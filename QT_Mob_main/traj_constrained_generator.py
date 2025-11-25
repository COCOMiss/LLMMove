import logging
import torch
from transformers import LogitsProcessor

# ===== Logger Setup =====
def get_logger(name):
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("QT-Mob-TrajGen")

# ===== Logits Processor =====
class InspectLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.allowed_tokens = []

    def set_allowed_tokens(self, allowed_tokens):
        self.allowed_tokens = allowed_tokens

    def __call__(self, input_ids, scores):
        # 将所有非 allowed 的 token 设为 -inf
        if self.allowed_tokens:
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[:, self.allowed_tokens] = False
            scores[mask] = float('-inf')
            
            # 兜底：防止 allowed tokens 里原本就是 -inf (极罕见情况)
            # 如果 allowed_tokens 全是 -inf，模型会崩溃，所以这里做一个救援
            for token_id in self.allowed_tokens:
                if scores[0, token_id] == float('-inf'):
                    scores[0, token_id] = -10.0 # 赋予一个较小的有效值
        return scores

class TrajConstrainedGenerator:
    def __init__(self, tokenizer, codebook):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        
        # 1. 定义起始 Token (允许 [ 和 " [")
        self.prefix_token_pure = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.prefix_token_space = self.tokenizer.encode(" [", add_special_tokens=False)[0]
        self.allowed_start_tokens = list(set([self.prefix_token_pure, self.prefix_token_space]))

        self.list_end_token_id = self.tokenizer.encode("]", add_special_tokens=False)[0]
        
        # 2. 定义结构 Token 序列
        self.obj_start_tokens = self.tokenizer.encode('{"id": "', add_special_tokens=False)
        self.sep_id_time_tokens = self.tokenizer.encode('", "start_time": "', add_special_tokens=False)
        self.sep_time_h3_tokens = self.tokenizer.encode('", "h3_index": "', add_special_tokens=False)
        self.sep_h3_dur_tokens = self.tokenizer.encode('", "stay_duration": "', add_special_tokens=False)
        
        self.quote_token_id = self.tokenizer.encode('"', add_special_tokens=False)[0]
        self.obj_end_token_id = self.tokenizer.encode('}', add_special_tokens=False)[0]
        self.comma_sep_tokens = self.tokenizer.encode(', ', add_special_tokens=False)

        # 3. 构建 Trie 树
        self._prepare_duration_trie()
        self._prepare_time_trie()
        self.h3_trie = self._build_h3_trie(codebook)
        
        # 4. 数字 Token (用于 ID)
        self.digit_tokens = []
        for i in range(10):
            tps = self.tokenizer.encode(str(i), add_special_tokens=False)
            self.digit_tokens.extend(tps)
        self.digit_tokens = list(set(self.digit_tokens))

        logger.info("TrajConstrainedGenerator initialized.")

    def _prepare_duration_trie(self):
        self.duration_variants = [f"{m} min" for m in range(30, 601, 30)]
        self.duration_trie = {}
        for text in self.duration_variants:
            self._add_to_trie(self.duration_trie, text)

    def _prepare_time_trie(self):
        self.time_trie = {}
        periods = ["AM", "PM"]
        for period in periods:
            for h in range(1, 13): 
                for m in range(0, 60): 
                    # 格式: 09:05 AM (补零) 和 9:05 AM (不补零) 都支持
                    self._add_to_trie(self.time_trie, f"{h:02d}:{m:02d} {period}")
                    self._add_to_trie(self.time_trie, f"{h}:{m:02d} {period}")

    def _build_h3_trie(self, codebook):
        trie = {}
        if not codebook:
            logger.warning("Codebook is empty! H3 constrained generation will fail.")
            return trie
            
        for h3_parts_list in codebook.values():
            token_ids = []
            for part in h3_parts_list:
                token_ids.extend(self.tokenizer.encode(part, add_special_tokens=False))
            
            node = trie
            for token_id in token_ids:
                node = node.setdefault(token_id, {})
            node['is_end'] = True
        return trie

    def _add_to_trie(self, trie_root, text):
        token_seq = self.tokenizer.encode(text, add_special_tokens=False)
        node = trie_root
        for token_id in token_seq:
            node = node.setdefault(token_id, {})
        node["is_end"] = True

    # === 核心辅助函数：计算 token_list 结尾匹配 target_list 前缀的长度 ===
    def _get_matched_len(self, token_list, target_list):
        # 比如 generated=[..., A, B], target=[A, B, C] -> matched_len=2 -> next is C
        max_len = min(len(token_list), len(target_list) - 1)
        for i in range(max_len, 0, -1):
            if token_list[-i:] == target_list[:i]:
                return i
        return 0

    # === 核心辅助函数：查找子序列最后一次出现的位置 ===
    def _rfind_sequence(self, full_list, pattern):
        n = len(full_list)
        m = len(pattern)
        if m == 0: return -1
        for i in range(n - m, -1, -1):
            if full_list[i:i+m] == pattern:
                return i
        return -1

    def get_prefix_allowed_tokens_fn(self, prompt_lengths, logits_inspector: InspectLogitsProcessor):
        
        def prefix_allowed_tokens_fn(batch_id: int, sentence):
            try:
                # 1. 基础处理
                if hasattr(prompt_lengths, "__getitem__"):
                    prompt_length = int(prompt_lengths[batch_id]) if hasattr(prompt_lengths[batch_id], "item") else int(prompt_lengths[batch_id])
                else:
                    prompt_length = int(prompt_lengths)
                
                full_seq = sentence.tolist()
                newly_generated = full_seq[prompt_length:]
                
                # 2. 检查是否已经生成 EOS
                if newly_generated and newly_generated[-1] == self.eos_token_id:
                    return [self.eos_token_id]

                # =========================================================
                # 状态 0: 强制生成列表开始符 "["
                # =========================================================
                if len(newly_generated) == 0:
                    logits_inspector.set_allowed_tokens(self.allowed_start_tokens)
                    return self.allowed_start_tokens
                
                # 获取内容部分 (跳过第1个token，即 '[')
                content_tokens = newly_generated[1:]

                # =========================================================
                # 状态机逻辑
                # =========================================================
                
                # 定位最后一个对象开始符和结束符
                last_obj_start_idx = self._rfind_sequence(content_tokens, self.obj_start_tokens)
                
                last_obj_end_token_idx = -1
                for i in range(len(content_tokens)-1, -1, -1):
                    if content_tokens[i] == self.obj_end_token_id:
                        last_obj_end_token_idx = i
                        break
                    
                    
                
                
                
                
                
                
                
                
                # Case A: 处理 "列表之间" 的状态
                if last_obj_start_idx == -1 or last_obj_end_token_idx > last_obj_start_idx:
                    
                    # 1. 还没开始任何对象 -> 必须生成第一个
                    if last_obj_start_idx == -1:
                        k = self._get_matched_len(content_tokens, self.obj_start_tokens)
                        return [self.obj_start_tokens[k]]

                    # 2. 上一个对象已经结束 -> 检查 ',' 或 ']'
                    if last_obj_end_token_idx > last_obj_start_idx:
                        after_brace = content_tokens[last_obj_end_token_idx+1:]
                        
                        # (a) 刚闭合 '}' -> 决定是继续还是结束
                        if not after_brace:
                            # === 修改开始：统计已生成的对象数量 ===
                            # 统计 content_tokens 中 '}' 的出现次数
                            num_objs = content_tokens.count(self.obj_end_token_id)
                            
                            # 设定最小轨迹长度，例如 3
                            MIN_TRAJ_LENGTH = 3 
                            
                            if num_objs < MIN_TRAJ_LENGTH:
                                # 如果还没够 3 个点，强制生成逗号，逼迫模型继续
                                allowed = [self.comma_sep_tokens[0]]
                            else:
                                # 够了之后，允许逗号或结束
                                allowed = [self.comma_sep_tokens[0], self.list_end_token_id]
                            
                            logits_inspector.set_allowed_tokens(allowed)
                            return allowed
                            # === 修改结束 ===
                        
                        # (b) 已经生成 ']' -> 结束
                        if after_brace[0] == self.list_end_token_id:
                            return [self.eos_token_id]
                        
                        # (c) 正在生成 ', '
                        k_comma = self._get_matched_len(after_brace, self.comma_sep_tokens)
                        # 如果逗号还没完
                        if k_comma < len(self.comma_sep_tokens) and after_brace != self.comma_sep_tokens:
                             return [self.comma_sep_tokens[k_comma]]
                        
                        # (d) 逗号已生成完毕 -> 必须开始新对象
                        # after_brace 此时是 [comma_tokens..., partial_start_tokens...]
                        # 我们需要看逗号后面生成了多少 start_tokens
                        tokens_after_comma = after_brace[len(self.comma_sep_tokens):]
                        k_start = self._get_matched_len(tokens_after_comma, self.obj_start_tokens)
                        return [self.obj_start_tokens[k_start]]

                # Case B: 处于对象内部 (Inside Object)
                # 此时 last_obj_start_idx 指向当前对象的开头
                
                current_obj_tokens = content_tokens[last_obj_start_idx:]
                
                # 1. 确保 obj_start_tokens 完整生成
                if len(current_obj_tokens) < len(self.obj_start_tokens):
                    return [self.obj_start_tokens[len(current_obj_tokens)]]
                
                inner_content = current_obj_tokens[len(self.obj_start_tokens):]
                
                # 寻找各个分隔符的位置
                idx_id_time = self._rfind_sequence(inner_content, self.sep_id_time_tokens)
                idx_time_h3 = self._rfind_sequence(inner_content, self.sep_time_h3_tokens)
                idx_h3_dur  = self._rfind_sequence(inner_content, self.sep_h3_dur_tokens)
                
                # --- State: ID Value ---
                if idx_id_time == -1:
                    # 检查是否正在生成 ", "start_time": "
                    k = self._get_matched_len(inner_content, self.sep_id_time_tokens)
                    if k > 0: return [self.sep_id_time_tokens[k]]
                    
                    # 允许生成数字，或者如果已经有数字了，允许生成分隔符的第一个字
                    allowed = self.digit_tokens[:]
                    # 只有当 ID 不为空时才允许分隔符（这里简化处理，允许空 ID 以防卡死，或者假设模型够聪明）
                    allowed.append(self.sep_id_time_tokens[0])
                    logits_inspector.set_allowed_tokens(allowed)
                    return allowed

                # --- State: Start Time ---
                if idx_time_h3 == -1:
                    time_part = inner_content[idx_id_time + len(self.sep_id_time_tokens):]
                    
                    k = self._get_matched_len(time_part, self.sep_time_h3_tokens)
                    if k > 0: return [self.sep_time_h3_tokens[k]]
                    
                    # 查 Trie
                    node = self.time_trie
                    for t in time_part:
                        if t in node: node = node[t]
                        else: return [] 
                    
                    allowed = [key for key in node.keys() if key != 'is_end']
                    if 'is_end' in node:
                        allowed.append(self.sep_time_h3_tokens[0])
                    logits_inspector.set_allowed_tokens(allowed)
                    return allowed

                # --- State: H3 Index ---
                if idx_h3_dur == -1:
                    h3_part = inner_content[idx_time_h3 + len(self.sep_time_h3_tokens):]
                    
                    k = self._get_matched_len(h3_part, self.sep_h3_dur_tokens)
                    if k > 0: return [self.sep_h3_dur_tokens[k]]
                    
                    # 查 Trie
                    node = self.h3_trie
                    for t in h3_part:
                        if t in node: node = node[t]
                        else: return []
                    
                    allowed = [key for key in node.keys() if key != 'is_end']
                    if 'is_end' in node:
                        allowed.append(self.sep_h3_dur_tokens[0])
                    logits_inspector.set_allowed_tokens(allowed)
                    return allowed

                # --- State: Duration ---
                dur_part = inner_content[idx_h3_dur + len(self.sep_h3_dur_tokens):]
                
                # 检查是否已有闭合引号
                if self.quote_token_id in dur_part:
                    # 如果只有引号，下一个必须是 '}'
                    if dur_part[-1] == self.quote_token_id:
                        return [self.obj_end_token_id]
                    # 如果 '}' 也生成了，那应该回到 Case A，但这里兜底返回逗号或结束
                    return [self.comma_sep_tokens[0], self.list_end_token_id]
                
                node = self.duration_trie
                for t in dur_part:
                    if t in node: node = node[t]
                    else: return []
                
                allowed = [key for key in node.keys() if key != 'is_end']
                if 'is_end' in node:
                    allowed.append(self.quote_token_id) # 允许生成引号结束
                
                logits_inspector.set_allowed_tokens(allowed)
                return allowed

            except Exception as e:
                logger.error(f"Error in constrained gen: {e}")
                return [self.eos_token_id]

        return prefix_allowed_tokens_fn