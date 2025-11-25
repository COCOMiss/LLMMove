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
        if self.allowed_tokens:
            # 只有当你想看详细调试信息时取消注释下面这行
            # logger.info(f"Allowed: {self.allowed_tokens}")
            
            # 兜底防止 -inf 导致生成崩溃
            for token_id in self.allowed_tokens:
                if scores[0, token_id] == float('-inf'):
                    # logger.warning(f"WARNING: Token {token_id} has -inf score. Forcing to -100 (clickable).")
                    scores[0, token_id] = -100.0 # 给一个很小但非inf的值，允许强行采样
        return scores

class TrajConstrainedGenerator:
    def __init__(self, tokenizer, codebook):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        
        
        self.prefix_token_pure = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.prefix_token_space = self.tokenizer.encode(" [", add_special_tokens=False)[0] # 应对 "prediction:" 后面没空格的情况
        
        # 将它们都放入允许列表（去重）
        self.allowed_start_tokens = list(set([self.prefix_token_pure, self.prefix_token_space]))
        
        # 为了后续逻辑方便，我们内部统一认为前缀长度是 1
        self.prefix_tokens = [self.prefix_token_pure]

        self.list_end_token_id = self.tokenizer.encode("]", add_special_tokens=False)[0]
        
        # 对象内部结构
        self.obj_start_tokens = self.tokenizer.encode('{"id": "', add_special_tokens=False)
        self.sep_id_time_tokens = self.tokenizer.encode('", "start_time": "', add_special_tokens=False)
        self.sep_time_h3_tokens = self.tokenizer.encode('", "h3_index": "', add_special_tokens=False)
        self.sep_h3_dur_tokens = self.tokenizer.encode('", "stay_duration": "', add_special_tokens=False)
        
        self.quote_token_id = self.tokenizer.encode('"', add_special_tokens=False)[0]
        self.obj_end_token_id = self.tokenizer.encode('}', add_special_tokens=False)[0]
        
        # 列表分隔符
        self.comma_sep_tokens = self.tokenizer.encode(', ', add_special_tokens=False)

        # ===== 2. 构建约束 Trie 树 =====
        self._prepare_duration_trie()
        self._prepare_time_trie()
        self.h3_trie = self._build_h3_trie(codebook)
        
        # 数字 Token (用于 ID)
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
                    time_str = f"{h:02d}:{m:02d} {period}" # 补零格式 09:05 AM
                    self._add_to_trie(self.time_trie, time_str)
                    
                    # 同时也支持一下非补零格式，以防万一: 9:05 AM
                    time_str_simple = f"{h}:{m:02d} {period}"
                    self._add_to_trie(self.time_trie, time_str_simple)

    def _build_h3_trie(self, codebook):
        trie = {}
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

    def _ends_with(self, token_list, suffix_list):
        if len(token_list) < len(suffix_list):
            return False
        return token_list[-len(suffix_list):] == suffix_list

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
                if hasattr(prompt_lengths, "__getitem__"):
                    prompt_length = int(prompt_lengths[batch_id]) if hasattr(prompt_lengths[batch_id], "item") else int(prompt_lengths[batch_id])
                else:
                    prompt_length = int(prompt_lengths)
                
                full_seq = sentence.tolist()
                newly_generated = full_seq[prompt_length:]
                
                # 检查是否已经生成 EOS
                if newly_generated and newly_generated[-1] == self.eos_token_id:
                    return [self.eos_token_id]

                # =========================================================
                # 状态 0: 强制生成前缀 "["
                # =========================================================
                if len(newly_generated) == 0:
                    # 如果还没有生成任何内容，允许 '[' 或 ' ['
                    logits_inspector.set_allowed_tokens(self.allowed_start_tokens)
                    return self.allowed_start_tokens
                content_tokens = newly_generated[1:] # 假设前缀长度为 1
                

                # =========================================================
                # 状态机逻辑
                # =========================================================
                
                last_obj_start_idx = self._rfind_sequence(content_tokens, self.obj_start_tokens)
                
                # 查找最后一个对象结束符 '}'
                last_obj_end_token_idx = -1
                for i in range(len(content_tokens)-1, -1, -1):
                    if content_tokens[i] == self.obj_end_token_id:
                        last_obj_end_token_idx = i
                        break

                # Case A: 准备开始新对象 (刚开始 或 刚结束上一个)
                is_start_new = False
                
                # 1. 还没开始任何对象
                if last_obj_start_idx == -1:
                    is_start_new = True
                
                # 2. 上一个对象已经结束
                elif last_obj_end_token_idx > last_obj_start_idx:
                    after_brace = content_tokens[last_obj_end_token_idx+1:]
                    
                    if not after_brace:
                        # 刚生成 '}' -> 可选 ', ' 或 ']'
                        allowed = [self.comma_sep_tokens[0], self.list_end_token_id]
                        logits_inspector.set_allowed_tokens(allowed)
                        return allowed
                    
                    if after_brace[0] == self.list_end_token_id:
                        # 已经生成 ']' -> 结束
                        return [self.eos_token_id]
                    
                    if len(after_brace) < len(self.comma_sep_tokens):
                        # 正在生成 ', '
                        return [self.comma_sep_tokens[len(after_brace)]]
                    elif after_brace == self.comma_sep_tokens:
                        # 生成完 ', ' -> 必须开始新对象
                        is_start_new = True
                
                if is_start_new:
                    # 必须生成 '{"id": "'
                    # 由于我们是一个一个token生成的，这里返回第一个
                    return [self.obj_start_tokens[0]]

                # Case B: 正在生成 '{"id": "' 的过程中
                # 检查是否处于 obj_start_tokens 的中间
                if last_obj_start_idx == -1:
                    # 还在第一个对象的 start tokens 序列中
                    # content_tokens 目前包含了部分 start tokens
                    # 比如 obj_start_tokens 是 [A, B, C]
                    # content_tokens 是 [A] -> return [B]
                    current_len = len(content_tokens)
                    if current_len < len(self.obj_start_tokens):
                         return [self.obj_start_tokens[current_len]]
                
                # Case C: 对象内部
                current_obj_tokens = content_tokens[last_obj_start_idx:]
                
                # 再次检查 start tokens 是否完整 (针对非第一个对象)
                if len(current_obj_tokens) < len(self.obj_start_tokens):
                    return [self.obj_start_tokens[len(current_obj_tokens)]]
                
                inner_content = current_obj_tokens[len(self.obj_start_tokens):]
                
                idx_id_time = self._rfind_sequence(inner_content, self.sep_id_time_tokens)
                idx_time_h3 = self._rfind_sequence(inner_content, self.sep_time_h3_tokens)
                idx_h3_dur  = self._rfind_sequence(inner_content, self.sep_h3_dur_tokens)
                
                # --- State: ID Value ---
                if idx_id_time == -1:
                    # 检查是否正在生成分隔符
                    matched_sep_len = 0
                    for i in range(len(self.sep_id_time_tokens), 0, -1):
                        if self._ends_with(inner_content, self.sep_id_time_tokens[:i]):
                            matched_sep_len = i
                            break
                    
                    if matched_sep_len > 0:
                        if matched_sep_len < len(self.sep_id_time_tokens):
                            return [self.sep_id_time_tokens[matched_sep_len]]
                            
                    allowed = self.digit_tokens + [self.sep_id_time_tokens[0]]
                    logits_inspector.set_allowed_tokens(allowed)
                    return allowed

                # --- State: Start Time ---
                if idx_time_h3 == -1:
                    time_part = inner_content[idx_id_time + len(self.sep_id_time_tokens):]
                    
                    matched_sep_len = 0
                    for i in range(len(self.sep_time_h3_tokens), 0, -1):
                        if self._ends_with(time_part, self.sep_time_h3_tokens[:i]):
                            matched_sep_len = i
                            break
                    
                    if matched_sep_len > 0:
                        if matched_sep_len < len(self.sep_time_h3_tokens):
                            return [self.sep_time_h3_tokens[matched_sep_len]]
                            
                    node = self.time_trie
                    for t in time_part:
                        if t in node: node = node[t]
                        else: return [] # Invalid path
                    
                    allowed = [k for k in node.keys() if k != 'is_end']
                    if 'is_end' in node:
                        allowed.append(self.sep_time_h3_tokens[0])
                    logits_inspector.set_allowed_tokens(allowed)
                    return allowed

                # --- State: H3 Index ---
                if idx_h3_dur == -1:
                    h3_part = inner_content[idx_time_h3 + len(self.sep_time_h3_tokens):]
                    
                    matched_sep_len = 0
                    for i in range(len(self.sep_h3_dur_tokens), 0, -1):
                        if self._ends_with(h3_part, self.sep_h3_dur_tokens[:i]):
                            matched_sep_len = i
                            break
                    
                    if matched_sep_len > 0:
                        if matched_sep_len < len(self.sep_h3_dur_tokens):
                            return [self.sep_h3_dur_tokens[matched_sep_len]]
                            
                    node = self.h3_trie
                    for t in h3_part:
                        if t in node: node = node[t]
                        else: return []
                    
                    allowed = [k for k in node.keys() if k != 'is_end']
                    if 'is_end' in node:
                        allowed.append(self.sep_h3_dur_tokens[0])
                    logits_inspector.set_allowed_tokens(allowed)
                    return allowed

                # --- State: Duration ---
                dur_part = inner_content[idx_h3_dur + len(self.sep_h3_dur_tokens):]
                
                # Check closing quote
                if self.quote_token_id in dur_part:
                    # Closing quote found, expect '}'
                    if dur_part[-1] == self.quote_token_id:
                        return [self.obj_end_token_id]
                    # This case should be handled by top-level logic (Case A), but just in case
                    return [self.comma_sep_tokens[0], self.list_end_token_id]
                
                node = self.duration_trie
                for t in dur_part:
                    if t in node: node = node[t]
                    else: return []
                
                allowed = [k for k in node.keys() if k != 'is_end']
                if 'is_end' in node:
                    allowed.append(self.quote_token_id)
                
                logits_inspector.set_allowed_tokens(allowed)
                return allowed

            except Exception as e:
                logger.error(f"Error: {e}")
                return [self.eos_token_id]

        return prefix_allowed_tokens_fn