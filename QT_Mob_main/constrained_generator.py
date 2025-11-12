import json
import logging

# 设置日志记录器
logger = logging.getLogger(__name__)

class ConstrainedGenerator:
    def __init__(self, tokenizer, codebook):
        self.tokenizer = tokenizer
        self.codebook = codebook
        self.allowed_tokens_h3 = None
        self.token_len_h3 = 0  # tokenizer-piece length offset (last id index)
        self.index_parts_len = None  # number of tokens composing one H3 index (e.g., 4)
        
        # 为了效率，预先计算好所有需要的token IDs
        self._prepare_token_sets()

    def _prepare_token_sets(self):
        """预计算并缓存所有约束所需的token集合，避免在每次调用时重复计算。"""
        # JSON 结构部分的 Tokens
        self.json_start_tokens = self.tokenizer.encode('{"h3_index": "', add_special_tokens=False)
        # stay_duration 为字符串，形如 "90min"：中间段包含起始引号
        self.json_mid_tokens = self.tokenizer.encode('", "stay_duration": "', add_special_tokens=False)
        self.json_end_token = self.tokenizer.encode('}', add_special_tokens=False)[0]
        self.quote_token = self.tokenizer.encode('"', add_special_tokens=False)[0]

        # Duration 数值部分的 Tokens: 仅允许 {"30min","60min",...,"600min"}
        self.duration_values = [f"{m}min" for m in range(30, 601, 30)]
        self.duration_token_seqs = [
            self.tokenizer.encode(v, add_special_tokens=False) for v in self.duration_values
        ]
        # 前缀->下一步允许的token集合；以及第一个位置的允许tokens
        self.duration_allowed_pos0 = set()
        self.duration_allowed_by_prefix = {}
        for seq in self.duration_token_seqs:
            if not seq:
                continue
            self.duration_allowed_pos0.add(seq[0])
            self.duration_allowed_by_prefix.setdefault(0, set()).add(seq[0])
            for i in range(1, len(seq)):
                prefix = tuple(seq[:i])
                self.duration_allowed_by_prefix.setdefault(prefix, set()).add(seq[i])

        # H3 Index部分的 Tokens (复用你原来的逻辑)
        if self.allowed_tokens_h3 is None:
            self.allowed_tokens_h3 = {}
            # H3 由多少个离散token组成（例如4），从 codebook 的第一项推断
            first_index = next(iter(self.codebook.values()))
            self.index_parts_len = len(first_index)
            # 计算“最后一位 id”的偏移，用于每个 token 的代表 id
            self.token_len_h3 = len(self.tokenizer(first_index[0])["input_ids"]) - 1 

            for index in self.codebook.values():
                token_ids = [self.tokenizer(token)["input_ids"][self.token_len_h3] for token in index]
                if token_ids[0] not in self.allowed_tokens_h3:
                    self.allowed_tokens_h3[token_ids[0]] = set()
                self.allowed_tokens_h3[token_ids[0]].add(token_ids[1])
                for i in range(2, len(token_ids)):
                    if tuple(token_ids[0:i]) not in self.allowed_tokens_h3:
                        self.allowed_tokens_h3[tuple(token_ids[0:i])] = set()
                    self.allowed_tokens_h3[tuple(token_ids[0:i])].add(token_ids[i])
            for index in self.codebook.values():
                for i, token in enumerate(index):
                    token_id = self.tokenizer(token)["input_ids"][self.token_len_h3]
                    if i not in self.allowed_tokens_h3:
                        self.allowed_tokens_h3[i] = set()
                    self.allowed_tokens_h3[i].add(token_id)
    
    def get_prefix_allowed_tokens_fn(self):
        """返回一个闭包函数，用于 transformers 的 generate 方法。"""
        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            num_generated = len(sentence)

            # 状态 0: 强制生成 '{"h3_index": "'
            if num_generated < len(self.json_start_tokens):
                return [self.json_start_tokens[num_generated]]

            # 状态 1: 正在生成 H3 Index
            h3_start_pos = len(self.json_start_tokens)
            # 使用从 codebook 推断的 H3 组成 token 数量（例如4）
            parts_len = self.index_parts_len  or 7
            h3_end_pos = h3_start_pos + parts_len
            if num_generated < h3_end_pos:
                h3_generated_tokens = sentence[h3_start_pos:]
                h3_generated_len = len(h3_generated_tokens)
                
                if h3_generated_len == 0:
                    return list(self.allowed_tokens_h3[0])
                elif h3_generated_len == 1:
                     return list(self.allowed_tokens_h3[h3_generated_tokens[0]])
                else:
                    return list(self.allowed_tokens_h3[tuple(h3_generated_tokens)])

            # 状态 2: 强制生成 '", "duration_seconds": '
            mid_start_pos = h3_end_pos
            mid_end_pos = mid_start_pos + len(self.json_mid_tokens)
            if num_generated < mid_end_pos:
                mid_generated_len = num_generated - mid_start_pos
                return [self.json_mid_tokens[mid_generated_len]]

            # 状态 3: 正在生成 Duration 数值（仅允许 {"30min","60min",...,"600min"} 的精确序列）
            duration_start_pos = mid_end_pos
            if sentence[-1] != self.json_end_token:
                value_prefix = sentence[duration_start_pos:]
                # 1) 如果刚好完整匹配某个 duration 序列，下一步应为关闭引号
                if any(tuple(seq) == tuple(value_prefix) for seq in self.duration_token_seqs):
                    return [self.quote_token]
                # 2) 如果已生成了关闭引号，则允许 '}' 结束 JSON
                if len(value_prefix) >= 1 and value_prefix[-1] == self.quote_token:
                    return [self.json_end_token]
                # 3) 空前缀：允许任何一个合法序列的起始 token
                if len(value_prefix) == 0:
                    return list(self.duration_allowed_pos0 or self.duration_allowed_by_prefix.get(0, set()))
                # 4) 尝试从精确前缀继续
                next_allowed = self.duration_allowed_by_prefix.get(tuple(value_prefix))
                if next_allowed:
                    return list(next_allowed)
                # 5) 如果前缀失配，则尽可能结束（先补全引号，否则结束大括号）
                return [self.quote_token] if (len(value_prefix) > 0 and value_prefix[-1] != self.quote_token) else [self.json_end_token]
            
            # 状态 4: 已生成 '}'，强制生成 EOS token
            if sentence[-1] == self.json_end_token:
                return [self.tokenizer.eos_token_id]

            # 兜底：如果逻辑出错，返回空列表，通常会停止生成
            return []

        return prefix_allowed_tokens_fn

# --- 如何在你的测试代码中使用 ---
# 假设 tokenizer 和 codebook 已经加载
# 1. 创建约束生成器的实例
# constrained_generator = ConstrainedGenerator(tokenizer, codebook)

# # 2. 获取前缀函数
# prefix_allowed_tokens = constrained_generator.get_prefix_allowed_tokens_fn()

# # 3. 在 model.generate 中使用它
# output = model.generate(
#     input_ids=inputs["input_ids"],
#     attention_mask=inputs["attention_mask"],
#     max_new_tokens=30, # 需要足够长以容纳整个JSON
#     # ... 其他采样参数 ...
#     prefix_allowed_tokens_fn=prefix_allowed_tokens,
#     # ... 其他参数 ...
# )