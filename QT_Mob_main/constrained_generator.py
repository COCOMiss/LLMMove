import json
import logging

# 设置日志记录器
logger = logging.getLogger(__name__)

class ConstrainedGenerator:
    def __init__(self, tokenizer, codebook):
        self.tokenizer = tokenizer
        self.codebook = codebook
        self.allowed_tokens_h3 = None
        self.token_len_h3 = 0
        
        # 为了效率，预先计算好所有需要的token IDs
        self._prepare_token_sets()

    def _prepare_token_sets(self):
        """预计算并缓存所有约束所需的token集合，避免在每次调用时重复计算。"""
        # JSON 结构部分的 Tokens
        self.json_start_tokens = self.tokenizer.encode('{"h3_index": "', add_special_tokens=False)
        self.json_mid_tokens = self.tokenizer.encode('", "duration_seconds": ', add_special_tokens=False)
        self.json_end_token = self.tokenizer.encode('}', add_special_tokens=False)[0]

        # Duration 数值部分的 Tokens
        # 使用集合以提高查找效率
        self.digit_tokens = {self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)}
        self.dot_token = self.tokenizer.encode('.', add_special_tokens=False)[0]
        self.allowed_duration_tokens = self.digit_tokens | {self.dot_token}

        # H3 Index部分的 Tokens (复用你原来的逻辑)
        if self.allowed_tokens_h3 is None:
            self.allowed_tokens_h3 = {}
            # 假设所有h3 index的token化长度都一样
            self.token_len_h3 = len(self.tokenizer(list(self.codebook.values())[0][0])["input_ids"]) - 1 

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
            # 这里的 4 是 H3 index token 的数量，你需要根据实际情况调整
            h3_end_pos = h3_start_pos + 4 
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

            # 状态 3: 正在生成 Duration 数值
            duration_start_pos = mid_end_pos
            if sentence[-1] != self.json_end_token:
                # 检查是否已经有小数点了
                duration_tokens = sentence[duration_start_pos:]
                if self.dot_token in duration_tokens:
                    # 如果已有小数点，则只允许数字和结束符
                    return list(self.digit_tokens | {self.json_end_token})
                else:
                    # 允许数字、小数点和结束符
                    return list(self.allowed_duration_tokens | {self.json_end_token})
            
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