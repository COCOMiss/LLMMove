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

    # def _prepare_token_sets(self):
    #     """预计算并缓存所有约束所需的token集合，避免在每次调用时重复计算。"""
    #     # JSON 结构部分的 Tokens
    #     self.json_start_tokens = self.tokenizer.encode('{"h3_index": "', add_special_tokens=False)
    #     # stay_duration 为字符串，形如 "90min"：中间段包含起始引号
    #     self.json_mid_tokens = self.tokenizer.encode('", "stay_duration": "', add_special_tokens=False)
    #     self.json_end_token = self.tokenizer.encode('}', add_special_tokens=False)[0]
    #     self.quote_token = self.tokenizer.encode('"', add_special_tokens=False)[0]

    #     # Duration 数值部分的 Tokens: 仅允许 {"30min","60min",...,"600min"}
    #     self.duration_values = [f"{m}min" for m in range(30, 601, 30)]
    #     self.duration_token_seqs = [
    #         self.tokenizer.encode(v, add_special_tokens=False) for v in self.duration_values
    #     ]
    #     # 前缀->下一步允许的token集合；以及第一个位置的允许tokens
    #     self.duration_allowed_pos0 = set()
    #     self.duration_allowed_by_prefix = {}
    #     for seq in self.duration_token_seqs:
    #         if not seq:
    #             continue
    #         self.duration_allowed_pos0.add(seq[0])
    #         self.duration_allowed_by_prefix.setdefault(0, set()).add(seq[0])
    #         for i in range(1, len(seq)):
    #             prefix = tuple(seq[:i])
    #             self.duration_allowed_by_prefix.setdefault(prefix, set()).add(seq[i])

    #     # H3 Index部分的 Tokens (复用你原来的逻辑)
    #     if self.allowed_tokens_h3 is None:
    #         self.allowed_tokens_h3 = {}
    #         # H3 由多少个离散token组成（例如4），从 codebook 的第一项推断
    #         first_index = next(iter(self.codebook.values()))
    #         self.index_parts_len = len(first_index)
    #         # 计算“最后一位 id”的偏移，用于每个 token 的代表 id
    #         self.token_len_h3 = len(self.tokenizer(first_index[0])["input_ids"]) - 1 

    #         for index in self.codebook.values():
    #             token_ids = [self.tokenizer(token)["input_ids"][self.token_len_h3] for token in index]
    #             if token_ids[0] not in self.allowed_tokens_h3:
    #                 self.allowed_tokens_h3[token_ids[0]] = set()
    #             self.allowed_tokens_h3[token_ids[0]].add(token_ids[1])
    #             for i in range(2, len(token_ids)):
    #                 if tuple(token_ids[0:i]) not in self.allowed_tokens_h3:
    #                     self.allowed_tokens_h3[tuple(token_ids[0:i])] = set()
    #                 self.allowed_tokens_h3[tuple(token_ids[0:i])].add(token_ids[i])
    #         for index in self.codebook.values():
    #             for i, token in enumerate(index):
    #                 token_id = self.tokenizer(token)["input_ids"][self.token_len_h3]
    #                 if i not in self.allowed_tokens_h3:
    #                     self.allowed_tokens_h3[i] = set()
    #                 self.allowed_tokens_h3[i].add(token_id)
                    
                    
    # --- FINAL CORRECTED VERSION ---

    # def get_prefix_allowed_tokens_fn(self):
    #     """
    #     Returns a closure function for constrained generation that correctly handles:
    #     1. A Trie-based lookup for multi-token H3 indices.
    #     2. A prefix-based lookup for allowed duration values.
    #     3. The overall JSON structure.
    #     """
    #     def prefix_allowed_tokens_fn(batch_id, sentence):
    #         sentence = sentence.tolist()
    #         num_generated = len(sentence)

    #         # STATE 0: Force generation of '{"h3_index": "'
    #         if num_generated < len(self.json_start_tokens):
    #             return [self.json_start_tokens[num_generated]]

    #         # After the initial JSON part, we determine which part of the content we are generating.
    #         # Check if we have finished the H3 index part. An H3 index is complete
    #         # when the sequence of generated tokens after `json_start_tokens` is a valid key in our Trie
    #         # marked with 'is_end', OR when the next token to be generated is the start of the middle part.
            
    #         # We search for the start of the middle JSON part '", "stay_duration": "'
    #         mid_part_start_index = -1
    #         # Simple sliding window to find the sublist
    #         for i in range(len(sentence) - len(self.json_mid_tokens) + 1):
    #             if sentence[i:i + len(self.json_mid_tokens)] == self.json_mid_tokens:
    #                 mid_part_start_index = i
    #                 break

    #         # If the middle part hasn't started yet, we must be in STATE 1
    #         if mid_part_start_index == -1:
    #             # STATE 1: Generating the H3 Index using the Trie
    #             h3_start_pos = len(self.json_start_tokens)
    #             h3_generated_tokens = sentence[h3_start_pos:]
                
    #             node = self.h3_trie
    #             for token_id in h3_generated_tokens:
    #                 if token_id in node:
    #                     node = node[token_id]
    #                 else:
    #                     # Generated an invalid H3 prefix, stop.
    #                     return [] 

    #             # The next allowed tokens are the children of the current Trie node
    #             allowed_next = [token for token in node.keys() if token != 'is_end']

    #             # If the current prefix marks the end of a valid H3 index,
    #             # the next token could also be the start of the middle JSON part.
    #             if 'is_end' in node:
    #                 allowed_next.append(self.json_mid_tokens[0])

    #             return allowed_next if allowed_next else []

    #         # If we found the middle part, we are either generating it or the duration afterwards.
    #         # STATE 2: We are currently generating the '", "stay_duration": "' part
    #         mid_end_pos = mid_part_start_index + len(self.json_mid_tokens)
    #         if num_generated < mid_end_pos:
    #             offset = num_generated - mid_part_start_index
    #             return [self.json_mid_tokens[offset]]

    #         # STATE 3: Generating the duration value (e.g., "30min")
    #         duration_start_pos = mid_end_pos
            
    #         # Check if the final '}' has been generated
    #         if self.json_end_token in sentence[duration_start_pos:]:
    #             # STATE 4: End of generation, force EOS
    #             return [self.tokenizer.eos_token_id]

    #         duration_prefix_tokens = tuple(sentence[duration_start_pos:])
            
    #         # Case 3.1: A full duration value has been generated, next must be the closing quote
    #         is_complete_duration = any(
    #             duration_prefix_tokens == tuple(seq) for seq in self.duration_token_seqs
    #         )
    #         if is_complete_duration:
    #             return [self.quote_token]

    #         # Case 3.2: The closing quote has just been generated, next must be the closing brace
    #         if len(duration_prefix_tokens) > 0 and duration_prefix_tokens[-1] == self.quote_token:
    #             return [self.json_end_token]

    #         # Case 3.3: We are in the middle of generating a duration value
    #         # Use the pre-computed dictionary to find the next allowed token(s)
    #         allowed_next_duration_tokens = self.duration_allowed_by_prefix.get(duration_prefix_tokens)
    #         if allowed_next_duration_tokens:
    #             return list(allowed_next_duration_tokens)
                
    #         # Fallback: If no valid prefix is found, something is wrong. Stop generation.
    #         return []

    #     return prefix_allowed_tokens_fn
    
    # def get_prefix_allowed_tokens_fn(self):
    #     """返回一个闭包函数，用于 transformers 的 generate 方法。"""
    #     def prefix_allowed_tokens_fn(batch_id, sentence):
    #         sentence = sentence.tolist()
    #         num_generated = len(sentence)

    #         # 状态 0: 强制生成 '{"h3_index": "'
    #         if num_generated < len(self.json_start_tokens):
    #             return [self.json_start_tokens[num_generated]]      
            
    #         # 状态 1: 正在生成 H3 Index
    #         h3_start_pos = len(self.json_start_tokens)
    #         parts_len = self.index_parts_len or 7 # 确保 self.index_parts_len 已被正确初始化
    #         h3_end_pos = h3_start_pos + parts_len

    #         if num_generated < h3_end_pos:
    #             h3_generated_tokens = sentence[h3_start_pos:]
                
    #             # 将 h3_generated_tokens 转换为元组作为 key
    #             prefix_tuple = tuple(h3_generated_tokens)
                
    #             # 检查这个前缀是否存在于允许的规则中
    #             if prefix_tuple in self.allowed_tokens_h3:
    #                 return list(self.allowed_tokens_h3[prefix_tuple])
    #             elif len(h3_generated_tokens) == 0:
    #                 # 处理初始情况，key 应该是一个特殊值或空元组，取决于你的构建方式
    #                 # 假设你为第一个token设置的key是整数0
    #                 if 0 in self.allowed_tokens_h3:
    #                     return list(self.allowed_tokens_h3[0])
    #                 else:
    #                     # 如果没有为位置0设置规则，说明逻辑有误，返回空列表
    #                     logger.warning("No rule found for the first H3 token (position 0).")
    #                     return []
    #             else:
    #                 # 如果一个非空前缀在字典里找不到，说明出现了意外情况，停止生成
    #                 logger.warning(f"Invalid H3 prefix {prefix_tuple} encountered. Halting generation.")
    #                 return []
           

    #         # 状态 2: 强制生成 '", "duration_seconds": '
    #         mid_start_pos = h3_end_pos
    #         mid_end_pos = mid_start_pos + len(self.json_mid_tokens)
    #         if num_generated < mid_end_pos:
    #             mid_generated_len = num_generated - mid_start_pos
    #             return [self.json_mid_tokens[mid_generated_len]]

    #         # 状态 3: 正在生成 Duration 数值（仅允许 {"30min","60min",...,"600min"} 的精确序列）
    #         duration_start_pos = mid_end_pos
    #         if sentence[-1] != self.json_end_token:
    #             value_prefix = sentence[duration_start_pos:]
    #             # 1) 如果刚好完整匹配某个 duration 序列，下一步应为关闭引号
    #             if any(tuple(seq) == tuple(value_prefix) for seq in self.duration_token_seqs):
    #                 return [self.quote_token]
    #             # 2) 如果已生成了关闭引号，则允许 '}' 结束 JSON
    #             if len(value_prefix) >= 1 and value_prefix[-1] == self.quote_token:
    #                 return [self.json_end_token]
    #             # 3) 空前缀：允许任何一个合法序列的起始 token
    #             if len(value_prefix) == 0:
    #                 return list(self.duration_allowed_pos0 or self.duration_allowed_by_prefix.get(0, set()))
    #             # 4) 尝试从精确前缀继续
    #             next_allowed = self.duration_allowed_by_prefix.get(tuple(value_prefix))
    #             if next_allowed:
    #                 return list(next_allowed)
    #             # 5) 如果前缀失配，则尽可能结束（先补全引号，否则结束大括号）
    #             return [self.quote_token] if (len(value_prefix) > 0 and value_prefix[-1] != self.quote_token) else [self.json_end_token]
            
    #         # 状态 4: 已生成 '}'，强制生成 EOS token
    #         if sentence[-1] == self.json_end_token:
    #             return [self.tokenizer.eos_token_id]

    #         # 兜底：如果逻辑出错，返回空列表，通常会停止生成
    #         return []

    #     return prefix_allowed_tokens_fn
        # --- 最终版方法1：请用此代码块完整替换 _prepare_token_sets ---

    def _prepare_token_sets(self):
        """
        预计算并缓存所有约束所需的token集合。
        该最终版本根据已验证的分词器行为，正确地从codebook构建H3 Trie。
        """
        logger.info("Preparing token sets for constrained generation...")

        # --- JSON 结构部分的 Tokens (这部分保持不变) ---
        self.json_start_tokens = self.tokenizer.encode('{"h3_index": "', add_special_tokens=False)
        self.json_mid_tokens = self.tokenizer.encode('", "stay_duration": "', add_special_tokens=False)
        self.json_end_token = self.tokenizer.encode('}', add_special_tokens=False)[0]
        self.quote_token = self.tokenizer.encode('"', add_special_tokens=False)[0]

        # --- Duration 数值部分的 Tokens (这部分保持不变) ---
        self.duration_values = [f"{m}min" for m in range(30, 601, 30)]
        self.duration_token_seqs = [
            self.tokenizer.encode(v, add_special_tokens=False) for v in self.duration_values
        ]
        self.duration_allowed_by_prefix = {}
        for seq in self.duration_token_seqs:
            if not seq:
                continue
            # 使用空元组作为初始前缀
            prefix = ()
            if prefix not in self.duration_allowed_by_prefix:
                self.duration_allowed_by_prefix[prefix] = set()
            self.duration_allowed_by_prefix[prefix].add(seq[0])
            # 后续的每个前缀
            for i in range(1, len(seq)):
                prefix = tuple(seq[:i])
                if prefix not in self.duration_allowed_by_prefix:
                    self.duration_allowed_by_prefix[prefix] = set()
                self.duration_allowed_by_prefix[prefix].add(seq[i])
        
        # --- H3 Index 部分的 Tokens (构建正确的Trie) ---
        logger.info("Building H3 Trie from codebook (Correct Method)...")
        self.h3_trie = {}
        
        # self.codebook.values() 是一个列表的列表, e.g., [['<a_68>',...], ['<a_244>',...]]
        for h3_parts_list in self.codebook.values():
            
            # 因为每个部分是一个独立的token，我们直接获取它们的ID
            # tokenizer.encode('<a_68>') -> [151716]
            # 我们需要解开这个列表，只取那个ID
            token_ids = [self.tokenizer.encode(part, add_special_tokens=False)[0] for part in h3_parts_list]
            
            # 将这个由7个token ID组成的序列添加到Trie中
            node = self.h3_trie
            for token_id in token_ids:
                if token_id not in node:
                    node[token_id] = {}
                node = node[token_id]
            
            # 在Trie的叶子节点标记这是一个合法的结束
            node['is_end'] = True
            
        
        
        logger.info("Token sets preparation complete.")
        
        if not self.h3_trie:
            logger.error("!!! CRITICAL ERROR: self.h3_trie is EMPTY after build process !!!")
        else:
            logger.info(f"Trie built successfully. Number of first-level keys: {len(self.h3_trie)}")
            # 打印一些样本key，确保它们是整数token ID
            import itertools
            sample_keys = list(itertools.islice(self.h3_trie.keys(), 5))
            logger.info(f"Sample of first-level Trie keys: {sample_keys}")

        logger.info("Token sets preparation complete.")
        
    # --- 最终修正版：请用此代码块完整替换 get_prefix_allowed_tokens_fn ---

    def get_prefix_allowed_tokens_fn(self):
        """
        返回一个约束生成闭包函数。
        此版本正确处理传入的完整sentence，仅对新生成的部分进行约束。
        """
        # 在闭包外部，获取一次输入prompt的长度。
        # 这假设了在一次generate调用中，输入的prompt是固定的。
        # 我们需要在调用model.generate之前知道输入有多长。
        # 这部分需要在您的测试循环中进行适配。
        
        # 一个更稳健的方法是，让闭包自己来计算。
        
        def prefix_allowed_tokens_fn(batch_id, sentence):
            # sentence 是完整的 token 序列: input_ids + new_tokens
            
            # --- 关键修正：动态计算新生成token的起始位置 ---
            # 我们知道 'prediction:' 后面是我们关心的部分。
            # 首先，找到 'prediction:' 在 sentence 中的位置。
            prediction_prompt_tokens = self.tokenizer.encode("prediction:", add_special_tokens=False)
            
            # 查找 prediction_prompt_tokens 这个子列表在 sentence 中最后出现的位置
            start_of_generation_marker = -1
            for i in range(len(sentence) - len(prediction_prompt_tokens), -1, -1):
                if sentence[i:i + len(prediction_prompt_tokens)] == prediction_prompt_tokens:
                    start_of_generation_marker = i + len(prediction_prompt_tokens)
                    break
            
            if start_of_generation_marker == -1:
                # 如果连 'prediction:' 都找不到，说明输入有问题，停止生成
                return []

            # `newly_generated_tokens` 是模型真正新生成的内容
            newly_generated_tokens = sentence[start_of_generation_marker:]
            num_generated = len(newly_generated_tokens)
            
            # --- 现在，所有的逻辑都基于 newly_generated_tokens，而不是整个 sentence ---

            # STATE 0: 检查 '{"h3_index": "' 是否已生成
            # 注意：因为我们已经把这部分放进了prompt，所以newly_generated_tokens在第一步应该是空的
            json_start_prefix = '{"h3_index": "'
            json_start_tokens = self.tokenizer.encode(json_start_prefix, add_special_tokens=False)

            # 因为我们把json_start_tokens放进了prompt，所以newly_generated_tokens的开头应该就是H3内容
            # 因此我们直接进入STATE 1
            
            # 检查是否已开始生成中间部分
            mid_part_str = '", "stay_duration": "'
            mid_part_tokens = self.tokenizer.encode(mid_part_str, add_special_tokens=False)
            
            mid_part_start_index = -1
            if len(newly_generated_tokens) >= len(mid_part_tokens):
                for i in range(len(newly_generated_tokens) - len(mid_part_tokens) + 1):
                    if newly_generated_tokens[i:i + len(mid_part_tokens)] == mid_part_tokens:
                        mid_part_start_index = i
                        break

            # 如果中间部分还没开始，我们一定是在生成H3索引
            if mid_part_start_index == -1:
                # STATE 1: 使用Trie生成H3 Index
                # 此时 h3_generated_tokens 就是 newly_generated_tokens
                h3_generated_tokens = newly_generated_tokens
                
                node = self.h3_trie
                try:
                    for token_id in h3_generated_tokens:
                        node = node[token_id]
                except KeyError:
                    return [] 

                allowed_next = [token for token in node.keys() if token != 'is_end']

                if 'is_end' in node:
                    allowed_next.append(mid_part_tokens[0])
                
                return allowed_next if allowed_next else []

            # 如果找到了中间部分...
            # STATE 2: 正在生成 '", "stay_duration": "'
            # 注意：这里的逻辑也需要基于 newly_generated_tokens 的相对位置
            # 但由于错误发生在STATE 1，我们暂时简化后续逻辑以确保STATE 1正确
            mid_end_pos_relative = mid_part_start_index + len(mid_part_tokens)
            if len(newly_generated_tokens) < mid_end_pos_relative:
                offset = len(newly_generated_tokens) - mid_part_start_index
                return [mid_part_tokens[offset]]

            # STATE 3: 正在生成 duration value
            duration_start_pos_relative = mid_end_pos_relative
            
            if self.json_end_token in newly_generated_tokens[duration_start_pos_relative:]:
                return [self.tokenizer.eos_token_id]

            duration_prefix_tokens = tuple(newly_generated_tokens[duration_start_pos_relative:])
            
            is_complete_duration = any(
                duration_prefix_tokens == tuple(seq) for seq in self.duration_token_seqs
            )
            if is_complete_duration:
                return [self.quote_token]

            if len(duration_prefix_tokens) > 0 and duration_prefix_tokens[-1] == self.quote_token:
                return [self.json_end_token]

            allowed_next_duration_tokens = self.duration_allowed_by_prefix.get(duration_prefix_tokens)
            if allowed_next_duration_tokens:
                return list(allowed_next_duration_tokens)
                
            return []

        return prefix_allowed_tokens_fn
            
            
