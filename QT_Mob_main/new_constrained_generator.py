# --- 第二步：请务必使用这个版本的生成器 ---
import logging
from logger_utils import get_logger

# 使用 logger_utils 的 get_logger，这样日志会同时输出到控制台和文件
logger = get_logger("QT-Mob-ConstrainedGen")
# --- 第一步：定义这个诊断类 ---
from transformers import LogitsProcessor

class InspectLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.allowed_tokens = []

    def set_allowed_tokens(self, allowed_tokens):
        # 这个方法让我们的约束函数可以把“允许列表”告诉这个诊断器
        self.allowed_tokens = allowed_tokens

    def __call__(self, input_ids, scores):
        # scores 是 logits tensor，形状为 (batch_size, vocab_size)
        if self.allowed_tokens:
            logger.info("\n--- INSIDE LOGITS PROCESSOR ---")
            logger.info(f"Logits shape: {scores.shape}")
            logger.info(f"Allowed token IDs: {self.allowed_tokens}")
            
            # 记录我们关心的token的logits
            for token_id in self.allowed_tokens:
                # 假设 batch size is 1
                logit_value = scores[0, token_id].item()
                token_str = self.tokenizer.decode([token_id])
                logger.info(f"    - Logit for token {token_id} ('{token_str}'): {logit_value}")
            
            # 检查是否有 -inf
            has_inf = any(scores[0, token_id].item() == float('-inf') for token_id in self.allowed_tokens)
            if has_inf:
                logger.info("    !!! WARNING: At least one allowed token has a logit of -inf!")
            logger.info("---------------------------------\n")

        return scores

class FinalConstrainedGenerator:
    def __init__(self, tokenizer, codebook):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

        # 预先编码所有片段
        self.prediction_tokens = self.tokenizer.encode("prediction:", add_special_tokens=False)
        self.json_start_tokens = self.tokenizer.encode('{"h3_index": "', add_special_tokens=False)
        self.json_mid_tokens = self.tokenizer.encode('", "stay_duration": "', add_special_tokens=False)
        
        
        
        self.quote_token_id = self.tokenizer.encode('"', add_special_tokens=False)[0]
        self.json_end_token_id = self.tokenizer.encode('}', add_special_tokens=False)[0]
        
        self._prepare_duration_rules()
        self.h3_trie = self._build_h3_trie(codebook)
        logger.info("FinalConstrainedGenerator initialized.")

    def _prepare_duration_rules(self):
        self.duration_variants = []
        for minutes in range(30, 601, 30):
            self.duration_variants.append(f"{minutes} min")
            # 如果还想允许“分钟”等其他写法，也在这里append

        self.duration_trie = {}
        for text in self.duration_variants:
            token_seq = self.tokenizer.encode(text, add_special_tokens=False)
            node = self.duration_trie
            for token_id in token_seq:
                node = node.setdefault(token_id, {})
            node["is_end"] = True

    def _build_h3_trie(self, codebook):
        trie = {}
        for h3_parts_list in codebook.values():
            token_ids = [self.tokenizer.encode(part, add_special_tokens=False)[0] for part in h3_parts_list]
            node = trie
            for token_id in token_ids:
                if token_id not in node: node[token_id] = {}
                node = node[token_id]
            node['is_end'] = True
        return trie

    # --- 最终调试版(修正后): 请用这段代码完整替换 get_prefix_allowed_tokens_fn ---

    def get_prefix_allowed_tokens_fn(self, prompt_lengths: int, logits_inspector: InspectLogitsProcessor):
        """
        一个状态机逻辑完全正确、支持完整JSON生成、并带有正确logits_inspector调用的约束函数。
        """
        def prefix_allowed_tokens_fn(batch_id: int, sentence):
            try:
                # =================================================================
                # 1. 初始化和分离新生成的部分
                # =================================================================
                if hasattr(prompt_lengths, "__getitem__"):
                    prompt_length = prompt_lengths[batch_id]
                    if hasattr(prompt_length, "item"):
                        prompt_length = int(prompt_length.item())
                else:
                    prompt_length = int(prompt_lengths)
                newly_generated = sentence[prompt_length:].tolist()
                num_generated = len(newly_generated)
                
                decoded_newly_generated = self.tokenizer.decode(newly_generated)
                logger.info(f"\n--- [Step prompt_len+{num_generated}] ---")
                logger.info(f"DEBUG: Newly generated tokens (IDs): {newly_generated}")
                logger.info(f"DEBUG: Decoded new text: '{decoded_newly_generated}'")
                logger.info(f"DEBUG: EOS token ID: {self.eos_token_id}, JSON end token ID: {self.json_end_token_id}")
                
                # 如果已经生成了 EOS token，强制返回 EOS
                if newly_generated and newly_generated[-1] == self.eos_token_id:
                    logger.info("LOG (Early Exit): EOS token already generated. Returning EOS.")
                    allowed_next = [self.eos_token_id]
                    logits_inspector.set_allowed_tokens(allowed_next)
                    return allowed_next
                
                # 检查是否已经生成了 JSON 结束符 `}`，如果是，强制返回 EOS
                if self.json_end_token_id in newly_generated:
                    logger.info(f"DEBUG: Found JSON end token {self.json_end_token_id} in newly_generated")
                    json_end_indices = [i for i, tid in enumerate(newly_generated) if tid == self.json_end_token_id]
                    first_json_end_idx = json_end_indices[0]
                    tokens_after_json_end = newly_generated[first_json_end_idx + 1:]
                    
                    # 检查 tokens_after_json_end 中是否有 EOS
                    has_eos_after = self.eos_token_id in tokens_after_json_end
                    
                    # 如果 `}` 后面没有 EOS，或者有多个 `}`，强制返回 EOS
                    if len(json_end_indices) > 1:
                        logger.info(f"LOG (Early Exit): Multiple JSON end '}}' found ({len(json_end_indices)}). Forcing EOS.")
                        logger.info(f"DEBUG: newly_generated tokens (last 10): {newly_generated[-10:]}")
                        allowed_next = [self.eos_token_id]
                        logits_inspector.set_allowed_tokens(allowed_next)
                        return allowed_next
                    elif not has_eos_after:
                        logger.info("LOG (Early Exit): JSON end '}' found but no EOS after it. Forcing EOS.")
                        logger.info(f"DEBUG: newly_generated tokens (last 10): {newly_generated[-10:]}")
                        logger.info(f"DEBUG: tokens_after_json_end: {tokens_after_json_end}")
                        logger.info(f"DEBUG: EOS token ID: {self.eos_token_id}, JSON end token ID: {self.json_end_token_id}")
                        allowed_next = [self.eos_token_id]
                        logits_inspector.set_allowed_tokens(allowed_next)
                        return allowed_next

                # =================================================================
                # 状态 0: 强制生成 "prediction:"
                # =================================================================
                if num_generated < len(self.prediction_tokens):
                    next_token = self.prediction_tokens[num_generated]
                    allowed_next = [next_token]
                    logger.info(f"LOG (State 0): Forcing 'prediction:'. Returning {allowed_next}")
                    logits_inspector.set_allowed_tokens(allowed_next)
                    return allowed_next
                
                content_tokens = newly_generated[len(self.prediction_tokens):]
                logger.info(f"LOG: Content tokens after 'prediction:': {content_tokens}")

                # =================================================================
                # 状态 1: 强制生成 '{"h3_index": "'
                # =================================================================
                if len(content_tokens) < len(self.json_start_tokens):
                    offset = len(content_tokens)
                    next_token = self.json_start_tokens[offset]
                    allowed_next = [next_token]
                    logger.info(f"LOG (State 1): Forcing JSON start. Returning {allowed_next}")
                    logits_inspector.set_allowed_tokens(allowed_next)
                    return allowed_next

                # =================================================================
                # 核心逻辑：顺序解析H3部分和后续部分
                # =================================================================
                content_after_h3_start = content_tokens[len(self.json_start_tokens):]
                
                h3_tokens, post_h3_tokens = [], []
                temp_node, h3_ended = self.h3_trie, False
                for token_id in content_after_h3_start:
                    if not h3_ended and token_id in temp_node:
                        h3_tokens.append(token_id)
                        temp_node = temp_node[token_id]
                    else:
                        h3_ended = True
                        post_h3_tokens.append(token_id)
                
                logger.info(f"LOG (Parse): Parsed h3_tokens: {h3_tokens}")
                logger.info(f"LOG (Parse): Parsed post_h3_tokens: {post_h3_tokens}")

                # =================================================================
                # 状态判断与执行
                # =================================================================

                if not post_h3_tokens:
                    logger.info("LOG (Decision): In STATE_H3_INDEX (post_h3_tokens is empty).")
                    node = temp_node
                    allowed_next = [key for key in node.keys() if key != 'is_end']
                    if 'is_end' in node:
                        logger.info("LOG (State 2): H3 sequence is complete. Allowing next part to start.")
                        allowed_next.append(self.json_mid_tokens[0])
                    
                    if not allowed_next:
                        logger.info(f"!!! CRITICAL EXIT 1: Dead end in H3 Trie. Returning [].")
                        logits_inspector.set_allowed_tokens([])
                        return []
                    
                    logits_inspector.set_allowed_tokens(allowed_next)
                    return allowed_next
                else:
                    if len(post_h3_tokens) < len(self.json_mid_tokens):
                        logger.info("LOG (State 3): Forcing JSON mid part.")
                        if post_h3_tokens[0] != self.json_mid_tokens[0]:
                            logger.info(f"!!! CRITICAL EXIT 2: First post-H3 token {post_h3_tokens[0]} != expected {self.json_mid_tokens[0]}. Returning [].")
                            logits_inspector.set_allowed_tokens([])
                            return []
                        next_token = self.json_mid_tokens[len(post_h3_tokens)]
                        allowed_next = [next_token]
                        logits_inspector.set_allowed_tokens(allowed_next)
                        return allowed_next
                    duration_tokens = post_h3_tokens[len(self.json_mid_tokens):]
                   
                    # 分离出纯 duration 数值 tokens 和结构 tokens
                    duration_value_tokens = []
                    has_quote = False
                    has_json_end = False
                    has_eos = False
                    json_end_count = 0
                    
                    for token_id in duration_tokens:
                        if token_id == self.quote_token_id:
                            has_quote = True
                            break
                        elif token_id == self.json_end_token_id:
                            has_json_end = True
                            json_end_count += 1
                            # 如果已经有 `}`，后面的 token 应该都是 `}` 或 EOS
                            # 如果遇到非 `}` 非 EOS 的 token，说明有问题
                        elif token_id == self.eos_token_id:
                            has_eos = True
                            break
                        else:
                            # 如果已经有 `}`，但遇到其他 token，说明有问题
                            if has_json_end:
                                logger.info(f"!!! WARNING: Found token {token_id} after JSON end '}}'. This should not happen.")
                            duration_value_tokens.append(token_id)
                    
                    # 按优先级处理结构 token
                    if has_eos:
                        allowed_next = [self.eos_token_id]
                        logits_inspector.set_allowed_tokens(allowed_next)
                        return allowed_next

                    if has_json_end:
                        # 如果已经有 `}`，强制返回 EOS（无论有多少个 `}`）
                        allowed_next = [self.eos_token_id]
                        logits_inspector.set_allowed_tokens(allowed_next)
                        return allowed_next

                    if has_quote:
                        allowed_next = [self.json_end_token_id]
                        logits_inspector.set_allowed_tokens(allowed_next)
                        return allowed_next

                    # 遍历 duration trie
                    node = self.duration_trie
                    for token_id in duration_value_tokens:
                        if token_id not in node:
                            logger.info(f"!!! CRITICAL EXIT 3: Invalid duration token {token_id}. Returning [].")
                            logits_inspector.set_allowed_tokens([])
                            return []
                        node = node[token_id]

                    allowed_next = [key for key in node.keys() if key != 'is_end']
                    if 'is_end' in node:
                        allowed_next.append(self.quote_token_id)
                    
                    if not allowed_next:
                        logger.info(f"!!! CRITICAL EXIT 4: Dead end in Duration Trie. Returning [].")
                        logits_inspector.set_allowed_tokens([])
                        return []

                    logger.info(f"DEBUG: Allowed tokens count: {len(allowed_next)}, Token IDs: {allowed_next}")
                    if not allowed_next:
                        logger.info("!!! WARNING: allowed_next is empty! This will cause all tokens to be masked!")
                    logits_inspector.set_allowed_tokens(allowed_next)
                    return allowed_next

            except Exception as e:
                logger.info(f"!!! CRITICAL EXIT 5: An unexpected exception occurred: {e}")
                import traceback
                traceback.print_exc()
                logits_inspector.set_allowed_tokens([])
                return []

        return prefix_allowed_tokens_fn