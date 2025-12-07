from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import torch
import json
import re
from logger_utils import get_logger

logger = get_logger(__name__)

SEQ_RESPONSE_TAG = "prediction:"
END_TAG = "<|im_end|>"

# class CompletionOnlyCollator:
#     def __init__(self, tokenizer, response_tag="prediction:", max_length=4096, pad_to_multiple_of=8):
#         self.tokenizer = tokenizer
#         self.response_tag = SEQ_RESPONSE_TAG
#         self.token_response_tag = tokenizer.encode(SEQ_RESPONSE_TAG, add_special_tokens=False)
#         self.max_length = max_length
#         self.pad_to_multiple_of = pad_to_multiple_of
        
   

#     def _coerce_to_text_list(self, features: List[Union[str, Dict[str, Any]]]) -> List[str]:
#         """Coerce input features to a list of text."""
#         if not isinstance(features, list) or len(features) == 0:
#             logger.error("features must be a non-empty list")
#             raise ValueError("features must be a non-empty list")

#         first = features[0]
        
#          # ✅ 新增诊断：检查 features 的内容
#         logger.info(f"\n=== _coerce_to_text_list Debug ===")
#         logger.info(f"features type: {type(features)}, len: {len(features)}")
#         logger.info(f"first type: {type(first)}")
        
#         if isinstance(first, dict):
#             logger.info(f"first keys: {first.keys()}")
#             if "text" in first:
#                 logger.info(f"first['text'] length: {len(first['text'])}")
#                 logger.info(f"first['text'][:100]: {first['text'][:100]}")
#                 logger.info(f"first['text'][-100:]: {first['text'][-100:]}")
        
#         if isinstance(first, str):
#             logger.debug(f"Coercing {len(features)} string features to text list")
#             return [str(x) for x in features]
        
#         if isinstance(first, dict):
#             texts = []
#             for ex in features:
#                 if "text" in ex:
#                     text = str(ex["text"])
#                     texts.append(text)
#                     # ✅ 诊断：打印提取的文本
#                     logger. info(f"Extracted text length: {len(text)}")
#                     if len(texts) < 3:  # 只打印前 3 个
#                         logger.info(f"  First 100 chars: {text[:100]}")
#                         logger.info(f"  Last 100 chars: {text[-100:]}")
#                 elif "input_ids" in ex:
#                     text = self. tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
#                     texts.append(text)
#                     logger.info(f"Decoded from input_ids, length: {len(text)}")
#                 else:
#                     logger.error(f"Dict element must have 'text' or 'input_ids' field.  Got keys: {ex.keys()}")
#                     raise ValueError(f"Dict element must have 'text' or 'input_ids' field. Got keys: {ex.keys()}")
#             logger.debug(f"Coerced {len(features)} dict features to text list")
#             logger.info(f"Final texts list: {len(texts)} items")
#             return texts
    
#         logger.error(f"Unsupported feature element type: {type(first)}")
#         raise ValueError("Unsupported feature element type; expected dict or str.")
    
    
    
#     def _extract_json_block(self, text: str) -> Optional[Dict[str, Any]]:
#         """Extract the first JSON object appearing after response_tag."""
#         pos = text.find(self.response_tag)
#         if pos == -1:
#             logger.debug("Response tag not found in text")
#             return None
#         rest = text[pos + len(self.response_tag):]
#         start = rest.find("{")
#         if start == -1:
#             logger.debug("No JSON object found after response tag")
#             return None
#         i = start
#         depth = 0
#         while i < len(rest):
#             if rest[i] == "{":
#                 depth += 1
#             elif rest[i] == "}":
#                 depth -= 1
#                 if depth == 0:
#                     try:
#                         return json.loads(rest[start:i + 1])
#                     except Exception as e:
#                         logger.debug(f"Failed to parse JSON block: {e}")
#                         return None
#             i += 1
#         logger.debug("JSON block extraction incomplete (unclosed braces)")
#         return None

#     # def extract_h3_index_and_duration(self, text: str):
#     #     """Extract h3_index and stay_duration (minutes) from the JSON prediction block."""
#     #     obj = self._extract_json_block(text)
#     #     if not obj or not isinstance(obj, dict):
#     #         logger.debug("No valid JSON object extracted from text")
#     #         return None, None
#     #     h3_index = obj.get("h3_index")
#     #     stay_duration = obj.get("stay_duration")
#     #     if isinstance(stay_duration, str):
#     #         digits = re.findall(r"\d+", stay_duration)
#     #         stay_minutes = int(digits[0]) if digits else None
#     #         if stay_minutes is None:
#     #             logger.debug(f"Could not extract duration from string: {stay_duration}")
#     #     elif isinstance(stay_duration, (int, float)):
#     #         stay_minutes = int(stay_duration)
#     #     else:
#     #         logger.debug(f"Unsupported stay_duration type: {type(stay_duration)}")
#     #         stay_minutes = None
#     #     return h3_index, stay_minutes

#     def _mask_until_response(self, texts: List[str]) -> Dict[str, torch.Tensor]:
#         """Mask the input until the response for loss calculation."""
#         enc = self.tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#             pad_to_multiple_of=self.pad_to_multiple_of,
#         )
#         input_ids = enc["input_ids"]
#         labels = input_ids.clone()

#         for i, text in enumerate(texts):
#             pos = text.find(self.response_tag)
#             if pos == -1:
#                 labels[i, :] = -100
#                 logger.warning(f"Sample {i}: No response tag '{self.response_tag}' found, masked all labels")
#                 continue

#             cutoff_text = text[: pos + len(self.response_tag)]
#             cutoff_ids = self.tokenizer(cutoff_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
#             cutoff_len = int(cutoff_ids.size(0))
            
            
#             # 调试：计算 mask 比例（仅在前几个样本时记录）
#             total_len = input_ids[i].shape[0]
#             active_len = total_len - cutoff_len
#             if i < 3:  # 只记录前 3 个样本的详细信息
#                 logger.debug(f"Sample {i}: Total={total_len}, Masked={cutoff_len}, Active={active_len}, Ratio={active_len/total_len:.2%}")

#         enc["labels"] = labels
        
#         # ✅ 确保返回的格式是 SFTTrainer 期望的
#         return {
#             "input_ids": enc["input_ids"],
#             "attention_mask": enc["attention_mask"],
#             "labels": enc["labels"],
#         }

    
    
    
    
   
#     def __call__(self, features: List[Union[str, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
#         logger.debug(f"Collator called with {len(features)} features")
#         texts = self._coerce_to_text_list(features)
#         result = self._mask_until_response(texts)
#         logger.debug(f"Collator returning batch with input_ids shape: {result['input_ids'].shape}")
#         return result



def prepare_tokenized_dataset(data_list, tokenizer, max_length, response_tag, postfix):
    """
    预处理数据，确保 response_tag 之后的内容完整
    支持两种输入格式：
    1.  {"input_ids": str, "labels": str}  - 旧格式
    2. {"prompt": str, "completion": str} - 新格式
    """
    
    processed = []
    skipped = 0
    
    for idx, item in enumerate(data_list):
        # ✅ 支持两种输入格式
        if "prompt" in item and "completion" in item:
            # 新格式：prompt + completion
            prompt_text = item["prompt"]
            completion_text = item["completion"] + postfix
            full_text = prompt_text + completion_text
            
            # 对于这种格式，我们直接用 prompt 和 completion 的边界
            # 而不是依赖 response_tag
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
            
        elif "input_ids" in item and "labels" in item:
            # 旧格式：input_ids + labels（都是字符串）
            full_text = item["input_ids"] + item["labels"] + postfix
            
            tag_pos = full_text.find(response_tag)
            if tag_pos == -1:
                skipped += 1
                continue
            
            prompt_text = full_text[:tag_pos + len(response_tag)]
            completion_text = full_text[tag_pos + len(response_tag):]
            
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        else:
            print(f"⚠️ Sample {idx}: Unknown format, keys = {item.keys()}")
            skipped += 1
            continue
        
        # 截断逻辑：优先保证 completion 完整
        total_len = len(prompt_ids) + len(completion_ids)
        
        if total_len > max_length:
            available_for_prompt = max_length - len(completion_ids) - 20
            
            if available_for_prompt < 200:
                # completion 太长，需要截断
                print(f"⚠️ Sample {idx}: completion too long ({len(completion_ids)}), truncating")
                completion_ids = completion_ids[:max_length - 200]
                available_for_prompt = 200
            
            if len(prompt_ids) > available_for_prompt:
                # 截断 prompt，保留头尾
                keep_start = int(available_for_prompt * 0.6)
                keep_end = available_for_prompt - keep_start
                prompt_ids = prompt_ids[:keep_start] + prompt_ids[-keep_end:]
        
        # 组合 input_ids 和 labels
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        
        processed. append({
            "input_ids": input_ids,
            "labels": labels,
        })
    
    print(f"✅ Processed {len(processed)} samples, skipped {skipped}")
    return processed

# class SimpleCollator:
#     """
#     简单的 Collator，支持两种数据格式：
#     1.  已预处理的 input_ids + labels 格式
#     2. SFTTrainer 处理后的 input_ids + completion_mask 格式
#     """
    
#     def __init__(self, tokenizer, max_length=4096, pad_to_multiple_of=8, debug=True):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.pad_to_multiple_of = pad_to_multiple_of
#         self.pad_token_id = tokenizer. pad_token_id if tokenizer.pad_token_id is not None else 0
#         self.debug = debug
#         self.debug_count = 0
    
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
#         # ✅ 检测数据格式并打印（调试用）
#         if self.debug and self.debug_count == 0:
#             print(f"🔍 Collator received keys: {features[0].keys()}")
        
#         # 获取 batch 中的最大长度
#         max_len = max(len(f["input_ids"]) for f in features)
#         max_len = min(max_len, self.max_length)
        
#         if self.pad_to_multiple_of:
#             max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
#         batch_input_ids = []
#         batch_attention_mask = []
#         batch_labels = []
        
#         for f in features:
#             input_ids = f["input_ids"][:self. max_length]
            
#             # ✅ 关键修改：支持两种格式
#             if "labels" in f:
#                 # 格式1：已经有 labels（预处理过的数据）
#                 labels = f["labels"][:self.max_length]
#             elif "completion_mask" in f:
#                 # 格式2：SFTTrainer 处理后的 completion_mask
#                 # completion_mask: 1 表示 completion（需要计算 loss），0 表示 prompt（不计算 loss）
#                 completion_mask = f["completion_mask"][:self.max_length]
#                 labels = [
#                     token_id if mask == 1 else -100 
#                     for token_id, mask in zip(input_ids, completion_mask)
#                 ]
#             else:
#                 # 格式3：没有任何 mask，全部计算 loss（不推荐）
#                 print("⚠️ Warning: No labels or completion_mask found, using input_ids as labels")
#                 labels = list(input_ids)
                
#              # Padding
#             pad_len = max_len - len(input_ids)
            
            
#             if "completion_mask" in f:
#                 completion_mask = f["completion_mask"][:self.max_length]
#                 completion_mask = list(completion_mask) if not isinstance(completion_mask, list) else completion_mask
#                 attention_mask = completion_mask + [0] * pad_len
#             elif "attention_mask" in f:
#                 attention_mask = f["attention_mask"][:self.max_length]
#                 attention_mask = list(attention_mask) if not isinstance(attention_mask, list) else attention_mask
#                 attention_mask = attention_mask + [0] * pad_len
#             else:
#                 attention_mask = [1] * len(input_ids)
#                 attention_mask = list(attention_mask) if not isinstance(attention_mask, list) else attention_mask
#                 attention_mask = attention_mask + [0] * pad_len
            
            
#             # 确保是 list
#             input_ids = list(input_ids) if not isinstance(input_ids, list) else input_ids
#             labels = list(labels) if not isinstance(labels, list) else labels
            
            
#             input_ids = input_ids + [self.pad_token_id] * pad_len
#             labels = labels + [-100] * pad_len
            
#             batch_input_ids.append(input_ids)
#             batch_attention_mask.append(attention_mask)
#             batch_labels.append(labels)
        
#         # Debug 输出
#         if self.debug and self.debug_count < 3:
#             self._debug_batch(features, batch_input_ids, batch_labels)
#             self.debug_count += 1
        
#         if "labels" in f:
#             return {
#                 "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
#                 "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
#                 "labels": torch. tensor(batch_labels, dtype=torch. long),
#             }
#         else:
           
#             return {
#                 "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
#                 "completion_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
#             }
      
    
#     def _debug_batch(self, features, batch_input_ids, batch_labels):
#         """调试：解码并打印 input 和 label"""
#         print("\n" + "=" * 80)
#         print("🔍 DEBUG: Collator Output Inspection")
#         print(f"📋 Input feature keys: {features[0].keys()}")
#         print("=" * 80)
        
#         for i, (input_ids, labels) in enumerate(zip(batch_input_ids, batch_labels)):
#             if i >= 2:
#                 break
                
#             print(f"\n--- Sample {i} ---")
            
#             # 1. 统计
#             num_ignored = sum(1 for l in labels if l == -100)
#             num_valid = sum(1 for l in labels if l != -100)
#             print(f"📊 Labels 统计: 总长度={len(labels)}, 忽略(-100)={num_ignored}, 有效={num_valid}")
            
#             # 2. 解码完整输入
#             input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
#             print(f"\n📝 完整输入 (前500字符):\n{input_text[:500]}...")
            
#             # 3. 解码有效 label
#             valid_label_ids = [l for l in labels if l != -100]
#             if valid_label_ids:
#                 label_text = self. tokenizer. decode(valid_label_ids, skip_special_tokens=False)
#                 print(f"\n🎯 有效 Label 内容 (前500字符):\n{label_text[:500]}...")
#             else:
#                 print("\n⚠️ 警告: 所有 label 都是 -100!")
            
#             # 4.  找到转换位置
#             transition_idx = None
#             for idx, l in enumerate(labels):
#                 if l != -100:
#                     transition_idx = idx
#                     break
            
#             if transition_idx is not None:
#                 print(f"\n🔀 Label 转换位置: index={transition_idx}")
#                 context_start = max(0, transition_idx - 20)
#                 context_end = min(len(input_ids), transition_idx + 20)
#                 context_ids = input_ids[context_start:context_end]
#                 context_text = self.tokenizer.decode(context_ids, skip_special_tokens=False)
#                 print(f"📍 转换点上下文: ... {context_text}...")
            
#             # 5. 检查一致性
#             mismatch_count = 0
#             for idx, (inp, lab) in enumerate(zip(input_ids, labels)):
#                 if lab != -100 and inp != lab:
#                     mismatch_count += 1
#                     if mismatch_count <= 5:
#                         print(f"⚠️ 不匹配 @ {idx}: input_id={inp}, label={lab}")
            
#             if mismatch_count == 0:
#                 print("✅ input_ids 和 labels (非-100部分) 完全一致")
#             else:
#                 print(f"❌ 发现 {mismatch_count} 处不匹配")
        
#         print("\n" + "=" * 80 + "\n")

class SimpleCollator:
    """简单的 Collator，数据已经预处理好，只需要 padding"""
    
    def __init__(self, tokenizer, max_length=4096, pad_to_multiple_of=8, debug=True, debug_count=3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer. pad_token_id if tokenizer.pad_token_id is not None else 0
        self.debug = debug
        self.debug_count = 0  # 只打印前几个样本
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 获取 batch 中的最大长度
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.max_length)
        
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for f in features:
            input_ids = f["input_ids"][:self.max_length]
            labels = f["labels"][:self.max_length]
            attention_mask = f. get("attention_mask", [1] * len(input_ids))[:self.max_length]
            
            # Padding
            pad_len = max_len - len(input_ids)
            
            input_ids = list(input_ids) + [self.pad_token_id] * pad_len
            attention_mask = list(attention_mask) + [0] * pad_len
            labels = list(labels) + [-100] * pad_len
            
            batch_input_ids. append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        
        if self.debug and self.debug_count < 3:  # 只打印前3个batch
            self._debug_batch(features, batch_input_ids, batch_labels)
            self.debug_count += 1
        # ==============================
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch. tensor(batch_labels, dtype=torch. long),
        }
        
    def _debug_batch(self, features, batch_input_ids, batch_labels):
        """调试：解码并打印 input 和 label"""
        print("\n" + "=" * 80)
        print("🔍 DEBUG: Collator Output Inspection")
        print("=" * 80)
        
        for i, (input_ids, labels) in enumerate(zip(batch_input_ids, batch_labels)):
            if i >= 2:  # 每个 batch 只看前2个样本
                break
                
            print(f"\n--- Sample {i} ---")
            
            # 1. 统计 label 中 -100 的数量
            num_ignored = sum(1 for l in labels if l == -100)
            num_valid = sum(1 for l in labels if l != -100)
            print(f"📊 Labels 统计: 总长度={len(labels)}, 忽略(-100)={num_ignored}, 有效={num_valid}")
            
            # 2. 解码完整输入
            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            print(f"\n📝 完整输入 (前500字符):\n{input_text}...")
            
            # 3. 解码有效 label（非 -100 的部分）
            valid_label_ids = [l for l in labels if l != -100]
            if valid_label_ids:
                label_text = self. tokenizer.decode(valid_label_ids, skip_special_tokens=False)
                print(f"\n🎯 有效 Label 内容 (前500字符):\n{label_text}...")
            else:
                print("\n⚠️ 警告: 所有 label 都是 -100!")
            
            # 4. 找到 label 从 -100 变为有效值的位置
            transition_idx = None
            for idx, l in enumerate(labels):
                if l != -100:
                    transition_idx = idx
                    break
            
            if transition_idx is not None:
                print(f"\n🔀 Label 转换位置: index={transition_idx}")
                # 解码转换点前后的文本
                context_start = max(0, transition_idx - 20)
                context_end = min(len(input_ids), transition_idx + 20)
                context_ids = input_ids[context_start:context_end]
                context_text = self. tokenizer.decode(context_ids, skip_special_tokens=False)
                print(f"📍 转换点上下文: ... {context_text}...")
            
            # 5. 对比 input_ids 和 labels 是否一致（非 -100 部分）
            mismatch_count = 0
            for idx, (inp, lab) in enumerate(zip(input_ids, labels)):
                if lab != -100 and inp != lab:
                    mismatch_count += 1
                    if mismatch_count <= 5:  # 只打印前5个不匹配
                        print(f"⚠️ 不匹配 @ {idx}: input_id={inp}, label={lab}")
            
            if mismatch_count == 0:
                print("✅ input_ids 和 labels (非-100部分) 完全一致")
            else:
                print(f"❌ 发现 {mismatch_count} 处不匹配")
        
        print("\n" + "=" * 80 + "\n")
        
       