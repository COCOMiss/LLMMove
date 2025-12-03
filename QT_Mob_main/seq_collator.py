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



class SimpleCollator:
    """简单的 Collator，数据已经预处理好，只需要 padding"""
    
    def __init__(self, tokenizer, max_length=4096, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer. pad_token_id if tokenizer.pad_token_id is not None else 0
    
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
        
        return {
            "input_ids": torch. tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch. long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }