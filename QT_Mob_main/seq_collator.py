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


# 在 seq_collator. py 或数据处理文件中修改
import torch

SEQ_RESPONSE_TAG = "Predicted trajectory:"  # 根据你的实际 tag 修改


def prepare_tokenized_dataset_with_yesterday(
    data_list, 
    tokenizer, 
    max_length, 
    response_tag, 
    postfix
):
    """预处理数据集，保留前一天轨迹用于复制惩罚"""
    processed = []
    
    for item in data_list:
        full_text = item["input_ids"] + item["labels"] + postfix
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        input_ids = tokenized["input_ids"]
        
        # 确保有 attention_mask
        attention_mask = tokenized. get("attention_mask", [1] * len(input_ids))
        
        # 找到 response 开始位置
        response_start = full_text.find(response_tag)
        if response_start != -1:
            instruction_part = full_text[:response_start + len(response_tag)]
            instruction_tokens = tokenizer(instruction_part, add_special_tokens=True)["input_ids"]
            response_start_idx = len(instruction_tokens)
        else:
            response_start_idx = len(input_ids) // 2
        
        # 构建 labels
        labels = [-100] * response_start_idx + input_ids[response_start_idx:]
        labels = labels[:len(input_ids)]
        
        # 提取前一天轨迹
        yesterday_traj = item. get("last_day_traj", "")
        if yesterday_traj and isinstance(yesterday_traj, str) and len(yesterday_traj. strip()) > 0:
            yesterday_token_ids = tokenizer(
                yesterday_traj, 
                add_special_tokens=False,
                truncation=True,
                max_length=max_length // 2
            )["input_ids"]
        else:
            yesterday_token_ids = []
        
        processed.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "yesterday_token_ids": yesterday_token_ids,
        })
    
    return processed


class SimpleCollatorWithYesterday:
    """支持复制惩罚的 Collator"""
    
    def __init__(self, tokenizer, max_length, debug=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer. pad_token_id if tokenizer.pad_token_id is not None else 0
        self.debug = debug
        self._first_call = True
    
    def __call__(self, batch): 
        # 处理 input_ids
        input_ids_list = []
        for item in batch:
            if isinstance(item["input_ids"], torch.Tensor):
                input_ids_list.append(item["input_ids"])
            else:
                input_ids_list.append(torch.tensor(item["input_ids"]))
        
        # 处理 labels
        labels_list = []
        for item in batch:
            if isinstance(item["labels"], torch.Tensor):
                labels_list.append(item["labels"])
            else:
                labels_list.append(torch.tensor(item["labels"]))
        
        # 处理 attention_mask
        attention_mask_list = []
        for item in batch:
            if "attention_mask" in item:
                if isinstance(item["attention_mask"], torch.Tensor):
                    attention_mask_list.append(item["attention_mask"])
                else:
                    attention_mask_list.append(torch.tensor(item["attention_mask"]))
            else:
                # 自动生成
                attention_mask_list.append(torch.ones(len(item["input_ids"]), dtype=torch.long))
        
        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn. pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils. rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        
        # 处理 yesterday_token_ids
        yesterday_padded = None
        if batch and "yesterday_token_ids" in batch[0]:
            yesterday_lists = []
            for item in batch:
                yt = item. get("yesterday_token_ids", [])
                if isinstance(yt, torch.Tensor):
                    yesterday_lists.append(yt. tolist())
                else:
                    yesterday_lists.append(yt if yt else [])
            
            max_yesterday_len = max(len(y) for y in yesterday_lists) if yesterday_lists else 0
            
            if max_yesterday_len > 0:
                yesterday_padded = torch.full(
                    (len(batch), max_yesterday_len), 
                    self.pad_token_id, 
                    dtype=torch.long
                )
                for i, y in enumerate(yesterday_lists):
                    if len(y) > 0:
                        yesterday_padded[i, :len(y)] = torch.tensor(y)
        
        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        
        if yesterday_padded is not None:
            result["yesterday_token_ids"] = yesterday_padded
        
        return result

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
 
       