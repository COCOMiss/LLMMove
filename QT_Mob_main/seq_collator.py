from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import torch
import json
import re

SEQ_RESPONSE_TAG = "prediction:"
END_TAG = "<|im_end|>"

class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_tag="prediction:", max_length=256, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.response_tag = SEQ_RESPONSE_TAG
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def _coerce_to_text_list(self, features: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """Coerce input features to a list of text."""
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("features must be a non-empty list")

        first = features[0]
        
        if isinstance(first, str):
            return [str(x) for x in features]
        
        if isinstance(first, dict):
            texts = []
            for ex in features:
                if "text" in ex:
                    texts.append(str(ex["text"]))
                elif "input_ids" in ex:
                    text = self.tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
                    texts.append(text)
                else:
                    raise ValueError(f"Dict element must have 'text' or 'input_ids' field. Got keys: {ex.keys()}")
            return texts
        
        raise ValueError("Unsupported feature element type; expected dict or str.")

    def _extract_json_block(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object appearing after response_tag."""
        pos = text.find(self.response_tag)
        if pos == -1:
            return None
        rest = text[pos + len(self.response_tag):]
        start = rest.find("{")
        if start == -1:
            return None
        i = start
        depth = 0
        while i < len(rest):
            if rest[i] == "{":
                depth += 1
            elif rest[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(rest[start:i + 1])
                    except Exception:
                        return None
            i += 1
        return None

    def extract_h3_index_and_duration(self, text: str):
        """Extract h3_index and stay_duration (minutes) from the JSON prediction block."""
        obj = self._extract_json_block(text)
        if not obj or not isinstance(obj, dict):
            return None, None
        h3_index = obj.get("h3_index")
        stay_duration = obj.get("stay_duration")
        if isinstance(stay_duration, str):
            digits = re.findall(r"\d+", stay_duration)
            stay_minutes = int(digits[0]) if digits else None
        elif isinstance(stay_duration, (int, float)):
            stay_minutes = int(stay_duration)
        else:
            stay_minutes = None
        return h3_index, stay_minutes

    def _mask_until_response(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Mask the input until the response for loss calculation."""
        # 🔍 调试输出
        # print(f"\n=== Collator Debug ===")
        # print(f"Processing {len(texts)} samples")
        
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()

        for i, text in enumerate(texts):
            pos = text.find(self.response_tag)
            if pos == -1:
                labels[i, :] = -100
                print(f"Sample {i}: ⚠️ No response tag found, masked all")
                continue

            cutoff_text = text[: pos + len(self.response_tag)]
            cutoff_ids = self.tokenizer(cutoff_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            cutoff_len = int(cutoff_ids.size(0))
            labels[i, :cutoff_len] = -100
            
            # 🔍 调试：计算 mask 比例
            total_len = input_ids[i].shape[0]
            active_len = total_len - cutoff_len
            # if i < 3:  # 只打印前 3 个
            #     print(f"Sample {i}: Total={total_len}, Masked={cutoff_len}, Active={active_len}, Ratio={active_len/total_len:.2%}")

        enc["labels"] = labels
        
        # ✅ 确保返回的格式是 SFTTrainer 期望的
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["labels"],
        }

    def __call__(self, features: List[Union[str, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        texts = self._coerce_to_text_list(features)
        return self._mask_until_response(texts)