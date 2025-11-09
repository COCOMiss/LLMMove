from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import torch

# # ！！！重要：与数据 & 测试严格一致
SEQ_RESPONSE_TAG = " will stay at h3 index "  # 样本中预测的目标部分
END_TAG = "<|im_end|>"  # 结束标签

# @dataclass
# class CompletionOnlyCollator:
#     tokenizer: Any
#     response_tag: str = SEQ_RESPONSE_TAG  # 默认为预测h3 index
#     max_length: int = 256
#     pad_to_multiple_of: Optional[int] = 8  # 便于 TensorCore

#     def _coerce_to_text_list(self, features: List[Union[str, Dict[str, Any]]]) -> List[str]:
#         """处理输入数据，将 input_ids 转换为文本。"""
#         texts = []
#         for feature in features:
#             if isinstance(feature, dict) and "input_ids" in feature:
#                 # Decode input_ids 为文本
#                 text = self.tokenizer.decode(feature["input_ids"], skip_special_tokens=True)
#                 texts.append(text)
#             else:
#                 raise ValueError("Features must contain 'input_ids' for processing")
#         return texts

#     def _mask_until_response(self, texts: List[str]) -> Dict[str, torch.Tensor]:
#         """将 RESPONSE_TAG 之前的 labels 置为 -100，只计算答案部分的损失。"""
#         # 将文本转换为 token ids
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

#         # 遍历每个样本，查找 RESPONSE_TAG 的位置并将其前面的 tokens 设置为 -100
#         for i, text in enumerate(texts):
#             pos = text.find(self.response_tag)  # 查找 response_tag 的位置
#             if pos == -1:
#                 print(f"[WARN] SEQ_RESPONSE_TAG not found in: {text}")
#                 labels[i, :] = -100  # 没找到则不计算损失
#                 continue

#             # 通过 RESPONSE_TAG 截取前部分并将其对应的 input_ids 设置为 -100
#             cutoff_text = text[: pos + len(self.response_tag)]
#             cutoff_ids = self.tokenizer(
#                 cutoff_text, add_special_tokens=False, return_tensors="pt"
#             )["input_ids"][0]
#             cutoff_len = int(cutoff_ids.size(0))
#             labels[i, :cutoff_len] = -100

#         enc["labels"] = labels
#         return enc

#     def __call__(self, features: List[Union[str, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
#         """
#         接收 features（包含 input_ids 的字典），
#         进行解码并调用 _mask_until_response 生成适合计算损失的 labels
#         """
#         texts = self._coerce_to_text_list(features)
#         return self._mask_until_response(texts)


import re
from typing import List, Dict, Any, Union

class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_tag=" will stay at h3 index ", max_length=256, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.response_tag = response_tag
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def _coerce_to_text_list(self, features: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """Coerce input features (input_ids) to a list of text."""
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("features must be a non-empty list")

        first = features[0]
        
        if isinstance(first, str):
            return [str(x) for x in features]
        
        if isinstance(first, dict):
            texts = []
            for ex in features:
                if "input_ids" in ex:
                    text = self.tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
                    texts.append(text)
            return texts
        raise ValueError("Unsupported feature element type; expected dict or str.")

    def extract_h3_index_and_duration(self, text: str):
        """Extract the H3 index and duration from the decoded text."""
        h3_index = re.search(r"h3 index (\S+)", text)
        duration = re.search(r"for (\S+) seconds", text)
        
        if h3_index:
            h3_index = h3_index.group(1)
        else:
            h3_index = None
        
        if duration:
            duration = float(duration.group(1))
        else:
            duration = None
        
        return h3_index, duration

    def _mask_until_response(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Mask the input until the response for loss calculation."""
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

        # Extract h3_index and duration
        h3_index_labels = []
        duration_labels = []

        for i, text in enumerate(texts):
            h3_index, duration = self.extract_h3_index_and_duration(text)
            h3_index_labels.append(h3_index)
            duration_labels.append(duration)
            
            pos = text.find(self.response_tag)
            if pos == -1:
                labels[i, :] = -100  # Masking the input part if no response found
                continue

            cutoff_text = text[: pos + len(self.response_tag)]
            cutoff_ids = self.tokenizer(cutoff_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            cutoff_len = int(cutoff_ids.size(0))
            labels[i, :cutoff_len] = -100

        enc["labels"] = labels
        enc["h3_index_labels"] = h3_index_labels
        enc["duration_labels"] = duration_labels
        return enc

    def __call__(self, features: List[Union[str, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        texts = self._coerce_to_text_list(features)
        return self._mask_until_response(texts)
