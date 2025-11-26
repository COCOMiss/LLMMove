from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import torch
import json
import re
import numpy as np

# ！！！重要：与数据 & 测试严格一致
SEQ_RESPONSE_TAG = "prediction:"  # 样本中预测的目标部分（JSON 输出起始）
END_TAG = "<|im_end|>"  # 结束标签

class CompletionOnlyCollator:
    def __init__(
        self, 
        tokenizer, 
        response_tag="prediction:", 
        max_length=256, 
        pad_to_multiple_of=8,
        # ✅ 新参数：防止过拟合
        add_repeat_penalty=True,           # 是否添加重复惩罚
        repeat_penalty_weight=0.1,         # 重复惩罚权重
        no_repeat_ngram_size=2,            # 禁止 n-gram 重复
        data_augmentation=False,           # 数据增强
        dropout_prob=0.1,                  # dropout 概率
    ):
        self.tokenizer = tokenizer
        self.response_tag = response_tag
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        
        # ✅ 新参数
        self.add_repeat_penalty = add_repeat_penalty
        self.repeat_penalty_weight = repeat_penalty_weight
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.data_augmentation = data_augmentation
        self.dropout_prob = dropout_prob

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

    def _extract_json_block(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object appearing after response_tag."""
        pos = text.find(self.response_tag)
        if pos == -1:
            return None
        rest = text[pos + len(self.response_tag):]
        # Find the first JSON object via brace balancing
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
        # Normalize duration to integer minutes
        if isinstance(stay_duration, str):
            # Accept formats like "90min" -> 90
            digits = re.findall(r"\d+", stay_duration)
            stay_minutes = int(digits[0]) if digits else None
        elif isinstance(stay_duration, (int, float)):
            stay_minutes = int(stay_duration)
        else:
            stay_minutes = None
        return h3_index, stay_minutes

    # ✅ 新增：计算重复惩罚权重
    def _compute_repeat_penalty_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        计算重复惩罚掩码。
        对于重复的 token 或 n-gram，增加 loss 权重。
        
        Args:
            input_ids: shape (batch_size, seq_len)
            
        Returns:
            repeat_penalty: shape (batch_size, seq_len)，值为 1.0 或 > 1.0
        """
        batch_size, seq_len = input_ids.shape
        repeat_penalty = torch.ones_like(input_ids, dtype=torch.float32)
        
        for i in range(batch_size):
            seq = input_ids[i].tolist()
            
            # 1. 检查连续相同的 token（强烈惩罚）
            for j in range(1, len(seq)):
                if seq[j] == seq[j-1] and seq[j] != self.tokenizer.pad_token_id:
                    repeat_penalty[i, j] = 1.0 + self.repeat_penalty_weight * 3  # 连续重复最严重
            
            # 2. 检查 n-gram 重复
            if self.no_repeat_ngram_size > 1:
                for j in range(self.no_repeat_ngram_size, len(seq)):
                    ngram = tuple(seq[j-self.no_repeat_ngram_size:j])
                    # 在历史中查找相同的 n-gram
                    for k in range(j - self.no_repeat_ngram_size):
                        if tuple(seq[k:k+self.no_repeat_ngram_size]) == ngram:
                            repeat_penalty[i, j] = 1.0 + self.repeat_penalty_weight * 2
                            break
        
        return repeat_penalty

    # ✅ 新增：数据增强（随机打乱、混淆）
    def _data_augmentation(self, text: str) -> str:
        """
        简单的数据增强：
        - 随机删除部分 prompt 中的非关键词
        - 添加随机噪声
        """
        if not self.data_augmentation or np.random.random() > 0.3:
            return text
        
        # 只在 response 之前的部分进行增强
        pos = text.find(self.response_tag)
        if pos == -1:
            return text
        
        prompt_part = text[:pos]
        response_part = text[pos:]
        
        # 随机删除某些非关键词（保留 prompt 主要结构）
        words = prompt_part.split()
        if len(words) > 5:
            # 保留 80% 的词
            mask = np.random.rand(len(words)) > 0.2
            augmented_prompt = " ".join([w for w, keep in zip(words, mask) if keep])
            return augmented_prompt + response_part
        
        return text

    def _mask_until_response(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Mask the input until the response for loss calculation."""
        
        # ✅ 新增：数据增强
        if self.data_augmentation:
            texts = [self._data_augmentation(text) for text in texts]
        
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
        
        # ✅ 新增：添加重复惩罚权重到 enc 中
        if self.add_repeat_penalty:
            repeat_penalty = self._compute_repeat_penalty_mask(input_ids)
            enc["repeat_penalty"] = repeat_penalty
  
        return enc

    def __call__(self, features: List[Union[str, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        texts = self._coerce_to_text_list(features)
        return self._mask_until_response(texts)