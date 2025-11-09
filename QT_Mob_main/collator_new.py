import torch
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataCollatorForCompletionOnlyLM:
    response_template_ids: List[int]
    tokenizer: any
    max_length: int = 4096
    pad_token_id: int = None

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        模拟原 HuggingFace TRL 的 DataCollatorForCompletionOnlyLM，
        支持 (response_template_ids, tokenizer) 参数形式。
        """
        # 提取文本数据
        texts = [ex["text"] for ex in examples]

        # 编码
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        labels = input_ids.clone()

        # 忽略 pad token
        pad_token_id = self.pad_token_id or self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        labels[labels == pad_token_id] = -100

        # ✅ 如果有 response_template_ids，就只训练 response 区段
        if self.response_template_ids:
            for i in range(len(labels)):
                input_list = input_ids[i].tolist()
                for j in range(len(input_list) - len(self.response_template_ids)):
                    if input_list[j:j + len(self.response_template_ids)] == self.response_template_ids:
                        # Mask掉指令部分，仅训练response部分
                        labels[i, : j + len(self.response_template_ids)] = -100
                        break

        enc["labels"] = labels
        return enc
