import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RepetitionAwareTrainer(Trainer):
    """
    自定义 Trainer，在 loss 中添加重复惩罚项。
    防止模型在训练时过度拟合和重复生成。
    """
    
    def __init__(self, *args, repeat_penalty_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeat_penalty_weight = repeat_penalty_weight
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算带有重复惩罚的 loss。
        """
        # 获取标准的 loss
        labels = inputs.get("labels")
        repeat_penalty = inputs.pop("repeat_penalty", None)  # 从 collator 获取
        
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
        
        # 计算标准的语言模型 loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 标准 cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                         shift_labels.view(-1))
        
        # ✅ 添加重复惩罚
        if repeat_penalty is not None:
            repeat_penalty = repeat_penalty[..., 1:].contiguous()  # 对齐
            losses = losses * repeat_penalty.view(-1)
        
        # 掩码：将 label 为 -100 的位置的 loss 设为 0
        mask = shift_labels.view(-1) != -100
        loss = (losses * mask).sum() / mask.sum().clamp(min=1)
        
        return (loss, outputs) if return_outputs else loss
