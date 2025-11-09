import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import SFTTrainer

class DualTaskTrainer(SFTTrainer):
    def __init__(self, *args, duration_loss_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # 从模型配置里取 hidden_size，构造一个极小的回归头
        hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size from model.config; please set one of hidden_size / n_embd.")
        self.duration_head = nn.Linear(hidden_size, 1)
        # 放到和模型相同的设备
        self.duration_head.to(next(self.model.parameters()).device)
        self.duration_loss_weight = float(duration_loss_weight)

    def _mean_pool_last_hidden(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        对最后一层隐状态做 mask 平均池化，得到每个样本一个向量 (B, H)
        last_hidden_state: (B, T, H)
        attention_mask:    (B, T)
        """
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, T, 1)
        summed = (last_hidden_state * mask).sum(dim=1)                    # (B, H)
        denom = mask.sum(dim=1).clamp_min(1e-6)                           # (B, 1)
        return summed / denom

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """
        标准 CE + 回归 MSE
        需要 collator 返回：
          - input_ids, attention_mask, labels（常规 SFT）
          - duration_labels（list[float] 或 tensor）
        """
        # 1) 先做常规的 CE（只把模型需要的键传进去，避免 HF 抱怨）
        model_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "labels")}
        # 让模型返回 hidden_states 以做回归
        outputs = model(**model_inputs, output_hidden_states=True)
        ce_loss = outputs.loss  # 这是你原有的 token-level CE

        total_loss = ce_loss

        # 2) 如果提供了 duration_labels，则做回归损失
        if "duration_labels" in inputs and inputs["duration_labels"] is not None:
            duration_labels = inputs["duration_labels"]
            # list -> tensor，并放到正确设备 / dtype
            if isinstance(duration_labels, (list, tuple)):
                duration_labels = torch.tensor(duration_labels, dtype=torch.float32)
            duration_labels = duration_labels.to(next(model.parameters()).device).float().view(-1, 1)  # (B,1)

            # 取最后一层隐状态 (B, T, H)，做 mean-pool 得到 (B, H)
            last_hidden = outputs.hidden_states[-1]
            pooled = self._mean_pool_last_hidden(last_hidden, model_inputs["attention_mask"])  # (B, H)
            pred_duration = self.duration_head(pooled)  # (B, 1)

            duration_mse = F.mse_loss(pred_duration, duration_labels)
            total_loss = total_loss + self.duration_loss_weight * duration_mse

            # 可选：记录到日志
            self.log({"ce_loss": ce_loss.detach().float(),
                      "duration_mse": duration_mse.detach().float()})

        return (total_loss, outputs) if return_outputs else total_loss
