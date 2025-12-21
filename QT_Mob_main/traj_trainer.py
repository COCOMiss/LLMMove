import torch
import torch.nn. functional as F
from trl import SFTTrainer
from logger_utils import get_logger


import torch
import torch.nn. functional as F
from trl import SFTTrainer



logger = get_logger(__name__)
logger.info("==== TrajTrainer started ====")
class SoftPositionMatchCopyPenaltyTrainer(SFTTrainer):
    """
    基于位置匹配的 Soft Matching 复制惩罚 Trainer（完全可微分）
    
    修复版本：确保梯度正确反向传播
    """
    
    def __init__(
        self, 
        *args, 
        copy_penalty_weight=0.3, 
        copy_threshold=0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.copy_penalty_weight = copy_penalty_weight
        self. copy_threshold = copy_threshold
        self._step_count = 0
        self._total_copy_penalty = 0.0
        self._total_std_loss = 0.0
        logger.info(
            f"✅ SoftPositionMatchCopyPenaltyTrainer initialized: "
            f"weight={copy_penalty_weight}, threshold={copy_threshold}"
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """重写损失计算，添加 soft position match 复制惩罚"""
        
        # 提取 yesterday_token_ids
        yesterday_token_ids = inputs.pop("yesterday_token_ids", None)
        
        # 计算标准损失
        outputs = model(**inputs)
        standard_loss = outputs.loss
        
        # 计算复制惩罚
        copy_penalty = torch.tensor(0.0, device=standard_loss.device)
        debug_info = {'avg_copy_prob': 0.0, 'has_gradient': False}
        
        if yesterday_token_ids is not None and self. copy_penalty_weight > 0:
            copy_penalty, debug_info = self._compute_soft_position_penalty(
                outputs.logits, 
                inputs["labels"], 
                yesterday_token_ids
            )
            total_loss = standard_loss + self.copy_penalty_weight * copy_penalty
        else:
            total_loss = standard_loss
        
        # 日志记录
        self._step_count += 1
        self._total_copy_penalty += copy_penalty.item()
        self._total_std_loss += standard_loss.item()
        
        if self._step_count % 100 == 0:
            avg_copy = self._total_copy_penalty / 100
            avg_std = self._total_std_loss / 100
            logger.info(
                f"Step {self._step_count}: "
                f"std_loss={standard_loss.item():.4f} (avg={avg_std:.4f}), "
                f"copy_penalty={copy_penalty.item():.4f} (avg={avg_copy:.4f}), "
                f"total={total_loss.item():.4f}, "
                f"avg_copy_prob={debug_info['avg_copy_prob']:.4f}, "
                f"has_grad={debug_info['has_gradient']}"
            )
            self._total_copy_penalty = 0.0
            self._total_std_loss = 0.0
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _compute_soft_position_penalty(self, logits, labels, yesterday_token_ids):
        """
        完全可微分的位置匹配惩罚
        
        确保梯度可以正确反向传播到模型参数
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # 1. 有效位置 mask
        valid_mask = (labels != -100).float()
        total_valid = valid_mask.sum()
        
        if total_valid == 0:
            # 返回需要梯度的零张量
            zero_loss = (logits * 0).sum()  # 保持计算图连接
            return zero_loss, {'avg_copy_prob': 0.0, 'has_gradient': True}
        
        # 2. 计算 softmax 概率（可微分）
        probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        
        # 3. 获取 pad_token_id
        pad_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id else 0
        
        # 4. 构建位置对齐的 yesterday token 索引
        aligned_yesterday = self._align_yesterday_to_response(
            labels, yesterday_token_ids, pad_id, seq_len, device
        )
        
        # 5. 限制在 vocab 范围内
        aligned_yesterday = aligned_yesterday. clamp(0, vocab_size - 1)
        
        # 6. 使用 gather 获取每个位置对应 yesterday token 的概率（可微分）
        gather_indices = aligned_yesterday.unsqueeze(-1)
        position_copy_probs = probs.gather(dim=-1, index=gather_indices). squeeze(-1)
        
        # 7. 只考虑有效位置且 yesterday 有效的位置
        yesterday_valid_mask = (aligned_yesterday != pad_id).float()
        combined_mask = valid_mask * yesterday_valid_mask
        
        # 8. 计算平均复制概率（可微分）
        masked_copy_probs = position_copy_probs * combined_mask
        total_combined = combined_mask.sum()
        
        if total_combined == 0:
            zero_loss = (logits * 0).sum()
            return zero_loss, {'avg_copy_prob': 0.0, 'has_gradient': True}
        
        avg_copy_prob = masked_copy_probs.sum() / (total_combined + 1e-8)
        
        # 9. 使用软阈值计算惩罚（可微分）
        # 方法1：直接使用 avg_copy_prob（始终有梯度）
        penalty = avg_copy_prob
        
        # 方法2：使用 soft ReLU（Softplus）保持平滑梯度
        # penalty = F.softplus(avg_copy_prob - self.copy_threshold)
        
        # 方法3：使用标准 ReLU（超过阈值才有梯度）
        # penalty = F.relu(avg_copy_prob - self.copy_threshold) / (1.0 - self.copy_threshold + 1e-8)
        
        # 检查是否有梯度
        has_gradient = penalty.requires_grad and (avg_copy_prob > self.copy_threshold). item()
        
        debug_info = {
            'avg_copy_prob': avg_copy_prob.item(),
            'has_gradient': has_gradient
        }
        
        return penalty, debug_info
    
    def _align_yesterday_to_response(self, labels, yesterday_token_ids, pad_id, seq_len, device):
        """
        将 yesterday_token_ids 对齐到 response 位置
        """
        batch_size = labels.size(0)
        aligned = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            # 找到 response 开始位置
            valid_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
            
            response_start = valid_positions[0]. item()
            response_len = len(valid_positions)
            
            # 获取有效的 yesterday tokens
            yesterday = yesterday_token_ids[i]
            yesterday_valid = yesterday[yesterday != pad_id]
            yesterday_len = len(yesterday_valid)
            
            # 对齐
            copy_len = min(response_len, yesterday_len)
            if copy_len > 0:
                aligned[i, response_start:response_start + copy_len] = yesterday_valid[:copy_len]
        
        return aligned
   