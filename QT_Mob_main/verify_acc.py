import torch
from typing import Dict, Optional, Tuple, Any, List
from functools import partial
from torch.utils.data import DataLoader
from logger_utils import get_logger

logger = get_logger(__name__)


def get_autocast_context(device):
    """根据设备返回合适的 autocast context"""
    if device is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        # CPU 或其他设备，使用空 context
        import contextlib
        return contextlib.nullcontext()


def simple_collate_fn_for_prompt_completion(
    features: List[Dict[str, Any]], 
    tokenizer,
    max_length: int = 4096
) -> Dict[str, torch.Tensor]:
    """适用于 prompt/completion 格式的 collate 函数"""
    
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for f in features:
        # 检查数据格式
        if "prompt" in f and "completion" in f:
            # prompt/completion 格式 - 需要 tokenize
            prompt_ids = tokenizer.encode(f["prompt"], add_special_tokens=True)
            completion_ids = tokenizer.encode(f["completion"], add_special_tokens=False)
            
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            
        elif "input_ids" in f and "labels" in f:
            # 已经 tokenize 的格式
            input_ids = f["input_ids"]
            labels = f["labels"]
            
            if hasattr(input_ids, 'tolist'):
                input_ids = input_ids.tolist()
            if hasattr(labels, 'tolist'):
                labels = labels.tolist()
        else:
            raise ValueError(f"Unknown data format: {f. keys()}")
        
        # 截断
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        
        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append([1] * len(input_ids))
        batch["labels"].append(labels)
    
    # Padding
    max_len = max(len(ids) for ids in batch["input_ids"])
    pad_token_id = tokenizer.pad_token_id or 0
    
    for i in range(len(batch["input_ids"])):
        pad_len = max_len - len(batch["input_ids"][i])
        batch["input_ids"][i] = batch["input_ids"][i] + [pad_token_id] * pad_len
        batch["attention_mask"][i] = batch["attention_mask"][i] + [0] * pad_len
        batch["labels"][i] = batch["labels"][i] + [-100] * pad_len
    
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

def compute_token_accuracy(
    logits: torch. Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> Tuple[float, int, int]:
    """计算 token 级别的准确率，与 SFTTrainer 的实现一致。"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    predictions = shift_logits.argmax(dim=-1)
    mask = shift_labels != ignore_index
    
    correct_predictions = (predictions == shift_labels) & mask
    correct_tokens = correct_predictions. sum(). item()
    total_tokens = mask.sum().item()
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return accuracy, correct_tokens, total_tokens


def compute_batch_token_accuracy(
    model,
    batch: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """对一个 batch 计算 token 准确率。"""
    model. eval()
    
    if device is not None:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # ✅ 使用 autocast
    with torch.no_grad():
        with get_autocast_context(device):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
    
    accuracy, correct, total = compute_token_accuracy(
        outputs.logits,
        batch["labels"]
    )
    
    return {
        "accuracy": accuracy,
        "correct_tokens": correct,
        "total_tokens": total,
        "loss": outputs.loss. item() if outputs.loss is not None else None
    }


def compute_dataset_token_accuracy(
    model,
    dataloader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """对整个数据集计算 token 准确率。"""
    model.eval()
    
    total_correct = 0
    total_tokens = 0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            if device is not None:
                batch = {k: v.to(device) if isinstance(v, torch. Tensor) else v for k, v in batch.items()}
            
            # ✅ 使用 autocast
            with get_autocast_context(device):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
            
            _, correct, total = compute_token_accuracy(
                outputs.logits,
                batch["labels"]
            )
            
            total_correct += correct
            total_tokens += total
            if outputs.loss is not None:
                total_loss += outputs. loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                current_acc = total_correct / total_tokens if total_tokens > 0 else 0
                logger.info(f"Batch {batch_idx}: running accuracy = {current_acc:.4f}")
    
    mean_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        "mean_token_accuracy": mean_accuracy,
        "total_correct": total_correct,
        "total_tokens": total_tokens,
        "mean_loss": mean_loss,
        "num_batches": num_batches
    }


def validate_accuracy_implementation(
    model,
    sample_batch: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """验证我们的实现与 SFTTrainer 一致。"""
    model.eval()
    
    if device is not None:
        sample_batch = {k: v. to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch. items()}
    
    # ✅ 使用 autocast 处理混合精度
    with torch. no_grad():
        with get_autocast_context(device):
            outputs = model(
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_mask"],
                labels=sample_batch["labels"]
            )
    
    logits = outputs.logits
    labels = sample_batch["labels"]
    
    logger.info("=" * 50)
    logger. info("Validation of Token Accuracy Implementation")
    logger. info("=" * 50)
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Labels shape: {labels.shape}")

    shift_logits = logits[..., :-1, :]. contiguous()
    shift_labels = labels[..., 1:].contiguous()

    logger.info(f"Shift logits shape: {shift_logits.shape}")
    logger.info(f"Shift labels shape: {shift_labels.shape}")
    
    predictions = shift_logits.argmax(dim=-1)
    logger.info(f"Predictions shape: {predictions.shape}")
    
    mask = shift_labels != -100
    logger. info(f"Mask shape: {mask.shape}")
    logger.info(f"Total valid tokens (mask. sum()): {mask.sum(). item()}")
    
    num_ignored = (labels == -100).sum().item()
    num_valid = (labels != -100).sum().item()
    logger.info(f"Labels: {num_ignored} ignored (-100), {num_valid} valid")
    
    correct_predictions = (predictions == shift_labels) & mask
    correct_tokens = correct_predictions.sum().item()
    total_tokens = mask.sum().item()
    
    logger.info(f"Correct tokens: {correct_tokens}")
    logger. info(f"Total tokens: {total_tokens}")
    if total_tokens > 0:
        logger.info(f"Accuracy: {correct_tokens / total_tokens:.6f}")
    else:
        logger. info("Accuracy: N/A (no valid tokens)")
    
    logger.info("")
    logger.info("Per-sample accuracy:")
    batch_size = labels.shape[0]
    for i in range(min(batch_size, 5)):
        sample_mask = mask[i]
        sample_correct = correct_predictions[i]
        sample_total = sample_mask.sum().item()
        sample_correct_count = (sample_correct & sample_mask).sum(). item()
        sample_acc = sample_correct_count / sample_total if sample_total > 0 else 0
        logger.info(f"  Sample {i}: {sample_correct_count}/{sample_total} = {sample_acc:.4f}")
    
    logger.info("=" * 50)
    
    return {
        "accuracy": correct_tokens / total_tokens if total_tokens > 0 else 0,
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "loss": outputs.loss. item() if outputs. loss is not None else None
    }


def evaluate_token_accuracy(
    model,
    dataset,
    tokenizer,
    batch_size: int = 4,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """评估数据集上的 token 准确率"""
    model.eval()
    
    if device is None:
        device = next(model.parameters()). device
    
    collate_fn = partial(simple_collate_fn_for_prompt_completion, pad_token_id=tokenizer.pad_token_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    total_correct = 0
    total_tokens = 0
    total_loss = 0.0
    num_batches = 0
    
    logger.info(f"Starting evaluation on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # ✅ 使用 autocast
            with get_autocast_context(device):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
            
            _, correct, total = compute_token_accuracy(outputs.logits, batch["labels"])
            
            total_correct += correct
            total_tokens += total
            if outputs.loss is not None:
                total_loss += outputs. loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                running_acc = total_correct / total_tokens if total_tokens > 0 else 0
                logger. info(f"Batch {batch_idx}/{len(dataloader)}: running_accuracy={running_acc:.4f}")
    
    mean_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    logger.info(f"Evaluation completed:")
    logger.info(f"  Mean Token Accuracy: {mean_accuracy:.6f}")
    logger. info(f"  Mean Loss: {mean_loss:.6f}")
    logger.info(f"  Total Correct: {total_correct}")
    logger.info(f"  Total Tokens: {total_tokens}")
    logger.info(f"  Num Batches: {num_batches}")
    
    return {
        "mean_token_accuracy": mean_accuracy,
        "mean_loss": mean_loss,
        "total_correct": total_correct,
        "total_tokens": total_tokens,
        "num_batches": num_batches,
    }


def validate_single_batch(
    model,
    dataset,
    tokenizer,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """验证单个 batch，打印详细信息用于调试"""
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    collate_fn = partial(simple_collate_fn_for_prompt_completion, pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    logger.info("=" * 60)
    logger. info("Single Batch Validation")
    logger.info("=" * 60)
    logger.info(f"input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"attention_mask shape: {batch['attention_mask'].shape}")
    logger.info(f"labels shape: {batch['labels'].shape}")
    
    labels = batch["labels"]
    num_ignored = (labels == -100).sum().item()
    num_valid = (labels != -100).sum(). item()
    logger.info(f"Labels: {num_ignored} ignored (-100), {num_valid} valid tokens")
    
    # ✅ 使用 autocast
    with torch.no_grad():
        with get_autocast_context(device):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
    
    logger.info(f"Loss: {outputs.loss.item():.4f}")
    logger.info(f"Logits shape: {outputs.logits.shape}")
    
    accuracy, correct, total = compute_token_accuracy(outputs.logits, batch["labels"])
    logger.info(f"Token Accuracy: {correct}/{total} = {accuracy:.4f}")
    logger.info("=" * 60)
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "loss": outputs.loss.item()
    }