import math
import torch.nn.functional as F

import re
import json
import logging


logger = logging.getLogger(__name__)

def mean_squared_error(predictions, targets):
    """计算均方误差 (MSE)"""
    return sum([(pred - target) ** 2 for pred, target in zip(predictions, targets)]) / len(predictions)




def get_topk_results(predictions, scores, targets, k, metrics,all_items=None):
    # predictions: List[str] size = B(batch size)*k
    # scores: List[float] or List[tensor] or tuple of step-wise logits size = B*k or generation steps
    # targets: List[str] size = B
    # k: int
    # all_items: List[str] or None
    import torch
    import torch.nn.functional as F
    
    results = []
    B = len(targets)
    
    # Convert scores to list of scalars
    # Handle different score formats: tensor, tuple of tensors (generation steps), list of tensors, etc.
    processed_scores = []
    
    # Check if scores is a tuple/list of generation step logits
    if isinstance(scores, (list, tuple)) and len(scores) > 0:
        first_score = scores[0]
        if isinstance(first_score, torch.Tensor) and first_score.dim() == 2:
            # This is likely generation step logits: tuple of [batch*num_seq, vocab_size] tensors
            # We need to compute sequence-level scores from step-wise logits
            num_sequences = first_score.shape[0]
            if num_sequences == len(predictions):
                # Compute sequence scores: average of max logits at each step
                sequence_scores = []
                for seq_idx in range(num_sequences):
                    seq_logits = []
                    for step_logits in scores:
                        if step_logits.dim() == 2:
                            seq_step_logits = step_logits[seq_idx]
                        else:
                            seq_step_logits = step_logits
                        # Get max logit for this sequence at this step (or mean of non-inf logits)
                        non_inf_mask = seq_step_logits != float('-inf')
                        if non_inf_mask.any():
                            seq_logits.append(seq_step_logits[non_inf_mask].max().item())
                        else:
                            seq_logits.append(float('-inf'))
                    # Use mean of logits as sequence score (or sum, or other aggregation)
                    if seq_logits:
                        # Filter out -inf values
                        valid_logits = [lg for lg in seq_logits if lg != float('-inf')]
                        if valid_logits:
                            sequence_scores.append(sum(valid_logits) / len(valid_logits))
                        else:
                            sequence_scores.append(float('-inf'))
                    else:
                        sequence_scores.append(float('-inf'))
                processed_scores = sequence_scores
            else:
                # Fallback: try to convert each score to scalar
                for score in scores:
                    if isinstance(score, torch.Tensor):
                        if score.numel() == 1:
                            processed_scores.append(score.item())
                        else:
                            processed_scores.append(score.mean().item())
                    elif hasattr(score, "item"):
                        processed_scores.append(score.item())
                    else:
                        processed_scores.append(float(score))
        else:
            # Regular list/tuple of scalars or single-element tensors
            for score in scores:
                if isinstance(score, torch.Tensor):
                    if score.numel() == 1:
                        processed_scores.append(score.item())
                    else:
                        processed_scores.append(score.mean().item())
                elif hasattr(score, "item"):
                    processed_scores.append(score.item())
                else:
                    processed_scores.append(float(score))
    elif isinstance(scores, torch.Tensor):
        # If scores is a single tensor, convert to list
        if scores.numel() == len(predictions):
            processed_scores = scores.tolist()
        else:
            # Flatten and take first len(predictions) elements
            processed_scores = scores.flatten()[:len(predictions)].tolist()
    else:
        # Try to convert to list
        if hasattr(scores, "tolist"):
            processed_scores = scores.tolist()
        else:
            processed_scores = list(scores)
    
    # Ensure we have the right number of scores
    if len(processed_scores) != len(predictions):
        print(f"Warning: Number of scores ({len(processed_scores)}) doesn't match predictions ({len(predictions)}). Using dummy scores.")
        processed_scores = [1.0] * len(predictions)
    
    for b in range(B):  # For each example in the batch
        batch_seqs = predictions[b * k: (b + 1) * k]  # k predicted items
        batch_scores = processed_scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)  # Sort by scores
        target_item = targets[b]
        for metric in metrics:
            if metric.lower().startswith("hit"):
                one_results = []  # List to store Top K results for the current example
                for sorted_pred in sorted_pairs:
                    sorted_pred_tokens = sorted_pred[0].split("><")
                    target_item_tokens= target_item.split("><")
                    hit_count = 0
                    for pred_token , target_token in zip(sorted_pred_tokens, target_item_tokens):
                        if pred_token == target_token:
                            hit_count+=1
                    one_results.append(hit_count/len(sorted_pred_tokens))
                 

            elif metric.lower().startswith("mse"):
                one_results = 0.0
                for sorted_pred in sorted_pairs:
                    one_results+=(float(sorted_pred[0][0])-float(target_item[0]))** 2
            
        results.append(one_results)  # Store results for each example

    return results







def get_top1_results(predictions, targets, all_items=None):
    # predictions: List[str] size = B(batch size)*k
    # targets: List[str] size = B
    # k: int
    # all_items: List[str] or None
    results = []
    B = len(targets)
    predictions = [_.split("user will visit POI index ")[-1].split(".")[0] for _ in predictions] # 取出最后一个Response:后的字符串，即预测的item
    predictions = [_.strip().replace(" ","") for _ in predictions] # 去掉空格
    # print(predictions[:1])
    if all_items is not None:
        predictions = [seq if seq in all_items else None for seq in predictions] # 如果预测的item不在all_items中，将其设置为None

    for b in range(B): # 对于一个batch里的每个样本
        batch_seqs = predictions[b: b + 1] # k个预测的item
        target_item = targets[b]
        one_results = [] # 长度为k的分数
        for pred in batch_seqs:
            if pred == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results





def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        elif m.lower().startswith("map"):
            k = int(m.split("@")[1])
            res[m] = map_k(topk_results, k)
        # elif m.lower().startswith("mse"):  # Add MSE for duration
        #     mean_squared_error(topk_results)
            
        else:
            raise NotImplementedError

    return res






def ndcg_k(topk_results, k):
    # 归一化折损累计增益（NDCG）
    # Note: topk_results contains hit scores (0.0 to 1.0), convert to binary for NDCG
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        # Convert hit scores to binary: > 0 means hit (1), otherwise 0
        binary_res = [1 if score > 0 else 0 for score in res]
        ndcg += next((1 / math.log(i + 2, 2) for i in range(len(binary_res)) if binary_res[i] == 1), 0.0)
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        # res is a list of hit scores (0.0 to 1.0)
        # Check if there's any hit (any value > 0)
        if any(score > 0 for score in res):
            hit += 1.0
    return hit


def map_k(topk_results, k):
    # Mean Average Precision (MAP)
    # Note: topk_results contains hit scores (0.0 to 1.0), convert to binary for MAP
    map_score = 0.0
    for row in topk_results:
        res = row[:k]
        # Convert hit scores to binary: > 0 means hit (1), otherwise 0
        binary_res = [1 if score > 0 else 0 for score in res]
        # Calculate average precision for this query
        if sum(binary_res) > 0:
            precision_sum = 0.0
            hit_count = 0
            for i, rel in enumerate(binary_res):
                if rel == 1:
                    hit_count += 1
                    precision_sum += hit_count / (i + 1)
            map_score += precision_sum / sum(binary_res)
    return map_score / len(topk_results) if len(topk_results) > 0 else 0.0


def formatting_func(text):

    pred_dict = {}
    try:
        # Extract the string after 'prediction:' (should be a JSON-like string)
        pred_part = text.split('prediction:',1)[1].strip()
        # Sometimes there may be accidental trailing chars, so find up to the first matching closing brace if possible
        m = re.search(r'(\{.*\})', pred_part)
        if m:
            pred_json_str = m.group(1)
        else:
            pred_json_str = pred_part
        # Replace single quotes with double quotes for json.loads, if necessary
        pred_json_str = pred_json_str.replace("'", '"')
        # Remove tokens like \xa0 if present
        pred_json_str = pred_json_str.strip()
        pred_dict = json.loads(pred_json_str)
        h3_index = pred_dict.get("h3_index", "")
        duration = pred_dict.get("stay_duration", "")
    except Exception as e:
        # fallback to manual extraction if json fails or format is messy
        h3_index = ""
        duration = ""
        m = re.search(r'"h3_index":\s*("(?:<[^>]+>)+")', text)
        if m:
            h3_index = m.group(1).strip('"')
        m = re.search(r'"stay_duration":\s*"([^"]+)"', text)
        if m:
            duration = m.group(1)
    return h3_index, duration


def formatting_labels(text):
    try:
    
        m = re.search(r'(\{.*\})', text)
        if m:
            pred_json_str = m.group(1)
        else:
            pred_json_str = text
        # Replace single quotes with double quotes for json.loads, if necessary
        pred_json_str = pred_json_str.replace("'", '"')
        # Remove tokens like \xa0 if present
        pred_json_str = pred_json_str.strip()
        pred_dict = json.loads(pred_json_str)
        h3_index = pred_dict.get("h3_index", "")
        duration = pred_dict.get("stay_duration", "")
    except Exception as e:
        # fallback to manual extraction if json fails or format is messy
        h3_index = ""
        duration = ""
        m = re.search(r'"h3_index":\s*("(?:<[^>]+>)+")', text)
        if m:
            h3_index = m.group(1).strip('"')
        m = re.search(r'"stay_duration":\s*"([^"]+)"', text)
        if m:
            duration = m.group(1)
    return h3_index, duration


def get_daily_traj_results(predictions, targets):
    # predictions: List[str] size = B(batch size)*k
    # targets: List[str] size = B
    # k: int
    # all_items: List[str] or None
    results = []
    B = len(targets)
    predictions = [_.split("user will visit POI index ")[-1].split(".")[0] for _ in predictions] # 取出最后一个Response:后的字符串，即预测的item
    predictions = [_.strip().replace(" ","") for _ in predictions] # 去掉空格
    for b in range(B): # 对于一个batch里的每个样本
        batch_seqs = predictions[b: b + 1] # k个预测的item
        target_item = targets[b]
        one_results = [] # 长度为k的分数
        for pred in batch_seqs:
            if pred == target_item:
                one_results.append(1)
            else:
                one_results.append(0)
        results.append(one_results)
    return results



def get_seq_results(output_text, targets,scores,metrics,all_items=None):
    # predictions: List[str] size = B(batch size)*k
    # targets: List[str] size = B
    # k: int
    # all_items: List[str] or None
    
    
    
    # 分离预测出的 H3 index 和 duration
    h3_predictions = []
    duration_predictions=[]
    
    for text in output_text:
        try:
            h3_index, duration = formatting_func(text)
            h3_predictions.append(h3_index)
            duration_predictions.append(duration)
        except (IndexError, ValueError) as e:
            # Handle parsing errors gracefully
            logger.warning(f"Error parsing text: {text[:100]}... Error: {e}")
            h3_predictions.append([])
            duration_predictions.append([])
    
    target_h3=[]
    target_duration=[] 
    for target in targets:
        t_h3_index, t_duration = formatting_labels(target)
        target_h3.append(t_h3_index)
        target_duration.append(t_duration)
        
    h3_topk_res = get_topk_results(h3_predictions,scores,target_h3,5,metrics=metrics,
                                all_items=all_items )

    h3_metrics_res = get_metrics_results(h3_topk_res, metrics)
    return h3_metrics_res
    
    
    
    # results = []
    # B = len(targets)
    # predictions = [_.split("user will visit POI index ")[-1].split(".")[0] for _ in predictions] # 取出最后一个Response:后的字符串，即预测的item
    # predictions = [_.strip().replace(" ","") for _ in predictions] # 去掉空格