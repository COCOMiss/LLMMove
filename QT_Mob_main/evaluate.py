import math
import torch.nn.functional as F

import re
import json
import logging
import torch


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


def formatting_traj_func(text):
    """
    解析模型生成的轨迹字符串。
    格式示例: prediction:[{"id": "1", "h3_index": "...", "stay_duration": "..."}, ...]
    返回: (h3_list, duration_list)
    """
    # 1. 清理前缀
    if "prediction:" in text:
        text = text.split("prediction:")[1]
    
    # 2. 清理后缀和空白
    text = text.strip()
    if "<|im_end|>" in text:
        text = text.replace("<|im_end|>", "")
    
    # 3. 尝试 JSON 解析
    try:
        # 有时模型可能生成未闭合的列表，尝试简单的修复（针对截断情况）
        if text.startswith("[") and not text.endswith("]"):
            if text.endswith("}"):
                text += "]"
            elif text.endswith('"'): # 假设断在属性值
                text += '"}]' 
            else:
                text += "}]" # 极其简单的兜底
                
        data = json.loads(text)
        
        h3_list = []
        dur_list = []
        start_time_list = []
        
        if isinstance(data, list):
            for item in data:
                # 提取 H3 和 Duration，容错处理
                h3_list.append(str(item.get("h3_index", "")).strip())
                dur_list.append(str(item.get("stay_duration", "")).strip())
                start_time_list.append(str(item.get("start_time", "")).strip())
                
        return h3_list, dur_list
        
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        # 解析失败时返回空列表
        # logger.debug(f"JSON Parse Error: {e} | Text snippet: {text[:50]}...")
        return [], []



def compute_lcs(s1, s2):
    """计算两个序列的最长公共子序列长度"""
    m, n = len(s1), len(s2)
    if m == 0 or n == 0: return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

# def get_daily_traj_results(output_text, targets, scores, metrics, all_items=None):
#     """
#     评估轨迹生成任务的结果。
#     注意：此函数主要评估 Beam Search 的 Top-1 结果（最可能的轨迹）。
    
#     Args:
#         output_text (List[str]): 模型生成的文本列表，长度为 Batch_Size * Num_Return_Sequences
#         targets (List[str]): 真实标签列表，长度为 Batch_Size
#         scores (List): 模型分数 (本函数暂主要使用 output_text 的顺序作为排名)
#         metrics (List[str]): 需要计算的指标列表
#         all_items: (Optional)
#     """
    
#     # 假设 test.py 中设置了 num_return_sequences (例如 5)
#     # 我们需要知道 K 是多少来对齐 targets
#     total_predictions = len(output_text)
#     batch_size = len(targets)
#     k = total_predictions // batch_size
    
#     # 存储指标统计
#     res_stats = {
#         "H3_Jaccard": 0.0,  # 集合重叠度 (忽略顺序)
#         "H3_LCS": 0.0,      # 最长公共子序列 (考虑顺序)
#         "Dur_Acc": 0.0      # Duration 序列完全匹配
#     }
    
#     valid_count = 0
    
#     for i in range(batch_size):
#         # 获取当前样本的 Target
#         t_h3_seq, t_dur_seq = formatting_traj_func(targets[i])
        
#         # 获取当前样本的 Predictions (取 Top-1，即 k 个中的第 0 个)
#         # 这里的 output_text 已经是按 beam score 排序过的
#         pred_idx_start = i * k
#         top1_text = output_text[pred_idx_start]
        
#         p_h3_seq, p_dur_seq = formatting_traj_func(top1_text)
        
      
            
#         # --- 2. H3 Jaccard Similarity (Set Overlap) ---
#         s1 = set(t_h3_seq)
#         s2 = set(p_h3_seq)
#         if len(s1) == 0 and len(s2) == 0:
#             res_stats["H3_Jaccard"] += 1.0
#         elif len(s1) > 0 or len(s2) > 0:
#             intersection = len(s1 & s2)
#             union = len(s1 | s2)
#             res_stats["H3_Jaccard"] += intersection / union
            
#         # --- 3. H3 LCS (Longest Common Subsequence) ---
#         # 归一化：LCS / max(len(pred), len(target))
#         lcs_len = compute_lcs(p_h3_seq, t_h3_seq)
#         denom = max(len(p_h3_seq), len(t_h3_seq))
#         if denom > 0:
#             res_stats["H3_LCS"] += lcs_len / denom
#         elif len(t_h3_seq) == 0 and len(p_h3_seq) == 0:
#              res_stats["H3_LCS"] += 1.0

#         # --- 4. Duration 序列完全匹配 ---
#         if p_dur_seq == t_dur_seq and len(t_dur_seq) > 0:
#             res_stats["Dur_Acc"] += 1.0

#         valid_count += 1

#     # 计算平均值
#     metrics_results = {}
#     if valid_count > 0:
#         for k, v in res_stats.items():
#             metrics_results[k] = v / valid_count
    
#     return metrics_results






# 假设 compute_lcs 和 formatting_traj_func 已经在外部定义，或者在同一个文件中
# 如果没有，请保留之前定义的这两个辅助函数

def get_daily_traj_results(output_text, targets, scores, metrics, all_items=None):
    """
    评估轨迹生成任务的结果，计算 Top-1 和 Top-5 指标。
    
    Top-K 定义: 在前 K 个候选项中，指标分数的最大值 (Best-of-K)。
    例如 Top-5 Jaccard 意味着前 5 个预测中与真实值重叠度最高的那个分数。
    """
    
    # 1. 确定 Beam Search 的束宽 (num_return_sequences)
    total_predictions = len(output_text)
    batch_size = len(targets)
    if batch_size == 0:
        return {}
        
    k = total_predictions // batch_size
    # 确保我们有足够的数据计算 Top-5，如果 k < 5，则 Top-5 等同于 Top-k
    top_k_limit = 5 
    
    # 2. 初始化统计字典
    # 使用 @1 和 @5 后缀区分
    res_stats = {
        "H3_Jaccard@1": 0.0, "H3_Jaccard@5": 0.0,
        "H3_LCS@1": 0.0,     "H3_LCS@5": 0.0,
        "Dur_Acc@1": 0.0,    "Dur_Acc@5": 0.0
    }
    
    valid_count = 0
    
    for i in range(batch_size):
        # 2.1 获取并解析当前样本的 Target
        t_h3_seq, t_dur_seq = formatting_traj_func(targets[i])
        
        # 目标如果是空的，视情况处理，这里假设总是有效的或者参与计算
        # 如果需要过滤无效数据，可以在这里加判断
        
        # 2.2 准备存储当前样本 K 个预测的单项分数
        # 只要计算前5个即可，多余的不需要计算以节省时间
        current_k_scores = {
            "jaccard": [],
            "lcs": [],
            "dur_acc": []
        }
        
        # 遍历该样本的前 K 个预测 (或者最多前5个)
        num_to_check = min(k, top_k_limit)
        start_idx = i * k
        
        for j in range(num_to_check):
            pred_text = output_text[start_idx + j]
            p_h3_seq, p_dur_seq = formatting_traj_func(pred_text)
            
            # --- Metric: H3 Jaccard ---
            s1 = set(t_h3_seq)
            s2 = set(p_h3_seq)
            jac = 0.0
            if len(s1) == 0 and len(s2) == 0:
                jac = 1.0
            elif len(s1) > 0 or len(s2) > 0:
                intersection = len(s1 & s2)
                union = len(s1 | s2)
                jac = intersection / union
            current_k_scores["jaccard"].append(jac)
            
            # --- Metric: H3 LCS ---
            lcs_val = 0.0
            lcs_len = compute_lcs(p_h3_seq, t_h3_seq)
            denom = max(len(p_h3_seq), len(t_h3_seq))
            if denom > 0:
                lcs_val = lcs_len / denom
            elif len(t_h3_seq) == 0 and len(p_h3_seq) == 0:
                lcs_val = 1.0
            current_k_scores["lcs"].append(lcs_val)
            
            # --- Metric: Duration Accuracy ---
            # 这是一个 Boolean 指标 (1.0 or 0.0)
            is_match = 1.0 if (p_dur_seq == t_dur_seq and len(t_dur_seq) > 0) else 0.0
            current_k_scores["dur_acc"].append(is_match)

        # 2.3 汇总当前样本的 @1 和 @5 分数
        
        # Top-1: 直接取列表的第 0 个
        res_stats["H3_Jaccard@1"] += current_k_scores["jaccard"][0]
        res_stats["H3_LCS@1"]     += current_k_scores["lcs"][0]
        res_stats["Dur_Acc@1"]    += current_k_scores["dur_acc"][0]
        
        # Top-5: 取列表中的最大值 (Best Match)
        # 注意：如果 k < 5，这里取 max 会自动处理实际长度
        res_stats["H3_Jaccard@5"] += max(current_k_scores["jaccard"])
        res_stats["H3_LCS@5"]     += max(current_k_scores["lcs"])
        res_stats["Dur_Acc@5"]    += max(current_k_scores["dur_acc"]) # 只要有一个全对，就是1.0

        valid_count += 1

    # 3. 计算平均值
    metrics_results = {}
    if valid_count > 0:
        for key in res_stats:
            metrics_results[key] = res_stats[key] / valid_count
    
    return metrics_results


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