import os
import json
import argparse

import torch
import pandas as pd
from unsloth import FastLanguageModel

from logger_utils import get_logger
from h3_prompt_mobility import *
from utils import set_seed

logger = get_logger(__name__)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../LLLMOVE/QT_Mob_main
PROJECT_ROOT = os.path.dirname(THIS_DIR)              # .../LLLMOVE

DEFAULT_LORA_PATH = os.path.join(
    PROJECT_ROOT, "my_qwen3_mobility_unsloth_lora"
)

# 和 train 一致的任务 prompt
PROMPT_TEMPLATE = (
    "Here is the mobility profile for user {user}."
    "The profile details their home and work locations (if known), a list of frequently visited locations with typical visit times, "
    "and their preferences for different POI categories based on visit history: {profile}"
    "Today is a {date}."
    "Task: Predict the daily trajectory of the user for this date."
    "Predict the sequence of visits, including the start time, the location (H3 index), and the stay duration (in minutes)."
    "Return ONLY a JSON list with the following format and no extra text:"
    'Example: [{{ "id": "1", "start_time": "HH:MM AM/PM", "h3_index": "...", "stay_duration": "... min" }}, ...]'
)

def build_instruction(row):
    sysprompt = system_prompt_new + traj_task_prompt
    base_prompt = sysprompt + PROMPT_TEMPLATE.format(
        user=row["user"],
        profile=row["profile"],
        date=row["date"],
    )
    base_prompt = base_prompt.rstrip(" \n\r\t")
    return base_prompt


def load_eval_df(args):
    data_path = args.eval_data_path
    logger.info(f"Loading eval data from: {data_path}")
    df = pd.read_feather(data_path)
    if args.max_eval_samples > 0 and len(df) > args.max_eval_samples:
        df = df.sample(
            n=args.max_eval_samples,
            random_state=args.seed,
        ).reset_index(drop=True)
        logger.info(f"Sampled {len(df)} examples for evaluation.")
    else:
        df = df.reset_index(drop=True)
        logger.info(f"Using all {len(df)} examples for evaluation.")
    return df


def load_lora_model(args):
    max_seq_len = args.max_seq_length
    logger.info(f"Loading LoRA model from: {args.lora_path}")
    logger.info(f"max_seq_length = {max_seq_len}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.lora_path,   # LoRA 目录
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 推理模式
    model = FastLanguageModel.for_inference(model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info(f"Model loaded on device: {device}")
    return model, tokenizer, device


# ---------- metrics: hit@k ----------

def parse_traj(traj_str):
    """
    把 ground_truth / model_output 字符串解析成 H3 index 序列: List[str]

    兼容：
      - 直接 JSON list of dicts [{"h3_index": "...", ...}, ...]
      - 前面带 'prediction:' 前缀
      - 前后有别的说明文字，只取第一个 [ ... ] 段
    """
    if traj_str is None:
        return []
    if isinstance(traj_str, (list, dict)):
        data = traj_str
    else:
        if not isinstance(traj_str, str):
            traj_str = str(traj_str)
        s = traj_str.strip()
        if not s:
            return []
        if s.startswith("prediction:"):
            s = s[len("prediction:"):].lstrip()
        # 只截取第一个 JSON list
        start = s.find("[")
        end = s.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        s_json = s[start:end+1]
        try:
            data = json.loads(s_json)
        except Exception as e:
            logger.warning(f"Failed to json.loads trajectory: {e}. snippet={s_json[:200]}")
            return []

    seq = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "h3_index" in item:
                seq.append(str(item["h3_index"]))
            else:
                # 如果不是 dict，就直接当成字符串
                seq.append(str(item))
    elif isinstance(data, dict):
        if "h3_index" in data:
            seq.append(str(data["h3_index"]))
    return seq


def hits_at_k(gt_seq, pred_seq, ks=(1, 5, 10)):
    """
    标准推荐任务的 hit@k:
    只要 ground truth 集合里有任意一个出现在 pred 的 Top-k 里，就记为 hit=1
    """
    # 去重后的预测序列
    seen = set()
    unique_preds = []
    for p in pred_seq:
        if p not in seen:
            seen.add(p)
            unique_preds.append(p)

    gt_set = set(gt_seq)
    res = {}
    for k in ks:
        topk = unique_preds[:k]
        res[k] = int(len(gt_set.intersection(topk)) > 0)
    return res


def compute_hit_metrics(results):
    """
    results: List[dict]，每个 dict 里有 ground_truth 和 model_output
    输出 hit@1, hit@5, hit@10 的平均值
    """
    ks = (1, 5, 10)
    hit_sums = {k: 0 for k in ks}
    valid_cnt = 0
    skipped = 0

    for r in results:
        gt_seq = parse_traj(r.get("ground_truth", ""))
        pred_seq = parse_traj(r.get("model_output", ""))
        if not gt_seq or not pred_seq:
            skipped += 1
            continue
        hits = hits_at_k(gt_seq, pred_seq, ks)
        for k, v in hits.items():
            hit_sums[k] += v
        valid_cnt += 1

    metrics = {}
    if valid_cnt == 0:
        logger.warning("No valid samples to compute hit@k (all empty or parse failed).")
        for k in ks:
            metrics[f"hit@{k}"] = 0.0
    else:
        for k in ks:
            metrics[f"hit@{k}"] = hit_sums[k] / valid_cnt

    logger.info(f"Computed hit@k on {valid_cnt} valid samples, skipped={skipped}")
    for k in ks:
        logger.info(f"hit@{k} = {metrics[f'hit@{k}']:.4f}")

    return metrics


# ---------- main inference + eval ----------

def run_inference(args):
    set_seed(args.seed)

    df = load_eval_df(args)
    model, tokenizer, device = load_lora_model(args)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_predictions.jsonl")
    metrics_path = os.path.join(args.output_dir, "metrics_hitk.json")

    results = []

    logger.info("Starting inference...")
    for idx, row in df.iterrows():
        instruction = build_instruction(row)
        # 训练时 text = "INSTRUCTION:\n{prompt}\n\nRESPONSE:\n{completion}"
        prompt_text = f"INSTRUCTION:\n{instruction}\n\nRESPONSE:\n"

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        full_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # 把 RESPONSE: 后面的当做模型输出
        if "RESPONSE:\n" in full_text:
            pred_text = full_text.split("RESPONSE:\n", 1)[1].strip()
        else:
            pred_text = full_text[len(prompt_text):].strip()

        result = {
            "user": str(row.get("user", "")),
            "date": str(row.get("date", "")),
            "profile": str(row.get("profile", "")),
            "ground_truth": str(row.get("prediction", "")),  # GT
            "model_output": pred_text,                       # 模型生成
        }
        results.append(result)

        if (idx + 1) % args.log_interval == 0:
            logger.info(f"Inference progress: {idx + 1}/{len(df)}")

    # 写预测结果 jsonl
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved predictions to: {out_path}")

    # 计算 hit@1, hit@5, hit@10
    metrics = compute_hit_metrics(results)

    # 存指标
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")
    logger.info("Done.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="QT-Mob Unsloth LoRA inference (test) with hit@k metrics"
    )

    # 基本配置
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=DEFAULT_LORA_PATH,
        help="Path to Unsloth LoRA model (trainer.save_model 的目录)",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="QT_Mob_main/dataset/test/zdc_h3_8/daily_traj_dataset.feather",
        help="Feather file for evaluation (可以改成 test 集)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save prediction jsonl & metrics",
    )

    # 生成参数
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max sequence length for model/tokenizer",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )

    # 其它
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=100,
        help="Max number of eval samples (0 = use all)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Logging interval during inference",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)
