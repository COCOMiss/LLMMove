
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from unsloth import FastLanguageModel
import json
import argparse
import torch
import pandas as pd
import re
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from logger_utils import get_logger
from h3_prompt_mobility import *
from utils import set_seed
import torch
logger = get_logger(__name__)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../LLLMOVE/QT_Mob_main
PROJECT_ROOT = os.path.dirname(THIS_DIR)              # .../LLLMOVE

DEFAULT_LORA_PATH = os.path.join(
    PROJECT_ROOT, "my_qwen3_mobility_unsloth_lora"
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 和 train 一致的任务 prompt
PROMPT_TEMPLATE = (
    "Here is the mobility profile for user {user}:\n"
    "{profile}\n\n"
    "The profile above summarizes their home and work locations (if known), "
    "frequently visited locations with typical visit times, and preferences for POI categories.\n"
    "The specific date to predict for is: {date}.\n\n"
    "You MUST construct a plausible daily trajectory for this user in Tokyo that is consistent "
    "with the profile and the given date. The visits in the JSON list must:\n"
    "- Be ordered in chronological order by \"start_time\" from earliest to latest.\n"
    "- Use \"id\" as a string index starting from \"1\" and increasing by 1 for each subsequent visit.\n"
    "- Use \"start_time\" in the format \"HH:MM AM/PM\" (for example, \"07:30 PM\").\n"
    "- Use \"h3_index\" as a single valid H3 index string representing the location of that visit.\n"
    "- Use \"stay_duration\" as one of 30, 60, 90, ..., 600 (step 30), formatted exactly as \"<N> min\".\n\n"
    "Return ONLY a valid JSON list (array) and NOTHING else. Do NOT include any natural language "
    "explanation, comments, or extra keys. The output should look like this:\n"
    "[\n"
    "  {{\"id\": \"1\", \"start_time\": \"10:25 AM\", \"h3_index\": \"<H3_INDEX_1>\", \"stay_duration\": \"210 min\"}},\n"
    "  {{\"id\": \"2\", \"start_time\": \"04:00 PM\", \"h3_index\": \"<H3_INDEX_2>\", \"stay_duration\": \"330 min\"}}\n"
    "]\n"
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
    logger.info(f"Loading LoRA adapter from: {args.lora_path}")
    logger.info(f"max_seq_length = {max_seq_len}")

    # 1️⃣ 从 LoRA 目录中读取 adapter_config，解析 base_model_name_or_path
    # 优先用 LoraConfig.from_pretrained，如果取不到再手动读 JSON
    lora_config = LoraConfig.from_pretrained(args.lora_path)
    base_model_name_or_path = getattr(lora_config, "base_model_name_or_path", None)

    if base_model_name_or_path is None:
        adapter_cfg_path = os.path.join(args.lora_path, "adapter_config.json")
        logger.info(f"base_model_name_or_path not in LoraConfig, fallback to {adapter_cfg_path}")
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
        base_model_name_or_path = cfg_json.get("base_model_name_or_path")

    if base_model_name_or_path is None:
        raise ValueError(
            f"Cannot resolve base_model_name_or_path from LoRA config under {args.lora_path}. "
            f"Please check adapter_config.json."
        )

    logger.info(f"Resolved base model from LoRA config: {base_model_name_or_path}")

    # 2️⃣ 用 base_model_name_or_path 加载基座 Qwen（Unsloth 4bit）
    #    注意：这里的 model_name 不再用 lora_path，避免 Unsloth 自动套 LoRA。
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name_or_path,
        max_seq_length=max_seq_len,
        dtype=None,                   # 或 "float16" 看你环境
        load_in_4bit=True,
        trust_remote_code=True,
        local_files_only=True,       # 如果你有本地快照会走缓存；完全离线再改 True
        use_gradient_checkpointing=False,  # 推理阶段关掉，省显存
    )

    # 3️⃣ 如果训练时加过自定义 token，这里也要做同样的事
    try:
        from utils import get_new_tokens
        new_tokens = get_new_tokens(args)
    except ImportError:
        logger.warning("utils.get_new_tokens not found, skip adding custom tokens.")
        new_tokens = []
    except Exception as e:
        logger.warning(f"Failed to get new tokens: {e}")
        new_tokens = []

    if new_tokens:
        num_added = tokenizer.add_tokens(list(new_tokens))
        if num_added > 0:
            logger.info(f"Added {num_added} new tokens to tokenizer.")
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized token embeddings to {len(tokenizer)}.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4️⃣ 把 LoRA 模块挂到基座上（用刚才的 lora_config）
    model = get_peft_model(model, lora_config)
    logger.info("LoRA modules added to base model.")

    # 5️⃣ 手动加载 LoRA 权重：只保留 LoRA 相关 key，避免覆盖 embedding / lm_head
    adapter_path = os.path.join(args.lora_path, "adapter_model.safetensors")
    logger.info(f"Loading LoRA weights from: {adapter_path}")
    full_state_dict = load_file(adapter_path)  # OrderedDict

    # 只保留 LoRA 参数（名字里带 lora_ / lora_A / lora_B）
    state_dict = {}
    for k, v in full_state_dict.items():
        if "lora_" in k or "lora_A" in k or "lora_B" in k:
            state_dict[k] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        f"Loaded LoRA weights (LoRA-only). "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    # 6️⃣ 推理模式
    model = FastLanguageModel.for_inference(model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model + LoRA loaded on device: {device}")

    return model, tokenizer, device
# ---------- metrics: hit@k ----------


def extract_first_json_list(text: str):
    """
    在 text 中寻找第一个能被 json.loads 成功解析的 JSON list。
    找到则返回对应的 Python 对象，否则返回 None。
    """
    if not isinstance(text, str):
        text = str(text)

    # 非贪心匹配任意 [...] 块，多次尝试
    for match in re.finditer(r"\[[\s\S]*?\]", text):
        snippet = match.group(0)
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return data
        except Exception:
            continue
    return None


def parse_traj(traj_str):
    if traj_str is None:
        return []
    if isinstance(traj_str, (list, dict)):
        data = traj_str
    else:
        data = extract_first_json_list(traj_str)
        if data is None:
            logger.warning("Failed to find a valid JSON list in trajectory output.")
            return []

    seq = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "h3_index" in item:
                seq.append(str(item["h3_index"]))
            else:
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

    batch_size = getattr(args, "eval_batch_size", 4)
    num_samples = len(df)
    logger.info(f"Starting batched inference: total={num_samples}, batch_size={batch_size}")

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_df = df.iloc[start:end]
        # 转成 list[dict]，方便后面取字段
        batch_records = batch_df.to_dict(orient="records")

        # 1. 构造一整个 batch 的 prompt 文本
        instructions = [build_instruction(r) for r in batch_records]
        prompts = [f"INSTRUCTION:\n{inst}\n\nRESPONSE:\n" for inst in instructions]

        # 2. 一次性 tokenizer，padding + truncation
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        ).to(device)

        # 3. 一次性 generate
        with torch.inference_mode():
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

        # 4. 按条解析输出
        for rec, prompt_text, output in zip(batch_records, prompts, outputs):
            full_text = tokenizer.decode(
                output,
                skip_special_tokens=True,
            )

            if "RESPONSE:\n" in full_text:
                pred_text = full_text.split("RESPONSE:\n", 1)[1].strip()
            else:
                # 兜底：截掉前面的 prompt
                pred_text = full_text[len(prompt_text):].strip()

            result = {
                "user": str(rec.get("user", "")),
                "date": str(rec.get("date", "")),
                "profile": str(rec.get("profile", "")),
                "ground_truth": str(rec.get("prediction", "")),  # GT
                "model_output": pred_text,                       # 模型生成
            }
            results.append(result)

        if end % args.log_interval == 0 or end == num_samples:
            logger.info(f"Inference progress: {end}/{num_samples}")

    # 5. 写预测结果 jsonl
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved predictions to: {out_path}")

    # 6. 计算 hit@1, hit@5, hit@10
    metrics = compute_hit_metrics(results)

    # 7. 存指标
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")
    logger.info("Done.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="QT-Mob Unsloth LoRA inference (test) with hit@k metrics"
    )
    parser.add_argument("--index_file", type=str, default="LLMMove/QT_Mob_main/dataset/location_r8.json", help="the item indices file, not path")
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
        default=2048,
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
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
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
        default=0,
        help="Max number of eval samples (0 = use all)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Logging interval during inference",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,   # 你可以根据显存改，比如 2/4/8
        help="Batch size for batched inference",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)
