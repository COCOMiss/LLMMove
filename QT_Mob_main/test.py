# -*- coding: utf-8 -*-
import os
import json
import logging
import argparse
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import PeftConfig, PeftModel

# Local imports
from collator import TestCollator
from h3_prompt_mobility import all_prompt
from evaluate import get_daily_traj_results, get_seq_results, extract_json_array
from utils import (
    set_seed, 
    load_test_dataset, 
    parse_global_args, 
    parse_dataset_args, 
    parse_test_args
)

# ===== Logger Configuration =====
def get_logger():
    logger = logging.getLogger("QT-Mob-Test")
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger()

# Environment Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Set working directory
try:
    WORK_DIR = "/home/linyuxi/LLM"
    os.chdir(WORK_DIR)
    logger.info(f"Changed working directory to: {os.getcwd()}")
except FileNotFoundError:
    logger.warning(f"Directory {WORK_DIR} not found, using current directory.")


def str2bool(x):
    return str(x).lower() == "true"


def load_model_and_tokenizer(args, device):
    """
    加载模型和分词器的辅助函数
    """
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {args.ckpt_path}")

    # Save testing args
    with open(os.path.join(args.ckpt_path, 'testing_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Determine dtype
    torch_dtype = torch.float16 if str(args.torch_dtype).lower() in ("float16", "fp16", "16") else torch.bfloat16

    # Quantization Config
    quantization_config = None
    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=torch_dtype,
        )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = args.cutoff_len

    # Load Base Model
    peft_config = PeftConfig.from_pretrained(args.ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize embeddings if needed
    if args.indexing:
        model.resize_token_embeddings(len(tokenizer))

    # Load Adapter
    model = PeftModel.from_pretrained(model, args.ckpt_path)
    
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer


def test(args):
    # ===== 1. 设置设备 =====
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ===== 2. 处理布尔参数 =====
    for attr in ['quantize', 'indexing', 'multi_seq', 'add_profile', 
                 'add_prefix', 'filter_items', 'multi_rec', 'single_rec']:
        if hasattr(args, attr) and isinstance(getattr(args, attr), str):
            setattr(args, attr, str2bool(getattr(args, attr)))

    set_seed(args.seed)
    
    # ===== 3. 加载模型 =====
    try:
        model, tokenizer = load_model_and_tokenizer(args, device)
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return

    # ===== 4. 准备 Prompt IDs =====
    if args.test_prompt_ids == "all":
        prompt_map = {
            "seq": all_prompt["seq"],
            "daily_traj": all_prompt["daily_traj"],
            "rec_single": all_prompt["rec_single"] 
        }
        if args.test_task in prompt_map:
             prompt_ids = range(len(prompt_map[args.test_task]))
        else:
             prompt_ids = range(len(all_prompt.get("rec_single", [])))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    # ===== 5. 准备数据 =====
    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    # ★★★ Beam Search 设置 ★★★
    BEAM_SIZE = 5  # 设置 Beam Size 为 5
    logger.info(f"Beam Search Config: Num Beams={BEAM_SIZE}, Num Return Sequences={BEAM_SIZE}")

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    metrics_list = args.metrics.split(",")
    all_prompt_results = []
    prediction_dict={}
    ground_truth_dict={}

    # ===== 6. 推理循环 =====
    with torch.no_grad():
        for prompt_id in prompt_ids:
            logger.info(f"=== Testing Prompt ID: {prompt_id} ===")
            test_loader.dataset.set_prompt(prompt_id)
            
            metrics_accumulator = {}
            total_samples = 0

            for batch_idx, batch in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx}/{len(test_loader)}")

                batch_inputs, targets = batch
                batch_size = len(targets)
                inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                total_samples += batch_size

                # === 生成 (Beam Search) ===
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=2048,
                    do_sample=False,            # 关闭采样，启用 Beam Search
                    num_beams=BEAM_SIZE,        # 设置 Beam Size
                    num_return_sequences=BEAM_SIZE, # 每个样本返回 Top-Beam 序列
                    repetition_penalty=1.1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # 解码所有生成的序列
                # output.sequences shape: [batch_size * BEAM_SIZE, seq_len]
                raw_generated_texts = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

                # 简单清洗数据：提取 JSON 数组部分 (因为模型可能会输出一些废话)
                # extract_json_array 返回的是一个 list，我们对每个生成结果都应用一下
                cleaned_generated_texts = extract_json_array([t.strip() for t in raw_generated_texts])
                
                # 确保清洗后的列表长度一致，如果有空值则回退到原始文本(避免 list 长度不匹配)
                final_texts_for_eval = []
                for raw, clean in zip(raw_generated_texts, cleaned_generated_texts):
                    final_texts_for_eval.append(clean if clean else raw.strip())

                # === 评估 ===
                # 这里直接传入所有 Beam 的结果 (batch_size * 5)
                # 评估函数内部会根据 total_predictions // batch_size 来计算 Top-1 和 Top-5
                eval_kwargs = {
                    "output_text": final_texts_for_eval, 
                    "targets": targets,
                    "scores": None,
                    "metrics": metrics_list,
                    "all_items": all_items if args.filter_items else None
                }

                if args.test_task == "daily_traj":
                    batch_metrics,best_index_list = get_daily_traj_results(**eval_kwargs)
                elif args.test_task == "seq":
                    batch_metrics = get_seq_results(**eval_kwargs)
                else:
                    raise ValueError(f"Invalid test task: {args.test_task}")
                
                # Debug 打印第一个样本的 Top-1 结果
                if batch_idx % 5 == 0: 
                    logger.info(f"\n[Sample Debug Info (Top-5 Beam)]")
                    logger.info(f"Ground Truth: {targets[0]}")
                    logger.info(f"Prediction: {final_texts_for_eval[best_index_list[0]]}\n")

                # 累加指标
                for m, res in batch_metrics.items():
                    metrics_accumulator[m] = metrics_accumulator.get(m, 0) + res

                # 定期打印临时结果
                if  batch_idx % 10 == 0:
                    temp_avg = {m: val / (batch_idx+1) for m, val in metrics_accumulator.items()}
                    logger.info(f"Intermediate Metrics (n={total_samples}): {temp_avg}")

                if args.limit_test_size and total_samples >= 1000:
                    logger.info("Hit test size limit (1000). Stopping prompt loop.")
                    break

            # === 当前 Prompt ID 最终结果 ===
            final_prompt_metrics = {m: val / (batch_idx+1) for m, val in metrics_accumulator.items()}
            all_prompt_results.append(final_prompt_metrics)
            logger.info(f"Prompt {prompt_id} Final Results: {final_prompt_metrics}")

    # ===== 7. 保存结果 =====
    save_data = {}
    if len(all_prompt_results) == 1:
        save_data = {
            "test_task": args.test_task,
            "test_prompt_ids": args.test_prompt_ids,
            "single_result": all_prompt_results[0]
        }
    else:
        # 计算所有 Prompt 的平均值
        if all_prompt_results:
            keys = all_prompt_results[0].keys()
            mean_results = {m: sum([r[m] for r in all_prompt_results]) / len(all_prompt_results) for m in keys}
        else:
            mean_results = {}
            
        save_data = {
            "test_task": args.test_task,
            "mean_results": mean_results,
            "all_prompt_results": all_prompt_results
        }

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)
    logger.info(f"All results saved to {args.results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    
    # 假设这些 parse 函数会向 parser 添加参数
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    
    args = parser.parse_args()

    # 强制覆盖参数 (调试用，生产环境建议移除或注释)
    logger.warning("Overriding args for testing purpose: indexing=True, filter_items=True, tasks=seq")
    args.indexing = True
    args.filter_items = True
    args.tasks = "seq"

    test(args)