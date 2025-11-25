import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collator import TestCollator
from h3_prompt_mobility import all_prompt
from evaluate import get_topk_results, get_metrics_results, get_daily_traj_results, get_seq_results
from peft import PeftConfig, PeftModel
from utils import set_seed, load_test_dataset, parse_global_args, parse_dataset_args, parse_test_args
from pathlib import Path
import json
from seq_constrained_generator import FinalConstrainedGenerator, InspectLogitsProcessor
from traj_constrained_generator import TrajConstrainedGenerator
import logging

# ===== 加载logger =====
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

# 设置 Tokenizers 并行
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# 切换工作路径
try:
    os.chdir("/home/linyuxi/LLM")
    logger.info(f"Changed working directory to: {os.getcwd()}")
except FileNotFoundError:
    logger.warning("Directory /home/linyuxi/LLM not found, using current directory.")

codebook_path = "LLMMove/QT_Mob_main/dataset/location_r8.json"

codebook = None
if os.path.exists(codebook_path):
    with open(codebook_path, "r", encoding="utf-8") as f:
        codebook = json.load(f)
else:
    logger.error(f"Codebook not found at {codebook_path}")
    # 这里根据需求，如果没有codebook是否需要报错退出
    # raise FileNotFoundError("Codebook not found")

def test(args):
    # ===== 1. 设置设备 =====
    # 获取用户指定的 device ID (例如 "0", "1", "2", "3")
    target_device_id = str(getattr(args, 'device', '0'))
    
    # 将可见设备设置为用户指定的那个 ID
    # 这样 PyTorch 内部只需要使用 "cuda:0"，它会自动映射到对应的物理显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = target_device_id
    logger.info(f"Running on Physical GPU: {target_device_id}")
    
    # 逻辑设备固定为 cuda:0 (因为我们只让一张卡可见)
    device = torch.device("cuda:0")

    # ===== 参数处理 =====
    if isinstance(args.quantize, str):
        args.quantize = args.quantize.lower() == "true"
    if isinstance(args.indexing, str):
        args.indexing = args.indexing.lower() == "true"
    if isinstance(args.multi_seq, str):
        args.multi_seq = args.multi_seq.lower() == "true"
    if isinstance(args.add_profile, str):
        args.add_profile = args.add_profile.lower() == "true"
    if isinstance(args.add_prefix, str):
        args.add_prefix = args.add_prefix.lower() == "true"
    if isinstance(args.filter_items, str):
        args.sft_json_output = args.sft_json_output.lower() == "true"
    if isinstance(args.multi_rec, str):
        args.multi_rec = args.multi_rec.lower() == "true"
    if isinstance(args.single_rec, str):
        args.single_rec = args.single_rec.lower() == "true"

    set_seed(args.seed)
    logger.info(f"Args: {vars(args)}")
    
    # 保存测试参数
    if not os.path.exists(args.ckpt_path):
        logger.error(f"Checkpoint path does not exist: {args.ckpt_path}")
        return

    with open(os.path.join(args.ckpt_path, 'testing_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    logger.info(f"Loading model from: {args.ckpt_path}")
    
    # 确定计算数据类型
    # 如果 args.torch_dtype 是 "float16" 则使用 float16，否则使用 bfloat16
    torch_dtype = torch.float16 if str(args.torch_dtype).lower() in ("float16", "fp16", "16") else torch.bfloat16
        
    # ===== 2. 加载 Tokenizer (必须从 checkpoint 加载) =====
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 4096
    
    logger.info("Use peft model with LoRA adapter")
    peft_config = PeftConfig.from_pretrained(args.ckpt_path)
    
    # ===== 3. 配置量化 =====
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    # ===== 4. 加载 Base Model =====
    # device_map="auto" 会自动将模型加载到当前可见的 GPU (即上面设置的 CUDA_VISIBLE_DEVICES)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch_dtype,               # ✅ 使用变量传入，而不是写死
        quantization_config=quantization_config if args.quantize else None,
        device_map="auto",                     
        trust_remote_code=True,
    )
        
    # ===== 5. 调整词表与加载 LoRA =====
    if args.indexing:
        logger.info(f"Resizing token embeddings to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, args.ckpt_path)
    
    # 后处理
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # ===== 6. 准备 Prompt IDs =====
    if args.test_prompt_ids == "all":
        if args.test_task == "seq":
            prompt_ids = range(len(all_prompt["seq"]))
        elif args.test_task == "daily_traj":
            prompt_ids = range(len(all_prompt["daily_traj"]))
        else:
            prompt_ids = range(len(all_prompt["rec_single"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    # ===== 7. 数据加载 =====
    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer) 
    
    all_items = test_data.get_all_items()
    
    # 初始化约束生成器
    constrained_generator = None
    if args.indexing:
        logger.info("Using indexing (Constrained Generation)")
        if args.test_task == "seq":
            constrained_generator = FinalConstrainedGenerator(tokenizer, codebook)
        else:
            # 确保 codebook 存在，否则 traj 生成器可能会报错
            if codebook is None:
                 raise ValueError("Codebook is required for TrajConstrainedGenerator")
            constrained_generator = TrajConstrainedGenerator(tokenizer, codebook)

    logger.info("Using Beam Search for evaluation")
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)
    
    if args.limit_test_size:
        logger.info("Limit test size to 1000")

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []
    
    # 初始化 Logits 检查器
    logits_inspector = InspectLogitsProcessor(tokenizer)
    
    # ===== 8. 推理循环 =====
    with torch.no_grad():
        for prompt_id in prompt_ids: 

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for batch_idx, batch in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx}")

                batch_inputs, targets = batch
                
                # 将输入移到 GPU (device此时是 cuda:0)
                inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                
                prompt_length = inputs["input_ids"].shape[1]
                
                # 获取约束函数
                prefix_fn = None
                if constrained_generator:
                    prefix_fn = constrained_generator.get_prefix_allowed_tokens_fn(
                        prompt_lengths=prompt_length,
                        logits_inspector=logits_inspector
                    )
              
                total += len(targets)
                
                # 生成
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=1024,
                    do_sample=True,             # 使用 beam search 的采样配置
                    prefix_allowed_tokens_fn=prefix_fn,
                    logits_processor=[logits_inspector], 
                    temperature=2.0,            # 之前代码设置的参数
                    top_p=1.0,
                    num_return_sequences=5,
                    output_scores=True,         # 需要返回 scores
                    return_dict_in_generate=True,
                    early_stopping=True
                )
                
                
                
                
                # 处理输出
                output_ids = output["sequences"]
                scores = output["scores"]
                
                # 解码
                output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                
                # 评估
                if args.test_task == "daily_traj":
                    metrics_res = get_daily_traj_results(output_text, targets, scores, metrics, all_items if args.filter_items else None)
                elif args.test_task == "seq":
                    metrics_res = get_seq_results(output_text, targets, scores, metrics, all_items if args.filter_items else None)
                else:
                    raise ValueError(f"Invalid test task: {args.test_task}")

                # 累加指标
                for m, res in metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if total % 20 == 0:
                    temp = {m: metrics_results[m] / total for m in metrics_results}
                    logger.info(f"Temp metric results at total {total}: {temp}")
                
                if args.limit_test_size and total >= 1000:
                    logger.info("Limit test size to 1000")
                    break

            # 计算当前 Prompt 的最终平均指标
            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total

            all_prompt_results.append(metrics_results)
            logger.info("======================================================")
            logger.info(f"Prompt {prompt_id} results: {metrics_results}")
            logger.info("======================================================")
            logger.info("")

    # ===== 9. 保存结果 =====
    if len(all_prompt_results) == 1:
        single_result = all_prompt_results[0]
        save_data = {
            "test_task": args.test_task,
            "test_prompt_ids": args.test_prompt_ids,
            "single_result": single_result
        }
    else:
        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics_results.keys(): # 使用最后一个 metrics_results 的 keys
            all_res = [_[m] for _ in all_prompt_results]
            mean_results[m] = sum(all_res) / len(all_res)
            min_results[m] = min(all_res)
            max_results[m] = max(all_res)
    
        save_data = {
            "test_task": args.test_task,
            "test_prompt_ids": args.test_prompt_ids,
            "mean_results": mean_results,
            "min_results": min_results,
            "max_results": max_results,
            "all_prompt_results": all_prompt_results
        }

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)
    logger.info(f"Results saved to {args.results_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="QT-Mob test")
    
    # 添加 device 参数，允许用户通过命令行指定
    parser.add_argument("--device", type=str, default="0", help="GPU Device ID (0, 1, 2, 3)")
    
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    
    # 硬编码参数 (如果需要覆盖命令行)
    args.indexing = True
    # args.ckpt_path = "checkpoints/qwen_tokyo" # 如需测试可取消注释
    args.filter_items = True
    args.tasks = "seq"
    
    # 如果你想在这里硬编码 device 也可以：
    # args.device = "1" 
    
    test(args)