import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collator import TestCollator
from h3_prompt_mobility import all_prompt
from evaluate import get_topk_results, get_metrics_results,get_daily_traj_results,get_seq_results
from peft import PeftConfig, PeftModel
from utils import set_seed,load_test_dataset,parse_global_args,parse_dataset_args,parse_test_args
from pathlib import Path
import json
import re
# from seq_collator import CompletionOnlyCollator,SEQ_RESPONSE_TAG
from constrained_generator import ConstrainedGenerator
from new_constrained_generator import FinalConstrainedGenerator,InspectLogitsProcessor

# ===== 加载logger =====
import logging

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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 只用物理 GPU 1（按需改成你想用的物理卡号）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# 切换到 /home/linyuxi/LLM 作为工作路径
os.chdir("/home/linyuxi/LLM")
logger.info(f"Changed working directory to: {os.getcwd()}")
codebook_path = "LLMMove/QT_Mob_main/dataset/location_r8.json"


codebook = None
if os.path.exists(codebook_path):
    with open(codebook_path, "r", encoding="utf-8") as f:
        codebook = json.load(f)
else:
    logger.error("Codebook not found")
    raise FileNotFoundError("Codebook not found")

def test(args):
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
    
    if "3.2" in args.ckpt_path:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "1").split(",")[0]
        
    with open(os.path.join(args.ckpt_path, 'testing_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device_map={"": 1}  # 使用 cuda:1
    # 指定在cuda:1上运行
    # torch_dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    torch_dtype = torch.float16 if str(args.torch_dtype).lower() in ("float16","fp16","16") else torch.bfloat16
    device = torch.device("cuda:0")
    logger.info(f"Loading model from: {args.ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 4096
    logger.info("Use peft model with LoRA adapter")
    peft_config = PeftConfig.from_pretrained(args.ckpt_path)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    # 单卡 + 量化推荐：让 HF 自动分配设备
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,   # 你已经是绝对路径 ✅
        dtype=torch.bfloat16,                  # 用 dtype（替代 torch_dtype）
        quantization_config=quantization_config if args.quantize else None,  # BitsAndBytesConfig 或 None
        device_map="auto",                     # ✅ 关键修复：不要用 {0:1}
        trust_remote_code=True,
    )
        
    if args.indexing:
        model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.ckpt_path)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if args.test_prompt_ids == "all":
        if args.test_task == "seq":
            prompt_ids = range(len(all_prompt["seq"]))
        else:
            prompt_ids = range(len(all_prompt["rec_single"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer) # collator是一个类，用于tokenize输入
    
    all_items = test_data.get_all_items()
    logger.info(f"All items: {all_items}")
    
    if args.indexing:
        logger.info("Using indexing")
        constrained_generator = FinalConstrainedGenerator(tokenizer, codebook)
        # 获取带日志的调试函数
        # prefix_allowed_tokens = constrained_generator.get_prefix_allowed_tokens_fn() # 假设你把上面的代码放到了这个新方法里
        # prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer, args.test_task.lower())

    logger.info("Using Beam Search for evaluation")
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)
    
    if args.limit_test_size:
        logger.info("Limit test size to 1000")

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []
      
    logits_inspector = InspectLogitsProcessor(tokenizer)
    
    with torch.no_grad():
        for prompt_id in prompt_ids: 

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for batch_idx, batch in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx}")

                batch_inputs, targets = batch
                # 把每个 tensor 放到 device
                inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                
                prompt_length = inputs["input_ids"].shape[1]
                
                prefix_fn = constrained_generator.get_prefix_allowed_tokens_fn(
                    prompt_lengths=prompt_length,
                    logits_inspector=logits_inspector # 传入实例
                )
              
                total += len(targets)
                
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    do_sample=True,  # 使用 beam search，不使用采样
                    prefix_allowed_tokens_fn=prefix_fn,
                    logits_processor=[logits_inspector], # 将诊断器作为logits processor传入
                    num_return_sequences=5,
                    top_k=None,
                    top_p=1.0,
                    output_scores=True, # 返回每个token的score
                    return_dict_in_generate=True,
                    early_stopping=True
                )
                
                # 获取H3 index和duration预测
                output_ids = output["sequences"]  # torch.Size([batch_size * num_return_sequences, seq_len])
                scores= output["scores"] 
                
                # scores 是长度为生成步数的list，每个元素是一个形如 [batch*num_return_sequences, vocab_size] 的tensor
                # 检查每个张量中不为-inf的token位置
                num_return_sequences = 5  # 与 generate 调用中的 num_return_sequences 保持一致
                batch_size = output_ids.shape[0] // num_return_sequences
                
                logger.info(f"\n=== DEBUGGING SCORES ===")
                logger.info(f"Total generation steps: {len(scores)}")
                logger.info(f"Batch size: {batch_size}, Number of return sequences: {num_return_sequences}")
                logger.info(f"Output sequences shape: {output_ids.shape}")
                
                for idx, score_tensor in enumerate(scores):
                    # score_tensor 形状通常是 [batch*num_return_sequences, vocab_size]
                    logger.info(f"\nStep {idx}: Score tensor shape: {score_tensor.shape}")
                    
                    # 检查第一个序列的 scores
                    if score_tensor.dim() == 2:
                        first_seq_scores = score_tensor[0]
                    else:
                        first_seq_scores = score_tensor
                    
                    # 查找不为-inf的位置
                    not_inf_mask = first_seq_scores != float('-inf')
                    not_inf_indices = not_inf_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                    not_inf_count = not_inf_mask.sum().item()
                    
                    logger.info(f"  Non -inf tokens count: {not_inf_count} / {len(first_seq_scores)}")
                    if not_inf_count > 0:
                        logger.info(f"  Non -inf token indices (first 20): {not_inf_indices[:20]}")
                        # 打印这些 token 的实际值
                        non_inf_values = first_seq_scores[not_inf_mask][:10]
                        logger.info(f"  Sample non-inf logit values: {non_inf_values.cpu().tolist()}")
                    else:
                        logger.warning(f"  ⚠️ WARNING: All tokens are -inf at step {idx}!")
                        logger.warning(f"  This means prefix_allowed_tokens_fn returned empty list or invalid tokens!")
                        logger.warning(f"  Check the debug logs from prefix_allowed_tokens_fn above!")

                # 解码
                output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                
                
                if args.test_task == "daily_traj":
                    metrics_res = get_daily_traj_results(output_text,targets,scores,metrics,all_items if args.filter_items else None)
                elif args.test_task == "seq":
                    metrics_res = get_seq_results(output_text,targets,scores,metrics,all_items if args.filter_items else None)
                # elif args.test_task == "recovery":
                #     metrics_res = get_recovery_results(output_text)
                else:
                    raise ValueError(f"Invalid test task: {args.test_task}")

                # 分离预测出的 H3 index 和 duration
                # h3_predictions = []
                # duration_predictions=[]
                
                # for text in output_text:
                #     try:
                #         h3_index, duration = formatting_func(text)
                #         h3_predictions.append(h3_index)
                #         duration_predictions.append(duration)
                #     except (IndexError, ValueError) as e:
                #         # Handle parsing errors gracefully
                #         logger.warning(f"Error parsing text: {text[:100]}... Error: {e}")
                #         h3_predictions.append([])
                #         duration_predictions.append([])
                
                # target_h3=[]
                # target_duration=[] 
                # for target in targets:
                #     t_h3_index, t_duration = formatting_labels(target)
                #     target_h3.append(t_h3_index)
                #     target_duration.append(t_duration)
                    
                # h3_topk_res = get_topk_results(h3_predictions,scores,target_h3,5,metrics=metrics,
                #                             all_items=all_items if args.filter_items else None)

                # h3_metrics_res = get_metrics_results(h3_topk_res, metrics)

                for m, res in metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if total % 20 == 0:
                    temp={}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    logger.info(f"Temp metric results at total {total}: {temp}")
                
                if args.limit_test_size and total >= 1000:
                    logger.info("Limit test size to 1000")
                    break

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total

            all_prompt_results.append(metrics_results)
            logger.info("======================================================")
            logger.info(f"Prompt {prompt_id} results: {metrics_results}")
            logger.info("======================================================")
            logger.info("")

    if len(all_prompt_results) == 1:
        single_result = all_prompt_results[0]
        logger.info("======================================================")
        logger.info(f"Single prompt result: {single_result}")
        logger.info("======================================================")
    
        save_data = {}
        save_data["test_task"] = args.test_task
        save_data["test_prompt_ids"] = args.test_prompt_ids
        save_data["single_result"] = single_result
    else:
        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics:
            all_res = [_[m] for _ in all_prompt_results]
            mean_results[m] = sum(all_res) / len(all_res)
            min_results[m] = min(all_res)
            max_results[m] = max(all_res)
    
        logger.info("======================================================")
        logger.info(f"Mean results: {mean_results}")
        logger.info(f"Min results: {min_results}")
        logger.info(f"Max results: {max_results}")
        logger.info("======================================================")
    
        save_data = {}
        save_data["test_task"] = args.test_task
        save_data["test_prompt_ids"] = args.test_prompt_ids
        save_data["mean_results"] = mean_results
        save_data["min_results"] = min_results
        save_data["max_results"] = max_results
        save_data["all_prompt_results"] = all_prompt_results

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="QT-Mob test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    args.indexing   = True
    # args.ckpt_path ="checkpoints/qwen_tokyo"
    args.filter_items= True
    args.tasks="seq"
    test(args)
