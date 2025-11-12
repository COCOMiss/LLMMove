import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import *
from collator import TestCollator
from h3_prompt_mobility import all_prompt
from evaluate import get_topk_results, get_metrics_results
from peft import PeftConfig, PeftModel
from utils import set_seed,load_test_dataset,parse_global_args,parse_dataset_args,parse_test_args
from pathlib import Path
# from seq_collator import CompletionOnlyCollator,SEQ_RESPONSE_TAG
from constrained_generator import ConstrainedGenerator


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 只用物理 GPU 1（按需改成你想用的物理卡号）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# codebook_path = Path("QT_Mob_main/index/QT-Mob-main/ckpt/location.index.json")
codebook_path = Path("LLMMove/QT_Mob_main/dataset/location.index.json")


codebook = None
if codebook_path.exists():
    with open(codebook_path, "r", encoding="utf-8") as f:
        codebook = json.load(f)###########

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
    print(vars(args))
    
    if "3.2" in args.ckpt_path:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "1").split(",")[0]
        
    with open(os.path.join(args.ckpt_path, 'testing_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device_map={"": 1}  # 使用 cuda:1
    # 指定在cuda:1上运行
    # torch_dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    torch_dtype = torch.float16 if str(args.torch_dtype).lower() in ("float16","fp16","16") else torch.bfloat16
    device = torch.device("cuda:0")
    print("Loading model from: ", args.ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 4096
    print("Use peft model with LoRA adapter") 
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
    
    # collator = CompletionOnlyCollator(
    #             tokenizer=tokenizer,
    #             response_tag=SEQ_RESPONSE_TAG,
    #             max_length=256
    #         )
    all_items = test_data.get_all_items()
    
    if args.indexing:
        print("Using indexing")
        prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer, args.test_task.lower())

    print("Using Beam Search for evaluation")
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)
    
    if args.limit_test_size:
        print("Limit test size to 1000")

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []
    constrained_generator = ConstrainedGenerator(tokenizer, codebook)##########
    prefix_allowed_tokens = constrained_generator.get_prefix_allowed_tokens_fn()##########
    with torch.no_grad():
        for prompt_id in prompt_ids: 

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for _, batch in enumerate(tqdm(test_loader)):

                batch_inputs, targets = batch
                # 把每个 tensor 放到 device
                inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                # inputs = batch[0].to(device)
                # targets = batch[1]
                total += len(targets)
                
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=1,
                    top_k=50,
                    top_p=0.92,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens if args.indexing else None,
                    num_beams=args.num_beams, # 使用的是beam search，并非sampling，所以会输出num_beams个结果
                    num_return_sequences=args.num_beams,
                    output_scores=True, # 返回每个token的score
                    return_dict_in_generate=True,
                    early_stopping=True
                    )
                
                
                # output = model.generate(
                #     input_ids=inputs["input_ids"],
                #     attention_mask=inputs["attention_mask"],
                #     max_new_tokens=20,
                #     do_sample=True,
                #     temperature=1,
                #     top_k=50,
                #     top_p=0.92,
                #     num_beams=args.num_beams, # 使用的是beam search，并非sampling，所以会输出num_beams个结果
                #     num_return_sequences=args.num_beams,
                #     output_scores=True, # 返回每个token的score
                #     return_dict_in_generate=True,
                #     early_stopping=True
                #     )

                # 获取H3 index和duration预测
                output_ids = output["sequences"]  # torch.Size([batch_size * num_beams, seq_len])
                scores = output["sequences_scores"]  # torch.Size([batch_size * num_beams])

                # 解码
                output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # 分离预测出的 H3 index 和 duration
                h3_predictions = []
                duration_predictions=[]
                
                for text in output_text:
                    try:
                        # Extract h3_index
                        h3_index = text.split(" assistant ")[1].split("will stay at h3 index ")[-1].split(" for")[0].split()
                        h3_predictions.append(h3_index)
                        
                        # Try to extract duration, handle cases where it might not exist
                        parts = text.split(" assistant ")[1].split("will stay at h3 index ")[-1].split(" for")
                        if len(parts) > 1:
                            duration = parts[1].split()
                            duration_predictions.append(duration)
                        else:
                            # No duration found, append empty or default value
                            duration_predictions.append([])
                    except (IndexError, ValueError) as e:
                        # Handle parsing errors gracefully
                        print(f"Error parsing text: {text[:100]}... Error: {e}")
                        h3_predictions.append([])
                        duration_predictions.append([])
                
                target_h3=[]
                target_duration=[] 
                for target in targets:
                    target_h3.append(target.split(" for ")[0].split())
                    target_duration.append(target.split(" for ")[1].split("seconds.")[0].split())
                    
                # 在这里可以继续对 predictions 和 ground truth 进行对比，计算准确率等指标
                
                # duration_topk_res = get_topk_results(duration_predictions,scores,target_duration,args.num_beams,metric='mse',
                #                             all_items=None)  
                
                h3_topk_res = get_topk_results(h3_predictions,scores,target_h3,args.num_beams,metrics=metrics,
                                            all_items=all_items if args.filter_items else None)  
                
                
               
                h3_metrics_res = get_metrics_results(h3_topk_res, metrics)

                for m, res in h3_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if total % 20 == 0:
                    temp={}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)
                
                if args.limit_test_size and total >= 1000:
                    print("Limit test size to 1000")
                    break

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total

            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")

    if len(all_prompt_results) == 1:
        single_result = all_prompt_results[0]
        print("======================================================")
        print("Single prompt result: ", single_result)
        print("======================================================")
    
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
    
        print("======================================================")
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
        print("======================================================")
    
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
    args.ckpt_path ="checkpoints/qwen_tokyo_latest"
    args.filter_items= True
    args.tasks="seq"
    test(args)
