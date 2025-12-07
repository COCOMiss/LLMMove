from logger_utils import get_logger
import os
from unsloth import FastLanguageModel

from unsloth.chat_templates import train_on_responses_only
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from logger_utils import get_logger
import torch
from trl import SFTConfig, SFTTrainer
import pandas as pd
from h3_prompt_mobility import *
import argparse
from utils import load_datasets, set_seed, ensure_dir, parse_global_args, parse_dataset_args, parse_train_args, get_new_tokens
from datasets import Dataset as HFDataset

logger = get_logger(__name__)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../LLLMOVE/QT_Mob_main
PROJECT_ROOT = os.path.dirname(THIS_DIR)              # .../LLLMOVE
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Qwen3-8B")

def process_data(args):
    # train_dataset,valid_dataset = load_datasets(args)
    train_dataset, valid_dataset = [], []

    prompt = (
        "Here is the mobility profile for user {user}."
        "The profile details their home and work locations (if known), a list of frequently visited locations with typical visit times, "
        "and their preferences for different POI categories based on visit history: {profile}"
        "Today is a {date}."
        "Task: Predict the daily trajectory of the user for this date."
        "Predict the sequence of visits, including the start time, the location (H3 index), and the stay duration (in minutes)."
        "Return ONLY a JSON list with the following format and no extra text:"
        'Example: [{{ "id": "1", "start_time": "HH:MM AM/PM", "h3_index": "...", "stay_duration": "... min" }}, ...]'
    )

    train_df = pd.read_feather("QT_Mob_main/dataset/train/zdc_h3_8/daily_traj_dataset.feather")
    valid_df = pd.read_feather("QT_Mob_main/dataset/valid/zdc_h3_8/daily_traj_dataset.feather")

    sysprompt = system_prompt_new + traj_task_prompt

    def remap_data(row):
        # 1️⃣ 先拼出完整 prompt
        base_prompt = sysprompt + prompt.format(
            user=row["user"],
            profile=row["profile"],
            date=row["date"],
        )
        # 2️⃣ 去掉 prompt 末尾多余空格/换行，避免“边界空格”被重新切分成不同 token
        base_prompt = base_prompt.rstrip(" \n\r\t")

        # 3️⃣ completion 前面主动加一个分隔（这里用换行），
        #    并去掉自己开头的空白，防止 prompt+completion 粘在一起被 tokenizer 合成新 token
        completion = str(row["prediction"]).lstrip(" \n\r\t")

        return {
            "prompt": base_prompt,
            "completion": "prediction:" + completion,
        }

    train_dataset = [remap_data(row) for _, row in train_df.iterrows()]
    valid_dataset = [remap_data(row) for _, row in valid_df.iterrows()]

    return train_dataset, valid_dataset



def train1(args):
    set_seed(getattr(args, "seed", 42))
    train_dataset, valid_dataset = process_data(args)
    train_dataset = HFDataset.from_list(train_dataset)
    valid_dataset = HFDataset.from_list(valid_dataset)

    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=args.base_model,      
    #     max_seq_length=max_seq_len,
    #     dtype=None,
    #     load_in_4bit=True,
    #     trust_remote_code=True,
    #     local_files_only=True,
    # )

    tokenizer=AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    num_added = 0
    if args.indexing:
        new_tokens = get_new_tokens(args)
        if new_tokens:
            num_added = tokenizer.add_tokens(new_tokens)
            
 
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16 ,   # 如果你的卡不支持 bf16，就换成 torch.float16
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,


    )


    model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto", 
            quantization_config=bnb_config,
        )

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False   
        
    training_args = SFTConfig(
        output_dir="./output",
        # 训练超参（全部来自你 parse_train_args 的 args）
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_checkpointing=True,
        # eval & save
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        save_strategy="steps",
        eval_strategy="steps",
        logging_steps=getattr(args, "logging_steps", 50),
        save_total_limit=getattr(args, "save_total_limit", 3),

        # LoRA + DDP 友好（很关键）
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,

        # 只对 completion 算 loss
        completion_only_loss=True,

        optim="adamw_torch",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules 需要根据你具体的 Qwen 版本来设
        # 比如 q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj 等
    )


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    # 这个会在主进程自动保存 LoRA + base 的合并或 adapter（视你设置）
    trainer.save_model("my_qwen3_mobility_lora")

    # 只在 world_process_zero (rank 0) 保存 tokenizer
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained("my_qwen3_mobility_lora")
        

def train(args):
    set_seed(getattr(args, "seed", 42))

    # 1. 构建 HF Dataset（保持你原来的）
    train_list, valid_list = process_data(args)
    train_dataset = HFDataset.from_list(train_list)
    valid_dataset = HFDataset.from_list(valid_list)

    # 1.1 映射出 text 字段：INSTRUCTION + RESPONSE
    def formatting_prompts_func(batch):
        prompts = batch["prompt"]
        completions = batch["completion"]

        # 兼容单条 / batched
        if isinstance(prompts, str):
            prompts = [prompts]
            completions = [completions]

        texts = []
        for p, c in zip(prompts, completions):
            p = p.rstrip(" \n\r\t")
            c = c.lstrip(" \n\r\t")
            # 你可以自定义这个格式，但 instruction_part / response_part 要和下面一致
            text = f"INSTRUCTION:\n{p}\n\nRESPONSE:\n{c}"
            texts.append(text)

        return {"text": texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    valid_dataset = valid_dataset.map(formatting_prompts_func, batched=True)
    remove_cols = [col for col in train_dataset.column_names if col != "text"]
    # 上面这行会删掉除了 text 以外的所有列（如果你还想保留别的列就手动写）
    train_dataset = train_dataset.remove_columns(remove_cols)
    valid_dataset = valid_dataset.remove_columns(remove_cols)
    max_seq_len = getattr(args, "max_seq_length", 512)

    # 2. Unsloth 加载 4bit 模型（QLoRA）
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = args.base_model,    # 本地路径 /Qwen3-8B
        max_seq_length  = max_seq_len,
        dtype           = None,               # 自动选 bf16 / fp16
        load_in_4bit    = True,
        trust_remote_code = True,
        local_files_only  = True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 新 token（如果有）
    if getattr(args, "indexing", False):
        new_tokens = get_new_tokens(args)
        if new_tokens:
            num_added = tokenizer.add_tokens(new_tokens)
            if num_added > 0:
                model.resize_token_embeddings(len(tokenizer))

    # 3. 给模型挂 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r              = getattr(args, "lora_r", 32),
        lora_alpha     = getattr(args, "lora_alpha", 16),
        bias           = "none",
    )

    # 4. TRL 的 SFTConfig
    training_args = SFTConfig(
        output_dir = getattr(args, "output_dir", "./output_unsloth"),

        # 如果你本地 TRL 支持 max_seq_length 就用这个；不支持就改成 max_length
        max_seq_length = max_seq_len,

        per_device_train_batch_size  = args.per_device_train_batch_size,
        per_device_eval_batch_size   = args.per_device_eval_batch_size,
        gradient_accumulation_steps  = args.gradient_accumulation_steps,

        learning_rate      = args.learning_rate,
        num_train_epochs   = args.epochs,
        lr_scheduler_type  = args.lr_scheduler_type,
        warmup_ratio       = args.warmup_ratio,
        weight_decay       = args.weight_decay,

        gradient_checkpointing = True,
        bf16 = torch.cuda.is_bf16_supported(),
        fp16 = not torch.cuda.is_bf16_supported(),

        eval_steps      = args.save_and_eval_steps,
        save_steps      = args.save_and_eval_steps,
        save_strategy   = "steps",
        eval_strategy   = "steps",
        logging_steps   = getattr(args, "logging_steps", 50),
        save_total_limit = getattr(args, "save_total_limit", 3),

        ddp_find_unused_parameters = False,
        remove_unused_columns      = True,

        # ❌ 不要再用 completion_only_loss，这里交给 train_on_responses_only 处理
        # completion_only_loss = True,

        optim  = "paged_adamw_8bit",
        packing = False,

        # 这一句很关键：告诉 TRL 文本在 "text" 这一列
        dataset_text_field = "text",
    )

    # 5. 不再给 SFTTrainer 传 formatting_func，数据已经预处理成 text
    trainer = SFTTrainer(
        model         = model,
        processing_class    = tokenizer,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = valid_dataset,
    )

    # 6. 用 Unsloth 的 train_on_responses_only —— 只对 RESPONSE 段算 loss
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "INSTRUCTION:\n",
        response_part    = "RESPONSE:\n",
    )

    trainer.train()
    trainer.save_model("my_qwen3_mobility_unsloth_lora")

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained("my_qwen3_mobility_unsloth_lora")
        
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="QT-Mob train with logging")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_train_args(parser)


    args = parser.parse_args()
    args.base_model = DEFAULT_MODEL_PATH
    # Normalize boolean args
    for key in [
        "quantize", "indexing", "multi_seq", "add_profile",
        "add_prefix", "sft_json_output", "multi_rec", "single_rec"
    ]:
        val = getattr(args, key, None)
        if isinstance(val, str):
            setattr(args, key, val.lower() == "true")

    
    # args.ckpt_path="checkpoints/qwen_tokyo"
    train(args)