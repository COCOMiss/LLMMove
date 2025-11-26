from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
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
def process_data(args):
    # train_dataset,valid_dataset = load_datasets(args)
    train_dataset, valid_dataset=[],[]
    prompt = (
    "Here is the mobility profile for user {user}. "
    "The profile details their home and work locations (if known), a list of frequently visited locations with typical visit times, "
    "and their preferences for different POI categories based on visit history: {profile} \n"
    "Today is a {date}. \n"
    "Task: Predict the daily trajectory of the user for this date. \n"
    "Predict the sequence of visits, including the start time, the location (H3 index), and the stay duration (in minutes). \n"
    "Return ONLY a JSON list with the following format and no extra text:\n"
    # 使用双花括号 {{ }} 来表示 JSON 的花括号，避免 .format() 报错
    """Example: [{{ "id": "1", "start_time": "HH:MM AM/PM", "h3_index": "...", "stay_duration": "... min" }}, ...]"""
)
    train_df=pd.read_feather("QT_Mob_main/dataset/train/zdc_h3_8/daily_traj_dataset.feather")
    valid_df=pd.read_feather("QT_Mob_main/dataset/valid/zdc_h3_8/daily_traj_dataset.feather")
    sysprompt=system_prompt_new+traj_task_prompt
    def remap_data(data):
        return {
        "prompt":sysprompt+prompt.format(
            user=data["user"],
            profile=data["profile"],
            date=data["date"],
        ),
        "completion": data["prediction"]
    }
    train_dataset = [remap_data(row) for index, row in train_df.iterrows()]
    valid_dataset = [remap_data(row) for index, row in valid_df.iterrows()]
        
    return train_dataset, valid_dataset


def train(args):
    train_dataset, valid_dataset=process_data(args)
    train_dataset = HFDataset.from_list(train_dataset)
    valid_dataset = HFDataset.from_list(valid_dataset)
    max_seq_len = 1024
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,      # 建议传本地目录
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.indexing:
        new_tokens = get_new_tokens(args)
        num_added = tokenizer.add_tokens(new_tokens)

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    


    training_args = SFTConfig(
        output_dir="./output",
        max_seq_length=max_seq_len,
        eos_token=tokenizer.eos_token,   # 直接跟 tokenizer 对齐
        completion_only_loss=True,       # prompt-completion 数据集默认就是只算 completion
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 官方推荐写法
        random_state=3407,
        max_seq_length=max_seq_len,
        use_rslora=False,
        loftq_config=None,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    model.save_pretrained("my_qwen3_mobility_lora")
    tokenizer.save_pretrained("my_qwen3_mobility_lora") 
        
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="QT-Mob train with logging")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_train_args(parser)

    args = parser.parse_args()

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