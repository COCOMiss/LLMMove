import os
import json
import torch
import argparse
from accelerate import Accelerator
from utils import load_datasets, set_seed, ensure_dir, parse_global_args, parse_dataset_args, parse_train_args, get_new_tokens,formatting_func
from datasets import Dataset as HF_Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from liger_kernel.transformers import apply_liger_kernel_to_llama
from logger_utils import get_logger
from trl import SFTTrainer, SFTConfig
from seq_collator import CompletionOnlyCollator,SEQ_RESPONSE_TAG,END_TAG
from seq_trainer import DualTaskTrainer
from collator import TestCollator
from peft import PeftConfig, PeftModel

logger = get_logger(__name__)
"""
Usage:

CUDA_VISIBLE_DEVICES="2,3,4,6"  torchrun --nproc_per_node=4 train.py 
"""
logger.info("==== Training script started ====")

def main(args):
    accelerator = Accelerator()
    try:
        set_seed(args.seed)
        model_id = os.path.join(args.path_to_sft_save_dir, args.experiment_name)

        if accelerator.is_main_process:
            ensure_dir(model_id)
            with open(os.path.join(model_id, 'training_args.json'), 'w') as f:
                json.dump(vars(args), f, indent=4)
            logger.info(f"Experiment initialized at: {model_id}")

        # ================= Tokenizer =================
        logger.info(f"Loading tokenizer: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Pad token set to EOS token.")

        if args.indexing:
            new_tokens = get_new_tokens(args)
            tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} new tokens.")
            
            
        torch_dtype = torch.float16 if str(args.torch_dtype).lower() in ("float16","fp16","16") else torch.bfloat16
       
       
       
       
        if os.path.exists(args.ckpt_path):
            logger.info("Loading tokenizer from checkpoint: ", args.ckpt_path)
            tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
            logger.info("Use peft model with LoRA adapter") 
            peft_config = PeftConfig.from_pretrained(args.ckpt_path)
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_storage=torch_dtype,
            )
            
            logger.info("Loading model from checkpoint: ", args.ckpt_path)
            # 单卡 + 量化推荐：让 HF 自动分配设备
            model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,   # 你已经是绝对路径 ✅
                dtype=torch.bfloat16,                  # 用 dtype（替代 torch_dtype）
                quantization_config=quantization_config if args.quantize else None,  # BitsAndBytesConfig 或 None
                device_map="auto",                     # ✅ 关键修复：不要用 {0:1}
                trust_remote_code=True,
            )
            
            model = PeftModel.from_pretrained(model, args.ckpt_path)   
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            
        else:
            logger.info(f"Loading tokenizer: {args.base_model}")
            tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Pad token set to EOS token.")
            
            if args.indexing:
                new_tokens = get_new_tokens(args)
                tokenizer.add_tokens(new_tokens)
                logger.info(f"Added {len(new_tokens)} new tokens.")
                
            logger.info(f"Loading model: {args.base_model}")
                
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_storage=torch_dtype,
            )

            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                use_cache=False,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config if args.quantize else None,
                device_map="auto",
                trust_remote_code=True,
            )

            logger.info("Model loaded successfully.")

            # ================= LoRA =================
            logger.info("LoRA configuration prepared.")
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules.split(","),
                modules_to_save=args.lora_modules_to_save.split(",") if args.indexing else None,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
        

        # ================= Datasets =================
        logger.info("Loading datasets...")
        train_data, valid_data = load_datasets(args)
        postfix = tokenizer.eos_token if args.indexing else ". " + tokenizer.eos_token

        if valid_data is None:
            logger.warning("⚠️ 没有加载到验证集，将使用训练集的一部分作为验证集。")
         # 随机划分一部分训练集当作验证集
            total_len = len(train_data)
            val_ratio = 0.1  # 按需调整
            val_size = max(1, int(total_len * val_ratio))
            valid_data = [train_data[i] for i in range(val_size)]
            train_data = [train_data[i] for i in range(val_size, total_len)]
        if valid_data is not None:
            valid_data_list = [valid_data[i] for i in range(len(valid_data))]
            valid_data = [{"text": item["labels"] + postfix} for item in valid_data_list]
            valid_data = HF_Dataset.from_list(valid_data)

        train_data_list = [train_data[i] for i in range(len(train_data))]
        train_data = [{"text": item["labels"] + postfix} for item in train_data_list]
        train_data = HF_Dataset.from_list(train_data)
        logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(valid_data) if valid_data is not None else 0}")

        if accelerator.is_main_process:
            random_indices = torch.randperm(len(train_data))[:5].tolist()
            for idx in random_indices:
                logger.info(f"Sample[{idx}]: {train_data[idx]}")

        # ================= Collator =================
        response_template_with_context = "<|im_end|>"
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
        # collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
        logger.info("Data collator initialized.")

        # ================= Model =================
        # torch_dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
        # device_index = accelerator.process_index
       

        

       

        # apply_liger_kernel_to_llama()  # 可选
        
        model.gradient_checkpointing_enable()
        # ================= Training =================
        tokenizer.model_max_length = args.cutoff_len
        if args.indexing:
                model.resize_token_embeddings(len(tokenizer))
                logger.info("Token embeddings resized for new tokens.")
    
        
        if args.tasks=='seq':
            train_args = SFTConfig(
                seed=args.seed,
                output_dir=model_id,
                eval_steps=args.save_and_eval_steps,
                save_steps=args.save_and_eval_steps,
                save_strategy="steps",
                eval_strategy="steps",
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                lr_scheduler_type=args.lr_scheduler_type,
                warmup_ratio=args.warmup_ratio,
                fp16=torch_dtype == torch.float16,
                bf16=torch_dtype == torch.bfloat16,
                dataloader_num_workers=8,
                num_train_epochs=args.epochs,
                optim="adamw_torch",
                report_to="none",
                ddp_find_unused_parameters=False,
                learning_rate=args.learning_rate,
                logging_steps=5,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                per_device_train_batch_size=args.per_device_train_batch_size,
                weight_decay=args.weight_decay,
                gradient_checkpointing=True,
                dataset_num_proc=4,
                packing=False
            )

            logger.info("Data collator initialized.")
            collator = CompletionOnlyCollator(
                tokenizer=tokenizer,
                response_tag=SEQ_RESPONSE_TAG,
                max_length=args.cutoff_len
            )
            
            
            trainer = SFTTrainer(
                model=model,
                args=train_args,
                train_dataset=train_data,
                eval_dataset=valid_data,
                peft_config=peft_config,
                data_collator=collator,
            )
            
            
            # trainer = DualTaskTrainer(
            #     model=model,
            #     args=train_args,
            #     train_dataset=train_data,
            #     eval_dataset=valid_data,
            #     peft_config=peft_config,
            #     data_collator=collator,
            # )
                     
        else:
            train_args = SFTConfig(
                gradient_checkpointing_kwargs={'use_reentrant': True},
                dataset_text_field="text",
                seed=args.seed,
                output_dir=model_id,
                eval_steps=args.save_and_eval_steps,
                save_steps=args.save_and_eval_steps,
                save_strategy="steps",
                eval_strategy="steps",
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                lr_scheduler_type=args.lr_scheduler_type,
                warmup_ratio=args.warmup_ratio,
                fp16=torch_dtype == torch.float16,
                bf16=torch_dtype == torch.bfloat16,
                dataloader_num_workers=8,
                num_train_epochs=args.epochs,
                optim="adamw_torch",
                report_to="none",
                ddp_find_unused_parameters=False,
                learning_rate=args.learning_rate,
                logging_steps=5,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                per_device_train_batch_size=args.per_device_train_batch_size,
                weight_decay=args.weight_decay,
                gradient_checkpointing=True,
            )
            logger.info("Trainer configuration created.")   
            
            
            trainer = SFTTrainer(
                model=model,
                args=train_args,
                train_dataset=train_data,
                eval_dataset=valid_data,
                peft_config=peft_config
            )
            
        

        if trainer.accelerator.is_main_process:
            trainer.model.print_trainable_parameters()
            logger.info("Trainable parameters printed.")

        logger.info("Starting fine-tuning...")
        trainer.train()
        logger.info("Training finished successfully.")

        # ================= Save =================
        logger.info("Saving model and tokenizer...")
        trainer.save_model(model_id)
        trainer.create_model_card()
        tokenizer.save_pretrained(model_id)
        logger.info(f"Model saved at: {model_id}")

    except Exception:
        logger.exception("Training failed due to an unexpected error.")
        raise



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

    logger.info("Arguments parsed successfully.")
    
    args.ckpt_path="checkpoints/qwen_tokyo_latest"
    main(args)