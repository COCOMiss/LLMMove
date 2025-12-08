import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
import json
import torch
import argparse
from accelerate import Accelerator
from utils import load_datasets, set_seed, ensure_dir, parse_global_args, parse_dataset_args, parse_train_args, get_new_tokens
from datasets import Dataset as HF_Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from liger_kernel.transformers import apply_liger_kernel_to_llama
from logger_utils import get_logger
from trl import SFTTrainer, SFTConfig
# from seq_collator import CompletionOnlyCollator,SEQ_RESPONSE_TAG
from seq_collator import prepare_tokenized_dataset_with_yesterday,SimpleCollatorWithYesterday,SEQ_RESPONSE_TAG
from collator import TestCollator
from peft import PeftConfig, PeftModel
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from verify_acc import compute_batch_token_accuracy, validate_accuracy_implementation,compute_dataset_token_accuracy,simple_collate_fn_for_prompt_completion
from transformers import EarlyStoppingCallback
from traj_trainer import SoftPositionMatchCopyPenaltyTrainer




logger = get_logger(__name__)

logger.info("==== Training script started ====")

def test_gradient_flow_safe(model, train_data, collator, logger):
    """
    安全的梯度测试函数，正确处理数据类型
    """
    logger.info("=" * 50)
    logger.info("测试梯度流动...")
    
    try:
        # 获取模型信息
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        logger.info(f"  Model device: {device}, dtype: {model_dtype}")
        
        # 准备测试数据
        batch = [train_data[i] for i in range(min(2, len(train_data)))]
        batch = collator(batch)
        
        # 移动到设备（Trainer 内部会处理 dtype）
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        model.train()
        
        # 使用 autocast
        with torch.cuda.amp. autocast(dtype=model_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        logger.info(f"  Loss: {outputs.loss.item():.4f}")
        logger. info(f"  requires_grad: {outputs.loss. requires_grad}")
        logger.info("  ✅ 梯度测试通过")
        
    except Exception as e:
        logger.warning(f"  ⚠️ 梯度测试跳过: {str(e)}")
        logger. info("  Trainer 会自动处理类型转换，继续训练...")
    
    logger.info("=" * 50)

def main(args):

    logger.info("Applying Liger Kernel optimizations for Qwen...")
    apply_liger_kernel_to_qwen2()  # <--- 这行代码价值 20GB 显存

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
                device_map="balanced",                     # ✅ 关键修复：不要用 {0:1}
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

            if args.quantize:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    # This enables offloading parts of the model to CPU if GPU is full
                    llm_int8_enable_fp32_cpu_offload=True 
                )
            else:
                quantization_config = None

            # 2. Update the model loading call
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                use_cache=False,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                # 'auto' is generally better for offloading than 'balanced' when memory is tight
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
 
        if valid_data is not None and len(valid_data) > 0:
            valid_data_list = [valid_data[i] for i in range(len(valid_data))]
        
        else:
            logger.warning("⚠️ 没有加载到验证集或验证集为空，将使用训练集的一部分作为验证集。")
         # 随机划分一部分训练集当作验证集
            total_len = len(train_data)
            val_ratio = 0.2  # 按需调整
            val_size = max(1, int(total_len * val_ratio))
            valid_data_list = [train_data[i] for i in range(val_size)]
            
        
        # 使用方式
        train_data_list = [train_data[i] for i in range(len(train_data))]
       
        processed_train = prepare_tokenized_dataset_with_yesterday(
            train_data_list, 
            tokenizer, 
            args.cutoff_len, 
            SEQ_RESPONSE_TAG, 
            postfix
        )
        train_data = HF_Dataset.from_list(processed_train)

        # ✅ 验证集也要用相同的处理方式
        processed_valid = prepare_tokenized_dataset_with_yesterday(
            valid_data_list, 
            tokenizer, 
            args.cutoff_len, 
            SEQ_RESPONSE_TAG, 
            postfix
        )
        valid_data = HF_Dataset.from_list(processed_valid)
        
        sample = train_data[0]
        logger.info(f"Sample keys: {sample.keys()}")
        logger.info(f"input_ids length: {len(sample['input_ids'])}")
        logger. info(f"labels length: {len(sample['labels'])}")
        logger.info(f"Number of -100 in labels: {sum(1 for x in sample['labels'] if x == -100)}")

        # else:
            
        #     train_data = [
        #             {"text": item["input_ids"] + item["labels"] + postfix} 
        #             for item in train_data_list
        #         ]
        #     train_data = HF_Dataset.from_list(train_data)
        #     # ✅ 验证集使用相同格式
        #     valid_data = [
        #         {"text": item["input_ids"] + item["labels"] + postfix} 
        #         for item in valid_data_list
        #     ]
        #     valid_data = HF_Dataset.from_list(valid_data)
                
        logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(valid_data) if valid_data is not None else 0}")
        logger.info("Data collator initialized.")
        
        model.gradient_checkpointing_enable()
        # ================= Training =================
        tokenizer.model_max_length = args.cutoff_len
        if args.indexing:
                model.resize_token_embeddings(len(tokenizer))
                logger.info("Token embeddings resized for new tokens.")
    
        
        # if args.tasks in ['seq','daily_traj','index','location']:
        
            
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,        # 容忍多少次评估没有改善
            early_stopping_threshold=0.02    # 最小改善阈值（可选）
        )
                
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
            max_grad_norm=1.0,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            ddp_find_unused_parameters=False,
            learning_rate=args.learning_rate,
            logging_steps=5,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            per_device_train_batch_size=args.per_device_train_batch_size,
            weight_decay=args.weight_decay,
            gradient_checkpointing=True,
            dataset_num_proc=4,
            packing=False,
            save_total_limit=args.save_total_limit,
            save_only_model=args.save_only_model,
            save_safetensors=True,
            max_length=args.cutoff_len,
            remove_unused_columns=False,
            dataset_kwargs={"remove_unused_columns": False}
            
        )
        
       
        collator = SimpleCollatorWithYesterday(
            tokenizer=tokenizer,
            max_length=args.cutoff_len
        )
        
        trainer = SoftPositionMatchCopyPenaltyTrainer(
            model=model,
            args=train_args,
            train_dataset=train_data,
            eval_dataset=valid_data,              # ✅ 验证集已传入
            peft_config=peft_config,
            callbacks=[early_stopping_callback],  # ✅ 早停回调已传入
            data_collator=collator,
            copy_penalty_weight=args.copy_penalty_weight,
            copy_threshold=args.copy_threshold,
        )
        
        
        if accelerator.is_main_process:
            test_gradient_flow_safe(model, train_data, collator, logger)
            
        
            
        from functools import partial
        from torch.utils.data import DataLoader 
        collate_fn = partial(
            simple_collate_fn_for_prompt_completion, 
            tokenizer=tokenizer,
            max_length=args.cutoff_len
        )
        # 创建 DataLoader 用于验证 accuracy
        test_dataloader = DataLoader(
            valid_data,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn
        )
                    
        # 获取一个 batch
        test_batch = next(iter(test_dataloader))
        
        # 获取设备
        device = next(model.parameters()).device
        
        # 验证实现
        result = validate_accuracy_implementation(model, test_batch, device)
        logger.info(f"Validation result: {result}")


        if trainer.accelerator.is_main_process:
            trainer.model.print_trainable_parameters()
            logger.info("Trainable parameters printed.")

        logger.info("Starting fine-tuning...")
        trainer.train()
        logger.info("Training finished successfully.")
        
          
        # eval_dataloader = DataLoader(
        #         valid_data,
        #         batch_size=args.per_device_eval_batch_size,
        #         shuffle=False,
        #         collate_fn=collate_fn
        #     )

        # device = next(model. parameters()).device

        # # 计算整个数据集的 accuracy
        # result = compute_dataset_token_accuracy(
        #     model,
        #     eval_dataloader,
        #     device=device,
        #     max_batches=50  # 只计算前50个 batch 用于快速验证
        # )
        
        # ================= Save =================
        logger.info("Saving model and tokenizer...")
        trainer.save_model(model_id)
        trainer.create_model_card()
        tokenizer.save_pretrained(model_id)
        logger.info(f"Model saved at: {model_id}")


        logger.info(f"Our implementation - Mean Token Accuracy: {result['mean_token_accuracy']:.4f}")
        logger.info(f"Our implementation - Mean Loss: {result['mean_loss']:.4f}")

      

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
    
    # args.ckpt_path="checkpoints/qwen_tokyo"
    main(args)