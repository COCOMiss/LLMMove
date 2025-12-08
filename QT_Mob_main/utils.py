import json
import os
import random
import datetime
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data_mobility_h3_new import *
import re
import json
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'

def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_model", type=str,
                        default="Qwen3-8B")#模型路径
    parser.add_argument("--quantize", type=str, default="true", help="whether to quantize the model")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="the dtype of the model")
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="zdc_h3_8",
                        help="data directory")
    parser.add_argument("--gridinfo_path", type=str, default="zdc_h3_8/grid_profile_codebook_qwen",
                        help="grid info path")
    # parser.add_argument("--data_filename", type=str, default=".pkl",help="data filename")
    parser.add_argument("--tasks", type=str, default="daily_traj",
                        help="Downstream tasks, separate by comma")
    parser.add_argument("--index_file", type=str, default="QT_Mob_main/dataset/location_r8.json", help="the item indices file, not path")
    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of location in history trajectory, -1 means no limit")
    parser.add_argument("--add_prefix",  type=str, default="false",
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=" ", help="The separator used for history")
    parser.add_argument("--sft_json_output", type=str,default="false", help="whether to output json file for sft")
    parser.add_argument("--indexing", type=str,default="true", help="whether to index the location")
    parser.add_argument("--multi_seq", type=str,default="true", help="whether to generate multiple trajectories")
    parser.add_argument("--add_profile", type=str,default="True", help="whether to add user profile")
    parser.add_argument("--multi_rec",  type=str, default="false", help="whether to use  multi mode for recovery task")
    parser.add_argument("--single_rec", type=str, default="false", help="whether to use single mode for recovery task")
    parser.add_argument("--ablation_location_prompt", type=str, default="1", help="ablation rows of location prompt")
    return parser

def parse_train_args(parser):

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=10)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--lora_r", type=int, default=96)
    parser.add_argument("--lora_alpha", type=int, default=192)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    help="target modules for LoRA")#for qwen3
    # parser.add_argument("--lora_target_modules", type=str,
    #                     default="q_proj,k_proj,v_proj,o_proj", help="separate by comma") #for llama q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj
    parser.add_argument("--lora_modules_to_save", type=str,
                        default="embed_tokens,lm_head", help="separate by comma")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_steps", type=int, default=200)
    parser.add_argument("--experiment_name", type=str, help="The name of the experiment")
    parser.add_argument("--path_to_sft_save_dir", type=str, default="QT_Mob_main/sft",help="QT_Mob_main/sft")

    parser.add_argument("--save_total_limit", type=int, default=3)   # 自动删旧
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--greater_is_better", action="store_true")  # 如果用准确率等需要 True；loss 用 False
    parser.add_argument("--save_only_model", action="store_true")    # LoRA 强烈建议打开
 

    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str,
                        default="",
                        help="The checkpoint path")
    parser.add_argument("--filter_items",  default=True,
                        help="whether filter illegal items")
    parser.add_argument("--results_file", type=str,
                        default="QT_Mob_main/daily_traj_results/metrics.json",
                        help="result output path")
    parser.add_argument("--prediction_file", type=str,
                        default="QT_Mob_main/daily_traj_results/prediction.json",
                        help="prediction output path")
    parser.add_argument("--ground_truth_file", type=str,
                        default="QT_Mob_main/daily_traj_results/ground_truth.json",
                        help="ground truth output path")
    
    parser.add_argument("--test_batch_size", type=int, default=10)
    
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    
    
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--test_task", type=str, default="daily_traj",
                        help="test task, one of [seq, daily_traj, recovery]")
    parser.add_argument("--limit_test_size",  default=False, help="whether to limit the test size to 1000")

    return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    
    
def get_new_tokens(args):
    indices = load_json(args.index_file)
    new_tokens = set()
    for id in indices:
        for index in indices[id]:
            new_tokens.add(index)
    return list(sorted(new_tokens))


def load_datasets(args):
    set_seed(args.seed)
    tasks = args.tasks.split(",")

    train_datasets = []
    for task in tasks:
        if task.lower() == "seq":
            dataset = SeqDataset(args, mode="train")
        elif task.lower() == "daily_traj":
            dataset = DailyTrajDataset(args, mode="train")
        elif task.lower() == "recovery":
            dataset = RecoveryDataset(args, mode="train")
        elif task.lower() == "index":
            dataset = Index2LocationDataset(args)
        elif task.lower() == "location":
            dataset = Location2IndexDataset(args)
        elif task.lower() == "trans":
            dataset = TrajectoryTranslationDataset(args, mode="train")
        else:
            raise NotImplementedError
        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)
    
    valid_datasets = []
    for task in tasks:
        if task.lower() == "seq":
            dataset = SeqDataset(args, mode="valid")
        elif task.lower() == "daily_traj":
            dataset = DailyTrajDataset(args, mode="valid")
        elif task.lower() == "recovery":
            dataset = RecoveryDataset(args, mode="valid")
        elif task.lower() == "trans":
            dataset = TrajectoryTranslationDataset(args, mode="valid")
        elif task.lower() == "index" or task.lower() == "location":
            continue
        else:
            raise NotImplementedError
        valid_datasets.append(dataset)
        
    if len(valid_datasets) > 0:
        valid_data = ConcatDataset(valid_datasets)
    else:
        valid_data = None

    return train_data, valid_data

def load_test_dataset(args):
    set_seed(args.seed)
    if args.test_task.lower() == "seq":
        test_data = SeqDataset(args, mode="test")
    elif args.test_task.lower() == "daily_traj":
        test_data = DailyTrajDataset(args, mode="test")
    elif args.test_task.lower() == "recovery":
        test_data = RecoveryDataset(args, mode="test")
    elif args.test_task.lower() == "index":
        test_data = Index2LocationDataset(args, mode="test")
    elif args.test_task.lower() == "location":
        test_data = Location2IndexDataset(args, mode="test")
    else:
        raise NotImplementedError

    return test_data

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file, indent=4):
    with open(file, 'w') as f:
        json.dump(data, f, indent=indent)


# Helper to ensure directory exists
def ensure_dir_for_file(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)