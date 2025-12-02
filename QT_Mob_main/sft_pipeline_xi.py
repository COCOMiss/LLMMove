import os
CUDA_VISIBLE_DEVICES = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
import multiprocessing as mp
from utils import *
import argparse
from test import test
import torch
import sys
import importlib.util
from pathlib import Path
from test import test
from utils import parse_dataset_args, parse_global_args, parse_train_args, parse_test_args
from logger_utils import get_logger
logger = get_logger(__name__)
logger.info("==== QT_Mob runner started ====")

# 开关
TRAIN = True
TEST = True

# PATH_TO_SFT_SAVE_DIR = "autodl-tmp/checkpoint"
PATH_TO_SFT_SAVE_DIR ="checkpoint"
# 你的 train.py 路径（优先用项目内的，若不存在则用上传的）
TRAIN_SCRIPT_PATH = "QT_Mob_main/train.py"
# if not Path(TRAIN_SCRIPT_PATH).exists() and Path("/mnt/data/train.py").exists():
#     TRAIN_SCRIPT_PATH = "/mnt/data/train.py"

# 切换到 /home/linyuxi/LLM 作为工作路径
# os.chdir("/home/linyuxi/LLM")
print(f"Changed working directory to: {os.getcwd()}")



def import_train_module(train_script_path: str):
    """动态导入 train.py 文件"""
    try:
        spec = importlib.util.spec_from_file_location("qt_mob_train", train_script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法从 {train_script_path} 导入 train.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["qt_mob_train"] = mod
        spec.loader.exec_module(mod)
        logger.info(f"✅ 成功导入训练脚本: {train_script_path}")
        return mod
    except Exception:
        logger.exception(f"❌ 导入 {train_script_path} 失败")
        raise

def train_model_trl(args):
    """单进程直接调用 train.py 的 main(args)"""
    try:
        train_mod = import_train_module(TRAIN_SCRIPT_PATH)
        if not hasattr(train_mod, "main"):
            raise AttributeError(f"{TRAIN_SCRIPT_PATH} 中未找到 main(args) 函数")
        logger.info(f"开始单进程训练: {TRAIN_SCRIPT_PATH}::main(args)")
        train_mod.main(args)
        logger.info("✅ 单进程训练完成。")
    except Exception:
        logger.exception("❌ 训练过程中出现错误")
        raise

def test_model(args):
    """执行测试"""
    try:
        args.ckpt_path = f"{args.path_to_sft_save_dir}/{args.experiment_name}"
        if args.test_task == "seq":
            args.results_file = f"{args.path_to_sft_save_dir}/{args.experiment_name}/test_results.json"
        elif args.test_task == "recovery":
            args.results_file = f"{args.path_to_sft_save_dir}/{args.experiment_name}/test_results_rec.json"

        logger.info(f"开始测试任务: {args.test_task}")
        test(args)
        logger.info(f"✅ 测试任务 {args.test_task} 完成，结果文件: {args.results_file}")
    except Exception:
        logger.exception("❌ 测试阶段出错")
        raise


def choose_model(base_model):
    """选择模型路径"""
    mapping = {
        "3.2": "path to Llama-3.2-1B-Instruct",
        "3.1": "path to Llama-3.1-8B-Instruct",
        "tiny": "path to TinyLlama_v1.1",
        "qwen": "Qwen3-8B",
        "phi": "path to phi-1_5",
        "olmo": "path to OLMo-1B-0724-hf",
    }
    if base_model not in mapping:
        logger.error(f"Unknown base model: {base_model}")
        raise NotImplementedError(f"Unknown base model: {base_model}")
    logger.info(f"Base model '{base_model}' 选择路径: {mapping[base_model]}")
    return mapping[base_model]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='QT_Mob')
    parser = parse_dataset_args(parser)
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_test_args(parser)
    args = parser.parse_args()
    
    

    # ==============================
    # 参数配置日志
    # ==============================
    TEST_METRICS = "hit@1,hit@5"
    args.path_to_sft_save_dir = PATH_TO_SFT_SAVE_DIR
    args.metrics = TEST_METRICS
    logger.info(f"CUDA_VISIBLE_DEVICES = {CUDA_VISIBLE_DEVICES}")
    BASE_MODEL = "qwen"
    args.base_model = "/root/autodl-tmp/Qwen/Qwen3-8B"  # 本地模型绝对路径
    # args.base_model = "Qwen3-8B"  # 其他本地模型路径示例
    args.index_file = "QT_Mob_main/dataset/location_r8.json"
    # DATASET_PATH = "./zdc_h3_index"

    TRAIN_TASKS = ["daily_traj"]
    TEST_TASK = "daily_traj"
    CUSTOM_NAME = "tokyo_daily_traj_with_last_day_traj"

    args.tasks = ",".join(TRAIN_TASKS)
    # args.data_path = DATASET_PATH
    args.experiment_name = BASE_MODEL + "_" + CUSTOM_NAME
    # args.num_workers=8
    # args.parallel_backend="process"
    # 布尔开关
    # 把这些改成 True/False（不加引号）
    args.indexing   = True
    args.multi_seq  = True
    args.add_profile= True
    args.multi_rec  = True
    args.single_rec = True
    args.add_last_day = True
    args.epochs = 1

    logger.info(f"训练任务: {args.tasks} | 测试任务: {TEST_TASK}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"实验名称: {args.experiment_name}")
    logger.info(f"模型: {args.base_model}")

    # ==============================
    # 训练与测试流程
    # ==============================
    try:
        if TRAIN:
            logger.info("🚀 开始单进程训练流程")
            # 如果GPU使用不均衡，可以尝试以下替代方案：
            # 1. 使用 torchrun 进行多进程分布式训练（取消下面的注释）
            # import subprocess
            # cmd = ["torchrun", "--nproc_per_node=4", "--master_port=29500", "LLMMove/QT_Mob_main/train.py"]
            # # 添加命令行参数
            # for key, value in vars(args).items():
            #     if isinstance(value, bool):
            #         if value:
            #             cmd.append(f"--{key}")
            #     else:
            #         cmd.append(f"--{key}={value}")
            # result = subprocess.run(cmd)
            # if result.returncode != 0:
            #     raise RuntimeError(f"torchrun failed with return code {result.returncode}")

            train_model_trl(args)

        if TEST:
            for task in TEST_TASK.split(","):
                args.test_task = task
                logger.info(f"🧪 开始测试任务: {task}")
                test_model(args)

        torch.cuda.empty_cache()
        logger.info("✅ 所有流程执行完毕，GPU缓存已释放。")

    except Exception:
        logger.exception("❌ Runner 主流程执行失败。")
        raise