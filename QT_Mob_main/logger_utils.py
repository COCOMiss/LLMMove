import os
import logging
from datetime import datetime

# 日志目录
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ 固定全局日志文件名（仅第一次生成）
# 使用环境变量或单例模式来避免重复生成
if not hasattr(logging, "_qt_mob_logfile"):
    logging._qt_mob_logfile = os.path.join(
        LOG_DIR,
        f"qt_mob_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
LOG_FILE = logging._qt_mob_logfile  # 全局共享路径


def get_logger(name: str, level=logging.INFO):
    """返回全局共享 logger 实例"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        ch = logging.StreamHandler()
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
