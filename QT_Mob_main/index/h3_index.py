 
     
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import ast

 
if __name__ == '__main__':
 
    df = pd.read_csv("QT-Mob-main/data/h3_emb/shinjuku_emb.csv", index_col='h3')
    
    # 读取 index 列表文件
    json_path = "ckpt/location.index.json"
    # 读取 json 列表
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        location_dict = json.load(f)
    # 获取新的索引列表
    new_index_list = list(df.index)
    # 构建新的字典
    new_location_dict = {}
    for idx, key in enumerate(location_dict.keys()):
        new_key = new_index_list[idx] if idx < len(new_index_list) else key
        new_location_dict[new_key] = location_dict[key]
        
    # 保存 new_location_dict 到 json 文件
    with open("ckpt/location_h3.index.json", "w", encoding="utf-8") as f:
        json.dump(new_location_dict, f, ensure_ascii=False, indent=4)

   

        


