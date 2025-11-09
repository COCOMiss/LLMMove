import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
import argparse
import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


parser = argparse.ArgumentParser(description="QT-Mob")
parser.add_argument('--ckpt_path', type=str, default="liandanlu/7dim/best_loss_model.pth")
parser.add_argument('--output_dir', type=str,default="liandanlu/7dim/")
parser.add_argument('--gpu_id', type=str, default='1', help='gpu id')
parser.add_argument('--output_file', type=str, default="location.index.json")
args = parser.parse_args()
    
ckpt_path = args.ckpt_path
output_dir = args.output_dir
output_file = args.output_file
output_file = os.path.join(output_dir,output_file)
device = torch.device("cuda:"+args.gpu_id)

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'),weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]


data = EmbDataset(args.data_path)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = {}
all_indices_str = {}

all_codes = []          # 位置 i 的 token 列表，比如 ["<a_93>", ...]
all_codes_str = []      # 位置 i 的字符串化版本，用于碰撞比较
all_h3 = []             # 位置 i 对应的 h3 字符串
all_ds_idx = []         # 位置 i 对应的 Dataset 索引（0..len(data)-1）
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>","<g_{}>","<h_{}>"]

cur_idx = 0  # 跟踪 Dataset 顺序（shuffle=False 前提下）
for (d, h3_batch) in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()

    for row_ind, (index, h3) in enumerate(zip(indices, h3_batch)):
        code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
        all_codes.append(code)
        all_codes_str.append(str(code))
        all_h3.append(str(h3))
        all_ds_idx.append(cur_idx)
        cur_idx += 1

# === 设置 sk_epsilon ===
for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon = 0.0
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.005

# === 碰撞消解循环（按“位置”操作，保持四个数组一致）===
tt = 0
all_codes_str_array = np.array(all_codes_str)

while True:
    if tt >= 20 or check_collision(all_codes_str_array):
        break

    collision_item_groups = get_collision_item(all_codes_str_array)  # 里面是“位置”的列表
    for pos_group in collision_item_groups:
        # 用 Dataset 索引取回原始特征
        batch_ds_idx = [all_ds_idx[pos] for pos in pos_group]
        d_collision = torch.stack([data[i][0] for i in batch_ds_idx], dim=0).to(device)

        new_indices = model.get_indices(d_collision, use_sk=True)
        new_indices = new_indices.view(-1, new_indices.shape[-1]).cpu().numpy()

        # 把新的 code 写回“位置”对应的四个并行容器
        for pos, index in zip(pos_group, new_indices):
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_codes[pos] = code
            all_codes_str[pos] = str(code)
            # all_h3 / all_ds_idx 不变

    all_codes_str_array = np.array(all_codes_str)
    tt += 1

# === 指标与导出 ===
indices_count = get_indices_count(all_codes_str_array)
print("All indices number: ", len(all_codes))
print("Max number of conflicts: ", max(indices_count.values()))
tot_item = len(all_codes_str_array)
tot_indice = len(set(all_codes_str_array.tolist()))
print("Collision Rate", (tot_item - tot_indice) / tot_item)

# 回填成字典（一个 h3 只能有一个 code）
all_indices_dict = {h3: code for h3, code in zip(all_h3, all_codes)}

with open(output_file, 'w', encoding='utf-8') as fp:
    json.dump(all_indices_dict, fp, ensure_ascii=False, indent=2)