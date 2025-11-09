import pandas as pd
import numpy as np

# === 1. 读取与预处理 ===
df = pd.read_feather("hex2vec-main/data/processed/Tokyo, Japan/location_tokyo.feather")
df = df.set_index('h3')
df[['hex_prefix', 'hex_suffix']] = df['hex_id'].str.split('_', n=1, expand=True)
df = df.drop(['neighbor_hexes', 'hex_id'], axis=1)

# 转为数值类型
df['hex_prefix'] = pd.to_numeric(df['hex_prefix'], errors='coerce')
df['hex_suffix'] = pd.to_numeric(df['hex_suffix'], errors='coerce')

# === 2. 计算角度 ===
num_per_level = np.maximum(1, 6 ** np.clip(df['hex_prefix'] , 0, None))
angle = 2 * np.pi * (df['hex_suffix'] ) / num_per_level

# === 3. 计算半径 r，并做 Z-score 标准化 ===
r = df['hex_prefix']  # 原始半径用层号表示
r_mean = r.mean()
r_std = r.std()
r_z = (r - r_mean) / r_std   # 标准化：均值为 0，方差为 1
df['r_z'] = r_z
# === 4. 计算 r·cosθ 与 r·sinθ ===
df['rcos'] = r_z * np.cos(angle)
df['rsin'] = r_z * np.sin(angle)

print(df[['hex_prefix', 'hex_suffix', 'rcos', 'rsin']].head())
df.to_csv("hex2vec-main/data/processed/Tokyo, Japan/location_tokyo_polar.csv", index=False)
