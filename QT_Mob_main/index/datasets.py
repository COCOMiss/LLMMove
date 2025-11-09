import numpy as np
import torch
import torch.utils.data as data
import pandas as pd


# class EmbDataset(data.Dataset):

#     def __init__(self,data_path):
        
#         # embeddings shape (len, dim) => (100, 4096)

#         self.data_path = data_path
#         self.embeddings = np.load(data_path)
        
     
#         self.dim = self.embeddings.shape[-1]

#     def __getitem__(self, index):
#         emb = self.embeddings[index]
#         tensor_emb=torch.FloatTensor(emb)
#         return tensor_emb

#     def __len__(self):
#         return len(self.embeddings)
    
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import ast

class EmbDataset(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        
        # 读取feather文件，然后设置h3作为索引
        self.df = pd.read_feather(data_path)
        self.df = self.df.set_index('h3')
        
        # 提取neighbor_hexes作为目标
        self.neighbor_hexes = self.df['neighbor_hexes']
        

        
        self.df[['hex_prefix', 'hex_suffix']] = self.df['hex_id'].str.split('_', n=1, expand=True)
        self.embeddings = self.df.drop(['neighbor_hexes','hex_id'], axis=1)
        # self.embeddings = self.df
        self.embeddings = self.embeddings.apply(lambda row: [float(x) for x in row.values], axis=1)
        
        
        # 将 pandas Series 转换为张量列表，然后堆叠
        embedding_tensors = [torch.tensor(x) for x in self.embeddings.values]
        self.embeddings = torch.stack(embedding_tensors, axis=0)

        # # 将neighbor_hexes字符串转换为实际的列表
        # self.neighbor_hexes = self.neighbor_hexes.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        self.dim = self.embeddings.shape[1]  # embedding的维度

    # def __getitem__(self, index):
    #     # 获取embedding特征
    #     emb_values = self.embeddings[index]
        
    #     # 获取对应的邻居列表 - 使用原始DataFrame的索引
    #     neighbors = self.neighbor_hexes.iloc[index]
        
    #     return emb_values, neighbors
    
    
    def __getitem__(self, index):
        # 获取embedding特征
        emb_values = self.embeddings[index]
        
        # 获取对应的邻居列表 - 使用原始DataFrame的索引
        # neighbors = self.neighbor_hexes.iloc[index]
        
        return emb_values, self.df.index[index]

    def __len__(self):
        return len(self.embeddings)
    
    def get_h3_id(self, index):
        """获取指定索引对应的h3 ID"""
        return self.df.index[index]
    
    def get_embedding_dim(self):
        """获取embedding的维度"""
        return self.dim
