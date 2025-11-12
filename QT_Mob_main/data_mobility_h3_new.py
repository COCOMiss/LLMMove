from math import log
import random
import os
import re
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from h3_prompt_mobility import *
import pandas as pd
import pickle
from tqdm import tqdm
from datetime import datetime
from logger_utils import get_logger
import gc
from functools import partial                                     
import multiprocessing as mp
from torch.utils.data import get_worker_info  # 用于检测是否在 DataLoader worker 内
import os, gc, json
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
logger = get_logger(__name__)
logger.info("==== Dataset module initialized ====")

EXPECTED_H3_PREFIXES = ("a", "b", "c", "d", "e", "f", "g")
H3_TOKEN_PATTERN = re.compile(r"<([a-g])_(\d+)>")
DURATION_TOKEN_PATTERN = re.compile(r"(\d+)\s*min", re.IGNORECASE)
ALLOWED_DURATION_MINUTES = {minutes for minutes in range(30, 601, 30)}

# # ---------- 无状态任务函数（可用于进程/线程池） ----------

# def _read_feather_task(fpath):
#     """读取单个 feather 文件并附加 trajectory_num 前缀。"""
#     base_name = os.path.splitext(os.path.basename(fpath))[0]
#     df = pd.read_feather(fpath)
#     df['trajectory_num'] = base_name + "_" + df['trajectory_num'].astype(str)
#     return base_name, df

# def _stay_sessions_for_day(day_time, trajectory_data):
#     """
#     复刻你当前 _process_stay_data() 的单日逻辑：
#     返回 (day_time, [traj_session, ...])；每个 traj_session 是若干 (h3, prev_time, user_id, traj_num, duration)
#     """
#     out_sessions = []

#     stay_data = trajectory_data[trajectory_data['transport_mode'] == 'STAY']
#     user_set = stay_data['user_id'].unique()

#     for user_id in user_set:
#         trajs = stay_data[stay_data['user_id'] == user_id]
#         trajs = trajs.sort_values(['trajectory_num', 'point_order'], ascending=True).reset_index()
#         traj_session = []

#         for idx, row in trajs.iterrows():
#             if idx % 2 == 0:
#                 prev_time = datetime.fromisoformat(row['time']) if isinstance(row['time'], str) else row['time'].to_pydatetime()
#                 prev_loc  = row['h3']
#                 prev_traj = row['trajectory_num']
#             else:
#                 if row['h3'] == prev_loc:
#                     cur_time = datetime.fromisoformat(row['time']) if isinstance(row['time'], str) else row['time'].to_pydatetime()
#                     duration = (cur_time - prev_time).total_seconds()
#                     if duration < 0:
#                         duration = -duration
#                     traj_session.append((str(row['h3']), prev_time, user_id, prev_traj, duration))
#                 else:
#                     # 与原逻辑一致：不同 h3 就跳过
#                     continue

#         if len(traj_session) >= 2:
#             out_sessions.append(traj_session)

#     return day_time, out_sessions

# def _remap_one_trajectory(codebook, trajectory):
#     """把一条 traj_session remap 成 index 串：返回 new_trajectory 列表。"""
#     # trajectory: [(h3, time, user_id, traj_id, duration), ...]
#     return [("".join(codebook[loc[0]]), loc[1], loc[0], loc[2], loc[3], loc[4]) for loc in trajectory]

# def _build_records_for_trajectory(traj, max_his_len, his_sep, add_prefix, add_profile, profile_map, mode, multi_seq):
#     """
#     按你现有 _process_data 的规则，把一条 remapped trajectory 生成若干训练样本 dict。
#     仅用到传入的参数，保持无状态（便于多进程）。
#     """
#     inter_data = []

#     # 划分 start/end 与原逻辑一致
#     if multi_seq and mode == "train":
#         start = 2
#         end = int(len(traj) * 0.7) if len(traj) > 2 else len(traj)
#     elif len(traj) >= 2 and mode == "valid":
#         start = int(len(traj) * 0.7)
#         end = int(len(traj) * 0.9)
#     elif len(traj) >= 2 and mode == "test":
#         start = int(len(traj) - 1)
#         end = len(traj)
#     else:
#         return inter_data  # 不满足长度

#     for i in range(start, end):
#         user_id = traj[i][3]
#         tstamp  = traj[i][1]
#         pred    = f"{traj[i][0]} for {traj[i][5]} seconds."

#         # 历史
#         history = traj[:i][-max_his_len:] if max_his_len > 0 else []
#         if max_his_len > 0:
#             history = [
#                 "At time " + str(item[1]) + ", user " + str(item[3]) + " stayed at H3 index " + item[0] + " for " + str(item[5]) + " seconds."
#                 for item in history
#             ]
#         if add_prefix:
#             history = [f"{k+1}. {txt}" for k, txt in enumerate(history)]

#         profile_txt = ""
#         if add_profile:
#             pr = profile_map.get(int(user_id))
#             if pr is not None:
#                 profile_txt = f"User {user_id}: {pr} "

#         one = {
#             "user": user_id,
#             "response": f"At time {tstamp}, user {user_id} will stay at h3 index ",
#             "prediction": pred,
#             "inters": his_sep.join(history),
#             "time": tstamp,
#             "profile": profile_txt,
#         }
#         inter_data.append(one)

#     return inter_data


# class BaseDataset(Dataset):

#     def __init__(self, args):
#         super().__init__()

#         self.args = args
#         self.data_path = args.data_path # 数据路径
#         self.max_his_len = args.max_his_len # 最大历史记录长度
#         self.his_sep = args.his_sep # The separator used for history
#         self.index_file = args.index_file
#         self.add_prefix = args.add_prefix # 是否加上序号
#         self.sft_json_output = args.sft_json_output # 是否输出json文件
#         self.indexing = args.indexing # 是否使用index表示, False表示使用(xxx,xxx)的location表示
#         self.new_tokens = None
#         self.allowed_tokens = None
#         self.all_items = None
#         self.task_prompt = None
#         self.data_filename_list = [f for f in os.listdir(self.data_path) if f.endswith(".feather")]
#         # self.data_filename = args.data_filename 
#         self.multi_seq = args.multi_seq
#         self.add_profile = args.add_profile
#         self.multi_rec = args.multi_rec
#         self.single_rec = args.single_rec
#         self.abalation_location_prompt = args.ablation_location_prompt
#         self.num_workers = getattr(args, "num_workers", 1)          # 并行度
#         self.parallel_backend = getattr(args, "parallel_backend", "thread")  # "thread" | "process"
#         logger.info(f"BaseDataset initialized with data path: {self.data_path}")
               
#     def _load_data(self):
#         raise NotImplementedError

#     def get_all_items(self):
        
#         # 返回所有item的index表示
#         if self.all_items is not None:
#             return self.all_items
#         self.all_items = set()
#         for index in self.codebook.values():
#             self.all_items.add("".join(index))
#         logger.info(f"Total unique items collected: {len(self.all_items)}")
#         return self.all_items
    
    
    
#     def get_prefix_allowed_tokens_fn(self, tokenizer, test_task):
#         # 返回一个函数，该函数返回当前token的allowed_tokens
#         if self.allowed_tokens is None:
#             self.allowed_tokens = {}
#             for index in self.codebook.values():
#                 self.token_len = len(index)
#                 len_of_token = len(tokenizer(index[0])["input_ids"]) - 1
#                 token_ids = [tokenizer(token)["input_ids"][len_of_token] for token in index]
#                 if token_ids[0] not in self.allowed_tokens.keys():
#                     self.allowed_tokens[token_ids[0]] = set()
#                 self.allowed_tokens[token_ids[0]].add(token_ids[1])
#                 for i in range(2, len(token_ids)):
#                     if tuple(token_ids[0:i]) not in self.allowed_tokens.keys():
#                         self.allowed_tokens[tuple(token_ids[0:i])] = set()
#                     self.allowed_tokens[tuple(token_ids[0:i])].add(token_ids[i])
#             for index in self.codebook.values():
#                 for i, token in enumerate(index):  # i表示token的位置,取值范围为0-n，token就n+1位
#                     token_id = tokenizer(token)["input_ids"][len_of_token]
#                     if i not in self.allowed_tokens.keys():
#                         self.allowed_tokens[i] = set()
#                     self.allowed_tokens[i].add(token_id)

#         # 根据test_task选择合适的分隔符
#         if test_task == "seq":
#             sep = tokenizer("will stay at h3 index ", add_special_tokens=False)["input_ids"][1:]
#         elif test_task == "recovery":
#             sep = tokenizer(" visited h3 index ", add_special_tokens=False)["input_ids"][1:]

#         def prefix_allowed_tokens_fn(batch_id, sentence):
#             sentence = sentence.tolist()
            
#             logger.info(f"prefix_allowed_tokens_fn sentence  {sentence}")
#             reversed_sent = sentence[::-1]

#             # 遍历逆向的句子，查找分隔符位置
#             for i in range(len(reversed_sent)):
#                 if reversed_sent[i:i + len(sep)] == sep[::-1]:
#                     # 如果在当前位置预测的是H3 index
#                     if i == self.token_len:
#                         logger.info(f"allowed_tokens i = self.token_len {tokenizer.eos_token_id}")
#                         return [tokenizer.eos_token_id]

#                     # 如果是第一个token位置，返回允许的H3 index相关tokens
#                     if i == 0 or reversed_sent[0] < 20:
#                         logger.info(f"allowed_tokens i = 0 {i}")
#                         return list(self.allowed_tokens[i])

#                     # 如果是duration位置，返回允许的duration tokens
#                     if i == 1:
#                         logger.info(f"allowed_tokens i = 1 {reversed_sent[0]}")
#                         return list(self.allowed_tokens[reversed_sent[0]])

#                     # 对于其他情况，检查当前H3 index的生成状态
#                     logger.info(f"allowed_tokens {tuple(reversed_sent[0:i][::-1])}")
#                     return list(self.allowed_tokens[tuple(reversed_sent[0:i][::-1])])

#             print("Warning: sep not found")
#             return []

#         return prefix_allowed_tokens_fn

 
#     # def get_prefix_allowed_tokens_fn(self, tokenizer, test_task):
#     #     # 返回一个函数，该函数返回当前token的allowed_tokens
#     #     if self.allowed_tokens is None:
#     #         self.allowed_tokens = {}
#     #         for index in self.codebook.values():
#     #             self.token_len = len(index)
#     #             len_of_token =  len(tokenizer(index[0])["input_ids"])-1
#     #             token_ids = [tokenizer(token)["input_ids"][len_of_token] for token in index]
#     #             if token_ids[0] not in self.allowed_tokens.keys():
#     #                 self.allowed_tokens[token_ids[0]] = set()
#     #             self.allowed_tokens[token_ids[0]].add(token_ids[1])
#     #             for i in range(2, len(token_ids)):
#     #                 if tuple(token_ids[0:i]) not in self.allowed_tokens.keys():
#     #                     self.allowed_tokens[tuple(token_ids[0:i])] = set()
#     #                 self.allowed_tokens[tuple(token_ids[0:i])].add(token_ids[i])
#     #         for index in self.codebook.values():
#     #             for i, token in enumerate(index): # i表示token的位置,取值范围为0-n，token就n+1位
#     #                 token_id = tokenizer(token)["input_ids"][len_of_token]
#     #                 if i not in self.allowed_tokens.keys():
#     #                     self.allowed_tokens[i] = set()
#     #                 self.allowed_tokens[i].add(token_id)
                
#     #     if test_task == "seq":
#     #         sep = tokenizer("will stay at h3 index ", add_special_tokens=False)["input_ids"][1:]
#     #     elif test_task == "recovery":
#     #         sep = tokenizer(" visited h3 index ", add_special_tokens=False)["input_ids"][1:]
        
#     #     def prefix_allowed_tokens_fn(batch_id, sentence):
#     #         sentence = sentence.tolist()
            
#     #         logger.info(f"prefix_allowed_tokens_fn sentence  {sentence}")
#     #         reversed_sent = sentence[::-1]
#     #         # print(tokenizer.decode(sentence))
#     #         for i in range(len(reversed_sent)):
#     #             if reversed_sent[i:i + len(sep)] == sep[::-1]:
#     #                 if i == self.token_len:
#     #                     logger.info(f"allowed_tokens i = self.token_len {tokenizer.eos_token_id}")
#     #                     return [tokenizer.eos_token_id]
#     #                 if i == 0 or reversed_sent[0]<20:
#     #                     logger.info(f"allowed_tokens i = 0 {i}")
#     #                     return list(self.allowed_tokens[i])
#     #                 if i == 1:
#     #                     logger.info(f"allowed_tokens i = 1 {reversed_sent[0]}")
#     #                     return list(self.allowed_tokens[reversed_sent[0]])
                    
#     #                 logger.info(f"allowed_tokens {tuple(reversed_sent[0:i][::-1])}")
#     #                 return list(self.allowed_tokens[tuple(reversed_sent[0:i][::-1])])
#     #         print("Warning: sep not found")

#     #     return prefix_allowed_tokens_fn

#     def _process_data(self):
#         raise NotImplementedError    
    
#     def set_prompt(self, prompt_id):
#         self.test_prompt_id = prompt_id
#         logger.info(f"Prompt ID set to {prompt_id}")

#     def __len__(self):
#         return len(self.inter_data)
    
#     def _get_text_data(self, data, prompt, sft_format=False):
#         if self.indexing:
#             sys_prompt = system_prompt
#         else:
#             # sys_prompt = system_prompt_not_indexing.format(max_poi=len(self.indices)-1)
#             sys_prompt = system_prompt_not_indexing
#         instruction = sys_prompt + self.task_prompt + prompt.format(**data)
#         response = data["response"]
#         prediction = data["prediction"] if "prediction" in data else ""

#         if self.mode == 'test':
#             input = sft_prompt.format(instruction = instruction, response = response, prediction = "")
#             return input, prediction
        
#         if sft_format:
#             input = sft_prompt.format(instruction = instruction, response = "", prediction = "")
#             output = sft_prompt.format(instruction = instruction, response = response, prediction = prediction)
#         else:
#             input = instruction
#             output = response + prediction
            
#         return input, output
    
#     def __getitem__(self, index):
#         d = self.inter_data[index]
#         if self.mode == 'test':
#             prompt_id = self.test_prompt_id # 测试时使用指定的prompt
#         else:
#             prompt_id = random.randint(0, len(self.prompts) - 1) # 随机选择一个prompt

#         prompt = self.prompts[prompt_id] # 获取prompt
#         input, output = self._get_text_data(d, prompt, not self.sft_json_output)
#         return dict(input_ids=input, labels=output)

    
#     def merge_data(self):
#         if self.inter_data_dict:
#             merged_data = pd.concat(list(self.inter_data_dict.values()), ignore_index=False)
#             logger.info(f"Merged {len(self.inter_data_dict)} dataframes.")
#         else:
#             merged_data = pd.DataFrame()
#             logger.warning("No data to merge.")
#         return merged_data

        
#     def load_multi_days_data(self):
#         all_data = {}
#         files = [os.path.join(self.data_path, fn) for fn in self.data_filename_list]
#         files = [f for f in files if os.path.exists(f)]

#         if not files:
#             logger.warning("No feather files found.")
#             return all_data

#         Executor = ThreadPoolExecutor if self.parallel_backend == "thread" else ProcessPoolExecutor

#         if self.num_workers > 1:
#             logger.info(f"Reading {len(files)} files with {self.parallel_backend} pool ({self.num_workers} workers)...")
#             with Executor(max_workers=self.num_workers) as ex:
#                 futures = {ex.submit(_read_feather_task, fpath): fpath for fpath in files}
#                 for fut in tqdm(as_completed(futures), total=len(futures), desc="Load feather"):
#                     try:
#                         base_name, df = fut.result()
#                         all_data[base_name] = df
#                         logger.info(f"Loaded {os.path.basename(futures[fut])} with {len(df)} records.")
#                     except Exception:
#                         logger.exception(f"Error reading file: {os.path.basename(futures[fut])}")
#         else:
#             for fpath in files:
#                 try:
#                     base_name, df = _read_feather_task(fpath)
#                     all_data[base_name] = df
#                     logger.info(f"Loaded file {os.path.basename(fpath)} with {len(df)} records.")
#                 except Exception:
#                     logger.exception(f"Error reading file: {fpath}")

#         return all_data


#         # if all_data:
#         #     merged_data = pd.concat(all_data, ignore_index=True)
#         # else:
#         #     merged_data = pd.DataFrame()
#     def _free_attrs(self, *names):
#         """将指定属性从内存中释放（置 None + 垃圾回收），并打印日志。"""
#         for n in names:
#             if hasattr(self, n) and getattr(self, n) is not None:
#                 try:
#                     obj = getattr(self, n)
#                     # 尽量清空容器，帮助释放
#                     if isinstance(obj, dict):
#                         obj.clear()
#                     elif hasattr(obj, "clear"):
#                         try:
#                             obj.clear()
#                         except Exception:
#                             pass
#                     setattr(self, n, None)
#                     logger.info(f"[MEM] Freed attribute: {n}")
#                 except Exception as e:
#                     logger.warning(f"[MEM] Free {n} failed: {e}")
#         gc.collect()
        

# class SeqDataset(BaseDataset):
#     # Task -- Next Location Prediction

#     def __init__(self, args, mode="train"):
#         super().__init__(args)

#         self.mode = mode # train, valid, test
        
#         self.prompts = all_prompt["seq"] # 所有的prompt
#         self.task_prompt = task_prompt
        
#         logger.info(f"Initializing SeqDataset (mode={self.mode})")
#         try:
#             self._load_data()
#             self.profile_map = dict(zip(self.user_profile['user_id'].astype(int), self.user_profile['prompt']))
#             self._remap_items()
#             self.inter_data = self._process_data()
#             logger.info(f"SeqDataset loaded successfully: {len(self.inter_data)} samples.")
#         except Exception:
#             logger.exception("SeqDataset initialization failed.")
#             raise
#         logger.info(f"SeqDataset data loaded ({len(self.inter_data)} STAY points).")



#     def _load_data(self):
#         # load data
        
#         logger.info("Loading data for SeqDataset...")       
#         self.inter_data_dict = self.load_multi_days_data()
        
#         self._process_stay_data()
#         self._free_attrs("inter_data_dict")
#         # 读取codebook文件
#         with open(os.path.join(self.data_path, self.index_file), 'r') as f:
#             self.codebook = json.load(f)
#         self.user_profile = pd.read_csv(
#             os.path.join(self.data_path, "user_profile_codebook.csv"),
#             converters={'latest_5_trips': eval}, sep="|"
#         )
#         # logger.info(f"SeqDataset data loaded ({len(self.inter_data)} STAY points).")
#     def _process_stay_data(self):
#         self.stay_data = {}
#         items = list(self.inter_data_dict.items())

#         Executor = ThreadPoolExecutor if self.parallel_backend == "thread" else ProcessPoolExecutor
#         if self.num_workers > 1:
#             logger.info(f"Extracting STAY sessions with {self.parallel_backend} pool ({self.num_workers} workers)...")
#             with Executor(max_workers=self.num_workers) as ex:
#                 futures = {ex.submit(_stay_sessions_for_day, day_time, df): day_time for day_time, df in items}
#                 for fut in tqdm(as_completed(futures), total=len(futures), desc="STAY sessions"):
#                     try:
#                         day_time, sess = fut.result()
#                         self.stay_data[day_time] = sess
#                     except Exception:
#                         logger.exception("Error while building STAY sessions")
#         else:
#             for day_time, df in tqdm(items, desc="STAY sessions (serial)"):
#                 day_time, sess = _stay_sessions_for_day(day_time, df)
#                 self.stay_data[day_time] = sess

#     def _remap_items(self):
#         self.remapped_inters = []

#         # 把 dict of list 展平成列表
#         all_trajs = [traj for day_list in self.stay_data.values() for traj in day_list]

#         Executor = ThreadPoolExecutor if self.parallel_backend == "thread" else ProcessPoolExecutor
#         if self.num_workers > 1:
#             logger.info(f"Remapping trajectories with {self.parallel_backend} pool ({self.num_workers} workers)...")
#             with Executor(max_workers=self.num_workers) as ex:
#                 futures = {ex.submit(_remap_one_trajectory, self.codebook, traj): i for i, traj in enumerate(all_trajs)}
#                 for fut in tqdm(as_completed(futures), total=len(futures), desc="Remap"):
#                     try:
#                         new_traj = fut.result()
#                         self.remapped_inters.append(new_traj)
#                     except Exception:
#                         logger.exception("Error in remap task")
#         else:
#             for traj in tqdm(all_trajs, desc="Remap (serial)"):
#                 self.remapped_inters.append(_remap_one_trajectory(self.codebook, traj))

#         logger.info(f"Remapping complete: {len(self.remapped_inters)} trajectories in SeqDataset·····.")
#         self._free_attrs("stay_data")


    
#     def _process_data(self):
#         logger.info("Processing SeqDataset trajectories...")
#         inter_data = []

#         Executor = ThreadPoolExecutor if self.parallel_backend == "thread" else ProcessPoolExecutor
#         args_tuple = (self.max_his_len, self.his_sep, self.add_prefix, self.add_profile, self.profile_map, self.mode, self.multi_seq)

#         if self.num_workers > 1:
#             logger.info(f"Building records with {self.parallel_backend} pool ({self.num_workers} workers)...")
#             with Executor(max_workers=self.num_workers) as ex:
#                 futures = {
#                     ex.submit(_build_records_for_trajectory, traj, *args_tuple): i
#                     for i, traj in enumerate(self.remapped_inters)
#                 }
#                 for fut in tqdm(as_completed(futures), total=len(futures), desc="Build records"):
#                     try:
#                         inter_data.extend(fut.result())
#                     except Exception:
#                         logger.exception("Error building records from one trajectory")
#         else:
#             for traj in tqdm(self.remapped_inters, desc="Build records (serial)"):
#                 inter_data.extend(_build_records_for_trajectory(traj, *args_tuple))

#         logger.info(f"SeqDataset processing complete: {len(inter_data)} records.")
#         self._free_attrs("remapped_inters", "user_profile", "profile_map")
#         return inter_data



class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.data_path = args.data_path # 数据路径
        self.max_his_len = args.max_his_len # 最大历史记录长度
        self.his_sep = args.his_sep # The separator used for history
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix # 是否加上序号
        self.sft_json_output = args.sft_json_output # 是否输出json文件
        self.indexing = args.indexing # 是否使用index表示, False表示使用(xxx,xxx)的location表示
        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None
        self.task_prompt = None
        # self.data_filename_list = [f for f in os.listdir(self.data_path) if f.endswith(".feather")]
        # self.data_filename = args.data_filename 
        self.multi_seq = args.multi_seq
        self.add_profile = args.add_profile
        self.multi_rec = args.multi_rec
        self.single_rec = args.single_rec
        self.abalation_location_prompt = args.ablation_location_prompt
        logger.info(f"BaseDataset initialized with data path: {self.data_path}")
               
    def _load_data(self):
        raise NotImplementedError

    def get_all_items(self):
        
        # 返回所有item的index表示
        if self.all_items is not None:
            return self.all_items
        self.all_items = set()
        for index in self.codebook.values():
            self.all_items.add("".join(index))
        logger.info(f"Total unique items collected: {len(self.all_items)}")
        return self.all_items
    
    
    
    def get_prefix_allowed_tokens_fn(self, tokenizer, test_task):
        # 返回一个函数，该函数返回当前token的allowed_tokens
        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.codebook.values():
                self.token_len = len(index)
                len_of_token = len(tokenizer(index[0])["input_ids"]) - 1
                token_ids = [tokenizer(token)["input_ids"][len_of_token] for token in index]
                if token_ids[0] not in self.allowed_tokens.keys():
                    self.allowed_tokens[token_ids[0]] = set()
                self.allowed_tokens[token_ids[0]].add(token_ids[1])
                for i in range(2, len(token_ids)):
                    if tuple(token_ids[0:i]) not in self.allowed_tokens.keys():
                        self.allowed_tokens[tuple(token_ids[0:i])] = set()
                    self.allowed_tokens[tuple(token_ids[0:i])].add(token_ids[i])
            for index in self.codebook.values():
                for i, token in enumerate(index):  # i表示token的位置,取值范围为0-n，token就n+1位
                    token_id = tokenizer(token)["input_ids"][len_of_token]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)

        # 根据test_task选择合适的分隔符
        if test_task == "seq":
            sep = tokenizer("will stay at h3 index ", add_special_tokens=False)["input_ids"][1:]
        elif test_task == "recovery":
            sep = tokenizer(" visited h3 index ", add_special_tokens=False)["input_ids"][1:]

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            
            logger.info(f"prefix_allowed_tokens_fn sentence  {sentence}")
            reversed_sent = sentence[::-1]

            # 遍历逆向的句子，查找分隔符位置
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # 如果在当前位置预测的是H3 index
                    if i == self.token_len:
                        logger.info(f"allowed_tokens i = self.token_len {tokenizer.eos_token_id}")
                        return [tokenizer.eos_token_id]

                    # 如果是第一个token位置，返回允许的H3 index相关tokens
                    if i == 0 or reversed_sent[0] < 20:
                        logger.info(f"allowed_tokens i = 0 {i}")
                        return list(self.allowed_tokens[i])

                    # 如果是duration位置，返回允许的duration tokens
                    if i == 1:
                        logger.info(f"allowed_tokens i = 1 {reversed_sent[0]}")
                        return list(self.allowed_tokens[reversed_sent[0]])

                    # 对于其他情况，检查当前H3 index的生成状态
                    logger.info(f"allowed_tokens {tuple(reversed_sent[0:i][::-1])}")
                    return list(self.allowed_tokens[tuple(reversed_sent[0:i][::-1])])

            print("Warning: sep not found")
            return []

        return prefix_allowed_tokens_fn

    def validate_output_constraints(self, output, raise_on_error=False):
        """
        校验大模型生成的文本是否满足 H3 index 及 duration 的输出格式要求。

        参数:
            output: 可以是字符串、序列或字典，函数会自动转换为字符串后再进行校验。
            raise_on_error: 如果为 True，且存在校验错误，则直接抛出 ValueError。

        返回:
            dict，包含以下字段:
                - valid: bool，是否全部校验通过
                - errors: list[str]，校验失败信息
                - h3_indices: list[str]，按照出现顺序提取出的 H3 index 片段
                - durations: list[str]，合法的 duration 片段（统一格式为 "xxx min"）
        """
        if isinstance(output, str):
            text = output
        elif isinstance(output, (list, tuple, set)):
            text = " ".join(map(str, output))
        elif isinstance(output, dict):
            try:
                text = json.dumps(output, ensure_ascii=False)
            except TypeError:
                text = str(output)
        else:
            text = str(output)

        errors = []

        # —— 校验 H3 index —— #
        h3_matches = list(H3_TOKEN_PATTERN.finditer(text))
        h3_tokens = [match.group(0) for match in h3_matches]
        h3_prefixes = [match.group(1) for match in h3_matches]
        ordered_h3_tokens = []

        if not h3_tokens:
            errors.append("未检测到符合 `<a_123>` 形式的 H3 index。")
        else:
            missing_prefixes = [prefix for prefix in EXPECTED_H3_PREFIXES if prefix not in h3_prefixes]
            if missing_prefixes:
                errors.append(
                    "H3 index 需要依次包含 <a_*>, <b_*>, ..., <g_*> 形式的 7 个片段，缺失: "
                    + ", ".join(f"<{p}_*>" for p in missing_prefixes)
                )
            else:
                first_positions = [h3_prefixes.index(prefix) for prefix in EXPECTED_H3_PREFIXES]
                if first_positions != sorted(first_positions):
                    errors.append("H3 index 片段的顺序需要依次为 a→b→c→d→e→f→g。")
                else:
                    first_seen = {}
                    for match in h3_matches:
                        prefix = match.group(1)
                        if prefix not in first_seen:
                            first_seen[prefix] = match.group(0)
                    ordered_h3_tokens = [first_seen[prefix] for prefix in EXPECTED_H3_PREFIXES]

        # —— 校验 duration —— #
        duration_matches = [int(match.group(1)) for match in DURATION_TOKEN_PATTERN.finditer(text)]
        duration_tokens = []
        if not duration_matches:
            errors.append("未检测到任何符合 `xx min` 形式的 duration。")
        else:
            for minutes in duration_matches:
                if minutes not in ALLOWED_DURATION_MINUTES:
                    errors.append(
                        f"duration `{minutes} min` 不在允许范围 (30min-600min，并以 30min 为步长) 内。"
                    )
                else:
                    duration_tokens.append(f"{minutes} min")

        valid = len(errors) == 0
        if raise_on_error and not valid:
            raise ValueError("; ".join(errors))

        return {
            "valid": valid,
            "errors": errors,
            "h3_indices": ordered_h3_tokens or h3_tokens,
            "durations": duration_tokens,
        }

 
    # def get_prefix_allowed_tokens_fn(self, tokenizer, test_task):
    #     # 返回一个函数，该函数返回当前token的allowed_tokens
    #     if self.allowed_tokens is None:
    #         self.allowed_tokens = {}
    #         for index in self.codebook.values():
    #             self.token_len = len(index)
    #             len_of_token =  len(tokenizer(index[0])["input_ids"])-1
    #             token_ids = [tokenizer(token)["input_ids"][len_of_token] for token in index]
    #             if token_ids[0] not in self.allowed_tokens.keys():
    #                 self.allowed_tokens[token_ids[0]] = set()
    #             self.allowed_tokens[token_ids[0]].add(token_ids[1])
    #             for i in range(2, len(token_ids)):
    #                 if tuple(token_ids[0:i]) not in self.allowed_tokens.keys():
    #                     self.allowed_tokens[tuple(token_ids[0:i])] = set()
    #                 self.allowed_tokens[tuple(token_ids[0:i])].add(token_ids[i])
    #         for index in self.codebook.values():
    #             for i, token in enumerate(index): # i表示token的位置,取值范围为0-n，token就n+1位
    #                 token_id = tokenizer(token)["input_ids"][len_of_token]
    #                 if i not in self.allowed_tokens.keys():
    #                     self.allowed_tokens[i] = set()
    #                 self.allowed_tokens[i].add(token_id)
                
    #     if test_task == "seq":
    #         sep = tokenizer("will stay at h3 index ", add_special_tokens=False)["input_ids"][1:]
    #     elif test_task == "recovery":
    #         sep = tokenizer(" visited h3 index ", add_special_tokens=False)["input_ids"][1:]
        
    #     def prefix_allowed_tokens_fn(batch_id, sentence):
    #         sentence = sentence.tolist()
            
    #         logger.info(f"prefix_allowed_tokens_fn sentence  {sentence}")
    #         reversed_sent = sentence[::-1]
    #         # print(tokenizer.decode(sentence))
    #         for i in range(len(reversed_sent)):
    #             if reversed_sent[i:i + len(sep)] == sep[::-1]:
    #                 if i == self.token_len:
    #                     logger.info(f"allowed_tokens i = self.token_len {tokenizer.eos_token_id}")
    #                     return [tokenizer.eos_token_id]
    #                 if i == 0 or reversed_sent[0]<20:
    #                     logger.info(f"allowed_tokens i = 0 {i}")
    #                     return list(self.allowed_tokens[i])
    #                 if i == 1:
    #                     logger.info(f"allowed_tokens i = 1 {reversed_sent[0]}")
    #                     return list(self.allowed_tokens[reversed_sent[0]])
                    
    #                 logger.info(f"allowed_tokens {tuple(reversed_sent[0:i][::-1])}")
    #                 return list(self.allowed_tokens[tuple(reversed_sent[0:i][::-1])])
    #         print("Warning: sep not found")

    #     return prefix_allowed_tokens_fn

    def _process_data(self):
        raise NotImplementedError    
    
    def get_json_prefix_allowed_tokens_fn(self, tokenizer, task="seq"):
        """
        Constrain generation for JSON outputs:
        - "h3_index": value must be a 4-token H3 index from codebook (e.g., "<a_244><b_61><c_35><d_14>")
        - "stay_duration": value must be one of {"30min","60min",...,"600min"} (30-minute steps, up to 10 hours)
        
        This function returns a callable suitable for use as prefix_allowed_tokens_fn in generation APIs.
        It detects whether the model is currently inside the JSON value for "h3_index" or "stay_duration"
        by matching the tokenized anchors '"h3_index": "' and '"stay_duration": "' and then restricts the
        next tokens accordingly.
        """
        # Tokenization cache to avoid repeated tokenizer calls
        if not hasattr(self, "_tok_cache"):
            self._tok_cache = {}
        tokcache = self._tok_cache
        def tok_ids(txt: str):
            r = tokcache.get(txt)
            if r is None:
                r = tokenizer(txt, add_special_tokens=False)["input_ids"]
                tokcache[txt] = r
            return r
        
        # Build constraints once
        if not hasattr(self, "_json_constraints_built") or not self._json_constraints_built:
            # 1) H3 index transitions from codebook
            # Map first-token -> second-token set; tuple(prefix) -> next-token set
            self._h3_token_len = None
            self._h3_allowed_pos0 = set()            # allowed first token ids at position 0
            self._h3_allowed_by_prefix = {}          # key: tuple(token_ids_prefix) or int 0 -> set(next_ids)
            
            # Determine final-piece id extraction offset by tokenizing the first token once
            # We use the "last id" approach consistent with the existing implementation.
            def last_id(token_str: str) -> int:
                ids = tok_ids(token_str)
                # Use the last token id in the tokenized piece to represent this token
                return ids[-1]
            
            for index in self.codebook.values():
                if self._h3_token_len is None:
                    self._h3_token_len = len(index)
                token_ids_seq = [last_id(t) for t in index]
                # allowed start tokens (position 0)
                self._h3_allowed_pos0.add(token_ids_seq[0])
                self._h3_allowed_by_prefix.setdefault(0, set()).add(token_ids_seq[0])
                # transitions
                if len(token_ids_seq) >= 2:
                    self._h3_allowed_by_prefix.setdefault((token_ids_seq[0],), set()).add(token_ids_seq[1])
                for i in range(2, len(token_ids_seq)):
                    prefix = tuple(token_ids_seq[:i])
                    self._h3_allowed_by_prefix.setdefault(prefix, set()).add(token_ids_seq[i])
            
            # 2) Duration allowed sequences: "30min" .. "600min" step 30
            self._dur_values = [f"{m}min" for m in range(30, 601, 30)]
            self._dur_allowed_pos0 = set()
            self._dur_allowed_by_prefix = {}          # tuple(prefix_ids) or int 0 -> set(next_ids)
            self._dur_token_seqs = []
            for s in self._dur_values:
                ids = tok_ids(s)
                if not ids:
                    continue
                self._dur_token_seqs.append(ids)
                self._dur_allowed_pos0.add(ids[0])
                self._dur_allowed_by_prefix.setdefault(0, set()).add(ids[0])
                for i in range(1, len(ids)):
                    prefix = tuple(ids[:i])
                    self._dur_allowed_by_prefix.setdefault(prefix, set()).add(ids[i])
            
            # 3) Anchors to detect which JSON field we are in
            self._sep_h3 = tok_ids('"h3_index": "')
            self._sep_dur = tok_ids('"stay_duration": "')
            # Token for closing quote
            q = tok_ids('"')
            self._quote_token = q[0] if len(q) > 0 else None
            
            self._json_constraints_built = True
        
        # Helper to find the nearest anchor in reversed tokens
        def find_anchor(reversed_tokens, sep):
            n = len(sep)
            if n == 0:
                return None
            sep_rev = sep[::-1]
            for i in range(len(reversed_tokens) - n + 1):
                if reversed_tokens[i:i+n] == sep_rev:
                    return i
            return None
        
        def prefix_allowed_tokens_fn(batch_id, sentence_tensor):
            sentence = sentence_tensor.tolist()
            reversed_sent = sentence[::-1]
            
            i_h3 = find_anchor(reversed_sent, self._sep_h3)
            i_dur = find_anchor(reversed_sent, self._sep_dur)
            
            # Choose the nearest anchor (smaller i means closer to the end)
            chosen = None
            if i_h3 is not None and (i_dur is None or i_h3 < i_dur):
                chosen = ("h3", i_h3)
            elif i_dur is not None:
                chosen = ("dur", i_dur)
            
            if chosen is None:
                # Not inside a constrained JSON value; do not restrict
                return []
            
            field, offset = chosen
            # prefix tokens (forward order) generated for the field value so far
            value_prefix = tuple(reversed_sent[:offset][::-1])
            
            if field == "h3":
                # If we have already produced 4 H3 tokens, force closing quote (if available)
                if len(value_prefix) >= (self._h3_token_len or 4):
                    if self._quote_token is not None:
                        return [self._quote_token]
                    return []
                if len(value_prefix) == 0:
                    return list(self._h3_allowed_by_prefix.get(0, set()))
                # Use the exact prefix if possible; otherwise, backoff to last token or start set
                allowed = self._h3_allowed_by_prefix.get(value_prefix)
                if allowed is None:
                    if len(value_prefix) >= 1:
                        allowed = self._h3_allowed_by_prefix.get((value_prefix[0],))
                if allowed is None:
                    allowed = self._h3_allowed_by_prefix.get(0, set())
                return list(allowed)
            
            # field == "dur"
            # Duration is a string; after finishing the value, require closing quote
            # Find which duration prefix we are on
            if len(value_prefix) == 0:
                return list(self._dur_allowed_by_prefix.get(0, set()))
            # If value_prefix matches a full allowed sequence, next should be closing quote
            if any(tuple(seq) == value_prefix for seq in self._dur_token_seqs):
                if self._quote_token is not None:
                    return [self._quote_token]
                return []
            # Otherwise continue along any matching prefix path
            allowed = self._dur_allowed_by_prefix.get(value_prefix)
            if allowed is None and len(value_prefix) >= 1:
                allowed = self._dur_allowed_by_prefix.get((value_prefix[0],))
            if allowed is None:
                allowed = self._dur_allowed_by_prefix.get(0, set())
            return list(allowed)
        
        return prefix_allowed_tokens_fn
    
    def set_prompt(self, prompt_id):
        self.test_prompt_id = prompt_id
        logger.info(f"Prompt ID set to {prompt_id}")

    def __len__(self):
        return len(self.inter_data)
    
    def _get_text_data(self, data, prompt, sft_format=False):
        if self.indexing:
            sys_prompt = system_prompt
        else:
            # sys_prompt = system_prompt_not_indexing.format(max_poi=len(self.indices)-1)
            sys_prompt = system_prompt_not_indexing
        instruction = sys_prompt + self.task_prompt + prompt.format(**data)
        response = data["response"]
        prediction = data["prediction"] if "prediction" in data else ""

        if self.mode == 'test':
            input = sft_prompt.format(instruction = instruction, response = response, prediction = "")
            return input, prediction
        
        if sft_format:
            input = sft_prompt.format(instruction = instruction, response = "", prediction = "")
            output = sft_prompt.format(instruction = instruction, response = response, prediction = prediction)
        else:
            input = instruction
            output = response + prediction
            
        return input, output
    
    def __getitem__(self, index):
        d = self.inter_data[index]
        if self.mode == 'test':
            prompt_id = self.test_prompt_id # 测试时使用指定的prompt
        else:
            prompt_id = random.randint(0, len(self.prompts) - 1) # 随机选择一个prompt

        prompt = self.prompts[prompt_id] # 获取prompt
        input, output = self._get_text_data(d, prompt, not self.sft_json_output)
        return dict(input_ids=input, labels=output)

    
    def merge_data(self):
        if self.inter_data_dict:
            merged_data = pd.concat(list(self.inter_data_dict.values()), ignore_index=False)
            logger.info(f"Merged {len(self.inter_data_dict)} dataframes.")
        else:
            merged_data = pd.DataFrame()
            logger.warning("No data to merge.")
        return merged_data

        
    def load_multi_days_data(self):
        # 读取所有 self.data_filename_list 中的文件，合并 data
        all_data = {}
        for file_name in self.data_filename_list:
            fpath = os.path.join(self.data_path, file_name)
            if os.path.exists(fpath):
                try:
                    df = pd.read_feather(fpath)
                    base_name = os.path.splitext(os.path.basename(file_name))[0]
                    df['trajectory_num'] = base_name + "_" + df['trajectory_num'].astype(str)
                    all_data[base_name] = df
                    logger.info(f"Loaded file {file_name} with {len(df)} records.")
                except Exception as e:
                    logger.exception(f"Error reading file: {fpath}")
            else:
                logger.warning(f"File not found: {fpath}")
        return all_data

        # if all_data:
        #     merged_data = pd.concat(all_data, ignore_index=True)
        # else:
        #     merged_data = pd.DataFrame()
    def _free_attrs(self, *names):
        """将指定属性从内存中释放（置 None + 垃圾回收），并打印日志。"""
        for n in names:
            if hasattr(self, n) and getattr(self, n) is not None:
                try:
                    obj = getattr(self, n)
                    # 尽量清空容器，帮助释放
                    if isinstance(obj, dict):
                        obj.clear()
                    elif hasattr(obj, "clear"):
                        try:
                            obj.clear()
                        except Exception:
                            pass
                    setattr(self, n, None)
                    logger.info(f"[MEM] Freed attribute: {n}")
                except Exception as e:
                    logger.warning(f"[MEM] Free {n} failed: {e}")
        gc.collect()
        

class SeqDataset(BaseDataset):
    # Task -- Next Location Prediction

    def __init__(self, args, mode="train"):
        super().__init__(args)

        self.mode = mode # train, valid, test
        
        self.prompts = all_prompt["seq"] # 所有的prompt
        self.task_prompt = task_prompt
        with open("LLMMove/QT_Mob_main/dataset/location.index.json", 'r') as f:
            self.codebook = json.load(f)
        # self.user_profile = pd.read_csv(
        #     os.path.join(self.data_path, "user_profile_codebook.csv"),
        #     converters={'latest_5_trips': eval}, sep="|"
        # )
        logger.info(f"Initializing SeqDataset (mode={self.mode})")
        try:
            if self.mode=="valid":
                # self._load_data()
                # self._remap_items()
                # self.inter_data = self._process_data()
                self.inter_data=pd.read_feather("LLMMove/QT_Mob_main/dataset/valid/inner_data_seq_dataset.feather")
                self.inter_data=self.inter_data.to_dict(orient="records")
            if self.mode == "train":
                self.inter_data=pd.read_feather("LLMMove/QT_Mob_main/dataset/train/inner_data_seq_dataset.feather")
                self.inter_data=self.inter_data.to_dict(orient="records")
            #     pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/train/inner_data_seq_dataset.feather")
            if self.mode=="test":
                self.inter_data=pd.read_feather("QT_Mob_main/dataset/test/inner_data_seq_dataset.feather")
                self.inter_data=self.inter_data.to_dict(orient="records")                
            #     pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/test/inner_data_seq_dataset.feather")
            logger.info(f"SeqDataset loaded successfully: {len(self.inter_data)} samples.")
        except Exception:
            logger.exception("SeqDataset initialization failed.")
            raise
        logger.info(f"SeqDataset data loaded ({len(self.inter_data)} STAY points).")

    def get_stay_duration(self, duration: float) -> int:
        """
        Round stay duration (in seconds) to nearest 30-minute bucket and clamp to [30, 600] minutes.
        Returns integer minutes.
        """
        total_minutes = int(round(duration / 60.0))
        bucket_minutes = int(round(total_minutes / 30.0) * 30)
        if bucket_minutes < 30:
            bucket_minutes = 30
        if bucket_minutes > 600:
            bucket_minutes = 600
        return bucket_minutes



    def _load_data(self):
        # load data
        
        logger.info("Loading data for SeqDataset...")       
        self.inter_data_dict = self.load_multi_days_data()
        
        self._process_stay_data()
        self._free_attrs("inter_data_dict")
        # 读取codebook文件
        with open(os.path.join(self.data_path, self.index_file), 'r') as f:
            self.codebook = json.load(f)
        self.user_profile = pd.read_csv(
            os.path.join(self.data_path, "user_profile_codebook.csv"),
            converters={'latest_5_trips': eval}, sep="|"
        )
        # logger.info(f"SeqDataset data loaded ({len(self.inter_data)} STAY points).")
    def _process_stay_data(self):
        self.stay_data={}
        
        for day_time, trajectory_data in self.inter_data_dict.items():
            if day_time not in self.stay_data.keys():
                self.stay_data[day_time]=[]
            stay_data = trajectory_data[trajectory_data['transport_mode'] == 'STAY']
            user_set = stay_data['user_id'].unique()
            
            for user_id in tqdm(user_set, desc="Processing STAY DATA"):
                trajs = stay_data[stay_data['user_id'] == user_id]
                trajs = trajs.sort_values(['trajectory_num', 'point_order'], ascending=True)
                trajs = trajs.reset_index()
                traj_session=[]
                # traj_nums = trajs['trajectory_num'].unique()
                for index, row in trajs.iterrows(): 
                    if index%2 ==0:
                        if type(row['time']) is str:
                            prev_time = datetime.fromisoformat(row['time'])
                        else:
                            prev_time = row['time'].to_pydatetime()
                        prev_loc= row['h3']
                    else:
                        if row['h3']==prev_loc:
                            if type(row['time']) is str:
                                duration = (datetime.fromisoformat(row['time'])-prev_time).total_seconds()
                            else:
                                duration = (row['time'].to_pydatetime()-prev_time).total_seconds()
                            
                            if duration < 0:
                                duration = -duration
                            traj_session.append((str(row['h3']), prev_time, user_id, row['trajectory_num'],duration))
                        else:
                            continue
                 
                if len(traj_session) >= 2:
                    self.stay_data[day_time].append(traj_session)
        
        
    def _remap_items(self):
           
        #源代码 loc[0],loc[1] -> lat,lon; loc[2]-> time ; loc[3] -> user_id;  loc[4] -> traj_id
        #现代码 loc[0] -> h3 ; loc[1]-> time ; loc[2] -> user_id;  loc[3] -> traj_id
        # item转换成index表示
        self.remapped_inters = []
        for date, day_trajectory in self.stay_data.items():

            for trajectory in day_trajectory:
                new_trajectory = [("".join(self.codebook[loc[0]]) ,loc[1],loc[0],loc[2],loc[3],loc[4]) for loc  in trajectory]
                # new_trajectory = [("".join(self.indices[str(self.loc2id[(loc[0],loc[1])])]),loc[2],loc[0],loc[1],loc[3],loc[4]) for loc in trajectory]
                self.remapped_inters.append(new_trajectory)
        logger.info(f"Remapping complete: {len(self.remapped_inters)} trajectories in SeqDataset·····.")    
        self._free_attrs("stay_data")

    
    def _process_data(self):
        logger.info("Processing SeqDataset trajectories...")       
        inter_data = []
        for trajectory in tqdm(self.remapped_inters):
            if self.multi_seq and self.mode == "train":
                start = 2
                if len(trajectory)>2 :
                    end = int(len(trajectory)*0.7)
                else: 
                    end = len(trajectory)
            elif len(trajectory)>=2 and self.mode == "valid":
                start = int(len(trajectory)*0.7)
                end = int(len(trajectory)*0.9)
            elif len(trajectory)>=2 and self.mode == "test":
                start = int(len(trajectory)-1)
                end = len(trajectory)
            
            for i in range(start, end):
                try:
                    one_data = dict()
                    one_data["user"] = trajectory[i][3]
                    # JSON output: assistant response tag + JSON prediction
                    one_data["response"] = "prediction:"
                    one_data["prediction"] = json.dumps(
                        {
                            "h3_index": trajectory[i][0],
                            "stay_duration": f"{self.get_stay_duration(trajectory[i][5])}min",
                        },
                        ensure_ascii=False,
                    )
                    # one_data['duration'] = trajectory[i][5]
                    history = trajectory[:i][-self.max_his_len:]
                    
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]# 只保留最近的max_his_len个历史记录
                        history = [
                            "At time " + str(item_idx[1]) + ", user " + str(item_idx[3]) + " stayed at H3 index " + item_idx[0] + " for " + self.get_stay_duration(trajectory[i][5]) + " min."
                            for item_idx in history
                        ]
                        
                        
                  
                        
                        # history = ["At time " + str(item_idx[1]) + ", user " + str(item_idx[3]) + " visited h3 index " + item_idx[0] + "." for item_idx in history]
                    if self.add_prefix:
                        history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)] # 添加序号前缀 1. item1 
                    one_data["inters"] = self.his_sep.join(history)
                    one_data["time"] = trajectory[i][1]
                    if self.add_profile:
                        profile = self.user_profile.loc[self.user_profile['user_id'] == int(trajectory[i][3])]
                        one_data["profile"] = f"User {trajectory[i][3]}: {profile['prompt'].values[0]} " if not profile.empty else ""
                    else:
                        one_data["profile"] = ""
                    inter_data.append(one_data)
                except Exception:
                    logger.exception("Error processing a trajectory sample.")
        logger.info(f"SeqDataset processing complete: {len(inter_data)} records.")
        self._free_attrs("remapped_inters", "user_profile")
        return inter_data



class RecoveryDataset(BaseDataset):
    # Task -- Trajectory Recovery --10 Prompt
    # 有训练集，验证集和测试集

    def __init__(self, args, mode="train"):
        super().__init__(args)

        self.mode = mode # train, valid, test
        
        self.prompts = all_prompt["recovery"] # 所有的prompt
        self.task_prompt = task_prompt
        logger.info(f"Initializing RecoveryDataset (mode={self.mode})")       

        try:
            self._load_data()
            self._remap_items()
            self.inter_data = self._process_data()
            if self.mode == "train":
                pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/train/inner_data_rec_dataset.feather")
            if self.mode=="test":
                pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/test/inner_data_rec_dataset.feather")            
            logger.info(f"RecoveryDataset loaded successfully: {len(self.inter_data)} samples.")
        except Exception:
            logger.exception("RecoveryDataset initialization failed.")
            raise



    def _load_data(self):
        logger.info("Loading data for RecoveryDataset...")
       
        # path = os.path.join(self.data_path, self.data_filename)
        self.inter_data_dict = self.load_multi_days_data()
        self.inter_data = self.merge_data()
        self._free_attrs("inter_data_dict")
                
        
        # self.inter_data = self.inter_data[self.inter_data["transport_mode"] == "STAY"]
        
        with open(os.path.join(self.data_path, self.index_file), "r") as f:
            self.codebook = json.load(f) 
            
        self.user_profile = pd.read_csv(os.path.join(self.data_path, "user_profile_codebook.csv"),
                                        converters={'latest_5_trips': eval}, sep="|")            
        # # 读取index文件
        # with open(os.path.join(self.data_path, self.index_file), 'r') as f:
        #     self.indices = json.load(f)
        # if not self.indexing:
        #     self.indices = {k: [f"{k}"] for k in self.indices.keys()}
            
        # with open(os.path.join(self.data_path, "loc2id"), 'rb') as file:
        #     self.loc2id = pickle.load(file)
        
        # self.user_profile = pd.read_csv(os.path.join(self.data_path, "user_profile.csv"), converters={'latest_5_trips': eval},sep="|")



    def _remap_items(self):
        all_trajectory = []
        user_set = self.inter_data['user_id'].unique()
        logger.info(f"Remapping {len(user_set)} users in RecoveryDataset...")
        for user_id in tqdm(user_set, desc="Remapping RecoveryDataset"):
            try:
                trajs = self.inter_data[self.inter_data['user_id'] == user_id]
                trajs = trajs.sort_values(['trajectory_num', 'point_order'], ascending=True)
                for traj_id in trajs['trajectory_num'].unique():
                    traj_session = []
                    pev = datetime.fromisoformat("1000-01-01 00:00:00+09:00")
                    for _, row in trajs[trajs['trajectory_num'] == traj_id].iterrows():
                        if type(row['time']) is str:
                            stamp = datetime.fromisoformat(row['time'])
                        else:
                            stamp = row['time'].to_pydatetime()
                        if (stamp - pev).total_seconds() > 180:
                            traj_session.append((str(row['h3']), row['time'], user_id, traj_id))
                            pev = stamp
                    if len(traj_session) >= 2:
                        all_trajectory.append(traj_session)
            except Exception:
                logger.exception(f"Error remapping user {user_id}")   
        #源代码 loc[0],loc[1] -> lat,lon; loc[2]-> time ; loc[3] -> user_id;  loc[4] -> traj_id
        #现代码 loc[0] -> h3 ; loc[1]-> time ; loc[2] -> user_id;  loc[3] -> traj_id
        # item转换成index表示
        self._free_attrs("inter_data")
        self.remapped_inters = []
        for trajectory in all_trajectory:
            try:
                new_trajectory = [("".join(self.codebook[loc[0]]) ,str(loc[1]),loc[0],loc[2],str(loc[3])) for loc  in trajectory]
                # new_trajectory = [("".join(self.indices[str(self.loc2id[(loc[0],loc[1])])]),loc[2],loc[0],loc[1],loc[3],loc[4]) for loc in trajectory]
                self.remapped_inters.append(new_trajectory)
            except Exception:
                logger.exception(f"Error remapping h3 {[loc[0] for loc in trajectory]}")
                continue
        
        
        
        logger.info(f"RecoveryDataset remap complete: {len(self.remapped_inters)} trajectories.")
        
        

        
        
        
        
    def _process_data(self):
        
        def generate_multi_mask(history):
            one_data = dict()
            one_data["user"] = history[-1][3]
            one_data["response"] = ""
            mask_count = random.randint(max(1, int(0.1 * len(history))), max(1, int(0.2 * len(history)))) # 随机选择20%-50%的位置作为mask
            mask_indices = random.sample(range(1,len(history)), mask_count) # 从第2个到最后一个位置随机选择这些位置作为mask
            if self.mode != "test":
                one_data["prediction"] = self.his_sep.join(["At time " + str(item_idx[1]) + ", user " + str(item_idx[3]) + " visited h3 index " + str(item_idx[0]) + "." for item_idx in history])
            else:
                one_data["prediction"] = [{"answer": "At time " + str(item_idx[1]) + ", user " + str(item_idx[3]) + " visited h3 index " + str(item_idx[0]) + ".", "mask": idx in mask_indices} for idx, item_idx in enumerate(history)]
            
            for mask_idx in mask_indices:
                history[mask_idx] = ("[MASK]", history[mask_idx][1], history[mask_idx][2], history[mask_idx][3], history[mask_idx][4])
            history = ["At time " + str(item_idx[1]) + ", user " + str(item_idx[3]) + " visited h3 index " + str(item_idx[0]) + "." for item_idx in history]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]  # 添加序号前缀 1. item1
            one_data["inters"] = self.his_sep.join(history)
            one_data["multi"] = " and output the complete current trajectory" 
            return one_data
        
        
        
                    
            # one_data = dict()
            # one_data["user"] = hist[i][3]
            # one_data["response"] = f"At time {hist[i][1]}, user {hist[i][3]} visited h3 index "
            # one_data["prediction"] = hist[i][0]
            # history = [
            #     f"At time {h[1]}, user {h[3]} visited h3 index "
            #     + ("[MASK]" if j == i else h[0]) + "."
            #     for j, h in enumerate(hist)
            # ]
            # if self.add_prefix:
            #     history = [f"{k + 1}. {v}" for k, v in enumerate(history)]
            # one_data["inters"] = self.his_sep.join(history)
            # profile = self.user_profile.loc[self.user_profile['user_id'] == hist[i][3]]
            # one_data["profile"] = (f"User {hist[i][3]}: {profile['prompt'].values[0]} "
            #                         if self.add_profile and not profile.empty else "")
            # one_data["time"] = hist[i][1]
            # inter_data.append(one_data)

        def generate_single_mask(history):
            one_data_list = []
            # one_trips_new = []
            # one_trips_sparse = []
            # one_num_label = []
            mask_count = random.randint(max(1, int(0.2 * len(history))), max(1, int(0.5 * len(history)))) # 随机选择20%-50%的位置作为mask
            mask_indices = random.sample(range(1,len(history)), mask_count) # 从第2个到最后一个位置随机选择这些位置作为mask
            # for i in range(len(history)):
            #     timestamp = pd.to_datetime(history[i][1]).timestamp()
            #     one_trips_new.append((int(history[i][0]), history[i][2], history[i][3], timestamp))
            #     if i not in mask_indices:
            #         one_num_label.append(0)
            #         one_trips_sparse.append((int(history[i][0]), history[i][2], history[i][3], timestamp))
            #     else:
            #         one_num_label[-1] += 1
            for mask_idx in mask_indices:
                one_data = dict()
                history_one = history.copy()
                one_data["user"] = history_one[mask_idx][3]
                one_data["response"] = "At time " + str(history_one[mask_idx][1]) + ", user " + str(history_one[mask_idx][3]) + " visited h3 index "
                one_data["prediction"] = history_one[mask_idx][0]
                history_one = [("At time " + str(item_idx[1]) + ", user " + str(item_idx[3]) + " visited h3 index " + ("[MASK]" if idx == mask_idx else "[UNKNOWN]" if idx in mask_indices else str(item_idx[0])) + ".") for idx, item_idx in enumerate(history_one)]
                if self.add_prefix:
                    history_one = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history_one)]
                one_data["inters"] = self.his_sep.join(history_one)
                one_data["multi"] = ""
                # try:
                #     profile = self.user_profile.loc[self.user_profile['user_id'] == int(one_data["user"])]
                # except Exception as e :
                #     import pdb; 
                #     pdb.set_trace()
                #     print ("user id",history_one[mask_idx][3],one_data["user"])
                
                
                # one_data["profile"] = (f"User {str(history_one[mask_idx][3])}: {profile['prompt'].values[0]} "
                #                     if self.add_profile and not profile.empty else "")
                # one_data["time"] = history_one[mask_idx][1]
                one_data_list.append(one_data)
            return one_data_list
            # return one_trips_new, one_trips_sparse, one_num_label
        
        
        
        
        
        logger.info("Processing RecoveryDataset...")        
        inter_data = []
        # start,end=self._data_selection()
        for traj in tqdm(self.remapped_inters):
            
            
         
            if self.max_his_len > 0:
                history = traj[:self.max_his_len]  # 只保留最近的max_his_len个历史记录
            if self.multi_rec and self.mode != "test":
                one_data = generate_multi_mask(history.copy())
                inter_data.append(one_data)
            if self.single_rec or self.mode == "test":
                one_data_list = generate_single_mask(history.copy())
                inter_data.extend(one_data_list)
                
                
             # num_labels.append(one_num_label)
        self._free_attrs("remapped_inters")
        if self.add_profile:
            for one_data in inter_data:
                try:
                    profile = self.user_profile[self.user_profile['user_id']==int(one_data["user"])]
                    one_data["profile"] = "User "+ str(one_data["user"]) +" has the following profile: "+profile['prompt'].values[0]+" "                    
                except Exception as e:
                    one_data["profile"] = "" 
        else:
            for one_data in inter_data:
                one_data["profile"] = "" 
        return inter_data
                
                
        #     hist = traj[:self.max_his_len]
        #     mask_num = random.randint(1, max(1, len(hist)//10))
        #     mask_idx = random.sample(range(len(hist)), mask_num)
            
        #     one_data=generate_multi_mask(hist.copy())
        #     for i in mask_idx:
        #         try:
        #             one_data = dict()
        #             one_data["user"] = hist[i][3]
        #             one_data["response"] = f"At time {hist[i][1]}, user {hist[i][3]} visited h3 index "
        #             one_data["prediction"] = hist[i][0]
        #             history = [
        #                 f"At time {h[1]}, user {h[3]} visited h3 index "
        #                 + ("[MASK]" if j == i else h[0]) + "."
        #                 for j, h in enumerate(hist)
        #             ]
        #             if self.add_prefix:
        #                 history = [f"{k + 1}. {v}" for k, v in enumerate(history)]
        #             one_data["inters"] = self.his_sep.join(history)
        #             profile = self.user_profile.loc[self.user_profile['user_id'] == hist[i][3]]
        #             one_data["profile"] = (f"User {hist[i][3]}: {profile['prompt'].values[0]} "
        #                                    if self.add_profile and not profile.empty else "")
        #             one_data["time"] = hist[i][1]
        #             inter_data.append(one_data)
        #         except Exception:
        #             logger.exception("Error processing a recovery sample.")
        # logger.info(f"RecoveryDataset processed: {len(inter_data)} records.")
        
        
        
       

        
        # 
        
        
                    

        # inter_data = []
        # # trips_new = [] # 用于baseline的复现
        # # trips_sparse = [] # 用于baseline的复现
        # # num_labels = [] # 用于baseline的复现
        # # user_list = [] # 用于baseline的复现
        # for trajectory in tqdm(self.remapped_inters):
        #     history = trajectory
        #     if self.max_his_len > 0:
        #         history = history[:self.max_his_len]  # 只保留最近的max_his_len个历史记录
        #     if self.multi_rec and self.mode != "test":
        #         one_data = generate_multi_mask(history.copy())
        #         inter_data.append(one_data)
        #     if self.single_rec or self.mode == "test":
        #         one_data_list = generate_single_mask(history.copy())
        #         inter_data.extend(one_data_list)
        #     # one_trips_new, one_trips_sparse, one_num_label = generate_single_mask(history)
        #     # trips_new.append(one_trips_new)
        #     # user_list.append(trajectory[0][4])
        #     # trips_sparse.append(one_trips_sparse)
        #     # num_labels.append(one_num_label)
        # if self.add_profile:
        #     for one_data in inter_data:
        #         profile = self.user_profile[self.user_profile['user_id']==int(one_data["user"])]
        #         one_data["profile"] = "User "+one_data["user"]+" has the following profile: "+profile['prompt'].values[0]+" "
        # else:
        #     for one_data in inter_data:
        #         one_data["profile"] = "" 
        # # df = pd.DataFrame({
        # #     'trips_new': trips_new,
        # #     'trips_sparse': trips_sparse,
        # #     'num_labels': num_labels,
        # #     'user_list': user_list
        # # })
        # # df.to_csv(os.path.join(self.data_path, f"{self.mode}_recovery_data.csv"), index=False)
        # return inter_data    

class Index2LocationDataset(BaseDataset):
    # Task -- Index to Location

    def __init__(self, args):
        super().__init__(args)
        self.prompts = all_prompt["index"]  # 所有的prompt
        self.task_prompt = task_prompt
        self.mode = "train"

        logger.info("Initializing Index2LocationDataset...")

        try:
            self._load_data()
            self.inter_data = self._process_data()
            if self.mode == "train":
                pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/train/inner_data_i2l_dataset.feather")
            if self.mode=="test":
                pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/test/inner_data_i2l_dataset.feather")            
            logger.info(f"Index2LocationDataset loaded successfully with {len(self.inter_data)} samples.")
        except Exception:
            logger.exception("Error initializing Index2LocationDataset.")
            raise


    def _load_data(self):
        # load data for abalation study
        # location_prompt = {}
        # for file in os.listdir(os.path.join(self.data_path, "prompts")): 
        #     with open(os.path.join(self.data_path, "prompts", file), 'r') as f:
        #         content = f.read().split("\n")
        #         if self.abalation_location_prompt=="1":
        #             content.pop(3)
        #         elif self.abalation_location_prompt=="2":
        #             content.pop(4)
        #         elif self.abalation_location_prompt=="3":
        #             content.pop(5)
        #         content = "\n".join(content)                
        #         location_prompt[file.split(".")[0]] = content
        # self.location_prompt = location_prompt
        # 读取index文件
        logger.info("Loading index-to-location mapping data...")
        try:
            index_path = os.path.join(self.data_path, self.index_file)
            with open(index_path, "r") as f:
                self.codebook = json.load(f)
            logger.info(f"Loaded index file: {index_path}")

            prompt_dir = os.path.join(self.data_path, "grid_profile_codebook")
            if not os.path.exists(prompt_dir):
                logger.error(f"Prompt directory not found: {prompt_dir}")
                raise FileNotFoundError(f"Missing prompt directory: {prompt_dir}")

            self.prompts_map = {}
            for fn in os.listdir(prompt_dir):
                path = os.path.join(prompt_dir, fn)
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                self.prompts_map[fn.split(".")[0]] = content
            logger.info(f"Loaded {len(self.prompts_map)} prompt files from {prompt_dir}")
        except Exception:
            logger.exception("Error loading data in Index2LocationDataset.")
            raise
        # 会有一模一样的location


    def _process_data(self):
        logger.info("Processing Index2LocationDataset samples...")
        data = []
        try:
            for idx, desc in self.prompts_map.items():
                if idx not in self.codebook:
                    logger.warning(f"Index '{idx}' not found in codebook; skipping.")
                    continue
                one_data = {
                    "index": "".join(self.codebook[idx]),
                    "response": desc
                }
                data.append(one_data)
            logger.info(f"Processed {len(data)} index-to-location pairs.")
        except Exception:
            logger.exception("Error processing Index2LocationDataset data.")
            raise
        return data
    
class Location2IndexDataset(BaseDataset):
    # Task -- Location to Index


    def __init__(self, args):
        super().__init__(args)
        self.prompts = all_prompt["location"]  # 所有的prompt
        self.task_prompt = task_prompt
        self.mode = "train"

        logger.info("Initializing Location2IndexDataset...")

        try:
            self._load_data()
            self.inter_data = self._process_data()
            if self.mode=="train":
                pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/train/inner_data_l2i_dataset.feather")
            if self.mode=="test":
                pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/test/inner_data_l2i_dataset.feather")
            logger.info(f"Location2IndexDataset loaded successfully with {len(self.inter_data)} samples.")
        except Exception:
            logger.exception("Error initializing Location2IndexDataset.")
            raise

    def _load_data(self):
        # # load data for abalation study
        # location_prompt = {}
        # for file in os.listdir(os.path.join(self.data_path, "prompts")): 
        #     with open(os.path.join(self.data_path, "prompts", file), 'r') as f:
        #         content = f.read().split("\n")
        #         if self.abalation_location_prompt=="1":
        #             content.pop(3)
        #         elif self.abalation_location_prompt=="2":
        #             content.pop(4)
        #         elif self.abalation_location_prompt=="3":
        #             content.pop(5)
        #         content = "\n".join(content)
        #         location_prompt[file.split(".")[0]] = content
        # self.location_prompt = location_prompt
        
        # 读取index文件
        logger.info("Loading location-to-index mapping data...")
        try:
            index_path = os.path.join(self.data_path, self.index_file)
            with open(index_path, "r") as f:
                self.codebook = json.load(f)
            logger.info(f"Loaded index file: {index_path}")

            prompt_dir = os.path.join(self.data_path, "grid_profile_codebook")
            if not os.path.exists(prompt_dir):
                logger.error(f"Prompt directory not found: {prompt_dir}")
                raise FileNotFoundError(f"Missing prompt directory: {prompt_dir}")

            self.prompts_map = {}
            for fn in os.listdir(prompt_dir):
                path = os.path.join(prompt_dir, fn)
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                self.prompts_map[fn.split(".")[0]] = content
            logger.info(f"Loaded {len(self.prompts_map)} prompt files from {prompt_dir}")
        except Exception:
            logger.exception("Error loading data in Location2IndexDataset.")
            raise
        # 会有一模一样的location


    def _process_data(self):
        logger.info("Processing Location2IndexDataset samples...")
        data = []
        try:
            for idx, desc in self.prompts_map.items():
                if idx not in self.codebook:
                    logger.warning(f"Index '{idx}' not found in codebook; skipping.")
                    continue
                one_data = {
                    "location": desc,
                    "response": "".join(self.codebook[idx])
                }
                data.append(one_data)
            logger.info(f"Processed {len(data)} location-to-index pairs.")
        except Exception:
            logger.exception("Error processing Location2IndexDataset data.")
            raise
        return data
    
    
class TrajectoryTranslationDataset(BaseDataset):
    # Task -- Trajectory Translation

    def __init__(self, args, mode="train"):
        super().__init__(args)
        self.mode = mode
        self.prompts = all_prompt["trans"]
        self.task_prompt = task_prompt
        logger.info(f"Initializing TrajectoryTranslationDataset (mode={self.mode})")

        try:
            self._load_data()
            self._remap_items()
            self.inter_data = self._process_data()
            if self.mode=="train":
                pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/train/inner_data_taj_dataset.feather")
            if self.mode=="test":
                pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/test/inner_data_taj_dataset.feather")            
            logger.info(f"TrajectoryTranslationDataset ready with {len(self.inter_data)} records.")
        except Exception:
            logger.exception("TrajectoryTranslationDataset initialization failed.")
            raise


    def _load_data(self):
        # load data
        logger.info("Loading translation data...")
        
        self.inter_data_dict = self.load_multi_days_data()
        self.inner_data = self.merge_data()
        self._free_attrs("inter_data_dict")
                
        
        # self.inner_data = pd.read_pickle(os.path.join(self.data_path, self.data_filename))
        # # load data for abalation study
        # location_prompt = {}
        # for file in os.listdir(os.path.join(self.data_path, "prompts")): 
        #     with open(os.path.join(self.data_path, "prompts", file), 'r') as f:
        #         content = f.read().split("\n")
        #         if self.abalation_location_prompt=="1":
        #             content.pop(3)
        #         elif self.abalation_location_prompt=="2":
        #             content.pop(4)
        #         elif self.abalation_location_prompt=="3":
        #             content.pop(5)
        #         content = "\n".join(content)
        #         location_prompt[file.split(".")[0]] = content
        # self.location_prompt = location_prompt
        
        with open(os.path.join(self.data_path, self.index_file), "r") as f:
            self.codebook = json.load(f)
        self.user_profile = pd.read_csv(os.path.join(self.data_path, "user_profile_codebook.csv"),
                                        converters={'latest_5_trips': eval}, sep="|")

    # trajectory: [(index, time, loc[0], loc[1], user_id, traj_id), ...]

    def _remap_items(self):
        all_trajectory = []
        user_set = self.inner_data['user_id'].unique()
        logger.info(f"Remapping {len(user_set)} users in TranslationDataset...")
        for user_id in tqdm(user_set, desc="Remapping TranslationDataset"):
            try:
                trajs = self.inner_data[self.inner_data['user_id'] == user_id]
                trajs = trajs.sort_values(['trajectory_num', 'point_order'], ascending=True)
                for traj_id in trajs['trajectory_num'].unique():
                    traj_session = []
                    pev = datetime.fromisoformat("1000-01-01 00:00:00+09:00")
                    for _, row in trajs[trajs['trajectory_num'] == traj_id].iterrows():
                        if type(row['time']) is str:
                            stamp = datetime.fromisoformat(row['time'])
                        else:
                            stamp = row['time'].to_pydatetime()
                        if (stamp - pev).total_seconds() > 180:
                            traj_session.append((str(row['h3']), row['time'], user_id, traj_id))
                            pev = stamp
                    if len(traj_session) >= 2:
                        all_trajectory.append(traj_session)
            except Exception:
                logger.exception(f"Error in TranslationDataset remapping for user {user_id}")

        self.remapped_inters = [
            [("".join(self.codebook[loc[0]]), loc[1], loc[0], loc[2], loc[3]) for loc in t]
            for t in all_trajectory
        ]
        self._free_attrs("inter_data")
                
        logger.info(f"Translation remap complete: {len(self.remapped_inters)} trajectories.")


    def _process_data(self):
        logger.info("Processing TranslationDataset...")
        data = []
        for traj in tqdm(self.remapped_inters):
            try:
                hist = traj[:self.max_his_len]
                one_data = dict()
                one_data["user"] = hist[0][3]
                one_data["response"] = self.his_sep.join([
                    f"[{k + 1}] At time {it[1]}, user {it[3]} visited h3 index {it[0]}."
                    for k, it in enumerate(hist)
                ])
                one_data["inters"] = self.his_sep.join([
                    f"[{k + 1}] Time: {it[1]}, Description: (index {it[0]}, h3={it[2]})"
                    for k, it in enumerate(hist)
                ])
                profile = self.user_profile.loc[self.user_profile['user_id'] == hist[0][3]]
                one_data["profile"] = (f"User {hist[0][3]}: {profile['prompt'].values[0]} "
                                       if self.add_profile and not profile.empty else "")
                # one_data["time"] = it[1]
                data.append(one_data)
            except Exception:
                logger.exception("Error processing translation sample.")
        logger.info(f"TranslationDataset processed: {len(data)} samples.")
        self._free_attrs("remapped_inters")
        return data