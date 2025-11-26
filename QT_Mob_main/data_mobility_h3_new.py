from math import log
import random
import os
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

import holidays


japan_holidays = holidays.country_holidays('JP')






def format_time_field(time_value):
    """
    Format a time field for display, handling various input types.
    """
    if hasattr(time_value, 'strftime'):
        return time_value.strftime("%I:%M %p")
    elif isinstance(time_value, str):
        try:
            return datetime.fromisoformat(time_value).strftime("%I:%M %p")
        except (ValueError, TypeError):
            return f"{time_value}:00 AM"
    else:
        return f"{time_value}:00 AM"

def is_holiday(date_str):
    """
    判断某一天是否为日本的节假日或周末。

    参数:
    - date_str: 字符串类型的日期，如 '2024-04-29' 或 '2024/04/29' 或 datetime/date对象

    返回:
    - bool: True if holiday or weekend (in Japan), False otherwise
    """

    # 支持datetime/date对象与字符串
    if isinstance(date_str, (datetime, )):
        date = date_str.date() if hasattr(date_str, "date") else date_str
    elif hasattr(date_str, "year") and hasattr(date_str, "month") and hasattr(date_str, "day"):
        date = date_str
    else:
        # 支持多种日期格式
        for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%Y%m%d'):
            try:
                date = datetime.strptime(str(date_str), fmt).date()
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Unrecognized date format: {date_str}")

    # 周末判定
    if date.weekday() == 5:  # 5=Sat, 6=Sun
        return "Saturday"
    elif date.weekday() == 6:
        return "Sunday"
    elif date in japan_holidays:
        return "Holiday"
    else:
        return "Workday"
    
   

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
        self.data_filename_list = ['20120809.feather','20120810.feather','20120811.feather','20120812.feather','20120817.feather','20120818.feather']
        # self.data_filename_list = [f for f in os.listdir(self.data_path) if f.endswith(".feather") and "2012081" in f]
        # # import re
        # self.data_filename_list = [f for f in os.listdir(self.data_path) if re.search(r"2\d\.feather$", f)]
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
    

    def _process_data(self):
        raise NotImplementedError    
    
   
    
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

        
    def load_multi_days_data(self,start=0,end=-1):
        # 读取所有 self.data_filename_list 中的文件，合并 data
        all_data = {}
        for file_name in self.data_filename_list[start:end]:
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
        



class DailyTrajDataset(BaseDataset):
    # Task --  Predict Daily Trajectory

    def __init__(self, args, mode="train"):
        super().__init__(args)

        self.mode = mode # train, valid, test
        
        self.prompts = all_prompt["daily_traj"] # 所有的prompt
        self.task_prompt = traj_task_prompt
        with open(self.index_file, 'r') as f:
            self.codebook = json.load(f)
        logger.info(f"Initializing daily trajectory dataset (mode={self.mode})")   
        
        # # 生成daily trajectory dataset
        # try:
        #     if self.mode=="valid":
        #         self._load_data()
        #         # self._remap_items()
        #         self.inter_data = self._process_data()
        #         pd.DataFrame(self.inter_data).to_feather("LLMMove/QT_Mob_main/dataset/valid/zdc_h3_8/daily_traj_dataset.feather")
        #     if self.mode == "train":
        #         self._load_data()
        #         # self._remap_items()
        #         self.inter_data = self._process_data()
        #         pd.DataFrame(self.inter_data).to_feather("LLMMove/QT_Mob_main/dataset/train/zdc_h3_8/daily_traj_dataset.feather")
        #     if self.mode=="test":
        #         self._load_data()
        #         # self._remap_items()
        #         self.inter_data = self._process_data()
        #         pd.DataFrame(self.inter_data).to_feather("LLMMove/QT_Mob_main/dataset/test/zdc_h3_8/daily_traj_dataset.feather")
        #     logger.info(f"daily trajectory data loaded successfully: {len(self.inter_data)} samples.")
        # except Exception:
        #     logger.exception("daily trajectory dataset initialization failed.")
        #     raise
        # logger.info(f"daily trajectory dataset generated ({len(self.inter_data)} STAY points).")
        
        # 加载next loc dataset
        try:
            if self.mode=="valid":
                self.inter_data=pd.read_feather("QT_Mob_main/dataset/valid/zdc_h3_8/daily_traj_dataset.feather")
                self.inter_data=self.inter_data.to_dict(orient="records")
            if self.mode == "train":
                self.inter_data=pd.read_feather("QT_Mob_main/dataset/train/zdc_h3_8/daily_traj_dataset.feather")
                self.inter_data=self.inter_data.to_dict(orient="records")
            # if self.mode=="test":
            #     self._load_data()
            #     # self._remap_items()
            #     self.inter_data = self._process_data()
            #     pd.DataFrame(self.inter_data).to_feather("LLMMove/QT_Mob_main/dataset/test/zdc_h3_8/daily_traj_dataset.feather")
               
            if self.mode=="test":
                self.inter_data=pd.read_feather("LLMMove/QT_Mob_main/dataset/test/zdc_h3_8/daily_traj_dataset.feather")
                self.inter_data=self.inter_data.to_dict(orient="records")                
          
            logger.info(f"daily trajectory dataset loaded successfully: {len(self.inter_data)} samples.")
        except Exception:
            logger.exception("daily trajectory dataset initialization failed.")
            raise
        logger.info(f"daily trajectory dataset loaded ({len(self.inter_data)} STAY points).")

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
        
        logger.info("Loading data for daily trajectory prediction...")
        
        # split_idx = int(len(self.data_filename_list) * 0.7)
        # if split_idx % 2 == 0:
        #     split_idx += 1 if split_idx < len(self.data_filename_list) else -1
        
        if self.mode == "train":
            self.inter_data_dict = self.load_multi_days_data(0,4)
        else:
            self.inter_data_dict = self.load_multi_days_data(4,len(self.data_filename_list))
        # self.inter_data_dict = self.load_multi_days_data()
        
        self._process_stay_data()
        self._free_attrs("inter_data_dict")
        with open(self.index_file, 'r') as f:
            self.codebook = json.load(f)
        self.user_profile_weekday = pd.read_csv(
            os.path.join(self.data_path, "user_profiles_weekday.csv"), sep=","
        )
        self.user_profile_weekend_holiday = pd.read_csv(
            os.path.join(self.data_path, "user_profiles_holiday.csv"), sep=","
        )
        
        # self.user_profile = pd.read_csv(
        #     os.path.join(self.data_path, "user_profile_codebook.csv"),
        #     converters={'latest_5_trips': eval}, sep="|"
        # )
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

                            # Only include trajectory points that belong to the current day
                            # Filter out points that are on different days
                            if prev_time.date() == datetime.strptime(day_time, '%Y%m%d').date():
                                traj_session.append((str(row['h3']), prev_time, user_id, row['trajectory_num'],duration))
                        else:
                            continue
                 
                if len(traj_session) >= 2:
                    self.stay_data[day_time].append(traj_session)
        
  
    
    def _process_data(self):
        logger.info("Processing Daily Trajectory Dataset...")       
        inter_data = []
        
        for date, day_trajectory in tqdm(self.stay_data.items(), desc="Processing Daily Trajectory Dataset"):
            
            for user_trajectory in day_trajectory:
                # 按照 trajectory_num (第4个元素, 即下标3) 对 user_trajectory 排序
                # 对 user_trajectory 按 trajectory_num 排序，若已有序则不变
                # 实际上 user_trajectory 默认应已按序，可以校验是否排序有变化：
                user_trajectory = sorted(user_trajectory, key=lambda x: int(x[3].split('_')[1]))

                # 数据验证：检查trajectory数据的完整性
                for traj in user_trajectory:
                    if len(traj) != 5:
                        logger.warning(f"Invalid trajectory length: {len(traj)}, expected 5. Trajectory: {traj}")
                        continue
                    h3_idx, time_val, user_id, traj_num, duration = traj
                    if h3_idx not in self.codebook:
                        logger.warning(f"H3 index '{h3_idx}' not found in codebook. Available keys sample: {list(self.codebook.keys())[:3]}")
                #0: h3 index, 1: time, 2: user id, 3: trajectory num, 4: duration
                one_data = dict()
                one_data["user"] = user_trajectory[0][2]
                one_data["response"] = "prediction:"
                # 获取date是否是节假日，假设有一个 is_holiday 方法可用
                one_data["date"] = is_holiday(date)
                if self.add_profile:
                    
                    
                    # profile = self.user_profile.loc[self.user_profile['user_id'] == int(one_data["user"])]
                    if one_data["date"] == "Workday":
                        profile = self.user_profile_weekday.loc[self.user_profile_weekday['user_id'] == int(one_data["user"])]
                    else:
                        profile = self.user_profile_weekend_holiday.loc[self.user_profile_weekend_holiday['user_id'] == int(one_data["user"])]
                    one_data["profile"] = profile['profile'].values[0] if not profile.empty else ""
                else:
                    one_data["profile"] = ""
                
                
                
                try:
                    one_data["prediction"] = json.dumps(
                        [
                            {   "id":str(i+1),
                                "start_time": format_time_field(trajectory[1]),
                                "h3_index": "".join(self.codebook[trajectory[0]]),
                                "stay_duration": f"{self.get_stay_duration(trajectory[4])} min"
                            } for i,trajectory in enumerate(user_trajectory)
                        ],
                        ensure_ascii=False)
                        
                except Exception:
                    logger.exception("Error processing a trajectory sample.")
                
                inter_data.append(one_data)
        logger.info(f"Daily Trajectory Dataset processing complete: {len(inter_data)} records.")
        
        self._free_attrs("stay_data", "user_profile")
        return inter_data
                    
      

class SeqDataset(BaseDataset):
    # Task -- Next Location Prediction

    def __init__(self, args, mode="train"):
        super().__init__(args)

        self.mode = mode # train, valid, test
        
        self.prompts = all_prompt["seq"] # 所有的prompt
        self.task_prompt = task_prompt
        with open(self.index_file, 'r') as f:
            self.codebook = json.load(f)
        # self.user_profile = pd.read_csv(
        #     os.path.join(self.data_path, "user_profile_codebook.csv"),
        #     converters={'latest_5_trips': eval}, sep="|"
        # )
        logger.info(f"Initializing SeqDataset (mode={self.mode})")
        
        
        ## 生成next loc dataset
        try:
            if self.mode=="valid":
                self._load_data()
                self._remap_items()
                self.inter_data = self._process_data()
                pd.DataFrame(self.inter_data).to_feather("LLMMove/QT_Mob_main/dataset/valid/zdc/seq_dataset.feather")
            if self.mode == "train":
                self._load_data()
                self._remap_items()
                self.inter_data = self._process_data()
                pd.DataFrame(self.inter_data).to_feather("LLMMove/QT_Mob_main/dataset/train/zdc/seq_dataset.feather")
            if self.mode=="test":
                self._load_data()
                self._remap_items()
                self.inter_data = self._process_data()
                pd.DataFrame(self.inter_data).to_feather("LLMMove/QT_Mob_main/dataset/test/zdc/seq_dataset.feather")
            logger.info(f"SeqDataset loaded successfully: {len(self.inter_data)} samples.")
        except Exception:
            logger.exception("SeqDataset initialization failed.")
            raise
        logger.info(f"SeqDataset data loaded ({len(self.inter_data)} STAY points).")
        
        # 加载next loc dataset
        # try:
        #     if self.mode=="valid":
        #         # self._load_data()
        #         # self._remap_items()
        #         # self.inter_data = self._process_data()
        #         self.inter_data=pd.read_feather("LLMMove/QT_Mob_main/dataset/valid/zdc/seq_dataset.feather")
        #         self.inter_data=self.inter_data.to_dict(orient="records")
        #     if self.mode == "train":
        #         self.inter_data=pd.read_feather("LLMMove/QT_Mob_main/dataset/train/zdc/seq_dataset.feather")
        #         self.inter_data=self.inter_data.to_dict(orient="records")
        #         # pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/train/inner_data_seq_dataset.feather")
        #     if self.mode=="test":
        #         self.inter_data=pd.read_feather("LLMMove/QT_Mob_main/dataset/test/zdc/seq_dataset.feather")
        #         self.inter_data=self.inter_data.to_dict(orient="records")                
        #     #     pd.DataFrame(self.inter_data).to_feather("QT_Mob_main/dataset/test/inner_data_seq_dataset.feather")
        #     logger.info(f"SeqDataset loaded successfully: {len(self.inter_data)} samples.")
        # except Exception:
        #     logger.exception("SeqDataset initialization failed.")
        #     raise
        # logger.info(f"SeqDataset data loaded ({len(self.inter_data)} STAY points).")

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
        with open(self.index_file, 'r') as f:
            self.codebook = json.load(f)
        
        self.user_profile_weekday = pd.read_csv(
            os.path.join(self.data_path, "user_profiles_weekday.csv"), sep=","
        )
        self.user_profile_weekend_holiday = pd.read_csv(
            os.path.join(self.data_path, "user_profiles_holiday.csv"), sep=","
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

                            # Only include trajectory points that belong to the current day
                            # Filter out points that are on different days
                            if prev_time.date() == datetime.strptime(day_time, '%Y%m%d').date():
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
                
                ##之前的排序似乎有问题，在这里重新排序
                trajectory = sorted(trajectory, key=lambda x: int(x[3].split('_')[1]))
                new_trajectory = [("".join(self.codebook[loc[0]]) ,loc[1],loc[0],loc[2],loc[3],loc[4]) for loc  in trajectory]
                # new_trajectory = [("".join(self.indices[str(self.loc2id[(loc[0],loc[1])])]),loc[2],loc[0],loc[1],loc[3],loc[4]) for loc in trajectory]
                self.remapped_inters.append(new_trajectory)
        logger.info(f"Remapping complete: {len(self.remapped_inters)} trajectories in SeqDataset·····.")    
        self._free_attrs("stay_data")

    
    def _process_data(self):
        logger.info("Processing SeqDataset trajectories...")       
        inter_data = []
        for trajectory in tqdm(self.remapped_inters):
            
            if len(trajectory)<7 :
                continue
            if self.multi_seq and self.mode == "train":
                start = 2
                end = int(len(trajectory)*0.7)
           
            else :
                start = int(len(trajectory)*0.7)
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
                            "stay_duration": f"{self.get_stay_duration(trajectory[i][5])} min",
                        },
                        ensure_ascii=False,
                    )
                    # one_data['duration'] = trajectory[i][5]
                    history = trajectory[:i][-self.max_his_len:]
                    
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]# 只保留最近的max_his_len个历史记录
                        
                        
                        history = [
                            "At time " + (item_idx[1].strftime("%I:%M %p") if hasattr(item_idx[1], 'strftime') else datetime.fromisoformat(item_idx[1]).strftime("%I:%M %p")) + ", user " + str(item_idx[3]) + " stayed at H3 index " + item_idx[0] + " for " + str(self.get_stay_duration(trajectory[i][5])) + " min."
                            for item_idx in history
                        ]

                        
                        # history = ["At time " + str(item_idx[1]) + ", user " + str(item_idx[3]) + " visited h3 index " + item_idx[0] + "." for item_idx in history]
                    if self.add_prefix:
                        history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)] # 添加序号前缀 1. item1 
                    one_data["inters"] = self.his_sep.join(history)
                    one_data["time"] = trajectory[i][1]
                    if self.add_profile:  
                        if one_data["date"] == "Workday":
                            profile = self.user_profile_weekday.loc[self.user_profile_weekday['user_id'] == int(trajectory[i][3])]
                        else:
                            profile = self.user_profile_weekend_holiday.loc[self.user_profile_weekend_holiday['user_id'] == int(trajectory[i][3])]
                        one_data["profile"] = f"User {trajectory[i][3]}: {profile['profile'].values[0]} " if not profile.empty else ""
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
        
        with open(self.index_file, "r") as f:
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
            # one_data["profile"] = (f"User {hist[i][3]}: {profile['profile'].values[0]} "
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
                
                
                # one_data["profile"] = (f"User {str(history_one[mask_idx][3])}: {profile['profile'].values[0]} "
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
                    one_data["profile"] = "User "+ str(one_data["user"]) +" has the following profile: "+profile['profile'].values[0]+" "                    
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
        #             one_data["profile"] = (f"User {hist[i][3]}: {profile['profile'].values[0]} "
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
        #         one_data["profile"] = "User "+one_data["user"]+" has the following profile: "+profile['profile'].values[0]+" "
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
           
            with open(self.index_path, "r") as f:
                self.codebook = json.load(f)
            logger.info(f"Loaded index file: {self.index_path}")

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
            
            with open(self.index_path, "r") as f:
                self.codebook = json.load(f)
            logger.info(f"Loaded index file: {self.index_path}")

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
        
        with open(self.index_file, "r") as f:
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
                one_data["profile"] = (f"User {hist[0][3]}: {profile['profile'].values[0]} "
                                       if self.add_profile and not profile.empty else "")
                # one_data["time"] = it[1]
                data.append(one_data)
            except Exception:
                logger.exception("Error processing translation sample.")
        logger.info(f"TranslationDataset processed: {len(data)} samples.")
        self._free_attrs("remapped_inters")
        return data