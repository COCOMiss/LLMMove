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
import numpy as np

logger = get_logger(__name__)
logger.info("==== Dataset module initialized ====")

class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()
        self._tok_cache = {}           # tokenizer 结果缓存（NEW）
        self.user_profile_map = {}     # 轻量 profile 映射（NEW）
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
        self.data_filename_list = [f for f in os.listdir(self.data_path) if f.endswith(".feather")]
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
        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            tokcache = self._tok_cache

            def tok_ids(token: str):
                r = tokcache.get(token)
                if r is None:
                    r = tokenizer(token)["input_ids"]
                    tokcache[token] = r
                return r

            for index in self.codebook.values():
                self.token_len = len(index)
                len_of_token = len(tok_ids(index[0])) - 1
                token_ids = [tok_ids(t)[len_of_token] for t in index]

                self.allowed_tokens.setdefault(token_ids[0], set()).add(token_ids[1])
                for i in range(2, len(token_ids)):
                    key = tuple(token_ids[0:i])
                    self.allowed_tokens.setdefault(key, set()).add(token_ids[i])

            for index in self.codebook.values():
                len_of_token = len(tok_ids(index[0])) - 1
                for i, token in enumerate(index):
                    token_id = tok_ids(token)[len_of_token]
                    self.allowed_tokens.setdefault(i, set()).add(token_id)

        if test_task == "seq":
            sep = tokenizer("will stay at h3 index ", add_special_tokens=False)["input_ids"][1:]
        elif test_task == "recovery":
            sep = tokenizer(" visited h3 index ", add_special_tokens=False)["input_ids"][1:]
        else:
            sep = []

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            logger.info(f"prefix_allowed_tokens_fn sentence  {sentence}")
            reversed_sent = sentence[::-1]

            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    if i == self.token_len:
                        logger.info(f"allowed_tokens i = self.token_len {tokenizer.eos_token_id}")
                        return [tokenizer.eos_token_id]
                    if i == 0 or reversed_sent[0] < 20:
                        logger.info(f"allowed_tokens i = 0 {i}")
                        return list(self.allowed_tokens[i])
                    if i == 1:
                        logger.info(f"allowed_tokens i = 1 {reversed_sent[0]}")
                        return list(self.allowed_tokens[reversed_sent[0]])
                    logger.info(f"allowed_tokens {tuple(reversed_sent[0:i][::-1])}")
                    return list(self.allowed_tokens[tuple(reversed_sent[0:i][::-1])])

            print("Warning: sep not found")
            return []

        return prefix_allowed_tokens_fn

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
      
    def _load_side_assets(self):
        with open(os.path.join(self.data_path, self.index_file), 'r') as f:
            self.codebook = json.load(f)

        prof_path = os.path.join(self.data_path, "user_profile_codebook.csv")
        if os.path.exists(prof_path):
            up = pd.read_csv(prof_path, sep="|", usecols=["user_id", "prompt"], dtype={"user_id": "int32"})
            self.user_profile_map = dict(zip(up["user_id"].tolist(), up["prompt"].tolist()))
            del up
        else:
            self.user_profile_map = {}
        gc.collect()
    def _file_iter(self, need_cols=None):
        if need_cols is None:
            need_cols = ["user_id", "trajectory_num", "point_order", "h3", "time", "transport_mode"]

        for file_name in self.data_filename_list:
            fpath = os.path.join(self.data_path, file_name)
            if not os.path.exists(fpath):
                logger.warning(f"File not found: {fpath}")
                continue
            try:
                df = pd.read_feather(fpath, columns=need_cols)
                base = os.path.splitext(os.path.basename(file_name))[0]

                if "user_id" in df:       df["user_id"] = df["user_id"].astype("int32", copy=False)
                if "point_order" in df:   df["point_order"] = df["point_order"].astype("int32", copy=False)
                if "trajectory_num" in df:
                    df["trajectory_num"] = base + "_" + df["trajectory_num"].astype("int32").astype("string")
                if "transport_mode" in df:
                    df["transport_mode"] = df["transport_mode"].astype("category")
                if "h3" in df:
                    df["h3"] = df["h3"].astype("category")
                if "time" in df:
                    df["time"] = pd.to_datetime(df["time"], errors="coerce")

                yield df, base
            except Exception:
                logger.exception(f"Error reading file: {fpath}")
  
    @staticmethod
    def _build_stay_pairs(df: pd.DataFrame) -> pd.DataFrame:
        need = ["user_id","trajectory_num","point_order","h3","time","transport_mode"]
        s = df.loc[df["transport_mode"] == "STAY", [c for c in need if c in df.columns]].copy()
        if s.empty:
            return s.assign(start_time=pd.NaT, duration=np.float32(0.0)).iloc[0:0]

        s.sort_values(["user_id","trajectory_num","point_order"], inplace=True)
        prev = s.shift(1)

        mask = (
            s["user_id"].eq(prev["user_id"]) &
            s["trajectory_num"].eq(prev["trajectory_num"]) &
            s["h3"].eq(prev["h3"]) &
            s["point_order"].eq(prev["point_order"] + 1)
        )

        end_rows = s.loc[mask]
        start_rows = prev.loc[mask]

        dur = (end_rows["time"].values - start_rows["time"].values) / np.timedelta64(1, "s")
        dur = np.abs(dur).astype("float32")

        out = pd.DataFrame({
            "h3": end_rows["h3"].astype("string").values,
            "start_time": start_rows["time"].values,
            "user_id": end_rows["user_id"].astype("int32").values,
            "trajectory_num": end_rows["trajectory_num"].astype("string").values,
            "duration": dur,
        })
        return out
                

class SeqDataset(BaseDataset):
    # Task -- Next Location Prediction

    def __init__(self, args, mode="train"):
        super().__init__(args)

        self.mode = mode # train, valid, test
        
        self.prompts = all_prompt["seq"] # 所有的prompt
        self.task_prompt = task_prompt
        
        logger.info(f"Initializing SeqDataset (mode={self.mode})")
        try:
            self._load_data()
            self._remap_items()
            self.inter_data = self._process_data()
            logger.info(f"SeqDataset loaded successfully: {len(self.inter_data)} samples.")
        except Exception:
            logger.exception("SeqDataset initialization failed.")
            raise
        logger.info(f"SeqDataset data loaded ({len(self.inter_data)} STAY points).")


    def _load_data(self):
        logger.info("Loading data for SeqDataset (streamed by file)...")
        self._load_side_assets()  # codebook + profile dict

        self.inter_data = []
        append = self.inter_data.append

        # 全局样本计数条（无限总量，按增量更新）
        pbar_samples = tqdm(total=0, desc="Seq • samples", unit="sample", leave=True, dynamic_ncols=True)

        # 文件进度条（按 .feather 数量）
        for df, base in tqdm(
            self._file_iter(need_cols=["user_id","trajectory_num","point_order","h3","time","transport_mode"]),
            total=len(self.data_filename_list),
            desc="Seq • files",
            unit="file",
            leave=True,
            dynamic_ncols=True
        ):
            before = len(self.inter_data)

            pairs = self._build_stay_pairs(df)
            if not pairs.empty:
                pairs.sort_values(["user_id","trajectory_num","start_time"], inplace=True)

                # 轨迹进度条（当前文件内）
                ng = pairs.groupby(["user_id","trajectory_num"], sort=False).ngroups
                with tqdm(total=ng, desc=f"{base} • traj", unit="traj", leave=False, dynamic_ncols=True) as pbar_traj:
                    for (uid, tid), grp in pairs.groupby(["user_id","trajectory_num"], sort=False):
                        # 序列：(idx_str, time, h3, user, traj, duration)
                        seq = []
                        for h3, t, du in zip(grp["h3"].tolist(), grp["start_time"].tolist(), grp["duration"].tolist()):
                            if h3 in self.codebook:
                                seq.append(("".join(self.codebook[h3]), t, h3, int(uid), str(tid), float(du)))
                        if len(seq) < 2:
                            pbar_traj.update(1)
                            continue

                        # 与原切片逻辑一致
                        if self.multi_seq and self.mode == "train":
                            start = 2
                            end = int(len(seq) * 0.7) if len(seq) > 2 else len(seq)
                        elif len(seq) >= 2 and self.mode == "valid":
                            start = int(len(seq) * 0.7)
                            end = int(len(seq) * 0.9)
                        elif len(seq) >= 2 and self.mode == "test":
                            start = len(seq) - 1
                            end = len(seq)
                        else:
                            pbar_traj.update(1)
                            continue

                        for i in range(start, max(end, start)):
                            try:
                                user = seq[i][3]
                                tgt_idx_str = seq[i][0]
                                tgt_time = seq[i][1]
                                tgt_dur = int(seq[i][5])

                                history = seq[:i][-self.max_his_len:] if self.max_his_len > 0 else []
                                history_texts = [
                                    f"At time {str(itm[1])}, user {str(itm[3])} stayed at H3 index {itm[0]} for {int(itm[5])} seconds."
                                    for itm in history
                                ]
                                if self.add_prefix:
                                    history_texts = [f"{k+1}. {line}" for k, line in enumerate(history_texts)]
                                history_joined = self.his_sep.join(history_texts)

                                profile_text = ""
                                if self.add_profile:
                                    p = self.user_profile_map.get(int(user))
                                    if p:
                                        profile_text = f"User {user}: {p} "

                                one_data = dict(
                                    user=user,
                                    response=f"At time {tgt_time}, user {user} will stay at h3 index ",
                                    prediction=f"{tgt_idx_str} for {tgt_dur} seconds.",
                                    inters=history_joined,
                                    time=tgt_time,
                                    profile=profile_text
                                )
                                append(one_data)
                            except Exception:
                                logger.exception("Error processing a trajectory sample.")

                        pbar_traj.update(1)

            # 更新样本条：按本文件新增量
            delta = len(self.inter_data) - before
            if delta > 0:
                pbar_samples.update(delta)

            del df, pairs
            gc.collect()

   
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
        return


    
    def _process_data(self):
        logger.info(f"SeqDataset processing complete: {len(self.inter_data)} records.")
        return self.inter_data


class RecoveryDataset(BaseDataset):
    def __init__(self, args, mode="train"):
        super().__init__(args)
        self.mode = mode
        self.prompts = all_prompt["recovery"]
        self.task_prompt = task_prompt
        logger.info(f"Initializing RecoveryDataset (mode={self.mode})")

        try:
            self._load_data()
            self._remap_items()      # 占位
            self.inter_data = self._process_data()
            logger.info(f"RecoveryDataset loaded successfully: {len(self.inter_data)} samples.")
        except Exception:
            logger.exception("RecoveryDataset initialization failed.")
            raise

    def _load_data(self):
        logger.info("Loading data for RecoveryDataset (streamed by file)...")
        self._load_side_assets()  # codebook + user_profile dict
        # 不再加载 inter_data_dict

    def _remap_items(self):
        return  # 不再常驻内存

    def _process_data(self):
        logger.info("Processing RecoveryDataset (streamed, full trajectories)...")
        inter_data = []
        append = inter_data.append
        extend = inter_data.extend

        # === 定义两个 mask 构造器（原逻辑保留） ===
        def generate_multi_mask(history):
            one_data = dict()
            one_data["user"] = history[-1][3]
            one_data["response"] = ""
            mask_count = random.randint(max(1, int(0.1 * len(history))), max(1, int(0.2 * len(history))))
            mask_indices = random.sample(range(1, len(history)), mask_count)
            if self.mode != "test":
                one_data["prediction"] = self.his_sep.join([
                    "At time " + str(it[1]) + ", user " + str(it[3]) + " visited h3 index " + str(it[0]) + "."
                    for it in history
                ])
            else:
                one_data["prediction"] = [
                    {"answer": "At time " + str(it[1]) + ", user " + str(it[3]) + " visited h3 index " + str(it[0]) + ".",
                     "mask": idx in mask_indices} for idx, it in enumerate(history)
                ]
            hist2 = [("[MASK]" if i in mask_indices else it[0], it[1], it[2], it[3], it[4]) for i, it in enumerate(history)]
            hist_txt = ["At time " + str(it[1]) + ", user " + str(it[3]) + " visited h3 index " + str(it[0]) + "." for it in hist2]
            if self.add_prefix:
                hist_txt = [f"{k+1}. {x}" for k, x in enumerate(hist_txt)]
            one_data["inters"] = self.his_sep.join(hist_txt)
            one_data["multi"] = " and output the complete current trajectory"
            return one_data

        def generate_single_mask(history):
            out = []
            mask_count = random.randint(max(1, int(0.2 * len(history))), max(1, int(0.5 * len(history))))
            mask_indices = set(random.sample(range(1, len(history)), mask_count))
            for m in mask_indices:
                one = dict()
                one["user"] = history[m][3]
                one["response"] = "At time " + str(history[m][1]) + ", user " + str(history[m][3]) + " visited h3 index "
                one["prediction"] = history[m][0]
                lines = []
                for idx, it in enumerate(history):
                    token = "[MASK]" if idx == m else ("[UNKNOWN]" if idx in mask_indices else str(it[0]))
                    lines.append(f"At time {str(it[1])}, user {str(it[3])} visited h3 index {token}.")
                if self.add_prefix:
                    lines = [f"{k+1}. {x}" for k, x in enumerate(lines)]
                one["inters"] = self.his_sep.join(lines)
                one["multi"] = ""
                out.append(one)
            return out

        # === 流式构造样本 ===
        need_cols = ["user_id","trajectory_num","point_order","h3","time"]
        for df, _ in self._file_iter(need_cols=need_cols):
            df.sort_values(["user_id","trajectory_num","point_order"], inplace=True)
            for (uid, tid), g in df.groupby(["user_id","trajectory_num"], sort=False):
                if len(g) < 2:
                    continue
                # 构造完整轨迹
                seq = []
                for h3, tt in zip(g["h3"].astype("string").tolist(), g["time"].tolist()):
                    if h3 in self.codebook:
                        seq.append(("".join(self.codebook[h3]), tt, h3, int(uid), str(tid)))
                if len(seq) < 2:
                    continue

                if self.max_his_len > 0:
                    seq = seq[:self.max_his_len]

                if self.multi_rec and self.mode != "test":
                    append(generate_multi_mask(seq.copy()))
                if self.single_rec or self.mode == "test":
                    extend(generate_single_mask(seq.copy()))
            del df
            gc.collect()

        # 添加用户画像
        if self.add_profile:
            for d in inter_data:
                p = self.user_profile_map.get(int(d["user"]), "")
                d["profile"] = f"User {d['user']} has the following profile: {p} " if p else ""
        else:
            for d in inter_data:
                d["profile"] = ""

        logger.info(f"RecoveryDataset processed: {len(inter_data)} samples.")
        return inter_data

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
    def __init__(self, args, mode="train"):
        super().__init__(args)
        self.mode = mode
        self.prompts = all_prompt["trans"]
        self.task_prompt = task_prompt
        logger.info(f"Initializing TrajectoryTranslationDataset (mode={self.mode})")

        try:
            self._load_data()
            self._remap_items()      # 占位
            self.inter_data = self._process_data()
            logger.info(f"TrajectoryTranslationDataset ready with {len(self.inter_data)} samples.")
        except Exception:
            logger.exception("TrajectoryTranslationDataset initialization failed.")
            raise

    def _load_data(self):
        logger.info("Loading translation data (streamed by file)...")
        self._load_side_assets()

    def _remap_items(self):
        return

    def _process_data(self):
        logger.info("Processing TranslationDataset (streamed, full trajectories)...")
        data = []
        append = data.append

        need_cols = ["user_id","trajectory_num","point_order","h3","time"]
        for df, _ in self._file_iter(need_cols=need_cols):
            df.sort_values(["user_id","trajectory_num","point_order"], inplace=True)
            for (uid, tid), g in df.groupby(["user_id","trajectory_num"], sort=False):
                if len(g) < 2:
                    continue

                seq = []
                for h3, tt in zip(g["h3"].astype("string").tolist(), g["time"].tolist()):
                    if h3 in self.codebook:
                        seq.append(("".join(self.codebook[h3]), tt, h3, int(uid), str(tid)))
                if not seq:
                    continue

                hist = seq[:self.max_his_len] if self.max_his_len > 0 else seq

                try:
                    one = dict()
                    one["user"] = hist[0][3]
                    one["response"] = self.his_sep.join([
                        f"[{k + 1}] At time {it[1]}, user {it[3]} visited h3 index {it[0]}."
                        for k, it in enumerate(hist)
                    ])
                    one["inters"] = self.his_sep.join([
                        f"[{k + 1}] Time: {it[1]}, Description: (index {it[0]}, h3={it[2]})"
                        for k, it in enumerate(hist)
                    ])
                    if self.add_profile:
                        p = self.user_profile_map.get(int(hist[0][3]))
                        one["profile"] = f"User {hist[0][3]}: {p} " if p else ""
                    else:
                        one["profile"] = ""
                    append(one)
                except Exception:
                    logger.exception("Error processing translation sample.")
            del df
            gc.collect()

        logger.info(f"TranslationDataset processed: {len(data)} samples.")
        return data
