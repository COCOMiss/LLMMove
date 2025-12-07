# =====================================================
# Qwen3 ChatML Prompt Template for Mobility Tasks
# =====================================================
# ✅ 改成单行字符串 ChatML 格式
# sft_prompt = "<|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n{response}{prediction}"

sft_prompt = "<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n{response}{prediction}"


system_prompt = """\
<<SYS>> You are a helpful assistant that predicts human mobility trajectories in Tokyo. \n
Do NOT output your thinking process or <think> tags.\n
Return ONLY a valid JSON LIST containing the sequence of visits.\n
The value of "h3_index" must be a valid H3 index string.\n
"stay_duration" must be one of 30, 60, 90, ..., 600 (step 30), formatted as "<N> min".\n
</SYS>>
"""


system_prompt_not_indexing = """\
<<SYS>> You are a helpful assistant that predicts human mobility trajectories in Tokyo. <</SYS>> \
Each "H3 index" is a unique string.
"""

# 2. Task Prompt: 设定字段格式
# 移除了 "predict next" 这种容易引起歧义的词，改为 "daily trajectory sequence"
traj_task_prompt = """\
Task: According to the user information, please predict the user's daily trajectory (a sequence of visits) for the specific date.\n
Each object in the list must have the fields: "id", "start_time", "h3_index", and "stay_duration".\n
"""


# H3_prompt = """\
# Your goal is to learn the spatial and locational information represented by each H3 index.
# Question: """


seq_task_prompt = """\
A trajectory is a time-ordered sequence of H3 indices, where each 4-token-length index represents the user's location within a specific time interval.
Task:According to the user information and the trajectories, please predict the next H3 index the user will stay and how long he/she will stay there"""

location2index_system_prompt = """<<SYS>>
You are a Tokyo urban geography expert specializing in spatial encoding systems.\n
Task:Given a location description, predict its unique 4-token hierarchical location code.\n
Input Information includes:\n
1. Geographic Location - Position relative to Tokyo landmarks (e.g., Tokyo Tower)\n
2.  Neighbors - Adjacent location codes\n
3. POI Category Distribution - Probability distribution of place types (≥0.05)\n
4. High-frequency Visit Times - Peak check-in hours for weekdays and weekends\n
5.  Location Summary - Comprehensive area description\n
Output Format:\n
<a_X><b_Y><c_Z><d_W>\n
Important:Output ONLY the 4-token location code. Do not include any explanation. \n
<</SYS>>
"""


index2loc_system_prompt_location = """\
<<SYS>> You are a Tokyo urban geography expert. Given a 4-token location code, you provide detailed location analysis including geographic context, POI distribution, visit patterns, and a comprehensive summary.\n
Location codes follow the format <a_X><b_Y><c_Z><d_W> where each token represents hierarchical spatial information.\n
Your response must be structured with the following sections:\n
- **Geographic Location**: Relative position within Tokyo\n
- **POI Category Distribution**: Category probabilities (≥0.05 only)\n
- **High-frequency Visit Times**: Peak hours for weekdays and weekends\n
- **Model-generated Location Summary**: A comprehensive description integrating all information\n
<</SYS>>
"""

user_history_prompt = """User {user} had the following HISTORICAL trajectories: """

all_prompt = {}



# =====================================================
# Task 1 -- Next H3 Prediction (index + stay duration in seconds) -- 10 Prompt
# =====================================================
seq_prompt = []

prompt = (
    "This is the user {user}'s profile, which includes the top 5 time intervals the user regularly checks in, as well as the locations the user frequently stays at and their respective frequencies.: {profile} \n"
    "The following is a time-ordered trajectory for user {user}: {inters} "
    "At {time}, predict the next H3 cell the user will stay at "
    "and how long he/she will stay there (in minutes). "
    "Return ONLY JSON with keys:\n"
    '{{"h3_index","stay_duration"}}.\n'
)
seq_prompt.append(prompt)





prompt = (
    "{profile}Given the continuous trajectory of user {user}: {inters} "
    "Forecast at {time} the most probable next H3 index (r=9, Tokyo) and the stay duration (minutes). "
    "Respond strictly in JSON using keys:\n"
    '{{"h3_index","stay_duration"}}.\n'
)
seq_prompt.append(prompt)

all_prompt["seq"] = seq_prompt










# ========================================================
# Task 2 -- Trajectory Recovery --10 Prompt
# ========================================================
recovery_prompt = []

prompt = (
    "{profile}Given this partial trajectory of user {user}: {inters} "
    "Each [MASK] corresponds to an unrecorded position, and [UNKNOWN] to an uncertain one. "
    "Recover the most probable H3 index for the missing part{multi}."
)
recovery_prompt.append(prompt)

prompt = (
    "{profile}User {user}'s trajectory shows missing grid cells: {inters} "
    "Predict the missing H3 index (not the unknown H3) by leveraging the temporal order and spatial proximity of surrounding cells{multi}."
)
recovery_prompt.append(prompt)



prompt = (
    "{profile}In the user {user}'s movement path {inters}, [MASK] denotes an unobserved H3 index and [UNKNOWN] an unreliable one. "
    "Infer the missing H3 index that maintains spatial-temporal consistency{multi}."
)
recovery_prompt.append(prompt)



prompt = (
    "{profile}Trajectory for user {user}: {inters} "
    "Some segments are missing ([MASK]) and others are unknown ([UNKNOWN]). "
    "Predict the missing H3 index (not the unknown H3) based on spatial continuity and trajectory smoothness{multi}."
)
recovery_prompt.append(prompt)


all_prompt["recovery"] = recovery_prompt


# ========================================================
# Task 3 -- Index to Location (H3 r=9, Tokyo) -- 8 Prompt
# ========================================================
index2location_prompts = []

# Prompt 1: 简洁直接
prompt = (
    "Location Code: {index}\n"
)
index2location_prompts. append(prompt)


# Prompt 3: 问答式
prompt = (
    "Given the location code {index}, "
    "what can you tell me about this location in Tokyo?"
)
index2location_prompts.append(prompt)

# Prompt 4: 结构化请求
prompt = (
    "Input: {index}\n\n"
    "Provide the location information:"
)
index2location_prompts. append(prompt)

# Prompt 5: 详细引导
prompt = (
    "The 4-token location code is: {index}\n\n"
    "Please describe this location including its geographic position, "
    "POI distribution, visit patterns, and a comprehensive summary."
)
index2location_prompts.append(prompt)


# Prompt 7: 专业分析
prompt = (
    "**Spatial Analysis Request**\n\n"
    "Code: {index}\n\n"
    "Provide a comprehensive location profile:"
)
index2location_prompts. append(prompt)

# 添加到 all_prompt
all_prompt["index2location"] = index2location_prompts

# ========================================================
# Task 4 -- Location to Index (H3 r=9, Tokyo) -- 6 Prompt
# ========================================================



location2index_prompts = []

# Prompt 1: 简洁直接
prompt = (
    "**Location Description:**\n"
    "{description}\n\n"
    "**Location Code:**"
)
location2index_prompts. append(prompt)

# Prompt 2: 分析引导
prompt = (
    "**Location Information:**\n"
    "{description}\n"
)
location2index_prompts. append(prompt)



# Prompt 7: 问答式
prompt = (
    "Given the following location information in Tokyo:\n\n"
    "{description}\n\n"
    "What is the 4-token location code for this area?"
)
location2index_prompts.append(prompt)

# 添加到 all_prompt
all_prompt["location2index"] = location2index_prompts


# ========================================================
# Task 5 -- Trajectory Translation
# ========================================================
trajectory_translation_prompt = []

# prompt = "Here's a trajectory description of user {user}:\n{inters}\nCan you translate it into a sequence of H3 indices?"
# trajectory_translation_prompt.append(prompt)

prompt = (
    "Here is a time-ordered trajectory description for user {user}:\n"
    "{inters}\n"
    "Translate it into a sequence of H3 indices?"
)

trajectory_translation_prompt.append(prompt)
prompt = "Given the following user {user} path:\n{inters}\nCan you convert it into a sequence of H3 indices?"
trajectory_translation_prompt.append(prompt)
prompt = (
    "User {user}'s path (a trajectory, not discrete check-ins) is described as follows:\n"
    "{inters}\n"
    "Can you transform it into a sequence of H3 indices?"
)

trajectory_translation_prompt.append(prompt)

all_prompt["trans"] = trajectory_translation_prompt







# =====================================================
# Task 6 -- daily trajectory prediction (index + stay duration in minutes) -- 10 Prompt
# =====================================================
daily_traj_prompt = []

prompt = (
    "Here is the mobility profile for user {user}. "
    "The profile details their home and work locations (if known), a list of frequently visited locations with typical visit times, "
    "and their preferences for different POI categories based on visit history: {profile} \n"
    "Today is a {date}. \n"
    "The last {date} trajectory of the user is: {last_day_traj} \n"
    "Task: Predict the daily trajectory of the user for this date. \n"
    "Predict the sequence of visits, including the start time, the location (H3 index), and the stay duration (in minutes). \n"
    "Return ONLY a JSON list with the following format and no extra text:\n"
    # 使用双花括号 {{ }} 来表示 JSON 的花括号，避免 .format() 报错
    """Example: [{{ "id": "1", "start_time": "HH:MM AM/PM", "h3_index": "...", "stay_duration": "... min" }}, ...]\n"""
)

daily_traj_prompt.append(prompt)

all_prompt["daily_traj"] = daily_traj_prompt




