# =====================================================
# Qwen3 ChatML Prompt Template for Mobility Tasks
# =====================================================
# ✅ 改成单行字符串 ChatML 格式
# sft_prompt = "<|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n{response}{prediction}"

sft_prompt = "<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n{response}{prediction}"


# =====================================================
# only last day traj
# =====================================================

# one_day_traj_system_prompt = (
#     "<<SYS>>\n"
#     "You are a helpful assistant that predicts human mobility trajectories in Tokyo.\n"
#     "Do NOT output your thinking process or <think> tags.\n"
#     "Return ONLY a valid JSON LIST containing the sequence of visits.\n"
#     "The value of \"h3_index\" must be a valid 4-token H3 index string.\n"
#     "\"stay_duration\" must be one of 30, 60, 90, ..., 600 (step 30), formatted as \"<N> min\".\n"
#     "<</SYS>>\n\n"
# )


# one_day_traj_task_prompt = """\
# Task: According to the user information, please predict the user's daily trajectory (a sequence of visits) for the specific date.\n
# Each object in the list must have the fields: "id", "start_time", "h3_index", and "stay_duration".\n
# """




system_prompt = (
    "<<SYS>>\n"
    "You are a helpful assistant that predicts human mobility trajectories in Tokyo.\n"
    "Please analyze multi-day historical patterns to predict future movements.\n"
    "Do NOT output your thinking process or <think> tags.\n"
    "Do NOT simply copy any single day's trajectory - synthesize patterns from all reference days.\n"
    "Return ONLY a valid JSON LIST containing the sequence of visits.\n"
    "The value of \"h3_index\" must be a valid 4-token H3 index string.\n"
    "\"stay_duration\" must be one of 30, 60, 90, .. ., 600 (step 30), formatted as \"<N> min\".\n"
    "<</SYS>>\n\n"
)




system_prompt_not_indexing = (
    "<<SYS>>\n"
    "You are a helpful assistant that predicts human mobility trajectories in Tokyo.\n"
    "Do NOT output your thinking process or <think> tags.\n"
    "Return ONLY a valid JSON LIST containing the sequence of visits.\n"
    "<</SYS>>\n\n"
)



traj_task_prompt = """\
Task: According to the user information and multi-day historical trajectories, predict the user's daily trajectory (a sequence of visits) for the specific target date.\n
Analyze the patterns across multiple reference days to make an informed prediction.\n
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

# ============================================================
# Prompt 1: 结构化详细版
# ============================================================
prompt = (
    "**Sequential Mobility Prediction Task**\n\n"
    "Given the following information about user {user}:\n\n"
    "---\n"
    "**1. User Profile:**\n{profile}\n\n"
    "---\n"
    "**2. Historical Trajectories (Past 3 Days for Reference):**\n\n"
    "Analyze the following multi-day movement patterns.  "
    "Look for recurring locations, timing patterns, and behavioral regularities.\n\n"
    "{historical_trajectories}\n\n"
    "---\n"
    "**3. Target Date:** {date}\n\n"
    "---\n"
    "**Instructions:**\n"
    "- Analyze patterns across ALL reference days (not just one day)\n"
    "- Consider which locations appear frequently\n"
    "- Note typical timing patterns for different activities\n"
    "- Generate a NEW trajectory that reflects the user's habits\n"
    "- Do NOT simply copy any single reference day\n\n"
    "**Output:** JSON list with id, start_time, h3_index, stay_duration for each visit.\n"
)
daily_traj_prompt.append(prompt)

# ============================================================
# Prompt 2: 简洁直接版
# ============================================================
prompt = (
    "**User:** {user}\n"
    "**Profile:** {profile}\n\n"
    "**Historical Trajectories (3 Days):**\n"
    "{historical_trajectories}\n\n"
    "**Target Date:** {date}\n\n"
    "**Predicted Trajectory:**"
)
daily_traj_prompt.append(prompt)

# ============================================================
# Prompt 3: 问答式
# ============================================================
prompt = (
    "Given the following mobility data for user {user} in Tokyo:\n\n"
    "User Profile:  {profile}\n\n"
    "Past 3 days trajectories:\n{historical_trajectories}\n\n"
    "What would be the most likely trajectory for {date}?\n"
    "Output as JSON list with id, start_time, h3_index, stay_duration."
)
daily_traj_prompt.append(prompt)

# ============================================================
# Prompt 4: 分析引导版
# ============================================================
prompt = (
    "**Mobility Pattern Analysis**\n\n"
    "**User Information:**\n"
    "- User ID: {user}\n"
    "- Profile: {profile}\n\n"
    "**Reference Data (Past 3 Days):**\n"
    "{historical_trajectories}\n\n"
    "**Prediction Target:** {date}\n\n"
    "Based on the observed patterns, predict the trajectory:\n"
)
daily_traj_prompt. append(prompt)

# ============================================================
# Prompt 5: 对话式
# ============================================================
prompt = (
    "I need to predict where user {user} will go on {date}.\n\n"
    "Here's what I know about them:\n"
    "{profile}\n\n"
    "Their recent movement history:\n"
    "{historical_trajectories}\n\n"
    "Please predict their trajectory for {date} as a JSON list."
)
daily_traj_prompt.append(prompt)

# ============================================================
# Prompt 6: 模板填充式
# ============================================================
prompt = (
    "**[User]** {user}\n"
    "**[Profile]** {profile}\n"
    "**[History]**\n{historical_trajectories}\n"
    "**[Date]** {date}\n"
    "**[Prediction]**"
)
daily_traj_prompt.append(prompt)

# ============================================================
# Prompt 7: 任务导向版
# ============================================================
prompt = (
    "Task: Predict daily trajectory\n"
    "User: {user}\n"
    "Profile: {profile}\n\n"
    "Historical Data:\n{historical_trajectories}\n\n"
    "Target:  {date}\n"
    "Output:  JSON list (id, start_time, h3_index, stay_duration)\n\n"
    "Prediction:"
)
daily_traj_prompt.append(prompt)

# ============================================================
# Prompt 8: 强调规律版
# ============================================================
prompt = (
    "**Trajectory Prediction for User {user}**\n\n"
    "**Profile:** {profile}\n\n"
    "**Movement Patterns (Last 3 Days):**\n"
    "{historical_trajectories}\n\n"
    "**Key Patterns to Consider:**\n"
    "- Frequent locations across days\n"
    "- Typical departure/arrival times\n"
    "- Duration patterns at each location\n\n"
    "**Predict trajectory for {date}:**"
)
daily_traj_prompt.append(prompt)

# ============================================================
# Prompt 9: 最简版
# ============================================================
prompt = (
    "User {user} | {profile}\n"
    "History:\n{historical_trajectories}\n"
    "Predict for {date}:"
)
daily_traj_prompt.append(prompt)

# ============================================================
# Prompt 10: 带示例格式说明版 ✅ 修复：使用 {{ }} 转义花括号
# ============================================================
prompt = (
    "**Input:**\n"
    "- User: {user}\n"
    "- Profile: {profile}\n"
    "- Historical trajectories (3 days):\n{historical_trajectories}\n"
    "- Target date: {date}\n\n"
    "**Output Format:** [{{\"id\": \"1\", \"start_time\": \"HH:MM\", \"h3_index\": \"xxxx\", \"stay_duration\": \"N min\"}}, ... ]\n\n"
    "**Predicted Trajectory:**"
)
daily_traj_prompt.append(prompt)

# ============================================================
# Prompt 11: 强调不复制版
# ============================================================
prompt = (
    "**Mobility Prediction Task**\n\n"
    "User: {user}\n"
    "Profile: {profile}\n\n"
    "Reference trajectories (DO NOT copy directly):\n"
    "{historical_trajectories}\n\n"
    "Generate a NEW realistic trajectory for {date} based on observed patterns.\n"
    "Output:"
)
daily_traj_prompt.append(prompt)

# ============================================================
# Prompt 12: 时间序列分析版
# ============================================================
prompt = (
    "**Time-Series Mobility Analysis**\n\n"
    "Analyze user {user}'s movement patterns:\n\n"
    "Profile: {profile}\n\n"
    "Day -3 to Day -1 trajectories:\n"
    "{historical_trajectories}\n\n"
    "Forecast Day 0 ({date}) trajectory:\n"
)
daily_traj_prompt. append(prompt)

# ============================================================
# 添加到 all_prompt
# ============================================================
all_prompt["daily_traj"] = daily_traj_prompt

print(f"Total daily_traj prompts: {len(daily_traj_prompt)}")

# prompt = (
#     "Here is the mobility profile for user {user}. "
#     "The profile details their home and work locations (if known), a list of frequently visited locations with typical visit times, "
#     "and their preferences for different POI categories based on visit history: {profile} \n"
#     "Today is a {date}. \n"
#     "The last {date} trajectory of the user is: {last_day_traj} \n"
#     "Task: Predict the daily trajectory of the user for this date. \n"
#     "Predict the sequence of visits, including the start time, the location (H3 index), and the stay duration (in minutes). \n"
#     "Return ONLY a JSON list with the following format and no extra text:\n"
#     # 使用双花括号 {{ }} 来表示 JSON 的花括号，避免 .format() 报错
#     """Example: [{{ "id": "1", "start_time": "HH:MM AM/PM", "h3_index": "...", "stay_duration": "... min" }}, ...]\n"""
# )




#     # =====================================================
#     # Prompt 3: 时间序列预测风格
#     # =====================================================
# prompt = (
#         "**Sequential Mobility Prediction Task**\n\n"
#         "Given the following information about user {user}:\n\n"
#         "1. **User Profile:**\n{profile}\n\n"
#         "2. **Reference Trajectory (Last {date}):**\n{last_day_traj}\n\n"
#         "3. **Target Date:** {date}\n\n"
#         "Predict the user's movement sequence for today. "
#         "The trajectory should follow the temporal order of visits throughout the day.\n\n"
#         "Output format: JSON list with id, start_time, h3_index, stay_duration for each visit.\n"
#     )
# daily_traj_prompt. append(prompt)

#     # =====================================================
#     # Prompt 4: 问答式 - 简洁版
#     # =====================================================
# prompt = (
#         "User {user}'s profile: {profile}\n\n"
#         "Previous {date} trajectory:\n{last_day_traj}\n\n"
#         "Today is also a {date}.\n\n"
#         "Question: What locations will this user visit today, in what order, and for how long?\n\n"
#         "Answer with a JSON list of visits:\n"
#     )
# daily_traj_prompt.append(prompt)

#     # =====================================================
#     # Prompt 5: 强调位置转移
#     # =====================================================
# prompt = (
#         "**Location Transition Prediction**\n\n"
#         "User: {user}\n"
#         "Profile Summary: {profile}\n\n"
#         "**Reference Day ({date}) Visit Sequence:**\n"
#         "{last_day_traj}\n\n"
#         "For today ({date}), predict the sequence of location transitions:\n"
#         "- Where does the user start their day?\n"
#         "- What locations do they visit and in what order?\n"
#         "- How long do they stay at each location?\n"
#         "- When do they arrive at each location?\n\n"
#         "Provide your prediction as a JSON list:\n"
#     )
# daily_traj_prompt.append(prompt)

#     # =====================================================
#     # Prompt 6: 结构化输入输出
#     # =====================================================
# prompt = (
#         "=== INPUT ===\n"
#         "User ID: {user}\n"
#         "Date Type: {date}\n"
#         "User Profile:\n{profile}\n\n"
#         "Historical Trajectory (Previous {date}):\n"
#         "{last_day_traj}\n\n"
#         "=== TASK ===\n"
#         "Generate the predicted daily trajectory for this user.\n\n"
#         "=== OUTPUT FORMAT ===\n"
#         "JSON list: [{{\"id\": \"N\", \"start_time\": \"HH:MM AM/PM\", \"h3_index\": \"<4-token code>\", \"stay_duration\": \"N min\"}}]\n\n"
#         "=== PREDICTION ===\n"
#     )
# daily_traj_prompt. append(prompt)






#     # =====================================================
#     # Prompt 9: 分步引导
#     # =====================================================
# prompt = (
#         "Let's predict the daily trajectory for user {user} step by step.\n\n"
#         "**Step 1 - Understand the user:**\n{profile}\n\n"
#         "**Step 2 - Review historical pattern:**\n"
#         "On the previous {date}, the trajectory was:\n{last_day_traj}\n\n"
#         "**Step 3 - Consider today's context:**\n"
#         "Today is a {date}, so the user's behavior should follow similar patterns.\n\n"
#         "**Step 4 - Generate prediction:**\n"
#         "Based on the above analysis, output the predicted trajectory as a JSON list:\n"
#     )
# daily_traj_prompt.append(prompt)

#     # =====================================================
#     # Prompt 10: 最简洁版本
#     # =====================================================
# prompt = (
#         "User: {user}\n"
#         "Profile: {profile}\n"
#         "Last {date}: {last_day_traj}\n"
#         "Today: {date}\n\n"
#         "Predict today's trajectory:\n"
#     )
# daily_traj_prompt. append(prompt)

#     # =====================================================
#     # Prompt 11: 强调 H3 索引
#     # =====================================================
# prompt = (
#         "**Spatial-Temporal Trajectory Prediction**\n\n"
#         "User {user}'s mobility profile:\n{profile}\n\n"
#         "Reference trajectory from previous {date}:\n{last_day_traj}\n\n"
#         "Note: Each location is encoded as a 4-token H3 index representing a specific area in Tokyo.\n\n"
#         "For today ({date}), predict the sequence of H3 locations the user will visit, "
#         "along with arrival times and stay durations.\n\n"
#         "Output format: JSON list with h3_index as 4-token location codes.\n"
#     )
# daily_traj_prompt.append(prompt)

#     # =====================================================
#     # Prompt 12: 角色扮演风格
#     # =====================================================
# prompt = (
#         "You are a mobility prediction system analyzing user {user}.\n\n"
#         "**User Data:**\n"
#         "- Profile: {profile}\n"
#         "- Historical {date} trajectory: {last_day_traj}\n\n"
#         "**Request:**\n"
#         "Generate a predicted trajectory for today ({date}).\n"
#         "The trajectory should be realistic and consistent with the user's historical behavior.\n\n"
#         "**Response format:**\n"
#         "JSON list with fields: id, start_time, h3_index, stay_duration\n"
#     )
# daily_traj_prompt.append(prompt)

#     # =====================================================
#     # Prompt 13: 强调模式匹配
#     # =====================================================
# prompt = (
#         "**Pattern-Based Trajectory Generation**\n\n"
#         "Analyze the mobility pattern of user {user}:\n\n"
#         "Profile information:\n{profile}\n\n"
#         "Observed pattern on previous {date}:\n{last_day_traj}\n\n"
#         "Target: Generate trajectory for today ({date})\n\n"
#         "Instructions:\n"
#         "- Match the temporal pattern (similar start times)\n"
#         "- Use consistent location codes (h3_index)\n"
#         "- Maintain realistic stay durations (30-600 minutes)\n\n"
#         "Generated trajectory:\n"
#     )
# daily_traj_prompt.append(prompt)

  

#     # =====================================================
#     # Prompt 15: 强调预测逻辑
#     # =====================================================
# prompt = (
#         "**Mobility Prediction Request**\n\n"
#         "User: {user}\n"
#         "Day type: {date}\n\n"
#         "**Context:**\n"
#         "The user's profile indicates: {profile}\n\n"
#         "**Reference Data:**\n"
#         "On the last {date}, the user's trajectory was:\n{last_day_traj}\n\n"
#         "**Prediction Logic:**\n"
#         "- Start from the likely home/origin location\n"
#         "- Follow the user's typical daily routine\n"
#         "- Account for work, shopping, dining patterns\n"
#         "- Use appropriate stay durations for each activity type\n\n"
#         "**Predicted Trajectory:**\n"
#     )
# daily_traj_prompt. append(prompt)

# 添加到 all_prompt
# all_prompt["daily_traj"] = daily_traj_prompt






