# =====================================================
# Qwen3 ChatML Prompt Template for Mobility Tasks
# =====================================================
# ✅ 改成单行字符串 ChatML 格式
# sft_prompt = "<|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n{response}{prediction}"

sft_prompt = "<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n{response}{prediction}"


system_prompt = """\
<<SYS>> You are a helpful assistant that predicts human mobility trajectories in Tokyo. <</SYS>> \
Each H3 index represents a spatial cell at resolution 9, encoded as a 4-token-length "H3 index" that integrates both spatial position and contextual POI-type information.
"""


system_prompt_not_indexing = """\
<<SYS>> You are a helpful assistant that predicts human mobility trajectories in Tokyo. <</SYS>> \
Each "H3 index" is a unique string.
"""

H3_prompt = """\
Your goal is to learn the spatial and locational information represented by each H3 index.
Question: """


task_prompt = """\
A trajectory is a time-ordered sequence of H3 indices, where each index represents the user's location within a specific time interval. 
The sequence captures continuous movement patterns rather than discrete check-ins. 
Each transition between H3 cells reflects spatial mobility and temporal continuity in the user's movement behavior. 
Task: """



user_history_prompt = """User {user} had the following HISTORICAL trajectories: """

all_prompt = {}


# =====================================================
# Task 1 -- Next H3 Prediction -- 10 Prompt
# =====================================================
# seq_prompt = []

# prompt = (
#     "{profile}The following data represents a time-ordered trajectory of user {user}. "
#     "Based on the recent trajectory {inters} "
#     "At {time} which H3 index will the user move to next? And how long will the user stay at there?"
# )
# seq_prompt.append(prompt)

# prompt = (
#     "{profile}Given the continuous trajectory below for user {user}: {inters} "
#     "Forecast the next H3 cell the user will likely move into at time {time}. And how long will the user stay at there?"
# )
# seq_prompt.append(prompt)


# all_prompt["seq"] = seq_prompt

# =====================================================
# Task 1 -- Next H3 Prediction (index + stay duration in seconds) -- 10 Prompt
# =====================================================
seq_prompt = []

prompt = (
    "{profile}The following is a time-ordered trajectory for user {user}: {inters} "
    "At {time}, predict the next H3 cell (resolution 9, Tokyo) the user will move into "
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
index2location_prompt = []

prompt = (
    "For H3 index {index} (resolution 9, Tokyo), produce a grounded location description.\n"
    "If available, include: 6 neighbors, top-5 POI categories with scores, top-3 weekday peak hours, "
    "top-3 weekend peak hours. Output strictly in JSON with keys:\n"
    '{{"index","neighbors","poi_top5","peaks_weekday","peaks_weekend","summary"}}.\n'
    "Use the project’s 4-token-length H3 index format. Use null for any unknown field."
)

index2location_prompt.append(prompt)

prompt = (
    "Interpret H3 cell {index} (r=9, Tokyo). Align the token with regional semantics by listing: "
    "its 6 neighboring H3 indices, the top-5 dominant POI categories with scores, and peak hours "
    "(weekday & weekend, top-3 each). Return JSON only with keys:\n"
    '{{"index","neighbors","poi_top5","peaks_weekday","peaks_weekend","summary"}}.\n'
    "Do not add extra text; fill missing info with null."
)
index2location_prompt.append(prompt)

prompt = (
    "Given H3 index {index}, map the token to its urban context in Tokyo (r=9). "
    "Provide neighbors (6), POI distribution (top-5 with scores), and temporal patterns "
    "(weekday/weekend top-3 hours). Answer ONLY as JSON with keys:\n"
   '{{"index","neighbors","poi_top5","peaks_weekday","peaks_weekend","summary"}}.\n'
    "Keep the index format as the 4-token-length code. Use null if unknown."
)
index2location_prompt.append(prompt)

prompt = (
    "Explain the urban meaning of H3 index {index} in Tokyo at r=9 by aligning the token with its local signals: "
    "list 6 neighboring cells, top-5 POI categories with socres, and top-3 peak hours for weekdays/weekends. "
    "Output JSON only, keys:\n"
   '{{"index","neighbors","poi_top5","peaks_weekday","peaks_weekend","summary"}}.\n'
    "Keep index in 4-token-length form; set null if unknown."
)
index2location_prompt.append(prompt)

all_prompt["index"] = index2location_prompt

# ========================================================
# Task 4 -- Location to Index (H3 r=9, Tokyo) -- 6 Prompt
# ========================================================
location2index_prompt = []

prompt = (
    "{location}\n"
    "Using the provided neighbors, dominant POI categories (with scores), and peak hours (weekday/weekend), "
    "infer the most likely H3 r=9 index in Tokyo. Output the 4-token-length H3 index ."
)
location2index_prompt.append(prompt)


all_prompt["location"] = location2index_prompt


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


