# 数据准备
def build_location_training_sample(location_code, location_info, tokenizer):
    """
    构建位置编码对齐的训练样本
    
    Args:
        location_code: 如 "<a_13><b_0><c_34><d_12>"
        location_info: 包含 geographic, poi, visit_times, summary 的字典
    """
    
    # System prompt
    system = """<<SYS>> You are a Tokyo urban geography expert. Given a 4-token location code, you provide detailed location analysis.  <</SYS>>"""
    
    # 构建完整的 prompt（输入部分）
    prompt = f"""{system}
    Task: Analyze the location represented by the given 4-token code. 

    Input: {location_code}

    Output:
    """
    
    # 构建 response（输出部分）
    response = f"""**Location Analysis**
    The codebook of this location is {location_code}.

    **Geographic Location**
    {location_info['geographic']}
    The neighbors of this location are: {', '.join(location_info['neighbors'])}.

    **POI Category Distribution**
    (Only categories with probability ≥ 0.05 are listed.)
    {location_info['poi_distribution']}

    **High-frequency Visit Times (1-hour bins)**
    (Only hours with at least 10 visits in total are treated as peaks.)
    The top frequent check-in weekday times of this grid are: {location_info['weekday_peaks']}
    The top frequent check-in weekend times of this grid are: {location_info['weekend_peaks']}

    **Model-generated Location Summary**
    {location_info['summary']}"""

    return {
        "input_ids": prompt,      # 用于训练时的输入
        "labels": response,       # 用于训练时的标签
        "text": prompt + response # 完整文本
    }




def prepare_location_alignment_data(location_data_list, tokenizer, max_length=2048):
    """
    准备位置编码对齐的训练数据
    """
    processed = []
    
    for item in location_data_list:
        location_code = item["code"]  # <a_13><b_0><c_34><d_12>
        location_info = item["info"]  # 包含所有描述信息
        
        sample = build_location_training_sample(location_code, location_info, tokenizer)
        
        # Tokenize
        prompt_ids = tokenizer.encode(sample["input_ids"], add_special_tokens=False)
        response_ids = tokenizer.encode(sample["labels"], add_special_tokens=False)
        
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids  # 只在 response 部分计算 loss
        
        # 截断
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        
        processed.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids)
        })
    
    return processed