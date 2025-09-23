#!/usr/bin/env python3

import os
import sys
from unsloth import FastModel
import json

def remove_fused_bnb_4bit_from_adapter_config(directory):
    """
    读取指定目录下的 'adapter_config.json' 文件，
    将其中 'base_model_name_or_path' 的值中的 '-fused-bnb-4bit' 部分移除，
    然后将修改后的配置写回原文件。
    
    参数:
        directory (str): 包含 adapter_config.json 的目录路径
    
    返回:
        str: 修改后的 base_model_name_or_path 值，或 None 如果文件不存在或无变化
    """
    config_path = os.path.join(directory, 'adapter_config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"文件不存在: {config_path}")
    
    # 读取 JSON 文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取 base_model_name_or_path 字段
    base_path = config.get("base_model_name_or_path")
    if not base_path:
        print("警告: 'base_model_name_or_path' 字段不存在，跳过修改。")
        return None
    
    # 移除 '-fused-bnb-4bit' 后缀（如果存在）
    suffix = "-fused-bnb-4bit"
    if base_path.endswith(suffix):
        new_base_path = base_path[:-len(suffix)]
    else:
        # 如果不以该后缀结尾，尝试替换中间出现的（更安全）
        new_base_path = base_path.replace(suffix, "")
    
    # 如果有变化，更新配置并写回文件
    if new_base_path != base_path:
        config["base_model_name_or_path"] = new_base_path
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"已修改: '{base_path}' -> '{new_base_path}'")
    else:
        print("无需修改: 'base_model_name_or_path' 不包含 '-fused-bnb-4bit' 后缀。")
    
    return new_base_path

def generate_test(model, tokenizer):
    messages = [
        {"role" : "user", "content" : "谁创造了你"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
    )
    from transformers import TextStreamer
    _ = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = 1000, # Increase for longer outputs!
        temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )


## need change local lora's adapter_config.json, revert "base_model_name_or_path" to none-fused
model_name = sys.argv[1]
full_path = sys.argv[2]

remove_fused_bnb_4bit_from_adapter_config(model_name)

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    load_in_4bit = False,
    full_finetuning = True
)

model.save_pretrained_merged(full_path, tokenizer, save_method = "merged_16bit",)
tokenizer.save_pretrained(full_path)
