#!/usr/bin/env python3
#
# Example to train a LoRA on the fused and quantized version of Qwen3-30B-A3B using Unsloth

import os

from unsloth import FastModel

# Import unsloth before others
from datasets import load_dataset,load_from_disk,concatenate_datasets
import torch

from trl import SFTConfig, SFTTrainer

from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer

MAX_CONTEXT_LEN=6144
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

def generate_test(model, tokenizer):
    prompt = "Give me a short introduction to large language model."
    prompt = "谁创造了你？"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids)
    print(content)

def main():
    patch_bnb_quantizer()
    # We can set a smaller rank for MoE layers
    # With rslora, we don't need to set a different alpha for them
    # TODO: Support rank_pattern in Unsloth
    patch_lora_config(
        rank_pattern={
            "q_proj": 16,
            "k_proj": 16,
            "v_proj": 16,
            "o_proj": 16,
            # "gate": 16,  # It's possible to create a LoRA on the routing gate, but this is unstable
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        }
    )
    patch_Qwen3MoeFusedSparseMoeBlock_forward()

    # This is Qwen3 2504. Nowadays you can use Qwen3 2507 for better intelligence
    model_id = "../Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit/"

    model, tokenizer = FastModel.from_pretrained(model_id,
            max_seq_length = MAX_CONTEXT_LEN,
            # device_map="auto", # only for multi gpu
            auto_model=Qwen3MoeFusedForCausalLM)

    ## best lora_rank? vllm default max is 16? 
    model = FastModel.get_peft_model(
        model,
        r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128; 4 for moe as fast train woct0rdho said
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,   # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    final_dataset = load_from_disk("../../datasets/mix_en_cn_whoami_recommend_3k/")
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen3-instruct",
    )

    def formatting_prompts_func(examples):
       convos = examples["conversations"]
       texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
       return { "text" : texts, }

    dataset = final_dataset.map(formatting_prompts_func, batched = True)

    # 假设 dataset 已经通过 map 操作生成，包含 "text" 字段
    texts = dataset["text"]  # 这是一个列表，包含所有转换后的文本字符串
    max_length = max(len(text) for text in texts)
    print(f"最大文本长度: {max_length}")
    import numpy as np

    lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in dataset["text"]]
    print(f"总datasets行数: {len(lengths)}")
    print(f"最大token长度: {max(lengths)}")
    print(f"平均token长度: {np.mean(lengths):.1f}")
    print(f"中位数token 长度: {np.median(lengths):.1f}")
    print(f"95% 分位数: {np.percentile(lengths, 95):.1f}")
    print(f"99% 分位数: {np.percentile(lengths, 99):.1f}")
    
    def filter_by_token_length(example):
        text = example["text"]
        # 使用 add_special_tokens=False，因为 apply_chat_template 已经添加了特殊 token
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return len(token_ids) <= (MAX_CONTEXT_LEN-500) ## sub 500 here for safe
    dataset = dataset.filter(filter_by_token_length)
    print(f"过滤后datasets行数: {len(dataset["text"])}")
    #exit(0)

    sft_config = SFTConfig(
        per_device_train_batch_size=4,  # Increase batch size if you have more memory
        gradient_accumulation_steps=4,
        learning_rate=7e-5,
        weight_decay=5e-3,  # For MoE models, weight decay can be smaller than dense models
        num_train_epochs=1,
        lr_scheduler_type="linear",
        #warmup_steps=1000,
        warmup_ratio = 0.1,
        logging_steps=1,
        save_steps=100,
        save_total_limit=5,
        bf16=True,
        optim="adamw_8bit",
        dataset_text_field="text",
        dataset_num_proc=1,
        torch_compile=True,
        torch_compile_mode="max-autotune",
        # ddp_find_unused_parameters = False, # for multi gpu usage
        report_to="none",  # You may report to Wandb
        seed=3407,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    ## only train on outputs, ignore inputs
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    trainer_stats = trainer.train()
    print("trainer_stats")
    print(trainer_stats)
    ### 保存lora
    normal_lora = "outputs/30b_lora_model_7e5_r8_decay5e3"
    full_path = normal_lora.replace('lora', 'full')
    fused_lora = normal_lora + "_fused"
    model.save_pretrained(fused_lora)  # Local saving
    tokenizer.save_pretrained(fused_lora)

    from qwen3_moe_fused.convert import convert_lora_to_unfused, convert_model_to_unfused
    convert_lora_to_unfused(fused_lora, normal_lora)

    generate_test(model, tokenizer)

    if True:
        print(f"CUDA_VISIBLE_DEVICES=0,1 python merge_full_model.py {normal_lora} {full_path}")
        return

    ## full model merge not work yet
    fused_full_path = full_path + "_fused"
    model.save_pretrained_merged(fused_full_path, tokenizer) # , save_method = "merged_16bit",)
    tokenizer.save_pretrained(fused_full_path)

    convert_model_to_unfused(fused_full_path, full_path)    
    tokenizer.save_pretrained(full_path)

if __name__ == "__main__":
    main()
