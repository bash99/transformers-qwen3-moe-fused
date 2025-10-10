from unsloth import FastLanguageModel, FastModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset,load_from_disk,concatenate_datasets

max_seq_length = 2048 # Supports RoPE Scaling internally, so choose any!
# Get LAION dataset

final_dataset = load_from_disk("./final_dataset_10k/")
# mix_en_cn_whoami_recommend_10k/
fourbit_models = [
    "../Qwen/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", # Qwen 4B 2x faster
    "../Qwen/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth

# model_name = "../Qwen/Qwen3-4B-Instruct-2507"
model_name = "../Qwen/Qwen3-30B-A3B-Instruct-2507"

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

## best lora_rank? vllm default max is 16? 
model = FastModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,   # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

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


## start training prepare token
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4, # vram limited
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 60, # test train in 60 steps
        learning_rate = 5e-5, # Reduce to 2e-5 for long training runs, range from 2e-4 to 5e-6
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001, # 0.01 for 4B dense, 0.001 for MoE 30B???
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

## only train on outputs, ignore inputs
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

## start the train
trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Hot-fix for qwen3-30B-A3B, comment below if train for other model
from transformers.generation.utils import GenerationMixin
original_f = GenerationMixin._prepare_cache_for_generation
def patched_f(self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device):
    generation_config.cache_implementation = 'dynamic'
    return original_f(self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device)

GenerationMixin._prepare_cache_for_generation = patched_f
# Hot-fix end

### 推理测试
messages = [
    {"role" : "user", "content" : "Continue the sequence: 1, 1, 2, 3, 5, 8,"}
]
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

### 保存lora
loar_path = "30b_lora_model_5e5"
model.save_pretrained(loar_path)  # Local saving
tokenizer.save_pretrained(loar_path)

### 如果需要从头单独 load 这个lora
if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

# Merge to 16bit to save the full model
if True:
    full_path = "30b_model_trained_5e5"
    model.save_pretrained_merged(full_path, tokenizer, save_method = "merged_16bit",)
    tokenizer.save_pretrained(full_path)
