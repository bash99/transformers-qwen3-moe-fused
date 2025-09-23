#!/usr/bin/env python3
#
# Randomly initialize a tiny model and its quantized version
# Then it can be trained in example_train_tiny.py

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3MoeConfig

from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


def main():
    patch_bnb_quantizer()

    model_dir = "../Qwen3-30B-A3B-Instruct-2507-fused"
    model_quantized_dir = "../Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #tokenizer.save_pretrained(model_dir)

    # Load and quantize the model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3MoeFusedForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config)
    model.save_pretrained(model_quantized_dir)

    tokenizer.save_pretrained(model_quantized_dir)


if __name__ == "__main__":
    main()
