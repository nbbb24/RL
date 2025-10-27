#!/bin/bash
# Quick script to chat with model(s)

# Usage scenarios:

# # Scenario 1: No arguments - default to base model
# echo "Loading base model only..."
# python chat.py \
#     --models base \
#     --base_model_path "meta-llama/Llama-3.2-3B-Instruct" \
#     --system_prompt "data/system_prompt.txt" \
#     --device 3

#     # Scenario 2: Single argument "sft" - load SFT model only
# echo "Loading SFT model only..."
# python chat.py \
#     --models sft \
#     --base_model_path "meta-llama/Llama-3.2-3B-Instruct" \
#     --sft_adapter_path "models/sft" \
#     --system_prompt "data/system_prompt.txt" \
#     --device 3


# Scenario 3: Multiple arguments - load all specified models
echo "Loading models"
python chat.py \
    --models base sft \
    --base_model_path "meta-llama/Llama-3.2-3B-Instruct" \
    --sft_adapter_path "models/sft" \
    --system_prompt "data/system_prompt.txt" \
    --device 3
