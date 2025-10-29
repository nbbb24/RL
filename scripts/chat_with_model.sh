#!/bin/bash
# Quick script to chat with model(s)
#
# Usage examples:
#   ./scripts/chat_with_model.sh                    # Load base model only
#   ./scripts/chat_with_model.sh sft                # Load SFT model only
#   ./scripts/chat_with_model.sh grpo               # Load GRPO model (checkpoint 126)
#   ./scripts/chat_with_model.sh base sft           # Compare base vs SFT
#   ./scripts/chat_with_model.sh base sft grpo      # Compare all three models
#   ./scripts/chat_with_model.sh grpo 100           # Load GRPO with checkpoint 100

# Configuration
BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
SFT_ADAPTER="models/sft"
GRPO_CHECKPOINT=126  # Default GRPO checkpoint
# GRPO path will be auto-constructed as:
GRPO_ADAPTER="models/grpo/meta-llama/Llama-3.2-3B-Instruct/global_step_${GRPO_CHECKPOINT}/actor/lora_adapter"
SYSTEM_PROMPT="data/system_prompt.txt"
GPU_DEVICE=3


# Run chat script
python chat.py \
    --models base sft grpo \
    --base_model_path $BASE_MODEL \
    --sft_adapter_path $SFT_ADAPTER \
    --grpo_adapter_path $GRPO_ADAPTER \
    --system_prompt $SYSTEM_PROMPT \
    --device $GPU_DEVICE
