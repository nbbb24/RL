#!/bin/bash
# Script to chat with Llama-3.2-3B-Instruct model as ECG QA expert

MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
SYSTEM_PROMPT="data/system_prompt.txt"

python chat.py --model_path $MODEL_PATH --system_prompt $SYSTEM_PROMPT
