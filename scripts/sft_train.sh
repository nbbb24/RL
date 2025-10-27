#!/bin/bash
# SFT training script with LoRA

python sft_train.py \
    --device 3 \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --train_file "data/processed/ECG_Knowledge_(Basic_Q&A)_train.jsonl" \
    --val_file "data/processed/ECG_Knowledge_(Basic_Q&A)_val.jsonl" \
    --system_prompt "data/system_prompt.txt" \
    --output_dir "models/sft" \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --max_seq_length 512 \
    --lora_r 8 \
    --lora_alpha 16

echo "SFT training complete!"
