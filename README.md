# RLHF with GRPO for ECG Medical Reasoning

GRPO Experiments before integrated into [ECG-Bench](https://github.com/willxxy/ECG-Bench)

## Overview

- **Base Model**: [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Dataset**: [ECG-Expert-QA](https://github.com/Zaozzz/ECG-Expert-QA)
- **Method**: GRPO (Group Relative Policy Optimization)

## Project Structure

```
RL/
├── data/                        # Dataset files
│   ├── raw/                    # Raw ECG-Expert-QA JSON files
│   ├── processed/              # Processed training data
│   └── system_prompt.txt       # ECG QA expert system prompt
├── models/                      # Model checkpoints
│   ├── base/                   # Base Llama model
│   ├── sft/                    # Supervised fine-tuned model
│   └── rlhf/                   # RLHF checkpoints
├── scripts/
│   ├── prepare_data.sh         # Prepare data for SFT/RL
│   ├── sft_train.sh            #SFT (LoRA)
│   └── chat_with_model.sh      # Quick script to chat with model
├── configs/                     # Training configurations
├── utils/                       # Helper functions
├── chat.py                      # Interactive chat with models
├── requirement.txt              # Python dependencies
├── README.md                    # This file
├── prepare_data.py              # Data preprocessing 
├── sft_train.py                 # SFT training
└── grpo_train.py                # GRPO training
```

## Step-by-Step Implementation Guide

### Phase 1: Environment Setup

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Add VERL as a git submodule for version control and reproducibility
```bash
git submodule add https://github.com/volcengine/verl.git verl
git submodule update --init --recursive
```
```bash
cd verl
pip install --no-deps -e .
cd ..
```


#### huggingface login
```bash
huggingface-cli login
```

#### Clone Dataset
```bash
git submodule add https://github.com/Zaozzz/ECG-Expert-QA data/raw/ECG-Expert-QA
git submodule update --init --recursive
```

---

### Phase 2: Data Preparation

#### Preprocess Data
Run the data preparation script:
```bash
bash scripts/prepare_data.sh
```
---

### Phase 3: Supervised Fine-Tuning (SFT)

#### Configure SFT Training

**Training Parameters:**
- `--model_name`: Base model (meta-llama/Llama-3.2-3B-Instruct)
- `--num_epochs`
- `--batch_size`
- `--gradient_accumulation_steps`
- `--learning_rate`
- `--max_seq_length`
- `--lora_r`
- `--lora_alpha`

#### Run SFT Training
```bash
bash scripts/sft_train.sh
```
---


### Phase 4: GRPO Training
https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/README.md

#### Configure GRPO

#### Run GRPO Training


---

### Phase 5: Evaluation & Testing

#### Quantitative Evaluation

#### Qualitative Analysis

#### Compare Models

---