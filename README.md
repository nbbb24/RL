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
└── sft_train.py                 # SFT training
```

## Step-by-Step Implementation Guide

### Phase 1: Environment Setup

#### Install Dependencies
Refer to ECG-Bench
Install the uv package manager via ```bash pip install uv.```

For Torch
```bash
uv pip uninstall -y vllm torch torchvision torchaudio
uv pip install "torch>=2.6.0" "torchvision>=0.21.0" \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

For base installation 
```bash 
uv pip install -e . --no-build-isolation
```

For installation with flash attention 
```bash 
uv pip install -e ".[flash]" --no-build-isolation
```

For installation with judge 
```bash 
uv pip install -e ".[judge]"
```

For installation of all packages 
```bash 
uv pip install -e ".[all]" --no-build-isolation
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

#### reward

Need to design reward function under "verl/verl/utils/reward_score/"

Also add this to "verl/verl/utils/reward_score/__init__.py"
```python
elif data_source == "ecg_expert_qa":
    from . import ecg_expert_qa

    res = ecg_expert_qa.compute_score(solution_str, ground_truth, method="hybrid")
```

#### Run GRPO Training

```bash
bash scripts/run_grpo.sh
```

---

### Phase 5: Evaluation & Testing

#### Quantitative Evaluation

#### Qualitative Analysis

#### Compare Models

---



## Issues

**Issue**: vLLM 0.8.4 has a bug that prevents LoRA training with VERL (`AttributeError: 'LoRALRUCache' object has no attribute '_LRUCache__update'`)

**Fix Applied**: Patched `~/anaconda3/envs/rlhf/lib/python3.10/site-packages/vllm/utils.py` lines 277-280

**What was changed**:
```python
# Before (buggy):
def touch(self, key: _K) -> None:
    self._LRUCache__update(key)  # type: ignore

# After (fixed):
def touch(self, key: _K) -> None:
    # Fix for LoRA LRU cache bug - use move_to_end instead
    if key in self._LRUCache__order:  # type: ignore
      