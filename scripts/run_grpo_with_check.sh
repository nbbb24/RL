#!/bin/bash
# GRPO Training Script for ECG-Expert-QA with Dependency Checking
# Based on VERL GRPO official guidelines: https://verl.readthedocs.io/en/latest/algo/grpo.html

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ECG RLHF GRPO Training Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${RED}ERROR: No conda environment is activated!${NC}"
    echo -e "${YELLOW}Please activate the environment first:${NC}"
    echo -e "  ${GREEN}conda activate rlhf${NC}"
    exit 1
elif [ "$CONDA_DEFAULT_ENV" != "rlhf" ]; then
    echo -e "${YELLOW}WARNING: Current environment is '$CONDA_DEFAULT_ENV', expected 'rlhf'${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ Using conda environment: $CONDA_DEFAULT_ENV${NC}"
echo ""

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
python check_dependencies.py
CHECK_EXIT=$?

if [ $CHECK_EXIT -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}Dependency check found issues.${NC}"
    read -p "Continue training anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Training aborted.${NC}"
        echo -e "${BLUE}Fix issues and try again, or run: python check_dependencies.py --fix${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}Starting GRPO training...${NC}"
echo ""

set -x  # Enable command echo

# GPU Selection (set which GPU to use)
export CUDA_VISIBLE_DEVICES=3  # Change this to select different GPU (e.g., 0, 1, 2, 3)

# Ray workarounds for OpenTelemetry compatibility
export RAY_USAGE_STATS_ENABLED=0
export RAY_OTEL_ENABLED=0
export RAY_DASHBOARD_METRICS_ENABLED=0

# Configuration
MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
TRAIN_DATA="data/processed/ECG_Knowledge_Basic_Q_A_grpo.parquet"
VAL_DATA="data/processed/ECG_Knowledge_Basic_Q_A_val.parquet"
OUTPUT_DIR="models/grpo"
PROJECT_NAME="ecg_rl"
EXPERIMENT_NAME="llama3.2_3b_grpo_lora"

# GPU Configuration
N_GPUS=1  # Adjust based on your hardware
N_NODES=1

# GRPO Specific Parameters
N_SAMPLES=2  # Number of samples per prompt (critical for GRPO, must be >1)
TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=16
PPO_MICRO_BATCH_SIZE=8  # Must divide ppo_mini_batch_size * n_samples

# Training Parameters
TOTAL_EPOCHS=3
SAVE_FREQ=20
TEST_FREQ=5

# LoRA Parameters
LORA_RANK=8
LORA_ALPHA=16

# Learning Parameters
LEARNING_RATE=3e-6
KL_LOSS_COEF=0.001

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$N_SAMPLES \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS "$@"
