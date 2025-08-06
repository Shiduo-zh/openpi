#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
source setup_env.sh
export WANDB_MODE=offline 
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

CONFIG=$1      
EXP_ID=$(date "+%Y%m%d-%H%M%S")  
EXP_NAME="${CONFIG}_${EXP_ID}"

uv run scripts/train.py $CONFIG --exp-name=$EXP_NAME --overwrite