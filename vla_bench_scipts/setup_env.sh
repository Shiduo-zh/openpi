#!/bin/bash
export OPENPI_DATA_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/model/
# export LEROBOT_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/lerobot
export HF_LEROBOT_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/lerobot
export HF_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/hf_cache/
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export WANDB_MODE=offline

# 可选：显示设置的环境变量
echo "Environment variables set:"
echo "OPENPI_DATA_HOME: $OPENPI_DATA_HOME"
echo "HF_LEROBOT_HOME: $HF_LEROBOT_HOME"
echo "HF_HOME: $HF_HOME"