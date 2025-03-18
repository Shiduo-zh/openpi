export HF_HOME=/remote-home1/sdzhang/huggingface

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_fast_real_mahjong_lora --exp-name=pi0_fast_real_mahjong_lora_new --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_real_mahjong_lora --exp-name=pi0_real_mahjong_lora_new --overwrite


