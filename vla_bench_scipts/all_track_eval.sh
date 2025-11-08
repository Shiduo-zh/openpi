#!/bin/bash
# export MUJOCO_EGL_DEVICE_ID=0

pkill -9 python
bash vla_bench_scipts/run_eval.sh \
    pi05_ft_vlabench_primitive \
    checkpoints/pi05_ft_vlabench_primitive/jax_pi05_base_NO_sg/29999

# pkill -9 python
# bash vla_bench_scipts/run_eval.sh \
#     pi05_ft_vlabench_primitive \
#     checkpoints/pi05_ft_vlabench_primitive/torch_pi05_base/30000 \

# pkill -9 python
# bash vla_bench_scipts/run_eval.sh \
#     pi0_posttrain_vlabench_primitive \
#     checkpoints/pi0_posttrain_vlabench_primitive/torch_pi0_base_vla_posttrain/77000 \

# e
# pkill -9 python
# bash vla_bench_scipts/run_eval.sh \
#     pi0_ft_vlabench_primitive \
#     checkpoints/pi0_ft_vlabench_primitive/torch_pi0_base_vla_ft/30000 \

uv run scripts/gpu_runner.py