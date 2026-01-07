#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate arvla

export HF_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/hf_cache/
export MUJOCO_GL=egl
NUM_GPUS=$(nvidia-smi -L | wc -l)
NUM_TRIALS=50
MAX_PROCS_PER_GPU=2
MAX_PROCS=$((NUM_GPUS * MAX_PROCS_PER_GPU))

# --------------- 解析参数 ---------------
SAVE_DIR=""
TRACK_OPT=""
TASK_OPT=""

if [ "$#" -lt 1 ]; then
    usage
fi
SAVE_DIR=$1
shift 1

# 解析可选参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --track)
            TRACK_OPT="$2"
            shift 2
            ;;
        --task)
            TASK_OPT="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            usage
            ;;
    esac
done

# --------------- 初始化全量数据 ---------------
ALL_TRACKS=(track_1_in_distribution track_2_cross_category track_3_common_sense track_4_semantic_instruction track_6_unseen_texture)
ALL_TASKS=(add_condiment insert_flower select_book select_drink select_chemistry_tube select_mahjong select_toy select_fruit select_painting select_poker select_nth_largest_poker select_unique_type_mahjong)

# --------------- 处理track与task ---------------
if [[ -n "$TRACK_OPT" ]]; then
    IFS=',' read -ra TRACKS <<< "$TRACK_OPT"
else
    TRACKS=("${ALL_TRACKS[@]}")
fi

if [[ -n "$TASK_OPT" ]]; then
    IFS=',' read -ra TASKS <<< "$TASK_OPT"
else
    TASKS=("${ALL_TASKS[@]}")
fi


job_idx=0
BASE_PORT=8000


for TRACK in "${TRACKS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        GPU_ID=$((job_idx % NUM_GPUS))
        NOTE="${CKPT_BASENAME}"
        port=$((BASE_PORT + GPU_ID))
        echo "[INFO] Submit JOB: ckpt=$CKPT, track=$TRACK, task=$TASK, gpu=$GPU_ID"

        CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_EGL_DEVICE_ID=$GPU_ID \
            python examples/vlabench/eval.py \
            --args.port $port \
            --args.eval_track $TRACK \
            --args.tasks $TASK \
            --args.n-episode $NUM_TRIALS \
            --args.save_dir $SAVE_DIR &

        job_idx=$((job_idx+1))
        while [ $(jobs -rp | wc -l) -ge $MAX_PROCS ]; do
            sleep 2
            wait -n
        done
    done
done

wait

python examples/vlabench/summarize.py
python scripts/gpu_runner.py

