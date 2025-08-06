#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
source setup_env.sh
set -e  # 遇到错误立即退出

usage() {
    echo "Usage: $0 <config_name> <checkpoint_path> [--track <track_name>] [--task <task_name>]"
    echo ""
    echo "Arguments:"
    echo "  config_name          配置名称"
    echo "  checkpoint           模型路径"
    echo ""
    echo "Options:"
    echo "  --track <track_name> 指定track (可选)"
    echo "  --task <task_name>   指定task (可选)"
    echo ""
    echo "Example:"
    echo "  $0 my_model_config ckpt"
    echo "  $0 my_model_config ckpt --track track_1_in_distribution --task add_condiment"
    exit 1
}

# 检查参数数量
if [ "$#" -lt 1 ]; then
    usage
fi

# 解析参数
CONFIG_NAME=$1
CKPT=$2
shift 2

TRACK_OPT=""
TASK_OPT=""

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
        --help|-h)
            usage
            ;;
        *)
            echo "未知参数: $1"
            usage
            ;;
    esac
done

# 配置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_SCRIPT="${SCRIPT_DIR}/serve_policy.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/multi_run_vlabench.sh"

# 检查脚本是否存在
if [ ! -f "$POLICY_SCRIPT" ]; then
    echo "错误: Policy server脚本不存在: $POLICY_SCRIPT"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "错误: Evaluation脚本不存在: $EVAL_SCRIPT"
    exit 1
fi

# 生成save_dir
model_step="${CKPT##*/}"
SAVE_DIR="evaluate_results/${CONFIG_NAME}/model_${model_step}"

echo "=========================================="
echo "统一评估脚本启动"
echo "=========================================="
echo "配置名称: $CONFIG_NAME"
echo "保存目录: $SAVE_DIR"
if [[ -n "$TRACK_OPT" ]]; then
    echo "指定Track: $TRACK_OPT"
fi
if [[ -n "$TASK_OPT" ]]; then
    echo "指定Task: $TASK_OPT"
fi
echo "=========================================="

# 创建日志目录
LOG_DIR="logs/unified_eval_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "日志目录: $LOG_DIR"

# 清理函数
cleanup() {
    echo ""
    echo "=========================================="
    echo "正在清理进程..."
    echo "=========================================="
    
    # 终止policy server进程
    echo "终止Policy Server进程..."
    pkill -f "serve_policy.py" || true
    
    # 终止evaluation进程
    echo "终止Evaluation进程..."
    pkill -f "vlabench/eval.py" || true
    
    echo "清理完成"
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# ==================== 第一步：启动Policy Servers ====================
echo ""
echo "=========================================="
echo "第一步: 启动Policy Servers"
echo "=========================================="

echo "执行命令: bash $POLICY_SCRIPT $CONFIG_NAME $CKPT"

# 在后台启动policy servers，并重定向输出到日志文件
bash "$POLICY_SCRIPT" "$CONFIG_NAME" "$CKPT"> "${LOG_DIR}/policy_servers.log" 2>&1 &
POLICY_PID=$!

echo "Policy servers正在启动... (PID: $POLICY_PID)"
echo "日志文件: ${LOG_DIR}/policy_servers.log"

# ==================== 等待Policy Servers启动 ====================
echo ""
echo "=========================================="
echo "等待Policy Servers完全启动..."
echo "=========================================="

WAIT_TIME=120
echo "等待 ${WAIT_TIME} 秒..."

for i in $(seq 1 $WAIT_TIME); do
    printf "\r等待中: %2d/${WAIT_TIME} 秒" $i
    sleep 1
done
echo ""

# 检查policy servers是否启动成功
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
BASE_PORT=8000
echo "验证Policy Servers状态..."

all_ready=true
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    port=$((BASE_PORT + gpu_id))
    if curl -s --connect-timeout 2 --max-time 5 "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "  ✓ GPU ${gpu_id} (端口 ${port}): 运行中"
    else
        echo "  ✗ GPU ${gpu_id} (端口 ${port}): 未响应"
        all_ready=false
    fi
done

if [ "$all_ready" = false ]; then
    echo "警告: 部分Policy Servers可能未正确启动，但继续执行evaluation..."
fi

# ==================== 第二步：启动Evaluation ====================
echo ""
echo "=========================================="
echo "第二步: 启动Evaluation"
echo "=========================================="

# 构建evaluation命令
EVAL_CMD="bash $EVAL_SCRIPT $SAVE_DIR"
if [[ -n "$TRACK_OPT" ]]; then
    EVAL_CMD="$EVAL_CMD --track $TRACK_OPT"
fi
if [[ -n "$TASK_OPT" ]]; then
    EVAL_CMD="$EVAL_CMD --task $TASK_OPT"
fi

echo "执行命令: $EVAL_CMD"

# 启动evaluation并等待完成
$EVAL_CMD > "${LOG_DIR}/evaluation.log" 2>&1
EVAL_EXIT_CODE=$?

# ==================== 结果处理 ====================
echo ""
echo "=========================================="
echo "评估完成"
echo "=========================================="

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation执行成功"
else
    echo "✗ Evaluation执行失败 (退出码: $EVAL_EXIT_CODE)"
fi

echo "日志目录: $LOG_DIR"
echo "  - Policy servers日志: ${LOG_DIR}/policy_servers.log"
echo "  - Evaluation日志: ${LOG_DIR}/evaluation.log"

# ==================== 清理 ====================
echo ""
echo "=========================================="
echo "清理资源"
echo "=========================================="

echo "终止Policy Server进程..."
kill $POLICY_PID 2>/dev/null || true
pkill -f "serve_policy.py" || true

echo "等待进程完全退出..."
sleep 5

echo ""
echo "=========================================="
echo "评估流程完成!"
echo "=========================================="
echo "配置: $CONFIG_NAME"
echo "结果保存在: $SAVE_DIR"
echo "日志保存在: $LOG_DIR"

# python scripts/gpu_runner.py