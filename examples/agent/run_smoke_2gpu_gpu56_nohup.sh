#!/usr/bin/env bash
# 真正的后台 2 卡冒烟：nohup 忽略 SIGHUP，关 IDE/SSH 断开（在未杀进程前提下）仍可继续跑。
#
# 使用前在**当前 shell** 中 export（勿把 key 写进仓库）：
#   export CUDA_VISIBLE_DEVICES=5,6
#   export LLM_AS_A_JUDGE_BASE="https://yunwu.ai/v1"
#   export LLM_AS_A_JUDGE_API_KEY="sk-..."
#   export LLM_AS_A_JUDGE_MODEL="qwen3.5-plus"   # 可选，有默认值
#
# 启动：
#   bash examples/agent/run_smoke_2gpu_gpu56_nohup.sh
#
# 看日志：
#   tail -f logs/nohup_smoke_2gpu_<时间戳>.log
#   # 训练本身仍会 tee 到 logs/smoke_2gpu_gpu56_subset512.log
#
# 停跑（慎用）：
#   kill "$(cat logs/nohup_smoke_2gpu_latest.pid)" && ray stop --force

set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?请先 export CUDA_VISIBLE_DEVICES，例如 5,6}"
: "${LLM_AS_A_JUDGE_API_KEY:?请先 export LLM_AS_A_JUDGE_API_KEY}"

export WORLD_SIZE="${WORLD_SIZE:-1}"
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-https://yunwu.ai/v1}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-qwen3.5-plus}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
NOHUP_LOG="${LOG_DIR}/nohup_smoke_2gpu_${STAMP}.log"
PIDFILE="${LOG_DIR}/nohup_smoke_2gpu_latest.pid"
LATEST_LINK="${LOG_DIR}/nohup_smoke_2gpu_latest.log"

ln -sf "${NOHUP_LOG}" "${LATEST_LINK}"

cd "${REPO}"

# nohup + 子 shell：与启动终端脱钩；env 显式传入敏感变量，避免写进磁盘脚本
nohup env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}" \
  LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE}" \
  LLM_AS_A_JUDGE_API_KEY="${LLM_AS_A_JUDGE_API_KEY}" \
  LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL}" \
  PYTHONUNBUFFERED=1 \
  bash -c '
    set -euo pipefail
    source "${HOME}/deepeyes_raid_env.sh"
    export PATH="/data/lsj/deepeyes/conda_env/bin:${PATH}"
    cd "'"${REPO}"'"
    exec bash examples/agent/local_smoke_2gpu_gpu56.sh
  ' >>"${NOHUP_LOG}" 2>&1 &

echo $! >"${PIDFILE}"
echo "[nohup-smoke-2gpu] 已启动"
echo "  PID 文件 : ${PIDFILE}  -> $(cat "${PIDFILE}")"
echo "  nohup 日志: ${NOHUP_LOG}  (软链: ${LATEST_LINK})"
echo "  训练 tee  : ${LOG_DIR}/smoke_2gpu_gpu56_subset512.log"
