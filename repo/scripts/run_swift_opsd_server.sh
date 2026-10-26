#!/bin/bash
# ms-swift OPSD (GKD/JSD) server 模式 — vLLM 服务独占一卡 (AsyncEngine), 训练独占一卡
#
# 与 run_swift_opsd.sh (colocate) 的唯一结构差异:
#   - rollout 由 `swift rollout` 常驻服务执行, 多轮 scheduler 跑在服务端,
#     每条轨迹在 AsyncEngine 上独立推进 (无 run_multi_turn 的轮级屏障)
#   - 训练侧不再持有 vLLM 引擎, 也不需要 sleep/wake; LoRA 每步通过 HTTP 同步
#   - 代价: 训练只有 1 卡 (colocate 是 2 卡), 训练相位会变慢
#
# 用法:
#   OPSD_DATA=/path/to/train.jsonl bash scripts/run_swift_opsd_server.sh
set -euo pipefail

REPO=.
PY=/opt/conda/envs/rl/bin/python
SWIFT=/opt/conda/envs/rl/bin/swift
PLUGIN=$REPO/kgqa/rl/swift_plugins/kgqa_scheduler.py

DEFAULT_DATA=$REPO/data/opsd/opsd_kgqa_teacher19.jsonl
DATA=${OPSD_DATA:-$DEFAULT_DATA}
if [[ ! -f "$DATA" ]]; then
  echo "missing OPSD_DATA=$DATA" >&2
  exit 1
fi
MODEL=${OPSD_MODEL:-models/Qwen3.5-4B}
EXP=${OPSD_EXP:-opsd_gkd_server}
OUT=${OPSD_OUT:-/tmp/$EXP}
STEPS=${OPSD_STEPS:-3}
# server 模式训练只有 1 卡: ACC=32 → 32 条/step（与 colocate ACC=16×2 卡对齐）
ACC=${OPSD_ACC:-32}
TOPK=${OPSD_TOPK:-256}
BETA=${OPSD_BETA:-0.5}
MAXLEN=${OPSD_MAX_LEN:-32768}
COMPLETION=${OPSD_COMPLETION:-1500}
SEED=${OPSD_SEED:-42}
PORT=${OPSD_ROLL_PORT:-8020}
SERVER_GPU=${OPSD_SERVER_GPU:-0}
TRAIN_GPU=${OPSD_TRAIN_GPU:-1}
UTIL=${OPSD_VLLM_UTIL:-0.85}
SERVER_LOG=${OPSD_SERVER_LOG:-$REPO/logs/rollout_server.log}

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export GTE_API_URLS=${GTE_API_URLS:-http://localhost:8003}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export MAX_ROUNDS=${MAX_ROUNDS:-24}
export SEQ_ADAPTIVE_ROUTES=${SEQ_ADAPTIVE_ROUTES:-1}
export SEQ_PROMPT=${SEQ_PROMPT:-RC2}
export KGQA_ROLLOUT_TEMP=${KGQA_ROLLOUT_TEMP:-0.7}
export KGQA_ROLLOUT_TOP_P=${KGQA_ROLLOUT_TOP_P:-0.8}
export KGQA_ROLLOUT_TOP_K=${KGQA_ROLLOUT_TOP_K:-20}
export KGQA_ROLLOUT_PRESENCE=${KGQA_ROLLOUT_PRESENCE:-1.5}
export KGQA_ROLLOUT_REPETITION=${KGQA_ROLLOUT_REPETITION:-1.0}
export KGQA_ROLLOUT_SEED=${KGQA_ROLLOUT_SEED:-}
export KGQA_THINKING_BUDGET=${KGQA_THINKING_BUDGET:-1000}
export KGQA_DROP_UNANSWERED=${KGQA_DROP_UNANSWERED:-0}
export KGQA_TEACHER_PREFIX=${KGQA_TEACHER_PREFIX:-system}
export KGQA_TRAIN_MAX_LEN=${KGQA_TRAIN_MAX_LEN:-$MAXLEN}
export KGQA_DYNAMIC_SAMPLE=${KGQA_DYNAMIC_SAMPLE:-1}
export KGQA_GROUP_BY_CASE=${KGQA_GROUP_BY_CASE:-1}
# server 模式的 rollout 是独立进程，可安全做 fresh resampling。
export KGQA_DYNAMIC_RESAMPLE=${KGQA_DYNAMIC_RESAMPLE:-1}
export KGQA_RESAMPLE_TRIES=${KGQA_RESAMPLE_TRIES:-2}
export KGQA_ROLLOUT_MAX_LEN=${KGQA_ROLLOUT_MAX_LEN:-$MAXLEN}
export KGQA_ALIGN_CHECK=${KGQA_ALIGN_CHECK:-1}
export SEQ_LICENSE_FILTER=${SEQ_LICENSE_FILTER:-1}
export SEQ_MULTISTEP=${SEQ_MULTISTEP:-1}
export SEQ_PLAN_PATH_HINTS=${SEQ_PLAN_PATH_HINTS:-1}
export SEQ_RENDER_V38=${SEQ_RENDER_V38:-1}
export SEQ_RENDER_V4=${SEQ_RENDER_V4:-1}
export WALK_BATCH_WINDOW=${WALK_BATCH_WINDOW:-1.0}
export WALK_POOL=${WALK_POOL:-3}
export KGQA_SCHEDULER=kgqa_opsd
export KGQA_TURN_SPLIT=0
export KGQA_TITO_STRIP_PREFIX=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ---- 1) rollout 服务 (GPU0) ------------------------------------------------
CUDA_VISIBLE_DEVICES="$SERVER_GPU" nohup "$SWIFT" rollout \
  --model "$MODEL" \
  --multi_turn_scheduler kgqa_opsd --max_turns "$MAX_ROUNDS" \
  --vllm_use_async_engine true --vllm_enable_lora true \
  --vllm_max_model_len "$MAXLEN" --vllm_gpu_memory_utilization "$UTIL" \
  --max_length "$MAXLEN" \
  --attn_impl sdpa --torch_dtype bfloat16 \
  --external_plugins "$PLUGIN" \
  --host 127.0.0.1 --port "$PORT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "rollout server pid=$SERVER_PID gpu=$SERVER_GPU port=$PORT log=$SERVER_LOG"

# ---- 2) 等待服务就绪 -------------------------------------------------------
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "rollout server ready"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "rollout server died; see $SERVER_LOG" >&2
    tail -40 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || {
  echo "rollout server not ready after 600s" >&2; exit 1; }

# ---- 3) 训练 (GPU1) --------------------------------------------------------
# 不用 exec: trap 需要在训练退出后回收 rollout 服务
# 不传 --teacher_model：同路径 + LoRA 会变成 disable_adapter 冻结基座。
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" NPROC_PER_NODE=1 "$SWIFT" rlhf --rlhf_type gkd \
  --model "$MODEL" \
  --beta "$BETA" --lmbda 1.0 --gkd_logits_topk "$TOPK" \
  --dataset "$DATA" \
  --dataset_shuffle false --train_dataloader_shuffle false \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 \
  --attn_impl sdpa --torch_dtype bfloat16 \
  --max_length "$MAXLEN" --max_completion_length "$COMPLETION" \
  --completion_length_limit_scope per_round \
  --use_vllm true --vllm_mode server \
  --vllm_server_host 127.0.0.1 --vllm_server_port "$PORT" \
  --vllm_enable_lora true --vllm_server_timeout 600 \
  --offload_optimizer true --deepspeed zero2 \
  --gradient_checkpointing true \
  --external_plugins "$PLUGIN" \
  --multi_turn_scheduler kgqa_opsd --remove_unused_columns false \
  --per_device_train_batch_size 1 --gradient_accumulation_steps "$ACC" \
  --num_generations 1 \
  --max_steps "$STEPS" --logging_steps 1 --save_steps 1000 \
  --seed "$SEED" \
  --output_dir "$OUT" --report_to none "$@"
