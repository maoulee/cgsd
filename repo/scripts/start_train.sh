#!/bin/bash
# 训练容器专用入口：GTE 服务拉起 + 健康检查 + 训练启动，一个 sh 全包含。
#
# 与 run_swift_opsd.sh 的关系（用户要求不冲突）:
#   - 本脚本只负责"启动前环境"：GTE 检查/拉起、GPU 拓扑、CPU 核数适配；
#   - 训练本体全部委托 run_swift_opsd.sh（env > yaml > 默认，不变）；
#   - 若 GTE 已在运行（交互容器或重复启动），直接复用，不重复拉起。
#
# 用法（训练容器内）:
#   bash scripts/start_train.sh                     # GTE + 100 步训练
#   STEPS=2 bash scripts/start_train.sh             # 快速验证
#   bash scripts/start_train.sh --save_steps 5      # 透传 swift 参数
set -euo pipefail

REPO=.
PY=/opt/conda/envs/rl/bin/python
GTE_PORT=${GTE_PORT:-8003}
GTE_URL=http://localhost:${GTE_PORT}
GTE_LOG=${GTE_LOG:-$REPO/logs/gte_server.log}
# 位置参数优先: `bash scripts/start_train.sh 2` = 跑 2 步（环境不支持 VAR= 前缀
# 时的唯一入口）; 其后参数继续透传给 swift。
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  STEPS="$1"; shift
else
  STEPS=${STEPS:-${OPSD_STEPS:-100}}
fi

mkdir -p "$REPO/logs"

# ---- 1) GTE 健康检查（model_loaded 必须为 true）---------------------------
gte_health() {
  curl -sf -m 3 "$GTE_URL/health" 2>/dev/null || return 1
}

gte_ready() {
  local h; h=$(gte_health) || return 1
  grep -q '"model_loaded": *true' <<< "$h"
}

# GTE 部署（2026-10-24 裁定修正: 必须留在 GPU）: 0.6B embedding CPU 前向
# ≈ 20 TFLOP/批（64 文本×256 token），4 核实测 400s/批、32 核也要 ~50s —
# 远超 1s 工具窗口，CPU 不可行（曾误判"绰绰有余"，已撤回）。
# GPU 共存挤压的防护 = 服务端请求切块 GTE_MAX_REQ_ITEMS=64（gte_api_server.py，
# 2026-10-24）: 批形状稳定 → cuda caching allocator 复用既有块 → 不在训练
# 显存峰值期新申请 → 不 OOM。并发回 8（max_num_seqs）也削薄了突发。
export GTE_DEVICE=${GTE_DEVICE:-balanced}   # 双卡摊平（用户设计: 降单卡峰值）

if gte_ready; then
  echo "[start_train] GTE 已在运行且模型已加载，复用: $GTE_URL"
else
  if gte_health >/dev/null 2>&1; then
    echo "[start_train] FATAL: GTE 端口有响应但模型未加载（model_loaded!=true）" >&2
    echo "[start_train]        查看日志: tail -50 $GTE_LOG；杀掉重启: pkill -f gte_api_server" >&2
    exit 1
  fi
  echo "[start_train] GTE 未运行，拉起: port=$GTE_PORT device=$GTE_DEVICE log=$GTE_LOG"
  nohup env GTE_DEVICE="$GTE_DEVICE" "$PY" "$REPO/scripts/gte_api_server.py" --port "$GTE_PORT" \
    > "$GTE_LOG" 2>&1 &
  GTE_PID=$!
  echo "[start_train] GTE pid=$GTE_PID，等待模型加载（最长 300s）..."
  for i in $(seq 1 150); do
    if gte_ready; then
      echo "[start_train] GTE 就绪 (${i}×2s): $(gte_health)"
      break
    fi
    if ! kill -0 "$GTE_PID" 2>/dev/null; then
      echo "[start_train] FATAL: GTE 进程退出，见 $GTE_LOG" >&2
      tail -30 "$GTE_LOG" >&2 || true
      exit 1
    fi
    sleep 2
  done
  gte_ready || { echo "[start_train] FATAL: GTE 300s 内未就绪" >&2; exit 1; }
fi

# ---- 2) GPU 拓扑检查（对齐用户裁定: 两卡 rollout + 两卡训练）---------------
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NGPU" -lt 2 ]]; then
  echo "[start_train] WARNING: 只检测到 $NGPU 卡（预期 2）；将降级 TP${NGPU}" >&2
fi

# ---- 3) CPU 配额确认（WALK_POOL 由 run_swift_opsd.sh 按 cgroup 自适应）----
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r Q P < /sys/fs/cgroup/cpu.max
  [[ "$Q" != "max" ]] && echo "[start_train] CPU 配额: $((Q/P)) 核 (WALK_POOL 将按此自适应)"
fi

# ---- 4) 训练（全部委托原 launcher；env 优先级不变）--------------------------
export OPSD_GPUS=${OPSD_GPUS:-$(seq -s, 0 $((NGPU-1)))}
export OPSD_STEPS="$STEPS"
echo "[start_train] 启动训练: gpus=$OPSD_GPUS steps=$STEPS"
exec bash "$REPO/scripts/run_swift_opsd.sh" "$@"
