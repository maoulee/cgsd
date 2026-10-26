#!/bin/bash
# ms-swift OPSD (GKD/JSD) 单入口启动器（自包含）。
#
# 配置来源优先级: OPSD_* 环境变量 > config/opsd_train.yaml > 脚本内默认。
# GPU 数 / vLLM TP / 训练 rank 数自动跟随环境可见卡数（不进 yaml）。
# 每步 F1 写入 $OUT/f1_steps.jsonl（插件侧，含 masked 统计）。
#
# 用法:
#   bash scripts/run_swift_opsd.sh                      # 读 config/opsd_train.yaml
#   OPSD_STEPS=2 bash scripts/run_swift_opsd.sh         # 单项覆盖
#   bash scripts/run_swift_opsd.sh --save_steps 5       # 透传 swift 参数
set -euo pipefail

REPO=.
PY=/opt/conda/envs/rl/bin/python
SWIFT=/opt/conda/envs/rl/bin/swift
CFG=${OPSD_CFG:-$REPO/config/opsd_train.yaml}

# ---- 读 YAML → CFG_<KEY>（扁平化；缺文件/缺键静默走默认）------------------
if [[ -f "$CFG" ]]; then
  eval "$("$PY" - "$CFG" <<'PYEOF'
import sys, yaml, shlex
try:
    d = yaml.safe_load(open(sys.argv[1])) or {}
except Exception:
    d = {}
def flat(x, p=""):
    for k, v in x.items():
        key = f"{p}_{k}".upper() if p else str(k).upper()
        if isinstance(v, dict):
            yield from flat(v, key)
        elif v is not None:
            yield key, v
for k, v in flat(d):
    print(f"CFG_{k}={shlex.quote(str(v))}")
PYEOF
)"
else
  echo "[launcher] WARN config $CFG 不存在，全部走内置默认" >&2
fi
cfg() { local v="CFG_$1"; [[ -n "${!v:-}" ]] && echo "${!v}" || echo "$2"; }

# ---- GPU 拓扑对齐（用户裁定 2026-10-24: 两卡 rollout + 两卡训练）----------
# 目标拓扑: rollout = 1 个 TP{NGPU} vLLM 引擎横跨全部卡；训练 = NGPU 个
# rank ZeRO-2 各占一卡。容器调度器常把 CUDA_VISIBLE_DEVICES 限制成单卡，
# 这会把整条训练静默降级成 TP1 单卡——OPSD_GPUS 显式夺回（最高优先）。
if [[ -n "${OPSD_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$OPSD_GPUS"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPU=$(( $(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES") ))
else
  NGPU=$(nvidia-smi -L | wc -l)
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NGPU-1)))"
fi
_AVAIL=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$_AVAIL" -gt 0 && "$NGPU" -lt "$_AVAIL" ]]; then
  echo "[launcher] WARNING: 宿主可见 $_AVAIL 卡但 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 只暴露 $NGPU 卡" >&2
  echo "[launcher]   → 降级为 TP${NGPU} 单引擎 / ${NGPU} rank 训练" >&2
  echo "[launcher]   → 修复: OPSD_GPUS=0,1 bash scripts/run_swift_opsd.sh 或 unset CUDA_VISIBLE_DEVICES" >&2
  if [[ "${OPSD_REQUIRE_GPUS:-}" -gt "$NGPU" ]] 2>/dev/null; then
    echo "[launcher] FATAL: OPSD_REQUIRE_GPUS=${OPSD_REQUIRE_GPUS} 但只有 $NGPU 卡可用" >&2
    exit 1
  fi
fi
# vLLM 引擎 TP 数，默认 = 全部可见卡。NPROC_PER_NODE 须能被 TP 整除。
VLLM_TP=${OPSD_VLLM_TP:-$NGPU}
export NPROC_PER_NODE=${NPROC_PER_NODE:-$NGPU}
echo "[launcher] topology: rollout=1xTP${VLLM_TP} engine on ${NGPU} GPU | training=${NPROC_PER_NODE} ranks ZeRO-2 on ${NGPU} GPU"
# 注意力: auto = flash_attn 可导入则 flash_attention_2，否则 sdpa
#（flash-attn 无 torch2.11+cu130 预编译轮；容器装入后自动升级）。
if [[ "${OPSD_ATTN_IMPL:-auto}" == "auto" ]]; then
  if "$PY" -c "import flash_attn" 2>/dev/null; then ATTN=flash_attention_2; else ATTN=sdpa; fi
else
  ATTN=${OPSD_ATTN_IMPL}
fi

# ---- 基本参数（env > yaml > 默认）----------------------------------------
MODEL=${OPSD_MODEL:-$(cfg MODEL models/Qwen3.5-4B)}
DATA=${OPSD_DATA:-$(cfg DATA $REPO/data/opsd/opsd_kgqa_teacher19.jsonl)}
[[ -f "$DATA" ]] || { echo "missing data: $DATA" >&2; exit 1; }
EXP=${OPSD_EXP:-$(cfg EXP opsd_gkd)}
OUT=${OPSD_OUT:-$REPO/outputs/$EXP}   # 用户裁定: 运行产物落 outputs/（容器重置不丢）
# 默认目录复用会让两次 run 的 f1_steps.jsonl / rollouts/stepNNN 混在一起
# （历史上出现过 step001 里 100 个文件 = 两个 run）。未显式指定 OPSD_OUT 且
# 目录已有 f1 记录时，自动加时间后缀，保证一次 run 一个目录。
if [[ -z "${OPSD_OUT:-}" && -e "$OUT/f1_steps.jsonl" ]]; then
  OUT="$OUT-$(date +%m%d-%H%M)"
  echo "[launcher] OUT 已存在旧记录 → 本次使用 $OUT"
fi
STEPS=${OPSD_STEPS:-$(cfg STEPS 3)}
BS=${OPSD_BS:-$(cfg BS 1)}
ACC=${OPSD_ACC:-$(cfg ACC 16)}
LR=${OPSD_LR:-$(cfg LR '')}
WARMUP=${OPSD_WARMUP:-$(cfg WARMUP_STEPS '')}
SAVESTEPS=${OPSD_SAVE_STEPS:-$(cfg SAVE_STEPS 1000)}
SEED=${OPSD_SEED:-$(cfg SEED 42)}
TOPK=${OPSD_TOPK:-$(cfg TOPK 256)}
BETA=${OPSD_BETA:-$(cfg BETA 0.0)}
TEMPERATURE=${OPSD_TEMPERATURE:-$(cfg TEMPERATURE '')}
UTIL=${OPSD_VLLM_UTIL:-$(cfg VLLM_UTIL 0.65)}
MAXLEN=${OPSD_MAX_LEN:-$(cfg MAX_LEN 49152)}   # 模型原生 256k；见 yaml KV 账注释
COMPLETION=${OPSD_COMPLETION:-$(cfg COMPLETION 1500)}
MAX_SEQS=${OPSD_MAX_NUM_SEQS:-$(cfg MAX_NUM_SEQS 16)}
PREFIX_CACHING=${OPSD_PREFIX_CACHING:-$(cfg PREFIX_CACHING 1)}
LORA_RANK=${OPSD_LORA_RANK:-$(cfg LORA_RANK 64)}
LORA_ALPHA=${OPSD_LORA_ALPHA:-$(cfg LORA_ALPHA 128)}
export KGQA_LORA_PLUS=${KGQA_LORA_PLUS:-$(cfg LORA_PLUS 16)}

mkdir -p "$OUT"
echo "[launcher] cfg=$CFG gpus=$NGPU vllm_tp=$VLLM_TP attn=$ATTN" \
     "steps=$STEPS acc=$ACC batch=$((BS*ACC*NGPU)) maxlen=$MAXLEN" \
     "lr=${LR:-default} out=$OUT"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export GTE_API_URLS=${GTE_API_URLS:-http://localhost:8003}
# GTE 批量上限（2026-10-25）: vLLM 的 KV 预分配后固定, 训练期峰值波动来自
# GTE 的编码激活（并发越高 drain 合并的文本越多）。GTE 不是瓶颈 → 收紧批量
# 换显存余量: 每次 forward 32 条、每次 drain 合并 ≤64 条（原 64/256）。
export GTE_ENCODE_BATCH=${GTE_ENCODE_BATCH:-32}
export GTE_BATCH_EARLY=${GTE_BATCH_EARLY:-64}

# ---- 9B 对齐环境（与 config/batches/opd_step.yaml 同口径）-----------------
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export MAX_ROUNDS=${MAX_ROUNDS:-24}
export SEQ_ADAPTIVE_ROUTES=${SEQ_ADAPTIVE_ROUTES:-1}
export SEQ_PROMPT=${SEQ_PROMPT:-RC2}
export SEQ_LICENSE_FILTER=${SEQ_LICENSE_FILTER:-1}
export SEQ_MULTISTEP=${SEQ_MULTISTEP:-1}
export SEQ_PLAN_PATH_HINTS=${SEQ_PLAN_PATH_HINTS:-1}
export SEQ_RENDER_V38=${SEQ_RENDER_V38:-1}
export SEQ_RENDER_V4=${SEQ_RENDER_V4:-1}
export WALK_BATCH_WINDOW=${WALK_BATCH_WINDOW:-1.0}
# walk 车道 = 独立 spawn 进程（真并行，无 GIL）。256 规模下每 (rank,轮)
# ~48 个并发工具调用、工具耗时总和最高 791s，3 车道是瓶颈；按核数自适应
# （4 核容器→3，32 核容器→24）。24 而非 32: 主进程+2×vLLM EngineCore+GTE
# ~5 常驻进程，24+5 ≈ 29 < 32 核不超卖；每步 32 个不同 case = 车道并行上限。
_walk_default="auto"
_walk_cfg=$(cfg WALK_POOL $_walk_default)
# CPU 配额从 cgroup 读（nproc 报宿主核数，容器里是错的）
_cpu_quota() {
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    local q p; read -r q p < /sys/fs/cgroup/cpu.max
    [[ "$q" != "max" ]] && echo $(( q / p )) && return
  fi
  if [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
    local q p; q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [[ "$q" -gt 0 ]] && echo $(( q / p )) && return
  fi
  echo "$(nproc)"
}
if [[ -n "${WALK_POOL:-}" ]]; then
  :  # env 显式指定，尊重
elif [[ "$_walk_cfg" == "auto" ]]; then
  _NCPU=$(_cpu_quota)
  if   [[ "$_NCPU" -ge 32 ]]; then export WALK_POOL=24
  elif [[ "$_NCPU" -ge 12 ]]; then export WALK_POOL=$(( _NCPU / 2 ))
  else                           export WALK_POOL=3
  fi
else
  export WALK_POOL="$_walk_cfg"
fi

# ---- rollout 采样（生产 9B 口径）------------------------------------------
export KGQA_ROLLOUT_TEMP=${KGQA_ROLLOUT_TEMP:-$(cfg ROLLOUT_TEMP 0.7)}
export KGQA_ROLLOUT_TOP_P=${KGQA_ROLLOUT_TOP_P:-$(cfg ROLLOUT_TOP_P 0.8)}
export KGQA_ROLLOUT_TOP_K=${KGQA_ROLLOUT_TOP_K:-$(cfg ROLLOUT_TOP_K 20)}
export KGQA_ROLLOUT_PRESENCE=${KGQA_ROLLOUT_PRESENCE:-$(cfg ROLLOUT_PRESENCE 1.5)}
export KGQA_ROLLOUT_REPETITION=${KGQA_ROLLOUT_REPETITION:-1.0}
export KGQA_ROLLOUT_SEED=${KGQA_ROLLOUT_SEED:-}
export KGQA_THINKING_BUDGET=${KGQA_THINKING_BUDGET:-$(cfg THINKING_BUDGET 1000)}

# ---- OPSD 机制 ------------------------------------------------------------
export KGQA_SCHEDULER=kgqa_opsd
export KGQA_TURN_SPLIT=0
export KGQA_TITO_STRIP_PREFIX=1
export KGQA_TEACHER_PREFIX=${KGQA_TEACHER_PREFIX:-$(cfg TEACHER_PREFIX system)}
export KGQA_TRAIN_MAX_LEN=${KGQA_TRAIN_MAX_LEN:-$MAXLEN}
export KGQA_ROLLOUT_MAX_LEN=$MAXLEN
export KGQA_DYNAMIC_SAMPLE=1
export KGQA_GROUP_BY_CASE=1
export KGQA_DYN_MODE=${KGQA_DYN_MODE:-$(cfg DYN_MODE mask)}
export KGQA_DYNAMIC_RESAMPLE=${KGQA_DYNAMIC_RESAMPLE:-0}
export KGQA_RESAMPLE_TRIES=${KGQA_RESAMPLE_TRIES:-2}
export KGQA_ALIGN_CHECK=1
export KGQA_CHUNKED_LOSS=${KGQA_CHUNKED_LOSS:-$(cfg CHUNKED_LOSS 1)}
export KGQA_LOSS_CHUNK=${KGQA_LOSS_CHUNK:-$(cfg LOSS_CHUNK 512)}
export KGQA_SESSION_DRAIN_CLOSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 静音 triton autotune 逐配置打印（autotune 本身照常运行+缓存，只关日志）
export TRITON_PRINT_AUTOTUNING=0
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
# 每步 [phase]（rollout/train 阶段耗时）+ [tool]（游走 lane / GTE 批量并发
# 统计：exec vs wait、slots/memo 去重）——训练容器确认工具层并发的唯一窗口
export KGQA_PHASE_PROFILE=${KGQA_PHASE_PROFILE:-1}
# 每步 F1 + masked 统计落盘（jsonl，便于画曲线）
export KGQA_F1_FILE="$OUT/f1_steps.jsonl"
# 每步 rollout 全量落盘（用户裁定 2026-10-24: step2 崩塌取证）:
# <dir>/step{N}/rank{R}/<case>__{uid}.md+.json —— md 是人读审计视图
# （thinking/action/tool 分栏, 工具回包截断），json 逐字保真。
# 轮次表含每轮 tool_s（工具层耗时应激增 = GTE 死亡的现场证据）。
# 体量 ~5-20MB/step（32 轨迹），100 步 ≈ 1-2GB。KGQA_DUMP_DIR="" 可关。
export KGQA_DUMP_DIR=${KGQA_DUMP_DIR:-$OUT/rollouts}

EXTRA=(--vllm_max_num_seqs "$MAX_SEQS"
       --vllm_enable_prefix_caching "$([[ $PREFIX_CACHING == 1 ]] && echo true || echo false)"
       --save_steps "$SAVESTEPS")
[[ "$VLLM_TP" != "1" ]] && EXTRA+=(--vllm_tensor_parallel_size "$VLLM_TP")
[[ -n "$LR" ]] && EXTRA+=(--learning_rate "$LR")
[[ -n "$WARMUP" ]] && EXTRA+=(--warmup_steps "$WARMUP")
[[ -n "$TEMPERATURE" ]] && EXTRA+=(--temperature "$TEMPERATURE")

exec "$SWIFT" rlhf --rlhf_type gkd \
  --model "$MODEL" \
  --beta "$BETA" --lmbda 1.0 --gkd_logits_topk "$TOPK" \
  --dataset "$DATA" \
  --dataset_shuffle false --train_dataloader_shuffle false \
  --tuner_type lora --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" \
  --attn_impl "$ATTN" --torch_dtype bfloat16 \
  --max_length "$MAXLEN" --max_completion_length "$COMPLETION" \
  --completion_length_limit_scope ${OPSD_LEN_SCOPE:-per_round} \
  --use_vllm true --vllm_mode colocate \
  --vllm_max_model_len "$MAXLEN" --vllm_gpu_memory_utilization "$UTIL" \
  --sleep_level 2 --offload_optimizer true --offload_model true --deepspeed zero2 \
  --gradient_checkpointing true \
  --external_plugins "$REPO/kgqa/rl/swift_plugins/kgqa_scheduler.py" \
  --multi_turn_scheduler kgqa_opsd --remove_unused_columns false \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$ACC" \
  --num_generations 1 \
  --max_steps "$STEPS" --logging_steps 1 \
  --seed "$SEED" \
  --output_dir "$OUT" --report_to none "${EXTRA[@]}" "$@"
