#!/usr/bin/env python3
"""SEQ offline-GRPO trainer entry (separate from SAPS).

Consumes the SEQ trainer-ready JSONL (kgqa/rl/seq_advantage.py reallocate output:
records with `messages` + `turn_advantages` + `advantage`). Tokenizes via
`_tokenize_seq` (per-turn advantage over <|im_start|> spans, format-agnostic) and
trains with the GENERIC OfflineGRPOTrainer + per-stage loss (imported from the SAPS
trainer module — shared infrastructure, not modified).

Adaptive granularity is baked into turn_advantages upstream:
  coarse (single-SG): connectivity+p0 → A_info/A_ans
  fine   (multi-SG) : c_i (LOO)       → A_plan/per-SG/A_ans

Run (mirrors scripts/run_routed.sh but for SEQ):
  accelerate launch --config_file configs/zero2_lora.yaml --num_processes 2 \
    kgqa/rl/seq_train_grpo.py --dataset /tmp/seq_train_1000x8.jsonl \
    --model models/Qwen3.5-9B --output checkpoint/seq_grpo
"""
import os, sys, argparse

# import the GENERIC trainer machinery from the SAPS trainer module (shared, unmodified)
_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
sys.path.insert(0, _SCRIPTS)
from train_offline_grpo import (  # noqa: E402
    OfflineGRPOTrainer, _make_routed_collator as _make_scalar_collator,
    _make_per_stage_collator, _patch_apply_chat_template,
)

import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from trl import SFTConfig  # noqa: E402
from peft import LoraConfig  # noqa: E402
from datasets import load_dataset  # noqa: E402


def _tokenize_seq(tokenizer, messages, turn_advantages, max_length):
    """SEQ tokenizer for SCALAR (limer) backend. Labels assistant turns (train only
    on assistant tokens). Computes a SCALAR per-example advantage = mean of
    turn_advantages (the block-level decomposition, already conservation-balanced).
    This avoids the per_stage double-forward OOM — limer fuses GRPO loss into
    a single forward with NO logits materialization."""
    txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(txt, add_special_tokens=False)
    if len(ids) > max_length:
        return None
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    starts = [i for i, t in enumerate(ids) if t == im_start]
    spans = [(starts[k], starts[k + 1] if k + 1 < len(starts) else len(ids)) for k in range(len(starts))]
    if len(spans) != len(messages):
        return "MISMATCH"
    labels = [-100] * len(ids)
    ai = 0
    for (s, e), m in zip(spans, messages):
        if m.get("role") != "assistant":
            continue
        for t in range(s, e):
            labels[t] = ids[t]
        ai += 1
    # scalar advantage = mean of turn advantages (conservation: Σa_t = A_k → mean = A_k/n_turns,
    # but we want the TOTAL signal per example, so use the SUM which equals A_k)
    scalar_adv = sum(turn_advantages) if turn_advantages else 0.0
    return {"input_ids": ids, "labels": labels, "advantage": scalar_adv}


def _tokenize_seq_perturn(tokenizer, messages, turn_advantages, max_length):
    """Per-TURN advantage tokenizer (gradient-decomposable credit): the k-th
    ASSISTANT message's tokens uniformly carry turn_advantages[k] — the per-reply
    design. Same <|im_start|> span alignment as _tokenize_per_stage; advantages
    come from the record's turn list (two-axis v0 recomputation), not stage
    mapping. Returns None (too long) or a skip-reason string."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=False)
    # trl×transformers combo returns list[int] | list[list[int]] | list[Encoding]
    # depending on patch state — unwrap all three
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if isinstance(ids, list) and ids and hasattr(ids[0], "ids"):
        ids = list(ids[0].ids)
    ids = list(ids)
    if len(ids) > max_length:
        return None
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    starts = [i for i, t in enumerate(ids) if t == im_start]
    spans = [(starts[k], starts[k + 1] if k + 1 < len(starts) else len(ids))
             for k in range(len(starts))]
    if len(spans) != len(messages):
        return "MISMATCH"
    asst_idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if len(asst_idxs) != len(turn_advantages):
        return "ADV_MISMATCH"
    labels = [-100] * len(ids)
    per_tok = [0.0] * len(ids)
    for k, mi in enumerate(asst_idxs):
        s, e = spans[mi]
        adv = float(turn_advantages[k])
        for t in range(s, e):
            labels[t] = ids[t]
            per_tok[t] = adv
    return {"input_ids": ids, "labels": labels, "per_token_advantage": per_tok}


def _prepare_seq_dataset(tokenizer, ds, max_length, drop_zero_adv=False):
    """Pre-tokenize SEQ records → {input_ids, labels, advantage} for SCALAR limer backend."""
def _prepare_seq_dataset(tokenizer, ds, max_length, drop_zero_adv=False,
                         per_turn=False):
    """Pre-tokenize SEQ records. per_turn=False → scalar limer backend
    {input_ids, labels, advantage}; per_turn=True → per-token advantage vector
    {input_ids, labels, per_token_advantage} consumed by the per-stage loss
    (gradient-decomposable per-reply credit)."""
    from collections import Counter

    def _tok_scalar(ex):
        r = _tokenize_seq(tokenizer, ex["messages"], ex["turn_advantages"], max_length)
        if not isinstance(r, dict):
            return {"input_ids": [], "labels": [], "advantage": 0.0, "_skip": (r or "too_long")}
        return {"input_ids": r["input_ids"], "labels": r["labels"],
                "advantage": r["advantage"], "_skip": ""}

    def _tok_perturn(ex):
        r = _tokenize_seq_perturn(tokenizer, ex["messages"], ex["turn_advantages"], max_length)
        if not isinstance(r, dict):
            return {"input_ids": [], "labels": [], "per_token_advantage": [],
                    "_skip": (r or "too_long")}
        return {"input_ids": r["input_ids"], "labels": r["labels"],
                "per_token_advantage": r["per_token_advantage"], "_skip": ""}

    n_in = len(ds)
    if drop_zero_adv:
        # zero scalar advantage = exactly-zero gradient (sum of turn advantages);
        # keeping them only burns forward/backward compute
        ds = ds.filter(lambda r: abs(float(r.get("advantage") or 0)) > 1e-6)
        print(f"  zero-adv drop: {len(ds)}/{n_in} kept")
        n_in = len(ds)
    ds = ds.map(_tok_perturn if per_turn else _tok_scalar, num_proc=4)
    reasons = Counter(r for r in ds["_skip"] if r)
    ds = ds.filter(lambda r: len(r["input_ids"]) > 0, num_proc=4)
    print(f"  seq tokenize ({'per-turn' if per_turn else 'scalar'}): kept {len(ds)}/{n_in} "
          f"(dropped: {dict(reasons)})")
    # length column for Trainer's LengthGroupedSampler (group_by_length=True)
    ds = ds.map(lambda r: {"length": len(r["input_ids"])}, num_proc=4)
    keep = {"input_ids", "labels", "length"}
    keep.add("per_token_advantage" if per_turn else "advantage")
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--output", default="checkpoint/seq_grpo")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--no-gc", action="store_true", default=False)
    ap.add_argument("--gc-reentrant", action="store_true", default=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--optim", default="adamw_torch_fused",
                    help="adamw_bnb_8bit lost on container reset; fused default")
    ap.add_argument("--per-stage-chunk", type=int, default=1024,
                    help="token chunk for the per-turn lm_head log_softmax; "
                         "long trajectories OOM at 1024 (float logits copy ≈ "
                         "chunk×V×4B), 384 fits 40GB GPUs at max_len 14336")
    ap.add_argument("--init-adapter", default="",
                    help="path to a trained adapter (e.g. checkpoint/seq_grpo_v15) "
                         "whose LoRA weights initialize this run — iterated RL: "
                         "round-2 training CONTINUES from the round-1 adapter "
                         "instead of starting from fresh LoRA")
    ap.add_argument("--drop-zero-adv", action="store_true", default=False,
                    help="drop samples whose recomputed advantage is exactly 0 — "
                         "zero gradient, pure wasted compute (17%% of display-mode "
                         "v6 data, 1%% of tf-mode)")
    ap.add_argument("--per-turn", action="store_true", default=False,
                    help="per-TURN credit: tokenize turn_advantages onto assistant "
                         "token spans and train via the per-stage per-token loss "
                         "(gradient-decomposable; ~2.5x slower/step than scalar)")
    args = ap.parse_args()

    print(f"=== SEQ offline GRPO: {args.dataset} ===")
    ds = load_dataset("json", data_files=args.dataset, split="train")
    print(f"  {len(ds)} samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _patch_apply_chat_template(tokenizer)
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    ds = _prepare_seq_dataset(tokenizer, ds, args.max_length,
                               drop_zero_adv=args.drop_zero_adv, per_turn=args.per_turn)

    peft_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"],
                             task_type="CAUSAL_LM")
    from transformers import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True)
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
        apply_liger_kernel_to_qwen3_5(model=model, fused_linear_cross_entropy=False,
                                      rms_norm=True, swiglu=True)
        print("  Applied Liger RMSNorm + SwiGLU patches")
    except Exception as e:
        print(f"  [warn] liger patch failed ({e})")

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio, max_length=args.max_length,
        completion_only_loss=False, assistant_only_loss=False,
        logging_steps=args.log_steps, save_steps=args.save_steps, save_total_limit=2,
        bf16=True, gradient_checkpointing=not args.no_gc,
        gradient_checkpointing_kwargs={"use_reentrant": args.gc_reentrant},
        optim=args.optim, report_to="none", packing=False,
        dataset_num_proc=4, remove_unused_columns=False)
    # this trl version's SFTConfig filters unknown ctor kwargs — inject the
    # length-grouping attrs post-construction (Trainer reads them as attributes)
    sft_config.group_by_length = True
    sft_config.length_column_name = "length"

    trainer = OfflineGRPOTrainer(
        model=model, args=sft_config, train_dataset=ds, processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=(_make_per_stage_collator(pad_id) if args.per_turn
                       else _make_scalar_collator(pad_id)),
        loss_backend="liger", per_stage=args.per_turn,
        per_stage_chunk=args.per_stage_chunk)
    print(f"\n=== Offline GRPO (SEQ, per-stage): {len(ds)} samples ===")
    if args.init_adapter:
        # iterated RL: warm-start this run's LoRA from the round-1 adapter.
        # Key mapping: saved adapters use '<module>.lora_A.weight'; the live
        # PeftModel state_dict uses '<module>.lora_A.default.weight' (adapter
        # name). Without the rename, 0/256 tensors match and the "continuation"
        # silently starts from a FRESH LoRA (caught in the v16 attempt).
        from safetensors.torch import load_file
        sd_raw = load_file(os.path.join(args.init_adapter, "adapter_model.safetensors"))
        sd = {}
        for k, v in sd_raw.items():
            nk = k.replace(".lora_A.weight", ".lora_A.default.weight") \
                  .replace(".lora_B.weight", ".lora_B.default.weight")
            sd[nk] = v
        missing, unexpected = trainer.model.load_state_dict(sd, strict=False)
        model_keys = set(trainer.model.state_dict().keys())
        lora_loaded = sum(1 for k in sd if k in model_keys)
        print(f"  init-adapter ← {args.init_adapter}: {lora_loaded}/{len(sd)} tensors "
              f"loaded (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
        assert lora_loaded >= len(sd) * 0.9, \
            f"adapter warm-start failed: only {lora_loaded}/{len(sd)} keys matched"
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"done → {args.output}")


if __name__ == "__main__":
    main()
