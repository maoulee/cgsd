#!/usr/bin/env python3
"""E-Zero strict: I0/I1/I2/I3 identity matrix + T1/T2 signal rerun.

Fixes vs ezero_identity_test.py per the 2026-11-03 audit:
  1. FP32 softmax (`.float()`), matching the real chunked GKD loss;
     I3 keeps BF16 to reproduce the original anomaly.
  2. Shifted labels: span [lo,hi) supervised from hidden positions
     [lo-1, hi-1) (the old harness was off by one).
  3. Token-mean loss normalization (real loss), gradients accumulated
     across sequences then one backward.
  4. Gradient-direction comparison: pairwise cosine over flattened LoRA
     grads (not just norms).

Matrix (all FP32 unless noted, teacher adapter per row):
  I0  teacher = same forward's logits.detach()   → math control, grad≡0
  I1  two forwards, same weights, adapter ON     → dual-forward divergence
  I2  two forwards, teacher adapter OFF          → adapter-path isolation
  I3  = I2 but BF16 softmax                      → reproduce old T0
  T1  answer-only suffix, teacher adapter ON     → signal reference
  T2  refined reference suffix, adapter ON       → real-OPSD-form signal

Usage:
  SEQ_PROMPT=C123 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/ezero_strict.py --n 5 --gpu 0
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SEQ_PROMPT", "C123")
os.environ.setdefault("SEQ_ADAPTIVE_ROUTES", "1")

from exp0_reference_shift import (  # noqa: E402
    VERDICTS, ROLLOUT, PROMPT_SHA, REF_TMPL, assistant_full_text,
    build_user_msg)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

TOPK = 256
TEMP = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="reports/ezero_strict.json")
    args = ap.parse_args()

    from kgqa.agent.seq_react_loop import seq_agents_md
    from build_teacher_prompts import render_fact_grouped, wrap_structured
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    system_base = seq_agents_md()
    import hashlib
    assert hashlib.sha256(system_base.encode()).hexdigest() == PROMPT_SHA

    verdicts = json.load(open(VERDICTS))
    rollout = json.load(open(ROLLOUT))
    by_key = {(r["case_id"], int(r["sample_idx"])): r for r in rollout}
    vrows, seen = [], set()
    for v in sorted([v for v in verdicts if v.get("f1", 0) > 0 and
                     (v["case_id"], int(v["sample"])) in by_key],
                    key=lambda v: v["case_id"]):
        if v["case_id"] in seen:
            continue
        seen.add(v["case_id"]); vrows.append(v)
        if len(vrows) >= args.n:
            break

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}", trust_remote_code=True)
    model.eval()
    pm = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    pm.gradient_checkpointing_enable()
    pm.enable_input_require_grads()
    adapter0 = {k: v.detach().clone() for k, v in pm.state_dict().items()
                if "lora_" in k}

    def walk(tr, row, suffix=""):
        msgs = [{"role": "system", "content": system_base + suffix},
                {"role": "user", "content": build_user_msg(row)}]
        t0 = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=False)
        n_prev = len(tok(t0, add_special_tokens=False).input_ids)
        spans = []
        for m in tr:
            if m.get("role") == "assistant":
                msgs.append({"role": "assistant",
                             "content": assistant_full_text(m)})
            else:
                msgs.append({"role": m.get("role"),
                             "content": m.get("content"), "name": m.get("name")})
            t = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=False)
            ids = tok(t, add_special_tokens=False).input_ids
            if m.get("role") == "assistant":
                spans.append((n_prev, len(ids)))
            n_prev = len(ids)
        return ids, spans

    seqs = []
    for v in vrows:
        row = by_key[(v["case_id"], int(v["sample"]))]
        tr = row["trajectory"]
        golds = v["golds"] or ["unknown"]
        ids, spans = walk(tr, row)
        seqs.append({"key": (v["case_id"], int(v["sample"])),
                     "row": row, "tr": tr, "ids": ids, "spans": spans,
                     "golds": golds,
                     "t2": "\n\n" + wrap_structured(render_fact_grouped(
                         tr, answer=golds, gold=golds)),
                     "t1": REF_TMPL.format(vals=" | ".join(golds))})
    print(f"[ez2] {len(seqs)} seqs", flush=True)

    def span_preds(spans):
        # SHIFTED labels: token range [lo,hi) ← hidden positions [lo-1,hi-1)
        return sorted({p for lo, hi in spans for p in range(lo - 1, hi - 1)})

    def fwd_span_logits(ids, preds, grad, teacher_off=False):
        ids_t = torch.tensor([ids], device=pm.device)
        am = torch.ones_like(ids_t)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            if teacher_off:
                with pm.disable_adapter():
                    h = pm.base_model.model.model(
                        input_ids=ids_t, attention_mask=am).last_hidden_state
            else:
                h = pm.base_model.model.model(
                    input_ids=ids_t, attention_mask=am).last_hidden_state
            idx = torch.tensor(preds, device=h.device)
            hc = h[0].index_select(0, idx).unsqueeze(0)
            logits = pm.base_model.model.lm_head(hc)
        return logits

    def kl_top256(s_logits, t_logits, fp32=True):
        tv, ti = t_logits.topk(TOPK, dim=-1)
        s_g = s_logits.gather(-1, ti)
        if fp32:
            s_g, tv = s_g.float(), tv.float()
        s_l = F.log_softmax(s_g / TEMP, dim=-1)
        t_l = F.log_softmax(tv / TEMP, dim=-1)
        return (t_l.exp() * (t_l - s_l)).sum(-1)   # [1, P]

    def run(name, mode, suffix_key=None, fp32=True):
        for k in adapter0:
            pm.state_dict()[k].copy_(adapter0[k])
        for p in pm.parameters():
            p.grad = None
        pm.zero_grad(set_to_none=True)
        pm.train()
        tot_kl, tot_n = 0.0, 0
        for seq in seqs:
            preds = span_preds(seq["spans"])
            suffix = seq[suffix_key] if suffix_key else ""
            t_ids, t_spans = (walk(seq["tr"], seq["row"], suffix)
                              if suffix else (seq["ids"], seq["spans"]))
            if mode == "I0":            # same forward, detached teacher
                s_logits = fwd_span_logits(seq["ids"], preds, grad=True)
                t_logits = s_logits.detach()
            else:
                s_logits = fwd_span_logits(seq["ids"], preds, grad=True)
                t_logits = fwd_span_logits(
                    t_ids, span_preds(t_spans), grad=False,
                    teacher_off=(mode in ("I2", "I3")))
            kl = kl_top256(s_logits, t_logits, fp32=(mode != "I3"))
            n = int(kl.shape[1])
            (kl.sum() / 1.0).backward()          # accumulate token sums
            tot_kl += float(kl.sum()); tot_n += n
            del s_logits, t_logits
        grads = []
        for _, p in pm.named_parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().float().flatten())
        gvec = torch.cat(grads) if grads else None
        gnorm = float(gvec.norm()) if gvec is not None else 0.0
        rec = {"condition": name, "mode": mode, "fp32": mode != "I3",
               "kl_top256_tokmean": tot_kl / max(tot_n, 1),
               "grad_norm": gnorm, "n_tokens": tot_n,
               "grad_vec": gvec}
        print(f"[ez2] {name:16s} KL={rec['kl_top256_tokmean']:.8f} "
              f"grad={gnorm:.4e} n={tot_n}", flush=True)
        return rec

    recs = [
        run("I0_same_forward", "I0"),
        run("I1_two_forward", "I1"),
        run("I2_adapter_off", "I2"),
        run("I3_bf16_repro", "I3"),
        run("T1_answer", "T1", suffix_key="t1"),
        run("T2_refined", "T2", suffix_key="t2"),
    ]
    # gradient-direction comparison (drop the big vectors from the dump)
    import math
    cos = {}
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            if a["grad_vec"] is None or b["grad_vec"] is None:
                continue
            na, nb = a["grad_vec"].norm(), b["grad_vec"].norm()
            if na > 0 and nb > 0:
                cos[f"{a['condition']}|{b['condition']}"] = round(
                    float((a["grad_vec"] @ b["grad_vec"]) / (na * nb)), 4)
    for r in recs:
        r.pop("grad_vec")
    out = {"results": recs, "grad_cosine": cos}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[ez2] wrote {args.out}")
    print("[ez2] grad cosines:", json.dumps(cos, indent=0)[:600])


if __name__ == "__main__":
    main()
