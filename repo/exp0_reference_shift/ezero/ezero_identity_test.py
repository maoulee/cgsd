#!/usr/bin/env python3
"""E-Zero: Identity Self-Distillation Test + minimal T1/T2 causal audit.

Question (Codex 2026-11-02): if teacher and student distributions were truly
identical, pure KL self-distillation gives zero gradient — so what actually
pushes OPSD updates? Engineering prerequisite: verify the REAL training loss
(chunked top-256 FKL, beta=0, T=1.0, kgqa_scheduler._chunk_loss form) on:

  T0 identity — teacher = student exactly (step-0 reality: fresh LoRA is a
                no-op, teacher runs with adapter disabled)  → must be ~0
  T1 answer   — teacher system += [Reference] Verified answer (exp0 form)
  T2 refined  — teacher system += wrap_structured(render_fact_grouped(...))
                (the ACTUAL OPSD pool teacher reference)

Per condition, on real r9a trajectories (C123 prefix, loss on assistant
spans), with 4B + LoRA(r64, α128, lora_plus×4 on B):
  loss (= top-256 FKL), full-vocab KL (diagnostic), grad_norm over LoRA,
  one AdamW step (lr 2e-5) → ||Δθ||₂, and policy drift (mean logp change of
  assistant spans on a held probe sequence).

Usage:
  SEQ_PROMPT=C123 python scripts/ezero_identity_test.py --n 8 --gpu 0
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
    build_user_msg, render)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

TOPK = 256
TEMP = 1.0
LR = 2e-5
LORA_PLUS = 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=8, help="trajectories")
    ap.add_argument("--out", default="reports/ezero_identity.json")
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
    vrows = []
    seen = set()
    for v in sorted([v for v in verdicts if v.get("f1", 0) > 0 and
                     (v["case_id"], int(v["sample"])) in by_key],
                    key=lambda v: v["case_id"]):
        if v["case_id"] in seen:         # one sample per case
            continue
        seen.add(v["case_id"]); vrows.append(v)
        if len(vrows) >= args.n:
            break

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}", trust_remote_code=True)
    model.eval()
    lcfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    pm = get_peft_model(model, lcfg)
    pm.gradient_checkpointing_enable()
    pm.enable_input_require_grads()
    adapter0 = copy.deepcopy(
        {k: v for k, v in pm.state_dict().items() if "lora_" in k})

    # ── build sequences: ids + assistant-span loss mask ────────────────
    seqs = []
    for v in vrows:
        row = by_key[(v["case_id"], int(v["sample"]))]
        tr = row["trajectory"]
        user = build_user_msg(row)
        golds = v["golds"] or ["unknown"]
        # full multi-turn conversation, loss on ALL assistant spans
        msgs = [{"role": "system", "content": system_base},
                {"role": "user", "content": user}]
        spans = []          # (start_tok, end_tok) per assistant turn
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
        n_prev = len(tok(text, add_special_tokens=False).input_ids)
        for i, m in enumerate(tr):
            if m.get("role") == "assistant":
                msgs.append({"role": "assistant",
                             "content": assistant_full_text(m)})
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)
                ids = tok(t, add_special_tokens=False).input_ids
                spans.append((n_prev, len(ids)))
                n_prev = len(ids)
            else:
                msgs.append({"role": m.get("role"), "content": m.get("content"),
                             "name": m.get("name")})
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)
                n_prev = len(tok(t, add_special_tokens=False).input_ids)
        ids = tok(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False),
            add_special_tokens=False).input_ids
        t2_inject = wrap_structured(render_fact_grouped(
            tr, answer=golds, gold=golds))
        seqs.append({
            "case_id": v["case_id"], "sample": int(v["sample"]),
            "ids": ids, "spans": spans,
            "suffix": {
                "t1": REF_TMPL.format(vals=" | ".join(golds)),
                "t2": "\n\n" + t2_inject,
            },
        })
    print(f"[ezero] {len(seqs)} sequences, len "
          f"p50={sorted(len(s['ids']) for s in seqs)[len(seqs)//2]}", flush=True)

    meta_of = {}
    for v in vrows:
        meta_of[(v["case_id"], int(v["sample"]))] = v

    def rebuild(seq, suffix):
        """Re-render the same trajectory walk with a modified system prompt;
        returns (ids, assistant_spans)."""
        v = meta_of[(seq["case_id"], seq["sample"])]
        row = by_key[(seq["case_id"], seq["sample"])]
        tr = row["trajectory"]
        msgs = [{"role": "system", "content": system_base + suffix},
                {"role": "user", "content": build_user_msg(row)}]
        spans, n_prev = [], 0
        t0 = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=False)
        n_prev = len(tok(t0, add_special_tokens=False).input_ids)
        for m in tr:
            if m.get("role") == "assistant":
                msgs.append({"role": "assistant",
                             "content": assistant_full_text(m)})
            else:
                msgs.append({"role": m.get("role"), "content": m.get("content"),
                             "name": m.get("name")})
            t = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=False)
            ids = tok(t, add_special_tokens=False).input_ids
            if m.get("role") == "assistant":
                spans.append((n_prev, len(ids)))
            n_prev = len(ids)
        return ids, spans

    # ── loss: chunked top-256 FKL over assistant spans (training form) ──
    def span_chunk_kl(s_logits, t_logits, tgt_span, want_full=False):
        """s_logits/t_logits: [1, T, V] slices at positions predicting span.
        Returns (kl_top256_mean, kl_full_mean|None)."""
        K = TOPK
        tv, ti = t_logits.topk(K, dim=-1)
        s_g = s_logits.gather(-1, ti)
        s_l = F.log_softmax(s_g / TEMP, dim=-1)
        t_l = F.log_softmax(tv / TEMP, dim=-1)
        kl_k = (t_l.exp() * (t_l - s_l)).sum(-1)      # FKL teacher||student
        if not want_full:
            return kl_k.mean(), None
        s_full = F.log_softmax(s_logits / TEMP, dim=-1)
        t_full = F.log_softmax(t_logits / TEMP, dim=-1)
        kl_f = (t_full.exp() * (t_full - s_full)).sum(-1)
        return kl_k.mean(), kl_f.mean()

    CH = 512

    # NOTE on identity path: with grad we must keep the graph; chunking with
    # detach breaks grad. For E-Zero scale (span preds ~2-4K positions) the
    # full-span logits tensor is feasible: [1, P, V] bf16.
    def run_condition(name, suffix_key, probe_seq):
        # reset adapter
        sd = pm.state_dict()
        for k in adapter0:
            sd[k].copy_(adapter0[k])
        opt = torch.optim.AdamW(
            [{"params": [p for n_, p in pm.named_parameters()
                         if "lora_A" in n_], "lr": LR},
             {"params": [p for n_, p in pm.named_parameters()
                         if "lora_B" in n_], "lr": LR * LORA_PLUS}],
            lr=LR, weight_decay=0.0)
        tot_kl, tot_full, n_tok = 0.0, 0.0, 0
        pm.train()
        opt.zero_grad(set_to_none=True)
        for seq in seqs:
            s_ids, s_spans = seq["ids"], seq["spans"]
            suffix = seq["suffix"].get(suffix_key, "") if suffix_key else ""
            t_ids, t_spans = (rebuild(seq, suffix) if suffix
                              else (seq["ids"], seq["spans"]))
            ids_s = torch.tensor([s_ids], device=pm.device)
            ids_t = torch.tensor([t_ids], device=pm.device)
            am_s, am_t = torch.ones_like(ids_s), torch.ones_like(ids_t)
            preds_s = sorted({p for lo, hi in s_spans for p in range(lo, hi)})
            idx_s = torch.tensor(preds_s, device=pm.device)
            with torch.enable_grad():
                h_s = pm.base_model.model.model(
                    input_ids=ids_s, attention_mask=am_s).last_hidden_state
                hc = h_s[0].index_select(0, idx_s).unsqueeze(0)
                s_logits = pm.base_model.model.lm_head(hc)
            with torch.no_grad():
                with pm.disable_adapter():
                    h_t = pm.base_model.model.model(
                        input_ids=ids_t, attention_mask=am_t).last_hidden_state
                    preds_t = sorted({p for lo, hi in t_spans
                                      for p in range(lo, hi)})
                    idx_t = torch.tensor(preds_t, device=pm.device)
                    hc_t = h_t[0].index_select(0, idx_t).unsqueeze(0)
                    t_logits = pm.base_model.model.lm_head(hc_t)
            kl_k, kl_f = span_chunk_kl(s_logits, t_logits, None, want_full=True)
            loss = kl_k * len(preds_s)
            loss.backward()
            tot_kl += float(kl_k) * len(preds_s)
            tot_full += float(kl_f) * len(preds_s)
            n_tok += len(preds_s)
            del s_logits, t_logits, h_s, h_t
        grads = [p.grad for _, p in pm.named_parameters() if p.grad is not None]
        gnorm = float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.float()) for g in grads]))) \
            if grads else 0.0
        opt.step()
        d2 = 0.0
        for n_, p in pm.named_parameters():
            if "lora_" in n_ and p.requires_grad:
                with torch.no_grad():
                    d2 += float((p.float() - adapter0[n_].float()
                                 .to(p.device)).pow(2).sum())
        dtheta = d2 ** 0.5
        # policy drift probe: mean logp change on probe spans
        seq = probe_seq
        ids = torch.tensor([seq["ids"]], device=pm.device)
        am = torch.ones_like(ids)
        preds = sorted({p for lo, hi in seq["spans"] for p in range(lo, hi)})
        idxp = torch.tensor(preds, device=pm.device)
        drift = None
        with torch.no_grad():
            h = pm.base_model.model.model(input_ids=ids,
                                          attention_mask=am).last_hidden_state
            hc = h[0].index_select(0, idxp).unsqueeze(0)
            lg = pm.base_model.model.lm_head(hc)[0].float()
            lp_new = torch.log_softmax(lg, -1).gather(
                -1, ids[0][idxp + 1].unsqueeze(-1)).squeeze(-1).mean()
            with pm.disable_adapter():
                h0 = pm.base_model.model.model(
                    input_ids=ids, attention_mask=am).last_hidden_state
                hc0 = h0[0].index_select(0, idxp).unsqueeze(0)
                lg0 = pm.base_model.model.lm_head(hc0)[0].float()
                lp_old = torch.log_softmax(lg0, -1).gather(
                    -1, ids[0][idxp + 1].unsqueeze(-1)).squeeze(-1).mean()
            drift = float(lp_new - lp_old)
        rec = {"condition": name, "n_seqs": len(seqs), "n_span_tokens": n_tok,
               "loss_top256_kl": tot_kl / max(n_tok, 1),
               "full_kl_diag": tot_full / max(n_tok, 1),
               "grad_norm_lora": gnorm, "dtheta_l2": dtheta,
               "probe_logp_drift": drift}
        print(f"[ezero] {name:10s} KL256={rec['loss_top256_kl']:.6f} "
              f"KLfull={rec['full_kl_diag']:.6f} grad={gnorm:.3e} "
              f"|dθ|={dtheta:.3e} drift={drift:+.5f}", flush=True)
        return rec

    probe = seqs[0]
    results = [
        run_condition("T0_identity", "", probe),
        run_condition("T1_answer", "t1", probe),
        run_condition("T2_refined", "t2", probe),
    ]
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[ezero] wrote {args.out}")


if __name__ == "__main__":
    main()
