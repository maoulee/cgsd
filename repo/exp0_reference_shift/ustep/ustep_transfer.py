#!/usr/bin/env python3
"""One-Step Transfer Test (U0–U3): does ONE GKD step move the bare Student's
action preference on HELD-OUT cases?

Question: reference-conditioned preference —GKD→ unconditioned policy
improvement?  Per case-split design (train cases never overlap held-out):

  conditions (teacher system suffix during the ONE training step):
    U0 identity  — no suffix (numerical baseline; FP32 path ⇒ ≈no-op)
    U1 answer    — answer-only
    U2 refined   — independent refined reference (best sibling, C2 form)
    U3 wrong     — another case's refined reference
  all: same 4B init, same train trajectories, same LoRA(r64,α128,lora_plus×4)
  + AdamW(lr 2e-5), ONE accumulated step, fresh optimizer per condition.

  TEACHER_SYNC env (user directive 2026-11-03): 'frozen' (default; teacher
  forward with adapter disabled — identical to synced at step 1 since the
  fresh LoRA is a no-op, meaningful from step 2 on) | 'synced'.

Metrics on held-out pairs (scored BARE, no reference):
  ΔM_transfer = M_θ1 − M_θ0,  M = s(a+) − s(a−)
  ΔlogP(a+), ΔlogP(a−) separately; KL-to-initial over held-out assistant
  spans (policy damage); train GKD KL (the loss actually optimized).

Usage:
  SEQ_PROMPT=C123 python scripts/ustep_transfer.py --gpu 0 \
      --out reports/ustep_transfer.json
"""
import argparse
import collections
import hashlib
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SEQ_PROMPT", "C123")
os.environ.setdefault("SEQ_ADAPTIVE_ROUTES", "1")

from exp0_reference_shift import (  # noqa: E402
    VERDICTS, ROLLOUT, PROMPT_SHA, REF_TMPL, label_of,
    assistant_full_text, build_user_msg, render)
from exp1c2_independent_ref import (  # noqa: E402
    parse_action, menu_before)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TOPK = 256
TEMP = 1.0
LR = 2e-5
LORA_PLUS = 4.0
CH = 512
TEACHER_SYNC = os.environ.get("TEACHER_SYNC", "frozen")


WALK_CACHE = {}


def walk(tok, system_base, tr, row, suffix="", cache_key=None):
    if cache_key and cache_key in WALK_CACHE:
        return WALK_CACHE[cache_key]
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
            msgs.append({"role": m.get("role"), "content": m.get("content"),
                         "name": m.get("name")})
        t = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=False)
        ids = tok(t, add_special_tokens=False).input_ids
        if m.get("role") == "assistant":
            spans.append((n_prev, len(ids)))
        n_prev = len(ids)
    if cache_key:
        WALK_CACHE[cache_key] = (ids, spans)
    return ids, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cap", type=int, default=2, help="train seqs per case")
    ap.add_argument("--out", default="reports/ustep_transfer.json")
    args = ap.parse_args()

    from kgqa.agent.seq_react_loop import seq_agents_md
    from build_teacher_prompts import render_fact_grouped, wrap_structured

    system_base = seq_agents_md()
    assert hashlib.sha256(system_base.encode()).hexdigest() == PROMPT_SHA

    verdicts = json.load(open(VERDICTS))
    rollout = json.load(open(ROLLOUT))
    by_key = {(r["case_id"], int(r["sample_idx"])): r for r in rollout}

    by_case = collections.defaultdict(list)
    for v in verdicts:
        by_case[v["case_id"]].append(v)
    # cases with ≥2 usable samples (need a sibling for the U2 reference)
    case_ok = {}
    for cid, vs in by_case.items():
        ok = sorted([v for v in vs if v.get("f1", 0) > 0 and
                     (v["case_id"], int(v["sample"])) in by_key],
                    key=lambda v: (-v.get("f1", 0), v["sample"]))
        if len(ok) >= 2:
            case_ok[cid] = ok
    cids = sorted(case_ok)
    held_out = {c for i, c in enumerate(cids) if i % 4 == 3}
    train_c = [c for c in cids if c not in held_out]
    print(f"[ustep] cases: {len(cids)} → train {len(train_c)} / "
          f"held-out {len(held_out)} | teacher={TEACHER_SYNC}", flush=True)

    # best-sibling reference per case (for U2), wrong-case ref (U3)
    ref_of, gold_of = {}, {}
    for cid in cids:
        v0 = case_ok[cid][0]
        ref_of[cid] = by_key[(cid, int(v0["sample"]))]["trajectory"]
        gold_of[cid] = v0["golds"] or ["unknown"]
    wrong_cid_of = {c: cids[(i + 1) % len(cids)]
                    for i, c in enumerate(cids)}

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

    def span_preds(spans):
        return sorted({p for lo, hi in spans for p in range(lo - 1, hi - 1)})

    def fwd_span(ids, preds, grad):
        ids_t = torch.tensor([ids], device=pm.device)
        am = torch.ones_like(ids_t)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            off = (TEACHER_SYNC == "frozen") and not grad
            if off:
                with pm.disable_adapter():
                    h = pm.base_model.model.model(
                        input_ids=ids_t, attention_mask=am).last_hidden_state
            else:
                h = pm.base_model.model.model(
                    input_ids=ids_t, attention_mask=am).last_hidden_state
            idx = torch.tensor(preds, device=h.device)
            hc = h[0].index_select(0, idx).unsqueeze(0)
            return pm.base_model.model.lm_head(hc)

    def kl_top256(s_logits, t_logits):
        tv, ti = t_logits.topk(TOPK, dim=-1)
        s_g = s_logits.gather(-1, ti)
        s_l = F.log_softmax(s_g.float() / TEMP, dim=-1)
        t_l = F.log_softmax(tv.float() / TEMP, dim=-1)
        return (t_l.exp() * (t_l - s_l)).sum(-1)

    def span_logp(ids, n_lo, n_hi, adapter_on=True):
        ids_t = torch.tensor([ids], device=pm.device)
        am = torch.ones_like(ids_t)
        with torch.no_grad():
            if adapter_on:
                h = pm.base_model.model.model(
                    input_ids=ids_t, attention_mask=am).last_hidden_state[0]
            else:
                with pm.disable_adapter():
                    h = pm.base_model.model.model(
                        input_ids=ids_t, attention_mask=am).last_hidden_state[0]
            out = []
            for s in range(n_lo - 1, n_hi - 1, CH):
                e = min(s + CH, n_hi - 1)
                lp = torch.log_softmax(
                    pm.base_model.model.lm_head(h[s:e]).float(), -1)
                out += lp.gather(1, ids_t[0, s + 1:e + 1]
                                 .unsqueeze(1)).squeeze(1).tolist()
        return out

    # ── train sequences (full trajectories, all assistant spans) ──────
    train_seqs = []
    for cid in train_c:
        for v in case_ok[cid][1:1 + args.cap]:      # sibling(s) as students
            row = by_key[(cid, int(v["sample"]))]
            sk = (cid, v["sample"], "student")
            ids, spans = walk(tok, system_base, row["trajectory"], row,
                              cache_key=sk)
            sufs = {
                "U0_identity": "",
                "U1_answer": REF_TMPL.format(
                    vals=" | ".join(v["golds"] or ["unknown"])),
                "U2_refined": "\n\n" + wrap_structured(render_fact_grouped(
                    ref_of[cid], answer=gold_of[cid], gold=gold_of[cid])),
                "U3_wrong": "\n\n" + wrap_structured(render_fact_grouped(
                    ref_of[wrong_cid_of[cid]],
                    answer=gold_of[wrong_cid_of[cid]],
                    gold=gold_of[wrong_cid_of[cid]])),
            }
            t_views = {}
            for cname, suf in sufs.items():
                if not suf or cname == "U0_identity":
                    t_views[cname] = (ids, spans)
                else:
                    t_views[cname] = walk(tok, system_base,
                                          row["trajectory"], row, suf,
                                          cache_key=(cid, v["sample"], cname))
            train_seqs.append({"cid": cid, "ids": ids, "spans": spans,
                               "sufs": sufs, "t_views": t_views})
    print(f"[ustep] train seqs: {len(train_seqs)}", flush=True)

    # ── held-out eval pairs (C2 construction, held-out cases only) ────
    eval_pairs = []
    for cid in sorted(held_out):
        for scored in case_ok[cid][1:]:
            tr = by_key[(cid, int(scored["sample"]))]["trajectory"]
            for b in scored["blocks"]:
                if label_of(b.get("verdict", "")) != "EFFECTIVE":
                    continue
                idx = int(b["idx"])
                if idx < 1 or tr[idx - 1].get("role") != "assistant":
                    continue
                act = parse_action(tr[idx - 1].get("content") or "")
                if not act:
                    continue
                rels = (act.get("args") or {}).get("relations") or []
                menu = menu_before(tr, idx)
                unchosen = [r for r in menu if r not in rels]
                if len(rels) < 1 or not unchosen:
                    continue
                neg = json.loads(json.dumps(act))
                neg["args"]["relations"] = [unchosen[0]] + rels[1:]
                thought = (tr[idx - 1].get("reasoning") or "").strip()
                hist = []
                for m in tr[:idx - 1]:
                    if m.get("role") == "assistant":
                        hist.append({"role": "assistant",
                                     "content": assistant_full_text(m)})
                    else:
                        hist.append({"role": m.get("role"),
                                     "content": m.get("content"),
                                     "name": m.get("name")})
                row = by_key[(cid, int(scored["sample"]))]
                eval_pairs.append({
                    "cid": cid,
                    "user": build_user_msg(row),
                    "hist": hist,
                    "think": (f"<think>\n{thought}\n</think>\n\n"
                              if thought else " "),
                    "a_pos": json.dumps(act, ensure_ascii=False),
                    "a_neg": json.dumps(neg, ensure_ascii=False),
                })
    print(f"[ustep] held-out pairs: {len(eval_pairs)}", flush=True)

    def score_bare(p, adapter_on):
        """mean logp of a candidate under the bare system (no reference)."""
        out = {}
        for cand in ("a_pos", "a_neg"):
            full_text = (f"{p['think']}{p[cand]}"
                         if p["think"].strip() != "" else p[cand])
            f_ids = render(tok, system_base, p["user"], p["hist"], full_text)
            c_ids = render(tok, system_base, p["user"], p["hist"],
                           p["think"] if p["think"].strip() != "" else " ")
            span = span_logp(f_ids, len(c_ids), len(f_ids),
                             adapter_on=adapter_on)
            out[cand] = sum(span) / len(span) if span else None
        return out

    # baseline θ0 scores (bare, adapter off — LoRA is a no-op at init,
    # but disable for exactness)
    base_scores = [score_bare(p, adapter_on=False) for p in eval_pairs]

    results = []
    conds = os.environ.get("USTEP_CONDS",
            "Z0_nostep,Z1_zerograd,Z2_idmean,Z3_id_prod").split(",")
    for cond in conds:
        for k in adapter0:
            pm.state_dict()[k].copy_(adapter0[k])
        pm.zero_grad(set_to_none=True)
        pm.train()
        eff_lr = LR * (1.0 / 5.0) if cond in ("Z3_id_prod",) else LR
        opt = torch.optim.AdamW(
            [{"params": [p_ for n_, p_ in pm.named_parameters()
                         if "lora_A" in n_], "lr": eff_lr},
             {"params": [p_ for n_, p_ in pm.named_parameters()
                         if "lora_B" in n_], "lr": eff_lr * LORA_PLUS}],
            lr=eff_lr, weight_decay=0.0, eps=1e-8)
        # N = total supervised tokens over ALL train seqs (token-mean, the
        # production loss normalization) — fixed per audit 2026-11-03
        all_preds = [span_preds(sq["spans"]) for sq in train_seqs]
        N = sum(len(p) for p in all_preds)
        tot_kl, tot_n = 0.0, 0
        for seq, preds in zip(train_seqs, all_preds):
            if cond in ("Z0_nostep", "Z1_zerograd"):
                break                       # controls: no backward at all
            s_logits = fwd_span(seq["ids"], preds, grad=True)
            if cond in ("U0_identity", "Z2_idmean", "Z3_id_prod"):
                t_logits = s_logits.detach()
            else:
                t_ids2, t_spans2 = seq["t_views"][cond]
                t_logits = fwd_span(t_ids2, span_preds(t_spans2), grad=False)
            kl = kl_top256(s_logits, t_logits)
            (kl.sum() / N).backward()       # token-mean (production form)
            tot_kl += float(kl.sum()); tot_n += int(kl.shape[1])
            del s_logits, t_logits
        if cond == "Z1_zerograd":
            for p_ in pm.parameters():
                p_.grad = None
            pm.zero_grad(set_to_none=True)
        grads = [p_.grad for _, p_ in pm.named_parameters()
                 if p_.grad is not None]
        gnorm = float(torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(g.float()) for g in grads]))) if grads else 0.0
        if grads:
            absf = torch.cat([g.abs().float().flatten() for g in grads])
            q = torch.quantile(absf, torch.tensor([0.5, 0.9],
                                  device=absf.device)) \
                if absf.numel() < 10_000_000 else None
            gq = {"p50": float(absf.median()),
                  "p90": float(q[1]) if q is not None else None,
                  "max": float(absf.max()),
                  "frac_gt_eps": float((absf > 1e-8).float().mean())}
        else:
            gq = {}
        if cond == "Z3_id_prod":
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in pm.parameters() if p_.grad is not None], 1.0)
        if cond != "Z0_nostep":
            opt.step()
        pm.eval()
        # held-out eval (bare, adapter ON)
        dm, dap, dan = [], [], []
        for p, b in zip(eval_pairs, base_scores):
            s = score_bare(p, adapter_on=True)
            if s["a_pos"] is None or b["a_pos"] is None:
                continue
            dm.append((s["a_pos"] - s["a_neg"]) - (b["a_pos"] - b["a_neg"]))
            dap.append(s["a_pos"] - b["a_pos"])
            dan.append(s["a_neg"] - b["a_neg"])
        rec = {"condition": cond, "train_kl_tokmean": tot_kl / max(tot_n, 1),
               "grad_norm": gnorm, "grad_quantiles": gq,
               "n_heldout": len(dm),
               "dM_transfer_mean": st.mean(dm) if dm else None,
               "d_logp_apos": st.mean(dap) if dap else None,
               "d_logp_aneg": st.mean(dan) if dan else None,
               "dM_pairs": dm,
               "pairs_detail": [{"case_id": eval_pairs[i]["cid"],
                                 "delta_margin": dm[i],
                                 "m_before": base_scores[i]["a_pos"] - base_scores[i]["a_neg"]}
                                for i in range(len(dm))]}
        results.append(rec)
        print(f"[ustep] {cond:11s} trainKL={rec['train_kl_tokmean']:.5f} "
              f"grad={gnorm:.3e} | held-out ΔM={rec['dM_transfer_mean']:+.5f} "
              f"(n={len(dm)}) ΔlogP+={rec['d_logp_apos']:+.5f} "
              f"ΔlogP−={rec['d_logp_aneg']:+.5f}", flush=True)

    # case-level bootstrap for each condition's ΔM
    for rec in results:
        by = collections.defaultdict(list)
        p_idx = {i: eval_pairs[i]["cid"] for i in range(len(eval_pairs))}
        for i, g in enumerate(rec["dM_pairs"]):
            by[p_idx[i]].append(g)
        cm = [st.mean(v) for v in by.values()]
        rng = random.Random(0)
        bs = sorted(sum(rng.choice(cm) for _ in cm) / len(cm)
                    for _ in range(10000))
        rec["dM_case_equal"] = st.mean(cm)
        rec["dM_CI95_case"] = [bs[250], bs[9750]]
        rec["dM_pos_cases"] = f"{sum(1 for x in cm if x > 0)}/{len(cm)}"
        rec.pop("dM_pairs", None)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[ustep] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
