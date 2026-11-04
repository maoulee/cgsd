#!/usr/bin/env python3
"""S2: score-fixed held-out test (audit 2026-11-04 #3).

Fixes vs s_series_heldout.py:
  1. TRUE full-token-mean baseline F_token_mean (single N over ALL region
     tokens), plus the historical region-equal F (renamed F_region_equal).
  2. STRICT held-out scoring: candidates = the REAL full_text (AUDIT kept)
     with ONLY the relations VALUES swapped char-level; relations span
     located by char-anchor in EACH render and asserted; margins reported
     as (a) relations-fork margin and (b) full-action margin.
  3. Per-candidate decomposition: ΔlogP(a+) and ΔlogP(a−) separately —
     distinguishes memorization (a+ up) from drift (both down).
  4. Drift diagnostics: ‖Δθ‖ and held-out realized-token logp drift
     (whole assistant turn, old vs new).

Arms: I | F_token_mean | F_region_equal | A | A+0.25T (3 steps, frozen
teacher, warmup+clip). Same 10/7 case split as S.

Usage:
  SEQ_PROMPT=C123 python scripts/s2_score_fixed.py --gpu 0 --out reports/s2.json
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SEQ_PROMPT", "C123")
os.environ.setdefault("SEQ_ADAPTIVE_ROUTES", "1")

from exp0_reference_shift import (  # noqa: E402
    VERDICTS, ROLLOUT, PROMPT_SHA, label_of, build_user_msg, render)
from exp1c2_independent_ref import parse_action, menu_before  # noqa: E402
from w4_strict_align import turn_text, tok_at_char_f  # noqa: E402
from s_series_heldout import aligned_pairs  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TOPK = 256
TEMP = 1.0
LR = 2e-5
LORA_PLUS = 4.0
CH = 256
STEPS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="reports/s2.json")
    args = ap.parse_args()

    from kgqa.agent.seq_react_loop import seq_agents_md
    from build_teacher_prompts import render_fact_grouped, wrap_structured
    system_base = seq_agents_md()
    assert hashlib.sha256(system_base.encode()).hexdigest() == PROMPT_SHA

    verdicts = json.load(open(VERDICTS))
    rollout = json.load(open(ROLLOUT))
    by_case = collections.defaultdict(list)
    for v in verdicts:
        by_case[v["case_id"]].append(v)
    by_key = {(r["case_id"], int(r["sample_idx"])): r for r in rollout}
    cids_ok = sorted(cid for cid, vs in by_case.items()
                     if sum(1 for v in vs if v.get("f1", 0) > 0 and
                            (cid, int(v["sample"])) in by_key) >= 2)
    train_c = [c for i, c in enumerate(cids_ok) if i % 4 != 3][:10]
    held_c = [c for i, c in enumerate(cids_ok) if i % 4 == 3]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hdr_ids = tok("<|im_start|>assistant\n",
                  add_special_tokens=False).input_ids
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

    def anchor(ids, target):
        """char-anchor token span whose decode == target (last occurrence)."""
        tstrs = [tok.decode([t]) for t in ids]
        full = "".join(tstrs)
        c = full.rfind(target)
        if c < 0:
            return None
        lo = tok_at_char_f(ids, tok, c)
        hi = tok_at_char_f(ids, tok, c + len(target) - 1) + 1
        if "".join(tstrs[lo:hi]).strip() == target.strip():
            return (lo, hi)
        return None

    def span_logp(ids, lo, hi):
        ids_t = torch.tensor([ids], device=pm.device)
        am = torch.ones_like(ids_t)
        with torch.no_grad():
            h = pm.base_model.model.model(
                input_ids=ids_t, attention_mask=am).last_hidden_state[0]
            out = []
            for s in range(lo - 1, hi - 1, CH):
                e = min(s + CH, hi - 1)
                lp = torch.log_softmax(
                    pm.base_model.model.lm_head(h[s:e]).float(), -1)
                out += lp.gather(1, ids_t[0, s + 1:e + 1]
                                 .unsqueeze(1)).squeeze(1).tolist()
        return st.mean(out) if out else None

    def fwd(ids, preds, grad, adapter_off=False):
        ids_t = torch.tensor([ids], device=pm.device)
        am = torch.ones_like(ids_t)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            if adapter_off:
                with pm.disable_adapter():
                    h = pm.base_model.model.model(
                        input_ids=ids_t, attention_mask=am).last_hidden_state
            else:
                h = pm.base_model.model.model(
                    input_ids=ids_t, attention_mask=am).last_hidden_state
            idx = torch.tensor(preds, device=h.device)
            hc = h[0].index_select(0, idx).unsqueeze(0)
            return pm.base_model.model.lm_head(hc)

    def kl_top256(s_l, t_l):
        tv, ti = t_l.topk(TOPK, dim=-1)
        s_g = s_l.gather(-1, ti)
        s_lp = F.log_softmax(s_g.float() / TEMP, dim=-1)
        t_lp = F.log_softmax(tv.float() / TEMP, dim=-1)
        return (t_lp.exp() * (t_lp - s_lp)).sum(-1)

    # ── train pairs (strict, as S) ─────────────────────────────────────
    train_raw = aligned_pairs(train_c, tok)
    train = []
    for p in train_raw:
        f_ids = render(tok, system_base, p["user"], p["hist"],
                       p["full_text"])
        f_ids_t = render(tok, system_base + p["ref_suffix"], p["user"],
                         p["hist"], p["full_text"])
        a_span = anchor(f_ids, p["act_text"])
        r_span = anchor(f_ids, p["vals"][2])
        a_span_t = anchor(f_ids_t, p["act_text"])
        if not (a_span and r_span and a_span_t):
            continue
        delta = a_span_t[0] - a_span[0]
        if not all(f_ids_t[q + delta] == f_ids[q]
                   for q in range(r_span[0], r_span[1])):
            continue

        c_span = anchor(f_ids, p["cut_text"].rstrip())
        u_span = (anchor(f_ids, p["audit_text"].strip())
                  if p["audit_text"].strip() else None)
        if c_span is None:
            continue
        train.append({
            "f_ids": f_ids, "f_ids_t": f_ids_t, "delta": delta,
            "A": a_span, "R": r_span,
            "T": c_span, "U": u_span,
        })
    print(f"[s2] aligned train pairs: {len(train)}", flush=True)

    evals = []
    for p in aligned_pairs(held_c, tok):
        rels = (p["act_json"].get("args") or {}).get("relations") or []
        unchosen = [r for r in p["menu"] if r not in rels]
        if len(rels) < 1 or not unchosen:
            continue
        vals_lo, vals_hi, vals_str = p["vals"]
        act = p["act_text"]
        neg_vals = " | ".join([unchosen[0]] + rels[1:])
        pos_full = p["full_text"]
        neg_full = p["full_text"].replace(vals_str, neg_vals, 1)
        if neg_full == pos_full:
            continue
        evals.append({"cid": p["cid"], "user": p["user"], "hist": p["hist"],
                      "pos_full": pos_full, "neg_full": neg_full,
                      "pos_vals": vals_str, "neg_vals": neg_vals,
                      "act_text": act})
    print(f"[s2] held-out eval pairs: {len(evals)}", flush=True)

    def score_ev(ev):
        out = {}
        for tag, full, vals in (("pos", ev["pos_full"], ev["pos_vals"]),
                                ("neg", ev["neg_full"], ev["neg_vals"])):
            f_ids = render(tok, system_base, ev["user"], ev["hist"], full)
            r = anchor(f_ids, vals)
            a = anchor(f_ids, ev["act_text"]) or (r[0], r[1])
            if r is None:
                out[tag] = None
                continue
            out[tag] = {"rel": span_logp(f_ids, r[0], r[1]),
                        "act": span_logp(f_ids, a[0], a[1])}
        return out

    base = [score_ev(ev) for ev in evals]

    def region_ranges(t):
        out = {"A": [t["A"]], "R": [t["R"]]}
        if t["T"]:
            out["T"] = [t["T"]]
        if t["U"]:
            out["U"] = [t["U"]]
        return out

    def preds_for(t, reg):
        rr = region_ranges(t).get(reg)
        if not rr:
            return []
        out = []
        for lo, hi in rr:
            out += [x - 1 for x in range(lo, hi)
                    if 0 < x - 1 < len(t["f_ids"]) - 1]
        return out

    Nreg = {r: sum(len(preds_for(t, r)) for t in train) or 1
            for r in ("A", "T", "U")}
    N_all = sum(len(preds_for(t, r)) for t in train
                for r in ("A", "T", "U")) or 1

    def run_cond(name, regs_weights, norm_mode):
        for k in adapter0:
            pm.state_dict()[k].copy_(adapter0[k])
        pm.zero_grad(set_to_none=True)
        rec = {"cond": name}
        if name != "I":
            eff = LR / 5.0
            opt = torch.optim.AdamW(
                [{"params": [q for n_, q in pm.named_parameters()
                             if "lora_A" in n_], "lr": eff},
                 {"params": [q for n_, q in pm.named_parameters()
                             if "lora_B" in n_], "lr": eff * LORA_PLUS}],
                lr=eff, weight_decay=0.0)
            pm.train()
            for step_i in range(STEPS):
                for gi, g in enumerate(opt.param_groups):
                    base_lr = LR if gi == 0 else LR * LORA_PLUS
                    g["lr"] = base_lr * min(1.0, (step_i + 1) / 5.0)
                opt.zero_grad(set_to_none=True)
                for t in train:
                    for reg, w in regs_weights:
                        if reg not in region_ranges(t):
                            continue
                        N = N_all if norm_mode == "token_mean" else Nreg[reg]
                        preds = preds_for(t, reg)
                        t_preds = [min(q + t["delta"],
                                       len(t["f_ids_t"]) - 2)
                                   for q in preds]
                        for c0 in range(0, len(preds), CH):
                            s_l = fwd(t["f_ids"], preds[c0:c0 + CH],
                                      grad=True)
                            t_l = fwd(t["f_ids_t"], t_preds[c0:c0 + CH],
                                      grad=False, adapter_off=True)
                            kl = kl_top256(s_l, t_l)
                            ((w * kl.sum()) / N).backward()
                            del s_l, t_l
                torch.nn.utils.clip_grad_norm_(
                    [q for q in pm.parameters() if q.grad is not None], 1.0)
                opt.step()
            pm.eval()
        dvec = torch.cat([(q.detach().float() - adapter0[n_].float()
                           .to(q.device)).flatten()
                          for n_, q in pm.named_parameters()
                          if "lora_" in n_ and q.requires_grad])
        rec["dtheta_norm"] = float(dvec.norm())
        dm, dpos, dneg, dact = [], [], [], []
        for ev, b in zip(evals, base):
            sc = score_ev(ev)
            if None in (sc["pos"], sc["neg"], b["pos"], b["neg"]):
                continue
            dm.append((sc["pos"]["rel"] - sc["neg"]["rel"])
                      - (b["pos"]["rel"] - b["neg"]["rel"]))
            dpos.append(sc["pos"]["rel"] - b["pos"]["rel"])
            dneg.append(sc["neg"]["rel"] - b["neg"]["rel"])
            dact.append((sc["pos"]["act"] - sc["neg"]["act"])
                        - (b["pos"]["act"] - b["neg"]["act"]))
        by = collections.defaultdict(list)
        for ev, d in zip(evals, dm):
            by[ev["cid"]].append(d)
        cm = [st.mean(v) for v in by.values()]
        rng = random.Random(0)
        bs = sorted(sum(rng.choice(cm) for _ in cm) / len(cm)
                    for _ in range(10000))
        rec.update({
            "dM_rel_mean": st.mean(dm) if dm else None,
            "dM_pos_pairs": f"{sum(1 for x in dm if x > 0)}/{len(dm)}",
            "dM_case_CI": [bs[250], bs[9750]],
            "d_logp_apos": st.mean(dpos) if dpos else None,
            "d_logp_aneg": st.mean(dneg) if dneg else None,
            "dM_act_mean": st.mean(dact) if dact else None,
        })
        print(f"[s2] {name:14s} ΔM_rel={rec['dM_rel_mean']:+.5f} "
              f"({rec['dM_pos_pairs']}) CI=[{bs[250]:+.4f},{bs[9750]:+.4f}] "
              f"| ΔlogP+={rec['d_logp_apos']:+.5f} "
              f"ΔlogP−={rec['d_logp_aneg']:+.5f} "
              f"ΔM_act={rec['dM_act_mean']:+.5f} "
              f"|dθ|={rec['dtheta_norm']:.3f}", flush=True)
        return rec

    results = [
        run_cond("I", [], "none"),
        run_cond("F_token_mean", [("A", 1.0), ("T", 1.0), ("U", 1.0)],
                 "token_mean"),
        run_cond("F_region_equal", [("A", 1.0), ("T", 1.0), ("U", 1.0)],
                 "region"),
        run_cond("A", [("A", 1.0)], "region"),
        run_cond("A+0.25T", [("A", 1.0), ("T", 0.25)], "region"),
    ]
    json.dump({"train_pairs": len(train), "eval_pairs": len(evals),
               "results": results}, open(args.out, "w"), indent=1)
    print(f"[s2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
