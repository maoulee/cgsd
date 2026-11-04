#!/usr/bin/env python3
"""W4b: region-split arms (audit 2026-11-04 #2) — isolate AUDIT vs THINKING.

W4's AT arm was a CONTIGUOUS span that silently included the AUDIT block
(think_hi..act_start). W4b uses DISJOINT-UNION masks:

  A = action | T = thinking | U = AUDIT (post-think, pre-tool text)
  arms: A, T, U, A∪T (no AUDIT), A∪U (no thinking), A∪T∪U (= W4's AT)

Plus λ-dose arms with PER-REGION normalization: L = L_A/N_A + λ·L_T/N_T
for λ ∈ {0.25, 0.5, 1} — separates supervision-scope effects from
token-count weighting. Diagnostics per arm: per-region token counts,
grad norm, per-pair Δ (persisted with case ids), the ACTUAL Adam update
vector's cosine with the R-arm gradient direction g_R.

Usage:
  SEQ_PROMPT=C123 python scripts/w4b_region_split.py --gpu 0 \
      --out reports/w4b.json
"""
import argparse
import collections
import hashlib
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SEQ_PROMPT", "C123")
os.environ.setdefault("SEQ_ADAPTIVE_ROUTES", "1")

from exp0_reference_shift import (  # noqa: E402
    VERDICTS, ROLLOUT, PROMPT_SHA, label_of, build_user_msg, render)
from exp1c2_independent_ref import parse_action  # noqa: E402
from w4_strict_align import turn_text, tok_at_char_f  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TOPK = 256
TEMP = 1.0
LR = 2e-5
LORA_PLUS = 4.0
CH = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="reports/w4b.json")
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
    cids_ok = sorted(cid for cid, vs in by_case.items()
                     if sum(1 for v in vs if v.get("f1", 0) > 0 and
                            (cid, int(v["sample"])) in by_key) >= 2)
    train_c = [c for i, c in enumerate(cids_ok) if i % 4 != 3]

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

    pts, n_skip = [], 0
    FAILS = collections.Counter()
    for cid in train_c:
        ok = sorted([v for v in by_case[cid] if v.get("f1", 0) > 0 and
                     (cid, int(v["sample"])) in by_key],
                    key=lambda v: -v.get("f1", 0))
        ref_row = by_key[(cid, int(ok[0]["sample"]))]
        golds = ok[0]["golds"] or ["unknown"]
        ref_suffix = "\n\n" + wrap_structured(render_fact_grouped(
            ref_row["trajectory"], answer=golds, gold=golds))
        for scored in ok[1:]:
            tr = by_key[(cid, int(scored["sample"]))]["trajectory"]
            row = by_key[(cid, int(scored["sample"]))]
            for b in scored["blocks"]:
                if label_of(b.get("verdict", "")) != "EFFECTIVE":
                    continue
                idx = int(b["idx"])
                if idx < 1 or tr[idx - 1].get("role") != "assistant":
                    continue
                st_ = tr[idx - 1]
                if not parse_action(st_.get("content") or ""):
                    continue
                tt = turn_text(st_)
                if tt is None:
                    continue
                full_text, cut_text, raw_str, tool_char = tt
                act_text = raw_str[tool_char:]
                audit_text = raw_str[:tool_char]
                m_rel = re.search(r"(?:^|\n)relations:\s*", act_text)
                if not m_rel:
                    continue
                vals_lo = m_rel.end()
                vals_hi = act_text.find("\n", vals_lo)
                vals_hi = vals_hi if vals_hi > 0 else len(act_text)
                vals_str = act_text[vals_lo:vals_hi]
                if not vals_str.strip():
                    continue
                hist = []
                bad = False
                for m2 in tr[:idx - 1]:
                    if m2.get("role") == "assistant":
                        t2 = turn_text(m2)
                        if t2 is None:
                            bad = True
                            break
                        hist.append({"role": "assistant", "content": t2[0]})
                    else:
                        hist.append({"role": m2.get("role"),
                                     "content": m2.get("content"),
                                     "name": m2.get("name")})
                if bad:
                    continue
                user = build_user_msg(row)
                f_ids = render(tok, system_base, user, hist, full_text)
                f_ids_t = render(tok, system_base + ref_suffix, user, hist,
                                 full_text)
                tok_strs = [tok.decode([t]) for t in f_ids]
                full_dec = "".join(tok_strs)
                cum, off = [], 0
                for tstr in tok_strs:
                    cum.append(off)
                    off += len(tstr)

                def tok_at_char(c):
                    lo, hi = 0, len(cum) - 1
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if cum[mid] <= c:
                            lo = mid
                        else:
                            hi = mid - 1
                    return lo
                hdr_txt = "<|im_start|>assistant\n"
                h_char = full_dec.rfind(hdr_txt)
                a_char = full_dec.rfind(act_text)
                if h_char < 0 or a_char < 0:
                    n_skip += 1
                    FAILS.update(["find"])
                    continue
                c_start = tok_at_char(h_char) + len(hdr_ids)
                act_start = tok_at_char(a_char)
                raw_start_char = a_char - tool_char
                think_hi = tok_at_char(raw_start_char)
                end = tok_at_char(a_char + len(act_text) - 1) + 1
                r_lo = tok_at_char(a_char + vals_lo)
                r_hi = tok_at_char(a_char + vals_hi - 1) + 1
                # asserts incl. AUDIT region
                fails = []
                for lo_, hi_, exp, tag in (
                        (act_start, end, act_text, "action"),
                        (r_lo, r_hi, vals_str, "relations"),
                        (c_start, think_hi, cut_text, "thinking"),
                        (think_hi, act_start, audit_text, "audit")):
                    got = "".join(tok_strs[lo_:hi_])
                    if got.strip() != (exp or "").strip() and exp.strip():
                        fails.append(tag)
                tok_strs_t = [tok.decode([t]) for t in f_ids_t]
                full_dec_t = "".join(tok_strs_t)
                a_char_t = full_dec_t.rfind(act_text)
                delta = (tok_at_char_f(f_ids_t, tok, a_char_t) - act_start
                         if a_char_t >= 0 else None)
                if delta is None:
                    fails.append("teacher-anchor")
                else:
                    if not all(f_ids_t[p + delta] == f_ids[p]
                               for lo_, hi_ in ((act_start, end),
                                                (r_lo, r_hi),
                                                (c_start, think_hi),
                                                (think_hi, act_start))
                               for p in range(lo_, hi_)):
                        fails.append("tids")
                if fails:
                    n_skip += 1
                    FAILS.update(fails)
                    continue
                pts.append({"cid": cid, "f_ids": f_ids, "f_ids_t": f_ids_t,
                            "delta": delta, "c_start": c_start,
                            "think_hi": think_hi, "act_start": act_start,
                            "end": end, "r_lo": r_lo, "r_hi": r_hi})
            if len(pts) >= args.n:
                break
        if len(pts) >= args.n:
            break
    print(f"[w4b] aligned: {len(pts)} | skipped: {n_skip} "
          f"| fails: {dict(FAILS)}", flush=True)

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

    REGION = {"R": lambda p: [(p["r_lo"], p["r_hi"])],
              "A": lambda p: [(p["act_start"], p["end"])],
              "T": lambda p: [(p["c_start"], p["think_hi"])],
              "U": lambda p: [(p["think_hi"], p["act_start"])]}

    def region_preds(p, reg):
        out = []
        for lo, hi in REGION[reg](p):
            out += [x - 1 for x in range(lo, hi)
                    if 0 < x - 1 < len(p["f_ids"]) - 1]
        return out

    def region_backward(regs_weights):
        """regs_weights: list of (region, weight, normN) — loss contribution
        sum(w * kl_sum)/normN per region."""
        for reg, w, N in regs_weights:
            for p in pts:
                preds = region_preds(p, reg)
                t_preds = [min(x + p["delta"], len(p["f_ids_t"]) - 2)
                           for x in preds]
                for c0 in range(0, len(preds), CH):
                    s_l = fwd(p["f_ids"], preds[c0:c0 + CH], grad=True)
                    t_l = fwd(p["f_ids_t"], t_preds[c0:c0 + CH], grad=False,
                              adapter_off=True)
                    kl = kl_top256(s_l, t_l)
                    ((w * kl.sum()) / N).backward()
                    del s_l, t_l

    def gvec_now():
        gs = [q.grad.detach().float().flatten()
              for _, q in pm.named_parameters() if q.grad is not None]
        return torch.cat(gs) if gs else None

    def new_opt():
        eff = LR / 5.0
        return torch.optim.AdamW(
            [{"params": [q for n_, q in pm.named_parameters()
                         if "lora_A" in n_], "lr": eff},
             {"params": [q for n_, q in pm.named_parameters()
                         if "lora_B" in n_], "lr": eff * LORA_PLUS}],
            lr=eff, weight_decay=0.0)

    # N per region (token counts)
    Nreg = {r: sum(len(region_preds(p, r)) for p in pts) for r in REGION}

    # g_R reference direction (once)
    for k in adapter0:
        pm.state_dict()[k].copy_(adapter0[k])
    pm.zero_grad(set_to_none=True)
    pm.train()
    region_backward([("R", 1.0, Nreg["R"])])
    g_R = gvec_now().clone()

    base_R = [span_logp(p["f_ids"], p["r_lo"], p["r_hi"]) for p in pts]

    def run_arm(name, regs_weights):
        for k in adapter0:
            pm.state_dict()[k].copy_(adapter0[k])
        pm.zero_grad(set_to_none=True)
        pm.train()
        opt = new_opt()
        region_backward(regs_weights)
        g_arm = gvec_now()
        gnorm = float(g_arm.norm())
        cos_gR = float((g_arm @ g_R) / (g_arm.norm() * g_R.norm())) \
            if gnorm > 0 else None
        torch.nn.utils.clip_grad_norm_(
            [q for q in pm.parameters() if q.grad is not None], 1.0)
        opt.step()
        # actual Adam update direction vs g_R
        dvec = torch.cat([(q.detach().float() - adapter0[n_].float()
                           .to(q.device)).flatten()
                          for n_, q in pm.named_parameters()
                          if "lora_" in n_ and q.requires_grad])
        cos_up = float((dvec @ g_R) / (dvec.norm() * g_R.norm()))
        pm.eval()
        dRs, cases = [], []
        for p, b0 in zip(pts, base_R):
            post = span_logp(p["f_ids"], p["r_lo"], p["r_hi"])
            if post is not None and b0 is not None:
                dRs.append(post - b0)
                cases.append(p["cid"])
        rec = {"arm": name,
               "regions": [f"{r}×{w}" for r, w, _ in regs_weights],
               "n_tokens": {r: Nreg[r] for r, w, _ in regs_weights},
               "grad_norm": gnorm, "cos_g_arm_vs_gR": cos_gR,
               "cos_adam_update_vs_gR": cos_up,
               "dR_mean": st.mean(dRs) if dRs else None,
               "dR_pos": f"{sum(1 for x in dRs if x > 0)}/{len(dRs)}",
               "dR_pairs": dRs, "pair_cases": cases}
        print(f"[w4b] {name:6s} ΔR={rec['dR_mean']:+.5f} "
              f"({rec['dR_pos']}) gnorm={gnorm:.2e} "
              f"cos(g,gR)={cos_gR:+.3f} cos(Δθ,gR)={cos_up:+.3f}",
              flush=True)
        return rec

    results = [
        run_arm("A", [("A", 1.0, Nreg["A"])]),
        run_arm("T", [("T", 1.0, Nreg["T"])]),
        run_arm("U", [("U", 1.0, Nreg["U"])]),
        run_arm("A∪T", [("A", 1.0, Nreg["A"] + Nreg["T"]),
                        ("T", 1.0, Nreg["A"] + Nreg["T"])]),
        
        run_arm("A∪U", [("A", 1.0, Nreg["A"] + Nreg["U"]),
                        ("U", 1.0, Nreg["A"] + Nreg["U"])]),
        run_arm("A∪T∪U", [("A", 1.0, Nreg["A"] + Nreg["T"] + Nreg["U"]),
                          ("T", 1.0, Nreg["A"] + Nreg["T"] + Nreg["U"]),
                          ("U", 1.0, Nreg["A"] + Nreg["T"] + Nreg["U"])]),
    ]
    # λ-dose (per-region normalization): L = L_A/N_A + λ·L_T/N_T
    for lam in (0.25, 0.5, 1.0):
        results.append(run_arm(f"λ={lam}",
                               [("A", 1.0, Nreg["A"]),
                                ("T", lam, Nreg["T"])]))
    out = {"aligned": len(pts), "skipped": n_skip,
           "Nreg": Nreg, "arms": results}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[w4b] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
