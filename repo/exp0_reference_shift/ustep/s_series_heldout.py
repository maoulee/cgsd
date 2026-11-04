#!/usr/bin/env python3
"""S-series: Selective GKD → HELD-OUT decision improvement (2026-11-04).

The decisive generalization test per the audit: with frozen teacher,
warmup+clip, strict masks, and CASE-ISOLATED train/test splits — does
selective supervision improve the bare policy's correct-vs-wrong RELATION
margin on held-out cases?

Conditions (identical train pairs, same init, 3 steps, warmup schedule):
  I        identity — no update (numerical control)
  F        current-turn FULL (A∪T∪U) — the "everything" baseline
  A        action-only mask
  A+0.25T  per-region-normalized λ=0.25 soft thinking

Eval: held-out pairs (a+ = the case's own effective action, a− = same-menu
distractor swap, exp1c2 construction), scored BARE before/after each
condition. ΔM = M(θ1) − M(θ0), case-level bootstrap.

Usage:
  SEQ_PROMPT=C123 python scripts/s_series_heldout.py --gpu 0 --out reports/s_series.json
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


def aligned_pairs(cid_list, tok, limit_per_case=99):
    """Strictly-aligned decision pairs (W4b machinery) for given cases."""
    verdicts = json.load(open(VERDICTS))
    rollout = json.load(open(ROLLOUT))
    by_key = {(r["case_id"], int(r["sample_idx"])): r for r in rollout}
    by_case = collections.defaultdict(list)
    for v in verdicts:
        by_case[v["case_id"]].append(v)
    pts = []
    for cid in cid_list:
        ok = sorted([v for v in by_case[cid] if v.get("f1", 0) > 0 and
                     (cid, int(v["sample"])) in by_key],
                    key=lambda v: -v.get("f1", 0))
        if len(ok) < 2:
            continue
        ref = ok[0]
        ref_row = by_key[(cid, int(ref["sample"]))]
        golds = ref["golds"] or ["unknown"]
        from build_teacher_prompts import render_fact_grouped, wrap_structured
        ref_suffix = "\n\n" + wrap_structured(render_fact_grouped(
            ref_row["trajectory"], answer=golds, gold=golds))
        n_this = 0
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
                act = parse_action(st_.get("content") or "")
                if not act:
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
                f_ids = render(tok, "S", user, hist, full_text)  # placeholder
                # real renders done by caller with system_base; here store
                # the pieces
                pts.append({
                    "cid": cid, "user": user, "hist": hist,
                    "full_text": full_text, "cut_text": cut_text,
                    "raw_str": raw_str, "tool_char": tool_char,
                    "act_text": act_text, "audit_text": audit_text,
                    "vals": (vals_lo, vals_hi, vals_str),
                    "ref_suffix": ref_suffix,
                    "menu": menu_before(tr, idx),
                    "act_json": act,
                })
                n_this += 1
                if n_this >= limit_per_case:
                    break
            if n_this >= limit_per_case:
                break
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="reports/s_series.json")
    args = ap.parse_args()

    from kgqa.agent.seq_react_loop import seq_agents_md
    system_base = seq_agents_md()
    assert hashlib.sha256(system_base.encode()).hexdigest() == PROMPT_SHA

    verdicts = json.load(open(VERDICTS))
    by_case = collections.defaultdict(list)
    for v in verdicts:
        by_case[v["case_id"]].append(v)
    rollout = json.load(open(ROLLOUT))
    cids_ok = sorted(cid for cid, vs in by_case.items()
                     if sum(1 for v in vs if v.get("f1", 0) > 0 and
                            any((v["case_id"], int(v["sample"])) ==
                                (r["case_id"], int(r["sample_idx"]))
                                for r in rollout if v.get("f1", 0) > 0)) >= 2)
    train_c = [c for i, c in enumerate(cids_ok) if i % 4 != 3][:10]
    held_c = [c for i, c in enumerate(cids_ok) if i % 4 == 3]
    print(f"[s] cases: train {len(train_c)} / held-out {len(held_c)}",
          flush=True)

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

    def finalize(p, suffix=""):
        """Render + char-anchor boundaries for one pair under a suffix."""
        f_ids = render(tok, system_base + suffix, p["user"], p["hist"],
                       p["full_text"])
        tok_strs = [tok.decode([t]) for t in f_ids]
        full_dec = "".join(tok_strs)
        cum, off = [], 0
        for tstr in tok_strs:
            cum.append(off)
            off += len(tstr)

        def tc(c):
            lo, hi = 0, len(cum) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if cum[mid] <= c:
                    lo = mid
                else:
                    hi = mid - 1
            return lo
        h_char = full_dec.rfind("<|im_start|>assistant\n")
        a_char = full_dec.rfind(p["act_text"])
        if h_char < 0 or a_char < 0:
            return None
        c_start = tc(h_char) + len(hdr_ids)
        act_start = tc(a_char)
        think_hi = tc(a_char - p["tool_char"])
        end = tc(a_char + len(p["act_text"]) - 1) + 1
        vals_lo, vals_hi, vals_str = p["vals"]
        r_lo = tc(a_char + vals_lo)
        r_hi = tc(a_char + vals_hi - 1) + 1
        # assertions
        if "".join(tok_strs[act_start:end]).strip() != p["act_text"].strip():
            return None
        if "".join(tok_strs[r_lo:r_hi]).strip() != vals_str.strip():
            return None
        return {"f_ids": f_ids, "c_start": c_start, "think_hi": think_hi,
                "act_start": act_start, "end": end, "r_lo": r_lo,
                "r_hi": r_hi}

    # train pairs (teacher = bare + ref suffix)
    train_raw = aligned_pairs(train_c, tok)
    train = []
    for p in train_raw:
        b = finalize(p)
        t = finalize(p, p["ref_suffix"])
        if b is None or t is None:
            continue
        a_char_t = "".join(tok.decode([x]) for x in t["f_ids"]).rfind(p["act_text"])
        delta = tok_at_char_f(t["f_ids"], tok, a_char_t) - b["act_start"] if a_char_t >= 0 else None
        if delta is None:
            continue
        if not all(t["f_ids"][q + delta] == b["f_ids"][q]
                   for lo_, hi_ in ((b["act_start"], b["end"]),
                                    (b["r_lo"], b["r_hi"]))
                   for q in range(lo_, hi_)):
            continue
        train.append({"p": p, "b": b, "t": t, "delta": delta})
    print(f"[s] aligned train pairs: {len(train)}", flush=True)

    # held-out eval pairs: a+ = own action; a− = menu distractor swap
    held_raw = aligned_pairs(held_c, tok)
    evals = []
    for p in held_raw:
        rels = (p["act_json"].get("args") or {}).get("relations") or []
        unchosen = [r for r in p["menu"] if r not in rels]
        if len(rels) < 1 or not unchosen:
            continue
        neg = json.loads(json.dumps(p["act_json"]))
        neg["args"]["relations"] = [unchosen[0]] + rels[1:]
        # rebuild the negative action text in flat form
        neg_text = re.sub(r"relations:.*",
                          "relations: " + " | ".join(neg["args"]["relations"]),
                          p["act_text"])
        evals.append({"p": p, "a_pos_text": p["act_text"],
                      "a_neg_text": neg_text})
    print(f"[s] held-out eval pairs: {len(evals)}", flush=True)

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

    def score_eval(p, cand_text):
        full = f"{p['cut_text']}{cand_text}" if p["cut_text"].strip() else cand_text
        f_ids = render(tok, system_base, p["user"], p["hist"], full)
        c_ids = render(tok, system_base, p["user"], p["hist"],
                       p["cut_text"] if p["cut_text"].strip() else " ")
        return span_logp(f_ids, len(c_ids), len(f_ids))

    base_M = []
    for ev in evals:
        sp = score_eval(ev["p"], ev["a_pos_text"])
        sn = score_eval(ev["p"], ev["a_neg_text"])
        base_M.append((sp, sn, (sp - sn) if sp is not None and sn is not None
                       else None))

    def region_ranges(b):
        return {"A": [(b["act_start"], b["end"])],
                "T": [(b["c_start"], b["think_hi"])],
                "U": [(b["think_hi"], b["act_start"])],
                "R": [(b["r_lo"], b["r_hi"])]}

    def preds_for(b, regs):
        out = []
        for reg in regs:
            for lo, hi in region_ranges(b)[reg]:
                out += [x - 1 for x in range(lo, hi)
                        if 0 < x - 1 < len(b["f_ids"]) - 1]
        return out

    def run_cond(name, regs_weights):
        for k in adapter0:
            pm.state_dict()[k].copy_(adapter0[k])
        pm.zero_grad(set_to_none=True)
        rec = {"cond": name}
        if name == "I":
            pass                                   # no update
        else:
            eff = LR / 5.0
            opt = torch.optim.AdamW(
                [{"params": [q for n_, q in pm.named_parameters()
                             if "lora_A" in n_], "lr": eff},
                 {"params": [q for n_, q in pm.named_parameters()
                             if "lora_B" in n_], "lr": eff * LORA_PLUS}],
                lr=eff, weight_decay=0.0)
            Ns = {}
            for reg, w in regs_weights:
                Ns[reg] = sum(len(preds_for(x["b"], [reg]))
                              for x in train) or 1
            pm.train()
            for step_i in range(STEPS):
                for gi, g in enumerate(opt.param_groups):
                    base = LR if gi == 0 else LR * LORA_PLUS
                    g["lr"] = base * min(1.0, (step_i + 1) / 5.0)
                opt.zero_grad(set_to_none=True)
                for x in train:
                    b, t, delta = x["b"], x["t"], x["delta"]
                    for reg, w in regs_weights:
                        preds = preds_for(b, [reg])
                        t_preds = [min(q + delta, len(t["f_ids"]) - 2)
                                   for q in preds]
                        for c0 in range(0, len(preds), CH):
                            s_l = fwd(b["f_ids"], preds[c0:c0 + CH],
                                      grad=True)
                            t_l = fwd(t["f_ids"], t_preds[c0:c0 + CH],
                                      grad=False, adapter_off=True)
                            kl = kl_top256(s_l, t_l)
                            ((w * kl.sum()) / (Ns[reg])).backward()
                            del s_l, t_l
                torch.nn.utils.clip_grad_norm_(
                    [q for q in pm.parameters() if q.grad is not None], 1.0)
                opt.step()
            pm.eval()
        dMs, cases = [], []
        for ev, (sp0, sn0, m0) in zip(evals, base_M):
            sp = score_eval(ev["p"], ev["a_pos_text"])
            sn = score_eval(ev["p"], ev["a_neg_text"])
            if None in (sp, sn, m0):
                continue
            dMs.append((sp - sn) - m0)
            cases.append(ev["p"]["cid"])
        rec["dM_mean"] = st.mean(dMs) if dMs else None
        rec["dM_pairs"] = dMs
        rec["pair_cases"] = cases
        by = collections.defaultdict(list)
        for c, d in zip(cases, dMs):
            by[c].append(d)
        cm = [st.mean(v) for v in by.values()]
        rng = random.Random(0)
        bs = sorted(sum(rng.choice(cm) for _ in cm) / len(cm)
                    for _ in range(10000))
        rec["dM_case_equal"] = st.mean(cm) if cm else None
        rec["dM_CI95_case"] = [bs[250], bs[9750]] if bs else None
        rec["dM_pos_pairs"] = f"{sum(1 for x in dMs if x > 0)}/{len(dMs)}"
        print(f"[s] {name:8s} held-out ΔM={rec['dM_mean']:+.5f} "
              f"({rec['dM_pos_pairs']}) case-equal={rec['dM_case_equal']:+.5f} "
              f"CI=[{rec['dM_CI95_case'][0]:+.4f},{rec['dM_CI95_case'][1]:+.4f}]",
              flush=True)
        return rec

    results = [
        run_cond("I", []),
        run_cond("F", [("A", 1.0), ("T", 1.0), ("U", 1.0)]),
        run_cond("A", [("A", 1.0)]),
        run_cond("A+0.25T", [("A", 1.0), ("T", 0.25)]),
    ]
    json.dump({"train_pairs": len(train), "eval_pairs": len(evals),
               "steps": STEPS, "results": results},
              open(args.out, "w"), indent=1)
    print(f"[s] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
