#!/usr/bin/env python3
"""W4: strict token alignment + current-turn-only arms (audit 2026-11-04).

Fixes vs wseries_singlepos.py:
  1. Causal shift done ONCE, correctly: predictor positions = target-1.
  2. DECODE-AND-ASSERT every supervised span: decode the target token ids
     and require exact match with the intended text (relations values /
     action text / thinking text); teacher/student target-id equality at
     aligned positions. Pairs failing alignment are SKIPPED and counted.
  3. History assistant turns rendered in the PRODUCTION shape:
     <think>…</think> + RAW flat action (not JSON).
  4. Arms supervise the CURRENT TURN ONLY (never history/context):
       R  = relations values        A = whole flat action
       T  = current-turn thinking   A+T = thinking + action
  5. Correct Δ metrics: same-scope post-minus-pre (no mixed metrics).
  6. Gradient interference: cos(g_R, g_T) over LoRA grads; plus a
     normalized L_R + λ·L_T dose arm (λ=1) to test interference beyond
     mask composition.

Usage:
  SEQ_PROMPT=C123 python scripts/w4_strict_align.py --gpu 0 \
      --out reports/w4_strict.json
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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TOPK = 256
TEMP = 1.0
LR = 2e-5
LORA_PLUS = 4.0
CH = 256


def tok_at_char_f(ids, tok_, char):
    cum, off = [], 0
    for t in ids:
        cum.append(off)
        off += len(tok_.decode([t]))
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cum[mid] <= char:
            lo = mid
        else:
            hi = mid - 1
    return lo


def turn_text(m):
    """Production-shape assistant turn: <think>{reasoning}</think> + RAW.

    The stored `raw` is the POST-THINKING output (may include an AUDIT
    block before the tool line); `reasoning` is the thinking itself.
    Returns (full_text, think_cut, raw_str, tool_char) where tool_char is
    the char offset of the `tool:` line inside raw."""
    raw = (m.get("raw") or m.get("content") or "")
    mm = re.search(r"(?:^|\n)tool:\s*\w+", raw)
    if not mm:
        return None
    reasoning = (m.get("reasoning") or "").strip()
    if reasoning:
        full = f"<think>\n{reasoning}\n</think>" + raw
        cut = f"<think>\n{reasoning}\n</think> "
    else:
        full, cut = raw, " "
    return full, cut, raw, mm.start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="reports/w4_strict.json")
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
    im_end_id = tok("<|im_end|>", add_special_tokens=False).input_ids[0]
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

    pts, n_skip_align, n_skip_other = [], 0, 0
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
                m_rel = re.search(r"(?:^|\n)relations:\s*", act_text)
                if not m_rel:
                    continue
                vals_lo = m_rel.end()
                vals_hi = act_text.find("\n", vals_lo)
                vals_hi = vals_hi if vals_hi > 0 else len(act_text)
                vals_str = act_text[vals_lo:vals_hi]
                if not vals_str.strip():
                    continue
                # production-shape history
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
                    n_skip_other += 1
                    continue
                user = build_user_msg(row)
                # renders
                f_ids = render(tok, system_base, user, hist, full_text)
                f_ids_t = render(tok, system_base + ref_suffix, user, hist,
                                 full_text)
                # ---- char-anchor boundary location (byte-level BPE: the
                # concatenation of per-token decodes == the full decode) ----
                tok_strs = [tok.decode([t]) for t in f_ids]
                full_dec = "".join(tok_strs)
                # token index that OWNS each char (start of token i at cum_i)
                cum, off = [], 0
                for tstr in tok_strs:
                    cum.append(off)
                    off += len(tstr)
                def tok_at_char(c):
                    # bisect: last token whose cum <= c
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
                if h_char < 0:
                    n_skip_align += 1
                    FAILS.update(["header"])
                    continue
                c_start = tok_at_char(h_char) + len(hdr_ids)
                a_char = full_dec.rfind(act_text)
                if a_char < 0:
                    n_skip_align += 1
                    FAILS.update(["action-find"])
                    continue
                act_start = tok_at_char(a_char)
                raw_start_char = a_char - tool_char
                think_hi = tok_at_char(raw_start_char)
                end_char = a_char + len(act_text)
                end = tok_at_char(end_char - 1) + 1
                r_char = a_char + vals_lo
                r_hi_char = a_char + vals_hi
                r_lo = tok_at_char(r_char)
                r_hi = tok_at_char(r_hi_char - 1) + 1
                # ---- asserts ----
                fails = []
                if "".join(tok_strs[act_start:end]).strip() != \
                        act_text.strip():
                    fails.append("action")
                if "".join(tok_strs[r_lo:r_hi]).strip() != \
                        vals_str.strip():
                    fails.append("relations")
                if "".join(tok_strs[c_start:think_hi]).strip() != \
                        cut_text.strip():
                    fails.append("thinking")
                # teacher alignment: same anchors recomputed on ITS decode
                tok_strs_t = [tok.decode([t]) for t in f_ids_t]
                full_dec_t = "".join(tok_strs_t)
                delta = None
                if full_dec_t.endswith(full_dec[h_char:]):
                    body = full_dec[h_char:]
                    b_char_t = len(full_dec_t) - len(body)
                    # char delta between teacher and student contexts
                    delta_c = b_char_t - h_char
                    # token delta at the same content anchor
                    delta = tok_at_char_f(f_ids_t, tok, a_char + delta_c) \
                        - act_start if False else None
                    # simpler: locate both act anchors and use per-span
                    a_char_t = full_dec_t.rfind(act_text)
                    act_start_t = tok_at_char_f(f_ids_t, tok, a_char_t)
                    delta = act_start_t - act_start
                if delta is None:
                    fails.append("teacher-anchor")
                else:
                    ok_ids = all(f_ids_t[p + delta] == f_ids[p]
                                 for lo_, hi_ in ((act_start, end),
                                                  (r_lo, r_hi),
                                                  (c_start, think_hi))
                                 for p in range(lo_, hi_))
                    if not ok_ids:
                        fails.append("tids")
                if fails:
                    n_skip_align += 1
                    FAILS.update(fails)
                    continue
                pts.append({"cid": cid, "f_ids": f_ids, "f_ids_t": f_ids_t,
                            "delta": delta, "c_start": c_start,
                            "act_start": act_start, "think_hi": think_hi,
                            "end": end, "r_lo": r_lo, "r_hi": r_hi})
            if len(pts) >= args.n:
                break
        if len(pts) >= args.n:
            break
    print(f"[w4] aligned pairs: {len(pts)} | skipped(align): {n_skip_align} "
          f"| skipped(other): {n_skip_other} | fail-modes: {dict(FAILS)}",
          flush=True)
    if not pts:
        json.dump({"aligned_pairs": 0, "skipped": n_skip_align,
                   "fail_modes": dict(FAILS)}, open(args.out, "w"), indent=1)
        return

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
                lp = torch.log_softmax(model.lm_head(h[s:e]).float(), -1)
                out += lp.gather(1, ids_t[0, s + 1:e + 1]
                                 .unsqueeze(1)).squeeze(1).tolist()
        return st.mean(out) if out else None

    SPANS = {"R": lambda p: (p["r_lo"], p["r_hi"]),
             "A": lambda p: (p["act_start"], p["end"]),
             "T": lambda p: (p["c_start"], p["think_hi"]),
             "AT": lambda p: (p["c_start"], p["end"])}

    # baseline (θ0) per scope
    base = {k: [span_logp(p["f_ids"], *SPANS[k](p)) for p in pts]
            for k in SPANS}

    def arm_loss_backward(scope, N, collect=False):
        gvec = None
        for p in pts:
            lo, hi = SPANS[scope](p)
            preds = [x - 1 for x in range(lo, hi)          # causal shift
                     if 0 < x - 1 < len(p["f_ids"]) - 1]
            t_preds = [x - 1 + p["delta"] for x in range(lo, hi)
                       if 0 < x - 1 < len(p["f_ids"]) - 1]
            t_max = len(p["f_ids_t"]) - 1
            t_preds = [min(x, t_max - 1) for x in t_preds]
            for c0 in range(0, len(preds), CH):
                s_l = fwd(p["f_ids"], preds[c0:c0 + CH], grad=True)
                t_l = fwd(p["f_ids_t"], t_preds[c0:c0 + CH], grad=False,
                          adapter_off=True)
                kl = kl_top256(s_l, t_l)
                (kl.sum() / N).backward()
                del s_l, t_l
        if collect:
            gs = [q.grad.detach().float().flatten()
                  for _, q in pm.named_parameters() if q.grad is not None]
            gvec = torch.cat(gs)
        return gvec

    # gradient cosine g_R vs g_T (fresh init, separate passes)
    for k in adapter0:
        pm.state_dict()[k].copy_(adapter0[k])
    pm.zero_grad(set_to_none=True)
    pm.train()
    N_R = sum(SPANS["R"](p)[1] - SPANS["R"](p)[0] for p in pts)
    g_R = arm_loss_backward("R", N_R, collect=True)
    pm.zero_grad(set_to_none=True)
    N_T = sum(SPANS["T"](p)[1] - SPANS["T"](p)[0] for p in pts)
    g_T = arm_loss_backward("T", N_T, collect=True)
    cos_RT = float((g_R @ g_T) / (g_R.norm() * g_T.norm()))
    print(f"[w4] cos(g_R, g_T) = {cos_RT:+.4f}", flush=True)
    del g_R, g_T

    def run_arm(name, scope):
        for k in adapter0:
            pm.state_dict()[k].copy_(adapter0[k])
        pm.zero_grad(set_to_none=True)
        pm.train()
        eff = LR / 5.0
        opt = torch.optim.AdamW(
            [{"params": [q for n_, q in pm.named_parameters()
                         if "lora_A" in n_], "lr": eff},
             {"params": [q for n_, q in pm.named_parameters()
                         if "lora_B" in n_], "lr": eff * LORA_PLUS}],
            lr=eff, weight_decay=0.0)
        N = sum(SPANS[scope](p)[1] - SPANS[scope](p)[0] for p in pts)
        arm_loss_backward(scope, N)
        torch.nn.utils.clip_grad_norm_(
            [q for q in pm.parameters() if q.grad is not None], 1.0)
        opt.step()
        pm.eval()
        rec = {"arm": name, "scope": scope, "n_pairs": len(pts)}
        for k in SPANS:
            post = [span_logp(p["f_ids"], *SPANS[k](p)) for p in pts]
            d = [a - b_ for a, b_ in zip(post, base[k])
                 if a is not None and b_ is not None]
            rec[f"d_{k}"] = st.mean(d)
            rec[f"d_{k}_pos"] = f"{sum(1 for x in d if x > 0)}/{len(d)}"
        print(f"[w4] {name:4s} " + " ".join(
            f"Δ{k}={rec[f'd_{k}']:+.5f}({rec[f'd_{k}_pos']})"
            for k in ("R", "A")), flush=True)
        return rec

    results = [run_arm("R", "R"), run_arm("A", "A"),
               run_arm("T", "T"), run_arm("AT", "AT")]
    out = {"aligned_pairs": len(pts), "skipped_align": n_skip_align,
           "skipped_other": n_skip_other, "cos_gR_gT": cos_RT,
           "arms": results}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[w4] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
