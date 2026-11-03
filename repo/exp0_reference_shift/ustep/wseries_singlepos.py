#!/usr/bin/env python3
"""W-series: single-position learning verification (Codex minimal experiment).

Question: from OPD first principles (grad = p − q), a frozen teacher with a
better distribution on the correct-relation tokens SHOULD teach the student
in-situ. If even a decision-token-only GKD step fails to raise p(a+) on the
very state it trained on, the fault is in the training function (loss/
alignment/top-K/optimizer), not knowledge transfer.

Per train-case EFFECTIVE decision point:
  premise   : q(a+|h,R) − p(a+|h) > 0 on the DECISION tokens (relation
              name tokens inside the retrieve_subgraph action JSON)
  arms      : W1 loss on decision tokens only | W2 whole action JSON
              | W3 thinking+action (full turn)
  readout   : after ONE stabilized Adam step (frozen teacher, token-mean),
              re-score p(a+) decision tokens on the SAME states; also the
              same-state full-action margin M.
Escalation logic: if W1 learns and W3 doesn't → multi-position gradient
interference; if W1 can't learn in-situ → training-function bug hunt.

Usage:
  SEQ_PROMPT=C123 python scripts/wseries_singlepos.py --gpu 0 \
      --out reports/wseries.json
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
    VERDICTS, ROLLOUT, PROMPT_SHA, label_of, assistant_full_text,
    build_user_msg, render)
from exp1c2_independent_ref import parse_action, menu_before  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TOPK = 256
TEMP = 1.0
LR = 2e-5
LORA_PLUS = 4.0
CH = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=10, help="decision points")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--out", default="reports/wseries.json")
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
                     if sum(1 for v in vs if v.get("f1", 0) > 0
                            and (cid, int(v["sample"])) in by_key) >= 2)
    train_c = [c for i, c in enumerate(cids_ok) if i % 4 != 3]

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

    def rel_span_in_action(action_text):
        """Token index range (within the action text) covering the relations
        array VALUES. Char-locate `"relations": [` ... `]` then map to tokens
        by prefix tokenization."""
        m = re.search(r'"relations"\s*:\s*\[', action_text)
        if not m:
            return None
        j = action_text.find("]", m.end())
        lo_c, hi_c = m.end(), j if j > 0 else len(action_text)
        pre = len(tok(action_text[:lo_c], add_special_tokens=False).input_ids)
        seg = len(tok(action_text[lo_c:hi_c], add_special_tokens=False)
                  .input_ids)
        return (pre, pre + seg)

    pts = []
    for cid in train_c:
        ok = sorted([v for v in by_case[cid] if v.get("f1", 0) > 0 and
                     (cid, int(v["sample"])) in by_key],
                    key=lambda v: -v.get("f1", 0))
        ref_row = by_key[(cid, int(ok[0]["sample"]))]
        ref_suffix = "\n\n" + wrap_structured(render_fact_grouped(
            ref_row["trajectory"], answer=ok[0]["golds"] or ["unknown"],
            gold=ok[0]["golds"] or ["unknown"]))
        for scored in ok[1:]:
            tr = by_key[(cid, int(scored["sample"]))]["trajectory"]
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
                a_text = json.dumps(act, ensure_ascii=False)
                rsp = rel_span_in_action(a_text)
                if not rsp:
                    continue
                thought = (st_.get("reasoning") or "").strip()
                think = (f"<think>\n{thought}\n</think>\n\n"
                         if thought else " ")
                hist = []
                for mmsg in tr[:idx - 1]:
                    if mmsg.get("role") == "assistant":
                        hist.append({"role": "assistant",
                                     "content": assistant_full_text(mmsg)})
                    else:
                        hist.append({"role": mmsg.get("role"),
                                     "content": mmsg.get("content"),
                                     "name": mmsg.get("name")})
                pts.append({"cid": cid, "user": build_user_msg(
                    by_key[(cid, int(scored["sample"]))]),
                    "hist": hist, "think": think, "a_text": a_text,
                    "rel_span": rsp, "ref_suffix": ref_suffix})
            if len(pts) >= args.n:
                break
        if len(pts) >= args.n:
            break
    print(f"[w] {len(pts)} decision points (train cases)", flush=True)

    def render_pair(p, suffix, full_text):
        f = render(tok, system_base + suffix, p["user"], p["hist"], full_text)
        c = render(tok, system_base + suffix, p["user"], p["hist"],
                   p["think"] if p["think"].strip() != "" else " ")
        return f, c

    def span_logp(ids, lo, hi, adapter_on=True):
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
            for s in range(lo - 1, hi - 1, CH):
                e = min(s + CH, hi - 1)
                lp = torch.log_softmax(
                    pm.base_model.model.lm_head(h[s:e]).float(), -1)
                out += lp.gather(1, ids_t[0, s + 1:e + 1]
                                 .unsqueeze(1)).squeeze(1).tolist()
        return out

    def fwd_logits(ids, preds, grad, adapter_off=False):
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

    # ── premise + baseline scores (θ0) ─────────────────────────────────
    base = []
    for p in pts:
        full = f"{p['think']}{p['a_text']}" if p["think"].strip() else p["a_text"]
        f_ids, c_ids = render_pair(p, "", full)
        n_act_start = len(c_ids)
        r_lo, r_hi = p["rel_span"]
        dec_lo, dec_hi = n_act_start + r_lo, n_act_start + r_hi
        lp = span_logp(f_ids, dec_lo, dec_hi, adapter_on=False)
        # teacher (frozen base + reference): positions recomputed on ITS render
        f_ids_t, c_ids_t = render_pair(p, p["ref_suffix"], full)
        dec_lo_t = len(c_ids_t) + r_lo
        lp_t = span_logp(f_ids_t, dec_lo_t, dec_lo_t + (r_hi - r_lo),
                         adapter_on=False)
        base.append({"f_ids": f_ids, "f_ids_t": f_ids_t,
                     "act_lo": n_act_start, "act_hi": len(f_ids),
                     "dec_lo": dec_lo, "dec_hi": dec_hi,
                     "p_dec": st.mean(lp), "q_dec": st.mean(lp_t),
                     "gap": st.mean(lp_t) - st.mean(lp)})
    gaps = [b["gap"] for b in base]
    print(f"[w] premise q−p on decision tokens: mean={st.mean(gaps):+.5f} "
          f"pos={sum(1 for g in gaps if g > 0)}/{len(gaps)}", flush=True)

    def run_arm(name, scope):
        for k in adapter0:
            pm.state_dict()[k].copy_(adapter0[k])
        pm.zero_grad(set_to_none=True)
        pm.train()
        eff_lr = LR / 5.0
        opt = torch.optim.AdamW(
            [{"params": [q for n_, q in pm.named_parameters()
                         if "lora_A" in n_], "lr": eff_lr},
             {"params": [q for n_, q in pm.named_parameters()
                         if "lora_B" in n_], "lr": eff_lr * LORA_PLUS}],
            lr=eff_lr, weight_decay=0.0)
        all_preds = []
        for p, b in zip(pts, base):
            if scope == "decision":
                rng = range(b["dec_lo"], b["dec_hi"])
            elif scope == "action":
                rng = range(b["act_lo"], b["act_hi"])
            else:                                  # full turn
                rng = range(0 if p["think"].strip() else b["act_lo"],
                            b["act_hi"])
            all_preds.append(sorted(rng))
        N = sum(len(x) for x in all_preds)
        for _ in range(args.steps):
            opt.zero_grad(set_to_none=True)
            for p, b, preds in zip(pts, base, all_preds):
                # teacher view boundaries shift by the suffix delta
                delta = len(b["f_ids_t"]) - len(b["f_ids"])
                t_preds = [x + delta for x in preds]
                # chunked fwd/bwd (peak [chunk, V] logits only)
                for c0 in range(0, len(preds), 256):
                    cp = preds[c0:c0 + 256]
                    ct = t_preds[c0:c0 + 256]
                    s_l = fwd_logits(b["f_ids"], cp, grad=True)
                    t_l = fwd_logits(b["f_ids_t"], ct, grad=False,
                                     adapter_off=True)  # frozen teacher
                    kl = kl_top256(s_l, t_l)
                    (kl.sum() / N).backward()
                    del s_l, t_l
            torch.nn.utils.clip_grad_norm_(
                [q for q in pm.parameters() if q.grad is not None], 1.0)
            opt.step()
        pm.eval()
        ddec, dact = [], []
        for p, b in zip(pts, base):
            lp = st.mean(span_logp(b["f_ids"], b["dec_lo"], b["dec_hi"]))
            ddec.append(lp - b["p_dec"])
            la = st.mean(span_logp(b["f_ids"], b["act_lo"], b["act_hi"]))
            dact.append(la - b["p_dec"])
        rec = {"arm": name, "scope": scope, "steps": args.steps,
               "d_p_dec_mean": st.mean(ddec),
               "d_p_dec_pos": f"{sum(1 for x in ddec if x > 0)}/{len(ddec)}",
               "d_act_mean": st.mean(dact)}
        print(f"[w] {name:12s} Δp(decision tokens)={rec['d_p_dec_mean']:+.5f} "
              f"pos={rec['d_p_dec_pos']} | Δp(action span)={rec['d_act_mean']:+.5f}",
              flush=True)
        return rec

    results = [run_arm("W1_decision", "decision"),
               run_arm("W2_action", "action"),
               run_arm("W3_turn", "turn")]
    out = {"premise_gap_mean": st.mean(gaps),
           "premise_gap_pos": f"{sum(1 for g in gaps if g > 0)}/{len(gaps)}",
           "arms": results}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[w] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
