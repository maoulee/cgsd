#!/usr/bin/env python3
"""Exp1-A: Full Distribution Audit (follows the 2026-11-02 Codex audit).

Exp0 measured only the SELECTED token's log-prob under reference injection.
GKD optimizes the distribution over the whole vocabulary (top-256 renorm).
Same realized-token logprob can coexist with a large KL(q||p) — this audit
measures, per position, on the same prefixes as Exp0 (system injection):

  kl_full    = D_KL(q_t || p_t)          over full vocab
  kl_top256  = D_KL(q̃ || p̃)             on teacher(=cond) top-256, renorm
               both sides (beta=0 FKL convention of opsd_train.yaml)
  + teacher/student mass on the top-256 support, top-1 change rate,
    entropy delta, and the realized-token shift (Exp0 metric, recomputed).

p = π(·|h)  (none), q = π(·|h,z) (correct / wrong / placebo).
Spans: thinking and action, reported separately.

Usage (one shard per GPU, one model at a time):
  SEQ_PROMPT=C123 python scripts/exp1_dist_audit.py --model models/Qwen3.5-9B \
      --gpu 0 --shards 2 --shard 0 --out reports/exp1_dist_9b.jsonl
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SEQ_PROMPT", "C123")
os.environ.setdefault("SEQ_ADAPTIVE_ROUTES", "1")

from exp0_reference_shift import (  # noqa: E402
    VERDICTS, ROLLOUT, PROMPT_SHA, REF_TMPL, NEUTRAL, label_of,
    assistant_full_text, build_user_msg, render)

import torch  # noqa: E402

TOPK = 256
CHUNK = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from kgqa.agent.seq_react_loop import seq_agents_md
    from transformers import AutoModelForCausalLM, AutoTokenizer

    system_base = seq_agents_md()
    import hashlib as _h
    assert _h.sha256(system_base.encode()).hexdigest() == PROMPT_SHA

    verdicts = json.load(open(VERDICTS))
    rollout = json.load(open(ROLLOUT))
    by_key = {(r["case_id"], int(r["sample_idx"])): r for r in rollout}
    vrows = sorted([v for v in verdicts if v.get("f1", 0) > 0 and
                    (v["case_id"], int(v["sample"])) in by_key],
                   key=lambda v: v["case_id"])
    wrong_of = {}
    for i, v in enumerate(vrows):
        w = vrows[(i + 1) % len(vrows)]
        g = v["golds"] or ["unknown"]
        wg = w["golds"] or ["unknown"]
        # entity-overlap assertion (audit item): reject a pairing that shares
        # any answer entity with the case
        if set(map(str.lower, g)) & set(map(str.lower, wg)):
            w2 = vrows[(i + 2) % len(vrows)]
            wg = w2["golds"] or ["unknown"]
        wrong_of[v["case_id"]] = [wg[j % len(wg)] for j in range(len(g))]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}", trust_remote_code=True)
    model.eval()
    mshort = "9b" if "9B" in args.model else "4b"

    # ── build turn list (block turns + answer turns, as exp0) ──────────
    from kgqa.rl.turn_credit import accepted_answer_turn
    turns = []          # (v, tr, idx_toolresult_or_assistant+1, lab, verdict)
    for v in vrows:
        tr = by_key[(v["case_id"], int(v["sample"]))]["trajectory"]
        for b in v["blocks"]:
            lab = label_of(b.get("verdict", ""))
            if lab == "UNKNOWN":
                continue
            idx = int(b["idx"])
            if idx < 1 or idx >= len(tr) or tr[idx - 1].get("role") != "assistant":
                continue
            turns.append((v, tr, idx, lab, b["verdict"]))
        ai = accepted_answer_turn(tr)
        if ai is not None:
            turns.append((v, tr, ai + 1, "ANSWER", "final-accepted-answer"))
    my_turns = [t for k, t in enumerate(turns) if k % args.shards == args.shard]
    print(f"[exp1a:{mshort}] shard{args.shard}: {len(my_turns)} turns "
          f"(of {len(turns)})", flush=True)

    def cond_suffixes(v):
        golds = v["golds"] or ["unknown"]
        return {
            "none": "",
            "correct": REF_TMPL.format(vals=" | ".join(golds)),
            "wrong": REF_TMPL.format(vals=" | ".join(wrong_of[v["case_id"]])),
            "placebo": REF_TMPL.format(
                vals=" | ".join(NEUTRAL[:max(1, len(golds))])),
        }

    def span_lsm(f_ids, n_lo, n_hi):
        """Full-vocab log_softmax rows for token positions [n_lo, n_hi)."""
        ids_t = torch.tensor([f_ids], device=model.device)
        am = torch.ones_like(ids_t)
        with torch.no_grad():
            hidden = model.model(input_ids=ids_t,
                                 attention_mask=am).last_hidden_state[0]
            rows = []
            for s in range(n_lo - 1, n_hi - 1, CHUNK):
                e = min(s + CHUNK, n_hi - 1)
                logits = model.lm_head(hidden[s:e])
                rows.append(torch.log_softmax(logits.float(), dim=-1))
        return torch.cat(rows), ids_t

    def kl_top256(lp_q, lp_p):
        tq, iq = torch.topk(lp_q, TOPK, dim=-1)
        qm = torch.softmax(tq, dim=-1)              # renorm teacher on support
        pm_raw = lp_p.gather(1, iq)
        pm = torch.softmax(pm_raw, dim=-1)          # renorm student on support
        return (qm * (torch.log(qm + 1e-12) - torch.log(pm + 1e-12))).sum(-1)

    def metrics(lp_p, lp_q, ids_t, lo):
        with torch.no_grad():
            n = lp_p.shape[0]
            tgt = ids_t[0, lo:lo + n]
            out = {}
            out["logp_shift_mean"] = float(
                (lp_q.gather(1, tgt.unsqueeze(1))
                 - lp_p.gather(1, tgt.unsqueeze(1))).mean())
            out["kl_full_mean"] = float(
                (lp_q.exp() * (lp_q - lp_p)).sum(-1).mean())
            out["kl_full_sum"] = float((lp_q.exp() * (lp_q - lp_p)).sum(-1).sum())
            out["kl_top256_mean"] = float(kl_top256(lp_q, lp_p).mean())
            tq, iq = torch.topk(lp_q, TOPK, dim=-1)
            out["teacher_mass256_mean"] = float(tq.exp().sum(-1).mean())
            out["student_mass256_mean"] = float(
                lp_p.gather(1, iq).exp().sum(-1).mean())
            out["top1_change_rate"] = float(
                (lp_q.argmax(-1) != lp_p.argmax(-1)).float().mean())
            Hq = -(lp_q.exp() * lp_q).sum(-1)
            Hp = -(lp_p.exp() * lp_p).sum(-1)
            out["entropy_delta_mean"] = float((Hq - Hp).mean())
            return out

    with open(f"{args.out}.shard{args.shard}", "w") as f:
        for n, (v, tr, idx, lab, verdict) in enumerate(my_turns):
            st = tr[idx - 1]
            full_text = assistant_full_text(st)
            thought = (st.get("reasoning") or "").strip()
            cut_wrap, cut_act = " ", (f"<think>\n{thought}\n</think>\n\n"
                                      if thought else " ")
            user = build_user_msg(by_key[(v["case_id"], int(v["sample"]))])
            hist = []
            for m in tr[:idx - 1]:
                if m.get("role") == "assistant":
                    hist.append({"role": "assistant",
                                 "content": assistant_full_text(m)})
                else:
                    hist.append({"role": m.get("role"),
                                 "content": m.get("content"),
                                 "name": m.get("name")})
            base = {"case_id": v["case_id"], "sample": v["sample"],
                    "block_idx": idx, "label5": lab, "verdict": verdict,
                    "model": mshort}
            try:
                sufs = cond_suffixes(v)
                lp_p = None
                ids_ref = None
                geoms = {}
                for cname, suf in sufs.items():
                    sys_text = system_base + suf
                    f_ids = render(tok, sys_text, user, hist, full_text)
                    a_ids = render(tok, sys_text, user, hist, cut_act)
                    w_ids = render(tok, sys_text, user, hist, cut_wrap)
                    if cname == "none":
                        lp_p_t, ids_ref = span_lsm(f_ids, len(w_ids), len(a_ids))
                        lp_p_a, _ = span_lsm(f_ids, len(a_ids), len(f_ids))
                        ids_t = ids_ref
                    else:
                        lp_q_t, _ = span_lsm(f_ids, len(w_ids), len(a_ids))
                        lp_q_a, _ = span_lsm(f_ids, len(a_ids), len(f_ids))
                        row = dict(base)
                        row["condition"] = cname
                        row["n_think"], row["n_action"] = (
                            int(lp_p_t.shape[0]), int(lp_p_a.shape[0]))
                        for span, lpq, lpp, lo in (
                                ("think", lp_q_t, lp_p_t, len(w_ids)),
                                ("act", lp_q_a, lp_p_a, len(a_ids))):
                            if lpp.shape[0] == 0:
                                continue
                            m_ = metrics(lpp, lpq, ids_t, lo)
                            for k, val in m_.items():
                                row[f"{span}_{k}"] = round(val, 6)
                        f.write(json.dumps(row) + "\n")
                f.flush()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                f.write(json.dumps({**base, "error": "oom"}) + "\n"); f.flush()
                continue
            if (n + 1) % 10 == 0:
                print(f"[exp1a:{mshort}] shard{args.shard} {n+1}/"
                      f"{len(my_turns)}", flush=True)
    print(f"[exp1a:{mshort}] shard{args.shard} done", flush=True)


if __name__ == "__main__":
    main()
