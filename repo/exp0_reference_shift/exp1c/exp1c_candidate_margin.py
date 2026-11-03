#!/usr/bin/env python3
"""Exp1-C: Candidate Action Preference under reference conditions.

The decisive direction test: does the OPSD refined reference (T2) shift the
model's RELATIVE preference toward the environment-validated effective action
(a+) over same-state legal distractors (a-), beyond what no-reference gives?

  G_c = [s_c(a+) - s_c(a-)] - [s_0(a+) - s_0(a-)],  s = action-span mean logp
  conditions c: T1 answer-only, T2 refined (render_fact_grouped + wrap)
  candidates: a+ = original action JSON (judged EFFECTIVE by RDL);
              a-soft = same JSON, 1 chosen relation swapped for an unchosen
                       menu relation; a-hard = all relations replaced.

Distractors come from the SAME retrieve_relations menu at the same state —
env-legal, format/length-matched. Scoring position = post-thinking (the
original thinking stays in the prefix; Codex decision point B).

Usage (dual GPU):
  SEQ_PROMPT=C123 python scripts/exp1c_candidate_margin.py --gpu 0 --shards 2 --shard 0 --out reports/exp1c_margin.jsonl
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SEQ_PROMPT", "C123")
os.environ.setdefault("SEQ_ADAPTIVE_ROUTES", "1")

from exp0_reference_shift import (  # noqa: E402
    VERDICTS, ROLLOUT, PROMPT_SHA, REF_TMPL, label_of,
    assistant_full_text, build_user_msg, render)

import torch  # noqa: E402

CH = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from kgqa.agent.seq_react_loop import seq_agents_md
    from build_teacher_prompts import render_fact_grouped, wrap_structured
    from transformers import AutoModelForCausalLM, AutoTokenizer

    system_base = seq_agents_md()
    assert hashlib.sha256(system_base.encode()).hexdigest() == PROMPT_SHA

    verdicts = json.load(open(VERDICTS))
    rollout = json.load(open(ROLLOUT))
    by_key = {(r["case_id"], int(r["sample_idx"])): r for r in rollout}

    def parse_action(c):
        for m in __import__("re").finditer(r"\{.*\}", c, 16):
            try:
                j = json.loads(m.group(0))
                if j.get("tool") == "retrieve_subgraph":
                    return j
            except Exception:
                pass
        return None

    def menu_before(tr, idx):
        for j in range(idx - 1, -1, -1):
            st = tr[j]
            if st.get("role") == "tool" and "retrieve_relations" in \
                    str(st.get("name", "")):
                for line in (st.get("content") or "").splitlines():
                    if line.startswith("candidate_relations:"):
                        return [x.strip() for x in
                                line.split(":", 1)[1].split("|")
                                if x.strip()]
        return []

    pairs = []
    for v in verdicts:
        if v.get("f1", 0) <= 0:
            continue
        key = (v["case_id"], int(v["sample"]))
        if key not in by_key:
            continue
        tr = by_key[key]["trajectory"]
        golds = v["golds"] or ["unknown"]
        for b in v["blocks"]:
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
            if len(rels) < 1 or len(unchosen) < 1:
                continue
            soft = json.loads(json.dumps(act))
            soft["args"]["relations"] = [unchosen[0]] + rels[1:]
            hard = json.loads(json.dumps(act))
            hard["args"]["relations"] = [unchosen[i % len(unchosen)]
                                         for i in range(len(rels))]
            pairs.append({
                "case_id": v["case_id"], "sample": v["sample"], "idx": idx,
                "golds": golds, "tr": tr,
                "a_pos": tr[idx - 1].get("content").strip(),
                "a_soft": json.dumps(soft, ensure_ascii=False),
                "a_hard": json.dumps(hard, ensure_ascii=False),
            })
    if args.limit:
        pairs = pairs[:args.limit]
    my = [p for k, p in enumerate(pairs) if k % args.shards == args.shard]
    print(f"[exp1c] {len(pairs)} pairs | shard{args.shard}: {len(my)}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}", trust_remote_code=True)
    model.eval()

    def span_logp(ids, n_lo, n_hi):
        ids_t = torch.tensor([ids], device=model.device)
        am = torch.ones_like(ids_t)
        with torch.no_grad():
            h = model.model(input_ids=ids_t,
                            attention_mask=am).last_hidden_state[0]
            out = []
            for s in range(n_lo - 1, n_hi - 1, CH):
                e = min(s + CH, n_hi - 1)
                lg = model.lm_head(h[s:e]).float()
                lp = torch.log_softmax(lg, -1)
                tgt = ids_t[0, s + 1:e + 1]
                out += lp.gather(1, tgt.unsqueeze(1)).squeeze(1).tolist()
        return out

    def suffixes(p):
        t2 = "\n\n" + wrap_structured(render_fact_grouped(
            p["tr"], answer=p["golds"], gold=p["golds"]))
        return {"none": "",
                "t1": REF_TMPL.format(vals=" | ".join(p["golds"])),
                "t2": t2}

    with open(f"{args.out}.shard{args.shard}", "w") as f:
        for n, p in enumerate(my):
            tr, idx = p["tr"], p["idx"]
            user = build_user_msg(by_key[(p["case_id"], p["sample"])])
            st = tr[idx - 1]
            thought = (st.get("reasoning") or "").strip()
            think_text = (f"<think>\n{thought}\n</think>\n\n"
                          if thought else " ")
            hist = []
            for m in tr[:idx - 1]:
                if m.get("role") == "assistant":
                    hist.append({"role": "assistant",
                                 "content": assistant_full_text(m)})
                else:
                    hist.append({"role": m.get("role"),
                                 "content": m.get("content"),
                                 "name": m.get("name")})
            row = {"case_id": p["case_id"], "sample": p["sample"],
                   "block_idx": idx}
            try:
                for cname, suf in suffixes(p).items():
                    for cand in ("a_pos", "a_soft", "a_hard"):
                        full = think_text if thought else think_text
                        full_text = (f"<think>\n{thought}\n</think>\n\n"
                                     + p[cand]) if thought else p[cand]
                        cut = (" " if not thought else think_text)
                        f_ids = render(tok, system_base + suf, user, hist,
                                       full_text)
                        c_ids = render(tok, system_base + suf, user, hist,
                                       cut if cut != " " else " ")
                        span = span_logp(f_ids, len(c_ids), len(f_ids))
                        if span:
                            row[f"{cname}_{cand}_mean"] = sum(span) / len(span)
                            row[f"{cname}_{cand}_sum"] = sum(span)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row["error"] = "oom"
            f.write(json.dumps(row) + "\n"); f.flush()
            if (n + 1) % 10 == 0:
                print(f"[exp1c] shard{args.shard} {n+1}/{len(my)}",
                      flush=True)
    print(f"[exp1c] shard{args.shard} done", flush=True)


if __name__ == "__main__":
    main()
