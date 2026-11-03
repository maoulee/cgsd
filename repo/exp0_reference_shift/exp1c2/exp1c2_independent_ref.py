#!/usr/bin/env python3
"""Exp1-C2: Independent-reference candidate preference (audit 2026-11-03).

Fixes vs exp1c_candidate_margin.py:
  1. NO same-trajectory leakage: the refined reference is rendered from a
     DIFFERENT successful sample of the same case (highest-f1 sibling);
     the scored trajectory never participates in reference construction.
  2. Wrong-reference control: t2_wrong = refined reference rendered from a
     DIFFERENT case's successful trajectory (format-matched, wrong case).
  3. Unified serialization: a_pos AND a_neg both go through the same
     json.dumps path (no original-text vs re-serialized asymmetry).
  4. Report mean AND sum (sequence) margins; soft distractor only (hard
     collapsed to soft whenever len(relations)==1 — 67/124 in exp1c).

Conditions: none / t1 answer-only / t2_ind independent correct reference /
t2_wrong cross-case reference. Pairs keep their case id for case-level
clustering; per-pair flag ref_overlap = whether the reference's relation
choices at the corresponding stage overlap the scored action's relations
(the clean subset is no-overlap).

Usage:
  SEQ_PROMPT=C123 python scripts/exp1c2_independent_ref.py --gpu 0 \
      --shards 2 --shard 0 --out reports/exp1c2_margin_9b.jsonl
"""
import argparse
import hashlib
import json
import os
import re
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


def parse_action(c):
    for m in re.finditer(r"\{.*\}", c, re.DOTALL):
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
                            line.split(":", 1)[1].split("|") if x.strip()]
    return []


def ref_relations_of(tr):
    """All relations the reference trajectory chose (for overlap flag)."""
    out = set()
    for st in tr:
        if st.get("role") != "assistant":
            continue
        a = parse_action(st.get("content") or "")
        if a:
            out.update((a.get("args") or {}).get("relations") or [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
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

    # group verdict rows by case; pick reference = best sibling sample
    by_case = {}
    for v in verdicts:
        by_case.setdefault(v["case_id"], []).append(v)
    pairs = []
    for cid, vs in sorted(by_case.items()):
        ok = sorted([v for v in vs if v.get("f1", 0) > 0 and
                     (v["case_id"], int(v["sample"])) in by_key],
                    key=lambda v: -v.get("f1", 0))
        if len(ok) < 2:
            continue                    # need a sibling distinct from scored
        ref = ok[0]                      # highest-f1 sibling = reference source
        ref_row = by_key[(cid, int(ref["sample"]))]
        for scored in ok[1:]:
            tr = by_key[(cid, int(scored["sample"]))]["trajectory"]
            golds = scored["golds"] or ["unknown"]
            ref_rels = ref_relations_of(ref_row["trajectory"])
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
                pairs.append({
                    "case_id": cid, "sample": scored["sample"], "idx": idx,
                    "golds": golds, "tr": tr,
                    "a_pos": json.dumps(act, ensure_ascii=False),
                    "a_neg": json.dumps(neg, ensure_ascii=False),
                    "ref_overlap": int(bool(set(rels) & ref_rels)),
                })
    # cross-case wrong reference: next case's best sibling (deterministic)
    case_refs = {}
    for cid, vs in sorted(by_case.items()):
        ok = sorted([v for v in vs if v.get("f1", 0) > 0 and
                     (v["case_id"], int(v["sample"])) in by_key],
                    key=lambda v: -v.get("f1", 0))
        if ok:
            v0 = ok[0]
            case_refs[cid] = (v0["golds"] or ["unknown"],
                              by_key[(cid, int(v0["sample"]))]["trajectory"])
    cids = list(case_refs)
    my = [p for k, p in enumerate(pairs) if k % args.shards == args.shard]
    print(f"[exp1c2] {len(pairs)} pairs | shard{args.shard}: {len(my)}",
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
                lp = torch.log_softmax(model.lm_head(h[s:e]).float(), -1)
                out += lp.gather(1, ids_t[0, s + 1:e + 1]
                                 .unsqueeze(1)).squeeze(1).tolist()
        return out

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
            rg, rtr = case_refs[p["case_id"]]
            ci = cids.index(p["case_id"])
            wg, wtr = case_refs[cids[(ci + 1) % len(cids)]]
            sufs = {
                "none": "",
                "t1": REF_TMPL.format(vals=" | ".join(p["golds"])),
                "t2_ind": "\n\n" + wrap_structured(render_fact_grouped(
                    rtr, answer=rg, gold=rg)),
                "t2_wrong": "\n\n" + wrap_structured(render_fact_grouped(
                    wtr, answer=wg, gold=wg)),
            }
            row = {"case_id": p["case_id"], "sample": p["sample"],
                   "block_idx": idx, "ref_overlap": p["ref_overlap"]}
            try:
                for cname, suf in sufs.items():
                    for cand in ("a_pos", "a_neg"):
                        full_text = (f"<think>\n{thought}\n</think>\n\n"
                                     + p[cand]) if thought else p[cand]
                        f_ids = render(tok, system_base + suf, user, hist,
                                       full_text)
                        c_ids = render(tok, system_base + suf, user, hist,
                                       think_text)
                        span = span_logp(f_ids, len(c_ids), len(f_ids))
                        if span:
                            row[f"{cname}_{cand}_mean"] = sum(span) / len(span)
                            row[f"{cname}_{cand}_sum"] = sum(span)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row["error"] = "oom"
            f.write(json.dumps(row) + "\n"); f.flush()
            if (n + 1) % 10 == 0:
                print(f"[exp1c2] shard{args.shard} {n+1}/{len(my)}",
                      flush=True)
    print(f"[exp1c2] shard{args.shard} done", flush=True)


if __name__ == "__main__":
    main()
