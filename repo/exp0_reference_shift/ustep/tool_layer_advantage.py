#!/usr/bin/env python3
"""Per-tool-layer teacher advantage: q−p on ACTION tokens, grouped by tool.

For every assistant turn (flat MD protocol, raw text), score the action
span (from the `tool:` line onward) under (a) bare system, (b) bare +
refined reference (sibling). Group by tool type: plan / retrieve_relations /
retrieve_subgraph / answer. This is the data-driven supervision mask:
supervise tool layers (and fields) where q−p > 0.

Usage:
  SEQ_PROMPT=C123 python scripts/tool_layer_advantage.py --gpu 0 \
      --out reports/tool_layer_advantage.json
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
    VERDICTS, ROLLOUT, PROMPT_SHA, build_user_msg, render)
from build_teacher_prompts import (  # noqa: E402
    render_fact_grouped, wrap_structured)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

CH = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="reports/tool_layer_advantage.json")
    args = ap.parse_args()

    from kgqa.agent.seq_react_loop import seq_agents_md
    system_base = seq_agents_md()
    assert hashlib.sha256(system_base.encode()).hexdigest() == PROMPT_SHA

    verdicts = json.load(open(VERDICTS))
    rollout = json.load(open(ROLLOUT))
    by_key = {(r["case_id"], int(r["sample_idx"])): r for r in rollout}
    by_case = collections.defaultdict(list)
    for v in verdicts:
        by_case[v["case_id"]].append(v)

    # one scored sample per case (its best sibling provides the reference)
    jobs = []
    for cid in sorted(by_case):
        ok = sorted([v for v in by_case[cid] if v.get("f1", 0) > 0 and
                     (cid, int(v["sample"])) in by_key],
                    key=lambda v: -v.get("f1", 0))
        if len(ok) < 2:
            continue
        ref = ok[0]
        ref_row = by_key[(cid, int(ref["sample"]))]
        golds = ref["golds"] or ["unknown"]
        ref_suffix = "\n\n" + wrap_structured(render_fact_grouped(
            ref_row["trajectory"], answer=golds, gold=golds))
        scored = ok[1]
        row = by_key[(cid, int(scored["sample"]))]
        tr = row["trajectory"]
        user = build_user_msg(row)
        hist, n_prev = [], None
        msgs = [{"role": "system", "content": system_base},
                {"role": "user", "content": user}]
        t0 = tok_text = None
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        t0 = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=False)
        n_prev = len(tok(t0, add_special_tokens=False).input_ids)
        for i, m in enumerate(tr):
            if m.get("role") == "assistant":
                raw = (m.get("raw") or m.get("content") or "")
                m_act = re.search(r"(?:^|\n)tool:\s*(\w+)", raw)
                if not m_act:
                    continue
                tool = m_act.group(1)
                act_start = m_act.start()
                think_part = raw[:act_start].rstrip() + "\n\n" \
                    if act_start > 0 else " "
                act_part = raw[act_start:].strip()
                hist_here = list(hist)
                jobs.append({"cid": cid, "tool": tool, "user": user,
                             "hist": hist_here, "think": think_part,
                             "act": act_part, "ref": ref_suffix,
                             "golds": golds})
                hist.append({"role": "assistant", "content": raw})
            else:
                hist.append({"role": m.get("role"), "content": m.get("content"),
                             "name": m.get("name")})
    print(f"[tla] {len(jobs)} tool turns across "
          f"{len({j['cid'] for j in jobs})} cases", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}", trust_remote_code=True)
    model.eval()

    def span_logp(ids, lo, hi):
        ids_t = torch.tensor([ids], device=model.device)
        am = torch.ones_like(ids_t)
        with torch.no_grad():
            h = model.model(input_ids=ids_t,
                            attention_mask=am).last_hidden_state[0]
            out = []
            for s in range(lo - 1, hi - 1, CH):
                e = min(s + CH, hi - 1)
                lp = torch.log_softmax(model.lm_head(h[s:e]).float(), -1)
                out += lp.gather(1, ids_t[0, s + 1:e + 1]
                                 .unsqueeze(1)).squeeze(1).tolist()
        return st.mean(out) if out else None

    groups = collections.defaultdict(list)
    for n, j in enumerate(jobs):
        full = f"{j['think']}{j['act']}" if j["think"].strip() else j["act"]
        cut = j["think"] if j["think"].strip() else " "
        f_b = render(tok, system_base, j["user"], j["hist"], full)
        c_b = render(tok, system_base, j["user"], j["hist"], cut)
        f_r = render(tok, system_base + j["ref"], j["user"], j["hist"], full)
        c_r = render(tok, system_base + j["ref"], j["user"], j["hist"], cut)
        pb = span_logp(f_b, len(c_b), len(f_b))
        pr = span_logp(f_r, len(c_r), len(f_r))
        if pb is not None and pr is not None:
            groups[j["tool"]].append(pr - pb)
        if (n + 1) % 20 == 0:
            print(f"[tla] {n+1}/{len(jobs)}", flush=True)

    out = {}
    print(f"\n{'tool':18s} {'n':>4s} {'q−p (action tokens)':>20s} "
          f"{'pos%':>6s}")
    for tool, xs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        out[tool] = {"n": len(xs), "mean": st.mean(xs),
                     "pos_pct": sum(1 for x in xs if x > 0) / len(xs) * 100}
        print(f"{tool:18s} {len(xs):4d} {st.mean(xs):+20.5f} "
              f"{out[tool]['pos_pct']:5.0f}%")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[tla] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
