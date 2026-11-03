#!/usr/bin/env python3
"""Exp1-C2 analysis: G margins with CASE-LEVEL bootstrap (reproducible).

Usage: python scripts/exp1c2_analyze.py reports/exp1c2_margin_9b.jsonl
"""
import collections
import glob
import json
import random
import statistics as st
import sys

IN = sys.argv[1] if len(sys.argv) > 1 else "reports/exp1c2_margin_9b.jsonl"


def main():
    rows = []
    for p in sorted(glob.glob(IN + ".shard*")):
        rows += [json.loads(l) for l in open(p)]
    ok = [r for r in rows if "none_a_pos_mean" in r
          and "t2_ind_a_pos_mean" in r]
    print(f"pairs: {len(ok)} | errors: {sum(1 for r in rows if 'error' in r)}"
          f" | cases: {len({r['case_id'] for r in ok})}")

    def m(r, c):
        return r[f"{c}_a_pos_mean"] - r[f"{c}_a_neg_mean"]

    def rep(sub, tag):
        if not sub:
            return
        nc = len({r["case_id"] for r in sub})
        print(f"\n{tag} (n={len(sub)}, cases={nc}) "
              f"base margin={st.mean([m(r,'none') for r in sub]):+.4f}")
        for cond in ("t1", "t2_ind", "t2_wrong"):
            g = [m(r, cond) - m(r, "none") for r in sub]
            by = collections.defaultdict(list)
            for r, gi in zip(sub, g):
                by[r["case_id"]].append(gi)
            cs = list(by.values())
            rng = random.Random(0)
            bs = sorted(sum(rng.choice(c) for c in cs) / len(cs)
                        for _ in range(10000))
            print(f"  G_{cond:8s} mean={st.mean(g):+.5f} "
                  f"CI95(case)=[{bs[250]:+.5f},{bs[9750]:+.5f}] "
                  f"p(<=0)={sum(1 for x in bs if x <= 0)/10000:.4f} "
                  f"pos={sum(1 for x in g if x > 0)/len(g)*100:.0f}%")

    rep(ok, "ALL")
    rep([r for r in ok if r["ref_overlap"] == 0],
        "no-overlap (reference never chose the scored relations)")
    rep([r for r in ok if r["ref_overlap"] == 1],
        "overlap (sibling shares >=1 relation)")


if __name__ == "__main__":
    main()
