#!/usr/bin/env python3
"""Exp1-C2 analysis — CORRECTED case-level bootstrap (audit 2026-11-03).

Fix vs previous version: the old bootstrap sampled one trajectory WITHIN
each fixed case (rng.choice(c) for c in cs) — that measures within-case
variation, not case resampling risk. Correct scheme: resample CASES with
replacement, then average the case means (case-equal weighting).

Also reports: true case count, case-equal-weight mean, leave-one-case-out
sensitivity, and accepts both `<in>.shard*` shards and a merged `<in>.jsonl`.

Usage: python scripts/exp1c2_analyze.py reports/exp1c2_margin_9b
"""
import collections
import glob
import json
import os
import random
import statistics as st
import sys

IN = sys.argv[1] if len(sys.argv) > 1 else "reports/exp1c2_margin_9b"
N_BOOT = 20000


def load_rows(in_path):
    shards = sorted(glob.glob(in_path + ".shard*"))
    if shards:
        rows = []
        for p in shards:
            rows += [json.loads(l) for l in open(p)]
        return rows
    if os.path.exists(in_path):
        return [json.loads(l) for l in open(in_path)]
    if os.path.exists(in_path + ".jsonl"):
        return [json.loads(l) for l in open(in_path + ".jsonl")]
    raise SystemExit(f"no input for {in_path} (.shard* / .jsonl / merged)")


def case_bootstrap(case_means, rng, n_boot=N_BOOT):
    """Resample CASES with replacement; return sorted bootstrap means."""
    cs = list(case_means)
    return sorted(sum(rng.choice(cs) for _ in cs) / len(cs)
                  for _ in range(n_boot))


def main():
    rows = load_rows(IN)
    ok = [r for r in rows if "none_a_pos_mean" in r
          and "t2_ind_a_pos_mean" in r]
    n_cases = len({r["case_id"] for r in ok})
    print(f"pairs: {len(ok)} | cases: {n_cases} | "
          f"errors: {sum(1 for r in rows if 'error' in r)}")

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
            pair_mean = st.mean(g)
            by = collections.defaultdict(list)
            for r, gi in zip(sub, g):
                by[r["case_id"]].append(gi)
            case_means = [st.mean(v) for v in by.values()]
            case_mean = st.mean(case_means)
            pos_cases = sum(1 for x in case_means if x > 0)
            rng = random.Random(0)
            bs = case_bootstrap(case_means, rng)
            lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))]
            p = sum(1 for x in bs if x <= 0) / len(bs)
            # leave-one-case-out sensitivity
            loco = [st.mean(case_means[:i] + case_means[i + 1:])
                    for i in range(len(case_means))]
            worst = min(loco)
            print(f"  G_{cond:8s} pair-mean={pair_mean:+.5f} | "
                  f"case-equal={case_mean:+.5f} "
                  f"CI95(case)=[{lo:+.5f},{hi:+.5f}] p(<=0)={p:.4f} | "
                  f"pos cases {pos_cases}/{len(case_means)} | "
                  f"LOCO-worst={worst:+.5f}")

    rep(ok, "ALL")
    rep([r for r in ok if r["ref_overlap"] == 0],
        "no-overlap (reference never chose the scored relations)")
    rep([r for r in ok if r["ref_overlap"] == 1],
        "overlap (sibling shares >=1 relation)")


if __name__ == "__main__":
    main()
