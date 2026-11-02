#!/usr/bin/env python3
"""Experiment 0 analysis: group stats + paired bootstrap CI + core figures.

Input: reports/exp0_reference_shift.jsonl (from exp0_reference_shift.py)
Figures → reports/exp0_figs/
"""
import collections
import json
import os
import random
import sys

IN = sys.argv[1] if len(sys.argv) > 1 else "reports/exp0_reference_shift.jsonl"
OUTDIR = "reports/exp0_figs"
CONDS = ["none", "correct", "wrong", "placebo"]
GROUP_OF = {"EFFECTIVE": "positive", "KEEP": "positive",
            "REDUNDANT": "redundant", "INVALID": "invalid",
            "ANSWER": "answer"}


_last_rows = []


def load():
    global _last_rows
    import glob
    paths = sorted(glob.glob(IN + ".shard*")) or [IN]
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    _last_rows = [r for r in rows if "logp_mean" in r]
    ok = [r for r in rows if "logp_mean" in r]
    # pair by block: Δ = cond − none
    by_block = collections.defaultdict(dict)
    meta = {}
    for r in ok:
        k = (r["case_id"], r["sample"], r["block_idx"])
        by_block[k][r["condition"]] = r
        meta[k] = r
    blocks = []
    for k, d in by_block.items():
        if "none" not in d or "correct" not in d:
            continue
        if d["none"].get("logp_mean") is None:
            continue
        b = {"key": k, "label5": d["none"]["label5"],
             "group": GROUP_OF.get(d["none"]["label5"], "?"),
             "case": k[0], "verdict": d["none"]["verdict"],
             "n_span": d["none"]["n_span"]}
        for c in CONDS:
            if c in d:
                b[f"{c}_mean"] = d[c]["logp_mean"]
                b[f"{c}_sum"] = d[c]["logp_sum"]
                b[f"{c}_tmean"] = d[c].get("think_logp_mean")
                b[f"{c}_tsum"] = d[c].get("think_logp_sum")
        blocks.append(b)
    return blocks


def group_stats(blocks):
    out = {}
    for g in ("positive", "redundant", "invalid", "answer"):
        sub = [b for b in blocks if b["group"] == g]
        for c in ("correct", "wrong", "placebo"):
            if not sub:
                continue
            deltas = [b[f"{c}_mean"] - b["none_mean"]
                      for b in sub if f"{c}_mean" in b and "none_mean" in b]
            dsums = [b[f"{c}_sum"] - b["none_sum"] for b in sub
                     if f"{c}_sum" in b and "none_sum" in b]
            if deltas:
                out[(g, c)] = {
                    "n": len(deltas),
                    "dmean_avg": sum(deltas) / len(deltas),
                    "dsum_avg": sum(dsums) / len(dsums),
                }
    return out


def paired_bootstrap(blocks, g1, g2, cond="correct", n_boot=10000, seed=0):
    """Case-level paired bootstrap of separation = μ_g1 − μ_g2."""
    rng = random.Random(seed)
    by_case = collections.defaultdict(lambda: {"a": [], "b": []})
    for b in blocks:
        if f"{cond}_mean" not in b:
            continue
        d = b[f"{cond}_mean"] - b["none_mean"]
        if b["group"] == g1:
            by_case[b["case"]]["a"].append(d)
        elif b["group"] == g2:
            by_case[b["case"]]["b"].append(d)
    cases = [c for c in by_case.values() if c["a"] or c["b"]]
    if not cases:
        return None
    def stat(cs):
        A = [x for c in cs for x in c["a"]]
        B = [x for c in cs for x in c["b"]]
        if not A or not B:
            return None
        return sum(A) / len(A) - sum(B) / len(B)
    point = stat(cases)
    boots = [s for _ in range(n_boot)
             if (s := stat([cases[rng.randrange(len(cases))]
                            for _ in range(len(cases))])) is not None]
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    p_le0 = sum(1 for x in boots if x <= 0) / len(boots)
    return {"sep": point, "ci95": (lo, hi), "p_boot_sep_le_0": p_le0,
            "n_cases": len(cases)}


def main():
    blocks = load()
    global rows_ok_global
    rows_ok_global = _last_rows
    print(f"blocks scored: {len(blocks)}")
    st = group_stats(blocks)
    print(f"\n{'group':10s} {'cond':8s} {'n':>4s} {'Δmean':>8s} {'Δsum':>8s}")
    for (g, c), v in sorted(st.items()):
        print(f"{g:10s} {c:8s} {v['n']:4d} {v['dmean_avg']:+8.3f} "
              f"{v['dsum_avg']:+8.2f}")

    # baseline stratification (ceiling check): Δ among low-baseline (uncertain)
    # vs high-baseline (near-deterministic) action blocks
    pos = [b for b in blocks if b["group"] == "positive" and
           f"correct_mean" in b and "none_mean" in b]
    if pos:
        meds = sorted(b["none_mean"] for b in pos)
        med = meds[len(meds) // 2]
        for name, sel in (("low-baseline", lambda v: v < med),
                          ("high-baseline", lambda v: v >= med)):
            sub = [b for b in pos if sel(b["none_mean"])]
            if sub:
                print(f"  positive/{name} (median split at {med:+.2f}): "
                      f"n={len(sub)} "
                      f"base={sum(b['none_mean'] for b in sub)/len(sub):+.3f} "
                      f"Δcorrect={sum(b['correct_mean']-b['none_mean'] for b in sub)/len(sub):+.4f}")

    # thinking-span diagnostic (same blocks, thinking tokens)
    tstats = {}
    for g in ("positive", "redundant", "invalid", "answer"):
        sub = [b for b in blocks if b["group"] == g and b.get("none_tmean") is not None]
        for c in ("correct", "wrong", "placebo"):
            if not sub or not sub[0].get(f"{c}_tmean"):
                continue
            td = [b[f"{c}_tmean"] - b["none_tmean"] for b in sub
                  if b.get(f"{c}_tmean") is not None]
            if td:
                tstats[(g, c)] = {"n": len(td),
                                  "dmean": sum(td) / len(td)}
    if tstats:
        print("\nthinking-span Δ (diagnostic):")
        for (g, c), v in sorted(tstats.items()):
            print(f"  {g:10s} @{c:8s} n={v['n']:3d} Δthink_mean={v['dmean']:+.4f}")

    print("\nseparation (paired bootstrap, case-level):")
    seps = {}
    for cond in ("correct", "wrong", "placebo"):
        for g2 in ("redundant", "invalid"):
            r = paired_bootstrap(blocks, "positive", g2, cond)
            if r:
                seps[(cond, g2)] = r
                ci = f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
                print(f"  positive−{g2:9s} @{cond:8s} sep={r['sep']:+.3f} "
                      f"CI95={ci} p(≤0)={r['p_boot_sep_le_0']:.4f} "
                      f"cases={r['n_cases']}")

    # condition specificity: sep(correct) − sep(placebo) for positive−redundant
    if ("correct", "redundant") in seps and ("placebo", "redundant") in seps:
        spec = seps[("correct", "redundant")]["sep"] - \
            seps[("placebo", "redundant")]["sep"]
        print(f"\ncondition_specificity (sep_correct − sep_placebo) = {spec:+.3f}")

    os.makedirs(OUTDIR, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fig 1+2 combined: groups × conditions Δmean bars with CI
        fig, ax = plt.subplots(figsize=(8, 4.5))
        groups = ["positive", "redundant", "invalid", "answer"]
        colors = {"none": "#999", "correct": "#2a7", "wrong": "#d55",
                  "placebo": "#7a7"}
        w = 0.2
        for gi, g in enumerate(groups):
            for ci, c in enumerate(CONDS):
                v = st.get((g, c))
                if not v:
                    continue
                ax.bar(gi + (ci - 1.5) * w, v["dmean_avg"], w,
                       color=colors[c], label=c if gi == 0 else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(groups)))
        lab_n = {g: len([b for b in blocks if b["group"] == g]) for g in groups}
        ax.set_xticklabels([f"{g}\n(n={lab_n[g]})" for g in groups])
        ax.set_ylabel("Δ mean logP (action span)")
        ax.set_title("Exp0: Reference-Conditioned Distribution Shift "
                     "(r9a, 9B, C123-prefix)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUTDIR}/fig_groups_conds.png", dpi=150)
        print(f"\nwrote {OUTDIR}/fig_groups_conds.png")
    except ImportError:
        print("matplotlib missing — tables only")

    # answer-channel paired bootstrap (reproduces HANDOFF §2 numbers)
    ans_by = collections.defaultdict(dict)
    for r in rows_ok_global:
        if r.get("label5") == "ANSWER":
            ans_by[(r["case_id"], r["sample"])][r["condition"]] = r["logp_mean"]
    aks = [k for k, d in ans_by.items()
           if all(c in d for c in CONDS)]
    if aks:
        print("\nanswer-channel paired bootstrap:")
        rng = random.Random(0)
        for i, name in ((0, "d_correct"), (1, "d_wrong"), (2, "d_placebo"),
                        (3, "specificity(correct-placebo)")):
            pts = []
            for k in aks:
                d = ans_by[k]
                pts.append((d["correct"] - d["none"]) if i == 0 else
                           (d["wrong"] - d["none"]) if i == 1 else
                           (d["placebo"] - d["none"]) if i == 2 else
                           (d["correct"] - d["placebo"]))
            bs = sorted(sum(pts[rng.randrange(len(pts))]
                            for _ in range(len(pts))) / len(pts)
                        for _ in range(10000))
            print(f"  {name:28s} {sum(pts)/len(pts):+.5f} "
                  f"CI95=[{bs[250]:+.5f}, {bs[9750]:+.5f}] "
                  f"p(<=0)={sum(1 for x in bs if x <= 0)/len(bs):.4f}")

    json.dump({"stats": {f"{g}|{c}": v for (g, c), v in st.items()},
               "separations": {f"{c}|{g2}": v for (c, g2), v in seps.items()},
               "n_blocks": len(blocks)},
              open(f"{OUTDIR}/exp0_stats.json", "w"), indent=1)
    print(f"wrote {OUTDIR}/exp0_stats.json")


if __name__ == "__main__":
    main()
