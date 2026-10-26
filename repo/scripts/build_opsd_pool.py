#!/usr/bin/env python3
"""Build the OPSD training pool: hit cases → 8-row blocks → stratified shuffle.

Replaces the fixed 32-case file (12.8 epochs over 100 steps, 10/32 WebQTest
contamination) with a large pre-shuffled pool drawn from TRAIN splits only.
The trainer consumes rows in FILE ORDER (KGQA_GROUP_BY_CASE only disables
shuffling), so a pre-shuffled pool gives random-without-replacement case
sampling per step with ZERO training-code changes: step k takes blocks
[32(k-1), 32k) — fresh cases every step, exactly 8 trajectories each.

Two modes:

  per-source  (rollout results + source pkl → one labeled pool):
    python scripts/build_opsd_pool.py reports/pool_cwq.json \
        --pkl data/cwq_processed/train_v4_repaired.pkl -o data/opsd/pool_cwq.jsonl

  --mix       (labeled pools → ratio-stratified training pool):
    python scripts/build_opsd_pool.py --mix \
        data/opsd/pool_cwq.jsonl=24 data/opsd/pool_webqsp.jsonl=8 \
        --seed 42 -o data/opsd/opsd_pool.jsonl

The mix emits 32-block cycles with labels shuffled WITHIN each cycle, so
every optimizer step (32 cases at batch 256) gets exactly the ratio
composition (e.g. 24 CWQ + 8 WebQSP) in random order.

Teacher references come from the rollout's OWN hit trajectory (self-
distillation semantics: reference = what the starting policy achieved), via
build_teacher_prompts.action_blocks_msgs — no dump files needed, the
results json carries the full trajectory per record.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_teacher_prompts import (  # noqa: E402
    render_fact_grouped, wrap_structured)

SYSTEM_PROMPT = "You are a knowledge-graph QA assistant."


def _candidate_score(rec: dict, script: str):
    """Rank candidate refined references of ONE case.

    PRIMARY (user ruling 2026-10-25, prior design): score = f1 + p — the SUM
    of the ORIGINAL trajectory's answer F1 and the REFINED trajectory's
    answer probability. Both live on [0,1], so the sum ranks "correct AND
    plausible" without either term dominating. Correctness alone lets a
    LUCKY path become the case standard; probability alone prefers easy
    paths. (Product was my misreading — retracted.)

    SECONDARY (tie-break, and the effective criterion when probabilities
    were not captured — pre-ROLL_LOGPROBS pools): completeness of the
    advanced plan facts, then fewer rounds, then shorter text."""
    f1 = float(rec.get("f1") or 0.0)
    lp = rec.get("answer_logprob")
    if lp is None:
        lp = rec.get("mean_logprob")
    prob = math.exp(lp) if isinstance(lp, (int, float)) else None
    primary = (f1 + prob) if prob is not None else f1
    served = script.count("这一步服务事实")
    rounds = script.count("回合（")
    return (round(primary, 6), served, -rounds, -len(script))


def _evidence_text(rec: dict) -> str:
    """Text blob the answer must be supported by: the rollout's evidence
    entities when recorded, else every tool-result message of the trajectory
    (works for dump-sourced smoke data too)."""
    ev = rec.get("evidence_entities")
    if ev:
        return " | ".join(str(x) for x in ev)
    # ACCUMULATED evidence (user correction 2026-10-25): everything the
    # retrievals delivered across the episode — NOT just the last couple of
    # replies. The NBA Finals case proved the point: all 17 gold answers
    # were in the tool results, the model simply answered one of them
    # (a REASONING-layer failure, not missing evidence). Restricting to the
    # tail threw away 16 supported answers and mislabelled the case as
    # "evidence incomplete".
    parts = []
    msgs = rec.get("trajectory") or rec.get("messages") or []
    for m in msgs:
        role = m.get("role")
        c = str(m.get("content") or "")
        if role == "tool":                      # offline pool trajectory form
            parts.append(c)
        elif role == "user" and c.lstrip().startswith("Tool result ("):
            parts.append(c)
    return "\n".join(parts)


def _answerable(rec: dict, min_f1: float) -> bool:
    """Refinement gate (user ruling 2026-10-25): a trajectory that answered
    nothing / scored 0 is NOT refined at all — refining a fully wrong path
    would manufacture a fake "best experience". Partial answers DO count
    (f1 > min_f1): they carry a correct sub-chain worth distilling."""
    try:
        f1 = float(rec.get("f1") or 0.0)
    except (TypeError, ValueError):
        return False
    if f1 <= min_f1:
        return False
    if not (rec.get("answer") or rec.get("trajectory")):
        return False
    return True


def _case_hit_rates(rows: list[dict]) -> dict:
    """case_id -> (hits, n) over ALL samples of that case (f1 >= 0.999)."""
    st = {}
    for r in rows:
        cid = r.get("case_id")
        if not cid:
            continue
        h, n = st.get(cid, (0, 0))
        st[cid] = (h + (1 if float(r.get("f1") or 0) >= 0.999 else 0), n + 1)
    return st


def build_source(rows: list[dict], pkl: dict, samples: int, min_blocks: int,
                 prefixes: list[str] | None = None,
                 min_f1: float = 0.0,
                 variance_only: bool = False) -> list[list[dict]]:
    """Per-case blocks of `samples` training rows.

    Grouped in ONE pass (a per-case rescan was O(n^2) — 24k pool rows made it
    unusable). Every answerable sample of a case is a candidate; the best one
    (`_candidate_score` = f1 + answer probability) becomes the frozen
    reference for that case, shared by all its rollouts (OPSD semantics).
    """
    by_case: dict[str, list[dict]] = {}
    total_hits: dict[str, int] = {}
    for r in rows:
        cid = r.get("case_id")
        if not cid:
            continue
        by_case.setdefault(cid, []).append(r)
        if float(r.get("f1") or 0) >= 0.999:
            total_hits[cid] = total_hits.get(cid, 0) + 1

    out_blocks: list[list[dict]] = []
    n_hit = n_no_src = n_thin = n_prefix = n_alt = n_prob = n_degen = 0
    for cid, recs in by_case.items():
        if prefixes and not cid.startswith(tuple(prefixes)):
            n_prefix += 1
            continue
        if variance_only:
            # the case-granular mask discards all-correct / all-wrong CASES
            # wholesale (swift's fixed sampling is the reason) — such a case
            # contributes ZERO gradient, i.e. wasted rollout budget.
            n = len(recs)
            h = total_hits.get(cid, 0)
            if n and (h == 0 or h == n):
                n_degen += 1
                continue
        cands = []
        for r in recs:
            if not _answerable(r, min_f1):
                continue
            sc = render_fact_grouped(r.get("trajectory") or [],
                                     answer=r.get("answer"),
                                     gold=(pkl.get(cid) or {}).get("a_entity"),
                                     evidence_text=_evidence_text(r))
            if sc:
                cands.append((_candidate_score(r, sc), sc, r))
        if not cands:
            n_no_src += 1
            continue
        sample = pkl.get(cid)
        if sample is None:
            n_no_src += 1
            continue
        if len(cands) > 1:
            n_alt += 1
        cands.sort(key=lambda c: c[0], reverse=True)
        _, script, best = cands[0]
        n_prob += 1 if (best.get("answer_logprob") is not None
                        or best.get("mean_logprob") is not None) else 0
        if min_blocks and "FACT" not in script and "Relation search" not in script:
            n_thin += 1          # no parseable decisions (fact-grouped form)
            continue
        n_hit += 1
        prompt = wrap_structured(script)
        question = str(sample.get("question") or best.get("question") or "")
        base = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "teacher_prompt": prompt,
            "kgqa_sample": sample,
            "kgqa_pilot": {
                "case_id": cid,
                "question": question,
                "gt_answers": list(sample.get("a_entity") or []),
            },
        }
        out_blocks.append([dict(base, kgqa_idx=i) for i in range(samples)])
    print(f"  refined_cases={n_hit} (f1>{min_f1}) no_source={n_no_src} "
          f"prefix_filtered={n_prefix} thin={n_thin} "
          f"multi_candidate={n_alt} prob_scored={n_prob} "
          f"degen_skipped={n_degen} → {len(out_blocks)} case blocks", flush=True)
    return out_blocks


def stratified_mix(pools: dict[str, list[list[dict]]], ratio: dict[str, int],
                   rng: random.Random, max_cycles: int = 0) -> list[list[dict]]:
    """Emit sum(ratio)-block cycles, labels shuffled within each cycle.

    Every FULL cycle carries exactly the ratio composition (one optimizer
    step at batch 256 = 32 cases = one 32-block cycle at 24:8). A label
    whose deck runs dry recycles its blocks (with replacement) while other
    pools still have stock; the run ends when every deck is empty, so the
    final partial cycle may under-fill (real pools are far larger than the
    3,200 case slots a 100-step run consumes — never hit in practice).
    """
    decks = {lab: list(blks) for lab, blks in pools.items()}
    for deck in decks.values():
        rng.shuffle(deck)
    cycle: list[str] = [lab for lab, n in ratio.items() for _ in range(n)]
    out: list[list[dict]] = []
    cycles = 0
    while any(decks.values()):
        rng.shuffle(cycle)
        emitted = 0
        for lab in cycle:
            if not decks[lab]:
                if not any(d for l2, d in decks.items() if l2 != lab):
                    continue            # everything else is dry: tail slot
                decks[lab] = list(pools[lab])
                rng.shuffle(decks[lab])
                print(f"  ⚠ {lab} pool exhausted — recycling its blocks "
                      f"(cases will repeat)", file=sys.stderr, flush=True)
            out.append(decks[lab].pop())
            emitted += 1
        if emitted < len(cycle):
            break                       # tail partial cycle; all pools dry
        cycles += 1
        if max_cycles and cycles >= max_cycles:
            break
    return out


def load_pool_blocks(path: Path) -> list[list[dict]]:
    """Group a pool jsonl back into case blocks (8 consecutive rows each)."""
    # iterate the FILE (split on "\n" only): str.splitlines() also breaks on
    # \u2028 / \x85 / \x1c, which appear inside tool-result text and would
    # slice a JSON line in half ("Unterminated string" on an intact file)
    with open(path, "r", encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    blocks: list[list[dict]] = []
    for i in range(0, len(rows), 8):
        blk = rows[i:i + 8]
        ids = {r["kgqa_pilot"]["case_id"] for r in blk}
        if len(blk) != 8 or len(ids) != 1:
            raise SystemExit(f"{path}: rows {i}..{i + len(blk)} are not one "
                             f"8-row case block (ids={ids})")
        blocks.append(blk)
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=str, nargs="*",
                    help="source mode: rollout OUT json | mix mode: pool.jsonl=ratio ...")
    ap.add_argument("--mix", action="store_true",
                    help="mix mode: every input is POOL.jsonl=RATIO")
    ap.add_argument("--pkl", type=Path, help="source case pkl (graph arrays)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=8,
                    help="rows per case block (= trajectories per case)")
    ap.add_argument("--max-cycles", type=int, default=0,
                    help="mix mode: stop after N cycles (= N*sum(ratio) case "
                         "blocks). One cycle fills exactly one optimizer step "
                         "at the ratio composition — 25 cycles = 800 cases = "
                         "100 steps at 8 cases/step.")
    ap.add_argument("--variance-only", action="store_true",
                    help="keep only cases whose 8-sample hit rate is strictly "
                         "between 0 and n (all-correct / all-wrong cases are "
                         "masked wholesale by the case-granular rule → zero "
                         "gradient, wasted rollout budget)")
    ap.add_argument("--min-f1", type=float, default=0.0,
                    help="refinement gate: refine only trajectories with "
                         "f1 > this (default 0 = 'answered or partially "
                         "answered'; fully wrong paths are never refined)")
    ap.add_argument("--min-blocks", type=int, default=1,
                    help="min action blocks in the reference solution")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="cap pool size in cases (staged rollout: first N hits)")
    ap.add_argument("--prefix", action="append", default=None,
                    help="keep only case ids starting with this prefix "
                         "(repeatable). NOTE 2026-10-24 user ruling: CWQ "
                         "train is used IN FULL — the WebQTrn-/WebQTest- "
                         "prefixes only mark which WebQSP split the BASE "
                         "question came from (CWQ is a WebQSP variant set); "
                         "the split that matters is the DATASET's own. "
                         "Prefix filtering is OFF by default; use only for "
                         "explicit ablations.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mix:
        if not args.inputs:
            ap.error("--mix needs POOL.jsonl=RATIO inputs")
        pools: dict[str, list[list[dict]]] = {}
        ratio: dict[str, int] = {}
        for spec in args.inputs:
            path_s, _, r_s = spec.partition("=")
            if not r_s:
                ap.error(f"bad mix spec {spec!r} (want POOL.jsonl=RATIO)")
            pools[path_s] = load_pool_blocks(Path(path_s))
            ratio[path_s] = int(r_s)
        blocks = stratified_mix(pools, ratio, random.Random(args.seed),
                                args.max_cycles)
        print(f"  mix: " + ", ".join(
                  f"{Path(p).name}={len(b)} blocks ×{ratio[p]}"
                  for p, b in pools.items())
              + f" | cycle={sum(ratio.values())} blocks", flush=True)
    else:
        if len(args.inputs) != 1 or args.pkl is None:
            ap.error("source mode: exactly one RESULTS json and --pkl "
                     "(mix mode: --mix POOL.jsonl=RATIO ...)")
        rows = json.loads(Path(args.inputs[0]).read_text())
        pkl_rows = pickle.loads(args.pkl.read_bytes())
        pkl = {s["id"]: s for s in pkl_rows if s.get("a_entity")}
        blocks = build_source(rows, pkl, args.samples, args.min_blocks,
                              args.prefix, args.min_f1, args.variance_only)
        if args.max_cases:
            blocks = blocks[:args.max_cases]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for blk in blocks:
            for row in blk:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(blocks)} cases × {len(blocks[0]) if blocks else 0} rows "
          f"→ {args.out} ({sum(len(b) for b in blocks)} rows)", flush=True)


if __name__ == "__main__":
    main()
