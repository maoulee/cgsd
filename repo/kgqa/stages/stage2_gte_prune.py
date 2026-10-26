"""Stage 2: GTE retrieval and LLM pruning for triple decomposition.

Per-triple GTE retrieval + LLM prune. Replaces old Stage 2 + 3 + 4.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional

from kgqa.core.utils import rel_to_text
from kgqa.stages.stage2_entity import gte_retrieve
from kgqa.llm.batch import batch_call_llm
from kgqa.llm.prompts import TRIPLE_PRUNE_PROMPT
from kgqa.core.utils import sample_triple_batch, extract_xml_tag
from kgqa.core.config import PRUNE_TOPK


# Patterns that implicitly ask for profession/occupation but GTE cannot map.
_IMPLICIT_PROFESSION_PATTERNS = re.compile(
    r"(?:what\s+(?:did|does)\s+.+?\s+do(?:\b|$)|"
    r"actions?\s+(?:or|and)\s+(?:deeds|contributions)|"
    r"career\s+activit|"
    r"(?:known|famous|notable)\s+for\s+doing|"
    r"profession|occupation|career|job\s+title)",
    re.IGNORECASE,
)


async def stage_2_gte_and_prune(session, cases, adaptive_routing: bool = False):
    """Per-triple GTE retrieval + LLM prune. Replaces old Stage 2 + 3 + 4."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # ── Phase A: Per-triple GTE retrieval (predicate + sub-question + expansion) ──
    gte_calls = []  # (cs, triple_idx, query, source)
    for cs in active:
        sub_questions = getattr(cs, 'sub_questions', [])
        for i, triple in enumerate(cs.triples):
            # Primary: predicate
            query = triple.get("predicate", "")
            if query.strip():
                gte_calls.append((cs, i, query.strip(), "predicate"))
            # Secondary: sub-question (richer semantics)
            src_sq = triple.get("source_subquestion", 0)
            sq_text = sub_questions[src_sq - 1] if 0 < src_sq <= len(sub_questions) else ""
            if sq_text.strip() and sq_text.strip() != query.strip():
                gte_calls.append((cs, i, sq_text.strip(), "subquestion"))
            # Expansion: detect implicit profession queries, add targeted GTE query
            combined = f"{query} {sq_text} {cs.question}"
            if _IMPLICIT_PROFESSION_PATTERNS.search(combined):
                gte_calls.append((cs, i, "profession or occupation", "expansion"))

    _gte_sem = asyncio.Semaphore(3)
    async def _gte_limited(coro):
        async with _gte_sem:
            return await coro

    gte_results = await asyncio.gather(*[
        _gte_limited(gte_retrieve(session, query, cs.rels,
                                  candidate_texts=getattr(cs, 'rel_texts', None),
                                  top_k=15))
        for cs, ti, query, src in gte_calls
    ], return_exceptions=True)

    # Distribute GTE results, merging predicate + subquestion hits per triple
    for (cs, ti, query, source), result in zip(gte_calls, gte_results):
        if isinstance(result, Exception):
            continue
        if not hasattr(cs, '_triple_gte') or cs._triple_gte is None:
            cs._triple_gte = {}
        if ti not in cs._triple_gte:
            cs._triple_gte[ti] = {"gte_all": {}, "topk": [], "queries": []}

        for rank, r in enumerate(result):
            cand = r.get("candidate", "")
            score = round(r.get("score", 0), 4)
            idx_in_rels = cs.rels.index(cand) if cand in cs.rels else None
            # Keep best score across both queries
            if idx_in_rels is not None:
                existing = cs._triple_gte[ti]["gte_all"].get(idx_in_rels)
                if existing is None or score > existing[1]:
                    cs._triple_gte[ti]["gte_all"][idx_in_rels] = (cand, score)

        cs._triple_gte[ti]["queries"].append({"query": query, "source": source})

    # Rebuild topk list from merged gte_all for each triple
    for cs in active:
        if not hasattr(cs, '_triple_gte') or not cs._triple_gte:
            continue
        for ti, gd in cs._triple_gte.items():
            sorted_rels = sorted(gd["gte_all"].items(), key=lambda x: -x[1][1])
            topk_list = []
            for rank, (idx_in_rels, (cand, score)) in enumerate(sorted_rels):
                topk_list.append({
                    "rank": rank + 1,
                    "rel_idx": idx_in_rels,
                    "candidate": cand,
                    "score": score,
                })
            gd["topk"] = topk_list

    # Also populate legacy step_candidates / gte_per_step for downstream compat
    for cs in active:
        cs.step_candidates = {}
        cs.gte_per_step = {}
        cs.relation_retrieval_details = []
        if hasattr(cs, '_triple_gte') and cs._triple_gte:
            for ti, gd in cs._triple_gte.items():
                step_num = ti + 1
                sorted_cands = sorted(gd["gte_all"].items(), key=lambda x: -x[1][1])
                cs.step_candidates[step_num] = [(idx, name, score) for idx, (name, score) in sorted_cands]
                cs.gte_per_step[step_num] = gd["gte_all"]
                cs.relation_retrieval_details.append({
                    "step": step_num,
                    "queries": gd.get("queries", []),
                    "gte_candidates_count": len(gd["gte_all"]),
                })

    gte_dt = time.perf_counter() - _t0
    print(f"  Stage 2a (Per-triple GTE): {gte_dt:.2f}s | {len(gte_calls)} calls")

    # ── Phase A.5: SIMPLE shortcut — take GTE top-PRUNE_TOPK directly (no LLM prune) ──
    # Adaptive routing only. When disabled, every case is complexity="complex"
    # and this branch is a no-op, so behavior is byte-identical to today.
    # The resulting cs.step_relations has the SAME shape the LLM prune path
    # produces: List[set[int]] of relation indices, one set per triple, merged
    # by depth at the end of this stage (same merge block runs for all cases).
    simple_cases = []
    if adaptive_routing:
        simple_cases = [cs for cs in active
                        if cs.active and getattr(cs, 'complexity', 'complex') == 'simple']
    for cs in simple_cases:
        cs.step_relations = []
        for ti in range(len(cs.triples)):
            cands = cs.step_candidates.get(ti + 1, [])
            cs.step_relations.append(set(idx for idx, _, _ in cands[:PRUNE_TOPK]))
        if not cs.triples:
            cs.step_relations = [set() for _ in range(max(len(cs.steps), 1))]
        cs.prune_debug = {"adaptive_shortcut": "simple_topk", "prune_skipped": True}

    # ── Phase B: LLM Prune with per-triple candidate blocks ──
    # COMPLEX cases (and ALL cases when adaptive_routing is off) run the
    # original LLM prune path unchanged. SIMPLE cases are excluded here.
    prune_active = [cs for cs in active
                    if not (adaptive_routing and getattr(cs, 'complexity', 'complex') == 'simple')]
    prompts = []
    prune_cases = []
    for cs in prune_active:
        if not cs.triples or not hasattr(cs, '_triple_gte') or not cs._triple_gte:
            # Fallback: top-3 GTE for each step
            cs.step_relations = []
            for ti in range(len(cs.triples)):
                cands = cs.step_candidates.get(ti + 1, [])
                cs.step_relations.append(set(idx for idx, _, _ in cands[:PRUNE_TOPK]))
            if not cs.triples:
                cs.step_relations = [set() for _ in range(max(len(cs.steps), 1))]
            continue

        # Build chain text (with sub-questions for context)
        sub_questions = getattr(cs, 'sub_questions', [])
        chain_lines = []
        for i, t in enumerate(cs.triples):
            src_sq = t.get("source_subquestion", 0)
            sq_text = sub_questions[src_sq - 1] if 0 < src_sq <= len(sub_questions) else ""
            if sq_text:
                chain_lines.append(f"  Step {i+1}: {sq_text}")
                chain_lines.append(f"    {t['subject']} —[{t['predicate']}]—> {t['object']}")
            else:
                chain_lines.append(f"  Step {i+1}: {t['subject']} —[{t['predicate']}]—> {t['object']}")
        chain_text = "\n".join(chain_lines)

        # Build candidate blocks
        all_rel_indices = set()
        for ti, gd in cs._triple_gte.items():
            for r in gd["topk"]:
                if r["rel_idx"] is not None:
                    all_rel_indices.add(r["rel_idx"])

        triple_samples = sample_triple_batch(
            list(all_rel_indices), cs.h_ids, cs.r_ids, cs.t_ids, cs.ents, cs.rels
        ) if all_rel_indices else {}

        blocks = []
        for i, t in enumerate(cs.triples):
            gd = cs._triple_gte.get(i, {})
            topk = gd.get("topk", [])
            cand_lines = []
            for r in topk:
                line = f"    {r['rank']}. {r['candidate']}"
                sample = triple_samples.get(r.get("rel_idx"), "")
                if sample:
                    line += f"\n       e.g. {sample}"
                cand_lines.append(line)
            if not cand_lines:
                cand_lines = ["    (no candidates)"]

            src_sq = t.get("source_subquestion", 0)
            sq_text = sub_questions[src_sq - 1] if 0 < src_sq <= len(sub_questions) else ""
            if sq_text:
                header = (
                    f"Step {i+1}: {sq_text}\n"
                    f"  {t['subject']} —[{t['predicate']}]—> {t['object']}\n"
                    f"Candidates:\n"
                )
            else:
                header = (
                    f"Step {i+1}: {t['subject']} —[{t['predicate']}]—> {t['object']}\n"
                    f"Candidates:\n"
                )
            blocks.append(header + "\n".join(cand_lines))

        blocks_text = "\n\n".join(blocks)

        prompt = TRIPLE_PRUNE_PROMPT.format(
            question=cs.question,
            chain_text=chain_text,
            blocks_text=blocks_text,
        )
        prompts.append([
            {"role": "system", "content": "You are a KGQA relation selector. Select exactly 3 best KG relations per step, ranked by semantic validity. Check e.g. examples for direction before selecting. Output <analysis> and <answer> XML tags with parseable JSON."},
            {"role": "user", "content": prompt},
        ])
        cs.prune_prompt = prompt
        prune_cases.append(cs)

    if prompts:
        _prune_t0 = time.perf_counter()
        prune_responses = await batch_call_llm(session, prompts, max_tokens=1000)
        _prune_dt = time.perf_counter() - _prune_t0
        print(f"    LLM prune batch: {_prune_dt:.2f}s | {len(prompts)} cases")

        for cs, raw in zip(prune_cases, prune_responses):
            cs.prune_debug = {"raw": raw, "prompt": getattr(cs, 'prune_prompt', '')}

            # Parse <answer> JSON
            selected = {}
            m = re.search(r'<answer>(.*?)</answer>', raw or "", re.DOTALL)
            if m:
                try:
                    ans = json.loads(m.group(1).strip().replace("'", '"'))
                    for sr in ans.get("selected_relations", []):
                        step_str = sr.get("step", "")
                        m2 = re.match(r'step[_ ]?(\d+)', step_str, re.IGNORECASE)
                        step_num = int(m2.group(1)) if m2 else 1
                        ids = sr.get("selected_relation_ids", [])
                        selected[step_num] = ids
                except (json.JSONDecodeError, ValueError):
                    pass

            # Fallback: try <selected> XML format too
            if not selected:
                sel_text = extract_xml_tag(raw or "", "selected") or ""
                for line in sel_text.split('\n'):
                    line = line.strip()
                    m3 = re.match(r'step_(\d+)\s*:\s*\[(.*?)\]', line)
                    if m3:
                        sn = int(m3.group(1))
                        nums = [int(x.strip()) for x in m3.group(2).split(',') if x.strip().isdigit()]
                        selected[sn] = nums

            # Convert selected IDs to relation indices, cap at topk
            cs.step_relations = []
            topk = PRUNE_TOPK
            for ti in range(len(cs.triples)):
                step_num = ti + 1
                cands = cs.step_candidates.get(step_num, [])
                ids = selected.get(step_num, [])
                rel_indices = []
                seen = set()
                # Take LLM's top-k ranked selections
                for rid in ids[:topk]:
                    if 1 <= rid <= len(cands):
                        idx = cands[rid - 1][0]
                        if idx not in seen:
                            rel_indices.append(idx)
                            seen.add(idx)
                # Pad from GTE only if LLM selected < topk
                if len(rel_indices) < topk and cands:
                    for idx, _, _ in cands:
                        if idx not in seen:
                            rel_indices.append(idx)
                            seen.add(idx)
                            if len(rel_indices) >= topk:
                                break
                if not rel_indices and cands:
                    rel_indices = [idx for idx, _, _ in cands[:topk]]
                cs.step_relations.append(set(rel_indices))

    else:
        # All cases had fallback
        for cs in active:
            if not getattr(cs, 'step_relations', None):
                cs.step_relations = [set() for _ in range(max(len(cs.steps), 1))]

    # Merge step_relations for triples at the same depth (branch grouping)
    for cs in active:
        steps = getattr(cs, 'steps', [])
        if not steps or len(cs.step_relations) <= 1:
            continue
        # Build triple_idx → depth mapping
        depths = {}
        for s in steps:
            ti = s.get("triple_idx")
            d = s.get("depth")
            if ti is not None and d is not None:
                depths[ti] = d
        if not depths:
            continue
        # Group by depth
        depth_groups = {}
        for ti, d in depths.items():
            depth_groups.setdefault(d, []).append(ti)
        # Only merge if there are actually branched triples (depth with 2+ triples)
        has_branch = any(len(tis) > 1 for tis in depth_groups.values())
        if not has_branch:
            continue
        # Merge: sort depths, build new step_relations
        sorted_depths = sorted(depth_groups.keys())
        merged = []
        for d in sorted_depths:
            combined = set()
            for ti in depth_groups[d]:
                if ti < len(cs.step_relations):
                    combined |= cs.step_relations[ti]
            merged.append(combined)
        cs.step_relations = merged

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["gte_prune"] = dt / max(len(active), 1)
    n_steps = sum(len(cs.step_relations) for cs in active)
    n_rels = sum(sum(len(s) for s in cs.step_relations) for cs in active)
    print(f"  Stage 2b (GTE+Prune): {dt:.2f}s | {len(active)} cases, {n_steps} steps, {n_rels} relations selected")
