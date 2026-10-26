"""Stage 6: Diagnosis and retry for candidate explosion cases.

Safety net that re-prunes relations (Level 3) and swaps anchors (Level 4)
when Stage 5 produced too many candidates.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import List

from kgqa.core.case_state import CaseState
from kgqa.core.config import CANDIDATE_THRESHOLD
from kgqa.core.utils import normalize, candidate_hit, strict_candidate_hit, compute_match_stats, extract_xml_tag
from kgqa.traversal.cvt import is_cvt_like, expand_through_cvt
from kgqa.traversal.k_queue import k_queue_traverse
from kgqa.traversal.path_utils import prefer_breakpoint_hit_paths
from kgqa.llm.batch import batch_call_llm


# ---------------------------------------------------------------------------
# Helpers shared with Stage 5 (re-prune prompt building / parsing)
# ---------------------------------------------------------------------------

def _effective_pattern_count(cs):
    """Count valid logical patterns. When endpoints exist, only count endpoint-hitting patterns."""
    lps = getattr(cs, 'logical_paths', [])
    if not lps:
        return len(cs.paths) if cs.paths else 0
    bp_indices = set(cs.breakpoints.values()) if cs.breakpoints else set()
    bp_indices.discard(cs.anchor_idx)
    if bp_indices:
        return sum(1 for lp in lps if lp.get("endpoint"))
    return len(lps)


def _build_prune_all_prompt(question, all_steps, step_candidates):
    """Build the prompt for relation pruning. Returns (prompt_text, system_text)."""
    chain_lines = []
    step_blocks = []
    for s in all_steps:
        sn = s["step"]
        ep_str = f" -> endpoint: {s['endpoint']}" if s.get("endpoint") else ""
        chain_lines.append(f"  Step {sn}: {s['question']}{ep_str}")

        cands = step_candidates.get(sn, [])
        if not cands:
            step_blocks.append(f"Step {sn}: {s['question']}\n  Purpose: {s.get('definition', '')}\n  Candidates: (none)")
            continue

        cand_lines = [f"    {i}. {name}"
                      for i, (idx, name, score) in enumerate(cands, 1)]
        step_blocks.append(
            f"Step {sn}: {s['question']}\n"
            f"  Purpose: {s.get('definition', '')}\n"
            f"  Candidates:\n" + "\n".join(cand_lines)
        )

    chain_text = "\n".join(chain_lines)
    blocks_text = "\n\n".join(step_blocks)

    prompt = f"""Analyze and select knowledge graph relations for each step of this reasoning chain.

Question: {question}

Reasoning chain:
{chain_text}

Step-by-step candidates:
{blocks_text}

Rules:
1. Each step connects FROM previous output TO next — select bridge relations
2. Select 2-4 relevant relations per step. Pick the best candidates that match the step's purpose.
3. When uncertain about a relation's relevance, INCLUDE it — missing a key relation is far worse
   than having an extra irrelevant one.
4. Ignore unrelated attributes (currency, codes when asking about geography)
5. If no relations fit a step, output empty list
6. ORDER matters: rank by relevance to the step (most relevant first)

Output format:
<analysis>
One sentence per step: what it needs and which relations fit.
</analysis>
<selected>
step_1: [1, 3]
step_2: [2, 5]
</selected>"""

    system = "You are a knowledge graph relation selector for multi-step QA. Analyze the full chain, then select relevant relations per step. Output <analysis> and <selected> XML tags."
    return prompt, system


def _parse_prune_result(raw, step_candidates, all_steps):
    """Parse LLM response for relation pruning. Returns dict mapping step_num -> set of selected indices."""
    selected_yaml = extract_xml_tag(raw, "selected")
    result = {}
    if selected_yaml:
        for line in selected_yaml.split('\n'):
            line = line.strip()
            m = __import__('re').match(r'step_(\d+)\s*:\s*\[(.*?)\]', line)
            if m:
                sn = int(m.group(1))
                nums = [int(x.strip()) for x in m.group(2).split(',') if x.strip().isdigit()]
                cands = step_candidates.get(sn, [])
                selected_indices = set()
                for n in nums:
                    if 1 <= n <= len(cands):
                        selected_indices.add(cands[n - 1][0])
                result[sn] = selected_indices

    for s in all_steps:
        sn = s["step"]
        if sn not in result or not result[sn]:
            cands = step_candidates.get(sn, [])
            result[sn] = set(idx for idx, _, _ in cands[:3])
    return result


# ---------------------------------------------------------------------------
# Candidate rebuild helper (shared between L3 and L4)
# ---------------------------------------------------------------------------

def _rebuild_candidates_from_paths(paths_new, anchor_idx, ents, h_ids, r_ids, t_ids):
    """Build deduplicated candidate list from last-2-hop nodes of new paths."""
    last2_new = set()
    for p in paths_new:
        nodes = p.get("nodes", [])
        for n in nodes[-2:]:
            if n != anchor_idx:
                last2_new.add(n)

    new_candidates = []
    for node_idx in sorted(last2_new):
        if node_idx == anchor_idx:
            continue
        name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
        if is_cvt_like(name):
            for cvt_idx, _ in expand_through_cvt(node_idx, h_ids, r_ids, t_ids, ents):
                if cvt_idx != anchor_idx and 0 <= cvt_idx < len(ents) and not is_cvt_like(ents[cvt_idx]):
                    new_candidates.append(ents[cvt_idx])
        else:
            new_candidates.append(name)
    seen_n = set()
    unique_n = []
    for c in new_candidates:
        nc = normalize(c)
        if len(nc) < 2 or not c.isascii():
            continue
        if nc not in seen_n:
            seen_n.add(nc)
            unique_n.append(c)
    return unique_n


def _apply_new_paths(cs, paths_new, new_step_relations, unique_n):
    """Update CaseState with new traversal results."""
    cs.paths = paths_new
    cs.step_relations = new_step_relations
    cs.answer_candidates = unique_n
    all_nodes_new = {cs.anchor_idx}
    for p in paths_new:
        all_nodes_new.update(p.get("nodes", []))
    cs.all_subgraph_nodes = all_nodes_new
    if paths_new:
        cs.max_depth = max(p.get("depth", 0) for p in paths_new)
        cs.max_cov = max(len(p.get("covered_steps", frozenset())) for p in paths_new)
    cs.path_candidates = list(unique_n)
    cs.gt_hit = candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
    cs.gt_hit_strict = strict_candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
    cs.gt_f1 = compute_match_stats(cs.path_candidates, cs.gt_answers)['f1']


# ---------------------------------------------------------------------------
# Stage 6 main
# ---------------------------------------------------------------------------

async def stage_6_diagnosis_retry(session, cases: List[CaseState]):
    """Safety net + Level 3-4 retry for candidate explosion."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    direct_count = sum(1 for cs in active if cs.needs_direct_answer)

    # -- Level 3: Re-prune relations for candidate explosion cases (batch) --
    explosive = [cs for cs in active
                 if _effective_pattern_count(cs) > CANDIDATE_THRESHOLD and not cs.needs_direct_answer]
    l3_count = 0
    explosive_with_steps = [cs for cs in explosive if cs.steps]
    if explosive_with_steps:
        l3_prompts = []
        for cs in explosive_with_steps:
            prompt, system = _build_prune_all_prompt(cs.question, cs.steps, cs.step_candidates)
            l3_prompts.append([
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ])
        l3_responses = await batch_call_llm(session, l3_prompts, max_tokens=1000)

        for cs, raw in zip(explosive_with_steps, l3_responses):
            if not raw:
                continue
            try:
                prune_result = _parse_prune_result(raw, cs.step_candidates, cs.steps)
                new_step_relations = []
                for step in cs.steps:
                    sn = step["step"]
                    ranked = prune_result.get(sn, [])
                    new_step_relations.append(set(ranked[:2]))
            except Exception:
                continue

            kq_rels = [(set(rs) if rs else set()) for rs in new_step_relations]
            try:
                paths_new, max_depth_new, max_cov_new = k_queue_traverse(
                    cs.anchor_idx, kq_rels,
                    cs.h_ids, cs.r_ids, cs.t_ids, cs.ents,
                    beam_width=80, max_hops_per_step=2,
                    relation_list=cs.rels)
            except Exception:
                continue
            paths_new = prefer_breakpoint_hit_paths(
                paths_new, cs.breakpoints, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents)

            unique_n = _rebuild_candidates_from_paths(
                paths_new, cs.anchor_idx, cs.ents, cs.h_ids, cs.r_ids, cs.t_ids)

            if len(unique_n) <= CANDIDATE_THRESHOLD:
                _apply_new_paths(cs, paths_new, new_step_relations, unique_n)
                l3_count += 1

    # -- Level 4: Anchor swap for persistent explosion (batch) --
    still_explosive = [cs for cs in active
                       if _effective_pattern_count(cs) > CANDIDATE_THRESHOLD and not cs.needs_direct_answer]
    l4_count = 0
    # Phase 1: Swap anchors (CPU work)
    swapped = []
    for cs in still_explosive:
        current_anchor = cs.anchor_name
        new_anchor = None
        new_anchor_idx = None
        for ent_name, gte_score in cs.ner_top_ents:
            if ent_name == current_anchor:
                continue
            ner_name_map = defaultdict(list)
            for idx, name in enumerate(cs.ents):
                if name == ent_name:
                    ner_name_map[name].append(idx)
            if ner_name_map.get(ent_name):
                new_anchor = ent_name
                new_anchor_idx = ner_name_map[ent_name][0]
                break
        if new_anchor is None:
            continue
        cs.anchor_name = new_anchor
        cs.anchor_idx = new_anchor_idx
        swapped.append(cs)

    # Phase 2: Batch prune for swapped anchors
    if swapped:
        l4_prompts = []
        for cs in swapped:
            prompt, system = _build_prune_all_prompt(cs.question, cs.steps, cs.step_candidates)
            l4_prompts.append([
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ])
        l4_responses = await batch_call_llm(session, l4_prompts, max_tokens=1000)

        for cs, raw in zip(swapped, l4_responses):
            if not raw:
                continue
            try:
                prune_result = _parse_prune_result(raw, cs.step_candidates, cs.steps)
                new_step_relations = []
                for step in cs.steps:
                    sn = step["step"]
                    ranked = prune_result.get(sn, [])
                    new_step_relations.append(set(ranked[:2]))
            except Exception:
                continue

            kq_rels = [(set(rs) if rs else set()) for rs in new_step_relations]
            try:
                paths_new, max_depth_new, max_cov_new = k_queue_traverse(
                    cs.anchor_idx, kq_rels,
                    cs.h_ids, cs.r_ids, cs.t_ids, cs.ents,
                    beam_width=80, max_hops_per_step=2,
                    relation_list=cs.rels)
            except Exception:
                continue
            paths_new = prefer_breakpoint_hit_paths(
                paths_new, cs.breakpoints, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents)

            unique_n = _rebuild_candidates_from_paths(
                paths_new, cs.anchor_idx, cs.ents, cs.h_ids, cs.r_ids, cs.t_ids)

            if len(unique_n) < len(cs.answer_candidates):
                _apply_new_paths(cs, paths_new, new_step_relations, unique_n)
                l4_count += 1

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["retries"] = dt / len(active)
    hits_after = sum(1 for cs in active if cs.gt_hit)
    print(f"  Stage 6 (Safety net): {dt:.2f}s | {direct_count} direct | L3_reprune={l3_count} L4_anchor_swap={l4_count} | GT={hits_after}/{len(active)}")
