"""Stage 5: Graph traversal — hybrid frontier-first + relation-prior-expand fallback.

Traverses the KG from the anchor entity using k_queue_traverse as primary
and relation_prior_expand as fallback for weak coverage cases.  Handles
candidate explosion via progressive relation/layer filtering.
"""
from __future__ import annotations

import asyncio
import time
from typing import List

from kgqa.core.case_state import CaseState
from kgqa.core.config import CANDIDATE_THRESHOLD
from kgqa.core.utils import normalize, candidate_hit, strict_candidate_hit, compute_match_stats
from kgqa.traversal.cvt import is_cvt_like, expand_through_cvt
from kgqa.traversal.k_queue import k_queue_traverse
from kgqa.traversal.frontier import relation_prior_expand
from kgqa.traversal.logical_paths import build_mode_level_logical_paths
from kgqa.traversal.path_utils import (
    prefer_breakpoint_hit_paths,
)


# ---------------------------------------------------------------------------
# Helpers local to Stage 5
# ---------------------------------------------------------------------------

def _collect_hr_frontier(anchor_idx, step_relations, h_ids, r_ids, t_ids,
                         path_nodes=None, target_nodes=None, paths=None):
    """Collect HR frontier from path-level (h+r) triples.

    For each hop (h, r, t) in existing paths, collect ALL t' from KG where
    (h, r, t') exists. This supplements path entities with siblings missed
    by relation_prior_expand's beam_width limits.
    """
    if not paths:
        return [], set()

    # Collect unique (h, r) pairs from all path hops
    hr_pairs = set()
    for path in paths:
        nodes = path.get("nodes", [])
        relations = path.get("relations", [])
        for i in range(min(len(relations), len(nodes) - 1)):
            hr_pairs.add((nodes[i], relations[i]))

    if not hr_pairs:
        return [], set()

    all_nodes = set()
    for i in range(len(h_ids)):
        if (h_ids[i], r_ids[i]) in hr_pairs and t_ids[i] != anchor_idx:
            all_nodes.add(t_ids[i])

    return [], all_nodes


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


def evaluate_step_relations(anchor_idx, paths, step_relations, ents):
    """Count unique non-anchor candidates per relation per step from traversal paths.

    Returns: dict {step_idx: {rel_idx: count_of_unique_candidates}}
    """
    rel_to_steps = {}
    for si, rs in enumerate(step_relations):
        for r in (rs if isinstance(rs, (set, list)) else set()):
            rel_to_steps.setdefault(r, set()).add(si)

    step_rel_targets = {}
    for path in paths:
        nodes = path.get("nodes", [])
        relations = path.get("relations", [])
        for hop_i, rel_idx in enumerate(relations):
            if hop_i + 1 < len(nodes):
                target_idx = nodes[hop_i + 1]
                if target_idx == anchor_idx:
                    continue
                steps_for_rel = rel_to_steps.get(rel_idx, set())
                for step_idx in steps_for_rel:
                    step_rel_targets.setdefault(step_idx, {}).setdefault(rel_idx, set()).add(target_idx)

    return {s: {r: len(t) for r, t in rels.items()} for s, rels in step_rel_targets.items()}


# ---------------------------------------------------------------------------
# Constraint step merging
# ---------------------------------------------------------------------------

def _merge_constraint_steps(cs: CaseState):
    """After pruning, merge all constraint steps into a single step k+1.

    Constraints are same-level filters, not sequential hops.
    Each constraint gets its own GTE retrieval and pruning (for precision),
    but their relations are pooled into one step before traversal.
    """
    if not cs.steps or not cs.step_relations:
        return

    fact_indices = []
    constraint_indices = []
    for i, step in enumerate(cs.steps):
        if 'constraint' in step:
            constraint_indices.append(i)
        else:
            fact_indices.append(i)

    if not constraint_indices:
        return

    new_steps = [cs.steps[i] for i in fact_indices]
    new_step_relations = [cs.step_relations[i] for i in fact_indices]

    # Merge all constraint relations into one step
    merged_rels = []
    merged_questions = []
    for i in constraint_indices:
        merged_rels.extend(cs.step_relations[i])
        merged_questions.append(cs.steps[i].get('question', ''))

    merged_step = dict(cs.steps[constraint_indices[0]])
    merged_step['question'] = ' | '.join(merged_questions)
    merged_step.pop('constraint', None)
    new_steps.append(merged_step)
    new_step_relations.append(merged_rels)

    # Renumber steps sequentially
    for i, step in enumerate(new_steps):
        step['step'] = i + 1

    cs.steps = new_steps
    cs.step_relations = new_step_relations


# ---------------------------------------------------------------------------
# Stage 5 main
# ---------------------------------------------------------------------------

async def stage_5_graph_traversal(cases: List[CaseState]):
    """Hybrid graph traversal: frontier-first, relation_prior_expand fallback for weak cases."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]

    def _traverse_one(cs: CaseState):
        if cs.anchor_idx is None:
            cs.needs_direct_answer = True
            return

        # Merge constraint steps into one step before traversal
        _merge_constraint_steps(cs)

        bp_set = set(cs.breakpoints.values()) - {cs.anchor_idx, None} if cs.breakpoints else set()
        n_steps = len(cs.steps)
        # explicit_targets from resolved endpoints (same as e2e)
        explicit_targets = list(bp_set) if bp_set else None

        # -- Primary: mode-level logical path traversal (handles step skip, endpoint bridge) --
        kq_step_rels = [(set(rs) if rs else set()) for rs in cs.step_relations]

        # Primary: mode-level logical path traversal
        logical_paths = build_mode_level_logical_paths(
            cs.anchor_idx, kq_step_rels, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents, cs.rels,
            bp_set, beam_width=80, max_hops_per_step=2, relation_list=cs.rels)

        if logical_paths:
            # Extract witness raw paths from logical paths
            seen_witness = set()
            paths = []
            for lp in logical_paths:
                witness = lp.get("best_raw_path")
                if not witness:
                    continue
                sig = (tuple(witness.get("nodes", [])), tuple(witness.get("relations", [])))
                if sig in seen_witness:
                    continue
                seen_witness.add(sig)
                paths.append(witness)
            max_depth = max((p.get("depth", 0) for p in paths), default=0)
            max_cov = max((len(p.get('covered_steps', frozenset())) for p in paths), default=0)
            cs.logical_paths = logical_paths
        else:
            # Fallback: k_queue_traverse
            paths, max_depth, max_cov = k_queue_traverse(
                cs.anchor_idx, kq_step_rels, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents,
                beam_width=80, max_hops_per_step=2, relation_list=cs.rels)

        # RPE fallback for broader coverage
        if max_cov < n_steps or n_steps <= 1:
            rpe_paths, rpe_depth, rpe_cov = relation_prior_expand(
                cs.anchor_idx, [set(rs) for rs in cs.step_relations],
                cs.h_ids, cs.r_ids, cs.t_ids, cs.ents,
                explicit_targets=explicit_targets,
                prefix_nodes=getattr(cs, "prefix_nodes", None))
            if rpe_cov > max_cov:
                paths, max_depth, max_cov = rpe_paths, rpe_depth, rpe_cov
            elif rpe_paths:
                existing_sigs = {(tuple(p["relations"][:3]), p["nodes"][-1]) for p in paths}
                for rp in rpe_paths:
                    sig = (tuple(rp["relations"][:3]), rp["nodes"][-1])
                    if sig not in existing_sigs:
                        paths.append(rp)
                        existing_sigs.add(sig)
        # Prefer paths hitting breakpoint endpoints
        paths = prefer_breakpoint_hit_paths(
            paths, cs.breakpoints, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents
        )
        if paths:
            max_depth = max(p.get("depth", 0) for p in paths)
            max_cov = max(len(p.get("covered_steps", frozenset())) for p in paths)
        cs.paths = paths
        cs.max_depth = max_depth
        cs.max_cov = max_cov

        # Collect subgraph nodes
        cs.all_subgraph_nodes = {cs.anchor_idx}
        for path in cs.paths:
            cs.all_subgraph_nodes.update(path["nodes"])

        # HR frontier: path-level (h+r) forward + (r+t) reverse triples
        expanded_rels = [set(rs) for rs in cs.step_relations]
        hr_triples, hr_nodes = _collect_hr_frontier(
            cs.anchor_idx, expanded_rels, cs.h_ids, cs.r_ids, cs.t_ids,
            paths=cs.paths)
        cs.all_subgraph_nodes |= hr_nodes

        # Answer candidates from last step's BFS walk (+ CVT expansion)
        answer_candidates = _last_step_candidates(
            cs.paths, cs.anchor_idx, cs.step_relations, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents)

        seen = set()
        unique = []
        for c in answer_candidates:
            nc = normalize(c)
            if len(nc) < 2:
                continue
            # REMOVED: if not c.isascii() - was too aggressive for non-English entities
            if nc not in seen:
                seen.add(nc)
                unique.append(c)

        # Merge logical_path candidates
        logical_paths = getattr(cs, 'logical_paths', [])
        for lp in logical_paths:
            for c in lp.get("candidates", []):
                nc = normalize(c)
                if len(nc) < 2 or nc in seen:
                    continue
                seen.add(nc)
                unique.append(c)

        cs.answer_candidates = unique

        # -- Level 1-2: Progressive relation/layer filtering for candidate explosion --
        lp_count = len(getattr(cs, 'logical_paths', [])) if getattr(cs, 'logical_paths', None) else _effective_pattern_count(cs)
        if lp_count > CANDIDATE_THRESHOLD and cs.paths:
            step_rel_quality = evaluate_step_relations(
                cs.anchor_idx, cs.paths, cs.step_relations, cs.ents)

            filtered_step_relations = []
            any_removed = False
            layers_skipped = 0
            for li, rs in enumerate(cs.step_relations):
                rs_set = set(rs) if not isinstance(rs, set) else rs
                if not rs_set:
                    filtered_step_relations.append(rs_set)
                    continue
                rel_counts = step_rel_quality.get(li, {})
                clean = {r for r in rs_set if rel_counts.get(r, 0) <= CANDIDATE_THRESHOLD}
                noisy = {r for r in rs_set if rel_counts.get(r, 0) > CANDIDATE_THRESHOLD}

                if clean:
                    filtered_step_relations.append(clean)
                    if noisy:
                        any_removed = True
                else:
                    # Level 2: all relations in this layer are noisy → skip
                    filtered_step_relations.append(set())
                    any_removed = True
                    layers_skipped += 1

            if any_removed:
                bp_set_filt = set(cs.breakpoints.values()) - {cs.anchor_idx, None}
                paths_new, max_depth_new, max_cov_new = relation_prior_expand(
                    cs.anchor_idx, filtered_step_relations,
                    cs.h_ids, cs.r_ids, cs.t_ids, cs.ents,
                    explicit_targets=list(bp_set_filt) if bp_set_filt else None,
                    max_hops=len(filtered_step_relations))
                paths_new = prefer_breakpoint_hit_paths(
                    paths_new, cs.breakpoints, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents)

                # Rebuild candidates from filtered paths (last 2 hops only)
                last2_new = set()
                for p in paths_new:
                    nodes = p.get("nodes", [])
                    for n in nodes[-2:]:
                        if n != cs.anchor_idx:
                            last2_new.add(n)

                new_candidates = []
                for node_idx in sorted(last2_new):
                    if node_idx == cs.anchor_idx:
                        continue
                    name = cs.ents[node_idx] if 0 <= node_idx < len(cs.ents) else ""
                    if is_cvt_like(name):
                        for cvt_idx, _ in expand_through_cvt(node_idx, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents):
                            if cvt_idx != cs.anchor_idx and 0 <= cvt_idx < len(cs.ents) and not is_cvt_like(cs.ents[cvt_idx]):
                                new_candidates.append(cs.ents[cvt_idx])
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

                old_max_cov = cs.max_cov
                old_path_count = len(cs.paths)
                new_max_cov = max((len(p.get("covered_steps", frozenset())) for p in paths_new), default=0)
                accepted_filter = (
                    bool(paths_new)
                    and new_max_cov >= old_max_cov
                    and len(unique_n) < len(cs.answer_candidates)
                )
                filter_debug = {
                    "triggered": True,
                    "accepted": accepted_filter,
                    "before_step_relations": [list(rs) for rs in cs.step_relations],
                    "after_step_relations": [list(rs) for rs in filtered_step_relations],
                    "before_candidates": len(cs.answer_candidates),
                    "after_candidates": len(unique_n),
                    "before_max_cov": old_max_cov,
                    "after_max_cov": new_max_cov,
                    "before_paths": old_path_count,
                    "after_paths": len(paths_new),
                    "relation_candidate_counts": {
                        str(li): {str(r): c for r, c in counts.items()}
                        for li, counts in step_rel_quality.items()
                    },
                }
                if not isinstance(cs.prune_debug, dict):
                    cs.prune_debug = {}
                cs.prune_debug["level_filter"] = filter_debug

                if accepted_filter:
                    cs.paths = paths_new
                    cs.step_relations = filtered_step_relations
                    cs.answer_candidates = unique_n
                    cs.all_subgraph_nodes = {cs.anchor_idx}
                    for p in paths_new:
                        cs.all_subgraph_nodes.update(p.get("nodes", []))
                    if paths_new:
                        cs.max_depth = max(p.get("depth", 0) for p in paths_new)
                        cs.max_cov = max(len(p.get("covered_steps", frozenset())) for p in paths_new)
                    cs.layers_skipped_by_filter = layers_skipped

        # GT recall is a traversal-end annotation: only entities produced by
        # the last step should count, not earlier bridge nodes from raw paths.
        cs.path_candidates = []
        seen_gt_candidates = set()
        for c in cs.answer_candidates:
            nc = normalize(c)
            if len(nc) < 2 or nc in seen_gt_candidates:
                continue
            seen_gt_candidates.add(nc)
            cs.path_candidates.append(c)
        cs.gt_hit = candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
        cs.gt_hit_strict = strict_candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
        gt_stats = compute_match_stats(cs.path_candidates, cs.gt_answers)
        cs.gt_f1 = gt_stats['f1']

        # Flag for direct answer if no paths found
        if not cs.paths:
            cs.needs_direct_answer = True

    # Run graph traversal (CPU-bound) concurrently
    await asyncio.gather(*[asyncio.to_thread(_traverse_one, cs) for cs in active])

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["graph_traverse"] = dt / len(active)
    hits = sum(1 for cs in active if cs.gt_hit)
    strict_hits = sum(1 for cs in active if cs.gt_hit_strict)
    direct = sum(1 for cs in active if cs.needs_direct_answer)
    print(f"  Stage 5 (Graph traverse): {dt:.2f}s | GT={hits}/{len(active)} | strict={strict_hits}/{len(active)} | direct={direct}")


def _last_step_candidates(paths, anchor_idx, step_relations, h_ids, r_ids, t_ids, ents):
    """Return raw candidate names from the last step's BFS walk (all nodes, not just terminal)."""
    last_si = -1
    for i in range(len(step_relations) - 1, -1, -1):
        if step_relations[i]:
            last_si = i
            break
    if last_si < 0:
        return []
    frontier = {anchor_idx}
    for p in paths:
        if last_si not in p.get("covered_steps", frozenset()):
            ns = p.get("nodes", [])
            if ns:
                frontier.add(ns[-1])
    if len(frontier) <= 1:
        for si in range(last_si):
            srels = step_relations[si]
            if not srels:
                continue
            for j in range(len(h_ids)):
                if r_ids[j] in srels:
                    frontier.add(h_ids[j])
                    frontier.add(t_ids[j])
    last_nodes = set()
    for p in paths:
        if last_si not in p.get("covered_steps", frozenset()):
            continue
        ns = p.get("nodes", [])
        if not ns:
            continue
        fp = -1
        for fi in range(len(ns)):
            if ns[fi] in frontier:
                fp = fi
        for ni in range(fp + 1, len(ns)):
            last_nodes.add(ns[ni])
    out = []
    for nidx in sorted(last_nodes):
        nm = ents[nidx] if 0 <= nidx < len(ents) else ""
        if is_cvt_like(nm):
            for ci, _ in expand_through_cvt(nidx, h_ids, r_ids, t_ids, ents):
                if ci != anchor_idx and 0 <= ci < len(ents) and not is_cvt_like(ents[ci]):
                    out.append(ents[ci])
        elif nidx != anchor_idx and nm:
            out.append(nm)
    return out
