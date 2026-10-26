"""K-queue BFS traversal with bitmask loop detection.

Single BFS from anchor, K watchers tracking target relations simultaneously.
Each watcher independently records when its step's relations are found.
"""
from __future__ import annotations

import os
from typing import List

from kgqa.traversal.logical_paths import DIRECTED_TRAVERSAL

_KQ_ADJ_CACHE: dict = {}   # (ids, len, directed, noisy) -> adjacency (walk-perf)


def _is_noisy_path_relation(rel_name):
    rel_name = rel_name or ""
    noisy_prefixes = (
        "type.",
        "common.",
        "freebase.",
        "kg.",
        "user.",
        "base.ontologies.",
    )
    noisy_shorts = {
        "type",
        "types",
        "instance",
        "instances",
        "notable_types",
        "topic_equivalent_webpage",
        "webpage",
        "mid",
        "guid",
        "key",
        "keys",
        "permission",
        "is_reviewed",
    }
    return rel_name.startswith(noisy_prefixes) or rel_name.rsplit(".", 1)[-1] in noisy_shorts


def _check_order(path_rels, step_relations):
    """Check which step relation sets appear in the path in correct order.

    Returns set of step indices matched in order.
    """
    if not path_rels:
        return set()
    matched = set()
    search_start = 0
    for step_idx, rel_set in enumerate(step_relations):
        if not rel_set:
            continue
        for pos in range(search_start, len(path_rels)):
            if path_rels[pos] in rel_set:
                matched.add(step_idx)
                search_start = pos + 1
                break
    return matched


def _coverage_tier(matched_steps):
    """Bitmask-based tier: each matched step contributes 2^position."""
    return sum(1 << s for s in matched_steps)


def k_queue_traverse(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                     beam_width=80, max_hops_per_step=2, relation_list=None):
    """Single BFS from anchor, K independent queues each tracking their own step_relations.

    Instead of processing steps sequentially, one BFS pass tracks ALL target
    relations simultaneously via rel_to_step mapping. After each hop that matches
    a target relation, check_order computes which steps are covered in order.

    Returns (paths, max_depth, max_coverage_tier).
    """
    n_steps = len(step_relations)
    if n_steps == 0:
        return [], 0, 0
    if not any(step_relations):
        return [], 0, 0

    # Build noise relation blacklist
    noisy_rel_ids = set()
    if relation_list is not None:
        for ri, rname in enumerate(relation_list):
            if _is_noisy_path_relation(rname):
                noisy_rel_ids.add(ri)

    # Build adjacency (directed if KGQA_DIRECTED_TRAVERSAL=1, else undirected).
    # Directed mode respects Freebase's named-direction relations (contains vs
    # containedby) instead of treating them as bidirectional.
    # MEMOIZED per case (walk-perf, 2026-09-07): same-arrays rebuilds are pure
    # waste — see logical_paths._build_adj; downstream is read-only iteration.
    _akey = (id(h_ids), id(r_ids), id(t_ids), len(h_ids),
             DIRECTED_TRAVERSAL,
             frozenset(noisy_rel_ids) if noisy_rel_ids else None)
    _hit = _KQ_ADJ_CACHE.get(_akey)
    if _hit is not None:
        adj = _hit[0]
    else:
        adj = None
    if adj is None:
        adj = {}
        for i in range(len(h_ids)):
            h, r, t = h_ids[i], r_ids[i], t_ids[i]
            if r in noisy_rel_ids:
                continue
            adj.setdefault(h, []).append((t, r))
            if not DIRECTED_TRAVERSAL:
                adj.setdefault(t, []).append((h, r))
        if len(_KQ_ADJ_CACHE) > 16:
            _KQ_ADJ_CACHE.clear()
        # pin the source arrays: id() keys stay valid while the entry lives
        # (audit 2026-09-07 — stale-id collision impossible)
        _KQ_ADJ_CACHE[_akey] = (adj, (h_ids, r_ids, t_ids))

    # Build rel_to_step mapping: each relation → set of step indices it belongs to
    rel_to_step = {}
    all_target_rels = set()
    for step_idx, rel_set in enumerate(step_relations):
        for rel in rel_set:
            rel_to_step.setdefault(rel, set()).add(step_idx)
            all_target_rels.add(rel)

    max_hops = max_hops_per_step * n_steps
    max_total_paths = 5000

    # -- Single BFS from anchor --
    # State: flat arrays with frozenset loop detection.
    # (Previously used bigint bitmask `1 << node_idx`, which creates huge bigints
    #  on large subgraphs — O(N/64) per bitwise op. frozenset is faster for
    #  sparse visited sets and avoids the latent `1 << negative` crash.)
    flat_nodes = [anchor_idx]
    flat_parent = [-1]
    flat_rel = [-1]
    flat_visited = [frozenset({anchor_idx})]
    flat_rels_list = [[]]  # accumulated path relations for check_order
    flat_depth = [0]

    frontier = [0]
    completed = []  # paths that hit at least one target relation

    for hop in range(max_hops):
        if not frontier:
            break
        new_frontier = []
        for eidx in frontier:
            cur = flat_nodes[eidx]
            visited = flat_visited[eidx]
            path_rels = flat_rels_list[eidx]
            cur_depth = flat_depth[eidx]

            for nb, rel in adj.get(cur, []):
                if nb in visited:
                    continue
                new_visited = visited | {nb}
                new_rels = path_rels + [rel]
                new_depth = cur_depth + 1

                nidx = len(flat_nodes)
                flat_nodes.append(nb)
                flat_parent.append(eidx)
                flat_rel.append(rel)
                flat_visited.append(new_visited)
                flat_rels_list.append(new_rels)
                flat_depth.append(new_depth)

                # Check if this hop uses a target relation
                if rel in all_target_rels:
                    # Reconstruct path nodes
                    r_nodes = [nb]
                    pi = eidx
                    while pi >= 0:
                        r_nodes.append(flat_nodes[pi])
                        pi = flat_parent[pi]
                    r_nodes.reverse()

                    # Compute covered steps via check_order
                    covered = _check_order(new_rels, step_relations)
                    tier = _coverage_tier(covered)

                    completed.append({
                        "nodes": r_nodes,
                        "relations": new_rels,
                        "covered_steps": frozenset(covered),
                        "covered_rels": frozenset(r for r in new_rels if r in all_target_rels),
                        "depth": new_depth,
                        "coverage_tier": tier,
                    })

                # Continue expanding regardless — other watchers may need further hops
                new_frontier.append(nidx)

        # Beam prune: keep most promising entries by coverage potential
        if len(new_frontier) > beam_width * 2:
            # Score by how many target relations found so far in path
            scored = []
            for nidx in new_frontier:
                path_rels = flat_rels_list[nidx]
                hit_count = sum(1 for r in path_rels if r in all_target_rels)
                scored.append((hit_count, nidx))
            scored.sort(key=lambda x: -x[0])
            new_frontier = [s[1] for s in scored[:beam_width * 2]]

        frontier = new_frontier

        if len(completed) > max_total_paths:
            break

    if not completed:
        return [], 0, 0

    # Dedup by (nodes, relations) signature
    seen_sigs = set()
    deduped = []
    for p in completed:
        sig = (tuple(p["nodes"]), tuple(p["relations"]))
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            deduped.append(p)

    all_steps = frozenset(range(n_steps))

    def _step_rank(path):
        covered = path.get("covered_steps", frozenset())
        return (
            covered == all_steps,
            len(covered),
            max(covered) if covered else -1,
            -path["depth"],
        )

    # Sort by matched decomposition steps first. Hop count is only a tie-breaker
    # because one logical step may span multiple graph hops.
    deduped.sort(key=_step_rank, reverse=True)

    if len(deduped) > max_total_paths:
        deduped = deduped[:max_total_paths]

    # Build output — include partial coverage paths too (not just full-coverage)
    output_paths = [p for p in deduped if p["depth"] > 0 and p.get("covered_rels")]

    if not output_paths:
        return [], 0, 0

    output_paths.sort(key=_step_rank, reverse=True)

    for p in output_paths:
        p["matched_relations"] = frozenset(p.get("covered_rels", frozenset()))

    max_depth = max(p["depth"] for p in output_paths)
    max_cov = max(len(p.get("covered_steps", frozenset())) for p in output_paths)

    return output_paths, max_depth, max_cov
