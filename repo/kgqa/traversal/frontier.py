"""Frontier-based graph expansion strategies for multi-hop KGQA traversal.

Includes chain expansion, bidirectional search, relation-prior expansion,
layer-wise frontier expansion, and layer diagnostics.
"""
from __future__ import annotations

import re
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from kgqa.traversal.cvt import is_cvt_like, expand_through_cvt, expand_node
from kgqa.traversal.logical_paths import DIRECTED_TRAVERSAL

_RPE_ADJ_CACHE: Dict[tuple, dict] = {}   # (ids, len, directed) -> tuple-adjacency (walk-perf)


# ---------------------------------------------------------------------------
# Causal tier scoring (shared by multiple expansion functions)
# ---------------------------------------------------------------------------

def score_causal_tier(hit_set, bridge_length=0):
    """Score path by decomposition layer coverage using lexicographic tuple.

    Returns (hit_count, max_layer_hit, -bridge_length) for comparison.
    Higher = better. Layer dedup: each layer counted once regardless of
    how many of its relations match.

    3-step: {R1,R2,R3} > {R1,R3}={R2,R3} > {R1,R2} > {R3} > {R2} > {R1}
      (3,2,*)       (2,2,*)  (2,2,*)    (2,1,*)   (1,2,*)  (1,1,*)  (1,0,*)
    """
    if not hit_set:
        return (0, -1, 0)
    return (len(hit_set), max(hit_set), -bridge_length)


# ---------------------------------------------------------------------------
# Chain expansion (v1): ordered step-by-step with forward validation
# ---------------------------------------------------------------------------

def chain_expand(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list):
    """Ordered chain expansion with forward validation: step 1 -> step 2 -> ...
    After expanding step K, validate that resulting nodes have step K+1 edges.
    Prunes structurally incomplete paths — if a tail node can't continue the
    chain, the triple is removed. Last step has no constraint (all results kept).
    Returns ALL paths, not just deepest.
    """
    paths = [{"nodes": [anchor_idx], "relations": [], "depth": 0}]

    for step_idx, rel_indices in enumerate(step_relations):
        if not rel_indices:
            continue
        # Look ahead: find next non-empty step for forward validation
        next_rel_indices = set()
        for nxt in range(step_idx + 1, len(step_relations)):
            if step_relations[nxt]:
                next_rel_indices = step_relations[nxt]
                break

        new_paths = []
        for path in paths:
            current = path["nodes"][-1]
            fwd = expand_node(current, rel_indices, h_ids, r_ids, t_ids)
            rev = expand_node(current, rel_indices, h_ids, r_ids, t_ids, reverse=True) if not DIRECTED_TRAVERSAL else []
            all_children = fwd + rev
            if all_children:
                seen = set(path["nodes"])
                for child_idx, rel_idx in all_children:
                    if child_idx in seen:
                        continue
                    # Forward validation: skip child if it can't continue the chain
                    if next_rel_indices and not _has_step_edges(
                            child_idx, next_rel_indices, h_ids, r_ids, t_ids, entity_list):
                        continue
                    new_path = {"nodes": path["nodes"] + [child_idx], "relations": path["relations"] + [rel_idx], "depth": path["depth"] + 1}
                    child_name = entity_list[child_idx] if 0 <= child_idx < len(entity_list) else ""
                    if is_cvt_like(child_name):
                        cvt_children = expand_through_cvt(child_idx, h_ids, r_ids, t_ids, entity_list)
                        new_seen = set(new_path["nodes"])
                        for cvt_idx, cvt_rel in cvt_children:
                            if cvt_idx in new_seen:
                                continue
                            new_paths.append({"nodes": new_path["nodes"] + [cvt_idx], "relations": new_path["relations"] + [cvt_rel], "depth": new_path["depth"] + 1})
                    else:
                        new_paths.append(new_path)
            else:
                # Dead end: keep path as-is for this step
                new_paths.append(path)
        paths = new_paths if new_paths else paths

    if not paths:
        return [], 0
    max_depth = max(p["depth"] for p in paths)
    return paths, max_depth


def _has_step_edges(node_idx, rel_indices, h_ids, r_ids, t_ids, entity_list):
    """Check if node has edges matching rel_indices within 2 hops.
    Direct edges first, then through CVT intermediary:
      node -> [any rel] -> CVT -> [target rel]
    This handles cases like child_labor_percent where the relation
    bridges through a CVT node rather than connecting directly.
    """
    # Direct check
    fwd = expand_node(node_idx, rel_indices, h_ids, r_ids, t_ids)
    rev = expand_node(node_idx, rel_indices, h_ids, r_ids, t_ids, reverse=True) if not DIRECTED_TRAVERSAL else []
    if fwd or rev:
        return True
    # 2-hop: node -> CVT -> [target rel]
    seen = {node_idx}
    for i in range(len(h_ids)):
        if h_ids[i] == node_idx:
            neighbor = t_ids[i]
        elif t_ids[i] == node_idx:
            neighbor = h_ids[i]
        else:
            continue
        if neighbor in seen:
            continue
        neighbor_name = entity_list[neighbor] if 0 <= neighbor < len(entity_list) else ""
        if not is_cvt_like(neighbor_name):
            continue
        seen.add(neighbor)
        fwd2 = expand_node(neighbor, rel_indices, h_ids, r_ids, t_ids)
        rev2 = expand_node(neighbor, rel_indices, h_ids, r_ids, t_ids, reverse=True) if not DIRECTED_TRAVERSAL else []
        if fwd2 or rev2:
            return True
    return False


# ---------------------------------------------------------------------------
# Chain expansion (v2): relation-anchored multi-hop with beam search
# ---------------------------------------------------------------------------

def chain_expand_v2(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                    max_hops=4, beam_width=80, per_branch_width=5):
    """Relation-anchored multi-hop expansion: anchor on relations, not hop count.

    For step_relations = [R1_set, R2_set, ..., Rn_set]:
    1. MAIN: Search all paths within max_hops where the LAST hop's relation is in Rn_set.
    2. BACKTRACK: Check if earlier relations (R1..Rn-1) appear in order in the path.
    3. SCORE: Full match (all relations in order) > partial match > single relation.
    4. FALLBACK: If no Rn paths found, try Rn-1, then Rn-2, ..., down to R1.

    CVT nodes pass through without counting toward hop limit.

    Returns (paths, max_depth, max_coverage_tier) where paths are dicts with keys:
      nodes, relations, depth, covered_steps, matched_relations, coverage_tier
    """
    n_steps = len(step_relations)
    if n_steps == 0:
        return [], 0, 0

    # Build adjacency list (directed if KGQA_DIRECTED_TRAVERSAL=1, else undirected)
    adj = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        adj.setdefault(h, []).append((t, r))
        if not DIRECTED_TRAVERSAL:
            adj.setdefault(t, []).append((h, r))

    # Build rel_to_step mapping
    rel_to_step = {}
    all_target_rels = set()
    for step_idx, rel_set in enumerate(step_relations):
        for rel in rel_set:
            rel_to_step.setdefault(rel, set()).add(step_idx)
            all_target_rels.add(rel)

    # Coverage tier computation:
    # Given a path's matched steps, compute tier score.
    # For [R1, R2]: R1+R2=3, R2_only=2, R1_only=1
    # For [R1, R2, R3]: R1+R2+R3=7, R1+R3=5, R2+R3=6, R3_only=4, R1+R2=3, R2_only=2, R1_only=1
    # General: priority by last-step presence, then by earlier steps, ordered correctly.
    def compute_tier(matched_steps_set):
        """Higher is better. Must include the target step for main strategy."""
        if not matched_steps_set:
            return 0
        score = 0
        for s in matched_steps_set:
            score += (1 << s)
        return score

    def check_order(path_rels, target_rels_ordered):
        """Check which target relation sets appear in the path in correct order.

        Returns set of step indices that are matched in order.
        """
        if not path_rels:
            return set()

        matched = set()
        search_start = 0
        for step_idx, rel_set in enumerate(target_rels_ordered):
            if not rel_set:
                continue
            # Find first occurrence of any rel in rel_set at or after search_start
            for pos in range(search_start, len(path_rels)):
                if path_rels[pos] in rel_set:
                    matched.add(step_idx)
                    search_start = pos + 1
                    break
        return matched

    def _beam_prune(paths, limit):
        if len(paths) <= limit:
            return paths
        # Group by branch signature (last node + covered steps pattern) to maintain diversity
        groups = {}
        for p in paths:
            sig = (p["nodes"][-1], tuple(sorted(p.get("covered_steps", set()))))
            groups.setdefault(sig, []).append(p)
        result = []
        overflow = []
        for sig, group in groups.items():
            group.sort(key=lambda x: (x.get("coverage_tier", 0), -x.get("depth", 0)), reverse=True)
            take = min(len(group), per_branch_width)
            result.extend(group[:take])
            overflow.extend(group[take:])
        if len(result) < limit and overflow:
            overflow.sort(key=lambda x: (x.get("coverage_tier", 0), -x.get("depth", 0)), reverse=True)
            result.extend(overflow[:limit - len(result)])
        if len(result) > limit:
            result.sort(key=lambda x: (x.get("coverage_tier", 0), -x.get("depth", 0)), reverse=True)
            result = result[:limit]
        return result

    def _search_from_anchor(target_step_indices):
        """BFS/DFS hybrid from anchor, looking for paths ending with target step relations.

        target_step_indices: list of step indices whose relations are acceptable as the last hop.
        Returns list of path dicts.
        """
        # Collect target relations for this search
        target_rels = set()
        for si in target_step_indices:
            target_rels |= step_relations[si]

        if not target_rels:
            return []

        # BFS with beam
        # State: list of path dicts
        initial_path = {
            "nodes": [anchor_idx],
            "relations": [],
            "depth": 0,
            "real_hops": 0,  # hops excluding CVT passthrough
            "covered_steps": set(),
            "matched_relations": set(),
            "coverage_tier": 0,
            "last_hop_is_target": False,
        }
        active = [initial_path]
        completed = []  # paths that ended with a target relation

        for _ in range(max_hops * 3):  # enough iterations for CVT passthroughs
            if not active:
                break

            new_active = []
            for path in active:
                current = path["nodes"][-1]
                current_name = entity_list[current] if 0 <= current < len(entity_list) else ""
                is_at_cvt = is_cvt_like(current_name)

                all_neighbors = adj.get(current, [])
                if not all_neighbors:
                    continue

                seen = set(path["nodes"])

                # 3-tier edge filtering: pool relations first, non-pool strictly limited
                target_edges = []   # matches current search target (e.g. R_n)
                pool_edges = []     # in any step's relation set
                other_edges = []    # not in any step — noise
                for neighbor, rel in all_neighbors:
                    if neighbor in seen:
                        continue
                    if rel in target_rels:
                        target_edges.append((neighbor, rel))
                    elif rel in all_target_rels:
                        pool_edges.append((neighbor, rel))
                    else:
                        other_edges.append((neighbor, rel))

                # Priority: target > pool > limited fallback
                if target_edges or pool_edges:
                    edges = target_edges + pool_edges
                else:
                    edges = other_edges[:5]

                for neighbor, rel in edges:
                    new_nodes = path["nodes"] + [neighbor]
                    new_rels = path["relations"] + [rel]
                    new_real_hops = path["real_hops"]

                    # CVT passthrough: don't count toward hop limit
                    neighbor_name = entity_list[neighbor] if 0 <= neighbor < len(entity_list) else ""
                    if not is_cvt_like(current_name) or not is_at_cvt:
                        new_real_hops += 1

                    # Check hop limit
                    if new_real_hops > max_hops:
                        continue

                    # Check if this hop uses a target relation
                    rel_step_matches = rel_to_step.get(rel, set())
                    is_target_hop = rel in target_rels

                    new_covered = set(path["covered_steps"])
                    new_matched = set(path["matched_relations"])
                    if rel_step_matches:
                        new_covered |= rel_step_matches
                        new_matched.add(rel)

                    new_path = {
                        "nodes": new_nodes,
                        "relations": new_rels,
                        "depth": path["depth"] + 1,
                        "real_hops": new_real_hops,
                        "covered_steps": new_covered,
                        "matched_relations": new_matched,
                        "coverage_tier": 0,
                        "last_hop_is_target": is_target_hop,
                    }

                    if is_target_hop:
                        # This path ends with a target relation — compute coverage
                        ordered_match = check_order(new_rels, step_relations)
                        new_path["covered_steps"] = ordered_match
                        new_path["coverage_tier"] = compute_tier(ordered_match)
                        completed.append(new_path)
                        # Continue expanding from here too (might find longer matches)
                        if new_real_hops < max_hops:
                            new_active.append(new_path)
                    else:
                        # Not a target hop, keep searching
                        new_active.append(new_path)

            # Beam prune active paths
            active = _beam_prune(new_active, beam_width)

        # Cap completed paths: sort by coverage tier, keep top beam_width
        if len(completed) > beam_width:
            completed.sort(key=lambda p: (p.get("coverage_tier", 0), -p.get("depth", 0)), reverse=True)
            completed = completed[:beam_width]
        return completed

    # ---- Main strategy: search for paths ending with R_n ----
    result_paths = []

    # Try from last step backwards (fallback cascade)
    for target_depth in range(n_steps - 1, -1, -1):
        target_indices = [target_depth]
        found = _search_from_anchor(target_indices)

        if found:
            # Deduplicate by (nodes tuple, relations tuple)
            seen_keys = set()
            for p in result_paths:
                seen_keys.add((tuple(p["nodes"]), tuple(p["relations"])))

            for p in found:
                key = (tuple(p["nodes"]), tuple(p["relations"]))
                if key not in seen_keys:
                    seen_keys.add(key)
                    result_paths.append(p)

            # If we found paths for the deepest target step, we can still look for earlier ones
            # but only if the main target was the last step (not a fallback)
            if target_depth == n_steps - 1:
                # Also search for earlier steps as supplementary (these are lower tier)
                for supplementary_depth in range(n_steps - 2, -1, -1):
                    supp_found = _search_from_anchor([supplementary_depth])
                    for p in supp_found:
                        key = (tuple(p["nodes"]), tuple(p["relations"]))
                        if key not in seen_keys:
                            seen_keys.add(key)
                            result_paths.append(p)
                break  # Main strategy succeeded, don't fallback further
        # If target_depth < n_steps-1, this is a fallback — accept and stop

    if not result_paths:
        return [], 0, 0

    # Sort by coverage tier (desc), then depth (desc for deeper=more info), then fewer nodes
    result_paths.sort(key=lambda p: (p.get("coverage_tier", 0), -p.get("depth", 0)), reverse=True)

    # Final beam prune on total results
    if len(result_paths) > beam_width * 2:
        result_paths = _beam_prune(result_paths, beam_width * 2)

    max_depth = max(p.get("depth", 0) for p in result_paths)
    max_tier = max(p.get("coverage_tier", 0) for p in result_paths)

    return result_paths, max_depth, max_tier


# ---------------------------------------------------------------------------
# Coverage ranking helpers (used by bidirectional_expand)
# ---------------------------------------------------------------------------

def _coverage_rank(path):
    """Causal tier ranking: layer coverage count > max layer hit > shorter bridge."""
    covered = path.get("covered_steps", frozenset())
    depth = path.get("depth", 0)
    return score_causal_tier(covered, bridge_length=depth)


def _merge_paths(fwd_path, bwd_path, entity_list):
    fwd_nodes = fwd_path["nodes"]
    bwd_nodes = bwd_path["nodes"]
    meeting_node = fwd_nodes[-1]
    if meeting_node != bwd_nodes[-1]:
        return None
    if set(fwd_nodes[:-1]) & set(bwd_nodes[:-1]):
        return None
    return {
        "nodes": fwd_nodes + list(reversed(bwd_nodes[:-1])),
        "relations": fwd_path["relations"] + list(reversed(bwd_path["relations"])),
        "depth": fwd_path["depth"] + bwd_path["depth"],
        "covered_steps": frozenset(set(fwd_path.get("covered_steps", frozenset())) | set(bwd_path.get("covered_steps", frozenset()))),
        "matched_relations": frozenset(set(fwd_path.get("matched_relations", frozenset())) | set(bwd_path.get("matched_relations", frozenset()))),
    }


# ---------------------------------------------------------------------------
# Bidirectional BFS expansion
# ---------------------------------------------------------------------------

def bidirectional_expand(anchor_idx, target_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                         max_hops=5, beam_width=50, per_branch_width=6):
    """Relation-prior-guided bidirectional BFS with step-aware scoring.

    Forward from anchor: preferentially expands via step-aligned relations (step 0 at hop 0, etc.)
    Backward from endpoint: preferentially expands via reversed step relations (step N-1 at hop 0, etc.)
    Three-tier edge priority: guided (step-aligned) > in-pool (other steps) > fallback (all edges).
    Post-hoc ranking by step coverage count.
    """
    n_steps = len(step_relations)
    # Build rel_to_steps mapping and pooled relation set
    rel_to_steps: Dict[int, set] = {}
    relation_pool: set = set()
    for step_idx, rel_set in enumerate(step_relations):
        for rel in rel_set:
            rel_to_steps.setdefault(rel, set()).add(step_idx)
            relation_pool.add(rel)

    # Build adjacency (directed if KGQA_DIRECTED_TRAVERSAL=1, else undirected)
    adj: Dict[int, List[tuple]] = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        adj.setdefault(h, []).append((t, r))
        if not DIRECTED_TRAVERSAL:
            adj.setdefault(t, []).append((h, r))

    def _make_path(nodes, relations, depth, covered=frozenset(), matched=frozenset()):
        return {"nodes": nodes, "relations": relations, "depth": depth,
                "covered_steps": covered, "matched_relations": matched}

    def _extend_path(path, neighbor, rel):
        r_steps = rel_to_steps.get(rel, set())
        new_covered = frozenset(set(path["covered_steps"]) | r_steps)
        new_matched = frozenset(set(path["matched_relations"]) | ({rel} if r_steps else set()))
        return _make_path(
            path["nodes"] + [neighbor], path["relations"] + [rel], path["depth"] + 1,
            new_covered, new_matched)

    def _branch_sig(path):
        """Branch signature: (step_idx, first_matched_rel) per covered step + last node."""
        covered = path.get("covered_steps", frozenset())
        sig = []
        for step_idx in sorted(covered):
            for r in sorted(path.get("matched_relations", frozenset())):
                if step_idx in rel_to_steps.get(r, set()):
                    sig.append((step_idx, r))
                    break
        sig.append(path["nodes"][-1])
        return tuple(sig)

    def _beam_prune(paths, limit):
        if len(paths) <= limit:
            return paths
        groups: Dict[tuple, list] = {}
        for p in paths:
            sig = _branch_sig(p)
            groups.setdefault(sig, []).append(p)
        result = []
        remaining = []
        for sig, group in groups.items():
            group.sort(key=_coverage_rank, reverse=True)
            result.extend(group[:per_branch_width])
            remaining.extend(group[per_branch_width:])
        if len(result) < limit and remaining:
            remaining.sort(key=_coverage_rank, reverse=True)
            result.extend(remaining[:limit - len(result)])
        if len(result) > limit:
            result.sort(key=_coverage_rank, reverse=True)
            result = result[:limit]
        return result

    def _get_edges(node_idx, hop, is_forward):
        """Get edges for expansion with 3-tier priority: guided > in-pool > fallback."""
        all_edges = adj.get(node_idx, [])
        if not all_edges:
            return []
        # Determine expected step for this hop
        if is_forward:
            expected_step = hop % n_steps if n_steps > 0 else -1
        else:
            expected_step = (n_steps - 1 - hop % n_steps) if n_steps > 0 else -1

        expected_rels = step_relations[expected_step] if 0 <= expected_step < n_steps else set()

        guided = []    # Matches expected step relation
        in_pool = []   # In some step but not expected
        fallback = []  # Not in any step

        for neighbor, rel in all_edges:
            if rel in expected_rels:
                guided.append((neighbor, rel))
            elif rel in relation_pool:
                in_pool.append((neighbor, rel))
            else:
                fallback.append((neighbor, rel))

        # Priority: guided first, then in-pool (limited), then fallback (very limited)
        if guided:
            return guided + in_pool[:5]
        elif in_pool:
            return in_pool + fallback[:5]
        else:
            return fallback[:10]

    # Initialize frontiers
    init_fwd = _make_path([anchor_idx], [], 0)
    init_bwd = _make_path([target_idx], [], 0)
    fwd_frontier: Dict[int, List[Dict]] = {anchor_idx: [init_fwd]}
    bwd_frontier: Dict[int, List[Dict]] = {target_idx: [init_bwd]}
    fwd_path_map: Dict[int, List[Dict]] = {anchor_idx: [init_fwd]}
    bwd_path_map: Dict[int, List[Dict]] = {target_idx: [init_bwd]}
    fwd_visited: Dict[int, int] = {anchor_idx: 0}
    bwd_visited: Dict[int, int] = {target_idx: 0}
    meeting_paths = []

    for hop in range(max_hops):
        if len(fwd_frontier) <= len(bwd_frontier):
            # Forward expansion with guided edges
            new_frontier: Dict[int, List[Dict]] = {}
            for node_idx, paths in fwd_frontier.items():
                edges = _get_edges(node_idx, hop, is_forward=True)
                for neighbor, rel in edges:
                    if neighbor in fwd_visited and fwd_visited[neighbor] < hop + 1:
                        continue
                    fwd_visited.setdefault(neighbor, hop + 1)
                    for path in paths:
                        if neighbor in set(path["nodes"]):
                            continue
                        new_path = _extend_path(path, neighbor, rel)
                        if neighbor in bwd_visited:
                            for bwd_path in bwd_path_map.get(neighbor, []):
                                merged = _merge_paths(new_path, bwd_path, entity_list)
                                if merged:
                                    meeting_paths.append(merged)
                        new_frontier.setdefault(neighbor, []).append(new_path)
            for node in new_frontier:
                new_frontier[node] = _beam_prune(new_frontier[node], per_branch_width * 3)
            fwd_frontier = new_frontier
            for node, node_paths in new_frontier.items():
                fwd_path_map.setdefault(node, []).extend(node_paths)
                fwd_path_map[node] = _beam_prune(fwd_path_map[node], beam_width)
        else:
            # Backward expansion with guided edges
            new_frontier: Dict[int, List[Dict]] = {}
            for node_idx, paths in bwd_frontier.items():
                edges = _get_edges(node_idx, hop, is_forward=False)
                for neighbor, rel in edges:
                    if neighbor in bwd_visited and bwd_visited[neighbor] < hop + 1:
                        continue
                    bwd_visited.setdefault(neighbor, hop + 1)
                    for path in paths:
                        if neighbor in set(path["nodes"]):
                            continue
                        new_path = _extend_path(path, neighbor, rel)
                        if neighbor in fwd_visited:
                            for fwd_path in fwd_path_map.get(neighbor, []):
                                merged = _merge_paths(fwd_path, new_path, entity_list)
                                if merged:
                                    meeting_paths.append(merged)
                        new_frontier.setdefault(neighbor, []).append(new_path)
            for node in new_frontier:
                new_frontier[node] = _beam_prune(new_frontier[node], per_branch_width * 3)
            bwd_frontier = new_frontier
            for node, node_paths in new_frontier.items():
                bwd_path_map.setdefault(node, []).extend(node_paths)
                bwd_path_map[node] = _beam_prune(bwd_path_map[node], beam_width)

    if not meeting_paths:
        return [], 0, 0

    # Deduplicate and rank
    seen = set()
    unique = []
    for p in meeting_paths:
        key = (tuple(p["nodes"]), tuple(p["relations"]))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    unique.sort(key=_coverage_rank, reverse=True)
    max_cov = len(unique[0].get("covered_steps", frozenset())) if unique else 0
    max_depth = max(p.get("depth", 0) for p in unique)

    return unique, max_depth, max_cov


# ---------------------------------------------------------------------------
# Relation-prior forward expansion
# ---------------------------------------------------------------------------

def relation_prior_expand(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                          explicit_targets=None, max_hops=3, beam_width=80, per_branch_width=5,
                          prefix_nodes=None):
    """Forward layer-by-layer relation-prior expansion.

    New behavior:
    1. Start from current entity frontier (initially the anchor).
    2. For layer i, search all paths within max_hops whose LAST hop relation is in R_i.
    3. Use the endpoints of those matched paths as the start frontier for the next layer.
    4. If a layer has no hit, skip it and continue from the current frontier.
    5. If explicit endpoint targets exist, connect the final frontier to those targets
       via a shortest path search within max_hops.

    This removes the backward-target template and avoids the repeated-relation
    penetration issue seen in bidirectional matching such as r1 -> r1 collapse.

    Performance optimizations (v2):
    - Paths stored as tuples (nodes, rels, depth, real_hops, covered, matched)
      instead of dicts, avoiding dict creation overhead in the hot inner loop.
    - CVT status pre-computed once as a boolean list.
    - Adjacency neighbor lists stored as tuples for faster iteration.
    - BFS in _connect_to_targets uses collections.deque.
    - Reduced frozenset churn: only create new frozensets when coverage changes.
    - _prune_paths uses frozenset directly as hash key instead of sorted tuple.
    """
    n_steps = len(step_relations)
    if n_steps == 0:
        return [], 0, 0

    # -- Build adjacency (directed if KGQA_DIRECTED_TRAVERSAL=1, else undirected) --
    # MEMOIZED per case (walk-perf, 2026-09-07): the single-step agent walk
    # fires RPE on EVERY call (n_steps<=1 makes the fallback unconditional),
    # and each call rebuilt the identical adjacency — dense cases pay 26ms+
    # per rebuild at degree 900+. Downstream is read-only (adj.get iteration).
    _rkey = (id(h_ids), id(r_ids), id(t_ids), len(h_ids), DIRECTED_TRAVERSAL)
    _hit = _RPE_ADJ_CACHE.get(_rkey)
    adj = _hit[0] if _hit is not None else None
    if adj is None:
        adj = {}
        for i in range(len(h_ids)):
            h, r, t = h_ids[i], r_ids[i], t_ids[i]
            if h in adj:
                adj[h] = adj[h] + ((t, r),)
            else:
                adj[h] = ((t, r),)
            if not DIRECTED_TRAVERSAL:
                if t in adj:
                    adj[t] = adj[t] + ((h, r),)
                else:
                    adj[t] = ((h, r),)
        if len(_RPE_ADJ_CACHE) > 16:
            _RPE_ADJ_CACHE.clear()
        # pin the source arrays: id() keys stay valid while the entry lives
        # (audit 2026-09-07 — stale-id collision impossible)
        _RPE_ADJ_CACHE[_rkey] = (adj, (h_ids, r_ids, t_ids))
    adj_empty = ()

    # -- Pre-compute CVT mask (avoids re.match per hop) --
    is_cvt = [is_cvt_like(name) for name in entity_list]
    n_ents = len(entity_list)

    # -- Build reverse mapping --
    rel_to_step: Dict[int, set] = {}
    for si, rs in enumerate(step_relations):
        for r in rs:
            rel_to_step.setdefault(r, set()).add(si)
    all_layer_rels = set(rel_to_step.keys())

    # -- Helpers --
    def _real_hop_inc(curr_idx, next_idx):
        """Real hop increment for the hop-limit check.

        CVT nodes count as a hop (NOT passthrough). Rationale: a CVT-mediated
        path (anchor -> CVT -> leaf) is 2 structural hops, and downstream logic
        (compress_paths, candidate collection, pattern matching) all operate on
        actual path length (len(rels)/len(nodes)). Treating CVT as 0-hop here
        would let RPE explore longer *structural* paths than max_hops allows,
        creating an inconsistency with logical_paths (which counts CVT as a hop)
        and with compress_paths (which uses len(rels) for depth/tier).
        With max_hops_per_step=2, single CVT chains (2 structural hops) are
        reachable within one step. Only 3+-hop CVT chains would be truncated —
        and those are rare (8% of GT is 2-hop CVT, ~0% need 3+ CVT hops).
        """
        return 1

    def _coverage_rank_fast(path):
        """Path is tuple: (nodes, rels, depth, real_hops, covered_steps, matched_rels)."""
        covered = path[4]
        depth = path[2]
        if not covered:
            return (0, -1, 0)
        return (len(covered), max(covered), -depth)

    def _prune_paths(paths, limit):
        if len(paths) <= limit:
            return paths
        # Group by (endpoint, covered_steps) -- frozenset is directly hashable
        grouped: Dict[tuple, list] = {}
        for p in paths:
            sig = (p[0][-1], p[4])  # (nodes[-1], covered_steps)
            if sig in grouped:
                grouped[sig].append(p)
            else:
                grouped[sig] = [p]
        result = []
        overflow = []
        for group in grouped.values():
            group.sort(key=_coverage_rank_fast, reverse=True)
            result.extend(group[:per_branch_width])
            overflow.extend(group[per_branch_width:])
        if len(result) < limit and overflow:
            overflow.sort(key=_coverage_rank_fast, reverse=True)
            result.extend(overflow[: limit - len(result)])
        if len(result) > limit:
            result.sort(key=_coverage_rank_fast, reverse=True)
            result = result[:limit]
        return result

    def _search_terminal_relation_paths(start_paths, target_rels, layer_idx):
        """From start_paths, search local segments that terminate at the FIRST hit of target_rels.

        Rules:
        - bridge hops may not use any selected layer relation
        - once a target relation is hit, the segment ends immediately
        - same-layer relations cannot chain within one segment
        - PATTERN-PREFIX mask (user design 2026-09-10, 状态保持/不重复回去):
          with prefix_nodes set (a continuation walk from a pattern tail), a
          neighbor INSIDE the earlier walk's territory is never EXPANDED —
          only a target-relation edge INTO it is recorded (connection
          evidence); bridge hops into the prefix are skipped entirely.
        """
        if not target_rels:
            return []
        pfx = set(prefix_nodes) if prefix_nodes else None
        active = list(start_paths)
        matched = []
        seen_matched = set()
        layer_idx_frozen = frozenset({layer_idx})

        for _ in range(max_hops * 3):
            if not active:
                break
            new_active = []
            for path in active:
                nodes, rels, depth, real_hops, covered, matched_rels = path
                current = nodes[-1]
                neighbors = adj.get(current, adj_empty)
                for neighbor, rel in neighbors:
                    if neighbor in nodes:
                        continue
                    # -- Inverse-pair loop detection: same rel twice = trivial cycle --
                    if rels and rels[-1] == rel:
                        continue
                    inc = _real_hop_inc(current, neighbor)
                    new_real_hops = real_hops + inc
                    if new_real_hops > max_hops:
                        continue

                    is_target_rel = rel in target_rels
                    is_any_layer_rel = rel in all_layer_rels
                    in_prefix = pfx is not None and neighbor in pfx

                    if not is_target_rel and is_any_layer_rel:
                        continue
                    if in_prefix and not is_target_rel:
                        continue          # back into walked territory — dead end

                    new_nodes = nodes + (neighbor,)
                    new_rels = rels + (rel,)
                    new_depth = depth + 1

                    rel_steps = rel_to_step.get(rel)
                    if rel_steps:
                        new_covered = covered | rel_steps
                        new_matched = matched_rels | frozenset({rel})
                    else:
                        new_covered = covered
                        new_matched = matched_rels

                    new_path = (new_nodes, new_rels, new_depth, new_real_hops, new_covered, new_matched)

                    if is_target_rel:
                        final_covered = new_covered | layer_idx_frozen
                        final_path = (new_nodes, new_rels, new_depth, new_real_hops, final_covered, new_matched)
                        key = (new_nodes, new_rels)
                        if key not in seen_matched:
                            seen_matched.add(key)
                            matched.append(final_path)
                    else:
                        new_active.append(new_path)
            active = _prune_paths(new_active, beam_width)
        return _prune_paths(matched, beam_width)

    def _connect_to_targets(paths, targets):
        """Attach final frontier endpoints to explicit targets by shortest unconstrained path."""
        if not targets or not paths:
            return paths
        targets_set = set(targets) - {anchor_idx, None}
        if not targets_set:
            return paths

        connected = []
        seen = set()
        for base in paths:
            start = base[0][-1]  # nodes[-1]
            queue = deque([(start, (start,), (), 0)])
            local_seen = {(start, 0)}
            best = []
            while queue:
                node, nodes_seq, rel_seq, rhops = queue.popleft()
                if node in targets_set and node != start:
                    merged = (
                        base[0] + nodes_seq[1:],
                        base[1] + rel_seq,
                        base[2] + len(rel_seq),
                        base[3] + rhops,
                        base[4],
                        base[5],
                    )
                    best.append(merged)
                    continue
                for neighbor, rel in adj.get(node, adj_empty):
                    if neighbor in nodes_seq:
                        continue
                    inc = _real_hop_inc(node, neighbor)
                    new_hops = rhops + inc
                    if new_hops > max_hops:
                        continue
                    state_key = (neighbor, new_hops)
                    if state_key in local_seen:
                        continue
                    local_seen.add(state_key)
                    queue.append((neighbor, nodes_seq + (neighbor,), rel_seq + (rel,), new_hops))
            best.sort(key=_coverage_rank_fast, reverse=True)
            for p in best[:per_branch_width]:
                key = (p[0], p[1])
                if key not in seen:
                    seen.add(key)
                    connected.append(p)
        return _prune_paths(connected, beam_width) if connected else paths

    # -- Main expansion logic --
    # Internal path format: (nodes_tuple, rels_tuple, depth, real_hops, covered_steps, matched_rels)
    frontier_paths = [((anchor_idx,), (), 0, 0, frozenset(), frozenset())]
    all_result_paths = []
    matched_layer_indices = []
    nonempty_layers = [i for i, rs in enumerate(step_relations) if rs]

    for layer_idx, target_rels in enumerate(step_relations):
        if not target_rels:
            continue
        matched = _search_terminal_relation_paths(frontier_paths, target_rels, layer_idx)
        if not matched:
            continue
        frontier_paths = matched
        all_result_paths = matched
        matched_layer_indices.append(layer_idx)

    # Minimal repair: only repair the final non-empty layer if it was missed.
    if nonempty_layers and matched_layer_indices:
        last_nonempty_idx = nonempty_layers[-1]
        if last_nonempty_idx not in matched_layer_indices and frontier_paths:
            repaired = _search_terminal_relation_paths(frontier_paths, step_relations[last_nonempty_idx], last_nonempty_idx)
            if repaired:
                merged = list(all_result_paths or []) + repaired
                merged.sort(key=_coverage_rank_fast, reverse=True)
                all_result_paths = _prune_paths(merged, beam_width)

    if explicit_targets:
        all_result_paths = _connect_to_targets(all_result_paths or frontier_paths, explicit_targets)
    elif not all_result_paths:
        all_result_paths = frontier_paths

    if not all_result_paths:
        return [], 0, 0

    # Dedup + post-hoc cycle filter
    dedup = []
    seen = set()
    for p in all_result_paths:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            dedup.append(p)

    # Remove paths with cycles (repeated nodes)
    acyclic = [p for p in dedup if len(set(p[0])) == len(p[0])]
    if acyclic:
        dedup = acyclic

    dedup.sort(key=_coverage_rank_fast, reverse=True)
    dedup = _prune_paths(dedup, beam_width)

    # Convert internal tuple format back to dict format for API compatibility
    result_dicts = []
    for p in dedup:
        result_dicts.append({
            "nodes": list(p[0]),
            "relations": list(p[1]),
            "depth": p[2],
            "real_hops": p[3],
            "covered_steps": p[4],
            "matched_relations": p[5],
        })

    max_cov = max((len(p[4]) for p in dedup), default=0)
    max_depth = max((p[2] for p in dedup), default=0)
    return result_dicts, max_depth, max_cov


# ---------------------------------------------------------------------------
# Layer diagnostics (requires graph_tool)
# ---------------------------------------------------------------------------

def diagnose_layers(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                    max_hops=3, beam_width=80, per_branch_width=5):
    """Diagnose each layer via a single-pass unified BFS from the anchor.

    One beam search traversal tracks TWO things per path simultaneously:
    - frontier_depth (int): maximum contiguous layer coverage (0,1,...,k).
      Advances when the NEXT expected layer's target relation is matched.
    - anchor_layers (frozenset of ints): ALL layers whose target relations
      were reached, regardless of order.

    After the single BFS:
    - anchor_hit[i] = any path has layer i in its anchor_layers
    - frontier_hit[i] = any path has frontier_depth >= i

    This replaces the original 2N separate BFS calls (frontier + anchor per
    layer) with one unified traversal, using CVT vertex properties for O(1)
    hop increments and tuple-based paths for speed.
    """
    try:
        import graph_tool as gt
    except ImportError:
        raise ImportError("graph_tool is required for diagnose_layers")

    if anchor_idx is None:
        return []

    # Build undirected graph_tool graph with CVT vertex property
    graph = gt.Graph(directed=False)
    graph.add_vertex(len(entity_list))
    ep_rel = graph.new_edge_property("int")
    graph.edge_properties["relation_id"] = ep_rel

    # Mark CVT nodes at construction time (O(1) check later instead of regex)
    _cvt_re = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
    vp_cvt = graph.new_vertex_property("bool")
    for i, name in enumerate(entity_list):
        vp_cvt[i] = bool(_cvt_re.match(name)) if name else False

    # Add edges
    for i in range(len(h_ids)):
        edge = graph.add_edge(h_ids[i], t_ids[i])
        ep_rel[edge] = r_ids[i]

    # Cached neighbor lookups via graph_tool C++ backend
    _nb_cache = {}

    def _get_neighbors(node_idx):
        if node_idx in _nb_cache:
            return _nb_cache[node_idx]
        v = graph.vertex(node_idx)
        nbs = []
        for e in v.out_edges():
            s, t = int(e.source()), int(e.target())
            nb = t if s == node_idx else s
            nbs.append((nb, int(ep_rel[e])))
        result = tuple(nbs)
        _nb_cache[node_idx] = result
        return result

    # rel -> set of layer indices mapping
    rel_to_layers = {}
    for si, rs in enumerate(step_relations):
        for r in rs:
            rel_to_layers.setdefault(r, set()).add(si)

    # Precompute step_relations as list of sets for fast membership test
    step_rel_sets = [set(rs) for rs in step_relations]
    n_layers = len(step_relations)
    max_iterations = max_hops * 3 * max(n_layers, 1)

    def _hop_inc(curr, nxt):
        return 0 if (vp_cvt[curr] or vp_cvt[nxt]) else 1

    def _prune(paths, limit):
        if len(paths) <= limit:
            return paths
        grouped = {}
        for p in paths:
            sig = (p[0][-1], p[4])
            grouped.setdefault(sig, []).append(p)
        result = []
        overflow = []
        for group in grouped.values():
            group.sort(key=lambda p: p[4], reverse=True)
            result.extend(group[:per_branch_width])
            overflow.extend(group[per_branch_width:])
        if len(result) < limit and overflow:
            overflow.sort(key=lambda p: p[4], reverse=True)
            result.extend(overflow[:limit - len(result)])
        return result[:limit]

    # Path tuple: (nodes, rels, depth, hops, frontier_depth, anchor_layers)
    active = [((anchor_idx,), (), 0, 0, -1, frozenset())]

    # Collect per-layer hit info during traversal
    frontier_hits = defaultdict(list)
    anchor_hits = defaultdict(list)

    for _ in range(max_iterations):
        if not active:
            break
        new_active = []
        for nodes, rels, dep, hops, f_depth, a_layers in active:
            current = nodes[-1]
            for nb, rel in _get_neighbors(current):
                if nb in nodes:
                    continue
                new_hops = hops + _hop_inc(current, nb)
                if new_hops > max_hops:
                    continue

                # Frontier: advance contiguous coverage
                new_f = f_depth
                next_expected = f_depth + 1
                if next_expected < n_layers and rel in step_rel_sets[next_expected]:
                    new_f = next_expected

                # Anchor: record all layers matched by this relation
                matched = rel_to_layers.get(rel, set())
                new_a = a_layers | frozenset(matched)

                new_path = (nodes + (nb,), rels + (rel,), dep + 1, new_hops, new_f, new_a)
                new_active.append(new_path)

                # Record hits
                for li in matched:
                    anchor_hits[li].append(new_path)
                if new_f > f_depth:
                    for li in range(f_depth + 1, new_f + 1):
                        frontier_hits[li].append(new_path)

        active = _prune(new_active, beam_width)

    # Build diagnostics
    diagnostics = []
    for li in range(n_layers):
        diagnostics.append({
            "layer_idx": li,
            "frontier_hit": bool(frontier_hits.get(li)),
            "anchor_hit": bool(anchor_hits.get(li)),
            "frontier_count": len(frontier_hits.get(li, [])),
            "anchor_count": len(anchor_hits.get(li, [])),
        })
    return diagnostics


# ---------------------------------------------------------------------------
# Canonicalization helper (used by frontier_expand_layers)
# ---------------------------------------------------------------------------

def _canonicalize_by_step_hits(path, layer_rels):
    """Canonicalize a raw path by truncating each step at its first target-relation hit.

    For each step, scan the relation sequence from the current position and find the
    first edge matching that step's target relations. Keep edges up to that hit, then
    continue scanning from the hit point for the next step. This prevents a step from
    accumulating extra relations after its first valid hit.

    CVT nodes at hit points are preserved — the next step continues from them.
    """
    nodes = path["nodes"]
    rels = path["relations"]

    if not rels or not layer_rels:
        return path

    keep_nodes = [nodes[0]]
    keep_rels = []
    committed_nodes = [nodes[0]]
    covered = set()
    scan_start = 0

    for step_idx, target_rels in enumerate(layer_rels):
        if not target_rels:
            continue
        target_set = set(target_rels)
        hit = None
        for e in range(scan_start, len(rels)):
            if rels[e] in target_set:
                hit = e
                break
        if hit is None:
            break
        # Keep all edges from scan_start through hit
        for e in range(scan_start, hit + 1):
            keep_rels.append(rels[e])
            keep_nodes.append(nodes[e + 1])
        committed_nodes.append(nodes[hit + 1])
        covered.add(step_idx)
        scan_start = hit + 1

    if not keep_rels:
        return path

    return {
        "nodes": keep_nodes,
        "relations": keep_rels,
        "committed_nodes": committed_nodes,
        "covered_steps": frozenset(covered),
        "depth": len(keep_nodes) - 1,
    }


# ---------------------------------------------------------------------------
# Multi-step frontier expansion
# ---------------------------------------------------------------------------

def frontier_expand_layers(anchor_idx, step_relations, steps,
                           h_ids, r_ids, t_ids, entity_list,
                           beam_width=80, max_hops_per_step=2):
    """Multi-step traversal: single BFS per step, hit-and-stop per branch.

    Per step:
    1. From each active path's endpoint, BFS up to K hops
    2. When an edge matches a target relation -> record that path, STOP the branch
    3. Other branches continue independently (backtrack to explore other edges)
    4. First encountered target relation per branch — no stacking
    5. Matched endpoints form the next step's frontier

    Step-skip fallback: if entire step produces zero matches, borrow the
    next step's target relations and re-explore from the current frontier.

    step_relations: List[List[int]] — target relation indices per step.
    Returns (paths, max_depth, max_cov) compatible with relation_prior_expand.
    """
    n_layers = len(step_relations)
    if n_layers == 0:
        return [], 0, 0

    layer_rels: List[List[int]] = step_relations
    if not any(layer_rels):
        return [], 0, 0

    # Build adjacency: node_idx -> list of (neighbor_idx, rel_idx)
    # (directed if KGQA_DIRECTED_TRAVERSAL=1, else undirected)
    adj: Dict[int, list] = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        adj.setdefault(h, []).append((t, r))
        if not DIRECTED_TRAVERSAL:
            adj.setdefault(t, []).append((h, r))

    n_ents = len(entity_list)

    def _expand_through_cvt(nb_idx, current_idx, seen):
        return [(nb_idx, [nb_idx], [])]

    def _bfs_step(paths, rels, cover_step):
        """Single BFS pass: find K-hop paths ending with a target relation."""
        results = []
        pb = max(beam_width // max(len(paths), 1), 5)
        for path in paths:
            start = path["nodes"][-1]
            committed_nodes = list(path.get("committed_nodes", [path["nodes"][0]]))
            p_seen = set(committed_nodes)
            frontier = [(start, [], [])]
            hits = []
            for hop in range(max_hops_per_step):
                nf = []
                for cur, in_nodes, in_rels in frontier:
                    seen = p_seen | set(in_nodes)
                    for nb, rel in adj.get(cur, []):
                        if nb in seen:
                            continue
                        # Inverse-pair loop detection
                        if in_rels and in_rels[-1] == rel:
                            continue
                        if rel in rels:
                            for fn, en, er in _expand_through_cvt(nb, cur, seen):
                                hits.append((fn, in_nodes + en, in_rels + [rel] + er))
                        else:
                            if hop < max_hops_per_step - 1:
                                nb_name = entity_list[nb] if 0 <= nb < n_ents else ""
                                if is_cvt_like(nb_name):
                                    for cvt_nb, cvt_rel in adj.get(nb, []):
                                        if cvt_nb not in seen and cvt_nb != cur:
                                            if cvt_rel in rels:
                                                hits.append((cvt_nb, in_nodes + [nb, cvt_nb],
                                                             in_rels + [rel, cvt_rel]))
                                            else:
                                                nf.append((cvt_nb, in_nodes + [nb, cvt_nb],
                                                            in_rels + [rel, cvt_rel]))
                                else:
                                    nf.append((nb, in_nodes + [nb], in_rels + [rel]))
                if len(nf) > pb:
                    nf = nf[:pb]
                frontier = nf
            for end_node, extra_nodes, extra_rels in hits:
                results.append({
                    "nodes": path["nodes"] + extra_nodes,
                    "relations": path["relations"] + extra_rels,
                    "committed_nodes": committed_nodes + [end_node],
                    "covered_steps": path["covered_steps"] | frozenset({cover_step}),
                    "depth": path["depth"] + len(extra_nodes),
                })
            if not hits:
                results.append(path)
        return results

    # Active paths: nodes, relations, covered_steps
    active = [{"nodes": [anchor_idx], "relations": [],
               "committed_nodes": [anchor_idx],
               "covered_steps": frozenset(), "depth": 0}]

    step_idx = 0
    while step_idx < n_layers:
        target_rels = set(layer_rels[step_idx])
        if not target_rels:
            step_idx += 1
            continue

        new_active = _bfs_step(active, target_rels, step_idx)
        advance = 1

        # Step-skip fallback: entire step unmatched -> borrow next step's relations
        any_matched = any(step_idx in p["covered_steps"] for p in new_active)
        if not any_matched and step_idx + 1 < n_layers:
            fallback_rels = set(layer_rels[step_idx + 1])
            if fallback_rels:
                fallback = _bfs_step(active, fallback_rels, step_idx + 1)
                if any(step_idx + 1 in p["covered_steps"] for p in fallback):
                    new_active = fallback
                    advance = 2

        if not new_active:
            break

        # Beam prune: group by (endpoint, covered_steps) for diversity
        if len(new_active) > beam_width:
            grouped: Dict[tuple, list] = {}
            for p in new_active:
                sig = (p["nodes"][-1], p["covered_steps"])
                grouped.setdefault(sig, []).append(p)
            for sig, group in grouped.items():
                group.sort(key=lambda x: (-len(x["covered_steps"]), x["depth"]))
            result = [group[0] for group in grouped.values()]
            overflow = []
            for group in grouped.values():
                overflow.extend(group[1:])
            overflow.sort(key=lambda x: (-len(x["covered_steps"]), x["depth"]))
            remaining = beam_width - len(result)
            if remaining > 0:
                result.extend(overflow[:remaining])
            new_active = result[:beam_width]

        active = new_active
        step_idx += advance

    # Convert to output format
    paths = [p for p in active if p["depth"] > 0]
    if not paths:
        return [], 0, 0

    # Post-processing: canonicalize each path by step-level first-hit truncation.
    # For each step, keep only edges up to the first hit of that step's target relations,
    # then continue from the hit point for the next step. This prevents a single step
    # from accumulating extra relations after its first valid hit.
    canonicalized = []
    for p in paths:
        canon = _canonicalize_by_step_hits(p, layer_rels)
        if canon and canon["depth"] > 0:
            canonicalized.append(canon)
    if canonicalized:
        paths = canonicalized

    # Sort by layers covered (desc), then depth (asc: prefer shorter)
    paths.sort(key=lambda p: (-len(p["covered_steps"]), p["depth"]))

    # Add matched_relations for compatibility
    for p in paths:
        p["matched_relations"] = frozenset(p["relations"])

    max_depth = max(p["depth"] for p in paths)
    max_cov = max(len(p["covered_steps"]) for p in paths)

    return paths, max_depth, max_cov
