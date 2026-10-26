"""CVT (Compound Value Type) node detection and expansion utilities.

Pure graph operations on index arrays. No kgqa-internal dependencies.
"""
from __future__ import annotations

import re
from collections import Counter


_CVT_CACHE: dict = {}


def is_cvt_like(name: str) -> bool:
    """Detect CVT (Compound Value Type) nodes.
    Matches m.xxx / g.xxx pattern. Results cached (called ~30k times/run)."""
    if not name or len(name) < 2:
        return False
    cached = _CVT_CACHE.get(name)
    if cached is not None:
        return cached
    result = name[0] in "mg" and len(name) > 2 and name[1] == "." and name[2:].replace("_", "").isalnum()
    _CVT_CACHE[name] = result
    return result


def expand_cvt_leaves(ents, rels, h_ids, r_ids, t_ids):
    """Auto-expand CVT leaf nodes (degree <= 1) by finding additional edges
    from other triples in the subgraph that share the same CVT node.
    Returns potentially augmented (ents, rels, h_ids, r_ids, t_ids)."""
    node_degree = Counter()
    for i in range(len(h_ids)):
        node_degree[h_ids[i]] += 1
        node_degree[t_ids[i]] += 1

    # Find CVT nodes with degree <= 1 (leaf/dead-end)
    cvt_leaves = []
    for idx, name in enumerate(ents):
        if is_cvt_like(name) and node_degree.get(idx, 0) <= 1:
            cvt_leaves.append(idx)

    if not cvt_leaves:
        return ents, rels, h_ids, r_ids, t_ids

    # For each CVT leaf, search the subgraph for any triples where it appears
    # that weren't included (e.g., via shared intermediate nodes)
    # Since we only have the subgraph, we can only find edges already present
    # but potentially missed due to indexing
    # No-op for now: full expansion requires KG API access
    # Flag count for debugging
    return ents, rels, h_ids, r_ids, t_ids


def expand_node(node_idx, rel_indices, h_ids, r_ids, t_ids, reverse=False):
    children = []
    for i in range(len(h_ids)):
        if reverse:
            if t_ids[i] == node_idx and r_ids[i] in rel_indices:
                children.append((h_ids[i], r_ids[i]))
        else:
            if h_ids[i] == node_idx and r_ids[i] in rel_indices:
                children.append((t_ids[i], r_ids[i]))
    return children


def expand_through_cvt(node_idx, h_ids, r_ids, t_ids, entity_list):
    name = entity_list[node_idx] if 0 <= node_idx < len(entity_list) else ""
    if not is_cvt_like(name):
        return []
    children, seen = [], set()
    for i in range(len(h_ids)):
        if h_ids[i] == node_idx and t_ids[i] not in seen:
            children.append((t_ids[i], r_ids[i])); seen.add(t_ids[i])
        if t_ids[i] == node_idx and h_ids[i] not in seen:
            children.append((h_ids[i], r_ids[i])); seen.add(h_ids[i])
    return children
