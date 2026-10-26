"""Path processing utilities for KGQA graph traversal.

Includes path compression, triple expansion, relation segmentation,
noise filtering, and candidate extraction.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from kgqa.traversal.cvt import is_cvt_like, expand_through_cvt


# ---------------------------------------------------------------------------
# Text normalization helpers (inlined since kgqa.core.utils doesn't exist yet)
# ---------------------------------------------------------------------------

def rel_to_text(rel: str) -> str:
    """Return original dot-notation format for LLM display: 'people.person.religion'"""
    return rel


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9%.' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Noisy relation detection
# ---------------------------------------------------------------------------

def _is_noisy_path_relation(rel_name):
    rel_name = rel_name or ""
    noisy_prefixes = (
        "type.",
        "common.",
        "freebase.",
        "kg.",
        "user.",
        "base.ontologies.",
        "rdf-schema#",       # W3C schema plumbing (domain/range) — same
        "owl#",              # universal-structural class as type.* (2026-08-21,
                             # Berlin specimen: outranked the gold relation)
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


def _path_relation_names(path, rels_list):
    return [
        rel_to_text(rels_list[rel_idx]) if 0 <= rel_idx < len(rels_list) else "?"
        for rel_idx in path.get("relations", [])
    ]


# ---------------------------------------------------------------------------
# Path -> triples expansion
# ---------------------------------------------------------------------------

def expand_to_triples(paths, ents, rels_list):
    """Expand raw paths to triples, bridging through CVT intermediate nodes.

    When a triple endpoint is a CVT node (m.xxx / g.xxx), we bridge forward
    through consecutive CVT nodes to the next real entity, merging relation
    names with " > ". This preserves connectivity while hiding gibberish.
    """
    triples = []
    seen = set()
    for path in paths:
        nodes = path["nodes"]
        rels = path["relations"]
        # Build raw (h_name, r_text, t_name) triples
        raw = []
        for i in range(min(len(rels), len(nodes) - 1)):
            h_idx, r_idx, t_idx = nodes[i], rels[i], nodes[i + 1]
            h_name = ents[h_idx] if 0 <= h_idx < len(ents) else "?"
            t_name = ents[t_idx] if 0 <= t_idx < len(ents) else "?"
            r_text = rel_to_text(rels_list[r_idx]) if 0 <= r_idx < len(rels_list) else "?"
            raw.append((h_name, r_text, t_name))
        # Bridge through CVT nodes
        i = 0
        while i < len(raw):
            h, r, t = raw[i]
            if is_cvt_like(h):
                i += 1
                continue
            if is_cvt_like(t):
                merged_rels = [r]
                j = i
                while j < len(raw) - 1 and is_cvt_like(raw[j][2]):
                    j += 1
                    merged_rels.append(raw[j][1])
                final_t = raw[j][2]
                if not is_cvt_like(final_t):
                    merged_r = " > ".join(merged_rels)
                    sig = (normalize(h), normalize(merged_r), normalize(final_t))
                    if sig not in seen:
                        seen.add(sig)
                        triples.append((h, merged_r, final_t))
                    i = j + 1
                else:
                    # Preserve tail relation even if it ends at a CVT/value node.
                    sig = (normalize(h), normalize(r), normalize(final_t))
                    if sig not in seen:
                        seen.add(sig)
                        triples.append((h, r, final_t))
                    i += 1
            else:
                sig = (normalize(h), normalize(r), normalize(t))
                if sig not in seen:
                    seen.add(sig)
                    triples.append((h, r, t))
                i += 1
    return triples


# ---------------------------------------------------------------------------
# Path compression into logical patterns
# ---------------------------------------------------------------------------

def compress_paths(paths, ents, rels_list, anchor_idx, breakpoint_indices):
    """Compress raw paths into logical patterns with role-aware node identification.

    Pattern format: Anchor --[rel_chain]--> <node> ... --> [Candidate] --> Endpoint
    - Anchor: starting entity name
    - <node>: abstracted bridge entities (not shown by name)
    - [Candidate]: potential answer entities
    - Endpoint: constraint entity (if exists)
    Merges paths with same relation chain pattern, collecting unique candidates.
    """
    if not paths or anchor_idx is None:
        return []

    anchor_name = ents[anchor_idx] if 0 <= anchor_idx < len(ents) else "?"
    patterns: Dict[tuple, dict] = {}

    def _has_non_cvt_loop(path):
        seen_nodes = set()
        for node_idx in path.get("nodes", []):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if is_cvt_like(name):
                continue
            if node_idx in seen_nodes:
                return True
            seen_nodes.add(node_idx)
        return False

    def _raw_path_to_readable(path):
        """Render one witness path with abstract node labels.

        Keep relation order and path shape, but hide concrete intermediate entities.
        This lets the model judge semantic fit of the path rather than overfitting
        to visible candidate names.
        """
        nodes = path["nodes"]
        rels = path["relations"]
        if not nodes:
            return anchor_name
        # If the path already reaches a non-CVT entity and then only trails into
        # CVT/value nodes, hide that trailing CVT tail from the logical path.
        # We only keep a CVT tail visible when the core relation itself lands on CVT
        # (i.e. no intermediate non-CVT endpoint has already been reached).
        cutoff_len = len(nodes)
        non_cvt_positions = []
        for i, node_idx in enumerate(nodes):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if not is_cvt_like(name):
                non_cvt_positions.append(i)
        if len(non_cvt_positions) >= 2:
            last_non_cvt_pos = non_cvt_positions[-1]
            trailing_count = len(nodes) - 1 - last_non_cvt_pos
            if trailing_count > 1:
                # Keep 1 trailing hop for relation visibility
                cutoff_len = last_non_cvt_pos + 2
        nodes = nodes[:cutoff_len]
        rels = rels[: max(0, cutoff_len - 1)]
        parts = []
        node_labels = {}
        next_label_id = 1
        for i, node_idx in enumerate(nodes):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else "?"
            if i == 0:
                if node_idx == anchor_idx:
                    parts.append(anchor_name)
                else:
                    parts.append("node0")
                continue
            prev_rel_idx = rels[i - 1] if i - 1 < len(rels) else None
            rel_text = rel_to_text(rels_list[prev_rel_idx]) if prev_rel_idx is not None and prev_rel_idx < len(rels_list) else "?"
            if node_idx in node_labels:
                node_text = node_labels[node_idx]
            elif node_idx in breakpoint_indices and node_idx != anchor_idx:
                node_text = f"node{next_label_id}[endpoint]"
                node_labels[node_idx] = node_text
                next_label_id += 1
            else:
                node_text = f"node{next_label_id}"
                node_labels[node_idx] = node_text
                next_label_id += 1
            parts.append(f"--[{rel_text}]--> {node_text}")
        return " ".join(parts)

    for path in paths:
        if _has_non_cvt_loop(path):
            continue
        # Extract non-CVT nodes with positions
        sig_nodes = []
        for i, node_idx in enumerate(path["nodes"]):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if not is_cvt_like(name):
                sig_nodes.append((i, node_idx, name))

        # Build relation chain
        rel_chain = []
        if len(sig_nodes) >= 2:
            # Normal: relation chain between consecutive non-CVT nodes
            for j in range(1, len(sig_nodes)):
                prev_pos = sig_nodes[j - 1][0]
                curr_pos = sig_nodes[j][0]
                rel_indices_between = path["relations"][prev_pos:curr_pos]
                rel_texts = [rel_to_text(rels_list[ri]) for ri in rel_indices_between if ri < len(rels_list)]
                rel_chain.append(" -> ".join(rel_texts) if rel_texts else "?")
            # Append trailing relations from last non-CVT to CVT/value tail
            last_non_cvt_pos = sig_nodes[-1][0]
            trailing_rels = path["relations"][last_non_cvt_pos:]
            if trailing_rels:
                rel_texts = [rel_to_text(rels_list[ri]) for ri in trailing_rels if ri < len(rels_list)]
                rel_chain.append(" -> ".join(rel_texts) if rel_texts else "?")
        elif path["relations"]:
            # Path ends at CVT: use full relation chain as pattern
            rel_chain = [rel_to_text(rels_list[ri]) for ri in path["relations"] if ri < len(rels_list)]
        else:
            continue

        # Identify endpoint
        endpoint_name = None
        for _, node_idx, name in sig_nodes:
            if node_idx in breakpoint_indices and node_idx != anchor_idx:
                endpoint_name = name
                break

        # Identify candidates: ALL non-anchor, non-endpoint, non-CVT nodes
        # The answer is often a bridge entity being verified by the last relation,
        # not the last node before the endpoint.
        candidates = set()
        bp_set = breakpoint_indices if breakpoint_indices else set()
        for _, node_idx, name in sig_nodes:
            if node_idx != anchor_idx and node_idx not in bp_set and not is_cvt_like(name):
                candidates.add(name)

        # Pattern key: relation chain + endpoint
        key = (tuple(rel_chain), endpoint_name)

        # Compute causal tier for this raw path
        from kgqa.traversal.frontier import score_causal_tier
        covered = path.get("covered_steps", frozenset())
        tier = score_causal_tier(covered, bridge_length=path.get("depth", 0))

        if key not in patterns:
            patterns[key] = {
                "rel_chain": rel_chain,
                "endpoint": endpoint_name,
                "candidates": set(),
                "raw_paths": [],
                "best_tier": (0, -1, 0),
                "best_raw_path": path,
                "best_depth": path.get("depth", 0),
            }
        patterns[key]["candidates"].update(candidates)
        if tier > patterns[key]["best_tier"]:
            patterns[key]["best_tier"] = tier
            patterns[key]["best_raw_path"] = path
            patterns[key]["best_depth"] = path.get("depth", 0)
        elif tier == patterns[key]["best_tier"] and path.get("depth", 0) < patterns[key].get("best_depth", 10**9):
            patterns[key]["best_raw_path"] = path
            patterns[key]["best_depth"] = path.get("depth", 0)
        if len(patterns[key]["raw_paths"]) < 200:
            patterns[key]["raw_paths"].append(path)

    # Convert to sorted list — rank only by the best causal tier of the pattern
    result = []
    for key, group in sorted(patterns.items(), key=lambda x: x[1]["best_tier"], reverse=True):
        cands = sorted(group["candidates"])[:20]
        # Use one witness path to preserve bridge structure for display.
        readable = _raw_path_to_readable(group["best_raw_path"])

        result.append({
            "rel_chain": group["rel_chain"],
            "endpoint": group["endpoint"],
            "candidates": cands,
            "best_tier": group["best_tier"],
            "readable": readable,
            "raw_paths": group["raw_paths"],
            "best_raw_path": group["best_raw_path"],
        })
    return result


# ---------------------------------------------------------------------------
# Relation segment extraction
# ---------------------------------------------------------------------------

def _extract_relation_segments_from_path(path, ents):
    """Collapse a raw path into relation segments.

    Keep:
    - non-CVT -> non-CVT segments
    - direct non-CVT -> ... -> CVT tail segment only when no later non-CVT entity
      has already been reached

    This preserves direct value constraints such as:
    country -> statistical_region.child_labor_percent -> g.xxx
    but removes tails such as:
    person -> religion -> Judaism -> membership -> CVT
    where the core relation has already reached a real entity.
    """
    sig_positions = []
    for i, node_idx in enumerate(path.get("nodes", [])):
        name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
        if not is_cvt_like(name):
            sig_positions.append(i)
    segments = []
    for j in range(1, len(sig_positions)):
        prev_pos = sig_positions[j - 1]
        curr_pos = sig_positions[j]
        rel_seq = path.get("relations", [])[prev_pos:curr_pos]
        if rel_seq:
            segments.append(list(rel_seq))
    # Preserve the tail segment when the endpoint is a CVT node.
    # Endpoint CVTs need attribute expansion (rate, date, etc.).
    # Middle CVTs are just path intermediaries — their direction is already determined.
    nodes = path.get("nodes", [])
    rels = path.get("relations", [])
    if len(sig_positions) >= 1:
        last_node_pos = len(nodes) - 1
        last_node_name = ents[nodes[last_node_pos]] if 0 <= last_node_pos < len(nodes) else ""
        if is_cvt_like(last_node_name) and sig_positions[-1] < last_node_pos:
            tail_rel_seq = rels[sig_positions[-1]:]
            if tail_rel_seq:
                segments.append(list(tail_rel_seq))
    return segments


# ---------------------------------------------------------------------------
# Breakpoint-based path selection
# ---------------------------------------------------------------------------

def _path_hits_breakpoints(path, breakpoint_nodes, h_ids, r_ids, t_ids, entity_list):
    """Return True if a path lands on, or terminates in a CVT adjacent to, any breakpoint."""
    if not path or not breakpoint_nodes:
        return False
    nodes = path.get("nodes", [])
    if not nodes:
        return False

    last = nodes[-1]
    if last in breakpoint_nodes:
        return True

    last_name = entity_list[last] if 0 <= last < len(entity_list) else ""
    if is_cvt_like(last_name):
        for nb_idx, _ in expand_through_cvt(last, h_ids, r_ids, t_ids, entity_list):
            if nb_idx in breakpoint_nodes:
                return True
    return False


def prefer_breakpoint_hit_paths(paths, breakpoints, h_ids, r_ids, t_ids, entity_list):
    """Prefer paths that hit explicit endpoint breakpoints, but never drop recall if none do."""
    if not paths or not breakpoints:
        return paths
    hit_paths = [
        p for p in paths
        if _path_hits_breakpoints(p, breakpoints, h_ids, r_ids, t_ids, entity_list)
    ]
    return hit_paths if hit_paths else paths


# ---------------------------------------------------------------------------
# Candidate extraction from paths
# ---------------------------------------------------------------------------

def _extract_path_candidates(paths, anchor_idx, ents, h_ids, r_ids, t_ids):
    """Extract deduplicated candidates from path traversal nodes only (no HR frontier).
    Used for GT recall — measures whether graph traversal can reach the answer.
    """
    path_nodes = {anchor_idx}
    for path in paths:
        path_nodes.update(path.get("nodes", []))
    cands = []
    for node_idx in sorted(path_nodes):
        if node_idx == anchor_idx:
            continue
        name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
        if is_cvt_like(name):
            for cvt_idx, _ in expand_through_cvt(node_idx, h_ids, r_ids, t_ids, ents):
                if cvt_idx != anchor_idx and 0 <= cvt_idx < len(ents) and not is_cvt_like(ents[cvt_idx]):
                    cands.append(ents[cvt_idx])
        else:
            cands.append(name)
    seen = set()
    unique = []
    for c in cands:
        nc = normalize(c)
        if len(nc) < 2 or not c.isascii():
            continue
        if nc not in seen:
            seen.add(nc)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Candidate name extraction from paths
# ---------------------------------------------------------------------------

def _extract_candidate_names_from_paths(paths, ents, anchor_idx, breakpoint_indices, limit=20):
    """Extract unique candidate names from paths, sorted by frequency."""
    from collections import defaultdict
    counts = _candidate_name_counts_from_paths(paths, ents, anchor_idx, breakpoint_indices)
    return [name for name, _ in counts[:limit]]


def _candidate_name_counts_from_paths(paths, ents, anchor_idx, breakpoint_indices):
    """Count candidate name occurrences across paths."""
    from collections import defaultdict
    counts = defaultdict(int)
    bp_set = breakpoint_indices if breakpoint_indices else set()
    for path in paths:
        nodes = path.get("nodes", [])
        for node_idx in nodes:
            if node_idx == anchor_idx:
                continue
            if node_idx in bp_set:
                continue
            if 0 <= node_idx < len(ents):
                name = ents[node_idx]
                if not is_cvt_like(name):
                    counts[normalize(name)] += 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))
