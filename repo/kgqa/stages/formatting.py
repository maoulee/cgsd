"""Evidence formatting helpers for the KGQA pipeline.

Builds and renders pattern-grouped evidence trees, CVT-resolved triples,
endpoint rescue patterns, and grouped-triple displays for LLM reasoning.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Tuple

from kgqa.core.utils import normalize, rel_to_text
from kgqa.traversal.cvt import is_cvt_like, expand_through_cvt


# Latin-script allow-ranges for the multilingual-noise filter. Freebase stores
# the same entity's aliases in many scripts (CJK/Arabic/Cyrillic/…) as separate
# "entities" — those flood evidence with unreadable duplicates, so non-Latin
# LETTERS are dropped. But Latin-with-diacritics/symbols MUST stay: U+2212 minus
# in 'UTC−05:00', Vietnamese 'Tiến Quân Ca', 'Zürich', 'Bø'. isascii() was too
# coarse (it dropped these legit entities → empty evidence); this script check
# keeps the multilingual-noise intent without the over-broad ASCII restriction.
_LATIN_RANGES = ((0x0000, 0x024F), (0x1E00, 0x1EFF), (0x2C60, 0x2C7F))

# per-CHARACTER memo (walk-perf, 2026-09-07): _is_latinish ran 5.3k times on
# ONE sparse-case walk (58% of its evidence build) — each call re-deriving
# unicodedata.category + range checks per character. Entity vocabularies draw
# from a tiny character alphabet, so a char-level table collapses the cost to
# a dict hit. Same inputs → same verdicts, byte-equivalent.
_LATIN_CHAR_OK: Dict[str, bool] = {}

# case-level static memos for build_pattern_evidence_triples (2026-09-10):
# per-CVT scored edge lists + (h,r,t) static filter results, keyed by the
# pinned case arrays — the same hub's CVTs are expanded once per CASE, not
# once per sg call. Entries pin (strong-ref) their source arrays so the
# id() keys stay valid while the entry lives (walk-perf audit pattern).
_CASE_STATIC_MEMO: Dict[tuple, dict] = {}


def _is_latinish(name) -> bool:
    """True if name contains no non-Latin LETTER (allows Latin+diacritics, digits,
    punctuation, and symbols like U+2212). Drops CJK/Arabic/Cyrillic/etc. words."""
    for ch in str(name):
        ok = _LATIN_CHAR_OK.get(ch)
        if ok is None:
            if not unicodedata.category(ch).startswith("L"):   # non-letter allowed
                ok = True
            else:
                cp = ord(ch)
                ok = any(lo <= cp <= hi for lo, hi in _LATIN_RANGES)
            _LATIN_CHAR_OK[ch] = ok
        if not ok:
            return False
    return True



# ---------------------------------------------------------------------------
# PatternEvidence data holder
# ---------------------------------------------------------------------------

class PatternEvidence:
    """Triples and metadata for one relation pattern, used in Stage 8 reasoning."""
    __slots__ = ("label", "readable", "candidates", "triples", "tree_data")

    def __init__(self, label, readable, candidates, triples, tree_data=None):
        self.label = label
        self.readable = readable
        self.candidates = candidates
        self.triples = triples
        self.tree_data = tree_data  # (step_ent_names, step_edge_names, cvt_attrs, cvt_to_named)


# ---------------------------------------------------------------------------
# Internal helpers for path / relation noise detection
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


# CVT attribute shorts with no semantic value for QA — stripped from CVT
# display names so the model sees only meaningful attributes (dates, roles,
# names), not Freebase bookkeeping markers like has_no_value=To.
_NOISY_CVT_ATTR_SHORTS = {
    "has_no_value", "no_value", "is_reviewed",
    "type", "types", "instance", "instances",
    "permission", "guid", "mid", "key", "keys",
}


def _is_noisy_cvt_attr_short(attr_pair: str) -> bool:
    """True if a 'short=value' CVT attr pair is bookkeeping noise."""
    key = attr_pair.split("=", 1)[0].replace(".inv", "")
    return key in _NOISY_CVT_ATTR_SHORTS


def _logical_path_needs_endpoint_rescue(lp, rels_list, min_depth=4):
    best = lp.get("best_raw_path") or {}
    rel_names = _path_relation_names(best, rels_list)
    return best.get("depth", len(best.get("relations", []))) >= min_depth or any(
        _is_noisy_path_relation(rel_name) for rel_name in rel_names
    )


# ---------------------------------------------------------------------------
# Endpoint rescue patterns
# ---------------------------------------------------------------------------

def build_endpoint_rescue_patterns(paths, selected_patterns, ents, rels_list,
                                   anchor_idx, breakpoint_indices,
                                   max_paths=4):
    """Build endpoint-constrained rescue patterns for noisy/overlong selections.

    This is a Stage 8 evidence supplement: when selected logical paths reach an
    endpoint through noisy or overlong chains, expose direct anchor-to-endpoint
    raw paths ranked by step coverage first, then noise and depth.
    """
    if not paths or not breakpoint_indices:
        return []
    if not all(_logical_path_needs_endpoint_rescue(lp, rels_list) for lp in selected_patterns):
        return []

    endpoint_set = {idx for idx in breakpoint_indices if idx is not None and idx != anchor_idx}
    if not endpoint_set:
        return []

    rescue_candidates = []
    for path in paths:
        nodes = path.get("nodes", [])
        rels = path.get("relations", [])
        if not nodes or not rels:
            continue
        hit_pos = None
        for pos, node_idx in enumerate(nodes):
            if node_idx in endpoint_set:
                hit_pos = pos
                break
        if hit_pos is None or hit_pos == 0:
            continue
        trimmed = dict(path)
        trimmed["nodes"] = nodes[:hit_pos + 1]
        trimmed["relations"] = rels[:hit_pos]
        trimmed["depth"] = len(trimmed["relations"])
        rel_names = _path_relation_names(trimmed, rels_list)
        noisy_count = sum(1 for rel_name in rel_names if _is_noisy_path_relation(rel_name))
        covered_count = len(trimmed.get("covered_steps", frozenset()))
        rescue_candidates.append((-covered_count, noisy_count, trimmed["depth"], trimmed))

    if not rescue_candidates:
        return []

    rescue_candidates.sort(key=lambda item: item[:3])
    support = []
    seen = set()
    for _, _, _, path in rescue_candidates:
        sig = (tuple(path.get("nodes", [])), tuple(path.get("relations", [])))
        if sig in seen:
            continue
        seen.add(sig)
        support.append(path)
        if len(support) >= max_paths:
            break
    if not support:
        return []

    best = support[0]
    endpoint_name = ents[best["nodes"][-1]] if 0 <= best["nodes"][-1] < len(ents) else None
    readable_parts = []
    for i, node_idx in enumerate(best.get("nodes", [])):
        if i == 0:
            readable_parts.append(ents[node_idx] if 0 <= node_idx < len(ents) else "?")
            continue
        rel_idx = best["relations"][i - 1]
        rel_text = rel_to_text(rels_list[rel_idx]) if 0 <= rel_idx < len(rels_list) else "?"
        node_text = f"node{i}"
        if node_idx in endpoint_set:
            node_text += "[endpoint]"
        readable_parts.append(f"--[{rel_text}]--> {node_text}")

    candidates = []
    for path in support:
        for node_idx in path.get("nodes", []):
            if node_idx == anchor_idx or node_idx in endpoint_set:
                continue
            if 0 <= node_idx < len(ents):
                name = ents[node_idx]
                if not is_cvt_like(name):
                    candidates.append(name)
    deduped_candidates = []
    seen_candidates = set()
    for name in candidates:
        key = normalize(name)
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        deduped_candidates.append(name)

    return [{
        "rel_chain": [],
        "endpoint": endpoint_name,
        "candidates": deduped_candidates[:20],
        "best_tier": (len(best.get("covered_steps", frozenset())), -best.get("depth", 0), 0),
        "readable": "ENDPOINT RESCUE: " + " ".join(readable_parts),
        "raw_paths": support,
        "best_raw_path": best,
    }]


# ---------------------------------------------------------------------------
# Pattern evidence triple builder
# ---------------------------------------------------------------------------

def build_pattern_evidence_triples(selected_patterns, ents, rels_list, h_ids, r_ids, t_ids,
                                   anchor_idx, max_grouped_lines=120,
                                   selected_rel_ids=None):
    """Build bounded pattern evidence from the witness path plus sibling raw paths.

    The Stage 7 logical path is grouped from many raw paths, but a single witness
    path is often too narrow for Stage 8 reasoning. Here we keep the witness path
    and then add a bounded number of sibling raw paths that introduce distinct
    non-CVT terminal entities, so the evidence remains representative without
    exploding into full pattern-level expansion.
    """
    node_edges = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        node_edges.setdefault(h, []).append((h, r, t))
        node_edges.setdefault(t, []).append((h, r, t))

    anchor_name = ents[anchor_idx] if 0 <= anchor_idx < len(ents) else ""

    # CASE-LEVEL STATIC MEMOS (user proposal 2026-09-10: CVT entities expand
    # ONCE per case; displays adjust on demand): the per-CVT scored edge
    # lists and the (h,r,t) static filter results depend only on the case
    # arrays — the SAME hub's CVTs were re-expanded on EVERY sg call of the
    # case. Keyed by the pinned arrays (walk-perf _RPE_ADJ_CACHE pattern:
    # values keep strong refs so id() keys cannot collide while live).
    _ck = (id(ents), id(rels_list), len(ents))
    _cm = _CASE_STATIC_MEMO.get(_ck)
    if _cm is None:
        if len(_CASE_STATIC_MEMO) > 8:
            _CASE_STATIC_MEMO.clear()
        _cm = _CASE_STATIC_MEMO[_ck] = {
            "pin": (ents, rels_list),
            "endpoint": {},       # cvt_idx -> static scored edge list
            "add": {},            # (h,r,t) -> static tuple | None
        }
    _cvt_endpoint_memo = _cm["endpoint"]
    _add_static_store = _cm["add"]
    _attr_disp_store = _cm.setdefault("attr_display", {})

    # schema/meta relation CLASS filter (single choke point for ALL evidence
    # sources: support paths, CVT expansion, Option-B leaf enumeration). The
    # walk-level blacklist (k_queue) already blocks traversal, but Option-B's
    # incident-edge enumeration reads raw node_edges and leaked ontology edges
    # (type.type.*, freebase.type_profile.*, notable_types...) — these connect
    # type-label hub nodes ('Film character' deg-23) whose edges then flood the
    # display and mislead the answer turn. Class-based: relation prefixes,
    # never entity names.
    _SCHEMA_REL_PREFIXES = ("type.", "common.", "freebase.", "kg.", "user.",
                            "base.ontologies.", "owl#", "rdf-schema#")

    def _is_schema_rel(rel_name: str) -> bool:
        r = (rel_name or "").strip().lower()
        if r.startswith(_SCHEMA_REL_PREFIXES):
            return True
        short = r.rsplit(".", 1)[-1]
        return short in ("type", "types", "instance", "instances",
                         "notable_types", "expected_type", "domain",
                         "properties", "included_types", "kind", "inverseof")

    # PATTERN-PATH DISCIPLINE (SAPS original semantics, restored): an edge enters
    # evidence ⟺ its relation is in the model-SELECTED set, or it is a CVT role
    # attribute auto-revealed from a CVT on a selected path. Non-selected
    # relations traversed en route (support-path intermediates, named-leaf
    # extensions like Denver--portrayed-->Film off any center path) are NOT
    # kept — the old pattern-first rendering never kept them either.
    _sel_ids = set(selected_rel_ids) if selected_rel_ids is not None else None
    _sib_selected_memo = {}         # build-local: depends on THIS call's _sel_ids

    def _make_adder(triples_list, seen_set):
        _add_static_memo = _add_static_store

        def _add(h_idx, r_idx, t_idx, cvt_attr=False, path_edge=False):
            _sk = (h_idx, r_idx, t_idx)
            _sv = _add_static_memo.get(_sk, 0)
            if _sv == 0:
                h_name = ents[h_idx] if 0 <= h_idx < len(ents) else "?"
                t_name = ents[t_idx] if 0 <= t_idx < len(ents) else "?"
                r_text = rel_to_text(rels_list[r_idx]) if 0 <= r_idx < len(rels_list) else "?"
                if _is_schema_rel(r_text):
                    _sv = _add_static_memo[_sk] = None
                elif (not _is_latinish(h_name) or not _is_latinish(t_name)
                        or len(normalize(h_name)) < 2
                        or len(normalize(t_name)) < 2):
                    _sv = _add_static_memo[_sk] = None
                else:
                    _sv = _add_static_memo[_sk] = (h_name, t_name, r_text,
                                                   (normalize(h_name),
                                                    normalize(r_text),
                                                    normalize(t_name)))
            if _sv is None:
                return
            if (_sel_ids is not None and r_idx not in _sel_ids
                    and not cvt_attr and not path_edge):
                return
            sig = _sv[3]
            if sig not in seen_set:
                seen_set.add(sig)
                triples_list.append((_sv[0], _sv[2], _sv[1]))
        # PER-ADDER (per-pattern) family dedup carrier: _expand_sibling_cvts
        # reads it — acceptance is static, so within ONE adder a family's
        # later touches append nothing (first-touch order is the result).
        _add._fam_done = set()
        return _add

    def _meta_rel_priority(rel_name):
        rel_name = rel_name or ""
        short = rel_name.rsplit(".", 1)[-1]
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
        }
        return rel_name.startswith(noisy_prefixes) or short in noisy_shorts

    def _expand_endpoint_cvt(cvt_idx, add, witness_nodes):
        # STATIC SCORED-EDGE MEMO (perf 2026-09-10): 100k calls per hub walk
        # re-scored the SAME CVT's edges (is_cvt_like/_meta_rel_priority/
        # _is_latinish/normalize/rel_to_text per edge per call). The static
        # score tail is deterministic per (cvt, edge); only the witness flag
        # varies per call. Same order, same add() sequence — pure caching.
        scored_static = _cvt_endpoint_memo.get(cvt_idx)
        if scored_static is None:
            scored_static = []
            for h_idx, r_idx, t_idx in node_edges.get(cvt_idx, []):
                other_idx = t_idx if h_idx == cvt_idx else h_idx
                other_name = ents[other_idx] if 0 <= other_idx < len(ents) else ""
                if not other_name or is_cvt_like(other_name):
                    continue
                rel_name = rels_list[r_idx] if 0 <= r_idx < len(rels_list) else ""
                scored_static.append((
                    (1 if _meta_rel_priority(rel_name) else 0,
                     0 if _is_latinish(other_name) else 1,
                     len(normalize(other_name)) < 2,
                     rel_to_text(rel_name) if rel_name else "",
                     other_name),
                    h_idx, r_idx, t_idx, other_idx))
            _cvt_endpoint_memo[cvt_idx] = scored_static
        witness_node_set = set(witness_nodes)
        scored = [((0 if other in witness_node_set else 1,) + tail,
                   h_idx, r_idx, t_idx)
                  for tail, h_idx, r_idx, t_idx, other in scored_static]
        for _, h_idx, r_idx, t_idx in sorted(scored):
            add(h_idx, r_idx, t_idx, cvt_attr=True)

    def _expand_sibling_cvts(path_nodes, path_rels, add):
        """For each CVT in the path, find and expand ALL sibling CVTs reachable
        via the same (head_entity, relation) pair.

        Key: must add the parent edge (head, rel, sibling_cvt) BEFORE expanding
        the sibling CVT's attributes, so the formatter sees it as pattern evidence.
        PER-ADDER FAMILY DEDUP (2026-09-10, user decomposition):: _add's
        acceptance is fully STATIC (schema/latinish/normalize/sel_ids — the
        witness flag orders but never admits/rejects), so a family's
        enterable-sig set is identical for every path; seen_set then makes
        every later path's expansion a pure no-op check. The old per-CALL
        expanded_rels set re-did the whole family walk per path (7221 calls
        on the hub specimen); hoisting it to build scope skips the repeats
        byte-identically (first-touch insertion order is preserved — exactly
        what the old code produced, since later calls appended nothing).
        """
        for i, node_idx in enumerate(path_nodes):
            if i == 0:
                continue
            node_name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if not is_cvt_like(node_name):
                continue
            prev_idx = path_nodes[i - 1]
            rel_idx = path_rels[i - 1] if i - 1 < len(path_rels) else None
            if rel_idx is None:
                continue
            sig = (prev_idx, rel_idx)
            _fam_done = getattr(add, "_fam_done", None)
            if _fam_done is None:
                _fam_done = add._fam_done = set()
            if sig in _fam_done:
                continue
            _fam_done.add(sig)
            for edge_h, edge_r, edge_t in node_edges.get(prev_idx, []):
                if edge_h != prev_idx or edge_r != rel_idx:
                    continue
                t_name = ents[edge_t] if 0 <= edge_t < len(ents) else ""
                if not is_cvt_like(t_name):
                    continue
                if edge_t != node_idx:
                    # sibling path keeps the pattern shape: the sibling CVT must
                    # itself touch a selected relation (its last hop rides one,
                    # like the path CVT it siblings). Only then may the —
                    # possibly unselected — parent edge enter evidence, via the
                    # cvt_attr channel (pattern-path discipline, user ruling).
                    _selhit = _sib_selected_memo.get(edge_t)
                    if _selhit is None:
                        _selhit = _sel_ids is None or any(
                            e[1] in _sel_ids for e in node_edges.get(edge_t, []))
                        _sib_selected_memo[edge_t] = _selhit
                    if _selhit:
                        # Add parent edge first so formatter recognizes sibling
                        add(prev_idx, rel_idx, edge_t, cvt_attr=True)
                        _expand_endpoint_cvt(edge_t, add, path_nodes)

    def _path_sig(path):
        return (tuple(path.get("nodes", [])), tuple(path.get("relations", [])))

    def _true_edge(u, r_idx, v):
        """Resolve the STORED orientation of a path hop (u, r, v): the undirected
        walk records hops in path order, so a reverse-traversed edge would display
        a semantically backwards triple ('mascot --team_mascot--> team' — Bernie
        Brewer specimen, 2026-08-19). Return the graph's actual (h, r, t); prefer
        the forward orientation when both exist."""
        for h_, r_, t_ in node_edges.get(u, ()):
            if r_ == r_idx and t_ == v:
                return h_, r_, t_
        for h_, r_, t_ in node_edges.get(v, ()):
            if r_ == r_idx and t_ == u:
                return h_, r_, t_
        return u, r_idx, v

    def _support_diversity_key(path):
        nodes = path.get("nodes", [])
        non_cvt_nodes = []
        cvt_nodes = []
        for node_idx in nodes:
            if 0 <= node_idx < len(ents):
                name = ents[node_idx]
                if is_cvt_like(name):
                    cvt_nodes.append(normalize(name))
                else:
                    non_cvt_nodes.append(normalize(name))
        return (tuple(non_cvt_nodes), tuple(cvt_nodes))

    def _select_support_paths(lp, max_paths=10000):
        # SUPPORT CAP REMOVED (user ruling 2026-09-13): a fixed pattern path
        # has no branching — instantiation keeps every chain satisfying the
        # WHOLE path; fan-out caps must not cut pattern-level completions
        witness = lp.get("best_raw_path")
        if not witness:
            return []

        support = [witness]
        seen_path_sigs = {_path_sig(witness)}
        seen_diversity = {_support_diversity_key(witness)}

        for rp in lp.get("raw_paths", []):
            sig = _path_sig(rp)
            if sig in seen_path_sigs:
                continue
            diversity_key = _support_diversity_key(rp)
            if diversity_key in seen_diversity:
                continue
            support.append(rp)
            seen_path_sigs.add(sig)
            seen_diversity.add(diversity_key)
            if len(support) >= max_paths:
                break
        return support

    result = {}

    for pat_idx, lp in enumerate(selected_patterns):
        label = f"P{pat_idx + 1}"
        witness = lp.get("best_raw_path")
        if not witness:
            continue

        pat_triples = []
        pat_seen = set()
        add = _make_adder(pat_triples, pat_seen)

        support_paths = _select_support_paths(lp)
        if _sel_ids is not None:
            # PATTERN-PATH ADJUDICATION (user rulings 2026-08-17/19): last hop
            # rides a selected relation, and SHORTEST-FIRST — if the CENTER
            # already has direct selected-relation edges, longer detour paths
            # through a NAMED mid node ('Nordic <--contains_major_portion_of--
            # Northern Europe --contains--> member' while 'Nordic --contains-->
            # member' exists) are NOT pattern paths — the detour edges must
            # not enter evidence. Without a direct selected edge (Düsseldorf
            # --kind_of→: no 1-hop kind_of) the 2-hop detour IS the only
            # route and stays. CVT mids are transparent either way.
            # PER-RELATION (Greeley fixture, 2026-08-19): the gate is keyed to
            # the path's LAST relation, not the whole selected set. A payload
            # relation with no direct instantiation at the center (school_type
            # from a city center) keeps its named-mid route even when a BRIDGE
            # relation selected in the same call (contains) has direct edges —
            # the set-level gate silently voided every 2-hop-only sibling the
            # GTE pool had guaranteed walkable.
            _direct_sel_rels = {er for _eh, er, _et in node_edges.get(anchor_idx, [])
                                if er in _sel_ids}

            def _hop_ok(sp):
                rels_, nodes_ = sp.get("relations", []), sp.get("nodes", [])
                if not rels_ or rels_[-1] not in _sel_ids:
                    return False
                if rels_[-1] not in _direct_sel_rels:
                    return True
                # SAME-RELATION chain detour (Nordics guard, 2026-08-19): a
                # chain whose mid hop REPEATS the last-hop relation (over a
                # NAMED node) while that relation has a direct instantiation
                # at the center is a back-edge detour — 'Nordic --contains-->
                # Scandinavia --contains--> member' when 'Nordic --contains-->
                # member' exists. CVT mids are transparent (a CVT same-rel
                # passage is the education-family bridge, not a detour).
                for j, r_ in enumerate(rels_[:-1]):
                    if (r_ == rels_[-1]
                            and 0 <= nodes_[j + 1] < len(ents)
                            and not is_cvt_like(ents[nodes_[j + 1]])):
                        return False
                for j, r_ in enumerate(rels_[:-1]):
                    if r_ in _sel_ids:
                        continue
                    nxt = ents[nodes_[j + 1]] if 0 <= nodes_[j + 1] < len(ents) else ""
                    if not is_cvt_like(nxt):
                        return False
                return True
            support_paths = [sp for sp in support_paths if _hop_ok(sp)]
            # CVT BRIDGE (Ethiopia reachability under the hop gate): the walk
            # expands only step_relations from the center, so when the model
            # selects the ROLE direction (office_holder) instead of the CONNEC-
            # TOR (governing_officials), the center's CVTs are unreachable
            # except by beam luck. Construct the bridges explicitly — center
            # --any-rel--> CVT --selected--> target — the exact unselected-hop
            # shape the gate admits. No named-entity detours can arise here
            # (the mid node IS a CVT by construction).
            for _eh, _er, _et in node_edges.get(anchor_idx, []):
                _cvt = _et if _eh == anchor_idx else _eh
                _nm = ents[_cvt] if 0 <= _cvt < len(ents) else ""
                if not (_nm and is_cvt_like(_nm)):
                    continue
                for _e2h, _e2r, _e2t in node_edges.get(_cvt, []):
                    if _e2r not in _sel_ids:
                        continue
                    _tgt = _e2t if _e2h == _cvt else _e2h
                    _tn = ents[_tgt] if 0 <= _tgt < len(ents) else ""
                    if _tn and not is_cvt_like(_tn):
                        support_paths.append({"nodes": [anchor_idx, _cvt, _tgt],
                                              "relations": [_er, _e2r]})
        expanded_cvts = set()
        for sp in support_paths:
            sp_nodes = sp.get("nodes", [])
            sp_rels = sp.get("relations", [])
            for i in range(min(len(sp_rels), len(sp_nodes) - 1)):
                _h_, _r_, _t_ = _true_edge(sp_nodes[i], sp_rels[i], sp_nodes[i + 1])
                add(_h_, _r_, _t_, path_edge=True)
            for node_idx in sp_nodes:
                node_name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
                if is_cvt_like(node_name) and node_idx not in expanded_cvts:
                    _expand_endpoint_cvt(node_idx, add, sp_nodes)
                    expanded_cvts.add(node_idx)
            _expand_sibling_cvts(sp_nodes, sp_rels, add)

        # Option B: leaf-set full enumeration. Same-prefix/different-leaf support
        # paths are ONE one-hop pattern with N leaves — the 24-path support cap
        # must bound path-SHAPE variety, never leaf cardinality. Enumerate ALL
        # edges of the witness's last hop from the local graph, BOTH orientations
        # (mirrors _expand_sibling_cvts, already unbounded for CVT siblings).
        # Both orientations is deliberate: (a) the walk is UNDIRECTED, so the
        # witness's stored orientation is an arbitrary ranking artifact — gating
        # on it silently empties the leaf set when the witness happened to be
        # traversed backwards (observed: offpool rejecting the model's answers
        # because the pool missed the displayed leaves); (b) WebQSP's list GT is
        # orientation-loose (books ABOUT Darwin are gold for 'books written by
        # Darwin'). Added in path direction; the renderer's canonicalizer aligns
        # raw direction downstream.
        w_nodes = witness.get("nodes", [])
        w_rels = witness.get("relations", [])
        if (len(w_nodes) >= 2 and len(w_rels) >= len(w_nodes) - 1
                and (_sel_ids is None or w_rels[-1] in _sel_ids)
                and (_sel_ids is None or _hop_ok(witness))):
            # witness channel too: the fan-out anchor w_nodes[-2] must be on a
            # pattern path (same shortest-first adjudication) — otherwise
            # detours re-enter through enumeration.
            _pen, _lrel = w_nodes[-2], w_rels[-1]
            # BOTH directions (user ruling, 2026-08-19): edge orientation does
            # not matter — `root --r1--> node1 <--r2-- node2` is as legal as
            # the forward shape. The ONLY constraint is the LAST hop rides the
            # selected relation (already enforced above). Reverse gold hops
            # ('institution --students_graduates--> person' queried from the
            # person center) MUST enumerate — an out-edge-only variant emptied
            # those subgraphs entirely.
            # ANCHOR (Bernie Brewer specimen, 2026-08-19): the fan-out node is
            # the TRUE HEAD of the last-hop edge, not the path's [-2] node — a
            # reverse-traversed final hop (center <--team_mascot-- team) left
            # the anchor on the center (a mascot has no team_mascot out-edges)
            # and the leaf enumeration silently yielded ONE witness instead of
            # the full tail set.
            _pen = _true_edge(w_nodes[-2], _lrel, w_nodes[-1])[0]
            for _eh, _er, _et in node_edges.get(_pen, []):
                if _er != _lrel:
                    continue
                _other = _et if _eh == _pen else _eh
                add(_pen, _lrel, _other)

        # CVT auto-penetration POST-PASS REMOVED (user ruling, 2026-08-17):
        # evidence lives on pattern paths whose LAST hop rides a selected
        # relation (support paths, sibling CVTs, Option-B leaves) — nothing is
        # disclosed from mere adjacency of the center or of entities that
        # happened to surface. The adjacency blowout this pass caused (China:
        # 13k edges, job records under a trade query) violated the original
        # pattern-path discipline; the chained-path support filter above now
        # covers the Ethiopia reachability case this pass was invented for.

        witness_nodes = witness.get("nodes", [])

        # candidates = lp candidates ∪ non-CVT entities of the (now fully
        # enumerated) triples — the pool must cover every leaf or the offpool
        # answer check rejects legitimately-enumerated entities. No [:50] cap:
        # list-type patterns legitimately carry hundreds of leaves (display is
        # capped downstream by the renderer).
        _lp_cands = list(lp.get("candidates", []))
        _tri_cands = [n for tr_ in pat_triples for n in (tr_[0], tr_[2])
                      if not is_cvt_like(n)]
        _seen_c = set()
        cand_list = []
        for _n in _lp_cands + _tri_cands:
            _k = normalize(_n)
            if _k and _k not in _seen_c:
                _seen_c.add(_k)
                cand_list.append(_n)

        # GHOST-CANDIDATE EDGE GUARANTEE (user ruling 2026-09-05, Disney
        # specimen): lp["candidates"] covers the walk's FULL leaf enumeration,
        # while pat_triples carries only the bounded support-path set (≤24
        # shapes + CVT expansion) and lp["raw_paths"] is itself capped — a
        # candidate whose paths lost every cut used to surface nameless (the
        # ack said "candidates not shown above: X (walk candidates — no
        # visible edge this case)"), severing the render from the actual
        # connectivity the walk found. Every candidate the render NAMES must
        # now carry ≥1 visible connecting edge: reconstruct anchor→candidate
        # through the LOCAL graph along the lp's own rel_chain (the chain the
        # walk used to reach it) and admit the hops via the path_edge channel.
        # Bounded ghosts per pattern; the renderer caps rows downstream.
        _ghost_paths = []
        if (os.environ.get("SEQ_GHOST_EDGES", "1") != "0" and _lp_cands):
            _have = {normalize(n) for tr_ in pat_triples
                     for n in (tr_[0], tr_[2])}
            _want = [c for c in cand_list if normalize(c) not in _have]
            if _want:
                _rel_id_by_name = {}
                for _ri, _rn in enumerate(rels_list):
                    _rel_id_by_name.setdefault(_rn, _ri)
                _chain_rids = set()
                for _rc in (lp.get("rel_chain") or []):
                    for _part in str(_rc).split(" -> "):
                        _rid = _rel_id_by_name.get(_part.strip())
                        if _rid is not None:
                            _chain_rids.add(_rid)
                if _chain_rids:
                    _ent_idx_by_name = {}
                    for _ei, _en in enumerate(ents):
                        if not is_cvt_like(_en):
                            _ent_idx_by_name.setdefault(normalize(_en), _ei)
                    _n_ghost = 0
                    if os.environ.get("SEQ_GHOST_DEBUG"):
                        print(f"[ghost] {label}: missing={len(_want)} "
                              f"chain_rids={len(_chain_rids)}", flush=True)
                    for _c in _want[:60]:
                        _tidx = _ent_idx_by_name.get(normalize(_c))
                        if _tidx is None or _tidx == anchor_idx:
                            continue
                        _prev = {anchor_idx: None}
                        _q = [(anchor_idx, 0)]
                        _found = False
                        while _q and not _found:
                            _u, _d = _q.pop(0)
                            if _d >= 4:
                                continue
                            for _eh, _er, _et in node_edges.get(_u, ()):
                                if _er not in _chain_rids:
                                    continue
                                _v = _et if _eh == _u else _eh
                                if _v in _prev:
                                    continue
                                _prev[_v] = (_eh, _er, _et)
                                if _v == _tidx:
                                    _found = True
                                    break
                                _q.append((_v, _d + 1))
                        if _found:
                            _seq_nodes = [_tidx]
                            _seq_rels = []
                            _node = _tidx
                            while _prev[_node] is not None:
                                _eh, _er, _et = _prev[_node]
                                _hops_edge = (_eh, _er, _et)
                                _seq_rels.append(_er)
                                _node = _eh if _et == _node else _et
                                _seq_nodes.append(_node)
                            _seq_nodes.reverse()
                            _seq_rels.reverse()
                            # admit every hop of the reconstructed chain
                            for _k in range(len(_seq_rels)):
                                _eh2 = _seq_nodes[_k]
                                _er2 = _seq_rels[_k]
                                _et2 = _seq_nodes[_k + 1]
                                _te = _true_edge(_eh2, _er2, _et2)
                                add(_te[0], _te[1], _te[2], path_edge=True)
                            _ghost_paths.append((list(_seq_nodes),
                                                 list(_seq_rels)))
                            _n_ghost += 1
                    if os.environ.get("SEQ_GHOST_DEBUG"):
                        print(f"[ghost] {label}: reconnected {_n_ghost}",
                              flush=True)

        sig_nodes = []
        for node_idx in witness_nodes:
            if 0 <= node_idx < len(ents):
                name = ents[node_idx]
                if not is_cvt_like(name):
                    sig_nodes.append((node_idx, name))

        step_ent_names = []
        if sig_nodes:
            step_ent_names = [[name] for _, name in sig_nodes]
        elif anchor_name:
            step_ent_names = [[anchor_name]]

        step_edge_names = []
        if len(sig_nodes) >= 2:
            for i in range(len(sig_nodes) - 1):
                from_name = sig_nodes[i][1]
                to_name = sig_nodes[i + 1][1]
                step_edge_names.append({from_name: {to_name}})

        # Resolve CVT attributes for inline display (dedup forward/reverse)
        cvt_attrs = {}
        cvt_to_named = {}  # CVT -> named entity mapping
        _cvt_fwd = {}  # cvt_name -> set of forward entity names
        _cvt_inv = {}  # cvt_name -> [(attr_name, t_name)] pending inv attrs
        for h_name, r_text, t_name in pat_triples:
            if is_cvt_like(h_name):
                is_inv = r_text.endswith('.inv')
                attr_name = r_text.rsplit('.', 1)[-1] if '.' in r_text else r_text
                fwd = _cvt_fwd.setdefault(h_name, set())
                if is_inv:
                    _cvt_inv.setdefault(h_name, []).append((attr_name, t_name))
                else:
                    fwd.add(normalize(t_name))
                    cvt_attrs.setdefault(h_name, []).append(f"{attr_name}={t_name}")
            if is_cvt_like(t_name) and not is_cvt_like(h_name):
                cvt_to_named[t_name] = h_name
        # Merge inv attrs not covered by forward
        for cvt_name, inv_list in _cvt_inv.items():
            fwd = _cvt_fwd.get(cvt_name, set())
            for attr_name, t_name in inv_list:
                n = normalize(t_name)
                if n not in fwd:
                    fwd.add(n)
                    cvt_attrs.setdefault(cvt_name, []).append(f"{attr_name}={t_name}")

        if not pat_triples:
            continue

        def _cvt_attr_display(cvt_idx, limit=20):
            # CASE-LEVEL DISPLAY MEMO (perf 2026-09-10 residual profile:
            # 104k calls = top tottime): pure per (cvt_idx, case arrays) —
            # the same CVT's attr list was rebuilt per pattern per path.
            # limit is constant at the single call site. Serves the first
            # build's insertion order; pickle-sha drift vs uncached runs is
            # insertion-order permutation only (content equality holds —
            # see the 2026-09-10 methodology note in SESSION_MEMORY).
            _hit = _attr_disp_store.get(cvt_idx)
            if _hit is not None:
                return _hit
            # Collect forward and inv attrs separately, then merge.
            # Ensures dedup regardless of edge iteration order.
            forward_items = []
            inv_items = []
            forward_seen = set()
            for h_idx, r_idx, t_idx in node_edges.get(cvt_idx, []):
                other_idx = t_idx if h_idx == cvt_idx else h_idx
                other_name = ents[other_idx] if 0 <= other_idx < len(ents) else ""
                if not other_name or is_cvt_like(other_name):
                    continue
                rel_name = rels_list[r_idx] if 0 <= r_idx < len(rels_list) else ""
                short = rel_to_text(rel_name).rsplit(".", 1)[-1] if rel_name else "?"
                is_inv = (t_idx == cvt_idx and h_idx != cvt_idx)
                n_other = normalize(other_name)
                if is_inv:
                    inv_items.append((f"{short}.inv", other_name, rel_name))
                else:
                    forward_seen.add(n_other)
                    forward_items.append((short, other_name, rel_name))

            # Add inv attrs only for entities not already shown as forward
            for item in inv_items:
                n = normalize(item[1])
                if n not in forward_seen:
                    forward_items.append(item)
                    forward_seen.add(n)

            prefix_groups = {}
            for short, val, rel_name in forward_items:
                parts = rel_name.rsplit(".", 1) if rel_name else ["?"]
                prefix = parts[0] if len(parts) > 1 else ""
                prefix_groups.setdefault(prefix, []).append((short, val))

            attrs = []
            for prefix, items in prefix_groups.items():
                for short, val in items:
                    attrs.append(f"{short}={val}")
            attrs = attrs[:limit]
            _attr_disp_store[cvt_idx] = attrs
            return attrs

        _node_disp_memo = {}   # per-pattern (cvt_attrs is fixed within it)

        def _node_display(node_idx, expand_full=False):
            _nd = _node_disp_memo.get(node_idx)
            if _nd is not None:
                return _nd
            name = ents[node_idx] if 0 <= node_idx < len(ents) else "?"
            if is_cvt_like(name):
                # CVT event node: show its attrs INLINE on the node — use the
                # FULL graph attrs (not just the narrow witness-path ones) so
                # every CVT node carries its attrs (e.g. adjoins=Montana).
                path_attrs = [a for a in cvt_attrs.get(name, [])
                              if not _is_noisy_cvt_attr_short(a)]
                graph_attrs = [a for a in _cvt_attr_display(node_idx)
                               if not _is_noisy_cvt_attr_short(a)]
                seen_a = set(path_attrs)
                merged = list(path_attrs)
                for a in graph_attrs:
                    if a not in seen_a:
                        merged.append(a)
                        seen_a.add(a)
                out = (f"{name}: [" + ", ".join(merged[:20]) + "]"
                       if merged else f"{name}: []")
            else:
                out = name
            _node_disp_memo[node_idx] = out
            return out

        tree_paths = []
        tree_seen = set()
        for sp in support_paths:
            sp_nodes = sp.get("nodes", [])
            sp_rels = sp.get("relations", [])
            if not sp_nodes or not sp_rels:
                continue
            n = len(sp_nodes)
            display_nodes = [_node_display(idx, expand_full=(pos >= n - 2)) for pos, idx in enumerate(sp_nodes)]
            display_rels = [
                rel_to_text(rels_list[rel_idx]) if 0 <= rel_idx < len(rels_list) else "?"
                for rel_idx in sp_rels[: max(0, len(display_nodes) - 1)]
            ]
            sig = (tuple(display_nodes), tuple(display_rels))
            if sig not in tree_seen:
                tree_seen.add(sig)
                tree_paths.append({"nodes": display_nodes, "relations": display_rels})
            # Mirror _expand_sibling_cvts: emit sibling-CVT display paths so the
            # tree shows disambiguating CVTs (e.g. a performance CVT and its
            # character_note) that the graph-side enrichment adds to triples but
            # which are absent from the raw witness-path nodes. Without this the
            # rendered tree shows only the shallow witness CVT and hides the
            # branch carrying the answer. Uses display_nodes[i-1] as the parent
            # so _render_path_tree nests the sibling under the same parent edge.
            sib_pair_seen = set()
            for i in range(1, len(sp_nodes)):
                cvt_idx = sp_nodes[i]
                cvt_name = ents[cvt_idx] if 0 <= cvt_idx < len(ents) else ""
                if not is_cvt_like(cvt_name):
                    continue
                prev_idx = sp_nodes[i - 1]
                rel_idx = sp_rels[i - 1] if i - 1 < len(sp_rels) else None
                if rel_idx is None:
                    continue
                pair = (prev_idx, rel_idx)
                if pair in sib_pair_seen:
                    continue
                sib_pair_seen.add(pair)
                rel_disp = rel_to_text(rels_list[rel_idx]) if 0 <= rel_idx < len(rels_list) else "?"
                prev_disp = display_nodes[i - 1]
                for edge_h, edge_r, edge_t in node_edges.get(prev_idx, []):
                    if edge_h != prev_idx or edge_r != rel_idx or edge_t == cvt_idx:
                        continue
                    t_name = ents[edge_t] if 0 <= edge_t < len(ents) else ""
                    if not is_cvt_like(t_name):
                        continue
                    sib_disp = _node_display(edge_t, expand_full=True)
                    s_nodes = [prev_disp, sib_disp]
                    s_rels = [rel_disp]
                    s_sig = (tuple(s_nodes), tuple(s_rels))
                    if s_sig in tree_seen:
                        continue
                    tree_seen.add(s_sig)
                    tree_paths.append({"nodes": s_nodes, "relations": s_rels})

        # GHOST-CANDIDATE paths also enter the tree (V38 renders from
        # tree_data.paths, not .triples): without this the reconnected edges
        # exist in triples only and the display still shows the candidate
        # nameless.
        for _g_nodes, _g_rels in _ghost_paths:
            _dn = [_node_display(_gi, expand_full=(_gp >= len(_g_nodes) - 2))
                   for _gp, _gi in enumerate(_g_nodes)]
            _dr = [rel_to_text(rels_list[_gr]) if 0 <= _gr < len(rels_list) else "?"
                   for _gr in _g_rels]
            _gsig = (tuple(_dn), tuple(_dr))
            if _gsig not in tree_seen:
                tree_seen.add(_gsig)
                tree_paths.append({"nodes": _dn, "relations": _dr})

        result[label] = PatternEvidence(
            label=label,
            readable=lp.get("readable", ""),
            candidates=cand_list,
            triples=pat_triples,
            tree_data={"paths": tree_paths},
        )

    return result


# ---------------------------------------------------------------------------
# Path tree renderer
# ---------------------------------------------------------------------------

def _render_path_tree(tree_data, max_lines=50, max_children=50, constraint_entities=None):
    """Render support raw paths as a compact YAML-like trie.

    This keeps the relation pattern aligned with the actual materialized paths:
    each child is nested under the parent entity/CVT that really leads to it.

    constraint_entities: list of entity names that should be shown first (sorted priority).
    """
    if not isinstance(tree_data, dict):
        return []
    paths = tree_data.get("paths") or []
    if not paths:
        return []

    root = {"name": None, "edges": {}}
    for path in paths:
        nodes = path.get("nodes") or []
        rels = path.get("relations") or []
        if not nodes:
            continue
        if root["name"] is None:
            root["name"] = nodes[0]
        cur = root
        for i, rel_name in enumerate(rels):
            if i + 1 >= len(nodes):
                break
            child_name = nodes[i + 1]
            rel_children = cur["edges"].setdefault(rel_name, {})
            cur = rel_children.setdefault(child_name, {"name": child_name, "edges": {}})

    if root["name"] is None:
        return []

    lines = [f"node0: {root['name']}"]

    def _is_cvt_display(name):
        return (
            isinstance(name, str)
            and (name.startswith("m.") or name.startswith("g.") or name.startswith("CVT:"))
        )

    def _parse_cvt_display(name):
        if not _is_cvt_display(name):
            return name, []
        if ": [" not in name or not name.endswith("]"):
            return name, []
        cvt_id, attrs_raw = name.split(": [", 1)
        attrs_raw = attrs_raw[:-1]
        if not attrs_raw:
            return cvt_id, []
        return cvt_id, [a.strip() for a in attrs_raw.split(", ") if a.strip()]

    def _extract_inv_entities(cvt_display_name):
        """Extract .inv attribute values, skipping entities already shown as forward attrs."""
        _, attrs = _parse_cvt_display(cvt_display_name)
        forward_entities = set()
        for attr in attrs:
            if ".inv=" not in attr and "=" in attr:
                _, val = attr.split("=", 1)
                forward_entities.add(normalize(val.strip()))
        inv_entities = []
        for attr in attrs:
            if ".inv=" in attr:
                rel_short, entity = attr.split(".inv=", 1)
                if normalize(entity.strip()) not in forward_entities:
                    inv_entities.append((rel_short, entity.strip()))
        return inv_entities

    def _format_cvt_display(cvt_id, attrs):
        return f"{cvt_id}: [" + ", ".join(attrs) + "]"

    def _compress_cvt_displays(child_names):
        if len(child_names) < 2 or not all(_is_cvt_display(name) for name in child_names):
            return [], child_names
        parsed = [_parse_cvt_display(name) for name in child_names]
        attr_sets = [set(attrs) for _, attrs in parsed]
        shared = set.intersection(*attr_sets) if attr_sets else set()
        shared_attrs = [attr for attr in parsed[0][1] if attr in shared]
        if not shared_attrs:
            return [], child_names
        compressed = []
        for cvt_id, attrs in parsed:
            own_attrs = [attr for attr in attrs if attr not in shared]
            compressed.append(_format_cvt_display(cvt_id, own_attrs))
        return shared_attrs, compressed

    def _has_constraint_in_subtree(node):
        """Check if node or its descendants match any constraint entity."""
        if not constraint_entities:
            return False
        name = node.get("name", "")
        if isinstance(name, str):
            for ce in constraint_entities:
                if ce and normalize(ce) in normalize(name):
                    return True
        for rn, ch in node.get("edges", {}).items():
            for cn, child in ch.items():
                if _has_constraint_in_subtree(child):
                    return True
        return False

    def _constraint_sort_key(name, node=None):
        """Sort key: constraint-hitting branches first, then alphabetical."""
        if not constraint_entities:
            return (0, name)
        # Check direct name match
        if isinstance(name, str):
            for ce in constraint_entities:
                if ce and normalize(ce) in normalize(name):
                    return (-1, name)
        # Check subtree
        if node and _has_constraint_in_subtree(node):
            return (-1, name)
        return (0, name)

    def _fold_cvt_leaf_edges(node):
        if not node.get("edges"):
            return ""
        parts = []
        for rel_name in sorted(node.get("edges", {})):
            children = node["edges"][rel_name]
            if not all(not child.get("edges") for child in children.values()):
                return ""
            child_names = sorted(children, key=lambda n: _constraint_sort_key(n))
            shown = child_names[:max_children]
            suffix = f" | ... (+{len(child_names) - len(shown)})" if len(child_names) > len(shown) else ""
            short_rel = rel_name.rsplit(".", 1)[-1]
            parts.append(f"{short_rel}=[" + " | ".join(shown) + suffix + "]")
        return "; ".join(parts)

    def _render_node(node, depth, indent):
        if len(lines) >= max_lines:
            return
        for rel_name in sorted(node.get("edges", {})):
            if len(lines) >= max_lines:
                return
            children = node["edges"][rel_name]
            lines.append(" " * indent + f"{rel_name}:")

            # Leaf-or-CVT-only check: compress when all children are leaves,
            # OR when all children are CVT-display nodes (their meaningful
            # attributes already live in the display name; any edges they
            # have are reverse back-edges that add no information and would
            # otherwise bypass compression, leaving shared attrs duplicated
            # across every CVT — e.g. basic_title=Prime minister repeated
            # 10× for a list of office holders).
            child_keys = list(children.keys())
            all_cvt = bool(child_keys) and all(_is_cvt_display(n) for n in child_keys)
            if all(not child.get("edges") for child in children.values()) or all_cvt:
                child_names = sorted(children, key=lambda n: _constraint_sort_key(n))
                if any(_is_cvt_display(child_name) for child_name in child_names):
                    shown_names = child_names[:max_children]
                    shared_attrs, shown_names = _compress_cvt_displays(shown_names)
                    if shared_attrs and len(lines) < max_lines:
                        lines.append(
                            " " * (indent + 2)
                            + "shared: [" + ", ".join(shared_attrs) + "]"
                        )
                    # CVT siblings stay one-per-line: each carries differentiating
                    # attributes (from=2014, office_holder=X) that the model must
                    # read to pick the right one. Joining them on one line buries
                    # these attrs in a wall of text. Plain-entity leaves (no CVT
                    # attrs) are joined with "|" on one line in the branch below.
                    for child_name in shown_names:
                        if len(lines) >= max_lines:
                            return
                        lines.append(" " * (indent + 2) + f"- node{depth + 1}: {child_name}")
                        for inv_rel, inv_ent in _extract_inv_entities(child_name):
                            if len(lines) < max_lines:
                                lines.append(" " * (indent + 4) + f"← also {inv_rel}: {inv_ent}")
                    omitted = len(child_names) - len(shown_names)
                    if omitted > 0 and len(lines) < max_lines:
                        lines.append(" " * (indent + 2) + f"- ... (+{omitted})")
                    continue
                shown_names = child_names[:max_children]
                suffix = f" | ... (+{len(child_names) - len(shown_names)})" if len(child_names) > len(shown_names) else ""
                lines.append(
                    " " * (indent + 2)
                    + f"node{depth + 1}: [" + " | ".join(shown_names) + suffix + "]"
                )
                continue

            child_items = sorted(children.items(), key=lambda x: _constraint_sort_key(x[0], x[1]))
            shown_items = child_items[:max_children]
            compressed_by_name = {}
            cvt_names = [name for name, _ in shown_items if _is_cvt_display(name)]
            if cvt_names:
                shared_attrs, compressed_names = _compress_cvt_displays(cvt_names)
                compressed_by_name = dict(zip(cvt_names, compressed_names))
                if shared_attrs and len(lines) < max_lines:
                    lines.append(
                        " " * (indent + 2)
                        + "shared: [" + ", ".join(shared_attrs) + "]"
                    )
            for child_name, child in shown_items:
                if len(lines) >= max_lines:
                    return
                child_display = compressed_by_name.get(child_name, child_name)
                folded = _fold_cvt_leaf_edges(child) if _is_cvt_display(child_name) else ""
                suffix = f" ; {folded}" if folded else ""
                lines.append(" " * (indent + 2) + f"- node{depth + 1}: {child_display}{suffix}")
                for inv_rel, inv_ent in _extract_inv_entities(child_display):
                    if len(lines) < max_lines:
                        lines.append(" " * (indent + 4) + f"← also {inv_rel}: {inv_ent}")
                if not folded:
                    _render_node(child, depth + 1, indent + 4)
            omitted = len(child_items) - len(shown_items)
            if omitted > 0 and len(lines) < max_lines:
                lines.append(" " * (indent + 2) + f"- ... (+{omitted})")

    _render_node(root, 0, 2)
    return lines


# ---------------------------------------------------------------------------
# Entity tree renderer
# ---------------------------------------------------------------------------

def _render_entity_tree(step_ent_names, step_edge_names, cvt_attrs,
                        max_branches=15, max_paths_per_branch=10, cvt_to_named=None):
    """Render entity tree with node position labels."""
    if len(step_ent_names) < 2 or not step_edge_names:
        return []

    if cvt_to_named is None:
        cvt_to_named = {}

    lines = []
    step1_ents = step_ent_names[1][:max_branches]
    num_steps = len(step_edge_names)

    def _resolve(name):
        if is_cvt_like(name) and name in cvt_attrs and cvt_attrs[name]:
            return "[" + ", ".join(cvt_attrs[name][:20]) + "]"
        if is_cvt_like(name) and name in cvt_to_named:
            return cvt_to_named[name]
        return name

    def _enumerate(step, entity, limit):
        if step >= num_steps:
            return [[]]
        children = step_edge_names[step].get(entity, set())
        if not children:
            return [[]]
        result = []
        for child in sorted(children):
            if len(result) >= limit:
                break
            node_label = f"node{step + 1}"
            sub_paths = _enumerate(step + 1, child, limit - len(result))
            for sp in sub_paths:
                result.append([f"{node_label}={_resolve(child)}"] + sp)
        return result

    for s1 in step1_ents:
        paths = _enumerate(1, s1, max_paths_per_branch)
        if not paths or paths == [[]]:
            lines.append(f"  node1={s1}")
        else:
            lines.append(f"  node1={s1}")
            for p in paths:
                if p:
                    lines.append("    " + " → ".join(p))
    return lines


# ---------------------------------------------------------------------------
# Public formatting functions
# ---------------------------------------------------------------------------

def format_pattern_evidence(pattern_evidence, max_total_lines=120, max_per_pattern=50, max_tails=50, constraint_entities=None):
    """Format pattern-grouped evidence as entity tree for LLM reasoning.

    Each pattern shows a tree of entities at each node position, grouped by
    the first-level entity. CVT nodes are resolved to attribute displays.
    """
    lines = []

    for label, pe in pattern_evidence.items():
        if not pe.triples:
            continue

        # Header with readable path
        lines.append(f"=== PATTERN {label}: {pe.readable} ===")

        tree_lines = _render_path_tree(pe.tree_data, max_lines=max_per_pattern, constraint_entities=constraint_entities)
        if tree_lines:
            lines.append("  Evidence tree:")
            lines.extend(f"  {tl}" for tl in tree_lines)
        else:
            cvt_text = format_subgraph_with_cvt(pe.triples, max_lines=max_per_pattern, max_tails=50)
            for tl in cvt_text.split('\n'):
                lines.append(f"  {tl}")

        lines.append("")

    return "\n".join(lines)


def format_subgraph_with_cvt(triples, max_lines=80, max_tails=50):
    """Format triples for LLM reasoning with inline CVT resolution.

    Normal triples: grouped by (head, relation) as before.
    CVT nodes are resolved inline:
      Entity → relation → [attr1=val1, attr2=val2]
    Orphan CVTs (no parent entity) fall back to bare block display.
    """
    def _is_noisy_cvt_attr(rel_name):
        rel_name = rel_name or ""
        short = rel_name.rsplit(".", 1)[-1]
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
            # Freebase valuenotation markers — no semantic content for QA
            "has_no_value",
            "is_reviewed",
            "no_value",
        }
        return rel_name.startswith(noisy_prefixes) or short in noisy_shorts

    def _sort_attrs(attrs):
        return sorted(
            attrs,
            key=lambda x: (
                1 if _is_noisy_cvt_attr(x[0]) else 0,
                1 if is_cvt_like(x[1]) else 0,
                0 if _is_latinish(x[1]) else 1,
                len(normalize(x[1])) < 2,
                x[0],
                x[1],
            ),
        )

    clean = []
    cvt_attrs = {}  # cvt_name -> [(short_rel, value)]
    cvt_parents = {}  # cvt_name -> [(parent_entity, parent_rel)]

    for h, r, t in triples:
        h_cvt = is_cvt_like(h)
        t_cvt = is_cvt_like(t)

        if not h_cvt and not t_cvt:
            clean.append((h, r, t))
        elif h_cvt and t_cvt:
            continue
        elif h_cvt:
            short = r.split(".")[-1]
            cvt_attrs.setdefault(h, []).append((short, t))
        elif t_cvt:
            cvt_parents.setdefault(t, []).append((h, r))

    # Resolve CVT nodes inline: merge parent edge + CVT attributes
    resolved_cvt = set()
    cvt_inline = []  # (parent, rel, attrs_str) for inline display
    for cvt_name, parents in cvt_parents.items():
        attrs = list(cvt_attrs.get(cvt_name, []))
        if len(parents) > 1:
            for extra_parent, extra_rel in parents[1:]:
                attrs.append((extra_rel.split(".")[-1] + ".inv", extra_parent))
        attrs = _sort_attrs(attrs)
        if parents and attrs:
            attr_str = ", ".join(f"{short}={val}" for short, val in attrs[:20])
            parent, rel = parents[0]
            cvt_inline.append((parent, rel, attr_str))
            resolved_cvt.add(cvt_name)
        elif parents:
            parent, rel = parents[0]
            cvt_inline.append((parent, rel, f"CVT:{cvt_name}"))
            resolved_cvt.add(cvt_name)

    lines = []

    # Part 1: Normal triples (existing grouped format)
    if clean:
        grouped_lines = format_grouped_triples(clean, max_lines=max_lines, max_tails=max_tails)
        lines.extend(grouped_lines)

    # Part 2: Inline-resolved CVT edges
    if cvt_inline:
        if lines:
            lines.append("")
        # Group by (parent, rel) for compression
        inline_groups = {}
        for parent, rel, attr_str in cvt_inline:
            inline_groups.setdefault((parent, rel), []).append(attr_str)
        for (parent, rel), attr_strs in inline_groups.items():
            if len(lines) >= max_lines:
                break
            if len(attr_strs) == 1:
                lines.append(f"({parent}, {rel}, [{attr_strs[0]}])")
            else:
                shown = attr_strs[:max_tails]
                suffix = f", ...(+{len(attr_strs) - len(shown)})" if len(attr_strs) > len(shown) else ""
                lines.append(f"({parent}, {rel}, [{']  ['.join(shown)}{suffix}])")

    # Part 3: Orphan CVTs (no parent connection) — bare block display
    orphan_cvts = {k: v for k, v in cvt_attrs.items() if k not in resolved_cvt}
    if orphan_cvts:
        if lines:
            lines.append("")
        for cvt_name, attrs in orphan_cvts.items():
            if len(lines) >= max_lines:
                break
            attrs = _sort_attrs(attrs)
            lines.append(f"[{cvt_name}]")
            for short_rel, val in attrs:
                if len(lines) >= max_lines:
                    break
                lines.append(f"  {short_rel} → {val}")

    return "\n".join(lines)


def format_grouped_triples(triples, max_lines=80, max_tails=50):
    """Group triples by (head, relation) and (relation, tail) to reduce token usage."""
    # Forward grouping: (head, relation) -> list of tails
    fwd_groups = {}
    for h, r, t in triples:
        fwd_groups.setdefault((h, r), []).append(t)

    # Deduplicate forward groups
    fwd_deduped = {}
    for (h, r), tails in fwd_groups.items():
        uniq = []
        seen = set()
        for t in tails:
            nt = normalize(t)
            if nt not in seen:
                seen.add(nt)
                uniq.append(t)
        fwd_deduped[(h, r)] = uniq

    # Identify reverse-compressible groups: (relation, tail) -> list of heads
    # Only when multiple heads share same (relation, tail)
    rev_groups = {}
    for (h, r), tails in fwd_deduped.items():
        for t in tails:
            rev_groups.setdefault((r, normalize(t)), []).append(h)

    # Mark which (head, relation) pairs are absorbed into a reverse group
    reverse_absorbed = set()
    reverse_lines = {}
    for (r, nt), heads in rev_groups.items():
        # Deduplicate heads
        uniq_heads = []
        seen_h = set()
        for h in heads:
            nh = normalize(h)
            if nh not in seen_h:
                seen_h.add(nh)
                uniq_heads.append(h)
        if len(uniq_heads) >= 2:
            # Find the original tail name from the triples
            orig_tail = None
            for (h2, r2), tails2 in fwd_deduped.items():
                for t2 in tails2:
                    if r2 == r and normalize(t2) == nt:
                        orig_tail = t2
                        break
                if orig_tail:
                    break
            if orig_tail and len(uniq_heads) <= max_tails:
                reverse_lines[(r, nt)] = (uniq_heads, r, orig_tail)
                for h in uniq_heads:
                    reverse_absorbed.add((normalize(h), r))

    lines = []
    for (h, r), tails in fwd_deduped.items():
        # Skip if this head is absorbed into a reverse group AND all tails match
        nh = normalize(h)
        if (nh, r) in reverse_absorbed:
            # Check if ALL tails of this (h, r) are absorbed into reverse groups
            all_absorbed = all(
                (r, normalize(t)) in reverse_lines for t in tails
            )
            if all_absorbed:
                continue
            # Partially absorbed — show only non-absorbed tails
            remaining = [t for t in tails if (r, normalize(t)) not in reverse_lines]
            if not remaining:
                continue
            tails = remaining

        if len(tails) == 1:
            lines.append(f"({h}, {r}, {tails[0]})")
        else:
            shown = tails[:max_tails]
            suffix = f", ...(+{len(tails) - len(shown)})" if len(tails) > len(shown) else ""
            lines.append(f"({h}, {r}, [{', '.join(shown)}{suffix}])")
        if len(lines) >= max_lines:
            break

    # Append reverse-compressed lines
    for (r, nt), (heads, rel, tail) in sorted(reverse_lines.items()):
        if len(lines) >= max_lines:
            break
        if len(heads) == 2:
            lines.append(f"([{heads[0]}, {heads[1]}], {rel}, {tail})")
        else:
            shown = heads[:max_tails]
            suffix = f", ...(+{len(heads) - len(shown)})" if len(heads) > len(shown) else ""
            lines.append(f"([{', '.join(shown)}{suffix}], {rel}, {tail})")

    return lines


def collect_local_subgraph_triples(seed_node_indices, ents, rels_list, h_ids, r_ids, t_ids,
                                   max_per_seed=6, global_limit=120):
    """Collect a small 1-hop local subgraph around selected witness/candidate nodes.

    This is for final reasoning only. It exposes local evidence around bridge nodes
    and candidate entities that is not visible from the witness path alone.
    """
    triples = []
    seen = set()

    def _add_triple(h_idx, r_idx, t_idx):
        h_name = ents[h_idx] if 0 <= h_idx < len(ents) else "?"
        t_name = ents[t_idx] if 0 <= t_idx < len(ents) else "?"
        r_text = rel_to_text(rels_list[r_idx]) if 0 <= r_idx < len(rels_list) else "?"
        # Filter out pure non-Latin entities (multilingual noise from Freebase)
        if not re.search(r'[a-zA-Z]', h_name) or not re.search(r'[a-zA-Z]', t_name):
            return
        sig = (normalize(h_name), normalize(r_text), normalize(t_name))
        if sig not in seen:
            seen.add(sig)
            triples.append((h_name, r_text, t_name))

    for node_idx in seed_node_indices:
        count = 0
        if node_idx is None or not (0 <= node_idx < len(ents)):
            continue
        for i in range(len(h_ids)):
            if len(triples) >= global_limit or count >= max_per_seed:
                break
            if h_ids[i] == node_idx or t_ids[i] == node_idx:
                _add_triple(h_ids[i], r_ids[i], t_ids[i])
                count += 1
        if len(triples) >= global_limit:
            break
    return triples


# ---------------------------------------------------------------------------
# Path candidate extraction
# ---------------------------------------------------------------------------

def _extract_path_candidates(paths, anchor_idx, ents, h_ids, r_ids, t_ids):
    """Extract deduplicated candidates from path traversal nodes only (no HR frontier).
    Used for GT recall -- measures whether graph traversal can reach the answer.
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
        if len(nc) < 2 or not _is_latinish(c):
            continue
        if nc not in seen:
            seen.add(nc)
            unique.append(c)
    return unique
