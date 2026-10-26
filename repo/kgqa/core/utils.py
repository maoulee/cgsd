"""Pure utility functions for the KGQA pipeline.

String normalization, relation formatting, XML extraction, entity context
building, candidate matching, and answer-set F1 computation.
"""
from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List


# ---------------------------------------------------------------------------
# Phase timing (perf attribution — where does the ~8s/case go?)
# Accumulators are process-global; the rollout prints them per stage.
# llm = chat_batch wall time; dispatch = turn-processing wall (contains
# walk + render + gte, which are timed separately inside).
# ---------------------------------------------------------------------------
PHASE_TIMES: Dict[str, float] = {
    "llm": 0.0, "dispatch": 0.0, "walk": 0.0, "render": 0.0, "gte": 0.0,
    # 2026-10-24 崩塌取证: seq_tools 的游走计时用裸下标 +=（walk_collect/
    # wait/exec 不在此初始化会 KeyError）。训练路径曾由 [tool] 统计钩子的
    # pop 触发（键被删 → step2 首 walk 即 KeyError: 'walk' → 工具层全错
    # → 空答雪崩）。所有裸写入键必须在此预置; 写入方仍应用 setdefault 防御。
    "walk_collect": 0.0, "walk_wait": 0.0, "walk_exec": 0.0,
    "walk_flush_wall": 0.0, "gte_n": 0.0, "gte_collect": 0.0,
    "gte_post_wall": 0.0,
}


class phase_timer:
    """Context manager accumulating wall time into PHASE_TIMES[key]."""
    __slots__ = ("key", "_t0")

    def __init__(self, key: str):
        self.key = key

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        # 2026-10-24 ruling: instrumentation must NEVER be able to break
        # execution. The bare `+=` raised KeyError when a key was removed
        # (the training [tool] hook once popped keys → every retrieve_subgraph
        # died with KeyError: 'walk'). The pure-inference environment
        # (react_loop/tools) has no timing hooks at all — the seq path's
        # instrumentation must be side-effect-free on any dict state.
        try:
            PHASE_TIMES[self.key] = PHASE_TIMES.get(self.key, 0.0) \
                + time.perf_counter() - self._t0
        except Exception:  # noqa: BLE001
            pass
        return False


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

# PERF-4 (2026-08-23): normalize is called ~6M times per 267×3 replay — the
# same entity/relation names re-normalized at every call site (render keys,
# accumulate, canonicalization, name resolution). It is a pure string →
# string function, so a bounded LRU memo removes ~90% of that work with zero
# behavior change. Bounded (2^18) to keep memory flat on unbounded query
# strings; walk-lane workers build their own process-local cache.
@lru_cache(maxsize=262_144)
def _normalize_impl(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9%.' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    # thin wrapper keeps the cached core small and the call sites unchanged
    # (non-str inputs fail exactly as before, inside the implementation)
    return _normalize_impl(text)


# vLLM reasoning_end_str force-injected at thinking-budget exhaustion. The Qwen3.5
# chat template puts the <think> OPEN tag in the prompt, so vLLM's boundary search
# (which scans the OUTPUT) can miss it — when the budget fires mid-word the phrase
# glues onto the last partial token ("...Engage The CrowdI will now emit...") and
# lands in content, corrupting tool args and checkpoint bindings downstream.
_REASONING_LEAK_RE = re.compile(
    r"I will now emit the tool call based on the reasoning above\.?", re.IGNORECASE)


def strip_reasoning_leak(text: str, keep_think_tags: bool = False) -> str:
    """Remove the vLLM-injected reasoning transition phrase (anywhere, glued or
    standalone) and stray </think> tags. Idempotent; returns text unchanged when
    clean.

    ``keep_think_tags`` is for contexts that must replay the sampled token
    stream exactly (rollout history / training transcript): the transition
    phrase is still removed, but the ``</think>`` boundary is preserved so the
    assistant block re-renders identically."""
    if not text:
        return text
    out = _REASONING_LEAK_RE.sub("", text)
    if not keep_think_tags:
        out = out.replace("</think>", "")
    return out


def rel_to_text(rel: str) -> str:
    """Return original dot-notation format for LLM display: 'people.person.religion'"""
    return rel


def rel_to_text_short(rel: str) -> str:
    """Short format (last 2 segments) for GTE retrieval: 'person religion'"""
    parts = rel.split(".")
    return " ".join(p.replace("_", " ") for p in parts[-2:]) if len(parts) >= 2 else rel.replace("_", " ")


def extract_xml_tag(text, tag):
    """Extract content between <tag>...</tag>, returns None if not found."""
    if not text:
        return None
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# Entity / relation context
# ---------------------------------------------------------------------------

def get_entity_contexts(entity_names, h_ids, r_ids, t_ids, ents, rels):
    """For each candidate entity, extract surrounding relation context from subgraph.
    Similar to check_entities' [Context: ...] annotation."""
    # Import here to avoid circular dependency — is_cvt_like lives in traversal/cvt.py
    from kgqa.traversal.cvt import is_cvt_like

    # Build name->idx lookup
    name_to_indices = {}
    for i, e in enumerate(ents):
        name_to_indices.setdefault(e, []).append(i)

    contexts = {}
    for cand in entity_names:
        indices = name_to_indices.get(cand, [])
        if not indices:
            continue
        idx_set = set(indices)
        outgoing = []
        incoming = []
        for i in range(len(h_ids)):
            if h_ids[i] in idx_set:
                rel_text = rel_to_text(rels[r_ids[i]]) if 0 <= r_ids[i] < len(rels) else "?"
                t_name = ents[t_ids[i]] if 0 <= t_ids[i] < len(ents) else "?"
                if not is_cvt_like(t_name) and t_name != cand:
                    outgoing.append(f"{rel_text}->{t_name}")
            if t_ids[i] in idx_set:
                rel_text = rel_to_text(rels[r_ids[i]]) if 0 <= r_ids[i] < len(rels) else "?"
                h_name = ents[h_ids[i]] if 0 <= h_ids[i] < len(ents) else "?"
                if not is_cvt_like(h_name) and h_name != cand:
                    incoming.append(f"{h_name}->{rel_text}")
        parts = outgoing[:2] + incoming[:1]
        if parts:
            contexts[cand] = "; ".join(parts)
    return contexts


# ---------------------------------------------------------------------------
# Candidate matching
# ---------------------------------------------------------------------------

def _contains_whole(short: str, long_: str) -> bool:
    """Token-boundary containment: `short` appears as whole consecutive
    word(s) inside `long_`. Raw substring scoring is WRONG: gold 'Europe'
    matched 'Central European Summer Time' because 'europe' is a substring
    of the token 'european' — a wrong answer scored F1=1.0 (user-reported
    2026-09-13, GMT/Belgium case). Token boundary keeps legitimate cases
    ('Netherlands' in 'Kingdom of the Netherlands') while rejecting
    morphological accidents ('europe' vs 'european')."""
    return (" " + short.strip() + " ") in (" " + long_.strip() + " ")


def candidate_hit(cands: List[str], targets: List[str]) -> bool:
    # Filter empties AFTER normalize: brackets/punctuation like "[]" or "()" normalize
    # to "" and `"" in target` is True for every string — an empty answer would match
    # everything (f1=1.0). Same guard as the empty-string _name_to_idx fix.
    norm_cands = [normalize(c) for c in cands if c.strip()]
    norm_cands = [c for c in norm_cands if len(c) >= 2]
    if not norm_cands:
        return False
    for t in targets:
        nt = normalize(t)
        if len(nt) < 2:
            continue
        for c in norm_cands:
            # ENTITY ONE-TO-ONE: exact normalized equality only —
            # containment across distinct surface forms miscounts hits
            if c == nt:
                return True
    # Fuzzy fallback: catch near-matches like "Connor" vs "Conner"
    # Only for entities >= 8 chars to avoid false positives on short names.
    # Threshold 0.95 (not 0.92): "2010 World Series" vs "2014 World Series" =
    # 0.94 — year-variant events must NOT collapse to a match.
    for t in targets:
        nt = normalize(t)
        if len(nt) < 8:
            continue
        for c in norm_cands:
            if len(c) < 8:
                continue
            if SequenceMatcher(None, c, nt).ratio() >= 0.95:
                return True
    return False


def strict_candidate_hit(cands: List[str], targets: List[str]) -> bool:
    """Strict matching: exact normalized equality only (entity one-to-one)."""
    norm_cands = [normalize(c) for c in cands]
    for t in targets:
        nt = normalize(t)
        if len(nt) < 2:
            continue
        for c in norm_cands:
            if len(c) < 2:
                continue
            if c == nt:
                return True
    return False


def sample_triple(rel_idx: int, h_ids, r_ids, t_ids, entity_list, relation_list) -> str:
    """Sample one (head, relation, tail) triple for a given relation index."""
    for i in range(len(r_ids)):
        if r_ids[i] == rel_idx:
            h_name = entity_list[h_ids[i]] if 0 <= h_ids[i] < len(entity_list) else "?"
            t_name = entity_list[t_ids[i]] if 0 <= t_ids[i] < len(entity_list) else "?"
            r_name = relation_list[r_ids[i]] if 0 <= r_ids[i] < len(relation_list) else "?"
            return f"({h_name}) -[{r_name}]-> ({t_name})"
    return ""


def sample_triple_batch(rel_indices, h_ids, r_ids, t_ids, entity_list, relation_list):
    """Sample one triple per relation index. Returns dict: rel_idx -> triple_str."""
    result = {}
    seen_rels = set(rel_indices)
    for i in range(len(r_ids)):
        r = r_ids[i]
        if r in seen_rels and r not in result:
            h_name = entity_list[h_ids[i]] if 0 <= h_ids[i] < len(entity_list) else "?"
            t_name = entity_list[t_ids[i]] if 0 <= t_ids[i] < len(entity_list) else "?"
            r_name = relation_list[r] if 0 <= r < len(relation_list) else "?"
            result[r] = f"({h_name}) -[{r_name}]-> ({t_name})"
            if len(result) == len(seen_rels):
                break
    return result


def compute_match_stats(predicted: List[str], gold: List[str]) -> Dict[str, float]:
    """Compute P/R/F1 between predicted and gold answer sets.

    Returns: {'precision': float, 'recall': float, 'f1': float,
              'matched_gold': int, 'matched_pred': int, 'n_gold': int, 'n_pred': int}
    Uses the same matching logic as candidate_hit (normalize + substring + fuzzy).
    """
    if not predicted or not gold:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'matched_gold': 0, 'matched_pred': 0, 'n_gold': len(gold), 'n_pred': len(predicted)}

    norm_pred = [normalize(c) for c in predicted if c.strip()]
    norm_gold = [normalize(t) for t in gold if t.strip()]
    # Drop empty/short normalizations: "[]"/"()" normalize to "" and `"" in target`
    # is True for every string, which would score an empty answer as f1=1.0.
    norm_pred = [c for c in norm_pred if len(c) >= 2]
    norm_gold = [t for t in norm_gold if len(t) >= 2]
    if not norm_pred or not norm_gold:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'matched_gold': 0, 'matched_pred': 0, 'n_gold': len(gold), 'n_pred': len(predicted)}

    def _matches(c: str, t: str) -> bool:
        # ENTITY ONE-TO-ONE (user ruling 2026-09-13): gold and prediction
        # are graph ENTITIES — 'Western Europe' vs 'Europe' are distinct
        # nodes (both live as separate entities across the corpus), so
        # token containment across different surface forms is a wrong
        # match. Exact normalized equality + typo-tolerant fuzzy only.
        if c == t:
            return True
        if len(c) >= 8 and len(t) >= 8 and SequenceMatcher(None, c, t).ratio() >= 0.95:
            return True
        return False

    # Recall: how many gold items matched by at least one prediction
    matched_gold = 0
    for t in norm_gold:
        if any(_matches(c, t) for c in norm_pred):
            matched_gold += 1

    # Precision: how many predictions matched at least one gold item
    matched_pred = 0
    for c in norm_pred:
        if any(_matches(c, t) for t in norm_gold):
            matched_pred += 1

    precision = matched_pred / len(norm_pred) if norm_pred else 0.0
    recall = matched_gold / len(norm_gold) if norm_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'matched_gold': matched_gold, 'matched_pred': matched_pred,
            'n_gold': len(norm_gold), 'n_pred': len(norm_pred)}


# ---------------------------------------------------------------------------
# Constraint entity extraction
# ---------------------------------------------------------------------------

def _extract_constraint_entities(cs):
    """Extract constraint entities for evidence tree sorting priority."""
    constraints = []
    # pathentity roles from decomposition
    for er in getattr(cs, 'entity_roles', []):
        if er.get('role') == 'pathentity':
            constraints.append(er['entity'])
    # breakpoint (endpoint) entities
    for idx in getattr(cs, 'breakpoints', {}).values():
        if idx is not None and hasattr(cs, 'ents') and 0 <= idx < len(cs.ents):
            constraints.append(cs.ents[idx])
    return constraints


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

def _extract_graph_evidence(reason_prompt) -> str:
    """Extract GRAPH EVIDENCE section from a stored reason prompt."""
    if not isinstance(reason_prompt, str):
        return ""
    m = re.search(r'GRAPH EVIDENCE:\s*\n(.*?)(?=━━━|CANDIDATE ENTITIES|STEP |\Z)', reason_prompt, re.DOTALL)
    if m:
        return m.group(1).rstrip()
    return ""


# ---------------------------------------------------------------------------
# Attempt scoring
# ---------------------------------------------------------------------------

def _attempt_score(state, candidate_threshold=50):
    """Score a planning attempt for comparison. Penalizes candidate explosion."""
    if state.get("error"):
        return (-1, -1, -1, -1)
    n_cand = len(state.get("answer_candidates", []))
    under_threshold = 1 if n_cand <= candidate_threshold else 0
    return (
        state.get("max_cov", 0),
        under_threshold,
        len(state.get("paths", [])),
        -n_cand,
    )
