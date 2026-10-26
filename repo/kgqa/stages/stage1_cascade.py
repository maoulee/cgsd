"""Stage 1 cascade: two-step decomposition (sub-questions -> triples).

Step 1: Decompose question into sub-questions (stable, high quality).
Step 2: Convert sub-questions into triples with anchor/entity_roles.

Populates cs.steps, cs.triples, cs.entity_roles, cs.anchor_idx, cs.anchor_name
so that downstream stages (3-8) work unchanged.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from kgqa.core.case_state import CaseState
from kgqa.core.utils import normalize
from kgqa.llm.batch import batch_call_llm
from kgqa.llm.prompts import CASCADE_DECOMP_PROMPT_EN, CASCADE_SUBQ_TO_TRIPLES_PROMPT_EN


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_subquestions(raw: str):
    """Parse Step 1 output: extract analysis + numbered sub-questions."""
    analysis = ""
    m = re.search(r'<analysis>(.*?)</analysis>', raw, re.DOTALL)
    if m:
        analysis = m.group(1).strip()

    subs = []
    m = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            sm = re.match(r'^\d+\.\s*(.+)', line.strip())
            if sm:
                s = sm.group(1).strip()
                if len(s) >= 3:
                    subs.append(s)
    return analysis, subs


def _parse_triples_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse Step 2 output: extract analysis + JSON answer."""
    answer_raw = ""
    m = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL)
    if m:
        answer_raw = m.group(1).strip()

    if not answer_raw:
        return None

    # Fix common JSON issues
    answer_raw = answer_raw.replace('“', '"').replace('”', '"')  # smart quotes
    answer_raw = answer_raw.replace('‘', "'").replace('’', "'")

    try:
        return json.loads(answer_raw)
    except json.JSONDecodeError:
        answer_raw = re.sub(r',\s*([}\]])', r'\1', answer_raw)
        try:
            return json.loads(answer_raw)
        except json.JSONDecodeError:
            return None


def _resolve_anchor(cs: CaseState, entity_roles: List[Dict]) -> bool:
    """Resolve anchor from entity_roles. Returns True if found."""
    q_entities = cs.sample.get("q_entity", [])

    for er in entity_roles:
        if er.get("role") == "anchorentity":
            anchor_name = er.get("entity", "").strip()
            if not anchor_name:
                continue

            sn = normalize(anchor_name)
            # Exact match
            for i, e in enumerate(cs.ents):
                if normalize(e) == sn:
                    cs.anchor_idx = i
                    cs.anchor_name = e
                    return True
            # Substring match
            for i, e in enumerate(cs.ents):
                en = normalize(e)
                if (sn in en or en in sn) and len(sn) >= 3:
                    cs.anchor_idx = i
                    cs.anchor_name = e
                    return True
            # Match via q_entities
            for qe in q_entities:
                qn = normalize(qe)
                if qn == sn or (len(sn) >= 3 and (sn in qn or qn in sn)):
                    for i, e in enumerate(cs.ents):
                        if normalize(e) == qn:
                            cs.anchor_idx = i
                            cs.anchor_name = e
                            return True

    return False


def _fallback_anchor(cs: CaseState):
    """Fallback: pick first q_entity or first entity."""
    q_entities = cs.sample.get("q_entity", [])
    if q_entities:
        qn = normalize(q_entities[0])
        for i, e in enumerate(cs.ents):
            if normalize(e) == qn:
                cs.anchor_idx = i
                cs.anchor_name = e
                return
    if cs.ents:
        cs.anchor_idx = 0
        cs.anchor_name = cs.ents[0]


def _validate_triples(triples: List[Dict], anchor_name: str) -> Optional[str]:
    """Validate triples: must be DAG with anchor as root, no duplicate sub-questions.

    Returns None if valid, or an error message string if invalid.
    """
    if not triples:
        return "no triples"

    def _norm(s):
        return s.strip().lower() if s else ""

    anchor_n = _norm(anchor_name)
    subjects = [_norm(t.get("subject", "")) for t in triples]
    objects = [_norm(t.get("object", "")) for t in triples]

    # Check 1: anchor must appear as subject of at least one triple
    anchor_is_subject = any(
        s == anchor_n or (anchor_n and s and (anchor_n in s or s in anchor_n) and len(s) >= 3)
        for s in subjects
    )
    if anchor_n and not anchor_is_subject:
        return f"anchor '{anchor_name}' never appears as triple subject — triples do not start from anchor"

    # Check 2: triples must form a DAG (no cycles)
    # Build adjacency: triple i's object -> triple j's subject
    n = len(triples)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and objects[i] and subjects[j] and objects[i] == subjects[j]:
                adj[i].append(j)

    # DFS cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    def has_cycle(u):
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and has_cycle(v):
                return True
        color[u] = BLACK
        return False

    for i in range(n):
        if color[i] == WHITE and has_cycle(i):
            return "triples contain a cycle — not a DAG"

    return None


def _validate_subquestions(subs: List[str]) -> Optional[str]:
    """Check for duplicate or degenerate sub-questions. Returns error message or None."""
    # 1-step: skip validation (single-hop questions are valid)
    if len(subs) == 1 and len(subs[0].strip()) >= 3:
        return None

    # Filter out degenerate sub-questions (too short, single chars)
    valid_subs = [(i, s) for i, s in enumerate(subs) if len(s.strip()) >= 3]
    if len(valid_subs) < len(subs):
        skipped = len(subs) - len(valid_subs)
        # If too many degenerate, the decomposition is bad
        if len(valid_subs) < 2:
            return f"degenerate sub-questions: {skipped} of {len(subs)} are too short"

    seen = {}
    for i, s in valid_subs:
        key = normalize(s)
        if key in seen:
            return f"duplicate sub-question: Q{seen[key]+1} and Q{i+1} are identical ('{s}')"
        seen[key] = i
    return None


def _is_trivial_decomposition(subs: List[str], question: str) -> bool:
    """Check if decomposition is trivial: only 1 sub-question that restates the original question.

    Only flags as trivial when the sub-question content is completely contained in
    the original question (not just word overlap). Genuinely rephrased single-step
    questions that add semantic value are NOT flagged.
    """
    if len(subs) != 1:
        return False

    def _clean(s):
        s = normalize(s)
        s = re.sub(r'["\'?,;.!?()]', '', s)
        for prefix in ("who ", "what ", "where ", "when ", "which ", "how ", "name "):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        return s.strip()

    sub_str = _clean(subs[0])
    q_str = _clean(question)
    if not sub_str or not q_str:
        return False
    # Exact match after cleaning
    if sub_str == q_str:
        return True
    # Sub-question content is a substring of the question (model just truncated it)
    if sub_str in q_str:
        return True
    # Question content is a substring of the sub-question (model just added filler)
    if q_str in sub_str:
        return True
    return False


def _compute_triple_depths(triples: List[Dict], anchor_name: str) -> List[int]:
    """Compute depth of each triple in the entity/variable DAG.

    Triples branching from the same intermediate entity get the same depth.
    E.g. h—r→t, t—r1→t2, t—r2→t3 → depths [0, 1, 1].
    """
    n = len(triples)
    if n == 0:
        return []

    def _norm(s):
        return s.strip().lower() if s else ""

    anchor_n = _norm(anchor_name)

    # subject/object of each triple (normalized)
    subjects = [_norm(t.get("subject", "")) for t in triples]
    objects = [_norm(t.get("object", "")) for t in triples]

    # For each triple, check if its subject is the anchor (direct depth 0)
    # or matches another triple's object (depth = other_depth + 1)
    depths = [None] * n

    # Iterative depth assignment
    changed = True
    max_iter = n + 1
    while changed and max_iter > 0:
        changed = False
        max_iter -= 1
        for i in range(n):
            if depths[i] is not None:
                continue
            subj = subjects[i]
            # Case 1: subject is the anchor → depth 0
            if subj == anchor_n or (anchor_n and subj and (anchor_n in subj or subj in anchor_n) and len(subj) >= 3):
                depths[i] = 0
                changed = True
                continue
            # Case 2: subject matches object of a triple with known depth
            for j in range(n):
                if depths[j] is None:
                    continue
                obj_j = objects[j]
                if subj and obj_j and subj == obj_j:
                    d = depths[j] + 1
                    if depths[i] is None or d < depths[i]:
                        depths[i] = d
                        changed = True

    # Fallback: any unassigned triples get sequential depth
    max_d = max((d for d in depths if d is not None), default=-1)
    for i in range(n):
        if depths[i] is None:
            max_d += 1
            depths[i] = max_d

    return depths


def _build_steps_from_triples(triples: List[Dict], sub_questions: List[str],
                               anchor_name: str = "") -> List[Dict]:
    """Build cs.steps from triples, grouping branches at the same depth.

    Instead of sequential step assignment, computes DAG depth so that
    triples branching from the same intermediate entity share a step.
    E.g. h—r→t, t—r1→t2, t—r2→t3 → step 1:{r}, step 2:{r1,r2}.
    """
    depths = _compute_triple_depths(triples, anchor_name)

    # Group triples by depth → step
    depth_groups = {}
    for i, d in enumerate(depths):
        depth_groups.setdefault(d, []).append(i)

    # Sort by depth, assign step numbers
    sorted_depths = sorted(depth_groups.keys())
    depth_to_step = {d: idx + 1 for idx, d in enumerate(sorted_depths)}

    steps = []
    for i, t in enumerate(triples):
        predicate = t.get("predicate", "")
        src_sq = t.get("source_subquestion", 0)
        sq_text = sub_questions[src_sq - 1] if 0 < src_sq <= len(sub_questions) else ""
        steps.append({
            "step": depth_to_step[depths[i]],
            "question": predicate,
            "definition": predicate,
            "subquestion": sq_text,
            "relation_query": predicate,
            "subject": t.get("subject", ""),
            "object": t.get("object", ""),
            "triple_idx": i,
            "depth": depths[i],
        })
    return steps


# ---------------------------------------------------------------------------
# Adaptive complexity router
# ---------------------------------------------------------------------------

# Constraint keywords that promote a 1-hop-shaped question to COMPLEX
# (superlative / temporal / negation / cardinality / ordinal).
_CONSTRAINT_KEYWORDS = (
    "before", "after", "during", "most", "largest", "biggest",
    "oldest", "youngest", "first", "last", "how many", "not ",
)


def classify_complexity(cs: CaseState) -> str:
    """Deterministic, zero-LLM router. SIMPLE iff all of:
      - exactly 1 sub-question
      - no pathentity / explicit endpoint role
      - question has no constraint keyword
    COMPLEX otherwise.
    """
    if len(cs.sub_questions) != 1:
        return "complex"

    # Any endpoint role beyond the anchor means an extra fixed entity → COMPLEX
    for er in cs.entity_roles:
        if er.get("role") == "pathentity":
            return "complex"

    q_lower = (cs.question or "").lower()
    if any(kw in q_lower for kw in _CONSTRAINT_KEYWORDS):
        return "complex"

    return "simple"


# ---------------------------------------------------------------------------
# Stage 1 cascade: main entry point
# ---------------------------------------------------------------------------

async def stage_1_cascade_decomposition(session, cases: List[CaseState],
                                          allow_1step: bool = False,
                                          adaptive_routing: bool = False):
    """Two-step cascade: sub-question decomposition -> triple generation.

    Populates cs.steps, cs.triples, cs.entity_roles, cs.answer_variable,
    cs.anchor_idx, cs.anchor_name for downstream stages.
    """
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # ── Step 1: sub-question decomposition ──
    step1_prompts = []
    for cs in active:
        cs.decomp_method = "cascade"
        prompt = CASCADE_DECOMP_PROMPT_EN.format(question=cs.question)
        cs.decomp_prompt_formatted = f"[CASCADE STEP 1: Sub-question Decomposition]\n{prompt}"
        step1_prompts.append([{"role": "user", "content": prompt}])

    step1_responses = await batch_call_llm(session, step1_prompts, max_tokens=1500)

    # Parse sub-questions
    step1_parsed = []
    for cs, raw in zip(active, step1_responses):
        cs.decomp_raw = raw or ""
        if not raw:
            step1_parsed.append(([], ""))
            continue
        analysis, subs = _parse_subquestions(raw)
        step1_parsed.append((subs, analysis))

    # ── Retry trivial 1-step decomposition ──
    trivial_cases = [(cs, i) for i, (cs, (subs, _)) in enumerate(zip(active, step1_parsed))
                     if _is_trivial_decomposition(subs, cs.question)]
    if trivial_cases and not allow_1step:
        retry_prompt_suffix = (
            "\n\nIMPORTANT: Your previous decomposition produced only 1 sub-question that restates "
            "the original question. This is NOT a valid decomposition.\n"
            "You MUST break the question into at least 2 distinct sub-questions, each covering "
            "a different relation, attribute, or constraint. If the question has multiple conditions "
            "or modifiers, each must become its own sub-question."
        )
        retry_prompts = []
        for cs, _ in trivial_cases:
            prompt = CASCADE_DECOMP_PROMPT_EN.format(question=cs.question) + retry_prompt_suffix
            retry_prompts.append([{"role": "user", "content": prompt}])

        retry_responses = await batch_call_llm(session, retry_prompts, max_tokens=1500)

        retried_trivial = 0
        for (cs, idx), raw in zip(trivial_cases, retry_responses):
            if not raw:
                continue
            new_subs, new_analysis = _parse_subquestions(raw)
            if len(new_subs) >= 2 or (len(new_subs) == 1 and not _is_trivial_decomposition(new_subs, cs.question)):
                step1_parsed[idx] = (new_subs, new_analysis)
                cs.decomp_raw = (cs.decomp_raw or "") + "\n\n[TRIVIAL RETRY]\n" + raw
                retried_trivial += 1

        print(f"    Trivial decomposition retry: {retried_trivial}/{len(trivial_cases)} recovered")

    # ── Step 2: sub-questions -> triples ──
    step2_prompts = []
    for cs, (subs, _) in zip(active, step1_parsed):
        entities = cs.sample.get("q_entity", [])
        subqs_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(subs))
        if not subqs_text:
            subqs_text = "(no sub-questions parsed)"

        prompt = CASCADE_SUBQ_TO_TRIPLES_PROMPT_EN.format(
            question=cs.question,
            entities=" | ".join(entities),
            subquestions=subqs_text,
        )
        step2_prompts.append([{"role": "user", "content": prompt}])
        cs.decomp_prompt_formatted = (cs.decomp_prompt_formatted or "") + f"\n\n[CASCADE STEP 2: Sub-questions → Triples]\n{prompt}"

    step2_responses = await batch_call_llm(session, step2_prompts, max_tokens=2000)

    # ── Parse & validate ──
    MAX_RETRIES = 1
    retry_cases = []  # (cs, subs, validation_error)

    def _parse_and_validate(cs, subs, s2_raw):
        """Parse step 2 output, run validation. Returns (parsed_dict | None, error | None)."""
        if not subs:
            return None, "cascade step 1: no sub-questions parsed"
        if not s2_raw:
            return None, "cascade step 2: empty response"

        parsed = _parse_triples_json(s2_raw)
        if not parsed or "triples" not in parsed:
            return None, "cascade step 2: JSON parse failed"

        triples = parsed.get("triples", [])
        if not triples:
            return None, "cascade step 2: no triples generated"

        # Check duplicate sub-questions
        dup_err = _validate_subquestions(subs)
        if dup_err:
            return None, f"cascade validation: {dup_err}"

        # Determine anchor name for DAG check
        anchor_name = ""
        for er in parsed.get("entity_roles", []):
            if er.get("role") == "anchorentity":
                anchor_name = er.get("entity", "")
                break

        # Check DAG + anchor as root
        dag_err = _validate_triples(triples, anchor_name)
        if dag_err:
            return None, f"cascade validation: {dag_err}"

        return parsed, None

    # First pass: parse all cases
    ok = 0
    parsed_results = []  # (cs, subs, parsed_or_None, error_or_None)
    for cs, (subs, _), s2_raw in zip(active, step1_parsed, step2_responses):
        cs.sub_questions = subs
        cs.answer_type = None

        parsed, err = _parse_and_validate(cs, subs, s2_raw)
        parsed_results.append((cs, subs, parsed, err))

        if err:
            cs.error = err
            cs.active = False
            retry_cases.append((cs, subs, err))
        else:
            ok += 1

    # ── Retry invalid cases ──
    retried_ok = 0
    if retry_cases:
        retry_prompts = []
        for cs, subs, err in retry_cases:
            entities = cs.sample.get("q_entity", [])
            subqs_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(subs)) if subs else "(no sub-questions)"
            retry_hint = (
                f"\n\n[VALIDATION FEEDBACK — previous attempt failed: {err}]\n"
                "Requirements:\n"
                "1. The anchorentity MUST appear as the subject of at least one triple (the starting point of traversal).\n"
                "2. Triples must form a directed acyclic graph rooted at the anchor.\n"
                "3. Sub-questions must not be duplicates.\n"
                "Please fix and regenerate."
            )
            prompt = CASCADE_SUBQ_TO_TRIPLES_PROMPT_EN.format(
                question=cs.question,
                entities=" | ".join(entities),
                subquestions=subqs_text,
            ) + retry_hint
            retry_prompts.append([{"role": "user", "content": prompt}])

        retry_responses = await batch_call_llm(session, retry_prompts, max_tokens=2000)

        for (cs, subs, _), s2_raw in zip(retry_cases, retry_responses):
            cs.active = True
            cs.error = None
            parsed, err = _parse_and_validate(cs, subs, s2_raw)

            if err:
                cs.error = f"cascade retry: {err}"
                cs.active = False
            else:
                # Update parsed_results entry
                for i, (ocs, _, _, _) in enumerate(parsed_results):
                    if ocs is cs:
                        parsed_results[i] = (cs, subs, parsed, None)
                        break
                retried_ok += 1
                cs.decomp_raw = (cs.decomp_raw or "") + "\n\n[STEP 2 RETRY]\n" + (s2_raw or "")

        print(f"    Cascade retry: {retried_ok}/{len(retry_cases)} recovered")

    # ── Populate CaseState from validated results ──
    for cs, subs, parsed, err in parsed_results:
        if not parsed:
            continue

        triples = parsed.get("triples", [])
        cs.triples = triples
        cs.entity_roles = parsed.get("entity_roles", [])
        cs.answer_variable = parsed.get("answer_variable", "")
        cs.answer_type = parsed.get("answer_type")

        # Build cs.steps from triples (group branches at same depth)
        anchor_for_steps = cs.anchor_name or ""
        if not anchor_for_steps and cs.entity_roles:
            for er in cs.entity_roles:
                if er.get("role") == "anchorentity":
                    anchor_for_steps = er.get("entity", "")
                    break
        cs.steps = _build_steps_from_triples(triples, subs, anchor_for_steps)

        # Resolve anchor
        if not _resolve_anchor(cs, cs.entity_roles):
            _fallback_anchor(cs)

        # Derive breakpoints from pathentity roles
        cs.breakpoints = {}
        if cs.entity_roles and cs.ents:
            ent_to_idx = {e.lower().strip(): i for i, e in enumerate(cs.ents)}
            for er in cs.entity_roles:
                if er.get("role") == "pathentity":
                    ent_name = er.get("entity", "").lower().strip()
                    idx = ent_to_idx.get(ent_name)
                    if idx is not None:
                        step_idx = len(cs.breakpoints)
                        cs.breakpoints[step_idx] = idx

        # Store raw for debugging
        if not cs.decomp_raw or "[STEP 2]" not in cs.decomp_raw:
            cs.decomp_raw = (cs.decomp_raw or "") + "\n\n[STEP 2]\n"

    # ── Adaptive routing: classify each active case as simple/complex ──
    # Zero-LLM classifier using fields already finalized above.
    # When adaptive_routing is off, every case stays complexity="complex"
    # (the CaseState default), so downstream behavior is byte-identical to today.
    if adaptive_routing:
        n_simple = 0
        for cs in active:
            if not cs.active:
                continue
            cs.complexity = classify_complexity(cs)
            if cs.complexity == "simple":
                n_simple += 1
        print(f"    Adaptive routing: {n_simple}/{len(active)} cases routed SIMPLE")

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["decomposition"] = dt / len(active)
    final_ok = ok + retried_ok
    print(f"  Stage 1 (Cascade Decomposition): {dt:.2f}s | {final_ok}/{len(active)} ok")
