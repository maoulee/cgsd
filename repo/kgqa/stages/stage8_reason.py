"""Stage 8: Answer reasoning via batch LLM.

Builds evidence from selected logical paths, constructs reasoning prompts
(either entity-centric, entity-lite, checklist, ecot, or v2 style),
performs batch LLM inference, and applies post-hoc retry logic for
invalid answers (raw years/timestamps) and None answers (path reselect).
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from kgqa.core.case_state import CaseState
from kgqa.core.config import REASON_STYLE
from kgqa.agent.loader import build_agent_reason_prompt
from kgqa.core.utils import (
    normalize,
    candidate_hit,
    strict_candidate_hit,
    compute_match_stats,
    extract_xml_tag,
    _extract_constraint_entities,
)
from kgqa.stages.formatting import (
    build_endpoint_rescue_patterns,
    build_pattern_evidence_triples,
    format_pattern_evidence,
)
from kgqa.traversal.cvt import is_cvt_like, expand_through_cvt
from kgqa.traversal.logical_paths import materialize_selected_logical_patterns
from kgqa.llm.batch import batch_call_llm


# ---------------------------------------------------------------------------
# _case_state_to_result_dict (used by the runner after stage 8)
# ---------------------------------------------------------------------------

def _case_state_to_result_dict(cs: CaseState) -> Dict[str, Any]:
    """Convert CaseState to result dict matching run_case() output format."""
    from kgqa.llm.prompts import DECOMP_PROMPT, CHAIN_PROMPT
    logical_paths = cs.logical_paths or []
    return {
        "case_id": cs.case_id,
        "question": cs.question,
        "gt_answers": cs.gt_answers,
        "decomposition_prompt": cs.decomp_prompt_formatted or (CHAIN_PROMPT if cs.use_ner else DECOMP_PROMPT),
        "decomposition_question": cs.decomp_question,
        "decomposition": cs.decomp_raw,
        "decomp_retry": cs.decomp_retry,
        "decomp_reflect_raw": cs.decomp_reflect_raw or "",
        "decomp_retry_reason": cs.decomp_retry_reason or "",
        "stage_1a_raw": cs._1a_raw or "",
        "stage_1a_prompt": cs._1a_prompt or "",
        "stage_1a_anchor": cs._1a_anchor or "",
        "stage_1a_endpoints": cs._1a_endpoints or "",
        "stage_1a_interpretation": cs._1a_interpretation or "",
        "stage_1a_answer_type": cs._1a_answer_type or "",
        "stage_1a_rewritten": cs._1a_rewritten or "",
        "steps_parsed": cs.steps,
        "anchor_idx": cs.anchor_idx,
        "anchor_name": cs.ents[cs.anchor_idx] if cs.anchor_idx is not None and 0 <= cs.anchor_idx < len(cs.ents) else cs.anchor_name,
        "breakpoints": {k: cs.ents[v] for k, v in cs.breakpoints.items() if v is not None and 0 <= v < len(cs.ents)},
        "step_relations": [list(r) for r in cs.step_relations],
        "entity_retrieval_details": cs.entity_retrieval_details,
        "relation_retrieval_details": cs.relation_retrieval_details,
        "prune_debug": cs.prune_debug,
        "layer_diagnostics": cs.layer_diagnostics,
        "planning_attempts": cs.planning_attempts,
        "max_depth": cs.max_depth,
        "num_paths": len(cs.paths),
        "answer_candidates": cs.answer_candidates,
        "gt_hit": cs.gt_hit,
        "gt_hit_strict": cs.gt_hit_strict,
        "gt_f1": cs.gt_f1,
        "num_patterns": len(logical_paths),
        "logical_paths": [lp["readable"] for lp in logical_paths[:10]],
        "pattern_details": [{
            "candidates": lp.get("candidates", []),
            "best_tier": lp.get("best_tier", (0, -1, 0)),
            "endpoint": lp.get("endpoint"),
            "witness_nodes": lp.get("best_raw_path", {}).get("nodes", []),
            "witness_relations": lp.get("best_raw_path", {}).get("relations", []),
        } for lp in logical_paths],
        "llm_answer": cs.llm_answer,
        "llm_hit": cs.llm_hit,
        "llm_f1": cs.llm_f1,
        "llm_precision": cs.llm_precision,
        "llm_recall": cs.llm_recall,
        "selected_paths": cs.selected_paths,
        "num_triples": cs.num_triples,
        "attempt_log": cs.attempt_log,
        "path_select_prompt": getattr(cs, 'path_select_prompt', ''),
        "path_select_response": getattr(cs, 'path_select_response', ''),
        "llm_reasoning_prompt": cs.llm_reasoning_prompt,
        "llm_reasoning_full": cs.llm_reasoning_full,
        "stage_times": cs.stage_times,
        "error": cs.error,
    }


def _json_safe(obj):
    """Recursively convert frozenset/set/tuple to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, frozenset):
        return sorted(_json_safe(v) for v in obj) if all(isinstance(v, (int, str)) for v in obj) else list(obj)
    if isinstance(obj, set):
        return sorted(obj) if all(isinstance(v, (int, str)) for v in obj) else list(obj)
    return obj


def serialize_casestate(cs: CaseState) -> Dict[str, Any]:
    """Full CaseState serialization for trajectory dump (JSONL)."""
    logical_paths = cs.logical_paths or []
    # raw_paths omitted to reduce dump size (can be very large)
    return {
        "case_id": cs.case_id,
        "question": cs.question,
        "gt_answers": cs.gt_answers,
        # Subgraph
        "ents": cs.ents,
        "rels": cs.rels,
        "h_ids": cs.h_ids,
        "r_ids": cs.r_ids,
        "t_ids": cs.t_ids,
        # Stage 0
        "use_ner": cs.use_ner,
        # Stage 1 decomposition
        "decomp_method": cs.decomp_method,
        "decomp_raw": cs.decomp_raw,
        "decomp_prompt_formatted": cs.decomp_prompt_formatted,
        "steps_parsed": cs.steps,
        "answer_type": cs.answer_type,
        "triples": cs.triples,
        "entity_roles": cs.entity_roles,
        "answer_variable": cs.answer_variable,
        "sub_questions": cs.sub_questions,
        # Stage 2
        "anchor_idx": cs.anchor_idx,
        "anchor_name": cs.anchor_name,
        "breakpoints": {k: v for k, v in cs.breakpoints.items()},
        "step_candidates": {str(k): [(idx, name, round(sc, 4)) for idx, name, sc in v]
                            for k, v in cs.step_candidates.items()},
        "relation_retrieval_details": cs.relation_retrieval_details,
        "step_relations": [list(r) for r in cs.step_relations],
        "prune_debug": cs.prune_debug,
        # Stage 5 traversal
        "num_paths": len(cs.paths),
        "max_depth": cs.max_depth,
        "max_cov": cs.max_cov,
        "answer_candidates": cs.answer_candidates,
        "gt_hit": cs.gt_hit,
        "gt_hit_strict": cs.gt_hit_strict,
        "gt_f1": cs.gt_f1,
        "needs_direct_answer": cs.needs_direct_answer,
        "all_subgraph_nodes": sorted(cs.all_subgraph_nodes) if cs.all_subgraph_nodes else [],
        # Stage 7
        "logical_paths": [{
            "readable": lp.get("readable", ""),
            "rel_chain": lp.get("rel_chain", []),
            "candidates": lp.get("candidates", [])[:20],
            "endpoint": lp.get("endpoint"),
            "best_tier": lp.get("best_tier"),
            "best_raw_path": _json_safe(lp.get("best_raw_path", {})),
        } for lp in logical_paths[:20]],
        "selected_paths": cs.selected_paths,
        "path_select_prompt": getattr(cs, 'path_select_prompt', ''),
        "path_select_response": getattr(cs, 'path_select_response', ''),
        # Stage 8
        "llm_answer": cs.llm_answer,
        "llm_hit": cs.llm_hit,
        "llm_f1": cs.llm_f1,
        "llm_precision": cs.llm_precision,
        "llm_recall": cs.llm_recall,
        "llm_reasoning_prompt": cs.llm_reasoning_prompt,
        "llm_reasoning_full": cs.llm_reasoning_full,
        "num_triples": cs.num_triples,
        # Control
        "active": cs.active,
        "error": cs.error,
        "stage_times": cs.stage_times,
    }


# ---------------------------------------------------------------------------
# Reasoning prompt builders (one per REASON_STYLE)
# ---------------------------------------------------------------------------

def _build_default_reason_prompt(cs, pattern_text, answer_type_hint, rewritten_hint):
    """Default entity-centric 3-step reasoning prompt."""
    return f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

━━━ ENTITY-CENTRIC REASONING ━━━

STEP 1 — QUESTION UNDERSTANDING
Answer type: what kind of entity the question seeks (person, country, event, year, etc.).
Explicit constraints: conditions directly stated (dates, locations, superlatives, quantities).
Implicit constraint heuristic (pick one):
  - UNIQUE ROLE ("the governor/president/leader") without time qualifier → MOST RECENT only
  - EVENTS/ACHIEVEMENTS ("wins/championships/movies/albums") → return ALL matching
  - ATTRIBUTES/PROPERTIES ("languages/religions/currency") → return ALL that apply
  - GROUP MEMBERSHIP ("countries in / states in / members of") → return ALL matching members

STEP 2 — PER-CANDIDATE CONSTRAINT VERIFICATION
For EVERY candidate entity found in GRAPH EVIDENCE (including intermediate nodes and CVT attribute values), check:

2a. TYPE MATCH: Does this candidate's type match the answer type?
  Format: entity → KEEP (type matches) / REMOVE (type mismatch, state why)

2b. EXPLICIT CONSTRAINT CHECK (for type-matched candidates):
  For each explicit constraint from the question, check against graph evidence:
  Format: Constraint "[description]": entity1 PASS (evidence: ...), entity2 FAIL (evidence: ...)
  If graph evidence does NOT show failure → KEEP the entity.
  No dates in evidence → do NOT filter by time.

2c. IMPLICIT CONSTRAINT CHECK (for candidates passing 2b):
  Apply the heuristic from Step 1:
  - If unique role → pick MOST RECENT by graph dates; no dates → output ALL
  - If events/attributes/group → ALL candidates PASS

  Format: entity → PASS / FAIL (one reason)

STEP 3 — OUTPUT DECISION
Collect ALL candidates that PASS steps 2a + 2b + 2c.
- 1 entity → output it
- Multiple + unique role heuristic → pick MOST RECENT by graph dates; no dates → output ALL
- Multiple + events/attributes/group → output ALL
- Zero entities passed → <answer>None</answer>

RULES:
- Graph evidence ONLY. No outside knowledge.
- ANY entity in GRAPH EVIDENCE is a valid answer — including intermediate nodes and CVT attribute values.
- NEVER decide answer count before completing Step 2. Evaluate ALL candidates first.
- Singular/plural phrasing alone does NOT determine answer count.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- If unsure whether an entity satisfies a constraint → KEEP it.
- Over-output is better than discarding valid answers.
- For "group/organization that fought in/participated in" questions, political entities (countries, confederacies, alliances) are valid answer types — not only military units.

━━━ EXAMPLES (illustrate METHOD only) ━━━

Example A — TYPE FILTER + unique role:
Q: "Who is the president of France?"
2a (type person): KEEP [Macron, Hollande]. REMOVE [France(country), President(role), 2017-05-14(date)].
2b: No explicit time constraint.
2c (unique role, most recent): Macron PASS (term start 2017-05-14), Hollande FAIL (term start 2012-05-15).
3: Macron.

Example B — EVENTS → return ALL:
Q: "What movies did the actor who played Forrest Gump star in?"
2a (type movie): KEEP [Forrest Gump, Saving Private Ryan, Cast Away]. REMOVE [Tom Hanks(person), Actor(role), 1994(year)].
2b: No additional constraints beyond basic fact.
2c (events → ALL): All PASS.
3: Forrest Gump | Saving Private Ryan | Cast Away.

━━━ OUTPUT FORMAT ━━━
<reasoning>
Step 1: answer type, explicit constraints, implicit heuristic.
Step 2a: type match per candidate.
Step 2b: explicit constraint check per candidate.
Step 2c: implicit constraint check per candidate.
Step 3: passing set and output decision.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
No valid entity: <answer>None</answer>
NO text after </answer> tag."""


def _build_entity_lite_prompt(cs, pattern_text, answer_type_hint, rewritten_hint):
    """Lite version for 9B models: 2-step high-recall, no None."""
    n_cand = len(cs.answer_candidates) if cs.answer_candidates else 0
    cand_limit = min(n_cand, 30)
    cand_names = list(dict.fromkeys(cs.answer_candidates[:cand_limit])) if cs.answer_candidates else []
    cand_list = "\n".join(f"  - {c}" for c in cand_names) if cand_names else "  (see graph evidence above)"
    atype = cs.answer_type or "(infer from question)"
    reason_prompt = f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

━━━ ANSWER SELECTION ━━━

STEP 1 — Identify what the question needs.
Answer type: {atype}
Key constraints from the question: list them briefly.
Cardinality: Does the question ask for ONE specific role (e.g. "the governor", "the president")? If yes → pick the most recent one with dates in evidence. Otherwise → output ALL matching candidates.

STEP 2 — Evaluate each candidate.
For each candidate in CANDIDATE ENTITIES:
- KEEP if type roughly matches answer type.
- KEEP if no evidence contradicts it.
- REMOVE only if graph evidence explicitly shows it is WRONG type or contradicts a constraint.

RULES:
- Use graph evidence only. No outside knowledge.
- Do NOT remove a candidate for missing evidence. Remove only if evidence explicitly contradicts.
- Do NOT output None. If uncertain, output the best matching candidate(s).
- Prefer over-output to under-output.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES.
- Copy entity strings exactly. Never output Freebase IDs (m.0xxx).
- Over-output is better than discarding valid answers.

━━━ OUTPUT ━━━
<reasoning>
Step 1: answer type, constraints, cardinality decision.
Step 2: per-candidate KEEP/REMOVE with one reason each.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer> tag."""
    system = "You are a precise graph QA assistant. Always output at least one entity from CANDIDATE ENTITIES. Never output None. Copy entity strings exactly."
    return reason_prompt, system


def _build_check_prompt(cs, pattern_text, answer_type_hint, rewritten_hint):
    """Checklist v2: mechanical checks + type/granularity verification."""
    n_cand = len(cs.answer_candidates) if cs.answer_candidates else 0
    cand_limit = min(n_cand, 30)
    cand_names = list(dict.fromkeys(cs.answer_candidates[:cand_limit])) if cs.answer_candidates else []
    cand_list = "\n".join(f"  - {c}" for c in cand_names) if cand_names else "  (see graph evidence above)"
    atype = cs.answer_type or "(infer from question)"
    type_check_lines = "\n".join(
        f"  {c} -> TYPE: [matches {atype}?] -> KEEP / REMOVE"
        for c in cand_names[:25]
    )
    reason_prompt = f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

=== CHECKLIST ===

□ ANSWER TYPE: {atype}

□ TYPE FILTER — check each candidate's type matches answer type:
{type_check_lines}

□ CONSTRAINTS from question:
  - ____
  Per kept candidate: PASS / FAIL

□ GRANULARITY CHECK:
  Any candidate is a parent/child of another (e.g. city vs stadium, country vs sport)?
  -> Pick the one matching question specificity.

□ CARDINALITY (pick one):
  [ ] unique role ("the X", no time) -> MOST RECENT only
  [ ] events / achievements -> ALL
  [ ] attributes / properties -> ALL
  [ ] group membership -> ALL
  [ ] none of the above -> ALL remaining

□ KEPT AFTER FILTERS: [list here]

RULES:
- Graph evidence only. No outside knowledge.
- No dates in evidence -> do NOT remove by time.
- "When" -> event NAME, not raw year.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- Answer may be at intermediate hop.
- In doubt -> KEEP.
- NEVER output "None". If all removed, pick most specific candidate.

ANSWER STRING (copy exactly from CANDIDATE ENTITIES above):
____

<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer>."""
    system = "You are a graph fact-checker. Fill the checklist mechanically. One word per judgement (KEEP/REMOVE, PASS/FAIL). No paragraphs. Always output an answer. Copy entity strings exactly from CANDIDATE ENTITIES."
    return reason_prompt, system


def _build_ecot_prompt(cs, pattern_text, answer_type_hint, rewritten_hint):
    """Evidence-COT hybrid: checklist structure + 1-sentence evidence per item."""
    n_cand = len(cs.answer_candidates) if cs.answer_candidates else 0
    cand_limit = min(n_cand, 30)
    cand_names = list(dict.fromkeys(cs.answer_candidates[:cand_limit])) if cs.answer_candidates else []
    cand_list = "\n".join(f"  - {c}" for c in cand_names) if cand_names else "  (see graph evidence above)"
    atype = cs.answer_type or "(infer from question)"
    type_check_lines = "\n".join(
        f"  {c} -> [{atype}?] KEEP/REMOVE because [one fact from graph]"
        for c in cand_names[:25]
    )
    reason_prompt = f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

=== STRUCTURED EVIDENCE CHECK ===

1. ANSWER TYPE: {atype}

2. TYPE + EVIDENCE CHECK (one fact per candidate):
{type_check_lines}

3. CONSTRAINT CHECK:
   Question requires: ____
   Kept candidates that satisfy it: ____

4. GRANULARITY: If any kept candidate is a parent of another (city vs venue), pick the specific one.

5. CARDINALITY: unique role -> MOST RECENT | events/attributes -> ALL | in doubt -> ALL

6. KEPT: [list final kept candidates here]

RULES:
- Graph evidence only. No outside knowledge.
- No dates -> do NOT filter by time.
- "When" -> event NAME (e.g. "2014 World Series"), NOT raw timestamp or year.
- NEVER output "None". If all removed, pick most specific kept.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.

<answer>\\boxed{{exact entity from CANDIDATE ENTITIES}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer>."""
    system = "You are a precise graph QA checker. For each candidate, give ONE fact from graph evidence. Keep total under 10 lines. Answer MUST be an exact string from the candidate list. No outside knowledge."
    return reason_prompt, system


def _build_entity_prompt(cs, pattern_text, answer_type_hint, rewritten_hint):
    """Entity-centric 3-step: basic fact → constraint check → output."""
    n_cand = len(cs.answer_candidates) if cs.answer_candidates else 0
    cand_limit = min(n_cand, 30)
    cand_names = list(dict.fromkeys(cs.answer_candidates[:cand_limit])) if cs.answer_candidates else []
    cand_list = "\n".join(f"  - {c}" for c in cand_names) if cand_names else "  (see graph evidence above)"
    atype = cs.answer_type or "(infer from question)"

    # Extract relation names from selected patterns for basic-fact disambiguation
    rel_names = set()
    for i in cs.selected_paths:
        if i < len(cs.logical_paths):
            lp = cs.logical_paths[i]
            for hop in lp.get("hops", []):
                r = hop.get("relation", "")
                if r:
                    rel_names.add(r)
    rel_list = ", ".join(sorted(rel_names)[:12]) if rel_names else "(see evidence)"

    reason_prompt = f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES (traversal endpoints):
{cand_list}

NOTE: Any entity name appearing in GRAPH EVIDENCE (including intermediate nodes, CVT attribute values, and ← also lines) is also a valid answer. Do NOT restrict answers to the list above only.

━━━ ENTITY-CENTRIC REASONING ━━━

STEP 1 — QUESTION UNDERSTANDING
Answer type: what kind of entity the question seeks (person, country, event, year, etc.).
Available relations in evidence: {rel_list}
Explicit constraints from the question (time, location, superlatives, quantities, etc.): ____
Implicit constraint heuristic (pick one):
  - UNIQUE ROLE ("the governor/president/leader") without time qualifier → MOST RECENT only
  - EVENTS/ACHIEVEMENTS ("wins/championships/movies/albums") → return ALL matching
  - ATTRIBUTES/PROPERTIES ("languages/religions/currency") → return ALL that apply
  - GROUP MEMBERSHIP ("countries in / states in / members of") → return ALL matching members

STEP 2 — PER-CANDIDATE CONSTRAINT VERIFICATION
For EVERY candidate entity found in GRAPH EVIDENCE (including intermediate nodes and CVT attribute values), check:

2a. TYPE MATCH: Does this candidate's type match the answer type?
  Format: entity → KEEP (type matches) / REMOVE (type mismatch, state why)

2b. EXPLICIT CONSTRAINT CHECK (for type-matched candidates):
  For each explicit constraint from the question, check against graph evidence:
  Format: Constraint "[description]": entity1 PASS (evidence: ...), entity2 FAIL (evidence: ...)
  If graph evidence does NOT show failure → KEEP the entity.
  No dates in evidence → do NOT filter by time.

2c. IMPLICIT CONSTRAINT CHECK (for candidates passing 2b):
  Apply the heuristic from Step 1:
  - If unique role → pick MOST RECENT by graph dates; no dates → output ALL
  - If events/attributes/group → ALL candidates PASS

  Format: entity → PASS / FAIL (one reason)

STEP 3 — OUTPUT DECISION
Collect ALL candidates that PASS steps 2a + 2b + 2c.
- 1 entity → output it
- Multiple + unique role heuristic → pick MOST RECENT by graph dates; no dates → output ALL
- Multiple + events/attributes/group → output ALL
- Zero entities passed → <answer>None</answer>

RULES:
- Graph evidence ONLY. No outside knowledge.
- ANY entity in GRAPH EVIDENCE is a valid answer — including intermediate nodes, CVT attribute values, and entities in ← also lines.
- NEVER decide answer count before completing Step 2. Evaluate ALL candidates first.
- Singular/plural phrasing alone does NOT determine answer count.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- If unsure whether an entity satisfies a constraint → KEEP it.
- Over-output is better than discarding valid answers.
- For "group/organization that fought in/participated in" questions, political entities (countries, confederacies, alliances) are valid answer types — not only military units.

━━━ EXAMPLES (illustrate METHOD only) ━━━

Example A — TYPE FILTER + unique role:
Q: "Who is the president of France?"
2a (type person): KEEP [Macron, Hollande]. REMOVE [France(country), President(role), 2017-05-14(date)].
2b: No explicit time constraint.
2c (unique role, most recent): Macron PASS (term start 2017-05-14), Hollande FAIL (term start 2012-05-15).
3: Macron.

Example B — EVENTS → return ALL:
Q: "What movies did the actor who played Forrest Gump star in?"
2a (type movie): KEEP [Forrest Gump, Saving Private Ryan, Cast Away]. REMOVE [Tom Hanks(person), Actor(role), 1994(year)].
2b: No additional constraints beyond basic fact.
2c (events → ALL): All PASS.
3: Forrest Gump | Saving Private Ryan | Cast Away.

━━━ OUTPUT FORMAT ━━━
<reasoning>
Step 1: answer type, explicit constraints, implicit heuristic.
Step 2a: type match per candidate.
Step 2b: explicit constraint check per candidate.
Step 2c: implicit constraint check per candidate.
Step 3: passing set and output decision.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
No valid entity: <answer>None</answer>
NO text after </answer> tag."""
    system = "You are a precise graph QA system using per-candidate constraint verification. First identify answer type and constraints, then check EACH candidate against type + explicit + implicit constraints, then output ALL passing candidates. Any entity in GRAPH EVIDENCE is valid, not just CANDIDATE ENTITIES. NEVER decide answer count before checking all candidates. Over-output is better than discarding."
    return reason_prompt, system


def _build_simple_reason_prompt(cs, pattern_text, answer_type_hint, rewritten_hint):
    """SIMPLE-route (1-hop direct lookup) prompt.

    Minimal capability frame: the model just picks the answer entity from the
    graph evidence. Shares the SAME <answer>\\boxed{...}</answer> contract the
    existing _extract_llm_answer parser reads — no parser changes needed.
    """
    cand_list = ", ".join(cs.answer_candidates[:20]) if cs.answer_candidates else "No candidates"

    reason_prompt = f"""
QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

This is a direct one-hop lookup. Identify which candidate entity (or entities) the evidence directly shows as the answer to the question, respecting any constraint stated in the question. Use only the graph evidence; copy entity strings verbatim (use full entity names, never bare years or Freebase IDs). If multiple candidates qualify, list them all.

<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
None: <answer>None</answer>
NO text after </answer> tag.
"""
    system = ("You are a precise knowledge-graph QA engine. Decide only from "
              "the graph evidence provided; reason freely, then emit the answer "
              "in the <answer> tag with \\boxed{}.")
    return reason_prompt, system


def _build_free_reason_prompt(cs, pattern_text, answer_type_hint, rewritten_hint):
    """Minimal-constraint FREE-reasoning prompt.

    The user's hypothesis: with sufficient evidence + a capable model, stripping the
    reasoning template/cardinality rules and letting the model reason freely beats the
    rigid v2 method. Keeps ONLY the output-essential rules: entity fidelity, time→event
    name, and the <answer>\\boxed{} format. Shares the existing parser (no parser change).
    """
    cand_list = ", ".join(cs.answer_candidates[:20]) if cs.answer_candidates else "No candidates"
    reason_prompt = f"""
QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

Based on the graph evidence above, determine the answer to the question. Reason about it however you find most natural — there is no required reasoning format or step structure.

Rules:
- The answer must be an entity that appears in the graph evidence. Copy it verbatim (use the full entity name).
- For a "when"-type question, answer with the event NAME (e.g. "2014 World Series"), not a raw year.
- If multiple entities satisfy the question, list all of them. If none satisfies, answer None.

<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
None: <answer>None</answer>
NO text after </answer> tag.
"""
    system = ("You are a knowledge-graph QA engine. Decide only from the graph evidence; "
              "reason freely, then emit the answer entity in the <answer> tag.")
    return reason_prompt, system


def _build_v2_prompt(cs, pattern_text, answer_type_hint, rewritten_hint):
    """V2 candidate-audit prompt: question focus → evidence path → per-candidate audit."""
    atype = cs.answer_type or "(infer from question)"
    rewritten = getattr(cs, 'rewritten_question', '') or cs.question

    reason_prompt = f"""
QUESTION: {cs.question}

{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

━━━ ANSWER SELECTION TASK ━━━

Your task is to select the final answer entities from the graph evidence.
You must audit each plausible candidate individually — do NOT use a single bulk filter_action.

Step A — Question focus:
Identify the exact answer focus from the question.
The answer focus is stronger than the answer type hint.
If the answer type hint conflicts with the question focus, follow the question focus.

Step B — Evidence path summary:
Summarize the relevant graph path(s) that connect known entities to candidate answers.
Distinguish:
- bridge nodes: needed only to connect the path, not the final answer
- direct answer candidates: nodes that directly satisfy the answer focus
- auxiliary candidates: nodes reached through weaker, broader, inverse, or expansion relations (e.g. includes, is_part_of, containedby, venue history)

Step C — Candidate audit:
For each plausible candidate, judge it as:
- keep: directly satisfies the answer focus, supported by the strongest relevant evidence
- remove: bridge node, wrong type, wrong relation meaning, weaker auxiliary expansion, or contradicted by evidence
- uncertain_keep: evidence is genuinely ambiguous and the candidate cannot be safely removed

For each candidate, give one short evidence-based reason.

Step D — Final answer:
Output:
- all keep candidates
- uncertain_keep candidates ONLY if no stronger keep candidate excludes them
- do NOT output remove candidates

RULES:
1. Graph evidence only. No outside knowledge.
2. Entity names encode meaningful information. Use this as valid evidence.
3. Do NOT invent hidden constraints such as time overlap, currentness, or uniqueness — unless the question or graph evidence explicitly supports it.
4. A constraint used to identify a bridge entity must NOT be reused to remove final candidates unless the question explicitly says it also restricts the final answer.
5. If candidates come from different paths, prefer the path whose relation semantics best match the question wording.
6. If one candidate is a direct answer and another is only reached through expansion (includes, is_part_of, containedby, venue history, auxiliary path), prefer the direct answer.
7. Do NOT keep bridge entities unless the question asks for them.
8. Do NOT keep wrong-type entities.
9. Over-output is allowed ONLY when candidates are equally direct and equally supported.
10. CRITICAL: This is an ENTITY evaluation task. You MUST output the COMPLETE entity name exactly as it appears in the graph evidence. Events must include their full name (e.g. "2014 World Series", NOT "2014"). Places must include their full name. Never truncate entity names. Never output bare years, bare numbers, or abbreviated entity names.
11. Copy entity strings exactly. Never output Freebase IDs (m.0xxx).
12. NO text after </answer> tag.

Reasoning format rules:
- <reasoning> must contain exactly four sections: A, B, C, D.
- Candidate audit (Step C) must use compact bullet lines, one per candidate.
- Keep each candidate reason short (one line).
- Do NOT self-correct or discuss alternative interpretations.
- Prefer compact lists over prose.

<reasoning>
Step A — Question focus: ...
Step B — Evidence path summary: ...
Step C — Candidate audit:
- candidate: ... | decision: keep/remove/uncertain_keep | reason: ...
- candidate: ... | decision: keep/remove/uncertain_keep | reason: ...
Step D — Final selection: ...
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}} \\boxed{{e3}}</answer>
No valid entity: <answer>None</answer>
"""
    system = "You are a precise graph QA system using per-candidate constraint verification. First identify answer type and constraints, then check EACH candidate against type + explicit + implicit constraints, then output ALL passing candidates. Any entity in GRAPH EVIDENCE is valid, not just CANDIDATE ENTITIES. NEVER decide answer count before checking all candidates. Over-output is better than discarding."
    return reason_prompt, system


def _build_v2_prompt_v3(cs, pattern_text, answer_type_hint, rewritten_hint):
    """V3 candidate-audit prompt: cardinality-aware, over-output preferred for ALL-answer classes."""
    atype = cs.answer_type or "(infer from question)"
    rewritten = getattr(cs, 'rewritten_question', '') or cs.question

    reason_prompt = f"""
QUESTION: {cs.question}

{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

━━━ ANSWER SELECTION TASK ━━━

Your task is to select the final answer entities from the graph evidence.
You must audit each plausible candidate individually — do NOT use a single bulk filter_action.

Step A — Question focus:
Identify the exact answer focus from the question.
The answer focus is stronger than the answer type hint.
If the answer type hint conflicts with the question focus, follow the question focus.

Step B — Evidence path summary:
Summarize the relevant graph path(s) that connect known entities to candidate answers.
Distinguish:
- bridge nodes: needed only to connect the path, not the final answer
- direct answer candidates: nodes that directly satisfy the answer focus
- auxiliary candidates: nodes reached through weaker, broader, inverse, or expansion relations (e.g. includes, is_part_of, containedby, venue history)

Step C — Candidate audit:
For each plausible candidate, judge it as:
- keep: matches the answer focus and is not contradicted by explicit question constraints
- remove: wrong answer type, pure bridge node, wrong relation meaning, or explicitly contradicted by graph evidence
- uncertain_keep: plausible answer type/focus and graph evidence does not contradict it

Candidates do NOT compete with each other unless the question is classified as UNIQUE_ROLE, SUPERLATIVE, ORDINAL, or COUNT_LIMIT.
For attributes, properties, events, achievements, group membership, containment, location, and unknown cardinality, one strong candidate never excludes another plausible candidate.

For each candidate, give one short evidence-based reason.

Step D — Cardinality and final answer:
First classify the question:
- UNIQUE_ROLE / CURRENT_ROLE -> choose most recent by graph dates; if no dates, output ALL keep + uncertain_keep
- SUPERLATIVE / ORDINAL / COUNT_LIMIT -> apply the stated ranking/count only if graph evidence supports it
- EVENTS / ACHIEVEMENTS / WORKS -> output ALL keep + uncertain_keep
- ATTRIBUTES / PROPERTIES -> output ALL keep + uncertain_keep
- GROUP_MEMBERSHIP / CONTAINMENT / LOCATION-IN -> output ALL keep + uncertain_keep
- UNKNOWN -> output ALL keep + uncertain_keep

RULES:
1. Graph evidence only. No outside knowledge.
2. Entity names encode meaningful information. Use this as valid evidence.
3. Do NOT invent hidden constraints such as time overlap, currentness, or uniqueness — unless the question or graph evidence explicitly supports it.
4. A constraint used to identify a bridge entity must NOT be reused to remove final candidates unless the question explicitly says it also restricts the final answer.
5. Do NOT prefer one path over another for ALL-answer classes. If different paths yield candidates that satisfy the focus, keep all of them.
6. Do NOT remove expansion, inverse, includes, is_part_of, containedby, venue history, or auxiliary-path candidates only because they are less direct. Remove them only for wrong type, wrong relation meaning, or explicit contradiction.
7. Do NOT keep bridge entities unless the question asks for them.
8. Do NOT keep wrong-type entities.
9. Over-output is preferred for ALL-answer and UNKNOWN classes. Use under-output only for explicit unique, ranked, ordinal, or count-limited questions.
10. CRITICAL: This is an ENTITY evaluation task. You MUST output the COMPLETE entity name exactly as it appears in the graph evidence. Events must include their full name (e.g. "2014 World Series", NOT "2014"). Places must include their full name. Never truncate entity names. Never output bare years, bare numbers, or abbreviated entity names.
11. Copy entity strings exactly. Never output Freebase IDs (m.0xxx).
12. NO text after </answer> tag.

Reasoning format rules:
- <reasoning> must contain exactly four sections: A, B, C, D.
- Candidate audit (Step C) must use compact bullet lines, one per candidate.
- Keep each candidate reason short (one line).
- Do NOT self-correct or discuss alternative interpretations.
- Prefer compact lists over prose.

<reasoning>
Step A — Question focus: ...
Step B — Evidence path summary: ...
Step C — Candidate audit:
- candidate: ... | decision: keep/remove/uncertain_keep | reason: ...
- candidate: ... | decision: keep/remove/uncertain_keep | reason: ...
Step D — Question class: ... | Final selection: ...
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}} \\boxed{{e3}}</answer>
No valid entity: <answer>None</answer>
"""
    system = "You are a precise graph QA system using per-candidate constraint verification. Classify question cardinality first. For ALL-answer classes (events, attributes, group membership, etc.), output ALL kept and uncertain candidates. Over-output is preferred over discarding valid answers."
    return reason_prompt, system


# ---------------------------------------------------------------------------
# Tier retry prompt builders
# ---------------------------------------------------------------------------

def _build_tier1_retry_prompt(cs):
    """Tier 1: same reasoning style with format emphasis."""
    cand_str = ", ".join(cs.answer_candidates[:20]) if cs.answer_candidates else "No candidates"
    answer_type_hint = f"\nAnswer type: {cs.answer_type}" if cs.answer_type else ""
    return [
        {"role": "system", "content": "You are a precise graph QA system. You MUST output <answer> tags with the exact entity name from the candidates."},
        {"role": "user", "content": f"""QUESTION: {cs.question}{answer_type_hint}

ANSWER CANDIDATES (pick from these):
{cand_str}

Output your answer using this EXACT format:
<answer>\\boxed{{exact entity name}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer> tag."""},
    ]


def _build_v2_hybrid_retry(cs):
    """Tier 2: V2 structured audit + explicit candidate list + no-None constraint."""
    cand_str = ", ".join(cs.answer_candidates[:20]) if cs.answer_candidates else "No candidates"
    atype = cs.answer_type or "(infer from question)"
    return [
        {"role": "system", "content": "You are a precise graph QA system. You MUST select at least one entity from CANDIDATE ENTITIES. NEVER output None. Copy entity strings exactly."},
        {"role": "user", "content": f"""QUESTION: {cs.question}

CANDIDATE ENTITIES (the answer is among these):
{cand_str}

━━━ MANDATORY ANSWER SELECTION ━━━

Step A — Answer type: {atype}

Step B — Per-candidate audit (one line each):
For each candidate, judge: keep | remove | uncertain_keep

Step C — Final answer: output ALL kept candidates.

RULES:
- You MUST output at least one entity. NEVER output None.
- Graph evidence only. No outside knowledge.
- Copy entity names EXACTLY from CANDIDATE ENTITIES above.
- If unsure → keep the candidate.

<reasoning>
Step A: ...
Step B:
- candidate: ... | decision: keep/remove | reason: ...
Step C: ...
</reasoning>
<answer>\\boxed{{entity from CANDIDATE ENTITIES}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer> tag."""},
    ]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_llm_answer(raw):
    """Two-stage parsing: XML tag completeness → content extraction.

    Returns (llm_preds, llm_answer_string, parse_status).
    parse_status: 'ok' | 'incomplete_xml' | 'malformed_content'
    """
    text = raw or ""
    # Use findall + take LAST match to handle retry overwrites appended to same string
    ans_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)

    if not ans_matches:
        # Check if <answer> was started but never closed (truncated output)
        if '<answer>' in text and '</answer>' not in text:
            return [], "", "incomplete_xml"
        # No answer tags at all — also incomplete
        return [], "", "incomplete_xml"

    content = ans_matches[-1].strip()
    if not content:
        return [], "", "malformed_content"

    # Extract \boxed{...} patterns from content
    boxed = re.findall(r'\\boxed\{([^}]+)\}', content)
    if boxed:
        preds = [b.strip() for b in boxed]
        return preds, " | ".join(preds), "ok"

    # No \boxed — treat entire content as the answer
    preds = [content]
    return preds, content, "ok"


def _validate_answer_content(preds, answer_str):
    """Check if parsed answer content is valid or needs retry.

    Returns True if the answer is malformed and should be retried.
    """
    if not preds or not answer_str or not answer_str.strip():
        return True
    a = answer_str.strip().lower()
    if a in ("none", "n/a", "null", "maybe", "yes", "no", "unknown"):
        return True
    if '\\boxed' in a or 'boxed{' in a:
        return True
    if 'no answer' in a or 'no valid answer' in a or 'not found' in a:
        return True
    if a.startswith('is there') or a.startswith('boxed'):
        return True
    return False


def _resolve_entity_in_graph(pred: str, ents: list, threshold: float = 0.95):
    """Try to resolve a predicted entity against the full subgraph entity list.

    Returns (resolved_entity_or_None, match_type).
    match_type: 'exact' | 'fuzzy' | None
    """
    from difflib import SequenceMatcher as SM

    pred_norm = normalize(pred)
    if len(pred_norm) < 2:
        return None, None

    # 1. Exact normalized match
    for e in ents:
        if normalize(e) == pred_norm:
            return e, 'exact'

    # 2. Substring containment (bidirectional, min length 4)
    for e in ents:
        en = normalize(e)
        if len(pred_norm) >= 4 and len(en) >= 4:
            if pred_norm in en or en in pred_norm:
                ratio = min(len(pred_norm), len(en)) / max(len(pred_norm), len(en))
                if ratio >= 0.5:
                    return e, 'exact'

    # 3. Fuzzy match with length pre-filter
    best_score = 0.0
    best_match = None
    for e in ents:
        en = normalize(e)
        if len(en) < 2:
            continue
        len_ratio = min(len(pred_norm), len(en)) / max(len(pred_norm), len(en), 1)
        if len_ratio < 0.6:
            continue
        score = SM(None, pred_norm, en).ratio()
        if score > best_score:
            best_score = score
            best_match = e

    if best_score >= threshold:
        return best_match, 'fuzzy'

    return None, None


def _validate_and_fix_entities(cs, preds):
    """Validate LLM answer entities against the full subgraph entity list.

    For each predicted entity:
    - If found in cs.ents (exact or substring) → keep
    - If fuzzy match ≥ 95% similarity → replace with matched entity
    - If no match → mark as unmatched

    Returns (fixed_preds, unmatched_preds)
    """
    fixed_preds = []
    unmatched = []

    for pred in preds:
        resolved, _ = _resolve_entity_in_graph(pred, cs.ents, threshold=0.95)
        if resolved is not None:
            fixed_preds.append(resolved)
        else:
            unmatched.append(pred)
            fixed_preds.append(pred)  # keep original for now

    return fixed_preds, unmatched


def _build_entity_mismatch_retry(cs, unmatched):
    """Build a retry prompt telling the model which entities are not in the graph."""
    cand_str = ", ".join(cs.answer_candidates[:30]) if cs.answer_candidates else "No candidates"
    unmatched_str = ", ".join(f'"{u}"' for u in unmatched)
    atype = cs.answer_type or "(infer from question)"

    # Find similar entities from the full entity list for hints
    from difflib import SequenceMatcher as SM
    hints = []
    for u in unmatched:
        u_norm = normalize(u)
        if len(u_norm) < 3:
            continue
        similar = []
        for e in cs.ents:
            en = normalize(e)
            if len(en) < 3:
                continue
            if SM(None, u_norm, en).ratio() >= 0.7:
                similar.append(e)
            if len(similar) >= 5:
                break
        if similar:
            hints.append(f'  "{u}" → similar graph entities: {similar[:5]}')

    hints_text = "\n".join(hints) if hints else ""

    prompt_content = (
        f"QUESTION: {cs.question}\n"
        f"Answer type: {atype}\n\n"
        f"Your previous answer contained entities NOT present in the knowledge graph:\n"
        f"{unmatched_str}\n\n"
        f"These entities do not exist in the subgraph. Do NOT output them again.\n\n"
        f"CANDIDATE ENTITIES (the answer MUST be from this list):\n"
        f"{cand_str}"
    )

    if hints_text:
        prompt_content += f"\n\nSimilar graph entities that might match:\n{hints_text}"

    prompt_content += (
        "\n\nRULES:\n"
        "- You MUST select entity names from CANDIDATE ENTITIES above.\n"
        "- Copy entity strings EXACTLY — do not paraphrase, abbreviate, or modify.\n"
        "- If unsure, output the most similar candidate from the list.\n\n"
        "<answer>\\boxed{exact entity from CANDIDATE ENTITIES}</answer>\n"
        "Multiple: <answer>\\boxed{e1} \\boxed{e2}</answer>\n"
        "NO text after </answer> tag."
    )

    return [
        {"role": "system",
         "content": "You are a precise graph QA system. You MUST output entity names that exist exactly in the knowledge graph. Copy entity strings exactly from CANDIDATE ENTITIES."},
        {"role": "user", "content": prompt_content},
    ]


def _update_cs_answer(cs, llm_preds, llm_answer):
    """Update CaseState with parsed LLM answer and match stats."""
    cs.llm_answer = llm_answer
    cs.llm_hit = candidate_hit(llm_preds, cs.gt_answers)
    llm_stats = compute_match_stats(llm_preds, cs.gt_answers)
    cs.llm_f1 = llm_stats['f1']
    cs.llm_precision = llm_stats['precision']
    cs.llm_recall = llm_stats['recall']


def _merge_candidate_names(base, extra):
    """Append candidate names by normalized form while preserving order."""
    merged = []
    seen = set()
    for c in list(base or []) + list(extra or []):
        nc = normalize(c)
        if len(nc) < 2 or nc in seen:
            continue
        seen.add(nc)
        merged.append(c)
    return merged


def _materialized_traversal_candidates(patterns, ents, h_ids, r_ids, t_ids, anchor_idx):
    """Collect final-step entities from materialized logical paths.

    Stage 5 records relation-pattern candidates before Stage 8 rebuilds the
    concrete evidence.  When the materialized raw path reaches an attribute
    value or reaches an answer through a CVT node, this function exposes those
    end entities to the traversal GT annotation.
    """
    candidates = []
    node_ids = set()
    for lp in patterns or []:
        candidates.extend(lp.get("candidates", []) or [])
        for raw_path in lp.get("raw_paths", []) or []:
            nodes = raw_path.get("nodes", []) or []
            if not nodes:
                continue
            if nodes[-1] != anchor_idx:
                node_ids.add(nodes[-1])
            for nidx in nodes:
                if not (0 <= nidx < len(ents)):
                    continue
                if not is_cvt_like(ents[nidx]):
                    continue
                for ci, _ in expand_through_cvt(nidx, h_ids, r_ids, t_ids, ents):
                    if ci != anchor_idx and 0 <= ci < len(ents) and not is_cvt_like(ents[ci]):
                        node_ids.add(ci)
    for nidx in sorted(node_ids):
        if 0 <= nidx < len(ents):
            candidates.append(ents[nidx])
    return _merge_candidate_names([], candidates)


def _refresh_traversal_gt_from_materialized(cs, selected_pattern_objs):
    """Augment traversal GT labels after selected logical paths are materialized."""
    extra_candidates = _materialized_traversal_candidates(
        selected_pattern_objs, cs.ents, cs.h_ids, cs.r_ids, cs.t_ids, cs.anchor_idx,
    )
    if not extra_candidates:
        return
    cs.answer_candidates = _merge_candidate_names(cs.answer_candidates, extra_candidates)
    cs.path_candidates = _merge_candidate_names(getattr(cs, "path_candidates", []), extra_candidates)
    cs.gt_hit = candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
    cs.gt_hit_strict = strict_candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
    cs.gt_f1 = compute_match_stats(cs.path_candidates, cs.gt_answers)['f1']


# ---------------------------------------------------------------------------
# Stage 8 main
# ---------------------------------------------------------------------------

async def stage_8_answer_reasoning(session, cases: List[CaseState]):
    """Batch LLM answer reasoning with direct-answer fallback for failed graph cases."""
    _t0 = time.perf_counter()

    # Split into normal cases and direct-answer (safety net) cases
    active = [cs for cs in cases if cs.active]
    normal_cases = [cs for cs in active if cs.selected_paths and not cs.needs_direct_answer]
    direct_cases = [cs for cs in active if cs.needs_direct_answer]

    # --- Normal cases: graph-based reasoning ---
    prompts = []
    cases_with_triples = []
    for cs in normal_cases:
        selected_pattern_objs = [cs.logical_paths[i] for i in cs.selected_paths if i < len(cs.logical_paths)]
        if not selected_pattern_objs:
            # Re-select from logical_paths: pick top diverse ones by relation chain
            if cs.logical_paths:
                seen_chains = set()
                cs.selected_paths = []
                for i, lp in enumerate(cs.logical_paths[:20]):
                    chain_sig = tuple(lp.get("rel_chain", [])[:2])
                    if chain_sig in seen_chains and len(seen_chains) >= 4:
                        continue
                    seen_chains.add(chain_sig)
                    cs.selected_paths.append(i)
                    if len(cs.selected_paths) >= 5:
                        break
                if not cs.selected_paths:
                    cs.selected_paths = list(range(min(3, len(cs.logical_paths))))
                selected_pattern_objs = [cs.logical_paths[i] for i in cs.selected_paths]
            else:
                direct_cases.append(cs)
                continue
        breakpoint_indices = set(cs.breakpoints.values())
        selected_pattern_objs = list(selected_pattern_objs)
        selected_pattern_objs = materialize_selected_logical_patterns(
            selected_pattern_objs, cs.ents, cs.rels, cs.h_ids, cs.r_ids, cs.t_ids,
            cs.anchor_idx, breakpoint_indices,
        )
        selected_pattern_objs.extend(build_endpoint_rescue_patterns(
            cs.paths, selected_pattern_objs, cs.ents, cs.rels, cs.anchor_idx, breakpoint_indices,
        ))
        _refresh_traversal_gt_from_materialized(cs, selected_pattern_objs)
        pat_evidence = build_pattern_evidence_triples(
            selected_pattern_objs, cs.ents, cs.rels, cs.h_ids, cs.r_ids, cs.t_ids, cs.anchor_idx,
            max_grouped_lines=120)
        cs.num_triples = sum(len(pe.triples) for pe in pat_evidence.values())
        # Bug fix: Don't skip cases with empty triples if we have candidates from Stage 7
        # The candidates might already contain the answer (e.g., Case [27]: Blue Ivy is in candidates)
        if not pat_evidence and not cs.answer_candidates:
            direct_cases.append(cs)
            continue
        cases_with_triples.append(cs)

        # Bug fix: When pat_evidence is empty but we have candidates, use candidates as evidence
        if not pat_evidence:
            # Generate a candidate-based prompt when triples are empty
            candidates_str = ", ".join(cs.answer_candidates[:20]) if cs.answer_candidates else "No candidates found"
            selected_paths_str = ", ".join([f"Path {i+1}: {cs.logical_paths[i].get('readable', '')}"
                                           for i in cs.selected_paths if i < len(cs.logical_paths)])
            pattern_text = f"""SELECTED PATHS:
{selected_paths_str}

ANSWER CANDIDATES FROM GRAPH TRAVERSAL:
{candidates_str}

Note: Graph triples could not be extracted, but answer candidates are available from the path traversal."""
        else:
            _constraints = _extract_constraint_entities(cs)
            pattern_text = format_pattern_evidence(pat_evidence, constraint_entities=_constraints)

        answer_type_hint = f"\nAnswer type: {cs.answer_type}" if cs.answer_type else ""
        rewritten_hint = ""
        if hasattr(cs, 'rewritten_question') and cs.rewritten_question and cs.rewritten_question != cs.question:
            rewritten_hint = f"\nRewritten: {cs.rewritten_question}"

        # --- Dispatch prompt construction by REASON_STYLE ---
        # Agent mode (--reason-style agent): system = AGENTS.md, user = the
        # reason_simple / reason_complex skill doc + turn inputs. The loader
        # picks the skill by cs.complexity. Same <answer>\boxed{}</answer>
        # contract, no parser change. Takes over both SIMPLE and COMPLEX.
        if REASON_STYLE == "agent":
            system_msg, reason_prompt = build_agent_reason_prompt(
                cs, pattern_text, answer_type_hint, rewritten_hint)
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": reason_prompt},
            ])
        elif REASON_STYLE == "free":
            reason_prompt, system_msg = _build_free_reason_prompt(
                cs, pattern_text, answer_type_hint, rewritten_hint)
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": reason_prompt},
            ])
        # Adaptive routing (non-agent): SIMPLE cases use the minimal synthesize_simple
        # prompt. The else-branch keeps the COMPLEX dispatch byte-identical to today.
        elif getattr(cs, 'complexity', 'complex') == "simple":
            reason_prompt, system_msg = _build_simple_reason_prompt(
                cs, pattern_text, answer_type_hint, rewritten_hint)
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": reason_prompt},
            ])
        elif REASON_STYLE == "entity-lite":
            reason_prompt, system_msg = _build_entity_lite_prompt(cs, pattern_text, answer_type_hint, rewritten_hint)
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": reason_prompt},
            ])
        elif REASON_STYLE == "check":
            reason_prompt, system_msg = _build_check_prompt(cs, pattern_text, answer_type_hint, rewritten_hint)
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": reason_prompt},
            ])
        elif REASON_STYLE == "ecot":
            reason_prompt, system_msg = _build_ecot_prompt(cs, pattern_text, answer_type_hint, rewritten_hint)
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": reason_prompt},
            ])
        elif REASON_STYLE == "entity":
            reason_prompt, system_msg = _build_entity_prompt(cs, pattern_text, answer_type_hint, rewritten_hint)
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": reason_prompt},
            ])
        elif REASON_STYLE == "v2":
            reason_prompt, system_msg = _build_v2_prompt(cs, pattern_text, answer_type_hint, rewritten_hint)
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": reason_prompt},
            ])
        elif REASON_STYLE == "v3":
            reason_prompt, system_msg = _build_v2_prompt_v3(cs, pattern_text, answer_type_hint, rewritten_hint)
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": reason_prompt},
            ])
        else:
            # Default entity-centric prompt (same as the original default)
            reason_prompt = _build_default_reason_prompt(cs, pattern_text, answer_type_hint, rewritten_hint)
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": "You are a precise graph QA system using per-candidate constraint verification. First identify answer type and constraints, then check EACH candidate against type + explicit + implicit constraints, then output ALL passing candidates. Any entity in GRAPH EVIDENCE is valid, not just CANDIDATE ENTITIES. NEVER decide answer count before checking all candidates. Over-output is better than discarding."},
                {"role": "user", "content": reason_prompt},
            ])

    # --- Direct answer cases: parametric-only reasoning ---
    for cs in direct_cases:
        prompt = f"""QUESTION: {cs.question}

No reliable graph paths were found for this question. Use your parametric knowledge to answer directly.

<reasoning>One short sentence.</reasoning>
<answer>\\boxed{{exact entity}}</answer>"""
        cs.llm_reasoning_prompt = prompt
        prompts.append([
            {"role": "system", "content": "You are a precise QA system over Freebase (circa 2015). Answer the question directly using your knowledge. Output <reasoning> and <answer> XML tags."},
            {"role": "user", "content": prompt},
        ])

    # Batch all prompts together (normal + direct)
    if prompts:
        mt = 2400
        responses = await batch_call_llm(session, prompts, max_tokens=mt)

        all_answered = cases_with_triples + direct_cases
        for cs, raw in zip(all_answered, responses):
            cs.llm_reasoning_full = raw or ""
            llm_preds, llm_answer, parse_status = _extract_llm_answer(raw)
            if parse_status == "ok":
                _update_cs_answer(cs, llm_preds, llm_answer)
            else:
                _update_cs_answer(cs, [], "")

    # ==================================================================
    # Tier 0.5: Entity-in-graph validation + fix
    # ==================================================================

    all_answered = cases_with_triples + direct_cases
    entity_retry_cases, entity_retry_prompts = [], []
    for cs in all_answered:
        if not cs.llm_answer or not cs.ents:
            continue
        preds = [p.strip() for p in cs.llm_answer.split(" | ") if p.strip()]
        if not preds:
            continue
        fixed_preds, unmatched = _validate_and_fix_entities(cs, preds)
        if unmatched:
            # Some entities not in graph → retry with feedback
            if fixed_preds != preds:
                # At least fix what we can
                fixed_answer = " | ".join(fixed_preds)
                _update_cs_answer(cs, fixed_preds, fixed_answer)
            entity_retry_cases.append(cs)
            entity_retry_prompts.append(_build_entity_mismatch_retry(cs, unmatched))
        elif fixed_preds != preds:
            # All resolved via fuzzy/substring → just update
            fixed_answer = " | ".join(fixed_preds)
            _update_cs_answer(cs, fixed_preds, fixed_answer)
            cs.llm_reasoning_full += f"\n\n[ENTITY FIX] {preds} → {fixed_preds}"

    if entity_retry_cases:
        entity_retry_responses = await batch_call_llm(session, entity_retry_prompts, max_tokens=600)
        for cs, raw in zip(entity_retry_cases, entity_retry_responses):
            preds, answer, status = _extract_llm_answer(raw)
            if status == "ok" and not _validate_answer_content(preds, answer):
                # Validate the new answer too
                fixed_preds, unmatched2 = _validate_and_fix_entities(cs, preds)
                _update_cs_answer(cs, fixed_preds, " | ".join(fixed_preds))
                cs.llm_reasoning_full += f"\n\n[ENTITY RETRY] {answer}"
            # else: keep previous answer (may have been partially fixed)

    # ==================================================================
    # Unified 3-tier retry system
    # ==================================================================

    all_answered = cases_with_triples + direct_cases

    # -- Tier 1: Lightweight format retry --
    # Triggered by: XML incomplete, empty output, malformed content,
    # truncated entity names (bare year/timestamp), or invalid answer content.
    # Includes direct_cases now (they were excluded before).
    tier1_cases, tier1_prompts = [], []
    for cs in all_answered:
        raw = cs.llm_reasoning_full
        _, _, parse_status = _extract_llm_answer(raw)
        needs_retry = False

        if parse_status != "ok":
            needs_retry = True
        elif _validate_answer_content(
                [p.strip() for p in (cs.llm_answer or "").split(" | ")],
                cs.llm_answer):
            needs_retry = True
        else:
            # Bare year/timestamp or truncated entity name check
            if cs.answer_candidates and cs.llm_answer:
                preds = [p.strip() for p in cs.llm_answer.split(" | ")]
                cand_norms = {normalize(c): c for c in cs.answer_candidates[:50]}
                for p in preds:
                    pn = normalize(p)
                    if pn in cand_norms:
                        continue
                    if re.fullmatch(r'\d{4}(-\d{2}-\d{2}.*|-08:00)?', p):
                        needs_retry = True
                        break
                    is_trunc = any(
                        orig.lower().startswith(p.lower() + " ")
                        for _cn, orig in cand_norms.items()
                    )
                    if is_trunc:
                        needs_retry = True
                        break

        if needs_retry:
            tier1_cases.append(cs)
            tier1_prompts.append(_build_tier1_retry_prompt(cs))

    if tier1_cases:
        tier1_responses = await batch_call_llm(session, tier1_prompts, max_tokens=600)
        for cs, raw in zip(tier1_cases, tier1_responses):
            preds, answer, status = _extract_llm_answer(raw)
            if status == "ok" and not _validate_answer_content(preds, answer):
                _update_cs_answer(cs, preds, answer)
                cs.llm_reasoning_full += f"\n\n[TIER1 RETRY] {answer}"
            # else: leave as-is, Tier 2 will handle

    # -- Tier 2: V2 Hybrid structured retry --
    # Triggered by: Tier 1 failed, or primary returned "None"/invalid.
    # Uses V2's structured audit format + explicit candidate list + no-None.
    tier2_cases, tier2_prompts = [], []
    for cs in all_answered:
        # Skip if already has a valid answer from Tier 1 or primary
        _, _, parse_status = _extract_llm_answer(cs.llm_reasoning_full)
        current_answer = cs.llm_answer or ""
        if (parse_status == "ok"
                and current_answer
                and current_answer.lower() not in ("none", "n/a", "null", "")
                and not _validate_answer_content(
                    [p.strip() for p in current_answer.split(" | ")],
                    current_answer)):
            continue  # valid answer, skip

        if not cs.answer_candidates:
            continue  # no candidates to reason about, skip to Tier 3

        tier2_cases.append(cs)
        tier2_prompts.append(_build_v2_hybrid_retry(cs))

    if tier2_cases:
        tier2_responses = await batch_call_llm(session, tier2_prompts, max_tokens=1200)
        for cs, raw in zip(tier2_cases, tier2_responses):
            preds, answer, status = _extract_llm_answer(raw)
            if status == "ok" and not _validate_answer_content(preds, answer):
                _update_cs_answer(cs, preds, answer)
                cs.llm_reasoning_full += f"\n\n[TIER2 RETRY] {answer}"

    # -- Tier 3: Candidate direct output (no LLM call needed) --
    for cs in all_answered:
        if not cs.llm_answer or cs.llm_answer.lower() in ("none", "n/a", "null", ""):
            if cs.answer_candidates:
                # Output top candidates directly
                top_cands = list(dict.fromkeys(cs.answer_candidates[:5]))
                cs.llm_answer = " | ".join(top_cands)
                _update_cs_answer(cs, top_cands, cs.llm_answer)
                cs.llm_reasoning_full += f"\n\n[TIER3 FALLBACK] Direct candidate output: {cs.llm_answer}"

    # Mark all cases as completed
    for cs in cases:
        if cs.active:
            cs.active = False
            cs.stage_times.setdefault("llm_reasoning", time.perf_counter() - _t0)

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["llm_reasoning"] = dt / max(len(active), 1)
    llm_hits = sum(1 for cs in cases_with_triples + direct_cases if cs.llm_hit)
    print(f"  Stage 8 (Answer reasoning): {dt:.2f}s | LLM={llm_hits}/{len(cases_with_triples)}+{len(direct_cases)}direct")
