"""Stage 7: Path selection via batch LLM.

Compresses raw paths into logical paths, then uses an LLM to select 2-4
diverse reasoning paths with a pre-filtering step for deduplication.
"""
from __future__ import annotations

import re
import time
from typing import List

from kgqa.core.case_state import CaseState
from kgqa.core.utils import extract_xml_tag
from kgqa.traversal.path_utils import compress_paths
from kgqa.llm.batch import batch_call_llm


# ---------------------------------------------------------------------------
# Stage 7 main
# ---------------------------------------------------------------------------

async def stage_7_path_selection(session, cases: List[CaseState]):
    """Batch LLM path selection. Pre-filtered, single round with fallback."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]

    # Compress paths (reuse pre-computed logical_paths if available)
    for cs in active:
        if getattr(cs, 'logical_paths', None):
            continue
        breakpoint_indices = set(cs.breakpoints.values())
        cs.logical_paths = compress_paths(
            cs.paths, cs.ents, cs.rels, cs.anchor_idx, breakpoint_indices) if cs.paths else []

    cases_with_paths = [cs for cs in active if cs.logical_paths]
    if not cases_with_paths:
        return

    # Single round: pre-filter and select
    prompts = []
    for cs in cases_with_paths:
        # Pre-filter: deduplicate by relation chain prefix, keep max 6
        seen_chains = set()
        filtered_indices = []
        for i, lp in enumerate(cs.logical_paths):
            # Use first 2 relations as chain signature for diversity
            chain_sig = tuple(lp["rel_chain"][:2])
            if chain_sig in seen_chains and len(seen_chains) >= 3:
                continue  # Skip duplicate chains if we already have 3+ diverse ones
            seen_chains.add(chain_sig)
            filtered_indices.append(i)
            if len(filtered_indices) >= 20:
                break
        # If filtering removed too many, keep original top-20
        if len(filtered_indices) < 2:
            filtered_indices = list(range(min(20, len(cs.logical_paths))))

        cs._idx_map = {}
        path_lines = []
        for display_num, actual_idx in enumerate(filtered_indices, 1):
            lp = cs.logical_paths[actual_idx]
            ncands = len(lp.get("candidates", []))
            has_ep = "→ endpoint" if lp.get("endpoint") else ""
            covered = sorted(lp.get("covered_steps", frozenset()))
            if covered:
                step_text = ",".join(str(i + 1) for i in covered)
                coverage_text = f"covers steps {step_text}"
            else:
                coverage_text = "covers no selected step"
            path_lines.append(f"{display_num}. {lp['readable']}  ({coverage_text}; {ncands} candidates{has_ep})")
            cs._idx_map[display_num] = actual_idx

        # Build input from triple decomposition (with step-based fallback)
        rewritten = getattr(cs, 'rewritten_question', '') or cs.question
        answer_type = getattr(cs, 'answer_type', '') or ''

        # Expected pattern: triples > steps fallback
        expected_pattern = ""
        if getattr(cs, 'triples', None):
            pattern_lines = []
            for i, t in enumerate(cs.triples, 1):
                subj = t.get("subject", "")
                pred = t.get("predicate", "")
                obj = t.get("object", "")
                pattern_lines.append(f"  {i}. {subj} —[{pred}]—> {obj}")
            if pattern_lines:
                expected_pattern = "\n".join(pattern_lines)
        elif cs.steps:
            pattern_lines = []
            for s in cs.steps:
                defn = s.get('definition', '')
                kw = s.get('keyword', '')
                if defn:
                    pattern_lines.append(f"  - {defn}")
                elif kw:
                    pattern_lines.append(f"  - {kw}")
            if pattern_lines:
                expected_pattern = "\n".join(pattern_lines)

        # Endpoint constraints: entity_roles pathentity > breakpoints fallback
        endpoints = ""
        if getattr(cs, 'entity_roles', None):
            ep_entities = [er["entity"] for er in cs.entity_roles if er.get("role") == "pathentity"]
            if ep_entities:
                endpoints = ", ".join(ep_entities)
        if not endpoints:
            endpoint_names = [v for v in cs.breakpoints.values() if v is not None and v != cs.anchor_idx]
            if endpoint_names:
                ep_strs = [cs.ents[i] for i in endpoint_names if 0 <= i < len(cs.ents)]
                if ep_strs:
                    endpoints = ", ".join(ep_strs)

        paths_text = "\n".join(path_lines)

        select_prompt = f"""
Select reasoning paths for this question.

Question:
{cs.question}

Rewritten question:
{rewritten}

Answer type:
{answer_type}

Expected relation pattern:
{expected_pattern}

Endpoint constraints:
{endpoints}

Paths sorted by step coverage:
{paths_text}

Task:
Evaluate each path against the expected relation pattern and select 2-4 paths.

Path categories:
- Strong Match: The relation chain closely matches the intended reasoning structure.
- Partial but Useful: The path does not fully match but captures an important part of the reasoning.
- Mismatch: The path is semantically off-track.

Selection rules:
1. Prefer Strong Match paths first.
2. If fewer than 2 Strong Match paths exist, add the best Partial but Useful paths until selecting 2-4 paths.
3. Prefer semantic correctness before diversity.
4. Use diversity only among paths that are Strong Match or Partial but Useful.
5. Prefer paths that satisfy endpoint constraints and lead to the expected answer type.
6. Prefer paths whose relation chain matches the expected relation pattern, not merely paths with similar entity names.
7. Avoid near-duplicate paths with the same relation chain unless they reach meaningfully different candidate sets.
8. Avoid Mismatch paths unless there are no better alternatives.
9. Do not invent new paths, new relations, or external facts.
10. A single reasoning step may span multiple graph hops; judge coverage by the listed step numbers, not by hop count.

Reasoning output rules:
1. <analysis> must contain exactly three short sentences.
2. Do not analyze every path one by one.
3. Do not redo the full KGQA reasoning.
4. The three sentences must follow this structure:
   - Answer semantics: state what answer type the selected paths should lead to.
   - Path matching: summarize which paths are Strong Match or Partial but Useful.
   - Selection rationale: explain why the final paths are selected.

Output format:

<analysis>
Answer semantics: The selected paths should lead to an answer of type {answer_type} and satisfy the endpoint constraints.
Path matching: Path ... is a Strong Match, Path ... is Partial but Useful, and the remaining paths are weaker or mismatched.
Selection rationale: Path ... are selected because they best match the expected relation pattern while preserving useful semantic diversity.
</analysis>

<selected>1,3,5</selected>

Rules for <selected>:
- Use comma-separated path numbers only.
- Do not include brackets.
- Do not include explanations.
- Order selected path numbers by final preference, best first.
"""

        prompts.append([
            {"role": "system", "content": "You select diverse reasoning paths for graph QA. Always select 2-4 paths with different relation types. Output <analysis> and <selected>."},
            {"role": "user", "content": select_prompt},
        ])
        cs.path_select_prompt = select_prompt

    responses = await batch_call_llm(session, prompts, max_tokens=1000)

    for cs, raw in zip(cases_with_paths, responses):
        sel_text = extract_xml_tag(raw or "", "selected") or ""
        sel_indices = []
        for m in re.finditer(r'\d+', sel_text):
            display_num = int(m.group())
            if display_num in cs._idx_map:
                sel_indices.append(cs._idx_map[display_num])

        cs.selected_paths = list(dict.fromkeys(sel_indices))
        cs.path_select_response = raw or ""

    # Ensure minimum 2 selected paths (fallback to top patterns)
    for cs in cases_with_paths:
        if len(cs.selected_paths) < 2 and cs.logical_paths:
            for i in range(min(3, len(cs.logical_paths))):
                if i not in cs.selected_paths:
                    cs.selected_paths.append(i)
                if len(cs.selected_paths) >= 2:
                    break

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["path_select"] = dt / len(active)
    sel_counts = [len(cs.selected_paths) for cs in cases_with_paths]
    print(f"  Stage 7 (Path select): {dt:.2f}s | selected={sel_counts}")
