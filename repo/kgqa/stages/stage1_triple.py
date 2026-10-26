"""Stage 1: Triple decomposition with verification.

Single-prompt triple decomposition with num_triples validation + retry,
followed by verification and re-decomposition of incomplete cases.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from kgqa.llm.batch import batch_call_llm, batch_call_llm_hot
from kgqa.llm.prompts import (
    DECOMP_VERIFY_PROMPT,
    REDECOMP_PROMPT,
    SIMPLE_DECOMP_PROMPT,
)
from kgqa.core.utils import sample_triple_batch


def _parse_triple_decomp(raw):
    """Parse triple decomposition output into structured data."""
    reason = ""
    m = re.search(r'<reason>(.*?)</reason>', raw, re.DOTALL)
    if m:
        reason = m.group(1).strip()

    answer_raw = ""
    m = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL)
    if m:
        answer_raw = m.group(1).strip()

    answer = {}
    if answer_raw:
        try:
            answer = json.loads(answer_raw.replace("'", '"'))
        except json.JSONDecodeError:
            answer_raw = re.sub(r',\s*([}\]])', r'\1', answer_raw)
            try:
                answer = json.loads(answer_raw)
            except json.JSONDecodeError:
                answer = {"parse_error": answer_raw[:200]}

    return reason, answer


def _apply_decomp_to_cs(cs, answer, raw):
    """Apply parsed decomposition answer to CaseState. Returns True on success."""
    from kgqa.core.utils import normalize

    if "parse_error" in answer:
        cs.active = False
        cs.error = f"Stage 1 triple: parse error — {answer['parse_error']}"
        return False

    cs.decomp_raw = raw
    cs.decomp_question = answer.get("question_decomposition", "")
    cs.triples = answer.get("triples", [])
    cs.entity_roles = answer.get("entity_roles", [])
    cs.answer_variable = answer.get("answer_variable", "")
    cs.answer_type = answer.get("answer_type", "") or cs.answer_type
    if answer.get("question_decomposition"):
        cs.rewritten_question = answer["question_decomposition"]

    # Resolve anchor from entity_roles
    anchor_resolved = False
    for er in cs.entity_roles:
        if er.get("role") == "anchorentity":
            ent_name = er.get("entity", "")
            if cs.ents:
                for i, e in enumerate(cs.ents):
                    if e.lower().strip() == ent_name.lower().strip():
                        cs.anchor_idx = i
                        cs.anchor_name = e
                        anchor_resolved = True
                        break
            if not anchor_resolved and cs.ents:
                el = ent_name.lower()
                for i, e in enumerate(cs.ents):
                    if el in e.lower() or e.lower() in el:
                        cs.anchor_idx = i
                        cs.anchor_name = e
                        anchor_resolved = True
                        break
            break

    if not anchor_resolved and cs.ents:
        cs.anchor_idx = 0
        cs.anchor_name = cs.ents[0]

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

    # Derive cs.steps from triples
    cs.steps = []
    for i, t in enumerate(cs.triples):
        step = {
            "step": i + 1,
            "question": t.get("predicate", ""),
            "definition": t.get("predicate", ""),
            "subquestion": "",
            "type": "hop",
            "relation_query": t.get("predicate", ""),
            "subject": t.get("subject", ""),
            "object": t.get("object", ""),
            "keyword": "",
            "endpoint": "",
        }
        cs.steps.append(step)
    return True


async def stage_1_triple_decomposition(session, cases):
    """Single-prompt triple decomposition with num_triples validation + retry."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # Build batch prompts
    prompts = []
    for cs in active:
        cs.decomp_method = "triple"
        q = cs.question
        entities = cs.sample.get("q_entity", []) or cs.ents[:10]
        prompt = SIMPLE_DECOMP_PROMPT.format(question=q, entities=" | ".join(str(e) for e in entities))
        prompts.append([{"role": "user", "content": prompt}])

    responses = await batch_call_llm(session, prompts, max_tokens=1500)

    n_ok = sum(1 for r in responses if r)
    n_none = sum(1 for r in responses if r is None)
    print(f"  Stage 1 decomp batch: {n_ok} ok, {n_none} failed, {len(responses)} total")

    # Parse and validate: check num_triples vs actual count
    retry_cases = []  # cases that need retry
    ok = 0
    for cs, raw in zip(active, responses):
        if not raw:
            cs.active = False
            cs.error = "Stage 1 triple: empty response"
            continue

        reason, answer = _parse_triple_decomp(raw)
        if "parse_error" in answer:
            cs.active = False
            cs.error = f"Stage 1 triple: parse error — {answer['parse_error']}"
            continue

        # Validate num_triples
        declared_n = answer.get("num_triples")
        actual_n = len(answer.get("triples", []))
        if declared_n is not None and declared_n != actual_n:
            cs.decomp_retry_reason = f"num_triples mismatch: declared={declared_n} actual={actual_n}"
            retry_cases.append(cs)
            continue

        _apply_decomp_to_cs(cs, answer, raw)
        ok += 1

    # Retry mismatched cases with higher temperature
    n_retried = 0
    if retry_cases:
        retry_prompts = []
        for cs in retry_cases:
            q = cs.question
            entities = cs.sample.get("q_entity", []) or cs.ents[:10]
            prompt = SIMPLE_DECOMP_PROMPT.format(question=q, entities=" | ".join(str(e) for e in entities))
            retry_prompts.append([{"role": "user", "content": prompt}])

        # Retry with temp=0.7 for more creative reasoning
        retry_responses = await batch_call_llm_hot(session, retry_prompts, max_tokens=1500)

        for cs, raw in zip(retry_cases, retry_responses):
            if not raw:
                # Fall back to original (even if mismatched)
                _apply_decomp_to_cs(cs, {"triples": [], "entity_roles": [], "parse_error": "retry failed"}, "")
                continue

            reason, answer = _parse_triple_decomp(raw)
            if _apply_decomp_to_cs(cs, answer, raw):
                n_retried += 1
            # If retry also fails, cs stays inactive

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["triple_decomp"] = dt / max(len(active), 1)
    print(f"  Stage 1 (Triple decomp): {dt:.2f}s | {ok}/{len(active)} ok, {len(retry_cases)} retry, {n_retried} retry ok")


async def stage_1_5_triple_verify(session, cases):
    """Verify all decompositions with CoT, then batch re-decompose incomplete ones."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active and getattr(cs, 'triples', None) and len(cs.triples) == 1]
    if not active:
        return

    # ── Batch 1: Verify all decompositions ──
    verify_prompts = []
    for cs in active:
        q = cs.question
        triples_json = json.dumps(cs.triples, ensure_ascii=False)
        prompt = DECOMP_VERIFY_PROMPT.format(
            question=q,
            triples_json=triples_json,
        )
        verify_prompts.append([{"role": "user", "content": prompt}])

    verify_responses = await batch_call_llm(session, verify_prompts, max_tokens=800)

    # Collect incomplete cases with their verify feedback
    incomplete_cases = []  # (cs, verify_raw, reason_text)
    continued = 0
    for cs, raw in zip(active, verify_responses):
        if not raw:
            continue
        judge = ""
        m = re.search(r'<judge>(.*?)</judge>', raw, re.DOTALL)
        if m:
            judge = m.group(1).strip().lower()

        if judge != "incomplete":
            continued += 1
            cs.decomp_verify_raw = raw
            continue

        # Extract reason from verify for feedback
        reason_match = re.search(r'<reason>(.*?)</reason>', raw, re.DOTALL)
        feedback = reason_match.group(1).strip() if reason_match else "Constraints are missing from the triples."
        cs.decomp_verify_raw = raw
        incomplete_cases.append((cs, feedback))

    print(f"  Stage 1.5 (Verify): {continued}/{len(active)} complete, {len(incomplete_cases)} incomplete")

    # ── Batch 2: Re-decompose incomplete cases ──
    patched = 0
    if incomplete_cases:
        redecomp_prompts = []
        for cs, feedback in incomplete_cases:
            entities = cs.sample.get("q_entity", []) or cs.ents[:10]
            prompt = REDECOMP_PROMPT.format(
                question=cs.question,
                entities=" | ".join(str(e) for e in entities),
                prev_triples=json.dumps(cs.triples, ensure_ascii=False),
                feedback=feedback,
            )
            redecomp_prompts.append([{"role": "user", "content": prompt}])

        redecomp_responses = await batch_call_llm(session, redecomp_prompts, max_tokens=1500)

        for (cs, feedback), raw in zip(incomplete_cases, redecomp_responses):
            if not raw:
                continue
            # Parse the re-decomposed output
            reason, answer = _parse_triple_decomp(raw)
            if "parse_error" in answer:
                continue

            new_triples = answer.get("triples", [])
            if not new_triples:
                continue  # re-decomp produced no triples

            # Apply patched decomposition
            cs.triples = new_triples
            if answer.get("entity_roles"):
                cs.entity_roles = answer["entity_roles"]
            if answer.get("answer_variable"):
                cs.answer_variable = answer["answer_variable"]
            if answer.get("answer_type"):
                cs.answer_type = answer["answer_type"]
            if answer.get("question_decomposition"):
                cs.rewritten_question = answer["question_decomposition"]

            # Re-resolve anchor from entity_roles
            anchor_resolved = False
            for er in cs.entity_roles:
                if er.get("role") == "anchorentity":
                    ent_name = er.get("entity", "")
                    if cs.ents:
                        for i, e in enumerate(cs.ents):
                            if e.lower().strip() == ent_name.lower().strip():
                                cs.anchor_idx = i
                                cs.anchor_name = e
                                anchor_resolved = True
                                break
                        if not anchor_resolved:
                            el = ent_name.lower()
                            for i, e in enumerate(cs.ents):
                                if el in e.lower() or e.lower() in el:
                                    cs.anchor_idx = i
                                    cs.anchor_name = e
                                    anchor_resolved = True
                                    break
                    break

            # Re-derive steps from patched triples
            cs.steps = []
            for i, t in enumerate(cs.triples):
                step = {
                    "step": i + 1,
                    "question": t.get("predicate", ""),
                    "definition": t.get("predicate", ""),
                    "subquestion": "",
                    "type": "hop",
                    "relation_query": t.get("predicate", ""),
                    "subject": t.get("subject", ""),
                    "object": t.get("object", ""),
                    "keyword": "",
                    "endpoint": "",
                }
                cs.steps.append(step)

            # Re-resolve pathentity breakpoints
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

            patched += 1

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["triple_verify"] = dt / max(len(active), 1)
    print(f"  Stage 1.5 (Verify+Re-decomp): {dt:.2f}s | checked={len(active)} patched={patched} continued={continued}")
