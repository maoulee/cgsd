"""Stage 1: Question decomposition — chain decomposition and reflection.

Includes parsing functions (parse_decomposition, parse_chain), LLM relation
pruning helpers (llm_prune_all_relations, llm_reselect_single_step_relation),
and the two main stage functions: stage_1_decomposition and
stage_1_5_decomposition_reflect.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from kgqa.core.case_state import CaseState
from kgqa.core.config import ALLOW_1STEP
from kgqa.core.utils import normalize, rel_to_text, extract_xml_tag
from kgqa.llm.client import call_llm
from kgqa.llm.batch import batch_call_llm
from kgqa.llm.prompts import (
    DECOMP_PROMPT,
    ENTITY_ANALYSIS_PROMPT,
    CHAIN_PROMPT,
)
from kgqa.traversal.cvt import is_cvt_like


# ---------------------------------------------------------------------------
# Decomposition parsing
# ---------------------------------------------------------------------------

def _parse_endpoint(endpoint_raw: str):
    """Parse endpoint string: 'entity (entity_query: term)' or 'none'."""
    if endpoint_raw.lower() == "none":
        return None, None
    m_ep = re.match(r"(.+?)\s*\(entity_query:\s*(.+?)\)", endpoint_raw)
    if m_ep:
        return m_ep.group(1).strip(), m_ep.group(2).strip()
    return endpoint_raw, endpoint_raw


def parse_decomposition(raw: str) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]], Optional[str]]:
    """Parse LLM decomposition: extract anchor name, entity_query, structured steps, and answer_type.

    Supports both new format (with type: find/verify, quoted questions) and old format.
    New:  1. "Who was the Governor?" (type: find; relation_query: ...; endpoint: ...)
    Old:  1. Who was the Governor? (relation_query: ...; endpoint: ...)
    """
    anchor_name = None
    anchor_entity_query = None
    steps = []
    answer_type = None
    for line in raw.strip().split("\n"):
        line = line.strip()
        # Extract anchor line: Anchor: name (entity_query: ...)
        m_anchor = re.match(r"^Anchor:\s*(.+?)\s*\(entity_query:\s*(.+?)\)\s*$", line, re.IGNORECASE)
        if m_anchor:
            anchor_name = m_anchor.group(1).strip()
            anchor_entity_query = m_anchor.group(2).strip()
            continue
        # Fallback: anchor without entity_query
        m_anchor2 = re.match(r"^Anchor:\s*(.+)$", line, re.IGNORECASE)
        if m_anchor2 and not anchor_name:
            anchor_name = m_anchor2.group(1).strip()
            anchor_entity_query = anchor_name
            continue
        # Extract answer type: Answer_type: [type]
        m_at = re.match(r"^Answer_type:\s*(.+)$", line, re.IGNORECASE)
        if m_at and not answer_type:
            answer_type = m_at.group(1).strip()
            continue

        # New format: 1. "question" (type: find/verify; relation_query: ...; endpoint: ...)
        m_new = re.match(
            r'^(\d+)\.\s*["""“](.+?)["""”]\s*\(type:\s*(find|verify);\s*relation_query:\s*(.+?);\s*endpoint:\s*(.+?)\)\s*$',
            line)
        if m_new:
            endpoint_raw = m_new.group(5).strip()
            endpoint, endpoint_query = _parse_endpoint(endpoint_raw)
            steps.append({
                "step": int(m_new.group(1)),
                "question": m_new.group(2).strip(),
                "type": m_new.group(3).strip(),
                "relation_query": m_new.group(4).strip(),
                "endpoint": endpoint,
                "endpoint_query": endpoint_query,
            })
            continue

        # Old format: 1. question (relation_query: ...; endpoint: ... or none)
        m = re.match(r"^(\d+)\.\s+(.+?)\s*\(relation_query:\s*(.+?);\s*endpoint:\s*(.+?)\)\s*$", line)
        if not m:
            continue
        endpoint_raw = m.group(4).strip()
        endpoint, endpoint_query = _parse_endpoint(endpoint_raw)
        steps.append({
            "step": int(m.group(1)),
            "question": m.group(2).strip(),
            "type": "find",
            "relation_query": m.group(3).strip(),
            "endpoint": endpoint,
            "endpoint_query": endpoint_query,
        })
    return anchor_name, anchor_entity_query, steps, answer_type


def parse_chain(text):
    """Parse CHAIN_PROMPT output into structured dict with anchor, hops, endpoints.

    Returns: {'anchor': str, 'hops': [{'relation': str, 'keyword': str, 'definition': str, 'endpoint': str}],
              'endpoint_entities': [{'entity': str, 'hop': int}], 'reasoning': str, 'raw': str}
    """
    text = text.strip()
    text = re.sub(r'^(Answer|答案)[：:]\s*', '', text)

    reasoning = ''
    m = re.search(r'Reasoning:\s*(.+?)(?=\n\s*Chain:|\nChain:)', text, re.DOTALL)
    if m:
        reasoning = m.group(1).strip()

    # Extract answer_type (XML tag or legacy format)
    answer_type = None
    at_xml = re.search(r'<answer_type>(.*?)</answer_type>', text, re.DOTALL)
    if at_xml:
        answer_type = at_xml.group(1).strip()
    else:
        at_match = re.search(r'Answer_type:\s*(.+)', text)
        if at_match:
            answer_type = at_match.group(1).strip()

    cm = re.search(r'Chain:\s*\n?(.*?)(?=\n\s*(Analysis:|Endpoints:)|\nAnalysis:|\nEndpoints:|$)', text, re.DOTALL)
    if not cm:
        return None
    chain_lines = [l.strip() for l in cm.group(1).split('\n') if '-(' in l]
    if not chain_lines:
        return None
    chain_text = chain_lines[-1]

    am = re.match(r'^(.+?)\s*-\(', chain_text)
    if not am:
        return None
    anchor = am.group(1).strip()
    rest = chain_text[am.end() - 1:]

    hops = []
    for m in re.finditer(r'\(([^)]+)\)\s*(?:\.inv\s*)?->\s*(.+?)(?=\s*-\(|$)', rest):
        node = m.group(2).strip()
        if not node or node.lower() in ('unknown', 'the', 'a'):
            node = 'node'
        hops.append({'relation': m.group(1).strip(), 'endpoint': node, 'keyword': '', 'definition': '', 'subquestion': ''})

    if not hops:
        return None

    endpoint_entities = []
    ep_match = re.search(r'Endpoints:\s*(.+)', text)
    if ep_match:
        ep_text = ep_match.group(1).strip()
        if ep_text.lower() != 'none':
            for em in re.finditer(r'\[([^\]]+)\]\s*(?:at\s*hop\s*(\d+))?', ep_text):
                ep_name = em.group(1).strip()
                hop_num = int(em.group(2)) if em.group(2) else None
                endpoint_entities.append({'entity': ep_name, 'hop': hop_num})

    anm = re.search(r'Analysis:\s*\n(.+)', text, re.DOTALL)
    if anm:
        kw = re.findall(r'-\s*Keyword:\s*(.+)', anm.group(1))
        df = re.findall(r'-\s*Definition:\s*(.+)', anm.group(1))
        sq = re.findall(r'-\s*Sub-question:\s*(.+)', anm.group(1))
        for i, hop in enumerate(hops):
            if i < len(kw): hop['keyword'] = kw[i].strip()
            if i < len(df): hop['definition'] = df[i].strip()
            if i < len(sq): hop['subquestion'] = sq[i].strip()

    return {'anchor': anchor, 'hops': hops, 'reasoning': reasoning,
            'endpoint_entities': endpoint_entities, 'answer_type': answer_type, 'raw': text}



# ---------------------------------------------------------------------------
# LLM relation pruning helpers
# ---------------------------------------------------------------------------

def _parse_prune_result(raw, step_candidates, all_steps):
    """Parse LLM response for relation pruning. Returns dict mapping step_num -> set of selected indices."""
    selected_yaml = extract_xml_tag(raw, "selected")
    result = {}
    if selected_yaml:
        for line in selected_yaml.split('\n'):
            line = line.strip()
            m = re.match(r'step_(\d+)\s*:\s*\[(.*?)\]', line)
            if m:
                sn = int(m.group(1))
                nums = [int(x.strip()) for x in m.group(2).split(',') if x.strip().isdigit()]
                cands = step_candidates.get(sn, [])
                selected_indices = set()
                for n in nums:
                    if 1 <= n <= len(cands):
                        selected_indices.add(cands[n - 1][0])
                result[sn] = selected_indices

    for s in all_steps:
        sn = s["step"]
        if sn not in result or not result[sn]:
            cands = step_candidates.get(sn, [])
            result[sn] = set(idx for idx, _, _ in cands[:3])
    return result


async def llm_prune_all_relations(session, question, all_steps, step_candidates):
    """Single LLM call to prune relations for ALL steps at once.

    Args:
        all_steps: list of parsed step dicts
        step_candidates: dict mapping step_num -> list of (idx, rel_name, score)

    Returns:
        dict mapping step_num -> set of selected relation indices,
        and debug dict with full prompt/response
    """
    from kgqa.llm.prompts import _build_prune_all_prompt

    prompt, system = _build_prune_all_prompt(question, all_steps, step_candidates)

    # Try up to 3 times (1 initial + 2 retries)
    for attempt in range(3):
        raw = await call_llm(session, [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ], max_tokens=2000)

        selected_yaml = extract_xml_tag(raw, "selected")
        if selected_yaml:
            break
    else:
        selected_yaml = None

    result = _parse_prune_result(raw, step_candidates, all_steps)

    debug = {
        "prompt": prompt,
        "response": raw,
        "parsed_yaml": selected_yaml,
    }
    return result, debug


async def llm_reselect_single_step_relation(session, question, step, step_candidates, current_indices):
    """Reselect relations for one failed step, avoiding the current failed choice set."""
    cands = step_candidates.get(step["step"], [])
    if not cands:
        return set()

    cand_lines = []
    current_pos = set()
    for i, (idx, name, score) in enumerate(cands, 1):
        marker = " [CURRENT]" if idx in current_indices else ""
        if idx in current_indices:
            current_pos.add(i)
        cand_lines.append(f"  {i}. {name}{marker}")

    prompt = f"""The current relation choice for one reasoning step appears to be wrong or too noisy.

Question: {question}
Failed step: {step['question']}
Step purpose: {step.get('definition', '')}

Candidate relations:
{chr(10).join(cand_lines)}

Select 1 to 3 BETTER alternative relations for this step.

Rules:
- Prefer relations that directly express the step semantics.
- Avoid the currently marked failed choices if better alternatives exist.
- Do not select generic or weakly related relations just because they are broad.

Output format:
<analysis>One short sentence.</analysis>
<selected>comma-separated candidate numbers only</selected>"""

    raw = await call_llm(session, [
        {"role": "system", "content": "You reselect better knowledge graph relations for a single failed reasoning step. Output <analysis> and <selected>."},
        {"role": "user", "content": prompt},
    ], max_tokens=400)

    selected = set()
    sel_text = extract_xml_tag(raw, "selected") or ""
    for m in re.finditer(r"\d+", sel_text):
        n = int(m.group())
        if 1 <= n <= len(cands):
            idx = cands[n - 1][0]
            selected.add(idx)

    if not selected:
        for idx, _, _ in cands:
            if idx not in current_indices:
                selected.add(idx)
            if len(selected) >= 3:
                break
    if not selected:
        selected = set(idx for idx, _, _ in cands[:3])
    return selected


# ---------------------------------------------------------------------------
# Stage 1: Main decomposition
# ---------------------------------------------------------------------------

async def stage_1_decomposition(session, cases: List[CaseState]):
    """Two-stage decomposition: 1a (entity analysis + rewrite) -> 1b (chain decomposition).
    Uses V1 prompts: ENTITY_ANALYSIS_PROMPT + CHAIN_PROMPT.
    Original monolith implementation matching.
    """
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # == Stage 1a: Entity analysis + question rewrite (V1) ==
    prompts_1a = []
    for cs in active:
        if cs.use_ner and cs.ner_top_ents:
            _ents = cs.sample.get("q_entity", [])
            ent_str = " | ".join(str(e) for e in _ents)
            prompt_text = ENTITY_ANALYSIS_PROMPT.format(entities=ent_str, question=cs.question)
            cs._1a_prompt = prompt_text
            prompts_1a.append([{"role": "user", "content": prompt_text}])
        else:
            # Non-NER mode: skip 1a, use DECOMP_PROMPT directly
            decomp_q = f"Question: {cs.question}"
            if cs.anchor_forbidden:
                decomp_q += f"\nDo not use this previous anchor again: {cs.anchor_forbidden}"
            cs.decomp_question = decomp_q
            prompts_1a.append(None)

    # Run 1a for NER cases only
    ner_prompts = [p for p in prompts_1a if p is not None]
    ner_responses = await batch_call_llm(session, ner_prompts, max_tokens=800) if ner_prompts else []

    # Parse 1a results using V1 parser
    ner_idx = 0
    for cs in active:
        if not (cs.use_ner and cs.ner_top_ents):
            continue
        raw_1a = ner_responses[ner_idx] if ner_idx < len(ner_responses) else ""
        ner_idx += 1

        q_ents = cs.sample.get("q_entity", [])
        a1 = _parse_1a_v1(raw_1a or "", cs.question, q_ents)

        cs._1a_anchor = a1['anchor']
        cs._1a_endpoints = a1['required_properties']  # backward compat
        cs._1a_required_properties = a1['required_properties']
        cs._1a_answer_type = a1['answer_type']
        cs._1a_rewritten = a1['rewritten']
        cs._1a_interpretation = a1['interpretation']
        cs._1a_raw = raw_1a or ""

    # == Stage 1b: Chain decomposition (V1) ==
    prompts_1b = []
    for cs in active:
        if cs.use_ner and cs.ner_top_ents:
            _ents = cs.sample.get("q_entity", [])
            ent_str = " | ".join(str(e) for e in _ents)
            anchor = cs._1a_anchor or (_ents[0].strip() if _ents else "N/A")
            endpoints = cs._1a_endpoints or "none"
            answer_type = cs._1a_answer_type or "other"
            interpretation = cs._1a_interpretation or ""
            rewritten = cs._1a_rewritten or cs.question
            prompt_text = CHAIN_PROMPT.format(
                entities=ent_str, anchor=anchor, endpoints=endpoints,
                answer_type=answer_type, interpretation=interpretation,
                rewritten=rewritten, question=cs.question)
            cs.decomp_prompt_formatted = prompt_text
            cs.decomp_question = prompt_text
            prompts_1b.append([{"role": "user", "content": prompt_text}])
        else:
            decomp_q = f"Question: {cs.question}"
            if cs.anchor_forbidden:
                decomp_q += f"\nDo not use this previous anchor again: {cs.anchor_forbidden}"
            cs.decomp_question = decomp_q
            prompts_1b.append([
                {"role": "system", "content": DECOMP_PROMPT},
                {"role": "user", "content": decomp_q},
            ])

    responses = await batch_call_llm(session, prompts_1b, max_tokens=1500)

    for cs, raw in zip(active, responses):
        cs.decomp_raw = raw or ""

        if cs.use_ner and cs.ner_top_ents:
            # -- NER mode: use parse_chain (V1) --
            parsed = parse_chain(raw or "")
            if not parsed or not parsed.get('hops'):
                cs.error = "decomposition failed"
                cs.active = False
                continue

            # V1: use Stage 1a anchor (authoritative), fallback to 1b chain anchor
            anchor_name = cs._1a_anchor or parsed['anchor']

            # Build steps from hops (V1 doesn't have separate constraints)
            cs.answer_type = cs._1a_answer_type or parsed.get('answer_type')
            cs.rewritten_question = cs._1a_rewritten or cs.question

            # Convert hops to steps
            cs.steps = []
            for j, hop in enumerate(parsed['hops']):
                cs.steps.append({
                    "step": j + 1,
                    "question": hop.get('subquestion') or hop['relation'],
                    "relation_query": hop['relation'],
                    "keyword": hop.get('keyword', ''),
                    "definition": hop.get('definition', ''),
                    "subquestion": hop.get('subquestion', ''),
                    "endpoint": "",
                })

            # -- Resolve breakpoints from endpoints (V1) --
            cs._pending_endpoints = []
            cs.breakpoints = {}
            q_entities = cs.sample.get("q_entity", [])

            # V1: endpoints come from endpoint_entities in parse_chain
            endpoint_entities = parsed.get('endpoint_entities', [])
            for ep in endpoint_entities:
                ep_name = ep['entity']
                ep_hop = ep.get('hop')

                # Find entity in ents list
                bp_idx = None
                en = normalize(ep_name)
                for i, e in enumerate(cs.ents):
                    if normalize(e) == en and not is_cvt_like(e):
                        bp_idx = i
                        break
                if bp_idx is None:
                    for i, e in enumerate(cs.ents):
                        e_name = normalize(e)
                        if not is_cvt_like(e) and min(len(en), len(e_name)) >= 4 and (en in e_name or e_name in en):
                            bp_idx = i
                            break
                if bp_idx is None or bp_idx == cs.anchor_idx:
                    # Try via q_entities
                    for qe in q_entities:
                        qn = normalize(qe)
                        if qn == en or (min(len(qn), len(en)) >= 4 and (qn in en or en in qn)):
                            for i, e in enumerate(cs.ents):
                                if normalize(e) == qn and not is_cvt_like(e):
                                    bp_idx = i
                                    break
                            if bp_idx is not None:
                                break

                if bp_idx is not None:
                    # Use hop number from endpoint_entities, or default to len(hops) + 1
                    bp_key = ep_hop if ep_hop is not None else (len(cs.steps) + 1)
                    while bp_key in cs.breakpoints:
                        bp_key += 100
                    cs.breakpoints[bp_key] = bp_idx
                else:
                    cs._pending_endpoints.append({"step_idx": ep_hop or (len(cs.steps) + 1), "query": ep_name})

            # -- Resolve anchor --
            if anchor_name:
                sn = normalize(anchor_name)
                for i, e in enumerate(cs.ents):
                    if normalize(e) == sn and not is_cvt_like(e):
                        cs.anchor_idx = i
                        cs.anchor_name = e
                        break
                if cs.anchor_idx is None:
                    for qe in q_entities:
                        qn = normalize(qe)
                        if qn in sn or sn in qn:
                            for i, e in enumerate(cs.ents):
                                if normalize(e) == qn and not is_cvt_like(e):
                                    cs.anchor_idx = i
                                    cs.anchor_name = e
                                    break
                            if cs.anchor_idx is not None:
                                break
                if cs.anchor_idx is None:
                    best = None
                    for i, e in enumerate(cs.ents):
                        en = normalize(e)
                        if (sn in en or en in sn) and not is_cvt_like(e) and len(sn) >= 3:
                            if best is None or len(e) > len(cs.ents[best]):
                                best = i
                    if best is not None:
                        cs.anchor_idx = best
                        cs.anchor_name = cs.ents[best]

            cs.entity_retrieval_details.append({
                "role": "anchor_ner",
                "ner_top_ents": cs.ner_top_ents[:6],
                "selected": cs.anchor_name,
                "selected_idx": cs.anchor_idx,
            })
            for step in cs.steps:
                step["entity_query"] = None
        else:
            # -- Non-NER mode: use old parse_decomposition --
            anchor_eq_name, anchor_eq, cs.steps, cs.answer_type = parse_decomposition(raw or "")
            if not cs.steps:
                cs.error = "decomposition failed"
                cs.active = False
                continue

            cs._pending_endpoints = []
            for step in cs.steps:
                if step["endpoint"] and step.get("endpoint_query"):
                    cs._pending_endpoints.append({"step_idx": step["step"], "query": step["endpoint_query"]})

            cs._pending_anchor_eq = anchor_eq
            cs._pending_anchor_eq_name = anchor_eq_name

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["decomposition"] = dt / len(active)
    ok = sum(1 for cs in active if cs.active)
    print(f"  Stage 1 (Decomposition): {dt:.2f}s | {ok}/{len(active)} ok")


# ---------------------------------------------------------------------------
# Stage 1.5: Decomposition reflection (retry 1-step decompositions)
# ---------------------------------------------------------------------------

async def stage_1_5_decomposition_reflect(session, cases: List[CaseState]):
    """Rule-based retry: re-invoke Stage 1b for 1-step decompositions using V1 logic.

    Uses V1 prompt (CHAIN_PROMPT) and parser (parse_chain) to produce a simple
    hop chain without the fact/constraint split of V2. This gives the model a
    different decomposition perspective on retry.
    """
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    retry_cases = []
    retry_prompts = []
    for cs in active:
        hop_count = sum(1 for s in cs.steps if 'constraint' not in s)
        if hop_count <= 1 and not ALLOW_1STEP:
            retry_cases.append(cs)
            _ents = cs.sample.get("q_entity", [])
            ent_str = " | ".join(_ents)
            anchor = cs._1a_anchor or cs.anchor_name or (_ents[0].strip() if _ents else "N/A")
            endpoints = cs._1a_endpoints if cs._1a_endpoints and cs._1a_endpoints.lower() != "none" else "none"
            answer_type = cs._1a_answer_type or cs.answer_type or "other"
            rewritten = cs._1a_rewritten or getattr(cs, 'rewritten_question', None) or cs.question
            interpretation = cs._1a_interpretation or ""

            prompt_text = CHAIN_PROMPT.format(
                entities=ent_str, anchor=anchor,
                endpoints=endpoints,
                answer_type=answer_type,
                interpretation=interpretation,
                rewritten=rewritten, question=cs.question)

            prev_step = cs.steps[0].get('question', '') if cs.steps else ''
            prev_rel = cs.steps[0].get('relation_query', cs.steps[0].get('keyword', '')) if cs.steps else ''
            hint = f"""

[NOTE: The previous decomposition produced only 1 hop ("{prev_step}" / rel: {prev_rel}), which is INSUFFICIENT for this complex question. You MUST produce at least 2 hops. Think about the intermediate entity between the anchor and the final answer.]"""
            prompt_text += hint
            retry_prompts.append([{"role": "user", "content": prompt_text}])

    n_retry = len(retry_cases)
    if n_retry == 0:
        dt = time.perf_counter() - _t0
        for cs in active:
            cs.stage_times["decomposition_reflect"] = dt / len(active)
        print(f"  Stage 1.5 (Reflect V1): {dt:.2f}s | 0 retry")
        return

    retry_responses = await batch_call_llm(session, retry_prompts, max_tokens=1500)

    for cs, raw in zip(retry_cases, retry_responses):
        cs.decomp_retry = True
        cs.decomp_reflect_raw = raw or ""
        cs.decomp_retry_reason = f"Only {sum(1 for s in cs.steps if 'constraint' not in s)} hop(s) for complex question"
        cs.decomp_raw = (cs.decomp_raw or "") + "\n\n[RETRY V1]\n" + (raw or "")

        parsed = parse_chain(raw or "")
        hop_count = len(parsed.get('hops', [])) if parsed else 0
        if not parsed or hop_count < 2:
            print(f"    Case {cs.case_id}: retry still has {hop_count} hops")
            continue

        anchor_name = cs._1a_anchor or parsed['anchor']

        # Build steps from V1 parsed (simple hops, no constraints)
        cs.steps = []
        for j, hop in enumerate(parsed['hops']):
            cs.steps.append({
                "step": j + 1,
                "question": hop.get('subquestion', '') or hop['relation'],
                "keyword": hop.get('keyword', ''),
                "definition": hop.get('definition', ''),
                "subquestion": hop.get('subquestion', ''),
                "relation_query": hop.get('keyword', '') or hop['relation'],
                "endpoint": None,
                "endpoint_query": None,
            })

        # Resolve endpoint entities from V1 parsed
        cs._pending_endpoints = []
        for ep in parsed.get('endpoint_entities', []):
            ep_name = ep['entity']
            ep_hop = ep.get('hop')
            step_idx = ep_hop if ep_hop else len(cs.steps)
            cs._pending_endpoints.append({"step_idx": step_idx, "query": ep_name})

        # Resolve breakpoints from endpoints
        cs.breakpoints = {}
        for ep_info in parsed.get('endpoint_entities', []):
            ep_name = ep_info['entity']
            en_norm = normalize(ep_name)
            bp_idx = None
            for i, e in enumerate(cs.ents):
                if normalize(e) == en_norm and not is_cvt_like(e):
                    bp_idx = i
                    break
            if bp_idx is None:
                for i, e in enumerate(cs.ents):
                    en = normalize(e)
                    if not is_cvt_like(e) and min(len(en_norm), len(en)) >= 4 and (en_norm in en or en in en_norm):
                        bp_idx = i
                        break
            if bp_idx is not None and bp_idx != cs.anchor_idx:
                hop_num = ep_info.get('hop') or len(cs.steps)
                while hop_num in cs.breakpoints:
                    hop_num += 100
                cs.breakpoints[hop_num] = bp_idx

        # Resolve anchor
        if anchor_name:
            sn = normalize(anchor_name)
            found = False
            for i, e in enumerate(cs.ents):
                if normalize(e) == sn and not is_cvt_like(e):
                    cs.anchor_idx = i
                    cs.anchor_name = e
                    found = True
                    break
            if not found:
                for i, e in enumerate(cs.ents):
                    en = normalize(e)
                    if (sn in en or en in sn) and not is_cvt_like(e) and len(sn) >= 3:
                        cs.anchor_idx = i
                        cs.anchor_name = e
                        found = True
                        break

        print(f"    Case {cs.case_id}: retry applied ({hop_count} hops, anchor={cs.anchor_name})")

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["decomposition_reflect"] = dt / len(active)
    print(f"  Stage 1.5 (Reflect V1): {dt:.2f}s | {n_retry}/{len(active)} retry")


def _parse_ner_start(raw):
    """Extract start entity from NER-style decomposition output."""
    m = re.search(r"^Start:\s*(.+)$", raw, re.M)
    return m.group(1).strip() if m else None
