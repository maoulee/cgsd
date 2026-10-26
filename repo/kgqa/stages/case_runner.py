"""Per-case execution: NER → decomposition → GTE → prune → traverse → reasoning."""
from __future__ import annotations
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from kgqa.core.config import SKIP_NER, CANDIDATE_THRESHOLD
from kgqa.core.utils import (
    normalize, get_entity_contexts, extract_xml_tag,
    candidate_hit, rel_to_text, _attempt_score, _extract_constraint_entities,
)
from kgqa.traversal.cvt import expand_cvt_leaves, is_cvt_like, expand_through_cvt
from kgqa.traversal.frontier import diagnose_layers
from kgqa.traversal.k_queue import k_queue_traverse
from kgqa.traversal.path_utils import compress_paths, prefer_breakpoint_hit_paths, _extract_path_candidates
from kgqa.traversal.logical_paths import (
    build_mode_level_logical_paths, materialize_selected_logical_patterns,
)
from kgqa.stages.stage2_entity import gte_retrieve, llm_resolve_entity, resolve_anchor_ner
from kgqa.stages.stage1_decomp import (
    parse_chain, parse_decomposition, llm_prune_all_relations, llm_reselect_single_step_relation,
)
from kgqa.stages.formatting import (
    build_endpoint_rescue_patterns, build_pattern_evidence_triples,
    format_pattern_evidence,
)
from kgqa.stages.stage5_traverse import _collect_hr_frontier
from kgqa.llm.client import call_llm
from kgqa.llm.prompts import DECOMP_PROMPT, CHAIN_PROMPT


# ---------------------------------------------------------------------------
# Helper functions extracted from monolith
# ---------------------------------------------------------------------------

def _parse_ner_steps(raw):
    """Parse steps from NER decomposition format."""
    steps = []
    for line in raw.split("\n"):
        m = re.match(r"^\d+\.\s+(.+?)\s*\(relation:\s*(.+?)\)\s*$", line.strip())
        if m:
            steps.append({
                "step": len(steps) + 1,
                "question": m.group(1).strip(),
                "relation_query": m.group(2).strip(),
                "endpoint": None,
                "endpoint_query": None,
            })
    return steps


def _parse_ner_start(raw):
    m = re.search(r"^Start:\s*(.+)$", raw, re.M)
    return m.group(1).strip() if m else None


def _parse_ner_end(raw):
    m = re.search(r"^End:\s*(.+)$", raw, re.M)
    end = m.group(1).strip() if m else "none"
    if end.lower() == "none":
        return None, None
    return end, end


def _last_step_candidates(paths, anchor_idx, step_relations, h_ids, r_ids, t_ids, ents):
    """Return raw candidate names from the last step's BFS walk (all nodes, not just terminal)."""
    last_si = -1
    for i in range(len(step_relations) - 1, -1, -1):
        if step_relations[i]:
            last_si = i
            break
    if last_si < 0:
        return []
    frontier = {anchor_idx}
    for p in paths:
        if last_si not in p.get("covered_steps", frozenset()):
            ns = p.get("nodes", [])
            if ns:
                frontier.add(ns[-1])
    if len(frontier) <= 1:
        for si in range(last_si):
            srels = step_relations[si]
            if not srels:
                continue
            for j in range(len(h_ids)):
                if r_ids[j] in srels:
                    frontier.add(h_ids[j])
                    frontier.add(t_ids[j])
    last_nodes = set()
    for p in paths:
        if last_si not in p.get("covered_steps", frozenset()):
            continue
        ns = p.get("nodes", [])
        if not ns:
            continue
        fp = -1
        for fi in range(len(ns)):
            if ns[fi] in frontier:
                fp = fi
        for ni in range(fp + 1, len(ns)):
            last_nodes.add(ni)
    out = []
    for nidx in sorted(last_nodes):
        nm = ents[nidx] if 0 <= nidx < len(ents) else ""
        if is_cvt_like(nm):
            for ci, _ in expand_through_cvt(nidx, h_ids, r_ids, t_ids, ents):
                if ci != anchor_idx and 0 <= ci < len(ents) and not is_cvt_like(ents[ci]):
                    out.append(ents[ci])
        elif nidx != anchor_idx and nm:
            out.append(nm)
    return out


# ---------------------------------------------------------------------------
# Main per-case execution function
# ---------------------------------------------------------------------------

async def run_case(session, sample, pilot_row):
    question = pilot_row["question"]
    gt_answers = pilot_row.get("gt", pilot_row.get("ground_truth", pilot_row.get("gt_answers", [])))
    ents = sample.get("text_entity_list", []) + sample.get("non_text_entity_list", [])
    rels = list(sample.get("relation_list", []))
    h_ids, r_ids, t_ids = sample.get("h_id_list", []), sample.get("r_id_list", []), sample.get("t_id_list", [])

    _stage_times = {}

    # ── NER entity resolution (BEFORE expand_cvt_leaves, using original data) ──
    _t0 = time.perf_counter()
    if SKIP_NER:
        q_ents = sample.get("q_entity", [])
        ner_scored = [{"entity": e, "gte": 1.0} for e in q_ents]
        ner_name_to_ids = {}
        for i, name in enumerate(ents):
            ner_name_to_ids.setdefault(name, []).append(i)
        ner_top_ents = [(e, 1.0) for e in q_ents]
    else:
        ner_scored, ner_name_to_ids = await resolve_anchor_ner(
            session, question, ents, rels, h_ids, r_ids, t_ids)
        ner_top_ents = []
        _seen_e = set()
        for s in ner_scored[:6]:
            if s["entity"] not in _seen_e:
                ner_top_ents.append((s["entity"], s["gte"]))
                _seen_e.add(s["entity"])
    _stage_times["ner_resolve"] = time.perf_counter() - _t0

    # Auto-expand CVT leaf nodes (degree ≤ 1)
    _t0 = time.perf_counter()
    ents, rels, h_ids, r_ids, t_ids = expand_cvt_leaves(ents, rels, h_ids, r_ids, t_ids)
    rel_texts = list(rels)
    _stage_times["cvt_expand"] = time.perf_counter() - _t0

    # Build entity candidates (non-CVT)
    ent_candidates = [e for e in ents if e and len(e) > 1 and not is_cvt_like(e)]

    # Re-map NER entity names to expanded entity list indices
    ner_name_to_ids_expanded = {}
    for i, name in enumerate(ents):
        ner_name_to_ids_expanded.setdefault(name, []).append(i)

    async def execute_planning(anchor_forbidden=None, step_relations_override=None,
                               retry_note=None, use_ner=True):
        entity_retrieval_details = []
        anchor_idx = None
        anchor_name = None

        # ── NER-based anchor resolution (with fallback) ────────────────
        ner_ok = False
        if use_ner and ner_top_ents:
            decomp_question = f"Q: {question} [NER mode]"
            _ents = sample.get("q_entity", [])
            ent_str = "\n".join(f"- {e}" for e in _ents)
            first_ent = _ents[0].strip() if _ents else "Entity"
            prompt_text = CHAIN_PROMPT.format(
                entities=ent_str, anchor=first_ent, endpoints="none",
                answer_type="other", interpretation="", rewritten=question,
                question=question)
            raw = await call_llm(session, [
                {"role": "user", "content": prompt_text},
            ], max_tokens=2000)

            parsed = parse_chain(raw)
            if parsed and parsed.get('hops'):
                start_name = parsed['anchor']
                # Convert hops to steps format
                steps = []
                for i, hop in enumerate(parsed['hops']):
                    ep = None
                    for ep_info in parsed.get('endpoint_entities', []):
                        if ep_info.get('hop') == i + 1:
                            ep = ep_info['entity']
                            break
                    steps.append({
                        "step": i + 1,
                        "question": hop['relation'],
                        "type": "find",
                        "relation_query": hop.get('keyword', hop['relation']),
                        "definition": hop.get('definition', ''),
                        "keyword": hop.get('keyword', ''),
                        "endpoint": ep,
                        "endpoint_query": ep,
                        "entity_query": None,
                    })

                # Resolve anchor from NER scored entities
                if start_name:
                    sn = normalize(start_name)
                    for s in ner_scored:
                        if normalize(s["entity"]) == sn:
                            anchor_idx = ner_name_to_ids_expanded.get(s["entity"], [None])[0]
                            anchor_name = s["entity"]
                            break
                    if anchor_idx is None:
                        _cands = []
                        for s in ner_scored:
                            en = normalize(s["entity"])
                            if sn in en or en in sn:
                                _cands.append(s)
                        if _cands:
                            _cands.sort(key=lambda s: -len(s["entity"]))
                            anchor_idx = ner_name_to_ids_expanded.get(_cands[0]["entity"], [None])[0]
                            anchor_name = _cands[0]["entity"]
                if anchor_idx is None and ner_scored:
                    anchor_name = ner_scored[0]["entity"]
                    anchor_idx = ner_name_to_ids_expanded.get(anchor_name, [None])[0]

                entity_retrieval_details.append({
                    "role": "anchor_ner",
                    "ner_top_ents": ner_top_ents[:6],
                    "selected": anchor_name,
                    "selected_idx": anchor_idx,
                })

                # Endpoint resolve from parse_chain endpoint_entities
                breakpoints = {}
                for ep_info in parsed.get('endpoint_entities', []):
                    ep_name = ep_info.get('entity')
                    ep_hop = ep_info.get('hop')
                    if ep_name:
                        ep_rows = await gte_retrieve(session, ep_name, ent_candidates, top_k=3)
                        ep_cands = [r.get("candidate", "") for r in ep_rows if r.get("candidate")]
                        ep_ctx = get_entity_contexts(ep_cands, h_ids, r_ids, t_ids, ents, rels)
                        ep_cands_with_ctx = [(n, ep_ctx.get(n, "")) for n in ep_cands]
                        best = await llm_resolve_entity(session, question, ep_name, ep_cands_with_ctx)
                        idx = ents.index(best) if best and best in ents else None
                        if idx is not None and ep_hop:
                            breakpoints[ep_hop] = idx
                        entity_retrieval_details.append({
                            "role": f"endpoint_step{ep_hop}",
                            "query": ep_name,
                            "selected": best,
                            "selected_idx": idx,
                            "llm_resolved": True,
                        })
                ner_ok = True
            else:
                # Fallback to old parser
                anchor_eq_name_ner, _, steps, _ = parse_decomposition(raw)
                start_name = _parse_ner_start(raw)
                if not start_name and anchor_eq_name_ner:
                    start_name = anchor_eq_name_ner
                if steps:
                    if start_name:
                        sn = normalize(start_name)
                        for s in ner_scored:
                            if normalize(s["entity"]) == sn:
                                anchor_idx = ner_name_to_ids_expanded.get(s["entity"], [None])[0]
                                anchor_name = s["entity"]
                                break
                    if anchor_idx is None and ner_scored:
                        anchor_name = ner_scored[0]["entity"]
                        anchor_idx = ner_name_to_ids_expanded.get(anchor_name, [None])[0]
                    for step in steps:
                        step["entity_query"] = None
                    breakpoints = {}
                    ner_ok = True

        # ── Original decomposition-based anchor resolution (fallback) ──
        if not ner_ok:
            decomp_question = f"Question: {question}"
            if retry_note:
                decomp_question += f"\n\nRetry instruction: {retry_note}"
            if anchor_forbidden:
                decomp_question += f"\nDo not use this previous anchor again: {anchor_forbidden}"

            raw = await call_llm(session, [
                {"role": "system", "content": DECOMP_PROMPT},
                {"role": "user", "content": decomp_question},
            ])
            anchor_eq_name, anchor_eq, steps, _ = parse_decomposition(raw)
            if not steps:
                return {"error": "decomposition failed", "raw": raw, "decomp_question": decomp_question}

            if anchor_eq:
                rows = await gte_retrieve(session, anchor_eq, ent_candidates, top_k=5)
                topk = [{"rank": i+1, "candidate": r.get("candidate", ""), "score": round(r.get("score", 0), 4)} for i, r in enumerate(rows)]
                anchor_cands = [r["candidate"] for r in topk if r["candidate"]]
                anchor_ctx = get_entity_contexts(anchor_cands, h_ids, r_ids, t_ids, ents, rels)
                anchor_cands_with_ctx = [(n, anchor_ctx.get(n, "")) for n in anchor_cands]
                selected = await llm_resolve_entity(session, question, anchor_eq, anchor_cands_with_ctx)
                anchor_idx = ents.index(selected) if selected and selected in ents else None
                anchor_name = selected or anchor_eq_name
                entity_retrieval_details.append({
                    "role": "anchor",
                    "query": anchor_eq,
                    "top_k": topk,
                    "selected": selected,
                    "selected_idx": anchor_idx,
                    "llm_resolved": True,
                })

            # Endpoint resolve for original format
            breakpoints = {}
            for step in steps:
                if step["endpoint"] and step.get("endpoint_query"):
                    ep_rows = await gte_retrieve(session, step["endpoint_query"], ent_candidates, top_k=3)
                    ep_topk = [{"rank": i+1, "candidate": r.get("candidate", ""), "score": round(r.get("score", 0), 4)} for i, r in enumerate(ep_rows)]
                    ep_cands = [r["candidate"] for r in ep_topk if r["candidate"]]
                    ep_ctx = get_entity_contexts(ep_cands, h_ids, r_ids, t_ids, ents, rels)
                    ep_cands_with_ctx = [(n, ep_ctx.get(n, "")) for n in ep_cands]
                    best = await llm_resolve_entity(session, question, step["endpoint_query"], ep_cands_with_ctx)
                    idx = ents.index(best) if best and best in ents else None
                    if idx is not None:
                        breakpoints[step["step"]] = idx
                    entity_retrieval_details.append({
                        "role": f"endpoint_step{step['step']}",
                        "query": step["endpoint_query"],
                        "top_k": ep_topk,
                        "selected": best,
                        "selected_idx": idx,
                        "llm_resolved": True,
                    })

        # ── Shared: multi-query GTE (definition + subquestion + question + keyword) + prune + expand ──
        step_candidates = {}
        gte_per_step = {}
        relation_retrieval_details = []
        for step in steps:
            # Build deduplicated queries: definition, subquestion, question, keyword, relation_query
            _seen_q = set()
            queries = []
            for field in ["definition", "subquestion", "question", "keyword", "relation_query"]:
                val = step.get(field, "")
                if val and val.strip():
                    ql = val.strip().lower()
                    if ql not in _seen_q:
                        _seen_q.add(ql)
                        queries.append(val.strip())
            gte_all = {}
            queries_detail = []
            for query in queries:
                rows = await gte_retrieve(session, query, rels, candidate_texts=rel_texts, top_k=10)
                topk = []
                for i, r in enumerate(rows):
                    cand = r.get("candidate", "")
                    score = round(r.get("score", 0), 4)
                    idx_in_rels = rels.index(cand) if cand in rels else None
                    topk.append({"rank": i+1, "candidate": cand, "score": score, "rel_idx": idx_in_rels,
                                 "rel_text": rel_to_text(cand) if idx_in_rels is not None else ""})
                    if idx_in_rels is not None:
                        if idx_in_rels not in gte_all or score > gte_all[idx_in_rels][1]:
                            gte_all[idx_in_rels] = (rels[idx_in_rels], score)
                queries_detail.append({"query": query, "top_k": topk})

            gte_candidates = sorted(gte_all.items(), key=lambda x: -x[1][1])
            candidate_list = [(idx, name, score) for idx, (name, score) in gte_candidates]
            step_candidates[step["step"]] = candidate_list
            gte_per_step[step["step"]] = gte_all
            relation_retrieval_details.append({
                "step": step["step"],
                "queries": queries_detail,
                "gte_candidates_count": len(gte_all),
                "gte_indices": sorted(gte_all.keys()),
            })

        prune_result, prune_debug = await llm_prune_all_relations(session, question, steps, step_candidates)

        step_relations = []
        for step in steps:
            sn = step["step"]
            pruned = prune_result.get(sn, set())
            step_relations.append(pruned)
            gte_all = gte_per_step.get(sn, {})
            resolved_names = [{"idx": ri, "name": gte_all[ri][0]} for ri in sorted(pruned)] if pruned else []
            for rd in relation_retrieval_details:
                if rd["step"] == sn:
                    rd["resolved_indices"] = sorted(pruned)
                    rd["resolved_names"] = resolved_names
                    break

        if step_relations_override:
            for layer_idx, override_set in step_relations_override.items():
                if 0 <= layer_idx < len(step_relations):
                    step_relations[layer_idx] = set(override_set)
                    for rd in relation_retrieval_details:
                        if rd["step"] == steps[layer_idx]["step"]:
                            rd["override_indices"] = sorted(step_relations[layer_idx])
                            rd["override_names"] = [{"idx": ri, "name": rels[ri]} for ri in sorted(step_relations[layer_idx]) if 0 <= ri < len(rels)]
                            break

        prune_debug_field = {
            "prompt": prune_debug["prompt"],
            "response": prune_debug["response"],
            "parsed_yaml": prune_debug.get("parsed_yaml"),
        }

        paths, max_depth, max_cov = [], 0, 0
        logical_paths = []
        answer_candidates = []
        all_subgraph_nodes = set()
        if anchor_idx is not None:
            kq_rels = [(set(rs) if rs else set()) for rs in step_relations]
            breakpoint_indices = set(breakpoints.values()) if breakpoints else set()
            breakpoint_indices.discard(anchor_idx)
            logical_paths = build_mode_level_logical_paths(
                anchor_idx, kq_rels,
                h_ids, r_ids, t_ids, ents, rels,
                breakpoint_indices,
                beam_width=80, max_hops_per_step=2,
                relation_list=rels)
            if logical_paths:
                seen_witness = set()
                for lp in logical_paths:
                    witness = lp.get("best_raw_path")
                    if not witness:
                        continue
                    sig = (tuple(witness.get("nodes", [])), tuple(witness.get("relations", [])))
                    if sig in seen_witness:
                        continue
                    seen_witness.add(sig)
                    paths.append(witness)
                max_depth = max((p.get("depth", 0) for p in paths), default=0)
                max_cov = max((len(p.get("covered_steps", frozenset())) for p in paths), default=0)
            else:
                paths, max_depth, max_cov = k_queue_traverse(
                    anchor_idx, kq_rels,
                    h_ids, r_ids, t_ids, ents,
                    beam_width=80, max_hops_per_step=2,
                    relation_list=rels)

            # Prefer paths hitting breakpoint endpoints
            paths = prefer_breakpoint_hit_paths(paths, breakpoints, h_ids, r_ids, t_ids, ents)
            if paths:
                max_depth = max(p.get("depth", 0) for p in paths)
                max_cov = max(len(p.get("covered_steps", frozenset())) for p in paths)

            # Collect path nodes
            all_subgraph_nodes = {anchor_idx}
            for path in paths:
                all_subgraph_nodes.update(path["nodes"])

            # HR frontier: path-level (h+r) forward + (r+t) reverse triples
            hr_triples, hr_nodes = _collect_hr_frontier(
                anchor_idx, step_relations, h_ids, r_ids, t_ids,
                paths=paths)
            all_subgraph_nodes |= hr_nodes

            # Extract answer candidates from last step's BFS segment (+ CVT expansion)
            answer_candidates = _last_step_candidates(
                paths, anchor_idx, step_relations, h_ids, r_ids, t_ids, ents)

            seen = set()
            unique = []
            for c in answer_candidates:
                nc = normalize(c)
                if nc not in seen:
                    seen.add(nc)
                    unique.append(c)
            for lp in logical_paths:
                for c in lp.get("candidates", []):
                    nc = normalize(c)
                    if nc in seen:
                        continue
                    seen.add(nc)
                    unique.append(c)
            answer_candidates = unique

        # GT recall from path entities only (not HR frontier)
        path_candidates = _extract_path_candidates(paths, anchor_idx, ents, h_ids, r_ids, t_ids)
        gt_hit = candidate_hit(path_candidates, gt_answers) if path_candidates else False
        return {
            "error": None,
            "raw": raw,
            "decomp_question": decomp_question,
            "steps": steps,
            "anchor_idx": anchor_idx,
            "anchor_name": ents[anchor_idx] if anchor_idx is not None else anchor_name,
            "breakpoints": breakpoints,
            "step_relations_sets": step_relations,
            "entity_retrieval_details": entity_retrieval_details,
            "relation_retrieval_details": relation_retrieval_details,
            "prune_debug": prune_debug_field,
            "step_candidates": step_candidates,
            "paths": paths,
            "logical_paths": logical_paths,
            "max_depth": max_depth,
            "max_cov": max_cov,
            "answer_candidates": answer_candidates,
            "gt_hit": gt_hit,
        }

    planning_attempts = []
    _t0 = time.perf_counter()
    active = await execute_planning()
    _stage_times["planning_primary"] = time.perf_counter() - _t0
    planning_attempts.append({"label": "primary", "score": _attempt_score(active, CANDIDATE_THRESHOLD), "anchor": active.get("anchor_name")})

    if active.get("error"):
        return {
            "case_id": pilot_row["case_id"],
            "question": question,
            "error": active["error"],
            "gt_answers": gt_answers,
            "answer_candidates": [],
            "gt_hit": False,
            "raw": active.get("raw"),
        }

    nonempty_layers = [i for i, rs in enumerate(active["step_relations_sets"]) if rs]
    _t0 = time.perf_counter()
    layer_diagnostics = diagnose_layers(
        active["anchor_idx"], active["step_relations_sets"], h_ids, r_ids, t_ids, ents,
        max_hops=3,
    ) if active.get("anchor_idx") is not None else []
    _stage_times["diagnose_layers"] = time.perf_counter() - _t0

    # ── Anchor shortcut: skip noisy layers using already-recorded anchor_hit ──
    # BFS recorded paths reaching each layer from anchor (anchor_hit) and from
    # sequential frontier (frontier_hit). If a layer misses frontier but hits
    # anchor, the previous layer is noise → clear it (no LLM call needed).
    # Re-diagnose after shortcuts to update subsequent layers' frontier info.
    _t0_retry = time.perf_counter()
    shortcut_applied = False
    for li in list(nonempty_layers):
        if li < len(layer_diagnostics) and li > 0:
            d = layer_diagnostics[li]
            if not d["frontier_hit"] and d["anchor_hit"]:
                # Previous layer is noise — clear it
                if active["step_relations_sets"][li - 1]:
                    active["step_relations_sets"][li - 1] = set()
                    shortcut_applied = True

    if shortcut_applied:
        layer_diagnostics = diagnose_layers(
            active["anchor_idx"], active["step_relations_sets"], h_ids, r_ids, t_ids, ents,
            max_hops=3,
        ) if active.get("anchor_idx") is not None else []
        planning_attempts.append({"label": "anchor_shortcut", "score": _attempt_score(active, CANDIDATE_THRESHOLD), "anchor": active.get("anchor_name")})

    # Find first truly unreachable layer (both frontier AND anchor miss)
    first_miss = None
    for li in nonempty_layers:
        if li < len(layer_diagnostics):
            d = layer_diagnostics[li]
            if not d["frontier_hit"] and not d["anchor_hit"]:
                first_miss = li
                break

    # Retry B: reselect relations for truly unreachable layer (LLM call)
    if first_miss is not None:
        reselection = await llm_reselect_single_step_relation(
            session, question, active["steps"][first_miss],
            active["step_candidates"], active["step_relations_sets"][first_miss],
        )
        overrides = {first_miss: reselection}
        alt = await execute_planning(
            step_relations_override=overrides,
            retry_note=f"Retry relation selection for step {first_miss + 1}. The previous relation choice was too weak or too noisy.",
        )
        planning_attempts.append({"label": "reselect_failed_layer", "score": _attempt_score(alt, CANDIDATE_THRESHOLD), "anchor": alt.get("anchor_name")})
        if _attempt_score(alt, CANDIDATE_THRESHOLD) > _attempt_score(active, CANDIDATE_THRESHOLD):
            active = alt

    # Retry C: change anchor if still weak
    active_nonempty = [i for i, rs in enumerate(active["step_relations_sets"]) if rs]
    if active_nonempty and active.get("max_cov", 0) < len(active_nonempty):
        explicit_entities = {active.get("anchor_name")} | set(ents[idx] for idx in active.get("breakpoints", {}).values() if idx is not None and 0 <= idx < len(ents))
        explicit_entities = {e for e in explicit_entities if e}
        if len(explicit_entities) >= 2 and active.get("anchor_name"):
            alt = await execute_planning(
                anchor_forbidden=active["anchor_name"],
                retry_note="Choose a different explicit anchor entity from the question and redecompose the plan from that anchor.",
            )
            planning_attempts.append({"label": "anchor_swap_redecompose", "score": _attempt_score(alt, CANDIDATE_THRESHOLD), "anchor": alt.get("anchor_name")})
            if _attempt_score(alt, CANDIDATE_THRESHOLD) > _attempt_score(active, CANDIDATE_THRESHOLD):
                active = alt

    final_layer_diagnostics = diagnose_layers(
        active["anchor_idx"], active["step_relations_sets"], h_ids, r_ids, t_ids, ents,
        max_hops=3,
    ) if active.get("anchor_idx") is not None else []
    _stage_times["retries"] = time.perf_counter() - _t0_retry

    raw = active["raw"]
    decomp_question = active["decomp_question"]
    steps = active["steps"]
    anchor_idx = active["anchor_idx"]
    breakpoints = active["breakpoints"]
    step_relations = active["step_relations_sets"]
    entity_retrieval_details = active["entity_retrieval_details"]
    relation_retrieval_details = active["relation_retrieval_details"]
    prune_debug_field = active["prune_debug"]
    paths = active["paths"]
    max_depth = active["max_depth"]
    answer_candidates = active["answer_candidates"]
    gt_hit = active["gt_hit"]

    # Step 5: Compress paths into logical patterns
    breakpoint_indices = set(breakpoints.values())
    logical_paths = active.get("logical_paths") or (
        compress_paths(paths, ents, rels, anchor_idx, breakpoint_indices) if paths else []
    )

    # Step 6: Multi-attempt LLM reasoning with rollback
    # Model selects paths, can trigger rollback to get more paths (up to 3 attempts)
    # After all attempts, model reasons over all selected paths' triples
    llm_answer = None
    llm_hit = False
    selected_paths = []
    num_triples = 0
    attempt_log = []
    llm_reasoning_prompt = None
    llm_reasoning_full = None
    _t0_pathsel = time.perf_counter() if logical_paths else None
    if logical_paths:
        try:
            remaining = list(range(len(logical_paths)))

            for attempt in range(3):
                if not remaining:
                    break
                # Present remaining patterns with sequential numbering
                path_lines = []
                idx_map = {}  # display_number -> actual index in logical_paths
                for display_num, actual_idx in enumerate(remaining[:15], 1):
                    path_lines.append(f"{display_num}. {logical_paths[actual_idx]['readable']}")
                    idx_map[display_num] = actual_idx
                paths_text = "\n".join(path_lines)

                select_prompt = f"""Analyze and select reasoning paths for this question.

Question: {question}

Paths:
{paths_text}

Instructions:
1. Ignore hidden entity identities. Judge each path only by its relation sequence and node structure.
2. Select paths by semantic relevance, not by shortest length.
3. Keep a path if it preserves the intended multi-step meaning of the question, even if it is longer, includes bridge nodes, or is not the most direct-looking path.
4. Do not eliminate a path only because it is longer, slightly noisy, or contains extra intermediate structure.
5. When uncertain, prefer recall over precision: keep all paths that are semantically plausible.
6. Remove only paths that clearly contradict the question semantics.

Output format:
<analysis>
Two short sentences max. Do NOT deliberate. Just state which paths fit and why.
</analysis>
<selected>comma-separated path indices only</selected>
<need_more>yes or no</need_more>

Rules:
- Do not copy example numbers.
- Do not leave out a semantically plausible path just because it is longer.
- If several paths are plausible, select all of them."""

                # Try up to 3 times for valid XML output
                sel_raw = ""
                for sel_attempt in range(3):
                    sel_raw = await call_llm(session, [
                        {"role": "system", "content": "You analyze and select reasoning paths for multi-step QA. Your goal is high-recall semantic path selection. Judge semantic fit of relation chains to the question. Keep all semantically plausible paths. Output exactly three XML tags: <analysis>, <selected>, and <need_more>."},
                        {"role": "user", "content": select_prompt},
                    ], max_tokens=600)
                    if extract_xml_tag(sel_raw, "selected"):
                        break

                # Parse selected indices from <selected> tag
                sel_indices = []
                sel_text = extract_xml_tag(sel_raw, "selected") or ""
                for m in re.finditer(r'\d+', sel_text):
                    display_num = int(m.group())
                    if display_num in idx_map:
                        sel_indices.append(idx_map[display_num])

                # Parse rollback flag from <need_more> tag
                need_more_text = extract_xml_tag(sel_raw, "need_more") or "no"
                rollback = "yes" in need_more_text.lower()

                selected_paths.extend(sel_indices)
                attempt_log.append({
                    "attempt": attempt + 1,
                    "selected": sel_indices,
                    "rollback": rollback,
                    "raw": sel_raw.strip(),
                    "prompt": select_prompt,
                    "full_response": sel_raw,
                })

                # Remove selected from remaining pool
                selected_set = set(sel_indices)
                remaining = [i for i in remaining if i not in selected_set]

                # Stop if no rollback or max attempts reached
                if not rollback or attempt >= 2:
                    break

            # Deduplicate selected paths
            selected_paths = list(dict.fromkeys(selected_paths))
            # Fallback: if nothing selected, use top 3 by candidate count
            if not selected_paths:
                selected_paths = list(range(min(3, len(logical_paths))))

            _stage_times["path_select"] = time.perf_counter() - _t0_pathsel

            # Build final reasoning subgraph from selected patterns, grouped by pattern.
            selected_pattern_objs = []
            for idx in selected_paths:
                lp = logical_paths[idx]
                selected_pattern_objs.append(lp)
            selected_pattern_objs = materialize_selected_logical_patterns(
                selected_pattern_objs, ents, rels, h_ids, r_ids, t_ids,
                anchor_idx, breakpoint_indices,
            )
            selected_pattern_objs.extend(build_endpoint_rescue_patterns(
                paths, selected_pattern_objs, ents, rels, anchor_idx, breakpoint_indices,
            ))
            pat_evidence = build_pattern_evidence_triples(
                selected_pattern_objs, ents, rels, h_ids, r_ids, t_ids, anchor_idx,
                max_grouped_lines=120,
            )
            num_triples = sum(len(pe.triples) for pe in pat_evidence.values())

            if pat_evidence:
                _run_case_constraints = [ents[i] for i in breakpoint_indices if i is not None and 0 <= i < len(ents)]
                pattern_text = format_pattern_evidence(pat_evidence, constraint_entities=_run_case_constraints)

                reason_prompt = f"""QUESTION: {question}

GRAPH EVIDENCE (from Freebase, snapshot circa 2015):
{pattern_text}

━━━ REASONING TASK ━━━

Use the graph evidence to answer the question.
Reason step by step, but keep each step concise: 1 to 2 sentences.

STEP 1 — QUESTION UNDERSTANDING
Explain what the question is asking for, what the answer type is, and what constraint(s) must be satisfied.
Also state whether the last hop is the answer itself or only a verification condition.

STEP 2 — PATTERN COMPARISON
You must evaluate at least two candidate patterns if more than one is available.
For each pattern, say MATCH or MISMATCH and give one short reason why its relation chain semantically matches or mismatches the question.
Then choose the best matching pattern and briefly justify why it is better than the others.

STEP 3 — ANSWER POSITION AND CANDIDATES
Explain where the answer is located in the selected pattern (which hop/node).
List the candidate entities at that position, and mention the graph evidence connecting them.

STEP 4 — CONSTRAINT VERIFICATION AND ANSWER SELECTION

4a — IDENTIFY CONSTRAINTS (explicit AND implicit)

Determine the expected ANSWER TYPE from the question (person, country, language, event, year, etc.).
Identify ALL constraints:

EXPLICIT constraints: directly stated in the question (dates, locations, superlatives, quantities, conditions).

IMPLICIT constraints — apply these heuristics based on question semantics:
- UNIQUE ROLE/POSITION ("the governor", "the president", "the leader", "the capital") WITHOUT a time qualifier → implicit: prefer the MOST RECENT or CURRENT holder
- EVENTS/ACHIEVEMENTS/AWARDS ("wins", "championships", "movies", "albums", "titles") → NO implicit "current" constraint; return ALL matching instances
- ATTRIBUTES/PROPERTIES ("languages spoken", "religions practiced", "government type", "currency") → return ALL that apply
- GROUP MEMBERSHIP ("countries in X", "states bisected by Y", "members of") → return ALL matching members

CRITICAL: The singular/plural form of the question ALONE does not determine answer count. Use the heuristics above.

4b — CONSTRAINT CHECK
For each candidate, verify:
1. TYPE MATCH: Does this candidate match the expected ANSWER TYPE? Discard non-matching types (e.g., discard countries, dates, roles when question asks for languages).
2. EXPLICIT constraints: Does it satisfy all stated conditions?
3. IMPLICIT constraints: Does it satisfy the applicable heuristic?
  - Constraint: [description] → candidate A: PASS/FAIL, candidate B: PASS/FAIL, ...
Collect ALL candidates that pass ALL constraints → PASSING SET.
If graph evidence does NOT show a candidate fails, KEEP it.

4c — OUTPUT DECISION
- PASSING SET has 1 member → output it.
- PASSING SET has multiple + implicit "most recent" applies → output the MOST RECENT member by graph dates. If no dates in evidence, output ALL.
- PASSING SET has multiple + no implicit limit → output ALL members.
- NEVER discard a candidate solely because the question uses singular phrasing.

RULES:
- NEVER decide answer count before constraint checking. Evaluate ALL candidates first.
- Prefer graph evidence over intuition. NEVER use external knowledge to filter.
- The answer may appear at an intermediate hop, not necessarily the terminal node.
- For geographic constraints, only remove if graph evidence explicitly contradicts.
- For temporal constraints, look for date values in evidence. No dates → do NOT filter.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- When in doubt, output ALL passing candidates. Over-output is better than discarding valid answers.

━━━ EXAMPLES (illustrate METHOD only — do NOT copy answers) ━━━

Example A — TYPE FILTER + IMPLICIT "CURRENT" for unique role:
Q: "Who is the president of France?"
Candidates: [Emmanuel Macron, François Hollande, France, President, 2017-05-14]
→ Answer type: person (president). Type filter: FAIL=[France(country), President(role), 2017-05-14(date)]
→ Remaining: [Emmanuel Macron, François Hollande]. Implicit: unique role no time qualifier → most recent
→ Output: Emmanuel Macron

Example B — EVENTS/ACHIEVEMENTS → return ALL:
Q: "What movies did the actor who played Forrest Gump star in?"
Candidates: [Forrest Gump, Tom Hanks, Saving Private Ryan, Cast Away, Actor, 1994]
→ Answer type: movie. Type filter: FAIL=[Tom Hanks(person), Actor(role), 1994(year)]
→ Remaining: [Forrest Gump, Saving Private Ryan, Cast Away]. Implicit: events/works → ALL
→ Output: Forrest Gump | Saving Private Ryan | Cast Away

Example C — PATTERN SELECTION:
Q: "What countries border France?"
Pattern A: location.location.adjoining_countries (direct, 1-hop)
Pattern B: location.location.containedby → location.location.adjoining_countries (via region, 2-hop)
→ Choose Pattern A: direct semantic match, shorter path, fewer noise entities in candidates

━━━ OUTPUT FORMAT ━━━
<reasoning>
Step 1: 1-2 sentences.
Step 2: 1-2 sentences per pattern evaluated, then 1 sentence for choice.
Step 3: 1-2 sentences.
Step 4a: Answer type + explicit/implicit constraints. Step 4b: TYPE MATCH and PASS/FAIL per candidate. Step 4c: PASSING SET and output decision.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>

Multiple answers: <answer>\\boxed{{cand1}} \\boxed{{cand2}}</answer>
NO text after </answer> tag."""

                _t0_reason = time.perf_counter()
                llm_raw = await call_llm(session, [
                    {"role": "system", "content": "You are a precise graph QA system. Follow the 4-step reasoning process. ALWAYS answer. Compare patterns before choosing. First identify answer type and ALL constraints (explicit + implicit), then filter candidates by type match and constraint check. Apply implicit constraint heuristics: unique role → most recent; events/achievements → all; attributes → all; group membership → all. Singular/plural alone does NOT determine answer count. NEVER skip constraint verification. Exact graph strings only."},
                    {"role": "user", "content": reason_prompt},
                ], max_tokens=1800)
                # Extract answer from <answer>...\boxed{...}...</answer> format
                ans_match = re.search(r'<answer>(.*?)</answer>', llm_raw, re.DOTALL)
                if ans_match:
                    boxed = re.findall(r'\\boxed\{([^}]+)\}', ans_match.group(1))
                    if boxed:
                        llm_answer = " | ".join(b.strip() for b in boxed)
                        llm_hit = candidate_hit([b.strip() for b in boxed], gt_answers)
                    else:
                        llm_answer = ans_match.group(1).strip()
                        llm_hit = candidate_hit([llm_answer], gt_answers)
                else:
                    # Legacy fallback
                    ans_match = re.search(r'ANSWER:\s*(.+)', llm_raw, re.IGNORECASE)
                    if ans_match:
                        llm_answer = ans_match.group(1).strip()
                    else:
                        lines = [l.strip() for l in llm_raw.strip().split('\n') if l.strip()]
                        llm_answer = lines[-1] if lines else llm_raw.strip()
                    llm_hit = candidate_hit([llm_answer], gt_answers)
                llm_reasoning_prompt = reason_prompt
                llm_reasoning_full = llm_raw
                _stage_times["llm_reasoning"] = time.perf_counter() - _t0_reason
        except Exception as e:
            print(f"  LLM reasoning error: {type(e).__name__}: {e}")
            llm_answer = None
            llm_hit = False

    return {
        "case_id": pilot_row["case_id"],
        "question": question,
        "gt_answers": gt_answers,
        "decomposition_prompt": DECOMP_PROMPT,
        "decomposition_question": decomp_question,
        "decomposition": raw,
        "steps_parsed": steps,
        "anchor_idx": anchor_idx,
        "anchor_name": ents[anchor_idx] if anchor_idx is not None else None,
        "breakpoints": {k: ents[v] for k, v in breakpoints.items()},
        "step_relations": [list(r) for r in step_relations],
        "entity_retrieval_details": entity_retrieval_details,
        "relation_retrieval_details": relation_retrieval_details,
        "prune_debug": prune_debug_field,
        "layer_diagnostics": final_layer_diagnostics,
        "planning_attempts": planning_attempts,
        "max_depth": max_depth,
        "num_paths": len(paths),
        "answer_candidates": answer_candidates,
        "gt_hit": gt_hit,
        "num_patterns": len(logical_paths),
        "logical_paths": [lp["readable"] for lp in logical_paths[:10]],
        "pattern_details": [{
            "candidates": lp.get("candidates", []),
            "best_tier": lp.get("best_tier", (0, -1, 0)),
            "endpoint": lp.get("endpoint"),
            "path_count": lp.get("path_count", len(lp.get("raw_paths", []))),
            "materialized": lp.get("materialized", False),
            "witness_nodes": lp.get("best_raw_path", {}).get("nodes", []),
            "witness_relations": lp.get("best_raw_path", {}).get("relations", []),
        } for lp in logical_paths],
        "llm_answer": llm_answer,
        "llm_hit": llm_hit,
        "selected_paths": selected_paths,
        "num_triples": num_triples,
        "attempt_log": attempt_log,
        "llm_reasoning_prompt": llm_reasoning_prompt,
        "llm_reasoning_full": llm_reasoning_full,
        "stage_times": _stage_times,
    }


async def _run_case_wrapper(session, sample, pilot_row, sem, case_num, total_cases):
    """Wrapper to run a single case with semaphore control and error handling."""
    import time
    t0 = time.perf_counter()
    case_id = pilot_row.get('case_id', '?')

    # Use semaphore if provided (parallel mode), otherwise run directly
    if sem is not None:
        async with sem:
            return await _execute_case(session, sample, pilot_row, case_num, total_cases, t0)
    else:
        return await _execute_case(session, sample, pilot_row, case_num, total_cases, t0)


async def _execute_case(session, sample, pilot_row, case_num, total_cases, t0):
    """Execute a single case and return result with timing."""
    case_id = pilot_row.get('case_id', '?')

    try:
        result = await run_case(session, sample, pilot_row)
        dt = time.perf_counter() - t0

        if result is None:
            return None, dt, case_id, case_num

        status = "HIT" if result["gt_hit"] else "MISS"
        llm_status = "LLM_HIT" if result.get("llm_hit") else "LLM_MISS"
        n_patterns = result.get("num_patterns", 0)
        n_triples = result.get("num_triples", 0)
        cid = result.get('case_id', '?') or '?'

        # Print result as it completes
        print(f"[{case_num}/{total_cases}] {status} | {cid[:30]} | paths={result.get('num_paths',0):6d} | patterns={n_patterns:3d} | triples={n_triples:3d} | {llm_status} | time={dt:.2f}s | llm={str(result.get('llm_answer',''))[:25]} | gt={result.get('gt_answers')}")
        # Print stage timing
        st = result.get("stage_times", {})
        if st:
            parts = " | ".join(f"{k}={v*1000:.0f}ms" for k, v in st.items())
            print(f"  stages: {parts}")

        return result, dt, case_id, case_num

    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"[{case_num}/{total_cases}] ERROR: {case_id}: {type(e).__name__}: {e} (time={dt:.2f}s)")
        return None, dt, case_id, case_num
