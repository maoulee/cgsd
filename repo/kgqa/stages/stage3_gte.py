"""Stage 3: GTE relation retrieval for decomposition steps.

Concurrent GTE retrieval across all steps of all active cases,
aggregating results from multiple query fields (definition, subquestion,
question, keyword, relation_query) plus the original question.
"""
from __future__ import annotations

import asyncio
import time
from typing import List

from kgqa.core.case_state import CaseState
from kgqa.core.utils import rel_to_text
from kgqa.stages.stage2_entity import gte_retrieve


async def stage_3_gte_relation_retrieval(session, cases: List[CaseState]):
    """Concurrent GTE relation retrieval for all steps of all cases."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # Build flat list of all GTE calls: (cs, step, query)
    # Queries: definition, subquestion + step question + original question (matching V2)
    gte_calls = []
    step_query_counts = []  # track per-(cs, step) query count for result distribution
    for cs in active:
        cs_qcounts = []
        for step in cs.steps:
            queries = []
            for field in ["definition", "subquestion"]:
                val = step.get(field, "")
                if val and val.strip():
                    queries.append(val.strip())
            # Add step question (chain relation intent)
            if step.get("question") and step["question"].strip():
                queries.append(step["question"].strip())
            # Add original question as stable semantic anchor
            if cs.question and cs.question.strip():
                queries.append(cs.question.strip())
            # Deduplicate (case-insensitive)
            seen = set()
            n = 0
            for q in queries:
                ql = q.lower()
                if ql not in seen:
                    seen.add(ql)
                    gte_calls.append((cs, step, q))
                    n += 1
            cs_qcounts.append(n)
        step_query_counts.append(cs_qcounts)

    _gte_sem3 = asyncio.Semaphore(3)
    async def _gte_limited3(coro):
        async with _gte_sem3:
            return await coro
    gte_results = await asyncio.gather(*[
        _gte_limited3(gte_retrieve(session, query, cs.rels, candidate_texts=cs.rel_texts, top_k=15))
        for cs, step, query in gte_calls
    ], return_exceptions=True)

    # Distribute results back
    call_idx = 0
    for cs_idx, cs in enumerate(active):
        cs.step_candidates = {}
        cs.gte_per_step = {}
        cs.relation_retrieval_details = []
        for step_idx, step in enumerate(cs.steps):
            n_queries = step_query_counts[cs_idx][step_idx]
            gte_all = {}
            queries_detail = []
            for qi in range(n_queries):
                result = gte_results[call_idx]
                query_text = gte_calls[call_idx][2] if call_idx < len(gte_calls) else ""
                call_idx += 1
                if isinstance(result, Exception):
                    queries_detail.append({"query": query_text, "top_k": [], "error": str(result)})
                    continue
                topk = []
                for i, r in enumerate(result):
                    cand = r.get("candidate", "")
                    score = round(r.get("score", 0), 4)
                    idx_in_rels = cs.rels.index(cand) if cand in cs.rels else None
                    topk.append({"rank": i + 1, "candidate": cand, "score": score,
                                 "rel_idx": idx_in_rels,
                                 "rel_text": rel_to_text(cand) if idx_in_rels is not None else ""})
                    if idx_in_rels is not None:
                        if idx_in_rels not in gte_all or score > gte_all[idx_in_rels][1]:
                            gte_all[idx_in_rels] = (cs.rels[idx_in_rels], score)
                queries_detail.append({"query": query_text, "top_k": topk})

            gte_candidates = sorted(gte_all.items(), key=lambda x: -x[1][1])
            candidate_list = [(idx, name, score) for idx, (name, score) in gte_candidates]
            cs.step_candidates[step["step"]] = candidate_list
            cs.gte_per_step[step["step"]] = gte_all
            cs.relation_retrieval_details.append({
                "step": step["step"], "queries": queries_detail,
                "gte_candidates_count": len(gte_all),
                "gte_indices": sorted(gte_all.keys()),
            })

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["gte_retrieve"] = dt / len(active)
    print(f"  Stage 3 (GTE relations): {dt:.2f}s | {call_idx} calls")
