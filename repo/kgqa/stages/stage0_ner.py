"""Stage 0: NER entity resolution using GTE retrieval.

GTE-only entity resolution with optional skip mode that uses q_entity directly.
No LLM calls in this stage.
"""
from __future__ import annotations

import asyncio
import time
from typing import List

from kgqa.core.case_state import CaseState
from kgqa.core.config import SKIP_NER
from kgqa.traversal.cvt import is_cvt_like, expand_cvt_leaves
from kgqa.stages.stage2_entity import resolve_anchor_ner


async def stage_0_ner_resolve(session, cases: List[CaseState], skip_ner=False):
    """GTE-only NER entity resolution. No LLM calls.
    When skip_ner=True, skips GTE call and uses q_entity directly.
    """
    _t0 = time.perf_counter()

    async def _resolve_one(cs: CaseState):
        cs.ents = cs.sample.get("text_entity_list", []) + cs.sample.get("non_text_entity_list", [])
        cs.rels = list(cs.sample.get("relation_list", []))
        cs.h_ids = cs.sample.get("h_id_list", [])
        cs.r_ids = cs.sample.get("r_id_list", [])
        cs.t_ids = cs.sample.get("t_id_list", [])

        if skip_ner:
            cs.use_ner = False
            q_ents = cs.sample.get("q_entity", [])
            # Filter out CVT nodes from q_entity
            q_ents = [e for e in q_ents if not is_cvt_like(e)]
            cs.ner_scored = [{"entity": e, "gte": 1.0} for e in q_ents]
            cs.ner_top_ents = [(e, 1.0) for e in q_ents]
        else:
            cs.ner_scored, _ = await resolve_anchor_ner(
                session, cs.question, cs.ents, cs.rels, cs.h_ids, cs.r_ids, cs.t_ids)
            cs.ner_top_ents = []
            _seen = set()
            for s in cs.ner_scored[:6]:
                if s["entity"] not in _seen:
                    cs.ner_top_ents.append((s["entity"], s["gte"]))
                    _seen.add(s["entity"])

        # CVT expansion
        cs.ents, cs.rels, cs.h_ids, cs.r_ids, cs.t_ids = expand_cvt_leaves(
            cs.ents, cs.rels, cs.h_ids, cs.r_ids, cs.t_ids)
        cs.rel_texts = list(cs.rels)
        cs.ent_candidates = [e for e in cs.ents if e and len(e) > 1 and not is_cvt_like(e)]

        cs.ner_name_to_ids_expanded = {}
        for i, name in enumerate(cs.ents):
            cs.ner_name_to_ids_expanded.setdefault(name, []).append(i)

    if skip_ner:
        await asyncio.gather(*[_resolve_one(cs) for cs in cases])
    else:
        _gte_sem = asyncio.Semaphore(3)
        async def _resolve_one_limited(cs):
            async with _gte_sem:
                return await _resolve_one(cs)
        await asyncio.gather(*[_resolve_one_limited(cs) for cs in cases])

    dt = time.perf_counter() - _t0
    for cs in cases:
        cs.stage_times["ner_resolve"] = dt / len(cases)
    mode_str = "skip-NER (q_entity)" if skip_ner else "GTE NER"
    print(f"  Stage 0 ({mode_str}): {dt:.2f}s for {len(cases)} cases")
