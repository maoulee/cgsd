"""Stage 2: Entity resolution via GTE + LLM.

Provides NER anchor resolution (resolve_anchor_ner), LLM entity disambiguation
(llm_resolve_entity), and the batch entity resolution pipeline stage.
"""
from __future__ import annotations

import asyncio
import atexit
import os
import re
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from kgqa.core.case_state import CaseState
from kgqa.core.config import GTE_API_URL
from kgqa.core.utils import normalize, rel_to_text, extract_xml_tag, get_entity_contexts
from kgqa.llm.client import call_llm
from kgqa.llm.batch import batch_call_llm
from kgqa.traversal.cvt import is_cvt_like

import aiohttp


# ---------------------------------------------------------------------------
# GTE retrieval helper
# ---------------------------------------------------------------------------

# Generic doc-similarity task instruct for Qwen3-Embedding (case-free: no
# KG/attribute/answer-type wording). Validated to maximize gold-relation recall
# without overfitting; passed explicitly so it applies regardless of the server
# default (TASK_DESC).
GTE_INSTRUCT = "Given a query, retrieve the document most semantically similar to it"


# ---------------------------------------------------------------------------
# Round-level request collector (perf phase 3, 2026-08-23)
# ---------------------------------------------------------------------------
# 267x3 server-side profile: n_jobs=8724, encode_s=285.6 (GPU busy 4.8min) but
# infer_wait_s=8799 — the request COUNT, not the batch window, is the
# bottleneck. Same fix shape as the walk coordinator (kgqa/agent/
# seq_tools.py): concurrent gte_retrieve calls collect in an ADAPTIVE
# two-stage window — first slice GTE_CLIENT_BATCH_FIRST; a second arrival
# within the slice extends the deadline to the full window
# GTE_CLIENT_BATCH_WINDOW (counted from the first arrival); a LONE request
# flushes right after the first slice (no fixed window tax). One flush = ONE
# /retrieve_batch POST, results distribute back per request. Identical
# (pool, query, top_k, instruct) requests share one in-flight future (G=3
# lockstep emits identical prelink/early queries), and a completed-memo
# serves repeats with zero HTTP. DETERMINISM: the same key returns the SAME
# result object — callers must treat rows as read-only (all current callers
# only read r.get(...) fields).
# Env: GTE_CLIENT_BATCH_WINDOW (full window s, default 1.0; <=0 = legacy
# per-request POST), GTE_CLIENT_BATCH_FIRST (first slice s, default 0.25),
# GTE_CLIENT_MEMO_MAX (completed-memo bound, default 50000).
_GTE_REQ_BATCH = None                    # open collection batch or None
_GTE_MEMO = OrderedDict()                # key -> results (completed, bounded)
_GTE_MEMO_MAX = int(os.environ.get("GTE_CLIENT_MEMO_MAX", "50000") or 0)
_GTE_BATCH_STATS = {"flushes": 0, "single": 0, "burst": 0, "reqs": 0,
                    "dedup": 0, "memo_hits": 0, "fallback": 0,
                    "http_items": 0, "collect_s": 0.0}


def _gte_batch_summary():
    st = _GTE_BATCH_STATS
    if st["reqs"]:
        print(f"  gte-batch(client): {st['flushes']} flushes "
              f"(single={st['single']} burst={st['burst']}) reqs={st['reqs']} "
              f"http_items={st['http_items']} dedup={st['dedup']} "
              f"memo={st['memo_hits']} fallback={st['fallback']} "
              f"collect={st['collect_s']:.1f}s", flush=True)


atexit.register(_gte_batch_summary)


def _gte_req_key(pool_key, candidates, candidate_texts, query, top_k, instruct):
    """Identity of one retrieval: pool (registry key, or the value-shipped
    candidate list) x query x top_k x instruct. Equal keys MUST get equal
    results, so they dedup onto one server item / one memo entry."""
    if pool_key is not None:
        pool = ("k", pool_key)
    else:
        pool = ("v", tuple(candidates or []),
                tuple(candidate_texts or ()))
    return (pool, query, top_k, instruct)


def _gte_open_batch(loop):
    """Open the collection batch and arm the first-slice callback."""
    global _GTE_REQ_BATCH
    first = float(os.environ.get("GTE_CLIENT_BATCH_FIRST", "0.25") or 0)
    b = {"loop": loop, "reqs": [], "by_key": {}, "t_first": time.perf_counter()}
    b["cb"] = loop.call_later(max(first, 0.001), _gte_flush_check)
    _GTE_REQ_BATCH = b
    return b


def _gte_flush_check():
    """End of the FIRST slice: flush now unless a second request arrived
    within it — then extend to the FULL window from the first arrival (the
    round's time cluster still coalesces)."""
    b = _GTE_REQ_BATCH
    if b is None:
        return
    if len(b["reqs"]) > 1:
        full = float(os.environ.get("GTE_CLIENT_BATCH_WINDOW", "1.0") or 0)
        remain = (b["t_first"] + full) - time.perf_counter()
        if remain > 0:
            b["loop"].call_later(remain, _gte_flush_now)
            return
    _gte_flush_now()


def _gte_flush_now():
    """Close the open batch (later arrivals open a fresh one) and hand it to
    the async flusher."""
    global _GTE_REQ_BATCH
    b, _GTE_REQ_BATCH = _GTE_REQ_BATCH, None
    if b is not None and b["reqs"]:
        b["loop"].create_task(_gte_flush_async(b))


def _gte_memo_put(key, rows):
    if key in _GTE_MEMO:
        _GTE_MEMO.move_to_end(key)
        return
    if _GTE_MEMO_MAX > 0:
        while len(_GTE_MEMO) >= _GTE_MEMO_MAX:
            _GTE_MEMO.popitem(last=False)
    _GTE_MEMO[key] = rows


async def _gte_flush_async(batch):
    """One flush = ONE /retrieve_batch POST for every collected request;
    same-key requests resolve to the SAME result object. A missing endpoint
    (stale server) or batch-transport failure falls back to the per-request
    path; any remaining failure raises to every waiter (callers' retry loops
    re-enter the collector)."""
    from kgqa.core.utils import PHASE_TIMES
    reqs = batch["reqs"]
    session = batch.get("session")
    st = _GTE_BATCH_STATS
    t_flush = time.perf_counter()
    st["flushes"] += 1
    st["single" if len(reqs) == 1 else "burst"] += 1
    st["reqs"] += len(reqs)
    st["collect_s"] += sum(t_flush - r["t0"] for r in reqs)
    PHASE_TIMES.setdefault("gte_collect", 0.0)
    PHASE_TIMES["gte_collect"] += sum(t_flush - r["t0"] for r in reqs)
    # POST wall (flush start → results distributed) — flushes overlap each
    # other/walk, so the SUM upper-bounds the gte share of the dispatch wall.
    _t_post0 = time.perf_counter()
    try:
        results = None
        if session is not None:
            items = [{"query": r["query"], "candidates": r["candidates"],
                      "candidate_texts": r["candidate_texts"],
                      "top_k": r["top_k"], "instruct": r["instruct"],
                      "pool_key": r["pool_key"]} for r in reqs]
            try:
                async with session.post(f"{GTE_API_URL}/retrieve_batch",
                                        json={"items": items},
                                        timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 200:
                        results = (await resp.json()).get("results")
            except Exception:
                results = None   # batch transport failed → per-request below
        if results is None:
            st["fallback"] += 1
            results = await asyncio.gather(*[
                _gte_post_one(session, r["query"], r["candidates"],
                              r["candidate_texts"], r["top_k"], r["instruct"],
                              r["pool_key"]) for r in reqs])
        if len(results) != len(reqs):
            raise RuntimeError(
                f"/retrieve_batch returned {len(results)} results "
                f"for {len(reqs)} requests")
        st["http_items"] += len(reqs)
        for r, rows in zip(reqs, results):
            rows = rows or []
            _gte_memo_put(r["key"], rows)
            if not r["fut"].done():
                r["fut"].set_result(rows)
    except Exception as e:
        for r in reqs:
            if not r["fut"].done():
                r["fut"].set_exception(e)
    finally:
        PHASE_TIMES["gte_post_wall"] = \
            PHASE_TIMES.get("gte_post_wall", 0.0) + (time.perf_counter() - _t_post0)


async def _gte_collect(session, key, query, candidates, candidate_texts,
                       top_k, instruct, pool_key):
    """Enter the open collection batch (opening one if needed) and await the
    flush's result. Same-key callers share ONE future."""
    loop = asyncio.get_running_loop()
    b = _GTE_REQ_BATCH
    if b is None or b["loop"] is not loop:
        # a new event loop (the rollout runs one asyncio.run per stage) must
        # not enqueue onto a dead loop's batch — its callbacks never fire.
        b = _gte_open_batch(loop)
    fut = b["by_key"].get(key)
    if fut is None:
        fut = loop.create_future()
        b["by_key"][key] = fut
        b["reqs"].append({
            "key": key, "fut": fut, "t0": time.perf_counter(),
            "query": query, "candidates": list(candidates or []),
            "candidate_texts": list(candidate_texts) if candidate_texts else None,
            "top_k": top_k, "instruct": instruct, "pool_key": pool_key,
        })
    else:
        _GTE_BATCH_STATS["dedup"] += 1
    if session is not None:
        b.setdefault("session", session)
    return await fut


async def _gte_post_one(session, query, candidates, candidate_texts, top_k,
                        instruct, pool_key):
    """Legacy per-request path (also the batch-transport fallback): one POST
    per call — the pre-2026-08-23 request, byte-for-byte."""
    payload = {"query": query, "candidates": candidates,
               "candidate_texts": candidate_texts, "top_k": top_k,
               "instruct": instruct}
    if pool_key is not None:
        payload["pool_key"] = pool_key
    async with session.post(f"{GTE_API_URL}/retrieve", json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)) as resp:
        data = await resp.json()
    return data.get("results", [])


async def gte_retrieve(session, query, candidates, candidate_texts=None, top_k=10, instruct=GTE_INSTRUCT, pool_key=None):
    """pool_key (2026-08-21): register-once candidate pools. First call sends
    candidates (server registers under pool_key); later calls with the same
    key may pass candidates=[] — no 10-30KB re-transfer/re-parse. The same
    (head, pool) is re-queried across facts with only the query changing.
    Batch-collected (2026-08-23): concurrent calls share one adaptive
    collection window and one /retrieve_batch POST (see the collector note
    above); signature and per-call semantics unchanged."""
    from kgqa.core.utils import PHASE_TIMES
    PHASE_TIMES["gte_n"] = PHASE_TIMES.get("gte_n", 0.0) + 1
    if float(os.environ.get("GTE_CLIENT_BATCH_WINDOW", "1.0") or 0) <= 0:
        return await _gte_post_one(session, query, candidates, candidate_texts,
                                   top_k, instruct, pool_key)
    key = _gte_req_key(pool_key, candidates, candidate_texts, query, top_k, instruct)
    hit = _GTE_MEMO.get(key)
    if hit is not None:
        _GTE_MEMO.move_to_end(key)
        _GTE_BATCH_STATS["memo_hits"] += 1
        return hit
    return await _gte_collect(session, key, query, candidates, candidate_texts,
                              top_k, instruct, pool_key)


# ---------------------------------------------------------------------------
# NER + GTE scoring for entity resolution
# ---------------------------------------------------------------------------

def _token_overlap(question: str, entity: str) -> float:
    """Ratio of entity tokens found in question."""
    q_tokens = set(normalize(question).split())
    e_tokens = set(normalize(entity).split())
    if not e_tokens:
        return 0.0
    return len(e_tokens & q_tokens) / len(e_tokens)


async def resolve_anchor_ner(session, question, entity_list, rel_list, h_ids, r_ids, t_ids):
    """NER + GTE + token overlap + relation overlap scoring for entity resolution."""
    clean_ents, seen = [], set()
    for e in entity_list:
        if not e or is_cvt_like(e) or len(e) <= 1:
            continue
        if e not in seen:
            clean_ents.append(e)
            seen.add(e)

    rel_texts = [rel_to_text(r) for r in rel_list]

    # GTE entity retrieval
    q_ent_rows = await gte_retrieve(session, question, clean_ents, top_k=12)
    gte_ents = [(r["candidate"], r.get("score", 0)) for r in q_ent_rows
                if r.get("candidate") and r["candidate"] in entity_list]

    # Token overlap filter (min 0.5)
    filtered = [(e, s) for e, s in gte_ents if _token_overlap(question, e) >= 0.5]

    # Relation overlap scoring
    q_rel_rows = await gte_retrieve(session, question, rel_list,
                                    candidate_texts=rel_texts, top_k=5)
    q_top_rel_idx = {rel_list.index(r["candidate"]) for r in q_rel_rows
                     if r.get("candidate") and r["candidate"] in rel_list}

    name_to_ids = defaultdict(list)
    for i, name in enumerate(entity_list):
        name_to_ids[name].append(i)

    scored = []
    for ent, gte_score in filtered:
        ent_idx = set(name_to_ids.get(ent, []))
        ent_rels = set()
        for h, r, t in zip(h_ids, r_ids, t_ids):
            if h in ent_idx or t in ent_idx:
                ent_rels.add(r)
        overlap = len(ent_rels & q_top_rel_idx)
        scored.append({"entity": ent, "gte": round(gte_score, 4), "overlap": overlap})
    scored.sort(key=lambda x: (-x["overlap"], -x["gte"]))

    return scored, name_to_ids


# ---------------------------------------------------------------------------
# LLM entity disambiguation
# ---------------------------------------------------------------------------

async def llm_resolve_entity(session, question, query, candidates_with_ctx):
    """LLM selects the correct entity from GTE top-k candidates using surrounding relation context.
    candidates_with_ctx: list of (name, context_str) tuples.
    Returns selected entity name or None."""
    if not candidates_with_ctx:
        return None
    if len(candidates_with_ctx) == 1:
        return candidates_with_ctx[0][0]

    cand_lines = []
    for i, (name, ctx) in enumerate(candidates_with_ctx, 1):
        cand_lines.append(f"  {i}. {name} [{ctx}]" if ctx else f"  {i}. {name}")

    prompt = f"""Search query: {query}

Candidate entities (with relation context from knowledge graph):
{chr(10).join(cand_lines)}

Which candidate best matches the search query? Use the relation context to identify what each entity actually IS (a person, a location, a schema type, etc). Pick the specific entity, not generic types or schema entries.

<analysis>Brief reasoning about which candidate matches the query</analysis>
<selected>entity name</selected>"""

    for _ in range(2):
        raw = await call_llm(session, [
            {"role": "system", "content": "Select the correct entity from candidates. Output <analysis> and <selected> XML tags."},
            {"role": "user", "content": prompt},
        ], max_tokens=300)
        sel = extract_xml_tag(raw, "selected")
        if sel:
            sel = sel.strip().strip('"').strip("'")
            # Match to candidate names
            for name, _ in candidates_with_ctx:
                if normalize(sel) == normalize(name):
                    return name
            # Fuzzy match
            for name, _ in candidates_with_ctx:
                if normalize(sel) in normalize(name) or normalize(name) in normalize(sel):
                    return name
    # Fallback to GTE top-1
    return candidates_with_ctx[0][0]


# ---------------------------------------------------------------------------
# Stage 2: Batch entity resolution
# ---------------------------------------------------------------------------

async def stage_2_entity_resolution(session, cases: List[CaseState]):
    """Batch entity resolution. Flatten all requests across cases into one batch."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # Phase A: GTE retrieval for all entity queries concurrently
    gte_tasks = []  # (kind, cs, query_str)
    for cs in active:
        if not (cs.use_ner and cs.ner_top_ents) and cs._pending_anchor_eq:
            gte_tasks.append(("anchor", cs, cs._pending_anchor_eq))
        for ep in cs._pending_endpoints:
            gte_tasks.append(("endpoint", cs, ep["query"]))

    _gte_sem = asyncio.Semaphore(3)
    async def _gte_limited(coro):
        async with _gte_sem:
            return await coro
    gte_results = await asyncio.gather(*[
        _gte_limited(gte_retrieve(session, query, cs.ent_candidates, top_k=5 if kind == "anchor" else 3))
        for kind, cs, query in gte_tasks
    ], return_exceptions=True)

    # Phase B: Build LLM prompts for entity resolution
    llm_items = []  # (cs, kind, step_idx, cands_with_ctx)
    for i, (kind, cs, query) in enumerate(gte_tasks):
        result = gte_results[i]
        if isinstance(result, Exception):
            continue
        candidates = [r.get("candidate", "") for r in result if r.get("candidate")]
        if len(candidates) <= 1:
            if kind == "anchor" and candidates:
                cs.anchor_name = candidates[0]
                cs.anchor_idx = cs.ents.index(candidates[0]) if candidates[0] in cs.ents else None
            continue

        ctx = get_entity_contexts(candidates, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents, cs.rels)
        cands_with_ctx = [(n, ctx.get(n, "")) for n in candidates]
        llm_items.append((cs, kind, query, cands_with_ctx))

    # Phase C: Batch LLM call
    if llm_items:
        prompts = []
        for cs, kind, query, cands_with_ctx in llm_items:
            cand_lines = []
            for j, (name, c) in enumerate(cands_with_ctx, 1):
                cand_lines.append(f"  {j}. {name} [{c}]" if c else f"  {j}. {name}")
            prompt = f"""Search query: {query}

Candidate entities (with relation context from knowledge graph):
{chr(10).join(cand_lines)}

Which candidate best matches the search query? Use the relation context to identify what each entity actually IS.

<analysis>Brief reasoning about which candidate matches the query</analysis>
<selected>entity name</selected>"""
            prompts.append([
                {"role": "system", "content": "Select the correct entity from candidates. Output <analysis> and <selected> XML tags."},
                {"role": "user", "content": prompt},
            ])

        responses = await batch_call_llm(session, prompts, max_tokens=300)

        for (cs, kind, query, cands_with_ctx), raw in zip(llm_items, responses):
            sel = extract_xml_tag(raw or "", "selected")
            selected = None
            if sel:
                sel = sel.strip().strip('"').strip("'")
                for name, _ in cands_with_ctx:
                    if normalize(sel) == normalize(name):
                        selected = name; break
                if not selected:
                    for name, _ in cands_with_ctx:
                        if normalize(sel) in normalize(name) or normalize(name) in normalize(sel):
                            selected = name; break
            if not selected and cands_with_ctx:
                selected = cands_with_ctx[0][0]

            if kind == "anchor":
                cs.anchor_name = selected
                cs.anchor_idx = cs.ents.index(selected) if selected and selected in cs.ents else None
                cs.entity_retrieval_details.append({
                    "role": "anchor", "query": query, "selected": selected,
                    "selected_idx": cs.anchor_idx, "llm_resolved": True,
                })
            else:
                # Find which endpoint this resolves
                for ep in cs._pending_endpoints:
                    if ep["query"] == query:
                        step_idx = ep["step_idx"]
                        idx = cs.ents.index(selected) if selected and selected in cs.ents else None
                        if idx is not None:
                            cs.breakpoints[step_idx] = idx
                        cs.entity_retrieval_details.append({
                            "role": f"endpoint_step{step_idx}", "query": query,
                            "selected": selected, "selected_idx": idx, "llm_resolved": True,
                        })
                        break

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["entity_resolve"] = dt / len(active)
    print(f"  Stage 2 (Entity resolve): {dt:.2f}s | {sum(1 for cs in active if cs.anchor_idx is not None)} anchored")
