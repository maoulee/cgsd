"""Tool JSON schemas + dispatch for the native tool-calling agent.

Schemas are the surface the model sees; dispatch executes each accepted tool
against a CaseContext using the ROBUST shared engines (GTE, k_queue_traverse,
compress_paths) — imported, never modified.

Key design: `retrieve` is MODEL-DRIVEN — the model's `subquestion` is the GTE
query (not the raw question). This is the fix for the 46-47% relation-retrieval
miss diagnosed in the stage pipeline.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List

# Structural prune was REMOVED (3-run multi-sample A/B, n=45 heldout): the
# chained-scope intersection was net-negative — F1 0.7438 with-prune vs 0.7649
# no-prune, recall 115 vs 121 gold-reached. The scope demands an accurate chain,
# and a contaminated chain (an upstream fact's noisy candidate consumes a
# downstream answer entity) cuts the bridging relation the last hop needs (e.g.
# WebQTest-1483: form_of_government, which GTE ranked #14, was structurally
# pruned). After the G1 instruct optimization GTE recall is strong enough that
# raw top-K suffices. The scope is still computed (scope_rel_ids) and used ONLY
# for candidate-text contextualization (a GTE ranking aid), never for pruning.
#
# GTE top-K (default 30). top-15 was tested and DROPS F1 to 0.6776 (too tight,
# recall 112) — keep 30.
_GTE_TOPK = int(os.environ.get("KGQA_GTE_TOPK", "30"))

from kgqa.core.utils import normalize, candidate_hit, strip_reasoning_leak
from kgqa.stages.stage2_entity import gte_retrieve
from kgqa.traversal.k_queue import k_queue_traverse
from kgqa.traversal.frontier import relation_prior_expand
from kgqa.traversal.cvt import is_cvt_like
from kgqa.stages.stage5_traverse import stage_5_graph_traversal
from kgqa.stages.formatting import build_pattern_evidence_triples, _render_path_tree
from kgqa.core.case_state import CaseState
from kgqa.traversal.path_utils import compress_paths
from kgqa.traversal.logical_paths import materialize_selected_logical_patterns


def _candidates_from_triples(triples, anchor_name: str):
    """Pull answer-entity names out of a PatternEvidence's triples.

    build_pattern_evidence_triples expands CVT mediators into triples like
    (CVT, office_holder, Kasich), so the answer NAME lives as the object of a
    CVT-headed triple. compress_paths, by contrast, only collects candidates
    from a path's non-CVT nodes — so CVT-terminated paths (anchor -> CVT, no
    further non-CVT node in the path) end up with an EMPTY candidate list even
    though the office_holder is present in the triples/tree.

    Rules:
    - take the OBJECT of any triple whose head is a CVT (the CVT attribute value),
    - also take a non-CVT, non-anchor object of a forward (non-CVT) triple,
    - skip CVT ids themselves, the anchor, and bare years/short tokens.
    Returns a deduped list preserving first-seen order.
    """
    anchor_norm = normalize(anchor_name) if anchor_name else ""
    seen = set()
    out = []
    for h, r, t in triples:
        for name in (t, h):
            if not name or is_cvt_like(name):
                continue
            if anchor_norm and normalize(name) == anchor_norm:
                continue
            nc = normalize(name)
            if len(nc) < 2:
                continue
            # Skip bare years / pure numbers (CVT value noise like from=2011).
            cleaned = nc.replace(" ", "")
            if cleaned.isdigit() and len(cleaned) <= 4:
                continue
            if nc not in seen:
                seen.add(nc)
                out.append(name)
    return out


# ---------------------------------------------------------------------------
# Tool JSON schemas (OpenAI tool-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "decompose",
            "description": (
                "Decompose the question into an INFORMATION FLOW of directed triples. "
                "Each triple is one hop (head | relation | tail); triples thread via "
                "shared ?variables to flow from the named ANCHORS to the ANSWER. "
                "MUST be the first tool call. The system runs GTE per triple (the "
                "relation clause is the hint) and walks each anchor along the flow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flow": {
                        "type": "array",
                        "description": ("Directed triples describing how information flows "
                                        "from the anchors to the answer. Thread via shared "
                                        "?variables: the tail of one triple is the head of "
                                        "the next. relation = a semantic clause (e.g. "
                                        "'shares a border with'); never a bare copula. Use a "
                                        "?variable for any unstated intermediate or the answer."),
                        "items": {
                            "type": "array",
                            "description": "[head, relation, tail]",
                            "items": {},
                            "minItems": 3,
                        },
                        "minItems": 1,
                    },
                    "entities": {
                        "type": "array",
                        "description": ("EVERY named entity the question names — the proper "
                                        "nouns the flow starts from (people, places, "
                                        "countries, orgs; NOT type words, NOT numbers, NOT "
                                        "?variables). List them ALL: each entity becomes its "
                                        "own converging walk to the answer. Required (>= 1)."),
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "answer": {
                        "type": "string",
                        "description": ("The ?variable that is the flow's terminal/goal "
                                        "(the thing the question asks 'what/which X' for), "
                                        "plus its type, e.g. '?country (a country)'."),
                    },
                },
                "required": ["flow", "entities", "answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Retrieve candidate entities for ONE fact. The system uses your "
                "subquestion as the retrieval query against KG relations, then "
                "walks the graph from the anchor. Call this once per fact_id, "
                "after decompose, before select."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_id": {"type": "string",
                                "description": "The id of the fact to retrieve (from decompose)."},
                    "subquestion": {
                        "type": "string",
                        "description": (
                            "a rephrased sub-question for this fact, closer to the "
                            "question's own words (the retrieval query)"
                        ),
                    },
                },
                "required": ["fact_id", "subquestion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_relations",
            "description": (
                "Each fact from decompose lists its candidate_relations next to "
                "its text/subquestion (the step's sub-question). Read each "
                "fact's text, then select ALL candidate relations that express "
                "what that step asks for — the system only guarantees the "
                "candidates are structurally reachable, you decide which ones "
                "actually answer the step. Call ONCE after decompose; the system "
                "traverses the chosen relations and returns the evidence tree."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selections": {
                        "type": "array",
                        "description": "One entry per fact.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fact_id": {"type": "string"},
                                "relations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Relation names chosen from that fact's candidate_relations.",
                                },
                            },
                            "required": ["fact_id", "relations"],
                        },
                    },
                },
                "required": ["selections"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_branches",
            "description": (
                "Expand one or more branches of the evidence tree to see their "
                "detailed candidates and CVT attributes. Pass the branch numbers "
                "whose overview looks relevant to the question, in a single batch "
                "call. You may also skip directly to answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "branch_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Branch numbers from the tree overview, e.g. ['1','2']. Pass all branches you want to expand in one call.",
                    },
                },
                "required": ["branch_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": (
                "Emit the final answer entities, copied verbatim from the evidence. "
                "Call LAST. The entities MUST come from the retrieved candidates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity-name strings, verbatim from evidence.",
                    },
                },
                "required": ["entities"],
            },
        },
    },
]


def tools_for_state(state_name: str) -> List[Dict[str, Any]]:
    """Return only the schema(s) allowed in the given harness state.

    Restricting the offered tools each turn makes the model's job trivial and
    makes out-of-order calls impossible at the API layer too.
    """
    allowed = {
        "INIT": {"decompose"},
        "RETRIEVE": {"retrieve", "select_relations"},  # retrieve = optional fallback
        "SELECT_RELATIONS": {"select_relations", "retrieve"},  # retrieve = fallback
        "EXPAND": {"expand_branches", "answer"},
        "ANSWER": {"answer"},
        "DONE": set(),
    }
    names = allowed.get(state_name, set())
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] in names]


# ---------------------------------------------------------------------------
# Dispatch — execute an accepted tool against a CaseContext
# ---------------------------------------------------------------------------


def _json_result(payload: Any) -> str:
    """Serialize a dispatch result to a compact JSON string for the tool role."""
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"error": "non-serializable result"}, ensure_ascii=False)


async def dispatch(tool_name: str, args: Dict[str, Any], ctx, session) -> str:
    """Execute one accepted tool. Returns a JSON-serializable result string."""
    if tool_name == "decompose":
        return await _do_decompose(args, ctx, session)

    if tool_name == "retrieve":
        return await _do_retrieve(args, ctx, session)

    if tool_name == "select_relations":
        return await _do_select_relations(args, ctx, session)

    if tool_name in ("expand_branch", "expand_branches"):
        return await _do_expand_branch(args, ctx, session)

    if tool_name == "answer":
        return _do_answer(args, ctx)

    return _json_result({"error": f"unknown tool: {tool_name}"})


# ── Matched GTE for the triple decompose ──
# This is the GTE debugged alongside scripts/clean_decompose_prompt.txt: the
# query AND candidates are LABELED TRIPLES "head:(h) relation:(r) tail:(t)" with
# a relation-semantics instruct, top-15. The full triple context (not a bare
# relation clause) is what makes "played for ?team" match the TEAM relation
# (sports.pro_athlete.teams) instead of the athlete relation — the failure mode
# where a terse hint drifted to the wrong relation surface.
_TRIPLE_GTE_INSTRUCT = ("Given a natural-language question about the head entity, "
                        "retrieve the knowledge-graph relation whose semantics match "
                        "what the question asks — the relation must connect the head "
                        "entity to the answer.")
# V3 semantic-match instruct (2026-08-22, user ruling 本质是关系语义是否与问题相符合):
# validated on the gold-relation suite (195 cases, production last2+head format):
# recall@5 0.231 vs V2's 0.210, @10/@15 parity (0.333/0.385 vs 0.328/0.385);
# MLK specimen film-bias: base phrasing film.* 9→6 slots, location family
# 6/7/11→2/5/7; "where" phrasing location family returns to the window
# (12th) from absent. Replaces V2 (whose "question's topic" wording coupled
# "shot"→film corpus and amplified paraphrase variance).
# lever that rescues the Iraq specimen (form_of_government #25 -> top-5).
_TRIPLE_GTE_TOPK = 15
_REGISTERED_POOLS = set()   # pool_keys registered with the GTE server (process lifetime)
# Grounded-GTE pool-size gate: each hop's candidate pool is its 2-hop-reachable
# relation set. If that set is smaller than _GTE_POOL_MIN (sparse anchor or a
# broken chain — too narrow to rank meaningfully), the hop expands to the full
# relation set. Tuned to keep sparse-anchor cases ranked.
_GTE_POOL_MIN = 8



def _rel_last2(rel_id: str) -> str:
    segs = str(rel_id).split(".")
    # LAST-3 (user ruling 2026-09-15, GTE decomposition experiment):
    # last-2 severed the domain prefix — "adjoining_relationship.adjoins"
    # without "location" ranked #13 for "border"; last-1/last-3/full all
    # rank #1. The domain prefix carries the semantic context the embedding
    # needs. Keep last-3 (domain.type.attribute) with underscores→spaces.
    return " ".join(p.replace("_", " ") for p in segs[-3:]) if len(segs) >= 3 else str(rel_id).replace("_", " ")


async def _gte_for_triple(ctx, session, head, rel_clause, tail, pool_relids=None):
    """Labeled-triple GTE (matched with the triple decompose). Returns matched
    relation indices (top-_TRIPLE_GTE_TOPK). The /retrieve service always returns
    top-K rows by similarity (never empty) — any "0 candidates" is a mapping bug,
    so this maps each row robustly via index (int OR string), candidate (rel id),
    or text (labeled), whichever yields a valid rel index.

    pool_relids: the candidate pool = these relation indices (the chain-wise
    2-hop-reachable set, grounded GTE). When None, the pool is the full relation
    set (entity-agnostic, prior behavior). Grounding the pool to the reachable
    set lets structurally-correct relations that are buried under surface-token
    matches in the full set (e.g. location.location.containedby for an
    airport->country hop) surface into top-K — GTE never sees unreachable
    relations. Returns FULL relation indices either way."""
    # Query = the triple's natural-language sub-question (the relation clause),
    # UNWRAPPED. Candidate = a triple-format QUESTION "head | relation(last2) | ?"
    # (generic ? blank, NOT the named ?var). Both sides are questions; the instruct
    # asks whether they are semantically consistent. The earlier "head:(h) relation:(r)
    # tail:(t)" wrapping poisoned ranking (a location head dragged matches to location
    # relations); the flat triple-question + pure-question query avoids that.
    query = rel_clause
    # Candidate pool: grounded (2-hop-reachable) when pool_relids given, else full.
    if pool_relids is not None:
        pool_idx = sorted(i for i in set(pool_relids) if 0 <= i < len(ctx.rels))
    else:
        pool_idx = list(range(len(ctx.rels)))
    rels_pool = [ctx.rels[i] for i in pool_idx]
    labeled = [f"{head} | {_rel_last2(r)} | ?" for r in rels_pool]
    lab2loc = {lab: i for i, lab in enumerate(labeled)}      # labeled text -> pool-local idx
    relid2loc = {r: i for i, r in enumerate(rels_pool)}      # rel id -> pool-local idx
    # POOL_KEY (2026-08-21): same (head, pool) re-queried across facts — register
    # the candidate list once per process; later queries ship the key only.
    import hashlib as _hl
    _pk = _hl.md5(("|".join(rels_pool) + f"#{head}").encode()).hexdigest()[:20]
    _first_time = _pk not in _REGISTERED_POOLS
    # The /retrieve service resets connections under batch concurrency (Errno 104);
    # a reset must NOT silently zero out a triple's candidates (cascades into a bad
    # walk). Retry with backoff — a fresh call succeeds (the ranking itself is fine).
    rows = None
    from kgqa.core.utils import phase_timer
    with phase_timer("gte"):
        for _attempt in range(4):
            try:
                rows = await gte_retrieve(
                    session, query,
                    rels_pool if _first_time else [],
                    candidate_texts=labeled if _first_time else None,
                    top_k=_TRIPLE_GTE_TOPK,
                    instruct=_TRIPLE_GTE_INSTRUCT,
                    pool_key=_pk)
                _REGISTERED_POOLS.add(_pk)
                break
            except Exception as _e:
                if _attempt < 3:
                    await asyncio.sleep(0.4 * (_attempt + 1))
                    continue
                import sys as _sys
                print(f"  GTE error in _gte_for_triple (4 retries failed): {_e}", file=_sys.stderr)
                rows = []
    rows = rows or []
    out = []
    for r in rows:
        loc = None
        # 1) index (service returns it as int OR string) — the position in the
        #    candidates list (pool-local), the most reliable mapping.
        idx = r.get("index")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = None
        if isinstance(idx, int) and 0 <= idx < len(rels_pool):
            loc = idx
        # 2) candidate / text may carry the rel id or the labeled text (pool-local)
        if loc is None:
            for key in (r.get("candidate"), r.get("text")):
                if not isinstance(key, str):
                    continue
                if key in lab2loc:
                    loc = lab2loc[key]; break
                if key in relid2loc:
                    loc = relid2loc[key]; break
        if loc is not None:
            full = pool_idx[loc]   # map pool-local position -> full relation index
            if full not in out:
                out.append(full)
    return out


async def _gte_for_fact(ctx, session, fact_id, hint):
    """Run GTE retrieve + structural prune for one fact. Returns (candidates, gte_raw).

    Mirrors _do_retrieve's candidate construction (start_type + last-two schema
    segments) and structural pruning. If structural prune yields empty, falls
    back to the raw GTE top-15 (relations outside 2-hop scope are better than
    no candidates).
    """
    scope_rel_ids = _chained_source_rel_ids(ctx, fact_id)
    _is_f1 = bool(ctx.fact_ids) and fact_id == ctx.fact_ids[0]
    start_type = (ctx.anchor_name if _is_f1 else (ctx.fact_start_types.get(fact_id) or ctx.anchor_name)) or ""
    if start_type:
        cand_texts = []
        for ri, rid in enumerate(ctx.rels):
            if ri in scope_rel_ids:
                parts = rid.split('.')
                tail = ' '.join(p.replace('_', ' ') for p in parts[-2:])
                cand_texts.append(f"{start_type} {tail}")
            else:
                cand_texts.append(rid)
    else:
        cand_texts = ctx.rel_texts
    try:
        rows = await gte_retrieve(
            session, hint, ctx.rels,
            candidate_texts=cand_texts, top_k=_GTE_TOPK)
    except Exception as _e:
        # Log the error so GTE failures are visible (don't silently return empty)
        import sys as _sys
        print(f"  GTE error in _gte_for_fact({fact_id}): {_e}", file=_sys.stderr)
        return [], []
    gte_rel_ids = []
    gte_raw = []
    for r in rows:
        rid = r.get("candidate")
        idx = r.get("index")
        # GTE returns either an int index or the candidate value itself.
        # If index is present and valid, use it; otherwise look up by name.
        if isinstance(idx, int) and 0 <= idx < len(ctx.rels):
            gte_rel_ids.append(idx)
            gte_raw.append(ctx.rels[idx])
        elif isinstance(rid, str) and rid in ctx.rels:
            ri = ctx.rels.index(rid)
            gte_rel_ids.append(ri)
            gte_raw.append(rid)
        elif isinstance(rid, int) and 0 <= rid < len(ctx.rels):
            gte_rel_ids.append(rid)
            gte_raw.append(ctx.rels[rid])
    # No structural prune (see module note): candidates = raw GTE top-K. The
    # scope is still used above only for candidate-text contextualization.
    pruned = list(gte_rel_ids)
    return pruned, gte_raw


async def _do_decompose(args: Dict[str, Any], ctx, session) -> str:
    """Triple decompose — run GTE PER-TRIPLE, driven by the information flow.

    The harness has already parsed the model's FLOW (triples) + ANCHORS + ANSWER
    into ``ctx.fact_ids`` / ``ctx.fact_texts`` (one fid per triple hop, in flow
    order) and ``ctx.chains`` (per-anchor ordered hops, with con layers). This
    handler runs GTE once per triple: the triple's RELATION CLAUSE is the GTE
    hint (the flow itself drives relation selection — not a separate subquestion
    per fact). It stores each triple's candidate relations on
    ``ctx.fact_relation_candidates`` and returns them inline so the model picks
    from them in select_relations.

    This REPLACES the legacy facts-array decompose: triples are the decomposition.
    The walk (per-anchor, along the flow) and the original expand_branches +
    answer run unchanged downstream.
    """
    entities = args.get("entities") or []
    if isinstance(entities, str):
        entities = [a.strip() for a in entities.split(",") if a.strip()]
    anchor = entities[0] if entities else ""
    if anchor and not ctx.anchor_name:
        ctx.anchor_name = anchor  # used by _gte_for_fact as f1's start_type

    # ---- Grounded GTE (SYSTEM step, deterministic): per chain, per hop, the
    # candidate pool is the 2-hop-reachable relations of the entities the
    # previous hop reached (the chain prefix). hop1 = anchor's 2-hop; hop_i =
    # prior-hop-hit entities' 2-hop. GTE ranks WITHIN this pool, so it never
    # sees structurally-unreachable relations — a correct relation buried under
    # surface-token matches in the full set (e.g. location.location.containedby
    # for an airport->country hop) surfaces into top-K. The LLM's
    # select_relations later picks from these grounded candidates (unchanged).
    chains = list(getattr(ctx, "chains", []) or [])
    if not chains:
        # No chain structure (single sequential question): one chain from the anchor.
        chains = [{"anchor": (ctx.anchor_name or ""),
                   "fact_steps": [[f] for f in ctx.fact_ids]}]

    def _parse_triple(fid):
        txt = ctx.fact_texts.get(fid, "")
        inner = txt.strip().strip("()")
        parts = [p.strip() for p in inner.split("|")] if inner else []
        head = parts[0] if len(parts) >= 1 else ""
        rel_clause = parts[1] if len(parts) >= 2 else (txt or ctx.question)
        tail = parts[2] if len(parts) >= 3 else ""
        return txt, head, rel_clause, tail

    entries = {}   # fid -> entry dict (output in fid order below)
    done_fids = set()
    for chain in chains:
        anc_name = (chain.get("anchor") or "").strip()
        anc_idx = _resolve_chain_anchor(ctx, anc_name) if anc_name else ctx.anchor_idx
        if anc_idx is None:
            continue   # chain's fids fall through to the full-pool orphan pass
        frontier = {anc_idx}
        visited = set(frontier)
        for step in (chain.get("fact_steps") or []):
            grp = step if isinstance(step, list) else [step]
            fids = [f for f in grp if f in ctx.fact_ids and f not in done_fids]
            if not fids:
                continue
            pool = _reach2_relids(ctx, frontier)
            if len(pool) < _GTE_POOL_MIN:
                pool = set(range(len(ctx.rels)))   # too narrow -> full set
            step_cands = {}
            for fid in fids:
                txt, head, rel_clause, tail = _parse_triple(fid)
                cands = await _gte_for_triple(ctx, session, head, rel_clause, tail,
                                              pool_relids=pool)
                ctx.fact_relation_candidates[fid] = cands
                step_cands[fid] = set(cands)
                entries[fid] = {
                    "id": fid, "triple": txt, "relation_hint": rel_clause,
                    "candidate_relations": [ctx.rels[i] for i in cands if i < len(ctx.rels)],
                }
                done_fids.add(fid)
            # Advance the frontier: entities reached via this step's candidate
            # relations become the prefix for the next hop's grounded pool.
            step_rels = set().union(*step_cands.values()) if step_cands else set()
            if step_rels:
                nxt = _entities_via_relations(frontier, step_rels, ctx, exclude=visited)
                nxt = _passthrough_cvts(nxt, ctx, exclude=visited)
                if nxt:
                    visited |= nxt
                    frontier = nxt
    # Orphan facts (not in any chain / unresolved anchor): full-pool GTE.
    for fid in ctx.fact_ids:
        if fid in done_fids:
            continue
        txt, head, rel_clause, tail = _parse_triple(fid)
        cands = await _gte_for_triple(ctx, session, head, rel_clause, tail)
        ctx.fact_relation_candidates[fid] = cands
        entries[fid] = {
            "id": fid, "triple": txt, "relation_hint": rel_clause,
            "candidate_relations": [ctx.rels[i] for i in cands if i < len(ctx.rels)],
        }
    triples_with_candidates = [entries[f] for f in ctx.fact_ids if f in entries]

    return _json_result({
        "flow": triples_with_candidates,
        "entities": entities,
        "answer": args.get("answer", ""),
        "note": ("Per-triple GTE done (driven by the information flow). For EACH "
                 "triple, select ALL relations that express its relation_hint, then "
                 "call select_relations with one entry per fact id. When there are "
                 "multiple entities the walk runs once per entity and the branches "
                 "are namespaced (F*, N*, ...); pick across all entities."),
    })


async def _do_retrieve(args: Dict[str, Any], ctx, session) -> str:
    """MODEL-DRIVEN GTE + STRUCTURAL PRUNING.

    Pipeline: gte_retrieve(hint, top_k=30) → intersect with anchor's outgoing
    relations (structural reachability) → store as CANDIDATES for the model to
    pick from in select_relations.

    The structural prune removes relations that are semantically close (high GTE
    score) but graph-unreachable from the anchor (e.g. geography.mountain_range.*
    when the answer needs location.location.containedby). It is pure gain —
    validated to drop 0 correct relations while cutting GTE-15 to median 4-5.

    This does NOT decide the final relations — the model picks from these
    pruned candidates via select_relations, because only the model can judge
    whether the full relation chain answers the question.
    """
    fact_id = str(args.get("fact_id", ""))
    hint = (args.get("subquestion") or args.get("relation_hint") or "").strip()
    if not hint:
        hint = ctx.fact_texts.get(fact_id, ctx.question)
    if ctx.anchor_idx is None:
        return _json_result({"fact_id": fact_id, "error": "no_anchor",
                             "candidates": []})

    # 1. GTE over ALL case relations — hint as query, keep top-15.
    # Candidate texts are contextualized for relations in this fact's structural
    # scope: "<start_type> <last-two-schema-segments>" instead of a bare
    # dot-notation id. The start_type is the anchor name (f1) or the type noun
    # the previous hop arrives at (f2+, e.g. "airport", "country"). Only the
    # LAST TWO schema segments are kept (dropping the leading domain bucket word
    # like "location"/"government" — it disturbs ordinary-case GTE ranking by
    # pulling toward every relation in that domain). This bridges the
    # deep-semantic ↔ surface-wording gap (model writes "bordering", KG stores
    # "adjoining_relationship") without answer leakage: the candidate carries
    # the relation's own schema name (legitimate KG structure), the hint carries
    # only the model's natural-language wording. Relations OUTSIDE the scope
    # keep the bare id. Zero candidate-count change — avoids 2-hop overload.
    scope_rel_ids = _chained_source_rel_ids(ctx, fact_id)
    _is_f1 = bool(ctx.fact_ids) and fact_id == ctx.fact_ids[0]
    start_type = (ctx.anchor_name if _is_f1 else (ctx.fact_start_types.get(fact_id) or ctx.anchor_name)) or ""
    if start_type:
        cand_texts = []
        for ri, rid in enumerate(ctx.rels):
            if ri in scope_rel_ids:
                parts = rid.split('.')
                tail = ' '.join(p.replace('_', ' ') for p in parts[-2:])
                cand_texts.append(f"{start_type} {tail}")
            else:
                cand_texts.append(rid)
    else:
        cand_texts = ctx.rel_texts

    try:
        rows = await gte_retrieve(
            session, hint, ctx.rels,
            candidate_texts=cand_texts, top_k=_GTE_TOPK)
    except Exception as e:
        return _json_result({"fact_id": fact_id, "error": f"gte_failed: {e}",
                             "candidates": []})

    gte_rel_ids: List[int] = []
    for r in rows:
        cand = r.get("candidate")
        if cand and cand in ctx.rels:
            gte_rel_ids.append(ctx.rels.index(cand))

    # 2. Structural prune: intersect GTE-15 with the structural scope for this
    #    fact (scope_rel_ids was computed above for candidate-text anchoring;
    #    reuse it here). For fact_1 this is the anchor's neighbors. For fact_i
    #    (i>1) it's the neighbors of entities reached via prior facts' selected
    #    relations (the chain: anchor --[f1]--> entity --[f2]--> ...). This
    #    ensures fact2 sees relations matching the entity-type fact1 arrives at,
    #    not the anchor's type — fixing CWQ multi-hop (e.g. person→country→
    #    religion: fact2 gets country.religions, not person.religion).
    # No structural prune (see module note): candidates = raw GTE top-K.
    pruned_rel_ids = list(gte_rel_ids)

    # 3. Store as CANDIDATES (not final). Model picks via select_relations.
    #    Also init fact_relations as the full candidate set so that if the
    #    model skips select_relations, the traversal still has something.
    ctx.fact_relation_candidates[fact_id] = list(pruned_rel_ids)
    ctx.fact_relations[fact_id] = set(pruned_rel_ids)  # default = all candidates

    return _json_result({
        "fact_id": fact_id,
        "subquestion": hint,
        "gte_relations": [ctx.rels[i] for i in gte_rel_ids[:15]],
        "candidate_relations": [ctx.rels[i] for i in pruned_rel_ids],
        "n_candidates": len(pruned_rel_ids),
        "note": (f"GTE-15 pruned to {len(pruned_rel_ids)} structurally-reachable "
                 "relations. Call select_relations to pick the ones that form "
                 "the answer chain, then select to traverse."),
    })


def _anchor_outgoing_rel_ids(ctx) -> set:
    """Structural scope for the anchor: 1-hop neighbor relations PLUS
    2-hop relations via ANY intermediate (CVT or regular entity).

    The traversal (build_mode_level_logical_paths, max_hops_per_step=2) allows
    a relation to span one hop, so the GTE candidate scope must match: a gold
    relation sitting on the 2nd hop (e.g. city -> capital -> country, then
    country -> combatants) is walkable by the traversal but would be pruned by
    a strict 1-hop-only scope. Replay-validated: 2-hop-any recalls +3 golds
    with 0 regression vs the prior 1-hop+CVT-only scope.

    1-hop: relations where anchor is h or t (undirected — Freebase stores some
    relations with reversed edge direction).
    2-hop: relations touching any of the anchor's direct neighbors (CVT or
    regular), mirroring the traversal's hop-spanning.
    """
    out = set()
    ai = ctx.anchor_idx
    if ai is None:
        return out

    # 1. Direct neighbors (undirected); collect ALL neighbors (CVT + regular)
    neighbors = set()
    for h, r, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        if h == ai or t == ai:
            out.add(r)
            other = t if h == ai else h
            if 0 <= other < len(ctx.ents):
                neighbors.add(other)

    # 2. 2-hop via ANY intermediate (CVT or regular), matching the traversal's
    # max_hops_per_step=2 hop-spanning.
    for mid in neighbors:
        for h, r, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
            if h == mid or t == mid:
                out.add(r)

    return out


def _full_adj(ctx):
    """Per-ctx full undirected adjacency (list per entity idx of (rel, other)),
    built ONCE per ctx — shared by _reach2_relids' BFS and the SEQ pool's
    CVT-transparent expansion (both previously rescanned all edges per call,
    blocking the event loop on hub cases)."""
    _adj_full = getattr(ctx, "_full_adj_idx", None)
    if _adj_full is None or len(_adj_full) != len(ctx.ents):
        _adj_full = [[] for _ in range(len(ctx.ents))]
        for hh, rr, tt in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
            if 0 <= hh < len(_adj_full) and 0 <= tt < len(_adj_full):
                _adj_full[hh].append((rr, tt))
                _adj_full[tt].append((rr, hh))
        ctx._full_adj_idx = _adj_full
    return _adj_full


def _reach2_relids(ctx, entity_set) -> set:
    """2-hop reachable relation indices from a SET of entities (undirected, any
    intermediate — CVT or regular). Generalizes _anchor_outgoing_rel_ids to a
    frontier set. Used by the chain-wise grounded GTE: hop_i's candidate pool
    = the 2-hop relations of the entities hop_{i-1} reached (the chain prefix).

    hop1 callers pass {anchor} (equivalent to _anchor_outgoing_rel_ids); hop_i
    callers pass the prior hop's hit entities. Same 1-hop + 2-hop-any structure
    as _anchor_outgoing_rel_ids so the GTE pool matches the traversal's
    max_hops_per_step=2 hop-spanning.
    """
    out = set()
    ent_set = set(e for e in entity_set if e is not None and 0 <= e < len(ctx.ents))
    if not ent_set:
        return out
    # per-case pool memo (user design: 同case内GTE/池常驻): rephrase loops
    # re-query the SAME center set — reachability is deterministic, so the
    # pool is memoized per (frozen center set) on ctx.
    _memo = getattr(ctx, "_pool_memo", None)
    if _memo is None:
        _memo = {}; ctx._pool_memo = _memo
    _mk = frozenset(ent_set)
    if _mk in _memo:
        return set(_memo[_mk])
    neighbors = set()
    for h, r, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        if h in ent_set or t in ent_set:
            out.add(r)
            if h in ent_set and 0 <= t < len(ctx.ents):
                neighbors.add(t)
            if t in ent_set and 0 <= h < len(ctx.ents):
                neighbors.add(h)
    neighbors -= ent_set
    # HIT LOGIC (user ruling, 2026-08-19): the budget is TWO NAMED hops;
    # CVT nodes are TRANSPARENT — passing through one costs no hop (and a
    # path ENDING on a CVT is also penetrated). BFS by named-hop cost:
    # cost 0 = centers, cost 1 = direct named neighbors (+ everything behind
    # transparent CVT chains at the same cost), cost 2 = one more named hop
    # (again with free CVT passage). Relations on any edge touched along the
    # way enter the pool.
    # full adjacency index, built ONCE per ctx (single O(E) pass) — the BFS
    # visits thousands of unique nodes on hub centers; per-node edge scans
    # are O(V×E) (France: 0.9s per pool call in the event loop).
    _adj_full = _full_adj(ctx)

    def _adj(i):
        return _adj_full[i] if 0 <= i < len(_adj_full) else ()

    import re as _re_idlike
    from kgqa.agent.entity_kinds import is_id_node as _id_like


    def _expand(seed, cost_budget):
        seen, frontier, cost = set(seed), set(seed), 0
        while frontier and cost < cost_budget:
            nxt, stack = set(), list(frontier)
            while stack:   # current named-hop layer; CVTs extend it for free
                i = stack.pop()
                for rr, other in _adj(i):
                    if other in seen or not (0 <= other < len(ctx.ents)):
                        continue
                    seen.add(other)
                    _nm = str(ctx.ents[other])
                    if _nm.startswith(("m.", "g.")) or _id_like(_nm):
                        stack.append(other)     # transparent: same layer
                    else:
                        nxt.add(other)          # named: costs the next hop
            frontier = nxt
            cost += 1
        return seen

    reach = _expand(ent_set, 2)
    for hh, rr, tt in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        if hh in reach or tt in reach:
            out.add(rr)
    # PER-CTX EDGE INDEX + PAYS CACHE (2026-08-22 perf): the walkability and
    # renderability filters were O(pool × all-edges) per call — hub cases
    # (France: 184 pool rels × 5189 edges ≈ 1M iterations, ~1s) ran this in
    # the event loop and blocked every concurrent case. Both filters now run
    # on a per-relation edge list built once per ctx; the center-independent
    # pays bit is precomputed with it.
    def _is_cvt(i):
        return 0 <= i < len(ctx.ents) and str(ctx.ents[i]).startswith(("m.", "g."))
    _edges = getattr(ctx, "_rel_edges_idx", None)
    if _edges is None or len(_edges) != len(ctx.rels):
        from collections import defaultdict as _dd
        _cvt_named = _dd(set)
        _cvt_touch = _dd(int)
        _edges = _dd(list)
        for hh, rr2, tt in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
            _edges[rr2].append((hh, tt))
            hc, tc = _is_cvt(hh), _is_cvt(tt)
            if hc:
                _cvt_touch[hh] += 1
            if tc:
                _cvt_touch[tt] += 1
            if hc and not tc and 0 <= tt < len(ctx.ents):
                _cvt_named[hh].add(tt)
            elif tc and not hc and 0 <= hh < len(ctx.ents):
                _cvt_named[tt].add(hh)
        # VALUE-STUB ADMISSION (user ruling 2026-09-14, gdp_deflator specimen):
        # a relation whose every edge ends on a DANGLING CVT/g terminal (the
        # dataset truncated the value chain at the node) was pays=False and
        # so never entered ANY GTE pool — the semantically top-ranked
        # relation (Monaco #1, 0.51) lost before ranking. But a stub edge
        # renders as `country --rel--> g.xxx`, and edge EXISTENCE is the
        # discriminator signal (deflator: only 2/9 neighbors carry it;
        # runtime: 1/19). A CVT touching exactly ONE edge overall is a
        # value stub, not a content CVT — its edge pays.
        _stub_pays = os.environ.get("SEQ_POOL_VALUE_STUB", "1") == "1"
        _pays = {}
        for rr2, es in _edges.items():
            _pays[rr2] = any(
                (hh != tt) and (
                    (not _is_cvt(hh) and not _is_cvt(tt))
                    or (hh in _cvt_named and (_cvt_named[hh] - {tt}))
                    or (tt in _cvt_named and (_cvt_named[tt] - {hh}))
                    or (_stub_pays
                        and (_is_cvt(hh) != _is_cvt(tt))
                        and _cvt_touch.get(hh if _is_cvt(hh) else tt, 0) <= 1)
                )
                for hh, tt in es)
        ctx._rel_edges_idx = _edges
        ctx._rel_pays_idx = _pays
    _edges = ctx._rel_edges_idx
    _pays = ctx._rel_pays_idx

    # WALKABILITY INVARIANT (user ruling, 2026-08-21): pool membership ⟺ the
    # relation has a WALKABLE instantiation from the centers. A pattern path
    # is ≤2 named hops with the SELECTED relation as the LAST hop — so the
    # relation is walkable iff it has an edge whose endpoint is within ONE
    # named hop (CVT-transparent) of a center. Edges whose endpoints sit at
    # 2 named hops (King specimen: featured_film_locations on the director's
    # OTHER films) are only usable as hop-3 — the walk correctly refuses
    # them, so the pool must not offer them as if they were current-fact
    # bridges ("关系可达但是子图不可达" seam). Pool = walkable-last-hop set.
    walk_reach = _expand(ent_set, 1)
    # RENDERABILITY INVARIANT (Norwood specimen, 2026-08-22): 能选必可达 must
    # hold through to the DISPLAY — bare-CVT-stub / self-loop instantiations
    # pay nothing (see the pays precompute above).
    from kgqa.traversal.path_utils import _is_noisy_path_relation
    out = {r for r in out
           if not _is_noisy_path_relation(ctx.rels[r])
           and _pays.get(r, True)
           and any((hh in walk_reach or tt in walk_reach)
                   for hh, tt in _edges.get(r, ()))}
    return out


def _reachable_rel_ids_from(entities, ctx) -> set:
    """Relation indices reachable from a set of entity indices (outgoing only),
    PLUS CVT-bridged 2-hop relations.

    Directed (h==entity): only follow edges where the entity is the HEAD.
    CVT-bridged: if entity → CVT → X, the CVT→X edge relations are also
    included (same logic as _anchor_outgoing_rel_ids — CVT mediates one
    logical step).
    """
    from kgqa.traversal.cvt import is_cvt_like
    out = set()
    ent_set = set(entities)
    cvt_neighbors = set()
    for h, r in zip(ctx.h_ids, ctx.r_ids):
        if h in ent_set:
            out.add(r)
    # Also find CVT nodes directly connected to these entities (undirected)
    for h, r, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        if h in ent_set and 0 <= t < len(ctx.ents) and is_cvt_like(ctx.ents[t]):
            cvt_neighbors.add(t)
        if t in ent_set and 0 <= h < len(ctx.ents) and is_cvt_like(ctx.ents[h]):
            cvt_neighbors.add(h)
    # Add CVT-bridged relations
    for cvt_idx in cvt_neighbors:
        for h, r, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
            if h == cvt_idx or t == cvt_idx:
                out.add(r)
    return out


def _entities_via_relations(source_entities, rel_ids, ctx, exclude=None) -> set:
    """Entities reachable from source via the given relation set (1 hop).

    Used by chained retrieve: given fact_{i-1}'s selected relations and the
    anchor (or the entities fact_{i-2} reached), compute where fact_{i-1}
    arrives — those entities' neighbors become fact_i's structural scope.

    exclude: a set of entity indices that have ALREADY been visited earlier in
    the chain. These are never returned, preventing loops where the chain
    revisits an entity via a different relation (e.g. anchor→CVT→Israel→CVT).
    """
    rel_set = set(rel_ids)
    src = set(source_entities)
    excl = set(exclude) if exclude else set()
    reached = set()
    for h, r, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        if r not in rel_set:
            continue
        if h in src and t not in excl:
            reached.add(t)
        if t in src and h not in excl:
            reached.add(h)
    # Also exclude the source itself (don't stay in place)
    reached -= src
    return reached


def _passthrough_cvts(entities, ctx, exclude=None) -> set:
    """Extend through CVT nodes to reach the named entities beyond them.

    When the chain arrives at a CVT (e.g. national_anthem_of → CVT), the
    actual entity of interest (e.g. the country) is one more hop beyond the
    CVT. Without this passthrough, fact_i's scope is limited to the CVT's
    own attributes (country, anthem, start_date...) and misses the named
    entity's domain relations (form_of_government, capital, etc.).

    This extends each CVT node one hop forward to its named (non-CVT) neighbors,
    adding them to the entity set. Non-CVT entities pass through unchanged.
    """
    from kgqa.traversal.cvt import is_cvt_like
    excl = set(exclude) if exclude else set()
    result = set()
    for idx in entities:
        if idx >= len(ctx.ents):
            continue
        name = ctx.ents[idx]
        if is_cvt_like(name):
            # CVT: extend one hop to non-CVT neighbors
            for h, r, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
                if h == idx and t not in excl:
                    t_name = ctx.ents[t] if t < len(ctx.ents) else ""
                    if t_name and not is_cvt_like(t_name):
                        result.add(t)
                if t == idx and h not in excl:
                    h_name = ctx.ents[h] if h < len(ctx.ents) else ""
                    if h_name and not is_cvt_like(h_name):
                        result.add(h)
        else:
            result.add(idx)
    return result


def _chained_source_rel_ids(ctx, fact_id) -> set:
    """Structural scope for fact_i: which relations are reachable?

    - fact_1 (first fact): anchor's neighbor relations (undirected).
    - fact_i (i>1): relations reachable from the entities that fact_{i-1}'s
      selected relations reach from the anchor (or from fact_{i-2}'s endpoint).
      This is the chain: anchor --[f1 rels]--> entity --[f2 rels]--> ...
    """
    # Determine fact order from ctx.fact_ids (all facts from decompose, in order)
    fids = list(ctx.fact_ids)
    if not fids:
        return _anchor_outgoing_rel_ids(ctx)

    try:
        idx = fids.index(fact_id)
    except ValueError:
        return _anchor_outgoing_rel_ids(ctx)

    if idx == 0:
        # First fact: scope = anchor neighbors
        return _anchor_outgoing_rel_ids(ctx)

    # Chained: walk from anchor through each prior fact's FULL candidate set.
    # At retrieve time, select_relations hasn't run yet, so we use all of each
    # prior fact's GTE-pruned candidates as the chain link — NOT just top-1
    # (top-1 is too fragile: if it's the wrong relation, the whole chain breaks
    # and fact2's scope becomes garbage). Using ALL candidates means the chain
    # reaches every entity reachable via any plausible first-hop relation,
    # giving fact2 a complete picture of what entity-types it might need to
    # handle. The scope stays tight because _reachable_rel_ids_from is directed
    # (only outgoing edges), so CVT→person back-links don't re-admit person
    # relations.
    #
    # Loop prevention: track ALL entities visited across the chain (visited
    # set). Each hop excludes everything in visited.
    source = {ctx.anchor_idx} if ctx.anchor_idx is not None else set()
    visited = set(source)
    for prior_fid in fids[:idx]:
        cands = ctx.fact_relation_candidates.get(prior_fid, [])
        if cands:
            prior_rels = set(cands)
        else:
            prior_rels = ctx.fact_relations.get(prior_fid, set())
        if prior_rels:
            source = _entities_via_relations(source, prior_rels, ctx, exclude=visited)
            # CVT passthrough: extend through CVT nodes to named entities.
            source = _passthrough_cvts(source, ctx, exclude=visited)
            visited |= source
            if not source:
                # Boundary 2: chain broken — model may have under-planned steps
                # (e.g. needed 3 facts but decomposed 2). Relax: try a 2-hop
                # structural connectivity check from the PREVIOUS source (before
                # this hop failed) through ALL its outgoing edges, not just the
                # GTE-pruned candidates. This lets the chain skip a missed step
                # by exploring one extra hop structurally.
                # Recover the source from before this hop:
                prev_source = visited - {ctx.anchor_idx} if len(visited) > 1 else {ctx.anchor_idx}
                # Walk one hop from prev_source via ANY relation (not just GTE candidates)
                relaxed = set()
                for h_idx, r_idx in zip(ctx.h_ids, ctx.r_ids):
                    if h_idx in prev_source:
                        relaxed.add(r_idx)
                if relaxed:
                    source = _entities_via_relations(prev_source, relaxed, ctx, exclude=visited)
                    source = _passthrough_cvts(source, ctx, exclude=visited)
                    visited |= source
                if not source:
                    break  # chain truly broken

    if not source:
        # Chain broken — fall back to anchor neighbors (best effort)
        return _anchor_outgoing_rel_ids(ctx)
    return _reachable_rel_ids_from(source, ctx)


def _compact_overview(overview: str) -> str:
    """Secondary compaction pass on the evidence tree overview.

    The main compaction (≤3 candidates per branch, no entity list in the
    RETRIEVED CANDIDATES header) is done inside _do_select. This function
    is a safety net: if _do_select produced a long overview, cap each branch
    block's tree lines to keep the total readable. No-op if already compact.
    """
    # _do_select already produces compact output; this is a placeholder for
    # future additional compaction if needed.
    return overview


def _resolve_chain_anchor(ctx, anchor_name: str):
    """Resolve a multi-anchor chain's declared anchor NAME to a graph entity idx.

    Exact normalize match first, then substring (len>=3, overlap>=40%) —
    tighter than loop._resolve_anchor to avoid cross-language false matches
    (e.g. a typo 'Lavinge' substring-matching an unrelated Thai entity).
    Returns None for empty/unresolvable names — the chain is DROPPED (its
    fids fall through to the full-pool GTE), NOT fallen back to the primary
    anchor (which would ground the wrong entity).
    """
    if not anchor_name:
        return None  # empty name = no anchor → drop chain (was: ctx.anchor_idx)
    mn = normalize(anchor_name)
    if not mn or len(mn) < 2:
        return None
    import re
    # System-layer block: a pure number/date/value is NOT a named entity — drop it
    # (keeps digit-bearing NAMES like "WW2", "Boeing 747", which contain letters).
    if re.match(r'^[\d.\-/\s]+$', anchor_name) and any(c.isdigit() for c in anchor_name):
        return None
    from kgqa.traversal.cvt import is_cvt_like
    def _named(i):  # CVT nodes are never valid named entities — system-layer block
        return not is_cvt_like(ctx.ents[i])
    for i, e in enumerate(ctx.ents):
        if normalize(e) == mn and _named(i):
            return i
    # Substring match: require BOTH names >=3 chars + meaningful overlap
    # (shorter >= 40% of longer) to prevent short/garbage/cross-language names
    # from false-matching.
    for i, e in enumerate(ctx.ents):
        en = normalize(e)
        if not en or len(en) < 3 or len(mn) < 3:
            continue
        if not _named(i):
            continue
        if mn in en or en in mn:
            shorter = min(len(mn), len(en))
            longer = max(len(mn), len(en))
            if shorter >= longer * 0.4:
                return i
    return None


async def _do_select_relations(args: Dict[str, Any], ctx, session) -> str:
    """Model decision: pick the relation(s) forming the answer chain, per fact,
    THEN run the graph traversal (formerly the separate `select` tool) and
    return the evidence tree inline.

    This merges select_relations + select into one step:
    1. Validate the model's relation picks against each fact's candidates.
    2. Lock the chosen relations into ctx.fact_relations.
    3. Run stage_5_graph_traversal + build the evidence tree overview.
    4. Return the tree (compact: ≤3 candidates per branch + ellipsis).

    Args:
        selections: [{"fact_id": "f1", "relations": ["people.person.parents"]}, ...]
    """
    selections = args.get("selections") or []
    if not selections:
        return _json_result({"error": "no selections provided"})

    chosen = {}
    skipped = []
    for sel in selections:
        fid = str(sel.get("fact_id", ""))
        rels_chosen = sel.get("relations") or []
        candidates = ctx.fact_relation_candidates.get(fid, [])
        cand_names = {ctx.rels[i] for i in candidates}
        valid_ids = []
        invalid = []
        for rn in rels_chosen:
            if rn in cand_names:
                valid_ids.append(ctx.rels.index(rn))
            else:
                invalid.append(rn)
        if valid_ids:
            chosen[fid] = set(valid_ids)
            ctx.fact_relations[fid] = set(valid_ids)
        if invalid:
            skipped.append({"fact_id": fid, "invalid": invalid})
        if not valid_ids:
            # model picked nothing valid — keep all candidates as fallback
            chosen[fid] = set(candidates)
            ctx.fact_relations[fid] = set(candidates)

    # Non-blocking validation (REPORT, not a hard reject): if the model emitted any
    # relation not in that fact's candidate set, the offending names are already
    # recorded in `skipped`; the VALID subset is locked into ctx.fact_relations
    # above (and if NOTHING was valid, that fact falls back to all candidates,
    # lines above). We PROCEED to run the traversal on the valid subset and surface
    # the dropped names in the result's `skipped_invalid` field (fact_id → invalid
    # list) so the model can see what was filtered — but we do NOT block or force a
    # re-select.
    #
    # Why no hard reject: in one-shot rollout each candidate gets a single
    # execution. A hard reject returns BEFORE the traversal → ctx.branches stays
    # empty → score_variant("plan") = 0 for EVERY rejected candidate → the whole
    # K-candidate group collapses to "skip" (no variance, none ≥ CORRECT).
    # Reporting + proceeding keeps the valid subset's traversal scoreable in BOTH
    # the agent loop and the rollout. (Was: reject-once via _select_invalid_retried;
    # that flag is now unused — the graceful path is always taken.)

    selected_summary = {fid: [ctx.rels[i] for i in ids] for fid, ids in chosen.items()}

    # ── Run the traversal ──
    # Single chain (the common case): walk once, EXACTLY as before (baseline).
    # Multiple chains (multi-anchor): walk EACH anchor independently via _do_select
    # (the single-anchor engine is reused UNCHANGED — no rewrite of ranking /
    # materialize / evidence-building), then present a COMPACT combined overview
    # (one header line per branch, namespaced F1../N1.. per anchor) so the model
    # picks branches across anchors. Full triples are revealed by expand_branches
    # — the overview's job is to let the model CHOOSE by relation chain, not to
    # dump every candidate. This is the validated mechanism: the mature pipeline
    # is sufficient; what was missing was walking every declared anchor instead
    # of only chains[0].
    chains = list(getattr(ctx, "chains", []) or [])

    if len(chains) <= 1:
        traverse_result = await _do_select(ctx)
        try:
            traverse_obj = json.loads(traverse_result)
        except Exception:
            traverse_obj = {"error": "traverse failed"}
        if "overview" in traverse_obj:
            traverse_obj["overview"] = _compact_overview(traverse_obj["overview"])
        traverse_obj["selected"] = selected_summary
        traverse_obj["skipped_invalid"] = skipped
        return _json_result(traverse_obj)

    # ── Multi-anchor: per-anchor _do_select, namespace branches, merge ──
    _ANCHOR_TAGS = ["F", "N", "P", "Q", "R", "S"]
    merged_branches: Dict[str, dict] = {}
    merged_all_candidates: List[str] = list(getattr(ctx, "all_candidates", []) or [])
    sections: List[str] = []
    per_anchor: Dict[str, dict] = {}
    for ci, chain in enumerate(chains[:len(_ANCHOR_TAGS)]):
        tag = _ANCHOR_TAGS[ci]
        anc_name = (chain.get("anchor") or "").strip()
        anc_idx = _resolve_chain_anchor(ctx, anc_name)
        if anc_idx is None:
            continue
        chain_steps = chain.get("fact_steps") or []
        chain_fids = []
        for grp in chain_steps:
            if isinstance(grp, str):
                grp = [grp]
            chain_fids.extend(f for f in grp if f in ctx.fact_relations)
        if not chain_fids:
            continue
        # Isolate this chain: temporarily scope ctx to THIS anchor + chain's
        # facts so _do_select (which reads anchor_idx / fact_ids / fact_steps)
        # walks only this chain. Dispatch is sequential per case, so save/restore
        # is safe. fact_relations holds all facts; only this chain's fids are
        # referenced via the scoped fact_ids/fact_steps.
        saved = (ctx.anchor_idx, ctx.anchor_name, ctx.fact_steps, ctx.fact_ids)
        ctx.anchor_idx = anc_idx
        ctx.anchor_name = anc_name or ctx.ents[anc_idx]
        ctx.fact_steps = chain_steps
        ctx.fact_ids = chain_fids
        ctx.branches = {}
        try:
            await _do_select(ctx)   # populates ctx.branches {"1":..,"2":..} (ranked)
        finally:
            ctx.anchor_idx, ctx.anchor_name, ctx.fact_steps, ctx.fact_ids = saved
        # Move this chain's ranked branches into the merged namespaced dict.
        chain_branches = list(ctx.branches.items())
        for bid, br in chain_branches:
            merged_branches[f"{tag}{bid}"] = br
        merged_all_candidates.extend(getattr(ctx, "all_candidates", []) or [])
        anc_display = ctx.ents[anc_idx] if 0 <= anc_idx < len(ctx.ents) else anc_name
        per_anchor[tag] = {"anchor": anc_display, "n_branches": len(chain_branches)}
        # Compact header per branch (relation chain + sample candidates + #ID).
        lines = [f"=== ANCHOR: {anc_display}  (branches #{tag}1..#{tag}{len(chain_branches)}) ==="]
        for bid, br in chain_branches:
            nbid = f"{tag}{bid}"
            cands = br.get("candidates", []) or []
            cs = ", ".join(cands[:5]) + (f"  (+{len(cands)-5})" if len(cands) > 5 else "")
            lines.append(f"  {nbid}: {br.get('readable','')}  ->  {len(cands)} cand [{cs}]")
        sections.append("\n".join(lines))

    ctx.branches = merged_branches
    # Dedup the cross-anchor candidate pool (for gt_hit / answer fallback).
    seen = set(); dedup = []
    for c in merged_all_candidates:
        nc = normalize(c)
        if nc and nc not in seen:
            seen.add(nc); dedup.append(c)
    ctx.all_candidates = dedup
    ctx.selected_candidates = dedup[:60]

    combined = "\n\n".join(sections) if sections else "(no branches)"
    overview_text = (
        f"MULTI-ANCHOR RETRIEVE: {len(merged_branches)} branches across "
        f"{len(per_anchor)} anchors (namespaced F*, N*, ...).\n"
        "Each branch shows its relation chain + a few candidates + a #ID marker. "
        "Branches are pre-ranked within each anchor by relation-chain alignment "
        "with your facts. ANALYZE both anchors' trees, pick the branches relevant "
        "to the question — the answer must satisfy EVERY anchor's constraint (e.g. "
        "borders France AND contains an airport serving Nijmegen) — and call "
        "expand_branches(['F1','N2',...]). Then answer from what you expanded.\n\n"
        + combined
    )
    return _json_result({
        "n_patterns": len(merged_branches),
        "overview": overview_text,
        "candidates": dedup[:50],
        "per_anchor": per_anchor,
        "selected": selected_summary,
        "skipped_invalid": skipped,
    })


def _paths_to_preview(paths, ctx, limit: int = 10) -> List[str]:
    """Render a few witness paths as 'anchor --[rel]--> cand' for the model."""
    out = []
    anchor_name = ctx.ents[ctx.anchor_idx] if ctx.anchor_idx is not None and 0 <= ctx.anchor_idx < len(ctx.ents) else "?"
    for p in paths[:limit]:
        nodes = p.get("nodes", [])
        rels = p.get("relations", [])
        if not nodes:
            continue
        parts = [anchor_name]
        for i, ni in enumerate(nodes[1:], 1):
            ri = rels[i - 1] if i - 1 < len(rels) else None
            rtxt = ctx.rels[ri] if ri is not None and 0 <= ri < len(ctx.rels) else "?"
            nm = ctx.ents[ni] if 0 <= ni < len(ctx.ents) else "?"
            parts.append(f"--[{rtxt}]--> {nm}")
        out.append(" ".join(parts))
    return out


def _hint_match_rank(patterns, fact_rel_sets, fids):
    """Rank relation-pattern branches by how well they align with the
    decomposed facts' relation hints.

    Coarse ranking (the system does this; the model does the fine selection
    by expanding branches). For each branch path, align the per-fact hint
    relation-id sets against the path's relation-id sequence in order:

        hints = [set_f1, set_f2, ...]   (one id-set per fact, in fact order)
        path  = [r1, r2, ...]           (relation id sequence of the branch)

    Walk the path hops left-to-right; consume hints left-to-right. Hop i
    counts as a "level hit" if it belongs to the next unconsumed hint's set,
    and that hint is then consumed (so a single hint is matched at most once —
    this avoids a step-1 relation matching at both hop 1 and hop 2 and being
    double-counted).

    Rank key: (#level-hits desc, path-length asc) — more aligned hops first,
    shorter paths break ties. Branches with zero hits sink to the bottom but
    are NOT dropped (the model may still want them).

    Returns the patterns list sorted by this key (stable).
    """
    if not fact_rel_sets:
        # No hints available (e.g. decompose/retrieve skipped) — leave as-is.
        return list(patterns)

    def _key(lp):
        rels = (lp.get("best_raw_path") or {}).get("relations", [])
        rel_seq = [r for r in rels if r is not None]
        hints_iter = iter(fact_rel_sets)
        try:
            cur_hint = next(hints_iter)
        except StopIteration:
            cur_hint = None
        hits = 0
        consumed_any = False
        for r in rel_seq:
            if cur_hint is not None and r in cur_hint:
                hits += 1
                consumed_any = True
                try:
                    cur_hint = next(hints_iter)
                except StopIteration:
                    cur_hint = None
                    # remaining hops can't raise hits further
                    break
        return (-hits, len(rel_seq))

    return sorted(patterns, key=_key)


def _steps_to_groups(fact_steps, fact_relations, fids):
    """Build ordered (step_fids, step_relations) from the explicit chain structure.

    ``fact_steps`` comes from harness parsing ``question_chains[].steps``: a list
    where each entry is either ``[fid]`` (a sequential step — its answer feeds the
    next) or ``[fid, fid, ...]`` (a conjunctive ``con`` layer — those facts
    constrain the SAME entity reached by the prior step, so their selected
    relations are UNIONED at one layer, not chained in series).

    Sequential facts each stay their own step; conjunctive facts merge into one
    step (relation union). This replaces the old ``_group_parallel_facts``, which
    INFERRED siblings from ``f{N}.k`` id naming — grouping is now explicit (the
    model writes ``con``), which keeps the decomposition (facts) and the structure
    (steps) from diverging.

    Falls back to all-facts-sequential when ``fact_steps`` is empty (single
    sequential question with no ``question_chains``). Only fids that have selected
    relations are walked; empty groups are dropped.
    """
    if not fact_steps:
        fact_steps = [[fid] for fid in fids]

    step_fids = []
    step_relations = []
    for grp in fact_steps:
        if isinstance(grp, str):
            grp = [grp]
        fids_in = [f for f in grp if f in fact_relations]
        if not fids_in:
            continue
        merged = []
        seen = set()
        for fid in fids_in:
            for r in fact_relations.get(fid, []):
                if r not in seen:
                    seen.add(r)
                    merged.append(r)
        step_fids.append(fids_in[0] if len(fids_in) == 1 else fids_in)
        step_relations.append(merged)

    if not step_fids:
        for fid in fids:
            step_fids.append(fid)
            step_relations.append(list(fact_relations.get(fid, [])))
    return step_fids, step_relations


async def _do_select(ctx) -> str:
    """Run the FULL multi-step traversal (stage_5_graph_traversal) over all
    facts' hinted relations, then build a TREE overview of evidence.

    This reuses the proven stage-5 engine (mode-level logical paths + CVT
    expansion + RPE fallback) for traversal, AND the proven Stage-8 evidence
    builders (build_pattern_evidence_triples + _render_path_tree) for
    presentation — so the overview shows CVT-expanded candidates (Kasich, not
    CVT node IDs) and a trie structure, not a flat list.

    The model uses this overview to SELECT which branch to expand (Stage 7's
    role), then expand_branches(['N']) drills into the chosen branch (Stage 8's role).

    Conjunctive layers: a step group with several fids (a ``con`` group from
    ``question_chains``) means those facts constrain the SAME entity reached by
    the prior step — they are NOT sequential hops. Their selected relations are
    UNIONED into one step, mirroring stage's ``_merge_constraint_steps``. Without
    this merge the traversal would walk them in series (f2 → entity → f3), which
    is semantically wrong: both filters read attributes of the entity reached by
    the prior step, they don't chain. Grouping is explicit (the model writes
    ``con``), parsed into ``ctx.fact_steps`` by the harness.
    """
    fids = [fid for fid in ctx.fact_ids if fid in ctx.fact_relations]
    if not fids or ctx.anchor_idx is None:
        return _json_result({"error": "no_facts_or_anchor",
                             "evidence": [], "candidates": []})
    # Build ordered steps from the explicit question_chains structure parsed in
    # harness (ctx.fact_steps): each entry is [fid] (sequential step) or
    # [fid, ...] (a conjunctive `con` layer — those facts' relations are unioned
    # at one layer, not chained in series). Falls back to all-facts-sequential
    # when no chains were declared.
    step_fids, step_relations = _steps_to_groups(
        getattr(ctx, "fact_steps", None) or [], ctx.fact_relations, fids)

    # Build a minimal CaseState and run the proven stage-5 traversal
    cs = CaseState(case_id=ctx.case_id or "agent", case_num=ctx.case_num or 0,
                   sample=ctx.sample, pilot_row=ctx.pilot_row)
    cs.anchor_idx = ctx.anchor_idx
    cs.anchor_name = ctx.anchor_name
    cs.h_ids, cs.r_ids, cs.t_ids = ctx.h_ids, ctx.r_ids, ctx.t_ids
    cs.ents, cs.rels, cs.rel_texts = ctx.ents, ctx.rels, ctx.rel_texts
    cs.step_relations = step_relations
    cs.steps = [{"id": "+".join(g) if isinstance(g, list) else g}
                for g in step_fids]   # n_steps = n_groups (siblings merged)
    cs.breakpoints = ctx.breakpoints or {}
    cs.active = True
    try:
        await stage_5_graph_traversal([cs])
    except Exception as e:
        return _json_result({"error": f"stage5_failed: {e}",
                             "evidence": [], "candidates": []})

    # stage-5 output: cs.paths / cs.logical_paths / cs.answer_candidates
    ctx.all_paths = cs.paths or []
    ctx.all_candidates = list(getattr(cs, "answer_candidates", []) or [])
    bp_indices = set(cs.breakpoints.values()) if cs.breakpoints else set()
    if cs.anchor_idx is not None:
        bp_indices.discard(cs.anchor_idx)
    if ctx.all_paths:
        patterns = compress_paths(ctx.all_paths, ctx.ents, ctx.rels, ctx.anchor_idx, bp_indices)
    else:
        patterns = cs.logical_paths or []
    ctx.logical_paths = patterns

    # ── Coarse rank by hint alignment (system does this; model does fine
    # selection via expand_branches). Aligns each branch's relation-id sequence
    # against the per-fact hint relation sets, in fact order. More aligned
    # hops + shorter path = more relevant. All patterns are kept — the model
    # sees the whole ranked tree and decides which branches to expand.
    fids = [fid for fid in ctx.fact_ids if fid in ctx.fact_relations]
    fact_rel_sets = [ctx.fact_relations[fid] for fid in fids]
    ranked_patterns = _hint_match_rank(patterns, fact_rel_sets, fids)

    # ── Build CVT-aware evidence via the proven Stage-8 builder ──
    # build_pattern_evidence_triples returns dict[label -> PatternEvidence]
    # where each PatternEvidence has:
    #   .candidates  — CVT-expanded named entities (office_holder names, not CVT IDs)
    #   .triples     — full (h, r, t) triples with CVT attributes expanded
    #   .tree_data   — nested trie (display-named nodes) for _render_path_tree
    # This fixes Blocker A (candidates now include CVT-expanded entities) and
    # Blocker B (overview rendered as a tree, not a flat list).
    valid_patterns = [lp for lp in ranked_patterns if isinstance(lp, dict) and lp.get("best_raw_path")]
    # ── Materialize logical patterns into ALL raw paths (Stage-7→8 bridge) ──
    # The logical pattern carries one witness path; the proven Stage-8 builder
    # expands evidence from the pattern's raw_paths. Without materialization,
    # only the single witness is seen (e.g. only 1 of 5 languages). This mirrors
    # the baseline case_runner exactly — keep the verified mechanism intact.
    valid_patterns = materialize_selected_logical_patterns(
        valid_patterns, ctx.ents, ctx.rels, ctx.h_ids, ctx.r_ids, ctx.t_ids,
        ctx.anchor_idx, set((ctx.breakpoints or {}).values()))
    pat_evidence = build_pattern_evidence_triples(
        valid_patterns, ctx.ents, ctx.rels, ctx.h_ids, ctx.r_ids, ctx.t_ids,
        ctx.anchor_idx, max_grouped_lines=120)

    # Preserve the hint-alignment order in the evidence dict output.
    # build_pattern_evidence_triples labels patterns P1, P2, ... in input
    # order, so iterating pat_evidence in insertion order == ranked order.
    ranked = list(pat_evidence.items())

    # Cap the overview to the top-20 hint-aligned branches per chain (the model
    # can still expand any branch whose #N appears). Hint-aligned order means the
    # most relevant branches are always at the top.
    ranked = ranked[:20]

    # ── Build numbered tree overview (numbers on the RIGHT, per user spec) ──
    ctx.branches = {}
    overview_blocks = []
    anchor_name = ctx.anchor_name or ""
    for i, (label, pe) in enumerate(ranked, 1):
        bid = str(i)
        # Render this branch's evidence as a YAML-like trie (Stage 8 form).
        tree_lines = _render_path_tree(pe.tree_data, max_lines=200)

        # Resolve the branch's candidates. pe.candidates comes from
        # compress_paths (path non-CVT nodes) and is EMPTY for CVT-terminated
        # paths (anchor -> CVT), even though the office_holder NAME is present
        # in pe.triples / the rendered tree. Fall back to deriving candidates
        # from the triples so the overview header reflects what's actually in
        # the branch (Blocker A completion).
        cands = list(pe.candidates)
        if len(cands) < 3 and pe.triples:
            derived = _candidates_from_triples(pe.triples, anchor_name)
            seen_c = {normalize(c) for c in cands}
            for d in derived:
                if normalize(d) not in seen_c:
                    cands.append(d)
                    seen_c.add(normalize(d))

        # Store the full branch data for expand_branches to re-render later.
        # Key: store the materialized pattern (with raw_paths) so expand can
        # re-run build_pattern_evidence_triples and get ALL candidates (not just
        # the witness). Without this, expand only sees the single witness path.
        ctx.branches[bid] = {
            "label": label,
            "candidates": cands,
            "triples": pe.triples,
            "readable": pe.readable,
            "tree_lines": tree_lines,
            "pattern": valid_patterns[i] if i < len(valid_patterns) else None,
        }

        # Compose the branch block with the #N marker on the RIGHT of each leaf.
        # overview shows FEW candidates per branch — this display is only to help
        # the model CHOOSE which branches to expand, NOT to answer from. The full
        # candidate set (the answer leaves) is surfaced by expand_branches (which
        # the model must call to answer). Keep this small to avoid clutter.
        cand_str = ", ".join(cands[:5])
        if len(cands) > 5:
            cand_str += f", ... (+{len(cands) - 5})"
        header = (f"  branch {bid}: {pe.readable}  →  {len(cands)} candidates "
                  f"[{cand_str}]    #{bid}")

        # Compact overview: branch HEADER only (relation chain + sample candidates
        # + #N marker). The node-level tree is NOT rendered here — it bloats the
        # context and the model picks branches by relation chain, not node detail.
        # The full tree is revealed by expand_branches (stored in ctx.branches).
        overview_blocks.append(header)

    # Collect all candidates (for gt_hit + fallback).
    seen = set(); dedup = []
    for c in ctx.all_candidates:
        nc = normalize(c)
        if nc and nc not in seen:
            seen.add(nc); dedup.append(c)
    ctx.selected_candidates = dedup[:60]

    overview_text = (
        f"RETRIEVED CANDIDATES: {len(dedup)} entities in the answer pool "
        "(expand branches to see them).\n\n"
        f"EVIDENCE TREE (ranked by relevance; #N on the right marks each "
        "branch):\n"
        "Branches are pre-ranked by how well their relation chain aligns with "
        "your decomposed facts (most aligned, shortest paths first). ANALYZE "
        "the tree, decide which branches are relevant to the question, and call "
        "expand_branches(['1','2']) on those you choose to see their full triples "
        "+ CVT attributes. Then answer from what you expanded.\n\n"
        + "\n\n".join(overview_blocks)
    )

    return _json_result({
        "n_patterns": len(ranked),
        "overview": overview_text,
        "candidates": ctx.selected_candidates[:50],
    })


def _cvt_attr_summary(triples, candidates=None, max_lines: int = 60) -> str:
    """Per-CVT attribute summary for the answer step.

    Inline-resolves CVT nodes so each holder's tenure attrs sit on one line
    (e.g. "Abdullah al-Thani: from=2014-06-09, to=(incumbent), basic_title=Prime minister").
    Unlike format_subgraph_with_cvt, KEEPS `has_no_value` (rendered as
    `<attr>=(incumbent)`) — that "no end date" marker is the signal that picks
    the current office-holder, and stripping it (as noise) is exactly why the
    model cannot tell who is incumbent on exclusive relations like Libya PMs.
    Filters only true noise (type./common./freebase bookkeeping minus has_no_value).
    """
    from kgqa.traversal.cvt import is_cvt_like
    _NOISY_PREFIX = ("type.", "common.", "kg.", "user.", "base.ontologies.")
    _NOISY_SHORT = {"type", "types", "instance", "instances", "notable_types",
                    "topic_equivalent_webpage", "webpage", "mid", "guid",
                    "key", "keys", "permission", "is_reviewed", "no_value"}
    _HOLDER_RELS = {"office_holder", "actor", "director", "spouse"}
    cand_set = {str(c) for c in (candidates or [])}  # attrs key by these names, not m-IDs

    def _short(r):
        return r.rsplit(".", 1)[-1] if r else r

    def _noisy(r):
        r = str(r)
        return r.startswith(_NOISY_PREFIX) or _short(r) in _NOISY_SHORT

    cvt_attrs = {}   # cvt_name -> [(short_rel, value)]
    cvt_holder = {}  # cvt_name -> holder entity (the named entity it points to)
    for tr in triples:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            continue
        h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
        if not (is_cvt_like(h) and not is_cvt_like(t)):
            continue
        if _noisy(r):
            continue
        s = _short(r)
        lst = cvt_attrs.setdefault(h, [])
        if (s, t) not in lst:              # per-CVT dedup: the same attr=value recurs across
            lst.append((s, t))             # inverse/variant relation patterns (4× explosion)
        # Resolve the CVT to a NAMED entity (never the opaque m-ID): holder-rel
        # first, else any named candidate it points to.
        if s in _HOLDER_RELS:
            cvt_holder[h] = t
        elif h not in cvt_holder and str(t) in cand_set:
            cvt_holder[h] = t
    # CVTs still unmapped: fall back to their first named entity (not the m-ID).
    for cvt in list(cvt_attrs):
        if cvt not in cvt_holder:
            named = [v for _, v in cvt_attrs[cvt] if not is_cvt_like(str(v))]
            if named:
                cvt_holder[cvt] = named[0]

    lines = []
    # incumbents (has_no_value = no end date) first so the current holder surfaces
    def _sort_key(item):
        cvt, attrs = item
        has_incumbent = any(s == "has_no_value" for s, _ in attrs)
        holder = cvt_holder.get(cvt, cvt)
        return (0 if has_incumbent else 1, holder)

    _seen = set()  # dedup identical (holder, attrs) lines — don't repeat identical attrs
    for cvt, attrs in sorted(cvt_attrs.items(), key=_sort_key):
        holder = cvt_holder.get(cvt, cvt)
        nh = normalize(holder)
        parts = []
        for s, v in attrs:
            if normalize(str(v)) == nh:
                continue                   # back-edge: holder listed as its own attr value
            if s == "has_no_value":
                # value names the attribute that is absent (e.g. "To") → incumbent
                parts.append(f"{str(v).lower()}=(incumbent)")
            else:
                parts.append(f"{s}={v}")
        if parts:
            ln = f"  - {holder}: {', '.join(parts[:8])}"
            if ln not in _seen:
                _seen.add(ln); lines.append(ln)
    return "\n".join(lines[:max_lines])


def _constraint_attr_summary(triples, candidates):
    """Detect DISCRIMINATING attributes among sibling candidates in the expand
    tree — attrs whose VALUES DIFFER across candidates (incl. present-vs-absent).

    Works on the expand triples (the tree), not a flat subgraph re-resolve. For
    each candidate, resolves its direct + CVT-mediated attrs from the triples,
    finds attrs where values differ across candidates, and returns a highlighted
    summary so the answer stage can apply the constraint.

    Handles both HOLDER-type (PM from/to via office_holders CVT) and VALUE-type
    (population/CPI/champion as direct or CVT attr of the candidate).
    """
    from kgqa.traversal.cvt import is_cvt_like
    _NOISY_PREFIX = ("type.", "common.", "kg.", "user.", "base.ontologies.")
    _NOISY_SHORT = {"type", "types", "instance", "instances", "notable_types",
                    "topic_equivalent_webpage", "webpage", "mid", "guid",
                    "key", "keys", "permission", "is_reviewed", "article",
                    "description", "alias", "name"}  # NOTE: no_value KEPT (incumbent signal)

    def _short(r):
        return r.rsplit(".", 1)[-1] if r else r

    def _noisy(r):
        r = str(r)
        return r.startswith(_NOISY_PREFIX) or _short(r) in _NOISY_SHORT

    # Build candidate -> {attr_short: set(values)} from triples.
    # Two patterns: direct (cand, rel, val) and CVT (cand, rel, CVT)->(CVT, rel2, val).
    cand_set = {str(c) for c in candidates if c}
    cvt_out = {}  # cvt -> [(short_rel, value)]
    for tr in triples:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            continue
        h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
        if is_cvt_like(h) and not is_cvt_like(t):
            if not _noisy(r):
                cvt_out.setdefault(h, []).append((_short(r), t))

    holder_attrs = {}  # candidate -> {attr_short: set(values)}
    for cvt, attrs in cvt_out.items():
        holder = None
        for s, v in attrs:
            if s in ("office_holder", "actor", "director", "spouse", "student",
                     "champion", "person", "student_"):
                holder = v
                break
        if holder is None:
            # resolve to a named entity (never the opaque m-ID): a candidate it
            # points to, else its first named value.
            holder = next((v for _, v in attrs if str(v) in cand_set), None)
            if holder is None:
                named = [v for _, v in attrs if not is_cvt_like(str(v))]
                holder = named[0] if named else None
        if holder is None:
            continue  # CVT with no named entity — skip it (don't emit an m-ID key)
        for s, v in attrs:
            holder_attrs.setdefault(str(holder), {}).setdefault(s, set()).add(str(v))
    # Also direct attrs of named candidates
    for tr in triples:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            continue
        h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
        if h in cand_set and not is_cvt_like(t) and not _noisy(r):
            holder_attrs.setdefault(h, {}).setdefault(_short(r), set()).add(str(t))

    if len(holder_attrs) < 2:
        return ""

    # Find discriminating attrs: values differ across >=2 candidates (incl ABSENT).
    all_attrs = set()
    for a in holder_attrs.values():
        all_attrs |= set(a.keys())
    discrim = {}
    for attr in all_attrs:
        vals = {h: (a[attr] if attr in a else "ABSENT") for h, a in holder_attrs.items()}
        if len(vals) >= 2:
            repr_vals = {h: (frozenset(v) if isinstance(v, set) else v) for h, v in vals.items()}
            if len(set(repr_vals.values())) > 1:
                discrim[attr] = vals

    if not discrim:
        return ""

    lines = ["DISCRIMINATING ATTRIBUTES (values differ across candidates — apply as constraint):"]
    for attr, vals in sorted(discrim.items()):
        parts = []
        for h, v in list(vals.items())[:6]:
            vstr = ", ".join(sorted(v)) if isinstance(v, set) and v else str(v)
            parts.append(f"{h}={vstr}")
        lines.append(f"  ⚡ {attr}: " + " | ".join(parts))
    return "\n".join(lines)


async def _relevant_constraint_summary(all_cands, ctx, session, top_k: int = 3, covered=None) -> str:
    """Sample the question-relevant CONSTRAINT relation from each answer
    candidate's neighborhood and surface its values across candidates.

    Why: the discriminator relation normally reaches expand evidence via
    decompose->select, but if the model selected the wrong branch, the
    constraint is absent and the model over-emits (e.g. emits every PM
    instead of the incumbent). This walks each candidate's 1-hop +
    CVT-bridged neighborhood from the FULL edge set on ctx (independent of
    which branches were selected), GTE-ranks the neighborhood relations
    against the question, and surfaces the top-k relations' values across
    candidates so the model can discriminate.

    Gates / self-limiting:
    - needs >=2 candidates and a non-empty question;
    - skips CVT-id values (empty mediator nodes) and bookkeeping relations;
    - masks relations whose values are identical across all candidates
      (no discrimination) — keeps present-vs-absent as discriminating;
    - returns "" when no question-relevant discriminating relation is present
      (so data-gap cases like empty statistical_region CVTs inject nothing).
    """
    from kgqa.traversal.cvt import is_cvt_like
    if session is None or len(all_cands) < 2 or not getattr(ctx, "question", ""):
        return ""

    _NOISY_PREFIX = ("type.", "common.", "kg.", "user.", "base.ontologies.", "freebase.")
    _NOISY_SHORT = {"type", "types", "instance", "instances", "notable_types",
                    "topic_equivalent_webpage", "webpage", "mid", "guid", "key",
                    "keys", "permission", "is_reviewed", "article", "description",
                    "alias", "name", "no_value"}

    def _short(r):
        r = str(r) if r else ""
        return r.rsplit(".", 1)[-1].replace("_", " ") if r else r

    def _noise(r):
        r = str(r) if r else ""
        return r.startswith(_NOISY_PREFIX) or r.rsplit(".", 1)[-1] in _NOISY_SHORT

    ents, rel_txt = ctx.ents, ctx.rel_texts
    h_ids, r_ids, t_ids = ctx.h_ids, ctx.r_ids, ctx.t_ids

    # candidate name -> idx; sibling set = candidates themselves. Sibling
    # values are structural links (stadium<->team), not constraints -> excluded.
    name_to_idx = {}
    for i, e in enumerate(ents):
        name_to_idx.setdefault(normalize(e), i)
    sibling_idx = {name_to_idx[n] for n in (normalize(c) for c in all_cands)
                   if n in name_to_idx}
    if len(sibling_idx) < 2:
        return ""

    # PASS 1 (single over edges): per candidate, split 1-hop relations into
    #   (a) CVT-pointing ENTRY relations -> cand_cvt_rels[ci][rel] = {cvt_idx}
    #   (b) direct named/literal values   -> cand_direct[ci][rel]   = {value}
    cand_cvt_rels = {ci: {} for ci in sibling_idx}
    cand_direct = {ci: {} for ci in sibling_idx}
    touched_cvts = set()
    for i in range(len(h_ids)):
        H, R, T = h_ids[i], r_ids[i], t_ids[i]
        rname = rel_txt[R] if 0 <= R < len(rel_txt) else ""
        if _noise(rname):
            continue
        rt = _short(rname)
        for ci, other in ((H, T), (T, H)):
            if ci not in sibling_idx:
                continue
            oname = ents[other] if 0 <= other < len(ents) else ""
            if is_cvt_like(oname):
                if covered and oname in covered:
                    continue  # CVT already in expand evidence (select hit) -> FALLBACK ONLY: skip to avoid loop
                cand_cvt_rels[ci].setdefault(rt, set()).add(other)
                touched_cvts.add(other)
            elif oname and other not in sibling_idx:
                cand_direct[ci].setdefault(rt, set()).add(oname)

    # GTE-rank the CVT-POINTING entry relations vs the question. Restricting to
    # CVT-pointing relations (the user's "is this relation a CVT attribute?")
    # avoids direct relations of noisy sibling candidates dominating the rank
    # (e.g. a country candidate's "national anthem" matching the question's
    # filter clause and crowding out the leaders' office relations).
    all_rels = sorted({r for ci in sibling_idx for r in cand_cvt_rels[ci].keys()})
    if not all_rels:
        return ""
    if len(all_rels) <= top_k:
        top_rels = all_rels
    else:
        try:
            rows = await gte_retrieve(session, ctx.question, all_rels,
                                      candidate_texts=all_rels, top_k=top_k)
        except Exception as _e:
            import sys as _sys
            print(f"  GTE error in _relevant_constraint_summary: {_e}", file=_sys.stderr)
            return ""
        top_rels = [r.get("candidate") for r in rows
                    if isinstance(r.get("candidate"), str) and r["candidate"] in all_rels]
    top_rels = list(dict.fromkeys(top_rels))[:top_k]
    if not top_rels:
        return ""

    # PASS 2 (single over edges): precompute attrs of every touched CVT once
    # (exclude siblings, CVT ids, noise). cvt_attrs[cvt] = {attr: {value}}.
    cvt_attrs = {}
    if touched_cvts:
        for i in range(len(h_ids)):
            H, R, T = h_ids[i], r_ids[i], t_ids[i]
            cvt = H if H in touched_cvts else (T if T in touched_cvts else None)
            if cvt is None:
                continue
            rname = rel_txt[R] if 0 <= R < len(rel_txt) else ""
            if _noise(rname):
                continue
            other = T if H == cvt else H
            oname = ents[other] if 0 <= other < len(ents) else ""
            if not oname or is_cvt_like(oname) or other in sibling_idx:
                continue
            cvt_attrs.setdefault(cvt, {}).setdefault(_short(rname), set()).add(oname)

    # Per candidate: merge attrs from the top-rel-pointed CVTs + direct values.
    cand_nb = {}
    for ci in sibling_idx:
        nb = {}
        for rt in top_rels:
            if rt in cand_direct[ci]:
                nb.setdefault(rt, set()).update(cand_direct[ci][rt])
            for cvt in cand_cvt_rels[ci].get(rt, ()):
                for at, vals in cvt_attrs.get(cvt, {}).items():
                    nb.setdefault(at, set()).update(vals)
        if nb:
            cand_nb[ci] = nb
    if len(cand_nb) < 2:
        return ""
    idx_to_name = {ci: ents[ci] for ci in cand_nb}

    # Surface DISCRIMINATING attrs (values differ across >=2 candidates OR
    # present-vs-absent) — the CVT attrs the model needs (e.g. from /
    # has_no_value=incumbent for office_holders).
    all_attr = sorted({a for nb in cand_nb.values() for a in nb})
    n_cand = len(cand_nb)
    kept = []
    for at in all_attr:
        present = [sorted(v) for nb in cand_nb.values() if (v := nb.get(at))]
        absent = n_cand - len(present)
        repr_vals = {tuple(v) for v in present}
        if (len(present) >= 2 and len(repr_vals) > 1) or (present and absent):
            kept.append(at)
    if not kept:
        return ""
    kept = kept[:8]

    lines = ["QUESTION-RELEVANT ATTRIBUTES (compare values across candidates to apply the question's constraint):"]
    for at in kept:
        per = {idx_to_name[ci]: sorted(v) for ci, nb in cand_nb.items() if (v := nb.get(at))}
        if not per:
            continue
        absent = [n for n in idx_to_name.values() if n not in per]
        parts = [f"{c}={','.join(v[:2])}" for c, v in list(per.items())[:6]]
        if absent:
            parts.append(f"{len(absent)} other(s) lack it")
        lines.append(f"  - {at}: " + " | ".join(parts))
    if len(lines) <= 1:
        return ""
    return "\n".join(lines[:9])


async def _do_expand_branch(args: Dict[str, Any], ctx, session) -> str:
    """Expand one or more branches to show their FULL evidence.

    Accepts both the batch form (branch_ids: ['1','2']) and the legacy single
    form (branch_id: '1'). Returns merged evidence across all selected branches
    using the pre-built triples + tree from select (no re-build needed — the
    select stage already ran build_pattern_evidence_triples with CVT expansion).
    """
    bids = args.get("branch_ids") or args.get("branch_id_list") or []
    if isinstance(bids, str):
        bids = [bids]
    bids = [str(b) for b in bids if str(b).strip()]
    if not bids:
        bids = [str(args.get("branch_id", ""))]
    bids = [b for b in bids if b]
    if not bids:
        return _json_result({"error": "no branch_ids provided. Pass a list like ['1','2'].",
                             "valid_branches": list(ctx.branches.keys())})

    unknown = [b for b in bids if b not in ctx.branches]
    if unknown:
        return _json_result({"error": f"unknown branch_ids {unknown}",
                             "valid_branches": list(ctx.branches.keys())})

    # Use pre-built evidence from select (triples + tree_lines already contain
    # the full CVT-expanded data). Just merge across selected branches.
    all_triples = []
    seen_t = set()
    all_cands = []
    seen_c = set()
    per_branch = []
    tree_sections = []
    for bid in bids:
        br = ctx.branches[bid]
        cands = br.get("candidates", [])
        triples = br.get("triples", [])
        tree_lines = br.get("tree_lines", [])
        readable = br.get("readable", "")
        for cand in cands:
            nc = normalize(cand)
            if nc not in seen_c:
                seen_c.add(nc); all_cands.append(cand)
        for tr in triples:
            key = tuple(str(x) for x in tr)
            if key not in seen_t:
                seen_t.add(key); all_triples.append(tr)
        per_branch.append({"branch_id": bid, "readable": readable,
                           "candidates": cands[:30], "n_triples": len(triples)})
        if tree_lines:
            tree_sections.append(f"=== branch {bid}: {readable} ===\n" +
                                 "\n".join(tree_lines))

    triple_strs = [f"({h}, {r}, {t})" for h, r, t in all_triples[:80]]
    cand_attrs = _cvt_attr_summary(all_triples, all_cands)
    # Constraint-awareness: detect discriminating attrs among sibling candidates
    # in the tree + append them so the answer stage can apply the constraint.
    constraint_attrs = _constraint_attr_summary(all_triples, all_cands)
    if constraint_attrs:
        cand_attrs = (cand_attrs + "\n" + constraint_attrs) if cand_attrs else constraint_attrs
    # FALLBACK constraint sampling: surface constraints NOT already in the
    # expand evidence (select missed them). `covered` = CVT entities already in
    # the selected-branch triples — the helper skips them, so it only fills
    # gaps (avoids re-deriving what select already provided = loop) AND drops
    # already-selected filter-clause CVTs that would otherwise dominate GTE.
    covered = {str(e) for tr in all_triples for e in (tr[0], tr[2]) if is_cvt_like(str(e))}
    rel_constraint = await _relevant_constraint_summary(all_cands, ctx, session, covered=covered)
    if rel_constraint:
        cand_attrs = (cand_attrs + "\n" + rel_constraint) if cand_attrs else rel_constraint
    # Merge expand-revealed candidates into ctx.all_candidates so the answer
    # off-pool check (in _do_answer) validates against the FULL select+expand
    # pool, not just the select-stage candidates. Without this, a correct
    # answer entity revealed only by expand looks "off-pool" and gets wrongly
    # rejected (the model then fails to recover -> empty answer). The check
    # should fire only when an entity NEVER appeared in any expanded branch.
    _existing_all = list(getattr(ctx, "all_candidates", []) or [])
    _seen_all = {normalize(c) for c in _existing_all}
    for _cand in all_cands:
        _nc = normalize(_cand)
        if _nc not in _seen_all:
            _seen_all.add(_nc); _existing_all.append(_cand)
    ctx.all_candidates = _existing_all
    payload = {
        "branches_expanded": bids,
        "n_branches": len(bids),
        "candidates": all_cands[:100],
        "n_candidates": len(all_cands),
        "triples": triple_strs,
        "n_triples": len(all_triples),
        "candidate_attrs": cand_attrs,
        "per_branch": per_branch,
    }
    if tree_sections:
        payload["tree"] = "\n\n".join(tree_sections)
    return _json_result(payload)

# '#center::relation' — '::' NOT '|': the flat protocol splits entity lists on
# '|', so a '#Smokey Robinson|track' ref parses as TWO entities. '|' still
# accepted in the regex for refs that arrive intact (quoted JSON).
_BRANCH_RE = re.compile(r"^#(.{1,200}?)(?:::|\|)(.{1,300})$")


def _expand_branch_refs(ctx, entities):
    """Expand '#center|relation' answer refs into ALL local entities of that edge
    pattern (list-question protocol). The retrieve_subgraph renderer caps huge
    pattern lines and advertises this ref; expanding here lets the model answer
    with the full list via a compressed submission instead of enumerating
    hundreds of entities. Relation matches full name, dotted suffix, or short
    form. Falls back to the literal string when nothing matches (a stray '#a|b'
    should still be visible in the offpool feedback, not silently dropped)."""
    from kgqa.core.utils import normalize as _norm
    out = []
    for e in entities:
        m = _BRANCH_RE.match(e.strip())
        if not m:
            out.append(e)
            continue
        center, rel = m.group(1).strip(), m.group(2).strip()
        H, R, T = ctx.h_ids, ctx.r_ids, ctx.t_ids
        ents, rels = ctx.ents, ctx.rels
        got, seen = [], set()
        for k in range(len(H)):
            r_name = str(rels[R[k]]) if 0 <= R[k] < len(rels) else ""
            if not (r_name == rel or r_name.endswith("." + rel)
                    or r_name.rsplit(".", 1)[-1] == rel):
                continue
            h_name, t_name = str(ents[H[k]]), str(ents[T[k]])
            if _norm(h_name) == _norm(center):
                other = t_name
            elif _norm(t_name) == _norm(center):
                other = h_name
            else:
                continue
            if (not other.strip() or is_cvt_like(other)
                    or _norm(other) in seen or _norm(other) == _norm(center)):
                continue
            seen.add(_norm(other))
            got.append(other)
        out.extend(got if got else [e])
    return out


def _do_answer(args: Dict[str, Any], ctx) -> str:
    """Capture the model's answer entities. The `entities` array is the
    authoritative answer surface (schema-required).

    Fallback: the model frequently emits the §Answer checklist's ANSWER field
    as the tool arg key (``ANSWER``) instead of the schema's ``entities`` — it
    knows the answer but the call slips format. Accept ``ANSWER``/``answer`` so
    a correct answer is never dropped to empty on a key-name mismatch."""
    entities = args.get("entities")
    if entities is None:
        entities = args.get("ANSWER") or args.get("answer") or []
    if isinstance(entities, str):
        # checklist ANSWER may be a comma/pipe-separated string
        entities = [e.strip() for e in entities.replace("|", ",").split(",") if e.strip()]
    if not isinstance(entities, list):
        entities = []
    entities = [str(e).strip() for e in entities if str(e).strip()]
    # value-level guard (harness fix P6): belt-and-braces behind the ingress
    # sanitizer — if a reasoning-leak fragment survived into an entity value
    # (glued onto a name), strip it here so the offpool check and the final
    # answer never carry the artifact.
    entities = [strip_reasoning_leak(e).strip() for e in entities]
    entities = [e for e in entities if e]
    # MID-format entities can never be correct (gold is always display names);
    # evidence trees sometimes display raw mids (m.0h2z5vr) which the offpool
    # check would otherwise pass — strip them outright. ALIAS FORM (Wave-1,
    # V2.2 specimen: the recorded final answer "m.0cr70y2 (Freemasonry)") —
    # an event node written as '<mid> (<name>)' slipped the bare-mid strip,
    # so a mid followed by an optional parenthetical/whitespace tail is
    # stripped too: an event node is never a legal answer in any written form.
    import re as _re_mid
    _pre_mid_strip = entities
    entities = [e for e in entities
                if not _re_mid.fullmatch(
                    r"[mg]\.[0-9a-z_]{2,}(?:\s*\([^)]*\))?\s*",
                    e.strip().lower())]
    # SYSTEM-LEVEL REJECTION (2026-08-21, Lauren/Kim specimens): a submission
    # consisting ENTIRELY of event-node mids was silently stripped to empty and
    # ACCEPTED as an empty answer — measured 34% of all empty answers. Reject
    # ONCE with the fix instruction (same one-shot semantics as the offpool
    # check); the retry with named attribute values passes normally.
    # NO BYPASS (Wave-1): the retry flag used to gate the whole check, letting
    # a SECOND all-event-node submission fall through and be recorded — that
    # is how the alias-form specimen above reached llm_answer_preds. Hard
    # invariant: answer ⊆ retrieved evidence, and an event node is never
    # evidence. After the one corrective error, a further all-event-node
    # submission falls through with only what survived the strip (possibly []).
    if _pre_mid_strip and not entities:
        if not getattr(ctx, "_answer_cvt_retried", False):
            ctx._answer_cvt_retried = True
            return _json_result({
                "error": ("answer entities were ALL event nodes (m./g. ids — "
                          "an id with a parenthetical name, e.g. "
                          "'m.0cr70y2 (Freemasonry)', is still the event "
                          "node: submit the named attribute itself). An event "
                          "node is an abstract RECORD — it names no thing, so accepting "
                          "it would score an EMPTY answer (see §7.5). The entities "
                          "INSIDE each event's bracket are the world: re-call `answer` "
                          "binding, per variable, the attribute entity whose KEY answers "
                          "the question (award question → the award= value; residence → "
                          "location=; who played → actor=; which film → film=)."),
            })
        # corrective error already spent and STILL all event nodes: accept the
        # sanitized (possibly empty) list — event-node strings are never
        # recorded as the answer.
    # branch-ref expansion must precede the offpool check: expanded entities are
    # graph-derived (always on-pool), the raw '#center|relation' string is not.
    entities = _expand_branch_refs(ctx, entities)
    entities = [e for e in entities if e]

    # Off-pool check: answer entities should come from the retrieved evidence.
    # If the model emits an entity NOT in the candidate pool, flag it ONCE (it
    # may be hallucinating or answering before expanding) and reject with the
    # pool so it re-answers from evidence. NO BYPASS (Wave-1): the retry flag
    # used to gate the whole check, letting a SECOND off-pool submission fall
    # through and be recorded verbatim. Hard invariant: answer ⊆ retrieved
    # evidence — a LATER off-pool submission is sanitized to its pool-hit
    # entities (possibly []) and accepted; the terminal answer is never
    # blocked forever, but off-pool entities are never recorded as it.
    # candidate_hit (normalized substring + fuzzy) lets legitimate name
    # variants pass.
    # LEGAL-ENTITY BASIS (user ruling 2026-09-08): the answer must be an
    # entity that APPEARED in the walk — not necessarily one the DISPLAY
    # showed. The license filter keeps out-of-license edges out of the
    # render (anti-drift for the model's eyes), but any entity the walk
    # enumerated is retrieved evidence and is a legal answer (1171:s2
    # specimen: James Earl Jones arrived on a wandered servicemembers edge,
    # filtered from the display yet the correct answer).
    # STRUCTURED SOURCE FIRST (redesign ZEROETH RULE 2026-09-24): the
    # answer-legal surface is the EvidenceLog's structured pool (endpoints
    # + CVT record values, walked edges' named endpoints). The legacy
    # multi-source assembly below stays as fallback until the log's
    # coverage is verified on the cohort.
    # STRUCTURED SOURCE FIRST (redesign ZEROETH RULE 2026-09-24): the
    # answer-legal surface is the EvidenceLog's structured pool (endpoints
    # + CVT record values, walked edges' named endpoints). The legacy
    # multi-source assembly below stays as fallback until the log's
    # coverage is verified on the cohort.
    # R3-b (SEQ_RENDER_V3, specs/r3b_render_pools): the off-pool domain is
    # the VIEW layer's existence('episode') — the normalized endpoints of
    # every fact ANY delivered call materialized (walk endpoints AND CVT
    # attr VALUE endpoints, the same domain the checkpoint hallucination
    # filter judges on since D3-0). The ad-hoc assembly (EvidenceLog pool
    # ∪ accumulated ∪ CVT-neighbor expansion) runs only on the OFF face /
    # when no ledger exists. The corrective REMINDER's candidate rows
    # (pool[:15] / candidate_pool) ride the same switched source.
    from kgqa.agent.views import render_v3 as _rv3_ans
    _rlog_ans = getattr(ctx, "evidence_records", None)
    if _rv3_ans() and _rlog_ans is not None:
        from kgqa.agent import views as _views_ans
        pool = sorted(_views_ans.existence(_rlog_ans, "episode"))
    else:
        _elog = getattr(ctx, "evidence_log", None)
        if _elog is not None and _elog.calls:
            pool = _elog.answer_pool()
        else:
            pool = list(getattr(ctx, "all_candidates", []) or [])
            pool += [e for e in (getattr(ctx, "walk_seen_entities", []) or [])
                     if e not in pool]
        # STRUCTURED HARVEST (user ruling 2026-09-22, revision — never parse
        # rendered text): the pool is built from the TRIPLES the text layer
        # renders. Sources: (a) the accumulated delivered triples — including
        # the CVT-expansion pass-through edges; (b) the case-graph expansion
        # around every delivered CVT node — the structured form of the bracket
        # attribute values the renderer shows. Origin: 1379-s0, where the
        # walk-side pool missed a same-edge value (Composer legal, Priest
        # off-pool on ONE rendered profession line) — the delivered-triple
        # record is the authority the display renders from.
        from kgqa.traversal.cvt import is_cvt_like as _icl_hv
        _named_extra, _cvts = set(), set()
        for _h, _r, _t in (getattr(ctx, "accumulated_triples", None) or set()):
            for _x in (_h, _t):
                _xs = str(_x) if _x is not None else ""
                if not _xs:
                    continue
                if _icl_hv(_xs):
                    _cvts.add(_xs)
                else:
                    _named_extra.add(_xs)
        if _cvts:
            # _full_adj is an INDEX-KEYED LIST adjacency (adj[ent_idx] = [(rel,
            # other)...]), and _cvts holds CVT NAMES — map name→idx first. The
            # old `.get(_cv, ())` treated the list as a dict and raised
            # "'list' object has no attribute 'get'" on every answer whose
            # delivered triples carried a CVT endpoint (124 crashes in the
            # 2026-09-23 v24b run — the answer dispatch died BEFORE the
            # off-pool corrective feedback, locking wrong names in).
            _adj_hv = _full_adj(ctx)
            _n2i = {}
            for _i, _e in enumerate(ctx.ents):
                _n2i.setdefault(str(_e), _i)
            for _cv in _cvts:
                _ci2 = _n2i.get(_cv)
                if _ci2 is None or not (0 <= _ci2 < len(_adj_hv)):
                    continue
                for _r2, _w in _adj_hv[_ci2]:
                    _wn = ctx.ents[_w] if 0 <= _w < len(ctx.ents) else ""
                    if _wn and not _icl_hv(str(_wn)):
                        _named_extra.add(str(_wn))
        pool += [e for e in _named_extra if e not in pool]
    if entities and pool:
        offpool = [e for e in entities if not candidate_hit(pool, [e])]
        if offpool:
            if not getattr(ctx, "_answer_offpool_retried", False):
                ctx._answer_offpool_retried = True
                return _json_result({
                    "error": (f"answer entities NOT in the retrieved evidence: {offpool}. "
                              "The answer must COMPLETELY MATCH an entity retrieved in the "
                              "evidence — use the full entity name exactly as it appears "
                              "(a partial word or extracted fragment is not an entity). "
                              "Entities in your retrieved evidence pool include: "
                              f"{pool[:15]}. Pick from these or expand the relevant "
                              "branches, then re-call `answer` with complete entity names."),
                    "offpool": offpool,
                    "candidate_pool": pool[:30],
                })
            # corrective error already spent and STILL off-pool: sanitize —
            # keep only the pool-hit entities (possibly []) and accept those.
            entities = [e for e in entities if e not in offpool]

    ctx.llm_answer_preds = entities
    ctx.llm_answer_str = " | ".join(entities)
    return _json_result({"entities": entities})


def join_path_rescue(ctx, fids, max_hops=3, reveal=False):
    """Shortest bridge between two subgraphs' walked endpoint sets
    (JOIN-PATH RESCUE, user ruling 2026-09-23): bidirectional BFS over
    the case graph from side A's pool to side B's pool; returns a
    rendered 'A --rel--> ... --rel--> B' line, or None. CVT/id transit
    nodes are transparent (same named-hop semantics as the pool).
    FAIRNESS CONTRACT (user ruling 2026-09-23): the search is POOL-TO-POOL
    shortest + GTE-on-question ranking — NEVER gold-anchored. The bridge
    must not be constructed to pass through the answer; any landing near
    the answer is the natural consequence of pool proximity, and the
    external feedback carries existence only (no chain, no endpoint)."""
    def _name_idx(name):
        nm = str(name)
        for i, e in enumerate(ctx.ents):
            if str(e) == nm:
                return i
        return None

    if len(fids) < 2:
        return None
    from kgqa.agent.tools import _full_adj
    adj = _full_adj(ctx)
    n = len(ctx.ents)
    import re as _re_id
    def _cvtl(i):
        # C1 single-source: CVT-or-ID transit — entity_kinds.is_transit
        from kgqa.agent.entity_kinds import is_transit as _ek_tr
        nm = str(ctx.ents[i]) if 0 <= i < n else ""
        return _ek_tr(nm)
    # KEY CANONICALIZATION (integration audit 2026-09-23): join_flag
    # passes DECLARED fids ('sg2.f2') but the pools are keyed through
    # fact_key_map's canonical fid ('f2') — the raw lookup missed every
    # pool and the rescue silently returned None in real rollouts
    # (function verified fine in isolation). Resolve both ways.
    # R3-b (SEQ_RENDER_V3, specs/r3b_render_pools): the search domain is
    # the VIEW layer — each side's existence('tree') ∪ frontier node ids
    # (the same domain _bridge_path_v2 bridges), replacing the tool-side
    # candidate pool as this mechanism's data source. The pool reads
    # below stay verbatim as the OFF face.
    from kgqa.agent.views import render_v3 as _rv3
    if _rv3():
        _rlog = getattr(ctx, "evidence_records", None)
        if _rlog is None:
            return None
        from kgqa.agent import views as _views
        from kgqa.core.utils import normalize as _nrv
        _n2i = {}
        for _i, _e in enumerate(ctx.ents):
            _n2i.setdefault(_nrv(str(_e)), _i)
        pools = {}
        for _fid in fids[:2]:
            _idxs = {int(_i) for _i in
                     (_views.frontier(_rlog, "tree", _fid)
                      .get("node_ids") or ())}
            for _nm in _views.existence(_rlog, "tree", _fid):
                _j = _n2i.get(_nm)
                if _j is not None:
                    _idxs.add(_j)
            if _idxs:
                pools[_fid] = _idxs
        if len(pools) < 2:
            return None
    else:
        _fkm = getattr(ctx, "fact_key_map", None) or {}
        _fcp = getattr(ctx, "fact_candidate_pool", None) or {}
        _fev = getattr(ctx, "fact_evidence", None) or {}
        pools = {}
        for _fid in fids[:2]:
            _key = _fkm.get(_fid, _fid)
            pl = _fcp.get(_key) or _fcp.get(_fid)
            if not pl:
                pl = set(_fev.get(_key) or ()) or set(_fev.get(_fid) or ())
            if pl:
                pools[_fid] = {_name_idx(p) for p in pl}
        if len(pools) < 2:
            return None
    (ka, sa), (kb, sb) = pools.items()
    sa, sb = {i for i in sa if i is not None}, {i for i in sb if i is not None}
    if sa & sb:
        shared = sa & sb
        return f"(pools already share: {' | '.join(str(ctx.ents[i]) for i in list(sorted(shared))[:5])})"
    # BFS from A's pool, target B's pool; named-hop cost, transit free
    parent = {}
    seen = set(sa)
    frontier, cost = set(sa), 0
    hits = []
    while frontier and cost < max_hops and not hits:
        nxt = set()
        stack = list(frontier)
        while stack:
            i = stack.pop()
            for _rr, other in (adj[i] if 0 <= i < len(adj) else ()):
                if other in seen or not (0 <= other < n):
                    continue
                seen.add(other)
                if other in sb:
                    parent[other] = (i, _rr)
                    hits.append(other)   # collect the whole landing layer
                    continue
                if _cvtl(other):
                    stack.append(other)
                    parent.setdefault(other, (i, _rr))
                else:
                    parent[other] = (i, _rr)
                    nxt.add(other)
        frontier = nxt
        cost += 1
    if not hits:
        return None
    # GTE SEMANTIC RANKING of the candidate bridges (user ruling
    # 2026-09-23): hop count first (BFS layer), semantics WITHIN equal
    # hops — the shortest bridge is only "best" when its relations
    # actually encode the question's join. Sync embed call: the rescue
    # fires at most once per case (empty-intersection only).
    def _full_path(hit_n):
        # backtrack the RAW node chain (CVT segments kept) with the
        # edge relation of each step — named-pair rendering failed on
        # CVT-mediated bridges (no DIRECT edge between the named ends).
        # GTE UNIFICATION (2026-10-02): the chain starts EMPTY and each
        # loop iteration appends the node it is LEAVING — the old
        # `[hit_n]` seed duplicated the landing node, rendering a bogus
        # `--?--> ?` tail on every rescue (surfaced by the JOIN bridge
        # now delegating here; see _bridge_path_v2).
        path, node = [], hit_n
        while node not in sa and node in parent:
            path.append(node)
            node = parent[node][0]
        if node in sa:
            path.append(node)      # the A-side seed ends the chain
        path.reverse()
        if len(path) < 2:
            return None, None
        named = [i for i in path if not _cvtl(i)]
        if len(named) < 2:
            return None, None
        return named, path

    cands = []
    for h in hits[:12]:
        named, path = _full_path(h)
        if not named:
            continue
        cands.append((named, path))
    if not cands:
        return None
    _scores = None
    try:
        import requests as _rq
        def _chain_txt(path):
            segs_r = []
            for a, b in zip(path, path[1:]):
                for rr, other in (adj[a] if 0 <= a < len(adj) else ()):
                    if other == b:
                        segs_r.append(
                            ".".join(str(ctx.rels[rr]).rsplit(".", 2)[-2:]))
                        break
            return " ⭢ ".join(segs_r)
        _txts = [_chain_txt(path) for _, path in cands]
        _q = str(getattr(ctx, "question", "") or "")
        import os as _os, random as _rnd
        _gus = [u for u in _os.environ.get("GTE_API_URLS", "").split(",") if u] \
            or ["http://127.0.0.1:8003"]
        _eb = _rq.post(_rnd.choice(_gus) + "/embed",
                       json={"texts": [_q] + _txts}, timeout=10).json()["embeddings"]
        import math as _m
        def _cos(a, b):
            d = sum(x * y for x, y in zip(a, b))
            na = _m.sqrt(sum(x * x for x in a)) or 1.0
            nb = _m.sqrt(sum(x * x for x in b)) or 1.0
            return d / (na * nb)
        _scores = [_cos(_eb[0], e) for e in _eb[1:]]
    except Exception:
        _scores = None
    order = sorted(range(len(cands)),
                   key=lambda k: (len(cands[k][0]) - 1,
                                  -(_scores[k] if _scores else 0)))
    named, path = cands[order[0]]
    segs = [str(ctx.ents[path[0]])]
    for a, b in zip(path, path[1:]):
        _rs = "?"
        for rr, other in (adj[a] if 0 <= a < len(adj) else ()):
            if other == b:
                _rs = ".".join(str(ctx.rels[rr]).rsplit(".", 2)[-2:])
                break
        if _cvtl(b):
            segs.append(f" --{_rs}-->")
        elif _rs == "object.name":
            continue          # id-node → same-name node, never info
        elif not reveal and b == path[-1]:
            # ANSWER-LEAK GUARD (user ruling 2026-09-23): the landing node
            # IS the merge point (the answer) — the harness never reveals
            # it. Mind-map form: full relation chain, hidden endpoint.
            segs.append(f" --{_rs}--> ?")
        else:
            segs.append(f" --{_rs}--> {ctx.ents[b]}")
    out = "".join(segs)[:600]
    if _scores is not None:
        out += f"  (GTE-ranked best of {len(cands)})"
    return out + (f"  ({len(named)-1} hop(s); walk this bridge to "
                  "merge the subgraphs)")



