"""Native tool-calling agent orchestrator.

Self-contained: imports the robust engines (GTE, k_queue, compress_paths) and
the stage pipeline's CaseState only as a data carrier for result-dict shape
compatibility. Does NOT call run_stage_mode or any kgqa/stages/* stage function.

Flow per case:
  build CaseContext (ents/rels/edges from sample, expand CVTs, resolve anchor)
  → messages = [system=AGENTS.md, user="Question: ..."]
  → loop (≤ max_iters):
       msg = agent_call(session, messages, tools_for_state(state), ...)
       if msg.tool_calls: harness.validate → dispatch (if ok) or guidance (if bad)
       if answer accepted: parse + return
  → on failure: return best-effort with agent_failed=True (entry decides fallback)
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from kgqa.core.utils import normalize, candidate_hit, strict_candidate_hit, compute_match_stats
from kgqa.traversal.cvt import is_cvt_like, expand_cvt_leaves
from kgqa.llm.client import agent_call
from kgqa.agent import harness as H
from kgqa.agent import tools as T
from kgqa.agent.loader import agents_md


# ---------------------------------------------------------------------------
# Relation definitions for symmetric GTE retrieval
#
# GTE matches the query (relation_hint) against candidate relation texts.
# Originally candidate_texts were raw dot-notation IDs (rel_to_text is a no-op),
# which put the query (natural language) and candidates (schema IDs) in
# DIFFERENT semantic spaces — GTE matched on substring coincidences (e.g.
# "champion" in the hint matching "championships" in the ID), and shared schema
# prefixes (sports.sports_team.*) created false-positive similarity.
#
# Fix (validated): pre-generate a DEFINITIONAL sentence per relation (what the
# relation semantically connects, noun-based). When BOTH the hint and the
# candidate use definitional text, GTE rank-1 hits the correct relation with
# noise suppressed. The skill guides the model to write definitional hints; the
# candidate side uses these definitions here.
# ---------------------------------------------------------------------------
_REL_DEF_PATH = Path(__file__).resolve().parents[2] / "data" / "relation_richtext.json"
_REL_DEFS: Dict[str, str] = {}
try:
    _REL_DEFS = json.loads(_REL_DEF_PATH.read_text())
except (FileNotFoundError, json.JSONDecodeError):
    pass


def _rel_def(rel_id: str) -> str:
    """Definition text for a relation, falling back to the raw id."""
    return _REL_DEFS.get(rel_id, rel_id)



# ---------------------------------------------------------------------------
# Per-case context (built once from a pipeline sample dict)
# ---------------------------------------------------------------------------

@dataclass
class CaseContext:
    """Holds the per-case KG data + agent-accumulated paths/candidates."""
    case_id: str = ""
    case_num: int = 0
    question: str = ""
    gt_answers: List[str] = field(default_factory=list)
    sample: Dict[str, Any] = field(default_factory=dict)
    pilot_row: Dict[str, Any] = field(default_factory=dict)

    # KG arrays (mirrors CaseState after stage0)
    ents: List[str] = field(default_factory=list)
    rels: List[str] = field(default_factory=list)
    rel_texts: List[str] = field(default_factory=list)
    h_ids: List[int] = field(default_factory=list)
    r_ids: List[int] = field(default_factory=list)
    t_ids: List[int] = field(default_factory=list)
    ent_candidates: List[str] = field(default_factory=list)

    # Anchor
    anchor_idx: Optional[int] = None
    anchor_name: Optional[str] = None
    breakpoints: Dict[int, int] = field(default_factory=dict)

    # Agent-accumulated state
    fact_ids: List[str] = field(default_factory=list)       # ordered fact ids (from decompose)
    fact_texts: Dict[str, str] = field(default_factory=dict)
    fact_satisfies: Dict[str, str] = field(default_factory=dict)  # id -> constraint text (if fact materializes a condition)
    fact_start_types: Dict[str, str] = field(default_factory=dict)  # id -> start_type (anchor name for f1, type noun for f2+)
    fact_start_entities: Dict[str, str] = field(default_factory=dict)  # id -> start_entity (multi-anchor chain roots only)
    fact_steps: List[List[str]] = field(default_factory=list)  # ordered step groups from question_chains: each = [fid] (sequential) or [fid,...] (a `con` conjunctive layer)
    chains: List[dict] = field(default_factory=list)  # parsed question_chains: [{anchor (name|""), fact_steps ([groups])}, ...]; one per independent anchor (multi-anchor). Single-chain = 1 entry.
    fact_relations: Dict[str, set] = field(default_factory=dict)  # fact_id -> GTE relation idx set
    fact_relation_candidates: Dict[str, list] = field(default_factory=dict)  # fact_id -> pruned candidate rel idx (model picks from these)
    fact_entities: Dict[str, List[int]] = field(default_factory=dict)  # SEQ pipeline: fact_id -> resolved entity idx frontier (fact_i's hits feed fact_{i+1} grounding). Empty for SAPS.
    subgraph_entities: set = field(default_factory=set)  # SEQ pipeline: accumulated entity idx seen across all retrieve_subgraph trees (+ anchor). Center-boundary check. Empty for SAPS.
    var_bindings: Dict[str, List[str]] = field(default_factory=dict)  # SEQ pipeline: ?variable -> bound entity NAMES, populated from the model's checkpoint declarations ([fid ✓] ?var = [v1,...]). Tools expand "?var" in `entities` via this map. Empty for SAPS.
    accumulated_triples: set = field(default_factory=set)  # SEQ pipeline: set of (h_name, r_name, t_name) seen across prior retrieve_subgraph calls. Used by display cycle-suppression: a chain whose edge repeats/inverses an accumulated edge (subgraph N looping back onto subgraph 1) is dropped from the tree. Empty for SAPS.
    fact_paths: Dict[str, list] = field(default_factory=dict)
    all_paths: list = field(default_factory=list)
    all_candidates: List[str] = field(default_factory=list)
    logical_paths: list = field(default_factory=list)
    selected_candidates: List[str] = field(default_factory=list)
    branches: Dict[str, dict] = field(default_factory=dict)  # branch_id -> {candidates, triples, readable}

    # Final answer
    llm_answer_preds: List[str] = field(default_factory=list)
    llm_answer_str: str = ""

    # Diagnostics
    trajectory: list = field(default_factory=list)   # [{role, content/tool_calls}, ...]
    stage_times: Dict[str, float] = field(default_factory=dict)


def build_context(sample: Dict[str, Any], pilot_row: Dict[str, Any], idx: int) -> CaseContext:
    """Build a CaseContext from a pipeline sample dict (same shape run_pipeline uses).

    Replicates stage0's subgraph extraction: ents from text/non-text entity lists,
    rels from relation_list, edges from h/r/t_id_list, CVT-leaf expansion, and
    rel_texts/ent_candidates. Anchor resolution reuses the q_entity→entity-list
    heuristic from stage1_cascade (_resolve_anchor / _fallback_anchor logic).
    """
    gt = pilot_row.get("gt", pilot_row.get("ground_truth", pilot_row.get("gt_answers", [])))
    ctx = CaseContext(
        case_id=pilot_row.get("case_id", sample.get("id", "")),
        case_num=idx,
        question=pilot_row.get("question", sample.get("question", "")),
        gt_answers=gt if isinstance(gt, list) else ([gt] if gt else []),
        sample=sample,
        pilot_row=pilot_row,
    )

    # Subgraph (identical to stage0_ner.resolve)
    ents = list(sample.get("text_entity_list", [])) + list(sample.get("non_text_entity_list", []))
    rels = list(sample.get("relation_list", []))
    h_ids = list(sample.get("h_id_list", []))
    r_ids = list(sample.get("r_id_list", []))
    t_ids = list(sample.get("t_id_list", []))
    ents, rels, h_ids, r_ids, t_ids = expand_cvt_leaves(ents, rels, h_ids, r_ids, t_ids)

    ctx.ents = ents
    ctx.rels = rels
    ctx.h_ids = h_ids
    ctx.r_ids = r_ids
    ctx.t_ids = t_ids
    # GTE candidate texts: use RAW schema ids (rel_to_text is a no-op), NOT the
    # _rel_def richtext. Batch verification (n=111 facts, WebQSP 100) showed
    # raw-id beats richtext on GT-relation recall at every top-K (top3 84.7% vs
    # 76.6%, top5 89.2% vs 84.7%). The richtext definitions inject semantic noise
    # (high-frequency words like state/jurisdiction/location surface unrelated
    # relations), pushing the correct relation down. The earlier "_rel_def
    # validated to improve GTE" claim in the comment above _rel_def did not hold
    # under the agent's natural-language hints.
    from kgqa.core.utils import rel_to_text as _rel_to_text
    ctx.rel_texts = [_rel_to_text(r) for r in rels]
    ctx.ent_candidates = [e for e in ents if e and len(e) > 1 and not is_cvt_like(e)]

    _resolve_anchor(ctx)
    return ctx


def _resolve_anchor(ctx: CaseContext):
    """Resolve the anchor from the model's LLM analysis (preferred) or fallback
    to q_entity heuristic.

    The model picks ``anchor`` and ``endpoints`` during decompose (stored on
    AgentState). This function resolves the model's choice to graph idx:
      - anchor name → ctx.anchor_idx / ctx.anchor_name
      - endpoint names → ctx.breakpoints (entity_name → idx)

    If the model didn't pick an anchor (empty/None), fall back to the q_entity
    heuristic: first q_entity that matches an entity in ctx.ents (exact, then
    substring len>=3). This mirrors stage1_cascade but is ONLY a fallback — the
    LLM's ambiguity analysis is authoritative.
    """
    # 1. Try model-chosen anchor first (LLM ambiguity analysis)
    model_anchor = getattr(ctx, '_model_anchor', None) or ""
    if model_anchor:
        mn = normalize(model_anchor)
        if mn:
            for i, e in enumerate(ctx.ents):
                if normalize(e) == mn:
                    ctx.anchor_idx = i
                    ctx.anchor_name = ctx.ents[i]
                    break
            if ctx.anchor_idx is None:
                for i, e in enumerate(ctx.ents):
                    en = normalize(e)
                    if len(mn) >= 3 and (mn in en or en in mn):
                        ctx.anchor_idx = i
                        ctx.anchor_name = ctx.ents[i]
                        break
            if ctx.anchor_idx is not None:
                # Resolve endpoints (model-chosen constraint entities)
                model_eps = getattr(ctx, '_model_endpoints', None) or []
                for ep_name in model_eps:
                    epn = normalize(str(ep_name))
                    if not epn:
                        continue
                    for i, e in enumerate(ctx.ents):
                        if normalize(e) == epn and i != ctx.anchor_idx:
                            ctx.breakpoints[ep_name] = i
                            break
                    else:
                        # substring fallback for endpoints
                        for i, e in enumerate(ctx.ents):
                            en = normalize(e)
                            if len(epn) >= 3 and (epn in en or en in epn) and i != ctx.anchor_idx:
                                ctx.breakpoints[ep_name] = i
                                break
                return

    # 2. Fallback: q_entity heuristic (only if model didn't pick an anchor)
    q_entities = ctx.sample.get("q_entity", []) or []
    if q_entities:
        for qe in q_entities:
            qn = normalize(qe)
            if not qn:
                continue
            for i, e in enumerate(ctx.ents):
                if normalize(e) == qn:
                    ctx.anchor_idx = i
                    ctx.anchor_name = ctx.ents[i]
                    return
            for i, e in enumerate(ctx.ents):
                en = normalize(e)
                if len(qn) >= 3 and (qn in en or en in qn):
                    ctx.anchor_idx = i
                    ctx.anchor_name = ctx.ents[i]
                    return
        # fallback: first q_entity even if no match (anchor_idx stays None)
        ctx.anchor_name = q_entities[0]
        ctx.anchor_idx = None
        return
    if ctx.ents:
        ctx.anchor_idx = 0
        ctx.anchor_name = ctx.ents[0]


# ---------------------------------------------------------------------------
# Single-case agent loop
# ---------------------------------------------------------------------------

async def run_agent_case(session: aiohttp.ClientSession, sample: Dict[str, Any],
                         pilot_row: Dict[str, Any], idx: int,
                         args) -> Dict[str, Any]:
    """Run the tool-calling agent on one case. Returns a result dict whose shape
    matches kgqa.stages.stage8_reason._case_state_to_result_dict so the same
    scoring/summary code path works for both modes."""
    t0 = time.perf_counter()
    ctx = build_context(sample, pilot_row, idx)
    state = H.AgentState()

    system_prompt = agents_md()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {ctx.question}"},
    ]

    max_iters = int(getattr(args, "agent_max_iters", 16))
    agent_failed = False
    failure_reason = ""

    for _ in range(max_iters):
        offered = T.tools_for_state(state.state)
        if not offered:
            break  # DONE
        try:
            msg = await agent_call(
                session, messages, offered,
                tool_choice="required" if state.state != "ANSWER" else "auto",
                max_tokens=int(getattr(args, "agent_max_tokens", 1024)),
                temperature=float(getattr(args, "agent_temperature", 0.3)),
            )
        except Exception as e:
            agent_failed = True
            failure_reason = f"agent_call error: {e}"
            break

        # Append the assistant message verbatim (preserves tool_calls for the API)
        assistant_msg = {"role": "assistant",
                         "content": msg.get("content") or "",
                         "tool_calls": msg.get("tool_calls") or None}
        # Strip None tool_calls to match OpenAI shape expectations
        if not assistant_msg["tool_calls"]:
            assistant_msg.pop("tool_calls")
        messages.append(assistant_msg)
        ctx.trajectory.append({"role": "assistant",
                               "content": assistant_msg["content"],
                               "tool_calls": assistant_msg.get("tool_calls")})

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # Model emitted plain content with no tool call mid-flow → nudge.
            if state.state != "ANSWER":
                messages.append({"role": "tool", "tool_call_id": "noop",
                                 "content": f"No tool call received. Next expected: "
                                            f"{H._allowed_hint(state)}. Call it now."})
                ctx.trajectory.append({"role": "tool", "content": "(nudge)"})
                continue
            # In ANSWER state, plain content means model gave up — finish.
            agent_failed = True
            failure_reason = "no answer tool_call in ANSWER state"
            break

        ok, err, state = H.validate(state, tool_calls)
        if not ok:
            # Relay the guidance as a tool-role message so the model self-corrects.
            messages.append({"role": "tool", "tool_call_id": tool_calls[0].get("id", "rej"),
                             "content": f"REJECTED: {err}"})
            ctx.trajectory.append({"role": "tool", "content": f"REJECTED: {err}"})
            continue

        # Accepted — execute via dispatch and feed the result back.
        call = tool_calls[0]
        fn = call.get("function") or {}
        tool_name = fn.get("name", "")
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception:
                parsed_args = {}
        else:
            parsed_args = raw_args or {}

        # Sync the ordered fact ids (from the harness state) so _do_select can
        # build multi-step step_relations in decompose order.
        ctx.fact_ids = list(getattr(state, "fact_ids", []) or [])
        ctx.fact_texts = dict(getattr(state, "fact_texts", {}) or {})
        ctx.fact_satisfies = dict(getattr(state, "fact_satisfies", {}) or {})
        try:
            result_str = await T.dispatch(tool_name, parsed_args, ctx, session)
        except Exception as e:
            result_str = json.dumps({"error": f"dispatch_{tool_name}: {e}"})

        messages.append({"role": "tool", "tool_call_id": call.get("id", tool_name),
                         "name": tool_name, "content": result_str})
        ctx.trajectory.append({"role": "tool", "name": tool_name, "content": result_str})

        if state.state == H.DONE:
            break
    else:
        agent_failed = True
        failure_reason = f"max_iters ({max_iters}) reached at state {state.state}"

    dt = time.perf_counter() - t0
    ctx.stage_times["agent_total"] = dt

    return _ctx_to_result_dict(ctx, state, agent_failed, failure_reason)


# ---------------------------------------------------------------------------
# Result-dict shape (matches _case_state_to_result_dict so runner scoring works)
# ---------------------------------------------------------------------------

def _step_dict(fid: str, ctx: CaseContext) -> Dict[str, Any]:
    """One entry of `steps_parsed`. Includes `satisfies` when the fact
    materializes a question constraint (so decompose-level condition info
    survives into the result dict for复盘 / trajectory training)."""
    step = {"id": fid, "text": ctx.fact_texts.get(fid, "")}
    sat = ctx.fact_satisfies.get(fid)
    if sat:
        step["satisfies"] = sat
    return step


from kgqa.agent.evidence import EvidenceLog


def _ctx_to_result_dict(ctx: CaseContext, state, agent_failed: bool,
                        failure_reason: str) -> Dict[str, Any]:
    preds = ctx.llm_answer_preds
    ans_str = ctx.llm_answer_str
    gt = ctx.gt_answers

    llm_hit = candidate_hit(preds, gt) if preds else False

    # gt_hit should reflect what the model could ACTUALLY see at answer time —
    # i.e. the expand_branch candidates (stage8), not just the stage5 pool.
    # The stage5 pool (ctx.all_candidates) only has last-step BFS candidates
    # WITHOUT sibling-CVT expansion, so it under-counts. We union with all
    # expand_branch candidate lists from the trajectory.
    expand_cands = list(ctx.all_candidates or [])
    for step in ctx.trajectory:
        # Accept both plural (current) and singular (legacy trajectory files)
        # names so old runs can still be replayed/scored.
        if step.get("role") == "tool" and step.get("name") in ("expand_branch", "expand_branches"):
            try:
                payload = json.loads(step.get("content", "{}"))
                expand_cands.extend(payload.get("candidates", []))
            except Exception:
                pass

    gt_hit = candidate_hit(expand_cands, gt) if expand_cands else False
    gt_strict = strict_candidate_hit(expand_cands, gt) if expand_cands else False
    llm_stats = compute_match_stats(preds, gt)
    gt_stats = compute_match_stats(ctx.all_candidates, gt)

    return {
        "case_id": ctx.case_id,
        "case_num": ctx.case_num,
        "question": ctx.question,
        "gt_answers": gt,
        "decomposition_method": "agent_toolcall",
        "anchor_idx": ctx.anchor_idx,
        "anchor_name": ctx.anchor_name,
        "breakpoints": {k: ctx.ents[v] for k, v in ctx.breakpoints.items()
                        if v is not None and 0 <= v < len(ctx.ents)},
        "steps_parsed": [_step_dict(fid, ctx) for fid in state.fact_ids],
        "n_paths": len(ctx.all_paths),
        "answer_candidates": ctx.all_candidates,
        # FULL-EXPANSION REFERENCE (recorded at walk time, free): the per-branch
        # CVT-expanded candidate + triple sets for EVERY branch select_relations
        # produced (not just the ones the model expanded). This is the data the
        # path-level S_plan/S_select scorer needs (which branch hits the answer,
        # including branches the model MISSED) — recorded here so it never has to
        # be re-traversed offline. Caps keep the record compact.
        "branches_ref": {
            bid: {"candidates": list(br.get("candidates", []))[:50],
                  "triples": [list(t) for t in br.get("triples", [])[:80]],
                  "readable": br.get("readable", "")}
            for bid, br in (getattr(ctx, "branches", {}) or {}).items()
        },
        "gt_hit": gt_hit,
        "gt_hit_strict": gt_strict,
        "gt_f1": gt_stats.get("f1", 0.0),
        "num_patterns": len(ctx.logical_paths),
        "logical_paths": [lp.get("readable", "") for lp in ctx.logical_paths[:10]],
        "llm_answer": ans_str,
        "llm_hit": llm_hit,
        "llm_f1": llm_stats.get("f1", 0.0),
        "llm_precision": llm_stats.get("precision", 0.0),
        "llm_recall": llm_stats.get("recall", 0.0),
        "agent_failed": agent_failed,
        "agent_failure_reason": failure_reason,
        "agent_state": state.state,
        "agent_trajectory": ctx.trajectory,
        # EVIDENCE LOG SERIALIZATION (redesign ZEROETH RULE, 2026-09-24):
        # per-call structured evidence for offline consumers (RSCC
        # attribution reads THIS, never re-parsing rendered text).
        "evidence_log": [
            {"root": ev.root, "patterns": ev.patterns,
             "edges": ev.edges,
             "records": {k: v for k, v in ev.records.items()},
             "endpoints": sorted(ev.endpoints)}
            for ev in (getattr(ctx, "evidence_log", None) or EvidenceLog()).calls
        ] if getattr(ctx, "evidence_log", None) else [],
        "stage_times": ctx.stage_times,
        "error": failure_reason if agent_failed else None,
    }


# ---------------------------------------------------------------------------
# Batch mode (async gather with limited concurrency)
# ---------------------------------------------------------------------------

async def run_agent_mode(cases_to_run, args):
    """Run the agent over all (sample, pilot_row, idx) tuples.

    Mirrors run_stage_mode's I/O contract: writes results.json + prints a summary
    with the same GT/LLM hit + F1 surface so the two modes are directly comparable.
    """
    total = len(cases_to_run)
    print(f"\n=== Agent tool-call mode: {total} cases ===")
    parallel = max(1, min(int(getattr(args, "parallel", 1)), 32))
    sem = asyncio.Semaphore(parallel)

    async def _one(session, sample, pilot_row, idx):
        async with sem:
            try:
                return await run_agent_case(session, sample, pilot_row, idx, args)
            except Exception as e:
                return {
                    "case_id": pilot_row.get("case_id", ""),
                    "case_num": idx,
                    "question": pilot_row.get("question", ""),
                    "gt_answers": pilot_row.get("gt_answers",
                                                pilot_row.get("gt", [])),
                    "agent_failed": True,
                    "agent_failure_reason": f"unhandled: {e}",
                    "llm_answer": "",
                    "llm_hit": False,
                    "gt_hit": False,
                    "gt_hit_strict": False,
                    "gt_f1": 0.0,
                    "llm_f1": 0.0,
                    "llm_precision": 0.0,
                    "llm_recall": 0.0,
                    "answer_candidates": [],
                    "n_paths": 0,
                    "steps_parsed": [],
                    "stage_times": {},
                    "error": f"unhandled: {e}",
                }

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[
            _one(session, sample, pilot_row, idx)
            for sample, pilot_row, idx in cases_to_run
        ])
    wall_time = time.perf_counter() - wall_start

    _write_results(results, args, wall_time)
    return results


def _write_results(results, args, wall_time):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

    cases = len(results)
    if cases == 0:
        print("\n=== No valid cases processed ===")
        return

    gt_hits = sum(1 for r in results if r.get("gt_hit"))
    llm_hits = sum(1 for r in results if r.get("llm_hit"))
    failures = sum(1 for r in results if r.get("agent_failed"))
    errors = sum(1 for r in results if r.get("error"))

    gt_f1_avg = sum(r.get("gt_f1", 0) for r in results) / cases
    llm_f1_avg = sum(r.get("llm_f1", 0) for r in results) / cases
    llm_p_avg = sum(r.get("llm_precision", 0) for r in results) / cases
    llm_r_avg = sum(r.get("llm_recall", 0) for r in results) / cases
    strict_hits = sum(1 for r in results if r.get("gt_hit_strict"))

    for r in results:
        gt_mark = "✓" if r.get("gt_hit") else "✗"
        llm_mark = "✓" if r.get("llm_hit") else "✗"
        fail_mark = "!" if r.get("agent_failed") else " "
        print(f"  [{r.get('case_num', '?')}] GT{gt_mark} LLM{llm_mark}{fail_mark} "
              f"(F1={r.get('llm_f1', 0):.2f}) facts={len(r.get('steps_parsed', []))} "
              f"paths={r.get('n_paths', 0)} | {r.get('question', '')[:55]}")

    print(f"\n=== Summary: GT_recall={gt_hits}/{cases} ({100*gt_hits/cases:.1f}%) | "
          f"strict={strict_hits}/{cases} | GT_F1={gt_f1_avg:.3f} | "
          f"LLM_reason={llm_hits}/{cases} ({100*llm_hits/cases:.1f}%) | "
          f"LLM_P={llm_p_avg:.3f} R={llm_r_avg:.3f} F1={llm_f1_avg:.3f} | "
          f"agent_failures={failures} | errors={errors} ===")
    print(f"    Wall time: {wall_time:.2f}s")

    if getattr(args, "dump_trajectories", False):
        traj_dir = out / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            (traj_dir / f"case_{r.get('case_num', 0):03d}.txt").write_text(
                _format_trajectory(r), encoding="utf-8")
        print(f"    Trajectories: {traj_dir}")


def _format_trajectory(r: Dict[str, Any]) -> str:
    lines = [f"CASE [{r.get('case_num')}] {r.get('case_id')}",
             f"Q: {r.get('question')}",
             f"GT: {r.get('gt_answers')}",
             f"LLM: {r.get('llm_answer')} | hit={r.get('llm_hit')} "
             f"F1={r.get('llm_f1', 0):.2f} failed={r.get('agent_failed')}",
             f"final_state={r.get('agent_state')} "
             f"reason={r.get('agent_failure_reason', '')}",
             "-" * 70]
    for step in r.get("agent_trajectory", []):
        role = step.get("role", "?")
        if role == "assistant":
            tc = step.get("tool_calls") or []
            names = ", ".join((c.get("function") or {}).get("name", "?") for c in tc) if tc else "(no tool call)"
            lines.append(f"[ASSISTANT tools={names}]")
            if step.get("content"):
                lines.append(step["content"][:500])
        else:
            name = step.get("name", "")
            lines.append(f"[TOOL {name}] {step.get('content', '')[:400]}")
    return "\n".join(lines)
