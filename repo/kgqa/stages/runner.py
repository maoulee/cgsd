"""Stage-based batch execution runner for the KGQA pipeline.

Orchestrates all 8 stages, handles golden injection, GT anchor override,
result collection, and fail trace generation.
"""

import asyncio
import json
import time as _time
from pathlib import Path
from typing import List

import aiohttp

from kgqa.core.case_state import CaseState
from kgqa.core.config import (
    ROOT, REASON_STYLE, SKIP_NER, CANDIDATE_THRESHOLD,
)
from kgqa.core.utils import normalize

from kgqa.stages.stage0_ner import stage_0_ner_resolve
from kgqa.stages.stage1_decomp import stage_1_decomposition, stage_1_5_decomposition_reflect
from kgqa.stages.stage1_cascade import stage_1_cascade_decomposition
from kgqa.stages.stage2_entity import stage_2_entity_resolution
from kgqa.stages.stage2_gte_prune import stage_2_gte_and_prune
from kgqa.stages.stage3_gte import stage_3_gte_relation_retrieval
from kgqa.stages.stage4_prune import stage_4_relation_pruning, stage_4_rerank_pruning
from kgqa.stages.stage5_traverse import stage_5_graph_traversal
from kgqa.stages.stage6_diagnosis import stage_6_diagnosis_retry
from kgqa.stages.stage7_select import stage_7_path_selection
from kgqa.stages.stage8_reason import stage_8_answer_reasoning, _case_state_to_result_dict, serialize_casestate
from kgqa.traversal.path_utils import compress_paths
from kgqa.llm.batch import get_usage_stats, reset_usage_stats


async def run_stage_mode(cases_to_run, args):
    """Process all cases stage-by-stage with batch LLM calls."""
    reset_usage_stats()

    # Initialize CaseStates
    case_states = []
    for sample, pilot_row, idx in cases_to_run:
        gt = pilot_row.get("gt", pilot_row.get("ground_truth", pilot_row.get("gt_answers", [])))
        cs = CaseState(
            case_id=pilot_row["case_id"],
            case_num=idx,
            sample=sample,
            pilot_row=pilot_row,
            question=pilot_row["question"],
            gt_answers=gt if isinstance(gt, list) else [gt],
        )
        case_states.append(cs)

    total = len(case_states)
    print(f"\n=== Stage-batch mode: {total} cases ===")

    wall_start = _time.perf_counter()

    # GT anchor override: use q_entity from ROG-CWQ pkl as anchor
    gt_anchor_map = {}
    if getattr(args, 'gt_anchor', False):
        import pickle
        pkl_dir = ROOT / "data/cwq_processed"
        for split in ['test', 'train', 'val']:
            pkl_path = pkl_dir / f"{split}.pkl"
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    pkl_data = pickle.load(f)
                for d in pkl_data:
                    qe_ids = d.get("q_entity_id_list", [])
                    if qe_ids:
                        gt_anchor_map[d["id"]] = {
                            "idx": qe_ids[0],
                            "name": d.get("q_entity", [""])[0],
                        }
        print(f"  GT anchor: loaded {len(gt_anchor_map)} cases with q_entity")

    async with aiohttp.ClientSession() as session:
        skip_ner = getattr(args, 'skip_ner', False)
        await stage_0_ner_resolve(session, case_states, skip_ner=skip_ner)

        # ── Inject golden Stage 1a/1/1.5 outputs if requested ──
        inject_map = {}
        if getattr(args, 'inject_decomp', None):
            inject_map = {r["case_id"]: r for r in json.loads(Path(args.inject_decomp).read_text())}
            injected = 0
            for cs in case_states:
                g = inject_map.get(cs.case_id)
                if not g:
                    continue
                cs._1a_raw = g.get("stage_1a_raw")
                cs._1a_anchor = g.get("stage_1a_anchor")
                cs._1a_endpoints = g.get("stage_1a_endpoints")
                cs._1a_answer_type = g.get("stage_1a_answer_type")
                cs._1a_rewritten = g.get("stage_1a_rewritten")
                cs._1a_interpretation = g.get("stage_1a_interpretation")
                cs.answer_type = g.get("stage_1a_answer_type") or g.get("answer_type")
                if cs._1a_rewritten:
                    cs.rewritten_question = cs._1a_rewritten
                cs.decomp_raw = g.get("decomposition")
                cs.decomp_question = g.get("decomposition_question", "")
                cs.steps = g.get("steps_parsed", [])
                cs.anchor_name = g.get("anchor_name")
                cs.anchor_idx = None
                if cs.anchor_name and cs.ents:
                    for i, e in enumerate(cs.ents):
                        if e == cs.anchor_name:
                            cs.anchor_idx = i
                            break
                    if cs.anchor_idx is None:
                        an_lower = cs.anchor_name.lower()
                        for i, e in enumerate(cs.ents):
                            if an_lower in e.lower() or e.lower() in an_lower:
                                cs.anchor_idx = i
                                break
                bp_raw = g.get("breakpoints", {})
                cs.breakpoints = {}
                if bp_raw:
                    ent_to_idx = {e: i for i, e in enumerate(cs.ents)}
                    for k, v in bp_raw.items():
                        idx = ent_to_idx.get(v)
                        if idx is not None:
                            cs.breakpoints[int(k)] = idx
                cs.decomp_retry = g.get("decomp_retry", False)
                cs.decomp_reflect_raw = g.get("decomp_reflect_raw")
                cs.decomp_retry_reason = g.get("decomp_retry_reason")
                injected += 1
            print(f"  Injected golden decomp for {injected}/{total} cases (skipping Stage 1/1.5)")

        # Override anchor with GT q_entity after stage 0
        if gt_anchor_map:
            overridden = 0
            for cs in case_states:
                gt_info = gt_anchor_map.get(cs.case_id)
                if not gt_info:
                    continue
                idx = gt_info["idx"]
                if 0 <= idx < len(cs.ents):
                    cs.anchor_idx = idx
                    cs.anchor_name = cs.ents[idx]
                    overridden += 1
            print(f"  GT anchor: overridden {overridden}/{total} anchors")

        adaptive_routing = getattr(args, 'adaptive_routing', False)

        if not inject_map:
            decomp_mode = getattr(args, 'decomp', 'cascade')
            if decomp_mode == 'cascade':
                allow_1step = getattr(args, 'allow_1step', False)
                await stage_1_cascade_decomposition(
                    session, case_states, allow_1step=allow_1step,
                    adaptive_routing=adaptive_routing)
            else:
                await stage_1_decomposition(session, case_states)
                # Stage 1.5 retry: uses V1 chain prompt to retry 1-step decompositions.
                await stage_1_5_decomposition_reflect(session, case_states)

        if decomp_mode in ('cascade', 'triple'):
            # Cascade/triple pipeline: combined GTE + prune per triple
            await stage_2_gte_and_prune(
                session, case_states, adaptive_routing=adaptive_routing)
        else:
            # Old pipeline: separate entity resolution, GTE, and prune
            await stage_2_entity_resolution(session, case_states)

            # Re-apply GT anchor after stage 2
            if gt_anchor_map:
                for cs in case_states:
                    gt_info = gt_anchor_map.get(cs.case_id)
                    if gt_info:
                        idx = gt_info["idx"]
                        if 0 <= idx < len(cs.ents):
                            cs.anchor_idx = idx
                            cs.anchor_name = cs.ents[idx]

            await stage_3_gte_relation_retrieval(session, case_states)
            if args.prune == "rerank":
                await stage_4_rerank_pruning(case_states, args.rerank_model, args.prune_top_k)
            else:
                await stage_4_relation_pruning(session, case_states)
        await stage_5_graph_traversal(case_states)

        # ── Adaptive routing: SIMPLE→COMPLEX fallback promotion ──
        # Any SIMPLE case that failed Stage5 (no anchor / no paths / needs direct
        # answer) is promoted to COMPLEX and runs the remaining complex stages.
        # Safe-by-construction: worst case = today's behavior.
        if adaptive_routing:
            promoted = 0
            for cs in case_states:
                if (cs.active and getattr(cs, 'complexity', 'complex') == 'simple'
                        and (cs.anchor_idx is None or not cs.paths
                             or getattr(cs, 'needs_direct_answer', False))):
                    cs.complexity = "complex"
                    promoted += 1
            if promoted:
                print(f"    Adaptive fallback: {promoted} SIMPLE cases promoted to COMPLEX")

        # Compute logical_paths early for pattern-based explosion detection
        for cs in case_states:
            if cs.active and cs.paths:
                bp_indices = set(cs.breakpoints.values()) if cs.breakpoints else set()
                bp_indices.discard(cs.anchor_idx)
                cs.logical_paths = compress_paths(
                    cs.paths, cs.ents, cs.rels, cs.anchor_idx, bp_indices)

        if adaptive_routing:
            # Group-based routing: SIMPLE cases skip Stage 6 (diagnosis) and
            # Stage 7 (path-select). A SIMPLE 1-hop case has a single logical
            # pattern, so set cs.selected_paths = [0] (or all indices if no
            # logical_paths yet) to give Stage8 evidence. Stage8 already has a
            # re-select fallback for empty selected_paths.
            simple_group = [cs for cs in case_states
                            if cs.active and getattr(cs, 'complexity', 'complex') == 'simple']
            complex_group = [cs for cs in case_states
                             if cs.active and getattr(cs, 'complexity', 'complex') == 'complex']

            # For SIMPLE cases that still lack selected_paths, set a safe default
            for cs in simple_group:
                if not cs.selected_paths:
                    if cs.logical_paths:
                        cs.selected_paths = [0]
                    elif cs.paths:
                        cs.selected_paths = [0]
                    else:
                        cs.selected_paths = []

            if complex_group:
                await stage_6_diagnosis_retry(session, complex_group)
                await stage_7_path_selection(session, complex_group)
            # SIMPLE group skips Stage 6 + Stage 7
            await stage_8_answer_reasoning(session, case_states)
        else:
            await stage_6_diagnosis_retry(session, case_states)
            await stage_7_path_selection(session, case_states)
            await stage_8_answer_reasoning(session, case_states)

    wall_time = _time.perf_counter() - wall_start

    # Collect results
    rows = [_case_state_to_result_dict(cs) for cs in case_states]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    llm_usage = get_usage_stats()
    (out / "llm_usage.json").write_text(json.dumps(llm_usage, ensure_ascii=False, indent=2))

    # Dump full trajectories (JSONL) for trace_viewer.py
    traj_path = out / "trajectory_dump.jsonl"
    with open(traj_path, 'w', encoding='utf-8') as tf:
        for cs in case_states:
            tf.write(json.dumps(serialize_casestate(cs), ensure_ascii=False) + '\n')
    print(f"  Trajectory dump: {traj_path} ({len(case_states)} cases)")

    cases = len(rows)
    if cases == 0:
        print("\n=== No valid cases processed ===")
        return

    hits = sum(1 for r in rows if r["gt_hit"])
    llm_hits = sum(1 for r in rows if r.get("llm_hit"))
    errors = sum(1 for r in rows if r.get("error"))

    # Timing
    all_times = []
    for cs in case_states:
        total_stage = sum(cs.stage_times.values())
        all_times.append(total_stage)
    avg_case = sum(all_times) / len(all_times) if all_times else 0
    total_case = sum(all_times)

    # Per-case detail
    for i, (r, cs) in enumerate(zip(rows, case_states)):
        gt_mark = "✓" if r["gt_hit"] else "✗"
        gt_strict_mark = "S" if r.get("gt_hit_strict") else " "
        llm_mark = "✓" if r.get("llm_hit") else "✗"
        gt_f1_val = r.get('gt_f1', 0)
        llm_f1_val = r.get('llm_f1', 0)
        cov = getattr(cs, 'max_cov', 0)
        nsteps = len(getattr(cs, 'steps', []))
        print(f"  [{i}] GT{gt_mark}{gt_strict_mark}(F1={gt_f1_val:.2f}) LLM{llm_mark}(F1={llm_f1_val:.2f}) cov={cov}/{nsteps} paths={len(getattr(cs,'paths',[]))} | {r['question'][:60]}")

    strict_hits = sum(1 for r in rows if r.get("gt_hit_strict"))
    gt_f1_avg = sum(r.get('gt_f1', 0) for r in rows) / cases if cases else 0
    llm_f1_avg = sum(r.get('llm_f1', 0) for r in rows) / cases if cases else 0
    llm_p_avg = sum(r.get('llm_precision', 0) for r in rows) / cases if cases else 0
    llm_r_avg = sum(r.get('llm_recall', 0) for r in rows) / cases if cases else 0

    # Hit@1: first prediction matches GT
    from kgqa.core.utils import normalize as _norm
    def _hit_at_1(row):
        ans = row.get('llm_answer', '')
        if not ans:
            return False
        preds = [p.strip() for p in ans.split(' | ') if p.strip()]
        if not preds:
            return False
        gt = row.get('gt_answers', [])
        if not gt:
            return False
        p1_norm = _norm(preds[0])
        return any(p1_norm in _norm(g) or _norm(g) in p1_norm for g in gt)
    llm_hit1 = sum(1 for r in rows if _hit_at_1(r))

    print(f"\n=== Summary: GT_recall={hits}/{cases} ({100*hits/cases:.1f}%) | strict={strict_hits}/{cases} ({100*strict_hits/cases:.1f}%) | GT_F1={gt_f1_avg:.3f} | LLM_reason={llm_hits}/{cases} ({100*llm_hits/cases:.1f}%) | Hit@1={llm_hit1}/{cases} ({100*llm_hit1/cases:.1f}%) | LLM_P={llm_p_avg:.3f} R={llm_r_avg:.3f} F1={llm_f1_avg:.3f} | Errors={errors} ===")
    print(f"    Timing: Wall={wall_time:.2f}s | Sum stages={total_case:.2f}s | Avg case={avg_case:.2f}s")
    if llm_usage:
        logical = int(llm_usage.get("logical_prompts", 0))
        batch_requests = int(llm_usage.get("batch_requests", 0))
        single_requests = int(llm_usage.get("single_requests", 0))
        cache_hits = int(llm_usage.get("cache_hits", 0))
        print(f"    LLM transport: prompts={logical} | batch_requests={batch_requests} | single_requests={single_requests} | cache_hits={cache_hits}")
    speedup = total_case / wall_time if wall_time > 0 else 1.0
    print(f"    Batch speedup: {speedup:.2f}x")

    # Fail traces
    fail_traces = _build_fail_traces(rows, case_states)
    if fail_traces:
        trace_path = out / "fail_traces.txt"
        trace_path.write_text('\n\n'.join(fail_traces), encoding='utf-8')
        print(f"  Fail traces written to: {trace_path} ({len(fail_traces)} cases)")

    # Full trajectory dumps for manual auditing
    if getattr(args, 'dump_trajectories', False):
        traj_dir = out / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        for i, (r, cs) in enumerate(zip(rows, case_states)):
            _dump_case_trajectory(traj_dir, i, r, cs)
        print(f"  Trajectories written to: {traj_dir} ({len(rows)} cases)")


def _dump_case_trajectory(traj_dir, idx, r, cs):
    """Write a per-case trajectory file showing every stage's complete I/O."""
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"CASE [{idx}] — {r.get('case_id','')}")
    lines.append(f"{'='*80}")
    gt_mark = "HIT" if r.get("gt_hit") else "MISS"
    llm_mark = "HIT" if r.get("llm_hit") else "MISS"
    lines.append(f"Q: {r.get('question','')}")
    lines.append(f"GT: {r.get('gt_answers',[])}")
    lines.append(f"LLM Answer: {r.get('llm_answer','')}")
    lines.append(f"GT={gt_mark}(F1={r.get('gt_f1',0):.2f}) | LLM={llm_mark}(P={r.get('llm_precision',0):.2f} R={r.get('llm_recall',0):.2f} F1={r.get('llm_f1',0):.2f})")
    if r.get("error"):
        lines.append(f"ERROR: {r['error']}")

    # Stage 0: NER
    lines.append(f"\n{'─'*80}")
    lines.append(f"STAGE 0: NER Entity Resolution")
    lines.append(f"{'─'*80}")
    ents = getattr(cs, 'ents', [])
    lines.append(f"Entities ({len(ents)}): {ents[:20]}{'...' if len(ents) > 20 else ''}")
    ner_top = getattr(cs, 'ner_top_ents', [])
    if ner_top:
        lines.append(f"NER top entities: {[(e, f'{s:.3f}') for e, s in ner_top]}")
    lines.append(f"use_ner={getattr(cs, 'use_ner', True)}")

    # Stage 1a: Entity Analysis
    lines.append(f"\n{'─'*80}")
    lines.append(f"STAGE 1a: Entity Analysis (LLM Input → Output)")
    lines.append(f"{'─'*80}")
    prompt_1a = r.get("stage_1a_prompt", "")
    if prompt_1a:
        lines.append(f"--- PROMPT (user) ---")
        lines.append(prompt_1a[:3000])
    raw_1a = r.get("stage_1a_raw", "")
    if raw_1a:
        lines.append(f"\n--- LLM OUTPUT ---")
        lines.append(raw_1a[:3000])
    lines.append(f"\nParsed: anchor={r.get('stage_1a_anchor','')} | endpoints={r.get('stage_1a_endpoints','')} | answer_type={r.get('stage_1a_answer_type','')}")
    lines.append(f"Rewritten: {r.get('stage_1a_rewritten','')}")
    lines.append(f"Interpretation: {r.get('stage_1a_interpretation','')}")

    # Stage 1b: Chain Decomposition
    lines.append(f"\n{'─'*80}")
    lines.append(f"STAGE 1b: Chain Decomposition (LLM Input → Output)")
    lines.append(f"{'─'*80}")
    decomp_prompt = r.get("decomposition_prompt", "")
    if decomp_prompt:
        lines.append(f"--- PROMPT (decomposition template) ---")
        lines.append(str(decomp_prompt)[:500] if len(str(decomp_prompt)) > 500 else str(decomp_prompt))
    decomp_raw = r.get("decomposition", "")
    if decomp_raw:
        lines.append(f"\n--- LLM OUTPUT ---")
        lines.append(decomp_raw)
    lines.append(f"\nParsed anchor: {r.get('anchor_name','')} (idx={r.get('anchor_idx','')})")
    bp = r.get("breakpoints", {})
    if bp:
        lines.append(f"Breakpoints: {bp}")
    steps = r.get("steps_parsed", [])
    for si, s in enumerate(steps):
        lines.append(f"  Step {si}: {s}")

    # Stage 1.5: Reflection retry
    if r.get("decomp_retry"):
        lines.append(f"\n{'─'*80}")
        lines.append(f"STAGE 1.5: Decomposition Retry")
        lines.append(f"{'─'*80}")
        lines.append(f"Retry reason: {r.get('decomp_retry_reason','')}")
        if r.get("decomp_reflect_raw"):
            lines.append(f"Reflect raw: {r.get('decomp_reflect_raw','')[:2000]}")

    # Stage 3-4: GTE + Pruning
    lines.append(f"\n{'─'*80}")
    lines.append(f"STAGE 3-4: GTE Retrieval + Relation Pruning")
    lines.append(f"{'─'*80}")
    # GTE retrieval details (top-k per step)
    gte_details = r.get("relation_retrieval_details", [])
    rels_list = getattr(cs, 'rels', [])
    if gte_details:
        for entry in gte_details:
            step = entry.get("step", "?")
            lines.append(f"  Step {step} — GTE candidates ({entry.get('gte_candidates_count',0)}):")
            for q in entry.get("queries", []):
                for tk in q.get("top_k", [])[:10]:
                    lines.append(f"    #{tk.get('rank','?')} {tk.get('rel_text', tk.get('candidate',''))} (score={tk.get('score',0):.4f})")
    # Prune LLM reasoning
    prune_dbg = r.get("prune_debug", {})
    prune_resp = prune_dbg.get("response", "")
    if prune_resp:
        lines.append(f"\n  --- Prune LLM Reasoning ---")
        lines.append(prune_resp[:1500])
    prune_parsed = prune_dbg.get("parsed_result", {})
    if prune_parsed:
        lines.append(f"\n  Prune result: {prune_parsed}")
    # Final selected relations
    step_rels = r.get("step_relations", [])
    if step_rels:
        lines.append(f"\n  Final selected relations per step:")
        for si, rels in enumerate(step_rels):
            named = [f"{ri}:{rels_list[ri]}" if ri < len(rels_list) else str(ri) for ri in (rels if isinstance(rels, list) else list(rels))]
            lines.append(f"    Step {si}: {named}")

    # Stage 5: Graph Traversal
    lines.append(f"\n{'─'*80}")
    lines.append(f"STAGE 5: Graph Traversal")
    lines.append(f"{'─'*80}")
    paths = getattr(cs, 'paths', [])
    lines.append(f"Raw paths found: {len(paths)}")
    lines.append(f"Max depth: {getattr(cs, 'max_depth', 0)} | Coverage: {getattr(cs, 'max_cov', 0)}/{len(steps)}")
    lines.append(f"Answer candidates ({len(r.get('answer_candidates', []))}): {r.get('answer_candidates', [])[:20]}")
    lines.append(f"GT hit: {r.get('gt_hit')} | GT strict: {r.get('gt_hit_strict')} | GT F1: {r.get('gt_f1',0):.2f}")

    # Stage 7: Path Selection
    lines.append(f"\n{'─'*80}")
    lines.append(f"STAGE 7: Path Selection")
    lines.append(f"{'─'*80}")
    logical_paths = r.get("logical_paths", [])
    lines.append(f"Logical paths ({len(logical_paths)}):")
    for li, lp in enumerate(logical_paths[:10]):
        lines.append(f"  {li}. {lp}")
    pattern_details = r.get("pattern_details", [])
    if pattern_details:
        lines.append(f"Pattern details:")
        for pi, pd in enumerate(pattern_details[:5]):
            lines.append(f"  Pattern {pi}: cands={pd.get('candidates',[])[:8]} tier={pd.get('best_tier','')}")
    lines.append(f"Selected path indices: {r.get('selected_paths', [])}")

    # Stage 8: Answer Reasoning
    lines.append(f"\n{'─'*80}")
    lines.append(f"STAGE 8: Answer Reasoning (LLM Input → Output)")
    lines.append(f"{'─'*80}")
    reason_prompt = r.get("llm_reasoning_prompt", "")
    if reason_prompt:
        lines.append(f"--- FULL PROMPT (sent to LLM) ---")
        lines.append(reason_prompt)
    lines.append(f"\n--- LLM OUTPUT (raw response) ---")
    lines.append(r.get("llm_reasoning_full", "(empty)"))

    lines.append(f"\n{'='*80}")
    lines.append(f"FINAL: LLM={llm_mark} | Answer: {r.get('llm_answer','')}")
    lines.append(f"{'='*80}")

    (traj_dir / f"case_{idx:03d}.txt").write_text('\n'.join(str(l) if l is not None else "" for l in lines), encoding='utf-8')


def _build_fail_traces(rows, case_states):
    """Build detailed fail trace strings for cases where GT or LLM missed."""
    fail_traces = []
    for i, (r, cs) in enumerate(zip(rows, case_states)):
        if r.get("gt_hit") and r.get("llm_hit"):
            continue
        lines = [f"{'='*70}"]
        lines.append(f"Case [{i}]: {r.get('case_id','')}")
        lines.append(f"Q: {r.get('question','')}")
        lines.append(f"GT: {r.get('gt_answers',[])}")
        lines.append(f"LLM: {r.get('llm_answer','')}")
        lines.append(f"GT_hit={r.get('gt_hit',False)} GT_F1={r.get('gt_f1',0):.2f} LLM_hit={r.get('llm_hit',False)} LLM_P={r.get('llm_precision',0):.2f} R={r.get('llm_recall',0):.2f} F1={r.get('llm_f1',0):.2f}")
        if r.get("error"):
            lines.append(f"Error: {r['error']}")

        # Stage 1a
        stage_1a_raw = r.get("stage_1a_raw", "")
        if stage_1a_raw:
            lines.append(f"\n{'='*40} STAGE 1a: Entity Analysis {'='*40}")
            lines.append(f"--- 1a Output ---")
            lines.append(stage_1a_raw[:2000])
            lines.append(f"  Anchor: {r.get('stage_1a_anchor','')}")
            lines.append(f"  Endpoints: {r.get('stage_1a_endpoints','')}")
            lines.append(f"  Answer_type: {r.get('stage_1a_answer_type','')}")

        # Stage 1b
        lines.append(f"\n{'='*40} STAGE 1b: Chain Decomposition {'='*40}")
        decomp_raw = r.get("decomposition", "")
        if decomp_raw:
            lines.append(f"--- 1b Result ---")
            lines.append(decomp_raw)
        lines.append(f"Anchor: {r.get('anchor_name','')} (idx={r.get('anchor_idx','')})")
        bp = r.get("breakpoints", {})
        if bp:
            lines.append(f"Breakpoints: {bp}")
        steps = r.get("steps_parsed", [])
        for si, s in enumerate(steps):
            lines.append(f"  Step {si}: {s}")

        # Stage 3-4
        lines.append(f"\n{'='*40} STAGE 3-4: GTE + Pruning {'='*40}")
        # GTE retrieval details
        gte_details = r.get("relation_retrieval_details", [])
        rels_list = getattr(cs, 'rels', [])
        if gte_details:
            for entry in gte_details:
                step = entry.get("step", "?")
                lines.append(f"  Step {step} — GTE candidates ({entry.get('gte_candidates_count',0)}):")
                for q in entry.get("queries", []):
                    for tk in q.get("top_k", [])[:10]:
                        lines.append(f"    #{tk.get('rank','?')} {tk.get('rel_text', tk.get('candidate',''))} (score={tk.get('score',0):.4f})")
        # Prune LLM reasoning
        prune_dbg = r.get("prune_debug", {})
        prune_resp = prune_dbg.get("response", "")
        if prune_resp:
            lines.append(f"\n  --- Prune LLM Reasoning ---")
            lines.append(prune_resp[:1500])
        prune_parsed = prune_dbg.get("parsed_result", {})
        if prune_parsed:
            lines.append(f"\n  Prune result: {prune_parsed}")
        # Final selected
        step_rels = getattr(cs, 'step_relations', [])
        if step_rels:
            lines.append(f"\n  Final selected relations per step:")
            for si, rels in enumerate(step_rels):
                named = [f"{ri}:{rels_list[ri]}" if ri < len(rels_list) else str(ri) for ri in (rels if isinstance(rels, list) else list(rels))]
                lines.append(f"    Step {si}: {named}")

        # Stage 5
        lines.append(f"\n{'='*40} STAGE 5: Graph Traversal {'='*40}")
        paths = getattr(cs, 'paths', [])
        ents = getattr(cs, 'ents', [])
        if paths:
            lines.append(f"--- Raw Paths ({len(paths)}) ---")
            for pi, p in enumerate(paths):
                nodes = p.get("nodes", [])
                rels_p = p.get("relations", [])
                chain_parts = []
                for ni in range(len(nodes)):
                    chain_parts.append(ents[nodes[ni]] if 0 <= nodes[ni] < len(ents) else f"#{nodes[ni]}")
                    if ni < len(rels_p):
                        chain_parts.append(f"--[{rels_list[rels_p[ni]]}]-->" if rels_p[ni] < len(rels_list) else f"--[{rels_p[ni]}]-->")
                lines.append(f"  Path {pi}: {' '.join(chain_parts)}")

        # Stage 7
        lines.append(f"\n{'='*40} STAGE 7: Path Selection {'='*40}")
        logical_paths = getattr(cs, 'logical_paths', [])
        if logical_paths:
            lines.append(f"--- Logical Paths ({len(logical_paths)}) ---")
            for li, lp in enumerate(logical_paths):
                readable = lp.get("readable", "")
                cands = lp.get("candidates", [])
                lines.append(f"  LP {li}: {readable}")
                lines.append(f"         candidates({len(cands)}): {cands[:10]}{'...' if len(cands)>10 else ''}")
        sel_paths = getattr(cs, 'selected_paths', [])
        if sel_paths is not None:
            lines.append(f"--- Selected path indices: {sel_paths} ---")

        # Stage 8
        lines.append(f"\n{'='*40} STAGE 8: Answer Reasoning {'='*40}")
        reasoning = r.get("llm_reasoning_full", "")
        if reasoning:
            lines.append(f"--- Stage 8 LLM Response ---")
            lines.append(reasoning)

        # Candidates
        ans_cands = r.get("answer_candidates", [])
        if ans_cands:
            lines.append(f"\n--- Answer Candidates ({len(ans_cands)}) ---")
            for ci in range(0, len(ans_cands), 10):
                lines.append(f"  {ans_cands[ci:ci+10]}")

        fail_traces.append('\n'.join(lines))
    return fail_traces
