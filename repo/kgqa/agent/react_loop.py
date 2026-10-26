"""ReAct Batch mode — content-only LLM calls with batch optimization.

Instead of tool-call round-trips (agent_call per turn), this mode:
1. Collects ALL active cases' LLM prompts each round
2. Sends them in ONE batch_call_llm request
3. Parses each case's JSON output → (tool_name, args)
4. Dispatches tools (reuses tools.dispatch + harness.validate unchanged)
5. Appends results, updates state, repeats until all cases DONE

The LLM returns content (not tool_calls), in a JSON envelope:
  {"tool": "retrieve", "args": {"fact_id": "f1", "subquestion": "..."}}

This achieves ~17x throughput vs per-case round-trips, matching baseline's
batch-call pattern while keeping the agent's free orchestration (model decides
which tool to call next, not a fixed stage sequence).

All _do_* tool handlers and harness state machine are reused UNCHANGED.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from kgqa.agent.loop import (
    CaseContext, build_context, _ctx_to_result_dict, _step_dict,
)
from kgqa.agent.harness import AgentState, validate, _allowed_hint
from kgqa.agent import tools as T
from kgqa.agent.loader import agents_md
from kgqa.llm.batch import batch_call_llm, batch_call_llm_with_reasoning


# ---------------------------------------------------------------------------
# Per-case ReAct state (wraps CaseContext + messages + harness state)
# ---------------------------------------------------------------------------
class ReactCase:
    """One case's evolving state through the ReAct batch loop."""

    def __init__(self, sample, pilot_row, idx):
        self.ctx = build_context(sample, pilot_row, idx)
        self.state = AgentState()
        self.messages: List[Dict[str, Any]] = []
        self.failed = False
        self.failure_reason = ""
        self.done = False
        self._init_messages()

    def _init_messages(self):
        sys_prompt = agents_md()
        user = f"Question: {self.ctx.question}"
        # Provide the known entities list so the model can do anchor/endpoint
        # analysis during decompose (pick the lowest-ambiguity concrete entity
        # as anchor, skip generic type words, mark constraint entities as
        # endpoints). This mirrors stage's ENTITY_ANALYSIS_PROMPT.
        q_ents = self.ctx.sample.get("q_entity", []) or []
        self.messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]

    @property
    def is_active(self):
        return not self.done and not self.failed

    def allowed_tools_hint(self):
        """What tools are legal in the current state (for prompt nudge)."""
        return _allowed_hint(self.state)


# ---------------------------------------------------------------------------
# JSON envelope parsing
# ---------------------------------------------------------------------------
def parse_react_output(content: str) -> Tuple[Optional[str], Optional[dict]]:
    """Parse the LLM's content-only output into (tool_name, args).

    Prompt convention (think-then-act):
      <reasoning>
      tool: {"tool": "<name>", "args": {<args>}}

    The ``tool:`` anchor pins the real tool call, so any hypothetical JSON
    examples inside the reasoning are ignored. If no anchor is present we fall
    back to the first valid tool-JSON anywhere (backward compat with pure-JSON
    output). Anchored parsing takes the FIRST tool call — the harness enforces
    one tool per turn and the model is told to emit only one.
    """
    if not content:
        return None, None
    text = content.strip()

    def _try_parse_tool(seg: str) -> Optional[dict]:
        try:
            obj = json.loads(seg)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and "tool" in obj:
            return obj
        return None

    def _first_tool_json(s: str) -> Optional[dict]:
        """Return the first balanced {...} in s that parses to a dict with 'tool'.

        Robust to truncated deep JSON: when the model omits trailing ``}`` (a
        common failure on deeply nested ``select_relations`` payloads), we take
        the segment from the first ``{`` to the last ``}`` in s and append the
        missing closing braces to balance the bracket depth before parsing.
        """
        depth, start = 0, -1
        last_close = -1
        for i, c in enumerate(s):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    obj = _try_parse_tool(s[start:i + 1])
                    if obj is not None:
                        return obj
                    last_close = i
                    start = -1
                elif depth > 0:
                    last_close = i
        # Truncation recovery: a ``{`` was opened but never closed back to 0.
        # Re-parse from the first ``{`` to the last seen ``}``, padding the
        # missing closers so deeply nested but otherwise-valid payloads survive.
        if start >= 0 and last_close > start:
            seg = s[start:last_close + 1]
            seg += '}' * (seg.count('{') - seg.count('}'))
            obj = _try_parse_tool(seg)
            if obj is not None:
                return obj
        return None

    # 1) Anchor path: look for `tool:` then take the first tool JSON after it.
    for m in re.finditer(r'(?<![A-Za-z])tool:\s*', text, re.IGNORECASE):
        rest = text[m.end():]
        rest = re.sub(r'^\s*```(?:json)?\s*', '', rest)  # strip optional fence
        obj = _first_tool_json(rest)
        if obj is not None:
            return obj["tool"], obj.get("args") or {}

    # 2) Fallback: no anchor — backward-compat first tool JSON anywhere.
    obj = _first_tool_json(text)
    if obj is not None:
        return obj["tool"], obj.get("args") or {}

    return None, None


# ---------------------------------------------------------------------------
# Tool call to harness-compatible format
# ---------------------------------------------------------------------------
def _to_tool_calls(tool_name: str, args: dict) -> list:
    """Wrap parsed (tool_name, args) into the format harness.validate expects."""
    return [{
        "id": "react-call",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(args) if args else "{}",
        },
    }]


# ---------------------------------------------------------------------------
# Batch ReAct main loop
# ---------------------------------------------------------------------------
async def run_react_batch(cases_to_run, args, output_dir: str):
    """Run the ReAct batch agent over all cases.

    Each round:
      1. Collect active cases' current messages
      2. batch_call_llm (one HTTP request for all)
      3. Parse JSON → (tool, args) per case
      4. Dispatch tools (concurrent)
      5. Append results, update states
    """
    total = len(cases_to_run)
    print(f"\n=== ReAct batch mode: {total} cases ===", flush=True)

    wall_start = time.perf_counter()

    # Initialize all cases
    react_cases = []
    for sample, pilot_row, idx in cases_to_run:
        rc = ReactCase(sample, pilot_row, idx)
        react_cases.append(rc)

    max_rounds = int(getattr(args, "agent_max_iters", 16))
    max_tokens = int(getattr(args, "agent_max_tokens", 1024))
    # When a thinking-token budget is active, reasoning tokens come out of
    # max_tokens — raise the floor so reasoning can't starve the tool-JSON
    # output (which manifests as finish_reason=length with an empty answer).
    from kgqa.llm.client import THINKING_TOKEN_BUDGET as _tb
    if _tb > 0:
        max_tokens = max(max_tokens, _tb + 2560)
    batch_chunk = int(getattr(args, "batch_chunk", 25))

    async with aiohttp.ClientSession() as session:
        for round_num in range(max_rounds):
            # 1. Collect active cases
            active = [rc for rc in react_cases if rc.is_active]
            if not active:
                break

            # 2. Build prompts for batch (system prompt is shared, context differs)
            prompts = []
            for rc in active:
                hint = rc.allowed_tools_hint()
                msgs = list(rc.messages)
                msgs.append({"role": "user", "content": hint})
                prompts.append(msgs)

            print(f"  Round {round_num+1}: {len(active)} active cases, "
                  f"batching {len(prompts)} LLM calls...", flush=True)
            t_batch = time.perf_counter()

            # 3. Batch LLM call — split into chunks, dispatch in parallel.
            # The batch endpoint scales poorly beyond ~25 prompts (no cross-prompt
            # KV-cache), so we chunk and fire chunks concurrently.
            chunks = [prompts[i:i+batch_chunk]
                      for i in range(0, len(prompts), batch_chunk)]

            async def _batch_chunk(chunk):
                return await batch_call_llm_with_reasoning(session, chunk, max_tokens=max_tokens)

            try:
                chunk_results = await asyncio.gather(*[_batch_chunk(c) for c in chunks])
            except Exception as e:
                print(f"  Batch call failed: {e}", flush=True)
                for rc in active:
                    rc.failed = True
                    rc.failure_reason = f"batch_call error: {e}"
                break

            # Flatten chunked results back to per-case order (contents + reasoning)
            responses = []
            responses_reasoning = []
            for cr_contents, cr_reasoning in chunk_results:
                responses.extend(cr_contents)
                responses_reasoning.extend(cr_reasoning)

            dt_batch = time.perf_counter() - t_batch
            print(f"  Batch done in {dt_batch:.1}s ({len(active)/dt_batch:.1f} cases/s)",
                  flush=True)

            # 4. Parse + dispatch per case (concurrent)
            async def process_case(rc, raw_response, reasoning=None):
                """Parse LLM output, validate, dispatch tool, append result."""
                if not raw_response or not raw_response.strip():
                    # Empty response — nudge
                    rc.messages.append({"role": "assistant", "content": ""})
                    rc.messages.append({"role": "user",
                                        "content": rc.allowed_tools_hint()})
                    return

                # Append assistant message (content-only, no tool_calls). Attach
                # the vLLM <think> reasoning so batch trajectories carry the
                # model's CoT (parity with run_react_case's per-call path).
                asst_msg = {"role": "assistant", "content": raw_response}
                if reasoning:
                    asst_msg["reasoning"] = reasoning
                rc.messages.append(asst_msg)
                traj_step = {"role": "assistant", "content": raw_response}
                if reasoning:
                    traj_step["reasoning"] = reasoning
                rc.ctx.trajectory.append(traj_step)

                # Parse JSON envelope
                tool_name, parsed_args = parse_react_output(raw_response)
                if not tool_name:
                    # Failed to parse — nudge
                    rc.messages.append({"role": "user",
                                        "content": f"Could not parse tool call. "
                                        f"Output JSON like: {{\"tool\": \"...\", \"args\": {{...}}}}. "
                                        f"{rc.allowed_tools_hint()}"})
                    return

                parsed_args = parsed_args or {}

                # Validate via harness (reuses existing state machine)
                tool_calls = _to_tool_calls(tool_name, parsed_args)
                ok, err, new_state = validate(rc.state, tool_calls)
                if not ok:
                    rc.messages.append({"role": "user", "content": f"REJECTED: {err}"})
                    rc.ctx.trajectory.append({"role": "tool",
                                              "content": f"REJECTED: {err}"})
                    return

                rc.state = new_state

                # Sync harness state into ctx (like loop.py does)
                rc.ctx.fact_ids = list(getattr(rc.state, "fact_ids", []) or [])
                rc.ctx.fact_texts = dict(getattr(rc.state, "fact_texts", {}) or {})
                rc.ctx.fact_satisfies = dict(getattr(rc.state, "fact_satisfies", {}) or {})
                rc.ctx.fact_start_types = dict(getattr(rc.state, "fact_start_types", {}) or {})
                rc.ctx.fact_start_entities = dict(getattr(rc.state, "fact_start_entities", {}) or {})
                rc.ctx.fact_steps = list(getattr(rc.state, "fact_steps", []) or [])
                rc.ctx.chains = list(getattr(rc.state, "chains", []) or [])
                # Re-resolve the anchor from the model-declared entities so the
                # single-anchor walk (_do_select uses ctx.anchor_idx) follows what
                # the model declared — not the build_context q_entity default
                # (which blindly takes the first q_entity, e.g. "Gold medal" over
                # "Dewitt High School"). Multi-anchor walks resolve each chain
                # anchor independently via ctx.chains + _resolve_chain_anchor.
                # (run_react_case already does this; the batch path must too.)
                if tool_name == "decompose":
                    rc.ctx._model_anchor = getattr(rc.state, "anchor", None) or ""
                    rc.ctx._model_endpoints = list(getattr(rc.state, "endpoints", []) or [])
                    from kgqa.agent.loop import _resolve_anchor
                    _resolve_anchor(rc.ctx)

                # Dispatch tool (reuses existing _do_* functions)
                try:
                    result_str = await T.dispatch(tool_name, parsed_args, rc.ctx, session)
                except Exception as e:
                    result_str = json.dumps({"error": f"dispatch_{tool_name}: {e}"})

                # Append tool result (as user message in content-only mode)
                rc.messages.append({"role": "user",
                                    "content": f"Tool result ({tool_name}): {result_str}"})
                rc.ctx.trajectory.append({"role": "tool", "name": tool_name,
                                          "content": result_str})

                # Check completion
                if rc.state.state == "DONE":
                    # ── Boundary defense: empty-answer retry (parity with run_react_case) ──
                    # An empty answer scores 0; a best-guess from the retrieved pool
                    # retains recall. When the model emits `answer` with NO entities
                    # (over-cautious "can't verify the constraint"), feed the signal
                    # back ONCE and force a guess. The model still picks which entity.
                    ans_entities = (parsed_args.get("entities")
                                    if tool_name == "answer" else None) or []
                    if (tool_name == "answer" and not ans_entities
                            and not getattr(rc, "_empty_answer_retried", False)
                            and rc.ctx.selected_candidates):
                        rc._empty_answer_retried = True
                        rc.state.state = "ANSWER"  # roll back so `answer` is legal again
                        cand_pool = rc.ctx.selected_candidates[:50]
                        nudge = (
                            "Your answer was empty. An empty answer scores 0 — you must "
                            "output your best guess. From the retrieved candidate pool "
                            f"{cand_pool}, pick the entity (or entities) MOST likely to "
                            "answer the question and call `answer` again. When you "
                            "cannot verify a constraint from the graph, default to "
                            "keeping the candidates that best match the rest of the "
                            "question; never output an empty list."
                        )
                        rc.messages.append({"role": "user", "content": nudge})
                        rc.ctx.trajectory.append({"role": "tool", "name": "answer",
                                                  "content": nudge})
                    else:
                        rc.done = True

            # Process all cases concurrently (dispatch is mostly fast except retrieve)
            dispatch_sem = asyncio.Semaphore(16)
            async def process_with_sem(rc, raw, rsn):
                async with dispatch_sem:
                    await process_case(rc, raw, rsn)

            await asyncio.gather(*[
                process_with_sem(rc, raw, rsn)
                for rc, raw, rsn in zip(active, responses, responses_reasoning)
            ])

            done_count = sum(1 for rc in react_cases if rc.done)
            failed_count = sum(1 for rc in react_cases if rc.failed)
            print(f"  Round {round_num+1} done: {done_count} completed, "
                  f"{failed_count} failed, {total - done_count - failed_count} active",
                  flush=True)

    wall_time = time.perf_counter() - wall_start
    print(f"\n=== {total} cases in {wall_time:.1f}s ({total/wall_time:.1f} cases/s) ===",
          flush=True)

    # Build result dicts (reuse existing _ctx_to_result_dict)
    results = []
    for rc in react_cases:
        if rc.failed or not rc.done:
            # Mark as failed if not done
            if not rc.done:
                rc.failed = True
                rc.failure_reason = f"max_rounds ({max_rounds}) reached at state {rc.state.state}"
        result = _ctx_to_result_dict(rc.ctx, rc.state, rc.failed, rc.failure_reason)
        result["decomposition_method"] = "react_batch"
        results.append(result)

    # Write results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda x: x.get("case_num", 0))
    json.dump(results, open(out / "results.json", "w"), indent=2, ensure_ascii=False)

    # Print summary
    _print_summary(results, wall_time)
    return results


def _print_summary(results, wall_time):
    """Print GT/LLM hit + F1 summary (same format as run_agent_mode)."""
    n = len(results)
    gt_hit = sum(1 for r in results if r.get("gt_hit"))
    llm_hit = sum(1 for r in results if r.get("llm_hit"))
    f1 = sum(r.get("llm_f1", 0) for r in results) / n if n else 0
    failed = sum(1 for r in results if r.get("agent_failed"))

    # Hit@1: first predicted entity matches gold (ranking-based, primary metric).
    # Requires the model to RANK its answers (best guess first) — see AGENTS.md
    # §answer. Without ranking, Hit@1 ≈ F1 (set emission, no discrimination).
    from kgqa.core.utils import normalize as _norm
    def _hit_at_1(r):
        ans = r.get("llm_answer", "") or ""
        preds = [p.strip() for p in ans.split(" | ") if p.strip()] if ans else []
        if not preds:
            return False
        gt = r.get("gt_answers") or []
        if not gt:
            return False
        p1 = _norm(preds[0])
        return any(p1 in _norm(g) or _norm(g) in p1 for g in gt)
    hit1 = sum(1 for r in results if _hit_at_1(r))

    gh = [r for r in results if r.get("gt_hit")]
    n2 = len(gh)
    f1g = sum(r.get("llm_f1", 0) for r in gh) / n2 if n2 else 0
    prec = sum(r.get("llm_precision", 0) for r in gh) / n2 if n2 else 0
    rec = sum(r.get("llm_recall", 0) for r in gh) / n2 if n2 else 0

    print(f"\n{'='*60}")
    print(f"ReAct Batch Results: {n} cases in {wall_time:.1f}s")
    print(f"{'='*60}")
    print(f"GT hit:   {gt_hit}/{n} ({gt_hit/n:.3f})" if n else "GT hit: 0")
    print(f"LLM hit:  {llm_hit}/{n} ({llm_hit/n:.3f})" if n else "LLM hit: 0")
    print(f"Hit@1:    {hit1}/{n} ({hit1/n:.3f})" if n else "Hit@1: 0")
    print(f"Failed:   {failed}")
    print(f"Overall F1:     {f1:.4f}")
    print(f"GT-hit F1:      {f1g:.4f}  (n={n2})")
    print(f"GT-hit Prec:    {prec:.4f}")
    print(f"GT-hit Recall:  {rec:.4f}")


# ---------------------------------------------------------------------------
# Single-case ReAct runner — mirrors loop.run_agent_case's signature but uses
# the CONTENT-ONLY tool: protocol (the format the model was TRAINED on). This
# is the serving path that matches the training data format, unlike
# loop.run_agent_case which uses native tool_calls (which the model never saw
# in training and which drops the model's CoT reasoning).
# ---------------------------------------------------------------------------
async def run_react_case(session: aiohttp.ClientSession, sample: Dict[str, Any],
                         pilot_row: Dict[str, Any], idx: int,
                         args) -> Dict[str, Any]:
    """Run one case through the ReAct content-only loop.

    Each turn: call_llm returns free-form content; parse_react_output extracts
    the ``tool:`` anchor + JSON; harness.validate + tools.dispatch run it.
    Identical parsing to sample_trajectories.py, so training and serving agree.
    """
    from kgqa.llm.client import _call_single_with_reasoning

    rc = ReactCase(sample, pilot_row, idx)
    max_rounds = int(getattr(args, "agent_max_iters", 16))
    max_tokens = int(getattr(args, "agent_max_tokens", 1024))
    # Raise the max_tokens floor when thinking is on (see run_react_batch).
    from kgqa.llm.client import THINKING_TOKEN_BUDGET as _tb
    if _tb > 0:
        max_tokens = max(max_tokens, _tb + 2560)

    for _ in range(max_rounds):
        if not rc.is_active:
            break
        # Nudge with the allowed-tools hint (the model needs this since there
        # are no injected tool schemas in content-only mode).
        msgs = list(rc.messages)
        msgs.append({"role": "user", "content": rc.allowed_tools_hint()})

        try:
            # _call_single_with_reasoning returns (content, reasoning) so we
            # can capture the model's CoT in the trajectory for training data
            # generation and for diagnosing reasoning-answer consistency.
            raw_response, reasoning = await _call_single_with_reasoning(
                session, msgs, max_tokens=max_tokens)
        except Exception as e:
            rc.failed = True
            rc.failure_reason = f"call_llm error: {e}"
            break

        if not raw_response or not raw_response.strip():
            rc.messages.append({"role": "assistant", "content": ""})
            rc.messages.append({"role": "user", "content": rc.allowed_tools_hint()})
            continue

        # Append assistant content verbatim (preserves the CoT reasoning).
        # Store reasoning on the message dict so trajectory_to_messages
        # (sample_trajectories.py) can include it in training data. vLLM
        # separates <think> into a `reasoning` field; we re-attach it here so
        # the message is self-contained for SFT/GRPO trajectory export.
        asst_msg = {"role": "assistant", "content": raw_response}
        if reasoning:
            asst_msg["reasoning"] = reasoning
        rc.messages.append(asst_msg)
        traj_step = {"role": "assistant", "content": raw_response}
        if reasoning:
            traj_step["reasoning"] = reasoning
        rc.ctx.trajectory.append(traj_step)

        tool_name, parsed_args = parse_react_output(raw_response)
        if not tool_name:
            rc.messages.append({"role": "user",
                                "content": f"Could not parse tool call. "
                                f"Output JSON like: {{\"tool\": \"...\", \"args\": {{...}}}}. "
                                f"{rc.allowed_tools_hint()}"})
            rc.ctx.trajectory.append({"role": "tool", "content": "(unparsed)"})
            continue

        parsed_args = parsed_args or {}
        tool_calls = _to_tool_calls(tool_name, parsed_args)
        ok, err, new_state = validate(rc.state, tool_calls)
        if not ok:
            rc.messages.append({"role": "user", "content": f"REJECTED: {err}"})
            rc.ctx.trajectory.append({"role": "tool", "content": f"REJECTED: {err}"})
            continue
        rc.state = new_state

        rc.ctx.fact_ids = list(getattr(rc.state, "fact_ids", []) or [])
        rc.ctx.fact_texts = dict(getattr(rc.state, "fact_texts", {}) or {})
        rc.ctx.fact_satisfies = dict(getattr(rc.state, "fact_satisfies", {}) or {})
        rc.ctx.fact_start_types = dict(getattr(rc.state, "fact_start_types", {}) or {})
        rc.ctx.fact_start_entities = dict(getattr(rc.state, "fact_start_entities", {}) or {})
        rc.ctx.fact_steps = list(getattr(rc.state, "fact_steps", []) or [])
        # Sync model-chosen anchor/endpoints so _resolve_anchor can use them.
        # Done right after decompose validate, before dispatch runs GTE.
        rc.ctx._model_anchor = getattr(rc.state, "anchor", None) or ""
        rc.ctx._model_endpoints = list(getattr(rc.state, "endpoints", []) or [])
        if tool_name == "decompose":
            # Re-resolve anchor/endpoints now that the model has chosen them.
            from kgqa.agent.loop import _resolve_anchor
            _resolve_anchor(rc.ctx)

        try:
            result_str = await T.dispatch(tool_name, parsed_args, rc.ctx, session)
        except Exception as e:
            result_str = json.dumps({"error": f"dispatch_{tool_name}: {e}"})

        rc.messages.append({"role": "user",
                            "content": f"Tool result ({tool_name}): {result_str}"})
        rc.ctx.trajectory.append({"role": "tool", "name": tool_name,
                                  "content": result_str})

        if rc.state.state == "DONE":
            # ── Boundary defense: empty-answer retry ──
            # When the model emits `answer` with NO entities (an over-cautious
            # "I can't verify the constraint, so no answer"), that is almost
            # always worse than guessing from the candidate pool — empty scores
            # 0 by definition, while a best-guess retains recall. Detect this
            # and feed the signal back to the model ONCE, forcing it to pick the
            # most likely candidate from the select-stage pool (which is fuller
            # than the expand pool — expand can drop candidates). This is a
            # boundary defense, not the harness choosing for the model: the
            # model still picks which entity.
            ans_entities = (parsed_args.get("entities")
                            if tool_name == "answer" else None) or []
            if (tool_name == "answer" and not ans_entities
                    and not getattr(rc, "_empty_answer_retried", False)
                    and rc.ctx.selected_candidates):
                rc._empty_answer_retried = True
                # Roll state back so `answer` is legal again on the next turn.
                rc.state.state = "ANSWER"
                cand_pool = rc.ctx.selected_candidates[:50]
                nudge = (
                    "Your answer was empty. An empty answer scores 0 — you must "
                    "output your best guess. From the retrieved candidate pool "
                    f"{cand_pool}, pick the entity (or entities) MOST likely to "
                    "answer the question and call `answer` again. When you "
                    "cannot verify a constraint from the graph, default to "
                    "keeping the candidates that best match the rest of the "
                    "question; never output an empty list."
                )
                rc.messages.append({"role": "user", "content": nudge})
                rc.ctx.trajectory.append({"role": "tool", "name": "answer",
                                          "content": nudge})
                continue
            rc.done = True
            break
    else:
        rc.failed = True
        rc.failure_reason = f"max_iters ({max_rounds}) reached at state {rc.state.state}"

    result = _ctx_to_result_dict(rc.ctx, rc.state, rc.failed, rc.failure_reason)
    result["decomposition_method"] = "react_single"
    return result
