"""KGQA 多轮工具 rollout 接入 ms-swift（colocate 与 server 两种模式）.

关键契约（swift 4.6.0，`swift/rollout/agent_loop.py:run_multi_turn`）
--------------------------------------------------------------------
- **colocate 模式**（`--vllm_mode colocate`，我们用的）走 `run_multi_turn`，
  它每轮调用：
    1. `scheduler.append_assistant_completion(req, completion)`（基类追加助手消息）
    2. `await scheduler.on_turn_end(req, choice, turn)`  ← **工具执行放这里（async）**
    3. 未结束时 `scheduler.step(req, choice, turn)`     ← 同步，设置下一轮 messages
    4. `scheduler.check_finished(...)`                  ← 终止判定
- 覆写 `run()` 只对 **server 模式**生效；colocate 下会被忽略（本文件早期版本
  即因此没有执行——已修正）。
- 每轮返回的 `response_token_ids` / `response_loss_mask` 由 driver 收集；
  基类 `get_response_token_data` 会自动为确定性前缀补 mask=0。

监督面
------
**所有 assistant 回复都监督**（用户裁定 2026-10-22）：每轮生成 token 的
mask 全 1；工具结果/提示不是模型生成，不进 ids（由下一轮 prompt 承载）。
"""

from __future__ import annotations

import os
import sys
import time as _time
import uuid as _uuid

import aiohttp

from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

from kgqa.agent.seq_react_loop import SeqReactCase
from kgqa.rl.verl_plugins.opsd_pair import split_thinking, strip_turn_terminator


def _dbg(msg: str) -> None:
    if os.environ.get("KGQA_DEBUG"):
        print(f"[kgqa {_time.strftime('%H:%M:%S')}] r{os.environ.get('RANK', '?')} {msg}",
              file=sys.stderr, flush=True)


# Wall-clock phase spans (KGQA_PHASE_PROFILE=1): every value is a single-clock
# interval on the main thread — concurrent activity inside a span is NOT
# summed twice. Sub-spans (prelink inside rollout, loss_fwd inside train) are
# reported nested, never added into the parent total.
_PHASE: dict[str, float] = {}
_PHASE_T0: dict[str, float] = {}
# absolute wallclock window per span (first start, last end) for aligning
# with external GPU-utilization timelines
_PHASE_TS: dict[str, list] = {}


def _phase_begin(name):
    if os.environ.get("KGQA_PHASE_PROFILE") == "1":
        _PHASE_T0[name] = _time.monotonic()
        w = _PHASE_TS.setdefault(name, [None, None])
        if w[0] is None:
            w[0] = _time.time()


def _phase_end(name):
    if os.environ.get("KGQA_PHASE_PROFILE") == "1":
        t0 = _PHASE_T0.pop(name, None)
        if t0 is not None:
            _PHASE[name] = _PHASE.get(name, 0.0) + (_time.monotonic() - t0)
            w = _PHASE_TS.setdefault(name, [None, None])
            w[1] = _time.time()


def _ts(name, i):
    w = _PHASE_TS.get(name)
    if not w or w[i] is None:
        return "--:--:--"
    return _time.strftime("%H:%M:%S", _time.localtime(w[i]))


# Sub-timers accumulated inside the chunked compute_loss (reported via the
# phase summary; wall intervals, sequentially exclusive).
_CHUNK_STATS: dict[str, float] = {}


class ToolLayerBroken(RuntimeError):
    """Raised by the [tool] hook when internal tool-execution errors exceed
    KGQA_TOOL_ERR_ABORT in one step — the rollout environment is broken and
    training on it is wasted GPU-hours. Dedicated type so the [tool] block's
    own error handler re-raises instead of swallowing the abort."""

# Tokenizer for the exact context guard. The scheduler instance has no
# trainer reference, so the dynamic sampler's _generate_completions (self =
# trainer) stashes processing_class here BEFORE the rollout runs — the
# on_turn_end guard then uses it. None until the first step / in server mode
# → the guard falls back to the coarse chars/4.5 estimate.
_GUARD_TOK = None
# Optimizer step the CURRENT rollout belongs to (1-based): set alongside
# _GUARD_TOK; rollout for step k runs while trainer.state.global_step is
# still k-1. Feeds the per-step rollout dump subdirectory so 100 steps of
# trajectories never overwrite each other (step1 vs step2 comparison is the
# whole point of the dump).
_STEP_NOW = 0


def _content_tokens(tok, msgs):
    """Exact token count of the message CONTENTS only (no wrapper)."""
    if tok is None or not msgs:
        return None
    text = "\n".join(str(m.get("content") or "") for m in msgs)
    try:
        return len(tok(text, add_special_tokens=False)["input_ids"])
    except Exception:  # noqa: BLE001
        return None


def _exact_msg_tokens(tok, msgs):
    """Tokenizer-accurate upper bound of the rendered conversation length:
    exact content tokens (via _content_tokens) + ~10 wrapper tokens per
    message (im_start/role/im_end; the real chat template uses 4-8) + a flat
    256 for system and generation-prompt overhead. Returns None when
    counting fails — callers must fall back to the coarse estimate, never
    to zero."""
    n = _content_tokens(tok, msgs)
    if n is None:
        return None
    return n + 10 * len(msgs) + 256


# ---- 人读轨迹导出（KGQA_DUMP_DIR=<dir>）-----------------------------------
def _dump_trajectory(st: dict, rc, reason: str) -> None:
    """One case → <dir>/step{N}/rank{R}/<case>.md + .json for human audit.

    The Markdown is the audit view (thinking / action / tool result / hint are
    separated, long tool payloads truncated); the JSON keeps every message
    verbatim so a suspected environment error can be checked without loss.
    Step-level subdirectory (2026-10-24): the same 4 cases roll EVERY step —
    a flat rank dir either overwrites (no uid) or drowns the step1-vs-step2
    comparison the dump exists for (crash forensics).
    """
    if not os.environ.get("KGQA_DUMP_DIR") or st.get("dumped"):
        return
    st["dumped"] = True
    import json
    import pathlib
    import re

    root = pathlib.Path(os.environ["KGQA_DUMP_DIR"])
    rank = os.environ.get("RANK", "0")
    out_dir = root / f"step{_STEP_NOW:03d}" / f"rank{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_id = st.get("case_id") or "case"
    uid = str(st.get("uuid") or "")[-8:]
    safe = (re.sub(r"[^A-Za-z0-9._-]+", "_", case_id)[:110] or "case")
    if uid:
        safe = f"{safe}__{uid}"
    pilot = st.get("pilot") or {}
    gt = pilot.get("gt_answers") or []
    try:
        pred = rc._last_answer_entities()
    except Exception:  # noqa: BLE001
        pred = []
    f1 = _f1_score(pred, gt)
    meta = st.get("meta") or []
    turns = st.get("turns", 0)
    wall = _time.monotonic() - st.get("t0", _time.monotonic())

    def _text(v) -> str:
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return "" if v is None else str(v)

    system_prompt = ""
    body: list[str] = []
    for msg in rc.messages:
        role = msg.get("role")
        content = _text(msg.get("content"))
        if role == "system":
            system_prompt = system_prompt or content
            continue
        if role == "assistant":
            reasoning, action = split_thinking(content)
            if not reasoning and msg.get("reasoning"):
                reasoning = msg["reasoning"]
                action = content[len(reasoning):] if content.startswith(reasoning) else content
                action = action.lstrip("\n")
                if action.startswith("</think>"):
                    action = action[len("</think>"):].lstrip("\n")
            body.append("\n---\n### ASSISTANT\n")
            if reasoning:
                body.append(f"**Thinking** ({len(reasoning)} chars)\n\n")
                body.append("\n".join("> " + ln for ln in reasoning.splitlines()))
                body.append("\n\n")
            body.append("**Action / Answer**\n\n```\n")
            body.append(action or "(empty)")
            body.append("\n```\n")
            continue
        if role == "user":
            if "Tool result (" in content[:400]:
                pos = content.find("Tool result (")
                head, _, rest = content[pos:].partition("\n")
                prefix = content[:pos].strip()
                limit = int(os.environ.get("KGQA_DUMP_TOOL_CHARS", "6000"))
                shown = rest if len(rest) <= limit else (
                    rest[:limit] + f"\n... (truncated; total {len(rest)} chars, "
                    f"full payload in {safe}.json)")
                body.append(f"\n---\n### TOOL\n`{head}`\n\n")
                if prefix:
                    body.append(f"{prefix}\n\n")
                body.append(f"```\n{shown}\n```\n")
            elif content.startswith("REJECTED:"):
                body.append(f"\n---\n### REJECTED / NUDGE\n\n```\n{content}\n```\n")
            else:
                body.append(f"\n---\n### USER (hint / question)\n\n{content}\n")
            continue
        body.append(f"\n---\n### {str(role).upper()}\n\n{content}\n")

    lines = [
        f"# {case_id}\n",
        f"- Step: {_STEP_NOW}\n",
        f"- Question: {pilot.get('question') or (rc.ctx.question if hasattr(rc, 'ctx') else '')}\n",
        f"- GT answers: {gt}\n",
        f"- Predicted: {pred}\n",
        f"- F1: {f1:.4f}\n",
        f"- Result: {reason} | turns={turns} | wall={wall:.1f}s | "
        f"failed={getattr(rc, 'failed', False)}"
        f"{(' | ' + rc.failure_reason) if getattr(rc, 'failure_reason', '') else ''}\n",
        "\n## Turn log\n\n",
        "| # | tokens | tool_s | done | failed | raw head |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for m in meta:
        head = (m.get("raw_head") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {m.get('turn')} | {m.get('ids')} | {m.get('tool_s')} | "
            f"{m.get('done')} | {m.get('failed')} | {head} |\n")
    lines.append("\n## Transcript\n")
    lines.extend(body)
    (out_dir / f"{safe}.md").write_text("".join(lines), encoding="utf-8")

    payload = {
        "step": _STEP_NOW,
        "case_id": case_id,
        "question": pilot.get("question"),
        "gt_answers": gt,
        "predicted": pred,
        "f1": f1,
        "result": reason,
        "turns": turns,
        "wall_s": round(wall, 1),
        "failed": bool(getattr(rc, "failed", False)),
        "failure_reason": getattr(rc, "failure_reason", ""),
        "turn_meta": meta,
        "messages": rc.messages,
    }
    (out_dir / f"{safe}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")

    if system_prompt:
        sp = out_dir / "_system_prompt.md"
        if not sp.exists():
            sp.write_text(system_prompt, encoding="utf-8")

    idx = out_dir / "_index.md"
    new = not idx.exists()
    with open(idx, "a", encoding="utf-8") as f:
        if new:
            f.write("# Trajectory index\n\n"
                    "| case | turns | result | F1 | pred | question |\n"
                    "|---|---|---|---|---|---|\n")
        q = (pilot.get("question") or "").replace("|", "\\|").replace("\n", " ")
        f.write(f"| [{safe}]({safe}.md) | {turns} | {reason} | {f1:.3f} | "
                f"{pred} | {q} |\n")


def _install_traj_dumper() -> None:
    """Dump finished + unfinished cases at process exit (gated by env)."""
    global _DUMPER_INSTALLED
    if _DUMPER_INSTALLED or not os.environ.get("KGQA_DUMP_DIR"):
        return
    _DUMPER_INSTALLED = True
    import atexit

    def _flush() -> None:
        for sched in _SCHEDULERS:
            for st in list(getattr(sched, "_state", {}).values()):
                try:
                    _dump_trajectory(st, st.get("rc"), "unfinished")
                except Exception:  # noqa: BLE001
                    pass

    atexit.register(_flush)


_SCHEDULERS: list = []
_DUMPER_INSTALLED = False


class KGQAScheduler(MultiTurnScheduler):
    """KGQA 多轮工具循环（colocate 兼容）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_rounds = int(os.environ.get(
            "MAX_ROUNDS", os.environ.get("SEQ_MAX_ROUNDS", "16")))
        self.gte_urls = os.environ.get("GTE_API_URLS", "http://localhost:8003")
        self._state: dict[str, dict] = {}
        self._session: aiohttp.ClientSession | None = None
        self._session_loop = None
        self.max_model_len = int(os.environ.get("KGQA_ROLLOUT_MAX_LEN", "0") or 0)
        if os.environ.get("KGQA_DUMP_DIR"):
            _SCHEDULERS.append(self)
            _install_traj_dumper()

    # ---- 生命周期 ------------------------------------------------------
    async def _get_session(self) -> aiohttp.ClientSession:
        # ms-swift 用临时 loop 执行 on_trajectory_start（invoke_async_hook），
        # 再用另一个 loop 驱动 run_multi_turn。session 若跨 loop 复用会抛
        # "Event loop is closed"（实测一次 rollout 120 次工具失败）。
        import asyncio
        loop = asyncio.get_running_loop()
        if (self._session is None or self._session.closed
                or self._session_loop is not loop):
            # Best-effort release of the replaced session (audit P2-E):
            # ClientSession.close()/TCPConnector.close() are coroutines bound
            # to the OLD loop, which is already closed here — awaiting is
            # impossible. The connector's sync core (_close) releases the
            # transports; mark the session closed to silence the warning.
            old = self._session
            if old is not None and not old.closed:
                try:
                    conn = getattr(old, "_connector", None)
                    if conn is not None:
                        try:
                            conn._close()
                        except Exception:  # noqa: BLE001
                            coro = conn.close()
                            if asyncio.iscoroutine(coro):
                                coro.close()
                    old._closed = True
                except Exception:  # noqa: BLE001
                    pass
            self._session = aiohttp.ClientSession()
            self._session_loop = loop
        return self._session

    async def close_session(self) -> None:
        """Public close inside the owning loop (audit R4-03)."""
        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
            except Exception:  # noqa: BLE001
                pass
        self._session = None
        self._session_loop = None

    async def on_trajectory_start(self, requests):
        # Use a per-call session. Sharing self._session here raced in server
        # mode (one trajectory's start hook closed the session another was
        # using, audit F06); keeping it local also avoids the colocate
        # cross-event-loop binding entirely.
        async with aiohttp.ClientSession() as session:
            for req in requests:
                key = getattr(req, "uuid", None) or _uuid.uuid4().hex
                req.uuid = key
                if key in self._state:
                    continue
                data = getattr(req, "data_dict", {}) or {}
                sample = data.get("kgqa_sample")
                if sample is None:
                    _dbg(f"uuid={key} 无 kgqa_sample → 回退基类")
                    continue
                pilot = data.get("kgqa_pilot") or {}
                idx = int(data.get("kgqa_idx", 0))
                rc = SeqReactCase(sample, pilot, idx)
                # 教师侧在 system 末尾追加精炼参考；训练 encode 的长度必须把它
                # 算进去，否则 student 侧刚好在 32k 内、teacher 侧越界。
                _tp = str(data.get("teacher_prompt") or "")
                self._state[key] = {"rc": rc, "turns": 0, "pilot": pilot, "uuid": key,
                                    "case_id": pilot.get("case_id", sample.get("id", "")),
                                    "teacher_tokens": int(len(_tp) / 4.5),
                                    "last_ids": [],
                                    "t0": _time.monotonic()}
                _dbg(f"start case={pilot.get('case_id', sample.get('id', ''))}")
                await rc.prelink(session)
                # prelink 把 canonical anchors 注入 rc.messages；不回写的话
                # 首轮 prompt 仍是数据集原始消息（缺 anchors，与 9B 环境不同）。
                # driver 在本钩子后会执行 sample.messages = req.messages。
                if getattr(rc, "messages", None) is not None:
                    req.messages = [dict(m) for m in rc.messages]

    # ---- 每轮：工具执行（async） ---------------------------------------
    async def on_turn_end(self, infer_request, response_choice, current_turn):
        key = getattr(infer_request, "uuid", None)
        st = self._state.get(key)
        if st is None:
            return {"done": True}
        rc = st["rc"]
        raw = response_choice.message.content or ""
        if os.environ.get("KGQA_DEBUG") and current_turn == 1 and st["turns"] == 0:
            _rc_content = getattr(response_choice.message, "reasoning_content", None) or ""
            _dbg(f"raw case={st['case_id']} head={raw[:60]!r} "
                 f"has_close_think={'</think>' in raw} "
                 f"reasoning_content={len(_rc_content)}")
        ids = list(getattr(response_choice, "token_ids", None) or [])
        # vLLM 的 token_ids 含 stop token，decode 后尾部会有字面量 <|im_end|>；
        # 先剥掉，避免污染动作/历史（模板渲染时会自己补回合终止符）。
        raw, ids = strip_turn_terminator(raw, ids)
        reasoning, content = split_thinking(raw)
        if os.environ.get("KGQA_DEBUG") and current_turn == 1 and st["turns"] == 0:
            _dbg(f"strip case={st['case_id']} raw_tail={raw[-24:]!r} "
                 f"ids_tail={ids[-2:]}")
        # ── 思考预算耗尽伪影（2026-10-25）─────────────────────────────────
        # 预算(THINKING_BUDGET)耗尽时 vLLM 强行注入 reasoning_end_str
        # "I will now emit the tool call based on the reasoning above.</think>"
        # 该字符串落在 assistant 输出里 → 进 messages → 进 labels → 模型学到
        # 在思考中途复现它并切碎动作(tool: retrieve + 短语) → ACTION_FIELDS
        # 拒绝 → 重试/复读。step5 有 37% 的轮中招（step1 仅 4%）。
        # 处理: ① 从写入历史的文本里删掉短语（TITO 的 ids 独立, 不影响编码）;
        #       ② 把该轮标记为伪影轮, mask 分支会把它的 loss_mask 置零。
        _ART_PHRASE = "I will now emit the tool call based on the reasoning above"
        _artifact = _ART_PHRASE in raw
        if _artifact:
            _i = raw.index(_ART_PHRASE)
            _tail = raw[_i + len(_ART_PHRASE):]
            if _tail.lstrip().startswith("</think>"):
                _tail = _tail.lstrip()[len("</think>"):]
            raw = (raw[:_i] + _tail).strip()
            content = raw.rsplit("</think>", 1)[-1].strip() if "</think>" in raw else raw
            st.setdefault("artifact_turns", []).append(int(current_turn))
            _dbg(f"budget-artifact case={st['case_id']} turn={current_turn} "
                 f"(phrase stripped from history)")
        t_tool = _time.monotonic()
        try:
            # 裁定 2026-10-22: 原始轨迹必须保留 Thinking。工具解析仍只吃
            # action 部分 (content)，但写回历史的 assistant 消息用完整 raw，
            # 否则规范轨迹被裁成"只有行为"，教师/学生上下文与监督目标脱节。
            await rc.process_turn(await self._get_session(), content, reasoning,
                                  history_text=raw)
        except Exception as e:  # noqa: BLE001
            rc.failed = True
            rc.failure_reason = f"scheduler tool error: {e}"
        # 轨迹级环境故障熔断（用户裁定 2026-10-24: 封闭错误集——模型只看
        # 得到 ENVIRONMENT_UNAVAILABLE 这类可操作协议错误, 内部细节走环境
        # 侧遥测; 训练复用离线的轨迹失败语义）: 连续
        # KGQA_EXEC_ERR_STREAK 轮（默认 3）工具回包都是环境错误 →
        # rc.failed="tool_layer_error"，与离线 rollouts 的轨迹失败同一通道
        # （终止 → discard → mask），不再烧轮次。
        if not (rc.done or rc.failed):
            streak = st.get("exec_err_streak", 0)
            hit = False
            for m in reversed(rc.messages or []):
                if m.get("role") != "user":
                    break               # this turn's tool feedback ends here
                c = str(m.get("content") or "")
                if c.startswith("Tool result (") and "ENVIRONMENT_UNAVAILABLE" in c[:400]:
                    hit = True
                    break
            streak = streak + 1 if hit else 0
            st["exec_err_streak"] = streak
            if streak >= int(os.environ.get("KGQA_EXEC_ERR_STREAK", "3") or 0):
                rc.failed = True
                rc.failure_reason = "tool_layer_error"
                st["discard"] = True
                _dbg(f"tool_layer_error case={st['case_id']} turn={current_turn} "
                     f"streak={streak} (env broken — trajectory failed, "
                     f"offline semantics)")
        _dbg(f"tool case={st['case_id']} turn={current_turn} "
             f"s={_time.monotonic() - t_tool:.2f} ids={len(ids)}")
        st["turns"] = current_turn
        st["last_ids"] = list(ids)
        # Context guard must run HERE, before `done` is reported. The driver
        # prefers the `done` field over check_finished()'s return value, so a
        # guard that only lives in check_finished() would set rc.failed but
        # still let the next generation happen (audit F04).
        if self.max_model_len and not (rc.done or rc.failed):
            try:
                # Context guard (2026-10-24, incremental-exact version).
                # Estimate from the REQUEST messages (rc.messages misses the
                # per-turn hints the driver appends). The old chars/4.5
                # coarse estimate can undercount entity-dense tool JSON by
                # >10% (a band-gated exact check still missed a 26k-coarse /
                # 50.8k-real row — chars/token ≈ 2.8 sits below ANY practical
                # band floor), letting a too-long prompt reach vLLM, which
                # rejects the WHOLE batch and kills the training process.
                # So: count tokens EXACTLY, but INCREMENTALLY — per turn only
                # the messages appended since the last guard run are
                # tokenized (O(delta), ~2-5ms) and accumulated in st; a full
                # recount is forced every 12 turns or when the list shrank
                # (empty-assistant deletion) to bound drift. Wrapper overhead
                # (10/msg + 256) is recomputed on the full list each turn.
                # trip = exact + 2048 > limit (2048 = next completion 1500 +
                # hints ~150 + slack). No tokenizer → coarse fallback with
                # margin 4096.
                est_msgs = getattr(infer_request, "messages", None) or rc.messages
                limit = self.max_model_len
                trip = None
                plen = None
                if _GUARD_TOK is not None and est_msgs:
                    upto = st.get("exact_upto", 0)
                    run = st.get("exact_running", 0)
                    if upto > len(est_msgs) or current_turn % 12 == 0:
                        upto, run = 0, 0       # shrink or periodic recount
                    if upto < len(est_msgs):
                        add = _content_tokens(_GUARD_TOK, est_msgs[upto:])
                        if add is not None:
                            run += add
                            upto = len(est_msgs)
                            st["exact_running"] = run
                            st["exact_upto"] = upto
                    if upto == len(est_msgs):
                        plen = run + 10 * len(est_msgs) + 256
                        trip = plen + 2048 > limit
                if trip is None:
                    plen = int(sum(len(str(m.get("content") or ""))
                                   for m in est_msgs) / 4.5)
                    trip = plen + 4096 > limit
                if trip:
                    rc.failed = True
                    rc.failure_reason = "context_limit"
                    st["discard"] = True
                    st["too_long"] = True
                    _dbg(f"context_limit case={st['case_id']} turn={current_turn} "
                         f"prompt~{plen} limit={limit}")
            except Exception:  # noqa: BLE001
                pass
        # 复读熔断（2026-10-25）: 连续 KGQA_REP_STREAK 轮 assistant 输出逐字相同
        # → 模型已陷入复读, 继续跑到 24 轮只会把上下文撑爆（step5: 27/256 条
        # 轨迹重复 ≥5 轮、19 条打满 24 轮、10 条因此被截断）。这类轨迹本就多被
        # mask（F1=0）, 早停只省 rollout 预算。
        # unchanged_query 升级（2026-10-25）: 环境回 "progress: unchanged_query"
        # 说明模型在原地重试同一查询; 连续 KGQA_UQ_MAX(默认3) 轮 → 轨迹失败,
        # 不再烧轮次（提示语本身无需改: 它已明确"先推断作答/换未执行的步骤"）。
        _uq = 0
        for _m in reversed(rc.messages or []):
            if _m.get("role") != "user":
                break
            if "unchanged_query" in str(_m.get("content") or ""):
                _uq += 1
            else:
                break
        _uq_max = int(os.environ.get("KGQA_UQ_MAX", "3") or 0)
        if _uq_max and _uq >= _uq_max and not (rc.done or rc.failed):
            rc.failed = True
            rc.failure_reason = "repeated_query"
            st["discard"] = True
            _dbg(f"repeated_query case={st['case_id']} turn={current_turn} "
                 f"consecutive={_uq}")
        if not (rc.done or rc.failed):
            _streak = int(os.environ.get("KGQA_REP_STREAK", "4") or 0)
            _hist = st.setdefault("raw_hist", [])
            _hist.append(raw.strip())
            if len(_hist) > _streak:
                _hist.pop(0)
            if _streak and len(_hist) >= _streak and len(set(_hist)) == 1:
                rc.failed = True
                rc.failure_reason = "degenerate_repetition"
                st["discard"] = True
                _dbg(f"degenerate_repetition case={st['case_id']} "
                     f"turn={current_turn} streak={_streak}")
        done = bool(rc.done or rc.failed or current_turn >= self.max_rounds)
        # Per-turn sampled-token counts (audit R4-02): text emptiness and id
        # emptiness disagree for whitespace-token turns — deletion must use
        # the TOKEN count so messages and id slots stay aligned.
        st.setdefault("ids_history", []).append(len(ids))
        if done:
            # Drop every assistant whose turn produced ZERO tokens
            # (stop-only), together with the nudge user message that directly
            # follows it. Whitespace-token turns (ids non-empty) are KEPT —
            # their message keeps a non-empty id slot (audit R4-02). Mutate
            # the list IN PLACE: the server driver captured this exact list
            # reference before the hook and returns it.
            msgs = infer_request.messages
            if msgs is not None:
                zero_ids = set(k for k, n in enumerate(st["ids_history"])
                               if n == 0)
                if zero_ids:
                    ai = -1          # assistant ordinal, tracks ids_history
                    keep = []
                    i = 0
                    while i < len(msgs):
                        m = msgs[i]
                        if m.get("role") == "assistant":
                            ai += 1
                            if ai in zero_ids:
                                nxt = (msgs[i + 1]
                                       if i + 1 < len(msgs) else None)
                                i += 2 if (nxt is not None and
                                           nxt.get("role") == "user") else 1
                                continue
                        keep.append(m)
                        i += 1
                    if len(keep) != len(msgs):
                        msgs[:] = keep
        if os.environ.get("KGQA_DUMP_DIR"):
            st.setdefault("meta", []).append({
                "turn": current_turn,
                "ids": len(ids),
                "tool_s": round(_time.monotonic() - t_tool, 2),
                "done": bool(done),
                "failed": bool(rc.failed),
                "raw_head": (raw[:160] + ("..." if len(raw) > 160 else "")),
            })
        if done:
            # 格式拒绝熔断（2026-10-25）: 一条轨迹里 action-format 拒绝
            # (MULTIPLE_ACTIONS / ACTION_REQUIRED / ACTION_FIELDS) >= 阈值,
            # 说明它陷入"多动作文章/字段错误"循环 —— 这些 token 是噪声,
            # 整条丢弃不进训练（KGQA_FMT_REJ_MAX=0 关闭）。
            _fmt_rej = 0
            for _m in (rc.messages or []):
                if _m.get("role") == "user":
                    _t = str(_m.get("content") or "").lstrip()
                    if _t.startswith(("error: MULTIPLE_ACTIONS",
                                      "error: ACTION_REQUIRED",
                                      "REJECTED: ACTION_FIELDS")):
                        _fmt_rej += 1
            _fmt_max = int(os.environ.get("KGQA_FMT_REJ_MAX", "3") or 0)
            if _fmt_max and _fmt_rej >= _fmt_max:
                st["discard"] = True
                st["fmt_rejected"] = True
                _dbg(f"fmt_rejected case={st['case_id']} "
                     f"rejections={_fmt_rej} -> discard")
            try:
                _pred = rc._last_answer_entities()
            except Exception:  # noqa: BLE001
                _pred = []
            try:
                _f1 = _f1_score(_pred, (st.get("pilot") or {}).get("gt_answers") or [])
            except Exception:  # noqa: BLE001
                _f1 = 0.0
            st["f1"] = _f1
            st["correct"] = _f1 >= 0.999
            st["answer_empty"] = not _pred
            # 训练侧剔除：打满轮数仍未完成、或最终空答的轨迹不提供梯度
            # （采样多样性仍然保留；真正 resample 属于 trainer 层，下一步再做）。
            st["discard"] = (not rc.done) or (not _pred)
            # Pre-encode length guard: an over-long prompt would raise
            # MaxLengthError in ms-swift's template.encode; mark it for the
            # postprocess drop instead.
            _max_len = int(os.environ.get("KGQA_TRAIN_MAX_LEN", "0") or 0)
            if _max_len and not st["discard"]:
                try:
                    # rc.messages already contains every assistant turn and
                    # tool result; adding snapshot ids again double-counts the
                    # response (audit F04). Use content estimate + teacher
                    # reference + safety margin only.
                    _pt = int(sum(len(str(m.get("content") or ""))
                                  for m in rc.messages) / 4.5)
                    if _pt + st.get("teacher_tokens", 0) + 2048 > _max_len:
                        st["discard"] = True
                        st["too_long"] = True
                except Exception:
                    pass
            _dbg(f"traj case={st['case_id']} turns={current_turn} "
                 f"wall={_time.monotonic() - st['t0']:.1f}s failed={rc.failed} "
                 f"empty={st['answer_empty']} discard={st['discard']} "
                 f"too_long={st.get('too_long', False)}")
            _dump_trajectory(
                st, rc, "failed" if rc.failed else ("done" if rc.done else "max_rounds"))
            _record_f1(rc, st.get("pilot") or {}, st.get("uuid"))
        _dbg(f"turn={current_turn} ids={len(ids)} done={done} "
             f"failed={rc.failed} case={st['case_id']}")
        # 按轮拆分的快照（用户方案 2026-10-22）: 训练期一条轨迹 → N 个单轮
        # 样本, 复用同一条完整 teacher。driver 每轮 update rollout_infos,
        # 这里带累计列表, 末轮即完整快照集。messages_before = 去掉 driver
        # 刚追加的 assistant 消息（append_assistant_completion 先于本钩子）。
        # 不拆轮时这些深拷贝没人消费，保留会让每个 case 的 CPU 占用随轮数
        # 近似二次增长（audit F09）。
        if os.environ.get("KGQA_TURN_SPLIT", "0") == "1":
            import copy as _copy
            msgs_before = [_copy.deepcopy(m) for m in infer_request.messages]
            if msgs_before and msgs_before[-1].get("role") == "assistant":
                msgs_before = msgs_before[:-1]
            st.setdefault("snapshots", []).append({
                "messages_before": msgs_before,
                "ids": ids,
                "text": raw,
            })
        # 动作段监督（2026-10-25 用户裁定）: 只监督 `tool:` 之后的动作,
        # thinking 与 AUDIT 前言一律 mask。理由: ① 监督长思考会把"写长篇
        # 分析"当成目标（预算耗尽→注入串→复读的源头）; ② 我们真正要学的是
        # 决策动作; ③ AUDIT 块是模型自创格式, 监督它等于固化噪声。
        _mask = [1] * len(ids)
        if os.environ.get("KGQA_MASK_THINK", "0") == "1" and ids:   # 默认保留 thinking 监督（用户裁定 2026-10-25）
            _pos = raw.rfind("tool:")
            if _pos <= 0:
                # 没有可解析的动作（纯分析/文章轮）→ 整轮不监督
                _mask = [0] * len(ids)
            else:
                _tok = _GUARD_TOK
                _cut = len(ids)
                if _tok is not None:
                    try:
                        _cut = len(_tok(raw[:_pos], add_special_tokens=False)["input_ids"])
                    except Exception:  # noqa: BLE001
                        _cut = int(len(ids) * _pos / max(len(raw), 1))
                else:
                    _cut = int(len(ids) * _pos / max(len(raw), 1))
                _cut = max(0, min(_cut, len(ids)))
                _mask = [0] * _cut + [1] * (len(ids) - _cut)
        st["last_mask"] = list(_mask)   # 动作段掩码, 由 step() 回传给 driver
        result = {
            "done": done,
            "response_token_ids": ids,
            "response_loss_mask": _mask,
            "rollout_infos": {
                "kgqa_case_id": st["case_id"],
                "kgqa_uuid": st["uuid"],
                "kgqa_done": rc.done,
                "kgqa_failed": rc.failed,
                "kgqa_failure_reason": rc.failure_reason,
                "kgqa_answer_empty": bool(st.get("answer_empty", False)),
                "kgqa_f1": round(float(st.get("f1", 0.0)), 4),
                "kgqa_correct": bool(st.get("correct", False)),
                "kgqa_max_rounds": bool(not rc.done),
                "kgqa_discard": bool(st.get("discard", False)),
                "kgqa_too_long": bool(st.get("too_long", False)),
                # 伪影轮(思考预算耗尽时 vLLM 注入的 reasoning_end 串): mask
                # 分支按轮把它们的 loss_mask 置零, 不训练注入痕迹。
                "kgqa_artifact_turns": list(st.get("artifact_turns", [])),
                # 累计快照: [ {messages_before, ids, text}, ... ]
                "kgqa_turns": list(st.get("snapshots", [])),
                # 案级 teacher 条件原料（teacher_prompt 来自数据列）
                "kgqa_teacher_prompt":
                    (getattr(infer_request, "data_dict", {}) or {})
                    .get("teacher_prompt"),
            },
        }
        if done:
            # No further step/check needs the live SeqReactCase; completed
            # states + snapshots used to accumulate for the whole run (F09).
            self._state.pop(key, None)
            if (os.environ.get("KGQA_SESSION_DRAIN_CLOSE", "0") == "1"
                    and not self._state and self._session is not None):
                # Last active trajectory of this batch, still inside the
                # driver loop: close the session the proper way (await) so
                # connector protocols are closed before the loop dies
                # (audit R4-03). Server mode keeps sessions across
                # concurrent requests and leaves this off.
                await self.close_session()
        return result

    def set_assistant_message_token_ids(self, infer_request, token_ids):
        # The driver writes the accumulated LAST turn's ids into the newest
        # assistant message. When the current turn was empty (stop-only), the
        # driver passes the PREVIOUS turn's ids — writing them into the empty
        # assistant corrupts history (audit R1). Only freeze ids that belong
        # to the current turn of this request.
        st = self._state.get(getattr(infer_request, "uuid", None))
        if st is not None and st.get("last_ids") != list(token_ids or []):
            return
        return super().set_assistant_message_token_ids(infer_request, token_ids)

    # ---- 每轮：下一轮消息（sync） --------------------------------------
    def step(self, infer_request, response_choice, current_turn):
        st = self._state.get(getattr(infer_request, "uuid", None))
        if st is None:
            return {"infer_request": infer_request}
        rc = st["rc"]
        if os.environ.get("KGQA_DEBUG") and current_turn <= 1:
            asst = [m for m in rc.messages if m.get("role") == "assistant"]
            if asst:
                _dbg(f"hist case={st['case_id']} turn={current_turn} "
                     f"think_in_hist={'</think>' in (asst[-1].get('content') or '')} "
                     f"reasoning_field={bool(asst[-1].get('reasoning'))}")
        rc.ctx._seq_turns_remaining = max(self.max_rounds - current_turn, 0)
        hint = rc.state_aware_hint() + rc.allowed_tools_hint()
        if getattr(rc, "_tool_repeat", 0) >= 2:
            hint = rc.loop_nudge() + "\n" + hint
        infer_request.messages = list(rc.messages) + [
            {"role": "user", "content": hint}]
        out = {"infer_request": infer_request}
        # TITO 掩码回传（2026-10-25）: driver 只有看到 step_result 里的
        # response_token_ids/response_loss_mask 才会用它们覆盖默认的全 1 掩码,
        # 否则 on_turn_end 算好的"只监督动作段"根本到不了训练侧。
        if st.get("last_ids") and st.get("last_mask") is not None:
            out["response_token_ids"] = list(st["last_ids"])
            out["response_loss_mask"] = list(st["last_mask"])
        return out

    def check_finished(self, infer_request, response_choice, current_turn):
        st = self._state.get(getattr(infer_request, "uuid", None))
        if st is None:
            return super().check_finished(infer_request, response_choice, current_turn)
        # Context guard: stop BEFORE the next turn's prompt is encoded, so the
        # engine does not raise MaxLengthError mid-rollout.
        rc = st["rc"]
        if self.max_model_len and not (rc.done or rc.failed):
            try:
                # cheap estimate: ~4.5 chars/token (RC2 system 40.2k chars /
                # 8.2k tokens); avoids tokenizing the full history each turn
                plen = int(sum(len(str(m.get("content") or ""))
                               for m in rc.messages) / 4.5)
                if plen + 2048 > self.max_model_len:
                    rc.failed = True
                    rc.failure_reason = "context_limit"
                    st["discard"] = True
                    st["too_long"] = True
                    _dbg(f"context_limit case={st['case_id']} turn={current_turn} "
                         f"prompt~{plen} limit={self.max_model_len}")
                    return True
            except Exception:
                pass
        rc = st["rc"]
        return bool(rc.done or rc.failed or current_turn >= self.max_rounds)


multi_turns["kgqa_opsd"] = KGQAScheduler


# ---- 调试探针（KGQA_DEBUG=1 时打印 rollout 入口与 scheduler 状态）----------
def _install_debug_probes() -> None:
    if not os.environ.get("KGQA_DEBUG"):
        return
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
    except Exception:
        return
    cls = getattr(RM, "RolloutTrainerMixin", None)
    if cls is None:
        return
    for name in ("prepare_rollout", "_prepare_scheduler", "_generate_completions",
                 "_fast_infer", "_infer_single_or_multi_turn"):
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_kgqa_probe", False):
            continue

        def make(orig, name):
            def wrapper(self, *a, **k):
                try:
                    sched = type(getattr(self, "multi_turn_scheduler", None)).__name__
                    srv = getattr(self, "enable_server_multi_turn", None)
                    print(f"[kgqa-dbg] {name}: sched={sched} server_mt={srv}",
                          file=sys.stderr, flush=True)
                except Exception:
                    pass
                return orig(self, *a, **k)
            wrapper._kgqa_probe = True
            return wrapper

        setattr(cls, name, make(orig, name))


_install_debug_probes()


# ---- 训练相位探针（KGQA_DEBUG=1，实例级，避免 import 顺序问题）------------
def _install_instance_probes(trainer) -> None:
    if not os.environ.get("KGQA_DEBUG"):
        return
    names = ("_prepare_inputs", "_rollout_samples", "_generate_completions",
             "_prepare_batch_inputs", "_encode_samples", "_postprocess_batch",
             "training_step", "_postprocess_rollout_outputs")
    for name in names:
        orig = getattr(trainer, name, None)
        if orig is None or getattr(orig, "_kgqa_t", False):
            continue

        def make(orig, name):
            def wrapper(*a, **k):
                t0 = _time.monotonic()
                r = orig(*a, **k)
                dt = _time.monotonic() - t0
                if dt >= 0.5:
                    _dbg(f"T {name} {dt:.1f}s")
                return r
            wrapper._kgqa_t = True
            return wrapper
        setattr(trainer, name, make(orig, name))


def _add_stop(tmpl) -> None:
    """vLLM must stop on the model's own turn terminator.

    Qwen3.5 emits ``<|im_end|>`` (id 248044) at the end of a turn, but
    ms-swift aligns the tokenizer eos to ``<|endoftext|>`` (248046), so the
    engine keeps generating past ``<|im_end|>`` and the marker lands in
    content (observed: action truncated mid-sentence at ``[actor:<|im_end|>``).
    """
    meta = getattr(tmpl, "template_meta", None)
    if meta is None:
        return
    words = list(getattr(meta, "stop_words", None) or [])
    for token in ("<|im_end|>", "<|endoftext|>"):
        if token not in words:
            words.append(token)
    meta.stop_words = words


# ---- 空 rank 参与 collective（跨 rank 补采/训练的前提）----------------------
def _install_empty_rank_collective_fix() -> None:
    """Make an empty rank enter the same collectives as a non-empty one.

    `_add_prompt_id_to_inputs` returned before `gather_object` when samples was
    empty (audit F03). Dynamic resampling deliberately calls the rollout with
    an empty list on one rank; that rank then skipped the gather while the
    other entered it, so the collective order diverged. Call the same
    gather with an empty contribution before returning.
    """
    if os.environ.get("KGQA_FIX_EMPTY_RANK", "1") != "1":
        return
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
    except Exception:
        return
    cls = getattr(RM, "RolloutTrainerMixin", None)
    if cls is None or getattr(cls._add_prompt_id_to_inputs, "_kgqa_empty_rank", False):
        return
    orig = cls._add_prompt_id_to_inputs

    def _add_prompt_id_to_inputs(self, samples):
        if not samples:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    from accelerate.utils import gather_object
                    gather_object([])
            except Exception:
                pass
            return samples
        return orig(self, samples)

    _add_prompt_id_to_inputs._kgqa_empty_rank = True
    cls._add_prompt_id_to_inputs = _add_prompt_id_to_inputs


_install_empty_rank_collective_fix()


# ---- 按 case 分组取样（动态采样的前提）------------------------------------
def _install_grouped_sampler() -> None:
    """Keep the 8 rows of a case in the same generation batch.

    ms-swift's GKD sampler shuffles individual rows (``self.shuffle_dataset``
    defaults True). With 32 rows/step that yields ~22 cases with 1-2 samples
    each, so "all-wrong / all-correct / variance" is decided on noise. The
    training jsonl is written as contiguous 8-row case blocks (32 cases × 8);
    disabling row shuffle makes each generation batch 4 complete cases (or
    all 32 cases when the batch is 256). Set KGQA_GROUP_BY_CASE=1 to keep the
    grouping contract (default), 0 for the old row-shuffled behaviour.
    """
    if os.environ.get("KGQA_GROUP_BY_CASE", "1") != "1":
        return
    try:
        from swift.rlhf_trainers.gkd_trainer import GKDTrainer
    except Exception:
        return
    if getattr(GKDTrainer._get_train_sampler, "_kgqa_group", False):
        return
    orig = GKDTrainer._get_train_sampler

    def _get_train_sampler(self, train_dataset=None):
        self.shuffle_dataset = False
        if hasattr(self.args, "dataset_shuffle"):
            self.args.dataset_shuffle = False
        sampler = orig(self, train_dataset)
        _dbg("grouped sampler: shuffle_dataset=False (contiguous 8-row case blocks)")
        return sampler

    _get_train_sampler._kgqa_group = True
    GKDTrainer._get_train_sampler = _get_train_sampler


_install_grouped_sampler()


# ---- 强制安装 scheduler（trainer args 缺 multi_turn_scheduler 字段）--------
def _install_scheduler_override() -> None:
    """`self.args` 是 GKDConfig，没有 multi_turn_scheduler 字段，原生
    `_prepare_scheduler` 会静默返回 None。这里在 orig 之后补装。"""
    if os.environ.get("KGQA_SCHEDULER", "") != "kgqa_opsd":
        return
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
    except Exception:
        return
    cls = getattr(RM, "RolloutTrainerMixin", None)
    if cls is None or getattr(cls._prepare_scheduler, "_kgqa_forced", False):
        return
    orig = cls._prepare_scheduler

    def _prepare_scheduler(self):
        orig(self)
        # GKDConfig 缺 max_turns; _colocate_multi_turn_infer 会直接读
        if not hasattr(self.args, "max_turns"):
            setattr(self.args, "max_turns",
                    int(os.environ.get("MAX_ROUNDS", "16")))
        # 用户裁定 2026-10-22: 轨迹保留 Thinking（有思考预算）。
        # ms-swift 默认给每个 assistant 生成轮注入空思考块
        # (`<think>\n\n</think>\n\n`, add_non_thinking_prefix) 并在 infer
        # 模式删除历史 thinking (preserve_thinking=False)。两者都会让
        # rollout 实际不思考 + prefix cache 失效，这里显式关掉/打开。
        _rt = os.environ.get("KGQA_ROLLOUT_TEMP", "").strip()
        if _rt and getattr(self, "request_config", None) is not None:
            self.request_config.temperature = float(_rt)
            _dbg(f"rollout sampling temperature={_rt}")
        # 生产 9B 批的采样参数（kgqa/llm/client.py + seq_rollout.main）:
        # top_p=0.8 / top_k=20 / presence=1.5。只覆盖 temperature 会让
        # 训练 rollout 实际走 vLLM 默认 (top_p=1, top_k=-1, presence=0)，
        # 与测试环境不是同一个分布。
        rc = getattr(self, "request_config", None)
        if rc is not None:
            for env, attr, cast in (
                    ("KGQA_ROLLOUT_TOP_P", "top_p", float),
                    ("KGQA_ROLLOUT_TOP_K", "top_k", int),
                    ("KGQA_ROLLOUT_PRESENCE", "presence_penalty", float),
                    ("KGQA_ROLLOUT_REPETITION", "repetition_penalty", float)):
                val = os.environ.get(env, "").strip()
                if val:
                    setattr(rc, attr, cast(val))
            _rs = os.environ.get("KGQA_ROLLOUT_SEED", "").strip()
            if _rs:
                rc.seed = int(_rs)
            _dbg(f"rollout sampling top_p={getattr(rc, 'top_p', None)} "
                 f"top_k={getattr(rc, 'top_k', None)} "
                 f"presence={getattr(rc, 'presence_penalty', None)} "
                 f"seed={getattr(rc, 'seed', None)}")
        if os.environ.get("KGQA_PRESERVE_THINKING", "1") == "1":
            tmpl = getattr(self, "template", None)
            if tmpl is not None:
                tmpl.preserve_thinking = True
                tmpl.add_non_thinking_prefix = False
                tmpl.enable_thinking = True
                _add_stop(tmpl)
                _dbg("template: preserve_thinking=True "
                     "add_non_thinking_prefix=False enable_thinking=True")
        if getattr(self, "multi_turn_scheduler", None) is None:
            self.multi_turn_scheduler = KGQAScheduler(
                max_turns=getattr(self.args, "max_turns", None),
                tokenizer=getattr(self, "processing_class", None),
                template=getattr(self, "template", None))
            _dbg("forced scheduler installed (args 无 multi_turn_scheduler 字段)")
        _install_instance_probes(self)

    _prepare_scheduler._kgqa_forced = True
    cls._prepare_scheduler = _prepare_scheduler


_install_scheduler_override()


# ---- server 模式: rollout 服务进程的模板也要开 thinking -------------------
def _install_rollout_template_patch() -> None:
    if os.environ.get("KGQA_PRESERVE_THINKING", "1") != "1":
        return
    try:
        from swift.pipelines.infer import rollout as RO
    except Exception:
        return
    orig = getattr(RO, "get_rollout_engine_type", None)
    if orig is None or getattr(orig, "_kgqa_think", False):
        return

    def get_rollout_engine_type(args, engine, *a, **k):
        res = orig(args, engine, *a, **k)
        tmpl = getattr(engine, "template", None)
        if tmpl is not None:
            tmpl.preserve_thinking = True
            tmpl.add_non_thinking_prefix = False
            tmpl.enable_thinking = True
            _add_stop(tmpl)
            _dbg("server template: preserve_thinking=True "
                 "add_non_thinking_prefix=False enable_thinking=True")
        return res

    get_rollout_engine_type._kgqa_think = True
    RO.get_rollout_engine_type = get_rollout_engine_type


_install_rollout_template_patch()


# ---- 思考预算（vLLM 原生 thinking_token_budget）----------------------------
def _install_thinking_budget() -> None:
    """按本地链同款机制限制每轮思考 token。

    vLLM 0.24 的 thinking_token_budget 需要 engine 级 reasoning_config
    (start/end 边界)；超预算时 vLLM 强制注入 reasoning_end_str，模型随即
    转入 action。没有它，Qwen3.5 会一直想到 max_completion 被截断，
    动作永远不出现（实测 1536 全被思考吃满）。
    """
    raw = os.environ.get("KGQA_THINKING_BUDGET", "1000").strip()
    if not raw or raw == "0":
        return
    try:
        budget = int(raw)
        from vllm.config import ReasoningConfig
        from swift.infer_engine.vllm_engine import VllmEngine
    except Exception:
        return
    if getattr(VllmEngine._prepare_engine_kwargs, "_kgqa_tb", False):
        return

    orig_ek = VllmEngine._prepare_engine_kwargs

    def _prepare_engine_kwargs(self, max_model_len, engine_kwargs):
        ek = dict(engine_kwargs or {})
        ek.setdefault('reasoning_config', ReasoningConfig(
            reasoning_start_str='<think>',
            reasoning_end_str=('I will now emit the tool call based on the '
                               'reasoning above.</think>')))
        return orig_ek(self, max_model_len, ek)

    _prepare_engine_kwargs._kgqa_tb = True
    VllmEngine._prepare_engine_kwargs = _prepare_engine_kwargs

    orig_gc = VllmEngine._prepare_generation_config

    def _prepare_generation_config(self, request_config):
        res = orig_gc(self, request_config)
        try:
            res.thinking_token_budget = budget
        except Exception:
            pass
        return res

    _prepare_generation_config._kgqa_tb = True
    VllmEngine._prepare_generation_config = _prepare_generation_config


_install_thinking_budget()


# ---- 按轮拆分（用户方案 2026-10-22）：一条轨迹 → N 个单轮样本 --------------
# 动机: ms-swift OPSD 的 teacher 编码与多轮 student 编码在模板层(think 块/
# 特殊 token)插入不对称 → Label count mismatch (student=470, teacher=494)。
# 单轮 OPSD 已验证全绿; 拆分后每个样本在结构上与单轮完全一致, 且每个轮次
# 复用同一条完整 teacher 条件(案级 teacher_prompt, 正典 OPSD: teacher 只看
# 精炼参考, 不看真实轨迹)。
def _install_turn_splitter() -> None:
    if os.environ.get("KGQA_TURN_SPLIT", "0") != "1":  # 默认单轨迹(TITO 修好后不再需要拆分)
        return
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
    except Exception:
        return
    cls = getattr(RM, "RolloutTrainerMixin", None)
    if cls is None or getattr(cls._postprocess_rollout_outputs, "_kgqa_split", False):
        return
    import copy as _copy
    orig = cls._postprocess_rollout_outputs

    def _postprocess_rollout_outputs(self, samples, outputs):
        results = orig(self, samples, outputs)
        expanded = []
        n_split = 0
        for s in results:
            turns = (s.rollout_infos or {}).pop("kgqa_turns", None)
            if not turns:
                expanded.append(s)
                continue
            tp = (s.rollout_infos or {}).pop("kgqa_teacher_prompt", None)
            # 案级正典 teacher 条件: system + question + teacher_prompt
            head = [m for m in turns[0]["messages_before"][:2]
                    if m.get("role") in ("system", "user")]
            for t in turns:
                if not t["ids"]:
                    continue
                c = _copy.deepcopy(s)
                # student: 该轮生成前的完整会话 + assistant 占位（单轮形态）
                c.messages = _copy.deepcopy(t["messages_before"]) + [
                    {"role": "assistant", "content": t["text"]}]
                c.response_token_ids = list(t["ids"])
                c.response_loss_mask = [1] * len(t["ids"])
                c.rollout_logprobs = []
                c.finish_reason = "stop"
                # teacher: 复用同一条完整参考（正典 OPSD 条件）
                if tp:
                    c.teacher_prompt = tp
                    c.teacher_messages = _copy.deepcopy(head) + [
                        {"role": "user", "content": tp},
                        {"role": "assistant", "content": t["text"]}]
                expanded.append(c)
                n_split += 1
        if os.environ.get("KGQA_DEBUG") and n_split:
            _dbg(f"turn-split: {len(results)} trajectories -> {len(expanded)} samples")
        return expanded

    _postprocess_rollout_outputs._kgqa_split = True
    cls._postprocess_rollout_outputs = _postprocess_rollout_outputs


_install_turn_splitter()


# ---- teacher 前缀对齐：精炼轨迹进教师侧 system，学生侧保持原始轨迹
def _install_teacher_prefix() -> None:
    """OPSD prefix alignment.

    ms-swift's default ``build_teacher_view`` REPLACES the last user message
    with ``teacher_prompt``; for multi-turn samples that only conditions the
    final turn. The intended comparison is: student prefix = original
    trajectory; teacher prefix = original trajectory PLUS the verified refined
    experience/answer; both score the SAME original response tokens. ``system``
    mode merges the refined reference into the teacher-side system prefix so
    every response token is conditioned on it, while the student keeps the
    original prefix. Set ``KGQA_TEACHER_PREFIX=replace`` to restore ms-swift's
    last-user substitution.
    """
    mode = os.environ.get("KGQA_TEACHER_PREFIX", "system").strip().lower()
    if mode == "replace":
        return
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
    except Exception:
        return
    cls = getattr(RM, "RolloutTrainerMixin", None)
    if cls is None or getattr(cls._postprocess_rollout_outputs, "_kgqa_tprefix", False):
        return
    orig = cls._postprocess_rollout_outputs

    def _postprocess_rollout_outputs(self, samples, outputs):
        res = orig(self, samples, outputs)
        n = 0
        for s in res:
            tp = getattr(s, "teacher_prompt", None)
            if not tp:
                continue
            msgs = [dict(m) for m in (getattr(s, "messages", None) or [])]
            if mode == "append":
                # Teacher prefix = original messages + one user transition
                # carrying the reference solution; the original rollout stays
                # the formal response being scored.
                msgs.append({"role": "user", "content": str(tp)})
            else:
                block = "\n\n" + str(tp) + "\n"
                idx = next((i for i, m in enumerate(msgs)
                            if m.get("role") == "system"), None)
                if idx is None:
                    msgs.insert(0, {"role": "system", "content": block})
                else:
                    msgs[idx] = {"role": "system",
                                 "content": str(msgs[idx].get("content") or "") + block}
            s.teacher_messages = msgs
            n += 1
        if n and os.environ.get("KGQA_DEBUG"):
            _dbg(f"teacher prefix=system for {n} samples "
                 f"(ref head={str(res[0].teacher_prompt)[:60]!r})")
        return res

    _postprocess_rollout_outputs._kgqa_tprefix = True
    cls._postprocess_rollout_outputs = _postprocess_rollout_outputs


_install_teacher_prefix()


# ---- teacher 视图透传 loss_mask（解除 mask 模式的框架阻塞）------------------
def _install_teacher_mask_passthrough() -> None:
    """``OnPolicySample.to_teacher_template_dict`` drops ``response_loss_mask``,
    so ``encode_teacher_view`` rebuilds teacher labels as all-active — zeroing
    the student mask then trips the label-count assertion. Pass the mask
    through so teacher labels respect it (all-ones default is unchanged).
    """
    try:
        from swift.rl_core.data import OnPolicySample
    except Exception:
        return
    if getattr(OnPolicySample.to_teacher_template_dict, "_kgqa_tmask", False):
        return
    orig = OnPolicySample.to_teacher_template_dict

    def to_teacher_template_dict(self):
        d = orig(self)
        if self.response_loss_mask:
            d['response_loss_mask'] = self.response_loss_mask
        return d

    to_teacher_template_dict._kgqa_tmask = True
    OnPolicySample.to_teacher_template_dict = to_teacher_template_dict


_install_teacher_mask_passthrough()


# ---- 空槽归一（独立于 teacher-prefix 模式，audit R4-02）--------------------
def _install_empty_slot_normalizer() -> None:
    """Drop empty response id slots on BOTH backends, for every teacher mode.

    The server driver keeps `[]` outer entries for zero-token turns while
    colocate drops them, and on_turn_end removed the matching empty assistant
    messages. Dropping empty slots keeps assistant count == non-empty slot
    count on both backends. This used to live inside the teacher-prefix
    wrapper, so KGQA_TEACHER_PREFIX=replace skipped it (audit R4-02).
    """
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
    except Exception:
        return
    cls = getattr(RM, "RolloutTrainerMixin", None)
    if cls is None or getattr(cls._postprocess_rollout_outputs,
                              "_kgqa_slots", False):
        return
    orig = cls._postprocess_rollout_outputs

    def _postprocess_rollout_outputs(self, samples, outputs):
        res = orig(self, samples, outputs)
        for s in res:
            rti = list(getattr(s, "response_token_ids", None) or [])
            if rti and any(not ids for ids in rti):
                rlm = list(getattr(s, "response_loss_mask", None) or [])
                pairs = [(ids, (rlm[i] if i < len(rlm) else [1] * len(ids)))
                         for i, ids in enumerate(rti) if ids]
                s.response_token_ids = [p[0] for p in pairs]
                s.response_loss_mask = [p[1] for p in pairs]
        return res

    _postprocess_rollout_outputs._kgqa_slots = True
    cls._postprocess_rollout_outputs = _postprocess_rollout_outputs


_install_empty_slot_normalizer()


# ---- teacher/student 同轨迹断言（非位置对齐）-------------------------------
def _install_alignment_check() -> None:
    """Verify teacher and student score the SAME response token sequence.

    ms-swift already asserts equal label *counts*; this checks the stronger
    property we rely on: the non-masked labels (the response token ids) are
    element-wise identical in order. If they match, pairing by each side's
    mask order is pairing the same original trajectory token-by-token, even
    when the teacher prompt shifts every absolute position. A mismatch means
    the teacher view was re-tokenized/expanded, so the JSD pairs would be
    meaningless.
    """
    if os.environ.get("KGQA_ALIGN_CHECK", "1") != "1":
        return
    try:
        import torch
        from swift.rlhf_trainers.gkd_trainer import GKDTrainer
    except Exception:
        return
    if getattr(GKDTrainer._compute_jsd_loss, "_kgqa_align", False):
        return
    orig = GKDTrainer._compute_jsd_loss
    calls = {"n": 0}

    def _compute_jsd_loss(self, student_logits, teacher_output, labels):
        calls["n"] += 1
        t_labels = getattr(teacher_output, "labels", None)
        if t_labels is not None and calls["n"] <= 3 or (
                t_labels is not None and calls["n"] % 50 == 0):
            try:
                s_ids = labels.reshape(-1)
                t_ids = t_labels.reshape(-1)
                s_val = s_ids[s_ids != -100]
                t_val = t_ids[t_ids != -100]
                same_count = int(s_val.numel()) == int(t_val.numel())
                same_ids = bool(same_count and torch.equal(s_val, t_val))
                step = getattr(getattr(self, "state", None), "global_step", "?")
                head = s_val[:8].tolist() if s_val.numel() else []
                print(f"[kgqa-align] step={step} student={int(s_val.numel())} "
                      f"teacher={int(t_val.numel())} same_ids={same_ids} "
                      f"head={head}", file=sys.stderr, flush=True)
                if not same_ids:
                    raise RuntimeError(
                        "teacher/student response token ids differ: "
                        f"student={int(s_val.numel())} teacher={int(t_val.numel())} "
                        f"student_head={head[:16]} teacher_head={t_val[:16].tolist()} "
                        "— teacher view must reuse the original sampled trajectory "
                        "token ids (TITO), not re-tokenize a rewritten trajectory.")
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"[kgqa-align] check failed: {exc}", file=sys.stderr, flush=True)
        return orig(self, student_logits, teacher_output, labels)

    _compute_jsd_loss._kgqa_align = True
    GKDTrainer._compute_jsd_loss = _compute_jsd_loss


_install_alignment_check()


# ---- teacher 权重守卫（独立于 KGQA_ALIGN_CHECK，audit R5）------------------
def _install_lora_plus() -> None:
    """LoRA+ (KGQA_LORA_PLUS=<ratio>): lora_B learns at ratio x the base lr.

    swift 4.6.0 has no lora_plus_ratio knob, so we split the optimizer's
    param groups ourselves: lora_B params move to a group with lr*ratio while
    every other hyperparameter is preserved. PyTorch LR schedulers keep
    per-group base_lrs, so warmup/decay scale both groups consistently.
    """
    ratio = float(os.environ.get("KGQA_LORA_PLUS", "0") or 0)
    if ratio <= 0:
        return
    try:
        import transformers
    except Exception:  # noqa: BLE001
        return
    orig = transformers.Trainer.create_optimizer
    if getattr(orig, "_kgqa_lora_plus", False):
        return

    def create_optimizer(self):
        opt = orig(self)
        try:
            names = {id(p): n for n, p in self.model.named_parameters()}
            groups, scaled = [], 0
            for g in opt.param_groups:
                base, hi = [], []
                for prm in g["params"]:
                    nm = names.get(id(prm), "")
                    (hi if ("lora_B" in nm or "lora_b" in nm) else base).append(prm)
                if not hi:
                    groups.append(g)
                    continue
                gh = {k: v for k, v in g.items() if k != "params"}
                gh["params"] = hi
                gh["lr"] = float(g["lr"]) * ratio
                if base:
                    gb = {k: v for k, v in g.items() if k != "params"}
                    gb["params"] = base
                    groups.append(gb)
                groups.append(gh)
                scaled += len(hi)
            opt.param_groups[:] = groups
            print(f"[kgqa-lora+] lora_B params={scaled} lr x{ratio} "
                  f"({len(groups)} groups)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[kgqa-lora+] patch failed: {exc}", flush=True)
        return opt

    create_optimizer._kgqa_lora_plus = True
    transformers.Trainer.create_optimizer = create_optimizer


_install_lora_plus()


# ---- F01: LM head 前分块 loss（不物化 [S,V] logits）------------------------
def _install_chunked_loss() -> None:
    """Chunked LM-head GKD/JSD: backbone hidden -> active rows -> per-chunk head.

    Verified by scripts/diag_chunked_lmhead.py on Qwen3.5-4B (V=248,320):
    baseline vs chunked |Δloss|=2.3e-10 / |Δgrad|rel=1.7e-05, peak 33.4G ->
    14.5G at 16k (2.30x), 30k runs at 19.9G where the baseline cannot even
    allocate one [S,V] logits tensor. Chunk head+loss runs under
    torch.utils.checkpoint (non-reentrant) so [C,V] activations are not held
    across chunks (a plain `for chunk: total += loss` keeps them all).
    """
    if os.environ.get("KGQA_CHUNKED_LOSS", "0") != "1":
        return
    try:
        import torch
        import torch.nn.functional as F
        from swift.rlhf_trainers.gkd_trainer import GKDTrainer
        from swift.trainers import disable_gradient_checkpointing
    except Exception as exc:  # noqa: BLE001
        # Explicitly requested but uninstallable: hard-fail (audit C05). A
        # silent fallback keeps the full-logits path that OOMs at 256×32k.
        raise RuntimeError(
            f"KGQA_CHUNKED_LOSS=1 but chunked loss cannot be installed: {exc}")
    if getattr(GKDTrainer.compute_loss, "_kgqa_chunked", False):
        return
    orig = GKDTrainer.compute_loss
    chunk = int(os.environ.get("KGQA_LOSS_CHUNK", "512") or 512)
    stats = {"n": 0}
    print(f"[kgqa-chunked] installed: chunk={chunk} "
          f"(wraps {getattr(orig, '__qualname__', orig)})", flush=True)

    def _active_flat(labels):
        """(flat positions into [B,S-1] hidden, supervised token ids).

        Prediction at position p uses label[p+1]; positions are flattened
        row-major so B>1 micro-batches work (audit C03).
        """
        mask = labels[:, 1:] != -100
        rows, pos = mask.nonzero(as_tuple=True)
        flat = rows * (labels.shape[1] - 1) + pos
        sup = labels[rows, pos + 1]
        return flat, sup

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        if self.use_liger_gkd_loss or self.use_teacher_api \
                or (getattr(self.args, 'sft_alpha', 0) or 0) > 0:
            return orig(self, model, inputs, return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch)
        # Shared pre-checks that the old chain ran for us (audit C01): the
        # fast path bypasses orig, so the teacher-policy guard and the TITO
        # token-id check must live here too.
        if os.environ.get("KGQA_TEACHER_SAME_WEIGHTS", "1") == "1":
            self_distill = bool(getattr(self, "_is_self_distillation", False))
            disable_adapter = bool(
                getattr(self, "_teacher_use_disable_adapter", False))
            if not self_distill or disable_adapter:
                raise RuntimeError(
                    "teacher is not the student policy: "
                    f"self_distillation={self_distill} "
                    f"disable_adapter={disable_adapter} (chunked fast path)")
        model_inputs = inputs['model_inputs']
        teacher_model_inputs = inputs['teacher_model_inputs']
        unwrapped = self.accelerator.unwrap_model(model)
        base_model = (unwrapped.get_base_model()
                      if hasattr(unwrapped, "get_base_model") else unwrapped)
        dec = base_model.get_decoder()
        W = base_model.get_output_embeddings().weight

        def _dec_fwd(inputs_dict):
            keep = ("input_ids", "attention_mask", "position_ids")
            kw = {k: v for k, v in inputs_dict.items() if k in keep}
            kw["use_cache"] = False
            return dec(**kw).last_hidden_state

        s_labels = model_inputs['labels']
        t_labels = teacher_model_inputs['labels']
        idx_s, sup_s = _active_flat(s_labels)
        A = int(idx_s.numel())
        idx_t, sup_t = _active_flat(t_labels)
        if A != int(idx_t.numel()):
            raise RuntimeError(
                f"chunked loss active mismatch student={A} "
                f"teacher={int(idx_t.numel())}")
        if A and os.environ.get("KGQA_ALIGN_CHECK", "1") == "1" \
                and (stats["n"] <= 3 or stats["n"] % 50 == 0):
            if not bool(torch.equal(sup_s, sup_t)):
                raise RuntimeError(
                    "chunked TITO check: supervised token ids differ "
                    f"(student head={sup_s[:8].tolist()} "
                    f"teacher head={sup_t[:8].tolist()})")

        _phase_begin("loss_student_fwd")
        hs = _dec_fwd(model_inputs)
        _phase_end("loss_student_fwd")
        h_s = hs[:, :-1, :].reshape(-1, hs.shape[-1]).index_select(0, idx_s)
        _phase_begin("loss_teacher_fwd")
        with torch.no_grad(), disable_gradient_checkpointing(
                model, self.args.gradient_checkpointing_kwargs):
            ht = _dec_fwd(teacher_model_inputs)
        _phase_end("loss_teacher_fwd")
        h_t = (ht[:, :-1, :].reshape(-1, ht.shape[-1])
               .index_select(0, idx_t).detach())
        if A == 0:
            # Keep the student graph connected (audit C03) and honor the
            # return_outputs contract.
            loss = hs.sum() * 0.0
            return (loss, None) if return_outputs else loss

        beta = self.beta
        temp = self.temperature
        K = self.gkd_logits_topk

        def _jsd(s_log, t_log):
            # Endpoint contract matches gkd_loss.jsd_loss (audit C04):
            # beta=0 -> KL(t||s), beta=1 -> KL(s||t), else the JSD mix.
            if beta == 0:
                return (t_log.exp() * (t_log - s_log)).sum(-1)
            if beta == 1:
                return (s_log.exp() * (s_log - t_log)).sum(-1)
            log_beta = torch.log(torch.tensor(beta, device=s_log.device,
                                              dtype=s_log.dtype))
            log_1mb = torch.log1p(-torch.tensor(beta, device=s_log.device,
                                                dtype=s_log.dtype))
            log_m = torch.logaddexp(s_log + log_1mb, t_log + log_beta)
            return beta * (t_log.exp() * (t_log - log_m)).sum(-1) \
                + (1 - beta) * (s_log.exp() * (s_log - log_m)).sum(-1)

        if K is None:
            def _chunk_loss(hc_s, hc_t):
                s_l = F.log_softmax((hc_s @ W.t()).float() / temp, dim=-1)
                t_l = F.log_softmax((hc_t @ W.t()).float() / temp, dim=-1)
                return _jsd(s_l, t_l).sum()
        else:
            def _chunk_loss(hc_s, hc_t):
                t_logits = hc_t @ W.t()
                tv, ti = t_logits.topk(K, dim=-1)
                s_g = (hc_s @ W.t()).gather(-1, ti).float()
                s_l = F.log_softmax(s_g / temp, dim=-1)
                t_l = F.log_softmax(tv.float() / temp, dim=-1)
                return _jsd(s_l, t_l).sum()

        total = hs.new_zeros((), dtype=torch.float32)
        _phase_begin("loss_chunk_loop")
        for s in range(0, A, chunk):
            e = min(s + chunk, A)
            total = total + torch.utils.checkpoint.checkpoint(
                _chunk_loss, h_s[s:e], h_t[s:e], use_reentrant=False)
        _phase_end("loss_chunk_loop")
        loss = total / A
        stats["n"] += 1
        if stats["n"] <= 3 or stats["n"] % 100 == 0:
            step = getattr(getattr(self, "state", None), "global_step", "?")
            print(f"[kgqa-chunked] step={step} active={A} chunk={chunk} "
                  f"K={K}", file=sys.stderr, flush=True)
        if return_outputs:
            return loss, None
        return loss

    compute_loss._kgqa_chunked = True
    GKDTrainer.compute_loss = compute_loss


_install_chunked_loss()


def _install_teacher_mode_guard() -> None:
    """Fail BEFORE any forward when the teacher is not the student policy.

    OPSD is self-distillation: teacher and student must share the same weights
    (including the LoRA adapter). ms-swift resolves same-path ``--teacher_model``
    + LoRA into ``disable_adapter()`` (frozen base), which silently shifts the
    target policy. Checking inside compute_loss (pre-forward) avoids running
    the student/teacher forwards first on a wrong configuration.
    """
    if os.environ.get("KGQA_TEACHER_SAME_WEIGHTS", "1") != "1":
        return
    try:
        from swift.rlhf_trainers.gkd_trainer import GKDTrainer
    except Exception:
        return
    if getattr(GKDTrainer.compute_loss, "_kgqa_tguard", False):
        return
    orig = GKDTrainer.compute_loss
    calls = {"n": 0}

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        self_distill = bool(getattr(self, "_is_self_distillation", False))
        disable_adapter = bool(getattr(self, "_teacher_use_disable_adapter", False))
        if not self_distill or disable_adapter:
            raise RuntimeError(
                "teacher is not the student policy: "
                f"self_distillation={self_distill} "
                f"disable_adapter={disable_adapter}. For OPSD self-distillation "
                "omit --teacher_model (same-path LoRA is resolved to the frozen "
                "base by ms-swift); set KGQA_TEACHER_SAME_WEIGHTS=0 only for an "
                "intentional frozen-teacher ablation.")
        calls["n"] += 1
        if calls["n"] <= 3 or calls["n"] % 100 == 0:
            step = getattr(getattr(self, "state", None), "global_step", "?")
            print(f"[kgqa-teacher] step={step} self_distill=True "
                  f"disable_adapter=False teacher_model_none="
                  f"{getattr(self, 'teacher_model', None) is None}",
                  file=sys.stderr, flush=True)
        return orig(self, model, inputs, return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch)

    compute_loss._kgqa_tguard = True
    GKDTrainer.compute_loss = compute_loss


_install_teacher_mode_guard()




# ---- 动态采样：全错 case 移除 + 满轮错轨迹重采样（同位替换，数量不变）----
def _install_dynamic_sampler() -> None:
    """Count-preserving dynamic sampling after rollout.

    User rule: a case with NO variance (all samples wrong, or all samples
    correct) has no contrast to learn from; a trajectory that hit the round
    cap and is empty/wrong is noise. ms-swift requires a fixed batch count
    (dropping breaks encode with ``KeyError: input_ids``), so slots are
    REPLACED:
      - degenerate-case slot -> draw an extra sample of a donor case with
        variance (0 < F1 < 1), preferring a correct/non-bad donor;
      - bad trajectory slot  -> resample the SAME case, up to N tries.
    """
    if os.environ.get("KGQA_DYNAMIC_SAMPLE", "1") != "1":
        return
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
    except Exception:
        return
    cls = getattr(RM, "RolloutTrainerMixin", None)
    if cls is None or getattr(cls._generate_completions, "_kgqa_dyn", False):
        return
    import copy as _copy
    import random as _random
    import uuid as _uuid2

    orig = cls._generate_completions
    tries = int(os.environ.get("KGQA_RESAMPLE_TRIES", "2") or 0)
    # 1 = fresh rollout (server mode / enough GPU headroom); 0 = duplicate a
    # donor's completed trajectory (colocate-safe, no second engine wake).
    resample = os.environ.get("KGQA_DYNAMIC_RESAMPLE", "1") == "1"
    # Zero retry budget means no fresh generation at all (audit F02): the old
    # `rounds=max(tries,1)` still issued one extra rollout.
    fresh_ok = resample and tries > 0
    # KGQA_DYN_MODE=mask (user ruling 2026-10-23): degenerate cases / bad /
    # over-long trajectories keep their batch SLOT but contribute ZERO
    # gradient — response_loss_mask zeroed on both student and teacher views
    # (see _install_teacher_mask_passthrough). No donors, no second rollout,
    # no duplicated gradients; the donor machinery below is skipped entirely.
    dyn_mode = os.environ.get("KGQA_DYN_MODE", "mask").strip().lower()

    def _inf(s):
        return getattr(s, "rollout_infos", None) or {}

    def _correct(s) -> bool:
        return float(_inf(s).get("kgqa_f1", 0) or 0) >= 0.999

    def _usable(s) -> bool:
        # Donor/keep eligibility: a discarded sample (teacher reference pushed
        # it over the training length, or context-limit) must not be copied
        # into another slot (audit R3).
        return not bool(_inf(s).get("kgqa_discard"))

    def _bad(s) -> bool:
        i = _inf(s)
        if bool(i.get("kgqa_discard")) and (bool(i.get("kgqa_too_long"))
                                            or bool(i.get("kgqa_max_rounds"))):
            return True
        return bool(i.get("kgqa_max_rounds")) and (
            bool(i.get("kgqa_answer_empty")) or not _correct(s))

    def _fresh(template, msgs):
        c = _copy.deepcopy(template)
        c.messages = _copy.deepcopy(msgs)
        c.response_token_ids = []
        c.response_loss_mask = []
        c.rollout_infos = {}
        c.finish_reason = None
        c.encoded = None
        rid = f"chatcmpl-{_uuid2.uuid4().hex}"
        c.request_id = rid
        c.uuid = rid
        return c

    def _generate_completions(self, samples):
        global _GUARD_TOK, _STEP_NOW
        if _GUARD_TOK is None:
            # stash the trainer's tokenizer for the on_turn_end context
            # guard BEFORE the rollout — the scheduler instance itself has
            # no trainer reference (exact guard arbitration, 2026-10-24).
            _GUARD_TOK = getattr(self, "processing_class", None)
        # which optimizer step this rollout feeds (global_step is still
        # k-1 while rolling out for step k) → per-step dump subdirectory
        try:
            _STEP_NOW = int(getattr(getattr(self, "state", None),
                                    "global_step", 0)) + 1
        except Exception:  # noqa: BLE001
            pass
        originals = [_copy.deepcopy(s.messages) for s in samples]
        res = orig(self, samples)

        # Global per-case F1s (a case can be split across ranks). The per-sample
        # bad flag rides the SAME collective (audit P1-A): a rank-local
        # `_any_bad` made the donor-records gather conditional on divergent
        # rank-local state, desynchronizing the collective order.
        local = [(_inf(s).get("kgqa_case_id"),
                  float(_inf(s).get("kgqa_f1", 0) or 0),
                  _bad(s)) for s in res]
        world = [local]
        _phase_begin("dyn_f1_gather")
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                world = [None] * dist.get_world_size()
                dist.all_gather_object(world, local)
        except Exception:
            pass
        _phase_end("dyn_f1_gather")
        by_case = {}
        any_bad_global = False
        for part in world:
            if not isinstance(part, list):
                continue
            for rec in part:
                if not (isinstance(rec, (tuple, list)) and len(rec) == 3):
                    continue
                cid, f1, bad = rec
                if cid and isinstance(f1, (int, float)):
                    by_case.setdefault(cid, []).append(float(f1))
                any_bad_global = any_bad_global or bool(bad)
        # "全错" = no trajectory reached F1 1.0 (partial F1 is still wrong and
        # gives no correct reference); "全对" = every trajectory F1 1.0.
        # Only a mixed case carries the contrast this objective needs.
        all_wrong = {cid for cid, v in by_case.items()
                     if v and all(x < 0.999 for x in v)}
        all_correct = {cid for cid, v in by_case.items()
                       if v and all(x >= 0.999 for x in v)}
        variance = {cid for cid, v in by_case.items()
                    if v and any(x >= 0.999 for x in v)
                    and any(x < 0.999 for x in v)}
        degenerate = all_wrong | all_correct

        if dyn_mode == "mask":
            # Case-granular masking (user ruling 2026-10-24): degenerate
            # cases (all-correct / all-wrong) mask ALL their trajectories —
            # masked counts are multiples of the group size. Variance cases
            # are kept WHOLESALE, including their wrong trajectories: the
            # contrast is the signal. The only per-trajectory exception is
            # over-length samples, which are masked AND truncated (below).
            _train_len = int(os.environ.get("KGQA_TRAIN_MAX_LEN", "0") or 0)
            # exact counter (step-10 crash: char/4.5 underestimated an entity-
            # dense row by >10% — est 29.7k passed truncation, encode hit
            # 32801 > 32768). The coarse estimate stays the fast pre-filter;
            # borderline rows get a REAL tokenizer count (module-level
            # _exact_msg_tokens; tok is the trainer's processing_class).
            _tok = getattr(self, "processing_class", None) or _GUARD_TOK

            def _est_tokens(msgs):
                return int(sum(len(str(m.get("content") or ""))
                               for m in msgs) / 4.5) + 3072

            def _exact_tokens(msgs):
                return _exact_msg_tokens(_tok, msgs)

            def _over_limit(msgs):
                """True if the row plausibly exceeds the training length.
                Coarse-estimate hit → True outright; within the 4096-token
                danger band below the limit → exact check decides (the
                coarse estimate can undercount entity-dense tool JSON by
                >10%, which is what crashed step 10). Used for the INITIAL
                too_long decision only — the truncate loop uses _fit_check
                (exact unconditionally): a band-gated condition flips back
                to False after the first dropped turn slides the coarse
                estimate below the band floor, exiting the loop with the
                row still over the real limit."""
                if not _train_len:
                    return False
                est = _est_tokens(msgs)
                if est > _train_len:
                    return True
                if est > _train_len - 4096:
                    exact = _exact_tokens(msgs)
                    if exact is not None and exact > _train_len:
                        return True
                return False

            def _fit_check(msgs):
                """Truncate-loop condition: exact token count when the
                tokenizer is available, coarse estimate otherwise. Never
                band-gated — the loop must keep dropping turns until the
                REAL count fits."""
                if not _train_len:
                    return False
                exact = _exact_tokens(msgs)
                return (exact if exact is not None
                        else _est_tokens(msgs)) > _train_len

            def _truncate_to_fit(smp):
                """Drop trailing (assistant, user) turns until the row fits
                the training length — masked samples still get encoded, so an
                over-long one must not reach template.encode (the step-4
                MaxLengthError). Exact-token driven (see _over_limit): the
                coarse char estimate alone let a 32801-token row through at
                step 10. Aligns to the nearest whole-turn prefix."""
                msgs = list(getattr(smp, "messages", None) or [])
                rti = list(getattr(smp, "response_token_ids", None) or [])
                rlm = list(getattr(smp, "response_loss_mask", None) or [])
                if not msgs or not _train_len:
                    return 0
                dropped = 0
                while _fit_check(msgs) and len(msgs) > 3:
                    # messages end with [..., assistant, user]; drop the pair
                    if msgs[-1].get("role") == "user":
                        msgs.pop()
                    if msgs and msgs[-1].get("role") == "assistant" \
                            and len(msgs) > 3:
                        msgs.pop()
                        dropped += 1
                        if rti:
                            rti.pop()
                            if rlm:
                                rlm.pop()
                    else:
                        break
                smp.messages = msgs
                smp.response_token_ids = rti
                smp.response_loss_mask = rlm
                # teacher_messages was copied from the FULL messages by the
                # teacher-prefix wrapper BEFORE this truncation ran. If it
                # keeps the dropped assistant turns, the teacher encode's
                # reverse id-injection only covers the tail and the leftover
                # TEXT assistant spans become teacher labels — mismatch
                # (student=0 teacher=240, first hit in the 100-step run).
                tmsgs = list(getattr(smp, "teacher_messages", None) or [])
                if tmsgs and dropped:
                    want_asst = sum(1 for m in msgs
                                    if m.get("role") == "assistant")
                    n_asst = sum(1 for m in tmsgs
                                 if m.get("role") == "assistant")
                    while n_asst > want_asst and len(tmsgs) > 2:
                        if tmsgs[-1].get("role") == "user":
                            tmsgs.pop()
                            continue
                        if tmsgs[-1].get("role") == "assistant":
                            tmsgs.pop()
                            n_asst -= 1
                            continue
                        break
                    smp.teacher_messages = tmsgs
                return dropped

            masked = truncated = masked_turns = 0
            for smp in res:
                inf = _inf(smp)
                cid = inf.get("kgqa_case_id")
                _artifact_turns = inf.get("kgqa_artifact_turns") or []
                too_long = bool(inf.get("kgqa_too_long")) or _over_limit(
                    getattr(smp, "messages", None) or [])
                if too_long:
                    d = _truncate_to_fit(smp)
                    truncated += 1 if d else 0
                    _dbg(f"truncated case={cid} turns_dropped={d}")
                # 伪影轮(预算注入)逐轮置零: 这些 token 是 vLLM 注入的, 不是
                # 模型行为, 训练它们等于教模型复现注入串。
                if _artifact_turns:
                    rlm = getattr(smp, "response_loss_mask", None) or []
                    for _t in _artifact_turns:
                        _k = int(_t) - 1
                        if 0 <= _k < len(rlm) and rlm[_k]:
                            rlm[_k] = [0] * len(rlm[_k])
                            masked_turns += 1
                    smp.response_loss_mask = rlm
                if cid in degenerate or too_long:
                    rlm = getattr(smp, "response_loss_mask", None) or []
                    rti = getattr(smp, "response_token_ids", None) or []
                    if len(rlm) == len(rti):
                        smp.response_loss_mask = [
                            [0] * len(turn) for turn in rlm]
                        inf["kgqa_masked"] = True
                        masked += 1
            n_masked = masked
            try:
                import torch
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    t = torch.tensor([masked])
                    dist.all_reduce(t)
                    n_masked = int(t.item())
            except Exception:
                pass
            # 复读指标（早警）: 逐字重复轮占比 + 重复≥5轮的轨迹数
            _dup_turns = _dup_trajs = _all_turns = 0
            for _smp in res:
                _meta = (_inf(_smp) or {}).get("kgqa_turn_meta") or []
                _hs = [str(m.get("raw_head") or "") for m in _meta]
                _all_turns += len(_hs)
                _d = sum(1 for _i in range(1, len(_hs))
                         if _hs[_i] == _hs[_i - 1] and _hs[_i])
                _dup_turns += _d
                if _d >= 5:
                    _dup_trajs += 1
            _MASK_STATE.update(
                masked=n_masked, degenerate_cases=len(degenerate),
                truncated=truncated, total=len(res), rep_trajs=_dup_trajs,
                artifact_turns_masked=masked_turns,
                rep_turn_pct=round(100.0 * _dup_turns / max(_all_turns, 1), 1))
            print(f"[kgqa] dyn_mode=mask: masked {masked} local "
                  f"({n_masked} global) of {len(res)} slots; "
                  f"degenerate_cases={len(degenerate)} "
                  f"truncated={truncated} artifact_turns_masked={masked_turns}",
                  flush=True)
            return res

        # Degenerate cases and variance cases can live on DIFFERENT ranks
        # (16 samples/rank = 2 complete cases). A rank-local donor list would
        # leave an all-correct rank with donors=0 and silently keep the case.
        # Gather donor TEMPLATES (initial messages, no rollout state) so every
        # rank can resample any variance case on its own engine.
        donors = []
        flat = []
        # `_any_bad` must be the GLOBAL flag (audit P1-A): with the rank-local
        # value, one rank entered this gather while the other went straight to
        # the needs gather — desynchronized collectives.
        if (degenerate and variance) or any_bad_global:
            def _donor_record(s, msgs):
                # Keep the COMPLETED sample too: duplicate mode reuses its
                # trajectory (no second vLLM rollout); resample mode calls
                # _fresh() which clears the response fields.
                return (_inf(s).get("kgqa_case_id"), s, _copy.deepcopy(msgs))

            _phase_begin("dyn_donor_pack")
            local_records = [_donor_record(s, originals[j])
                             for j, s in enumerate(res)]
            if os.environ.get("KGQA_PHASE_PROFILE") == "1":
                import pickle
                _PHASE_BYTES = len(pickle.dumps(local_records, -1))
            _phase_end("dyn_donor_pack")
            records = [local_records]
            _phase_begin("dyn_donor_gather")
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    records = [None] * dist.get_world_size()
                    dist.all_gather_object(records, local_records)
            except Exception:
                records = [local_records]
            _phase_end("dyn_donor_gather")
            if os.environ.get("KGQA_PHASE_PROFILE") == "1" \
                    and os.environ.get("RANK", "0") == "0":
                print(f"[dyn-size] rank0 donor payload "
                      f"{_PHASE_BYTES / 1e6:.1f}MB x "
                      f"{len(local_records)} records", file=sys.stderr,
                      flush=True)
            flat = [rec for part in records if isinstance(part, list)
                    for rec in part
                    if isinstance(rec, (tuple, list)) and len(rec) == 3]
            flat = [(cid, s, msgs) for cid, s, msgs in flat]
            for cid, s, msgs in flat:
                if cid in variance and not _bad(s) and _usable(s) and _correct(s):
                    donors.append((s, msgs))
            if not donors:
                for cid, s, msgs in flat:
                    if cid in variance and not _bad(s) and _usable(s):
                        donors.append((s, msgs))
            if not donors:
                for cid, s, msgs in flat:
                    if cid not in degenerate and _usable(s):
                        donors.append((s, msgs))

        # Plan every replacement first, then run them in ONE batched rollout.
        # Sequential per-slot resampling was the wall-clock killer: 16 donor
        # rollouts × ~150s = 40min while the other rank waited at the DDP
        # barrier. All requests go to the engine together (vLLM interleaves).
        plans = []
        _phase_begin("dyn_plan")
        for i, (s, msgs) in enumerate(zip(res, originals)):
            cid = _inf(s).get("kgqa_case_id")
            uid = _inf(s).get("kgqa_uuid") or getattr(s, "request_id", None)
            if cid in degenerate and donors:
                donor_sample, donor_msgs = _random.choice(donors)
                if fresh_ok:
                    plans.append({"idx": i, "kind": "degen", "orig": s,
                                  "uid": uid,
                                  "req": _fresh(donor_sample, donor_msgs)})
                else:
                    # Colocate-safe mode: reuse the donor's completed
                    # trajectory. A second vLLM rollout in this process leaves
                    # the engine resident and OOMs the training forward; true
                    # fresh resampling is for server mode.
                    dup = _copy.deepcopy(donor_sample)
                    rid = f"chatcmpl-{_uuid2.uuid4().hex}"
                    dup.request_id = rid
                    dup.uuid = rid
                    plans.append({"idx": i, "kind": "degen", "orig": s,
                                  "uid": uid, "req": None, "dup": dup})
            elif cid in degenerate:
                # No variance donor: keep the sample rather than break batch
                # shape; the F1 log makes the degenerate case visible.
                plans.append({"idx": i, "kind": "skip", "orig": s,
                              "uid": uid, "req": None})
            elif _bad(s):
                if fresh_ok:
                    plans.append({"idx": i, "kind": "bad", "orig": s,
                                  "uid": uid, "req": _fresh(s, msgs)})
                else:
                    # No fresh rollout: prefer a correct trajectory from the
                    # SAME case (duplicate) over keeping a max-round failure.
                    # Search the GLOBAL records — the correct trajectory may
                    # live on the other rank (audit R6).
                    pool = flat or [
                        (_inf(t).get("kgqa_case_id"), t, originals[j])
                        for j, t in enumerate(res)]
                    good_same = [t for c2, t, _m in pool
                                 if c2 == cid and not _bad(t)
                                 and _usable(t) and _correct(t)]
                    if good_same:
                        dup = _copy.deepcopy(_random.choice(good_same))
                        rid = f"chatcmpl-{_uuid2.uuid4().hex}"
                        dup.request_id = rid
                        dup.uuid = rid
                        plans.append({"idx": i, "kind": "bad", "orig": s,
                                      "uid": uid, "req": None, "dup": dup})
                    else:
                        plans.append({"idx": i, "kind": "bad_keep",
                                      "orig": s, "uid": uid, "req": None})
            else:
                plans.append({"idx": i, "kind": "keep", "orig": s,
                              "uid": uid, "req": None})

        _phase_end("dyn_plan")
        pending = [p for p in plans if p["req"] is not None]
        done, last = {}, {}
        needs = [bool(pending)]
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                world_needs = [None] * dist.get_world_size()
                dist.all_gather_object(world_needs, needs)
                needs = [x for part in world_needs for x in part]
        except Exception:
            pass
        rounds = max(tries, 1) if any(needs) else 0
        for _ in range(rounds):
            # EVERY rank enters _generate_completions here, even with no local
            # request: run_multi_turn's gather_object is cross-rank, so letting
            # an empty rank skip the call deadlocks the rank that resamples.
            local_reqs = [p["req"] for p in pending]
            outs = orig(self, local_reqs)
            nxt = []
            for p, out in zip(pending, outs):
                last[p["idx"]] = out
                if not _bad(out) and _usable(out):
                    done[p["idx"]] = out
                    continue
                # Still bad: retry this slot with a fresh request (new uuid),
                # same donor for degen / same case for bad.
                if p["kind"] == "bad":
                    p = dict(p)
                    p["req"] = _fresh(p["orig"], originals[p["idx"]])
                else:
                    donor_sample, donor_msgs = _random.choice(donors)
                    p = dict(p)
                    p["req"] = _fresh(donor_sample, donor_msgs)
                nxt.append(p)
            pending = nxt
            # Synchronize the retry decision across ranks (same collective on
            # every rank), so no rank advances while another is still rolling.
            still = [bool(pending)]
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    world_still = [None] * dist.get_world_size()
                    dist.all_gather_object(world_still, still)
                    still = [x for part in world_still for x in part]
            except Exception:
                pass
            if not any(still):
                break

        # Defensive cleanup after a fresh resample call. The trainer's
        # _fast_infer finally already sleeps on the normal path; this is not
        # proof of release (audit F10), so the helper's warnings must be read.
        if rounds:
            try:
                if getattr(self, "vllm_mode", None) == "colocate" and \
                        int(getattr(self.args, "sleep_level", 0) or 0) > 0:
                    from swift.rlhf_trainers.utils import sleep_vllm_engine
                    sleep_vllm_engine(self.engine.engine,
                                      int(self.args.sleep_level),
                                      suppress_errors=True)
                    _dbg("post-resample vLLM sleep")
            except Exception as exc:  # noqa: BLE001
                _dbg(f"post-resample sleep failed: {exc}")

        kept, replaced_degenerate, replaced_bad, skipped = [], 0, 0, 0
        for p in plans:
            idx = p["idx"]
            if p["req"] is None:
                if p.get("dup") is not None:
                    kept.append(p["dup"])
                    if p["kind"] == "degen":
                        replaced_degenerate += 1
                    else:
                        replaced_bad += 1
                    if p["uid"]:
                        _F1_DROP.add(p["uid"])
                else:
                    if p["kind"] in ("skip", "bad_keep"):
                        skipped += 1
                    kept.append(p["orig"])
                continue
            # Retry outcome: accept only a USABLE result; a too_long/bad last
            # retry must not enter training (audit P1-C).
            out = done.get(idx)
            if out is None:
                cand = last.get(idx)
                if cand is not None and not _bad(cand) and _usable(cand):
                    out = cand
            if out is None:
                kept.append(p["orig"])
                skipped += 1
                continue
            kept.append(out)
            if p["kind"] == "degen":
                replaced_degenerate += 1
            else:
                replaced_bad += 1
            if p["uid"]:
                _F1_DROP.add(p["uid"])

        # Final validity pass (audit P1-C): a slot that is still invalid
        # (discard/too_long/max-round failure with no donor and exhausted
        # retries) must not reach training encode. Replace it with ANY usable
        # trajectory from the global pool; if the whole batch is unusable,
        # keep it but log loudly (batch shape forbids dropping).
        invalid = [i for i, s in enumerate(kept)
                   if _bad(s) or not _usable(s)]
        if invalid:
            _phase_begin("dyn_final")
            # Candidate sources (audit R4-01): the FINAL kept batch already
            # contains fresh-success trajectories that the first-rollout pool
            # never saw; prefer them, then the first-rollout pool.
            usable = [t for t in kept if not _bad(t) and _usable(t)]
            if not usable:
                pool = flat or [
                    (_inf(t).get("kgqa_case_id"), t, originals[j])
                    for j, t in enumerate(res)]
                usable = [t for _c, t, _m in pool
                          if not _bad(t) and _usable(t)]
            for i in invalid:
                if usable:
                    dup = _copy.deepcopy(_random.choice(usable))
                    rid = f"chatcmpl-{_uuid2.uuid4().hex}"
                    dup.request_id = rid
                    dup.uuid = rid
                    uid = _inf(kept[i]).get("kgqa_uuid") or getattr(
                        kept[i], "request_id", None)
                    if uid:
                        _F1_DROP.add(uid)
                    kept[i] = dup
                    replaced_bad += 1
                else:
                    cid = _inf(kept[i]).get("kgqa_case_id")
                    # Unconditional (audit C05/R4-01): _dbg is invisible
                    # without KGQA_DEBUG, and 256 configs depend on this
                    # fallback being observable.
                    print(f"[kgqa] ERROR dynamic_sample: no usable "
                          f"replacement for case={cid}; invalid sample kept "
                          "(whole batch unusable)", flush=True)
            _phase_end("dyn_final")

        _per_case = [len(v) for v in by_case.values()]
        _local_degen = sum(1 for s in res
                           if _inf(s).get("kgqa_case_id") in degenerate)
        _local_bad = sum(1 for s in res if _bad(s))
        _dbg(f"dynamic_sample cases={len(by_case)} "
             f"samples_per_case={min(_per_case) if _per_case else 0}-"
             f"{max(_per_case) if _per_case else 0} "
             f"cases_all_wrong={len(all_wrong)} "
             f"cases_all_correct={len(all_correct)} cases_variance={len(variance)} "
             f"local_degen={_local_degen} local_bad={_local_bad} "
             f"donors={len(donors)} replaced_degenerate={replaced_degenerate} "
             f"replaced_bad={replaced_bad} skipped={skipped} kept={len(kept)}")
        return kept

    _generate_completions._kgqa_dyn = True
    cls._generate_completions = _generate_completions


_install_dynamic_sampler()

# ---- 训练侧剔除：打满/空答/过长轨迹在编码前丢弃 --------------------------
def _install_discard_filter() -> None:
    """Drop discarded trajectories BEFORE encoding.

    ms-swift's teacher view has no loss mask (it rebuilds labels from ids as
    all-ones), so mask zeroing is invalid; clearing ids/mask still encodes an
    over-long prompt and hits MaxLengthError. Dropping is the only safe
    removal. Per-rank drop counts differ, so truncate all ranks to the global
    minimum length to keep gradient accumulation in sync.
    """
    if os.environ.get("KGQA_DROP_UNANSWERED", "0") != "1":
        return
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
    except Exception:
        return
    cls = getattr(RM, "RolloutTrainerMixin", None)
    if cls is None or getattr(cls._postprocess_rollout_outputs, "_kgqa_discard", False):
        return
    orig = cls._postprocess_rollout_outputs

    def _postprocess_rollout_outputs(self, samples, outputs):
        res = orig(self, samples, outputs)
        kept, dropped = [], 0
        for s in res:
            if (getattr(s, "rollout_infos", None) or {}).get("kgqa_discard"):
                dropped += 1
                continue
            kept.append(s)
        before_sync = len(kept)
        try:
            import torch
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and kept:
                t = torch.tensor([len(kept)], device=torch.cuda.current_device())
                dist.all_reduce(t, op=dist.ReduceOp.MIN)
                kept = kept[:min(len(kept), int(t.item()))]
        except Exception:
            pass
        if dropped or len(kept) != before_sync:
            _dbg(f"discard {dropped} trajectories; kept {len(kept)} "
                 f"(pre-sync {before_sync})")
        return kept

    _postprocess_rollout_outputs._kgqa_discard = True
    cls._postprocess_rollout_outputs = _postprocess_rollout_outputs


_install_discard_filter()


# ---- TITO 前缀剥离（用户裁定 2026-10-22）----------------------------------
# 采样 token 就是 response 的事实；模板的响应前缀属于 prompt，不应补进
# response（补了会让 student/teacher 两侧各补一次、长度差 12）。剥离后
# 两侧 response = 原始采样 ids，天然等长。
def _install_tito_prefix_strip() -> None:
    if os.environ.get("KGQA_TITO_STRIP_PREFIX", "1") != "1":
        return
    try:
        from swift.rlhf_trainers import gkd_helpers as GH
        from swift.rlhf_trainers import utils as U
        from swift.rollout.multi_turn import RolloutScheduler
    except Exception:
        return
    if getattr(RolloutScheduler.get_response_token_data, "_kgqa_tito", False):
        return
    empty_prefix = lambda *a, **k: []  # noqa: E731
    U.get_response_prefix_ids = empty_prefix
    GH.get_response_prefix_ids = empty_prefix

    def get_response_token_data(self, infer_request, response_choice,
                                is_continuation=False,
                                response_token_ids=None,
                                response_loss_mask=None):
        raw_ids = (list(response_choice.token_ids or [])
                   if response_token_ids is None else list(response_token_ids))
        # Validate the caller's alignment BEFORE any trimming (audit R4): the
        # mask must cover the raw ids exactly; silently truncating hid
        # upstream length bugs.
        mask = (None if response_loss_mask is None
                else list(response_loss_mask))
        if mask is not None and len(mask) != len(raw_ids):
            raise ValueError(
                f'loss mask length {len(mask)} != raw ids length {len(raw_ids)} '
                '— response ids/mask must be aligned before stop-stripping')
        # Single source of truth for the response-token contract: vLLM's
        # token_ids carry the stop token (<|im_end|>/<|endoftext|>). The
        # colocate terminal turn used to strip it via on_turn_end overrides,
        # while intermediate turns/server paths kept it (audit F08). Strip
        # here for every path, then trim the mask by the same count.
        ids = raw_ids
        if ids:
            _, ids = strip_turn_terminator("", ids)
        if mask is None:
            mask = [1] * len(ids)
        else:
            mask = mask[:len(ids)]
        assert len(mask) == len(ids), 'loss mask length mismatch'
        assert all(v in (0, 1) for v in mask), 'loss mask must be 0/1'
        return ids, mask

    get_response_token_data._kgqa_tito = True
    RolloutScheduler.get_response_token_data = get_response_token_data


_install_tito_prefix_strip()


# ---- 相位计时探针（KGQA_DEBUG=1）------------------------------------------
def _install_timing_probes() -> None:
    if not os.environ.get("KGQA_DEBUG"):
        return
    import time as _t
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
    except Exception:
        return
    cls = getattr(RM, "RolloutTrainerMixin", None)
    if cls is None or getattr(cls._generate_completions, "_kgqa_t", False):
        return
    for name in ("_preprocess_inputs", "_generate_completions",
                 "_infer_single_or_multi_turn", "_fast_infer"):
        orig = getattr(cls, name, None)
        if orig is None:
            continue

        def make(orig, name):
            def wrapper(self, *a, **k):
                t0 = _t.time()
                print(f"[kgqa-t] {name} START", file=sys.stderr, flush=True)
                r = orig(self, *a, **k)
                print(f"[kgqa-t] {name} DONE {_t.time()-t0:.1f}s",
                      file=sys.stderr, flush=True)
                return r
            wrapper._kgqa_t = True
            return wrapper
        setattr(cls, name, make(orig, name))


_install_timing_probes()


# ---- GKD 缺 completion_length_limit_scope 字段（2026-10-22）-------------
# 该字段定义在 GRPOArgumentsMixin, GKDConfig 不继承 → trainer 侧 None →
# multi_turn_completion_length_context 把 engine.max_model_len 冻结为
# 首个 prompt + max_tokens, 后续长 prompt 被挤到 max_tokens=14。
# 在 prepare_rollout 之前补 per_round, 交给 vLLM 自己编排。
def _install_len_scope_override() -> None:
    if os.environ.get("KGQA_LEN_SCOPE_PER_ROUND", "1") != "1":
        return
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
    except Exception:
        return
    cls = getattr(RM, "RolloutTrainerMixin", None)
    if cls is None or getattr(cls.prepare_rollout, "_kgqa_scope", False):
        return
    orig = cls.prepare_rollout

    def prepare_rollout(self):
        if not hasattr(self.args, "completion_length_limit_scope"):
            setattr(self.args, "completion_length_limit_scope", "per_round")
        orig(self)

    prepare_rollout._kgqa_scope = True
    cls.prepare_rollout = prepare_rollout


_install_len_scope_override()


# ---- 每步 F1 监控（rollout 质量）------------------------------------------
_F1_BUF: list[tuple[str, float, str]] = []
# Original trajectories replaced by dynamic sampling: skip their buffered F1
# so the logger reports the KEPT set (resamples record their own F1).
_F1_DROP: set[str] = set()
# Per-step mask/truncation stats set by the dynamic sampler; the F1 logger
# folds them into the step line and the jsonl file.
_MASK_STATE: dict = {"masked": None, "degenerate_cases": None,
                     "truncated": None, "total": None}


def _f1_score(pred, gt) -> float:
    """Strict entity-level F1 for training-side selection (audit R2).

    One prediction may match at most one gold (max bipartite matching); fuzzy
    matching never crosses differing digits (years/counts are part of the
    entity identity); short non-empty answers ("3") are valid — only empty
    normalizations are dropped. This is the training-side selector口径; the
    pipeline evaluator (compute_match_stats) keeps its own versioned口径 for
    offline eval comparability.
    """
    import re as _re2
    from difflib import SequenceMatcher

    def _norm(s):
        s = _re2.sub(r"[^\w\s]", " ", str(s).lower())
        return " ".join(s.split())

    def _digits(s):
        return _re2.sub(r"\D", "", s)

    pred_n = [_norm(p) for p in (pred or [])]
    gold_n = [_norm(g) for g in (gt or [])]
    pred_n = [p for p in pred_n if p]
    gold_n = [g for g in gold_n if g]
    if not pred_n or not gold_n:
        return 1.0 if not pred_n and not gold_n else 0.0

    def _match(a, b):
        if a == b:
            return True
        if len(a) >= 8 and len(b) >= 8 and _digits(a) == _digits(b) \
                and SequenceMatcher(None, a, b).ratio() >= 0.95:
            return True
        return False

    # Maximum-cardinality bipartite matching (audit P2-D): a greedy pass is
    # order-dependent — an earlier fuzzy edge can block two later matches.
    # Kuhn's augmenting path, exact edges first so ties prefer exact matches.
    adj = {i: [j for j, b in enumerate(gold_n) if _match(a, b)]
           for i, a in enumerate(pred_n)}
    exact_adj = {i: [j for j in adj[i] if pred_n[i] == gold_n[j]]
                 for i in adj}
    match_g = {}          # gold index -> pred index

    def _augment(i, seen):
        for j in exact_adj[i] + [j for j in adj[i] if j not in exact_adj[i]]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_g or _augment(match_g[j], seen):
                match_g[j] = i
                return True
        return False

    matched = 0
    for i in sorted(adj):
        if _augment(i, set()):
            matched += 1
    if not matched:
        return 0.0
    precision, recall = matched / len(pred_n), matched / len(gold_n)
    return 2 * precision * recall / (precision + recall)


def _record_f1(rc, pilot, uid: str | None = None) -> None:
    try:
        pred = rc._last_answer_entities()
        _F1_BUF.append((pilot.get("case_id", ""),
                        _f1_score(pred, pilot.get("gt_answers") or []),
                        str(uid or "")))
    except Exception:
        pass


def _install_f1_logger() -> None:
    if os.environ.get("KGQA_F1_LOG", "1") != "1":
        return
    cls = None
    for mod, name in (("swift.rlhf_trainers.rollout_mixin", "RolloutTrainerMixin"),
                      ("swift.trainers.mixin", "SwiftMixin"),
                      ("transformers", "Trainer")):
        try:
            m = __import__(mod, fromlist=[name])
            cand = getattr(m, name, None)
        except Exception:
            cand = None
        if cand is not None and hasattr(cand, "log"):
            cls = cand
            break
    if cls is None:
        return
    if getattr(getattr(cls, "log", None), "_kgqa_f1", False):
        return
    orig = cls.log

    def log(self, logs, *a, **k):
        r = orig(self, logs, *a, **k)
        try:
            acc = getattr(self, "accelerator", None)
            vals = [v for _, v, uid in _F1_BUF if not uid or uid not in _F1_DROP]
            local_raw, local_kept = len(_F1_BUF), len(vals)
            # Aggregate the KEPT rollout F1 across ranks: the main rank only
            # trains half the generation batch, so a local mean is misleading.
            # Must run on every rank (collective), even if one side is empty.
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    world = [None] * dist.get_world_size()
                    dist.all_gather_object(world, vals)
                    vals = [x for part in world for x in part]
            except Exception:
                pass
            if vals and (acc is None or acc.is_main_process):
                hit = sum(1 for v in vals if v >= 0.999)
                empty = sum(1 for v in vals if v < 1e-6)
                mean = sum(vals) / len(vals)
                step = getattr(self.state, "global_step", "?")
                ms = dict(_MASK_STATE)
                print(f"[f1] step={step} n={len(vals)} mean={mean:.4f} "
                      f"hit={hit} empty={empty} dropped={local_raw - local_kept} "
                      f"masked={ms.get('masked')}/{ms.get('total')} "
                      f"degen_cases={ms.get('degenerate_cases')} "
                      f"truncated={ms.get('truncated')} "
                      f"rep_trajs={ms.get('rep_trajs')} "
                      f"rep_turn_pct={ms.get('rep_turn_pct')} "
                      f"art_trn={ms.get('artifact_turns_masked')}",
                      file=sys.stderr, flush=True)
                f1_file = os.environ.get("KGQA_F1_FILE")
                if f1_file:
                    try:
                        import json as _json
                        with open(f1_file, "a", encoding="utf-8") as f:
                            f.write(_json.dumps({
                                "step": step, "n": len(vals),
                                "rep_trajs": ms.get("rep_trajs"),
                                "rep_turn_pct": ms.get("rep_turn_pct"),
                                "mean_f1": round(mean, 4), "hit": hit,
                                "empty": empty,
                                "dropped": local_raw - local_kept,
                                **ms}, ensure_ascii=False) + "\n")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[f1] jsonl write failed: {exc}",
                              file=sys.stderr, flush=True)
                _MASK_STATE.update(masked=None, degenerate_cases=None,
                                   truncated=None, total=None)
                if isinstance(logs, dict):
                    logs["train/rollout_f1"] = mean
                    logs["train/rollout_hit"] = hit / len(vals)
            _F1_BUF.clear()
            _F1_DROP.clear()
        except Exception:
            pass
        return r

    log._kgqa_f1 = True
    cls.log = log


_install_f1_logger()


# ---- 相位 profiler（KGQA_PHASE_PROFILE=1，wall 区间，无并发重复计数）------
def _install_phase_profiler() -> None:
    """Per-rank wall-clock phase spans for one optimizer step.

    All spans are single-clock intervals on the main thread; nested spans
    (prelink ⊂ rollout, loss_* ⊂ train) are reported as sub-items and never
    added into their parent. vLLM engine construction happens at process
    init (before step 1) and is therefore excluded by construction; the
    summary reports step-internal spans only.
    """
    if os.environ.get("KGQA_PHASE_PROFILE") != "1":
        return
    try:
        from swift.rlhf_trainers import rollout_mixin as RM
        from swift.rlhf_trainers.gkd_trainer import GKDTrainer
        from swift.trainers.mixin import SwiftMixin
    except Exception:
        return

    def _wrap(cls, name, span):
        if getattr(getattr(cls, name, None), "_kgqa_phase", False):
            return
        orig = getattr(cls, name, None)
        if orig is None:
            return

        def wrapper(self, *a, **k):
            _phase_begin(span)
            try:
                return orig(self, *a, **k)
            finally:
                _phase_end(span)
        wrapper._kgqa_phase = True
        setattr(cls, name, wrapper)

    _wrap(RM.RolloutTrainerMixin, "_generate_completions", "rollout_total")
    _wrap(RM.RolloutTrainerMixin, "_prepare_batch_inputs", "batch_inputs")
    _wrap(RM.RolloutTrainerMixin, "_prepare_inputs", "prepare_inputs")
    _wrap(RM.RolloutTrainerMixin, "_move_model_to_vllm", "wake_sync")
    _wrap(RM.RolloutTrainerMixin, "_postprocess_rollout_outputs", "postproc")
    _wrap(GKDTrainer, "_encode_samples", "encode")
    # sleep/cleanup runs in _fast_infer's finally via module reference
    if not getattr(RM.sleep_vllm_engine, "_kgqa_phase", False):
        _orig_sleep = RM.sleep_vllm_engine

        def sleep_vllm_engine(*a, **k):
            _phase_begin("sleep_cleanup")
            try:
                return _orig_sleep(*a, **k)
            finally:
                _phase_end("sleep_cleanup")
        sleep_vllm_engine._kgqa_phase = True
        RM.sleep_vllm_engine = sleep_vllm_engine
    # the executed training_step lives on transformers.Trainer in this stack
    # (SwiftMixin's is shadowed); wrap both defensively.
    import transformers
    _wrap(transformers.Trainer, "training_step", "train_total")

    # prelink is async on the scheduler; accumulate per-call wall
    orig_start = KGQAScheduler.on_trajectory_start
    if not getattr(orig_start, "_kgqa_phase", False):
        async def start(self, requests):
            _phase_begin("prelink")
            try:
                return await orig_start(self, requests)
            finally:
                _phase_end("prelink")
        start._kgqa_phase = True
        KGQAScheduler.on_trajectory_start = start

    # summary on every log() (logging_steps=1 → after each optimizer step)
    if getattr(SwiftMixin.log, "_kgqa_phase_sum", False):
        return
    orig_log = SwiftMixin.log

    def log(self, logs, *a, **k):
        r = orig_log(self, logs, *a, **k)
        try:
            step = getattr(self.state, "global_step", "?")
            def g(n):
                return f"{_PHASE.get(n, 0.0):.1f}"
            print(f"[phase] r{os.environ.get('RANK', '?')} step={step} "
                  f"rollout={g('rollout_total')} "
                  f"(prelink={g('prelink')} wake_sync={g('wake_sync')} "
                  f"sleep={g('sleep_cleanup')} postproc={g('postproc')} \
                  dyn:[f1g={g('dyn_f1_gather')} pack={g('dyn_donor_pack')} \
gather={g('dyn_donor_gather')} plan={g('dyn_plan')} \
final={g('dyn_final')}]) "
                  f"prep={g('prepare_inputs')} encode={g('encode')} "
                  f"batch={g('batch_inputs')} "
                  f"train={g('train_total')} "
                  f"(sfwd={g('loss_student_fwd')} tfwd={g('loss_teacher_fwd')} "
                  f"chunk={g('loss_chunk_loop')} "
                  f"bwd_opt~{float(_PHASE.get('train_total', 0)) - float(_PHASE.get('loss_student_fwd', 0)) - float(_PHASE.get('loss_teacher_fwd', 0)) - float(_PHASE.get('loss_chunk_loop', 0)) - float(_PHASE.get('rollout_total', 0)):.1f})",
                  file=sys.stderr, flush=True)
            for nm in ("rollout_total", "wake_sync", "sleep_cleanup",
                       "postproc", "prepare_inputs", "encode", "train_total"):
                print(f"[phase-ts] r{os.environ.get('RANK', '?')} {nm} "
                      f"{_ts(nm, 0)} -> {_ts(nm, 1)}",
                      file=sys.stderr, flush=True)
            # tool-layer parallelism (walk lanes + GTE client batching) for
            # THIS step, this rank's half of the requests. exec≫wait = lanes
            # had spare capacity (fanout working, not serialized); slots<reqs
            # and memo>0 = dedup/memo absorbing repeated cases; items<reqs =
            # GTE burst coalescing. Zero flushes with lanes=0 = every walk
            # was a memo hit (same case pool across steps) — idle by design.
            try:
                from kgqa.agent import seq_tools as _st
                from kgqa.stages.stage2_entity import _GTE_BATCH_STATS as _gs
                from kgqa.core.utils import PHASE_TIMES as _PT
                ws = _st._WALK_BATCH_STATS
                # INTERNAL tool-execution errors this step (env/GTE/code bugs
                # surfaced as EXECUTION: strings — the KeyError:'walk' run
                # poisoned every trajectory silently; fail fast instead of
                # burning GPU-hours on broken rollouts). KGQA_TOOL_ERR_ABORT=0
                # disables; default 24 ≈ a quarter step's trajectories already
                # degraded — transient one-offs never reach it.
                _errs = {}
                try:
                    from kgqa.agent.seq_adaptive import EXEC_ERR_COUNTS
                    _errs = dict(EXEC_ERR_COUNTS)
                    EXEC_ERR_COUNTS.clear()
                except Exception:  # noqa: BLE001
                    pass
                _nerr = sum(_errs.values())
                _top = max(_errs.items(), key=lambda kv: kv[1])[0] if _errs else "-"
                print(f"[tool] r{os.environ.get('RANK', '?')} step={step} "
                      f"pool={os.environ.get('WALK_POOL', '0')} "
                      f"lanes={len(_st._WALK_LANES) if _st._WALK_LANES else 0} "
                      f"exec_err={_nerr}{'' if _top == '-' else f'({_top})'} "
                      f"walk: {ws['batches']} flushes "
                      f"(single={ws['single']} burst={ws['burst']}) "
                      f"reqs={ws['reqs']} slots={ws['slots']} "
                      f"memo={ws['memo_hits']} dedup={ws['dedup_shares']} "
                      f"exec={_PT.get('walk_exec', 0.0):.1f}s "
                      f"wait={_PT.get('walk_wait', 0.0):.1f}s "
                      f"flushwall={_PT.get('walk_flush_wall', 0.0):.1f}s "
                      f"| gte: {_gs['flushes']} flushes reqs={_gs['reqs']} "
                      f"items={_gs['http_items']} memo={_gs['memo_hits']} "
                      f"dedup={_gs['dedup']} "
                      f"post={_PT.get('gte_post_wall', 0.0):.1f}s",
                      file=sys.stderr, flush=True)
                _abort = int(os.environ.get("KGQA_TOOL_ERR_ABORT", "24") or 0)
                if _abort and _nerr >= _abort:
                    print(f"[tool] FATAL r{os.environ.get('RANK', '?')}: "
                          f"{_nerr} internal tool-execution errors this step "
                          f"{_errs} — environment/code is broken (see "
                          f"rollouts/step dumps); aborting instead of training "
                          f"on poisoned trajectories "
                          f"(KGQA_TOOL_ERR_ABORT=0 to disable)",
                          file=sys.stderr, flush=True)
                    raise ToolLayerBroken(
                        f"tool layer broken: {_nerr} exec errors {_errs}")
                for _k in ws:
                    ws[_k] = 0.0 if isinstance(ws[_k], float) else 0
                for _k in _gs:
                    _gs[_k] = 0.0 if isinstance(_gs[_k], float) else 0
                # RESET, never pop (2026-10-24 崩塌根因): the walk path
                # writes PHASE_TIMES with bare subscripts (phase_timer and
                # walk_collect/wait/exec +=) — popping the keys here made
                # step 2's first retrieve_subgraph raise KeyError: 'walk',
                # which the adaptive layer turned into a tool error for
                # every trajectory (empty-answer avalanche, two runs lost).
                for _k in ("walk", "walk_collect", "walk_wait", "walk_exec",
                           "walk_flush_wall", "gte", "gte_n", "gte_collect",
                           "gte_post_wall"):
                    _PT[_k] = 0.0
            except Exception as exc:  # noqa: BLE001
                print(f"[tool] stats failed: {exc}", file=sys.stderr)
                if isinstance(exc, ToolLayerBroken):
                    raise          # fail-fast must survive this handler
            _PHASE.clear()
            _CHUNK_STATS.clear()
            _PHASE_TS.clear()
        except Exception as exc:  # noqa: BLE001
            print(f"[phase] summary failed: {exc}", file=sys.stderr)
            if isinstance(exc, ToolLayerBroken):
                raise          # fail-fast survives every handler here
        return r
    log._kgqa_phase_sum = True
    SwiftMixin.log = log


_install_phase_profiler()


# ---- C02 验收：跨 rank 可训练参数一致性（KGQA_RANK_PARAM_CHECK=1）---------
def _install_rank_param_check() -> None:
    """After each optimizer step, all-gather the flat trainable-parameter
    vector and compare ranks (audit C02 acceptance).

    With correct ZeRO-2 gradient reduction both ranks must apply the same
    update to the same (unsharded, ZeRO-2) parameters, so the flat vectors
    must match to float tolerance. A silent per-rank divergence — the risk
    of bypassing the wrapped forward — shows up here as a nonzero max diff.
    """
    if os.environ.get("KGQA_RANK_PARAM_CHECK") != "1":
        return
    try:
        import torch
        import torch.distributed as dist
        from swift.trainers.mixin import SwiftMixin
    except Exception:
        return
    if getattr(SwiftMixin.log, "_kgqa_rparam", False):
        return
    checks = {"n": 0}
    orig_log = SwiftMixin.log

    def log2(self, logs, *a, **k):
        r = orig_log(self, logs, *a, **k)
        try:
            if dist.is_available() and dist.is_initialized():
                params = [p for p in self.model.parameters()
                          if p.requires_grad]
                flat = torch.cat([p.detach().reshape(-1)
                                  for p in params]).float().contiguous()
                world = [torch.empty_like(flat)
                         for _ in range(dist.get_world_size())]
                dist.all_gather(world, flat)
                base = world[0]
                md = max(float((w - base).abs().max()) for w in world[1:])
                n = int(flat.numel())
                same = "OK" if md <= 1e-6 else "DIVERGED"
                checks["n"] += 1
                print(f"[rparam] r{os.environ.get('RANK', '?')} "
                      f"check#{checks['n']} n={n} max|Δrank0|={md:.3e} "
                      f"{same}", file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[rparam] check failed: {exc}", file=sys.stderr)
        return r
    log2._kgqa_rparam = True
    SwiftMixin.log = log2


_install_rank_param_check()
