"""KGQA 的 verl agent loop (OPSD 自蒸馏, S2).

设计要点 (2026-10-21)
--------------------
1. **复用领域机制**: 直接驱动 `kgqa.agent.seq_react_loop.SeqReactCase`
   (prelink / state_aware_hint / allowed_tools_hint / loop_nudge /
   process_turn) —— 工具、账本、渲染全走现有实现, 不在 verl 里重写。
2. **不用 verl 的工具解析**: verl 的 `tool_agent_loop` 走 `tool_parser`
   (hermes/function-call 结构化 tool_calls); 我们的协议是模型在 content 里
   写 `tool: ...` 行, 由 `process_turn` 解析。定制 loop 直接接管解析,
   `tool_parser` 不参与。
3. **训练样本 = 答案轮配对** (`opsd_pair.assemble_answer_pair`):
   prompt = 真实轨迹到答案轮之前 + 该轮提示; response = 该轮 raw 生成
   (thinking+action, on-policy token); mask 全 1。teacher 打分由上游预计算
   (extra_info) 注入, `opsd_teacher.install()` 保证不被教师服务覆盖。
4. 多轮轨迹本身放进 `extra_fields["kgqa_trajectory"]` 供判定/审计。

数据集契约 (`extra_info`):
    kgqa_sample : the pipeline sample dict (text_entity_list / relation_list /
                  h/r/t_id_list ... — build_context 的输入)
    kgqa_pilot  : {case_id, question, gt_answers}
    kgqa_idx    : int
    teacher_ids / teacher_logprobs (可选, 长度 = 答案轮 token 数)
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import aiohttp

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

from kgqa.agent.seq_react_loop import SeqReactCase

from .opsd_pair import assemble_answer_pair, last_action, split_thinking


@register("kgqa_opsd_agent")
class KGQAOPSDAgentLoop(AgentLoopBase):
    """OPSD 轨迹采样: 现有 KGQA harness + 答案轮配对输出。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 配置源: YAML 的 `kgqa:` 块 (config/verl/opsd.yaml); env 仅作回退
        kgqa = self.config.get("kgqa", {}) if self.config is not None else {}
        self.max_rounds = int(kgqa.get(
            "max_rounds", os.environ.get(
                "MAX_ROUNDS", os.environ.get("SEQ_MAX_ROUNDS", "16"))))
        self.think_budget = int(kgqa.get(
            "think_budget", os.environ.get("THINK_BUDGET", "1000")))
        self.act_budget = int(kgqa.get(
            "act_budget", os.environ.get("ACT_BUDGET", "500")))
        self.gte_urls = kgqa.get(
            "gte_api_urls", os.environ.get("GTE_API_URLS", "http://localhost:8003"))

    # ---- helpers -----------------------------------------------------
    def _final_hint(self, rc: SeqReactCase) -> str:
        return ("FINAL TURN (budget extended by one): submit your final answer "
                "NOW with `tool: answer` from the bindings you hold — this is "
                "the LAST turn; further retrieval will not be executed. "
                + rc.allowed_tools_hint())

    def _turn_hint(self, rc: SeqReactCase) -> str:
        hint = rc.state_aware_hint() + rc.allowed_tools_hint()
        if rc._tool_repeat >= 2:
            hint = rc.loop_nudge() + "\n" + hint
        return hint

    async def _one_turn(self, session, messages_before, hint,
                        sampling_params, priority):
        """生成一轮并交给 harness 处理; 返回 (prompt_ids, token_ids, raw)。"""
        from .opsd_pair import build_prompt_ids
        prompt_ids = build_prompt_ids(self.tokenizer, messages_before, hint)
        out = await self.server_manager.generate(
            request_id=uuid.uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            priority=priority,
        )
        token_ids = list(out.token_ids)
        raw = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return prompt_ids, token_ids, raw

    async def run(self, sampling_params: dict[str, Any], priority: int = 0,
                  **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info") or {}
        sample = extra.get("kgqa_sample")
        if sample is None:
            raise ValueError(
                "kgqa_opsd_agent: extra_info 缺 kgqa_sample (数据集必须带 "
                "build_context 所需的 sample dict)")
        pilot = extra.get("kgqa_pilot") or {}
        idx = int(extra.get("kgqa_idx", 0))

        sp = dict(sampling_params)
        # 裁定② 单轮总预算 (1000/1500 规格); 覆盖 verl 由 response_length
        # 推来的 max_tokens, 否则思考膨胀会把预算放大回 2560+ 档
        sp["max_tokens"] = self.think_budget + self.act_budget

        rc = SeqReactCase(sample, pilot, idx)
        t0 = time.time()
        trajectory: list[dict] = []
        pair: dict | None = None
        last_prompt_ids: list[int] = []
        last_token_ids: list[int] = []
        turns = 0

        try:
            async with aiohttp.ClientSession() as session:
                await rc.prelink(session)
                for turn in range(self.max_rounds):
                    if not rc.is_active:
                        break
                    rc.ctx._seq_turns_remaining = self.max_rounds - turn
                    hint = self._turn_hint(rc)
                    before = list(rc.messages)
                    prompt_ids, token_ids, raw = await self._one_turn(
                        session, before, hint, sp, priority)
                    reasoning, content = split_thinking(raw)
                    if last_action(content) == "answer":
                        pair = assemble_answer_pair(
                            self.tokenizer, before, hint, raw, token_ids,
                            teacher_ids=extra.get("teacher_ids"),
                            teacher_logprobs=extra.get("teacher_logprobs"))
                        pair["reasoning"], pair["content"] = reasoning, content
                    trajectory.append({
                        "role": "assistant", "raw": raw,
                        "reasoning": reasoning, "content": content,
                        "prompt_len": len(prompt_ids),
                    })
                    last_prompt_ids, last_token_ids = prompt_ids, token_ids
                    await rc.process_turn(session, content, reasoning)
                    turns += 1
                    if rc.done:
                        break
                if rc.is_active and turns < self.max_rounds + 1:
                    # 末轮兜底 (与 rollout 的 final-turn +1 同语义)
                    hint = self._final_hint(rc)
                    before = list(rc.messages)
                    prompt_ids, token_ids, raw = await self._one_turn(
                        session, before, hint, sp, priority)
                    reasoning, content = split_thinking(raw)
                    if last_action(content) == "answer":
                        pair = assemble_answer_pair(
                            self.tokenizer, before, hint, raw, token_ids,
                            teacher_ids=extra.get("teacher_ids"),
                            teacher_logprobs=extra.get("teacher_logprobs"))
                        pair["reasoning"], pair["content"] = reasoning, content
                    trajectory.append({"role": "assistant", "raw": raw,
                                       "reasoning": reasoning,
                                       "content": content,
                                       "prompt_len": len(prompt_ids)})
                    last_prompt_ids, last_token_ids = prompt_ids, token_ids
                    await rc.process_turn(session, content, reasoning)
                    turns += 1
        except Exception as e:                       # noqa: BLE001
            rc.failed = True
            rc.failure_reason = f"kgqa_opsd_agent: {e}"

        # 无 answer 轮 (弃答/失败): 以末轮作 response 但 mask 全 0 —
        # 该样本不参与蒸馏, 数据侧可按 extra_fields["kgqa_trainable"] 过滤
        trainable = pair is not None
        if pair is None:
            pair = {
                "prompt_ids": last_prompt_ids,
                "response_ids": last_token_ids,
                "response_mask": [0] * len(last_token_ids),
                "teacher_ids": None,
                "teacher_logprobs": None,
            }

        return AgentLoopOutput(
            prompt_ids=pair["prompt_ids"],
            response_ids=pair["response_ids"],
            response_mask=pair["response_mask"],
            reward_score=None,
            num_turns=max(turns, 1),
            metrics=AgentLoopMetrics(
                generate_sequences=time.time() - t0, tool_calls=0.0,
                compute_score=0.0, num_preempted=-1),
            extra_fields={
                "kgqa_case_id": pilot.get("case_id", sample.get("id", "")),
                "kgqa_idx": idx,
                "kgqa_trainable": trainable,
                "kgqa_failed": rc.failed,
                "kgqa_failure_reason": rc.failure_reason,
                "kgqa_trajectory": trajectory,
                # 预计算 teacher 打分 (verl 的 _postprocess 会 pop 并打包;
                # opsd_teacher.install() 防止教师服务覆盖)
                **({"teacher_ids": extra["teacher_ids"],
                    "teacher_logprobs": extra["teacher_logprobs"]}
                   if trainable and extra.get("teacher_ids") is not None
                   else {}),
            },
        )
