"""OPSD (one-teacher-per-case self-distillation) 与 verl 的配对装配.

为什么单独拆出来
----------------
verl 的 OPD 假定 teacher 与 student 用**同一个 prompt** 打分, 直接按 token
打分即可。OPSD 的 teacher 条件不同 (refined 轨迹 + OPSD 提示语), 而且我们
只监督**答案轮** (thinking+action 的 raw 生成), 不是整条多轮序列。因此
rollout 结束后要独立装配一个训练样本:

    prompt_ids   = 学生条件 (真实轨迹到答案轮之前) + 该轮提示
    response_ids = 答案轮的 raw 生成 token (on-policy)
    response_mask= 全 1 (只监督这一段)
    teacher_*    = 预计算: 同一段 token 在 refined 条件 (teacher) 下的分布

本模块是纯函数 (无 verl 依赖), 便于单测; verl agent loop 只是它的调用方。
"""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"^\s*<think>(.*?)(?:</think>|\Z)", re.S)
_TOOL_RE = re.compile(r"""["']?tool["']?\s*[:=]\s*["']?(\w+)""")


def split_thinking(raw: str) -> tuple[str, str]:
    """把原始生成拆成 (reasoning, content)。

    与 vLLM reasoning parser 同语义: `<think>...</think>` 归 reasoning, 其余
    归 content; 未闭合时整段都是 reasoning、content 为空 (被截断的思考轮)。
    """
    m = _THINK_RE.match(raw or "")
    if not m:
        return "", raw or ""
    reasoning = m.group(1)
    rest = (raw or "")[m.end():].lstrip("\n")
    return reasoning.strip(), rest


def last_action(raw: str) -> str | None:
    """从生成文本里取最后一个工具名 (与 seq_react_loop 的 canonical 解析同口径)。"""
    mm = _TOOL_RE.findall(raw or "")
    return mm[-1] if mm else None


def build_prompt_ids(tokenizer, messages: list[dict], hint: str) -> list[int]:
    """学生条件 token 化: 会话 + 本轮提示 + generation prompt。"""
    msgs = list(messages) + [{"role": "user", "content": hint}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def assemble_answer_pair(
    tokenizer,
    messages_before: list[dict],
    hint: str,
    answer_raw: str,
    answer_token_ids: list[int],
    teacher_ids: list[int] | None = None,
    teacher_logprobs: list[list[float]] | None = None,
) -> dict:
    """装配一个 OPSD 训练对 (纯数据, 不碰 verl 类型)。

    `answer_token_ids` 必须是该轮**实际生成**的 token (on-policy), 不能由
    `answer_raw` 重新 tokenize —— 那会与生成时的分词不一致 (例如服务端
    reasoning parser 处理过的边界)。
    """
    if not answer_token_ids:
        raise ValueError("answer_token_ids 为空: 该轮没有监督 token")
    if teacher_ids is not None and len(teacher_ids) != len(answer_token_ids):
        raise ValueError(
            f"teacher 对齐长度不一致: {len(teacher_ids)} vs "
            f"{len(answer_token_ids)} (teacher 必须按同一段 on-policy token 打分)")
    prompt_ids = build_prompt_ids(tokenizer, messages_before, hint)
    return {
        "prompt_ids": prompt_ids,
        "response_ids": list(answer_token_ids),
        "response_mask": [1] * len(answer_token_ids),
        "answer_raw": answer_raw,
        "teacher_ids": teacher_ids,
        "teacher_logprobs": teacher_logprobs,
    }


def find_answer_turn(trajectory: list[dict]) -> int | None:
    """返回 trajectory 中最后一个 `tool: answer` 的 assistant 轮下标。"""
    idx = None
    for i, m in enumerate(trajectory):
        if m.get("role") == "assistant" and last_action(
                m.get("raw") or m.get("content") or "") == "answer":
            idx = i
    return idx


# Qwen3.5 的回合终止符。ms-swift 的 `decode_generate_ids` 直接 decode 采样
# token_ids，而 vLLM 命中 stop 时这些 token 仍留在 token_ids 里 → 文本尾部
# 会带字面量 `<|im_end|>`，训练模板又会再加一个，历史/动作都会被污染。
_TURN_END_IDS = {248044, 248046}  # <|im_end|>, <|endoftext|> (tokenizer 对齐后的 eos)
_TURN_END_STRS = ("<|im_end|>", "<|endoftext|>")


def strip_turn_terminator(raw: str, ids):
    """去掉本轮末尾的回合终止符（文本后缀 + token 尾 id）。

    返回 (text, ids)；只剥尾部，不改动中间内容。模板负责在渲染时补回合
    终止符，所以这里剥掉不会丢监督目标的位置（终止符本身不是模型行为）。
    """
    text = raw or ""
    for suf in _TURN_END_STRS:
        stripped = text.rstrip()
        if stripped.endswith(suf):
            text = stripped[:-len(suf)].rstrip()
            break
    out = list(ids or [])
    while out and out[-1] in _TURN_END_IDS:
        out.pop()
    return text, out
