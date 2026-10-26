"""OPSD teacher 打分客户端 (在 rollout 之内, 但条件与 student 不同).

与 verl OPD 的差别: OPD 用同一 prompt 让 teacher 给 response 打分; 我们的
teacher 条件是 refined 轨迹 + OPSD 提示语 (上一步判定产出的案级参考), 打分
对象仍是学生本轮 on-policy 的答案轮 token. 因此在 agent loop 里生成完答案
轮后, 用本模块向教师服务 (冻结基座 vLLM) 请求这些 token 的 top-k logprob.

服务端要求: vLLM `/v1/completions` 支持 `prompt_logprobs` (token id 数组作为
prompt). 返回的 `prompt_logprobs` 是按位置的 {token_id: {logprob, rank}}; 取
最后 `n_answer` 个位置即答案轮的 teacher 分布.

纯函数 (`build_teacher_messages` / `parse_prompt_logprobs`) 可离线单测;
`score_answer` 需要在线教师服务.
"""

from __future__ import annotations

from typing import Any

# 与 kd_real_smoke.build_pair 完全同文 (正典 OPSD)
OPSD_PROMPT_HEAD = "Here is a reference solution (a worked agent trajectory):\n\n"
OPSD_PROMPT_TAIL = ("\n\nAfter understanding the reference solution, please try "
                    "to solve this problem using your own approach below:")


def serialize_refined(turns: list[dict]) -> str:
    """refined 轨迹序列化: plan + 工具调用 + 正文 (+ 穿插工具结果)."""
    out = []
    for m in turns:
        tag = "Agent: " if m.get("role") == "assistant" else ""
        out.append(tag + (m.get("content") or ""))
    return "\n\n".join(out)


def build_teacher_messages(system_prompt: str, question: str,
                           refined_turns: list[dict]) -> list[dict]:
    """teacher 条件 = 系统提示语 + 问题 + 内嵌 refined 参考的 OPSD 提示语."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "user",
         "content": OPSD_PROMPT_HEAD + serialize_refined(refined_turns)
                    + OPSD_PROMPT_TAIL},
    ]


def parse_prompt_logprobs(prompt_logprobs: list[Any], n_answer: int,
                          topk: int) -> tuple[list[list[int]], list[list[float]]]:
    """取最后 n_answer 个位置的 top-k (id, logprob), 不足 topk 的补齐。

    补齐用 id=0 / logprob=-1e9: exp(-1e9)=0, 不影响截断质量归一与 JSD。
    """
    tail = prompt_logprobs[-n_answer:]
    if len(tail) != n_answer:
        raise ValueError(f"prompt_logprobs 长度不足: {len(prompt_logprobs)} "
                         f"< {n_answer}")
    ids, lps = [], []
    for d in tail:
        if not d:
            raise ValueError("答案轮位置缺 logprob (检查 prompt_logprobs 参数)")
        items = sorted(d.items(), key=lambda kv: kv[1].get("logprob", -1e9),
                       reverse=True)[:topk]
        row_ids = [int(k) for k, _ in items]
        row_lps = [float(v.get("logprob", -1e9)) for _, v in items]
        while len(row_ids) < topk:
            row_ids.append(0)
            row_lps.append(-1e9)
        ids.append(row_ids)
        lps.append(row_lps)
    return ids, lps


async def score_answer(session, url: str, model: str, prompt_ids: list[int],
                       n_answer: int, topk: int,
                       timeout: float = 120.0) -> tuple[list[list[int]], list[list[float]]]:
    """向教师服务请求答案轮的 top-k logprob (prompt = teacher条件+答案token)."""
    body = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": topk,
    }
    async with session.post(url.rstrip("/") + "/v1/completions",
                            json=body,
                            timeout=timeout) as r:
        r.raise_for_status()
        data = await r.json()
    pls = data["choices"][0]["prompt_logprobs"]
    return parse_prompt_logprobs(pls, n_answer, topk)
