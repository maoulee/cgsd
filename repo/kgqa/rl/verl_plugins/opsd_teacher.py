"""让 verl 使用我们预计算的 teacher 打分 (OPSD), 而不是教师服务池.

背景 (实证, verl 0.9.1)
----------------------
`verl/experimental/agent_loop/agent_loop.py`
  :830-840  优先消费 `AgentLoopOutput.extra_fields["teacher_ids"/"teacher_logprobs"]`
  :1031     `AgentLoopWorker._compute_teacher_logprobs` —— 只要
            `distillation.enabled` 就调用教师服务, 并**覆盖** extra_fields.

我们的 teacher = 同一冻结基座在 refined 前缀下打分, teacher 条件与 student
不同, 无法用 OPD 的"同 prompt 打分"路径; 而且打分在判定阶段已离线算好
(复用 TDH hidden 缓存). 因此在 worker 上打一个极小补丁: 若样本已带预计算
teacher 字段, 则跳过教师服务调用.

用法: launcher 最前面 `import kgqa.rl.verl_plugins.opsd_teacher as t; t.install()`
"""

from __future__ import annotations

import os


def install() -> None:
    """幂等: 让 _compute_teacher_logprobs 在已有预计算字段时成为 no-op。"""
    from verl.experimental.agent_loop import agent_loop as A

    cls = A.AgentLoopWorker
    if getattr(cls._compute_teacher_logprobs, "_opsd_patched", False):
        return
    orig = cls._compute_teacher_logprobs

    async def _compute_teacher_logprobs(self, output, prompt_ids,
                                        response_ids, validate,
                                        sample_kwargs=None):
        extra = getattr(output, "extra_fields", None) or {}
        if extra.get("teacher_ids") is not None and \
                extra.get("teacher_logprobs") is not None:
            # 预计算 teacher 打分: 不改写, 直接沿用 (verl 后续会 pop 并打包)
            return None
        if os.environ.get("OPSD_NO_TEACHER_POOL", "0") == "1":
            # 没有教师服务可调 (同一基座在 refined 条件下打分, 已预计算);
            # 缺字段的样本按"无 teacher"继续, 数据侧过滤
            return None
        return await orig(self, output, prompt_ids, response_ids,
                          validate, sample_kwargs=sample_kwargs)

    _compute_teacher_logprobs._opsd_patched = True
    cls._compute_teacher_logprobs = _compute_teacher_logprobs
