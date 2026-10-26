"""sleep 等级插件: hybrid 让位时把 vLLM 权重卸载到 CPU (用户裁定 2026-10-21).

背景
----
verl `vLLMHttpServer._resolve_sleep_level()` 在 `lora_as_adapter` (我们用 LoRA)
或 MTP rollout 时返回 **1** —— 只 offload KV cache, **模型权重留在 GPU**。
用户要求 hybrid 让位时把权重也卸载到内存, 即 level 2。

风险与前置条件
--------------
level 2 会丢弃/卸载 vLLM 侧权重; wake 后 verl 若只同步 LoRA adapter, 基座
权重可能处于未定义状态。**启用前必须验证**: 加载 adapter → 记录生成 →
sleep(2) → wake → 同输入生成, 逐 token 一致。若不一致, 改回 level 1 或
wake 后做全量权重同步。

配置
----
`config/verl/opsd.yaml` 的 `kgqa.sleep_level`; launcher 在启动前把它导出为
`OPSD_SLEEP_LEVEL` (vLLM server actor 拿不到 trainer config, 只能走 env)。
未设置时**保守回到 verl 原逻辑** (不强制), 避免静默改变训练语义。
"""

from __future__ import annotations

import os


def install() -> None:
    """幂等: 用 OPSD_SLEEP_LEVEL 覆盖 _resolve_sleep_level。"""
    from verl.workers.rollout.vllm_rollout import vllm_async_server as S

    cls = S.vLLMHttpServer
    if getattr(cls._resolve_sleep_level, "_opsd_patched", False):
        return
    orig = cls._resolve_sleep_level

    def _resolve_sleep_level(self) -> int:
        lvl = os.environ.get("OPSD_SLEEP_LEVEL", "").strip()
        if lvl:
            return int(lvl)
        return orig(self)

    _resolve_sleep_level._opsd_patched = True
    cls._resolve_sleep_level = _resolve_sleep_level
