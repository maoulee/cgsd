"""跳过 verl 的教师池 (OPSD 设计: 没有独立 teacher 模型).

用户设计 (2026-10-21 重申): 两卡闭环 —— rollout → 判定/精炼参考 → 同一
vLLM 按 refined 条件给学生的答案跨度打分 → sleep(2) 卸载权重+KV 到内存 →
两卡 FSDP 蒸馏 (同一冻结基座, teacher=refined 前缀, student=真实轨迹) →
wake。**没有独立教师模型, 也就不需要教师池。**

verl 的框架假设相反: `distillation.enabled=true` 时必定分配
`teacher_pool` (`n_gpus_per_node × nnodes`)、建 MultiTeacherModelManager、
并在 agent loop 里调用教师服务。2 卡机器上多 1 卡就会排不上。

本插件只改 **trainer 侧** 的三处 (worker 侧保持蒸馏损失启用):
  1. `_init_resource_pool_mgr`: 临时把 distillation.enabled 设 False → 不分配
     teacher_pool、不做 nnodes>0 校验;
  2. `_setup`: 临时把 use_teacher_policy 设 False → 不建
     MultiTeacherModelManager; 结束后恢复 distillation_config,
     让 `_update_actor` 的 distillation_use_topk / distillation_only 仍正确;
  3. `get_teacher_client` 返回 None 的路径保持原样 (teacher_client 允许 None)。
worker 进程收到的是原始 config (enabled=true), 因此
`engine_workers.EngineWorker.distillation_enabled` 仍为 True, 损失照用。

由 `OPSD_NO_TEACHER_POOL=1` 开启 (launcher 从 YAML 的
`kgqa.no_teacher_pool` 导出)。
"""

from __future__ import annotations

import os


def install() -> None:
    if os.environ.get("OPSD_NO_TEACHER_POOL", "0") != "1":
        return
    import verl.trainer.ppo.v1.trainer_base as TB

    cls = TB.PPOTrainer
    if getattr(cls._init_resource_pool_mgr, "_opsd_no_teacher", False):
        return

    orig_pool = cls._init_resource_pool_mgr

    def _init_resource_pool_mgr(self):
        d = self.config.get("distillation")
        saved = None if d is None else d.enabled
        if d is not None:
            d.enabled = False          # 跳过 teacher_pool 分配与 nnodes 校验
        try:
            return orig_pool(self)
        finally:
            if d is not None:
                d.enabled = saved

    _init_resource_pool_mgr._opsd_no_teacher = True
    cls._init_resource_pool_mgr = _init_resource_pool_mgr

    orig_setup = cls._setup

    def _setup(self, *args, **kwargs):
        saved_flag = bool(getattr(self, "use_teacher_policy", False))
        if saved_flag:
            self.use_teacher_policy = False   # 不建教师服务
        try:
            return orig_setup(self, *args, **kwargs)
        finally:
            if saved_flag:
                self.use_teacher_policy = True
                if getattr(self, "distillation_config", None) is None:
                    # _setup 在 flag=False 时把蒸馏配置置 None; _update_actor
                    # 的 distillation_only/topk 判定仍需要它
                    from verl.utils.config import omega_conf_to_dataclass
                    self.distillation_config = omega_conf_to_dataclass(
                        self.config.distillation)

    _setup._opsd_no_teacher = True
    cls._setup = _setup
