"""JSD distillation loss plugin for verl (OPSD alignment, 2026-10-21).

背景
----
verl 0.9.1 的蒸馏损失分两段:
  1) logits processor 段 (`verl.trainer.distillation.losses.compute_topk_loss`)
     —— 按 strategy 硬编码为 `compute_forward_kl_topk`(fsdp/megatron), 产出
     `model_output["distillation_losses"] / ["student_mass"] / ["teacher_mass"]`;
  2) 最终损失段 (`distillation_loss`) —— 走 `DISTILLATION_LOSS_REGISTRY`,
     消费第 1 段产物。
注册表只覆盖第 2 段, 所以 JSD 需要: 注册同名最终损失函数 + 在第 1 段分派处
插入一个分支 (本模块 `install()` 的 monkeypatch, 不改 site-packages 文件)。

语义
----
teacher top-k 支撑集上的截断 JSD:
  lp_t, lp_s: [B, T, K] (在 teacher top-k ids 上 gather)
  先按各自支撑集质量归一 (p̃ = p / mass), 再
  JSD = 0.5*KL(p̃_t || m) + 0.5*KL(p̃_s || m),  m = (p̃_t + p̃_s)/2
截断误差由 `distillation_loss.topk` 控制 —— 越大越接近全词表 JSD; 与自研
脚本(全词表分块 JSD)对照时先跑一次 topk=V 的自检。

用法
----
launcher 最前面:
    import kgqa.rl.verl_plugins.jsd_loss as jsd; jsd.install()
配置: `distillation.distillation_loss.loss_mode: jsd`
自检: `python -m kgqa.rl.verl_plugins.jsd_loss`
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _as_dense(t: torch.Tensor) -> torch.Tensor:
    """thd 格式下 teacher 张量是 nested 形式; 展开成 (1, total, K)。"""
    if getattr(t, "is_nested", False):
        return t.values().unsqueeze(0)
    return t


def _slice_sp(t: torch.Tensor) -> torch.Tensor:
    """与 verl FSDP 实现对齐: 序列并行时切 teacher 张量。"""
    try:
        from verl.utils.ulysses import (
            get_ulysses_sequence_parallel_world_size,
            slice_input_tensor,
        )
        if get_ulysses_sequence_parallel_world_size() > 1:
            return slice_input_tensor(t, dim=1)
    except Exception:
        pass
    return t


def _student_topk_logprobs(
    student_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """log_softmax(logits).gather(topk_ids), 分块避免物化 [B,T,V]。"""
    b, t, v = student_logits.shape
    k = topk_ids.shape[-1]
    flat = student_logits.reshape(-1, v)
    ids = topk_ids.reshape(-1, k)
    out = torch.empty(ids.shape, dtype=torch.float32, device=student_logits.device)
    for s in range(0, flat.shape[0], chunk_size):
        e = min(s + chunk_size, flat.shape[0])
        lg = flat[s:e].float()
        log_z = torch.logsumexp(lg, dim=-1, keepdim=True)
        out[s:e] = lg.gather(-1, ids[s:e]) - log_z
    return out.reshape(b, t, k)


def compute_jsd_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config=None,
    data_format: str = "bshd",
) -> dict[str, torch.Tensor]:
    """logits processor 段: 返回与 compute_forward_kl_topk 相同的核心键。"""
    lp_t = _slice_sp(_as_dense(teacher_topk_log_probs)).float()
    ids = _slice_sp(_as_dense(teacher_topk_ids))
    lp_s = _student_topk_logprobs(student_logits, ids)
    assert lp_t.shape == lp_s.shape, f"{lp_t.shape=} != {lp_s.shape=}"
    assert lp_t.shape[:2] == student_logits.shape[:2], (
        f"{lp_t.shape[:2]=} != {student_logits.shape[:2]=}")

    t_mass = lp_t.exp().sum(dim=-1)
    s_mass = lp_s.exp().sum(dim=-1)
    lp_t_n = lp_t - torch.log(t_mass.clamp_min(1e-30)).unsqueeze(-1)
    lp_s_n = lp_s - torch.log(s_mass.clamp_min(1e-30)).unsqueeze(-1)

    # log 域 JSD (与自研脚本 logaddexp 实现同式, 全程有界)
    log_m = torch.logaddexp(lp_t_n, lp_s_n) - math.log(2.0)
    jsd = 0.5 * (lp_t_n.exp() * (lp_t_n - log_m)).sum(-1) \
        + 0.5 * (lp_s_n.exp() * (lp_s_n - log_m)).sum(-1)
    return {"distillation_losses": jsd,
            "student_mass": s_mass,
            "teacher_mass": t_mass}


def install() -> None:
    """注册 jsd 损失并给 logits processor 段加分派分支 (幂等)。"""
    import verl.trainer.distillation.losses as L

    if "jsd" not in L.DISTILLATION_LOSS_REGISTRY:
        # 最终损失段与 forward_kl_topk 完全相同: 只读 logits processor 产物
        L.register_distillation_loss(
            L.DistillationLossSettings(names="jsd", use_topk=True)
        )(L.compute_forward_kl_topk)

    if not getattr(L.compute_topk_loss, "_jsd_patched", False):
        orig = L.compute_topk_loss

        def compute_topk_loss(config, distillation_config, data,
                              student_logits, data_format):
            if distillation_config.distillation_loss.loss_mode == "jsd":
                return compute_jsd_topk(
                    student_logits=student_logits,
                    teacher_topk_log_probs=data["teacher_logprobs"],
                    teacher_topk_ids=data["teacher_ids"],
                    config=distillation_config,
                    data_format=data_format,
                )
            return orig(config, distillation_config, data,
                        student_logits, data_format)

        compute_topk_loss._jsd_patched = True
        L.compute_topk_loss = compute_topk_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    b, t, v = 2, 5, 512
    st = torch.randn(b, t, v)   # student logits
    te = torch.randn(b, t, v)   # teacher logits

    lp_t_full = F.log_softmax(te.float(), -1)
    lp_s_full = F.log_softmax(st.float(), -1)

    # 1) topk = V 时必须与全词表 JSD 逐值一致
    top_t = torch.topk(lp_t_full, k=v, dim=-1)
    got = compute_jsd_topk(st, top_t.values, top_t.indices)["distillation_losses"]
    m = 0.5 * (lp_t_full.exp() + lp_s_full.exp())
    ref = 0.5 * (lp_t_full.exp() * (lp_t_full - m.log())).sum(-1) \
        + 0.5 * (lp_s_full.exp() * (lp_s_full - m.log())).sum(-1)
    d_full = (got - ref).abs().max().item()

    # 2) 同分布 JSD = 0 (teacher 值必须按 ids 顺序 gather, 不能直接传整表)
    ids_s = torch.topk(lp_s_full, k=v, dim=-1).indices
    vals_s = torch.gather(lp_s_full, -1, ids_s)
    same = compute_jsd_topk(st, vals_s, ids_s)
    d_same = same["distillation_losses"].abs().max().item()

    # 3) 对称性: JSD(t||s) == JSD(s||t)
    fwd = compute_jsd_topk(st, top_t.values, top_t.indices)["distillation_losses"]
    top_s = torch.topk(lp_s_full, k=v, dim=-1)
    bwd = compute_jsd_topk(te, top_s.values, top_s.indices)["distillation_losses"]
    d_sym = (fwd - bwd).abs().max().item()

    print(f"[jsd self-test] full-vocab max|diff|={d_full:.2e} "
          f"same-dist={d_same:.2e} symmetry={d_sym:.2e}")
    assert d_full < 1e-5, d_full
    assert d_same < 1e-6, d_same
    assert d_sym < 1e-5, d_sym
