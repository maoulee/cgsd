"""JSD 蒸馏损失插件测试 (verl 对齐 S1, 2026-10-21).

契约 (与 verl compute_forward_kl_topk 的返回键/形状对齐):
  - topk = V 时与全词表 JSD 逐值一致 (<1e-5)
  - 同分布 JSD ≈ 0
  - 对称性
  - 截断支撑集上 JSD ∈ [0, log2]
  - install() 幂等且注册进 verl 的 DISTILLATION_LOSS_REGISTRY
"""

import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from kgqa.rl.verl_plugins.jsd_loss import compute_jsd_topk, install


def _ref_jsd(lp_t, lp_s):
    m = 0.5 * (lp_t.exp() + lp_s.exp())
    return (0.5 * (lp_t.exp() * (lp_t - m.log())).sum(-1)
            + 0.5 * (lp_s.exp() * (lp_s - m.log())).sum(-1))


def test_full_vocab_parity():
    torch.manual_seed(0)
    b, t, v = 2, 4, 256
    st, te = torch.randn(b, t, v), torch.randn(b, t, v)
    lp_t = F.log_softmax(te.float(), -1)
    top = torch.topk(lp_t, k=v, dim=-1)
    got = compute_jsd_topk(st, top.values, top.indices)["distillation_losses"]
    ref = _ref_jsd(lp_t, F.log_softmax(st.float(), -1))
    assert (got - ref).abs().max().item() < 1e-5


def test_same_distribution_zero():
    torch.manual_seed(1)
    st = torch.randn(2, 3, 128)
    lp = F.log_softmax(st.float(), -1)
    ids = torch.topk(lp, k=128, dim=-1).indices
    vals = torch.gather(lp, -1, ids)
    z = compute_jsd_topk(st, vals, ids)["distillation_losses"]
    assert z.abs().max().item() < 1e-6


def test_symmetry():
    torch.manual_seed(2)
    st, te = torch.randn(2, 3, 64), torch.randn(2, 3, 64)
    lp_s, lp_t = F.log_softmax(st.float(), -1), F.log_softmax(te.float(), -1)
    a = compute_jsd_topk(st, torch.topk(lp_t, k=64, dim=-1).values,
                         torch.topk(lp_t, k=64, dim=-1).indices)["distillation_losses"]
    b = compute_jsd_topk(te, torch.topk(lp_s, k=64, dim=-1).values,
                         torch.topk(lp_s, k=64, dim=-1).indices)["distillation_losses"]
    assert (a - b).abs().max().item() < 1e-5


def test_truncated_support_bounded():
    """截断到 top-k 时 JSD 仍在 [0, log2] (归一化后最大为 log2)。"""
    torch.manual_seed(3)
    st, te = torch.randn(1, 2, 1024), torch.randn(1, 2, 1024)
    lp_t = F.log_softmax(te.float(), -1)
    top = torch.topk(lp_t, k=16, dim=-1)
    jsd = compute_jsd_topk(st, top.values, top.indices)["distillation_losses"]
    assert (jsd >= -1e-6).all()
    assert (jsd <= math.log(2) + 1e-5).all()


def test_install_registers_and_is_idempotent():
    pytest.importorskip("verl")
    install()
    install()
    import verl.trainer.distillation.losses as L
    assert "jsd" in L.DISTILLATION_LOSS_REGISTRY
    assert getattr(L.compute_topk_loss, "_jsd_patched", False)
