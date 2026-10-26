#!/usr/bin/env python3
"""F01/F10/F11 prototype: chunked LM-head GKD/JSD on the real Qwen3.5-4B.

Verifies, on one A100-40G, that computing the LM head per active-token chunk
(instead of materializing full [S,V] logits) is:
  1. numerically equivalent to the ms-swift baseline (full student logits +
     full teacher logits -> to_topk -> conditional JSD over K), and
  2. actually lower peak memory, with a working backward, and
  3. (F11) measures conditional top-k JSD vs full-vocab JSD on the same
     tokens, to inform the loss-target ruling.

Teacher differs from the student by a perturbed 128-token prefix (stands in
for the OPSD teacher reference) so the JSD is nonzero and the two paths can
be compared on a nontrivial signal. LoRA r16 trainable (matches the training
regime; full-param grads would add ~8G ZeRO-2/LoRA never materialize).

Single process, single GPU, synthetic sequence; no vLLM, no NCCL.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/diag_chunked_lmhead.py \
      [--seq 16384 --active 4096 --chunk 512 --topk 256 --skip-baseline]
"""
from __future__ import annotations

import argparse
import gc
import time

import torch
import torch.nn.functional as F

MODEL = "models/Qwen3.5-4B"
BETA = 0.5
TEMP = 1.0


def mem(tag):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[mem {tag}] alloc={alloc:.2f}G reserved={reserved:.2f}G "
          f"peak={peak:.2f}G", flush=True)
    return peak


def jsd_per_token(s_log, t_log, beta):
    """Log-domain JSD over the last dim, matching gkd_loss.jsd_loss."""
    log_beta = torch.log(torch.tensor(beta, device=s_log.device))
    log_1mb = torch.log1p(-torch.tensor(beta, device=s_log.device))
    log_m = torch.logaddexp(s_log + log_1mb, t_log + log_beta)
    return beta * (t_log.exp() * (t_log - log_m)).sum(-1) \
        + (1 - beta) * (s_log.exp() * (s_log - log_m)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--active", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()
    S, A, C, K = args.seq, args.active, args.chunk, args.topk

    from transformers import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True).cuda()
    model.config.use_cache = False
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    base = model.get_base_model()
    dec, lm_head = base.get_decoder(), base.get_output_embeddings()
    W = lm_head.weight
    V, H = W.shape
    print(f"model={type(base).__name__} V={V} H={H} S={S} active={A} "
          f"chunk={C} topk={K} LoRA-r16", flush=True)

    torch.manual_seed(0)
    input_ids = torch.randint(1000, 100000, (1, S), device="cuda")
    attn = torch.ones_like(input_ids)
    teacher_ids = input_ids.clone()
    teacher_ids[0, :128] = torch.randint(1000, 100000, (128,), device="cuda")
    labels = torch.full((1, S), -100, dtype=torch.long, device="cuda")
    labels[0, S - A - 1:S - 1] = input_ids[0, S - A:S]
    idx = (labels[0] != -100).nonzero(as_tuple=True)[0]
    assert idx.numel() == A

    def grad_sum():
        return sum(p.grad.abs().sum().item() for p in model.parameters()
                   if p.requires_grad and p.grad is not None)

    def clean():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    res = {}

    # ---------------- baseline: full [S,V] logits (ms-swift path) ----------
    if not args.skip_baseline:
        clean()
        t0 = time.time()
        s_logits_full = model(input_ids=input_ids,
                              attention_mask=attn).logits      # [1,S,V] grad
        with torch.no_grad():
            t_logits_full = model(input_ids=teacher_ids,
                                  attention_mask=attn).logits
        mem("baseline fwd")
        # faithful to ms-swift to_topk: torch.topk on bf16 logits; fp32 only
        # after selection. Free the teacher's full logits immediately —
        # keeping both [S,V] tensors alive IS the F01 pathology.
        with torch.no_grad():
            t_act = t_logits_full[0].index_select(0, idx)
            del t_logits_full
            t_vals, t_inds = t_act.topk(K, dim=-1)
            t_vals = t_vals.float()
            del t_act
        s_g = (s_logits_full[0].index_select(0, idx)
               .gather(-1, t_inds).float())
        s_l = F.log_softmax(s_g / TEMP, dim=-1)
        t_l = F.log_softmax(t_vals / TEMP, dim=-1)
        loss_base = jsd_per_token(s_l, t_l, BETA).mean()
        loss_base.backward()
        torch.cuda.synchronize()
        res["base"] = (loss_base.item(), grad_sum(), time.time() - t0,
                       mem("baseline bwd"))
        print(f"[base ] loss={res['base'][0]:.8f} gsum={res['base'][1]:.3e} "
              f"time={res['base'][2]:.1f}s", flush=True)
        del s_logits_full, s_g, s_l, t_l, t_vals, t_inds, loss_base
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        clean()

    # ------------- chunked: backbone hidden + per-chunk LM head ------------
    t0 = time.time()
    hs = dec(input_ids=input_ids, attention_mask=attn).last_hidden_state
    h_s = hs[0, :-1, :].index_select(0, idx)                   # [A,H] grad
    with torch.no_grad():
        ht = dec(input_ids=teacher_ids, attention_mask=attn).last_hidden_state
        h_t = ht[0, :-1, :].index_select(0, idx)
    mem("chunked hidden")

    def chunk_loss(hc_s, hc_t):
        t_logits = (hc_t @ W.t())
        tv, ti = t_logits.topk(K, dim=-1)
        s_g = (hc_s @ W.t()).gather(-1, ti).float()
        s_l = F.log_softmax(s_g / TEMP, dim=-1)
        t_l = F.log_softmax(tv.float() / TEMP, dim=-1)
        return jsd_per_token(s_l, t_l, BETA).sum()

    total = hs.new_zeros((), dtype=torch.float32)
    for s in range(0, A, C):
        e = min(s + C, A)
        total = total + torch.utils.checkpoint.checkpoint(
            chunk_loss, h_s[s:e], h_t[s:e], use_reentrant=False)
    loss_chunk = total / A
    loss_chunk.backward()
    torch.cuda.synchronize()
    res["chunk"] = (loss_chunk.item(), grad_sum(), time.time() - t0,
                    mem("chunked bwd"))
    print(f"[chunk] loss={res['chunk'][0]:.8f} gsum={res['chunk'][1]:.3e} "
          f"time={res['chunk'][2]:.1f}s", flush=True)

    if "base" in res:
        dloss = abs(res["base"][0] - res["chunk"][0])
        dg = abs(res["base"][1] - res["chunk"][1]) / max(res["base"][1], 1e-12)
        print(f"\n[result] |Δloss|={dloss:.3e}  |Δgrad|rel={dg:.3e}  "
              f"peak {res['base'][3]:.2f}G -> {res['chunk'][3]:.2f}G "
              f"({res['base'][3]/res['chunk'][3]:.2f}x)  "
              f"time {res['base'][2]:.1f}s -> {res['chunk'][2]:.1f}s",
              flush=True)

    # ------------------- F11: conditional vs full-vocab JSD ----------------
    with torch.no_grad():
        vals, cens, tm, sm = [], [], [], []
        for s in range(0, A, C):
            e = min(s + C, A)
            t_log = (h_t[s:e] @ W.t()).float().log_softmax(-1)
            s_log = (h_s[s:e] @ W.t()).float().log_softmax(-1)
            tv, ti = t_log.topk(K, dim=-1)
            s_l = F.log_softmax(s_log.gather(-1, ti) / TEMP, -1)
            t_l = F.log_softmax(tv / TEMP, -1)
            vals.append(jsd_per_token(s_l, t_l, BETA))
            cens.append(jsd_per_token(s_log / TEMP, t_log / TEMP, BETA))
            tm.append(tv.exp().sum(-1))
            sm.append(s_log.gather(-1, ti).exp().sum(-1))
        cond = torch.cat(vals).mean().item()
        full = torch.cat(cens).mean().item()
        t_mass = torch.cat(tm).mean().item()
        s_mass = torch.cat(sm).mean().item()
    print(f"[F11  ] conditional-topk-JSD={cond:.6f}  full-vocab-JSD={full:.6f}"
          f"  teacher_mass_in_K={t_mass:.4f}  student_mass_in_K={s_mass:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
