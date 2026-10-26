#!/usr/bin/env python3
"""Team9 two-GPU e2e simulation — the FULL manual loop (degrade B).

rollout (our SEQ state machine, lockstep mode, in-process 4B engine, GPU0)
  → judge (structure layer via scripts/annotate_layer_ops + probability arm
    via engine.gold_logprob_batch, wave-batched reverse removal)
  → turn-level advantage assembly (kgqa.rl.turn_credit, λ=0.2)
  → LoRA training step (peft, dual-head pre-declared-denominator loss, GPU1)
  → adapter handoff (save_pretrained → engine-level mount, old one unmounted)
  → next batch.  Max 10 steps, NO checkpoints, monitors F1 / G1 / G2 / timing
  / VRAM per step.

Env: qwen35 (vllm 0.25.1). Run:

  CUDA_VISIBLE_DEVICES=0,1 /opt/conda/envs/rl/bin/python \
      -m kgqa.rl.sim_loop > tmp/team9/logs/sim_loop.log 2>&1

Tunables via env: SIM_STEPS(10) N_CASES(10) G(4) TEMP(0.3) LAMBDA(0.2)
  LR(1e-5) LORA_RANK(32) MAX_ROUNDS(16) THINK_BUDGET(1000)
  SLEEP_EACH_STEP(1) OUT_DIR(tmp/team9)
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "./scripts")

# ── canonical hp-recipe env (SESSION_MEMORY §0) — set BEFORE kgqa.agent import
_ENV = {
    "PYTHONHASHSEED": "0",
    "SEQ_ADAPTIVE_ROUTES": "1", "SEQ_LICENSE_FILTER": "1",
    "SEQ_MULTISTEP": "1", "SEQ_PLAN_PATH_HINTS": "1", "SEQ_PROMPT": "C123",
    "SEQ_RENDER_V38": "1", "SEQ_RENDER_V4": "1",
    "WALK_BATCH_WINDOW": "1.0", "WALK_POOL": "3",
    "SEQ_EVIDENCE_SIDECAR": "1",
}
for _k, _v in _ENV.items():
    os.environ.setdefault(_k, _v)

import aiohttp  # noqa: E402

from kgqa.rl.sim_engine import SimEngine  # noqa: E402

PKL = {"test_v4": "./data/cwq_processed/test_v4_repaired.pkl"}


def _gpu_mib(idx: int) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"]).decode().strip().split("\n")
        return int(out[idx].strip())
    except Exception:
        return -1


# ══════════════════════════════ rollout ══════════════════════════════

def load_cases(split: str, n: int):
    import pickle
    from pathlib import Path
    samples = pickle.loads(Path(PKL[split]).read_bytes())
    pool = [s for s in samples if s.get("a_entity")]
    return pool[:n]


async def rollout_step(engine: SimEngine, cases, n_samples: int, temp: float,
                       thinking_budget: int, max_rounds: int,
                       lora_name: str | None):
    """Lockstep SEQ rollout with per-turn §4 snapshots.

    Mirrors kgqa/rl/seq_rollout.rollout()'s lockstep branch (bubble OFF):
    same hint assembly, repeat nudges, final-turn +1, rescue, extraction.
    Additionally captures (prompt_ids, gen_ids, gen_logprobs) per turn.
    """
    from kgqa.agent.seq_react_loop import SeqReactCase, run_round_dispatch
    from kgqa.agent.loop import _ctx_to_result_dict
    from kgqa.agent.seq_adaptive import attach_result
    from kgqa.core.utils import phase_timer

    T0 = time.time()
    pilot_rows = [{"case_id": s["id"], "question": s["question"],
                   "gt_answers": s.get("a_entity", [])} for s in cases]
    react_cases = []
    for ci, (s, pr) in enumerate(zip(cases, pilot_rows)):
        for gi in range(n_samples):
            react_cases.append((ci, SeqReactCase(s, pr, ci)))
    snaps: dict[int, list] = {id(rc): [] for _, rc in react_cases}
    print(f"  expanded {len(cases)} cases × {n_samples} = "
          f"{len(react_cases)} trajectories", flush=True)

    max_tokens = thinking_budget + 2560

    async with aiohttp.ClientSession() as session:
        pl_sem = asyncio.Semaphore(16)

        async def _pl(rc):
            async with pl_sem:
                await rc.prelink(session)

        await asyncio.gather(*[_pl(rc) for _, rc in react_cases])
        t_admit = time.time()
        for _, rc in react_cases:
            rc._admit_t = t_admit

        async def _round(active):
            prompts = []
            for rc in active:
                rc.ctx._seq_turns_remaining = max_rounds
                hint = rc.state_aware_hint() + rc.allowed_tools_hint()
                if rc._tool_repeat >= 2:
                    hint = rc.loop_nudge() + "\n" + hint
                msgs = list(rc.messages)
                msgs.append({"role": "user", "content": hint})
                prompts.append(msgs)
            with phase_timer("llm"):
                batch = engine.chat_batch_snap(
                    prompts, thinking_budget=thinking_budget,
                    temperature=temp, max_tokens=max_tokens,
                    lora_name=lora_name)
            for rc, s in zip(active, batch):
                snaps[id(rc)].append(s)
            pairs = [(rc, s.text, s.reasoning)
                     for rc, s in zip(active, batch)]
            with phase_timer("dispatch"):
                await run_round_dispatch(pairs, session)

        for rnd in range(max_rounds):
            active = [rc for _, rc in react_cases if rc.is_active]
            if not active:
                break
            await _round(active)
            done = sum(1 for _, rc in react_cases if rc.done)
            failed = sum(1 for _, rc in react_cases if rc.failed)
            print(f"    round {rnd+1}: {len(active)} active → done={done} "
                  f"failed={failed}  [t={time.time()-T0:.0f}s]", flush=True)

        # budget+1 final turn (user ruling 2026-08-26)
        finalists = [rc for _, rc in react_cases if rc.is_active]
        if finalists:
            print(f"    final-turn +1 for {len(finalists)} cases", flush=True)
            prompts = []
            for rc in finalists:
                hint = ("FINAL TURN (budget extended by one): submit your "
                        "final answer NOW with `tool: answer` from the "
                        "bindings you hold — this is the LAST turn; further "
                        "retrieval will not be executed. "
                        + rc.allowed_tools_hint())
                msgs = list(rc.messages)
                msgs.append({"role": "user", "content": hint})
                prompts.append(msgs)
            with phase_timer("llm"):
                batch = engine.chat_batch_snap(
                    prompts, thinking_budget=thinking_budget,
                    temperature=temp, max_tokens=max_tokens,
                    lora_name=lora_name)
            for rc, s in zip(finalists, batch):
                snaps[id(rc)].append(s)
            with phase_timer("dispatch"):
                await run_round_dispatch(
                    [(rc, s.text, s.reasoning)
                     for rc, s in zip(finalists, batch)], session)

    # extract results (same row shape as seq_rollout, minus reward fields)
    out = []
    for i, (ci, rc) in enumerate(react_cases):
        rc.rescue_terminal_answer()
        if getattr(rc, "_done_t", None) is None:
            rc._done_t = time.time()
        r = _ctx_to_result_dict(rc.ctx, rc.state, rc.failed, rc.failure_reason)
        attach_result(r, rc.ctx)
        answer = r.get("llm_answer_str", "") or r.get("llm_answer", "") or ""
        out.append({
            "case_id": cases[ci]["id"], "case_idx": ci,
            "sample_idx": i % n_samples,
            "question": cases[ci]["question"],
            "gold": cases[ci].get("a_entity", []),
            "answer": answer, "f1": round(float(r.get("llm_f1", 0) or 0), 3),
            "hit": bool(r.get("llm_hit", False)),
            "failed": bool(rc.failed),
            "failure_reason": rc.failure_reason,
            "n_turns": sum(1 for st in rc.ctx.trajectory
                           if st.get("role") == "assistant"),
            "trajectory": rc.ctx.trajectory,
            "_snaps": snaps[id(rc)],
        })
    return out


# ═════════════════════ judge (structure + probability arm) ═════════════════════

def judge_step(engine: SimEngine, results: list[dict], out_dir: str, step: int):
    """Structure layer (annotate_layer_ops) + probability arm (wave-batched
    port of scripts/prob_arm_pathway.py via in-process gold_logprob_batch)."""
    import contextlib
    import io
    import annotate_layer_ops as A

    path = os.path.join(out_dir, f"rollout_{step}.json")
    slim = [{k: v for k, v in r.items() if not k.startswith("_")}
            for r in results]
    with open(path, "w") as f:
        json.dump(slim, f, ensure_ascii=False)

    recs = {(rec["case_id"], str(rec.get("sample_idx"))): rec
            for rec in slim}
    for rec in recs.values():          # annotate_layer_ops expects _gold_list
        gl = rec["gold"] if isinstance(rec["gold"], list) else [rec["gold"]]
        rec["_gold_list"] = gl
    with contextlib.redirect_stderr(io.StringIO()):
        t0 = time.time()
        ops_all, cases = A.annotate(path)
        A.mark_necessity(cases, recs)
        A.mark_break_points(cases, PKL["test_v4"])
        A.reconcile_necessity(cases)
        A.mark_removable(cases, recs)
        t_struct = time.time() - t0
    blocks = run_prob_arm(engine, cases, recs)
    return (blocks, path, {"struct_s": round(t_struct, 1),
                           "probarm_s": round(time.time() - t0 - t_struct, 1)})


TAU = 0.05
FLOOR = 0.05


def run_prob_arm(engine: SimEngine, cases, recs):
    """Wave-batched reverse dynamic removal — semantics identical to
    scripts/prob_arm_pathway.py (per-pathway isolation, per-gold vectors,
    decayed threshold, env-repeat kill, breadth rule), probes batched across
    trajectories per wave (each generator has ≤1 outstanding probe)."""

    def case_gen(c):
        rec = recs[(c["case_id"], str(c["sample"]))]
        if not (rec.get("f1") or 0) > 0:
            yield {"blocks": [{"idx": b["idx"], "verdict": "不评估(f1=0)",
                               "moved": [], "max_delta": None}
                              for b in sorted(c["blocks"],
                                              key=lambda b: b["idx"])],
                   "f1": rec.get("f1"), "golds": [], "p0_vec": [],
                   "pathways": []}
            return
        q = rec.get("question", "")
        golds = sorted({str(g) for g in (rec["gold"] or [])
                        if str(g).strip()}) or ["unknown"]
        blocks = sorted(c["blocks"], key=lambda b: b["idx"])
        tr = rec["trajectory"]
        round_of = {}
        for b in blocks:
            act = tr[b["idx"] - 1] if b["idx"] > 0 and \
                tr[b["idx"] - 1].get("role") == "assistant" else {}
            round_of[b["idx"]] = (
                f"[action] {act.get('content') or ''}\n"
                f"[result] {tr[b['idx']].get('content') or ''}")
        p0_vec = yield ("probe", q, "", golds)
        by_pw: dict[str, list] = {}
        for b in blocks:
            by_pw.setdefault(b.get("pathway") or "r?", []).append(b)
        out_blocks, pw_summary = [], []
        for tid, bs in by_pw.items():
            texts = [round_of[b["idx"]] for b in bs]
            ev_all = "\n\n".join(t for t in texts if t)
            ref_vec = yield ("probe", q, ev_all, golds)
            start_ref = list(ref_vec)
            alive = list(range(len(texts)))
            for k in range(len(texts) - 1, -1, -1):
                if "UNCHANGED EVIDENCE (repeat)" in texts[k]:
                    out_blocks.append({"idx": bs[k]["idx"],
                                       "verdict": "可移除-环境重复(判死)",
                                       "moved": [], "max_delta": None})
                    alive = [j for j in alive if j != k]
                    continue
                if bs[k].get("necessary") == "yes" \
                        or bs[k].get("necessary_disc") == "yes":
                    # tiered judging (user ruling 2026-10-11 裁定①):
                    # structure-necessary blocks skip the probability arm
                    # entirely — structure layer already adjudicated them;
                    # they stay in `alive` (always part of evidence).
                    out_blocks.append({"idx": bs[k]["idx"],
                                       "verdict": "有效-结构(必要)",
                                       "moved": [], "max_delta": None})
                    continue
                cf_ev = "\n\n".join(texts[j] for j in alive if j != k)
                cf_vec = yield ("probe", q, cf_ev, golds)
                deltas = [None if (r is None or f_ is None) else r - f_
                          for r, f_ in zip(ref_vec, cf_vec)]
                low_mass = start_ref and max(
                    (x for x in start_ref if x is not None),
                    default=0) < FLOOR
                shares = []
                if low_mass:
                    shares = [None if (r is None or f_ is None
                                       or p0v is None or r - p0v <= 1e-9)
                              else (r - f_) / (r - p0v)
                              for r, f_, p0v in zip(ref_vec, cf_vec, p0_vec)]
                share_moved = [(golds[i], sh) for i, sh in enumerate(shares)
                               if sh is not None and sh > 0.5]

                def _tau_of(base):
                    return max(TAU * (1.0 - (base or 0.0)), 0.03)
                hits = [(golds[i], d) for i, d in enumerate(deltas)
                        if d is not None and d > _tau_of(ref_vec[i])]
                ups = [(golds[i], d) for i, d in enumerate(deltas)
                       if d is not None and d > 0]
                downs_n = sum(1 for d in deltas
                              if d is not None and d < 0)
                if hits:
                    out_blocks.append({"idx": bs[k]["idx"],
                                       "verdict": "有效-概率(通路内)",
                                       "moved": [(g, round(d, 3))
                                                 for g, d in hits],
                                       "max_delta": round(
                                           max(d for _, d in hits), 3)})
                elif len(ups) >= 2 and len(ups) > downs_n:
                    out_blocks.append({"idx": bs[k]["idx"],
                                       "verdict": "有效-概率(通路内·多数上升)",
                                       "moved": [(g, round(d, 3))
                                                 for g, d in ups],
                                       "max_delta": round(
                                           max(d for _, d in ups), 3)})
                elif share_moved:
                    out_blocks.append({"idx": bs[k]["idx"],
                                       "verdict": "有效-概率(通路内·质量归因)",
                                       "moved": [(g, round(sh, 2))
                                                 for g, sh in share_moved],
                                       "max_delta": round(
                                           max(sh for _, sh in share_moved),
                                           2)})
                else:
                    mx = max((d for d in deltas if d is not None),
                             default=None)
                    out_blocks.append({"idx": bs[k]["idx"],
                                       "verdict": "可移除-概率(通路内)",
                                       "moved": [],
                                       "max_delta": round(mx, 3)
                                       if mx is not None else None})
                    ref_vec = list(cf_vec)
                    alive = [j for j in alive if j != k]
            gains = [None if (r is None or p is None) else r - p
                     for r, p in zip(start_ref, p0_vec)]
            pw_summary.append({
                "pathway": tid, "n_blocks": len(bs),
                "gain_vec_max": round(max((g for g in gains
                                           if g is not None), default=0.0),
                                      3)})
        yield {"blocks": sorted(out_blocks, key=lambda b: b["idx"]),
               "f1": rec.get("f1"), "golds": golds,
               "p0_vec": [round(x, 3) if x is not None else None
                          for x in p0_vec],
               "pathways": pw_summary}

    pending: dict[tuple, tuple] = {}
    results_out: dict[tuple, dict] = {}
    for c in cases:
        key = (c["case_id"], str(c["sample"]))
        g = case_gen(c)
        pending[key] = (g, next(g))
    wave = 0
    while pending:
        wave += 1
        probes, done = [], []
        for key, (g, req) in list(pending.items()):
            if isinstance(req, dict):
                results_out[key] = req.get("blocks", [])
                done.append(key)
            else:
                probes.append((key, req))
        for key in done:
            del pending[key]
        if not probes:
            break
        triples = [(req[1], req[2], req[3]) for _, req in probes]
        flat = [(q, g) for q, _, gs in triples for g in gs]
        t0 = time.time()
        lps = engine.gold_logprob_batch(flat)
        vecs, i = [], 0
        for _, _, gs in triples:
            vecs.append([None if lps[i + j] is None else math.exp(lps[i + j])
                         for j in range(len(gs))])
            i += len(gs)
        t_probe = time.time() - t0
        for (key, _), vec in zip(probes, vecs):
            g, req = pending[key]
            try:
                nxt = g.send(vec)
            except StopIteration:
                nxt = {}
            pending[key] = (g, nxt)
        print(f"    prob-arm wave {wave}: {len(flat)} probes "
              f"({t_probe:.0f}s)", flush=True)
    return results_out


# ═════════════════════ advantage assembly + guards ═════════════════════

def assemble_advantages(results, blocks_by_key, lam: float):
    """turn_credit coefficients per trajectory + G1/G2 guard stats."""
    from kgqa.rl.turn_credit import build_turn_coefficients, \
        action_tool_of_turn

    group_f1s: dict[int, list] = {}
    for r in results:
        group_f1s.setdefault(r["case_idx"], []).append(r["f1"])

    rows, guards = [], {"g1": {}, "g2": {}}
    n_allzero = n_nonzero = n_groups_answer = 0
    n_answer_turns = n_explore_turns = 0
    sg_actions_total = judged_blocks_total = align_drops = 0

    for r in results:
        key = (r["case_id"], str(r["sample_idx"]))
        blocks = blocks_by_key.get(key, [])
        coeffs, prov = build_turn_coefficients(
            r["trajectory"], blocks, r["f1"],
            group_f1s.get(r["case_idx"], [r["f1"]]), lam=lam)
        if any(abs(c) > 1e-9 for c in coeffs):
            n_nonzero += 1
        else:
            n_allzero += 1
        if any(p.get("label") == "answer_group" and abs(p["coef"]) > 1e-9
               for p in prov):
            n_groups_answer += 1
        sg_n = sum(1 for i, st in enumerate(r["trajectory"])
                   if st.get("role") == "assistant"
                   and action_tool_of_turn(r["trajectory"], i)
                   == "retrieve_subgraph")
        sg_actions_total += sg_n
        judged_blocks_total += len(blocks)

        # snapshots ↔ assistant turns alignment (content prefix, in order)
        snaps = r.get("_snaps", [])
        asst = [i for i, st in enumerate(r["trajectory"])
                if st.get("role") == "assistant"]
        used, match = set(), []
        for ai in asst:
            content = (r["trajectory"][ai].get("content") or "").strip()
            found = None
            if content:
                for k, s in enumerate(snaps):
                    if k not in used and content[:80] in s.raw_text:
                        found = k
                        break
            used.add(found) if found is not None else None
            match.append(found)
        if len(snaps) != len(asst):
            align_drops += 1

        for j, (ai, k) in enumerate(zip(asst, match)):
            if k is None:
                continue
            label = prov[j]["label"]
            head = ("answer" if label == "answer_group"
                    else "explore" if label == "effective" else "zero")
            if head == "answer":
                n_answer_turns += 1
            elif head == "explore":
                n_explore_turns += 1
            rows.append({
                "case_idx": r["case_idx"], "sample_idx": r["sample_idx"],
                "turn": j, "head": head, "coef": float(coeffs[j]),
                "label": label,
                "prompt_ids": snaps[k].prompt_ids,
                "gen_ids": snaps[k].gen_ids,
                "gen_logprobs": snaps[k].gen_logprobs,
            })

    f1s = [r["f1"] for r in results]
    guards["g1"] = {
        "n_traj": len(results), "n_allzero_adv": n_allzero,
        "n_nonzero_adv": n_nonzero,
        "allzero_frac": round(n_allzero / max(len(results), 1), 3),
        "groups_with_answer_signal": n_groups_answer,
        "n_groups": len(group_f1s),
        "answer_head_nonzero_frac": round(
            n_groups_answer / max(len(results), 1), 3),
        "lambda_turns": n_explore_turns,
        "n_answer_turns": n_answer_turns, "n_explore_turns": n_explore_turns,
    }
    guards["g2"] = {
        "sg_actions": sg_actions_total, "judged_blocks": judged_blocks_total,
        "no_evidence_bucket": sg_actions_total - judged_blocks_total,
        "snap_align_drops": align_drops,
    }
    return rows, guards, {
        "f1_mean": round(sum(f1s) / max(len(f1s), 1), 4),
        "f1_per_case": {str(k): [r["f1"] for r in results
                                 if r["case_idx"] == k]
                        for k in sorted(group_f1s)},
        "n_hit": sum(1 for r in results if r["hit"]),
        "mean_turns": round(sum(r["n_turns"] for r in results)
                            / max(len(results), 1), 2),
        "n_failed": sum(1 for r in results if r["failed"]),
    }


# ═════════════════════ LoRA trainer (dual head) ═════════════════════

class LoRATrainer:
    """peft LoRA on the text backbone + dual-head PPO-style loss.

    Per codex plan §3: NO full-sequence logits — backbone hidden states,
    then checkpointed 128-position chunks through the (frozen) LM head.
    Per teamB2 §2c: both heads reduce with the SAME pre-declared denominator
    (total generated tokens of the batch — zero-coefficient turns included).
    Single on-policy pass per batch (ppo_epochs=1 semantics).
    """

    def __init__(self, model_path="models/Qwen3.5-4B", rank=32,
                 alpha=64, lr=1e-5, device="cuda:1", max_len=32768,
                 clip_ratio=0.2):
        import torch
        from transformers import AutoConfig, Qwen3_5ForCausalLM
        # explicit TEXT config: inside the engine-loaded process, Auto* config
        # normalization returned the composite Qwen3_5Config (no flat
        # vocab_size) — sidestep deterministically (works in fresh processes
        # too).
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(cfg, "text_config"):
            cfg = cfg.text_config
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_path, config=cfg, dtype=torch.bfloat16,
            trust_remote_code=True)
        model = model.to(device)
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        for p in model.parameters():
            p.requires_grad_(False)

        from peft import LoraConfig, get_peft_model
        names = set()
        for n_, m_ in model.named_modules():
            if "visual" in n_ or "embed_tokens" in n_ or "lm_head" in n_:
                continue
            if isinstance(m_, torch.nn.Linear):
                names.add(n_.split(".")[-1])
        self.target_modules = sorted(names)
        lcfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                          target_modules=self.target_modules, bias="none",
                          task_type="CAUSAL_LM")
        self.model = get_peft_model(model, lcfg)
        self.model.print_trainable_parameters()
        self.device = device
        self.max_len = max_len
        self.clip_ratio = clip_ratio
        import torch.optim as optim
        self.opt = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, weight_decay=0.0)

    def train_rows(self, rows: list[dict]) -> dict:
        import torch
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint

        fit = [r for r in rows
               if len(r["prompt_ids"]) + len(r["gen_ids"]) <= self.max_len]
        decl = sum(len(r["gen_ids"]) for r in fit)      # pre-declared denom
        stats = {"n_rows": len(fit), "rows_dropped_long": len(rows) - len(fit),
                 "decl_tokens": decl,
                 "tokens_answer": sum(len(r["gen_ids"]) for r in fit
                                      if r["head"] == "answer"),
                 "tokens_explore": sum(len(r["gen_ids"]) for r in fit
                                       if r["head"] == "explore")}
        if decl == 0 or not fit:
            stats["skipped"] = True
            return stats
        m = self.model.base_model.model      # Qwen3_5ForConditionalGeneration
        backbone, lm_head = m.model, m.lm_head

        def _chunk_logp(h_chunk, w, tgt_chunk):
            logits = F.linear(h_chunk, w).float()      # [C, V] transient
            return logits.log_softmax(-1).gather(
                -1, tgt_chunk.unsqueeze(-1)).squeeze(-1)

        t0 = time.time()
        self.opt.zero_grad(set_to_none=True)
        loss_ans = loss_exp = 0.0
        clip_events = 0
        n_gen_tok = 0
        for r in fit:
            ids = r["prompt_ids"] + r["gen_ids"]
            n_gen = len(r["gen_ids"])
            x = torch.tensor([ids], device=self.device)
            pos = torch.arange(len(ids) - n_gen, len(ids),
                               device=self.device)
            tgt = torch.tensor(r["gen_ids"], device=self.device)
            old_lp = (torch.tensor(r["gen_logprobs"], device=self.device)
                      if r.get("gen_logprobs") else None)
            adv = r["coef"]
            out = backbone(input_ids=x, use_cache=False)
            h = out.last_hidden_state[0]
            lps = [checkpoint(_chunk_logp, h[pos[s:s + 128]], lm_head.weight,
                              tgt[s:s + 128], use_reentrant=False)
                     for s in range(0, n_gen, 128)]
            logp = torch.cat(lps)
            with torch.no_grad():
                if r["head"] == "answer":
                    loss_ans += float((-adv * logp).sum()) / decl
                elif r["head"] == "explore":
                    loss_exp += float((-adv * logp).sum()) / decl
            if old_lp is not None and old_lp.shape == logp.shape:
                ratio = torch.exp(logp - old_lp.float())
                pg = -adv * ratio
                pg_clip = -adv * torch.clamp(ratio, 1 - self.clip_ratio,
                                             1 + self.clip_ratio)
                clip_events += int((ratio < 1 - self.clip_ratio).sum()
                                   + (ratio > 1 + self.clip_ratio).sum())
                tok_loss = torch.maximum(pg, pg_clip)
            else:
                tok_loss = -adv * logp
            (tok_loss.sum() / decl).backward()
            n_gen_tok += n_gen
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        stats.update({"loss_answer": round(loss_ans, 6),
                      "loss_explore": round(loss_exp, 6),
                      "grad_norm": round(float(gn), 4),
                      "clip_events": clip_events,
                      "gen_tokens": n_gen_tok,
                      "train_s": round(time.time() - t0, 1)})
        return stats

    def load_adapter_state(self, path: str):
        """Resume the trainer LoRA from a saved adapter dir, so trainer
        state == rollout policy (keeps PPO ratios on-policy across the
        team9→team11 handoff).

        Direct per-tensor copy into module.lora_{A,B}['default'].weight —
        peft's set_peft_model_state_dict silently no-ops on this stack
        (transformers 5.14 conversion path drops every key without raising;
        verified: delta_norm stays 0)."""
        import torch
        import safetensors.torch as st
        sd = st.load_file(os.path.join(path, "adapter_model.safetensors"))
        n = 0
        for k, v in sd.items():
            body, comp = k.rsplit(".lora_", 1)
            side = comp.split(".")[0]               # "A" | "B"
            mod = self.model.get_submodule(body)    # wrapped base Linear
            layer = getattr(mod, f"lora_{side}")["default"]
            if layer.weight.shape != v.shape:
                raise RuntimeError(
                    f"adapter resume shape mismatch {k}: "
                    f"file {tuple(v.shape)} vs model "
                    f"{tuple(layer.weight.shape)}")
            with torch.no_grad():
                layer.weight.copy_(v.to(layer.weight.dtype))
            n += 1
        if n != len(sd):
            raise RuntimeError(f"adapter resume incomplete: {n}/{len(sd)}")

    def adapter_delta_norm(self) -> float:
        s = 0.0
        for n_, p in self.model.named_parameters():
            if p.requires_grad and "lora_B" in n_:
                s += float(p.detach().float().pow(2).sum())
        return round(s ** 0.5, 6)

    def save_adapter(self, path: str):
        self.model.save_pretrained(path)


# ═════════════════════ main loop ═════════════════════

def _verdict_buckets(blocks_by_key) -> dict:
    vc: dict[str, int] = {}
    for blks in blocks_by_key.values():
        for b in blks:
            v = ("有效-结构" if "有效-结构" in b["verdict"]
                 else "有效" if "有效" in b["verdict"]
                 else "可移除" if "可移除" in b["verdict"] else "不评估")
            vc[v] = vc.get(v, 0) + 1
    return vc


def main():
    # team11 run: steps START_STEP..SIM_STEPS-1 (batch 0 was rolled out and
    # trained by team9 → rollout_0.json + adapters/policy_0 are reused).
    steps = int(os.environ.get("SIM_STEPS", "10"))       # exclusive bound
    start_step = int(os.environ.get("START_STEP", "1"))
    n_cases = int(os.environ.get("N_CASES", "10"))
    G = int(os.environ.get("G", "4"))
    temp = float(os.environ.get("TEMP", "0.3"))
    lam = float(os.environ.get("LAMBDA", "0.2"))
    lr = float(os.environ.get("LR", "1e-5"))
    rank = int(os.environ.get("LORA_RANK", "32"))
    max_rounds = int(os.environ.get("MAX_ROUNDS", "16"))
    tb = int(os.environ.get("THINK_BUDGET", "1000"))
    out_dir = os.environ.get("OUT_DIR", "./tmp/team11")
    adapt_dir = os.path.join(out_dir, "adapters")
    shm_dir = os.environ.get("SHM_DIR", "/dev/shm/team11_adapters")
    init_adapter = os.environ.get(
        "INIT_ADAPTER", "./tmp/team9/adapters/policy_0")
    judge_batch0 = os.environ.get("JUDGE_BATCH0", "1") == "1"
    mount_src = os.environ.get("MOUNT_SRC", "shm")   # shm | nfs (per-step)
    os.makedirs(adapt_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    from kgqa.agent.seq_react_loop import seq_agents_md
    import hashlib
    ph = hashlib.sha256(seq_agents_md().encode()).hexdigest()[:16]

    print("═══ team11 full-loop run (team9 loop + team10 wake fix) ═══",
          flush=True)
    print(f"  steps {start_step}..{steps-1} cases={n_cases} G={G} "
          f"temp={temp} λ={lam} lr={lr} rank={rank} prompt_sha={ph}",
          flush=True)
    print(f"  init_adapter={init_adapter} mount_src={mount_src} "
          f"judge_batch0={judge_batch0} shm_dir={shm_dir}", flush=True)
    print("  env: " + " ".join(f"{k}={os.environ[k]}" for k in sorted(_ENV)),
          flush=True)

    import requests
    try:
        requests.post("http://localhost:8003/retrieve",
                      json={"query": "x", "candidates": ["a"], "top_k": 1},
                      timeout=10)
        print("  GTE :8003 up", flush=True)
    except Exception as e:
        raise SystemExit(f"GTE :8003 down: {e}")

    engine_dev = int(os.environ.get("ENGINE_DEV", "0"))
    train_dev = int(os.environ.get("TRAIN_DEV", "1"))
    print(f"  engine=GPU{engine_dev} train=GPU{train_dev}", flush=True)

    cases = load_cases("test_v4", n_cases)
    print(f"  loaded {len(cases)} cases: {[c['id'] for c in cases]}",
          flush=True)

    def M(m):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    t_boot = time.time()
    trainer = LoRATrainer(lr=lr, rank=rank,
                          device=f"cuda:{0 if train_dev == 0 else 1}")
    if init_adapter and os.path.isdir(init_adapter):
        trainer.load_adapter_state(init_adapter)
        dn = trainer.adapter_delta_norm()
        if dn <= 0:
            raise SystemExit("trainer adapter resume failed (delta_norm==0)")
        print(f"  trainer resumed from policy_0 (delta_norm={dn})",
              flush=True)
    print(f"  trainer ready on {trainer.device} "
          f"(GPU{train_dev} {_gpu_mib(train_dev)} MiB, "
          f"targets={trainer.target_modules})", flush=True)

    engine = SimEngine()
    print(f"  engine loaded in {engine.load_s:.0f}s "
          f"(GPU{engine_dev} now {_gpu_mib(engine_dev)} MiB)", flush=True)
    if not engine.verify_template_once():
        raise SystemExit("template mismatch — training rows would be wrong")
    engine.reset_prefix_cache()

    # ── boot mount A/B: SAME adapter bytes via NFS path vs tmpfs copy ──
    # separates file-load cost by source; tmpfs == RAM ("内存直挂" as close
    # as LoRARequest's path-only API allows in vllm 0.25.1).
    current = None
    mount_paths: dict[str, str] = {}
    if init_adapter and os.path.isdir(init_adapter):
        bm = {}
        t0 = time.time()
        engine.mount_adapter("policy_0", init_adapter)
        bm["nfs_mount_s"] = round(time.time() - t0, 2)
        mount_paths["policy_0"] = init_adapter
        current = "policy_0"
        engine.reset_prefix_cache()
        bm["sanity_nfs"] = engine.sanity_gen("policy_0")[:40]
        print(f"  [mount-A/B] NFS mount: {bm['nfs_mount_s']}s  "
              f"sanity={bm['sanity_nfs']!r}", flush=True)
        os.makedirs(shm_dir, exist_ok=True)
        shm0 = os.path.join(shm_dir, "policy_0")
        t0 = time.time()
        shutil.copytree(init_adapter, shm0, dirs_exist_ok=True)
        bm["copy_to_shm_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        engine.mount_adapter("policy_0_shm", shm0)
        bm["shm_mount_s"] = round(time.time() - t0, 2)
        bm["sanity_shm"] = engine.sanity_gen("policy_0_shm")[:40]
        engine.unmount_adapter("policy_0_shm")
        shutil.rmtree(shm0, ignore_errors=True)
        print(f"  [mount-A/B] tmpfs mount: {bm['shm_mount_s']}s "
              f"(copy {bm['copy_to_shm_s']}s) sanity={bm['sanity_shm']!r}"
              f"  → twin unmounted", flush=True)
        if bm["nfs_mount_s"] < 30 and bm["shm_mount_s"] < 30:
            # warm reads fast → chase the smoke's 488/517s with a genuinely
            # COLD NFS read: fresh copy, then drop its pages from the client
            # cache (per-file fadvise — no global cache drop).
            cold = os.path.join(adapt_dir, "policy_0_coldprobe")
            shutil.copytree(init_adapter, cold, dirs_exist_ok=True)
            st_file = os.path.join(cold, "adapter_model.safetensors")
            fd = os.open(st_file, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            t0 = time.time()
            engine.mount_adapter("policy_0_cold", cold)
            bm["cold_nfs_mount_s"] = round(time.time() - t0, 2)
            engine.unmount_adapter("policy_0_cold")
            shutil.rmtree(cold, ignore_errors=True)
            print(f"  [mount-A/B] cold-NFS mount (fadvise-dropped cache): "
                  f"{bm['cold_nfs_mount_s']}s", flush=True)
        M({"boot_mount_ab": bm})

    # ── batch-0 judge reconstruction (rollout_0.json reuse, saves a step;
    #    no _snaps → no train, this only fills step-0 monitor + validates
    #    the judge path incl. the structure-necessary skip) ──
    src0 = "./tmp/team9/rollout_0.json"
    if judge_batch0 and os.path.isfile(src0) and current:
        with open(src0) as f:
            b0 = json.load(f)
        t0 = time.time()
        blocks0, _, jt0 = judge_step(engine, b0, out_dir, 0)
        _, guards0, f1s0 = assemble_advantages(b0, blocks0, lam)
        m0 = {"step": 0, "policy": "base (team9 rollout)",
              "tag": "judge-reconstructed (no snaps → no train)",
              "judge": jt0, "judge_s": round(time.time() - t0, 1),
              "judge_blocks": sum(len(v) for v in blocks0.values()),
              "judge_verdicts": _verdict_buckets(blocks0),
              "g1": guards0["g1"], "g2": guards0["g2"], **f1s0}
        M(m0)
        print(f"  ── step 0 (judge-reconstructed): f1={m0['f1_mean']} "
              f"turns={m0['mean_turns']} "
              f"allzero={m0['g1']['allzero_frac']} "
              f"ansnz={m0['g1']['answer_head_nonzero_frac']} "
              f"λturns={m0['g1']['lambda_turns']} "
              f"sg={m0['g2']['sg_actions']}/blk={m0['g2']['judged_blocks']} "
              f"verdicts={m0['judge_verdicts']} "
              f"judge={m0['judge_s']}s [st={jt0['struct_s']}/"
              f"pa={jt0['probarm_s']}]", flush=True)

    for step in range(start_step, steps):
        m = {"step": step, "policy": current}
        ts = time.time()
        results = asyncio.run(rollout_step(
            engine, cases, G, temp, tb, max_rounds, current))
        m["rollout_s"] = round(time.time() - ts, 1)
        m["gpu_engine_rollout_mib"] = _gpu_mib(engine_dev)
        from kgqa.core.utils import PHASE_TIMES as PT
        m["phase_llm_s"] = round(PT.get("llm", 0), 1)
        m["phase_dispatch_s"] = round(PT.get("dispatch", 0), 1)
        for k in PT:
            PT[k] = 0.0

        ts = time.time()
        blocks_by_key, path, jt = judge_step(engine, results, out_dir, step)
        m["judge_s"] = round(time.time() - ts, 1)
        m["judge"] = jt
        m["judge_blocks"] = sum(len(v) for v in blocks_by_key.values())
        m["gpu_engine_judge_mib"] = _gpu_mib(engine_dev)
        m["judge_verdicts"] = _verdict_buckets(blocks_by_key)

        rows, guards, f1stats = assemble_advantages(results, blocks_by_key,
                                                    lam)
        m.update({"g1": guards["g1"], "g2": guards["g2"], **f1stats})

        tstats = trainer.train_rows(rows)
        m["train"] = tstats
        m["train_s"] = tstats.get("train_s", 0.0)
        m["gpu_train_mib"] = _gpu_mib(train_dev)
        m["adapter_delta_norm"] = trainer.adapter_delta_norm()

        new_name = f"policy_{step}"
        swap: dict = {"mount_src": mount_src}
        if mount_src == "nfs":
            new_path = os.path.join(adapt_dir, new_name)
        else:
            os.makedirs(shm_dir, exist_ok=True)
            new_path = os.path.join(shm_dir, new_name)
        t0 = time.time()
        trainer.save_adapter(new_path)
        swap["save_s"] = round(time.time() - t0, 2)   # file CREATION
        ts = time.time()
        if os.environ.get("SLEEP_EACH_STEP", "1") == "1":
            mib_before = _gpu_mib(engine_dev)
            swap["sleep_s"] = round(engine.sleep(1), 2)
            swap["mib_released"] = mib_before - _gpu_mib(engine_dev)
            swap["wake_s"] = round(engine.wake(), 2)
            swap["mib_after_wake"] = _gpu_mib(engine_dev)
        swap["mount_s"] = round(engine.mount_adapter(new_name, new_path), 2)
        engine.reset_prefix_cache()
        swap["sanity"] = engine.sanity_gen(new_name)
        if current:
            swap["unmount_s"] = round(engine.unmount_adapter(current), 2)
            old = mount_paths.get(current)
            if old and os.path.isdir(old) and (
                    old.startswith(shm_dir) or old.startswith(adapt_dir)):
                t0 = time.time()
                shutil.rmtree(old)
                swap["rmtree_s"] = round(time.time() - t0, 2)
        m["swap"] = swap
        m["swap_s"] = round(time.time() - ts, 2)
        m["gpu_engine_after_swap_mib"] = _gpu_mib(engine_dev)
        mount_paths[new_name] = new_path
        m["total_s"] = round(
            m["rollout_s"] + m["judge_s"] + m["train_s"] + m["swap_s"], 1)
        current = new_name
        M(m)
        print(f"  ── step {step}: f1={m['f1_mean']} "
              f"turns={m['mean_turns']} allzero={m['g1']['allzero_frac']} "
              f"ansnz={m['g1']['answer_head_nonzero_frac']} "
              f"λturns={m['g1']['lambda_turns']} "
              f"sg={m['g2']['sg_actions']}/blk={m['g2']['judged_blocks']} "
              f"La={m['train'].get('loss_answer')} "
              f"Le={m['train'].get('loss_explore')} "
              f"gn={m['train'].get('grad_norm')} "
              f"total={m['total_s']}s "
              f"(ro={m['rollout_s']} jd={m['judge_s']}"
              f"[st={jt['struct_s']}/pa={jt['probarm_s']}] "
              f"tr={m['train_s']} sw={m['swap_s']}"
              f"[sv={swap.get('save_s')}/mo={swap.get('mount_s')}])",
              flush=True)

    # cleanup — adapters were throwaway (no checkpoints stored)
    if current:
        engine.unmount_adapter(current)
        last = mount_paths.get(current)
        if last and os.path.isdir(last) and (
                last.startswith(shm_dir) or last.startswith(adapt_dir)):
            shutil.rmtree(last, ignore_errors=True)
    shutil.rmtree(shm_dir, ignore_errors=True)
    print(f"═══ done: steps {start_step}..{steps-1} in "
          f"{(time.time()-t_boot)/60:.1f} min (boot incl.) → {metrics_path} "
          f"═══", flush=True)


if __name__ == "__main__":
    main()
