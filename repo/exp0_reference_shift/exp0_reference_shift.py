#!/usr/bin/env python3
"""Experiment 0: Reference-Conditioned Distribution Shift (offline, no training).

Question: does the SAME model, given the correct reference answer, raise the
probability of effective actions and lower that of invalid ones — on the SAME
trajectory prefix, teacher-forced on the original action tokens?

Conditions (only the system-prompt suffix differs; history is byte-identical):
  A none      — original system prompt
  B correct   — Verified answer: <case golds>
  C wrong     — Verified answer: <another case's golds> (format/count matched)
  D placebo   — Verified answer: <neutral phrases> (format/length matched)

Spans scored separately per assistant action turn (block):
  action   — the tool-call JSON (content)  ← primary
  thinking — the <think> reasoning         ← diagnostic only

Full-vocab log-softmax per token via vLLM prompt_logprobs=0 (never top-256).

Usage:
  SEQ_PROMPT=RC1 python scripts/exp0_reference_shift.py --limit 4     # smoke
  SEQ_PROMPT=RC1 python scripts/exp0_reference_shift.py              # full
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SEQ_PROMPT", "C123")
os.environ.setdefault("SEQ_ADAPTIVE_ROUTES", "1")

VERDICTS = "tmp/prob_pathway_r9a_v5.json"
ROLLOUT = "reports/r9a_hard_36x3.json"
OUT_JSONL = "reports/exp0_reference_shift.jsonl"
# r9a_hard_36x3 provenance (logs/r9a_hard_36x3.log): 9B via http server,
# SEQ_PROMPT=C123 + SEQ_ADAPTIVE_ROUTES=1; prompt sha256 recorded below.
PROMPT_SHA = "36edaf911956e4e4a11555810380d2acbe443c9fe5e0e97bca6b2e2f018bb132"

REF_TMPL = "\n\n[Reference]\nVerified answer: {vals}"
NEUTRAL = [
    "archived working note", "intermediate indexing summary",
    "routine processing record", "background catalog entry",
    "standard reference filing", "periodic maintenance log",
    "general purpose annotation", "auxiliary tracking detail",
    "placeholder catalog remark", "neutral bookkeeping phrase",
]

LABEL5 = {  # verdict → 5-class label (user spec §C)
    "有效-概率": "EFFECTIVE",      # covers 通路内 / 多数上升 / 质量归因
    "有效-结构": "KEEP",
    "可移除-概率": "REDUNDANT",
    "可移除-环境重复": "INVALID",
    "不评估": "UNKNOWN",
}


def label_of(verdict: str) -> str:
    v = verdict or ""
    for k, lab in LABEL5.items():
        if v.startswith(k):
            return lab
    return "UNKNOWN"


def assistant_full_text(st: dict) -> str:
    """History-style assistant text (training shape): <think>...</think> + action."""
    action = (st.get("content") or "").strip()
    thought = (st.get("reasoning") or "").strip()
    if thought:
        return f"<think>\n{thought}\n</think>\n\n{action}"
    return action


def build_user_msg(row: dict) -> str:
    """Question + anchor entities, as prelink injected (dataset order)."""
    q = f"Question: {row['question']}"
    ents = []
    for st in row["trajectory"]:
        if st.get("role") == "tool" and str(st.get("name", "")).find("plan") >= 0:
            c = st.get("content") or ""
            for line in c.splitlines():
                if line.startswith("question_entities:"):
                    ents = [e.strip() for e in
                            line.split(":", 1)[1].split("|") if e.strip()]
            break
    if not ents:                       # plan echo missing — bare question
        return q
    lines = "\n".join(f"  - {e}" for e in ents)
    return q + (
        "\n\nQuestion entities (canonical graph names — use these spellings as "
        "your anchors for the entities the question names). The authoritative "
        "relation set for each anchor comes from retrieve_relations:\n" + lines)


def render(tok, system: str, user: str, hist: list[dict], final: str | None):
    """Chat-template render. final=None → prefix only (boundary token count);
    final=str → same conversation with the last assistant turn's text = final
    (full = thinking+action, cut = thinking only → action span between)."""
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}] + hist
    if final is not None:
        msgs = msgs + [{"role": "assistant", "content": final}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False)
    return tok(text, add_special_tokens=False).input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N rows")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", default=OUT_JSONL)
    ap.add_argument("--position", choices=["system", "near"], default="system")
    args = ap.parse_args()

    from kgqa.agent.seq_react_loop import seq_agents_md

    system_base = seq_agents_md()
    import hashlib as _h
    _sha = _h.sha256(system_base.encode("utf-8")).hexdigest()
    assert _sha == PROMPT_SHA, \
        f"system prompt drift: {_sha} != {PROMPT_SHA} — prefix fidelity broken"
    verdicts = json.load(open(VERDICTS))
    rollout = json.load(open(ROLLOUT))
    by_key = {(r["case_id"], int(r["sample_idx"])): r for r in rollout}

    # wrong-reference pairing: next f1>0 case in sorted order (deterministic)
    vrows = [v for v in verdicts if v.get("f1", 0) > 0 and
             (v["case_id"], int(v["sample"])) in by_key]
    vrows.sort(key=lambda v: v["case_id"])
    if args.limit:
        vrows = vrows[:args.limit]
    wrong_of = {}
    for i, v in enumerate(vrows):
        w = vrows[(i + 1) % len(vrows)]
        g = v["golds"] or ["unknown"]
        wg = (w["golds"] or ["unknown"])
        wrong_of[v["case_id"]] = [wg[j % len(wg)] for j in range(len(g))]

    llm = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ckpt_hash = hashlib.sha256(
        open(args.model + "/config.json", "rb").read()).hexdigest()[:12]

    # ── build all (block × condition) prompts ─────────────────────────
    jobs = []          # (meta, prompt_full, prompt_cut)
    for v in vrows:
        key = (v["case_id"], int(v["sample"]))
        row = by_key[key]
        tr = row["trajectory"]
        user = build_user_msg(row)
        golds = v["golds"] or ["unknown"]
        suffix = {
            "none": "",
            "correct": REF_TMPL.format(vals=" | ".join(golds)),
            "wrong": REF_TMPL.format(vals=" | ".join(wrong_of[v["case_id"]])),
            "placebo": REF_TMPL.format(
                vals=" | ".join(NEUTRAL[:max(1, len(golds))])),
        }

        def _add_turn_jobs(v, tr, user, suffix, idx, lab, verdict):
            """Score the assistant turn at tr[idx-1] under each condition.
            --position system: suffix appended to the system prompt (OPSD
            teacher view). --position near: suffix appended to the LAST
            history message, immediately before the scored turn
            (attention-salience control)."""
            st = tr[idx - 1]
            full_text = assistant_full_text(st)
            thought = (st.get("reasoning") or "").strip()
            # boundaries: wrap = assistant-template sentinel (thinking start),
            # act = end of thinking / start of action, end = full turn end
            cut_wrap = " "
            cut_act = (f"<think>\n{thought}\n</think>\n\n"
                       if thought else " ")
            hist = []
            for m in tr[:idx - 1]:
                if m.get("role") == "assistant":
                    hist.append({"role": "assistant",
                                 "content": assistant_full_text(m)})
                else:
                    hist.append({"role": m.get("role"), "content": m.get("content"),
                                 "name": m.get("name")})
            for cname, suf in suffix.items():
                if args.position == "near" and cname != "none" and hist:
                    h = [dict(m) for m in hist]
                    h[-1]["content"] = (h[-1].get("content") or "") + suf
                    sys_text = system_base
                else:
                    h = hist
                    sys_text = system_base + suf
                f_ids = render(tok, sys_text, user, h, full_text)
                a_ids = render(tok, sys_text, user, h, cut_act)
                w_ids = render(tok, sys_text, user, h, cut_wrap)
                jobs.append(({
                    "case_id": v["case_id"], "sample": v["sample"],
                    "block_idx": idx, "verdict": verdict, "label5": lab,
                    "condition": cname, "checkpoint_hash": ckpt_hash,
                    "n_action_tokens": len(f_ids) - len(a_ids),
                    "n_think_tokens": len(a_ids) - len(w_ids),
                }, f_ids, len(w_ids), len(a_ids)))

        for b in v["blocks"]:
            lab = label_of(b.get("verdict", ""))
            if lab == "UNKNOWN":
                continue
            idx = int(b["idx"])
            if idx < 1 or idx >= len(tr) or tr[idx - 1].get("role") != "assistant":
                continue
            _add_turn_jobs(v, tr, user, suffix, idx, lab, b["verdict"])

        # answer-turn channel (spec §七 "只有 Final Answer 变化" check):
        # final ACCEPTED answer turn on f1>0 trajectories
        from kgqa.rl.turn_credit import accepted_answer_turn
        ans_i = accepted_answer_turn(tr)
        if ans_i is not None:
            _add_turn_jobs(v, tr, user, suffix, ans_i + 1, "ANSWER",
                           "final-accepted-answer")

    # ── length stats ────────────────────────────────────────────────────
    lens = [len(j[1]) for j in jobs]
    lens.sort()
    print(f"[exp0] {len(vrows)} trajectories → {len(jobs)} block×cond jobs | "
          f"prompt len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} "
          f"max={lens[-1]}", flush=True)

    # ── HF chunked forward (vLLM prompt_logprobs materializes [T,V] logits
    # for the whole 24K-token prompt — 15GB+ per request, OOM/NCCL-deadlock.
    # Here: hidden states once, lm_head+log_softmax in 512-position chunks.) ──
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}", trust_remote_code=True)
    model.eval()
    del llm  # (never constructed in this path)

    def span_logprobs(ids, n_lo, n_hi):
        """logprobs of tokens ids[n_lo:n_hi] (full-vocab log_softmax,
        lm_head in 512-position chunks off a single forward)."""
        ids_t = torch.tensor([ids], device=model.device)
        am = torch.ones_like(ids_t)
        with torch.no_grad():
            hidden = model.model(input_ids=ids_t,
                                 attention_mask=am).last_hidden_state[0]
            lps = []
            CH = 512
            for s in range(n_lo - 1, n_hi - 1, CH):
                e = min(s + CH, n_hi - 1)
                logits = model.lm_head(hidden[s:e])
                lp = torch.log_softmax(logits.float(), dim=-1)
                tgt = ids_t[0, s + 1:e + 1]
                lps.append(lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
                           .cpu().tolist())
        return [x for part in lps for x in part]

    # shard across GPUs: --shards N --shard i keeps every i-th job
    my_jobs = [(m, f, w, a) for k, (m, f, w, a) in enumerate(jobs)
               if k % args.shards == args.shard]

    with open(args.out + f".shard{args.shard}", "w") as f:
        for n, (meta, f_ids, n_wrap, n_act) in enumerate(my_jobs):
            try:
                think = span_logprobs(f_ids, n_wrap, n_act)
                act = span_logprobs(f_ids, n_act, len(f_ids))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                meta.update({"error": "oom"}); f.write(json.dumps(meta) + "\n"); f.flush(); continue
            meta.update({
                "n_span": len(act), "n_think": len(think),
                "logp_mean": (sum(act) / len(act)) if act else None,
                "logp_sum": sum(act) if act else None,
                "think_logp_mean": (sum(think) / len(think)) if think else None,
                "think_logp_sum": sum(think) if think else None,
            })
            f.write(json.dumps(meta) + "\n"); f.flush()
            if (n + 1) % 10 == 0:
                print(f"[exp0] shard{args.shard} {n+1}/{len(my_jobs)}",
                      flush=True)
    print(f"[exp0] shard{args.shard} done → {args.out}.shard{args.shard}",
          flush=True)


if __name__ == "__main__":
    main()
