#!/usr/bin/env python3
"""MTP speculative-decoding A/B on a rollout-shaped workload (Qwen3.5-4B).

Mirrors one rollout turn: ~9k-token prompt (RC2 system + question + tool
rounds), ~800-1500 generated tokens per request with the engine reasoning
config + thinking budget + production sampling (temp .7 / top_p .8 / k 20).

Run twice (spec decode is fixed at engine construction):
  CUDA_VISIBLE_DEVICES=0 python scripts/diag_mtp_bench.py            # baseline
  CUDA_VISIBLE_DEVICES=0 python scripts/diag_mtp_bench.py --mtp 2    # MTP k=2
"""
from __future__ import annotations

import argparse
import time

MODEL = "models/Qwen3.5-4B"
RC2 = "./kgqa/agent/SEQ_AGENTS_RC2.md"


def build_prompts(n):
    system = open(RC2, encoding="utf-8").read()
    tool_round = (
        "\n\nTool result (retrieve_subgraph):\n"
        "triples:\n  (Lou Seal) -[mascot.of]-> (San Francisco Giants)\n"
        "  (San Francisco Giants) -[championship.won]-> (2014 World Series)\n"
    )
    hint = "\n\nCurrent state: RETRIEVE. Use delivered evidence or continue retrieval.\n"
    out = []
    for i in range(n):
        q = (f"Question: case-{i}: which team does the mascot Lou Seal belong "
             f"to, and when did that team last win the World Series?")
        msgs = (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{q}{tool_round}{hint}<|im_end|>\n"
                f"<|im_start|>assistant\n")
        out.append(msgs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtp", type=int, default=0,
                    help="0 = baseline; k>0 = MTP with k speculative tokens")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1200)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config import ReasoningConfig

    kwargs = dict(
        model=MODEL, max_model_len=16384, gpu_memory_utilization=0.85,
        enable_prefix_caching=True, dtype="bfloat16", trust_remote_code=True,
        disable_log_stats=False,
        reasoning_config=ReasoningConfig(
            reasoning_start_str="<think>",
            reasoning_end_str=("I will now emit the tool call based on the "
                               "reasoning above.</think>")),
    )
    if args.mtp:
        kwargs["speculative_config"] = {
            "method": "mtp", "num_speculative_tokens": args.mtp}
    t0 = time.time()
    llm = LLM(**kwargs)
    load_s = time.time() - t0

    prompts = build_prompts(args.n)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.7,
                        top_p=0.8, top_k=20,
                        thinking_token_budget=1000)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"\n[mtp-bench] mtp={args.mtp or 'off'} load={load_s:.0f}s "
          f"gen={gen_s:.1f}s reqs={len(prompts)} tokens={ntok} "
          f"throughput={ntok/gen_s:.0f} tok/s "
          f"({gen_s/len(prompts):.1f}s/req avg {ntok/len(prompts):.0f} tok)",
          flush=True)


if __name__ == "__main__":
    main()
