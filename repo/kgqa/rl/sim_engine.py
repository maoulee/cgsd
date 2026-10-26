#!/usr/bin/env python3
"""Team9 two-GPU e2e simulation — engine side (degrade B: manual loop).

SimEngine subclasses OfflineVLLM's pattern (without importing its __init__
path — we need enable_lora with max_loras=2 from the start) and adds:

  1. chat_batch_snap(): generation via EXPLICIT token ids (chat template
     applied by us, TokensPrompt sent to the engine) + per-turn snapshot
     (prompt_ids / generated ids / sampled-token logprobs). The snapshot is
     the codex plan §4 token record — training consumes ACTUAL sampled
     tokens, never re-tokenized text.
  2. Dynamic LoRA mount/unmount at engine level (llm_engine.lora_manager,
     tmp/lora_unblock_note.md recipe) — adapter swap WITHOUT engine restart.
  3. sleep()/wake() passthrough (vLLM sleep level=1) — the production
     time-sharing mechanism, exercised + timed each step.
  4. gold_logprob_batch() inherited (probability arm of the judge).

Everything runs in the qwen35 env (vllm 0.25.1) on ONE GPU (engine), the
training copy lives on the other GPU.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


@dataclass
class TurnSnap:
    """§4 token snapshot for ONE assistant turn."""
    prompt_ids: list[int]
    gen_ids: list[int]
    gen_logprobs: list[float] | None   # sampled-token logprobs (PPO old_lp)
    raw_text: str                      # full generation (reasoning + content)
    reasoning: str
    text: str                          # content after </think>


class SimEngine:
    """Offline vLLM engine for the sim loop: 4B, LoRA-capable, sleep-capable."""

    def __init__(self, model_path: str = "models/Qwen3.5-4B",
                 gpu_memory_utilization: float = 0.6,
                 max_num_seqs: int = 64, tp_size: int = 1,
                 enable_sleep: bool = True):
        from vllm import LLM
        from vllm.config import ReasoningConfig
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.reasoning_end = ("I will now emit the tool call based on the "
                              "reasoning above.</think>")
        _extra = {}
        if enable_sleep:
            _extra["enable_sleep_mode"] = True
        t0 = time.time()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=65536,
            dtype="auto",
            trust_remote_code=True,
            enable_prefix_caching=True,
            kv_cache_dtype="auto",    # fp8 KV breaks wake_up in vllm 0.25.1
                                     # (init_fp8_kv_scales: .zero_() on list)
            max_num_seqs=max_num_seqs,
            enable_lora=True,
            max_loras=2,
            max_lora_rank=64,
            reasoning_config=ReasoningConfig(
                reasoning_start_str="<think>",
                reasoning_end_str=self.reasoning_end,
            ),
            disable_log_stats=True,
            **_extra,
        )
        self.load_s = time.time() - t0
        self.tok = AutoTokenizer.from_pretrained(model_path,
                                                 trust_remote_code=True)
        self._mounted: dict[str, int] = {}      # name -> lora id
        self._next_id = 9000

    # ---------------- generation with §4 snapshots ----------------

    def _prompt_ids(self, messages: list[dict]) -> list[int]:
        ret = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
        if hasattr(ret, "input_ids") or isinstance(ret, dict):
            ret = ret["input_ids"]      # transformers 5.14 → BatchEncoding
        if ret and isinstance(ret[0], (list, tuple)):
            ret = ret[0]
        return [int(i) for i in ret]

    def verify_template_once(self) -> bool:
        """Sanity check of OUR tokenization path (template → int ids).

        vllm 0.25.1's offline llm.chat (VL-processor rendering) is broken in
        this env (returns str tokens → TypeError in input validation), so we
        verify the properties that matter for training correctness instead:
        ids are ints, inside vocab, and re-tokenizing the rendered template
        string reproduces them (round-trip)."""
        msgs = [{"role": "user", "content": "Say OK."},
                {"role": "assistant", "content": "OK."},
                {"role": "user", "content": "Again."}]
        ids = self._prompt_ids(msgs)
        ok_int = all(isinstance(i, int) for i in ids)
        ok_vocab = ok_int and max(ids) < len(self.tok)   # incl. added tokens
        text = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)
        ids2 = list(self.tok(text, add_special_tokens=False).input_ids)
        ok_rt = list(ids) == list(ids2)
        ok_suffix = text.rstrip().endswith("assistant\n<think>")
        print(f"  [engine] template verify: n={len(ids)} ints={ok_int} "
              f"vocab={ok_vocab} roundtrip={ok_rt} "
              f"gen_prompt_suffix={ok_suffix}", flush=True)
        return ok_int and ok_vocab and ok_rt and ok_suffix

    def chat_batch_snap(self, messages_list: list[list[dict]], *,
                        thinking_budget: int = 1000,
                        temperature: float = 0.3,
                        max_tokens: int | None = None,
                        top_p: float = 0.8, top_k: int = 20,
                        presence_penalty: float = 1.5,
                        lora_name: str | None = None,
                        want_logprobs: bool = True) -> list[TurnSnap]:
        """Batch generation; prompt sent as EXACT token ids we tokenized.

        Returns one TurnSnap per prompt — the §4 record for training."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if max_tokens is None:
            max_tokens = thinking_budget + 2560
        my_ids = [self._prompt_ids(m) for m in messages_list]
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in my_ids]
        sp = SamplingParams(
            thinking_token_budget=thinking_budget,
            temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, top_k=top_k, presence_penalty=presence_penalty,
            logprobs=0 if want_logprobs else None,
        )
        kwargs = {}
        if lora_name:
            kwargs["lora_request"] = self._lora_request(lora_name)
        outs = self.llm.generate(prompts, sp, **kwargs)

        snaps = []
        for i, o in enumerate(outs):
            g = o.outputs[0]
            raw = g.text or ""
            reasoning, text = "", raw
            core = self.reasoning_end.replace("</think>", "").strip()
            if "</think>" in raw:
                idx = raw.rfind("</think>")
                reasoning, text = raw[:idx], raw[idx + len("</think>"):].strip()
                if core and reasoning.rstrip().endswith(core):
                    reasoning = reasoning.rstrip()[:-len(core)]
            lps = None
            if want_logprobs and g.logprobs:
                lps = [next(iter(pos.values())).logprob for pos in g.logprobs]
                if len(lps) != len(g.token_ids):
                    lps = None      # defensive: never misaligned old_lp
            snaps.append(TurnSnap(
                prompt_ids=my_ids[i],
                gen_ids=list(g.token_ids),
                gen_logprobs=lps, raw_text=raw,
                reasoning=reasoning.strip(), text=text))
        return snaps

    # ---------------- probability arm ----------------

    def gold_logprob_batch(self, pairs: list[tuple[str, str]]) -> list[float | None]:
        """mean gold-token logprob per (prefix, gold) — judge probability arm."""
        from vllm import SamplingParams
        marker = "\n\nAnswer: "
        prompts = [p + marker + g for p, g in pairs]
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
        outs = self.llm.generate(prompts, sp)
        res = []
        for (prefix, gold), o in zip(pairs, outs):
            pl = o.prompt_logprobs
            if not pl:
                res.append(None)
                continue
            start = len(self.tok(prefix + marker,
                                 add_special_tokens=False).input_ids)
            if start >= len(pl):
                start = max(0, len(pl) - 2)
            lps = []
            for j in range(start, len(pl)):
                item = pl[j]
                if item:
                    v = next(iter(item.values()))
                    if v and getattr(v, "logprob", None) is not None:
                        lps.append(v.logprob)
            res.append(sum(lps) / len(lps) if lps else None)
        return res

    # ---------------- LoRA lifecycle ----------------

    def _lora_request(self, name: str):
        from vllm.lora.request import LoRARequest
        if name not in self._mounted:
            raise ValueError(f"adapter {name!r} not mounted "
                             f"(have {list(self._mounted)})")
        i = self._mounted[name]
        return LoRARequest(name, i, self._path_of[name])

    def mount_adapter(self, name: str, path: str) -> float:
        """Load adapter from disk into the live engine (returns seconds).

        vllm 0.25.1 v1: the lora_manager lives in the EngineCore process —
        the frontend LLMEngine only exposes engine_core.add_lora()."""
        from vllm.lora.request import LoRARequest
        if name in self._mounted:
            raise ValueError(f"{name} already mounted")
        i = self._next_id
        self._next_id += 1
        t0 = time.time()
        self.llm.llm_engine.engine_core.add_lora(
            LoRARequest(name, i, path))
        self._mounted[name] = i
        self._path_of = getattr(self, "_path_of", {})
        self._path_of[name] = path
        return time.time() - t0

    def unmount_adapter(self, name: str) -> float:
        i = self._mounted.pop(name, None)
        if i is None:
            return 0.0
        t0 = time.time()
        self.llm.llm_engine.engine_core.remove_lora(i)
        return time.time() - t0

    def sanity_gen(self, lora_name: str | None = None, n_tokens: int = 16) -> str:
        """Short generation proving the engine + (new) adapter answer."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        msgs = [{"role": "user", "content":
                 "Answer with one word: what is 2+2?"}]
        sp = SamplingParams(temperature=0.0, max_tokens=n_tokens)
        kwargs = {}
        if lora_name:
            kwargs["lora_request"] = self._lora_request(lora_name)
        out = self.llm.generate(
            [TokensPrompt(prompt_token_ids=self._prompt_ids(msgs))], sp,
            **kwargs)[0].outputs[0]
        return (out.text or "").strip()[:60]

    # ---------------- sleep / wake (production time-sharing) ----------------

    def sleep(self, level: int = 1) -> float:
        t0 = time.time()
        self.llm.sleep(level=level)
        return time.time() - t0

    def wake(self) -> float:
        t0 = time.time()
        self.llm.wake_up()
        return time.time() - t0

    def reset_prefix_cache(self):
        try:
            self.llm.llm_engine.reset_prefix_cache()
        except Exception as e:      # non-fatal: cache is only a perf knob
            print(f"  [engine] reset_prefix_cache failed: {e}", flush=True)
