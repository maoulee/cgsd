"""Offline vLLM batch interface — two capabilities:
  1. chat_batch: generation with thinking budget (rollout)
  2. logprob_batch: prompt_logprobs (IG / teacher forcing)

Cleanly separated from pipeline logic. Can be swapped with HTTP vLLM,
transformers, or other model backends without changing the pipeline.

Usage:
    llm = OfflineVLLM(model_path="models/Qwen3.5-9B")
    # Rollout
    results = llm.chat_batch(messages_list, thinking_budget=1000)
    # IG
    logprobs = llm.logprob_batch(prompt_strings)
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ChatResult:
    """One case's generation result."""
    reasoning: str   # <think> content (empty if no reasoning)
    text: str        # generated content (the actual tool call / answer)
    logprob: float | None = None   # mean sampled-token logprob (ROLL_LOGPROBS=1)


class OfflineVLLM:
    """Offline vLLM with chat (generation) + logprob (teacher forcing) batch APIs.

    Config choices:
    - prefix_caching=False: Mamba experimental crash fix (Qwen3.5 hybrid arch)
    - CUDA graph ON (enforce_eager=False): speed
    - reasoning_config: thinking boundary for thinking_token_budget
    - No tool_call_parser: SEQ uses flat text protocol (tool: ...), not native tools
    """

    def __init__(
        self,
        model_path: str = "models/Qwen3.5-9B",
        tp_size: int = 2,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 65536,
        kv_cache_dtype: str = "fp8",
        max_num_seqs: int = 128,
        max_num_batched_tokens: int | None = None,
        enforce_eager: bool = False,
        reasoning_start: str = "<think>",
        reasoning_end: str = "I will now emit the tool call based on the reasoning above.</think>",
        lora_modules: dict | None = None,
    ):
        from vllm import LLM
        from vllm.config import ReasoningConfig

        self.model_path = model_path
        # Optional LoRA serving (iterated RL: rollout with a TRAINED adapter as
        # policy). Same dynamic-LoRA mechanism as the HTTP server: callers pass
        # the adapter name as chat_batch(model=...); the engine hot-loads it.
        self.lora_modules = lora_modules or None
        _lora_kwargs = {}
        if self.lora_modules:
            _lora_kwargs = dict(enable_lora=True, max_lora_rank=64,
                                max_loras=len(self.lora_modules))
        # Store for post-hoc reasoning split: vLLM's reasoning_config searches for
        # reasoning_start_str in the *generated* output, but the Qwen3.5 chat template
        # injects `<think>` into the prompt — so the open tag is never in the output,
        # vLLM cannot find the start boundary, dumps everything into `text`, and the
        # thinking_token_budget never fires. We recover the boundary by splitting the
        # generated text on the end marker (what the qwen3 reasoning_parser does).
        self.reasoning_end = reasoning_end
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="auto",
            trust_remote_code=True,
            enable_prefix_caching=True,    # works with Mamba 'align' mode (vLLM 0.25.1 verified, 9.6x prefix reuse)
            kv_cache_dtype=kv_cache_dtype,
            max_num_seqs=max_num_seqs,     # cap concurrent decode (thrash guard)
            enforce_eager=enforce_eager,   # True for prompt_logprobs (IG): no CUDA-graph memory reserve
            **({"max_num_batched_tokens": max_num_batched_tokens}
               if max_num_batched_tokens is not None else {}),
            reasoning_config=ReasoningConfig(
                reasoning_start_str=reasoning_start,
                reasoning_end_str=reasoning_end,
            ),
            disable_log_stats=True,
            # NO tool_call_parser — flat text protocol
            **_lora_kwargs,
        )
        # Tokenizer for gold offset computation (logprob_batch)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _lora_request(self, model: str):
        """Build a vLLM LoRARequest for a registered adapter name."""
        from vllm.lora.request import LoRARequest
        path = (self.lora_modules or {}).get(model)
        if not path:
            raise ValueError(f"unknown LoRA name {model!r}; registered: {list(self.lora_modules or [])}")
        return LoRARequest(model, abs(hash(model)) % 100000, path)

    def chat_batch(
        self,
        messages_list: list[list[dict]],
        thinking_budget: int = 1000,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        top_p: float = 1.0,
        top_k: int = -1,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        model: str | None = None,
    ) -> list[ChatResult]:
        """Batch generation with thinking budget. Returns ChatResult per case.

        Args:
            messages_list: list of message lists (chat format, one per case)
            thinking_budget: max reasoning tokens (vLLM thinking_token_budget)
            temperature: sampling temperature (>0 for diverse GRPO rollouts)
            max_tokens: total token budget (thinking + content). If None, auto = thinking_budget + 2560
            top_p, top_k, presence_penalty, repetition_penalty: sampling params.
                For Qwen3.5 rollout diversity matching the HTTP production path, use
                temperature=0.8, top_p=0.8, top_k=20, presence_penalty=1.5.
            seed: RNG seed (None = nondeterministic; set for reproducible sampling)
        """
        from vllm import SamplingParams

        if max_tokens is None:
            max_tokens = thinking_budget + 2560

        sp = SamplingParams(
            thinking_token_budget=thinking_budget,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
            # ROLL_LOGPROBS=1: return sampled-token logprobs so the pool builder
            # can rank refined references by f1 x probability (user design:
            # correctness alone lets a lucky path become the standard).
            logprobs=0 if os.environ.get("ROLL_LOGPROBS") == "1" else None,
        )
        outputs = self.llm.chat(
            messages_list, sp,
            **({"lora_request": self._lora_request(model)} if model else {}))
        # vLLM docs (Limitations): the separated `reasoning` field is SERVER-ONLY
        # (/v1/chat/completions etc.); offline llm.chat() returns raw `text` =
        # "reasoning...</think>content". The budget (reasoning_config +
        # thinking_token_budget) still tracks & caps reasoning offline, but the
        # field split does not happen — recover it here by splitting on the end
        # marker, mirroring what the qwen3 reasoning_parser does server-side.
        END_TAG = "</think>"
        phrase_core = (self.reasoning_end or "").replace(END_TAG, "").strip()
        results = []
        for output in outputs:
            o = output.outputs[0]
            raw = o.text or ""
            reasoning, text = "", raw
            if END_TAG in raw:
                idx = raw.rfind(END_TAG)
                reasoning = raw[:idx]
                text = raw[idx + len(END_TAG):].strip()
                # strip the budget force-inject transition phrase from the
                # reasoning tail (it's vLLM filler, not model reasoning)
                if phrase_core and reasoning.rstrip().endswith(phrase_core):
                    reasoning = reasoning.rstrip()[:-len(phrase_core)]
            elif hasattr(o, "reasoning") and getattr(o, "reasoning"):
                # future-proof: if a vLLM version populates the field offline
                reasoning = o.reasoning
            lp_mean = None
            lp = getattr(o, "logprobs", None)
            ids = getattr(o, "token_ids", None) or []
            if lp:
                vals = []
                for pos, tid in enumerate(ids):
                    if pos < len(lp) and lp[pos]:
                        ent = lp[pos].get(tid)
                        if ent is not None:
                            vals.append(ent.logprob)
                if vals:
                    lp_mean = sum(vals) / len(vals)
            results.append(ChatResult(reasoning=reasoning.strip(), text=text,
                                      logprob=lp_mean))
        return results

    def logprob_batch(
        self,
        prompts: list[str],
    ) -> list[list[dict | None]]:
        """Batch prompt_logprobs. Returns FlatLogprobs per prompt.

        Each element is a list of {token_id: Logprob} per prompt position.
        Use extract_gold_logprob() to get mean gold token logprob.

        Args:
            prompts: list of prompt strings (gold already appended)
        """
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
        outputs = self.llm.generate(prompts, sp)
        return [output.prompt_logprobs for output in outputs]

    def gold_logprob(self, prefix: str, gold: str) -> float | None:
        """Teacher-forcing: gold appended to prefix, return mean gold token logprob.

        prompt = prefix + "\\n\\nAnswer: " + gold
        Returns mean logprob of gold tokens.
        """
        marker = "\n\nAnswer: "
        prompt = prefix + marker + gold
        pls = self.logprob_batch([prompt])
        pl = pls[0]
        if not pl:
            return None
        gold_start = len(self.tok(prefix + marker, add_special_tokens=False).input_ids)
        if gold_start >= len(pl):
            gold_start = max(0, len(pl) - 2)
        lps = []
        for i in range(gold_start, len(pl)):
            item = pl[i]
            if item:
                v = next(iter(item.values()))
                if v and getattr(v, "logprob", None) is not None:
                    lps.append(v.logprob)
        return sum(lps) / len(lps) if lps else None

    def gold_logprob_batch(
        self,
        prefix_gold_pairs: list[tuple[str, str]],
    ) -> list[float | None]:
        """Batch version of gold_logprob. Efficient: one llm.generate() call for all.

        Args:
            prefix_gold_pairs: [(prefix, gold), ...]
        Returns:
            [mean_gold_logprob, ...]
        """
        marker = "\n\nAnswer: "
        prompts = [prefix + marker + gold for prefix, gold in prefix_gold_pairs]
        pls = self.logprob_batch(prompts)
        results = []
        for i, (prefix, gold) in enumerate(prefix_gold_pairs):
            pl = pls[i]
            if not pl:
                results.append(None)
                continue
            gold_start = len(self.tok(prefix + marker, add_special_tokens=False).input_ids)
            if gold_start >= len(pl):
                gold_start = max(0, len(pl) - 2)
            lps = []
            for j in range(gold_start, len(pl)):
                item = pl[j]
                if item:
                    v = next(iter(item.values()))
                    if v and getattr(v, "logprob", None) is not None:
                        lps.append(v.logprob)
            results.append(sum(lps) / len(lps) if lps else None)
        return results
