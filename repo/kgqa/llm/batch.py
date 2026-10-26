"""Batch LLM execution for the KGQA pipeline.

The stage runner already groups prompts by pipeline stage.  This module keeps
that batching efficient, but degrades cleanly when a provider does not expose
the local vLLM ``/chat/completions/batch`` endpoint.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

from kgqa.llm.client import (
    LLM_API_URL,
    LLM_MODEL,
    LLM_TIMEOUT_SEC,
    batch_api_url,
    build_payload,
    request_headers,
    _call_single_direct,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# Conservative default batch chunk for the first stable end-to-end run. Raise
# via KGQA_LLM_BATCH_SIZE once the vLLM deployment is confirmed not to OOM or
# prefill-queue at this size; 100 was the agreed tuning target.
_BATCH_SIZE = max(1, _env_int("KGQA_LLM_BATCH_SIZE", 100))
_BATCH_CONCURRENCY = max(1, _env_int("KGQA_LLM_BATCH_CONCURRENCY", 1))
_SINGLE_CONCURRENCY = max(1, _env_int("KGQA_LLM_SINGLE_CONCURRENCY", 8))
_LOCAL_API = any(host in LLM_API_URL for host in ("localhost", "127.0.0.1", "0.0.0.0"))
_USE_BATCH_ENDPOINT = _env_bool("KGQA_LLM_USE_BATCH_ENDPOINT", _LOCAL_API)
_BATCH_ENDPOINT_DISABLED = False
_BATCH_TIMEOUT_SEC = _env_float("KGQA_LLM_BATCH_TIMEOUT_SEC", 900.0)
_ENABLE_CACHE = _env_bool("KGQA_LLM_CACHE", False)
_CACHE_DIR = Path(os.getenv("KGQA_LLM_CACHE_DIR", ".cache/kgqa_llm"))

_TRUNCATE_RETRY_MSG: Dict[str, str] = {
    "role": "user",
    "content": (
        "Your previous response was cut off due to length. "
        "Answer concisely in under 200 words; avoid repetition and explanation."
    ),
}

_USAGE = defaultdict(float)


def get_usage_stats() -> Dict[str, float]:
    """Return aggregate LLM transport stats for the current process."""
    return dict(_USAGE)


def reset_usage_stats() -> None:
    """Clear aggregate LLM transport stats."""
    _USAGE.clear()


def _cache_key(messages: List[Dict[str, str]], max_tokens: int,
               temperature: float, top_p: float) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> Optional[str]:
    if not _ENABLE_CACHE:
        return None
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get("response")
        return value if isinstance(value, str) else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(key: str, response: Optional[str]) -> None:
    if not _ENABLE_CACHE or response is None:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(key).write_text(json.dumps({"response": response}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return


async def batch_call_llm(
    session: aiohttp.ClientSession,
    prompts: List[List[Dict[str, str]]],
    max_tokens: int = 500,
) -> List[Optional[str]]:
    """Send conversations efficiently and return responses aligned to prompts.

    Local vLLM deployments can use the custom batch endpoint.  OpenAI-compatible
    providers without that endpoint should set ``KGQA_LLM_USE_BATCH_ENDPOINT=0``;
    if they do not, this client will still fall back to concurrent single calls.

    Sampling defaults (temperature=0.3, top_p=0.8) can be overridden via
    ``KGQA_LLM_BATCH_TEMPERATURE`` / ``KGQA_LLM_BATCH_TOP_P`` — useful for
    deterministic (temperature=0) reproduction.
    """
    results, _reasoning = await _call_many(
        session,
        prompts,
        max_tokens=max_tokens,
        temperature=_env_float("KGQA_LLM_BATCH_TEMPERATURE", 0.3),
        top_p=_env_float("KGQA_LLM_BATCH_TOP_P", 0.8),
    )
    return results


async def batch_call_llm_with_reasoning(
    session: aiohttp.ClientSession,
    prompts: List[List[Dict[str, str]]],
    max_tokens: int = 500,
) -> tuple:
    """Same as ``batch_call_llm`` but ALSO returns the vLLM ``<think>`` reasoning
    field per prompt. Returns ``(results, reasoning)``. Used by run_react_batch
    so batch trajectories carry the model's CoT (parity with the per-call path
    which already captures reasoning via _call_single_with_reasoning)."""
    return await _call_many(
        session,
        prompts,
        max_tokens=max_tokens,
        temperature=_env_float("KGQA_LLM_BATCH_TEMPERATURE", 0.3),
        top_p=_env_float("KGQA_LLM_BATCH_TOP_P", 0.8),
    )


async def batch_call_llm_hot(
    session: aiohttp.ClientSession,
    prompts: List[List[Dict[str, str]]],
    max_tokens: int = 500,
) -> List[Optional[str]]:
    """Same as ``batch_call_llm`` but with hotter retry sampling."""
    results, _reasoning = await _call_many(
        session,
        prompts,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    return results


async def _call_many(
    session: aiohttp.ClientSession,
    prompts: List[List[Dict[str, str]]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple:
    """Returns (results, reasoning)."""
    if not prompts:
        return [], []

    start_time = time.perf_counter()
    _USAGE["logical_prompts"] += len(prompts)

    results: List[Optional[str]] = [None] * len(prompts)
    reasoning: List[Optional[str]] = [None] * len(prompts)
    uncached: List[tuple[int, str, List[Dict[str, str]]]] = []
    for idx, prompt in enumerate(prompts):
        key = _cache_key(prompt, max_tokens, temperature, top_p)
        cached = _read_cache(key)
        if cached is not None:
            results[idx] = cached
            _USAGE["cache_hits"] += 1
        else:
            uncached.append((idx, key, prompt))

    if uncached:
        chunks = [
            uncached[start:start + _BATCH_SIZE]
            for start in range(0, len(uncached), _BATCH_SIZE)
        ]

        sem = asyncio.Semaphore(_BATCH_CONCURRENCY)

        async def _run_chunk(chunk):
            async with sem:
                chunk_prompts = [item[2] for item in chunk]
                chunk_contents, chunk_reasoning = await _call_chunk(
                    session,
                    chunk_prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return chunk, chunk_contents, chunk_reasoning

        for chunk, chunk_results, chunk_reasoning in await asyncio.gather(*[_run_chunk(chunk) for chunk in chunks]):
            for (orig_idx, key, _), response, rsn in zip(chunk, chunk_results, chunk_reasoning):
                results[orig_idx] = response
                reasoning[orig_idx] = rsn
                _write_cache(key, response)

    _USAGE["wall_seconds"] += time.perf_counter() - start_time
    return results, reasoning


async def _call_chunk(
    session: aiohttp.ClientSession,
    prompts: List[List[Dict[str, str]]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple:
    """Returns (results, reasoning_results). reasoning_results[idx] is the vLLM
    `<think>` field; None when the batch endpoint wasn't used or fell back to
    single calls (no reasoning separation there)."""
    global _BATCH_ENDPOINT_DISABLED

    if not prompts:
        return [], []
    if len(prompts) == 1 or not _USE_BATCH_ENDPOINT or _BATCH_ENDPOINT_DISABLED:
        indiv = await _call_individual_concurrent(
            session, prompts, max_tokens=max_tokens, temperature=temperature, top_p=top_p
        )
        return indiv, [None] * len(prompts)

    payload = build_payload(prompts, max_tokens, temperature=temperature, top_p=top_p)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            _USAGE["batch_requests"] += 1
            async with session.post(
                batch_api_url(),
                headers=request_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=max(_BATCH_TIMEOUT_SEC, LLM_TIMEOUT_SEC)),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"LLM batch request failed with HTTP {resp.status}: {body[:500]}")
                data = await resp.json()

            results: List[Optional[str]] = [None] * len(prompts)
            reasoning_results: List[Optional[str]] = [None] * len(prompts)
            truncated: List[int] = []
            for choice in data.get("choices", []):
                idx = choice.get("index")
                if isinstance(idx, int) and 0 <= idx < len(prompts):
                    msg = choice.get("message", {})
                    results[idx] = msg.get("content", "")
                    reasoning_results[idx] = msg.get("reasoning", "") or ""
                    if choice.get("finish_reason") == "length":
                        truncated.append(idx)

            if truncated:
                _USAGE["truncated_retries"] += len(truncated)
                retry_prompts = [prompts[idx] + [_TRUNCATE_RETRY_MSG] for idx in truncated]
                retry_results = await _call_individual_concurrent(
                    session,
                    retry_prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                for idx, retry_result in zip(truncated, retry_results):
                    if retry_result:
                        results[idx] = retry_result

            missing = [i for i, value in enumerate(results) if value is None]
            if missing:
                _USAGE["missing_choice_fallbacks"] += len(missing)
                fallback_results = await _call_individual_concurrent(
                    session,
                    [prompts[i] for i in missing],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                for idx, fallback_result in zip(missing, fallback_results):
                    results[idx] = fallback_result or ""

            return results, reasoning_results
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(2)

    _USAGE["batch_failures"] += 1
    _BATCH_ENDPOINT_DISABLED = True
    if last_error is not None:
        print(f"  LLM batch endpoint failed; falling back to concurrent single calls: {last_error}")
    indiv = await _call_individual_concurrent(
        session, prompts, max_tokens=max_tokens, temperature=temperature, top_p=top_p
    )
    return indiv, [None] * len(prompts)


async def _call_individual_concurrent(
    session: aiohttp.ClientSession,
    prompts: List[List[Dict[str, str]]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Optional[str]]:
    sem = asyncio.Semaphore(_SINGLE_CONCURRENCY)

    async def _one(prompt):
        async with sem:
            try:
                _USAGE["single_requests"] += 1
                return await _call_single_direct(
                    session,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout_seconds=LLM_TIMEOUT_SEC,
                )
            except Exception:
                _USAGE["single_failures"] += 1
                return ""

    return await asyncio.gather(*[_one(prompt) for prompt in prompts])
