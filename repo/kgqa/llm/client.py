"""Low-level LLM client functions for the KGQA pipeline.

Provides `call_llm()` for coalesced batch-optimized calls and
`_call_single_direct()` for single-shot requests.
"""
from __future__ import annotations

import asyncio
import os

import aiohttp

# ---------------------------------------------------------------------------
# Configuration — mirrors the constants in the monolith until kgqa.core.config
# is established.
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _completion_url(raw_url: str) -> str:
    url = raw_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


_RAW_LLM_API_URL = os.getenv("KGQA_LLM_API_URL", os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1"))
LLM_API_URL = _completion_url(_RAW_LLM_API_URL)
LLM_MODEL = os.getenv("KGQA_MODEL_NAME", os.getenv("KGQA_LLM_MODEL", "Qwen3.5-9B"))
LLM_API_KEY = os.getenv("KGQA_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_TIMEOUT_SEC = _env_float("KGQA_LLM_TIMEOUT_SEC", 120.0)
LLM_TEMPERATURE = _env_float("KGQA_LLM_TEMPERATURE", 0.3)
LLM_TOP_P = _env_float("KGQA_LLM_TOP_P", 0.8)
# Qwen3.5/3.6 official recommendation for instruction mode: top_k=20 +
# presence_penalty=1.5. presence_penalty is essential here — without it the
# tool-calling agent loops on the same expand_branch call (repeated tokens).
LLM_TOP_K = int(os.getenv("KGQA_LLM_TOP_K", "20"))
LLM_PRESENCE_PENALTY = _env_float("KGQA_LLM_PRESENCE_PENALTY", 1.5)
_seed = os.getenv("LLM_SEED", "").strip()
LLM_SEED = int(_seed) if _seed else None

_LOCAL_API = any(host in LLM_API_URL for host in ("localhost", "127.0.0.1", "0.0.0.0"))
SEND_CHAT_TEMPLATE_KWARGS = _env_bool("KGQA_LLM_SEND_CHAT_TEMPLATE_KWARGS", _LOCAL_API)
ENABLE_THINKING = _env_bool("KGQA_ENABLE_THINKING", False)
# SOFT reasoning hint via the model's chat template (thinking_budget kwarg).
# This was the original reasoning control but is only a hint the model may
# ignore — superseded by the hard cap below (THINKING_TOKEN_BUDGET). Kept at 0
# so the hard cap alone governs reasoning length under the default config; set
# KGQA_THINKING_BUDGET > 0 only for the legacy soft-cap path (rarely useful now).
THINKING_BUDGET = int(os.getenv("KGQA_THINKING_BUDGET", "0"))
# Top-level vLLM hard cap on reasoning tokens (SamplingParams.thinking_token_budget).
# When the reasoning token count hits this budget, vLLM force-injects
# reasoning_end_str (`</think>`) to terminate the <think> block cleanly, then the
# model proceeds to emit content/answer normally. This is the HARD enforcement
# path — distinct from the SOFT chat_template_kwargs.thinking_budget above (which
# is only a hint the model may ignore).
#
# Verified working on vLLM 0.23.0 + Qwen3.5-9B (scripts/ab_thinking_budget.py):
# with max_tokens sufficiently large (≥ a few k), budgets of 128/512 produce
# clean stop finish_reason + full answer in content. CRITICAL: max_tokens must
# exceed the budget generously, else reasoning eats the whole completion and
# finish_reason becomes 'length' with an empty answer (the original probe
# failure cause). 0 = no hard cap.
#
# A/B results (100 cases each, react path, excluding network errors):
#   WebQSP:  budget0=84%  budget512=87%(+3)   — budget512 also drops model
#                                              failures 6→0 (no more 16-iter
#                                              loops on hard cases)
#   CWQ:     budget0=72%  budget512=83.5%(+11.5) budget2048=80.6%
#            — both thinking budgets drop model_failed 6→0. budget512 wins:
#            this task is retrieval+extraction, NOT deep reasoning, so 512 is
#            enough to "gather thoughts" and longer budgets only invite
#            world-knowledge contamination (model abandons subgraph evidence).
# Default 512 is the verified sweet spot; override via KGQA_THINKING_TOKEN_BUDGET.
THINKING_TOKEN_BUDGET = int(os.getenv("KGQA_THINKING_TOKEN_BUDGET", "512"))
USE_BATCH_ENDPOINT = _env_bool("KGQA_LLM_USE_BATCH_ENDPOINT", _LOCAL_API)


def batch_api_url() -> str:
    """Return the local vLLM-style batch endpoint for chat completions."""
    return LLM_API_URL.replace("/chat/completions", "/chat/completions/batch")


def request_headers() -> dict:
    """Headers for OpenAI-compatible services."""
    if LLM_API_KEY and LLM_API_KEY != "EMPTY":
        return {"Authorization": f"Bearer {LLM_API_KEY}"}
    return {}


def build_payload(messages, max_tokens: int, temperature: float | None = None,
                  top_p: float | None = None,
                  tools: list | None = None, tool_choice: str | None = None) -> dict:
    """Build a chat-completions payload shared by single and batch clients.

    ``tools``/``tool_choice`` are optional and only attached when tools are
    provided — the content-only path used by ``call_llm`` is untouched.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
        "top_p": LLM_TOP_P if top_p is None else top_p,
        "top_k": LLM_TOP_K,
        "presence_penalty": LLM_PRESENCE_PENALTY,
    }
    # per-request sampling seed (OpenAI-standard param; vLLM honors it). Set
    # LLM_SEED for reproducible sampling runs (e.g. multi-seed stability tests).
    if LLM_SEED is not None:
        payload["seed"] = LLM_SEED
    # Top-level hard cap wins precedence: when THINKING_TOKEN_BUDGET > 0 we want
    # <think> ON so there is reasoning to cap. Override the soft enable_thinking
    # flag in that case (it would otherwise suppress thinking entirely).
    hard_cap_active = THINKING_TOKEN_BUDGET > 0
    effective_thinking = ENABLE_THINKING or hard_cap_active
    if SEND_CHAT_TEMPLATE_KWARGS:
        kws = {"enable_thinking": effective_thinking}
        # thinking_budget is the model's real (soft) reasoning control. 0 = no
        # <think>, which is what we want for stable tool-calling when no hard
        # cap is requested. Under a hard cap the soft budget is redundant but
        # harmless; keep 0 so only the hard cap governs reasoning length.
        kws["thinking_budget"] = 0 if hard_cap_active else THINKING_BUDGET
        payload["chat_template_kwargs"] = kws
    # Top-level vLLM sampling param: force-injects </think> once reasoning hits
    # the budget. Only attached when explicitly set (>0); omitted otherwise so
    # the payload stays OpenAI-compatible for non-vLLM backends.
    if hard_cap_active:
        payload["thinking_token_budget"] = THINKING_TOKEN_BUDGET
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    return payload


async def agent_call(
    session: aiohttp.ClientSession,
    messages: list,
    tools: list,
    tool_choice: str = "auto",
    max_tokens: int = 1024,
    temperature: float | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    """Single tool-call LLM POST. Returns the FULL ``choices[0].message`` dict
    (with ``content``, ``tool_calls``, ``reasoning``).

    Intentionally bypasses the batch coalescer — agent tool loops are single,
    turn-taking requests that must see the model's tool_calls immediately.
    HTTP-error handling mirrors ``_call_single_direct``.
    """
    payload = build_payload(
        messages, max_tokens,
        temperature=temperature,
        tools=tools, tool_choice=tool_choice,
    )
    async with session.post(
        LLM_API_URL,
        headers=request_headers(),
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout_seconds or LLM_TIMEOUT_SEC),
    ) as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise RuntimeError(f"LLM agent_call failed with HTTP {resp.status}: {body[:500]}")
        data = await resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM agent_call response has no choices: {str(data)[:500]}")
    msg = choices[0].get("message") or {}
    # Normalise: ensure content is a string (may be None when tool_calls present)
    if msg.get("content") is None:
        msg["content"] = ""
    if "tool_calls" not in msg:
        msg["tool_calls"] = None
    return msg


async def _call_single_direct(
    session: aiohttp.ClientSession,
    messages: list,
    max_tokens: int = 500,
    temperature: float | None = None,
    top_p: float | None = None,
    timeout_seconds: float = None,
) -> str:
    """Direct single LLM call (no coalescing).

    Used by ``batch_call_llm`` for single-item fallback.
    """
    payload = build_payload(messages, max_tokens, temperature=temperature, top_p=top_p)
    async with session.post(
        LLM_API_URL,
        headers=request_headers(),
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout_seconds or LLM_TIMEOUT_SEC),
    ) as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise RuntimeError(f"LLM request failed with HTTP {resp.status}: {body[:500]}")
        data = await resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM response has no choices: {str(data)[:500]}")
    return data["choices"][0]["message"]["content"]


async def _call_single_with_reasoning(
    session: aiohttp.ClientSession,
    messages: list,
    max_tokens: int = 500,
    temperature: float | None = None,
    top_p: float | None = None,
    timeout_seconds: float | None = None,
) -> tuple:
    """Same as ``_call_single_direct`` but also returns the reasoning field.

    vLLM with ``thinking_token_budget`` puts the model's chain-of-thought in a
    separate ``reasoning`` field on the response message (NOT inside content).
    This wrapper exposes it so callers (e.g. trajectory generation) can capture
    the full reasoning alongside the tool-call content.

    Returns ``(content, reasoning)`` where reasoning may be "" if vLLM did not
    separate it (e.g. thinking disabled or model wrote CoT inline in content).
    """
    payload = build_payload(messages, max_tokens, temperature=temperature, top_p=top_p)
    async with session.post(
        LLM_API_URL,
        headers=request_headers(),
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout_seconds or LLM_TIMEOUT_SEC),
    ) as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise RuntimeError(f"LLM request failed with HTTP {resp.status}: {body[:500]}")
        data = await resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM response has no choices: {str(data)[:500]}")
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    return content, reasoning


class _BatchCoalescer:
    """Coalesces concurrent call_llm requests into batch API calls.

    When multiple call_llm calls arrive within a short collection window,
    they are merged into a single /v1/chat/completions/batch request.
    This avoids GPU contention between independent LLM requests.
    """

    _BATCH_WINDOW = 0.05  # 50ms collection window
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._pending: list = []  # list of (messages, max_tokens, future)
        self._timer = None
        self._lock = asyncio.Lock()

    async def submit(self, session, messages, max_tokens=500, retries=3):
        """Submit a request. Returns response string."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        async with self._lock:
            self._pending.append((session, messages, max_tokens, retries, fut))
            if self._timer is None or self._timer.done():
                self._timer = asyncio.ensure_future(self._flush_after_window())
        return await fut

    async def _flush_after_window(self):
        await asyncio.sleep(self._BATCH_WINDOW)
        await self._flush()

    async def _flush(self):
        async with self._lock:
            batch = self._pending[:]
            self._pending.clear()

        if not batch:
            return

        # Group by (session, max_tokens) — same params can share a batch
        groups: dict = {}
        for session, messages, max_tokens, retries, fut in batch:
            key = (id(session), max_tokens)
            groups.setdefault(key, []).append((session, messages, max_tokens, retries, fut))

        for key, items in groups.items():
            session = items[0][0]
            max_tokens = items[0][2]
            retries = items[0][3]

            if len(items) == 1 or not USE_BATCH_ENDPOINT:
                # Single request — use normal endpoint
                async def _run_one(item):
                    _, messages, _, _, fut = item
                    try:
                        result = await self._call_single(session, messages, max_tokens, retries)
                        if not fut.done():
                            fut.set_result(result)
                    except Exception:
                        if not fut.done():
                            fut.set_result("")
                await asyncio.gather(*[_run_one(item) for item in items])
            else:
                # Multiple requests — use batch endpoint
                messages_list = [m for _, m, _, _, _ in items]
                futs = [f for _, _, _, _, f in items]
                try:
                    results = await self._call_batch(session, messages_list, max_tokens, retries)
                    for f, r in zip(futs, results):
                        if not f.done():
                            f.set_result(r)
                except Exception:
                    # Fallback: individual calls
                    async def _fallback_one(item):
                        _, messages, _, _, fut = item
                        if fut.done():
                            return
                        try:
                            r = await self._call_single(session, messages, max_tokens, retries)
                            fut.set_result(r)
                        except Exception:
                            fut.set_result("")
                    await asyncio.gather(*[_fallback_one(item) for item in items])

    @staticmethod
    async def _call_single(session, messages, max_tokens, retries):
        payload = build_payload(messages, max_tokens)
        for attempt in range(retries):
            try:
                async with session.post(
                    LLM_API_URL,
                    headers=request_headers(),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=LLM_TIMEOUT_SEC),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        raise RuntimeError(f"LLM request failed with HTTP {resp.status}: {body[:500]}")
                    data = await resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"LLM response has no choices: {str(data)[:500]}")
                return choices[0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    raise

    @staticmethod
    async def _call_batch(session, messages_list, max_tokens, retries):
        payload = build_payload(messages_list, max_tokens)
        for attempt in range(retries):
            try:
                async with session.post(
                    batch_api_url(),
                    headers=request_headers(),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=max(LLM_TIMEOUT_SEC, 120.0)),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        raise RuntimeError(f"LLM batch request failed with HTTP {resp.status}: {body[:500]}")
                    data = await resp.json()
                results = [None] * len(messages_list)
                for choice in data.get("choices", []):
                    idx = choice["index"]
                    if 0 <= idx < len(messages_list):
                        results[idx] = choice["message"]["content"]
                # Fill any missing results with individual calls
                for i, r in enumerate(results):
                    if r is None:
                        results[i] = await _BatchCoalescer._call_single(
                            session, messages_list[i], max_tokens, retries)
                return results
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    raise


async def call_llm(
    session: aiohttp.ClientSession,
    messages: list,
    max_tokens: int = 500,
    retries: int = 3,
) -> str:
    """LLM call with automatic request coalescing for batch optimization."""
    return await _BatchCoalescer.get().submit(session, messages, max_tokens, retries)
