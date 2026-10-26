"""HTTP chat_batch adapter — the SAME SYNC interface as OfflineVLLM.chat_batch
but against a PERSISTENT vLLM server (default :8000). Kills the ~2.5-min
per-process engine spawn/compile tax: the server loads once, every rollout
batch-injects prompts (user's original 固化为类API design, 2026-08-22).

Usage: LLM_MODE=http on seq_rollout; server must be up
(bash scripts/start_local_qwen35_server.sh with the same reasoning config).
"""
import asyncio
import os
import threading

import aiohttp

from kgqa.llm.client import LLM_MODEL, request_headers
from kgqa.llm.offline_vllm import ChatResult

CHAT_URL = (os.environ.get("KGQA_LLM_API_URL",
                           "http://localhost:8000/v1/chat/completions"))
BATCH_URL = CHAT_URL.replace("/chat/completions", "/chat/completions/batch")


class HTTPChatBatch:
    """Sync chat_batch over the persistent server — a dedicated background
    event loop bridges the sync call (the rollout calls chat_batch without
    await, matching OfflineVLLM)."""

    def __init__(self, thinking_budget: int = 1000, max_num_seqs: int = 128):
        self._tb = thinking_budget
        self._sem_n = max_num_seqs
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever,
                                        daemon=True)
        self._thread.start()
        self._session = None   # created INSIDE the loop

    def _payload(self, messages, thinking_budget, temperature,
                 max_tokens, top_p, top_k, presence_penalty):
        p = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens or (thinking_budget + 2560),
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "thinking_token_budget": thinking_budget,
        }
        if top_k and top_k > 0:
            p["top_k"] = top_k
        return p

    async def _achat(self, messages, **kw):
        sem = getattr(self, "_sem", None)
        if sem is None:
            self._sem = sem = asyncio.Semaphore(self._sem_n)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        async with sem:
            for attempt in range(3):
                try:
                    async with self._session.post(
                        CHAT_URL, json=self._payload(messages, **kw),
                        headers=request_headers(),
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}: "
                                               f"{(await resp.text())[:200]}")
                        data = await resp.json()
                    msg = data["choices"][0]["message"]
                    return ChatResult(reasoning=msg.get("reasoning") or "",
                                      text=msg.get("content") or "")
                except (aiohttp.ClientError, asyncio.TimeoutError,
                        RuntimeError, KeyError, IndexError):
                    if attempt < 2:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    return ChatResult(reasoning="", text="")

    async def _achat_batch(self, messages_list, kw):
        # ONE batch POST per round (user design: 批量注入) — 267 individual
        # POSTs fought for the semaphore in two waves; the server's batch
        # endpoint continuous-batches natively. Fall back to per-request if
        # the batch call fails.
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        payload = dict(self._payload(messages_list, **kw))
        try:
            async with self._session.post(
                BATCH_URL, json=payload, headers=request_headers(),
                timeout=aiohttp.ClientTimeout(total=1800),
            ) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"batch HTTP {resp.status}: "
                                       f"{(await resp.text())[:200]}")
                data = await resp.json()
            out = [None] * len(messages_list)
            for ch in data.get("choices", []):
                i = ch.get("index")
                if isinstance(i, int) and 0 <= i < len(out):
                    m = ch.get("message") or {}
                    out[i] = ChatResult(reasoning=m.get("reasoning") or "",
                                        text=m.get("content") or "")
            if all(x is not None for x in out):
                return out
            messages_list = [m for m, x in zip(messages_list, out)
                             if x is None] or messages_list
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
            pass
        sem = getattr(self, "_sem", None)
        if sem is None:
            self._sem = sem = asyncio.Semaphore(self._sem_n)
        return await asyncio.gather(*[
            self._achat(m, **kw) for m in messages_list])

    # ── sync bridge (OfflineVLLM-compatible signature) ──────────────────
    def chat_batch(self, messages_list, thinking_budget=None,
                   temperature=0.3, max_tokens=None, top_p=0.8,
                   top_k=20, presence_penalty=1.5, **_):
        tb = thinking_budget or self._tb
        kw = dict(thinking_budget=tb, temperature=temperature,
                  max_tokens=max_tokens, top_p=top_p, top_k=top_k,
                  presence_penalty=presence_penalty)
        fut = asyncio.run_coroutine_threadsafe(
            self._achat_batch(messages_list, kw), self._loop)
        return fut.result(timeout=1800)

    async def achat_one(self, messages, thinking_budget=None,
                        temperature=0.3, max_tokens=None, top_p=0.8,
                        top_k=20, presence_penalty=1.5, model=None, **_):
        """Single-request async chat (bubble mode, 2026-09-07): one POST per
        turn under the client semaphore — the server continuous-batches
        stream arrivals natively, so per-request injection keeps vLLM's
        throughput while the rollout drops the round barrier."""
        tb = thinking_budget or self._tb
        kw = dict(thinking_budget=tb, temperature=temperature,
                  max_tokens=max_tokens, top_p=top_p, top_k=top_k,
                  presence_penalty=presence_penalty)
        return await self._achat(messages, **kw)

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
