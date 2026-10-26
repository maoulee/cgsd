#!/usr/bin/env python3
"""Qwen3-Embedding-0.6B API server for relation/entity retrieval.

Uses last-token pooling (required by Qwen3-Embedding) and Instruct format
for retrieval queries.

Endpoints:
  POST /embed - Encode texts to embeddings
  POST /retrieve - Retrieve top-k relations/entities by NL query
  POST /retrieve_batch - One POST for a round's requests (list in, list out)
  POST /precompute - Pre-encode and cache candidates for fast retrieval

Usage:
    python scripts/gte_api_server.py --port 8003
"""

import argparse
import hashlib
import os
import time as _time
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel
import uvicorn

app = FastAPI(title="Qwen3 Embedding Server")

# Global model
tokenizer = None
model = None
DEVICE = os.environ.get("GTE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

TASK_DESC = "Given a query, retrieve the document most semantically similar to it"

# ── Config ──────────────────────────────────────────────────────────
MAX_CAND_LEN = 128
MAX_QUERY_LEN = 256
# 编码 batch 上限可配: 大并发时 drain 合并的文本数直接决定 GTE 激活显存
# （vLLM 的 KV 是启动时预分配、锁死的, 变动的峰值来自 GTE 与训练激活）
DEFAULT_BATCH_SIZE = int(os.environ.get("GTE_ENCODE_BATCH", "64"))

# ── Embedding Cache (LRU) ──────────────────────────────────────────
MAX_CACHE_SIZE = 50000


class EmbeddingCache:
    """LRU cache for text → embedding vectors (thread-safe: endpoints run
    inference in worker threads since the 2026-08-21 perf fix)."""

    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        import threading
        self._lock = threading.Lock()

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._hash(text)
        with self._lock:
            emb = self._cache.get(key)
            if emb is not None:
                self._cache.move_to_end(key)
        return emb

    def put(self, text: str, emb: np.ndarray):
        key = self._hash(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = emb
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def get_many(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        return [self.get(t) for t in texts]

    def put_many(self, texts: List[str], embs: np.ndarray):
        for t, e in zip(texts, embs):
            self.put(t, e)

    @property
    def size(self):
        return len(self._cache)

    def clear(self):
        with self._lock:
            self._cache.clear()


emb_cache = EmbeddingCache()


def last_token_pool(last_hidden_states, attention_mask):
    """Last-token pooling — required by Qwen3-Embedding."""
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), seq_lens]


class EmbedRequest(BaseModel):
    texts: List[str]
    batch_size: int = DEFAULT_BATCH_SIZE


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int
    count: int


class RetrieveRequest(BaseModel):
    query: str
    candidates: List[str] = Field(default_factory=list)
    candidate_texts: Optional[List[str]] = None
    top_k: int = 5
    instruct: Optional[str] = None
    pool_key: Optional[str] = None   # register-once / query-by-key mode


class RetrieveResponse(BaseModel):
    results: List[dict]


class RetrieveBatchRequest(BaseModel):
    items: List[RetrieveRequest]


class RetrieveBatchResponse(BaseModel):
    results: List[List[dict]]


class PrecomputeRequest(BaseModel):
    candidates: List[str]
    candidate_texts: Optional[List[str]] = None


class CacheStatsResponse(BaseModel):
    size: int
    max_size: int
    device: str
    infer_wait_s: float = 0.0    # cumulative submit→pickup (queueing)
    encode_s: float = 0.0        # cumulative GPU forward time
    n_jobs: int = 0              # encode jobs through the inference worker
    n_texts: int = 0             # texts encoded (cache misses)
    batch_calls: int = 0         # /retrieve_batch requests served
    batch_items: int = 0         # items inside them
    resp_cache_hits: int = 0     # response-cache hits (either retrieve endpoint)
    resp_cache_size: int = 0
    resp_cache_max: int = 0


@app.on_event("startup")
async def load_model():
    global tokenizer, model
    model_path = os.environ.get("GTE_MODEL_PATH", "models/Qwen3-Embedding-0.6B")
    print(f"Loading Qwen3-Embedding-0.6B from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
    if DEVICE == "balanced":
        # 单服务双卡均摊 (用户设计 2026-10-21: 一个 GTE 服务, 显存平均到
        # 两张卡 ~1.6G/卡, 而非每卡一个完整实例 3.2G) — accelerate 按层切分
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="balanced",
        )
        _dm = model.hf_device_map
        print(f"Model loaded with device_map=balanced: "
              f"{len(set(_dm.values()))} devices, layers split "
              f"{sorted(set(str(v) for v in _dm.values()))}. "
              f"Hidden dim: {model.config.hidden_size}")
    else:
        # CPU 分支用 fp32: torch CPU 没有原生 fp16 内核，逐算子上转换比
        # fp32 慢 5-20×（2026-10-24 冒烟实测 100 items×20 cands 120s 超时
        # 的主因之一）。排序任务对精度不敏感，fp32 足够。
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        model.to(DEVICE)
        print(f"Model loaded on {DEVICE} "
              f"(dtype={'fp32' if DEVICE == 'cpu' else 'fp16'}). "
              f"Hidden dim: {model.config.hidden_size}")
    model.eval()
    model.config.use_cache = False


def _encode(
    texts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = MAX_CAND_LEN,
) -> np.ndarray:
    all_embs = []
    # balanced 模式无单一 DEVICE — 输入落在模型首层所在设备, hook 负责跨卡搬运
    _in_dev = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        ).to(_in_dev)
        with torch.inference_mode():
            outputs = model(**tokens, use_cache=False)
        embs = last_token_pool(
            outputs.last_hidden_state,
            tokens['attention_mask'].to(outputs.last_hidden_state.device),
        )
        embs = F.normalize(embs.float(), p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
        del tokens, outputs, embs
    return np.concatenate(all_embs, axis=0)


def _encode_cached(
    texts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = MAX_CAND_LEN,
) -> np.ndarray:
    """Sync encode with cache (startup/compat path — blocks the caller)."""
    cached = emb_cache.get_many(texts)
    miss_indices = [i for i, c in enumerate(cached) if c is None]
    result = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        miss_embs = _infer_submit(miss_texts, batch_size, max_length)
        emb_cache.put_many(miss_texts, miss_embs)
        for j, idx in enumerate(miss_indices):
            result[idx] = miss_embs[j]
    for i, c in enumerate(cached):
        if c is not None:
            result[i] = c
    return result


async def _encode_cached_async(
    texts: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = MAX_CAND_LEN,
) -> np.ndarray:
    """Async encode with cache — the request-path entry. Cache hits resolve
    without touching the worker; misses go to the SINGLE INFERENCE WORKER
    (2026-08-21 refactor): one dedicated thread owns the model, concurrent
    requests coalesce into combined batches (dynamic batching). Replaces both
    legacy modes measured 2026-08-21: inline sync (loop-blocking, 2542 gte
    lane-s/100 cases) and asyncio.to_thread (GIL-thrash, 7663 lane-s)."""
    cached = emb_cache.get_many(texts)
    miss_indices = [i for i, c in enumerate(cached) if c is None]
    result = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        miss_embs = await _infer_submit_async(miss_texts, batch_size, max_length)
        emb_cache.put_many(miss_texts, miss_embs)
        for j, idx in enumerate(miss_indices):
            result[idx] = miss_embs[j]
    for i, c in enumerate(cached):
        if c is not None:
            result[i] = c
    return result


# ── single inference worker (dynamic batching) ─────────────────────────────
import asyncio as _aio
import queue as _queue
import threading as _threading
from concurrent.futures import Future as _CFuture

_infer_q: "_queue.Queue" = _queue.Queue()
_worker_up = _threading.Event()

# server-side attribution counters (perf audit 2026-08-23): queue_wait = submit
# → worker pickup (how long a request sat behind the inference worker's batch
# window / other jobs); encode = GPU forward time. Client RTT − (wait+encode)
# ≈ HTTP + scoring. Exposed read-only via /cache/stats.
_SRV_STATS = {"infer_wait_s": 0.0, "encode_s": 0.0, "n_jobs": 0, "n_texts": 0,
              "batch_calls": 0, "batch_items": 0, "resp_cache_hits": 0}
_SRV_STATS_LOCK = _threading.Lock()


def _srv_stat_bump(key, n=1):
    with _SRV_STATS_LOCK:
        _SRV_STATS[key] += n


def _merge_jobs(jobs):
    """Encode a drain-cycle's jobs: same-(batch_size,max_length) jobs merge
    into ONE encode call (true dynamic batching); result slices distributed.
    jobs entries are (texts, bs, ml, fut, t_submit) — t_submit feeds the
    queue-wait attribution."""
    by_params = {}
    for texts, bs, ml, fut, t_submit in jobs:
        by_params.setdefault((bs, ml), []).append((texts, fut, t_submit))
    for (bs, ml), group in by_params.items():
        flat, bounds, futs = [], [], []
        _now = _time.monotonic()
        for texts, fut, t_submit in group:
            bounds.append((len(flat), len(flat) + len(texts)))
            flat.extend(texts)
            futs.append(fut)
        try:
            _t0 = _time.perf_counter()
            embs = _encode(flat, batch_size=bs, max_length=ml)
            _enc_dt = _time.perf_counter() - _t0
            with _SRV_STATS_LOCK:
                _SRV_STATS["infer_wait_s"] += sum(_now - t for _, _, t in group)
                _SRV_STATS["encode_s"] += _enc_dt
                _SRV_STATS["n_jobs"] += len(group)
                _SRV_STATS["n_texts"] += len(flat)
            for (a, b), fut in zip(bounds, futs):
                fut.set_result(embs[a:b])
        except Exception as e:
            for _, fut, _ in group:
                fut.set_exception(e)


def _collect_batch(q, window_s, first_s, early_n, clock):
    """Drain the inference queue under the ADAPTIVE two-stage window
    (perf audit 2026-08-22). After the first job: wait only a short FIRST_SLICE
    (5ms default) — if nothing else arrives, compute immediately (kills the
    fixed window's tax on an empty queue / lone late caller); any arrival
    within the slice extends the deadline to the FULL window counted from the
    first job, so a round's time cluster still coalesces. EARLY-CLOSE keeps
    the 256-text cap. `clock` is injected (monotonic seconds) for unit tests.

    Env: GTE_BATCH_WINDOW (full window, default 0.015s; <=0 disables waiting
    entirely), GTE_BATCH_FIRST_SLICE (short probe slice, default 0.005s),
    GTE_BATCH_EARLY (early-close text cap, default 256).
    """
    jobs = [q.get()]
    n_texts = len(jobs[0][0])
    if window_s <= 0:
        while True:
            try:
                jobs.append(q.get_nowait())
            except _queue.Empty:
                break
        return jobs
    t0 = clock()
    deadline = t0 + first_s
    full_deadline = t0 + max(window_s, first_s)
    extended = False
    while n_texts < early_n:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        try:
            j = q.get(timeout=max(0.0, remaining))
        except _queue.Empty:
            break
        jobs.append(j)
        n_texts += len(j[0])
        if not extended:
            extended = True
            deadline = full_deadline
    return jobs


def _inference_worker():
    """The ONLY thread that touches the model. ROUND-SYNCHRONOUS BATCHING
    (2026-08-21, user design): the rollout dispatches a round's GTE calls in a
    time cluster — after the first job arrives, hold a short BATCH_WINDOW to
    collect the cluster, then merge-encode everything in one pass. Window
    tuned 60→15ms + EARLY-CLOSE at 256 merged texts (2026-08-21 measurement:
    the fixed 60ms wait was a per-call tax exceeding its coalescing value —
    same-round GTE concurrency is ~10-30, not 64). The window itself is
    ADAPTIVE since 2026-08-22 — see _collect_batch."""
    import os as _os
    window_s = float(_os.environ.get("GTE_BATCH_WINDOW", "0.015"))
    first_s = float(_os.environ.get("GTE_BATCH_FIRST_SLICE", "0.005"))
    _early_n = int(_os.environ.get("GTE_BATCH_EARLY", "256"))
    _worker_up.set()
    while True:
        _merge_jobs(_collect_batch(_infer_q, window_s, first_s, _early_n,
                                   _time.monotonic))


def _infer_submit(texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
    """Sync submit (blocks caller — startup/compat path only)."""
    fut: "_CFuture" = _CFuture()
    _infer_q.put((texts, batch_size, max_length, fut, _time.monotonic()))
    return fut.result()


async def _infer_submit_async(texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
    """Async submit — awaits without blocking the event loop."""
    fut: "_CFuture" = _CFuture()
    _infer_q.put((texts, batch_size, max_length, fut, _time.monotonic()))
    return await _aio.wrap_future(fut)


_threading.Thread(target=_inference_worker, daemon=True).start()


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    embs = await _encode_cached_async(req.texts, req.batch_size)
    return EmbedResponse(
        embeddings=embs.tolist(),
        dim=embs.shape[1],
        count=embs.shape[0],
    )


# ── pool registry + response cache (shared by both retrieve endpoints) ─────
from collections import OrderedDict as _OD
_pool_registry: "_OD[str, List[str]]" = _OD()
_POOL_REG_MAX = 20000

# Response cache (perf phase 3, 2026-08-23): (pool_sig, query, top_k,
# instruct) → results. The resident server outlives runs and G=3 lockstep +
# formulaic sub-questions make repeats the norm — a hit answers with ZERO
# encode and ZERO queueing (the second run over the same cohort is near-free,
# which is the iteration-experiment win). Loop-only access (async endpoints),
# so no lock is needed.
_RESP_CACHE: "_OD[tuple, List[dict]]" = _OD()
_RESP_CACHE_MAX = int(os.environ.get("GTE_RESP_CACHE_MAX", "100000") or 0)


def _pool_sig(req: RetrieveRequest) -> str:
    """Pool identity for the response cache. pool_key mode uses the key
    itself (the client derives it from pool content); value mode hashes the
    (candidates, candidate_texts) pair — both fields appear in the rows."""
    if req.pool_key:
        return f"p:{req.pool_key}"
    h = hashlib.md5()
    cand_texts = req.candidate_texts if req.candidate_texts else req.candidates
    for c, t in zip(req.candidates, cand_texts):
        h.update(c.encode("utf-8", "replace"))
        h.update(b"\x1f")
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x1f")
    return f"v:{h.hexdigest()}"


def _register_pool(pool_key: str, cand_texts: List[str]):
    _pool_registry[pool_key] = list(cand_texts)
    if len(_pool_registry) > _POOL_REG_MAX:
        _pool_registry.popitem(last=False)


def _resolve_pool(req: RetrieveRequest):
    """Resolve/register a retrieve request's candidate pool. Returns
    (cand_texts, candidates) — the texts scored vs the labels returned.
    /retrieve_batch registers pool_key+candidates items BEFORE calling this
    so key-only siblings in the same batch resolve."""
    if req.pool_key and not req.candidates:
        stored = _pool_registry.get(req.pool_key)
        assert stored is not None, f"unknown pool_key {req.pool_key!r}"
        return stored, stored
    cand_texts = req.candidate_texts if req.candidate_texts else req.candidates
    assert len(cand_texts) == len(req.candidates), "candidate_texts must match candidates length"
    if req.pool_key:
        _register_pool(req.pool_key, cand_texts)
    return cand_texts, req.candidates


def _resp_cache_get(key):
    hit = _RESP_CACHE.get(key)
    if hit is not None:
        _RESP_CACHE.move_to_end(key)
        _srv_stat_bump("resp_cache_hits")
    return hit


def _resp_cache_put(key, results: List[dict]):
    if _RESP_CACHE_MAX <= 0:
        return
    if key in _RESP_CACHE:
        _RESP_CACHE.move_to_end(key)
        return
    _RESP_CACHE[key] = results
    if len(_RESP_CACHE) > _RESP_CACHE_MAX:
        _RESP_CACHE.popitem(last=False)


def _query_text(req: RetrieveRequest) -> str:
    task = req.instruct if req.instruct else TASK_DESC
    return f'Instruct: {task}\nQuery: {req.query}'


def _top_rows(q_embs, c_embs, candidates, cand_texts, top_k) -> List[dict]:
    """Score + top-k shared by /retrieve and /retrieve_batch — the ONE place
    the ranking math lives, so batch == per-request for the same input."""
    scores = (q_embs @ c_embs.T)[0]
    top_k = min(top_k, len(candidates))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"index": int(idx),
             "candidate": candidates[idx],
             "text": cand_texts[idx],
             "score": float(scores[idx])} for idx in top_indices]


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Retrieve top-k candidates by NL query similarity.

    Queries are prefixed with Instruct: format for Qwen3-Embedding.
    Candidates are encoded as-is (no prefix).
    Single-inference-worker + dynamic batching (2026-08-21 refactor); the
    QUERY embedding is also cached (sub-questions are formulaic and repeat
    heavily across cases).
    POOL_KEY MODE (2026-08-21): the same (head, pool) is re-queried across
    facts with only the QUERY changing — the 10-30KB candidate payload was
    re-transferred and re-parsed each time. Pass pool_key + candidates ONCE
    (registers the pool); later calls send pool_key with no candidates.
    RESPONSE CACHE (2026-08-23): a repeated (pool, query, top_k, instruct)
    answers from the LRU below — zero encode, zero queueing.
    """
    if req.pool_key and req.candidates:
        # keep the registry warm even when the query itself is a response-
        # cache hit (key-only callers on the same pool depend on it)
        _register_pool(req.pool_key,
                       req.candidate_texts if req.candidate_texts else req.candidates)
    sig = _pool_sig(req)
    key = (sig, req.query, req.top_k, req.instruct or "")
    hit = _resp_cache_get(key)
    if hit is not None:
        return RetrieveResponse(results=hit)
    cand_texts, candidates = _resolve_pool(req)
    # query through the cache too (key = full instruct+query string).
    # batch_size=32, not 1 (perf audit 2026-08-22): _merge_jobs groups by
    # (batch_size, max_length) — bs=1 put every merged query group on a
    # one-by-one GPU forward path (30 coalesced queries = 30 forwards); at 32
    # the whole group encodes in 32-wide forwards. (bs,ml)=(32,256) also keeps
    # queries in their OWN merge group, away from the (64,128) candidates.
    q_embs = await _encode_cached_async([_query_text(req)], batch_size=32, max_length=MAX_QUERY_LEN)
    c_embs = await _encode_cached_async(cand_texts, batch_size=DEFAULT_BATCH_SIZE,
                                        max_length=MAX_CAND_LEN)
    results = _top_rows(q_embs, c_embs, candidates, cand_texts, req.top_k)
    _resp_cache_put(key, results)
    return RetrieveResponse(results=results)


@app.post("/retrieve_batch", response_model=RetrieveBatchResponse)
async def retrieve_batch(req: RetrieveBatchRequest):
    """One POST carries a whole round's requests (perf phase 3, 2026-08-23;
    the client-side collector flushes here). Response-cache hits answer with
    zero encode/queue; the misses get ONE merged query encode and ONE encode
    per unique pool, then score through the shared _top_rows — identical
    results to /retrieve for the same input.

    REQUEST-LEVEL CHUNKING (2026-10-24): the forward path was already
    bounded (DEFAULT_BATCH_SIZE texts per forward, GTE_BATCH_EARLY=256
    merged texts per drain) but one thick-burst POST processed ALL items in
    a single gather — peak latency, parse memory and pool-encode fanout all
    scaled with burst size. Items now process in sequential slices of
    GTE_MAX_REQ_ITEMS (default 64; ordering preserved, registration first).
    """
    _srv_stat_bump("batch_calls")
    _srv_stat_bump("batch_items", len(req.items))
    # registration pass FIRST: a pool_key item without candidates may share
    # the batch with (and must not precede) the item that registers its pool.
    for it in req.items:
        if it.pool_key and it.candidates:
            _register_pool(it.pool_key,
                           it.candidate_texts if it.candidate_texts else it.candidates)
    cap = int(os.environ.get("GTE_MAX_REQ_ITEMS", "64") or 0)
    out: List[Optional[List[dict]]] = []
    for s in range(0, len(req.items), max(cap, 1)):
        out.extend(await _retrieve_batch_core(req.items[s:s + max(cap, 1)]))
    return RetrieveBatchResponse(results=out)


async def _retrieve_batch_core(items: List[RetrieveRequest]):
    """Plan/encode/score one slice of a /retrieve_batch — cache hits resolve
    in place, misses get the merged query encode + one encode per unique
    pool."""
    out: List[Optional[List[dict]]] = [None] * len(items)
    # plan the misses (cache hits resolve in place)
    plans = []   # (idx, item, resp_key)
    for i, it in enumerate(items):
        key = (_pool_sig(it), it.query, it.top_k, it.instruct or "")
        hit = _resp_cache_get(key)
        if hit is not None:
            out[i] = hit
        else:
            plans.append((i, it, key))
    if plans:
        # ONE encode call for every unique query text of the misses (same
        # (bs,ml)=(32,256) group as /retrieve; the infer worker merges it
        # into a single GPU pass).
        q_idx, q_texts = {}, []
        for _i, it, _k in plans:
            qt = _query_text(it)
            if qt not in q_idx:
                q_idx[qt] = len(q_texts)
                q_texts.append(qt)
        q_embs = await _encode_cached_async(q_texts, batch_size=32, max_length=MAX_QUERY_LEN)
        # resolve each item's pool — labels follow the ITEM's own mode (a
        # registering call labels rows with its candidates, a key-only call
        # with the stored texts, exactly like /retrieve) — but ENCODE only
        # one copy per unique pool (G=3 lockstep re-queries the same pool
        # under different queries)
        p_idx, pool_texts = {}, []
        resolved = []   # (i, key, q_idx, pool_idx, candidates, cand_texts, top_k)
        for i, it, key in plans:
            cand_texts, candidates = _resolve_pool(it)
            sig = _pool_sig(it)
            if sig not in p_idx:
                p_idx[sig] = len(pool_texts)
                pool_texts.append(cand_texts)
            resolved.append((i, key, q_idx[_query_text(it)], p_idx[sig],
                             candidates, cand_texts, it.top_k))
        c_embs_list = await _aio.gather(*[
            _encode_cached_async(ct, batch_size=DEFAULT_BATCH_SIZE, max_length=MAX_CAND_LEN)
            for ct in pool_texts])
        for i, key, qi, pi, candidates, cand_texts, top_k in resolved:
            rows = _top_rows(q_embs[qi:qi + 1], c_embs_list[pi],
                             candidates, cand_texts, top_k)
            _resp_cache_put(key, rows)
            out[i] = rows
    return out


@app.post("/precompute")
async def precompute(req: PrecomputeRequest):
    """Pre-encode candidates and cache them for fast subsequent retrieval."""
    cand_texts = req.candidate_texts if req.candidate_texts else req.candidates
    _encode_cached(cand_texts)
    return {"status": "ok", "cached": len(cand_texts), "cache_size": emb_cache.size}


@app.post("/cache/clear")
async def cache_clear():
    emb_cache.clear()
    return {"status": "cleared"}


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    with _SRV_STATS_LOCK:
        s = dict(_SRV_STATS)
    return CacheStatsResponse(size=emb_cache.size, max_size=emb_cache.max_size,
                              device=DEVICE, resp_cache_size=len(_RESP_CACHE),
                              resp_cache_max=_RESP_CACHE_MAX, **s)


@app.get("/health")
async def health():
    return {"status": "healthy", "device": DEVICE, "model_loaded": model is not None, "cache_size": emb_cache.size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
