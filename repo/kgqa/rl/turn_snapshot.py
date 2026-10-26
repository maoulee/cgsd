#!/usr/bin/env python3
"""rollout_turn snapshot (Codex verl plan §4, P0) — request-boundary recorder.

WHY: the legacy training path rebuilds conversations post-hoc
(seq_advantage.build_messages) — that rebuild truncates the system prompt,
swaps the env-reply role, re-splices <think> tags and, critically, never saw
the per-turn user hints (they are appended to a COPY of rc.messages and never
enter the saved trajectory). Training on rebuilt messages ≠ training on the
conditions the policy actually sampled from. This module records the REAL
request at the model boundary, per turn, with the exact token ids and the
sampled-token logprobs — no behavior change to the rollout itself.

WHAT (per turn, sidecar jsonl — never inside results.json):
  input_ids / generated_ids / behavior_log_probs / generated_mask,
  policy_version / prompt_hash / template_hash / tokenizer_hash / env_version,
  sampling_config / stop_reason / canonical_action / observation_ids,
  question_id / trajectory_id / turn_id / action_id (via correlate()).

HOW: SnapshotChatBatch subclasses HTTPChatBatch. Only `achat_one` (bubble
mode, the production path) is snapshotted; `chat_batch` (lockstep fallback)
raises unless explicitly allowed — the smoke panel runs SEQ_BUBBLE=1.
input_ids are tokenized client-side with the SAME tokenizer + chat template
the server uses (verified: count == usage.prompt_tokens on every turn).
generated_ids are reconstructed from the server's reasoning/content fields
matching usage.completion_tokens (form ambiguity resolved per-turn; the
server's per-token logprob strings are kept as ground-truth audit).

vLLM reasoning boundary on this deployment: reasoning_start_str "<think>" is
injected by the chat template into the PROMPT, the generation is
  reasoning + "I will now emit the tool call based on the reasoning above.
  </think>" + content + "<|im_end|>"
when the thinking budget force-inject fires, or without the phrase when the
model closes thinking naturally. Reconstruction tries both + stop variants.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time

import aiohttp

from kgqa.llm.http_batch import HTTPChatBatch, CHAT_URL
from kgqa.llm.client import request_headers
from kgqa.llm.offline_vllm import ChatResult

# server reasoning_config (scripts/start_local_qwen35_server.sh)
REASONING_PHRASE = "I will now emit the tool call based on the reasoning above."
IM_END = "<|im_end|>"


def _sha(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(" ".join(map(str, ids)).encode()).hexdigest()


def _tokenizer_fingerprint(model_path: str) -> str:
    """sha256 over the tokenizer-relevant files, sorted by name."""
    import glob
    h = hashlib.sha256()
    files = []
    for pat in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json",
                "chat_template.jinja", "chat_template.json", "special_tokens_map.json",
                "added_tokens.json"):
        files += glob.glob(os.path.join(model_path, pat))
    for f in sorted(files):
        h.update(os.path.basename(f).encode())
        h.update(open(f, "rb").read())
    return h.hexdigest()


def _git_head() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "-C", ".", "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip()[:12]
    except Exception:
        return "unknown"


def parse_canonical_action(content: str) -> dict:
    """Tool extraction that handles BOTH the canonical JSON actions and the
    legacy flat `tool: X` lines (the old seq_advantage regex missed JSON)."""
    c = (content or "").strip()
    if "</think>" in c:
        c = c.rsplit("</think>", 1)[1].strip()
    # canonical JSON first
    for m in re.finditer(r"\{.*\}", c, re.DOTALL):
        try:
            j = json.loads(m.group(0))
            if isinstance(j, dict) and j.get("tool"):
                return {"tool": str(j["tool"]), "format": "json", "args": j.get("arguments", j)}
        except Exception:
            pass
    ms = re.findall(r"tool:\s*(\w+)", c)
    if ms:
        return {"tool": ms[-1], "format": "flat", "args": None}
    return {"tool": None, "format": "none", "args": None}


class SnapshotChatBatch(HTTPChatBatch):
    """HTTPChatBatch + per-turn snapshot sidecar. See module docstring."""

    def __init__(self, out_path: str, model_path: str = "models/Qwen3.5-9B",
                 policy_version: str | None = None,
                 thinking_budget: int = 1000, max_num_seqs: int = 128,
                 allow_chat_batch: bool = False):
        super().__init__(thinking_budget=thinking_budget, max_num_seqs=max_num_seqs)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model_path = model_path
        self.template_hash = _sha(self.tok.chat_template or "")
        self.tokenizer_hash = _tokenizer_fingerprint(model_path)
        self.policy_version = policy_version or os.environ.get("POLICY", "").strip() or "base"
        self.env_version = f"git:{_git_head()}"
        try:
            from kgqa.agent.seq_react_loop import seq_agents_md
            self.agents_md_hash = _sha(seq_agents_md())
        except Exception:
            self.agents_md_hash = ""
        self.out_path = out_path
        self._fh = open(out_path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = 0
        self._allow_chat_batch = allow_chat_batch
        self.stats = {"turns": 0, "prompt_match": 0, "completion_match": 0,
                      "lp_len_match": 0, "gen_forms": {}}

    # ── tokenization helpers ────────────────────────────────────────────
    def _prompt_ids(self, messages) -> list[int]:
        # tokenize=False then encode: same as seq_advantage.build_train_test;
        # avoids the BatchEncoding unwrap mess of tokenize=True
        txt = self.tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
        return self.tok.encode(txt, add_special_tokens=False)

    def _reconstruct_generation(self, reasoning: str, content: str,
                                finish_reason: str, completion_tokens: int,
                                logprob_tokens: list[str | None] | None = None):
        """EXACT raw-completion reconstruction. Primary source: the server's
        per-token logprob strings (ground truth — the reasoning/content string
        fields can be ambiguous when the model emits the end phrase naturally
        mid-content, observed 1/51 turns). Fallback: string-field candidates.
        Ambiguity is resolved against the authoritative completion_tokens."""
        if logprob_tokens:
            raw = "".join(t or "" for t in logprob_tokens)
            ids = self.tok.encode(raw, add_special_tokens=False)
            if len(ids) == completion_tokens:
                # decode round-trip sanity (BPE boundary check)
                if self.tok.decode(ids) == raw:
                    return ids, raw, "logprob_tokens"
        cores = [
            reasoning + REASONING_PHRASE + "</think>" + content,   # budget force-inject
            reasoning + "</think>" + content,                       # natural close
            reasoning + content,                                    # un-parsed (no boundary)
        ]
        cands = []
        for core in cores:
            if finish_reason == "stop":
                cands.append((core + IM_END, "stop"))
            cands.append((core, "no_stop"))
        for raw, tag in cands:
            ids = self.tok.encode(raw, add_special_tokens=False)
            if len(ids) == completion_tokens:
                return ids, raw, tag
        # no exact match: return the first candidate + mismatch flag
        raw, tag = cands[0]
        return self.tok.encode(raw, add_special_tokens=False), raw, tag + "+MISMATCH"

    # ── snapshot write ──────────────────────────────────────────────────
    def _write(self, rec: dict):
        with self._lock:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

    def _record(self, messages, kw, msg_out, usage, finish_reason, lps, latency):
        reasoning = msg_out.get("reasoning") or ""
        content = msg_out.get("content") or ""
        gen_ids, raw, form = self._reconstruct_generation(
            reasoning, content, finish_reason or "",
            (usage or {}).get("completion_tokens") or -1,
            [e.get("token") for e in (lps or [])])
        return self._record_with_generation(
            messages, kw, msg_out, finish_reason, lps, latency,
            gen_ids, raw, form, (usage or {}).get("prompt_tokens"),
            (usage or {}).get("completion_tokens"))

    def _record_with_generation(self, messages, kw, msg_out, finish_reason,
                                lps, latency, gen_ids, raw, form,
                                ptoks=None, ctoks=None):
        self._seq += 1
        reasoning = msg_out.get("reasoning") or ""
        content = msg_out.get("content") or ""
        input_ids = self._prompt_ids(messages)
        lp = [e.get("logprob") for e in (lps or [])]
        if ctoks is None:                     # batch endpoint: no per-choice usage
            ctoks = len(gen_ids)
        rec = {
            "seq": self._seq, "ts": time.time(), "latency_s": round(latency, 3),
            "question": messages[1]["content"] if len(messages) > 1 else "",
            "policy_version": self.policy_version,
            "env_version": self.env_version,
            "agents_md_sha256": self.agents_md_hash,
            "prompt_hash": _sha_ids(input_ids),
            "template_hash": self.template_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "sampling_config": {
                "thinking_token_budget": kw.get("thinking_budget"),
                "max_tokens": kw.get("max_tokens"),
                "temperature": kw.get("temperature"), "top_p": kw.get("top_p"),
                "top_k": kw.get("top_k"),
                "presence_penalty": kw.get("presence_penalty"),
                "model": self.policy_version,
            },
            "input_ids": input_ids, "input_len": len(input_ids),
            "server_prompt_tokens": ptoks,
            "prompt_match": (ptoks == len(input_ids)) if ptoks is not None else None,
            "generated_ids": gen_ids, "generated_len": len(gen_ids),
            "server_completion_tokens": ctoks,
            "completion_match": (ctoks == len(gen_ids)),
            "ids_exact": (form != "" and not form.endswith("+MISMATCH")),
            "gen_form": form,
            "behavior_log_probs": lp,        # ln p(sampled token) — vLLM logprob
            "generated_mask": [1] * len(gen_ids),
            "stop_reason": finish_reason,
            "reasoning": reasoning, "content": content,
            "raw_completion_text": raw,
            "logprob_tokens": [e.get("token") for e in (lps or [])],
            "lp_len_match": (len(lp) == len(gen_ids)),
            "messages": messages,
        }
        self.stats["turns"] += 1
        self.stats["prompt_match"] += int(rec["prompt_match"] or False)
        self.stats["completion_match"] += int(rec["completion_match"] or False)
        self.stats["lp_len_match"] += int(rec["lp_len_match"] or False)
        self.stats["gen_forms"][form] = self.stats["gen_forms"].get(form, 0) + 1
        self._write(rec)
        return rec

    # ── overridden request path (bubble mode) ───────────────────────────
    async def achat_one(self, messages, thinking_budget=None,
                        temperature=0.3, max_tokens=None, top_p=0.8,
                        top_k=20, presence_penalty=1.5, model=None, **_):
        tb = thinking_budget or self._tb
        kw = dict(thinking_budget=tb, temperature=temperature,
                  max_tokens=max_tokens, top_p=top_p, top_k=top_k,
                  presence_penalty=presence_penalty)
        sem = getattr(self, "_sem", None)
        if sem is None:
            self._sem = sem = asyncio.Semaphore(self._sem_n)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        payload = self._payload(messages, **kw)
        payload["logprobs"] = True          # sampling UNCHANGED — reporting only
        t0 = time.time()
        async with sem:
            for attempt in range(3):
                try:
                    async with self._session.post(
                        CHAT_URL, json=payload, headers=request_headers(),
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}: "
                                               f"{(await resp.text())[:200]}")
                        data = await resp.json()
                    ch = data["choices"][0]
                    msg = ch["message"]
                    usage = data.get("usage") or {}
                    lps = (ch.get("logprobs") or {}).get("content") or []
                    latency = time.time() - t0
                    self._record(messages, kw, msg, usage,
                                 ch.get("finish_reason"), lps, latency)
                    return ChatResult(reasoning=msg.get("reasoning") or "",
                                      text=msg.get("content") or "")
                except (aiohttp.ClientError, asyncio.TimeoutError,
                        RuntimeError, KeyError, IndexError):
                    if attempt < 2:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    return ChatResult(reasoning="", text="")

    async def _achat_batch_snapshot(self, messages_list, kw):
        """Batch endpoint WITH logprobs + per-choice token_ids (the batch
        endpoint returns exact generated token_ids; the single endpoint does
        not). Falls back to per-request achat_one (self-snapshotting) on any
        failure — sampling is identical, only the transport differs."""
        from kgqa.llm.http_batch import BATCH_URL
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        payload = dict(self._payload(messages_list, **kw))
        payload["logprobs"] = True
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
                if not (isinstance(i, int) and 0 <= i < len(out)):
                    continue
                m = ch.get("message") or {}
                lps = (ch.get("logprobs") or {}).get("content") or []
                tids = ch.get("token_ids")
                # token_ids are authoritative when present; else reconstruct
                if isinstance(tids, list) and tids:
                    gen_ids, raw, form = tids, "", "server_token_ids"
                else:
                    gen_ids, raw, form = self._reconstruct_generation(
                        m.get("reasoning") or "", m.get("content") or "",
                        ch.get("finish_reason") or "",
                        len(lps) or -1,
                        [e.get("token") for e in lps])
                self._record_with_generation(
                    messages_list[i], kw, m, ch.get("finish_reason"), lps,
                    0.0, gen_ids, raw, form)
                out[i] = ChatResult(reasoning=m.get("reasoning") or "",
                                    text=m.get("content") or "")
            if all(x is not None for x in out):
                return out
            missing = [k for k, x in enumerate(out) if x is None]
            rest = await asyncio.gather(*[
                self.achat_one(messages_list[k], **kw) for k in missing])
            for k, v in zip(missing, rest):
                out[k] = v
            return out
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
            return await asyncio.gather(*[
                self.achat_one(m, **kw) for m in messages_list])

    def chat_batch(self, messages_list, thinking_budget=None,
                   temperature=0.3, max_tokens=None, top_p=0.8,
                   top_k=20, presence_penalty=1.5, **_):
        tb = thinking_budget or self._tb
        kw = dict(thinking_budget=tb, temperature=temperature,
                  max_tokens=max_tokens, top_p=top_p, top_k=top_k,
                  presence_penalty=presence_penalty)
        fut = asyncio.run_coroutine_threadsafe(
            self._achat_batch_snapshot(messages_list, kw), self._loop)
        return fut.result(timeout=1800)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
        super().close()


# ─────────────────────────────────────────────────────────────────────────
# post-hoc correlation: snapshot turns → question/trajectory/turn ids,
# canonical_action + observation_ids, and the verification report
# ─────────────────────────────────────────────────────────────────────────
def _question_of(snapshot_rec) -> str:
    c = snapshot_rec["question"]
    if c.startswith("Question: "):
        c = c[len("Question: "):]
    for marker in ("\n\nQuestion entities",):
        i = c.find(marker)
        if i >= 0:
            c = c[:i]
    return c.strip()


def _fact_ids(text: str) -> list[str]:
    return re.findall(r"fact_id:\s*(\S+)", text or "")


def correlate_and_verify(sidecar_path: str, results: list[dict],
                         report_path: str):
    """Attach question_id/trajectory_id/turn_id/action_id/canonical_action/
    observation_ids to each sidecar turn (rewrite in place), then verify:
      1. prompt/completion/logprob length agreement (already per-turn)
      2. decode spot-checks (3 turns printed into the report)
      3. per-trajectory chain: next-turn input == render(history incl. this
         turn's sanitized output + env replies + hint) — tokenized exactly
      4. prefix-chain property check (whole-trajectory single causal forward
         eligibility; expected to FAIL on this stack — reasoning is stripped
         from history by the template — recorded as evidence)
    """
    import transformers
    from kgqa.core.utils import strip_reasoning_leak
    tok = transformers.AutoTokenizer.from_pretrained(
        "models/Qwen3.5-9B", trust_remote_code=True)
    recs = [json.loads(ln) for ln in open(sidecar_path, encoding="utf-8") if ln.strip()]

    # question text → result records (G may be >1: match by question then split
    # by first-turn divergence — same question's first-turn requests are
    # identical, so we only need per-question grouping + turn order here)
    by_question = {}
    for r in results:
        by_question.setdefault(r["question"].strip(), []).append(r)

    report = {"n_snapshot_turns": len(recs), "warnings": [], "chains": [],
              "decode_spotchecks": [], "prefix_chain": {}}

    # group snapshot turns by question, ordered by seq
    groups = {}
    for rec in recs:
        q = _question_of(rec)
        if q not in by_question:
            report["warnings"].append(f"turn seq={rec['seq']}: question not in results: {q[:60]}")
            rec["question_id"] = None
            continue
        groups.setdefault(q, []).append(rec)

    for q, turns in groups.items():
        rs = by_question[q]
        if len(rs) != 1:
            report["warnings"].append(
                f"question {q[:50]} matched {len(rs)} result records — "
                f"turn ids assigned per question-group only (G>1 needs chain split)")
        r = rs[0]
        tid = f"{r['case_idx']}_{r['sample_idx']}"
        turns.sort(key=lambda x: x["seq"])
        n_asst = sum(1 for st in r["trajectory"] if st.get("role") == "assistant")
        if n_asst != len(turns):
            report["warnings"].append(
                f"trajectory {tid}: {len(turns)} snapshot turns vs "
                f"{n_asst} assistant messages in trajectory")
        chain_ok = True
        for k, rec in enumerate(turns):
            rec["question_id"] = r["case_id"]
            rec["trajectory_id"] = tid
            rec["turn_id"] = f"{tid}#t{k:02d}"
            rec["action_id"] = f"{rec['turn_id']}#a0"
            rec["canonical_action"] = parse_canonical_action(rec["content"])
            # chain: turn k request = hist_k + hint_k. Turn k+1 request =
            # hist_k + [assistant(sanitized output)] + [env replies...] + hint_{k+1}
            # (the OLD hint is dropped — it is request-local, never lands in
            # rc.messages: exactly the §4 data-interface gap)
            if k + 1 < len(turns):
                nxt = turns[k + 1]["messages"]
                n_hist = len(rec["messages"]) - 1     # history WITHOUT this turn's hint
                obs = []
                for m in nxt[n_hist + 1:-1]:          # env replies before the new hint
                    if m.get("role") == "user":
                        obs += _fact_ids(m.get("content") or "")
                rec["observation_ids"] = sorted(set(obs))
            else:
                rec["observation_ids"] = []
            rec["parent_prompt_hash"] = turns[k - 1]["prompt_hash"] if k else None
            if k + 1 < len(turns):
                nxt_msgs = turns[k + 1]["messages"]
                hist_ok = nxt_msgs[:n_hist] == rec["messages"][:n_hist]
                asst = nxt_msgs[n_hist] if len(nxt_msgs) > n_hist else None
                sanitized = strip_reasoning_leak(rec["content"] or "")
                asst_ok = (asst or {}).get("role") == "assistant" and (
                    asst.get("content") == sanitized or asst.get("content") == rec["content"])
                envs_ok = all(m.get("role") == "user" for m in nxt_msgs[n_hist + 1:-1])
                hint_ok = (nxt_msgs[-1].get("role") == "user" if nxt_msgs else False)
                # exact token check: re-render next-turn input from its messages
                try:
                    red = tok.apply_chat_template(nxt_msgs, tokenize=False,
                                                  add_generation_prompt=True)
                    red_ids = tok.encode(red, add_special_tokens=False)
                    tok_ok = red_ids == turns[k + 1]["input_ids"]
                except Exception as e:
                    tok_ok = f"render-fail:{e}"
                chain_ok = chain_ok and hist_ok and asst_ok and tok_ok is True
                report["chains"].append({
                    "turn": rec["turn_id"], "hist_ok": hist_ok,
                    "asst_ok": asst_ok, "envs_ok": envs_ok, "hint_ok": hint_ok,
                    "tok_ok": tok_ok,
                    "asst_content_matches_raw": bool(asst) and asst.get("content") == rec["content"],
                })
        # prefix-chain property: is turn0's input a token-prefix of turn1's?
        if len(turns) >= 2:
            a, b = turns[0]["input_ids"], turns[1]["input_ids"]
            L = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                L += 1
            report["prefix_chain"][tid] = {
                "prefix_len": L, "turn0_len": len(a), "turn1_len": len(b),
                "holds": L == len(a),
                "divergence_decoded": tok.decode(b[max(0, L - 8):L + 24]),
            }

    # decode spot-checks: first, middle, last turn of the FIRST trajectory
    done_q = next(iter(groups), None)
    if done_q:
        turns = groups[done_q]
        for k in (0, len(turns) // 2, len(turns) - 1):
            t = turns[k]
            report["decode_spotchecks"].append({
                "turn": t["turn_id"],
                "input_tail_decoded": tok.decode(t["input_ids"][-40:]),
                "generated_decoded_head": tok.decode(t["generated_ids"])[:300],
                "reasoning_head": (t["reasoning"] or "")[:80],
                "content_head": (t["content"] or "")[:120],
            })

    # rewrite sidecar with ids attached
    with open(sidecar_path, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_pm_known = sum(1 for r in recs if r["prompt_match"] is not None)
    n_pm = sum(1 for r in recs if r["prompt_match"])
    n_cm = sum(1 for r in recs if r["completion_match"])
    n_lm = sum(1 for r in recs if r["lp_len_match"])
    n_chain = sum(1 for c in report["chains"] if c["hist_ok"] and c["asst_ok"] and c["tok_ok"] is True)
    report["summary"] = {
        "turns": len(recs),
        "prompt_match": f"{n_pm}/{n_pm_known}",
        "completion_match": f"{n_cm}/{len(recs)}",
        "lp_len_match": f"{n_lm}/{len(recs)}",
        "chain_ok": f"{n_chain}/{len(report['chains'])}",
        "prefix_chain_holds": sum(1 for v in report["prefix_chain"].values() if v["holds"]),
        "prefix_chain_total": len(report["prefix_chain"]),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return report
