"""SEQ ReAct loop — the per-fact sequential pipeline (parallel to react_loop).

Copies react_loop's GENERIC turn body (parse → validate → dispatch → append) and
rewires three things only:
  - state machine  → seq_harness (INIT → decompose → RESOLVE_FACT loop → ANSWER)
  - dispatch       → seq_tools (decompose grounds f1; resolve_fact walks one hop +
                     auto-penetrates CVTs + grounds the next fact on real entities)
  - system prompt  → SEQ_AGENTS.md

parse_react_output / _to_tool_calls / _call_single_with_reasoning / build_context /
_ctx_to_result_dict are REUSED from the SAPS code paths unchanged. SAPS react_loop,
harness, tools, AGENTS.md are not touched.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp

from kgqa.agent.loop import build_context, _ctx_to_result_dict
from kgqa.core.utils import strip_reasoning_leak
from kgqa.agent.seq_harness import (
    SeqAgentState, validate as seq_validate, _allowed_hint as seq_allowed_hint,
    commit_due, blocking_facts, norm_fact_key, ready_to_commit,
    unconcluded_facts,
)
from kgqa.agent import seq_tools as ST
from kgqa.agent import seq_protocol as SP
import copy
from kgqa.agent.seq_schemas import validate_args
from kgqa.agent.react_loop import parse_react_output, _to_tool_calls  # reuse verbatim

_AGENT_DIR = Path(__file__).resolve().parent


def _history_with_thinking(raw_response: str, reasoning: str | None) -> str | None:
    """Rebuild the full assistant turn for history (opt-in, eval parity).

    Production vLLM separates ``content`` and ``reasoning``; the pre-2026-10-22
    eval loops appended only content, so prior thinking was invisible to the
    model. The training scheduler appends ``<think>...</think>\\n\\n<action>``.
    ``SEQ_HISTORY_THINKING=1`` makes the pure-vLLM paths use the same shape.
    """
    if os.environ.get("SEQ_HISTORY_THINKING", "0").strip().lower() not in (
            "1", "true", "on", "yes"):
        return None
    text = raw_response or ""
    if "<think>" in text:
        return text
    thought = (reasoning or "").strip()
    if not thought:
        return None
    return f"<think>\n{thought}\n</think>\n\n{text}"


# Model checkpoint declaration — MD-LINE PRIMARY (user ruling 2026-09-20):
#   - sg1.f1 ✓ ?var = v1 | v2 | v3
#   - sg1.f2 ✗ empty
# Models follow markdown far more reliably than bespoke bracket grammar (the
# 2152 specimen burned 6 turns on bracket-format retries). The legacy bracket
# forms (`[fid ✓] ?var = [v1 | v2]`, `[fid ✗ empty]`) remain accepted for
# transition compatibility — old trajectories and tests still parse.
_CKPT_RE = re.compile(r'\[[^\]]*?✓\]\s*(\?\w+)\s*=\s*\[([^\]]*)\]')
_CKPT_RE_MD = re.compile(
    r'(?:^|\n)\s*[-*]?\s*(?:sg\d+(?:\.f\d+)?|f\d+)\s*[✓✔]\s*(\?\w+)'
    r'\s*=\s*\[?([^\]\n]*)')

# Plan-contract v2 closure declarations (2026-08-21): a fact may be CLOSED without
# bindings — `✗ empty` (no advancing relation/content in the reachable pool)
# or `✗ moot` (the terminal variable was already bound by earlier evidence).
# Closures free the model from the complete-every-fact contract (the Ramble/VP
# specimen looped 16 rounds on a dead-end fact and scored empty).
# Wave-1 (2026-09-20): the status set is a CLOSED enumeration — `mismatch`
# (queried bindings never matched the fact's evidence) and `exhausted` (the
# relation pool was tried out with nothing advancing) join empty/moot, and an
# optional `: reason` suffix is tolerated so a motivated closure line still
# parses. Anything outside the enumeration is NOT a closure (see the unknown-
# closure warning at the checkpoint ack) — the old grammar silently dropped
# free-text statuses and completion accounting kept the fact open.
# MD-line closures (user ruling): `- sg1.f2 ✗ empty` accepted alongside the
# bracket form.
_CKPT_CLOSE_RE = re.compile(r'\[([^\]]*?)✗\s*(empty|moot|mismatch|exhausted)(?::[^\]]*)?\]')
_CKPT_CLOSE_RE_MD = re.compile(
    r'(?:^|\n)\s*[-*]?\s*((?:sg\d+(?:\.f\d+)?|f\d+))\s*✗\s*'
    r'(empty|moot|mismatch|exhausted)(?:\s*:[^\n]*)?\s*$', re.M)


def _canonical_decl_fid(ctx, raw: str, var: str) -> str:
    """Canonical declared fact id for a checkpoint declaration's fid slot.

    The model sometimes copies the DOC PLACEHOLDER (`[fid ✓]`, `[fid ✗ empty]`)
    instead of a real fact id — the declaration then lands under a junk key
    that fact_key_map/norm_fact_key never resolve, so completion accounting
    misses the fact and the EVIDENCE COMMIT never fires (22% of fuse-on cohort
    trajectories, 2026-08-22). Resolution: real ids pass through fact_key_map;
    the placeholder family resolves via the DECLARED VARIABLE against the
    plan's fact_edges (the open fact having that var on either side), falling
    back to the single declared fact for var-less ✗ closures. The mapping is
    memoized per raw key on ctx so re-statements hit the same freeze-lock
    entry regardless of how the open/declared sets shift later."""
    k = (raw or "").strip()
    if not k:
        return k
    alias = getattr(ctx, "_fid_alias", None)
    if alias is None:
        alias = {}; ctx._fid_alias = alias
    if k in alias:
        return alias[k]
    km = getattr(ctx, "fact_key_map", None) or {}
    fids = getattr(ctx, "fact_ids", None) or []
    if k in km:
        alias[k] = km[k]
        return km[k]
    if k in fids or "." in k:
        alias[k] = k
        return k
    if k.lower() in ("fid", "f", "fact", "id"):
        edges = getattr(ctx, "fact_edges", None) or {}
        taken = set(getattr(ctx, "declared_facts", None) or {}) | \
            set(getattr(ctx, "closed_facts", None) or {})
        # tail-exact first: a ✓ declaration binds the fact's OUTPUT side; a
        # chain var (f1 tail ?x, f2 head ?x) is ambiguous under plain
        # membership but unique under tail-matching
        tail_m = [f for f, ht in edges.items()
                  if var and len(ht) >= 2 and ht[1] == var and f not in taken]
        any_m = [f for f, ht in edges.items()
                 if var and var in ht and f not in taken]
        pick = None
        if len(tail_m) == 1:
            pick = tail_m[0]
        elif len(any_m) == 1:
            pick = any_m[0]
        elif not var and len(fids) == 1:
            pick = fids[0]      # var-less ✗ closure on a single-fact plan
        if pick:
            alias[k] = pick
            return pick
    return k


def _join_v2() -> bool:
    """R3-a (2026-10-01): JOIN/COMMIT predicate switch (user ruling §6.8-3 /
    §7-R3). ON (default): JOIN triggers on the DeclarationLedger's closure
    state with views.join_projection as the single data source (empty
    intersection ⇒ the environment owes a CONNECTING PATH; single-side
    closure is reported, never silent), and EVIDENCE COMMIT triggers on
    all-facts-concluded (declaration clock). OFF: the checkpoint-content
    JOIN / retrieval-clock COMMIT behavior is preserved byte-for-byte — the
    rollback + A/B face."""
    return os.environ.get("SEQ_JOIN_V2", "1").strip().lower() \
        not in ("0", "off", "false", "no")


def _observed_episode(ctx) -> set:
    """OBSERVED-EPISODE domain for the hallucination filter (D3-0, 2026-10-01):
    normalized endpoint names of every triple delivered this episode, from the
    EvidenceRecord ledger's observed view (evidence.observed_episode — walk
    endpoints + CVT attr VALUE endpoints). An EMPTY set is meaningful, not a
    skip signal: nothing has been delivered, so nothing can support a
    declaration (F08 legacy hole closed — the old `if fe:` skipped the check
    exactly when the domain was empty)."""
    _rlog = getattr(ctx, "evidence_records", None)
    if _rlog is None:
        return set()
    return _rlog.observed_episode()


def _halluc_strip_receipt(ctx, fid, bad, hit_exact, hit_substr) -> str:
    """Receipt for checkpoint values removed by the hallucination filter
    (observed_episode domain, D3-0). Submission-first (ruling 2026-09-21):
    name the violation AND what to submit — the model's ORIGINAL declared
    strings ride verbatim in `bad` (never silently rewritten), the episode
    scope and the no-alias rule are stated so an old-name re-declaration
    knows exactness is what counts, and the kept-value hit split (exact vs
    len≥4 substring) surfaces the F08 over-admission face for measurement."""
    _hstand = ((getattr(ctx, "fact_bindings", None) or {}).get(fid)) or []
    _hvar = ((getattr(ctx, "fact_vars", None) or {}).get(fid)) or \
        "the fact's variable"
    _pool = ((getattr(ctx, "fact_evidence", None) or {}).get(fid)) or []
    _pool_str = (" | ".join(list(_pool)[:10])
                 + (" …" if len(_pool) > 10 else "")) if _pool else \
        "(that subgraph displayed no named entities)"
    _hits = (f" The other declared values were kept ({hit_exact} exact-name"
             f" match(es), {hit_substr} substring match(es))."
             if (hit_exact or hit_substr) else "")
    return (f"⚠ Your checkpoint [{fid} ✓] bound entities that NEVER "
            f"appeared in ANY evidence delivered this episode: {bad}. "
            "They were removed — the episode's delivered triples contain no "
            "such endpoint (checked by name: an alias or old name does NOT "
            "count as appeared)." + _hits +
            (f"The standing bindings are unchanged:\n"
             f"- {fid} ✓ {_hvar} = "
             + " | ".join(_hstand[:12])
             + (" …" if len(_hstand) > 12 else "") + "\n"
             if _hstand else "\n")
            + f"SUBMIT values drawn from {fid}'s OWN evidence pool "
            f"(includes: {_pool_str}) — re-declare "
            f"`- {fid} ✓ {_hvar} = <values from that pool>` or answer "
            "from the standing bindings.")


def _update_system_join_v2(ctx, state=None) -> None:
    """SYSTEM JOIN V2 (R3-a, user ruling 2026-10-01 §6.8-3): the trigger is
    the DeclarationLedger's CLOSURE STATE, not the checkpoint content clock
    — every fact of the variable's join group (both trees) concluded (✓
    binding OR explicit ✗; harness-side closures count) ∧ the sides not yet
    merged. The data source is views.join_projection — the OUTPUT-variable
    projection of each tree's delivered evidence (the single legal
    implementation); checkpoint CONTENT gates nothing, so a side that
    retrieved-but-cannot-bind (3084 sg2 face) closes `✗ empty` and the JOIN
    still fires. A single concluded side is REPORTED once (the old
    `len(_fids) < 2 → continue` silence is what left 36/58 join-type plans
    with no JOIN at all).

    Fid namespacing (3084 s1/s2 specimens, first p2a batch): the model may
    number facts GLOBALLY while fact_key_map is per-sg (`sg2.f2` = the third
    plan fact, not the second of sg2) — raw declaration keys then miss the
    map. Group membership and the conclusion set therefore canonicalize
    every key through norm_fact_key (the same single source COMMIT uses;
    D2-4 direction) and DROP keys that resolve outside state.fact_ids — a
    junk-keyed declaration cannot open or close a join side."""
    from collections import defaultdict as _dj
    from kgqa.agent import views as _views
    from kgqa.agent.seq_harness import norm_fact_key as _nfk
    _fb = getattr(ctx, "fact_bindings", None) or {}
    _fv = getattr(ctx, "fact_vars", None) or {}
    _edges = getattr(ctx, "fact_edges", None) or {}
    _fid_set = {f for f in (getattr(ctx, "fact_ids", None) or [])}

    def _canon(k):
        # canonical declared fid: fact_key_map first, then the harness's
        # bare-sg resolution (identical rule to COMMIT's clock — the two
        # predicates must not disagree on which fact a declaration closes)
        if k in _fid_set:
            return k
        c = (getattr(ctx, "fact_key_map", None) or {}).get(k)
        if c in _fid_set:
            return c
        if state is not None:
            c = _nfk(state, k)
            if c in _fid_set:
                return c
        return None                      # outside the plan — junk key

    # conclusion set: ✓ binding (an accepted checkpoint with content) ∪
    # explicit ✗ (closed_facts, any status word incl. harness-side purity/CVT
    # closures) ∪ ✓ locks (declared_facts) — all canonicalized
    _concluded = set()
    for _k in list(_fb) + list(getattr(ctx, "closed_facts", None) or {}) \
            + list(getattr(ctx, "declared_facts", None) or {}):
        _c = _canon(_k)
        if _c is not None:
            _concluded.add(_c)
    # join groups keyed by OUTPUT variable: the plan edge TAIL is the binding
    # site (a chained fact's HEAD var is a walk-on site, not a join side);
    # a ✓-declared fact missing from fact_edges keeps its declared var, but
    # ONLY under its canonical fid (junk keys never group)
    _by_var = _dj(set)
    for _f in _fid_set:
        _tail = (_edges.get(_f) or ("", ""))[1]
        if isinstance(_tail, str) and _tail.startswith("?"):
            _by_var[_tail].add(_f)
    for _f, _v in _fv.items():
        _c = _canon(_f)
        if _c is not None and _c not in _edges:
            _by_var[_v].add(_c)
    _joins = getattr(ctx, "var_joins", None)
    if _joins is None:
        _joins = {}; ctx.var_joins = _joins
    _log = getattr(ctx, "evidence_records", None)
    for _v, _fids in _by_var.items():
        if len(_fids) < 2:
            continue          # one tree binds the var — no join to report
        _fids = sorted(_fids)
        _done = [f for f in _fids if f in _concluded]
        _wait = [f for f in _fids if f not in _concluded]
        if _wait:
            # SINGLE-SIDE CLOSURE, no longer silent: say which side is
            # concluded and what the JOIN still waits for — once per
            # variable (episode-scope latch; reset clears it).
            if _done:
                _noted = getattr(ctx, "_join2_side_noted", None)
                if _noted is None:
                    _noted = set(); ctx._join2_side_noted = _noted
                if _v not in _noted:
                    _noted.add(_v)
                    ctx._join2_side_flag = (_v, list(_done), list(_wait))
            continue
        # BOTH TREES DECLARED COMPLETE → the join answer is due. Intersection
        # on the output-variable projection (views.join_projection — the one
        # legal implementation); the EMPTY intersection is an ANSWER, not an
        # error — the consumption site owes the CONNECTING-PATH search
        # (§6.8-3) or the explicit no-path verdict.
        try:
            _inter = _views.join_projection(_log, _fids, _v)["intersection"] \
                if _log is not None else []
        except Exception:
            _inter = []
        _cur = (tuple(_fids), tuple(_inter))
        if _joins.get(_v) != _cur:
            ctx.join_flag = (_v, _fids, list(_inter))
        _joins[_v] = _cur
        # NOT ENFORCED either way (Thundera specimen, 2026-08-19): missing
        # evidence ≠ negative evidence. The intersection is AUTHORITATIVE as
        # displayed information but ADVISORY as control flow — same contract
        # as the V1 note.


def _update_var_bindings(ctx, content: str, state=None) -> None:
    """Parse the model's checkpoint declarations and merge into ctx.var_bindings
    (?variable -> bound entity names). The model declares the curated binding for a
    variable after the retrieve_subgraph that resolves it; downstream tool calls that
    reference `?var` are expanded from this map (seq_tools._expand_entities).
    Closure declarations (`[fid ✗ empty|moot]`) are recorded in ctx.closed_facts —
    they bind nothing, but the harness (and later hints) can distinguish "fact
    resolved" from "fact closed" from "fact still open".
    Wave-2 split (2026-09-20): fact_bindings/var_bindings are the LEDGER — the
    retrieval record, only written on the ✓ path (fresh evidence); a no-new-
    evidence narrowing writes ctx.answer_selection (the model's discrimination
    layer) instead, never the ledger."""
    if not content:
        return
    vb = getattr(ctx, "var_bindings", None)
    if vb is None:
        ctx.var_bindings = vb = {}
    cf = getattr(ctx, "closed_facts", None)
    if cf is None:
        ctx.closed_facts = cf = {}
    for m in _CKPT_CLOSE_RE.finditer(content):
        cf[_canonical_decl_fid(ctx, m.group(1), "")] = m.group(2)
    for m in _CKPT_CLOSE_RE_MD.finditer(content):
        cf[_canonical_decl_fid(ctx, m.group(1), "")] = m.group(2)
    # unified ✓ entries: (fid_raw, var, vals) from BOTH grammars — the md
    # line form carries its fid natively; the legacy bracket form re-extracts
    # it below (the formats are line-disjoint, so no double-count)
    _entries = []
    for m in _CKPT_RE.finditer(content):
        fid_m = re.match(r"\[([^\]]*?)✓\]", m.group(0))
        _entries.append(((fid_m.group(1) if fid_m else "").strip(),
                         m.group(1), m.group(2)))
    for m in _CKPT_RE_MD.finditer(content):
        _entries.append((m.group(0).strip().lstrip("-* ").split()[0],
                         m.group(1), m.group(2)))
    for _fid_raw, var, vals in _entries:
        parts = [p.strip() for p in re.split(r'\s*\|\s*', vals) if p.strip()]
        # CVT-BINDING GUARD (2026-08-21, Angelina awards specimen): the model
        # bound raw m./g. event nodes as variable values. Events are records,
        # never entities — strip them and flag for a harness reminder on the
        # next turn (the prose note alone did not stop the behavior).
        from kgqa.traversal.cvt import is_cvt_like as _is_cvt
        cvt_bound = [p for p in parts if _is_cvt(p)]
        _all_cvt = False
        if cvt_bound:
            # CVT-BINDING LEGAL (user ruling 2026-09-24): event nodes MAY
            # bind variables — they are legitimate MID-CHAIN references
            # (a bound CVT can center later retrievals within its 2-hop
            # relation pool; the pool's CVT/id transparency handles the
            # reach). They are NEVER the final ANSWER — the answer tool's
            # mid-strip (§7.5) stays the only gate. The old strip+warn
            # (2026-08-21) cost a re-declare round on every CVT binding
            # (537 specimen: 12 performance CVTs bound, rejected, re-typed).
            parts = parts          # binding kept as declared
            _all_cvt = False
        # PER-SUBGRAPH HALLUCINATION CHECK (2026-08-21, user proposal): every
        # binding of [fid ✓] must have appeared in THAT subgraph's displayed
        # evidence (ctx.fact_evidence[fid]). Entities from other subgraphs or
        # thin air (Costner specimen: Twilight merged into sg1.f1) are stripped
        # and reported — the model re-declares from the correct subgraph.
        # VARIABLE FREEZING (2026-08-21, user ruling): a VALID ✓ declaration
        # locks the fact's bindings; a later re-declaration without a NEW
        # retrieve_subgraph for that fid is rejected (the Costner specimen
        # mutated sg1.f1 at answer time). The lock opens only when fresh
        # evidence for the same fid arrives (fact_evidence_seq advanced).
        _fid_key = _canonical_decl_fid(ctx, _fid_raw, var)
        if _all_cvt and _fid_key:
            # CVT-CHAIN DEADLOCK (Norwood specimen): every value was an
            # event — this variable can never bind, and a later fact
            # planned to walk from it is stranded. Remember (var, fid) so
            # the unbound-variable error can point at the moot-closure
            # exit instead of the generic re-declare instruction (which
            # looped declare→error for 16 turns before this).
            if getattr(ctx, "cvt_empty_var", None) is None:
                ctx.cvt_empty_var = (var, _fid_key)
        _decl = getattr(ctx, "declared_facts", None)
        if _decl is None:
            _decl = {}; ctx.declared_facts = _decl
        _cur_seq = getattr(ctx, "fact_evidence_seq", {}).get(_fid_key, 0)
        # PER-SUBGRAPH HALLUCINATION FILTER runs BEFORE the freeze check: the
        # standing fact_bindings store POST-filter values, so the idempotence
        # comparison below must see the same filtered shape (otherwise a
        # faithful re-statement of a once-filtered declaration misreads as a
        # value change).
        # HALLUCINATION FILTER DOMAIN = observed_episode (D3-0, 2026-10-01).
        # Detection principle (user): a hallucination = the model declared
        # something NO delivered triple contains. The check therefore judges
        # against EVERY triple delivered this episode (EvidenceRecord ledger's
        # observed view: walk endpoints + CVT attr VALUE endpoints), NOT the
        # declaring fid's own fact_evidence — a value that arrived only as a
        # CVT bracket attribute (2784 'Juice' specimen) or via ANOTHER tree's
        # subgraph (cross-tree declaration is a placement question for the
        # multi-tree reminder, not a hallucination) is LEGAL here. By-name
        # only, NO alias folding (ruling 2026-10-01: old-name declarations
        # with only the current name delivered still strip — 2152 face).
        # Empty domain = nothing delivered = nothing supports the declaration
        # → strip (F08 legacy hole: the old `if fe:` skipped the check exactly
        # when the fid's evidence set was absent). fact_evidence itself is
        # untouched (other readers unchanged; disposal is a later phase).
        # MATCH PREDICATE unchanged (exact OR len≥4 substring, both ways):
        # substring is a known over-admission source (F08) — the strip
        # receipt reports exact/substring hit counts so the narrowing has a
        # measured surface; semantics NOT changed in D3-0.
        _obs = _observed_episode(ctx)
        halluc = []
        _hit_exact = _hit_substr = 0
        if _obs:
            from kgqa.core.utils import normalize as _norm
            keep = []
            for p in parts:
                _np = _norm(p)
                _ex = _np in _obs
                _sb = (not _ex
                       and any((len(_np) >= 4 and _np in e)
                               or (len(e) >= 4 and e in _np)
                               for e in _obs))
                if _ex or _sb:
                    keep.append(p)
                    if _ex:
                        _hit_exact += 1
                    else:
                        _hit_substr += 1
                else:
                    halluc.append(p)
            parts = keep
        else:
            halluc = list(parts)
            parts = []
        if _fid_key in _decl and _decl[_fid_key] == _cur_seq:
            # IDEMPOTENT RE-STATEMENT (Stop-motion specimen, 2026-08-19): the
            # model often re-states a standing declaration verbatim (possibly
            # re-ordered) when composing the §18 answer checklist. That is NOT
            # a hallucination — reject only when the VALUE SET actually moves.
            from kgqa.core.utils import normalize as _nz2
            _fb = getattr(ctx, "fact_bindings", {}).get(_fid_key) or []
            _new_s, _old_s = {_nz2(p) for p in parts}, {_nz2(p) for p in _fb}
            if _new_s == _old_s:
                continue                  # identical re-statement: keep the lock silently
            if _new_s <= _old_s:
                # REFINEMENT (Belgium specimen, 2026-08-22): narrowing the
                # standing set to a SUBSET is discrimination, not hallucination —
                # the two-stage analysis may collapse [W-Europe|Europe|Eurasia|
                # N-Hemisphere] to [Europe]. LEDGER vs SELECTION (Wave-2,
                # 2026-09-20): fact_bindings/var_bindings are the RETRIEVAL
                # RECORD — immutable absent fresh evidence (the ✓ path below
                # is their only writer). The narrowed subset lands in
                # ctx.answer_selection (dict var → list), the model's
                # discrimination layer; the answer-stage hints (EVIDENCE
                # COMMIT CANDIDATES, second-refusal basis) read it with the
                # ledger as fallback. The old overwrite made a curated pick
                # masquerade as what retrieval delivered.
                _sel = getattr(ctx, "answer_selection", None)
                if _sel is None:
                    _sel = ctx.answer_selection = {}
                _sel[var] = list(parts)
                continue
            ctx.frozen_binding_flag = (_fid_key, parts[:8])
            ctx.halluc_binding_flag = None    # the whole declaration is rejected; one message
            continue                      # frozen: no new evidence, keep the lock
        if halluc:
            # counts ride the flag so the receipt can separate exact-name
            # hits from len≥4 substring hits (F08 narrowing surface)
            ctx.halluc_binding_flag = (_fid_key, halluc[:8],
                                       _hit_exact, _hit_substr)
        if parts:
            vb[var] = parts
            _decl[_fid_key] = _cur_seq    # (re)lock at the current evidence seq
            # fresh evidence supersedes any discrimination recorded against
            # the OLD evidence — this ✓ declaration is the new retrieval-
            # driven pick; a later no-new-evidence narrowing re-populates the
            # selection (LEDGER vs SELECTION, Wave-2)
            _stale_sel = getattr(ctx, "answer_selection", None)
            if _stale_sel:
                _stale_sel.pop(var, None)
            # PATTERN-STATE (user design 2026-09-10): a variable declared from
            # this fact's subgraph carries THAT walk's node set — a later
            # single-?var retrieval continues over it as a prefix mask (the
            # bindings are the pattern's tails; their walks must not re-traverse
            # the territory the declaring walk already covered).
            _fpat = (getattr(ctx, "fid_pattern", None) or {}).get(_fid_key)
            if _fpat and _fpat.get("nodes"):
                _ps = getattr(ctx, "pattern_state", None)
                if _ps is None:
                    _ps = ctx.pattern_state = {}
                _prev = _ps.get(var)
                if _prev is None:
                    _ps[var] = {"nodes": _fpat["nodes"]}
                else:
                    _ps[var] = {"nodes": _prev["nodes"] | _fpat["nodes"]}
            # PER-FID binding store (join support): vb is keyed by VARIABLE, so
            # a second subgraph declaring the same var overwrites the first —
            # the harness join below needs each fact's OWN values to intersect.
            _fbind = getattr(ctx, "fact_bindings", None)
            if _fbind is None:
                _fbind = {}; ctx.fact_bindings = _fbind
            _fvars = getattr(ctx, "fact_vars", None)
            if _fvars is None:
                _fvars = {}; ctx.fact_vars = _fvars
            _fbind[_fid_key] = list(parts)
            _fvars[_fid_key] = var
            # Seed declared bindings into subgraph_entities. The model curated these from
            # a prior retrieve_subgraph (the checkpoint marks the fact resolved); they ARE
            # valid centers. Without seeding, the boundary check rejects bindings that
            # appeared in the dense tree but weren't accumulated into the subgraph set
            # (display/accumulate mismatch) — e.g. ?religion=[Catholicism|...] then
            # retrieve_relations(center:?religion) failed "not in retrieved subgraph".
            # Only seed bindings that resolve to a graph entity (hallucinated names stay out).
            se = getattr(ctx, "subgraph_entities", None)
            if se is not None and getattr(ctx, "ents", None):
                from kgqa.core.utils import normalize as _norm
                # multi-map: a name can match several indices (text+non_text dup); add ALL
                from collections import defaultdict as _dd
                n2i_all = _dd(list)
                for _i, _e in enumerate(ctx.ents):
                    if _e:
                        n2i_all[_norm(_e)].append(_i)
                for p in parts:
                    for idx in n2i_all.get(_norm(p), ()):
                        se.add(idx)

    # SYSTEM JOIN (user ruling, 2026-08-19): when TWO OR MORE subgraphs bind the
    # SAME variable (multi-anchor constraint plans, e.g. 'genre Stop motion'
    # and 'Miley Cyrus films' both → ?films), every constraint must hold — the
    # effective binding is the INTERSECTION, not the last declaration. Compute
    # it harness-side and expose via ctx.var_joins (injected as a system note
    # next turn) so the model never has to eyeball 40-item lists.
    # R3-a (2026-10-01): SEQ_JOIN_V2 replaces this content-clock block with
    # the declaration-closure predicate + views.join_projection data source
    # (§6.8-3); the block below stays verbatim as the OFF/rollback face.
    # `state` rides along so the V2 conclusion set canonicalizes fids through
    # norm_fact_key — the SAME resolution COMMIT's clock uses (the two
    # predicates must not disagree on which fact a declaration closes).
    if _join_v2():
        _update_system_join_v2(ctx, state)
        return
    # Wave-2 (2026-09-20): the intersection runs over TOOL-SIDE CANDIDATE POOLS
    # (ctx.fact_candidate_pool — seq_tools._sg_finalize records each walk's
    # full entity enumeration per fid), not the model-declared fact_bindings:
    # a checkpoint list is CURATED, so intersecting two curated lists conflates
    # "the model listed representatives" with "the graph holds nothing
    # shared". A fid with no recorded pool (pre-pool episode, errored walk)
    # falls back to its declared fact_bindings.
    _fbind = getattr(ctx, "fact_bindings", None) or {}
    _fvars = getattr(ctx, "fact_vars", None) or {}
    if _fbind:
        from kgqa.core.utils import normalize as _nj
        from collections import defaultdict as _dj
        _pool = getattr(ctx, "fact_candidate_pool", None) or {}
        _by_var = _dj(list)
        for _f, _v in _fvars.items():
            if _f in _fbind:
                _by_var[_v].append(_f)
        _joins = getattr(ctx, "var_joins", None)
        if _joins is None:
            _joins = {}; ctx.var_joins = _joins
        for _v, _fids in _by_var.items():
            if len(_fids) < 2:
                continue
            _fids = sorted(_fids)
            _sets = []
            _orig = {}
            for _f in _fids:
                _s, _o = set(), {}
                for p in (_pool.get(_f) or _fbind[_f]):
                    k = _nj(p)
                    _s.add(k); _o.setdefault(k, p)
                _sets.append(_s); _orig.update(_o)
            _inter = sorted(_sets[0].intersection(*_sets[1:]))
            _inter_orig = [_orig[k] for k in _inter]   # original casing for display/rescue
            _prev = _joins.get(_v)
            _cur_join = (tuple(_fids), tuple(_inter_orig))
            if _prev != _cur_join:
                ctx.join_flag = (_v, _fids, _inter_orig)
            _joins[_v] = _cur_join
            # NOT ENFORCED either way (Thundera specimen, 2026-08-19; V2.1,
            # 2026-08-22): missing evidence ≠ negative evidence — and the pool
            # join only enumerates what the SUBMITTED relations walked. The
            # intersection is AUTHORITATIVE as displayed information (every
            # walked constraint holds on exactly these entities) but ADVISORY
            # as control flow: bindings are never force-collapsed; support is
            # computed at EVIDENCE COMMIT and the answer stage decides under
            # the support policy.


def _parse_flat(content: str):
    return SP.parse_action(content)


def _parse_with_repair(content: str):
    return SP.parse_action(content)




# LEGACY DISPATCH PATH (R5, user ruling 2026-10-06): SEQ_ADAPTIVE_ROUTES=0
# routes tools through the C-segment below. This path is RETIRED — no
# supported batch configuration uses it. Known inconsistencies vs adaptive:
# (1) accumulated_triples includes pre-filter walk entities (answer pool
# scope differs); (2) the multi-anchor/join gates live only here. The
# adaptive path (SEQ_ADAPTIVE_ROUTES=1) is the sole supported dispatcher.
_LEGACY_UNSUPPORTED = True


def seq_agents_md() -> str:
    """The SEQ system prefix (loaded once per process). SEQ_PROMPT=V21 selects
    the V2.1 evidence-grounded prototype (contract/extend/commit/two-stage)."""
    import os
    # Ad-hoc A/B: load any prompt file without registering a token.
    _file = os.environ.get("SEQ_PROMPT_FILE", "").strip()
    if _file:
        from pathlib import Path as _Path
        return _Path(_file).read_text()
    if os.environ.get("SEQ_ADAPTIVE_ROUTES", "0").strip().lower() in ("1", "true", "on", "yes"):
        # Within adaptive mode only these tokens switch the file; every other
        # SEQ_PROMPT value stays ignored here (replay setdefault V21 trap).
        _ap = os.environ.get("SEQ_PROMPT", "").strip().upper()
        if _ap in ("PATTERN", "P8"):
            return (_AGENT_DIR / "SEQ_AGENTS_PATTERN.md").read_text()
        if _ap in ("FALLBACK", "P10"):
            return (_AGENT_DIR / "SEQ_AGENTS_FALLBACK.md").read_text()
        if _ap in ("AUDIT", "P11"):
            return (_AGENT_DIR / "SEQ_AGENTS_AUDIT.md").read_text()
        if _ap in ("C2S", "C2-S"):
            return (_AGENT_DIR / "SEQ_AGENTS_C2S.md").read_text()
        if _ap in ("C2R", "C2-R"):
            return (_AGENT_DIR / "SEQ_AGENTS_C2R.md").read_text()
        if _ap in ("CALA", "CAL-A"):
            return (_AGENT_DIR / "SEQ_AGENTS_CALA.md").read_text()
        if _ap in ("CALAB", "CAL-AB"):
            return (_AGENT_DIR / "SEQ_AGENTS_CALAB.md").read_text()
        if _ap in ("C1",):
            return (_AGENT_DIR / "SEQ_AGENTS_C1.md").read_text()
        if _ap in ("C2",):
            return (_AGENT_DIR / "SEQ_AGENTS_C2.md").read_text()
        if _ap in ("C3",):
            return (_AGENT_DIR / "SEQ_AGENTS_C3.md").read_text()
        if _ap in ("C123",):
            return (_AGENT_DIR / "SEQ_AGENTS_C123.md").read_text()
        if _ap in ("C124",):
            # M12 arm A: standing rules — candidate growth (§3.3),
            # role-keyed bounded reading (§1.4), accumulated-set tiering (§23)
            return (_AGENT_DIR / "SEQ_AGENTS_C124.md").read_text()
        if _ap in ("C125",):
            # M12 arm B: answer-time self-check table (§0.5 answer entry —
            # candidate table + ownership line; env never parses audit text)
            return (_AGENT_DIR / "SEQ_AGENTS_C125.md").read_text()
        if _ap in ("RC2", "RC1-SHORT"):
            return (_AGENT_DIR / "SEQ_AGENTS_RC2.md").read_text()
        if _ap in ("RC1", "RELCOV"):
            # Relation-Coverage V1 (C125 lineage, team20 delivery 2026-10-14):
            # relation coverage — not record sparsity — decides retrieval
            # completeness (§10/§12/§16 + §16.2 matrix); AUDIT §0.5 keeps the
            # explicit self-check (env never parses it).
            return (_AGENT_DIR / "SEQ_AGENTS_RC1.md").read_text()
        return (_AGENT_DIR / "SEQ_AGENTS_ADAPTIVE.md").read_text()
    import os
    _p = os.environ.get("SEQ_PROMPT", "").upper()
    if _p in ("V21", "V2.1"):
        return (_AGENT_DIR / "SEQ_AGENTS_V21.md").read_text()
    if _p in ("V22", "V2.2"):
        return (_AGENT_DIR / "SEQ_AGENTS_V22.md").read_text()
    if _p in ("V23", "V2.3"):
        return (_AGENT_DIR / "SEQ_AGENTS_V23.md").read_text()
    if _p in ("V24", "V2.4"):
        return (_AGENT_DIR / "SEQ_AGENTS_V24.md").read_text()
    if _p in ("V241", "V2.4.1", "V24-R1"):
        return (_AGENT_DIR / "SEQ_AGENTS_V241.md").read_text()
    return (_AGENT_DIR / "SEQ_AGENTS.md").read_text()


def _answer_analysis_vocab() -> Tuple[str, str]:
    """(field-list, leading field) of the two-stage ANSWER_ANALYSIS block,
    synced to the ACTIVE prompt version the way seq_agents_md selects it
    (Wave-1 vocabulary sync): V22/V23 teach BASE_BINDINGS / CONSTRAINT_CHECK /
    FINAL_BINDINGS; the V21-lineage prompt teaches the legacy BASE_CANDIDATES /
    REQUIREMENT_CHECK / COUNT_CONTRACT / PROVISIONAL_FINAL framing. Injecting
    one version's field names while the system prompt teaches the other's
    taught the model vocabulary its own prompt never defines. The literal
    ANSWER_ANALYSIS header is identical in every version and the stage gate
    matches the header only — the gate regex stays untouched."""
    _p = os.environ.get("SEQ_PROMPT", "").upper()
    if _p in ("V241", "V2.4.1", "V24-R1"):
        return ("(BASE_BINDINGS / ANSWER_CARRIERS / CROSS_VIEW_SUPPORT / "
                "DISCRIMINATOR_EVIDENCE / CONSTRAINT_CHECK / FINAL_BINDINGS)",
                "BASE_BINDINGS")
    if _p in ("V22", "V2.2", "V23", "V2.3", "V24", "V2.4"):
        return ("(BASE_BINDINGS / CONSTRAINT_CHECK / FINAL_BINDINGS)",
                "BASE_BINDINGS")
    return ("(BASE_CANDIDATES / REQUIREMENT_CHECK / COUNT_CONTRACT / "
            "PROVISIONAL_FINAL)", "BASE_CANDIDATES")


_ZH_Q_CACHE: Dict[str, dict] = {}


def _zh_question_for(case_id: str) -> str:
    """Chinese restatement for this case from the SEQ_ZH_QUESTION annotation
    file ({case_id: {q, gold, zh}}), or "" when unset/unmatched."""
    path = os.environ.get("SEQ_ZH_QUESTION", "")
    if not path or not case_id:
        return ""
    if path not in _ZH_Q_CACHE:
        try:
            with open(path) as f:
                _ZH_Q_CACHE[path] = json.load(f)
        except Exception as e:
            print(f"  ⚠ SEQ_ZH_QUESTION load failed ({path}): {e}", flush=True)
            _ZH_Q_CACHE[path] = {}
    entry = _ZH_Q_CACHE[path].get(case_id) or {}
    return str(entry.get("zh") or "")


def reset_seq_episode_state(ctx) -> None:
    """Full episode-local reset for the ONE none-verdict restart (Wave-1,
    2026-09-20). The restart block used to hand-maintain a field list and
    silently missed every store added later — attempt-2 inherited attempt-1's
    territory masks (fid_pattern/pattern_state), idempotence cache
    (_sg_served), anchor bookkeeping (consumed_anchors/anchor_seqs) and
    similarity streaks (_qsim_hist), so the "fresh" second attempt planned
    differently but RETRIEVED against attempt-1's frozen world. Single source
    of truth: every ctx field with episode lifetime resets HERE, each to the
    exact type/value its init site declares (seq_tools.py lazily inits most of
    these via `getattr(...) or default`; loop.py's CaseContext declares
    subgraph_entities as a set). `restarted` is deliberately untouched — the
    none-verdict contract reads it to refuse a second restart."""
    # dict-typed stores → {} (Wave-2 adds answer_selection — the model's
    # discrimination layer — and fact_candidate_pool — the tool-side join
    # source; both are episode-local like the ledger they annotate)
    for f in ("var_bindings", "declared_facts", "fact_bindings", "fact_vars",
              "var_joins", "closed_facts", "fact_evidence",
              "fact_evidence_seq", "_sg_served", "_qsim_hist", "_qsim_streak",
              "pattern_state", "fid_pattern", "anchor_seqs",
              "anchor_seq_memo", "_fid_alias", "fid_tree_roots",
              "answer_selection", "fact_candidate_pool"):
        if hasattr(ctx, f):
            setattr(ctx, f, {})
    # list-typed stores → [] (walk_seen_entities/walk_extra MUST be lists —
    # the 1379-s1 crash family reset them to {} and the ledger append died)
    for f in ("walk_seen_entities", "walk_extra", "all_candidates",
              "_ma_gate_unc"):
        if hasattr(ctx, f):
            setattr(ctx, f, [])
    # set-typed stores → fresh set
    if hasattr(ctx, "accumulated_triples"):
        ctx.accumulated_triples = set()
    if hasattr(ctx, "consumed_anchors"):    # seq_tools.py inits it as set()
        ctx.consumed_anchors = set()
    # tuple/payload flags → None (their consumers test truthiness);
    # _join2_side_flag is the R3-a single-side JOIN STATUS payload.
    # R4 (2026-10-01): cvt_binding_flag retired — zero producers since the
    # 2026-09-24 CVT-binding ruling (see the parse-site retirement note).
    for f in ("cvt_empty_var", "join_flag",
              "halluc_binding_flag", "frozen_binding_flag", "_join2_side_flag"):
        if hasattr(ctx, f):
            setattr(ctx, f, None)
    # R3-a V2 latches: the per-variable JOIN-STATUS note set + the
    # DECLARATION WAIT one-shot
    if hasattr(ctx, "_join2_side_noted"):
        ctx._join2_side_noted = set()
    if hasattr(ctx, "_decl_wait_noted"):
        ctx._decl_wait_noted = False
    # one-shot latches → False (getattr defaults in the loop/tools gates)
    # F07 (Codex audit 2026-09-30): _join_selfcheck_done (pre-answer join
    # self-check) and _merge_noted (reasoning-time merge note) are one-shot
    # EPISODE latches — leaving them set let attempt-2 skip both notices.
    for f in ("plan_declared", "_ma_gate_fired", "_multi_tree_prompted",
              "_answer_offpool_retried", "_answer_cvt_retried",
              "_varmismatch_retried", "_join_selfcheck_done", "_merge_noted"):
        if hasattr(ctx, f):
            setattr(ctx, f, False)
    # EVIDENCE LOG (F07): the case-scope EvidenceLog is episode-local —
    # attempt-2 inherited attempt-1's delivered-key union and call log.
    # Reset to None: the finalize-side builder lazily re-inits it.
    if hasattr(ctx, "evidence_log"):
        ctx.evidence_log = None
    # SHADOW LEDGER (phase-2, 2026-09-30): same episode locality — the
    # EvidenceRecord log and its is_new history must not leak into
    # attempt-2. (The _ev_rebuild_truncs counter needs no reset: the
    # watermark in _sg_finalize reads per-call deltas.)
    # R1/R04 (2026-10-01): the episode switch no longer DISCARDS the old
    # attempt — its three-layer ledger snapshot (records/commits/facts with
    # attempt_id/episode_id/call_id/tool_message_idx anchors) goes into the
    # append-only ctx.evidence_archive first. The archive is CASE-scope
    # (deliberately absent from the reset lists above): the rollout sidecar
    # and the replay dumper export ALL attempts from it.
    if hasattr(ctx, "evidence_records"):
        _ev_arch = getattr(ctx, "evidence_archive", None)
        if _ev_arch is None:
            _ev_arch = []
            ctx.evidence_archive = _ev_arch
        if ctx.evidence_records is not None:
            _ev_arch.append(ctx.evidence_records.archive())
        ctx.evidence_records = None
    # counters → 0
    for f in ("_cvt_loop_n", "_refusal_n"):
        if hasattr(ctx, f):
            setattr(ctx, f, 0)
    if hasattr(ctx, "plan_answer_type"):
        ctx.plan_answer_type = ""
    # Nearest-Principle anchor (Team-A 2026-09-27): a fid-less call follows
    # the most recently delivered subgraph — episode-local like the stores
    # it feeds.
    if hasattr(ctx, "_last_canon_fid"):
        ctx._last_canon_fid = ""
    # two-stage flags (COMMIT-WIDENING): the fresh attempt re-enters
    # retrieval — a stale ledger/analysis gate would dead-end it.
    for f in ("_ledger_injected", "_analysis_pending", "_analysis_done"):
        if hasattr(ctx, f):
            setattr(ctx, f, False)
    # subgraph_entities: fresh set RESEEDED from the question entities — the
    # prelink seeding is what makes turn-0 anchor centers pass the center
    # boundary; an empty set would reject fact-1 retrieval outright. Mirror
    # of the prelink loop (same idx validity check, same add).
    qe_idxs = ctx.sample.get("q_entity_id_list") or []
    ctx.subgraph_entities = {
        idx for idx in qe_idxs
        if isinstance(idx, int) and 0 <= idx < len(getattr(ctx, "ents", None)
                                                   or ())}


class SeqReactCase:
    """One case's evolving state through the SEQ per-fact loop."""

    def __init__(self, sample, pilot_row, idx):
        self.ctx = build_context(sample, pilot_row, idx)
        self.state = SeqAgentState()
        self.messages: List[Dict[str, Any]] = []
        self.failed = False
        self.failure_reason = ""
        self.done = False
        self._empty_answer_retried = False
        # anti-loop: track consecutive identical tool calls so a stuck model (re-issuing
        # the same retrieve_relations/subgraph) is nudged to converge instead of burning
        # the turn budget (root cause of 2209/1864 empty-FAILs).
        self._last_tool_sig = None
        self._tool_repeat = 0
        self._init_messages()

    def loop_nudge(self) -> str:
        if os.environ.get("SEQ_ADAPTIVE_ROUTES", "0").strip().lower() in ("1", "true", "on", "yes"):
            from kgqa.agent.seq_adaptive import hint
            return hint(self.ctx)
        tool = self._last_tool_sig[0] if self._last_tool_sig else "the tool"
        try:
            fid = json.loads(self._last_tool_sig[1]).get("sg")
        except (TypeError, ValueError, IndexError):
            fid = None
        closure = (f"If useful alternatives are exhausted, declare `- {fid} ✗ exhausted`. "
                   if fid in self.state.fact_ids else "")
        return (f"REPEAT: `{tool}` was already served in this tree/query state. "
                "Use the prior tool result: select a relevant unused relation, "
                "continue another fact from a supported center, or submit answer_values. "
                + closure + "An unchanged retry cannot add evidence.")

    def state_aware_hint(self) -> str:
        if os.environ.get("SEQ_ADAPTIVE_ROUTES", "0").strip().lower() in ("1", "true", "on", "yes"):
            from kgqa.agent.seq_adaptive import hint
            return hint(self.ctx)
        if not self._last_tool_sig:
            return ""
        if self._last_tool_sig[0] == "retrieve_relations":
            return ("Current relation candidates are available. Select relevant "
                    "relations for retrieve_subgraph, revise the sub-question, "
                    "or answer from delivered evidence.\n")
        if self._last_tool_sig[0] == "retrieve_subgraph":
            return ("Use the delivered triples and attribute values. Declare "
                    "needed bindings before a dependent retrieval, or submit "
                    "answer_values now. Missing evidence is unknown.\n")
        return ""

    def _maybe_inject_evidence_commit(self) -> None:
        """BINDING LEDGER → EVIDENCE COMMIT (V2.1; COMMIT-WIDENING 2026-08-22).

        Trigger: commit_due — the READY predicate (all declared facts CLOSED
        (retrieved or ✗ closure), OR answer var bound with no OPEN fact able to
        change it). SEQ_WIDE_COMMIT=off falls back to the legacy all_retrieved.
        R3-a SEQ_JOIN_V2 (2026-10-01, default on): the trigger is the
        DECLARATION clock — every declared fact has a ✓ binding or an explicit
        ✗; retrieval-done-but-undeclared emits the one-shot DECLARATION WAIT
        note instead of committing.
        Idempotent via ctx._ledger_injected (PLAN EXTEND / RESTART reset it).
        Informational only: it never requires a separate analysis turn.

        CANDIDATES come from state.answer_var (the plan's declared answer
        variables) — the old last-bound-var guess printed the wrong variable's
        values under early binding. The heuristic survives ONLY as a display
        fallback when no declared answer var has bindings, and then the
        SUPPORT_BY_REQUIREMENT matrix is skipped (a guessed var yields a wrong
        support matrix)."""
        if (self.state.state != "RETRIEVE"
                or getattr(self.ctx, "_ledger_injected", False)):
            return
        _cf = getattr(self.ctx, "closed_facts", None) or {}
        _fb = getattr(self.ctx, "fact_bindings", None) or {}
        _vj = getattr(self.ctx, "var_joins", None) or {}
        if not commit_due(self.state, _cf, _fb, _vj):
            # DECLARATION WAIT (R3-a, SEQ_JOIN_V2): the retrieval clock may
            # say DONE while the declaration ledger does not — COMMIT now
            # waits for conclusions, so SAY SO once instead of committing on
            # the retrieval clock (the 87.3% single-side-CANDIDATES face).
            if _join_v2() and not getattr(self.ctx, "_decl_wait_noted", False):
                if (ready_to_commit(self.state, _cf, _fb, _vj)
                        or self.state.all_retrieved):
                    self.ctx._decl_wait_noted = True
                    _dw = ("⌗ DECLARATION WAIT: retrieval is complete, but "
                           + str(unconcluded_facts(self.state, _cf, _fb))
                           + " still has no conclusion — EVIDENCE COMMIT fires "
                             "only when EVERY declared fact has one: "
                             "`- fid ✓ ?var = values` or "
                             "`- fid ✗ empty | moot | mismatch | exhausted` "
                             "(an honest ✗ is a legal conclusion). Close them "
                             "now.")
                    self.messages.append({"role": "user", "content": _dw})
                    self.ctx.trajectory.append({"role": "tool", "content": _dw})
            return
        _fb = getattr(self.ctx, "fact_bindings", None) or {}
        _fv = getattr(self.ctx, "fact_vars", None) or {}
        _joins = getattr(self.ctx, "var_joins", None) or {}
        from collections import defaultdict as _dl
        _by_var = _dl(list)
        for _f, _v in _fv.items():
            if _f in _fb:
                _by_var[_v].append(_f)
        if not _by_var:
            return    # nothing committed to show — no analysis stage to gate

        _sel = getattr(self.ctx, "answer_selection", None) or {}
        from kgqa.agent.views import render_v3 as _rv3
        _v3face = _rv3()

        # RULING ① (2026-10-01, V3 face): CANDIDATES abolishes the [:40]
        # cut and the selection priority — the environment's
        # output_candidates (the V3-01 projection over the var's fact
        # trees) displays IN FULL, segmented across rows (a >80 merged-edge
        # group keeps its 80 in the tree block; the complete candidate
        # face lives HERE); the model's selection keeps its own
        # independent row. Admission criterion = has an output variable +
        # a legal path usage record + a projection match — no ratio
        # threshold, not limited to submitted one-hop; members OUTSIDE the
        # projection are never injected.
        _CAND_SEG = 40

        def _cand_face(_v, _vals):
            if len(_vals) <= _CAND_SEG:
                return [f"CANDIDATES ({_v}): [{' | '.join(_vals)}]"]
            rows = [f"CANDIDATES ({_v}) (n={len(_vals)}): "
                    f"[{' | '.join(_vals[:_CAND_SEG])}]"]
            for _i in range(_CAND_SEG, len(_vals), _CAND_SEG):
                _seg = _vals[_i:_i + _CAND_SEG]
                rows.append(f"  {_v} [{_i + 1}-{_i + len(_seg)} "
                            f"of {len(_vals)}]: [{' | '.join(_seg)}]")
            return rows

        def _vals_for(_v, _fids):
            # LEDGER vs SELECTION (Wave-2): the model's narrowed pick
            # (answer_selection) leads the CANDIDATES display — it is the
            # current discrimination; the join / fact_bindings ladder is the
            # retrieval-record fallback for vars the model never narrowed.
            # R3-b (SEQ_RENDER_V3, §6.3): the retrieval-record basis is the
            # ENVIRONMENT's answer-variable projection (views.
            # answer_projection over the var's fact trees — the pool's
            # legal successor: the walk's full enumeration, never the
            # model's curated subset). Curated fact_bindings remains only
            # as the degraded fallback when no ledger exists / projected
            # empty. Display keeps the facts' own spellings (a projection
            # of the view's answer, not a second legality implementation).
            # RULING ①: under the V3 face the selection and join branches
            # are GONE from this ladder (selection = its own row; the join
            # intersection is a subset of the full projection).
            if not _v3face and _sel.get(_v):
                return list(_sel[_v])
            if not _v3face and _v in _joins and _joins[_v][1]:
                return list(_joins[_v][1])
            if _v3face:
                _rlog = getattr(self.ctx, "evidence_records", None)
                if _rlog is not None and _rlog.delivered_commits():
                    from kgqa.agent import views as _views
                    _proj = _views.answer_projection(_rlog, _fids, _v)
                    if _proj:
                        # display casing: the projection is normalized —
                        # recover the facts' own spellings from the ledger's
                        # public surface (a projection of the view's answer,
                        # never a second legality implementation; the legacy
                        # record's triples cover loaded sidecar ledgers)
                        _orig = {}
                        for _c in _rlog.delivered_commits():
                            for _ob in (_c.observations or ()):
                                _f = _rlog.facts.get(_ob.fact_key)
                                if _f is None:
                                    continue
                                for _nm2 in (_f.h_id, _f.t_id):
                                    _nz = _nzv(_nm2)
                                    if _nz in _proj and _nz not in _orig:
                                        _orig[_nz] = _nm2
                            _lg = getattr(_c, "legacy", None)
                            for _tr in (getattr(_lg, "triples", None) or ()):
                                for _nm2 in (_tr.h, _tr.t):
                                    _nz = _nzv(_nm2)
                                    if _nz in _proj and _nz not in _orig:
                                        _orig[_nz] = _nm2
                        return [_orig.get(_x, _x) for _x in sorted(_proj)]
            _ordered = sorted(_fids)
            return _fb[_ordered[-1]] if _ordered else []

        # one normalize import for the display-casing projection above
        from kgqa.core.utils import normalize as _nzv

        _avs = [v for v in (getattr(self.state, "answer_var", None) or []) if v]
        _live = [v for v in _avs if _by_var.get(v) and _vals_for(v, _by_var[v])]
        _guessed = False
        if not _live:
            # no declared answer var is bound — display fallback to the last
            # bound var; SUPPORT lines suppressed (wrong-matrix risk)
            for _v, _fids in _by_var.items():
                if _vals_for(_v, _fids):
                    _live = [_v]
                    _guessed = True
        self.ctx._ledger_injected = True
        _lines = []
        for _v in _live:
            _vals = _vals_for(_v, _by_var[_v])
            if _v3face:
                # RULING ①: the full environment face, segmented — never
                # the [:40] cut; the model's own discrimination rides its
                # own row so the two faces cannot masquerade as each other.
                _lines.extend(_cand_face(_v, _vals))
                _sel_v = [str(x) for x in (_sel.get(_v) or ())]
                if _sel_v:
                    _lines.append(f"MODEL SELECTION ({_v}): "
                                  f"[{' | '.join(_sel_v)}]")
            else:
                _lines.append(f"CANDIDATES ({_v}): [{' | '.join(_vals[:40])}]"
                              + (f" …({len(_vals)})" if len(_vals) > 40
                                 else ""))
        # SUPPORT_BY_REQUIREMENT from the contract covers mapping — only when the
        # answer var is the plan-declared one (not the fallback guess)
        _reqs = getattr(self.state, "requirements", None) or {}
        _cov = getattr(self.state, "fact_covers", None) or {}
        if _reqs and _cov and _live and not _guessed:
            from kgqa.core.utils import normalize as _nz2
            _answer_var = _live[0]
            _req_sets = {}
            for _f, _rid in _cov.items():
                if _f in _fv and _fv[_f] == _answer_var and _f in _fb:
                    _set = { _nz2(x) for x in _fb[_f] }
                    _req_sets.setdefault(_rid, set()).update(_set)
            for _rid, _sup in _req_sets.items():
                _rt = (_reqs.get(_rid) or {}).get("text", "")[:60]
                _lines.append(f"SUPPORT {_rid} ({_rt}): "
                              f"{len(_sup)} candidate(s) positively supported")
            if len(_req_sets) > 1:
                _full = set.intersection(*_req_sets.values())
                _lines.append(f"FULL-SUPPORT (all requirements): "
                              f"{len(_full)} candidate(s)")
        _led = ("=== EVIDENCE COMMIT ===\n  " + "\n  ".join(_lines)
                + "\n  (UNRESOLVED ≠ CONTRADICTED: a requirement whose "
                "fact closed empty gives NO positive support — it does "
                "not erase candidates supported elsewhere.)"
                "\n  COUNT CONTRACT: CASE A — any candidate fully supported → "
                "submit ALL fully-supported; CASE B — none fully supported → "
                "submit the single best-supported. Never mix full and partial."
                "\n  Next: submit your supported answer_values, or one useful retrieval. "
                "Analysis may accompany the same action; no confirmation turn is needed.")
        self.messages.append({"role": "user", "content": _led})
        self.ctx.trajectory.append({"role": "tool", "content": _led[:300]})
        self.ctx._analysis_pending = False

    def _init_messages(self):
        q = f"Question: {self.ctx.question}"
        # SEQ_ZH_QUESTION (2026-09-07): annotation JSON {case_id: {q, gold, zh}}.
        # ZH RESTATEMENT INJECTION REMOVED (user ruling 2026-09-20): the test
        # benchmark is English-only; the restatement acted as both a
        # wrong-reading amplifier (1379: career→职务, 3/3 wrong with zh) and
        # an untracked semantic annotator. Bare-English plan readings were
        # measurably stabler (2×2 probe). _zh_question_for stays for
        # diagnostic tooling only — never injected into agent context.
        self.messages = [
            {"role": "system", "content": seq_agents_md()},
            {"role": "user", "content": q},
        ]

    async def prelink(self, session, top_k: int = 6):
        """Plan-前置: 用数据自带的 qentity（清洁问题实体，正确 FB 名）作为锚点参考注入
        question + 植入 ctx.subgraph_entities。解决模型抽取错实体名的问题（'Vienna,
        Austria'→'Vienna'；GTE 噪声 'National Anthem'→'Tiến Quân Ca'）。qentity 覆盖 100%、
        泄漏 2%、idx 100% 有效。
        Plan-contract v2 (2026-08-21, user ruling B): 锚点实体按【原始问题】GTE 相关性
        排序。注入语义限定为锚点参考，不覆盖问题主语（Lala/Carmelo 标本：q_entity 与
        问题主语不符时，以问题为准）。
        Wave-1 (2026-09-20): the per-entity question-ranked RELATION sample is
        REMOVED — a pre-chewed neighbor shortlist competed with (and outranked)
        the model's own question reading at plan time; the authoritative
        relation set comes from retrieve_relations, never from the prompt."""
        ctx = self.ctx
        qe_names = ctx.sample.get("q_entity") or []
        qe_idxs = ctx.sample.get("q_entity_id_list") or []
        if not qe_names:
            return
        se = getattr(ctx, "subgraph_entities", None)
        names = []
        for name, idx in zip(qe_names, qe_idxs):
            if not isinstance(idx, int) or not (0 <= idx < len(ctx.ents)):
                continue
            names.append(name)
            if se is not None:
                se.add(idx)
        if not names:
            return
        # rank the anchors themselves against the ORIGINAL question (user
        # ruling: 锚点按问题排序)
        try:
            from kgqa.stages.stage2_entity import gte_retrieve
            from kgqa.core.utils import phase_timer
            with phase_timer("gte"):   # prelink anchor ranking — phase-attributed
                rows = await gte_retrieve(session, ctx.question, names,
                                          candidate_texts=names, top_k=len(names))
            if rows:
                rank = {r.get("candidate"): i for i, r in enumerate(rows)}
                names.sort(key=lambda n: rank.get(n, len(rank)))
        except Exception:
            pass   # keep dataset order — the names still anchor
        lines = [f"  - {name}" for name in names]
        self.messages[1]["content"] += (
            "\n\nQuestion entities (canonical graph names — use these spellings as "
            "your anchors for the entities the question names). The authoritative "
            "relation set for each anchor comes from retrieve_relations:\n"
            + "\n".join(lines))

    @property
    def is_active(self):
        return not self.done and not self.failed

    def allowed_tools_hint(self):
        if os.environ.get("SEQ_ADAPTIVE_ROUTES", "0").strip().lower() in ("1", "true", "on", "yes"):
            return " Available tools: plan, retrieve_relations, retrieve_subgraph, answer. No fact/checkpoint declaration is required."
        hint = seq_allowed_hint(self.state)
        # plan-contract v2: keep the turn-0 question analysis in force — every hint
        # that permits answering also restates the declared answer type. Wave-1
        # demotion: the type word is a CONSISTENCY CHECK, not a filter — stated
        # as a hard contract it licensed rejecting graph-backed bindings whose
        # surface type word disagreed (the question remains the semantic
        # authority).
        at = getattr(self.ctx, "plan_answer_type", "")
        if at and "answer" in hint:
            hint += (f" PLANNED ANSWER TYPE: {at} — use it as a consistency "
                     f"check when judging bindings; the original question "
                     f"remains the semantic authority (the plan's type is a "
                     f"hint, not a filter).")
        return hint


    async def _search_join_paths(self, unconsumed, session,
                                 max_paths=3, max_hops=4, n_cands=8,
                                 targets=None):
        """JOIN PATH SEARCH on the full case graph, calibrated per user
        ruling 2026-09-08 #2: (a) the path domain is RETRIEVABLE relations —
        noisy/hub edges are excluded (the Band-of-Brothers->German
        coincidence bridge cannot appear); (b) SHORTEST paths win; (c) the
        final ordering is GTE relevance against the question."""
        try:
            from collections import defaultdict, deque
            from kgqa.traversal.k_queue import _is_noisy_path_relation
            deg = defaultdict(int)
            ents, rels = self.ctx.ents, self.ctx.rels
            for h, r, t in zip(self.ctx.h_ids, self.ctx.r_ids, self.ctx.t_ids):
                hn = str(ents[h]) if 0 <= h < len(ents) else str(h)
                tn = str(ents[t]) if 0 <= t < len(ents) else str(t)
                if hn != tn:
                    deg[hn] += 1
                    deg[tn] += 1
            # HUB CAP (user calibration: the path domain is RETRIEVABLE
            # relations — coincidence bridges cross high-degree plumbing
            # nodes like Band-of-Brothers->German Language; a node with
            # degree > HUB_DEG is not a retrievable bridge)
            HUB_DEG = 120
            # endpoints (the unconsumed starts + the target anchors) are
            # exempt — a plan anchor is naturally high-degree (Darth Vader
            # 172); the cap exists for INTERMEDIATE plumbing nodes only
            _pe0 = [str(e).strip() for e in (getattr(self.ctx, "plan_entities", None) or [])]
            _keep = {str(u).strip() for u in unconsumed} | set(_pe0)
            adj = defaultdict(set)
            rel_of = defaultdict(set)
            for h, r, t in zip(self.ctx.h_ids, self.ctx.r_ids, self.ctx.t_ids):
                hn = str(ents[h]) if 0 <= h < len(ents) else str(h)
                tn = str(ents[t]) if 0 <= t < len(ents) else str(t)
                rn = (str(rels[r]) if 0 <= r < len(rels) else "?")
                if (hn == tn or _is_noisy_path_relation(rn)
                        or (deg[hn] > HUB_DEG and hn not in _keep)
                        or (deg[tn] > HUB_DEG and tn not in _keep)):
                    continue
                adj[hn].add(tn)
                adj[tn].add(hn)
                rel_of[frozenset((hn, tn))].add((hn, rn))
            if not adj:
                return ""
            # user calibration 2026-09-08 #3: the join target is the OTHER
            # PLAN ANCHORS (start-entity to start-entity closure — 75th
            # Ranger Regiment ↔ Darth Vader), NOT the candidate pool: we
            # bridge UNCLOSED retrieval graphs between different starting
            # entities, not entity-to-answer
            _pe = [str(e).strip() for e in (getattr(self.ctx, "plan_entities", None) or [])]
            _cons = {str(c).strip().lower()
                     for c in (getattr(self.ctx, "consumed_anchors", None) or set())}
            reached = {e for e in _pe if e.lower() in _cons}
            reached -= {str(u).strip() for u in unconsumed}
            if targets is not None:
                # CONNECTIVITY MODE (2026-09-09): every anchor consumed —
                # search anchor→anchor directly (user ruling: start-entity
                # to start-entity closure) instead of unconsumed→consumed
                reached = {str(t).strip() for t in targets}
            cands, seen_sig = [], set()
            for unc in unconsumed[:2]:
                ucs = str(unc).strip()
                if ucs not in adj:
                    continue
                prev = {ucs: None}
                q = deque([ucs])
                hit_layer, depth = None, 0
                while q and hit_layer is None and depth < max_hops:
                    cur = list(q)
                    q.clear()
                    depth += 1
                    nxt = []
                    for x in cur:
                        for y in adj[x]:
                            if y in prev:
                                continue
                            prev[y] = x
                            nxt.append(y)
                            q.append(y)
                    hits = [y for y in nxt if y in reached]
                    if hits:
                        hit_layer = hits
                        break
                if hit_layer is None:
                    continue
                for tgt in hit_layer:
                    path = [tgt]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    path.reverse()
                    if len(path) > max_hops + 1:
                        continue
                    sig = frozenset(path)
                    if sig in seen_sig:
                        continue
                    seen_sig.add(sig)
                    # PATTERN FORM (user ruling 2026-10-05): the join hint
                    # gives the relation PATTERN between the anchors — the
                    # model expands a subgraph conforming to it; intermediate
                    # entity names do not leak. rel_of carries (src, full
                    # rel) per pair so traversal direction is derivable.
                    segs = []
                    for a, b in zip(path, path[1:]):
                        for src, rn in sorted(rel_of.get(frozenset((a, b)), (("", "?"),)))[:2]:
                            segs.append(rn if src == a else f"<-{rn}")
                    cands.append(f"{path[0][:24]}  ⭢  " + " ⭢ ".join(segs)
                                 + f"  ⭢  {path[-1][:24]}")
                    if len(cands) >= n_cands:
                        break
            if not cands:
                return ""
            try:
                from kgqa.stages.stage2_entity import gte_retrieve
                rows = await gte_retrieve(
                    session, str(getattr(self.ctx, "question", "") or ""),
                    cands, top_k=len(cands))
                ordered = [r.get("candidate") for r in (rows or [])
                           if r.get("candidate") in cands]
                ordered += [c for c in cands if c not in ordered]
                cands = ordered
            except Exception:
                pass                    # GTE unavailable → shortest-first as-is
            return "\n".join(f"  {i+1}. {x}" for i, x in enumerate(cands[:max_paths]))
        except Exception:
            return ""

    def _unconsumed_edge_hint(self, unconsumed, max_rels=3, max_vals=6):
        """RETRIEVAL HINT for unconsumed plan entities (user audit
        2026-09-08: the join bridge may not exist as a named path, and even
        when it does it can be a semantically irrelevant coincidence — the
        ACTIONABLE fallback is the unconsumed entity's OWN one-hop
        relations with their CVT attribute values AGGREGATED PER KEY: the
        servicemembers edge's military_person roster lists every soldier
        including the gold). Structure-only, from the case arrays."""
        try:
            from collections import defaultdict
            from kgqa.traversal.path_utils import _is_noisy_path_relation as _noisy
            ents, rels = self.ctx.ents, self.ctx.rels
            unc = {str(u).strip() for u in unconsumed}
            if not unc:
                return ""
            # NOISE-FILTERED relations (user audit 2026-09-17, WebQTrn-21:
            # degree-desc ranking put topic/webpage noise FIRST). Same
            # filter class as the prelink hint pool / retrieval pool.
            def _rel_ok(ri):
                return 0 <= ri < len(rels) and not _noisy(str(rels[ri]))
            mid_attrs = defaultdict(list)      # mid -> [(short_rel, value)]
            # unc -> rel -> attr_key -> vals (per-entity: labels must not mix)
            rel_keys = {u: defaultdict(lambda: defaultdict(list)) for u in unc}
            for h, r, t in zip(self.ctx.h_ids, self.ctx.r_ids, self.ctx.t_ids):
                hn = str(ents[h]) if 0 <= h < len(ents) else str(h)
                tn = str(ents[t]) if 0 <= t < len(ents) else str(t)
                rn = (str(rels[r]).rsplit(".", 1)[-1]
                      if 0 <= r < len(rels) else "?")
                hc, tc = hn[:2] in ("m.", "g."), tn[:2] in ("m.", "g.")
                if hc and not tc:
                    mid_attrs[hn].append((rn, tn))
                elif tc and not hc:
                    mid_attrs[tn].append((rn, hn))
            for h, r, t in zip(self.ctx.h_ids, self.ctx.r_ids, self.ctx.t_ids):
                if not _rel_ok(r):
                    continue
                hn = str(ents[h]) if 0 <= h < len(ents) else str(h)
                tn = str(ents[t]) if 0 <= t < len(ents) else str(t)
                rn = (str(rels[r]).rsplit(".", 1)[-1]
                      if 0 <= r < len(rels) else "?")
                for a, b in ((hn, tn), (tn, hn)):
                    if a not in rel_keys:
                        continue
                    if b[:2] in ("m.", "g."):
                        for ak, av in mid_attrs.get(b, []):
                            if av == a:
                                continue        # self-referential noise
                            lst = rel_keys[a][rn][ak]
                            if av not in lst:
                                lst.append(av)
                    else:
                        lst = rel_keys[a][rn][""]
                        if b not in lst:
                            lst.append(b)
            lines = []
            for u in unc:
                u_rels = rel_keys[u]
                top = sorted(u_rels.items(),
                             key=lambda kv: -sum(len(v) for v in kv[1].values()))[:max_rels]
                for rn, kmap in top:
                    parts = []
                    for ak, vals in sorted(kmap.items(), key=lambda kv: -len(kv[1])):
                        if not vals:
                            continue
                        shown = " | ".join(v[:40] for v in vals[:max_vals])
                        more = (f" …(+{len(vals)-max_vals})"
                                if len(vals) > max_vals else "")
                        parts.append((ak + "=" if ak else "") + shown + more)
                    if parts:
                        lines.append(f"'{u[:32]}' --{rn}--> " + " ; ".join(parts[:2]))
            return "\n".join(f"  {i+1}. {x}" for i, x in enumerate(lines[:max_rels * 2]))
        except Exception:
            return ""

    def _emit_tool_note(self, note: str):
        """Deliver a harness note as both a user message and a trajectory
        tool step (the standard dual-write for every gate/reminder/nudge)."""
        self.messages.append({"role": "user", "content": note})
        self.ctx.trajectory.append({"role": "tool", "content": note})

    def _anchors_disconnected(self, exclude=()) -> bool:
        """CONNECTIVITY ≠ CONSUMPTION (user audit 2026-09-09): a multi-center
        retrieve_subgraph consumes EVERY center, so the consumption latch can
        no longer detect the unclosed case — both sides consumed, yet no
        walked path links them. BFS the ACCUMULATED walk graph between the
        plan's anchor names; any anchor unreachable from the first → the
        join gate still owes its one-time check. `exclude` drops type-word
        entities (never retrieved by design) from the anchor set."""
        from collections import defaultdict, deque
        from kgqa.core.utils import normalize as _nz
        _ex = {str(e).strip().lower() for e in exclude}
        pe = [str(e).strip() for e in (getattr(self.ctx, "plan_entities", None) or [])
              if str(e).strip().lower() not in _ex]
        at = getattr(self.ctx, "accumulated_triples", None) or set()
        if len(pe) < 2:
            return False
        if not at:
            return True          # nothing walked at all — every anchor stranded
        adj = defaultdict(set)
        for h, _r, t in at:
            hn, tn = _nz(str(h)), _nz(str(t))
            if hn and tn and hn != tn:
                adj[hn].add(tn)
                adj[tn].add(hn)
        start = _nz(pe[0])
        seen = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return any(_nz(e) not in seen for e in pe[1:])

    def _type_word_exempt(self, name):
        """A plan entity whose EVERY graph edge rides a noisy relation
        (type./common./webpage/topic…) is a TYPE WORD ('Prime minister',
        'Continent'), not a retrievable subject — exempting it from the
        unconsumed set stops the gate from demanding a retrieval that has
        no meaningful form (WebQTrn-21 specimen: answer already committed,
        gate still burned a round on webpage edges)."""
        try:
            from kgqa.traversal.path_utils import _is_noisy_path_relation as _noisy
            ents = self.ctx.ents
            n = str(name).strip()
            for h, r, t in zip(self.ctx.h_ids, self.ctx.r_ids, self.ctx.t_ids):
                for i in (h, t):
                    if 0 <= i < len(ents) and str(ents[i]) == n:
                        if 0 <= r < len(self.ctx.rels) \
                                and not _noisy(self.ctx.rels[r]):
                            return False    # has at least one structural edge
            return True                     # only noisy edges → type word
        except Exception:
            return False

    def _ma_gate_would_fire(self, parsed_args):
        """SYNC trigger test for the multi-anchor consumption gate (the
        latch lives here so the deferred async run cannot double-fire)."""
        if (not (parsed_args.get("entities") or [])
                or getattr(self.ctx, "_ma_gate_fired", False)):
            return False
        _pe = getattr(self.ctx, "plan_entities", None) or []
        _cons = {str(c).strip().lower()
                 for c in (getattr(self.ctx, "consumed_anchors", None) or set())}
        _tw = [e for e in _pe if self._type_word_exempt(e)]
        _unc = [e for e in _pe if str(e).strip().lower() not in _cons
                and e not in _tw]
        if len(_pe) < 2:
            return False
        if not _unc and not self._anchors_disconnected(exclude=_tw):
            # all consumed AND the accumulated walk graph already links the
            # anchors — consumption closed AND connectivity closed
            return False
        self.ctx._ma_gate_fired = True
        self.ctx._ma_gate_unc = _unc
        return True

    async def _ma_gate_check(self, tool_name, parsed_args, session):
        """MULTI-ANCHOR CONSUMPTION GATE + JOIN PATHS (2026-09-01/07/08).
        Async: the join search is GTE-ranked against the question (user
        calibration 2026-09-08 #2 — retrievable-relation domain, shortest
        first, GTE relevance ordering).
        CONNECTIVITY MODE (2026-09-09): when every anchor was consumed
        (multi-center calls consume all centers) but the accumulated walk
        graph never linked them, the gate still fires — anchor→anchor join
        paths, no retrieval hint (the entities WERE retrieved from)."""
        _unc = getattr(self.ctx, "_ma_gate_unc", None) or []
        _pe = [str(e).strip() for e in (getattr(self.ctx, "plan_entities", None) or [])]
        if not _unc:
            _jp = (await self._search_join_paths(_pe[:1], session,
                                                 targets=_pe[1:])
                   if len(_pe) >= 2 else "")
            _hint = ("CONNECTIVITY CHECK (one-time): every plan entity was "
                "retrieved from, but no walked path links their subgraphs "
                "yet. If the question needs the two sides CONNECTED (an "
                "entity on one side reachable from the other), judge these "
                "join paths and retrieve the bridge; if the sides answer "
                "independently (their candidate sets intersect), answer now.")
            if _jp:
                _hint += ("\n\nJOIN PATHS (shortest first, GTE-ranked against the "
                          "question — judge relevance):\n" + _jp)
            self.messages.append({"role": "user", "content": _hint})
            self.ctx.trajectory.append({"role": "tool", "content":
                "connectivity gate (all anchors consumed, subgraphs unlinked)"
                + ((f"\nJOIN PATHS:\n{_jp}") if _jp else "")})
            return
        _jp = await self._search_join_paths(_unc, session)
        _hint = ("UNCONSUMED ENTITIES (one-time check): declared "
            + " | ".join(f"'{e}'" for e in _unc[:3])
            + " never started any subgraph retrieval. Retrieve FROM them only "
            "if the question's pattern still has an unexecuted segment that "
            "needs their neighborhood; if their role is only a supporting "
            "condition already covered, infer and answer from the delivered "
            "records.")
        if _jp:
            _hint += ("\n\nJOIN PATHS (shortest first, GTE-ranked against the "
                      "question — judge relevance):\n" + _jp)
        _eh = self._unconsumed_edge_hint(_unc)
        if _eh:
            _hint += ("\n\nRETRIEVAL HINT (the unconsumed entity's own one-hop relations):\n" + _eh)
        self.messages.append({"role": "user", "content": _hint})
        # trajectory carries the FULL paths/hint (user audit 2026-09-08:
        # the short marker hid the bridges from credit adjudication and
        # human review; join paths hitting a subgraph's core relations
        # CREDIT that subgraph's behavior — the adjudicator reads them here)
        self.ctx.trajectory.append({"role": "tool", "content":
            "unconsumed-entities gate: " + " | ".join(_unc[:3])
            + ((f"\nJOIN PATHS:\n{_jp}") if _jp else "")
            + ((f"\nRETRIEVAL HINT:\n{_eh}") if _eh else "")})

    def rescue_terminal_answer(self):
        if os.environ.get("SEQ_ADAPTIVE_ROUTES", "0").strip().lower() in ("1", "true", "on", "yes"):
            return
        if getattr(self, "done", False) or getattr(self.ctx, "llm_answer_preds", None):
            return
        values = self._last_answer_entities()
        if not values: return
        result = ST._do_seq_answer({"entities": values}, self.ctx)
        if not SP.result_failed(result): self._rescued = True

    def _join_path_rescue(self, fids, max_hops=3):
        from kgqa.agent.tools import join_path_rescue
        return join_path_rescue(self.ctx, fids, max_hops)

    def _join_v2_message(self, var, fids, inter) -> str:
        """§6.8-3 JOIN note body (R3-a, both sides declared complete):
        non-empty → the intersection report (advisory, semantics preserved,
        data source = the output-variable projection); EMPTY → the
        environment OWES a connecting path — render the bridge as a short
        structured suggestion the model can open a retrieval along, or state
        explicitly that no ≤2-hop bridge exists (never silent)."""
        if inter:
            return (f"⌗ SYSTEM JOIN (output-variable projection): "
                    f"{len(fids)} subgraphs bind {var} "
                    f"({', '.join(fids)}) — EVERY constraint must hold. "
                    f"{var} = {' ∩ '.join(fids)} (answer-typed projections) = "
                    f"[{' | '.join(inter[:40])}]"
                    + (f" … ({len(inter)} entities)" if len(inter) > 40 else "")
                    + ". Both sides are declared complete. Advisory as always: "
                      "a too-narrow relation set on either side can shrink "
                      "this intersection — verify against both subgraphs' "
                      "triples before finalizing.")
        _msg = (f"⌗ SYSTEM JOIN (output-variable projection): "
                f"{len(fids)} subgraphs bind {var} ({', '.join(fids)}) — "
                f"EVERY constraint must hold. {var} = {' ∩ '.join(fids)} = "
                "EMPTY: the sides' output-variable projections share NO "
                "entity.")
        try:
            _path = self._bridge_path_v2(fids[:2])
            if _path:
                _msg += ("\nCONNECTING PATH — the case graph bridges the two "
                         "subgraphs' evidence domains:\n" + _path +
                         "\nYou may open a NEW retrieval along this path to "
                         "merge the sides, then re-declare the checkpoint.")
            else:
                _msg += ("\nNo connecting path within 2 hops exists between "
                         "the two subgraphs' evidence domains. Do not force a "
                         "merge — answer from the delivered evidence by your "
                         "own discrimination (an honest empty is legal).")
        except Exception:
            _msg += "\n(bridge-path search unavailable)"
        return _msg

    def _bridge_path_v2(self, fids, radius: int = 2):
        """CONNECTING-PATH search (§6.8-3, R3-a) — GTE UNIFICATION (user
        ruling 2026-10-02): the bridge suggestion now rides join_path_
        rescue's ONE ordering search (POOL-TO-POOL shortest + GTE-on-
        question ranking, the fairness contract — never gold-anchored, the
        landing point hidden by the rescue's answer-leak guard) instead of
        this method's own deterministic first-hit BFS. No second ranking
        implementation exists anymore; this is a thin adapter: the
        domain-overlap pre-check keeps its explanatory message (the sides'
        existence∪frontier domains share nodes that are all outside the
        answer-typed projection), everything else delegates with the ≤
        `radius` hop bound. Returns the rendered suggestion ('' = no path
        within radius / a side delivered nothing)."""
        if len(fids) < 2:
            return ""
        from kgqa.agent import views as _views
        from kgqa.core.utils import normalize as _nrm
        ctx = self.ctx
        ents = list(getattr(ctx, "ents", None) or ())
        log = getattr(ctx, "evidence_records", None)
        n2i = {}
        for _i, _e in enumerate(ents):
            n2i.setdefault(_nrm(str(_e)), _i)

        def _domain(fid):
            idxs = set()
            if log is not None:
                for nm in _views.existence(log, "tree", fid):
                    _j = n2i.get(nm)
                    if _j is not None:
                        idxs.add(_j)
                idxs |= {int(_i) for _i in
                         (_views.frontier(log, "tree", fid)
                          .get("node_ids") or ())}
            return idxs

        try:
            sa, sb = _domain(fids[0]), _domain(fids[1])
        except Exception:
            sa, sb = set(), set()
        if log is not None and sa and sb and (sa & sb):
            # same-evidence overlap with an EMPTY answer projection (the
            # shared nodes are all transit/non-answer-typed) — say so instead
            # of pretending a bridge search ran
            _names = " | ".join(str(ents[_i]) for _i in sorted(sa & sb)[:5])
            return (f"(the two subgraphs' evidence domains already share: "
                    f"{_names} — but nothing shared is in the answer-typed "
                    "projection)")
        # GTE path ordering (user ruling 2026-10-02): one shared search —
        # POOL-TO-POOL shortest with GTE-on-question ranking inside equal
        # hops; the rescue's render carries the answer-leak guard.
        try:
            from kgqa.agent.tools import join_path_rescue
            return join_path_rescue(ctx, fids, radius) or ""
        except Exception:
            return ""

    def _name_idx(self, name):
        nm = str(name)
        for i, e in enumerate(self.ctx.ents):
            if str(e) == nm:
                return i
        return None


    def _maybe_merge_note(self):
        """REASONING-TIME merge note (user ruling 2026-09-23): when the
        model starts its final reasoning (ANSWER_ANALYSIS detected) and the
        walked pools never merged while a bridge exists, deliver the
        EXISTENCE note ONCE as an informational message — no interruption,
        no redo demand, no per-subgraph repetition.
        R3-b (SEQ_RENDER_V3): the merge test reads the VIEW layer —
        join_projection over the two earliest delivered trees (the §6.3
        answer_pool successor of the tool-side pool); the bridge search
        domain comes from existence∪frontier inside join_path_rescue. The
        legacy fact_candidate_pool reads stay verbatim as the OFF face."""
        try:
            ctx = self.ctx
            if getattr(ctx, "_merge_noted", False):
                return
            from kgqa.agent.tools import join_path_rescue
            from kgqa.agent.views import render_v3 as _rv3
            if _rv3():
                _rlog = getattr(ctx, "evidence_records", None)
                if _rlog is None:
                    return
                _trees = []
                for _c in _rlog.delivered_commits():
                    if _c.canon_fid and _c.canon_fid not in _trees:
                        _trees.append(_c.canon_fid)
                if len(_trees) < 2:
                    return
                from kgqa.agent import views as _views
                _a, _b = _trees[:2]
                if _views.join_projection(_rlog, [_a, _b])["intersection"]:
                    return      # the sides' projections share an entity
                if not join_path_rescue(ctx, [_a, _b]):
                    return
                ctx._merge_noted = True
                msg = ("NOTE (informational): your subgraphs' walked pools have "
                       "not merged; an unretrieved connecting path EXISTS "
                       "between their centers.")
                self.messages.append({"role": "user", "content": msg})
                self.ctx.trajectory.append({"role": "tool", "content": msg})
                return
            fcp = getattr(ctx, "fact_candidate_pool", None) or {}
            if len(fcp) < 2:
                return
            ks = list(fcp.keys())
            if set(fcp.get(ks[0]) or ()) & set(fcp.get(ks[1]) or ()):
                return
            if not join_path_rescue(ctx, ks[:2]):
                return
            ctx._merge_noted = True
            msg = ("NOTE (informational): your subgraphs' walked pools have "
                   "not merged; an unretrieved connecting path EXISTS "
                   "between their centers.")
            self.messages.append({"role": "user", "content": msg})
            self.ctx.trajectory.append({"role": "tool", "content": msg})
        except Exception:
            pass

    def _last_answer_entities(self) -> list:
        # Recover only an explicit answer action, never analysis/JOIN candidates.
        for step in reversed(self.ctx.trajectory):
            if step.get("role") != "assistant": continue
            try:
                tool, args = SP.parse_action(step.get("content") or "")
            except SP.ProtocolError:
                return []
            if tool == "answer":
                value, error = validate_args(tool, args)
                return [] if error else value["entities"]
        return []

    async def process_turn(self, session, raw_response, reasoning,
                           history_text=None) -> str:
        """Process ONE turn's LLM output: append → bind vars → parse → validate →
        dispatch → record. Shared by the per-call runner (run_seq_react_case) and
        the batch runner (run_seq_react_batch) so both follow identical semantics.

        Returns "done" when the case is complete (answer accepted / failed), else
        "continue". The LLM call itself is the caller's responsibility — this method
        only consumes (raw_response, reasoning).

        THREE-PHASE COMPOSITION (perf-4, 2026-08-23): the turn is A
        (_parse_prepare: pure CPU — parse/bind/gate/validate, produces the tool
        request) → B (_execute_tool: IO — GTE / walk lanes) → C (_finalize: pure
        CPU — result landing, evidence merge, COMMIT injection, message
        assembly). process_turn keeps the sequential composition so the
        per-case runner, run_seq_react_batch and every test see IDENTICAL
        semantics; the round-level scheduler (seq_rollout / replay_dispatch)
        drives the phases directly so one round runs A for all cases, fires
        every GTE/walk request together, then runs C — the event loop does
        orchestration only (sync tax no longer interleaves as 500-coroutine
        fragments)."""
        prep = self._parse_prepare(raw_response, reasoning, history_text)
        if "_gate_deferred" in prep:
            await self._ma_gate_check(*prep["_gate_deferred"], session)
            return "continue"
        if "ret" in prep:
            return prep["ret"]
        result_str = await self._execute_tool(prep, session)
        return self._finalize(prep, result_str)

    def _parse_prepare(self, raw_response, reasoning, history_text=None) -> dict:
        """Reject malformed actions before evaluating accompanying declarations.

        Declarations are staged until action admission. A later executor error
        leaves accepted declarations intact and acknowledges them separately.

        ``history_text`` is the full assistant turn as it must appear in the
        context (thinking + action). Tool parsing still runs on ``raw_response``
        (the action part), so retaining reasoning never changes dispatch.

        Pure-vLLM callers pass content/reasoning separately and no
        ``history_text``; ``SEQ_HISTORY_THINKING=1`` rebuilds the training-side
        shape there (see ``_history_with_thinking``).
        """
        if history_text is None:
            history_text = _history_with_thinking(raw_response, reasoning)
        if os.environ.get("SEQ_ADAPTIVE_ROUTES", "0").strip().lower() in ("1", "true", "on", "yes"):
            from kgqa.agent.seq_adaptive import parse_prepare
            return parse_prepare(self, raw_response, reasoning, history_text)
        clean = strip_reasoning_leak(raw_response or "")
        history = (strip_reasoning_leak(history_text, keep_think_tags=True)
                   if history_text is not None else clean)
        def reject(message):
            item = {"role": "assistant", "content": history}
            if reasoning: item["reasoning"] = reasoning
            self.messages.append(item)
            self.ctx.trajectory.append(dict(item))
            self.messages.append({"role": "user", "content": "REJECTED: " + message})
            self.ctx.trajectory.append({"role": "tool", "content": "REJECTED: " + message})
            return {"ret": "continue"}
        try:
            tool, parsed = SP.parse_action(clean)
        except SP.ProtocolError as exc:
            return reject("ACTION_PARSE: " + str(exc))
        if tool:
            _, error = validate_args(tool, parsed)
            if error: return reject(error)
        # Explicit controls are standalone actions, not embedded prose.
        control = clean.lstrip().startswith("[PLAN EXTEND]") or bool(
            re.match(r"\s*\[explore(?:\s|:)", clean))
        if control and tool:
            return reject("MULTIPLE_ACTIONS: a restart/plan extension and a tool must be separate responses.")
        if control:
            return self._parse_prepare_impl(clean, reasoning, history)
        declarations = SP.declaration_text(clean)
        live_ctx, live_state, live_messages = self.ctx, self.state, self.messages
        prior_plan_context = None
        if tool in ("plan", "decompose"):
            prior_plan_context = {
                "values": {key: copy.deepcopy(getattr(live_ctx, key)) for key in SP.DECL_FIELDS if hasattr(live_ctx, key)},
                "absent": [key for key in SP.DECL_FIELDS if not hasattr(live_ctx, key)]}
        staged = SP.stage_context(live_ctx, bool(declarations))
        staged._previous_plan_declared = getattr(live_ctx, "plan_declared", False)
        staged.trajectory = list(live_ctx.trajectory)
        self.messages = list(live_messages)
        message_start, trajectory_start = len(live_messages), len(live_ctx.trajectory)
        self.ctx, self.state = staged, copy.deepcopy(live_state)
        old_replan = getattr(self, "_replan_used", False)
        old_latch = getattr(self, "_post_restart_plan_ok", False)
        problem = None
        try:
            if declarations:
                SP.validate_declaration_lines(declarations)
                for line in declarations.splitlines():
                    m = re.search(r"(?:sg\d+(?:\.f\d+)?|f\d+)", line)
                    if m and norm_fact_key(self.state, m[0]) not in self.state.fact_ids:
                        raise SP.ProtocolError("DECLARATION_SCOPE: unknown fact " + m[0])
            prep = self._parse_prepare_impl(clean, reasoning, history)
            staged_state = self.state
        except SP.ProtocolError as exc:
            problem = str(exc)
            prep, staged_state = {"ret": "continue"}, self.state
        except Exception:
            self.state = live_state
            self._replan_used, self._post_restart_plan_ok = old_replan, old_latch
            raise
        finally:
            new_messages = self.messages[message_start:]
            new_steps = staged.trajectory[trajectory_start:]
            self.ctx, self.messages = live_ctx, live_messages
        accepted = problem is None and ("tool" in prep or (not tool and bool(declarations)))
        if accepted:
            SP.commit_context(live_ctx, staged)
            self.state = staged_state
            live_messages.extend(new_messages)
            live_ctx.trajectory.extend(new_steps)
            if "tool" in prep:
                prep["_previous_state"] = copy.deepcopy(live_state)
                prep["_previous_plan_context"] = prior_plan_context
                prep["_previous_plan_declared"] = getattr(live_ctx, "plan_declared", False) if tool not in ("plan", "decompose") else getattr(staged, "_previous_plan_declared", False)
                prep["_previous_replan_used"] = old_replan
                prep["_previous_restart_latch"] = old_latch
                prep["_declarations_accepted"] = bool(declarations)
        else:
            live_state.reminded = getattr(staged_state, "reminded", live_state.reminded)
            self.state = live_state
            self._replan_used, self._post_restart_plan_ok = old_replan, old_latch
            if problem: return reject(problem)
            # Preserve the actual attempted action and its rejection, not
            # JOIN/COMMIT previews computed from rolled-back declarations.
            live_messages.extend(m for m in new_messages if m.get("role") == "assistant")
            live_ctx.trajectory.extend(m for m in new_steps if m.get("role") == "assistant")
            feedback = next((m for m in reversed(new_messages) if m.get("role") == "user"), None)
            if feedback:
                live_messages.append(feedback)
                final_step = next((m for m in reversed(new_steps) if m.get("role") == "tool"), None)
                live_ctx.trajectory.append(final_step or {"role": "tool", "content": feedback["content"]})
            if prep.get("_gate_deferred"):
                for key in ("_ma_gate_fired", "_ma_gate_unc"):
                    if hasattr(staged, key): setattr(live_ctx, key, getattr(staged, key))
        return prep

    def _parse_prepare_impl(self, raw_response, reasoning, history_text=None) -> dict:
        """A段 (pure CPU): everything from ingress sanitization through the
        validated tool request. Early-exit paths complete their message/ctx
        side effects here and return {"ret": "continue"}; the tool path
        returns {"tool": name, "args": args, "raw": post-quoted content}."""
        # empty response — nudge, keep going
        if not raw_response or not raw_response.strip():
            self.messages.append({"role": "assistant", "content": history_text or ""})
            self.messages.append({"role": "user", "content": self.allowed_tools_hint()})
            return {"ret": "continue"}

        # ingress sanitizer (harness fix P5): strip the vLLM reasoning_end leak BEFORE
        # anything consumes the content — history, checkpoint bindings, tool parsing.
        # The phrase glues onto the last partial word mid-entity-list when the thinking
        # budget force-inject fires; downstream parsers must never see it.
        raw_response = strip_reasoning_leak(raw_response)

        asst_msg = {"role": "assistant", "content": history_text if history_text is not None else raw_response}
        if os.environ.get("KGQA_HISTDBG"):
            print(f"[histdbg] in_tail={history_text[-28:]!r} "
                  f"stored_tail={asst_msg['content'][-28:]!r}",
                  file=sys.stderr, flush=True)
        if reasoning:
            asst_msg["reasoning"] = reasoning
        self.messages.append(asst_msg)
        traj_step = {"role": "assistant", "content": history_text if history_text is not None else raw_response}
        if reasoning:
            traj_step["reasoning"] = reasoning
        self.ctx.trajectory.append(traj_step)
        # NONE VERDICT → system restart (user ruling, 2026-08-22): the model
        # does NOT ask for a restart — it declares the exploration found
        # NOTHING answer-relevant (`[explore ✗ none]`), PARTIAL relevance
        # means continue (ladder levels 1-3). On the FIRST none the system
        # restarts the case ONCE: fresh conversation seeded with what the
        # first attempt tried. The second attempt MUST answer (a second none
        # is rejected — no variance loops); turn budget is shared.
        _rst = re.search(r"^\s*\[explore\s*(?:✗|x)\s*none([^\]]*)\]|^\s*\[explore:\s*none([^\]]*)\]",
                         raw_response, re.I)
        if _rst:
            diag = (_rst.group(1) or _rst.group(2) or "").strip(" —-")[:300]
            if getattr(self.ctx, "restarted", False):
                self.messages.append({"role": "user", "content":
                    "REJECTED: the restart was already used — this attempt MUST "
                    "answer. Submit the most justified entities from whatever "
                    "evidence exists (or an explicit empty answer). Declare "
                    "checkpoints (✓/✗) per fact and call answer."})
                self.ctx.trajectory.append({"role": "tool", "content":
                    "REJECTED none: must answer on the second attempt"})
                return {"ret": "continue"}
            self.ctx.restarted = True
            # what the first attempt tried — seed the lesson so attempt-2 plans
            # differently instead of resampling the same dead ends
            _tried = []
            for s in self.ctx.trajectory:
                c = s.get("content") or ""
                if s.get("role") == "assistant" and c.startswith("tool: retrieve_subgraph"):
                    m = re.search(r"relations:\s*(.+?)(?:\s+sg:|\s*$)", c, re.S)
                    if m:
                        _tried.extend(x.strip() for x in m.group(1).split("|") if x.strip())
            _tried = sorted(set(_tried))[:12]
            # reset conversation → [system, question+anchors]
            self.messages = self.messages[:2]
            self.state = SeqAgentState()
            # full episode-local reset (Wave-1): every ctx store with episode
            # lifetime resets in ONE place — the old hand list silently missed
            # later-added stores (territory masks, idempotence cache, anchor
            # bookkeeping, similarity streaks), so attempt-2 retrieved against
            # attempt-1's frozen world. subgraph_entities is RESEEDED from the
            # question entities there, mirroring prelink's seeding.
            reset_seq_episode_state(self.ctx)
            self._empty_answer_retried = False
            self._tool_repeat = 0
            self._last_tool_sig = None
            # F07: the answer-bounce counter is CASE-object state (not ctx)
            # and the restart REUSES this case object — without this reset
            # attempt-2 could hit the 3rd-bounce fill-in template early.
            self._answer_ready_n = 0
            # POST-RESTART PLAN LATCH (Wave-1, 1812 specimen): the restart
            # lesson demands a DIFFERENT plan, but the phase gate rejects any
            # `plan` call once the machine has left INIT — a first attempt-2
            # plan that parses yet proves unusable dead-ends the re-plan. The
            # latch makes the FIRST tool call of the fresh attempt legal as
            # `plan` regardless of residual phase; cleared after the first
            # ACCEPTED plan (see the seq_validate site in _parse_prepare).
            self._post_restart_plan_ok = True
            self._replan_used = False
            _lesson = (f"RESTART (the only one): your first attempt's verdict was "
                       f"NONE — nothing retrieved related to the answer"
                       + (f" ({diag})" if diag else "") + ".")
            if _tried:
                _lesson += (" Relations already tried (all dead ends): "
                            + " | ".join(_tried) + ".")
            _lesson += (" Plan a DIFFERENT decomposition. This second attempt "
                        "MUST produce an answer — none cannot be declared again; "
                        "if the evidence is again irrelevant, answer empty.")
            self.messages.append({"role": "user", "content": _lesson})
            self.ctx.trajectory.append({"role": "tool", "content":
                                        f"⌗ RESTART on none-verdict: {_lesson[:200]}"})
            return {"ret": "continue"}
        # PLAN EXTEND (V2.1, append-only): `[PLAN EXTEND]` + `covers: R2` +
        # sgN.anchor/sgN.fM lines APPEND facts covering a contract requirement.
        # TWO legal reasons (Wave-1, 2026-09-20): (1) COVER_MISSING — the
        # requirement exists in the original contract and no current fact
        # covers it; (2) ADD_ANCHOR_VIEW — the requirement IS covered, but
        # every new subgraph is anchored on a plan-declared entity that
        # anchors NO covering fact for it: an independent evidence view on a
        # DIFFERENT question entity (multi-entity questions whose second
        # entity constrains the same requirement — previously rejected as
        # "already covered", forcing the model into relation repair on the
        # wrong side). The anchor must be a question entity or a bound
        # variable. Successful earlier facts are never rewritten — extension
        # is the ONLY topology change after freeze.
        _ext = re.search(r"^\s*\[PLAN EXTEND\]\s*\n(.*)", raw_response, re.S)
        if _ext:
            ext_body = _ext.group(1)
            cov_m = re.search(r"covers:\s*(\S+)", ext_body)
            rid = cov_m.group(1).strip().upper() if cov_m else ""
            _reqs = getattr(self.state, "requirements", None) or {}
            _cov = getattr(self.state, "fact_covers", None) or {}
            _ents = set(self.state.entities or []) | set(
                v for v in (getattr(self.ctx, "var_bindings", None) or {}))
            _planned = {str(e).strip() for e in (self.state.entities or [])}
            new_sgs = re.findall(
                r"(sg\d+)\.anchor:\s*(.+?)\n\1\.(f\d+):\s*(.+?)\n", ext_body)
            anchored_ok = all(a.strip() in _ents or a.strip().startswith("?")
                              for _, a, _, _ in new_sgs)
            # entities already anchoring a covering fact for THIS rid: the
            # declared anchor of every subgraph holding one (chains[i]
            # parallels sg_facts insertion order at parse time), plus any
            # covering fact's own entity head token
            _cov_fids = {f for f, r in (_cov or {}).items() if r == rid}
            _chains = list(getattr(self.state, "chains", None) or [])
            _edges = getattr(self.state, "fact_edges", None) or {}
            _serving = {str((_edges.get(f) or ("", ""))[0]).strip()
                        for f in _cov_fids}
            for _i, (_sg, _fids) in enumerate(
                    (getattr(self.state, "sg_facts", None) or {}).items()):
                if any(f in _cov_fids for f in _fids) and _i < len(_chains):
                    _serving.add(str(_chains[_i].get("anchor") or "").strip())
            _serving.discard("")
            add_view_ok = bool(new_sgs) and all(
                a.strip() in _planned and a.strip() not in _serving
                for _, a, _, _ in new_sgs)
            # RID-LESS PLANS (user ruling 2026-09-20): prompt V2.3 plans
            # carry no R1.text requirement lines — they overlap the question
            # segment and were dropped. EXTEND does not essentially depend
            # on them: when the plan declares no requirements, an extend
            # anchored on a plan entity is legal on its own merits (nothing
            # is declared covered, so COVER_MISSING holds trivially)
            _rid_ok = (rid in _reqs) if _reqs else bool(rid)
            legal = (rid and _rid_ok and new_sgs and anchored_ok
                     and (rid not in (_cov.values() if _cov else ())
                          or add_view_ok))
            if legal:
                _new_tails = []
                for sg_id, anchor, fnum, fact_line in new_sgs:
                    parts = [p.strip() for p in fact_line.split("|")]
                    if len(parts) < 3:
                        continue
                    n_existing = sum(1 for f in self.state.fact_ids)
                    fid = f"f{n_existing + 1}"
                    self.state.fact_ids.append(fid)
                    self.state.fact_texts[fid] = f"({parts[0]} | {parts[1]} | {parts[2]})"
                    self.state.fact_covers[fid] = rid
                    self.state.sg_plan[sg_id] = self.state.sg_plan.get(sg_id, 0) + 1
                    # COMMIT-WIDENING: register the appended fact in the key map /
                    # edge store so READY keeps judging the extended plan
                    self.state.fact_key_map[fid] = fid
                    self.state.fact_edges[fid] = (parts[0], parts[2])
                    self.state.sg_facts.setdefault(sg_id, []).append(fid)
                    self.state.fact_key_map[
                        f"{sg_id}.f{len(self.state.sg_facts[sg_id])}"] = fid
                    if parts[2].startswith("?"):
                        _new_tails.append(parts[2])
                if ("answer" in str((_reqs.get(rid) or {}).get("kind", "")).lower()
                        and _new_tails):
                    self.state.answer_var = list(self.state.answer_var) + [
                        v for v in _new_tails if v not in self.state.answer_var]
                # extension returns the case to retrieval mode: the committed
                # ledger is stale — re-inject once the extended facts close
                self.ctx._ledger_injected = False
                self.ctx._analysis_pending = False
                self.ctx._analysis_done = False
                # R3-a: the extended facts re-open the declaration ledger —
                # the DECLARATION WAIT note may legitimately fire again
                self.ctx._decl_wait_noted = False
                self.state.retrieved_fids = [f for f in self.state.retrieved_fids]
                self.ctx.fact_ids = list(self.state.fact_ids)
                self.ctx.fact_texts = dict(self.state.fact_texts)
                self.ctx.fact_key_map = dict(getattr(self.state, "fact_key_map", None) or {})
                self.ctx.fact_edges = dict(getattr(self.state, "fact_edges", None) or {})
                _ext_msg = (f"⌗ PLAN EXTENDED: facts appended covering {rid}. "
                            "Continue retrieve_relations for the new facts; "
                            "existing facts are unchanged.")
                self.messages.append({"role": "user", "content": _ext_msg})
                self.ctx.trajectory.append({"role": "tool", "content": _ext_msg})
            else:
                _covd = [f"{f}→{rv}" for f, rv in (_cov or {}).items()]
                why = (f"requirement {rid} not in the original contract"
                       if rid and rid not in _reqs else
                       (f"requirement {rid} is already covered (facts: {_covd}) "
                        f"and the new subgraph is not an independent view — its "
                        f"anchor must be a plan-declared entity that anchors NO "
                        f"covering fact for {rid} (anchors already serving it: "
                        f"{sorted(_serving) or 'none'})")
                       if rid in (_cov.values() if _cov else ()) else
                       "no valid sgN.anchor/sgN.fM block, or anchor is not a "
                       "question entity / bound variable")
                self.messages.append({"role": "user", "content":
                    f"REJECTED PLAN EXTEND: {why}. Extension is legal for one "
                    "of two reasons: (1) COVER_MISSING — the requirement is in "
                    "the contract but no current fact covers it; (2) "
                    "ADD_ANCHOR_VIEW — it is covered, but the new subgraph is "
                    "anchored on a DIFFERENT plan-declared entity, an "
                    "independent evidence view on another question entity. "
                    "Otherwise repair RELATIONS on the covering fact instead."})
                self.ctx.trajectory.append({"role": "tool", "content":
                                            "REJECTED PLAN EXTEND: " + why})

        # merge any checkpoint variable-bindings the model just declared, so the
        # dispatch below (and later turns) can expand `?var` in tool `entities`.
        # state rides along for the V2 JOIN's fid canonicalization (R3-a).
        _update_var_bindings(self.ctx, SP.declaration_text(raw_response), self.state)
        # MULTI-TREE PROMPT (user audit 2026-09-17, 567_df97 specimen): the
        # first fact resolved is the moment to check the OTHER declared
        # entities — if one anchors no tree yet, prompt NOW, not at the
        # answer gate (by then a 49-member single tree has absorbed several
        # wasted filter layers; the join side never got its own cheap walk).
        if not getattr(self.ctx, "_multi_tree_prompted", False):
            _fb = getattr(self.ctx, "fact_bindings", None) or {}
            _pe = [str(e) for e in (getattr(self.ctx, "plan_entities", None) or [])]
            _cons = {str(c).strip().lower() for c in
                     (getattr(self.ctx, "consumed_anchors", None) or set())}
            _unanchored = [e for e in _pe
                           if e.strip().lower() not in _cons
                           and not self._type_word_exempt(e)]
            if _fb and len(_pe) >= 2 and _unanchored:
                self.ctx._multi_tree_prompted = True
                _mt = ("MULTI-TREE REMINDER: declared entity "
                       + " | ".join(f"'{e}'" for e in _unanchored[:2])
                       + " anchors no tree yet. Anchor it as its OWN subgraph"
                       " (retrieve_relations FROM it), then let SYSTEM JOIN"
                       " intersect the two sides — anchored retrieval"
                       " discriminates better than filter layers on one"
                       " tree.")
                self._emit_tool_note(_mt)
        # COMMIT-WIDENING: normalize this turn's closures (✗) and accepted
        # declarations (✓) into the state's done-sets — READY and completion
        # counting live in the harness, which sees state only. Pure `sgN`
        # declarations resolve to the group's first unfinished fact.
        self.state.closed_fids = {norm_fact_key(self.state, k)
                                  for k in (getattr(self.ctx, "closed_facts", None) or {})}
        self.state.declared_fids = {norm_fact_key(self.state, k)
                                    for k in (getattr(self.ctx, "fact_vars", None) or {})}
        # R4 (2026-10-01): the cvt_binding_flag reminder block was RETIRED here.
        # Its producer died with the 2026-09-24 ruling (086ece5: CVT bindings are
        # LEGAL mid-chain references — the strip+warn was removed), leaving a
        # read guarded by a flag nothing ever sets; the dead branch + its reset
        # entry were deleted together (specs/r4_cleanup_2026-10-01.md).
        # harness reminder: the checkpoint re-declared a FROZEN fact without
        # new subgraph evidence — tell the model it is a hallucination AND
        # hand it the complete standing bindings + the exact next submission
        # (user ruling 2026-09-21: feedback never names only the error; it
        # names what to submit — 576-s1 specimen burned turns re-deriving
        # what "the original bindings stand" referred to).
        fb = getattr(self.ctx, "frozen_binding_flag", None)
        if fb:
            self.ctx.frozen_binding_flag = None
            _fid = fb[0]
            _stand = ((getattr(self.ctx, "fact_bindings", None) or {})
                      .get(_fid)) or []
            _var = ((getattr(self.ctx, "fact_vars", None) or {})
                    .get(_fid)) or "the fact's variable"
            _stand_str = (" | ".join(_stand[:12])
                          + (" …" if len(_stand) > 12 else "")) if _stand \
                else "(none recorded — the fact has no standing ✓ binding)"
            _fz_msg = (f"⚠ Your checkpoint [{_fid} ✓] tried to change bindings "
                       f"({fb[1]}) WITHOUT any new subgraph retrieval for {_fid} — "
                       "the binding update was rejected by the revision rule. "
                       "The COMPLETE standing bindings are:\n"
                       f"- {_fid} ✓ {_var} = {_stand_str}\n"
                       "SUBMIT answer_values from these, with "
                       "the subset the evidence supports. No separate analysis turn is needed. "
                       "Changing the retrieval binding requires one fresh "
                       f"retrieve_subgraph for {_fid} — nothing else unlocks it.")
            self.messages.append({"role": "user", "content": _fz_msg})
            self.ctx.trajectory.append({"role": "tool", "content": _fz_msg})
        # harness reminder: the checkpoint bound entities ABSENT from every
        # subgraph delivered this episode (invented names; the observed view
        # of the EvidenceRecord ledger found no triple endpoint by name).
        # Same submission-first rule: show what stands and where legal values
        # live, not just the violation (receipt text in _halluc_strip_receipt).
        hb = getattr(self.ctx, "halluc_binding_flag", None)
        if hb:
            self.ctx.halluc_binding_flag = None
            _hal_msg = _halluc_strip_receipt(self.ctx, hb[0], hb[1],
                                             hb[2], hb[3])
            self.messages.append({"role": "user", "content": _hal_msg})
            self.ctx.trajectory.append({"role": "tool", "content": _hal_msg})
        # SYSTEM JOIN note: multiple subgraphs now bind the same variable —
        # V2 (R3-a, §6.8-3) reports the OUTPUT-VARIABLE PROJECTION
        # intersection (both sides declared complete); V1 intersected the
        # TOOL-SIDE candidate pools on the checkpoint-content clock. Tell the
        # model the effective binding so it answers with the intersection,
        # never one side's full list.
        jf = getattr(self.ctx, "join_flag", None)
        if jf:
            self.ctx.join_flag = None
            _var, _fids, _inter = jf
            if _join_v2():
                _j_msg = self._join_v2_message(_var, _fids, _inter)
            elif _inter:
                _j_msg = (f"⌗ SYSTEM JOIN over tool-side candidate pools: "
                          f"{len(_fids)} subgraphs bind {_var} "
                          f"({', '.join(_fids)}) — EVERY constraint must hold. "
                          f"{_var} = {' ∩ '.join(_fids)} (walk candidate pools) = "
                          f"[{' | '.join(_inter[:40])}]"
                          + (f" … ({len(_inter)} entities)" if len(_inter) > 40 else "")
                          + ". Every walked constraint holds on these entities — "
                            "but a too-narrow relation set on either side can "
                            "shrink this intersection. Verify against both "
                            "subgraphs' triples before finalizing (advisory).")
            else:
                _j_msg = (f"⌗ SYSTEM JOIN over tool-side candidate pools: "
                          f"{len(_fids)} subgraphs bind {_var} "
                          f"({', '.join(_fids)}) — EVERY constraint must hold. "
                          f"{_var} = {' ∩ '.join(_fids)} = EMPTY: the two walks' "
                          "candidate pools share NO entity. CAUTION before "
                          "answering empty: a pool only enumerates what the "
                          "SUBMITTED relations walked — a missing or too-narrow "
                          "relation set on either side can hide the shared "
                          "entity. Re-examine both subgraphs' triples (or "
                          "re-retrieve the weaker side with a broader relation "
                          "set) and answer by your own discrimination; do not "
                          "dump one side's list.")
                # JOIN-PATH RESCUE (user ruling 2026-09-23, mechanism return —
                # the 2026-09-20 shallow-join rescue was lost in edc8004's
                # revert): when the pools fail to intersect, the harness
                # searches the SHORTEST BRIDGE between the two anchors'
                # completion sets and renders it — two independent pathways
                # that never met get their connecting evidence from the
                # environment instead of an empty-intersection dead end.
                try:
                    _rescue = self._join_path_rescue(_fids)
                    if _rescue:
                        _j_msg += "\nJOIN-PATH RESCUE — shortest bridge " \
                                  "between the two subgraphs' endpoints:\n" \
                                  + _rescue
                except Exception:
                    pass
            self.messages.append({"role": "user", "content": _j_msg})
            self.ctx.trajectory.append({"role": "tool", "content": _j_msg})
        # V2 JOIN STATUS note (R3-a): one side of a join variable is declared
        # complete while the other is not — REPORTED once per variable. The
        # old predicate skipped the whole group silently at this exact state.
        if _join_v2():
            _sn = getattr(self.ctx, "_join2_side_flag", None)
            if _sn:
                self.ctx._join2_side_flag = None
                _sv, _sd, _sw = _sn
                _sn_msg = (f"⌗ JOIN STATUS ({_sv}): {' and '.join(_sd)} "
                           f"declared complete; still waiting for "
                           f"{' and '.join(_sw)}. When its retrieval is done, "
                           "declare its conclusion (`- fid ✓ ?var = values`, or "
                           "`- fid ✗ empty | moot | mismatch | exhausted`) — "
                           "SYSTEM JOIN evaluates the sides the moment every "
                           "side of the variable is concluded.")
                self.messages.append({"role": "user", "content": _sn_msg})
                self.ctx.trajectory.append({"role": "tool", "content": _sn_msg})

        # quote unquoted ?variables — ONLY for JSON format (flat format breaks if
        # quoted: answer: ?x → answer: "?x" → Pydantic sees '"?x"' not '?x')
        if re.search(r'tool:\s*\{', raw_response):
            raw_response = re.sub(r'(?<=[,\[\s:])\?(\w+)(?=[,\]\s}])', r'"?\1"', raw_response)

        tool_name, parsed_args = _parse_with_repair(raw_response)
        if not tool_name:
            # DIAGNOSTIC: report the SPECIFIC JSON parse error so the model can fix it
            _diag = None
            for m in re.finditer(r'tool:\s*(\{)', raw_response):
                _seg = raw_response[m.start(1):]
                _depth, _end = 0, 0
                for _i, _c in enumerate(_seg):
                    if _c == '{': _depth += 1
                    elif _c == '}': _depth -= 1
                    if _depth == 0 and _i > 0:
                        _end = _i + 1; break
                _candidate = _seg[:_end] if _end else _seg
                try:
                    json.loads(_candidate)
                except json.JSONDecodeError as je:
                    _diag = (f"The `tool:` JSON has a syntax error: {je.msg} at char {je.pos}. "
                             f"Re-emit ONE clean `tool:` line with complete, valid JSON.")
                    break
                except Exception:
                    _diag = "The `tool:` JSON could not be parsed. Re-emit ONE clean `tool:` line."
                    break
            if _diag:
                fmt_nudge = _diag
            elif re.search(r"ANSWER_ANALYSIS", raw_response):
                fmt_nudge = ("Analysis does not change execution state. Submit one "
                             "useful retrieval or tool: answer with answer_values. "
                             "No separate analysis/confirmation turn is required.")
                self.messages.append({"role": "user", "content": fmt_nudge})
                self._last_tool_sig = None
                self._tool_repeat = 0
                return {"ret": "continue"}
            elif (re.search(r"\[[^\]]*[✓✗][^\]]*\]", raw_response)
                  or re.search(r"(?:^|\n)\s*[-*]?\s*(?:sg\d+(?:\.f\d+)?|f\d+)"
                               r"\s*[✓✔✗]", raw_response)):
                # checkpoint-only turn: the declarations were already merged by
                # _update_var_operations above — acknowledge, don't scold.
                # Wave-1: the old trigger `\[[^\]]*[✓✗]\]` matched only
                # "[fid ✓]"-shaped brackets (marker flush with `]`) — ✗
                # CLOSURE turns ("[f1 ✗ empty]") fell through to the generic
                # "No `tool:` call found" scold; any bracket carrying ✓/✗ is a
                # checkpoint turn. Var binding status rides along (ladder
                # pack ②, 2026-08-24): information, never enforcement — vars
                # link facts, binding is not forced.
                _fbv = getattr(self.ctx, "fact_bindings", None) or {}
                _fvv = getattr(self.ctx, "fact_vars", None) or {}
                _vstat = "; ".join(
                    f"{_v}=bound({len(_fbv[_f])})" if _fbv.get(_f)
                    else f"{_v}=unbound"
                    for _f, _v in list(_fvv.items())[:8])
                # UNKNOWN CLOSURE GUARD (Wave-1): a ✗ declaration outside the
                # closure grammar (`[f2 ✗ unresolved-after-repair]` & friends,
                # bracket or md form) binds nothing and closes nothing — a
                # bare "Checkpoints recorded" ack let the model believe the
                # fact was closed while completion accounting kept it open.
                # Name the offending line and teach the enumeration instead.
                _bad_close = [
                    b for b in (re.findall(r"\[[^\]]*✗[^\]]*\]", raw_response)
                                + [ln.strip() for ln in raw_response.splitlines()
                                   if re.search(r"✗", ln)
                                   and not ln.strip().startswith("[")])
                    if not (_CKPT_CLOSE_RE.search(b)
                            or _CKPT_CLOSE_RE_MD.search("\n" + b))]
                fmt_nudge = ("Checkpoints recorded. "
                             + (f"Subgraph vars: {_vstat}\n" if _vstat else "")
                             + ("⚠ Unrecognized closure status NOT recorded: "
                                + "; ".join(_bad_close[:2])
                                + ". A ✗ closure must use one of the status "
                                "words empty | moot | mismatch | exhausted "
                                "(an optional `: reason` may follow), e.g. "
                                "`- sg1.f2 ✗ empty`.\n" if _bad_close else "")
                             + "Now emit your next tool call, "
                             "flat format (one key per line):\n"
                             "tool: retrieve_relations\ncenter: <entity or ?var>\n"
                             "question: <sub-question>\n" + self.allowed_tools_hint())
            elif any(k in raw_response for k in ("CANDIDATES", "ANSWER:", "BASE_BINDINGS",
                                                 "FINAL_BINDINGS", "CONSTRAINT_CHECK",
                                                 "PROVISIONAL_FINAL")):
                fmt_nudge = ("Your content has the answer reasoning but no valid tool "
                             "call. Submit with flat format:\n"
                             "tool: answer\nanswer_values: A | B")
            else:
                fmt_nudge = (f"No `tool:` call found. Emit flat format — one `tool:` "
                             f"line then key: value lines (no JSON needed):\n"
                             f"tool: retrieve_relations\ncenter: <entity>\n"
                             f"question: <sub-question>\n"
                             f"{self.allowed_tools_hint()}")
            self.messages.append({"role": "user", "content": fmt_nudge})
            self.ctx.trajectory.append({"role": "tool", "content": "(no tool call)"})
            self._last_tool_sig = None; self._tool_repeat = 0   # different action — reset anti-loop
            return {"ret": "continue"}

        parsed_args = parsed_args or {}
        _pre_val_args = parsed_args          # pre-schema parse (see merge below)
        validated_args, schema_err = validate_args(tool_name, parsed_args)
        if schema_err:
            self.messages.append({"role": "user", "content": f"REJECTED: {schema_err}"})
            self.ctx.trajectory.append({"role": "tool", "content": f"REJECTED: {schema_err}"})
            self._last_tool_sig = None; self._tool_repeat = 0
            return {"ret": "continue"}
        parsed_args = validated_args
        # Legacy compatibility: requirements/covers are now schema fields.
        # Keep the merge for older/custom schema adapters.
        if tool_name in ("plan", "decompose"):
            if not validated_args.get("requirements") and _pre_val_args.get("requirements"):
                validated_args["requirements"] = _pre_val_args["requirements"]
            for _src, _dst in zip(_pre_val_args.get("subgraphs") or [],
                                  validated_args.get("subgraphs") or []):
                if _src.get("covers") and not _dst.get("covers"):
                    _dst["covers"] = _src["covers"]
        # anti-loop: consecutive IDENTICAL tool calls
        if tool_name in ("retrieve_relations", "retrieve_subgraph"):
            raw_fid = str(parsed_args.get("sg") or "")
            fid = norm_fact_key(self.state, raw_fid,
                center_hint=str((parsed_args.get("center") or [""])[0]))
            if raw_fid and fid not in self.state.fact_ids:
                self._emit_tool_note(f"REJECTED: FACT_SCOPE: unknown fact {raw_fid}; use one of {self.state.fact_ids}.")
                return {"ret": "continue"}
            if fid in self.state.fact_ids: parsed_args["sg"] = fid
        parsed_args["_protocol_scope"] = SP.context_scope(self.ctx)
        sig = (tool_name, json.dumps(parsed_args, sort_keys=True, ensure_ascii=False))
        if getattr(self, "_last_tool_errored", False):
            # an ERRORED call was never "served" — the checkpoint-corrected
            # retry is legitimate, not a loop (25_db96/62_bce8 specimens:
            # unbound-center error → retry with checkpoint → REJECTED (repeat)
            # → hallucinated relations → dump-all)
            self._last_tool_sig = None
            self._tool_repeat = 0
            self._last_tool_errored = False
        if sig == self._last_tool_sig:
            self._tool_repeat += 1
        else:
            self._last_tool_sig = sig
            self._tool_repeat = 1
        # REJECT identical repeats (not just nudge) — prevents wasted turns.
        # The model re-calling retrieve_relations with the same args gets the same
        # candidates. Force it to proceed: pick relations → retrieve_subgraph, or
        # declare checkpoint → next fact, or answer.
        # `answer` is EXEMPT (harness fix P1): it is an idempotent TERMINAL tool — an
        # identical re-call is confirmation of intent, not a wasted retrieval. Rejecting
        # it trapped compliant retries (the model followed a prior rejection's "call
        # answer again" instruction straight into this gate, forever). The premature/
        # offpool answer rejections are one-shot by design; the retry must pass through.
        if self._tool_repeat >= 2 and tool_name != "answer":
            nudge = self.loop_nudge()
            self.messages.append({"role": "user", "content": nudge})
            self.ctx.trajectory.append({"role": "tool", "content": f"REJECTED (repeat): {nudge}"})
            return {"ret": "continue"}

        # EVIDENCE COMMIT is a summary; READY answers proceed on this turn.
        self._maybe_inject_evidence_commit()

        _ans_entities = parsed_args.get("entities") if tool_name == "answer" else None
        _ready, _missing = False, None
        if tool_name == "answer":
            _cf = getattr(self.ctx, "closed_facts", None) or {}
            _fb2 = getattr(self.ctx, "fact_bindings", None) or {}
            _vj2 = getattr(self.ctx, "var_joins", None) or {}
            _ready = commit_due(self.state, _cf, _fb2, _vj2)
            # restart contract: after the ONE restart, an empty answer IS
            # the terminal the contract mandates — never remind again
            if (not _ready and getattr(self.ctx, "restarted", False)
                    and not _ans_entities):
                _ready = True
            if not _ready:
                _missing = blocking_facts(self.state, _cf, _fb2, _vj2)
                if _join_v2():
                    # R3-a: the reminder lists DECLARATION debt (facts with
                    # neither ✓ nor ✗), not retrieval debt — the V2 gate is
                    # the declaration clock
                    _missing = unconcluded_facts(self.state, _cf, _fb2)

        # MULTI-ANCHOR CONSUMPTION GATE (V3.7g, 2026-09-01, restored
        # 2026-09-07): intercept the answer ONE round when declared plan
        # entities never anchored a retrieval — MUST run before seq_validate
        # transitions the state to DONE: intercepting after the transition
        # leaves state=DONE with done unset, and every later call dies as
        # "REJECTED: Conversation finished" until the round budget burns out
        # (24-zombie family in the 2026-09-07 full-fix run).
        # READY-EXEMPT (user audit 2026-09-17, WebQTrn-21 specimen): when
        # every fact is closed and the answer is due, an unconsumed type
        # word has no retrieval form anyway — the gate only burned a round
        # between EVIDENCE COMMIT and the accepted answer. Skip it on READY.
        if tool_name == "answer" and _ans_entities and not _ready \
                and not self.state.reminded and self._ma_gate_would_fire(parsed_args):
            # One early-answer reminder covers both anchor and fact debt.
            self.state.reminded = True
            # deferred: the gate's join search is GTE-ranked (async) — the
            # caller runs it BEFORE anything else; state never transitions
            return {"ret": "continue", "_gate_deferred": (tool_name, parsed_args)}
        # POST-RESTART PLAN LATCH (Wave-1, 1812 specimen family): while the
        # latch is live (set by the restart block), the fresh attempt's FIRST
        # tool call may be `plan` regardless of residual phase — validate's
        # RETRIEVE branch would otherwise kill it as "Wrong tool in the
        # retrieve phase". Roll the machine (and the tool layer's
        # immutability flag) back so both gates accept the re-plan; the latch
        # clears right after the first ACCEPTED plan below.
        if (getattr(self, "_post_restart_plan_ok", False)
                and tool_name in ("plan", "decompose")
                and self.state.state != "INIT"):
            self.state = SeqAgentState()
            self.ctx.plan_declared = False
        # ONE MID-CASE RE-PLAN (user ruling 2026-09-20): restructure after a
        # dead-end is legitimate (2576 specimen: three re-plan attempts after
        # closing a dead fact, all REJECTED, burned to the round cap) — allow
        # exactly ONE re-plan per attempt beyond the initial plan; a second
        # is rejected as before. Uses the same roll-back as the restart
        # latch so both phase gates accept it.
        elif (tool_name in ("plan", "decompose")
                and self.state.state != "INIT"
                and not getattr(self, "_replan_used", False)):
            self._replan_used = True
            self.state = SeqAgentState()
            self.ctx.plan_declared = False
        ok, err, new_state = seq_validate(
            self.state, _to_tool_calls(tool_name, parsed_args),
            ready=_ready, missing_facts=_missing)
        if not ok:
            self.messages.append({"role": "user", "content": f"REJECTED: {err}"})
            self.ctx.trajectory.append({"role": "tool", "content": f"REJECTED: {err}"})
            return {"ret": "continue"}
        self.state = new_state
        if tool_name == "retrieve_subgraph":
            # CANONICAL FID FOR THE JOIN STORES (Team-A 2026-09-27): the
            # SYSTEM JOIN reads fact_bindings/fact_vars under the CANONICAL
            # declared fid (_canonical_decl_fid → fact_key_map), but the
            # tool-side pool/evidence stores were keyed by whatever string
            # the call carried — a walk invoked as `sg: sg2` (the prompt's
            # own continuation recipe) landed under the unresolvable "sg2"
            # and the cross-tree JOIN never saw that tree's pool, silently
            # degrading to curated-list intersection. Resolve the fid HERE
            # with the same norm_fact_key validate() used for retrieval
            # booking (single source of truth: bare-sg group + center-head
            # match) and hand it to seq_tools via the args.
            parsed_args["_canon_fid"] = norm_fact_key(
                self.state,
                str(parsed_args.get("sg") or parsed_args.get("fact_id")
                    or parsed_args.get("step") or ""),
                center_hint=str((parsed_args.get("center")
                                 or parsed_args.get("entities")
                                 or [""])[0] or ""))
        if tool_name in ("plan", "decompose"):
            self._post_restart_plan_ok = False

        # Repetition is handled by the scoped action identity above.
        # Query similarity never marks a fact resolved or contradicted.

        # sync the decompose projection onto ctx so dispatch sees the fact structure
        self.ctx.fact_ids = list(self.state.fact_ids)
        self.ctx.fact_texts = dict(self.state.fact_texts)
        self.ctx.fact_key_map = dict(getattr(self.state, "fact_key_map", None) or {})
        self.ctx.fact_edges = dict(getattr(self.state, "fact_edges", None) or {})
        self.ctx._model_anchor = self.state.anchor or ""
        if tool_name in ("decompose", "plan"):
            # plan-contract v2: capture the declared answer-type word — the
            # question-analysis anchor (who→person, what year→year...). Echoed at
            # answer time so the turn-0 reading stays in force all episode.
            at = str(parsed_args.get("answer_type") or "").strip()
            if at:
                self.ctx.plan_answer_type = at
            from kgqa.agent.loop import _resolve_anchor
            _resolve_anchor(self.ctx)
        return {"tool": tool_name, "args": parsed_args, "raw": raw_response}

    async def _execute_tool(self, prep, session) -> str:
        """B段 (IO): execute the prepared tool request. Exceptions land here as
        the dispatch error result — same semantics as the original single try
        around ST.dispatch (the finalization steps still run on error)."""
        tool = prep["tool"]
        try:
            result = await ST.dispatch(tool, prep["args"], self.ctx, session)
            if (os.environ.get("SEQ_ADAPTIVE_ROUTES", "0").strip().lower() in ("1", "true", "on", "yes")
                    and tool in ("plan", "decompose") and not SP.result_failed(result)
                    and os.environ.get("SEQ_PLAN_PATH_HINTS", "0") == "1"):
                names = list(getattr(self.ctx, "plan_entities", ()) or ())
                if len(names) >= 2:
                    try:
                        path = await self._search_join_paths(names[:1], session,
                            max_paths=3, max_hops=4, targets=names[1:])
                        if path:
                            result += ("\noptional_path_hint: " + path +
                                       "\nnote: This is a question-ranked reminder only. Choose other relations/centers freely; it is not delivered walk evidence or a JOIN verdict.")
                    except Exception:
                        result += "\npath_hint_status: unavailable; normal exploration remains available."
            return result
        except Exception as e:
            return json.dumps({"error": f"dispatch_{tool}: {e}"})

    def _finalize(self, prep, result_str) -> str:
        """C段 (pure CPU): land the tool result, merge evidence, fire the
        post-dispatch EVIDENCE COMMIT, assemble the messages."""
        if os.environ.get("SEQ_ADAPTIVE_ROUTES", "0").strip().lower() in ("1", "true", "on", "yes"):
            from kgqa.agent.seq_adaptive import loop_finalize
            return loop_finalize(self, prep, result_str)
        tool_name = prep["tool"]
        parsed_args = prep["args"]
        raw_response = prep["raw"]
        if SP.result_failed(result_str):
            self._last_tool_errored = "NO_EVIDENCE" not in result_str
            if tool_name in ("plan", "decompose") and prep.get("_previous_state") is not None:
                self.state = prep["_previous_state"]
                snapshot = prep.get("_previous_plan_context")
                if snapshot:
                    for key, value in snapshot["values"].items(): setattr(self.ctx, key, value)
                    for key in snapshot["absent"]:
                        if hasattr(self.ctx, key): delattr(self.ctx, key)
                self.ctx.plan_declared = prep.get("_previous_plan_declared", False)
                self._replan_used = prep.get("_previous_replan_used", False)
                self._post_restart_plan_ok = prep.get("_previous_restart_latch", False)
            note = ("Declaration checks completed; this action failed. "
                    if prep.get("_declarations_accepted") else "")
            self.messages.append({"role": "user", "content": note + result_str})
            self.ctx.trajectory.append({"role": "tool", "name": tool_name, "content": result_str})
            return "continue"
        from kgqa.agent.seq_harness import record_tool_success
        if not (tool_name == "retrieve_subgraph" and SP.result_repeated(result_str)):
            record_tool_success(self.state, tool_name, parsed_args)
        # errored calls are not "served": flag for the repeat detector so the
        # corrected retry is not rejected as a loop (see _parse_prepare).
        # SEQ's _json_result renders errors as FLAT "error: ..." / "entity_error: ..."
        # text, not JSON — the old '{"error"' probe never matched, so the
        # unbound-?var error → checkpoint declaration → corrected re-call hit
        # the repeat gate and dead-ended (1278d3da s2 specimen)
        _head = result_str[:120].lstrip()
        if (_head.startswith('{"error"') or '"error"' in result_str[:80]
                or _head.startswith("error:") or _head.startswith("entity_error:")):
            self._last_tool_errored = True
        self.messages.append({"role": "user",
                              "content": f"Tool result ({tool_name}): {result_str}"})
        self.ctx.trajectory.append({"role": "tool", "name": tool_name,
                                    "content": result_str})

        # plan-phase rollback: validate() transitions INIT→RETRIEVE on PARSE
        # success, but the tool handler can still reject (gate mismatch). If a
        # plan/decompose call was rejected downstream and NO plan was ever
        # accepted (plan_declared False), roll the state machine back to INIT
        # so re-emitting the (corrected) plan stays legal. Otherwise the phase
        # gate blocks exactly the re-emission the rejection message demanded
        # (CH specimen). When a plan IS already declared, the rejection is the
        # immutability gate — state stays RETRIEVE so the workflow continues.
        if (tool_name in ("plan", "decompose")
                and self.state.state == "RETRIEVE"
                and not getattr(self.ctx, "plan_declared", False)
                and ('"error"' in result_str or "REJECTED" in result_str)):
            self.state.state = "INIT"

        # BINDING LEDGER → EVIDENCE COMMIT (V2.1, 2026-08-22; COMMIT-WIDENING:
        # trigger is the READY predicate, not all_retrieved — see the method).
        # Post-dispatch site: a retrieval that completed the evidence fires the
        # ledger on the SAME turn (the pre-gate site runs before validate, so it
        # cannot see this turn's retrieval).
        self._maybe_inject_evidence_commit()

        if self.state.state == "DONE":
            self.done = True
            return "done"
        return "continue"


async def run_seq_react_case(session: aiohttp.ClientSession, sample: Dict[str, Any],
                             pilot_row: Dict[str, Any], idx: int, args) -> Dict[str, Any]:
    """Run one case through the SEQ per-fact loop (content-only `tool:` protocol).

    Each turn: LLM returns free-form content; parse_react_output extracts the
    `tool:` anchor + JSON; seq_harness.validate + seq_tools.dispatch run it.
    """
    from kgqa.llm.client import _call_single_with_reasoning, THINKING_TOKEN_BUDGET as _tb

    rc = SeqReactCase(sample, pilot_row, idx)
    await rc.prelink(session)
    max_rounds = int(getattr(args, "agent_max_iters", 16))
    max_tokens = int(getattr(args, "agent_max_tokens", 1024))
    if _tb > 0:
        max_tokens = max(max_tokens, _tb + 2560)

    for _ in range(max_rounds):
        if not rc.is_active:
            break
        rc.ctx._seq_turns_remaining = max_rounds - _
        msgs = list(rc.messages)
        hint = rc.state_aware_hint() + rc.allowed_tools_hint()
        if rc._tool_repeat >= 2:  # stuck on the same call — force convergence
            hint = rc.loop_nudge() + "\n" + hint
        msgs.append({"role": "user", "content": hint})

        try:
            raw_response, reasoning = await _call_single_with_reasoning(
                session, msgs, max_tokens=max_tokens)
        except Exception as e:
            rc.failed = True
            rc.failure_reason = f"call_llm error: {e}"
            break

        if await rc.process_turn(session, raw_response, reasoning) == "done":
            break
    else:
        rc.failed = True
        rc.failure_reason = f"max_iters ({max_rounds}) reached at state {rc.state.state}"

    rc.rescue_terminal_answer()   # harness fix P4: recover trapped answers
    result = _ctx_to_result_dict(rc.ctx, rc.state, rc.failed, rc.failure_reason)
    result["decomposition_method"] = "seq_per_fact"
    from kgqa.agent.seq_adaptive import attach_result
    attach_result(result, rc.ctx)
    if getattr(rc, "_rescued", False):
        result["terminal_rescue"] = True
    return result


async def run_seq_react_batch(cases_to_run, args):
    """Turn-synchronous SEQ batch runner — the production eval entry.

    Each round every active case advances exactly one turn: all cases' prompts are
    sent in ONE batched LLM call (``batch_call_llm_with_reasoning``, chunked to
    ``batch_chunk``), then each case's ``process_turn`` runs concurrently. This
    replaces the per-call path (``run_seq_react_case``) for evaluation: it is far
    faster (no per-turn HTTP round-trip per case) and avoids the per-call
    connection drops under concurrency that produced empty-answer regressions.

    ``cases_to_run``: list of ``(sample, pilot_row, idx)``. Returns
    ``[(sample, pilot_row, result_dict), ...]`` where result_dict carries
    ``agent_trajectory`` for dumping.
    """
    import time
    from kgqa.llm.client import THINKING_TOKEN_BUDGET as _tb
    from kgqa.llm.batch import batch_call_llm_with_reasoning

    total = len(cases_to_run)
    print(f"\n=== SEQ batch mode: {total} cases ===", flush=True)
    wall_start = time.perf_counter()

    react_cases = [SeqReactCase(s, pr, idx) for s, pr, idx in cases_to_run]
    max_rounds = int(getattr(args, "agent_max_iters", 16))
    max_tokens = int(getattr(args, "agent_max_tokens", 1024))
    if _tb > 0:
        max_tokens = max(max_tokens, _tb + 2560)
    batch_chunk = int(getattr(args, "batch_chunk", 25))

    async with aiohttp.ClientSession() as session:
        # plan-前置实体预检索（每 case 一次 GTE；并发受限避免 GTE 连接 reset）
        _pl_sem = asyncio.Semaphore(16)

        async def _pl(_rc):
            async with _pl_sem:
                await _rc.prelink(session)

        await asyncio.gather(*[_pl(rc) for rc in react_cases])

        for round_num in range(max_rounds):
            active = [rc for rc in react_cases if rc.is_active]
            if not active:
                break

            # build prompts (append allowed-tools hint + anti-loop nudge) — mirrors
            # the per-call runner's prompt assembly exactly.
            prompts = []
            for rc in active:
                rc.ctx._seq_turns_remaining = max_rounds - round_num
                hint = rc.state_aware_hint() + rc.allowed_tools_hint()
                if rc._tool_repeat >= 2:
                    hint = rc.loop_nudge() + "\n" + hint
                msgs = list(rc.messages)
                msgs.append({"role": "user", "content": hint})
                prompts.append(msgs)

            # batched LLM call (chunked — batch endpoint scales poorly past ~25)
            try:
                if len(prompts) <= batch_chunk:
                    contents, reasoning = await batch_call_llm_with_reasoning(
                        session, prompts, max_tokens=max_tokens)
                    chunk_results = [(contents, reasoning)]
                else:
                    chunks = [prompts[i:i + batch_chunk]
                              for i in range(0, len(prompts), batch_chunk)]

                    async def _b(chunk):
                        return await batch_call_llm_with_reasoning(
                            session, chunk, max_tokens=max_tokens)
                    chunk_results = await asyncio.gather(*[_b(c) for c in chunks])
            except Exception as e:
                print(f"  batch call failed: {e}", flush=True)
                for rc in active:
                    rc.failed = True
                    rc.failure_reason = f"batch_call error: {e}"
                break

            responses, responses_reasoning = [], []
            for cr_c, cr_r in chunk_results:
                responses.extend(cr_c)
                responses_reasoning.extend(cr_r)

            # process each case concurrently (dispatch is cheap except retrieve_*,
            # which is bounded in-process)
            dispatch_sem = asyncio.Semaphore(16)

            async def _proc(rc, raw, rsn):
                async with dispatch_sem:
                    if not raw:
                        # batch endpoint returns None on a per-prompt failure — treat
                        # like the per-call empty-response path: nudge and keep going.
                        rc.messages.append({"role": "assistant", "content": ""})
                        rc.messages.append(
                            {"role": "user", "content": rc.allowed_tools_hint()})
                        return
                    await rc.process_turn(session, raw, rsn)

            await asyncio.gather(*[_proc(rc, raw, rsn)
                                   for rc, raw, rsn in
                                   zip(active, responses, responses_reasoning)])

            done = sum(1 for rc in react_cases if rc.done)
            failed = sum(1 for rc in react_cases if rc.failed)
            print(f"  round {round_num + 1}: {len(active)} active → "
                  f"done={done} failed={failed}", flush=True)

    dt = time.perf_counter() - wall_start
    print(f"SEQ batch done: {total} cases in {dt:.0f}s ({dt / total:.1f}s/case)", flush=True)

    results: List[Tuple[Any, Any, Dict[str, Any]]] = []
    for (s, pr, idx), rc in zip(cases_to_run, react_cases):
        rc.rescue_terminal_answer()   # harness fix P4: recover trapped answers
        r = _ctx_to_result_dict(rc.ctx, rc.state, rc.failed, rc.failure_reason)
        r["decomposition_method"] = "seq_per_fact"
        from kgqa.agent.seq_adaptive import attach_result
        attach_result(r, rc.ctx)
        r["agent_trajectory"] = rc.ctx.trajectory
        if getattr(rc, "_rescued", False):
            r["terminal_rescue"] = True
        results.append((s, pr, r))
    return results


async def run_round_dispatch(cases_raws, session):
    """Round-level scheduler (perf-4): A loop for ALL active cases → B fires
    every GTE/walk request together (the seq_tools collectors coalesce the
    whole round into few lane jobs; GTE GPU and walk CPU lanes overlap) →
    C loop for all. Per-case semantics IDENTICAL to process_turn — that
    method keeps the sequential composition for the per-case runner,
    run_seq_react_batch and every test; only the phase interleaving differs
    (the event loop does orchestration only, sync CPU no longer fragments
    into 500-coroutine interleave)."""
    import time as _time_mod
    _perf = _time_mod.perf_counter
    prepared = []
    _t0 = _perf()
    for rc, raw, reasoning in cases_raws:
        if rc.done or rc.failed:
            continue
        prep = rc._parse_prepare(raw, reasoning)
        if "_gate_deferred" in prep:
            await rc._ma_gate_check(*prep["_gate_deferred"], session)
            continue
        if "ret" in prep:
            continue      # early-exit path: message side effects done in A
        prepared.append((rc, prep))
    _tA = _perf()
    if not prepared:
        return
    bres = await asyncio.gather(
        *(rc._execute_tool(prep, session) for rc, prep in prepared),
        return_exceptions=True)
    _tB = _perf()
    for (rc, prep), br in zip(prepared, bres):
        # _execute_tool already renders tool errors as result strings; an
        # exception here is harness-level — render it the same way so the
        # finalization steps still run and the case continues instead of
        # taking the whole round down.
        if isinstance(br, BaseException):
            print(f"  ⚠ dispatch exception ({prep['tool']}): "
                  f"{type(br).__name__}: {br}", flush=True)
            br = json.dumps({"error": f"dispatch_{prep['tool']}: {br}"})
        rc._finalize(prep, br)
    _tC = _perf()
    from kgqa.core.utils import PHASE_TIMES
    PHASE_TIMES["rd_A"] = PHASE_TIMES.get("rd_A", 0.0) + (_tA - _t0)
    PHASE_TIMES["rd_B"] = PHASE_TIMES.get("rd_B", 0.0) + (_tB - _tA)
    PHASE_TIMES["rd_C"] = PHASE_TIMES.get("rd_C", 0.0) + (_tC - _tB)


