"""SEQ harness — boundary-only state machine for iterative subgraph retrieval.
States: INIT → decompose → RETRIEVE → DONE (answer accepted).
The harness enforces ONLY boundaries — it never decides which relation, which center
entity, or when to answer. Those are the model's job. The boundaries:
  - order:  decompose first; answer last.
  - to-do:  track declared facts vs retrieved facts → progress hint + answer guard.
  - answer guard: if the model answers before every declared fact is retrieved, it is
            reminded ONCE (rejected with the missing list); a second answer is accepted.
The DATA boundary (next center ∈ accumulated subgraph) is enforced in the tool
(seq_tools.retrieve_subgraph), not here, because it needs ctx.
SAPS baseline (react_loop / harness.py / tools.py) is untouched.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from kgqa.agent.harness import _tool_name, _tool_args, _parse_triple_decompose
INIT = "INIT"
RETRIEVE = "RETRIEVE"   # retrieve_relations / retrieve_subgraph / answer
DONE = "DONE"
def _parse_subgraphs_decompose(args: dict) -> dict:
    """Parse the subgraphs decompose format (each named entity anchors a subgraph).
    Returns the same structure as _parse_triple_decompose so downstream code is unchanged."""
    subgraphs = args.get("subgraphs") or []
    entities_raw = args.get("entities")
    entities = entities_raw or []
    if isinstance(entities, str):
        entities = [e.strip() for e in entities.split(",") if e.strip()]
    answer = str(args.get("answer") or "").strip()
    if not subgraphs:
        return {"error": "plan must emit `subgraphs` (list of {id, anchor, facts})."}
    if not answer.startswith("?"):
        # internal repair: the answer field may be missing due to partial JSON parse
        # (reasoning-leak truncation). Infer from the last subgraph's last fact's tail.
        for sg in reversed(subgraphs):
            if isinstance(sg, dict):
                for fact in reversed(sg.get("facts") or []):
                    if isinstance(fact, (list, tuple)) and len(fact) >= 3:
                        tail = str(fact[-1]).strip()
                        if tail.startswith("?"):
                            answer = tail
                            break
                if answer.startswith("?"):
                    break
        if not answer.startswith("?"):
            return {"error": ("plan `answer` must be a ?variable (e.g. '?answer'). The answer "
                              "field was missing — if your JSON was truncated mid-write, re-emit "
                              "the plan as ONE complete `tool:` line with the full JSON.")}
    fact_ids, fact_texts, chains = [], {}, []
    sg_info = {}   # {sg_id → n_facts} for per-fact budget tracking
    fact_covers = {}   # fid → requirement id (V2.1 Question Contract linkage)
    # COMMIT-WIDENING (2026-08-22): fact identity spans three namespaces at
    # runtime — f1 (declared id), sg1 (retrieve_subgraph's `sg:` arg),
    # sg1.f1 (checkpoint declarations). Key everything to the DECLARED id so
    # completion counting and the READY predicate see one key per fact.
    sg_fact_ids = {}   # {sg_id → [fid, ...]} in declaration order
    fact_key_map = {}  # alias → declared fid
    fact_edges = {}    # fid → (head_token, tail_token), ?var kept verbatim
    for sg in subgraphs:
        if not isinstance(sg, dict):
            continue
        sg_id = str(sg.get("id") or f"sg{len(sg_info) + 1}")
        sg_anchor = str(sg.get("anchor") or "").strip()
        sg_facts = sg.get("facts") or []
        sg_cov = sg.get("covers") or {}
        sg_fids = []
        for fi, fact in enumerate(sg_facts, 1):
            if not isinstance(fact, (list, tuple)) or len(fact) < 3:
                continue
            head, rel, tail = str(fact[0]).strip(), str(fact[1]).strip(), str(fact[2]).strip()
            if not head or not rel:
                continue
            fid = f"f{len(fact_ids) + 1}"
            fact_ids.append(fid)
            fact_texts[fid] = f"({head} | {rel} | {tail})"
            fact_edges[fid] = (head, tail)
            fact_key_map[fid] = fid
            fact_key_map[f"{sg_id}.f{fi}"] = fid
            sg_fids.append(fid)
            cov = sg_cov.get(f"f{fi}") or sg_cov.get(str(fi))
            if cov:
                fact_covers[fid] = str(cov).strip().upper()
        if sg_fids:
            chains.append({"anchor": sg_anchor, "fact_steps": [[f] for f in sg_fids]})
            sg_info[sg_id] = len(sg_fids)
            sg_fact_ids[sg_id] = list(sg_fids)
    if not fact_ids:
        return {"error": "subgraphs contained no valid facts. Each fact: [head, sub-question, tail]."}
    if not entities:
        if entities_raw is None:
            return {"error": ("The `entities` field was not found in the parsed JSON — the plan JSON "
                              "may have been partially parsed (truncated/corrupted). Re-emit the plan "
                              "as ONE complete `tool:` line with valid JSON including the `entities` list."),
                    "entities_missing": True}
        return {"error": "`entities` is empty. List every named entity in the question.", "entities_missing": True}
    if answer not in {tail for _, tail in fact_edges.values()}:
        return {"error": f"ANSWER_PRODUCER: no planned fact produces {answer}; keep the requested answer slot and add its producer."}
    first_anchor = chains[0].get("anchor", "") if chains else ""
    requirements = args.get("requirements") if isinstance(args.get("requirements"), dict) else {}
    # answer var(s): the declared `answer` ?variable is authoritative; when the
    # plan format carries no explicit marker, fall back to the tail variables of
    # facts covered by an answer-kind requirement.
    answer_var = [answer] if answer.startswith("?") else []
    if not answer_var:
        for rid, req in (requirements or {}).items():
            if not isinstance(req, dict) or "answer" not in str(req.get("kind", "")).lower():
                continue
            for fid, cov_rid in fact_covers.items():
                if cov_rid == str(rid).upper():
                    tail = fact_edges.get(fid, ("", ""))[1]
                    if tail.startswith("?") and tail not in answer_var:
                        answer_var.append(tail)
    return {"fact_ids": fact_ids, "fact_texts": fact_texts, "chains": chains,
            "anchor": first_anchor, "entities": entities, "answer": answer,
            "sg_info": sg_info, "requirements": requirements,
            "fact_covers": fact_covers, "sg_facts": sg_fact_ids,
            "fact_key_map": fact_key_map, "fact_edges": fact_edges,
            "answer_var": answer_var}
@dataclass
class SeqAgentState:
    state: str = INIT
    # decompose projection
    fact_ids: list = field(default_factory=list)        # declared fact ids, in flow order
    fact_texts: dict = field(default_factory=dict)
    chains: list = field(default_factory=list)
    anchor: Optional[str] = None
    entities: list = field(default_factory=list)
    # to-do / answer guard
    retrieved_fids: list = field(default_factory=list)  # fact ids retrieved (when the model passes fact_id)
    n_subgraphs: int = 0                                # successful executed retrievals; never fact coverage
    reminded: bool = False                              # early-answer reminder fired once
    n_decomposes: int = 0
    # subgraph tracking (SEQ redesign — per-fact budget + deviation detection)
    sg_plan: dict = field(default_factory=dict)    # {sg_id → n_facts} from decompose
    requirements: dict = field(default_factory=dict)  # V2.1: R/D id → {kind, text}
    fact_covers: dict = field(default_factory=dict)   # V2.1: fid → requirement id
    sg_calls: dict = field(default_factory=dict)   # {sg_id → retrieve_subgraph call count}
    sg_done: set = field(default_factory=set)      # resolved/failed sg_ids
    # COMMIT-WIDENING pack (2026-08-22): READY-predicate state, filled at plan
    # parse / PLAN EXTEND; closed_fids+declared_fids are synced from ctx each turn.
    answer_var: list = field(default_factory=list)   # ?vars carrying the answer
    fact_key_map: dict = field(default_factory=dict) # alias (f1/sg1/sg1.f2) → declared fid
    fact_edges: dict = field(default_factory=dict)   # fid → (head, tail) tokens
    sg_facts: dict = field(default_factory=dict)     # sg_id → [fid, ...] in order
    closed_fids: set = field(default_factory=set)    # fids closed by ✗ declarations
    declared_fids: set = field(default_factory=set)  # fids with an accepted ✓ checkpoint
    def sg_budget(self, sg_id: str) -> int:
        """Max retrieve calls for this subgraph = 3 × n_facts (1 relations + 1 subgraph + 1 repair per fact)."""
        return 3 * self.sg_plan.get(sg_id, 1)
    def done_fids(self) -> set:
        """Declared fids that no longer need retrieval: retrieved, ✓ checkpointed,
        ✗ closed, or inside a budget-intercepted subgraph."""
        done = set(self.retrieved_fids) | self.closed_fids | self.declared_fids
        for sg in self.sg_done:
            done.update(self.sg_facts.get(sg, ()))
        return {f for f in done if f in self.fact_ids}
    @property
    def n_done(self) -> int:
        # Count facts, never calls.
        return len(self.done_fids())
    @property
    def all_retrieved(self) -> bool:
        # Every declared fact must have a recorded conclusion or retrieval.
        if not self.fact_ids:
            return False
        if self.done_fids() >= set(self.fact_ids):
            return True
        return False  # call counts are costs, never fact coverage
def norm_fact_key(state: SeqAgentState, key: str, center_hint: str = "") -> str:
    """Canonical (declared) fact id for any of the three runtime namespaces:
    `f1` (declared), `sg1` (retrieve_subgraph's sg arg), `sg1.f2` (checkpoint
    declaration). Unknown keys pass through unchanged so budget/closure stores
    stay inspectable.
    BARE-sg RESOLUTION (576 s2 specimen, 2026-09-21): a bare group id used to
    resolve to the group's first NOT-YET-retrieved fact — but a REPAIR
    re-retrieval of an already-retrieved fact (the model re-queried f1's head
    to widen its bindings) got booked under the NEXT fact instead: f1's
    evidence clock never advanced, the variable freeze stayed shut, and the
    model's legitimate fresh-evidence widening was rejected as a
    "hallucination" (while the never-queried f2 lit up as retrieved).
    center_hint is the call's own declared center string: when it equals a
    fact's declared HEAD token (the model's own strings, exact match —
    mechanical, no graph semantics), that fact wins over the sequential
    guess."""
    k = str(key or "").strip()
    if not k:
        # Missing optional ids can be inferred only when structurally unique.
        matching = [f for f in state.fact_ids if center_hint and
                    state.fact_edges.get(f, ("", ""))[0] == center_hint]
        if len(matching) == 1: return matching[0]
        if len(state.fact_ids) == 1: return state.fact_ids[0]
        return ""
    for cand in (k, k.lower()):
        if cand in state.fact_key_map:
            return state.fact_key_map[cand]
    if k in state.fact_ids:
        return k
    base = k.split(".")[0]
    fids = ()
    for cand in (k, base, k.lower(), base.lower()):
        fids = state.sg_facts.get(cand) or ()
        if fids:
            break
    if fids:
        _hint = str(center_hint or "").strip()
        if _hint:
            for f in fids:
                _head = state.fact_edges.get(f, ("", ""))[0]
                if _head and _head == _hint:
                    return f
        busy = set(state.retrieved_fids) | set(state.closed_fids) | set(state.declared_fids)
        for f in fids:
            if f not in busy:
                return f
        return fids[-1]
    return k
def _norm_done(state: SeqAgentState, closed_facts: dict) -> set:
    """done fids, counting ctx-side closures (✗ keys may be any namespace)."""
    done = state.done_fids()
    done |= {norm_fact_key(state, k) for k in (closed_facts or {})}
    return {f for f in done if f in state.fact_ids}
def _answer_vars_bound(state: SeqAgentState, fact_bindings: dict,
                       var_joins: dict) -> bool:
    """Every declared answer var has non-empty values — via a join intersection
    or any fact binding whose plan edge touches the var (tail = binding site,
    head = discriminator-walk site)."""
    for av in state.answer_var:
        if not av:
            continue
        join = (var_joins or {}).get(av)
        if join and join[1]:
            continue
        bound = False
        for raw_fid, vals in (fact_bindings or {}).items():
            fid = norm_fact_key(state, raw_fid)
            head, tail = state.fact_edges.get(fid, ("", ""))
            if av in (head, tail) and vals:
                bound = True
                break
        if not bound:
            return False
    return True
def blocking_facts(state: SeqAgentState, closed_facts: dict, fact_bindings: dict,
                   var_joins: dict) -> list:
    """OPEN declared facts that can still change WHICH entity answers — the
    inverse of ready_to_commit branch 2. The early-answer reminder lists only
    these; an empty list means READY and the reminder must stay silent."""
    done = _norm_done(state, closed_facts)
    open_facts = [f for f in state.fact_ids if f not in done]
    if not open_facts:
        return []
    avs = {v for v in state.answer_var if v}
    if not avs or not _answer_vars_bound(state, fact_bindings, var_joins):
        return open_facts
    tail_ans = {f for f in state.fact_ids
                if state.fact_edges.get(f, ("", ""))[1] in avs}
    tail_ans_rids = {state.fact_covers.get(f) for f in tail_ans}
    blocking = []
    for f in open_facts:
        head, tail = state.fact_edges.get(f, ("", ""))
        # tail==answer: would re-bind / intersect the answer set;
        # head==answer: discriminator walk (overshoot) — must finish first;
        # shared covers: a second anchor constraining the same requirement
        # under an alias variable can still eliminate candidates.
        if tail in avs or head in avs:
            blocking.append(f)
        elif state.fact_covers.get(f) and state.fact_covers.get(f) in tail_ans_rids:
            blocking.append(f)
    return blocking
def ready_to_commit(state: SeqAgentState, closed_facts: dict, fact_bindings: dict,
                    var_joins: dict) -> bool:
    """EVIDENCE COMMIT / answer-readiness (COMMIT-WIDENING, 2026-08-22):
    branch 1 — every declared fact is CLOSED (retrieved or ✗ closure, via the
    key map); branch 2 — the answer var is bound AND no OPEN fact can still
    change it (see blocking_facts). Deliberately NO transitive closure."""
    if not state.fact_ids:
        return False
    if _norm_done(state, closed_facts) >= set(state.fact_ids):
        return True
    if not state.answer_var:
        return False
    if not _answer_vars_bound(state, fact_bindings, var_joins):
        return False
    return not blocking_facts(state, closed_facts, fact_bindings, var_joins)
def all_facts_concluded(state: SeqAgentState, closed_facts: dict,
                        fact_bindings: dict) -> bool:
    """R3-a V2 COMMIT predicate (user ruling 2026-10-01, proposal §2 COMMIT
    row / §6.8-3): every DECLARED fact has a conclusion in the Declaration
    Ledger — a ✓ binding (an accepted checkpoint with content = a
    fact_bindings entry) or an explicit ✗ closure (closed_facts; harness-side
    closures — purity loop / CVT deadlock — count as ✗). The RETRIEVAL clock
    (retrieved_fids: sg delivered ⇒ retrieved) no longer qualifies a fact:
    the gap between that clock and the declaration clock is exactly the
    87.3% single-side-CANDIDATES COMMIT face (dataflow inventory C-③)."""
    if not state.fact_ids:
        return False
    return not unconcluded_facts(state, closed_facts, fact_bindings)
def unconcluded_facts(state: SeqAgentState, closed_facts: dict,
                      fact_bindings: dict) -> list:
    """Declared fids with NEITHER a ✓ binding nor an explicit ✗ — the
    DECLARATION WAIT list (R3-a): what COMMIT and JOIN still wait on."""
    done = {norm_fact_key(state, k) for k in (fact_bindings or {})}
    done |= {norm_fact_key(state, k) for k in (closed_facts or {})}
    return [f for f in state.fact_ids if f not in done]
def commit_due(state: SeqAgentState, closed_facts: dict, fact_bindings: dict,
               var_joins: dict) -> bool:
    """SEQ_WIDE_COMMIT (default on) selects the READY predicate; setting it off
    falls back to the legacy all_retrieved trigger — the regression bisect fuse.
    SEQ_JOIN_V2 (default ON, R3-a 2026-10-01): the trigger switches to the
    DECLARATION clock — all_facts_concluded (✓ binding or explicit ✗ per
    fact); retrieval-done-but-undeclared no longer commits (the environment
    prompts instead). SEQ_JOIN_V2=0 restores the retrieval-clock predicate
    byte-for-byte (rollback + A/B face)."""
    import os
    if os.environ.get("SEQ_JOIN_V2", "1").strip().lower() in ("0", "off", "false", "no"):
        if os.environ.get("SEQ_WIDE_COMMIT", "1").strip().lower() in ("0", "off", "false", "no"):
            return bool(state.all_retrieved)
        return ready_to_commit(state, closed_facts, fact_bindings, var_joins)
    return all_facts_concluded(state, closed_facts, fact_bindings)
def _todo(state: SeqAgentState) -> str:
    done = state.done_fids()
    marks = " ".join(f"{'✓' if f in done else '⬜'}{f}" for f in state.fact_ids) or "(no facts)"
    return f"{marks}  (retrieved {state.n_done}/{len(state.fact_ids)})"
def _allowed_hint(state: SeqAgentState) -> str:
    if state.state == INIT:
        return "`plan` (or `decompose`)"
    if state.state == RETRIEVE:
        todo = _todo(state)
        if state.all_retrieved:
            return (f"`retrieve_relations`/`retrieve_subgraph` (extra probes on seen entities) "
                    f"or `answer`. facts: {todo}")
        return (f"`retrieve_relations` finds relations for a NEW subgraph; "
                f"`retrieve_subgraph` continues an existing tree (anchor + relations — "
                f"the system walks the full chain). Both are valid. "
                f"facts: {todo}")
    return "(done)"
def validate(state: SeqAgentState, tool_calls, ready: bool = False,
             missing_facts=None) -> tuple:
    """One tool per turn. Returns (ok, err_msg, new_state).
    ``ready``/``missing_facts`` are supplied by the loop ONLY for `answer`
    calls: ready = the ctx-aware READY predicate held (skip the early-answer
    reminder entirely); missing_facts = the constraining-facts list for the
    reminder message. Both default to the legacy pure-state behavior."""
    if not tool_calls:
        return (False, "No tool call. " + _allowed_hint(state), state)
    if len(tool_calls) > 1:
        return (False, "ONE tool call per turn. " + _allowed_hint(state), state)
    name = _tool_name(tool_calls[0])
    args = _tool_args(tool_calls[0]) or {}
    # ── INIT: only decompose/plan ──
    if state.state == INIT:
        if name not in ("decompose", "plan"):
            return (False, "Call `plan` (or `decompose`) first.", state)
        # parse: subgraphs (new format) or flow (old fallback)
        if "subgraphs" in args:
            tri = _parse_subgraphs_decompose(args)
        else:
            tri = _parse_triple_decompose(args)
        if tri is None:
            return (False, "decompose must emit `subgraphs` (or `flow`) + entities + answer.", state)
        if "error" in tri:
            err = tri["error"]
            if "could not trace" in err:
                err += (" — every named ENTITY must have its own chain TO the answer ?variable. "
                        "If an entity is a CONSTRAINT, declare its chain FROM the constraint TO the answer "
                        "(e.g., [constraint_entity, 'which X are in this place', ?answer]), "
                        "not FROM the answer TO the constraint.")
            return (False, err, state)
        if tri.get("entities_missing") and state.n_decomposes == 0:
            state.n_decomposes += 1
            return (False, "decompose `entities` is empty. List EVERY named entity in the question.", state)
        # answer_type REQUIRED (plan-contract v2) — checked HERE, before the INIT→
        # RETRIEVE transition, so a rejected plan leaves the state machine in INIT
        # and re-emitting the plan stays legal. The CH specimen (2026-08-19) showed
        # the desync: validate transitioned on parse success, the tool-layer gate
        # then rejected, and the phase gate blocked the re-emission it had demanded
        # ("Wrong tool in the retrieve phase") — deadlock.
        if not str(args.get("answer_type", "") or "").strip():
            return (False, "plan requires `answer_type` — ONE word for what the "
                           "question asks for (person, country, movie, year, language, ...), "
                           "derived from the question's interrogative. Re-emit the plan with it.",
                    state)
        state.fact_ids = tri["fact_ids"]
        state.fact_texts = tri["fact_texts"]
        state.chains = tri["chains"]
        state.requirements = tri.get("requirements") or {}
        state.fact_covers = tri.get("fact_covers") or {}
        state.anchor = tri["anchor"]
        state.entities = tri.get("entities") or []
        state.sg_plan = tri.get("sg_info", {})       # {sg_id → n_facts} for per-fact budget
        state.sg_facts = tri.get("sg_facts", {}) or {}
        state.fact_key_map = tri.get("fact_key_map", {}) or {}
        state.fact_edges = tri.get("fact_edges", {}) or {}
        state.answer_var = tri.get("answer_var", []) or []
        state.n_decomposes += 1
        state.state = RETRIEVE
        return (True, "", state)
    # ── RETRIEVE: retrieve_relations / retrieve_subgraph / answer ──
    if state.state == RETRIEVE:
        if name == "retrieve_relations":
            # RR PER-CENTER BUDGET (user ruling 2026-09-22, cap 3): the
            # retrieve_relations call is deterministic (same center +
            # question → same GTE candidate list) and used to be entirely
            # unmetered — the repeat audit found trajectories burning 4+
            # identical rr calls on one center (2576-s2) while the sg budget
            # had no lever. Count by the call's declared center; exhaustion
            # blocks only that center's rr, with the submission-first
            # directive of the sg budget gate.
            _rrc = args.get("center") or args.get("entities") or []
            if isinstance(_rrc, str):
                _rrc = [_rrc]
            _rc0 = str(_rrc[0]).strip() if _rrc else ""
            if _rc0:
                _rkey = repr(("rr", args.get("sg"), tuple(_rrc), args.get("_protocol_scope"), args.get("question")))
                count = state.sg_calls.get(_rkey, 0) + 1
                if count > 3:
                    return (False, (f"retrieve_relations on center '{_rc0}' used "
                                    f"{count} calls in this query scope (budget 3 — the "
                                    f"program and query have not changed). Further rr in "
                                    f"THIS SCOPE is "
                                    f"rejected. NEXT: call retrieve_subgraph with "
                                    f"relations from the candidate list you already "
                                    f"have, or move to another center / the answer."),
                            state)
            return (True, "", state)                       # no state change
        if name == "retrieve_subgraph":
            # Admission never records a delivered fact. Finalize does that.
            # PER-CENTER RETRIEVAL BUDGET (user ruling 2026-09-20): count
            # retrieve_subgraph calls by their DECLARED START ENTITY, not per
            # subgraph — the model declares centers reliably (sg/fact ids are
            # the format-fragile part), and chained single-tree plans
            # legitimately need more probes inside one sg when an
            # intermediate binding set is wide (567 specimen: the 28-film
            # discrimination starved under the old 3×facts-per-sg cap; the
            # cap fired 15× in the run). Cap: 4 calls per declared center;
            # exhaustion blocks only THAT center, never the whole sg.
            centers = args.get("center") or args.get("entities") or []
            if isinstance(centers, str):
                centers = [centers]
            c0 = str(centers[0]).strip() if centers else ""
            if c0:
                key = repr(("sg", args.get("sg"), tuple(centers), args.get("_protocol_scope")))
                count = state.sg_calls.get(key, 0) + 1
                if count > 4:
                    return (False, (f"Retrieval in this scope used {count} calls "
                                    f"(budget 4 per fact/program scope). Choose another useful "
                                    f"program or submit the supported answer. This scope is "
                                    f"rejected. Answer NOW from the evidence already retrieved, or "
                                    f"continue retrieval from a DIFFERENT center."),
                            state)
            return (True, "", state)
        if name == "answer":
            # READY (loop-computed with ctx bindings) ⇒ no reminder at all — the
            # EVIDENCE COMMIT signal owns that turn; a reminder here would
            # double-reject. Otherwise list only facts that still CONSTRAIN the
            # answer (blocking set), not every unretrieved fact.
            if not ready and not state.all_retrieved and not state.reminded:
                state.reminded = True
                missing = (missing_facts if missing_facts is not None
                           else [f for f in state.fact_ids if f not in state.retrieved_fids])
                return (False, (f"You answered before retrieving every declared fact "
                                f"(missing: {missing}). Retrieve them first — or call `answer` "
                                f"again if you are certain the missing facts are unanswerable."), state)
            return (True, "", state)  # DONE only after answer executor succeeds
        return (False, f"UNKNOWN_ACTION: {name!r} is not available in this phase. "
                       + _allowed_hint(state), state)
    return (False, f"Conversation finished (state={state.state}).", state)


def record_tool_success(state, tool, args):
    """Record successful execution; attempted/rejected calls do not cover facts."""
    if tool == "retrieve_subgraph":
        state.n_subgraphs += 1
        fid = args.get("_canon_fid") or norm_fact_key(state, args.get("sg") or "",
            center_hint=str((args.get("center") or [""])[0]))
        if fid in state.fact_ids and fid not in state.retrieved_fids:
            state.retrieved_fids.append(fid)
        key = repr(("sg", args.get("sg"), tuple(args.get("center") or ()), args.get("_protocol_scope")))
        state.sg_calls[key] = state.sg_calls.get(key, 0) + 1
    elif tool == "retrieve_relations":
        key = repr(("rr", args.get("sg"), tuple(args.get("center") or ()), args.get("_protocol_scope"), args.get("question")))
        state.sg_calls[key] = state.sg_calls.get(key, 0) + 1
    elif tool == "answer":
        state.state = DONE
