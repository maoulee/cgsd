"""Tool-call ORDER state machine for the agent.

Enforces the strict sequence:
    INIT  → only `decompose`        (parses FLOW/ANCHORS/ANSWER triples; runs GTE per triple)
    RETRIEVE → optional `retrieve`  (fallback to re-fetch a fact's candidates)
    SELECT_RELATIONS → only `select_relations`  (picks relations + traverses)
    EXPAND → `expand_branches` (0-N) or `answer`
    ANSWER → only `answer`          (terminal)

Any out-of-order, skipped, or merged call is REJECTED with a guidance message.
The loop appends that guidance as a `tool` role message and re-prompts the model,
so the model can self-correct without losing the conversation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# State constants
INIT = "INIT"
RETRIEVE = "RETRIEVE"          # optional fallback (re-retrieve with new hint)
SELECT_RELATIONS = "SELECT_RELATIONS"  # select relations + traverse (merged)
EXPAND = "EXPAND"
ANSWER = "ANSWER"
DONE = "DONE"

# Order, for the "allowed next" hint
_ORDER = [INIT, RETRIEVE, SELECT_RELATIONS, EXPAND, ANSWER]

# Safety-net caps on model-driven rewrites. The MODEL decides when a choice was
# wrong and re-submits; these only intercept abuse (the system is a guardrail,
# not a decision-maker). Tool-internal errors (structural pruning, traversal
# reach) are fixed in the tool, NOT compensated by rewrites.
MAX_SELECTS = 2   # select_relations: 1 original + 1 model-driven rewrite
MAX_EXPANDS = 6   # expand_branches: loop-guard against meaningless repeated expands
# retrieve: 1 rewrite per fact (tracked per-fact in state.retrieved)


@dataclass
class AgentState:
    """Mutable order-enforcement state for one case."""
    state: str = INIT
    fact_ids: list = field(default_factory=list)       # all fact ids from decompose
    fact_texts: dict = field(default_factory=dict)     # id -> text (for guidance)
    fact_satisfies: dict = field(default_factory=dict)  # id -> constraint text (if fact materializes a condition)
    fact_start_types: dict = field(default_factory=dict)  # id -> start_type (anchor name for f1, type noun for f2+)
    fact_start_entities: dict = field(default_factory=dict)  # id -> start_entity (only for multi-anchor chain roots)
    fact_steps: list = field(default_factory=list)      # ordered step groups (each=[fid] seq, or [fid,...] a `con` conjunctive layer)
    chains: list = field(default_factory=list)          # parsed question_chains: [{anchor (name), fact_steps ([groups])}, ...]; one entry per independent anchor (multi-anchor). Single-chain = 1 entry.
    anchor: Optional[str] = None                       # model-chosen anchor entity name (= entities[0], compat for single-chain _resolve_anchor)
    entities: list = field(default_factory=list)        # ALL named entities the model declared (each → its own chain)
    endpoints: list = field(default_factory=list)      # model-chosen endpoint entity names (constraints)
    retrieved: list = field(default_factory=list)      # fact_ids already retrieved (via fallback retrieve)
    n_selects: int = 0
    n_expands: int = 0
    n_decomposes: int = 0
    terminal_answer: Optional[str] = None              # set when answer accepted

    @property
    def pending_facts(self) -> list:
        return [f for f in self.fact_ids if f not in self.retrieved]


def _tool_name(tool_call: dict) -> str:
    """Extract the tool name from either an OpenAI tool_call dict or a plain dict."""
    if not isinstance(tool_call, dict):
        return ""
    fn = tool_call.get("function") or {}
    if isinstance(fn, dict):
        return fn.get("name", "") or tool_call.get("name", "")
    return tool_call.get("name", "")


def _tool_args(tool_call: dict) -> dict:
    """Extract parsed arguments dict from a tool_call."""
    if not isinstance(tool_call, dict):
        return {}
    fn = tool_call.get("function") or {}
    if isinstance(fn, dict):
        raw = fn.get("arguments", {})
    else:
        raw = tool_call.get("arguments", {})
    if isinstance(raw, str):
        import json
        try:
            return json.loads(raw) if raw.strip() else {}
        except Exception:
            return {}
    return raw or {}


def _parse_triple_decompose(args: dict):
    """Parse triple-format decompose args {flow, anchors, answer} into the SAME
    structures the facts-format produces, so all downstream code (GTE, select,
    walk, expand, answer) is reused unchanged.

    Returns:
      None              — not triple-format (no `flow` key); caller falls back to
                          the facts-format parser.
      {"error": str}    — malformed triple decompose; caller rejects w/ guidance.
      {fact_ids, fact_texts, chains, anchor, ...} — parsed.

    Chain derivation (分解一致): each declared ANCHOR traces the FLOW to the ANSWER
    ?variable via shared ?variables. At each spine node, the spine edge (tail=?var)
    AND any parallel outgoing triples (constraints on that node) ALL depart that
    node → one conjunctive layer [spine_fid, *constraint_fids] (relations unioned,
    matching facts-format `con` semantics). Constraints on the TERMINAL answer
    node form their own con layer (e.g. worked-from / worked-to on the answer).
    """
    flow = args.get("flow")
    if flow is None:
        return None
    if not isinstance(flow, list) or not flow:
        return {"error": "decompose `flow` is empty. Emit FLOW triples "
                         "(head|relation|tail)."}

    entities = args.get("entities") or []
    if isinstance(entities, str):
        entities = [a.strip() for a in entities.split(",") if a.strip()]
    # Strip surrounding <> (some prompts bracket placeholders; the model can copy
    # them literally, which would break head==anchor matching in the trace).
    def _ent(s):
        return str(s).strip().strip("<>").strip()
    entities = [_ent(a) for a in entities if _ent(a)]
    entities_missing = not entities

    ans = str(args.get("answer") or "").strip()
    ans_var = ""
    ans_type = ""
    if ans.startswith("?"):
        paren = ans.split("(", 1)
        head = paren[0].strip()
        ans_var = head.split()[0] if head else ""
        if len(paren) > 1:
            inside = paren[1].split(")", 1)[0].strip()
            parts = inside.split()
            if parts and parts[0] in ("a", "an"):
                parts = parts[1:]
            ans_type = " ".join(parts)
    if not ans_var:
        return {"error": "decompose `answer` must be a ?variable, e.g. "
                         '"?country (a country)".'}

    triples = []
    for t in flow:
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            triples.append((_ent(t[0]), str(t[1]).strip(), _ent(t[2])))
        elif isinstance(t, str):
            parts = [p.strip() for p in t.split("|")]
            if len(parts) >= 3:
                triples.append((_ent(parts[0]), parts[1], _ent("|".join(parts[2:]).strip())))
    if not triples:
        return {"error": "decompose `flow` had no parseable (head|relation|tail) "
                         "triples."}

    # Empty `entities` → the model didn't list the named entities. The flow's
    # first non-variable head IS the seed entity (the model always places it as
    # f1's head); use it as a graceful fallback so downstream still runs, and
    # flag it so validate can reject-and-retry on the first attempt (the schema
    # requires entities >= 1). This replaces the old silent fallback that left
    # anchor="" and let _resolve_anchor pick a hub/generic q_entity (e.g.
    # "Gold medal" over "Dewitt High School").
    if entities_missing:
        for tr in triples:
            h = tr[0]
            if h and not h.startswith("?"):
                entities = [h]
                break

    fact_ids = []
    fact_texts = {}
    tri_fid = {}
    for tr in triples:
        if tr not in tri_fid:
            fid = f"f{len(fact_ids) + 1}"
            tri_fid[tr] = fid
            fact_ids.append(fid)
            fact_texts[fid] = f"({tr[0]} | {tr[1]} | {tr[2]})"
    out = {}
    for tr in triples:
        out.setdefault(tr[0], []).append(tr)

    def trace(anchor):
        steps = []
        cur = anchor
        visited = set()
        while cur != ans_var and cur not in visited:
            visited.add(cur)
            outs = out.get(cur, [])
            if not outs:
                break
            spine = next((t for t in outs if t[2].startswith("?")), None) or outs[0]
            departing = [tri_fid[spine]] + [tri_fid[t] for t in outs if t is not spine]
            steps.append(departing)  # con layer if >1 (all depart `cur`)
            cur = spine[2]
        if cur == ans_var:
            cons = [tri_fid[t] for t in out.get(ans_var, [])]
            if cons:
                steps.append(cons)
        return steps

    if entities:
        chains = [{"anchor": a, "fact_steps": trace(a)} for a in entities]
        for ch in chains:
            if not ch["fact_steps"]:
                return {"error": f"entity '{ch['anchor']}' could not trace to "
                                 f"answer {ans_var} via FLOW."}
    else:
        chains = [{"anchor": "", "fact_steps": [[fid] for fid in fact_ids]}]

    return {"fact_ids": fact_ids, "fact_texts": fact_texts, "chains": chains,
            "anchor": entities[0] if entities else "",
            "entities": entities, "entities_missing": entities_missing,
            "answer_type": ans_type}


def validate(state: AgentState, tool_calls) -> tuple:
    """Validate a batch of tool_calls against the order state machine.

    Returns ``(ok: bool, error_msg: str, new_state: AgentState)``.
    ``new_state`` is the SAME object (mutated in place) when ok, unchanged on
    rejection. The model is expected to emit exactly one tool call per turn in
    this workflow; a multi-call turn is treated as an attempted merge/skip and
    rejected so the model issues them one at a time.

    On rejection, ``error_msg`` is guidance the loop relays to the model.
    """
    calls = tool_calls or []
    if not calls:
        return (False,
                "You emitted no tool call. Call the next tool in order: "
                f"{_allowed_hint(state)}.",
                state)

    if len(calls) > 1:
        names = ", ".join(_tool_name(c) or "?" for c in calls)
        return (False,
                f"You emitted multiple tool calls in one turn ({names}). "
                "Issue ONE tool call per turn, in the required order. "
                f"Next expected: {_allowed_hint(state)}.",
                state)

    call = calls[0]
    name = _tool_name(call)
    args = _tool_args(call)

    # ── expand_branch name/arg normalize (defensive) ──
    # The schema/tool family was pluralized to `expand_branches` (batch), but
    # a model under prompt pressure can still emit the legacy singular form
    # `expand_branch` + `branch_id`. Normalize it here so the rest of validate
    # and dispatch see the canonical plural form — never reject on name shape.
    # (This was the latent bug behind the select-rules 0.80→0.15 regression:
    # harness rejected singular while dispatch accepted it.)
    if name == "expand_branch":
        name = "expand_branches"
        if "branch_id" in args and "branch_ids" not in args:
            bid = args.pop("branch_id")
            args["branch_ids"] = [bid] if isinstance(bid, str) else (bid or [])

    # ── INIT: only decompose ──
    if state.state == INIT:
        if name != "decompose":
            return (False,
                    f"Wrong order: '{name}' called before decompose. "
                    "You MUST call `decompose` first to emit FLOW + ANCHORS + ANSWER.",
                    state)
        # ── decompose = TRIPLE format (FLOW + ANCHORS + ANSWER) — replaces facts ──
        # Triples ARE the decomposition (the legacy facts/question_chains mechanism
        # is removed): each triple is one hop of the information flow; ANCHORS are
        # the named entities the flow starts from; ANSWER is the terminal ?variable.
        # GTE runs PER-TRIPLE (hint = the triple's relation clause) and the walk
        # runs PER-ANCHOR along the flow — both driven by this information flow.
        # expand_branches (path selection) + answer reuse the original engines
        # unchanged. Produces state.chains in the shape downstream expects.
        tri = _parse_triple_decompose(args)
        if tri is None:
            return (False,
                    "decompose must emit the TRIPLE format: args.flow (a list of "
                    "[head, relation, tail] triples), args.anchors (the named "
                    "entities the flow starts from), args.answer (the ?variable "
                    "the question asks for, + its type). No `flow` found.", state)
        if "error" in tri:
            return (False, tri["error"], state)
        # Empty-entities intercept (reject once, then accept the flow-head
        # fallback): the schema requires entities >= 1. On the FIRST empty
        # attempt, reject with guidance so the model re-declares the named
        # entities (it knows them — they are the flow heads). On a second empty
        # attempt, accept _parse_triple_decompose's flow-head fallback (avoids
        # a stuck loop; never silently picks a q_entity hub).
        if tri.get("entities_missing") and state.n_decomposes == 0:
            state.n_decomposes += 1
            return (False,
                    "decompose `entities` is empty. Declare EVERY named entity the "
                    "question names (the proper nouns — people, places, countries, "
                    "orgs — that the flow starts from); `entities` requires >= 1. "
                    "Each entity becomes its own walk. Re-call `decompose` with "
                    "`entities` filled.", state)
        state.fact_ids = tri["fact_ids"]
        state.fact_texts = tri["fact_texts"]
        state.fact_satisfies = {}
        state.fact_start_types = {}
        state.fact_start_entities = {}
        state.chains = tri["chains"]
        state.fact_steps = tri["chains"][0]["fact_steps"] if tri["chains"] else []
        state.anchor = tri["anchor"]
        state.entities = tri.get("entities") or []
        state.endpoints = []
        state.n_decomposes += 1
        # GTE runs inline in _do_decompose (per-triple) → skip RETRIEVE, go
        # straight to SELECT_RELATIONS. (retrieve remains available as a fallback
        # via the RETRIEVE state if the model re-enters it.)
        state.state = SELECT_RELATIONS
        return (True, "", state)

    # ── RETRIEVE: optional fallback (re-retrieve with a new hint) ──
    # Only reached if the model explicitly calls retrieve after decompose
    # (e.g. GTE candidates were structurally pruned to empty and the model
    # wants to retry with a different subquestion). In the normal flow,
    # decompose already returns GTE candidates and this state is skipped.
    if state.state == RETRIEVE:
        if name == "select_relations":
            state.state = EXPAND
            return (True, "", state)
        if name != "retrieve":
            return (False,
                    f"Wrong order: '{name}' during retrieve fallback. "
                    "Call `retrieve` to re-fetch a fact's candidates with a new "
                    "hint, or `select_relations` to proceed.",
                    state)
        fid = str(args.get("fact_id", ""))
        if not fid or fid not in state.fact_ids:
            return (False,
                    f"retrieve fact_id '{fid}' is unknown. Valid ids: {state.fact_ids}.",
                    state)
        if fid in state.retrieved:
            return (False,
                    f"retrieve limit reached for fact '{fid}' (1 rewrite per fact). "
                    "Proceed to `select_relations` with the current candidates, or "
                    "revise a different fact.",
                    state)
        state.retrieved.append(fid)
        return (True, "", state)

    # ── SELECT_RELATIONS: select relations + traverse (merged) ──
    # select_relations now ALSO runs the graph traversal (formerly the
    # separate `select` tool) and returns the evidence tree inline. So
    # after select_relations the model goes straight to EXPAND.
    if state.state == SELECT_RELATIONS:
        if name == "retrieve":
            # Allow retrieve as a fallback even here (1 rewrite per fact)
            fid = str(args.get("fact_id", ""))
            if not fid or fid not in state.fact_ids:
                return (False,
                        f"retrieve fact_id '{fid}' is unknown. Valid ids: {state.fact_ids}.",
                        state)
            if fid in state.retrieved:
                return (False,
                        f"retrieve limit reached for fact '{fid}' (1 rewrite per fact). "
                        "Proceed with `select_relations`.",
                        state)
            state.retrieved.append(fid)
            return (True, "", state)
        if name != "select_relations":
            return (False,
                    f"Wrong order: '{name}' called before select_relations. "
                    "Call `select_relations` once to pick the relation chain "
                    "from each fact's candidates. The system will traverse "
                    "automatically and return the evidence tree.",
                    state)
        # select_relations now also runs traverse → go straight to EXPAND
        state.n_selects += 1
        state.state = EXPAND
        return (True, "", state)

    # ── EXPAND: expand_branches (batch, 0-N calls), 1 backward select rewrite, or answer ──
    if state.state == EXPAND:
        # Backward rewrite: model may re-issue select_relations ONCE if it judges
        # the traversal overview wrong (safety-net cap; the MODEL decides when).
        # Dispatch re-runs the traverse and returns a fresh overview; stay in EXPAND.
        if name == "select_relations":
            if state.n_selects < MAX_SELECTS:
                state.n_selects += 1
                return (True, "", state)
            return (False,
                    "select_relations rewrite limit reached (1 rewrite). Work "
                    "with the current evidence: call `expand_branches` or `answer`.",
                    state)
        if name == "expand_branches":
            bids = args.get("branch_ids")
            if not bids:
                return (False,
                        "expand_branches missing `branch_ids`. Pass a list of "
                        "branch numbers from the tree overview, e.g. ['1','2'].",
                        state)
            state.n_expands += 1
            if state.n_expands > MAX_EXPANDS:
                return (False,
                        f"expand_branches loop limit reached ({MAX_EXPANDS} calls). "
                        "Stop expanding and call `answer` with the best evidence so far.",
                        state)
            return (True, "", state)  # accept; loop executes
        if name == "answer":
            state.state = DONE
            return (True, "", state)
        return (False,
                f"Wrong order: '{name}' during expand phase. "
                "Call `expand_branches(['1','2',...])`, re-issue `select_relations` "
                "(once, if the overview looks wrong), or `answer`.",
                state)

    # ── ANSWER: only answer, terminal ──
    if state.state == ANSWER:
        if name != "answer":
            return (False,
                    f"Wrong order: '{name}' called during answer phase. "
                    "Call `answer` with the entities from the evidence.",
                    state)
        state.state = DONE
        return (True, "", state)

    # Already DONE
    return (False,
            f"Conversation already finished (state={state.state}). "
            "Stop calling tools.",
            state)


def _allowed_hint(state: AgentState) -> str:
    """Human-readable hint of what tool is expected next."""
    if state.state == INIT:
        return "`decompose`"
    if state.state == RETRIEVE:
        return "`retrieve` (fallback) or `select_relations`"
    if state.state == SELECT_RELATIONS:
        return "`select_relations` (picks relations + traverses the graph)"
    if state.state == EXPAND:
        return "`expand_branches(['1','2'])`, `select_relations` (1 rewrite if the overview is wrong), or `answer`"
    if state.state == ANSWER:
        return "`answer`"
    return "(done)"
