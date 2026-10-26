"""SEQ dispatch — iterative subgraph retrieval (two tools) + decompose + answer.

Per declared fact the model runs:
  retrieve_relations(entity, question)  → GTE candidate relations around the entity
  retrieve_subgraph(entity, relations)  → walk + CVT penetration → DENSE tree

CRITICAL: retrieve_subgraph REUSES the proven SAPS walk (stage_5_graph_traversal:
multi-hop, K-path, CVT-penetrating, relation-centric subgraph) — NOT a 1-hop walk.
Scoped to ONE step from the current center. SAPS files stay untouched.

Boundaries enforced here (need ctx):
  - center entity must be in the accumulated subgraph (or the anchor for fact 1).
  - answer values must occur in delivered evidence (SEQ-specific handler).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict

from kgqa.core.utils import normalize
from kgqa.traversal.cvt import is_cvt_like
# attribute-key → answer-type class (display roster alignment, 2026-09-08;
# unknown keys map to "?" = keep — never hide a potential answer)
_ATTR_TYPE_CLASSES = {
    "character": "person", "actor": "person", "person": "person",
    "military_person": "person", "leader": "person", "member": "person",
    "office_holder": "person", "presenter": "person", "director": "person",
    "portrayed_in_films": "person", "employee": "person", "winner": "person",
    "founders": "person", "member": "person", "leader": "person",
    "starring": "film", "members": "organization",
    "film": "film", "movie": "film",
    "language": "language",
    "location": "location", "place": "location",
    "organization": "organization", "employer": "organization",
    "school": "organization",
}
from kgqa.agent.tools import (
    _gte_for_triple, _reach2_relids, _cvt_attr_summary,
    _constraint_attr_summary, _GTE_POOL_MIN,
)
from kgqa.core.case_state import CaseState
from kgqa.stages.stage5_traverse import stage_5_graph_traversal
from kgqa.traversal.path_utils import compress_paths
from kgqa.traversal.logical_paths import materialize_selected_logical_patterns
from kgqa.stages.formatting import build_pattern_evidence_triples

# perf-attribution lanes (values live in core.utils.PHASE_TIMES; registered
# here so utils stays untouched): walk_wait/walk_exec = the affine worker's
# submit→pickup / pickup→done split (perf_counter is CLOCK_MONOTONIC on Linux
# → cross-process comparable), walk_collect = the round-level coordinator's
# batch window (arrival→flush), gte_n = gte_retrieve call count.
from kgqa.core.utils import PHASE_TIMES as _PHASE_TIMES
for _k in ("walk_wait", "walk_exec", "walk_collect", "gte_n"):
    _PHASE_TIMES.setdefault(_k, 0.0)
del _k


# ───────────────────────── helpers ─────────────────────────

def _json_result(payload: Any) -> str:
    """SEQ tool-result renderer: readable key:value text (NOT JSON).
    Overrides tools._json_result so SEQ results read naturally for the model; SAPS
    keeps its own JSON renderer untouched. Skips empty/zero fields; lists → 'a | b';
    multi-line strings (e.g. triples) get their own indented block."""
    if not isinstance(payload, dict):
        return str(payload)
    lines = []
    for k, v in payload.items():
        if v is None or v == "" or v == [] or v == 0:
            continue
        if isinstance(v, list):
            lines.append(f"{k}: {' | '.join(str(x) for x in v)}")
        elif isinstance(v, str) and "\n" in v:
            lines.append(f"{k}:")
            lines.append(v.rstrip())
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines) if lines else "(empty)"


def _name_to_idx(name: str, ctx) -> int | None:
    """Map entity name → graph idx. Robust layered resolution:
    1. RAW exact (name == e) — strongest; excludes prefix-noise entities (:Caribbean) that
       normalize-collide with the clean name but are isolated pool=0 nodes. Without this,
       normalize(':Caribbean')=='caribbean'==normalize('Caribbean') and the first collision
       wins → retrieve_relations gets a pool=0 entity → fallback all-rels → GTE returns
       unreachable candidates → walk reaches nothing (idx88 root cause).
    2. normalize exact + COLLISION resolution — when several entities normalize to one key,
       prefer raw==name, else the SHORTEST raw (prefix-noise ':X' is longer than 'X').
    3. accent-insensitive (Vietnamese diacritics — normalize corrupts them, strip first).
    4. substring containment + similarity gate (≥0.85) — variant containment
       ('United States' ⊂ 'United States of America'), gated so a fragment like 'museum'
       ⊂ 'harvard museum' (ratio <0.85) doesn't match.
    5. fuzzy typo (Larr Baer → Larry Baer, difflib ≥0.85).

    PERF (2026-08-23): each step rescanned all ents with fresh normalize calls
    (~10k calls/run, O(V) each, in the event loop). The result depends only on
    (name, ctx.ents) — both immutable per case — so the whole resolution is
    memoized per ctx, and the normalize list is computed once."""
    import difflib, unicodedata
    if not name or not str(name).strip():
        return None
    _memo = getattr(ctx, "_n2i_memo", None)
    if _memo is None:
        _memo = {}
        ctx._n2i_memo = _memo
    _mk = str(name)
    if _mk in _memo:
        return _memo[_mk]
    name = str(name).strip()
    n = normalize(name)
    res = None
    if n:
        norms = getattr(ctx, "_ent_norms", None)
        if norms is None or len(norms) != len(ctx.ents):
            norms = [normalize(e) for e in ctx.ents]
            ctx._ent_norms = norms

        # 1. RAW exact
        for i, e in enumerate(ctx.ents):
            if e == name:
                res = i
                break

        if res is None:
            # 2. normalize exact + collision resolution (multiple entities → one normalize key)
            n_matches = [i for i, en in enumerate(norms) if en == n]
            if n_matches:
                raw_eq = next((i for i in n_matches if ctx.ents[i] == name), None)
                if raw_eq is not None:
                    res = raw_eq
                else:
                    res = min(n_matches, key=lambda i: len(ctx.ents[i]))

        if res is None:
            # 3. accent-insensitive (strip accents on RAW before normalize)
            def _strip_accents(text):
                return ''.join(c for c in unicodedata.normalize('NFKD', text)
                               if not unicodedata.combining(c))
            n_na = normalize(_strip_accents(name))
            if n_na and n_na != n:
                for i, e in enumerate(ctx.ents):
                    if normalize(_strip_accents(e)) == n_na:
                        res = i
                        break

        if res is None and len(n) >= 3:
            # 4. substring containment + similarity gate (variants yes, fragments no)
            for i, en in enumerate(norms):
                if (en and len(en) >= 2 and (n in en or en in n)
                        and _match_sim(name, ctx.ents[i]) >= 0.85):
                    res = i
                    break

        if res is None:
            # 5. fuzzy typo
            close = difflib.get_close_matches(n, norms, n=1, cutoff=0.85)
            if close:
                res = norms.index(close[0])
    _memo[_mk] = res
    return res


def _match_sim(a, b) -> float:
    """Name-similarity ratio (0–1) between an input and a matched entity, to tell a
    confident match from a substring-FRAGMENT match. _name_to_idx's substring step
    matches 'museum' inside 'harvard museum of modern colors' (a fragment → wrong
    entity), which a None-check alone misses. difflib fuzzy in _name_to_idx already
    gates at 0.85, so a real fuzzy/typo match scores ≥0.85; a fragment scores far
    below — 0.80 separates them."""
    import difflib
    return difflib.SequenceMatcher(None, normalize(str(a)), normalize(str(b))).ratio()


def _split_pipe_entities(entities) -> list:
    """Defensively split a '|'-joined center into its intended multiple entities.

    The model sometimes copies the DISPLAY format ('A | B | C') as ONE center string
    (e.g. inside a JSON-array element), and the parser keeps it as one entity name →
    a wrong, unwieldy center. Split on ' | ' (the display separator, which entity
    names essentially never contain) to recover the intended list. System-layer fix:
    recovers the intent silently rather than erroring or nudging (no burned turn)."""
    out = []
    for e in entities:
        e = str(e)
        if " | " in e:
            out.extend([x.strip() for x in e.split(" | ") if x.strip()])
        else:
            out.append(e)
    return out


def _looks_like_value(name) -> bool:
    """True if the input is a literal VALUE (numeric/coordinate/raw date/UTC offset),
    not a named entity. Value-like strings resolve UNRELIABLY via fuzzy/substring —
    'UTC-05:00' fuzzy-matches 'UTC−04:00' (a DIFFERENT timezone, high string-sim but
    wrong entity) — so callers require EXACT match for value-like centers (see the
    resolve loops). Named entities ('2014 World Series', 'Boeing 747') are NOT mostly
    digits and pass through normal resolution."""
    s = str(name).strip().strip("'\"")
    if len(s) < 2:
        return False
    digits = sum(c.isdigit() for c in s)
    if digits >= max(2, len(s) * 0.5):
        return True                          # coordinates, numbers, ISO dates
    # UTC/GMT offsets: 'UTC-05:00', 'UTC+12', 'GMT-5', 'UTC−04:00' (U+2212 minus)
    if re.search(r'\b(UTC|GMT)\s*[+\-−]\s*\d', s, re.IGNORECASE):
        return True
    return False


async def _entity_correction(name, question, ctx, session, top_k_candidates: int = 5,
                             top_k_rels: int = 3):
    """Entity-name CORRECTION (model-in-the-loop, not auto-resolve).

    When an input entity name isn't in the graph (_name_to_idx fails), GTE-find
    candidate entities by NAME similarity, and for each candidate show its TOP-K
    question-relevant NEIGHBOR relations (GTE: original question vs the candidate's
    1-hop relations). The model reads the neighbor relations to disambiguate (a
    battle has commander/participants; a city has population/area) and re-calls with
    the right entity. Name-match alone is unreliable (several high-similarity
    candidates), so the relations are the disambiguation signal — but the MODEL
    decides, the system only informs.

    Only for the initial named entity (literal center), NOT for ?var bindings.
    Returns a list of {name, neighbor_relations} candidates, or None if no
    session / no named entities / GTE fails (caller falls back to a plain error)."""
    from kgqa.stages.stage2_entity import gte_retrieve
    if session is None or not str(name).strip():
        return None
    # named entities only (skip CVT m-ids + short/empty norms that match anything)
    named = [(i, e) for i, e in enumerate(ctx.ents)
             if e and not is_cvt_like(e) and len(normalize(e)) >= 3]
    if not named:
        return None
    idxs = [i for i, _ in named]
    names = [e for _, e in named]
    name2idx = {e: i for i, e in zip(idxs, names)}   # name → graph idx (first occurrence)
    # 1. GTE candidate entities by name similarity to the (wrong) input.
    #    gte_retrieve returns rows with 'candidate' (the matched name) + 'score'.
    try:
        from kgqa.core.utils import phase_timer
        with phase_timer("gte"):   # UNTIMED BLIND SPOT closed 2026-08-21: this
            # pool is the FULL case entity list (thousands of texts) — a major
            # dispatch cost that the phase attribution previously missed.
            rows = await gte_retrieve(session, str(name), names,
                                      candidate_texts=names, top_k=top_k_candidates)
    except Exception:
        return None
    out = []
    for row in rows:
        ename = row.get("candidate")
        if not ename:
            continue
        eidx = name2idx.get(ename)
        if eidx is None:
            continue
        # 2. this candidate's 1-hop NEIGHBOR relations
        rels_1hop = set()
        for k in range(len(ctx.h_ids)):
            if (ctx.h_ids[k] == eidx or ctx.t_ids[k] == eidx) and 0 <= ctx.r_ids[k] < len(ctx.rels):
                rels_1hop.add(ctx.r_ids[k])
        if not rels_1hop:
            continue
        rel_names = sorted(rels_1hop, key=lambda r: ctx.rels[r])
        # 3. top-K question-relevant NEIGHBOR relations — reuse the proven _gte_for_triple
        #    (raw relation names embed poorly; _gte_for_triple labels them as
        #    'head | rel | ?' triples, which GTE ranks correctly).
        try:
            ranked = await _gte_for_triple(ctx, session, ename, question or str(name),
                                           "", pool_relids=rels_1hop)
            top_rels = [ctx.rels[r] for r in ranked[:top_k_rels] if 0 <= r < len(ctx.rels)]
        except Exception:
            top_rels = [ctx.rels[r] for r in rel_names[:top_k_rels]]
        if top_rels:
            out.append({"name": ename, "neighbor_relations": top_rels[:top_k_rels]})
        if len(out) >= top_k_candidates:
            break
    # Seed the candidates into subgraph_entities — when the model re-calls with one,
    # the boundary check (which requires prior retrieve_subgraph) would otherwise reject
    # a correction-picked entity (deadlock: correction → pick candidate → "not in prior
    # retrieve"). The correction sanctions these candidates, so they're valid centers.
    if out:
        se = getattr(ctx, "subgraph_entities", None)
        if se is not None:
            n2i_all = _norm_idx_all(ctx)
            for c in out:
                for idx in n2i_all.get(normalize(c["name"]), ()):
                    se.add(idx)
    return out if out else None


def _expand_entities(entities, ctx):
    """Expand any ?variable in an entity-name list to its declared bindings
    (ctx.var_bindings, populated from the model's checkpoints). Literal named
    entities (no `?` prefix) pass through. Returns (names, error). An unresolved
    ?var yields an error nudge telling the model to declare it first. De-dups
    preserving order."""
    vb = getattr(ctx, "var_bindings", {}) or {}
    out = []
    for e in entities:
        e = str(e)
        if e.startswith("?"):
            bound = vb.get(e)
            if not bound:
                # CVT-CHAIN DEADLOCK EXIT (Norwood specimen, 2026-08-22): the
                # variable's only declared values were EVENT nodes (stripped by
                # the §7.5 guard) — re-declaring them can never bind it, and a
                # later fact planned to walk FROM this variable is stranded.
                # The event's ATTRIBUTES are the entities: point the model at
                # them instead of the generic re-declare instruction (which
                # caused a 16-turn declare/error loop).
                _cev = getattr(ctx, "cvt_empty_var", None)
                if _cev and _cev[0] == e:
                    return [], (
                        f"variable {e} can NEVER bind: its declared values were "
                        f"EVENT nodes (m./g. ids) and events are not entities "
                        f"(§7.5) — re-declaring them returns here forever. The "
                        f"event's bracketed ATTRIBUTES are the entities (e.g. "
                        f"award_nominee=…). Close the stranded fact "
                        f"`[{_cev[1]} ✗ moot]` (earlier evidence already shows "
                        f"its target inside the brackets) and continue the next "
                        f"fact with an ATTRIBUTE entity as the center.")
                return [], (
                    f"variable {e} has no declared bindings. After the "
                    f"retrieve_subgraph that resolves it, declare its values in a "
                    f"checkpoint line `[<fact_id> ✓] {e} = [value1 | value2 | ...]`, "
                    f"then reference {e}.")
            out.extend(bound)
            # WANDERED-CANDIDATE COMPLETION (user audit 2026-09-08, 1797
            # specimen): the declared bindings are the DISPLAY entities; a
            # walk candidate the display filter showed only as "not shown
            # above" (gold Pemberton — its date_of_death edge is the sole
            # date source in the graph) never joins the centers, so the
            # relation pool and the walk can never reach it. Attach the
            # accumulated not-shown candidates — RECENCY-CAPPED (walk cost
            # audit 2026-09-10): the ledger grows monotonically every sg
            # call, and each ?var expansion re-walks EVERY entry as a full
            # environment traversal (RPE is relset-independent — user cost
            # model: one walk + per-relation accounting). Late-turn ?var
            # calls were walking 30-50 centers (slot count 2343/run, walk
            # exec ×6). The mechanism only ever needed the RECENT wanderers
            # (1797: Pemberton from the immediately prior sg); ancient ones
            # were already reachable when they were fresh.
            _wxcap = int(os.environ.get("SEQ_WALLEXTRA_CAP", "6") or 0)
            _wx = getattr(ctx, "walk_extra", None) or []
            if _wxcap > 0 and len(_wx) > _wxcap:
                _wx = _wx[-_wxcap:]
            out.extend(x for x in _wx if x not in out)
        else:
            out.append(e)
    seen, dedup = set(), []
    for e in out:
        if e not in seen:
            seen.add(e); dedup.append(e)
    return dedup, None


def _variable_nudge(raw_entities, ctx) -> str:
    """REMOVED (user ruling 2026-09-21): the warning existed because binding
    loss used to mean core-path loss — under the current design, bindings
    feed the relation-discovery pool only, never gate the walk; quantity is
    never limited and the only requirement is non-empty. Always returns ''.
    Wiring sites (2957, 3605 + 6 prepend points) call this for future
    re-enablement; the function body is inert."""
    if getattr(ctx, "_adaptive_worker", False):
        return ""
    return ""


def _cvt_repipe(name: str) -> str:
    """Repipe a CVT inline-attr node display from comma- to |-separated.
    'm.02xgww5: [actor=X, character=Y]' → 'm.02xgww5: [actor=X | character=Y]'.
    Commas collide with entity names that contain commas (e.g. 'St. John's,
    Montgomery, Alabama'); '|' never appears in Freebase names. Non-CVT names
    (no ': [' marker) pass through unchanged."""
    if not isinstance(name, str) or ": [" not in name or not name.endswith("]"):
        return name
    prefix, attrs = name.split(": [", 1)
    attrs = attrs[:-1]
    parts = [a.strip() for a in attrs.split(",") if a.strip()]
    return f"{prefix}: [{' | '.join(parts)}]" if parts else f"{prefix}: []"


def _is_cvt_node(name) -> bool:
    return (isinstance(name, str)
            and (name.startswith("m.") or name.startswith("g.") or name.startswith("CVT:")))


# Inverse-relation table for display cycle-suppression. Freebase stores directed
# inverses as distinct names (no .inv twin); these are the directional pairs whose
# second traversal re-establishes the first (a containment/administrative loop), plus
# the symmetric (self-inverse) set. A node revisited via a DIFFERENT relation is NOT
# a cycle — only an edge that repeats or whose inverse was already established.
_INVERSE_PAIR = {
    "location.location.contains": "location.location.containedby",
    "location.location.containedby": "location.location.contains",
    "location.location.partially_contains": "location.location.partially_contains",  # treat as symmetric fallback
    "base.aareas.schema.administrative_area.administrative_children": "base.aareas.schema.administrative_area.administrative_parent",
    "base.aareas.schema.administrative_area.administrative_parent": "base.aareas.schema.administrative_area.administrative_children",
    "location.location.contains_major_portion_of": "location.location.containedby",
}
_INVERSE_SYM = {
    "location.location.adjoin_s",
    "location.adjoining_relationship.adjoins",
}


def _edge_inverse(h, r, t):
    """Return the inverse edge of (h,r,t) if r has a known inverse/symmetric twin,
    else None. Symmetric relations invert to (t, r, h); directional pairs to
    (t, r_partner, h)."""
    if r in _INVERSE_SYM:
        return (t, r, h)
    partner = _INVERSE_PAIR.get(r)
    if partner:
        return (t, partner, h)
    return None


def _chain_is_cyclic(nodes, rels, accumulated) -> bool:
    """Edge-level cycle test (node-revisit via a DIFFERENT relation is allowed — a
    node is not unvisitable). A chain is cyclic if any edge (h,r,t):
      - is itself, OR whose inverse is, in `accumulated` (the PRIOR calls' triples —
        subgraph N looping back onto an earlier subgraph's edge), OR
      - repeats, or whose inverse repeats, an edge earlier in the SAME chain.
    `accumulated` must be the snapshot from BEFORE the current call's _accumulate,
    so a call never suppresses its own freshly-retrieved edges. Keys on relation +
    both endpoints, never on a seen node alone — a forward critical path that reuses
    an entity via a new relation (e.g. airport.serves → containedby → Germany) is kept."""
    accumulated = accumulated or set()
    seen_in_path = set()
    for k in range(len(nodes) - 1):
        h, t = nodes[k], nodes[k + 1]
        r = rels[k] if k < len(rels) else ""
        fwd = (h, r, t)
        inv = _edge_inverse(h, r, t)
        if fwd in accumulated or fwd in seen_in_path:
            return True
        if inv is not None and (inv in accumulated or inv in seen_in_path):
            return True
        seen_in_path.add(fwd)
    return False


def _clean_cvt_node(name: str, head: str, rel: str) -> str:
    """Suppress the redundant back-edge attr from a CVT inline-attr display, then
    dedup identical attrs, re-emitting '|'-separated.

    A CVT reached FROM `head` via `rel` lists that same relation pointing back at
    `head` (e.g. 'Denmark --adjoins--> m.02wrxld [adjoins=Denmark | adjoins=Germany]'
    — `adjoins=Denmark` just restates the entity we walked from). That entry is
    dropped; only the new endpoint(s) remain. Safe because the rule is
    relation-aware: it only fires when the attr key == the traversed relation's
    short name AND the value == the predecessor. Forrest-Gump-style CVTs (reached
    via 'starring', attrs actor/character/character_note) match no key → nothing
    dropped. Non-CVT names pass through."""
    if not isinstance(name, str) or ": [" not in name or not name.endswith("]"):
        return _cvt_repipe(name)
    prefix, attrs = name.split(": [", 1)
    attrs = attrs[:-1]
    parts = [a.strip() for a in attrs.split(",") if a.strip()]
    rel_short = (rel.rsplit(".", 1)[-1] if rel else "").replace(".inv", "")
    seen, kept = set(), []
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            k_base = k.strip().replace(".inv", "")
            if rel_short and k_base == rel_short and v.strip() == head:
                continue                       # back-edge to the entity we came from
        if p not in seen:
            seen.add(p); kept.append(p)
    return f"{prefix}: [{' | '.join(kept)}]" if kept else f"{prefix}: []"


_REL_CHAIN_RE = re.compile(r'--\[([^\]]+)\]-->')


def _relation_chain(readable: str) -> str:
    """Extract the PURE relation chain from a PatternEvidence.readable, stripping
    the starting-entity prefix. 'Americas --[location.location.time_zones]--> node1
    [materialized paths=2]' -> 'location.location.time_zones'; a 2-hop readable ->
    'r1 → r2'. The same chain shared across centers thus groups under one header."""
    rels = _REL_CHAIN_RE.findall(readable or "")
    return " → ".join(rels) if rels else (readable or "")


def _pattern_template(readable: str) -> str:
    """The PATTERN header as a node-template path: 'node1 --r1--> node2 [--r2--> node3 …]'.
    Relations live ONLY in the header; the body is pure entity chains that fill the
    slots in order. Center-agnostic, so multi-center (variable) groups merge under it."""
    rels = _REL_CHAIN_RE.findall(readable or "")
    if not rels:
        return readable or ""
    s = "node1"
    for i, r in enumerate(rels):
        s += f" --{r}--> node{i + 2}"
    return s


def _format_pattern_body(pe, accumulated=None, max_tails: int = 60, max_chains: int = 4) -> list:
    """Body lines for ONE PatternEvidence (no header). Length-2 paths group by
    (head, relation) into 'head → t1 | t2 | t3' ('|' avoids comma-in-entity-name
    collisions). ≥3-hop paths render as explicit chains 'head --r1--> mid --r2-->
    leaf' — hops preserved: a named intermediate is a selectable center, so the
    next relation points forward and is NEVER collapsed to an inline '(attr)'.
    CVT m-id nodes inline their radiating entities (hub-style — they are entities).
    Capped to prevent prompt explosion on high-degree relations."""
    paths = (getattr(pe, "tree_data", None) or {}).get("paths") or []
    out = []
    short, chains = [], []           # length-2 paths / length-3+ chains
    for p in paths:
        nodes = p.get("nodes") or []
        rels = p.get("relations") or []
        if len(nodes) < 2:
            continue
        # edge-level cycle filter (RAW nodes/rels, before CVT display-cleaning): skip
        # chains that repeat/inverse a PRIOR-call edge (subgraph N→subgraph 1) or an
        # earlier edge in the same chain. `accumulated` is the snapshot from BEFORE this
        # call's _accumulate, so a call never suppresses its own freshly-retrieved edges.
        # A node revisit via a DIFFERENT relation is allowed — only edge cycles drop.
        if _chain_is_cyclic(nodes, rels, accumulated):
            continue
        # clean each CVT node's inline attrs against its predecessor (drop the
        # back-edge that restates the entity we just came from), then repipe to '|'.
        cleaned = []
        for pos, n in enumerate(nodes):
            if pos > 0 and _is_cvt_node(n):
                prev_rel = rels[pos - 1] if pos - 1 < len(rels) else ""
                cleaned.append(_clean_cvt_node(n, nodes[pos - 1], prev_rel))
            else:
                cleaned.append(_cvt_repipe(n))
        nodes = cleaned
        if len(nodes) == 2:
            short.append((nodes[0], rels[0] if rels else "", nodes[1]))
        else:
            rlen = min(len(rels), len(nodes) - 1)
            chains.append((list(nodes), list(rels[:rlen])))   # keep (nodes, rels) for cycle filter (Part 4)
    # 1-hop body: entity-only (relation lives in the header template); group by head.
    groups = {}
    for head, _rel, tail in short:
        groups.setdefault(head, []).append(tail)
    for head, tails in groups.items():
        if len(tails) > max_tails:
            tails = tails[:max_tails] + [f"... +{len(tails) - max_tails} more (see candidates)"]
        out.append(f"  {head} → {' | '.join(tails)}")
    # multi-hop body: entity-only chains ('n0 → n1 → n2'); dedup by entity sequence, cap.
    seen, kept, dropped = set(), 0, 0
    for cnodes, crels in chains:
        cstr = " → ".join(cnodes)
        if cstr in seen:
            continue
        seen.add(cstr)
        if kept < max_chains:
            out.append(f"  {cstr}"); kept += 1
        else:
            dropped += 1
    if dropped:
        out.append(f"  ... +{dropped} chain paths truncated (walk exploration; see candidates)")
    return out


def _short_rel(r: str) -> str:
    """Last path segment of a relation name (organization.organization.parent → parent)."""
    r = str(r)
    return r.rsplit(".", 1)[-1] if r else r


# Noisy relation short-names to drop from the rendered tree — BOOKKEEPING and
# COMMON-TOPIC noise ONLY (zero answer signal), plus the role/type HUB quartet:
# member/organization/role/appointees radiate from title entities (Governor,
# team, organization) to EVERY holder across the graph — via LEGAL pattern
# paths (they are the answer edge for what-group questions), so the path
# discipline cannot stop them; measured 2026-08-18: removing them cost −3.5pp
# (0.8186→0.7834) — flooding beats the 30 what-group questions it would serve.
# "profession" REMOVED (answer edge, +1.5pp); alias/name still blocked — alias
# floods every entity, and the 6 another-name questions pay less than the noise.
_EDGE_NOISY_SHORT = {
    "type", "types", "instance", "instances", "notable_types", "notable_type",
    "webpage", "mid", "guid", "key", "keys", "permission",
    "article", "description", "is_reviewed", "image", "webpage_topic",
    "topic_equivalent_webpage", "alias", "name", "descriptive_name",
    "organization", "appointees",
    # "member"/"role" REMOVED (user audit 2026-08-26, Randy Jackson specimen):
    # these are RECORD-SEMANTIC edges (group_membership.member/.role carry who
    # played what in which group), not bookkeeping — filtering them from the
    # render indexes (a) killed the CVT record's inline attrs ([member=Randy;
    # role=Vocals; group=Journey] — the gold), (b) broke direction resolution
    # and instance reconstruction for membership-bridge chains, and (c) silently
    # dropped musical_group.member from tier-1 blocks even when the model
    # SELECTED it. Unselected member/role edges still never render standalone
    # (selection discipline in blocks/facts); they surface only inside CVT
    # brackets and as chain hops, both sanctioned.
}


def _canonicalize_triples(triples, ctx) -> list:
    """Flip triples whose rendered direction is REVERSED vs the raw graph edge.

    The walk is undirected (DIRECTED_TRAVERSAL=0): an edge is traversed from
    whichever side the center sits on, and the recorded triple puts the CENTER
    first. That reverses relations which canonically point INTO the center — e.g.
    `sports.mascot.team` is mascot→team, but retrieved from the team center it
    renders as 'team --team--> mascot', the wrong way. The base model reads the
    arrow literally, so a reversed arrow misleads it. This flips direct (both
    named, non-CVT) triples back to the raw canonical (h→t) direction. CVT-
    mediated triples keep their traversal direction (the CVT is the raw source
    of its out-edges, so they are already canonical).

    PERF (2026-08-23): the raw edge set + both name maps were rebuilt per call
    (O(E)+O(V) in the event loop); the arrays are immutable per case, so all
    three are cached on ctx."""
    raw = getattr(ctx, "_canon_raw_edges", None)
    if raw is None:
        raw = set(zip(ctx.h_ids, ctx.r_ids, ctx.t_ids))
        ctx._canon_raw_edges = raw
    n2i = getattr(ctx, "_canon_n2i", None)
    if n2i is None:
        n2i = {normalize(e): i for i, e in enumerate(ctx.ents) if e}
        ctx._canon_n2i = n2i
    r2i = getattr(ctx, "_canon_r2i", None)
    if r2i is None:
        r2i = {}
        for i, r in enumerate(ctx.rels):
            r2i.setdefault(r, i)
        ctx._canon_r2i = r2i
    out = []
    for tr in triples:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            out.append(tr); continue
        h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
        if is_cvt_like(h) or is_cvt_like(t):
            out.append(tr); continue                       # CVT-mediated: keep
        hi, ri, ti = n2i.get(normalize(h)), r2i.get(r), n2i.get(normalize(t))
        if hi is None or ri is None or ti is None:
            out.append(tr); continue
        if (hi, ri, ti) in raw:
            out.append(tr)                                 # already canonical
        elif (ti, ri, hi) in raw:
            out.append((t, r, h))                          # reversed → flip
        else:
            out.append(tr)
    return out


def _resolve_cvt_edges(triples) -> list:
    """Flatten CVT-mediated 2-paths into named→named edges for display.

    A Freebase compound fact runs (named --r1--> CVT --r2--> named2); the CVT is a
    transparent mediator and r2 is the meaningful role edge. This collapses such
    2-paths to (named, r2, named2). Direct named→named edges are kept. CVT tails
    with no named head and pure CVT↔CVT edges are dropped (not answerable).

    Inverse / variant relations that surface the SAME entity pair via different
    relation names (film.film_subject.films vs film.film.subjects; people.person.
    sibling_s vs people.sibling_relationship.sibling) all collapse onto one named
    edge per entity pair after resolution — the root cause of the 4× pattern
    explosion. Returns [(head_name, rel_short, tail_name)] all named.
    """
    cvt_out = {}                       # cvt_name -> [(rel_short, named_tail)]
    direct = []
    for tr in triples:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            continue
        h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
        if is_cvt_like(h) and not is_cvt_like(t):
            cvt_out.setdefault(h, []).append((_short_rel(r), t))
        elif not is_cvt_like(h) and not is_cvt_like(t):
            sr = _short_rel(r)
            if sr in _EDGE_NOISY_SHORT or normalize(h) == normalize(t):
                continue                   # drop bookkeeping noise + self-loops
            direct.append((h, sr, t))
    resolved = []
    seen = set()
    for h, r, t in direct:
        k = (normalize(h), r, normalize(t))
        if k not in seen:
            seen.add(k); resolved.append((h, r, t))
    for tr in triples:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            continue
        h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
        if not is_cvt_like(h) and is_cvt_like(t):     # named → CVT: continue through it
            for r2, t2 in cvt_out.get(t, ()):
                if r2 == "has_no_value":
                    # incumbent signal: t2 names the ABSENT attr (e.g. "To") → render as
                    # 'holder --to--> (incumbent)', the discriminator for current/latest
                    # holder on exclusive relations. Carried as a triple so it survives
                    # without the separate discriminating_attrs view.
                    attr = str(t2).lower()
                    k = (normalize(h), attr, "(incumbent)")
                    if k not in seen:
                        seen.add(k); resolved.append((h, attr, "(incumbent)"))
                    continue
                if is_cvt_like(t2) or r2 in _EDGE_NOISY_SHORT:
                    continue
                if normalize(h) == normalize(t2):
                    continue                           # self-loop (e.g. sibling listing self)
                k = (normalize(h), r2, normalize(t2))
                if k not in seen:
                    seen.add(k); resolved.append((h, r2, t2))
    return resolved


_MERGE_TAIL_CAP = 120   # per-line tail cap; beyond it advertise a branch ref (below)


def _inline_events(edges):
    """Replace CVT/MID endpoints with inline attribute blocks gathered from the
    same edge set: 'FC --starring--> Portrayal' + 'Portrayal --character--> Denver'
    + 'Portrayal --actor--> JF' renders as 'FC --starring--> [character=Denver;
    actor=JF]' — the event node's IDENTITY is its attributes (user spec: point
    out WHAT the event is, e.g. someone's position/performance), no bare mids.
    ATTR BRACKET PRINTED ONCE per CVT per render (2026-08-21, Angelina awards
    specimen): hub CVTs appear under several heads (nominee/ceremony/award/
    work spokes) — inlining attrs at EVERY appearance quadrupled the display.
    First appearance carries the block; later spokes reference the bare id."""
    cvt_attrs = {}
    for h, r, t in edges:
        if is_cvt_like(h) and not is_cvt_like(t):
            cvt_attrs.setdefault(h, []).append((_short_rel(r), t))
    inlined = set()

    def _repl(name):
        if is_cvt_like(name) and name in cvt_attrs:
            if name in inlined:
                return name
            inlined.add(name)
            attrs = cvt_attrs[name][:4]
            return name + " [" + "; ".join(f"{a}={v}" for a, v in attrs) + "]"
        return name
    # drop the CVT-headed attribute edges (their content now lives in the inline
    # block); keep named↔named and named→CVT edges with the CVT inlined
    return [(_repl(h), r, _repl(t)) for h, r, t in edges
            if not (is_cvt_like(h) and not is_cvt_like(t))]


def _merge_edges(edges, max_per_line: int = 8) -> list:
    """Merge resolved edges into compact list-form lines, BOTH directions:
      same (h, r) with differing t      → 'h --r--> t1 | t2 | ...'      (t-list)
      same (r, t-set) with differing h  → 'h1 | h2 | ... --r--> t1|t2'  (h-list)
    Grouping by (r, t-set) catches the multi-t case too: 8 siblings each pointing
    at the same parent-set collapse to '[Ted|Robert|…] --parents--> Joseph|Rose'.
    Entities joined by ' | '. Returns rendered strings (caller indents).

    List-type patterns larger than _MERGE_TAIL_CAP render the first cap entities +
    a '#center|relation' BRANCH REF: the model submits that ref as one answer
    entity and the backend expands it to the FULL pattern (see _do_answer) —
    compressed submission, system-level expansion."""
    by_hr = {}                          # (h, r) -> [t ...] (ordered, deduped)
    for h, r, t in edges:
        by_hr.setdefault((h, r), [])
        if t not in by_hr[(h, r)]:
            by_hr[(h, r)].append(t)
    # group the per-h t-lists by (r, t-set) so heads sharing the same tail-set merge
    groups = {}                         # (r, tuple(ts)) -> [ts_list, [h...]]
    for (h, r), ts in by_hr.items():
        key = (r, tuple(ts))
        if key not in groups:
            groups[key] = [ts, []]
        groups[key][1].append(h)
    out = []
    for (r, ts), (ts_list, hs) in groups.items():
        hs = list(dict.fromkeys(hs))
        if len(ts_list) > _MERGE_TAIL_CAP:
            extra = len(ts_list) - _MERGE_TAIL_CAP
            # '::' separator — the flat protocol splits on '|', a '|' ref would
            # parse as two entities.
            branch = (f"+{extra} more (total {len(ts_list)}). To answer with ALL of "
                      f"them, include \"#{hs[0]}::{r}\" as one answer entity — the "
                      f"system expands the ref to the full list.")
            out.append(f"{' | '.join(hs)} --{r}--> {' | '.join(ts_list[:_MERGE_TAIL_CAP])} {branch}")
        elif len(hs) > _MERGE_TAIL_CAP:
            extra = len(hs) - _MERGE_TAIL_CAP
            out.append(f"{' | '.join(hs[:_MERGE_TAIL_CAP])} +{extra} more --{r}--> {' | '.join(ts_list)}")
        else:
            out.append(f"{' | '.join(hs)} --{r}--> {' | '.join(ts_list)}")
    return out


def _shown_edge_key(h: str, r: str, t: str) -> tuple:
    return (normalize(h), r, normalize(t))


def _ctx_inverse_rels(ctx) -> dict:
    """Per-case inverse-relation folding (user ruling 2026-09-23).

    Freebase double-materializes many facts under BOTH direction relations
    (music.artist.concert_tours vs music.concert_tour.artist): the same
    (entity, entity) pair carries two raw edges with complementary
    directions and different relation ids. Any dedup key comparing exact
    (h, r, t) therefore misses the redelivery of an already-shown fact
    under its inverse id (1731 specimen: `Journey --artist.concert_tours->
    Raised on Radio Tour` delivered, later re-entered as `Raised on Radio
    Tour --concert_tour.artist--> Journey`).

    Derivation: relations that occur as direction-complementary edges on
    >= 2 shared entity pairs AND on >= half of the smaller relation's edge
    count are an inverse pair; both fold onto the lexicographically smaller
    full name. Cached on ctx, keyed by the (immutable-per-case) edge count.
    """
    n_edges = len(ctx.h_ids)
    cached = getattr(ctx, "_inv_rels_cache", None)
    if cached is not None and cached[0] == n_edges:
        return cached[1]
    from collections import defaultdict
    pair_edges = defaultdict(set)          # (min_idx, max_idx) -> {(h, t, r)}
    for h, r, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        if h != t:
            pair_edges[(min(h, t), max(h, t))].add((h, t, r))
    deg = defaultdict(int)
    co = defaultdict(int)                  # sorted (r1, r2) -> complementary pairs
    for members in pair_edges.values():
        for (_h, _t, r) in members:
            deg[r] += 1
        ml = sorted(members)
        for i in range(len(ml)):
            for j in range(i + 1, len(ml)):
                (h1, t1, r1), (h2, t2, r2) = ml[i], ml[j]
                if r1 != r2 and h1 == t2 and t1 == h2:
                    co[(min(r1, r2), max(r1, r2))] += 1
    rep = {}
    for (r1, r2), c in co.items():
        if c >= 2 and c * 2 >= min(deg[r1], deg[r2]):
            n1 = str(ctx.rels[r1]) if 0 <= r1 < len(ctx.rels) else str(r1)
            n2 = str(ctx.rels[r2]) if 0 <= r2 < len(ctx.rels) else str(r2)
            keep = min(n1, n2)
            rep[n1] = keep
            rep[n2] = keep
    ctx._inv_rels_cache = (n_edges, rep)
    return rep


def _edge_fact_key(h: str, r: str, t: str, inv: dict) -> tuple:
    """Direction-normalized + inverse-folded fact key: unordered endpoints
    plus the relation pair's representative. (A,r,B) and (B,r,A) — the same
    edge traversed from either side — key identically; so do (A,r1,B) and
    (B,r2,A) for a folded inverse pair r1/r2."""
    return (frozenset((normalize(h), normalize(t))), inv.get(r, r))


def layer_evidence(fid, ctx, paths, all_triples, centers, entities):
    """EVIDENCE SEMANTIC LAYERS (user ruling 2026-08-24, V4 / Codex synthesis):
    the harness — not the model — knows which slot of the fact is the ANSWER
    role (the plan's ?var side in fact_edges). Partition THIS call's evidence
    into answer-role / attribute-values / source layers so the model reads
    "what satisfies the fact" before the graph soup. Evidence-derived only:
    no decision-making, bindings stay the model's checkpoint job.
    Returns dict of display lines (empty strings when not derivable)."""
    from kgqa.core.utils import normalize as _nz

    def _disp_name(n):
        return str(n).split(":", 1)[0].strip()

    fk = None
    for cand in (fid, str(fid).lower()):
        if cand in (getattr(ctx, "fact_key_map", None) or {}):
            fk = ctx.fact_key_map[cand]
            break
    if fk is None or fk not in edges:
        # model omitted `sg:` (the placeholder family): fall back to the
        # first OPEN fact — display-layer heuristic, same shape as the purity
        # detector's open-fact pick
        _taken = set(getattr(ctx, "declared_facts", None) or {}) | \
            set(getattr(ctx, "closed_facts", None) or {})
        for cand in reversed(getattr(ctx, "fact_ids", None) or []):
            if cand not in _taken:
                fk = cand
                break
        if fk is None:
            fids_l = getattr(ctx, "fact_ids", None) or []
            fk = fids_l[0] if fids_l else None
    edges = getattr(ctx, "fact_edges", None) or {}
    ftext = (getattr(ctx, "fact_texts", None) or {}).get(fk or fid, "")
    head_t, tail_t = (edges.get(fk) or (None, None)) if fk else (None, None)
    # answer side: the fact's ?var slot (usually tail); head-var facts answer
    # at the anchor side
    if tail_t and str(tail_t).startswith("?"):
        var = str(tail_t)
    elif head_t and str(head_t).startswith("?"):
        var = str(head_t)
    else:
        var = None

    center_names = [e for e in (entities or []) if not str(e).startswith("?")]
    term_by_rel = {}
    for p in paths or []:
        nodes = p.get("nodes") or []
        rels = p.get("relations") or []
        if not nodes or not rels:
            continue
        n = _disp_name(nodes[-1])
        if n and not is_cvt_like(n) and _nz(n) not in {_nz(c) for c in center_names}:
            term_by_rel.setdefault(_short_rel(rels[-1]), []).append(n)
    answer = list(dict.fromkeys(t for ts in term_by_rel.values() for t in ts))
    attrs = []
    for tr in (all_triples or []):
        if isinstance(tr, (tuple, list)) and len(tr) == 3:
            h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
            if is_cvt_like(h) and not is_cvt_like(t) and t not in attrs:
                attrs.append(t)
    attrs = attrs[:40]
    # head-var facts: the anchor entities ARE the answer-role instances
    answer_role = answer if (var and str(tail_t or "").startswith("?")) else (
        center_names if (var and str(head_t or "").startswith("?")) else [])
    mids = []
    for p in paths or []:
        for n in (p.get("nodes") or [])[1:-1]:
            n = _disp_name(n)
            if n and not is_cvt_like(n) and n not in answer and n not in mids:
                mids.append(n)
    return {
        "fact": ftext,
        "var": var,
        "answer": answer_role[:60],
        "by_rel": [(rel, list(dict.fromkeys(ts))[:60])
                   for rel, ts in term_by_rel.items()],
        "n_answer": len(answer_role),
        "attrs": attrs,
        "source": (center_names + mids)[:15],
    }


def render_pattern_chains(paths, selected_rels=None, max_shapes=6,
                          lines_per_shape=30):
    """MULTI-HOP PATTERN EVIDENCE (SUPERSEDED in production by
    render_evidence_sections, 2026-08-25; kept as the A/B before-arm for
    scripts/render_ab_report.py — do not wire back into _sg_finalize).
    User rulings 2026-08-24/25:
    ① SELECTION DISCIPLINE — a shape renders only when EVERY hop's relation
    is in the model's selected set (no extra unselected-relation edges swept
    in by the walk's enumeration); facts section filters likewise.
    ② INSTANCE COHESION — one line per instance chain (anchor --r1--> n1
    --r2--> n2), instances sharing all-but-the-terminal node fold their
    tails with ' | '. The old (h,r) fan-out grouping mixed edges from
    different instances under one shape and broke path continuity
    (Fitzgerald specimen). Global (h,r,t) dedup + CVT back-edge suppression
    retained; attrs single-print."""
    from kgqa.core.utils import normalize as _nz

    def _disp(n):
        s = str(n)
        if ": [" in s and s.endswith("]"):
            i, attrs = s.split(": [", 1)
            attrs = "; ".join(a.strip() for a in attrs[:-1].split(", ") if a.strip())
            return f"{i} [{attrs}]" if attrs else i
        return s

    printed = set()

    def _once(n):
        s = _disp(n)
        rid = s.split(" [", 1)[0]
        if " [" in s and rid in printed:
            return rid
        if " [" in s:
            printed.add(rid)
        return s

    sel = {_short_rel(r) for r in (selected_rels or []) if r}
    shapes = {}
    for p in paths or []:
        rels = p.get("relations") or []
        nodes = p.get("nodes") or []
        shorts = [_short_rel(r) for r in rels]
        if len(shorts) < 2 or len(nodes) < 2:
            continue
        if sel and not all(s in sel for s in shorts):
            continue          # selection discipline: no unselected hops
        shapes.setdefault(tuple(shorts), []).append(nodes)
    out, covered = [], set()

    def _edge_key(a, r, b):
        return (_nz(str(a).split(":", 1)[0].strip()), r,
                _nz(str(b).split(":", 1)[0].strip()))

    ordered = sorted(shapes.items(), key=lambda kv: -len(kv[1]))[:max_shapes]
    for key, insts in ordered:
        # fold instances sharing all-but-terminal nodes: one line, tails '|'
        groups, order = {}, []
        for nodes in insts:
            gk = tuple(nodes[:-1])
            if gk not in groups:
                groups[gk] = []
                order.append(gk)
            groups[gk].append(nodes[-1])
        sh_lines = []
        for gk in order:
            prefix, tails0 = gk, groups[gk]
            tails = list(dict.fromkeys(str(t).split(":", 1)[0].strip()
                                       for t in tails0))
            if len(prefix) != len(key):
                continue          # malformed instance — conservative skip
            # backtrack guard on the prefix edges (same-rel swapped ends)
            pre_eks = [_edge_key(prefix[k], key[k], prefix[k + 1])
                       for k in range(len(key) - 1)]
            if any((ek[2], ek[1], ek[0]) in covered for ek in pre_eks):
                continue
            line = _once(prefix[0])
            for k in range(len(key) - 1):
                covered.add(pre_eks[k])
                line += f" --{key[k]}--> {_once(prefix[k + 1])}"
            shown = " | ".join(_disp(t) for t in tails[:120])
            more = f" …(+{len(tails)} instances)" if len(tails) > 120 else ""
            for t0 in tails0:
                covered.add(_edge_key(prefix[-1], key[-1], t0))
            line += f" --{key[-1]}--> {shown}{more}"
            sh_lines.append(line)
            if len(sh_lines) >= lines_per_shape:
                break
        if not sh_lines:
            continue
        out.append(f"[{' --'.join(key)}]  ({len(insts)} instances)")
        out.extend(f"  {l}" for l in sh_lines)
    return out, covered


_SHAPE_INSTANCE_CAP = 3000   # complete instances enumerated per shape; overflow is
                              # explicitly marked (I5: no silent truncation)


def _select_uncovered(all_triples, covered, sel_rels):
    """FACTS INPUT SELECTION for _render_records (user rulings 2026-08-20/25).

    ① Selection discipline (ruling ⑧): only edges whose relation ∈ the
    selected set (empty selection = no filter; matching on FULL dotted
    names — covered keys and the selected set share the full-name form) and
    not consumed by a structured block (`covered`).
    ② WITHIN-CALL dedup only: duplicates arriving via multiple patterns/
    channels collapse by exact (h, r, t) key — cross-call suppression is
    FORBIDDEN (each subgraph re-renders the full evidence its pattern paths
    justify); the old `(all N already shown in a prior subgraph)` message
    counted these within-call duplicates on a FRESH shown-set and lied about
    a "prior subgraph" (Kim Richards specimen, first retrieval of the case).
    ③ CVT ATTR REVIVAL (ruling ④, the specimen's real failure): when a CVT
    is displayed via a selected-rel edge (e.g. `Show --regular_cast--> m.x`),
    its attribute edges (unselected relation names like actor/series) are
    fed to the record renderer as PAYLOAD — _render_records only ever shows
    them inside the inline `[k=v; …]` bracket, never as standalone edge
    lines, so the selection discipline's letter (no unselected relation
    LINES) holds while the record's content stays visible. Without this the
    record renders as a bare CVT tail and the legacy BARE-CVT TAIL DROP
    discards it — an empty evidence section.
    """
    from kgqa.core.utils import normalize as _nz
    sel = set(sel_rels or [])
    seen = set()
    out = []
    for tr in all_triples or []:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            continue
        r = str(tr[1])
        if sel and r not in sel:
            continue
        k = (_nz(str(tr[0])), r, _nz(str(tr[2])))
        if k in covered or k in seen:
            continue
        seen.add(k)
        out.append(tr)
    # ③ revive attr edges of CVTs that the facts section will display
    need = set()
    for h, _r, t in out:
        if is_cvt_like(str(h)):
            need.add(_nz(str(h)))
        if is_cvt_like(str(t)):
            need.add(_nz(str(t)))
    if need:
        for tr in all_triples or []:
            if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
                continue
            h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
            if not is_cvt_like(h) or is_cvt_like(t) \
                    or _short_rel(r) in _EDGE_NOISY_SHORT:
                continue
            k = (_nz(h), r, _nz(t))
            if k in covered or k in seen or _nz(h) not in need:
                continue
            seen.add(k)
            out.append(tr)
    return out


_FACT_REL_RE = re.compile(r"--<?\s*([A-Za-z0-9_.]+)\s*>?--")


def _group_facts_by_rel(lines):
    """FACTS BY RELATION (user ruling 2026-08-25, Debussy specimen: "the
    facts marker has weak semantics"): after the selected-relation sweep the
    facts section holds only true remainders (record renderings, revived
    payload, residual edges) — group them under per-relation headers so no
    unorganized pipe list remains. Lines without a relation token (record
    group headers, (all:) uplifts, indented sub-entries) inherit the
    PRECEDING line's relation; a single group keeps its lines unchanged
    under one header."""
    groups, order, cur = {}, [], None
    for ln in lines:
        m = _FACT_REL_RE.search(ln)
        rel = m.group(1) if m else cur
        if rel is None:
            rel = "records"
        cur = rel
        if rel not in groups:
            groups[rel] = []
            order.append(rel)
        groups[rel].append(ln)
    out = []
    for rel in order:
        out.append(f"  {rel}:")
        out.extend(groups[rel])
    return out


def render_evidence_sections(paths, centers, var_label, all_triples, selected_rels,
                             per_rel=3, total_cap=12, lines_per_shape=30):
    """UNIFIED EVIDENCE RENDERER (user ruling 2026-08-25: audit the mechanism,
    then redesign once). Supersedes build_pattern_overview + render_pattern_chains
    in production (both kept below for the A/B before-arm).

    PRINCIPLE 1 (per selected relation, short-to-long): ① one 1-hop block per
    selected relation FIRST (anchor-anchored dense line, synthesized from the
    anchor's own selected-relation edges even when the walk enumerated no
    1-hop pattern), then ② k-hop path blocks, length ASCENDING, within one
    length by instance count desc. The overview is the shape INDEX of exactly
    the blocks that render, in the same order (概览与细节同源: same instance
    sets, consistent counts).

    PRINCIPLE 2 (path consistency): a k-hop block contains ONLY instances that
    COMPLETE the block's path shape — every row traces back to a full
    anchor→terminal chain. Instances are RECONSTRUCTED from the canonicalized
    evidence edges (tree paths are bounded witnesses; the flattened edge set
    carries the walk's full leaf enumeration), so a hop-2 continuation edge of
    a non-witness hop-1 entity lands INSIDE the 2-hop block behind its hop-1
    anchor — never in facts as an anchorless fragment. An entity reached by
    hop 1 with no hop-2 continuation stays in the 1-hop block only.

    Standing rulings kept: selection discipline (every hop ∈ selected set;
    unselected relations reach neither blocks nor facts), start anchoring +
    ?slots + `<--r--` reverse notation in the overview, dense line grammar +
    CVT attrs inline + (all:) uplift + 120 cap + branch refs, CVT back-edge
    suppression (canonical-direction reconstruction: a same-rel swapped-end
    backtrack cannot re-enter as a forward hop), attr single-print, shared-
    prefix tail folding with tail dedup, separator hierarchy ('|' = hop/tail
    axis, never inside a cell).
    Returns (overview_lines, block_lines, covered_edge_keys)."""
    from kgqa.core.utils import normalize as _nz

    def _node_key(n):
        # tree paths carry expanded display names ("m.xxx: [attrs]") — keys use
        # the id/name before the colon
        return _nz(str(n).split(":", 1)[0].strip())

    sel_full = [str(r) for r in (selected_rels or []) if r]
    sel = set(sel_full)
    centers = list(centers or [])
    center_set = {_nz(str(c)) for c in centers if _nz(str(c))}

    # SHORT-NAME COLLISION DISAMBIGUATION (user ruling 2026-08-25, Debussy
    # specimen): two selected FULL relations may share a short name
    # (music.composition.composer vs base.musiteca.composition.composer) —
    # short-name display would merge them indistinguishably. All INTERNAL
    # keys (indexes, shape hops, selection discipline, coverage) use the FULL
    # dotted name; only DISPLAY tokens shorten, and a short shared by ≥2
    # selected relations renders as `short[family]` (family = first segment).
    from collections import Counter as _Ctr
    _short_n = _Ctr(_short_rel(r) for r in sel_full)

    def _disp_rel(rfull):
        s = _short_rel(rfull)
        if _short_n.get(s, 0) > 1:
            return f"{s}[{str(rfull).split('.', 1)[0]}]"
        return s

    # ── canonical edge indexes (graph direction) + display names + CVT attrs ──
    # NOTE on CVT detection: normalize() maps '_'→' ' ("m.02sd_zh"→"m.02sd zh")
    # which breaks is_cvt_like on NORMALIZED keys — collect cvt_keys from the
    # RAW names instead and test membership.
    fwd_idx, rev_idx, disp, cvt_attrs = {}, {}, {}, {}
    by_rel = {}      # full rel -> [(hn, tn)] — the selected-relation sweep
    cvt_keys = set()
    edge_keys = set()

    def _is_cvt_key(nk):
        return nk in cvt_keys

    for tr in all_triples or []:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            continue
        h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
        if _short_rel(r) in _EDGE_NOISY_SHORT:
            continue                       # bookkeeping noise (mirror facts filter)
        hn, tn = _nz(h), _nz(t)
        if not hn or not tn or hn == tn:
            continue                       # empty / self-loop
        edge_keys.add((hn, r, tn))
        disp.setdefault(hn, h)
        disp.setdefault(tn, t)
        if tn not in fwd_idx.setdefault((hn, r), []):
            fwd_idx[(hn, r)].append(tn)
        if hn not in rev_idx.setdefault((tn, r), []):
            rev_idx[(tn, r)].append(hn)
        by_rel.setdefault(r, []).append((hn, tn))
        if is_cvt_like(h):
            cvt_keys.add(hn)
        if is_cvt_like(t):
            cvt_keys.add(tn)
        if is_cvt_like(h) and not is_cvt_like(t):
            k, v = (r, t)
            lst = cvt_attrs.setdefault(hn, [])
            if (k, _nz(v)) not in [(a, _nz(w)) for a, w in lst]:
                lst.append((k, v))

    # ── shapes from tree paths (the walk's pattern vocabulary; hop direction
    #    resolved against the canonical edges) ─────────────────────────────────
    # SHAPE KEY = the hop sequence ALONE (user audit 2026-08-26, Randy Jackson
    # specimen): keying on (hops, term_cvt) split one shape into two entries
    # whose instance reconstruction (edge-based, witness-independent) was
    # IDENTICAL — the same block rendered twice verbatim. Terminal-CVT-ness
    # is a DISPLAY decision computed from the reconstructed instances.
    shapes = {}   # (hops) -> {hops, anchors, order}
    for p in paths or []:
        rels = p.get("relations") or []
        nodes = p.get("nodes") or []
        if not rels or len(nodes) < 2 or len(nodes) > len(rels) + 1:
            continue
        # NO-FREE-HEAD GATE (user calibration 2026-08-26, Poetic Justice /
        # Vocals / Sonora specimens): a tree path may anchor a shape only
        # when its FIRST node is a CENTER. The sibling-CVT display mirror
        # (formatting.py) roots continuation paths at the path's MID node,
        # and those previously registered as non-center 1-hop shapes that
        # rendered as free-head rows detached from the chain that produced
        # them (`Vocals --instrumentalists--> ...` while the center's own
        # connection lived in the chain). The same edges stay visible INSIDE
        # the center-rooted chain shapes; truly centerless selected-relation
        # edges fall to the terminal-anchored sweep below (Debussy ruling).
        a0 = _node_key(nodes[0])
        if not a0 or a0 not in center_set:
            continue
        # LAST-HOP discipline replaces whole-path selection discipline (user
        # calibration 2026-08-26, "绕道链归各自真实锚的形状或长度正确的多跳
        # 形状"): a detour chain whose bridge hops are unselected relations is
        # the REAL provenance of its last-hop selected edge — rendering it as
        # a correctly-lengthed multi-hop chain row is the only form in which
        # that edge stays attached to its center. An unselected LAST hop
        # still never renders (ruling ⑧ core); detour chains order AFTER
        # all-selected shapes below.
        if sel and str(rels[-1]) not in sel:
            continue
        hops = []
        for k in range(len(nodes) - 1):
            rf = str(rels[k])
            a, b = _node_key(nodes[k]), _node_key(nodes[k + 1])
            rev = (a, rf, b) not in edge_keys and (b, rf, a) in edge_keys
            hops.append((rf, rev))
        key = tuple(hops)
        ent = shapes.setdefault(key, {"hops": hops,
                                      "anchors": [], "seen": set(),
                                      "order": len(shapes)})
        if a0 not in ent["seen"]:
            ent["seen"].add(a0)
            ent["anchors"].append(a0)

    # 1-hop synthesis (principle 1): every selected relation gets its
    # anchor-anchored 1-hop block whenever the anchor itself has such edges —
    # anchor-headed selected edges always live in tier-1 blocks, never
    # scattered through facts.
    center_keys = [_nz(str(c)) for c in centers if _nz(str(c))]
    for ri, rf in enumerate(dict.fromkeys(sel_full)):
        for rev in (False, True):
            tails = []
            for ck in center_keys:
                tails.extend(rev_idx.get((ck, rf), []) if rev
                             else fwd_idx.get((ck, rf), []))
            if not tails:
                continue
            hops_1 = [(rf, rev)]
            key = tuple(hops_1)
            ent = shapes.setdefault(key, {"hops": hops_1,
                                          "anchors": [], "seen": set(),
                                          "order": len(shapes) + ri})
            for ck in center_keys:
                if ck not in ent["seen"]:
                    ent["seen"].add(ck)
                    ent["anchors"].append(ck)

    # ── instance reconstruction per shape (principle 2: complete paths only) ──
    for ent in shapes.values():
        prefixes = [(a,) for a in ent["anchors"]]
        overflow = False
        for sr, rev in ent["hops"]:
            nxt, seenp = [], set()
            for pfx in prefixes:
                nbrs = rev_idx.get((pfx[-1], sr), ()) if rev \
                    else fwd_idx.get((pfx[-1], sr), ())
                for n in nbrs:
                    npfx = pfx + (n,)
                    if npfx not in seenp:
                        seenp.add(npfx)
                        nxt.append(npfx)
            if len(nxt) > _SHAPE_INSTANCE_CAP:
                nxt = nxt[:_SHAPE_INSTANCE_CAP]
                overflow = True
            prefixes = nxt
            if not prefixes:
                break
        ent["instances"] = prefixes
        ent["overflow"] = overflow

    # ── ordering (principle 1): 1-hop per selected relation first (selected
    #    order, fwd before rev), then length ascending / instance count desc.
    #    Detour chains (unselected bridge hops, admitted by the last-hop
    #    discipline above) order AFTER all-selected shapes of any length —
    #    the direct selected-relation evidence always reads first. ──────────
    sel_order = {r: i for i, r in enumerate(dict.fromkeys(sel_full))}
    enriched = []
    for key, ent in shapes.items():
        n = len(ent["instances"])
        if n == 0:
            continue                       # no complete instance → no block
        hops = ent["hops"]
        if len(hops) == 1:
            sort_k = (0, sel_order.get(hops[0][0], len(sel_order)),
                      1 if hops[0][1] else 0, ent["order"])
        else:
            all_sel = all(h[0] in sel for h in hops) if sel else True
            sort_k = (0 if all_sel else 1, len(hops), -n, ent["order"])
        enriched.append((sort_k, key, ent))
    enriched.sort(key=lambda x: x[0])
    per_first, kept = {}, []
    for _sk, key, ent in enriched:
        fr = key[0][0]
        per_first[fr] = per_first.get(fr, 0) + 1
        if per_first[fr] > per_rel or len(kept) >= total_cap:
            continue
        kept.append((key, ent))

    # ── CHAIN-BEATS-ORPHAN PROMOTION (user calibration 2026-08-26, VOG /
    #    Lost-Symbol specimens): a selected-relation edge whose endpoints are
    #    both non-centers must surface as a COMPLETE chain row whenever a
    #    center-rooted shape instance carries it. The per_rel/total_cap cuts
    #    could drop exactly that shape, and the completeness sweep would then
    #    re-render the edge as a detached terminal-anchored row — the
    #    free-head form the audit banned. Promote (bounded) capped-out shapes
    #    whose instances would otherwise land in the sweep; only truly
    #    orphan edges (no covering shape at all — the Debussy family) keep
    #    the terminal-anchored sweep form. ─────────────────────────────────
    def _inst_edges(ent):
        out = set()
        for inst in ent["instances"]:
            for k, (sr, rv) in enumerate(ent["hops"]):
                out.add((inst[k + 1], sr, inst[k]) if rv
                        else (inst[k], sr, inst[k + 1]))
        return out

    kept_keys = {key for key, _ent in kept}
    kept_edges = set()
    for _k, ent in kept:
        kept_edges |= _inst_edges(ent)

    def _sweep_leftovers(edges):
        out = []
        for rf in dict.fromkeys(sel_full):
            for hn, tn in by_rel.get(rf, ()):
                if ((hn, rf, tn) in edges or hn in center_set or tn in center_set
                        or _is_cvt_key(hn) or _is_cvt_key(tn)):
                    continue
                out.append((hn, rf, tn))
        return out

    for _promo_round in range(2):
        leftovers = _sweep_leftovers(kept_edges)
        if not leftovers:
            break
        promoted = 0
        for _sk, key, ent in enriched:
            if key in kept_keys or promoted >= 4:
                continue
            ie = _inst_edges(ent)
            if ie & set(leftovers):
                kept.append((key, ent))
                kept_keys.add(key)
                kept_edges |= ie
                promoted += 1
                if not _sweep_leftovers(kept_edges):
                    break
        if not promoted:
            break

    # ── display helpers ───────────────────────────────────────────────────────
    printed = set()   # CVT keys whose attr bracket already printed (ruling ⑦)

    def _disp_node(nk):
        base = disp.get(nk, nk)
        if not _is_cvt_key(nk) or nk in printed:
            return base
        printed.add(nk)
        attrs = cvt_attrs.get(nk, [])[:4]
        if attrs:
            # attr KEYS display short (payload, not a selected-relation arrow
            # — no disambiguation needed; internal keys stay full for coverage)
            return f"{base} [" + "; ".join(f"{_short_rel(a)}={v}" for a, v in attrs) + "]"
        return base

    def _cover_cvt(nk, covered):
        # a displayed CVT's attr edges are consumed by its node display
        for a, v in cvt_attrs.get(nk, []):
            covered.add((nk, a, _nz(v)))

    def _tails_display(tails):
        """CVT tail list with uniform-attr uplift (ruling ③): an attr that is
        present in EVERY tail with the SAME SINGLE value hoists to one
        '(all: k=v)' prefix; multi-valued / non-uniform attrs stay in the
        per-record brackets (no last-wins collapse — values must survive)."""
        attrs = [cvt_attrs.get(t, [])[:4] for t in tails]
        if len(attrs) >= 2 and all(attrs):
            uniform = {}
            for k in set(attrs[0]):
                vs = [[v for a, v in lst if a == k] for lst in attrs]
                if all(len(v) == 1 for v in vs) and len({v[0] for v in vs}) == 1:
                    uniform[k] = vs[0][0]
        else:
            uniform = {}
        hoist = ""
        if uniform:
            hoist = "(all: " + "; ".join(f"{_short_rel(k)}={str(v)[:24]}"
                                        for k, v in uniform.items()) + ") "
        outs = []
        for t, lst in zip(tails, attrs):
            base = disp.get(t, t)
            if not _is_cvt_key(t) or t in printed:
                outs.append(base)          # bare id — attr payload printed earlier
                continue
            kept = [(a, v) for a, v in lst if a not in uniform]
            outs.append(base + (" [" + "; ".join(f"{_short_rel(a)}={v}" for a, v in kept) + "]"
                                if kept else ""))
            printed.add(t)
        return hoist, outs

    covered = set()

    def _mark_hop(a, sr, b, rev):
        covered.add((b, sr, a) if rev else (a, sr, b))

    def _center_first(seq):
        """④③ CENTER-PRIORITY TRUNCATION (user audit 2026-08-26, Randy
        Jackson specimen): when a fold truncates a member/tail list, CENTER
        entities survive first — the center's own presence inside a shared
        list is exactly the evidence a fold must never cut (the center was
        dropped from a 6-member instrumentalists row, and with him the gold
        binding). Stable sort: non-centers keep their arrival order."""
        return sorted(seq, key=lambda x: _nz(str(x)) not in center_set)

    def _anchor_head(ent):
        """②a SAME-ANCHOR ATTRIBUTION (user audit 2026-08-26, ABM specimen):
        the overview head is the shape's REAL first-node population — the
        declared variable label when there is one (a variable head honestly
        stands for all its bindings), else the distinct anchors folded in
        the dense multi-head grammar. centers[0] is never borrowed for
        instances it does not anchor: `A Beautiful Mind --subjects--> ?x
        (4 instances)` previously claimed four instances anchored at OTHER
        films while ABM itself had none."""
        heads = []
        for inst in ent["instances"]:
            if inst[0] not in heads:
                heads.append(inst[0])
        if var_label:
            return str(var_label)
        if len(heads) == 1:
            return disp.get(heads[0], heads[0])
        shown = " | ".join(disp.get(h, h) for h in heads[:4])
        if len(heads) > 4:
            shown += f" | …(+{len(heads) - 4} centers)"
        return shown

    # ── render: overview lines + detail blocks, same order, same counts ──────
    ov_lines, block_lines = [], []
    tier1_end = 0        # splice point for the selected-relation sibling sweep
    for key, ent in kept:
        hops = key
        instances = ent["instances"]

        # overview line (ruling ① grammar; counts = block instance counts);
        # head = the shape's real anchor population (see _anchor_head)
        line = _anchor_head(ent)
        slot = ord("x")
        # terminal-CVT notation is a display decision over the RECONSTRUCTED
        # instances (all-terminal-CVT → `[N records]`), not a witness property
        term_cvt = bool(instances) and all(_is_cvt_key(i[-1]) for i in instances)
        for k, (rf, rev) in enumerate(hops):
            last = k == len(hops) - 1
            if last and term_cvt:
                nxt = f"[{len(instances)} records]"
            else:
                nxt = f"?{chr(slot)}"
                slot += 1
            line += f" <--{_disp_rel(rf)}-- {nxt}" if rev \
                else f" --{_disp_rel(rf)}--> {nxt}"
        if not term_cvt and len(instances) > 1:
            line += f"  ({len(instances)} instances)"
        ov_lines.append(line)

        # detail block header (multi-hop only — a 1-hop dense line names its
        # own shape; header = relation sequence, legacy grammar). Direction
        # is part of the shape: two shapes differing only in a hop's
        # direction must not render identical headers (reads as a duplicate
        # block with inconsistent counts — ④'s twin-block report). Reverse
        # hops carry the `<--` prefix, mirroring the dense-line grammar.
        if len(hops) > 1:
            block_lines.append("[{}]  ({} instances)".format(
                " --".join(_disp_rel(rf) if not rv else f"<--{_disp_rel(rf)}"
                           for rf, rv in hops), len(instances)))

        if len(hops) == 1:
            rf, rev = hops[0]
            # fold members sharing a tail-set (dense grammar, ruling ③/⑨)
            by_ts, order = {}, []
            for inst in instances:
                a, t = inst[0], inst[1]
                if a not in by_ts:
                    by_ts[a] = []
                    order.append(a)
                if t not in by_ts[a]:
                    by_ts[a].append(t)
            ts_group, ts_order = {}, []
            for a in order:
                gk = tuple(by_ts[a])
                if gk not in ts_group:
                    ts_group[gk] = []
                    ts_order.append(gk)
                ts_group[gk].append(a)
            n_lines = 0
            for gk in ts_order:
                if n_lines >= lines_per_shape:
                    break
                members = _center_first(ts_group[gk])
                hoist, tdisp = _tails_display(_center_first(list(gk)))
                m_shown = " | ".join(disp.get(m, m) for m in members[:_MERGE_TAIL_CAP])
                if len(members) > _MERGE_TAIL_CAP:
                    m_shown += f" +{len(members) - _MERGE_TAIL_CAP} more"
                t_list = " | ".join(tdisp[:_MERGE_TAIL_CAP])
                if len(tdisp) > _MERGE_TAIL_CAP:
                    extra = len(tdisp) - _MERGE_TAIL_CAP
                    t_list += (f" +{extra} more (total {len(tdisp)}). To answer with "
                               f"ALL of them, include \"#{disp.get(members[0], members[0])}::{_short_rel(rf)}\" "
                               f"as one answer entity — the system expands the ref to "
                               f"the full list.")
                arrow = f" <--{_disp_rel(rf)}-- " if rev else f" --{_disp_rel(rf)}--> "
                block_lines.append(f"  {m_shown}{arrow}{hoist}{t_list}")
                n_lines += 1
                for m in members:
                    for t in gk:
                        _mark_hop(m, rf, t, rev)
                        if _is_cvt_key(t):
                            _cover_cvt(t, covered)
            tier1_end = len(block_lines)
            continue

        # multi-hop: shared-prefix tail folding, one line per prefix (ruling ⑨);
        # additionally, prefixes sharing the SAME tail-set whose nodes differ
        # in exactly ONE position merge into a single fanout line — the
        # _merge_edges (relation, tail-set) semantics carried to chains (the
        # round-trip family 'A --r--> X1..Xn --r2--> A' collapses to one line)
        groups, order = {}, []
        for inst in instances:
            gk = inst[:-1]
            if gk not in groups:
                groups[gk] = []
                order.append(gk)
            if inst[-1] not in groups[gk]:
                groups[gk].append(inst[-1])
        by_ts, ts_order = {}, []
        for gk in order:
            ts = tuple(groups[gk])
            if ts not in by_ts:
                by_ts[ts] = []
                ts_order.append(ts)
            by_ts[ts].append(gk)
        entries = []   # (rep_prefix, vary_pos|None, vary_ents, tails)
        for ts in ts_order:
            gks = by_ts[ts]
            if len(gks) == 1:
                entries.append((gks[0], None, [], list(ts)))
                continue
            vary = [i for i in range(len(gks[0]))
                    if len({g[i] for g in gks}) > 1]
            if len(vary) == 1:
                ents = list(dict.fromkeys(g[vary[0]] for g in gks))
                entries.append((gks[0], vary[0], ents, list(ts)))
            else:
                for g in gks:
                    entries.append((g, None, [], list(ts)))
        n_shown_inst = 0
        rendered = 0
        # FRONTIER MEMBER CAP (user audit 2026-09-17, 567_df97 specimen: a
        # story_by layer instantiated 8 films' writer blocks — structurally
        # unnecessary bulk). K>0 caps the DISTINCT prefix members RENDERED
        # per multi-hop shape; 0 (default) keeps the full render. Display-
        # only: the _mark_hop/_cover_cvt loop still runs for every entry,
        # so licensing and CVT coverage are unchanged.
        _fk = int(os.environ.get("SEQ_FRONTIER_RENDER_K", "0") or 0)
        _fk_hidden = 0
        for _ei, (rep, vpos, vents, tails) in enumerate(entries):
            if _fk and _ei >= _fk:
                _fk_hidden += 1
                tails = _center_first(tails)
                if vpos is not None:
                    vents = _center_first(vents)
                prefixes = ([tuple(x if i == vpos else p for i, p in enumerate(rep))
                             for x in vents] if vpos is not None else [rep])
                for pfx in prefixes:
                    for k in range(len(hops) - 1):
                        rf, rev = hops[k]
                        _mark_hop(pfx[k], rf, pfx[k + 1], rev)
                    lrf, lrev = hops[-1]
                    for t in tails:
                        _mark_hop(pfx[-1], lrf, t, lrev)
                        if _is_cvt_key(t):
                            _cover_cvt(t, covered)
                    for m in pfx[1:]:
                        if _is_cvt_key(m):
                            _cover_cvt(m, covered)
                continue
            if rendered >= lines_per_shape:
                break
            tails = _center_first(tails)
            if vpos is not None:
                vents = _center_first(vents)
            tdisp = [_disp_node(t) for t in tails[:_MERGE_TAIL_CAP]]
            shown = " | ".join(tdisp)
            if len(tails) > _MERGE_TAIL_CAP:
                shown += f" …(+{len(tails)} instances)"
            segs = []
            for k in range(len(hops)):           # prefix node positions
                if k == 0:
                    segs.append(" | ".join(_disp_node(x) for x in
                                           (vents if vpos == 0 else [rep[0]])))
                    continue
                rf, rev = hops[k - 1]
                nodes = (vents if vpos == k else [rep[k]])
                segs.append((f" <--{_disp_rel(rf)}-- " if rev else f" --{_disp_rel(rf)}--> ")
                            + " | ".join(_disp_node(x) for x in nodes))
            lrf, lrev = hops[-1]
            segs.append((f" <--{_disp_rel(lrf)}-- " if lrev else f" --{_disp_rel(lrf)}--> ")
                        + shown)
            block_lines.append("  " + "".join(segs))
            rendered += 1
            n_shown_inst += (len(vents) if vpos is not None else 1) * len(tails)
            # cover the instance edges of every folded prefix/tail (incl.
            # beyond-cap — the cap marker owns the disclosure)
            prefixes = ([tuple(x if i == vpos else p for i, p in enumerate(rep))
                         for x in vents] if vpos is not None else [rep])
            for pfx in prefixes:
                for k in range(len(hops) - 1):
                    rf, rev = hops[k]
                    _mark_hop(pfx[k], rf, pfx[k + 1], rev)
                lrf, lrev = hops[-1]
                for t in tails:
                    _mark_hop(pfx[-1], lrf, t, lrev)
                    if _is_cvt_key(t):
                        _cover_cvt(t, covered)
                for m in pfx[1:]:
                    if _is_cvt_key(m):
                        _cover_cvt(m, covered)
        if _fk_hidden:
            block_lines.append(
                f"  …(+{_fk_hidden} members follow the same relation — not shown; "
                f"discriminate from the shown ones or retrieve their edges "
                f"explicitly)")
        rest = len(instances) - n_shown_inst
        if rendered < len(entries) and (rest > 0 or ent.get("overflow")):
            suffix = "+" if ent.get("overflow") else ""
            block_lines.append(f"  …(+{rest}{suffix} more instances)")

    # ── SELECTED-RELATION COMPLETENESS SWEEP (user ruling 2026-08-25, Debussy
    #    specimen): a selected relation's instances may be sibling edges whose
    #    NEITHER endpoint is a center (works --author--> Claude Debussy while
    #    the center is one particular work) — previously they all fell into
    #    facts as one unlabeled merged line. Each selected relation must have
    #    a STRUCTURAL presentation: group the leftovers by shared terminal and
    #    render terminal-anchored reverse dense lines, spliced right after the
    #    tier-1 blocks (same 1-hop grammar, no overview line — the overview
    #    indexes center-anchored shapes only, ruling ①). Selected-relation
    #    edges never reach facts unsemantically.
    sib_lines = []
    for rf in dict.fromkeys(sel_full):
        groups, g_order = {}, []
        for hn, tn in by_rel.get(rf, ()):
            if ((hn, rf, tn) in covered or hn in center_set or tn in center_set
                    or _is_cvt_key(hn) or _is_cvt_key(tn)):
                continue
            if tn not in groups:
                groups[tn] = []
                g_order.append(tn)
            if hn not in groups[tn]:
                groups[tn].append(hn)
        n_ln = 0
        for tn in g_order:
            if n_ln >= lines_per_shape:
                remaining = sum(len(groups[x]) for x in g_order[n_ln:])
                sib_lines.append(f"  …(+{remaining} more via {_disp_rel(rf)})")
                break
            heads = _center_first(groups[tn])
            h_list = " | ".join(disp.get(h, h) for h in heads[:_MERGE_TAIL_CAP])
            if len(heads) > _MERGE_TAIL_CAP:
                extra = len(heads) - _MERGE_TAIL_CAP
                h_list += (f" +{extra} more (total {len(heads)}). To answer with "
                           f"ALL of them, include \"#{disp.get(tn, tn)}::{_short_rel(rf)}\" "
                           f"as one answer entity — the system expands the ref to "
                           f"the full list.")
            sib_lines.append(f"  {disp.get(tn, tn)} <--{_disp_rel(rf)}-- {h_list}")
            n_ln += 1
            for hn in heads:
                covered.add((hn, rf, tn))
    if sib_lines:
        block_lines[tier1_end:tier1_end] = sib_lines
    # EXPLICIT ZERO-COVERAGE (user ruling 2026-08-25, Kim Richards specimen):
    # a selected relation with NO instance anywhere in the evidence must SAY
    # so — silence lets the model conclude "no instance exists" from absence.
    # (Instance may live in the remainder section rather than blocks — check
    # all_triples.) ALL-∅ (user ruling 2026-08-25, repair hook): when every
    # selected relation is empty the output must also carry the cross-subgraph
    # repair guidance (same family as the walk-empty RELATION_MISMATCH note)
    # — a non-empty return whose selected relations are ALL silent is exactly
    # the Kim-Richards-class miss that previously self-healed only when the
    # walk returned nothing at all.
    _present_rels = {str(tr[1]) for tr in (all_triples or [])
                     if isinstance(tr, (tuple, list)) and len(tr) == 3}
    _sel_uniq = list(dict.fromkeys(sel_full))
    for rf in _sel_uniq:
        if rf not in _present_rels:
            ov_lines.append(f"  --{_disp_rel(rf)}--> ∅ (no instances from these centers)")
    if _sel_uniq and all(rf not in _present_rels for rf in _sel_uniq):
        ov_lines.append(
            "  selected relations hit NOTHING from these centers — consider "
            "re-calling with OTHER candidate relations, or with a center "
            "BORROWED from any earlier subgraph's evidence (cross-subgraph "
            "center: any prior retrieve_subgraph's entities are legal centers).")
    return ov_lines, block_lines, covered


def build_pattern_overview(paths, centers, var_label, all_triples,
                            selected_rels, per_rel=3, total_cap=12):
    """PATTERN OVERVIEW (SUPERSEDED in production by render_evidence_sections,
    2026-08-25; kept as the A/B before-arm for scripts/render_ab_report.py).
    User ruling 2026-08-24, V3: at most `per_rel` pattern
    lines per first-hop relation — path-terminal shapes FIRST (last hop is one
    of the SELECTED relations, i.e. the walk ended by entering what the model
    asked for), then by instance count. Anchor = the walk start (single entity
    or the ?var set label); every line is start-anchored, slots are ?x/?y.
    Direction per hop is resolved against the flattened edges (reverse hops
    render `<--r--`). Terminal CVT nodes render `[records]`. Dense by design:
    this is an index, the detail below carries the payload."""
    from kgqa.core.utils import normalize as _nz

    def _node_key(n):
        # tree paths carry expanded display names ("m.xxx: [attrs]") — match
        # flattened edges on the id/name before the colon
        return _nz(str(n).split(":", 1)[0].strip())

    edge_dir = set()
    for tr in all_triples or []:
        if isinstance(tr, (tuple, list)) and len(tr) == 3:
            edge_dir.add((_nz(str(tr[0])), _short_rel(tr[1]), _nz(str(tr[2]))))

    anchor = var_label or (centers[0] if centers else "?start")
    sel_short = {_short_rel(r) for r in (selected_rels or [])}

    shapes = {}   # shape key -> [count, terminal_last, rels_line, first_rel]
    for p in paths or []:
        rels = p.get("relations") or []
        nodes = p.get("nodes") or []
        if not rels or len(nodes) < 2:
            continue
        hops, terminal_last = [], False
        for k, r in enumerate(rels):
            sr = _short_rel(r)
            a, b = _node_key(nodes[k]), _node_key(nodes[k + 1]) if k + 1 < len(nodes) else ""
            rev = (a, sr, b) not in edge_dir and (b, sr, a) in edge_dir
            hops.append((sr, rev))
        last_rel = hops[-1][0]
        if last_rel in sel_short:
            terminal_last = True
        key = (tuple(hops), str(nodes[-1]).startswith(("m.", "g.")))
        if key not in shapes:
            shapes[key] = [0, terminal_last, hops, _short_rel(rels[0])]
        shapes[key][0] += 1

    # group by first-hop relation; per group: terminal-first, then count desc
    groups = {}
    for key, (cnt, term, hops, first_rel) in shapes.items():
        groups.setdefault(first_rel, []).append((not term, -cnt, key, hops, cnt))
    out = []
    for first_rel, items in sorted(groups.items(), key=lambda kv: -sum(i[4] for i in kv[1])):
        items.sort()
        for _prio, _ncnt, key, hops, cnt in items[:per_rel]:
            if len(out) >= total_cap:
                return out
            ends_record = key[1]
            line = str(anchor)
            slot = ord("x")
            for k, (sr, rev) in enumerate(hops):
                last = k == len(hops) - 1
                if last and ends_record:
                    nxt = f"[{cnt} records]"
                else:
                    nxt = f"?{chr(slot)}"; slot += 1
                line += f" <--{sr}-- {nxt}" if rev else f" --{sr}--> {nxt}"
            if not ends_record:
                line += f"  ({cnt} instances)" if cnt > 1 else ""
            out.append(line)
    return out


def _render_records(all_triples, shown_edges):
    """Record-centric renderer (replaces _resolve_cvt_edges + _merge_edges).

    CVT-with-holder → record `holder → title [from=..; to=..; ..]`, direction-normalized:
    holder = the person reached via an `office_holder` (singular) or `positions_held` edge
    in EITHER direction; `office_holders` (plural) is the title→CVT reverse and is skipped
    (it was the source of the holder/title direction confusion). CVT-without-holder
    (non-record schemas) → flattened named→named. Direct named→named edges → merged lines.
    `has_no_value` → `to=(incumbent)`. Cross-subgraph dedup by record key or edge key.
    Returns (lines, n_overlap)."""
    cvt_attrs = {}         # cvt -> [(attr_short, val)]  (CVT→named, excl has_no_value)
    cvt_holders = {}       # cvt -> [holder name]  (either direction, holder edge)
    cvt_incumbent = set()  # cvt with a has_no_value edge
    direct = []

    def _is_holder_rel(r):
        return r.endswith("office_holder") or "positions_held" in r

    for tr in all_triples:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            continue
        h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
        sr = _short_rel(r)
        if is_cvt_like(h) and not is_cvt_like(t):
            if sr == "has_no_value" or "has_no_value" in r:
                cvt_incumbent.add(h); continue
            cvt_attrs.setdefault(h, []).append((sr, t))
            if _is_holder_rel(r):
                cvt_holders.setdefault(h, []).append(t)
        elif not is_cvt_like(h) and is_cvt_like(t):
            if _is_holder_rel(r):
                cvt_holders.setdefault(t, []).append(h)
            # title→CVT (office_holders plural) reverse: skip — holder/title surface via CVT attrs
        elif not is_cvt_like(h) and not is_cvt_like(t):
            if sr in _EDGE_NOISY_SHORT or normalize(h) == normalize(t):
                continue
            direct.append((h, sr, t))

    # build records (CVT with holder)
    records = []
    seen_rec = set()
    for cvt, holders in cvt_holders.items():
        attrs = cvt_attrs.get(cvt, [])
        title = next((v for a, v in attrs if any(k in a for k in ("basic_title", "office_position_or_title"))), None)
        from_v = next((v for a, v in attrs if a == "from"), None)
        to_v = next((v for a, v in attrs if a == "to"), None)
        if cvt in cvt_incumbent and not to_v:
            to_v = "(incumbent)"
        juris = next((v for a, v in attrs if "jurisdiction" in a), None)
        used = {title, from_v, to_v, juris}
        extra = [(a, v) for a, v in attrs
                 if v not in used and a not in ("office_holder", "office_holders",
                     "office_position_or_title", "basic_title", "from", "to")
                 and "type" not in a and "review" not in a and not a.startswith("is_")]
        for hd in holders:
            key = ("REC", normalize(hd), normalize(title or ""), from_v or "", to_v or "")
            if key in seen_rec:
                continue
            seen_rec.add(key)
            records.append((hd, title, from_v, to_v, juris, extra))

    # fallback: CVT without holder.
    # MEASUREMENT CVTs (dated_percentage/dated_integer/dated_metric_ton — carry a
    # date+rate/number pair) are kept as PRESERVED EVENT ENTITIES so the (date, value)
    # pairing survives: render as `head --rel--> rec[date=D; value=V]` instead of
    # flattening to two independent `head --date--> D` / `head --rate--> V` lists
    # (which destroys the date↔value correspondence the model needs to pick a year).
    # Other holder-less CVTs → flatten their attrs onto the named head (edge-centric).
    _MEAS_ATTRS = {"rate", "number", "index", "amount", "cubic_meters", "days", "tons",
                   "percentage", "metric_tons", "kgoe"}
    _DATE_ATTRS = {"date", "year", "initial_date", "final_date", "valid_date",
                   "valid_from", "valid_to", "start_date", "end_date"}
    measurement_records = []   # (head, rel_short, [(attr, val)]) — one per CVT, attrs paired
    seen_meas = set()
    # a CVT is a PURE measurement (date+value, no named-entity answer) ONLY if ALL its
    # attrs are date/value/literal — if it carries a named entity (e.g. religion_percentage
    # has religion=Islam + percentage=86.1 + date), it is an ENTITY-bearing CVT and must
    # keep the old flatten path (the named entity IS the answer; the value is auxiliary).
    def _is_pure_measurement_cvts(cvt_name):
        attrs = cvt_attrs.get(cvt_name, [])
        if not attrs:
            return False
        attr_keys = {a for a, _ in attrs}
        if not (attr_keys & _MEAS_ATTRS) or not (attr_keys & _DATE_ATTRS):
            return False
        # every attr must be a meas/date attr (no named-entity-bearing attr like religion/holder/title)
        non_meas = attr_keys - _MEAS_ATTRS - _DATE_ATTRS
        return len(non_meas) == 0
    fallback = []
    for tr in all_triples:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            continue
        h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
        if not is_cvt_like(h) and is_cvt_like(t) and t not in cvt_holders:
            if _is_pure_measurement_cvts(t):
                # preserve as one event-record line keyed by the CVT (pairing kept)
                # use the incoming relation short-name as the record type
                rel_short = _short_rel(r)
                key = ("MEAS", normalize(h), t)
                if key in seen_meas:
                    continue
                seen_meas.add(key)
                # pick the value attr (first meas attr) + the date attr
                m_attrs = cvt_attrs.get(t, [])
                val = next((v for a, v in m_attrs if a in _MEAS_ATTRS), None)
                dt = next((v for a, v in m_attrs if a in _DATE_ATTRS), None)
                if val is not None:
                    measurement_records.append((h, rel_short, dt, val))
                continue
            # CVT stays VISIBLE as the path intermediate (path fidelity — the
            # rollout data keeps CVTs as intermediate nodes; flattening attrs
            # onto the named head fabricated direct named↔named edges and hid
            # the path structure). Keep the incoming edge + hang the CVT's
            # attrs ON the CVT.
            fallback.append((h, _short_rel(r), t))
            for a, v in cvt_attrs.get(t, []):
                if is_cvt_like(v) or a in _EDGE_NOISY_SHORT or normalize(h) == normalize(v):
                    continue
                fallback.append((t, a, v))

    # dedup against shown_edges (records by record key, edges by edge key)
    n_overlap = 0
    new_records = []
    for rec in records:
        hd, title, fv, tv, juris, extra = rec
        rkey = ("REC", normalize(hd), normalize(title or ""), fv or "", tv or "")
        if rkey in shown_edges:
            n_overlap += 1; continue
        shown_edges.add(rkey)
        new_records.append(rec)
    new_direct = []
    for h, r, t in direct + fallback:
        ekey = _shown_edge_key(h, r, t)
        if ekey in shown_edges:
            n_overlap += 1; continue
        shown_edges.add(ekey)
        new_direct.append((h, r, t))

    # render
    def _attrs_str(fv, tv, juris, extra):
        parts = []
        if fv:
            parts.append(f"from={fv[:10]}")
        if tv:
            if tv == "(incumbent)":
                parts.append("to=incumbent")
            else:
                parts.append(f"to={tv[:10] if isinstance(tv, str) and len(tv) > 10 else tv}")
        if juris:
            parts.append(f"jurisdiction={juris}")
        for a, v in extra[:2]:
            parts.append(f"{a}={v}")
        return "; ".join(parts)

    # zero-information-gain compression (user principle): within a comparison
    # group, an attribute IDENTICAL across all records cannot discriminate the
    # candidates — repeating it per record only pollutes the prompt. Collapse
    # uniform attrs (jurisdiction + extra keys; from/to stay per-record — they
    # are the usual discriminators and often the literal answer) into ONE
    # "(all: k=v)" clause on the group header; varying attrs stay per-record.
    def _uniform_attrs(recs):
        def _normv(v):
            return normalize(str(v)) if v is not None else None
        uniform = []
        juris_vals = [_normv(j) for _, _, _, j, _ in recs]
        has_juris = any(v is not None for v in juris_vals)
        if has_juris and len(set(v for v in juris_vals if v is not None)) == 1 \
                and all(v is not None for v in juris_vals):
            uniform.append(("jurisdiction", recs[0][3]))
        keys = []
        for _, _, _, _, ex in recs:
            for a, _v in (ex or []):
                if a not in keys:
                    keys.append(a)
        for a in keys:
            vals = [next((v for k, v in (ex or []) if k == a), None) for _, _, _, _, ex in recs]
            if all(v is not None for v in vals) and len(set(_normv(v) for v in vals)) == 1:
                uniform.append((a, vals[0]))
        return set(a for a, _ in uniform), uniform

    lines = []
    by_title = {}
    for hd, title, fv, tv, juris, extra in new_records:
        by_title.setdefault(title or "(office)", []).append((hd, fv, tv, juris, extra))
    for title, recs in by_title.items():
        if len(recs) == 1:
            hd, fv, tv, juris, extra = recs[0]
            astr = _attrs_str(fv, tv, juris, extra)
            lines.append(f"  {hd} → {title}" + (f"  [{astr}]" if astr else ""))
        else:
            hide, uniform_kv = _uniform_attrs(recs)
            hdr = f"  {title}"
            if uniform_kv:
                hdr += f" (all: {'; '.join(f'{a}={str(v)[:24]}' for a, v in uniform_kv)})"
            lines.append(hdr + ":")
            for hd, fv, tv, juris, extra in recs:
                extra = [(a, v) for a, v in (extra or []) if a not in hide]
                if "jurisdiction" in hide:
                    juris = None
                astr = _attrs_str(fv, tv, juris, extra)
                lines.append(f"    - {hd}" + (f"  [{astr}]" if astr else ""))
    # measurement records: render as preserved event entities, (date, value) PAIRED.
    # group by (head, rel) so one country's CPI records list together, each as
    # `head --rel--> rec[date=D; value=V]` — the date↔value correspondence the model
    # needs to pick a year. Values truncated for readability.
    meas_by_hr = {}
    for h, rel, dt, val in measurement_records:
        meas_by_hr.setdefault((h, rel), []).append((dt, val))
    for (h, rel), pairs in meas_by_hr.items():
        # sort by date if present (chronological), else keep order
        def _yr(d):
            m = re.match(r"(\d{4})", str(d or ""))
            return int(m.group(1)) if m else 9999
        pairs.sort(key=lambda dv: _yr(dv[0]))
        rec_strs = []
        for dt, val in pairs:
            d = str(dt)[:7] if dt else "?"
            v = str(val)
            if re.match(r"^-?\d+\.\d{4,}$", v):
                v = str(round(float(v), 2))
            rec_strs.append(f"date={d}; value={v}")
        # cap per line to avoid explosion (many years of data)
        if len(rec_strs) > 12:
            rec_strs = rec_strs[:12] + [f"... +{len(rec_strs)-12} more"]
        lines.append(f"  {h} --{rel}--> " + " | ".join(f"[{rs}]" for rs in rec_strs))
    # BARE-CVT TAIL DROP (2026-08-19 audit): tails that are still raw m./g.
    # IDs after inlining carry NO attributes — pure noise (the Gingrich/Nordic
    # dumps showed dozens per line). Attribute-inlined CVTs ('m.xxx [k=v;..]')
    # keep their block; the walk path edges in records are unaffected.
    # UNIFORM-ATTR COLLAPSE (2026-08-21, user original design extended to the
    # inline path): CVTs on ONE pattern path often share identical attributes
    # (every personal-appearance event carries type_of_appearance=Him/Herself;
    # person=X) — zero information gain, repeated N times per line. Post-merge
    # string transform: attrs common to ALL bracketed CVTs of a line are
    # hoisted into one '(all: k=v; …)' prefix; each bracket keeps only its
    # DISCRIMINATING attrs.
    import re as _re_line
    _cvt_tok = _re_line.compile(r'\b([mg]\.[0-9a-z_]+) \[([^\]]+)\]')

    def _collapse_uniform_line(ln: str) -> str:
        toks = _cvt_tok.findall(ln)
        if len(toks) < 2:
            return ln

        def _kv(s):
            return {p.partition("=")[0].strip(): p.partition("=")[2].strip()
                    for p in s.split("; ") if "=" in p}
        first = _kv(toks[0][1])
        if not first:
            return ln
        uniform = {k: v for k, v in first.items()
                   if all(_kv(t[1]).get(k) == v for t in toks[1:])}
        if not uniform:
            return ln

        def _strip_bracket(m):
            keep = [p for p in m.group(2).split("; ")
                    if p.partition("=")[0].strip() not in uniform]
            return m.group(1) + (f" [{'; '.join(keep)}]" if keep else "")
        body = _cvt_tok.sub(_strip_bracket, ln)
        hdr = "(all: " + "; ".join(f"{k}={str(v)[:24]}" for k, v in uniform.items()) + ") "
        # attach the header right after the line's relation arrow when present
        m = _re_line.search(r'^(.*?--[^>]+-->\s*)', body)
        return (m.group(1) + hdr + body[m.end():]) if m else hdr + body

    _evs = [(h, r, t) for h, r, t in _inline_events(new_direct)
            if not (is_cvt_like(t) and "[" not in t)]
    for ln in _merge_edges(_evs):
        lines.append(f"  {_collapse_uniform_line(ln)}")
    return lines, n_overlap


def _format_merged(collected, accumulated=None, max_tails: int = 60, max_chains: int = 4) -> list:
    """Cross-center merged display. `collected` = list of (center_name,
    PatternEvidence) across all centers of one retrieve_subgraph call. Group by
    PURE relation chain (anchor stripped) → ONE header per relation pattern; each
    center's body lines follow (every line starts with its own root, so the
    variable's bindings appear as the roots under the shared pattern). Hops
    preserved per center. `accumulated` = PRIOR calls' triples snapshot for
    edge-level cycle suppression (None disables cross-call suppression)."""
    if not collected:
        return []
    groups, order = {}, []
    for _center, pe in collected:
        rc = _relation_chain(getattr(pe, "readable", ""))
        if rc not in groups:
            groups[rc] = []; order.append(rc)
        groups[rc].append(pe)
    out = []
    for rc in order:
        out.append(f"PATTERN: {_pattern_template(groups[rc][0].readable)}")
        for pe in groups[rc]:
            out.extend(_format_pattern_body(pe, accumulated, max_tails, max_chains))
    return out


def _format_pattern(pe, accumulated=None, max_tails: int = 60, max_chains: int = 4) -> list:
    """Single-PatternEvidence display (header + body). Kept for unit tests; the
    production path uses _format_merged for cross-center grouping."""
    body = _format_pattern_body(pe, accumulated, max_tails, max_chains)
    if not body:
        return []
    return [f"PATTERN: {_pattern_template(getattr(pe, 'readable', ''))}"] + body


def _in_subgraph(idx: int, ctx) -> bool:
    return idx in getattr(ctx, "subgraph_entities", set())


def _norm_idx_all(ctx):
    """normalize(name) → [all entity idxs with that name] (multi-map), built
    once per ctx — the MULTI-map matters: a single-valued dict drops duplicate
    names ('Catholicism' at two idxs). Shared by _accumulate and
    _entity_correction (both rebuilt it O(V) per call in the event loop)."""
    m = getattr(ctx, "_norm_idx_multi", None)
    if m is None:
        from collections import defaultdict
        m = defaultdict(list)
        for i, e in enumerate(ctx.ents):
            if e:
                m[normalize(e)].append(i)
        ctx._norm_idx_multi = m
    return m


def _accumulate(ctx, center_idx: int, pe) -> None:
    """Add every entity the model just saw (the center + all named entities in the
    dense tree, incl. CVT-attribute entities like jurisdiction/actor/note) to
    ctx.subgraph_entities, and named candidates to ctx.all_candidates."""
    se = getattr(ctx, "subgraph_entities", None)
    if se is None:
        se = set(); ctx.subgraph_entities = se
    se.add(center_idx)
    # accumulate edges (h_name, r_name, t_name) for cross-call display cycle-suppression
    # R4 DISPOSAL STATUS (2026-10-01, specs/r4_cleanup_2026-10-01.md): KEEP —
    # the evaluator left text reconstruction behind in R3-c (the dataflow D3-5
    # "delete once the evaluator is on the sidecar" precondition is met), but
    # the RENDER chain still reads this store on BOTH faces: the chain-level
    # render delta (_shown_facts / _delta_new below: all-old chains render
    # nothing, 09-24 exact-triple keys) and the per-call prior snapshot
    # (treq["prior"] → seq_render_v38's edge-shown comparison); plus the
    # answer-pool OFF face and the rollout's evidence_entities. Delete when
    # the render delta switches to the ledger's is_new (O3 / D3-3b).
    at = getattr(ctx, "accumulated_triples", None)
    if at is None:
        at = set(); ctx.accumulated_triples = at
    for tr in (pe.triples or []):
        if len(tr) == 3:
            at.add((str(tr[0]), str(tr[1]), str(tr[2])))
    # MULTI-map: a name may normalize to several entity indices (text + non_text entity
    # lists merge duplicates — 'Catholicism' at idx 69 AND 2007). A single-valued dict
    # keeps only the LAST, but _name_to_idx returns the FIRST → the boundary check
    # (_in_subgraph) looks for the wrong idx and rejects a valid center. Add ALL matches.
    n2i_all = _norm_idx_all(ctx)
    # DETERMINISTIC FIRST-APPEARANCE ORDER (perf-4 replay oracle, 2026-08-23):
    # the old `names = set(...)` + `for nm in names` made all_candidates
    # ordering hash-seed-dependent PER PROCESS — the same case replayed in a
    # new process emitted a differently-ordered candidate_pool (same SET).
    # dict.fromkeys dedups identically while preserving arrival order.
    ordered = list(dict.fromkeys(
        list(pe.candidates or [])
        + [x for tr in (pe.triples or []) if len(tr) == 3
           for x in (tr[0], tr[2])]))
    seen_cand = {normalize(c) for c in getattr(ctx, "all_candidates", []) or []}
    for nm in ordered:
        if not nm:
            continue
        for idx in n2i_all.get(normalize(nm), ()):   # add ALL entities with this name
            se.add(idx)
        if not is_cvt_like(nm):          # only NAMED entities join the answer candidate pool
            nn = normalize(nm)
            if nn not in seen_cand:
                seen_cand.add(nn); ctx.all_candidates.append(nm)


def _collect_cvt_neighbors_to_pool(ctx) -> None:
    """Ensure ALL non-CVT entities connected to any CVT in the subgraph are in
    all_candidates. The dense tree (_cvt_attr_display) shows ALL of a CVT's edges,
    but pe.triples (_expand_endpoint_cvt) may miss some due to scoring/limits —
    so entities visible in the tree (like an olympic event name) can be absent from
    the answer pool, causing off-pool rejection of a correct answer. This scans the
    raw graph edges of every CVT in subgraph_entities and adds every named neighbor.
    Rule: if an entity appears in any retrieved tree, it's answerable."""
    se = getattr(ctx, "subgraph_entities", None)
    if not se:
        return
    cvt_idxs = {i for i in se if 0 <= i < len(ctx.ents) and is_cvt_like(ctx.ents[i])}
    if not cvt_idxs:
        return
    seen_cand = {normalize(c) for c in getattr(ctx, "all_candidates", []) or []}
    for h, r, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        other = None
        if h in cvt_idxs and 0 <= t < len(ctx.ents) and not is_cvt_like(ctx.ents[t]):
            other = t
        elif t in cvt_idxs and 0 <= h < len(ctx.ents) and not is_cvt_like(ctx.ents[h]):
            other = h
        if other is not None:
            nm = ctx.ents[other]
            nn = normalize(nm)
            if nn not in seen_cand:
                seen_cand.add(nn); ctx.all_candidates.append(nm)
            se.add(other)


def _seq_pool_relids(ctx, entity_set) -> set:
    """SEQ relation-candidate pool = SAPS 2-hop reach (`_reach2_relids`, untouched)
    PLUS one CVT-transparent hop: relations on entities reached THROUGH a CVT
    mediator (E -> CVT -> E2). The flat 2-hop pool stops at the CVT's own edges, so
    a relation sitting BEHIND a CVT bridge (e.g. an org's leadership CVT leads to a
    person whose `people.person.religion` is the answer) is never ranked by GTE.
    Extending one CVT-transparent hop surfaces it. Only affects what GTE can rank
    (the walk's bounds are unchanged).

    PERF (2026-08-23): the CVT expansion rescanned ALL edges 3× per call with no
    memo — a hub first call measured 324ms IN THE EVENT LOOP (blocking every
    concurrent dispatcher). The pool is deterministic per (ctx, center set), so
    the whole result is memoized per ctx; the expansion itself runs on the
    per-ctx adjacency index instead of full edge scans."""
    _memo = getattr(ctx, "_seq_pool_memo", None)
    if _memo is None:
        _memo = {}
        ctx._seq_pool_memo = _memo
    ents = frozenset(e for e in entity_set
                     if e is not None and 0 <= e < len(ctx.ents))
    if ents in _memo:
        return set(_memo[ents])
    out = set(_reach2_relids(ctx, ents))
    if ents:
        from kgqa.agent.tools import _full_adj
        adj = _full_adj(ctx)
        n_ents = len(ctx.ents)
        # 1-hop neighbors, isolate the CVT mediators
        direct = set()
        for e in ents:
            for _r, other in adj[e]:
                if 0 <= other < n_ents:
                    direct.add(other)
        cvts = {x for x in direct if is_cvt_like(ctx.ents[x])}
        if cvts:
            # entities lying behind each CVT (the CVT's other endpoints)
            behind = set()
            for cvt in cvts:
                for _r, other in adj[cvt]:
                    if 0 <= other < n_ents and other not in ents:
                        behind.add(other)
            # add relations on those behind-entities' edges (the CVT-transparent hop)
            for mid in behind:
                for r, _other in adj[mid]:
                    out.add(r)
        from kgqa.traversal.path_utils import _is_noisy_path_relation
        out = {r for r in out if not _is_noisy_path_relation(ctx.rels[r])}
    _memo[ents] = out
    return set(out)


# Walk parallelism (rollout throughput): the graph walk is pure-CPU Python, so
# gather() over cases is SERIAL under the GIL — with CASE_BATCH=1000 the LLM
# batch takes 84s while the per-round walk pile-up takes 150-200s. WALK_POOL=<n>
# runs the one-step pipeline (walk + compress + materialize + evidence build)
# in n single-worker SPAWN lanes, ONE POOL TASK PER retrieve_subgraph CALL: a
# call's centers share the case arrays, so they traverse together in one worker
# stage_5 batch (task count = calls, not centers — 7416 → ~1814 on 267×3).
# Per-CASE affinity: lane = hash(case_key) % n, each lane keeps its OWN
# sent-set — a case's ~1MB arrays ship exactly once and stay resident in that
# lane's worker cache (a shared pool paired a GLOBAL sent-set with PER-WORKER
# caches → ~2/3 of tasks MISSed and reshipped the arrays). Spawn, NOT fork:
# forked children corrupt a live vLLM EngineCore ("WorkerProc initialization
# failed"). Off by default (production HTTP eval keeps the inline path).
_WALK_LANES = None
_SENT_BY_LANE = []


def _walk_worker_init():
    """Lane-worker GC policy (walk-perf, 2026-09-07): the per-case adjacency
    memo (logical_paths/frontier/k_queue) parks large container graphs that
    make gen-2 collections scan-happy — a dense-case walk showed 300ms GC
    spikes between 4ms walks. Raise the gen-0 threshold so collections stay
    rare (walk objects are mostly acyclic; the pool recycles the worker if
    memory ever grows)."""
    import gc
    gc.set_threshold(2_000_000, 100, 100)


def _get_walk_lanes(n: int):
    """n single-worker spawn executors + the matching per-lane sent-sets.
    Created once; a later n mismatch keeps the existing lanes (same contract
    as the old shared pool)."""
    global _WALK_LANES, _SENT_BY_LANE
    if _WALK_LANES is None:
        import concurrent.futures as cf
        import multiprocessing as mp
        _spawn = mp.get_context("spawn")
        _WALK_LANES = [cf.ProcessPoolExecutor(max_workers=1, mp_context=_spawn,
                                              initializer=_walk_worker_init)
                       for _ in range(n)]
        _SENT_BY_LANE = [set() for _ in range(n)]
    return _WALK_LANES


# worker-side: case_key -> (sample, pilot_row, ents, rels, h_ids, r_ids, t_ids,
# rel_texts). Simple bounded cache (spawn workers are long-lived within a run).
_W_CASE_CACHE = {}


def _walk_case_steps(case_key, d, steps):
    """Worker-side traversal of ONE case's steps (d = its cached arrays).
    Pure: identical inputs → identical output list (per-center dict[label ->
    PatternEvidence], {} when the walk reached nothing). Shared by the
    per-call lane task (_walk_call_spawn) and the round-level batch task
    (_walk_batch_spawn).
    Step tuples are (center_idx, rel_idxs, fid) or, for a PATTERN-PREFIX
    continuation (user design 2026-09-10), (center_idx, rel_idxs, fid,
    prefix_names): the center is a tail of an earlier pattern and the
    prefix = the earlier walk's node NAMES — RPE records target edges INTO
    that territory but never expands through it (状态保持,不重复回去)."""
    sample, pilot_row, ents, rels, h_ids, r_ids, t_ids, rel_texts = d
    import asyncio as _aio
    n2i = None
    cases = []
    for step in steps:
        center_idx, rel_idxs, fid = step[0], step[1], step[2]
        cs = CaseState(case_id=case_key[0], case_num=case_key[1],
                       sample=sample, pilot_row=pilot_row)
        cs.anchor_idx = center_idx
        cs.anchor_name = ents[center_idx] if 0 <= center_idx < len(ents) else ""
        cs.h_ids, cs.r_ids, cs.t_ids = h_ids, r_ids, t_ids
        cs.ents, cs.rels, cs.rel_texts = ents, rels, rel_texts
        if isinstance(rel_idxs, tuple):
            # MULTI-STEP derived pattern (realignment spec pillar 2+3):
            # rel_idxs is a tuple of per-hop frozensets — the engine walks
            # the sequence with forward validation (chain_expand lookahead);
            # RPE falls back only on weak coverage (n_steps > 1).
            cs.step_relations = [set(x) for x in rel_idxs]
            cs.steps = [{"id": (fid or "f") + f"#s{k}"} for k in range(len(rel_idxs))]
        else:
            cs.step_relations = [set(rel_idxs)]
            cs.steps = [{"id": fid or "f"}]
        cs.breakpoints = {}
        cs.active = True
        if len(step) > 3 and step[3]:
            # names → idx set (once per call: the map is shared by all steps)
            if n2i is None:
                n2i = {}
                for j, e in enumerate(ents):
                    n2i.setdefault(str(e), j)
            cs.prefix_nodes = frozenset(
                n2i[nm] for nm in step[3]
                if nm in n2i and n2i[nm] != center_idx)
        cases.append(cs)
    try:
        _aio.run(stage_5_graph_traversal(cases))
    except Exception:
        return [{} for _ in steps]
    out = []
    for step, cs in zip(steps, cases):
        center_idx, rel_idxs = step[0], step[1]
        _sel_ids = (set().union(*rel_idxs)
                    if isinstance(rel_idxs, tuple) else set(rel_idxs))
        paths = cs.paths or []
        patterns = (compress_paths(paths, ents, rels, center_idx, set())
                    if paths else (cs.logical_paths or []))
        valid = [lp for lp in patterns if isinstance(lp, dict) and lp.get("best_raw_path")]
        valid = materialize_selected_logical_patterns(
            valid, ents, rels, h_ids, r_ids, t_ids, center_idx, set())
        if not valid:
            out.append({})
            continue
        out.append(build_pattern_evidence_triples(
            valid, ents, rels, h_ids, r_ids, t_ids, center_idx,
            max_grouped_lines=120, selected_rel_ids=_sel_ids))
    return out


def _walk_call_spawn(task):
    """Worker body for ONE retrieve_subgraph call: task = (case_key, ctx_data,
    steps) with steps = [(center_idx, rel_idxs, fid), ...]. All centers of the
    call share the case arrays, so they traverse in ONE stage_5 batch (its
    native multi-case mode — per-case state only, thread-isolated inside the
    worker). Returns a per-center list aligned with steps, each entry a
    dict[label -> PatternEvidence], or {} when that center's walk reached
    nothing; "MISS" when the case arrays are not cached (caller reships with
    data)."""
    case_key, ctx_data, steps = task
    if ctx_data is not None:
        if len(_W_CASE_CACHE) > 512:
            _W_CASE_CACHE.clear()
        _W_CASE_CACHE[case_key] = ctx_data
    d = _W_CASE_CACHE.get(case_key)
    if d is None:
        return "MISS"          # sentinel: caller reships this task with data
    return _walk_case_steps(case_key, d, steps)


def _walk_batch_spawn(payload):
    """Worker body for ONE FLUSH of the round-level coordinator: payload =
    [(case_key, ctx_data_or_None, steps), ...] — every case the flush assigned
    to THIS lane, one stage_5 batch per case. Returns a list aligned with
    payload; each entry is "MISS" (case arrays evicted — caller reships that
    one case with data) or the per-step evidence list."""
    out = []
    for case_key, ctx_data, steps in payload:
        if ctx_data is not None:
            if len(_W_CASE_CACHE) > 512:
                _W_CASE_CACHE.clear()
            _W_CASE_CACHE[case_key] = ctx_data
        d = _W_CASE_CACHE.get(case_key)
        out.append("MISS" if d is None else _walk_case_steps(case_key, d, steps))
    return out


async def _run_walk_one_step(ctx, center_idx: int, rel_idxs, fid: str,
                             prefix_names=None):
    """Inline (in-process) one-center walk: the WALK_POOL=0 path, and the exact
    per-center unit the packed worker (_walk_call_spawn) replicates off-process.
    Reuses the proven SAPS walk (stage_5_graph_traversal: multi-hop, K-path,
    CVT-penetrating) scoped to ONE step from center_idx along rel_idxs
    (prefix_names: pattern-prefix continuation, see _walk_case_steps). Mirrors
    _do_select's cs setup + walk + evidence build (tools.py:1280-1337), but for a
    single step whose anchor is the current center. Returns dict[label -> PatternEvidence]
    (each carries .triples + .candidates + .tree_data for the dense CVT-inline tree)."""
    cs = CaseState(case_id=ctx.case_id or "seq", case_num=ctx.case_num or 0,
                   sample=ctx.sample, pilot_row=ctx.pilot_row)
    cs.anchor_idx = center_idx
    cs.anchor_name = ctx.ents[center_idx] if 0 <= center_idx < len(ctx.ents) else ""
    cs.h_ids, cs.r_ids, cs.t_ids = ctx.h_ids, ctx.r_ids, ctx.t_ids
    cs.ents, cs.rels, cs.rel_texts = ctx.ents, ctx.rels, ctx.rel_texts
    if isinstance(rel_idxs, tuple):
        cs.step_relations = [set(x) for x in rel_idxs]
        cs.steps = [{"id": (fid or "f") + f"#s{k}"} for k in range(len(rel_idxs))]
    else:
        cs.step_relations = [set(rel_idxs)]        # ONE step
        cs.steps = [{"id": fid or "f"}]
    cs.breakpoints = {}
    cs.active = True
    if prefix_names:
        _n2i = {}
        for j, e in enumerate(ctx.ents):
            _n2i.setdefault(str(e), j)
        cs.prefix_nodes = frozenset(
            _n2i[nm] for nm in prefix_names
            if nm in _n2i and _n2i[nm] != center_idx)
    try:
        await stage_5_graph_traversal([cs])
    except Exception:
        return {}
    paths = cs.paths or []
    patterns = (compress_paths(paths, ctx.ents, ctx.rels, center_idx, set())
                if paths else (cs.logical_paths or []))
    valid = [lp for lp in patterns if isinstance(lp, dict) and lp.get("best_raw_path")]
    valid = materialize_selected_logical_patterns(
        valid, ctx.ents, ctx.rels, ctx.h_ids, ctx.r_ids, ctx.t_ids, center_idx, set())
    if not valid:
        return {}
    return build_pattern_evidence_triples(
        valid, ctx.ents, ctx.rels, ctx.h_ids, ctx.r_ids, ctx.t_ids, center_idx,
        max_grouped_lines=120,
        # tuple (multistep layer sets) must FLATTEN — set(rel_idxs) on a
        # tuple yields a set of frozensets the filter never matches (the
        # worker path at _walk_case_steps already unions; this inline
        # WALK_POOL=0 path silently lost the whole selection)
        selected_rel_ids=(set().union(*rel_idxs)
                          if isinstance(rel_idxs, tuple) else set(rel_idxs)))


def _walk_call_spawn_timed(task):
    """Affine-lane wrapper around _walk_call_spawn splitting the lane timer:
    wait = submit→worker pickup (queue depth behind other cases' tasks), exec =
    actual traversal time. task carries the parent-side submit timestamp;
    time.perf_counter() is CLOCK_MONOTONIC on Linux, so the difference is
    valid across processes. Returns (result, wait_s, exec_s)."""
    import time as _t
    t0 = _t.perf_counter()
    res = _walk_call_spawn(task[:3])
    return res, t0 - task[3], _t.perf_counter() - t0


def _walk_batch_spawn_timed(task):
    """Same timing split for a whole flush's multi-case lane task."""
    import time as _t
    t0 = _t.perf_counter()
    res = _walk_batch_spawn(task[0])
    return res, t0 - task[1], _t.perf_counter() - t0


# ─── round-level walk coordinator (perf2 2026-08-23) ─────────────────────
# The rollout is round-synchronized: after each LLM batch, every dispatcher's
# retrieve_subgraph call arrives within one burst. The per-call lane path let
# dozens of concurrent submits pile onto 3 single-worker lanes — the measured
# walk lane is ~98% queue wait. The coordinator ports the GTE server's
# ADAPTIVE two-stage window: the first arrival opens a short FIRST slice; a
# second arrival within it extends the deadline to the FULL window counted
# from the first arrival (a burst); a LONE request flushes right after the
# first slice — no fixed full-window tax on single/tail requests. At flush:
# identical (center, rels) steps across the batch share one execution slot
# (G=3 lockstep trajectories emit identical deterministic walks), cases are
# grouped and sticky-balanced onto the affine lanes by slot count, and each
# lane receives ONE packed multi-case task (queue depth 1/lane per flush).
# Completed slots are memoized for cross-batch reuse (reworded loops).
# Env: WALK_BATCH_WINDOW (full window s, default 1.0; <=0 = legacy per-call
# lane path), WALK_BATCH_FIRST (first slice s, default 0.3).
_WALK_BATCH = None                       # open collection batch or None
_WALK_CASE_LANE = {}                     # case_key -> lane idx (sticky residency)
_WALK_MEMO = {}                          # case_key -> {(center_idx, frozenset): pe}
_WALK_MEMO_MAX_CASES = 96
_WALK_BATCH_STATS = {"batches": 0, "single": 0, "burst": 0, "reqs": 0,
                     "slots": 0, "memo_hits": 0, "dedup_shares": 0,
                     "ipc_bytes": 0.0, "collect_s": 0.0, "burst_span_s": 0.0}


def _walk_batch_enqueue(loop, ctx, case_key, steps):
    """Register one request on the open batch (opening one if none) and return
    its future. The flush callback is scheduled here (see the adaptive window
    above). ctx rides along only to ship the case arrays on first sight."""
    global _WALK_BATCH
    import time as _t
    if _WALK_BATCH is None:
        _first = float(os.environ.get("WALK_BATCH_FIRST", "0.3") or 0)
        _WALK_BATCH = {"loop": loop, "reqs": [], "t_first": _t.perf_counter()}
        _WALK_BATCH["cb"] = loop.call_later(
            max(_first, 0.001), _walk_flush_check)
    fut = loop.create_future()
    _WALK_BATCH["reqs"].append({"ctx": ctx, "case_key": case_key,
                                "steps": steps, "t0": _t.perf_counter(),
                                "fut": fut})
    return fut


def _walk_flush_check():
    """Fires at the end of the FIRST slice: flush now unless the slice saw a
    second arrival — that signals the round's burst, so extend to the FULL
    window counted from the first arrival (mirrors the GTE _collect_batch
    adaptive two-stage window)."""
    b = _WALK_BATCH
    if b is None:
        return
    if len(b["reqs"]) > 1:
        import time as _t
        _full = float(os.environ.get("WALK_BATCH_WINDOW", "1.0") or 0)
        remain = (b["t_first"] + _full) - _t.perf_counter()
        if remain > 0:
            b["loop"].call_later(remain, _walk_flush_now)
            return
    _walk_flush_now()


def _walk_flush_now():
    """Close the open batch (later arrivals open a fresh one) and hand it to
    the async flusher."""
    global _WALK_BATCH
    b, _WALK_BATCH = _WALK_BATCH, None
    if b is not None and b["reqs"]:
        b["loop"].create_task(_walk_flush_async(b))


async def _walk_flush_async(batch):
    """One flush: dedup → memo lookup → sticky lane grouping (ONE packed task
    per lane) → MISS reship → memo fill → resolve every request future."""
    import asyncio as _aio
    import pickle as _pkl
    import time as _t
    from kgqa.core.utils import PHASE_TIMES
    reqs = batch["reqs"]
    st = _WALK_BATCH_STATS
    n_req_steps = sum(len(r["steps"]) for r in reqs)
    t_flush = _t.perf_counter()
    st["batches"] += 1
    st["single" if len(reqs) == 1 else "burst"] += 1
    st["reqs"] += len(reqs)
    st["collect_s"] += sum(t_flush - r["t0"] for r in reqs)
    st["burst_span_s"] += max(r["t0"] for r in reqs) - min(r["t0"] for r in reqs)
    PHASE_TIMES.setdefault("walk_collect", 0.0)
    PHASE_TIMES["walk_collect"] += sum(t_flush - r["t0"] for r in reqs)
    # flush WALL (first arrival → every future resolved) — the batch-view cost;
    # flushes overlap each other/GTE, so the SUM upper-bounds the walk share.
    _t_first = min(r["t0"] for r in reqs)
    try:
        # 1. unique execution slots per case, memo hits served for free.
        # slot key = (center_idx, frozenset(rel_idxs), prefix) — the walk
        # consumes rel_idxs as a SET, so order never affects the result; the
        # PATTERN-PREFIX set (continuation walks, user design 2026-09-10) is
        # part of the identity: the same center+rels walked with a different
        # prefix is a different result.
        def _step_key(s):
            i, rel_idxs = s[0], s[1]
            _pf = s[3] if len(s) > 3 else None
            # MULTI-STEP sequences (tuple of per-hop frozensets) key by the
            # ordered sequence itself — hop ORDER is the pattern's identity
            return (i, rel_idxs if isinstance(rel_idxs, tuple)
                    else frozenset(rel_idxs), _pf)

        case_slots, case_ctx = {}, {}
        for r in reqs:
            ck = r["case_key"]
            slots = case_slots.setdefault(ck, {})
            case_ctx.setdefault(ck, r["ctx"])
            for s in r["steps"]:
                slots.setdefault(_step_key(s), (s[0], s[1],
                                                s[3] if len(s) > 3 else None))
        # memo-covered slots are NOT re-executed: drop them from the lane
        # payload, and resolve requests whose steps are ALL memo hits right
        # now (they must not wait out the lane tasks).
        n_memo = 0
        pending = {}
        for ck, slots in case_slots.items():
            mcase = _WALK_MEMO.get(ck) or {}
            pend = {}
            for k, step in slots.items():
                if k in mcase:
                    n_memo += 1
                else:
                    pend[k] = step
            pending[ck] = pend
        done_reqs = []
        for r in reqs:
            ck = r["case_key"]
            mcase = _WALK_MEMO.get(ck) or {}
            if all(_step_key(s) in mcase for s in r["steps"]):
                if not r["fut"].done():
                    r["fut"].set_result([mcase[_step_key(s)] for s in r["steps"]])
                done_reqs.append(r)
        done_ids = {id(r) for r in done_reqs}
        reqs = [r for r in reqs if id(r) not in done_ids]
        n_slots = sum(len(p) for p in pending.values())
        st["memo_hits"] += n_memo
        st["slots"] += n_slots
        st["dedup_shares"] += n_req_steps - n_slots - n_memo
        if not reqs:
            return
        # 2. sticky lane assignment over the PENDING (non-memo) slots:
        # existing map / resident sent-set first, then NEW cases greedy-LPT
        # by slot count (balanced AND resident).
        lanes = _get_walk_lanes(int(os.environ.get("WALK_POOL", "0") or 0))
        n = len(lanes)
        load = [0] * n
        by_lane = [[] for _ in range(n)]
        run_cases = [ck for ck in case_slots if pending[ck]]
        for ck in run_cases:
            lane = _WALK_CASE_LANE.get(ck)
            if lane is None:
                lane = next((i for i, s in enumerate(_SENT_BY_LANE) if ck in s),
                            None)
            if lane is not None and 0 <= lane < n:
                _WALK_CASE_LANE[ck] = lane
                by_lane[lane].append(ck)
                load[lane] += len(pending[ck])
        for ck in sorted((c for c in run_cases if c not in _WALK_CASE_LANE),
                         key=lambda c: -len(pending[c])):
            lane = min(range(n), key=lambda i: load[i])
            _WALK_CASE_LANE[ck] = lane
            by_lane[lane].append(ck)
            load[lane] += len(pending[ck])
        # 3. one packed task per lane: ship the arrays of cases this lane has
        # not seen yet (sticky lanes keep them resident across rounds).
        loop = batch["loop"]
        t_submit = _t.perf_counter()
        lane_jobs = []
        for li in range(n):
            if not by_lane[li]:
                continue
            payload = []
            for ck in by_lane[li]:
                ctx_data = None
                if ck not in _SENT_BY_LANE[li]:
                    ctx = case_ctx[ck]
                    ctx_data = (ctx.sample, ctx.pilot_row, ctx.ents, ctx.rels,
                                ctx.h_ids, ctx.r_ids, ctx.t_ids, ctx.rel_texts)
                    _SENT_BY_LANE[li].add(ck)
                payload.append((ck, ctx_data,
                                [(i, rel_idxs, "", pf) for i, rel_idxs, pf
                                 in pending[ck].values()]))
            lane_jobs.append((li, payload, loop.run_in_executor(
                lanes[li], _walk_batch_spawn_timed, (payload, t_submit))))
        results = await _aio.gather(*(j[2] for j in lane_jobs),
                                    return_exceptions=True)
        # lane results are (entry_list, lane_wait, lane_exec) — accumulate the
        # wait/exec split (queue wait behind other lanes' tasks vs traversal).
        slot_pe = {}
        reship = {}
        for (li, payload, _f), res in zip(lane_jobs, results):
            if isinstance(res, BaseException):
                # crashed lane job (huge-flush pickling/OOM) — visible, not
                # silent: affected slots resolve {} ("walk reached nothing")
                import sys as _sys
                print(f"  ⚠ walk lane job failed (lane {li}, {len(payload)} cases): "
                      f"{type(res).__name__}: {res}", file=_sys.stderr, flush=True)
                continue                       # crashed lane → {} per slot below
            entries, _wait, _exec = res
            PHASE_TIMES["walk_wait"] = PHASE_TIMES.get("walk_wait", 0.0) + _wait
            PHASE_TIMES["walk_exec"] = PHASE_TIMES.get("walk_exec", 0.0) + _exec
            st["ipc_bytes"] += len(_pkl.dumps(entries, protocol=4))
            for (ck, _d, steps), entry in zip(payload, entries):
                if entry == "MISS":
                    reship.setdefault(li, []).append(ck)
                else:
                    for s, pe in zip(steps, entry):
                        _pf = s[3] if len(s) > 3 else None
                        slot_pe.setdefault(
                            (ck, s[0], frozenset(s[1]), _pf), pe)
        if reship:
            # 4. worker cache evicted for these cases → reship WITH data on the
            # SAME lanes (sticky), one follow-up task per affected lane.
            jobs2 = []
            for li, cks in reship.items():
                payload = []
                for ck in cks:
                    ctx = case_ctx[ck]
                    payload.append((ck,
                                    (ctx.sample, ctx.pilot_row, ctx.ents, ctx.rels,
                                     ctx.h_ids, ctx.r_ids, ctx.t_ids, ctx.rel_texts),
                                    [(i, rel_idxs, "", pf) for i, rel_idxs, pf
                                     in pending[ck].values()]))
                jobs2.append((payload, loop.run_in_executor(
                    lanes[li], _walk_batch_spawn_timed,
                    (payload, _t.perf_counter()))))
            for (payload, _f), res in zip(
                    jobs2, await _aio.gather(*(f for _, f in jobs2),
                                             return_exceptions=True)):
                if isinstance(res, BaseException):
                    continue
                entries, _wait, _exec = res
                PHASE_TIMES["walk_wait"] = PHASE_TIMES.get("walk_wait", 0.0) + _wait
                PHASE_TIMES["walk_exec"] = PHASE_TIMES.get("walk_exec", 0.0) + _exec
                for (ck, _d, steps), entry in zip(payload, entries):
                    if entry == "MISS":
                        continue
                    for s, pe in zip(steps, entry):
                        _pf = s[3] if len(s) > 3 else None
                        slot_pe.setdefault(
                            (ck, s[0], frozenset(s[1]), _pf), pe)
        # 5. fill the cross-batch memo (evict whole oldest cases when bound).
        for (ck, i, rels_fs, _pf), pe in slot_pe.items():
            mcase = _WALK_MEMO.get(ck)
            if mcase is None:
                if len(_WALK_MEMO) >= _WALK_MEMO_MAX_CASES:
                    _WALK_MEMO.pop(next(iter(_WALK_MEMO)))
                mcase = _WALK_MEMO[ck] = {}
            mcase[(i, rels_fs, _pf)] = pe
        # 6. resolve every request from memo + fresh slots.
        for r in reqs:
            ck = r["case_key"]
            mcase = _WALK_MEMO.get(ck) or {}
            out = []
            for s in r["steps"]:
                _pf = s[3] if len(s) > 3 else None
                k = (s[0], frozenset(s[1]), _pf)
                pe = mcase.get(k)
                if pe is None:
                    pe = slot_pe.get((ck, s[0], k[1], _pf), {})
                out.append(pe)
            if not r["fut"].done():
                r["fut"].set_result(out)
    except Exception:
        # never leave a waiter hanging: the legacy per-call path returned
        # [{} for steps] on failure — resolve the same way.
        for r in reqs:
            if not r["fut"].done():
                r["fut"].set_result([{} for _ in r["steps"]])
    finally:
        PHASE_TIMES["walk_flush_wall"] = \
            PHASE_TIMES.get("walk_flush_wall", 0.0) + (_t.perf_counter() - _t_first)


async def _run_walk_coordinated(ctx, case_key, steps):
    """Coordinator entry: enqueue and await the flush's result."""
    import asyncio as _aio
    fut = _walk_batch_enqueue(_aio.get_running_loop(), ctx, case_key, steps)
    try:
        return await fut
    except Exception:
        return [{} for _ in steps]


async def _run_walk_lane(ctx, case_key, steps, pool_n):
    """Legacy per-call lane path (WALK_POOL>0, WALK_BATCH_WINDOW<=0): ONE
    packed task per retrieve_subgraph CALL on the case's hash-affine lane.
    Kept for non-round-synced callers (per-case runner)."""
    lanes = _get_walk_lanes(pool_n)
    lane_i = hash(case_key) % len(lanes)
    lane, sent = lanes[lane_i], _SENT_BY_LANE[lane_i]
    ctx_data = None
    if case_key not in sent:
        ctx_data = (ctx.sample, ctx.pilot_row, ctx.ents, ctx.rels,
                    ctx.h_ids, ctx.r_ids, ctx.t_ids, ctx.rel_texts)
        sent.add(case_key)
    import asyncio as _aio
    import time as _t
    from kgqa.core.utils import PHASE_TIMES
    loop = _aio.get_running_loop()

    async def _submit(ctx_data):
        return await loop.run_in_executor(
            lane, _walk_call_spawn_timed,
            (case_key, ctx_data, steps, _t.perf_counter()))

    try:
        res, _wait, _exec = await _submit(ctx_data)
    except Exception:
        return [{} for _ in steps]
    if res == "MISS":
        # worker cache was evicted — reship once WITH the case arrays, on the
        # SAME lane (affinity must not wobble on the retry)
        sent.add(case_key)
        try:
            res, _wait2, _exec2 = await _submit(
                (ctx.sample, ctx.pilot_row, ctx.ents, ctx.rels,
                 ctx.h_ids, ctx.r_ids, ctx.t_ids, ctx.rel_texts))
        except Exception:
            return [{} for _ in steps]
        _wait += _wait2
        _exec += _exec2
    # wait = lane-queue time (submit→pickup); exec = traversal time in the
    # worker. walk − (wait+exec) is parent-side pickle/event-loop overhead.
    PHASE_TIMES["walk_wait"] = PHASE_TIMES.get("walk_wait", 0.0) + _wait
    PHASE_TIMES["walk_exec"] = PHASE_TIMES.get("walk_exec", 0.0) + _exec
    return res


async def _run_walk_packed(ctx, steps):
    """All centers of ONE retrieve_subgraph call → per-center evidence dicts
    aligned with steps (accumulate/collect side effects stay in the caller —
    they write ctx). WALK_POOL=0/unset: inline per-center walk. WALK_POOL>0 +
    WALK_BATCH_WINDOW>0: the round-level coordinator (rollout path). WALK_POOL>0
    otherwise: the legacy per-call affine-lane submit."""
    _pool_n = int(os.environ.get("WALK_POOL", "0") or 0)
    if _pool_n <= 0:
        return [await _run_walk_one_step(
                    ctx, s[0], s[1], s[2],
                    s[3] if len(s) > 3 else None)
                for s in steps]
    case_key = (ctx.case_id or "seq", ctx.case_num or 0)
    if float(os.environ.get("WALK_BATCH_WINDOW", "0") or 0) > 0:
        return await _run_walk_coordinated(ctx, case_key, steps)
    return await _run_walk_lane(ctx, case_key, steps, _pool_n)


# ───────────────────────── decompose (declare + seed anchor; no grounding) ─────────────────────────

def _do_seq_decompose(args: Dict[str, Any], ctx, session=None) -> str:
    """SEQ decompose: declare the fact sequence + step-1 anchor. Does NOT ground — each
    fact is grounded by the model's own retrieve_relations call. Seeds ctx.subgraph_entities
    with the anchor so fact-1's center passes the boundary.
    Plan-contract v2 (2026-08-21): the plan is IMMUTABLE after declaration — a second
    plan call is rejected with a close-only directive (facts may be closed ✓/✗, never
    added). The declared answer_type word is captured and echoed at answer time."""
    # immutability gate: one plan per episode
    if getattr(ctx, "plan_declared", False):
        return _json_result({
            "error": "REJECTED: the plan was already declared — do NOT re-emit it. "
                     "To ADD evidence for an uncovered contract requirement, emit "
                     "an append-only extension instead:\n"
                     "[PLAN EXTEND]\ncovers: R2\n\nsg2.anchor: <question entity>\n"
                     "sg2.f1: <head> | <sub-question> | ?var\n\n"
                     "To change RELATIONS or CENTER of an existing fact, just "
                     "re-call retrieve_relations/retrieve_subgraph — the plan "
                     "locks facts, not retrieval choices.",
            "note": ("The plan is a closed contract. Its only legal transitions are "
                     "checkpoints (one md line per fact): `- fid ✓ ?var = v1 | v2` (resolved), "
                     "`- fid ✗ empty` (the reachable pool holds no relation advancing "
                     "this fact), `[fid ✗ moot]` (earlier evidence already bound this "
                     "fact's target). Continue the workflow with retrieve_relations / "
                     "retrieve_subgraph on the next OPEN fact. Answer when "
                     "every fact that can constrain/discriminate the answer variable "
                     "is closed — binding the variable early is AVAILABLE, not READY.")})
    # answer_type REQUIRED (2026-08-21, CH specimen: the model planned ?language
    # for a "which COUNTRIES" question — the optional type word let the inverted
    # answer slot through). The gate forces the turn-0 type commitment; a plan
    # whose answer slot contradicts the type word surfaces immediately.
    # (harness validate pre-checks this so a rejection never leaves the state
    # machine in RETRIEVE — kept here as the backstop.)
    _at = str(args.get("answer_type", "") or "").strip()
    if not _at:
        return _json_result({
            "error": "REJECTED: plan requires `answer_type` — ONE word for what the "
                     "question asks for (person, country, movie, year, language, ...), "
                     "derived from the question's interrogative. Re-emit the plan with it.",
        })
    ctx.plan_answer_type = _at
    # set only AFTER every rejection gate: a rejected plan must not poison the
    # immutability flag (the model's corrected re-emission must go through)
    ctx.plan_declared = True
    entities = args.get("entities") or []
    if isinstance(entities, str):
        entities = [a.strip() for a in entities.split(",") if a.strip()]
    if entities and not ctx.anchor_name:
        ctx.anchor_name = entities[0]
    # declared plan entities — the answer-time consumption gate compares
    # these against ctx.consumed_anchors (retrieval starts)
    ctx.plan_entities = [str(e).strip() for e in entities if str(e).strip()]
    # seed ALL declared entities (not just the anchor) — multi-entity questions have
    # several valid starting centers (e.g. "X and Y" → both X and Y are seedable starts).
    seed = set()
    if getattr(ctx, "anchor_idx", None) is not None:
        seed.add(ctx.anchor_idx)
    for name in entities:
        i = _name_to_idx(name, ctx)
        if i is not None:
            seed.add(i)
    ctx.subgraph_entities = seed
    flow = []
    for fid in ctx.fact_ids:
        txt = ctx.fact_texts.get(fid, "")
        inner = txt.strip().strip("()")
        parts = [p.strip() for p in inner.split("|")] if inner else []
        flow.append({"id": fid, "triple": txt,
                     "subquestion": parts[1] if len(parts) >= 2 else (txt or ctx.question)})
    # MULTI-ANCHOR REMINDER (user ruling 2026-09-01, 1171 specimen): a declared
    # entity that never LITERALLY heads a fact can only arrive as a constraint
    # tail — its own neighborhood (roster/members/siblings) never gets walked,
    # so the fact that filters on it discriminates nothing (single-candidate
    # binding + empty filter → satisficed wrong answer). One short nudge at
    # plan time (Arizona gate lesson: keep nudge text minimal).
    # Wave-1: nudge text aligned with the dual PLAN EXTEND gate — extending is
    # legal either to COVER_MISSING or to ADD_ANCHOR_VIEW, not only to cover.
    anchored = set()
    for fid in ctx.fact_ids:
        head = (ctx.fact_texts.get(fid, "").strip("()")
                .split("|") + [""])[0].strip().lower()
        if head and not head.startswith("?"):
            anchored.add(head)
    for k, v in (args or {}).items():          # sgN.anchor specs, any format
        if str(k).endswith(".anchor") and isinstance(v, str) \
                and v.strip() and not v.strip().startswith("?"):
            anchored.add(v.strip().lower())
        if isinstance(v, list):                 # structured subgraphs payload
            for sg in v:
                if isinstance(sg, dict):
                    a = str(sg.get("anchor") or "").strip()
                    if a and not a.startswith("?"):
                        anchored.add(a.lower())
    unused = [str(e).strip() for e in entities
              if str(e).strip() and str(e).strip().lower() not in anchored]
    _ma = ""
    if len(entities) >= 2 and unused:
        _ma = ("declared entity " +
               " | ".join(f"'{u}'" for u in unused[:3]) +
               " never anchors a fact — it can only filter. When its own "
               "roster/membership is the candidate pool, anchor a fact ON it "
               "([PLAN EXTEND]) and intersect; anchored retrieval "
               "discriminates better than tail-filtering. [PLAN EXTEND] is "
               "legal for either reason: COVER_MISSING (the requirement is "
               "not yet covered) or ADD_ANCHOR_VIEW (a second anchored "
               "evidence view on a DIFFERENT declared question entity — "
               "legal even when the requirement is already covered).")
    result = {
        "flow": flow, "entities": entities, "answer": args.get("answer", ""),
        "answer_type": str(args.get("answer_type", "") or "").strip(),
    }
    if _ma:
        # SEPARATE FIELD (2026-09-01: buried in the note the model ignored
        # it — 1171 V37e specimen, reminder present, plan unchanged)
        result["multi_anchor"] = _ma
    result["note"] = ("Iterative subgraph retrieval. For EACH OPEN fact: call "
                      "retrieve_relations to find question-RELEVANT relations (all that "
                      "match, typically ≤3, max 5 — not just the single best), then "
                      "retrieve_subgraph with ALL of them together. After each "
                      "retrieve_subgraph, close the fact with ONE of three checkpoints: "
                      "resolve it `- fid ✓ ?var = v1 | v2`; close it EMPTY "
                      "`- fid ✗ empty` when no candidate relation advances it; close it "
                      "MOOT `[fid ✗ moot]` when earlier evidence already bound its target. "
                      "MULTI-TREE (question names SEVERAL entities): prefer a "
                      "separate anchor view per entity when the entities provide "
                      "independent constraints, and let the system JOIN them (an "
                      "entity reachable from both sides intersects the candidate "
                      "sets); an entity that adds no independent constraint does "
                      "not need its own tree. For later facts continuing the "
                      "SAME tree, just pass the ANCHOR or any retrieved entity + the "
                      "new relation — the system walks the full chain from the root "
                      "automatically. Fact-1 center = the literal named anchor. "
                      "The system auto-penetrates CVTs.")
    return _json_result(result)


# ───────────────────────── retrieve_relations (GTE per entity+question) ─────────────────────────
# THREE-PHASE SPLIT (perf-4, 2026-08-23): _rr_prepare (A: pure CPU — entity
# resolution + memoized pool construction) → _rr_execute (B: the GTE awaits,
# batched by the round-level collector) → _rr_finalize (C: pure CPU — ranking
# merge + result rendering). The public async retrieve_relations keeps the
# sequential composition; the round-level scheduler drives the phases directly.

def _rr_prepare(args: Dict[str, Any], ctx) -> dict:
    """A段: terminal errors render here (kind="done"); an unresolvable entity
    defers to the GTE correction in execute (kind="corr"); the clean path
    emits the per-entity GTE requests (kind="rr")."""
    raw = args.get("center") or args.get("entities") or ([args.get("entity")] if args.get("entity") else [])
    entities = _split_pipe_entities([str(e) for e in raw if e])
    question = args.get("question") or args.get("subquestion") or ""
    if not entities:
        return {"kind": "done", "result": _json_result({"error": "no entity provided."})}
    # CALLING-FACT RESOLUTION: explicit sg/fact_id, else the most recently
    # born chain (rr calls usually omit `sg:`; dict insertion order = chain
    # birth order, and the loop works one fact at a time).
    _fid = str(args.get("sg") or args.get("fact_id") or args.get("step") or "")
    _fidt = getattr(ctx, "fid_tree_roots", None) or {}
    if not _fid and _fidt:
        _fid = next(reversed(list(_fidt)))
    entities, _err = _expand_entities(entities, ctx)
    if _err:
        # UNBOUND-?VAR FRONTIER FALLBACK (route 1 C): the frontier is the
        # rr pool's primary source, so an undeclared variable is no longer
        # a dead end when the calling fact's chain exists — rank from the
        # chain frontier and continue (no error round-trip to the model).
        _has_chain = any(
            _fr in (getattr(ctx, "anchor_seqs", None) or {})
            for _fk in ([_fid, _fid.split(".", 1)[0]] if _fid else [])
            for _fr in (_fidt.get(_fk) or ()))
        if not (_has_chain and os.environ.get("SEQ_REL_SEQ", "1") == "1"):
            return {"kind": "done", "result": _json_result({"error": _err})}
        entities = []
    _nudge = _variable_nudge(raw, ctx)
    # ENTITY RESOLUTION GATE (user ruling 2026-09-27: dispatch-level
    # single gate, never per-path patches — the 25_7cec crash came from
    # the frontier-pool path bypassing this check): every literal entity
    # name must resolve BEFORE any internal path (frontier-pool, direct,
    # future additions). ?var entries skip (they ride the frontier
    # fallback above). Failure → GTE correction candidates, regardless
    # of which path would have been taken.
    for e in entities:
        if e.startswith("?"):
            continue
        i = _name_to_idx(e, ctx)
        if _looks_like_value(e) and (i is None or normalize(e) != normalize(ctx.ents[i])):
            continue                                              # value-like → skip
        if i is None or _match_sim(e, ctx.ents[i]) < 0.95:
            return {"kind": "corr", "entity": e, "question": question, "nudge": _nudge}
    # PATTERN-GRAPH RR POOL (user ruling 2026-09-26, route 1 C): when the
    # calling fact's chain exists, the step-k+1 relation candidates come
    # from the step-k materialized FRONTIER — the FULL supernode endpoint
    # set (CVT supernodes included), neighborhood relations aggregated with
    # the two-hop extension. This is the PRIMARY source and it never
    # consults the model's ?var binding sets: bindings are a curated
    # subset and relations that live only on specific frontier entities
    # (25_892f: the gold TV work's runtime family hangs on a frontier
    # performance CVT the model never bound) would be missed. Unbound
    # ?vars ride the same path (frontier is the source whether or not the
    # variable was ever declared — bindings feed display/answer only).
    _root = None
    for _fk in ([_fid, _fid.split(".", 1)[0]] if _fid else []):
        for _fr in (_fidt.get(_fk) or ()):
            if _fr in (getattr(ctx, "anchor_seqs", None) or {}):
                _root = _fr
                break
        if _root is not None:
            break
    if _root is not None and os.environ.get("SEQ_REL_SEQ", "1") == "1":
        from kgqa.traversal.pattern_walk import get_pattern_index
        _ast = getattr(ctx, "anchor_seqs", None) or {}
        # RR RETRIEVAL START = the anchored step's landing set (constitution
        # B2 round-4): the model's declared entity only anchors; the GTE
        # pool starts from the step landing — never the declaration alone.
        # Falls back to the layer-completion projection when the chain has
        # no delivered landing yet (born-but-not-delivered).
        _land = (getattr(ctx, "chain_landing", None) or {}).get(_root) or {}
        _segs_l = list(_land.get("segs") or ())
        if _segs_l and _segs_l[-1]:
            _front = set(_segs_l[-1]) | set(_land.get("attrs") or frozenset())
        else:
            _front = set(_seq_completions(
                ctx, get_pattern_index(ctx), _root, _ast[_root])[-1])
        _front = set(_front) - {_root}
        _pool = _seq_pool_relids(ctx, _front) if _front else set()
        if len(_pool) < _GTE_POOL_MIN:
            _pool |= set(range(len(ctx.rels)))
        _head = (entities[0] if entities else str(ctx.ents[_root]))
        reqs = [(_head, _pool)]
        if not entities:
            entities = [str(ctx.ents[_root])]
        return {"kind": "rr", "requests": reqs, "entities": entities,
                "question": question, "nudge": _nudge}
    # FRONTIER-DEFAULT POOL (user ruling 2026-10-05): a mid-chain RR without a
    # fact id must not let the model's named entity HIJACK the candidate pool.
    # By design, every step after a route's initial anchor retrieves relations
    # from the PREVIOUS step's materialized frontier ("以某一层最后的关系实体
    # 为检索点"). The sg-attached branch above implements that when the model
    # passes a fact id; adaptive-mode models never do, so the design was dead
    # code. Here: pool = deepest-layer frontiers of ALL accumulated sequences
    # UNION the named entities' own pools — named entities are usually frontier
    # members and only ever ADD coverage, never replace the frontier. Entities
    # that fail the delivered-center gate fall through to the detailed errors
    # below instead of riding this path.
    if _root is None and os.environ.get("SEQ_REL_SEQ", "1") == "1":
        _st = getattr(ctx, "anchor_seqs", None) or {}
        if any(_layers for _layers in _st.values()):
            _nidx, _blocked = [], False
            for e in entities:
                if e.startswith("?"):
                    continue
                i = _name_to_idx(e, ctx)
                if i is None or not _in_subgraph(i, ctx):
                    _blocked = True
                    break
                _nidx.append(i)
            if not _blocked:
                from kgqa.traversal.pattern_walk import get_pattern_index
                _ix = get_pattern_index(ctx)
                _clx = getattr(ctx, "chain_landing", None) or {}
                _front = set()
                for _a, _layers in _st.items():
                    if not _layers:
                        continue
                    _land = _clx.get(_a) or {}
                    _segs_l = list(_land.get("segs") or ())
                    if _segs_l and _segs_l[-1]:
                        _front |= set(_segs_l[-1]) | set(
                            _land.get("attrs") or frozenset())
                    else:
                        _front |= set(_seq_completions(
                            ctx, _ix, _a, _layers)[-1])
                _front -= set(_st.keys())
                _pool = _seq_pool_relids(ctx, _front | set(_nidx))
                if len(_pool) >= _GTE_POOL_MIN:
                    _head = entities[0] if entities else str(ctx.ents[next(iter(_st))])
                    return {"kind": "rr", "requests": [(_head, _pool)],
                            "entities": entities, "question": question,
                            "nudge": _nudge}
    # Entity resolution: for non-resolving centers, offer GTE correction candidates
    # (non-blocking — model picks one OR continues with current evidence). For values
    # and no-candidate cases, skip (lenient — don't block the workflow).
    idxs, unresolved = [], []
    for e in entities:
        i = _name_to_idx(e, ctx)
        if _looks_like_value(e) and (i is None or normalize(e) != normalize(ctx.ents[i])):
            unresolved.append(e); continue                      # value-like → skip
        if i is None or _match_sim(e, ctx.ents[i]) < 0.95:
            return {"kind": "corr", "entity": e, "question": question, "nudge": _nudge}
        if not _in_subgraph(i, ctx):
            # Entity IS in the graph (resolved well) but wasn't retrieved in a prior
            # subgraph — DISTINCT from the resolution error above. Give feedback.
            return {"kind": "done", "result": _json_result({
                "entity_error": f"'{e}' is in the graph but was not retrieved in any "
                    "prior retrieve_subgraph.",
                "note": f"A center must come from a prior retrieve_subgraph's triples "
                    f"(or the plan anchor for fact 1). '{e}' wasn't in any prior result. "
                    "If it's a variable binding, pass the ?VARIABLE. Otherwise, retrieve "
                    "from the question's named entity first. "
                    "WORKFLOW: retrieve_relations → retrieve_subgraph."})}
        idxs.append(i)
    if not idxs:
        return {"kind": "done", "result": _json_result({
            "entities": entities, "question": question,
            "candidate_relations": [],
            "note": ((_nudge + " ") if _nudge else "") +
                    f"No valid center resolved from {entities} "
                    f"(skipped: {unresolved}). Continue with current "
                    "evidence — answer from what you have, or try a "
                    "different entity from the question / a previous "
                    "retrieve_subgraph's triples."})}
    # FIRST-FACT rr (no chain yet): per-entity pools from the anchor's own
    # neighborhood — the frontier does not exist before step-1 materializes
    # it (GTE per-entity, NOT a generic "these entities" head: a multi-entity
    # call is a variable expansion; one generic-head call on the union pool
    # loses the entity-specific ranking signal — e.g. [San Francisco Giants,
    # Crazy Crab] dropped `sports.sports_team.championships` (ranked #1 for
    # the Giants alone) out of the top-15).
    reqs = []
    for ent, i in zip(entities, idxs):
        ent_pool = _seq_pool_relids(ctx, {i})
        if len(ent_pool) < _GTE_POOL_MIN:
            ent_pool = set(range(len(ctx.rels)))
        reqs.append((ent, ent_pool))
    return {"kind": "rr", "requests": reqs, "entities": entities,
            "question": question, "nudge": _nudge}


def _rr_bridge_reserved(ctx, reqs, cands, scored, top_n):
    """CVT-BRIDGE reserved slots for the rr FIRST screen (Ethiopia specimen
    21_660138373d19bbffdd3d3f7a30234e4a, 2026-09-23). A mediation CVT adjacent
    to the center (leadership / government_position_held / …) is often the
    ONLY route from the center to the answer, but every relation crossing it —
    the center↔CVT edge (jurisdiction_of_office / governing_officials) AND the
    CVT's terminal attributes (office_holder / basic_title) — labels as
    "Ethiopia | government position held office holder | ?", a non-sequitur
    the union-rank GTE buries (specimen: office_holder #38,
    jurisdiction_of_office #51 of a 74 pool — all below the top-30 cands cut,
    so the first screen misses the whole family and the model's first
    submission can't reach the CVT). The subgraph side re-discovers them as
    relation_expansion bridges only AFTER a failed walk; these slots surface
    them up front.

    Bridge set = pool relations on edges touching a 1-hop CVT of any request
    center (the same carriers the sg-side bridge scan would attach). The GTE
    tail scores order them; relations already ranked into cands consume no
    slot. Pure CPU, bounded by the centers' adjacency."""
    if top_n <= 0:
        return []
    from kgqa.agent.tools import _full_adj
    adj = _full_adj(ctx)
    n_ents = len(ctx.ents)
    cand_set = set(cands)
    bridge = set()
    for ent, pool in reqs:
        ci = _name_to_idx(ent, ctx)
        if not (0 <= ci < len(adj)):
            continue
        for _r, nb in adj[ci]:
            if not (0 <= nb < n_ents) or not is_cvt_like(ctx.ents[nb]):
                continue
            for r2, _o in adj[nb]:
                if r2 in pool:
                    bridge.add(r2)
    scored_bridge = [(s, r) for r, s in scored.items()
                     if r in bridge and r not in cand_set]
    scored_bridge.sort(reverse=True)
    return [r for _s, r in scored_bridge[:top_n]]


async def _rr_execute(treq, ctx, session):
    """B段: the GTE correction (or the ranking awaits).

    UNION-THEN-RANK (user ruling 2026-09-14): the candidate pool is the
    UNION of all requesting entities' reachable relations, and ONE labeled
    GTE ranking orders that union by the sub-question. The old path ranked
    each entity separately (top-15 each) and concatenated in entity order —
    a relation ranked #16 for every entity never entered the union, and a
    later entity's #1 sat behind the first entity's cluster. Entity
    specificity is preserved by the candidate LABELS ("entity | relation |
    ?" — one label per relation, headed by an entity that actually reaches
    it); the Giants/Crazy Crab failure that motivated per-entity CALLS was
    a generic head, not a specific one. Also saves N-1 GTE round-trips.

    ATTRIBUTE-FIRST RANKING (user design 2026-09-08): one GTE call ranks
    BOTH the attribute names AND the full relation names together. The
    attribute ranking identifies the question's target semantic role; the
    relation ranking provides secondary ordering within each attribute
    group — no extra GTE call needed."""
    if treq["kind"] == "corr":
        cands = await _entity_correction(treq["entity"], treq["question"], ctx, session)
        return {"cands": cands}
    if os.environ.get("SEQ_RR_UNION_RANK", "1") == "1":
        from kgqa.agent.tools import _rel_last2, _TRIPLE_GTE_INSTRUCT
        from kgqa.stages.stage2_entity import gte_retrieve
        pool_ids = sorted({i for _ent, pool in treq["requests"] for i in pool})
        head_of = {}
        for ent, pool in treq["requests"]:
            for i in pool:
                head_of.setdefault(i, ent)
        pool_names = [str(ctx.rels[i]) if 0 <= i < len(ctx.rels) else ""
                      for i in pool_ids]
        labeled = [f"{head_of.get(i, '?')} | {_rel_last2(ctx.rels[i])} | ?"
                   for i in pool_ids]
        name2pos = {n: k for k, n in enumerate(pool_names) if n}
        # CVT-BRIDGE RESERVED SLOTS (Ethiopia specimen 21_6601…, 2026-09-23):
        # one extra SCORE WINDOW beyond the top-30 that feeds cands — the
        # server embeds the whole pool either way, the tail rows only score
        # the bridge candidates (below). With slots off, top_k stays 30 and
        # behavior is byte-identical to the pre-fix call.
        _slots = int(os.environ.get("SEQ_RR_BRIDGE_SLOTS", "3") or 0)
        rows = await gte_retrieve(session, treq["question"], pool_names,
                                  candidate_texts=labeled,
                                  top_k=30 + (30 if _slots > 0 else 0),
                                  instruct=_TRIPLE_GTE_INSTRUCT)
        cands, scored = [], {}
        for r in rows or []:
            pos = r.get("index")
            try:
                pos = int(pos)
            except (TypeError, ValueError):
                pos = None
            if not (isinstance(pos, int) and 0 <= pos < len(pool_ids)):
                pos = name2pos.get(r.get("candidate"))
            if isinstance(pos, int) and 0 <= pos < len(pool_ids):
                full = pool_ids[pos]
                if full not in scored and r.get("score") is not None:
                    scored[full] = float(r.get("score") or 0.0)
                if len(cands) < 30 and full not in cands:
                    cands.append(full)
        reserved = (_rr_bridge_reserved(ctx, treq["requests"], cands, scored,
                                        _slots) if _slots > 0 else [])
    else:
        cands, seen = [], set()
        reserved = []
        for ent, pool in treq["requests"]:
            ranked = await _gte_for_triple(ctx, session, ent, treq["question"], "",
                                           pool_relids=pool)
            for r in ranked:
                if r not in seen:
                    seen.add(r); cands.append(r)
    # GLOBAL TOP-K FLOOR (user ruling 2026-10-06, batch 2): a small
    # question-ranked slice of the FULL case relation set rides every RR.
    # Local pools starve when the previous step's landing composition
    # shifts (892f: a character-heavy step-1 pushed the runtime family out
    # while the works were already delivered). Local and global stay
    # SEPARATE in the face; relations already shown locally are dropped.
    _gk = int(os.environ.get("SEQ_RR_GLOBAL_TOPK", "5") or 0)
    global_top = []
    if _gk > 0:
        try:
            from kgqa.stages.stage2_entity import gte_retrieve
            from kgqa.agent.tools import _TRIPLE_GTE_INSTRUCT
            _local = set(cands) | set(reserved)
            _idx_of = {str(r): i for i, r in enumerate(ctx.rels)}
            _rest = [str(ctx.rels[i]) for i in range(len(ctx.rels))
                     if i not in _local][:1000]
            if _rest:
                _grow = await gte_retrieve(session, treq["question"], _rest,
                                           top_k=max(_gk * 4, 20),
                                           instruct=_TRIPLE_GTE_INSTRUCT)
                for r in _grow or []:
                    i = _idx_of.get(r.get("candidate"))
                    if i is not None and i not in _local:
                        global_top.append(i)
                        _local.add(i)
                    if len(global_top) >= _gk:
                        break
        except Exception:
            global_top = []
    # ATTRIBUTE-FIRST: derive attribute names from the ranked relations and
    # rank them in the SAME GTE call (add as extra candidates — piggyback,
    # no second round-trip). We use the /retrieve endpoint directly since
    # _gte_for_triple works on indices, not names.
    try:
        from kgqa.stages.stage2_entity import gte_retrieve, GTE_INSTRUCT
        # cands are relation INDICES — convert to full names first
        pool_rel_names = [str(ctx.rels[i]) for i in cands
                          if 0 <= i < len(ctx.rels)]
        attr_names = sorted({r.rsplit(".", 1)[-1] for r in pool_rel_names
                             if "." in r})
        # GLOBAL SEMANTIC MERGE (user ruling 2026-09-14, deflator specimen):
        # re-rank the WHOLE per-entity union, not the first 15 by entity
        # order. cands is concatenated in ENTITY order (per-entity GTE
        # scores are dropped at _gte_for_triple's boundary), so [:15] is
        # the first entity's cluster — a later entity's #1 (Monaco's
        # gdp_deflator_change, GTE 0.51, union position ~29) never entered
        # the re-rank and the menu showed the first entity's generic
        # relations. Re-ranking all union members by the SAME question
        # restores semantic order for every entity's contributions.
        if os.environ.get("SEQ_RR_GLOBAL_MERGE", "1") == "1":
            merged_pool = pool_rel_names[:int(os.environ.get(
                "SEQ_RR_MERGE_CAP", "80"))]
        else:
            merged_pool = pool_rel_names[:15]
        combined = attr_names + merged_pool
        rows = await gte_retrieve(session, treq["question"], combined,
                                  top_k=len(combined), instruct=GTE_INSTRUCT)
        ranked_combined = [r.get("candidate") for r in (rows or [])
                           if r.get("candidate") in combined]
        attr_set = set(attr_names)
        attr_ranked = [c for c in ranked_combined if c in attr_set]
        rel_ranked = [c for c in ranked_combined if c not in attr_set]
        attr_ranked += [a for a in attr_names if a not in set(attr_ranked)]
        rel_ranked += [r for r in merged_pool if r not in set(rel_ranked)]
        return {"cands": cands, "attr_ranked": attr_ranked,
                "rel_ranked": rel_ranked, "reserved": reserved,
                "global_top": global_top}
    except Exception:
        return {"cands": cands, "reserved": reserved,
                "global_top": global_top}


def _rr_finalize(treq, bres, ctx) -> str:
    """C段: render the correction / ranking result (pure CPU)."""
    if treq["kind"] == "corr":
        e = treq["entity"]
        cands = bres["cands"]
        if cands:
            return _json_result({
                "entity_error": f"'{e}' is not a confident graph entity.",
                "candidates": cands,
                "note": "Pick the correct entity from `candidates` (each shows "
                        "NEIGHBOR relations to disambiguate) and RE-CALL with THAT "
                        "name (a different name passes the repeat gate). If NONE of "
                        "the candidates is the right entity, do NOT re-call with the "
                        "same name (it will be rejected) — stop pursuing this step "
                        "(an optional `- fid ✗ empty` note suffices) or continue from "
                        "another question entity's evidence. CROSS-SUBGRAPH CENTER: if "
                        "THIS fact's selected relations are right but its anchor "
                        "entity is wrong, re-call retrieve_subgraph with a center "
                        "BORROWED from any prior subgraph's evidence (e.g. an "
                        "earlier fact's binding) plus this fact's relations — the "
                        "boundary check accepts any retrieved entity as a center."})
        # No candidates found — GTE searched and found nothing confident.
        # Give the model FEEDBACK (not silent skip) so it can adjust.
        return _json_result({
            "entity_error": f"'{e}' was not found in the graph and no similar "
                "entities were detected by the system's search.",
            "note": f"The system searched for entities matching '{e}' but found "
                "nothing confident. This entity may not exist in this graph snapshot, "
                "or its name may be very different. Try a different entity from the "
                "question, or if this is a later fact, pick an entity from a previous "
                "retrieve_subgraph's triples. "
                "WORKFLOW: retrieve_relations → retrieve_subgraph."})
    _nudge = treq["nudge"]
    cands = bres["cands"]
    # TYPE+ATTRIBUTE GROUPED DISPLAY (user audit 2026-09-09): grouping by
    # the last component alone collapsed semantically different relations
    # (film.producer.film and film.director.film both showed as 'film';
    # division/facility/league/location all as 'teams') and the model could
    # neither see nor express the from/to difference. Group key is now the
    # TYPE+ATTRIBUTE (last TWO components — freebase domain.type.attribute),
    # so each group is one semantic relation family. Ordering stays
    # attribute-first: the GTE attribute ranking orders groups; within a
    # group, relation GTE rank orders members.
    attr_ranked = bres.get("attr_ranked") or []
    rel_ranked = bres.get("rel_ranked") or []
    _ATTR_GROUP_THRESHOLD = 5
    if attr_ranked and rel_ranked:
        from collections import defaultdict as _dd

        def _typed(r: str) -> str:
            return ".".join(r.rsplit(".", 2)[-2:])

        groups = _dd(list)          # typed key -> [full rel names, GTE-ordered]
        rel_set = set(rel_ranked)
        for r in rel_ranked:
            groups[_typed(r)].append(r)
        # also include relations from the original cands that GTE didn't rank
        # (cands are indices — convert, fill the tail of each group)
        for i in cands[:30]:
            if not isinstance(i, int) or not (0 <= i < len(ctx.rels)):
                continue
            r = str(ctx.rels[i])
            if r not in rel_set and "." in r:
                t = _typed(r)
                if t in groups:
                    groups[t].append(r)
        lines = []
        # ORDERING (user ruling 2026-09-09): ① attributes by GTE rank;
        # ② within an attribute, typed groups sorted by type+attribute key
        # (deterministic — dict insertion order varied with GTE rel rank).
        for attr in attr_ranked:    # attribute GTE rank orders the sections
            for t in sorted(t for t in groups
                            if groups[t] and t.rsplit(".", 1)[-1] == attr):
                members = groups[t]
                shown = members[:_ATTR_GROUP_THRESHOLD]
                more = (f" …(+{len(members)-_ATTR_GROUP_THRESHOLD})"
                        if len(members) > _ATTR_GROUP_THRESHOLD else "")
                # SEMANTIC-EQUIVALENCE MARKER (user ruling 2026-09-15):
                # same domain.type prefix + same attribute base = likely
                # semantic equivalents (adjoins ≡ adjoin_s) — signal "these
                # belong in ONE submission, not split across calls"
                _pref = t.rsplit(".", 1)[0]
                _equiv = [m for m in shown if m.rsplit(".", 1)[0] == _pref]
                _mark = " ≡" if len(_equiv) > 1 else ""
                lines.append(f"{t} ← {', '.join(shown)}{more}{_mark}")
                if len(lines) >= 12:
                    break
            if len(lines) >= 12:
                break
        if lines:
            _grouped = "\n".join(f"  {ln}" for ln in lines)
            _apos = {a: i for i, a in enumerate(attr_ranked)}

            def _order(rn: str):
                t = _typed(rn)
                return (_apos.get(t.rsplit(".", 1)[-1], 999), t)

            _flat = sorted((str(ctx.rels[i]) for i in cands[:30]
                            if isinstance(i, int) and 0 <= i < len(ctx.rels)),
                           key=_order)[:15]
            # CVT-BRIDGE RESERVED SLOTS (see _rr_bridge_reserved): guaranteed
            # first-screen presence regardless of the second re-rank's
            # opinion — appended AFTER the 15-cut, never displacing a ranked
            # candidate. ⚑ tells the model these are the center's CVT
            # mediator crossings: submit them WITH the family they carry.
            _res = [str(ctx.rels[i]) for i in (bres.get("reserved") or [])
                    if isinstance(i, int) and 0 <= i < len(ctx.rels)
                    and str(ctx.rels[i]) not in _flat]
            if _res:
                _flat += _res
                _grouped += ("\n  ⚑ cvt-bridge (reserved — paths through the "
                             "center's CVT mediators; carry the answer behind "
                             "them): " + ", ".join(_res))
            # GLOBAL FLOOR RENDER (user ruling 2026-10-06): separate face
            # field + inclusion in the submit-able candidate list. These
            # matched the QUESTION across ALL case relations — the local
            # neighborhood may not reach them (892f specimen).
            _gtop = [str(ctx.rels[i]) for i in (bres.get("global_top") or [])
                     if isinstance(i, int) and 0 <= i < len(ctx.rels)
                     and str(ctx.rels[i]) not in _flat]
            if _gtop:
                _flat += _gtop
            if _res:
                # reserved cvt-bridge slots: marker folds into the note
                # (grouped_relations removed — it duplicated
                # candidate_relations; the grouping was a legacy GTE
                # optimization with little value, user ruling 2026-09-23)
                _bridge_note = ("  ⚑ the last relations listed are reserved "
                                "cvt-bridge slots (paths through the "
                                "center's CVT mediators).")
            else:
                _bridge_note = ""
            return _json_result({
                "entities": treq["entities"], "question": treq["question"],
                "candidate_relations": _flat,
                **({"global_relations": _gtop} if _gtop else {}),
                "advice": ("Check semantic applicability group by group, "
                         "then submit every group that passes. Do not "
                         "perform a second best-or-backup selection. "
                         "Submit the TYPED NAME (e.g. relations: "
                         "baseball_division.teams | producer.film) — it "
                         "expands to that group's relations. A BARE "
                         "attribute (relations: teams) expands to ALL "
                         "relations with that attribute across types. "
                         "After execution, a sparse result is not a "
                         "reason to add more relations for the same step. "
                         + (_nudge + " " if _nudge else "")),
            })
    # FALLBACK: no attribute ranking (GTE failure) — flat list as before
    # THREE-WAY RELATION CLASSIFICATION (user + Codex design, 2026-09-03):
    # selection = uncertainty management, not hard filtering. The model
    # classifies candidates Required (fact cannot be evidenced without one
    # of them) / Supporting (alternative or complementary encodings of the
    # SAME fact — equivalent, inverse, sibling schema) / Irrelevant, and
    # the retrieve_subgraph `relations:` field carries Required ∪ Supporting
    # (hard cap 10). Call-level data: live hard selection captured gold 58%
    # per call; three-way recovers 62% of the missed golds at ×4 set size.
    _note = ("Identify the encodings of the CURRENT step — typically "
             "≤3 core relations, max 5. Do NOT pick only the single best "
             "name: relation semantics are unstable, so include EVERY "
             "semantically equivalent encoding of the SAME step "
             "(film.director.film and film.film.directed_by are the same "
             "fact from two ends; every release-date form is the same "
             "attribute) — never discard an equivalent merely because its "
             "name differs. Reject relations that merely share vocabulary "
             "or topic. Then call retrieve_subgraph with the step's "
             "equivalent, inverse and sibling encodings together.")
    if _nudge:
        _note = _nudge + " " + _note
    cand_rel_names = [ctx.rels[i] for i in cands if 0 <= i < len(ctx.rels)]
    # CVT-BRIDGE reserved slots — same guaranteed presence as the grouped path
    cand_rel_names += [str(ctx.rels[i]) for i in (bres.get("reserved") or [])
                       if isinstance(i, int) and 0 <= i < len(ctx.rels)
                       and str(ctx.rels[i]) not in {str(n) for n in cand_rel_names}]
    return _json_result({
        "entities": treq["entities"], "question": treq["question"],
        "candidate_relations": cand_rel_names,
        "advice": _note,
    })


async def retrieve_relations(args: Dict[str, Any], ctx, session) -> str:
    """Subgraph relation retrieval: entities + question → candidate relations.
    Accepts a single entity or a LIST of entities. When multiple entities are passed,
    the GTE candidate pool is the UNION of all entities' 2-hop reachable relations —
    ensuring relations visible from ANY candidate are surfaced (not just the first).
    Sequential composition of the three phases (see _rr_prepare)."""
    from kgqa.agent.seq_adaptive import enabled as _adaptive_enabled
    if _adaptive_enabled(ctx):
        return await dispatch("retrieve_relations", args, ctx, session)
    treq = _rr_prepare(args, ctx)
    if treq["kind"] == "done":
        return treq["result"]
    bres = await _rr_execute(treq, ctx, session)
    return _rr_finalize(treq, bres, ctx)


_TIME_VALUE_ATTRS = {"from", "to", "date", "year", "rate", "number", "amount",
                     "start_date", "end_date", "initial_date", "final_date"}

def _relation_snapshot(ctx, center_idxs, cand_rel_idxs, top_k: int = 8) -> str:
    """For each top candidate relation, what 2nd-hop attribute relations does it expose
    via 1-hop + CVT penetration from the centers? Returns a hint string tagging candidates
    that reach TIME/VALUE dims (from/to/date/rate/number). Only names relations, not entities.
    `center_idxs`: the entity idxs this call is centered on. `cand_rel_idxs`: GTE-ranked
    candidate relation idxs (ordered)."""
    if not center_idxs or not cand_rel_idxs:
        return ""
    centers = set(center_idxs)
    ents = ctx.ents
    rels = ctx.rels
    H, R, T = ctx.h_ids, ctx.r_ids, ctx.t_ids
    # pre-index edges by head and tail for speed
    out_by_h = {}
    in_by_t = {}
    for i in range(len(H)):
        out_by_h.setdefault(H[i], []).append((R[i], T[i]))
        in_by_t.setdefault(T[i], []).append((R[i], H[i]))
    def _short(ri):
        s = str(rels[ri]) if 0 <= ri < len(rels) else ""
        return s.rsplit(".", 1)[-1] if s else s
    lines = []
    for ri in cand_rel_idxs[:top_k]:
        # 1-hop neighbors of centers via this relation (both directions)
        nbs = set()
        for c in centers:
            for r, t in out_by_h.get(c, []):
                if r == ri:
                    nbs.add(t)
            for r, h in in_by_t.get(c, []):
                if r == ri:
                    nbs.add(h)
        if not nbs:
            continue
        # 2nd-hop relations from those neighbors (collect short-names)
        second = set()
        for nb in list(nbs)[:6]:
            for r, t in out_by_h.get(nb, []):
                second.add(_short(r))
            for r, h in in_by_t.get(nb, []):
                second.add(_short(r))
        exposed_time = sorted(second & _TIME_VALUE_ATTRS)
        if exposed_time:
            lines.append("%s →exposes[%s]" % (_short(ri), ",".join(exposed_time)))
    if not lines:
        return ""
    return ("Relation 2nd-hop hints (which candidates reach time/value dims like from/to/date): "
            + " | ".join(lines) + ".")


# ───────────────────────── retrieve_subgraph (mature multi-hop walk + dense tree) ─────────────────────────
# THREE-PHASE SPLIT (perf-4, 2026-08-23): _sg_prepare (A: center resolution +
# boundary checks + walk step construction) → _sg_execute (B: the walk await,
# batched by the round-level coordinator; GTE correction for bad names first)
# → _sg_finalize (C: accumulate + render + evidence sets, pure CPU).


# ───────────────────────── RELATION-SEQUENCE STATE (SEQ_REL_SEQ, user design 2026-09-15) ─────────────────────────
# The authoritative walk state is the ANCHOR + its accumulated relation LAYERS
# (ctx.anchor_seqs: {anchor_idx: [frozenset(rel_idx), ...]}). A later
# retrieve_subgraph CENTERED ON THE SAME ANCHOR appends: each submitted
# relation joins the deepest layer whose completions make it structurally
# feasible (continuation-first; anchor-direct feasible = layer-1 widening).
# The walk then instantiates the FULL sequence from the anchor — typed entity
# lists never become walk centers (the runtime/tvrage specimens' rosters
# dropped exactly the gold; constructive instantiation cannot). Gate is OFF by
# default: same-anchor call semantics change (repair re-selection becomes
# union-accumulate), so it ships dark until the 48×3 verdict flips it.

def _anchor_seq_layers(ctx) -> dict:
    st = getattr(ctx, "anchor_seqs", None)
    if st is None:
        st = {}
        ctx.anchor_seqs = st
    return st


from kgqa.agent.entity_kinds import (
    is_cvt as _ek_is_cvt, is_transit as _ek_is_transit, entity_kind as _ek_kind)


def _cvt_like_name(n) -> bool:
    """C1 single-source: delegate to entity_kinds (the authoritative
    four-kind classifier) — this file's old inline copy was one of five
    divergent CVT predicates."""
    return _ek_is_cvt(n)


def _supernode_step(ctx, ix, cur, rels) -> set:
    """PATTERN-LAYER BASE STEP (user ruling 2026-09-26, route 1): ONE
    kind-agnostic hop over the layer's relation set — every landed entity
    (named, CVT, id-node alike) is a supernode member of the layer's
    frontier. No CVT/ named distinction, no transparent pass-through: a
    mediator hop is an ordinary hop and must be carried by the pattern's
    own relation sequence. This is the ONLY stepping the decision layer
    (completions / feasibility / frontier pools) consumes."""
    nxt = set()
    for r in rels:
        f, rv = ix.fwd.get(r, {}), ix.rev.get(r, {})
        for e in cur:
            nxt.update(f.get(e, ()))
            nxt.update(rv.get(e, ()))
    return nxt


def _trans_named_step(ctx, ix, cur, rels) -> set:
    """DISPLAY PROJECTION ONLY (route 1): the retired transparent step —
    landed CVT/id-nodes pass through their edges to NAMED endpoints. The
    decision layer no longer consumes this (see _supernode_step); it
    survives solely to keep display echoes (anchor_sequence completion
    counts) byte-stable against the pre-migration surface."""
    from kgqa.agent.tools import _full_adj
    adj = _full_adj(ctx)
    ents = ctx.ents
    nxt = set()
    for r in rels:
        f, rv = ix.fwd.get(r, {}), ix.rev.get(r, {})
        for e in cur:
            nxt.update(f.get(e, ()))
            nxt.update(rv.get(e, ()))
    def _cvtl(i):
        return _cvt_like_name(ents[i] if 0 <= i < len(ents) else i)
    named = {n for n in nxt if not _cvtl(n)}
    stack = [n for n in nxt if _cvtl(n)]
    seen = set(nxt) | set(cur)
    while stack:
        n = stack.pop()
        for _rr, o in (adj[n] if 0 <= n < len(adj) else ()):
            if o in seen:
                continue
            seen.add(o)
            if _cvtl(o):
                stack.append(o)
            else:
                named.add(o)
    return named


def _seq_completions(ctx, ix, anchor_idx, layers, named_only=False):
    """[ {anchor}, supernode completions through L1, ..., through Lk ] — one
    kind-agnostic layer step each (see _supernode_step; route 1 base).
    named_only=True keeps the pre-migration DISPLAY projection (NAMED
    endpoints through the transparent step) for the anchor_sequence echo —
    the decision layer never passes it. Memoized per (anchor, layers,
    mode); deterministic, so the memo is always valid.

    PREFIX REUSE (walk speedup, user request 2026-09-17): completion set k
    depends only on (anchor, layers[:k]) — an EXTEND (new tail layer) or
    UPDATE (replace layer j) invalidates only the SUFFIX, but the old
    whole-tuple key forced a full-chain rewalk on every layer change.
    Every computed prefix is cached with its cross-layer visited set, so
    continuation calls walk only the layers they actually changed."""
    memo = getattr(ctx, "anchor_seq_memo", None)
    if memo is None:
        memo = {}
        ctx.anchor_seq_memo = memo
    lkey = tuple(frozenset(l) for l in layers)
    key = (anchor_idx, lkey, bool(named_only))
    hit = memo.get(key)
    if hit is not None:
        return hit[0]
    _step = _trans_named_step if named_only else _supernode_step
    # longest cached prefix (values: (out_list, visited_set)). A stored
    # value must have k+1 entries (anchor + k walked layers) — a CHAINED-OFF
    # full key (the chain broke at layer j) also matches layer-j prefix
    # lookups by tuple equality but has FEWER entries; reusing it would
    # resume walking from the wrong layer (equivalence harness catch).
    out, visited, k = None, None, 0
    for k in range(len(layers) - 1, 0, -1):
        phit = memo.get((anchor_idx, lkey[:k], bool(named_only)))
        if phit is not None and len(phit[0]) == k + 1:
            out, visited = list(phit[0]), set(phit[1])
            break
    if out is None:
        out, visited = [{anchor_idx}], {anchor_idx}
        k = 0
    cur = out[-1]
    for j in range(k, len(layers)):
        nxt = _step(ctx, ix, cur, layers[j]) - visited
        if not nxt:
            break
        visited |= nxt
        out.append(nxt)
        cur = nxt
        # out has m entries ⇔ layers[:m-1] walked — cache that prefix
        memo[(anchor_idx, lkey[:len(out) - 1], bool(named_only))] = (
            list(out), set(visited))
    # the full key shares the OUT OBJECT itself (aliasing contract:
    # repeat calls return the same object — test_memo_returns_same_object)
    memo[key] = (out, visited)
    return out


def _feasible_rels_one_hop(ctx, ent_set) -> set:
    """Relation idxs with at least one edge touching ent_set (either
    direction) — the structural half of the layer-assignment judgment."""
    from kgqa.agent.tools import _full_adj
    adj = _full_adj(ctx)
    out = set()
    for e in ent_set:
        if 0 <= e < len(adj):
            for rr, _o in adj[e]:
                out.add(rr)
    return out


def _classify_seq_submit(ctx, ix, anchor_idx, layers, new_rel_idxs):
    """Deepest-feasible-layer assignment for appended relations.
    Returns (updated_layers, {rel_idx: action}, completions). Actions:
    'append' (new layer beyond the last — frontier relations submitted
    TOGETHER share that one new layer), 'join:k' (added to existing layer
    k), 'infeasible' (no layer's completions reach it). Two-pass: every
    relation is assigned against the PRE-update completions, then applied —
    mutating mid-loop would both index past feas and split same-call
    frontier siblings into separate layers."""
    comps = _seq_completions(ctx, ix, anchor_idx, layers)
    feas = [_feasible_rels_one_hop(ctx, comps[k]) for k in range(len(comps))]
    layers = [set(l) for l in layers]
    n0 = len(layers)
    assign = {}
    for r in new_rel_idxs:
        join_at = None
        for k in range(len(feas) - 1, -1, -1):      # deepest first
            if r in feas[k]:
                join_at = k + 1
                break
        assign[r] = join_at
    actions = {}
    for r in new_rel_idxs:
        join_at = assign[r]
        if join_at is None:
            actions[r] = "infeasible"
            continue
        if join_at > n0:
            if n0 >= 3:                             # depth cap = derive's
                actions[r] = "depth_cap"
                continue
            if len(layers) == n0:                   # first frontier rel
                layers.append(set())
            layers[-1].add(r)
            actions[r] = "append"
        else:
            layers[join_at - 1].add(r)
            actions[r] = f"join:{join_at}"
    return [frozenset(l) for l in layers], actions, comps


from kgqa.agent.tools import _full_adj as _full_adj_imp


def _chain_feasible(ctx, ix, anchor_idx, chain) -> bool:
    """RETIRED SHELL (walk-bridge unification 2026-10-10): per-leg
    realization — direct channel plus the step-junction bridge — now
    lives inside _patterns_from_layers, and feasibility IS the
    construction (a leg with neither channel kills the combo). Kept as
    a thin direct-walk predicate for external probes/tests that only
    need "is this chain directly walkable" (no bridge)."""
    cur = {anchor_idx}
    for r in chain:
        cur = _supernode_step(ctx, ix, cur, [r])
        if not cur:
            return False
    return True


def _patterns_from_layers(ctx, ix, anchor_idx, layers, submitted):
    """Cross-product of the declared layers into multistep pattern tuples.
    PRIORITY QUEUE (user ruling 2026-10-06/R4): the CURRENT submission's
    relations sort to the FRONT of each layer; historical relations fill
    remaining slots. Budget trimming cuts from the BACK (old relations
    first) — a truncated relation stays in the layer and the environment,
    it just doesn't participate in THIS pattern derivation. This guarantees
    a selectable+submitted relation is never silently starved by history.

    WALK-BRIDGE UNIFICATION (user ruling 2026-10-10,
    specs/walk_bridge_unification_2026-10-10.md): combos are realized LEG
    BY LEG from the anchor — every step junction (incl. the entry
    junction) independently takes the DIRECT channel (the leg's relation
    attaches to the supernode) or the BRIDGE channel: one hop through any
    relation to mediators, then the submitted relation — direction
    variants accepted at the second sub-hop via _ctx_inverse_rels
    (bridging is STEP-JUNCTION semantics; 直挂则免). Feasibility is the
    construction itself (a leg with neither channel kills the combo) —
    the retired _chain_feasible mirror is absorbed here. _rebuild_paths
    stays STRICT: bridged legs are inlined as ordinary hops. Branches are
    bounded on the RELATION dimension (distinct bridge relations), under
    the same 24 budget + SEQ_CHAIN_TOPK — no new parameters.

    Returns (patterns, realizes): patterns = {pk: hops-tuple} for the
    execution lane (realized hops carry ACTUAL relation indices — a
    bridged leg is (rb, actual_terminal)); realizes = {pk: ((leg_idx,
    submitted_r, channel), ...)} — the NATIVE step attribution consumers
    read instead of matching names."""
    from itertools import product as _prod
    from kgqa.agent.tools import _full_adj
    _sub = set(submitted)
    # priority sort: submitted relations first (ascending), then historical
    # (ascending) — deterministic, no reliance on relation index magnitude
    lay = [sorted(l, key=lambda r: (0 if r in _sub else 1, r))
           for l in layers if l]
    if not lay:
        return {}, {}
    widths = [len(l) for l in lay]
    n = 1
    for w in widths:
        n *= w
    while n > 24:                     # trim widest layer from the BACK
        k = widths.index(max(widths))
        if widths[k] <= 1:
            break
        widths[k] -= 1
        lay[k] = lay[k][:widths[k]]   # front-priority: submitted kept
        n = 1
        for w in widths:
            n *= w
    _POOL_CAP = 24                    # same budget as the combo trim above
    n_ents = len(ctx.ents)
    _adj = _full_adj(ctx)
    _fold = _ctx_inverse_rels(ctx)
    _var_cache = {}

    def _variants(r):
        """r plus its inverse-folded variant indices — direction variants
        ARE different relations, but the walk may realize a leg under the
        graph's complementary encoding (fold at the second sub-hop)."""
        if r in _var_cache:
            return _var_cache[r]
        vs = [r]
        rn = str(ctx.rels[r]) if 0 <= r < len(ctx.rels) else str(r)
        rep = _fold.get(rn)
        if rep is not None:
            for i, nm in enumerate(ctx.rels):
                if i != r and _fold.get(str(nm), str(nm)) == rep:
                    vs.append(i)
        _var_cache[r] = tuple(vs)
        return _var_cache[r]

    def _bridge_cands(cur, r):
        """STEP-JUNCTION BRIDGE, relation-dim candidates: from supernode
        `cur`, one hop through ANY relation to mediators that carry r or
        a fold variant. Returns {rb: mediator set} — the branch count is
        the number of DISTINCT rb (user ruling 2026-10-10: the derivation
        layer records relations, not entities, so the bound is relational)."""
        vs = _variants(r)
        cands = {}
        for e in sorted(cur):
            if not (0 <= e < len(_adj)):
                continue
            for rb, m in _adj[e]:
                if m == e or not (0 <= m < n_ents):
                    continue
                for v in vs:
                    if m in ix.fwd.get(v, {}) or m in ix.rev.get(v, {}):
                        cands.setdefault(rb, set()).add(m)
                        break
        return cands

    out, realizes = {}, {}
    for combo in _prod(*lay):
        if combo[-1] not in submitted:
            continue
        # leg-by-leg realization; the frontier list branches ONLY where a
        # leg needs the bridge (direct attachment emits a single state)
        frontier = [((), {anchor_idx}, ())]
        for k, r in enumerate(combo):
            nxt = []
            for hops, cur, legs in frontier:
                step = _supernode_step(ctx, ix, cur, [r])
                if step:
                    nxt.append((hops + (r,), step,
                                legs + ((k, r, "direct"),)))
                    continue
                for rb, mids in sorted(_bridge_cands(cur, r).items()):
                    for v in _variants(r):
                        vstep = _supernode_step(ctx, ix, mids, [v])
                        if vstep:
                            nxt.append((hops + (rb, v), vstep,
                                        legs + ((k, r, "bridge"),)))
                            break
            if not nxt:
                frontier = []
                break
            # per-leg flood control, deterministic (shortest hops first)
            frontier = sorted(nxt, key=lambda x: (len(x[0]), x[0]))
            frontier = frontier[:_POOL_CAP]
        for hops, _cur, legs in frontier:
            if hops and hops not in realizes:
                realizes[hops] = legs
                out[hops] = tuple(frozenset([h]) for h in hops)
    if len(out) > _POOL_CAP:
        _ranked = sorted(out, key=lambda pk: (len(pk), pk))[:_POOL_CAP]
        out = {k: out[k] for k in _ranked}
        realizes = {k: realizes[k] for k in _ranked}
    # TOP-K CHAIN SELECTION (user ruling 2026-09-15, 24353bbc specimen):
    # the second subgraph's patterns are the OVERALL top-K chains, not the
    # full layer cross-product — L1(3 rels) × L2(3 rels) = 9 sections was
    # the explosion the trajectory review caught. Rank: (a) shorter chains
    # first (fewer intermediates = tighter evidence); (b) deterministic
    # (rel-idx sorted). K=3 matches the derive path's per-terminal quota.
    _K_CHAINS = int(os.environ.get("SEQ_CHAIN_TOPK", "3"))
    if len(out) > _K_CHAINS:
        _ranked = sorted(out, key=lambda pk: (len(pk), pk))
        out = {k: out[k] for k in _ranked[:_K_CHAINS]}
        realizes = {k: realizes[k] for k in out}
    return out, realizes


def _sg_prepare(args: Dict[str, Any], ctx) -> dict:
    """A段: terminal errors render here (kind="done"); a wrong/low-conf center
    name defers its GTE correction to execute (corr field — when the
    correction yields NO candidates the call still walks the resolved
    centers); the clean path emits the walk steps."""
    raw = args.get("center") or args.get("entities") or ([args.get("entity")] if args.get("entity") else [])
    entities = _split_pipe_entities([str(e) for e in raw if e])
    rel_names = args.get("relations") or []
    fid = str(args.get("sg") or args.get("fact_id") or args.get("step") or "")
    # single-?var call marker for the PATTERN-PATH walk (seq_tools B段):
    # the expansion produced the binding set; _sg_execute may replace the
    # per-binding walks with one set-state pattern walk
    _var_name = ""
    _raw0 = str(raw[0]).strip() if raw else ""
    if _raw0.startswith("?") and len([e for e in raw if e]) == 1:
        _var_name = _raw0
    if not entities:
        return {"kind": "done", "result": _json_result({"error": "no entities provided."})}
    # PATTERN-PREFIX CONTINUATION (user design 2026-09-10): a single-?var
    # call whose variable was declared from a prior single-center retrieval
    # carries that walk's node set as the prefix — the bindings are the
    # prior pattern's tails, so their walks mask the already-walked
    # territory (状态保持,不重复回去) and spend the budget on the novel fringe.
    _prefix = None
    _raw0 = str(raw[0]).strip() if raw else ""
    if (_raw0.startswith("?") and len(entities) == 1
            and os.environ.get("SEQ_PATTERN_PREFIX", "0") == "1"):
        # OFF by default — A/B 2026-09-10: RPE's dominant cost is each
        # binding's 3-hop NOVEL fringe (never in the prefix), so the mask
        # saved only the hop-1 back-edges while cutting legitimate paths
        # through shared territory: speed flat (exec 2408 vs 2152s), f1
        # -2.4pp (0.6930 -> 0.6692). Kept behind the env for the record.
        _pst = getattr(ctx, "pattern_state", None) or {}
        _org = _pst.get(_raw0)
        if _org and _org.get("nodes"):
            _prefix = _org["nodes"]
    entities, _err = _expand_entities(entities, ctx)
    _var_frontier_root = None
    if _err:
        # UNBOUND-?VAR FRONTIER FALLBACK (user ruling 2026-09-27; same rule
        # as the rr side): an undeclared variable is NOT an error when the
        # calling fact's chain exists — the center only indicates the STEP,
        # so the call extends the chain from its root. The error round-trip
        # burned a turn the budget later needed (1797 msg8 specimen). No
        # chain yet → keep the declare-first nudge unchanged.
        if os.environ.get("SEQ_REL_SEQ", "1") == "1" and fid:
            _ft = getattr(ctx, "fid_tree_roots", None) or {}
            for _fk in (fid, fid.split(".", 1)[0]):
                for _fr in (_ft.get(_fk) or ()):
                    if _fr in (getattr(ctx, "anchor_seqs", None) or {}):
                        _var_frontier_root = _fr
                        break
                if _var_frontier_root is not None:
                    break
        if _var_frontier_root is None:
            return {"kind": "done", "result": _json_result({"error": _err})}
        entities = [str(ctx.ents[_var_frontier_root])]
    _nudge = _variable_nudge(raw, ctx)
    # ATTRIBUTE-FAMILY pre-check: if any submitted name lacks dots, it's an
    # attribute — skip the early validation (the full-name check would reject
    # it here); the expansion after center resolution handles it
    _relset = {str(r) for r in ctx.rels}

    def _is_family(rs: str) -> bool:
        # family name = bare attribute ('actor') OR typed name that is not a
        # full relation ('baseball_division.teams' — the rr display's group key)
        return bool(rs) and ("." not in rs or rs not in _relset)

    _has_attr_name = any(_is_family(str(r).strip()) for r in rel_names)
    if not _has_attr_name:
        rel_idxs = [ctx.rels.index(r) for r in rel_names
                    if isinstance(r, str) and r in ctx.rels]
        if not rel_idxs:
            return {"kind": "done", "result": _json_result({"error": "no valid relations provided. Pick from the candidate_relations "
                                          "returned by retrieve_relations."})}
    else:
        rel_idxs = []  # will be populated after attribute expansion

    # resolve + boundary-check each center. value-like inputs require EXACT match (fuzzy
    # resolves unreliably: 'UTC-05:00' -> 'UTC−04:00'); skip entities not yet in the
    # subgraph (boundary); flag wrong/low-conf named entities → correction.
    centers, skipped = [], []
    bad_name = None                     # an entity whose NAME is wrong/low-conf → correct
    for e in entities:
        i = _name_to_idx(e, ctx)
        if _looks_like_value(e) and (i is None or normalize(e) != normalize(ctx.ents[i])):
            # Values (numbers, UTC offsets, IDs) have no semantic name → GTE returns random
            # entities. Skip GTE; tell the model directly.
            return {"kind": "done", "result": _json_result({
                "entity_error": f"'{e}' is a VALUE (numeric/coordinate/date/ID), not a named "
                    "graph entity — it's stored as an ATTRIBUTE, not a center. Pass the ENTITY "
                    "that carries it, or the ?variable. WORKFLOW: retrieve_relations → retrieve_subgraph."})}
        if i is None or _match_sim(e, ctx.ents[i]) < 0.95:
            if bad_name is None:
                bad_name = e            # no-match or substring-fragment match → correct
            if i is None:
                continue
            # low-conf match present but flagged → still try it as a center below if in-subgraph
        if i is not None and not _in_subgraph(i, ctx):
            skipped.append(e)           # in graph but not yet retrieved → boundary skip
        elif i is not None:
            centers.append((e, i))
    if bad_name is None and not centers:
        return {"kind": "done", "result": _json_result({"error": (f"none of {entities} appeared in a prior retrieve_subgraph. "
                                       f"A center MUST be an entity from a previous retrieve_subgraph's "
                                       f"triples (or the plan anchor for fact 1) — ANY prior subgraph's "
                                       f"entities are legal centers (cross-subgraph borrowing). If any of "
                                       f"these is a variable binding, pass the ?VARIABLE "
                                       f"(center: [\"?var\"]) — the runtime expands it to all declared "
                                       f"bindings.")})}
    # snapshot the PRIOR calls' triples BEFORE this call's _accumulate (which
    # lands in finalize), so the cycle filter suppresses cross-call loops
    # (subgraph N→subgraph 1) but never this call's own freshly-retrieved edges.
    # CONSUMPTION TRACK (user ruling 2026-09-01): record every resolved
    # retrieval START (literal or ?var-expanded) — the answer-time gate
    # intercepts once when a declared plan entity never started anything.
    # ATTRIBUTE-FAMILY EXPANSION (user design 2026-09-08): a submitted
    # family name — BARE attribute ('actor', 'film') or TYPED group name
    # ('baseball_division.teams', the rr display's group key, user audit
    # 2026-09-09) — expands to relations that can serve as a path edge from
    # the centers. MUST run after center resolution: expanding from the full
    # graph picks unreachable relations and the walk dies as
    # RELATION_MISMATCH (30 vs 18 in the first rollout).
    # CVT BRIDGE (user ruling 2026-09-09, position rule): expansion is
    # POSITIONAL, not hop-based — a CVT as the last or second-to-last path
    # entity is expanded; the 2-hop budget governs relation SELECTION only,
    # and a CVT mid-path stretches the window by one edge. The family's
    # terminal edge (film.performance.actor) or its behind-entity
    # (people.person.religion) never touches the center, so the walk needs
    # the BRIDGE (center→CVT) submitted alongside — the existing penetration
    # then reveals the terminal edge. Matching the terminal edge alone walked
    # ZERO edges (sg 707 vs 2884 chars, attrfamily2): the bridge is what
    # makes a family walkable.
    # UNIFIED EXPANSION (typedgroup audit 2026-09-09): every submitted
    # relation — bare attribute, typed group, OR full name — goes through
    # the same pool-match + bridge attachment. Bridges are a reachability
    # property of (center, relation), not of the name form: the model
    # copying a full terminal name from the display walked ZERO edges
    # exactly like a bare family did before bridges existed (29/35
    # mismatches in the typedgroup rollout were full-name submissions).
    _fam_echo = {}
    if centers and rel_names:
        from kgqa.agent.tools import _full_adj
        _adj = _full_adj_imp(ctx)
        _n_ents = len(ctx.ents)
        _lmemo = getattr(ctx, "_rel_last_memo", None)
        if _lmemo is None:
            _lmemo = ctx._rel_last_memo = {}
        _fmemo = getattr(ctx, "_fam_expand_memo", None)
        if _fmemo is None:
            _fmemo = ctx._fam_expand_memo = {}
        _ckey = tuple(sorted(_ci for _cn, _ci in centers))

        def _last(ri):
            lc = _lmemo.get(ri)
            if lc is None:
                lc = _lmemo[ri] = (str(ctx.rels[ri]).rsplit(".", 1)[-1]
                                   if 0 <= ri < len(ctx.rels) else "?")
            return lc

        def _typed(ri):
            rn = str(ctx.rels[ri]) if 0 <= ri < len(ctx.rels) else "?"
            return ".".join(rn.rsplit(".", 2)[-2:])

        _expanded = []
        _fam0 = ""
        for r in rel_names:
            rs = str(r).strip()
            if not rs:
                continue
            _fam = _is_family(rs)
            if _fam:
                _fam0 = _fam0 or rs
            _typedq = _fam and "." in rs   # typed query: match last TWO comps
            _mk = (_ckey, rs)
            if _mk in _fmemo:
                _names, _bids = _fmemo[_mk]
            else:
                # MATCHED SET for this name — WALK-PORTS ALIGNED (user
                # ruling 2026-10-09, "relation retrieval follows the
                # pathway FRONTIER, never the declared center alone"): the
                # pool comes from kgqa.agent.walk_ports.pool_for_center —
                # frontier pool of the declared entity's chain ∪ its own
                # neighborhood (members add coverage). Same source as the
                # RR menu, so the gate can no longer veto a relation the
                # menu itself offered.
                from kgqa.agent.walk_ports import pool_for_center
                _pool = set()
                for _cn, _ci in centers:
                    _pool |= pool_for_center(ctx, _ci)
                if _fam:
                    _names = sorted(ri for ri in _pool
                                    if ((_typed(ri) == rs) if _typedq
                                        else (_last(ri) == rs)))[:10]
                else:
                    _ri = ctx.rels.index(rs)
                    _names = [_ri] if _ri in _pool else []
                # BRIDGE part (adjacency): the center→carrier edge when the
                # carrier — the terminal edge's own CVT, or a holder behind
                # it — touches the matched set. Terminal pool matches never
                # touch the center (hop-1 death); the bridge is what makes
                # them walkable. Bridges are a reachability property of the
                # (center, relation) pair, NOT of the name form — a FULL
                # name copied verbatim from the display gets the same
                # bridges as its typed/bare family (typedgroup rollout:
                # 29/35 mismatches were full-name submissions bypassing
                # this).
                # DIRECT-FIRST (user audit 2026-09-10, Belgium specimen):
                # bridges exist to REACH a family the center cannot touch
                # at hop 1. When a center ALREADY carries a matched
                # relation on its own edges, bridges for this name are
                # pure noise — the behind-CVT carrier test is loose (any
                # co-endpoint touching the family fires), so a hub center
                # surrounded by CVTs whose co-endpoints are big entities
                # (Belgium: adjoin/combat/partial-containment CVTs whose
                # co-endpoints all carry countries.continent) pulled a
                # dozen irrelevant relations into rel_idxs and the render
                # labeled them "(retrieved)" — evidence unrelated to the
                # asked relations. Skip bridges whenever ANY center has a
                # direct matched edge.
                _mset = set(_names)
                _bids = set()
                # A/B verdict 2026-09-10: skipping bridges when the center
                # carries the family directly cleans the render (Belgium's
                # adjoin/combat sections) but costs -4.4pp — the bridged
                # walks' penetrations feed discrimination evidence. Gated
                # OFF by default; SEQ_DIRECT_FIRST=1 enables.
                _center_direct = False
                if os.environ.get("SEQ_DIRECT_FIRST", "0") == "1":
                    _center_direct = any(
                        ri in _mset
                        for _cn, _ci in centers
                        if 0 <= _ci < len(_adj)
                        for ri, _ni in _adj[_ci])
                if not _center_direct:
                    for _cn, _ci in centers:
                        if not (0 <= _ci < len(_adj)):
                            continue
                        _scanned = set()
                        for _ri, _ni in _adj[_ci]:
                            if (_ni in _scanned or not (0 <= _ni < _n_ents)
                                    or len(_bids) >= 6):
                                continue
                            _scanned.add(_ni)
                            _hit = any(_rj in _mset
                                       for _rj, _nj in _adj[_ni] if _nj != _ci)
                            if not _hit and is_cvt_like(ctx.ents[_ni]):
                                for _rj, _mj in _adj[_ni]:
                                    if _mj == _ci or not (0 <= _mj < _n_ents):
                                        continue
                                    if any(_rk in _mset
                                           for _rk, _nk in _adj[_mj]):
                                        _hit = True
                                        break
                            if _hit:
                                _bids.add(_ri)
                _fmemo[_mk] = (_names, _bids)
            _name_set = set(_names)
            _dnames = [str(ctx.rels[i]) for i in _names]
            _bridges = [str(ctx.rels[i]) for i in sorted(_bids)
                        if i not in _name_set][:6]
            if _fam or _bridges:
                # echo every family expansion; for full names echo only when
                # bridges were attached (the surprising, audit-worthy case)
                _fam_echo[rs] = {"direct": _dnames or [rs], "bridge": _bridges}
            _expanded.extend(_dnames if _dnames else [rs])
            # BRIDGES NEVER ENTER rel_idxs (user ruling 2026-09-23): a bridge
            # is a PATTERN-PATH mid-segment relation only. The old
            # SEQ_BRIDGE_TERMINAL=1 injection (2026-09-11, measured on the
            # pre-rebuild walk engine as the mismatch fix) put bridges into
            # the submitted-relation set, where the rebuild era's lane-2
            # walk consumed them as frontier one-hop termini — re-admitting
            # already-delivered facts under their inverse relation id (1731
            # specimen: Raised-on-Radio row). Reach on the rebuild lane is
            # carried by the mseq pattern hops (the bridge IS hop-1 of the
            # derived chain), so the injection has no remaining consumer.
            # The expansion echo still records bridges for audit.
        rel_names = list(dict.fromkeys(_expanded))
        rel_idxs = [ctx.rels.index(r) for r in rel_names
                    if isinstance(r, str) and r in ctx.rels]
        if not rel_idxs:
            # HALLUCINATION DETECTION (user ruling 2026-10-09): the
            # submitted relation is not in ANY pool — the center's, the
            # frontier's, or the case graph's. The model invented it
            # from world knowledge. Never execute: point out the
            # hallucination and redirect to RR or answer mode.
            _all_case_rels = set(str(r) for r in ctx.rels)
            _any_known = any(r in _all_case_rels for r in rel_names)
            if _any_known:
                return {"kind": "done", "result": _json_result({
                    "error": f"HALLUCINATED_RELATION: the relation "
                             f"'{_fam0 or (rel_names[0] if rel_names else '?')}' "
                             f"exists in the graph but is NOT reachable from "
                             f"your declared center or its pathway frontier. "
                             f"It was not offered by any retrieve_relations "
                             f"result. Use retrieve_relations from the correct "
                             f"step's frontier to find valid encodings, or "
                             f"answer from the already-delivered evidence."})}
            else:
                return {"kind": "done", "result": _json_result({
                    "error": f"HALLUCINATED_RELATION: the relation "
                             f"'{_fam0 or (rel_names[0] if rel_names else '?')}' "
                             f"does not exist anywhere in this case's graph. "
                             f"The model appears to have invented it. Use "
                             f"retrieve_relations to find actual available "
                             f"relations, or answer from already-delivered "
                             f"evidence."})}

    _cons = getattr(ctx, "consumed_anchors", None)
    if _cons is None:
        _cons = ctx.consumed_anchors = set()
    for _cn, _ci in centers:
        if _cn:
            _cons.add(str(_cn).strip().lower())
    # MULTI-STEP PATTERN DERIVATION (realignment spec pillar 1+2+3): for
    # centers WITHOUT direct 1-hop family support, derive the relation
    # SEQUENCE center→…→family-terminal (BFS ≤2 named hops, CVT-free) —
    # submitted to the walk as multi-step step_relations so the engine's
    # forward validation enforces path consistency and reach stops needing
    # bridges-as-termini.
    _multistep = {}
    _ms_cov = {}                 # (center_idx, pattern) -> step coverage
    _ast = {}                    # anchor tree layers (guarded: REL_SEQ may be off)
    _root_of = {}
    # UNCONDITIONAL INIT (crash fix 2026-09-19): the return at the end of
    # _sg_prepare references these unconditionally; they used to be born
    # inside the SEQ_MULTISTEP block, so SEQ_MULTISTEP=0 (e.g. replay
    # without the standing env) raised UnboundLocalError on EVERY sg call.
    _seq_echo = ""
    _cont_compare = False
    _n_resolved = len(centers)   # pre-collapse resolved-center count (note)
    _scope_ids = sorted({ci for _n, ci in centers if ci is not None})
    _cont_frontier = {}
    _infeas_map = {}
    _prefix_map = {}   # chain → the walk segment list for THIS submission
    _pure_repeat_served = False   # R1-scope init (unconditional)
    _layer_action = ""
    _seq_touched = set()   # roots this call actually edited (F1) — MUST be
                            # at function scope (R1: inside the multistep
                            # conditional it's UnboundLocalError when off)
    if centers and os.environ.get("SEQ_MULTISTEP", "1") == "1":
        try:
            from kgqa.traversal.pattern_walk import get_pattern_index
            _ix = get_pattern_index(ctx)
            # PATTERN FAMILY = the SUBMITTED relations, resolved by name
            # (user ruling 2026-09-22, design realignment): single-hop
            # paths are DIRECTLY PERCEIVABLE — a full submitted name
            # resolves to itself, and the derivation enumerates 1..3-hop
            # realizable sequences ending in a submitted relation, ranked
            # by (length, semantics). The expansion's direct/bridge BUCKETS
            # are name-resolution + reachability machinery for the WALK
            # layer; they must never feed the pattern layer (the 567
            # specimen: 'film.director.film | film.film.directed_by'
            # expanded to a 7-relation pool whose bridge relations became
            # pattern terminals → 60 enumerated patterns, 20+ rendered —
            # vs the design's direct + ≤3/terminal ≈ a handful).
            # Resolution precedence: (1) exact full name; (2) family/bare
            # name → the expansion's direct members (the bucket's intended
            # use — typed groups and bare attributes have no exact match);
            # (3) short-name equivalence. Bridges are never terminals.
            _matched = set()
            _l1 = lambda rn: str(rn).rsplit(".", 1)[-1]
            _resolved_any = False
            for _rs in [str(r) for r in (args.get("relations") or [])]:
                for _ri, _rn in enumerate(ctx.rels):
                    if str(_rn) == _rs:
                        _matched.add(_ri)
                        _resolved_any = True
                        break
                else:
                    _exp = (_fam_echo or {}).get(_rs) or {}
                    _got = False
                    for _dn in (_exp.get("direct") or []):
                        if _dn in ctx.rels:
                            _matched.add(ctx.rels.index(_dn))
                            _got = _resolved_any = True
                    if not _got:
                        for _ri, _rn in enumerate(ctx.rels):
                            if _l1(_rn) == _rs or _l1(_rs) == _l1(_rn):
                                _matched.add(_ri)
                                _resolved_any = True
                                break
            _fam = _matched if _resolved_any else set(rel_idxs)
            # RELATION-SEQUENCE ACCUMULATION — ROUTE 1 ATTRIBUTION (user
            # ruling 2026-09-26): a center's ONLY walk-side meaning is the
            # STEP it indicates. Attribution NEVER inspects center entity
            # identity — no completion-set membership, no root preference:
            # every resolved center of a call routes to the CALLING FACT's
            # chain (fid_tree_roots), and a fact with no chain yet BIRTHS
            # one at its centers (the fact-1 anchor call). The walk always
            # re-runs from the chain root with the submitted relations;
            # non-attaching inputs are not a class anymore. Render scope is
            # preserved by the downstream scope filter (chains must pass
            # through the resolved center set — A-line lesson from the
            # -4.6pp rollback), and the COMPARE note reads the PRE-collapse
            # resolved-center count (n_resolved).
            _declared = {}
            _seq_echo = ""      # (re-initialized at function scope — see above)
            _cont_compare = False
            _cont_frontier = {}
            _layer_action = ""
            _layer_ops = os.environ.get("SEQ_LAYER_OPS", "1") == "1"
            if os.environ.get("SEQ_REL_SEQ", "1") == "1" and rel_idxs:
                _ast = _anchor_seq_layers(ctx)
                # FID→CHAIN REGISTRY: chain roots by birth fid (raw + dot
                # prefix). Routed centers map here; the registry is the
                # single attribution source.
                _fidt = getattr(ctx, "fid_tree_roots", None)
                if _fidt is None:
                    _fidt = {}
                    ctx.fid_tree_roots = _fidt
                _fid_roots = []
                for _fk in ([fid, fid.split(".", 1)[0]] if fid else []):
                    for _fr in (_fidt.get(_fk) or ()):
                        if _fr in _ast and _fr not in _fid_roots:
                            _fid_roots.append(_fr)
                _root_of = {}
                _anchor_step = {}   # chain → anchored step (0 = the head)
                _center_layer = {}  # RETIRED from decisions (kept for echo)
                # ATTRIBUTION (constitution B2 rounds 3-4 + root-prefix fix,
                # 2026-10-06): the center anchors the chain whose segment
                # LANDING (per-segment, via the prefix-depth table "pref")
                # or CVT-attribute endpoints contains it; center == chain
                # head anchors step 0. Tie on two chains → trajectory
                # recency. No landing → the calling fact's chain; none →
                # birth. The anchored step is the WALK PREFIX depth for
                # the new segment: a root re-exploration (center == head)
                # walks from the head unconstrained by earlier segments —
                # the 452 specimen (place_lived from Carmelo after the
                # sibling segment died to the joint-command filter).
                _cl = getattr(ctx, "chain_landing", None) or {}
                _ct = getattr(ctx, "chain_touch", None) or {}
                for _cn, _ci in centers:
                    if not (0 <= _ci < len(ctx.ents)):
                        continue
                    _hits = []
                    for _a in _ast:
                        if _ci == _a:
                            _hits.append((_a, 0))
                            continue
                        _land = _cl.get(_a) or {}
                        _segs_l = list(_land.get("segs") or ())
                        _pref_l = list(_land.get("pref") or
                                       list(range(len(_segs_l))))
                        # SAME-LAYER REPEAT (user ruling 2026-10-09, 212
                        # specimen): a center that IS a segment's declared
                        # start re-walks that segment's layer as a SIBLING
                        # — anchored at the segment's own PREFIX DEPTH
                        # (walk = same parent prefix + the new relation
                        # group), NOT appended at the chain tail. Latest
                        # matching segment wins.
                        _starts_l = list(_land.get("starts") or ())
                        for _k in range(len(_starts_l) - 1, -1, -1):
                            if _ci in (_starts_l[_k] or frozenset()):
                                _hits.append(
                                    (_a, _pref_l[_k] if _k < len(_pref_l)
                                     else _k))
                                break
                        else:
                            # LATEST segment first (user mechanism ③:
                            # multiple hits → the most recent)
                            for _k in range(len(_segs_l) - 1, -1, -1):
                                if _ci in _segs_l[_k]:
                                    _hits.append((_a, _pref_l[_k] + 1))
                                    break
                            else:
                                if _ci in (_land.get("attrs") or frozenset()):
                                    _hits.append((_a, len(_segs_l)))
                    if len(_hits) > 1:
                        _hits.sort(key=lambda x: -(_ct.get(x[0], -1)))
                    if _hits:
                        _a0, _s0 = _hits[0]
                        _root_of[_ci] = _a0
                        _anchor_step[_a0] = min(_anchor_step.get(_a0, 99), _s0)
                        continue
                    if _fid_roots:
                        _astats = getattr(ctx, "_anchor_stats", None)
                        if _astats is not None:
                            _astats["fid_fallback"] = _astats.get(
                                "fid_fallback", 0) + 1
                        _root_of[_ci] = _fid_roots[0]
                        continue
                    _astats = getattr(ctx, "_anchor_stats", None)
                    if _astats is not None:
                        _astats["birth"] = _astats.get("birth", 0) + 1
                    _ast[_ci] = [frozenset(rel_idxs)]    # chain birth
                    _seq_touched.add(_ci)
                    if fid:
                        for _fk in dict.fromkeys([fid, fid.split(".", 1)[0]]):
                            _rts = _fidt.setdefault(_fk, [])
                            if _ci not in _rts:
                                _rts.append(_ci)
                if _root_of:
                    # continuation of an existing chain — collapse the call
                    # onto the chain root (the walk re-runs from there).
                    _cont_compare = True
                    _newc, _seen = [], set()
                    for _cn, _ci in centers:
                        _r = _root_of.get(_ci, _ci)
                        if _r in _seen:
                            continue
                        _seen.add(_r)
                        _newc.append((str(ctx.ents[_r]), _r))
                    centers = _newc
                for _a in sorted(set(_root_of.values())):
                    _layers = _ast[_a]
                    _have = set().union(*_layers) if _layers else set()
                    _new = [r for r in rel_idxs if r not in _have]
                    if not _new and _layer_ops:
                        _layer_action = "repeat"
                        # PURE-REPEAT RULING (user 2026-10-09): a fully
                        # repeated command never calls the environment.
                        # A repeat of a command that FAILED with
                        # NO_EVIDENCE is an error to point out; a repeat
                        # of a served one reports already-retrieved.
                        _fs = (getattr(ctx, "_failed_steps", None) or {})
                        _rn = frozenset(str(r) for r in rel_names)
                        if any(_rn == fs
                               for fss in _fs.values() for fs in fss):
                            return {"kind": "done", "result": (
                                "error: REPEATED_FAILED_COMMAND: this exact "
                                "relation group was already executed and "
                                "delivered nothing (NO_EVIDENCE). Repeating "
                                "it cannot add information — change the "
                                "relations or the step, or move to answer "
                                "mode.\nrelations: "
                                + " | ".join(rel_names)), "tool": "retrieve_subgraph"}
                        _pure_repeat_served = True
                        continue
                    if not _new:
                        # pre-ops pure repeat — same ruling as above
                        _fs = (getattr(ctx, "_failed_steps", None) or {})
                        _rn = frozenset(str(r) for r in rel_names)
                        if any(_rn == fs
                               for fss in _fs.values() for fs in fss):
                            return {"kind": "done", "result": (
                                "error: REPEATED_FAILED_COMMAND: this exact "
                                "relation group was already executed and "
                                "delivered nothing (NO_EVIDENCE). Repeating "
                                "it cannot add information — change the "
                                "relations or the step, or move to answer "
                                "mode.\nrelations: "
                                + " | ".join(rel_names)), "tool": "retrieve_subgraph"}
                        _pure_repeat_served = True
                        continue    # pure repeat → already-retrieved reply

                    # APPEND-ONLY SEGMENTS (constitution B2, batch 1): the
                    # submission is the command's NEW segment — a
                    # terminal-retrieval relation group appended after the
                    # ANCHORED STEP. The walk prefix = commands[:anchor
                    # step]: a root re-exploration (anchor step 0) walks
                    # from the head unconstrained by earlier segments;
                    # a landing continuation joins the prefix chain.
                    _astep = _anchor_step.get(_a, len(_layers))
                    _up = list(_layers) + [frozenset(_new)]
                    _layer_action = "extend segment %d" % len(_up)
                    _ast[_a] = _up
                    _seq_touched.add(_a)
                    # PREFIX DEPTH rides the landing record (linear
                    # default; this call's segment records its own anchor).
                    # STARTS: the DECLARED centers this segment was walked
                    # from — the same-layer anchoring key (user ruling
                    # 2026-10-09: a repeat on a segment's own start centers
                    # re-walks THAT layer, not the chain tail).
                    _land = (_cl.get(_a) or {})
                    _segs_l = list(_land.get("segs") or ())
                    _pref_l = list(_land.get("pref") or
                                   list(range(len(_segs_l))))
                    _starts_l = list(_land.get("starts") or ())
                    while len(_starts_l) < len(_layers):
                        _starts_l.append(frozenset())
                    _starts_l = _starts_l[:len(_layers)]
                    _starts_l.append(frozenset(
                        [_ci2 for _ci2, _r2 in _root_of.items()
                         if _r2 == _a] or [_a]))
                    _pref_l = _pref_l[:len(_up) - 1] + [_astep]
                    _cl[_a] = {"segs": tuple(_segs_l),
                               "attrs": _land.get("attrs") or frozenset(),
                               "pref": tuple(_pref_l),
                               "starts": tuple(_starts_l)}
                    if _segs_l and _segs_l[-1]:
                        _fr = sorted(_segs_l[-1] - {_a})[:12]
                        if _fr:
                            _cont_frontier[_a] = _fr
                    _walk_segs = list(_layers[:_astep]) + [frozenset(_new)]
                    _prefix_map[_a] = _walk_segs
                    _pats_d, _pats_real = _patterns_from_layers(
                        ctx, _ix, _a, _walk_segs, set(rel_idxs))
                    if _pats_d:
                        _declared[_a] = _pats_d
                    # L0-5 COVERAGE (replaces the classifier's
                    # "infeasible" tag): a submitted relation is covered
                    # when some surviving pattern's LAST LEG realizes it —
                    # read from the NATIVE realizes tag (walk-bridge
                    # unification 2026-10-10), never from name matching:
                    # a bridged leg may end in a fold variant's index.
                    # Uncovered relations (no direct AND no bridge within
                    # the per-junction 2-hop design) ride the wider FREE
                    # DERIVATION below.
                    _cov = {_legs[-1][1] for _legs in _pats_real.values()
                            if _legs}
                    _unc = {r for r in _new if r not in _cov}
                    if _unc:
                        _infeas_map[_a] = _unc
                    # ECHO shows the COMMAND whenever a segment is appended
                    # (CVT-mediated steps derive their patterns instead of
                    # declaring them — the echo must not depend on _pats_d);
                    # counts read the SAME landing record as the decision
                    # layer (B3: one projection).
                    _short = lambda ri: ".".join(
                        str(ctx.rels[ri]).rsplit(".", 2)[-2:]) \
                        if 0 <= ri < len(ctx.rels) else str(ri)
                    _parts = [str(ctx.ents[_a])]
                    for _li, _lay in enumerate(_up):
                        _lbl = " | ".join(_short(r) for r in sorted(_lay))
                        _cnt = len(_segs_l[_li]) \
                            if _li < len(_segs_l) else 0
                        _parts.append(f"{_lbl} ({_cnt})")
                    _seq_echo = " ⭢ ".join(_parts)
            # F1 scoped-publish basis: _seq_touched rides the return dict
            # SEMANTIC IDEMPOTENCE (user ruling 2026-09-15, trajectory
            # review): the model re-called the same walk with surface-
            # different args (typed roster → ?country → short rel name →
            # full rel name) four times, each producing the same evidence.
            # The repeat gate catches exact-match only. Check the RESOLVED
            # (root, relation-set, COMMAND SEGMENTS) — batch 1 adds the
            # segment tuple so the same relation at a DIFFERENT segment
            # position (a genuinely new command) is no longer swallowed.
            _sem_key = tuple(sorted(
                (ci, frozenset(rel_idxs),
                 tuple(tuple(sorted(s)) for s in (_ast.get(ci) or ())))
                for _n, ci in centers))
            _served = getattr(ctx, "_sg_served", None)
            if _served is None:
                _served = {}
                ctx._sg_served = _served
            _hit = _served.get(_sem_key)
            if _hit and os.environ.get("SEQ_SEM_IDEMPOTENT", "1") == "1":
                # KEY-ORDER FIX (user review 2026-09-22, repeat-audit
                # finding): the check used to read the centers tuple's
                # FIRST element (the NAME) while the mark read the SECOND
                # (the INDEX) — str vs int keys never matched, so the
                # idempotency cache was structurally dead. Both sides now
                # key on the center INDEX (canonical across surface forms).
                # RESPONSE (user ruling): do NOT re-paste cached evidence —
                # a byte-identical re-call is information-less (its scoring
                # block is removed at evaluation time); one concise
                # submission-first note is the whole reply.
                _rep = {
                    "evidence_repeat": True,
                    "note": ("This retrieval is IDENTICAL to your previous "
                             "call (same centers + same resolved relations — "
                             "surface-form changes do not alter it) and "
                             "returns nothing new. The evidence from that "
                             "call is already in your context. NEXT, one of: "
                             "(1) discriminate the candidates using the "
                             "values/dates/symbols already displayed; (2) "
                             "submit the answer from the standing bindings; "
                             "(3) pick a DIFFERENT relation or move to the "
                             "next fact.")}
                return {"kind": "done", "result": _json_result(_rep)}
            for _cn, _ci in centers:
                if not (0 <= _ci < len(ctx.ents)):
                    continue
                # CHAIN-CONSTRAINED (user ruling 2026-09-15, trajectory
                # review): when a sequence exists the second subgraph is a
                # RE-WALK of the full chain — patterns must go THROUGH the
                # accumulated layers, satisfying the entire relation chain.
                # Free derivation (which produced military_combatant→co2
                # chains that bypass L1) only applies when NO sequence
                # exists. The declared-only chains now survive because the
                # render pipeline fixes (collapse matching, out-and-back
                # guard with node-return, section-scoped attrs) landed.
                # DERIVE ALWAYS (user audit 2026-09-12, Belgium/GMT specimen):
                # the top-K PATTERN PATHS ending in the submitted relation —
                # applies to first-call anchors only.
                # TREE-KEYED RESOLUTION (user ruling 2026-09-23, design
                # realignment): _declared/_infeas_map are keyed on the TREE
                # ROOT — a continuation whose center is a FRONTIER entity
                # (Andorra, a movie) used to miss both lookups and fall into
                # per-center free derivation (the execute truncation: the
                # layer's re-walk never ran; 537/241 specimens). The center
                # confirms RELATIONS only; the walk starts at the root.
                _ci_r = _root_of.get(_ci, _ci) if isinstance(_root_of, dict) else _ci
                if _ci_r in _declared:
                    _multistep[_ci_r] = _declared[_ci_r]
                    _inf = _infeas_map.get(_ci_r)
                    if _inf:
                        # BRIDGE-LEGAL FALLBACK (user ruling 2026-09-23): the
                        # direct-unreachable submissions ride FREE DERIVATION
                        # — 1..3-hop patterns ENDING in the submitted relation
                        # (the design's "two hops, submitted relation last"
                        # semantics) — sharing the render with the declared
                        # chain instead of being silently dropped.
                        try:
                            _der_i, _ = _derive_multistep_seq(
                                _ix, ctx, _ci_r, _inf, topk=0)
                        except Exception as _exc:
                            # R2b (Codex audit v2): the bridge fallback
                            # must NOT silently drop submitted relations —
                            # if derivation fails, report it so the model
                            # knows which encoding was not covered.
                            _der_i = None
                            import sys as _sys
                            print(f"[BRIDGE-DERIVE-FAIL] rels={_inf}: "
                                  f"{type(_exc).__name__}: {_exc}",
                                  file=_sys.stderr)
                        if _der_i:
                            _multistep[_ci_r].update(_der_i)
                    continue
                _sem = os.environ.get("SEQ_PAT_SEMANTIC", "1") == "1"
                # STEP-COVERAGE (user ruling 2026-09-22): the ranking's
                # PRIMARY key is how many of the anchor tree's accumulated
                # step relation-sets the pattern path hits — a pattern
                # covering step1 AND step2 outranks one covering only
                # step2; semantics rank WITHIN equal coverage. Hit count
                # counts STEPS, never entities.
                _steps_now = None
                _root = _root_of.get(_ci, _ci) if isinstance(_root_of, dict) else _ci
                _steps_now = (_prefix_map.get(_root)
                              or (_ast.get(_root) if isinstance(_ast, dict)
                                  else None))
                # ROOT-ANCHORED DERIVATION (user ruling 2026-10-06, the
                # 567 repro): bridge patterns derive FROM THE CHAIN ROOT
                # under the walk prefix — never from the declared center.
                # A landing center is a mid-node; deriving from it produced
                # "middle-node bridging" that the command model forbids.
                _der, _der_cov = _derive_multistep_seq(
                    _ix, ctx, _root, _fam, topk=0 if _sem else 3,
                    steps=_steps_now)
                if _der:
                    _multistep[_root] = _der
                    for _pk, _cv in (_der_cov or {}).items():
                        _ms_cov[(_root, _pk)] = _cv
        except Exception as _exc:
            # F9 (user ruling 2026-10-06): "selectable relations are always
            # walkable" — pattern construction cannot legitimately fail. A
            # failure here is an internal bug; propagate it (the adapter's
            # EXECUTION error path reports it with evidence retained).
            raise RuntimeError(
                f"PATTERN_CONSTRUCTION: {type(_exc).__name__}: {_exc}. "
                "This is an internal error — the relations were selectable "
                "and therefore walkable; pattern derivation must not fail."
            ) from _exc
    # PURE-REPEAT SHORT-CIRCUIT (user ruling 2026-10-09): every matched
    # chain already carries this exact relation group and nothing new was
    # submitted — the environment is not called; the reply states the
    # commands were already retrieved and the records are fixed.
    if _pure_repeat_served and not _prefix_map and not _multistep:
        return {"kind": "done",
                "_seq_touched": set(),
                "result": (
            "unchanged_query: these relations were already retrieved for "
            "this step — the records are FIXED. No new walk was executed. "
            "Infer and answer from the delivered evidence; only a "
            "still-unexecuted step or a concrete repair justifies a "
            "different relation or center.\nrelations: "
            + " | ".join(rel_names)
            + "\nprogress: unchanged_query"), "tool": "retrieve_subgraph"}
    return {"kind": "sg", "corr": bad_name, "centers": centers, "skipped": skipped,
            "_seq_touched": _seq_touched,
            "rel_idxs": rel_idxs, "rel_names": rel_names, "fid": fid,
            "fid_c": str(args.get("_canon_fid") or ""),
            "entities": entities, "nudge": _nudge, "attr_expansion": _fam_echo,
            "prefix": _prefix, "var_name": _var_name, "multistep": _multistep,
            "multistep_cov": _ms_cov,
            "anchor_seq": _seq_echo, "cont_compare": _cont_compare,
            "cont_frontier": _cont_frontier, "layer_action": _layer_action,
            "root_of": dict(_root_of), "_prefix_map": _prefix_map,
            "n_resolved": _n_resolved, "scope_ids": list(_scope_ids),
            "prior": set(getattr(ctx, "accumulated_triples", set()) or set())}


def _derive_multistep_seq(ix, ctx, ci, fam_idxs, topk=3, steps=None):
    """PATTERN-LEVEL enumeration (user design 2026-09-11, corrected): distinct
    relation-sequences (r1, …, rk→family) from ci — pure relation pairs, not
    entity BFS. One hop = the pattern set; GTE/support ranks within the same
    hop count; top-K patterns survive. Each pattern = multi-step
    step_relations for the engine's forward validation.

    Cost: relation-set operations, milliseconds — no entity-path enumeration
    (the entity-BFS version exploded exponentially on wide hop sets).
    Returns {pattern_key: tuple(frozenset, ...)} — multiple patterns, the
    caller submits each as a separate step."""
    from collections import defaultdict
    from kgqa.agent.tools import _full_adj
    n = len(ctx.ents)
    # PER-NODE INVERTED ADJACENCY (walk speedup, user request 2026-09-17):
    # the O(|rels|) full-relation scans below (hop1/_reach each enumerated
    # every relation dict per node) dominated the walk profile — 64% of
    # walk time on dense cases. _full_adj is the same edge set as
    # (fwd, rev) indexed by node, so the scans become O(deg(node)).
    # KIND-AGNOSTIC (route 1, user ruling 2026-09-26): every neighbor is a
    # supernode member — no named filter, no CVT behind-set. Mediator hops
    # are enumerated as their own pattern hops (depth 1..3).
    _adj_all = _full_adj(ctx)

    # local adjacency (center's 1-hop, kind-agnostic)
    hop1 = defaultdict(set)          # rel_idx -> {entity idxs reached}
    if 0 <= ci < n:
        for r, t2 in _adj_all[ci]:
            if 0 <= t2 < n:
                hop1[r].add(t2)
    if not hop1:
        return None
    # PATTERN ENUMERATION, ANY HOP DEPTH 1..3 (user ruling 2026-09-13):
    # patterns are pure RELATION sequences ending in a submitted relation —
    # 1-hop (the direct pattern, forward+reverse), 2-hop, 3-hop. Enumeration
    # is CONSTRUCTIVE from index reachability: every yielded pattern has at
    # least one entity-level instantiation by construction (顺延 is inherent
    # — selection ranks only realizable patterns and fills the quota down
    # the ranking). Fan-out counts are never tracked (support is not in the
    # design); same-rel out-and-back steps are skipped (trivial loops).
    fams = fam_idxs
    patterns = set()

    # depth 1: the direct pattern
    for f in fams:
        if (((ix.head_mask.get(f, 0) >> ci) & 1)
                or ((ix.tail_mask.get(f, 0) >> ci) & 1)):
            patterns.add((f,))

    # reach helper for one more hop from a node set — kind-agnostic
    # (per-node inverted adjacency, same O(deg) rewrite as hop1 above)
    def _reach(nodes):
        out = defaultdict(set)          # rel -> {idx}
        for x in nodes:
            if not (0 <= x < n):
                continue
            for r2, o in _adj_all[x]:
                out[r2].add(o)
        return out

    def _fam_incident(node):
        return any(f in fams and (node in ix.fwd[f] or node in ix.rev[f])
                   for f in fams)

    # TYPE-POINTER RELATIONS NEVER TRANSIT (user audit 2026-09-24, 2540
    # per-pattern diagnosis): notable_types / characters_with_this_occupation
    # etc. are entity→type-class POINTERS, not path carriers — used as a
    # mid-hop they mint hub mega-patterns ('Visual Artist --notable_types-->
    # every visual artist', 74-86 chains) whose mid-edges then render as
    # edges with no (sane) pattern correspondence. Mid-hop exclusion only;
    # the submitted relation itself is unaffected.
    import re as _re_type
    _TYPE_MID_RE = _re_type.compile(
        r"notable_types$|characters_with_this_occupation$|type\.object|"
        r"\.type$|\.types$")

    def _mid_ok(ri):
        return not _TYPE_MID_RE.search(str(ctx.rels[ri]))

    # depth 2: (r1, fam)
    for r1, reach in hop1.items():
        if not _mid_ok(r1):
            continue
        for node in reach:
            if not (0 <= node < n) or node == ci:
                continue
            for f in fams:
                if f == r1:
                    continue
                if node in ix.fwd[f] or node in ix.rev[f]:
                    patterns.add((r1, f))

    # depth 3: (r1, r2, fam) — hop2 named reach from hop1 nodes, bounded
    hop2_cache = {}
    for r1, reach in hop1.items():
        if not _mid_ok(r1):
            continue
        for node in reach:
            if not (0 <= node < n) or node == ci:
                continue
            if node not in hop2_cache:
                hop2_cache[node] = _reach({node})
            for r2, reach2 in hop2_cache[node].items():
                if r2 == r1 or not _mid_ok(r2):
                    continue
                for node2 in reach2:
                    if node2 == ci or node2 == node:
                        continue
                    for f in fams:
                        if f == r2 or f == r1:
                            continue
                        if node2 in ix.fwd[f] or node2 in ix.rev[f]:
                            patterns.add((r1, r2, f))
    # STEP COVERAGE (user ruling 2026-09-22): primary rank = number of the
    # tree's accumulated step relation-SETS the pattern hits (step1 then
    # step2 beats step2-only); semantics rank within equal coverage (B
    # phase). Counting STEPS, never entities.
    # INVERSE-FOLDED MATCH (user ruling 2026-10-09): direction variants
    # ARE different relations, but the walk may realize a step leg under
    # the graph's complementary encoding (partiallycontains submitted;
    # partially_containedby in the pattern). The coverage check folds
    # through _ctx_inverse_rels — pattern relation r matches step-set st
    # if r ∈ st OR r's inverse-folded representative ∈ st. This is NOT
    # hardcoding name pairs: the fold table is derived per-case from
    # co-occurring complementary edges (see _ctx_inverse_rels).
    _inv_rels = _ctx_inverse_rels(ctx)

    def _rel_matches_step(r, st):
        if r in st:
            return True
        rn = str(ctx.rels[r]) if 0 <= r < len(ctx.rels) else str(r)
        rep = _inv_rels.get(rn)
        if rep is not None:
            for s in st:
                sn = str(ctx.rels[s]) if 0 <= s < len(ctx.rels) else str(s)
                if _inv_rels.get(sn, sn) == rep:
                    return True
        # STRUCTURAL MATCH RETIRED (walk-bridge unification 2026-10-10,
        # was bf36270): frontier-entity overlap no longer covers a step —
        # declared patterns carry NATIVE realizes tags from the per-leg
        # bridge channel in _patterns_from_layers, so this fallback
        # filter shrinks back to exact + inverse-fold.
        return False

    def _cov(pk):
        if not steps:
            return 0
        return sum(1 for st in steps
                   if any(_rel_matches_step(r, st) for r in pk))
    cov = {pk: _cov(pk) for pk in patterns}
    if not patterns:
        return None, None
    # JOINT-COMMAND CONSTRAINT (constitution B2, batch 1): with ≥2 command
    # segments a pattern must instantiate EVERY segment — coverage becomes
    # a hard filter, not a rank. A bare (terminal,) pattern from the head
    # bypasses step 1 and re-walks the anchor's own edges (the co2
    # specimen: France --co2--> its own series instead of the frontier's).
    # First-call derivation (steps=None / single segment) is unfiltered.
    if steps and len(steps) >= 2:
        _need = len(steps)
        patterns = {pk for pk in patterns if cov[pk] >= _need}
        cov = {pk: c for pk, c in cov.items() if pk in patterns}
        if not patterns:
            return None, None
    # ORDERING: (hops, name) — length first, semantics second (B phase);
    # support never influences selection.
    # ENUMERATION CEILING (user audit 2026-09-19): topk=0 (semantic mode)
    # used to become [:None] — the FULL enumeration returned, and when the
    # B-phase selection failed (its exception was swallowed) every
    # enumerated pattern walked and rendered (567: 39 sections for 4
    # submitted relations). Cap the semantic-mode candidate pool at 60 so
    # selection failure degrades gracefully instead of exploding.
    _cap = max(topk, 0) or 60
    ranked = sorted(patterns, key=lambda p: (len(p), p))[:_cap]
    out = {p: tuple(frozenset([r]) for r in p) for p in ranked}
    return out, cov


# ───────────────── RECONSTRUCTION LANE (user ruling 2026-09-19) ─────────────────
# The pattern layer has ALREADY searched (enumerate → rank → select); the
# instance layer must RECONSTRUCT, not search again: walk each selected
# pattern hop by hop deterministically, keep only THROUGH chains (an
# h-r1-neighbor that cannot continue r2 is not on the path), CVT landings
# pass through their edges (the CVT stays on the chain so the renderer can
# inline its attributes), no beam, no witness collection.

_REBUILD_BUDGET = 400          # per-hop breadth flood control (hub safety)


def _rebuild_paths(ctx, ix, start_idx, hops, budget=_REBUILD_BUDGET):
    """Deterministic through-chain reconstruction for ONE selected pattern —
    KIND-AGNOSTIC (route 1, user ruling 2026-09-26): every landed node
    (named, CVT, id-node alike) is an ordinary chain node and every edge
    is a real pattern hop. No CVT pass-through: a mediator hop must be
    carried by the pattern's own relation sequence (the 25_892f ③
    specimen — selected 2-hop patterns rendering as 3-relation chains
    because hidden transparency edges slipped into hops_edges — dies
    here by construction).

    hops: tuple of frozensets (rel idxs) — one per pattern hop (a plain
    direct step is a single hop with the full submitted rel set).
    Returns (chains, edges, dirs): chains = [{"nodes": [idx...],
    "edges": [(h_idx, rel_idx, t_idx)...]}]; edges = the union of
    directed triples; dirs (V3-03) = {traversal idx triple: 'fwd'|'rev'}
    — the discovery direction of every walked edge, so the ledger can
    stage facts in RAW graph orientation and keep the traversal
    direction as usage metadata (same raw edge from either side = one
    fact; (A,r,B) and (B,r,A) stay the two distinct facts they are).
    Only chains reaching the LAST hop survive."""
    n = len(ctx.ents)
    parents = {}                        # node -> (prev, rel_idx)
    dirs = {}                           # V3-03: traversal triple -> fwd/rev
    # LAST-HOP RE-DISCOVERY EDGES (user ruling 2026-09-24, Eleanor specimen
    # 1392): an already-parented terminal re-discovered on the last hop
    # satisfies `found` without recording the discovering edge — remember
    # the discovering (src, rel, tgt) so chain assembly can materialize
    # the submitted relation's terminal hop.
    # F02 (Codex audit 2026-09-30): keep ALL discovering edges per terminal,
    # not just the first — the single-predecessor backbone can carry only
    # one branch, and the other witnesses used to vanish outright (diamond
    # A-r1-{B,D}-r2-C kept 2 of 4 edges). They enter the returned union
    # edge_set below even when no backbone chain carries them.
    term_edges = {}                     # end -> [(src, rel_idx, tgt), ...]
    level = {start_idx}
    for hop in hops:
        _last_hop = hop is hops[-1]
        nxt, found = set(), False
        for u in sorted(level):
            for r in sorted(hop):
                f, rv = ix.fwd.get(r, {}), ix.rev.get(r, {})
                # V3-03: an edge expanded from the FORWARD adjacency is the
                # raw (u --r--> v) fact; one expanded from the REVERSE
                # adjacency corresponds to the raw (v --r--> u) fact. When
                # both raw twins exist the forward one represents the step
                # (deterministic; the reverse twin still materializes when
                # walked from its own side).
                _fset = set(f.get(u, ()))
                _rset = set(rv.get(u, ())) - _fset
                for v in sorted(_fset | _rset):
                    if not (0 <= v < n) or v == start_idx:
                        continue
                    dirs[(u, r, v)] = "fwd" if v in _fset else "rev"
                    if v not in parents:
                        parents[v] = (u, r)
                        nxt.add(v)
                        found = True
                    elif _last_hop:
                        # TERMINAL RE-DISCOVERY (audit ③ specimen 576/1812,
                        # 2026-09-21): a node discovered at an EARLIER hop
                        # (single-parent DAG) may join the terminal level on
                        # the LAST hop; its backtrack follows the original
                        # parent path and the discovering edge is remembered
                        # for materialization below. BOTH shapes: outgoing
                        # (self-loop v==u) and INCOMING (2540: Monet's 18
                        # influenced in-neighbors re-discovered on the last
                        # hop — their discovering edge (u --r--> v) heads
                        # into the chain's end v).
                        nxt.add(v)
                        found = True
                        term_edges.setdefault(v, []).append((u, r, v))
        if not found:
            return [], set(), {}
        if len(nxt) > budget:
            nxt = set(sorted(nxt)[:budget])
            # SHADOW LEDGER (phase-2, additive; F03): per-hop breadth cut —
            # count it so the call's EvidenceRecord completeness can say
            # PARTIAL instead of the note's unconditional completeness.
            ctx._ev_rebuild_truncs = getattr(ctx, "_ev_rebuild_truncs", 0) + 1
        level = nxt

    def _backtrack(node):
        """Parent-DAG backtrack -> directed-contiguous edge list from
        start_idx to node, or None when the trail is orphaned
        (budget-severed)."""
        rev = []
        cur = node
        while cur != start_idx:
            _par = parents.get(cur)
            if _par is None:
                return None
            prev, rel = _par
            rev.append((prev, rel, cur))
            cur = prev
        return list(reversed(rev))

    def _contiguous(edges):
        """F02 post-assembly invariant: a chain's edges must join segment
        by segment (edges[i+1].head == edges[i].tail, first head = start).
        A dis-contiguous edge list is never shipped — dropped, not faked."""
        return bool(edges) and edges[0][0] == start_idx and all(
            edges[i + 1][0] == edges[i][2] for i in range(len(edges) - 1))

    # backtrack every final-level node to the start (chains share prefixes
    # through the parent DAG; each end yields its own chain)
    chains, edge_set = [], set()
    if len(level) > budget:
        # SHADOW LEDGER (phase-2, additive; F03): terminal backtrack cap.
        ctx._ev_rebuild_truncs = getattr(ctx, "_ev_rebuild_truncs", 0) + 1
    for end in sorted(level)[:budget]:
        edges = _backtrack(end)
        if not edges:
            continue
        # LAST-HOP MATERIALIZATION (user ruling 2026-09-24): a terminal
        # re-discovered on the last hop reached this level WITHOUT its
        # discovering edge — the submitted relation's terminal hop is then
        # missing from the chain and the render layer's full-depth check
        # drops it (the relation parsed, selected, and silently vanished).
        # When the chain is otherwise SHORTER than the pattern's hop count,
        # the remembered discovering edge materializes the last hop — under
        # a CONTIGUITY contract (F02): OUTGOING shape (edge head == chain
        # end, self-loop/CVT-attr-back) appends; INCOMING shape (edge tail
        # == chain end) REBUILDS the backbone from the edge's source, so
        # the discovering edge becomes the terminal segment of a real path
        # (the old blind append spliced (B,r2,C) onto the A-r1-C chain and
        # shipped nodes=[A,C,C] with the mid B missing). Complete chains
        # (576/1812 rosters) are untouched.
        if len(edges) < len(hops):
            for _te in (term_edges.get(end) or ()):
                if _te[0] == end:
                    _cand = list(edges) + [(_te[0], _te[1], _te[2])]
                else:
                    _pre = _backtrack(_te[0])
                    if not _pre:
                        continue
                    _cand = _pre + [(_te[0], _te[1], _te[2])]
                if _contiguous(_cand):
                    edges = _cand
                    break
        if not _contiguous(edges):
            continue
        nodes = [start_idx] + [t for (_h, _r, t) in edges]
        chains.append({"nodes": nodes,
                       "edges": edges,
                       "full_nodes": nodes, "full_edges": list(edges)})
        edge_set.update(edges)
    # F02 WITNESS PRESERVATION: every last-hop discovering edge — and its
    # source's backtrack branch — enters the union edge set even when the
    # single-predecessor backbone keeps it out of every chain (diamond: the
    # D-branch A-r1-D / D-r2-C is a true pattern instantiation; losing it
    # shrank the walk's evidence to one branch). Backbone chains stay
    # single; the full layered witness graph is phase 2.
    for _v, _tes in term_edges.items():
        for (_u, _r, _t2) in _tes:
            edge_set.add((_u, _r, _t2))
            edge_set.update(_backtrack(_u) or ())
    return chains, edge_set, dirs


def _rebuild_pe_list(ctx, treq, _mseq, _ms_map):
    """Rebuild pe_list directly from the SELECTED patterns (no search lane,
    no PG completeness block). Also archives treq["confirmed"] =
    {(center_idx, pattern-name-tuple): [chain...]} — the render layer reads
    this instead of re-inferring the pattern↔path mapping by name matching
    (the old mismatch that killed heterogeneous multi-hop sections).
    UNIFIED (user ruling 2026-09-21): every call is pattern → rebuild →
    expand. Derived patterns walk from the center; a plain direct query is
    the trivial single-hop pattern walking from the step's starts (frontier
    members on a continuation, else the center). Both lanes share ONE
    assembly: paths = pattern hops only (license path-admission needs a
    submitted last hop), triples = full_edges (the terminal-CVT passthrough
    expansion rides along — the renderer inlines it at the CVT node),
    candidates = named full terminals."""
    from kgqa.stages.formatting import PatternEvidence
    from kgqa.traversal.pattern_walk import get_pattern_index
    ix = get_pattern_index(ctx)
    # RENDER DELTA (user ruling 2026-09-20, superseding the 2026-08-20
    # full-re-render): a continuation call renders only what is NEW —
    # chains whose every edge was already displayed by a prior subgraph of
    # this case render nothing. First calls (empty accumulation) render
    # everything; EXTEND naturally shows the new layer's chains.
    # FACT KEYS (user ruling 2026-09-23): the comparison key is the FACT —
    # direction-normalized endpoints + the inverse-folded relation — not the
    # exact (h, r, t) walk record. Exact keys let the same fact re-enter as
    # "new" under a reversed traversal direction or its inverse relation id
    # (1731 specimen).
    _root_of = treq.get("root_of") or {}
    # EXACT-TRIPLE DELTA (user realignment 2026-09-24): a prior call's
    # (h, r, t) suppresses ONLY the same exact triple (direction-
    # normalized — the same edge walked from either side). The inverse-
    # fold merged act/cat storage twins as one fact and re-killed
    # sg2's 76 chains (2540); '完全对应' is the charter.
    _inv = _ctx_inverse_rels(ctx)
    _shown_facts = {(frozenset((normalize(str(h)), normalize(str(t)))),
                     str(r))
                    for (h, r, t) in (getattr(ctx, "accumulated_triples",
                                              None) or set())}
    _scope_set = set(treq.get("scope_ids") or ())
    _scope_cvt_only = bool(_scope_set) and all(
        _cvt_like_name(ctx.ents[i]) if 0 <= i < len(ctx.ents) else False
        for i in _scope_set)
    confirmed = {}
    out = []
    _seen_roots = set()
    for (_cn, _ci) in treq["centers"]:
        # TREE-KEYED (execute side): resolve to the walk root; duplicate
        # centers collapsing onto one root walk once.
        _ci = _root_of.get(_ci, _ci)
        if _ci in _seen_roots:
            out.append({})
            continue
        _seen_roots.add(_ci)
        combined = {}
        if not (0 <= _ci < len(ctx.ents)):
            out.append({})
            continue
        _raw_fallback = None      # (chains, names-key) when the delta filter
                                  # ate everything — see the empty-fallback below

        def _delta_new(chains):
            # RENDER DELTA: drop chains whose EVERY edge was already
            # displayed by a prior subgraph (all-old chains render nothing;
            # a chain with at least one new edge keeps — its rows carry the
            # new relation's context)
            return [ch for ch in chains
                    if not all(
                        (frozenset((normalize(str(ctx.ents[h])),
                                    normalize(str(ctx.ents[t])))),
                         str(ctx.rels[r])) in _shown_facts
                        for (h, r, t) in ch["full_edges"])]

        def _assemble(names, chains):
            """The ONE PatternEvidence assembly. triples = full_edges (the
            terminal-CVT passthrough expansion rides along — the renderer
            inlines it at the CVT node); paths = the PATTERN HOPS ONLY
            (ending at the submitted relation — license path-admission
            requires it); candidates = NAMED full terminals. The retired
            plain-direct branch — born before the edges/full_edges split
            (e9a10fd) redefined ch["edges"] to pattern hops — assembled
            triples from the pattern-hop edges and candidates from mid
            terminals, silently losing the passthrough evidence (2784
            specimen: Tupac's performance CVTs rendered nameless)."""
            edges = {e for ch in chains for e in ch["full_edges"]}
            label = " ⭢ ".join(
                ".".join(str(r).rsplit(".", 2)[-2:]) for r in names)
            triples = sorted(
                (str(ctx.ents[h]), str(ctx.rels[r]), str(ctx.ents[t]))
                for (h, r, t) in edges)
            paths = [{"nodes": [str(ctx.ents[i]) for i in ch["nodes"]],
                      "relations": [str(ctx.rels[r])
                                    for (_h, r, _t) in ch["edges"]]}
                     for ch in chains]
            _cands = {str(ctx.ents[ch["full_nodes"][-1]])
                      for ch in chains
                      if not _cvt_like_name(
                          str(ctx.ents[ch["full_nodes"][-1]]))}
            combined[label] = PatternEvidence(
                label, label, sorted(_cands), triples, {"paths": paths})
            confirmed[(_ci, names)] = list(
                (confirmed.get((_ci, names)) or [])) + chains

        # lane 1 — derived patterns (multi-hop or layered), from the center.
        # SINGLE-LAYER declared chains (pattern key len==1, the most common
        # continuation shape) ride here too: a 1-layer chain IS one hop from
        # the root — the old len>1 guard dropped them entirely (block
        # renders empty).
        # TRIVIAL FALLBACK (user ruling 2026-09-23): when derivation produced
        # NO pattern for this center, the submitted relations themselves ride
        # as 1-hop patterns FROM THE CENTER — the retired lane-2's first-call
        # duty, now on the same single code path (root-anchored, like every
        # other pattern). Lane-2's continuation duty (frontier-started,
        # rel_idxs-including-injected-bridges one-hop walks) is DELETED: a
        # continuation is a tree update followed by a re-walk of the FULL
        # chain from the root, not a fresh retrieval seeded at the frontier.
        _ms = dict(_mseq.get(_ci) or {})
        if not _ms:
            _ms = {(ri,): (frozenset({ri}),)
                   for ri in sorted(set(treq["rel_idxs"]))
                   if 0 <= ri < len(ctx.rels)}
        for _pk, _seq in _ms.items():
            chains, _edges, _dirs = _rebuild_paths(ctx, ix, _ci, _seq)
            if _edges:
                # F02: the rebuild lane's edge union (backbone chains +
                # last-hop witness branches that no chain carries) — the
                # finalize-side EvidenceSet builder merges this so diamond
                # branches do not vanish from the call's evidence.
                _wu = treq.get("_witness_edges")
                if _wu is None:
                    _wu = treq["_witness_edges"] = set()
                _wu |= _edges
                # V3-03: the per-edge discovery directions ride the same
                # treq — the ledger builder stages RAW-orientation facts
                # and records the traversal direction per usage.
                _edm = treq.get("_edge_dirs")
                if _edm is None:
                    _edm = treq["_edge_dirs"] = {}
                _edm.update(_dirs or {})
            if not chains:
                continue
            # RENDER SCOPE FILTER (A-line lesson, -4.6pp rollback), SCOPED
            # TO TRANSIT CENTER SETS ONLY (route 1 refinement per the
            # frontier-全域 ruling): when the call resolved ≥2 centers and
            # EVERY one is a transit node (CVT/id supernode — the 2784-s0 /
            # 25_db96 class where the declared centers ARE the render
            # contract), instantiation keeps only chains passing THROUGH
            # the declared center set. Named/curated rosters do NOT gate
            # coverage — bindings are a curated subset and the gold may
            # live beyond them (25_892f: the gold TV work's runtime must
            # materialize from the WORK FRONTIER, not the model's film
            # roster), so named-center calls instantiate the full frontier.
            # Falls back to unfiltered when the filter would empty the
            # pattern.
            if _scope_cvt_only and len(_scope_set) >= 2 and len(chains) > 1:
                _kept = [ch for ch in chains
                         if _scope_set.intersection(ch["full_nodes"])]
                if _kept:
                    chains = _kept
            _new = _delta_new(chains)
            if not _new:
                # EMPTY-RENDER FALLBACK (user ruling 2026-09-21): an
                # all-old call used to render NOTHING — the model then held
                # NO evidence in context for these relations and exploded
                # into reasoning/re-asking loops. When the whole call would
                # render empty, the REPEATED evidence is shown again
                # (marked) instead.
                if _raw_fallback is None:
                    _raw_fallback = (
                        chains, tuple(str(ctx.rels[r]) for r in _pk))
                continue
            _assemble(tuple(str(ctx.rels[r]) for r in _pk), _new)

        # EMPTY-RENDER FALLBACK: the delta filter ate every chain but the
        # walk DID produce chains — re-show the repeated evidence (first
        # candidate, marked) instead of an empty render. All-old repeats
        # used to fall through to a spurious NO_EVIDENCE — the walk had
        # evidence, it was merely repeated.
        if not combined and _raw_fallback is not None:
            _assemble(_raw_fallback[1], _raw_fallback[0])
            treq["_repeat_evidence"] = True
        out.append(combined)
    treq["confirmed"] = confirmed
    # CHAIN LANDING CAPTURE (constitution B1/B2, batch 1): per-segment
    # landing sets + terminal-CVT attribute endpoints from the DELIVERED
    # chains — the SINGLE frontier record feeding attribution anchoring,
    # the RR retrieval start, and the echo counts. Segment k's landing =
    # the node after the last hop whose relation belongs to segment k
    # (derived mid-hop relations that belong to no segment are skipped
    # without advancing); terminal-CVT attribute VALUES (b1/b3) are
    # anchorable entities and ride "attrs".
    _astx = getattr(ctx, "anchor_seqs", None) or {}
    _clx = getattr(ctx, "chain_landing", None)
    if _clx is None:
        _clx = {}
        ctx.chain_landing = _clx
    _ctouch = getattr(ctx, "chain_touch", None)
    if _ctouch is None:
        _ctouch = {}
        ctx.chain_touch = _ctouch
    _clk = (getattr(ctx, "_chain_clock", 0) or 0) + 1
    ctx._chain_clock = _clk
    for (_ci, _pk), chains in confirmed.items():
        _segs = _astx.get(_ci)
        if not _segs or not chains:
            continue
        _seg_rels = [set(s) for s in _segs]
        _landings = [set() for _ in _segs]
        _attrs = set()
        for ch in chains:
            # LAST-SEGMENT LANDING = the chain TERMINALS (direction- and
            # name-agnostic): derived patterns pick their own relation
            # names for hops (actor.film ≠ the submitted film.starring),
            # so name-matched segment partition is only a best-effort
            # echo source — the pool/attribution frontier must be the
            # terminals themselves (the 892f specimen: the runtime
            # relation hung on terminals name-matching missed).
            _term = (ch.get("full_nodes") or [None])[-1]
            if _term is not None and _landings:
                _landings[-1].add(_term)
            nodes = ch.get("nodes") or []
            k = 0
            for j, (_h, _r, _t) in enumerate(ch.get("edges") or ()):
                if k < len(_seg_rels) and _r in _seg_rels[k]:
                    if j + 1 < len(nodes):
                        _landings[k].add(nodes[j + 1])
                    if k + 1 < len(_seg_rels):
                        k += 1
            for (_h, _r, _t) in (ch.get("full_edges") or ()):
                if _landings and _h in _landings[-1] and \
                        _cvt_like_name(str(ctx.ents[_h])):
                    _attrs.add(_t)
        _prev_pref = list(((_clx.get(_ci) or {}).get("pref") or ())
                          [:len(_segs)])
        _old_pref = _prev_pref + list(range(len(_prev_pref), len(_landings)))
        _prev_starts = list(((_clx.get(_ci) or {}).get("starts") or ())
                            [:len(_segs)])
        while len(_prev_starts) < len(_landings):
            _prev_starts.append(frozenset())
        _clx[_ci] = {"segs": tuple(frozenset(s) for s in _landings),
                     "pref": tuple(_old_pref),
                     "attrs": frozenset(_attrs),
                     "starts": tuple(frozenset(s) for s in _prev_starts)}
        _ctouch[_ci] = _clk
    return out


async def _sg_execute(treq, ctx, session):
    """B段: entity-name correction (GTE) when a bad name was flagged — with
    candidates the turn ends with the correction result; otherwise (and when
    the correction found nothing) the packed walk runs (coordinator batches)."""
    # CVT ATTR-KEY GTE RANKING (user design 2026-09-08): pre-compute the
    # question→attribute-key relevance ONCE per case (cached on ctx), so the
    # renderer can show only the top-K most question-relevant attributes per
    # CVT instead of flooding every bracket. Avg 96 keys/case — random top-3
    # covers 5% of gold keys; GTE top-3 covers 50%.
    _rank = getattr(ctx, "_cvt_key_rank", None)
    _qnow = str(getattr(ctx, "question", "") or "")
    if _rank is None or _rank[0] != _qnow:
        _keys = set()
        for _h, _r, _t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
            _hn = str(ctx.ents[_h]) if 0 <= _h < len(ctx.ents) else str(_h)
            _tn = str(ctx.ents[_t]) if 0 <= _t < len(ctx.ents) else str(_t)
            if is_cvt_like(_hn) != is_cvt_like(_tn):
                _keys.add(str(ctx.rels[_r]).rsplit(".", 1)[-1]
                          if 0 <= _r < len(ctx.rels) else "?")
        if _keys and _qnow:
            try:
                from kgqa.stages.stage2_entity import gte_retrieve
                _kl = sorted(_keys)
                _rows = await gte_retrieve(session, _qnow, _kl, top_k=len(_kl))
                _ranked = [r.get("candidate") for r in (_rows or [])
                           if r.get("candidate") in _keys]
                _ranked += [k for k in _kl if k not in _ranked]
                ctx._cvt_key_rank = (_qnow, _ranked)
            except Exception:
                ctx._cvt_key_rank = (_qnow, sorted(_keys))
        else:
            ctx._cvt_key_rank = (_qnow, [])
    if treq.get("corr") is not None:
        bad_name = treq["corr"]
        cands = await _entity_correction(bad_name, getattr(ctx, "question", ""), ctx, session)
        if cands:
            return {"corr_cands": cands, "bad_name": bad_name}
        if not treq["centers"]:
            # correction found nothing AND no center survived resolution — the
            # original flow's `if not centers` error (never reaches the walk)
            return {"no_centers": True}
    if getattr(ctx, "_adaptive_worker", False):
        from kgqa.agent.seq_adaptive import start_walk
        start_walk(ctx, treq)
    from kgqa.core.utils import phase_timer
    # PATTERN-PATH WALK (user design 2026-09-10, A+ integration): a call
    # whose centers are one ?var's binding set (≥4) walks ONCE at the
    # relation level — set-state transitions, bridge = any non-target
    # relation, terminal = the model's selected relations; witnesses are
    # materialized for the top-K ranked patterns (先模式,再实例化). Empty
    # pattern result falls back to the per-binding walks.
    _pf = treq.get("prefix")
    _steps = [(i, treq["rel_idxs"], treq["fid"], _pf) if _pf is not None
              else (i, treq["rel_idxs"], treq["fid"])
              for _, i in treq["centers"]]
    if (os.environ.get("SEQ_PATTERN_WALK", "0") == "1"
            and treq.get("var_name") and len(treq["centers"]) >= 4):
        try:
            _pw = _pattern_walk_evidence(ctx, treq)
        except Exception:
            _pw = None
        if _pw is not None:
            return _pw
    # MULTI-STEP PATTERN WALK (realignment spec 2026-09-11, pillar 1+2+3):
    # when the submitted family has NO direct 1-hop support on a center
    # (the case bridges were computed for), derive the relation SEQUENCE
    # center→…→family-terminal from the pattern index (≤2 named hops,
    # CVT-transparent) and submit it as MULTI-STEP step_relations — the
    # engine's forward validation (chain_expand lookahead) enforces the
    # path-consistency pillar natively, RPE falls back only on weak
    # coverage (n_steps>1), and reach no longer needs bridges-as-termini.
    _mseq = treq.get("multistep")     # {center_idx: {pat_key: (fs1, fs2)}}
    _ms_map = {}                      # bound OUTSIDE the gate (audit F1: an
                                      # env flip between phases left it unbound
                                      # and crashed the merge below)
    if _mseq and os.environ.get("SEQ_MULTISTEP", "1") == "1":
      if os.environ.get("SEQ_PAT_SEMANTIC", "1") == "1":
        # SEMANTIC PATTERN SELECTION (user ruling 2026-09-12): the derivation
        # layer enumerates ALL 2-hop patterns; HERE (B phase, GTE available)
        # they are ranked semantically against the question and each center
        # keeps top-3 MULTI-HOP patterns total (the 1-hop direct rides the
        # plain step). GATED OFF after two 48x3 A/Bs: offline it retains
        # gold patterns better (6/6 vs 5/6 top-3) but end-to-end it costs
        # -2.0pp hit / -4.4pp f1 vs the support-topk (semantic 78.5/0.641
        # with section filter, 79.2/0.657 without vs pattern4 81.2/0.701) —
        # high-fan-out patterns carry discrimination context beyond the
        # gold. DEFAULT ON after the 3-run mean verdict (2026-09-12):
        # semantic 78.7±1.4 hit / 65.1±2.0 f1 vs support 77.3±3.1 / 64.6±2.7
        # — better mean AND half the variance; pattern4's 81.2 was the top
        # of support's variance band. SEQ_PAT_SEMANTIC=0 restores support.
        try:
            from kgqa.stages.stage2_entity import gte_retrieve
            _q = str(getattr(ctx, "question", "") or "")
            _cands, _cmap = [], {}
            for _ci, _pats in _mseq.items():
                for _pk in _pats:
                    if len(_pk) == 1:
                        continue        # 1-hop direct: first-class, no quota
                    _s = " -> ".join(str(ctx.rels[_r]) for _r in _pk)
                    _k = (_ci, _s)     # KEY FORM (route 1 D, constraint A):
                                        # TWO-PART (chain-root anchor, pattern
                                        # string) — post-D1 `_ci` is the chain
                                        # root, not a per-center entity
                                        # identity; one string under one root
                                        # occupies exactly one seat.
                    if _k not in _cmap:
                        _cmap[_k] = _pk
                        _cands.append(_s)
            _rank = {}
            if _cands and _q and session is not None:
                _rows = await gte_retrieve(session, _q, sorted(set(_cands)),
                                           top_k=len(set(_cands)))
                _rank = {r.get("candidate"): i
                         for i, r in enumerate(_rows or [])}
            _sel = {}
            _cov = treq.get("multistep_cov") or {}
            # GLOBAL TOP-5 BY TOTAL ORDER (user directive 2026-09-24): one
            # leaderboard after the coarse pass (step coverage -> length ->
            # GTE semantics within equal length) — the most valuable
            # patterns live in the top five; top-5 bounds the render. ONE
            # narrowing point (selection), replacing the per-terminal x3
            # quota stack. FLOOR: each submitted terminal relation keeps
            # its single best pattern even outside the top-5 (Charlie-
            # Hunnam specimen: a relation crowded to zero lost its lane).
            _TOP = int(os.environ.get("SEQ_PAT_TOPK", "5"))
            # COV KEY ALIGNMENT (user mechanism realignment 2026-09-24):
            # step coverage = the pattern path walked FRONT TO BACK —
            # step1's relation set matched, then step2's; both hit =
            # coverage 2. The write keyed (center, idx-tuple) while the
            # read keyed (center, name-string) — the primary sort key
            # NEVER fired since introduction. Resolve through cmap's pk.
            _cov_of = {k: _cov.get((k[0], tuple(_cmap[k])), 0)
                       for k in _cmap}
            _ranked = sorted(
                _cmap,
                # ROUTE 1 D (user ruling 2026-09-26): the leaderboard ORDER
                # is pattern-layer only — (-coverage, length, GTE), ZERO
                # entity identity (the retired trailing k[0] broke ties by
                # center idx and decided WHICH centers' witnesses
                # instantiated a pattern; 2784-s0: Juice's identical
                # pattern lost to a lower idx). The anchor part survives
                # in the KEY (constraint A), never in the ORDER.
                key=lambda k: (-_cov_of.get(k, 0),
                               len(_cmap[k]),
                               _rank.get(k[1], 999),
                               k[1]))
            _chosen = list(_ranked[:_TOP])
            _floors = set()
            for (_ci, _s) in _ranked[_TOP:]:
                _term = _cmap[(_ci, _s)][-1]
                # FLOOR per (chain-root, terminal) (route 1 D): every root
                # keeps its single best pattern per submitted terminal even
                # outside the top-N — coverage by ANOTHER root's chosen
                # pattern does not spend this root's floor lane.
                if any(_cmap[k][-1] == _term and k[0] == _ci
                       for k in _chosen):
                    continue
                if (_ci, _term) in _floors:
                    continue
                _floors.add((_ci, _term))
                _chosen.append((_ci, _s))
            for (_ci, _s) in _chosen:
                _sel.setdefault(_ci, {})[_cmap[(_ci, _s)]] = _mseq[_ci][_cmap[(_ci, _s)]]
            # 1-hop direct patterns always survive (first-class, no quota)
            for _ci, _pats in _mseq.items():
                for _pk, _v in _pats.items():
                    if len(_pk) == 1:
                        _sel.setdefault(_ci, {})[_pk] = _v
            treq["multistep"] = _mseq = _sel
        except Exception:
            # SELECTION-FAILURE FALLBACK (user audit 2026-09-19): swallowing
            # the error left derive's UNBOUNDED enumeration (topk=0 under
            # SEQ_PAT_SEMANTIC) as treq["multistep"] — a hub case then
            # rendered every enumerated pattern. Fall back to length-first
            # top-3, the quota the selection was supposed to enforce.
            _fb = {}
            for _ci, _pats in (treq.get("multistep") or {}).items():
                for _pk in sorted(_pats, key=lambda k: (len(k), k))[:3]:
                    _fb.setdefault(_ci, {})[_pk] = _pats[_pk]
            treq["multistep"] = _fb
        if os.environ.get("SEQ_DEBUG_CHAIN", "0") == "1":
            import sys as _sys
            for _ci, _pats in (treq.get("multistep") or {}).items():
                _multi = {("->".join(str(ctx.rels[r]) for r in k)): len(k)
                          for k in _pats if len(k) > 1}
                print(f"[CHAIN-SEL] center={str(ctx.ents[_ci])[:24]!r} "
                      f"2hop+={_multi}", file=_sys.stderr, flush=True)
      # PATTERN-LEVEL MULTI-STEP (user design corrected 2026-09-11):
        # each center's top-K patterns become separate multi-step steps;
        # their pe results merge at finalize (same center, same fid).
      _ms_steps = []
      from kgqa.traversal.pattern_walk import get_pattern_index as _gpi
      _ix2 = _gpi(ctx)
      # direct family edges per center (the 1-hop pattern — shortest, so it
      # keeps its plain walk step ALONGSIDE the derived 2-hop patterns;
      # Belgium specimen: the CET/CEST direct row and the
      # containedby→time_zones chain both render, length-first ordered)
      _direct_fam = set()
      _echo_sh = set()
      for _rs, _exp in (treq.get("attr_expansion") or {}).items():
          _echo_sh.add(str(_rs))
          for _dn in (_exp.get("direct") or []):
              if _dn in ctx.rels:
                  _direct_fam.add(ctx.rels.index(_dn))
      # echo-less submitted names count too (same fallback as prepare's
      # _matched — Charlie-Hunnam specimen lost the plain direct step)
      _l1x = lambda rn: str(rn).rsplit(".", 1)[-1]
      for _rs in (treq.get("rel_names") or []):
          if str(_rs) in _echo_sh:
              continue
          for _ri, _rn in enumerate(ctx.rels):
              if (str(_rn) == str(_rs) or _l1x(_rn) == str(_rs)
                      or _l1x(_rs) == _l1x(_rn)):
                  _direct_fam.add(_ri)
                  break
      _fr_of = treq.get("cont_frontier") or {}
      for _cn, _ci in treq["centers"]:
          _pats = _mseq.get(_ci) or {}
          if _pats:
              _n_walk = 0
              for _pk, _seq in _pats.items():
                  if len(_pk) == 1:
                      continue     # 1-hop direct: the plain step walks it
                  _ms_steps.append((_ci, _seq, treq["fid"]))
                  _n_walk += 1
              _ms_map[_ci] = _n_walk
              # SEQUENCE CONTINUATION: the plain direct step applies the
              # submitted relation to the tree's FRONTIER members (per-member
              # rows, the per-binding compare render) — never to the root
              # (the root's OWN edges on the submitted relation flood the
              # render: France's 100 co2 edges swamped the 9-country rows)
              _fr = _fr_of.get(_ci)
              if _fr:
                  for _fi in _fr:
                      if any((((_ix2.head_mask.get(r, 0) >> _fi) & 1)
                              or ((_ix2.tail_mask.get(r, 0) >> _fi) & 1))
                             for r in _direct_fam):
                          _ms_steps.append((_fi, treq["rel_idxs"], treq["fid"]))
                          _ms_map[_ci] += 1
              elif (0 <= _ci < len(ctx.ents)) and any(
                      (((_ix2.head_mask.get(r, 0) >> _ci) & 1)
                       or ((_ix2.tail_mask.get(r, 0) >> _ci) & 1))
                      for r in _direct_fam):
                  _ms_steps.append((_ci, treq["rel_idxs"], treq["fid"]))
                  _ms_map[_ci] += 1
          else:
              _fr = _fr_of.get(_ci)
              if _fr:
                  for _fi in _fr:
                      _ms_steps.append((_fi, treq["rel_idxs"], treq["fid"]))
                      _ms_map[_ci] = _ms_map.get(_ci, 0) + 1
              else:
                  _ms_steps.append((_ci, treq["rel_idxs"], treq["fid"]))
                  _ms_map[_ci] = 1
      if _ms_steps:
          _steps = _ms_steps
    if os.environ.get("SEQ_REBUILD", "1") == "1":
        # RECONSTRUCTION LANE (user ruling 2026-09-19): the pattern layer
        # already searched; rebuild the selected patterns' through-chains
        # deterministically (no beam lane, no PG completeness block) and
        # archive treq["confirmed"] for the renderer.
        with phase_timer("walk"):
            pe_list = _rebuild_pe_list(ctx, treq, _mseq, _ms_map)
    else:
        with phase_timer("walk"):
            pe_list = await _run_walk_packed(ctx, _steps)
        # merge multi-pattern pe back per-center (concatenate pattern dicts)
        if _mseq and _ms_map:
            _merged, _idx = [], 0
            for _cn, _ci in treq["centers"]:
                _n = _ms_map.get(_ci, 1)
                _combined = {}
                for _pe in pe_list[_idx:_idx + _n]:
                    if isinstance(_pe, dict):
                        _combined.update(_pe)
                _merged.append(_combined if _combined else {})
                _idx += _n
            pe_list = _merged
    # PATTERN-COMPLETENESS GUARANTEE (user ruling 2026-09-13: the walk runs
    # on the reconstructed PATTERN graph — pattern branches are few, and
    # ENTITY-layer caps (beam / per-branch / support-path limits) must never
    # sever a SELECTED pattern's instantiations). Every selected 2-hop
    # pattern is instantiated EXHAUSTIVELY from the pattern index (bounded
    # only by a generous per-pattern budget; display caps apply later in the
    # render) and any hop the capped walk missed joins the pe paths.
    # [2026-09-19: reconstruction lane already rebuilds every selected
    #  pattern exhaustively-with-budget — this block only runs on the old
    #  search lane, and is DELETED on the rebuild lane per user ruling]
    if _mseq and _ms_map and os.environ.get("SEQ_REBUILD", "1") != "1":
        try:
            from kgqa.traversal.pattern_walk import get_pattern_index as _gpi2
            from kgqa.stages.formatting import PatternEvidence as _PE
            _ixg = _gpi2(ctx)
            _n = len(ctx.ents)
            for (_cn, _ci) in treq["centers"]:
                _pats = _mseq.get(_ci) or {}
                if not _pats or not (0 <= _ci < _n):
                    continue
                _slot = next((k for k, (_c2, _i2) in enumerate(treq["centers"])
                              if _i2 == _ci), None)
                if _slot is None or _slot >= len(pe_list):
                    continue
                _pe = pe_list[_slot]
                _have = set()
                for _p in (_pe.values() if isinstance(_pe, dict) else []):
                    for _tp in ((getattr(_p, "tree_data", None) or {})
                                .get("paths") or []):
                        _have.add((tuple(str(x) for x in (_tp.get("nodes") or [])),
                                   tuple(str(x) for x in (_tp.get("relations") or []))))
                _add = []
                for _pk in _pats:
                    if len(_pk) == 1:
                        continue     # 1-hop direct: plain step owns it
                    _budget = 200

                    def _enum(level, cur_nodes, chain):
                        if len(_add) >= _budget:
                            return
                        if level == len(_pk):
                            _nodes = tuple([str(ctx.ents[_ci])]
                                           + [str(ctx.ents[_x]) for _x in chain])
                            _rels = tuple(str(ctx.rels[_r]) for _r in _pk)
                            if (_nodes, _rels) not in _have:
                                _have.add((_nodes, _rels))
                                _add.append({"nodes": list(_nodes),
                                             "relations": list(_rels)})
                            return
                        r = _pk[level]
                        srcs = set()
                        for _cur in cur_nodes:
                            srcs.update(_ixg.fwd[r].get(_cur, ()))
                            srcs.update(_ixg.rev[r].get(_cur, ()))
                        for _x in sorted(srcs):
                            if not (0 <= _x < _n) or _x == _ci or _x in chain:
                                continue
                            _enum(level + 1, {_x}, chain + [_x])

                    _enum(0, {_ci}, [])
                if _add:
                    if not isinstance(_pe, dict):
                        _pe = {}
                    _pe["PG"] = _PE(label="PG", readable="",
                                    candidates=[], triples=[],
                                    tree_data={"paths": _add})
                    pe_list[_slot] = _pe
        except Exception:
            pass
    return {"pe_list": pe_list}


def _pattern_walk_evidence(ctx, treq):
    """Build PatternEvidence-compatible output for the pattern-path walk.
    Returns the bres dict ({"pe_list", "pattern_display"}) or None when the
    walk found no patterns (caller falls back to per-binding walks)."""
    from kgqa.traversal.pattern_walk import (get_pattern_index,
                                             pattern_walk, rank_display)
    from kgqa.stages.formatting import PatternEvidence
    ix = get_pattern_index(ctx)
    seeds = [cn for cn, _ci in treq["centers"]]
    # TERMINALS (1797 postmortem, two iterations): v1 = full rel_idxs
    # (direct+bridge, f1 0.6508); v2 = direct-only (0.6269 — recall loss
    # when the model's pick is off). The measured quality hole was NOT the
    # terminal set but the missing tier-1 rows (weaved below) — back to v1.
    _term_idxs = list(treq["rel_idxs"])
    pats = pattern_walk(ix, seeds, _term_idxs)
    if not pats:
        return None
    triples, candidates, paths = [], [], []
    seen_c = set()
    for k, p in enumerate(pats):
        for ch in p["witnesses"]:
            triples.extend(ch)
            paths.append({
                "nodes": [ch[0][0]] + [tr[2] for tr in ch],
                "relations": [tr[1] for tr in ch],
            })
        for a in p["answer"]:
            if a not in seen_c:
                seen_c.add(a)
                candidates.append(a)
    if not triples:
        return None
    # TIER-1 FEEDING (1797 postmortem, 2nd iteration): the v38 renderer
    # builds rows ONLY from pe tree paths — the per-binding DIRECT
    # (binding, terminal-rel) edges that _guarantee_center_direct_edges
    # appends to all_triples never reach it, so discrimination facts lost
    # the binding→value mapping entirely. Weave those edges in as paths
    # (capped per binding): the model reads the pattern groups AND every
    # binding's own terminal values.
    _n2i = {}
    for _j, _e in enumerate(ctx.ents):
        _n2i.setdefault(str(_e), _j)
    for _cn, _ci in treq["centers"]:
        if not (0 <= _ci < len(ctx.ents)):
            continue
        _n_add = 0
        for _ri in _term_idxs:
            for _t2 in ix.fwd[_ri].get(_ci, ()):
                paths.append({"nodes": [str(ctx.ents[_ci]),
                                        str(ctx.ents[_t2])],
                              "relations": [str(ctx.rels[_ri])]})
                triples.append((str(ctx.ents[_ci]), str(ctx.rels[_ri]),
                                str(ctx.ents[_t2])))
                _n_add += 1
                if _n_add >= 12:
                    break
            if _n_add >= 12:
                break
    pe = {"pat%d" % k: PatternEvidence(
        label="pat%d" % k,
        readable=" → ".join(str(ctx.rels[r]).rsplit(".", 2)[-1]
                            for r, _f in p["rels"]),
        candidates=[a for a in p["answer"][:20]],
        triples=[tr for p in pats for ch in p["witnesses"] for tr in ch],
        tree_data={"paths": paths}) for k, p in enumerate(pats)}
    pe_list = [pe] + [{} for _ in treq["centers"][1:]]
    return {"pe_list": pe_list,
            "pattern_display": rank_display(ctx, pats, _term_idxs)}


def _select_patterns_for_render(pe_values, center_name, sel_names, top_n=5):
    """① FEEDING GUARANTEE (user calibration 2026-08-26, walk-starvation
    audit): the renderer's tier-1 blocks are synthesized from the CENTER'S
    OWN selected-relation edges, so those edges must survive this
    collection. The top-N rank cut could starve them when a center's walk
    produced more detour patterns than N above its 1-hop one — every
    pattern that carries a center-anchored 1-hop selected-relation witness
    is therefore kept BEYOND the cut (bounded: at most 2 extras per selected
    relation, fwd+rev). Pure/bottom-level so the guarantee is testable."""
    keep = list(pe_values)[:top_n]
    sel = {str(r) for r in (sel_names or []) if r}
    if len(keep) == top_n and len(pe_values) > top_n and sel:
        from kgqa.core.utils import normalize as _nz
        ck = _nz(str(center_name))
        cap = top_n + 2 * len(sel)
        rest = list(pe_values)[top_n:]
        for p in rest:
            if len(keep) >= cap:
                break
            for tp in ((getattr(p, "tree_data", None) or {}).get("paths") or []):
                nds = tp.get("nodes") or []
                rls = tp.get("relations") or []
                if len(nds) == 2 and len(rls) == 1 and str(rls[0]) in sel:
                    k0 = _nz(str(nds[0]).split(":", 1)[0].strip())
                    k1 = _nz(str(nds[1]).split(":", 1)[0].strip())
                    if ck in (k0, k1):
                        keep.append(p)
                        break
    return keep


def _guarantee_center_direct_edges(ctx, centers, rel_idxs, all_triples):
    """① WALK-SIDE FEEDING GUARANTEE (multi-center starvation audit,
    2026-08-26): collect every DIRECT (center, selected relation) edge of the
    case graph that the walk's evidence missed, so it can be appended to
    all_triples. The renderer's tier-1 rows synthesize from exactly these
    edges, and the walk can drop them wholesale WITHOUT its patterns going
    empty: the 24-path support cap bounds a center's INVERSE 1-hop family
    (center = TAIL of the selected relation — 'Film --produced_by--> Ron'
    walked from center Ron Howard anchors Option-B's leaf enumeration on the
    witness's own HEAD, so the center's remaining same-relation partners
    never enumerate; Ron Howard specimen: 35 graph edges, 24 carried, 7
    producer credits invisible). Qualification is the definition of a 1-hop
    pattern instance — relation SELECTED + one endpoint a CENTER of this
    call — so the pattern-path discipline holds by construction. 补采 scans
    ctx's full edge arrays; dedup covers both orientations (the walk records
    hops in path order). Also lands the new endpoints in the same pools
    _accumulate feeds (subgraph_entities / all_candidates) so they are
    answerable and legal future centers. Pure w.r.t. the walk: same inputs →
    same additions."""
    _SCHEMA_PREFIXES = ("type.", "common.", "freebase.", "kg.", "user.",
                        "base.ontologies.", "owl#", "rdf-schema#")
    seen = set()
    for h, r, t in all_triples:
        nh, nt = normalize(str(h)), normalize(str(t))
        seen.add((nh, str(r), nt))
        seen.add((nt, str(r), nh))
    cset = {i for _n, i in centers if i is not None}
    rset = set(rel_idxs or ())
    if not cset or not rset:
        return []
    ents, rels = ctx.ents, ctx.rels
    se = getattr(ctx, "subgraph_entities", None)
    if se is None:
        se = set(); ctx.subgraph_entities = se
    at = getattr(ctx, "accumulated_triples", None)
    if at is None:
        at = set(); ctx.accumulated_triples = at
    seen_cand = {normalize(c) for c in getattr(ctx, "all_candidates", []) or []}
    added = []
    for h, r, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        if r not in rset or (h not in cset and t not in cset):
            continue
        if not (0 <= h < len(ents) and 0 <= t < len(ents)):
            continue
        r_name = str(rels[r]) if 0 <= r < len(rels) else ""
        if not r_name or r_name.lower().startswith(_SCHEMA_PREFIXES):
            continue                       # class-based schema filter (mirrors
                                          # the walk's evidence admission)
        h_name, t_name = str(ents[h]), str(ents[t])
        if len(normalize(h_name)) < 2 or len(normalize(t_name)) < 2:
            continue
        nh, nt = normalize(h_name), normalize(t_name)
        if (nh, r_name, nt) in seen or (nt, r_name, nh) in seen:
            continue
        seen.add((nh, r_name, nt)); seen.add((nt, r_name, nh))
        added.append((h_name, r_name, t_name))
        at.add((h_name, r_name, t_name))
        se.update((h, t))
        for nm in (h_name, t_name):
            if not is_cvt_like(nm):
                nn = normalize(nm)
                if nn not in seen_cand:
                    seen_cand.add(nn); ctx.all_candidates.append(nm)
    return added



def _display_license_filter(treq, bres, centers, all_triples, candidates):
    """DISPLAY LICENSE FILTER (user ruling 2026-09-07): edges/candidates the
    walk wandered to OUT of license — riding relations NOT submitted by this
    call, beyond the centers' relation-licensed + CVT/identity-mirror
    transparent path — are removed from every display variable and never
    rendered (the 567 language-hub escape specimen). Display-only: ctx
    accumulation, bindings and the answer-time join gate are untouched.
    Transparency: CVT endpoints always pass (two-hop + CVT-extension scope);
    common.topic.*/user.* plumbing and same-name identity mirrors
    (representations_in_fiction / based_on between name-mirrored nodes) pass
    (1379:s1: person facts hang on the topic-mirror node).

    PATH-LEVEL ADMISSION (user ruling 2026-09-12, pattern-walk era): the
    edge-level rule below belongs to the OLD architecture (walk entities,
    induce patterns afterwards) — it evaluates every hop against the
    submitted names, so an engine-DERIVED pattern's mid hop (行政辖属→属地
    before the submitted time_zones terminal) reads as drift and the whole
    2-hop path is dropped, both hops, even though the walk succeeded (the
    realign5 UK/ETZ specimens: walk fine, evidence zeroed, misreported as
    RELATION_MISMATCH). Under the pattern-first architecture the unit of
    admission is the PATH: a path is legal iff its LAST hop is a submitted
    relation; mid hops are the pattern's own segments and license their
    nodes by path membership. Filtering still runs AFTER the walk completes
    (walk full, then filter). Active only for calls that carried derived
    patterns (treq.multistep) — plain calls keep the legacy rule, so the
    standing config's behavior is unchanged."""
    from kgqa.core.utils import normalize as _nz
    subs = {str(r) for r in (treq.get("rel_names") or [])}
    subs_typed = {".".join(r.rsplit(".", 2)[-2:]) for r in subs}
    center_names = [str(e) for e, _i in centers]

    def _cvt(x):
        return is_cvt_like(str(x))

    def _head(s):
        s = str(s)
        h = s.split(":", 1)[0].strip()
        return h if h != s else s

    def _mirror(a, b):
        a, b = str(a), str(b)
        return (a in b or b in a) and min(len(a), len(b)) >= 5

    def _transparent(r, a, b):
        r = str(r)
        if r.startswith(("common.topic.", "user.")):
            return True
        if r.endswith(("representations_in_fiction", "fictional_character.based_on")):
            return _mirror(a, b)
        return False

    edges = [(str(tr[0]), str(tr[1]), str(tr[2])) for tr in all_triples
             if len(tr) == 3]
    path_mode = bool(treq.get("multistep"))
    _path_admitted = None
    if path_mode:
        # license = nodes of paths whose LAST hop is a submitted relation
        # (full or typed-name match; CVT endpoints stay transparent)
        def _path_admitted(tp):
            rels = tp.get("relations") or []
            nodes = tp.get("nodes") or []
            if not rels or not nodes:
                return False
            last = str(rels[-1])
            last_typed = ".".join(last.rsplit(".", 2)[-2:])
            return (last in subs or last_typed in subs_typed
                    or _cvt(nodes[-1]) or _cvt(nodes[0]))

        lic = set(center_names)
        for pe in (bres.get("pe_list") or []):
            if not isinstance(pe, dict):
                continue
            for p in pe.values():
                for tp in ((getattr(p, "tree_data", None) or {}).get("paths")
                           or []):
                    if _path_admitted(tp):
                        for n in (tp.get("nodes") or []):
                            lic.add(str(n))
                            lic.add(_head(n))
        kept_triples = [tr for tr in all_triples
                        if len(tr) == 3
                        and (str(tr[0]) in lic or str(tr[2]) in lic)]
        kept_candidates = [c for c in candidates if str(c) in lic]
    else:
        lic = set(center_names)
        frontier = list(center_names)
        while frontier:
            x = frontier.pop()
            for h, r, t in edges:
                for a, b in ((h, t), (t, h)):
                    if a == x and b not in lic:
                        if (r in subs or _cvt(a) or _cvt(b)
                                or _transparent(r, a, b)):
                            lic.add(b)
                            frontier.append(b)
        kept_triples = [tr for tr in all_triples
                        if len(tr) == 3
                        and (str(tr[1]) in subs or _cvt(tr[0]) or _cvt(tr[2]))
                        and (str(tr[0]) in lic or str(tr[2]) in lic)]
        kept_candidates = [c for c in candidates if str(c) in lic]
    # filtered pe_list copy for the renderers (V38 reads pe_list directly)
    import copy as _copy

    def _node_ok(n):
        s = str(n)
        if s in lic or _cvt(s):
            return True
        head = _head(s)
        return head != s and (head in lic or _cvt(head))

    bres2 = dict(bres)
    pe2 = []
    for pe in (bres.get("pe_list") or []):
        if not isinstance(pe, dict):
            pe2.append(pe)
            continue
        pe2.append({lbl: p for lbl, p in pe.items()})
    for pe in pe2:
        if not isinstance(pe, dict):
            continue
        for lbl, p in pe.items():
            tris = [tr for tr in (getattr(p, "triples", None) or [])
                    if len(tr) == 3
                    and (path_mode or str(tr[1]) in subs
                         or _cvt(tr[0]) or _cvt(tr[2]))
                    and (str(tr[0]) in lic or str(tr[2]) in lic)]
            cands = [c for c in (getattr(p, "candidates", None) or [])
                     if str(c) in lic]
            try:
                p.triples = tris
                p.candidates = cands
            except AttributeError:
                pass
            # tree-path nodes carry the RENDER form 'm.xxx: [attr=.., ..]' —
            # strip the attribute tail before the lic/cvt test, or every
            # CVT-mediated path fails both tests and the V38 renderer (which
            # synthesizes its rows from these paths) goes empty (537 specimen:
            # --film--> edges present in kept_triples yet "triples: (empty)").
            # MUST stay INSIDE the lbl loop: an empty pattern-dict otherwise
            # reads a stale/unbound p (UnboundLocalError killed the whole
            # dispatch — 48 crashes / 14 cases, 2026-09-07 audit).
            td = getattr(p, "tree_data", None)
            if isinstance(td, dict):
                td["paths"] = [tp for tp in (td.get("paths") or [])
                               if (_path_admitted(tp) if path_mode else
                                   all(_node_ok(n) for n in (tp.get("nodes") or [])))]
    bres2["pe_list"] = pe2
    return kept_triples, kept_candidates, bres2


_SG_SERVED_CAP = 4000        # cached finalize result per semantic walk key —
                             # bounds _sg_served memory across an episode's
                             # walks while keeping the repeat reply's
                             # cached_evidence field self-sufficient


def _ev_record_canon_fid(treq, ctx, fid: str) -> str:
    """SHADOW LEDGER (phase-2, additive): canonical fid for the call's
    EvidenceRecord — MIRRORS the resolution at the pool-write site below
    (_fid_c in _sg_finalize). Inputs (treq fid_c / fact_key_map / fid) are
    fixed for the call and ctx._last_canon_fid is only written AFTER that
    late block, so this early evaluation yields the identical value. Keep
    the two in sync; the late one stays authoritative for live behavior."""
    _fkm = getattr(ctx, "fact_key_map", None) or {}
    _fc = str(treq.get("fid_c") or "") or _fkm.get(fid, "")
    if not _fc and not fid:
        _fc = getattr(ctx, "_last_canon_fid", "") or ""
    if not _fc:
        _fc = fid
    return _fc


def _sg_finalize(treq, bres, ctx) -> str:
    """C段: correction result render OR the post-walk CPU — per-center
    accumulation, CVT-neighbor pool seeding, canonicalization, record-centric
    rendering, per-subgraph evidence sets."""
    if "corr_cands" in bres:
        return _json_result({
            "entity_error": f"'{bres['bad_name']}' is not an entity in the graph.",
            "candidates": bres["corr_cands"],
            "note": "Pick the correct entity from `candidates` (each shows NEIGHBOR "
                    "relations to disambiguate). Re-call THIS retrieve_subgraph with the "
                    "correct entity + your selected relations. CROSS-SUBGRAPH CENTER: "
                    "if this fact's RELATIONS are right but its anchor entity is wrong, "
                    "re-call with a center BORROWED from any prior subgraph's evidence "
                    "(e.g. an earlier fact's binding) plus these relations. WORKFLOW: "
                    "retrieve_relations → retrieve_subgraph per fact."})
    if bres.get("no_centers"):
        entities = treq["entities"]
        return _json_result({"error": (f"none of {entities} appeared in a prior retrieve_subgraph. "
                                       f"A center MUST be an entity from a previous retrieve_subgraph's "
                                       f"triples (or the plan anchor for fact 1) — ANY prior subgraph's "
                                       f"entities are legal centers (cross-subgraph borrowing). If any of "
                                       f"these is a variable binding, pass the ?VARIABLE "
                                       f"(center: [\"?var\"]) — the runtime expands it to all declared "
                                       f"bindings.")})
    centers = treq["centers"]
    entities = treq["entities"]
    rel_names = treq["rel_names"]
    fid = treq["fid"]
    _nudge = treq["nudge"]
    skipped = treq["skipped"]
    prior_triples = treq["prior"]
    # SEMANTIC IDEMPOTENCE: mark this (center-set, relation-set) as served
    # (see the check in _sg_prepare — same resolved walk returns cached note).
    # Wave-2 (2026-09-20): the mark is the finalize RESULT STRING (capped at
    # _SG_SERVED_CAP), written at each return below — a repeat hit replays the
    # cached evidence (cached_evidence field), not just the advice to re-read
    # what the context may already have dropped.
    _sem_key = tuple(sorted((ci, frozenset(treq.get("rel_idxs") or ()))
                            for _n, ci in (treq.get("centers") or ())))

    def _mark_served(_res_str: str) -> None:
        _served = getattr(ctx, "_sg_served", None)
        if _served is None:
            _served = {}
            ctx._sg_served = _served
        _served[_sem_key] = _res_str[:_SG_SERVED_CAP]
    # per-center walk; accumulate evidence inline, collect PatternEvidence for the
    # cross-center merged display (one block per relation pattern, all roots under it)
    candidates, all_triples = [], []
    collected = []   # (center_name, PatternEvidence)
    _TOP_PATTERNS = 5   # keep only the top-N patterns (ordered by the walk: path length +
                        # relation hit) per center — excess patterns are noise; the walk already
                        # prunes partial-pattern paths (full-pattern only), this caps the count.
    # SUBMITTED-TERMINAL EVIDENCE DISCIPLINE (user design 2026-09-11,
    # Ron-Howard awards specimen): the walk keeps loose bridge targets for
    # REACH (traversal-only measured mismatch 39 vs 1 — the loose terminal
    # IS the mismatch fix), but the EVIDENCE admits only paths TERMINATED
    # by the model's SUBMITTED relations: bridge carrier edges may appear
    # mid-path, yet segments ENDING at a bridge relation (every award edge
    # in the 3-hop environment) are not answer paths — they are what blew
    # tens of triples into hundreds on hub centers and, over turns, the
    # context. Filter AFTER _accumulate (legality pool keeps the full walk
    # enumeration — the answer gate is unchanged), BEFORE pattern
    # collection for render.
    _sub_terms = set()
    for _exp in (treq.get("attr_expansion") or {}).values():
        for _dn in (_exp.get("direct") or []):
            _sub_terms.add(".".join(str(_dn).rsplit(".", 2)[-2:]))
    _has_bridges = (os.environ.get("SEQ_SUBTERM", "0") == "1"
                    and any(_exp.get("bridge") for _exp in
                            (treq.get("attr_expansion") or {}).values()))

    def _pe_filter(p):
        """Keep only tree paths whose LAST relation is a submitted terminal
        (+ their CVT-penetration edges). Returns a filtered PatternEvidence
        copy, or the original when nothing to filter (no bridges, or the
        filter keeps everything)."""
        if not _has_bridges or not _sub_terms:
            return p
        from kgqa.stages.formatting import PatternEvidence as _PE
        paths = (p.tree_data or {}).get("paths") or []
        keep_p = [tp for tp in paths
                  if tp.get("relations")
                  and ".".join(str(tp["relations"][-1]).rsplit(".", 2)[-2:])
                  in _sub_terms]
        if len(keep_p) == len(paths):
            return p
        keep_nodes = {str(_pn.split(":", 1)[0].strip())
                      for tp in keep_p
                      for _pn in (tp.get("nodes") or [])}
        keep_tr = []
        for tr in (p.triples or []):
            if len(tr) != 3:
                continue
            h, _r, t = str(tr[0]), str(tr[1]), str(tr[2])
            if h in keep_nodes or t in keep_nodes:
                keep_tr.append(tr)
        keep_c = [c for c in (p.candidates or []) if str(c) in keep_nodes]
        if not keep_p and not keep_tr:
            return None
        return _PE(label=p.label, readable=p.readable,
                   candidates=keep_c, triples=keep_tr,
                   tree_data={"paths": keep_p})

    for (e, i), pe in zip(centers, bres["pe_list"]):
        if not pe:
            continue
        # ① feeding guarantee (see _select_patterns_for_render): the top-N
        # cut alone could starve a center's direct selected-relation edges
        # when its walk produced many higher-ranked detour patterns.
        for p in _select_patterns_for_render(pe.values(), e, rel_names,
                                             _TOP_PATTERNS):
            _accumulate(ctx, i, p)      # legality pool: FULL walk output
            pf = _pe_filter(p)
            if pf is None:
                continue
            collected.append((e, pf))
            for tr in (pf.triples or []):
                if len(tr) == 3:
                    all_triples.append(tr)
            for c in (pf.candidates or []):
                if not is_cvt_like(c) and c not in candidates:
                    candidates.append(c)
    # EVIDENCE-LOG APPEND (redesign ZEROETH RULE, first wiring 2026-09-24):
    # one EvidenceSet per call, built from the confirmed chains (structured
    # edge/record ownership). Downstream consumers migrate onto it — the
    # answer pool reads it FIRST (tools._do_answer), idempotency and RSCC
    # attribution follow. The rendered text is a projection; nothing reads
    # it back.
    # F01 (Codex audit 2026-09-30, approved stopbleed): the builder ran with
    # TWO construction bugs its whole life — `_inv` was never defined
    # (NameError whenever any edge existed) and the pattern loop iterated
    # the multistep OUTER dict (center-index ints -> TypeError on any
    # multi-step call) — both swallowed by the blanket except below, so the
    # log was permanently empty. Fixed: inv map from _ctx_inverse_rels (the
    # same map the render delta uses), pattern iteration over the inner
    # per-center pattern keys, and the witness-edge union from the rebuild
    # lane (F02) rides along. The except stays ONLY as a safety net and
    # must never fire in normal operation — it now prints a stack trace to
    # stderr instead of passing silently.
    try:
        from kgqa.agent.evidence import EvidenceSet, EvidenceLog
        _chains_flat = []
        for _pw, _chs in (treq.get("confirmed") or {}).items():
            _chains_flat.extend(_chs)
        _ev = EvidenceSet()
        _ev.root = str(ctx.ents[treq["centers"][0][1]]
                       ) if treq.get("centers") else ""
        # multistep = {center_idx: {pattern_key: hop-frozensets}} — the
        # pattern key is the INNER dict key (tuple of rel idxs); iterating
        # the outer dict yields center ints (the old TypeError).
        for _ci, _pats in (treq.get("multistep") or {}).items():
            for _pk in (_pats or {}):
                _ev.patterns.append(" ⭢ ".join(
                    ".".join(str(ctx.rels[_r]).rsplit(".", 2)[-2:])
                    for _r in _pk))
        # edge source: confirmed chains + the rebuild lane's witness union
        # (F02: last-hop discovering edges that never enter a backbone
        # chain still belong to this call's evidence).
        _edge_idx_rows = [(_h, _r, _t)
                          for _ch in _chains_flat
                          for (_h, _r, _t) in (_ch.get("full_edges") or ())]
        _edge_idx_rows.extend(treq.get("_witness_edges") or ())
        for (_h, _r, _t) in _edge_idx_rows:
            _hname, _rname, _tname = (str(ctx.ents[_h]),
                                      str(ctx.rels[_r]), str(ctx.ents[_t]))
            if (_hname, _rname, _tname) not in _ev.edges:
                _ev.edges.append((_hname, _rname, _tname))
            from kgqa.agent.entity_kinds import entity_kind as _ek2
            for _nm in (_hname, _tname):
                if _ek2(_nm) == "NAMED":
                    _ev.endpoints.add(_nm)
            if _rname.rsplit(".", 1)[-1] not in ("type", "types") \
                    and _ek2(_tname) == "LITERAL":
                _ev.records.setdefault(_hname, {}).setdefault(
                    _rname.rsplit(".", 1)[-1], []).append(_tname)
        _ev_inv = _ctx_inverse_rels(ctx)
        _ev.delivered_keys = {_edge_fact_key(h, r, t, _ev_inv)
                              for (h, r, t) in _ev.edges}
        # CVT bracket attrs (§5.6 CVT-attr carrier channel): mirror the v38
        # renderer — attrs reconstructed from the case graph for every CVT
        # this call's evidence touches, MERGED with the walked-literal
        # records of the EvidenceSet.
        from kgqa.agent.entity_kinds import is_cvt as _ek_cvt
        from kgqa.agent.evidence import (backfill_case_attrs as _bca,
                                         cvt_case_attrs)
        _ev_cvt_names = {n for (_a, _r, _b) in _ev.edges
                         for n in (_a, _b) if _ek_cvt(n)}
        _ev_cvt_names |= {n for n in _ev.records}
        _ev_attrs = cvt_case_attrs(_ev_cvt_names, ctx.h_ids, ctx.r_ids,
                                   ctx.t_ids, ctx.ents, ctx.rels)
        # R3-b (§6.8-2, SEQ_RENDER_V3): CVT auto-expansion IS part of
        # materialization — this pull runs at evidence-build time and the
        # renderer PROJECTS the stashed store (the PRE-merge case channel;
        # walked-literal records ride their own value rows, merging them
        # here would double-display). The v38 renderer's own case-graph
        # scan is its OFF-mode legacy source.
        _ev_case_attrs = None
        from kgqa.agent.views import render_v3 as _render_v3_fn
        if _render_v3_fn():
            _ev_case_attrs = {c: {k: list(vs) for k, vs in ka.items()}
                              for c, ka in _ev_attrs.items()}
            treq["_ev_case_attrs"] = _ev_case_attrs
        for _c, _ka in _ev.records.items():
            for _k, _vs in _ka.items():
                _ev_attrs.setdefault(_c, {}).setdefault(_k, [])
                for _v in _vs:
                    if _v not in _ev_attrs[_c][_k]:
                        _ev_attrs[_c][_k].append(_v)
        # R01 (Codex audit 2026-10-01): the case-graph attr channel is part
        # of the LIVE delivery too — backfill the merged attrs into the
        # EvidenceSet's records so EvidenceLog.answer_pool() carries the named
        # bracket values. Until now these values were shadow-side only
        # (build_evidence_record's records_attrs), leaving value-form gold
        # outside the live answer pool (2784 'Juice' specimen: walked edges
        # hold only Actor→m.cvt, the film= value arrives via this channel).
        # Delivery-enters-the-domain (unified triple layer): live pool and
        # shadow observed view now share ONE domain. SCOPE NOTE (Codex
        # warning, unresolved): which attrs legally belong to THIS fact has
        # no retrieval rule yet — cvt_case_attrs pulls every non-noisy,
        # non-CVT, <60-char attr of the touched CVTs, and the backfill keeps
        # that scope verbatim (live/shadow parity); narrowing it is a later
        # ruling, NOT part of D3-0. Render is untouched (v38 rebuilds its
        # brackets from the case graph itself, never reads this).
        _bca(_ev.records, _ev_attrs)
        _log = getattr(ctx, "evidence_log", None)
        if _log is None:
            _log = ctx.evidence_log = EvidenceLog()
        _log.append(_ev)
        # ── SHADOW LEDGER (phase-2, 2026-09-30, additive-only) ─────────────
        # One EvidenceRecord per call, appended right beside the EvidenceSet
        # (1:1 pairing). Purely additive: no live mechanism reads it; the
        # rollout sidecar (SEQ_EVIDENCE_SIDECAR) ships it for offline shadow
        # judgment. Failure policy = the F01 convention above (stderr stack,
        # never fail the call).
        from kgqa.agent.evidence import (EvidenceRecordLog,
                                         build_call_ledger)
        _rlog = getattr(ctx, "evidence_records", None)
        if _rlog is None:
            # R1/R04: the attempt ordinal comes from the append-only archive
            # (reset_seq_episode_state appends there before clearing this
            # slot) — attempt-2 ledgers know they are attempt-2.
            _rlog = ctx.evidence_records = EvidenceRecordLog(
                attempt_id=len(getattr(ctx, "evidence_archive", None) or []))
        _fid_raw = str(treq.get("fid") or "")
        _canon = _ev_record_canon_fid(treq, ctx, _fid_raw)
        # truncation watermark: _rebuild_paths counts its budget cuts on ctx;
        # the per-call delta = current − watermark (self-maintaining, no
        # touch of prepare/execute).
        _tr_now = int(getattr(ctx, "_ev_rebuild_truncs", 0) or 0)
        _tr_delta = _tr_now - int(getattr(ctx, "_ev_rebuild_trunc_wm", 0) or 0)
        ctx._ev_rebuild_trunc_wm = _tr_now
        # post-delivery frontier = this call's named chain terminals (the
        # surface the next continuation walks from). R1/R07: the ledger ALSO
        # keeps the FULL structural frontier (terminal node idxs) — the ≤50
        # named list below is the truncated PREVIEW face the legacy record
        # has always carried.
        _fr_names = []
        _fr_ids = []
        _fr_id_seen = set()
        for _chg in (treq.get("confirmed") or {}).values():
            for _chg_ch in (_chg or ()):
                if not _chg_ch.get("full_nodes"):
                    continue
                _tidx = _chg_ch["full_nodes"][-1]
                _tn = str(ctx.ents[_tidx]) if 0 <= _tidx < len(ctx.ents) else ""
                if (_tn and not _cvt_like_name(_tn) and _tn not in _fr_names):
                    _fr_names.append(_tn)
                if _tidx not in _fr_id_seen:
                    _fr_id_seen.add(_tidx)
                    _fr_ids.append(int(_tidx))
        # R05 companion map: full relation names behind the bracket-attr
        # rows — the attr lane's fact identity is (carrier, full.rel, value)
        # and the short key stays a display form of the SAME fact. V3-04:
        # MULTI-VALUED — two complete relations sharing one short key each
        # materialize as their own SourceFact in the ledger; the renderer's
        # display face keeps ONE representative per row (the last-scanned
        # full rel, the pre-V3 map's value — display bytes unchanged).
        from kgqa.agent.evidence import (cvt_case_attr_full_rels,
                                         cvt_case_attr_stats)
        _attr_full = cvt_case_attr_full_rels(_ev_cvt_names, ctx.h_ids,
                                             ctx.r_ids, ctx.t_ids,
                                             ctx.ents, ctx.rels)
        # V3-04 completeness face: the ≥60-char attr values the retrieval
        # contract excludes at materialize time (never delivered — the
        # count keeps the exclusion honest, it is NOT display truncation).
        _attr_stats = cvt_case_attr_stats(_ev_cvt_names, ctx.h_ids,
                                          ctx.r_ids, ctx.t_ids,
                                          ctx.ents, ctx.rels)
        if _render_v3_fn():
            # R05 companion for the render's attr dedup identity (F09 face):
            # the bracket row's identity is (carrier, full.rel, value); the
            # short key stays its display form.
            treq["_ev_attr_full_rels"] = _attr_full
        # (CVT bracket attrs + their R01 live backfill ran ABOVE, before the
        # EvidenceSet entered the log — _ev_attrs is ready for the record.)
        # R1 THREE-LAYER (2026-10-01): build_call_ledger calls the UNCHANGED
        # build_evidence_record for the legacy projection, then enriches it
        # into SourceFact/Observation/CallCommit. begin_call opens the
        # ATOMIC commit (PENDING): its identities only enter delivered
        # history on the SUCCESS finalize at the pool-write point below —
        # the NO_EVIDENCE early return finalizes NO_EVIDENCE instead, so a
        # failed call consumes no NEW (R03, the 1/309 corner).
        _ev_tool_idx = sum(1 for _st in (getattr(ctx, "trajectory", None)
                                         or ()) if _st.get("role") == "tool")
        _draft = build_call_ledger(
            seq=len(_rlog.records), canon_fid=_canon,
            center=_ev.root,
            submitted_rel_idxs=(treq.get("rel_idxs") or ()),
            rel_full_names=ctx.rels, inv=_ev_inv,
            confirmed=treq.get("confirmed"),
            witness_edges=(treq.get("_witness_edges") or ()),
            ents=ctx.ents, rels=ctx.rels,
            records_attrs=_ev_attrs, attr_full_rels=_attr_full,
            prior_episode=_rlog.prior_episode(),
            prior_tree=_rlog.prior_tree(_canon),
            truncations=_tr_delta, render_lines_dropped=0,
            frontier_names=_fr_names[:50], frontier_node_ids=_fr_ids,
            frontier_total=len(_fr_names),
            attempt_id=_rlog.attempt_id,
            tool_message_idx=_ev_tool_idx,
            # V3-03: per-edge discovery directions — facts stage in RAW
            # graph orientation, traversal_dir rides the usages.
            edge_dirs=(treq.get("_edge_dirs") or None),
            # V3-01: ALL walk anchors (the projection's anchor-exclusion
            # set).
            centers=[str(_cn) for (_cn, _ci2) in (treq.get("centers")
                                                  or ())],
            # V3-04: attr-lane retrieval-contract exclusions (≥60 values).
            attr_stats=_attr_stats)
        _rlog.begin_call(_draft)
        # RULING ② dependency provenance (2026-10-01): the call's centers
        # are the walk's seeds — a center an EARLIER delivered call first
        # delivered makes THIS call depend on it (its usage of the fact
        # space dies in that call's counterfactual; an INDEPENDENT
        # re-retrieval — anchored on a question entity — survives). Every
        # earlier commit of the SAME tree is a dependency too (a
        # continuation's frontier walk rides the tree's own deliveries).
        try:
            _dep = set()
            _first_del = {}
            for _pc in _rlog.delivered_commits():
                _ends = getattr(_pc, "_dep_endpoints", None)
                if _ends is None:
                    _ends = set()
                    for _ob2 in (_pc.observations or ()):
                        _f2 = _rlog.facts.get(_ob2.fact_key)
                        if _f2 is not None:
                            _ends.add(normalize(str(_f2.h_id)))
                            _ends.add(normalize(str(_f2.t_id)))
                    try:
                        _pc._dep_endpoints = _ends
                    except Exception:
                        pass
                for (_cn3, _ci3) in (treq.get("centers") or ()):
                    if normalize(str(_cn3)) in _ends:
                        _dep.add(_pc.call_id)
                if _pc.canon_fid == _canon:
                    _dep.add(_pc.call_id)
                for _ob3 in (_pc.observations or ()):
                    _first_del.setdefault(_ob3.fact_key, _pc.call_id)
            for _ob4 in _draft.observations:
                _ob4.depends_on = sorted(_dep)
                _ob4.origin_call_id = (_first_del.get(_ob4.fact_key)
                                       or _draft.commit.call_id)
        except Exception:
            import sys as _sys_dep
            import traceback as _tb_dep
            print(f"[evidence-log] depends_on fill failed: "
                  f"{_tb_dep.format_exc()}", file=_sys_dep.stderr)
        treq["_ev_record"] = _draft.legacy
        treq["_ev_commit"] = _draft.commit
    except Exception:
        # safety net ONLY — surface it (stack trace to stderr), never fail
        # the tool call on the ledger's account (behavior policy = phase 2+)
        import sys as _sys_ev
        import traceback as _tb_ev
        print(f"[evidence-log] build failed: {_tb_ev.format_exc()}",
              file=_sys_ev.stderr)
    # NOTE: leaf-set full enumeration now lives in the evidence builder
    # (build_pattern_evidence_triples, Option B) — same-prefix/different-leaf
    # paths are one pattern with N leaves; the support-path cap no longer
    # bounds leaf cardinality. No parallel enumeration here.
    # ① walk-side feeding guarantee (see _guarantee_center_direct_edges): the
    # renderer's tier-1 rows synthesize from the centers' OWN selected-relation
    # edges — those edges must be in all_triples whatever the walk's support
    # caps did to their patterns (inverse 1-hop families were silently capped).
    # Under the submitted-terminal discipline the guarantee is scoped to the
    # SUBMITTED relations (full rel_idxs would re-admit bridge edges).
    if _has_bridges and _sub_terms:
        _guar_rels = set()
        for _exp in (treq.get("attr_expansion") or {}).values():
            for _dn in (_exp.get("direct") or []):
                if _dn in ctx.rels:
                    _guar_rels.add(ctx.rels.index(_dn))
        _guar_idxs = sorted(_guar_rels) or treq["rel_idxs"]
    else:
        _guar_idxs = treq["rel_idxs"]
    for tr in _guarantee_center_direct_edges(ctx, centers, _guar_idxs,
                                             all_triples):
        all_triples.append(tr)
        for x in (tr[0], tr[2]):
            if not is_cvt_like(x) and x not in candidates:
                candidates.append(x)
    if os.environ.get("SEQ_LICENSE_FILTER", "1") != "0":
        # ANSWER-LEGALITY LEDGER (user ruling 2026-09-08): record every entity
        # the walk enumerated BEFORE the display filter — the answer may be any
        # entity that appeared in the walk, not only what the render showed
        # (the filter is anti-drift for the model's eyes, not an answer gate).
        _seen = getattr(ctx, "walk_seen_entities", None)
        if _seen is None:
            _seen = ctx.walk_seen_entities = []
        _seen_len0 = len(_seen)      # PATTERN-STATE provenance: this call's delta
        _seen_n = {str(c) for c in _seen}
        for _tr in all_triples:
            if len(_tr) == 3:
                for _x in (_tr[0], _tr[2]):
                    if str(_x) not in _seen_n:
                        _seen_n.add(str(_x))
                        _seen.append(str(_x))
        for _c in candidates:
            if str(_c) not in _seen_n:
                _seen_n.add(str(_c))
                _seen.append(str(_c))
        # PATTERN-STATE (user design 2026-09-10): remember THIS walk's node set
        # (the ledger delta) per fact — when the model binds a ?var from this
        # fact, later single-?var calls from that var continue with this
        # territory as the prefix mask instead of re-walking it per binding.
        _fp = getattr(ctx, "fid_pattern", None)
        if _fp is None:
            _fp = ctx.fid_pattern = {}
        _fp[fid] = {
            "nodes": frozenset(str(x) for x in _seen[_seen_len0:]),
            "single_center": (centers[0][1] if len(centers) == 1 else None),
        }
        all_triples, candidates, bres = _display_license_filter(
            treq, bres, centers, all_triples, candidates)
    if not all_triples:
        # WALK-NOTHING (Wave-1): observation only — the old text asserted
        # "the fact is right" (a verdict an empty walk cannot support) and
        # jumped straight to relation substitution. Layered diagnosis,
        # cheapest layer first: unused sibling relations → one reworded
        # retrieve_relations → borrow a center from earlier evidence →
        # close the fact as mismatched and continue.
        _err = {"error": "NO_EVIDENCE: this selection delivered no advancing triples. "
                "Repeating the unchanged query cannot add evidence. Choose a relevant unused "
                "relation or a supported center; if neither can advance this fact, "
                "declare it exhausted and continue. This does not prove the answer is absent.",
                "entities": entities, "relations": rel_names}
        if getattr(ctx, "_adaptive_worker", False):
            _err["error"] = ("NO_EVIDENCE: this executed query delivered no advancing triples — "
                             "the query ran; nothing new was delivered.")
            _err["advice"] = ("A missing record after an appropriate execution is UNKNOWN data, "
                              "not proof of absence and not a reason to keep switching paths. "
                              "Preserve the candidates already delivered; repair only a concrete "
                              "executable error, otherwise move to answer-mode comparison and "
                              "fallback — do not treat missing confirmation as a global no-answer.")
        _ne_str = _json_result(_err)
        _mark_served(_ne_str)   # NO_EVIDENCE walks are served too — a repeat
        # SHADOW LEDGER (phase-2, additive): the call delivered nothing —
        # mark the record's outcome so the offline aligner (blocks ↔ records)
        # can skip it.
        _evr_ne = treq.get("_ev_record")
        if _evr_ne is not None:
            _evr_ne.outcome = "no_evidence"
            _evr_ne.completeness["status"] = "FAILED"
            _evr_ne.pool_delta = 0
        # R1 ATOMIC COMMIT (R03 root-out): the call's terminal state is
        # NO_EVIDENCE — its staged identities NEVER enter the delivered
        # history, so a later successful delivery of the same facts is
        # still first-delivery NEW (phase-2 order committed the ids at
        # build time, before this marker — the 1/309 corner).
        _rlog_ne = getattr(ctx, "evidence_records", None)
        if _rlog_ne is not None:
            _cmt_ne = treq.get("_ev_commit")
            if _cmt_ne is not None:
                _cmt_ne.pool_delta = 0
                _cmt_ne.completeness["retrieval"] = {
                    "status": "FAILED", "rebuild_truncations":
                        _cmt_ne.completeness.get("retrieval", {}).get(
                            "rebuild_truncations", 0)}
            _rlog_ne.finalize_call("NO_EVIDENCE")
        return _ne_str          # replays the diagnosis instead of re-walking

    # ensure ALL tree-visible entities (incl CVT-attr entities pe.triples may miss) are answerable
    _collect_cvt_neighbors_to_pool(ctx)

    # ── relation-grouped tree (replaces pattern-based _format_merged) ──────────────
    # First flip any triples whose direction is reversed vs the raw graph (the undirected
    # walk records center-first, which reverses relations pointing INTO the center, e.g.
    # mascot.team rendered as 'team --team--> mascot'). Then resolve CVT-mediated edges
    # → named→named, deduped. Inverse / variant relations that surfaced the SAME entity
    # pair via N patterns collapse to ONE edge per entity pair here.
    all_triples = _canonicalize_triples(all_triples, ctx)
    # record-centric rendering (replaces _resolve_cvt_edges + _merge_edges):
    # CVT-with-holder → record `holder → title [from=..; to=..]` (direction-normalized);
    # CVT-without-holder → flattened named→named; direct edges → merged.
    # CROSS-CALL DEDUP REMOVED (user ruling, 2026-08-20): the old suppression
    # (ctx.shown_edges persisting across calls, keyed (h,r,t)) was born as an
    # economy hack for walk-bloated subgraphs; admission now lives in the walk
    # layer (pattern-path discipline). Suppressing a triple that is a link of
    # THIS subgraph's pattern path severs the path and masks entities from an
    # edge's displayed extension (the Faroese/Denmark bug: sg2 rendered only
    # the new tails of Denmark--languages_spoken, hiding the discriminating
    # Faroese|Danish). The set is now per-call: duplicates arriving via multi-
    # ple channels/patterns within ONE call still collapse; every subgraph
    # re-renders the full evidence its own pattern paths justify.
    shown = set()
    from kgqa.core.utils import phase_timer
    from kgqa.agent.views import render_v3 as _render_v3_fn
    _v3_on = _render_v3_fn()
    with phase_timer("render"):
        if _v3_on:
            # per-call display-truncation scratch (§6.6/F03): the row layer
            # fills it; None until then (a served-cache or legacy-path call
            # contributes no display events)
            ctx._render_display_stats = None
        # EVIDENCE DISPLAY (2026-08-24, user ruling V3): DENSE legacy detail
        # lines (h --r--> t1 | t2 | …, CVT records inline [attr=val], 120 cap
        # + branch refs) + a CAPPED pattern overview (≤3 shapes per relation,
        # path-terminal first, start-anchored). The V2 wide-table experiment
        # (SEQ_RENDER_V2=1) lost density and hid CVT attrs — kept for A/B only.
        _al = next((e for e in entities if str(e).startswith("?")), None)
        _paths = []
        for _e2, _p2 in collected:
            _paths.extend((_p2.tree_data or {}).get("paths") or [])
        if os.environ.get("SEQ_RENDER_V38", "1") == "1":
            # V3.8 (2026-09-02, user + Codex design ①): TRIPLE-FORM rows —
            # only two legal compression shapes (same h+r→many tails /
            # many heads→same r+t); CVT attrs inline; no pattern-paths
            # overview, no forced candidates line (row entities are
            # candidates). Walk-faithful edges, V37h dedup discipline.
            from kgqa.agent.seq_render_v38 import render_v38_ack
            v38 = render_v38_ack(treq, bres, ctx)
            if v38 and v38 != "(empty)":
                tree_lines = v38.split("\n")
                _ov, _blocks, _covered = [], [], set()
            else:
                _ov, _blocks, _covered = render_evidence_sections(
                    _paths, [e for e, _i in centers], _al, all_triples,
                    rel_names or [])
                _uncovered = _select_uncovered(all_triples, _covered,
                                               set(rel_names or []))
                tree_lines, n_overlap = _render_records(_uncovered, shown)
                if tree_lines:
                    tree_lines = _group_facts_by_rel(tree_lines)
        elif os.environ.get("SEQ_RENDER_V37", "") == "1":
            # V3.7 (2026-08-31): candidate-centric lines — the candidate leads,
            # its instantiated pattern path follows (user design: Iowa ◂
            # Missouri River ←partially_contains─ m.xxx ─partially_contained_by→).
            # Entity-terminal-first combo selection + all centers render.
            from kgqa.agent.seq_render_v37 import render_v37_ack
            v37 = render_v37_ack(treq, bres, ctx)
            if v37 and v37 != "(empty)":
                tree_lines = v37.split("\n")
                _ov, _blocks, _covered = [], [], set()
            else:
                _ov, _blocks, _covered = render_evidence_sections(
                    _paths, [e for e, _i in centers], _al, all_triples,
                    rel_names or [])
                _uncovered = _select_uncovered(all_triples, _covered,
                                               set(rel_names or []))
                tree_lines, n_overlap = _render_records(_uncovered, shown)
                if tree_lines:
                    tree_lines = _group_facts_by_rel(tree_lines)
        elif os.environ.get("SEQ_RENDER_V36", "") == "1":
            from kgqa.agent.seq_render_v36 import render_v36_ack
            v36 = render_v36_ack(treq, bres, ctx)
            if v36 and v36 != "(empty)":
                tree_lines = v36.split("\n")
                _ov, _blocks, _covered = [], [], set()
            else:
                _ov, _blocks, _covered = render_evidence_sections(
                    _paths, [e for e, _i in centers], _al, all_triples,
                    rel_names or [])
                _uncovered = _select_uncovered(all_triples, _covered,
                                               set(rel_names or []))
                tree_lines, n_overlap = _render_records(_uncovered, shown)
                if tree_lines:
                    tree_lines = _group_facts_by_rel(tree_lines)
        elif os.environ.get("SEQ_RENDER_V35", "") == "1":
            # V3.5 (2026-08-31): relation-signature groups + name-trie with
            # R3 sibling merge + zero-loss folds + CVT attribute brackets
            from kgqa.agent.seq_render_v35 import render_v35_ack
            v35 = render_v35_ack(treq, bres, ctx)
            if v35 and v35 != "(empty)":
                tree_lines = v35.split("\n")
                n_overlap = 0
        elif os.environ.get("SEQ_RENDER_V2", "") == "1":
            from kgqa.stages.evidence_display import render_records_compat
            tree_lines, n_overlap = render_records_compat(
                all_triples, shown, anchors=[e for e, _i in centers],
                anchor_label=_al, patterns=_paths)
        else:
            # V3.3 (2026-08-25 redesign): ONE unified renderer organizes ALL
            # structured evidence — 1-hop blocks per selected relation first,
            # then k-hop blocks length-ascending (principle 1), each block
            # holding only instances that COMPLETE its path shape,
            # reconstructed from the canonicalized edges (principle 2).
            # Facts = legacy dense remainder (_render_records), fed by
            # _select_uncovered (within-call dedup + CVT attr revival — the
            # Kim Richards specimen: selection filtered the record attrs,
            # every remaining edge collapsed as a bare-CVT tail, and the
            # fresh shown-set counted the duplicates as "already shown in a
            # prior subgraph" on the case's FIRST retrieval).
            if os.environ.get("SEQ_RENDER_V35", "") == "1":
                _ov, _blocks, _covered = [], [], set()
                tree_lines = v35.split("\n")
            else:
                _ov, _blocks, _covered = render_evidence_sections(
                    _paths, [e for e, _i in centers], _al, all_triples,
                    rel_names or [])
            _uncovered = _select_uncovered(all_triples, _covered,
                                           set(rel_names or []))
            tree_lines, n_overlap = _render_records(_uncovered, shown)
            # REMAINDER BY RELATION (user rulings 2026-08-25, Debussy + naming
            # specimens): the leftover section is per-relation grouped and its
            # name is decoupled from the plan's fact vocabulary (f1/f2/sg) —
            # "facts" as a section name is BANNED; selected-relation edges were
            # already swept into blocks, so this holds only true remainders
            if tree_lines:
                tree_lines = _group_facts_by_rel(tree_lines)
            # V4 semantic layers (user ruling: candidates are the core; the
            # answer ROLE is harness-known from the fact's ?var slot).
            # SEQ_EVIDENCE_LAYERS=1 enables; DEFAULT OFF after two cohort
            # runs showed the header costs ~4pp without gain (V4 0.6634 /
            # V4b 0.6490 vs V3.2 0.7037) — kept as experimental arm.
            _layers_on = os.environ.get("SEQ_EVIDENCE_LAYERS", "0") == "1"
            _lay = (layer_evidence(fid, ctx, _paths, all_triples,
                                   centers, entities) if _layers_on else {})
            _head = []
            if _lay.get("fact"):
                _head.append(f"fact: {_lay['fact']}")
            if _lay.get("var") and _lay.get("answer"):
                _head.append(f"answer-role: {_lay['var']}")
                _head.append("── answer candidates (this subgraph) ──")
                # per-LAST-HOP-RELATION groups (Lala Anthony specimen): multi-
                # relation selection mixes terminals of different semantics
                # (place_of_birth → Red Hook vs people_born_here → its
                # residents) — attribute each group to its relation so the
                # model filters by the fact's semantics itself
                for _r3, _ts in _lay.get("by_rel") or []:
                    _shown = " | ".join(_ts[:40])
                    _more = f" …(+{len(_ts) - 40})" if len(_ts) > 40 else ""
                    _head.append(f"via {_r3}: {_shown}{_more}")
            if _ov or _blocks:
                tree_lines = ([f"  {l}" for l in _head] + ["pattern paths:"]
                              + [f"  {l}" for l in _ov]
                              + [f"  {l}" for l in _blocks]
                              + (["  ── other relations ──"] + tree_lines if tree_lines else [])
                              + ([f"  attribute values: {' | '.join(_lay['attrs'])}"]
                                 if _lay.get("attrs") else [])
                              + ([f"  source/intermediate: {' | '.join(_lay['source'])}"]
                                 if _lay.get("source") else [])
                              + ([f"  {_lay['var']} (this call): {_lay['n_answer']} entities"]
                                 if _lay.get("var") else []))
    # NO "already shown in a prior subgraph" message (user ruling 2026-08-20
    # restated 2026-08-25, Kim Richards specimen): cross-call suppression is
    # forbidden — every subgraph re-renders the full evidence its pattern
    # paths justify. n_overlap now only reflects _render_records' internal
    # record-key collapse (the record itself still renders), never a hidden
    # prior display.
    # GRAPH-STRUCTURE GATE attempted 2026-08-18 (Arizona audit: reworded-subquestion
    # iteration burning the budget) and REVERTED after measurement: unconditional
    # 2-strike −1.6pp, same-centers-conditioned −2.7pp (3-seed 0.7850 vs 0.8117
    # base). The nudge text perturbs the retrieval distribution more than the rare
    # budget-exhaustion event costs. Budget-exhaustion fallback belongs in the
    # answer rules (§18 fallback clause), not in the retrieval layer.
    # DISPLAY TRUNCATION RULING (user 2026-10-01, proposal §6.6): the ONLY
    # legal display truncation point is the MERGED EDGE GROUP — `h --rel-->
    # many tails` with >80 leaves, or `many heads --rel--> t` with >80 heads.
    # R3-b (SEQ_RENDER_V3, 2026-10-01) EXECUTES the ruling: this global
    # 200-line budget is ABOLISHED under V3 (the >80 group cut lives in the
    # row layer, seq_rows._cut_members, and its counted events flow into
    # completeness['display'] at the late fill below); the legacy budget and
    # its "... +N lines truncated" marker run only on the OFF face.
    if not _v3_on:
        _TREE_LINE_BUDGET = 200
        if len(tree_lines) > _TREE_LINE_BUDGET:
            dropped = len(tree_lines) - _TREE_LINE_BUDGET
            tree_lines = tree_lines[:_TREE_LINE_BUDGET]
            tree_lines.append(f"  ... +{dropped} lines truncated")
    # CANDIDATE PROVENANCE REMOVED (user ruling 2026-09-12, asked repeatedly):
    # the "candidates not shown above" annotation listed bare entity names —
    # valid information renders as TRIPLES (the subgraph's atom), anything
    # without a visible edge is removed. The answer-legality pool and the
    # wandered-candidate ledger below are unchanged.
    _fe = getattr(ctx, "fact_evidence", None)
    if _fe is None:
        _fe = {}; ctx.fact_evidence = _fe
    # WANDERED-CANDIDATE LEDGER (see _expand_entities): remember the walk
    # candidates that have NO visible edge in this render — they stay legal
    # answer entities and future ?var expansions carry them as centers
    _blob = "\n".join(tree_lines)
    _wx = getattr(ctx, "walk_extra", None)
    if _wx is None:
        _wx = ctx.walk_extra = []
    _wxset = {str(x) for x in _wx}
    for _c in candidates:
        _cs = str(_c)
        if (not is_cvt_like(_cs) and _cs not in _blob
                and _cs not in _wxset):
            _wxset.add(_cs)
            _wx.append(_cs)

    # The merged triples above ARE the evidence view — discriminator attrs (dates,
    # incumbent) live on their own edges ('Robert --to--> 1968', 'Ted --to-->
    # (incumbent)'), so the model compares candidates directly from the triples.
    # candidate_attrs / discriminating_attrs are intentionally NOT displayed: they
    # re-stated the same triples (tree/attrs overlap) and the present-vs-ABSENT flood
    # over-loaded the prompt. Builders kept (not discarded) for offline scoring.
    multi = (treq.get("n_resolved") or len(centers)) > 1
    # PER-SUBGRAPH EVIDENCE SET (2026-08-21, user proposal — hallucination check):
    # record the named entities THIS subgraph displayed, keyed by fact id. The
    # checkpoint validator rejects bindings that never appeared in the declaring
    # fact's subgraph (Kevin Costner specimen: the sg1.f1 re-declaration merged
    # the OTHER actor's films into sg1 — cross-subgraph contamination passed
    # the global pool check but dies against sg1's own evidence set).
    # R4 DISPOSAL STATUS (2026-10-01, specs/r4_cleanup_2026-10-01.md): KEEP —
    # the filter DOMAIN left this store in D3-0 (observed_episode judges), but
    # two readers remain: the strip receipt's "SUBMIT values drawn from {fid}'s
    # OWN evidence pool (includes: …)" display (seq_react_loop.
    # _halluc_strip_receipt) and the OFF-face join_path_rescue fallback
    # (tools.py, SEQ_RENDER_V3=0). Retire when the receipt switches to the
    # view layer's per-fid projection AND the render OFF face is retired.
    # MULTI-TREE JOIN KEY CANONICALIZATION (Team-A 2026-09-27): the three
    # join-side stores (fact_evidence / fact_candidate_pool /
    # fact_evidence_seq) must land on the CANONICAL declared fid the join
    # and the checkpoint handler read (fact_bindings / fact_vars /
    # _canonical_decl_fid → fact_key_map). The old bare dict lookup keyed a
    # walk invoked as `sg: sg2` under the unresolvable "sg2" — the SYSTEM
    # JOIN never saw that tree's pool (every cross-tree join silently
    # degraded to curated-list intersection), and the per-subgraph
    # hallucination filter + evidence clock were inert for bare-sg walks.
    # Resolution order: the harness-computed _canon_fid (norm_fact_key
    # parity: alias map, bare-sg group with center-head match) →
    # fact_key_map on the raw fid → for a call with NO explicit fact, the
    # MOST RECENTLY delivered subgraph's fid (Nearest Principle, user
    # 2026-09-26/27: multi-tree attribution follows the latest delivery) →
    # raw passthrough (inspectability, matches norm_fact_key).
    _fkm_l = getattr(ctx, "fact_key_map", None) or {}
    _fid_c = str(treq.get("fid_c") or "") or _fkm_l.get(fid, "")
    if not _fid_c and not fid:
        _fid_c = getattr(ctx, "_last_canon_fid", "") or ""
    if not _fid_c:
        _fid_c = fid
    ctx._last_canon_fid = _fid_c
    _fe = getattr(ctx, "fact_evidence", None)
    if _fe is None:
        _fe = {}; ctx.fact_evidence = _fe
    from kgqa.core.utils import normalize as _nz
    _fe[_fid_c] = _fe.get(_fid_c, set()) | \
        {_nz(c) for c in candidates if not is_cvt_like(c)} | \
        {_nz(x) for h, r, t in all_triples for x in (h, t) if not is_cvt_like(x)}
    # TOOL-SIDE CANDIDATE POOL (Wave-2, 2026-09-20): the SYSTEM JOIN source.
    # Same names the per-subgraph hallucination filter admits (candidates +
    # triple endpoints, CVT records excluded) but kept in ORIGINAL casing and
    # UNIONed across this fid's retrievals — the pool is the walk's full
    # enumeration, never the model's curated checkpoint subset. Keyed
    # canonically (see above) so every call form lands on the fid the join
    # reads (fact_bindings/fact_vars keys); a fid with no pool falls back
    # to its declared bindings there.
    # R4 DISPOSAL STATUS (2026-10-01, specs/r4_cleanup_2026-10-01.md): KEEP as
    # a SHADOW WRITE — every DEFAULT-face consumer left in R3-a/R3-b (V2 JOIN
    # reads views.join_projection; merge note / join_path_rescue / COMMIT
    # CANDIDATES / off-pool read the view layer). Remaining readers: the V1
    # JOIN face (SEQ_JOIN_V2=0: checkpoint-content JOIN + pre-answer
    # self-check), the SEQ_RENDER_V3=0 face (merge note + join_path_rescue
    # fallback), and the sidecar's pool_delta computation below (len delta
    # over this store — a serialized field). DELETE TOGETHER WITH the V1/OFF
    # faces when the user rules their retirement; pool_delta then needs a
    # ledger-side source.
    _fcp = getattr(ctx, "fact_candidate_pool", None)
    if _fcp is None:
        _fcp = ctx.fact_candidate_pool = {}
    _ev_pool_n0 = len(_fcp.get(_fid_c) or ())   # SHADOW LEDGER: pre-write size
    _fcp[_fid_c] = (_fcp.get(_fid_c) or set()) | \
        {str(c) for c in candidates if not is_cvt_like(c)} | \
        {str(x) for h, r, t in all_triples for x in (h, t) if not is_cvt_like(x)}
    # SHADOW LEDGER (phase-2, additive): complete the call's EvidenceRecord —
    # pool delta (canonical-fid candidate pool) + the render-side truncation
    # the completeness field reports (F03). V3: the §6.6 face — the >80
    # merged-edge-group cut events and the also-in-evidence spill count flow
    # from the row layer's stats (never read back from a text marker); the
    # OFF face keeps the legacy "+N lines truncated" marker parse.
    _evr_late = treq.get("_ev_record")
    if _evr_late is not None:
        _evr_late.pool_delta = len(_fcp[_fid_c]) - _ev_pool_n0
        if _v3_on:
            _ds = getattr(ctx, "_render_display_stats", None) or {}
            _gt = int(_ds.get("group_truncations", 0) or 0)
            _evr_late.completeness["display_group_truncations"] = _gt
            _evr_late.completeness["display_truncated_members"] = int(
                _ds.get("truncated_members", 0) or 0)
            _evr_late.completeness["display_also_in_evidence"] = int(
                _ds.get("also_in_evidence", 0) or 0)
            if _gt:
                _evr_late.completeness["status"] = "PARTIAL"
        else:
            _ev_drop = 0
            if tree_lines:
                _m = re.match(r"\s*\.\.\. \+(\d+) lines truncated$",
                              tree_lines[-1])
                if _m:
                    _ev_drop = int(_m.group(1))
            if _ev_drop:
                _evr_late.completeness["render_lines_dropped"] = _ev_drop
                _evr_late.completeness["status"] = "PARTIAL"
    # R1 ATOMIC COMMIT (R03): the call reached its SUCCESS terminal state —
    # ONLY NOW do its identities enter the delivered history. (Failure paths
    # finalize NO_EVIDENCE/FAILED above and fold nothing.)
    _rlog_late = getattr(ctx, "evidence_records", None)
    if _rlog_late is not None and treq.get("_ev_commit") is not None:
        _rlog_late.finalize_call("SUCCESS",
                                 pool_delta=(len(_fcp[_fid_c]) - _ev_pool_n0),
                                 render_lines_dropped=(
                                     (_evr_late.completeness.get(
                                         "render_lines_dropped", 0)
                                      if _evr_late is not None else 0)
                                     if not _v3_on else 0),
                                 display_stats=(
                                     getattr(ctx, "_render_display_stats",
                                             None) if _v3_on else None))
    # retrieval sequence per fid (variable-freezing clock): each NEW subgraph
    # retrieval for this fid advances it; a frozen declaration may only be
    # replaced after the clock advanced (fresh evidence for the same fact).
    # Evidence set is a UNION across retrievals — the legal binding domain is
    # "original sg1 ∪ new sg1 retrievals" (user ruling: 注册必须基于对应的子图).
    # R4 DISPOSAL STATUS (2026-10-01, specs/r4_cleanup_2026-10-01.md): KEEP.
    # COMMIT stopped reading this clock in R3-a (all_facts_concluded =
    # declaration clock), but the VARIABLE-FREEZE unlock still does
    # (dataflow C-③: "新证据解锁声明更新" — seq_react_loop compares the ✓ lock's
    # seq against it to admit a re-declaration). Its semantic successor, the
    # explicit REVISION event (proposal §5.6-④ 显式 revision 修复事件), is NOT
    # implemented; delete only after revision lands and the freeze check
    # switches to it.
    _fseq = getattr(ctx, "fact_evidence_seq", None)
    if _fseq is None:
        _fseq = {}; ctx.fact_evidence_seq = {}
    _fseq[_fid_c] = _fseq.get(_fid_c, 0) + 1
    _v36 = (os.environ.get("SEQ_RENDER_V36", "") == "1"
            or os.environ.get("SEQ_RENDER_V37", "") == "1"
            or os.environ.get("SEQ_RENDER_V38", "1") == "1")
    # ANSWER-TYPE ALIGNMENT of the display roster (user audit 2026-09-08):
    # CVT attribute VALUES (character role names in a film question) flooded
    # the flat candidates line. A value appearing ONLY as a CVT attribute —
    # never as a named direct-edge endpoint — is dropped unless its
    # attribute key's type class matches the plan's answer_type (a person
    # question KEEPS actor= values — 1171's gold is one; a film question
    # DROPS character= names — 25 specimen). Display-only: the legality
    # pool (walk_seen_entities / all_candidates) is untouched.
    _res = {
        "fact_id": fid,
        "entities": [e for e, _ in centers] if not _v36 else "",
        "triples": "\n".join(tree_lines) if tree_lines else "(empty)",

        "skipped_centers": skipped,
        "advice": ((_nudge + " ") if _nudge else "") +
                (("UNCHANGED EVIDENCE (repeat): these relations produced no NEW edges since "
                  "your previous subgraph — the same evidence is displayed again so it stays "
                  "in context. The records for this step are fixed: infer on them (answer, "
                  "or move to a still-unexecuted step). "
                  "Re-retrieving will return the same. ") if treq.get("_repeat_evidence") else "") +
                (("Multiple centers retrieved with one shared relation set — COMPARE them via "
                  "the triples (an edge '--to--> (incumbent)' marks the current holder). ") if multi else "") +
                (("SEQUENCE EXTENSION applied to several frontier members — the new layer's "
                  "edges are per-candidate: COMPARE them across the candidates (values, dates, "
                  "ids) and select the values the evidence supports (any non-empty count). "
                  "Final discrimination happens at answer analysis. "
                  "Mid-chain entities are HOPS, not answers. ") if treq.get("cont_compare") else "") +
                ("First check the executed path against your intended step (roles, "
                 "direction, bindings). If it matches, its records are FIXED for that "
                 "step: infer your answer from them instead of re-querying. Take a next "
                 "center from these triples only for a still-unexecuted step or a "
                 "specific repair. "
                 "Delivered records feed answer selection: keep the question's output "
                 "role and its stated conditions; with no exact binding, compare the "
                 "delivered actual partial support and select by the answer tiers "
                 "(EXACT / BEST-SUPPORTED / BEST-GUESS). Never rename a neighboring "
                 "role into the question's relation. A missing confirmation does not "
                 "require an empty answer."),
        "note": ("Evidence blocks group triples by entity: 'h --rel--> t1 | "
                  "t2' merges tails, 'h1 | h2 --rel--> t' merges heads — a wrapped "
                  "head list places the arrow+tail on its LAST line, so everything "
                  "before an arrow reads as heads. "
                  "m.xxx/g.xxx may be intermediate bindings, never final answers. "
                  "Their named ATTRIBUTES (actor, character, office "
                  "holder, jurisdiction) are shown inline in brackets and may be answers. "
                  "Discriminator attributes (dates, incumbent) appear as "
                  "their own edges — compare them to pick."),
    }
    # relation_expansion ECHO REMOVED from the model-facing result (user
    # ruling 2026-10-03: environment-facing bookkeeping, no model value —
    # and it over-promised bridges that are never walked when the center
    # has direct family support). treq["attr_expansion"] stays internal.
    if treq.get("anchor_seq"):
        # SEQUENCE-STATE ECHO (SEQ_REL_SEQ): the anchor's accumulated
        # relation layers + per-layer completion counts — the model's cue
        # that a later fact continuing this subgraph just re-calls with the
        # ANCHOR + the new relation (the system appends the layer).
        _res["anchor_sequence"] = treq["anchor_seq"]
    if treq.get("layer_action"):
        # LAYER-ACTION ECHO (user ruling 2026-09-15): the model's first
        # visibility into what the system DID with its submission —
        # extend (new layer), update:k (replaced layer k's relations), or
        # repeat (already in layers, pick different relations)
        _res["layer_action"] = treq["layer_action"]
    if bres.get("pattern_display"):
        # PATTERN-PATH WALK echo (2026-09-10): the call ran as ONE set-state
        # walk over the binding set; this is the ranked pattern list the
        # witness chains below re-instantiate
        _res["pattern_paths"] = bres["pattern_display"]
    _res_str = _json_result(_res)
    _mark_served(_res_str)
    return _res_str


async def retrieve_subgraph(args: Dict[str, Any], ctx, session) -> str:
    """Subgraph retrieval for ONE fact: one shared relation set applied to one or more
    center entities (the fact's candidate centers, packed into a single call). For each
    center the proven SAPS walk runs (multi-hop, K-path, CVT-penetrating); results are
    merged — dense tree + a candidate_attrs summary grouped by candidate under the shared
    relation pattern, so the model can compare across candidates (e.g. latest/largest).
    Accumulates seen entities into the subgraph.
    Sequential composition of the three phases (see _sg_prepare)."""
    from kgqa.agent.seq_adaptive import enabled as _adaptive_enabled
    if _adaptive_enabled(ctx):
        return await dispatch("retrieve_subgraph", args, ctx, session)
    treq = _sg_prepare(args, ctx)
    if treq["kind"] == "done":
        return treq["result"]
    bres = await _sg_execute(treq, ctx, session)
    return _sg_finalize(treq, bres, ctx)


# ───────────────────────── SEQ dispatch ─────────────────────────
# THREE-PHASE SPLIT (perf-4, 2026-08-23): dispatch_prepare (A: pure CPU —
# decompose/answer compute fully, the retrieval tools resolve entities and
# build their GTE/walk requests) → dispatch_execute (B: the IO — GTE calls and
# walk-lane submits, batched by the round-level collectors) → dispatch_finalize
# (C: pure CPU — evidence accumulation, rendering, result assembly). dispatch()
# keeps the sequential composition for the per-case runner and every existing
# caller; the round-level scheduler (seq_rollout / replay_dispatch) drives the
# phases directly so one round runs A for all cases, fires every GTE/walk
# request together, then runs C.

def _expand_seq_branch_refs(ctx, values):
    """Expand legacy branch references from delivered triples only."""
    if not any(value.startswith("#") for value in values):
        return values
    log = getattr(ctx, "evidence_records", None)
    if log is not None:
        edges = [edge for commit in log.delivered_commits()
                 for edge in commit.fact_keys]
    else:
        edges = list(getattr(ctx, "accumulated_triples", None) or ())
    out = []
    for value in values:
        match = re.fullmatch(r"#([^|:]+)(?:::|\|)(.+)", value)
        if not match:
            out.append(value)
            continue
        center, relation = match[1].strip(), match[2].strip()
        members = []
        for head, rel, tail in edges:
            if not (rel == relation or rel.endswith("." + relation)):
                continue
            other = tail if normalize(head) == normalize(center) else head if normalize(tail) == normalize(center) else None
            if other and not _ek_is_transit(other) and normalize(other) != normalize(center):
                members.append(other)
        out.extend(list(dict.fromkeys(members)) or [value])
    return out


def _do_seq_answer(args, ctx):
    """SEQ-only strict answer validation (user ruling 2026-10-06: SINGLE
    ENTRY — the answer must be an entity in the delivered subgraph. No
    candidate-pool distinction, no ledger-view vs shadow-pool fallback:
    the pool is the set of all entities in the accumulated delivered
    triples — period)."""
    values = args.get("entities")
    if not isinstance(values, list):
        return _json_result({"error": "ANSWER_FIELDS: provide answer_values explicitly (a list)."})
    if any(not isinstance(value, str) or not value.strip() for value in values):
        return _json_result({"error": "ANSWER_FIELDS: every answer value must be a nonempty string."})
    values = [value.strip() for value in values]
    values = _expand_seq_branch_refs(ctx, values)
    # V4 compressed-group auto-completion (user ruling 2026-10-05, continuing
    # the answer-face injection ruling): submitting exactly the displayed
    # batch of a compressed enumeration completes the omitted members.
    _v4g = getattr(ctx, "_v4_compressed_groups", None)
    if _v4g:
        from kgqa.agent.seq_render_v4 import complete_compressed_groups
        values = complete_compressed_groups(values, _v4g)
    if any(_ek_is_transit(v) for v in values):
        return _json_result({"error": "TRANSIT_VALUE: submit named entities or attribute values, not event/CVT ids."})
    # SINGLE ENTRY (user ruling 2026-10-06): all entities in the delivered
    # subgraph — nodes of the accumulated triples. No candidate/existence
    # distinction; the graph knows every entity it has delivered.
    pool = {normalize(str(v)) for edge in (getattr(ctx, "accumulated_triples", None) or ())
            for v in (edge[0], edge[2])}
    # NO-ABSTENTION VALVE (user ruling 2026-10-03): an empty answer is a lazy
    # escape whenever ANY evidence was delivered — reject it and make the
    # model select from the pool. [] stays valid only when nothing was ever
    # delivered (degenerate case: every retrieval failed / no budget).
    if not values and pool:
        return _json_result({"error": (
            "EMPTY_ANSWER: delivered evidence exists — an empty answer is not "
            "accepted. Select the values that best satisfy the question's "
            "answer role from the delivered records (SUPPORTED first, then the "
            "strongest preserved binding; missing confirmation is not "
            "abstention grounds) and resubmit:\n"
            "tool: answer\nanswer_values: <best supported delivered values>")})
    bad = [v for v in values if normalize(v) not in pool]
    if bad:
        # SUGGESTION (user ruling 2026-10-06): if the rejected value is
        # close to a delivered entity, suggest it — the model knows what
        # it meant and can resubmit immediately.
        sug = []
        for b in bad:
            bn = normalize(b)
            close = sorted((p for p in pool if p != bn
                            and (bn in p or p in bn)))[:3]
            sug.extend(close)
        sug_text = (f" Did you mean: {' | '.join(sorted(set(sug))[:3])}?"
                    if sug else "")
        return _json_result({"error": (
            f"UNDELIVERED_ANSWER: {bad}. The answer must be an entity in the "
            f"delivered subgraph.{sug_text} Repair answer_values using "
            "delivered evidence; this answer was not accepted.")})
    values = list(dict.fromkeys(values))
    ctx.llm_answer_preds = values
    ctx.llm_answer_str = " | ".join(values)
    return _json_result({"answer_values": values}) if values else "answer_values: []"


def dispatch_prepare(tool_name: str, args: Dict[str, Any], ctx) -> dict:
    """A段: returns a tool request. kind="done" carries the fully-computed
    result (no IO); kind="rr"/"sg" carry the GTE/walk request for
    dispatch_execute; kind="unknown" defers the error render to finalize."""
    from kgqa.agent.seq_adaptive import enabled as _adaptive_enabled, prepare as _adaptive_prepare
    if _adaptive_enabled(ctx):
        return _adaptive_prepare(tool_name, args, ctx)
    if tool_name in ("plan", "decompose"):
        return {"kind": "done", "result": _do_seq_decompose(args, ctx)}
    if tool_name == "retrieve_relations":
        return _rr_prepare(args, ctx)
    if tool_name == "retrieve_subgraph":
        return _sg_prepare(args, ctx)
    if tool_name == "answer":
        return {"kind": "done", "result": _do_seq_answer(args, ctx)}
    return {"kind": "unknown", "tool": tool_name}


async def dispatch_execute(treq: dict, ctx, session):
    """B段: the IO half of the prepared request (no-op for done/unknown)."""
    k = treq.get("kind")
    if k == "adaptive":
        from kgqa.agent.seq_adaptive import execute
        return await execute(treq, ctx, session)
    if k in ("rr", "corr"):
        return await _rr_execute(treq, ctx, session)
    if k == "sg":
        return await _sg_execute(treq, ctx, session)
    return None


def dispatch_finalize(treq: dict, bres, ctx) -> str:
    """C段: render the result string (pure CPU)."""
    k = treq.get("kind")
    if k == "adaptive":
        from kgqa.agent.seq_adaptive import finalize
        return finalize(treq, bres, ctx)
    if k == "done":
        return treq["result"]
    if k in ("rr", "corr"):
        return _rr_finalize(treq, bres, ctx)
    if k == "sg":
        return _sg_finalize(treq, bres, ctx)
    return _json_result({"error": f"unknown tool: {treq.get('tool')}"})


async def dispatch(tool_name: str, args: Dict[str, Any], ctx, session) -> str:
    # accept both new SEQ names and old names (backward-compat for stale trajectories)
    treq = dispatch_prepare(tool_name, args, ctx)
    if treq["kind"] in ("done", "unknown"):
        return dispatch_finalize(treq, None, ctx)
    bres = await dispatch_execute(treq, ctx, session)
    return dispatch_finalize(treq, bres, ctx)

