"""R2 VIEW LAYER — ONE LEGAL IMPLEMENTATION PER QUESTION (proposal §6.7-5,
§6.8, §7-R2; 2026-10-01). R1 built the three-layer ledger; R2 mounts the
eight questions every mechanism asks on top of it. Views are PURE READS
over the ledger (they never write it), each view composes with the others
only through their PUBLIC functions (never each other's internals), and
every returned collection is deterministically ordered (sorted or
first-seen) so outputs are stable across processes (PYTHONHASHSEED face).

The eight questions (a mechanism may project a view's answer, but must
never re-implement, side-collect, or parse text behind it):

    existence(scope)           — WHAT evidence exists? (fact | tree | episode)
                                 Absorbs the D3-0 observed_episode view: the
                                 episode scope IS that domain, value-for-value.
    binding_legality(var, val) — may THIS variable bind THIS value? (role and
                                 type: intermediate vars may bind CVT/ID transit
                                 nodes — 2026-09-24 ruling — answer vars may not)
    answer_legality(value)     — may THIS value be the FINAL answer? (support
                                 ∧ answer-variable projection ∧ final type:
                                 terminal CVTs and anonymous ID nodes never)
    join_projection(ids, var)  — the legal projection of both sides onto one
                                 OUTPUT variable: intersection + union (the
                                 §6.8-3 JOIN face; V3-01 true terminal-hop
                                 landing projection — a shared anchor is
                                 never a shared answer; the pool never moves)
    frontier(scope)            — the structural frontier, FULL node ids incl
                                 CVTs (the rr-pool source); the named ≤50
                                 preview face rides along
    attribution(fact)          — many-to-many ownership of one fact: which
                                 trees' legal evidence contains it + the first
                                 delivery per scope (§6.7-2's three questions,
                                 minus the counterfactual which is the
                                 evaluator's to ask)
    carrier(gold)              — §5.6 FINAL gold-carrier test on a call's NEW
                                 delivered facts of submitted family with gold
                                 as an endpoint (CVT-attr special case rides
                                 the same test: the attr fact's VALUE endpoint
                                 with origin='submitted')
    display_projection(budget) — the render projection INPUT for one call,
                                 merged edge groups with the ONE legal
                                 truncation point (§6.6: >budget tails/heads
                                 per group); truncation events are counted and
                                 returned, never silent

Consumers switch in R3; nothing live reads this module yet. The contracts
live in tests/test_r2_view_contracts.py (per-view Q/A contracts + the
cross-view consistency assertions + the read-only invariant).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from kgqa.agent.entity_kinds import entity_kind

# scopes of the existence question
SCOPE_FACT = "fact"          # key = fact_key (h, rel, t) — full names
SCOPE_TREE = "tree"          # key = tree_id (canon fid)
SCOPE_EPISODE = "episode"    # key ignored

# kinds that may terminate an answer (NAMED entities and typed LITERALs —
# value-typed final hops; entity_kinds.py is the single authority)
_ANSWER_KINDS = ("NAMED", "LITERAL")
# transit surfaces: legal mid-chain bindings, never final answers
_TRANSIT_KINDS = ("CVT", "ID_NODE")

# §6.6 display truncation: the ONE legal cut point — a merged edge group
# with more than this many tails (or heads) keeps this many
DISPLAY_GROUP_BUDGET = 80

# R3-b consumer-switch flag (2026-10-01, specs/r3b_render_pools): the
# render/display + pool-display face moves onto this view layer. ON
# (default): display truncation = the >DISPLAY_GROUP_BUDGET merged-edge-
# group cut ONLY (the 200-line global budget is abolished, §6.6), render
# CVT attrs PROJECT the evidence-build materialized store (§6.8-2), and
# the fact_candidate_pool consumers (merge note / join rescue / COMMIT
# CANDIDATES / off-pool + its candidate rows) read answer_projection /
# existence here instead — the pool keeps its writes as a shadow until R4
# retires it. OFF (=0): byte-identical legacy behavior — rollback + A/B.
RENDER_V3_DEFAULT = "1"


def render_v3() -> bool:
    import os
    return os.environ.get("SEQ_RENDER_V3", RENDER_V3_DEFAULT).strip().lower() \
        not in ("0", "off", "false", "no")


@dataclass
class Verdict:
    """Machine-readable legality answer (receipt building is R3's job)."""
    ok: bool
    code: str                     # OK | UNDELIVERED | TRANSIT_VALUE | ...
    detail: str = ""

    def __bool__(self) -> bool:   # convenient truthiness at call sites
        return self.ok


# ─────────────────────────────────────────────────────────────────────────────
# shared read plumbing (public ledger surface only)
# ─────────────────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    # normalize() is the SINGLE norm implementation (phase-2 trap: bare
    # .lower() misses 'The Twilight Saga: New Moon')
    from kgqa.core.utils import normalize
    return normalize(name)


def _delivered(log) -> List:
    """Delivered commits, in ledger order — the legal-evidence gate (the
    ledger's own _delivered test: SUCCESS state + legacy outcome mirror)."""
    return log.delivered_commits()


def _call(log, call_id: Optional[str]):
    """Resolve one call by id (None = the LATEST delivered commit)."""
    cs = _delivered(log)
    if not cs:
        return None
    if call_id is None:
        return cs[-1]
    for c in cs:
        if c.call_id == call_id:
            return c
    return None


def _fact_endpoints(log, fact_key: Tuple[str, str, str]) -> Set[str]:
    """Normalized endpoint names of one fact — the fact-scope existence set.
    Attr facts answer with their VALUE endpoint naturally (t_id IS the
    value); the attr display_key never creates a second identity (R05)."""
    f = log.facts.get(tuple(fact_key)) if getattr(log, "facts", None) else None
    if f is not None:
        # only legal evidence if some delivered observation used it
        used = any(ob.fact_key == f.fact_key for c in _delivered(log)
                   for ob in (c.observations or ()))
        return {_norm(f.h_id), _norm(f.t_id)} if used else set()
    # legacy path (loaded sidecars / legacy commit()): identity == the
    # legacy triple itself
    for c in _delivered(log):
        if c.legacy is None:
            continue
        for tr in c.legacy.triples:
            if (tr.h, tr.rel, tr.t) == tuple(fact_key):
                return {_norm(tr.h), _norm(tr.t)}
    return set()


# ─────────────────────────────────────────────────────────────────────────────
# 1. existence — what evidence exists (D3-0 observed_episode absorbed)
# ─────────────────────────────────────────────────────────────────────────────

def existence(log, scope: str = SCOPE_EPISODE, key=None) -> Set[str]:
    """The legal-evidence EXISTENCE set at the asked scope: NORMALIZED
    endpoint names of every fact delivered there — walk-edge endpoints AND
    CVT attr VALUE endpoints (role='attr'; the 2784 'Juice' specimen's gold
    lives only there). By-name only, NO alias folding (ruling 2026-10-01:
    an old name with only the current name delivered does NOT count);
    normalize() is the single norm implementation. no_evidence / failed
    calls delivered nothing and contribute nothing.

    scope='episode' returns EXACTLY the D3-0 observed_episode domain — the
    hallucination filter's question ("did ANY delivered triple contain this
    name?") is this scope, and EvidenceRecordLog.observed_episode() now
    delegates here (call-site signatures unchanged, output value-for-value
    equal). scope='tree' answers the placement question for one canon fid;
    scope='fact' answers whether ONE fact is legal evidence yet (empty set
    = undelivered)."""
    if scope == SCOPE_FACT:
        return _fact_endpoints(log, key)
    out: Set[str] = set()
    for c in _delivered(log):
        if scope == SCOPE_TREE and c.canon_fid != key:
            continue
        if c.observations:
            for ob in c.observations:
                f = log.facts.get(ob.fact_key)
                if f is not None:
                    out.add(_norm(f.h_id))
                    out.add(_norm(f.t_id))
        elif c.legacy is not None:
            for tr in c.legacy.triples:
                out.add(_norm(tr.h))
                out.add(_norm(tr.t))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# shared answer projection (the common core of answer_legality and
# join_projection — ONE implementation so the two answers cannot drift).
# V3-01 (Codex final review 2026-10-01): this is a TRUE output-variable
# projection, not a kind-filtered endpoint pool. For each delivered fact
# the projection keeps the LANDING endpoint of the pattern's TERMINAL hop
# (the far end of the chain's last hop, resolved per usage from the
# Observation's chain position + traversal direction) plus the attr lane's
# VALUE endpoints (the legal value channel, R01/R05). Anchors (the call's
# walk centers) and bridge/mid-chain nodes are excluded BY CONSTRUCTION —
# a two-tree JOIN can no longer intersect on a shared anchor: two trees
# anchored on the same node with different answer tails get an EMPTY
# intersection, as they must. The plan's fact→var mapping enters through
# the caller's tree_ids (the join group / the var's bound facts); `var`
# rides as the consumer's key.
# ─────────────────────────────────────────────────────────────────────────────

def _usage_landings(ob, f) -> List[str]:
    """The landing endpoint names of one observation's TERMINAL-hop
    usages: 'forward' usage lands on the raw tail, 'reversed' on the raw
    head (V3-03 raw orientation); 'unknown' (legacy sidecars) keeps both
    answer-typed ends. Witness usages are last-hop discoverers — terminal
    by construction. cvt_passthrough usages never land (transparency)."""
    from kgqa.agent.evidence import FactUsage
    out: List[str] = []
    us = list(ob.usages or ())
    if not us:
        # legacy/loader observations carry the primary usage inline
        if ob.chain_len and ob.chain_pos == ob.chain_len - 1:
            us = [FactUsage("hop", chain_pos=ob.chain_pos,
                            chain_len=ob.chain_len,
                            traversal_dir=ob.traversal_dir)]
        elif ob.role == "pattern_hop" and ob.chain_len is None \
                and ob.chain_pos is None:
            us = [FactUsage("witness")]
    for u in us:
        if u.lane == "witness":
            out.extend(_dir_ends(f, u.traversal_dir or ob.traversal_dir))
        elif u.lane == "full":
            continue                      # CVT passthrough never lands
        elif u.chain_len and u.chain_pos == u.chain_len - 1:
            out.extend(_dir_ends(f, u.traversal_dir or ob.traversal_dir))
        else:
            continue                      # bridge / mid-chain hop
    return out


def _dir_ends(f, traversal_dir) -> List[str]:
    if traversal_dir == "forward":
        return [f.t_id]
    if traversal_dir == "reversed":
        return [f.h_id]
    return [f.h_id, f.t_id]               # legacy tolerance: both ends


def answer_projection(log, tree_ids: Optional[Sequence[str]] = None,
                      var: Optional[str] = None) -> Set[str]:
    """The ANSWER-VARIABLE projection of the legal evidence (§6.3
    answer_pool, kinds-resolved, V3-01 true projection): per delivered
    fact, the TERMINAL-hop LANDING endpoint of its pattern usage (anchors
    and bridge nodes excluded by construction; the raw orientation + the
    usage's traversal direction resolve which end lands) and the attr
    lane's VALUE endpoints (role='attr' — value-form gold projects like
    any endpoint, R01/R05), restricted to kinds that may terminate an
    answer — NAMED entities and typed LITERALs; CVT carriers and anonymous
    ID nodes never. Scoped to the given trees when tree_ids is given, else
    the whole episode. The plan's fact→var mapping is the caller's
    tree_ids grouping (the join group keyed by the plan edge's tail
    variable); `var` rides as the consumer's key. answer_legality and
    join_projection BOTH read this — the same-source contract."""
    want = None if tree_ids is None else {str(t) for t in tree_ids}
    out: Set[str] = set()
    for c in _delivered(log):
        if want is not None and c.canon_fid not in want:
            continue
        # V3-01 anchor exclusion: the call's walk centers are givens, never
        # this call's output candidates ([center] fallback for old commits).
        # Per-commit subtraction — one tree's landing on ANOTHER tree's
        # anchor stays legal (order-free).
        anchors = {_norm(x) for x in (c.centers or ()) if x}
        if not anchors and c.center:
            anchors = {_norm(c.center)}
        contribution: Set[str] = set()
        if c.observations:
            for ob in c.observations:
                f = log.facts.get(ob.fact_key)
                if f is None:
                    continue
                if ob.lane == "attr" or ob.role == "attr":
                    if entity_kind(f.t_id) in _ANSWER_KINDS:
                        contribution.add(_norm(f.t_id))  # legal value channel
                    continue
                for name in _usage_landings(ob, f):
                    if entity_kind(name) in _ANSWER_KINDS:
                        contribution.add(_norm(name))
        elif c.legacy is not None:
            # legacy projection (phase-2 sidecars): attr rows project their
            # value; walk rows project the terminal hop when the position
            # is known, else the answer-typed ends (both).
            for tr in c.legacy.triples:
                if tr.role == "attr":
                    if entity_kind(tr.t) in _ANSWER_KINDS:
                        contribution.add(_norm(tr.t))
                    continue
                if tr.chain_len and tr.chain_pos == tr.chain_len - 1:
                    ends = [tr.t]
                elif tr.chain_len is None and tr.role == "pattern_hop":
                    ends = [tr.h, tr.t]   # witness: terminal by construction
                else:
                    continue
                for name in ends:
                    if entity_kind(name) in _ANSWER_KINDS:
                        contribution.add(_norm(name))
        out |= (contribution - anchors)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. binding_legality / 3. answer_legality (可绑定 ≠ 可作答 — two questions)
# ─────────────────────────────────────────────────────────────────────────────

def binding_legality(log, var, value,
                     answer_vars: Optional[Set[str]] = None) -> Verdict:
    """May VARIABLE `var` bind VALUE `value`? (D3 semantics + the
    hallucination principle, one legal answer.)

    Conjuncts:
      1. support — the value exists in the episode's legal evidence
         (existence('episode'), by name, no alias folding);
      2. role/type — an ANSWER variable never binds a transit node (CVT
         event / anonymous ID): events are records, not entities. An
         INTERMEDIATE variable MAY bind a delivered transit node (ruling
         2026-09-24: legitimate mid-chain reference, e.g. a bound CVT can
         center later retrievals).

    `answer_vars` is the caller's set of output-variable names (the plan's
    answer side — DeclarationLedger's face in R3); vars outside it are
    intermediate by the D3 ruling (CVT≡普通实体 at the pattern layer)."""
    v = str(value)
    kind = entity_kind(v)
    if _norm(v) not in existence(log, SCOPE_EPISODE):
        return Verdict(False, "UNDELIVERED",
                       "no delivered triple of this episode contains the "
                       "name (checked by name; alias/old names do not count)")
    if answer_vars and str(var) in {str(x) for x in answer_vars} \
            and kind in _TRANSIT_KINDS:
        return Verdict(False, "TRANSIT_AS_ANSWER_VAR",
                       f"{kind} nodes are records, never entities: an "
                       "answer variable may only bind named/typed values")
    return Verdict(True, "OK", kind)


def answer_legality(log, value) -> Verdict:
    """May VALUE be the FINAL answer? Three conjuncts, one legal answer:
      1. support — exists in the episode's legal evidence (by name);
      2. answer projection — sits in the answer-variable projection of the
         evidence (answer_projection — same source as join_projection;
         V3-01: the terminal-hop landing set, anchors and bridge nodes
         excluded);
      3. final type — a terminal CVT is never an answer (§7.5 mid-strip
         semantics) and neither is an anonymous ID node; typed LITERALs of
         value-typed final hops ARE (1812: [number: 610])."""
    v = str(value)
    kind = entity_kind(v)
    if kind in _TRANSIT_KINDS:
        return Verdict(False, "TRANSIT_VALUE",
                       f"{kind} nodes name no thing — the entities INSIDE "
                       "the brackets are the answer surface")
    if _norm(v) not in existence(log, SCOPE_EPISODE):
        return Verdict(False, "UNDELIVERED",
                       "no delivered triple of this episode contains the "
                       "name (checked by name; alias/old names do not count)")
    if _norm(v) not in answer_projection(log):
        return Verdict(False, "NOT_IN_PROJECTION",
                       "delivered, but not on any answer-typed endpoint "
                       "(projection excludes transit carriers)")
    return Verdict(True, "OK", kind)


# ─────────────────────────────────────────────────────────────────────────────
# 4. join_projection — both sides' legal projection onto one output variable
# ─────────────────────────────────────────────────────────────────────────────

def join_projection(log, tree_ids: Sequence[str],
                    var: Optional[str] = None) -> dict:
    """The legal projections of the given trees onto ONE shared output
    variable, plus their intersection and union (§5.6-⑥/§6.8-3: JOIN uses
    only the OUTPUT-variable projection, the pools themselves never move;
    an EMPTY intersection is meaningful — with both sides declared and no
    merge, the environment owes a connecting path). V3-01: each side's
    projection is the TRUE terminal-hop landing set (answer_projection —
    anchors and bridge nodes excluded), so two trees sharing an anchor
    with different answer tails get an EMPTY intersection; the plan's
    fact→var mapping is the caller's tree grouping (the join group keyed
    by the plan edge's tail variable), `var` rides as its key."""
    tids = [str(t) for t in (tree_ids or ())]
    per_tree: Dict[str, List[str]] = {}
    for tid in tids:
        per_tree[tid] = sorted(answer_projection(log, [tid], var))
    sets = [set(v) for v in per_tree.values()]
    inter = sorted(set.intersection(*sets)) if sets else []
    union = sorted(set.union(*sets)) if sets else []
    return {"var": var, "trees": tids,
            "per_tree": per_tree,
            "intersection": inter, "union": union,
            "nonempty_per_side": bool(sets) and all(sets)}


# ─────────────────────────────────────────────────────────────────────────────
# 5. frontier — the FULL structural frontier (rr pool source)
# ─────────────────────────────────────────────────────────────────────────────

def frontier(log, scope: str = SCOPE_EPISODE,
             tree_id: Optional[str] = None) -> dict:
    """The post-delivery structural frontier: FULL node ids, CVTs INCLUDED
    (R07 — the rr pool continues from the whole supernode set; transit
    transparency handles their reach). The named ≤50 PREVIEW face rides
    along with its truncation counts — display faces never gate structural
    ones. Episode scope merges the LATEST delivered commit of EVERY tree
    (each tree's continuation surface); tree scope takes the given tree's."""
    per_tree: Dict[str, dict] = {}
    for c in _delivered(log):
        if scope == SCOPE_TREE and c.canon_fid != tree_id:
            continue
        # later commits of the same tree supersede earlier ones (the
        # continuation surface is the tree's newest frontier)
        per_tree[c.canon_fid] = {
            "call_id": c.call_id,
            "node_ids": list(c.frontier_node_ids or ()),
            "named_preview": list(c.frontier_preview or ()),
            "total": int(c.frontier_total or 0),
            "dropped": int(c.frontier_dropped or 0)}
    if scope == SCOPE_TREE and tree_id is not None:
        per_tree = {k: v for k, v in per_tree.items() if k == str(tree_id)}
    node_ids: List[int] = []
    seen: Set[int] = set()
    preview: List[str] = []
    pseen: Set[str] = set()
    total = dropped = 0
    for face in per_tree.values():
        for i in face["node_ids"]:
            if i not in seen:
                seen.add(i)
                node_ids.append(int(i))
        for n in face["named_preview"]:
            if n not in pseen:
                pseen.add(n)
                preview.append(n)
        total += face["total"]
        dropped += face["dropped"]
    return {"scope": scope,
            "node_ids": node_ids,           # FULL structural set, CVT incl.
            "named_preview": preview,       # the ≤50 named face (R07)
            "frontier_total": total, "frontier_dropped": dropped,
            "per_tree": {k: per_tree[k] for k in sorted(per_tree)}}


# ─────────────────────────────────────────────────────────────────────────────
# 6. attribution — one fact's many-to-many ownership (§6.7-2)
# ─────────────────────────────────────────────────────────────────────────────

def attribution(log, fact_key: Tuple[str, str, str]) -> dict:
    """The three ownership questions of §6.7-2 that a READ view can answer:
    the SCOPE RANGE (which trees' legal evidence contains the fact — facts
    ↔ trees are many-to-many), the FIRST DELIVERY per scope (which call
    first delivered it, episode-wide and per tree), and the usage census
    (lanes/roles seen). The third question — the ACTION CONTRIBUTION
    (what disappears if a call is removed) — is counterfactual and stays
    the evaluator's (R3 sidecar face)."""
    fk = tuple(fact_key)
    trees: Set[str] = set()
    first_episode: Optional[str] = None
    first_by_tree: Dict[str, str] = {}
    lanes: Set[str] = set()
    roles: Set[str] = set()
    n_uses = 0
    for c in _delivered(log):
        hit = False
        if c.observations:
            for ob in c.observations:
                if ob.fact_key == fk:
                    hit = True
                    n_uses += 1
                    lanes.add(ob.lane)
                    roles.add(ob.role)
        elif c.legacy is not None:
            for tr in c.legacy.triples:
                if (tr.h, tr.rel, tr.t) == fk:
                    hit = True
                    n_uses += 1
                    roles.add(tr.role)
        if hit:
            trees.add(c.canon_fid)
            if first_episode is None:
                first_episode = c.call_id
            first_by_tree.setdefault(c.canon_fid, c.call_id)
    return {"fact_key": list(fk),
            "delivered": bool(trees),
            "trees": sorted(trees),            # the ownership SCOPE RANGE
            "first_delivery": {"episode": first_episode,
                               "by_tree": {k: first_by_tree[k]
                                           for k in sorted(first_by_tree)}},
            "uses": n_uses, "lanes": sorted(lanes), "roles": sorted(roles)}


# ─────────────────────────────────────────────────────────────────────────────
# 7. carrier — the §5.6 FINAL gold-carrier test
# ─────────────────────────────────────────────────────────────────────────────

def carrier(log, gold, call_id: Optional[str] = None) -> dict:
    """Is GOLD carried by a call? §5.6 FINAL: a NEW-delivered fact of the
    call whose relation is in the ENVIRONMENT-RESOLVED submitted family
    (origin='submitted' — folded from rel_idxs incl. inverses, never the
    raw submitted string) with gold as an ENDPOINT (by name). The CVT-attr
    special case rides the SAME test: an attr fact whose carrier CVT was
    hit by a submitted-origin edge carries origin='submitted', and its
    VALUE endpoint is then a legal gold side (1392 education specimen).
    Redisplayed gold never re-counts (only the block's NEW delivery;
    372 blk1) — `new_to_episode` rides the CANONICAL fact-key identity
    (V3-02; ruling ⑥: exact identity forever, the family fold is data
    for retrieval/relevance only and never moves a flag)."""
    ng = _norm(str(gold))
    calls = [_call(log, call_id)] if call_id is not None else _delivered(log)
    hits: List[dict] = []
    for c in calls:
        if c is None:
            continue
        if c.observations:
            for ob in c.observations:
                if ob.origin != "submitted" or not ob.new_to_episode:
                    continue
                f = log.facts.get(ob.fact_key)
                if f is None or ng not in (_norm(f.h_id), _norm(f.t_id)):
                    continue
                hits.append({"call_id": c.call_id, "tree_id": c.canon_fid,
                             "fact_key": list(f.fact_key),
                             "role": ob.role, "lane": ob.lane,
                             "origin": ob.origin})
        elif c.legacy is not None:
            for tr in c.legacy.triples:
                if tr.origin != "submitted" or not tr.new_to_episode:
                    continue
                if ng not in (_norm(tr.h), _norm(tr.t)):
                    continue
                hits.append({"call_id": c.call_id, "tree_id": c.canon_fid,
                             "fact_key": [tr.h, tr.rel, tr.t],
                             "role": tr.role, "lane": "legacy",
                             "origin": tr.origin})
    return {"gold": str(gold), "carried": bool(hits), "hits": hits}


# ─────────────────────────────────────────────────────────────────────────────
# 8. display_projection — the render projection input (sole truncation point)
# ─────────────────────────────────────────────────────────────────────────────

def display_rows(log, call_id: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """The call's DISPLAY rows: (h, rel, t) in display form — walk-lane
    facts under their full relation name, attr-lane facts under their
    display_key bracket form (the same fact's display face, R05; a fact
    seen through BOTH lanes keeps BOTH display rows, as the legacy
    lossless projection always did)."""
    c = _call(log, call_id)
    if c is None:
        return []
    rows: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()
    if c.observations:
        for ob in c.observations:
            f = log.facts.get(ob.fact_key)
            if f is None:
                continue
            rel = f.display_key if (ob.lane == "attr" and f.display_key) \
                else f.rel_id
            row = (f.h_id, rel, f.t_id)
            if row not in seen:
                seen.add(row)
                rows.append(row)
    elif c.legacy is not None:
        for tr in c.legacy.triples:
            rows.append((tr.h, tr.rel, tr.t))
    return rows


def display_projection(log, call_id: Optional[str] = None,
                       budget: int = DISPLAY_GROUP_BUDGET) -> dict:
    """The RENDER projection INPUT for one call: the display rows merged
    into edge groups — shape A `h --rel--> t1 | t2 | …` (same h+rel, many
    tails) and shape B `h1 | h2 --rel--> t` (many heads, same rel+t) — with
    the ONE legal truncation point (§6.6 ruling): a group with more than
    `budget` members keeps `budget` and the cut is RETURNED as a countable
    truncation event, never silent. Judgment views never read this; the
    display budget is recorded separately from the walk budget (R1
    completeness['display'])."""
    rows = display_rows(log, call_id)
    tails: Dict[Tuple[str, str], List[str]] = {}
    heads: Dict[Tuple[str, str], List[str]] = {}
    for (h, r, t) in rows:
        tl = tails.setdefault((h, r), [])
        if t not in tl:
            tl.append(t)
        hd = heads.setdefault((r, t), [])
        if h not in hd:
            hd.append(h)
    groups: List[dict] = []
    truncations: List[dict] = []
    for (h, r), ts in sorted(tails.items()):
        cut = len(ts) > budget
        kept = sorted(ts)[:budget] if cut else sorted(ts)
        groups.append({"shape": "tails", "h": h, "rel": r, "members": kept,
                       "total": len(ts)})
        if cut:
            truncations.append({"shape": "tails", "group": [h, r],
                                "side": "tails", "total": len(ts),
                                "kept": budget, "dropped": len(ts) - budget})
    for (r, t), hs in sorted(heads.items()):
        cut = len(hs) > budget
        kept = sorted(hs)[:budget] if cut else sorted(hs)
        groups.append({"shape": "heads", "rel": r, "t": t, "members": kept,
                       "total": len(hs)})
        if cut:
            truncations.append({"shape": "heads", "group": [r, t],
                                "side": "heads", "total": len(hs),
                                "kept": budget, "dropped": len(hs) - budget})
    return {"call_id": _call(log, call_id).call_id if _call(log, call_id)
            else None,
            "budget": int(budget),
            "n_rows": len(rows),
            "groups": groups,
            "truncations": truncations,
            "n_truncations": len(truncations)}
