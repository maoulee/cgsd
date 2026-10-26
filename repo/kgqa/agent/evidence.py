"""EVIDENCE SET — the structured single source of truth for evidence
(redesign charter 2026-09-24, ZEROETH RULE):

    Apart from TWO legal exceptions — (a) logprob scoring must construct a
    text prompt, (b) the MODEL's own declarations (checkpoint bindings /
    answer entities) must be parsed from input text — EVERY consumer of
    evidence (render, dedup, idempotency, answer-pool harvest, join
    checks, evaluation attribution) reads THIS structured object. The
    rendered text is a PROJECTION of the EvidenceSet for the model's eyes;
    the environment NEVER reads its own rendered text back.

Lifecycle per retrieve_subgraph call:
    walker chains -> EvidenceSet.build(chains, kinds, ctx) -> presenter
    projection; the call's EvidenceSet is appended to ctx.evidence_log and
    unioned into ctx.delivered (fact-key set, via entity_kinds-adjacent
    fact keys from seq_tools._edge_fact_key).

Consumers (migrated in stages behind SEQ_EVIDENCE_V2):
    - answer legal pool   : endpoints + CVT-record attrs from evidence_log
    - idempotency (_sg_served): (root, pattern-set) keys — no text
    - join/merge checks   : pool endpoints from evidence_log
    - RSCC attribution    : per-call edge ownership from evidence_log
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from kgqa.agent.entity_kinds import entity_kind, is_cvt, is_transit

Edge = Tuple[str, str, str]            # (head_name, rel_full_name, tail_name)
Record = Tuple[str, List[Tuple[str, str]]]   # (cvt_name, [(attr_key, value)])


@dataclass
class EvidenceSet:
    """One call's structured evidence. rows = merged display rows (visual
    only); edges/records are the truth; delivered_keys are fact keys."""
    root: str = ""
    patterns: List[str] = field(default_factory=list)     # selected pattern labels
    edges: List[Edge] = field(default_factory=list)        # ALL walked edges incl. passthrough
    records: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)  # cvt -> attr -> values
    endpoints: Set[str] = field(default_factory=set)       # NAMED endpoints (answer-legal surface)
    delivered_keys: Set[Tuple] = field(default_factory=set)  # fact keys contributed THIS call

    # ---------- construction ----------
    @classmethod
    def build(cls, chains, ctx, fact_key_fn) -> "EvidenceSet":
        """chains: walker WalkChain dicts with full edge lists (names).
        fact_key_fn: seq_tools._edge_fact_key (h, r, t, inv_map) -> key."""
        ev = cls()
        seen_edges: Set[Edge] = set()
        for ch in chains or []:
            for (h, r, t, role) in ch.get("named_edges", ()):
                e = (h, r, t)
                if e in seen_edges:
                    continue
                seen_edges.add(e)
                ev.edges.append(e)
                for name in (h, t):
                    if entity_kind(name) == "NAMED":
                        ev.endpoints.add(name)
                    elif is_cvt(name):
                        ev.records.setdefault(name, {})
            for (cvt, key, value) in ch.get("record_attrs", ()):
                ev.records.setdefault(cvt, {}).setdefault(key, []).append(value)
        ev.delivered_keys = {fact_key_fn(h, r, t) for (h, r, t) in ev.edges}
        return ev

    # ---------- queries (the ONLY interfaces consumers may use) ----------
    def answer_pool(self) -> List[str]:
        """Named endpoints + CVT attribute values (the answer-legal surface)."""
        pool = list(self.endpoints)
        for attrs in self.records.values():
            for values in attrs.values():
                pool.extend(values)
        return pool

    def owns_edge(self, key) -> bool:
        return key in self.delivered_keys

    def to_projection_meta(self) -> dict:
        """Structured metadata for the presenter (labels/records); the
        presenter renders TEXT from this + edges, never re-parses text."""
        return {"patterns": list(self.patterns),
                "records": {k: dict(v) for k, v in self.records.items()}}


class EvidenceLog:
    """Case-scope ledger: append-only log of EvidenceSets + the unioned
    delivered-key set. THIS replaces ctx.accumulated_triples /
    fact_evidence / fact_candidate_pool overlaps as the single 'already
    delivered' authority (stage-gated migration)."""

    def __init__(self):
        self.calls: List[EvidenceSet] = []
        self.delivered: Set[Tuple] = set()

    def append(self, ev: EvidenceSet):
        self.calls.append(ev)
        self.delivered |= ev.delivered_keys

    def answer_pool(self, fid: Optional[str] = None) -> List[str]:
        if fid is None:
            pool: List[str] = []
            for ev in self.calls:
                pool.extend(ev.answer_pool())
            return pool
        # fid-scoped pools remain keyed by the caller during migration
        pool: List[str] = []
        for ev in self.calls:
            pool.extend(ev.answer_pool())
        return pool

    def is_delivered(self, key) -> bool:
        return key in self.delivered


# ─────────────────────────────────────────────────────────────────────────────
# PHASE-2 SHADOW LEDGER (two-ledger proposal §1.1 + user rulings 2026-09-30:
# gold-carrier final version / F10 origin). EvidenceRecord is a PURELY
# ADDITIVE per-call record: nothing in the live decision/render/judgment
# chain consumes it yet — it is appended alongside the EvidenceSet and
# shipped in the rollout sidecar (SEQ_EVIDENCE_SIDECAR) for offline shadow
# comparison against the text-era evaluator. Do NOT wire any live mechanism
# to it without the ±2.4pp dual-batch protocol.
# ─────────────────────────────────────────────────────────────────────────────

TripleId = Tuple[str, str, str]       # (h full name, rel full name, t full name)


@dataclass
class EvidenceTriple:
    """One delivered triple of one call, with provenance.

    role    : pattern_hop     — an edge of a confirmed chain's pattern hops
                                 (incl. rebuild-lane witness branches);
              cvt_passthrough — terminal-CVT passthrough expansion (in
                                 full_edges but not the pattern hops);
              attr            — a CVT-record attribute value (rel = the raw
                                 attr key, h = the carrier node).
    origin  : submitted — the relation belongs to the call's ENVIRONMENT-
              RESOLVED submitted family (rel_idxs + its inverse-fold family,
              per §5.6: judge by rel_idxs, never the raw submitted string);
              bridge    — every other relation on the chain (pattern prefix
              / environment expansion). Attr triples inherit submitted when
              their carrier CVT is an endpoint of a submitted-origin edge.
    is_new  : multi-dimensional first-delivery flags over the (h, rel, t)
              EXACT identity (Codex §7.6 adopted 2026-09-30):
              new_to_episode (whole case), new_to_tree (same canon_fid
              history), redisplayed (= delivered by an EARLIER call of the
              episode, i.e. not new_to_episode)."""
    h: str
    rel: str
    t: str
    role: str
    origin: str
    chain_pos: Optional[int] = None    # position within the owning chain
    chain_len: Optional[int] = None    # hops of the owning chain (witness-only: None)
    new_to_episode: bool = True
    new_to_tree: bool = True
    redisplayed: bool = False

    def to_dict(self) -> dict:
        return {"h": self.h, "rel": self.rel, "t": self.t,
                "role": self.role, "origin": self.origin,
                "chain_pos": self.chain_pos, "chain_len": self.chain_len,
                "new_to_episode": self.new_to_episode,
                "new_to_tree": self.new_to_tree,
                "redisplayed": self.redisplayed}


@dataclass
class EvidenceRecord:
    """One retrieve_subgraph call's ledger row (shadow; see module header).
    `frontier` = this call's post-delivery frontier (named chain terminals);
    `pool_delta` = the canonical-fid candidate-pool increment (filled at the
    pool-write point in _sg_finalize; None on early-return paths)."""
    seq: int = 0                        # call ordinal within the episode
    canon_fid: str = ""
    center: str = ""                    # first center display name
    submitted_rels: List[str] = field(default_factory=list)    # full names
    submitted_rel_idxs: List[int] = field(default_factory=list)
    completeness: Dict[str, object] = field(default_factory=dict)
    triples: List[EvidenceTriple] = field(default_factory=list)
    frontier: Optional[List[str]] = None
    pool_delta: Optional[int] = None
    outcome: str = "pending"            # delivered | no_evidence

    def to_dict(self) -> dict:
        return {"seq": self.seq, "canon_fid": self.canon_fid,
                "center": self.center,
                "submitted_rels": list(self.submitted_rels),
                "submitted_rel_idxs": list(self.submitted_rel_idxs),
                "completeness": dict(self.completeness),
                "triples": [tr.to_dict() for tr in self.triples],
                "frontier": list(self.frontier) if self.frontier is not None else None,
                "pool_delta": self.pool_delta,
                "outcome": self.outcome}


# ─────────────────────────────────────────────────────────────────────────────
# R1 THREE-LAYER LEDGER (2026-10-01, proposal §6.7 label v2 + §6.8 rulings +
# §7 R1; Codex audit R03/R04/R05/R07 root-outs). ZERO BEHAVIOR CHANGE for the
# live chain: render / filters / pools / JOIN / COMMIT / verdicts are untouched
# — this stage only rebuilds the ledger's DATA STRUCTURE; consumer switching
# is R3. Layers:
#   SourceFact   — one fact, one identity: (h, rel, t) with FULL names, kinds,
#                  raw-graph direction and source. The attr short key is the
#                  SAME fact's display_key, never a second identity (R05).
#   Observation  — one usage record per (attempt, call, fact): role / origin /
#                  pattern / position / traversal direction, plus EVERY
#                  same-call usage (edge_meta first-wins is dead — R05/§6.7-1).
#                  Facts↔trees are many-to-many (§6.7-2): one fact may carry
#                  observations from many calls and canon_fids.
#   CallCommit   — per-call ATOMIC commit: pending draft → SUCCESS /
#                  NO_EVIDENCE / FAILED; only SUCCESS writes delivered
#                  history (R03: a failed call consumes no NEW). Frontier is
#                  kept FULL (structural node ids) with a truncated preview
#                  face + counts (R07); completeness splits
#                  retrieval/ledger/display dimensions (R07).
# The legacy EvidenceRecord remains the PROJECTION surface (sidecar format,
# offline aligners) — build_evidence_record is unchanged and every commit
# carries its record, so the old serialization stays byte-compatible and
# old sidecars load (EvidenceRecordLog.from_legacy_records).
# ─────────────────────────────────────────────────────────────────────────────

LEDGER_SCHEMA_VERSION = 2

# Ruling ⑥ (user 2026-10-01, PERMANENT): is_new / carrier identity is the
# EXACT (h, rel, t) fact key, forever. The R2-era SEQ_LEDGER_FAMILY_ISNEW
# switch (inverse-family fold with endpoint swap rewriting the is_new trio)
# is DEAD — no code path reads it anymore; contract tests pin that even a
# stray =1 in the environment cannot move an identity. SourceFact.family_key
# remains as DATA for retrieval/relevance consumers only (GTE relation
# families), never as an identity.



@dataclass
class SourceFact:
    """Layer 1 — the immutable fact identity (§6.7-1). `fact_key` is the
    canonical (h_id, rel_id, t_id) with FULL relation names; a bracket-attr
    row [k: v] of the same fact rides as `display_key` (R05: the walked
    full name and the attr short key no longer create two identities).
    `direction` records the RAW graph orientation of the stored endpoints —
    V3-03: the execution layer keeps the raw h/r/t and the rebuild lane's
    fwd/rev TRAVERSAL direction lives on the Observation (traversal_dir /
    FactUsage), never as a rewritten fact; the same raw edge walked from
    either side is ONE fact (no double NEW), while (A,r,B) and (B,r,A) of
    one relation coexist as the two distinct facts they are.
    `family_key` is DATA ONLY (ruling ⑥ permanent): retrieval/relevance
    consumers (GTE relation families) may group on it — it never resolves
    identity and never moves is_new."""
    h_id: str
    rel_id: str
    t_id: str
    h_kind: str = ""            # NAMED | CVT | ID_NODE | LITERAL (kinds.py)
    t_kind: str = ""
    t_is_literal: bool = False
    direction: str = "forward"  # endpoints stored in RAW graph orientation
    source: str = "walk"        # walk | case_attr | records_unresolved
    display_key: str = ""       # attr display form ('' when the walk full
                                # name IS the display form)
    family_key: tuple = None    # DATA (retrieval/relevance only, ruling ⑥)

    @property
    def fact_key(self) -> tuple:
        return (self.h_id, self.rel_id, self.t_id)

    def to_dict(self) -> dict:
        return {"schema_version": LEDGER_SCHEMA_VERSION,
                "h_id": self.h_id, "rel_id": self.rel_id, "t_id": self.t_id,
                "h_kind": self.h_kind, "t_kind": self.t_kind,
                "t_is_literal": self.t_is_literal,
                "direction": self.direction, "source": self.source,
                "display_key": self.display_key,
                "family_key": [sorted(x) if isinstance(x, frozenset) else x
                               for x in (self.family_key or ())]}


@dataclass
class FactUsage:
    """One in-call occurrence of a fact (walk lane). The Observation keeps
    ALL of them; the first is the primary usage the legacy projection
    shows (identical to the old edge_meta setdefault)."""
    lane: str                   # hop | full | witness
    pattern_id: Optional[str] = None
    chain_pos: Optional[int] = None
    chain_len: Optional[int] = None
    traversal_dir: str = "unknown"   # forward | reversed | unknown

    def to_dict(self) -> dict:
        return {"lane": self.lane, "pattern_id": self.pattern_id,
                "chain_pos": self.chain_pos, "chain_len": self.chain_len,
                "traversal_dir": self.traversal_dir}


@dataclass
class Observation:
    """Layer 2 — one usage record per (attempt, call, fact) (§6.7-1/2).
    role/origin/chain position are USAGE-level, never fact-level. The
    is_new trio mirrors the legacy EvidenceTriple flags of this usage's
    display form (canonical fact-key history — V3-02: walked and bracket
    views of the SAME SourceFact can never double-count a NEW delivery).
    depends_on / origin_call_id (ruling ② 2026-10-01) carry the
    dependency provenance the evaluator's counterfactual masks on: the
    call_ids whose delivered surface seeded THIS call's walk (empty when
    the call anchored on a question entity directly), and the episode
    FIRST-DELIVERY call of the fact (== call_id for a NEW delivery)."""
    obs_id: int = -1                    # assigned at ledger append
    attempt_id: int = 0
    call_id: str = ""
    seq: int = 0
    fact_key: tuple = None
    tree_id: str = ""                   # canon_fid — fact↔tree many-to-many
    lane: str = "walk"                  # walk | attr
    role: str = "pattern_hop"           # primary role (legacy projection)
    origin: str = "bridge"
    rel_display: str = ""               # full name (walk) / display_key (attr)
    pattern_id: Optional[str] = None
    chain_pos: Optional[int] = None
    chain_len: Optional[int] = None
    traversal_dir: str = "unknown"
    order_key: tuple = None             # projection order anchor (idx triple)
    new_to_episode: bool = True
    new_to_tree: bool = True
    redisplayed: bool = False
    usages: List[FactUsage] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    origin_call_id: str = ""

    def to_dict(self) -> dict:
        return {"obs_id": self.obs_id, "attempt_id": self.attempt_id,
                "call_id": self.call_id, "seq": self.seq,
                "fact_key": list(self.fact_key or ()),
                "tree_id": self.tree_id, "lane": self.lane,
                "role": self.role, "origin": self.origin,
                "rel_display": self.rel_display,
                "pattern_id": self.pattern_id,
                "chain_pos": self.chain_pos, "chain_len": self.chain_len,
                "traversal_dir": self.traversal_dir,
                "new_to_episode": self.new_to_episode,
                "new_to_tree": self.new_to_tree,
                "redisplayed": self.redisplayed,
                "usages": [u.to_dict() for u in self.usages],
                "depends_on": list(self.depends_on),
                "origin_call_id": self.origin_call_id}


@dataclass
class CallCommit:
    """Layer 3 — one call's atomic commit (R03/R07). `legacy` is the
    unchanged build_evidence_record output this commit projects to (the
    sidecar / aligner surface); the three-layer fields are the new source
    of truth for R2+ consumers. `centers` = ALL walk anchors of the call
    (V3-01: the projection's anchor-exclusion set; [center] when the
    caller passes none — old sidecars)."""
    attempt_id: int = 0
    episode_id: int = 0
    seq: int = 0
    canon_fid: str = ""
    center: str = ""
    centers: List[str] = field(default_factory=list)
    submitted_rels: List[str] = field(default_factory=list)
    submitted_rel_idxs: List[int] = field(default_factory=list)
    state: str = "PENDING"               # PENDING | SUCCESS | NO_EVIDENCE | FAILED
    outcome: str = "pending"             # legacy mirror for quick filters
    ids: List[TripleId] = field(default_factory=list)   # exact ids staged
    fact_keys: List[tuple] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    completeness: Dict[str, dict] = field(default_factory=dict)
    frontier_node_ids: List[int] = field(default_factory=list)   # FULL (R07)
    frontier_preview: List[str] = field(default_factory=list)    # ≤50 named
    frontier_total: int = 0
    frontier_dropped: int = 0
    pool_delta: Optional[int] = None
    tool_message_idx: Optional[int] = None
    legacy: Optional[EvidenceRecord] = None

    @property
    def call_id(self) -> str:
        return f"att{self.attempt_id}.call{self.seq}"

    def to_dict(self) -> dict:
        return {"schema_version": LEDGER_SCHEMA_VERSION,
                "call_id": self.call_id,
                "attempt_id": self.attempt_id, "episode_id": self.episode_id,
                "seq": self.seq, "canon_fid": self.canon_fid,
                "center": self.center,
                "centers": list(self.centers),
                "submitted_rels": list(self.submitted_rels),
                "submitted_rel_idxs": list(self.submitted_rel_idxs),
                "state": self.state, "outcome": self.outcome,
                "completeness": {k: dict(v) for k, v in
                                 (self.completeness or {}).items()},
                "frontier_node_ids": list(self.frontier_node_ids),
                "frontier_preview": list(self.frontier_preview),
                "frontier_total": self.frontier_total,
                "frontier_dropped": self.frontier_dropped,
                "pool_delta": self.pool_delta,
                "tool_message_idx": self.tool_message_idx,
                "fact_keys": [list(k) for k in self.fact_keys],
                "observations": [o.to_dict() for o in self.observations]}


@dataclass
class LedgerDraft:
    """build_call_ledger output — the pending bundle begin_call admits."""
    facts: List[SourceFact] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    commit: CallCommit = None
    legacy: EvidenceRecord = None
    ids: List[TripleId] = field(default_factory=list)


class EvidenceRecordLog:
    """Episode-scope holder for the THREE-LAYER ledger (R1, 2026-10-01;
    proposal §6.7/§7). Zero-behavior-change refactor of the phase-2 shadow
    ledger: the D3-0 views and every legacy reader keep their exact
    surface (`records`, `commit`, `prior_episode`, `prior_tree`,
    `observed_episode`), but the bottom layer is now
        facts         — SourceFact registry (one fact, one identity)
        observations  — append-only per-(call, fact) usage records
        commits       — per-call CallCommit, ATOMIC (R03): a pending draft
                        only writes delivered history on SUCCESS; a
                        NO_EVIDENCE/FAILED call consumes no NEW.
    Lazy-initialized on ctx (reset_seq_episode_state ARCHIVES the attempt
    into ctx.evidence_archive before clearing it — R04 — instead of
    dropping it)."""

    schema_version = 2

    def __init__(self, attempt_id: int = 0, episode_id: Optional[int] = None):
        self.attempt_id = int(attempt_id)
        self.episode_id = self.attempt_id if episode_id is None \
            else int(episode_id)
        self.commits: List["CallCommit"] = []
        self.facts: Dict[tuple, "SourceFact"] = {}
        self.observations: List["Observation"] = []
        # V3-02 (2026-10-01): the delivered history is CANONICAL — it folds
        # SourceFact.fact_keys ((h, rel, t) with FULL names) ONLY. The
        # legacy short-key attr identity ((carrier, short_key, value)) lives
        # exclusively inside the legacy projection's rows and never gates
        # is_new: a fact first WALKED and later seen as a bracket row (or
        # vice versa) is ONE first delivery, never two. Ruling ⑥ (same day)
        # permanently closed the inverse-family identity rewrite —
        # SEQ_LEDGER_FAMILY_ISNEW is dead; SourceFact.family_key survives as
        # DATA for retrieval/relevance views only, never as an identity.
        self.episode_seen: Set[TripleId] = set()   # fact keys, SUCCESS-only
        self.tree_seen: Dict[str, Set[TripleId]] = {}
        self._pending: Optional["CallCommit"] = None
        self._observed_cache: Optional[Set[str]] = None   # lazy, commit-invalidated

    # ---------- public read surface for the R2 view layer ----------
    def delivered_commits(self) -> List["CallCommit"]:
        """Delivered commits in ledger order — the legal-evidence gate the
        R2 views (kgqa/agent/views.py) iterate. Same test as _delivered:
        SUCCESS state (+ the legacy outcome mirror). Views are READ-ONLY;
        this is the one sanctioned entry into the commit list."""
        return [c for c in self.commits if self._delivered(c)]

    # ---------- legacy surface (phase-2 readers, verbatim semantics) ----------
    @property
    def records(self) -> List[EvidenceRecord]:
        """The per-call legacy EvidenceRecords — now a PROJECTION of the
        commit layer (each commit carries the record built by
        build_evidence_record; late _sg_finalize fills mutate it in place,
        so the projection always reflects them)."""
        return [c.legacy for c in self.commits if c.legacy is not None]

    def prior_episode(self) -> Set[TripleId]:
        """Identities delivered before the CURRENT call — snapshot helper.
        R03: only SUCCESS calls fold into this history."""
        return set(self.episode_seen)

    def prior_tree(self, canon_fid: str) -> Set[TripleId]:
        return set(self.tree_seen.get(canon_fid) or ())

    def observed_episode(self) -> Set[str]:
        """OBSERVED view over the unified triple layer (proposal §6.3; first
        live consumer = the checkpoint hallucination filter, D3-0 2026-10-01):
        the NORMALIZED endpoint names of every fact ANY successful call of
        this episode delivered — walk-edge endpoints AND the VALUE endpoints
        of CVT attr facts (role='attr'; the 2784 'Juice' specimen's
        value-form gold lives only there, which the old per-fid
        fact_evidence domain missed). R2 (2026-10-01): the implementation is
        now the STANDARD-LAYER existence view — kgqa/agent/views.existence(
        self, 'episode') — this method keeps its call-site signature (the
        D3-0 filter helper reads it) and its lazy cache; the output is
        value-for-value equal to the D3-0/R1 algorithm (pinned by contract
        tests on both layer paths).
        Scope is the whole EPISODE, not the declaring fid: a value another
        tree delivered is a placement question, not a hallucination.
        By-name only, NO alias folding (user ruling 2026-10-01: an old name
        with only the current name delivered does NOT count). normalize() is
        the single norm implementation (phase-2 trap: bare .lower() misses
        'The Twilight Saga: New Moon'). CVT carrier mids stay in the set —
        binding a DELIVERED event node is legal (ruling 2026-09-24); an
        invented one still fails the match. no_evidence calls delivered no
        block and contribute nothing."""
        if self._observed_cache is None:
            from kgqa.agent.views import existence as _existence
            self._observed_cache = _existence(self, "episode")
        return self._observed_cache

    @staticmethod
    def _delivered(c: "CallCommit") -> bool:
        """Delivered-surface gate: SUCCESS state, with the legacy path
        additionally mirroring the old `outcome != 'delivered'` filter
        (a test/loader commit of a no_evidence record contributed nothing
        in phase 2 and must keep contributing nothing)."""
        if c.state != "SUCCESS":
            return False
        if c.legacy is not None and c.legacy.outcome != "delivered":
            return False
        return True

    # ---------- legacy atomic-commit path (kept for tests + loaders) ----------
    def commit(self, rec: EvidenceRecord, ids: List[TripleId]) -> None:
        """Append the record and fold its identities into the history
        IMMEDIATELY (phase-2 semantics — a delivered call). Production
        wiring uses the atomic begin_call/finalize_call pair instead."""
        c = CallCommit(attempt_id=self.attempt_id,
                       episode_id=self.episode_id, seq=rec.seq,
                       canon_fid=rec.canon_fid, center=rec.center,
                       legacy=rec)
        c.state = "SUCCESS"
        c.outcome = rec.outcome
        self.commits.append(c)
        self._fold(c, ids)
        self._observed_cache = None

    # ---------- R1 atomic path (R03: failure consumes no NEW) ----------
    def begin_call(self, draft: "LedgerDraft") -> CallCommit:
        """Open a call's PENDING commit: register the draft's facts (first
        identity wins; later observations of the same fact reference it —
        §6.7-2 fact↔tree is many-to-many), append its observations, and
        hold the draft's identities OUT of the delivered history until the
        call reaches a terminal state. A leftover pending (crashed call)
        is promoted to FAILED first — a failed call must never write
        delivered history."""
        if self._pending is not None:
            self.finalize_call("FAILED")
        for f in draft.facts:
            cur = self.facts.get(f.fact_key)
            if cur is None:
                self.facts[f.fact_key] = f
            elif not cur.display_key and f.display_key:
                # V3-02: one fact, one identity — a later call's bracket
                # view of an already-registered fact only teaches its
                # display form (never a second identity)
                cur.display_key = f.display_key
        for ob in draft.observations:
            ob.obs_id = len(self.observations)
            self.observations.append(ob)
        self._pending = draft.commit
        self.commits.append(draft.commit)
        return draft.commit

    def finalize_call(self, state: str, pool_delta: Optional[int] = None,
                      render_lines_dropped: Optional[int] = None,
                      display_stats: Optional[dict] = None) -> None:
        """Terminate the pending commit. SUCCESS folds its identities into
        the delivered history (the ONLY write — R03); NO_EVIDENCE / FAILED
        leave it untouched. pool_delta / render_lines_dropped complete the
        commit's own fields (the legacy record's copies are filled by the
        unchanged _sg_finalize sites). display_stats (R3-b, §6.6/F03) is the
        render's counted truncation face: the >80 merged-edge-group cut
        events and the also-in-evidence spill count — the display dimension
        of completeness, recorded separately from the walk budget."""
        c = self._pending
        if c is None:
            return
        c.state = state
        if state == "SUCCESS":
            c.outcome = "delivered"
            self._fold(c, c.ids)
            self._observed_cache = None
        elif state == "NO_EVIDENCE":
            c.outcome = "no_evidence"
        else:
            c.outcome = "failed"
        if pool_delta is not None:
            c.pool_delta = pool_delta
        if render_lines_dropped:
            c.completeness["display"] = {
                "status": "PARTIAL",
                "render_lines_dropped": int(render_lines_dropped)}
        if display_stats:
            # §6.6: completeness records ONLY the two countable group events
            # (tails>80 / heads>80); the also-in-evidence index is zero-loss
            # (names all render, in the appendix) so it counts as a stat but
            # never flips the status.
            _gt = int(display_stats.get("group_truncations", 0) or 0)
            c.completeness["display"] = {
                "status": "PARTIAL" if _gt else "COMPLETE",
                "group_truncations": _gt,
                "truncated_members":
                    int(display_stats.get("truncated_members", 0) or 0),
                "also_in_evidence":
                    int(display_stats.get("also_in_evidence", 0) or 0)}
        self._pending = None

    def _fold(self, c: "CallCommit", ids: List[TripleId]) -> None:
        # V3-02: ids are CANONICAL fact keys on the atomic path (the
        # LedgerDraft carries the facts' own keys); the legacy commit()
        # path keeps passing whatever identities its caller chose (tests
        # and from_legacy_records — the legacy projection's own keys).
        self.episode_seen.update(ids or ())
        tree = self.tree_seen.setdefault(c.canon_fid, set())
        tree.update(ids or ())

    # ---------- R04 archive / sidecar export ----------
    def archive(self) -> dict:
        """Append-only attempt snapshot for the episode switch
        (reset_seq_episode_state). A still-pending commit (crashed call)
        is promoted to FAILED first — the archive is a closed ledger."""
        if self._pending is not None:
            self.finalize_call("FAILED")
        return self.export()

    def export(self) -> dict:
        """Full three-layer snapshot (rollout sidecar `evidence_attempts`
        entries / replay DUMP_EVIDENCE `attempts`)."""
        return {"schema_version": self.schema_version,
                "attempt_id": self.attempt_id,
                "episode_id": self.episode_id,
                "records": [c.legacy.to_dict() for c in self.commits
                            if c.legacy is not None],
                "commits": [c.to_dict() for c in self.commits],
                "facts": [f.to_dict() for f in self.facts.values()]}

    # ---------- R1 loaders (backward compatibility) ----------
    @classmethod
    def from_sidecar(cls, attempts) -> "EvidenceRecordLog":
        """Rebuild a CASE-scope three-layer log from the sidecar export
        (rollout `evidence_attempts` / replay DUMP_EVIDENCE `attempts`)
        for OFFLINE adjudication — the R3-c evaluator's ledger source.

        Attempts are merged in list order into ONE log (the case is the
        judgment scope). Only delivered (SUCCESS) commits are kept — the
        offline judge never scores failed calls. The is_new trio and
        origin_call_id are RECOMPUTED case-wide: a restarted attempt's
        live log could not see its predecessor's history, so the stored
        flags are per-attempt episode scope; with a SINGLE attempt the
        recompute reproduces the stored flags verbatim (verified on the
        p3 dual batch), with several attempts it restores the case-wide
        first-delivery truth. Identity is the canonical fact key (V3-02,
        ruling ⑥: exact identity forever — no family fold on load)."""
        log = cls(attempt_id=0)
        _first_delivery: Dict[tuple, str] = {}
        for att in attempts or ():
            if not isinstance(att, dict):
                continue
            for fd in (att.get("facts") or ()):
                try:
                    fam = fd.get("family_key")
                    fk_fam = ((frozenset(fam[0]), fam[1])
                              if isinstance(fam, (list, tuple)) and len(fam) == 2
                              and isinstance(fam[0], (list, tuple))
                              else None)
                    sf = SourceFact(
                        h_id=str(fd.get("h_id") or ""),
                        rel_id=str(fd.get("rel_id") or ""),
                        t_id=str(fd.get("t_id") or ""),
                        h_kind=str(fd.get("h_kind") or ""),
                        t_kind=str(fd.get("t_kind") or ""),
                        t_is_literal=bool(fd.get("t_is_literal")),
                        direction=str(fd.get("direction") or "forward"),
                        source=str(fd.get("source") or "walk"),
                        display_key=str(fd.get("display_key") or ""),
                        family_key=fk_fam)
                    if sf.h_id and sf.rel_id and sf.t_id:
                        log.facts.setdefault(sf.fact_key, sf)
                except Exception:
                    continue
            for cd in (att.get("commits") or ()):
                if not isinstance(cd, dict) or cd.get("state") != "SUCCESS":
                    continue
                c = CallCommit(
                    attempt_id=int(cd.get("attempt_id") or 0),
                    episode_id=int(cd.get("episode_id") or 0),
                    seq=int(cd.get("seq") or 0),
                    canon_fid=str(cd.get("canon_fid") or ""),
                    center=str(cd.get("center") or ""),
                    centers=[str(x) for x in (cd.get("centers") or ())],
                    submitted_rels=[str(x) for x in (cd.get("submitted_rels")
                                                     or ())],
                    submitted_rel_idxs=[int(i) for i in
                                        (cd.get("submitted_rel_idxs") or ())],
                    state="SUCCESS", outcome="delivered",
                    completeness=dict(cd.get("completeness") or {}),
                    frontier_node_ids=[int(i) for i in
                                       (cd.get("frontier_node_ids") or ())],
                    frontier_preview=[str(x) for x in
                                      (cd.get("frontier_preview") or ())],
                    frontier_total=int(cd.get("frontier_total") or 0),
                    frontier_dropped=int(cd.get("frontier_dropped") or 0),
                    pool_delta=(int(cd["pool_delta"])
                                if cd.get("pool_delta") is not None else None),
                    tool_message_idx=(int(cd["tool_message_idx"])
                                      if cd.get("tool_message_idx")
                                      is not None else None))
                for od in (cd.get("observations") or ()):
                    fk = tuple(od.get("fact_key") or ())
                    if len(fk) != 3:
                        continue
                    # V3-06: deserialize the FULL usage list (multi-pattern /
                    # multi-position occurrences) — the export/load/export
                    # roundtrip is pinned by test; pre-V3 sidecars (no
                    # "usages" key) fall back to the primary-usage fields.
                    usages = [FactUsage(
                        lane=str(u.get("lane") or "hop"),
                        pattern_id=u.get("pattern_id"),
                        chain_pos=(int(u["chain_pos"])
                                   if u.get("chain_pos") is not None else None),
                        chain_len=(int(u["chain_len"])
                                   if u.get("chain_len") is not None else None),
                        traversal_dir=str(u.get("traversal_dir") or "unknown"))
                        for u in (od.get("usages") or ())
                        if isinstance(u, dict)]
                    ob = Observation(
                        attempt_id=c.attempt_id, call_id=c.call_id,
                        seq=c.seq, fact_key=fk,
                        tree_id=str(od.get("tree_id") or c.canon_fid),
                        lane=str(od.get("lane") or "walk"),
                        role=str(od.get("role") or ""),
                        origin=str(od.get("origin") or ""),
                        rel_display=str(od.get("rel_display") or ""),
                        pattern_id=od.get("pattern_id"),
                        chain_pos=od.get("chain_pos"),
                        chain_len=od.get("chain_len"),
                        traversal_dir=str(od.get("traversal_dir")
                                          or "unknown"),
                        usages=usages,
                        depends_on=[str(x) for x in
                                    (od.get("depends_on") or ())],
                        origin_call_id=str(od.get("origin_call_id") or ""))
                    # case-wide is_new recompute (see docstring)
                    ob.new_to_episode = fk not in log.episode_seen
                    ob.new_to_tree = fk not in log.tree_seen.get(
                        ob.tree_id, ())
                    ob.redisplayed = not ob.new_to_episode
                    # origin_call_id: the case-wide FIRST delivery — always
                    # recomputed (a restarted attempt's stored value only
                    # knows its own attempt's history)
                    if ob.new_to_episode:
                        ob.origin_call_id = ob.call_id
                    else:
                        ob.origin_call_id = (_first_delivery.get(fk)
                                             or ob.call_id)
                    _first_delivery.setdefault(fk, ob.call_id)
                    ob.obs_id = len(log.observations)   # V3-06 roundtrip
                    c.observations.append(ob)
                    c.ids.append(fk)
                    log.observations.append(ob)
                c.fact_keys = sorted({ob.fact_key
                                      for ob in c.observations})
                # V3-06 lossless roundtrip: rebuild the LEGACY projection
                # from the observations (every field the row carried rides
                # the observation: rel_display IS the row's rel — full name
                # on the walk lane, short display key on the attr lane)
                rec = EvidenceRecord(
                    seq=c.seq, canon_fid=c.canon_fid, center=c.center,
                    submitted_rels=list(c.submitted_rels),
                    submitted_rel_idxs=list(c.submitted_rel_idxs),
                    completeness=_legacy_completeness(c.completeness),
                    frontier=list(c.frontier_preview),
                    pool_delta=c.pool_delta, outcome="delivered")
                for ob in c.observations:
                    f = log.facts.get(ob.fact_key)
                    if f is None:
                        continue
                    rec.triples.append(EvidenceTriple(
                        h=f.h_id, rel=(ob.rel_display or f.rel_id),
                        t=f.t_id, role=ob.role, origin=ob.origin,
                        chain_pos=ob.chain_pos, chain_len=ob.chain_len,
                        new_to_episode=ob.new_to_episode,
                        new_to_tree=ob.new_to_tree,
                        redisplayed=ob.redisplayed))
                c.legacy = rec
                log.commits.append(c)
                log._fold(c, c.ids)
                log._observed_cache = None
        return log

    @classmethod
    def from_legacy_records(cls, rows: List[dict],
                            attempt_id: int = 0) -> "EvidenceRecordLog":
        """Load a PRE-R1 sidecar (`evidence_records` list of legacy record
        dicts) into the three-layer holder: every record becomes a SUCCESS
        commit over its legacy triples; the exact-identity history is
        rebuilt so is_new semantics continue for anything appended
        afterwards. R3-c: each legacy triple is ALSO registered as a
        SourceFact + Observation (identity = the legacy (h, rel, t)
        itself, role/origin/is_new read off the triple, tree_id = the
        record's canon_fid) so the offline views/evaluator can adjudicate
        a phase-2 sidecar through the same observation path as the
        three-layer format — the legacy projection (`records`) and the
        observed view are unchanged by construction."""
        log = cls(attempt_id=attempt_id)
        for row in rows or ():
            rec = evidence_record_from_dict(row)
            if rec is None:
                continue
            ids = sorted({(t.h, t.rel, t.t) for t in rec.triples})
            log.commit(rec, ids)
            c = log.commits[-1]
            # V3-02 (one fact, one identity — the runtime walk_join
            # semantics, mirrored on load): walk-lane rows register first;
            # an attr row whose short key fronts a registered walked fact
            # of the same endpoints JOINS it (the bracket row is a display
            # form of the same fact), else it keeps its legacy identity —
            # the short-key identity survives ONLY inside the legacy
            # projection, never as a second fact.
            _walk_facts: Dict[tuple, List[SourceFact]] = {}
            for t in rec.triples:
                if t.role != "attr":
                    fk = (t.h, t.rel, t.t)
                    f = log.facts.setdefault(fk, SourceFact(
                        h_id=t.h, rel_id=t.rel, t_id=t.t, source="walk"))
                    _short = (t.h, t.rel.rsplit(".", 1)[-1], t.t)
                    _walk_facts.setdefault(_short, [])
                    if f not in _walk_facts[_short]:
                        _walk_facts[_short].append(f)
            for t in rec.triples:
                fk = (t.h, t.rel, t.t)
                if t.role == "attr":
                    join = _walk_facts.get((t.h, t.rel, t.t)) or []
                    if join:
                        fk = join[0].fact_key
                        if t.rel and not join[0].display_key:
                            join[0].display_key = t.rel
                    else:
                        log.facts.setdefault(fk, SourceFact(
                            h_id=t.h, rel_id=t.rel, t_id=t.t,
                            source="case_attr"))
                else:
                    log.facts.setdefault(fk, SourceFact(
                        h_id=t.h, rel_id=t.rel, t_id=t.t, source="walk"))
                ob = Observation(
                    attempt_id=c.attempt_id, call_id=c.call_id, seq=c.seq,
                    fact_key=fk, tree_id=rec.canon_fid,
                    lane=("attr" if t.role == "attr" else "walk"),
                    role=t.role, origin=t.origin, rel_display=t.rel,
                    new_to_episode=t.new_to_episode,
                    new_to_tree=t.new_to_tree, redisplayed=t.redisplayed,
                    origin_call_id=(c.call_id if t.new_to_episode else ""))
                ob.obs_id = len(log.observations)
                log.observations.append(ob)
                c.observations.append(ob)
        return log


def _legacy_completeness(dimensions: Dict[str, dict]) -> dict:
    """The legacy record's mixed-dimension completeness dict, folded back
    out of the three split dimensions (from_sidecar's V3-06 lossless
    reconstruction)."""
    dims = dimensions or {}
    ret = dims.get("retrieval", {}) or {}
    disp = dims.get("display", {}) or {}
    led = dims.get("ledger", {}) or {}
    part = any(d.get("status") == "PARTIAL"
               for d in (ret, disp, led)) or bool(ret.get(
                   "attr_values_excluded_long"))
    return {"status": "PARTIAL" if part else "COMPLETE",
            "rebuild_truncations": int(ret.get("rebuild_truncations", 0)
                                       or 0),
            "render_lines_dropped": int(disp.get("render_lines_dropped", 0)
                                        or 0)}


def evidence_record_from_dict(row: dict) -> Optional[EvidenceRecord]:
    """Deserialize one legacy EvidenceRecord dict (old sidecar rows)."""
    if not isinstance(row, dict):
        return None
    rec = EvidenceRecord(
        seq=int(row.get("seq") or 0),
        canon_fid=str(row.get("canon_fid") or ""),
        center=str(row.get("center") or ""),
        submitted_rels=[str(r) for r in (row.get("submitted_rels") or ())],
        submitted_rel_idxs=[int(i) for i in (row.get("submitted_rel_idxs")
                                             or ())],
        completeness=dict(row.get("completeness") or {}),
        frontier=(list(row["frontier"]) if row.get("frontier") is not None
                  else None),
        pool_delta=(int(row["pool_delta"])
                    if row.get("pool_delta") is not None else None),
        outcome=str(row.get("outcome") or "pending"))
    for tr in (row.get("triples") or ()):
        if not isinstance(tr, dict):
            continue
        rec.triples.append(EvidenceTriple(
            h=str(tr.get("h") or ""), rel=str(tr.get("rel") or ""),
            t=str(tr.get("t") or ""), role=str(tr.get("role") or ""),
            origin=str(tr.get("origin") or ""),
            chain_pos=(int(tr["chain_pos"])
                       if tr.get("chain_pos") is not None else None),
            chain_len=(int(tr["chain_len"])
                       if tr.get("chain_len") is not None else None),
            new_to_episode=bool(tr.get("new_to_episode", True)),
            new_to_tree=bool(tr.get("new_to_tree", True)),
            redisplayed=bool(tr.get("redisplayed", False))))
    return rec


def folded_reps(rel_names, inv: dict) -> set:
    """Representative names of `rel_names` under the inverse-fold map (the
    same folding _edge_fact_key uses): a relation and its inverse family
    collapse onto one representative — the §5.6 'submitted family incl.
    inverses' membership test."""
    return {inv.get(str(r), str(r)) for r in rel_names}


def backfill_case_attrs(records: Dict[str, Dict[str, List[str]]],
                        attrs: Dict[str, Dict[str, List[str]]]) -> None:
    """R01 (Codex audit 2026-10-01): fold the case-graph CVT attr channel
    into the LIVE EvidenceSet.records (short-attr-key convention, order-
    preserving dedup). Until this, the named bracket values cvt_case_attrs
    reconstructs were shadow-side only (build_evidence_record's
    records_attrs) — EvidenceLog.answer_pool() missed them, so value-form
    gold sat outside the live answer pool while the shadow observed view
    had it (2784 'Juice' specimen). Delivery-enters-the-domain: one domain
    for the live pool and the shadow view. SCOPE (unresolved, kept
    verbatim): which attrs legally belong to THIS fact has no retrieval
    rule yet — the caller passes cvt_case_attrs' full per-CVT scope."""
    for carrier, ka in (attrs or {}).items():
        slot = records.setdefault(carrier, {})
        for key, values in (ka or {}).items():
            dst = slot.setdefault(key, [])
            for v in values or ():
                if v not in dst:
                    dst.append(v)


def _case_attr_rows(cvt_names, h_ids, r_ids, t_ids, ents, rels):
    """Shared scan behind cvt_case_attrs: yields (carrier, key, value,
    full_rel) for every case-graph bracket-attr row under the exact filter
    cvt_case_attrs has always applied (non-CVT tail, <60 chars, non-noisy
    key, key = last rel segment). R1 (2026-10-01) needs the FULL relation
    name too — the SourceFact identity is (h, full.rel, t) and the short
    key is only the display form (§6.7-3; R05) — so the scan surfaces it
    instead of discarding it.

    RETRIEVAL CONTRACT (V3-04, 2026-10-01): the ≥60-char VALUE exclusion
    below is a MATERIALIZE-TIME admission rule, not a display truncation —
    an excluded value never enters the ledger at all (no fact, no
    observation) and is therefore not compensated downstream; the count is
    surfaced through cvt_case_attr_stats into completeness['retrieval']
    (attr_values_excluded_long) so the sidecar can say PARTIAL-of-contract
    instead of silently narrowing the attr lane."""
    from kgqa.agent.entity_kinds import is_cvt as _is_cvt
    from kgqa.agent.seq_render_v38 import _NOISY_ATTR
    want = set(cvt_names or ())
    if not want:
        return
    for h, rl, t in zip(h_ids, r_ids, t_ids):
        if not (0 <= h < len(ents)) or not (0 <= t < len(ents)):
            continue
        hn = str(ents[h])
        if hn not in want or not _is_cvt(hn):
            continue
        tv = str(ents[t])
        full = str(rels[rl]) if 0 <= rl < len(rels) else "?"
        key = full.rsplit(".", 1)[-1]
        if _is_cvt(tv) or len(tv) >= 60 or key in _NOISY_ATTR:
            continue
        yield hn, key, tv, full


def cvt_case_attrs(cvt_names, h_ids, r_ids, t_ids, ents, rels):
    """Bracket-attr reconstruction for the CVTs of ONE call, mirroring the
    v38 renderer's cvt_graph_all semantics (seq_render_v38.py): every
    case-graph out-edge of a CVT to a NON-CVT tail (<60 chars, non-noisy
    key) yields an attr triple (cvt, key-last-segment, value). This is the
    channel the §5.6 CVT-attr carrier form judges on (1392 specimen:
    education CVT → [institution: gold]) — the walked-edges-only records
    (LITERAL tails) miss named values, and the renderer inlines them from
    the case graph, so the ledger must too."""
    out: Dict[str, Dict[str, List[str]]] = {}
    for hn, key, tv, _full in _case_attr_rows(cvt_names, h_ids, r_ids,
                                              t_ids, ents, rels):
        out.setdefault(hn, {}).setdefault(key, []).append(tv)
    return out


def cvt_case_attr_full_rels(cvt_names, h_ids, r_ids, t_ids, ents, rels):
    """R05/V3-04 companion map: (carrier, key, value) -> the LIST of FULL
    relation names behind the bracket-attr row. Two complete relations can
    share one short key (film.actor / tv.actor both display as 'actor') —
    the pre-V3 single-value map silently kept only the LAST one, deleting
    the other fact from the ledger; the multi-valued map lets the attr
    lane materialize EACH complete relation as its own SourceFact (same
    value, distinct identities), with the short key staying their shared
    display form."""
    out: Dict[Tuple[str, str, str], List[str]] = {}
    for (c, k, v, fr) in _case_attr_rows(cvt_names, h_ids, r_ids, t_ids,
                                         ents, rels):
        lst = out.setdefault((c, k, v), [])
        if fr not in lst:
            lst.append(fr)
    return out


def cvt_case_attr_stats(cvt_names, h_ids, r_ids, t_ids, ents, rels):
    """V3-04 completeness face of the attr lane's retrieval contract: the
    number of DISTINCT (carrier, key, value) rows the ≥60-char value rule
    excluded at materialize time (never delivered, never displayed —
    recorded so the exclusion is countable, not silent)."""
    seen: Set[Tuple[str, str, str]] = set()
    n = 0
    from kgqa.agent.entity_kinds import is_cvt as _is_cvt
    from kgqa.agent.seq_render_v38 import _NOISY_ATTR
    want = set(cvt_names or ())
    for h, rl, t in zip(h_ids, r_ids, t_ids):
        if not (0 <= h < len(ents)) or not (0 <= t < len(ents)):
            continue
        hn = str(ents[h])
        if hn not in want or not _is_cvt(hn):
            continue
        tv = str(ents[t])
        full = str(rels[rl]) if 0 <= rl < len(rels) else "?"
        key = full.rsplit(".", 1)[-1]
        if _is_cvt(tv) or key in _NOISY_ATTR:
            continue
        row = (hn, key, tv)
        if len(tv) >= 60 and row not in seen:
            seen.add(row)
            n += 1
    return {"attr_values_excluded_long": n}


def build_evidence_record(*, seq, canon_fid, center, submitted_rel_idxs,
                          rel_full_names, inv, confirmed, witness_edges,
                          ents, rels, records_attrs, prior_episode,
                          prior_tree, truncations, render_lines_dropped,
                          frontier_names, edge_dirs=None):
    """Pure builder (no ctx import — pass primitives; keeps evidence.py
    import-cycle-free and unit-testable). Returns (EvidenceRecord, ids) —
    ids = the exact identities this call delivers (for the history commit).

    confirmed      : {(center_idx, pattern_key): [chain...]} — chain dicts
                     with idx-space `edges` (pattern hops) / `full_edges`
                     (hops + terminal-CVT passthrough).
    witness_edges  : the rebuild lane's witness union (idx triples).
    records_attrs  : {carrier_name: {attr_key: [values]}} (EvidenceSet.records).
    inv            : the inverse-fold map (ctx-level, from the walk side).
    edge_dirs      : V3-03 — {traversal idx triple: 'fwd' | 'rev'} from the
                     rebuild lane; a 'rev' discovery means the RAW graph
                     edge runs t→h, so the staged identity (and the legacy
                     row) keep the graph's own orientation and the
                     traversal direction rides the Observation instead.
                     Missing/None = legacy chains (orientation as stored)."""

    def _nm(i):
        return str(ents[i]) if 0 <= i < len(ents) else str(i)

    def _raw(e):
        """RAW-oriented idx triple of one traversal edge (V3-03)."""
        te = tuple(e)
        if edge_dirs and edge_dirs.get(te) == "rev":
            return (te[2], te[1], te[0])
        return te

    sub_reps = folded_reps((rel_full_names[i] for i in submitted_rel_idxs
                            if 0 <= i < len(rel_full_names)), inv)

    def _origin(ridx: int) -> str:
        rn = str(rels[ridx]) if 0 <= ridx < len(rels) else str(ridx)
        return "submitted" if inv.get(rn, rn) in sub_reps else "bridge"

    # edge RAW-idx-triple -> (role, chain_pos, chain_len); first chain wins
    # (dedup), witness-only edges carry None position.
    edge_meta: Dict[Tuple[int, int, int], Tuple[str, Optional[int], Optional[int]]] = {}
    for _key, chains in (confirmed or {}).items():
        for ch in chains or ():
            hops = list(ch.get("edges") or ())
            full = list(ch.get("full_edges") or ())
            for pos, e in enumerate(hops):
                edge_meta.setdefault(_raw(e),
                                     ("pattern_hop", pos, len(hops)))
            for e in full:
                edge_meta.setdefault(_raw(e), ("cvt_passthrough", None, None))
    for e in (witness_edges or ()):
        edge_meta.setdefault(_raw(e), ("pattern_hop", None, None))

    rec = EvidenceRecord(seq=seq, canon_fid=canon_fid, center=center)
    rec.submitted_rel_idxs = [int(i) for i in (submitted_rel_idxs or ())]
    rec.submitted_rels = [str(rel_full_names[i]) for i in rec.submitted_rel_idxs
                          if 0 <= i < len(rel_full_names)]
    status = "PARTIAL" if (truncations or render_lines_dropped) else "COMPLETE"
    rec.completeness = {"status": status,
                        "rebuild_truncations": int(truncations or 0),
                        "render_lines_dropped": int(render_lines_dropped or 0)}

    seen_ids: Set[TripleId] = set()
    # walk edges first (dedup, stable order), then CVT-record attrs
    sub_hit_nodes: Set[str] = set()     # endpoints of submitted-origin edges
    for e in sorted(edge_meta):
        h_i, r_i, t_i = e
        role, pos, clen = edge_meta[e]
        h, r, t = _nm(h_i), (str(rels[r_i]) if 0 <= r_i < len(rels) else str(r_i)), _nm(t_i)
        tid = (h, r, t)
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        origin = _origin(r_i)
        if origin == "submitted":
            sub_hit_nodes.update((h, t))
        rec.triples.append(EvidenceTriple(
            h=h, rel=r, t=t, role=role, origin=origin,
            chain_pos=pos, chain_len=clen,
            new_to_episode=tid not in prior_episode,
            new_to_tree=tid not in prior_tree,
            redisplayed=tid in prior_episode))
    for carrier, attrs in (records_attrs or {}).items():
        for key, values in (attrs or {}).items():
            for v in values or ():
                tid = (carrier, key, str(v))
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
                rec.triples.append(EvidenceTriple(
                    h=carrier, rel=key, t=str(v), role="attr", origin="",
                    chain_pos=None, chain_len=None,
                    new_to_episode=tid not in prior_episode,
                    new_to_tree=tid not in prior_tree,
                    redisplayed=tid in prior_episode))
    # attr origin: submitted iff the carrier CVT is hit by a submitted-origin
    # edge of THIS call (§5.6: 提交关系命中的 CVT 的括号属性)
    for tr in rec.triples:
        if tr.role == "attr":
            tr.origin = "submitted" if tr.h in sub_hit_nodes else "bridge"
    rec.frontier = list(frontier_names or [])
    rec.outcome = "delivered"
    return rec, seen_ids


# ─────────────────────────────────────────────────────────────────────────────
# R1 three-layer builder (2026-10-01). build_evidence_record above is
# UNCHANGED and remains the legacy projection's author — build_call_ledger
# calls it, then ENRICHES its rows into SourceFact/Observation/CallCommit.
# Reading role/origin/chain position/is_new back from the legacy rows
# (instead of recomputing) pins the two layers together by construction:
# the projection can never drift from the new source of truth.
# ─────────────────────────────────────────────────────────────────────────────

def _pattern_id(key) -> str:
    """Stable pattern identity from the confirmed-dict key
    ((center_idx, hops-tuple) at the _rebuild_pe_list write site)."""
    ci, pk = key if isinstance(key, tuple) and len(key) == 2 else ("?", key)

    def _hop(h):
        if isinstance(h, str):
            return h
        try:
            return "{" + ".".join(str(x) for x in sorted(h, key=str)) + "}"
        except TypeError:
            return str(h)

    hops = ">".join(_hop(h) for h in pk) if isinstance(pk, (tuple, list)) \
        else str(pk)
    return f"c{ci}|{hops}"


def _traversal_dir(full_nodes, e, edge_dirs=None) -> str:
    """Traversal direction of one chain edge, V3-03 semantics: with the
    rebuild lane's discovery map the direction is the BFS fact (fwd = the
    walk follows the raw h→t edge, rev = against it). Legacy chains (no
    edge_dirs) fall back to visitation order against the STORED triple —
    which was traversal-oriented, so this stays the phase-2 face."""
    if edge_dirs is not None:
        d = edge_dirs.get(e)
        if d == "fwd":
            return "forward"
        if d == "rev":
            return "reversed"
        return "unknown"
    pos = {n: i for i, n in enumerate(full_nodes or ())}
    hi, ti = pos.get(e[0]), pos.get(e[2])
    if hi is None or ti is None:
        return "unknown"
    return "forward" if hi < ti else "reversed"


def build_call_ledger(*, seq, canon_fid, center, submitted_rel_idxs,
                      rel_full_names, inv, confirmed, witness_edges,
                      ents, rels, records_attrs, prior_episode, prior_tree,
                      truncations, render_lines_dropped, frontier_names,
                      frontier_node_ids=None, frontier_total=None,
                      attr_full_rels=None, attempt_id=0, episode_id=None,
                      tool_message_idx=None, edge_dirs=None, centers=None,
                      attr_stats=None
                      ) -> LedgerDraft:
    """Pure three-layer builder (R1 + V3 fixes 2026-10-01). Same primitive
    inputs as build_evidence_record (which it calls for the legacy
    projection), plus:
      frontier_node_ids — the FULL structural frontier (node idxs, R07);
      frontier_total    — the named frontier size BEFORE the ≤50 preview
                          cap (R07 dropped count);
      attr_full_rels    — {(carrier, key, value): [full rel names]} from
                          cvt_case_attr_full_rels — the R05/V3-04
                          resolution: EVERY complete relation behind a
                          short key materializes as its own SourceFact,
                          the short key stays their shared display form;
      attempt_id / episode_id / tool_message_idx — R04 alignment anchors;
      edge_dirs         — V3-03 {traversal idx triple: 'fwd'|'rev'}: the
                          walk lane stages facts in RAW graph orientation
                          and records the traversal direction on the
                          usage/observation (the same raw edge walked from
                          either side is ONE fact, never two NEWs);
      centers           — ALL walk anchors of the call (V3-01 projection
                          anchor-exclusion set; defaults to [center]);
      attr_stats        — V3-04 completeness face of the attr lane's
                          retrieval contract (cvt_case_attr_stats — the
                          ≥60-char value exclusions, countable not silent).
    is_new (V3-02): computed ONCE per fact over the canonical fact key
    against the canonical priors (episode_seen / tree_seen fold fact keys
    only) — a walked→bracket (or bracket→walked) re-sight of the same
    SourceFact is a REDISPLAY, never a second NEW; the flags propagate to
    the legacy projection's rows so the sidecar and the ledger agree.
    Ruling ⑥: no family rewrite exists — the identity is exact, forever.
    Returns a LedgerDraft for EvidenceRecordLog.begin_call (PENDING)."""

    def _nm(i):
        return str(ents[i]) if 0 <= i < len(ents) else str(i)

    def _raw(e):
        """RAW-oriented idx triple of one traversal edge (V3-03)."""
        te = tuple(e)
        if edge_dirs and edge_dirs.get(te) == "rev":
            return (te[2], te[1], te[0])
        return te

    rec, _legacy_ids = build_evidence_record(
        seq=seq, canon_fid=canon_fid, center=center,
        submitted_rel_idxs=submitted_rel_idxs,
        rel_full_names=rel_full_names, inv=inv, confirmed=confirmed,
        witness_edges=witness_edges, ents=ents, rels=rels,
        records_attrs=records_attrs, prior_episode=prior_episode,
        prior_tree=prior_tree, truncations=truncations,
        render_lines_dropped=render_lines_dropped,
        frontier_names=frontier_names, edge_dirs=edge_dirs)
    legacy_by_tid = {(t.h, t.rel, t.t): t for t in rec.triples}

    # walk usages — the twin of the legacy edge_meta loop, keeping EVERY
    # occurrence (R05/§6.7-1: first-wins is dead at the data layer; the
    # legacy projection still shows the first usage, as it always did).
    # Keys are RAW idx triples (V3-03).
    from kgqa.agent.entity_kinds import entity_kind as _ek
    usages: Dict[tuple, List[FactUsage]] = {}
    for _key, chains in (confirmed or {}).items():
        pid = _pattern_id(_key)
        for ch in chains or ():
            hops = list(ch.get("edges") or ())
            fnodes = list(ch.get("full_nodes") or ())
            for pos, e in enumerate(hops):
                te = tuple(e)
                usages.setdefault(_raw(te), []).append(FactUsage(
                    "hop", pid, pos, len(hops),
                    _traversal_dir(fnodes, te, edge_dirs)))
            for e in (ch.get("full_edges") or ()):
                te = tuple(e)
                usages.setdefault(_raw(te), []).append(FactUsage(
                    "full", pid, None, None,
                    _traversal_dir(fnodes, te, edge_dirs)))
    for e in (witness_edges or ()):
        te = tuple(e)
        usages.setdefault(_raw(te), []).append(
            FactUsage("witness", traversal_dir=(
                _traversal_dir((), te, edge_dirs)
                if (edge_dirs and edge_dirs.get(te)) else "unknown")))

    _role = {"hop": "pattern_hop", "full": "cvt_passthrough",
             "witness": "pattern_hop"}
    facts: List[SourceFact] = []
    fact_by_key: Dict[tuple, SourceFact] = {}
    # (h, last-seg, t) -> [facts] — V3-04: a short key can front SEVERAL
    # walked full relations; the attr lane joins by exact full name first.
    walk_join: Dict[tuple, List[SourceFact]] = {}
    legacy_tid_facts: Dict[TripleId, List[tuple]] = {}   # legacy tid -> fact keys
    observations: List[Observation] = []
    call_id = f"att{int(attempt_id)}.call{int(seq)}"
    ep_id = int(attempt_id) if episode_id is None else int(episode_id)
    seen_tids: Set[TripleId] = set()

    # walk lane (legacy order: sorted idx triples; NAME-level dedup keeps
    # the first — the legacy seen_ids semantics). Keys are RAW idx triples
    # (V3-03): the fact keeps the graph's own h→t orientation.
    for e in sorted(usages):
        h_i, r_i, t_i = e
        h = _nm(h_i)
        r = str(rels[r_i]) if 0 <= r_i < len(rels) else str(r_i)
        t = _nm(t_i)
        tid = (h, r, t)
        if tid in seen_tids:
            continue
        seen_tids.add(tid)
        hk, tk = _ek(h), _ek(t)
        f = fact_by_key.get(tid)
        if f is None:
            f = SourceFact(h_id=h, rel_id=r, t_id=t, h_kind=hk, t_kind=tk,
                           t_is_literal=(tk == "LITERAL"), source="walk",
                           family_key=(frozenset((h, t)), inv.get(r, r)))
            fact_by_key[tid] = f
            facts.append(f)
        walk_join.setdefault((h, r.rsplit(".", 1)[-1], t), [])
        if f not in walk_join[(h, r.rsplit(".", 1)[-1], t)]:
            walk_join[(h, r.rsplit(".", 1)[-1], t)].append(f)
        legacy_tid_facts.setdefault(tid, []).append(tid)
        prim = usages[e][0]
        lg = legacy_by_tid.get(tid)
        observations.append(Observation(
            attempt_id=int(attempt_id), call_id=call_id, seq=int(seq),
            fact_key=tid, tree_id=canon_fid, lane="walk",
            role=_role[prim.lane], origin=(lg.origin if lg else "bridge"),
            rel_display=r, pattern_id=prim.pattern_id,
            chain_pos=prim.chain_pos, chain_len=prim.chain_len,
            traversal_dir=prim.traversal_dir, order_key=e,
            origin_call_id=call_id,
            usages=list(usages[e])))

    # attr lane (legacy order: records_attrs iteration). R05/V3-04: resolve
    # EVERY full-relation identity behind the bracket row — each complete
    # relation sharing the short key materializes as its OWN SourceFact
    # (the pre-V3 single-value map silently dropped the others); the short
    # key stays their shared display form.
    afull = attr_full_rels or {}
    for carrier, attrs in (records_attrs or {}).items():
        for key, values in (attrs or {}).items():
            for v in (values or ()):
                v = str(v)
                tid = (carrier, key, v)
                if tid in seen_tids:
                    continue
                seen_tids.add(tid)
                # candidate walked facts under this short key: join by the
                # exact full relation when one was walked, else create one
                # fact per complete relation the case graph carries.
                walked = list(walk_join.get((carrier, key, v)) or ())
                fulls = list(afull.get((carrier, key, v)) or [])
                mat = list(walked)
                for full in fulls:
                    if any(f.rel_id == full for f in mat):
                        continue               # already materialized/walked
                    fk = (carrier, full, v)
                    if fk in fact_by_key:
                        mat.append(fact_by_key[fk])
                    else:
                        nf = SourceFact(
                            h_id=carrier, rel_id=full, t_id=v,
                            h_kind=_ek(carrier), t_kind=_ek(v),
                            t_is_literal=(_ek(v) == "LITERAL"),
                            source="case_attr", display_key=key,
                            family_key=(frozenset((carrier, v)),
                                        inv.get(full, full)))
                        fact_by_key[fk] = nf
                        facts.append(nf)
                        mat.append(nf)
                if not mat:
                    # degraded: no full relation recoverable and nothing
                    # walked — keep the row (lossless) with the short key AS
                    # the identity and flag it (never hit when the caller
                    # passes the case-graph map).
                    fk = (carrier, key, v)
                    if fk in fact_by_key:
                        f = fact_by_key[fk]
                    else:
                        f = SourceFact(
                            h_id=carrier, rel_id=key, t_id=v,
                            h_kind=_ek(carrier), t_kind=_ek(v),
                            t_is_literal=(_ek(v) == "LITERAL"),
                            source="records_unresolved", display_key="",
                            family_key=(frozenset((carrier, v)), key))
                        fact_by_key[fk] = f
                        facts.append(f)
                    mat = [f]
                for f in mat:
                    if key and not f.display_key:
                        f.display_key = key   # same fact, attr display form
                    legacy_tid_facts.setdefault(tid, []).append(f.fact_key)
                    lg = legacy_by_tid.get(tid)
                    observations.append(Observation(
                        attempt_id=int(attempt_id), call_id=call_id,
                        seq=int(seq), fact_key=f.fact_key, tree_id=canon_fid,
                        lane="attr", role="attr",
                        origin=(lg.origin if lg else "bridge"),
                        rel_display=(key if f.display_key == key else f.rel_id),
                        pattern_id=None, chain_pos=None, chain_len=None,
                        traversal_dir="unknown",
                        order_key=("attr", carrier, key, v),
                        origin_call_id=call_id))

    # V3-02 canonical is_new: ONE truth per fact over the canonical fact
    # key (the priors fold fact keys); observations and the legacy
    # projection rows both read it, so walked↔bracket re-sights of one
    # fact can never double-count a NEW delivery.
    _prior_ep = set(prior_episode or ())
    _prior_tree = set(prior_tree or ())
    _flags_by_fact: Dict[tuple, Tuple[bool, bool, bool]] = {}
    for f in facts:
        _flags_by_fact[f.fact_key] = (
            f.fact_key not in _prior_ep,
            f.fact_key not in _prior_tree,
            f.fact_key in _prior_ep)
    for ob in observations:
        fl = _flags_by_fact.get(ob.fact_key)
        if fl is not None:
            ob.new_to_episode, ob.new_to_tree, ob.redisplayed = fl
    for tr in rec.triples:
        for fk in set(legacy_tid_facts.get((tr.h, tr.rel, tr.t), ())):
            fl = _flags_by_fact.get(fk)
            if fl is not None:
                tr.new_to_episode, tr.new_to_tree, tr.redisplayed = fl
                break

    # R07 frontier: FULL structural ids + the ≤50 named preview face and
    # its truncation counts; completeness split into retrieval / ledger /
    # display dimensions (the legacy completeness dict mixes them — it stays
    # untouched on the projection). V3-04: the attr lane's retrieval-
    # contract exclusions (≥60-char values, never materialized) ride the
    # retrieval dimension — countable, and PARTIAL-of-contract when >0.
    preview = list(frontier_names or [])
    total = len(preview) if frontier_total is None else int(frontier_total)
    dropped = max(0, total - len(preview))
    _attr_long = int((attr_stats or {}).get("attr_values_excluded_long", 0) or 0)
    _truncs = int(truncations or 0)
    _canon_ids = sorted({f.fact_key for f in facts})
    commit = CallCommit(
        attempt_id=int(attempt_id), episode_id=ep_id, seq=int(seq),
        canon_fid=canon_fid, center=center,
        centers=[str(x) for x in centers] if centers else
                ([center] if center else []),
        submitted_rels=list(rec.submitted_rels),
        submitted_rel_idxs=list(rec.submitted_rel_idxs),
        ids=_canon_ids, fact_keys=list(_canon_ids),
        observations=observations,
        completeness={
            "retrieval": {"status": "PARTIAL" if (_truncs or _attr_long)
                          else "COMPLETE",
                          "rebuild_truncations": _truncs,
                          **({"attr_values_excluded_long": _attr_long}
                             if _attr_long else {})},
            "ledger": {"status": "PARTIAL" if dropped else "COMPLETE",
                       "frontier_total": total, "frontier_dropped": dropped},
            "display": {"status": "PARTIAL" if render_lines_dropped
                        else "COMPLETE",
                        "render_lines_dropped": int(render_lines_dropped or 0)}},
        frontier_node_ids=[int(i) for i in (frontier_node_ids or ())],
        frontier_preview=preview, frontier_total=total,
        frontier_dropped=dropped,
        pool_delta=None, tool_message_idx=tool_message_idx, legacy=rec)
    return LedgerDraft(facts=facts, observations=observations, commit=commit,
                       legacy=rec, ids=_canon_ids)
