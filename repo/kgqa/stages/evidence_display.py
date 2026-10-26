"""Evidence presentation layer: pattern-path overview + relation table blocks.

2026-08-23 user ruling (specs/rendering_mechanism_design.md, "呈现层" section):
the model first sees a coarse INDEX of the subgraph's pattern paths, then the
evidence details as ONE BLOCK PER RELATION instead of the unified line
grammar's inline ``h --rel--> t1 | t2 | ...`` rows.

Phase-2 corrections (2026-08-23 user review of render_v2_g3, Mandela case):
  C1 record blocks are PATH TERMINALS — a CVT record is never walked THROUGH;
      its attributes are terminal values, not parallel entity columns. Single
      record renders as ``m.xxx [k=v; k=v]``; multi-record keeps the attr-
      column table but the block TITLE marks terminality.
  C2 sibling-row folding — the unified line's many-to-one semantics return to
      the table: rows of one edge block that share the same tail-SET merge
      into one row (join side becomes a multi-entity cell), and one head with
      many tails folds its tails into one cell (per _merge_edges semantics).
  C3 multi-hop chains render as PATHS — relations that stitch through
      pass-through nodes (in-degree 1, out-degree 1, non-record) become one
      chain block: column headers are the hop relation sequence, each row is
      one concrete instance chain (start | middle | ... | terminal). When in
      doubt the stitch is NOT made (保守不串).

SEPARATOR HIERARCHY (hard constraint, user ruling 2026-08-23):
  L1 ``|``        hop-position/record layer: column separation in table and
                  chain rows; overview tail enumeration ``--> t1 | t2``.
                  NEVER inside a cell.
  L2 ``(a; b; c)`` in-cell entity enumeration (folded cells, chain fanout
                  cells, multi-value attr cells) — parens wrap when >1.
  L3 ``[k=v; k=v]`` record attribute brackets (existing grammar, unchanged).
  Fallback: an entity containing ``;`` ``(`` or ``)`` is never joined into an
  L2 cell — edge rows stay unfolded; attr cells degrade to first value plus
  an exact ``…+K more`` marker. Hierarchy clarity beats compactness.

Relationship to the S0-S4 pipeline (unchanged internals):
  S0  direction canonicalization stays at the CALLER (_canonicalize_triples in
      seq_tools) — this module is a PURE function of the triples it receives
      (no ctx access, zero cross-call state, I4);
  S1  eventize: every CVT becomes exactly one record object {id, attrs};
  S2  content-key dedup (edges by (h,rel,t); records by (head, signature));
  S2b chain stitch: pass-through named edges chain into path groups (C3);
  S3  group: leftover named edges by (head, rel); records by (head, rel);
  S4  compress (L1 uniform-attr hoist / L2 single-print / L3 budget caps with
      explicit markers) and SYNTHESIZE as overview + blocks.

Budget law (L3) table semantics: per-block row cap, per-cell entity cap,
block-count cap, overview cap, value truncation — every truncation carries an
explicit marker (I5). candidates / n_candidates / note are assembled VERBATIM
by the caller; ``render_records_compat`` is a drop-in replacement for
seq_tools._render_records (same signature, same (lines, n_overlap) contract).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from kgqa.core.utils import normalize
from kgqa.traversal.cvt import is_cvt_like

# ── noise rules ─────────────────────────────────────────────────────────────
# Keep in sync with seq_tools._EDGE_NOISY_SHORT (bookkeeping + hub-quartet
# edges). Duplicated locally: importing kgqa.agent.seq_tools from kgqa.stages
# would close a stages→agent→stages import cycle.
_EDGE_NOISY_SHORT = {
    "type", "types", "instance", "instances", "notable_types", "notable_type",
    "webpage", "mid", "guid", "key", "keys", "permission",
    "article", "description", "is_reviewed", "image", "webpage_topic",
    "topic_equivalent_webpage", "alias", "name", "descriptive_name",
    "member", "organization", "role", "appointees",
}
# Attribute-key noise (record columns) — same classes the production record
# renderer filters (type./common. bookkeeping, is_* flags). has_no_value is
# handled separately (incumbent marker), never as a column of its own.
_ATTR_NOISY_PREFIX = ("type.", "common.", "kg.", "user.", "base.ontologies.",
                      "freebase.")
_ATTR_NOISY_SHORT = _EDGE_NOISY_SHORT
_DATE_ATTR_KEYS = {"from", "to", "date", "year", "initial_date", "final_date",
                   "valid_date", "valid_from", "valid_to", "start_date",
                   "end_date"}

# budget defaults (L3 — correction 8: integrity first, economy only on
# extreme hubs; 120 is the capacity the OLD stack ran all history on)
ROW_CAP = 40          # folded rows per block (line-count control)
BLOCK_CAP = 12        # relation blocks per render
OVERVIEW_CAP = 20     # pattern-path overview lines
CELL_CAP = 120        # single value display cap — ABOVE the longest real
                     # entity names: named entities are ANSWER CANDIDATES and
                     # must survive untruncated (Gingrich's 54-char title);
                     # the cap only marks pathological non-entity garbage
CELL_LIST_CAP = 120   # entities per L2 multi-entity cell — the OLD pipeline's
                     # _MERGE_TAIL_CAP level: ≤120 entities ALWAYS fully
                     # visible; only >120 truncates, and then WITH a branch
                     # ref (C5). No …+N more / hidden below 120 (C8).
ATTR_LINE_CAP = 120   # attrs on a single-record terminal line (same principle)
MAX_CHAIN_HOPS = 3    # conservative chain-stitch ceiling (C3)

_INCUMBENT = "(incumbent)"
_ELLIPSIS = "…"
_RULE_MIN_WIDTH = 48
_TERMINAL_TAG = " (terminal records)"
# branch-ref compressed submission (C5, restored): when a budget cap hides
# answer candidates, emit a `#anchor::relation` ref the answer layer expands
# into the FULL list (tools._expand_branch_refs / _do_answer). Keep the token
# grammar in sync with tools._BRANCH_RE = ^#(.{1,200}?)(?:::|\|)(.{1,300})$
# — '::' is the live separator; expansion matches rel by full/suffix/short
# name and the anchor on EITHER edge side (so both fold directions are legal).
_REF_INSTR = ('To answer with ALL of them, include "#{ref}" as one answer '
              'entity — the system expands the ref to the full list.')


def _ref_safe(anchor: str) -> bool:
    s = str(anchor)
    return bool(s) and "#" not in s and "::" not in s and "|" not in s \
        and len(s) <= 200 and _cell_safe(s)


def _short_rel(r: str) -> str:
    r = str(r or "")
    return r.rsplit(".", 1)[-1] if r else r


def _attr_noisy(rel_full: str, short: str) -> bool:
    return rel_full.startswith(_ATTR_NOISY_PREFIX) or short in _ATTR_NOISY_SHORT


def _fmt_val(key: str, v) -> str:
    """Format a single value. Dates cut to 10 chars, long values capped — real
    truncation is MARKED with an ellipsis (I5); cutting the time suffix off an
    ISO timestamp is date FORMATTING, not truncation (no marker)."""
    s = str(v)
    if re.fullmatch(r"-?\d+\.\d{4,}", s):       # measurement round-off (display only)
        s = str(round(float(s), 2))
    if key and key.lower() in _DATE_ATTR_KEYS:
        if re.match(r"^\d{4}-\d{2}-\d{2}", s):
            return s[:10]
        if len(s) > CELL_CAP:
            return s[:CELL_CAP - 1] + _ELLIPSIS
        return s
    if len(s) > CELL_CAP:
        return s[:CELL_CAP - 1] + _ELLIPSIS
    return s


def _year_key(s: str):
    m = re.match(r"(\d{4})", str(s or ""))
    return int(m.group(1)) if m else 9999


# ── separator hierarchy helpers (L1 | / L2 (;) / L3 [k=v]) ─────────────────

def _cell_safe(e) -> bool:
    """An entity that may live inside an L2 cell. Names containing the L2/L3
    delimiters are never joined (user fallback ruling: hierarchy clarity
    beats compactness)."""
    s = str(e)
    return not any(ch in s for ch in ";()" )


def _join_cell(entities: Sequence, cap: int):
    """L2 cell: 1 entity → bare; >1 → parens + '; '. Returns None when any
    entity is unsafe (caller falls back to unfolded rows). Hidden beyond cap
    marked exactly: ``(a; b; …+K more)`` (I5). Returns
    (text, hidden, shown_entities)."""
    ents = [str(e) for e in entities]
    if any(not _cell_safe(e) for e in ents):
        return None
    shown = [_fmt_val("", e) for e in ents[:cap]]   # display-faithful
    hidden = max(0, len(ents) - cap)
    parts = [_fmt_val("", e) for e in shown]
    if hidden > 0:
        parts.append(f"…+{hidden} more")
    inner = "; ".join(parts)
    return (f"({inner})" if len(ents) > 1 else inner), hidden, shown


def _attr_cell(key: str, values: List, cap: int) -> Tuple[str, int]:
    """Attr cell under the hierarchy: 1 value → bare; >1 all-safe → L2 parens;
    >1 with unsafe values → first value + exact ``…+K more`` marker."""
    if len(values) == 1:
        return _fmt_val(key, values[0]), 0
    jc = _join_cell(values, cap)
    if jc is not None:
        return jc[0], jc[1]
    hidden = len(values) - 1
    return _fmt_val(key, values[0]) + f" …+{hidden} more", hidden


# ── view model (S1-S3 output) ───────────────────────────────────────────────

class RecordRow:
    __slots__ = ("rec_id", "head", "attrs", "ref_only")

    def __init__(self, rec_id, head, attrs, ref_only=False):
        self.rec_id = rec_id          # CVT id (m.xxx / g.xxx)
        self.head = head              # named head the record hangs off
        self.attrs = attrs            # ordered [(key, [values])] non-noisy
        self.ref_only = ref_only      # hub-CVT later appearance: bare id (L2 单次打印)

    def signature(self) -> tuple:
        return (normalize(self.head),
                tuple(sorted((k, tuple(normalize(str(v)) for v in vals))
                             for k, vals in self.attrs)))


class _Group:
    """One S3 group: (head, relation) → payloads (tails or records)."""
    __slots__ = ("head", "rel_full", "rel_short", "tails", "rows")

    def __init__(self, head, rel_full, kind):
        self.head = head
        self.rel_full = rel_full
        self.rel_short = _short_rel(rel_full) or rel_full or "(relation)"
        self.tails: List[str] = []       # kind == "edge"
        self.rows: List[RecordRow] = []  # kind == "record"


class ChainGroup:
    """C3 chain group: all instance chains sharing (start, rel sequence, end).
    ``middles[i]`` is the list of i-th-position middle entities across the
    parallel chains (fanout merges into ONE L2 cell per position)."""
    __slots__ = ("start", "rels", "end", "middles", "n_chains")

    def __init__(self, start, rels, end, middles, n_chains):
        self.start = start
        self.rels = rels                # full relation names, hop order
        self.end = end
        self.middles = middles          # [ [mid-entity per position] per chain ]
        self.n_chains = n_chains


class Block:
    """S4 block: one relation (or one chain relation-sequence) + its groups."""
    __slots__ = ("rel_full", "label", "edge_groups", "record_groups",
                 "chain_groups")

    def __init__(self, rel_full, label):
        self.rel_full = rel_full
        self.label = label               # display label (short name(s), deduped)
        self.edge_groups: List[_Group] = []
        self.record_groups: List[_Group] = []
        self.chain_groups: List[ChainGroup] = []

    @property
    def n_rows_total(self) -> int:
        return (sum(len(g.tails) for g in self.edge_groups)
                + sum(len(g.rows) for g in self.record_groups)
                + sum(c.n_chains for c in self.chain_groups))

    @property
    def n_entities_total(self) -> int:
        return (sum(len(g.tails) for g in self.edge_groups)
                + sum(len(g.rows) for g in self.record_groups)
                + sum(c.n_chains * (len(c.rels) + 1) for c in self.chain_groups))

    @property
    def n_groups(self) -> int:
        return (len(self.edge_groups) + len(self.record_groups)
                + len(self.chain_groups))


class EvidenceView:
    def __init__(self):
        self.blocks: List[Block] = []
        self.edge_groups: List[_Group] = []
        self.record_groups: List[_Group] = []
        self.chain_groups: List[ChainGroup] = []
        # C4/C6 pattern-level overview: the call's center entities (normalized
        # members) + the display label. Multi-center (?var expansion) walks
        # anchor patterns at the SET, never per member.
        self.anchors: set = set()
        self.anchor_members: List[str] = []   # display order
        self.anchor_label: Optional[str] = None   # e.g. "?film" (var token)
        # C-raw: the walk's REAL pattern structures (PatternEvidence
        # tree_data paths: {"nodes": [center, ...], "relations": [...]}).
        # C11: shapes derived from them are the SINGLE SOURCE OF TRUTH for
        # the overview AND the detail blocks; flattened edges only SUPPLEMENT
        # what the shapes do not cover.
        self.raw_paths: List[dict] = []
        self.shape_groups: List[dict] = []   # parsed shapes (see build_view)
        self.covered_edges: set = set()      # (norm_h, rel_full, norm_t)
        self.record_attrs: Dict[str, List[Tuple[str, List[str]]]] = {}
        self.dropped = {"noisy_edges": 0, "self_loops": 0, "cvt_cvt": 0,
                        "bare_cvts": 0, "dup_edges": 0, "dup_records": 0,
                        "cycles_not_stitched": 0}

    @property
    def set_mode(self) -> bool:
        return len(self.anchors) > 1

    @property
    def anchor_header(self) -> str:
        """Column-header text for the anchor slot (C7)."""
        if not self.anchors:
            return "?node"
        if len(self.anchors) == 1:
            return self.anchor_members[0] if self.anchor_members else "?anchor"
        if self.anchor_label:
            return f"{self.anchor_label}({len(self.anchors)} centers)"
        first = self.anchor_members[0] if self.anchor_members else "?"
        return f"({first}; +{len(self.anchors) - 1} more)"

    @property
    def anchor_display(self) -> str:
        """Pattern-line start symbol (C6)."""
        if not self.anchors:
            return "?node"
        if len(self.anchors) == 1:
            return self.anchor_members[0] if self.anchor_members else "?anchor"
        if self.anchor_label:
            return self.anchor_label
        first = self.anchor_members[0] if self.anchor_members else "?"
        return f"({first}; +{len(self.anchors) - 1} more)"

    @property
    def n_groups(self) -> int:
        return (len(self.edge_groups) + len(self.record_groups)
                + len(self.chain_groups))


# ── S0-S3 ───────────────────────────────────────────────────────────────────

def _stitch_chains(edges: List[Tuple[str, str, str]],
                   record_heads_norm: set) -> Tuple[List[ChainGroup], set]:
    """C3: chain pass-through named edges into path groups. A middle node is
    stitched only when it is UNAMBIGUOUS: exactly one incoming and one
    outgoing edge in the deduped named-edge set, never a record head, never a
    self-loop, chain length ≤ MAX_CHAIN_HOPS, no revisits (a node revisit
    stops the walk; unabsorbed cycle members stay single-hop — 保守不串).
    Returns (chain groups, consumed edge indexes)."""
    out_map: Dict[str, List[int]] = {}      # head -> edge indexes
    in_map: Dict[str, List[int]] = {}       # tail -> edge indexes
    for i, (h, _r, t) in enumerate(edges):
        out_map.setdefault(h, []).append(i)
        in_map.setdefault(t, []).append(i)

    def _pass_through(n: str) -> Optional[int]:
        if normalize(n) in record_heads_norm:
            return None
        outs, ins = out_map.get(n, ()), in_map.get(n, ())
        if len(outs) != 1 or len(ins) != 1:
            return None
        i = outs[0]
        if edges[i][2] == n:                # defensive self-loop
            return None
        return i

    consumed: set = set()
    chains: List[ChainGroup] = []
    key_order: Dict[tuple, int] = {}
    for i0 in range(len(edges)):
        if i0 in consumed:
            continue
        a, r1, _b = edges[i0]
        if _pass_through(a) is not None:
            continue                        # a is a middle — its head's chain absorbs it
        used, nodes, rels, visited, cur = [i0], [a], [r1], {a}, i0
        while len(rels) < MAX_CHAIN_HOPS:
            nxt = edges[cur][2]
            if nxt in visited:
                break
            ni = _pass_through(nxt)
            if ni is None or ni in consumed:
                break
            visited.add(nxt)
            used.append(ni)
            nodes.append(nxt)
            rels.append(edges[ni][1])
            cur = ni
        if len(rels) < 2:
            continue                        # single hop — not a chain
        end = edges[cur][2]
        for u in used:
            consumed.add(u)
        key = (a, tuple(rels), end)
        mids = nodes[1:]                    # one middle per stitched hop
        if key in key_order:
            cg = chains[key_order[key]]
            cg.middles.append(mids)
            cg.n_chains += 1
        else:
            key_order[key] = len(chains)
            chains.append(ChainGroup(a, list(rels), end, [mids], 1))
    return chains, consumed


def build_view(triples: Sequence, row_cap: int = ROW_CAP,
               block_cap: int = BLOCK_CAP,
               overview_cap: int = OVERVIEW_CAP,
               cell_list_cap: int = CELL_LIST_CAP,
               anchors: Sequence[str] = (),
               anchor_label: Optional[str] = None,
               patterns: Sequence[dict] = ()) -> EvidenceView:
    """Grouped evidence structure from canonicalized (h, r, t) triples.

    ``anchors`` (C4/C6) = the retrieve_subgraph call's center entities —
    multi-center (?var-expanded) walks anchor overview patterns at the SET.
    ``anchor_label`` = the ?var token when the raw center was a variable
    (rendered ``?film`` in patterns / ``?film(43 centers)`` in headers).
    Without anchors, a structural fallback merges groups that share one tail
    into reverse patterns (≥2 groups, singleton tail-set).

    Pure: never mutates ``triples``; no cross-call state (I4)."""
    # S1 eventize — one pass over the triples:
    cvt_attrs: Dict[str, List[Tuple[str, str]]] = {}
    cvt_incumbent: Dict[str, set] = {}
    rec_edges: List[Tuple[str, str, str]] = []      # (head, rel_full, cvt_id)
    named_edges: List[Tuple[str, str, str]] = []    # (h, rel_full, t)
    view = EvidenceView()
    for tr in triples or []:
        if not (isinstance(tr, (tuple, list)) and len(tr) == 3):
            continue
        h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
        short = _short_rel(r)
        if is_cvt_like(h):
            if is_cvt_like(t):
                view.dropped["cvt_cvt"] += 1
                continue
            if "has_no_value" in r:
                # value NAMES the absent attribute ("To" → to=(incumbent))
                key = str(t).strip().lower()
                if key:
                    cvt_incumbent.setdefault(h, set()).add(key)
                continue
            if _attr_noisy(r, short):
                continue
            lst = cvt_attrs.setdefault(h, [])
            if (short, t) not in lst:
                lst.append((short, t))
            continue
        if is_cvt_like(t):
            rec_edges.append((h, r, t))
            continue
        # named ↔ named
        if short in _EDGE_NOISY_SHORT or _attr_noisy(r, short):
            # (production's short-name set + the pool's bookkeeping prefixes —
            # the pool filter stops type./common. relations from being WALKED;
            # the display applies the same classes defensively)
            view.dropped["noisy_edges"] += 1
            continue
        if normalize(h) == normalize(t):
            view.dropped["self_loops"] += 1
            continue
        named_edges.append((h, r, t))

    # S2 content-key dedup
    seen_e, edges = set(), []
    for h, r, t in named_edges:
        k = (normalize(h), r, normalize(t))
        if k in seen_e:
            view.dropped["dup_edges"] += 1
            continue
        seen_e.add(k)
        edges.append((h, r, t))
    seen_r, records = set(), []
    for h, r, cvt in rec_edges:
        k = (normalize(h), r, cvt)
        if k in seen_r:
            view.dropped["dup_records"] += 1
            continue
        seen_r.add(k)
        records.append((h, r, cvt))

    # record objects; attrs grouped by key (multi-value keys keep all values
    # for the L2 cell form); content dedup; zero-attr CVTs dropped (Kim)
    row_by_cvt: Dict[str, RecordRow] = {}
    rec_sigs = set()
    record_heads = {normalize(h) for h, _r, _c in records}
    _kv_by_cvt: Dict[str, List[Tuple[str, List[str]]]] = {}
    for h, r, cvt in records:
        kv: Dict[str, List[str]] = {}
        for k, v in cvt_attrs.get(cvt, []):
            kv.setdefault(k, [])
            if v not in kv[k]:
                kv[k].append(v)
        for key in sorted(cvt_incumbent.get(cvt, ())):
            if key not in kv:
                kv[key] = [_INCUMBENT]
        if not kv:
            view.dropped["bare_cvts"] += 1
            continue
        attrs = list(kv.items())
        if cvt not in _kv_by_cvt:
            _kv_by_cvt[cvt] = attrs
        row = RecordRow(cvt, h, attrs)
        sig = ("REC", row.signature())
        if sig in rec_sigs:
            view.dropped["dup_records"] += 1
            continue
        rec_sigs.add(sig)
        row_by_cvt.setdefault(cvt, row)
    view.record_attrs = _kv_by_cvt

    # ── C11: parse the walk's REAL pattern shapes FIRST (single source of
    # truth); their hop edges become COVERED — the flattened edge set keeps
    # only the supplement, and chains stitch on the leftovers.
    view.raw_paths = [pp for pp in (patterns or [])
                      if isinstance(pp, dict) and pp.get("nodes")
                      and pp.get("relations")]
    _anchor_norm_pre = {normalize(a) for a in (anchors or ()) if a}

    def _pid(n) -> str:
        return str(n).split(":")[0].strip()

    def _mem_pre(x) -> bool:
        return normalize(x) in _anchor_norm_pre

    _fwd_edges = {(normalize(h), r, normalize(t)) for h, r, t in edges}

    def _dir_for(a, b, rf):
        if (normalize(a), rf, normalize(b)) in _fwd_edges:
            return "f"
        if (normalize(b), rf, normalize(a)) in _fwd_edges:
            return "r"
        return "f"

    shape_order: Dict[tuple, dict] = {}
    for path in view.raw_paths:
        nodes, rels = path["nodes"], path["relations"]
        if len(rels) < 1 or len(nodes) < 2:
            continue
        pids = [_pid(n) for n in nodes]
        start = pids[0]
        _mem_list = [a for a in (anchors or ()) if a]
        if _anchor_norm_pre:
            if not _mem_pre(start):
                continue              # C9: only walk starts anchor shapes
            if len(_anchor_norm_pre) > 1:
                disp = (str(anchor_label) if anchor_label else
                        f"({_mem_list[0]}; +{len(_anchor_norm_pre) - 1} more)")
            else:
                disp = _mem_list[0] if _mem_list else start
        else:
            disp = start
        rels_full = [str(r) for r in rels]
        hops = []
        dirs = []
        for i, rf in enumerate(rels_full):
            d = _dir_for(pids[i], pids[i + 1], rf)
            dirs.append(d)
            hops.append(((_short_rel(rf) or rf), d))
        terminal = "records" if is_cvt_like(pids[-1]) else None
        key = (disp, tuple(hops), terminal)
        grp = shape_order.get(key)
        if grp is None:
            grp = {"disp": disp, "hops": hops, "hops_full": rels_full,
                   "dirs": dirs, "terminal": terminal, "paths": [],
                   "starts": []}
            shape_order[key] = grp
        grp["paths"].append(path)
        grp["starts"].append(start)
        for i, rf in enumerate(rels_full):
            view.covered_edges.add((normalize(pids[i]), rf, normalize(pids[i + 1])))
    view.shape_groups = list(shape_order.values())

    # supplement = deduped named edges AND record edges NOT covered by any
    # shape (C11 rule 2: flattened data only fills the shapes' gaps)
    edges = [e for e in edges
             if (normalize(e[0]), e[1], normalize(e[2])) not in view.covered_edges]
    records = [rc for rc in records
               if (normalize(rc[0]), rc[1], normalize(rc[2]))
               not in view.covered_edges]

    # S2b C3 chain stitch over the SUPPLEMENT edges
    chains, consumed = _stitch_chains(edges, record_heads)
    leftover = [e for i, e in enumerate(edges) if i not in consumed]

    # S3 group by (head, relation); blocks keyed by relation (chain blocks by
    # their relation sequence), placed at the first constituent's arrival
    e_groups: Dict[Tuple[str, str], _Group] = {}
    r_groups: Dict[Tuple[str, str], _Group] = {}
    blocks: Dict[str, Block] = {}
    used_labels: set = set()

    def _block(rel_full: str) -> Block:
        b = blocks.get(rel_full)
        if b is None:
            short = _short_rel(rel_full) or rel_full or "(relation)"
            label = short if short not in used_labels else rel_full
            used_labels.add(label)
            b = Block(rel_full, label)
            blocks[rel_full] = b
        return b

    for h, r, t in leftover:
        g = e_groups.get((normalize(h), r))
        if g is None:
            g = _Group(h, r, "edge")
            e_groups[(normalize(h), r)] = g
            view.edge_groups.append(g)
            _block(r).edge_groups.append(g)
        if t not in g.tails:
            g.tails.append(t)
    detailed: set = set()
    for h, r, cvt in records:
        row = row_by_cvt.get(cvt)
        if row is None:
            continue                       # bare CVT: dropped before grouping
        g = r_groups.get((normalize(h), r))
        if g is None:
            g = _Group(h, r, "record")
            r_groups[(normalize(h), r)] = g
            view.record_groups.append(g)
            _block(r).record_groups.append(g)
        if any(x.rec_id == cvt for x in g.rows):
            continue
        if cvt in detailed:
            # hub CVT later appearance → bare-id reference row (L2 单次打印)
            g.rows.append(RecordRow(cvt, h, [], ref_only=True))
        else:
            g.rows.append(row)
            detailed.add(cvt)
    for c in chains:
        b = _block("chain::" + "›".join(c.rels))
        if not b.chain_groups:
            seq = " → ".join((_short_rel(r) or r) for r in c.rels)
            label = f"{seq} ({len(c.rels)}-hop)"
            if label in used_labels:
                label = f"{seq} ({len(c.rels)}-hop) [{'›'.join(c.rels)}]"
            used_labels.add(label)
            b.label = label
        b.chain_groups.append(c)
        view.chain_groups.append(c)
    # ── C11: synthesize the PATTERN blocks — the single source of truth.
    # Records-terminal shapes → record groups (attrs from the CVT map);
    # 1-hop named shapes → edge groups (C2 folding applies); multi-hop
    # named shapes → chain groups (one instance chain per path). The synth
    # blocks are PREPENDED so the overview's promise is delivered first;
    # the (coverage-filtered) supplement blocks follow.
    synth: List[Block] = []
    detailed_recs: set = set()
    for grp in view.shape_groups:
        rel0 = grp["hops_full"][0]
        seq = "›".join(grp["hops_full"])
        inst = []
        seen_inst: set = set()
        for path, st in zip(grp["paths"], grp["starts"]):
            pids_all = [_pid(n) for n in path["nodes"]]
            sig = (st, tuple(pids_all))
            if sig not in seen_inst:
                seen_inst.add(sig)
                inst.append((st, pids_all))
        grp["instances"] = inst
        if grp["terminal"] == "records":
            b = Block("shape::rec::" + seq, _short_rel(rel0) or rel0)
            groups_by_start: Dict[str, _Group] = {}
            for st, pids_all in inst:
                rid = pids_all[-1]
                g = groups_by_start.get(st)
                if g is None:
                    g = _Group(st, rel0, "record")
                    groups_by_start[st] = g
                    b.record_groups.append(g)
                if rid in detailed_recs:
                    g.rows.append(RecordRow(rid, st, [], ref_only=True))
                else:
                    g.rows.append(RecordRow(rid, st,
                                            list(_kv_by_cvt.get(rid, []))))
                    detailed_recs.add(rid)
            if b.record_groups:
                synth.append(b)
                grp["block"] = b
        elif len(grp["hops_full"]) == 1:
            b = Block("shape::edge::" + seq, _short_rel(rel0) or rel0)
            groups_by_start = {}
            for st, pids_all in inst:
                g = groups_by_start.get(st)
                if g is None:
                    g = _Group(st, rel0, "edge")
                    groups_by_start[st] = g
                    b.edge_groups.append(g)
                if pids_all[-1] not in g.tails:
                    g.tails.append(pids_all[-1])
            if b.edge_groups:
                synth.append(b)
                grp["block"] = b
        else:
            b = Block("shape::chain::" + seq,
                      " → ".join((_short_rel(r) or r)
                                 for r in grp["hops_full"]))
            cg: Dict[tuple, ChainGroup] = {}
            for st, pids_all in inst:
                end = pids_all[-1]
                mids = pids_all[1:-1]
                key = (st, tuple(grp["hops_full"]), end)
                if key in cg:
                    cg[key].middles.append(mids)
                    cg[key].n_chains += 1
                else:
                    cg[key] = ChainGroup(st, list(grp["hops_full"]), end,
                                         [mids], 1)
            b.chain_groups = list(cg.values())
            if b.chain_groups:
                synth.append(b)
                grp["block"] = b

    # ── C11 merge: 1-hop shapes absorb the UNCOVERED flattened instances of
    # the same (member head, relation) — the pattern block then carries the
    # TRUE instance count (tree_data paths are bounded witnesses; the
    # flattened set holds the rest). Multi-hop leftovers stay in supplement.
    for grp in view.shape_groups:
        rel0 = grp["hops_full"][0]
        b = grp.get("block")
        if b is None or len(grp["hops_full"]) != 1:
            continue
        if grp["terminal"] == "records":
            for h, r, cvt in list(records):
                if r != rel0 or not _mem_pre(h):
                    continue
                if (normalize(h), r, normalize(cvt)) in view.covered_edges:
                    continue
                view.covered_edges.add((normalize(h), r, normalize(cvt)))
                g = next((x for x in b.record_groups
                          if normalize(x.head) == normalize(h)), None)
                if g is None:
                    g = _Group(h, rel0, "record")
                    b.record_groups.append(g)
                if any(x.rec_id == cvt for x in g.rows):
                    continue
                if cvt in detailed_recs:
                    g.rows.append(RecordRow(cvt, h, [], ref_only=True))
                else:
                    g.rows.append(RecordRow(cvt, h, list(_kv_by_cvt.get(cvt, []))))
                    detailed_recs.add(cvt)
            grp["instances"] = [(g.head, [g.head, row.rec_id])
                                for g in b.record_groups for row in g.rows]
        else:
            for h, r, t in list(edges):
                if r != rel0 or not _mem_pre(h):
                    continue
                if (normalize(h), r, normalize(t)) in view.covered_edges:
                    continue
                view.covered_edges.add((normalize(h), r, normalize(t)))
                g = next((x for x in b.edge_groups
                          if normalize(x.head) == normalize(h)), None)
                if g is None:
                    g = _Group(h, rel0, "edge")
                    b.edge_groups.append(g)
                if t not in g.tails:
                    g.tails.append(t)
            grp["instances"] = [(g.head, [g.head, t])
                                for g in b.edge_groups for t in g.tails]

    # supplement record rows demote to bare-id refs once their attrs have a
    # pattern-block home (I2: attr set printed exactly once across blocks)
    for blk in blocks.values():
        for g in blk.record_groups:
            g.rows = [r if r.rec_id not in detailed_recs
                      else RecordRow(r.rec_id, r.head, [], ref_only=True)
                      for r in g.rows]

    view.blocks = synth + list(blocks.values())
    view.anchor_members = [a for a in (anchors or ()) if a]
    view.anchors = {normalize(a) for a in view.anchor_members if a}
    view.anchor_label = anchor_label
    view.raw_paths = [p for p in (patterns or ())
                      if isinstance(p, dict) and p.get("nodes")
                      and p.get("relations")]
    return view


# ── S4: synthesis (overview + blocks) ───────────────────────────────────────

_SLOTS = ["?x", "?y", "?z", "?n4", "?n5", "?n6"]


class _Pattern:
    """C4/C6 overview pattern: one MODE SHAPE per line. Only the start is an
    entity (or the SET symbol for multi-center walks); every later node is a
    placeholder slot. ``hops`` = [(rel_short, 'f'|'r')], ``terminal`` = None
    (edge pattern, slot tail) | 'records' | 'chains'."""
    __slots__ = ("start", "hops", "terminal", "count", "block", "from_shape")

    def __init__(self, start, rel, direction, terminal=None, count=0,
                 block=None, from_shape=False):
        self.start = start
        self.hops = [((_short_rel(rel) or rel), direction)]
        self.terminal = terminal
        self.count = count
        self.block = block
        self.from_shape = from_shape

    def _body(self, generic_terminal=False):
        out = f"{self.start}"
        n_hops = len(self.hops)
        for i, (rel, d) in enumerate(self.hops):
            slot = _SLOTS[i] if i < len(_SLOTS) else "?n" + str(i)
            if self.terminal == "records" and i == n_hops - 1:
                slot = "[records]" if generic_terminal else \
                    f"[{self.count} record{'s' if self.count != 1 else ''}]"
            out += (f" --{rel}--> {slot}" if d == "f"
                    else f" <--{rel}-- {slot}")
        return out

    def render(self) -> str:
        out = self._body()
        if self.terminal == "chains":
            if self.count > 1:
                out += f" ({self.count} chains)"
        elif self.terminal != "records" and self.count > 1:
            out += f" ({self.count} instances)"
        return "  " + out

    def shape(self) -> str:
        """Block-title form: full shape, generic terminals."""
        return self._body(generic_terminal=True)


def _collect_patterns(view: EvidenceView) -> List[_Pattern]:
    """Distinct pattern SHAPES (C4/C6/C9). The overview anchors at the WALK
    START only — the call's center entity (or the SET symbol for ?var
    expansions). Any group NOT rooted at the start produces NO pattern line
    (C9 rule 1): record back-edges fold into the record's own presentation
    (their entities surface as record attribute values), boundary
    continuations live in the blocks/chain shapes. When NO anchors are known
    (anchor-less calls), the correction-4 structural fallback applies
    (per-group anchoring) — production always passes anchors."""
    pats: List[_Pattern] = []
    anchor_norm = view.anchors

    def _member(x) -> bool:
        return normalize(x) in anchor_norm

    # ── C11: pattern lines come from the parsed SHAPE GROUPS (build_view) —
    # the same structures that synthesize the detail blocks (single source
    # of truth: overview count == block instance rows by construction).
    covered: set = set()
    for grp in view.shape_groups:
        pat = _Pattern(grp["disp"], grp["hops_full"][0],
                       grp["dirs"][0] if grp["dirs"] else "f",
                       terminal=grp["terminal"],
                       count=len(grp.get("instances") or grp["paths"]),
                       block=grp.get("block"), from_shape=True)
        pat.hops = list(grp["hops"])
        pats.append(pat)
        covered.add((grp["hops"][0][0], grp["dirs"][0] if grp["dirs"] else "f",
                     grp["terminal"] or "edge"))

    # supplement blocks produce NO overview lines when real shapes exist
    # (rule 2: flattened edges are display-only 补充); the anchor-less
    # fallback keeps the correction-4 structural behavior.
    _supplement_only = bool(view.raw_paths) and bool(anchor_norm)

    for block in view.blocks:
        if _supplement_only and block.rel_full.startswith("shape::"):
            continue          # synth blocks carry no patterns of their own
        def _covered(rel_full, direction, terminal):
            return ((_short_rel(rel_full) or rel_full), direction,
                    terminal) in covered

        fwd: Dict[str, _Pattern] = {}
        rev: Dict[str, _Pattern] = {}
        parked: Dict[tuple, List[_Group]] = {}
        # ── set-anchor merges (multi-center): one pattern per (rel, dir)
        set_fwd = None
        set_rev = None
        if _supplement_only:
            pass                    # display-only supplement (C11 rule 2)
        elif view.set_mode:
            for g in block.edge_groups:
                if _member(g.head):
                    if set_fwd is None and not _covered(g.rel_full, "f", "edge"):
                        set_fwd = _Pattern(view.anchor_display, g.rel_full,
                                           "f", block=block)
                        pats.append(set_fwd)
                    if set_fwd is not None:
                        set_fwd.count += len(g.tails)
                elif any(_member(t) for t in g.tails):
                    if set_rev is None and not _covered(g.rel_full, "r", "edge"):
                        set_rev = _Pattern(view.anchor_display, g.rel_full,
                                           "r", block=block)
                        pats.append(set_rev)
                    if set_rev is not None:
                        set_rev.count += len(g.tails)
                else:
                    parked.setdefault(
                        tuple(normalize(t) for t in g.tails), []).append(g)
        else:
            for g in block.edge_groups:
                hn = normalize(g.head)
                if hn in anchor_norm:
                    if _covered(g.rel_full, "f", "edge"):
                        continue
                    p = fwd.get(hn)
                    if p is None:
                        p = _Pattern(g.head, g.rel_full, "f", block=block)
                        fwd[hn] = p
                        pats.append(p)
                    p.count += len(g.tails)
                else:
                    anch = next((t for t in g.tails
                                 if normalize(t) in anchor_norm), None)
                    if anch is not None:
                        if _covered(g.rel_full, "r", "edge"):
                            continue
                        key = normalize(anch)
                        p = rev.get(key)
                        if p is None:
                            p = _Pattern(anch, g.rel_full, "r", block=block)
                            rev[key] = p
                            pats.append(p)
                        p.count += len(g.tails)
                    else:
                        parked.setdefault(
                            tuple(normalize(t) for t in g.tails), []).append(g)
        for ts, gs in parked.items():
            if anchor_norm or _supplement_only:
                continue      # C9 rule 1: not start-rooted → no pattern line
            # anchor-less fallback (legacy): shared-tail reverse merge / local
            if len(gs) >= 2 and len(ts) == 1:
                p = _Pattern(gs[0].tails[0], gs[0].rel_full, "r", block=block)
                p.count = sum(len(g.tails) for g in gs)
                pats.append(p)
            else:
                for g in gs:
                    p = _Pattern(g.head, g.rel_full, "f", block=block)
                    p.count = len(g.tails)
                    pats.append(p)
        # ── record groups: start-rooted only (C9 rule 2 — back-edge heads
        # fold into the record presentation, no pattern line of their own)
        if _supplement_only:
            pass
        elif view.set_mode:
            set_rec = None
            for g in block.record_groups:
                if _member(g.head):
                    if set_rec is None and not _covered(g.rel_full, "f",
                                                        "records"):
                        set_rec = _Pattern(view.anchor_display, g.rel_full,
                                           "f", terminal="records", count=0,
                                           block=block)
                        pats.append(set_rec)
                    if set_rec is not None:
                        set_rec.count += len(g.rows)
                elif not anchor_norm:
                    p = _Pattern(g.head, g.rel_full, "f", terminal="records",
                                 count=len(g.rows), block=block)
                    pats.append(p)
        else:
            for g in block.record_groups:
                if _member(g.head) or not anchor_norm:
                    if anchor_norm and _covered(g.rel_full, "f", "records"):
                        continue
                    p = _Pattern(g.head, g.rel_full, "f", terminal="records",
                                 count=len(g.rows), block=block)
                    pats.append(p)
        # ── chains: start-rooted only (rule 3 — transitive heads already
        # live inside anchor-rooted chain shapes when the stitch formed)
        if _supplement_only:
            pass
        elif view.set_mode:
            set_chain: Dict[tuple, _Pattern] = {}
            for c in block.chain_groups:
                if _covered(c.rels[0], "f", "edge") or \
                        _covered(c.rels[0], "f", "records"):
                    continue
                if not _member(c.start):
                    if not anchor_norm:
                        p = _Pattern(c.start, c.rels[0], "f",
                                     terminal="chains", count=c.n_chains,
                                     block=block)
                        p.hops = [((_short_rel(r) or r), "f") for r in c.rels]
                        pats.append(p)
                    continue
                key = tuple(c.rels)
                p = set_chain.get(key)
                if p is None:
                    p = _Pattern(view.anchor_display, c.rels[0], "f",
                                 terminal="chains", count=0, block=block)
                    p.hops = [((_short_rel(r) or r), "f") for r in c.rels]
                    set_chain[key] = p
                    pats.append(p)
                p.count += c.n_chains
        else:
            for c in block.chain_groups:
                if anchor_norm and (_covered(c.rels[0], "f", "edge") or
                                    _covered(c.rels[0], "f", "records")):
                    continue
                if not (_member(c.start) or not anchor_norm):
                    continue
                p = _Pattern(c.start, c.rels[0], "f", terminal="chains",
                             count=c.n_chains, block=block)
                p.hops = [((_short_rel(r) or r), "f") for r in c.rels]
                pats.append(p)
    return pats


def _overview(view: EvidenceView, cap: int) -> Tuple[List[str], List[_Pattern],
                                                     int]:
    pats = _collect_patterns(view)
    shown, hidden = pats[:cap], max(0, len(pats) - cap)
    return [p.render() for p in shown], pats, hidden


def _table(headers: List[str], rows: List[List[str]]) -> Tuple[List[str], int]:
    """Render a padded table. Returns (lines, total_width)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    total = sum(widths) + 3 * (len(headers) - 1)

    def _fmt(row):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip()

    return [_fmt(headers)] + [_fmt(row) for row in rows], total


def _rule(label: str, width: int) -> str:
    body = "── " + label + " "
    return body + "─" * max(4, _RULE_MIN_WIDTH - len(body), width - len(body))


def render_view(view: EvidenceView, row_cap: int = ROW_CAP,
                block_cap: int = BLOCK_CAP,
                overview_cap: int = OVERVIEW_CAP,
                cell_list_cap: int = CELL_LIST_CAP) -> Tuple[List[str], dict]:
    """Synthesize the two-layer form. Returns (lines, report); the report is
    the ground truth the invariant checker audits against.

    C7: block TITLES carry the full pattern shape; table COLUMN headers are
    relation short names (first column = the anchor slot) — the formal
    head/heads/tail/tails column names are gone."""
    if view.n_groups == 0:
        return [], {}

    ov, pats, ov_hidden = _overview(view, overview_cap)
    lines = ["pattern paths:"] + ov
    if ov_hidden:
        lines.append(f"  ... +{ov_hidden} more paths (see blocks below)")

    by_block: Dict[int, List[_Pattern]] = {}
    for p in pats:
        by_block.setdefault(id(p.block), []).append(p)

    def block_title(block) -> str:
        """C9 companion: the title shows the DOMINANT shape only (most
        instances) — no slash-joined direction pairs (confusable with the
        L1 '|' hierarchy); other directions stay visible in the rows."""
        bp = by_block.get(id(block)) or []
        if not bp:
            return block.label
        best = max(bp, key=lambda p: p.count)
        return best.shape()

    def anchor_col_header(block) -> str:
        """First-column header: the anchor slot (entity / set symbol)."""
        bp = by_block.get(id(block)) or []
        starts = {p.start for p in bp}
        if view.anchors and starts == {view.anchor_display}:
            return view.anchor_header
        if len(starts) == 1:
            return next(iter(starts))
        return "?node"

    def _pattern_of(block):
        for p in pats:
            if p.block is block and p.from_shape:
                return p        # C-consist binds SHAPE-backed blocks only
        return None

    report: dict = {
        "blocks": [], "overview_lines": len(ov), "overview_hidden": ov_hidden,
        "n_groups": view.n_groups,
        "n_patterns": len(pats),
        "patterns": [{"start": p.start,
                      "hops": [[r, d] for r, d in p.hops],
                      "terminal": p.terminal, "count": p.count}
                     for p in pats],
        "dropped": dict(view.dropped)}
    hidden_blocks = view.blocks[block_cap:]
    _blk_by_title: Dict[str, Block] = {}
    for block in view.blocks[:block_cap]:
        title = block_title(block)
        _blk_by_title.setdefault(title, block)
        # ── C3 chain block: one column per hop relation, rows = instance chains
        if block.chain_groups:
            headers = ([anchor_col_header(block)]
                       + [(_short_rel(r) or r) for r in
                          block.chain_groups[0].rels])
            rows: List[List[str]] = []
            emit_rows: List[List[List[str]]] = []
            hidden_ents = 0
            budget = row_cap
            total_chains = 0
            shown_chains = 0
            for c in block.chain_groups:
                total_chains += c.n_chains
                if budget <= 0:
                    continue
                budget -= 1
                shown_chains += c.n_chains
                cells_raw: List[List[str]] = [[c.start]]
                for pos in range(len(c.rels) - 1):
                    vals: List[str] = []
                    for m in c.middles:
                        if m[pos] not in vals:
                            vals.append(m[pos])
                    cells_raw.append(vals)
                cells_raw.append([c.end])
                row, safe = [], True
                shown_cells: List[List[str]] = []
                for vals in cells_raw:
                    if len(vals) == 1:
                        row.append(_fmt_val("", vals[0]))
                        shown_cells.append([_fmt_val("", vals[0])])
                    else:
                        jc = _join_cell(vals, cell_list_cap)
                        if jc is None:            # unsafe entities → no merge
                            safe = False
                            break
                        row.append(jc[0])
                        shown_cells.append(jc[2])
                        hidden_ents += jc[1]
                if not safe:
                    # fallback: one row per parallel chain (no L2 merge)
                    for m in c.middles or [[]]:
                        seq = [c.start] + list(m) + [c.end]
                        row = [_fmt_val("", e) for e in seq]
                        rows.append(row)
                        emit_rows.append([[_fmt_val("", e)] for e in seq])
                else:
                    rows.append(row)
                    emit_rows.append(shown_cells)
            tbl, width = _table(headers, rows)
            lines.append(_rule(title, width))
            lines.extend(tbl)
            hidden_rows = max(0, total_chains - shown_chains)
            clauses = []
            if hidden_rows > 0:
                clauses.append(f"+{hidden_rows} more chains "
                               f"(total {total_chains})")
            if hidden_ents > 0:
                clauses.append(f"+{hidden_ents} entities hidden")
            if clauses:
                lines.append("  ... " + "; ".join(clauses))
            report["blocks"].append({
                "title": title, "label": block.label, "kind": "chain",
                "cols": headers,
                "n_rows_total": total_chains,
                "n_rows_emitted": shown_chains,
                "hidden_rows": hidden_rows, "hidden_entities": hidden_ents,
                "emit_rows": [[[str(x) for x in cell] for cell in r]
                              for r in emit_rows],
                "n_groups": block.n_groups,
            })
            continue

        # ── record block: C1 terminal semantics (edge-only blocks skip)
        uniform: List[Tuple[_Group, List[Tuple[str, str]]]] = []
        for g in block.record_groups:
            det = [r for r in g.rows if not r.ref_only]
            if len(det) < 2:
                continue
            keys: List[str] = []
            for k, _v in det[0].attrs:
                if k not in keys:
                    keys.append(k)
            uni = []
            for k in keys:
                vals = [next((v for kk, v in r.attrs if kk == k), None)
                        for r in det]
                if all(v is not None for v in vals) and \
                        len({normalize(";".join(str(x) for x in v))
                             for v in vals}) == 1:
                    uni.append((k, vals[0]))
            if uni:
                uniform.append((g, uni))
        uniform_keys = {id(g): {k for k, _ in uni} for g, uni in uniform}

        rec_assoc: List[Tuple[_Group, RecordRow]] = []
        for g in block.record_groups:
            for r in g.rows:
                rec_assoc.append((g, r))

        use_table = (any(len(g.rows) >= 2 for g in block.record_groups)
                     or len({normalize(g.head)
                             for g in block.record_groups}) > 1)
        if rec_assoc and not use_table:
            # single head, every group single-record → terminal LINES form
            lines.append(_rule(title + _TERMINAL_TAG, 0))
            emit_recs = []
            for g, r in rec_assoc:
                if r.ref_only:
                    lines.append(f"  {r.rec_id}")
                    emit_recs.append({"head": normalize(g.head),
                                      "id": r.rec_id, "attrs": []})
                    continue
                parts = []
                hidden_attrs = 0
                for k, vals in r.attrs[:ATTR_LINE_CAP]:
                    cell, hid = _attr_cell(k, vals, cell_list_cap)
                    parts.append(f"{k}={cell}")
                    hidden_attrs += hid
                hidden_attrs += max(0, len(r.attrs) - ATTR_LINE_CAP)
                if len(r.attrs) > ATTR_LINE_CAP:
                    parts.append(f"…+{len(r.attrs) - ATTR_LINE_CAP} more attrs")
                lines.append(f"  {r.rec_id} [{'; '.join(parts)}]")
                emit_recs.append({"head": normalize(g.head), "id": r.rec_id,
                                  "attrs": parts, "hidden": hidden_attrs})
            report["blocks"].append({
                "title": title, "label": block.label, "kind": "records-lines",
                "cols": [],
                "n_rows_total": len(rec_assoc), "n_rows_emitted": len(rec_assoc),
                "hidden_rows": 0, "hidden_entities": 0,
                "emit_records": emit_recs, "uniform": {},
                "n_groups": block.n_groups,
            })
            continue

        if block.record_groups:
            # record TABLE (multi-record and/or multi-head): first column =
            # the anchor slot (when heads differ), then the RECORD relation
            # column (cells = record ids), then attr-relation columns
            multi_head = len({normalize(g.head)
                              for g in block.record_groups}) > 1
            col_keys: List[str] = []
            for g in block.record_groups:
                for r in g.rows:
                    if r.ref_only:
                        continue
                    for k, _v in r.attrs:
                        if k not in col_keys and k not in uniform_keys.get(id(g), ()):
                            col_keys.append(k)
            rec_budget = row_cap
            rec_rows, rec_pairs = [], []
            for g in block.record_groups:
                if rec_budget <= 0:
                    break
                for r in g.rows:
                    if rec_budget <= 0:
                        break
                    hide = uniform_keys.get(id(g), ())
                    row = ([_fmt_val("", g.head)] if multi_head else []) \
                        + [r.rec_id]
                    for k in col_keys:
                        if k in hide:
                            row.append("")      # hoisted → lives in (all:) only
                            continue
                        vals = next((v for kk, v in r.attrs if kk == k), None)
                        row.append(_attr_cell(k, vals or [], cell_list_cap)[0]
                                   if vals else "")
                    rec_rows.append(row)
                    rec_pairs.append((g, r))
                    rec_budget -= 1
            cells_emit: List[List[List[str]]] = []
            for g, r in rec_pairs:
                hide = uniform_keys.get(id(g), ())
                cell_ents = [[_fmt_val("", g.head)] if multi_head else []]
                for k in col_keys:
                    if k in hide:
                        cell_ents.append([])
                        continue
                    vals = next((v for kk, v in r.attrs if kk == k), None)
                    cell_ents.append(list(vals or []))
                cells_emit.append(cell_ents)
            headers = (([anchor_col_header(block)] if multi_head else [])
                       + [(_short_rel(block.rel_full) or block.rel_full)]
                       + col_keys)
            date_col = next((i for i, k in enumerate(col_keys)
                             if k.lower() in _DATE_ATTR_KEYS), None)
            if date_col is not None:
                off = (1 if multi_head else 0) + 1
                order = sorted(range(len(rec_rows)),
                               key=lambda i: (_year_key(rec_rows[i][off + date_col]), i))
                rec_rows = [rec_rows[i] for i in order]
                cells_emit = [cells_emit[i] for i in order]
                rec_pairs = [rec_pairs[i] for i in order]
            if rec_rows:
                tbl, width = _table(headers, rec_rows)
                lines.append(_rule(title + _TERMINAL_TAG, width))
                lines.extend(tbl)
            total_records = sum(len(g.rows) for g in block.record_groups)
            hidden_rows = max(0, total_records - len(rec_rows))
            if hidden_rows > 0:
                lines.append(f"  ... +{hidden_rows} more records "
                             f"(total {total_records})")
            for g, uni in uniform:
                multi = len({normalize(x.head)
                             for x in block.record_groups}) > 1
                body = "; ".join(f"{k}={_attr_cell(k, v, cell_list_cap)[0]}"
                                 for k, v in uni)
                lines.append(f"  (all {g.head}: {body})" if multi
                             else f"  (all: {body})")
            report["blocks"].append({
                "title": title, "label": block.label, "kind": "records-table",
                "cols": headers, "anchor_col": multi_head,
                "n_rows_total": total_records,
                "n_rows_emitted": len(rec_rows),
                "hidden_rows": hidden_rows, "hidden_entities": 0,
                "emit_records": [{"head": normalize(g.head), "id": r.rec_id}
                                 for g, r in rec_pairs],
                "emit_cells": cells_emit, "col_keys": col_keys,
                "uniform": {f"{normalize(g.head)}|{g.rel_short}":
                            [[k, ";".join(str(x) for x in v)] for k, v in uni]
                            for g, uni in uniform},
                "multi_head": multi_head,
                "n_groups": block.n_groups,
            })

        # ── edge block: C2 sibling-row folding (+ C5 branch refs on caps).
        # C7: columns = [anchor slot, relation]; ANCHOR-SIDE cell first.
        if block.edge_groups:
            total_edges = sum(len(g.tails) for g in block.edge_groups)
            by_ts: Dict[tuple, List[str]] = {}
            ts_order: List[tuple] = []
            for g in block.edge_groups:
                key = tuple(normalize(t) for t in g.tails)
                if key not in by_ts:
                    by_ts[key] = []
                    ts_order.append(key)
                by_ts[key].append(g.head)
            anchor_norm = view.anchors

            def _is_rev(tails) -> bool:
                return any(normalize(t) in anchor_norm for t in tails)

            budget = row_cap
            emitted_edges = 0
            rows_e: List[List[str]] = []
            emit_rows_e: List[List[List[str]]] = []
            full_rows: List[List[List[str]]] = []   # C8 full-display audit
            hidden_ents = 0
            hidden_cells: List[int] = []
            first_ref = None
            n_rev = 0
            for key in ts_order:
                if budget <= 0:
                    break
                heads = by_ts[key]
                tails = next(g.tails for g in block.edge_groups
                             if tuple(normalize(x) for x in g.tails) == key)
                budget -= 1
                emitted_edges += len(heads) * len(tails)
                jc_h = _join_cell(heads, cell_list_cap)
                jc_t = _join_cell(tails, cell_list_cap)
                if jc_h is None or jc_t is None:
                    # unsafe entity names → NO folding, one row per edge
                    for h in heads:
                        for t in tails:
                            rows_e.append([_fmt_val("", h), _fmt_val("", t)])
                            emit_rows_e.append([[_fmt_val("", h)],
                                                 [_fmt_val("", t)]])
                            full_rows.append([[_fmt_val("", h)],
                                              [_fmt_val("", t)]])
                    continue
                hidden_ents += jc_h[1] + jc_t[1]
                hidden_cells.extend(c for c in (jc_h[1], jc_t[1]) if c > 0)
                if _is_rev(tails):
                    n_rev += 1
                    cells = [jc_t, jc_h]         # anchor side first
                    full = [[_fmt_val("", t) for t in tails],
                            [_fmt_val("", h) for h in heads]]
                else:
                    cells = [jc_h, jc_t]
                    full = [[_fmt_val("", h) for h in heads],
                            [_fmt_val("", t) for t in tails]]
                rows_e.append([cells[0][0], cells[1][0]])
                # C5: the FIRST cell-cap truncation carries the compressed
                # submission ref — anchor = the OPPOSITE cell's first shown
                # entity (expansion collects this group's full hidden side)
                if first_ref is None and jc_t[1] > 0 and jc_h[2]:
                    first_ref = (jc_t[1], len(tails), jc_h[2][0])
                elif first_ref is None and jc_h[1] > 0 and jc_t[2]:
                    first_ref = (jc_h[1], len(heads), jc_t[2][0])
                emit_rows_e.append([cells[0][2], cells[1][2]])
                full_rows.append(full)
            all_rev = n_rev == len(emit_rows_e) and n_rev > 0
            rel_col = ("<--" + (_short_rel(block.rel_full)
                                or block.rel_full)) if all_rev \
                else (_short_rel(block.rel_full) or block.rel_full)
            headers = [anchor_col_header(block), rel_col]
            if rows_e:
                tbl, width = _table(headers, rows_e)
                lines.append(_rule(title, width))
                lines.extend(tbl)
            hidden_edges = total_edges - emitted_edges   # row-budget drops
            emitted_one_marker = False
            if hidden_edges > 0:
                # row-cap: whole groups dropped → refs for the dropped groups
                dropped = ts_order[len(emit_rows_e):]
                ref = ""
                if dropped:
                    dh = by_ts[dropped[0]][0]
                    if _ref_safe(dh):
                        ref = (f' {_REF_INSTR.format(ref=f"{dh}::{block.label}")}')
                        extra = [f"#{by_ts[k][0]}::{block.label}"
                                 for k in dropped[1:3]
                                 if _ref_safe(by_ts[k][0])]
                        if extra:
                            ref += f" (also: {' | '.join(extra)})"
                lines.append(f"  ... +{hidden_edges} more edges "
                             f"(total {total_edges}).{ref}")
                emitted_one_marker = True
            if first_ref is not None and not emitted_one_marker:
                h, t, anchor = first_ref
                if _ref_safe(anchor):
                    lines.append(f"  +{h} more (total {t}). "
                                 + _REF_INSTR.format(
                                     ref=f"{anchor}::{block.label}"))
            report["blocks"].append({
                "title": title, "label": block.label, "kind": "edges",
                "cols": headers,
                "n_rows_total": total_edges, "n_rows_emitted": len(rows_e),
                "hidden_rows": max(0, total_edges - emitted_edges),
                "hidden_entities": hidden_ents,
                "hidden_cells": hidden_cells,
                "branch_ref": (f"#{first_ref[2]}::{block.label}"
                               if first_ref and _ref_safe(first_ref[2]) else None),
                "emit_rows": [[[str(x) for x in c] for c in r]
                              for r in emit_rows_e],
                "full_rows": [[[str(x) for x in c] for c in r]
                              for r in full_rows],
                "n_groups": block.n_groups,
            })
    # C-consist (a): shape-backed blocks carry their pattern's count —
    # overview promise == block instance rows
    for b in report["blocks"]:
        blk = _blk_by_title.get(b.get("title"))
        if blk is not None:
            pp = _pattern_of(blk)
            if pp is not None:
                b["pattern_count"] = pp.count
    if hidden_blocks:
        names = "; ".join(block_title(b) for b in hidden_blocks)
        lines.append(f"... +{len(hidden_blocks)} more relations truncated: {names}")
        report["hidden_blocks"] = [[block_title(b), b.rel_full]
                                   for b in hidden_blocks]
    return lines, report


def render_evidence(triples: Sequence, row_cap: int = ROW_CAP,
                    block_cap: int = BLOCK_CAP,
                    overview_cap: int = OVERVIEW_CAP,
                    cell_list_cap: int = CELL_LIST_CAP,
                    anchors: Sequence[str] = (),
                    anchor_label: Optional[str] = None,
                    patterns: Sequence[dict] = ()) -> List[str]:
    """One-shot: canonicalized triples → overview + relation blocks.
    ``anchors`` = the call's centers; ``anchor_label`` = the ?var token
    (C6 set-anchor rendering for multi-center walks)."""
    view = build_view(triples, row_cap, block_cap, overview_cap,
                      cell_list_cap, anchors, anchor_label, patterns)
    lines, _ = render_view(view, row_cap, block_cap, overview_cap,
                           cell_list_cap)
    return lines


# ── production drop-in (signature-compatible with seq_tools._render_records) ─

def render_records_compat(all_triples, shown_edges=None, anchors=None,
                          anchor_label=None, patterns=None,
                          row_cap: int = ROW_CAP, block_cap: int = BLOCK_CAP,
                          overview_cap: int = OVERVIEW_CAP,
                          cell_list_cap: int = CELL_LIST_CAP):
    """Drop-in for ``_render_records(all_triples, shown_edges)``.

    Same contract: returns ``(lines, n_overlap)``, dedups against (and adds to)
    ``shown_edges`` so a per-call set collapses within-call duplicates and a
    cross-call set would count overlaps. ``anchors`` (C4) = the call's center
    entities, used ONLY by the pattern-level overview. Rendering itself never
    reads any cross-call state (I4)."""
    if shown_edges is None:
        shown_edges = set()
    view = build_view(all_triples, row_cap, block_cap, overview_cap,
                      cell_list_cap, anchors or (), anchor_label, patterns)
    n_overlap = 0
    for g in view.edge_groups:
        for t in g.tails:
            key = ("E", normalize(g.head), g.rel_short, normalize(t))
            if key in shown_edges:
                n_overlap += 1
            else:
                shown_edges.add(key)
    for g in view.record_groups:
        for r in g.rows:
            key = ("REC", normalize(g.head), g.rel_short, r.signature())
            if key in shown_edges:
                n_overlap += 1
            else:
                shown_edges.add(key)
    for c in view.chain_groups:
        key = ("C", normalize(c.start), "›".join(c.rels), normalize(c.end))
        if key in shown_edges:
            n_overlap += 1
        else:
            shown_edges.add(key)
    lines, _ = render_view(view, row_cap, block_cap, overview_cap,
                           cell_list_cap)
    return lines, n_overlap


# ── programmatic invariants (I1-I5) ─────────────────────────────────────────

_RE_RULE = re.compile(r"^── (.+?) ─+$")
_RE_ALL = re.compile(r"^  \(all(?: (.+?))?: (.+)\)$")
_RE_RECLINE = re.compile(r"^  ((?:m|g)\.[0-9a-z_]+)(?: \[(.*)\])?$")
_RE_MORE = re.compile(r"^  \.\.\. \+(\d+) more (records|edges|chains)")
# C5 branch-ref marker: `  +K more (total T). To answer with ALL of them,
# include "#anchor::rel" as one answer entity — ...`
_RE_REFMORE = re.compile(
    r"^  \+(\d+) more \(total (\d+)\)\. To answer with ALL of them, "
    r'include "(#[^"]+)" as one answer entity')
# local copy of tools._BRANCH_RE (keep in sync — stages must not import agent)
_RE_BRANCH_TOKEN = re.compile(r"^#(.{1,200}?)(?:::|\|)(.{1,300})$")


def _parse_cell(cell: str) -> List[str]:
    """Parse a rendered cell back into entities: bare name or L2 parens."""
    c = cell.strip()
    if c.startswith("(") and c.endswith(")"):
        inner = c[1:-1]
        out = []
        for part in inner.split("; "):
            part = part.strip()
            if part.startswith("…+"):
                continue                 # in-cell cap marker (hidden entities)
            out.append(part)
        return out
    return [c] if c else []


def check_invariants(triples: Sequence, row_cap: int = ROW_CAP,
                     block_cap: int = BLOCK_CAP,
                     overview_cap: int = OVERVIEW_CAP,
                     cell_list_cap: int = CELL_LIST_CAP,
                     anchors: Sequence[str] = (),
                     anchor_label: Optional[str] = None,
                     patterns: Sequence[dict] = (),
                     candidates: Sequence[str] = None) -> List[str]:
    """Full I1-I5 + C4/C5/C7/C8 audit on (triples → rendered text). Returns
    violation strings; empty list = all green. Used by tests AND the replay
    evaluator."""
    v: List[str] = []
    snapshot = [tuple(t) if isinstance(t, (tuple, list)) else t
                for t in (triples or [])]
    view = build_view(triples, row_cap, block_cap, overview_cap,
                      cell_list_cap, anchors, anchor_label, patterns)
    lines, report = render_view(view, row_cap, block_cap, overview_cap,
                                cell_list_cap)
    after = [tuple(t) if isinstance(t, (tuple, list)) else t
             for t in (triples or [])]
    if snapshot != after:
        v.append("I4: renderer mutated its input triples")

    lines2 = render_evidence(triples, row_cap, block_cap, overview_cap,
                             cell_list_cap, anchors, anchor_label, patterns)
    if lines2 != lines:
        v.append("I4: non-deterministic render (two runs differ)")

    n_ov = sum(1 for ln in lines
               if ln.startswith("  ") and ("-->" in ln or "<--" in ln))
    if report:
        if report["overview_hidden"]:
            if not any(f"+{report['overview_hidden']} more paths" in ln
                       for ln in lines):
                v.append("I5: overview truncation missing explicit marker")
        elif n_ov != report["n_patterns"]:
            v.append(f"I1-index: overview lines {n_ov} != distinct patterns "
                     f"{report['n_patterns']}")
        # C4 slot purity: every overview line starts with ONE entity (or the
        # SET symbol) and all later payloads are slots / [N records] / counts
        re_count = re.compile(r" \(\d+ (?:instances|chains)\)$")
        re_hop = re.compile(
            r" --(.+?)--> (\?[a-z0-9]+|\[\d+ records?\])"
            r"| <--(.+?)-- (\?[a-z0-9]+|\[\d+ records?\])")
        for ln in lines:
            if not ln.startswith("  ") or ("-->" not in ln and "<--" not in ln):
                continue
            body = re_count.sub("", ln.strip())
            hops = list(re_hop.finditer(body))
            if not hops:
                v.append(f"C4: overview line has no parseable hops: "
                         f"{ln[:70]!r}")
                continue
            start = body[:hops[0].start()].strip()
            if not start or start.startswith("?") and "(" not in start \
                    and not start.startswith("?x"):
                # set symbols like "?film" ARE legal starts (C6); bare slots
                # ("?x") as a START are not
                if start in _SLOTS:
                    v.append(f"C4: overview line starts with a bare slot: "
                             f"{ln[:70]!r}")
            tail = body[hops[-1].end():].strip()
            if tail and not re.fullmatch(r"\[\d+ records?\]", tail):
                v.append(f"C4: non-slot payload in overview line "
                         f"({tail[:24]!r}): {ln[:70]!r}")
            if "|" in ln:
                v.append(f"C4: instance enumeration '|' in overview line: "
                         f"{ln[:70]!r}")

    # ── parse the detail layer back, driven by the report's block sequence ──
    parsed_edges: List[Tuple[List[str], List[str]]] = []
    parsed_chains: List[List[List[str]]] = []
    parsed_recs: List[Tuple[str, str, str]] = []           # (head, title, id)
    parsed_uniform: List[Tuple[str, Optional[str], str]] = []
    rb = list(report.get("blocks", []))
    bi = 0
    cur: Optional[dict] = None
    got_headers: List[str] = []
    seen_header_rows: List[List[str]] = []
    for ln in lines:
        m = _RE_RULE.match(ln)
        if m:
            title = m.group(1)
            while bi < len(rb) and not title.startswith(rb[bi]["title"]):
                bi += 1
            cur = rb[bi] if bi < len(rb) else None
            bi += 1
            got_headers = []
            continue
        if cur is None or not ln.strip():
            continue
        if ln.lstrip().startswith(("...", "+", "…")):
            continue
        ma = _RE_ALL.match(ln)
        if ma:
            kvs = [p.split("=", 1) for p in ma.group(2).split("; ") if "=" in p]
            head = ma.group(1) or None
            attr_cols = (cur.get("col_keys") if cur.get("kind") ==
                         "records-table" else got_headers) or []
            for k, _ in kvs:
                parsed_uniform.append((cur["title"], head, k.strip()))
                if head is None and k.strip() in attr_cols:
                    v.append(f"I3: uniform attr {k.strip()!r} also a column "
                             f"of single-head block {cur['title']!r}")
            continue
        ml = _RE_RECLINE.match(ln)
        if ml and not got_headers:
            parsed_recs.append(("", cur["title"].replace(_TERMINAL_TAG, ""),
                                ml.group(1)))
            continue
        cells = [c.strip() for c in ln.split("|")]
        if not got_headers:
            got_headers = cells
            seen_header_rows.append(cells)
            if cur.get("cols") and cells != cur["cols"]:
                v.append(f"C7: header row {cells} != report cols "
                         f"{cur['cols']} in {cur['title']!r}")
            continue
        if len(cells) != len(got_headers):
            v.append(f"I1: unparseable table line (col count — in-cell '|'?): "
                     f"{ln[:90]!r}")
            continue
        if cur.get("kind") == "chain":
            parsed_chains.append([_parse_cell(c) for c in cells])
        elif cur.get("kind") == "edges":
            parsed_edges.append((_parse_cell(cells[0]), _parse_cell(cells[1])))
        elif cur.get("kind") == "records-table":
            off = 1 if cur.get("anchor_col") else 0
            parsed_recs.append((cells[0] if off else "",
                                cur["title"].replace(_TERMINAL_TAG, ""),
                                cells[off]))

    # C7 hard rule: no formal head/tail column names anywhere (the exact
    # user-specified literal set — a RELATION whose short name coincides is
    # still a relation column, not a formal slot name)
    for hdr in seen_header_rows:
        bad = [h for h in hdr if h in ("head", "heads", "tail", "tails")]
        if bad:
            v.append(f"C7: formal column name(s) {bad} survived: {hdr}")

    # I1 + C8: every FULL side list must be represented by a rendered row —
    # exact when all sides are ≤ cap; a subset match is legal ONLY above cap
    # (markers audited separately by I5/C5)
    for b in report.get("blocks", []):
        if b.get("kind") not in ("edges", "chain"):
            continue
        full = b.get("full_rows") or b.get("emit_rows")
        pool = parsed_edges if b["kind"] == "edges" else parsed_chains
        for row_full in full:
            exact = subset = False
            for prow in pool:
                if len(prow) != len(row_full):
                    continue
                if all(set(p) == set(f) for p, f in zip(prow, row_full)):
                    exact = True
                    break
                if all(set(p) <= set(f) for p, f in zip(prow, row_full)):
                    subset = True
            if exact:
                continue
            over_cap = any(len(set(f)) > cell_list_cap for f in row_full)
            if subset and over_cap:
                continue
            if subset:
                v.append(f"C8: cell with ≤{cell_list_cap} entities not fully "
                         f"shown in {b['title']!r}")
            else:
                v.append(f"I1: no rendered row matches emitted row in "
                         f"{b['title']!r}")

    # records: ids per block (order-insensitive multiset)
    exp_recs = sorted((b["title"].replace(_TERMINAL_TAG, ""), d["id"])
                      for b in report.get("blocks", [])
                      for d in b.get("emit_records", []))
    got_recs = sorted((t, rid) for h, t, rid in parsed_recs)
    if exp_recs != got_recs:
        v.append(f"I1: record rows mismatch — parsed {len(got_recs)} vs "
                 f"emitted {len(exp_recs)}")

    # I2: each CVT's attr set printed at most once (ref lines carry no attrs)
    with_attrs: Dict[str, int] = {}
    for ln in lines:
        ml = _RE_RECLINE.match(ln)
        if ml and ml.group(2):
            with_attrs[ml.group(1)] = with_attrs.get(ml.group(1), 0) + 1
    for rid, n in with_attrs.items():
        if n > 1:
            v.append(f"I2: record {rid} printed with attrs {n}x")

    # I3: (all:) line count == uniform record GROUPS
    n_all = sum(1 for ln in lines if _RE_ALL.match(ln))
    n_uni = sum(len(b["uniform"]) for b in report.get("blocks", [])
                if b.get("uniform"))
    if n_all != n_uni:
        v.append(f"I3: (all:) lines {n_all} != uniform groups {n_uni}")

    # I5 + C5: budget truncations and branch refs
    for b in report.get("blocks", []):
        if b.get("hidden_rows", 0) > 0:
            want = f"+{b['hidden_rows']} more"
            total = f"(total {b['n_rows_total']})"
            if not any(ln.lstrip().startswith("...") and want in ln and total in ln
                       for ln in lines):
                v.append(f"I5: block {b['title']} hid {b['hidden_rows']} rows "
                         "without an exact marker")
        he = b.get("hidden_entities", 0)
        hidden_cells = b.get("hidden_cells", [])
        if he > 0:
            ok = any(f"…+{c} more" in ln or f"+{c} entities hidden" in ln
                     or f"+{c} more (total" in ln
                     for c in (hidden_cells or [he]) for ln in lines)
            if not ok:
                v.append(f"I5: block {b['title']} hid {he} entities "
                         "without an exact marker")
        if b.get("kind") == "edges" and (he > 0 or b.get("hidden_rows", 0) > 0):
            refs = [m.group(3) for ln in lines for m in
                    [_RE_REFMORE.search(ln)] if m] + \
                   [tok for ln in lines if "as one answer entity" in ln
                    for tok in re.findall(r'"(#[^"]+)"', ln)]
            if not refs:
                v.append(f"C5: block {b['title']} hides candidates with no "
                         "branch ref")
            for tok in refs:
                if not _RE_BRANCH_TOKEN.match(tok):
                    v.append(f"C5: ref token fails _BRANCH_RE grammar: "
                             f"{tok[:40]!r}")
            for ln in lines:
                m = _RE_REFMORE.search(ln)
                if m:
                    more, total = int(m.group(1)), int(m.group(2))
                    if more > total:
                        v.append(f"I5: ref marker more({more}) > total({total})")
                    if not any(len(cell) + more == total
                               for r in b.get("emit_rows", []) for cell in r):
                        v.append(f"I5: ref marker counts inconsistent "
                                 f"(+{more} of {total} matches no emitted cell)")
    if report.get("hidden_blocks"):
        want = f"+{len(report['hidden_blocks'])} more relations truncated"
        if not any(want in ln for ln in lines):
            v.append("I5: block truncation missing marker")

    # C-consist (a): overview count == block instance rows for every
    # shape-backed block (single source of truth; no truncation pending)
    for b in report.get("blocks", []):
        pc = b.get("pattern_count")
        if pc is None:
            continue
        if b.get("hidden_rows", 0) == 0 and b.get("hidden_entities", 0) == 0                 and b.get("n_rows_total") != pc:
            v.append(f"C-consist: pattern {b['title']!r} promises {pc} "
                     f"instances but block has {b.get('n_rows_total')} rows")

    # C-consist (b): every candidate must be traceable to some block (rows,
    # (all:) attr values, record-line attrs, or bare-id refs)
    if candidates:
        pool = set()
        for b in report.get("blocks", []):
            for r in b.get("emit_rows", []):
                for cell in r:
                    pool.update(normalize(x) for x in cell)
            for cells in b.get("emit_cells", []):
                for cell in cells:
                    pool.update(normalize(x) for x in cell)
            for d in b.get("emit_records", []):
                pool.add(normalize(d.get("id", "")))
            for head_key, kvs in (b.get("uniform") or {}).items():
                for k, val in kvs:
                    pool.update(normalize(x) for x in str(val).split(";"))
        for ln in lines:
            ml = _RE_RECLINE.match(ln)
            if ml and ml.group(2):
                for part in ml.group(2).split("; "):
                    if "=" in part:
                        pool.add(normalize(part.split("=", 1)[1]
                                           .strip("()") if part.split("=", 1)[1]
                                           .startswith("(")
                                           else part.split("=", 1)[1]))
        for c in candidates or ():
            nc = normalize(str(c))
            if nc and nc not in pool:
                v.append(f"C-consist: candidate {c!r} not traceable to any "
                         "block")
                break
    return v
