"""WALK STATE PORTS (user ruling 2026-10-09): the single, shared data
interface for every mechanism that consumes or produces per-chain walk
state.

Problem this module kills: mechanisms used to pass their own
differently-shaped copies of the same state around (chain_landing written
by prepare, rebuilt by _rebuild_pe_list, whitelisted field-by-field by the
staged fork/publish) — one missed field name and a consumer goes silently
blind (the _failed_steps incident). Here the cross-mechanism contract is
declared ONCE:

  SEED_FIELDS   — every per-case state attribute that must cross a
                  staged fork; the fork and the publish iterate THIS
                  list, so a new state field is added in exactly one
                  place.
  PUBLISH_FIELDS — the subset that survives a scoped publish on success.

Function-local names stay free; what crosses function boundaries comes
from these definitions only.

The frontier accessor is likewise single-sourced: frontier_of() is the
only definition of "the pathway frontier of a chain" — the RR menu pools,
the family-expansion gate, and any future consumer all call it, so
relation retrieval can never re-align to a different notion of "center".
"""
from __future__ import annotations

# (attr, seed_kind) — seed kinds:
#   "map"       deep-ish copy: dict of mutable values, value-copied
#   "dict"      shallow dict copy
#   "set"       new set
#   None        reset to empty on fork (per-call scratch, never crosses)
SEED_FIELDS = (
    ("anchor_seqs", "map"),        # {root: [frozenset(rel_idxs), ...]}
    ("fid_tree_roots", "map"),     # {fid: [root, ...]}
    ("chain_landing", "map"),      # {root: {"segs","attrs","pref","starts"}}
    ("chain_touch", "dict"),       # {root: recency clock}
    ("_failed_steps", "map"),      # {root: [frozenset(rel_names), ...]}
    ("consumed_anchors", "set"),
    ("_sg_served", None),          # per-call memo — reset
    ("anchor_seq_memo", None),     # per-call memo — reset
    ("_rel_last_memo", None),
    ("_fam_expand_memo", None),
    ("var_bindings", "dict"),
    ("pattern_state", "dict"),
)


def seed_fork(staged, live):
    """Seed the staged fork's SEED_FIELDS from live. One definition — the
    fork can never again forget a field."""
    import copy as _copy
    for name, kind in SEED_FIELDS:
        v = getattr(live, name, None)
        if kind == "map":
            setattr(staged, name,
                    {k: _copy.deepcopy(x) for k, x in (v or {}).items()}
                    if v else {})
        elif kind == "dict":
            setattr(staged, name, dict(v or {}))
        elif kind == "set":
            setattr(staged, name, set(v or ()))
        else:
            setattr(staged, name, {})


def publish(staged, live, touched_roots=()):
    """Scoped publish on success: anchor_seqs/fid_tree_roots merge only the
    roots this worker edited; the remaining PUBLISH fields copy wholesale
    (they are per-case, not per-root)."""
    touched = set(touched_roots or ())
    st_sq = getattr(staged, "anchor_seqs", None) or {}
    if touched:
        lv = getattr(live, "anchor_seqs", None) or {}
        for a in touched:
            if a in st_sq:
                lv[a] = st_sq[a]
        live.anchor_seqs = lv
    st_ft = getattr(staged, "fid_tree_roots", None) or {}
    if touched and st_ft:
        lv = getattr(live, "fid_tree_roots", None) or {}
        for k, roots in st_ft.items():
            lr = lv.setdefault(k, [])
            for r in roots:
                if r not in lr:
                    lr.append(r)
        live.fid_tree_roots = lv
    for name in ("chain_landing", "chain_touch", "_failed_steps",
                 "consumed_anchors", "var_bindings", "pattern_state"):
        v = getattr(staged, name, None)
        if v is not None:
            setattr(live, name, v)


def frontier_of(ctx, root):
    """THE pathway-frontier definition (single source): the chain's
    last-segment landing ∪ terminal CVT attr endpoints; falls back to the
    layer-completion projection for born-but-not-delivered chains. Never
    includes the root itself."""
    land = (getattr(ctx, "chain_landing", None) or {}).get(root) or {}
    segs = list(land.get("segs") or ())
    if segs and segs[-1]:
        front = set(segs[-1]) | set(land.get("attrs") or frozenset())
    else:
        from kgqa.agent.seq_tools import _seq_completions
        from kgqa.traversal.pattern_walk import get_pattern_index
        ast = getattr(ctx, "anchor_seqs", None) or {}
        if root not in ast:
            return set()
        front = set(_seq_completions(
            ctx, get_pattern_index(ctx), root, ast[root])[-1])
    return set(front) - {root}


def pool_for_center(ctx, ci, own_pool=True):
    """THE relation-retrieval pool rule (user alignment ruling): a
    declared entity's pool is the FRONTIER pool of the chain it belongs
    to — the declaration only anchors; the retrieval starts from the
    pathway frontier. Entities attached to no chain (heads, unattached)
    keep their own neighborhood pool. own_pool: also union the entity's
    own pool (named entities are usually frontier members and only ever
    ADD coverage — RR frontier-default design 2026-10-05)."""
    from kgqa.agent.seq_tools import _seq_pool_relids
    pool = set()
    hit = False
    cl = getattr(ctx, "chain_landing", None) or {}
    ast = getattr(ctx, "anchor_seqs", None) or {}
    for root, land in cl.items():
        segs = list(land.get("segs") or ())
        starts = list(land.get("starts") or ())
        member = (ci == root
                  or any(ci in s for s in segs)
                  or ci in (land.get("attrs") or frozenset())
                  or any(ci in st for st in starts))
        if member:
            front = frontier_of(ctx, root)
            if front:
                pool |= _seq_pool_relids(ctx, front)
                hit = True
    if own_pool or not hit:
        pool |= _seq_pool_relids(ctx, {ci})
    return pool
