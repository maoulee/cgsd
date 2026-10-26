"""PATTERN-PATH WALK (user design 2026-09-10): set-state symbolic traversal.

Replaces per-binding entity traversal for multi-binding calls (?var centers):
the binding set is ONE set state (a bitmap over entity idx); a walk step is a
RELATION MODE transition  S ∩ H_r -> N_r(S ∩ H_r);  bridge hops use any
non-target relation, a target-relation hit terminates the segment (RPE's
segment semantics preserved). Branching collapses from thousands of entity
edges to a few dozen relation modes. Entity witnesses are materialized ONLY
for the top-K ranked patterns at the end (先模式,再实例化).

Prototype validation (tmp/proto_pattern_walk.py, wallexcap replay): 109s ->
6.3-8.9s over 77 real multi-binding calls (12-17x), bridge semantics intact.

Ranking (A+ design, user ruling 2026-09-10):
  (terminal-relation GROUP order  — the model's own submission order first,
   path length ascending,          — shortest first (join-path ruling)
   support descending,
   GTE only tie-breaks)            — the model's choice beats GTE.
"""
from collections import defaultdict


class CasePatternIndex:
    """One O(E) relation index per case. Bitmaps are Python ints over ent idx."""

    def __init__(self, ents, rels, h_ids, r_ids, t_ids):
        self.ents, self.rels = ents, rels
        n = len(ents)
        self.head_mask = defaultdict(int)
        self.tail_mask = defaultdict(int)
        self.fwd = defaultdict(lambda: defaultdict(list))
        self.rev = defaultdict(lambda: defaultdict(list))
        for h, r, t in zip(h_ids, r_ids, t_ids):
            if not (0 <= h < n and 0 <= t < n):
                continue
            self.head_mask[r] |= 1 << h
            self.tail_mask[r] |= 1 << t
            self.fwd[r][h].append(t)
            self.rev[r][t].append(h)

    def transition(self, state, r, forward):
        if forward:
            src = state & self.head_mask[r]
            f, nxt = self.fwd[r], 0
        else:
            src = state & self.tail_mask[r]
            f, nxt = self.rev[r], 0
        m = src
        while m:
            b = m & -m
            for t in f.get(b.bit_length() - 1, ()):
                nxt |= 1 << t
            m ^= b
        return nxt

    @staticmethod
    def bits(m):
        while m:
            b = m & -m
            yield b.bit_length() - 1
            m ^= b


def get_pattern_index(ctx):
    """Per-ctx memoized index (deterministic over the immutable case arrays)."""
    ix = getattr(ctx, "_pattern_index", None)
    if ix is None:
        ix = ctx._pattern_index = CasePatternIndex(
            ctx.ents, ctx.rels, ctx.h_ids, ctx.r_ids, ctx.t_ids)
    return ix


def pattern_walk(ix, seed_names, target_rel_idxs, max_hops=3, beam=60,
                 topk_patterns=10, witness_k=12):
    """Set-state BFS from the binding set. Returns a ranked pattern list.

    Each pattern: {'rels': [(ridx, fwd), ...], 'answer': [names],
                   'support': int, 'witnesses': [chains of (h, r, t) names]}.
    CVT members of a terminal answer set are penetrated one hop (named
    entities behind the event join the answer).
    """
    from kgqa.traversal.cvt import is_cvt_like
    name2idx = {}
    for j, e in enumerate(ix.ents):
        name2idx.setdefault(str(e), j)
    seed = 0
    for nm in seed_names:
        j = name2idx.get(str(nm))
        if j is not None:
            seed |= 1 << j
    if not seed:
        return []
    tset = set(target_rel_idxs)

    cvt = [is_cvt_like(str(e)) for e in ix.ents]

    def penetrate(members):
        out = []
        for i in members:
            if not cvt[i]:
                continue
            for rmap in (ix.fwd, ix.rev):
                for adj in rmap.values():
                    for t2 in adj.get(i, ()):
                        if not cvt[t2]:
                            out.append(t2)
        return out

    frontier = [(seed, [])]
    seen_states = {seed}
    patterns = []
    for _depth in range(max_hops):
        nxt_layer = []
        for state, prefix in frontier:
            for r in range(len(ix.rels)):
                if not ix.head_mask[r] and not ix.tail_mask[r]:
                    continue
                for forward in (True, False):
                    if prefix and prefix[-1][0] == r \
                            and prefix[-1][1] == forward:
                        continue          # same-rel trivial cycle
                    t_state = ix.transition(state, r, forward)
                    if not t_state or t_state == state:
                        continue
                    supp = bin(state & (ix.head_mask[r] if forward
                                        else ix.tail_mask[r])).count("1")
                    npath = prefix + [(r, forward)]
                    if r in tset:
                        members = list(ix.bits(t_state))
                        ans = [i for i in members if not cvt[i]]
                        ans += penetrate(members)
                        patterns.append({
                            "rels": npath,
                            "answer": [str(ix.ents[i]) for i in ans[:40]],
                            "support": supp,
                        })
                    else:
                        if t_state in seen_states:
                            continue
                        seen_states.add(t_state)
                        nxt_layer.append((t_state, npath))
        if len(nxt_layer) > beam:
            nxt_layer.sort(key=lambda sp: -bin(sp[0]).count("1"))
            nxt_layer = nxt_layer[:beam]
        frontier = nxt_layer
        if not frontier:
            break
    if not patterns:
        return []
    # Ranking (user ruling 2026-09-12): terminal-HIT first, then length asc,
    # then name — support/fan-out is not a design criterion anywhere
    patterns.sort(key=lambda p: (len(p["rels"]), p["rels"]))
    for p in patterns[:topk_patterns]:
        p["witnesses"] = [
            ch for ch in (_materialize(ix, seed_names, p["rels"], a, name2idx)
                          for a in p["answer"][:witness_k * 2])
            if ch][:witness_k]
    return patterns[:topk_patterns]


def _materialize(ix, seed_names, rels_path, target_name, name2idx):
    """One concrete entity chain realizing the pattern (backward DFS)."""
    tgt = name2idx.get(str(target_name))
    if tgt is None:
        return None
    cur = [tgt]
    nodes = [tgt]
    for r, forward in reversed(rels_path):
        prev = []
        for c in cur:
            if forward:
                prev += ix.rev[r].get(c, [])
            else:
                prev += ix.fwd[r].get(c, [])
        if not prev:
            return None
        cur = [prev[0]]
        nodes.append(prev[0])
    seed_set = {str(s) for s in seed_names}
    if str(ix.ents[nodes[-1]]) not in seed_set:
        return None
    nodes.reverse()
    chain = []
    for k in range(len(rels_path)):
        h = nodes[k]
        t = nodes[k + 1] if k + 1 < len(nodes) else nodes[-1]
        r, forward = rels_path[k]
        if not forward:
            h, t = t, h
        chain.append((str(ix.ents[h]), str(ix.rels[r]), str(ix.ents[t])))
    return chain


def rank_display(ctx, patterns, rel_idxs):
    """Grouped display text (A+): terminal-relation groups ordered by the
    model's own submission order (rel_idxs), patterns within by (len, -supp)."""
    pos = {r: i for i, r in enumerate(rel_idxs)}

    def tkey(p):
        return (pos.get(p["rels"][-1][0], 999), len(p["rels"]), p["rels"])

    lines = []
    for p in sorted(patterns, key=tkey):
        arrow = " → ".join(
            ("›" if fwd else "‹") + str(ctx.rels[r]).rsplit(".", 2)[-1]
            for r, fwd in p["rels"])
        ans = " | ".join(p["answer"][:6])
        more = f" …(+{len(p['answer']) - 6})" if len(p["answer"]) > 6 else ""
        lines.append(f"  {arrow}  (support {p['support']}) → {ans}{more}")
    return "\n".join(lines)
