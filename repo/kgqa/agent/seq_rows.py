"""ROW LAYER — entity-block primary with inline CVT brackets (user ruling
2026-09-16): entities are the primary grouping unit; pattern paths render
as an index line only. Within each entity block:
  * outgoing edges: tail-merged
  * incoming edges: head-merged (many→one)
  * CVT connections: INLINE brackets — m.xxx [key: value; key: value]
  * cross-block dedup: each (h, rel, t) edge appears exactly once
Two rules (user): ① related attributes together, ② no duplicate rendering.

R3-b (2026-10-01, SEQ_RENDER_V3, specs/r3b_render_pools):
  * F09 fix — the dedup/grouping key is the relation name AS CARRIED
    (two-segment short for pattern hops), not its lowercased LAST segment:
    two full relations sharing the last segment (film.actor / tv.actor)
    no longer merge into one row. Display strings and sort logic unchanged.
  * §6.6 — the ONLY legal display truncation is the merged edge group
    (same h+rel → >80 tails, or >80 heads → same rel+t): the group keeps
    the budget, the cut is a COUNTED event stashed on the store
    (`_display_stats`) for completeness['display'] — never silent. The
    also-in-evidence appendix keeps its zero-loss semantics; its name
    count rides the same stats.
"""
from collections import defaultdict
from kgqa.agent.entity_kinds import is_cvt as _cvt
from kgqa.agent.views import DISPLAY_GROUP_BUDGET

# NOTE CONSOLIDATED (user request 2026-09-20): the two per-message notes
# (block-formatting + triple-grammar) merged into ONE compact note emitted
# by seq_tools' result assembly; this renderer no longer appends its own.
_NOTE = ""

_TAIL_CAP = 40
_HEAD_CAP = 12
_VAL_CAP = 6
_GROUP_CAP = 10
_MULTI_MIN = 3

# ── FULL-NAME ENUMERATION (user ruling 2026-09-27b: folding is allowed,
# HIDING NAMES IS NOT — an entity hidden behind a `…(+N)` count can be THE
# answer, and the answer-legality ledger admits every walked entity; value
# tails too — 241-style discriminators are dates/numbers). Every former
# `…(+N)` tail now enumerates ALL names, pipe-separated and wrapped at the
# visual width. Boundedness: each row may spend a small wrap budget
# (_ENUM_WRAP_PER_ROW continuation lines); a row needing more spills its
# remainder to the always-complete `also-in-evidence` index — the index
# itself wraps unlimited (bounded by the distinct-name count, zero loss).
_ENUM_WIDTH = 110           # visual chars per wrapped line
_ENUM_WRAP_PER_ROW = 3      # wrapped lines one row's enumeration may spend


def _enum_lines_capped(items, first_len, cap):
    """Full enumeration wrapped at _ENUM_WIDTH. Returns (lines, rest):
    `lines` are pipe-joined segments (first one rides the row's own line
    after `first_len` prefix chars); when producing another line would
    exceed `cap`, the REMAINING items return as `rest` for the index."""
    lines, cur, cur_len = [], [], first_len
    for i, it in enumerate(items):
        add = (3 if cur else 0) + len(it)
        if cur and cur_len + add > _ENUM_WIDTH:
            if len(lines) + 1 >= cap:
                return (lines + [" | ".join(cur)]) if cur else lines, items[i:]
            lines.append(" | ".join(cur))
            cur, cur_len = [it], len(it)
        else:
            cur.append(it)
            cur_len += add
    if cur:
        lines.append(" | ".join(cur))
    return lines, []


def _is_value(n):
    s = str(n)
    if _cvt(s):
        return False
    return (len(s) <= 14 and any(c.isdigit() for c in s)
            and not any(c.isalpha() for c in s.replace("-", "").replace(":", "")))


def _short_rel(r):
    return str(r).rsplit(".", 1)[-1].lower()


def _cut_members(items, stats):
    """§6.6 display truncation: ONE merged edge group with more than
    DISPLAY_GROUP_BUDGET members keeps the budget. The kept set sorts by
    name when a cut happens (deterministic, the view contract's
    sorted[:budget]); the cut counts into `stats` — never silent."""
    if len(items) <= DISPLAY_GROUP_BUDGET:
        return items, 0
    dropped = len(items) - DISPLAY_GROUP_BUDGET
    stats["group_truncations"] += 1
    stats["truncated_members"] += dropped
    return sorted(items, key=str)[:DISPLAY_GROUP_BUDGET], dropped


def render_rows(store):
    # NO CENTER ECHO (user ruling 2026-09-23, correction): the call command
    # already carries the centers — repeating them (or the frontier) in the
    # result is duplication, not a budget/truncation matter. Root/layer
    # context is carried by anchor_sequence + the SEQUENCE EXTENSION note.
    L = []
    # R3-b faces: V3 gates the full-name identity keys (F09) and the >80
    # group cut; the stats dict rides the store back to _sg_finalize.
    _v3 = bool(store.get("_v3"))
    _stats = {"group_truncations": 0, "truncated_members": 0,
              "also_in_evidence": 0}

    def _rk(r):
        # relation IDENTITY key: the name as carried under V3 (F09 — the
        # bare last segment merged distinct relations); legacy last-segment
        # face otherwise (OFF = byte-identical output).
        return str(r) if _v3 else _short_rel(r)

    _attr_full = store.get("_attr_full") or {}   # (mid, key, v) -> full rel

    # ── flatten + dedup all edges ──
    _seen_e = set()
    all_edges = []
    pattern_labels = []
    for pat in store["patterns"]:
        pattern_labels.append(pat["label"])
        for sh, hop in pat["hops"]:
            for h, ts in hop.items():
                for t in ts:
                    k = (str(h), _rk(sh), str(t))
                    if k not in _seen_e:
                        _seen_e.add(k)
                        all_edges.append((h, sh, t))
    for h, r, t in store.get("env_triples", ()):
        k = (str(h), _rk(r), str(t))
        if k not in _seen_e:
            _seen_e.add(k)
            all_edges.append((h, r, t))

    # CVT attrs (deduped). Attr identity under V3 = the R05 FULL relation
    # when the materialization companion map carries it; the bracket still
    # DISPLAYS the short display_key (the same fact's display form).
    cvt_kv = defaultdict(lambda: defaultdict(set))
    _seen_a = set()
    for pat in store["patterns"]:
        for _hi, (mid, kk, v) in pat.get("attrs", ()):
            _fk = _short_rel(kk)
            if _v3:
                _fr = _attr_full.get((str(mid), str(kk), str(v)))
                if _fr:
                    _fk = str(_fr)
            k2 = (str(mid), _fk, str(v))
            if k2 not in _seen_a:
                _seen_a.add(k2)
                cvt_kv[mid][str(kk)].add(v)

    # pattern index (one line)
    if pattern_labels:
        L.append(f"▸ patterns: {' | '.join(pattern_labels)}")

    def _tail_str(t):
        if _cvt(t) and t in cvt_kv and cvt_kv[t]:
            pairs = []
            for k in sorted(cvt_kv[t]):
                vs = sorted(cvt_kv[t][k])
                # FULL enumeration (ruling 2026-09-27b): value tails are
                # discriminating answers (241: dates/rates) — never capped.
                pairs.append(f"{k}: {' | '.join(vs)}")
            return f"{t} [{'; '.join(pairs)}]"
        return str(t)

    # enumeration emitter: the row's first segment rides its own line; the
    # wrap spends the per-row budget; the remainder spills to the index.
    # `tail` carries the §6.6 truncation marker on the row's first line.
    _spill = set()          # names beyond any row budget -> also-in-evidence

    def _emit_enum(prefix, items, indent="        ", tail=""):
        if _v3:
            items, _dropped = _cut_members(items, _stats)
            if _dropped:
                tail = tail + f" …(+{_dropped} truncated)"
        lines, rest = _enum_lines_capped(
            [str(x) for x in items], len(prefix) + len(tail),
            _ENUM_WRAP_PER_ROW)
        L.append(prefix + (lines[0] if lines else "") + tail)
        for cont in lines[1:]:
            L.append(indent + cont)
        _spill.update(rest)

    # ── CVT owner: which entity connects to each CVT ──
    cvt_owner = {}
    for h, r, t in all_edges:
        if _cvt(t) and not _cvt(h) and not _is_value(h):
            cvt_owner.setdefault(t, h)
        elif _cvt(h) and not _cvt(t) and not _is_value(t):
            cvt_owner.setdefault(h, t)

    # DISCRIMINATOR VALUE LANE (audit ③ specimen 626, 2026-09-21): a
    # CVT --date/number--> value edge had NO render lane — ent_out/ent_in
    # require a non-CVT non-value endpoint on that side, and the context
    # tail dropped CVT heads, so walked dates/runtimes never reached the
    # screen (the model abstained "no date evidence" with the values in
    # the walk). Detected purely by VALUE SHAPE (_is_value) — never by the
    # model-declared answer type (user design). Rendered under the CVT's
    # owning entity block, merged per (cvt, relation).
    val_by_mid = defaultdict(lambda: defaultdict(list))
    for h, r, t in all_edges:
        if _cvt(h) and _is_value(t):
            val_by_mid[h][_rk(r)].append(t)

    # ── entity classification ──
    centers = set(store.get("centers", []))
    ent_out = defaultdict(list)
    ent_in = defaultdict(list)
    for h, r, t in all_edges:
        if not _cvt(h) and not _is_value(h):
            ent_out[h].append((h, r, t))
        if not _cvt(t) and not _is_value(t):
            ent_in[t].append((h, r, t))

    # block candidates: anchor + entities with CVT connections or high degree
    # CVT PARTNERS COUNT TOO (audit ③ specimen 241): the named ENDPOINT of
    # a CVT edge carries CVT-mediated evidence — without it, degree-1
    # environment entities (French arrondissement…) alphabetically starved
    # Monaco/Germany out of the 10-block budget while their edges existed.
    cvt_connected = set(cvt_owner.values())
    cvt_connected |= {t for h, r, t in all_edges
                      if _cvt(h) and not _cvt(t) and not _is_value(t)}
    all_ents = set(ent_out) | set(ent_in)
    scored = sorted(
        all_ents,
        key=lambda e: (
            0 if e in centers else 1,
            0 if e in cvt_connected else 1,
            -(len(ent_out.get(e, ())) + len(ent_in.get(e, ()))),
            e))
    block_ents = scored[:_GROUP_CAP]

    # ── render blocks with cross-block dedup ──
    rendered_edges = set()
    for ent in block_ents:
        out_e = ent_out.get(ent, [])
        in_e = ent_in.get(ent, [])
        has_new = any((str(h), _rk(r), str(t)) not in rendered_edges
                      for h, r, t in out_e + in_e)
        if not has_new:
            continue    # all edges already shown in earlier blocks — skip
        L.append(f"── {ent} ──")
        # outgoing: tail-merge
        by_hr = defaultdict(lambda: [None, []])
        for h, r, t in out_e:
            k = (str(h), _rk(r), str(t))
            if k in rendered_edges:
                continue
            rendered_edges.add(k)
            sk = (h, _rk(r))
            by_hr[sk][0] = r
            by_hr[sk][1].append(t)
        for (h, _), (r, ts) in sorted(by_hr.items()):
            # FULL enumeration (ruling 2026-09-27b): every tail name is
            # visible — wrapped, with over-budget remainder spilling to the
            # also-in-evidence index. No `…(+N)` hiding.
            _emit_enum(f"    --{r}--> ", [_tail_str(t) for t in ts])
        # incoming: skip shown, head-merge many→one
        by_rt = defaultdict(list)
        for h, r, t in in_e:
            k = (str(h), _rk(r), str(t))
            if k in rendered_edges:
                continue
            rendered_edges.add(k)
            by_rt[(_rk(r), t)].append((h, r))
        for (_, t), hrs in sorted(by_rt.items()):
            if len(hrs) >= _MULTI_MIN:
                # CVT heads carry their attrs bracket too (audit 2026-09-24,
                # Eleanor-1392 specimen): the incoming lane rendered bare
                # mids while the outgoing lane (_tail_str) bracketed them —
                # a CVT at a chain's SOURCE side (m.xxx --student--> person)
                # lost its institution/degree attributes entirely.
                hs_u = sorted(set(h for h, _ in hrs))
                # FULL enumeration of every head name (ruling 2026-09-27b)
                _suffix = f" --{hrs[0][1]}--> {t}"
                if _v3:
                    hs_u, _dropped = _cut_members(hs_u, _stats)
                else:
                    _dropped = 0
                _marker = (f" …(+{_dropped} truncated)" if _dropped else "")
                _lines, _rest = _enum_lines_capped(
                    [_tail_str(x) for x in hs_u],
                    len(_suffix) + 4 + len(_marker),
                    _ENUM_WRAP_PER_ROW)
                # HEAD-side wrap, POSITIONAL form (user ruling 2026-10-05):
                # continuation chunks are plain head lists; the arrow+tail
                # rides the LAST line — everything before an arrow reads as
                # heads, so no side marker is needed.
                if len(_lines) <= 1:
                    L.append(f"    {_lines[0] if _lines else ''}{_suffix}{_marker}")
                else:
                    L.append(f"    {_lines[0]}")
                    for _cont in _lines[1:-1]:
                        L.append(f"    {_cont}")
                    L.append(f"    {_lines[-1]}{_suffix}{_marker}")
                _spill.update(_rest)
            else:
                for h, r in sorted(hrs):
                    L.append(f"    {_tail_str(h)} --{r}--> {t}")
        # discriminator value rows for this block's owned CVTs
        for mid, ent2 in sorted(cvt_owner.items(), key=lambda kv: str(kv[0])):
            if ent2 != ent or mid not in val_by_mid:
                continue
            for sr, vs in sorted(val_by_mid[mid].items()):
                fresh = [v for v in vs
                         if (str(mid), sr, str(v)) not in rendered_edges]
                if not fresh:
                    continue
                for v in fresh:
                    rendered_edges.add((str(mid), sr, str(v)))
                # FULL value enumeration (ruling 2026-09-27b): values are
                # discriminating answers — never capped, never +N-hidden.
                _emit_enum(f"    {mid} --{sr}--> ",
                           sorted(set(fresh)))

    # OVERFLOW FOLD (user ruling 2026-09-27, 1797 specimen): _GROUP_CAP
    # bounds BLOCKS, never evidence — a group is display compression, and
    # overflow entities must not lose their edges. Entities beyond the
    # block budget get their unrendered edges as COMPACT MERGED ROWS
    # (super-leaf folding form, `A --rel--> t1 | t2 …(+N)`); when the row
    # budget exhausts, the remaining entity NAMES still ride a final index
    # row — the model binds names from rows (1797: Farragut et al. were
    # bound from merged rows), so a folded row preserves the answer
    # surface. Row count stays bounded (the cap's reason is context
    # budget): fold rows share the tail's budget; entity loss is zero.
    _TAIL_ROWS = _GROUP_CAP * 3
    _n_fold = 0
    _idx_names = []
    _overflow = scored[_GROUP_CAP:]

    def _edge_ok(h, r, t):
        # same admission as the tail below: CVT tails only with walked
        # attrs, value tails excluded from fold rows (their lanes are the
        # discriminator rows above + the tail).
        return ((str(h), _rk(r), str(t)) not in rendered_edges
                and (not _cvt(t) or (t in cvt_kv and cvt_kv[t]))
                and not _is_value(t))

    for _oent in _overflow:
        _outs = [(h, r, t) for h, r, t in ent_out.get(_oent, ())
                 if _edge_ok(h, r, t)]
        _ins = [(h, r, t) for h, r, t in ent_in.get(_oent, ())
                if _edge_ok(h, r, t) and not _cvt(h) and not _is_value(h)]
        if not _outs and not _ins:
            continue
        if _n_fold >= _TAIL_ROWS:
            _idx_names.append(_oent)
            continue
        _by_hr = defaultdict(list)
        for h, r, t in _outs:
            _by_hr[(h, _rk(r))].append((r, t))
        for (h, _), rts in sorted(_by_hr.items()):
            if _n_fold >= _TAIL_ROWS:
                _idx_names.append(_oent)
                break
            r0 = rts[0][0]
            ts_u = sorted(set(t for _, t in rts))
            for t in ts_u:
                rendered_edges.add((str(h), _rk(r0), str(t)))
            _emit_enum(f"    {_tail_str(h)} --{r0}--> ",
                       [_tail_str(t) for t in ts_u])
            _n_fold += 1
        _by_rt = defaultdict(list)
        for h, r, t in _ins:
            _by_rt[(_rk(r), t)].append(h)
        for (_, t), hs in sorted(_by_rt.items()):
            if _n_fold >= _TAIL_ROWS:
                _idx_names.append(_oent)
                break
            r0 = next(r for h, r, tt in _ins if tt == t)
            hs_u = sorted(set(hs))
            if _v3:
                hs_u, _dropped = _cut_members(hs_u, _stats)
            else:
                _dropped = 0
            for h in hs_u:
                rendered_edges.add((str(h), _rk(r0), str(t)))
            _sfx = f" --{r0}--> {_tail_str(t)}"
            _marker = (f" …(+{_dropped} truncated)" if _dropped else "")
            _lines, _rest = _enum_lines_capped(
                [_tail_str(x) for x in hs_u],
                len(_sfx) + 4 + len(_marker),
                _ENUM_WRAP_PER_ROW)
            # HEAD-side wrap, POSITIONAL form (see the incoming lane above).
            if len(_lines) <= 1:
                L.append(f"    {_lines[0] if _lines else ''}{_sfx}{_marker}")
            else:
                L.append(f"    {_lines[0]}")
                for _cont in _lines[1:-1]:
                    L.append(f"    {_cont}")
                L.append(f"    {_lines[-1]}{_sfx}{_marker}")
            _spill.update(_rest)
            _n_fold += 1
    # FALLBACK INDEX (last line of defense, ruling 2026-09-27/27b): every
    # budget-exhausted or over-wrap entity NAME rides here, FULLY
    # enumerated and wrapped — the index itself is never capped and never
    # hides behind +N (bounded by the distinct-name count, zero loss).
    _idx_all = sorted(dict.fromkeys(list(_idx_names) + sorted(_spill)))
    if _idx_all:
        _ilines, _ = _enum_lines_capped(_idx_all, len("    also-in-evidence: "),
                                        len(_idx_all) + 2)
        L.append(f"    also-in-evidence: {_ilines[0] if _ilines else ''}")
        for _cont in _ilines[1:]:
            L.append(f"        {_cont}")

    # context tail: remaining unshown edges (block entities' leftovers and
    # overflow edges that missed the fold budget). CVT-HEADED edges with a
    # named endpoint are ADMITTED (audit ③ specimen 241): Monaco's only
    # edge was CVT-headed and had no lane anywhere; the non-CVT endpoint is
    # a real walked entity, so the edge is evidence, not noise.
    # CVT-TAILED edges whose CVT carries WALKED ATTRS are admitted too
    # (user ruling 2026-09-24, 1812 specimen): a pure-literal terminal CVT
    # (Barbados --size_of_armed_forces--> g.xxx [number: 610]) is a legal
    # VALUE-ANSWER shape. When its owner missed the block budget (root-
    # collapsed off the center list; 21 centers filled all 10 blocks), the
    # old `not _cvt(t)` left the submitted relation's terminal hop with NO
    # lane — patterns row promised it, no edge row ever showed it. Bare
    # mids without attrs stay excluded (a bare id is noise, not evidence).
    ctx = [(h, r, t) for h, r, t in all_edges
           if (str(h), _rk(r), str(t)) not in rendered_edges
           and (not _cvt(t) or (t in cvt_kv and cvt_kv[t]))
           and not _is_value(t)]
    if ctx:
        _ctx_by = defaultdict(list)
        _budget = max(0, _TAIL_ROWS - _n_fold)
        for h, r, t in ctx[:_budget]:
            _ctx_by[(h, _rk(r))].append((r, t))
        for (h, _), rts in sorted(_ctx_by.items()):
            r0 = rts[0][0]
            ts_u = sorted(set(t for _, t in rts))
            # FULL enumeration (ruling 2026-09-27b); _tail_str keeps the
            # CVT attr bracket inline ([number: 610]) — the value-answer's
            # display form
            _emit_enum(f"    {_tail_str(h)} --{r0}--> ",
                       [_tail_str(t) for t in ts_u])
    # ownerless CVT discriminator values (no named owner block to ride on)
    for mid, srs in sorted(val_by_mid.items()):
        if mid in cvt_owner:
            continue
        for sr, vs in sorted(srs.items()):
            fresh = [v for v in vs
                     if (str(mid), sr, str(v)) not in rendered_edges]
            if not fresh:
                continue
            for v in fresh:
                rendered_edges.add((str(mid), sr, str(v)))
            _emit_enum(f"    {mid} --{sr}--> ",
                       [str(v) for v in sorted(set(fresh))])

    if _v3:
        # §6.6/F03: the display face's counted truncation events ride the
        # store back to _sg_finalize (completeness['display']); the
        # also-in-evidence count is a zero-loss stat, never a status flip.
        _stats["also_in_evidence"] = len(_idx_all)
        store["_display_stats"] = dict(_stats)
    if _NOTE:
        L.append(_NOTE)
    # BRIDGE-CONTEXT ROW LABELS (user ruling 2026-10-05): rows whose relation
    # reached the center only as an expansion bridge are continuation
    # context. Post-process: tag the relation label inside the arrow of any
    # row line whose relation name (full or short face) is a bridge. Set by
    # the V38 store builder from treq attr_expansion; absent = no bridges.
    _br = store.get("_bridge_shorts") or ()
    if _br:
        for _bn in _br:
            for _i, _ln in enumerate(L):
                if f"--{_bn}-->" in _ln:
                    L[_i] = _ln.replace(f"--{_bn}-->", f"--{_bn} (bridge ctx)-->")
                _bs = _short_rel(_bn)
                if _bs != _bn and f"--{_bs}-->" in L[_i]:
                    L[_i] = L[_i].replace(f"--{_bs}-->", f"--{_bs} (bridge ctx)-->")
    return "\n".join(L)
