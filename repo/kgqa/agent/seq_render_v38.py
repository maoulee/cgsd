"""V3.8 triple-form renderer (user + Codex design ①, 2026-09-02).

RETURN TO TRIPLE-FORM evidence. Rows are compressed triples and ONLY two
compression shapes are legal (user ruling):

    same h + r → many tails    `h --r--> t1 | t2 | t3`
    many heads + same r + t    `h1 | h2 | h3 --r--> t`

CVT expansion unchanged (attrs inline in brackets, graph-reconstructed).
The pattern-paths overview and the forced candidates line are DROPPED —
in triple form every entity in a row IS a usable candidate (the ack's
env-computed `candidates` field still carries the walk's roster).

Input discipline inherited from V37h (walk-faithful): tree_data paths with
base-name + per-hop-direction dedup, same-relation out-and-back loop
suppression, suffix continuation drop. Edges derive from the kept paths'
hops with TRUE storage orientation (h→t as the graph stores it), so each
relation's own triples stay distinct (member vs member_of are different
rows by construction).

Gate: SEQ_RENDER_V38=1"""
import os
import re
from kgqa.agent.entity_kinds import is_cvt as _cvt
from collections import defaultdict

_VAL_RE = re.compile(
    r"^\d{4}-\d{2}|^\d{2}:\d{2}|^[+-]?[\d.,]+$|^\d{4}s?$")
_NOISY_ATTR = {"type", "types", "instance", "instances", "notable_types",
               "topic_equivalent_webpage", "webpage", "mid", "guid",
               "key", "keys", "permission", "is_reviewed", "article",
               "description", "alias", "name"}



def _parse_node(s):
    s = str(s)
    if s.endswith("]") and ": [" in s:
        base, _, attr_txt = s.partition(": [")
        base = base.strip()
        if _cvt(base):
            return base, [a.strip() for a in attr_txt[:-1].split(",")
                          if a.strip()]
    return s, []


def render_v38_ack(treq, bres, ctx):
    from kgqa.agent.views import render_v3 as _render_v3
    _v3 = _render_v3()
    ents, rels = ctx.ents, ctx.rels
    rel_adj = defaultdict(set)
    for h, rl, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        rel_adj[rl].add((h, t))
    ent_idx = {}
    for j, e in enumerate(ents):
        ent_idx.setdefault(str(e), j)
    center_names = [e for e, _i in (treq.get("centers") or [])]
    sel_rels = {str(r) for r in (treq.get("rel_names") or [])}
    # R3-b (§6.8-2): CVT auto-expansion is part of MATERIALIZATION — the
    # case-graph attr pull runs at evidence-build time (_sg_finalize stashes
    # the PRE-merge case channel here) and this renderer PROJECTS it instead
    # of re-scanning the case graph. Render and materialization therefore
    # read ONE store; the walked-literal records ride their own value rows
    # (merging them into the render source would double-display). mids the
    # materialization never touched show no bracket (render ⊆ materialized).
    _attrs_mat = treq.get("_ev_case_attrs") if _v3 else None
    if _v3:
        # per-call scratch: _sg_finalize reads it for completeness['display']
        # (F03); None until a renderer fills it.
        ctx._render_display_stats = None

    # EDGE-LEVEL PRIOR DEDUP (user ruling 2026-09-23): paths stay whole —
    # path adjudication keeps the chain connectivity — but an edge whose
    # FACT was already delivered by a prior subgraph of this case renders no
    # row. Keys are direction-normalized + inverse-folded facts; the
    # comparison set is treq["prior"], the accumulation snapshot taken
    # BEFORE this call's own edges joined it (render runs after
    # _accumulate, so the live set would suppress this call's new edges
    # too).
    from kgqa.agent.seq_tools import _ctx_inverse_rels, _edge_fact_key
    _inv = _ctx_inverse_rels(ctx)
    _prior_facts = {_edge_fact_key(str(h), str(r), str(t), _inv)
                    for (h, r, t) in (treq.get("prior") or ())}

    def _edge_shown(hn, rn, tn):
        return _edge_fact_key(hn, rn, tn, _inv) in _prior_facts

    def hop_dir(rname, a_name, b_name):
        if rname in rels:
            rid = rels.index(rname)
            ai = ent_idx.get(a_name)
            bi = ent_idx.get(b_name)
            if ai is not None and bi is not None:
                if (ai, bi) in rel_adj.get(rid, ()):
                    return "f"
                if (bi, ai) in rel_adj.get(rid, ()):
                    return "r"
        return "?"

    # ── collect walked paths (V37h discipline: base names + per-hop
    # direction key; same-relation out-and-back paths dropped) ──
    raw = {}
    for (cname, cidx), pe in zip(treq["centers"], bres["pe_list"]):
        for pid, p in (pe.items() if isinstance(pe, dict) else []):
            td = (getattr(p, "tree_data", None) or {})
            for tp in (td.get("paths") or []):
                nodes = tp.get("nodes") or []
                if len(nodes) < 2:
                    continue
                key = tuple(_parse_node(n)[0] for n in nodes)
                relsn = tuple(str(r) for r in
                              (tp.get("relations") or [])[:len(nodes) - 1])
                if len(relsn) >= 2:
                    rd = {i: hop_dir(relsn[i], key[i], key[i + 1])
                          for i in range(len(relsn))}
                    # same-relation out-and-back = returns to a VISITED node.
                    # A CVT bridge legitimately walks one relation r-then-f
                    # (France←m.adjoins→Spain); killing any same-rel
                    # direction change severed every CVT-mediated chain
                    # (1278d3da: the adjoins⭢co2 composite never rendered).
                    if any(relsn[a] == relsn[b] and rd[a] != rd[b]
                           and key[a] == key[b + 1]
                           for a in range(len(relsn))
                           for b in range(a + 1, len(relsn))):
                        continue
                dirs = tuple(
                    hop_dir(relsn[i], key[i], key[i + 1])
                    if relsn[i] in sel_rels else "?"
                    for i in range(len(relsn)))
                v = raw.setdefault((key, dirs), {"rels": [], "attrs": None})
                if relsn not in v["rels"]:
                    v["rels"].append(relsn)
                if v["attrs"] is None:
                    v["attrs"] = [_parse_node(n)[1] for n in nodes]
    if not raw:
        return "(empty)"

    # COMPOSITE-FIRST (user ruling 2026-09-15): a MULTI-RELATION chain path
    # is the completed walk command — the question-relevant composite. Its
    # per-member fragments (the frontier plain steps' last-hop paths) are
    # CONTAINED inside the chain's hop rows and are the redundant copies.
    # The old suffix-suppression kept the fragment and dropped the chain
    # (1278d3da: adjoins⭢co2 never rendered because (Belgium,g) rendered
    # standalone). Inverted: drop a path only when it is a proper contiguous
    # subsequence of a LONGER path. Perf guard: fall back to the old rule on
    # path-count blowups.
    if len(raw) > 400:
        node_seqs = {k[0] for k in raw}
        kept = {k: v for k, v in raw.items()
                if not any(k[0][i:] in node_seqs for i in range(1, len(k[0])))}
    else:
        _seqs = [k[0] for k in raw]
        _seq_set = set(_seqs)

        def _fragment_of(seq):
            for o in _seq_set:
                if len(o) > len(seq) and any(
                        o[i:i + len(seq)] == seq
                        for i in range(1, len(o) - len(seq) + 1)):
                    return True
            return False
        kept = {k: v for k, v in raw.items() if not _fragment_of(k[0])}

    # ── derive EDGES with true storage orientation ──
    # EDGE DISPLAY SHORT NAME (user audit 2026-09-09): last TWO components
    # (type.attribute). The bare attribute collapsed semantically different
    # relations in one subgraph (division/facility/league/location all
    # rendered '--teams-->') — the model could not tell which relation a row
    # came from. Typed shorts disambiguate rows AND match the rr display's
    # group keys.
    def _short(rname: str) -> str:
        return ".".join(rname.rsplit(".", 2)[-2:])

    # PATH-CONSISTENT EVIDENCE (user ruling 2026-09-10, Belgium specimen):
    # tier-1 admits only hops on CENTER-ANCHORED chains — the abstract
    # pattern instantiated with the center (Belgium --containedby--> Europe).
    # Terminal edges of RPE bridge segments (Belgium →adjoin-CVT→ Luxembourg
    # --containedby→ …) and sibling expansions are still WALKED evidence but
    # are environment rows: they demote to "other walked relations" instead
    # of posing as the asked relation's answers.
    # GATED OFF by default (A/B: pathcons 0.6539 / pathcons2 0.6661 vs
    # speed5 0.6978 — the demoted environment rows are load-bearing for the
    # model's discrimination, same lesson as the pattern-walk verdict;
    # chain-shaped rendering is the untried alternative). SEQ_TIER_ANCHOR=1
    # enables.
    _tier_anchor = os.environ.get("SEQ_TIER_ANCHOR", "0") == "1"
    _center_names = {str(c) for c in center_names}

    def _anchored_hops(key):
        if key and key[0] in _center_names:
            return {(key[i], key[i + 1]) for i in range(len(key) - 1)}
        return set()

    anchored = set()                  # (h_name, t_name) on center-anchored paths

    def _anch_pair(a, b):
        if not _tier_anchor:
            return True
        return (a, b) in anchored or (b, a) in anchored

    edges = set()                       # (h_name, rel_short, t_name)
    for (key, _dirs), v in kept.items():
        anchored |= _anchored_hops(key)
        for relsn in v["rels"]:
            for i in range(len(key) - 1):
                rname = relsn[i] if i < len(relsn) else None
                if rname is None:
                    continue
                sh = _short(rname)
                d = hop_dir(rname, key[i], key[i + 1])
                if d == "r":
                    if not _edge_shown(key[i + 1], rname, key[i]):
                        edges.add((key[i + 1], sh, key[i]))
                        anchored.add((key[i + 1], key[i]))
                else:
                    if not _edge_shown(key[i], rname, key[i + 1]):
                        edges.add((key[i], sh, key[i + 1]))
                        anchored.add((key[i], key[i + 1]))

    # CVT PENETRATION + CENTER-PARENTED SIBLINGS are part of the center's
    # pattern instantiation (pathcons rollout: demoting them cost -4.4pp —
    # film names behind performance CVTs left tier-1). A penetration pair
    # (cvt, named) is anchored when the CVT sits on ANY walked path or is
    # reached directly from a center; a sibling row is anchored when its
    # head/parent is a center or a path node.
    _path_nodes = {n for (key, _d), _v in kept.items() for n in key}
    for (cname, cidx), pe in zip(treq["centers"], bres["pe_list"]):
        for pid, p in (pe.items() if isinstance(pe, dict) else []):
            for tr in (getattr(p, "triples", None) or []):
                if len(tr) != 3:
                    continue
                h, _r, t = str(tr[0]), str(tr[1]), str(tr[2])
                # tier-1 is gated by the SELECTED relations anyway, so
                # anchoring every CVT-headed walk triple cannot re-admit
                # unselected environment sections — it only keeps the
                # selected relations' CVT-mediated instantiations visible
                if _cvt(h) or h in _center_names:
                    anchored.add((h, t))
    # CVT PENETRATION EDGES from p.triples (user audit 2026-09-08: the walk
    # ENUMERATES CVT→named edges — 54 in the 25 specimen — but tree_data
    # paths are step-relation-constrained and only record anchor→CVT→anchor
    # round-trips; the named endpoints (film names, rosters) never join the
    # render. Add them from the walk's actual enumeration.)
    for (cname, cidx), pe in zip(treq["centers"], bres["pe_list"]):
        for pid, p in (pe.items() if isinstance(pe, dict) else []):
            for tr in (getattr(p, "triples", None) or []):
                if len(tr) != 3:
                    continue
                h, r, t = str(tr[0]), str(tr[1]), str(tr[2])
                if _cvt(h) and not _cvt(t):
                    if _edge_shown(h, r, t) or _edge_shown(t, r, h):
                        continue
                    sh = _short(r)
                    d = hop_dir(r, h, t)
                    if d == "r":
                        edges.add((t, sh, h))
                    else:
                        edges.add((h, sh, t))
                elif not _cvt(h) and _cvt(t):
                    if _edge_shown(h, r, t) or _edge_shown(t, r, h):
                        continue
                    sh = _short(r)
                    d = hop_dir(r, h, t)
                    if d == "r":
                        edges.add((t, sh, h))
                    else:
                        edges.add((h, sh, t))

    # ── CVT attrs (graph-reconstructed, decoration-independent) ──
    # SELECTED-relation CVT edges are EXCLUDED from attrs: they render as
    # rows instead (1171 specimen: film.dubbing_performance.character folded
    # into attrs made the selected relation invisible while bridge edges
    # showed — "relations and walk results misaligned")
    sel_shorts = {_short(r) for r in sel_rels}
    sel_bare = {r.rsplit(".", 1)[-1] for r in sel_rels}
    cvt_graph = defaultdict(list)
    cvt_graph_all = defaultdict(list)   # same WITHOUT the sel_bare exclusion
    for h, rl, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        hn = str(ents[h]) if 0 <= h < len(ents) else ""
        if not _cvt(hn):
            continue
        k = (str(rels[rl]).rsplit(".", 1)[-1]
             if 0 <= rl < len(rels) else "?")
        tv = str(ents[t]) if 0 <= t < len(ents) else str(t)
        if not _cvt(tv) and len(tv) < 60 and k not in _NOISY_ATTR:
            # COMPRESSION summary source: the SELECTED key (film=, for a
            # film-family row) is exactly the content the compressed row
            # must show — excluding it (the bracket rule) left the summary
            # with secondary keys (character:) that mislead at that position
            # (cvtinline3: 0.6570 with character-first summaries vs 0.6836
            # with the summary hidden entirely)
            cvt_graph_all[hn].append(f"{k}={tv}")
            if k not in sel_bare:
                cvt_graph[hn].append(f"{k}={tv}")

    # ── row building: the two legal compression shapes ──
    # direct entity↔entity rows
    direct = sorted(e for e in edges
                    if not _cvt(e[0]) and not _cvt(e[2]))
    # entity—CVT edges → record rows; CVT→entity edges whose relation was
    # SELECTED also become rows (see above) — only non-selected ones fold
    rec_out = defaultdict(set)          # (h, r) -> {cvt}
    rec_in = defaultdict(set)           # (r, t) -> {cvt}   (reverse shape)
    sel_cvt = defaultdict(set)          # (r, t) -> {cvt}   selected CVT edges
    for h, r, t in edges:
        if _cvt(t) and not _cvt(h):
            rec_out[(h, r)].add(t)
        elif _cvt(h) and not _cvt(t):
            if r in sel_shorts:
                sel_cvt[(r, t)].add(h)
            else:
                rec_in[(r, t)].add(h)

    # CVT ATTR-KEY TOP-K (user design 2026-09-08): the question→key GTE
    # ranking pre-computed in _sg_execute selects which attributes each CVT
    # bracket shows — top-K keys only, the rest suppressed (avg 96 keys/
    # case would flood every bracket; GTE top-3 covers 50% of gold keys vs
    # random 5%). Fallback when no ranking: show all (cap 6, old behavior).
    _krank = getattr(ctx, "_cvt_key_rank", None)
    _kranked_keys = _krank[1] if _krank and _krank[1] else None
    _K = int(os.environ.get("SEQ_CVT_ATTR_TOPK", "5") or 0)  # 0 = off; 3->5 (user 2026-09-11): GTE ranks discriminator keys (initial_release_date #3 for date questions) just outside top-3
    _submitted_components = set()
    for _sr in (treq.get("rel_names") or []):
        for _seg in str(_sr).split("."):
            if _seg:
                _submitted_components.add(_seg)

    def cvt_disp(mid, red):
        # BARE MID (user ruling 2026-09-12, unified-triple design): the CVT's
        # attributes are its domain — they render as their own triple rows
        # right after the pattern row that surfaced the mid (see
        # _emit_attr_rows), not as an inline bracket. Bracket content that
        # duplicated compressed sections is gone by construction.
        return mid

    # PER-CVT ATTRIBUTE ADMISSION (spec pillar 4b, user ruling 2026-09-12):
    # the key set is THIS CVT's own keys (97% of CVTs have <=5); the case-level
    # GTE ranking (_kranked_keys) serves only as the keys' RELATIVE order,
    # never as a global admission quota — the 09-08 case-level top-5 gate
    # blanked brackets whenever a CVT's only informative key ranked 6+
    # (Eleanor specimen: institution #7 of 143). Cap applies ONLY to CVTs
    # with more keys than the cap (3% of CVTs).
    def _mid_attr_pairs(mid, red, section_bare=frozenset()):
        kv = defaultdict(set)
        if _attrs_mat is not None:
            # R3-b: the materialized store is the source (§6.8-2); the
            # selection below (red / noisy / cap / ranking / section_bare)
            # is this layer's DISPLAY projection of it, unchanged.
            for k, vs in ((_attrs_mat.get(mid) or {}).items()):
                for v in vs:
                    if (not v or v.lower() in red or k in _NOISY_ATTR
                            or len(v) >= 60):
                        continue
                    kv[k].add(v)
        else:
            for a in cvt_graph_all.get(mid, ()):
                if "=" not in a:
                    continue
                k, v = a.split("=", 1)
                k, v = k.strip(), v.strip()
                if not v or v.lower() in red or k in _NOISY_ATTR or len(v) >= 60:
                    continue
                kv[k].add(v)
        if not kv:
            return []
        keys = [k for k in kv if k not in section_bare]
        if not keys:
            return []
        if len(keys) > _K:
            _ord = {k: i for i, k in enumerate(_kranked_keys or [])}
            keys.sort(key=lambda k: (k not in _submitted_components,
                                     _ord.get(k, 99), k))
            keys = keys[:_K]
        # TARGET-HIT KEYS FIRST (pillar 4b)
        keys.sort(key=lambda k: k not in _submitted_components)
        return [(k, sorted(kv[k])) for k in keys]

    _section_mids = defaultdict(set)     # section rel -> mids to expand
    _mid_red = {}                        # mid -> row-endpoint names to suppress
    _mid_done = set()                    # mids whose attrs live in a
                                         # compression summary already
    _attr_emitted = set()
    _mid_sec = {}                        # mid -> registered section

    def _reg_mid(mid, section_r, red):
        if not _cvt(mid):
            return
        cur = _mid_sec.get(mid)
        # a SELECTED section owns the mid's domain rows (pattern-related
        # content stays with the pattern): it overrides an earlier
        # environment-section registration (person.education rec_out rows
        # register before sel_cvt does)
        if cur is not None and not (cur not in sel_shorts
                                    and section_r in sel_shorts):
            return
        if cur is not None:
            _section_mids[cur].discard(mid)
        _mid_sec[mid] = section_r
        _mid_red[mid] = {x.lower() for x in red}
        _section_mids[section_r].add(mid)

    def _emit_attr_rows(r):
        """The CVT domain as triple rows (user design 2026-09-12): each
        admitted attribute edge renders `mids --key--> value`; same value
        across many mids merges the HEAD side (pillar 4c compression), the
        mids whose attributes a tail-compression summary already carries are
        skipped (no duplicated display after compression)."""
        kv_groups = defaultdict(lambda: defaultdict(set))
        # SECTION-SCOPED exclusion (user ruling 2026-09-15, 24353bbc
        # specimen): the sel_bare exclusion used to be CALL-level, so a CVT
        # admitted by section A (administrative_division) never disclosed a
        # key because the relation owning it was selected for section B —
        # Nebraska-m.04st86j-date_adopted is a perfect path but its value
        # never appeared next to the association row. Exclusion applies only
        # where the key's relation IS this section (there it renders as rows).
        _sec_bare = {str(r).rsplit(".", 1)[-1]}
        for mid in sorted(_section_mids.get(r, ())):
            if mid in _mid_done or mid in _attr_emitted:
                continue
            pairs = _mid_attr_pairs(mid, _mid_red.get(mid, ()),
                                    section_bare=_sec_bare)
            if not pairs:
                continue
            _attr_emitted.add(mid)
            for k, vs in pairs:
                for v in vs:
                    kv_groups[k][v].add(mid)
        if not kv_groups:
            return []
        _ord = {k: i for i, k in enumerate(_kranked_keys or [])}
        out = []
        for k in sorted(kv_groups, key=lambda k: (k not in _submitted_components,
                                                  _ord.get(k, 99), k)):
            for v in sorted(kv_groups[k]):
                ms = sorted(kv_groups[k][v])
                head = " | ".join(ms[:6]) + (f" …(+{len(ms)-6} mids)"
                                             if len(ms) > 6 else "")
                out.append(f"{head} --{k}--> {v}")
        return out[:20]

    lines = []

    # ── CVT-TAIL COMPRESSION (user approval 2026-09-09) ─────────────────────
    # A row with MANY CVT tails (Miley --actor.film--> 18 bare mids) is noise:
    # mids are never answers, and per-mid brackets repeat the overlapping
    # values 18× (actor=Miley Cyrus in every record). For (h,r) rows with ≥4
    # CVT tails, collapse the tail set to its GTE-ranked attribute content —
    # per KEY: the DISTINCT values (question-ranked keys first, submitted
    # components next); a key with ONE distinct value folds to `k=v (×n)`.
    # A/B (user design 2026-09-09): SEQ_CVT_STYLE=inline puts the collapsed
    # form IN the row; =block leaves a pointer in the row and appends an
    # `event attributes` section after the relation sections.
    _CVT_COMPRESS_MIN = 4

    def _cvt_tail_keys(mids, red):
        # per-key distinct values across the tail set (top-K filtered, same
        # discipline as cvt_disp), keys ordered: GTE rank, then the rest.
        # Reads cvt_graph_all (V3: the materialized store — §6.8-2) — the
        # compression summary needs the SELECTED family's own key first
        # (see cvt_graph_all build note).
        kv = defaultdict(set)
        for mid in mids:
            if _attrs_mat is not None:
                for k, vs in ((_attrs_mat.get(mid) or {}).items()):
                    for v in vs:
                        if (not v or v.lower() in red or k in _NOISY_ATTR
                                or len(v) >= 60):
                            continue
                        kv[k].add(v)
                continue
            for a in cvt_graph_all.get(mid, ()):
                if "=" not in a:
                    continue
                k, v = a.split("=", 1)
                k, v = k.strip(), v.strip()
                if not v or v.lower() in red or k in _NOISY_ATTR:
                    continue
                if len(v) >= 60:
                    continue
                kv[k].add(v)
        if not kv:
            return []
        if _kranked_keys and _K > 0:
            _topk = set(_kranked_keys[:_K]) | _submitted_components
            kv = {k: vs for k, vs in kv.items() if k in _topk} or kv
        order = {k: i for i, k in enumerate(_kranked_keys or [])}

        def _kord(k):
            # submitted components rank right after the GTE top keys
            return (0, order.get(k, 99)) if k in order else (1, k)
        return sorted(kv.items(), key=lambda kv_: _kord(kv_[0]))

    def _cvt_tail_summary(h, r, mids):
        red = {h.lower()}
        kv = _cvt_tail_keys(mids, red)
        if not kv:
            return ""
        # FAMILY-KEY CAP (user ruling 2026-09-11, 567/25 specimens): the key
        # matching the SUBMITTED relation family (film: for a film question)
        # is the answer surface — the old flat 8-value cap cut it exactly
        # like the pre-compression mid floods did (The Journey, sorted at T,
        # invisible in 3/3 samples). Family keys get the leaf discipline
        # (SEQ_CVT_VALCAP, default 40 = roster parity); secondary keys
        # stay at 8.
        _sub_last = {str(x).rsplit(".", 1)[-1]
                     for x in (sel_rels or ()) if x}
        _vcap_fam = int(os.environ.get("SEQ_CVT_VALCAP", "40") or 40)
        parts = []
        for k, vs in kv[:3]:
            vs = sorted(vs)
            if len(vs) == 1:
                # one distinct value across the whole tail set (overlap) —
                # fold to k=v (×n) instead of echoing it per event
                parts.append(f"{k}={next(iter(vs))} (×{len(mids)})")
            else:
                cap = _vcap_fam if k in _sub_last else 8
                shown = " | ".join(vs[:cap])
                more = f" …(+{len(vs) - cap})" if len(vs) > cap else ""
                parts.append(f"{k}: {shown}{more}")
        return f"{len(mids)} event records · " + " · ".join(parts)

    _cvt_block = []               # block-mode deferred summaries
    _style = os.environ.get("SEQ_CVT_STYLE", "inline").strip().lower()
    n = 0                         # synthetic-entry counter for compressed rows

    if treq.get("multistep"):
        # TWO-LAYER RENDER (user ruling 2026-09-12): the triple layer
        # rebuilds the walk's logical paths into the subgraph's triples
        # (dedup/merge evaluation live there); the row layer formats rows
        # and accepts ONLY the triple store. UNIFORM ROUTING (2026-09-13,
        # supersedes SEQ_UNICHAIN): the DIRECT relation is a first-class
        # 1-hop pattern in the derivation (key (None, fam)) — multistep is
        # non-empty whenever direct support exists, so no call falls to the
        # legacy renderer for lack of 2-hop patterns; the UNICHAIN gate is
        # retired.
        # The legacy CVT-tail compression's -4pp context loss is addressed
        # by the pattern framework itself (direct-first + deferral), not by
        # a renderer split.
        from kgqa.agent.seq_triples import collect_pattern_triples
        from kgqa.agent.seq_rows import render_rows
        _store = collect_pattern_triples(
            treq, bres, ctx, kept, edges, hop_dir, _mid_attr_pairs)
        # BRIDGE-CONTEXT labels (user ruling: default ON) — compute the
        # bridge short-names BEFORE any arm returns; both the V38 rows face
        # and the V4 arm consume them (deep-audit fix: the V4 return used
        # to execute before this was set, so the arm could never tag).
        if os.environ.get("SEQ_BRIDGE_LABEL", "1") != "0":
            _store["_bridge_shorts"] = {
                _short(b) for _exp in (treq.get("attr_expansion") or {}).values()
                for b in (_exp.get("bridge") or [])}
        # V4 PARALLEL ARM (user-approved triple-form display, 2026-10-05):
        # same store snapshot, triple rows + one-side merge; acceptance =
        # expand(render(F)) == F. Gated; V38 rows face unchanged.
        if os.environ.get("SEQ_RENDER_V4", "0") == "1":
            from kgqa.agent.seq_render_v4 import render_v4
            if _v3:
                _store["_v3"] = True
                _store["_attr_full"] = {
                    _k: (_v[-1] if isinstance(_v, (list, tuple)) else _v)
                    for _k, _v in (treq.get("_ev_attr_full_rels") or {}).items()}
            _out = render_v4(_store, str(getattr(ctx, "question", "") or ""))
            ctx._render_display_stats = _store.get("_display_stats")
            # compressed groups ride ctx for answer-face auto-completion;
            # _v4_render_call stamps THIS call so the adapter publishes only
            # groups this render actually produced (Codex F4: a non-V4 call
            # must neither resurrect stale groups nor clear a fresher one).
            ctx._v4_compressed_groups = _store.get("_compressed_groups") or []
            _ev = getattr(ctx, "_adaptive_event", None)
            ctx._v4_render_call = (_ev or {}).get("call_id") or "v4"
            return _out
        if _v3:
            # R3-b row-layer faces: full-name identity keys (F09) + the
            # §6.6 >80 group cut; the attr full-rel companion resolves the
            # attr dedup identity (R05) while brackets keep the display_key.
            # V3-04: the companion map is MULTI-VALUED at the ledger layer
            # (every complete relation behind a short key is its own fact);
            # the DISPLAY face keeps one representative per row — the
            # last-scanned full rel, exactly the pre-V3 single value, so
            # render bytes do not move with the ledger fix.
            _store["_v3"] = True
            _store["_attr_full"] = {
                _k: (_v[-1] if isinstance(_v, (list, tuple)) else _v)
                for _k, _v in (treq.get("_ev_attr_full_rels") or {}).items()}
        _out = render_rows(_store)
        if _v3 and _store.get("_display_stats") is not None:
            ctx._render_display_stats = _store["_display_stats"]
        return _out

    # shape 1: same h + r → many tails (direct + records share the row space)
    tails_of = defaultdict(dict)        # (h, r) -> {t: display}
    for h, r, t in direct:
        tails_of[(h, r)][t] = t
    for (h, r), mids in rec_out.items():
        _mset = sorted(m for m in mids if _cvt(m))
        if len(_mset) >= _CVT_COMPRESS_MIN:
            _sum = _cvt_tail_summary(h, r, _mset)
            if _sum:
                # the whole mid tail set becomes ONE entry (repeating the
                # summary per mid would echo it 18×) — named direct tails
                # in the same (h, r) row are untouched
                for mid in mids:
                    tails_of[(h, r)].pop(mid, None)
                    _mid_done.add(mid)      # attrs live in the summary
                if _style == "block":
                    _cvt_block.append((r, h, _sum))
                    tails_of[(h, r)][f"__evt{n}"] = f"(↓ {len(mids)} event records)"
                else:
                    tails_of[(h, r)][f"__evt{n}"] = f"({_sum})"
                n += 1
                continue
        for mid in mids:
            tails_of[(h, r)][mid] = cvt_disp(mid, {h.lower()})
            _reg_mid(mid, r, {h})
    # shape 2: singletons regroup many heads → same r + t. Synthetic
    # compressed entries (__evt*) must NOT enter the regroup — it keys by
    # the TAIL NAME, which would print the raw key instead of the summary;
    # they render directly as one row.
    rows = []
    single = [(h, r, t) for (h, r), ts in tails_of.items() for t in ts
              if len(ts) == 1 and not t.startswith("__evt")]
    for (h, r), ts in tails_of.items():
        if len(ts) == 1:
            t = next(iter(ts))
            if t.startswith("__evt"):
                rows.append((r, f"{h} --{r}--> {ts[t]}", _anch_pair(h, t), 1))
    multi = {(h, r): ts for (h, r), ts in tails_of.items() if len(ts) >= 2}
    heads_of = defaultdict(set)         # (r, t) -> {h}
    for h, r, t in single:
        heads_of[(r, t)].add(h)

    for (h, r) in sorted(multi, key=lambda k: (k[1], k[0])):
        # join the DISPLAY values (CVT tails carry inline attrs) — joining
        # the dict keys printed bare mids and hid every measurement value
        # (co2/population specimens: 100+ bare mids, values unreachable).
        # STRUCTURE-PRESERVING fold (user ruling 2026-09-03 + measurement):
        # entities are core and NEVER dropped; only row STRUCTURE compresses
        # — long tails keep the first records verbatim, the remainder fold
        # to their full attr pairs grouped per record (`;` within, `|`
        # between). No caps on names.
        disp_list = [d for _t, d in sorted(multi[(h, r)].items())]
        joined = " | ".join(disp_list)
        # DISPLAY TRUNCATION RULING (user 2026-10-01, proposal §6.6 — D3-3
        # anchor, registration only): the ONLY legal truncation point is the
        # merged edge GROUP itself (>80 tails on one `h --rel-->` row, or
        # >80 heads on one `--rel--> t` row). The char fold below is
        # structure-preserving COMPRESSION (records regrouped, never dropped)
        # and stays; any future cut must land as the >80 group truncation,
        # not as a new cap here.
        if len(joined) > 1200:
            head_ds, used = [], 0
            for dd in disp_list:
                if used + len(dd) > 600 and head_ds:
                    break
                head_ds.append(dd)
                used += len(dd) + 3
            rest = disp_list[len(head_ds):]
            recs = ["; ".join(re.findall(r"[a-zA-Z_]+=[^;\]]+", dd))
                    for dd in rest]
            recs = [x for x in recs if x] or rest
            joined = (" | ".join(head_ds)
                      + f" (+{len(rest)}: {' | '.join(recs)}"
                      + ")")
        rows.append((r, f"{h} --{r}--> {joined}",
                     any(_anch_pair(h, _t) for _t in multi[(h, r)]), 1))
    for (r, t) in sorted(heads_of, key=lambda k: (k[0], k[1])):
        hs = sorted(heads_of[(r, t)])
        disp = cvt_disp(t, {x.lower() for x in hs}) if _cvt(t) else t
        rows.append((r, f"{' | '.join(hs)} --{r}--> {disp}",
                     any(_anch_pair(h, t) for h in hs), 1))
    # SELECTED CVT→entity rows: `m.xxx [attrs] --selected_rel--> entity` —
    # every relation the model asked for stays visible as a triple
    for (r, t) in sorted(sel_cvt):
        disps = [cvt_disp(m, {t.lower()}) for m in sorted(sel_cvt[(r, t)])]
        for m in sorted(sel_cvt[(r, t)]):
            _reg_mid(m, r, {t})
        rows.append((r, f"{' | '.join(disps)} --{r}--> {t}",
                     any(_anch_pair(m, t) for m in sel_cvt[(r, t)]), 1))
    # record reverse-shape rows (CVT→entity edges that stayed as records)
    seen_recs = {mid for mids in rec_out.values() for mid in mids}
    seen_recs |= {mid for mids in sel_cvt.values() for mid in mids}
    for (r, t), mids in rec_in.items():
        extra = {m for m in mids if m not in seen_recs}
        if not extra:
            continue
        for m in sorted(extra):
            _reg_mid(m, r, {t})
            rows.append((r, f"{cvt_disp(m, {t.lower()})} --{r}--> {t}",
                         _anch_pair(m, t), 1))

    # RELATION-SECTIONED DISPLAY (user design 2026-09-02) + PATH
    # CONSISTENCY (user ruling 2026-09-10): tier-1 admits only rows on
    # CENTER-ANCHORED chains (the abstract pattern instantiated with the
    # center); environment rows — RPE bridge-segment terminals, sibling
    # expansions — all demote to "other walked relations", never posing as
    # the asked relation's answers.
    # Leftover FULL-PATH CHAINS whose terminal edge never became a singleton
    # flat row (merged into multi-tail rows or dropped by row shapes) still
    # render — capped per relation to bound context growth.
    by_rel = defaultdict(list)
    for r, row, anch, hops in rows:
        by_rel[r].append((row, anch, hops))

    def roster(r):
        tails, heads = set(), set()
        for h, rr, t in edges:
            if rr == r and not _cvt(t) and not _cvt(h):
                tails.add(t)
                heads.add(h)
        side = tails if len(tails) >= len(heads) else heads
        ns = sorted(side)
        if not ns:
            return ""
        # FULL enumeration (ruling 2026-09-27b): candidates are named
        # entities — never hidden behind +N; wrapped at the visual width.
        from kgqa.agent.seq_rows import _enum_lines_capped
        lines, _rest = _enum_lines_capped(
            ns, len("▸ --") + len(r) + len("-->  (retrieved · candidates: "),
            len(ns) + 2)
        return ("\n           ".join(lines))

    L = [f"entities: {' | '.join(center_names)}"]
    sel_shorts = {_short(r) for r in sel_rels}
    # LENGTH-FIRST SECTION ORDER (user ruling 2026-09-12): sections whose
    # evidence is 1-hop from the center come first, then bridged/pattern
    # sections by their minimum path length — submission order only
    # tie-breaks. Within a section rows sort by hop count (short paths
    # first), preserving the build order for equal lengths. The tie-break
    # uses the MODEL's own submission order (attr_expansion keys); bridge
    # relations expanded into rel_idxs sort after every submitted name.
    _sub_order = {}
    for _i, _rn in enumerate((treq.get("attr_expansion") or {}).keys()):
        _sub_order.setdefault(_short(str(_rn)), _i)
    for _i, _rn in enumerate(treq.get("rel_names") or []):
        _sub_order.setdefault(_short(str(_rn)), 50 + _i)
    _min_hops = {r: min(h for _row, _a, h in v) for r, v in by_rel.items()}
    # BRIDGE SECTION LABELING (user audit 2026-09-11, Ron-Howard awards
    # specimen): relations that entered rel_idxs as BRIDGES (behind-CVT
    # carrier hits — co-endpoints of a CVT carrying the family, e.g. an
    # award CVT whose co-nominee is a producer) render their tier-1 section
    # as bridge CONTEXT, not as the asked relation. Rows unchanged (A/B
    # history: removing/demoting them cost -4.4/-3.2pp — they carry
    # discrimination evidence); only the label tells the model these
    # sections were not its selection.
    _bridge_shorts = {_short(b) for _exp in (treq.get("attr_expansion") or {}).values()
                      for b in (_exp.get("bridge") or [])}
    _row_seq = {}
    for r in by_rel:
        _row_seq[r] = {row: i for i, (row, _a, _h) in enumerate(by_rel[r])}
    for r in sorted((r for r in by_rel if r in sel_shorts),
                    key=lambda r: (_min_hops[r], _sub_order.get(r, 99), r)):
        tier = sorted(((row, h) for row, anch, h in by_rel[r] if anch),
                      key=lambda rh: (rh[1], _row_seq[r].get(rh[0], 0)))
        tier_rows = [row for row, _h in tier]
        if not tier_rows:
            continue         # selected but NO center-anchored instantiation
        ros = roster(r)
        # DEFAULT ON (user ruling 2026-10-05: bridge carriers are continuation
        # context, not new centers — the core design makes intermediate
        # relations auto-bridged and center-independent, and the label states
        # that in the render). The old A/B (2026-09-11, pre-P8 stack: hit
        # 82.6% -> 79.2%) gated it off; the pattern-centric C-stack teaches
        # bridge semantics explicitly, so the verdict is re-validated per
        # batch. SEQ_BRIDGE_LABEL=0 restores the old unlabeled face.
        _bl = (r in _bridge_shorts
               and os.environ.get("SEQ_BRIDGE_LABEL", "1") != "0")
        tag = ("bridge context — reached via your selected family's carriers"
               if _bl else "retrieved")
        L.append(f"▸ --{r}-->  ({tag}"
                 + (f" · candidates: {ros}" if ros else "") + ")")
        L.extend(f"    {row}" for row in tier_rows)
        # the CVT domain as its own triple rows — pattern-related content
        # stays in the section that surfaced the mids (user 2026-09-12)
        for _ar in _emit_attr_rows(r):
            L.append(f"    {_ar}")
    L.append("▸ other walked relations (environment — not center-anchored paths):")
    # MID-HOP VISIBILITY (user audit 2026-09-12, Belgium specimen — logical
    # chain break): a 2-hop witness decomposed into triples leaves its mid
    # edge in a section whose relation was never submitted; with
    # tier-anchoring off every row is "anchored", so the old anch=False-only
    # environment section rendered NOTHING and the mid hop vanished entirely
    # — the terminal `Europe --time_zones--> GMT` showed with no visible
    # Europe↔Belgium connection. Non-selected sections now render here
    # (their full row set, capped per section): the connective tissue of
    # multi-hop witnesses stays visible as triples.
    _ENV_ROWS_PER_REL = 6
    for r in sorted(by_rel):
        if r in sel_shorts:
            env_rows = [row for row, anch, _h in by_rel[r] if not anch]
        else:
            env_rows = [row for row, _a, _h in by_rel[r]]
        for row in env_rows[:_ENV_ROWS_PER_REL]:
            L.append(f"    {row}")
    if _cvt_block:
        # BLOCK MODE (SEQ_CVT_STYLE=block): the compressed attribute content
        # of large CVT-tail rows lives here, one line per (relation, head)
        L.append("▸ event attributes (compressed from the ↓-marked rows):")
        for r, h, s in _cvt_block:
            L.append(f"    {h} --{r}--> {s}")
    return "\n".join(L)
