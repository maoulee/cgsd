"""TRIPLE LAYER (user core-principle ruling 2026-09-12): the RECONSTRUCTED
TRIPLE SET is each subgraph's core content. The full case graph is the base;
the logical paths obtained from the pattern walk are rebuilt into the
subgraph's triples; dedup / merge evaluation / CVT admission all operate on
those triples. Everything except plan parsing works on the walk's ACTUAL
outputs. This module owns that reconstruction — it knows treq/bres/ctx but
knows NOTHING about rows or display.

Output (the "store") is consumed exclusively by the row layer (seq_rows.py).
"""
from collections import defaultdict
from kgqa.agent.entity_kinds import is_cvt as _cvt



def _short_r(rname):
    return ".".join(str(rname).rsplit(".", 2)[-2:])


def _collapse_rt(rt):
    """Consecutive-duplicate collapse: a CVT mid-node repeats the relation
    across its two edges ((France,m,Belgium) walks (adjoins,adjoins)), so a
    walked 2-layer chain carries a 3-tuple while its pattern key is the
    2-tuple. Pattern identity = the DISTINCT relation sequence (user ruling
    2026-09-15: CVT traversal is an expansion detail, not a pattern hop)."""
    out = []
    for r in rt:
        if not out or out[-1] != r:
            out.append(r)
    return tuple(out)


def collect_pattern_triples(treq, bres, ctx, kept, edges, hop_dir,
                            mid_attr_pairs):
    """Rebuild the pattern-walk's logical paths into the subgraph's triples.

    kept/edges/hop_dir/mid_attr_pairs are render_v38_ack's collection-phase
    products (the walk's actual outputs — the only source of truth)."""
    groups = defaultdict(list)          # rels tuple -> [nodes tuple]
    for (key, _dirs), v in kept.items():
        for relsn in v["rels"]:
            groups[tuple(str(r) for r in relsn)].append(key)

    # submitted-terminal priority: the model's own submission order
    # (attr_expansion keys; rel_names fallback for echo-less full names)
    _sub_sh = {_short_r(k): i for i, k in
               enumerate((treq.get("attr_expansion") or {}).keys())}
    for _i, _rn in enumerate(treq.get("rel_names") or []):
        _sub_sh.setdefault(_short_r(str(_rn)), 50 + _i)
    _cn = {str(c) for c, _i in (treq["centers"] or [])}

    # the SUBMITTED relation's direct edges (CVT penetration edges ride
    # p.triples, not kept paths) join their 1-hop pattern
    _full_of_short = {}
    for _rn in (treq.get("rel_names") or []):
        _full_of_short.setdefault(_short_r(_rn), str(_rn))
    for _h, _r, _t in list(edges):
        _fr = _full_of_short.get(_r)
        if _fr is not None:
            groups.setdefault((_fr,), []).append((_h, _t))

    def _pkey(rt):
        insts = {k for k in groups[rt]}
        anchored = any(k[0] in _cn or k[-1] in _cn for k in insts)
        return (0 if _short_r(rt[-1]) in _sub_sh else 1,
                0 if anchored else 1,
                len(rt), rt)

    # ORDER walked patterns, never cut (semantic choice happened in B phase)
    _env_cap = 2
    # SECTION MEMBERSHIP ALIGNED WITH THE SELECTION (user ruling 2026-09-13,
    # Belgium specimen: 6 multi-hop sections for ONE submitted relation —
    # the per-relation top-3 quota must hold at display too). Only the
    # SELECTED patterns (treq.multistep keys — B phase, per-terminal-relation
    # quota) and the 1-hop SUBMITTED-direct patterns render as sections;
    # loose RPE-walked patterns go to the environment edge block below.
    _selected = set()
    for _ci, _pats in (treq.get("multistep") or {}).items():
        for _pk in _pats:
            _selected.add(tuple(str(ctx.rels[_r]) for _r in _pk))
    _selected_c = {_collapse_rt(k) for k in _selected}
    # ADMISSION TERMINALS (user audit 2026-09-19, 567 specimen: 39 patterns
    # rendered for 4 submitted relations — design cap is direct + 3 per
    # terminal). Two tightenings restore the designed count:
    #  (1) branch-(b) admission accepts only SUBMITTED relations as legal
    #      terminals — bridges ride the walk but may not mint render
    #      sections (rel_names carries them; attr_expansion keys are the
    #      model's own submission);
    #  (2) branch-(b) admits at most 3 walked tuples PER TERMINAL,
    #      mirroring the B-phase quota the selection already enforces.
    #      Overflow tuples drop through to the environment edge block.
    _ae_keys = list((treq.get("attr_expansion") or {}).keys())
    _submitted_names = ({str(k) for k in _ae_keys} or
                        {str(r) for r in (treq.get("rel_names") or [])})
    _admit_terms = ({k[-1] for k in _selected if k} | _submitted_names)
    # GLOBAL per-terminal budget: selected patterns SPEND the terminal's
    # 3-slot quota (user design: ≤3 multi-hop per terminal in TOTAL —
    # the B-phase quota is per (center, terminal), so a multi-center call
    # stacks; display enforces the per-terminal total). Pre-fill is itself
    # capped at 3 so over-quota selected sets still leave branch (a) room
    # to show the sorted-first representatives.
    ADMIT_PER_TERM = 3
    _admit_count = {}
    for k in _selected:
        if k and _admit_count.get(k[-1], 0) < ADMIT_PER_TERM:
            _admit_count[k[-1]] = _admit_count.get(k[-1], 0) + 1
    # RECONSTRUCTION LANE (user ruling 2026-09-19): treq["confirmed"] maps
    # each selected pattern key to its reconstructed THROUGH-chains; a
    # walked tuple renders iff it is one of those chains' ACTUAL rel
    # sequences (CVT pass-through hops included) — no name-form matching,
    # the mismatch that used to kill heterogeneous multi-hop sections.
    # Budget counts per selected KEY (B-phase quota already bounded the
    # key set; this is the display-side control point).
    _confirmed = treq.get("confirmed")
    _rt_of_key = {}
    if _confirmed:
        for (_ci, _names), _chains in _confirmed.items():
            _s = _rt_of_key.setdefault(_names, set())
            for _ch in _chains:
                _rel_t = tuple(str(ctx.rels[_r])
                               for (_h, _r, _t) in _ch["edges"])
                # FULL-DEPTH ONLY (user ruling 2026-09-23): a chain whose
                # edge sequence is SHORTER than the pattern's hop count is
                # an expansion remnant (hop landed on a CVT, passthrough
                # satisfied the last-hop "found" without instantiating the
                # submitted relation — 567: [award_winner⭢directed_by]'s
                # hop-1-only chain minted the bare `award_winner` label).
                # Remnants never become independent render groups.
                # LONGER sequences stay admitted (537 specimen, 2026-09-24):
                # the extra edges are MID-CHAIN CVT pass-throughs (root
                # --actor.film--> m.xxx --performance.film--> film
                # --netflix_id--> id = 3 edges for the 2-hop pattern) — the
                # bridge-legal instantiation the design demands. Trailing
                # passthroughs are already excluded upstream (_last_pat in
                # _rebuild_paths), so > hop-count length can ONLY be the
                # mid-chain bridge form; the equality used to zero the
                # whole continuation render (triples: (empty)).
                if _rel_t and len(_rel_t) >= len(_names):
                    _s.add(_rel_t)
    _key_count = {}
    _shown = []
    # TOTAL display budget: multi-center calls mint one key-set per center
    # (Tempus-Unbound specimen: 8 terminals × N centers = 125 sections).
    # The per-key ≤3 keeps each pattern tight; the TOTAL cap keeps the
    # whole render inside the selection-layer design count.
    ADMIT_TOTAL = 24
    _a_seen = set()      # collapse-key dedup: one raw representative per
    for rt in sorted(groups, key=_pkey):   # SELECTED pattern (variants of
        _rtc = _collapse_rt(rt)            # the same collapsed key merged)
        if _confirmed is not None:
            # reconstruction lane: render only the confirmed chains' actual
            # sequences of a selected key; 1-hop submitted-directs ride
            # their own confirmed entries ((rel,) keys)
            _owner = next((_k for _k, _rts in _rt_of_key.items()
                           if rt in _rts), None)
            if _owner is not None:
                if _rtc in _a_seen:
                    continue
                # RENDER=SELECTED (user directive 2026-09-24): every
                # SELECTED pattern renders; the old ADMIT_PER_TERM/
                # ADMIT_TOTAL caps made displayed ⊂ selected (65
                # 'outside-every-displayed-pattern' edges) — selection
                # alone decides, the render layer adds nothing.
                _a_seen.add(_rtc)
                _shown.append(rt)
                continue
            continue                    # not on any confirmed path → nothing
        # SELECTED-PATH RECONSTRUCTION (user ruling 2026-09-19: the render
        # layer RECONSTRUCTS the chosen pattern paths — it makes NO display
        # decisions of its own. The admission budget IS the selection-layer
        # control (it picks which complete paths qualify); walked-but-
        # unselected paths render NOTHING (no environment fallback). CVT
        # rendering mechanics are unchanged — they are answer-critical.)
        # ACTUAL-PATH GROUNDED (2026-09-15): the walked tuple is the ground
        # truth of a declared key — a collapse-equivalent walk of a selected
        # pattern IS that selected path's actual form.
        if rt in _selected or _rtc in _selected_c:
            if _rtc in _a_seen:
                continue            # collapse-variant of an already-shown
            _a_seen.add(_rtc)
            _shown.append(rt)
            continue
        if len(_rtc) == 1 and _short_r(_rtc[0]) in _sub_sh:
            _shown.append(rt)      # 1-hop submitted-direct (CVT mids collapse)
            continue
        # walked but NOT selected → not rendered (user ruling 2026-09-19)

    patterns = []
    shown_edges = set()
    ms_emitted = set()
    for rt in _shown:
        insts = sorted({k for k in groups[rt]})
        hops = []                       # [(rel_short, {h: [t...]})]
        attrs = []                      # [(hop_idx, (mid, key, value))]
        for i, r in enumerate(rt):
            sh = _short_r(r)
            # F7 (user ruling 2026-10-06): the IDENTITY key uses the FULL
            # relation name — _short_r collapses domain1.foo.r and
            # domain2.foo.r into the same key, merging distinct facts.
            # The short name stays as the display label in hops/labels.
            rid = str(r)
            tails_of_h = {}
            mids_here = []
            for key in insts:
                a, b = key[i], key[i + 1]
                d = hop_dir(r, a, b)
                h, t = (b, a) if d == "r" else (a, b)
                if (h, rid, t) not in shown_edges:
                    shown_edges.add((h, rid, t))
                    tails_of_h.setdefault(h, set()).add(t)
                for x in (h, t):
                    if _cvt(x) and x not in ms_emitted:
                        ms_emitted.add(x)
                        mids_here.append((x, {a.lower(), b.lower()}))
            # R3 (Codex audit v2): hops carry the FULL relation name as the
            # identity key — downstream (V4 unify_facts) uses this for
            # dedup. The short name stays in the pattern label only.
            hops.append((rid, {h: sorted(ts) for h, ts in tails_of_h.items()}))
            # SECTION-SCOPED attr exclusion: a key folds into attrs unless
            # its relation is part of THIS pattern (there it renders as hop
            # rows). Call-level exclusion severed cross-section disclosures —
            # the CVT admitted by administrative_division never showed its
            # date_adopted value next to the owning state (24353bbc).
            # TYPED (audit ③ specimen 537, 2026-09-21): the bare key is
            # excluded only when THIS hop actually renders the CVT→named
            # edge for it (a CVT-headed edge in tails_of_h). For a
            # named→CVT one-hop pattern (actor.film → performance mids) the
            # old bare-name rule excluded the CVT's OWN film attribute —
            # the answer surface — while no hop ever rendered it (Pacific
            # Rim vanished from the bracket, only character: survived).
            _thishop_cvt_headed = any(
                str(h)[:2] in ("m.", "g.") for h in tails_of_h)
            _rt_bare = {sh} if _thishop_cvt_headed else set()
            for mid, red in mids_here:
                for k, vs in mid_attr_pairs(mid, red, section_bare=_rt_bare):
                    for v in vs:
                        attrs.append((i, (mid, k, v)))
        # label = the ACTUAL walked sequence (doubled relations included —
        # the CVT in/out is part of the real pattern; collapsing the label
        # while rows show actual hops recreated the pattern/walk mismatch)
        patterns.append({"label": " ⭢ ".join(_short_r(r) for r in rt),
                         "hops": hops, "attrs": attrs, "n_inst": len(insts)})

    # environment: walked edges not on any rendered pattern
    _edges = set()
    for (key, _dirs), v in kept.items():
        for relsn in v["rels"]:
            for i in range(len(key) - 1):
                d = hop_dir(relsn[i], key[i], key[i + 1])
                _edges.add((key[i + 1], _short_r(relsn[i]), key[i]) if d == "r"
                           else (key[i], _short_r(relsn[i]), key[i + 1]))
    for e in edges:
        _edges.add(e)
    # WALKED-BUT-UNSELECTED edges render NOTHING (user ruling 2026-09-19:
    # only the chosen paths' entities, meeting the whole-path requirement,
    # are rendered — no environment fallback for unselected walks). The
    # key stays for interface compatibility; downstream renders it empty.
    env_triples = []

    return {"centers": [str(c) for c, _i in treq["centers"]],
            "frontier": [str(ctx.ents[i]) for i in (treq.get("cont_frontier") or {}).get(
                treq["centers"][0][1], []) if 0 <= i < len(ctx.ents)]
            if treq.get("cont_frontier") else [],
            "patterns": patterns, "env_triples": env_triples}
