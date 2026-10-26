"""V4 evidence renderer — triple-form display (user-approved design 2026-10-05).

Every fact renders as an exact triple; ordering gathers related facts; ONE-
SIDED merging compresses repeated text. No entity headers, no headerless
folds, no name-only index: connections live in the rows themselves.

Pipeline (design §9):
    facts   = unify(store)                      # exact identity dedup
    ordered = order(facts, centers)             # deterministic BFS, CVT attrs adjacent
    rows    = compress_adjacent_one_side(ordered)
    text, coverage = format_rows(rows)

Acceptance (design §10.1): expand(render(F)) == F and each fact covered by
exactly one row. Gated by SEQ_RENDER_V4=1 (parallel arm; V38 unchanged).
"""
from __future__ import annotations

import os
from collections import defaultdict, deque

_WIDTH = 110            # wrap width for enumerations
_ENUM_PER_LINE_MIN = 3  # don't split an enumeration below this many members
_COMPRESS_AT = 40       # user ruling 2026-10-05: the ONLY budget point —
                        # pattern filtering already happened upstream; a
                        # merged side beyond this many members compresses,
                        # and a whole-batch answer submission auto-completes
                        # the omitted members (answer-face injection).


def _esc(name: str) -> str:
    """Escape syntax characters (|, arrow, newline/CR) — JSON quoting."""
    s = str(name)
    if "|" in s or "-->" in s or "\n" in s or "\r" in s:
        import json
        return json.dumps(s, ensure_ascii=False)
    return s


def _fmt_seg(x: str) -> str:
    return _esc(x)


def unify_facts(store):
    """Exact-identity triples from the delivered snapshot, SPLIT into
    structural facts and CVT attribute facts (user ruling 2026-10-06:
    the CVT neighborhood renders as its OWN block — attributes act as
    relations, values are answerable entities — the triple section keeps
    only the CVT marker).

    Path hops + env triples contribute structure; CVT attrs contribute
    the neighborhood block. Identity = the string triple; dedup drops
    repeated TEXT, never ownership (ownership stays upstream in the
    ledger; this layer never re-derives it).
    """
    facts, cvt_facts = {}, {}
    for pat in store.get("patterns", ()):
        for _rel, heads in pat.get("hops", ()):
            for h, ts in heads.items():
                for t in ts:
                    facts[(str(h), str(_rel), str(t))] = True
        for _hi, (mid, key, val) in pat.get("attrs", ()):
            cvt_facts[(str(mid), str(key), str(val))] = True
    for e in store.get("env_triples", ()) or ():
        h, r, t = e
        facts[(str(h), str(r), str(t))] = True
    # cross-side dedup: a fact already shown as a structural hop wins —
    # the neighborhood block only carries the CVT's OWN expansion.
    for k in list(cvt_facts):
        if k in facts:
            del cvt_facts[k]
    return list(facts), list(cvt_facts)


def order_facts(facts, centers):
    """Deterministic adjacency ordering: BFS from the centers, CVT attrs
    emitted immediately after their record's first appearance; remaining
    facts (disconnected from centers) appended in identity order."""
    adj = defaultdict(set)
    for h, r, t in facts:
        adj[h].add(t)
        adj[t].add(h)
    seen_nodes, ordered = set(), []
    rest = sorted(facts)

    def node_key(n):
        return (0 if n in set(centers) else 1, n)

    dq = deque(sorted(set(centers), key=node_key))
    emitted = set()
    while dq:
        n = dq.popleft()
        if n in seen_nodes:
            continue
        seen_nodes.add(n)
        # facts touching n, attrs of n first (record-local info adjacency)
        touching = sorted((f for f in rest if f[0] == n or f[2] == n),
                          key=lambda f: (f[1], f[0], f[2]))
        for f in touching:
            if f not in emitted:
                emitted.add(f)
                ordered.append(f)
        for nb in sorted(adj[n]):
            if nb not in seen_nodes:
                dq.append(nb)
    for f in rest:
        if f not in emitted:
            emitted.add(f)
            ordered.append(f)
    return ordered


def order_facts_witness(store, facts):
    """PATTERN-MAJOR ordering (the actual execution witness): one pattern's
    facts render contiguously — entry hops first, each record's attrs right
    after the hop that introduced it (attrs carry their hop index). First
    occurrence wins across patterns; leftover facts follow in identity
    order. This is the evidence-concentration ordering the design calls for;
    BFS remains the no-witness fallback."""
    emitted, ordered = set(), []
    for pi, pat in enumerate(store.get("patterns", ())):
        # attrs no longer ride the structural stream — they render in the
        # dedicated CVT neighborhood block (user ruling 2026-10-06)
        for hi, (rel, heads) in enumerate(pat.get("hops", ())):
            for f in sorted(((str(h), str(rel), str(t))
                             for h, ts in heads.items() for t in ts)):
                if f not in emitted:
                    emitted.add(f)
                    ordered.append(f)
    for f in sorted(facts):
        if f not in emitted:
            emitted.add(f)
            ordered.append(f)
    return ordered


def compress_adjacent_one_side(ordered):
    """Rows of (heads, rel, tails, fact_ids); exactly one side may hold >1.

    Greedy adjacent scan (design §5.3): consecutive facts sharing (h, r)
    tail-merge; consecutive sharing (r, t) head-merge; otherwise single.
    A merge run stops at the first key change — no global regrouping.
    """
    rows, i, n = [], 0, len(ordered)
    while i < n:
        h, r, t = ordered[i]
        # tail-merge run: same (h, r)
        j = i + 1
        tails = [t]
        while j < n and ordered[j][0] == h and ordered[j][1] == r:
            tails.append(ordered[j][2])
            j += 1
        if len(tails) > 1:
            rows.append(([h], r, tails, [ordered[k] for k in range(i, j)]))
            i = j
            continue
        # head-merge run: same (r, t)
        j = i + 1
        heads = [h]
        while j < n and ordered[j][1] == r and ordered[j][2] == t:
            heads.append(ordered[j][0])
            j += 1
        rows.append((heads, r, [t], [ordered[k] for k in range(i, j)]))
        i = j
    return rows


def _split_enum(items, budget):
    """Split an enumeration into line chunks; every line keeps >= min items
    unless a single item alone exceeds the budget (it rides its own line)."""
    if not items:
        return [[]]
    segs, cur, cur_w = [], [], 0
    for it in items:
        w = len(it) + 3
        if cur and cur_w + w > budget and len(cur) >= _ENUM_PER_LINE_MIN:
            segs.append(cur)
            cur, cur_w = [], 0
        cur.append(it)
        cur_w += w
    if cur:
        segs.append(cur)
    return segs


def format_rows(rows):
    """Full output. The single budget point (user ruling 2026-10-05): a
    merged side beyond _COMPRESS_AT members shows the first batch with an
    explicit ` …(+K compressed)` marker — never silent; the omitted members
    ride the compressed-group registry for answer-face auto-completion.

    HEAD-WRAP (user ruling 2026-10-06, aligns the code with the face
    note's teaching): a wrapped head list places the arrow+tail on its
    LAST line only — earlier lines are pure head lists ending with " …".
    One logical group stays identifiable as one group at any width."""
    lines, displayed = [], []
    compressed = []      # (displayed_members, omitted_members) raw names
    for heads, rel, tails, fids in rows:
        if len(heads) == 1:
            prefix = f"{_fmt_seg(heads[0])} --{rel}--> "
            items = [_fmt_seg(t) for t in tails]
            if len(items) > _COMPRESS_AT:
                shown, omitted = items[:_COMPRESS_AT], items[_COMPRESS_AT:]
                compressed.append((tails[:_COMPRESS_AT], tails[_COMPRESS_AT:]))
                displayed.extend(fids[:_COMPRESS_AT])
                for chunk in _split_enum(shown, _WIDTH - len(prefix)):
                    lines.append(prefix + " | ".join(chunk))
                lines[-1] += f" …(+{len(omitted)} compressed)"
            else:
                displayed.extend(fids)
                for chunk in _split_enum(items, _WIDTH - len(prefix)):
                    lines.append(prefix + " | ".join(chunk))
        else:
            # head-merge wrap: arrow+tail on the LAST line; continuation
            # lines are pure head lists ending with " …".
            suffix = f" --{rel}--> {_fmt_seg(tails[0])}"
            items = [_fmt_seg(h) for h in heads]
            if len(items) > _COMPRESS_AT:
                shown, omitted = items[:_COMPRESS_AT], items[_COMPRESS_AT:]
                compressed.append((heads[:_COMPRESS_AT], heads[_COMPRESS_AT:]))
                displayed.extend(fids[:_COMPRESS_AT])
                chunks = _split_enum(shown, _WIDTH - len(suffix))
            else:
                displayed.extend(fids)
                chunks = _split_enum(items, _WIDTH - len(suffix))
            for k, chunk in enumerate(chunks):
                last = (k == len(chunks) - 1)
                if last:
                    lines.append(" | ".join(chunk) + suffix)
                else:
                    lines.append(" | ".join(chunk) + " …")
    return "\n".join(lines), displayed, compressed


_NUM_RE = None  # compiled lazily; module keeps zero side effects at import

_PROX_GATE = 0.5  # relative-distance gate: matched values must sit within
                  # the question number's order of magnitude ("near" a year
                  # means years apart, not six orders of magnitude apart)


def _nums(s):
    """Numeric tokens of a display string (commas folded, dates yield
    year/month/day tokens). Deterministic — this is the value-proximity
    signal, deliberately NOT GTE: embedding distance does not encode
    that 2006 and 2008 are two years apart."""
    global _NUM_RE
    if _NUM_RE is None:
        import re
        _NUM_RE = re.compile(r"\d+(?:\.\d+)?")
    return [float(x) for x in _NUM_RE.findall(str(s).replace(",", ""))]


def _proximity(attrs, qnums):
    """Best (rel_distance, key, value) over question numbers × this CVT's
    attr values; None when neither side holds a commensurable number.
    RELATIVE distance with the _PROX_GATE magnitude gate: a value counts
    as 'near' only within the question number's order of magnitude
    ("Season 5" is not near netflix_id 60031289 — spec 537 lesson)."""
    best = None
    for key, val in attrs:
        for v in _nums(val):
            for q in qnums:
                rel = abs(q - v) / max(1.0, abs(q), abs(v))
                if rel <= _PROX_GATE and (best is None or rel < best[0]):
                    best = (rel, key, val)
    return best


def format_cvt_block(cvt_facts, question=""):
    """The CVT NEIGHBORHOOD block (user ruling 2026-10-06): each CVT node
    gets its own sub-block — the marker line, then one indented
    `attribute: value` line per neighbor fact. Attributes act as
    relations; values are answerable entities.

    VALUE-PROXIMITY RANKING (user ruling 2026-10-08): many-CVT cases are
    one schema with different VALUES — an ordering problem, not a
    retrieval problem. When the question carries numbers, CVT nodes whose
    values sit within the question number's magnitude gate rank by exact
    numeric distance (nearest first); the matched attribute leads its
    sub-block. Nodes without a gated match keep their id order AFTER all
    scored ones. Questions without numbers — or blocks without
    commensurable values — leave the block byte-identical to the
    unranked form (single-variable attribution). Overflow beyond
    _COMPRESS_AT no longer vanishes behind a count: it renders as packed
    one-line summaries `mid (key≈value)` continuing the same rank order,
    so every id stays visible and selectable (L0-5)."""
    by_mid = defaultdict(list)
    for mid, key, val in cvt_facts:
        by_mid[mid].append((str(key).rsplit(".", 1)[-1], str(val)))
    lines, displayed = [], []
    scored = {}
    qnums = _nums(question) if question else []
    if qnums:
        for mid, kvs in by_mid.items():
            best = _proximity(sorted(set(kvs)), qnums)
            if best is not None:
                scored[mid] = best
    if scored:
        # ties in relative distance break by the matched value's string
        # (ascending) — same-format numbers and ISO dates then order
        # numerically/chronologically, not by opaque mid (241 lesson:
        # Germany 10.1284 vs Belgium 10.125 equidistant from the
        # question literal)
        mids = sorted(by_mid,
                      key=lambda m: ((0, scored[m][0], scored[m][2], m)
                                     if m in scored else (1, 0.0, "", m)))
    else:
        mids = sorted(by_mid)
    omitted = 0
    summarize = bool(scored)  # summaries exist only when ranking is active
    if len(mids) > _COMPRESS_AT:
        omitted = len(mids) - _COMPRESS_AT
        shown, rest = mids[:_COMPRESS_AT], mids[_COMPRESS_AT:]
    else:
        shown, rest = mids, []
    for mid in shown:
        lines.append(_fmt_seg(mid))
        kvs = sorted(set(by_mid[mid]))
        if mid in scored:
            # the proximity-matched attribute leads its sub-block
            mk = scored[mid][1]
            kvs = ([kv for kv in kvs if kv[0] == mk]
                   + [kv for kv in kvs if kv[0] != mk])
        for key, val in kvs:
            lines.append(f"  {key}: {_fmt_seg(val)}")
            displayed.append((mid, key, val))
    if rest and summarize:
        cap = int(os.environ.get("SEQ_CVT_SUMMARY_CAP", "120") or 0)
        if cap > 0:
            listed, further = rest[:cap], max(0, len(rest) - cap)
        else:
            listed, further = rest, 0
        items = []
        for m in listed:
            if m in scored:
                _d, _k, _v = scored[m]
                items.append(f"{_fmt_seg(m)} ({_k}≈{_fmt_seg(_v)})")
            else:
                first = sorted(set(by_mid[m]))[:1]
                if first:
                    items.append(f"{_fmt_seg(m)} ({first[0][0]}: "
                                 f"{_fmt_seg(first[0][1])})")
                else:
                    items.append(_fmt_seg(m))
        lines.append("more cvt nodes (nearest values first):")
        for chunk in _split_enum(items, _WIDTH):
            lines.append(" | ".join(chunk))
        if further > 0:
            lines.append(f"…(+{further} further, farthest from question "
                         f"values)")
    elif omitted:
        lines.append(f"…(+{omitted} cvt nodes)")
    text = "\n".join(lines)
    header = ""
    if lines:
        header = ("cvt nodes (attribute = relation, value = answerable "
                  "entity):\n")
    return header + text, displayed, omitted


def complete_compressed_groups(values, groups):
    """Answer-face auto-completion (user ruling, continuing the existing
    injection ruling): when the submitted values ARE the whole displayed
    batch of a compressed group, the omitted members join the submission."""
    if not groups:
        return values
    vset = {str(v).strip() for v in values}
    out = list(values)
    for displayed, omitted in groups:
        dset = {str(x).strip() for x in displayed}
        if vset and dset and vset == dset:
            seen = set(vset)
            for x in omitted:
                if str(x).strip() not in seen:
                    out.append(x)
                    seen.add(str(x).strip())
            break
    return out


def render_v4(store, question=""):
    facts, cvt_facts = unify_facts(store)
    # witness ordering when patterns exist (evidence concentration); BFS
    # adjacency otherwise (degenerate stores without pattern structure).
    if store.get("patterns"):
        ordered = order_facts_witness(store, facts)
    else:
        ordered = order_facts(facts, store.get("centers") or [])
    rows = compress_adjacent_one_side(ordered)
    text, displayed, compressed = format_rows(rows)
    cvt_text, cvt_disp, cvt_omitted = format_cvt_block(cvt_facts, question)
    store["_display_stats"] = {
        "group_truncations": len(compressed), "truncated_members": 0,
        "also_in_evidence": 0,
        "delivered_facts": len(facts) + len(cvt_facts),
        "displayed_facts": len(displayed) + len(cvt_disp),
        "omitted_facts": (len(facts) + len(cvt_facts)
                          - len(displayed) - len(cvt_disp)),
    }
    store["_compressed_groups"] = compressed
    # pattern index (structure only — design §2.2 keeps the short index)
    labels = [pat.get("label") for pat in store.get("patterns", ()) if pat.get("label")]
    head = ("▸ patterns: " + " | ".join(labels) + "\n") if labels else ""
    # BRIDGE-CONTEXT tags (user ruling: default ON; deep-audit fix — the
    # store's bridge names now reach this arm): tag rows whose relation
    # reached the center only as an expansion bridge.
    _br = store.get("_bridge_shorts") or ()
    if _br:
        lines = text.splitlines()
        for _bn in _br:
            for _i, _ln in enumerate(lines):
                if f"--{_bn}-->" in _ln:
                    lines[_i] = _ln.replace(
                        f"--{_bn}-->", f"--{_bn} (bridge ctx)-->")
        text = "\n".join(lines)
    legend = ""
    if compressed:
        legend = ("\nlegend: a `…(+K compressed)` enumeration shows its first "
                  f"{_COMPRESS_AT} members; submitting that whole displayed "
                  "batch as the answer auto-completes the compressed members.")
    if cvt_text:
        text = text + "\n" + cvt_text
    return head + text + legend


# ── acceptance helper (tests): recover the exact fact set from the text ──
import re as _re


def _top_level_positions(s, needle):
    """Positions of needle occurrences OUTSIDE JSON-quoted spans."""
    pos, i, in_q = [], 0, False
    while i < len(s):
        c = s[i]
        if in_q:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_q = False
        elif c == '"':
            in_q = True
        elif s.startswith(needle, i):
            pos.append(i)
            i += len(needle)
            continue
        i += 1
    return pos


def _split_top(s, sep=" | "):
    """Split on top-level separators only (quote-aware)."""
    cuts = [0] + [p + len(sep) for p in _top_level_positions(s, sep)]
    ends = [p for p in _top_level_positions(s, sep)] + [len(s)]
    return [s[a:b] for a, b in zip(cuts, [e for e in ends])]


def _unescape(x):
    s = x.strip()
    if s.startswith('"') and s.endswith('"'):
        import json
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def expand_text(text):
    """Inverse of format_rows/format_cvt_block at the FACT level: every
    line expands to its full triple set. Handles the head-wrap
    continuation format (pure head lines ending " …" prepend to the next
    arrow line's head list) and the CVT neighborhood block (marker line +
    indented `attr: value` lines → (mid, attr, value))."""
    facts = set()
    pending_heads = []
    cur_mid = None
    for ln in text.splitlines():
        raw = ln.rstrip()
        if not raw or raw.startswith("▸") or raw.startswith("legend:"):
            continue
        if raw.startswith("cvt nodes ("):
            cur_mid = None
            continue
        if raw.startswith("more cvt nodes"):
            # packed overflow summaries (ids stay visible; their facts
            # were budgeted out — expand recovers exactly the displayed)
            cur_mid = None
            continue
        if raw.startswith("  ") and ": " in raw and cur_mid is not None:
            key, _, val = raw.strip().partition(": ")
            facts.add((_unescape(cur_mid), _unescape(key), _unescape(val)))
            continue
        arrows = _top_level_positions(raw, "--> ")
        if not arrows:
            # head-wrap continuation: pure head list ending with " …"
            if raw.endswith("…") and pending_heads is not None:
                pending_heads.extend(_split_top(raw[:-1].rstrip()))
                cur_mid = None
                continue
            cur_mid = _unescape(raw) if raw and not raw.startswith("…") else None
            continue
        a = arrows[-1]
        right_raw = raw[a + 4:]
        left_raw = raw[:a]
        # rel sits between the nearest preceding top-level " --" and the arrow
        dashes = _top_level_positions(left_raw, " --")
        if dashes:
            d = dashes[-1]
            rel, left_raw = left_raw[d + 3:], left_raw[:d]
        else:
            rel = "?"
        if _top_level_positions(left_raw, " | ") or pending_heads:
            # head-merged line (possibly preceded by continuation lines):
            # left enumerates heads, right is the single tail
            heads = list(pending_heads) + _split_top(left_raw)
            pending_heads = []
            for h in heads:
                facts.add((_unescape(h), rel, _unescape(right_raw)))
            cur_mid = None
        else:
            for t in _split_top(right_raw):
                facts.add((_unescape(left_raw), rel, _unescape(t)))
            cur_mid = None
    return facts


def enabled():
    return os.environ.get("SEQ_RENDER_V4", "0") == "1"
