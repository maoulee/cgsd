"""V3.7 candidate-centric renderer (user design 2026-08-31: 候选为主键).

WALK-FAITHFUL: v3.6 discarded the walked node sequences (kept only each
pattern's first path's RELATIONS, then re-enumerated directions by max
instance count). That reconstruction inverted CVT-mediated patterns — the
fan-out direction won ([center→state→cvt ×88], candidates buried mid-tree,
CVT records carrying ◂) while the walk itself had the semantic witnesses
([center ←cvt─ state ×7]). V3.7 organizes the walked paths directly:

One line per candidate, self-contained — the candidate(s) lead the line,
their shared instantiated path follows, the final arrow points back at them:

    Iowa ◂  Missouri River ←partially_contains─ m.0wg906w ─partially_contained_by→
    Iowa ◂  Missouri River ─partially_containedby→ ─adjoin_s→ m.02tbl_w [adjoins=North Dakota; …]

Mechanics (folding is structure-only; entity names are never folded):
- every walked tree path renders as one candidate line (all patterns, ALL
  centers — v3.6 broke after the first center);
- a path that is a node-suffix of a longer walked path drops (the longer
  form carries it) — kills the walk's mid-rooted continuation duplicates;
- trailing CVT/value hops strip into a records suffix on the candidate's
  line; len-2 paths keep the CVT as the candidate with attrs inline;
- candidates whose chains differ only by the final node fold onto one line
  (fan-out compression: `A | B | C ◂ <shared chain>`);
- attr pairs present on EVERY record of a group fold to one header line
  (L1 uniform compression); attrs restating an instantiated path node drop;
  per-record attrs cap at 6;
- group order = (hops asc, entity-last-hop first, instances desc) — the
  last hop defines the candidate (user ruling), so patterns whose last hop
  LANDS on a named entity outrank record-fanout patterns of equal length.

Gate: SEQ_RENDER_V37=1"""
import re
from kgqa.agent.entity_kinds import is_cvt as _cvt
from collections import defaultdict

_VAL_RE = re.compile(
    r"^\d{4}-\d{2}|^\d{2}:\d{2}|^[+-]?[\d.,]+$|^\d{4}s?$")



def _val(n):
    return bool(_VAL_RE.match(str(n).strip()))


def _parse_node(s):
    """'m.xxx: [k=v, k2=v2]' → (name, [k=v, ...]); bare name → (name, [])."""
    s = str(s)
    if s.endswith("]") and ": [" in s:
        base, _, attr_txt = s.partition(": [")
        base = base.strip()
        if _cvt(base):
            attrs = [a.strip() for a in attr_txt[:-1].split(",") if a.strip()]
            return base, attrs
    return s, []


_NOISY_ATTR = {"type", "types", "instance", "instances", "notable_types",
               "topic_equivalent_webpage", "webpage", "mid", "guid",
               "key", "keys", "permission", "is_reviewed", "article",
               "description", "alias", "name"}


def render_v37_ack(treq, bres, ctx):
    ents, rels = ctx.ents, ctx.rels
    rel_adj = defaultdict(set)
    for h, rl, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        rel_adj[rl].add((h, t))
    ent_idx = {}
    for j, e in enumerate(ents):
        ent_idx.setdefault(str(e), j)

    # graph-reconstructed CVT attrs (decoration-independent — the walk's
    # decoration is sometimes empty for CVTs whose content the graph HAS:
    # 31 bare mids with film= edges, 2784 specimen)
    cvt_graph = defaultdict(list)      # mid -> [k=v, ...]
    for h, rl, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        hn = str(ents[h]) if 0 <= h < len(ents) else ""
        if not _cvt(hn):
            continue
        k = (str(rels[rl]).rsplit(".", 1)[-1]
             if 0 <= rl < len(rels) else "?")
        tv = str(ents[t]) if 0 <= t < len(ents) else str(t)
        if not _cvt(tv) and len(tv) < 60 and k not in _NOISY_ATTR:
            cvt_graph[hn].append(f"{k}={tv}")

    def merged_pairs(mid, deco_pairs):
        seen = set(deco_pairs)
        return deco_pairs + [a for a in cvt_graph.get(mid, ())
                             if a not in seen]

    center_names = [e for e, _i in (treq.get("centers") or [])]
    # BRIDGE-HOP ANNOTATION (user flag 2026-09-01, Liszt specimen): the walk
    # deliberately enumerates 2-hop completions whose LAST hop rides a
    # selected relation from ANY first hop (Bernie Brewer semantics) — hops
    # the model did NOT select render parenthesized so "gave relations but
    # walked something else" is visible instead of silent. Exact full-name
    # match: parallel encodings of a selected relation also read as bridges.
    sel_rels = {str(r) for r in (treq.get("rel_names") or [])}

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

    # ── collect walked paths (all patterns, all centers), dedup ──
    # DEDUP RULE (user ruling 2026-09-01, corrected): each relation keeps its
    # OWN path and its OWN candidate — member and member_of designate
    # different candidates (their edge DIRECTIONS differ), so member's path
    # must NEVER fold into member_of's display. Lines merge only when node
    # sequence AND per-hop direction both coincide (parallel same-direction
    # spellings union their labels; opposite-direction relations split into
    # separate lines, each with its own arrows).
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
                # SAME-RELATION LOOP RULE (user ruling 2026-09-02, Freemasonry
                # specimen): cycles through DIFFERENT relations are legitimate
                # evidence, but a path that goes out and comes back on the
                # SAME relation (`A ─members→ m.x ←members─ …`) is a
                # degenerate out-and-back — drop it. Same-direction repeats
                # (nested contains→ contains→) stay.
                if len(relsn) >= 2:
                    raw_d = {i: hop_dir(relsn[i], key[i], key[i + 1])
                             for i in range(len(relsn))}
                    _degen = False
                    for a in range(len(relsn)):
                        for b in range(a + 1, len(relsn)):
                            if (relsn[a] == relsn[b]
                                    and raw_d[a] != raw_d[b]):
                                _degen = True
                                break
                        if _degen:
                            break
                    if _degen:
                        continue
                # direction is CANDIDATE-DEFINING only on selected hops —
                # bridge hops (unselected parallel edges) fold their two
                # storage directions into one neutral line; selected
                # relations keep per-direction lines (member vs member_of)
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

    # ── drop continuations: any proper node-suffix of a longer walked path ──
    node_seqs = {k[0] for k in raw}
    items = [(k[0], raw[k]["rels"], raw[k]["attrs"], k[1]) for k in raw
             if not any(k[0][i:] in node_seqs for i in range(1, len(k[0])))]

    def seg_for(relsns, dirs, names, ent_idx, hi, a_i, b_i):
        # per-hop relation label: union of the parallel SAME-DIRECTION
        # spellings that walked this node pair (the raw key guarantees every
        # variant shares this hop's direction)
        shorts = []
        for rv in relsns:
            rn = rv[hi] if hi < len(rv) else None
            if rn is None:
                continue
            sh = rn.rsplit(".", 1)[-1]
            if sh not in shorts:
                shorts.append(sh)
        if not shorts:
            sh = "?"
        elif len(shorts) == 1:
            sh = shorts[0]
        else:
            sh = "/".join(shorts[:2]) + ("+" if len(shorts) > 2 else "")
        if not any((rv[hi] if hi < len(rv) else None) in sel_rels
                   for rv in relsns):
            sh = f"({sh})"          # bridge hop — no variant was selected
        d = dirs[hi] if hi < len(dirs) else "?"
        if d == "f":
            return f"─{sh}→"
        if d == "r":
            return f"←{sh}─"
        return f"─{sh}─"

    # ── pass 1: decompose paths; collect attr stats for uniform folding ──
    # TERMINAL-TYPE LINE FORM (user ruling 2026-09-01, Mandela specimen): the
    # walked terminal IS the line's terminal — a named entity leads the line
    # as candidate (◂); a CVT/value terminal is a RECORD the chain completes
    # INTO, attrs inline (the model picks among its attrs at answer time).
    # No trailing-CVT stripping: it pulled mid-chain entities to the head and
    # left adjacent arrows where the entity used to sit.
    groups = {}          # gsig → {"n", "ent_last", "paths": [entry...]}
    for key, relsns, attrs, dirs in items:
        names = list(key)
        q = list(range(len(names)))
        ci = q[-1]
        cand = names[ci]
        red = {names[j].lower() for j in q}

        def clean_pairs(attr_list):
            out = []
            for a in attr_list:
                if "=" in a:
                    _k, v = a.split("=", 1)
                    if v.strip().lower() in red:
                        continue        # restates an instantiated path node
                out.append(a)
            return out

        chain = [names[q[0]]]
        sig = []
        for hi in range(len(q) - 1):
            ar = seg_for(relsns, dirs, names, ent_idx, hi, q[hi], q[hi + 1])
            sig.append(ar)
            chain.append(ar)
            if hi + 1 < len(q) - 1:
                nd = names[q[hi + 1]]
                if _cvt(nd):
                    # body CVT: inline its graph attrs (decoration-independent)
                    at = "; ".join(clean_pairs(merged_pairs(
                        nd, attrs[q[hi + 1]]))[:3])
                    chain.append(f"{nd} [{at}]" if at else nd)
                else:
                    chain.append(nd)
        cand_pairs = (clean_pairs(merged_pairs(cand, attrs[ci]))
                      if _cvt(cand) else [])
        # record paths (CVT/value terminal) hang under their OWNER — the
        # last named entity on the chain before the terminal
        owner = cand
        if _cvt(cand) or _val(cand):
            for j in range(len(q) - 2, -1, -1):
                nm_ = names[q[j]]
                if not _cvt(nm_) and not _val(nm_):
                    owner = nm_
                    break
        entry = {
            "sig": tuple(sig),
            "chain": " ".join(chain),          # ends with arrow INTO cand
            "cand": cand,
            "owner": owner,
            "cand_pairs": cand_pairs,
            "ent_last": not (_cvt(cand) or _val(cand)),
            "mark": "" if (_cvt(cand) or _val(cand)) else " ◂",
        }
        g = groups.setdefault(entry["sig"],
                              {"n": 0, "ent_last": 0, "paths": []})
        g["n"] += 1
        g["ent_last"] += 1 if entry["ent_last"] else 0
        g["paths"].append(entry)

    if not groups:
        return "(empty)"

    # ── pass 2: uniform attr folding, then ENTITY-FIRST blocks ──
    # (user ruling 2026-09-01: group by candidate FIRST — each entity's
    # pattern paths and relations list UNDER it, `Reiner Schöne：路径…` —
    # not entities re-listed per pattern. Record paths hang under their
    # owner entity; candidates with IDENTICAL path sets fold onto one
    # block head.)
    L = [f"entities: {' | '.join(center_names)}"]
    all_cands = set()
    stats = {}
    for gsig, g in groups.items():
        pair_cnt, hosts = defaultdict(int), 0
        for e in g["paths"]:
            if e["cand_pairs"]:
                hosts += 1
            for a in set(e["cand_pairs"]):
                pair_cnt[a] += 1
        stats[gsig] = {a for a, c in pair_cnt.items()
                       if hosts > 2 and c == hosts}

    blocks = {}
    for gsig, g in groups.items():
        uniform = stats[gsig]
        for e in g["paths"]:
            lead = str(e["cand"])
            if e["cand_pairs"]:
                at = "; ".join([a for a in e["cand_pairs"]
                                if a not in uniform][:6])
                if at:
                    lead = f"{e['cand']} [{at}]"
            if e["mark"]:
                all_cands.add(e["cand"])
                bkey = e["cand"]
            else:
                bkey = e["owner"]
            b = blocks.setdefault(bkey, {"entity": False, "fold": {}, "n": 0})
            if e["mark"]:
                b["entity"] = True
            b["fold"].setdefault(e["chain"], []).append(
                (e["cand"], lead, bool(e["mark"])))
            b["n"] += 1

    rendered = {}                      # line-tuple -> block info (merge twins)
    for bkey, b in blocks.items():
        lines = []
        for chain, items in sorted(b["fold"].items()):
            recs = [lead for _c, lead, mk in sorted(items) if not mk]
            ents = [c for c, _l, mk in sorted(items) if mk]
            if ents:
                # entity-candidate path: the chain's final arrow points at
                # the block head itself
                lines.append(chain)
            if recs:
                shown, used = [], 0
                for lead in recs:
                    if used + len(lead) > 480 and shown:
                        break
                    shown.append(lead)
                    used += len(lead) + 3
                line = f"{chain} {' | '.join(shown)}"
                more = recs[len(shown):]
                if more:
                    nm, used_n = [], 0
                    for x in more:
                        if used_n + len(x) > 360:
                            break
                        nm.append(x)
                        used_n += len(x) + 3
                    dot = " …" if len(nm) < len(more) else ""
                    line += f" (+{len(more)}: {' | '.join(nm)}{dot})"
                lines.append(line)
        lines = sorted(set(lines))
        key = tuple(lines)
        if key in rendered:
            rendered[key]["heads"].append(bkey)
            rendered[key]["entity"] = rendered[key]["entity"] or b["entity"]
            rendered[key]["n"] += b["n"]
        else:
            rendered[key] = {"heads": [bkey], "entity": b["entity"],
                             "n": b["n"], "lines": lines}

    for blk in sorted(rendered.values(),
                      key=lambda x: (not x["entity"], -x["n"],
                                     sorted(x["heads"])[0])):
        heads = " | ".join(sorted(set(blk["heads"])))
        mark = " ◂" if blk["entity"] else ""
        L.append(f"{heads}{mark}  ({blk['n']} paths)")
        for ln in blk["lines"][:88]:
            L.append(f"    {ln}")
        if len(blk["lines"]) > 88:
            L.append(f"    ... +{len(blk['lines']) - 88} paths "
                     f"(see candidates)")

    named = sorted(all_cands)
    L.append(f"candidates ({len(named)}): {' | '.join(named)}")
    return "\n".join(L)
