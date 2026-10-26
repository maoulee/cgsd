"""V3.5 evidence renderer — SAPS tree format (user ruling 2026-08-31).

SAPS-style tree connectors (├── └── │) with center as explicit root.
Relation names on each connector. R3 sibling merge when subtrees identical.
L1 uniform-compression for CVT attrs. Zero-loss: all entity names visible.

门控: SEQ_RENDER_V35=1"""
import itertools
from collections import defaultdict


def _cvt_like(n):
    s = str(n)
    return s[:2] in ("m.", "g.") and len(s) > 4


def render_v35_ack(treq, bres, ctx):
    ents, rels = ctx.ents, ctx.rels
    rel_adj = defaultdict(list)
    for h, rl, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        rel_adj[rl].append((h, t))
    cvt_attrs = defaultdict(dict)
    for h, rl, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        hn = str(ents[h]) if 0 <= h < len(ents) else ""
        if _cvt_like(hn):
            key = rels[rl].rsplit(".", 1)[-1] if 0 <= rl < len(rels) else str(rl)
            tv = str(ents[t]) if 0 <= t < len(ents) else str(t)
            if not _cvt_like(tv) and len(tv) < 40:
                cvt_attrs[hn].setdefault(key, tv)

    def enum(center, hops, cap=400):
        paths = [[center]]
        for rid, rev in hops:
            es = rel_adj.get(rid) or []
            nxt = []
            for p in paths:
                tail = p[-1]
                for h, t in es:
                    if (t == tail) if rev else (h == tail):
                        nxt.append(p + [h if rev else t])
                        if len(nxt) > cap:
                            break
                if len(nxt) > cap:
                    break
            paths = nxt
            if not paths:
                break
        return paths

    def nm(i):
        return str(ents[i]) if 0 <= i < len(ents) else str(i)

    def _short(rid):
        return rels[rid].rsplit(".", 1)[-1] if 0 <= rid < len(rels) else "?"

    center_names = [e for e, _i in (treq.get("centers") or [])]
    L = [f"entities: {' | '.join(center_names)}", "triples:"]
    seen_sig = set()

    for (cname, cidx), pe in zip(treq["centers"], bres["pe_list"]):
        for pid, p in (pe.items() if isinstance(pe, dict) else []):
            td = (getattr(p, "tree_data", None) or {})
            tp = (td.get("paths") or [None])[0]
            if not tp:
                continue
            rseq = tp.get("relations") or []
            rids = []
            ok = True
            for x in rseq:
                if isinstance(x, int) and 0 <= x < len(rels):
                    rids.append(x)
                elif x in rels:
                    rids.append(rels.index(x))
                else:
                    ok = False
                    break
            if not ok:
                continue
            best, bn, bc = None, -1, None
            for combo in itertools.product([False, True], repeat=len(rids)):
                inst = enum(cidx, list(zip(rids, combo)))
                if len(inst) > bn:
                    best, bn, bc = inst, len(inst), combo
            if not best or bn <= 0:
                continue
            key = (tuple(rids), bc)
            if key in seen_sig:
                continue
            seen_sig.add(key)
            sig = list(zip(rids, bc))

            # group header: relation chain declared once
            chain = " → ".join(
                f"←{_short(r)}" if rv else f"{_short(r)}→" for r, rv in sig)
            L.append(f"  ▸ [{chain}]  ({bn} instances)")

            # name-trie
            name_paths = sorted({tuple(nm(x) for x in p) for p in best[:300]})
            tree = {}
            for p in name_paths:
                for d in range(1, len(p)):
                    tree.setdefault(p[d - 1], set()).add(p[d])
            root = name_paths[0][0]

            # L1 uniform-compression for CVT attrs
            cvt_here = sorted({n for p in name_paths for n in p if _cvt_like(n)})
            attr_vals = defaultdict(list)
            for cn in cvt_here:
                for k, v in cvt_attrs.get(cn, {}).items():
                    attr_vals[k].append(v)
            uniform_pairs = [(k, vs[0]) for k, vs in attr_vals.items()
                             if len(set(vs)) == 1 and len(vs) == len(cvt_here)]
            disc_attrs = sorted(
                [k for k, vs in attr_vals.items()
                 if not (len(set(vs)) == 1 and len(vs) == len(cvt_here))],
                key=lambda k: -len(set(attr_vals[k])))
            used_cvt = set()

            def cvt_str(n):
                if _cvt_like(n) and n in cvt_attrs and n not in used_cvt:
                    used_cvt.add(n)
                    pairs = [(k, cvt_attrs[n][k]) for k in disc_attrs
                             if k in cvt_attrs[n]]
                    at = "; ".join(f"{k}={v}" for k, v in pairs[:6])
                    return f"{n} [{at}]" if at else n
                return n

            def sub_sig(n, d):
                if d >= len(sig):
                    return ()
                kids = sorted(tree.get(n, ()))
                return (_short(sig[d][0]), sig[d][1],
                        tuple((c,) + sub_sig(c, d + 1) for c in kids))

            def render_level(siblings, depth, prefix, is_last):
                """SAPS tree: ├── / └── / │ connectors with relation names."""
                if depth >= len(sig):
                    return
                rn = _short(sig[depth][0])
                arrow = "←" if sig[depth][1] else "→"
                conn = f"{rn} {arrow}"
                # R3: merge siblings with identical subtrees
                groups = defaultdict(list)
                for k in siblings:
                    groups[sub_sig(k, depth + 1)].append(k)
                items = list(groups.items())
                for gi, (_gs, members) in enumerate(items):
                    last_g = (gi == len(items) - 1)
                    branch = "└──" if last_g else "├──"
                    names = " | ".join(cvt_str(m) for m in sorted(members))
                    L.append(f"{prefix}{branch} {conn} {names}")
                    # recurse into the shared subtree
                    sub = sorted(tree.get(members[0], ()))
                    if depth + 1 < len(sig) and sub:
                        new_p = prefix + ("    " if last_g else "│   ")
                        render_level(sub, depth + 1, new_p, last_g)

            L.append(f"  {root}")
            kids0 = sorted(tree.get(root, ()))
            if kids0:
                render_level(kids0, 0, "  ", True)
            if uniform_pairs:
                uv = "; ".join(f"{k}={v}" for k, v in uniform_pairs[:4])
                if uv:
                    L.append(f"    (all: {uv})")
            ends = {p[-1] for p in name_paths}
            L.append(f"  candidates: {len(ends)} endpoints")
        break
    return "\n".join(L) if len(L) > 2 else "(empty)"
