"""V3.6 candidate-centric renderer v2 (user design).
每个候选标注从起点到它的完整模式路径。按模式分组渲染,叶=候选(◂),
CVT 属性内联,关系名在每个连接符上。门控: SEQ_RENDER_V36=1"""
import itertools
from kgqa.agent.entity_kinds import is_cvt as _cvt
from collections import defaultdict

_NOISY_ATTR = {"type", "types", "instance", "instances", "notable_types",
               "topic_equivalent_webpage", "webpage", "mid", "guid",
               "key", "keys", "permission", "is_reviewed", "article",
               "description", "alias", "name"}

def _short(rels, rid):
    return rels[rid].rsplit(".", 1)[-1] if 0 <= rid < len(rels) else "?"

def render_v36_ack(treq, bres, ctx):
    ents, rels = ctx.ents, ctx.rels
    rel_adj = defaultdict(list)
    for h, rl, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        rel_adj[rl].append((h, t))
    cvt_attrs = defaultdict(dict)
    for h, rl, t in zip(ctx.h_ids, ctx.r_ids, ctx.t_ids):
        hn = str(ents[h]) if 0 <= h < len(ents) else ""
        if _cvt(hn):
            k = _short(rels, rl)
            tv = str(ents[t]) if 0 <= t < len(ents) else str(t)
            if not _cvt(tv) and len(tv) < 50 and k not in _NOISY_ATTR:
                cvt_attrs[hn].setdefault(k, tv)

    def nm(i):
        return str(ents[i]) if 0 <= i < len(ents) else str(i)

    def enum(center, hops, cap=300):
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

    center_names = [e for e, _i in (treq.get("centers") or [])]
    all_terminals = set()
    groups = []

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
            sig = list(zip(rids, bc))
            chain = "→".join(
                (f"←{_short(rels, r)}" if rv else f"{_short(rels, r)}→")
                for r, rv in sig)
            name_paths = sorted({tuple(nm(x) for x in p) for p in best[:300]})
            for np in name_paths:
                all_terminals.add(np[-1])
            groups.append((sig, name_paths, chain))
        break

    if not groups:
        return "(empty)"

    groups.sort(key=lambda g: (len(g[0]), -len(g[1])))

    cvt_here = sorted({n for _s, ps, _c in groups for p in ps for n in p
                       if _cvt(n)})
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

    def cvt_line(n):
        if _cvt(n) and n in cvt_attrs and n not in used_cvt:
            used_cvt.add(n)
            pairs = [(k, cvt_attrs[n][k]) for k in disc_attrs
                     if k in cvt_attrs[n]]
            at = "; ".join(f"{k}={v}" for k, v in pairs[:6])
            return f"{n} [{at}]" if at else n
        return n

    L = [f"entities: {' | '.join(center_names)}"]
    if uniform_pairs:
        uv = "; ".join(f"{k}={v}" for k, v in uniform_pairs[:4])
        L.append(f"all records: ({uv})")

    for sig, name_paths, chain in groups:
        n_hops = len(sig)
        root = name_paths[0][0]
        L.append(f"  ▸ {chain}  ({len(name_paths)} instances)")
        tree = {}
        for p in name_paths:
            for d in range(1, len(p)):
                tree.setdefault(p[d-1], set()).add(p[d])

        def sub_sig(n, d):
            if d >= n_hops:
                return ()
            kids = sorted(tree.get(n, ()))
            return (_short(rels, sig[d][0]), sig[d][1],
                    tuple((c,) + sub_sig(c, d + 1) for c in kids))

        def render_level(siblings, depth, prefix):
            if depth >= n_hops:
                return
            rn = _short(rels, sig[depth][0])
            arrow = "←" if sig[depth][1] else "→"
            r3 = defaultdict(list)
            for k in siblings:
                r3[sub_sig(k, depth + 1)].append(k)
            items = list(r3.items())
            for gi, (_gs, members) in enumerate(items):
                last_g = gi == len(items) - 1
                branch = "└──" if last_g else "├──"
                names = " | ".join(cvt_line(m) for m in sorted(members))
                is_cand = all(m in all_terminals and not tree.get(m)
                              for m in members)
                mark = " ◂" if is_cand else ""
                L.append(f"{prefix}{branch} {rn}{arrow} {names}{mark}")
                sub = sorted(tree.get(members[0], ()))
                if depth + 1 < n_hops and sub:
                    new_p = prefix + ("    " if last_g else "│   ")
                    render_level(sub, depth + 1, new_p)

        L.append(f"  {root}")
        kids0 = sorted(tree.get(root, ()))
        if kids0:
            render_level(kids0, 0, "  ")

    named_cands = sorted(t for t in all_terminals if not _cvt(t))
    L.append(f"candidates ({len(named_cands)}): {' | '.join(named_cands)}")
    return "\n".join(L)
