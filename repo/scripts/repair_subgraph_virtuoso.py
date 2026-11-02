#!/usr/bin/env python3
"""
Repair Freebase subgraphs (RoG/SubgraphRAG pkl format) by injecting the GOLD
answer path, materialized AUTHORITATIVELY from the Freebase Virtuoso dump.

Problem
-------
The k-hop BFS subgraphs miss ~21-54% of gold edges (mostly CVT-mediated
relations: film.actor.film, education, religions, places_lived, containedby,
from/to/start_date ...). The relation NAME is often in relation_list but the
actual gold triple (gold_subject_mid -> gold_relation -> object) is absent, so
the answer is unreachable. The previous heuristic repair (that produced the
*_path_completed pkls) is incomplete; this re-materializes from the real
Freebase via Virtuoso.

Method (per case)
-----------------
1. Take the gold SPARQL (CWQ: cwq_sparql/test.json; WebQSP: WebQSP.test.json
   Questions[].Parses[0].Sparql).
2. Rewrite SELECT to `SELECT *`, keep WHERE (incl. FILTERs).
3. Execute against Virtuoso (localhost:8890/sparql) -> ALL variable bindings
   (answer var + intermediate CVT nodes + literal values).
4. Materialize each BGP triple pattern from every binding row ->
   concrete (subj_term, relation, obj_term).
5. Merge into the pkl record following the storage convention:
     - named entity -> text_entity_list (by /type/object/name)
     - unnamed MID (m./g.) -> non_text_entity_list
     - literal (date/number/string) -> non_text_entity_list (as string)
   append relations + h/r/t index triples, dedup.

Output: *_virtuoso_patched.pkl (originals preserved).

Usage
-----
  # smoke (5 cases, verbose)
  python scripts/repair_subgraph_virtuoso.py --dataset cwq --limit 5 --verbose
  # full
  python scripts/repair_subgraph_virtuoso.py --dataset cwq
  python scripts/repair_subgraph_virtuoso.py --dataset webqsp
"""
import argparse, json, os, pickle, re, sys, threading, time, urllib.parse, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ENDPOINT = os.environ.get("VIRTUOSO_ENDPOINT", "http://localhost:8890/sparql")
ROOT = os.environ.get("CGSD_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NS = "http://rdf.freebase.com/ns/"
DATASETS = {
    # from-RoG reconstruction entries: base pkls produced by
    # convert_rog_to_pkl.py from the public RoG parquet releases (see README §4).
    # These are the recommended path when rebuilding the data from sources.
    "cwq_test_from_rog": {
        "pkl": f"{ROOT}/data/cwq_processed/test_rog.pkl",
        "out": f"{ROOT}/data/cwq_processed/test_virt_r2.pkl",
        "sparql": f"{ROOT}/data/cwq_sparql/test.json",        # list[{ID, sparql}]
        "id_key": "ID", "sparql_key": "sparql",
    },
    "cwq_train_from_rog": {
        "pkl": f"{ROOT}/data/cwq_processed/train_rog.pkl",
        "out": f"{ROOT}/data/cwq_processed/train_virt_r2.pkl",
        "sparql": f"{ROOT}/data/cwq_sparql_full/train.json",  # dict{ID: {sparql}}
        "id_key": "ID", "sparql_key": "sparql",
    },
    "webqsp_test_from_rog": {
        "pkl": f"{ROOT}/data/webqsp/test_rog.pkl",
        "out": f"{ROOT}/data/webqsp/test_virt_r2.pkl",
        "sparql": f"{ROOT}/data/webqsp/WebQSP.test.json",     # {Questions:[{QuestionId,Parses}]}
        "id_key": "QuestionId",
        "sparql_key": "Parses",       # special: list of {Sparql}
    },
    "webqsp_train_from_rog": {
        "pkl": f"{ROOT}/data/webqsp/train_rog.pkl",
        "out": f"{ROOT}/data/webqsp/train_virt_r2.pkl",
        "sparql": f"{ROOT}/data/webqsp/WebQSP.train.sparql.json",  # dict{QuestionId: sparql}
        "id_key": "ID", "sparql_key": "sparql",
    },
    # round-2 internal snapshots (author-side lineage; files not in this repo).
    "cwq_test_r2": {
        "pkl": f"{ROOT}/data/cwq_processed/test_v4_repaired.pkl",
        "out": f"{ROOT}/data/cwq_processed/test_virt_r2.pkl",
        "sparql": f"{ROOT}/data/cwq_sparql/test.json",        # list[{ID, sparql}]
        "id_key": "ID", "sparql_key": "sparql",
    },
    "cwq_train_r2": {
        "pkl": f"{ROOT}/data/cwq_processed/train_v4_repaired.pkl",
        "out": f"{ROOT}/data/cwq_processed/train_virt_r2.pkl",
        "sparql": f"{ROOT}/data/cwq_sparql_full/train.json",  # dict{ID: {sparql}}
        "id_key": "ID", "sparql_key": "sparql",
    },
    "webqsp_train_r2": {
        "pkl": f"{ROOT}/data/webqsp/train_sparql_patched.pkl",
        "out": f"{ROOT}/data/webqsp/train_virt_r2.pkl",
        "sparql": f"{ROOT}/data/webqsp/WebQSP.train.sparql.json",  # dict{QuestionId: sparql}
        "id_key": "ID", "sparql_key": "sparql",
    },
    # legacy entries below reference files removed in the 2026-10-28 cleanup;
    # kept for history (they will fail loudly on load).
    "cwq": {
        "pkl": f"{ROOT}/data/cwq_processed/test_literal_and_language_fixed_path_completed.pkl",
        "out": f"{ROOT}/data/cwq_processed/test_virtuoso_patched.pkl",
        "sparql": f"{ROOT}/data/cwq_sparql/test.json",        # list[{ID, sparql}]
        "id_key": "ID",
        "sparql_key": "sparql",
    },
    "cwq_train": {
        "pkl": f"{ROOT}/data/cwq_processed/train_rog.pkl",
        "out": f"{ROOT}/data/cwq_processed/train_sparql_patched.pkl",
        "sparql": f"{ROOT}/data/cwq_sparql_full/train.json",   # dict{ID: {sparql}}
        "id_key": "ID", "sparql_key": "sparql",
    },
    "cwq_val": {
        "pkl": f"{ROOT}/data/cwq_processed/validation_rog.pkl",
        "out": f"{ROOT}/data/cwq_processed/validation_sparql_patched.pkl",
        "sparql": f"{ROOT}/data/cwq_sparql_full/validation.json",
        "id_key": "ID", "sparql_key": "sparql",
    },
    "webqsp": {
        "pkl": f"{ROOT}/data/webqsp/test_fixed_path_completed.pkl",
        "out": f"{ROOT}/data/webqsp/test_virtuoso_patched.pkl",
        "sparql": f"{ROOT}/data/webqsp/WebQSP.test.json",      # {Questions:[{QuestionId,Parses}]}
        "id_key": "QuestionId",
        "sparql_key": "Parses",       # special: list of {Sparql}
    },
    "webqsp_train": {
        "pkl": f"{ROOT}/data/webqsp/train_rog.pkl",
        "out": f"{ROOT}/data/webqsp/train_sparql_patched.pkl",
        "sparql": f"{ROOT}/data/webqsp/WebQSP.train.sparql.json",  # dict{QuestionId: sparql}
        "id_key": "ID", "sparql_key": "sparql",
    },
}

# ---------------------------------------------------------------------------
# Virtuoso SPARQL client (stdlib only; respects no proxy — it's localhost)
# ---------------------------------------------------------------------------
# localhost endpoint — never route through http_proxy (the proxy 502s localhost).
_NOPROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def sparql_query(query, timeout=60):
    """Execute SPARQL against Virtuoso, return list of binding dicts."""
    url = ENDPOINT + "?" + urllib.parse.urlencode({
        "query": query, "format": "json", "timeout": str(timeout * 1000)})
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    last = None
    for attempt in range(2):
        try:
            with _NOPROXY_OPENER.open(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data.get("results", {}).get("bindings", [])
        except Exception as e:
            last = e; time.sleep(1.5 * (attempt + 1))
    print(f"  [sparql] query failed after retries: {last}", file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# Gold SPARQL loading
# ---------------------------------------------------------------------------
def load_gold_sparql(cfg):
    raw = json.load(open(cfg["sparql"]))
    out = {}
    if isinstance(raw, dict) and "Questions" in raw:    # WebQSP {Questions:[{QuestionId,Parses}]}
        for q in raw["Questions"]:
            parses = q.get("Parses", []) or []
            if parses and parses[0].get("Sparql"):
                out[q["QuestionId"]] = parses[0]["Sparql"]
    elif isinstance(raw, dict):                         # CWQ train/val: {ID: {sparql,...}} (or id->str)
        for k, v in raw.items():
            out[k] = v.get(cfg["sparql_key"], "") if isinstance(v, dict) else v
    elif cfg["id_key"] == "ID":                         # CWQ test: list of {ID, sparql}
        for d in raw:
            out[d["ID"]] = d.get(cfg["sparql_key"], "")
    return out


# ---------------------------------------------------------------------------
# SPARQL BGP parsing
# ---------------------------------------------------------------------------
# captures triple patterns: subject predicate object  (until . or })
# 2026-10-28: also accept GrailQA's ':' prefix dialect (:m.xxx / :rel.name)
TRIPLE_RE = re.compile(
    r"(\?\w+|ns:[\w./-]+|:[\w./-]+|<[^>]+>)\s+"             # subject
    r"(ns:[\w./-]+|:[\w./-]+|<[^>]+>|a)\s+"                   # predicate
    r"(\?\w+|ns:[\w./-]+|:[\w./-]+|<[^>]+>|\"[^\"]*\"(?:\^\^[^ ]+|@[A-Za-z-]+)?)\s*"
    r"(?:\.|\})", re.DOTALL)

def parse_bgp(sparql):
    """Return list of (subj_tok, pred_tok, obj_tok) triple patterns from WHERE."""
    m = re.search(r"\bWHERE\s*\{", sparql, re.IGNORECASE)
    if not m:
        return []
    # extract the WHERE body (brace-matched, one level is enough for these queries)
    start = m.end(); depth = 1; i = start
    while i < len(sparql) and depth > 0:
        if sparql[i] == "{": depth += 1
        elif sparql[i] == "}": depth -= 1
        i += 1
    body = sparql[start:i-1]
    pats = []
    for grp in TRIPLE_RE.finditer(body):
        s, p, o = grp.group(1), grp.group(2), grp.group(3)
        # only keep freebase-ns predicates (skip rdf:type 'a', misc)
        if p == "a" or not (p.startswith("ns:") or p.startswith(":") or p.startswith("<")):
            continue
        pats.append((s, p, o))
    return pats


def to_select_star(sparql):
    """Replace SELECT projections with SELECT * — for GrailQA's nested form
    (outer 'SELECT (?x0 AS ?value) WHERE { SELECT DISTINCT ?x0 WHERE { ...')
    BOTH levels must expand, else intermediate vars never bind."""
    s = re.sub(r"\bSELECT\b\s+(?:DISTINCT\s+|REDUCED\s+)?[^.]*?\bWHERE\b",
               "SELECT * WHERE", sparql, count=1, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"\bSELECT\s+DISTINCT\s+\?\w+\s+WHERE\b",
               "SELECT * WHERE", s, flags=re.IGNORECASE)   # inner subselect
    if not re.search(r"\bLIMIT\b", s, re.IGNORECASE):
        s = s.rstrip().rstrip(";") + "\nLIMIT 2000"
    return s


# ---------------------------------------------------------------------------
# Term helpers
# ---------------------------------------------------------------------------
def term_value(tok, binding):
    """Resolve a triple-pattern token to a concrete (kind, value) given a binding row.
    kind in {'mid','literal','uri','const'}; value is the string."""
    if tok.startswith("?"):
        b = binding.get(tok[1:])   # SPARQL JSON binds vars by name without '?'
        if not b:
            return None
        typ, val = b.get("type"), b.get("value", "")
        if typ == "uri":
            if val.startswith(NS):
                mid = val[len(NS):]
                if re.match(r"^[mg]\.\w", mid):
                    return ("mid", mid)
            return ("uri", val)
        return ("literal", val)              # typed/plain literal
    if tok.startswith("ns:"):
        return ("mid", tok[3:])
    if tok.startswith(":"):                        # GrailQA dialect
        v = tok[1:]
        return ("mid", v) if re.match(r"^[mg]\.\w", v) else ("literal", v)
    if tok.startswith("<"):
        v = tok[1:-1]
        return ("mid", v[len(NS):]) if v.startswith(NS) else ("uri", v)
    if tok.startswith('"'):
        return ("literal", tok.strip('"'))
    return ("const", tok)


def pred_local(tok):
    if tok.startswith("ns:"):
        return tok[3:]
    if tok.startswith(":"):                        # GrailQA dialect
        return tok[1:]
    if tok.startswith("<") and tok[1:-1].startswith(NS):
        return tok[1:-1][len(NS):]
    return None


def uri_mid(b):
    """Extract a Freebase m./g. MID from a binding's uri value, or None."""
    v = b.get("value", "") if isinstance(b, dict) else ""
    if v.startswith(NS):
        mid = v[len(NS):]
        if re.match(r"^[mg]\.\w", mid):
            return mid
    return None


# ---------------------------------------------------------------------------
# Entity name resolution (cached, batched, thread-safe)
# ---------------------------------------------------------------------------
_NAME_CACHE = {}
_NAME_LOCK = threading.Lock()
_NAME_BATCH_SIZE = 400

def get_name(mid):
    """Return the English /type/object/name for a Freebase MID, cached."""
    return _NAME_CACHE.get(mid, "")

def resolve_names(mids):
    """Batch-resolve /type/object.name for a set of MIDs (one VALUES query per
    BATCH_SIZE). Populates _NAME_CACHE. Thread-safe (coarse lock)."""
    todo = [m for m in set(mids) if m and m not in _NAME_CACHE]
    if not todo:
        return
    with _NAME_LOCK:
        todo = [m for m in todo if m not in _NAME_CACHE]   # re-check under lock
        if not todo:
            return
        # mark as in-progress to avoid duplicate work
        for m in todo:
            _NAME_CACHE[m] = _NAME_CACHE.get(m, "")
    for i in range(0, len(todo), _NAME_BATCH_SIZE):
        chunk = todo[i:i + _NAME_BATCH_SIZE]
        vals = " ".join("ns:%s" % m for m in chunk)
        q = ("PREFIX ns: <http://rdf.freebase.com/ns/>\n"
             "SELECT ?s ?n WHERE { VALUES ?s { %s } "
             "?s ns:type.object.name ?n . FILTER(lang(?n)='en') }" % vals)
        rows = sparql_query(q, timeout=60)
        found = {}
        for b in rows:
            s = b["s"]["value"]; n = b["n"]["value"].strip()
            if s.startswith(NS):
                found.setdefault(s[len(NS):], n)
        with _NAME_LOCK:
            for m in chunk:
                _NAME_CACHE[m] = found.get(m, "")   # "" = looked up, no english name


# ---------------------------------------------------------------------------
# Per-case repair
# ---------------------------------------------------------------------------
def repair_case(rec, gold_sparql, add_name=True, verbose=False):
    """Mutate rec in place: add gold-path triples materialized via Virtuoso.
    Returns stats dict."""
    te = rec["text_entity_list"]; nte = rec["non_text_entity_list"]
    rels = rec["relation_list"]
    h, r, t = rec["h_id_list"], rec["r_id_list"], rec["t_id_list"]
    nodes = te + nte
    node_idx = {}
    for i, n in enumerate(nodes):
        node_idx.setdefault(n, i)            # first occurrence wins
    rel_idx = {rr: i for i, rr in enumerate(rels)}
    existing_triples = set(zip(h, r, t))
    orig_rel_set = set(rel_idx.keys())     # snapshot before any append

    # 2026-10-29 INDEX-SPACE FIX: node_slot now returns a MARKER, not an
    # index. The old idx = len(te)+len(nte)-1 / len(te)-1 formulas collide
    # when TE and NTE growth interleaves (a later TE append shifts every NTE
    # index) — repaired pkls came out with permuted (h, r, t) targets. The
    # stable layout is materialized ONCE after all passes (see the remap
    # block before h.extend).
    _reg_order, _reg_named = [], {}

    def node_slot(kind, value):
        """Return ('E', idx) for an existing node key or ('N', key) for a
        newly registered one. Markers hash/dedup; indices come later."""
        if kind == "literal":
            key = value
            named = False
        else:
            name = get_name(value) if add_name else ""
            key = name or value
            named = bool(name) and not key.startswith("g.")
        if key in node_idx:
            return ("E", node_idx[key])
        if key not in _reg_named:
            _reg_named[key] = named
            _reg_order.append(key)
        return ("N", key)

    bgp = parse_bgp(gold_sparql)
    if not bgp:
        return {"n_new": 0, "reason": "no BGP parsed", "n_bind": 0}
    sel = to_select_star(gold_sparql)
    rows = sparql_query(sel, timeout=90)
    # collect every MID that will be materialized (binding vars + concrete tokens),
    # then batch-resolve english names in a single VALUES query (huge speedup)
    mid_terms = set()
    for (s, p, o) in bgp:
        for tok in (s, o):
            if tok.startswith("ns:") and re.match(r"ns:[mg]\.", tok):
                mid_terms.add(tok[3:])
    for b in rows:
        for (s, p, o) in bgp:
            sv = term_value(s, b); ov = term_value(o, b)
            if sv and sv[0] == "mid": mid_terms.add(sv[1])
            if ov and ov[0] == "mid": mid_terms.add(ov[1])
    if mid_terms:
        resolve_names(mid_terms)
    n_new = 0; new_triples = []
    for b in rows:
        for (s, p, o) in bgp:
            prel = pred_local(p)
            if not prel:
                continue
            sv = term_value(s, b); ov = term_value(o, b)
            if not sv or not ov:
                continue
            hi = node_slot(sv[0], sv[1])
            ti = node_slot(ov[0], ov[1])
            if prel not in rel_idx:
                rels.append(prel); rel_idx[prel] = len(rels) - 1
            ri = rel_idx[prel]
            if (hi, ri, ti) in existing_triples:
                continue
            existing_triples.add((hi, ri, ti))
            new_triples.append((hi, ri, ti)); n_new += 1
    # ---- second pass: relations not materialized by SELECT * (CVT attributes
    # hidden inside FILTER(EXISTS) — their object var isn't returned). Query the
    # subject MID(s) directly for those relations.
    added_rels = {rels[x[1]] for x in new_triples}
    var_mids = {}
    for b in rows:
        for (s, p, o) in bgp:
            if s.startswith("?"):
                sv = term_value(s, b)
                if sv and sv[0] == "mid":
                    var_mids.setdefault(s, set()).add(sv[1])
    sp2 = []
    for (s, p, o) in bgp:
        pl = pred_local(p)
        if not pl or pl in orig_rel_set or pl in added_rels:
            continue
        if s.startswith("?"):
            smids = var_mids.get(s, set())
        elif s.startswith("ns:") and re.match(r"ns:[mg]\.", s):
            smids = {s[3:]}
        else:
            smids = set()
        if not smids:
            continue
        vals = " ".join("ns:%s" % m for m in list(smids)[:60])
        q = ("PREFIX ns: <http://rdf.freebase.com/ns/>\n"
             "SELECT ?s ?o WHERE { VALUES ?s { %s } ?s ns:%s ?o } LIMIT 500"
             % (vals, pl))
        for b2 in sparql_query(q, timeout=45):
            sm = uri_mid(b2.get("s", {})); ob = b2.get("o", {})
            if not sm:
                continue
            if ob.get("type") == "uri":
                om = uri_mid(ob)
                o_term = ("mid", om) if om else ("literal", ob.get("value", ""))
            else:
                o_term = ("literal", ob.get("value", ""))
            sp2.append((sm, pl, o_term))
    if sp2:
        resolve_names([s for s, _, o in sp2] + [o[1] for _, _, o in sp2 if o[0] == "mid"])
        for sm, pl, o_term in sp2:
            hi = node_slot("mid", sm)
            ti = node_slot(o_term[0], o_term[1])
            if pl not in rel_idx:
                rels.append(pl); rel_idx[pl] = len(rels) - 1
            ri = rel_idx[pl]
            if (hi, ri, ti) in existing_triples:
                continue
            existing_triples.add((hi, ri, ti))
            new_triples.append((hi, ri, ti)); n_new += 1
    # ---- pass 3 (2026-10-28 round-2): subgraph-guided completion. Gold
    # relations still absent by NAME (CVT terminal attrs: sports_team_roster.to,
    # government_position_held.to, ...) are queried directly on MIDs already
    # present in the subgraph — no dependency on the gold query's bindings
    # (pass 2's blind spot: the CVT var sits inside FILTER/EXISTS and never
    # binds). Subjects are pruned by relation FAMILY (gov...position_held.*)
    # to the MIDs touched by existing same-family edges; fallback = all NTE mids.
    missing_names = []
    for (_s, p, _o) in bgp:
        pl = pred_local(p)
        if pl and pl not in rel_idx and pl not in missing_names:
            missing_names.append(pl)
    if missing_names:
        nodes_snapshot = te + nte
        fam_mids = defaultdict(set)                 # family -> mids in its edges
        for _i in range(len(h)):
            rr = rels[r[_i]]
            if "." not in rr:
                continue
            fam = rr.rsplit(".", 1)[0]
            for nid in (h[_i], t[_i]):
                nm = nodes_snapshot[nid]
                if re.match(r"^[mg]\.\w", nm):
                    fam_mids[fam].add(nm)
        cand_all = [n for n in nte if re.match(r"^[mg]\.\w", n)]
        p3 = []
        for pl in missing_names:
            fam = pl.rsplit(".", 1)[0]
            subj = sorted(fam_mids.get(fam) or cand_all)
            for i in range(0, len(subj), 60):
                vals = " ".join("ns:%s" % m for m in subj[i:i + 60])
                q = ("PREFIX ns: <http://rdf.freebase.com/ns/>\n"
                     "SELECT ?s ?o WHERE { VALUES ?s { %s } ?s ns:%s ?o } "
                     "LIMIT 200" % (vals, pl))
                for b2 in sparql_query(q, timeout=45):
                    sm = uri_mid(b2.get("s", {}))
                    ob = b2.get("o", {})
                    if not sm:
                        continue
                    if ob.get("type") == "uri":
                        om = uri_mid(ob)
                        o_term = ("mid", om) if om else ("literal", ob.get("value", ""))
                    else:
                        o_term = ("literal", ob.get("value", ""))
                    p3.append((sm, pl, o_term))
        if p3:
            resolve_names([s for s, _, _ in p3] +
                          [o[1] for _, _, o in p3 if o[0] == "mid"])
            for sm, pl, o_term in p3:
                hi = node_slot("mid", sm)
                ti = node_slot(o_term[0], o_term[1])
                if pl not in rel_idx:
                    rels.append(pl); rel_idx[pl] = len(rels) - 1
                ri = rel_idx[pl]
                if (hi, ri, ti) in existing_triples:
                    continue
                existing_triples.add((hi, ri, ti))
                new_triples.append((hi, ri, ti)); n_new += 1
    # ---- pass 4 (round-2b): two-step targeted completion. Predicates still
    # missing EXIST in the dump (verified: place_lived.start_date etc.) but
    # their CVT subjects never entered the subgraph, so pass 3 cannot see
    # them. Query the dump directly: any mid within 2 hops of the topic
    # entities -> (entry edge) -> subject -> missing relation -> object.
    missing4 = []
    for (_s, p, _o) in bgp:
        pl = pred_local(p)
        if pl and pl not in rel_idx and pl not in missing4:
            missing4.append(pl)
    if missing4:
        nodes4 = te + nte
        # 2-hop mid set around the topic entities (BFS over pkl edges)
        start = [i for i in rec.get("q_entity_id_list", []) if 0 <= i < len(nodes4)]
        adj = defaultdict(list)
        for _i in range(len(h)):
            adj[h[_i]].append(t[_i]); adj[t[_i]].append(h[_i])
        seen2, dq = set(start), list(start)
        for _d in (0, 1):                       # expand two layers
            for _ in range(len(dq)):
                u = dq.pop(0)
                for v in adj.get(u, ()): 
                    if v not in seen2:
                        seen2.add(v); dq.append(v)
        mids2 = [nodes4[i] for i in seen2
                 if i >= len(te) and re.match(r"^[mg]\.\w", nodes4[i])][:200]
        p4 = []
        if mids2:
            for pl in missing4:
                vals = " ".join("ns:%s" % m for m in mids2)
                # (a) direct: the CVT itself is the subject (it sits 1 hop from
                #     the topic entity, whose entry-edge family differs from
                #     the missing relation's family — pass 3's blind spot)
                # (b) two-step: mid -> entry edge -> subject -> missing rel
                for q in (
                    "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
                    "SELECT ?s ?o WHERE { VALUES ?s { %s } ?s ns:%s ?o } LIMIT 200"
                    % (vals, pl),
                    "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
                    "SELECT ?s ?o WHERE { VALUES ?m { %s } "
                    "?m ?pe ?s . ?s ns:%s ?o . "
                    "FILTER(STRSTARTS(STR(?pe), \"http://rdf.freebase.com/ns/\")) } "
                    "LIMIT 200" % (vals, pl),
                ):
                    for b2 in sparql_query(q, timeout=60):
                        sm = uri_mid(b2.get("s", {}))
                        ob = b2.get("o", {})
                        if not sm:
                            continue
                        if ob.get("type") == "uri":
                            om = uri_mid(ob)
                            o_term = ("mid", om) if om else ("literal", ob.get("value", ""))
                        else:
                            o_term = ("literal", ob.get("value", ""))
                        p4.append((sm, pl, o_term))
        if p4:
            resolve_names([s for s, _, _ in p4] +
                          [o[1] for _, _, o in p4 if o[0] == "mid"])
            for sm, pl, o_term in p4:
                hi = node_slot("mid", sm)
                ti = node_slot(o_term[0], o_term[1])
                if pl not in rel_idx:
                    rels.append(pl); rel_idx[pl] = len(rels) - 1
                ri = rel_idx[pl]
                if (hi, ri, ti) in existing_triples:
                    continue
                existing_triples.add((hi, ri, ti))
                new_triples.append((hi, ri, ti)); n_new += 1
    # deferred index materialization: new TE nodes shift existing NTE indices
    new_te = [k for k in _reg_order if _reg_named[k]]
    new_nte = [k for k in _reg_order if not _reg_named[k]]
    k_new = len(new_te)
    n_te0, n_nte0 = len(te), len(nte)
    final = {}
    for _i, _k in enumerate(new_te):
        final[("N", _k)] = n_te0 + _i
    for _j, _k in enumerate(new_nte):
        final[("N", _k)] = n_te0 + k_new + n_nte0 + _j

    def _shift(_x):
        return _x + k_new if _x >= n_te0 else _x

    h2 = [_shift(x) for x in h]
    t2 = [_shift(x) for x in t]
    rec["q_entity_id_list"] = [_shift(x) for x in rec.get("q_entity_id_list", [])]
    rec["a_entity_id_list"] = [_shift(x) for x in rec.get("a_entity_id_list", [])]
    te = te + new_te
    nte = nte + new_nte
    base_existing = set(zip(h2, r, t2))
    out_new = []
    for _hm, _ri, _tm in new_triples:
        _hi = _hm[1] if _hm[0] == "E" else final[_hm]
        _ti = _tm[1] if _tm[0] == "E" else final[_tm]
        if (_hi, _ri, _ti) in base_existing:
            continue
        base_existing.add((_hi, _ri, _ti))
        out_new.append((_hi, _ri, _ti))
    h2.extend([x[0] for x in out_new])
    r.extend([x[1] for x in out_new])
    t2.extend([x[2] for x in out_new])
    n_new = len(out_new)
    rec["text_entity_list"] = te; rec["non_text_entity_list"] = nte
    rec["relation_list"] = rels
    rec["h_id_list"] = h2; rec["r_id_list"] = r; rec["t_id_list"] = t2
    return {"n_new": n_new, "n_bind": len(rows), "n_bgp": len(bgp)}


# ---------------------------------------------------------------------------
# Closed-loop verification
# ---------------------------------------------------------------------------
def verify_case(rec, gold_sparql):
    """Return # structurally-missing gold edges after repair."""
    te = rec["text_entity_list"]; nte = rec["non_text_entity_list"]
    nodes = te + nte
    rels = rec["relation_list"]; h, r, t = rec["h_id_list"], rec["r_id_list"], rec["t_id_list"]
    rel_idx = {rr: i for i, rr in enumerate(rels)}
    edges = defaultdict(set)
    for i in range(len(h)):
        edges[(nodes[h[i]], rels[r[i]])].add(nodes[t[i]])
    bgp = parse_bgp(gold_sparql)
    miss = 0
    for (s, p, o) in bgp:
        prel = pred_local(p)
        if prel is None or not s.startswith("ns:"):
            continue
        s_mid = s[3:]
        # subgraph stores named entities by NAME (TE), unnamed/CVT by MID (NTE);
        # check both the MID and its resolved name to avoid false "missing"
        keys = {s_mid}
        nm = get_name(s_mid)
        if nm:
            keys.add(nm)
        if not any(edges.get((k, prel)) for k in keys):
            miss += 1
    return miss, len(bgp)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def gold_missing_rels(rec, sq):
    """Local (no Virtuoso) check: which gold relations are absent from the
    subgraph's relation_list. Used to pre-filter — only cases with a missing
    relation need an (expensive) Virtuoso SELECT *."""
    have = set(rec["relation_list"])
    miss = set()
    for _, p, _ in parse_bgp(sq):
        pl = pred_local(p)
        if pl and pl not in have:
            miss.add(pl)
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--cases", default="", help="comma-list of case ids to repair (smoke)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="don't write output pkl")
    ap.add_argument("--workers", type=int, default=12, help="concurrent Virtuoso queries")
    ap.add_argument("--skip-complete", dest="skip_complete", action="store_true", default=True,
                    help="skip cases whose gold relations are all present (default)")
    ap.add_argument("--no-skip-complete", dest="skip_complete", action="store_false")
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]

    recs = pickle.load(open(cfg["pkl"], "rb"))
    gold = load_gold_sparql(cfg)
    sel_ids = [s.strip() for s in args.cases.split(",")] if args.cases else None
    def id_matches(cid):
        if not sel_ids:
            return True
        return any(cid == s or cid.startswith(s) for s in sel_ids)

    targets = []; skipped = 0
    for r in recs:
        cid = r["id"]
        if cid not in gold or not id_matches(cid):
            continue
        sq = gold[cid]
        if args.skip_complete and not gold_missing_rels(r, sq):
            skipped += 1
            continue
        targets.append((cid, r, sq))
        if args.limit and len(targets) >= args.limit:
            break
    print(f"[{args.dataset}] {len(targets)} to repair (skipped {skipped} already-complete) "
          f"[workers={args.workers}]", flush=True)

    def work(item):
        cid, r, sq = item
        return cid, repair_case(r, sq, verbose=False)

    t0 = time.time(); n_new_tot = 0; n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, it) for it in targets]
        for fut in as_completed(futs):
            cid, st = fut.result()
            n_done += 1; n_new_tot += st.get("n_new", 0)
            if args.verbose or (n_done and n_done % 1000 == 0):
                print(f"  [{n_done}/{len(targets)}] {cid[:28]}: +{st.get('n_new',0)} "
                      f"(bind={st.get('n_bind',0)})", flush=True)
    dt = time.time() - t0
    rate = n_done / max(dt, 1)
    print(f"\n[done] {n_done} cases in {dt:.0f}s ({rate:.1f}/s); +{n_new_tot} triples", flush=True)

    if not args.dry_run and targets:
        pickle.dump(recs, open(cfg["out"], "wb"))
        print(f"  wrote {cfg['out']} ({os.path.getsize(cfg['out'])//1024//1024} MB)", flush=True)


if __name__ == "__main__":
    main()
