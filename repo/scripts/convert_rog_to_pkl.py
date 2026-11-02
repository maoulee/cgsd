#!/usr/bin/env python3
"""
Convert RoG-format subgraph parquet (rmanluo/RoG-CWQ, ml1996/webqsp, RoG-WebQSP)
into the pkl format the agent loads:

  id, question, q_entity, q_entity_id_list, text_entity_list, non_text_entity_list,
  relation_list, h_id_list, r_id_list, t_id_list, a_entity, a_entity_id_list

Bucketing convention (reverse-engineered from the existing test pkls):
  - m./g. Freebase MIDs  -> non_text_entity_list (NTE)
  - date / number / slash-key literals -> NTE
  - clean entity names -> text_entity_list (TE)
Node index = position in (TE ++ NTE).

Usage:
  python scripts/convert_rog_to_pkl.py --dataset cwq   --split train
  python scripts/convert_rog_to_pkl.py --dataset cwq   --split test   --validate
  python scripts/convert_rog_to_pkl.py --dataset webqsp --split train
"""
import argparse, glob, os, pickle, re, sys
import pyarrow.parquet as pqf

ROOT = os.environ.get("CGSD_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCES = {
    "cwq":    f"{ROOT}/data/rog_cwq_raw/data",         # rmanluo/RoG-CWQ
    "webqsp": f"{ROOT}/data/ml1996_webqsp_raw/data",   # ml1996/webqsp (fallback RoG-WebQSP)
}
OUTDIR = {"cwq": f"{ROOT}/data/cwq_processed", "webqsp": f"{ROOT}/data/webqsp"}

MID_RE = re.compile(r"^[mg]\.\w")
DATE_RE = re.compile(r"^\d{4}-\d{2}")
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")

def bucket(node):
    """Return 'nte' if node should go to non_text, else 'te'."""
    if MID_RE.match(node):
        return "nte"
    if DATE_RE.match(node) or NUM_RE.match(node) or node.startswith("/"):
        return "nte"
    return "te"


def convert_record(r):
    te, nte = [], []
    te_idx, nte_idx = {}, {}
    rels, rel_idx = [], {}
    h, rr, t = [], [], []

    def node_id(s):
        if bucket(s) == "nte":
            if s not in nte_idx:
                nte_idx[s] = len(te) + len(nte); nte.append(s)
            return nte_idx[s]
        if s not in te_idx:
            te_idx[s] = len(te); te.append(s)
        return te_idx[s]

    def rel_id(rname):
        if rname not in rel_idx:
            rel_idx[rname] = len(rels); rels.append(rname)
        return rel_idx[rname]

    for hh, rname, tt in r.get("graph", []):
        hi = node_id(hh); ri = rel_id(rname); ti = node_id(tt)
        h.append(hi); rr.append(ri); t.append(ti)

    def ent_ids(names):
        out = []
        for n in (names or []):
            if not n:
                continue
            out.append(node_id(n))
        return out

    return {
        "id": r["id"],
        "question": r.get("question", ""),
        "q_entity": list(r.get("q_entity") or []),
        "q_entity_id_list": ent_ids(r.get("q_entity") or []),
        "text_entity_list": te,
        "non_text_entity_list": nte,
        "relation_list": rels,
        "h_id_list": h, "r_id_list": rr, "t_id_list": t,
        "a_entity": list(r.get("a_entity") or r.get("answer") or []),
        "a_entity_id_list": ent_ids(r.get("a_entity") or r.get("answer") or []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SOURCES), required=True)
    ap.add_argument("--split", required=True, help="train / validation / test")
    ap.add_argument("--validate", action="store_true",
                    help="compare converter output to the existing test pkl bucketing")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = SOURCES[args.dataset]
    shards = sorted(glob.glob(os.path.join(src, f"{args.split}-*.parquet")))
    if not shards:
        sys.exit(f"no parquet shards for split={args.split} in {src}")
    print(f"[{args.dataset}/{args.split}] {len(shards)} shards")
    recs = []
    for sh in shards:
        for r in pqf.read_table(sh).to_pylist():
            recs.append(convert_record(r))
    if args.limit:
        recs = recs[:args.limit]
    print(f"  converted {len(recs)} records")

    if args.validate and args.split in ("test",):
        # compare bucketing vs an existing repaired pkl, if present
        existing = {
            "cwq": f"{ROOT}/data/cwq_processed/test_virt_r2.pkl",
            "webqsp": f"{ROOT}/data/webqsp/test_virt_r2.pkl",
        }[args.dataset]
        if not os.path.exists(existing):
            print("  [validate] skipped (no reference pkl; expected before the repair step)")
        else:
            ex = {r["id"]: r for r in pickle.load(open(existing, "rb"))}
            match = mism = 0
            for r in recs[:500]:
                e = ex.get(r["id"])
                if not e:
                    continue
                ete = set(e["text_entity_list"]); ente = set(e["non_text_entity_list"])
                for n in r["text_entity_list"]:
                    if n in ete: match += 1
                    elif n in ente: mism += 1
                for n in r["non_text_entity_list"]:
                    if n in ente: match += 1
                    elif n in ete: mism += 1
            print(f"  [validate] vs existing test pkl: bucket-match={match} mismatch={mism}")

    outdir = OUTDIR[args.dataset]
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{args.split}_rog.pkl")
    pickle.dump(recs, open(out, "wb"))
    print(f"  wrote {out} ({os.path.getsize(out)//1024//1024} MB)")


if __name__ == "__main__":
    main()
