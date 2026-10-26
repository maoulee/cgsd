#!/usr/bin/env python3
"""Re-score trajectory jsonl with the CURRENT agent_stage_scorer.

The current scorer's recall source = walk-result (expand triples endpoints) +
candidates + CVT attr values (expand_full) — broader than the older
candidates-only version that under-counted anchor/intermediate entities
(e.g. Castlemont as gold, present in triples but excluded from candidates).

Reconstructs agent_trajectory from chat messages (assistant + "Tool result (X)"
user msgs) and runs score_case. Writes corrected S_plan/S_select/S_reason.
"""
import json, sys, argparse, statistics
from collections import Counter
sys.path.insert(0, '.')
from scripts.agent_stage_scorer import score_case


def reconstruct_trajectory(messages):
    """Chat messages → agent_trajectory [{role:assistant/tool, name, content}]."""
    traj = []
    for m in messages:
        role = m.get("role"); c = m.get("content", "") or ""
        if role == "assistant":
            traj.append({"role": "assistant", "content": c})
        elif role == "user" and c.startswith("Tool result ("):
            nm = c[len("Tool result ("):].split(")")[0]
            traj.append({"role": "tool", "name": nm, "content": c.split(":", 1)[1].strip()})
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="reports/decompose_audit/traj_100_greedy.jsonl")
    ap.add_argument("--out", default="reports/decompose_audit/traj_100_greedy_rescored.jsonl")
    args = ap.parse_args()
    R = [json.loads(l) for l in open(args.traj)]
    out = []
    for r in R:
        traj = reconstruct_trajectory(r.get("messages", []))
        case = {"case_id": r["case_id"], "question": r["question"],
                "gt_answers": r.get("gt_answers"), "llm_answer": r.get("llm_answer", ""),
                "answer_candidates": r.get("answer_candidates", []), "agent_trajectory": traj}
        try:
            sc = score_case(case)
        except Exception as e:
            sc = {"S_plan": None, "S_select": None, "S_reason": None, "notes": [f"scorer_error:{e}"]}
        rec = dict(r)
        rec["S_plan_old"] = r.get("S_plan"); rec["S_select_old"] = r.get("S_select"); rec["S_reason_old"] = r.get("S_reason")
        rec["S_plan"] = sc.get("S_plan"); rec["S_select"] = sc.get("S_select"); rec["S_reason"] = sc.get("S_reason")
        rec["scorer_notes"] = sc.get("notes", [])
        out.append(rec)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(out)
    def ms(k, new=True):
        v = [r[("S_"+k) if new else ("S_"+k+"_old")] for r in out]
        v = [x for x in v if x is not None]
        return statistics.mean(v) if v else 0.0
    print(f"=== re-scored {n} cases ===")
    print(f"{'stage':<10}{'old':>8}{'new':>8}{'delta':>8}")
    for st in ["plan", "select", "reason"]:
        o = ms(st, new=False); nw = ms(st, new=True)
        print(f"  S_{st:<7}{o:>8.3f}{nw:>8.3f}{nw-o:>+8.3f}")
    # the key bucket: plan=1 & select=1 & reason<1
    p1s1rL = [r for r in out if (r["S_plan"] or 0) >= 0.999 and (r["S_select"] or 0) >= 0.999 and (r["S_reason"] or 0) < 0.999]
    print(f"\n=== plan=1 & select=1 & reason<1 (answer-stage failures, upstream correct): {len(p1s1rL)}/{n} ===")
    for r in p1s1rL[:15]:
        gt = r.get("gt_answers"); ans = r.get("llm_answer")
        print(f"  {r['case_id'][:16]} S_reason={r['S_reason']:.2f} F1={r.get('llm_f1',0):.2f} | gold={str(gt)[:40]} | ans={str(ans)[:40]}")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
