#!/usr/bin/env python3
"""IG → advantage allocation for SEQ offline GRPO.

Two parts:
  1. alloc_advantages()  — PURE credit allocation (the locked spec). No GPU.
                           Testable now via --simulate.
  2. compute_ig()        — real IG (V_0/V_F/c_i) via OfflineVLLM (frozen base).
                           Needs GPU; run with --data after the rollout frees it.

Locked spec (per trajectory):
  A   = GroupNorm(F1)                      # group-relative; A>0 = relative-good, NOT "correct"
  V_0 = log p_base(y* | q)                 # no evidence
  V_F = log p_base(y* | q, E_full)         # all subgraph triples (NO plan/reasoning text)
  p_0 = e^{V_0},  p_F = e^{V_F}
  c_i = IG-based signed evidence contribution   # NOT Shapley

  Plan:    A>0 → A·max(0, p_F−p_0)         # absolute gain (simple Q auto-weakened, no threshold)
           A<0 → A·(1−p_F)                  # how much info still missing
  Answer:  A>0 → A                         # full (frozen base may under-rate actor)
           A<0 → A·p_F                      # penalize only when evidence was sufficient
  Retrieve:A>0 → a_i = A·max(0,c_i)/Σmax(0,c_j)    # reward helpful SGs
           A<0 → a_i = A·max(0,−c_i)/Σmax(0,−c_j)   # penalize harmful SGs
                   (c_i≈0 → ~0 both;  no c_i<0 & A<0 → retrieve ~0, Plan bears via low p_F)

Responsibility closure:
  wrong retrieval executed (c_i<0)   → Retrieve
  should-retrieve-but-didn't (low p_F, no c_i<0) → Plan
  evidence sufficient but answered wrong (high p_F) → Answer
"""
import json, math, random, re, sys, os
from typing import Any


# ─────────────────────────────────────────────────────────────────────────
# 1. PURE allocation (the locked spec) — no GPU, fully testable
# ─────────────────────────────────────────────────────────────────────────
def alloc_advantages(A: float, p0: float, pF: float,
                     loo_credits: list[tuple[str, float | None]],
                     n_sg: int,
                     forward_credits: list[tuple[str, float | None]] | None = None,
                     p_yhat: float | None = None,
                     r_k: float = 1.0,
                     rho: float = 0.2,
                     rho_plan: float = 0.2) -> dict:
    """CONSERVATION framework: Σa_t = A_k.  (converged design — TWO-LEVEL)

    TWO-LEVEL allocation:
      1. BLOCK LEVEL: info_block (Plan + all SGs) vs Answer.
         info r = (pF−p0)⁺ or (1−pF);  Answer r = F1_k or answer-blame.
         Normalize → A_info, A_ans.
      2. INFO INTERNAL (multi-SG only): Plan gets ρ_plan × A_info (decision);
         SG group gets (1−ρ_plan) × A_info (execution).  SG group split by
         responsibility: r_i = max(L_i⁺, ρ·F_i⁺) for reward, max((−L)⁺, ρ·(−F)⁺)
         for blame.  This prevents Plan from absorbing the full pF−p0 that
         actually belongs to the SGs' joint contribution.

    Single-SG: coarse (info block = Plan+Retrieve as one, no internal split).
    ALL signals in PROBABILITY domain.  Σa_t = A_k (conserved).
    """
    fwd = dict(forward_credits) if forward_credits else {}

    # ── 1. BLOCK LEVEL: info vs Answer ──
    if A > 0:
        r_info, r_ans = max(0.0, pF - p0), max(0.0, r_k)
    else:
        r_info = 1.0 - pF
        r_ans = pF * pF / (pF + (p_yhat or 0) + 1e-8) if p_yhat is not None else pF
    Z_blk = r_info + r_ans + 1e-8
    A_info = A * r_info / Z_blk
    A_ans = A * r_ans / Z_blk

    # ── 2. INFO INTERNAL (multi-SG: Plan vs SGs; single-SG: no split) ──
    if n_sg <= 1:
        advantages = {"info": A_info, "ans": A_ans}
    else:
        A_plan = A_info * rho_plan          # Plan = decision (small share)
        A_sg_grp = A_info * (1.0 - rho_plan)  # SGs = execution (bulk)
        # SG group responsibility scores → internal weights
        sg_raw = []
        for sg, L in loo_credits:
            L = L or 0.0
            F = fwd.get(sg) or 0.0
            if A > 0:
                ri = max(max(0.0, L), rho * max(0.0, F))
            else:
                ri = max(max(0.0, -L), rho * max(0.0, -F))
            sg_raw.append((sg, ri))
        Z_sg = sum(r for _, r in sg_raw)
        advantages = {"plan": A_plan, "ans": A_ans}
        if Z_sg < 1e-8:
            # all SG scores ≈ 0 (e.g. all redundant + no forward rescue)
            # distribute A_sg_grp equally so the budget doesn't vanish
            for sg, _ in sg_raw:
                advantages[sg] = A_sg_grp / len(sg_raw)
        else:
            for sg, ri in sg_raw:
                advantages[sg] = A_sg_grp * ri / Z_sg

    return {"granularity": "coarse" if n_sg <= 1 else "fine",
            "A": A, "advantages": advantages,
            "conservation": sum(advantages.values())}


def _has_mid(r: dict) -> bool:
    return any(re.match(r"^[mg]\.", str(x)) for x in r.get("gold", []))


def _connected(r: dict) -> bool:
    """gold is a retrieved OBJECT (target of a triple), not just substring anywhere.
    Parses 'subj --rel--> obj1 | obj2' from retrieve_subgraph triples — avoids the
    false-positive where gold matches a SUBJECT or relation name (substring-anywhere
    would wrongly pass). For single-SG (anchor-centered retrieval) gold-as-object ≈
    anchor→gold 1-hop connected. NOTE: v1 is object-membership, NOT full multi-hop
    BFS reachability (codex P0-4: a real graph-traversal is the eventual target)."""
    objs = set()
    for m in r.get("messages", []):
        if m.get("role") != "tool":
            continue
        c = m.get("content", "") or ""
        if "triples:" not in c:
            continue
        for line in c.split("\n"):
            for sep in ("-->", "→"):
                if sep in line:
                    rhs = line.split(sep, 1)[1]
                    for o in re.split(r"[|]", rhs):
                        o = o.strip().strip("[]().,;:'\"")
                        # drop bracketed CVT attrs [from=…] and fragments
                        if o and not o.startswith("[") and len(o) > 1:
                            objs.add(o.lower())
                    break
    gold = [str(g).lower().strip() for g in r.get("gold", [])
            if g and not str(g).startswith("g.")]
    return any(g in objs for g in gold)


# ─────────────────────────────────────────────────────────────────────────
# 2. REAL IG computation — needs GPU (OfflineVLLM, frozen base)
# ─────────────────────────────────────────────────────────────────────────
MAX_SG_LINES = 15
END_TAG = "</think>"
SEQ_AGENTS_MD = "./kgqa/agent/SEQ_AGENTS.md"


def _tool_of_assistant(content: str) -> str | None:
    """Tool name from an assistant turn. Parses AFTER the last </think> (so a
    reasoning phrase like 'I should use tool: retrieve_relations' inside <think>
    can't false-match), then takes the LAST 'tool:' match (the actual action)."""
    c = content or ""
    if "</think>" in c:
        c = c.rsplit("</think>", 1)[1]
    ms = re.findall(r"tool:\s*(\w+)", c)
    return ms[-1] if ms else None


def _sg_of_following_result(trajectory: list[dict], asst_idx: int) -> str | None:
    """fact_id (sgX) from the next tool message after the assistant turn at asst_idx."""
    for j in range(asst_idx + 1, len(trajectory)):
        st = trajectory[j]
        if st.get("role") == "tool":
            m = re.search(r"fact_id:\s*(\S+)", st.get("content", "") or "")
            return m.group(1) if m else None
    return None


def build_turn_advantages(trajectory: list[dict], adv: dict) -> list[float]:
    """Map each ASSISTANT turn → its block advantage from adv["advantages"].

    TURN-LEVEL CONSERVATION: each block's advantage is divided by the number of
    assistant turns in that block, so Σ(turn_advs) = Σ(block_advs) = A_k.
    (codex P0-7: without this, a 2-turn SG block would get 2× its advantage.)"""
    g = adv["granularity"]
    advs = adv["advantages"]  # {block_name: advantage_value}

    # precompute sg for retrieve_subgraph turns
    sg_of = {}
    for i, st in enumerate(trajectory):
        if st.get("role") == "assistant" and _tool_of_assistant(st.get("content", "")) == "retrieve_subgraph":
            sg_of[i] = _sg_of_following_result(trajectory, i)
    subgraph_idxs = sorted(sg_of)

    # assign each assistant turn to a block + count turns per block
    blocks, counts = [], {}
    for i, st in enumerate(trajectory):
        if st.get("role") != "assistant":
            continue
        tool = _tool_of_assistant(st.get("content", ""))
        if g == "coarse":
            b = "ans" if tool == "answer" else "info"
        else:
            if tool == "plan":
                b = "plan"
            elif tool == "answer":
                b = "ans"
            elif tool == "retrieve_subgraph":
                b = sg_of.get(i, "_unknown")
            elif tool == "retrieve_relations":
                nxt = next((k for k in subgraph_idxs if k > i), None)
                b = sg_of.get(nxt, "_unknown") if nxt else "_unknown"
            else:
                b = "_unknown"
        blocks.append(b)
        counts[b] = counts.get(b, 0) + 1

    # each turn = block_advantage / n_turns_in_block (turn-level conservation)
    return [advs.get(b, 0.0) / counts[b] for b in blocks]


def build_messages(record: dict, sys_cap: int = 8000) -> list[dict]:
    """Reconstruct the full trainer conversation: system + user(question) +
    trajectory. Each assistant turn's content = <think>{reasoning}</think>{content}
    (the full generation the model produced; template preserves it verbatim).
    """
    sys_text = ""
    try:
        sys_text = open(SEQ_AGENTS_MD).read()[:sys_cap]
    except OSError:
        sys_text = "You are a KGQA agent."
    msgs = [{"role": "system", "content": sys_text},
            {"role": "user", "content": f"Question: {record['question']}"}]
    for st in record.get("trajectory", []):
        if st.get("role") == "assistant":
            reasoning = (st.get("reasoning") or "").strip()
            content = st.get("content", "") or ""
            full = f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content
            msgs.append({"role": "assistant", "content": full})
        elif st.get("role") == "tool":
            msgs.append({"role": "tool", "content": st.get("content", "") or ""})
    return msgs


def _trunc(t: str, n: int = MAX_SG_LINES) -> str:
    lines = t.strip().splitlines()
    return "\n".join(lines[:n]) if len(lines) > n else t


def extract_subgraphs(trajectory: list[dict]) -> dict:
    """Extract {sg_id: triples_text} from retrieve_subgraph tool messages."""
    sgs, order = {}, []
    for st in trajectory:
        if st.get("role") != "tool":
            continue
        name = (st.get("name") or "")
        if "retrieve_subgraph" not in name:
            continue
        c = st.get("content", "") or ""
        m = re.search(r"fact_id:\s*(\S+)", c)
        sg = m.group(1) if m else f"sg{len(sgs) + 1}"
        tm = re.search(r"triples:\s*\n(.+?)(?=\ncandidates:|\nn_candidates:|\nnote:|\n--- T|$)",
                       c, re.DOTALL)
        triples = _trunc(tm.group(1).strip()) if tm else ""
        if triples and sg not in sgs:
            sgs[sg] = triples
            order.append(sg)
    return {s: sgs[s] for s in order}


def _ig_credit(V0, Vseq, Vloo, Vfull):
    """Signed LOO in PROBABILITY domain: L_i = e^{V_F} − e^{V_{−i}}.
    NOT logprob diff — Plan/Answer use prob diffs (p_F−p_0, 1−p_F), so Retrieve
    must too (codex P0-5: scale consistency). None if uncomputable."""
    credits = []
    pF = math.exp(Vfull) if Vfull is not None else None
    for sg in Vseq:
        vl = Vloo.get(sg)
        if pF is not None and vl is not None:
            credits.append((sg, pF - math.exp(vl)))
        else:
            credits.append((sg, None))
    return credits


def compute_ig_for_trajectory(record: dict, llm) -> dict:
    """Compute V_0, V_F, c_i for one trajectory via frozen-base prompt_logprobs.
    Needs GPU (OfflineVLLM.gold_logprob_batch).

    n_sg==1 → only V_0, V_F (2 passes). Single SG has no coalition/compensation,
              so c_info = V_F−V_0 trivially; no LOO/Shapley needed.
    n_sg >1 → V_0 + V_seq(cumulative) + V_loo(per-SG) for signed c_i.
    """
    q = record["question"]
    gold = " | ".join(sorted(record.get("gold", []))) or "unknown"
    sgs = extract_subgraphs(record.get("trajectory", []))
    sg_data = {sg: _trunc(t) for sg, t in sgs.items()}
    n_sg = len(sgs)

    if n_sg <= 1:
        # COARSE: just V_0 and V_F
        prefix0 = f"Question: {q}"
        prefixF = f"Question: {q}\n\nEvidence:\n{sg_data[list(sgs)[0]]}" if sgs else prefix0
        v0, vF = llm.gold_logprob_batch([(prefix0, gold), (prefixF, gold)])
        c_info = (vF - v0) if (v0 is not None and vF is not None) else None
        credits = [(list(sgs)[0], round(c_info, 4))] if (sgs and c_info is not None) else []
        return {"n_sg": n_sg, "V0": v0, "V_F": vF,
                "p0": math.exp(v0) if v0 is not None else None,
                "pF": math.exp(vF) if vF is not None else None, "credits": credits}

    # FINE: V_0 + V_seq + V_loo
    pairs, meta = [], []
    pairs.append((f"Question: {q}", gold)); meta.append(("V0", None))
    acc = ""
    for sg in sgs:
        acc = (acc + "\n" + sg_data[sg]).strip()
        pairs.append((f"Question: {q}\n\nEvidence:\n{acc}", gold)); meta.append(("Vseq", sg))
    for sg in sgs:
        minus = "\n".join(sg_data[s] for s in sgs if s != sg)
        prefix = f"Question: {q}\n\nEvidence:\n{minus}" if minus else f"Question: {q}"
        pairs.append((prefix, gold)); meta.append(("Vloo", sg))
    vals = llm.gold_logprob_batch(pairs)
    V = {}
    for (kind, sg), v in zip(meta, vals):
        if kind == "V0":
            V["V0"] = v
        elif kind == "Vseq":
            V.setdefault("Vseq", {})[sg] = v
        else:
            V.setdefault("Vloo", {})[sg] = v
    V0, Vfull = V["V0"], (V["Vseq"].get(list(sgs)[-1]) if sgs else V["V0"])
    credits = _ig_credit(V0, V.get("Vseq", {}), V.get("Vloo", {}), Vfull)
    return {"n_sg": n_sg, "V0": V0, "V_F": Vfull,
            "p0": math.exp(V0) if V0 is not None else None,
            "pF": math.exp(Vfull) if Vfull is not None else None, "credits": credits}


def _assemble_ig(d: dict, n_sg: int) -> dict:
    """Assemble IG result from a {V0, VF|Vseq, Vloo} dict of precomputed values.
    Saves RAW V values (V0/VF/Vseq/Vloo) so the credit formula can be recomputed
    offline WITHOUT re-running the GPU IG — never store only the final credits."""
    v0 = d.get("V0")
    if n_sg <= 1:
        vF = d.get("VF", v0)
        c_info = (vF - v0) if (v0 is not None and vF is not None) else None
        credits = [(d.get("sg0", "sg1"), c_info)] if c_info is not None else []
        return {"n_sg": n_sg, "V0": v0, "V_F": vF, "Vseq": d.get("VF"), "Vloo": None,
                "p0": math.exp(v0) if v0 is not None else None,
                "pF": math.exp(vF) if vF is not None else None, "credits": credits}
    vseq = d.get("Vseq", {}); vloo = d.get("Vloo", {})
    sgs = list(vseq.keys())
    vF = vseq.get(sgs[-1]) if sgs else v0
    credits = _ig_credit(v0, vseq, vloo, vF)
    return {"n_sg": n_sg, "V0": v0, "V_F": vF, "Vseq": vseq, "Vloo": vloo,
            "p0": math.exp(v0) if v0 is not None else None,
            "pF": math.exp(vF) if vF is not None else None, "credits": credits}


def run_real(data_path: str, out_path: str, limit: int | None = None,
             chunk: int = 1000):
    """Full IG + allocation on rollout data. NEEDS GPU. Batched across trajectories
    (collect ALL prompt pairs → chunked gold_logprob_batch → map back). Single-SG
    skips LOO (2 pairs); multi-SG uses V_seq+V_loo. Output is trainer-ready
    (messages + turn_advantages + advantages).
    """
    from kgqa.llm.offline_vllm import OfflineVLLM
    from collections import defaultdict
    import statistics, time
    recs = json.loads(open(data_path).read())
    if limit:
        recs = recs[:limit]

    # 1. GroupNorm(F1) → A_k (per case, across 8 samples); r_k=F1 preserved
    by_case = defaultdict(list)
    for r in recs:
        by_case[r["case_idx"]].append(r)
    A_of = {}
    for cid, rs in by_case.items():
        f1s = [r["f1"] for r in rs]
        mu, sd = statistics.mean(f1s), statistics.pstdev(f1s) or 1e-6
        for r in rs:
            A_of[id(r)] = (r["f1"] - mu) / sd

    # 2. per-trajectory SG extraction + build ALL (prefix, gold) pairs
    # LENGTH-DEBIAS: teacher-forcing the JOINT gold string multiplies token probs
    # (0.77^20 ≈ 0.02 — a perfect 20-token answer reads as near-zero). Every V is
    # therefore scored PER-ENTITY and averaged in the probability domain:
    # V = log(mean_i exp(logP(gold_i | prefix))). Entity count capped (deterministic
    # sample) to bound the pair blow-up on big-list golds.
    MAX_IG_ENTITIES = 8
    traj_meta = []  # per trajectory: (n_sg, sgs_order, sg_data)
    all_pairs, pair_meta = [], []  # pair_meta: (traj_idx, kind, sg)
    for i, r in enumerate(recs):
        sgs = extract_subgraphs(r.get("trajectory", []))
        sg_data = {sg: _trunc(t) for sg, t in sgs.items()}
        q = r["question"]
        gold_list = sorted(set(str(g) for g in r.get("gold", []) if str(g).strip())) or ["unknown"]
        if len(gold_list) > MAX_IG_ENTITIES:
            rng = random.Random(1234 + i)
            gold_list = sorted(rng.sample(gold_list, MAX_IG_ENTITIES))
        n_sg = len(sgs)
        traj_meta.append((n_sg, list(sgs), sg_data))
        for g in gold_list:
            all_pairs.append((f"Question: {q}", g)); pair_meta.append((i, "V0", None))
            if n_sg <= 1:
                prefixF = (f"Question: {q}\n\nEvidence:\n{sg_data[list(sgs)[0]]}"
                           if sgs else f"Question: {q}")
                all_pairs.append((prefixF, g)); pair_meta.append((i, "VF", "sg1"))
            else:
                acc = ""
                for sg in sgs:
                    acc = (acc + "\n" + sg_data[sg]).strip()
                    all_pairs.append((f"Question: {q}\n\nEvidence:\n{acc}", g))
                    pair_meta.append((i, "Vseq", sg))
                for sg in sgs:
                    minus = "\n".join(sg_data[s] for s in sgs if s != sg)
                    prefix = f"Question: {q}\n\nEvidence:\n{minus}" if minus else f"Question: {q}"
                    all_pairs.append((prefix, g)); pair_meta.append((i, "Vloo", sg))
                # V_alone (forward IG, multi-SG): each SG alone — for ρ·F rescue (France)
                for sg in sgs:
                    all_pairs.append((f"Question: {q}\n\nEvidence:\n{sg_data[sg]}", g))
                    pair_meta.append((i, "Valone", sg))
        # V_yhat: Actor's ACTUAL answer given full evidence — for answer responsibility
        actual = r.get("answer", "")
        if isinstance(actual, list):
            actual = " | ".join(sorted(str(a) for a in actual)) if actual else "unknown"
        _full_ev = "\n".join(sg_data[s] for s in sgs) if sgs else ""
        _pf_full = f"Question: {q}\n\nEvidence:\n{_full_ev}" if _full_ev else f"Question: {q}"
        all_pairs.append((_pf_full, str(actual) or "unknown"))
        pair_meta.append((i, "Vyhat", None))
    n_single = sum(1 for t in traj_meta if t[0] <= 1)
    print(f"  {len(recs)} traj, {len(by_case)} cases, {n_single} single-SG (coarse), "
          f"{len(recs)-n_single} multi-SG (fine)", flush=True)
    print(f"  {len(all_pairs)} IG pairs total → chunked gold_logprob_batch ({chunk}/chunk)",
          flush=True)

    # 3. load frozen-base LLM + batched gold_logprob
    # IG: only need logits (prompt_logprobs). gpu_mem_util=0.75 leaves headroom on
    # GPU 0 (GTE shares it) AND absorbs the prompt_logprobs logits spike that OOMs
    # at 0.9/0.85. Keep CUDA graphs (enforce_eager=False — eager is too slow). Cap
    # max_num_batched_tokens to bound the per-step logits tensor (vocab≈152K).
    print("loading OfflineVLLM (frozen base, IG-tuned: util=0.75) ...", flush=True)
    t = time.time()
    llm = OfflineVLLM(max_model_len=4096, gpu_memory_utilization=0.75,
                      max_num_seqs=64, max_num_batched_tokens=2048)
    print(f"  loaded in {time.time()-t:.0f}s", flush=True)
    t = time.time()
    all_vals = []
    for k in range(0, len(all_pairs), chunk):
        vals = llm.gold_logprob_batch(all_pairs[k:k + chunk])
        all_vals.extend(vals)
        print(f"  IG {min(k+chunk, len(all_pairs))}/{len(all_pairs)} "
              f"({(time.time()-t)/max(k+chunk,1)*1000:.2f}s/pair)", flush=True)

    # 4. map back → assemble IG → allocate → trainer-ready record
    # accumulate PER-ENTITY logprobs as lists, then probability-average:
    # V = log(mean(exp(v_j))) so downstream exp(V) is the length-debiased mean.
    _raw = defaultdict(lambda: defaultdict(list))
    for (ti, kind, sg), v in zip(pair_meta, all_vals):
        if v is None:
            continue
        if kind in ("Vseq", "Vloo", "Valone"):
            _raw[ti][(kind, sg)].append(v)
        else:
            _raw[ti][kind].append(v)

    def _lme(vs):
        """log-mean-exp of a list of logprobs (None-safe)."""
        vs = [x for x in vs if x is not None]
        if not vs:
            return None
        m = max(vs)
        return m + math.log(sum(math.exp(x - m) for x in vs) / len(vs))

    by_traj = defaultdict(dict)
    for ti, kv in _raw.items():
        d = by_traj[ti]
        for key, vs in kv.items():
            lv = _lme(vs)
            if isinstance(key, tuple):
                d.setdefault(key[0], {})[key[1]] = lv
            else:
                d[key] = lv
        d.setdefault("sg0", "sg1")   # single-SG VF tag (legacy key in _assemble_ig)

    out = []
    n_skip = 0
    for i, r in enumerate(recs):
        n_sg = traj_meta[i][0]
        ig = _assemble_ig(by_traj[i], n_sg)
        if ig["p0"] is None or ig["pF"] is None:
            n_skip += 1
            continue
        # build forward_credits (F_i = p_i^alone − p0, prob domain) for multi-SG
        fwd_credits = None
        if n_sg > 1 and "Valone" in by_traj[i]:
            p0_v = ig["p0"]
            fwd_credits = []
            for sg, va in by_traj[i]["Valone"].items():
                if va is not None and ig["V0"] is not None:
                    fwd_credits.append((sg, math.exp(va) - p0_v))
                else:
                    fwd_credits.append((sg, None))
        # p_yhat from V_yhat
        p_yhat = math.exp(by_traj[i].get("Vyhat")) if by_traj[i].get("Vyhat") is not None else None
        adv = alloc_advantages(A=A_of[id(r)], p0=ig["p0"], pF=ig["pF"],
                               loo_credits=ig["credits"], n_sg=n_sg,
                               forward_credits=fwd_credits,
                               p_yhat=p_yhat,
                               r_k=r["f1"])
        conn = _connected(r)  # diagnostic only (not in alloc)
        msgs = build_messages(r)
        tadv = build_turn_advantages(r.get("trajectory", []) or msgs, adv)
        out.append({"case_id": r["case_id"], "case_idx": r["case_idx"],
                    "sample_idx": r["sample_idx"], "question": r["question"],
                    "gold": r.get("gold", []), "r_k": r["f1"], "A_k": A_of[id(r)],
                    "n_sg": n_sg, "connected": conn,
                    "V0": ig["V0"], "V_F": ig["V_F"], "Vseq": ig.get("Vseq"),
                    "Vloo": ig.get("Vloo"), "p0": ig["p0"], "pF": ig["pF"],
                    "credits": ig["credits"], "advantage": adv,
                    "p_yhat": p_yhat, "forward_credits": fwd_credits,
                    "messages": msgs, "turn_advantages": tadv,
                    # structured fields from the new-harness rollout — authoritative
                    # evidence/pred for the v0 advantage recomputation (display mode)
                    "evidence_entities": r.get("evidence_entities"),
                    "pred_entities": r.get("pred_entities")})
    print(f"  skipped {n_skip} records with None scoring (P1-3)")

    json.dump(out, open(out_path, "w"), ensure_ascii=False)
    dt = time.time() - t
    # 5. reports
    print(f"\n  IG+alloc: {len(out)} traj in {dt:.0f}s ({dt/max(len(recs),1):.2f}s/traj)")
    high_f1 = [r for r in out if r["r_k"] >= 0.8]
    for tau in (0.2, 0.3, 0.5):
        n_sc = sum(1 for r in high_f1 if (r["pF"] or 0) < tau)
        print(f"  shortcut P(pF<{tau} | F1≥0.8) = {n_sc}/{len(high_f1)} "
              f"= {n_sc/max(len(high_f1),1):.1%}")
    coarse = sum(1 for r in out if r["advantage"]["granularity"] == "coarse")
    print(f"  granularity: {coarse} coarse / {len(out)-coarse} fine")
    print(f"  wrote → {out_path}")


# ─────────────────────────────────────────────────────────────────────────
# 3. SIMULATION test — verify allocation across responsibility scenarios
# ─────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    # (name, A, p0, pF, [(sg, c_i)...], n_sg, connected, expected)
    # ── SINGLE-SG (coarse: connectivity + p0; A_info = A·(1−p0) on connected+success) ──
    ("S1. single-SG success-hard (needed, low p0)", +0.60, 0.10, 0.70, [("sg1", 0.60)], 1, True,
     "connected+success+low-p0 → A_info=A·(1−p0)=0.54 (credited), A_ans=0.60"),
    ("S2. single-SG success-easy (prior, high p0)", +0.40, 0.95, 0.97, [("sg1", 0.02)], 1, True,
     "connected+success+high-p0 → A_info=A·(1−p0)=0.02 (prior ~0), A_ans=0.40"),
    ("S3. single-SG retrieve-error (wrong relation, gold MISSED)", -0.70, 0.10, 0.15, [("sg1", -0.10)], 1, False,
     "disconnected+fail → A_info=A=−0.70 (block blamed), A_ans=0 (shielded)"),
    ("S4. single-SG sufficient-but-wrong (gold reached, answer blew it)", -0.50, 0.20, 0.92, [("sg1", 0.50)], 1, True,
     "connected+fail → A_info=0 (SHIELD retrieval), A_ans=A=−0.50 (answer bears)"),
    # ── MULTI-SG (fine: signed-LOO c_i) ──
    ("M1. multi-SG success-hard", +0.60, 0.10, 0.70,
     [("sg1", +0.45), ("sg2", +0.25), ("sg3", -0.05)], 3, True,
     "Plan=A·max(0,pF−p0)=0.36, Answer=0.60, sg1/sg2 rewarded (c>0), sg3 ~0"),
    ("M2. multi-SG failure-missing-retrieval", -0.70, 0.10, 0.18,
     [("sg1", +0.06), ("sg2", +0.04)], 2, True,
     "Plan=A·(1−pF)=−0.57 (bears), Answer=A·pF=−0.13, retrieve ~0 (no c<0)"),
    ("M3. multi-SG failure-wrong-retrieval", -0.60, 0.20, 0.35,
     [("sg1", +0.10), ("sg2", -0.40), ("sg3", +0.05)], 3, True,
     "sg2 penalized (c<0 under A<0), Plan=A·(1−pF), Answer=A·pF"),
    ("M4. multi-SG sufficient-but-wrong", -0.50, 0.20, 0.92,
     [("sg1", +0.40), ("sg2", +0.30)], 2, True,
     "Answer=A·pF=−0.46 (bears), Plan=A·(1−pF)=−0.04 (shielded), retrieve ~0"),
]


def simulate():
    print("=" * 80)
    print("  ADAPTIVE-GRANULARITY ALLOCATION (single-SG coarse / multi-SG fine)")
    print("=" * 80)
    for name, A, p0, pF, credits, n_sg, conn, expected in SCENARIOS:
        r = alloc_advantages(A, p0, pF, credits, n_sg, conn)
        g = r["granularity"]
        print(f"\n{name}  [{g}, conn={conn}]")
        print(f"  A={A:+.2f}  p0={p0:.2f}  pF={pF:.2f}")
        if g == "coarse":
            print(f"  → B_info (plan+retrieve) : {r['info']:+.4f}")
            print(f"  → Answer                 : {r['ans']:+.4f}")
            if A < 0:
                cons = r["info"] + r["ans"]
                print(f"  [check] A_info+A_ans = {cons:+.4f} (≈A={A:+.2f}? "
                      f"{'✓' if abs(cons - A) < 1e-6 else '✗'})")
        else:
            print(f"  → Plan    : {r['plan']:+.4f}")
            print(f"  → Answer  : {r['ans']:+.4f}")
            for sg, a in r["retrieve"].items():
                print(f"  → Retrieve[{sg}]: {a:+.4f}")
            if not r["retrieve"]:
                print(f"  → Retrieve: (all ~0)")
            if A < 0:
                cons = r["plan"] + r["ans"]
                print(f"  [check] A_plan+A_ans = {cons:+.4f} (≈A={A:+.2f}? "
                      f"{'✓' if abs(cons - A) < 1e-6 else '✗'})")
        print(f"  expect: {expected}")


def build_train_test(data_path: str, n: int = 8):
    """Validate the message-reconstruction + turn_advantage mapping on real
    trajectories (mock advantages — real IG needs GPU). Checks: apply_chat_template
    renders, <|im_start|> spans align 1:1 with messages, turn_advantages length
    == #assistant turns."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("models/Qwen3.5-9B", trust_remote_code=True)
    ims_id = tok.convert_tokens_to_ids("<|im_start|>")
    recs = json.loads(open(data_path).read())[:n]
    mock_coarse = {"granularity": "coarse", "info": 0.5, "ans": 0.8}
    mock_fine = {"granularity": "fine", "plan": 0.3, "ans": 0.8,
                 "retrieve": {"sg1": 0.2, "sg2": 0.1, "sg3": 0.05}}
    print(f"=== build_train_test on {len(recs)} trajectories ===")
    ok = 0
    for r in recs:
        n_sg = len(extract_subgraphs(r["trajectory"]))
        adv = mock_coarse if n_sg <= 1 else mock_fine
        msgs = build_messages(r)
        tadv = build_turn_advantages(r["trajectory"], adv)
        n_asst = sum(1 for m in msgs if m["role"] == "assistant")
        # render + span check
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok.encode(txt, add_special_tokens=False)
        nmark = sum(1 for t in ids if t == ims_id)
        aligned = (nmark == len(msgs))
        tadv_ok = (len(tadv) == n_asst)
        status = "✓" if (aligned and tadv_ok) else "✗"
        if aligned and tadv_ok:
            ok += 1
        print(f"  {status} id={r['case_id'][:18]:18s} n_sg={n_sg} msgs={len(msgs)} "
              f"asst={n_asst} markers={nmark} align={aligned} tadv={len(tadv)} "
              f"(ok={tadv_ok}) toks={len(ids)}")
        if not (aligned and tadv_ok):
            print(f"      turn tools: {[_tool_of_assistant(m.get('content','')) for m in r['trajectory'] if m.get('role')=='assistant']}")
    print(f"\n  {ok}/{len(recs)} trajectories pass (render + span align + turn_adv match)")
    # show one full example's turn_adv mapping
    r = recs[0]
    n_sg = len(extract_subgraphs(r["trajectory"]))
    adv = mock_coarse if n_sg <= 1 else mock_fine
    tadv = build_turn_advantages(r["trajectory"], adv)
    tools = [_tool_of_assistant(m.get("content", "")) for m in r["trajectory"] if m.get("role") == "assistant"]
    print(f"  example turn→adv: {list(zip(tools, [round(a,3) for a in tadv]))}")


def reallocate(in_path: str, out_path: str):
    """Recompute the VALIDATED allocation on existing IG output (no GPU, no model).
    Filters MID outliers; single-SG uses connectivity+p0, multi-SG uses c_i.
    Recomputes turn_advantages. Writes trainer-ready JSONL (one record/line):
      {messages, turn_advantages, advantage, A_k, r_k, n_sg, p0, pF, credits, case_id, ...}
    """
    recs = json.loads(open(in_path).read())
    n_mid = sum(1 for r in recs if _has_mid(r))
    recs = [r for r in recs if not _has_mid(r)]
    print(f"reallocate: {len(recs)} records (filtered {n_mid} MID)", flush=True)
    n_conn = 0
    n_skip = 0
    with open(out_path, "w") as f:
        for i, r in enumerate(recs):
            if r.get("p0") is None or r.get("pF") is None:  # P1-3: skip None scoring
                n_skip += 1
                continue
            conn = _connected(r)
            n_conn += int(conn)
            adv = alloc_advantages(A=r["A_k"], p0=r["p0"], pF=r["pF"],
                                   loo_credits=r.get("credits", []), n_sg=r["n_sg"],
                                   forward_credits=r.get("forward_credits"),
                                   p_yhat=r.get("p_yhat"),
                                   r_k=r.get("r_k", r.get("f1", 0)))
            tadv = build_turn_advantages(r.get("messages", []), adv)
            rec = {
                "case_id": r.get("case_id"), "case_idx": r.get("case_idx"),
                "sample_idx": r.get("sample_idx"), "question": r.get("question"),
                "gold": r.get("gold", []), "r_k": r.get("r_k", r.get("f1", 0)),
                "A_k": r["A_k"], "n_sg": r["n_sg"], "p0": r.get("p0"), "pF": r.get("pF"),
                "credits": r.get("credits", []), "connected": conn,
                "p_yhat": r.get("p_yhat"),
                "forward_credits": r.get("forward_credits"),
                "advantage": adv, "messages": r.get("messages", []),
                "turn_advantages": tadv,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(recs)}", flush=True)
    n_single = sum(1 for r in recs if r["n_sg"] <= 1 and r.get("p0") is not None and r.get("pF") is not None)
    print(f"  wrote → {out_path}  (skipped {n_skip} None-scoring records)")
    print(f"  single-SG connected: {n_conn}/{n_single}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    if mode == "simulate":
        simulate()
    elif mode == "data":
        run_real(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif mode == "reallocate":
        reallocate(sys.argv[2], sys.argv[3])
    elif mode == "build_train_test":
        build_train_test(sys.argv[2] if len(sys.argv) > 2 else "/tmp/rollout_train_100x8.json")
    else:
        print("usage: ig_to_advantage.py [simulate | data IN OUT [limit] | "
              "reallocate IG.json OUT.jsonl | build_train_test IN.json]")
