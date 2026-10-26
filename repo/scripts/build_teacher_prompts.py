#!/usr/bin/env python3
"""Build OPSD teacher_prompt candidates from verified-correct trajectories.

Input: a JSONL manifest with one row per case:
  {"case_id", "question", "gt", "answer", "turns", "path"}
(produced by the trajectory dump analysis, e.g.
 reports/correct_manifest_union.jsonl).

Output: JSONL rows with ``teacher_prompt`` in the OPSD format: the verified
action chain + final answer, then "Now solve the problem using your own
approach:". Thinking is deliberately excluded — the teacher reference is the
action/evidence recipe, while the student rollout keeps its own thinking.

Usage:
  python scripts/build_teacher_prompts.py reports/correct_manifest_union.jsonl \
      -o reports/teacher_prompts_union.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kgqa.rl.verl_plugins.opsd_pair import (split_thinking,  # noqa: E402
                                             strip_turn_terminator)


_BAD_FEEDBACK = ("error:", "REJECTED:", "NO_EVIDENCE", "EMPTY_ANSWER",
                 "HALLUCINATED_RELATION", "REPEATED_FAILED_COMMAND",
                 "UNDELIVERED_ANSWER")


def action_blocks_msgs(msgs: list[dict]) -> list[str]:
    """Extract the successful action recipe from a messages-like list.

    Failed/retried steps (error / rejected / no-evidence feedback) are dropped:
    a teacher reference should be the path that actually worked, not the
    student's detours. The final answer step is always kept. Accepts BOTH
    sources: rollout-result ``trajectory`` records and scheduler dump
    ``messages`` (assistant content carries the raw text in both).
    """
    blocks = []
    for i, msg in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        raw = msg.get("raw_full") or msg.get("raw") or msg.get("content") or ""
        if not isinstance(raw, str):
            continue
        text, _ = strip_turn_terminator(raw, [])
        _, action = split_thinking(text)
        action = (action or text).strip()
        pos = action.rfind("tool:")
        if pos < 0:
            continue
        block = action[pos:].strip()
        is_answer = block.startswith("tool: answer")
        feedback = ""
        for nxt in msgs[i + 1:]:
            c = nxt.get("content") or ""
            if not isinstance(c, str):
                continue
            if nxt.get("role") == "tool" or c.startswith("Tool result ("):
                feedback = c
                break
        if not is_answer and any(sig in feedback[:600] for sig in _BAD_FEEDBACK):
            continue
        blocks.append(block)
    return blocks


def action_blocks(traj_path: str) -> list[str]:
    """Dump-JSON wrapper around action_blocks_msgs."""
    d = json.load(open(traj_path))
    return action_blocks_msgs(d.get("messages", []))


def decision_digest(msgs: list[dict], answer=None) -> str:
    """Verified trajectory → DECISION digest (user design 2026-10-25).

    The transcript's supervision value is the CHOICES, not the action
    syntax: which relations to pick from what retrieve_relations returned,
    which subgraph to pull, the final reasoning rule (declared by the
    plan's last fact, e.g. "which latest year"), and answering directly
    once evidence covers the facts. The model already knows the behavioral
    policy — failures concentrate at these decision points. Render ONLY
    them, as guidance prose: never emit tool-block syntax or multi-block
    layout (a reference written IN action syntax became an output-format
    template and collapsed the policy into essay-mode reproduction).
    """
    import re
    blocks = action_blocks_msgs(msgs)
    anchor = None
    atype = None
    plan_facts = []          # (head, sub-question, var)
    picks = []               # (center, relations)
    final_answer = answer
    for b in blocks:
        tool, kv = None, {}
        for ln in b.splitlines():
            if ln.startswith("tool:"):
                tool = ln.split(":", 1)[1].strip()
                continue
            if ":" in ln:
                k, v = ln.split(":", 1)
                kv[k.strip()] = v.strip()
        if tool == "plan":
            anchor = kv.get("question_entities") or anchor
            atype = kv.get("answer_type")
            for k, v in kv.items():
                if re.match(r"sg\d+\.f\d+$", k) and v and "|" in v:
                    parts = [p.strip() for p in v.split("|")]
                    if len(parts) >= 3:
                        plan_facts.append((parts[0], parts[1], parts[2]))
        elif tool == "retrieve_subgraph":
            picks.append((kv.get("center"), kv.get("relations")))
        elif tool == "answer":
            final_answer = kv.get("answer_values") or final_answer
        # retrieve_relations calls are protocol steps; the CHOICE they feed
        # is the subgraph relation pick — their syntax carries no extra
        # supervision and is deliberately omitted.
    lines = []
    if plan_facts:
        chain = " -> ".join(f"{' '.join(subq.split())} (binds {var})"
                            for _, subq, var in plan_facts)
        lines.append(f"Decomposition: anchor {anchor}; chain {chain}."
                     + (f" Answer type: {atype}." if atype else ""))
    _mid = re.compile(r"^[mg]\.\w+$")      # Freebase MID — unreadable as a name
    for i, (c, r) in enumerate(picks):
        fact = next((f"{' '.join(subq.split())} -> {var}"
                     for h, subq, var in plan_facts
                     if (h and c and h.lower() == c.lower())
                     or (var and var == c)), None)
        # advance target: name-matched first, else positional (the i-th
        # subgraph call advances the i-th declared fact; plan order is the
        # chain order). Without this a MID center showed up as a bare
        # "for m.048bcbz, pick ..." — meaningless to the model.
        head_var = plan_facts[i][2] if i < len(plan_facts) else None
        if c and _mid.match(str(c)):
            subj = (f"the entity bound to {plan_facts[i][0]}"
                    if i < len(plan_facts) else "the entity you just resolved")
        else:
            subj = str(c)
        if fact is None and i < len(plan_facts):
            h, subq, var = plan_facts[i]
            fact = f"{' '.join(subq.split())} -> {var}"
        lines.append(f"Key choice: for {subj}, pick the relations {r} from what "
                     f"retrieve_relations returned"
                     + (f" (this advances: {fact})" if fact else "") + ".")
    if plan_facts:
        last_q = " ".join(plan_facts[-1][1].split())
        # the final-reasoning rule must match THIS question's operation.
        # A hardcoded "(latest = maximum)" example told a religion question
        # to take a maximum — actively misleading supervision.
        if re.search(r"\b(latest|most recent|last|newest|youngest)\b", last_q, re.I):
            rule = " (e.g. latest = the maximum)"
        elif re.search(r"\b(earliest|oldest|first)\b", last_q, re.I):
            rule = " (e.g. earliest = the minimum)"
        elif re.search(r"\b(highest|largest|greatest|most)\b", last_q, re.I):
            rule = " (e.g. highest = take the maximum)"
        elif re.search(r"\b(lowest|smallest|fewest|least)\b", last_q, re.I):
            rule = " (e.g. lowest = take the minimum)"
        else:
            rule = ""       # no numeric/ordering operation — say nothing
        lines.append(f"Final reasoning: resolve '{last_q}' "
                     f"over the collected values{rule}, then answer directly "
                     f"— no further retrieval.")
    if final_answer:
        vals = " | ".join(final_answer) if isinstance(final_answer, list) \
            else str(final_answer)
        lines.append(f"Answer once every fact is covered: {vals}.")
    return "\n".join(lines)


def _norm(x) -> str:
    return " ".join(str(x or "").lower().split())


def gold_in_evidence(gold, evidence_text) -> list:
    """Gold entities that the evidence actually contains (user ruling).

    Answer line = gold ∩ evidence: with FULL coverage that is the complete
    gold answer (and the remaining gap is purely an answer-strategy problem);
    with partial coverage it is the sub-answer the chain can support — never
    the full gold (unsupported entities would teach hallucination) and never
    the model's arbitrary pick."""
    if not gold:
        return []
    ev = _norm(evidence_text)
    if not ev:
        return []
    import re as _re
    vals = gold if isinstance(gold, (list, tuple)) else [gold]
    out = []
    for v in vals:
        n = _norm(v)
        if not n:
            continue
        if _re.search(r"(?<![a-z0-9])" + _re.escape(n) + r"(?![a-z0-9])", ev):
            out.append(v)
    return out


def gold_supported(gold, evidence_text) -> bool:
    """True when EVERY gold answer entity appears in the collected evidence.

    User ruling 2026-10-25: the reference's answer line may use GOLD only
    when the evidence actually contains it (then it is a verified correct
    answer); with incomplete evidence we keep the trajectory's own answer
    (what the chain can actually support) instead of teaching an
    unsupported gold."""
    if not gold:
        return False
    ev = _norm(evidence_text)
    if not ev:
        return False
    vals = gold if isinstance(gold, (list, tuple)) else [gold]
    vals = [v for v in vals if str(v or "").strip()]
    if not vals:
        return False
    import re as _re
    def _has(v):
        n = _norm(v)
        # word-boundary match: a substring hit inside a relation name
        # ("...french..." in "sports.french_open") must not count as the
        # entity being present in the evidence
        return _re.search(r"(?<![a-z0-9])" + _re.escape(n) + r"(?![a-z0-9])",
                          ev) is not None
    return all(_has(v) for v in vals)


def behavior_script(msgs: list[dict], answer=None,
                    gold=None, evidence_text=None) -> str:
    """Verified trajectory → TURN-BY-TURN behavior script (user design
    2026-10-25: 行为序列, 逐回合"下一步该做什么").

    Unlike the abstract decision digest (which collapses into the plan's
    decomposition), each entry here is ONE turn's single action, keyed to
    the OBSERVATION that preceded it: "after receiving the relation
    results, submit retrieve_subgraph with the entity as anchor, picking
    relations X from the previous turn". That is the part the plan does
    NOT carry: how to act at each observation boundary. Rejected/retried
    turns are dropped (the script records the path that worked).

    Phrased as directive prose ("应提交…") — never as tool-block syntax,
    so it cannot serve as a copy-paste output template (the essay-mode
    collapse cause); the wrapper still anchors one action per turn.
    """
    import re
    entries = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        raw = m.get("raw_full") or m.get("raw") or m.get("content") or ""
        if not isinstance(raw, str):
            continue
        text, _ = strip_turn_terminator(raw, [])
        _, action = split_thinking(text)
        action = (action or text).strip()
        pos = action.rfind("tool:")
        if pos < 0:
            continue
        tool, kv = None, {}
        if pos >= 0:
            # TEXT form (scheduler dumps): "tool: plan\nquestion_entities: ..."
            block = action[pos:].strip()
            for ln in block.splitlines():
                if ln.startswith("tool:"):
                    tool = ln.split(":", 1)[1].strip()
                    continue
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    kv[k.strip()] = v.strip()
        else:
            # JSON form (offline pool trajectories): {"tool": "plan", "args": {...}}
            try:
                obj = json.loads(action)
            except Exception:  # noqa: BLE001
                obj = None
            if not isinstance(obj, dict) or not obj.get("tool"):
                continue
            tool = str(obj["tool"]).strip()
            args = obj.get("args") or {}
            for k, v in args.items():
                if isinstance(v, (list, tuple)):
                    kv[k] = " | ".join(str(x) for x in v)
                elif isinstance(v, dict):
                    continue
                else:
                    kv[k] = str(v)
            if tool == "plan":
                kv["question_entities"] = kv.get("entities") or kv.get("anchor")
                for si, sg in enumerate(args.get("subgraphs") or [], 1):
                    for fi, f in enumerate((sg or {}).get("facts") or [], 1):
                        if isinstance(f, (list, tuple)) and len(f) >= 3:
                            kv[f"sg{si}.f{fi}"] = f"{f[0]} | {f[1]} | {f[2]}"
            if tool == "answer" and "answer" in kv and "answer_values" not in kv:
                kv["answer_values"] = kv["answer"]
        if not tool:
            continue
        # the observation BEFORE this action (= what the model had seen)
        prev = None
        for j in range(i - 1, -1, -1):
            if msgs[j].get("role") in ("user", "tool"):
                prev = str(msgs[j].get("content") or "")
                break
        # did THIS action get rejected? then it is not part of the path
        nxt = None
        for k2 in msgs[i + 1:]:
            c = k2.get("content") or ""
            if not isinstance(c, str):
                continue
            if k2.get("role") in ("user", "tool"):
                nxt = c
                break
        if nxt and (nxt.lstrip().startswith("error:")
                    or nxt.lstrip().startswith("REJECTED:")):
            continue
        # NOTE: a retry after a rejection is NOT dropped here — dropping it
        # removed the ACCEPTED plan (the common "rejected plan → corrected
        # plan" path), which is exactly what the script must show. Retries
        # are collapsed by the same-target dedupe below (keep the last).
        if prev and prev.lstrip().startswith("Tool result (plan"):
            after = "收到 plan 被接受后"
        elif prev and prev.lstrip().startswith("Tool result (retrieve_relations"):
            after = "收到关系检索结果后"
        elif prev and prev.lstrip().startswith("Tool result (retrieve_subgraph"):
            after = "收到子图结果后"
        elif prev and prev.lstrip().startswith("Tool result (answer"):
            after = "作答被受理后"
        else:
            after = "开场"
        # a REAL tool result (not a rejection, not a duplicate-query echo)
        ok = bool(nxt) and not nxt.lstrip().startswith(("error:", "REJECTED:")) \
            and "unchanged_query" not in nxt[:300]
        entries.append((after, tool, kv, ok))
    if not entries:
        return ""
    # collapse consecutive same-target retries (keep the LAST — the one that
    # actually advanced the chain; earlier ones returned nothing usable).
    # Same (tool, center): plan/answer have no center → last of that tool;
    # retrieve_* with the SAME center → last (a repeat of one target is a
    # retry, different relations on one center are distinct facts and kept).
    dedup = []
    for e in entries:
        same = bool(dedup) and dedup[-1][1] == e[1] and (
            dedup[-1][2].get("center") == e[2].get("center"))
        if same:
            dedup[-1] = e          # retry on the same target → keep the last
        else:
            dedup.append(e)
    entries = dedup
    # relabel from the PREVIOUS KEPT entry (a collapsed retry's observation no
    # longer exists — an entry claiming "after the subgraph result" when the
    # kept predecessor was a relation retrieval reads as a wrong instruction)
    _lab = {"plan": "收到 plan 被接受后",
            "retrieve_relations": "收到关系检索结果后",
            "retrieve_subgraph": "收到子图结果后",
            "answer": "作答被受理后"}
    entries = [("开场" if i == 0 else _lab.get(entries[i - 1][1], e[0]),
                e[1], e[2], e[3]) for i, e in enumerate(entries)]
    # Reference path = actions BEFORE the first answer + the FINAL answer.
    # A verified trajectory can answer prematurely (the tool accepts any
    # delivery; correctness is scored at the end) and then keep retrieving.
    # Earlier attempts and the detour between them are not the recipe; the
    # student should see one clean trajectory that ends in the final answer.
    entries = [e for e in entries if e[3] or e[1] == "answer"]
    ans_idx = [i for i, e in enumerate(entries) if e[1] == "answer"]
    if ans_idx:
        pre = [e for i, e in enumerate(entries) if i < ans_idx[0]]
        fin = entries[ans_idx[-1]]
        after = fin[0] if fin[0] not in ("开场", "作答被受理后") \
            else "证据已覆盖全部事实后"
        entries = pre + [(after, fin[1], fin[2], True)]

    lines, plan_facts, plan_anchor, atype = [], [], None, None
    fact_i = 0          # plan fact the NEXT action advances
    prev_fb = None      # what the environment returns after the previous action
    prev_tool = None
    for idx, (after, tool, kv, _ok) in enumerate(entries, 1):
        head = (f"Turn {idx} (no environment result yet — only the system rules): "
                if idx == 1 else
                f"Turn {idx} (received the previous turn's {prev_fb}): ")
        if tool == "plan":
            plan_anchor = kv.get("question_entities") or plan_anchor
            atype = kv.get("answer_type")
            for k, v in kv.items():
                if re.match(r"sg\d+\.f\d+$", k) and v and "|" in v:
                    parts = [p.strip() for p in v.split("|")]
                    if len(parts) >= 3:
                        plan_facts.append(parts)
            chain = "; ".join(f"{' '.join(p[1].split())} -> {p[2]}"
                              for p in plan_facts)
            kw = [f"anchor {plan_anchor}"] if plan_anchor else []
            if atype:
                kw.append(f"answer type {atype}")
            if chain:
                kw.append(f"fact chain {chain}")
            lines.append(head + "make the PLANNING move — use the plan tool. "
                         "Key points: " + "; ".join(kw) + ".")
            fb = "plan result"
        elif tool == "retrieve_relations":
            c = kv.get("center") or plan_anchor or "the entity from the previous fact"
            if re.match(r"^[mg]\.\w+$", str(c)):
                c = f"the entity returned last turn ({c})"
            q = kv.get("question")
            purpose = ""
            if fact_i < len(plan_facts):
                h, subq, var = plan_facts[fact_i]
                purpose = f" This step serves: {' '.join(subq.split())} -> {var}."
            path = (" (next fact, another path)" if prev_tool == "retrieve_subgraph"
                    else "")
            basis = {"plan": "the plan result",
                     "retrieve_relations": "the candidate relations returned last turn",
                     "retrieve_subgraph": "the subgraph returned last turn",
                     None: "the system rules"}.get(prev_tool, "the previous result")
            lines.append(head + f"based on {basis} and the protocol rules, this turn "
                         f"performs RELATION RETRIEVAL{path} — use retrieve_relations, "
                         f"center entity {c}"
                         + (f", retrieval target (condensed from this fact's question) "
                            f"\"{' '.join(q.split())}\"" if q else "")
                         + "." + purpose)
            fb = "candidate relation list"
        elif tool == "retrieve_subgraph":
            basis = {"plan": "the plan result",
                     "retrieve_relations": "the candidate relations returned last turn",
                     "retrieve_subgraph": "the subgraph returned last turn",
                     None: "the system rules"}.get(prev_tool, "the previous result")
            c = kv.get("center")
            if not c:
                c = "the entity from the previous relation lookup"
            elif re.match(r"^[mg]\.\w+$", str(c)):
                c = "the entity returned last turn"
            r = kv.get("relations")
            purpose = ""
            if fact_i < len(plan_facts):
                h, subq, var = plan_facts[fact_i]
                purpose = f" This step serves: {' '.join(subq.split())} -> {var}."
                fact_i += 1
            how = ("choose from the candidates returned last turn"
                   if prev_tool == "retrieve_relations" else "use")
            lines.append(head + f"based on {basis} and the protocol rules, this turn "
                         f"performs SUBGRAPH RETRIEVAL — use retrieve_subgraph, "
                         f"anchor: {c}; relations: {how} {r}." + purpose)
            fb = "subgraph result"
        elif tool == "answer":
            vals = kv.get("answer_values") or answer
            if gold and evidence_text and gold_supported(gold, evidence_text):
                vals = gold
            if isinstance(vals, list):
                vals = " | ".join(vals)
            last_q = (" ".join(plan_facts[-1][1].split())
                      if plan_facts else None)
            rule = ""
            if last_q:
                if re.search(r"\b(latest|most recent|last|newest|youngest)\b", last_q, re.I):
                    rule = " Final reasoning: take the maximum over the collected values."
                elif re.search(r"\b(earliest|oldest|first)\b", last_q, re.I):
                    rule = " Final reasoning: take the minimum."
                elif re.search(r"\b(highest|largest|greatest|most)\b", last_q, re.I):
                    rule = " Final reasoning: take the maximum."
                elif re.search(r"\b(lowest|smallest|fewest|least)\b", last_q, re.I):
                    rule = " Final reasoning: take the minimum."
            lines.append(head + "the evidence covers every fact — make the ANSWER "
                         "move: use the answer tool and choose"
                         + (f" {vals}" if vals else " the answer")
                         + " from the evidence collected so far; stop retrieving." + rule)
            fb = None
        if fb:
            lines.append(f"  -> After acting, wait for the environment's {fb}; "
                         f"base the next turn's single move on it.")
        prev_fb, prev_tool = fb, tool
    return "\n".join(lines)


def _action_entries(msgs, answer=None):
    """Reuse behavior_script's parsing to get cleaned (tool, kv) entries."""
    import re
    entries = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        raw = m.get("raw_full") or m.get("raw") or m.get("content") or ""
        if not isinstance(raw, str):
            continue
        text, _ = strip_turn_terminator(raw, [])
        _, action = split_thinking(text)
        action = (action or text).strip()
        pos = action.rfind("tool:")
        tool, kv = None, {}
        if pos >= 0:
            block = action[pos:].strip()
            for ln in block.splitlines():
                if ln.startswith("tool:"):
                    tool = ln.split(":", 1)[1].strip()
                elif ":" in ln:
                    k, v = ln.split(":", 1)
                    kv[k.strip()] = v.strip()
        else:
            try:
                obj = json.loads(action)
            except Exception:  # noqa: BLE001
                obj = None
            if not isinstance(obj, dict) or not obj.get("tool"):
                continue
            tool = str(obj["tool"]).strip()
            args = obj.get("args") or {}
            for k, v in args.items():
                if isinstance(v, (list, tuple)):
                    kv[k] = " | ".join(str(x) for x in v)
                elif isinstance(v, dict):
                    continue
                else:
                    kv[k] = str(v)
            if tool == "plan":
                kv["question_entities"] = kv.get("entities") or kv.get("anchor")
                for si, sg in enumerate(args.get("subgraphs") or [], 1):
                    for fi, f in enumerate((sg or {}).get("facts") or [], 1):
                        if isinstance(f, (list, tuple)) and len(f) >= 3:
                            kv[f"sg{si}.f{fi}"] = f"{f[0]} | {f[1]} | {f[2]}"
            if tool == "answer" and "answer" in kv and "answer_values" not in kv:
                kv["answer_values"] = kv["answer"]
        if not tool:
            continue
        nxt = next((str(k.get("content") or "") for k in msgs[i + 1:]
                    if k.get("role") in ("user", "tool")), "")
        if nxt.lstrip().startswith(("error:", "REJECTED:")) or "unchanged_query" in nxt[:300]:
            continue
        entries.append((tool, kv))
    # collapse same-target retries
    out = []
    for e in entries:
        if out and out[-1][0] == e[0] and out[-1][1].get("center") == e[1].get("center"):
            out[-1] = e
        else:
            out.append(e)
    # keep pre-first-answer + final answer
    ai = [i for i, e in enumerate(out) if e[0] == "answer"]
    if ai:
        pre = out[:ai[0]]
        fin = out[ai[-1]]
        out = pre + [fin]
    return out


def render_flat_action(msgs, answer=None, gold=None, evidence_text=None):
    """flat_action renderer: decision records in execution order, NO turn
    numbers, NO narrative, NO instructions (GSD ref v2 — tests whether
    removing the Turn-script alone stops the rehearsal effect)."""
    entries = _action_entries(msgs, answer)
    if not entries:
        return ""
    vals = None
    for t, kv in entries:
        if t == "answer":
            vals = kv.get("answer_values") or answer
    if gold and evidence_text and gold_supported(gold, evidence_text):
        vals = gold
    if isinstance(vals, (list, tuple)):
        vals = " | ".join(str(x) for x in vals)
    lines = ["[REFINED DECISION DATA]"]
    if vals:
        lines.append(f"Answer target: {vals}")
    for t, kv in entries:
        if t == "plan":
            at = kv.get("answer_type")
            lines.append(f"Answer type: {at}" if at else "Answer type: -")
        elif t == "retrieve_relations":
            c = kv.get("center") or "-"
            q = kv.get("question") or ""
            lines.append(f"Relation search:\n  Center: {c}"
                         + (f"\n  Query: {q}" if q else ""))
        elif t == "retrieve_subgraph":
            c = kv.get("center") or "-"
            r = kv.get("relations") or "-"
            import re as _re
            if _re.match(r"^[mg]\.\w+$", str(c)):
                c = "entity returned by the previous search"
            lines.append(f"Subgraph selection:\n  Center: {c}\n  Relations: {r}")
    lines.append("[END REFINED DECISION DATA]")
    return "\n".join(lines)


def render_fact_grouped(msgs, answer=None, gold=None, evidence_text=None):
    """fact_grouped renderer: actions grouped by the plan fact they serve,
    with binding dependencies (GSD ref v2 — tests whether fact structure
    adds value over flat records)."""
    import re
    entries = _action_entries(msgs, answer)
    if not entries:
        return ""
    plan = next((kv for t, kv in entries if t == "plan"), {})
    facts = []
    for k, v in plan.items():
        if re.match(r"sg\d+\.f\d+$", k) and v and "|" in v:
            parts = [p.strip() for p in v.split("|")]
            if len(parts) >= 3:
                facts.append(parts)
    vals = None
    for t, kv in entries:
        if t == "answer":
            vals = kv.get("answer_values") or answer
    if gold and evidence_text and gold_supported(gold, evidence_text):
        vals = gold
    if isinstance(vals, (list, tuple)):
        vals = " | ".join(str(x) for x in vals)
    lines = ["[REFINED EXPERIENCE DATA]", ""]
    # group subgraph actions to facts positionally (same heuristic as the
    # turn renderer: the i-th subgraph advances the i-th fact)
    subs = [(t, kv) for t, kv in entries if t == "retrieve_subgraph"]
    rels = [(t, kv) for t, kv in entries if t == "retrieve_relations"]
    if vals:
        lines.append(f"Answer target: {vals}")
    if plan.get("answer_type"):
        lines.append(f"Answer type: {plan['answer_type']}")
    if facts:
        lines.append("Question requirements:")
        for h, q, var in facts:
            lines.append(f"  {h} | {q} | {var}")
    lines.append("")
    lines.append("Relevant decision information:")
    for i, (h, q, var) in enumerate(facts):
        lines.append("")
        lines.append(f"FACT {i + 1}: {q} -> {var}")
        r = rels[i] if i < len(rels) else None
        sb = subs[i] if i < len(subs) else None
        if r:
            lines.append(f"  Anchor: {r[1].get('center') or '-'}")
            if r[1].get("question"):
                lines.append(f"  Relation query: {r[1]['question']}")
        if sb:
            lines.append("  Selected relations:")
            for rr in str(sb[1].get("relations") or "-").split(" | "):
                lines.append(f"    {rr}")
        if i > 0:
            lines.append(f"  Depends on the binding of FACT {i}")
    lines.append("")
    lines.append("Reference constraints: relation choices depend on the "
                 "candidates actually returned in the current environment; "
                 "the answer target is a goal, not evidence already "
                 "retrieved.")
    lines.append("[END REFINED EXPERIENCE DATA]")
    return "\n".join(lines)


REF_HEADER = (
    "The following block contains refined experience data for the current "
    "question. Use it to guide decisions under the current environment "
    "state. It is not a transcript to reproduce: do not rehearse or "
    "restate its records. The actual interaction history determines what "
    "has already been retrieved and which actions are valid. Follow the "
    "original environment protocol.\n\n")


def wrap_structured(ref_text: str) -> str:
    return REF_HEADER + ref_text


def wrap_behavior_script(script: str, n_steps: int) -> str:
    """Turn-structured wrapper: the round mechanic is stated BEFORE the body
    so the script can never be read as "emit everything at once" (the
    essay-mode failure), and the reference is scoped as a behavioral
    skeleton (what/who/which/why-fact), not an argued motivation."""
    return (
        "Here is the verified reference behavior for this problem. Rules: each "
        "turn makes EXACTLY ONE move; after a move you must wait for the "
        "environment's result, and only then decide the next turn. Never merge "
        "several turns into one reply — the system rules allow one tool action "
        "per turn. (The reference gives only the behavioral skeleton: which "
        "tool, which anchor/center, which relations to pick from the returned "
        "candidates, which fact a step serves, and when to answer; do your own "
        "reasoning.)\n\n"
        f"{str(script).strip()}\n\n"
        "Now solve the problem turn by turn: one move per turn, wait for the "
        "environment before continuing.")


def wrap_reference_digest(digest: str) -> str:
    """Digest-mode transition: guidance framing + explicit single-action
    anchor (the digest must never read as a multi-action output template)."""
    return (
        "Here is the reference experience of this case — the KEY DECISIONS "
        "at the points where the solution succeeds or fails (decomposition, "
        "which relations to pick from each retrieval, the final reasoning "
        "rule, and when to answer). This is guidance for your choices, NOT "
        "an output format: keep submitting exactly ONE action per turn as "
        "the protocol requires.\n\n"
        f"{str(digest).strip()}\n\n"
        "Now solve this problem on your own, turn by turn:")


def wrap_reference_solution(solution: str) -> str:
    """The single teacher-prefix transition used by every case.

    One uniform wrapper for trajectory-derived solutions and gold answers:
    the teacher always sees the same sentence, only ``[Solution]`` differs.
    """
    return (
        f"Here is a reference solution: {str(solution).strip()}. "
        "After understanding the reference solution, please try to solve "
        "this problem using your own approach below:")


def normalize_teacher_prompt(prompt: str) -> str:
    """Wrap a gold-style teacher prompt in the standard transition sentence.

    Gold rows from earlier manifests look like::

        Question: ...
        Reference solution:
        The answer is: ...

        Now solve the problem using your own approach:

    Only the answer-bearing part is the solution; the question is already in
    the student/teacher conversation. Already-wrapped prompts pass through.
    """
    text = str(prompt or "").strip()
    if not text or text.startswith("Here is a reference solution:"):
        return text
    parts = re.split(r"\n\s*Reference solution:\s*", text, maxsplit=1)
    solution = parts[1] if len(parts) > 1 else text
    solution = re.split(r"\n\s*Now solve the problem", solution, maxsplit=1)[0]
    return wrap_reference_solution(solution)


def build(row: dict) -> dict:
    blocks = action_blocks(row["path"])
    chain = "\n\n".join(f"[step {i + 1}]\n{b}" for i, b in enumerate(blocks))
    answer = " | ".join(row.get("answer") or [])
    solution = f"{chain}\n\nFinal answer: {answer}"
    prompt = wrap_reference_solution(solution)
    return {
        "case_id": row["case_id"],
        "question": row["question"],
        "gt": row.get("gt"),
        "answer": row.get("answer"),
        "turns": row.get("turns"),
        "source_traj": row["path"],
        "solution": solution,
        "teacher_prompt": prompt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path, nargs="?")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--normalize-data", type=Path,
                    help="rewrite teacher_prompt in a training jsonl "
                         "(same schema as the rollout dataset)")
    ap.add_argument("--redigest-data", type=Path,
                    help="rewrite teacher_prompt in a training jsonl as a "
                         "DECISION DIGEST re-rendered from each row's "
                         "source_traj (2026-10-25 design; requires rows to "
                         "carry source_traj + answer)")
    args = ap.parse_args()
    if args.normalize_data:
        rows = [json.loads(line) for line in
                args.normalize_data.read_text().splitlines() if line.strip()]
        changed = 0
        for row in rows:
            old = row.get("teacher_prompt") or ""
            new = normalize_teacher_prompt(old)
            if new != old:
                row["teacher_prompt"] = new
                changed += 1
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"normalized {changed}/{len(rows)} teacher prompts -> {args.out}")
        return
    if args.redigest_data:
        rows = [json.loads(line) for line in
                args.redigest_data.read_text().splitlines() if line.strip()]
        out, miss = [], 0
        for row in rows:
            src = row.get("source_traj")
            try:
                d = json.load(open(src))
                digest = decision_digest(d.get("messages", []),
                                         answer=row.get("answer"))
                if "Decomposition" not in digest and "Key choice" not in digest:
                    raise ValueError("no parseable decisions")
                row["teacher_prompt"] = wrap_reference_digest(digest)
            except Exception:  # noqa: BLE001
                miss += 1              # keep the old prompt for unusable rows
            out.append(row)
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
        print(f"redigested {len(out) - miss}/{len(out)} rows "
              f"({miss} kept old prompt) -> {args.out}")
        return
    if args.manifest is None:
        ap.error("manifest is required unless --normalize-data is used")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    out = []
    for row in rows:
        try:
            out.append(build(row))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {row.get('case_id')}: {exc}", file=sys.stderr)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"{len(out)}/{len(rows)} teacher prompts -> {args.out}")


if __name__ == "__main__":
    main()
