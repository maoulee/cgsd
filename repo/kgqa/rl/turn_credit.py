#!/usr/bin/env python3
"""RSCC turn-level coefficients (Codex verl plan §5) — index-direct mapping.

Replaces the S1 turn mapping in seq_advantage.build_turn_advantages, which
(a) re-derives tool names from TEXT REGEX (misses canonical JSON actions —
verified on r9a: `_tool_of_assistant` returns None for JSON turns), and
(b) feeds a scalar-summing trainer path (`_tokenize_seq`: scalar_adv =
sum(turn_advantages)) that collapses e.g. [0.4, 0.4, -0.8] to 0.0.

Index-direct: the credit judgement blocks (tmp/prob_pathway_r9a_v5.json shape)
carry b["idx"] = trajectory index of the SG tool RESULT message; the ACTION
turn is idx-1 (the assistant message right before it). No text guessing.

Coefficient rules (plan §5, locked):
  retrieve_subgraph 有效* (effective exploration)      → +λ
  retrieve_subgraph 可移除* (removable / env repeat)   → 0
  判定未知 / 不评估                                     → 0, flagged UNKNOWN (never
                                                          faked as a definite label)
  plan / retrieve_relations / rejected answer attempts → 0
  FINAL ACCEPTED answer turn                           → (F1_i - group_mean) /
                                                          (group_std + eps)
If the whole group shares one F1, the answer coefficient is 0 but retrieval
positives survive (no scalar cancellation).

The output is a VECTOR over assistant turns (one coefficient per turn), plus
per-turn provenance. It is consumed by _tokenize_seq_perturn (per-token
advantage铺设) — never summed into a scalar.
"""
from __future__ import annotations

import json
import re
from statistics import mean, pstdev

UNKNOWN_MARK = "UNKNOWN"

_VERDICT_RULES = [
    # (substring of verdict, label)
    ("有效", "effective"),          # 有效-概率(通路内) / 有效·质量归因
    ("保留-结构必要", "keep"),      # structural necessity (audit F03 fix,
                                    #   2026-10-12): ladder-protected block,
                                    #   never probability-measured — also
                                    #   lambda-worthy (保留=结构必要∪概率有效)
    ("可移除", "removable"),        # 可移除-概率 / 可移除-环境重复(判死)
    ("不评估", UNKNOWN_MARK),       # f1=0 not evaluated
]


def verdict_label(verdict: str) -> str:
    v = verdict or ""
    for sub, lab in _VERDICT_RULES:
        if sub in v:
            return lab
    return UNKNOWN_MARK


def action_tool_of_turn(trajectory: list[dict], asst_idx: int) -> str | None:
    """Canonical tool name of an assistant turn: JSON action first, flat
    `tool: X` fallback (the old regex path in ONE place, never used for
    credit mapping — idx is)."""
    c = (trajectory[asst_idx].get("content") or "").strip()
    if "</think>" in c:
        c = c.rsplit("</think>", 1)[1].strip()
    for m in re.finditer(r"\{.*\}", c, re.DOTALL):
        try:
            j = json.loads(m.group(0))
            if isinstance(j, dict) and j.get("tool"):
                return str(j["tool"])
        except Exception:
            pass
    ms = re.findall(r"tool:\s*(\w+)", c)
    return ms[-1] if ms else None


def accepted_answer_turn(trajectory: list[dict]) -> int | None:
    """Index of the FINAL ACCEPTED answer assistant turn. An answer is
    accepted when its following tool message is not a rejection/error
    (REJECTED / error: / EMPTY_ANSWER). Earlier format-failed attempts get no
    answer coefficient (plan §5 首版裁定)."""
    out = None
    for i, st in enumerate(trajectory):
        if st.get("role") != "assistant":
            continue
        if action_tool_of_turn(trajectory, i) != "answer":
            continue
        nxt = next((trajectory[j] for j in range(i + 1, len(trajectory))
                    if trajectory[j].get("role") == "tool"), None)
        c = (nxt or {}).get("content") or ""
        rejected = ("REJECTED" in c or c.lstrip().startswith("error:")
                    or "EMPTY_ANSWER" in c or c.lstrip().startswith("entity_error:"))
        if not rejected:
            out = i          # keep scanning: a LATER accepted answer wins
    return out


def group_advantage(f1: float, group_f1s: list[float], eps: float = 1e-6) -> float:
    """(F1_i - mean) / (std + eps); identical-group → 0.0 (answer channel
    closes, retrieval positives survive)."""
    if len(group_f1s) < 2:
        return 0.0
    mu = mean(group_f1s)
    sd = pstdev(group_f1s)
    if sd < eps:
        return 0.0
    return (f1 - mu) / (sd + eps)


def build_turn_coefficients(trajectory: list[dict],
                            blocks: list[dict],
                            f1: float,
                            group_f1s: list[float],
                            lam: float = 0.2):
    """→ (coeff_vector over assistant turns, per-turn provenance list).

    blocks carry b["idx"] = trajectory idx of the SG tool RESULT; the action
    turn is idx-1. All mapping is by INDEX — never by text regex.
    """
    asst_idxs = [i for i, st in enumerate(trajectory)
                 if st.get("role") == "assistant"]
    n = len(asst_idxs)
    coeffs = [0.0] * n
    prov = [{"traj_idx": i, "tool": action_tool_of_turn(trajectory, i),
             "label": "none", "coef": 0.0} for i in asst_idxs]

    # 1. retrieval blocks → action turns by index (idx-1)
    for b in blocks:
        idx = b.get("idx")
        if not isinstance(idx, int) or not (0 < idx < len(trajectory)):
            continue
        if trajectory[idx].get("role") != "tool":
            continue
        if trajectory[idx - 1].get("role") != "assistant":
            continue
        lab = verdict_label(b.get("verdict"))
        # F03 audit fix (2026-10-12): prefer the explicit final_keep field
        # when the judgment producer provides the unified schema; the
        # verdict substring rules remain the fallback for legacy maps.
        if isinstance(b.get("final_keep"), bool):
            if b["final_keep"]:
                lab = "keep" if lab == "keep" else "effective"
            elif lab != "removable":
                lab = UNKNOWN_MARK    # 未知/不评估 blocks stay coefficient 0
        try:
            k = asst_idxs.index(idx - 1)
        except ValueError:
            continue
        if lab in ("effective", "keep"):
            coeffs[k] = lam
            prov[k]["label"] = lab
        elif prov[k]["label"] in ("none",):
            prov[k]["label"] = lab      # removable / UNKNOWN stay coefficient 0

    # 2. answer turn: final accepted answer → group-relative F1 coefficient
    ans = accepted_answer_turn(trajectory)
    ans_coef = group_advantage(f1, group_f1s)
    if ans is not None:
        k = asst_idxs.index(ans)
        coeffs[k] = ans_coef
        prov[k]["label"] = "answer_group"
        prov[k]["coef"] = ans_coef

    for c, p in zip(coeffs, prov):
        p["coef"] = c
    return coeffs, prov
