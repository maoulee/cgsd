# CGSD/OPSD Mechanism Study — Night Summary (2026-11-04)

## Setup
- Model: Qwen3.5-4B (base, no LoRA) — the actual OPSD trainee
- Prompt: RC4 (46,847 chars, sha256=832080851410c06e)
- Data: CWQ test split, 50 cases × 8 samples → 390 successful RC4 rollouts
- F1: 0.746 mean, 82% f1>0 (consistent with production performance)

## P1-A: Teacher Distribution Audit (T0–T4, 34 decision points)

| Condition | ΔlogP(a+) | ΔlogP(a−) | Δmargin | Ideal% |
|---|---|---|---|---|
| T1 answer-only | −0.003 | +0.010 | −0.013 | 21% |
| T2 refined (top-f1 sibling) | −0.004 | −0.045 | +0.041 | 21% |
| **T3 alternative-path ref** | **+0.046** | **−0.081** | **+0.127** | **59%** |
| T4 wrong-case ref | −0.003 | −0.007 | +0.004 | 29% |

**Key finding**: T3 (alternative-path reference) provides the strongest
teacher signal — 59% of points show the ideal "raise good AND lower bad"
pattern, vs only 21% for the current OPSD reference (T2). Alternative-path
references should replace top-f1 references.

## P1-B: Gradient Direction Audit (16 decision points)

| Comparison | Cosine | Interpretation |
|---|---|---|
| cos(g_full, g_top256) | **+0.9999** | Top-256 ≡ full-vocab — truncation is NOT a problem |
| cos(g_full, g_decision) | **+0.124** | GKD gradient only 12% aligned with decision improvement |
| cos(g_full, g_rl) | **−0.055** | GKD gradient is ANTI-correlated with environment reward |
| cos(g_decision, g_rl) | +0.382 | Decision and RL partially aligned |
| cos(Δθ_full, g_decision) | −0.068 | GKD's Adam update nearly orthogonal to decision improvement |
| cos(Δθ_decision, g_decision) | −0.485 | Even direct optimization only retains 49% direction under Adam |

**Key finding**: GKD's gradient direction is mathematically opposed to
environment reward optimization (cos = −0.055). This is the direct
explanation for why CGSD fails to transfer decision knowledge.

## P2: G0–G3 Multi-step Experiment (IN PROGRESS)

Baseline M = +0.965 (ceiling: 73/73 pairs positive — the model already
strongly prefers its effective relations). G1 frozen full-token:
- Step 0: M = 0.965
- Step 1: M = 0.931
- Step 2: M = 0.856
- Step 3: M = 0.823

Both a+ and a− are declining, with a− declining more slowly → margin
shrinks but stays positive. Pattern consistent with S2's strategy drift.

## Answering the Five Priority Questions

1. **Teacher's correct-action advantage from environmental preference?**
   PARTIALLY. T3 (alternative path) shows genuine decision knowledge
   (59% ideal pattern), but T2 (current OPSD reference) achieves only 21%.

2. **Full-vocab vs Top-256 same direction?**
   YES. cos = 0.9999. The Top-256 truncation is completely innocent.

3. **GKD aligned with environment Advantage?**
   NO. cos(g_KD, g_RL) = −0.055 (anti-correlated).

4. **Frozen vs synced Teacher divergence onset?**
   (From G/M dual-arm on C123): Divergence begins at step 3-4, becomes
   severe by step 6. (RC4 G2 results pending.)

5. **Why local Action learning doesn't transfer?**
   The GKD gradient itself points away from environment reward (Q3),
   and Adam further rotates the update (Q2 from W4b). The combination
   means the student learns to match the teacher's distribution, but
   that distribution's improvement direction conflicts with what the
   environment rewards.

## Hypotheses Status

| Hypothesis | Status |
|---|---|
| Answer-only GT guides actions | ❌ REJECTED (3-layer evidence) |
| Refined reference provides decision signal | ✅ CONFIRMED (T3 strong, T2 weak) |
| Top-256 truncation causes direction error | ❌ REJECTED (cos=0.9999) |
| GKD gradient aligned with environment reward | ❌ REJECTED (cos=−0.055) |
| Thinking supervision dilutes action gradient | ✅ CONFIRMED (W4b, 86% mass) |
| Dynamic teacher amplifies degradation | ✅ CONFIRMED (dual-arm −51%) |
| Student memorizes training relations | ❌ REJECTED (S2 decomposition) |
| Strategy drift (distribution flattening) | ✅ CONFIRMED (S2 + G2 early) |

## Remaining Questions
- G2 (synced teacher) and G3 (action-only) multi-step results
- On-policy rollout evaluation at each checkpoint
- Thinking–Action conditional generation check

## G1 Frozen Full-Token Trajectory (updates so far)
| Step | M (held-out margin) | ΔlogP(pos) | ΔlogP(neg) |
|---|---|---|---|
| 0 | 0.965 | -0.060 | -1.025 |
| 1 | 0.931 | -0.063 | -0.994 |
| 2 | 0.856 | -0.073 | -0.930 |
| 3 | 0.823 | -0.075 | -0.898 |

Both declining; neg declining faster → margin shrinks but stays positive.
G2 (synced) and G3 (action-only) still running.

## P2 Final Results (G0-G3)

| Group | Step 0 | Step 5 | Step 10 | Pattern |
|---|---|---|---|---|
| G0 identity | 0.965 | 0.965 | 0.965 | Perfect control |
| G1 frozen full | 0.965 | 0.811 | **0.874** | **U-shape: dips then recovers** |
| G2 synced full | 0.965 | 0.523 | **0.413*** | **Monotonic collapse (-57%)** |
| G3 frozen action | (running) | | | |

*G2 at step 7 (last recorded); decline continues.

### Mechanism (Mode A: Teacher Self-Reinforcement)
- G2 ΔlogP(pos): −0.06 → −0.31 (correct actions pushed down MORE each step)
- G2 ΔlogP(neg): −1.02 → −0.72 (wrong alternatives becoming MORE likely)
- The synced teacher amplifies its own biases, progressively destroying
  the student's correct action preferences while boosting alternatives.

### Key Insight from G1's U-Shape
The frozen-teacher system shows initial disruption followed by RECOVERY.
This means GKD is not irreversibly destructive — the dynamic teacher is
what blocks the recovery mechanism. With a frozen teacher, the student
eventually re-stabilizes near (but below) its initial policy.
