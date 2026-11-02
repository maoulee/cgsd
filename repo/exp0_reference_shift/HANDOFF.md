# Experiment 0: Reference-Conditioned Distribution Shift — Handoff

Offline, zero-training validation of the CGSD/OPSD mechanism premise:
**does giving the same model the correct reference answer change its behavior
probability distribution on the same trajectory prefix?**

Verdict (narrowed 2026-11-02 after audit): **a simple answer-only reference
shows no detectable action/thinking log-probability shift on realized
trajectories; answer-knowledge transfer is weakly supported.** The ACTUAL
OPSD teacher condition (refined decision reference from
build_teacher_prompts) is NOT tested here — see §5. This experiment measures
selected-token log-probs only; full-vocabulary distribution shift (what GKD
optimizes) is the follow-up Exp1-A.

## 1. Setup

- **Data**: `results/prob_pathway_r9a_v5.json` — Reverse Dynamic Leave
  verdicts over 108 trajectories (r9a hard-cohort, 36 cases × 3 samples).
  f1>0 surface: 155 judged blocks — EFFECTIVE 125 / REDUNDANT 27 / INVALID 3;
  plus 81 final-accepted answer turns. Labels per the 5-class scheme
  (removable≠invalid; unknown excluded).
- **Model**: Qwen3.5-9B — the same policy that generated the trajectories
  (r9a ran LLM_MODE=http default 9B). Verified empirically: baseline
  logp of own action tokens ≈ −0.28/token.
- **Prefix fidelity**: system prompt = C123 + adaptive routes, reconstructed
  **byte-exact** (sha256 `36edaf91…`, 91,502 chars, matches the run log).
  History assistant turns rendered in the training shape
  `<think>…</think>\n\n<action>` (TITO-consistent).
- **Conditions** (only the reference differs; history token-identical):
  - `none` — original prompt
  - `correct` — `\n\n[Reference]\nVerified answer: <case golds>`
  - `wrong` — same format, another case's golds (count-matched)
  - `placebo` — same format, neutral phrases (length-matched)
- **Positions** (both run):
  - `system` — suffix appended to the system prompt (the actual OPSD
    teacher-conditioning design)
  - `near` — suffix appended to the LAST history message, immediately before
    the scored turn (attention-salience control)
- **Spans scored separately** (full-vocab log_softmax, never top-256):
  action span (tool-call JSON; primary), thinking span (diagnostic),
  answer turn. Scoring = HF forward → hidden states → lm_head in
  512-position chunks (vLLM prompt_logprobs materializes [T,V] logits for
  whole 24K-token prompts → OOM/NCCL-deadlock; chunked path peaks ~300MB).
- **Runs**: 944 forwards per position (236 turns × 4 conditions), dual-GPU
  sharded, ~18 min per position. Row schema:
  `case_id, sample, block_idx, verdict, label5, condition, checkpoint_hash,
  n_span/n_think, logp_mean/sum (action), think_logp_mean/sum`.

## 2. Results (Δ = condition − none, per-token mean; paired by block)

### Action spans (tool name + args; thinking excluded)

| group | n | Δcorrect | Δplacebo | note |
|---|---|---|---|---|
| EFFECTIVE | 125 | **−0.0001** (p10/p90 ±0.003) | +0.000 | symmetric noise band; 27% up vs 26% down |
| REDUNDANT | 27 | +0.0002 | +0.000 | identical band — no separation |
| INVALID | 3 | +0.0014 | +0.001 | n too small; same band |

Separation (case-level paired bootstrap): all CIs cross zero,
condition specificity ≈ 0. **Same at both injection positions.**

### Thinking spans

EFFECTIVE: Δcorrect = −0.0012 (p10/p90 [−0.006, +0.005]).
**75% of turns have genuine uncertainty headroom** (baseline
thinking logp < −0.3, i.e. p < 0.74/token) — on those 177 turns
Δcorrect = **−0.0015 (still zero)**. The null is not a ceiling artifact.

### Answer turns (the only non-zero channel)

| metric | value |
|---|---|
| Δcorrect (system) | **+0.0062** CI95 [+0.0011, +0.0121], p=0.009 |
| Δwrong (system) | +0.0024 (CI crosses 0) |
| Δplacebo (system) | −0.0034 |
| specificity (correct−placebo) | **+0.0095** CI95 [+0.0049, +0.0148], p<1e-4 |
| Δcorrect (near) | +0.0077; specificity +0.010, p=0.03 |

Distribution is **right-skewed**: median sequence-likelihood ratio ×1.07,
p90 **×1.81**, max ×12.6. Interpretation: the shift concentrates where the
answer string literally echoes the reference (answer_values tokens) — a
**copy effect, not a decision effect**.

## 3. Verdict (pre-registered acceptance table)

Hits the row **"only the Final Answer changes → answer-knowledge transfer
only; agent-policy transfer unsupported"**. Per the task-book gate:
do NOT start 100-step OPSD long training; do not claim Agent Policy
Distillation; Answer Knowledge Distillation is the defensible framing.

## 4. Why it fails — mechanism

1. **Action = transcription of thinking.** Given its own <think>, the
   model's action JSON is near-determined (baseline −0.28/token, p≈0.75).
   The reference can only act upstream, in thinking.
2. **Thinking doesn't move either** — and NOT for ceiling reasons (75% of
   turns have headroom; Δ still −0.0015). Teacher-forcing locks the path:
   turn k's behavior is bound by turns 0..k−1 (plan structure, chosen
   centers, delivered evidence). The reference does not retroactively
   re-route an on-policy history. The model treats it as information about
   the ANSWER, not guidance about the WALK.
3. **Position control**: near-injection (reference right before the scored
   turn) leaves action/thinking at zero → not an attention-distance
   artifact.

Net: **π(a|h,z) ≈ π(a|h) on realized trajectories.** The GT-conditioned
teacher distribution ≈ student distribution → FKL/GKD gradient on these
tokens carries almost no contrast. Consistent with (a) the paper-side
CGSD teacher−student deltas of only −0.04..−0.15, (b) the code audit that
counterfactual verdicts never enter the OPSD gradient, (c) the GrailQA
finding that cognition is not the bottleneck.

## 5. Known limitations

- **Ceiling on the raise-direction (by design)**: scoring the REALIZED
  action conditions on the model's own thinking — the maximum-likelihood
  continuation is itself. This limits how much Δcorrect can rise, though
  the suppression side (demoting removable/invalid realized actions) had
  room and showed nothing.
- One injection template (one-line `Verified answer:` block) — this is
  answer-only privileged information, NOT the OPSD refined decision
  reference (`[REFINED EXPERIENCE DATA]` with answer target / FACT chains /
  relation choices, build_teacher_prompts L586-680, injected by
  kgqa_scheduler L1185-1243). The OPSD teacher condition itself is untested.
- Model: 9B (the trajectory generator). OPSD trains 4B (opsd_train.yaml) —
  a 4B replication is required before inference to the trainee.
- Selected-token log-prob only: a zero shift does NOT rule out
  redistribution over un-sampled candidates (KL(q||p)>0 with equal realized
  token prob). Full-vocab + top-256 GKD KL = Exp1-A.
- Near injection appends to the last history message (often a tool result)
  and falls back to system on turn 0 — exploratory only, not a clean
  position control.
- INVALID n=3 (env-repeat blocks only in this dataset) — the negative
  group is thin; REDUNDANT carries the contrast.

## 6. Remaining open tests (Exp1-A/B/C; see audit 2026-11-02)

Exp1-A full-distribution audit (full KL + top-256 GKD KL + entropy +
top-1 margin); Exp1-B reference decomposition (answer-only /
decision-only / full refined reference); Exp1-C candidate preference G.

### Original open test (Figure 3, candidate margin)

The design cannot answer whether z changes **which action would be chosen**
— only whether it re-weights the taken one. Construct, per state,
validated candidate pairs (a+ = effective action, a− = a plausible wrong
action at the same point, matched format); compute
G = [s_z(a+) − s_z(a−)] − [s_0(a+) − s_0(a−)]. G > 0 ⇒ the reference
shifts RELATIVE preference (a DPO-shaped signal) even though realized
actions are saturated. Given baseline saturation the prior is weak, but
this is the decisive remaining check before abandoning the mechanism.

## 7. Files

| path | what |
|---|---|
| `exp0_reference_shift.py` | runner (both positions, dual spans, chunked lm_head) |
| `exp0_analyze.py` | group stats + paired bootstrap + figures |
| `results/system_position.jsonl` | 944 rows, system-injection |
| `results/near_position.jsonl` | 944 rows, near-injection |
| `results/{system,near}_stats.json` | analyzer outputs per position |
| `results/fig_{system,near}_position.png` | groups × conditions bars |
| `results/prob_pathway_r9a_v5.json` | RDL verdicts + golds + p0 vectors |

Note: row field `checkpoint_hash` is the sha256 of the model's
`config.json` (architecture/version marker), NOT a weights hash.

The 55MB source rollout report is not shipped (regenerable; rows in the
results carry all analysis fields).

## 8. Exp1-A result (2026-11-02, executed same day as the audit)

The audit's central objection — "selected-token logp ≈ 0 does not rule out
full-distribution redistribution" — is now measured directly
(`exp1a/exp1_dist_audit.py`, both 9B and 4B, system injection, all four
conditions, thinking+action spans):

| quantity | 9B | 4B | reading |
|---|---|---|---|
| KL_full (action span, correct) | **0.0002** | 0.0004 | full 248K-vocab distribution unchanged |
| KL_top256 (GKD support) | 0.0002 | 0.0004 | identical to full — teacher top-256 mass = 1.000, so the GKD support ≈ the whole distribution |
| KL_full (thinking, correct) | 0.0022 | 0.0024 | trace perturbation, correct > wrong (0.0012) > placebo (0.0009) — ordered but ~0.002 nats |
| KL_full (answer turn, correct) | 0.0029 | 0.0017 | same order of magnitude |
| top-1 change rate | 0.002–0.016 | 0.002–0.016 | argmax flips on ≤1.6% of positions |
| entropy delta | ≤0.0016 nats | ≤0.0016 | no sharpening/flattening |

**Per the audit's quadrant table: Full KL ≈ 0 AND Top-256 KL ≈ 0 ⇒ the
reference does not change the predictive distribution.** The measurement-
artifact hypothesis is rejected: there is no hidden redistribution that
Exp0's selected-token metric missed. Both models agree (the 4B trainee
shows the same null as the 9B generator). One anomaly: any REAL answer
(correct or wrong) perturbs slightly more than placebo — but at 0.002–0.005
nats this is not a training-scale signal.

Remaining per the audit: Exp1-B (reference decomposition — the refined
decision reference is now the ONLY untested injection with a live
mechanism hypothesis) and Exp1-C (candidate preference G). Files:
`exp1a/dist_audit_{9b,4b}.jsonl` (row schema adds kl_full/kl_top256 means
and sums, mass256 both sides, top1_change_rate, entropy_delta, recomputed
logp_shift, per span).
