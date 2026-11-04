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

## 9. E-Zero identity test result (2026-11-03)

Real training loss form (chunked top-256 FKL, beta=0, T=1.0, LoRA r64/α128,
lora_plus×4, AdamW lr 2e-5, one step), 5 distinct-case r9a trajectories,
4B, loss on all assistant spans. `ezero/ezero_identity_test.py`.

| condition | teacher input | KL_top256 | KL_full | grad_norm | ‖Δθ‖ after 1 step | probe drift |
|---|---|---|---|---|---|---|
| T0 identity | = student | **0.000000** | 0.000000 | **7.74** | **0.6807** | −0.027 |
| T1 answer-only | +Verified answer | 0.004308 | 0.004321 | 1.34e3 | 0.6809 | −0.140 |
| T2 refined (real OPSD form) | +wrap_structured(render_fact_grouped) | **0.032040** | 0.032089 | 2.97e3 | 0.6809 | −0.089 |

Three findings:

1. **The identity test FAILS at the update level.** KL is exactly 0.000000
   (second-order in the distribution diff), yet grad_norm = 7.74 (FIRST-order
   p−q): bf16 numerical noise between the two forward paths (student:
   grad+checkpointing+adapter wrapper; teacher: no_grad+disable_adapter)
   yields a nonzero gradient from a zero-valued loss. One Adam step then
   moves LoRA params by 0.68 with measurable policy drift. **KL≈0 does not
   imply Δθ≈0 under Adam.**
2. **Update size is condition-independent** (0.6807 / 0.6809 / 0.6809):
   Adam's per-parameter sign-like normalization decouples ‖Δθ‖ from signal
   strength. In parameters the signal does not reach, numerical-noise
   gradients still take full-size ±lr·(lora_plus) steps — a per-step
   diffusion channel that iterates over hundreds of steps.
3. **The real OPSD reference DOES carry signal**: T2 KL = 0.032 = **7.4×**
   the answer-only injection (0.0043) — Exp0's answer-only null does NOT
   transfer to the actual teacher condition (audit hypothesis #1 confirmed).
   And KL_full ≈ KL_top256 in all conditions (truncation blind spot empty —
   hypothesis #3 excluded, consistent with teacher top-256 mass = 1.000).

Caveat: this harness replicates the loss form and optimizer setup, not the
literal swift GKDTrainer path — the real stack should log grad_norm /
update-norm per step and run an identity-batch probe in-situ. Mitigation
candidates worth testing before any long run: (a) skip steps with
loss < floor (1e-4) — kills the pure-noise step; (b) compute the identity
teacher from the SAME forward (detach) instead of a second pass;
(c) larger Adam ε or unnormalized scaling for distillation params.

## 10. E-Zero strict matrix (2026-11-03) — T0 smoking gun RETRACTED as harness artifact

The 2026-11-03 audit found the first E-Zero computed the loss softmax in
BF16 while the real chunked GKD uses `.float()`, and supervised unshifted
positions. Strict rerun (`ezero/ezero_strict.py`; FP32, shifted labels
[lo−1,hi−1), token-mean, teacher adapter ON for T1/T2):

| condition | KL_top256 (tok-mean) | grad_norm | note |
|---|---|---|---|
| I0 same-forward detach | 0.00000000 | **3.7e-03** | math control, fp32 roundoff floor |
| I1 two forwards | 0.00000000 | 3.7e-03 | dual-forward adds nothing |
| I2 teacher adapter off | 0.00000000 | 3.7e-03 | adapter path adds nothing |
| **I3 BF16 (old harness)** | 0.00000000 | **7.74** | the §9 T0 anomaly, reproduced |
| T1 answer-only | 0.00430 | 1.34e3 | survives all fixes |
| T2 refined reference | 0.03202 | 2.97e3 | survives all fixes |

Verdict per the audit's decision rule: **I0–I2 clean, I3 anomalous ⇒ the §9
T0 result was a BF16-softmax backward-rounding artifact of OUR harness, not
a property of the production loss.** The production FP32 path's identity
gradient floor is ~3.7e-3 — six orders below the T2 signal. §9's
"KL≈0 ⇒ full-size Adam update" claim is retracted for the real stack.

Gradient-direction cosines (new): I0/I1/I2 ≡ 1.0 (roundoff is common-mode);
I3 vs anything ≈ 0.13 (random direction — the artifact is uncorrelated
noise); **T1 vs T2 = 0.056 — the answer-only and refined-reference signals
are nearly orthogonal gradient directions**, i.e. the OPSD reference is not
"the answer, stronger" but a qualitatively different signal.

Status per the audit's three layers: VERIFIED — T2 carries a real, 7.4×,
reproducible distribution/gradient signal (this form; `evidence_text` from
the production pool still untested); NOT VERIFIED — whether in-situ GKD
steps show the same; NOT ESTABLISHED — any causal link to the historical
collapses (environment-death forensics stands). The loss<1e-4 skip is a
control arm only, not a fix. Open: in-situ identity probe inside
GKDTrainer.compute_loss; before/after-checkpoint × normal/broken-env 2×2
(OPSD adapters live on the training container, not here).

## 11. Exp1-C: candidate action preference (2026-11-03) — first POSITIVE significant signal

The decisive direction test: does the refined reference shift RELATIVE
preference toward the environment-validated effective action over same-state
legal distractors? (`exp1c/exp1c_candidate_margin.py`; 124 EFFECTIVE blocks;
a- constructed from the SAME retrieve_relations menu at the same state —
soft = 1 chosen relation swapped, hard = all replaced; post-thinking
scoring; 9B.)

| metric | value |
|---|---|
| base margin (none): s(a+) − s(a−soft) | +0.276 |
| **G_T2 = [margin|T2] − [margin|none]** (soft) | **+0.0142, CI95 [+0.0052, +0.0240], p=0.0004, 66% of pairs >0** |
| G_T1 (answer-only, soft) | **−0.0028, CI95 [−0.0043, −0.0014]** — significantly NEGATIVE |
| G_T2 (hard) | +0.0042 (n.s., base margin already 0.32) |
| G_T1 (hard) | −0.0045 (significantly negative) |

**Reading**: the refined decision reference significantly strengthens the
model's preference for its validated effective action over legal
alternatives — the mechanism premise holds for the REFINED reference, in
direction. The answer-only reference is neutral-to-harmful for action
discrimination at every layer measured (exp0 realized logp, exp1a full
distribution, exp1c relative preference). Combined chain:

| layer | answer-only | refined reference |
|---|---|---|
| realized-token shift (exp0) | ≈0 | — |
| full/top-256 KL (exp1a/ezero) | 0.004 | 0.032 (7.4×) |
| gradient direction | orthogonal to refined (cos 0.056) | independent |
| **relative preference G (exp1c)** | **negative** | **positive, significant** |

CGSD's defensible mechanism = refined-DECISION distillation, not
answer-knowledge injection. Magnitude is modest (~5% enhancement of the
base margin) — whether it compounds into training gain needs the M0–M3
ablation. Caveats: a- candidates are constructed (not sampled), but G is a
within-pair contrast so construction bias is symmetric across conditions;
the production-pool `evidence_text` rendering of T2 remains untested.

## 12. Exp1-C2: independent-reference replication (2026-11-03, per audit)

Construction per audit: reference rendered from a DIFFERENT successful
sample of the same case (highest-f1 sibling; scored trajectory never
participates); wrong-reference control = another case's reference;
a+/a- both through the same json.dumps; mean AND sum margins;
per-pair `ref_overlap` flag splits leakage-clean vs correlated-choice
pairs. 78 pairs / ~17 cases; 9B (on-policy) and 4B (off-policy on 9B
trajectories — the actual trainee). Reproducible stats:
`exp1c2/exp1c2_analyze.py` (case-level bootstrap, seed 0).

| model | subset | G_t2_ind | G_t2_wrong | G_t1 |
|---|---|---|---|---|
| 9B | ALL (n=78) | +0.0092, CI [−0.008, +0.016], p=0.28 | −0.0022 | −0.0022* |
| 9B | no-overlap (n=9) | **+0.0041, p<1e-4** | −0.0024** | −0.0019** |
| 4B | ALL (n=78) | **+0.0171, CI [+0.003, +0.020], p=0.0025, 68% pos** | +0.0021 (n.s.) | −0.0033** |
| 4B | no-overlap (n=9) | −0.0043 (fragile) | +0.0062 (fragile) | −0.0041 |

Reading (deliberately narrow):
1. **The effect survives the independent-reference construction at the
   population level in 4B (significant, case-clustered) and directionally
   in 9B**, and the wrong-case reference does NOT reproduce it (≈0 or
   negative) — the shift is case-specific, not generic format priming.
2. **Explanation A vs B (decision competence vs reference-action matching)
   is NOT settled**: the leakage-free no-overlap subset is n=9 pairs from
   ~2–3 cases — both models' subset statistics are directionally unstable
   (9B positive, 4B negative) and the bootstrap CIs over ≤3 clusters are
   degenerate. The binding constraint is DATA (36 cases × 3 samples), not
   compute: a 50-case × 8-sample regeneration would give the no-overlap
   subset enough mass to settle it.
3. 4B shows a LARGER effect than 9B (+0.017 vs +0.009) despite off-policy
   scoring — the trainee may be more reference-responsive than the
   generator (untested hypothesis, consistent with distillation having
   something to teach a weaker policy).

Verdict (updated from §11, per the audit wording): refined decision
references produce a statistically positive, case-specific shift in
preference toward actions observed as effective, constructed
independently of the scored trajectory. Whether this reflects transferable
decision competence rather than reference-action matching requires a
larger no-overlap sample; whether GKD transfers it to the Student is the
One-Step Transfer Test (next).

## 12a. Analyzer correction (2026-11-03, audit #2)

The §12 case bootstrap was WRONG: `rng.choice(c) for c in cs` sampled one
trajectory within each FIXED case — within-case variation, not case
resampling risk. Corrected scheme (cases resampled with replacement,
case-equal weighting; `exp1c2_analyze.py` also now reports true case
count and leave-one-case-out sensitivity, and accepts merged jsonl):

| model | pair-mean | case-equal | correct CI95(case) | pos cases | LOCO-worst |
|---|---|---|---|---|---|
| 9B G_t2_ind | +0.0092 | **+0.00365** | [−0.0066, +0.0153] — crosses 0 | 12/28 | +0.0002 |
| 4B G_t2_ind | +0.0171 | **+0.01166** | **[+0.0004, +0.0248], p=0.021** | 19/28 | +0.0071 |

Also corrected: 78 pairs span **28 cases** (not ~17), and the no-overlap
subset is 9 pairs from **8 cases** — under correct case bootstrap BOTH
models' no-overlap CIs include zero. Verdict: the independent-reference
positive signal retains statistical support in 4B only, and barely; the
A-vs-B question (decision competence vs reference matching) remains open
pending data expansion (diverse-path successes + environment-validated
negatives). Decomposition note (audit): 4B G = Δs(a+) +0.0080 − Δs(a−)
(−0.0092) — the shift raises the effective action AND lowers the
alternative, not a uniform boost.

## 13. One-Step Transfer Test (2026-11-03): FAILS — optimizer dynamics swamp the teacher signal

U0–U3 per design (case split 21 train / 7 held-out, 21 held-out pairs,
frozen teacher, 4B, LoRA r64/α128/lora_plus×4, AdamW lr 2e-5, ONE
accumulated step, held-out scored BARE). `ustep/ustep_transfer.py` +
`ustep/ustep_transfer.json`.

| condition | train KL | grad_norm | held-out ΔM | ΔM − ΔM(U0) | case-equal [CI95] | pos cases |
|---|---|---|---|---|---|---|
| U0 identity | 0.00000 | 2.1e-02 | **−0.0766** | 0 | −0.071 [−0.089,−0.051] | 0/7 |
| U1 answer | 0.00353 | 4.7e3 | −0.0636 | +0.013 | −0.054 [−0.084,−0.024] | 1/7 |
| **U2 refined** | **0.02838** | 1.1e4 | **−0.0733** | **+0.003** | −0.065 [−0.093,−0.041] | 0/7 |
| U3 wrong | 0.01013 | 5.2e3 | −0.0684 | +0.008 | −0.063 [−0.085,−0.043] | 0/7 |

Findings:
1. **One-step transfer fails**: all conditions NEGATIVE on held-out cases
   (0–1/7 positive). U2's differential vs the identity baseline is +0.003
   — indistinguishable from the wrong-reference control (+0.008) and
   SMALLER than answer-only (+0.013). The teacher's positive preference
   (C2) does not survive one full-token GKD step.
2. **The failure is optimizer dynamics, not teacher signal.** The identity
   step U0 — gradient 2.1e-2, pure fp32 roundoff — still moves held-out
   ΔM by −0.077. Adam's first step is sign-normalized
   (g/(|g|+ε)≈sign(g)): every LoRA parameter takes a full ±lr·4 step
   regardless of gradient size. The systematic direction is FLATTENING:
   ΔlogP(a+) < 0 and ΔlogP(a−) > 0 in every condition — high-probability
   tokens down, low-probability up. The one-step noise floor (0.077) is
   ~5× the signal we are trying to transfer (0.014).
3. Per the pre-agreed decision rule ("full-token GKD fails while the
   teacher's independent-reference preference is positive ⇒ test
   selective distillation, don't abandon"), the next interventions are:
   (a) loss on ACTION-SPAN tokens only (thinking/format excluded) to cut
   the flattening surface; (b) optimizer-side controls (loss floor skip,
   higher Adam ε or unnormalized scaling, lr warmup) to kill the sign-step
   floor. Caveats: 7 held-out cases / 21 pairs (small); one step with
   production lr does not extrapolate to multi-step with momentum — but
   it establishes that the signal is invisible at single-step granularity.

## 14. Z-series controls + normalized U-series rerun (2026-11-03): §13 mechanism RETRACTED; transfer failure CONFIRMED and sharper

Audit fixes applied: token-mean normalization in backward (production form),
per-pair records + PAIRED case-level differential, gradient quantiles,
production first-step optimizer variant (warmup lr 2e-5/5 + clip 1.0).

Z-controls (identity, held-out ΔM): Z0 no-step = **+0.000000**,
Z1 zero-grad+step = **+0.000000** (eval path clean); Z2 identity with
token-mean = **−0.0009** (grad 3.6e-7, ZERO coordinates above Adam ε);
Z3 production first-step config = −0.0011. **The §13 U0 −0.077 was an
artifact of our unnormalized kl.sum() backward** (5 orders of gradient
scale inflation → sign-step regime). Retracted per the audit's warning;
the production-equivalent identity noise floor is −0.001.

Normalized U-series rerun (the decisive table):

| condition | train KL | grad | held-out ΔM | paired diff vs U0 (case CI95) |
|---|---|---|---|---|
| U0 identity | 0 | 3.5e-7 | −0.0014 | — |
| U1 answer | 0.0035 | 9.9e-2 | −0.0636 | **−0.052 [−0.081,−0.023] p=1e-4** |
| U2 refined | 0.0284 | 2.3e-1 | −0.0728 | **−0.063 [−0.091,−0.038] p≈0** |
| U3 wrong | 0.0101 | 5.2e-2 | −0.0670 | **−0.066 [−0.081,−0.041] p≈0** |

Reading (per the audit's calibrated wording): one-step transfer was not
observed — and now with a clean floor: **every REAL reference condition
(correct / answer-only / WRONG-case) damages held-out margins by the same
−0.06 with zero reference-specificity** (U2≈U3≈U1). The damage is a
property of a full-token GKD step at production lr under Adam sign
normalization (real KL gradients put coordinates >> ε), not of the
identity noise (now −0.001) and not of reference content. The teacher's
positive preference (C2) does not survive the step; the step itself is
destructive regardless of what the teacher knows. Next per plan: V-series
2×2 (loss scope × optimizer stability, e.g. U2 at Z3's warmup lr+clip =
V1) to test whether the damage can be suppressed while signal survives.

## 15. V-series 2×2 + 5-step probe (2026-11-03): FINAL verdict of the audit arc

V-series (refined reference U2-form; scope × optimizer; paired vs the
identity floor; 21 held-out pairs / 7 cases):

| group | steps | held-out ΔM | diff vs floor (case CI95) | pos cases |
|---|---|---|---|---|
| V0 full + orig lr | 1 | −0.0728 | −0.071 [−0.091,−0.038] | 0/7 |
| V1 full + prod (warmup+clip) | 1 | **−0.0089** | −0.008 [−0.0088,−0.0051] | 0/7 |
| V2 action-only + orig | 1 | −0.0540 | −0.053 [−0.066,−0.027] | 0/7 |
| V3 action-only + prod | 1 | **−0.0069** | −0.006 [−0.0067,−0.0035] | 0/7 |
| V1 full + prod | 5 | −0.0084 | −0.006 [−0.0120,+0.0052] **CI crosses 0** | 3/7 |
| V3 action-only + prod | 5 | **−0.0239** | −0.020 [−0.0333,−0.0050] worse | 1/7 |

FINAL READINGS:
1. **Optimizer stabilization works**: warmup+clip removes ~90% of the
   one-step damage (V0→V1); with it, 5 steps are STABLE (V1 flat at
   −0.008, CI crossing zero — no drift accumulation). The historical
   "training wrecks the policy" phenomenon is reproducible and fixable
   at the optimizer level.
2. **Positive transfer is NOT observed under any tested form**: full /
   action-only × 1 / 5 steps × orig / prod optimizer — best case is
   neutral (V1), never positive. The teacher's +0.014 reference-
   conditioned preference does not arrive in the unconditioned policy.
3. **Action-only is not the fix**: at 1 step it matches full+prod
   (−0.007 vs −0.009), but at 5 steps it DEGRADES (−0.024, sig.) —
   concentrating supervision on fewer tokens concentrates the residual
   flattening damage instead of the signal.
4. Per the pre-agreed decision rule ("if no arm improves held-out
   margin, re-evaluate the teacher's training signal rather than tuning
   lr further"): the remaining explanations are (a) the teacher's
   preference is reference-matching rather than transferable decision
   competence (the unsettled A/B question, now the leading hypothesis),
   or (b) per-token GKD is the wrong vehicle for decision-level signal
   regardless of scope (→ contrastive/advantage forms, GRPO/RSCC track).
   Multi-step accumulation shows flat, not rising — (c) "just needs more
   steps" is not supported.

## 16. G_k/M_k dual-arm tracker (2026-11-03): frozen vs synced teacher — the user's question answered

Fixed held-out states + validated candidates, evaluated every 2 steps over
10 steps (V1 form: full-token GKD, stabilized optimizer; the ONLY
difference between arms is teacher co-movement; `ustep/gmk_{frozen,synced}.json`).

| step | M_k frozen (bare preference) | G_k frozen | M_k SYNCED | G_k SYNCED |
|---|---|---|---|---|
| 0 | +0.2633 | +0.0091 | +0.2633 | +0.0091 |
| 2 | +0.2361 | +0.0125 | +0.2192 | +0.0125 |
| 4 | +0.2550 | +0.0110 | **+0.1378** | +0.0188 |
| 6 | +0.2540 | +0.0113 | **+0.1039** | +0.0216 |
| 8 | +0.2519 | +0.0137 | +0.1180 | +0.0265 |
| 10 | +0.2594 | +0.0136 | **+0.1282** | +0.0267 |

Readings (per the audit's quadrant table, plus one pattern beyond it):
1. **FROZEN arm**: M stable (−1.5% over 10 steps, recovers from the step-2
   dip), G persists at +0.009..0.014 — quadrant "teacher retains guidance
   advantage, student does not absorb it". No harm, no transfer.
2. **SYNCED arm** (= the production configuration KGQA_TEACHER_SAME_WEIGHTS=1):
   M COLLAPSES −51% by step 10 (0.263→0.128, floor 0.104) while G RISES 3×
   (0.009→0.027). The policy is becoming MORE reference-dependent while its
   unconditioned decision skill erodes — it learns to need the crutch while
   losing the ability to walk. A dynamic teacher does not just fail to
   help; it amplifies student degradation.
3. Combined with §15: the final mechanism verdict — per-token GKD never
   transfers the teacher's reference-conditioned decision advantage to the
   bare policy (M never rises in ANY arm), and with a co-moving teacher it
   actively destroys it. The two production-relevant fixes (frozen teacher,
   stabilized optimizer) remove the harm and expose the vacancy: the
   decision knowledge does not travel through this loss.

## 17. W-series: single-position learning verification (2026-11-03) — the interference localized to THINKING tokens

Codex's minimal experiment, executed: 8 train-case decision points where
the premise holds (frozen teacher + refined reference assigns higher
probability to the correct-relation tokens: q−p = +0.032, 5/8 points).
One stabilized Adam step per arm, frozen teacher, token-mean, in-situ
re-scoring of the SAME states. `ustep/wseries_singlepos.py` +
`ustep/wseries_1step.json`.

| arm | supervision scope | Δp(correct-relation tokens) | positive points |
|---|---|---|---|
| premise | — (teacher advantage) | +0.032 | 5/8 |
| **W1** | decision tokens only (relation names) | **+0.044** | **5/8 — LEARNS, exceeds the teacher gap** |
| **W2** | whole action JSON | **+0.039** | 4/8 — still learns |
| **W3** | + thinking (full turn) | **−0.003** | 2/8 — **the signal dies** |

Readings:
1. **The training function is NOT fundamentally broken**: W1/W2 learn
   in-situ exactly as OPD math predicts (gradient p−q → p(correct) rises).
   Nor is the environment involved (fully offline).
2. **The interference is localized at the W2→W3 boundary: THINKING-token
   gradients cancel the decision-token learning** within a single turn
   (decision delta collapses +0.039 → −0.003 even as the overall action
   span still rises +0.043 on non-decision tokens).
3. Combined with V-series: a second dilution layer exists across turns
   (multi-turn action-only never transfers and degrades over steps), but
   the primary killer within a turn is thinking supervision.
4. The working recipe (W1) is now fully specified: frozen teacher +
   stabilized optimizer + token-mean + DECISION-TOKEN-ONLY loss →
   in-situ learning +0.044. This is the mechanistically-grounded version
   of "selective distillation" — the selectivity criterion is the
   relation-choice positions, and it follows directly from the escalation
   chain, not from crash-avoidance heuristics.
   Open: whether W1-scope learning generalizes to held-out cases (the
   C2 A/B question still governs that step).

## 17a. W-series RAW-FORMAT rerun (2026-11-03, user caught a fidelity gap)

The user pointed out the environment uses the FLAT MD protocol, not JSON.
Verified: the model's actual output (raw field) is
`tool: retrieve_subgraph\ncenter: ...\nrelations: a | b | c`; the JSON in
`content` is the environment's canonicalized parse. The original W-series
supervised a re-serialized JSON form — a fidelity deviation. Rerun on the
model's REAL output tokens (decision span = the values after `relations:`
to end of line, within the raw text):

| arm | JSON version (re-serialized) | **RAW version (real output)** |
|---|---|---|
| premise q−p | +0.032 (5/8) | **+0.023 (4/8)** — holds |
| W1 decision tokens only | +0.044 (5/8) | **+0.017 (4/8)** — still learns |
| W2 whole flat action | +0.039 (4/8) | **+0.018 (4/8)** — still learns |
| W3 + thinking | −0.003 (2/8) | **+0.001 (5/8)** — dead (−95% vs W2) |

The structure fully transfers to the production format: decision-token-only
GKD learns in-situ; thinking supervision collapses it (0.018 → 0.001).
NOTE: the JSON magnitudes were inflated (out-of-distribution serialization
→ larger gradients); **the RAW numbers (+0.023 premise / +0.017 learning)
are the production-fidelity effect sizes** and should be the cited ones.
Production implementation of selective distillation supervises the
`relations:` line values in the flat protocol.

## 18. Per-tool-layer teacher advantage (2026-11-04): the supervision mask is NOT the whole tool layer

q−p on ACTION tokens per tool type (raw flat protocol, 4B, 170 turns,
sibling refined reference; `ustep/tool_layer_advantage.{py,json}`):

| tool | n | q−p (whole action) | pos% |
|---|---|---|---|
| retrieve_subgraph | 63 | +0.0021 | 52% |
| retrieve_relations | 48 | −0.0004 | 56% |
| answer | 30 | **+0.0049** | 57% |
| **plan** | 29 | **−0.0468** | **34%** |

Readings:
1. **plan turns carry NEGATIVE teacher advantage** (−0.047, only 34%
   positive): the reference restates a decomposition, and the teacher's
   plan output moves AWAY from the student's original (successful-trajectory)
   plan. Supervising plan turns would pull the student toward
   reference-style plan text — a distribution shift, not a decision gain.
   This also retro-explains part of V-series action-only failure (its
   scope included plan).
2. retrieve_subgraph's whole-action average (+0.002) is the relations
   sub-field (+0.023, W-series) diluted by scaffolding tokens.
3. answer is mildly positive (+0.005) — largely answer-echo (exp0), the
   correctness materialization point; optional weak supervision.
4. Data-driven supervision mask: **relations values (primary, +0.023) +
   optionally answer_values (weak, echo) — exclude plan, retrieve_relations,
   thinking, and scaffolding**.

## 19. Answer-layer decomposition (2026-11-04): no supervision needed — student ceiling

Final accepted answer turns (f1>0 trajectories, n=28, sibling refined
reference, raw flat format, 4B; `ustep/answer_decomp.json`):

| quantity | value |
|---|---|
| q−p on answer_values tokens | **+0.00002 (14/28 — coin flip)** |
| q−p on whole answer action (incl. scaffolding) | +0.0004 |
| **student baseline p(own correct answer values)** | **logp −0.015/token ≈ p 0.985 — CEILING** |

Reading: on successful trajectories the student already produces its
correct answer tokens with near-certainty given its own evidence — there
is no headroom for the teacher to raise and nothing to teach. This also
reconciles the earlier exp0 +0.006 "answer echo": that was measured under
DIRECT answer injection (the GT string copied into the prompt) across all
answer turns; as TRAINING supervision on successful trajectories the
answer layer contributes ≈ zero gradient. **The supervision mask
finalizes to: relations values ONLY.** Excluded: plan (negative −0.047),
retrieve_relations (zero), answer_values (zero, ceiling), thinking
(cancels), scaffolding (dilutes).
