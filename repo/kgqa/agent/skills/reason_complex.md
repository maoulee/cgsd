---
name: reason_complex
stage: 8
route: COMPLEX
parser_tag: "<answer>\\boxed{...}</answer>"
---

# Capability: reason_complex

## When
COMPLEX turn, final stage. You receive a multi-hop and/or constrained question, its materialized
graph evidence (possibly several relation paths), and candidate entities. Trace the evidence and
derive the answer.

## Inputs
- QUESTION
- answer_type_hint (+ rewritten_hint)
- GRAPH EVIDENCE — one or more reasoning paths / triples across hops
- CANDIDATE ENTITIES

## What you do
Trace how the known entities in the question connect, through the evidence, to candidate answers.
Identify the answer type and every constraint the question states (time, place, quantity,
superlative, uniqueness, negation). Keep every candidate the evidence supports against those
constraints; remove only candidates the evidence explicitly contradicts.

Soft heuristics (apply only when the evidence supports them, never invent):
- A unique role ("the governor / president / leader") with no time qualifier → most recent holder.
- Events / achievements / works / group membership / attributes → return ALL that qualify.
- If the evidence shows no dates, do NOT filter by time.

## Output (emit exactly this; nothing after)
```
<answer>\boxed{exact entity}</answer>
```
Multiple: `<answer>\boxed{e1} \boxed{e2}</answer>`
None: `<answer>None</answer>`

## Hard rules (output only)
- Entity strings copied verbatim from the evidence; full names, never bare years / Freebase IDs.
- "When"-type questions → the event NAME, not a raw year.
- Decide from graph evidence only; over-output is preferred when uncertain.
