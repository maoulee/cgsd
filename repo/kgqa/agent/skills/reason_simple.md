---
name: reason_simple
stage: 8
route: SIMPLE
parser_tag: "<answer>\\boxed{...}</answer>"
---

# Capability: reason_simple

## When
SIMPLE turn, final stage. You receive a direct one-hop question, its graph evidence, and the
candidate entities found by traversal. Derive the answer.

## Inputs
- QUESTION
- answer_type_hint (+ rewritten_hint)
- GRAPH EVIDENCE — the relation chain / triples materialized from the anchor
- CANDIDATE ENTITIES — entities reached by traversal

## What you do
This is a direct one-hop lookup. Identify which candidate entity (or entities) the evidence
**directly shows** as the answer to the question, honoring any constraint the question states.
Use only the graph evidence. If several candidates qualify, list them all.

## Output (emit exactly this; nothing after)
```
<answer>\boxed{exact entity}</answer>
```
Multiple: `<answer>\boxed{e1} \boxed{e2}</answer>`
None: `<answer>None</answer>`

## Hard rules (output only)
- Entity strings copied verbatim from the evidence; full names, never bare years / Freebase IDs.
- "When"-type questions → the event NAME (e.g. "2014 World Series"), not a raw year.
- Decide from graph evidence only.
