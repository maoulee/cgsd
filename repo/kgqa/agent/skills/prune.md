---
name: prune
stage: 2
route: COMPLEX
parser_tag: "<selected>step_N:[...]</selected>"
---

# Capability: prune

## When
COMPLEX turn, after candidate KG relations are retrieved (by embedding similarity) for each step of
the decomposition. Keep the relations that actually express each step's intent.

## Inputs
- QUESTION
- The reasoning steps (the decomposition triples)
- Per-step candidate relations, each with an index and an example triple

## What you do
For each step, pick the 2-5 candidate relations whose semantics directly express what that step
needs (a bridge from the previous output to the next). Drop generic / weakly-related / attribute
relations that don't serve the step. Rank the kept relations by relevance to the step (most relevant
first). If no relation fits a step, give it an empty list.

## Output (emit exactly this)
```
<selected>
step_1: [3, 1]
step_2: [5, 2]
</selected>
```
Numbers are relation indices, ranked (first = most relevant). One line per step.

## Hard rules
- Use only the candidate indices provided for each step. Order by relevance.
- If a step has no fitting relation, emit `step_N: []`.
