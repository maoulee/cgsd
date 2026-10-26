---
name: select
stage: 7
route: COMPLEX
parser_tag: "<selected>[...]</selected>"
---

# Capability: select

## When
COMPLEX turn, after traversal materializes several reasoning paths. Pick the paths to feed to the
answer stage.

## Inputs
- QUESTION (+ rewritten question, answer type)
- Expected relation pattern
- Candidate paths, each a relation chain with endpoint entities, sorted by step coverage

## What you do
Pick 2-4 paths whose relation chain best matches the expected reasoning pattern and that lead to the
expected answer type. Prefer direct semantic matches; add a diverse path only if it captures a
distinct part of the reasoning. Skip paths that are semantically off-track or near-duplicates of an
already-selected path unless they reach a meaningfully different candidate set.

## Output (emit exactly this)
```
<selected>1,3,5</selected>
```
Comma-separated path numbers, best first. No brackets, no explanation.

## Hard rules
- Use only the path numbers provided. Order by final preference (best first).
- Do not invent paths or relations.
