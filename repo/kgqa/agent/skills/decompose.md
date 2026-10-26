---
name: decompose
stage: 1
route: COMPLEX
parser_tag: "<answer>{json triples}</answer>"
---

# Capability: decompose

## When
COMPLEX turn, decomposition stage. Break a multi-hop / constrained question into an ordered set of
retrieval triples that form a reasoning path from a single anchor entity to the answer.

## Inputs
- QUESTION
- KNOWN ENTITIES

## What you do
Choose ONE anchor entity (the unique starting point the question is about). Produce an ordered list
of triples that a graph walk can follow from the anchor, through intermediate variables, to the
answer variable, attaching any constraint (place/time/quantity/value) to the variable it constrains.
Label other mentioned known entities as `pathentity` only if they appear in a triple; otherwise
`unusedentity`.

## Output (emit exactly this)
```
<answer>{"anchor": "<entity>", "answer_variable": "?answer", "answer_type": "<noun phrase>", "entity_roles": [{"entity": "...", "role": "anchorentity|pathentity|unusedentity", "reason": "..."}], "triples": [{"source_subquestion": 1, "subject": "...", "predicate": "<relation phrase>", "object": "...|?var"}]}</answer>
```

## Hard rules
- Predicates are short natural-language relation phrases (not KG relation IDs).
- Exactly one anchor. Reuse intermediate variables across triples. Do not duplicate a fact forward
  and reverse. Do not answer the question or substitute real entities for unknowns.
