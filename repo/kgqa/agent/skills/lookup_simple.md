---
name: lookup_simple
stage: 1
route: SIMPLE
parser_tag: "<answer>{json triple}</answer>"
---

# Capability: lookup_simple

## When
SIMPLE turn, decomposition stage. A direct one-hop question needs to become a single retrieval
triple anchored on the known entity. (This replaces the heavy multi-step cascade for simple cases.)

## Inputs
- QUESTION
- KNOWN ENTITIES (the anchor candidates)

## What you do
Pick the single known entity the question is about (the anchor) and express the question as ONE
retrieval triple: anchor — relation phrase — unknown. The relation phrase is a short natural
description of the link the question asks about (e.g. "capital of", "place of birth", "language
spoken"). The object is the unknown answer variable `?`.

## Output (emit exactly this)
```
<answer>{"subject": "<anchor>", "predicate": "<relation phrase>", "object": "?"}</answer>
```

## Hard rules
- Copy the anchor name verbatim from the known entities.
- Exactly one triple. Do not invent endpoints or constraints not in the question.
