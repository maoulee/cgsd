"""ENTITY KINDS — the single source of truth for entity classification
(cleanup charter C1, user ruling 2026-09-24: the data channel is
pattern-layer walk -> recorded pattern paths -> pattern paths restored to
entity evidence, with EVERY participant variable consistent across layers).

Four kinds:
  NAMED    — a displayable entity (answer surface, walk node)
  CVT      — event/record mediator node (m./g. prefix): transparent in
             walks, its attribute edges inline at render
  ID_NODE  — opaque code node (no space + 2+ consecutive digits, <=24
             chars): a transit surface exactly like a CVT (25_db96)
  LITERAL  — value node (dates, numbers, strings): legal terminal of a
             value-typed final hop (1812: [number: 610])

The walk/render/pool layers previously carried FIVE divergent copies of the
CVT predicate and THREE non-equivalent literal checks; this module is the
only place those distinctions live.
"""
from __future__ import annotations

import re

_ID_RE = re.compile(r"\d{2,}")


def entity_kind(name) -> str:
    """NAMED | CVT | ID_NODE | LITERAL — one authority for every layer."""
    s = str(name)
    if s[:2] in ("m.", "g.") and len(s) > 4:
        return "CVT"
    if _is_literal(s):
        return "LITERAL"
    # ID code nodes carry LETTERS + digits (travelid, TAC123, kr1206);
    # pure numbers/dates are LITERALs (checked above).
    if (len(s) <= 24 and " " not in s and _ID_RE.search(s)
            and re.search(r"[A-Za-z]", s)):
        return "ID_NODE"
    return "NAMED"


def is_cvt(name) -> bool:
    return entity_kind(name) == "CVT"


def is_id_node(name) -> bool:
    return entity_kind(name) == "ID_NODE"


def is_literal(name) -> bool:
    return entity_kind(name) == "LITERAL"


def is_transit(name) -> bool:
    """Transit surfaces cost no named hop in walks/pools: CVTs and ID nodes."""
    return entity_kind(name) in ("CVT", "ID_NODE")


def _is_literal(s: str) -> bool:
    # Value nodes: bare dates/times/numbers (the discriminating terminals of
    # value-typed relations — '610', '1994-03-23-08:00', '1985-08:00').
    # Kept conservative: anything with a space or letters beyond a trailing
    # 's' is a name, not a literal.
    return bool(re.fullmatch(
        r"[+-]?[\d.,:;-]+(?:s)?|19\d{2}s?|20\d{2}s?", s.strip()))
