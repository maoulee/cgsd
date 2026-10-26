"""Agent + skill prompt loader for agent-mode reasoning.

Reads kgqa/agent/AGENTS.md (the system prefix, loaded once) and the capability
docs under kgqa/agent/skills/*.md, and assembles (system, user) messages so the
strong model is guided by a short capability description + the single
load-bearing output tag — instead of a rigid rule-wall prompt.

The output tags are byte-identical to the existing parsers' expectations, so
agent mode needs NO parser changes:
  - reason_simple / reason_complex  -> <answer>\\boxed{...}</answer>
  - (decompose / prune / select skill builders provided for the full vision;
     Stage 8 reason is wired first as the highest-leverage swap.)
"""
from __future__ import annotations

import functools
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent
_SKILLS_DIR = _AGENT_DIR / "skills"


@functools.lru_cache(maxsize=1)
def agents_md() -> str:
    """The root AGENTS.md — the shared system prefix for every agent-mode turn."""
    return (_AGENT_DIR / "AGENTS.md").read_text()


def _skill_body(name: str) -> str:
    """Return a skill doc's body (everything after the YAML frontmatter)."""
    raw = (_SKILLS_DIR / f"{name}.md").read_text()
    if raw.startswith("---"):
        end = raw.find("---", 3)
        raw = raw[end + 3:]
    return raw.strip()


def assemble(skill_name: str, inputs_block: str) -> tuple[str, str]:
    """system = AGENTS.md; user = skill body + the per-turn inputs."""
    user = f"{_skill_body(skill_name)}\n\n─── TURN INPUTS ───\n{inputs_block}"
    return agents_md(), user


# ---------------------------------------------------------------------------
# Stage 8 reason builders (drop-in for the REASON_STYLE dispatch)
# ---------------------------------------------------------------------------

def build_agent_reason_prompt(cs, pattern_text, answer_type_hint, rewritten_hint):
    """Agent-mode Stage 8 prompt. Picks reason_simple vs reason_complex by
    cs.complexity. Returns (system, user) — note the reversed order vs the
    legacy _build_*_prompt helpers, which return (prompt, system). The Stage 8
    dispatch wraps this consistently (see stage8_reason.py agent branch)."""
    cand = ", ".join(cs.answer_candidates[:50]) if cs.answer_candidates else "No candidates"
    inputs = (
        f"QUESTION: {cs.question}\n"
        f"{answer_type_hint}{rewritten_hint}\n\n"
        f"GRAPH EVIDENCE:\n{pattern_text}\n\n"
        f"CANDIDATE ENTITIES:\n{cand}\n"
    )
    skill = "reason_simple" if getattr(cs, "complexity", "complex") == "simple" else "reason_complex"
    return assemble(skill, inputs)
