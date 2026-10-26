#!/usr/bin/env python3
"""Render a react_case result JSON into a human-readable trajectory TXT.

Reads /tmp/case1_raw.json (or a path arg) and writes a formatted trajectory
showing: system prompt, user messages, assistant reasoning + content, tool
results — all with clear visual separators.

Usage: python scripts/render_trajectory.py [input.json] [output.txt]
"""
import json, sys, textwrap
from pathlib import Path

def render(result_json_path, output_path):
    r = json.loads(Path(result_json_path).read_text())
    traj = r.get("agent_trajectory", [])
    lines = []

    # Header
    q = r.get("question", "")
    gt = r.get("gt_answers", [])
    pred = r.get("llm_answer", "")
    f1 = r.get("llm_f1", 0)
    hit = "✓" if r.get("llm_hit") else "✗"
    lines.append("=" * 78)
    lines.append(f"CASE {r.get('case_num')}  [{hit}]  F1={f1:.2f}")
    lines.append("=" * 78)
    lines.append(f"Question: {q}")
    lines.append(f"Gold Answer: {gt}")
    lines.append(f"Model Answer: {pred}")
    anchor = r.get("anchor_name", "")
    if anchor:
        lines.append(f"Anchor Entity: {anchor}")
    if r.get("agent_failed"):
        lines.append(f"FAILED: {r.get('agent_failure_reason','')}")
    lines.append("")

    # System prompt note (the AGENTS.md is loaded at runtime; we note it)
    lines.append("─" * 78)
    lines.append("[SYSTEM PROMPT]  (kgqa/agent/AGENTS.md — the full tool spec & rules)")
    lines.append("─" * 78)
    lines.append("")

    step_num = 0
    for step in traj:
        role = step.get("role", "?")
        name = step.get("name", "")

        if role == "assistant":
            step_num += 1
            content = step.get("content", "")
            reasoning = step.get("reasoning", "")

            lines.append("╔" + "═" * 76 + "╗")
            lines.append(f"║  TURN {step_num} — MODEL OUTPUT (assistant)")
            lines.append("╚" + "═" * 76 + "╝")
            lines.append("")

            # Reasoning (the <think> chain)
            if reasoning:
                lines.append("  ┌─ THINKING (reasoning field, model's <think> chain) " + "─" * 22)
                for rl in reasoning.split("\n"):
                    wrapped = textwrap.wrap(rl, width=72) if rl.strip() else [""]
                    for w in wrapped:
                        lines.append(f"  │ {w}")
                lines.append("  └" + "─" * 74)
                lines.append("")

            # Content (the tool call / answer text)
            lines.append("  ┌─ CONTENT (what the runtime parses for the tool call) " + "─" * 14)
            for cl in content.split("\n"):
                wrapped = textwrap.wrap(cl, width=72) if cl.strip() else [""]
                for w in wrapped:
                    lines.append(f"  │ {w}")
            lines.append("  └" + "─" * 74)
            lines.append("")

        elif role == "tool":
            content = step.get("content", "")
            tag = f"TOOL RESULT ({name})" if name else "TOOL RESULT"
            lines.append("  ▼ " + tag)
            lines.append("  " + "." * 74)
            # Try to pretty-print JSON, else show raw
            try:
                obj = json.loads(content)
                pretty = json.dumps(obj, ensure_ascii=False, indent=2)
                # Truncate very long tool results
                pretty_lines = pretty.split("\n")
                if len(pretty_lines) > 60:
                    shown = pretty_lines[:55]
                    shown.append(f"  ... ({len(pretty_lines) - 55} more lines truncated)")
                else:
                    shown = pretty_lines
                for pl in shown:
                    wrapped = textwrap.wrap(pl, width=72) if pl.strip() else [""]
                    for w in wrapped:
                        lines.append(f"  │ {w}")
            except Exception:
                for cl in content.split("\n"):
                    wrapped = textwrap.wrap(cl, width=72) if cl.strip() else [""]
                    for w in wrapped:
                        lines.append(f"  │ {w}")
            lines.append("  " + "─" * 74)
            lines.append("")

        elif role == "user":
            # Non-tool user messages (nudges, rejections, hints)
            content = step.get("content", "")
            if content and not content.startswith("Tool result"):
                lines.append("  ➜ SYSTEM NUDGE (user):")
                for cl in content.split("\n")[:3]:
                    lines.append(f"    {cl}")
                lines.append("")

    # Footer
    lines.append("=" * 78)
    lines.append(f"END OF TRAJECTORY — {step_num} turns, {len(traj)} total steps")
    lines.append("=" * 78)

    output = "\n".join(lines)
    Path(output_path).write_text(output)
    print(f"Wrote {output_path} ({len(output)} chars, {len(lines)} lines)")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "/tmp/case1_raw.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/case1_trajectory.txt"
    render(inp, out)
