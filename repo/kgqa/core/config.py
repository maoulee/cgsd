"""Pipeline configuration constants and prompt templates.

Central location for all tunable parameters, file paths, and LLM prompts
used by the KGQA chain-decomposition pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# File-system paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root (kgqa/core/ → project/)

# ---------------------------------------------------------------------------
# LLM / GTE service endpoints
# ---------------------------------------------------------------------------
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:8000/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3.5-9B")
GTE_API_URL = os.environ.get("GTE_API_URL", "http://localhost:8003")
# 双卡 GTE 并行 (2026-10-20): 每卡一个实例, 逗号分隔多地址; 客户端轮询
# 分散负载, 避免单实例/单卡峰值。缺省回落单地址 (原行为)。
GTE_API_URLS = [u.strip() for u in os.environ.get(
    "GTE_API_URLS", GTE_API_URL).split(",") if u.strip()]
GTE_TASK_DESC = "Given a knowledge graph question, retrieve relevant graph relations that answer the question"

# ---------------------------------------------------------------------------
# Pipeline switches
# ---------------------------------------------------------------------------
REASON_STYLE = "v2"  # "default", "check", "v2", "entity-lite", etc.
SKIP_NER = False  # set by --skip-ner flag
ALLOW_1STEP = False  # set by --allow-1step flag: skip 1-step -> 2-step retry
CANDIDATE_THRESHOLD = 50  # max candidates before triggering re-prune / anchor swap
PRUNE_TOPK = 3            # max relations per step after LLM prune (local top-k from LLM ranking)


def set_reason_style(v: str):
    global REASON_STYLE
    REASON_STYLE = v


def set_skip_ner(v: bool):
    global SKIP_NER
    SKIP_NER = v


def set_allow_1step(v: bool):
    global ALLOW_1STEP
    ALLOW_1STEP = v

# ---------------------------------------------------------------------------
# Default I/O paths
# ---------------------------------------------------------------------------
DEFAULT_PILOT = ROOT / "reports/stage_pipeline_test/find_check_plan_pilot_10cases/results.json"
DEFAULT_CWQ = ROOT / "data/cwq_processed/test_v4_repaired.pkl"
MASK_WRONG_TYPE = ROOT / "data/cwq_processed/mask_wrong_type_ids.json"
DEFAULT_OUTPUT = ROOT / "reports/stage_pipeline_test/chain_decompose_test"

# ---------------------------------------------------------------------------
# Decomposition prompt template
# ---------------------------------------------------------------------------
DECOMP_PROMPT = '''Decompose the question into natural language sub-questions.

Break the original question into an ordered sequence of sub-questions. Each sub-question must be a complete, natural sentence that a human would ask.

For each sub-question, label its type:
- find: retrieves new information by following a relation in the knowledge graph.
- verify: checks a filter constraint on results already found (temporal: "before 1998", "most recent"; superlative: "largest", "biggest"; geographic: "bordering a specified place"; numeric: equals a value; intersection: must satisfy both A and B).

Rules:
1. START from the most specific named entity in the question — the one with the fewest possible neighbors. Prefer unique names (people, events, titles) over countries, regions, or groups.
2. Use 1 to 4 ordered steps. At most ONE verify step, placed at the end.
3. If the question is a straightforward chain of lookups with no filter constraint, all steps are find.
4. Each sub-question must be a complete natural language sentence, not a keyword phrase.
5. For each step, also provide a compact relation_query using domain nouns and verbs for retrieval.
6. Endpoint rule: only the LAST step may carry an endpoint with a fixed entity explicitly from the question. Otherwise use none.
7. Never output placeholder endpoints such as "[Country Name]", "team name", or bracketed templates.
8. Do not enumerate or name entities not present in the question.
9. Do not output chain-of-thought, hidden reasoning, explanations, examples, or alternative plans.

Output format:
Anchor: [entity name] (entity_query: [search term for entity retrieval])
Answer_type: [free-form noun phrase describing what the answer IS, e.g. person, country, government_type, language, sport, event, year, monetary_value, percentage, etc.]
1. "Who was the Governor of Arizona in 2009?" (type: find; relation_query: governor of state; endpoint: none)
2. "Did that governor hold a governmental position before 1998?" (type: verify; relation_query: tenure start date; endpoint: none)

Return only the decomposition in the exact format above.
'''
