"""CaseState dataclass — per-case intermediate state for the KGQA pipeline.

Holds all fields that flow between pipeline stages (NER, decomposition,
entity resolution, GTE retrieval, pruning, traversal, answer reasoning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CaseState:
    """Holds all intermediate state for one case between pipeline stages."""
    # Identity
    case_id: str
    case_num: int
    sample: Dict[str, Any]
    pilot_row: Dict[str, Any]

    # Input data (set at initialization)
    question: str = ""
    gt_answers: List[str] = field(default_factory=list)
    ents: List[str] = field(default_factory=list)
    rels: List[str] = field(default_factory=list)
    h_ids: List[int] = field(default_factory=list)
    r_ids: List[int] = field(default_factory=list)
    t_ids: List[int] = field(default_factory=list)
    rel_texts: List[str] = field(default_factory=list)
    ent_candidates: List[str] = field(default_factory=list)

    # Stage 0: NER
    ner_scored: List[Dict] = field(default_factory=list)
    ner_top_ents: List[Tuple[str, float]] = field(default_factory=list)
    ner_name_to_ids_expanded: Dict[str, List[int]] = field(default_factory=dict)

    # Stage 1: Decomposition
    decomp_raw: Optional[str] = None
    decomp_question: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    answer_type: Optional[str] = None
    use_ner: bool = True
    anchor_forbidden: Optional[str] = None

    # Stage 1.5: Decomposition reflection
    decomp_retry: bool = False
    decomp_reflect_raw: Optional[str] = None
    decomp_retry_reason: Optional[str] = None

    # Stage 2: Entity resolution
    anchor_idx: Optional[int] = None
    anchor_name: Optional[str] = None
    breakpoints: Dict[int, int] = field(default_factory=dict)
    entity_retrieval_details: List[Dict] = field(default_factory=list)

    # Stage 3: GTE relation retrieval
    step_candidates: Dict[int, List[Tuple[int, str, float]]] = field(default_factory=dict)
    gte_per_step: Dict[int, Dict[int, Tuple[str, float]]] = field(default_factory=dict)
    relation_retrieval_details: List[Dict] = field(default_factory=list)

    # Stage 4: Relation pruning
    step_relations: List[set] = field(default_factory=list)
    prune_debug: Dict = field(default_factory=dict)

    # Stage 5: Graph traversal
    paths: List[Dict] = field(default_factory=list)
    max_depth: int = 0
    max_cov: int = 0
    answer_candidates: List[str] = field(default_factory=list)
    gt_hit: bool = False
    gt_hit_strict: bool = False
    gt_f1: float = 0.0
    all_subgraph_nodes: set = field(default_factory=set)

    # Stage 5/6: Frontier traversal + safety net
    layer_diagnostics: List[Dict] = field(default_factory=list)
    planning_attempts: List[Dict] = field(default_factory=list)
    needs_direct_answer: bool = False

    # Stage 7: Path selection
    logical_paths: List[Dict] = field(default_factory=list)
    selected_paths: List[int] = field(default_factory=list)
    attempt_log: List[Dict] = field(default_factory=list)

    # Stage 8: Answer reasoning
    llm_answer: Optional[str] = None
    llm_hit: bool = False
    llm_f1: float = 0.0
    llm_precision: float = 0.0
    llm_recall: float = 0.0
    llm_reasoning_prompt: Optional[str] = None
    llm_reasoning_full: Optional[str] = None
    num_triples: int = 0

    # Control
    active: bool = True
    error: Optional[str] = None
    stage_times: Dict[str, float] = field(default_factory=dict)
    complexity: str = "complex"  # "simple" | "complex" (set by adaptive router)

    # Stage 1a: Entity analysis output
    _1a_prompt: Optional[str] = None  # actual input prompt sent for 1a
    _1a_raw: Optional[str] = None
    _1a_anchor: Optional[str] = None
    _1a_endpoints: Optional[str] = None
    _1a_required_properties: Optional[str] = None  # V2: entities constraining the answer
    _1a_constraints: Optional[str] = None  # V2: ordered hop descriptions from 1a
    _1a_answer_type: Optional[str] = None
    _1a_rewritten: Optional[str] = None
    _1a_interpretation: Optional[str] = None
    decomp_prompt_formatted: Optional[str] = None  # actual formatted prompt sent to LLM
    decomp_method: str = ""  # "chain", "triple", or "cascade"

    # Cascade decomposition (two-step: sub-questions → triples)
    triples: List[Dict[str, Any]] = field(default_factory=list)
    entity_roles: List[Dict[str, Any]] = field(default_factory=list)
    answer_variable: str = ""
    sub_questions: List[str] = field(default_factory=list)

    # Internal temp fields for cross-stage data
    _pending_endpoints: List[Dict] = field(default_factory=list)
    _pending_anchor_eq: Optional[str] = None
    _pending_anchor_eq_name: Optional[str] = None
    _prev_attempt_score: tuple = (-1, -1, -1)
