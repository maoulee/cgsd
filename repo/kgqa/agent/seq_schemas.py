"""SEQ contract: explicit public names, legacy entities internally."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator

class StrictArgs(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True, str_strip_whitespace=True)

class Subgraph(StrictArgs):
    id: str = ''
    anchor: str = ''
    facts: List[List[str]] = Field(min_length=1)
    covers: Dict[str, str] = Field(default_factory=dict)

    @field_validator('facts')
    @classmethod
    def triples(cls, facts):
        if any(len(f) != 3 or any(not str(x).strip() for x in f) for f in facts):
            raise ValueError('each fact must contain exactly three nonempty strings')
        return facts

class PlanArgs(StrictArgs):
    subgraphs: List[Subgraph] = Field(min_length=1)
    entities: List[str] = Field(min_length=1, validation_alias=AliasChoices('question_entities', 'entities'))
    answer: str = Field(min_length=2, pattern=r'^\?[A-Za-z_]\w*$')
    answer_type: str = Field(min_length=1)
    requirements: Optional[Dict[str, Any]] = None

class RetrieveRelationsArgs(StrictArgs):
    center: List[str] = Field(min_length=1)
    question: str = Field(min_length=1)
    sg: Optional[str] = None

class RetrieveSubgraphArgs(StrictArgs):
    center: List[str] = Field(min_length=1)
    relations: List[str] = Field(min_length=1)
    sg: Optional[str] = None

class AnswerArgs(StrictArgs):
    entities: List[str] = Field(validation_alias=AliasChoices('answer_values', 'entities'))  # required; [] means intentional abstention

def adaptive_enabled():
    return os.environ.get('SEQ_ADAPTIVE_ROUTES', '0').strip().lower() in ('1', 'true', 'on', 'yes')

class AdaptiveSubgraph(StrictArgs):
    id: str = ''
    anchor: str = ''
    facts: List[List[str]] = Field(default_factory=list)
    covers: Dict[str, str] = Field(default_factory=dict)

class AdaptivePlanArgs(StrictArgs):
    entities: List[str] = Field(min_length=1, validation_alias=AliasChoices('question_entities', 'entities'))
    subgraphs: List[AdaptiveSubgraph] = Field(default_factory=list)
    answer: str = '?answer'
    answer_type: str = ''
    requirements: Optional[Dict[str, Any]] = None

SCHEMAS = {'plan': AdaptivePlanArgs if adaptive_enabled() else PlanArgs,
    'decompose': AdaptivePlanArgs if adaptive_enabled() else PlanArgs,
    'retrieve_relations': RetrieveRelationsArgs,
    'retrieve_subgraph': RetrieveSubgraphArgs, 'answer': AnswerArgs}
PARAM_MAP = {
    'plan': {'question_entities': 'entities'},
    'decompose': {'question_entities': 'entities'},
    'answer': {'answer_values': 'entities', 'answer': 'entities'},
    'retrieve_relations': {'entities': 'center', 'entity': 'center', 'fact_id': 'sg', 'step': 'sg'},
    'retrieve_subgraph': {'entities': 'center', 'entity': 'center', 'fact_id': 'sg', 'step': 'sg'},
}

# Corrective-channel contract (user ruling 2026-10-03): a field-level error
# states the problem AND the complete format to resubmit — the model never
# has to guess the expected shape.
FORMAT_HINTS = {
    'plan': ('tool: plan\nquestion_entities: EntityA | EntityB\nanswer: ?variable\n'
             'answer_type: semantic_type\nsg1.anchor: EntityA\n'
             'sg1.f1: head | semantic sub-question | tail'),
    'decompose': ('tool: plan\nquestion_entities: EntityA | EntityB\nanswer: ?variable\n'
                  'answer_type: semantic_type\nsg1.anchor: EntityA\n'
                  'sg1.f1: head | semantic sub-question | tail'),
    'retrieve_relations': 'tool: retrieve_relations\ncenter: <anchor or delivered entity>\nquestion: <complete semantic sub-question>',
    'retrieve_subgraph': 'tool: retrieve_subgraph\ncenter: <anchor, delivered entity, CVT id or @frontier reference>\nrelations: relation.one | relation.two',
    'answer': 'tool: answer\nanswer_values: EntityA | EntityB',
}

def _with_format(problem: str, tool_name: str) -> str:
    hint = FORMAT_HINTS.get(tool_name)
    if not hint:
        return problem
    return f'{problem} Resubmit in the complete format:\n{hint}'

def validate_args(tool_name: str, args: Dict[str, Any]) -> Tuple[Optional[dict], Optional[str]]:
    model = (AdaptivePlanArgs if adaptive_enabled() else PlanArgs) if tool_name in ('plan', 'decompose') else SCHEMAS.get(tool_name)
    if model is None:
        return None, f'UNKNOWN_ACTION: {tool_name!r}; choose plan, retrieve_relations, retrieve_subgraph or answer.'
    if not isinstance(args, dict):
        return None, 'ACTION_FIELDS: arguments must be an object.'
    mapped = {}
    for key, value in args.items():
        key = str(key).strip().lower()
        key = PARAM_MAP.get(tool_name, {}).get(key, key)
        if key in ('entities', 'center', 'relations') and isinstance(value, str):
            value = [x.strip() for x in value.split('|') if x.strip()]
        if key in mapped and mapped[key] != value:
            return None, _with_format(f'ALIAS_CONFLICT: conflicting values for {key!r}; submit it once.', tool_name)
        mapped[key] = value
    try:
        value = model(**mapped).model_dump(exclude_none=True)
        for key in ('entities', 'center', 'relations'):
            if key in value:
                if any(not item.strip() for item in value[key]):
                    return None, _with_format(f'ACTION_FIELDS: {key!r} contains an empty value.', tool_name)
                value[key] = [item.strip() for item in value[key]]
        return value, None
    except ValidationError as exc:
        details = '; '.join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return None, _with_format(
            f'ACTION_FIELDS ({tool_name}): {details}. Repair this action; delivered evidence is retained.',
            tool_name)

SCHEMA_DOC = '''
plan: question_entities, answer (?variable), answer_type, subgraphs
retrieve_relations: center, question, sg (optional fact id)
retrieve_subgraph: center, relations, sg (optional fact id)
answer: answer_values (required list; [] is an explicit empty answer)
Legacy entities is accepted separately in plan and answer. Conflicting aliases
are rejected. Exactly one action per turn; analysis is not an action.
'''
if adaptive_enabled():
    SCHEMA_DOC = '''
plan: question_entities (required); answer_type and subgraphs are optional notes.
retrieve_relations: center, question; retrieve_subgraph: center, relations.
Centers can be canonical delivered entities or @frontier:<returned reference>.
answer: answer_values (required list; [] is intentional abstention).
No fact/checkpoint declarations are required. Legacy sg is accepted as metadata
and does not select a route or reset a resource budget.
'''
