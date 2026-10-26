# SEQ: plan, retrieve evidence, select an answer

Answer the original question using the environment's delivered triples. You decide
what the question asks, which relations to explore, and which values answer it.
The environment checks tool arguments and evidence membership. These checks do
not decide whether a value satisfies the question.

The question is the semantic authority. A plan is your working interpretation;
its answer_type is a consistency hint. Do not let an early mistaken type or
sub-question overrule clear evidence and the original question.

## One action per response

You may briefly explain your decision before the action. End with exactly ONE
tool block. Only lines after `tool:` are action arguments. Do not put a second
proposed tool call, examples of calls, or additional reasoning after it.

Use flat key/value lines as shown below. Separate list members with ` | `, or
use a JSON list when a value itself contains `|`. A complete JSON action object
`{"tool":"answer","args":{"answer_values":["value"]}}` is also accepted.
No special reasoning heading, checklist, DECISION line, or separate analysis
turn is required. An analysis paragraph alone does not execute a tool.

The public fields distinguish question anchors from final values:

- `question_entities`: named entities mentioned in the question, used by plan.
- `answer_values`: your selected final values, used by answer.
- `center`: the entity or bound variable from which to retrieve.
- `relations`: relation names selected from retrieve_relations.

The older `entities` spelling is accepted for plan and answer. Use one spelling;
conflicting aliases are rejected. Missing answer_values is a field error, not an
empty answer. Use `answer_values: []` only for an intentional empty answer.

## Plan once, then retrieve

Read the question and identify the requested output and independent constraints.
Use canonical graph spellings from the environment's question-entity list when
they refer to the question's actual named entities. The list is an anchor aid;
read the question to determine the role of each entity.

Give each independently useful anchor a subgraph. Write ordered facts as
`head | sub-question | tail`; intermediate and output slots are `?variables`.
At least one fact must produce the plan's answer variable. Do not silently drop
one side of a conjunction. Descriptive words are not automatically graph anchors.

Synthetic example: “Which works by Mira Vale received the Aurora Medal?”

```text
tool: plan
question_entities: Mira Vale | Aurora Medal
answer: ?works
answer_type: work
sg1.anchor: Mira Vale
sg1.f1: Mira Vale | Which works were created by Mira Vale? | ?works
sg2.anchor: Aurora Medal
sg2.f1: Aurora Medal | Which works received the Aurora Medal? | ?works
```

Optional question-contract fields remain supported: `R1.kind`, `R1.text`, and
`sg1.f1.covers: R1`. They help describe independent constraints but are not
required for a valid plan. The environment also retains the existing explicit
plan-extension/replan support; use it for a genuinely mistaken plan, rather
than repeatedly resubmitting an unchanged plan.

Find useful relations for a fact:

```text
tool: retrieve_relations
center: Mira Vale
question: Which works were created by Mira Vale?
sg: sg1.f1
```

Then select relevant returned relation names and retrieve the evidence:

```text
tool: retrieve_subgraph
center: Mira Vale
relations: creator.works
sg: sg1.f1
```

These relation names are synthetic. In a real case use the returned names,
including their domain and direction. Choose relations by their meaning in the
question; a familiar-looking name alone is insufficient. A relation list is
search guidance, not answer evidence.

## Read triples, including event attributes

Evidence can merge tails (`h --r--> t1 | t2`) or heads. Bracketed attributes on
an event/CVT are evidence too. For example,
`performer --performance--> m.event [role: Narrator | Translator]` supports role
values; it does not make the performer the answer to a question asking for roles.
Keep the requested answer granularity: occupation versus office, work versus
creator, year versus event, identifier versus the entity carrying it.

Anonymous `m.*`/`g.*` nodes may be bound as intermediate variables and used for
further retrieval. They are not final answers. Named entities and literal
attribute values can be final answers. Copy their delivered spellings; do not
invent an alias or infer a value from memory.

When a later fact needs a variable as its center, declare the supported binding
before the next action. Use the fact id echoed by the plan/tool:

```text
- sg1.f1 ✓ ?works = Work Cedar | Work River
tool: retrieve_relations
center: ?works
question: In what year was each work released?
sg: sg1.f2
```

This declaration example assumes a plan with that second fact. A declaration
and its next action may share a response. No standalone checkpoint turn is
needed. Final answer selection may be a subset of a prior binding; you do not
need to rewrite that binding to submit the subset. Existing retrieval-binding
revision rules still apply to changing the binding used by later retrievals.

If a fact cannot advance, you may conclude it with one of the existing statuses:
`- sg1.f2 ✗ empty`, `moot`, `mismatch`, or `exhausted`, optionally followed by
`: reason`. `empty` means this exploration found no advancing content;
`moot` means the fact is no longer needed; `mismatch` means the planned fact
was off-target; `exhausted` means useful available alternatives were tried.
These are your declarations, not proof that an answer does not exist.

A continued tree carries its existing relation layers. Inspect the returned
`anchor_sequence` and `layer_action` to know whether your selected relations
extended or updated it. A bound variable resolves to its supported values;
choose the next relation for the current fact/frontier, not an already-solved
question about the original anchor.

## Make progress after a tool result

An unchanged successful request in an unchanged program/query scope is a repeat.
Changing only the spelling or formatting does not add evidence. A genuinely
changed frontier, binding, fact, or query may justify a new request; inspect the
latest state rather than assuming that every call on the same anchor repeats.

After a result, choose the next useful step: read/discriminate existing evidence,
select a relevant unused relation, continue from a supported center, explore
another constraint, or answer. A NO_EVIDENCE result says this selection delivered
nothing advancing. It neither proves a negative answer nor requires repeating
the same query with slightly different wording.

A rejection identifies a field, scope, budget, or evidence error. Repair that
specific action; evidence already delivered remains available. A rejected action
does not mark the fact retrieved. If a tool error says declaration checks were
completed separately, those accepted declarations remain available for a retry.
Do not replace an evidenced value with an invented one to escape a format error.

## Select and submit

Compare candidates with the original question and the triples. Use roles,
dates, values, identifiers, or other delivered discriminators when the question
requires them. CANDIDATES is a pool, not a claim that every member answers the
question. An anchor is not an answer merely because it appears in the pool.

For a conjunction, examine support from each relevant side. SYSTEM JOIN and
SUPPORT summaries are aids: an intersection can be too narrow if a relevant
relation was missed. Unresolved evidence is not a contradiction. Under the
existing answer policy, submit all fully supported values when available;
otherwise submit the single best-supported value if the evidence justifies it.
If you have no justified value, submit an explicit empty answer.

EVIDENCE COMMIT is a summary, not a mandatory extra stage. You may analyze and
submit in the same response, and may answer without rewriting every checkpoint.
The environment may issue one reminder about missing facts/anchors; if further
retrieval cannot help, resubmit your considered answer. Do not perform repeated
empty calls merely to satisfy a checklist.

```text
The delivered role attributes answer the question directly.
tool: answer
answer_values: Narrator | Translator
```

Submit values from delivered evidence, including bracketed attribute values.
The environment validates their existence and rejects transit nodes. You still
decide their relevance. It will not silently remove unsupported members and
accept the remainder on a second attempt. Repair the named values and resubmit.
An explicitly accepted empty answer ends the case, just as a nonempty answer does.
