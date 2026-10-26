"""Prompt templates and prompt-building functions for the KGQA pipeline.

All LLM prompt constants and dynamic prompt construction functions are
centralised here so that stage modules can import them without depending
on the monolith script.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Stage 0 — Decomposition prompt (direct mode, no NER)
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


# ---------------------------------------------------------------------------
# Stage 1a — Entity analysis + question interpretation + rewrite
# ---------------------------------------------------------------------------
ENTITY_ANALYSIS_PROMPT = """Entities from knowledge graph: {entities}

Question: {question}

Task: Analyze the given entities, interpret the question, and rewrite it as a declarative sentence.

IMPORTANT: You may ONLY use entities from the list above as anchor or endpoints.
Do NOT infer, guess, or invent entities not in the list. If only one entity is given, it is the anchor.
The answer is usually NOT one of the listed entities. Listed non-anchor entities are constraints unless the
question explicitly asks for that exact entity.

## Entity Analysis
For each entity, assess its suitability as a traversal starting point (anchor).
Consider: how many possible graph neighbors does this entity have? Lower ambiguity = better anchor.
The entity list contains exact names from entity recognition — trust them as-is, do NOT judge
their validity. Even unusual names are valid entities.

Rate each entity: HIGH / MEDIUM / LOW ambiguity as starting point.
- Specific names (people, events, unique titles) → LOW ambiguity → good anchor
- Generic types (Country, Person, College/University) → HIGH ambiguity → bad anchor
- Active subject of the sentence → better anchor than entities in prepositional phrases

## Anchor Selection
Pick the entity with LOWEST ambiguity as anchor.
Rule: the ACTIVE subject of the sentence is preferred over locative constraints.
("Who founded X" → X is anchor; "in Y" → Y is endpoint)
Rule: When choosing between a specific named entity (event, unique title, person) and a
generic entity (country, place, sport), ALWAYS prefer the specific named entity — it has
fewer graph neighbors and produces more focused traversal.
("2010 FIFA World Cup" > "Spain"; "I Am... World Tour" > "Beyoncé")

## Endpoint Selection (optional)
From the REMAINING entities in the list, pick those that CONSTRAINT the answer path:
- Locative constraints: "in [Place]", "near [Location]"
- Co-participants: entities that shape the answer alongside the anchor
Skip generic type words. If none qualify or only one entity exists, output "none".

## Question Interpretation
Before rewriting, explain what the question is actually asking in your own words.
Identify: What is the answer's role? (person who did X, place where Y happened, year when Z occurred, etc.)
This interpretation guides the answer type and rewrite.
Use the wh-phrase to determine Answer_type:
- "Which country..." → country
- "Which region/globe region..." → region
- "Who..." → person
- "Where..." → location/place
Do NOT set Answer_type to an endpoint constraint such as a time zone, country, person, or date mentioned after
"with", "in", "near", "bordering", "during", or similar constraint phrases.

## Question Rewrite
Based on your interpretation, rewrite the question as a declarative sentence.
The rewrite must preserve all constraints and clearly express what entity/value is sought.
Start with "The [answer type] that..." or "The [answer type] where/when/which...".
The rewrite must contain a placeholder for the sought answer, e.g. "[Answer]".
Do NOT rewrite the endpoint constraint as the answer.

## Output format
Wrap each field in XML tags. Example outputs:

Example 1:
Entities from knowledge graph: [time zone] | [country]
Question: Which region with a specified time zone contains a given country?

<reasoning>time zone is generic with MEDIUM ambiguity. country is generic with HIGH ambiguity. Both are constraint entities; the answer is the region containing them.</reasoning>
<anchor>[country]</anchor>
<endpoints>[time zone]</endpoints>
<interpretation>The question asks for the region that contains the given country and has the specified time zone.</interpretation>
<answer_type>region</answer_type>
<rewritten>The region [Answer] that contains the given country and has the specified time zone.</rewritten>

Example 2:
Entities from knowledge graph: [city] | [border country]
Question: What country bordering a given country contains an airport that serves a given city?

<reasoning>city is a specific location with MEDIUM ambiguity. border country is a constraint entity. The answer is a country bordering the given country that has an airport for the given city.</reasoning>
<anchor>[city]</anchor>
<endpoints>[border country]</endpoints>
<interpretation>The question asks for a country that borders the given country and contains an airport serving the given city.</interpretation>
<answer_type>country</answer_type>
<rewritten>The country [Answer] that borders the given country and contains an airport serving the given city.</rewritten>

Now analyze the given question and entities. Output:

<reasoning>[one sentence per entity analyzing ambiguity]</reasoning>
<anchor>[chosen entity from the list only]</anchor>
<endpoints>[Entity from the list only, or "none"]</endpoints>
<interpretation>[1-2 sentences explaining what the question asks, what role the answer plays]</interpretation>
<answer_type>[free-form noun phrase: person, country, government_type, language, sport, event, year, religion, college, movie, sports team, location, currency, etc.]</answer_type>
<rewritten>[declarative sentence based on interpretation]</rewritten>"""


# ---------------------------------------------------------------------------
# Stage 1b — Chain decomposition (given anchor, endpoints, rewritten question)
# ---------------------------------------------------------------------------
CHAIN_PROMPT = """Entities: {entities}
Anchor: {anchor}
Endpoints: {endpoints}
Answer type: {answer_type}
Interpretation: {interpretation}
Rewritten question: {rewritten}
Original question: {question}

Decompose this question into a chain of abstract relation hops.
Each hop describes WHAT kind of graph edge to look for — NOT which specific entity it leads to.
Think first, then write the chain. Do NOT write the chain before doing the per-hop analysis.

IMPORTANT RULES:
1. This is QUESTION DECOMPOSITION, not graph traversal. You do NOT know what entities exist in the graph.
2. Intermediate nodes MUST be written as literally "node". NEVER fill in guessed entity names.
3. Each hop = ONE atomic relation. Compound actions MUST be split into separate hops.
4. No verification hops for superlatives ("find the most recent" → post-processing, not a hop).
5. No circular chains.
6. Endpoints are entities that must be reached or constrain the path.
7. Count hops carefully: each "that/which/who/where" clause typically adds ONE hop. Do NOT merge clauses.
8. "Return the answer", "the result is", "verify it", or "implicit return" is NOT a hop.
9. A constraint entity should appear only at the hop where it is reached or checked.
10. For "X that/which contains/has an airport that serves Y", start from Y with:
    Y -(served by airport)-> node -(airport located in country/region)-> node.
11. For "country/region with [endpoint constraint]", first find the answer node, then add one hop from that
    answer node to the endpoint constraint. Do NOT make the endpoint itself the answer.

Examples:
  Question: "What is the capital of the country where the Eiffel Tower is located?"
  Entities: Eiffel Tower
  Anchor: Eiffel Tower
  Chain: Eiffel Tower -(located in country)-> node -(capital city)-> node  [2 hops]

  Question: "What language is spoken in the country where the leader was appointed to office?"
  Entities: Leader
  Anchor: Leader
  Chain: Leader -(government position held)-> node -(jurisdiction of office)-> node -(language spoken)-> node  [3 hops]

  Question: "What sport does the most popular team in the country containing Paris play?"
  Entities: Paris
  Anchor: Paris
  Chain: Paris -(contained by country)-> node -(popular sport)-> node  [2 hops, "most popular" is post-processing]

  Question: "What country bordering a given country contains an airport that serves a given city?"
  Entities: [city] | [border country]
  Anchor: [city]
  Endpoints: [border country]
  Chain: [city] -(served by airport)-> node -(airport located in country)-> node -(borders country)-> node  [3 hops; endpoint is reached only by the border hop]

  Question: "Which region with a specified time zone contains a given country?"
  Entities: [country] | [time zone]
  Anchor: [country]
  Endpoints: [time zone]
  Chain: [country] -(contained in region)-> node -(has time zone)-> node  [2 hops; answer is the region node, endpoint is the time zone constraint]

WRONG Chain: Leader -(appointed to role in country where language is spoken)-> node  ← merged 3 hops into 1!
WRONG Chain: [city] -(located in country)-> node -(borders endpoint)-> node -(located in country)-> node  ← final "return" hop is fake and creates a loop.
WRONG Chain: [country] -(has time zone)-> node  ← answers the endpoint constraint, not the requested region.

Output (exact format — think first, chain last):
Reasoning: [1-3 short sentences. Identify the answer node and which endpoint constraints must be reached.]

Analysis:
1. [action description]
   - Keyword: core concept word
   - Definition: A concise dictionary-style definition of the relation concept in 15 words or less. Describe what the relation MEANS naturally. Do NOT use graph terminology (no "link", "node", "entity", "edge"). Examples: "neighboring countries sharing a common border", "the sport an athlete plays professionally", "a country's top-level administrative regions", "the official song representing a nation's identity".
   - Sub-question: a complete natural language question for this specific hop, as if asking a person (e.g., "What country contains this department?", "Who is the governor of this state?", "What sport does this team play?").

Chain:
{anchor} -(action description)-> node ... -(final action)-> node"""

# System prompt strings (used alongside the templates above)
# ---------------------------------------------------------------------------
SYSTEM_ENTITY_SELECT = "Select the correct entity from candidates. Output <analysis> and <selected> XML tags."
SYSTEM_RELATION_RESELECT = "You reselect better knowledge graph relations for a single failed reasoning step. Output <analysis> and <selected>."
SYSTEM_PATH_SELECT_RECALL = (
    "You analyze and select reasoning paths for multi-step QA. Your goal is high-recall semantic path selection. "
    "Judge semantic fit of relation chains to the question. Keep all semantically plausible paths. "
    "Output exactly three XML tags: <analysis>, <selected>, and <need_more>."
)
SYSTEM_RELATION_SELECTOR = (
    "You are a knowledge graph relation selector for multi-step QA. "
    "Analyze the full chain, then select relevant relations per step. "
    "Output <analysis> and <selected> XML tags."
)
SYSTEM_PRUNE_ALL = (
    "You are a knowledge graph relation selector for multi-step QA. "
    "Analyze the full chain, then select relevant relations per step. "
    "Output <analysis> and <selected> XML tags."
)
SYSTEM_PATH_SELECT_DIVerse = (
    "You select diverse reasoning paths for graph QA. "
    "Always select 2-4 paths with different relation types. "
    "Output <analysis> and <selected>."
)
SYSTEM_ENTITY_LITE = (
    "You are a precise graph QA assistant. Always output at least one entity from CANDIDATE ENTITIES. "
    "Never output None. Copy entity strings exactly."
)
SYSTEM_CHECK = (
    "You are a graph fact-checker. Fill the checklist mechanically. One word per judgement "
    "(KEEP/REMOVE, PASS/FAIL). No paragraphs. Always output an answer. "
    "Copy entity strings exactly from CANDIDATE ENTITIES."
)
SYSTEM_ECOT = (
    "You are a precise graph QA checker. For each candidate, give ONE fact from graph evidence. "
    "Keep total under 10 lines. Answer MUST be an exact string from the candidate list. No outside knowledge."
)
SYSTEM_ENTITY = (
    "You are a precise graph QA system using per-candidate constraint verification. "
    "First identify answer type and constraints, then check EACH candidate against type + explicit + "
    "implicit constraints, then output ALL passing candidates. Any entity in GRAPH EVIDENCE is valid, "
    "not just CANDIDATE ENTITIES. NEVER decide answer count before checking all candidates. "
    "Over-output is better than discarding."
)
SYSTEM_V2 = (
    "You are a precise graph QA system. Step 1: list ALL candidates matching the answer type. "
    "Step 2: filter by constraints, output ALL remaining. Always answer. Copy entity strings exactly. "
    "Over-output is better than discarding."
)
SYSTEM_DIRECT_ANSWER = (
    "You are a precise QA system over Freebase (circa 2015). Answer the question directly using your knowledge. "
    "Output <reasoning> and <answer> XML tags."
)
SYSTEM_YEAR_RETRY = (
    "Your previous answer was NOT a valid entity from the graph. You MUST pick from the candidate list. "
    "Use exact entity strings only — never raw years or timestamps."
)
SYSTEM_PATH_RESELECT_RETRY = (
    "You are a precise graph QA system. Previous paths had insufficient evidence. "
    "These are NEW alternative paths. Find the answer from the graph evidence. ALWAYS answer. "
    "Exact graph strings only."
)
SYSTEM_V1_REASON = (
    "You are a precise graph QA system. Follow the 4-step reasoning process. ALWAYS answer. "
    "Compare patterns before choosing. First identify answer type and ALL constraints (explicit + implicit), "
    "then filter candidates by type match and constraint check. Apply implicit constraint heuristics: "
    "unique role → most recent; events/achievements → all; attributes → all; group membership → all. "
    "Singular/plural alone does NOT determine answer count. NEVER skip constraint verification. "
    "Exact graph strings only."
)


# ---------------------------------------------------------------------------
# Prompt-building functions
# ---------------------------------------------------------------------------

def _build_prune_all_prompt(question, all_steps, step_candidates):
    """Build the prompt for relation pruning. Returns (prompt_text, system_text)."""
    chain_lines = []
    step_blocks = []
    for s in all_steps:
        sn = s["step"]
        ep_str = f" -> endpoint: {s['endpoint']}" if s.get("endpoint") else ""
        chain_lines.append(f"  Step {sn}: {s['question']}{ep_str}")

        cands = step_candidates.get(sn, [])
        if not cands:
            step_blocks.append(f"Step {sn}: {s['question']}\n  Purpose: {s.get('definition', '')}\n  Candidates: (none)")
            continue

        cand_lines = [f"    {i}. {name}"
                      for i, (idx, name, score) in enumerate(cands, 1)]
        step_blocks.append(
            f"Step {sn}: {s['question']}\n"
            f"  Purpose: {s.get('definition', '')}\n"
            f"  Candidates:\n" + "\n".join(cand_lines)
        )

    chain_text = "\n".join(chain_lines)
    blocks_text = "\n\n".join(step_blocks)

    prompt = f"""Analyze and select knowledge graph relations for each step of this reasoning chain.

Question: {question}

Reasoning chain:
{chain_text}

Step-by-step candidates:
{blocks_text}

Rules:
1. Each step connects FROM previous output TO next — select bridge relations
2. Select 2-4 relevant relations per step. Pick the best candidates that match the step's purpose.
3. When uncertain about a relation's relevance, INCLUDE it — missing a key relation is far worse
   than having an extra irrelevant one.
4. Ignore unrelated attributes (currency, codes when asking about geography)
5. If no relations fit a step, output empty list
6. ORDER matters: rank by relevance to the step (most relevant first)

Output format:
<analysis>
One sentence per step: what it needs and which relations fit.
</analysis>
<selected>
step_1: [1, 3]
step_2: [2, 5]
</selected>"""

    system = "You are a knowledge graph relation selector for multi-step QA. Analyze the full chain, then select relevant relations per step. Output <analysis> and <selected> XML tags."
    return prompt, system


def build_reselect_prompt(question, step, cands, current_indices):
    """Build prompt for reselecting a better relation for a failed step.

    Returns (prompt_text, system_text).
    """
    cand_lines = []
    current_pos = set()
    for i, (idx, name, score) in enumerate(cands):
        marker = " [CURRENT]" if idx in current_indices else ""
        if idx in current_indices:
            current_pos.add(i)
        cand_lines.append(f"  {i}. {name}{marker}")

    prompt = f"""The current relation choice for one reasoning step appears to be wrong or too noisy.

Question: {question}
Failed step: {step['question']}
Step purpose: {step.get('definition', '')}

Candidate relations:
{chr(10).join(cand_lines)}

Select 1 to 3 BETTER alternative relations for this step.

Rules:
- Prefer relations that directly express the step semantics.
- Avoid the currently marked failed choices if better alternatives exist.
- Do not select generic or weakly related relations just because they are broad.

Output format:
<analysis>One short sentence.</analysis>
<selected>comma-separated candidate numbers only</selected>"""

    system = SYSTEM_RELATION_RESELECT
    return prompt, system


def build_entity_disambig_prompt(query, candidates_with_ctx):
    """Build prompt for entity disambiguation from candidates.

    Args:
        query: search query string.
        candidates_with_ctx: list of (name, context_str) tuples.

    Returns (prompt_text, system_text).
    """
    cand_lines = []
    for i, (name, ctx) in enumerate(candidates_with_ctx, 1):
        cand_lines.append(f"  {i}. {name} [{ctx}]" if ctx else f"  {i}. {name}")

    prompt = f"""Search query: {query}

Candidate entities (with relation context from knowledge graph):
{chr(10).join(cand_lines)}

Which candidate best matches the search query? Use the relation context to identify what each entity actually IS (a person, a location, a schema type, etc). Pick the specific entity, not generic types or schema entries.

<analysis>Brief reasoning about which candidate matches the query</analysis>
<selected>entity name</selected>"""

    system = SYSTEM_ENTITY_SELECT
    return prompt, system


def build_stage4_prune_prompt(question, all_steps, step_candidates):
    """Build prompt for Stage 4 relation pruning (simplified variant).

    Returns (prompt_text, system_text).
    """
    chain_lines = []
    step_blocks = []
    for s in all_steps:
        sn = s["step"]
        ep_str = f" -> endpoint: {s['endpoint']}" if s.get("endpoint") else ""
        chain_lines.append(f"  Step {sn}: {s['question']}{ep_str}")
        cands = step_candidates.get(sn, [])
        if not cands:
            step_blocks.append(f"Step {sn}: {s['question']}\n  Purpose: {s.get('definition', '')}\n  Candidates: (none)")
            continue
        cand_lines = [f"    {i}. {name}" for i, (idx, name, score) in enumerate(cands, 1)]
        step_blocks.append(
            f"Step {sn}: {s['question']}\n  Purpose: {s.get('definition', '')}\n  Candidates:\n" + "\n".join(cand_lines))
    chain_text = "\n".join(chain_lines)
    blocks_text = "\n\n".join(step_blocks)
    prompt = f"""Analyze and select knowledge graph relations for each step of this reasoning chain.

Question: {question}

Reasoning chain:
{chain_text}

Step-by-step candidates:
{blocks_text}

Rules:
1. Each step connects FROM previous output TO next — select bridge relations
2. Select 2-4 relevant relations per step. Pick the best candidates that match the step's purpose.
3. Ignore unrelated attributes
4. If no relations fit a step, output empty list
5. ORDER matters: rank by relevance to the step (most relevant first)

Output format:
<analysis>
One sentence per step: what it needs and which relations fit.
</analysis>
<selected>
step_1: [3, 1]
step_2: [5, 2]
</selected>

Numbers are RANKED: first = most relevant."""
    system = SYSTEM_RELATION_SELECTOR
    return prompt, system


def build_path_select_prompt_recall(question, paths_text, idx_map_remaining):
    """Build prompt for path selection with recall-oriented heuristics (run_single_case).

    Returns (prompt_text, system_text).
    """
    select_prompt = f"""Analyze and select reasoning paths for this question.

Question: {question}

Paths:
{paths_text}

Instructions:
1. Ignore hidden entity identities. Judge each path only by its relation sequence and node structure.
2. Select paths by semantic relevance, not by shortest length.
3. Keep a path if it preserves the intended multi-step meaning of the question, even if it is longer, includes bridge nodes, or is not the most direct-looking path.
4. Do not eliminate a path only because it is longer, slightly noisy, or contains extra intermediate structure.
5. When uncertain, prefer recall over precision: keep all paths that are semantically plausible.
6. Remove only paths that clearly contradict the question semantics.

Output format:
<analysis>
Two short sentences max. Do NOT deliberate. Just state which paths fit and why.
</analysis>
<selected>comma-separated path indices only</selected>
<need_more>yes or no</need_more>

Rules:
- Do not copy example numbers.
- Do not leave out a semantically plausible path just because it is longer.
- If several paths are plausible, select all of them."""
    return select_prompt, SYSTEM_PATH_SELECT_RECALL


def build_path_select_prompt_diverse(question, decomp_context, path_lines):
    """Build prompt for diverse path selection (Stage 7 pipeline).

    Returns (prompt_text, system_text).
    """
    select_prompt = f"""Select reasoning paths for this question.

Question: {question}
{decomp_context}
Paths (sorted by relevance, top is best):
{chr(10).join(path_lines)}

Task:
Evaluate each path against the expected relation pattern and assign it to one of three categories:

- Strong Match: The relation chain closely matches the intended reasoning structure.
- Partial but Useful: The path does not fully match the intended structure, but captures an important part of the reasoning and may still help identify or verify the answer.
- Mismatch: The path is semantically off-track and should not be selected unless no better alternatives exist.

Selection rules:
1. Prefer Strong Match paths first.
2. If fewer than 2 Strong Match paths exist, add Partial but Useful paths until you have 2 to 4 total.
3. Prefer semantic diversity among the selected paths.
4. Prefer paths that reach different candidate sets.
5. Avoid Mismatch paths unless there are no better alternatives.

Important: It is acceptable that only one path is a Strong Match. In that case, you must supplement it with the best Partial but Useful path(s), rather than selecting a Mismatch path.

<analysis>
Path 1: [Strong Match / Partial / Mismatch] — one short reason.
Path 2: [Strong Match / Partial / Mismatch] — one short reason.
...
Selected: explain in 1-2 sentences why these paths were chosen.
</analysis>
<selected>comma-separated path numbers</selected>"""
    return select_prompt, SYSTEM_PATH_SELECT_DIVerse


# ---------------------------------------------------------------------------
# V1 / Default reasoning prompt (4-step structured reasoning)
# ---------------------------------------------------------------------------

def build_v1_reason_prompt(question, pattern_text):
    """Build the V1/default 4-step reasoning prompt.

    Returns (prompt_text, system_text).
    """
    reason_prompt = f"""QUESTION: {question}

GRAPH EVIDENCE (from Freebase, snapshot circa 2015):
{pattern_text}

━━━ REASONING TASK ━━━

Use the graph evidence to answer the question.
Reason step by step, but keep each step concise: 1 to 2 sentences.

STEP 1 — QUESTION UNDERSTANDING
Explain what the question is asking for, what the answer type is, and what constraint(s) must be satisfied.
Also state whether the last hop is the answer itself or only a verification condition.

STEP 2 — PATTERN COMPARISON
You must evaluate at least two candidate patterns if more than one is available.
For each pattern, say MATCH or MISMATCH and give one short reason why its relation chain semantically matches or mismatches the question.
Then choose the best matching pattern and briefly justify why it is better than the others.

STEP 3 — ANSWER POSITION AND CANDIDATES
Explain where the answer is located in the selected pattern (which hop/node).
List the candidate entities at that position, and mention the graph evidence connecting them.

STEP 4 — CONSTRAINT VERIFICATION AND ANSWER SELECTION

4a — IDENTIFY CONSTRAINTS (explicit AND implicit)

Determine the expected ANSWER TYPE from the question (person, country, language, event, year, etc.).
Identify ALL constraints:

EXPLICIT constraints: directly stated in the question (dates, locations, superlatives, quantities, conditions).

IMPLICIT constraints — apply these heuristics based on question semantics:
- UNIQUE ROLE/POSITION ("the governor", "the president", "the leader", "the capital") WITHOUT a time qualifier → implicit: prefer the MOST RECENT or CURRENT holder
- EVENTS/ACHIEVEMENTS/AWARDS ("wins", "championships", "movies", "albums", "titles") → NO implicit "current" constraint; return ALL matching instances
- ATTRIBUTES/PROPERTIES ("languages spoken", "religions practiced", "government type", "currency") → return ALL that apply
- GROUP MEMBERSHIP ("countries in X", "states bisected by Y", "members of") → return ALL matching members

CRITICAL: The singular/plural form of the question ALONE does not determine answer count. Use the heuristics above.

4b — CONSTRAINT CHECK
For each candidate, verify:
1. TYPE MATCH: Does this candidate match the expected ANSWER TYPE? Discard non-matching types (e.g., discard countries, dates, roles when question asks for languages).
2. EXPLICIT constraints: Does it satisfy all stated conditions?
3. IMPLICIT constraints: Does it satisfy the applicable heuristic?
  - Constraint: [description] → candidate A: PASS/FAIL, candidate B: PASS/FAIL, ...
Collect ALL candidates that pass ALL constraints → PASSING SET.
If graph evidence does NOT show a candidate fails, KEEP it.

4c — OUTPUT DECISION
- PASSING SET has 1 member → output it.
- PASSING SET has multiple + implicit "most recent" applies → output the MOST RECENT member by graph dates. If no dates in evidence, output ALL.
- PASSING SET has multiple + no implicit limit → output ALL members.
- NEVER discard a candidate solely because the question uses singular phrasing.

RULES:
- NEVER decide answer count before constraint checking. Evaluate ALL candidates first.
- Prefer graph evidence over intuition. NEVER use external knowledge to filter.
- The answer may appear at an intermediate hop, not necessarily the terminal node.
- For geographic constraints, only remove if graph evidence explicitly contradicts.
- For temporal constraints, look for date values in evidence. No dates → do NOT filter.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- When in doubt, output ALL passing candidates. Over-output is better than discarding valid answers.

━━━ EXAMPLES (illustrate METHOD only — do NOT copy answers) ━━━

Example A — TYPE FILTER + IMPLICIT "CURRENT" for unique role:
Q: "Who is the president of France?"
Candidates: [Emmanuel Macron, François Hollande, France, President, 2017-05-14]
→ Answer type: person (president). Type filter: FAIL=[France(country), President(role), 2017-05-14(date)]
→ Remaining: [Emmanuel Macron, François Hollande]. Implicit: unique role no time qualifier → most recent
→ Output: Emmanuel Macron

Example B — EVENTS/ACHIEVEMENTS → return ALL:
Q: "What movies did the actor who played Forrest Gump star in?"
Candidates: [Forrest Gump, Tom Hanks, Saving Private Ryan, Cast Away, Actor, 1994]
→ Answer type: movie. Type filter: FAIL=[Tom Hanks(person), Actor(role), 1994(year)]
→ Remaining: [Forrest Gump, Saving Private Ryan, Cast Away]. Implicit: events/works → ALL
→ Output: Forrest Gump | Saving Private Ryan | Cast Away

Example C — PATTERN SELECTION:
Q: "What countries border France?"
Pattern A: location.location.adjoining_countries (direct, 1-hop)
Pattern B: location.location.containedby → location.location.adjoining_countries (via region, 2-hop)
→ Choose Pattern A: direct semantic match, shorter path, fewer noise entities in candidates

━━━ OUTPUT FORMAT ━━━
<reasoning>
Step 1: 1-2 sentences.
Step 2: 1-2 sentences per pattern evaluated, then 1 sentence for choice.
Step 3: 1-2 sentences.
Step 4a: Answer type + explicit/implicit constraints. Step 4b: TYPE MATCH and PASS/FAIL per candidate. Step 4c: PASSING SET and output decision.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>

Multiple answers: <answer>\\boxed{{cand1}} \\boxed{{cand2}}</answer>
NO text after </answer> tag."""

    return reason_prompt, SYSTEM_V1_REASON


# ---------------------------------------------------------------------------
# Entity-centric reasoning prompt (default for Stage 8, REASON_STYLE == "default")
# ---------------------------------------------------------------------------

def build_entity_reason_prompt(question, pattern_text, answer_type_hint, rewritten_hint):
    """Build the entity-centric 3-step reasoning prompt (Stage 8 default).

    Returns (prompt_text, system_text).
    """
    reason_prompt = f"""QUESTION: {question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

━━━ ENTITY-CENTRIC REASONING ━━━

STEP 1 — QUESTION UNDERSTANDING
Answer type: what kind of entity the question seeks (person, country, event, year, etc.).
Explicit constraints: conditions directly stated (dates, locations, superlatives, quantities).
Implicit constraint heuristic (pick one):
  - UNIQUE ROLE ("the governor/president/leader") without time qualifier → MOST RECENT only
  - EVENTS/ACHIEVEMENTS ("wins/championships/movies/albums") → return ALL matching
  - ATTRIBUTES/PROPERTIES ("languages/religions/currency") → return ALL that apply
  - GROUP MEMBERSHIP ("countries in / states in / members of") → return ALL matching members

STEP 2 — PER-CANDIDATE CONSTRAINT VERIFICATION
For EVERY candidate entity found in GRAPH EVIDENCE (including intermediate nodes and CVT attribute values), check:

2a. TYPE MATCH: Does this candidate's type match the answer type?
  Format: entity → KEEP (type matches) / REMOVE (type mismatch, state why)

2b. EXPLICIT CONSTRAINT CHECK (for type-matched candidates):
  For each explicit constraint from the question, check against graph evidence:
  Format: Constraint "[description]": entity1 PASS (evidence: ...), entity2 FAIL (evidence: ...)
  If graph evidence does NOT show failure → KEEP the entity.
  No dates in evidence → do NOT filter by time.

2c. IMPLICIT CONSTRAINT CHECK (for candidates passing 2b):
  Apply the heuristic from Step 1:
  - If unique role → pick MOST RECENT by graph dates; no dates → output ALL
  - If events/attributes/group → ALL candidates PASS

  Format: entity → PASS / FAIL (one reason)

STEP 3 — OUTPUT DECISION
Collect ALL candidates that PASS steps 2a + 2b + 2c.
- 1 entity → output it
- Multiple + unique role heuristic → pick MOST RECENT by graph dates; no dates → output ALL
- Multiple + events/attributes/group → output ALL
- Zero entities passed → <answer>None</answer>

RULES:
- Graph evidence ONLY. No outside knowledge.
- ANY entity in GRAPH EVIDENCE is a valid answer — including intermediate nodes and CVT attribute values.
- NEVER decide answer count before completing Step 2. Evaluate ALL candidates first.
- Singular/plural phrasing alone does NOT determine answer count.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- If unsure whether an entity satisfies a constraint → KEEP it.
- Over-output is better than discarding valid answers.
- For "group/organization that fought in/participated in" questions, political entities (countries, confederacies, alliances) are valid answer types — not only military units.

━━━ EXAMPLES (illustrate METHOD only) ━━━

Example A — TYPE FILTER + unique role:
Q: "Who is the president of France?"
2a (type person): KEEP [Macron, Hollande]. REMOVE [France(country), President(role), 2017-05-14(date)].
2b: No explicit time constraint.
2c (unique role, most recent): Macron PASS (term start 2017-05-14), Hollande FAIL (term start 2012-05-15).
3: Macron.

Example B — EVENTS → return ALL:
Q: "What movies did the actor who played Forrest Gump star in?"
2a (type movie): KEEP [Forrest Gump, Saving Private Ryan, Cast Away]. REMOVE [Tom Hanks(person), Actor(role), 1994(year)].
2b: No additional constraints beyond basic fact.
2c (events → ALL): All PASS.
3: Forrest Gump | Saving Private Ryan | Cast Away.

━━━ OUTPUT FORMAT ━━━
<reasoning>
Step 1: answer type, explicit constraints, implicit heuristic.
Step 2a: type match per candidate.
Step 2b: explicit constraint check per candidate.
Step 2c: implicit constraint check per candidate.
Step 3: passing set and output decision.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
No valid entity: <answer>None</answer>
NO text after </answer> tag."""

    return reason_prompt, SYSTEM_ENTITY


# ---------------------------------------------------------------------------
# entity-lite reasoning prompt (for 9B models)
# ---------------------------------------------------------------------------

def build_entity_lite_reason_prompt(question, pattern_text, answer_type_hint, rewritten_hint,
                                    cand_list, atype):
    """Build the entity-lite 2-step reasoning prompt (high-recall, no None).

    Returns (prompt_text, system_text).
    """
    reason_prompt = f"""QUESTION: {question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

━━━ ANSWER SELECTION ━━━

STEP 1 — Identify what the question needs.
Answer type: {atype}
Key constraints from the question: list them briefly.
Cardinality: Does the question ask for ONE specific role (e.g. "the governor", "the president")? If yes → pick the most recent one with dates in evidence. Otherwise → output ALL matching candidates.

STEP 2 — Evaluate each candidate.
For each candidate in CANDIDATE ENTITIES:
- KEEP if type roughly matches answer type.
- KEEP if no evidence contradicts it.
- REMOVE only if graph evidence explicitly shows it is WRONG type or contradicts a constraint.

RULES:
- Use graph evidence only. No outside knowledge.
- Do NOT remove a candidate for missing evidence. Remove only if evidence explicitly contradicts.
- Do NOT output None. If uncertain, output the best matching candidate(s).
- Prefer over-output to under-output.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES.
- Copy entity strings exactly. Never output Freebase IDs (m.0xxx).
- Over-output is better than discarding valid answers.

━━━ OUTPUT ━━━
<reasoning>
Step 1: answer type, constraints, cardinality decision.
Step 2: per-candidate KEEP/REMOVE with one reason each.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer> tag."""

    return reason_prompt, SYSTEM_ENTITY_LITE


# ---------------------------------------------------------------------------
# check-style reasoning prompt (checklist v2)
# ---------------------------------------------------------------------------

def build_check_reason_prompt(question, pattern_text, answer_type_hint, rewritten_hint,
                              cand_list, atype, type_check_lines):
    """Build the checklist v2 reasoning prompt.

    Returns (prompt_text, system_text).
    """
    reason_prompt = f"""QUESTION: {question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

=== CHECKLIST ===

□ ANSWER TYPE: {atype}

□ TYPE FILTER — check each candidate's type matches answer type:
{type_check_lines}

□ CONSTRAINTS from question:
  - ____
  Per kept candidate: PASS / FAIL

□ GRANULARITY CHECK:
  Any candidate is a parent/child of another (e.g. city vs stadium, country vs sport)?
  -> Pick the one matching question specificity.

□ CARDINALITY (pick one):
  [ ] unique role ("the X", no time) -> MOST RECENT only
  [ ] events / achievements -> ALL
  [ ] attributes / properties -> ALL
  [ ] group membership -> ALL
  [ ] none of the above -> ALL remaining

□ KEPT AFTER FILTERS: [list here]

RULES:
- Graph evidence only. No outside knowledge.
- No dates in evidence -> do NOT remove by time.
- "When" -> event NAME, not raw year.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- Answer may be at intermediate hop.
- In doubt -> KEEP.
- NEVER output "None". If all removed, pick most specific candidate.

ANSWER STRING (copy exactly from CANDIDATE ENTITIES above):
____

<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer>."""

    return reason_prompt, SYSTEM_CHECK


# ---------------------------------------------------------------------------
# ecot (Evidence-COT hybrid) reasoning prompt
# ---------------------------------------------------------------------------

def build_ecot_reason_prompt(question, pattern_text, answer_type_hint, rewritten_hint,
                             cand_list, atype, type_check_lines):
    """Build the evidence-COT hybrid reasoning prompt.

    Returns (prompt_text, system_text).
    """
    reason_prompt = f"""QUESTION: {question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

=== STRUCTURED EVIDENCE CHECK ===

1. ANSWER TYPE: {atype}

2. TYPE + EVIDENCE CHECK (one fact per candidate):
{type_check_lines}

3. CONSTRAINT CHECK:
   Question requires: ____
   Kept candidates that satisfy it: ____

4. GRANULARITY: If any kept candidate is a parent of another (city vs venue), pick the specific one.

5. CARDINALITY: unique role -> MOST RECENT | events/attributes -> ALL | in doubt -> ALL

6. KEPT: [list final kept candidates here]

RULES:
- Graph evidence only. No outside knowledge.
- No dates -> do NOT filter by time.
- "When" -> event NAME (e.g. "2014 World Series"), NOT raw timestamp or year.
- NEVER output "None". If all removed, pick most specific kept.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.

<answer>\\boxed{{exact entity from CANDIDATE ENTITIES}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer>."""

    return reason_prompt, SYSTEM_ECOT


# ---------------------------------------------------------------------------
# entity-centric reasoning prompt (REASON_STYLE == "entity")
# ---------------------------------------------------------------------------

def build_entity_full_reason_prompt(question, pattern_text, answer_type_hint, rewritten_hint,
                                    cand_list, atype, rel_list):
    """Build the full entity-centric 3-step reasoning prompt with relation disambiguation.

    Returns (prompt_text, system_text).
    """
    reason_prompt = f"""QUESTION: {question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES (traversal endpoints):
{cand_list}

NOTE: Any entity name appearing in GRAPH EVIDENCE (including intermediate nodes, CVT attribute values, and ← also lines) is also a valid answer. Do NOT restrict answers to the list above only.

━━━ ENTITY-CENTRIC REASONING ━━━

STEP 1 — QUESTION UNDERSTANDING
Answer type: what kind of entity the question seeks (person, country, event, year, etc.).
Available relations in evidence: {rel_list}
Explicit constraints from the question (time, location, superlatives, quantities, etc.): ____
Implicit constraint heuristic (pick one):
  - UNIQUE ROLE ("the governor/president/leader") without time qualifier → MOST RECENT only
  - EVENTS/ACHIEVEMENTS ("wins/championships/movies/albums") → return ALL matching
  - ATTRIBUTES/PROPERTIES ("languages/religions/currency") → return ALL that apply
  - GROUP MEMBERSHIP ("countries in / states in / members of") → return ALL matching members

STEP 2 — PER-CANDIDATE CONSTRAINT VERIFICATION
For EVERY candidate entity found in GRAPH EVIDENCE (including intermediate nodes and CVT attribute values), check:

2a. TYPE MATCH: Does this candidate's type match the answer type?
  Format: entity → KEEP (type matches) / REMOVE (type mismatch, state why)

2b. EXPLICIT CONSTRAINT CHECK (for type-matched candidates):
  For each explicit constraint from the question, check against graph evidence:
  Format: Constraint "[description]": entity1 PASS (evidence: ...), entity2 FAIL (evidence: ...)
  If graph evidence does NOT show failure → KEEP the entity.
  No dates in evidence → do NOT filter by time.

2c. IMPLICIT CONSTRAINT CHECK (for candidates passing 2b):
  Apply the heuristic from Step 1:
  - If unique role → pick MOST RECENT by graph dates; no dates → output ALL
  - If events/attributes/group → ALL candidates PASS

  Format: entity → PASS / FAIL (one reason)

STEP 3 — OUTPUT DECISION
Collect ALL candidates that PASS steps 2a + 2b + 2c.
- 1 entity → output it
- Multiple + unique role heuristic → pick MOST RECENT by graph dates; no dates → output ALL
- Multiple + events/attributes/group → output ALL
- Zero entities passed → <answer>None</answer>

RULES:
- Graph evidence ONLY. No outside knowledge.
- ANY entity in GRAPH EVIDENCE is a valid answer — including intermediate nodes, CVT attribute values, and entities in ← also lines.
- NEVER decide answer count before completing Step 2. Evaluate ALL candidates first.
- Singular/plural phrasing alone does NOT determine answer count.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- If unsure whether an entity satisfies a constraint → KEEP it.
- Over-output is better than discarding valid answers.
- For "group/organization that fought in/participated in" questions, political entities (countries, confederacies, alliances) are valid answer types — not only military units.

━━━ EXAMPLES (illustrate METHOD only) ━━━

Example A — TYPE FILTER + unique role:
Q: "Who is the president of France?"
2a (type person): KEEP [Macron, Hollande]. REMOVE [France(country), President(role), 2017-05-14(date)].
2b: No explicit time constraint.
2c (unique role, most recent): Macron PASS (term start 2017-05-14), Hollande FAIL (term start 2012-05-15).
3: Macron.

Example B — EVENTS → return ALL:
Q: "What movies did the actor who played Forrest Gump star in?"
2a (type movie): KEEP [Forrest Gump, Saving Private Ryan, Cast Away]. REMOVE [Tom Hanks(person), Actor(role), 1994(year)].
2b: No additional constraints beyond basic fact.
2c (events → ALL): All PASS.
3: Forrest Gump | Saving Private Ryan | Cast Away.

━━━ OUTPUT FORMAT ━━━
<reasoning>
Step 1: answer type, explicit constraints, implicit heuristic.
Step 2a: type match per candidate.
Step 2b: explicit constraint check per candidate.
Step 2c: implicit constraint check per candidate.
Step 3: passing set and output decision.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
No valid entity: <answer>None</answer>
NO text after </answer> tag."""

    return reason_prompt, SYSTEM_ENTITY


# ---------------------------------------------------------------------------
# V2 fast 2-step reasoning prompt
# ---------------------------------------------------------------------------

def build_v2_reason_prompt(question, pattern_text, answer_type_hint, rewritten_hint,
                           cand_list, atype):
    """Build the V2 fast 2-step reasoning prompt (fact-match → constraint filter).

    Returns (prompt_text, system_text).
    """
    reason_prompt = f"""QUESTION: {question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

━━━ QUICK MATCH ━━━

Step 1 — Fact-matching: find ALL candidates that match answer type ({atype}) and basic relation facts.
Output the FULL list of type-matched entities. Do NOT pick just one — list every candidate that fits the basic fact pattern.

Step 2 — Constraint filter: from the Step 1 list, remove only those explicitly contradicted by constraints (time, location, quantity). Output ALL remaining.
If the question asks for a unique role ("the governor", "the president"), pick the MOST RECENT one.
Otherwise, output ALL remaining entities.

RULES:
- Graph evidence only. No outside knowledge.
- "When" → event NAME (e.g. "2014 World Series"), NOT raw year.
- Copy entity strings exactly. Never output Freebase IDs (m.0xxx).
- Over-output > under-output. If uncertain, KEEP.
- No dates in evidence → do NOT filter by time.
- Reasoning ≤ 8 lines. Do NOT restate the question.

<reasoning>
Step 1: [fact-matched candidates: list all that match type/relation]
Step 2: [constraint filter: which are removed, which remain; cardinality decision]
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}} \\boxed{{e3}}</answer>
NO text after </answer> tag."""

    return reason_prompt, SYSTEM_V2


# ---------------------------------------------------------------------------
# Direct-answer prompt (parametric fallback)
# ---------------------------------------------------------------------------

def build_direct_answer_prompt(question):
    """Build the direct-answer prompt for cases with no graph paths.

    Returns (prompt_text, system_text).
    """
    prompt = f"""QUESTION: {question}

No reliable graph paths were found for this question. Use your parametric knowledge to answer directly.

<reasoning>One short sentence.</reasoning>
<answer>\\boxed{{exact entity}}</answer>"""
    return prompt, SYSTEM_DIRECT_ANSWER


# ---------------------------------------------------------------------------
# Frontend retry prompts
# ---------------------------------------------------------------------------

def build_year_retry_prompt(question, llm_answer, cand_str, year_hint=""):
    """Build retry prompt when LLM output a raw year instead of an entity.

    Returns (prompt_text, system_text).
    """
    prompt = f"""QUESTION: {question}

Your previous answer was: {llm_answer}
This is NOT a valid entity from the graph.

VALID CANDIDATES (pick one or more exact strings):
{cand_str}
{year_hint}
Pick the candidate that best answers the question. Output ONLY the exact candidate string.

<answer>\\boxed{{candidate}}</answer>"""
    return prompt, SYSTEM_YEAR_RETRY


def build_path_reselect_retry_prompt(question, pattern_text, answer_type_hint, rewritten_hint):
    """Build retry prompt when previous paths had no answer.

    Returns (prompt_text, system_text).
    """
    retry_prompt = f"""QUESTION: {question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE (re-selected paths — previous paths had insufficient evidence):
{pattern_text}

Previous reasoning found NO valid answer. These are ALTERNATIVE paths. Find the answer here.

<evidence_summary>
Brief evaluation of new paths.
</evidence_summary>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer>."""
    return retry_prompt, SYSTEM_PATH_RESELECT_RETRY


# ---------------------------------------------------------------------------
# Decomposition retry hint (Stage 1.5)
# ---------------------------------------------------------------------------

def build_decomp_retry_hint(prev_step, prev_rel):
    """Build the hint appended when a 1-step decomposition needs retry."""
    return f"""

[NOTE: The previous decomposition produced only 1 hop ("{prev_step}" / rel: {prev_rel}), which is INSUFFICIENT for this complex question. You MUST produce at least 2 hops. Think about the intermediate entity between the anchor and the final answer.]"""


# ---------------------------------------------------------------------------
# Year detection hint for retry
# ---------------------------------------------------------------------------

YEAR_RETRY_HINT = """
NOTE: You output a raw year/timestamp. The question asks "when" — the answer must be the
EVENT ENTITY NAME (e.g. "2008 NBA Finals"), not just the year number. Pick the event entity."""


# ---------------------------------------------------------------------------
# Cascade decomposition — Step 1: sub-question decomposition (EN)
# ---------------------------------------------------------------------------

CASCADE_DECOMP_PROMPT_EN = """
Read the question and restructure it, based solely on its literal semantics, into logical sub-questions that must be solved sequentially.

Note: This is question-logic decomposition only. Do not perform factual reasoning or choose specific KG relations.

Example 1:
Question: Which of the seven Central American countries had co2 emissions per capita once of 2009 metric ton?
Result:
<analysis>
Answer focus: The question ultimately asks for countries satisfying the given condition.
Known entities: "Central American countries" and "2009" are given by the question.
Answer type hint: country
Decomposition strategy: First determine the candidate country set, then query their CO2 emissions per capita in 2009.
</analysis>
<answer>
1. What are the seven Central American countries?
2. What was the CO2 emissions per capita of these countries in 2009?
</answer>

Example 2:
Question: what did peter tchaikovsky do
Result:
<analysis>
Answer focus: The question asks for the person's profession, occupation, or primary role.
Known entities: "Peter Tchaikovsky" is the known entity.
Answer type hint: occupation/profession
Decomposition strategy: Directly query the person's profession or occupation.
</analysis>
<answer>
1. What is Peter Tchaikovsky's profession or occupation?
</answer>

Input:
Question: {question}

Requirements:
1. Only decompose the solving logic expressed by the question itself. Do not answer the question, look up facts, or verify entities.
2. All concrete entities appearing in the question are known by default. Do not generate entity-explanation sub-questions.
3. Do NOT generate sub-questions like "Who is X?", "What is X?", "Where is X?", or "What type is X?" unless the original question explicitly asks about that entity itself.
4. Each sub-question should correspond to one independent relation, attribute, or constraint. Do not repeat sub-questions or express the same fact in both forward and reverse directions.
5. Do not over-decompose. Only keep the logical steps necessary to solve the original question.
6. Sub-questions must serve the final answer. Do not add background-knowledge questions.
7. If the question contains a known entity as a starting point, directly ask about the relation between that entity and the target variable.
8. If the question contains modifier clauses, geographic scope, time scope, or other constraints, decompose them into constraint sub-questions.
9. If the original question contains only one core relation and one constraint, it can be decomposed into two sub-questions: one relation sub-question and one constraint sub-question.
10. Output must contain both <analysis> and <answer> tags.
11. <analysis> must contain exactly four lines:
    - Answer focus: State what the original question ultimately asks for.
    - Known entities: List entities from the question that are known and should not be explained.
    - Answer type hint: Identify the expected answer type or relevant KG relation category (e.g. occupation, population, capital, birthPlace). Be specific — use "occupation/profession" when the question asks what someone does or did for a living.
    - Decomposition strategy: State the decomposition approach by relation or constraint.
12. <answer> must contain only a numbered list of sub-questions, one per line, with no extra explanation.

Output format:
<analysis>
Answer focus: ...
Known entities: ...
Answer type hint: ...
Decomposition strategy: ...
</analysis>
<answer>
1. ...
2. ...
</answer>
"""


# ---------------------------------------------------------------------------
# Cascade decomposition — Step 2: sub-questions → triples (EN)
# ---------------------------------------------------------------------------

CASCADE_SUBQ_TO_TRIPLES_PROMPT_EN = """
You are a KGQA sub-question-to-triples conversion assistant.

Convert the given sequential sub-questions into a minimal ordered reasoning subgraph for retrieval.

Important:
This is structured graph decomposition for retrieval, not actual answering, fact-checking, ranking, or entity verification.
Use the original question and the provided sub-questions only.
Do not use outside knowledge.
Do not output operations.

Input:
Original question:
{question}

Known entities:
{entities}

Sub-question decomposition:
{subquestions}

Requirements:
1. Use the sub-questions as the primary guide.
2. Each sub-question must be represented by at least one triple unless it only restates a previous fact.
3. Do not add facts not expressed by the question or sub-questions.
4. Use variables for unknown entities or values.
5. Copy known entity strings exactly.
6. Reuse variables across triples when later sub-questions depend on earlier ones.
7. Do not duplicate the same fact in reverse direction.
8. Predicate descriptions should be clear 4-8 word relation phrases.
9. If a sub-question expresses ranking, comparison, time, quantity, or numeric constraints, represent the needed attribute or value as a normal retrieval triple. Do not output operations.
10. Output must contain <analysis> and <answer>.
11. <answer> must contain valid JSON only.

Entity role principles:
1. The known entity list is a candidate pool, not a required set.
2. Entity roles must be assigned after triples are drafted, not before.
3. The generated triples are the primary evidence for entity_roles.
4. If a known entity appears exactly as a subject or object in any generated triple, it must appear in entity_roles.
5. Do not silently drop a concrete known entity that appears in the generated triples.
6. Generic type words or broad labels such as Country, Person, Governor, State, Film, Super Bowl, College/University, Location, or Organization should not become anchorentity or pathentity unless the question uses them as exact named entities.
7. If a concrete known entity appears in the question or sub-questions but is not used in any triple, either add the missing retrieval triple if it is required, or mark it as unusedentity with a short reason.

Entity role definitions:
- anchorentity: the single known concrete entity used as the starting point of graph traversal.
- pathentity: a known concrete entity that appears later in the reasoning chain or acts as an explicit constraint, but is not the starting point.
- unusedentity: a known entity from the input list that is not required by the question logic or is only a generic type label.

Anchor selection rules:
1. If at least one relevant concrete known entity is used in the triples, exactly one entity must be labeled anchorentity.
2. The anchorentity must be the unique start node for the ordered reasoning graph.
3. Prefer the known entity that appears in the first triple subject.
4. If the first triple subject is a variable, choose the known entity closest to the answer variable in the main reasoning path.
5. Other known entities used in triples must be labeled pathentity.
6. Do not mark multiple entities as anchorentity.
7. Do not mark an entity as anchorentity only because it is mentioned in the question.

Validation rule:
If any generated triple contains an exact known entity string, entity_roles must not be empty.

Analysis requirements:
<analysis> must contain exactly five lines:
- Answer focus: state the final answer variable and answer type.
- Sub-question mapping: state how the sub-questions map to triples.
- Entity role analysis: for each known entity, briefly state used_in_triples=yes/no and role decision.
- Anchor decision: state the single selected anchorentity and why.
- Triple strategy: state how the ordered triples support graph retrieval.

Output format:
<analysis>
Answer focus: ...
Sub-question mapping: ...
Entity role analysis: ...
Anchor decision: ...
Triple strategy: ...
</analysis>
<answer>
{{
  "answer_variable": "?answer",
  "answer_type": "answer type",
  "entity_roles": [
    {{
      "entity": "EntityName",
      "role": "anchorentity | pathentity | unusedentity",
      "reason": "short reason"
    }}
  ],
  "triples": [
    {{
      "source_subquestion": 1,
      "subject": "Node",
      "predicate": "relation description",
      "object": "Node"
    }}
  ]
}}
</answer>
"""


# ---------------------------------------------------------------------------
# Cascade decomposition — Step 2 variant: enhanced entity validation (EN)
# ---------------------------------------------------------------------------

CASCADE_SUBQ_TO_TRIPLES_PROMPT_EN_V2 = """
You are a KGQA sub-question-to-triples conversion assistant.

Your task is to convert a sequential sub-question decomposition into a minimal ordered reasoning subgraph for retrieval.

Important:
This is structured graph decomposition for retrieval, not actual answering, fact-checking, ranking, filtering, or entity verification.
Use the original question and the provided sub-questions only.
Do not use outside knowledge.

Input:
Original question:
{question}

Known entities:
{entities}

Sub-question decomposition:
{subquestions}

Task:
Convert the sub-questions into ordered triples that can be used for knowledge graph retrieval.

Requirements:
1. Use the sub-questions as the primary decomposition guide. Do not ignore, skip, or merge independent sub-questions.
2. Each triple should correspond to one relation, attribute, or constraint expressed by a sub-question.
3. Do not add triples that are not supported by the original question or the sub-questions.
4. Do not answer the question or replace unknown answers with real-world entities.
5. If the target entity is unknown, use a variable such as ?country, ?person, ?team, ?event, ?religion, ?year, or ?answer.
6. Known concrete entities from the entity list should be copied exactly. Do not shorten, normalize, or partially copy entity names.
7. The qentity list is a candidate pool, not a required set.
8. Use a qentity only when it is directly needed by the original question or by one of the sub-questions.
9. Before using any qentity as anchorentity or pathentity, check:
   - Surface support: it is explicitly mentioned in the original question or sub-questions.
   - Reasoning function: it constrains the answer variable or a necessary intermediate variable.
   - Attachment point: it can be attached to a specific variable through a triple derived from a sub-question.
10. If a qentity fails any of these checks, mark it as unusedentity.
11. There must be exactly one anchorentity if at least one relevant concrete entity exists.
12. The anchorentity is the unique starting point for graph traversal.
13. Choose the anchorentity by this priority:
    - the entity directly connected to the answer variable;
    - otherwise, the entity used in the first sub-question;
    - otherwise, the entity that best starts the main reasoning path.
14. Other relevant known entities must be labeled pathentity, not anchorentity.
15. A pathentity must appear in at least one triple as subject or object. Otherwise it must be unusedentity.
16. If a sub-question identifies an intermediate entity, represent it as a variable and reuse that variable in later triples.
17. If a sub-question expresses a constraint, attach that constraint to the correct variable instead of turning it into an unrelated path.
18. If a sub-question expresses comparison, ranking, time filtering, quantity filtering, or numeric filtering, represent the required attribute or value as a normal retrieval triple. Do not output operations.
19. Do not duplicate the same fact in both forward and reverse directions.
20. Keep predicates as natural-language relation descriptions, not KG relation IDs.
21. Predicate descriptions should be clear 4-8 word phrases describing the actual relation meaning.
22. The triples must form an ordered reasoning tree or path that a graph-walk tool can follow from the single anchor entity to variables, constraints, and the final answer.
23. Output must contain both <analysis> and <answer> tags.
24. <analysis> must contain exactly four lines:
    - Answer focus: State the final answer variable and answer type.
    - Anchor entity: State the single selected anchor entity and why it starts the path.
    - Entity filtering: State used qentities and unused qentities briefly.
    - Triple strategy: State how sub-questions map to retrieval triples.
25. <answer> must contain valid JSON only. Escape internal double quotes if needed.

Output format:
<analysis>
Answer focus: ...
Anchor entity: ...
Entity filtering: used=[...]; unused=[...].
Triple strategy: ...
</analysis>
<answer>
{{
  "answer_variable": "?answer",
  "answer_type": "answer type",
  "entity_roles": [
    {{
      "entity": "EntityName",
      "role": "anchorentity | pathentity | unusedentity",
      "reason": "short reason"
    }}
  ],
  "triples": [
    {{
      "source_subquestion": 1,
      "subject": "Node",
      "predicate": "relation description",
      "object": "Node"
    }}
  ]
}}
</answer>
"""

# ---------------------------------------------------------------------------
# Triple decomposition prompts (Stage 1)
# ---------------------------------------------------------------------------

SIMPLE_DECOMP_PROMPT = """
You are a KGQA question subgraph decomposition assistant. Given a question and an entity list, decompose the question into a sequential reasoning subgraph of triples.

Input:
Question: {question}
Entities: {entities}

Task:
Based solely on the question, select an appropriate anchor entity from the entity list, and decompose the question into a triple subgraph usable for knowledge graph traversal.

Rules:
1. Only decompose what the question expresses. Do not add information not stated in the question. Decompose complex semantics into easier-to-understand text descriptions.
2. If the target answer entity is not in the entity list, use a ?variable to represent the unknown answer node, e.g. ?actor, ?character, ?movie, ?country. Do not fill in specific answer entities using common sense or external knowledge.
3. If the entity list contains type words like College/University, Person, Country, City, Film that correspond to the answer type being asked, do not treat them as regular entity nodes; they only indicate the type of the answer variable.
4. Each triple represents one independent semantic relation. Relations must not overlap, and must not query the same fact in both forward and reverse directions.
5. The predicate field must describe the actual meaning of the relation. Do not use single short words; use a 4-6 word phrase to express the relation semantics.
6. Triples should form a sequential logical path, so that a traversal tool can start from the anchorentity, pass through unknown answer variables, path nodes, or constraint nodes, and reach the target subgraph.

Entity role rules:
For each known entity, assign exactly one role:
- anchorentity: the single known entity used as the starting point of graph traversal.
- pathentity: a known entity that appears later in the reasoning chain or acts as a constraint, but is not the starting point.
- unusedentity: a known entity that is not needed by the question logic.

Anchor selection rules:
1. If there is at least one relevant known concrete entity, exactly one entity must be labeled anchorentity.
2. The anchorentity must be the unique start node for the ordered reasoning graph.
3. Prefer the entity that is closest to the answer variable in the main answer path.
4. If the sub-questions imply a clear first step, choose the entity used in the first sub-question as anchorentity.
5. Other known entities that constrain an intermediate variable or final answer must be labeled pathentity.
6. Do not mark multiple entities as anchorentity.
7. Do not mark an entity as anchorentity only because it is mentioned in the question.

7. In <reason>, write exactly three sentences:
   - Sentence 1: state the answer semantics, i.e. the final answer variable and answer type the question asks for.
   - Sentence 2: based on the answer semantics, decompose the question into easier-to-understand text, clearly distinguishing "the queried object" from "constraint conditions".
   - Sentence 3: describe the reasoning chain.
10. After the three sentences, write a SEPARATE line stating the total number of triples needed. Format: "Required triples: N"
    This N must exactly match the number of triples in your <answer>.
11. In the JSON <answer>, the "num_triples" field must equal the actual number of items in the "triples" array. If they do not match, your answer is invalid.
12. Complete reasoning within 500 tokens. Avoid excessive reflection and irrelevant derivation.
13. Output must contain two XML tags:
   - <reason>: the reasoning explanation, ending with the "Required triples: N" line.
   - <answer>: the final JSON result.

Output format:

<reason>
Answer semantics: the question asks for ...
Question decomposition: need to find ..., subject to ...
Reasoning chain: starting from ..., through ..., finally obtain ...
Required triples: N
</reason>

<answer>
{{
  "question_decomposition": "easy-to-understand text description after decomposing the question",
  "answer_variable": "?answer_variable",
  "answer_type": "answer type",
  "num_triples": N,
  "entity_roles": [
    {{
      "entity": "EntityName",
      "role": "anchorentity | pathentity | unusedentity"
    }}
  ],
  "triples": [
    {{
      "subject": "Node",
      "predicate": "relation description",
      "object": "Node"
    }}
  ]
}}
</answer>
"""


DECOMP_VERIFY_PROMPT = """Given a question and a list of reasoning triples, judge whether the triples cover ALL literal constraints in the question.

Question: {question}
Triples: {triples_json}

First, analyze in <reason> tag with exactly 3 steps. Each step at most 3 sentences:

Step 1 — Extract constraints from the question:
List ALL constraints in the question. A constraint is any noun phrase modifier, prepositional phrase, relative clause, superlative/comparative, or entity-binding condition that restricts the answer.

Step 2 — Map each constraint to a triple:
For each constraint found in Step 1, identify which triple covers it. A constraint is "covered" if the triple's predicate or variable binding directly resolves it. Mark constraints that have NO corresponding triple as "uncovered".

Step 3 — Conclusion:
List the uncovered constraints. If all constraints are covered, state "all covered".

Then output your judgment:
- <judge>complete</judge> if ALL constraints are covered.
- <judge>incomplete</judge> if any constraint is missing.

<reason>
Step 1: constraints=[...]
Step 2: mapping=[...]
Step 3: uncovered=[...]; verdict=[...]
</reason>
<judge>complete</judge>
OR
<judge>incomplete</judge>"""


REDECOMP_PROMPT = """You are a KGQA question subgraph decomposition assistant. A previous decomposition was judged INCOMPLETE by a reviewer. The reviewer found uncovered constraints. You MUST produce a corrected decomposition that covers ALL constraints.

Question: {question}
Entities: {entities}
Previous triples: {prev_triples}
Reviewer analysis:
{feedback}

Produce a corrected decomposition with ALL constraints covered. Follow the same rules and output format.

<reason>
Answer semantics: ...
Question decomposition: ...
Reasoning chain: ...
</reason>

<answer>
{{
  "question_decomposition": "...",
  "answer_variable": "?...",
  "answer_type": "...",
  "entity_roles": [{{"entity": "...", "role": "anchorentity | pathentity | unusedentity"}}],
  "triples": [{{"subject": "...", "predicate": "...", "object": "..."}}]
}}
</answer>"""


TRIPLE_PRUNE_PROMPT = """
You are a KGQA relation selection assistant.

Given the question, sub-question guided reasoning triples, and candidate KG relations for each step, select exactly the top 3 KG relation IDs for each reasoning step.

Input:
Question:
{question}

Reasoning triples:
{chain_text}

Candidate relations per step:
{blocks_text}

Task:
For each step, first briefly identify which candidate relations are relevant to the sub-question, then select exactly 3 KG relations from the corresponding candidate list.

Rules:
1. Use the sub-question and current triple as the main intent of the step.
2. Select only from the given candidate relations. Do not create or modify relation IDs.
3. Select exactly 3 relation IDs for each step if at least 3 candidates are available.
4. Rank the 3 selected IDs by semantic validity, not by listing order.
5. The relation must match the predicate meaning and support subject -> object traversal.
6. DIRECTION IS CRITICAL: check each candidate's "e.g." line. The example shows (SubjectEntity) -[relation]-> (ObjectEntity). If the triple needs A->B but the example shows B->A, the relation traverses in the WRONG direction. Reject it unless the triple's subject/object are swapped.
7. If the current object variable is reused in the next step, the relation output must be compatible with the next step's subject.
8. Prefer SPECIFIC relations over generic ones. A relation like "olympic_host_city.olympics_hosted" is better than "olympic_games.host_city" when the triple starts from a host city. Specific domain relations beat broad category relations.
9. Do not select relations solely by keyword overlap.
10. Reject wrong-domain relations even if they share words with the predicate.
11. Reject attribute/date/value relations when the object should be an entity, unless the sub-question asks for that attribute/date/value.
12. Reject entity-returning relations when the object should be an attribute/date/value.
13. Reject opposite-direction relations unless the examples clearly support the required traversal.
14. Candidate examples are PRIMARY evidence for relation behavior, direction, and output kind. You MUST read them before deciding.
15. If fewer than 3 relations are clearly valid, fill the remaining slots with the closest fallback relations.
16. selection_type must be:
    - matched: at least one selected relation is an exact or compatible semantic match.
    - fallback: all selected relations are weak fallback choices.

Output format:
1. First, for each step, write a brief `<analysis>` block listing the relevant relations and why they fit.
2. Then provide the JSON answer under `<answer>`.

<analysis>
Step 1: [sub-question intent]. Relevant candidates: ID=X (reason), ID=Y (reason), ID=Z (reason). Top 3: [X, Y, Z].
Step 2: [sub-question intent]. Relevant candidates: ID=A (reason), ID=B (reason), ID=C (reason). Top 3: [A, B, C].
</analysis>

<answer>
{{
  "selected_relations": [
    {{
      "step": "step_1",
      "selected_relation_ids": [3, 1, 5],
      "selection_type": "matched",
      "audit_summary": "3 fits perfectly; 1 matches direction; 5 is closest fallback."
    }},
    {{
      "step": "step_2",
      "selected_relation_ids": [4, 2, 7],
      "selection_type": "fallback",
      "audit_summary": "No exact match; all three are closest fallback relations."
    }}
  ]
}}
</answer>
"""
