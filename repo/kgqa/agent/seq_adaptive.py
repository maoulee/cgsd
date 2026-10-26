"""Opt-in adaptive SEQ interaction adapter; existing traversal stays intact.

Each query walks from the chosen actual centers. Its evidence graph joins the
matching delivered route; it is not forced through an accumulated fact program.
Private worker contexts isolate prepare/failed mutations. Full ledger commits
are required for graph state. Credit export never reads reasoning text.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

from kgqa.core.utils import normalize
from kgqa.agent.seq_schemas import adaptive_enabled, validate_args
from kgqa.agent.seq_routes import RouteStore, RouteBoundary, credit_messages, credit_text


def enabled(ctx=None):
    return adaptive_enabled() and not getattr(ctx, '_adaptive_worker', False)


def store(ctx):
    value = getattr(ctx, '_seq_route_store', None)
    if value is None:
        value = ctx._seq_route_store = RouteStore()
    return value


def action(tool, args):
    public = {k: v for k, v in args.items() if not k.startswith('_')}
    return json.dumps({'tool': tool, 'args': public}, ensure_ascii=False, sort_keys=True)


def error(message):
    return 'error: ' + str(message)


def node_key(index):
    return f'n:{index}'


def index_of(key):
    return int(key[2:]) if key.startswith('n:') else None


def _resolve_centers(args, ctx):
    from kgqa.agent import seq_tools as st
    manager = store(ctx)
    values, ids, preferred = [], [], {}
    for raw in args.get('center', []):
        if raw.startswith('@frontier:'):
            ref = raw.split(':', 1)[1]
            if ref in manager.routes:
                keys = manager.routes[ref].frontier
                rid = ref
            else:
                event = next((e for e in manager.events if e['call_id'] == ref), None)
                keys = event.get('frontier', []) if event else []
                rid = event['route_id'] if event else None
            usable = [index_of(k) for k in sorted(keys) if index_of(k) is not None]
            if not usable:
                raise RouteBoundary(f'FRONTIER: {raw} has no delivered graph-node frontier; choose a displayed entity or question anchor.')
            ids.extend(usable)
            for i in usable:
                preferred.setdefault(node_key(i), set()).add(rid)
            continue
        if raw.startswith('?'):
            raise RouteBoundary(f'CENTER_REFERENCE: {raw} does not select an environment frontier. Use a displayed entity or @frontier:{manager.active or "r1"}; no fact declaration is required.')
        i = st._name_to_idx(raw, ctx)
        if i is None or st._match_sim(raw, str(ctx.ents[i])) < 0.95:
            suggestion = f' Possible graph spelling: {ctx.ents[i]!r}.' if i is not None else ''
            raise RouteBoundary(f'CENTER_NAME: {raw!r} was not resolved confidently.{suggestion} Use a canonical question/entity spelling; no SG walk ran.')
        if i not in (getattr(ctx, 'subgraph_entities', None) or set()):
            raise RouteBoundary(f'CENTER_SOURCE: {raw!r} is not a question anchor or delivered center. Select from existing evidence; no SG walk ran.')
        if st._looks_like_value(raw) and normalize(raw) != normalize(str(ctx.ents[i])):
            raise RouteBoundary(f'CENTER_VALUE: {raw!r} is an attribute value; use its carrier entity.')
        ids.append(i)
    ids = sorted(set(ids))
    values = [str(ctx.ents[i]) for i in ids]
    return values, ids, preferred


def _worker(ctx, manager, ticket):
    staged = copy.copy(ctx)
    # Project graph arrays/indices are read-only and shared. Mutable result
    # state and the atomic evidence ledger are private to this transaction.
    mutable = ('consumed_anchors', 'subgraph_entities', 'all_candidates',
               'fact_evidence', 'fact_evidence_seq', 'fact_candidate_pool',
               'fid_pattern', 'walk_seen_entities', 'var_bindings', 'pattern_state',
               'subgraph_triples', 'cvt_attrs', 'evidence_entities',
               '_ev_rebuild_truncs', '_ev_rebuild_trunc_wm', '_render_display_stats',
               '_v4_compressed_groups')
    for key in mutable:
        if hasattr(ctx, key):
            setattr(staged, key, copy.deepcopy(getattr(ctx, key)))
    if getattr(ctx, 'evidence_records', None) is not None:
        staged.evidence_records = copy.deepcopy(ctx.evidence_records)
    route = manager.routes.get(ticket.route_id)
    raw = [f for e in manager.events if e.get('route_id') == ticket.route_id
           and e.get('outcome') == 'delivered' for f in e.get('raw_facts', [])]
    staged.accumulated_triples = {tuple(f) for f in raw}
    staged.all_candidates = list(dict.fromkeys(x for h, r, t in raw for x in (h, t)))
    staged.walk_seen_entities = list(staged.all_candidates)
    # Local queries: do not let fact IDs or old root programs override the
    # actual selected center. The route graph preserves prior constraints.
    # SEQUENCE DEPTH (2026-10-05 audit): anchor_seqs now seeds from LIVE so a
    # same-anchor continuation EXTENDS the accumulated layers instead of
    # restarting at depth 1 — the frontier-pool contract reads the previous
    # step's completions through the full sequence. Publish-back merges
    # longest-wins, so a shorter local walk never erases a deeper one.
    # WALK-STATE PORTS (user ruling 2026-10-09): every per-case state
    # field crosses the fork through ONE declaration — kgqa/agent/
    # walk_ports.SEED_FIELDS. The fork can never again drop a field (the
    # _failed_steps incident class). Mechanics of each field's seed live
    # in walk_ports.seed_fork; comments for anchor_seqs/fid_tree_roots/
    # chain_landing history: 2026-10-05 audit (sequences froze at depth 1
    # when the seed reset them) and B1/B2 batch 1 (a missing seed silently
    # kills the mechanism).
    from kgqa.agent.walk_ports import seed_fork
    seed_fork(staged, ctx)
    staged._adaptive_worker = True
    staged._adaptive_manager = manager
    staged._adaptive_ticket = ticket
    staged._adaptive_event = None
    return staged


def _plan(args, ctx):
    from kgqa.agent import seq_tools as st
    manager = store(ctx)
    names = list(dict.fromkeys(args.get('entities', [])))
    seeds = set(getattr(ctx, 'subgraph_entities', None) or ())
    roots = []
    for name in names:
        i = st._name_to_idx(name, ctx)
        if i is not None and st._match_sim(name, str(ctx.ents[i])) >= 0.95:
            seeds.add(i)
            roots.append(node_key(i))
    sample = getattr(ctx, 'sample', None) or {}
    grounded = [i for i in sample.get('q_entity_id_list', ())
                if isinstance(i, int) and 0 <= i < len(ctx.ents)]
    question_roots = [node_key(i) for i in grounded]
    seeds.update(grounded)
    manager.seed(roots + question_roots, question_roots=question_roots)
    manager.plan_args = copy.deepcopy(args)
    ctx.plan_entities = list(dict.fromkeys(names + [str(ctx.ents[i]) for i in grounded]))
    ctx.plan_answer_type = args.get('answer_type', '')
    ctx.plan_declared = True
    ctx.subgraph_entities = seeds
    # Plans are semantic guidance. No answer-producer/fact-closure contract.
    result = ('plan: accepted\nquestion_entities: ' + ' | '.join(names) +
              '\nadvice: Choose relations and actual centers freely. Routes are assigned from delivered graph evidence; no f/checkpoint declaration is required. Path reminders are optional and do not count as walk evidence.')
    manager.background = [{'role': 'assistant', 'content': action('plan', args)},
                          {'role': 'tool', 'name': 'plan', 'content': result}]
    return {'kind': 'done', 'result': result}


def _query_key(tool, args, req, ctx, ticket):
    manager = store(ctx)
    route = manager.routes.get(ticket.route_id)
    # INPUT CONSISTENCY (Codex F2, 2026-10-06): the RR candidate pool is a
    # query INPUT — another route's published sequence changes this route's
    # frontier-default pool without touching this route's version, so the
    # old key returned stale "unchanged_query" hits. Include the canonical
    # per-center relation pools (and the sequence layers they derive from)
    # in the key: same pools + same layers => same key; either changes =>
    # fresh key.
    _rr_requests = req.get('requests')
    _rr_pool_sig = None
    if tool == 'retrieve_relations' and _rr_requests:
        _rr_pool_sig = sorted(
            (str(_h), tuple(sorted(getattr(_p, 'difference', _p)() if isinstance(_p, frozenset) else _p)))
            if False else (str(_h), tuple(sorted(int(_r) for _r in _p)))
            for _h, _p in _rr_requests)
    _seq_sig = None
    if tool == 'retrieve_relations':
        _sq = getattr(ctx, 'anchor_seqs', None) or {}
        _seq_sig = sorted((int(_a), tuple(sorted(int(_r) for _l in _ls for _r in _l)))
                          for _a, _ls in _sq.items()) if _sq else 0
    payload = {'tool': tool, 'route': ticket.route_id,
               'centers': sorted(ticket.centers),
               'relations': sorted(req.get('rel_idxs', [])),
               'patterns': req.get('multistep', {}),
               'rr_requests': _rr_pool_sig,
               'rr_sequence_layers': _seq_sig,
               'question': args.get('question') or getattr(ctx, 'question', ''),
               'graph_version': getattr(ctx, '_seq_graph_version', 'immutable-case-graph'),
               'input_evidence_version': route.version if route else 0,
               'config': {k: v for k, v in os.environ.items() if k.startswith('SEQ_')}}
    def encode(value):
        if isinstance(value, (set, frozenset)):
            return sorted(value, key=repr)
        raise TypeError(type(value).__name__)
    # Multistep tuple keys are canonicalized separately for JSON.
    payload['patterns'] = repr(sorted((ci, sorted((repr(k), repr(v)) for k, v in pats.items()))
                                      for ci, pats in req.get('multistep', {}).items()))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=encode).encode()).hexdigest()


def prepare(tool, args, ctx):
    from kgqa.agent import seq_tools as st
    manager = store(ctx)
    if tool in ('plan', 'decompose'):
        return _plan(args, ctx)
    if tool == 'answer':
        return {'kind': 'done', 'result': st._do_seq_answer(args, ctx)}
    if not getattr(ctx, 'plan_declared', False):
        # Older policies often retrieve first. Bootstrap ONLY from already
        # grounded question seeds, never from gold or an invented fact plan.
        sample = getattr(ctx, 'sample', None) or {}
        ids = [i for i in (sample.get('q_entity_id_list') or ())
               if isinstance(i, int) and 0 <= i < len(ctx.ents)]
        if not ids:
            ids = [i for i in (getattr(ctx, 'subgraph_entities', ()) or ())
                   if isinstance(i, int) and 0 <= i < len(ctx.ents)]
        if not ids:
            return {'kind': 'done', 'result': error('PLAN_CONTEXT: submit plan with question_entities first. No facts/checkpoints are needed.')}
        manager.seed(node_key(i) for i in sorted(set(ids)))
        ctx.plan_entities = [str(ctx.ents[i]) for i in sorted(set(ids))]
        ctx.plan_declared = True
        ctx.subgraph_entities = set(getattr(ctx, 'subgraph_entities', ()) or ()) | set(ids)
    if tool not in ('retrieve_relations', 'retrieve_subgraph'):
        return {'kind': 'done', 'result': error('UNKNOWN_ACTION: choose plan, retrieve_relations, retrieve_subgraph or answer.')}
    try:
        _, ids, preferred = _resolve_centers(args, ctx)
        groups = manager.groups((node_key(i) for i in ids), preferred)
        children = []
        for rid, centers in groups.items():
            ticket = manager.propose(rid, centers, tool, check_budget=False)
            staged = _worker(ctx, manager, ticket)
            local = {k: v for k, v in args.items() if not k.startswith('_')}
            local['center'] = [str(ctx.ents[index_of(k)]) for k in sorted(centers)]
            local['sg'] = local['_canon_fid'] = rid
            # RR is about this actual center, not the latest fact frontier.
            req = st.dispatch_prepare(tool, local, staged)
            if req['kind'] == 'done':
                return req  # rejected before IO; no execution budget spent
            if tool == 'retrieve_subgraph' and len(centers) >= 4:
                req['var_name'] = '@frontier'  # reuse the existing set-state lane
            key = _query_key(tool, local, req, ctx, ticket)
            cached = manager.cache.get(key)
            if cached:
                msg = (f'route: {rid}\ncache_reference: {cached["call_id"]}\n'
                       'progress: unchanged_query\nadvice: The same query in the same input evidence '
                       'returned no additional information — the records for this step are fixed. '
                       'First infer and answer on the existing result; only a specific unexecuted '
                       'step justifies a different relation, known center, or route.')
                children.append({'cached': msg, 'ticket': ticket, 'tool': tool,
                                 'args': local, 'cached_call': cached['call_id']})
            else:
                manager.check(ticket, tool)
                children.append({'ctx': staged, 'req': req, 'ticket': ticket,
                                 'args': local, 'key': key, 'tool': tool})
        return {'kind': 'adaptive', 'children': children, 'tool': tool}
    except (RouteBoundary, TypeError, ValueError) as exc:
        return {'kind': 'done', 'result': error(exc)}
    except RuntimeError as exc:
        # PATTERN_CONSTRUCTION (R2, Codex audit v2): an internal failure
        # must reach the model as an error result, not crash the dispatch.
        # Evidence is retained; the model can retry or answer.
        return {'kind': 'done', 'result': error(exc)}


def start_walk(ctx, req):
    """Called inside _sg_execute AFTER corrections and BEFORE walking IO."""
    if not getattr(ctx, '_adaptive_worker', False) or ctx._adaptive_event is not None:
        return
    ctx._adaptive_event = ctx._adaptive_manager.begin(
        ctx._adaptive_ticket, 'retrieve_subgraph', req.get('rel_names', []))


async def execute(req, ctx, session):
    from kgqa.agent import seq_tools as st
    results = []
    # Per-case sequential finalization is intentional: ledger begin/finalize
    # is a transaction. Different cases still batch/concurrently execute.
    for child in req['children']:
        if 'cached' in child:
            event = store(ctx).reuse(child['ticket'], child['tool'], child['cached'], child['cached_call'])
            event['action'] = action(child['tool'], {k: v for k, v in child['args'].items() if not k.startswith('_') and k != 'sg'})
            results.append(child['cached'])
            continue
        staged = child['ctx']
        # Refresh the global evidence log after a preceding independent child.
        if getattr(ctx, 'evidence_records', None) is not None:
            staged.evidence_records = copy.deepcopy(ctx.evidence_records)
        try:
            if child['tool'] == 'retrieve_relations':
                staged._adaptive_event = store(ctx).begin(child['ticket'], child['tool'])
            bres = await st.dispatch_execute(child['req'], staged, session)
            result = st.dispatch_finalize(child['req'], bres, staged)
            results.append(_finish(child, result, ctx))
        except Exception as exc:
            # INTERNAL-error handling (user design ruling 2026-10-24, 9B-era
            # closed error taxonomy): the model may only ever see a SMALL,
            # actionable set of errors (bad parameter → name it + offer
            # candidates). These exceptions are OUR failures — including
            # environment-internal functions like the graph walk, which is
            # NOT a tool and can never appear in a tool list — so exception
            # types/keys must NOT leak into model-facing strings (the
            # KeyError:'walk' run showed the model just burns turns retrying
            # on opaque internals). Details go to environment-side telemetry
            # (EXEC_ERR_COUNTS → [tool] exec_err → step fuse) and stderr; the
            # model gets ONE closed-set protocol error.
            k = type(exc).__name__
            # RouteBoundary is a CONTROL-FLOW signal (the route layer's own
            # boundary), surfaced to the model like any protocol error — it is
            # not an environment failure and must not trip the step fuse.
            if not isinstance(exc, RouteBoundary):
                EXEC_ERR_COUNTS[k] = EXEC_ERR_COUNTS.get(k, 0) + 1
            print(f"[kgqa-env] tool exec failed ({child['tool']}): "
                  f"{k}: {exc}", flush=True)
            msg = error('ENVIRONMENT_UNAVAILABLE: the retrieval environment '
                        'failed on this call. Prior evidence is retained; '
                        'retry a different query or answer from what you '
                        'already have.')
            event = staged._adaptive_event
            if event:
                event.update(outcome='execution_failed', result=msg,
                             action=action(child['tool'], {k: v for k, v in child['args'].items() if not k.startswith('_') and k != 'sg'}))
            results.append(msg)
    return results


# per-process counts of INTERNAL tool-execution exceptions, keyed by exception
# type; consumed+reset per optimizer step by the training scheduler's [tool]
# hook (offline rollouts never reset it — their stage print can read cumulatives)
EXEC_ERR_COUNTS: dict = {}


def _fact_node(name, ctx):
    from kgqa.agent import seq_tools as st
    i = st._name_to_idx(str(name), ctx)
    if i is not None and normalize(str(ctx.ents[i])) == normalize(str(name)):
        return node_key(i)
    return 'v:' + normalize(str(name))


def _finish(child, result, live):
    from kgqa.agent import seq_protocol as sp
    manager = store(live)
    staged, req = child['ctx'], child['req']
    event = staged._adaptive_event
    if event is None:
        # Name correction, not a graph execution. No stage shift is claimed.
        return result
    event['action'] = action(child['tool'], {k: v for k, v in child['args'].items() if k != 'sg' and not k.startswith('_')})
    if child['tool'] == 'retrieve_relations':
        manager.finish(event, result=result)
    elif sp.result_failed(result):
        manager.finish(event, result=result)
        if 'NO_EVIDENCE' not in result:
            event['outcome'] = 'execution_failed'
        # Failed ledger observations are safe to retain; no delivered IDs.
        if 'NO_EVIDENCE' in result and getattr(staged, 'evidence_records', None) is not None:
            live.evidence_records = staged.evidence_records
        # FAILED-COMMAND RETENTION (user ruling 2026-10-09): a NON-repeat
        # failure still counts as an executed command — the append stays
        # published so later start-overlap checks can match it, and the
        # exact relation group is recorded so a repeat is flagged.
        if getattr(staged, 'anchor_seqs', None) is not None:
            live.anchor_seqs = staged.anchor_seqs
        if 'NO_EVIDENCE' in result and child['tool'] == 'retrieve_subgraph':
            _fs = getattr(live, '_failed_steps', None)
            if _fs is None:
                _fs = live._failed_steps = {}
            _rels = frozenset(str(r) for r in (child['args'].get('relations') or ()))
            if _rels:
                for _root in set((req.get('root_of') or {}).values()) or {None}:
                    _fs.setdefault(_root, []).append(_rels)
    else:
        commit = req.get('_ev_commit')
        raw_facts = list(getattr(commit, 'fact_keys', ()) or ())
        if commit is None or not raw_facts:
            raise RouteBoundary('EVIDENCE_LEDGER_REQUIRED: successful SG lacks full delivered fact_keys; update the ledger adapter before enabling adaptive routes.')
        facts = [(_fact_node(h, live), str(r), _fact_node(t, live)) for h, r, t in raw_facts]
        frontier = [node_key(int(i)) for i in (getattr(commit, 'frontier_node_ids', ()) or ())]
        attributes = any((getattr(o, 'lane', '') == 'attr' or getattr(o, 'role', '') == 'attr')
                         and getattr(o, 'new_to_tree', True)
                         for o in (getattr(commit, 'observations', ()) or ()))
        def display_name(key, fallback):
            i = index_of(key)
            return str(live.ents[i]) if i is not None else str(fallback)
        display_facts = [[display_name(head, h), str(r), display_name(tail, t)]
                         for (h, r, t), (head, _, tail) in zip(raw_facts, facts)]
        observations = [o.to_dict() if hasattr(o, 'to_dict') else
                        {'lane': getattr(o, 'lane', ''), 'role': getattr(o, 'role', '')}
                        for o in (getattr(commit, 'observations', ()) or ())]
        metadata = {'raw_facts': display_facts,
                    'ledger_call_id': str(getattr(commit, 'call_id', '')),
                    'frontier_node_ids': list(getattr(commit, 'frontier_node_ids', ()) or ()),
                    'observations': observations,
                    'submitted_relations': [str(staged.rels[i]) for i in req.get('rel_idxs', ())]}
        delivered_ids = {index_of(k) for h, r, t in facts for k in (h, t) if index_of(k) is not None}
        new_entities = set(getattr(live, 'subgraph_entities', ()) or ()) | delivered_ids
        new_triples = set(getattr(live, 'accumulated_triples', ()) or ()) | {tuple(f) for f in display_facts}
        new_candidates = list(dict.fromkeys(list(getattr(live, 'all_candidates', ()) or ()) +
                                           [x for h, r, t in display_facts for x in (h, t)]))
        new_consumed = set(getattr(live, 'consumed_anchors', ()) or ()) | {str(live.ents[index_of(k)]).lower() for k in event['centers']}
        # Validate and prepare all metadata before publishing route evidence.
        json.dumps(metadata, ensure_ascii=False)
        manager.finish(event, facts, frontier, result, delivered=True, attributes=attributes)
        event.update(metadata)
        # Only complete, validated delivery commits mutable working state.
        live.evidence_records = staged.evidence_records
        live.subgraph_entities = new_entities
        live.accumulated_triples = new_triples
        live.all_candidates = new_candidates
        live.consumed_anchors = new_consumed
        # SEQUENCE STATE PUBLISH — SCOPED + REVISION-CHECKED (Codex F1,
        # 2026-10-06): longest-wins over the WHOLE staged snapshot rolled
        # back a sibling route's same-depth edit (worker B's stale copy of
        # route A's layer overwrote A's just-committed relations — sequential
        # delivery suffices, no concurrency needed). Publish ONLY the roots
        # this worker actually edited (req['_seq_touched'], set by
        # _sg_prepare at birth and at every layer write); the staged value
        # for those roots was computed from the live value this worker read
        # at its own start, and per-case children finalize sequentially, so
        # the edit is the newest committed version of exactly those roots.
        # WALK-STATE PORTS publish (user ruling 2026-10-09): scoped
        # root-merge for anchor_seqs/fid_tree_roots (Codex F1: publish
        # ONLY the roots this worker edited), wholesale copy for the
        # per-case fields — all through walk_ports.publish.
        from kgqa.agent.walk_ports import publish as _wpublish
        _wpublish(staged, live, req.get('_seq_touched') or ())
        # V4 compressed groups (Codex F4): a render RESULT is transaction-
        # local — publish only when THIS call actually rendered V4 (the
        # render arm stamps _v4_render_call); a non-V4 call publishes
        # nothing, so it can neither resurrect stale groups nor clear a
        # fresher one from an earlier V4 delivery in the same episode.
        _stamp = getattr(staged, '_v4_render_call', None)
        if _stamp == event.get('call_id'):
            live._v4_compressed_groups = getattr(
                staged, '_v4_compressed_groups', None) or []
    # Terminal no-evidence results are deterministic too. Cache per route
    # and actual input version, never solely (physical root, relation).
    if event['outcome'] in ('delivered', 'no_evidence', 'relations'):
        manager.cache[child['key']] = {'call_id': event['call_id']}
        # The just-delivered version can also serve an identical re-call.
        post_key = _query_key(child['tool'], child['args'], req, live, child['ticket'])
        manager.cache[post_key] = {'call_id': event['call_id']}
    root_i = index_of(manager.routes[event['route_id']].root)
    root_name = str(live.ents[root_i]) if root_i is not None else manager.routes[event['route_id']].root
    feedback = '\nroute_root: ' + root_name
    feedback += '\nroute_state: ' + manager.status(event['route_id'])
    feedback += '\nroute_transition: ' + event.get('transition', 'continue_route')
    feedback += '\nprogress: ' + event.get('progress', event['outcome'])
    frontier = manager.routes[event['route_id']].frontier
    if frontier:
        ref = event['call_id'] if event.get('frontier') else event['route_id']
        feedback += f'\nnext_center_reference: @frontier:{ref}'
    event['result'] = result + feedback
    return event['result']


def finalize(req, results, ctx):
    return '\n\n'.join(results or [])


def parse_prepare(case, raw_response, reasoning=None, history_text=None):
    from kgqa.agent import seq_protocol as sp
    from kgqa.core.utils import strip_reasoning_leak
    try:
        clean = strip_reasoning_leak(raw_response or '')
        history = (strip_reasoning_leak(history_text, keep_think_tags=True)
                   if history_text is not None else clean)
        tool, args = sp.parse_adaptive_action(clean,
                                            known_values=getattr(case.ctx, 'all_candidates', ()) or ())
        if not tool:
            raise sp.ProtocolError('ACTION_REQUIRED: submit one tool action. No checkpoint/analysis turn is needed.')
        args, problem = validate_args(tool, args or {})
        if problem:
            raise sp.ProtocolError(problem)
        canonical = action(tool, args)
        # ROLLOT = complete raw record (user ruling 2026-10-03: credit
        # evaluation is a POST-PROCESSING step on the full trajectory in
        # its own files — rscc_credit/annotate — and does its own
        # filtering. The rollout layer records everything; the "tools
        # only" filter belongs to the credit layer, not here.)
        case.messages.append({'role': 'assistant', 'content': history})
        case.ctx.trajectory.append({'role': 'assistant',
                                    'content': canonical,
                                    'reasoning': reasoning or '',
                                    'raw': clean,
                                    'raw_full': history})
        case._last_tool_sig = None
        case._tool_repeat = 0
        return {'tool': tool, 'args': args, 'raw': canonical, '_adaptive': True}
    except sp.ProtocolError as exc:
        # FIX-① (user ruling 2026-10-06): the REJECTED output must enter
        # the conversation — previously the raw was never appended, so the
        # model never saw what failed and repeated the same non-action
        # turn until the round cap (co2 s0: 11 consecutive rejections →
        # empty answer). The rejection itself stays (workflow compliance:
        # a prose answer bypasses the answer gate); it now follows the
        # visible failed turn, and the trajectory records the raw for
        # review.
        case.messages.append({'role': 'assistant', 'content': history})
        case.ctx.trajectory.append({'role': 'assistant',
                                    'content': '(rejected: no tool action parsed)',
                                    'reasoning': reasoning or '',
                                    'raw': clean,
                                    'raw_full': history})
        message = error(f'{exc} Prior evidence is retained.')
        case.messages.append({'role': 'user', 'content': message})
        # Parse failures are diagnostic, not graph observations.
        case.ctx.trajectory.append({'role': 'tool', 'name': 'protocol', 'content': message})
        return {'ret': 'continue'}


def loop_finalize(case, prep, result):
    from kgqa.agent import seq_protocol as sp
    tool = prep['tool']
    case.messages.append({'role': 'user', 'content': f'Tool result ({tool}): {result}'})
    case.ctx.trajectory.append({'role': 'tool', 'name': tool, 'content': result})
    manager = store(case.ctx)
    case.state.n_subgraphs = sum(e['tool'] == 'retrieve_subgraph' and e['outcome'] == 'delivered' for e in manager.events)
    case.state.sg_calls = {f'{r.route_id}/{s.stage_id}': s.sg_used
                           for r in manager.routes.values() for s in r.stages}
    if tool in ('plan', 'decompose') and not sp.result_failed(result):
        case.state.state = 'RETRIEVE'
        case.state.entities = list(prep['args'].get('entities', []))
        if store(case.ctx).background:
            store(case.ctx).background[-1]['content'] = result
    if tool == 'answer' and not sp.result_failed(result):
        case.state.state = 'DONE'
        case.done = True
        return 'done'
    if tool in ('retrieve_relations', 'retrieve_subgraph') and getattr(case.ctx, 'plan_declared', False):
        case.state.state = 'RETRIEVE'
    return 'continue'


def hint(ctx):
    manager = store(ctx)
    if getattr(ctx, '_seq_turns_remaining', 2) <= 1:
        return ('This is the final available tool turn.\n'
                'advice: Submit your best-supported answer_values from the delivered '
                'evidence NOW — an empty answer is not accepted while any evidence '
                'exists. No completion declarations are needed.')
    if not getattr(ctx, 'plan_declared', False):
        return ('advice: Start with plan and question_entities. '
                'Fact/checkpoint declarations are not required.')
    lines = []
    for rid, route in manager.routes.items():
        i = index_of(route.root)
        label = str(ctx.ents[i]) if i is not None else route.root
        lines.append(f'Root {label}: ' + manager.status(rid))
    return '\n'.join(lines) + ('\nadvice: If every necessary semantic step already has an '
                               'appropriately executed pattern, its records are fixed: infer and '
                               'answer now. Otherwise execute the next unexecuted step — a useful '
                               'unused relation, a supported successor, or another starting route. '
                               'Budgets do not assert facts are false.')


def attach_result(result, ctx):
    if not adaptive_enabled():
        return result
    sidecar = store(ctx).to_dict()
    sidecar['question'] = str(getattr(ctx, 'question', ''))
    result['route_sidecar'] = sidecar
    result['credit_tools_only'] = credit_messages(sidecar)
    result['credit_tools_only_text'] = credit_text(sidecar)
    result['route_state_stats'] = {
        'routes': len(sidecar['routes']),
        'sg_attempts': sum(e['tool'] == 'retrieve_subgraph' and e['executed'] for e in sidecar['events']),
        'sg_delivered': sum(e['tool'] == 'retrieve_subgraph' and e['outcome'] == 'delivered' for e in sidecar['events']),
        'rr_attempts': sum(e['tool'] == 'retrieve_relations' and e['executed'] for e in sidecar['events']),
        'cache_replies': sum(e['outcome'] == 'cached' for e in sidecar['events']),
        'center_changes': {r['route_id']: max(0, sum(s['activated'] for s in r['stages']) - 1) for r in sidecar['routes']},
    }
    if getattr(ctx, '_seq_diagnostics', None):
        result['adaptive_diagnostics'] = copy.deepcopy(ctx._seq_diagnostics)
    result['decomposition_method'] = 'seq_adaptive_routes'
    return result
