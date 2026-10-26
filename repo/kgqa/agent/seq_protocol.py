"""Deterministic ingress and declaration staging for SEQ only."""
from __future__ import annotations
import copy
import hashlib
import json
import re

class ProtocolError(ValueError):
    pass

_LIST_FIELDS = {'entities', 'question_entities', 'answer_values', 'center', 'relations', 'candidates'}
_TOOL = re.compile(r'^[ \t]*tool:[ \t]*(.*?)[ \t]*$', re.I | re.M)

def _repair_json(text):
    """Repair syntax outside strings only; never rewrite quoted evidence names."""
    out, i = [], 0
    while i < len(text):
        char = text[i]
        if char == '"':
            start = i
            i += 1
            while i < len(text):
                if text[i] == '\\': i += 2; continue
                if text[i] == '"': i += 1; break
                i += 1
            out.append(text[start:i])
            continue
        if char == ',':
            j = i + 1
            while j < len(text) and text[j].isspace(): j += 1
            if j < len(text) and text[j] in '}]': i += 1; continue
        if char == '?' and ''.join(out).rstrip()[-1:] in (':', '[', ','):
            match = re.match(r'\?[A-Za-z_]\w*(?=\s*[,}\]])', text[i:])
            if match:
                out.append(json.dumps(match[0])); i += len(match[0]); continue
        out.append(char)
        i += 1
    return ''.join(out)

def _json_object(text):
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try: obj = json.loads(_repair_json(text))
        except json.JSONDecodeError as exc: raise ProtocolError(f'Invalid action JSON: {exc.msg}') from exc
    if not isinstance(obj, dict) or not isinstance(obj.get('tool'), str):
        raise ProtocolError('The action object must contain a string tool and an args object.')
    if set(obj) - {'tool', 'args'} or not isinstance(obj.get('args', {}), dict):
        raise ProtocolError('Use one action object with only tool and args.')
    return obj['tool'].strip().lower(), obj.get('args', {})

def parse_action(content):
    """Parse one explicit action; fields before tool never leak into it."""
    text = (content or '').strip()
    matches = list(_TOOL.finditer(text))
    if len(matches) > 1: raise ProtocolError('Multiple tool blocks; submit exactly one action.')
    if not matches:
        bare = re.sub(r'^```(?:json)?\s*|\s*```$', '', text).strip()
        if bare.startswith('{'): return _json_object(bare)
        return None, None
    start = matches[0]
    name, tail = start.group(1).strip(), text[start.end():].strip()
    if name.startswith('{') or (not name and tail.startswith('{')):
        return _json_object(re.sub(r'\s*```$', '', (name + '\n' + tail).strip()).strip())
    if not re.fullmatch(r'[A-Za-z_]+', name): raise ProtocolError('Invalid tool name.')
    tool = name.lower()
    kv, sg_data, previous = {}, {}, None
    for raw in tail.splitlines():
        line = raw.strip()
        if not line: continue
        if line.startswith('```'): break
        if line.startswith('|') and previous in _LIST_FIELDS:
            old = kv[previous]
            if not isinstance(old, list): old = [old]
            kv[previous] = old + [x.strip() for x in line.strip('|').split('|') if x.strip()]
            continue
        m = re.fullmatch(r'([A-Za-z_][\w.]*)\s*:\s*(.*)', line)
        if not m: raise ProtocolError('Action fields must be key: value lines; put analysis before the action.')
        key, value = m[1].lower(), m[2].strip()
        if key in _LIST_FIELDS:
            if value.startswith('['):
                try: value = json.loads(value)
                except json.JSONDecodeError as exc: raise ProtocolError(f'Invalid list in {key}.') from exc
                if not isinstance(value, list): raise ProtocolError(f'{key} must be a list.')
            else: value = [x.strip() for x in value.split('|') if x.strip()]
        if key in kv and kv[key] != value: raise ProtocolError(f'Conflicting duplicate field {key}.')
        kv[key], previous = value, key
    if tool in ('plan', 'decompose'):
        args = {k: v for k, v in kv.items() if not re.match(r'^sg\d+\.', k) and not re.match(r'^[rd]\d+\.', k)}
        for key, value in kv.items():
            m = re.fullmatch(r'(sg\d+)\.(anchor|f\d+|f\d+\.covers)', key)
            if not m:
                if re.match(r'^sg\d+\.', key): raise ProtocolError(f'Unknown plan field {key}.')
                continue
            sg, field = m.groups()
            row = sg_data.setdefault(sg, {'id': sg, 'anchor': '', 'facts': [], 'covers': {}, '_facts': {}})
            if field == 'anchor': row['anchor'] = value
            elif field.endswith('.covers'): row['covers'][field.split('.')[0]] = value
            else:
                fact = [x.strip() for x in value.split('|')]
                if len(fact) != 3: raise ProtocolError(f'{key} requires head | question | tail.')
                row['_facts'][int(field[1:])] = fact
        for row in sg_data.values():
            order = sorted(row.pop('_facts'))
            if order != list(range(1, len(order) + 1)):
                raise ProtocolError(f"{row['id']} facts must be numbered f1, f2, ... without gaps.")
            row['facts'] = [[x.strip() for x in kv[f"{row['id']}.f{i}"].split('|')] for i in order]
        args['subgraphs'] = [sg_data[k] for k in sorted(sg_data, key=lambda k: int(k[2:]))]
        reqs = {}
        for key, value in kv.items():
            m = re.fullmatch(r'([rd]\d+)\.(kind|text)', key)
            if m: reqs.setdefault(m[1].upper(), {})[m[2]] = value
        if reqs: args['requirements'] = reqs
        return tool, args
    return tool, kv

def declaration_text(content):
    """Explicit declaration lines before the action; ignore fenced examples."""
    lines, fenced = [], False
    for raw in (content or '').splitlines():
        line = raw.strip()
        if line.startswith('```'):
            fenced = not fenced
            continue
        if _TOOL.fullmatch(raw): break
        if not fenced and (re.match(r'^[-*]?\s*(?:sg\d+(?:\.f\d+)?|f\d+)\s*[✓✔✗]', line)
                           or re.match(r'^\[(?:sg\d+(?:\.f\d+)?|f\d+)\s*[✓✗]', line)):
            lines.append(raw)
    return '\n'.join(lines)

def parse_adaptive_action(content, known_values=()):
    """Ignore prose/declaration artifacts, never guess an executable value.

    Identical action blocks collapse; different executable actions still fail.
    Unknown key/value fields still reach schema validation. JSON uses the same
    deterministic repair as the legacy parser. Thought text is not returned.
    """
    text = (content or '').strip()
    known = {str(v).strip().casefold() for v in known_values}
    matches = list(_TOOL.finditer(text))
    if not matches:
        return parse_action(text)
    actions = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[match.start():end].strip()
        # JSON is parsed as one complete object, not edited line by line.
        name = match.group(1).strip()
        tail = block[block.find('\n') + 1:] if '\n' in block else ''
        if name.startswith('{') or (not name and tail.lstrip().startswith('{')):
            actions.append(parse_action(block))
            continue
        kept = [block.splitlines()[0]]
        previous = None
        for raw in block.splitlines()[1:]:
            line = raw.strip()
            if line.startswith('```'):
                continue
            if re.match(r'^[-*]?\s*(?:sg\d+(?:\.f\d+)?|f\d+)\s*[✓✔✗]', line) or re.match(r'^\[(?:sg\d+(?:\.f\d+)?|f\d+)\s*[✓✗]', line):
                continue
            if name.lower() in ('plan', 'decompose') and re.match(r'^sg\d+\.join\s*:', line, re.I):
                continue  # obsolete advisory field, not an executable JOIN
            if re.fullmatch(r'[A-Za-z_][\w.]*\s*:.*', line) or line.startswith('|'):
                kept.append(raw)
                field = re.match(r'([A-Za-z_][\w.]*)\s*:', line)
                if field:
                    previous = field[1].lower()
            elif previous in _LIST_FIELDS and line and (
                    line.casefold() in known or re.fullmatch(r'[-*]\s+.+', line)
                    or re.fullmatch(r'[\w?.@/-]+', line)):
                raise ProtocolError('AMBIGUOUS_LIST: put each value on the same list or prefix its continuation with |. No partial action was executed.')
        actions.append(parse_action('\n'.join(kept)))
    first = actions[0]
    if any(action != first for action in actions[1:]):
        raise ProtocolError('MULTIPLE_ACTIONS: choose one action; different tool blocks were not executed.')
    return first

DECL_FIELDS = ('var_bindings', 'closed_facts', 'declared_facts', 'fact_bindings',
    'fact_vars', 'answer_selection', 'pattern_state', 'subgraph_entities',
    'var_joins', '_fid_alias', '_join2_side_noted', '_join2_side_flag', 'join_flag',
    'frozen_binding_flag', 'halluc_binding_flag', 'cvt_empty_var', 'fact_ids',
    'fact_texts', 'fact_key_map', 'fact_edges', '_model_anchor', 'plan_answer_type',
    'anchor_name', 'anchor_idx', 'plan_entities', 'plan_declared', '_ledger_injected',
    '_decl_wait_noted', '_analysis_pending', '_analysis_done', '_multi_tree_prompted')

def stage_context(ctx, with_declarations):
    staged = copy.copy(ctx)
    if with_declarations:
        for key in DECL_FIELDS:
            if hasattr(ctx, key): setattr(staged, key, copy.deepcopy(getattr(ctx, key)))
    return staged

def commit_context(ctx, staged):
    for key in DECL_FIELDS:
        if hasattr(staged, key): setattr(ctx, key, getattr(staged, key))

def context_scope(ctx):
    # Program identity; redisplay alone does not create a new scope.
    payload = {k: getattr(ctx, k, None) for k in ('anchor_seqs', 'fid_tree_roots', 'var_bindings')}
    def encode(obj):
        if isinstance(obj, (set, frozenset)): return sorted(obj, key=repr)
        return str(obj)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=encode).encode()).hexdigest()[:20]

def result_failed(result):
    text = (result or '').lstrip()
    if re.search(r'^(?:error|entity_error):', text, re.M): return True
    try:
        obj = json.loads(text)
        return isinstance(obj, dict) and bool(obj.get('error') or obj.get('entity_error'))
    except (ValueError, TypeError): return False


def validate_declaration_lines(content):
    """Reject malformed explicit declarations rather than silently dropping them."""
    for line in content.splitlines():
        line = line.strip()
        md = re.fullmatch(r'[-*]?\s*(sg\d+(?:\.f\d+)?|f\d+)\s*(?:[✓✔]\s+\?\w+\s*=\s*\[?[^\n]+|✗\s+(?:empty|moot|mismatch|exhausted)(?:\s*:[^\n]*)?)', line)
        old = re.fullmatch(r'\[(sg\d+(?:\.f\d+)?|f\d+)\s*(?:✓|✗\s+(?:empty|moot|mismatch|exhausted)(?::[^\]]*)?)\](?:\s+\?\w+\s*=\s*\[[^\]]+\])?', line)
        if not (md or old):
            raise ProtocolError('DECLARATION_FORMAT: use - sg1.f1 ✓ ?var = value | value, or - sg1.f1 ✗ empty/moot/mismatch/exhausted.')
        if old and '✓' in line and not re.search(r'\]\s+\?\w+\s*=\s*\[[^\]]+\]', line):
            raise ProtocolError('DECLARATION_FORMAT: a ✓ declaration needs a variable and values.')

def result_repeated(result):
    try:
        obj = json.loads(result)
        return isinstance(obj, dict) and bool(obj.get('evidence_repeat'))
    except (TypeError, ValueError):
        return bool(re.search(r'^evidence_repeat:\s*(?:true|True|1)\s*$', result or '', re.M))
