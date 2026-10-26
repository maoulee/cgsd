"""Stage 4: Relation pruning — LLM-based and reranker-based variants.

LLM pruning selects 2-4 relevant relations per decomposition step.
Reranker pruning uses Qwen3-Reranker-0.6B yes/no logit scoring.
Both include a relation preflight probe to filter unreachable relations.
"""
from __future__ import annotations

import re
import time
from typing import List

from kgqa.core.case_state import CaseState
from kgqa.core.utils import rel_to_text, extract_xml_tag, sample_triple_batch
from kgqa.llm.batch import batch_call_llm
from kgqa.traversal.cvt import is_cvt_like
from kgqa.traversal.path_utils import _is_noisy_path_relation

# ---------------------------------------------------------------------------
# Preflight probe constants
# ---------------------------------------------------------------------------
RELATION_PREFLIGHT_LEVEL0_FILTER = False
RELATION_PREFLIGHT_MAX_FRONTIER = 80
RELATION_PREFLIGHT_LARGE_FANOUT = 50


def _anchor_reachability_filter(anchor_idx, step_relations, adj):
    """Level 0: BFS from anchor, collect relations reachable within hop budget.

    Step i budget = i + 2 hops (step 0: 2, step 1: 3, step 2: 4, ...).
    Returns: (filtered_step_relations, debug_info)
    """
    n_steps = len(step_relations)
    max_budget = n_steps + 1

    # BFS from anchor, collect reachable relations cumulatively at each hop
    reachable_rel_by_hop = {}
    visited = {anchor_idx}
    frontier = {anchor_idx}
    all_reachable_rels = set()

    for hop in range(1, max_budget + 1):
        hop_rels = set()
        next_frontier = set()
        for node in frontier:
            for nb, rel in adj.get(node, []):
                hop_rels.add(rel)
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        all_reachable_rels |= hop_rels
        reachable_rel_by_hop[hop] = set(all_reachable_rels)
        frontier = next_frontier

    # Filter each step's relations against its hop budget
    filtered = []
    removed_debug = []
    for si, rs in enumerate(step_relations):
        budget = si + 2
        reachable = reachable_rel_by_hop.get(budget, set())
        kept = [r for r in rs if r in reachable]
        removed = [r for r in rs if r not in reachable]
        filtered.append(kept)
        if removed:
            removed_debug.append({"step": si, "budget": budget, "removed": removed})
    return filtered, removed_debug


def relation_preflight_probe(anchor_idx, step_relations, steps,
                             h_ids, r_ids, t_ids, ents, rels_list,
                             max_frontier=RELATION_PREFLIGHT_MAX_FRONTIER):
    """Frontier-conditioned relation preflight.

    Level 0: anchor reachability pre-filter (BFS with step-dependent hop budget).
    Level 1/2: diagnostic metrics from current frontier.
    """
    if anchor_idx is None or not step_relations:
        return step_relations, {"enabled": False, "reason": "missing_anchor_or_relations"}

    adj = {}
    for h_idx, r_idx, t_idx in zip(h_ids, r_ids, t_ids):
        adj.setdefault(h_idx, []).append((t_idx, r_idx))
        adj.setdefault(t_idx, []).append((h_idx, r_idx))

    # -- Level 0: anchor reachability pre-filter --
    l0_filtered, l0_removed = _anchor_reachability_filter(anchor_idx, step_relations, adj)
    if RELATION_PREFLIGHT_LEVEL0_FILTER:
        step_relations = l0_filtered

    def _rel_name(rel_idx):
        return rel_to_text(rels_list[rel_idx]) if 0 <= rel_idx < len(rels_list) else "?"

    def _sample(items):
        return sorted(items)[:max_frontier]

    frontier = {anchor_idx}
    committed = {anchor_idx}
    filtered_layers = []
    debug_steps = []

    for step_idx, rels_for_step in enumerate(step_relations):
        rels_ordered = list(rels_for_step)
        next_rels = set(step_relations[step_idx + 1]) if step_idx + 1 < len(step_relations) else set()
        rel_reports = []
        reachable_rels = []
        next_frontier_by_rel = {}

        for rel_idx in rels_ordered:
            hits = set()
            loop_hits = 0
            for node_idx in frontier:
                for nb_idx, edge_rel in adj.get(node_idx, []):
                    if edge_rel != rel_idx:
                        continue
                    if nb_idx in committed:
                        loop_hits += 1
                        continue
                    hits.add(nb_idx)

            cvt_hits = {idx for idx in hits if 0 <= idx < len(ents) and is_cvt_like(ents[idx])}
            normal_hits = hits - cvt_hits
            sidecar_next_hits = 0
            direct_next_hits = 0
            if next_rels:
                for hit_idx in _sample(hits):
                    for nb_idx, edge_rel in adj.get(hit_idx, []):
                        if edge_rel in next_rels:
                            direct_next_hits += 1
                            break
                    if hit_idx in cvt_hits:
                        for nb_idx, edge_rel in adj.get(hit_idx, []):
                            if edge_rel in next_rels:
                                sidecar_next_hits += 1
                                break

            rel_text = _rel_name(rel_idx)
            reachable = bool(hits)
            next_reachable = (not next_rels) or direct_next_hits > 0 or sidecar_next_hits > 0
            noisy = _is_noisy_path_relation(rel_text)
            fanout = len(hits)
            status = "PASS"
            reason = "reachable from current frontier"
            if not reachable:
                status = "BLOCK"
                reason = "unreachable from current frontier"
            elif fanout > RELATION_PREFLIGHT_LARGE_FANOUT:
                status = "WARN"
                reason = "large fanout"
            elif noisy:
                status = "WARN"
                reason = "noisy relation"
            elif next_rels and not next_reachable:
                status = "WARN"
                reason = "no next-step reachability in local probe"

            if reachable:
                reachable_rels.append(rel_idx)
                next_frontier_by_rel[rel_idx] = hits

            rel_reports.append({
                "rel_idx": rel_idx,
                "rel": rel_text,
                "status": status,
                "reason": reason,
                "hit_count": fanout,
                "normal_hit_count": len(normal_hits),
                "cvt_hit_count": len(cvt_hits),
                "loop_hits": loop_hits,
                "direct_next_hits": direct_next_hits,
                "sidecar_next_hits": sidecar_next_hits,
                "next_reachable": next_reachable,
                "noisy": noisy,
            })

        # Level 1 filter: only keep reachable relations
        if RELATION_PREFLIGHT_LEVEL0_FILTER and reachable_rels:
            accepted = [rel_idx for rel_idx in rels_ordered if rel_idx in set(reachable_rels)]
            fallback_keep_original = False
        else:
            accepted = rels_ordered
            fallback_keep_original = bool(rels_ordered and not reachable_rels)

        next_frontier = set()
        for rel_idx in accepted:
            next_frontier.update(next_frontier_by_rel.get(rel_idx, set()))
        if next_frontier:
            frontier = set(_sample(next_frontier))
            committed.update(frontier)

        filtered_layers.append(accepted)
        debug_steps.append({
            "step": steps[step_idx].get("step", step_idx + 1) if step_idx < len(steps) else step_idx + 1,
            "frontier_size": len(frontier),
            "input_relations": rels_ordered,
            "accepted_relations": accepted,
            "fallback_keep_original": fallback_keep_original,
            "relation_reports": rel_reports,
        })

    return filtered_layers, {
        "enabled": True,
        "mode": "level0_multihop_filter_level1_diagnostic",
        "level0_removed": l0_removed,
        "level0_active": RELATION_PREFLIGHT_LEVEL0_FILTER,
        "steps": debug_steps,
    }


# ---------------------------------------------------------------------------
# Stage 4: LLM relation pruning
# ---------------------------------------------------------------------------

async def stage_4_relation_pruning(session, cases: List[CaseState]):
    """Batch LLM relation pruning for all active cases."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # Build prune prompts (reuse existing logic from llm_prune_all_relations)
    prompts = []
    for cs in active:
        # Collect all relation indices that have candidates for triple sampling
        all_rel_indices = set()
        for cands in cs.step_candidates.values():
            for idx, _, _ in cands:
                all_rel_indices.add(idx)

        # Sample triples for each relation
        triple_samples = sample_triple_batch(
            list(all_rel_indices), cs.h_ids, cs.r_ids, cs.t_ids, cs.ents, cs.rels
        )

        chain_lines = []
        step_blocks = []
        for s in cs.steps:
            sn = s["step"]
            ep_str = f" -> endpoint: {s['endpoint']}" if s.get("endpoint") else ""
            chain_lines.append(f"  Step {sn}: {s['question']}{ep_str}")
            cands = cs.step_candidates.get(sn, [])
            if not cands:
                step_blocks.append(
                    f"Step {sn}: {s['question']}\n"
                    f"  Purpose: {s.get('definition', '')}\n"
                    f"  Sub-question: {s.get('subquestion', '')}\n"
                    f"  Candidates: (none)"
                )
                continue

            cand_lines = []
            for i, (idx, name, score) in enumerate(cands, 1):
                line = f"    {i}. {name}"
                triple = triple_samples.get(idx, "")
                if triple:
                    line += f"\n       e.g. {triple}"
                cand_lines.append(line)

            step_blocks.append(
                f"Step {sn}: {s['question']}\n"
                f"  Purpose: {s.get('definition', '')}\n"
                f"  Sub-question: {s.get('subquestion', '')}\n"
                f"  Candidates:\n" + "\n".join(cand_lines)
            )

        chain_text = "\n".join(chain_lines)
        blocks_text = "\n\n".join(step_blocks)

        prompt = f"""Analyze and select knowledge graph relations for each step of this reasoning chain.

Question: {cs.question}

Reasoning chain:
{chain_text}

Step-by-step candidates (each with an example triple):
{blocks_text}

Rules:
1. Each step connects FROM previous output TO next — select bridge relations
2. Select 2-5 relevant relations per step. Pick the best candidates that match the step's purpose.
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
        prompts.append([
            {"role": "system", "content": "You are a knowledge graph relation selector for multi-step QA. Analyze the full chain, then select relevant relations per step. Output <analysis> and <selected> XML tags."},
            {"role": "user", "content": prompt},
        ])

    responses = await batch_call_llm(session, prompts, max_tokens=2000)

    for ci, (cs, raw) in enumerate(zip(active, responses)):
        selected_yaml = extract_xml_tag(raw or "", "selected")
        result = {}
        if selected_yaml:
            for line in selected_yaml.split('\n'):
                line = line.strip()
                m = re.match(r'step_(\d+)\s*:\s*\[(.*?)\]', line)
                if m:
                    sn = int(m.group(1))
                    nums = [int(x.strip()) for x in m.group(2).split(',') if x.strip().isdigit()]
                    cands = cs.step_candidates.get(sn, [])
                    # Preserve LLM ranking order
                    ranked_indices = []
                    seen_idx = set()
                    for n in nums:
                        if 1 <= n <= len(cands):
                            idx = cands[n - 1][0]
                            if idx not in seen_idx:
                                ranked_indices.append(idx)
                                seen_idx.add(idx)
                    result[sn] = ranked_indices

        # Fallback: top-3 GTE for missing steps
        for s in cs.steps:
            sn = s["step"]
            if sn not in result or not result[sn]:
                cands = cs.step_candidates.get(sn, [])
                result[sn] = [idx for idx, _, _ in cands[:3]]

        # Build step_relations: LLM-selected only (no padding)
        cs.step_relations = []
        for step in cs.steps:
            sn = step["step"]
            ranked = result.get(sn, [])
            cs.step_relations.append(ranked[:5])

        cs.prune_debug = {
            "prompt": prompts[ci][1]["content"],
            "response": raw,
            "parsed_yaml": selected_yaml,
            "parsed_result": {sn: list(indices) for sn, indices in result.items()},
            "stage4_step_relations": [list(rs) for rs in cs.step_relations],
        }

        # -- Stage 4.5: Relation Preflight Probe --
        filtered_rels, preflight_dbg = relation_preflight_probe(
            cs.anchor_idx, cs.step_relations, cs.steps,
            cs.h_ids, cs.r_ids, cs.t_ids, cs.ents, cs.rels,
        )
        cs.step_relations = filtered_rels
        cs.prune_debug["preflight_step_relations"] = [list(rs) for rs in cs.step_relations]
        cs.prune_debug["preflight"] = preflight_dbg

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["pruning"] = dt / len(active)
    print(f"  Stage 4 (Relation pruning): {dt:.2f}s")


# ---------------------------------------------------------------------------
# Reranker-based pruning (alternative to LLM)
# Uses Qwen3-Reranker-0.6B: CausalLM with yes/no logit scoring
# ---------------------------------------------------------------------------

_RERANK_MODEL = None
_RERANK_TOKENIZER = None
_RERANK_PREFIX_TOKENS = None
_RERANK_SUFFIX_TOKENS = None
_RERANK_TOKEN_FALSE_ID = None
_RERANK_TOKEN_TRUE_ID = None

_RERANK_PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
_RERANK_MAX_LENGTH = 8192
_RERANK_INSTRUCT = "Given a query, retrieve the document most semantically similar to it"


def _load_rerank_model(model_path):
    global _RERANK_MODEL, _RERANK_TOKENIZER, _RERANK_PREFIX_TOKENS, _RERANK_SUFFIX_TOKENS
    global _RERANK_TOKEN_FALSE_ID, _RERANK_TOKEN_TRUE_ID
    if _RERANK_MODEL is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading reranker: {model_path} ...")
    _RERANK_TOKENIZER = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
    _RERANK_MODEL = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
    _RERANK_PREFIX_TOKENS = _RERANK_TOKENIZER.encode(_RERANK_PREFIX, add_special_tokens=False)
    _RERANK_SUFFIX_TOKENS = _RERANK_TOKENIZER.encode(_RERANK_SUFFIX, add_special_tokens=False)
    _RERANK_TOKEN_FALSE_ID = _RERANK_TOKENIZER.convert_tokens_to_ids("no")
    _RERANK_TOKEN_TRUE_ID = _RERANK_TOKENIZER.convert_tokens_to_ids("yes")
    print(f"  Reranker loaded. (yes_id={_RERANK_TOKEN_TRUE_ID}, no_id={_RERANK_TOKEN_FALSE_ID})")


def _rerank_score_pairs(pairs):
    """Score query-document pairs using Qwen3-Reranker yes/no logit method.

    pairs: list of formatted instruction strings (one per candidate)
    Returns: list of float scores (probability of "yes")
    """
    import torch
    max_content = _RERANK_MAX_LENGTH - len(_RERANK_PREFIX_TOKENS) - len(_RERANK_SUFFIX_TOKENS)
    inputs = _RERANK_TOKENIZER(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_content
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = _RERANK_PREFIX_TOKENS + ele + _RERANK_SUFFIX_TOKENS
    inputs = _RERANK_TOKENIZER.pad(inputs, padding=True, return_tensors="pt", max_length=_RERANK_MAX_LENGTH)
    inputs = {k: v.to(_RERANK_MODEL.device) for k, v in inputs.items()}

    with torch.no_grad():
        batch_scores = _RERANK_MODEL(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, _RERANK_TOKEN_TRUE_ID]
        false_vector = batch_scores[:, _RERANK_TOKEN_FALSE_ID]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
    return scores


async def stage_4_rerank_pruning(cases: List[CaseState], model_path: str, top_k: int = 5):
    """Reranker-based relation pruning using Qwen3-Reranker-0.6B (CausalLM yes/no scoring)."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    _load_rerank_model(model_path)

    for cs in active:
        cs.step_relations = []
        for step in cs.steps:
            sn = step["step"]
            cands = cs.step_candidates.get(sn, [])
            if not cands:
                cs.step_relations.append([])
                continue

            step_query = step.get('definition', '') or step.get('question', '') or step.get('relation_query', '') or cs.question
            cand_names = [name for _, name, _ in cands]

            # Format each candidate as an instruction pair for the reranker
            pairs = []
            for cname in cand_names:
                text = f"<Instruct>: {_RERANK_INSTRUCT}\n<Query>: {step_query}\n<Document>: {cname}"
                pairs.append(text)

            scores = _rerank_score_pairs(pairs)

            # Sort candidates by reranker score, keep top_k
            ranked = sorted(zip(cands, scores), key=lambda x: -x[1])
            selected_indices = [idx for (idx, _, _), _ in ranked[:top_k]]
            cs.step_relations.append(selected_indices)

        cs.prune_debug = {"method": "rerank", "model": model_path}

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["pruning"] = dt / len(active)
    print(f"  Stage 4 (Rerank pruning): {dt:.2f}s")
