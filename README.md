# OPSD / CGSD — On-Policy Self-Distillation for Multi-hop KGQA (Anonymous Artifact)

Code, data, and workflow for reproducing the training and evaluation pipeline of
the paper. Anonymized for double-blind review.

## 1. What this is

- **Agent**: a 4-tool KGQA agent (`plan` / `retrieve_relations` / `retrieve_subgraph`
  / `answer`) operating on per-case Freebase subgraphs delivered as a
  constrained environment (relation candidates come from an embedding retrieval
  service; subgraph execution returns ground triples only — the agent never
  writes SPARQL).
- **Training**: on-policy self-distillation on top of ms-swift 4.6.0 GKD
  (colocate mode: a TP-2 vLLM engine and ZeRO-2 training share the same
  2×A100-40G GPUs). Loss: forward KL with beta=0, temperature 1.0, top-256
  renormalized support. The teacher condition is the *same base model* with a
  distilled **behavior reference** (fact-grouped script rendered from verified
  trajectories) injected into the system prompt; the student condition is the
  plain prompt. Only divergent decision tokens carry the correction signal.
- **Reference construction**: only trajectories with f1>0 are refined; the
  reference is one script per case (answer target, fact chain, per-round
  relation queries), selected by f1 + answer log-probability.

## 2. Repository layout

```
repo/                     pipeline source
  kgqa/                     agent + tool layer + RL plugins
  kgqa/rl/swift_plugins/    the colocate training plugin (scheduler, masking,
                            fuses, telemetry) — loaded via --external_plugins
  scripts/                  runners: run_batch.py, start_train.sh,
                            run_swift_opsd.sh / _server.sh, gte_api_server.py,
                            build_opsd_pool.py, build_teacher_prompts.py,
                            rescore_trajectories.py, render_trajectory.py
  config/                   opsd_train.yaml + batch profiles
  tests/                    unit test for the injection-artifact stripper
third_party/swift/        reference copies of the ms-swift 4.6.0 internals the
                          plugin integrates with (for review; the plugin runs
                          on stock ms_swift==4.6.0)
data/                     datasets (see §4)
requirements.txt          full pip freeze of the training environment
env/key_packages.txt      the load-bearing versions
```

## 3. Environment

Hardware used in the paper: 2×A100-40G (PCIe), ~32 CPU cores.

```bash
conda create -n opsd python=3.12 -y
conda activate opsd
pip install -r requirements.txt      # ms_swift==4.6.0, vllm==0.24.0,
                                     # torch==2.11.0+cu130, transformers==5.9.0,
                                     # peft==0.19.1, deepspeed==0.19.3
```

Base models (download to `models/`, or point env vars at your copies):

| role | model | env var |
|---|---|---|
| policy / backbone | Qwen3.5-4B (base) | `OPSD_MODEL` (default `models/Qwen3.5-4B`) |
| retrieval encoder | Qwen3-Embedding-0.6B | `GTE_MODEL_PATH` (default `models/Qwen3-Embedding-0.6B`) |

## 4. Data

This revision ships the **code**; the processed benchmark pkls (CWQ train/test,
WebQSP train/test, schemas below) are added in a follow-up commit — restore
commands and volume layout will appear here then. Included now:

- `data/opsd/opsd_pool.sample16.jsonl.gz` — pre-built training-pool sample
  (16 cases × 8 rows) for the §6 smoke test (`gunzip -k` it first);
- `data/cwq_processed/mask_wrong_type_ids.json` — evaluation mask list.

**Pkl row schema** (identical across CWQ/WebQSP): `id, question, q_entity,
q_entity_id_list, text_entity_list, non_text_entity_list, relation_list,
h_id_list, r_id_list, t_id_list, a_entity, a_entity_id_list` — the case subgraph
is stored as parallel head/relation/tail id arrays plus name lists; gold answers
in `a_entity`. The runtime environment (relation retrieval + subgraph execution)
is reconstructed from these arrays alone.

**Training-pool row schema** (`opsd_pool.*.jsonl`): `messages` (one sampled
trajectory), `teacher_prompt` (the rendered behavior reference),
`kgqa_sample / kgqa_pilot / kgqa_idx` (provenance keys).

## 5. Pipeline

### 5.0 Start the retrieval (GTE) server

```bash
GTE_DEVICE=balanced GTE_ENCODE_BATCH=16 \
GTE_MODEL_PATH=models/Qwen3-Embedding-0.6B \
python scripts/gte_api_server.py --port 8003 &
```

### 5.1 (Optional) Rebuild the training pool

The pool shipped in the paper was built from base-model rollouts over the
training sets; to regenerate at small scale:

```bash
# 1) rollouts: copy config/batches/rc1_36x3.yaml, set SPLIT/N_CASES/N_SAMPLES,
#    point at the train pkl, run:
python scripts/run_batch.py config/batches/<your_rollout>.yaml
# 2) pool construction (f1 gating, ranking, reference rendering are inside):
python scripts/build_opsd_pool.py reports/<rollout_results>.json \
    --pkl data/cwq_processed/train_v4_repaired.subset2000.pkl \
    -o data/opsd/opsd_pool.jsonl --samples 8
```

Reference formatting lives in `scripts/build_teacher_prompts.py`
(`render_fact_grouped`). To skip this step, smoke-test with the provided
sample pool (§6).

### 5.2 Train

```bash
bash scripts/start_train.sh 5          # positional arg = number of steps
```

Knobs live in `config/opsd_train.yaml` (`acc` rows/step, `beta`, `temperature`,
`lora_rank/alpha/plus`, `thinking_budget`, `max_num_seqs`, `vllm_util`).
Artifacts land in `outputs/<run>/` (LoRA checkpoints, per-step rollout dumps,
`f1_steps.jsonl`). Telemetry printed per step: `[f1]` rollout quality,
`[phase]` latency breakdown (rollout / encode / train), `[tool]` retrieval
layer stats, and the three behavior fuses (verbatim repetition, repeated
query, format rejections).

### 5.3 Evaluate

```bash
# serve the trained adapter as the "student" alias:
bash scripts/run_swift_opsd_server.sh
# then run the test batch (config/batches/opd_eval_test100.yaml):
python scripts/run_batch.py config/batches/opd_eval_test100.yaml
# rescore / inspect:
python scripts/rescore_trajectories.py --traj logs/<batch>.traj.jsonl \
    --out reports/<batch>.rescored.jsonl
python scripts/render_trajectory.py <trajectory_json>   # human-readable view
```

## 6. Smoke test (~10 min on 2×A100)

```bash
gunzip -k data/opsd/opsd_pool.sample16.jsonl.gz
# edit config/opsd_train.yaml: dataset: data/opsd/opsd_pool.sample16.jsonl,
#                                acc: 8   # 16 rows/step
bash scripts/start_train.sh 1
# expect: [f1] step=1 ... rep_trajs=0 art_trn=0, one LoRA checkpoint in outputs/
```

## 7. Cost reference (2×A100-40G, 256 trajectories/step)

| component | per-trajectory (amortized) |
|---|---|
| rollout generation | 4.2 s |
| retrieval layer (GTE + graph) | 0.16 s |
| training fwd/bwd (student+teacher) | 3.2 s |
| sync / offload overhead | 0.8 s |

## 8. Notes

- `data/` derives from the public ComplexWebQuestions and WebQSP benchmarks;
  only preprocessing artifacts (repair, id-array subgraph packing) are ours.
- `third_party/swift/` are verbatim copies of the installed ms-swift 4.6.0
  modules the plugin hooks into, vendored for reviewability.
