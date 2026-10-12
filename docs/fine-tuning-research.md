# Fine-tuning research search

The executable question set is [topics.fine-tuning.yaml](../configs/topics.fine-tuning.yaml):
seven topics, 58 research questions and 42 coverage facets. All seven topics
target the Pheasant source `swarm-lab-fine-tuning` through
[experiment.fine-tuning.yaml](../configs/experiment.fine-tuning.yaml) and
[pheasant-fine-tuning.yaml](../configs/pheasant-fine-tuning.yaml). Hosted role
models are pinned in [models.fine-tuning.yaml](../configs/models.fine-tuning.yaml).

| Order | Topic ID | Research decision or deliverable |
| --- | --- | --- |
| 1 | `topic-ft-scope-objectives` | Decide whether to tune, what to change, and which objective fits. |
| 2 | `topic-ft-data-pipeline` | Define raw, curated and annotated records, provenance, splits and versions. |
| 3 | `topic-ft-example-to-tensor` | Show exact serialization, token tensors, masks and collator invariants. |
| 4 | `topic-ft-algorithms` | Derive full and parameter-efficient update rules and memory costs. |
| 5 | `topic-ft-preference-reinforcement` | Specify preference pairs, reward data and alignment training state. |
| 6 | `topic-ft-training-system` | Specify distributed state, checkpoints, recovery, registry and serving. |
| 7 | `topic-ft-evaluation-release` | Design holdouts, paired tests, regression gates and rollback. |

## Evidence and output expectations

For every algorithmic claim, ask for the objective or update rule, assumptions,
the required record or tensor shape, compute and memory implications, a worked
example, and an empirical comparison with an untuned or simpler baseline.
Record the model family, tokenizer and template version, dataset version,
training split, hardware, implementation version, and evaluation protocol.
Distinguish method papers, independent comparisons and tool documentation;
flag results that have no independent replication or use contaminated tests.

The search covers concrete structures such as `prompt`/`completion` and
`prompt`/`chosen`/`rejected` records, conversation messages, token IDs,
attention and loss masks, sample weights, dataset manifests, optimizer states,
checkpoint shards and model-registry links. These are research targets, not a
prescribed universal schema. Current TRL documentation, for example, defines
different accepted [SFT dataset formats](https://huggingface.co/docs/trl/sft_trainer)
and [DPO preference formats](https://huggingface.co/docs/trl/dpo_trainer).

Starting primary sources for method discovery include the original
[LoRA paper](https://arxiv.org/abs/2106.09685),
[QLoRA paper](https://arxiv.org/abs/2305.14314), and
[DPO paper](https://arxiv.org/abs/2305.18290). These are seeds for the swarm to
test against later comparisons, not predetermined answers.

## Run sequence in Docker

Run each topic separately. The CLI defaults to the first topic when `--topic`
is omitted. `collect` researches and ingests into the same configured source;
`run` additionally freezes a benchmark and evaluates all five arms.

The existing Docker UI at port 8770 watches `pheasant-swarm-lab_lab-runs`.
Run the collector in a separate container sharing that exact volume and the
lab's local-data volume. The fleet API currently accepts
`host.docker.internal:8765` in its host allow-list. This command uses the
existing UI container's volumes and keeps the UI online:

```powershell
docker compose -f docker-compose.fleet.yml build lab
docker run -d --name pheasant-swarm-fine-tuning-worker `
  --network pheasant-swarm-lab_default `
  --volumes-from pheasant-swarm-lab-lab-1 `
  --env-file .env `
  -e PHEASANT_MCP_URL=http://host.docker.internal:8765/mcp `
  --entrypoint python pheasant-swarm-lab:local `
  -u scripts/run_fine_tuning_collection.py
```

The runner calls `collect` for all seven topic IDs in table order. Each topic
has its own run budget and directory, while the source name stays fixed.
Its status and per-topic logs are in `/app/runs/.fine-tuning-collection/`, in
the shared volume. For an evaluation of one topic, use `run --topic <id>`
instead of `collect`.

This configuration uses scholarly indexes, caps any one provider's contribution
per query, and prefers original full text with an abstract fallback.
With Brave and Tavily credentials, `--set collection.profile=balanced` adds
tool and vendor documentation, provided the region and model settings pass
`doctor`. The live region must advertise binary submission because this
configuration maps `content_encoding` for original PDFs.

Before a live search, supply `OPENAI_API_KEY` and `PHEASANT_API_TOKEN` in the
process environment. The CLI reads these secrets from the environment even
when `.env` supplies values for configuration interpolation. The source is
registered by successful ingestion, not by writing these files.
