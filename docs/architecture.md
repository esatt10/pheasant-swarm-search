# Swarm architecture

[Back to the README](../README.md) · [Setup](setup.md) · [Performance guide](metrics.md)

The hierarchy is **orchestrator → planner → research branches**, with a coverage
audit after each round. It runs in one Python process. The workers make model
and provider calls; they do not need a message broker, an agent server, or a
distributed cluster. Pheasant is the separate service that retains the source
material and serves later searches.

## Roles and responsibilities

| Component | Implementation | Responsibility |
|---|---|---|
| Orchestrator | [orchestrator.py](../src/pheasant_lab/orchestration/orchestrator.py) | Run rounds, bound concurrency, request replanning, and record the stop decision. This is a Python state machine. |
| Planner | [planner.py](../src/pheasant_lab/orchestration/planner.py) | Ask a model for subtopics, facet assignments, alternative terminology, and provider/date choices. It generates search instructions. |
| Researcher | [researcher.py](../src/pheasant_lab/orchestration/researcher.py) | Discover candidates, apply admission rules, extract claims from abstracts, and submit source documents. |
| Shared registry | [state.py](../src/pheasant_lab/orchestration/state.py) | Track sources, duplicates, claims, disagreements, and round history. |
| Coverage auditor | [auditor.py](../src/pheasant_lab/orchestration/auditor.py) | Compute facet coverage and source-quality gaps. An optional model narrates the computed result. |
| Stop evaluator | [stopping.py](../src/pheasant_lab/orchestration/stopping.py) | Check explicit conditions and return `sufficient`, `continue`, or an incomplete stop reason. |
| Cost ledger | [budget.py](../src/pheasant_lab/budget.py) | Reserve estimated maximum call cost and reconcile usage against run and role budgets. |
| Pheasant adapter | [pheasant/](../src/pheasant_lab/pheasant/) | Discover configured MCP capabilities, submit documents, read receipts, synchronize indexing, and retrieve evidence. |

Model settings include `orchestrator` and `benchmark_builder` roles, but the
current collection coordinator and question construction are deterministic
Python code. The benchmark recording step does make a model call and records
its cost; its response does not replace the constructed questions. A configured
role name does not imply another autonomous agent is launched. The `agents`
package extra is not required for this workflow.

## A research round has three phases

1. **Discover in parallel.** Each branch asks its selected literature providers
   for candidates using a ladder of search terms. This phase does not modify
   the shared source registry.
2. **Select in a fixed order.** Branches take turns presenting one candidate at
   a time. The registry applies source-type, identity, available-text, and
   duplicate checks. This is sequential because the first branch to retain a
   shared source determines which question drives its extraction.
3. **Extract and submit in parallel.** Each branch owns its selected records,
   asks the researcher model for claims and disagreements, and sends source
   metadata and abstract text to Pheasant as Markdown.

Taking turns during source selection prevents a broad branch from consuming
all the shared candidates before a narrower branch gets a chance. Separating
selection from concurrent network work also prevents thread timing from
changing source ownership across otherwise identical runs.

After each round, the auditor checks coverage, concentration, duplication,
disagreements, and new-claim yield. When conditions allow more work, the planner
widens the search toward gaps. The loop stops when every sufficiency condition
passes or a budget, runtime, or round limit is reached.

## Settings that control the shape and size

These are the defaults in [experiment.example.yaml](../configs/experiment.example.yaml):

| Setting | Default | Meaning in the current implementation |
|---|---:|---|
| `collection.max_research_agents` | 6 | Maximum branches scheduled in a round. |
| `collection.max_concurrent_agents` | 3 | Maximum threads working concurrently in discovery or extraction/submission. |
| `collection.max_depth` | 2 | Maximum outer research rounds, including replanning. This is not arbitrary recursive agent spawning. |
| `collection.max_search_rounds_per_agent` | 8 | Upper bound on each branch's query ladder. Actual queries can be fewer. |
| `collection.max_sources_per_subtopic` | 30 | Candidate/admission bound for a branch in a round; not a whole-run source cap. |
| `experiment.cost_budget_usd` | 10 | Total budget tracked by the lab's cost ledger. |
| `experiment.runtime_budget_minutes` | 120 | Time bound checked at collection decisions; not an interrupt for every in-flight call. |
| `stopping.evaluation_budget_reserve_fraction` | 0.40 | Share collection preserves for later evaluation. Separate per-bucket allocations also apply. |

More workers can reduce waiting for external services, but also increase calls,
overlap, and cost. More sources can improve coverage while increasing
redundancy. Use the audit to decide which limit to change for a new run.

## What crosses the Pheasant boundary

```text
Scholarly providers
  → candidates with metadata and abstracts
  → source selection and claim extraction
      → Pheasant: Markdown source documents + provenance metadata
      → local raw logs: source history, claims, contradictions, errors

Pheasant document submission
  → accepted receipt (stored)
  → source sync (indexing)
  → acknowledgment (index barrier crossed)
  → reconciliation (submitted items versus held items)
  → later search results with source locators
```

An accepted receipt alone does not prove searchability. After collection the
CLI requests sync, acknowledgment, and reconciliation where configured tools
are available. Evaluation uses receipt and reconciliation evidence in its gates;
inspect indexed status separately from acceptance. Missing capabilities limit
what can be verified and must remain visible.

Extracted claims are used by the local benchmark builder. The searchable
document is built from the source's metadata and abstract; it is not an
agent-written synthesis of all claims. Full-paper acquisition is not currently
implemented, even though configuration and records include licensing and
full-text fields.

Tool names and argument spellings are in
[pheasant-mcp.example.yaml](../configs/pheasant-mcp.example.yaml).
`doctor` checks mapped names against advertised schemas. An omitted argument
is not sent. The shipped search map omits `snapshot_id` and `as_of`, so the
default live adapter does not pin searches even if a newer server offers that
capability. A custom map must match the connected server's actual schema.

## Optional evaluation and isolation

The benchmark builder derives questions, matchable expected facts, and expected
source references from the local collection. Freezing writes content digests
before any comparison arm answers. The [five arms](metrics.md#the-five-comparison-arms)
receive different permitted inputs; [arms/base.py](../src/pheasant_lab/arms/base.py)
rejects prohibited constructor inputs. Pheasant arms do not receive the research
package or answer key.

Only `P0` attempts a snapshot pin, and only if the configured adapter supports
it. `P1` introduces memory and `P2` adapts queries; the run checks the sealed
snapshot for changes outside the memory section. A snapshot records a reference
state and supports drift checks. It does not create an independently queryable
historical copy of the whole corpus.

Benchmark files, answers, reports, and refinement candidates stay outside the
ordinary retrieval source. Memory and tuning are derived only from the
designated learned cohort, so the held-out cohort can test transfer to questions
that did not create those changes.

## Traceability and reproducibility

```text
run → branch → discovery → source → claim → ingest request → receipt
    → index/snapshot → question → arm search → passage → answer/citation
    → proof event → metric operands → comparison → report
```

Raw JSONL is authoritative and append-only. Later status records supersede
earlier ones; readers fold them by identity. `state.json` is a checkpoint, and
DuckDB is a disposable projection of raw events. The `replay` CLI command
rebuilds that projection; it does not call models or regenerate evaluation
answers and scores.

Runs are directly comparable only when their `config_digest` matches. The
digest excludes machine-specific paths and output location. Topic definitions,
models, prices, prompts, and evaluation settings are part of the reproducibility
record; changing them requires explicit accounting, not silently resuming.

## Extend the swarm

For another discovery source, implement `LiteratureProvider.search` and return
`SourceCandidate` records, then register the adapter in
[providers/base.py](../src/pheasant_lab/providers/base.py). Preserve identifiers,
dates, text availability, source type, and provenance. For non-scholarly sources,
also review admission rules, source-family identity, authority checks, prompts,
and benchmark assumptions. Add offline fixtures and contract tests for the new
path so a permissive mock cannot conceal a live API mismatch.

Changing a server tool name usually needs only a capability-map edit. Changing
a persisted record shape also requires regenerated schemas. The repository's
[contributor instructions](../CLAUDE.md) describe the invariants and required
checks.
