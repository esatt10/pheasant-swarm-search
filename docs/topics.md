# Choose a topic, build the knowledge base, and use it

[Back to the README](../README.md) · [Setup](setup.md) · [Architecture](architecture.md)

Start with a question that matters to your next project decision. For example,
"What evidence should guide a remote employee onboarding program?" gives the
swarm a clearer job than "business research."

## Choose the source path

| Your project needs | Supported path today |
|---|---|
| Scientific literature | Define a topic and collect from built-in scholarly providers. |
| Research about business, education, or social questions | Use the same workflow to collect scholarly work on that subject. |
| Specific websites, internal documents, manuals, or notes | Add material through Pheasant's source tools/UI and search it there. Automatic discovery of these sources is outside this swarm's current provider set. |
| Broad web or news discovery by swarm workers | Requires a new provider adapter and suitable source-quality rules; it is not an existing command or YAML switch. |

Pheasant supports supplied folders, documents, websites, and connectors. Follow
its [source guide](https://github.com/esatt10/pheasant-kb/blob/main/docs/how-to/sources.md)
for those inputs. Material added independently will not automatically acquire
this lab's collection trace or benchmark coverage. If you plan to benchmark the
swarm, keep its experimental knowledge base dedicated until evaluation finishes.

For a project based on your own documents or selected websites:

1. Start Pheasant as described in [setup step 3](setup.md#3-start-pheasant-for-a-live-collection).
2. Use **+ Add source** in its UI to supply a supported path or URL, or use
   the document upload flow. For a Docker instance, local folders must be
   mounted into the container; uploaded documents avoid that extra step.
3. Wait for the source's background sync to finish on the Sources page.
4. Search the knowledge base for your project questions and use the cited
   passages in the [handoff below](#use-the-knowledge-in-your-next-project-phase).

That path uses Pheasant directly and needs no swarm topic YAML. It produces a
searchable project knowledge base, but no swarm run ID or collection audit.

## Define your coverage checklist

Edit `configs/topics.yaml`, created during [setup](setup.md). A topic has a
title, starting search terms, and **facets**: the parts of the question you want
the collection to cover. A facet is simply an item on your research checklist.

For a non-scientific project:

```yaml
topics:
  - id: topic-onboarding
    title: Designing an effective onboarding program for remote employees
    seed_terms:
      - remote employee onboarding
      - organizational socialization newcomer adjustment
      - virtual teams mentoring role clarity
    date_range: { from: 2015-01-01, to: null }
    facets:
      - { id: adjustment, label: "Newcomer adjustment and role clarity", weight: 2 }
      - { id: mentoring, label: "Mentoring and social connection in remote teams", weight: 2 }
      - { id: outcomes, label: "Onboarding outcomes and evaluation methods", weight: 1 }
      - { id: uncertainty, label: "Conflicting findings and limits of the evidence", weight: 1 }
```

For a scientific project, the committed
[project examples](../configs/topics.project.example.yaml) include
`topic-battery-lifetime`, with facets for aging mechanisms, temperature,
operating strategies, and uncertainty. More detailed scientific examples are
in [topics.example.yaml](../configs/topics.example.yaml).

| Field | How to choose it |
|---|---|
| `id` | A short, unique label. Use this exact value with `--topic`. |
| `title` | The research question or decision area in ordinary language. |
| `seed_terms` | A few phrases and synonyms researchers might use for it. |
| `date_range` | Starting and ending dates for the planner/provider search window; `null` leaves an end open. Inspect retained dates if a strict cutoff matters. |
| `facets` | Three to six distinct areas your next decision needs. At least one is required. Include uncertainty or conflicting evidence where relevant. |
| `weight` | Relative importance. Weight 2 counts twice as much as weight 1 in coverage; all facets still have to meet minimums for `sufficient`. |

Prefer facets such as "role clarity" and "social connection" over overlapping
labels such as "onboarding benefits" and "good onboarding." Coverage is only as
useful as this checklist. The planner can expand vocabulary, but it cannot
prove that your checklist includes every important aspect of a field.

## Choose providers and evidence minimums

Available live adapters are `openalex`, `crossref`, `arxiv`, and `pubmed`.
For onboarding, start with this field in the existing `collection` section
of `configs/experiment.yaml`:

```yaml
collection:
  providers: [openalex, crossref]
```

For the battery example, `[openalex, crossref, arxiv]` is a starting selection.
Use `pubmed` for biomedical literature. These choices describe which adapters
to try, not a guarantee of coverage or provider availability. Keep the rest of
the experiment file in place when editing a field.

The default live collection requires each facet to have at least six sources,
three independent source families, and two sources classified as peer reviewed.
These checks are controlled by fields in `stopping`:

```yaml
stopping:
  minimum_sources_per_subtopic: 6
  minimum_independent_source_families: 3
  minimum_review_or_primary_sources: 2
```

Despite the `per_subtopic` name, the coverage calculation applies these
minimums to each **facet**. The implementation infers peer-review status from
source type; it does not verify a publication's review history. The
`source_authority` metadata in the older topic examples does not override these
stopping thresholds.

Keep minimums tied to what your project needs. A low threshold makes a small
pilot easier to complete but changes what "sufficient" means. Adapting the
swarm to company pages or internal documents also requires replacing scholarly
quality assumptions with suitable checks; changing a topic title alone does
not do that.

## Collect one topic

After [live setup](setup.md), run:

```text
uv run --env-file .env pheasant-lab plan --config configs/experiment.yaml --topic topic-onboarding
uv run --env-file .env pheasant-lab collect --config configs/experiment.yaml --topic topic-onboarding
```

To use the battery example, replace `topic-onboarding` with
`topic-battery-lifetime` in both commands. One invocation collects one topic;
with no `--topic`, the first topic is selected. `plan` previews cost;
the actual model-generated research plan is produced during `collect`.

Copy the printed `run-...` ID into the audit command:

```text
uv run --env-file .env pheasant-lab audit --config configs/experiment.yaml --run RUN_ID
```

Read the facet rows and gap explanations. `sufficient` means all configured
collection conditions passed. `stopped_budget_incomplete`,
`stopped_time_incomplete`, or `stopped_round_limit_incomplete` means the saved
collection still has unmet conditions. A small pilot can be useful without
being complete; carry those gaps into the next project phase.

For a rerun with a changed topic, budget, provider list, or thresholds, start a
new collection. `--resume RUN_ID` reconnects under matching settings and skips
an already completed collection stage. It does not resume an interrupted
worker at its last individual source.

## What is saved

| Location | What you get |
|---|---|
| Pheasant knowledge base and source named in `.env` | Markdown documents with title, authors, date, identifier/URL, license metadata, and abstract text. |
| `runs/RUN_ID/raw/sources.jsonl` | Source records, discovery/admission state, and provenance. |
| `runs/RUN_ID/raw/claims.jsonl` | Extracted claims and source references. |
| `runs/RUN_ID/raw/contradictions.jsonl` | Disagreements recorded during extraction, when present. |
| `runs/RUN_ID/raw/ingest-receipts.jsonl` | Submission and index acknowledgments; later rows can supersede earlier status. |
| `runs/RUN_ID/state.json` | A checkpoint with collection results, coverage, stop reason, and reconciliation. |
| `runs/RUN_ID/run-manifest.json` | Run identity and configuration digest. |
| `runs/RUN_ID/raw/errors.jsonl` | Failures and exclusions to inspect alongside coverage. |

JSON is a structured text format; JSONL stores one JSON record per line. Raw
files are append-only history, so repeated IDs can represent later status
records. Do not count lines as unique sources or rewrite earlier rows.

The live adapters collect metadata and abstracts. Full-text URLs and license
fields may be present, but automatic full-paper acquisition is not implemented.
Pheasant receives source documents; it does not receive the local claims ledger,
benchmark answers, or evaluation reports as ordinary search content.

## Use the knowledge in your next project phase

Open Pheasant's web UI, select your knowledge base, and search for a key question.
Or connect your next agent to the same MCP endpoint using Pheasant's
[agent connection guide](https://github.com/esatt10/pheasant-kb/blob/main/docs/how-to/attach-to-coding-agent.md).
The lab has no separate interactive `search` subcommand; Pheasant provides that
interface.

Give the next agent the endpoint, knowledge-base name, source name, topic, run
ID, and audit gaps. A handoff prompt might be:

```text
We are designing a remote employee onboarding pilot.

Search the Pheasant knowledge base <your knowledge-base name>, source
project-research, for evidence about role clarity, mentoring, and social
connection. The collection came from topic-onboarding, run <your run ID>.

Draft a pilot plan with activities, assumptions, and ways to evaluate outcomes.
Cite the retrieved sources for each evidence-based recommendation. Separate
research findings from your proposed design choices. Flag conflicting findings,
missing evidence, and claims that need full-paper review.

Known collection gaps from the audit: <paste the gaps here>.
```

For battery research, the next deliverable might be an experiment plan listing
aging hypotheses, operating conditions, measurement methods, and open questions.
For onboarding, it might be a pilot program and evaluation plan. Both are
subsequent tasks using the collected knowledge; the swarm's output is the
evidence base and research record.

Before using it for that deliverable, inspect source passages and unresolved
gaps. To measure how well a fresh agent can retrieve and answer from the
collection, run the [optional evaluation](metrics.md#run-the-optional-evaluation).
