# Understand swarm and knowledge-base performance

[Back to the README](../README.md) · [Setup](setup.md) · [Topic workflow](topics.md)

Start with five questions: **Did we collect enough? Was it saved? Can we find
it? Can an agent answer from it? Do memory and query changes help?** These
measure different stages. A strong answer score cannot make up for missing
source coverage or a failed isolation check.

For a first review, read the collection gaps, ingestion status, retrieval
recall, answer facts and citations, then cost. The tables below put the report's
technical names beside their everyday meanings. All numerical examples here
are illustrative, not results measured on a real project.

## Read a score correctly

A **denominator** is what the measurement is out of: 8 supported claims out of
10 checked claims is `8 / 10 = 0.80`, or 80%. Always read both counts. A score
from two checked examples carries less evidence than the same score from 200.

| Term in the output | Meaning |
|---|---|
| `value: 0.0` | We could measure this, and its result was zero. |
| `insufficient_evidence`, `value: null` | The required evidence or denominator was missing. This is an unmeasured point. |
| `excluded` | An item could not be judged and was left out of the calculation; its reason should be visible. |
| `primary` | A measure intended to support a performance claim, subject to its evidence and gate requirements. |
| `diagnostic` | A clue for investigating behavior; read it alongside primary evidence. |
| `descriptive` | Context with no universal good/bad direction, such as how concentrated a research field is. |
| `limitation` | What the calculation cannot establish. This is part of the result. |

For example, an answer with no citations has no citation-validity denominator.
It reports insufficient evidence. An answer that cites three IDs it never read
has a measurable validity of `0 / 3 = 0`.

### Decode the first block in summary.md

The opening block is called a **health vector**: several status indicators
shown together. It is not a combined grade.

| Summary field | What it tells you |
|---|---|
| `run_status` | Workflow status. `complete` does not mean every quality check passed. |
| `collection_sufficient` | Whether every configured collection stopping condition passed. |
| `evidence_coverage` | Fraction of proof **events** with a positive or negative judgment: `1 - unknown_events / events`. This is different from topic coverage and from the fraction of questions with known evidence. |
| `pheasant_ingest_receipt_rate` | Fraction of receipt records in an accepted, indexed, or verified state under the report's folding/counting rules. Check indexing separately. |
| `P0_vs_C0_fact_f1_delta` | Change in F1 when the agent gets the collected Pheasant evidence. |
| `P1_vs_P0_memory_gain` | Change in F1 when learned memory is enabled. |
| `P2_vs_P1_tuning_gain` | Change in F1 from adapting queries on top of memory. |
| `P2_specialist_gap` | P2's F1 minus the source-aware specialist's F1. Negative means P2 scored lower. |
| `specialist_noninferiority` | Whether the configured “rivals the specialist” decision passed, using P2 when present, otherwise P1. Read its reasons below the block. |
| `negative_exposure_delta` | Change in known-negative exposure for P1 versus P0; a decrease is favorable. |
| `unsupported_claim_rate` | P1's mean unsupported-claim rate. |
| `control_regression_rate` | The control-regression gate's observed rate. |
| `hard_gates` | Combined `PASS`, `FAIL`, or `INCOMPLETE` gate verdict. |
| `cost_usd` | Total cost recorded by the lab. |

The delta fields are numerical differences; check the later classification,
paired sample count, and interval before calling one an improvement. A demo
cost of zero comes from its free replay model, not from a live cost estimate.

## 1. Collection: did the swarm gather useful coverage?

The collection audit is available immediately through `audit`. After evaluation,
`reports/collection.md` shows its hard conditions, facet table, gaps, and yield
history. The following names also identify formulas in
[collection_metrics.py](../src/pheasant_lab/evaluation/collection_metrics.py).
That helper's complete metric list is not automatically emitted by the current
CLI; the ordinary collection report uses the saved audit and stop conditions.

| Measure | In everyday language | Calculation and how to use it |
|---|---|---|
| `facet_coverage` | Did each important part of the topic get enough evidence? | Weight of facets meeting source, family, and peer-review minimums / total facet weight. Higher is better relative to your checklist; it does not validate the checklist itself. |
| `duplicate_rate` | How much effort rediscovered material already held? | Duplicate/equivalent items / acquired items. Lower usually means less repeated effort. Matching uses DOI or normalized title and can miss differently named versions. |
| `source_family_diversity` | Are we hearing from several independent groups? | Distinct source families / retained sources, with family counts and largest-family size. Interpret concentration; there is no universal target. Missing affiliations can force a weaker first-author-surname grouping. |
| `marginal_claim_yield` | Is another round still finding new information? | New eligible unique claims / claims held before the round, with a minimum base of 1. Two new claims after 20 is 10%. A falling yield can mean exhausted search terms even when coverage is poor. |
| `critical_contradiction_closure` | Did important disagreements get carried forward? | Resolved or converted-to-benchmark disagreements / critical disagreements found. Conversion preserves uncertainty; it does not settle the dispute. No critical disagreements means no rate to measure. |
| `provenance_completeness` | Can we trace each retained source back to where it came from? | Sources with required provenance / retained sources. Target 100%; complete fields do not prove the recorded information is correct. |

Example: onboarding has facet weights `2, 2, 1, 1`. If only the first two meet
minimums, coverage is `(2 + 2) / 6 = 67%`. The other two remain gaps, even if
the first facets contain many papers.

`sufficient` requires **all** configured conditions: facet minimums, complete
provenance, receipt rate, acceptable duplicate rate, handled critical
contradictions, low recent marginal yield, remaining evaluation reserve, and
no unassigned critical gap. Money, time, and round limits produce explicit
`stopped_*_incomplete` outcomes. Running out of new results alone is not proof
that the field was covered.

## 2. Persistence: did Pheasant receive and index it?

| Measure or check | In everyday language | Technical interpretation |
|---|---|---|
| `ingest_receipt_rate` | Did the server acknowledge each submitted item? | Verified/accepted receipt count under the caller's receipt policy / submitted eligible sources. Read receipt statuses; acceptance alone does not establish indexing. |
| `indexed_content_verification` | Did acknowledged items cross into the searchable index? | The collection helper takes indexed/verified item counts over sampled receipts and reports digest-comparison availability separately. An absent server digest leaves exact content equality unverified. This helper is not automatically in the default report. |
| `silent_loss` | Does a receipt claim an item that the server does not actually hold? | Reconciliation checks item identities, not just whether two totals are equal. The desired count is zero. |
| Snapshot drift | Did the evidence store change while we were comparing answers? | Compare sealed snapshot sections. Memory changes from the intended treatment are allowed; unrelated corpus/retrieval changes invalidate the controlled comparison. |

The distinction is like delivery tracking: **accepted** says the package
arrived; **indexed** says it is on a shelf where search can find it. Inspect
`raw/ingest-receipts.jsonl` and reconciliation in `state.json`. When a newer
receipt updates an earlier status, count the latest status per identity.

## 3. Retrieval: did a fresh agent find the evidence?

A **known positive** is evidence the benchmark has marked as acceptable for a
question. A **known negative** is evidence marked wrong for that question.
Everything else is unjudged. Being returned or ignored does not make an item
good or bad.

`k` is the result cutoff, normally 10. The lab builds the first `k` distinct
artifact results across an answer's search rounds in arrival order. Recorded
ranks retain arrival positions, so repeated results can affect rank gaps.

| Measure | In everyday language | Calculation / direction |
|---|---|---|
| `query_evidence_coverage` | Can this question's retrieval be judged at all? | 1 if it has a known-positive set, otherwise 0. Describes benchmark coverage, not retrieval quality. |
| `known_positive_hit_at_k` | Did any known useful evidence appear? | 1 for at least one known positive among the first `k` results, otherwise 0. Higher is better. |
| `known_positive_recall_at_k` | How much of the known useful evidence appeared? | Known-positive identities reached / known-positive set size. Higher is better. Unlisted useful evidence is outside this denominator. |
| `known_positive_reciprocal_rank` | How soon did useful evidence appear? | `1 / first positive rank`; 0 if none appears. Rank 1 gives 1; rank 4 gives 0.25. Higher is better. |
| `negative_exposure_at_k` | How much known unsuitable material was shown? | Known negatives / returned results in the cutoff. Lower is better. Unjudged results do not count as negative, so this can understate unsuitable exposure. |
| `evidence_weighted_dcg` | Does the ranking put judged useful material early and unsuitable material later? | Sum of `relevance / log2(rank + 1)`, divided by ideal positive gain. Relevance is +1, -1, or 0 for positive, negative, or unjudged. Diagnostic; higher is better under these judgments, and values can be negative. |
| `pairwise_proof_accuracy` | Was judged-good evidence ranked above judged-bad evidence? | Correctly ordered judged pairs / judged pairs. Only pairs with explicit opposite judgments count. A helper exists in code; the normal evaluation engine does not currently emit it. |

Example: of four known useful sources, three appear in the first ten results.
Recall is 75%, and hit rate for this question is 1. If the first useful item is
fourth, reciprocal rank is 0.25. These describe different aspects of the same
search. None establishes that the other seven results were bad.

## 4. Answering: did the answer use the evidence well?

A **required fact** is a benchmark expectation with a deterministic matcher.
An **eligible claim** is a factual assertion in the answer that the scorer
counts. An **abstention** is the agent saying it cannot answer from its evidence.

| Measure | In everyday language | Calculation / direction |
|---|---|---|
| `fact_recall` | Did the answer include the required facts? | Matched required facts / eligible required facts. Higher is better. |
| `fact_precision` | How much of what it asserted could the scorer verify? | Claims matching an expected fact or supported by a cited passage / eligible claims. Higher is better; unverifiable does not necessarily mean false. |
| `fact_f1` | How well did it balance finding required facts and avoiding unsupported extras? | `2TP / (2TP + FP + FN)`. Higher is better. See the example below. |
| `evidence_support_rate` | Are its assertions supported by passages it cites? | Supported claims / eligible claims. Higher is better. |
| `citation_validity` | Do citations point to material the agent actually read in this session? | Valid citation occurrences / citation occurrences. Higher is better. A valid citation can still be attached to an unsupported claim. |
| `unsupported_claim_rate` | How much did it assert without cited support? | `1 - evidence_support_rate`. Lower is better. An uncited expected fact may pass fact precision but still lack cited support. |
| `abstention_accuracy` | Did it decline to answer when the benchmark expected it to? | 1 when abstention behavior matches the question's answerability, otherwise 0; aggregated over questions. Higher is better. |
| `contradiction_handling_accuracy` | Did it acknowledge both sides of a disagreement? | On contradiction questions, checks disagreement language and the required positions. Higher is better; it is a deterministic wording check. |
| `temporal_validity` | Did it satisfy the question's date/version checks? | On temporal questions, checks expected facts and citations against the benchmark's partial known-negative set. Higher is better; this is not a fresh date check on every citation. |
| `answer_completeness_by_facet` | Did it cover facts in the question's named areas? | Matched required facts within the question's facets / required facts in those facets. Higher is better. The current row combines those facets, rather than emitting a score for each one. |

For F1, **TP** is required facts matched, **FN** is required facts missed, and
**FP** is eligible returned claims that neither match a required fact nor pass
the support test. Suppose four facts are required: three are returned and one
is missed, with one unsupported extra assertion. F1 is
`2 × 3 / (2 × 3 + 1 + 1) = 0.75`.

If an answer abstains on an answerable question, precision has no denominator,
but F1 can still be zero because required facts were missed. The scorer computes
F1 from counts so these abstentions remain in comparisons.

These checks are deliberately limited. Fact matching is largely lexical; an
unlisted paraphrase may be missed. Support checks use token containment in a
cited passage, which does not establish that the passage logically supports
the claim. **Treat a good score as evidence about these checks, not expert
certification that every statement is true.**

## 5. Learning: do changes help on questions kept aside?

An **arm** is a version of the answering setup. A **cohort** is a group of
questions with a particular evaluation role.

### The five comparison arms

| ID | Everyday description | Inputs and behavior |
|---|---|---|
| `S0` | The researcher with its original notes | Sources and research trace; reference performance, scored by the same rules and capable of mistakes. |
| `C0` | An agent answering from what its model already knows | No collected sources and no Pheasant tools. |
| `P0` | A fresh agent using the collected knowledge base | Pheasant retrieval with memory and steering off. |
| `P1` | The Pheasant agent with learned memory | Memory derived from the learned cohort's first pass; corpus otherwise held stable. |
| `P2` | The Pheasant agent with memory and adapted queries | Query strategy derived from learned first-pass retrieval behavior; fresh answers generated afterward. |

`P0 - C0` asks what the collection adds. `P1 - P0` asks what memory adds.
`P2 - P1` asks what query adaptation adds when memory is already enabled.
`P1 - S0` and `P2 - S0` compare the fresh agent with the source-aware reference.
This five-arm design does not separately test query tuning with memory off.

| Cohort | Why keep it separate? |
|---|---|
| `learned` | First-pass behavior on these questions may create memory or query changes. Improvement here can include memorization. |
| `temporal_holdout` | Kept out of treatment creation. Improvement here is evidence of transfer within this benchmark, not proof of future real-world performance. |
| `anchor` | Stable reference questions for comparison. |
| `control` | Questions monitored for unintended changes. |
| `invariant` | Questions whose expected behavior should remain stable. |

The cohort name `temporal_holdout` does not itself make a question an independent
future dataset. Inspect how the frozen benchmark constructed and assigned it.

### Interpret the comparison

Paired comparisons use the same questions answered by both arms. Missing or
unscorable pairs are excluded and pairing coverage is reported. The live
example requires at least 12 paired questions and 80% pairing coverage. A small
pilot or small cohort can therefore produce `insufficient_evidence`.

| Output | Plain-language meaning |
|---|---|
| Absolute delta | Treatment minus baseline. A rise from 0.60 to 0.70 is +0.10, or **10 percentage points**. Read raw proportion units even where a report labels the delta as percentage points. |
| Relative delta | Change divided by baseline magnitude: the same rise is about +16.7%. Near-zero baselines report `not_applicable`. |
| Practical threshold | The configured minimum difference that matters for the project; commonly 0.05 for proportion metrics in the example. |
| Paired bootstrap interval | A range obtained by resampling question pairs. An interval spanning zero leaves the direction uncertain under the configured rule. |
| Wins / ties / losses | Counts of positive, zero, and negative paired deltas. The current effect-size helper uses raw signs; for a lower-is-better metric, interpret a negative delta as favorable. |
| `cohens_dz` / `dz` | Mean paired change divided by the standard deviation of paired changes. Describes change relative to variability; undefined when that variability is zero. Read the raw deltas and sample count alongside it. |
| McNemar / Wilcoxon tests | Statistical checks on paired outcomes or paired score differences. Their sample-size requirements still apply. |
| Multiple-comparison correction | Adjusts statistical evidence when many comparisons are tested, reducing chance findings. The configured method is Benjamini-Hochberg. |
| `improved` / `regressed` | A qualifying favorable/adverse change. Improvement checks the configured interval rule; regression can be triggered by an adverse threshold or a hard gate failure without an interval excluding zero. Read the stated reasons. |
| `unchanged` / `mixed` | No qualifying overall change, or conflicting results such as improvement alongside a protected subgroup regression. |
| `not_comparable` | The comparison failed a comparability requirement; do not read its difference as an experimental finding. |

**“Rivals the specialist”** means the lower paired confidence bound is within
the configured acceptable loss, overall and on the required holdout, with
required gates passing, cost within budget, and no protected subgroup
regression. The example margins are 0.05 for F1/support and 0.03 for citation
validity. A slightly higher average alone is insufficient. Latency is reported
separately; the current specialist decision does not enforce a latency limit.

## Cost, speed, and reliability

Operational scores are in `metrics/aggregates.json` under `operational`.
Reliability and stage timing appear in `reports/errors-and-retries.md`.

| Measure | Meaning and limitation |
|---|---|
| `cost_per_verified_source` | Total recorded run cost / sources counted as verified by the receipt policy. Includes evaluation cost, not just collection. |
| `cost_per_benchmark_question` | Total recorded run cost / benchmark questions. More questions can lower this ratio without making research cheaper. |
| `cost_per_proven_successful_answer` | Intended to relate cost to deterministic success. The current engine supplies the number of deterministic validation-pass **events**, not distinct successful answers; multiple claims can contribute multiple events. Read it as cost per pass event until that denominator is changed. |
| `answer_latency_p95_ms` | The 95th percentile of recorded answer wall time: roughly 95% of measured answers completed within it. Includes model and local work, not only Pheasant server time. Lower is faster. |
| `token_usage` | Total reported input/output tokens, with breakdowns. Token counts measure processing volume, not correctness; replay tokens are estimates. |
| MCP success / partial / failure rates | Fractions of recorded tool calls in each status. Partial results remain separate because missing content can affect later scores. |
| MCP retry rate | Extra recorded attempts / logical calls. Read with the error log and attempt-recording behavior; it is not an answer-quality measure. |
| Stage p50 / p95 / maximum | Median, slower-tail, and worst observed duration for a recorded operation, with its sample count. Helps locate waiting time. |

## Gates: checks an average cannot hide

A **gate** is a required check. `PASS` means every gate in the set was evaluated
and passed. `FAIL` means an evaluated gate failed. `INCOMPLETE` means at least
one required check could not be evaluated and no failure already decides the
set. Skipping a check does not turn it into a pass.

| Configured gate | Question it asks |
|---|---|
| `acl_leak` | Did an answer/search expose content the caller was not allowed to access? ACL means access-control list. |
| `stale_memory_leak` | Did superseded memory appear when only current memory should be used? |
| `as_of_correctness` | Were the tested time restrictions respected? |
| `abstention` | Did the agent handle answerable and unanswerable questions adequately? |
| `known_positive_exclusion` | Did a filter exclude evidence known to be needed? |
| `control_regression` | Did too many control questions get worse? |
| `negative_exposure_increase` | Did exposure to known unsuitable evidence rise beyond the allowed amount? |
| `benchmark_leakage` | Did answer keys or treatment-building information contaminate the comparison? |
| `ingest_receipt_rate` | Were enough submitted items acknowledged? |
| `silent_loss` | Does reconciliation find items claimed but missing? |
| `snapshot_drift` | Did unexpected parts of the reference state change? |
| `budget` | Did recorded spend exceed the allowed budget? |

See [metrics.example.yaml](../configs/metrics.example.yaml) for thresholds and
each gate row's evidence for what was actually exercised. Listing a gate in
configuration does not establish that the run had evidence to evaluate it.
The offline mock commonly leaves ACL and stale-memory checks incomplete.

## Evidence behind the numbers

Every `MetricResult` carries its formula, operands, exclusions, supported claim,
unsupported claim, and limitation. `raw/proof-events.jsonl` records typed
judgments. Merely serving, considering, including, or not selecting a result
carries no positive/negative weight. Where weighted proof is summarized,
directness, independence, specificity, and recency multiply the weight;
positive and negative totals and conflict remain separately visible.

`positive_weight` and `negative_weight` sum the weights of favorable and adverse
judgments. `net_weight` is their difference; zero can mean no judgments or
equally strong conflicting judgments, so read both totals. `conflict_rate` is
targets with both kinds of judgment divided by judged targets. It is undefined
when there are no judged targets.

The benchmark is fixed before evaluation and checked for leakage. A fact with
no deterministic matcher is an exclusion, not an answer failure. Question text
must not satisfy its own answer matcher, and held-out questions must not create
the memory or query rules used to test them. Preserve those boundaries when
editing prompts or extending the benchmark.

The configuration mentions `token_f1`, `embedding_alignment`, and
`model_judge_score` as diagnostics. The current engine does not implement or
emit those scores. A configured name is not a measured result.

## Run the optional evaluation

After collection, replace `RUN_ID` with its printed identifier and use the same
config, environment, and overrides throughout:

```text
uv run --env-file .env pheasant-lab freeze-benchmark --config configs/experiment.yaml --run RUN_ID
uv run --env-file .env pheasant-lab evaluate --config configs/experiment.yaml --run RUN_ID --arms S0,C0,P0,P1,P2
uv run --env-file .env pheasant-lab report --config configs/experiment.yaml --run RUN_ID
uv run --env-file .env pheasant-lab verify --config configs/experiment.yaml --run RUN_ID
```

Inspect receipts and indexing before evaluating. If a stage refuses, read its
stated reason; freezing an answer key alone does not establish searchability.
For the optional DuckDB projection:

```text
uv run --env-file .env pheasant-lab replay --config configs/experiment.yaml --run RUN_ID
```

`replay` rebuilds the projection from existing raw records; it does not rerun
the agents or recalculate scores. `report` renders saved evaluation results.
The deterministic **model provider** named `replay` is a separate concept.

Start with `reports/summary.md`, then use:

| File under the run directory | What to inspect |
|---|---|
| `reports/collection.md` | Coverage gaps and why collection stopped |
| `reports/arm-comparison.md` | Paired differences, counts, uncertainty, and cohort results |
| `reports/worst-regressions.md` | Questions that got worse |
| `reports/errors-and-retries.md` | Failures and operational timing |
| `reports/refinement-candidates.md` | Suggested changes supported by the run; review before applying |
| `metrics/per-query.jsonl` | Individual answer/retrieval formulas and evidence references |
| `metrics/paired-deltas.csv` | Comparisons for inspection in a spreadsheet |
| `metrics/gates.json` | Each required check and its evaluation status |

For a demo run, use `--config configs/demo.yaml` and omit uv's `.env` option.
Real performance claims require live evidence, sufficient paired samples, and
applicable gates that were actually evaluated.
