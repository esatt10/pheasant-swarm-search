# CLAUDE.md — pheasant-swarm-lab

Context hand-off for any agent working on this repository. Read it first. It
describes the system **as it is now**; where this file and the code disagree,
the code is authoritative.

---

## 1. What this is

A laptop-local lab that drives a hierarchical research swarm into a
[Pheasant](https://github.com/esatt10/pheasant-kb) knowledge region, freezes an
evidence-linked benchmark, and compares five isolated arms against it. The
product question: **can a stateless agent with only Pheasant rival the
source-aware specialist that built the collection?**

The five questions it keeps apart — collection, persistence, retrieval,
answering, learning — are in `README.md`. This file is the parts a contributor
needs that the README does not carry.

## 2. Rules

1. **A metric that cannot carry its denominator reports
   `insufficient_evidence` with `value: None`.** Never `0.0`. A point that
   could not be measured is not one that measured badly.
   `MetricResult.validate()` enforces it and refuses a result with no
   `limitation`.
2. **A gate set cannot be constructed empty**, and a verdict is tri-state.
   `all([])` is `True`; `GateSet.__init__` refuses, and `PASS` requires that
   every gate was *evaluated*.
3. **Unknown is not negative.** Served, considered, included and
   `not_selected` carry weight zero. Only a caller's judgement or a
   deterministic matcher produces polarity.
4. **Isolation is code, not convention.** `arms/base.py` declares each arm's
   permitted inputs and raises. A change that hands a Pheasant arm the
   research package must fail a test, not a review.
5. **Raw JSONL is authoritative and append-only.** Corrections supersede; no
   row is ever rewritten. Every derived thing — the DuckDB projection, the
   metrics, the reports — rebuilds from it.
6. **Reserve before you spend.** `CostLedger.spend` reserves the worst case,
   runs, then reconciles. A guard that reserves an average lets a run overshoot
   on exactly the population of calls a budget exists to stop.
7. **Tool names are configured, never assumed**, and heuristic matching is off.
   `doctor` refuses before any spend.
8. **The offline suite must stay offline.** `tests/conftest.py` strips real
   credentials and points every default at something local.
9. **Two runs are comparable only when their `config_digest` matches.** The
   digest deliberately excludes *where* a run happened (output root, absolute
   config paths) — a run copied to another machine must still be resumable.
10. **The demo measures the demo.** Anything produced against the mock region
    or the `replay` provider says so in its own limitations section.

## 3. Canonical commands

```bash
./scripts/bootstrap.sh
uv run pheasant-lab demo --config configs/demo.yaml   # offline, ~12s, free
make check                                            # lint + schemas + tests
uv run python scripts/export_schemas.py               # after changing a record shape
```

## 4. Design decisions worth knowing before you change something

- **The three-phase round.** `orchestrator._run_round` runs discovery
  concurrently, **admission sequentially and round-robin across subtopics**,
  then extraction concurrently. Admission is the only phase touching the shared
  registry: which branch reaches a shared candidate first decides which
  subtopic's question drives its extraction, so scheduling decided the corpus.
  Round-robin rather than one-branch-at-a-time, because draining a broad
  subtopic first starves a narrow one and its facet reports as uncovered when
  the corpus in fact covers it.
- **`fact_f1` is computed from TP/FP/FN, not from P and R.** An abstaining arm
  has no precision denominator; driving F1 off the ratios would drop every
  abstention out of the comparison and flatter the arm that abstained by
  removing it from its own denominator.
- **`read_passages` exists for every arm.** "A citation names something this
  session read" is the question. Defining it as "something a Pheasant search
  returned" scored the specialist's every citation invalid, because it does not
  use Pheasant.
- **Only `P0` is pinned to the sealed snapshot, and only where the region
  offers a pin.** A pinned search is answered from that state *or refused*,
  and `P1`'s own treatment moves the snapshot's memory section. The snapshot
  still guards `P1`/`P2`: the drift check fails the run if any section other
  than `memory` moved. pheasant >= 0.12 exposes `snapshot_id` and a
  corpus-level `as_of` on its HTTP surface and **not** on `search_context`, so
  the shipped example config maps neither, `Retriever.supports_pinning` is
  false there, and the run records a limitation saying `P0` ran unpinned.
- **An argument absent from `argument_map` is one the lab does not send.**
  That is how a capability the region lacks is declared, and `doctor` checks
  every mapped name against the tool's advertised schema
  (`capabilities.configured_arguments`) so a map claiming something the tool
  never heard of is refused before any spend.
- **The benchmark version is content-addressed over the evidence ledger and
  the composition**, not over the run. A version carrying the run id would give
  the same question a different id in every run, and there would be no trend
  line.
- **Question wording and its matcher are disjoint by construction.** The
  builder splits a claim into subject terms (which the question names) and
  answer terms (which the matcher requires), and `_harden` rebuilds any matcher
  the question would satisfy anyway. The leakage checker is the independent
  net, not the only one.

## 5. Traps this repository has already fallen into

- **A `Retry-After: 0` is a server saying "immediately".** `retry_after or
  backoff` swallowed it and waited the configured 30s instead. Falsy-zero.
- **A recorded snapshot id that is never sent is a run that looks pinned and is
  not.** `SearchRequest.as_arguments` built every other field and dropped
  `snapshot_id` and `as_of` on the floor. Found by the contract test that
  asserts a drifted snapshot refuses; no unit test could have seen it.
- **A fixed temp filename is a collision waiting for a second writer.** The
  mock region's save used `<name>.<pid>.tmp`; two branches in one process share
  a pid, so the first rename took the file out from under the second and the
  branch died. Unique per write, and unlink on failure.
- **A check-then-write across threads is not one step.** `state.known()`
  followed by `add_source()` let two branches admit the same candidate, and two
  runs of one configuration disagreed about their own corpus by a claim or two
   — enough to move a metric, not enough for anyone to notice why.
- **Restoring a thread-local you never set writes `None` back.** `Tracer.span`
  captured `getattr(self._local, "trace_id", None)` and restored it, so every
  event after the first span carried a null trace id. Captured through the
  properties now. Found by validating a real run against the published schema.
- **A config digest that includes the output directory makes a copied run
  unresumable.** Two runs are comparable on their experiment, not on where they
  were written.
- **A question built from a claim's own salient terms contains its own
  answer.** The first builder used the same terms for the wording and the
  matcher; the leakage checker refused three questions per run, correctly.
- **`ruff format` rewrites the lines your `sed` was aiming at.** Two test edits
  silently no-oped after a format pass and left an undefined name.
- **A mock that accepts an argument the real server rejects hides the bug it
  was built to expose.** The adapter mapped `snapshot_id` and `as_of` onto
  `search_context`, which pheasant exposes on HTTP only. The mock accepted
  both, so the demo, the contract tests and the drift-refusal test all passed
  while a live run would have failed at `P0`'s first search — or, worse,
  ignored the pin and looked pinned. Two fixes, because one was not enough:
  the adapter sends only what the map declares (`pin_sent` records which), and
  preflight now checks every *configured* name rather than a static list that
  was written before the map existed.
