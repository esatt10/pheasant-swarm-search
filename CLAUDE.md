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
uv run pheasant-lab demo --config configs/demo.yaml   # offline, ~20s, free
uv run pheasant-lab run --config configs/demo.yaml    # the supervised pipeline; --resume <run> carries one on
uv run pheasant-lab draft-topic --config configs/demo.yaml --intent "…"  # draft a topic, writes nothing
make check                                            # lint + schemas + tests
uv run python scripts/export_schemas.py               # after changing a record shape
make ui && uv run pheasant-lab serve                  # the console, http://127.0.0.1:8770
uv run pheasant-lab mcp                               # the console's tools over MCP stdio (/mcp over HTTP)
uv run pheasant-lab settings --section models         # every setting, explained
docker compose up -d --build                          # lab + a pheasant 0.13.11 region (docs/docker.md)
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
  and `P1`'s own treatment moves the snapshot. pheasant >= 0.12.6 accepts
  `snapshot_id` and `as_of` on `search_context` (before that, HTTP only), so
  the shipped example config maps both and `P0` is pinned; against an older
  region, drop them from the map and `Retriever.supports_pinning` goes false
  and the run records that `P0` ran unpinned. The drift check runs once,
  before any arm, so it guards the freeze-to-evaluate gap. In pheasant a
  memory record is an indexed artifact: a memory write moves `corpus`,
  `graph` and `retrieval` too, never `memory` alone, so a second `evaluate`
  after `P1` has seeded is refused - correctly.
- **Pheasant's submission contract has three steps, and the lab owns two.**
  `submit_documents` lands bytes in a directory and receipts them; the caller
  registers that directory as a `document_folder` source (`Ingestor.register`,
  which needs the region's state path allow-listed); a sync indexes it; and
  `acknowledge_ingest` promotes receipts, answering with *counts* while the
  receipts themselves are read back through `get_ingest_status`. On a fleet
  the sync is published, so the barrier polls `still_accepted` rather than
  trusting one call.
- **A hit's text is a preview; the answerer reads documents.** Pheasant caps
  a hit at a 500-character `text_preview`, and the lab's documents open with
  front matter, so `Retriever.hydrate` reads each passage back whole through
  `get_file_summary` and records `text_source`. The join from a hit to the
  lab's source id is the receipts (`artifact_sources`), never
  `provenance.source_id`, which is the *region's* source name.
- **An argument absent from `argument_map` is one the lab does not send.**
  That is how a capability the region lacks is declared, and `doctor` checks
  every mapped name against the tool's advertised schema
  (`capabilities.configured_arguments`) so a map claiming something the tool
  never heard of is refused before any spend.
- **The benchmark version is content-addressed over the evidence ledger and
  the composition**, not over the run. A version carrying the run id would give
  the same question a different id in every run, and there would be no trend
  line.
- **A collection profile supplies defaults, never overrides.**
  `collection.profile` (`scholarly` | `web` | `balanced`, `profiles.py`) fills
  only the `collection`/`stopping` keys the file leaves unset, which is why
  `experiment.example.yaml` states none of them. "Authoritative" is the
  profile's `authoritative_source_types`, not peer review: under `web` a
  press piece counts toward sources and families but never toward
  `minimum_review_or_primary_sources`. `balanced` sets
  `max_results_per_provider`, because without it the first provider in the
  list fills every subtopic and the second is never asked.
- **Graph expansion is recorded, never read, and declared like the pin.**
  `replay.graph_expansion` is sent only when the pheasant file maps `expand`
  (`expand_sent` records which; the shipped file maps it, for pheasant >=
  0.13.1, and an older region unmaps it with `--set
  argument_map.search.expand=null`), and the neighbourhood lands on each search
  record as `graph_neighbors`. No arm reads it: handing answerers neighbours
  changes what `P0` measures, and on this corpus a depth-2 walk reaches other
  papers through shared authors and venues, decoys included. That is a design
  decision with its own evidence, not a side effect of an adapter update.
- **Question wording and its matcher are disjoint by construction.** The
  builder splits a claim into subject terms (which the question names) and
  answer terms (which the matcher requires), and `_harden` rebuilds any matcher
  the question would satisfy anyway. The leakage checker is the independent
  net, not the only one.
- **The console is a fold and a launcher, never a second runtime.**
  `console/projection.py` builds the live view from `raw/*.jsonl` alone, in
  Python, once - the browser renders the server's snapshot rather than
  re-deriving it, because a second fold in TypeScript would drift. Runs are
  CLI child processes (`console/launcher.py`), so a browser-started run and a
  typed one leave identical traces. Nothing the console computes is a number a
  report states; arm progress is counted, never scored (rule 1).
- **The console writes one file, and only for topics.** A research topic is
  content, not a setting, so it cannot be a `--set`; `console/topics.py`
  writes the config's current topics plus the new one to
  `configs/topics.local.yaml` (git-ignored) after validating with the same
  `Topic` model, re-resolves the whole config against what it wrote, and the
  run points at it with `--set experiment.topics_file=...`. The shipped topics
  files are never rewritten, and the argv is still the whole story.
- **Traces are joined, not recomputed.** `console/traces.py` attributes every
  event and span to one actor (`agent_id`, else `arm_id`, else
  orchestration), nests events by `span_id`, and joins an MCP log row to the
  `mcp.tool.call` event that recorded it on `(tool, attempt, duration_ms)` -
  the log carries no span id. An event whose span was never exported is
  placed in its own actor's narrowest enclosing span (same question when both
  carry one) and marked `placed: by_time`/`by_question`; otherwise it stays
  loose. Prompts and completions are not recorded, only their digests,
  tokens and spend, and the trace view says so.
- **A run is its directory, and it survives the process running it.**
  `raw/*.jsonl` is authoritative and append-only; `state.json` is a
  checkpoint. Whole-file writes go through `lifecycle.durable_write_text`
  (unique temp, `fsync`, rename, directory `fsync`), and `RunState.save`
  runs `Tracer.sync` first, so no checkpoint names work the raw trace could
  lose. A resumed `Tracer` cuts a torn final line into `integrity/torn/` and
  records `run.repaired`. Collection checkpoints `collect_progress` after the
  plan and after every round, and resumes from it over state rehydrated from
  `raw/`. Evaluation reuses recorded answers by (arm, question, repetition).
  `pheasant-lab run` (`supervisor.py`) runs each stage as a child and resumes
  one that crashes. `docs/durability.md` has the whole table.
- **A topic may carry an intent.** `Topic.intent` is the person's own words,
  and the planner's brief on every run. It is left out of the digest while
  unset, like every field added after runs existed. "Seed terms or an intent"
  is enforced where a topic is *added* (`console/topics.py`), not at load,
  because topics files with neither loaded before and must keep loading.
  `draft-topic` (`orchestration/drafter.py`) proposes one from an intent
  under its own reserve-first budget; under `replay` it is rule-based and
  says so.
- **The pre-claim interval is recorded, not inferred.** On a role-split
  Pheasant `sync_source` answers `status: queued`; the lab emits
  `ingest.sync` with that disposition, one `ingest.barrier` per acknowledge
  poll, and - when the region offers `get_index_queue` (capability
  `index_queue`, optional) - each task's claim state. A document is shown
  `awaiting_claim` only on that evidence; silence never moves a document
  forward. `index_queue` and `mock_claim_seconds` are left out of the config
  digest: they change what a run reports, not what it measures.
- **P1 waits for its memory to be searchable.** A memory write on a fleet
  answers `sync: {status: queued}` like any other sync (captured from 0.13.2),
  and the record is not searchable until an indexer runs the task.
  `MemorySeeder` keeps the queued task ids and, after seeding, polls
  `get_index_queue` until they leave the listing (`memory.indexed`:
  `indexed`, `timed_out`, `failed` for a dead task, `unknown` with no
  complete listing). Anything but `indexed` becomes a limitation on the
  report; the run is not refused, because the paired difference is still a
  measurement of *something*, and the report says what.
- **P1's memory is a retrieval proof, not a note.** `arms/memory_proof.py`
  writes one readable record per learned question: the question, only the
  claims `supporting_citations` validates (the rule `evidence_support_rate`
  scores with), and each validating passage as a Markdown link to its
  region `relative_path`, with the search and rank that found it and a quote.
  Fewer than `replay.memory_min_evidence` (k, default 1) validating passages
  writes nothing, and `memory.seeded` says why. The links are what pheasant's
  cross-source pass turns into `about` edges (3 per record, which is why 3
  are listed by default). Alias and preference rules come only from a proven
  answer, keep the rule on line one (where pheasant parses it) and explain
  themselves below; the explanation is scrubbed of rule markers. No opaque id
  is ever an alias term. Checked against a real pheasant region, not only the
  mock.

- **A document is named by its title.** `IngestRequest.path()` is
  `<topic>/<title-slug>-<first-author>-<year>.md`, lowercase ASCII and
  hyphens only, because pheasant shows the path as a hit's title, weights it
  most in BM25, splits preference rules on commas and ends a Markdown link at
  `)`. `Ingestor.assign_path` keeps a source's first path for good (restored
  from `ingest-requests.jsonl` on resume, or a resubmission would be a second
  document) and never gives one path to two sources (`-2`, `-3` …).

- **A citation names a source through the receipts, not its spelling.**
  `answer_metrics.compute(citation_sources=…)` resolves each cited region id
  to the lab source before a `source` matcher or `temporal_validity`'s
  known-negative test reads it. Before, a correct citation counted only when
  the region id happened to embed the DOI (`10.5555_x` never matches
  `10.5555/x`), and the forward-reach test compared artifact ids to source
  ids and never fired. Validity and support still read citations as given.

- **Reports name things.** `reports/context.py` loads question text,
  expected facts, document titles, answers, memory records and the tuning
  strategy from the run directory once; every report labels ids through it.
  A comparison whose treatment was never applied (P2 with no tuned rule) says
  "not measured" and its health-vector entry is `null`, never a measured 0.

- **One selection drives the live page.** `ui/src/live/focus.ts` is the
  whole vocabulary: a focus is a branch or an arm, every panel reads it, and
  the feed, custody, facets and both visuals filter or dim from it. Panels do
  not keep a selection of their own, so selecting in one place can never
  disagree with what another shows. Deselecting is always available four
  ways (again, empty canvas, the ×, `Esc`).
- **The console has a key, like a fleet.** `PHEASANT_LAB_CONSOLE_TOKEN`
  guards every `/api` call; the shell and `/api/auth` stay public so the page
  can ask for it, and a console beyond loopback with no key refuses to start.
  The live stream is read with `fetch`, not `EventSource`, because an
  `EventSource` cannot send a header and a key in a URL ends up in logs.
- **Logs are kept whole or deleted whole.** `console/logs.py` deletes launch
  logs, a run's rebuildable projection or an entire run - never part of a raw
  trace (rule 5) and never anything a launch is writing - and audits every
  deletion. Retention defaults to keeping everything; the policy is console
  bookkeeping under `.console/`, not experiment configuration, so it never
  moves a digest. Run logging (`logging.yaml`) is configuration and does.

- **One operation, two surfaces.** `console/operations.py` is the only
  implementation of what the console does; the HTTP routes and the MCP tools
  (`console/mcp.py`, stdlib JSON-RPC at `/mcp` and on stdio) parse, call it and
  marshal. A behaviour that differs between them must be a visible difference
  in an adapter, the shape pheasant-kb's `services/` keeps.
- **One settings draft per console.** `console/state.Workspace`
  (`.console/workspace.json`) holds the config file, its overrides, topic,
  connection and launch caps; the UI and MCP both edit it, and a launch turns
  it into an argv. An invalid draft is kept and reported (`valid: false`)
  unless `strict`, because linked fields - five budget shares summing to one
  - can only be edited one at a time.
- **Every setting is explained, and the explanation cannot drift.**
  `catalog.py` derives keys, types and defaults from the file models and adds
  only meaning, values, ranges and role recommendations;
  `tests/unit/test_catalog.py` fails on a field with no explanation or an
  explanation with no field.
- **`--set` routing is derived, and an unrouted head is refused.** The owner
  of each head comes from the file models (`settings.OVERRIDE_FILES`).
- **A connection is overrides, a token is environment.** Selecting a
  connection writes `transport/url/knowledge_base/source_name/token_env` into
  the draft; a stored token (`.console/secrets.json`, 0600, never returned) is
  handed to launched children under its variable, where the redactor already
  registers every `*_TOKEN`.
- **A run records where it was made.** `environment.deployment` (from
  `PHEASANT_LAB_DEPLOYMENT`, `docker` in the image); a console with a run
  scope lists, reads and deletes only those runs. Not digested.
- **A run digests the prices of the models it uses, not the whole list.**
  `LabConfig.digest` filters `pricing.models` to the roles' models (and the
  judge's), so pricing a new model moves no existing run; changing a used
  price, or the list's `version`, still does. Bump `version` when a price
  changes, not when one is added. The GPT-6 family ships priced, and
  `catalog.MODEL_REASONING` records which reasoning levels each takes:
  `doctor` refuses a level the model would reject, and the form offers only
  the accepted ones.
- **What a person calls a run is not the run.** Labels and notes live in
  `.console/run-labels.json`; the run directory stays append-only evidence.

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
- **A mock kinder than the server hid every adapter bug at once.** Until the
  lab was first run end to end against a real pheasant (0.12.16), the mock had
  auto-indexed unregistered submissions, returned receipts under `receipts`
  rather than `accepted`/`rejected`, listed acknowledgements rather than
  counting them, put the lab's source id in `provenance`, returned whole
  passages, flattened the memory record and parsed a preference grammar
  pheasant does not have. So a live `collect` died at its first sync; with
  that fixed, every receipt read as `no_receipt`, acknowledging crashed, every
  hit collapsed onto one source id, answerers read front matter, memory ids
  were empty and every seeded preference was silently ignored. The mock now
  mirrors captured wire shapes, and `tests/contract/test_pheasant_wire.py`
  reads the captured payloads themselves (`tests/fixtures/pheasant/`) so the
  next drift fails offline. Regenerate those fixtures against a new release
  with `scripts/capture_pheasant_fixtures.py`, and keep the old ones: the
  readers are tested against every supported release, not only the newest.
- **A capture that never contained a hit shape cannot test it.** pheasant's
  graph arm puts a node hit's source on the hit, not under `provenance`, and
  has since before 0.12.16 - but that capture ranked no node hit into its
  result list, so `region_source` was read from `provenance` alone and such
  hits were fetched back with no `source_name`. The 0.13.0 capture ranked one
  second. Read every shape a capture holds, not just the first hit.
- **A field added to the config changes every existing run's digest.** The
  digest is over the whole resolved config, so a new setting - even unset -
  makes every run started before it refuse to resume without `--fork`.
  `LabConfig.digest` leaves `replay.graph_expansion` out while it is unset,
  and the `expand` mapping with it, since a mapping nothing asks for sends
  nothing; do the same for the next one.
- **A sequence number taken under a lock and written after it is not
  sequential.** `EventLog.emit` numbered events inside the lock and appended
  outside it, so concurrent branches wrote 10 before 9 and `verify` reported a
  discontinuity in a stream that had lost nothing - a replay test that failed
  one run in eight.
- **A mock that accepts an argument the real server rejects hides the bug it
  was built to expose.** The adapter mapped `snapshot_id` and `as_of` onto
  `search_context`, which pheasant then exposed on HTTP only. The mock accepted
  both, so the demo, the contract tests and the drift-refusal test all passed
  while a live run would have failed at `P0`'s first search — or, worse,
  ignored the pin and looked pinned. Two fixes, because one was not enough:
  the adapter sends only what the map declares (`pin_sent` records which), and
  preflight now checks every *configured* name rather than a static list that
  was written before the map existed.
- **A run that prints nothing is a run the launcher could not find.** The
  console first learnt a launch's run id from its output, and `demo` is
  silent for the whole of a barrier wait - exactly when someone wants to
  watch. The run directory is watched apart from the output now.
- **A backoff can step over a whole state.** The barrier polls with doubling
  backoff, so a claim that starts and finishes between two polls is never
  observed as `claimed`. The indexer lane says "claim between polls" rather
  than drawing the whole interval as a wait nobody serviced, and the test
  fixture's claim window is sized against the backoff so every state is seen.
- **A receipt's timestamp is its first one.** `ingest-receipts.jsonl`
  re-appends a receipt as `indexed` carrying the time it was *accepted*, so a
  replay that placed receipts by time showed documents indexed before the
  sync that indexed them. A replay admits `indexed` only after it has folded
  the crossed barrier, which is how the lab learnt it in the first place.
- **A span id on an event is not a span in `spans.jsonl`.** Events emitted on
  a worker thread (an arm's MCP calls and model calls, most orchestration
  bookkeeping) carry a span id the tracer never exported, so nesting by
  `span_id` alone left every arm event outside its question's span. The trace
  view places them by time and question, and says it did.
- **A drag ends in a click.** Panning a canvas by dragging across a node
  selected that node on release. `components/zoom.tsx` swallows the click
  after a real drag, in the capture phase so no node handler sees it.
- **An unhandled exception exited `1`, which already meant "completed, and a
  decision failed".** A supervisor could not tell a crash to resume from a
  result to keep. Crashes exit `3` now; `1` stays a result.
- **A pipe is a leash.** The console read its children's output through a
  pipe, so a console that died took every run with it at the child's next
  `print` (`BrokenPipeError`), contradicting the README's "closing the
  console does not stop a run". Children get their own session and a log
  file now, and launch records are files a restarted console re-attaches to.
- **A resumed process knew only its own receipts.** Documents the dead
  process submitted were never acknowledged and stayed `accepted` forever:
  searchable in the region, counted as never indexed. `ReceiptLedger.restore`
  rebuilds the ledger from `ingest-receipts.jsonl`. Found by comparing a
  crashed-and-resumed run to a clean one, the property the durability tests
  now assert.
- **Events and receipts are two files, read in either order.** Stamping
  "awaiting claim" on documents when the queued sync arrived missed every
  document whose acceptance was read afterwards. The claim state lives on the
  model and is applied at snapshot time.
- **P1 started 21 ms after its memory was written, and on a fleet that
  memory was not yet searchable.** Standalone, `memory_write(sync=True)`
  indexes in the call; role-split, it publishes the sync and returns. So `P1`
  ran its first questions with no memory at all and `P1 - P0` understated the
  treatment by however many questions beat the indexer - 57 against 59 of 63
  memory hits between two otherwise identical runs. Found by running the lab
  against a real fleet (0.13.2), not by the mock, which indexed memory
  in-call regardless of `mock_claim_seconds` until it was taught otherwise.
- **A redirect followed on every call is a round trip paid on every call.**
  pheasant mounts MCP at `/mcp/` and answers `/mcp` - the URL every client
  config names - with a 307, and `httpx` follows it per request. So each of a
  run's ~140 tool calls was two HTTP round trips, invisible in every test
  because the mock is in-process. The transport adopts a same-origin redirect
  once per session (never a cross-origin one: the bearer token stays on the
  host the config named). Found by counting `307`s in a live run's log
  against pheasant 0.13.4: 136 calls, 3 redirects after the fix.
- **A probe that swallows a 401 paints a green chip over a locked door.** On a
  region with `security.api_auth`, `/ready` is public and everything else is
  not, so a console with no `PHEASANT_API_TOKEN` read `ready` and then got 401
  from `/queue` - which it treated as "no queue to show". The chip went green
  over a region that would refuse every call the run was about to make. A 401
  is its own notice now, worded for "unset" and for "wrong". Found by probing
  a real 0.13.4 api replica with the token unset.
- **pheasant's two surfaces decide "publish or run" differently.** HTTP `/sync`
  on `--role api` refuses to run in-call; MCP `sync_source` publishes only
  when `graph.query_service_url` is set, and otherwise indexes in the call
  even on an api replica. The shipped fleet always sets that URL, so the lab
  sees `queued` there - but a hand-built role-split region without a graph
  service answers `completed`, and a capture taken against one records the
  wrong shape. `scripts/capture_pheasant_fixtures.py --queued-sync` needs the
  graph-service replica for that reason.
- **A setting with no reader looked like a working one for the whole life of
  the repo.** `logging.yaml`'s `level`, `format` and `file` were declared,
  documented and shipped, and every stage logged at INFO, as text, to stderr
  regardless. `logsetup.configure_logging` is the reader now; a relative
  `file` lands in the run directory, and it is excluded from checksums
  because it grows after a stage has checksummed: digested, `verify` would
  call every run that set it tampered with.
- **A key entered after a refused load does not re-run the load.** `/live`
  resolves "the latest run" once on mount; on a keyed console that call was
  refused before the key existed, so the page sat on "no runs yet" after a
  correct key. The routed page remounts when the key changes. Found by
  driving the real console in a browser, as was the next one.
- **A floating notice covers whatever is under it.** The live page's toasts
  sat bottom-left over the scrubber and the folded bottom strip's toggle,
  and moving them over the canvas only covered a branch instead. They are a
  strip in the layout now, which moves things down a line and covers nothing.
- **`answered` already counts abstentions.** The arm bars added `abstained`
  to it and read "28/14" for the prior-only arm, which abstains on every
  question. A count's definition belongs next to the count; it is commented
  at both places that read it now.
- **A form that renders `override ?? resolved` cannot be typed into.**
  Clearing a field deleted its override and the resolved value snapped back
  mid-keystroke: "12" over "6" became "612", and an experiment name could not
  be retyped at all. Inputs own their text while focused and commit after a
  pause (`ui/src/configure/fields.tsx`). Found by typing, in a real browser.
- **An override head no file claimed was silently dropped.** `budget.*`,
  `pricing.*`, `token_env` and `source_name` resolved to the file's value
  whatever `--set` said - a budget control that did not control the budget.
  Routing is derived from the models now, and an unknown head is refused.
- **A console started with `--runs` elsewhere never saw what it launched.**
  Children wrote to the config's own `experiment.output_root`. The launcher
  passes its output root (not digested) when they differ.
- **A region reached by service name answers MCP 421.** pheasant builds its
  DNS-rebinding allow-list from `server.api.cors_origins`, so the lab's
  container dialling `http://pheasant:8765/mcp` was refused on every call
  while every in-process test (loopback) passed. The vendored region config
  lists the origin, `tests/unit/test_docker_deployment.py` holds it, and
  pheasant-kb's lab-fleet answers list `http://api:8765`. Found by running
  the Compose stack. The fleet half was claimed here before it was true:
  pheasant-kb 0.13.11's answers listed `host.docker.internal` and not `api`,
  and its `test_fleet_manifests.py` now holds both names.
- **An idempotency key that does not name the destination folds across
  destinations.** The key was `pheasant-lab:<lab source>:<digest>`, so a
  second run on the same region into a new `source_name` resubmitted known
  keys - and pheasant folds a known key onto its receipt and keeps
  `indexed` while writing the bytes into the new source. The barrier crossed
  in 25 ms with the new source empty and every Pheasant arm abstained. The
  key names the region source now (`IngestRequest.region_source`), and the
  mock folds as pheasant does. Found by running against a real role-split
  0.13.11 region that an earlier run had used.

