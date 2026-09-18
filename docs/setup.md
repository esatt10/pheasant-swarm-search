# Setup: from a fresh checkout to your first live search

[Back to the README](../README.md) · [Topic examples](topics.md) · [Performance guide](metrics.md)

Install the lab once, then choose a starting point:

| Route | What you need | What it proves |
|---|---|---|
| Offline demo (step 2) | Python and uv | The workflow runs with a simulated knowledge base. |
| Real Pheasant, no model (step 2b) | Python, uv, and Docker | Documents reach a real database, become searchable, and flow through all five arms. |
| Your own research (steps 3–7) | Pheasant and a model API account | The swarm searches live literature for your topic. |

## 1. Install the lab

Use PowerShell on Windows or Terminal on macOS/Linux. Install
[Git](https://git-scm.com/downloads) if `git --version` is unavailable.
Install uv using the command for your system from its
[installation guide](https://docs.astral.sh/uv/getting-started/installation/):

Windows PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reopen your terminal after installation and check `uv --version`. Then run:

```text
git clone https://github.com/esatt10/pheasant-swarm-search.git
cd pheasant-swarm-search
uv python install 3.12
uv sync --extra dev --python 3.12
uv run pheasant-lab --help
```

If the repository is already downloaded, use that folder and skip cloning.
Python 3.12 or newer is required; these commands use 3.12. `uv sync` creates
`.venv` and installs the package and development/reporting dependencies. You do
not need to activate the environment, install Make, or run a shell script.
Run all lab commands from the folder containing `pyproject.toml`.

**Checkpoint:** help lists `demo`, `doctor`, `plan`, `collect`, and the other
commands.

## 2. Try the demo

```text
uv run pheasant-lab demo --config configs/demo.yaml
```

The demo searches the included synthetic tardigrade literature, submits it to
an in-process mock, builds a fixed question set, runs the five comparison arms,
and writes reports. It needs no Pheasant installation or credentials.

**Checkpoint:** find `demo run: run-...` in the terminal and open:

```text
runs/<the-printed-run-id>/reports/summary.md
runs/<the-printed-run-id>/reports/collection.md
```

The demo may return exit code `1`: some collection conditions or evaluation
checks can remain incomplete. Read the output and reports. An ACL check, for
example, cannot pass when the mock provides no access-control enforcement.
Exit code `2` means the command refused to proceed; inspect the error message.

The demo uses fixed replay model settings and ignores the lab's `.env` file.
Exported model settings cannot select a paid provider in this configuration.
Demo scores measure the mock and synthetic fixture, and do not establish live
search or model performance.

## 2b. Test real Pheasant without a model

Install and start [Docker Desktop](https://docs.docker.com/get-started/get-docker/)
or Docker Engine with Compose. From this repository's root, run these commands
in PowerShell, macOS Terminal, or a Linux shell:

```text
docker compose -f deploy/stub/compose.yaml up -d --wait
uv run python scripts/check_pheasant_setup.py
```

The first command downloads Pheasant **0.12.5** if needed and waits for it to
start. The second runs the setup check. Allow a few minutes; it prints each
stage and saves detailed logs rather than flooding your terminal with tool calls.
No `.env`, model download, model account, API key, or pricing edits are needed.

Open **<http://127.0.0.1:8766>**. The knowledge base is **`pheasant-swarm-stub`**;
the MCP endpoint is **`http://127.0.0.1:8766/mcp`**. This profile uses its own
Docker volumes and port, so another service on port 8765 can keep running.
Pheasant uses SQLite text search, with embeddings disabled and its assistant
set to `none`. Every lab role uses the fixed `replay` provider. The sample
documents are synthetic tardigrade literature, not evidence to use in a project.

**Checkpoint:** the final JSON prints `"setup": "PASS"`, 19 indexed documents,
70 answers, and zero model cost. Search `Dsup` in the Pheasant UI to see the
submitted material. Search works without a model; generated chat answers are
not part of this check.

The check runs health/readiness checks, `doctor`, `plan`, `collect`, `audit`,
`freeze-benchmark`, `evaluate`, `replay`, `report`, and `verify`. It requires
indexed receipts, matching content hashes, source-linked retrieval, all arms
completing, and intact run files. Its output names the report and evidence
folder under `runs/setup-check-.../`.

**Setup success and experiment quality are separate results.** `verify` can
return `1` because experiment gates are `INCOMPLETE` or `FAIL`. The setup check
still requires the core ingestion, benchmark-leakage, and snapshot gates to
pass, and rejects structural or execution failures. Scores from replay models
do not establish model quality; ACL and stale-memory checks can lack evidence.
Read [the performance guide](metrics.md) before interpreting those scores.

Stop the service when finished:

```text
docker compose -f deploy/stub/compose.yaml stop
```

Start it again with the same `up -d --wait` command. The volumes retain
documents, receipts, and memory. Running the check again tests the existing
fixture collection; previous evaluation memory may still be present. Use a
separate knowledge base for real research or a controlled performance study.

The profile is in [compose.yaml](../deploy/stub/compose.yaml), its validated
Pheasant configuration in [pheasant.yaml](../deploy/stub/pheasant.yaml), and the
lab configuration in [stub.yaml](../configs/stub.yaml). Pheasant's setup wizard
generated the server configuration from [answers.json](../deploy/stub/answers.json).
To change it, edit the answers and follow the regeneration instructions in
[the profile README](../deploy/stub/README.md).

## 3. Start Pheasant for a live collection

Pheasant is a separate service. This lab connects to it; installing the lab does
not install or launch Pheasant.

If you already have a dedicated instance, use its address and knowledge-base
name. For a new local instance, [install and start Docker](https://docs.docker.com/get-started/get-docker/), then run this in a
second terminal. It uses an empty workspace volume and a persistent state
volume dedicated to this project:

```text
docker run --name pheasant-swarm-kb -p 127.0.0.1:8765:8765 -v pheasant-swarm-workspace:/workspace -v pheasant-swarm-state:/state ghcr.io/esatt10/pheasant
```

Keep that terminal running. Open <http://127.0.0.1:8765> and note the actual
knowledge-base name shown by Pheasant. The lab must use that existing name;
`PHEASANT_KNOWLEDGE_BASE` does not create a knowledge base. Later, restart an
existing stopped container with `docker start pheasant-swarm-kb`.

The server's MCP endpoint is `http://127.0.0.1:8765/mcp`. MCP is the tool
connection the swarm uses to submit and search documents. This command adapts
Pheasant's [official quick start](https://github.com/esatt10/pheasant-kb#quick-start);
see its [setup guide](https://github.com/esatt10/pheasant-kb/blob/main/docs/how-to/setup.md)
for host installation and customization.

Keep the lab checkout and `runs/` outside Pheasant's indexed sources: they
contain benchmark answers and reports. The swarm submits its source documents
over MCP, so Pheasant does not need the lab folder mounted.

**Checkpoint:** the Pheasant UI loads. Record its knowledge-base name for step 5.

## 4. Create your local configuration files

These copy commands are for first setup. If you already have local files,
edit those instead of overwriting them.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
Copy-Item configs/experiment.example.yaml configs/experiment.yaml
Copy-Item configs/topics.project.example.yaml configs/topics.yaml
Copy-Item configs/pricing.example.yaml configs/pricing.yaml
```

macOS/Linux:

```bash
cp .env.example .env
cp configs/experiment.example.yaml configs/experiment.yaml
cp configs/topics.project.example.yaml configs/topics.yaml
cp configs/pricing.example.yaml configs/pricing.yaml
```

Open `configs/experiment.yaml` in a text editor. In its existing `experiment`
section, change these fields, leaving the other fields in place:

```yaml
experiment:
  name: my-project-research
  topics_file: configs/topics.yaml
  cost_budget_usd: 10.00
  runtime_budget_minutes: 120
```

Do not replace the whole file with that fragment. The remaining file references
can continue to point to the committed examples. Paths resolve from the
repository root. All four local files are ignored by Git.

The project topic file includes battery research and employee onboarding. For
onboarding, set the existing `collection.providers` field to
`[openalex, crossref]`. For battery research you can also include `arxiv`.
See [choosing a topic](topics.md) to replace the examples with your own.

## 5. Connect the model and knowledge base

Edit `.env`. Choose `MODEL_PROVIDER=openai` or `MODEL_PROVIDER=anthropic` and
fill in the corresponding API key. Change **all six model IDs**, since the
copied file otherwise names offline replay models:

| Variable | Used for |
|---|---|
| `ORCHESTRATOR_MODEL` | Planner and audit narration, plus the configured orchestrator role |
| `RESEARCHER_MODEL` | Claim extraction during collection |
| `SPECIALIST_MODEL` | Source-aware reference answers in evaluation |
| `TEST_AGENT_MODEL` | Pheasant-based answers in evaluation |
| `CONTROL_MODEL` | Answers without collected evidence in evaluation |
| `BENCHMARK_MODEL` | Model call made while recording the benchmark; question construction itself is deterministic |

You can start with one model ID supported by the chosen adapter for all six
variables. Use your provider's exact ID and confirm that it supports the
adapter's structured response format. This repository does not maintain a
current list of supported model IDs or prices. Set the four
`*_REASONING_EFFORT` values to `null` unless you have verified that your chosen
model and adapter support the requested reasoning settings.

Also set:

```dotenv
MODEL_PRICING_FILE=configs/pricing.yaml
PHEASANT_MCP_TRANSPORT=streamable_http
PHEASANT_MCP_URL=http://127.0.0.1:8765/mcp
PHEASANT_KNOWLEDGE_BASE=the-actual-name-from-pheasant
PHEASANT_SOURCE_NAME=project-research
```

Replace `the-actual-name-from-pheasant` with the name from step 3. Choose a new
source name that belongs to this project; do not point it at an unrelated
configured source. Set `PHEASANT_MCP_TOKEN` if your server requires one.

Use **`uv run --env-file .env` for every live command**. The lab reads `.env`
for configuration substitution, while adapters read API keys and connection
tokens from the process environment. The uv option makes them available to
both, as described in [uv's environment-file documentation](https://docs.astral.sh/uv/concepts/configuration-files/#environment-variable-files).
Already-exported shell variables take precedence; avoid conflicting
model or Pheasant values in your terminal.

## 6. Add prices and check the budget

In `configs/pricing.yaml`, add each distinct model ID under the existing
`models` mapping, using the provider's current input and output prices in
**USD per million tokens**. Copy the exact model ID from `.env` as the key.
Each entry has the shape `"your-model-id": { input: INPUT_PRICE, output: OUTPUT_PRICE }`;
replace both price placeholders with actual numbers before running. Keep the
replay entries if you also want to use this file with offline models.
The commented live entries in the example are placeholders, not real prices.

Check the connection and preview the work:

```text
uv run --env-file .env pheasant-lab doctor --config configs/experiment.yaml
uv run --env-file .env pheasant-lab plan --config configs/experiment.yaml --topic topic-onboarding
```

`doctor` checks configuration, local price entries, provider construction, and
the server's advertised MCP tools and argument schemas. **It does not make a
paid model call or prove that an API key, model ID, or literature endpoint
works.** Its `PASS` is a preflight result.

`plan` estimates volume and cost without model calls or document writes. It
includes the later benchmark/evaluation stages and uses assumed prompt sizes;
it is a planning estimate, not a quote. The runtime ledger reserves estimated
worst-case call cost before calling and reconciles reported usage afterward.
The default USD 10 budget covers costs recorded by this lab, not an independent
Pheasant hosting or API bill. Collection preserves the configured evaluation
reserve even if you only intend to collect.

If the plan is too large, lower `max_research_agents`, `max_depth`, or
`max_sources_per_subtopic` under `collection`, or choose a less expensive model
and update its price. Set these before starting a run. Smaller collections can
leave coverage incomplete, which the audit will report.

## 7. Search and inspect the result

```text
uv run --env-file .env pheasant-lab collect --config configs/experiment.yaml --topic topic-onboarding
```

It prints a `run: run-...` identifier, stop decision, source/claim counts, and
recorded cost. Replace `RUN_ID` below with that identifier:

```text
uv run --env-file .env pheasant-lab audit --config configs/experiment.yaml --run RUN_ID
```

**Checkpoint:** inspect the audit's gaps, then search your selected knowledge
base in Pheasant for terms from the topic. Check that results name the
`project-research` source and link to the original source material.

The topic guide explains [what is saved and what to hand off](topics.md#use-the-knowledge-in-your-next-project-phase).
The [performance guide](metrics.md#run-the-optional-evaluation) gives optional
freeze/evaluate/report commands. Reports require evaluation output;
`audit` is the immediate check after collection.

## Common problems

| What you see | What to do |
|---|---|
| `uv` or `git` is not recognized | Install it, reopen the terminal, and retry its `--version` command. |
| Missing configuration file | Run from the checkout root; create local files in step 4. `configs/experiment.yaml` is not shipped. |
| `unknown topic` | Use the topic's exact `id`, such as `topic-onboarding`, from the configured file. |
| API key is not set | Fill the matching key in `.env` and put `--env-file .env` immediately after `uv run`. |
| Model has no price | Add its exact ID and real prices to the file named by `MODEL_PRICING_FILE`. |
| Unauthorized or unsupported model request | Check account access, model ID, response-format support, and reasoning settings. `doctor` does not test model calls. |
| Cannot connect to MCP | Keep Pheasant running; check the URL ends in `/mcp`, the port matches, and the token is correct. The stub profile uses port **8766**. |
| Docker cannot start / port 8766 is occupied | Start Docker Desktop; inspect `docker compose -f deploy/stub/compose.yaml ps` and `logs --tail 50`. If changing the host port, also change the URL in `configs/pheasant-mcp.stub.yaml`. |
| Setup check fails | Open the stage log under the printed `runs/setup-check-.../` folder. Restarting the service preserves its volumes; do not delete volumes to troubleshoot a connection error. |
| Missing tool or invalid configured argument | Compare `configs/pheasant-mcp.example.yaml` with the server's advertised tools. Copy it to `configs/pheasant-mcp.yaml`, set `experiment.pheasant_file` to that path, and edit the map. |
| Unknown knowledge base / rejected source | Use the knowledge-base name registered in Pheasant and a new source name dedicated to the lab. |
| Few sources or no abstracts | Review `raw/errors.jsonl`, source rejection reasons, and topic vocabulary. Provider construction does not guarantee live access or abstract availability. |
| `stopped_*_incomplete` | A budget, time, or round limit was reached. Review saved evidence and unmet conditions; use a new run for changed settings. |
| Configuration changed on resume | Use the original config, `.env`, pricing, and `--set` overrides. Intentional changes belong in a new run; `--fork` records a deliberate change within the existing run. |
| `no run at ...` | Pass the printed ID and original config/output root. For demo follow-up commands, use `--config configs/demo.yaml`. |
| `insufficient_evidence` or `INCOMPLETE` | Read the stated limitation or unevaluated gates in the [performance guide](metrics.md). These are not scores of zero. |
