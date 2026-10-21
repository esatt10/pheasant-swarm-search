# Running the lab in Docker

```bash
cp .env.docker.example .env          # set PHEASANT_API_TOKEN and PHEASANT_LAB_CONSOLE_TOKEN
docker compose up -d --build
```

| | URL | Key |
|---|---|---|
| The console (UI) | <http://127.0.0.1:8770> | `PHEASANT_LAB_CONSOLE_TOKEN` (the browser asks once per tab) |
| The console's MCP endpoint | `http://127.0.0.1:8770/mcp` | the same key, as `Authorization: Bearer …` |
| The bundled Pheasant region | <http://127.0.0.1:8765> (its own UI) | `PHEASANT_API_TOKEN` |

Two services:

- **`lab`** — this repository's image: the console (UI, HTTP API, MCP) and the
  CLI it launches runs with. Runs go to the `lab-runs` volume; what the console
  writes (topics, connections, prices) goes to `lab-local`.
- **`pheasant`** — pheasant-kb **0.13.11**, one container (`role: all`),
  configured by `deploy/pheasant/pheasant.yaml`. That file is maintained here
  (pheasant-kb's only lab setup is the pheasant-lab fleet, below); it is what
  lets the lab actually load documents and have them indexed:

  | Setting | Why the lab needs it |
  |---|---|
  | `server.mcp.transports.streamable_http: true` | the lab is an MCP client |
  | `pheasant.name: pheasant-lab` | the knowledge base every connection names |
  | `/state` in `security.allow_workspace_roots` | `submit_documents` lands bytes in `/state/uploads/<source>`; the lab registers that directory as a `document_folder` source and syncs it. Without the allow-list the registration is refused and nothing is indexed |
  | `http://pheasant:8765` in `server.api.cors_origins` | pheasant derives its MCP DNS-rebinding allow-list from this list; a container reaching it by service name is otherwise answered **421** on every call |
  | a `memory` source at `/memory`, `memory.steering_enabled` | the P1 arm writes memory and steering records |
  | `security.api_auth.token_env: PHEASANT_API_TOKEN` | the region refuses unauthenticated calls; both containers read the one value from `.env` |
  | no embeddings, no assistant model | the first run needs no key |

## The first run is real and free

The image's default experiment is `configs/docker.yaml`: the offline fixture
literature and the deterministic `replay` models, against the **real**
bundled region. Launch it from the console (**Configure → Launch run**) or over
MCP (`lab_launch_run`). The swarm collects, every document is submitted,
registered, synced and acknowledged as indexed, the benchmark is frozen and
the five arms answer — P0, P1 and P2 by searching Pheasant. The report says
the replay provider was used, so the numbers measure retrieval, not a model.

From there, change anything from the console or its MCP tools — every change is
an ordinary `--set`:

- **Agents & models → Use recommended for all** (GPT-6.1 Sol and GPT-6 Luna;
  they need `OPENAI_API_KEY` in `.env`). Their prices
  ship in `configs/pricing.example.yaml`; any other model needs one under
  **Model prices**, because a model with no price is refused before any spend.
- **Collection profile** for the scholarly or web providers.
- **Budget** for the dollar ceiling, the per-phase split and a per-launch cap.
- **Pheasant connection** for another region.

## Only the runs made here

Every run records `environment.deployment` in its manifest, and the image sets
it to `docker`. The console in the image runs with
`PHEASANT_LAB_RUN_SCOPE=docker`, so the Runs, Live, Reports and Logs pages —
and the MCP tools — list, read and delete only runs made in this deployment. A
directory of laptop runs mounted over `/app/runs` stays invisible to it, and
cannot be deleted through it. Unset the variable to see everything.

## Pointing the lab at another region

**pheasant-kb's lab fleet** (Postgres, NATS, an indexer, workers): start it
from a pheasant-kb checkout, then the lab with the fleet Compose file, which
joins the fleet's network and dials `http://api:8765/mcp`:

```bash
# in pheasant-kb
docker compose --env-file .env -f deploy/compose/docker-compose.pheasant-lab.yml up -d
# here, with the fleet's PHEASANT_API_TOKEN in .env
docker compose -f docker-compose.fleet.yml up -d --build
```

The fleet's answer file admits `http://api:8765` for the same 421 reason as
above. pheasant-kb 0.13.11's did not, so pull a checkout after it and
regenerate the fleet's `pheasant.yaml` from the answer file. On the fleet a sync is
*queued* and claimed by the indexer; the Live view shows that interval.

**Any other region**: add it under **Configure → Pheasant connection** (URL,
knowledge base, source, token variable, and optionally the token itself, which
the console stores `0600` and hands to the runs it launches), or with
`lab_save_connection` over MCP. The region needs the same four things as the
table above: MCP over HTTP, its state path allow-listed, the origin the lab
dials in `cors_origins`, and the token.

## Volumes and upgrades

| Volume | Holds | Safe to delete? |
|---|---|---|
| `lab-runs` | every run, the console's launch records, the settings draft, the retention policy, stored connection tokens | deleting it deletes every run |
| `lab-local` | `topics.local.yaml`, `connections.local.yaml`, `pricing.local.yaml` | loses topics, connections and prices added in the console |
| `pheasant-state` | the region's index, receipts, snapshots | the region forgets what the lab sent; old runs can no longer be re-evaluated against it |

The image's own `configs/` is content an upgrade replaces, which is why nothing
the console writes lives there.
