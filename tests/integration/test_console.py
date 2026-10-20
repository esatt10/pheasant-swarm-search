"""The console: one fold over a run's raw files, served over stdlib HTTP.

What is held here is what a person watching a run relies on: that the live
view is a fold of the append-only trace and nothing else, that the pre-claim
interval a fleet introduces is shown as its own state rather than as an
unexplained wait, and that the server cannot be walked out of the run and
config directories it was pointed at.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from pheasant_lab.console.launcher import Launcher
from pheasant_lab.console.projection import LiveModel, RunWatcher, _Tail, list_runs
from pheasant_lab.console.region import base_url, notices_for
from pheasant_lab.console.server import serve
from pheasant_lab.settings import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fleet_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A demo run against a mock region that behaves like a role-split fleet."""

    root = tmp_path_factory.mktemp("console")
    environment = {
        **os.environ,
        "PHEASANT_LAB_FIXTURES": str(REPO_ROOT / "tests" / "fixtures" / "literature"),
        "PHEASANT_LAB_PROMPTS": str(REPO_ROOT / "prompts"),
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pheasant_lab.cli",
            "demo",
            "--config",
            str(REPO_ROOT / "configs" / "demo.yaml"),
            "--set",
            # The barrier polls with doubling backoff capped at 1s (the mock
            # file's retry_backoff_max_seconds): polls near 0.75s and 1.55s
            # land either side of a 1s claim, and the 2s finish lands before
            # the next one, so every state is observed once at least.
            "mock_claim_seconds=1.0",
            "--output-root",
            str(root / "runs"),
            "--project-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    runs = sorted((root / "runs").glob("run-*"))
    assert runs, "the fleet demo produced no run"
    yield runs[-1]


def _events(run: Path, kind: str) -> list[dict]:
    lines = (run / "raw" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [row for row in map(json.loads, lines) if row["event_type"] == kind]


# ---------------------------------------------------------------------------
# the lab records the pre-claim interval
# ---------------------------------------------------------------------------


def test_a_queued_sync_is_recorded_as_queued_not_as_indexed(fleet_run: Path) -> None:
    (sync,) = _events(fleet_run, "ingest.sync")
    assert sync["payload"]["disposition"] == "queued"
    assert sync["payload"]["region"]["task_id"].startswith("idx-")


def test_the_barrier_reports_the_claim_before_it_crosses(fleet_run: Path) -> None:
    barrier = _events(fleet_run, "ingest.barrier")
    states = [{task["state"] for task in row["payload"].get("queue") or []} for row in barrier[:-1]]
    assert {"awaiting_claim"} in states
    assert {"claimed"} in states
    assert states.index({"awaiting_claim"}) < states.index({"claimed"})
    assert barrier[-1]["payload"]["disposition"] == "crossed"
    assert barrier[-1]["status"] == "succeeded"


def test_p1_waits_for_its_memory_to_be_indexed(fleet_run: Path) -> None:
    """A fleet publishes the memory sync too; P1 must not start before it lands."""

    (indexed,) = _events(fleet_run, "memory.indexed")
    assert indexed["payload"]["outcome"] == "indexed"
    assert indexed["payload"]["tasks"], "the memory sync was not queued"
    first_p1 = min(
        row["sequence"]
        for row in _events(fleet_run, "arm.answered")
        if row["payload"].get("arm_id") == "P1" or row.get("arm_id") == "P1"
    )
    assert indexed["sequence"] < first_p1


def test_a_standalone_region_reports_an_in_call_sync(demo_run: tuple[Path, str]) -> None:
    run, _ = demo_run
    (sync,) = _events(run, "ingest.sync")
    assert sync["payload"]["disposition"] == "completed"
    # No queue to read, and nothing was ever awaiting a claim.
    assert all("queue" not in row["payload"] for row in _events(run, "ingest.barrier"))


# ---------------------------------------------------------------------------
# the projection
# ---------------------------------------------------------------------------


def test_the_projection_folds_a_finished_fleet_run(fleet_run: Path) -> None:
    watcher = RunWatcher(fleet_run)
    watcher.refresh()
    view = watcher.model.snapshot()

    custody = view["custody"]
    assert custody["discovered"] >= custody["acquired"] > 0
    assert custody["indexed"] == custody["accepted"]
    assert custody["awaiting_claim"] == 0 and custody["claimed"] == 0

    indexer = next(lane for lane in view["lanes"] if lane["id"] == "indexer")
    labels = [bar["label"] for bar in indexer["bars"]]
    assert labels == ["awaiting claim", "indexing"]
    assert all(bar["end"] is not None for bar in indexer["bars"])

    history = {notice["code"]: notice for notice in view["notice_history"]}
    assert history["queued"].get("closed") is True
    assert history["claimed"].get("closed") is True
    assert "crossed" in {notice["code"] for notice in view["notices"]}

    assert [phase["state"] for phase in view["run"]["phases"]] == ["done"] * 5
    assert {arm["arm_id"] for arm in view["arms"]} == set(view["run"]["arms_configured"])
    researchers = [agent for agent in view["agents"] if agent["role"] == "researcher"]
    assert researchers and all(agent["subtopic_label"] for agent in researchers)


def test_a_document_mid_claim_is_shown_awaiting_its_claim(fleet_run: Path) -> None:
    """Stop the fold at the first barrier poll: the live view a person saw then."""

    model = LiveModel(run_id=fleet_run.name)
    for row in map(json.loads, (fleet_run / "raw/events.jsonl").read_text().splitlines()):
        model.apply_event(row)
        if row["event_type"] == "ingest.barrier":
            break
    for row in map(json.loads, (fleet_run / "raw/ingest-receipts.jsonl").read_text().splitlines()):
        if row["status"] == "accepted":
            model.apply_receipt(row)

    view = model.snapshot()
    assert view["custody"]["awaiting_claim"] == view["custody"]["accepted"] > 0
    assert view["custody"]["indexed"] == 0
    assert view["notices"][-1]["code"] == "queued"
    assert [phase["state"] for phase in view["run"]["phases"]] == [
        "done",
        "active",
        "pending",
        "pending",
        "pending",
    ]
    indexer = next(lane for lane in view["lanes"] if lane["id"] == "indexer")
    assert indexer["bars"][-1]["status"] == "pre_claim" and indexer["bars"][-1]["end"] is None


def test_a_standalone_funnel_never_claims_a_wait(demo_run: tuple[Path, str]) -> None:
    watcher = RunWatcher(demo_run[0])
    watcher.refresh()
    custody = watcher.model.snapshot()["custody"]
    assert custody["awaiting_claim"] == 0
    assert custody["indexed"] == custody["accepted"] > 0


def test_the_tail_never_reads_half_a_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"a": 1}\n{"b": ', encoding="utf-8")
    tail = _Tail(path)
    assert tail.read() == [{"a": 1}]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("2}\n")
    assert tail.read() == [{"b": 2}]
    assert tail.read() == []


# ---------------------------------------------------------------------------
# the region probe
# ---------------------------------------------------------------------------


def test_the_region_base_url_is_the_mcp_url_without_its_mount() -> None:
    assert base_url("http://127.0.0.1:8765/mcp") == "http://127.0.0.1:8765"
    assert base_url("https://kb.example/region/mcp/") == "https://kb.example/region"
    assert base_url("") is None


def test_each_readiness_state_is_its_own_notice() -> None:
    assert (
        notices_for({"status": "draining", "draining_for_seconds": 3}, None)[0]["code"]
        == "draining"
    )
    assert notices_for({"status": "standby"}, None)[0]["tone"] == "info"
    assert (
        notices_for({"status": "not_ready", "reason": "state store unreachable"}, None)[0]["detail"]
        == "state store unreachable"
    )
    stale = notices_for(
        {"status": "ready", "graph_generation": {"loaded": "a", "published": "b"}}, None
    )
    assert stale[0]["code"] == "stale_generation"
    assert notices_for({"status": "ready"}, None) == []


def test_a_pre_claim_task_is_a_warning_and_a_claimed_one_is_not() -> None:
    queue = {
        "tasks": [
            {
                "task_id": "idx-1",
                "source": "lit",
                "state": "awaiting_claim",
                "waiting_seconds": 42,
                "position": 1,
            },
            {"task_id": "idx-2", "source": "lit", "state": "claimed"},
            {"task_id": "idx-3", "source": "lit", "state": "dead", "last_error": "boom"},
        ]
    }
    notices = notices_for({"status": "ready"}, queue)
    assert [(n["tone"], n["code"]) for n in notices] == [
        ("warn", "queue:idx-1"),
        ("danger", "queue:idx-3"),
    ]


def _probe_against(monkeypatch, handler, *, token: str | None):
    """``probe_region`` with its HTTP client answered by ``handler``, offline."""

    import httpx

    from pheasant_lab.console import region

    real = httpx.Client

    def client(**kwargs):
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(region.httpx, "Client", client)
    if token is None:
        monkeypatch.delenv("PHEASANT_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("PHEASANT_API_TOKEN", token)
    return region.probe_region(
        {
            "transport": "streamable_http",
            "url": "http://region.test:8765/mcp",
            "knowledge_base": "pheasant-lab",
            "source_name": "swarm-lab-literature",
        }
    )


def _authenticated_region(request):
    import httpx

    wire = json.loads((REPO_ROOT / "tests/fixtures/pheasant/responses-0.13.11.json").read_text())
    if request.url.path == "/ready":
        return httpx.Response(200, json={"status": "ready", "role": "api"})
    if request.headers.get("authorization") != "Bearer right":
        return httpx.Response(401, json={"detail": "this region requires a bearer token"})
    if request.url.path == "/queue":
        return httpx.Response(200, json=wire["get_index_queue"])
    if request.url.path == "/sources/swarm-lab-literature/overview":
        return httpx.Response(200, json=wire["describe_source"])
    return httpx.Response(404, json={"detail": "Not Found"})


def test_the_probe_reports_what_the_region_holds_for_the_labs_source(monkeypatch) -> None:
    probe = _probe_against(monkeypatch, _authenticated_region, token="right")
    assert probe["inventory"]["documents"] == 19
    assert probe["inventory"]["source_name"] == "swarm-lab-literature"
    assert not any(n["code"] == "auth" for n in probe["notices"])


@pytest.mark.parametrize(
    ("token", "title"), [(None, "Region needs a token"), ("wrong", "Region refused the token")]
)
def test_a_region_that_refuses_the_probe_says_so_rather_than_going_green(
    monkeypatch, token, title
) -> None:
    """``/ready`` is public on a token-protected region; a silent 401 read as healthy."""

    probe = _probe_against(monkeypatch, _authenticated_region, token=token)
    assert probe["notices"][0]["code"] == "auth"
    assert probe["notices"][0]["title"] == title
    assert probe["notices"][0]["tone"] == "danger"
    assert probe["inventory"] is None


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_neither_console_setting_moves_the_config_digest() -> None:
    """A run started before either existed must still resume without --fork."""

    base = load_config(REPO_ROOT / "configs/demo.yaml", project_root=REPO_ROOT)
    plain = base.model_copy(deep=True)
    plain.pheasant.capabilities.pop("index_queue", None)
    assert base.digest() == plain.digest()
    simulated = load_config(
        REPO_ROOT / "configs/demo.yaml",
        project_root=REPO_ROOT,
        overrides={"mock_claim_seconds": "0"},
    )
    assert simulated.digest() == base.digest()
    moved = load_config(
        REPO_ROOT / "configs/demo.yaml",
        project_root=REPO_ROOT,
        overrides={"mock_claim_seconds": "2"},
    )
    assert moved.digest() != base.digest()


def test_the_launcher_refuses_what_it_was_not_built_to_run(tmp_path: Path) -> None:
    launcher = Launcher(REPO_ROOT, tmp_path)
    with pytest.raises(ValueError, match="not a config file"):
        launcher.launch(kind="demo", config="../../etc/passwd")
    with pytest.raises(ValueError, match="unknown launch kind"):
        launcher.launch(kind="rm", config="configs/demo.yaml")
    with pytest.raises(ValueError, match="needs a run"):
        launcher.launch(kind="evaluate", config="configs/demo.yaml")
    with pytest.raises(ValueError, match=r"a\.b=value"):
        launcher.launch(kind="demo", config="configs/demo.yaml", overrides=["oops"])


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------


@pytest.fixture
def console(fleet_run: Path, tmp_path: Path) -> Iterator[str]:
    server = serve(
        host="127.0.0.1",
        port=0,
        project_root=REPO_ROOT,
        output_root=fleet_run.parent,
        default_config="configs/demo.yaml",
        ui_dist=tmp_path / "no-ui",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _get(url: str) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode()
            status = response.status
    except urllib.error.HTTPError as error:
        body, status = error.read().decode(), error.code
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


def test_the_api_lists_and_folds_runs(console: str, fleet_run: Path) -> None:
    status, runs = _get(f"{console}/api/runs")
    assert status == 200 and runs[0]["run_id"] == fleet_run.name
    listed = list_runs(fleet_run.parent)
    assert [
        {k: v for k, v in row.items() if k not in {"live", "resumable", "label", "notes", "kept"}}
        for row in runs
    ] == listed
    assert runs[0]["live"] is False and runs[0]["resumable"] is (not listed[0]["complete"])

    status, view = _get(f"{console}/api/runs/{fleet_run.name}")
    assert status == 200 and view["run"]["run_id"] == fleet_run.name
    # The progress bars' denominator: every question, once per repetition.
    assert view["run"]["repetitions"] == 1
    # `answered` counts every answer, abstentions included: never more
    # answers than are due, which is what the arm bars draw.
    for arm in view["arms"]:
        assert arm["abstained"] <= arm["answered"] <= view["questions_total"]
    # The mock describes the lab's source after the barrier, like 0.13.11.
    inventory = view["region"]["inventory"]
    assert inventory["disposition"] == "consistent"
    assert inventory["region_documents"] == inventory["receipts_indexed"] > 0

    status, page = _get(f"{console}/api/runs/{fleet_run.name}/events?after=0&limit=5")
    assert [row["sequence"] for row in page["events"]] == [1, 2, 3, 4, 5]


def test_the_stream_opens_with_the_backlog_then_the_model(console: str, fleet_run: Path) -> None:
    with urllib.request.urlopen(
        f"{console}/api/runs/{fleet_run.name}/stream", timeout=10
    ) as stream:
        assert stream.headers["Content-Type"] == "text/event-stream"
        kinds: list[str] = []
        while "model" not in kinds:
            line = stream.readline().decode().strip()
            if line.startswith("event: "):
                kinds.append(line.removeprefix("event: "))
    assert kinds[0] == "events" and kinds[-1] == "model"


def test_the_server_cannot_be_walked_out_of_its_directories(console: str, fleet_run: Path) -> None:
    assert _get(f"{console}/api/runs/..%2F..%2Fetc")[0] == 404
    assert _get(f"{console}/api/runs/{fleet_run.name}/reports/..%2Frun-manifest.json")[0] == 404
    status, _ = _get(f"{console}/api/region?config=../pyproject.toml")
    assert status == 400


def test_an_unbuilt_ui_says_how_to_build_it(console: str) -> None:
    status, body = _get(f"{console}/live")
    assert status == 200 and "make ui" in str(body)


def test_a_replay_position_shows_the_run_as_it_stood(fleet_run: Path) -> None:
    from pheasant_lab.console.projection import fold_until

    (sync,) = _events(fleet_run, "ingest.sync")
    then = fold_until(fleet_run, int(sync["sequence"])).snapshot()
    watcher = RunWatcher(fleet_run)
    watcher.refresh()
    now = watcher.model.snapshot()

    assert then["run"]["sequence"] == sync["sequence"] < now["run"]["sequence"]
    assert then["custody"]["indexed"] == 0 and now["custody"]["indexed"] > 0
    assert then["custody"]["awaiting_claim"] > 0
    assert not then["arms"] and now["arms"]


# ---------------------------------------------------------------------------
# every agent's trace
# ---------------------------------------------------------------------------


def _walk(spans: list[dict]) -> Iterator[dict]:
    for span in spans:
        yield span
        yield from _walk(span["children"])


def test_every_event_lands_in_exactly_one_actors_trace(fleet_run: Path) -> None:
    from pheasant_lab.console.traces import RunTraces

    traces = RunTraces(fleet_run)
    actors = traces.actors()
    kinds = {row["kind"] for row in actors}
    assert kinds == {"orchestration", "agent", "arm"}

    seen: list[str] = []
    for row in actors:
        trace = traces.trace(row["actor"])
        placed = [e["event_id"] for s in _walk(trace["spans"]) for e in s["events"]]
        loose = [e["event_id"] for e in trace["events"]]
        assert len(placed) + len(loose) == row["events"]
        seen += placed + loose
    events = (fleet_run / "raw" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(seen) == sorted(json.loads(line)["event_id"] for line in events)


def test_an_arms_trace_carries_its_answers_and_its_mcp_traffic(fleet_run: Path) -> None:
    from pheasant_lab.console.traces import RunTraces

    traces = RunTraces(fleet_run)
    trace = traces.trace("arm:P0")
    spans = list(_walk(trace["spans"]))
    assert spans and all(s["name"] == "arm.answer" for s in spans)
    # Every P0 event sits in the span of its own question, never another's.
    for span in spans:
        for event in span["events"]:
            if event["question_id"]:
                assert event["question_id"] == span["attributes"]["question_id"]
    calls = [e for s in spans for e in s["events"] if "mcp_call" in e]
    assert calls, "P0 searched the region, so its trace holds MCP calls"
    call = traces.mcp_call(calls[0]["mcp_call"])
    assert call["request"]["method"] == "tools/call" and call["response"]["result"]
    assert call["tool"] == calls[0]["payload"]["tool"]
    answers = {a["question_id"]: a for a in trace["answers"]}
    assert set(answers) == {s["attributes"]["question_id"] for s in spans}
    assert all(a["question"] for a in answers.values())


def test_a_branchs_trace_carries_the_claims_it_extracted(fleet_run: Path) -> None:
    from pheasant_lab.console.traces import RunTraces

    traces = RunTraces(fleet_run)
    branches = [row for row in traces.actors() if row["kind"] == "agent"]
    claims = [len(traces.trace(row["actor"])["claims"]) for row in branches]
    total = len((fleet_run / "raw" / "claims.jsonl").read_text(encoding="utf-8").splitlines())
    assert sum(claims) == total > 0


def test_the_trace_routes_answer_and_refuse(console: str, fleet_run: Path) -> None:
    status, body = _get(f"{console}/api/runs/{fleet_run.name}/traces")
    assert status == 200 and body["actors"][0]["actor"] == "orchestration"
    status, trace = _get(f"{console}/api/runs/{fleet_run.name}/traces/arm%3AP0")
    assert status == 200 and trace["actor"] == "arm:P0"
    assert _get(f"{console}/api/runs/{fleet_run.name}/traces/arm%3AZ9")[0] == 404
    assert _get(f"{console}/api/runs/{fleet_run.name}/mcp/0")[0] == 200
    assert _get(f"{console}/api/runs/{fleet_run.name}/mcp/999999")[0] == 404


# ---------------------------------------------------------------------------
# adding a research topic
# ---------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project root of its own, so a written topics file never lands in the repo."""

    import shutil

    shutil.copytree(
        REPO_ROOT / "configs",
        tmp_path / "configs",
        ignore=shutil.ignore_patterns("topics.local.yaml"),
    )
    return tmp_path


TOPIC = {
    "id": "topic-solid-state-dendrites",
    "title": "Solid-state battery dendrite suppression",
    "seed_terms": ["lithium dendrite solid electrolyte", " ", "garnet LLZO interphase"],
    "date_range": {"from": "2018-01-01", "to": ""},
    "facets": [
        {"id": "interphase", "label": "Interphase chemistry", "weight": 3},
        {"id": "pressure", "label": "Stack pressure", "weight": 2},
    ],
    "source_authority": {
        "preferred_types": ["journal_article", "review"],
        "minimum_peer_reviewed": 2,
    },
}


def test_a_topic_is_written_beside_the_current_ones_and_the_config_resolves(project: Path) -> None:
    from pheasant_lab.console.topics import LOCAL_TOPICS, add_topic, topic_rows

    config = project / "configs" / "demo.yaml"
    shipped = (project / "configs" / "topics.demo.yaml").read_text(encoding="utf-8")
    result = add_topic(project, config, {}, TOPIC)

    assert result["override"] == f"experiment.topics_file={LOCAL_TOPICS}"
    assert (project / "configs" / "topics.demo.yaml").read_text(encoding="utf-8") == shipped
    resolved = load_config(
        config, overrides={"experiment.topics_file": LOCAL_TOPICS}, project_root=project
    )
    ids = [topic.id for topic in resolved.topics]
    assert ids == ["topic-tardigrade-radiotolerance", TOPIC["id"]]
    added = resolved.topics[1]
    assert added.seed_terms == ["lithium dendrite solid electrolyte", "garnet LLZO interphase"]
    assert added.date_range.from_ == "2018-01-01" and added.date_range.to is None

    listed = topic_rows(project, config, {"experiment.topics_file": LOCAL_TOPICS})
    assert listed["topics_file"] == LOCAL_TOPICS and len(listed["topics"]) == 2
    assert listed["topics"][1]["date_range"] == {"from": "2018-01-01", "to": None}

    # A second topic is added to the local file, not to the shipped one again.
    second = dict(TOPIC, id="topic-second", title="Second")
    add_topic(project, config, {"experiment.topics_file": LOCAL_TOPICS}, second)
    again = topic_rows(project, config, {"experiment.topics_file": LOCAL_TOPICS})
    assert [t["id"] for t in again["topics"]][-2:] == [TOPIC["id"], "topic-second"]


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"facets": []}, "no facets"),
        ({"id": "Topic With Spaces"}, "lowercase"),
        ({"facets": [{"id": "a-b", "label": "x", "weight": 0}]}, "weights are positive"),
        ({"facets": [{"id": "dup", "label": "x"}, {"id": "dup", "label": "y"}]}, "repeats"),
        ({"id": "topic-tardigrade-radiotolerance"}, "already exists"),
        ({"surprise": True}, "surprise"),
    ],
)
def test_a_topic_the_cli_would_refuse_is_refused_and_nothing_is_written(
    project: Path, patch: dict, message: str
) -> None:
    from pydantic import ValidationError

    from pheasant_lab.console.topics import LOCAL_TOPICS, add_topic

    with pytest.raises((ValueError, ValidationError), match=message):
        add_topic(project, project / "configs" / "demo.yaml", {}, {**TOPIC, **patch})
    assert not (project / LOCAL_TOPICS).exists()


def test_the_topic_routes_list_add_and_refuse(
    project: Path, fleet_run: Path, tmp_path: Path
) -> None:
    server = serve(
        host="127.0.0.1",
        port=0,
        project_root=project,
        output_root=fleet_run.parent,
        default_config="configs/demo.yaml",
        ui_dist=tmp_path / "no-ui",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(body: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"{base}/api/topics",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        status, listed = _get(f"{base}/api/topics?config=configs/demo.yaml")
        assert status == 200 and listed["topics_file"] == "configs/topics.demo.yaml"
        status, created = post({"config": "configs/demo.yaml", "topic": TOPIC})
        assert status == 201 and created["count"] == 2
        status, refused = post({"config": "configs/demo.yaml", "topic": {**TOPIC, "facets": []}})
        assert status == 400 and "facet" in refused["detail"]
        status, listed = _get(
            f"{base}/api/topics?config=configs/demo.yaml&set={created['override']}"
        )
        assert [t["id"] for t in listed["topics"]][-1] == TOPIC["id"]
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# a reload on any tab is the app, never JSON
# ---------------------------------------------------------------------------


def test_every_console_route_reloads_to_the_app(fleet_run: Path, tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    server = serve(
        host="127.0.0.1",
        port=0,
        project_root=REPO_ROOT,
        output_root=fleet_run.parent,
        default_config="configs/demo.yaml",
        ui_dist=dist,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    run = fleet_run.name
    # The routes ui/src/App.tsx declares, with real parameters.
    app = (REPO_ROOT / "ui" / "src" / "App.tsx").read_text(encoding="utf-8")
    import re

    declared = set(re.findall(r'<Route path="([^"*]+)"', app))
    paths = {p.replace(":runId", run).replace(":actor", "arm%3AP0") for p in declared}
    assert {"/", "/configure", f"/reports/{run}/traces/arm%3AP0"} <= paths
    try:
        for path in sorted(paths):
            with urllib.request.urlopen(f"{base}{path}", timeout=10) as response:
                assert response.status == 200, path
                assert response.headers["Content-Type"].startswith("text/html"), path
                assert b"id=root" in response.read(), path
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# a topic from an intent
# ---------------------------------------------------------------------------


INTENT = (
    "Why do some tardigrade species survive extreme radiation while others do not, "
    "and which survival claims failed to replicate?"
)


def test_a_draft_is_proposed_from_an_intent_and_writes_nothing(project: Path) -> None:
    from pheasant_lab.cli import _model
    from pheasant_lab.orchestration.drafter import draft_topic

    config = load_config(project / "configs" / "demo.yaml", project_root=project)
    before = sorted(p.name for p in (project / "configs").iterdir())
    draft = draft_topic(config, _model(config, "planner"), intent=INTENT, seed_terms=["Dsup"])
    again = draft_topic(config, _model(config, "planner"), intent=INTENT, seed_terms=["Dsup"])
    payload = draft.as_dict()

    assert payload == again.as_dict(), "the offline draft is not deterministic"
    assert payload["intent"] == INTENT and payload["seed_terms"][0] == "Dsup"
    assert any("replicate" in f["label"] for f in payload["facets"])
    assert payload["drafted_by"]["deterministic"] is True
    assert sorted(p.name for p in (project / "configs").iterdir()) == before
    with pytest.raises(ValueError, match="sentence"):
        draft_topic(config, _model(config, "planner"), intent="dsup")


def test_an_intent_alone_is_enough_and_the_planner_reads_it(project: Path) -> None:
    from pheasant_lab.console.topics import LOCAL_TOPICS, add_topic
    from pheasant_lab.orchestration.planner import _render

    topic = {**TOPIC, "id": "topic-intent-only", "seed_terms": [], "intent": INTENT}
    add_topic(project, project / "configs" / "demo.yaml", {}, topic)
    resolved = load_config(
        project / "configs" / "demo.yaml",
        overrides={"experiment.topics_file": LOCAL_TOPICS},
        project_root=project,
    )
    saved = resolved.topic("topic-intent-only")
    assert saved.intent == INTENT and saved.seed_terms == []
    assert INTENT in _render(saved, 3)

    with pytest.raises(ValueError, match="neither seed terms nor an intent"):
        add_topic(
            project,
            project / "configs" / "demo.yaml",
            {},
            {**TOPIC, "id": "topic-nothing", "seed_terms": [], "intent": "  "},
        )


def test_a_topic_without_an_intent_keeps_its_digest() -> None:
    """`intent` arrived after topics did; a topic that sets none digests as before."""

    config = load_config(REPO_ROOT / "configs" / "demo.yaml", project_root=REPO_ROOT)
    payload = config.redacted(__import__("pheasant_lab.redaction").redaction.Redactor())
    assert all("intent" in t for t in payload["topics"])  # present in the model
    with_intent = config.model_copy(deep=True)
    with_intent.topics[0].intent = INTENT
    assert with_intent.digest() != config.digest()
    # The pinned digest of the shipped demo config, computed before `intent` existed.
    assert config.digest() == (
        "sha256:db7e1d01b6d2af85a85d00d91ec3cf1388edff34f146b48e30bf1c471bb02876"
    )


def test_the_draft_route_runs_the_cli_and_refuses_in_words(
    project: Path, fleet_run: Path, tmp_path: Path
) -> None:
    server = serve(
        host="127.0.0.1",
        port=0,
        project_root=REPO_ROOT,
        output_root=fleet_run.parent,
        default_config="configs/demo.yaml",
        ui_dist=tmp_path / "no-ui",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(body: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"{base}/api/topics/draft",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        status, draft = post({"config": "configs/demo.yaml", "intent": INTENT})
        assert status == 200 and draft["intent"] == INTENT and draft["facets"]
        status, refused = post({"config": "configs/demo.yaml", "intent": "dsup"})
        assert status == 400 and "sentence" in refused["detail"]
    finally:
        server.shutdown()
        server.server_close()
