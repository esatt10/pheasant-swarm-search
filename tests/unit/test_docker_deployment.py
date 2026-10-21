"""The Docker deployment: the image, the Compose files and the bundled region agree.

Each assertion is a property a container run depended on. The 421 one was
found by running the lab's image against the bundled region: pheasant derives
its MCP host guard from ``server.api.cors_origins``, and a lab that reaches
the region by service name is refused on every call unless the origin is
admitted - which no in-process test can see, because a test client is
loopback.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from pheasant_lab.settings import load_config

REPO = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((REPO / path).read_text(encoding="utf-8"))


def _connections() -> dict[str, dict]:
    return _yaml("configs/connections.example.yaml")["connections"]


def test_the_bundled_region_takes_what_the_lab_sends() -> None:
    region = _yaml("deploy/pheasant/pheasant.yaml")
    docker = _connections()["docker"]
    assert region["pheasant"]["name"] == docker["knowledge_base"]
    assert region["server"]["mcp"]["enabled"] is True
    assert region["server"]["mcp"]["transports"]["streamable_http"] is True
    # The lab registers <state>/uploads/<source> as a document_folder source.
    assert region["pheasant"]["state_path"] == "/state"
    assert "/state" in region["security"]["allow_workspace_roots"]
    assert region["security"]["api_auth"]["token_env"] == docker["token_env"]
    assert any(source["type"] == "memory" for source in region["sources"])
    # The MCP host guard admits the service name the lab dials.
    origin = "{0.scheme}://{0.netloc}".format(urlsplit(docker["url"]))
    assert origin in region["server"]["api"]["cors_origins"]


def test_compose_points_the_lab_at_the_bundled_region() -> None:
    compose = _yaml("docker-compose.yml")
    lab = compose["services"]["lab"]
    pheasant = compose["services"]["pheasant"]
    url = re.search(r":-([^}]+)\}", lab["environment"]["PHEASANT_MCP_URL"]).group(1)
    assert url == _connections()["docker"]["url"]
    assert urlsplit(url).hostname == "pheasant"
    assert "./deploy/pheasant/pheasant.yaml:/config/pheasant.yaml:ro" in pheasant["volumes"]
    assert "0.14.0" in pheasant["image"]
    assert lab["depends_on"]["pheasant"]["condition"] == "service_healthy"
    assert {"lab-runs:/app/runs", "lab-local:/data/local"} <= set(lab["volumes"])
    # Both keys are demanded, never defaulted.
    for name in ("PHEASANT_API_TOKEN", "PHEASANT_LAB_CONSOLE_TOKEN"):
        value = str(lab["environment"][name])
        assert ":?" in value, name


def test_the_fleet_file_joins_the_lab_fleet_network() -> None:
    compose = _yaml("docker-compose.fleet.yml")
    lab = compose["services"]["lab"]
    url = re.search(r":-([^}]+)\}", lab["environment"]["PHEASANT_MCP_URL"]).group(1)
    assert url == _connections()["lab-fleet"]["url"]
    network = compose["networks"]["fleet"]
    assert network["external"] is True and "pheasant-lab_default" in network["name"]


def test_the_image_is_scoped_to_its_own_runs() -> None:
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    for line in (
        "PHEASANT_LAB_DEPLOYMENT=docker",
        "PHEASANT_LAB_RUN_SCOPE=docker",
        "PHEASANT_LAB_LOCAL_DIR=/data/local",
    ):
        assert line in dockerfile, line
    assert '"--config", "configs/docker.yaml"' in dockerfile
    assert '"--host", "0.0.0.0"' in dockerfile


def test_the_docker_experiment_talks_to_a_real_region(monkeypatch) -> None:
    monkeypatch.setenv("PHEASANT_MCP_URL", "http://pheasant:8765/mcp")
    config = load_config(REPO / "configs/docker.yaml", project_root=REPO, env_file=None)
    assert config.pheasant.transport == "streamable_http"
    assert config.pheasant.url == "http://pheasant:8765/mcp"
    for capability in ("ingest", "register_source", "sync", "search", "write_memory"):
        assert config.pheasant.capabilities[capability].tool
    # Free on first run: no hosted model, no paid provider.
    assert {spec.provider for spec in config.models.values()} == {"replay"}
    assert config.collection.providers == ["fixtures"]
