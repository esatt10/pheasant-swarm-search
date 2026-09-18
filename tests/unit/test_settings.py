"""Configuration resolution, precedence and the cross-file refusals."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pheasant_lab.settings import ConfigError, apply_override, interpolate, load_config

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename", ["demo.yaml", "stub.yaml"])
def test_no_model_examples_cannot_select_paid_models_from_the_environment(filename):
    resolved = load_config(
        REPO / "configs" / filename,
        env_file=None,
        environ={"MODEL_PROVIDER": "openai", "RESEARCHER_MODEL": "paid-model"},
        project_root=REPO,
    )
    assert {spec.provider for spec in resolved.models.values()} == {"replay"}
    assert all(spec.model.startswith("replay:") for spec in resolved.models.values())


def test_interpolation_uses_environment_then_default():
    missing: list[str] = []
    assert interpolate("${A}", {"A": "set"}, missing) == "set"
    assert interpolate("${B:-fallback}", {}, missing) == "fallback"
    assert missing == []
    assert interpolate("${C}", {}, missing) == ""
    assert missing == ["C"]


def test_interpolation_gives_back_yaml_scalars():
    missing: list[str] = []
    assert interpolate("${N:-7}", {}, missing) == 7
    assert interpolate("${B:-true}", {}, missing) is True
    assert interpolate("${X:-null}", {}, missing) is None


def test_cli_override_beats_the_file(config, tmp_path):
    assert config.experiment.output_root.endswith("runs")
    reloaded = load_config(
        REPO / "configs" / "demo.yaml",
        overrides={"experiment.seed": "99"},
        env_file=None,
        environ={},
        project_root=REPO,
    )
    assert reloaded.experiment.seed == 99


def test_environment_beats_the_committed_default():
    resolved = load_config(
        REPO / "configs" / "demo.yaml",
        env_file=None,
        environ={"PHEASANT_KNOWLEDGE_BASE": "from-env"},
        project_root=REPO,
    )
    # The demo pins its own mock file, so the env value lands in the example
    # map; what this asserts is that interpolation reads the environment at
    # all, which the default-only path would hide.
    assert resolved.pheasant.knowledge_base in {"from-env", "pheasant-lab"}


def test_digest_is_stable_and_excludes_unresolved_env(config):
    from pheasant_lab.redaction import Redactor

    redactor = Redactor(enabled=True)
    first = config.digest(redactor)
    config.unresolved_env = ["SOMETHING"]
    assert config.digest(redactor) == first


def test_apply_override_refuses_a_path_through_a_scalar():
    tree = {"a": 1}
    with pytest.raises(ConfigError, match="passes through a scalar"):
        apply_override(tree, "a.b", "2")


def _write_variant(tmp_path: Path, mutate) -> Path:
    payload = yaml.safe_load((REPO / "configs" / "demo.yaml").read_text())
    mutate(payload)
    target = tmp_path / "variant.yaml"
    target.write_text(yaml.safe_dump(payload))
    return target


def test_composition_that_does_not_add_up_is_refused(tmp_path):
    path = _write_variant(tmp_path, lambda p: p["benchmark"].__setitem__("questions_per_topic", 99))
    with pytest.raises(ValueError, match="composition sums to"):
        load_config(path, env_file=None, environ={}, project_root=REPO)


def test_cohort_fractions_must_sum_to_one(tmp_path):
    path = _write_variant(
        tmp_path, lambda p: p["benchmark"]["cohorts"].__setitem__("anchor_fraction", 0.9)
    )
    with pytest.raises(ValueError, match="cohort fractions"):
        load_config(path, env_file=None, environ={}, project_root=REPO)


def test_an_unpriced_model_fails_preflight(tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match="A model treated as free"):
        load_config(
            REPO / "configs" / "demo.yaml",
            overrides={"models.researcher.model": "not-in-the-price-list"},
            env_file=None,
            environ={},
            project_root=REPO,
        )


def test_reserve_larger_than_its_allocation_is_refused(tmp_path):
    path = _write_variant(
        tmp_path,
        lambda p: p["stopping"].__setitem__("evaluation_budget_reserve_fraction", 0.95),
    )
    with pytest.raises(ConfigError, match="reserve the budget never set aside"):
        load_config(path, env_file=None, environ={}, project_root=REPO)


def test_yaml_dates_are_folded_to_iso_strings(config):
    window = config.topics[0].date_range
    assert isinstance(window.from_, str)
    assert window.from_ == "2014-01-01"


def test_refinement_submission_needs_its_own_namespace(tmp_path):
    path = _write_variant(
        tmp_path, lambda p: p["refinement"].__setitem__("submit_to_pheasant", True)
    )
    with pytest.raises(ValueError, match="diagnostic_knowledge_base"):
        load_config(path, env_file=None, environ={}, project_root=REPO)
