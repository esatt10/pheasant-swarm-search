"""Configuration resolution.

Precedence, lowest first:

1. committed defaults (the field defaults on the models below);
2. the experiment YAML and the sibling files it names;
3. ``.env`` / process environment, through ``${VAR}`` interpolation;
4. explicit ``--set dotted.path=value`` CLI overrides.

The resolved object is written to ``run-manifest.json`` with secrets removed
and a SHA-256 digest over it. Two runs are comparable only when every
differing resolved field is enumerated, which is what the digest makes
mechanical: equal digests need no enumeration, and unequal ones get one.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .hashing import digest
from .redaction import Redactor

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

ArmId = Literal["S0", "C0", "P0", "P1", "P2"]
ALL_ARMS: tuple[str, ...] = ("S0", "C0", "P0", "P1", "P2")
PHEASANT_ARMS: tuple[str, ...] = ("P0", "P1", "P2")


class ConfigError(ValueError):
    """A configuration a run must not start with."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# ---------------------------------------------------------------------------
# experiment.yaml
# ---------------------------------------------------------------------------


class ExperimentSection(_Model):
    name: str = "unnamed-experiment"
    seed: int = 0
    cost_budget_usd: float = 10.0
    runtime_budget_minutes: int = 120
    topics_file: str = "configs/topics.example.yaml"
    output_root: str = "runs"
    models_file: str = "configs/models.example.yaml"
    metrics_file: str = "configs/metrics.example.yaml"
    proof_policy_file: str = "configs/proof-policy.example.yaml"
    pheasant_file: str = "configs/pheasant-mcp.example.yaml"
    logging_file: str = "configs/logging.example.yaml"

    @field_validator("cost_budget_usd")
    @classmethod
    def _positive_budget(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("cost_budget_usd must be positive; a zero budget cannot reserve")
        return value


class CollectionSection(_Model):
    max_depth: int = 2
    max_research_agents: int = 6
    max_concurrent_agents: int = 3
    max_search_rounds_per_agent: int = 8
    max_sources_per_subtopic: int = 30
    allowed_source_types: list[str] = Field(
        default_factory=lambda: ["journal_article", "preprint", "review", "proceedings", "dataset"]
    )
    require_stable_identifier: bool = True
    permit_abstract_only: bool = True
    download_full_text_only_when_licensed: bool = True
    providers: list[str] = Field(
        default_factory=lambda: ["openalex", "crossref", "arxiv", "pubmed"]
    )
    provider_timeout_seconds: float = 20.0
    provider_max_retries: int = 3


class StoppingSection(_Model):
    minimum_sources_per_subtopic: int = 6
    minimum_independent_source_families: int = 3
    minimum_review_or_primary_sources: int = 2
    maximum_duplicate_rate: float = 0.25
    maximum_unresolved_critical_contradictions: int = 0
    marginal_unique_claim_window: int = 3
    marginal_unique_claim_threshold: float = 0.08
    minimum_ingest_receipt_rate: float = 0.98
    consecutive_saturated_rounds: int = 2
    evaluation_budget_reserve_fraction: float = 0.40


class CohortSplit(_Model):
    anchor_fraction: float = 0.25
    learned_fraction: float = 0.25
    temporal_holdout_fraction: float = 0.25
    control_fraction: float = 0.15
    invariant_fraction: float = 0.10

    @model_validator(mode="after")
    def _sums_to_one(self) -> CohortSplit:
        total = (
            self.anchor_fraction
            + self.learned_fraction
            + self.temporal_holdout_fraction
            + self.control_fraction
            + self.invariant_fraction
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"cohort fractions must sum to 1.0, got {total}")
        return self


class BenchmarkSection(_Model):
    freeze_before_evaluation: bool = True
    questions_per_topic: int = 24
    composition: dict[str, int] = Field(
        default_factory=lambda: {
            "atomic_fact": 6,
            "multi_source_synthesis": 6,
            "mechanism_or_causality": 4,
            "contradiction_or_uncertainty": 3,
            "temporal_or_versioned": 2,
            "source_identification": 1,
            "abstention": 2,
        }
    )
    require_expected_evidence_ids: bool = True
    blind_arm_order: bool = True
    cohorts: CohortSplit = Field(default_factory=CohortSplit)

    @model_validator(mode="after")
    def _composition_matches_total(self) -> BenchmarkSection:
        total = sum(self.composition.values())
        if total != self.questions_per_topic:
            raise ValueError(
                "benchmark.composition sums to "
                f"{total} but questions_per_topic is {self.questions_per_topic}; "
                "a composition that does not add up silently changes the question mix"
            )
        return self


class ReplaySection(_Model):
    mode: Literal["current_state", "historical"] = "current_state"
    as_of: str | None = None
    repetitions: int = 3
    fresh_session_per_question: bool = True
    randomize_arm_order: bool = True
    record_pheasant_snapshot: bool = True
    pairing_policy: Literal["complete_pairs_only", "available_pairs"] = "complete_pairs_only"
    max_search_rounds_per_answer: int = 4
    max_results_per_search: int = 10


class PrivacySection(_Model):
    store_prompt_text: bool = True
    store_response_text: bool = True
    redact_secrets: bool = True
    hash_principal_identifiers: bool = True
    allow_remote_trace_export: bool = False
    raw_prompt_retention_days: int | None = None
    retrieved_passage_retention_days: int | None = None
    downloaded_content_retention_days: int | None = None


class RefinementSection(_Model):
    enabled: bool = True
    submit_to_pheasant: bool = False
    diagnostic_knowledge_base: str = ""
    minimum_frequency: int = 2

    @model_validator(mode="after")
    def _isolated_namespace(self) -> RefinementSection:
        if self.submit_to_pheasant and not self.diagnostic_knowledge_base:
            raise ValueError(
                "refinement.submit_to_pheasant needs its own diagnostic_knowledge_base: "
                "diagnostic artifacts must not enter the namespace ordinary retrieval reads"
            )
        return self


class ExperimentFile(_Model):
    experiment: ExperimentSection = Field(default_factory=ExperimentSection)
    collection: CollectionSection = Field(default_factory=CollectionSection)
    stopping: StoppingSection = Field(default_factory=StoppingSection)
    benchmark: BenchmarkSection = Field(default_factory=BenchmarkSection)
    arms: list[str] = Field(default_factory=lambda: list(ALL_ARMS))
    replay: ReplaySection = Field(default_factory=ReplaySection)
    privacy: PrivacySection = Field(default_factory=PrivacySection)
    refinement: RefinementSection = Field(default_factory=RefinementSection)

    @field_validator("arms")
    @classmethod
    def _known_arms(cls, value: list[str]) -> list[str]:
        unknown = [arm for arm in value if arm not in ALL_ARMS]
        if unknown:
            raise ValueError(f"unknown arms: {unknown}; known arms are {list(ALL_ARMS)}")
        if not value:
            raise ValueError("at least one arm is required")
        return value


# ---------------------------------------------------------------------------
# models.yaml
# ---------------------------------------------------------------------------


class RoleModel(_Model):
    provider: str = "replay"
    model: str = "replay:default"
    reasoning_effort: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 4000
    tool_call_limit: int = 0
    criteria: dict[str, Any] = Field(default_factory=dict)


class PricingRef(_Model):
    source: str = "configs/pricing.example.yaml"
    fail_when_model_price_missing: bool = True


class BudgetAllocation(_Model):
    planning: float = 0.10
    collection: float = 0.40
    benchmark: float = 0.10
    evaluation: float = 0.30
    reserve: float = 0.10

    @model_validator(mode="after")
    def _sums_to_one(self) -> BudgetAllocation:
        total = self.planning + self.collection + self.benchmark + self.evaluation + self.reserve
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"budget.allocation must sum to 1.0, got {total}")
        return self

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class BudgetSection(_Model):
    allocation: BudgetAllocation = Field(default_factory=BudgetAllocation)
    reserve_output_at_max_tokens: bool = True
    assumed_tool_call_output_tokens: int = 400


class ModelsFile(_Model):
    models: dict[str, RoleModel] = Field(default_factory=dict)
    pricing: PricingRef = Field(default_factory=PricingRef)
    budget: BudgetSection = Field(default_factory=BudgetSection)


class ModelPrice(_Model):
    input: float
    output: float


class PricingFile(_Model):
    version: str = "unversioned"
    currency: str = "USD"
    unit: Literal["per_million_tokens", "per_thousand_tokens"] = "per_million_tokens"
    models: dict[str, ModelPrice] = Field(default_factory=dict)

    @field_validator("version", mode="before")
    @classmethod
    def _stringify(cls, value: Any) -> Any:
        return value.isoformat() if isinstance(value, date | datetime) else value


# ---------------------------------------------------------------------------
# metrics.yaml
# ---------------------------------------------------------------------------


class MetricsSection(_Model):
    primary: list[str] = Field(default_factory=list)
    diagnostic: list[str] = Field(default_factory=list)
    k: int = 10
    relative_lift_epsilon: float = 0.02


class ClassificationSection(_Model):
    practical_threshold: dict[str, float] = Field(default_factory=lambda: {"default": 0.05})
    lower_is_better: list[str] = Field(default_factory=list)
    minimum_paired_questions: int = 12
    minimum_pairing_coverage: float = 0.80
    require_interval_excludes_zero: bool = True
    protected_subgroups: list[str] = Field(default_factory=list)

    def threshold_for(self, metric: str) -> float:
        return self.practical_threshold.get(metric, self.practical_threshold.get("default", 0.05))

    def higher_is_better(self, metric: str) -> bool:
        return metric not in self.lower_is_better


class StatisticsSection(_Model):
    enabled: bool = True
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 0
    confidence_level: float = 0.95
    paired_tests: list[str] = Field(default_factory=lambda: ["mcnemar", "wilcoxon"])
    multiple_comparison_correction: Literal["benjamini_hochberg", "none"] = "benjamini_hochberg"
    false_discovery_rate: float = 0.10
    minimum_n_for_tests: int = 12


class NonInferioritySection(_Model):
    margin: dict[str, float] = Field(default_factory=dict)
    require_holdout_cohort: bool = True
    require_gates_pass: bool = True
    require_cost_within_budget: bool = True
    require_no_protected_subgroup_regression: bool = True


class GateSpec(_Model):
    enabled: bool = True
    max_violations: int | None = None
    max_rate: float | None = None
    min_rate: float | None = None
    min_accuracy: float | None = None
    max_delta: float | None = None
    max_findings: int | None = None
    max_items: int | None = None
    max_drifted_sections: int | None = None
    max_overrun_usd: float | None = None


class NormalisationSection(_Model):
    casefold: bool = True
    strip_accents: bool = True
    collapse_whitespace: bool = True
    strip_punctuation: bool = True
    number_words_to_digits: bool = True


class AnswerMatchingSection(_Model):
    normalize: NormalisationSection = Field(default_factory=NormalisationSection)
    numeric_relative_tolerance: float = 0.02
    require_citation_for_support: bool = True
    citation_must_be_retrieved: bool = True
    # Token overlap between a returned claim and the passage it cites, above
    # which the claim counts as supported. Not entailment; the metric says so.
    support_overlap_threshold: float = 0.5


class MetricsFile(_Model):
    metrics: MetricsSection = Field(default_factory=MetricsSection)
    classification: ClassificationSection = Field(default_factory=ClassificationSection)
    statistics: StatisticsSection = Field(default_factory=StatisticsSection)
    specialist_noninferiority: NonInferioritySection = Field(default_factory=NonInferioritySection)
    gates: dict[str, GateSpec] = Field(default_factory=dict)
    answer_matching: AnswerMatchingSection = Field(default_factory=AnswerMatchingSection)


# ---------------------------------------------------------------------------
# proof-policy.yaml
# ---------------------------------------------------------------------------


class ProofEventSpec(_Model):
    polarity: Literal["positive", "negative", "unknown"]
    base_weight: float


class MinimumEvidence(_Model):
    per_question_proof_events: int = 1
    per_metric_questions: int = 12
    per_cohort_questions: int = 6


class ConflictPolicy(_Model):
    report_rate: bool = True
    resolution: Literal["none", "majority"] = "none"


class ProofSection(_Model):
    version: int = 1
    publish_separately: list[str] = Field(default_factory=list)
    event_types: dict[str, ProofEventSpec] = Field(default_factory=dict)
    multipliers: dict[str, dict[str, float]] = Field(default_factory=dict)
    minimum_evidence: MinimumEvidence = Field(default_factory=MinimumEvidence)
    conflict: ConflictPolicy = Field(default_factory=ConflictPolicy)

    @model_validator(mode="after")
    def _unknown_weighs_nothing(self) -> ProofSection:
        for name, spec in self.event_types.items():
            if spec.polarity == "unknown" and spec.base_weight != 0.0:
                raise ValueError(
                    f"proof event '{name}' is polarity unknown with weight {spec.base_weight}: "
                    "an event nobody judged cannot carry weight"
                )
        return self


class JudgingSection(_Model):
    enabled: bool = False
    model: str = ""
    role: Literal["diagnostic_only"] = "diagnostic_only"
    requires_deterministic_agreement_sample: float = 0.20


class ProofPolicyFile(_Model):
    proof: ProofSection = Field(default_factory=ProofSection)
    judging: JudgingSection = Field(default_factory=JudgingSection)


# ---------------------------------------------------------------------------
# pheasant-mcp.yaml
# ---------------------------------------------------------------------------


class CapabilitySpec(_Model):
    tool: str = ""
    required: bool = False


class DiscoverySection(_Model):
    verify_configured_tool_exists: bool = True
    verify_input_schema: bool = True
    fail_on_ambiguous_capability: bool = True
    allow_heuristic_name_matching: bool = False


class IsolationSection(_Model):
    refuse_shared_knowledge_base: bool = False
    require_dedicated_source: bool = True
    forbid_memory_writes_in_arms: list[str] = Field(default_factory=lambda: ["S0", "C0", "P0"])


class PheasantFile(_Model):
    transport: Literal["streamable_http", "stdio", "mock"] = "streamable_http"
    url: str = ""
    command: str = ""
    token_env: str = "PHEASANT_MCP_TOKEN"
    protocol_version: str = "2026-07-28"
    timeout_seconds: float = 60.0
    connect_timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    retry_backoff_max_seconds: float = 20.0
    knowledge_base: str = ""
    source_name: str = "swarm-lab-literature"
    capabilities: dict[str, CapabilitySpec] = Field(default_factory=dict)
    discovery: DiscoverySection = Field(default_factory=DiscoverySection)
    argument_map: dict[str, Any] = Field(default_factory=dict)
    isolation: IsolationSection = Field(default_factory=IsolationSection)

    @model_validator(mode="after")
    def _transport_is_configured(self) -> PheasantFile:
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("transport streamable_http needs a url (PHEASANT_MCP_URL)")
        if self.transport == "stdio" and not self.command:
            raise ValueError(
                "transport stdio needs an explicit command (PHEASANT_MCP_COMMAND); "
                "a stdio transport that guesses its command starts the wrong server"
            )
        return self


# ---------------------------------------------------------------------------
# logging.yaml
# ---------------------------------------------------------------------------


class LoggingSection(_Model):
    level: str = "INFO"
    format: Literal["text", "json"] = "text"
    file: str | None = None


class SpansSection(_Model):
    enabled: bool = True
    file: str = "spans.jsonl"
    adopt_otel_ids_when_available: bool = True


class TranscriptSection(_Model):
    enabled: bool = True
    file: str = "mcp-calls.redacted.jsonl"
    store_request_body: bool = True
    store_response_body: bool = True
    max_body_bytes: int = 65536
    forbidden_headers: list[str] = Field(default_factory=lambda: ["authorization", "cookie"])


class TracingSection(_Model):
    raw_dir: str = "raw"
    flush_every_event: bool = True
    fsync_on_flush: bool = False
    record_payload_digest: bool = True
    verify_sequence_continuity: bool = True
    spans: SpansSection = Field(default_factory=SpansSection)
    mcp_transcript: TranscriptSection = Field(default_factory=TranscriptSection)


class ProjectionSection(_Model):
    backend: Literal["duckdb", "none"] = "duckdb"
    file: str = "projections/run.duckdb"
    rebuild_on_replay: bool = True
    verify_foreign_keys: bool = True


class ExportSection(_Model):
    otlp_enabled: bool = False
    otlp_endpoint: str | None = None
    otlp_headers_env: str | None = None


class LoggingFile(_Model):
    logging: LoggingSection = Field(default_factory=LoggingSection)
    tracing: TracingSection = Field(default_factory=TracingSection)
    projection: ProjectionSection = Field(default_factory=ProjectionSection)
    export: ExportSection = Field(default_factory=ExportSection)


# ---------------------------------------------------------------------------
# topics.yaml
# ---------------------------------------------------------------------------


class Facet(_Model):
    id: str
    label: str
    weight: float = 1.0


class DateRange(_Model):
    """A topic's publication window.

    YAML parses an unquoted ``2014-01-01`` into a ``date``, so the field
    accepts one and folds it to an ISO string. Everything downstream compares
    these as strings against provider metadata that is also a string, and a
    silently-typed date would compare unequal to every one of them.
    """

    from_: str | None = Field(default=None, alias="from")
    to: str | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("from_", "to", mode="before")
    @classmethod
    def _iso(cls, value: Any) -> Any:
        if isinstance(value, date | datetime):
            return value.isoformat()[:10]
        return value


class SourceAuthority(_Model):
    family_key: list[str] = Field(default_factory=lambda: ["corresponding_author"])
    preferred_types: list[str] = Field(default_factory=list)
    minimum_peer_reviewed: int = 0


class Topic(_Model):
    id: str
    title: str
    seed_terms: list[str] = Field(default_factory=list)
    date_range: DateRange = Field(default_factory=DateRange)
    facets: list[Facet] = Field(default_factory=list)
    source_authority: SourceAuthority = Field(default_factory=SourceAuthority)

    @model_validator(mode="after")
    def _has_facets(self) -> Topic:
        if not self.facets:
            raise ValueError(
                f"topic {self.id} has no facets; FacetCoverage would have no denominator"
            )
        return self


class TopicsFile(_Model):
    topics: list[Topic] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# the resolved whole
# ---------------------------------------------------------------------------


class LabConfig(_Model):
    """Every configuration file, resolved, in one object."""

    experiment: ExperimentSection
    collection: CollectionSection
    stopping: StoppingSection
    benchmark: BenchmarkSection
    arms: list[str]
    replay: ReplaySection
    privacy: PrivacySection
    refinement: RefinementSection
    models: dict[str, RoleModel]
    pricing_ref: PricingRef
    pricing: PricingFile
    budget: BudgetSection
    metrics: MetricsFile
    proof: ProofPolicyFile
    pheasant: PheasantFile
    logging: LoggingFile
    topics: list[Topic]
    # Provenance of the resolution itself.
    source_files: dict[str, str] = Field(default_factory=dict)
    overrides: dict[str, str] = Field(default_factory=dict)
    unresolved_env: list[str] = Field(default_factory=list)

    def redacted(self, redactor: Redactor) -> dict[str, Any]:
        return redactor.payload(self.model_dump(mode="json"))

    #: Fields that describe *where* a run happened rather than *what* it did.
    #: They are excluded from the configuration digest, because two runs are
    #: comparable or not on their experiment - not on the directory they were
    #: written to or the absolute path the config file sat at. Including them
    #: made a run that had been copied to another machine unresumable, which
    #: is the opposite of what the digest is for.
    ENVIRONMENTAL_FIELDS: ClassVar[tuple[str, ...]] = (
        "source_files",
        "overrides",
        "unresolved_env",
    )

    def digest(self, redactor: Redactor | None = None) -> str:
        """Digest the *redacted* resolution.

        Deliberately over the redacted form: a run manifest that two people
        can compare must not have a digest only one of them can reproduce.
        """

        payload = self.redacted(redactor or Redactor(enabled=True))
        for field_name in self.ENVIRONMENTAL_FIELDS:
            payload.pop(field_name, None)
        experiment = payload.get("experiment")
        if isinstance(experiment, dict):
            for field_name in (
                "output_root",
                "topics_file",
                "models_file",
                "metrics_file",
                "proof_policy_file",
                "pheasant_file",
                "logging_file",
            ):
                experiment.pop(field_name, None)
        return digest(payload)

    def role(self, name: str) -> RoleModel:
        try:
            return self.models[name]
        except KeyError as exc:
            raise ConfigError(f"no model configured for role '{name}'") from exc

    def topic(self, topic_id: str) -> Topic:
        for topic in self.topics:
            if topic.id == topic_id:
                return topic
        raise ConfigError(f"unknown topic '{topic_id}'")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def interpolate(value: Any, environ: Mapping[str, str], missing: list[str]) -> Any:
    """Substitute ``${VAR}`` / ``${VAR:-default}`` throughout a loaded tree."""

    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in environ and environ[name] != "":
                return environ[name]
            if default is not None:
                return default
            missing.append(name)
            return ""

        substituted = _ENV_REF.sub(_sub, value)
        return _coerce_scalar(substituted) if substituted != value else value
    if isinstance(value, Mapping):
        return {key: interpolate(item, environ, missing) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate(item, environ, missing) for item in value]
    return value


def _coerce_scalar(text: str) -> Any:
    """An interpolated value arrives as a string; give YAML scalars back."""

    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None if lowered != "" else ""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_dotenv(path: str | Path) -> dict[str, str]:
    """Read a ``.env`` file. Missing file is not an error; it is the default."""

    values: dict[str, str] = {}
    file = Path(path)
    if not file.is_file():
        return values
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return loaded


def apply_override(tree: dict[str, Any], dotted: str, raw: str) -> None:
    """Apply one ``a.b.c=value`` override to a loaded tree."""

    parts = dotted.split(".")
    node: Any = tree
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
            continue
        node = node.setdefault(part, {})
        if not isinstance(node, dict | list):
            raise ConfigError(f"override path '{dotted}' passes through a scalar at '{part}'")
    leaf = parts[-1]
    value = yaml.safe_load(raw)
    if isinstance(node, list):
        node[int(leaf)] = value
    else:
        node[leaf] = value


def resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate)
    return path if path.is_absolute() else (base / path)


def load_config(
    config_path: str | Path,
    *,
    overrides: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = ".env",
    project_root: str | Path | None = None,
) -> LabConfig:
    """Resolve every configuration file into one :class:`LabConfig`."""

    config_file = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root else Path.cwd()

    env: dict[str, str] = {}
    if env_file is not None:
        env.update(load_dotenv(resolve_path(root, str(env_file))))
    env.update(dict(environ if environ is not None else os.environ))

    missing: list[str] = []
    raw_experiment = interpolate(_read_yaml(config_file), env, missing)
    # An override is routed by its head key to the file that owns that
    # section. Applying every override to the experiment file first would make
    # `--set models.researcher.model=...` a validation error about an extra
    # field, which tells the operator nothing about what they got wrong.
    experiment_sections = set(ExperimentFile.model_fields)
    for dotted, value in (overrides or {}).items():
        if dotted.split(".", 1)[0] in experiment_sections:
            apply_override(raw_experiment, dotted, value)

    experiment_file = ExperimentFile.model_validate(raw_experiment)
    exp = experiment_file.experiment

    def _sibling(name: str) -> tuple[Path, dict[str, Any]]:
        path = resolve_path(root, name)
        return path, interpolate(_read_yaml(path), env, missing)

    models_path, raw_models = _sibling(exp.models_file)
    metrics_path, raw_metrics = _sibling(exp.metrics_file)
    proof_path, raw_proof = _sibling(exp.proof_policy_file)
    pheasant_path, raw_pheasant = _sibling(exp.pheasant_file)
    logging_path, raw_logging = _sibling(exp.logging_file)
    topics_path, raw_topics = _sibling(exp.topics_file)

    for dotted, value in (overrides or {}).items():
        head = dotted.split(".", 1)[0]
        if head == "models":
            apply_override(raw_models, dotted, value)
        elif head in {
            "metrics",
            "classification",
            "statistics",
            "gates",
            "specialist_noninferiority",
            "answer_matching",
        }:
            apply_override(raw_metrics, dotted, value)
        elif head in {"proof", "judging"}:
            apply_override(raw_proof, dotted, value)
        elif head in {
            "transport",
            "url",
            "capabilities",
            "discovery",
            "knowledge_base",
            "isolation",
            "argument_map",
        }:
            apply_override(raw_pheasant, dotted, value)
        elif head in {"tracing", "projection", "export"}:
            apply_override(raw_logging, dotted, value)

    models_file = ModelsFile.model_validate(raw_models)
    metrics_file = MetricsFile.model_validate(raw_metrics)
    proof_file = ProofPolicyFile.model_validate(raw_proof)
    pheasant_file = PheasantFile.model_validate(raw_pheasant)
    logging_file = LoggingFile.model_validate(raw_logging)
    topics_file = TopicsFile.model_validate(raw_topics)

    pricing_path = resolve_path(root, models_file.pricing.source or "configs/pricing.example.yaml")
    pricing = PricingFile.model_validate(interpolate(_read_yaml(pricing_path), env, missing))

    config = LabConfig(
        experiment=exp,
        collection=experiment_file.collection,
        stopping=experiment_file.stopping,
        benchmark=experiment_file.benchmark,
        arms=experiment_file.arms,
        replay=experiment_file.replay,
        privacy=experiment_file.privacy,
        refinement=experiment_file.refinement,
        models=models_file.models,
        pricing_ref=models_file.pricing,
        pricing=pricing,
        budget=models_file.budget,
        metrics=metrics_file,
        proof=proof_file,
        pheasant=pheasant_file,
        logging=logging_file,
        topics=topics_file.topics,
        source_files={
            "experiment": str(config_file),
            "models": str(models_path),
            "metrics": str(metrics_path),
            "proof_policy": str(proof_path),
            "pheasant": str(pheasant_path),
            "logging": str(logging_path),
            "topics": str(topics_path),
            "pricing": str(pricing_path),
        },
        overrides=dict(overrides or {}),
        unresolved_env=sorted(set(missing)),
    )
    _validate_cross_file(config)
    return config


def _validate_cross_file(config: LabConfig) -> None:
    """Checks no single file can make on its own."""

    required_roles = {"orchestrator", "planner", "researcher", "auditor", "benchmark_builder"}
    arm_roles = {
        "S0": "specialist",
        "C0": "control",
        "P0": "test_agent",
        "P1": "test_agent",
        "P2": "test_agent",
    }
    for arm in config.arms:
        required_roles.add(arm_roles[arm])
    absent = sorted(required_roles - set(config.models))
    if absent:
        raise ConfigError(
            f"roles configured nowhere in models.yaml: {absent}. "
            "A role with no model is a stage that cannot run, discovered mid-run."
        )

    if config.pricing_ref.fail_when_model_price_missing:
        priced = set(config.pricing.models)
        used = {role.model for role in config.models.values()}
        unpriced = sorted(used - priced)
        if unpriced:
            raise ConfigError(
                f"no price for {unpriced} in {config.source_files['pricing']}. "
                "A model treated as free is a budget guard that does not exist; "
                "add its price or set pricing.fail_when_model_price_missing: false."
            )

    if not config.topics:
        raise ConfigError("no topics configured")

    reserve = config.stopping.evaluation_budget_reserve_fraction
    evaluation_share = config.budget.allocation.evaluation + config.budget.allocation.reserve
    if reserve > evaluation_share + 1e-9:
        raise ConfigError(
            f"stopping.evaluation_budget_reserve_fraction ({reserve}) exceeds the evaluation "
            f"plus reserve allocation ({evaluation_share}); collection would be stopped by a "
            "reserve the budget never set aside"
        )
