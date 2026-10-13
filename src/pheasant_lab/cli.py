"""The command line.

Every mutating command takes ``--resume``, ``--dry-run`` and
``--max-cost-usd``. A resumed run uses the **original** resolved-configuration
digest unless a forked run is explicitly created: resuming with a different
configuration produces one run directory whose halves are not comparable, and
nothing downstream could tell.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__, ids, promptlib
from .benchmark.builder import BenchmarkBuilder
from .benchmark.freezer import freeze, load_frozen, source_ledger_digest, verify_freeze
from .benchmark.leakage import check_leakage
from .budget import BudgetExceeded, CostLedger
from .lifecycle import (
    RunPaths,
    RunState,
    build_manifest,
    read_checksums,
    read_json,
    run_paths,
    write_checksums,
    write_json,
)
from .logsetup import configure_logging
from .models import build as build_model
from .pheasant.capabilities import CapabilityMap, MissingCapability, enforce, resolve
from .pheasant.client import PheasantClient
from .pheasant.ingestion import Ingestor
from .pheasant.mock import MockPheasantServer
from .pheasant.retrieval import Retriever
from .providers.base import ProviderError, ProviderRegistry, build_provider
from .redaction import Redactor
from .settings import ConfigError, LabConfig, RoleModel, load_config
from .tracing.events import Tracer, read_jsonl

LOG = logging.getLogger("pheasant_lab")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
#: An unhandled exception. Kept apart from ``1`` - "completed, and a decision
#: failed" - because a supervisor resumes a crash and must not resume a result.
EXIT_CRASHED = 3


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


@dataclass
class Session:
    config: LabConfig
    redactor: Redactor
    run_id: str
    paths: RunPaths
    tracer: Tracer
    ledger: CostLedger
    state: RunState
    manifest: dict[str, Any]
    client: PheasantClient | None = None
    capabilities: CapabilityMap | None = None
    ingestor: Ingestor | None = None
    retriever: Retriever | None = None
    mock: MockPheasantServer | None = None
    dry_run: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        self.tracer.close()


def _redactor(config: LabConfig) -> Redactor:
    redactor = Redactor(enabled=config.privacy.redact_secrets)
    redactor.register_environment()
    redactor.register("pheasant_mcp_token", os.environ.get(config.pheasant.token_env))
    return redactor


def _resolve_config(args: argparse.Namespace) -> LabConfig:
    overrides = dict(pair.split("=", 1) for pair in (getattr(args, "set", None) or []))
    return load_config(
        args.config,
        overrides=overrides,
        env_file=getattr(args, "env_file", ".env"),
        project_root=getattr(args, "project_root", None) or Path.cwd(),
    )


def open_session(
    args: argparse.Namespace,
    *,
    create: bool = True,
    connect: bool = True,
) -> Session:
    """Resolve configuration, open (or resume) a run, connect to the region."""

    if getattr(args, "run", None):
        return resume_session(args, connect=connect)

    config = _resolve_config(args)
    redactor = _redactor(config)
    config_digest = config.digest(redactor)
    nonce = ids.new_nonce()
    run_id = ids.run_id(config.experiment.name, config_digest, nonce)
    paths = run_paths(config.experiment.output_root, run_id)
    if create:
        paths.ensure()
        configure_logging(
            config.logging.logging, paths.root, verbose=bool(getattr(args, "verbose", False))
        )

    manifest = build_manifest(
        config=config,
        run_id=run_id,
        nonce=nonce,
        config_digest=config_digest,
        redactor=redactor,
        command=" ".join(sys.argv[1:]),
        package_version=__version__,
    )
    manifest["prompt_digests"] = promptlib.digest_all()
    if create:
        write_json(paths.manifest, manifest)
        _write_resolved(paths, config, redactor)

    tracer = Tracer(
        paths,
        run_id=run_id,
        config_digest=config_digest,
        redactor=redactor,
        flush=config.logging.tracing.flush_every_event,
        fsync=config.logging.tracing.fsync_on_flush,
        spans_enabled=config.logging.tracing.spans.enabled,
    )
    ledger = CostLedger.from_config(config, budget_override=getattr(args, "max_cost_usd", None))
    state = RunState.load(paths.state)
    state.barrier = tracer.sync
    state.update(run_id=run_id, config_digest=config_digest, started_at=manifest["created_at"])

    session = Session(
        config=config,
        redactor=redactor,
        run_id=run_id,
        paths=paths,
        tracer=tracer,
        ledger=ledger,
        state=state,
        manifest=manifest,
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    tracer.emit(
        "run.started", payload={"command": manifest["command"], "config_digest": config_digest}
    )
    if connect and not session.dry_run:
        _connect(session, args)
    return session


def resume_session(
    args: argparse.Namespace, *, connect: bool = True, read_only: bool = False
) -> Session:
    """Continue an existing run under its original resolved configuration."""

    config = _resolve_config(args)
    paths = run_paths(config.experiment.output_root, args.run)
    if not paths.manifest.is_file():
        raise ConfigError(f"no run at {paths.root}")
    configure_logging(
        config.logging.logging, paths.root, verbose=bool(getattr(args, "verbose", False))
    )
    manifest = read_json(paths.manifest)
    redactor = _redactor(config)
    current_digest = config.digest(redactor)
    recorded = str(manifest.get("config_digest"))
    if current_digest != recorded and not getattr(args, "fork", False):
        raise ConfigError(
            f"the configuration has changed since this run started "
            f"({recorded} -> {current_digest}). Resuming under a different configuration produces "
            "one run directory whose halves are not comparable. Re-run without --run to start a "
            "new run, or pass --fork to record this as a deliberate fork."
        )
    if current_digest != recorded:
        manifest.setdefault("forks", []).append(
            {
                "from_digest": recorded,
                "to_digest": current_digest,
                "command": " ".join(sys.argv[1:]),
            }
        )
        manifest["config_digest"] = current_digest
        write_json(paths.manifest, manifest)

    tracer = Tracer(
        paths,
        run_id=str(manifest["run_id"]),
        config_digest=str(manifest["config_digest"]),
        redactor=redactor,
        flush=config.logging.tracing.flush_every_event,
        spans_enabled=config.logging.tracing.spans.enabled,
        resume=True,
        read_only=read_only,
    )
    ledger = CostLedger.from_config(config, budget_override=getattr(args, "max_cost_usd", None))
    _restore_spend(ledger, paths)
    session = Session(
        config=config,
        redactor=redactor,
        run_id=str(manifest["run_id"]),
        paths=paths,
        tracer=tracer,
        ledger=ledger,
        state=RunState.load(paths.state),
        manifest=manifest,
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    if not read_only:
        session.state.barrier = tracer.sync
        for repaired in tracer.repairs:
            # The previous process died mid-append. The fragment is kept in
            # integrity/torn/ and the cut is part of the trace, not a secret.
            tracer.emit("run.repaired", status="partial", payload=repaired)
        tracer.emit("run.resumed", payload={"command": " ".join(sys.argv[1:])})
    if connect and not session.dry_run:
        _connect(session, args)
    return session


def _restore_spend(ledger: CostLedger, paths: RunPaths) -> None:
    """Re-commit what the previous process already spent.

    A resumed run that starts its budget at zero can spend the whole allowance
    twice, which is the one failure mode a budget guard must not have.
    """

    for record in read_jsonl(paths.raw_file("events.jsonl")):
        if record.get("event_type") != "cost.model_call":
            continue
        payload = record.get("payload") or {}
        bucket = str(payload.get("bucket") or "reserve")
        ledger.commit(bucket, float(payload.get("actual_usd") or 0.0))


def _connect(session: Session, args: argparse.Namespace) -> None:
    config = session.config
    token = os.environ.get(config.pheasant.token_env) or None
    if config.pheasant.transport == "mock" or getattr(args, "mock", False):
        server = MockPheasantServer(
            knowledge_base=config.pheasant.knowledge_base or "pheasant-lab",
            protocol_version=config.pheasant.protocol_version,
            # The mock region lives beside the run so that every command in a
            # run talks to the same region, and so a second run starts from a
            # clean one. A shared region would make two demo runs disagree
            # about what they contain.
            state_path=session.paths.root / "mock-region.json",
            claim_seconds=config.pheasant.mock_claim_seconds,
        )
        session.mock = server
        client = PheasantClient.in_process(
            config.pheasant,
            server,
            redactor=session.redactor,
            tracer=session.tracer,
            package_version=__version__,
        )
    else:
        client = PheasantClient(
            config.pheasant,
            redactor=session.redactor,
            tracer=session.tracer,
            token=token,
            package_version=__version__,
        )
    session_info = client.connect()
    capabilities = resolve(config.pheasant, session_info.tools)
    session.client = client
    session.capabilities = capabilities
    from .pheasant.receipts import ReceiptLedger

    session.ingestor = Ingestor(
        client,
        capabilities,
        config.pheasant,
        run_id=session.run_id,
        tracer=session.tracer,
        # A resumed run carries on with the receipts the previous process held.
        ledger=ReceiptLedger.restore(
            read_jsonl(session.paths.raw_file("ingest-receipts.jsonl")),
            read_jsonl(session.paths.raw_file("ingest-requests.jsonl")),
        ),
    )
    session.retriever = Retriever(
        client, capabilities, config.pheasant, store_text=config.privacy.store_response_text
    )


def _write_resolved(paths: RunPaths, config: LabConfig, redactor: Redactor) -> None:
    import yaml

    paths.resolved_config.write_text(
        yaml.safe_dump(config.redacted(redactor), sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )


def _providers(config: LabConfig, *, offline: bool, failures: list[str] | None = None) -> list[Any]:
    names = ["fixtures"] if offline else list(config.collection.providers)
    built: list[Any] = []
    for name in names:
        try:
            key_env = ProviderRegistry.get(name).api_key_env
            built.append(
                build_provider(
                    name,
                    timeout=config.collection.provider_timeout_seconds,
                    contact_email=os.environ.get("LITERATURE_CONTACT_EMAIL") or None,
                    api_key=(os.environ.get(key_env) or None) if key_env else None,
                )
            )
        except ProviderError as exc:
            LOG.warning("provider %s unavailable: %s", name, exc)
            if failures is not None:
                failures.append(str(exc))
    return built


#: The error-ledger stage a role's model calls belong to.
_ROLE_STAGE = {
    "planner": "discovery",
    "orchestrator": "discovery",
    "auditor": "discovery",
    "researcher": "extraction",
    "benchmark_builder": "evaluation",
}


def _model(
    config: LabConfig, role: str, tracer: Tracer | None = None, *, spec: RoleModel | None = None
) -> Any:
    from .models.retry import RetryingProvider

    spec = spec or config.role(role)
    key_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(spec.provider)
    inner = build_model(spec, role=role, api_key=os.environ.get(key_env) if key_env else None)

    def record(exc: BaseException, attempt: int, wait: float) -> None:
        if tracer is not None:
            tracer.errors.record(
                exc,
                stage=_ROLE_STAGE.get(role, "answer"),
                component=f"models.{spec.provider}",
                operation=f"{role}.complete",
                attempt=attempt,
                retryable=True,
                resolution="retried",
                backoff_seconds=wait,
            )

    return RetryingProvider(inner, on_retry=record)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Refuse a paid run that cannot work, before it costs anything."""

    findings: list[str] = []
    notes: list[str] = []

    try:
        config = _resolve_config(args)
    except (ConfigError, ValueError) as exc:
        print(f"configuration: FAIL\n  {exc}")
        return EXIT_REFUSED

    notes.append(f"configuration resolved from {config.source_files['experiment']}")
    if config.unresolved_env:
        findings.append(
            f"environment variables referenced by the config and unset: {config.unresolved_env}"
        )

    for role in sorted(config.models):
        try:
            promptlib.load(role if role in promptlib.ROLE_FILES else "planner")
        except promptlib.PromptNotFound as exc:
            findings.append(str(exc))
    notes.append(f"{len(promptlib.digest_all())} prompts found and digested")

    ledger = CostLedger.from_config(config)
    for role, spec in sorted(config.models.items()):
        try:
            ledger.price(spec.model)
        except Exception as exc:
            findings.append(f"pricing for role {role}: {exc}")
    notes.append(f"budget {config.experiment.cost_budget_usd:.2f} {config.pricing.currency}")

    offline = bool(getattr(args, "offline", False))
    provider_failures: list[str] = []
    providers = _providers(config, offline=offline, failures=provider_failures)
    # A configured provider that cannot be built is a finding, not a note: a
    # web profile whose only web providers lack keys would otherwise collect
    # nothing and report every facet short.
    findings.extend(f"provider {failure}" for failure in provider_failures)
    if not providers:
        findings.append("no literature provider could be constructed")
    else:
        notes.append(
            f"collection profile: {config.collection.profile}; "
            f"providers: {', '.join(p.name for p in providers)}"
        )

    redactor = _redactor(config)
    notes.append(f"{redactor.known} secret value(s) registered for redaction")

    # The experiment asks for neighbourhoods; only the pheasant file can say
    # how to send them. Without the mapping every search would go out
    # unexpanded and the run would record nothing where it meant to record
    # the graph - the unpinned-pin shape, one argument over.
    search_map = config.pheasant.argument_map.get("search") or {}
    if config.replay.expansion is not None and "expand" not in search_map:
        findings.append(
            "replay.graph_expansion is set but the pheasant file's argument_map.search does "
            "not map `expand`; map it (pheasant >= 0.13.1) or "
            "unset replay.graph_expansion"
        )
    from .catalog import unsupported_effort

    for role, spec in sorted(config.models.items()):
        problem = unsupported_effort(spec.model, spec.reasoning_effort)
        if problem:
            findings.append(f"models.{role}: {problem}")
    if config.replay.min_score is not None and "min_score" not in search_map:
        findings.append(
            "replay.min_score is set but the pheasant file's argument_map.search does not map "
            "`min_score`; map it or unset replay.min_score"
        )

    transport = config.pheasant.transport
    if getattr(args, "mock", False) or transport == "mock":
        notes.append(
            "connecting to the in-process mock region. A mock result measures the mock: "
            "doctor will not report a mock as a live target."
        )
    try:
        namespace = argparse.Namespace(**vars(args))
        session = open_session(namespace, create=False, connect=True)
    except (MissingCapability, ConnectionError, OSError, RuntimeError) as exc:
        findings.append(f"Pheasant MCP: {exc}")
        session = None
    else:
        assert session.client is not None and session.capabilities is not None
        info = session.client.session
        assert info is not None
        notes.append(
            f"MCP: {info.server_name} {info.server_version}, protocol {info.protocol_version}, "
            f"{len(info.tools)} tools"
        )
        if info.protocol_version != config.pheasant.protocol_version:
            notes.append(
                f"protocol negotiated down from {config.pheasant.protocol_version} to "
                f"{info.protocol_version}"
            )
        for resolution in sorted(session.capabilities.resolutions.values(), key=lambda r: r.name):
            mark = "ok" if resolution.usable else ("MISSING" if resolution.required else "absent")
            line = f"  capability {resolution.name} -> {resolution.configured_tool or '(unset)'}: {mark}"
            if resolution.reason:
                line += f" ({resolution.reason})"
            notes.append(line)
        try:
            enforce(session.capabilities)
        except MissingCapability as exc:
            findings.append(str(exc))
        session.close()

    print("# doctor\n")
    for note in notes:
        print(f"- {note}")
    print()
    if findings:
        print(f"**FAIL** — {len(findings)} finding(s):\n")
        for finding in findings:
            print(f"- {finding}")
        return EXIT_REFUSED
    print("**PASS** — this laptop can run the experiment.")
    return EXIT_OK


def cmd_plan(args: argparse.Namespace) -> int:
    """Project cost and volume without a model or ingest call."""

    config = _resolve_config(args)
    ledger = CostLedger.from_config(config, budget_override=getattr(args, "max_cost_usd", None))
    topic = config.topic(args.topic) if getattr(args, "topic", None) else config.topics[0]

    agents = config.collection.max_research_agents * config.collection.max_depth
    searches = agents * config.collection.max_search_rounds_per_agent
    questions = config.benchmark.questions_per_topic
    arms = len(config.arms)
    answers = questions * arms * config.replay.repetitions
    query_calls = answers * config.replay.max_search_rounds_per_answer

    def estimate(role: str, calls: int, prompt_chars: int) -> float:
        spec = config.role(role)
        one = ledger.estimate(
            model=spec.model,
            prompt="x" * prompt_chars,
            max_output_tokens=spec.max_output_tokens,
            tool_calls=spec.tool_call_limit,
        )
        return one.total_usd * calls

    # Every query goes to every configured provider, and a web search API
    # bills per request - so a web profile's collection cost is not only
    # model tokens. Read off the class, which needs no key to construct.
    per_query_usd = 0.0
    for name in config.collection.providers:
        try:
            per_query_usd += ProviderRegistry.get(name).usd_per_request
        except ProviderError:
            continue

    projection = {
        "topic": topic.id,
        "planning_usd": estimate("planner", config.collection.max_depth, 4000)
        + estimate("orchestrator", config.collection.max_depth, 4000),
        "collection_usd": estimate("researcher", agents, 12000),
        "search_api_usd": round(searches * per_query_usd, 4),
        "benchmark_usd": estimate("benchmark_builder", 1, 20000),
        "evaluation_usd": estimate("test_agent", answers, 8000)
        + estimate("specialist", questions * config.replay.repetitions, 8000),
        "searches": searches,
        "answers": answers,
        "mcp_search_calls": query_calls,
        "questions": questions,
        "budget_usd": ledger.total_budget_usd,
    }
    projection["projected_total_usd"] = round(
        projection["planning_usd"]
        + projection["collection_usd"]
        + projection["search_api_usd"]
        + projection["benchmark_usd"]
        + projection["evaluation_usd"],
        4,
    )
    projection["fits_budget"] = projection["projected_total_usd"] <= ledger.total_budget_usd

    if getattr(args, "json", False):
        print(json.dumps(projection, indent=2, sort_keys=True))
    else:
        print(f"# plan — {topic.title}\n")
        for key in (
            "planning_usd",
            "collection_usd",
            "search_api_usd",
            "benchmark_usd",
            "evaluation_usd",
            "projected_total_usd",
            "budget_usd",
        ):
            print(f"- {key}: {projection[key]:.4f}")
        print(f"- collection profile: {config.collection.profile}")
        print(f"- research branches: {agents} · provider searches: {searches}")
        print(f"- benchmark questions: {questions} · arms: {arms} · answers: {answers}")
        print(f"- MCP search calls: {query_calls}")
        print()
        print("fits budget" if projection["fits_budget"] else "**does not fit the budget**")
        print()
        print(
            "This is a worst-case projection: every model call is reserved at its output cap, "
            "which is what the guard does at run time."
        )
    return EXIT_OK if projection["fits_budget"] else EXIT_FAILED


def cmd_collect(args: argparse.Namespace) -> int:
    from .orchestration.orchestrator import Orchestrator
    from .orchestration.state import rehydrate

    session = open_session(args)
    try:
        config = session.config
        progress = session.state.get("collect_progress") if getattr(args, "run", None) else None
        if getattr(args, "topic", None):
            topic = config.topic(args.topic)
        elif progress:
            topic = config.topic(str(progress["topic_id"]))
        else:
            topic = config.topics[0]
        if session.dry_run:
            print(f"dry run: would collect topic {topic.id} into {session.paths.root}")
            return EXIT_OK
        if session.state.completed("collect") and not getattr(args, "force", False):
            print(f"collect already completed for {session.run_id}; pass --force to redo it")
            return EXIT_OK
        if progress and progress.get("topic_id") != topic.id:
            progress = None
        # Resuming an interrupted collection: the registry and the evidence
        # ledger come back from the raw trace, not from the checkpoint, and the
        # loop continues from the last round boundary that was made durable.
        prior = (
            rehydrate(
                session.run_id,
                topic.id,
                read_jsonl(session.paths.raw_file("sources.jsonl")),
                read_jsonl(session.paths.raw_file("claims.jsonl")),
                contradiction_rows=read_jsonl(session.paths.raw_file("contradictions.jsonl")),
            )
            if getattr(args, "run", None)
            else None
        )
        if prior is not None and not progress and not prior.sources:
            prior = None

        offline = bool(getattr(args, "offline", False)) or session.mock is not None
        orchestrator = Orchestrator(
            config,
            run_id=session.run_id,
            tracer=session.tracer,
            ledger=session.ledger,
            providers=_providers(config, offline=offline),
            planner_model=_model(config, "planner", session.tracer),
            researcher_model=_model(config, "researcher", session.tracer),
            auditor_model=_model(config, "auditor", session.tracer),
            ingestor=session.ingestor,
        )
        session.state.mark_stage("collect", "running")
        result = orchestrator.collect(
            topic,
            resume=progress or None,
            prior=prior,
            checkpoint=lambda record: session.state.set("collect_progress", record),
        )

        if session.ingestor is not None:
            session.ingestor.sync()
            session.ingestor.acknowledge()
            reconcile = session.ingestor.reconcile()
            session.state.set("reconcile", reconcile)
            inventory = session.ingestor.inventory()
            if inventory is not None:
                session.state.set("inventory", inventory)

        session.state.update(
            topic_id=topic.id,
            collection=result.as_dict(),
            budget=session.ledger.snapshot(),
        )
        session.state.mark_stage("collect", "completed", decision=result.decision.outcome)
        write_checksums(session.paths)

        print(f"run: {session.run_id}")
        print(f"decision: {result.decision.outcome} — {result.decision.reason}")
        summary = result.state.summary()
        print(
            f"sources: {summary['sources_retained']} retained of {summary['sources_discovered']} "
            f"discovered · claims: {summary['claims_eligible']} eligible"
        )
        print(f"cost: {session.ledger.committed_usd:.4f} of {session.ledger.total_budget_usd:.2f}")
        return EXIT_OK if result.sufficient else EXIT_FAILED
    except BudgetExceeded as exc:
        print(f"budget: {exc}")
        session.state.mark_stage("collect", "stopped_budget_incomplete")
        return EXIT_FAILED
    finally:
        session.close()


def cmd_audit(args: argparse.Namespace) -> int:
    from .orchestration.auditor import CoverageAuditor
    from .orchestration.state import rehydrate

    session = resume_session(args, connect=False)
    try:
        config = session.config
        topic = config.topic(str(session.state.get("topic_id") or config.topics[0].id))
        state = _rehydrated(session, topic.id, rehydrate)
        auditor = CoverageAuditor(config, topic, tracer=session.tracer)
        audit = auditor.audit(state)
        session.state.set("audit", audit.as_dict())
        if getattr(args, "json", False):
            print(json.dumps(audit.as_dict(), indent=2, sort_keys=True))
        else:
            print(f"# audit — {topic.id}\n")
            print(f"facet coverage: {audit.coverage_fraction:.2%}")
            print(
                f"duplicate rate: {audit.duplicate_rate:.2%} ({audit.duplicates}/{audit.acquired})"
            )
            for row in audit.facet_coverage:
                mark = "ok" if row["meets_minimum"] else "SHORT"
                print(
                    f"  {row['facet_id']}: {mark} — {row['sources']} sources, {row['families']} families, "
                    f"{row['peer_reviewed']} peer-reviewed, "
                    f"{row.get('authoritative', row['peer_reviewed'])} authoritative"
                )
            for gap in audit.quality_gaps:
                print(f"  gap {gap.facet_id}: {gap.kind} — {gap.detail}")
        return EXIT_OK
    finally:
        session.close()


def cmd_freeze_benchmark(args: argparse.Namespace) -> int:
    from .orchestration.state import rehydrate

    session = resume_session(args, connect=True)
    try:
        config = session.config
        topic = config.topic(str(session.state.get("topic_id") or config.topics[0].id))
        state = _rehydrated(session, topic.id, rehydrate)
        builder = BenchmarkBuilder(
            config,
            model=_model(config, "benchmark_builder", session.tracer),
            ledger=session.ledger,
            tracer=session.tracer,
            run_id=session.run_id,
        )
        built = builder.build(topic, state)

        snapshot: dict[str, Any] = {}
        if session.ingestor is not None:
            snapshot = session.ingestor.seal_snapshot(label=f"{session.run_id}:frozen")
        package = freeze(
            built,
            session.paths,
            config,
            run_id=session.run_id,
            source_ledger_digest=source_ledger_digest(
                read_jsonl(session.paths.raw_file("claims.jsonl"))
            ),
            snapshot=snapshot,
            prompt_digests=promptlib.digest_all(),
        )
        for question in built.questions:
            session.tracer.append("questions.jsonl", question.as_dict(run_id=session.run_id))

        corpus = {
            record["source_id"]: f"{record.get('title', '')} {record.get('abstract', '')}"
            for record in read_jsonl(session.paths.raw_file("sources.jsonl"))
        }
        leakage = check_leakage(
            built.questions,
            built.facts,
            research_prompts=[promptlib.load("researcher")],
            corpus_texts=corpus,
            namespace_paths=[
                record.get("relative_path", "")
                for record in read_jsonl(session.paths.raw_file("ingest-requests.jsonl"))
            ],
        )
        write_json(session.paths.benchmark / "leakage.json", leakage.as_dict())
        session.state.update(
            benchmark_version=package.version,
            benchmark_digest=package.digest,
            snapshot=snapshot,
            leakage=leakage.as_dict(),
        )
        session.state.mark_stage("freeze-benchmark", "completed", questions=len(built.questions))
        write_checksums(session.paths)

        print(f"benchmark {package.version} frozen at {package.directory}")
        print(f"  questions: {len(built.questions)} · expected facts: {len(built.facts)}")
        print(f"  composition: {built.composition_built}")
        if built.exclusions:
            print(f"  exclusions: {len(built.exclusions)} (see exclusions.jsonl)")
        print(f"  leakage: {'clean' if leakage.clean else 'FINDINGS'} ({len(leakage.findings)})")
        if snapshot.get("snapshot_id"):
            print(f"  snapshot: {snapshot['snapshot_id']}")
        return EXIT_OK if leakage.clean else EXIT_FAILED
    finally:
        session.close()


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .arms.base import ArmContext
    from .arms.pheasant_memory import MemorySeeder
    from .arms.tuned_replay import QueryTuner
    from .benchmark.leakage import LeakageFinding, LeakageReport
    from .evaluation.engine import EvaluationEngine
    from .evaluation.evidence import ProofLedger
    from .orchestration.state import rehydrate, research_package

    session = resume_session(args, connect=True)
    try:
        config = session.config
        if getattr(args, "arms", None):
            config.arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
        package = load_frozen(session.paths)
        findings = verify_freeze(session.paths)
        if findings:
            print("the frozen benchmark does not match its manifest:")
            for finding in findings:
                print(f"  - {finding}")
            return EXIT_REFUSED

        topic = config.topic(str(session.state.get("topic_id") or config.topics[0].id))
        state = _rehydrated(session, topic.id, rehydrate)
        snapshot = dict(session.state.get("snapshot") or {})
        snapshot_id = snapshot.get("snapshot_id")

        # A region whose search tool takes no snapshot pin cannot be pinned,
        # and the run says so rather than recording an id it never sent.
        pinning = session.retriever is not None and session.retriever.supports_pinning
        limitations: list[str] = []
        if snapshot_id and not pinning:
            limitations.append(
                f"This region's search tool accepts no snapshot pin, so `P0` ran unpinned. "
                f"Snapshot `{snapshot_id}` was used as a drift check instead: the run fails its "
                "snapshot gate if the region moved between the freeze and the start of "
                "evaluation."
            )
        elif not snapshot_id:
            limitations.append(
                "No snapshot was sealed for this run, so no drift check was possible."
            )

        contexts: dict[str, ArmContext] = {}
        for arm_id in config.arms:
            role = {"S0": "specialist", "C0": "control"}.get(arm_id, "test_agent")
            contexts[arm_id] = ArmContext(
                config=config,
                run_id=session.run_id,
                tracer=session.tracer,
                ledger=session.ledger,
                model=_model(config, role, session.tracer),
                retriever=session.retriever if arm_id in {"P0", "P1", "P2"} else None,
                # Only the arm that writes nothing is pinned to the sealed
                # snapshot. A pinned search is answered from that state *or
                # refused*, and P1's own treatment - writing memory - moves the
                # snapshot's memory section, so pinning it would refuse every
                # query it makes. The snapshot still guards P1 and P2: the
                # drift check below fails the run if any section other than
                # `memory` moved, which is the property those arms need.
                snapshot_id=snapshot_id if arm_id == "P0" else None,
                research_package=research_package(state) if arm_id == "S0" else None,
                memory_enabled=arm_id in {"P1", "P2"},
            )

        seeder = (
            MemorySeeder(
                session.client,
                session.capabilities,
                config.pheasant,
                tracer=session.tracer,
                matching=config.metrics.answer_matching,
                min_evidence=config.replay.memory_min_evidence or 1,
            )
            if session.client is not None and session.capabilities is not None
            else None
        )
        leakage_payload = session.state.get("leakage") or {}
        leakage = LeakageReport(
            findings=[
                LeakageFinding(
                    check=str(row.get("check")),
                    severity=str(row.get("severity")),
                    question_id=row.get("question_id"),
                    detail=str(row.get("detail")),
                    evidence=dict(row.get("evidence") or {}),
                )
                for row in leakage_payload.get("findings", [])
            ],
            checked=dict(leakage_payload.get("checked") or {}),
        )

        from .pheasant.receipts import fold_receipts

        receipts = list(
            fold_receipts(read_jsonl(session.paths.raw_file("ingest-receipts.jsonl"))).values()
        )
        accepted = sum(
            1 for r in receipts if str(r.get("status")) in {"accepted", "indexed", "verified"}
        )
        submitted = len(receipts)
        if session.retriever is not None:
            # The receipts are the join between what the region calls a hit
            # and which of this lab's sources it is; see `normalise_results`.
            session.retriever.artifact_sources = {
                str(r["artifact_id"]): str(r["source_id"])
                for r in receipts
                if r.get("artifact_id") and r.get("source_id")
            }
            paths = {
                str(row["idempotency_key"]): str(row["relative_path"])
                for row in read_jsonl(session.paths.raw_file("ingest-requests.jsonl"))
                if row.get("idempotency_key") and row.get("relative_path")
            }
            session.retriever.artifact_paths = {
                str(r["artifact_id"]): paths[str(r["idempotency_key"])]
                for r in receipts
                if r.get("artifact_id") and str(r.get("idempotency_key")) in paths
            }

        # An evaluation a crash interrupted resumes from the answers it had
        # already recorded. Only possible when answers were stored whole: with
        # response text withheld a recorded answer cannot be re-scored.
        recorded: list[dict[str, Any]] = []
        if (
            session.state.stage_status("evaluate") == "running"
            and config.privacy.store_response_text
        ):
            recorded = list(read_jsonl(session.paths.raw_file("answers.jsonl")))
        engine = EvaluationEngine(
            config,
            run_id=session.run_id,
            paths=session.paths,
            tracer=session.tracer,
            ledger=session.ledger,
            package=package,
            arm_contexts=contexts,
            proof_ledger=ProofLedger(config.proof, run_id=session.run_id),
            leakage=leakage,
            memory_seeder=seeder,
            query_tuner=QueryTuner(),
            region_artifact_ids=[
                str(r.get("artifact_id")) for r in receipts if r.get("artifact_id")
            ],
            snapshot_verification=_verify_snapshot(session, snapshot_id),
            reconcile=dict(session.state.get("reconcile") or {}),
            receipt_rate=(accepted, submitted),
            limitations=limitations,
            recorded_answers=recorded,
        )
        session.state.mark_stage("evaluate", "running")
        result = engine.run()
        if engine.reused:
            session.tracer.emit(
                "evaluation.resumed",
                payload={"answers_reused": engine.reused, "answers_total": len(result.answers)},
            )
        session.state.update(budget=session.ledger.snapshot())
        session.state.mark_stage("evaluate", "completed", answers=len(result.answers))
        write_checksums(session.paths)

        print(f"evaluated {len(result.answers)} answers over {len(package.questions)} questions")
        print(f"hard gates: {result.gates.verdict.value}")
        for arm, decision in sorted(result.non_inferiority.items()):
            print(
                f"  {arm} rivals S0: {decision.get('rivals_specialist')} ({decision.get('status')})"
            )
        print(f"cost: {session.ledger.committed_usd:.4f} of {session.ledger.total_budget_usd:.2f}")
        return EXIT_OK
    finally:
        session.close()


# Sections a treatment is *supposed* to move. Memory is P1's independent
# variable; a corpus that moved between arms is a different experiment.
EXPECTED_DRIFT = frozenset({"memory"})


def _verify_snapshot(session: Session, snapshot_id: str | None) -> dict[str, Any]:
    if not snapshot_id or session.ingestor is None:
        return {}
    payload = session.ingestor.get_snapshot(snapshot_id)
    verification = payload.get("verification") or {}
    drifted = list(verification.get("drifted_sections") or [])
    return {
        "snapshot_id": snapshot_id,
        "current": verification.get("current"),
        "drifted_sections": drifted,
        "expected_drift": sorted(section for section in drifted if section in EXPECTED_DRIFT),
        "unexpected_drift": sorted(section for section in drifted if section not in EXPECTED_DRIFT),
    }


def cmd_replay(args: argparse.Namespace) -> int:
    """Rebuild every projection and metric from the raw events alone."""

    from .tracing.duckdb_projection import ProjectionUnavailable, project, verify_raw

    session = resume_session(args, connect=False, read_only=True)
    try:
        findings = verify_raw(session.paths, check_digests=True)
        if findings:
            print("raw trace findings:")
            for finding in findings:
                print(f"  - {finding}")
        try:
            report = project(
                session.paths,
                verify_foreign_keys=session.config.logging.projection.verify_foreign_keys,
            )
        except ProjectionUnavailable as exc:
            print(f"projection skipped: {exc}")
            return EXIT_OK if not findings else EXIT_FAILED
        write_json(session.paths.projections / "projection.json", report.as_dict())
        print(f"projected {sum(report.tables.values())} rows into {report.database}")
        for name, count in sorted(report.tables.items()):
            if count:
                print(f"  {name}: {count}")
        for finding in report.findings:
            print(f"  finding: {finding}")
        return EXIT_OK if report.clean and not findings else EXIT_FAILED
    finally:
        session.close()


def cmd_verify(args: argparse.Namespace) -> int:
    """Hashes, completeness, pairing, leakage and gates."""

    from .tracing.duckdb_projection import verify_raw
    from .tracing.lineage import LineageIndex

    session = resume_session(args, connect=False, read_only=True)
    try:
        paths = session.paths
        findings: list[str] = []

        recorded = read_checksums(paths)
        if not recorded:
            findings.append("no integrity/checksums.sha256 was written for this run")
        else:
            from .hashing import digest_file

            for name, value in sorted(recorded.items()):
                path = paths.root / name
                if not path.is_file():
                    findings.append(f"file recorded in checksums is missing: {name}")
                    continue
                if digest_file(path) != value:
                    findings.append(f"file changed after it was checksummed: {name}")

        findings.extend(verify_raw(paths, check_digests=True))
        findings.extend(verify_freeze(paths))

        lineage = LineageIndex(paths)
        for kind, items in sorted(lineage.orphans().items()):
            findings.append(f"{kind}: {len(items)} (e.g. {items[:3]})")
        missing_receipts = lineage.sources_without_receipt()
        if missing_receipts:
            findings.append(
                f"{len(missing_receipts)} submitted sources carry no ingest receipt: "
                f"{missing_receipts[:3]}"
            )

        leakage = session.state.get("leakage") or {}
        invalidating = [
            f for f in leakage.get("findings", []) if f.get("severity") == "invalidating"
        ]
        if invalidating:
            findings.append(f"{len(invalidating)} invalidating leakage findings")

        gates_path = paths.metrics / "gates.json"
        verdict = "not evaluated"
        if gates_path.is_file():
            gates = read_json(gates_path)
            verdict = str(gates.get("verdict"))
            if verdict != "PASS":
                findings.append(f"hard gates: {verdict}")

        verification = {
            "run_id": session.run_id,
            "checked_files": len(recorded),
            "findings": findings,
            "gates": verdict,
            "clean": not findings,
        }
        write_json(paths.integrity / "verification.json", verification)

        print(f"# verify — {session.run_id}\n")
        print(f"- files checksummed: {len(recorded)}")
        print(f"- hard gates: {verdict}")
        if findings:
            print(f"\n**{len(findings)} finding(s):**\n")
            for finding in findings:
                print(f"- {finding}")
            return EXIT_FAILED
        print("\n**clean** — every checksum, digest, lineage link and gate checked out.")
        return EXIT_OK
    finally:
        session.close()


def cmd_report(args: argparse.Namespace) -> int:
    from .evaluation import operational_metrics
    from .reports import (
        build_candidates,
        health_vector,
        write_arm_comparison,
        write_collection_report,
        write_errors_report,
        write_query_detail,
        write_refinements,
        write_regressions,
        write_summary,
    )
    from .tracing.lineage import LineageIndex

    session = resume_session(args, connect=False)
    try:
        paths = session.paths
        result = _load_result(paths)
        questions = {
            row["question_id"]: row for row in read_jsonl(paths.raw_file("questions.jsonl"))
        }
        errors = list(read_jsonl(paths.raw_file("errors.jsonl")))
        events = list(read_jsonl(paths.raw_file("events.jsonl")))
        mcp_calls = [
            {**(event.get("payload") or {}), "status": event.get("status")}
            for event in events
            if event.get("event_type") == "mcp.tool.call"
        ]
        collection = session.state.get("collection") or {}
        budget = session.state.get("budget") or session.ledger.snapshot()

        from .pheasant.receipts import fold_receipts

        receipts = list(fold_receipts(read_jsonl(paths.raw_file("ingest-receipts.jsonl"))).values())
        accepted = sum(
            1 for r in receipts if str(r.get("status")) in {"accepted", "indexed", "verified"}
        )
        submitted = len(receipts) or 1

        vector = health_vector(
            result,
            run_status=str(session.state.get("run_status") or "complete"),
            collection_sufficient=bool(
                (collection.get("decision") or {}).get("outcome") == "sufficient"
            ),
            receipt_rate=accepted / submitted,
            cost_usd=float(budget.get("committed_usd") or 0.0),
        )

        write_summary(
            paths,
            result,
            manifest=session.manifest,
            vector=vector,
            collection=collection,
            limitations=_provider_limitations(session.config),
        )
        write_arm_comparison(paths, result)
        write_collection_report(paths, collection)
        write_query_detail(
            paths, result.per_query, questions=questions, lineage=LineageIndex(paths)
        )
        write_regressions(paths, result, questions=questions)
        write_errors_report(
            paths,
            errors,
            mcp_reliability=operational_metrics.mcp_reliability(mcp_calls),
            latency=operational_metrics.latency_by_stage(events),
        )
        candidates = build_candidates(
            result,
            errors=errors,
            receipts=receipts,
            duplicate_rate=(collection.get("audit") or {}).get("duplicate_rate"),
            minimum_frequency=session.config.refinement.minimum_frequency,
        )
        write_refinements(paths, candidates, submitted=session.config.refinement.submit_to_pheasant)
        write_json(paths.reports / "health-vector.json", vector)
        write_checksums(paths)

        print(f"reports written to {paths.reports}")
        for key, value in vector.items():
            print(f"  {key}: {value}")
        return EXIT_OK
    finally:
        session.close()


def _provider_limitations(config: LabConfig) -> list[str]:
    limitations: list[str] = []
    providers = {spec.provider for spec in config.models.values()}
    if "replay" in providers:
        limitations.append(
            "The deterministic `replay` provider was used for at least one role. It has no prior "
            "knowledge, so the prior-only control `C0` abstains everywhere and `P0 − C0` is a "
            "**floor** on the corpus's contribution rather than an estimate of it. Its answering "
            "is extractive, so answer quality here measures retrieval."
        )
    if config.pheasant.transport == "mock":
        limitations.append(
            "The run answered against the in-process mock region: BM25 only, no vector or graph "
            "arm. It measures the plumbing, not Pheasant."
        )
    return limitations


def _load_result(paths: RunPaths) -> Any:
    from .evaluation.classification import Classification, Status, SubgroupResult
    from .evaluation.engine import EvaluationResult
    from .evaluation.gates import GateOutcome, GateSet, GateSetResult
    from .evaluation.statistics import Interval, TestResult

    result = EvaluationResult()
    result.per_query = list(read_jsonl(paths.metrics / "per-query.jsonl"))
    if (paths.metrics / "aggregates.json").is_file():
        result.aggregates = read_json(paths.metrics / "aggregates.json")
    if (paths.metrics / "classification.json").is_file():
        payload = read_json(paths.metrics / "classification.json")
        result.limitations = list(payload.get("limitations") or [])
        result.non_inferiority = dict(payload.get("non_inferiority") or {})
        for row in payload.get("comparisons") or []:
            interval = row.get("interval")
            result.classifications.append(
                Classification(
                    metric=str(row["metric"]),
                    baseline_arm=str(row["baseline_arm"]),
                    treatment_arm=str(row["treatment_arm"]),
                    status=Status(str(row["status"])),
                    n=int(row["n"]),
                    coverage=row.get("coverage"),
                    baseline_mean=row.get("baseline_mean"),
                    treatment_mean=row.get("treatment_mean"),
                    absolute_delta=row.get("absolute_delta"),
                    relative_delta=row.get("relative_delta"),
                    threshold=float(row.get("practical_threshold") or 0.0),
                    higher_is_better=bool(row.get("higher_is_better", True)),
                    interval=Interval(
                        lower=float(interval["lower"]),
                        upper=float(interval["upper"]),
                        level=float(interval["level"]),
                        method=str(interval["method"]),
                        resamples=int(interval["resamples"]),
                    )
                    if interval
                    else None,
                    tests=[
                        TestResult(
                            test=str(test.get("test")),
                            statistic=test.get("statistic"),
                            p_value=test.get("p_value"),
                            n=int(test.get("n") or 0),
                            detail={
                                k: v
                                for k, v in test.items()
                                if k not in {"test", "statistic", "p_value", "n"}
                            },
                        )
                        for test in row.get("tests") or []
                    ],
                    effect=dict(row.get("effect") or {}),
                    subgroups=[
                        SubgroupResult(
                            key=str(s["key"]),
                            value=str(s["value"]),
                            n=int(s["n"]),
                            delta=float(s["delta"]),
                            status=str(s["status"]),
                        )
                        for s in row.get("subgroups") or []
                    ],
                    reasons=list(row.get("reasons") or []),
                    cohort=row.get("cohort"),
                )
            )
    if (paths.metrics / "gates.json").is_file():
        payload = read_json(paths.metrics / "gates.json")
        gates = GateSetResult()
        for name, body in (payload.get("gate_sets") or {}).items():
            outcomes = [
                GateOutcome(
                    gate=str(row["gate"]),
                    evaluated=bool(row["evaluated"]),
                    passed=row.get("passed"),
                    observed=row.get("observed"),
                    threshold=row.get("threshold"),
                    detail=str(row.get("detail") or ""),
                    evidence_refs=list(row.get("evidence_refs") or []),
                    skip_reason=row.get("skip_reason"),
                )
                for row in body.get("gates") or []
            ]
            if outcomes:
                gates.sets[name] = GateSet(name, outcomes)
        result.gates = gates
    return result


def _rehydrated(session: Session, topic_id: str, rehydrate: Any) -> Any:
    paths = session.paths
    return rehydrate(
        session.run_id,
        topic_id,
        read_jsonl(paths.raw_file("sources.jsonl")),
        read_jsonl(paths.raw_file("claims.jsonl")),
        contradiction_rows=read_jsonl(paths.raw_file("contradictions.jsonl")),
        round_rows=(session.state.get("collection") or {}).get("rounds") or [],
    )


def cmd_demo(args: argparse.Namespace) -> int:
    """End to end, offline, against fixtures and the in-process mock region.

    This exists so the whole pipeline can be exercised without a network, a
    key or a running Pheasant - and so the tests drive the same code path a
    person does.
    """

    namespace = argparse.Namespace(
        config=args.config,
        set=list(getattr(args, "set", None) or []),
        env_file=None,
        project_root=getattr(args, "project_root", None),
        max_cost_usd=getattr(args, "max_cost_usd", None),
        dry_run=False,
        mock=True,
        offline=True,
        topic=getattr(args, "topic", None),
        force=True,
        json=False,
        run=None,
        fork=False,
        arms=getattr(args, "arms", None),
    )
    code = cmd_collect(namespace)
    run_id = _latest_run(namespace)
    namespace.run = run_id
    for command in (cmd_freeze_benchmark, cmd_evaluate, cmd_replay, cmd_report, cmd_verify):
        result = command(namespace)
        if result == EXIT_REFUSED:
            return result
        code = code or result
    print(f"\ndemo run: {run_id}")
    return code


def cmd_draft_topic(args: argparse.Namespace) -> int:
    """Draft a research topic from a stated intent. Writes nothing."""

    import yaml

    from .orchestration.drafter import draft_topic, drafting_role

    config = _resolve_config(args)
    seeds = [t for t in (args.seed_terms or "").split(",") if t.strip()]
    try:
        context = json.loads(args.context_json) if args.context_json else {}
        if not isinstance(context, dict):
            raise ValueError("--context-json must be a JSON object")
        spec = drafting_role(
            config,
            model=args.model,
            provider=args.provider,
            reasoning_effort=args.reasoning_effort,
        )
        draft = draft_topic(
            config,
            _model(config, "planner", spec=spec),
            intent=args.intent or "",
            seed_terms=seeds,
            max_cost_usd=args.max_cost_usd,
            context=context,
            spec=spec,
        )
    except (RuntimeError, ValueError) as exc:
        # A missing key, an unpriced model, an intent too thin to read: each
        # is an answer for the person, not a crash.
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    payload = draft.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        topic = {k: v for k, v in payload.items() if k not in {"notes", "drafted_by"}}
        topic["facets"] = [{k: f[k] for k in ("id", "label", "weight")} for f in topic["facets"]]
        print(yaml.safe_dump({"topics": [topic]}, sort_keys=False, allow_unicode=True))
        print(f"# {payload['notes']}")
        by = payload["drafted_by"]
        print(f"# drafted by {by['provider']}:{by['model']} for ${by['cost_usd']:.4f}")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """The whole pipeline, resuming any stage that crashes. See ``supervisor``."""

    from .supervisor import RunLocked, supervise

    config = _resolve_config(args)
    project_root = Path(getattr(args, "project_root", None) or Path.cwd()).resolve()
    output_root = Path(config.experiment.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    common = ["--config", str(args.config)]
    for item in getattr(args, "set", None) or []:
        common += ["--set", item]
    if getattr(args, "env_file", None):
        common += ["--env-file", str(args.env_file)]
    if getattr(args, "project_root", None):
        common += ["--project-root", str(args.project_root)]
    for flag in ("mock", "offline"):
        if getattr(args, flag, False):
            common.append(f"--{flag}")
    try:
        code, run_id = supervise(
            common,
            output_root=output_root,
            run_id=getattr(args, "run", None),
            topic=getattr(args, "topic", None),
            arms=getattr(args, "arms", None),
            max_cost_usd=getattr(args, "max_cost_usd", None),
            attempts=max(1, int(args.attempts)),
            cwd=project_root,
        )
    except RunLocked as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if run_id:
        print(f"\nrun: {run_id}")
    return code


def cmd_serve(args: argparse.Namespace) -> int:
    """The console: configure, launch and watch runs from a browser.

    Loopback by default. It starts runs - paid ones, with the right config -
    so binding it anywhere else is a decision to say out loud.
    """

    from .console.server import Unauthenticated, serve
    from .settings import load_dotenv

    project_root = Path(getattr(args, "project_root", None) or Path.cwd()).resolve()
    env_file = getattr(args, "env_file", ".env")
    config = load_config(args.config, project_root=project_root, env_file=env_file)
    output_root = Path(args.output_root_dir or config.experiment.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    # The key, like every secret here, comes from the environment or `.env`,
    # never from a flag: argv is visible to every process on the machine.
    token_env = getattr(args, "token_env", None) or "PHEASANT_LAB_CONSOLE_TOKEN"
    environment = {**(load_dotenv(project_root / env_file) if env_file else {}), **os.environ}
    try:
        server = serve(
            host=args.host,
            port=args.port,
            project_root=project_root,
            output_root=output_root,
            default_config=str(Path(args.config)),
            ui_dist=Path(args.ui_dist) if args.ui_dist else None,
            token=environment.get(token_env),
            allow_unauthenticated=bool(getattr(args, "allow_unauthenticated", False)),
            env_file=env_file,
            run_scope=_run_scope(args, environment),
        )
    except Unauthenticated as exc:
        print(f"refused: {exc}")
        return EXIT_REFUSED
    console = server.console  # type: ignore[attr-defined]
    print(f"pheasant-lab console on http://{args.host}:{server.server_address[1]}")
    if console.token is not None:
        print(f"access key required ({token_env}); the browser asks for it once per tab")
    print(
        f"runs: {output_root}"
        + (f" (only runs made in: {console.run_scope})" if console.run_scope else "")
    )
    print(
        f"MCP: http://{args.host}:{server.server_address[1]}/mcp (or `pheasant-lab mcp` on stdio)"
    )
    if console.ui_dist is None:
        print("UI not built: run `make ui`. The API is live at /api.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("console stopped; launched runs keep running")
    finally:
        server.server_close()
    return EXIT_OK


def _run_scope(args: argparse.Namespace, environment: dict[str, str]) -> str | None:
    scope = getattr(args, "run_scope", None)
    if scope is None:
        scope = environment.get("PHEASANT_LAB_RUN_SCOPE")
    return (scope or "").strip() or None


def cmd_mcp(args: argparse.Namespace) -> int:
    """The console's tools over MCP stdio: configure, launch and manage runs from an agent."""

    from .console.mcp import serve_stdio
    from .console.server import Console, default_ui_dist
    from .settings import load_dotenv

    project_root = Path(getattr(args, "project_root", None) or Path.cwd()).resolve()
    env_file = getattr(args, "env_file", ".env")
    config = load_config(args.config, project_root=project_root, env_file=env_file)
    output_root = Path(args.output_root_dir or config.experiment.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    environment = {**(load_dotenv(project_root / env_file) if env_file else {}), **os.environ}
    console = Console(
        project_root=project_root,
        output_root=output_root,
        default_config=str(Path(args.config)),
        ui_dist=default_ui_dist(project_root),
        run_scope=_run_scope(args, environment),
    )
    console.env_file = env_file
    serve_stdio(console)
    return EXIT_OK


def cmd_settings(args: argparse.Namespace) -> int:
    """Explain every setting: meaning, valid values, current value, recommendation."""

    from .catalog import catalog, describe

    config = _resolve_config(args)
    if args.key:
        row = describe(args.key, config)
        print(json.dumps(row, indent=2, default=str))
        return EXIT_OK
    result = catalog(config, include_advanced=args.advanced)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return EXIT_OK
    titles = {s["key"]: s["title"] for s in result["sections"]}
    current = None
    for row in result["fields"]:
        if args.section and row["section"] != args.section:
            continue
        if row["section"] != current:
            current = row["section"]
            print(f"\n## {titles.get(current, current)}")
        hint = f"  (recommended: {row['recommended']})" if row.get("recommended") else ""
        print(f"{row['key']} = {json.dumps(row.get('current'), default=str)}{hint}")
        print(f"    {row['help']}")
        if row.get("values"):
            print(f"    values: {row['values']}")
    return EXIT_OK


def _latest_run(args: argparse.Namespace) -> str:
    config = _resolve_config(args)
    root = Path(config.experiment.output_root)
    runs = sorted(
        (path for path in root.glob("run-*") if (path / "run-manifest.json").is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    if not runs:
        raise ConfigError(f"no run directory under {root}")
    return runs[-1].name


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pheasant-lab",
        description="Pheasant Scientific Swarm Evaluation Lab",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser, *, mutating: bool = True) -> None:
        sub.add_argument("--config", default="configs/experiment.example.yaml")
        sub.add_argument(
            "--set", action="append", metavar="a.b=value", help="override a config field"
        )
        sub.add_argument("--env-file", default=".env")
        sub.add_argument("--project-root", default=None)
        if mutating:
            sub.add_argument("--resume", dest="run", default=None, help="continue an existing run")
            sub.add_argument("--dry-run", action="store_true")
            sub.add_argument("--max-cost-usd", type=float, default=None)
            sub.add_argument("--fork", action="store_true", help="allow a changed config on resume")
            sub.add_argument("--mock", action="store_true", help="use the in-process mock region")
            sub.add_argument(
                "--offline", action="store_true", help="use fixture literature providers"
            )

    doctor = subparsers.add_parser("doctor", help="verify laptop, models, providers and Pheasant")
    common(doctor)
    doctor.set_defaults(func=cmd_doctor)

    plan = subparsers.add_parser("plan", help="project cost without model or ingest calls")
    common(plan, mutating=False)
    plan.add_argument("--topic", default=None)
    plan.add_argument("--max-cost-usd", type=float, default=None)
    plan.add_argument("--json", action="store_true")
    plan.set_defaults(func=cmd_plan)

    collect = subparsers.add_parser("collect", help="run collection and Pheasant ingestion")
    common(collect)
    collect.add_argument("--topic", default=None)
    collect.add_argument("--force", action="store_true")
    collect.set_defaults(func=cmd_collect)

    audit = subparsers.add_parser("audit", help="audit saturation and coverage")
    common(audit)
    audit.add_argument("--run", dest="run", required=True)
    audit.add_argument("--json", action="store_true")
    audit.set_defaults(func=cmd_audit)

    freeze_cmd = subparsers.add_parser("freeze-benchmark", help="build and freeze the question set")
    common(freeze_cmd)
    freeze_cmd.add_argument("--run", dest="run", required=True)
    freeze_cmd.set_defaults(func=cmd_freeze_benchmark)

    evaluate = subparsers.add_parser("evaluate", help="run the isolated comparison arms")
    common(evaluate)
    evaluate.add_argument("--run", dest="run", required=True)
    evaluate.add_argument("--arms", default=None, help="comma-separated subset, e.g. S0,C0,P0")
    evaluate.set_defaults(func=cmd_evaluate)

    replay = subparsers.add_parser("replay", help="rebuild projections from raw events")
    common(replay)
    replay.add_argument("--run", dest="run", required=True)
    replay.set_defaults(func=cmd_replay)

    verify = subparsers.add_parser("verify", help="verify hashes, lineage, leakage and gates")
    common(verify)
    verify.add_argument("--run", dest="run", required=True)
    verify.set_defaults(func=cmd_verify)

    report = subparsers.add_parser("report", help="render Markdown, CSV and JSON reports")
    common(report)
    report.add_argument("--run", dest="run", required=True)
    report.set_defaults(func=cmd_report)

    demo = subparsers.add_parser(
        "demo", help="end-to-end offline run against fixtures and a mock region"
    )
    common(demo)
    demo.add_argument("--topic", default=None)
    demo.add_argument("--arms", default=None)
    demo.add_argument("--output-root", default=None)
    demo.set_defaults(func=cmd_demo)

    draft = subparsers.add_parser(
        "draft-topic", help="draft a research topic (title, seed terms, facets) from an intent"
    )
    common(draft, mutating=False)
    draft.add_argument("--intent", default="", help="what you want to find out, in your words")
    draft.add_argument("--seed-terms", default=None, help="comma-separated terms you already have")
    draft.add_argument(
        "--model",
        default=None,
        help="draft with this model instead of the planner's, e.g. gpt-6.1-sol",
    )
    draft.add_argument("--provider", default=None, help="openai, anthropic or replay (inferred)")
    draft.add_argument("--reasoning-effort", default=None, help="minimal, low, medium or high")
    draft.add_argument(
        "--context-json",
        default=None,
        help="the form so far as JSON: title, details, facets, date_range, preferred_types",
    )
    draft.add_argument(
        "--max-cost-usd", type=float, default=0.25, help="cap for this one model call"
    )
    draft.add_argument("--json", action="store_true")
    draft.set_defaults(func=cmd_draft_topic)

    run_cmd = subparsers.add_parser(
        "run",
        help="collect → freeze → evaluate → replay → report → verify, resuming crashed stages",
    )
    common(run_cmd)
    run_cmd.add_argument("--topic", default=None)
    run_cmd.add_argument("--arms", default=None)
    run_cmd.add_argument(
        "--attempts", type=int, default=3, help="attempts per stage before stopping resumable"
    )
    run_cmd.add_argument("--output-root", default=None)
    run_cmd.set_defaults(func=cmd_run)

    serve_cmd = subparsers.add_parser(
        "serve", help="the console: configure, launch and watch runs in a browser"
    )
    serve_cmd.add_argument("--config", default="configs/demo.yaml")
    serve_cmd.add_argument("--env-file", default=".env")
    serve_cmd.add_argument("--project-root", default=None)
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8770)
    serve_cmd.add_argument("--runs", dest="output_root_dir", default=None)
    serve_cmd.add_argument("--ui-dist", default=None, help="a built UI (default: ui/dist)")
    serve_cmd.add_argument(
        "--token-env",
        default="PHEASANT_LAB_CONSOLE_TOKEN",
        help="environment variable (or .env key) holding the console's access key; "
        "when it is set, every /api call needs it as a bearer token",
    )
    serve_cmd.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="serve beyond loopback with no access key (only behind an authenticating proxy)",
    )
    serve_cmd.add_argument(
        "--run-scope",
        default=None,
        help="list only runs made in this deployment, e.g. docker "
        "(default: $PHEASANT_LAB_RUN_SCOPE, else every run)",
    )
    serve_cmd.set_defaults(func=cmd_serve)

    mcp_cmd = subparsers.add_parser(
        "mcp", help="the console's tools over MCP stdio, for an agent to configure and run the lab"
    )
    mcp_cmd.add_argument("--config", default="configs/demo.yaml")
    mcp_cmd.add_argument("--env-file", default=".env")
    mcp_cmd.add_argument("--project-root", default=None)
    mcp_cmd.add_argument("--runs", dest="output_root_dir", default=None)
    mcp_cmd.add_argument("--run-scope", default=None)
    mcp_cmd.set_defaults(func=cmd_mcp)

    settings_cmd = subparsers.add_parser(
        "settings", help="explain every setting: meaning, valid values, recommendation"
    )
    common(settings_cmd, mutating=False)
    settings_cmd.add_argument("--section", default=None)
    settings_cmd.add_argument("--key", default=None)
    settings_cmd.add_argument("--advanced", action="store_true", help="include advanced fields")
    settings_cmd.add_argument("--json", action="store_true")
    settings_cmd.set_defaults(func=cmd_settings)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if getattr(args, "output_root", None):
        args.set = [
            *(getattr(args, "set", None) or []),
            f"experiment.output_root={args.output_root}",
        ]
    started = time.monotonic()
    try:
        code = int(args.func(args))
    except (ConfigError, MissingCapability) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except BudgetExceeded as exc:
        print(f"budget: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130
    except Exception:
        # Without this an unhandled exception exits 1, which this CLI already
        # uses for "completed, and a decision failed" - so a supervisor could
        # not tell a crash it should resume from a result it must not.
        import traceback

        traceback.print_exc()
        print("crashed: the run is resumable with --resume/--run", file=sys.stderr)
        return EXIT_CRASHED
    LOG.debug("%s finished in %.1fs", args.command, time.monotonic() - started)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
