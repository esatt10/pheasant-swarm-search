"""Every setting a run can be given, explained.

The console's form, its MCP tools and ``pheasant-lab settings`` all read this
one catalog, so a field is described once: what it means, what may be entered,
its range, and - for the agent roles - which model and reasoning level we
recommend. Each entry is keyed by the ``--set`` path that changes it, which is
the only way any surface changes a setting (the argv stays the whole story).

Types and defaults are *derived* from the configuration models, never typed
here: a hand-kept list of defaults is a list of what somebody believed when
they last edited it. What is written here is only what a model cannot know -
the meaning. ``tests/unit/test_catalog.py`` fails when a configurable field
has no entry, or an entry names a field that no longer exists.
"""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass, field
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel

from .profiles import PROFILES, SCHOLARLY_TYPES, WEB_TYPES
from .settings import (
    ALL_ARMS,
    ExperimentFile,
    LabConfig,
    LoggingFile,
    MetricsFile,
    ModelsFile,
    PheasantFile,
    ProofPolicyFile,
    RoleModel,
)

#: The agent roles ``models.yaml`` configures, in the order a run meets them.
ROLES: tuple[str, ...] = (
    "orchestrator",
    "planner",
    "researcher",
    "auditor",
    "benchmark_builder",
    "specialist",
    "control",
    "test_agent",
)

PROVIDERS: tuple[str, ...] = ("replay", "openai", "anthropic")
#: Every reasoning level some OpenAI model accepts. Which ones a given model
#: takes is model-dependent (``MODEL_REASONING``); the GPT-6 family takes none
#: of ``minimal``, and gpt-6.1-sol refuses ``none`` too.
REASONING_EFFORTS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
#: What the models we recommend accept, checked 2026-10-07 (OpenAI's model
#: pages as relayed by the listings cited in configs/pricing.example.yaml). A
#: model not named here is not checked: the provider decides.
MODEL_REASONING: dict[str, tuple[str, ...]] = {
    "gpt-6.1-sol": ("low", "medium", "high", "xhigh", "max"),
    "gpt-6-sol": ("none", "low", "medium", "high", "xhigh", "max"),
    "gpt-6-luna": ("none", "low", "medium", "high", "xhigh", "max"),
}


def unsupported_effort(model: str, effort: str | None) -> str | None:
    """Why ``effort`` would be refused by ``model``, or ``None`` if it is fine or unknown."""

    accepted = MODEL_REASONING.get(model)
    if effort is None or accepted is None or effort in accepted:
        return None
    return f"{model} does not accept reasoning effort {effort!r}; it takes {', '.join(accepted)}"


#: Offered as suggestions, never enforced: a model id is the provider's to
#: define, and a list here would refuse next month's model.
MODEL_SUGGESTIONS: tuple[str, ...] = (
    "gpt-6.1-sol",
    "gpt-6-sol",
    "gpt-6-luna",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
)
#: The two models the topic drafter offers on one click.
DRAFT_MODELS: tuple[dict[str, str], ...] = (
    {
        "model": "gpt-6.1-sol",
        "provider": "openai",
        "reasoning_effort": "high",
        "label": "GPT-6.1 Sol",
        "note": "broader vocabulary and sharper facets; slower and dearer",
    },
    {
        "model": "gpt-6-luna",
        "provider": "openai",
        "reasoning_effort": "medium",
        "label": "GPT-6 Luna",
        "note": "fast and cheap; a good first draft for a field you know",
    },
)


@dataclass(frozen=True)
class RoleAdvice:
    """What a role does, and the model we recommend for it."""

    does: str
    model: str
    reasoning_effort: str
    why: str
    calls: str


ROLE_ADVICE: dict[str, RoleAdvice] = {
    "orchestrator": RoleAdvice(
        does="Runs the rounds: reads the audit, decides which gaps to send researchers after "
        "and when collection has saturated.",
        model="gpt-6.1-sol",
        reasoning_effort="high",
        why="Few calls per run, and each decides where the whole budget goes; a stop decision "
        "made badly wastes every later round.",
        calls="one per round (max_depth)",
    ),
    "planner": RoleAdvice(
        does="Turns the topic (intent, seed terms, facets) into subtopics and search vocabulary: "
        "synonyms, older names, identifiers.",
        model="gpt-6.1-sol",
        reasoning_effort="high",
        why="Vocabulary expansion is where a niche field is won or lost, and it is called a "
        "handful of times per run.",
        calls="one per depth level",
    ),
    "researcher": RoleAdvice(
        does="Each research branch: issues provider searches, admits sources and extracts "
        "claims with provenance.",
        model="gpt-6-luna",
        reasoning_effort="medium",
        why="The volume role - agents x rounds calls, many concurrent - so cost and latency "
        "dominate; extraction is grounded in the passage in front of it.",
        calls="max_research_agents x max_depth x max_search_rounds_per_agent",
    ),
    "auditor": RoleAdvice(
        does="Audits coverage between rounds: facet gaps, duplicates, unresolved contradictions.",
        model="gpt-6-luna",
        reasoning_effort="medium",
        why="A structured checklist over the round's own records; it needs care, not breadth.",
        calls="one per round",
    ),
    "benchmark_builder": RoleAdvice(
        does="Writes the frozen question set from the evidence ledger, with expected evidence ids "
        "and matchers kept disjoint from the wording.",
        model="gpt-6.1-sol",
        reasoning_effort="high",
        why="Called once, and every metric downstream is measured against what it writes.",
        calls="one per run",
    ),
    "specialist": RoleAdvice(
        does="Arm S0: answers with the collected sources and the research trace - the attainable "
        "reference, not ground truth.",
        model="gpt-6.1-sol",
        reasoning_effort="high",
        why="Keep S0, C0 and the Pheasant arms on one model and one reasoning level: then "
        "P1 - S0 measures Pheasant against the sources, not one model against another.",
        calls="questions x repetitions",
    ),
    "control": RoleAdvice(
        does="Arm C0: answers from the model's prior alone, no tools - what the model already knew.",
        model="gpt-6.1-sol",
        reasoning_effort="high",
        why="Must match test_agent exactly, or P0 - C0 is a difference between two models.",
        calls="questions x repetitions",
    ),
    "test_agent": RoleAdvice(
        does="Arms P0, P1 and P2: a fresh agent with nothing but Pheasant search, memory and "
        "steering.",
        model="gpt-6.1-sol",
        reasoning_effort="high",
        why="The thing being measured. Match the specialist so the comparison isolates what "
        "Pheasant contributes; drop all three to gpt-6-luna together to halve evaluation cost.",
        calls="questions x arms x repetitions x search rounds",
    ),
}


@dataclass(frozen=True)
class Doc:
    """The part of a field no model can derive: what it means."""

    help: str
    values: str = ""
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    choices: tuple[Any, ...] | None = None
    suggestions: tuple[Any, ...] | None = None
    advanced: bool = False
    #: False for the few fields that say *where* a run happens rather than
    #: *what* it does; changing them leaves the config digest alone.
    digest: bool = True
    label: str | None = None


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    blurb: str


SECTIONS: tuple[Section, ...] = (
    Section("experiment", "Experiment", "Name, seed and the run's two budgets."),
    Section("budget", "Budget", "How the dollar budget is split and reserved before each call."),
    Section("search", "Pheasant search", "What every arm's search asks the region for."),
    Section("connection", "Pheasant connection", "Which region the run talks to, and how."),
    Section("collection", "Collection", "How wide and deep the research swarm goes."),
    Section("stopping", "Stopping", "When collection counts as saturated."),
    Section("benchmark", "Benchmark", "The frozen question set and its cohorts."),
    Section("arms", "Arms & evaluation", "Which arms answer, how often, and how pairs are formed."),
    Section("models", "Agents & models", "The model and reasoning level for each agent role."),
    Section("privacy", "Privacy", "What a run's trace keeps."),
    Section("refinement", "Refinement", "Candidates proposed from what the run learnt."),
    Section("logging", "Logging & tracing", "How a run records itself."),
    Section("metrics", "Metrics & statistics", "How numbers are computed and compared."),
    Section("proof", "Proof policy", "What counts as evidence, and how much of it is enough."),
    Section("files", "Config files", "The sibling files a run resolves."),
)

_FRACTION = "a fraction from 0 to 1"
_SOURCE_TYPES = tuple(dict.fromkeys((*SCHOLARLY_TYPES, "book_chapter", *WEB_TYPES)))
_PROVIDER_NAMES = ("openalex", "crossref", "arxiv", "pubmed", "brave", "tavily", "fixtures")

#: Keyed by the ``--set`` path. ``models.<role>.*`` applies to every role.
DOCS: dict[str, Doc] = {
    # -- experiment ---------------------------------------------------------
    "experiment.name": Doc(
        "A label for the experiment, recorded in every run's manifest and shown in the run list.",
        "Free text; letters, digits, hyphens read best. Changing it moves the config digest.",
    ),
    "experiment.seed": Doc(
        "Seeds every random choice: cohort split, arm order, bootstrap. Two runs with one seed and "
        "one config make the same choices.",
        "Any integer.",
    ),
    "experiment.cost_budget_usd": Doc(
        "The hard dollar ceiling for the whole run. Every model call reserves its worst case "
        "against it first, so a run cannot overshoot; collection stops early to leave the "
        "evaluation reserve.",
        "A positive amount in USD. 10 is a first live run; the offline demo costs nothing.",
        minimum=0.01,
        maximum=10000,
        unit="USD",
        label="cost budget",
    ),
    "experiment.runtime_budget_minutes": Doc(
        "Wall-clock ceiling. A stage that would start past it is skipped and the run says so.",
        "Whole minutes; 30 for a demo, 120 for a live run.",
        minimum=1,
        maximum=10080,
        unit="min",
    ),
    "experiment.topics_file": Doc(
        "The research topics this run can choose from. Topics added in the console are written "
        "to configs/topics.local.yaml and selected here.",
        "A path relative to the project root.",
        digest=False,
        advanced=True,
    ),
    "experiment.output_root": Doc(
        "Where run directories are written. In the Docker image this is the runs volume.",
        "A path; relative paths are under the project root.",
        digest=False,
        advanced=True,
    ),
    "experiment.models_file": Doc(
        "models.yaml: each role's model, the pricing file and the budget split.",
        "A path.",
        digest=False,
        advanced=True,
    ),
    "experiment.metrics_file": Doc(
        "metrics.yaml: metrics, statistics and gates.", "A path.", digest=False, advanced=True
    ),
    "experiment.proof_policy_file": Doc(
        "proof-policy.yaml: what counts as evidence.", "A path.", digest=False, advanced=True
    ),
    "experiment.pheasant_file": Doc(
        "pheasant-mcp.yaml: the region connection and capability map. Choosing a connection in "
        "the console sets the fields of this file instead.",
        "A path.",
        digest=False,
        advanced=True,
    ),
    "experiment.logging_file": Doc(
        "logging.yaml: run logging and tracing.", "A path.", digest=False, advanced=True
    ),
    # -- collection -----------------------------------------------------------
    "collection.profile": Doc(
        "Which evidence the swarm collects. Supplies defaults for providers, admitted and "
        "authoritative source types and the stopping minimums; any of those you set yourself win.",
        "scholarly (OpenAlex, Crossref, arXiv, PubMed; authoritative = peer-reviewed); web "
        "(Brave, Tavily; authoritative = a primary statement, filing or posting); balanced (both, "
        "capped per provider).",
        choices=PROFILES,
    ),
    "collection.max_depth": Doc(
        "Rounds of plan -> research -> audit. Each round re-plans against the gaps the last one "
        "left.",
        "1-5. 2 is usually enough; every extra round multiplies researcher calls.",
        minimum=1,
        maximum=5,
    ),
    "collection.max_research_agents": Doc(
        "Research branches per round, one per subtopic the planner proposes.",
        "1-24. More branches cover more facets in parallel and cost proportionally more.",
        minimum=1,
        maximum=24,
    ),
    "collection.max_concurrent_agents": Doc(
        "How many branches run at once. Raises speed, not cost; provider rate limits are the "
        "ceiling.",
        "1 to max_research_agents. 3 is safe for the scholarly APIs.",
        minimum=1,
        maximum=24,
    ),
    "collection.max_search_rounds_per_agent": Doc(
        "Provider searches one branch may issue before it reports back.",
        "1-20.",
        minimum=1,
        maximum=20,
    ),
    "collection.max_sources_per_subtopic": Doc(
        "Admission cap per subtopic. Past it, a branch stops admitting even relevant sources.",
        "1-200.",
        minimum=1,
        maximum=200,
    ),
    "collection.allowed_source_types": Doc(
        "Source types a branch may admit. Anything else is recorded as rejected.",
        "A list, e.g. [journal_article, review, preprint].",
        suggestions=_SOURCE_TYPES,
    ),
    "collection.authoritative_source_types": Doc(
        "Types that count toward stopping.minimum_review_or_primary_sources. At least one must "
        "also be allowed, or the minimum could never be met.",
        "A list drawn from the source types.",
        suggestions=_SOURCE_TYPES,
    ),
    "collection.require_stable_identifier": Doc(
        "Admit only sources with a DOI, arXiv id, PMID or similar, so a citation can be checked.",
        "true for scholarly work; the web profile turns it off.",
    ),
    "collection.permit_abstract_only": Doc(
        "Admit a source whose full text is not available, on its abstract alone.",
        "true / false.",
    ),
    "collection.download_full_text_only_when_licensed": Doc(
        "Fetch full text only when the licence allows it (open access).",
        "true / false. Leave true.",
    ),
    "collection.providers": Doc(
        "Search providers, tried left to right. `fixtures` is the offline literature the demo uses.",
        "A list. Web providers need BRAVE_SEARCH_API_KEY / TAVILY_API_KEY.",
        suggestions=_PROVIDER_NAMES,
    ),
    "collection.max_results_per_provider": Doc(
        "Cap on what one provider returns for one query, so the first provider cannot fill a "
        "subtopic alone.",
        "Empty for no cap, or 1-100. The balanced profile sets it.",
        minimum=1,
        maximum=100,
    ),
    "collection.provider_timeout_seconds": Doc(
        "How long one provider request may take before it is recorded as an error.",
        "1-120 seconds.",
        minimum=1,
        maximum=120,
        unit="s",
        advanced=True,
    ),
    "collection.provider_max_retries": Doc(
        "Retries of a failed provider request before the next provider is tried.",
        "0-10.",
        minimum=0,
        maximum=10,
        advanced=True,
    ),
    # -- stopping -------------------------------------------------------------
    "stopping.minimum_sources_per_subtopic": Doc(
        "A subtopic is not saturated below this many admitted sources.",
        "1-100.",
        minimum=1,
        maximum=100,
    ),
    "stopping.minimum_independent_source_families": Doc(
        "Distinct source families (by the topic's family key) a subtopic needs; two papers from "
        "one group count once.",
        "1-50.",
        minimum=1,
        maximum=50,
    ),
    "stopping.minimum_review_or_primary_sources": Doc(
        "Authoritative sources a subtopic needs (see collection.authoritative_source_types).",
        "0-50.",
        minimum=0,
        maximum=50,
    ),
    "stopping.maximum_duplicate_rate": Doc(
        "Above this share of duplicate admissions a round counts as saturated.",
        _FRACTION,
        minimum=0,
        maximum=1,
    ),
    "stopping.maximum_unresolved_critical_contradictions": Doc(
        "Collection keeps going while more critical contradictions than this stay unresolved.",
        "0 or more.",
        minimum=0,
        maximum=100,
    ),
    "stopping.marginal_unique_claim_window": Doc(
        "Rounds over which the share of new unique claims is measured.",
        "1-10.",
        minimum=1,
        maximum=10,
    ),
    "stopping.marginal_unique_claim_threshold": Doc(
        "Below this share of new unique claims across the window, collection is saturated.",
        _FRACTION,
        minimum=0,
        maximum=1,
    ),
    "stopping.minimum_ingest_receipt_rate": Doc(
        "Share of submitted documents Pheasant must acknowledge as indexed before evaluation may "
        "start. Below it the run refuses rather than measuring a partial corpus.",
        "A fraction; 0.98 tolerates a stray rejection.",
        minimum=0,
        maximum=1,
    ),
    "stopping.consecutive_saturated_rounds": Doc(
        "Saturated rounds in a row before collection stops.", "1-5.", minimum=1, maximum=5
    ),
    "stopping.evaluation_budget_reserve_fraction": Doc(
        "Share of the budget collection may never touch, kept for the arms and reports. Must not "
        "exceed budget.allocation.evaluation + reserve.",
        _FRACTION,
        minimum=0,
        maximum=0.9,
    ),
    # -- benchmark ------------------------------------------------------------
    "benchmark.freeze_before_evaluation": Doc(
        "Freeze (content-address) the question set before any arm answers. Leave on: an unfrozen "
        "benchmark can be changed by the results it scores.",
        "true / false.",
        advanced=True,
    ),
    "benchmark.questions_per_topic": Doc(
        "Questions in the frozen set. Must equal the sum of benchmark.composition.",
        "4-200. Fewer than 12 leaves most paired statistics insufficient_evidence.",
        minimum=4,
        maximum=200,
    ),
    "benchmark.composition": Doc(
        "How many questions of each type. The sum must equal questions_per_topic.",
        "A map of question type to count.",
        advanced=True,
    ),
    "benchmark.require_expected_evidence_ids": Doc(
        "Refuse a question whose expected evidence ids are missing; without them retrieval "
        "recall has no denominator.",
        "true / false. Leave true.",
        advanced=True,
    ),
    "benchmark.blind_arm_order": Doc(
        "Hide which arm produced an answer from anything that scores it.",
        "true / false.",
        advanced=True,
    ),
    "benchmark.cohorts.anchor_fraction": Doc(
        "Share of questions in the frozen anchor cohort - the trend line across runs.",
        _FRACTION,
        minimum=0,
        maximum=1,
    ),
    "benchmark.cohorts.learned_fraction": Doc(
        "Share whose first-pass evidence may create memory or tuning (P1/P2 learn from these).",
        _FRACTION,
        minimum=0,
        maximum=1,
    ),
    "benchmark.cohorts.temporal_holdout_fraction": Doc(
        "Share never used for learning - where 'rivals the specialist' must hold.",
        _FRACTION,
        minimum=0,
        maximum=1,
    ),
    "benchmark.cohorts.control_fraction": Doc(
        "Share where no steering rule can fire, to catch memory regressions.",
        _FRACTION,
        minimum=0,
        maximum=1,
    ),
    "benchmark.cohorts.invariant_fraction": Doc(
        "Share of synthetic invariant questions. The five fractions must sum to 1.",
        _FRACTION,
        minimum=0,
        maximum=1,
    ),
    # -- arms -----------------------------------------------------------------
    "arms": Doc(
        "Which isolated arms answer the benchmark. S0 specialist, C0 prior-only, P0 Pheasant "
        "corpus, P1 + memory, P2 tuned replay.",
        "A list, e.g. [S0, C0, P0, P1, P2]. P1 - P0 needs both.",
        suggestions=ALL_ARMS,
    ),
    "replay.mode": Doc(
        "current_state answers against the region as it is now; historical asks for the state "
        "at replay.as_of.",
        "current_state or historical.",
        choices=("current_state", "historical"),
        advanced=True,
    ),
    "replay.as_of": Doc(
        "The instant historical replay asks for (ISO 8601).",
        "Empty, or e.g. 2026-09-30T00:00:00Z.",
        advanced=True,
    ),
    "replay.repetitions": Doc(
        "How many times each arm answers each question. More repetitions tighten intervals and "
        "multiply evaluation cost.",
        "1-10. 3 for a live run, 1 for the demo.",
        minimum=1,
        maximum=10,
    ),
    "replay.fresh_session_per_question": Doc(
        "Start every answer in a fresh session, so nothing carries over between questions.",
        "true / false. Leave true.",
        advanced=True,
    ),
    "replay.randomize_arm_order": Doc(
        "Shuffle arm order per question so no arm always goes first against a warm cache.",
        "true / false.",
        advanced=True,
    ),
    "replay.record_pheasant_snapshot": Doc(
        "Seal a Pheasant snapshot before the arms run and pin P0 to it.",
        "true / false.",
        advanced=True,
    ),
    "replay.pairing_policy": Doc(
        "complete_pairs_only compares a question only when every arm in the pair answered it; "
        "available_pairs uses whatever pairs exist.",
        "complete_pairs_only (recommended) or available_pairs.",
        choices=("complete_pairs_only", "available_pairs"),
    ),
    "replay.max_search_rounds_per_answer": Doc(
        "Search rounds a Pheasant arm may make for one answer, each refining its query.",
        "1-10.",
        minimum=1,
        maximum=10,
        label="search rounds per answer",
    ),
    "replay.max_results_per_search": Doc(
        "Results each search asks the region for (pheasant's max_results).",
        "1-100. 10 matches retrieval@k; more passages cost more answer tokens.",
        minimum=1,
        maximum=100,
        label="results per search",
    ),
    "replay.graph_expansion": Doc(
        "Ask the region to attach each hit's graph neighbourhood (pheasant >= 0.13.1 `expand`). "
        "Recorded only; no arm reads it.",
        "Empty (off), true, a depth 1-3, or {depth, max_neighbors, edge_types, exclude_edge_types}.",
        label="graph expansion",
    ),
    "replay.search_mode": Doc(
        "Which retrieval arms answer: hybrid fuses text (BM25), vector and graph by reciprocal "
        "rank; the others isolate one arm to see what it alone contributes.",
        "Empty (= hybrid, the region's default), hybrid, text, vector or graph.",
        choices=(None, "hybrid", "text", "vector", "graph"),
        label="search mode",
    ),
    "replay.min_score": Doc(
        "A floor on the region's fused score; hits below it are dropped. Fused scores have no "
        "absolute scale, so prefer fewer results to a floor.",
        "Empty (no floor) or a positive number. Sent only when the pheasant file maps min_score.",
        minimum=0,
        label="minimum score",
        advanced=True,
    ),
    "replay.memory_min_evidence": Doc(
        "How many distinct retrieved passages must validate a learned answer before P1's memory "
        "records it as a retrieval proof. An answer with fewer writes no memory at all, and no "
        "steering rule is learned from it.",
        "Empty (= 1) or 1-10. Higher keeps only well-corroborated answers; P1 then has less memory "
        "to be measured with.",
        minimum=1,
        maximum=10,
        label="memory evidence (k)",
    ),
    # -- privacy --------------------------------------------------------------
    "privacy.store_prompt_text": Doc(
        "Keep prompt text in the trace (digests are always kept).", "true / false."
    ),
    "privacy.store_response_text": Doc(
        "Keep model responses and retrieved passages in the trace.", "true / false."
    ),
    "privacy.redact_secrets": Doc(
        "Replace every configured secret with a stable token before anything is written.",
        "true / false. Leave true.",
    ),
    "privacy.hash_principal_identifiers": Doc(
        "Hash principal ids before they are written.", "true / false."
    ),
    "privacy.allow_remote_trace_export": Doc(
        "Permit OTLP export of spans off this machine (see logging export).",
        "true / false.",
        advanced=True,
    ),
    "privacy.raw_prompt_retention_days": Doc(
        "Days prompt text is kept. Empty keeps it until deleted by hand.",
        "Empty or 1-3650.",
        minimum=1,
        maximum=3650,
        advanced=True,
    ),
    "privacy.retrieved_passage_retention_days": Doc(
        "Days retrieved passages are kept.",
        "Empty or 1-3650.",
        minimum=1,
        maximum=3650,
        advanced=True,
    ),
    "privacy.downloaded_content_retention_days": Doc(
        "Days downloaded full text is kept.",
        "Empty or 1-3650.",
        minimum=1,
        maximum=3650,
        advanced=True,
    ),
    # -- refinement -----------------------------------------------------------
    "refinement.enabled": Doc(
        "Propose refinement candidates (aliases, steering rules) in the report.", "true / false."
    ),
    "refinement.submit_to_pheasant": Doc(
        "Also submit them to a Pheasant namespace. Needs its own diagnostic knowledge base, never "
        "the one retrieval reads.",
        "true / false.",
        advanced=True,
    ),
    "refinement.diagnostic_knowledge_base": Doc(
        "The knowledge base refinement candidates go to.",
        "A knowledge base name, distinct from the lab's.",
        advanced=True,
    ),
    "refinement.minimum_frequency": Doc(
        "Times a pattern must recur before it becomes a candidate.",
        "1-20.",
        minimum=1,
        maximum=20,
        advanced=True,
    ),
    # -- models ---------------------------------------------------------------
    "models.<role>.provider": Doc(
        "The model provider adapter. replay is offline, deterministic and free; openai and "
        "anthropic call the hosted APIs with OPENAI_API_KEY / ANTHROPIC_API_KEY.",
        "replay, openai or anthropic.",
        choices=PROVIDERS,
    ),
    "models.<role>.model": Doc(
        "The model id sent to the provider. It must have a price in the pricing file, or doctor "
        "refuses the run: a model treated as free is a budget guard that does not exist.",
        "A provider model id, e.g. gpt-6.1-sol or gpt-6-luna; replay:<name> offline.",
        suggestions=MODEL_SUGGESTIONS,
    ),
    "models.<role>.reasoning_effort": Doc(
        "How long the model may think before answering; thinking tokens bill as output. OpenAI "
        "receives it as reasoning.effort; Anthropic maps medium and up to extended thinking. "
        "Reasoning models ignore temperature.",
        "Empty (the model's default, medium for GPT-6), or none, low, medium, high, xhigh, max. "
        "Model-dependent: gpt-6.1-sol refuses none, and no GPT-6 model takes minimal.",
        choices=(None, *REASONING_EFFORTS),
        label="reasoning level",
    ),
    "models.<role>.temperature": Doc(
        "Sampling temperature. 0 is the reproducible choice; not sent to reasoning models.",
        "0-2.",
        minimum=0,
        maximum=2,
        advanced=True,
    ),
    "models.<role>.max_output_tokens": Doc(
        "The most a call may write. The budget guard reserves this worst case before every call, "
        "so it bounds cost as well as length.",
        "256-64000.",
        minimum=256,
        maximum=64000,
        unit="tokens",
    ),
    "models.<role>.tool_call_limit": Doc(
        "Tool calls one invocation may make; each is reserved at budget.assumed_tool_call_output_tokens.",
        "0-200.",
        minimum=0,
        maximum=200,
        advanced=True,
    ),
    "models.<role>.criteria": Doc(
        "What the role's prompt is told to prioritise (structured output, no prior context, ...).",
        "A map; edit the models file for this.",
        advanced=True,
    ),
    "pricing.source": Doc(
        "The versioned price list (USD per million tokens). Prices edited in the console go to "
        "configs/pricing.local.yaml and are selected here.",
        "A path.",
        advanced=True,
    ),
    "pricing.fail_when_model_price_missing": Doc(
        "Refuse a run whose model has no price. Leave on.", "true / false.", advanced=True
    ),
    # -- budget ---------------------------------------------------------------
    "budget.allocation.planning": Doc(
        "Share of the budget for planning and orchestration. The five shares must sum to 1.",
        _FRACTION,
        minimum=0,
        maximum=1,
    ),
    "budget.allocation.collection": Doc(
        "Share for the research swarm.", _FRACTION, minimum=0, maximum=1
    ),
    "budget.allocation.benchmark": Doc(
        "Share for building the question set.", _FRACTION, minimum=0, maximum=1
    ),
    "budget.allocation.evaluation": Doc(
        "Share for the arms answering.", _FRACTION, minimum=0, maximum=1
    ),
    "budget.allocation.reserve": Doc(
        "Share held back for retries and reconciliation.", _FRACTION, minimum=0, maximum=1
    ),
    "budget.reserve_output_at_max_tokens": Doc(
        "Reserve a call's full max_output_tokens before making it. Turning this off reserves an "
        "estimate and lets a run overshoot on exactly the calls a budget exists to stop.",
        "true / false. Leave true.",
        advanced=True,
    ),
    "budget.assumed_tool_call_output_tokens": Doc(
        "Tokens reserved per allowed tool call.",
        "0-10000.",
        minimum=0,
        maximum=10000,
        unit="tokens",
        advanced=True,
    ),
    # -- pheasant connection --------------------------------------------------
    "transport": Doc(
        "How the lab reaches Pheasant: streamable_http (a running region's /mcp), stdio (a "
        "command the lab starts) or mock (in-process, offline; measures nothing about Pheasant).",
        "streamable_http, stdio or mock.",
        choices=("streamable_http", "stdio", "mock"),
    ),
    "url": Doc(
        "The region's MCP endpoint. In Docker Compose the bundled region is http://pheasant:8765/mcp.",
        "An http(s) URL ending in /mcp.",
        digest=True,
    ),
    "command": Doc(
        "The command a stdio transport starts, e.g. `pheasant mcp --transport stdio`.",
        "A command line.",
        advanced=True,
    ),
    "token_env": Doc(
        "The environment variable holding the region's API token (pheasant's "
        "security.api_auth). A connection saved in the console can hold the token itself.",
        "A variable name, e.g. PHEASANT_API_TOKEN.",
    ),
    "protocol_version": Doc(
        "The MCP protocol version the client offers.", "A date string.", advanced=True
    ),
    "timeout_seconds": Doc(
        "Per-call timeout for a tool call.",
        "1-600 seconds.",
        minimum=1,
        maximum=600,
        unit="s",
        advanced=True,
    ),
    "connect_timeout_seconds": Doc(
        "Connection timeout.", "1-120 seconds.", minimum=1, maximum=120, unit="s", advanced=True
    ),
    "max_retries": Doc(
        "Retries of a retryable tool failure.", "0-10.", minimum=0, maximum=10, advanced=True
    ),
    "retry_backoff_seconds": Doc(
        "First retry delay; doubles each attempt.",
        "0-60 seconds.",
        minimum=0,
        maximum=60,
        unit="s",
        advanced=True,
    ),
    "retry_backoff_max_seconds": Doc(
        "Longest retry delay.", "0-600 seconds.", minimum=0, maximum=600, unit="s", advanced=True
    ),
    "knowledge_base": Doc(
        "The pheasant knowledge base (its `pheasant.name`) the lab writes to and searches. Use one "
        "that holds nothing else.",
        "A knowledge base name, e.g. pheasant-lab.",
    ),
    "source_name": Doc(
        "The source the lab registers for the documents it submits. The lab writes only to a "
        "source it created.",
        "A source name, e.g. swarm-lab-literature.",
    ),
    "capabilities": Doc(
        "Which pheasant tool answers each capability. Names are configured, never guessed; doctor "
        "checks each against tools/list.",
        "A map of capability to {tool, required}.",
        advanced=True,
    ),
    "argument_map": Doc(
        "How the lab's request fields are spelled for this server. An argument absent here is "
        "one the lab does not send.",
        "A map; `--set argument_map.search.expand=null` unmaps one.",
        advanced=True,
    ),
    "mock_claim_seconds": Doc(
        "Mock only: above 0 the mock behaves like a fleet - syncs are queued, and claimed after "
        "this many seconds.",
        "0-120 seconds.",
        minimum=0,
        maximum=120,
        unit="s",
    ),
    "discovery.verify_configured_tool_exists": Doc(
        "doctor refuses a configured tool missing from tools/list.", "true / false.", advanced=True
    ),
    "discovery.verify_input_schema": Doc(
        "doctor checks every mapped argument against the tool's schema.",
        "true / false.",
        advanced=True,
    ),
    "discovery.fail_on_ambiguous_capability": Doc(
        "Refuse when two tools could answer one capability.", "true / false.", advanced=True
    ),
    "discovery.allow_heuristic_name_matching": Doc(
        "Guess tool names by similarity. Off by design.",
        "true / false. Leave false.",
        advanced=True,
    ),
    "isolation.refuse_shared_knowledge_base": Doc(
        "Refuse a knowledge base that holds other sources.", "true / false.", advanced=True
    ),
    "isolation.require_dedicated_source": Doc(
        "Write only to a source the lab created.", "true / false. Leave true.", advanced=True
    ),
    "isolation.forbid_memory_writes_in_arms": Doc(
        "Arms that may never write memory.",
        "A list of arm ids; S0, C0 and P0 by design.",
        suggestions=ALL_ARMS,
        advanced=True,
    ),
    # -- logging --------------------------------------------------------------
    "logging.level": Doc(
        "Log level for every stage.",
        "DEBUG, INFO, WARNING or ERROR.",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    ),
    "logging.format": Doc(
        "text, or json (one object per line).", "text or json.", choices=("text", "json")
    ),
    "logging.file": Doc(
        "A log file; relative paths land inside the run (e.g. logs/lab.log).",
        "Empty for stderr only, or a path.",
    ),
    "tracing.raw_dir": Doc(
        "The append-only raw trace directory inside a run.", "A relative path.", advanced=True
    ),
    "tracing.flush_every_event": Doc(
        "Flush the trace after every event.", "true / false.", advanced=True
    ),
    "tracing.fsync_on_flush": Doc(
        "fsync each flush: slower, survives power loss.", "true / false.", advanced=True
    ),
    "tracing.record_payload_digest": Doc(
        "Record a digest of every payload.", "true / false.", advanced=True
    ),
    "tracing.verify_sequence_continuity": Doc(
        "verify checks the event sequence has no gaps.", "true / false.", advanced=True
    ),
    "tracing.spans.enabled": Doc("Write spans.jsonl.", "true / false.", advanced=True),
    "tracing.spans.file": Doc("The span file name.", "A file name.", advanced=True),
    "tracing.spans.adopt_otel_ids_when_available": Doc(
        "Use OpenTelemetry ids when an SDK is installed.", "true / false.", advanced=True
    ),
    "tracing.mcp_transcript.enabled": Doc(
        "Keep a redacted transcript of every MCP call.", "true / false."
    ),
    "tracing.mcp_transcript.file": Doc("The transcript file name.", "A file name.", advanced=True),
    "tracing.mcp_transcript.store_request_body": Doc("Keep MCP request bodies.", "true / false."),
    "tracing.mcp_transcript.store_response_body": Doc(
        "Keep what the region answered.", "true / false."
    ),
    "tracing.mcp_transcript.max_body_bytes": Doc(
        "Bodies are cut at this size.",
        "0-10000000 bytes.",
        minimum=0,
        maximum=10_000_000,
        unit="bytes",
    ),
    "tracing.mcp_transcript.forbidden_headers": Doc(
        "Headers never written.", "A list; authorization and cookie always.", advanced=True
    ),
    "projection.backend": Doc(
        "duckdb builds a queryable projection of the trace; none skips it.",
        "duckdb or none.",
        choices=("duckdb", "none"),
        advanced=True,
    ),
    "projection.file": Doc("The projection file inside a run.", "A relative path.", advanced=True),
    "projection.rebuild_on_replay": Doc(
        "Rebuild the projection on `replay`.", "true / false.", advanced=True
    ),
    "projection.verify_foreign_keys": Doc(
        "Check the projection's joins.", "true / false.", advanced=True
    ),
    "export.otlp_enabled": Doc(
        "Export spans over OTLP. Needs privacy.allow_remote_trace_export.",
        "true / false.",
        advanced=True,
    ),
    "export.otlp_endpoint": Doc("The OTLP endpoint.", "Empty or a URL.", advanced=True),
    "export.otlp_headers_env": Doc(
        "Variable holding OTLP headers.", "Empty or a variable name.", advanced=True
    ),
    # -- metrics --------------------------------------------------------------
    "metrics.primary": Doc(
        "Metrics the decision is made on.", "A list of metric names.", advanced=True
    ),
    "metrics.diagnostic": Doc(
        "Metrics reported but never decided on.", "A list of metric names.", advanced=True
    ),
    "metrics.k": Doc("k for retrieval@k metrics.", "1-100.", minimum=1, maximum=100),
    "metrics.relative_lift_epsilon": Doc(
        "Smallest relative lift worth reporting.", _FRACTION, minimum=0, maximum=1, advanced=True
    ),
    "classification.practical_threshold": Doc(
        "Per-metric difference that counts as practical.",
        "A map of metric to number.",
        advanced=True,
    ),
    "classification.lower_is_better": Doc(
        "Metrics where smaller is better.", "A list of metric names.", advanced=True
    ),
    "classification.minimum_paired_questions": Doc(
        "Below this many paired questions a comparison reports insufficient_evidence.",
        "1-200.",
        minimum=1,
        maximum=200,
    ),
    "classification.minimum_pairing_coverage": Doc(
        "Share of questions that must pair.", _FRACTION, minimum=0, maximum=1, advanced=True
    ),
    "classification.require_interval_excludes_zero": Doc(
        "A difference counts only if its interval excludes zero.", "true / false.", advanced=True
    ),
    "classification.protected_subgroups": Doc(
        "Question types that may not regress under the mean.", "A list.", advanced=True
    ),
    "statistics.enabled": Doc("Compute intervals and paired tests.", "true / false."),
    "statistics.bootstrap_resamples": Doc(
        "Bootstrap resamples per interval.",
        "100-100000.",
        minimum=100,
        maximum=100_000,
        advanced=True,
    ),
    "statistics.bootstrap_seed": Doc("Seed for the bootstrap.", "Any integer.", advanced=True),
    "statistics.confidence_level": Doc(
        "Interval confidence.", "0.5-0.999; 0.95 is conventional.", minimum=0.5, maximum=0.999
    ),
    "statistics.paired_tests": Doc(
        "Paired tests run.",
        "A list: mcnemar, wilcoxon.",
        suggestions=("mcnemar", "wilcoxon"),
        advanced=True,
    ),
    "statistics.multiple_comparison_correction": Doc(
        "Correction across comparisons.",
        "benjamini_hochberg or none.",
        choices=("benjamini_hochberg", "none"),
        advanced=True,
    ),
    "statistics.false_discovery_rate": Doc(
        "FDR for Benjamini-Hochberg.", _FRACTION, minimum=0, maximum=1, advanced=True
    ),
    "statistics.minimum_n_for_tests": Doc(
        "Below this n a test is not run.", "1-200.", minimum=1, maximum=200, advanced=True
    ),
    "specialist_noninferiority.margin": Doc(
        "Per-metric margin within which P1/P2 count as no worse than S0.",
        "A map of metric to number.",
        advanced=True,
    ),
    "specialist_noninferiority.require_holdout_cohort": Doc(
        "Rivalry must hold on the temporal holdout.", "true / false.", advanced=True
    ),
    "specialist_noninferiority.require_gates_pass": Doc(
        "Rivalry needs every gate to pass.", "true / false.", advanced=True
    ),
    "specialist_noninferiority.require_cost_within_budget": Doc(
        "Rivalry needs cost within budget.", "true / false.", advanced=True
    ),
    "specialist_noninferiority.require_no_protected_subgroup_regression": Doc(
        "Rivalry fails if a protected question type regressed.", "true / false.", advanced=True
    ),
    "gates": Doc(
        "Go/no-go gates evaluated before any aggregate.",
        "A map of gate to {enabled, thresholds}.",
        advanced=True,
    ),
    "answer_matching.normalize.casefold": Doc(
        "Compare answers case-insensitively.", "true / false.", advanced=True
    ),
    "answer_matching.normalize.strip_accents": Doc(
        "Ignore accents.", "true / false.", advanced=True
    ),
    "answer_matching.normalize.collapse_whitespace": Doc(
        "Collapse whitespace.", "true / false.", advanced=True
    ),
    "answer_matching.normalize.strip_punctuation": Doc(
        "Ignore punctuation.", "true / false.", advanced=True
    ),
    "answer_matching.normalize.number_words_to_digits": Doc(
        "Read 'four' as 4.", "true / false.", advanced=True
    ),
    "answer_matching.numeric_relative_tolerance": Doc(
        "Relative tolerance for numeric answers.", _FRACTION, minimum=0, maximum=1
    ),
    "answer_matching.require_citation_for_support": Doc(
        "A claim is supported only if it cites a passage.", "true / false.", advanced=True
    ),
    "answer_matching.citation_must_be_retrieved": Doc(
        "A citation must name something the session read.", "true / false.", advanced=True
    ),
    "answer_matching.support_overlap_threshold": Doc(
        "Token overlap with the cited passage above which a claim counts as supported (not entailment).",
        _FRACTION,
        minimum=0,
        maximum=1,
    ),
    # -- proof ----------------------------------------------------------------
    "proof.version": Doc("Proof-policy version.", "An integer.", advanced=True),
    "proof.publish_separately": Doc("Evidence classes published apart.", "A list.", advanced=True),
    "proof.event_types": Doc(
        "Evidence event types, their polarity and weight. Unknown carries weight 0.",
        "A map.",
        advanced=True,
    ),
    "proof.multipliers": Doc("Reported weight multipliers.", "A map.", advanced=True),
    "proof.minimum_evidence.per_question_proof_events": Doc(
        "Proof events a question needs.", "1-20.", minimum=1, maximum=20, advanced=True
    ),
    "proof.minimum_evidence.per_metric_questions": Doc(
        "Questions a metric needs, else insufficient_evidence.", "1-200.", minimum=1, maximum=200
    ),
    "proof.minimum_evidence.per_cohort_questions": Doc(
        "Questions a cohort needs.", "1-200.", minimum=1, maximum=200, advanced=True
    ),
    "proof.conflict.report_rate": Doc("Publish the conflict rate.", "true / false.", advanced=True),
    "proof.conflict.resolution": Doc(
        "How conflicting evidence resolves.",
        "none or majority.",
        choices=("none", "majority"),
        advanced=True,
    ),
    "judging.enabled": Doc(
        "A model judge, diagnostic only; never decides a metric.", "true / false.", advanced=True
    ),
    "judging.model": Doc(
        "The judge's model id.", "A model id.", suggestions=MODEL_SUGGESTIONS, advanced=True
    ),
    "judging.role": Doc(
        "Always diagnostic_only.", "diagnostic_only.", choices=("diagnostic_only",), advanced=True
    ),
    "judging.requires_deterministic_agreement_sample": Doc(
        "Share checked against the deterministic matcher.",
        _FRACTION,
        minimum=0,
        maximum=1,
        advanced=True,
    ),
}

#: Which section each head key belongs to in the form.
_SECTION_OF_HEAD = {
    "experiment": "experiment",
    "collection": "collection",
    "stopping": "stopping",
    "benchmark": "benchmark",
    "arms": "arms",
    "replay": "arms",
    "privacy": "privacy",
    "refinement": "refinement",
    "models": "models",
    "pricing": "budget",
    "budget": "budget",
    "logging": "logging",
    "tracing": "logging",
    "projection": "logging",
    "export": "logging",
    "metrics": "metrics",
    "classification": "metrics",
    "statistics": "metrics",
    "specialist_noninferiority": "metrics",
    "gates": "metrics",
    "answer_matching": "metrics",
    "proof": "proof",
    "judging": "proof",
}
_SEARCH_KEYS = {
    "replay.search_mode",
    "replay.max_results_per_search",
    "replay.max_search_rounds_per_answer",
    "replay.min_score",
    "replay.graph_expansion",
    "metrics.k",
}
_BUDGET_KEYS = {"experiment.cost_budget_usd", "experiment.runtime_budget_minutes"}
_FILE_KEYS = {
    "experiment.topics_file",
    "experiment.output_root",
    "experiment.models_file",
    "experiment.metrics_file",
    "experiment.proof_policy_file",
    "experiment.pheasant_file",
    "experiment.logging_file",
}
_PHEASANT_HEADS = set(PheasantFile.model_fields)


def section_of(key: str) -> str:
    if key in _SEARCH_KEYS:
        return "search"
    if key in _BUDGET_KEYS:
        return "budget"
    if key in _FILE_KEYS:
        return "files"
    head = key.split(".", 1)[0]
    if head in _PHEASANT_HEADS:
        return "connection"
    return _SECTION_OF_HEAD.get(head, "experiment")


# ---------------------------------------------------------------------------
# derivation from the models
# ---------------------------------------------------------------------------


@dataclass
class Leaf:
    key: str
    annotation: Any
    default: Any


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _leaves(model: type[BaseModel], prefix: str) -> list[Leaf]:
    rows: list[Leaf] = []
    for name, info in model.model_fields.items():
        key = f"{prefix}{info.alias or name}"
        if _is_model(info.annotation):
            rows.extend(_leaves(info.annotation, key + "."))
            continue
        default = info.default_factory() if info.default_factory else info.default  # type: ignore[call-arg]
        rows.append(Leaf(key, info.annotation, default))
    return rows


def configurable_leaves() -> list[Leaf]:
    """Every field a ``--set`` can reach, as the models declare it."""

    rows = _leaves(ExperimentFile, "")
    rows += _leaves(RoleModel, "models.<role>.")
    rows += [leaf for leaf in _leaves(ModelsFile, "") if leaf.key != "models"]
    rows += _leaves(PheasantFile, "")
    rows += _leaves(LoggingFile, "")
    rows += _leaves(MetricsFile, "")
    rows += _leaves(ProofPolicyFile, "")
    return rows


def _kind(annotation: Any) -> tuple[str, bool, tuple[Any, ...] | None]:
    """(kind, nullable, literal choices) for an annotation."""

    origin = get_origin(annotation)
    nullable = False
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        nullable = len(args) < len(get_args(annotation))
        if len(args) == 1:
            kind, _, choices = _kind(args[0])
            return kind, nullable, choices
        return "any", nullable, None
    if origin is Literal:
        return "enum", False, get_args(annotation)
    if origin is list:
        return "list", False, None
    if origin is dict:
        return "map", False, None
    if annotation is bool:
        return "bool", False, None
    if annotation is int:
        return "int", False, None
    if annotation is float:
        return "float", False, None
    if annotation is str:
        return "str", False, None
    return "any", False, None


def _label(key: str) -> str:
    return key.rsplit(".", 1)[-1].replace("_", " ")


def _entry(leaf: Leaf, key: str, doc: Doc) -> dict[str, Any]:
    kind, nullable, literal = _kind(leaf.annotation)
    choices = doc.choices if doc.choices is not None else literal
    if choices is not None and kind not in ("enum",):
        kind = "enum"
    return {
        "key": key,
        "section": section_of(leaf.key),
        "label": doc.label or _label(key),
        "kind": kind,
        "nullable": nullable or (choices is not None and None in choices),
        "default": leaf.default,
        "choices": list(choices) if choices is not None else None,
        "suggestions": list(doc.suggestions) if doc.suggestions else None,
        "minimum": doc.minimum,
        "maximum": doc.maximum,
        "unit": doc.unit,
        "help": doc.help,
        "values": doc.values,
        "advanced": doc.advanced,
        "moves_digest": doc.digest,
    }


def catalog(config: LabConfig | None = None, *, include_advanced: bool = True) -> dict[str, Any]:
    """The whole catalog: sections, fields (with current values), role advice."""

    roles = sorted(config.models) if config is not None else list(ROLES)
    roles = [r for r in ROLES if r in roles] + [r for r in roles if r not in ROLES]
    fields: list[dict[str, Any]] = []
    for leaf in configurable_leaves():
        doc = DOCS.get(leaf.key)
        if doc is None:
            continue
        keys = (
            [leaf.key.replace("<role>", role) for role in roles]
            if "<role>" in leaf.key
            else [leaf.key]
        )
        for key in keys:
            if doc.advanced and not include_advanced:
                continue
            row = _entry(leaf, key, doc)
            if "<role>" in leaf.key:
                role = key.split(".")[1]
                row["role"] = role
                advice = ROLE_ADVICE.get(role)
                field_name = key.rsplit(".", 1)[-1]
                if advice and field_name in ("model", "reasoning_effort"):
                    row["recommended"] = getattr(advice, field_name)
                if advice and field_name == "provider":
                    row["recommended"] = "openai"
            if config is not None:
                row["current"] = current_value(config, key)
                if "<role>" in leaf.key and key.endswith(".reasoning_effort"):
                    model = config.models[row["role"]].model
                    if model in MODEL_REASONING:
                        # Offer only what this role's model takes.
                        row["choices"] = [None, *MODEL_REASONING[model]]
            fields.append(row)
    # Role fields read role by role (all of the researcher's, then the
    # auditor's), where they were derived field by field.
    role_rows = [row for row in fields if "role" in row]
    if role_rows:
        first = fields.index(role_rows[0])
        order = {role: index for index, role in enumerate(roles)}
        role_rows.sort(key=lambda row: order.get(row["role"], len(order)))
        rest = [row for row in fields if "role" not in row]
        fields = rest[:first] + role_rows + rest[first:]
    return {
        "sections": [vars(section) for section in SECTIONS],
        "fields": fields,
        "roles": [
            {"role": role, **vars(ROLE_ADVICE[role])} if role in ROLE_ADVICE else {"role": role}
            for role in roles
        ],
        "draft_models": [dict(row) for row in DRAFT_MODELS],
        "reasoning_efforts": list(REASONING_EFFORTS),
        "providers": list(PROVIDERS),
    }


def describe(key: str, config: LabConfig | None = None) -> dict[str, Any]:
    """One field's entry, by its ``--set`` key."""

    for row in catalog(config)["fields"]:
        if row["key"] == key:
            return row
    raise KeyError(key)


def current_value(config: LabConfig, key: str) -> Any:
    """The resolved value a ``--set`` key would change."""

    from .settings import override_file

    owner = override_file(key)
    parts = key.split(".")
    node: Any
    if owner == "experiment":
        node = config
    elif owner == "models":
        head = parts[0]
        node = {"models": config.models, "pricing": config.pricing_ref, "budget": config.budget}
        node = node[head]
        parts = parts[1:]
    else:
        node = {
            "metrics": config.metrics,
            "proof": config.proof,
            "pheasant": config.pheasant,
            "logging": config.logging,
        }[owner]
    for part in parts:
        if node is None:
            return None
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, BaseModel):
            fields = type(node).model_fields
            name = next((n for n, f in fields.items() if (f.alias or n) == part), part)
            node = getattr(node, name, None)
        else:
            return None
    if isinstance(node, BaseModel):
        return node.model_dump(mode="json", by_alias=True)
    if isinstance(node, dict):
        return {
            k: v.model_dump(mode="json") if isinstance(v, BaseModel) else v for k, v in node.items()
        }
    return node


@dataclass
class Check:
    """A setting the catalog would advise against, and why."""

    key: str
    level: Literal["warn", "info"]
    message: str
    related: list[str] = field(default_factory=list)


def advise(config: LabConfig) -> list[dict[str, Any]]:
    """Advice the form shows beside the fields: things that resolve, and mislead."""

    notes: list[Check] = []
    for role, spec in sorted(config.models.items()):
        problem = unsupported_effort(spec.model, spec.reasoning_effort)
        if problem:
            notes.append(Check(f"models.{role}.reasoning_effort", "warn", f"{role}: {problem}."))
    answerers = [r for r in ("specialist", "control", "test_agent") if r in config.models]
    pairs = {(config.models[r].model, config.models[r].reasoning_effort) for r in answerers}
    # The offline replay provider names a model per role on purpose and has
    # no prior to compare; the advice is about hosted models.
    hosted_answerers = any(config.models[r].provider != "replay" for r in answerers)
    if hosted_answerers and len(pairs) > 1:
        notes.append(
            Check(
                "models.test_agent.model",
                "warn",
                "specialist, control and test_agent differ in model or reasoning level, so S0, C0 "
                "and the Pheasant arms are not compared on one model.",
                [f"models.{r}.model" for r in answerers],
            )
        )
    if config.collection.max_concurrent_agents > config.collection.max_research_agents:
        notes.append(
            Check(
                "collection.max_concurrent_agents",
                "info",
                "more concurrent slots than research agents; the extra slots are never used.",
            )
        )
    if config.benchmark.questions_per_topic < 12:
        notes.append(
            Check(
                "benchmark.questions_per_topic",
                "info",
                "below 12 questions most paired comparisons report insufficient_evidence.",
            )
        )
    hosted = [r for r, spec in config.models.items() if spec.provider != "replay"]
    if hosted and config.pheasant.transport == "mock":
        notes.append(
            Check(
                "transport",
                "warn",
                "hosted models against the mock region: you pay for a run that measures the mock.",
            )
        )
    return [vars(n) for n in notes]
