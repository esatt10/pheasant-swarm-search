import type { LabEvent } from "../types";

/** One line a person can read, per event type. Unknown types show their payload keys. */
export function summarize(event: LabEvent): string {
  const p = event.payload as Record<string, any>;
  switch (event.event_type) {
    case "collection.search":
      return event.status === "started"
        ? `${p.provider ?? ""} · “${String(p.query ?? "").slice(0, 90)}”`
        : `${p.results ?? 0} results`;
    case "collection.discovered":
      return `${p.title ?? p.source_id} · ${p.stable_identifier ?? ""}`;
    case "collection.acquired":
      return `${p.source_id} · ${p.mode ?? ""}${p.licence ? ` · ${p.licence}` : ""}`;
    case "collection.round":
      return `round ${p.round} closed · ${p.acquired ?? 0} acquired · decision ${p.decision ?? "—"}`;
    case "collection.audit":
      return `coverage ${p.coverage_fraction} · duplicates ${p.duplicates ?? 0}`;
    case "collection.plan":
      return `${(p.subtopics ?? []).length} subtopics planned`;
    case "cost.model_call":
      return `${p.role ?? ""} · ${p.input_tokens ?? 0} in / ${p.output_tokens ?? 0} out · reserved $${Number(p.reserved_usd ?? 0).toFixed(4)} → $${Number(p.actual_usd ?? 0).toFixed(4)}`;
    case "mcp.tool.call":
      return `${p.tool}${p.refusal_code ? ` · refused ${p.refusal_code}` : ""} · ${Number(p.duration_ms ?? 0).toFixed(0)} ms${p.attempt > 1 ? ` · attempt ${p.attempt}` : ""}`;
    case "mcp.session.initialized":
      return `${p.server_name} ${p.server_version} · protocol ${p.protocol_version}`;
    case "ingest.sync":
      return p.disposition === "queued"
        ? `queued · task ${p.region?.task_id ?? ""} · waiting for an indexer to claim it`
        : `sync ${p.disposition}`;
    case "ingest.barrier": {
      const queue = (p.queue ?? []) as { state: string; claimed_by?: string; waiting_seconds?: number }[];
      const head = queue[0];
      return `${p.disposition} · still_accepted ${p.still_accepted ?? 0} · ${p.waited_seconds}s${
        head ? ` · ${head.state.replace("_", " ")}${head.claimed_by ? ` by ${head.claimed_by}` : ""}` : ""
      }`;
    }
    case "arm.answered":
      return `${p.question_id} · ${p.abstained ? "abstained" : `${p.claims} claims`} · ${p.retrieved ?? 0} retrieved · ${Number(p.latency_ms ?? 0).toFixed(0)} ms`;
    case "benchmark.built":
      return `${Object.values(p.cohorts ?? {}).reduce((a: number, b: any) => a + Number(b), 0)} questions frozen`;
    case "memory.seeded":
      return p.proven_questions === undefined
        ? `${p.records} records from ${p.learned_questions} learned questions`
        : `${p.records} records · ${p.proven_questions}/${p.learned_questions} learned answers proven by ≥${p.min_evidence} passage(s)`;
    case "memory.indexed":
      return `${(p.tasks ?? []).length} memory sync(s) ${String(p.outcome ?? "").replace("_", " ")} after ${p.waited_seconds}s · ${p.polls} polls`;
    default:
      return Object.keys(p).slice(0, 5).join(", ");
  }
}

export function tone(event: LabEvent): "warn" | "danger" | "ok" | "" {
  const p = event.payload as Record<string, any>;
  if (event.status === "failed") return "danger";
  if (event.event_type === "ingest.sync" && p.disposition === "queued") return "warn";
  if (event.event_type === "ingest.barrier") {
    if (p.disposition === "crossed") return "ok";
    if (p.disposition === "timed_out") return "danger";
    return "warn";
  }
  if (p.refusal_code) return "warn";
  return "";
}

export const FILTERS: { key: string; label: string; test: (e: LabEvent) => boolean }[] = [
  { key: "all", label: "all", test: () => true },
  { key: "collection", label: "collection.*", test: (e) => e.event_type.startsWith("collection.") },
  { key: "pheasant", label: "pheasant", test: (e) => e.event_type.startsWith("mcp.") || e.event_type.startsWith("ingest.") },
  { key: "ingest", label: "ingest", test: (e) => e.event_type.startsWith("ingest.") },
  { key: "arms", label: "arms", test: (e) => e.event_type.startsWith("arm.") },
  { key: "cost", label: "cost.*", test: (e) => e.event_type.startsWith("cost.") },
  { key: "warnings", label: "warnings", test: (e) => tone(e) === "warn" || tone(e) === "danger" },
];
