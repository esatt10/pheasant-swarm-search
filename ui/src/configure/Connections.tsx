import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { Connection, RegionProbe, Resolved, Settings } from "../types";

/**
 * Which pheasant-kb region the run talks to.
 *
 * A connection is a named profile - transport, MCP URL, knowledge base, the
 * lab's source and the variable its token is read from. Choosing one writes
 * those as ordinary overrides into the shared draft, so the argv still names
 * the region. A token typed here is stored by the console (0600) and handed
 * to launched runs; it is never shown again, only whether one is held.
 */

const EMPTY = {
  name: "",
  description: "",
  transport: "streamable_http" as Connection["transport"],
  url: "http://pheasant:8765/mcp",
  command: "",
  knowledge_base: "pheasant-lab",
  source_name: "swarm-lab-literature",
  token_env: "",
  mock_claim_seconds: "",
};

type FormState = typeof EMPTY;

function probeTone(probe: RegionProbe | undefined): [string, string] {
  if (!probe) return ["pill", "not probed"];
  if (probe.mock) return ["pill", "mock"];
  if (probe.reachable === false) return ["pill pill--danger", "unreachable"];
  const bad = probe.notices.find((n) => n.tone === "danger");
  if (bad) return ["pill pill--danger", bad.title];
  if (probe.ready?.status === "ready") return ["pill pill--accent", `ready${probe.ready.role ? ` · ${probe.ready.role}` : ""}`];
  return ["pill pill--warn", probe.ready?.status ?? "unknown"];
}

export function Connections({
  selected,
  pheasant,
  onSelected,
  render,
}: {
  selected: string | null;
  pheasant?: Resolved["pheasant"];
  onSelected: (settings: Settings) => void;
  render: (key: string, compact?: boolean) => React.ReactNode;
}) {
  const [rows, setRows] = useState<Connection[]>([]);
  const [probes, setProbes] = useState<Record<string, RegionProbe>>({});
  const [form, setForm] = useState<FormState | null>(null);
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [showTools, setShowTools] = useState(false);

  const load = useCallback(() => {
    void api
      .connections()
      .then((result) => setRows(result.connections))
      .catch((e: Error) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  const probe = async (name: string) => {
    setBusy(`probe:${name}`);
    try {
      const result = await api.probeConnection(name);
      setProbes((current) => ({ ...current, [name]: result }));
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const choose = async (name: string | null) => {
    setBusy(`select:${name}`);
    setError(null);
    try {
      onSelected(await api.selectConnection(name));
      if (name) void probe(name);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const save = async () => {
    if (!form) return;
    setBusy("save");
    setError(null);
    try {
      await api.saveConnection(
        {
          name: form.name.trim(),
          description: form.description,
          transport: form.transport,
          url: form.url,
          command: form.command,
          knowledge_base: form.knowledge_base,
          source_name: form.source_name,
          token_env: form.token_env || null,
          mock_claim_seconds: form.mock_claim_seconds === "" ? null : Number(form.mock_claim_seconds),
        },
        token === "" ? undefined : token,
      );
      setForm(null);
      setToken("");
      load();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const remove = async (name: string) => {
    if (!window.confirm(`Delete the connection ${name}? Its stored token goes with it.`)) return;
    try {
      await api.deleteConnection(name);
      load();
    } catch (caught) {
      setError((caught as Error).message);
    }
  };

  const edit = (row: Connection) => {
    setToken("");
    setForm({
      name: row.name,
      description: row.description,
      transport: row.transport,
      url: row.url ?? "",
      command: row.command ?? "",
      knowledge_base: row.knowledge_base ?? "",
      source_name: row.source_name ?? "",
      token_env: row.token_env ?? "",
      mock_claim_seconds: row.mock_claim_seconds == null ? "" : String(row.mock_claim_seconds),
    });
  };

  const patch = (change: Partial<FormState>) => setForm((current) => (current ? { ...current, ...change } : current));

  return (
    <div className="card" id="connection">
      <div className="card__head">
        Pheasant connection <span className="sub">which pheasant-kb region the run writes to, indexes and searches</span>
        <div className="r">
          {selected ? (
            <button className="btn btn--ghost btn--small" onClick={() => void choose(null)}>
              use the config file's region
            </button>
          ) : null}
          <button className="btn btn--small btn--primary" onClick={() => { setToken(""); setForm({ ...EMPTY }); }}>
            + Connection
          </button>
        </div>
      </div>
      {error ? (
        <div className="card__body" style={{ paddingBottom: 0 }}>
          <span className="pill pill--danger" style={{ whiteSpace: "normal" }}>{error}</span>
        </div>
      ) : null}
      <div className="card__body conns">
        {rows.map((row) => {
          const on = row.name === selected;
          const [tone, text] = probeTone(probes[row.name]);
          return (
            <div key={row.name} className={`conn${on ? " conn--on" : ""}`}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span className={`radio${on ? " radio--on" : ""}`} />
                <b>{row.name}</b>
                <span className="pill">{row.transport === "streamable_http" ? "http" : row.transport}</span>
                <span className={tone} style={{ marginLeft: "auto" }}>{busy === `probe:${row.name}` ? <span className="spinner spinner--tiny" /> : null}{text}</span>
              </div>
              <div className="mono small">{row.transport === "mock" ? "in-process mock" : row.url ?? row.command}</div>
              <div className="muted small">
                kb <b>{row.knowledge_base || "—"}</b> · source <b>{row.source_name || "—"}</b> · token{" "}
                {row.token_stored ? "stored" : row.token_in_environment ? `from $${row.token_env}` : row.transport === "mock" ? "not needed" : `$${row.token_env} unset`}
              </div>
              {row.description ? <div className="muted small">{row.description}</div> : null}
              <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
                <button className={`btn btn--small${on ? "" : " btn--primary"}`} disabled={on || busy === `select:${row.name}`} onClick={() => void choose(row.name)}>
                  {on ? "selected" : "Use"}
                </button>
                <button className="btn btn--small" onClick={() => void probe(row.name)}>Probe</button>
                <button className="btn btn--small btn--ghost" onClick={() => edit(row)}>Edit</button>
                {!row.shipped ? (
                  <button className="btn btn--small btn--ghost" onClick={() => void remove(row.name)} aria-label={`Delete ${row.name}`}>
                    ✕
                  </button>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>

      {form ? (
        <div className="card__body" style={{ borderTop: "1px solid var(--border)" }}>
          <div className="grid3">
            <div className="fld">
              <label>name</label>
              <input className="input mono" value={form.name} onChange={(e) => patch({ name: e.target.value })} placeholder="e.g. team-region" />
              <div className="h">lowercase, digits, hyphens</div>
            </div>
            <div className="fld">
              <label>transport</label>
              <select className="input" value={form.transport} onChange={(e) => patch({ transport: e.target.value as Connection["transport"] })}>
                <option value="streamable_http">streamable_http — a running region's /mcp</option>
                <option value="stdio">stdio — a command the lab starts</option>
                <option value="mock">mock — in-process, offline</option>
              </select>
            </div>
            {form.transport === "streamable_http" ? (
              <div className="fld">
                <label>MCP URL</label>
                <input className="input mono" value={form.url} onChange={(e) => patch({ url: e.target.value })} />
                <div className="h">pheasant-kb ≥ 0.13.11, ending in /mcp. In Compose: http://pheasant:8765/mcp</div>
              </div>
            ) : form.transport === "stdio" ? (
              <div className="fld">
                <label>command</label>
                <input className="input mono" value={form.command} onChange={(e) => patch({ command: e.target.value })} placeholder="pheasant mcp --transport stdio" />
              </div>
            ) : (
              <div className="fld">
                <label>mock_claim_seconds</label>
                <input className="input" inputMode="decimal" value={form.mock_claim_seconds} onChange={(e) => patch({ mock_claim_seconds: e.target.value })} placeholder="0" />
                <div className="h">above 0: behave like a fleet — syncs are queued, then claimed</div>
              </div>
            )}
            <div className="fld">
              <label>knowledge_base</label>
              <input className="input mono" value={form.knowledge_base} onChange={(e) => patch({ knowledge_base: e.target.value })} />
              <div className="h">the region's <code>pheasant.name</code>; one that holds nothing else</div>
            </div>
            <div className="fld">
              <label>source_name</label>
              <input className="input mono" value={form.source_name} onChange={(e) => patch({ source_name: e.target.value })} />
              <div className="h">the document_folder source the lab registers and syncs</div>
            </div>
            <div className="fld">
              <label>token variable</label>
              <input className="input mono" value={form.token_env} onChange={(e) => patch({ token_env: e.target.value })} placeholder="PHEASANT_API_TOKEN" />
              <div className="h">where the region's API token is read (its <code>security.api_auth.token_env</code>)</div>
            </div>
            <div className="fld" style={{ gridColumn: "1 / 3" }}>
              <label>description</label>
              <input className="input" value={form.description} onChange={(e) => patch({ description: e.target.value })} />
            </div>
            <div className="fld">
              <label>token</label>
              <input className="input mono" type="password" autoComplete="new-password" value={token} onChange={(e) => setToken(e.target.value)} placeholder="leave blank to keep" />
              <div className="h">stored 0600, never shown again; blank keeps the current one</div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button className="btn" style={{ marginLeft: "auto" }} onClick={() => setForm(null)}>Cancel</button>
            <button className="btn btn--primary" disabled={busy === "save" || !form.name.trim()} onClick={() => void save()}>
              {busy === "save" ? <span className="spinner" /> : null} Save connection
            </button>
          </div>
        </div>
      ) : null}

      <div className="card__body grid3" style={{ borderTop: "1px solid var(--border)" }}>
        <div className="fld">
          <label>in force</label>
          <input className="input mono" readOnly value={pheasant?.transport === "mock" ? "mock (in-process)" : `${pheasant?.transport ?? ""} ${pheasant?.url ?? ""}`} />
        </div>
        <div className="fld">
          <label>knowledge_base · source</label>
          <input className="input mono" readOnly value={`${pheasant?.knowledge_base ?? ""} · ${pheasant?.source_name ?? ""}`} />
        </div>
        {pheasant?.transport === "mock" ? render("mock_claim_seconds", true) : render("timeout_seconds", true)}
      </div>
      <div style={{ padding: "0 14px 12px" }}>
        <button className="btn btn--ghost btn--small" onClick={() => setShowTools((v) => !v)} aria-expanded={showTools}>
          {showTools ? "▾" : "▸"} capability map ({Object.keys(pheasant?.capabilities ?? {}).length} tools)
        </button>
        {showTools ? (
          <table className="table">
            <thead>
              <tr>
                <th>capability</th>
                <th>tool</th>
                <th>required</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(pheasant?.capabilities ?? {}).map(([name, spec]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td className="mono">{spec.tool}</td>
                  <td>{spec.required ? <span className="pill pill--accent">required</span> : <span className="pill">optional</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </div>
    </div>
  );
}
