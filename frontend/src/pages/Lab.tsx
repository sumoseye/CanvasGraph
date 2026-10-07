import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from "recharts";
import { Loader2, Play, RefreshCw } from "lucide-react";
import TraceViewer from "../components/TraceViewer";
import { api } from "../api";
import type { EvalRow, ResultsPayload, Summary } from "../types";

const PIPE_COLORS: Record<string, string> = {
  rag: "#2B6CB0",
  graphrag: "#5A67D8",
  agentic: "#2F855A",
  routed: "#D69E2E",
};

export default function Lab() {
  const [data, setData] = useState<ResultsPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [limit, setLimit] = useState(6);
  const [err, setErr] = useState("");
  const [selected, setSelected] = useState<EvalRow | null>(null);
  const [filterPipe, setFilterPipe] = useState<string>("all");

  const load = () =>
    api
      .results("latest")
      .then(setData)
      .catch((e) => setErr(String(e)));

  useEffect(() => {
    load();
  }, []);

  const summary: Summary | undefined = data?.summary;
  const rows = data?.rows || [];

  const accData = useMemo(() => {
    const p = summary?.pipelines || {};
    return Object.entries(p).map(([name, v]) => ({
      name,
      accuracy: Number(((v.accuracy || 0) * 100).toFixed(1)),
      tokens: v.avg_tokens || 0,
      n: v.n,
    }));
  }, [summary]);

  const tokData = accData;

  const scatterData = useMemo(
    () =>
      accData.map((d) => ({
        name: d.name,
        x: d.tokens,
        y: d.accuracy,
        z: d.n,
      })),
    [accData],
  );

  const byTypeData = useMemo(() => {
    const bd = summary?.breakdown_by_type || {};
    return Object.entries(bd).map(([type, pipes]) => {
      const row: Record<string, string | number> = { type };
      for (const [p, v] of Object.entries(pipes || {})) {
        row[p] = Number(((v.accuracy || 0) * 100).toFixed(1));
      }
      return row;
    });
  }, [summary]);

  const filteredRows = rows.filter(
    (r) => filterPipe === "all" || r.pipeline === filterPipe,
  );

  // side-by-side: pick one question id, show all pipelines
  const [sideId, setSideId] = useState<string>("");
  useEffect(() => {
    if (!sideId && rows.length) setSideId(rows[0].id);
  }, [rows, sideId]);
  const sideRows = rows.filter((r) => r.id === sideId && r.pipeline !== "routed");

  const runBench = async () => {
    setBusy(true);
    setErr("");
    setLog([]);
    try {
      await api.benchmark({ limit, file: "data/sample_questions.json" }, (ev) => {
        if (ev.event === "row") {
          setLog((L) =>
            [
              `[${ev.pipeline}] ${ev.id} correct=${ev.correct} tok=${ev.total_tokens}`,
              ...L,
            ].slice(0, 40),
          );
        } else if (ev.event === "start") {
          setLog((L) => [`start jobs=${ev.jobs}`, ...L]);
        } else if (ev.event === "done") {
          setLog((L) => [`done run=${(ev.summary as any)?.run}`, ...L]);
        }
      });
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto scrollbar-thin p-5 space-y-5">
      {/* controls */}
      <div className="flex flex-wrap items-center gap-3 glass rounded-xl p-3">
        <div>
          <div className="text-sm font-semibold">Benchmark Lab</div>
          <div className="text-xs text-slate-400">
            run={summary?.run || "—"} · Jev routing accuracy{" "}
            <span className="text-accent font-mono">
              {((summary?.jev_routing_accuracy || 0) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <label className="text-xs text-slate-400 flex items-center gap-1">
            limit
            <input
              type="number"
              min={1}
              max={50}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value) || 6)}
              className="w-14 bg-panel-2 border border-border rounded px-1 py-1 text-xs"
            />
          </label>
          <button
            onClick={load}
            className="text-xs px-2 py-1.5 rounded border border-border hover:bg-panel-2 flex items-center gap-1"
          >
            <RefreshCw size={12} /> Reload summary.json
          </button>
          <button
            disabled={busy}
            onClick={runBench}
            className="text-xs px-3 py-1.5 rounded bg-accent text-white hover:bg-accent/90 disabled:opacity-50 flex items-center gap-1"
          >
            {busy ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
            Run pipelines + Jev
          </button>
        </div>
      </div>
      {err && <p className="text-sm text-rose-400">{err}</p>}

      {/* charts */}
      <div className="grid lg:grid-cols-2 gap-4">
        <ChartCard title="Accuracy by pipeline">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={accData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} unit="%" />
              <Tooltip contentStyle={{ background: "#121826", border: "1px solid #2a3548" }} />
              <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                {accData.map((d) => (
                  <Cell key={d.name} fill={PIPE_COLORS[d.name] || "#6c8cff"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Tokens by pipeline">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={tokData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip contentStyle={{ background: "#121826", border: "1px solid #2a3548" }} />
              <Bar dataKey="tokens" radius={[4, 4, 0, 0]}>
                {tokData.map((d) => (
                  <Cell key={d.name} fill={PIPE_COLORS[d.name] || "#6c8cff"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Accuracy vs tokens">
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
              <XAxis
                dataKey="x"
                name="tokens"
                stroke="#94a3b8"
                fontSize={11}
                label={{ value: "avg tokens", position: "insideBottom", offset: -2, fill: "#64748b" }}
              />
              <YAxis
                dataKey="y"
                name="accuracy"
                stroke="#94a3b8"
                fontSize={11}
                unit="%"
              />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                contentStyle={{ background: "#121826", border: "1px solid #2a3548" }}
                formatter={(v: any, name: string) => [v, name]}
                labelFormatter={() => ""}
              />
              <Scatter data={scatterData}>
                {scatterData.map((d) => (
                  <Cell key={d.name} fill={PIPE_COLORS[d.name] || "#6c8cff"} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title='Accuracy by question type — "when agents matter"'>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={byTypeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
              <XAxis dataKey="type" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} unit="%" />
              <Tooltip contentStyle={{ background: "#121826", border: "1px solid #2a3548" }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="rag" fill={PIPE_COLORS.rag} />
              <Bar dataKey="graphrag" fill={PIPE_COLORS.graphrag} />
              <Bar dataKey="agentic" fill={PIPE_COLORS.agentic} />
            </BarChart>
          </ResponsiveContainer>
          {summary?.when_agents_matter && (
            <div className="px-3 pb-2 flex flex-wrap gap-2">
              {Object.entries(summary.when_agents_matter).map(([t, v]) => (
                <span
                  key={t}
                  className="text-[10px] px-2 py-0.5 rounded-full bg-panel-2 border border-border text-slate-300"
                >
                  {t}: <span className="text-accent">{v}</span>
                </span>
              ))}
            </div>
          )}
        </ChartCard>
      </div>

      {/* side by side answers */}
      <div className="glass rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <h2 className="text-sm font-semibold">Side-by-side answers</h2>
          <select
            className="ml-auto text-xs bg-panel-2 border border-border rounded px-2 py-1"
            value={sideId}
            onChange={(e) => setSideId(e.target.value)}
          >
            {[...new Set(rows.map((r) => r.id))].map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </div>
        <div className="grid md:grid-cols-3 gap-3">
          {sideRows.map((r) => (
            <div
              key={r.pipeline}
              className={`rounded-lg border p-3 bg-panel-2 ${
                r.correct ? "border-emerald-500/40" : "border-border"
              }`}
            >
              <div className="flex text-xs mb-1">
                <span className="font-medium" style={{ color: PIPE_COLORS[r.pipeline] }}>
                  {r.pipeline}
                </span>
                <span className="ml-auto">{r.correct ? "PASS" : "FAIL"}</span>
              </div>
              <p className="text-xs text-slate-300 line-clamp-6 whitespace-pre-wrap">{r.answer}</p>
              <div className="text-[10px] text-slate-500 mt-2 font-mono">
                {r.total_tokens} tok · {r.latency_ms} ms
              </div>
            </div>
          ))}
          {!sideRows.length && (
            <p className="text-xs text-slate-500">No rows — run a benchmark or reload summary.</p>
          )}
        </div>
      </div>

      {/* results table + trace */}
      <div className="grid lg:grid-cols-5 gap-4">
        <div className="lg:col-span-3 glass rounded-xl overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
            <h2 className="text-sm font-semibold">Results</h2>
            <select
              className="ml-auto text-xs bg-panel-2 border border-border rounded px-2 py-1"
              value={filterPipe}
              onChange={(e) => setFilterPipe(e.target.value)}
            >
              <option value="all">all pipelines</option>
              {["rag", "graphrag", "agentic", "routed"].map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
          <div className="max-h-96 overflow-auto scrollbar-thin">
            <table className="w-full text-xs">
              <thead className="text-slate-500 sticky top-0 bg-panel">
                <tr>
                  <th className="text-left p-2">id</th>
                  <th className="text-left p-2">type</th>
                  <th className="text-left p-2">pipe</th>
                  <th className="text-left p-2">ok</th>
                  <th className="text-right p-2">tok</th>
                  <th className="text-right p-2">ms</th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((r, i) => (
                  <tr
                    key={`${r.id}-${r.pipeline}-${i}`}
                    onClick={() => setSelected(r)}
                    className={`border-t border-border cursor-pointer hover:bg-panel-2 ${
                      selected === r ? "bg-accent/10" : ""
                    }`}
                  >
                    <td className="p-2 font-mono">{r.id}</td>
                    <td className="p-2">{r.type}</td>
                    <td className="p-2" style={{ color: PIPE_COLORS[r.pipeline] }}>
                      {r.pipeline}
                    </td>
                    <td className="p-2">{r.correct ? "✓" : "✗"}</td>
                    <td className="p-2 text-right font-mono">{r.total_tokens}</td>
                    <td className="p-2 text-right font-mono">{r.latency_ms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-2 glass rounded-xl p-3 min-h-64">
          <h2 className="text-sm font-semibold mb-2">Trace viewer</h2>
          {selected ? (
            <div className="space-y-3">
              <div className="text-xs text-slate-400">
                {selected.id} · {selected.pipeline} · {selected.verdict || (selected.correct ? "PASS" : "FAIL")}
              </div>
              <p className="text-xs text-slate-300 whitespace-pre-wrap line-clamp-4">
                {selected.answer}
              </p>
              <p className="text-[11px] text-slate-500">{selected.reason}</p>
              <TraceViewer trace={selected.trace} />
              {!!selected.opt_logs?.length && (
                <ul className="text-[10px] font-mono text-slate-500">
                  {selected.opt_logs.map((l, i) => (
                    <li key={i}>· {l}</li>
                  ))}
                </ul>
              )}
            </div>
          ) : (
            <p className="text-xs text-slate-500">Select a result row to inspect its agent trace.</p>
          )}
        </div>
      </div>

      {/* live log */}
      {!!log.length && (
        <div className="glass rounded-xl p-3">
          <h2 className="text-sm font-semibold mb-2">Run log</h2>
          <pre className="text-[11px] font-mono text-slate-400 max-h-40 overflow-auto scrollbar-thin">
            {log.join("\n")}
          </pre>
        </div>
      )}
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="px-3 py-2 border-b border-border text-sm font-medium">{title}</div>
      <div className="p-2">{children}</div>
    </div>
  );
}