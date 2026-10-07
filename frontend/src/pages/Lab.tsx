import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Loader2, Play, RefreshCw, AlertCircle } from "lucide-react";
import TraceViewer from "../components/TraceViewer";
import { api } from "../api";
import type { EvalRow, ResultsPayload, Summary } from "../types";

const PIPE_COLORS: Record<string, string> = {
  rag: "#2B6CB0",
  graphrag: "#5A67D8",
  agentic: "#2F855A",
  routed: "#D69E2E",
};

const DATASETS = [
  { label: "eval_public.jsonl (official)", file: "data/questions/eval_public.jsonl" },
  { label: "sample_questions.json (local)", file: "data/sample_questions.json" },
];

export default function Lab() {
  const [data, setData] = useState<ResultsPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [limit, setLimit] = useState(6);
  const [fastJudge, setFastJudge] = useState(true);
  const [datasetFile, setDatasetFile] = useState(DATASETS[0].file);
  const [err, setErr] = useState("");
  const [selected, setSelected] = useState<EvalRow | null>(null);
  const [filterPipe, setFilterPipe] = useState<string>("all");

  const load = () =>
    api
      .results("latest")
      .then((res) => {
        setData(res);
        setErr("");
      })
      .catch((e) => setErr("No completed summary found. Click 'Run pipelines + Jev' to start."));

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

  const [sideId, setSideId] = useState<string>("");
  useEffect(() => {
    if (!sideId && rows.length) setSideId(rows[0].id);
  }, [rows, sideId]);
  const sideRows = rows.filter((r) => r.id === sideId && r.pipeline !== "routed");

  const runBench = async () => {
    setBusy(true);
    setErr("");
    setLog(["Initializing benchmark runner..."]);
    try {
      await api.benchmark(
        { limit, file: datasetFile, fast: fastJudge } as any,
        (ev) => {
          if (ev.event === "row") {
            setLog((L) =>
              [
                `[${ev.pipeline}] ${ev.id} correct=${ev.correct} tok=${ev.total_tokens}`,
                ...L,
              ].slice(0, 50),
            );
          } else if (ev.event === "start") {
            setLog((L) => [`Started benchmark: ${ev.questions} questions, ${ev.jobs} jobs`, ...L]);
          } else if (ev.event === "done") {
            setLog((L) => [`Benchmark complete! Saved run ${ev.run}`, ...L]);
          } else if (ev.event === "error") {
            setLog((L) => [`Error: ${ev.error}`, ...L]);
            setErr(String(ev.error));
          }
        },
      );
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const hasData = accData.length > 0 && accData.some((d) => d.n > 0);

  return (
    <div className="h-full overflow-y-auto scrollbar-thin p-5 space-y-5">
      {/* Controls Header */}
      <div className="flex flex-wrap items-center gap-3 glass rounded-xl p-3">
        <div>
          <div className="text-sm font-semibold">Benchmark Lab</div>
          <div className="text-xs text-slate-400 flex items-center gap-3">
            <span>run={summary?.run || "—"}</span>
            <span>
              Jev routing <span className="text-accent font-mono font-bold">{((summary?.jev_routing_accuracy || 0) * 100).toFixed(1)}%</span>
            </span>
            {summary?.breakdown_by_type?.conflict && (
               <span className="border-l border-border pl-3">
                 Agentic Conflict Accuracy <span className="text-rose-400 font-mono font-bold">{((summary.breakdown_by_type.conflict.agentic?.accuracy || 0) * 100).toFixed(1)}%</span> 
                 <span className="text-slate-500"> (vs RAG {((summary.breakdown_by_type.conflict.rag?.accuracy || 0) * 100).toFixed(1)}%)</span>
               </span>
            )}
          </div>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          {/* Dataset Selector */}
          <select
            value={datasetFile}
            onChange={(e) => setDatasetFile(e.target.value)}
            className="text-xs bg-panel-2 border border-border rounded px-2 py-1 text-slate-200"
          >
            {DATASETS.map((d) => (
              <option key={d.file} value={d.file}>
                {d.label}
              </option>
            ))}
          </select>

          {/* Fast Judge Checkbox */}
          <label className="text-xs text-slate-300 flex items-center gap-1 cursor-pointer bg-panel-2 px-2 py-1 rounded border border-border">
            <input
              type="checkbox"
              checked={fastJudge}
              onChange={(e) => setFastJudge(e.target.checked)}
              className="accent-accent"
            />
            Fast Judge (0 LLM tokens)
          </label>

          {/* Limit input */}
          <label className="text-xs text-slate-400 flex items-center gap-1">
            limit
            <input
              type="number"
              min={1}
              max={50}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value) || 6)}
              className="w-12 bg-panel-2 border border-border rounded px-1 py-1 text-xs text-center"
            />
          </label>

          <button
            onClick={load}
            className="text-xs px-2.5 py-1.5 rounded border border-border hover:bg-panel-2 flex items-center gap-1"
          >
            <RefreshCw size={12} /> Reload
          </button>

          <button
            disabled={busy}
            onClick={runBench}
            className="text-xs px-3.5 py-1.5 rounded bg-accent text-white font-medium hover:bg-accent/90 disabled:opacity-50 flex items-center gap-1 shadow-lg shadow-accent/20"
          >
            {busy ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
            Run pipelines + Jev
          </button>
        </div>
      </div>

      {err && (
        <div className="flex items-center gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded-lg p-2.5">
          <AlertCircle size={14} className="shrink-0" />
          <span>{err}</span>
        </div>
      )}

      {/* Recharts Grid */}
      <div className="grid lg:grid-cols-2 gap-4">
        <ChartCard title="Accuracy by pipeline">
          {hasData ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={accData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} unit="%" domain={[0, 100]} />
                <Tooltip contentStyle={{ background: "#121826", border: "1px solid #2a3548" }} />
                <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                  {accData.map((d) => (
                    <Cell key={d.name} fill={PIPE_COLORS[d.name] || "#6c8cff"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChartState />
          )}
        </ChartCard>

        <ChartCard title="Tokens by pipeline">
          {hasData ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={accData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip contentStyle={{ background: "#121826", border: "1px solid #2a3548" }} />
                <Bar dataKey="tokens" radius={[4, 4, 0, 0]}>
                  {accData.map((d) => (
                    <Cell key={d.name} fill={PIPE_COLORS[d.name] || "#6c8cff"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChartState />
          )}
        </ChartCard>

        <ChartCard title="Accuracy vs tokens">
          {hasData ? (
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
                <YAxis dataKey="y" name="accuracy" stroke="#94a3b8" fontSize={11} unit="%" domain={[0, 100]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{ background: "#121826", border: "1px solid #2a3548" }}
                />
                <Scatter data={scatterData}>
                  {scatterData.map((d) => (
                    <Cell key={d.name} fill={PIPE_COLORS[d.name] || "#6c8cff"} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChartState />
          )}
        </ChartCard>

        <ChartCard title='Accuracy by question type — "when agents matter"'>
          {hasData ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={byTypeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
                <XAxis dataKey="type" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} unit="%" domain={[0, 100]} />
                <Tooltip contentStyle={{ background: "#121826", border: "1px solid #2a3548" }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="rag" fill={PIPE_COLORS.rag} radius={[2, 2, 0, 0]} />
                <Bar dataKey="graphrag" fill={PIPE_COLORS.graphrag} radius={[2, 2, 0, 0]} />
                <Bar dataKey="agentic" fill={PIPE_COLORS.agentic} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChartState />
          )}
        </ChartCard>
      </div>

      {/* Side by side answers */}
      <div className="glass rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <h2 className="text-sm font-semibold">Side-by-side answers</h2>
          <select
            className="ml-auto text-xs bg-panel-2 border border-border rounded px-2 py-1 text-slate-200"
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
                r.correct ? "border-emerald-500/40" : "border-rose-500/30"
              }`}
            >
              <div className="flex text-xs mb-1">
                <span className="font-medium capitalize" style={{ color: PIPE_COLORS[r.pipeline] }}>
                  {r.pipeline}
                </span>
                <span className={`ml-auto font-mono ${r.correct ? "text-emerald-400" : "text-rose-400"}`}>
                  {r.correct ? "PASS" : "FAIL"}
                </span>
              </div>
              <p className="text-xs text-slate-300 line-clamp-6 whitespace-pre-wrap">{r.answer}</p>
              <div className="text-[10px] text-slate-500 mt-2 font-mono flex gap-3">
                <span>{r.total_tokens} tok</span>
                <span>{r.latency_ms} ms</span>
              </div>
            </div>
          ))}
          {!sideRows.length && (
            <p className="text-xs text-slate-500">No answer rows available. Run a benchmark above.</p>
          )}
        </div>
      </div>

      {/* Results table + Trace */}
      <div className="grid lg:grid-cols-5 gap-4">
        <div className="lg:col-span-3 glass rounded-xl overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
            <h2 className="text-sm font-semibold">Results ({filteredRows.length})</h2>
            <select
              className="ml-auto text-xs bg-panel-2 border border-border rounded px-2 py-1 text-slate-200"
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
                    <td className="p-2 font-mono text-slate-300">{r.id}</td>
                    <td className="p-2 text-slate-400">{r.type}</td>
                    <td className="p-2 font-medium" style={{ color: PIPE_COLORS[r.pipeline] }}>
                      {r.pipeline}
                    </td>
                    <td className={`p-2 ${r.correct ? "text-emerald-400" : "text-rose-400"}`}>
                      {r.correct ? "✓" : "✗"}
                    </td>
                    <td className="p-2 text-right font-mono text-slate-400">{r.total_tokens}</td>
                    <td className="p-2 text-right font-mono text-slate-400">{r.latency_ms}</td>
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
            </div>
          ) : (
            <p className="text-xs text-slate-500">Select a row from the table to inspect its agent trace.</p>
          )}
        </div>
      </div>

      {/* Run Log */}
      {!!log.length && (
        <div className="glass rounded-xl p-3">
          <h2 className="text-sm font-semibold mb-2">Live Execution Log</h2>
          <pre className="text-[11px] font-mono text-slate-400 max-h-40 overflow-auto scrollbar-thin leading-relaxed">
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

function EmptyChartState() {
  return (
    <div className="h-[220px] flex items-center justify-center text-xs text-slate-500">
      Click "Run pipelines + Jev" above to populate benchmark charts.
    </div>
  );
}