import { useEffect, useMemo, useState } from "react";
import { Loader2, Search, Sparkles } from "lucide-react";
import ForceGraph from "../components/ForceGraph";
import TraceViewer from "../components/TraceViewer";
import { api } from "../api";
import type { AskResult, GraphNode, Pipeline, Session } from "../types";

const EXAMPLES = [
  "What is the official 2025 revenue for PulseDB, and what was previously reported?",
  "Who is the current CEO of Aether Labs?",
  "Who led the Series B funding round for Aether Labs?",
];

type Props = {
  active: Session | null;
  onResult: (s: Session) => void;
  onGraphMaybeChanged: () => void;
};

export default function Canvas({ active, onResult }: Props) {
  const [q, setQ] = useState(EXAMPLES[0]);
  const [mode, setMode] = useState<Pipeline>("auto");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [result, setResult] = useState<AskResult | null>(null);
  const [tab, setTab] = useState<"answer" | "agent">("answer");
  const [graph, setGraph] = useState<{ nodes: GraphNode[]; edges: any[] }>({
    nodes: [],
    edges: [],
  });

  useEffect(() => {
    api.graph(250).then((g) => setGraph({ nodes: g.nodes || [], edges: g.edges || [] })).catch(() => {});
  }, []);

  useEffect(() => {
    if (active) {
      setResult(active.result);
      setQ(active.question);
      setTab("answer");
    }
  }, [active]);

  const highlight = useMemo(() => {
    const ids = new Set<string>();
    (result?.citations || []).forEach((c) => ids.add(String(c)));
    (result?.subgraph?.nodes || []).forEach((n) => ids.add(String(n.id)));
    return ids;
  }, [result]);

  const merged = useMemo(() => {
    const map = new Map(graph.nodes.map((n) => [n.id, n]));
    for (const n of result?.subgraph?.nodes || []) {
      if (!map.has(n.id)) map.set(n.id, { ...n, label: n.label || n.id });
    }
    return {
      nodes: [...map.values()],
      edges: [...graph.edges, ...(result?.subgraph?.edges || [])],
    };
  }, [graph, result]);

  const ask = async (question = q) => {
    if (!question.trim()) return;
    setBusy(true);
    setErr("");
    try {
      const r = await api.ask(question.trim(), mode);
      setResult(r);
      setTab("answer");
      onResult({
        id: `${Date.now()}`,
        question: question.trim(),
        mode,
        result: r,
        ts: Date.now(),
      });
      api.graph(250).then((g) => setGraph({ nodes: g.nodes || [], edges: g.edges || [] })).catch(() => {});
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const probs = result?.jev_probs || {};
  const tok = result?.tokens || { prompt: 0, completion: 0, total: 0 };

  return (
    <div className="h-full flex min-h-0 relative overflow-hidden">
      {/* center stage - ADDED overflow-hidden to prevent canvas bleed */}
      <div className="flex-1 relative min-w-0 flex flex-col overflow-hidden">
        <div className="flex-1 min-h-0">
          <ForceGraph nodes={merged.nodes} edges={merged.edges} highlightIds={highlight} />
        </div>

        {/* glass search */}
        <div className="absolute left-1/2 -translate-x-1/2 bottom-6 w-[min(720px,92%)] z-10">
          <div className="glass rounded-2xl shadow-2xl p-3">
            <div className="flex gap-2 mb-2">
              {(["auto", "rag", "graphrag", "agentic"] as Pipeline[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`text-[11px] px-2 py-1 rounded-full border ${
                    mode === m
                      ? "border-accent bg-accent/20 text-accent"
                      : "border-border text-slate-400"
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
            <div className="flex gap-2">
              <div className="flex-1 flex items-center gap-2 bg-panel-2 rounded-xl px-3 border border-border">
                <Search size={16} className="text-slate-500" />
                <input
                  className="flex-1 bg-transparent outline-none py-2.5 text-sm"
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && ask()}
                  placeholder="Ask the knowledge graph…"
                />
              </div>
              <button
                onClick={() => ask()}
                disabled={busy}
                className="px-4 rounded-xl bg-accent hover:bg-accent/90 text-white text-sm font-medium disabled:opacity-50 flex items-center gap-1.5 cursor-pointer"
              >
                {busy ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
                Ask
              </button>
            </div>
            <div className="flex gap-2 mt-2 flex-wrap">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => {
                    setQ(ex);
                    ask(ex);
                  }}
                  className="text-[10px] px-2 py-1 rounded-full bg-panel-2 text-slate-400 hover:text-slate-200 border border-border cursor-pointer"
                >
                  {ex.length > 48 ? ex.slice(0, 48) + "…" : ex}
                </button>
              ))}
            </div>
            {err && <p className="text-xs text-rose-400 mt-2">{err}</p>}
          </div>
        </div>
      </div>

      {/* right drawer - ADDED z-20 and relative to stay above graph */}
      <aside className="w-[400px] shrink-0 border-l border-border bg-panel flex flex-col min-h-0 relative z-20 shadow-2xl">
        <div className="flex border-b border-border">
          {(["answer", "agent"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`flex-1 py-3 text-sm capitalize cursor-pointer transition-colors ${
                tab === t ? "text-accent border-b-2 border-accent bg-accent/5" : "text-slate-400 hover:bg-panel-2"
              }`}
            >
              {t === "answer" ? "Answer" : "Jev + Trace"}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
          {!result && (
            <p className="text-sm text-slate-500">
              Run a query to see the answer, citations, highlighted subgraph, Jev routing and agent
              trace.
            </p>
          )}

          {result && tab === "answer" && (
            <>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="px-2 py-0.5 rounded bg-accent/20 text-accent font-medium">
                  {result.pipeline_used}
                </span>
                {result.mode_used && result.mode_used !== result.pipeline_used && (
                  <span>via {result.mode_used}</span>
                )}
                <span className="ml-auto font-mono">{result.latency_ms ?? 0} ms</span>
              </div>
              <article className="text-sm leading-relaxed whitespace-pre-wrap">{result.answer}</article>

              <section>
                <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-2 mt-4">Citations</h3>
                <div className="flex flex-wrap gap-1">
                  {(result.citations || []).map((c) => (
                    <span
                      key={c}
                      className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-documents/20 text-documents border border-documents/30"
                    >
                      {c}
                    </span>
                  ))}
                  {!result.citations?.length && (
                    <span className="text-xs text-slate-600">none</span>
                  )}
                </div>
              </section>
              {/* Highlighted subgraph view removed from here per your request */}
            </>
          )}

          {result && tab === "agent" && (
            <>
              <section>
                <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-2">
                  Jev probabilities
                </h3>
                <div className="space-y-2">
                  {["rag", "graphrag", "agentic"].map((k) => {
                    const p = Number(probs[k] || 0);
                    return (
                      <div key={k}>
                        <div className="flex text-xs mb-0.5">
                          <span className="capitalize">{k}</span>
                          <span className="ml-auto font-mono">{(p * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 rounded bg-panel-2 overflow-hidden">
                          <div
                            className="h-full rounded bg-accent"
                            style={{ width: `${Math.max(2, p * 100)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
                <p className="text-[11px] text-slate-500 mt-2">
                  rule: <span className="font-mono text-slate-300">{result.route_rule || "—"}</span>
                </p>
              </section>

              <section>
                <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-2">Token meter</h3>
                <div className="grid grid-cols-3 gap-2 text-center">
                  {[
                    ["prompt", tok.prompt],
                    ["completion", tok.completion],
                    ["total", tok.total],
                  ].map(([k, v]) => (
                    <div key={k as string} className="bg-panel-2 rounded border border-border p-2">
                      <div className="text-[10px] text-slate-500">{k}</div>
                      <div className="font-mono text-sm">{v as number}</div>
                    </div>
                  ))}
                </div>
                <div className="text-[11px] text-slate-500 mt-2 flex gap-3">
                  <span>ctx {result.context_tokens ?? 0}</span>
                  <span>steps {result.steps ?? 0}</span>
                  <span>stop {result.stop_reason || "—"}</span>
                </div>
              </section>

              <section>
                <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-2">
                  Agent trace
                </h3>
                <TraceViewer trace={result.trace} />
              </section>

              <section>
                <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-2">
                  Optimization logs
                </h3>
                <ul className="text-[11px] font-mono text-slate-400 space-y-1">
                  {(result.opt_logs || []).map((l, i) => (
                    <li key={i} className="truncate" title={l}>
                      · {l}
                    </li>
                  ))}
                  {!result.opt_logs?.length && <li className="text-slate-600">none</li>}
                </ul>
              </section>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}