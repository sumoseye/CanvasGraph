import type { TraceStep } from "../types";

export default function TraceViewer({ trace }: { trace?: TraceStep[] }) {
  if (!trace?.length) {
    return <p className="text-xs text-slate-500">No agent trace for this run.</p>;
  }
  return (
    <ol className="space-y-2">
      {trace.map((t, i) => (
        <li
          key={i}
          className={`relative pl-4 border-l-2 ${
            t.strategy_changed ? "border-amber-400" : "border-border"
          }`}
        >
          <div className="flex items-center gap-2 text-xs">
            <span className="font-mono text-slate-500">#{t.step}</span>
            <span className="font-medium text-accent">{t.tool}</span>
            {t.strategy_changed && (
              <span className="text-[10px] px-1 rounded bg-amber-400/20 text-amber-300">
                strategy Δ
              </span>
            )}
            <span className="ml-auto text-slate-500">{t.duration_ms ?? 0} ms</span>
          </div>
          {t.reason && <div className="text-[11px] text-slate-400 mt-0.5">{t.reason}</div>}
          <div className="text-[10px] text-slate-500 font-mono mt-0.5 flex gap-3">
            <span>tok {t.tokens?.total ?? 0}</span>
            <span>chunks {t["#chunks"] ?? 0}</span>
          </div>
        </li>
      ))}
    </ol>
  );
}