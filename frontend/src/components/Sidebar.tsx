import { useState } from "react";
import {
  Database,
  FileUp,
  Link2,
  Loader2,
  MessageSquare,
  StickyNote,
} from "lucide-react";
import { api } from "../api";
import type { Session, Stats } from "../types";

type Props = {
  online: boolean;
  stats: Stats | null;
  sessions: Session[];
  onSelectSession: (s: Session) => void;
  onIngested: () => void;
};

export default function Sidebar({
  online,
  stats,
  sessions,
  onSelectSession,
  onIngested,
}: Props) {
  const [tab, setTab] = useState<"file" | "text" | "url">("text");
  const [text, setText] = useState("");
  const [title, setTitle] = useState("note");
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setMsg("");
    try {
      const r = await fn();
      setMsg(`Ingested ✓ ${JSON.stringify(r).slice(0, 80)}`);
      onIngested();
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside className="w-72 shrink-0 border-r border-border bg-panel flex flex-col">
      <div className="p-4 border-b border-border">
        <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">Graph stats</div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <Stat label="chunks" value={stats?.chunks} />
          <Stat label="nodes" value={stats?.nodes} />
          <Stat label="edges" value={stats?.edges} />
          <Stat label="backend" value={stats?.settings?.backend || stats?.backend} />
        </div>
        {stats?.ntypes && (
          <div className="mt-3 flex flex-wrap gap-1">
            {Object.entries(stats.ntypes).map(([k, v]) => (
              <span key={k} className="text-[10px] px-1.5 py-0.5 rounded bg-panel-2 text-slate-400 border border-border">
                {k}:{String(v)}
              </span>
            ))}
          </div>
        )}
        <div className={`mt-3 text-xs ${online ? "text-emerald-400" : "text-rose-400"}`}>
          {online ? "● API connected" : "○ API offline — start uvicorn"}
        </div>
      </div>

      {/* Add Knowledge */}
      <div className="p-4 border-b border-border">
        <div className="text-xs uppercase tracking-widest text-slate-500 mb-2 flex items-center gap-1">
          <Database size={12} /> Add Knowledge
        </div>
        <div className="flex gap-1 mb-2">
          {(["text", "url", "file"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`text-xs px-2 py-1 rounded ${
                tab === t ? "bg-accent/20 text-accent" : "text-slate-400 hover:bg-panel-2"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {tab === "text" && (
          <div className="space-y-2">
            <input
              className="w-full bg-panel-2 border border-border rounded px-2 py-1 text-xs"
              placeholder="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
            <textarea
              className="w-full h-24 bg-panel-2 border border-border rounded px-2 py-1 text-xs resize-none"
              placeholder="Paste text…"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <button
              disabled={busy || !text.trim()}
              onClick={() => run(() => api.ingestText(text, title))}
              className="w-full text-xs py-1.5 rounded bg-accent/90 hover:bg-accent text-white disabled:opacity-40 flex items-center justify-center gap-1"
            >
              {busy ? <Loader2 size={12} className="animate-spin" /> : <StickyNote size={12} />}
              Ingest text
            </button>
          </div>
        )}

        {tab === "url" && (
          <div className="space-y-2">
            <input
              className="w-full bg-panel-2 border border-border rounded px-2 py-1 text-xs"
              placeholder="https://…"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <button
              disabled={busy || !url.trim()}
              onClick={() => run(() => api.ingestUrl(url))}
              className="w-full text-xs py-1.5 rounded bg-accent/90 hover:bg-accent text-white disabled:opacity-40 flex items-center justify-center gap-1"
            >
              {busy ? <Loader2 size={12} className="animate-spin" /> : <Link2 size={12} />}
              Fetch & ingest
            </button>
          </div>
        )}

        {tab === "file" && (
          <label className="flex flex-col items-center justify-center gap-2 h-28 border border-dashed border-border rounded cursor-pointer hover:border-accent/50 text-xs text-slate-400">
            <FileUp size={16} />
            Drop or click .txt/.md
            <input
              type="file"
              className="hidden"
              accept=".txt,.md,.markdown,.json"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) run(() => api.ingestFile(f));
              }}
            />
          </label>
        )}
        {msg && <p className="mt-2 text-[10px] text-slate-400 break-all">{msg}</p>}
      </div>

      {/* Sessions */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-3">
        <div className="text-xs uppercase tracking-widest text-slate-500 mb-2 flex items-center gap-1">
          <MessageSquare size={12} /> Sessions
        </div>
        {sessions.length === 0 && (
          <p className="text-xs text-slate-600">Ask a question on the Canvas to save a session.</p>
        )}
        <ul className="space-y-1">
          {sessions.map((s) => (
            <li key={s.id}>
              <button
                onClick={() => onSelectSession(s)}
                className="w-full text-left px-2 py-1.5 rounded hover:bg-panel-2 border border-transparent hover:border-border"
              >
                <div className="text-xs truncate">{s.question}</div>
                <div className="text-[10px] text-slate-500 flex gap-2">
                  <span className="text-accent">{s.result.pipeline_used}</span>
                  <span>{s.result.tokens?.total ?? 0} tok</span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}

function Stat({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="bg-panel-2 rounded px-2 py-1.5 border border-border">
      <div className="text-[10px] text-slate-500">{label}</div>
      <div className="font-mono text-sm">{value ?? "—"}</div>
    </div>
  );
}