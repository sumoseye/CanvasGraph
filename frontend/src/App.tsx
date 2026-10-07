import { NavLink, Route, Routes } from "react-router-dom";
import { useEffect, useState } from "react";
import { Activity, FlaskConical, Hexagon } from "lucide-react";
import Sidebar from "./components/Sidebar";
import Canvas from "./pages/Canvas";
import Lab from "./pages/Lab";
import { api } from "./api";
import type { Session, Stats } from "./types";

export default function App() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [online, setOnline] = useState(false);
  const [sessions, setSessions] = useState<Session[]>(() => {
    try {
      return JSON.parse(localStorage.getItem("cg_sessions") || "[]");
    } catch {
      return [];
    }
  });
  const [active, setActive] = useState<Session | null>(null);

  const refreshStats = () =>
    api
      .stats()
      .then((s) => {
        setStats(s);
        setOnline(true);
      })
      .catch(() => setOnline(false));

  useEffect(() => {
    refreshStats();
    const t = setInterval(refreshStats, 15000);
    return () => clearInterval(t);
  }, []);

  const pushSession = (s: Session) => {
    setSessions((prev) => {
      const next = [s, ...prev.filter((x) => x.id !== s.id)].slice(0, 30);
      localStorage.setItem("cg_sessions", JSON.stringify(next));
      return next;
    });
    setActive(s);
  };

  return (
    <div className="h-full flex bg-canvas-bg text-slate-100">
      <Sidebar
        online={online}
        stats={stats}
        sessions={sessions}
        onSelectSession={setActive}
        onIngested={refreshStats}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 shrink-0 border-b border-border flex items-center gap-6 px-5 glass">
          <div className="flex items-center gap-2 font-semibold tracking-wide">
            <Hexagon className="text-accent" size={20} />
            <span>CanvasGraph</span>
            <span className="text-xs font-normal text-slate-400 ml-1">TigerGraph Agentic GraphRAG</span>
          </div>
          <nav className="flex gap-1 ml-4">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `px-3 py-1.5 rounded-md text-sm flex items-center gap-1.5 ${
                  isActive ? "bg-accent/20 text-accent" : "text-slate-400 hover:text-slate-200"
                }`
              }
            >
              <Activity size={14} /> Canvas
            </NavLink>
            <NavLink
              to="/lab"
              className={({ isActive }) =>
                `px-3 py-1.5 rounded-md text-sm flex items-center gap-1.5 ${
                  isActive ? "bg-accent/20 text-accent" : "text-slate-400 hover:text-slate-200"
                }`
              }
            >
              <FlaskConical size={14} /> Lab
            </NavLink>
          </nav>
          <div className="ml-auto flex items-center gap-3 text-xs text-slate-400">
            <span className={`inline-flex items-center gap-1.5 ${online ? "text-emerald-400" : "text-rose-400"}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${online ? "bg-emerald-400" : "bg-rose-400"}`} />
              {online ? "backend online" : "backend offline"}
            </span>
            <span className="font-mono">{stats?.settings?.model || "—"}</span>
          </div>
        </header>

        <main className="flex-1 min-h-0">
          <Routes>
            <Route
              path="/"
              element={
                <Canvas
                  active={active}
                  onResult={pushSession}
                  onGraphMaybeChanged={refreshStats}
                />
              }
            />
            <Route path="/lab" element={<Lab />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}