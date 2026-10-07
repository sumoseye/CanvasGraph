import type { AskResult, GraphPayload, ResultsPayload, Stats, Summary } from "./types";

const BASE = "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} — ${text.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => req<{ ok: boolean; model?: string }>("/health"),

  stats: () => req<Stats>("/stats"),

  graph: (limit = 250) => req<GraphPayload>(`/graph?limit=${limit}`),

  ask: (question: string, mode: string = "auto") =>
    req<AskResult>("/ask", {
      method: "POST",
      body: JSON.stringify({ question, mode }),
    }),

  ingestPath: (path: string) =>
    req<unknown>("/ingest", { method: "POST", body: JSON.stringify({ path }) }),

  ingestText: (text: string, title = "pasted") =>
    req<unknown>("/ingest/text", {
      method: "POST",
      body: JSON.stringify({ text, title }),
    }),

  ingestUrl: (url: string) =>
    req<unknown>("/ingest/url", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),

  ingestFile: async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${BASE}/ingest/file`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  results: (run = "latest") => req<ResultsPayload>(`/results/${run}`),

  listRuns: () => req<{ runs: Array<{ run: string; file: string }> }>("/results"),

  summary: async (): Promise<Summary> => {
    const r = await req<ResultsPayload>("/results/latest");
    return r.summary;
  },

  /** SSE benchmark — calls onEvent for each progress frame, resolves with final summary event */
  benchmark: (
    opts: {
      file?: string;
      pipelines?: string[];
      limit?: number;
      run?: string;
    },
    onEvent: (ev: Record<string, unknown>) => void,
  ) =>
    new Promise<Record<string, unknown>>((resolve, reject) => {
      fetch(`${BASE}/benchmark`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file: opts.file || "data/sample_questions.json",
          pipelines: opts.pipelines,
          limit: opts.limit,
          run: opts.run,
        }),
      })
        .then(async (res) => {
          if (!res.ok || !res.body) throw new Error(`benchmark ${res.status}`);
          const reader = res.body.getReader();
          const dec = new TextDecoder();
          let buf = "";
          let last: Record<string, unknown> = {};
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            const parts = buf.split("\n\n");
            buf = parts.pop() || "";
            for (const block of parts) {
              const line = block.trim();
              if (!line.startsWith("data:")) continue;
              try {
                const ev = JSON.parse(line.slice(5).trim());
                onEvent(ev);
                last = ev;
                if (ev.event === "done") resolve(ev);
                if (ev.event === "error") reject(new Error(String(ev.error)));
              } catch {
                /* ignore partial */
              }
            }
          }
          resolve(last);
        })
        .catch(reject);
    }),
};