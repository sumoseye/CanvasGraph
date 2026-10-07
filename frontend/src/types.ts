export type Pipeline = "rag" | "graphrag" | "agentic" | "auto" | "routed";

export type GraphNode = {
  id: string;
  ntype?: string;
  label?: string;
  text?: string;
  source?: string;
  x?: number;
  y?: number;
};

export type GraphEdge = {
  source: string | GraphNode;
  target: string | GraphNode;
  rel?: string;
};

export type GraphPayload = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  n_nodes?: number;
  n_edges?: number;
};

export type TraceStep = {
  step: number;
  tool: string;
  args?: Record<string, unknown>;
  duration_ms?: number;
  tokens?: { prompt?: number; completion?: number; total?: number };
  "#chunks"?: number;
  strategy_changed?: boolean;
  reason?: string;
};

export type AskResult = {
  answer: string;
  pipeline_used: string;
  tokens: { prompt: number; completion: number; total: number };
  citations: string[];
  subgraph?: GraphPayload;
  trace?: TraceStep[];
  opt_logs?: string[];
  context_tokens?: number;
  steps?: number;
  tools_used?: string[];
  stop_reason?: string;
  latency_ms?: number;
  jev_probs?: Record<string, number>;
  route_rule?: string;
  mode_requested?: string;
  mode_used?: string;
  evidence?: Array<{ id: string; text: string; source?: string }>;
};

export type Stats = {
  settings?: { top_k?: number; hops?: number; backend?: string; model?: string };
  backend?: string;
  nodes?: number;
  edges?: number;
  chunks?: number;
  ntypes?: Record<string, number>;
  [k: string]: unknown;
};

export type EvalRow = {
  id: string;
  question: string;
  type: string;
  pipeline: string;
  answer: string;
  correct: boolean;
  verdict?: string;
  completeness?: number;
  reason?: string;
  total_tokens?: number;
  latency_ms?: number;
  steps?: number;
  tools_used?: string[];
  citations?: number;
  trace?: TraceStep[];
  opt_logs?: string[];
  stop_reason?: string;
  jev_probs?: Record<string, number> | null;
  routed_pipeline?: string;
  route_rule?: string;
};

export type PipeAgg = {
  n: number;
  accuracy: number;
  avg_tokens: number;
  accuracy_per_1k_tokens?: number;
  avg_latency_ms?: number;
  avg_completeness?: number;
};

export type Summary = {
  run?: string;
  pipelines?: Record<string, PipeAgg>;
  breakdown_by_type?: Record<string, Record<string, PipeAgg>>;
  when_agents_matter?: Record<string, string>;
  jev_routing_accuracy?: number;
  routed_vs_always_agentic?: {
    routed?: PipeAgg;
    always_agentic?: PipeAgg;
  };
  limit?: number;
  fast_judge?: boolean;
};

export type ResultsPayload = {
  run: string;
  summary: Summary;
  rows: EvalRow[];
};

export type Session = {
  id: string;
  question: string;
  mode: string;
  result: AskResult;
  ts: number;
};