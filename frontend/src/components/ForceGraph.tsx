import { useEffect, useMemo, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { GraphEdge, GraphNode } from "../types";

const COLORS: Record<string, string> = {
  Person: "#2B6CB0",
  People: "#2B6CB0",
  Entity: "#5A67D8",
  Entities: "#5A67D8",
  Org: "#5A67D8",
  Document: "#D69E2E",
  Documents: "#D69E2E",
  chunk: "#D69E2E",
  Tech: "#2F855A",
  Attribute: "#4A5568",
  Attributes: "#4A5568",
  Community: "#9F7AEA",
};

type Props = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  highlightIds?: Set<string>;
  height?: number | string;
  onNodeClick?: (n: GraphNode) => void;
};

export default function ForceGraph({
  nodes,
  edges,
  highlightIds,
  height = "100%",
  onNodeClick,
}: Props) {
  const ref = useRef<any>(null);
  
  const data = useMemo(() => {
    const ns = nodes.map((n) => ({ ...n }));
    const idSet = new Set(ns.map((n) => n.id));
    const es = edges
      .map((e) => ({
        source: typeof e.source === "object" ? (e.source as GraphNode).id : String(e.source),
        target: typeof e.target === "object" ? (e.target as GraphNode).id : String(e.target),
        rel: e.rel,
      }))
      .filter((e) => idSet.has(e.source) && idSet.has(e.target));
    return { nodes: ns, links: es };
  }, [nodes, edges]);

  useEffect(() => {
    const fg = ref.current;
    if (!fg) return;
    fg.d3Force("charge")?.strength(-120);
    fg.d3Force("link")?.distance(50);
  }, [data]);

  return (
    <div style={{ height, width: "100%" }} className="dot-grid">
      <ForceGraph2D
        ref={ref}
        graphData={data}
        backgroundColor="rgba(0,0,0,0)"
        nodeId="id"
        
        // Red color for SUPERSEDED_BY, faint blue for standard relationships
        linkColor={(l: any) => l.rel === "SUPERSEDED_BY" ? "#ef4444" : "rgba(148,163,184,0.25)"}
        
        // Make SUPERSEDED_BY dashed
        linkLineDash={(l: any) => l.rel === "SUPERSEDED_BY" ? [4, 4] : []}
        
        linkWidth={(l: any) => {
          if (l.rel === "SUPERSEDED_BY") return 2;
          const isHi = highlightIds && highlightIds.has(String(l.source.id || l.source)) && highlightIds.has(String(l.target.id || l.target));
          return isHi ? 2.2 : 0.6;
        }}
        
        nodeCanvasObject={(node: any, ctx, globalScale) => {
          const id = String(node.id);
          const hi = highlightIds?.has(id);
          const color = COLORS[node.ntype || "Entity"] || "#5A67D8";
          const r = hi ? 7 : 4;
          
          ctx.beginPath();
          ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
          ctx.fillStyle = color;
          ctx.globalAlpha = hi || !highlightIds?.size ? 1 : 0.25;
          ctx.fill();
          
          if (hi) {
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1.5 / globalScale;
            ctx.stroke();
          }
          
          const label = node.label || id;
          if (globalScale > 1.4 || hi) {
            ctx.font = `${11 / globalScale}px IBM Plex Sans`;
            ctx.fillStyle = "#cbd5e1";
            ctx.globalAlpha = 0.9;
            ctx.fillText(String(label).slice(0, 28), node.x + r + 2, node.y + 3);
          }
          ctx.globalAlpha = 1;
        }}
        onNodeClick={(n: any) => onNodeClick?.(n as GraphNode)}
        cooldownTicks={80}
      />
    </div>
  );
}