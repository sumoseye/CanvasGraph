from __future__ import annotations

import time

from backend.config import settings
from backend.graph_opt import deferred_fetch, mst_skeleton, pagerank_topk
from backend.llm import ZERO, complete, estimate_tokens
from backend.store import GraphStore


def run_graphrag(
    question: str,
    store: GraphStore,
    hops: int | None = None,
    k: int | None = None,
) -> dict:
    t0 = time.time()
    hops = hops or settings.HOPS
    k = k or settings.PAGERANK_K
    opt_logs: list[str] = []

    # 1. Entity linking
    linked = store.link_entities(question) or []
    opt_logs.append(f"linked {len(linked)} entities: {linked}")

    # 2. Neighbors traversal
    nb = store.neighbors(linked, hops=hops) if linked else {"nodes": [], "edges": []}
    nodes = nb.get("nodes") or []

    # If entity linking yields no graph nodes, fall back to vector search seed
    if not nodes:
        hits = store.vector_search(question, k=settings.TOP_K) or []
        seed = [h["id"] for h in hits]
        if hasattr(store, "g"):
            for hid in list(seed):
                try:
                    for _, tgt, ed in store.g.out_edges(hid, data=True):
                        if ed.get("rel") == "MENTIONS":
                            seed.append(tgt)
                except Exception:
                    pass
        nb = store.neighbors(seed, hops=hops) if seed else {"nodes": [], "edges": []}
        nodes = nb.get("nodes") or []
        opt_logs.append(f"fallback vector seed -> {len(nodes)} graph nodes")

    opt_logs.append(f"traverse hops={hops} -> {len(nodes)} nodes")

    # 3. PageRank top-K pruning
    pruned, plog = pagerank_topk(store, nodes, k)
    opt_logs.append(plog)

    # 4. MST skeleton graph
    skeleton, clog = mst_skeleton(store, pruned or nodes)
    opt_logs.append(clog)

    # Extract winning node IDs
    win_nodes = skeleton.get("nodes") or []
    win_ids = [n["id"] if isinstance(n, dict) else n for n in win_nodes] or pruned or nodes

    # 5. Deferred text fetch
    fetched = deferred_fetch(store, win_ids) or []

    # If graph traversal returned no text chunks, fall back directly to vector search chunks
    if not any(f.get("text") for f in fetched):
        hits = store.vector_search(question, k=settings.TOP_K) or []
        fetched = [{"id": h["id"], "text": h.get("text", ""), "source": h.get("source", ""), "ntype": "chunk"} for h in hits]
        opt_logs.append(f"fallback fetched {len(fetched)} chunks from vector search")
    else:
        opt_logs.append(f"deferred fetch {len(fetched)} texts")

    # 6. Context packing (hard limit: CONTEXT_CHAR_LIMIT chars, max 4 chunks)
    ctx_parts: list[str] = []
    used: list[dict] = []
    budget = settings.CONTEXT_CHAR_LIMIT

    for f in fetched:
        text = (f.get("text") or "").strip()
        if not text:
            continue
        text_clip = text[:500]
        piece = f"[{f['id']}] {text_clip}"
        if used and len(piece) > budget:
            break
        ctx_parts.append(piece)
        used.append({**f, "text": text_clip})
        budget -= len(piece)
        if len(used) >= 4:
            break

    ctx = "\n\n".join(ctx_parts)
    opt_logs.append(f"context built: {len(used)} chunks, {len(ctx)} chars")

    usage = dict(ZERO)
    answer = ""
    if settings.GROQ_API_KEY and ctx:
        prompt = (
            f"Answer using graph evidence. Cite ids like [c_xxx]. Be concise.\n"
            f"Question: {question}\n\nEvidence:\n{ctx}"
        )
        answer, usage = complete(
            prompt,
            system="You are a GraphRAG answerer. Prefer recent high-authority facts. Cite ids.",
            max_tokens=settings.ANSWER_MAX_TOKENS,
        )

    if not answer:
        answer = ctx[:2000] if ctx else "No graph evidence found."

    citations = [f["id"] for f in used if f.get("id")]
    return {
        "answer": answer,
        "pipeline_used": "graphrag",
        "tokens": usage,
        "citations": citations[:20],
        "subgraph": skeleton if skeleton.get("nodes") else {"nodes": [], "edges": []},
        "trace": [
            {
                "step": 1,
                "tool": "entity_linker",
                "args": {"n": len(linked)},
                "tokens": dict(ZERO),
                "#chunks": 0,
            },
            {
                "step": 2,
                "tool": "graph_traverse",
                "args": {"hops": hops, "pruned": len(pruned or [])},
                "tokens": dict(ZERO),
                "#chunks": len(used),
            },
            {
                "step": 3,
                "tool": "doc_retriever",
                "args": {"sent": len(used)},
                "tokens": dict(ZERO),
                "#chunks": len(used),
            },
        ],
        "opt_logs": opt_logs,
        "context_tokens": estimate_tokens(ctx),
        "steps": 3,
        "tools_used": ["entity_linker", "graph_traverse", "doc_retriever"],
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }