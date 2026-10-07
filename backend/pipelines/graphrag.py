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

    linked = store.link_entities(question)
    opt_logs.append(f"linked {len(linked)} entities")

    nb = store.neighbors(linked, hops=hops) if linked else {"nodes": [], "edges": []}
    if not nb.get("nodes"):
        hits = store.vector_search(question, k=settings.TOP_K)
        seed = [h["id"] for h in hits]
        # follow MENTIONS edges into entities if NetworkX store exposes .g
        if hasattr(store, "g"):
            for hid in list(seed):
                try:
                    for _, t, ed in store.g.out_edges(hid, data=True):
                        if ed.get("rel") == "MENTIONS":
                            seed.append(t)
                except Exception:
                    pass
        nb = store.neighbors(seed, hops=hops) if seed else {"nodes": [], "edges": []}
        opt_logs.append("fallback seed from vector search")

    opt_logs.append(f"traverse hops={hops} -> {len(nb.get('nodes') or [])} nodes")

    pruned, plog = pagerank_topk(store, nb.get("nodes") or [], k)
    opt_logs.append(plog)
    skeleton, clog = mst_skeleton(store, pruned or nb.get("nodes") or [])
    opt_logs.append(clog)

    win_ids = [n["id"] for n in skeleton.get("nodes", [])] or pruned
    fetched = deferred_fetch(store, win_ids)
    opt_logs.append(f"deferred fetch {len(fetched)} texts")

    # --- hard context cap: max 3 chunks, CONTEXT_CHAR_LIMIT chars ---
    ctx_parts: list[str] = []
    used: list[dict] = []
    budget = settings.CONTEXT_CHAR_LIMIT
    for f in fetched:
        text = (f.get("text") or "")[:500]
        piece = f"[{f['id']}] {text}"
        if used and len(piece) > budget:
            break
        ctx_parts.append(piece)
        used.append({**f, "text": text})
        budget -= len(piece)
        if len(used) >= 3:
            break

    ctx = "\n\n".join(ctx_parts)
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

    citations = [f["id"] for f in used]
    return {
        "answer": answer,
        "pipeline_used": "graphrag",
        "tokens": usage,
        "citations": citations[:20],
        "subgraph": skeleton,
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