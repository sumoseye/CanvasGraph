from __future__ import annotations

from backend.config import settings
from backend.graph_opt import deferred_fetch, mst_skeleton, pagerank_topk
from backend.llm import ZERO, complete
from backend.store import GraphStore


def run_graphrag(question: str, store: GraphStore, hops: int | None = None, k: int | None = None) -> dict:
    hops = hops or settings.HOPS
    k = k or settings.PAGERANK_K
    opt_logs = []
    linked = store.link_entities(question)
    opt_logs.append(f"linked {len(linked)} entities")
    nb = store.neighbors(linked, hops=hops) if linked else {"nodes": [], "edges": []}
    if not nb["nodes"]:
        hits = store.vector_search(question, k=settings.TOP_K)
        seed = [h["id"] for h in hits]
        if hasattr(store, "g"):
            for hid in list(seed):
                for _, t, ed in store.g.out_edges(hid, data=True):
                    if ed.get("rel") == "MENTIONS":
                        seed.append(t)
        nb = store.neighbors(seed, hops=hops)
        opt_logs.append("fallback seed from vector search")
    opt_logs.append(f"traverse hops={hops} -> {len(nb['nodes'])} nodes")
    pruned, log = pagerank_topk(store, nb["nodes"], k)
    opt_logs.append(log)
    skeleton, clog = mst_skeleton(store, pruned or nb["nodes"])
    opt_logs.append(clog)
    win_ids = [n["id"] for n in skeleton["nodes"]] or pruned
    fetched = deferred_fetch(store, win_ids)
    opt_logs.append(f"deferred fetch {len(fetched)} texts")
    ctx = "\n\n".join(f"[{f['id']}] {f['text']}" for f in fetched[:16])
    usage = dict(ZERO)
    answer = ""
    if settings.GROQ_API_KEY and ctx:
        prompt = (
            f"Answer using graph evidence. Cite ids like [c_xxx].\nQuestion: {question}\n\nEvidence:\n{ctx}"
        )
        answer, usage = complete(prompt, system="You are a GraphRAG answerer. Prefer recent high-authority facts.")
    if not answer:
        answer = ctx[:4000] if ctx else "No graph evidence found."
    citations = [f["id"] for f in fetched if f.get("ntype") == "chunk"] or [f["id"] for f in fetched]
    return {
        "answer": answer,
        "pipeline_used": "graphrag",
        "tokens": usage,
        "citations": citations[:20],
        "subgraph": skeleton,
        "trace": [
            {"step": 1, "tool": "entity_linker", "args": {"n": len(linked)}, "tokens": ZERO, "#chunks": 0},
            {"step": 2, "tool": "graph_traverse", "args": {"hops": hops}, "tokens": ZERO, "#chunks": len(fetched)},
        ],
        "opt_logs": opt_logs,
        "context_tokens": sum(len(f["text"].split()) for f in fetched),
        "steps": 2,
        "tools_used": ["entity_linker", "graph_traverse", "doc_retriever"],
    }
