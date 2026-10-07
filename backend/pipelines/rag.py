from __future__ import annotations

from backend.config import settings
from backend.llm import ZERO, complete, estimate_tokens
from backend.store import GraphStore


def run_rag(question: str, store: GraphStore, k: int | None = None) -> dict:
    k = k or settings.TOP_K + 2  # fetch slightly more to cover conflicting dates
    hits = store.vector_search(question, k=k)

    ctx_hits: list[dict] = []
    budget = settings.CONTEXT_CHAR_LIMIT
    for h in hits:
        text = (h.get("text") or "")[:600]
        piece = f"[{h['id']}] source={h.get('source', '')}\n{text}"
        if ctx_hits and len(piece) > budget:
            break
        ctx_hits.append({**h, "text": text})
        budget -= len(piece)
        if len(ctx_hits) >= 4:
            break

    ctx = "\n\n".join(
        f"[{h['id']}] source={h.get('source', '')}\n{h['text']}" for h in ctx_hits
    )
    usage = dict(ZERO)
    answer = ""
    if settings.GROQ_API_KEY and ctx_hits:
        prompt = (
            f"Answer the question accurately using ONLY the context below.\n"
            f"If context contains conflicting facts (e.g. 2023 vs 2025), ALWAYS prefer the most recent (2025) press release/audited fact.\n"
            f"Cite chunk ids like [c_xxx]. Be direct and state the key facts clearly.\n\n"
            f"Question: {question}\n\nContext:\n{ctx}"
        )
        answer, usage = complete(
            prompt,
            system="You are a precise GraphRAG answerer. Prefer recent (2025) facts when conflicts exist. Cite sources.",
            max_tokens=settings.ANSWER_MAX_TOKENS,
        )

    if not answer:
        answer = "\n\n".join(f"[{h['id']}] {h['text'][:400]}" for h in ctx_hits) if ctx_hits else "No evidence."

    citations = [h["id"] for h in ctx_hits]
    nodes = [{"id": h["id"], "ntype": "chunk", "text": h["text"][:200]} for h in ctx_hits]
    return {
        "answer": answer,
        "pipeline_used": "rag",
        "tokens": usage,
        "citations": citations,
        "subgraph": {"nodes": nodes, "edges": []},
        "trace": [
            {
                "step": 1,
                "tool": "similarity_search",
                "args": {"k": k, "used": len(ctx_hits)},
                "tokens": dict(ZERO),
                "#chunks": len(ctx_hits),
            }
        ],
        "opt_logs": [f"top-{k} retrieved, {len(ctx_hits)} sent"],
        "context_tokens": estimate_tokens(ctx),
        "steps": 1,
        "tools_used": ["similarity_search"],
        "latency_ms": 0,
    }