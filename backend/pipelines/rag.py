from __future__ import annotations

from backend.config import settings
from backend.llm import ZERO, complete, estimate_tokens
from backend.store import GraphStore


def _extractive(question: str, hits: list[dict]) -> str:
    if not hits:
        return "No evidence in the corpus."
    bits = "\n\n".join(f"[{h['id']}] {h['text'][:400]}" for h in hits)
    return f"Evidence for: {question}\n\n{bits}"


def run_rag(question: str, store: GraphStore, k: int | None = None) -> dict:
    k = k or settings.TOP_K
    hits = store.vector_search(question, k=k)

    # only the chunks we actually send to the LLM
    ctx_hits: list[dict] = []
    budget = settings.CONTEXT_CHAR_LIMIT
    for h in hits:
        text = (h.get("text") or "")[:600]
        piece = f"[{h['id']}] source={h.get('source', '')}\n{text}"
        if ctx_hits and len(piece) > budget:
            break
        ctx_hits.append({**h, "text": text})
        budget -= len(piece)
        if len(ctx_hits) >= 3:
            break

    ctx = "\n\n".join(
        f"[{h['id']}] source={h.get('source', '')}\n{h['text']}" for h in ctx_hits
    )
    usage = dict(ZERO)
    answer = ""
    if settings.GROQ_API_KEY and ctx_hits:
        prompt = (
            f"Answer using only the context. Cite chunk ids like [c_xxx].\n"
            f"Question: {question}\n\nContext:\n{ctx}"
        )
        answer, usage = complete(
            prompt,
            system="You are a precise RAG answerer. Cite sources. Be concise.",
            max_tokens=settings.ANSWER_MAX_TOKENS,
        )
    if not answer:
        answer = _extractive(question, ctx_hits)

    citations = [h["id"] for h in ctx_hits]  # only what was sent
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
        "opt_logs": [f"top-{k} retrieved, {len(ctx_hits)} sent ({settings.CONTEXT_CHAR_LIMIT} char cap)"],
        "context_tokens": estimate_tokens(ctx),
        "steps": 1,
        "tools_used": ["similarity_search"],
        "latency_ms": 0,
    }