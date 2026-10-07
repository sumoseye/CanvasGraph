from __future__ import annotations

from backend.config import settings
from backend.llm import ZERO, complete
from backend.store import GraphStore


def _extractive(question: str, hits: list[dict]) -> str:
    if not hits:
        return "No evidence in the corpus."
    bits = "\n\n".join(f"[{h['id']}] {h['text'][:500]}" for h in hits)
    return f"Evidence for: {question}\n\n{bits}"


def run_rag(question: str, store: GraphStore, k: int | None = None) -> dict:
    k = k or settings.TOP_K
    hits = store.vector_search(question, k=k)
    ctx_hits = hits[:3]
    ctx = "\n\n".join(f"[{h['id']}] source={h.get('source','')}\n{h['text'][:600]}" for h in ctx_hits)
    usage = dict(ZERO)
    answer = ""
    if settings.GROQ_API_KEY and hits:
        prompt = (
            f"Answer using only the context. Cite chunk ids like [c_xxx].\n"
            f"Question: {question}\n\nContext:\n{ctx}"
        )
        answer, usage = complete(prompt, system="You are a precise GraphRAG answerer. Cite sources.", max_tokens=300)
    if not answer:
        answer = _extractive(question, hits)
    citations = [h["id"] for h in hits]
    nodes = [{"id": h["id"], "ntype": "chunk", "text": h["text"][:200]} for h in hits]
    return {
        "answer": answer,
        "pipeline_used": "rag",
        "tokens": usage,
        "citations": citations,
        "subgraph": {"nodes": nodes, "edges": []},
        "trace": [{"step": 1, "tool": "similarity_search", "args": {"k": k}, "tokens": ZERO, "#chunks": len(hits)}],
        "opt_logs": [f"top-{k} chunks from TF-IDF"],
        "context_tokens": sum(len(h["text"].split()) for h in hits),
        "steps": 1,
        "tools_used": ["similarity_search"],
    }
