from __future__ import annotations

import time
from typing import Any, Callable
from datetime import datetime

from backend.config import settings
from backend.graph_opt import deferred_fetch, mst_skeleton, pagerank_topk
from backend.llm import ZERO, complete, complete_json, estimate_tokens
from backend.store import GraphStore


def _clip(text: str, n: int = 500) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n]


class ToolResult:
    __slots__ = ("name", "ok", "data", "chunks", "tokens", "duration_ms", "error")

    def __init__(
        self,
        name: str,
        ok: bool,
        data: Any,
        chunks: int = 0,
        tokens: dict | None = None,
        duration_ms: float = 0.0,
        error: str = "",
    ):
        self.name = name
        self.ok = ok
        self.data = data
        self.chunks = chunks
        self.tokens = tokens or dict(ZERO)
        self.duration_ms = duration_ms
        self.error = error

    def as_trace(self, step: int, args: dict, strategy_changed: bool = False, reason: str = "") -> dict:
        return {
            "step": step,
            "tool": self.name,
            "args": args,
            "duration_ms": round(self.duration_ms, 1),
            "tokens": self.tokens,
            "#chunks": self.chunks,
            "strategy_changed": strategy_changed,
            "reason": reason or self.error or ("ok" if self.ok else "fail"),
        }


def entity_linker(question: str, store: GraphStore, **_kw) -> ToolResult:
    t0 = time.time()
    linked = store.link_entities(question) or []
    return ToolResult(
        "entity_linker", True, {"entities": linked, "n": len(linked)},
        chunks=0, duration_ms=(time.time() - t0) * 1000,
    )


def similarity_search(question: str, store: GraphStore, k: int | None = None, **_kw) -> ToolResult:
    t0 = time.time()
    k = k or (settings.TOP_K + 2)
    hits = store.vector_search(question, k=k) or []
    rows = [
        {
            "id": h["id"],
            "text": _clip(h.get("text", "")),
            "source": h.get("source", ""),
            "score": h.get("score", 0),
        }
        for h in hits
    ]
    return ToolResult(
        "similarity_search", True, {"hits": rows},
        chunks=len(rows), duration_ms=(time.time() - t0) * 1000,
    )


def graph_traverse(question: str, store: GraphStore, entities: list[str] | None = None, hops: int | None = None, **_kw) -> ToolResult:
    t0 = time.time()
    hops = hops or settings.HOPS
    linked = list(entities or []) or (store.link_entities(question) or [])
    nb = store.neighbors(linked, hops=hops) if linked else {"nodes": [], "edges": []}
    if not nb.get("nodes"):
        hits = store.vector_search(question, k=settings.TOP_K + 2) or []
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

    pruned, plog = pagerank_topk(store, nb.get("nodes") or [], settings.PAGERANK_K)
    skeleton, slog = mst_skeleton(store, pruned or nb.get("nodes") or [])
    win_ids = [n["id"] if isinstance(n, dict) else n for n in skeleton.get("nodes", [])] or pruned
    fetched = deferred_fetch(store, win_ids) or []
    rows = [
        {
            "id": f["id"], "text": _clip(f.get("text", "")),
            "source": f.get("source", ""), "ntype": f.get("ntype", "chunk"),
        }
        for f in fetched
    ]
    return ToolResult(
        "graph_traverse", True,
        {"entities": linked, "pruned": pruned, "skeleton": skeleton, "hits": rows, "opt_logs": [plog, slog, f"fetched {len(rows)}"]},
        chunks=len(rows), duration_ms=(time.time() - t0) * 1000,
    )


def doc_retriever(question: str, store: GraphStore, ids: list[str] | None = None, **_kw) -> ToolResult:
    t0 = time.time()
    if ids:
        fetched = deferred_fetch(store, ids) or []
    else:
        hits = store.vector_search(question, k=settings.TOP_K + 2) or []
        fetched = hits
    rows = [{"id": f.get("id"), "text": _clip(f.get("text", "")), "source": f.get("source", "")} for f in fetched]
    return ToolResult("doc_retriever", True, {"hits": rows}, chunks=len(rows), duration_ms=(time.time() - t0) * 1000)


def aggregator(question: str, store: GraphStore, evidence: list[dict] | None = None, **_kw) -> ToolResult:
    t0 = time.time()
    ev = list(evidence or [])
    all_chunks = []
    if hasattr(store, "chunks") and isinstance(store.chunks, dict):
        all_chunks = [{"id": cid, "text": _clip(c.get("text", ""), 400), "source": c.get("source", "")} for cid, c in store.chunks.items()]
    if not all_chunks:
        hits = store.vector_search(question, k=8) or []
        all_chunks = [{"id": h["id"], "text": _clip(h.get("text", ""), 400), "source": h.get("source", "")} for h in hits]

    seen = {e.get("id") for e in ev}
    for c in all_chunks:
        if c.get("id") not in seen:
            ev.append(c)
            seen.add(c.get("id"))

    from collections import Counter
    import re
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", " ".join(e.get("text", "") for e in ev))
    stop = {"the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "have", "has", "been", "their", "they", "which", "will", "into", "about"}
    cnt = Counter(w for w in words if w.lower() not in stop)
    top = cnt.most_common(10)
    
    return ToolResult("aggregator", True, {"hits": ev, "top_terms": top, "n_docs": len(ev)}, chunks=len(ev), duration_ms=(time.time() - t0) * 1000)


def community_summary(question: str, store: GraphStore, **_kw) -> ToolResult:
    t0 = time.time()
    try:
        comms = store.communities() if hasattr(store, "communities") else []
    except Exception:
        comms = []
    rows = []
    if comms:
        for c in comms[:3]:
            cid = c.get("id") or c.get("community") or "comm"
            members = c.get("members") or c.get("nodes") or []
            rows.append({"id": str(cid), "text": _clip(f"Community {cid} members={len(members)}: {', '.join(map(str, members[:12]))}"), "source": "community"})
    if not rows:
        hits = store.vector_search(question, k=4) or []
        rows = [{"id": h["id"], "text": _clip(h.get("text", "")), "source": h.get("source", "")} for h in hits]
    return ToolResult("community_summary", True, {"hits": rows, "n_communities": len(comms) if comms else 0}, chunks=len(rows), duration_ms=(time.time() - t0) * 1000)


def conflict_checker(question: str, store: GraphStore, evidence: list[dict] | None = None, **_kw) -> ToolResult:
    t0 = time.time()
    ev = list(evidence or [])
    hits = store.vector_search(question, k=8) or []
    
    # Enrich chunks with store metadata (dates, authority)
    seen = {e.get("id") for e in ev}
    for h in hits:
        if h.get("id") not in seen:
            ev.append(h)
            seen.add(h.get("id"))
            
    for e in ev:
        if "valid_from" not in e and hasattr(store, "chunks") and e.get("id") in store.chunks:
            e["valid_from"] = store.chunks[e["id"]].get("valid_from", "")
            e["authority"] = store.chunks[e["id"]].get("source_authority", 0.5)

    cues = ("however", "but", "contrary", "previously", "no longer", "updated", "retracted", "instead", "superseded", "2023", "2024", "2025")
    flagged = [e for e in ev if any(c in (e.get("text") or "").lower() for c in cues)]
    
    tokens = dict(ZERO)
    uncertainty = 0.0
    synth_edges = []
    
    if len(flagged) >= 2 and settings.GROQ_API_KEY:
        blob = "\n".join(f"[{e.get('id')}] Date:{e.get('valid_from','none')} Auth:{e.get('authority', 0.5)} Text:{_clip(e.get('text', ''), 280)}" for e in ev[:6])
        data, tokens = complete_json(
            f"Question: {question}\nEvidence:\n{blob}\n"
            'Return JSON {"conflicts":[{"old_id":"c_...","new_id":"c_...","reason":"str"}],"uncertainty":0.0-1.0}',
            system="Analyze facts. If a newer date or higher authority source contradicts an older one, mark the older as superseded. Uncertainty is 1.0 if unresolved, 0.0 if clearly superseded. Strict JSON.",
            max_tokens=settings.JSON_MAX_TOKENS,
        )
        if data:
            uncertainty = float(data.get("uncertainty", 0.0))
            for c in data.get("conflicts", []):
                oid, nid = c.get("old_id"), c.get("new_id")
                if oid and nid:
                    synth_edges.append({"src": oid, "dst": nid, "rel": "SUPERSEDED_BY", "reason": c.get("reason")})

    report = {
        "conflicts_found": len(synth_edges) > 0,
        "uncertainty": round(uncertainty, 2),
        "superseded_edges": synth_edges,
        "note": "Weighed by recency & authority." if synth_edges else "No explicit conflicts resolved."
    }

    # Inject virtual SUPERSEDED_BY edges into the skeleton for UI rendering
    skeleton_injection = {"nodes": [{"id": e.get("id"), "label": "Chunk", "ntype": "chunk"} for e in ev[:6]], "edges": synth_edges}

    return ToolResult(
        "conflict_checker", True, 
        {"report": report, "hits": ev, "skeleton": skeleton_injection},
        chunks=len(ev), tokens=tokens, duration_ms=(time.time() - t0) * 1000
    )


def evidence_evaluator(question: str, store: GraphStore, evidence: list[dict] | None = None, **_kw) -> ToolResult:
    t0 = time.time()
    ev = evidence or []
    blob = "\n".join(f"[{e.get('id')}] {_clip(e.get('text', ''), 320)}" for e in ev[:5])
    conf = min(1.0, len(ev) / 3.0) if ev else 0.0
    sufficient = conf >= settings.EVIDENCE_CONF_THRESHOLD and len(blob) > 80
    tokens = dict(ZERO)
    gaps: list[str] = [] if sufficient else (["need more evidence"] if not ev else [])

    if not sufficient and settings.GROQ_API_KEY and blob:
        data, tokens = complete_json(
            f"Question: {question}\nEvidence:\n{blob}\nReturn JSON {{\"sufficient\":bool,\"gaps\":[str],\"conf\":0-1}}",
            system="Does the evidence answer the question? Strict, concise JSON.",
            max_tokens=settings.JSON_MAX_TOKENS,
        )
        if data:
            sufficient = bool(data.get("sufficient"))
            gaps = data.get("gaps") or gaps
            try: conf = float(data.get("conf") or conf)
            except (TypeError, ValueError): pass

    return ToolResult("evidence_evaluator", True, {"sufficient": sufficient, "gaps": gaps, "conf": conf, "n": len(ev)}, chunks=len(ev), tokens=tokens, duration_ms=(time.time() - t0) * 1000)


def answer_synthesizer(question: str, store: GraphStore, evidence: list[dict] | None = None, **_kw) -> ToolResult:
    t0 = time.time()
    ev = evidence or []
    parts, used = [], []
    budget = settings.CONTEXT_CHAR_LIMIT
    for e in ev:
        text = _clip(e.get("text", ""), 500)
        piece = f"[{e.get('id', '?')}] source={e.get('source', '')}\n{text}"
        if used and len(piece) > budget: break
        parts.append(piece)
        used.append(e)
        budget -= len(piece)
        if len(used) >= 5: break
    ctx = "\n\n".join(parts)
    tokens = dict(ZERO)
    answer = ""
    if settings.GROQ_API_KEY and ctx:
        answer, tokens = complete(
            f"Answer the question using evidence below.\nIf conflicting facts exist, ALWAYS prefer the highest authority / most recent.\nCite chunk ids like [c_xxx].\n\nQuestion: {question}\n\nEvidence:\n{ctx}",
            system="Synthesize a final answer from evidence. Cite ids. Be complete and concise.",
            max_tokens=settings.ANSWER_MAX_TOKENS,
        )
    if not answer:
        answer = ctx[:2000] if ctx else "No evidence."
    return ToolResult("answer_synthesizer", True, {"answer": answer, "used": used, "ctx": ctx}, chunks=len(used), tokens=tokens, duration_ms=(time.time() - t0) * 1000)


TOOLS: dict[str, Callable[..., ToolResult]] = {
    "entity_linker": entity_linker, "similarity_search": similarity_search,
    "graph_traverse": graph_traverse, "doc_retriever": doc_retriever,
    "aggregator": aggregator, "community_summary": community_summary,
    "conflict_checker": conflict_checker, "evidence_evaluator": evidence_evaluator,
    "answer_synthesizer": answer_synthesizer,
}

TOOL_NAMES = list(TOOLS.keys())

def run_tool(name: str, question: str, store: GraphStore, **kwargs) -> ToolResult:
    fn = TOOLS.get(name)
    if fn is None: return ToolResult(name, False, {}, error=f"unknown tool {name}")
    try: return fn(question=question, store=store, **kwargs)
    except Exception as e: return ToolResult(name, False, {}, error=str(e))