from __future__ import annotations

import time

from backend.config import settings
from backend.graph_opt import deferred_fetch, mst_skeleton, pagerank_topk
from backend.llm import ZERO, complete, complete_json
from backend.store import GraphStore


def _add(a: dict, b: dict) -> dict:
    return {"prompt": a.get("prompt", 0) + b.get("prompt", 0), "completion": a.get("completion", 0) + b.get("completion", 0), "total": a.get("total", 0) + b.get("total", 0)}


def _eval_evidence(question: str, evidence: list[dict]) -> tuple[dict, dict]:
    blob = "\n".join(e.get("text", "")[:400] for e in evidence[:8])
    if settings.GROQ_API_KEY and blob:
        data, usage = complete_json(
            f'Question: {question}\nEvidence:\n{blob}\nReturn JSON {{"sufficient":bool,"gaps":[str],"conf":0-1}}',
            system="You evaluate whether evidence can answer the question.",
        )
        if data:
            return {"sufficient": bool(data.get("sufficient")), "gaps": data.get("gaps") or [], "conf": float(data.get("conf") or 0)}, usage
    conf = min(1.0, len(evidence) / 4.0) if evidence else 0.0
    return {"sufficient": conf >= settings.EVIDENCE_CONF_THRESHOLD and len(blob) > 80, "gaps": [] if evidence else ["no evidence"], "conf": conf}, dict(ZERO)


def run_agentic(question: str, store: GraphStore) -> dict:
    t0 = time.time()
    usage = dict(ZERO)
    evidence: list[dict] = []
    trace: list[dict] = []
    visited: list[str] = []
    opt_logs: list[str] = []
    strategy = "search"
    stop = "max_steps"

    def log(step: int, tool: str, args: dict, dt: float, toks: dict, n: int, changed: bool, reason: str):
        trace.append(
            {
                "step": step,
                "tool": tool,
                "args": args,
                "duration_ms": round(dt, 1),
                "tokens": toks,
                "#chunks": n,
                "strategy_changed": changed,
                "reason": reason,
            }
        )

    for step in range(1, settings.MAX_AGENT_STEPS + 1):
        st = time.time()
        if step == 1 or (strategy == "search" and not evidence):
            hits = store.vector_search(question, k=settings.TOP_K)
            evidence.extend({"id": h["id"], "text": h["text"], "source": h.get("source", ""), "confidence": h.get("score", 0), "step": step} for h in hits)
            log(step, "similarity_search", {"k": settings.TOP_K}, (time.time() - st) * 1000, ZERO, len(hits), False, "seed chunks")
            strategy = "graph"
            continue
        if strategy == "graph":
            linked = store.link_entities(question)
            visited.extend(linked)
            nb = store.neighbors(linked, hops=settings.HOPS) if linked else {"nodes": []}
            pruned, plog = pagerank_topk(store, nb.get("nodes") or visited, settings.PAGERANK_K)
            opt_logs.append(plog)
            sk, clog = mst_skeleton(store, pruned or nb.get("nodes") or [])
            opt_logs.append(clog)
            fetched = deferred_fetch(store, [n["id"] for n in sk.get("nodes", [])] or pruned)
            for f in fetched:
                evidence.append({"id": f["id"], "text": f["text"], "source": f.get("source", ""), "confidence": 0.7, "step": step})
            log(step, "graph_traverse", {"hops": settings.HOPS, "linked": len(linked)}, (time.time() - st) * 1000, ZERO, len(fetched), True, "PageRank+MST")
            strategy = "evaluate"
            continue
        ev, u = _eval_evidence(question, evidence)
        usage = _add(usage, u)
        log(step, "evidence_evaluator", {"n": len(evidence)}, (time.time() - st) * 1000, u, len(evidence), False, str(ev.get("gaps")))
        if ev.get("sufficient"):
            stop = "evidence_sufficient"
            break
        strategy = "search"
        question = question + " " + " ".join(ev.get("gaps") or [])
        if step == settings.MAX_AGENT_STEPS:
            stop = "max_steps"

    ctx = "\n\n".join(f"[{e['id']}] {e['text']}" for e in evidence[:16])
    answer = ""
    st = time.time()
    if settings.GROQ_API_KEY and ctx:
        answer, u = complete(
            f"Answer with citations [id].\nQuestion: {question}\n\nEvidence:\n{ctx}",
            system="Synthesize a final answer from agent evidence. Cite ids.",
        )
        usage = _add(usage, u)
        log(len(trace) + 1, "answer_synthesizer", {}, (time.time() - st) * 1000, u, len(evidence), False, stop)
    if not answer:
        answer = ctx[:4000] if ctx else "No evidence."
    cites = list(dict.fromkeys(e["id"] for e in evidence))
    return {
        "answer": answer,
        "pipeline_used": "agentic",
        "tokens": usage,
        "citations": cites[:20],
        "subgraph": store.subgraph(visited[:30]) if visited else {"nodes": [], "edges": []},
        "trace": trace,
        "opt_logs": opt_logs,
        "context_tokens": sum(len((e.get("text") or "").split()) for e in evidence),
        "steps": len(trace),
        "tools_used": [t["tool"] for t in trace],
        "stop_reason": stop,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "evidence": evidence[:12],
    }
