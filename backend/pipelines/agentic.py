from __future__ import annotations

import time

from backend.config import settings
from backend.llm import ZERO, complete_json, estimate_tokens
from backend.store import GraphStore
from backend.tools import TOOL_NAMES, run_tool

ORCHESTRATOR_SYS = (
    "You are an agentic GraphRAG orchestrator. Pick ONE next tool as JSON:\n"
    '{"tool":"<name>","args":{},"reason":"<short>","done":false}\n'
    f"Tools: {', '.join(TOOL_NAMES)}.\n"
    "Strategy: start with similarity_search or graph_traverse; "
    "use conflict_checker for opposing facts; aggregator for counts/lists; "
    "evidence_evaluator before finishing; answer_synthesizer when ready. "
    'Set "done":true only with tool=answer_synthesizer. Stay minimal.'
)


def _add(a: dict, b: dict) -> dict:
    return {
        "prompt": a.get("prompt", 0) + b.get("prompt", 0),
        "completion": a.get("completion", 0) + b.get("completion", 0),
        "total": a.get("total", 0) + b.get("total", 0),
    }


def _merge_hits(evidence: list[dict], hits: list[dict]) -> list[dict]:
    seen = {e.get("id") for e in evidence}
    for h in hits or []:
        hid = h.get("id")
        if hid and hid not in seen:
            evidence.append(h)
            seen.add(hid)
    return evidence


def _plan(question: str, step: int, evidence: list[dict], last_tool: str, gaps: list[str]) -> tuple[dict, dict]:
    """LLM orchestrator — falls back to a deterministic plan if JSON fails / no key."""
    summary = f"n_evidence={len(evidence)} last_tool={last_tool or 'none'} gaps={gaps[:3]}"
    prompt = (
        f"Question: {question}\nStep: {step}/{settings.MAX_AGENT_STEPS}\nState: {summary}\n"
        "Pick the single best next tool."
    )
    if settings.GROQ_API_KEY:
        data, usage = complete_json(
            prompt,
            system=ORCHESTRATOR_SYS,
            max_tokens=settings.JSON_MAX_TOKENS,
        )
        if data.get("tool") in TOOL_NAMES:
            return data, usage

    # deterministic fallback (still produces a real multi-step trace)
    if step == 1 and not evidence:
        return {"tool": "similarity_search", "args": {}, "reason": "seed", "done": False}, dict(ZERO)
    if step == 1 or last_tool == "similarity_search":
        return {"tool": "graph_traverse", "args": {}, "reason": "expand", "done": False}, dict(ZERO)
    if last_tool not in ("evidence_evaluator", "answer_synthesizer") and step < settings.MAX_AGENT_STEPS:
        return {"tool": "evidence_evaluator", "args": {}, "reason": "check", "done": False}, dict(ZERO)
    return {"tool": "answer_synthesizer", "args": {}, "reason": "finalize", "done": True}, dict(ZERO)


def run_agentic(question: str, store: GraphStore) -> dict:
    t0 = time.time()
    usage = dict(ZERO)
    evidence: list[dict] = []
    trace: list[dict] = []
    opt_logs: list[str] = []
    visited: list[str] = []
    tools_used: list[str] = []
    stop = "max_steps"
    last_tool = ""
    gaps: list[str] = []
    strategy = "plan"
    answer = ""
    skeleton = {"nodes": [], "edges": []}
    original_q = question

    for step in range(1, settings.MAX_AGENT_STEPS + 1):
        # 1) orchestrator decides
        plan, pu = _plan(original_q, step, evidence, last_tool, gaps)
        usage = _add(usage, pu)
        tool = plan.get("tool") or "answer_synthesizer"
        args = plan.get("args") if isinstance(plan.get("args"), dict) else {}
        reason = str(plan.get("reason") or "")
        done_flag = bool(plan.get("done"))
        changed = tool != last_tool and last_tool != ""
        if changed:
            strategy = tool

        trace.append(
            {
                "step": step,
                "tool": "orchestrator",
                "args": {"pick": tool, **args},
                "duration_ms": 0,
                "tokens": pu,
                "#chunks": len(evidence),
                "strategy_changed": changed,
                "reason": reason,
            }
        )

        # 2) execute tool
        # inject current evidence for tools that accept it
        call_kwargs = dict(args)
        call_kwargs["evidence"] = evidence
        if visited:
            call_kwargs.setdefault("entities", visited)

        result = run_tool(tool, original_q, store, **call_kwargs)
        usage = _add(usage, result.tokens)
        tools_used.append(tool)
        trace.append(
            result.as_trace(
                step=step,
                args={"tool": tool, **args},
                strategy_changed=changed,
                reason=reason or result.error,
            )
        )
        last_tool = tool

        data = result.data or {}
        if "opt_logs" in data:
            opt_logs.extend(data.get("opt_logs") or [])
        if "skeleton" in data and data["skeleton"]:
            skeleton = data["skeleton"]
        if "entities" in data and data["entities"]:
            visited.extend(x for x in data["entities"] if x not in visited)

        hits = data.get("hits") or []
        if hits:
            _merge_hits(evidence, hits)

        if tool == "evidence_evaluator":
            if data.get("sufficient"):
                stop = "evidence_sufficient"
                # final synthesize on next loop or immediately if last step
                if step >= settings.MAX_AGENT_STEPS:
                    syn = run_tool("answer_synthesizer", original_q, store, evidence=evidence)
                    usage = _add(usage, syn.tokens)
                    tools_used.append("answer_synthesizer")
                    trace.append(syn.as_trace(step, {"forced": True}, False, stop))
                    answer = (syn.data or {}).get("answer") or ""
                    evidence = (syn.data or {}).get("used") or evidence
                    break
            else:
                gaps = list(data.get("gaps") or gaps)
                if gaps:
                    # light gap injection for next retrieval
                    original_q_gap = original_q  # keep original for final answer
                    question = f"{original_q} {' '.join(gaps[:3])}"
                    del original_q_gap

        if tool == "answer_synthesizer" or done_flag:
            answer = (data.get("answer") if tool == "answer_synthesizer" else "") or answer
            if tool == "answer_synthesizer":
                evidence = data.get("used") or evidence
            stop = "agent_done" if done_flag else stop
            break

        if step >= settings.MAX_AGENT_STEPS:
            stop = "max_steps"

    # guarantee an answer
    if not answer:
        syn = run_tool("answer_synthesizer", original_q, store, evidence=evidence)
        usage = _add(usage, syn.tokens)
        tools_used.append("answer_synthesizer")
        trace.append(syn.as_trace(len(trace) + 1, {"forced_final": True}, False, stop))
        answer = (syn.data or {}).get("answer") or ""
        evidence = (syn.data or {}).get("used") or evidence
        if not stop:
            stop = "forced_synthesize"

    # context string for token estimate / citations
    ctx_parts = []
    budget = settings.CONTEXT_CHAR_LIMIT
    used_ids = []
    for e in evidence:
        text = (e.get("text") or "")[:500]
        piece = f"[{e.get('id')}] {text}"
        if used_ids and len(piece) > budget:
            break
        ctx_parts.append(piece)
        used_ids.append(e.get("id"))
        budget -= len(piece)
        if len(used_ids) >= 3:
            break
    ctx = "\n\n".join(ctx_parts)

    if not answer:
        answer = ctx[:2000] if ctx else "No evidence."

    sub = skeleton if skeleton.get("nodes") else (
        store.subgraph(visited[:30]) if visited else {"nodes": [], "edges": []}
    )

    return {
        "answer": answer,
        "pipeline_used": "agentic",
        "tokens": usage,
        "citations": [i for i in used_ids if i][:20],
        "subgraph": sub,
        "trace": trace,
        "opt_logs": opt_logs,
        "context_tokens": estimate_tokens(ctx),
        "steps": len(trace),
        "tools_used": tools_used,
        "stop_reason": stop,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "evidence": evidence[:8],
        "strategy": strategy,
    }