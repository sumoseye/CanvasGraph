from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.config import RESULTS_DIR, ROOT, settings
from backend.pipelines.agentic import run_agentic
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.router import default_router
from backend.store import get_store


def _to_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(_to_str(x) for x in val)
    return str(val)


def run_hidden_evaluation(
    questions_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    store = get_store()
    
    # Resolve questions file
    q_path = Path(questions_path or (ROOT / "data" / "questions" / "eval_hidden.jsonl"))
    if not q_path.exists():
        q_path = ROOT / "data" / "questions" / "eval_public.jsonl"
    if not q_path.exists():
        q_path = ROOT / "data" / "sample_questions.json"

    out_p = Path(output_path or (RESULTS_DIR / "hidden_submission.json"))
    out_p.parent.mkdir(parents=True, exist_ok=True)

    text = q_path.read_text(encoding="utf-8").strip()
    raw = []
    if q_path.suffix.lower() == ".jsonl" or (text.startswith("{") and "\n" in text and not text.startswith("[")):
        for line in text.splitlines():
            if line.strip():
                try:
                    raw.append(json.loads(line.strip()))
                except Exception:
                    pass
    else:
        try:
            raw = json.loads(text)
        except Exception:
            raw = [json.loads(l) for l in text.splitlines() if l.strip()]

    if isinstance(raw, dict):
        items = raw.get("questions") or raw.get("items") or raw.get("data") or []
    else:
        items = raw

    runners = {"rag": run_rag, "graphrag": run_graphrag, "agentic": run_agentic}
    results = []

    print("\n" + "=" * 60)
    print(f"🚀 RUNNING HIDDEN EVALUATION ({len(items)} QUESTIONS)")
    print(f"Source: {q_path}")
    print(f"Output: {out_p}")
    print("=" * 60 + "\n")

    t0_global = time.time()
    total_tokens_all = 0
    pipe_counts = {"rag": 0, "graphrag": 0, "agentic": 0}

    for i, q in enumerate(items, 1):
        if not isinstance(q, dict):
            continue
        qid = _to_str(q.get(settings.Q_ID_FIELD) or q.get("id") or q.get("question_id") or f"hidden_{i}")
        qtext = _to_str(q.get(settings.Q_TEXT_FIELD) or q.get("question") or q.get("query") or q.get("prompt"))
        qtype = _to_str(q.get(settings.Q_TYPE_FIELD) or q.get("type") or q.get("difficulty") or "unknown")

        if not qtext:
            continue

        # Jev router prediction
        chosen_pipe, probs, rule = default_router.choose(qtext)
        runner = runners.get(chosen_pipe, run_graphrag)

        t0_q = time.time()
        out = runner(qtext, store)
        ms = round((time.time() - t0_q) * 1000, 1)

        tok = out.get("tokens") or {"prompt": 0, "completion": 0, "total": 0}
        q_tok = tok.get("total", 0)
        total_tokens_all += q_tok
        pipe_counts[chosen_pipe] = pipe_counts.get(chosen_pipe, 0) + 1

        print(
            f"[{i:02d}/{len(items)}] ID:{qid:<8} | Routed: {chosen_pipe:<8} | "
            f"Tok:{q_tok:<5} | Latency:{ms}ms | Cites:{len(out.get('citations', []))}"
        )

        results.append(
            {
                "id": qid,
                "question": qtext,
                "type": qtype,
                "routed_pipeline": chosen_pipe,
                "jev_probabilities": probs,
                "route_rule": rule,
                "answer": _to_str(out.get("answer", "")),
                "citations": out.get("citations", []),
                "tokens": tok,
                "context_tokens": out.get("context_tokens", 0),
                "steps": out.get("steps", 0),
                "tools_used": out.get("tools_used", []),
                "stop_reason": out.get("stop_reason"),
                "agent_trace": out.get("trace", []),
                "opt_logs": out.get("opt_logs", []),
                "latency_ms": ms,
            }
        )

    duration_sec = round(time.time() - t0_global, 2)
    avg_tok = round(total_tokens_all / len(results), 1) if results else 0
    avg_lat = round(duration_sec / len(results), 2) if results else 0

    submission_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.GROQ_MODEL,
        "total_questions": len(results),
        "total_tokens_burned": total_tokens_all,
        "avg_tokens_per_query": avg_tok,
        "total_latency_seconds": duration_sec,
        "avg_latency_seconds": avg_lat,
        "router_breakdown": pipe_counts,
        "results": results,
    }

    out_p.write_text(json.dumps(submission_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("✅ HIDDEN EVALUATION COMPLETE & SAVED")
    print("=" * 60)
    print(f"Total Questions Executed:  {len(results)}")
    print(f"Total Tokens Burned:      {total_tokens_all}")
    print(f"Avg Tokens / Question:    {avg_tok}")
    print(f"Avg Response Latency:     {avg_lat}s")
    print(f"Jev Router Breakdown:     RAG={pipe_counts['rag']}, GraphRAG={pipe_counts['graphrag']}, Agentic={pipe_counts['agentic']}")
    print(f"Saved Submission File:    {out_p}")
    print("=" * 60 + "\n")

    return submission_payload


def main():
    import argparse

    p = argparse.ArgumentParser(description="CanvasGraph Hidden Dataset Evaluator")
    p.add_argument("--file", default=None, help="Path to hidden questions file")
    p.add_argument("--out", default=None, help="Output JSON path")
    args = p.parse_args()

    run_hidden_evaluation(args.file, args.out)


if __name__ == "__main__":
    main()