from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backend.config import RESULTS_DIR, settings
from backend.pipelines.agentic import run_agentic
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.router import default_router
from backend.store import get_store


def run_hidden_evaluation(
    questions_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """
    Evaluates hidden/unseen questions using the Jev Router to pick the optimal pipeline,
    logging detailed agentic traces, token usage, and citations for hackathon submission.
    """
    store = get_store()
    q_path = Path(questions_path or (RESULTS_DIR.parent / "data" / "hidden_questions.json"))
    out_p = Path(output_path or (RESULTS_DIR / "hidden_submission.json"))

    if not q_path.exists():
        # Fallback to visible sample questions if no hidden dataset file is provided
        q_path = RESULTS_DIR.parent / "data" / "sample_questions.json"

    raw = json.loads(q_path.read_text(encoding="utf-8"))
    items = raw.get("questions") or raw.get("items") or (raw if isinstance(raw, list) else [])

    runners = {"rag": run_rag, "graphrag": run_graphrag, "agentic": run_agentic}
    results = []

    print(f"Running hidden submission pipeline over {len(items)} questions...")

    for i, q in enumerate(items):
        qid = str(q.get(settings.Q_ID_FIELD) or q.get("id") or f"hidden_{i}")
        qtext = q.get(settings.Q_TEXT_FIELD) or q.get("question") or q.get("query") or ""
        qtype = q.get(settings.Q_TYPE_FIELD) or q.get("type") or "unknown"

        # Jev router prediction
        chosen_pipe, probs, rule = default_router.choose(qtext)
        runner = runners.get(chosen_pipe, run_graphrag)

        out = runner(qtext, store)

        results.append(
            {
                "id": qid,
                "question": qtext,
                "type": qtype,
                "routed_pipeline": chosen_pipe,
                "jev_probabilities": probs,
                "route_rule": rule,
                "answer": out.get("answer", ""),
                "citations": out.get("citations", []),
                "tokens": out.get("tokens", {}),
                "context_tokens": out.get("context_tokens", 0),
                "steps": out.get("steps", 0),
                "tools_used": out.get("tools_used", []),
                "stop_reason": out.get("stop_reason"),
                "agent_trace": out.get("trace", []),
                "opt_logs": out.get("opt_logs", []),
                "latency_ms": out.get("latency_ms", 0),
            }
        )
        print(f"  [{qid}] Jev routed to {chosen_pipe} -> {len(out.get('citations', []))} citations")

    submission_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.GROQ_MODEL,
        "total_questions": len(results),
        "results": results,
    }

    out_p.write_text(json.dumps(submission_payload, indent=2), encoding="utf-8")
    print(f"\nSaved hidden submission to: {out_p}")
    return submission_payload


def main():
    import argparse

    p = argparse.ArgumentParser(description="CanvasGraph Hidden Dataset Evaluator")
    p.add_argument("--file", default=None, help="Path to hidden questions JSON file")
    p.add_argument("--out", default=None, help="Output JSON path")
    args = p.parse_args()

    run_hidden_evaluation(args.file, args.out)


if __name__ == "__main__":
    main()