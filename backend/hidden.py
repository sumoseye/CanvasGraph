from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backend.config import RESULTS_DIR, ROOT
from backend.evaluate import load_questions, _ensure_store
from backend.pipelines.agentic import run_agentic
from backend.router import default_router


def run_hidden(questions_path: str | Path, out_path: str | Path | None = None) -> dict:
    store = _ensure_store()
    questions = load_questions(questions_path)
    items = []
    for q in questions:
        out = run_agentic(q["question"], store)
        pick, probs, rule = default_router.choose(q["question"])
        items.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": out.get("answer", ""),
                "tokens": out.get("tokens"),
                "pipeline_used": out.get("pipeline_used", "agentic"),
                "jev_probs": probs,
                "route_rule": rule,
                "routed_would_pick": pick,
                "agentic_trace": out.get("trace") or [],
                "stop_reason": out.get("stop_reason"),
                "citations": out.get("citations") or [],
            }
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file": str(questions_path),
        "n": len(items),
        "questions": items,
    }
    dest = Path(out_path) if out_path else RESULTS_DIR / "hidden_submission.json"
    dest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?", default=str(ROOT / "data" / "sample_questions.json"))
    args = p.parse_args()
    run_hidden(args.file)
    print(str(RESULTS_DIR / "hidden_submission.json"))


if __name__ == "__main__":
    main()
