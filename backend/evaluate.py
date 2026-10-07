from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from backend.config import RESULTS_DIR, ROOT, settings
from backend.ingest import ingest_path
from backend.llm import ZERO, complete_json
from backend.pipelines.agentic import run_agentic
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.router import default_router
from backend.store import get_store

RUNNERS = {"rag": run_rag, "graphrag": run_graphrag, "agentic": run_agentic}
COST = {"rag": 1, "graphrag": 2, "agentic": 3}
JUDGE_SYS = (
    "You are a strict grader. Given question, ground truth, and model answer, "
    'return JSON {"verdict":"PASS"|"FAIL","completeness":0-1,"reason":"..."}. '
    "PASS if the answer contains the key facts of the ground truth (conflicts may mention both sides)."
)


def load_questions(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("questions") or raw.get("items") or []
    out = []
    for i, q in enumerate(raw):
        out.append(
            {
                "id": str(q.get(settings.Q_ID_FIELD) or q.get("id") or f"q{i}"),
                "question": q.get(settings.Q_TEXT_FIELD) or q.get("question") or q.get("query") or "",
                "ground_truth": q.get(settings.Q_ANSWER_FIELD) or q.get("answer") or q.get("ground_truth") or "",
                "type": q.get(settings.Q_TYPE_FIELD) or q.get("type") or q.get("difficulty") or "unknown",
            }
        )
    return out


def _done_keys(jsonl: Path) -> set[tuple[str, str]]:
    done = set()
    if not jsonl.exists():
        return done
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((str(r.get("id")), r.get("pipeline")))
    return done


def _load_rows(jsonl: Path) -> list[dict]:
    if not jsonl.exists():
        return []
    return [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]


def heuristic_judge(gt: str, answer: str) -> tuple[dict, dict]:
    stop = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "was", "as", "with"}
    gt_t = [w for w in re.findall(r"[a-z0-9]+", (gt or "").lower()) if w not in stop and len(w) > 1]
    ans = (answer or "").lower()
    if not gt_t:
        return {"verdict": "PASS", "completeness": 1.0, "reason": "empty gt"}, dict(ZERO)
    hits = sum(1 for w in dict.fromkeys(gt_t) if w in ans)
    uniq = len(dict.fromkeys(gt_t))
    comp = hits / uniq
    return {"verdict": "PASS" if comp >= 0.45 else "FAIL", "completeness": round(comp, 3), "reason": "token overlap"}, dict(ZERO)


def judge(question: str, gt: str, answer: str) -> tuple[dict, dict]:
    if settings.GROQ_API_KEY:
        data, usage = complete_json(
            f"Question: {question}\nGround truth: {gt}\nAnswer: {answer[:6000]}",
            system=JUDGE_SYS,
        )
        if data.get("verdict") in ("PASS", "FAIL"):
            try:
                comp = float(data.get("completeness", 0))
            except (TypeError, ValueError):
                comp = 0.0
            return {"verdict": data["verdict"], "completeness": max(0.0, min(1.0, comp)), "reason": data.get("reason", "")}, usage
    return heuristic_judge(gt, answer)


def _row(q: dict, pipe: str, out: dict, j: dict, ju: dict, ms: float) -> dict:
    tok = out.get("tokens") or ZERO
    return {
        "id": q["id"],
        "question": q["question"],
        "type": q["type"],
        "pipeline": pipe,
        "answer": out.get("answer", ""),
        "correct": j["verdict"] == "PASS",
        "verdict": j["verdict"],
        "completeness": j["completeness"],
        "reason": j.get("reason", ""),
        "context_tokens": out.get("context_tokens", 0),
        "llm_input_tokens": tok.get("prompt", 0) + ju.get("prompt", 0),
        "llm_output_tokens": tok.get("completion", 0) + ju.get("completion", 0),
        "total_tokens": tok.get("total", 0) + ju.get("total", 0),
        "latency_ms": out.get("latency_ms") or ms,
        "steps": out.get("steps", 0),
        "tools_used": out.get("tools_used") or [],
        "citations": len(out.get("citations") or []),
        "opt_logs": out.get("opt_logs") or [],
        "trace": out.get("trace") or [],
        "stop_reason": out.get("stop_reason"),
        "jev_probs": None,
    }


def _agg(rows: list[dict]) -> dict:
    n = len(rows) or 1
    acc = sum(1 for r in rows if r.get("correct")) / n
    avg_tok = sum(r.get("total_tokens", 0) for r in rows) / n
    return {
        "n": len(rows),
        "accuracy": round(acc, 4),
        "avg_tokens": round(avg_tok, 2),
        "accuracy_per_1k_tokens": round(acc / (avg_tok / 1000.0), 4) if avg_tok else 0.0,
        "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in rows) / n, 1),
        "avg_completeness": round(sum(r.get("completeness", 0) for r in rows) / n, 4),
    }


def summarize(rows: list[dict]) -> dict:
    by_pipe: dict[str, list] = {}
    for r in rows:
        by_pipe.setdefault(r["pipeline"], []).append(r)
    types = sorted({r["type"] for r in rows})
    breakdown = {}
    for t in types:
        breakdown[t] = {}
        for p, rs in by_pipe.items():
            sub = [r for r in rs if r["type"] == t]
            if sub:
                breakdown[t][p] = _agg(sub)
    where = {}
    for t in types:
        accs = {p: breakdown[t].get(p, {}).get("accuracy", 0) for p in ("rag", "graphrag", "agentic") if p in breakdown[t]}
        if not accs:
            continue
        best = max(accs, key=accs.get)
        if accs.get("agentic", 0) > max(accs.get("rag", 0), accs.get("graphrag", 0)):
            where[t] = "agentic_wins"
        elif accs.get("agentic", 0) <= accs.get("rag", 0) and "agentic" in accs:
            where[t] = "agentic_overkill"
        else:
            where[t] = f"best={best}"

    qids = sorted({r["id"] for r in rows if r["pipeline"] != "routed"})
    route_hits, route_n = 0, 0
    for qid in qids:
        group = [r for r in rows if r["id"] == qid and r["pipeline"] in COST]
        if not group:
            continue
        routed = next((r for r in rows if r["id"] == qid and r["pipeline"] == "routed"), None)
        if not routed:
            continue
        correct = [r for r in group if r.get("correct")]
        cheapest = min(correct, key=lambda r: (r.get("total_tokens", 0), COST.get(r["pipeline"], 9))) if correct else None
        route_n += 1
        picked = routed.get("routed_pipeline")
        if cheapest and picked == cheapest["pipeline"]:
            route_hits += 1
        elif not cheapest and picked:
            pass
    return {
        "pipelines": {p: _agg(rs) for p, rs in by_pipe.items()},
        "breakdown_by_type": breakdown,
        "when_agents_matter": where,
        "jev_routing_accuracy": round(route_hits / route_n, 4) if route_n else 0.0,
        "routed_vs_always_agentic": {
            "routed": _agg(by_pipe.get("routed") or []),
            "always_agentic": _agg(by_pipe.get("agentic") or []),
        },
    }


def _ensure_store():
    st = get_store()
    if st.stats().get("chunks", 0) == 0:
        ingest_path(ROOT / "data" / "sample")
        return get_store()
    return st


def _run_one(q: dict, pipe: str, store) -> dict:
    t0 = time.time()
    out = RUNNERS[pipe](q["question"], store)
    ms = round((time.time() - t0) * 1000, 1)
    j, ju = judge(q["question"], q["ground_truth"], out.get("answer", ""))
    return _row(q, pipe, out, j, ju, ms)


def _append(jsonl: Path, row: dict):
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def run_eval(
    questions_path: str | Path,
    pipelines: list[str] | None = None,
    run_id: str | None = None,
    progress=None,
) -> dict:
    pipelines = pipelines or ["rag", "graphrag", "agentic"]
    print(f"LLM: groq / {settings.GROQ_MODEL}")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    jsonl = RESULTS_DIR / f"{run_id}.jsonl"
    jsonl.touch(exist_ok=True)
    questions = load_questions(questions_path)
    store = _ensure_store()
    done = _done_keys(jsonl)
    jobs = [(q, p) for q in questions for p in pipelines if (q["id"], p) not in done and p in RUNNERS]
    if progress:
        progress({"event": "start", "run": run_id, "jobs": len(jobs), "questions": len(questions)})

    def emit(row):
        _append(jsonl, row)
        if progress:
            progress({"event": "row", "id": row["id"], "pipeline": row["pipeline"], "correct": row["correct"]})

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_run_one, q, p, store): (q, p) for q, p in jobs}
        for fut in as_completed(futs):
            emit(fut.result())

    rows = _load_rows(jsonl)
    by_q: dict[str, list] = {}
    for r in rows:
        if r["pipeline"] in RUNNERS:
            by_q.setdefault(r["id"], []).append(r)
    qmap = {q["id"]: q for q in questions}
    for q in questions:
        if (q["id"], "routed") in _done_keys(jsonl):
            continue
        group = by_q.get(q["id"]) or []
        if not group:
            continue
        pick, probs, rule = default_router.choose(q["question"])
        chosen = next((r for r in group if r["pipeline"] == pick), None)
        if rule == "graphrag_then_escalate":
            g = next((r for r in group if r["pipeline"] == "graphrag"), None)
            a = next((r for r in group if r["pipeline"] == "agentic"), None)
            if g is not None and not g.get("correct") and a is not None:
                chosen, pick = a, "agentic"
            elif g is not None:
                chosen, pick = g, "graphrag"
        if chosen is None:
            chosen = min(group, key=lambda r: COST.get(r["pipeline"], 9))
            pick = chosen["pipeline"]
        routed = dict(chosen)
        routed["pipeline"] = "routed"
        routed["routed_pipeline"] = pick
        routed["jev_probs"] = probs
        routed["route_rule"] = rule
        emit(routed)

    rows = _load_rows(jsonl)
    summary = {"run": run_id, "file": str(questions_path), "pipelines": pipelines, **summarize(rows)}
    (RESULTS_DIR / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if progress:
        progress({"event": "done", "run": run_id, "summary": summary})
    return summary


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--pipelines", default="rag,graphrag,agentic")
    p.add_argument("--run", default=None)
    args = p.parse_args()
    s = run_eval(args.file, [x.strip() for x in args.pipelines.split(",") if x.strip()], args.run)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
