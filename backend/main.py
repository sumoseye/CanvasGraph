import json
import queue
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import RESULTS_DIR, settings
from backend.evaluate import run_eval
from backend.hidden import run_hidden
from backend.ingest import ingest_path
from backend.pipelines.agentic import run_agentic
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.router import default_router
from backend.store import get_store

app = FastAPI(title="CanvasGraph")


class IngestBody(BaseModel):
    path: str = Field(..., description="Folder or file of txt/md/pdf/json")


class AskBody(BaseModel):
    question: str
    mode: str = "rag"


class BenchmarkBody(BaseModel):
    file: str
    pipelines: list[str] | None = None
    run: str | None = None


class HiddenBody(BaseModel):
    file: str


@app.post("/ingest")
def ingest(body: IngestBody):
    return ingest_path(body.path)


@app.post("/ask")
def ask(body: AskBody):
    store = get_store()
    mode = (body.mode or "rag").lower()
    probs = default_router.predict(body.question)
    pick, _, rule = default_router.choose(body.question)
    if mode == "auto":
        mode = pick
    if mode == "graphrag":
        out = run_graphrag(body.question, store)
    elif mode == "agentic":
        out = run_agentic(body.question, store)
    else:
        out = run_rag(body.question, store)
    out["jev_probs"] = probs
    out["route_rule"] = rule
    return out


@app.get("/stats")
def stats():
    return {"settings": {"top_k": settings.TOP_K, "hops": settings.HOPS, "backend": settings.GRAPH_BACKEND, "model": settings.GROQ_MODEL}, **get_store().stats()}


@app.post("/benchmark")
def benchmark(body: BenchmarkBody):
    q: queue.Queue = queue.Queue()

    def progress(ev: dict):
        q.put(ev)

    def work():
        try:
            run_eval(body.file, body.pipelines, body.run, progress=progress)
        except Exception as e:
            q.put({"event": "error", "error": str(e)})
        finally:
            q.put(None)

    import threading

    threading.Thread(target=work, daemon=True).start()

    def gen():
        while True:
            ev = q.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/results/{run}")
def results(run: str):
    jsonl = RESULTS_DIR / f"{run}.jsonl"
    summary = RESULTS_DIR / f"{run}_summary.json"
    if run == "summary":
        p = RESULTS_DIR / "summary.json"
        if not p.exists():
            raise HTTPException(404)
        return json.loads(p.read_text(encoding="utf-8"))
    if not jsonl.exists() and not summary.exists():
        raise HTTPException(404, f"no run {run}")
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()] if jsonl.exists() else []
    summ = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
    return {"run": run, "summary": summ, "rows": rows}


@app.post("/hidden")
def hidden(body: HiddenBody):
    return run_hidden(body.file)
