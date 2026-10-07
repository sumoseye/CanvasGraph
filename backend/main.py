import json
import queue
from pathlib import Path
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import RESULTS_DIR, ROOT, settings
from backend.evaluate import run_eval
from backend.hidden import run_hidden_evaluation as run_hidden
from backend.ingest import ingest_path
from backend.pipelines.agentic import run_agentic
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.router import default_router
from backend.store import get_store

app = FastAPI(title="CanvasGraph")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.middleware("http")
async def strip_api_prefix(request, call_next):
    if request.url.path.startswith("/api/"):
        request.scope["path"] = request.url.path[4:]
    elif request.url.path == "/api":
        request.scope["path"] = "/"
    return await call_next(request)


class IngestBody(BaseModel):
    path: str = Field(..., description="Folder or file of txt/md/pdf/json")


class IngestTextBody(BaseModel):
    text: str
    title: str = "pasted"


class IngestUrlBody(BaseModel):
    url: str


class AskBody(BaseModel):
    question: str
    mode: str = "auto"


class BenchmarkBody(BaseModel):
    file: str = "data/questions/eval_public.jsonl"
    pipelines: list[str] | None = None
    run: str | None = None
    limit: int | None = 6
    fast: bool = True


class HiddenBody(BaseModel):
    file: str = "data/questions/eval_hidden.jsonl"


@app.get("/health")
def health():
    return {"ok": True, "model": settings.GROQ_MODEL, "backend": settings.GRAPH_BACKEND}


@app.post("/ingest")
def ingest(body: IngestBody):
    return ingest_path(body.path)


@app.post("/ingest/text")
def ingest_text(body: IngestTextBody):
    d = ROOT / "data" / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in body.title)[:40] or "pasted"
    p = d / f"{safe}.txt"
    p.write_text(body.text, encoding="utf-8")
    return ingest_path(str(p))


@app.post("/ingest/url")
def ingest_url(body: IngestUrlBody):
    try:
        req = Request(body.url, headers={"User-Agent": "CanvasGraph/1.0"})
        with urlopen(req, timeout=20) as resp:
            raw = resp.read()
        text = raw.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(400, f"fetch failed: {e}") from e
    d = ROOT / "data" / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    name = body.url.rstrip("/").split("/")[-1][:40] or "url"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    if not safe.endswith((".txt", ".md", ".html", ".htm")):
        safe += ".txt"
    p = d / safe
    p.write_text(text, encoding="utf-8")
    return ingest_path(str(p))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    d = ROOT / "data" / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / (file.filename or "upload.txt")
    dest.write_bytes(await file.read())
    return ingest_path(str(dest))


@app.post("/ask")
def ask(body: AskBody):
    store = get_store()
    mode = (body.mode or "auto").lower()
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
    out["mode_requested"] = body.mode
    out["mode_used"] = mode
    return out


@app.get("/stats")
def stats():
    return {
        "settings": {
            "top_k": settings.TOP_K,
            "hops": settings.HOPS,
            "backend": settings.GRAPH_BACKEND,
            "model": settings.GROQ_MODEL,
        },
        **get_store().stats(),
    }


@app.get("/graph")
def graph(limit: int = 250):
    store = get_store()
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = set()

    g = getattr(store, "g", None)
    if g is None and hasattr(store, "shadow"):
        g = getattr(store.shadow, "g", None)

    if g is not None:
        ranked = list(g.nodes(data=True))
        for nid, data in ranked[:limit]:
            ntype = data.get("ntype") or data.get("type") or "Entity"
            label = data.get("name") or data.get("label") or str(nid)
            text = (data.get("text") or "")[:160]
            nodes.append(
                {
                    "id": str(nid),
                    "ntype": ntype,
                    "label": label[:48],
                    "text": text,
                    "source": data.get("source", ""),
                }
            )
            seen.add(str(nid))
        for u, v, ed in g.edges(data=True):
            if str(u) in seen and str(v) in seen:
                edges.append(
                    {
                        "source": str(u),
                        "target": str(v),
                        "rel": ed.get("rel") or ed.get("type") or "RELATED",
                    }
                )
            if len(edges) >= limit * 2:
                break
    else:
        st = store.stats()
        nodes = [{"id": "empty", "ntype": "Document", "label": f"{st.get('chunks', 0)} chunks", "text": ""}]

    return {"nodes": nodes, "edges": edges, "n_nodes": len(nodes), "n_edges": len(edges)}


@app.post("/benchmark")
def benchmark(body: BenchmarkBody):
    q: queue.Queue = queue.Queue()

    # Resolve dataset file path
    target_file = Path(body.file)
    if not target_file.exists():
        if (ROOT / body.file).exists():
            target_file = ROOT / body.file
        elif (ROOT / "data" / "questions" / "eval_public.jsonl").exists():
            target_file = ROOT / "data" / "questions" / "eval_public.jsonl"
        elif (ROOT / "data" / "sample_questions.json").exists():
            target_file = ROOT / "data" / "sample_questions.json"

    def progress(ev: dict):
        q.put(ev)

    def work():
        try:
            run_eval(
                target_file,
                body.pipelines,
                body.run,
                progress=progress,
                limit=body.limit,
                sequential=True,
                fast=body.fast,
            )
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
    if run == "summary" or run == "latest":
        p = RESULTS_DIR / "summary.json"
        if not p.exists():
            raise HTTPException(404, "no summary yet — run a benchmark first")
        summ = json.loads(p.read_text(encoding="utf-8"))
        run_id = summ.get("run", "")
        rows = []
        if run_id and (RESULTS_DIR / f"{run_id}.jsonl").exists():
            rows = [
                json.loads(l)
                for l in (RESULTS_DIR / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
        return {"run": run_id or "summary", "summary": summ, "rows": rows}
    if not jsonl.exists() and not summary.exists():
        raise HTTPException(404, f"no run {run}")
    rows = (
        [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if jsonl.exists()
        else []
    )
    summ = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
    return {"run": run, "summary": summ, "rows": rows}


@app.get("/results")
def list_results():
    files = sorted(RESULTS_DIR.glob("*_summary.json"), reverse=True)
    out = []
    for f in files[:20]:
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
            out.append({"run": s.get("run") or f.stem.replace("_summary", ""), "file": f.name})
        except Exception:
            out.append({"run": f.stem, "file": f.name})
    return {"runs": out}


@app.post("/hidden")
def hidden(body: HiddenBody):
    return run_hidden(body.file)