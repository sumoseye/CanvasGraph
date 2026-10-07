from __future__ import annotations

import argparse
import json

from backend.ingest import ingest_path
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.store import get_store


def main():
    p = argparse.ArgumentParser(prog="canvasgraph")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("path")
    pa = sub.add_parser("ask")
    pa.add_argument("question")
    pa.add_argument("--mode", choices=["rag", "graphrag"], default="rag")
    sub.add_parser("stats")
    ps = sub.add_parser("serve")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    if args.cmd == "ingest":
        print(json.dumps(ingest_path(args.path), indent=2, default=str))
        return
    if args.cmd == "stats":
        print(json.dumps(get_store().stats(), indent=2))
        return
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=False)
        return
    store = get_store()
    fn = run_graphrag if args.mode == "graphrag" else run_rag
    print(json.dumps(fn(args.question, store), indent=2, default=str))


if __name__ == "__main__":
    main()
