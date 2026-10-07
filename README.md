#  CanvasGraph: Agentic GraphRAG with Jev Router

> **Built for the TigerGraph Agentic GraphRAG Hackathon (2026)**  
> **Headline Question Answered:** *Which questions NEED an agent, and which don't?*

[![Medium Blog](https://img.shields.io/badge/Blog-Medium-black?logo=medium)](https://medium.com/@24b55.livona/canvasgraph-teaching-an-ai-when-not-to-think-too-hard-b975a9db3912)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Database](https://img.shields.io/badge/Graph_DB-TigerGraph_Savanna-orange)](https://www.tigergraph.com/)
[![LLM](https://img.shields.io/badge/LLM-openai%2Fgpt--oss--120b_(Groq)-purple)](https://groq.com/)

---

##  Quick Links to Submission Deliverables

*  **50 Hidden Dataset Results Payload:** [`results/hidden_submission.json`](./results/hidden_submission.json)
*  **Visible Benchmark Summary:** [`results/summary.json`](./results/summary.json)
*  **Medium Technical Article:** [CanvasGraph: Teaching an AI When Not to Think Too Hard](https://medium.com/@24b55.livona/canvasgraph-teaching-an-ai-when-not-to-think-too-hard-b975a9db3912)

---

## 💡 Overview & Key Innovations

CanvasGraph compares three pipelines over document corpora — **RAG**, **GraphRAG**, and **Agentic GraphRAG** — to prove where agentic reasoning adds value vs where it is token overkill.

### Key Features
1. ** Jev Router:** A non-generative (<150ms) pre-classifier that calculates routing probabilities across `{rag, graphrag, agentic}` before calling any LLM, reducing average token burn by **~65%** compared to "Always Agentic".
2. ** TigerGraph Savanna Integration:** Dual-writes document chunks and extracted entities to TigerGraph Cloud via `pyTigerGraph` with a persistent NetworkX local shadow-store fallback.
3. ** Temporal Conflict Resolution:** Detects contradictory/superseded facts using recency dates (`valid_from`) and `source_authority`, outputting an uncertainty score and rendering `SUPERSEDED_BY` relationships as **thick red-dashed lines** on the Canvas graph.
4. ** React Canvas & Benchmark Lab UI:** Interactive force-graph UI (`react-force-graph-2d`) with live tool traces, token meters, side-by-side answer comparisons, and 4 Recharts benchmark visualization panels.

---

## Architecture Diagram

<img width="1774" height="887" alt="archi" src="https://github.com/user-attachments/assets/b7434bfe-75bd-4606-92b7-458fd4b2ff02" />

## Project structure
CanvasGraph/
├── backend/
│   ├── main.py              # FastAPI Server (/ask, /graph, /benchmark SSE, /results)
│   ├── main_cli.py          # CLI for local testing and batch ingestion
│   ├── config.py            # Environment configuration & token limits
│   ├── llm.py               # Groq API client, disk caching, retry logic
│   ├── store.py             # GraphStore interface (TigerGraphStore + InMemoryGraphStore)
│   ├── ingest.py            # Ingestion pipeline for .txt, .pdf, .json, and .jsonl files
│   ├── graph_opt.py         # PageRank top-K pruning & MST skeleton graph algorithms
│   ├── router.py            # Jev Router pre-classifier
│   ├── tools.py             # Agent tools (conflict_checker, aggregator, evaluator, etc.)
│   ├── evaluate.py         # Benchmark evaluator & LLM-as-a-Judge script
│   └── hidden.py           # Hidden dataset evaluation runner
├── frontend/
│   ├── src/
│   │   ├── components/      # ForceGraph (2D canvas), Sidebar, TraceViewer
│   │   ├── pages/           # Canvas.tsx (Playground), Lab.tsx (Benchmark Dashboard)
│   │   ├── api.ts           # Central API & SSE streaming client
│   │   └── types.ts         # Shared TypeScript interfaces
├── data/
│   ├── corpus/              # Ingested corpus documents
│   └── questions/           # Public & Hidden evaluation questions
├── results/
│   ├── hidden_submission.json # 📍 RAW OUTPUTS FOR 50 HIDDEN QUESTIONS
│   └── summary.json         # Benchmark summary metrics
├── .env.example
├── requirements.txt
└── README.md

## Quick Start & Setup Instructions
# Clone the repository
git clone https://github.com/sumoseye/CanvasGraph.git
cd CanvasGraph

# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt
2. Environment Configuration (.env)
Create a .env file in the root directory:

env

GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1

GRAPH_BACKEND=tigergraph
TG_HOST=https://your-workspace.i.tgcloud.com
TG_USER=tigergraph
TG_PASSWORD=your_password
TG_GRAPH=CanvasGraph

TOP_K=3
HOPS=2
PAGERANK_K=8
MAX_AGENT_STEPS=3
3. Ingest Corpus & Run Benchmarks
Bash

# Ingest official corpus into TigerGraph + Shadow Store
python -m backend.main_cli ingest data/corpus/corpus.jsonl

# Run 100 visible public questions evaluation
python -m backend.evaluate data/questions/eval_public.jsonl --all --fast --sequential

# Generate 50 hidden questions submission JSON payload
python -m backend.hidden --file data/questions/eval_hidden.jsonl --out results/hidden_submission.json
4. Run Frontend Development Server
Bash

# Start backend API (Terminal 1)
.venv/bin/uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend UI (Terminal 2)
cd frontend
npm install
npm run dev
Open http://localhost:5173 to view the Canvas Graph Playground and Benchmark Lab!

 Key Benchmark Results
Metric	RAG	GraphRAG	Agentic GraphRAG	Jev Routed
Accuracy %	50.0%	66.7%	66.7%	66.7%
Avg Tokens / Query	692	657	1,851	669
Avg Latency	2.5s	2.8s	7.7s	3.9s
*Key Takeaway*: GraphRAG achieved a +16.7% accuracy boost over RAG at lower token cost. While Agentic GraphRAG matched GraphRAG's accuracy, it burned 3x tokens. The Jev Router successfully preserved peak 66.7% accuracy while reducing token burn by ~64% compared to "Always Agentic".

