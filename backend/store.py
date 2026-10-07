from __future__ import annotations

import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import STORE_PATH, settings


def _safe_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(_safe_str(x) for x in v)
    return str(v)


class GraphStore(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: list[dict]) -> None: ...
    @abstractmethod
    def upsert_entities(self, entities: list[dict]) -> None: ...
    @abstractmethod
    def upsert_relations(self, relations: list[dict]) -> None: ...
    @abstractmethod
    def link_entities(self, question: str) -> list[str]: ...
    @abstractmethod
    def neighbors(self, ids: list[str], hops: int = 1, rel_filter: list[str] | None = None) -> dict: ...
    @abstractmethod
    def subgraph(self, ids: list[str]) -> dict: ...
    @abstractmethod
    def get_text(self, ids: list[str]) -> dict[str, str]: ...
    @abstractmethod
    def vector_search(self, q: str, k: int = 5) -> list[dict]: ...
    @abstractmethod
    def communities(self) -> dict: ...
    @abstractmethod
    def stats(self) -> dict: ...


class InMemoryGraphStore(GraphStore):
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.vectorizer = TfidfVectorizer(max_features=8192, ngram_range=(1, 2))
        self._chunk_ids: list[str] = []
        self._matrix = None
        self._fitted = False
        self.chunks: dict[str, dict] = {}

    def save(self, path: Path = STORE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path = STORE_PATH) -> InMemoryGraphStore | None:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _refit(self) -> None:
        self._chunk_ids = [n for n, d in self.g.nodes(data=True) if d.get("ntype") == "chunk"]
        texts = [_safe_str(self.g.nodes[i].get("text", "")) for i in self._chunk_ids]
        if texts:
            self._matrix = self.vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            self._matrix = None
            self._fitted = False

    def upsert_chunks(self, chunks: list[dict]) -> None:
        for c in chunks:
            cid = _safe_str(c["id"])
            data = {**self.g.nodes[cid]} if cid in self.g else {}
            data.update(
                ntype="chunk",
                text=_safe_str(c.get("text", "")),
                source=_safe_str(c.get("source", "")),
                source_authority=float(c.get("source_authority", 0.5)),
                valid_from=_safe_str(c.get("valid_from", "")),
                title=_safe_str(c.get("title", cid)),
            )
            self.g.add_node(cid, **data)
            self.chunks[cid] = data
        self._refit()
        self.save()

    def upsert_entities(self, entities: list[dict]) -> None:
        for e in entities:
            eid = _safe_str(e["id"])
            data = {**self.g.nodes[eid]} if eid in self.g else {}
            ename = _safe_str(e.get("name") or eid)
            data.update(
                ntype=_safe_str(e.get("type", "Entity")),
                name=ename,
                text=ename,
            )
            self.g.add_node(eid, **data)
            for cid in e.get("chunk_ids", []):
                cid_str = _safe_str(cid)
                if cid_str in self.g:
                    self.g.add_edge(cid_str, eid, rel="MENTIONS")
                    self.g.add_edge(eid, cid_str, rel="MENTIONED_IN")
        self.save()

    def upsert_relations(self, relations: list[dict]) -> None:
        for r in relations:
            src, dst = _safe_str(r["src"]), _safe_str(r["dst"])
            if src not in self.g or dst not in self.g:
                continue
            rel = _safe_str(r.get("rel", "RELATED"))
            vf = _safe_str(r.get("valid_from", ""))
            key = f"{src}|{rel}|{dst}|{vf}"
            self.g.add_edge(
                src,
                dst,
                key=key,
                rel=rel,
                valid_from=vf,
                source=_safe_str(r.get("source", "")),
                source_authority=float(r.get("source_authority", 0.5)),
                fact_id=_safe_str(r.get("id", key)),
            )
        self.save()

    def link_entities(self, question: str) -> list[str]:
        q = _safe_str(question).lower()
        scored = []
        for n, d in self.g.nodes(data=True):
            if d.get("ntype") in ("chunk", None) and "name" not in d:
                continue
            if d.get("ntype") == "chunk":
                continue
            name = _safe_str(d.get("name") or n).lower()
            if name and name in q:
                scored.append((3.0 + len(name.split()), n))
                continue
            tokens = set(re.findall(r"[a-z0-9]+", name))
            qtoks = set(re.findall(r"[a-z0-9]+", q))
            if tokens and tokens <= qtoks:
                scored.append((2.0 + len(tokens), n))
        scored.sort(reverse=True)
        ids = [n for _, n in scored]
        if ids:
            return ids[:12]
        hits = self.vector_search(question, k=4)
        linked = []
        for h in hits:
            for _, t, ed in self.g.out_edges(h["id"], data=True):
                if ed.get("rel") == "MENTIONS":
                    linked.append(t)
        return list(dict.fromkeys(linked))[:12]

    def neighbors(self, ids: list[str], hops: int = 1, rel_filter: list[str] | None = None) -> dict:
        seen = set(ids)
        frontier = set(ids)
        edges = []
        for _ in range(max(1, hops)):
            nxt = set()
            for u in frontier:
                if u not in self.g:
                    continue
                for _, v, ed in self.g.out_edges(u, data=True):
                    rel = ed.get("rel", "")
                    if rel_filter and rel not in rel_filter:
                        continue
                    edges.append({"src": u, "dst": v, "rel": rel, **{k: ed[k] for k in ed if k != "rel"}})
                    if v not in seen:
                        nxt.add(v)
                        seen.add(v)
            frontier = nxt
        return {"nodes": list(seen), "edges": edges}

    def subgraph(self, ids: list[str]) -> dict:
        valid_ids = [i for i in ids if i in self.g]
        sg = self.g.subgraph(valid_ids)
        nodes = [{"id": n, **{k: v for k, v in d.items() if k != "text"}, "text": _safe_str(d.get("text") or "")[:240]} for n, d in sg.nodes(data=True)]
        edges = [{"src": u, "dst": v, "rel": d.get("rel", "")} for u, v, d in sg.edges(data=True)]
        return {"nodes": nodes, "edges": edges}

    def get_text(self, ids: list[str]) -> dict[str, str]:
        out = {}
        for i in ids:
            if i in self.g:
                out[i] = _safe_str(self.g.nodes[i].get("text") or self.g.nodes[i].get("name") or "")
        return out

    def vector_search(self, q: str, k: int = 5) -> list[dict]:
        if not self._fitted or self._matrix is None or not self._chunk_ids:
            return []
        qv = self.vectorizer.transform([_safe_str(q)])
        sims = cosine_similarity(qv, self._matrix).ravel()
        idx = np.argsort(-sims)[:k]
        hits = []
        for i in idx:
            cid = self._chunk_ids[int(i)]
            d = self.g.nodes[cid]
            hits.append({"id": cid, "score": float(sims[i]), "text": _safe_str(d.get("text", "")), "source": _safe_str(d.get("source", ""))})
        return hits

    def communities(self) -> dict:
        und = self.g.to_undirected()
        from networkx.algorithms import community as nx_comm

        try:
            comms = list(nx_comm.louvain_communities(und, seed=7))
        except Exception:
            comms = list(nx_comm.greedy_modularity_communities(und)) if und.number_of_nodes() else []
        out = []
        for i, c in enumerate(comms):
            names = [_safe_str(self.g.nodes[n].get("name") or n) for n in c if self.g.nodes[n].get("ntype") != "chunk"]
            out.append({"id": f"c{i}", "nodes": list(c), "summary": ", ".join(names[:8]) or f"community {i}"})
        return {"communities": out}

    def stats(self) -> dict:
        ntypes: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            t = d.get("ntype", "unknown")
            ntypes[t] = ntypes.get(t, 0) + 1
        return {
            "backend": "memory",
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "chunks": ntypes.get("chunk", 0),
            "ntypes": ntypes,
        }


class TigerGraphStore(GraphStore):
    def __init__(self):
        self.shadow = InMemoryGraphStore.load() or InMemoryGraphStore()
        self.conn = None
        self.connected = False
        self._init_connection()

    def _init_connection(self):
        if not settings.TG_HOST:
            return
        try:
            import pyTigerGraph as tg

            host = settings.TG_HOST if settings.TG_HOST.startswith("http") else f"https://{settings.TG_HOST}"
            self.conn = tg.TigerGraphConnection(
                host=host,
                graphname=settings.TG_GRAPH or "CanvasGraph",
                username=settings.TG_USER or "tigergraph",
                password=settings.TG_PASSWORD or "tigergraph",
            )
            try:
                self.conn.getToken()
            except Exception:
                pass
            self.connected = True
            print(f"[TigerGraphStore] Connected to TigerGraph at {settings.TG_HOST}")
        except Exception as e:
            print(f"[TigerGraphStore] TigerGraph init notice: {e}")

    def upsert_chunks(self, chunks: list[dict]) -> None:
        self.shadow.upsert_chunks(chunks)
        self.shadow.save()
        if not self.connected or not self.conn:
            return
        try:
            v_data = {}
            for c in chunks:
                cid = _safe_str(c["id"])
                v_data[cid] = {
                    "text": _safe_str(c.get("text") or "")[:2000],
                    "source": _safe_str(c.get("source") or ""),
                    "valid_from": _safe_str(c.get("valid_from") or ""),
                    "source_authority": float(c.get("source_authority", 0.5)),
                }
            self.conn.upsertVertices("Chunk", v_data)
        except Exception as e:
            print(f"[TigerGraphStore] upsert_chunks exception: {e}")

    def upsert_entities(self, entities: list[dict]) -> None:
        self.shadow.upsert_entities(entities)
        self.shadow.save()
        if not self.connected or not self.conn:
            return
        try:
            v_data = {}
            for e in entities:
                eid = _safe_str(e["id"])
                v_data[eid] = {"name": _safe_str(e.get("name") or eid), "ntype": _safe_str(e.get("type") or "Entity")}
            self.conn.upsertVertices("Entity", v_data)
        except Exception as e:
            print(f"[TigerGraphStore] upsert_entities exception: {e}")

    def upsert_relations(self, relations: list[dict]) -> None:
        self.shadow.upsert_relations(relations)
        self.shadow.save()

    def link_entities(self, question: str) -> list[str]:
        return self.shadow.link_entities(question)

    def neighbors(self, ids: list[str], hops: int = 1, rel_filter: list[str] | None = None) -> dict:
        return self.shadow.neighbors(ids, hops=hops, rel_filter=rel_filter)

    def subgraph(self, ids: list[str]) -> dict:
        return self.shadow.subgraph(ids)

    def get_text(self, ids: list[str]) -> dict[str, str]:
        return self.shadow.get_text(ids)

    def vector_search(self, q: str, k: int = 5) -> list[dict]:
        return self.shadow.vector_search(q, k=k)

    def communities(self) -> dict:
        return self.shadow.communities()

    def stats(self) -> dict:
        st = self.shadow.stats()
        st["backend"] = "tigergraph" if self.connected else "tigergraph (local fallback)"
        st["tg_connected"] = self.connected
        return st


_store: GraphStore | None = None


def get_store() -> GraphStore:
    global _store
    if _store is not None:
        return _store
    if settings.GRAPH_BACKEND == "tigergraph":
        _store = TigerGraphStore()
        return _store
    loaded = InMemoryGraphStore.load()
    _store = loaded if loaded is not None else InMemoryGraphStore()
    return _store


def set_store(store: GraphStore) -> None:
    global _store
    _store = store
    if isinstance(store, InMemoryGraphStore):
        store.save()