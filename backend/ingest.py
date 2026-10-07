from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from backend.config import settings
from backend.llm import ZERO, complete_json
from backend.store import GraphStore, InMemoryGraphStore, get_store, set_store

EXTRACT_SYS = (
    "Extract entities and relations from the document as JSON: "
    '{"entities":[{"name":str,"type":"Person|Org|Tech|Attribute|Document"}],'
    '"relations":[{"src":str,"rel":str,"dst":str}]}. '
    "Types: Person, Org, Tech, Attribute. Use short canonical names."
)


def _sid(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _norm_ent(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")[:64]


def _token_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    wsize = max(20, int(size * 0.75))
    wov = int(overlap * 0.75)
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i : i + wsize]))
        i += max(1, wsize - wov)
    return out


def _meta_from_text(text: str, source: str) -> tuple[str, float]:
    m = re.search(r"date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text, re.I)
    valid = m.group(1) if m else ""
    am = re.search(r"authority:\s*([0-9.]+)", text, re.I)
    auth = float(am.group(1)) if am else (0.9 if "official" in source.lower() or "press" in text.lower()[:400] else 0.5)
    return valid, auth


def _parse_file_docs(fp: Path) -> list[dict]:
    """Parse file into a list of raw document dicts {text, source, title, valid_from, authority}."""
    ext = fp.suffix.lower()
    docs = []

    if ext == ".jsonl":
        lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    txt = str(item.get("text") or item.get("content") or item.get("body") or item.get("passage") or "")
                    if not txt.strip():
                        continue
                    doc_id = str(item.get("id") or item.get("doc_id") or f"{fp.stem}_{i}")
                    valid, auth = _meta_from_text(txt, str(fp))
                    if "date" in item or "valid_from" in item:
                        valid = str(item.get("date") or item.get("valid_from") or valid)
                    if "authority" in item:
                        try:
                            auth = float(item["authority"])
                        except (ValueError, TypeError):
                            pass
                    docs.append({
                        "text": txt,
                        "source": str(fp),
                        "title": str(item.get("title") or doc_id),
                        "valid_from": valid,
                        "authority": auth,
                        "doc_id": doc_id,
                    })
            except Exception:
                continue
        return docs

    if ext in {".txt", ".md"}:
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        if txt.strip():
            valid, auth = _meta_from_text(txt, str(fp))
            docs.append({
                "text": txt,
                "source": str(fp),
                "title": fp.stem,
                "valid_from": valid,
                "authority": auth,
                "doc_id": fp.stem,
            })
        return docs

    if ext == ".json":
        try:
            data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                txt = str(data.get("text") or data.get("content") or json.dumps(data))
                valid, auth = _meta_from_text(txt, str(fp))
                docs.append({
                    "text": txt,
                    "source": str(fp),
                    "title": str(data.get("title") or fp.stem),
                    "valid_from": valid,
                    "authority": auth,
                    "doc_id": fp.stem,
                })
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    txt = str(item.get("text") or item.get("content") or str(item)) if isinstance(item, dict) else str(item)
                    if txt.strip():
                        valid, auth = _meta_from_text(txt, str(fp))
                        docs.append({
                            "text": txt,
                            "source": str(fp),
                            "title": f"{fp.stem}_{i}",
                            "valid_from": valid,
                            "authority": auth,
                            "doc_id": f"{fp.stem}_{i}",
                        })
        except Exception:
            pass
        return docs

    if ext == ".pdf":
        try:
            from pypdf import PdfReader

            txt = "\n".join(page.extract_text() or "" for page in PdfReader(str(fp)).pages)
            if txt.strip():
                valid, auth = _meta_from_text(txt, str(fp))
                docs.append({
                    "text": txt,
                    "source": str(fp),
                    "title": fp.stem,
                    "valid_from": valid,
                    "authority": auth,
                    "doc_id": fp.stem,
                })
        except Exception:
            pass
        return docs

    return []


def _heuristic_extract(text: str) -> dict:
    caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\b", text)
    skip = {"The", "This", "In", "On", "A", "An", "According", "Date", "Source", "Authority"}
    names = []
    seen = set()
    for c in caps:
        if c.split()[0] in skip:
            continue
        k = c.lower()
        if k not in seen:
            seen.add(k)
            names.append(c)
    entities = []
    for n in names[:24]:
        t = "Person" if re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+$", n) else "Org" if any(w in n for w in ("Labs", "Inc", "Corp", "Agency", "Reuters", "Ventures")) else "Tech"
        entities.append({"name": n, "type": t})
    rels = []
    for m in re.finditer(r"([A-Z][a-zA-Z0-9 ]{1,40})\s+(?:is|was|became|leads|founded|acquired|launched|released)\s+(?:the\s+)?([A-Z][a-zA-Z0-9 ]{1,40})", text):
        rels.append({"src": m.group(1).strip(), "rel": "RELATED", "dst": m.group(2).strip()})
    return {"entities": entities, "relations": rels}


def extract_from_chunk(text: str, source: str, valid_from: str, authority: float) -> tuple[list[dict], list[dict], dict]:
    usage = dict(ZERO)
    data = {}
    if settings.GROQ_API_KEY:
        data, usage = complete_json(f"SOURCE={source}\nDATE={valid_from}\n\n{text[:4000]}", system=EXTRACT_SYS)
    if not data.get("entities"):
        data = _heuristic_extract(text)
    ents, rels = [], []
    name_to_id = {}
    for e in data.get("entities") or []:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        eid = _norm_ent(name)
        name_to_id[name.lower()] = eid
        name_to_id[eid] = eid
        ents.append({"id": eid, "name": name, "type": e.get("type") or "Entity"})
    for r in data.get("relations") or []:
        s, d = (r.get("src") or "").strip(), (r.get("dst") or "").strip()
        if not s or not d:
            continue
        sid = name_to_id.get(s.lower()) or _norm_ent(s)
        did = name_to_id.get(d.lower()) or _norm_ent(d)
        if sid not in {e["id"] for e in ents}:
            ents.append({"id": sid, "name": s, "type": "Entity"})
        if did not in {e["id"] for e in ents}:
            ents.append({"id": did, "name": d, "type": "Entity"})
        rels.append(
            {
                "id": _sid(f"{sid}|{r.get('rel')}|{did}|{valid_from}|{source}"),
                "src": sid,
                "dst": did,
                "rel": r.get("rel") or "RELATED",
                "valid_from": valid_from,
                "source": source,
                "source_authority": authority,
            }
        )
    return ents, rels, usage


def ingest_path(path: str | Path, store: GraphStore | None = None) -> dict:
    root = Path(path)
    store = store or get_store()
    files = []
    if root.is_file():
        files = [root]
    else:
        for ext in ("*.txt", "*.md", "*.json", "*.jsonl", "*.pdf"):
            files.extend(sorted(root.rglob(ext)))

    chunks, all_ents, all_rels = [], [], []
    usage = dict(ZERO)

    for fp in files:
        docs = _parse_file_docs(fp)
        for doc in docs:
            text = doc["text"]
            source = doc["source"]
            valid_from = doc["valid_from"]
            authority = doc["authority"]
            title = doc["title"]
            doc_id = doc["doc_id"]

            parts = _token_chunks(text, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP)
            for i, part in enumerate(parts):
                cid = "c_" + _sid(f"{doc_id}:{i}:{part[:80]}")
                chunks.append(
                    {
                        "id": cid,
                        "text": part,
                        "source": source,
                        "source_authority": authority,
                        "valid_from": valid_from,
                        "title": title,
                    }
                )
                ents, rels, u = extract_from_chunk(part, source, valid_from, authority)
                usage["prompt"] += u.get("prompt", 0)
                usage["completion"] += u.get("completion", 0)
                usage["total"] += u.get("total", 0)
                for e in ents:
                    e.setdefault("chunk_ids", []).append(cid)
                all_ents.extend(ents)
                all_rels.extend(rels)

    store.upsert_chunks(chunks)

    # Merge entity chunk_ids
    merged: dict[str, dict] = {}
    for e in all_ents:
        m = merged.setdefault(e["id"], {"id": e["id"], "name": e["name"], "type": e.get("type", "Entity"), "chunk_ids": []})
        m["chunk_ids"] = list(dict.fromkeys(m["chunk_ids"] + e.get("chunk_ids", [])))
        m["name"] = e.get("name") or m["name"]

    store.upsert_entities(list(merged.values()))
    store.upsert_relations(all_rels)

    if isinstance(store, InMemoryGraphStore):
        set_store(store)

    return {
        "files": len(files),
        "chunks": len(chunks),
        "entities": len(merged),
        "relations": len(all_rels),
        "tokens": usage,
        "stats": store.stats(),
    }