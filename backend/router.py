from __future__ import annotations

import math
import re

from backend.config import settings

MULTI = re.compile(r"\b(compare|between|both|after|before|which\b.*\bthat|and then|as well as)\b", re.I)
AGG = re.compile(r"\b(how many|most|average|themes|list all|count|total|customers)\b", re.I)
TEMP = re.compile(r"\b(when|date|before|after|in 20\d{2}|launched|became)\b", re.I)
ENT = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex) or 1.0
    return [e / s for e in ex]


class Router:
    def predict(self, question: str) -> dict[str, float]:
        q = question or ""
        nlen = min(len(q.split()) / 28.0, 1.5)
        nent = min(len(ENT.findall(q)) / 4.0, 1.5)
        mh = 1.0 if MULTI.search(q) else 0.0
        ag = 1.0 if AGG.search(q) else 0.0
        tm = 1.0 if TEMP.search(q) else 0.0
        # logits: rag, graphrag, agentic
        rag = 1.2 - 0.9 * mh - 0.5 * ag - 0.3 * nlen + 0.1 * (nent < 0.4)
        graph = 0.4 + 0.8 * mh + 0.5 * nent + 0.3 * tm - 0.2 * ag
        agent = 0.1 + 1.0 * ag + 0.7 * mh + 0.4 * tm + 0.3 * nlen
        p = _softmax([rag, graph, agent])
        return {"rag": p[0], "graphrag": p[1], "agentic": p[2]}

    def choose(self, question: str) -> tuple[str, dict[str, float], str]:
        probs = self.predict(question)
        mx = max(probs, key=probs.get)
        if probs[mx] >= settings.ROUTER_ARGMAX_MIN:
            return mx, probs, "argmax"
        if all(v < 0.5 for v in probs.values()):
            return "graphrag", probs, "graphrag_then_escalate"
        return mx, probs, "argmax_weak"


default_router = Router()
