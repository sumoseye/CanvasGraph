from __future__ import annotations

import math
import re

from backend.config import settings

MULTI = re.compile(
    r"\b(compare|between|both|after|before|which\b.*\bthat|and then|as well as|"
    r"relationship|connected|via|through|path|linked)\b",
    re.I,
)
AGG = re.compile(
    r"\b(how many|most|average|themes|list all|count|total|customers|sum|top\s+\d+)\b",
    re.I,
)
TEMP = re.compile(r"\b(when|date|before|after|in 20\d{2}|launched|became|since|until)\b", re.I)
CONFLICT = re.compile(
    r"\b(conflict|contradict|supersede|instead|updated|retract|vs\.?|versus|disagree)\b",
    re.I,
)
ENT = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex) or 1.0
    return [e / s for e in ex]


class Router:
    """
    Jev — non-generative cheap classifier.
    Bias: prefer the cheapest pipeline that is still likely correct.
    Rule: argmax if p>=ROUTER_ARGMAX_MIN; if all <0.5 → graphrag_then_escalate.
    """

    def predict(self, question: str) -> dict[str, float]:
        q = question or ""
        tokens = q.split()
        nlen = min(len(tokens) / 28.0, 1.5)
        nent = min(len(ENT.findall(q)) / 4.0, 1.5)
        mh = 1.0 if MULTI.search(q) else 0.0
        ag = 1.0 if AGG.search(q) else 0.0
        tm = 1.0 if TEMP.search(q) else 0.0
        cf = 1.0 if CONFLICT.search(q) else 0.0

        # strong prior toward RAG (cheapest) — only climb when cues demand it
        rag = 2.0 - 1.2 * mh - 0.9 * ag - 0.8 * cf - 0.35 * nlen - 0.2 * tm
        # mild lift when short + few entities (classic single-hop)
        if nlen < 0.4 and nent < 0.5 and mh == 0 and ag == 0 and cf == 0:
            rag += 0.8

        graph = 0.3 + 1.0 * mh + 0.6 * nent + 0.4 * tm - 0.3 * ag - 0.2 * cf
        agent = -0.2 + 1.3 * ag + 0.9 * cf + 0.5 * mh + 0.35 * tm + 0.25 * nlen

        # if literally no complexity cues, crush agent/graph
        if mh == 0 and ag == 0 and cf == 0 and tm == 0 and nent < 0.5:
            graph -= 0.6
            agent -= 1.0
            rag += 0.6

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