from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from backend.config import ROOT, settings

CACHE_PATH = ROOT / ".cache" / "llm.json"
ZERO: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}

_client_singleton = None


def _client():
    """Return OpenAI-compatible Groq client, or None if no key."""
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    if not settings.GROQ_API_KEY:
        return None
    try:
        from openai import OpenAI

        _client_singleton = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url=settings.GROQ_BASE_URL,
        )
        return _client_singleton
    except Exception as e:
        raise RuntimeError(f"Failed to init Groq client: {e}") from e


def _cache_load() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cache_key(system: str, prompt: str) -> str:
    raw = f"{settings.GROQ_MODEL}|{system}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> dict | None:
    v = _cache_load().get(key)
    # must have non-empty text AND a usage dict
    if isinstance(v, dict) and v.get("text") and isinstance(v.get("usage"), dict):
        return v
    return None


def _cache_put(key: str, text: str, usage: dict) -> None:
    if not text:
        return  # never cache empty answers
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _cache_load()
    data[key] = {
        "text": text,
        "usage": {
            "prompt": int(usage.get("prompt", 0)),
            "completion": int(usage.get("completion", 0)),
            "total": int(usage.get("total", 0)),
        },
    }
    CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")


def _usage_from(resp: Any, prompt_text: str, completion_text: str) -> dict:
    """Extract usage; fall back to rough char/4 estimate if API omits it."""
    u = getattr(resp, "usage", None)
    if u is not None:
        prompt = int(getattr(u, "prompt_tokens", 0) or 0)
        completion = int(getattr(u, "completion_tokens", 0) or 0)
        total = int(getattr(u, "total_tokens", 0) or (prompt + completion))
        return {"prompt": prompt, "completion": completion, "total": total}
    # fallback estimate
    p = max(1, len(prompt_text) // 4)
    c = max(1, len(completion_text) // 4) if completion_text else 0
    return {"prompt": p, "completion": c, "total": p + c}


def _retry_sleep_429(err: Exception) -> float:
    """Parse 'try again in Xs' from Groq 429 body; default ~8s."""
    msg = str(err)
    m = re.search(r"try again in ([\d.]+)\s*s", msg, re.I)
    if m:
        return float(m.group(1)) + 1.0
    m = re.search(r"Please try again in ([\d.]+)s", msg, re.I)
    if m:
        return float(m.group(1)) + 1.0
    return 8.0


def complete(
    prompt: str,
    system: str = "",
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> tuple[str, dict]:
    """
    Return (text, usage{prompt,completion,total}).
    Cache hit returns ORIGINAL stored usage (never zero).
    Fails loudly if no API key / all retries exhausted.
    Never caches empty text.
    """
    if max_tokens is None:
        max_tokens = settings.ANSWER_MAX_TOKENS

    key = _cache_key(system, prompt)
    hit = _cache_get(key)
    if hit is not None:
        return hit["text"], dict(hit["usage"])

    client = _client()
    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing or empty. "
            "Set it in .env — silent offline fallback is disabled."
        )

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": settings.GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if "gpt-oss" in (settings.GROQ_MODEL or ""):
        kwargs["extra_body"] = {"reasoning_effort": "low"}

    last_err: Exception | None = None
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            usage = _usage_from(resp, system + prompt, text)

            # gpt-oss reasoning can eat the budget → empty text; retry bigger
            if not text and attempt < 4:
                kwargs["max_tokens"] = int(kwargs["max_tokens"] * 2)
                time.sleep(0.4)
                continue

            if text:
                _cache_put(key, text, usage)
            return text, usage

        except Exception as e:
            last_err = e
            code = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None
            )
            msg = str(e).lower()
            if code == 429 or "429" in msg or "rate limit" in msg:
                time.sleep(_retry_sleep_429(e))
            else:
                time.sleep(0.6 * (attempt + 1))

    raise RuntimeError(f"LLM failed after retries: {last_err}")


def complete_json(
    prompt: str,
    system: str = "",
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> tuple[dict, dict]:
    """
    Return (parsed_dict, usage). Retries parse failures up to 3 times.
    On total failure returns ({}, usage_or_ZERO) — callers should handle empty dict.
    """
    if max_tokens is None:
        max_tokens = settings.JSON_MAX_TOKENS

    last_usage = dict(ZERO)
    for attempt in range(3):
        text, usage = complete(
            prompt,
            system=system,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        last_usage = usage
        if not text:
            max_tokens = int(max_tokens * 2)
            continue
        # strip markdown fences if model wraps JSON
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data, usage
        except json.JSONDecodeError:
            # ask model to fix on next loop by tightening prompt slightly
            prompt = (
                prompt
                + "\n\nPREVIOUS OUTPUT WAS INVALID JSON. Reply with ONLY a single JSON object."
            )
            continue
    return {}, last_usage


def estimate_tokens(text: str) -> int:
    """Rough token estimate for context budgeting (no API call)."""
    if not text:
        return 0
    return max(1, len(text) // 4)