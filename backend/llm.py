def _cache_get(key: str) -> dict | None:
    v = _cache_load().get(key)
    return v if isinstance(v, dict) and v.get("text") else None


def _cache_put(key: str, text: str, usage: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _cache_load()
    data[key] = {"text": text, "usage": usage}
    CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")


def complete(
    prompt: str,
    system: str = "",
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> tuple[str, dict]:
    """Return (text, usage{prompt,completion,total})."""
    key = _cache_key(system, prompt)
    hit = _cache_get(key)
    if hit is not None:
        return hit["text"], dict(hit["usage"])  # original token counts, not zero

    client = _client()
    if client is None:
        return "", dict(ZERO)

    messages = []
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

    last_err = None
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            usage = _usage_from(resp, system + prompt, text)
            if not text and attempt < 3:  # reasoning ate the budget: retry with more room
                kwargs["max_tokens"] = int(kwargs["max_tokens"] * 2)
                continue
            if text:
                _cache_put(key, text, usage)
            return text, usage
        except Exception as e:
            last_err = e
            code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e).lower()
            if code == 429 or "429" in msg or "rate limit" in msg:
                time.sleep(_retry_sleep_429(e))
            else:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"LLM failed after retries: {last_err}")