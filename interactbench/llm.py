"""LLM API helpers used by generate.py."""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from openai import AsyncOpenAI

from interactbench.model_profiles import ModelProfile


def _resolve_max_tokens(profile: ModelProfile, max_tokens: int | None) -> int | None:
    if max_tokens is not None:
        return int(max_tokens)
    if profile.max_tokens is not None:
        return int(profile.max_tokens)
    return None


def _api_key(profile: ModelProfile) -> str:
    api_key = os.getenv(profile.credential_env)
    if not api_key:
        raise ValueError(f"{profile.name}: env var not set: {profile.credential_env}")
    return api_key


def _openai_client_kwargs(profile: ModelProfile) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "api_key": _api_key(profile),
        "base_url": profile.endpoint,
    }
    if profile.timeout_s is not None:
        kwargs["timeout"] = float(profile.timeout_s)
    return kwargs


async def async_call_model(
    profile: ModelProfile,
    system_prompt: str,
    user_msg: str,
    max_tokens: int | None = None,
    temperature: float = 1.0,
) -> str:
    """Async model call used by the generation pipeline."""
    resolved_max_tokens = _resolve_max_tokens(profile, max_tokens)
    if profile.provider == "gemini_v1beta":
        return await asyncio.to_thread(
            _call_gemini_v1beta,
            profile=profile,
            system_prompt=system_prompt,
            user_msg=user_msg,
            max_tokens=resolved_max_tokens,
            temperature=temperature,
        )
    if profile.provider != "openai_compatible":
        raise ValueError(f"{profile.name}: unsupported provider: {profile.provider}")
    return await _async_call_openai_compatible(
        profile=profile,
        system_prompt=system_prompt,
        user_msg=user_msg,
        max_tokens=resolved_max_tokens,
        temperature=temperature,
    )


def _openai_create_kwargs(
    *,
    profile: ModelProfile,
    system_prompt: str,
    user_msg: str,
    max_tokens: int | None,
    temperature: float,
) -> dict[str, Any]:
    create_kwargs: dict[str, Any] = {
        "model": profile.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens is not None:
        create_kwargs["max_tokens"] = int(max_tokens)
    return create_kwargs


async def _async_call_openai_compatible(
    *,
    profile: ModelProfile,
    system_prompt: str,
    user_msg: str,
    max_tokens: int | None,
    temperature: float,
) -> str:
    client = AsyncOpenAI(**_openai_client_kwargs(profile))
    stream = await client.chat.completions.create(
        **_openai_create_kwargs(
            profile=profile,
            system_prompt=system_prompt,
            user_msg=user_msg,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )
    content_parts: list[str] = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content_parts.append(delta.content)
    return "".join(content_parts)


def _normalize_gemini_v1beta_base_url(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        return ""
    endpoint = endpoint.rstrip("/")
    prefix, sep, _rest = endpoint.partition("/v1beta")
    if sep:
        return prefix + "/v1beta"
    return endpoint + "/v1beta"


def _append_query_param(url: str, *, key: str, value: str) -> str:
    parsed = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    q[str(key)] = str(value)
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(q)))


def _extract_gemini_text(resp: object) -> str:
    if not isinstance(resp, dict):
        return ""
    candidates = resp.get("candidates")
    if isinstance(candidates, list) and candidates:
        cand0 = candidates[0]
        if isinstance(cand0, dict):
            content = cand0.get("content")
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list) and parts:
                    texts = [
                        str(part.get("text") or "")
                        for part in parts
                        if isinstance(part, dict) and "text" in part
                    ]
                    if texts:
                        return "".join(texts)
    if "text" in resp:
        return str(resp.get("text") or "")
    return ""


def _extract_gemini_text_from_stream_events(events: list[object]) -> str:
    full = ""
    for ev in events:
        chunk = _extract_gemini_text(ev)
        if not chunk:
            continue
        if chunk.startswith(full):
            full = chunk
            continue
        if not full.startswith(chunk):
            full += chunk
    return full


def _add_json_event(events: list[object], data: str) -> None:
    if not data or data == "[DONE]":
        return
    obj = json.loads(data)
    if isinstance(obj, list):
        events.extend(obj)
    else:
        events.append(obj)


def _iter_sse_events(resp) -> list[object]:
    events: list[object] = []
    buf: list[str] = []
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")
        if not line:
            if buf:
                _add_json_event(events, "".join(buf).strip())
                buf = []
            continue
        if line.startswith("data:"):
            buf.append(line[len("data:") :].lstrip())
        elif not buf:
            _add_json_event(events, line.strip())
    if buf:
        _add_json_event(events, "".join(buf).strip())
    return events


def _gemini_urls(profile: ModelProfile) -> tuple[str, str, str]:
    base = _normalize_gemini_v1beta_base_url(profile.endpoint)
    if not base:
        raise ValueError(f"{profile.name}: invalid Gemini endpoint: {profile.endpoint}")
    fmodel = profile.model
    if fmodel.startswith("models/"):
        fmodel = fmodel[len("models/") :]
    url_stream = f"{base}/models/{fmodel}:streamGenerateContent"
    url_non_stream = f"{base}/models/{fmodel}:generateContent"
    return url_stream, _append_query_param(url_stream, key="alt", value="sse"), url_non_stream


def _gemini_auth(profile: ModelProfile) -> tuple[dict[str, str], str | None]:
    api_key = _api_key(profile)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    auth_mode = profile.auth_mode
    query_key: str | None = None
    if auth_mode in {"google_api_key", "google", "key"}:
        query_key = "key"
        headers["x-goog-api-key"] = api_key
    elif auth_mode in {"query", "query_param"}:
        query_key = profile.auth_param or "key"
    elif auth_mode in {"header", "x-goog-api-key"}:
        headers[profile.auth_header or "x-goog-api-key"] = api_key
    elif auth_mode in {"bearer", "authorization"}:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        raise ValueError(f"{profile.name}: unknown Gemini auth mode: {auth_mode}")
    return headers, query_key


def _gemini_payload(
    *,
    profile: ModelProfile,
    system_prompt: str,
    user_msg: str,
    max_tokens: int | None,
    temperature: float,
) -> bytes:
    payload: dict[str, object] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_msg}],
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
        },
    }
    if max_tokens is not None:
        payload["generationConfig"]["maxOutputTokens"] = int(max_tokens)
    if system_prompt and not profile.disable_system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    elif system_prompt and profile.disable_system_instruction:
        payload["contents"][0]["parts"][0]["text"] = f"{system_prompt}\n\n{user_msg}"
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _post_json(url: str, *, data: bytes, headers: dict[str, str], timeout_s: float):
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout_s)


def _try_gemini_stream(url: str, *, data: bytes, headers: dict[str, str], timeout_s: float) -> str:
    with _post_json(url, data=data, headers=headers, timeout_s=timeout_s) as resp:
        events = _iter_sse_events(resp)
    return _extract_gemini_text_from_stream_events(events)


def _call_gemini_v1beta(
    *,
    profile: ModelProfile,
    system_prompt: str,
    user_msg: str,
    max_tokens: int | None,
    temperature: float,
) -> str:
    url_stream, url_stream_sse, url_non_stream = _gemini_urls(profile)
    headers, query_key = _gemini_auth(profile)
    if query_key:
        api_key = _api_key(profile)
        url_stream = _append_query_param(url_stream, key=query_key, value=api_key)
        url_stream_sse = _append_query_param(url_stream_sse, key=query_key, value=api_key)
        url_non_stream = _append_query_param(url_non_stream, key=query_key, value=api_key)

    data = _gemini_payload(
        profile=profile,
        system_prompt=system_prompt,
        user_msg=user_msg,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    timeout_s = float(profile.timeout_s or 120.0)
    stream_errors: list[str] = []

    for url, accept_sse in ((url_stream_sse, True), (url_stream, False)):
        stream_headers = dict(headers)
        if accept_sse:
            stream_headers.setdefault("Accept", "text/event-stream")
        try:
            text = _try_gemini_stream(url, data=data, headers=stream_headers, timeout_s=timeout_s)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            stream_errors.append(str(exc))
            continue
        if text:
            return text

    try:
        with _post_json(url_non_stream, data=data, headers=headers, timeout_s=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{profile.name}: gemini_v1beta HTTP {exc.code}: {err_body[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{profile.name}: gemini_v1beta request failed: {exc}") from exc

    try:
        parsed = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{profile.name}: gemini_v1beta invalid JSON response: {body[:2000]}") from exc

    if isinstance(parsed, dict) and "error" in parsed:
        raise RuntimeError(f"{profile.name}: gemini_v1beta error: {parsed.get('error')}")

    text = _extract_gemini_text(parsed)
    if text:
        return text
    if stream_errors:
        raise RuntimeError(f"{profile.name}: gemini_v1beta produced no text; stream errors: {stream_errors}")
    return json.dumps(parsed, ensure_ascii=False)
