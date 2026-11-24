"""SSE client helpers for the Streamlit UI (pure Python + requests, no Streamlit imports)."""

from __future__ import annotations

import json
from typing import Iterable, Iterator, Tuple

import requests


class BackendError(RuntimeError):
    """The backend reported an unrecoverable error through an `event: error` message."""


def iter_events(lines: Iterable[str | bytes | None]) -> Iterator[Tuple[str, str]]:
    """Parse SSE lines into (event, data) pairs. A blank line terminates a message."""
    event, data = None, []
    for line in lines:
        if line is None:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line == "":
            if event is not None or data:
                yield (event or "message", "\n".join(data))
            event, data = None, []
            continue
        if line.startswith(":"):  # comment / keep-alive
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if event is not None or data:
        yield (event or "message", "\n".join(data))


def decode_events(lines: Iterable[str | bytes | None]) -> Iterator[Tuple[str, object]]:
    """Yield (kind, payload) with JSON already decoded. Raises BackendError on `error` events."""
    for event, data in iter_events(lines):
        payload = json.loads(data) if data else None
        if event == "error":
            raise BackendError((payload or {}).get("message", "unknown backend error"))
        yield event, payload


def stream_query(api_url: str, question: str, k: int) -> Iterator[Tuple[str, object]]:
    """POST /query and yield (kind, payload): sources, notice, token, done."""
    resp = requests.post(
        f"{api_url}/query",
        json={"question": question, "k": k},
        stream=True,
        timeout=(5, 300),
    )
    resp.raise_for_status()
    # chunk_size=None -> yield data as soon as it arrives (true progressive rendering)
    yield from decode_events(resp.iter_lines(chunk_size=None))


def fetch_info(api_url: str) -> dict | None:
    """GET /info; returns None when the API is unreachable."""
    try:
        r = requests.get(f"{api_url}/info", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None
