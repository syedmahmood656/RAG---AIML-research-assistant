"""Tiny Server-Sent-Events encoder (no third-party imports so it is trivially testable)."""

import json


def encode(event: str, payload) -> str:
    """
    Encode one SSE message. The payload is ALWAYS JSON-encoded, so newlines, quotes and
    non-ASCII characters inside answer tokens survive the round trip (json.dumps escapes
    newlines as the two characters backslash-n, which can never terminate an SSE line).
    """
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"
