"""Common response decoding and error handling for BPS HTTP adapters."""

from __future__ import annotations

import json
from typing import Any

import httpx


class BpsProtocolError(RuntimeError):
    pass


def decode_payload(response: httpx.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except json.JSONDecodeError:
        return response.text


def require_success_payload(response: httpx.Response) -> Any:
    response.raise_for_status()
    return decode_payload(response)
