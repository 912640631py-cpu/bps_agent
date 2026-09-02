"""Common response decoding and error handling for BPS HTTP adapters."""

from __future__ import annotations

import json
from typing import Any

import httpx

from bps_agent.errors import BpsError, ErrorCode


class BpsProtocolError(BpsError):
    default_code = ErrorCode.BPS_PROTOCOL_ERROR.value


def decode_payload(response: httpx.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except UnicodeDecodeError as exc:
        raise BpsProtocolError(
            "BPS response body is not valid UTF-8",
            code=ErrorCode.BPS_PROTOCOL_ERROR,
        ) from exc
    except json.JSONDecodeError:
        return response.text


def require_success_payload(response: httpx.Response) -> Any:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        code = (
            ErrorCode.BPS_AUTH_FAILED
            if response.status_code in {401, 403}
            else ErrorCode.BPS_PROTOCOL_ERROR
        )
        raise BpsProtocolError(
            f"BPS request failed with HTTP {response.status_code}",
            code=code,
        ) from exc
    return decode_payload(response)
