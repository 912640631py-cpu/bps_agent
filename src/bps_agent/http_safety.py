"""Shared URL-origin safety rules for authenticated HTTP traffic."""

from __future__ import annotations

from urllib.parse import urlsplit

Origin = tuple[str, str, int]


def canonical_origin(url: str) -> Origin:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid HTTP URL: {url!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("HTTP URL must not contain credentials")
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid HTTP URL port: {url!r}") from exc
    port = parsed_port if parsed_port is not None else (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname.casefold(), port


def require_same_origin(candidate: str, expected: str) -> None:
    if canonical_origin(candidate) != canonical_origin(expected):
        raise ValueError("authenticated HTTP URL must remain on the configured origin")
