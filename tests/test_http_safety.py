from __future__ import annotations

import pytest

from bps_agent.http_safety import canonical_origin, require_same_origin


def test_canonical_origin_normalizes_default_port_and_hostname_case() -> None:
    assert canonical_origin("https://BPS.Example.TEST/reports") == (
        "https",
        "bps.example.test",
        443,
    )
    assert canonical_origin("https://bps.example.test:443/api") == (
        "https",
        "bps.example.test",
        443,
    )


def test_canonical_origin_supports_ipv6_and_preserves_explicit_port() -> None:
    assert canonical_origin("http://[2001:db8::1]:8080/report") == (
        "http",
        "2001:db8::1",
        8080,
    )


def test_canonical_origin_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        canonical_origin("https://user:secret@bps.example.test/report")


def test_require_same_origin_rejects_cross_origin_url() -> None:
    with pytest.raises(ValueError, match="configured origin"):
        require_same_origin(
            "https://reports.example.test/export",
            "https://bps.example.test/api",
        )
