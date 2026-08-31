from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bps_agent.adapters.bps import BpsClient
from bps_agent.adapters.bps_protocol import BpsProtocolError
from bps_agent.artifacts import ArtifactStore
from bps_agent.http_safety import require_same_origin
from bps_agent.models.config import BpsConfig


def test_artifact_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="unsafe path"):
        store.ensure_evaluation("../outside")


def test_configuration_rejects_credentials_in_endpoint() -> None:
    with pytest.raises(ValueError, match="credential-free"):
        BpsConfig(
            endpoint="https://user:password@bps.example.test",
            template="template",
            slot=4,
            ports=(4, 5),
            group=10,
        )


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("https://bps.example.test:443/report", "https://bps.example.test"),
        ("https://BPS.Example.TEST/report", "https://bps.example.test/api"),
    ],
)
def test_http_origin_safety_accepts_equivalent_origins(
    candidate: str,
    expected: str,
) -> None:
    require_same_origin(candidate, expected)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("http://bps.example.test/report", "https://bps.example.test"),
        ("https://reports.example.test/export", "https://bps.example.test"),
        ("https://user:secret@bps.example.test/report", "https://bps.example.test"),
        ("https://bps.example.test:70000/report", "https://bps.example.test"),
    ],
)
def test_http_origin_safety_rejects_unsafe_or_different_origins(
    candidate: str,
    expected: str,
) -> None:
    with pytest.raises(ValueError):
        require_same_origin(candidate, expected)


def test_bps_api_rejects_cross_origin_request_before_network_access(
    bps_config: BpsConfig,
) -> None:
    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("cross-origin request reached the network")

    with httpx.Client(transport=httpx.MockTransport(unexpected_request)) as http:
        client = BpsClient(bps_config, username="user", password="password", client=http)

        with pytest.raises(BpsProtocolError, match="another origin"):
            client._request("GET", "https://attacker.example.test/api")


def test_bps_report_rejects_cross_origin_redirect(
    tmp_path: Path,
    bps_config: BpsConfig,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/reports/operations/exportReport"):
            return httpx.Response(200, json={"url": "/download/report.csv"})
        if request.url.path == "/download/report.csv":
            return httpx.Response(
                302,
                headers={"location": "https://attacker.example.test/report.csv"},
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = BpsClient(bps_config, username="user", password="password", client=http)

        with pytest.raises(BpsProtocolError, match="escaped the authenticated origin"):
            client.export_report("run-1", tmp_path / "report.csv", ("1",))
