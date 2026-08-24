from __future__ import annotations

import json
from typing import Any

import httpx

from bps_agent.adapters.bps import BpsClient
from bps_agent.adapters.deepseek import DeepSeekJudge
from bps_agent.adapters.dut import DutClient
from bps_agent.models import BpsConfig, DutConfig, ObservationPhase, ProviderConfig


def test_bps_run_contract_never_forces_reservation() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith("/auth/session"):
            return httpx.Response(200, json={"sessionId": "session", "apiKey": "key"})
        if path.endswith("/auth/login"):
            return httpx.Response(200, json={})
        if path.endswith("/testmodel/operations/search"):
            return httpx.Response(200, json=[{"name": "template", "version": 1}])
        if path.endswith("/topology/operations/reserve"):
            return httpx.Response(200, json={})
        if path.endswith("/testmodel/operations/run"):
            return httpx.Response(200, json={"runId": "1873"})
        raise AssertionError(f"unexpected request: {request.method} {path}")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = BpsClient(
        BpsConfig(
            endpoint="https://bps.example.test",
            template="template",
            slot=4,
            ports=(4, 5),
            group=10,
        ),
        username="user",
        password="password",
        client=http,
    )

    client.authenticate()
    assert client.find_template("template")["version"] == 1
    client.reserve_ports()
    assert client.start_run() == "1873"

    reserve = next(
        request for request in requests if request.url.path.endswith("operations/reserve")
    )
    assert json.loads(reserve.content)["force"] is False


def test_dut_contract_collects_five_resources_and_three_snapshots() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/login/checkLoginAuth"):
            return httpx.Response(200, json={})
        if request.url.path.endswith("/account/code"):
            return httpx.Response(200, content=b"image", headers={"content-type": "image/png"})
        if request.url.path.endswith("/login/login"):
            return httpx.Response(
                200,
                json={"code": 0, "data": {"apiKey": "api", "securityKey": "security"}},
            )
        return httpx.Response(200, json={"code": 0, "data": {}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = DutClient(
        DutConfig(
            endpoint="https://dut.example.test",
            interfaces=("T1/1", "T1/2"),
            read_retry_backoff_seconds=0,
        ),
        username="user",
        password="password",
        captcha_reader=lambda _image, _media: "1234",
        client=http,
    )

    client.authenticate()
    resources = client.collect_resources(ObservationPhase.BASELINE.value)
    supplemental = client.collect_supplemental()

    assert resources.is_complete(("T1/1", "T1/2"))
    assert supplemental.is_complete
    assert paths.count("/api/dashboards/system/traffic") == 2
    assert "/api/dashboards/system/interface" in paths
    assert "/api/dashboards/system/hardware" in paths
    assert "/api/dashboards/system/systemInfo" in paths


def test_deepseek_contract_sends_json_and_max_reasoning() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"verdict":"pass"}'}}],
                "usage": {},
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    judge = DeepSeekJudge(
        "official",
        ProviderConfig(
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            token_env="DEEPSEEK_API_KEY",
            attempts=1,
        ),
        token="secret",
        client=http,
    )

    judge.validate_compatibility()

    assert captured["authorization"] == "Bearer secret"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["thinking"] == {"type": "enabled"}
    assert captured["body"]["reasoning_effort"] == "max"


def test_deepseek_retries_invalid_json_at_most_three_times() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "not-json" if calls < 3 else '{"verdict":"pass"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    judge = DeepSeekJudge(
        "official",
        ProviderConfig(
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            token_env="DEEPSEEK_API_KEY",
            attempts=3,
        ),
        token="secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    judge.validate_compatibility()

    assert calls == 3


def test_bps_completion_requires_report_when_run_was_never_observed() -> None:
    report_checks = 0
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal report_checks
        if request.url.path.endswith("/topology/runningTest"):
            return httpx.Response(200, json=[])
        if request.url.path.endswith("/reports/operations/getReportContents"):
            report_checks += 1
            if report_checks == 1:
                return httpx.Response(503, json={"message": "not ready"})
            return httpx.Response(200, json={"sections": ["3.2"]})
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    client = BpsClient(
        BpsConfig(
            endpoint="https://bps.example.test",
            template="template",
            slot=4,
            ports=(4, 5),
            group=10,
            poll_interval_seconds=0.001,
            run_timeout_seconds=1,
            registration_grace_seconds=0,
        ),
        username="user",
        password="password",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    def on_poll() -> None:
        nonlocal polls
        polls += 1

    completion = client.wait_for_completion("1873", on_poll)

    assert completion.terminal
    assert completion.details["completion"] == "report-ready-before-registration-observed"
    assert report_checks == polls == 2
