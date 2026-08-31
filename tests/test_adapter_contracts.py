from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from bps_agent.adapters.bps import BpsClient, PortReleaseError
from bps_agent.adapters.deepseek import (
    DeepSeekJudge,
    ProviderCompatibilityError,
    ProviderRequestError,
    ProviderResponseError,
)
from bps_agent.adapters.dut import DutClient
from bps_agent.models.bps import PortReservationState
from bps_agent.models.common import DutCollectionMethod, ReasoningEffort
from bps_agent.models.config import BpsConfig, DutConfig, DutFrontendConfig, ProviderConfig
from bps_agent.models.dut import ObservationPhase, SupplementalSnapshot


def make_deepseek_judge(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    attempts: int = 3,
    reasoning_effort: ReasoningEffort = "max",
) -> DeepSeekJudge:
    return DeepSeekJudge(
        "official",
        ProviderConfig(
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            token_env="DEEPSEEK_API_KEY",
            attempts=attempts,
        ),
        token="secret",
        reasoning_effort=reasoning_effort,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


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
        if path.endswith("/tests/operations/getSharedComponentSettings"):
            return httpx.Response(
                200,
                json={
                    "result": json.dumps(
                        {
                            "testModelName": "template",
                            "sharedComponentSettings": [
                                {
                                    "name": "totalBandwidth",
                                    "originalValue": "800.0",
                                    "currentValue": "400.0",
                                    "percentage": "50",
                                    "enabled": True,
                                }
                            ],
                        }
                    )
                },
            )
        if path.endswith("/topology/operations/reserve"):
            return httpx.Response(200, json={})
        if path.endswith("/tests/operations/setSharedComponentSettings"):
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
    template = client.find_template("template")
    assert template["version"] == 1
    assert template["totalBandwidthMbps"] == 800.0
    assert template["sharedComponentSettings"][0]["currentValue"] == "400.0"
    client.reserve_ports()
    client.set_total_bandwidth(75.5)
    assert client.start_run() == "1873"

    reserve = next(
        request for request in requests if request.url.path.endswith("operations/reserve")
    )
    assert json.loads(reserve.content)["force"] is False
    bandwidth = next(
        request
        for request in requests
        if request.url.path.endswith("operations/setSharedComponentSettings")
    )
    assert bandwidth.url.path == "/api/v1/bps/tests/operations/setSharedComponentSettings"
    assert json.loads(bandwidth.content) == {
        "modelName": "template",
        "sharedComponentSettings": [{"paramName": "totalBandwidth", "paramValue": 75.5}],
    }
    settings = next(
        request
        for request in requests
        if request.url.path.endswith("operations/getSharedComponentSettings")
    )
    assert json.loads(settings.content) == {"modelName": "template"}
    assert requests.index(bandwidth) < next(
        index
        for index, request in enumerate(requests)
        if request.url.path.endswith("/testmodel/operations/run")
    )


def test_bps_queries_actual_reservation_owners_and_active_runs() -> None:
    topology = {
        "runningTest": [
            {
                "id": "TEST-1922",
                "completed": False,
                "user": "agent-user",
                "port": [{"pi": {"slot": 4, "port": 4}}, {"pi": {"slot": 4, "port": 5}}],
            }
        ],
        "slot": [
            {
                "id": "4",
                "port": [
                    {
                        "id": "4",
                        "owner": "BreakingPoint/agent-user/10",
                        "reservedBy": "agent-user",
                    },
                    {"id": "5", "owner": "None", "reservedBy": ""},
                ],
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/bps/api/v2/core/topology"
        return httpx.Response(200, json=topology)

    client = BpsClient(
        BpsConfig(
            endpoint="https://bps.example.test",
            template="template",
            slot=4,
            ports=(4, 5),
            group=10,
        ),
        username="agent-user",
        password="password",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    status = client.port_reservation_status()

    assert status.state == PortReservationState.PARTIAL_AGENT
    assert [(item.port, item.owner, item.owned_by_agent) for item in status.reservations] == [
        (4, "agent-user", True),
        (5, None, False),
    ]
    assert client.find_active_runs_for_ports() == ("TEST-1922",)


def test_bps_total_bandwidth_rejects_invalid_percentage() -> None:
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
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))),
    )

    for invalid in (0.0, -1.0, 100.1):
        with pytest.raises(ValueError, match="between 0 and 100"):
            client.set_total_bandwidth(invalid)


@pytest.mark.parametrize(
    "result",
    [
        "not-json",
        json.dumps({"sharedComponentSettings": []}),
        json.dumps(
            {
                "sharedComponentSettings": [
                    {
                        "name": "totalBandwidth",
                        "originalValue": "not-a-number",
                        "enabled": True,
                    }
                ]
            }
        ),
    ],
)
def test_bps_template_preflight_rejects_unusable_bandwidth_json(result: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/testmodel/operations/search"):
            return httpx.Response(200, json=[{"name": "template"}])
        if request.url.path.endswith("/tests/operations/getSharedComponentSettings"):
            return httpx.Response(200, json={"result": result})
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

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
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError):
        client.find_template("template")


def test_dut_contract_reads_history_once_and_filters_the_traffic_window() -> None:
    paths: list[str] = []
    requests: list[httpx.Request] = []

    points = [
        {"time": "08-24 09:49:59", "value": 1},
        {"time": "08-24 09:50:00", "value": 2},
        {"time": "08-24 09:59:59", "value": 3},
        {"time": "08-24 10:00:00", "value": 4},
        {"time": "08-24 10:05:00", "value": 5},
        {"time": "08-24 10:05:10", "value": 6},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
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
        if request.url.path.endswith(("/cpu", "/mem", "/traffic")):
            return httpx.Response(
                200,
                json={"code": 0, "data": {"data": points, "metadata": "preserved"}},
            )
        if request.url.path.endswith(("/newSess", "/concurrentSess")):
            return httpx.Response(200, json={"code": 0, "data": points})
        return httpx.Response(200, json={"code": 0, "data": {}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = DutClient(
        DutConfig(
            collection_method=DutCollectionMethod.FRONTEND_API,
            interfaces=("T1/1", "T1/2"),
            frontend=DutFrontendConfig(
                endpoint="https://dut.example.test",
                period="three-hour-fixture",
                read_retry_backoff_seconds=0,
            ),
        ),
        username="user",
        password="password",
        captcha_reader=lambda _image, _media: "1234",
        client=http,
    )

    client.authenticate()
    client.keepalive()
    before = SupplementalSnapshot(
        captured_at="2026-08-24T01:59:52+00:00",
        values={
            "interfaces": {},
            "hardware": {},
            "system": {"data": {"current_time": "2026-08-24 10:00:00,Asia@Shanghai"}},
        },
    )
    after = SupplementalSnapshot(
        captured_at="2026-08-24T02:05:02+00:00",
        values={
            "interfaces": {},
            "hardware": {},
            "system": {"data": {"current_time": "2026-08-24 10:05:10,Asia@Shanghai"}},
        },
    )

    observations = client.collect_monitoring_window(
        "2026-08-24T01:59:52+00:00",
        "2026-08-24T02:04:52+00:00",
        before,
        after,
    )

    assert [item.phase for item in observations] == [
        ObservationPhase.BASELINE,
        ObservationPhase.DURING,
        ObservationPhase.RECOVERY,
    ]
    assert [point["value"] for point in observations[0].resources["cpu"]["data"]["data"]] == [2, 3]
    assert [point["value"] for point in observations[1].resources["cpu"]["data"]["data"]] == [4, 5]
    assert [point["value"] for point in observations[2].resources["cpu"]["data"]["data"]] == [6]
    assert observations[1].resources["cpu"]["data"]["metadata"] == "preserved"
    assert paths.count("/api/dashboards/system/cpu") == 1
    assert paths.count("/api/dashboards/system/mem") == 1
    assert paths.count("/api/dashboards/system/newSess") == 1
    assert paths.count("/api/dashboards/system/concurrentSess") == 1
    assert paths.count("/api/dashboards/system/traffic") == 2
    keepalive_request = next(
        request for request in requests if request.url.path == "/api/dashboards/system/systemInfo"
    )
    assert "period" not in keepalive_request.url.params
    resource_requests = [
        request
        for request in requests
        if request.url.path
        in {
            "/api/dashboards/system/cpu",
            "/api/dashboards/system/mem",
            "/api/dashboards/system/newSess",
            "/api/dashboards/system/concurrentSess",
            "/api/dashboards/system/traffic",
        }
    ]
    assert all(
        request.url.params["period"] == "three-hour-fixture" for request in resource_requests
    )


def test_deepseek_contract_sends_configured_reasoning_effort() -> None:
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

    judge = make_deepseek_judge(handler, attempts=1, reasoning_effort="high")

    judge.validate_compatibility()

    assert captured["authorization"] == "Bearer secret"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["thinking"] == {"type": "enabled"}
    assert captured["body"]["reasoning_effort"] == "high"


def test_deepseek_retries_invalid_json_at_most_three_times() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "not-json" if calls < 3 else '{"verdict":"pass"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    judge = make_deepseek_judge(handler)

    judge.validate_compatibility()

    assert calls == 3


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test_deepseek_wraps_non_retryable_http_status(status_code: int) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="rejected")

    judge = make_deepseek_judge(handler)

    with pytest.raises(ProviderRequestError, match=f"HTTP {status_code}"):
        judge.validate_compatibility()


def test_deepseek_wraps_invalid_response_after_retries() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": []})

    judge = make_deepseek_judge(handler, attempts=2)

    with pytest.raises(ProviderResponseError, match="omitted choices"):
        judge.validate_compatibility()
    assert calls == 2


def test_deepseek_wraps_transport_failure_after_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused", request=request)

    judge = make_deepseek_judge(handler, attempts=2)

    with pytest.raises(ProviderRequestError, match="transport failed"):
        judge.validate_compatibility()
    assert calls == 2


@pytest.mark.parametrize(
    ("phase", "response_body", "error_type", "message_fragments"),
    [
        (
            "probe",
            {"text": "unsupported reasoning"},
            ProviderCompatibilityError,
            ("unsupported reasoning",),
        ),
        (
            "adjudication",
            {
                "json": {
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "input exceeds the model context window",
                    }
                }
            },
            ProviderRequestError,
            ("HTTP 400", "context_length_exceeded", "input exceeds the model context window"),
        ),
        (
            "probe",
            {"json": {"error": {"code": "invalid_request", "message": "malformed messages"}}},
            ProviderRequestError,
            ("malformed messages",),
        ),
    ],
)
def test_deepseek_classifies_bad_requests_by_call_phase(
    phase: str,
    response_body: dict[str, Any],
    error_type: type[Exception],
    message_fragments: tuple[str, ...],
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, **response_body)

    judge = make_deepseek_judge(handler, attempts=1)

    with pytest.raises(error_type) as captured:
        if phase == "probe":
            judge.validate_compatibility()
        else:
            judge._request_verdict([{"role": "user", "content": "real Evidence Bundle"}])

    assert all(fragment in str(captured.value) for fragment in message_fragments)


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


def test_bps_reconciliation_filters_running_runs_by_template_and_group() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/topology/runningTest")
        return httpx.Response(
            200,
            json=[
                {"runId": "11", "modelName": "template", "group": 10},
                {"runId": "12", "modelName": "other", "group": 10},
                {"runId": "13", "modelName": "template", "group": 11},
            ],
        )

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
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.find_running_runs(template="template", group=10) == ("11",)


def test_bps_temporary_disappearance_does_not_complete_without_ready_report() -> None:
    running_responses: list[list[dict[str, Any]]] = [
        [{"id": "TEST-1873", "completed": False}],
        [],
        [],
        [{"id": "TEST-1873", "completed": True}],
    ]
    report_checks = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal report_checks
        if request.url.path.endswith("/topology/runningTest"):
            return httpx.Response(200, json=running_responses.pop(0))
        if request.url.path.endswith("/reports/operations/getReportContents"):
            report_checks += 1
            return httpx.Response(503, json={"message": "not ready"})
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

    completion = client.wait_for_completion("1873", lambda: None)

    assert completion.details["completion"] == "explicit-terminal-state"
    assert report_checks == 2
    assert running_responses == []


def test_bps_seen_run_disappearance_requires_ready_report() -> None:
    running_responses: list[list[dict[str, Any]]] = [
        [{"id": "TEST-1873", "completed": False}],
        [],
        [],
    ]
    report_responses = [
        httpx.Response(503, json={"message": "not ready"}),
        httpx.Response(200, json={"sections": ["3.2"]}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/topology/runningTest"):
            return httpx.Response(200, json=running_responses.pop(0))
        if request.url.path.endswith("/reports/operations/getReportContents"):
            return report_responses.pop(0)
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

    completion = client.wait_for_completion("1873", lambda: None)

    assert completion.details["completion"] == "running-test-absent-and-report-ready"
    assert report_responses == []


def test_bps_port_release_retries_a_transient_bad_request() -> None:
    statuses = [400, 204]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(statuses.pop(0), json={"message": "run cleanup pending"})

    client = BpsClient(
        BpsConfig(
            endpoint="https://bps.example.test",
            template="template",
            slot=4,
            ports=(4, 5),
            group=10,
            port_release_attempts=2,
            port_release_retry_backoff_seconds=0,
        ),
        username="user",
        password="password",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    client.release_ports()

    assert len(requests) == 2
    assert statuses == []


def test_bps_port_release_can_target_only_agent_owned_subset() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

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
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    client.release_ports((4,))

    assert json.loads(requests[0].content) == {
        "unreservation": [{"slot": 4, "port": 4}]
    }


def test_bps_port_release_reports_retry_exhaustion_separately() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"message": "run cleanup pending"})

    client = BpsClient(
        BpsConfig(
            endpoint="https://bps.example.test",
            template="template",
            slot=4,
            ports=(4, 5),
            group=10,
            port_release_attempts=3,
            port_release_retry_backoff_seconds=0,
        ),
        username="user",
        password="password",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(PortReleaseError, match=r"after 3 attempts.*HTTP 400"):
        client.release_ports()

    assert calls == 3


def test_bps_export_report_uses_runtime_section_selection(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/reports/operations/exportReport"):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"url": "/reports/export.csv"})
        if request.url.path == "/reports/export.csv":
            return httpx.Response(
                200,
                content=b"section,result\n3.2,pass\n",
                headers={"content-type": "text/csv"},
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

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
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    destination = client.export_report("1873", tmp_path / "report.csv", ("3.2", "7.19"))

    assert destination.read_text(encoding="utf-8") == "section,result\n3.2,pass\n"
    assert captured["body"]["sectionIds"] == "3.2,7.19"
    assert captured["body"]["includeSubsections"] is False


def test_bps_exports_complete_pdf_with_pdf_limits(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/reports/operations/exportReport"):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"url": "/reports/full.pdf"})
        if request.url.path == "/reports/full.pdf":
            return httpx.Response(
                200,
                content=b"%PDF-1.7\nfixture",
                headers={"content-type": "application/pdf"},
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    client = BpsClient(
        BpsConfig(
            endpoint="https://bps.example.test",
            template="template",
            slot=4,
            ports=(4, 5),
            group=10,
            pdf_report_timeout_seconds=900,
            max_pdf_report_bytes=1024 * 1024,
        ),
        username="user",
        password="password",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    destination = client.export_full_report_pdf(
        "1873",
        tmp_path / "bps-report-full.pdf",
        ("1", "1.1", "2"),
    )

    assert destination.read_bytes() == b"%PDF-1.7\nfixture"
    assert captured["body"]["reportType"] == "PDF"
    assert captured["body"]["sectionIds"] == "1,1.1,2"
    assert captured["body"]["includeSubsections"] is True
    assert captured["body"]["dataType"] == "ALL"
