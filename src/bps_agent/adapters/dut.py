"""Read-only DUT authentication and monitoring adapter."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx

from bps_agent.artifacts import ArtifactStore
from bps_agent.models import (
    DutCaptureResult,
    DutConfig,
    DutObservations,
    FrontendDutEvidence,
    ObservationPhase,
    ResourceObservation,
    SupplementalSnapshot,
)

_RSA_EXPONENT = 0x10001
_RSA_MODULUS = int(
    "AF018735A835021A030719232C2D86556FC29369B0A6224E2ECB12477091400C"
    "CED6054E8498CCF985293C1641671121D1F7F40A978364ED2A51A6BA0707E62E"
    "B121B0AD38AE03A8F75B725ECC63540AEEF564A61825A208F9955C97601657E5"
    "D468B5CBD0E47ED48830E3A33E018DB48CCEB8E7E55326DAF2B848166E49F62"
    "D7CAAE6CDCE23C5D94F6D018DEF6A11FC9A0F0B4FE8E7BD5831E12CB204DFC2"
    "9AA6294AB3E79FD7E60F936966078BCDA0A36E743C0E122493E7271CC6E959A7"
    "2B76A7E72267F2C7B498B49F4BA3C8A710D119DDAB0ACA91889F0D745BAD80DC"
    "EB2514A4CD304BAE397DDF35D15D2A1CC2DE5D78B289C814F00A9DF90C267D0"
    "133",
    16,
)
_RSA_CHUNK_SIZE = 2 * ((_RSA_MODULUS.bit_length() - 1) // 16)
_SUCCESS_CODES = {0, 2000}
_RETRYABLE_STATUS = {429, 502, 503, 504}


def _aware_datetime(value: str, label: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _snapshot_device_time(snapshot: SupplementalSnapshot) -> datetime:
    system = snapshot.values.get("system")
    if not isinstance(system, dict) or not isinstance(system.get("data"), dict):
        raise ValueError("DUT system summary omitted data.current_time")
    current_time = system["data"].get("current_time")
    if not isinstance(current_time, str) or not current_time:
        raise ValueError("DUT system summary omitted data.current_time")
    time_text, separator, zone_text = current_time.partition(",")
    if not separator or not zone_text:
        raise ValueError("DUT system current_time omitted its timezone")
    timezone = ZoneInfo(zone_text.replace("@", "/", 1))
    return datetime.strptime(time_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone)


def _clock_offset(
    before: SupplementalSnapshot, after: SupplementalSnapshot
) -> tuple[timedelta, datetime]:
    device_before = _snapshot_device_time(before)
    device_after = _snapshot_device_time(after)
    local_before = _aware_datetime(before.captured_at, "DUT before captured_at")
    local_after = _aware_datetime(after.captured_at, "DUT after captured_at")
    before_offset = device_before.astimezone(UTC) - local_before.astimezone(UTC)
    after_offset = device_after.astimezone(UTC) - local_after.astimezone(UTC)
    average = (before_offset + after_offset) / 2
    return average, device_after


def _point_time(point: dict[str, Any], reference: datetime) -> datetime | None:
    timestamp = point.get("timestamp")
    if isinstance(timestamp, (str, int, float)) and not isinstance(timestamp, bool):
        try:
            return datetime.fromtimestamp(float(timestamp), UTC).astimezone(reference.tzinfo)
        except (OSError, OverflowError, ValueError):
            pass
    value = point.get("time")
    if not isinstance(value, str) or not value:
        return None
    for date_format in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, date_format).replace(tzinfo=reference.tzinfo)
        except ValueError:
            pass
    for date_format in ("%m-%d %H:%M:%S", "%m-%d %H:%M"):
        try:
            partial = datetime.strptime(value, date_format)
        except ValueError:
            continue
        candidates = [
            partial.replace(year=reference.year + year_delta, tzinfo=reference.tzinfo)
            for year_delta in (-1, 0, 1)
        ]
        return min(candidates, key=lambda candidate: abs(candidate - reference))
    return None


def _filtered_document(
    document: dict[str, Any],
    *,
    start: datetime,
    finish: datetime,
    include_start: bool,
    include_finish: bool,
    reference: datetime,
) -> tuple[dict[str, Any], int]:
    filtered = deepcopy(document)
    data = filtered.get("data")
    if isinstance(data, list):
        container = filtered
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        container = data
    else:
        raise ValueError("DUT resource response omitted a supported time-series data array")
    points: list[Any] = []
    for point in container["data"]:
        if not isinstance(point, dict):
            continue
        captured_at = _point_time(point, reference)
        if captured_at is None:
            continue
        after_start = captured_at >= start if include_start else captured_at > start
        before_finish = captured_at <= finish if include_finish else captured_at < finish
        if after_start and before_finish:
            points.append(point)
    container["data"] = points
    return filtered, len(points)


def encrypt_login_password(password: str) -> str:
    encoded = password.encode("utf-16-le", errors="surrogatepass")
    code_units = [encoded[index] | encoded[index + 1] << 8 for index in range(0, len(encoded), 2)]
    code_units.extend([0] * ((-len(code_units)) % _RSA_CHUNK_SIZE))
    blocks: list[str] = []
    for offset in range(0, len(code_units), _RSA_CHUNK_SIZE):
        block = code_units[offset : offset + _RSA_CHUNK_SIZE]
        message = sum(unit << (8 * index) for index, unit in enumerate(block))
        hexadecimal = f"{pow(message, _RSA_EXPONENT, _RSA_MODULUS):x}"
        blocks.append(hexadecimal.zfill(max(4, ((len(hexadecimal) + 3) // 4) * 4)))
    return "".join(blocks)


def _json_object(response: httpx.Response, operation: str) -> dict[str, Any]:
    if response.is_redirect:
        raise RuntimeError(f"{operation} returned a redirect")
    response.raise_for_status()
    try:
        document = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{operation} did not return JSON") from exc
    if not isinstance(document, dict):
        raise RuntimeError(f"{operation} returned non-object JSON")
    return document


def _require_success(document: dict[str, Any], operation: str) -> None:
    code = document.get("code")
    if not isinstance(code, int) or isinstance(code, bool) or code not in _SUCCESS_CODES:
        raise RuntimeError(f"{operation} returned unsuccessful device code {code!r}")


def _session_value(document: dict[str, Any], *names: str) -> str:
    containers = [document]
    if isinstance(document.get("data"), dict):
        containers.append(document["data"])
    for container in containers:
        for name in names:
            value = container.get(name)
            if isinstance(value, str) and value:
                return value
    raise RuntimeError("DUT login response omitted session material")


class DutClient:
    def __init__(
        self,
        config: DutConfig,
        *,
        username: str,
        password: str,
        captcha_reader: Callable[[bytes, str], str],
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        if config.frontend is None:
            raise ValueError("frontend_api DUT collection requires dut.frontend")
        self._frontend = config.frontend
        self.username = username
        self._password = password
        self._captcha_reader = captcha_reader
        self._client = client or httpx.Client(
            verify=self._frontend.verify_tls,
            follow_redirects=False,
            timeout=httpx.Timeout(30.0, read=120.0),
            trust_env=False,
        )
        self._owns_client = client is None
        self._api_key: str | None = None
        self._security_key: str | None = None
        self._before: SupplementalSnapshot | None = None
        self._before_path: Path | None = None
        self._traffic_started_at: str | None = None
        self._traffic_finished_at: str | None = None
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        self._warnings: list[str] = []

    def _url(self, path: str) -> str:
        return f"{self._frontend.endpoint}{path}"

    def authenticate(self) -> None:
        _json_object(
            self._client.post(self._url("/api/system/account/login/checkLoginAuth")),
            "DUT login preflight",
        )
        _json_object(
            self._client.post(
                self._url("/api/system/account/login/checkLoginAuth"),
                headers={"authType": "web"},
            ),
            "DUT web login preflight",
        )
        challenge = self._client.get(
            self._url("/api/system/account/code"),
            params={"num": secrets.randbelow(10) + 1},
        )
        if challenge.is_redirect:
            raise RuntimeError("DUT CAPTCHA returned a redirect")
        challenge.raise_for_status()
        media_type = challenge.headers.get("content-type", "application/octet-stream").split(
            ";", 1
        )[0]
        if not media_type.casefold().startswith("image/"):
            raise RuntimeError("DUT CAPTCHA did not return an image")
        captcha = self._captcha_reader(challenge.content, media_type).strip()
        if not captcha:
            raise RuntimeError("DUT CAPTCHA must not be empty")
        response = self._client.post(
            self._url("/api/system/account/login/login"),
            headers={"authType": "web"},
            json={
                "username": self.username,
                "password": encrypt_login_password(self._password),
                "vcode": captcha,
                "lang": "zh_CN",
                "guid": str(uuid4()),
            },
        )
        document = _json_object(response, "DUT login")
        _require_success(document, "DUT login")
        self._api_key = _session_value(document, "api_key", "apiKey")
        self._security_key = _session_value(document, "security_key", "securityKey")
        self._password = ""

    @staticmethod
    def _timestamp_ms() -> int:
        return time.time_ns() // 1_000_000

    def _headers(self, path: str, timestamp_ms: int) -> dict[str, str]:
        if not self._api_key or not self._security_key:
            raise RuntimeError("DUT client is not authenticated")
        signing_text = (
            f"security-key:{self._security_key};api-key:{self._api_key};"
            f"time:{timestamp_ms};rest-uri:{path};data:"
        )
        return {
            "Accept": "application/json, text/plain, */*",
            "apikey": self._api_key,
            "authType": "web",
            "sign": hashlib.sha256(signing_text.encode()).hexdigest(),
            "time": str(timestamp_ms),
        }

    def _get(
        self,
        path: str,
        parameters: dict[str, str | int] | None = None,
        *,
        include_period: bool = False,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._frontend.read_attempts):
            timestamp = self._timestamp_ms()
            query: dict[str, str | int] = {"t": timestamp}
            if include_period and self._frontend.period is not None:
                query["period"] = self._frontend.period
            if parameters:
                query.update(parameters)
            try:
                response = self._client.get(
                    self._url(path), params=query, headers=self._headers(path, timestamp)
                )
                if (
                    response.status_code in _RETRYABLE_STATUS
                    and attempt + 1 < self._frontend.read_attempts
                ):
                    time.sleep(self._frontend.read_retry_backoff_seconds * (2**attempt))
                    continue
                document = _json_object(response, f"DUT GET {path}")
                _require_success(document, f"DUT GET {path}")
                return document
            except (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt + 1 >= self._frontend.read_attempts:
                    raise
                time.sleep(self._frontend.read_retry_backoff_seconds * (2**attempt))
        if last_error:
            raise last_error
        raise RuntimeError(f"DUT GET {path} exhausted retries")

    def keepalive(self) -> None:
        self._get("/api/dashboards/system/systemInfo")

    def collect_monitoring_window(
        self,
        traffic_started_at: str,
        traffic_finished_at: str,
        before: SupplementalSnapshot,
        after: SupplementalSnapshot,
    ) -> tuple[ResourceObservation, ...]:
        started = datetime.now(UTC).isoformat()
        documents: dict[str, Any] = {}
        read_errors: dict[str, str] = {}
        endpoints = {
            "cpu": "/api/dashboards/system/cpu",
            "memory": "/api/dashboards/system/mem",
            "new_sessions": "/api/dashboards/system/newSess",
            "concurrent_sessions": "/api/dashboards/system/concurrentSess",
        }
        for name, path in endpoints.items():
            try:
                documents[name] = self._get(path, include_period=True)
            except Exception as exc:
                read_errors[name] = str(exc)
        traffic_documents: dict[str, Any] = {}
        for interface in self.config.interfaces:
            try:
                traffic_documents[interface] = self._get(
                    "/api/dashboards/system/traffic",
                    {"interface": interface},
                    include_period=True,
                )
            except Exception as exc:
                read_errors[f"traffic.{interface}"] = str(exc)

        finished = datetime.now(UTC).isoformat()
        offset, device_reference = _clock_offset(before, after)
        device_capture = (
            _aware_datetime(finished, "DUT monitoring finished_at").astimezone(UTC) + offset
        ).astimezone(device_reference.tzinfo)
        local_start = _aware_datetime(traffic_started_at, "traffic_started_at")
        local_finish = _aware_datetime(traffic_finished_at, "traffic_finished_at")
        if local_finish <= local_start:
            raise ValueError("traffic_finished_at must be later than traffic_started_at")
        device_start = (local_start.astimezone(UTC) + offset).astimezone(device_reference.tzinfo)
        device_finish = (local_finish.astimezone(UTC) + offset).astimezone(device_reference.tzinfo)
        baseline_start = device_start - timedelta(seconds=self._frontend.baseline_seconds)
        phase_windows = (
            (ObservationPhase.BASELINE, baseline_start, device_start, True, False),
            (ObservationPhase.DURING, device_start, device_finish, True, True),
            (ObservationPhase.RECOVERY, device_finish, device_capture, False, True),
        )
        observations: list[ResourceObservation] = []
        for phase, window_start, window_finish, include_start, include_finish in phase_windows:
            resources: dict[str, Any] = {}
            errors = dict(read_errors)
            for name, document in documents.items():
                try:
                    filtered, count = _filtered_document(
                        document,
                        start=window_start,
                        finish=window_finish,
                        include_start=include_start,
                        include_finish=include_finish,
                        reference=device_capture,
                    )
                    resources[name] = filtered
                    if count == 0 and phase != ObservationPhase.RECOVERY:
                        errors[name] = f"no DUT samples in the {phase.value} window"
                except ValueError as exc:
                    errors[name] = str(exc)
            traffic: dict[str, Any] = {}
            for interface, document in traffic_documents.items():
                error_name = f"traffic.{interface}"
                try:
                    filtered, count = _filtered_document(
                        document,
                        start=window_start,
                        finish=window_finish,
                        include_start=include_start,
                        include_finish=include_finish,
                        reference=device_capture,
                    )
                    traffic[interface] = filtered
                    if count == 0 and phase != ObservationPhase.RECOVERY:
                        errors[error_name] = f"no DUT samples in the {phase.value} window"
                except ValueError as exc:
                    errors[error_name] = str(exc)
            resources["traffic"] = traffic
            observations.append(
                ResourceObservation(
                    phase=phase,
                    started_at=started,
                    finished_at=finished,
                    window_started_at=window_start.isoformat(),
                    window_finished_at=window_finish.isoformat(),
                    dut_clock_offset_seconds=offset.total_seconds(),
                    resources=resources,
                    errors=errors,
                )
            )
        return tuple(observations)

    def collect_supplemental(self) -> SupplementalSnapshot:
        values: dict[str, Any] = {}
        errors: dict[str, str] = {}
        endpoints = {
            "interfaces": "/api/dashboards/system/interface",
            "hardware": "/api/dashboards/system/hardware",
            "system": "/api/dashboards/system/systemInfo",
        }
        for name, path in endpoints.items():
            try:
                values[name] = self._get(path)
            except Exception as exc:
                errors[name] = str(exc)
        return SupplementalSnapshot(
            captured_at=datetime.now(UTC).isoformat(), values=values, errors=errors
        )

    def prepare_attempt(self, attempt_dir: Path) -> None:
        self._stop_keepalive()
        self._warnings = []
        self._traffic_started_at = None
        self._traffic_finished_at = None
        self._before = self.collect_supplemental()
        if not self._before.is_complete:
            raise RuntimeError("required pre-traffic DUT evidence is incomplete")
        self._before_path = attempt_dir / "dut-frontend-before.json"
        ArtifactStore.write_json(self._before_path, self._before)

    def traffic_started(self, started_at: str) -> None:
        if self._before is None:
            raise RuntimeError("DUT Attempt was not prepared")
        self._traffic_started_at = started_at
        self._traffic_finished_at = None
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            name="dut-frontend-keepalive",
            daemon=True,
        )
        self._keepalive_thread.start()

    def restore_attempt(
        self,
        attempt_dir: Path,
        started_at: str,
        finished_at: str | None,
    ) -> None:
        if self._before is not None and self._traffic_started_at == started_at:
            if finished_at is not None and self._traffic_finished_at is None:
                self.traffic_finished(finished_at)
            return
        self._stop_keepalive()
        self._before_path = attempt_dir / "dut-frontend-before.json"
        if not self._before_path.is_file():
            raise RuntimeError("resumed frontend DUT Attempt omitted its pre-traffic snapshot")
        self._before = SupplementalSnapshot.model_validate(
            ArtifactStore.read_json(self._before_path)
        )
        self._warnings = []
        self._traffic_started_at = started_at
        self._traffic_finished_at = finished_at
        if finished_at is None:
            self.traffic_started(started_at)

    def _keepalive_loop(self) -> None:
        while not self._keepalive_stop.wait(self._frontend.keepalive_interval_seconds):
            try:
                self.keepalive()
            except Exception as exc:
                self._warnings.append(f"DUT keepalive failed: {exc}")

    def _stop_keepalive(self) -> None:
        self._keepalive_stop.set()
        thread = self._keepalive_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._keepalive_thread = None

    def traffic_finished(self, finished_at: str) -> None:
        self._traffic_finished_at = finished_at
        self._stop_keepalive()

    def finalize_attempt(self) -> DutCaptureResult:
        started_at = self._traffic_started_at
        finished_at = self._traffic_finished_at
        before = self._before
        if started_at is None or finished_at is None or before is None:
            raise RuntimeError("DUT traffic window is incomplete")
        time.sleep(self._frontend.cooldown_seconds)
        after = self.collect_supplemental()
        if not after.is_complete:
            raise RuntimeError("required post-traffic DUT evidence is incomplete")
        observations = self.collect_monitoring_window(started_at, finished_at, before, after)
        compact = DutObservations.from_resource_observations(observations)
        evidence = FrontendDutEvidence(
            endpoint=self._frontend.endpoint,
            interfaces=self.config.interfaces,
            traffic_started_at=started_at,
            traffic_finished_at=finished_at,
            observations=compact,
            before=before,
            after=after,
            warnings=tuple(self._warnings),
        )
        return DutCaptureResult(evidence=evidence, warnings=tuple(self._warnings))

    def close(self) -> None:
        self._stop_keepalive()
        self._api_key = None
        self._security_key = None
        self._password = ""
        if self._owns_client:
            self._client.close()
