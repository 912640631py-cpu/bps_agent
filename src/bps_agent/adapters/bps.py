"""Keysight BreakingPoint 9.22 adapter."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from bps_agent.models import BpsConfig, RunCompletion

LOGGER = logging.getLogger(__name__)

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_RETRYABLE_REPORT_STATUSES = {404, 409, 500, 503}
_RETRYABLE_PORT_RELEASE_STATUSES = {400, 409, 423, 500, 502, 503, 504}
_EXPLICIT_TERMINAL_STATES = {
    "aborted",
    "complete",
    "completed",
    "done",
    "failed",
    "finished",
    "passed",
    "stopped",
}


class PortOccupiedError(RuntimeError):
    pass


class PortReleaseError(RuntimeError):
    pass


class BpsProtocolError(RuntimeError):
    pass


def _normalize_list(payload: Any, names: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    if isinstance(payload, dict):
        for name in names:
            value = payload.get(name)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
        if any(key in payload for key in ("id", "runid", "runId")):
            return [payload]
        object_values = [
            value
            for value in payload.values()
            if isinstance(value, dict) and any(key in value for key in ("id", "runid", "runId"))
        ]
        if object_values:
            return object_values
        if not payload:
            return []
    if payload is None:
        return []
    raise BpsProtocolError("BPS returned an unrecognized list document")


def _shared_component_settings(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise BpsProtocolError("BPS shared component settings response is not an object")
    result = payload.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError as exc:
            raise BpsProtocolError(
                "BPS shared component settings result is not valid JSON"
            ) from exc
    if not isinstance(result, dict):
        raise BpsProtocolError("BPS shared component settings result is not an object")
    settings = result.get("sharedComponentSettings")
    if not isinstance(settings, list) or not all(isinstance(item, dict) for item in settings):
        raise BpsProtocolError("BPS template omitted sharedComponentSettings")
    return result


def _original_total_bandwidth_mbps(settings: dict[str, Any]) -> float:
    matches = [
        item
        for item in settings["sharedComponentSettings"]
        if str(item.get("name", "")).casefold() == "totalbandwidth"
    ]
    if len(matches) != 1:
        raise BpsProtocolError(
            "BPS template must contain exactly one totalBandwidth shared setting"
        )
    setting = matches[0]
    if setting.get("enabled") is False:
        raise BpsProtocolError("BPS template totalBandwidth shared setting is disabled")
    try:
        value = float(setting["originalValue"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BpsProtocolError("BPS template totalBandwidth originalValue is not numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise BpsProtocolError(
            "BPS template totalBandwidth originalValue must be a positive finite number"
        )
    return value


def _extract_run_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise BpsProtocolError("BPS run response is not a JSON object")
    for key in ("runid", "runId", "id"):
        value = payload.get(key)
        if value is not None:
            run_id = str(value)
            if _RUN_ID.fullmatch(run_id):
                return run_id
            raise BpsProtocolError("BPS returned an unsafe run ID")
    raise BpsProtocolError("BPS run response omitted a run ID")


def _run_aliases(run_id: str) -> set[str]:
    if run_id.startswith("TEST-"):
        return {run_id, run_id[5:]}
    return {run_id, f"TEST-{run_id}"}


def _is_explicit_terminal_state(details: dict[str, Any]) -> bool:
    completed = details.get("completed")
    if completed is True or (isinstance(completed, str) and completed.casefold() == "true"):
        return True
    return any(
        isinstance(details.get(key), str)
        and str(details[key]).strip().casefold() in _EXPLICIT_TERMINAL_STATES
        for key in ("state", "phase", "status")
    )


class BpsClient:
    def __init__(
        self,
        config: BpsConfig,
        *,
        username: str,
        password: str,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.username = username
        self._password = password
        self._client = client or httpx.Client(
            verify=config.verify_tls,
            follow_redirects=False,
            timeout=httpx.Timeout(30.0, read=300.0),
            trust_env=False,
            headers={"Content-Type": "application/json"},
        )
        self._owns_client = client is None
        self._session_id: str | None = None
        self._api_key: str | None = None

    def _url(self, path: str) -> str:
        return f"{self.config.endpoint}{path}"

    @staticmethod
    def _origin(url: str) -> tuple[str, str, int]:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise BpsProtocolError(f"invalid BPS URL: {url!r}")
        if parsed.username is not None or parsed.password is not None:
            raise BpsProtocolError("BPS URL must not contain credentials")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return parsed.scheme, parsed.hostname.casefold(), port

    def _request(self, method: str, path_or_url: str, **kwargs: Any) -> httpx.Response:
        url = path_or_url if path_or_url.startswith("http") else self._url(path_or_url)
        if self._origin(url) != self._origin(self.config.endpoint):
            raise BpsProtocolError("refusing to send BPS credentials to another origin")
        response = self._client.request(method, url, follow_redirects=False, **kwargs)
        if response.is_redirect:
            raise BpsProtocolError(
                f"unexpected BPS API redirect: {response.status_code} "
                f"{response.headers.get('location', '<missing>')}"
            )
        return response

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        if not response.content:
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

    @classmethod
    def _checked(cls, response: httpx.Response) -> Any:
        response.raise_for_status()
        return cls._decode(response)

    def authenticate(self) -> None:
        auth = self._checked(
            self._request(
                "POST",
                "/bps/api/v1/auth/session",
                json={"username": self.username, "password": self._password},
                timeout=30,
            )
        )
        if not isinstance(auth, dict) or not auth.get("sessionId") or not auth.get("apiKey"):
            raise BpsProtocolError("BPS authentication response omitted session material")
        self._session_id = str(auth["sessionId"])
        self._api_key = str(auth["apiKey"])
        self._client.headers.update({"sessionId": self._session_id, "X-API-KEY": self._api_key})
        self._checked(
            self._request(
                "POST",
                "/bps/api/v2/core/auth/login",
                json={
                    "username": self.username,
                    "password": self._password,
                    "sessionId": self._session_id,
                },
                timeout=30,
            )
        )

    def find_template(self, name: str) -> dict[str, Any]:
        payload = self._checked(
            self._request(
                "POST",
                "/bps/api/v2/core/testmodel/operations/search",
                json={
                    "searchString": name,
                    "limit": 20,
                    "sort": "name",
                    "sortorder": "ascending",
                },
                timeout=30,
            )
        )
        candidates = _normalize_list(payload, ("items", "results", "data", "models"))
        exact = [item for item in candidates if str(item.get("name", "")) == name]
        if len(exact) != 1:
            raise BpsProtocolError(
                f"expected exactly one BPS template named {name!r}, found {len(exact)}"
            )
        settings = _shared_component_settings(
            self._checked(
                self._request(
                    "POST",
                    "/api/v1/bps/tests/operations/getSharedComponentSettings",
                    json={"modelName": name},
                    timeout=60,
                )
            )
        )
        return {
            **exact[0],
            "sharedComponentSettings": settings["sharedComponentSettings"],
            "totalBandwidthMbps": _original_total_bandwidth_mbps(settings),
        }

    def reserve_ports(self) -> None:
        reservation = [
            {
                "slot": self.config.slot,
                "port": port,
                "group": self.config.group,
                "capture": False,
            }
            for port in self.config.ports
        ]
        response = self._request(
            "POST",
            "/bps/api/v2/core/topology/operations/reserve",
            json={"reservation": reservation, "force": False},
            timeout=60,
        )
        if response.status_code in {400, 409, 423}:
            raise PortOccupiedError(f"BPS ports are unavailable (HTTP {response.status_code})")
        response.raise_for_status()

    def release_ports(self) -> None:
        attempts = self.config.port_release_attempts
        last_failure = "unknown failure"
        last_cause: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._request(
                    "POST",
                    "/bps/api/v2/core/topology/operations/unreserve",
                    json={
                        "unreservation": [
                            {"slot": self.config.slot, "port": port} for port in self.config.ports
                        ]
                    },
                    timeout=60,
                )
            except httpx.TransportError as exc:
                last_failure = type(exc).__name__
                last_cause = exc
            else:
                last_cause = None
                if response.is_success:
                    return
                last_failure = f"HTTP {response.status_code}"
                if response.status_code not in _RETRYABLE_PORT_RELEASE_STATUSES:
                    raise PortReleaseError(
                        f"BPS rejected port release without retry ({last_failure})"
                    )
            if attempt < attempts:
                time.sleep(self.config.port_release_retry_backoff_seconds)
        message = f"BPS port release failed after {attempts} attempts ({last_failure})"
        if last_cause is not None:
            raise PortReleaseError(message) from last_cause
        raise PortReleaseError(message)

    def set_total_bandwidth(self, percentage: float) -> None:
        numeric_percentage = float(percentage)
        if not 0 < numeric_percentage <= 100:
            raise ValueError("BPS total bandwidth percentage must be between 0 and 100")
        param_value: int | float = (
            int(numeric_percentage) if numeric_percentage.is_integer() else numeric_percentage
        )
        self._checked(
            self._request(
                "POST",
                "/api/v1/bps/tests/operations/setSharedComponentSettings",
                json={
                    "modelName": self.config.template,
                    "sharedComponentSettings": [
                        {
                            "paramName": "totalBandwidth",
                            "paramValue": param_value,
                        }
                    ],
                },
                timeout=60,
            )
        )

    def start_run(self) -> str:
        payload = self._checked(
            self._request(
                "POST",
                "/bps/api/v2/core/testmodel/operations/run",
                json={
                    "modelname": self.config.template,
                    "group": self.config.group,
                    "allowMalware": self.config.allow_malware,
                },
                timeout=60,
            )
        )
        return _extract_run_id(payload)

    def _running_test(self, run_id: str) -> dict[str, Any] | None:
        payload = self._checked(
            self._request("GET", "/bps/api/v2/core/topology/runningTest", timeout=30)
        )
        aliases = _run_aliases(run_id)
        for item in _normalize_list(
            payload, ("items", "results", "data", "runningTest", "runningTests")
        ):
            if any(str(item.get(key)) in aliases for key in ("id", "runid", "runId")):
                return item
        return None

    def _report_contents(self, run_id: str) -> Any:
        response = self._request(
            "POST",
            "/bps/api/v2/core/reports/operations/getReportContents",
            json={"runid": run_id, "getTableOfContents": True},
            timeout=60,
        )
        if response.status_code in _RETRYABLE_REPORT_STATUSES:
            return None
        return self._checked(response)

    def wait_for_completion(self, run_id: str, on_poll: Callable[[], None]) -> RunCompletion:
        deadline = time.monotonic() + self.config.run_timeout_seconds
        registration_deadline = time.monotonic() + self.config.registration_grace_seconds
        seen_running = False
        last_details: dict[str, Any] = {}
        while True:
            running = self._running_test(run_id)
            on_poll()
            if running is not None:
                seen_running = True
                last_details = running
                if _is_explicit_terminal_state(running):
                    return RunCompletion(
                        terminal=True,
                        details={
                            "last_running_state": last_details,
                            "completion": "explicit-terminal-state",
                        },
                    )
            else:
                if self._report_contents(run_id) is not None:
                    completion = (
                        "running-test-absent-and-report-ready"
                        if seen_running
                        else "report-ready-before-registration-observed"
                    )
                    return RunCompletion(
                        terminal=True,
                        details={
                            "last_running_state": last_details,
                            "completion": completion,
                        },
                    )
                if not seen_running and time.monotonic() < registration_deadline:
                    pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for BPS run {run_id}")
            time.sleep(min(self.config.poll_interval_seconds, remaining))

    def wait_for_report(self, run_id: str) -> Any:
        for attempt in range(self.config.report_attempts):
            contents = self._report_contents(run_id)
            if contents is not None:
                return contents
            if attempt + 1 < self.config.report_attempts:
                time.sleep(self.config.report_poll_interval_seconds)
        raise TimeoutError(f"BPS report for run {run_id} did not become ready")

    def _download(
        self,
        reference: str,
        destination: Path,
        *,
        max_bytes: int,
        timeout_seconds: float,
        require_pdf: bool,
    ) -> Path:
        url = urljoin(f"{self.config.endpoint}/", reference)
        for _ in range(6):
            if self._origin(url) != self._origin(self.config.endpoint):
                raise BpsProtocolError("BPS report download escaped the authenticated origin")
            with self._client.stream(
                "GET",
                url,
                follow_redirects=False,
                timeout=timeout_seconds,
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise BpsProtocolError("BPS report redirect omitted Location")
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                if response.headers.get("content-type", "").casefold().startswith("text/html"):
                    raise BpsProtocolError("BPS report download unexpectedly returned HTML")
                declared = response.headers.get("content-length")
                if declared and declared.isdecimal() and int(declared) > max_bytes:
                    raise BpsProtocolError("BPS report exceeds configured size limit")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        prefix=f".{destination.name}.",
                        suffix=".part",
                        dir=destination.parent,
                        delete=False,
                    ) as handle:
                        temporary = Path(handle.name)
                        size = 0
                        prefix = bytearray()
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise BpsProtocolError("BPS report exceeds configured size limit")
                            if len(prefix) < 1024:
                                prefix.extend(chunk[: 1024 - len(prefix)])
                            handle.write(chunk)
                        if size == 0:
                            raise BpsProtocolError("BPS report download was empty")
                        if require_pdf and b"%PDF-" not in prefix:
                            raise BpsProtocolError("BPS PDF export did not contain a PDF signature")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, destination)
                    temporary = None
                    return destination
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        raise BpsProtocolError("too many BPS report redirects")

    def _export_report(
        self,
        run_id: str,
        destination: Path,
        section_ids: tuple[str, ...],
        *,
        report_type: str,
        max_bytes: int,
        timeout_seconds: float,
        include_subsections: bool,
    ) -> Path:
        response = self._request(
            "POST",
            "/bps/api/v2/core/reports/operations/exportReport",
            json={
                "filepath": str(destination),
                "runid": run_id,
                "reportType": report_type,
                "sectionIds": ",".join(section_ids),
                "includeSubsections": include_subsections,
                "dataType": self.config.report_data_type,
            },
            timeout=timeout_seconds,
        )
        payload = self._checked(response)
        reference: str | None = None
        if isinstance(payload, str):
            reference = payload.strip().strip('"')
        elif isinstance(payload, dict):
            for key in ("url", "downloadUrl", "downloadURL", "download", "href", "path", "file"):
                if isinstance(payload.get(key), str) and payload[key].strip():
                    reference = payload[key].strip()
                    break
        if not reference:
            raise BpsProtocolError("BPS export response omitted a download reference")
        return self._download(
            reference,
            destination,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
            require_pdf=report_type == "PDF",
        )

    def export_report(
        self,
        run_id: str,
        destination: Path,
        section_ids: tuple[str, ...],
    ) -> Path:
        return self._export_report(
            run_id,
            destination,
            section_ids,
            report_type=self.config.report_type,
            max_bytes=self.config.max_report_bytes,
            timeout_seconds=300,
            include_subsections=False,
        )

    def export_full_report_pdf(
        self,
        run_id: str,
        destination: Path,
        section_ids: tuple[str, ...],
    ) -> Path:
        return self._export_report(
            run_id,
            destination,
            section_ids,
            report_type="PDF",
            max_bytes=self.config.max_pdf_report_bytes,
            timeout_seconds=self.config.pdf_report_timeout_seconds,
            include_subsections=True,
        )

    def schedule_full_report_pdf(
        self,
        run_id: str,
        destination: Path,
        section_ids: tuple[str, ...],
    ) -> None:
        """Start an isolated best-effort PDF export outside the Evaluation critical path."""

        if not self._session_id or not self._api_key:
            LOGGER.warning(
                "Optional full PDF report export was not scheduled: BPS is not logged in"
            )
            return
        config = self.config
        username = self.username
        headers = dict(self._client.headers)
        cookies = dict(self._client.cookies.items())

        def export() -> None:
            try:
                with httpx.Client(
                    verify=config.verify_tls,
                    follow_redirects=False,
                    timeout=httpx.Timeout(30.0, read=config.pdf_report_timeout_seconds),
                    trust_env=False,
                    headers=headers,
                    cookies=cookies,
                ) as client:
                    worker = BpsClient(
                        config,
                        username=username,
                        password="",
                        client=client,
                    )
                    worker.export_full_report_pdf(run_id, destination, section_ids)
            except Exception as exc:
                LOGGER.warning("Optional full PDF report export failed: %s", exc)

        Thread(
            target=export,
            name=f"bps-full-pdf-{run_id}",
            daemon=True,
        ).start()

    def stop_run(self, run_id: str) -> None:
        self._checked(
            self._request(
                "POST",
                "/bps/api/v2/core/testmodel/operations/stopRun",
                json={"runid": run_id},
                timeout=30,
            )
        )

    def close(self) -> None:
        if self._session_id:
            try:
                with suppress(Exception):
                    self._request(
                        "POST",
                        "/bps/api/v2/core/auth/logout",
                        json={
                            "username": self.username,
                            "password": self._password,
                            "sessionId": self._session_id,
                        },
                        timeout=30,
                    )
                with suppress(Exception):
                    self._request("DELETE", "/bps/api/v1/auth/session", timeout=30)
            finally:
                self._session_id = None
                self._api_key = None
                self._client.headers.pop("sessionId", None)
                self._client.headers.pop("X-API-KEY", None)
        self._password = ""
        if self._owns_client:
            self._client.close()
