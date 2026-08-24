"""Read-only DUT authentication and monitoring adapter."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from bps_agent.models import DutConfig, ObservationPhase, ResourceObservation, SupplementalSnapshot

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
        self.username = username
        self._password = password
        self._captcha_reader = captcha_reader
        self._client = client or httpx.Client(
            verify=config.verify_tls,
            follow_redirects=False,
            timeout=httpx.Timeout(30.0, read=120.0),
            trust_env=False,
        )
        self._owns_client = client is None
        self._api_key: str | None = None
        self._security_key: str | None = None

    def _url(self, path: str) -> str:
        return f"{self.config.endpoint}{path}"

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

    def _get(self, path: str, parameters: dict[str, str | int] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.config.read_attempts):
            timestamp = self._timestamp_ms()
            query: dict[str, str | int] = {"t": timestamp}
            if self.config.period is not None:
                query["period"] = self.config.period
            if parameters:
                query.update(parameters)
            try:
                response = self._client.get(
                    self._url(path), params=query, headers=self._headers(path, timestamp)
                )
                if (
                    response.status_code in _RETRYABLE_STATUS
                    and attempt + 1 < self.config.read_attempts
                ):
                    time.sleep(self.config.read_retry_backoff_seconds * (2**attempt))
                    continue
                document = _json_object(response, f"DUT GET {path}")
                _require_success(document, f"DUT GET {path}")
                return document
            except (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt + 1 >= self.config.read_attempts:
                    raise
                time.sleep(self.config.read_retry_backoff_seconds * (2**attempt))
        if last_error:
            raise last_error
        raise RuntimeError(f"DUT GET {path} exhausted retries")

    def collect_resources(self, phase: str) -> ResourceObservation:
        observation_phase = ObservationPhase(phase)
        started = datetime.now(UTC).isoformat()
        resources: dict[str, Any] = {}
        errors: dict[str, str] = {}
        endpoints = {
            "cpu": "/api/dashboards/system/cpu",
            "memory": "/api/dashboards/system/mem",
            "new_sessions": "/api/dashboards/system/newSess",
            "concurrent_sessions": "/api/dashboards/system/concurrentSess",
        }
        for name, path in endpoints.items():
            try:
                resources[name] = self._get(path)
            except Exception as exc:
                errors[name] = str(exc)
        traffic: dict[str, Any] = {}
        for interface in self.config.interfaces:
            try:
                traffic[interface] = self._get(
                    "/api/dashboards/system/traffic", {"interface": interface}
                )
            except Exception as exc:
                errors[f"traffic.{interface}"] = str(exc)
        resources["traffic"] = traffic
        return ResourceObservation(
            phase=observation_phase,
            started_at=started,
            finished_at=datetime.now(UTC).isoformat(),
            resources=resources,
            errors=errors,
        )

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

    def close(self) -> None:
        self._api_key = None
        self._security_key = None
        self._password = ""
        if self._owns_client:
            self._client.close()
