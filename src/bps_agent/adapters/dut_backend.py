"""Continuous DUT backend metrics collection over SSH."""

from __future__ import annotations

import base64
import csv
import io
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko

from bps_agent.artifacts import ArtifactStore
from bps_agent.models import (
    BackendDutEvidence,
    BackendDutTarget,
    DutCaptureResult,
    DutConfig,
)

_REMOTE_API_DIR = "/opt/nsfocus/web/www/api"
_REMOTE_SYSTEM_CHART = "Dashboard/SystemChart.php"
_MAX_RETRY_DELAY_SECONDS = 5.0


class DutBackendError(RuntimeError):
    """The DUT backend did not return a valid metrics snapshot."""


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def _php_program(interfaces: tuple[str, ...]) -> str:
    encoded_interfaces = base64.b64encode(
        json.dumps(interfaces, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    return f'''\
$output = file_get_contents("/opt/nsfocus/etc/timezone.conf");
if ($output !== false) {{
    $current_timezone = trim(str_replace("TZ=Etc/GMT", "", $output));
    date_default_timezone_set("Etc/GMT" . $current_timezone);
}}

set_include_path(get_include_path() . PATH_SEPARATOR . "/opt/nsfocus/web/www/api/lib");
require_once "{_REMOTE_SYSTEM_CHART}";

function latest_item($items) {{
    if (!is_array($items) || count($items) === 0) {{
        return null;
    }}
    return $items[count($items) - 1];
}}

$interfaces = json_decode(base64_decode("{encoded_interfaces}"), true);
$obj = new systemChart();
$cpu = $obj->cpuUtilization(1);
$mem = $obj->memUtilization(1);
$newSess = $obj->newSessions(1);
$concurrentSess = $obj->concurrentSessions(1);

$result = array(
    "collected_at" => date("c"),
    "cpu" => array(
        "latest" => latest_item(isset($cpu["data"]) ? $cpu["data"] : array()),
        "threshold" => isset($cpu["threshold"]) ? $cpu["threshold"] : null
    ),
    "memory" => array(
        "latest" => latest_item(isset($mem["data"]) ? $mem["data"] : array()),
        "threshold" => isset($mem["threshold"]) ? $mem["threshold"] : null
    ),
    "new_sessions" => latest_item($newSess),
    "concurrent_sessions" => latest_item($concurrentSess),
    "traffic" => array()
);

foreach ($interfaces as $interface) {{
    $traffic = $obj->trafficChart(1, $interface);
    $result["traffic"][$interface] = latest_item(
        isset($traffic["data"]) ? $traffic["data"] : array()
    );
}}

echo json_encode($result, JSON_UNESCAPED_SLASHES), PHP_EOL;
'''


def _shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _remote_command(interfaces: tuple[str, ...]) -> str:
    return (
        f"cd {_shell_single_quote(_REMOTE_API_DIR)} && "
        f"php -r {_shell_single_quote(_php_program(interfaces))}"
    )


def _extract_json(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise DutBackendError("DUT backend returned empty stdout")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        try:
            document = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as line_error:
            raise DutBackendError(
                f"DUT backend returned invalid JSON: {text[:500]!r}"
            ) from line_error
    if not isinstance(document, dict):
        raise DutBackendError("DUT backend JSON root is not an object")
    return document


def _validate_snapshot(document: dict[str, Any], interfaces: tuple[str, ...]) -> None:
    required = {
        "collected_at",
        "cpu",
        "memory",
        "new_sessions",
        "concurrent_sessions",
        "traffic",
    }
    missing = sorted(required.difference(document))
    if missing:
        raise DutBackendError(f"DUT backend snapshot omitted fields: {', '.join(missing)}")
    traffic = document.get("traffic")
    if not isinstance(traffic, dict):
        raise DutBackendError("DUT backend traffic field is not an object")
    missing_interfaces = [interface for interface in interfaces if interface not in traffic]
    if missing_interfaces:
        raise DutBackendError(
            "DUT backend snapshot omitted interfaces: " + ", ".join(missing_interfaces)
        )


class DutSshBackendClient:
    def __init__(self, config: DutConfig, *, username: str, password: str) -> None:
        self.config = config
        self.username = username
        self._password = password
        self._client: paramiko.SSHClient | None = None

    def connect(self) -> None:
        self.close()
        backend = self.config.backend
        client = paramiko.SSHClient()
        # This temporary lab mode deliberately does not load or persist host keys.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=backend.host,
            port=backend.port,
            username=self.username,
            password=self._password,
            timeout=backend.connect_timeout_seconds,
            banner_timeout=backend.connect_timeout_seconds,
            auth_timeout=backend.connect_timeout_seconds,
            look_for_keys=False,
            allow_agent=False,
        )
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(15)
        self._client = client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def clear_credentials(self) -> None:
        self.close()
        self.username = ""
        self._password = ""

    def _ensure_connected(self) -> paramiko.SSHClient:
        if self._client is None:
            self.connect()
        assert self._client is not None
        transport = self._client.get_transport()
        if transport is None or not transport.is_active():
            self.connect()
        assert self._client is not None
        return self._client

    def read_snapshot(self, interfaces: tuple[str, ...]) -> dict[str, Any]:
        backend = self.config.backend
        command = _remote_command(interfaces)
        last_error: BaseException | None = None
        for attempt in range(backend.read_attempts):
            try:
                client = self._ensure_connected()
                stdin, stdout, stderr = client.exec_command(
                    command,
                    timeout=backend.command_timeout_seconds,
                    get_pty=False,
                )
                stdin.close()
                exit_status = stdout.channel.recv_exit_status()
                stdout_text = stdout.read().decode("utf-8", errors="replace")
                stderr_text = stderr.read().decode("utf-8", errors="replace").strip()
                if exit_status != 0:
                    detail = stderr_text or stdout_text.strip() or "no diagnostic output"
                    raise DutBackendError(
                        f"remote PHP collector exited with status {exit_status}: {detail[:1000]}"
                    )
                document = _extract_json(stdout_text)
                _validate_snapshot(document, interfaces)
                return document
            except (
                paramiko.SSHException,
                TimeoutError,
                OSError,
                EOFError,
                DutBackendError,
            ) as exc:
                last_error = exc
                self.close()
                if attempt + 1 < backend.read_attempts:
                    delay = min(
                        backend.read_retry_backoff_seconds * (2**attempt),
                        _MAX_RETRY_DELAY_SECONDS,
                    )
                    time.sleep(delay)
        assert last_error is not None
        raise DutBackendError(
            f"failed to collect DUT backend metrics: {last_error}"
        ) from last_error


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _elapsed_seconds(origin: str, current: Any) -> str:
    if not isinstance(current, str) or not origin:
        return ""
    try:
        elapsed = (datetime.fromisoformat(current) - datetime.fromisoformat(origin)).total_seconds()
    except ValueError:
        return ""
    if abs(elapsed) < 0.0005:
        elapsed = 0.0
    return f"{elapsed:.3f}".rstrip("0").rstrip(".")


def render_metrics_csv(samples: list[dict[str, Any]], interfaces: tuple[str, ...]) -> str:
    fixed_columns = [
        "sample_index",
        "time_origin",
        "elapsed_seconds",
        "cpu_mgt_percent",
        "cpu_data_percent",
        "cpu_mgt_threshold_percent",
        "cpu_data_threshold_percent",
        "memory_percent",
        "memory_threshold_percent",
        "new_sessions_count",
        "concurrent_sessions_count",
    ]
    interface_columns = [
        f"traffic[{interface}].{field}" for interface in interfaces for field in ("ibps", "obps")
    ]
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([*fixed_columns, *interface_columns])
    origin = str(samples[0].get("started_at") or "") if samples else ""
    for index, sample in enumerate(samples, start=1):
        resources = _object(sample.get("resources"))
        cpu = _object(resources.get("cpu"))
        cpu_latest = _object(cpu.get("latest"))
        cpu_threshold = _object(cpu.get("threshold"))
        memory = _object(resources.get("memory"))
        memory_latest = _object(memory.get("latest"))
        new_sessions = _object(resources.get("new_sessions"))
        concurrent_sessions = _object(resources.get("concurrent_sessions"))
        traffic = _object(resources.get("traffic"))
        row: list[Any] = [
            index,
            origin if index == 1 else "",
            _elapsed_seconds(origin, sample.get("started_at")),
            cpu_latest.get("mgt"),
            cpu_latest.get("data"),
            cpu_threshold.get("mgt"),
            cpu_threshold.get("data"),
            memory_latest.get("percent"),
            memory.get("threshold"),
            new_sessions.get("count"),
            concurrent_sessions.get("count"),
        ]
        for interface in interfaces:
            interface_traffic = _object(traffic.get(interface))
            row.extend(interface_traffic.get(field) for field in ("ibps", "obps"))
        writer.writerow(row)
    return output.getvalue()


class DutBackendCollector:
    def __init__(
        self,
        config: DutConfig,
        *,
        username: str,
        password: str,
        client: DutSshBackendClient | None = None,
    ) -> None:
        self.config = config
        self._client = client or DutSshBackendClient(
            config,
            username=username,
            password=password,
        )
        self._samples: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._attempt_dir: Path | None = None
        self._raw_path: Path | None = None
        self._csv_path: Path | None = None
        self._traffic_started_at: str | None = None
        self._traffic_finished_at: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def prepare_attempt(self, attempt_dir: Path) -> None:
        self._stop_worker()
        self._client.close()
        self._samples = []
        self._errors = []
        self._traffic_started_at = None
        self._traffic_finished_at = None
        self._attempt_dir = attempt_dir
        self._raw_path = attempt_dir / "dut-metrics.json"
        self._csv_path = attempt_dir / "dut-metrics.csv"
        self._client.connect()
        self._write_checkpoint()

    def traffic_started(self, started_at: str) -> None:
        if self._attempt_dir is None:
            raise RuntimeError("DUT backend Attempt was not prepared")
        self._traffic_started_at = started_at
        self._traffic_finished_at = None
        self._start_worker()

    def _start_worker(self) -> None:
        self._stop.clear()
        self._write_checkpoint()
        self._thread = threading.Thread(
            target=self._collect_loop,
            name="dut-backend-collector",
            daemon=True,
        )
        self._thread.start()

    def restore_attempt(
        self,
        attempt_dir: Path,
        started_at: str,
        finished_at: str | None,
    ) -> None:
        raw_path = attempt_dir / "dut-metrics.json"
        if self._traffic_started_at == started_at and self._raw_path == raw_path:
            if finished_at is not None and self._traffic_finished_at is None:
                self.traffic_finished(finished_at)
            return
        self._stop_worker()
        if not raw_path.is_file():
            raise RuntimeError("resumed backend DUT Attempt omitted dut-metrics.json")
        document = ArtifactStore.read_json(raw_path)
        if not isinstance(document, dict):
            raise RuntimeError("resumed backend DUT metrics artifact is not an object")
        target = _object(document.get("target"))
        if (
            target.get("host") != self.config.backend.host
            or target.get("port") != self.config.backend.port
        ):
            raise RuntimeError("resumed backend DUT target differs from the checkpoint")
        if tuple(document.get("interfaces", ())) != self.config.interfaces:
            raise RuntimeError("resumed backend DUT interfaces differ from the checkpoint")
        raw_samples = document.get("samples")
        raw_errors = document.get("errors")
        if not isinstance(raw_samples, list) or not isinstance(raw_errors, list):
            raise RuntimeError("resumed backend DUT artifact omitted samples or errors")
        self._attempt_dir = attempt_dir
        self._raw_path = raw_path
        self._csv_path = attempt_dir / "dut-metrics.csv"
        self._samples = [item for item in raw_samples if isinstance(item, dict)]
        self._errors = [item for item in raw_errors if isinstance(item, dict)]
        self._traffic_started_at = started_at
        self._traffic_finished_at = finished_at
        if finished_at is None:
            self._client.connect()
            self._start_worker()

    def _collect_loop(self) -> None:
        origin = time.monotonic()
        sample_number = 0
        while not self._stop.is_set():
            scheduled_at = origin + sample_number * self.config.backend.interval_seconds
            delay = scheduled_at - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                break
            started_at = _iso_now()
            try:
                snapshot = self._client.read_snapshot(self.config.interfaces)
                sample = {
                    "started_at": started_at,
                    "finished_at": _iso_now(),
                    "resources": snapshot,
                }
                with self._lock:
                    self._samples.append(sample)
                    self._write_checkpoint_locked()
            except Exception as exc:
                failure = {
                    "scheduled_at": started_at,
                    "started_at": started_at,
                    "finished_at": _iso_now(),
                    "error": str(exc),
                }
                with self._lock:
                    self._errors.append(failure)
                    self._write_checkpoint_locked()
            sample_number += 1

    def traffic_finished(self, finished_at: str) -> None:
        self._traffic_finished_at = finished_at
        self._stop_worker()
        self._write_checkpoint()

    def _stop_worker(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._thread = None

    def _result_document(self) -> dict[str, Any]:
        return {
            "target": {
                "host": self.config.backend.host,
                "port": self.config.backend.port,
                "transport": "ssh",
                "backend": "PHP Dashboard/SystemChart.php",
            },
            "interfaces": list(self.config.interfaces),
            "traffic_started_at": self._traffic_started_at,
            "traffic_finished_at": self._traffic_finished_at,
            "interval_seconds": self.config.backend.interval_seconds,
            "samples": list(self._samples),
            "errors": list(self._errors),
        }

    def _write_checkpoint(self) -> None:
        with self._lock:
            self._write_checkpoint_locked()

    def _write_checkpoint_locked(self) -> None:
        if self._raw_path is not None:
            ArtifactStore.write_json(self._raw_path, self._result_document())

    def finalize_attempt(self) -> DutCaptureResult:
        self._stop_worker()
        started_at = self._traffic_started_at
        finished_at = self._traffic_finished_at
        if started_at is None or finished_at is None:
            raise RuntimeError("DUT backend traffic window is incomplete")
        if self._raw_path is None or self._csv_path is None:
            raise RuntimeError("DUT backend artifact paths are unavailable")
        metrics_csv = render_metrics_csv(self._samples, self.config.interfaces)
        ArtifactStore.write_text(self._csv_path, metrics_csv)
        self._write_checkpoint()
        self._client.close()
        evidence = BackendDutEvidence(
            target=BackendDutTarget(
                host=self.config.backend.host,
                port=self.config.backend.port,
            ),
            interfaces=self.config.interfaces,
            traffic_started_at=started_at,
            traffic_finished_at=finished_at,
            interval_seconds=self.config.backend.interval_seconds,
            successful_sample_count=len(self._samples),
            failed_sample_count=len(self._errors),
            errors=tuple(self._errors),
            metrics_csv=metrics_csv,
        )
        warnings = tuple(
            f"DUT backend sample failed at {item['started_at']}: {item['error']}"
            for item in self._errors
        )
        return DutCaptureResult(
            evidence=evidence,
            raw_artifact_path=str(self._raw_path),
            csv_artifact_path=str(self._csv_path),
            warnings=warnings,
        )

    def close(self) -> None:
        self._stop_worker()
        self._client.clear_credentials()
