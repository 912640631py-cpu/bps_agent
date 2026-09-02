"""BPS report readiness, export, and authenticated download implementation."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from bps_agent.adapters.bps_protocol import BpsProtocolError, require_success_payload
from bps_agent.errors import BpsError, ErrorCode
from bps_agent.http_safety import require_same_origin
from bps_agent.models.config import BpsConfig

_RETRYABLE_REPORT_STATUSES = {404, 409, 500, 503}


class BpsReports:
    """Deep internal module for the complete report export workflow."""

    def __init__(
        self,
        config: BpsConfig,
        *,
        client: httpx.Client,
        request: Callable[..., httpx.Response],
    ) -> None:
        self._config = config
        self._client = client
        self._request = request

    def report_contents(self, run_id: str) -> Any | None:
        response = self._request(
            "POST",
            "/bps/api/v2/core/reports/operations/getReportContents",
            json={"runid": run_id, "getTableOfContents": True},
            timeout=60,
        )
        if response.status_code in _RETRYABLE_REPORT_STATUSES:
            return None
        try:
            return require_success_payload(response)
        except BpsProtocolError as exc:
            code = (
                ErrorCode.BPS_AUTH_FAILED
                if exc.code == ErrorCode.BPS_AUTH_FAILED.value
                else ErrorCode.BPS_REPORT_ERROR
            )
            raise BpsError(str(exc), code=code) from exc

    def wait_for_report(self, run_id: str) -> Any:
        for attempt in range(self._config.report_attempts):
            contents = self.report_contents(run_id)
            if contents is not None:
                return contents
            if attempt + 1 < self._config.report_attempts:
                time.sleep(self._config.report_poll_interval_seconds)
        raise BpsError(
            f"BPS report for run {run_id} did not become ready",
            code=ErrorCode.BPS_REPORT_ERROR,
        )

    def _download(
        self,
        reference: str,
        destination: Path,
        *,
        max_bytes: int,
        timeout_seconds: float,
        require_pdf: bool,
    ) -> Path:
        url = urljoin(f"{self._config.endpoint}/", reference)
        for _ in range(6):
            try:
                require_same_origin(url, self._config.endpoint)
            except ValueError as exc:
                raise BpsProtocolError(
                    "BPS report download escaped the authenticated origin",
                    code=ErrorCode.BPS_PROTOCOL_ERROR,
                ) from exc
            try:
                with self._client.stream(
                    "GET", url, follow_redirects=False, timeout=timeout_seconds
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise BpsProtocolError(
                                "BPS report redirect omitted Location",
                                code=ErrorCode.BPS_REPORT_ERROR,
                            )
                        url = urljoin(url, location)
                        continue
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise BpsError(
                            f"BPS report download failed with HTTP {response.status_code}",
                            code=(
                                ErrorCode.BPS_AUTH_FAILED
                                if response.status_code in {401, 403}
                                else ErrorCode.BPS_REPORT_ERROR
                            ),
                        ) from exc
                    if response.headers.get("content-type", "").casefold().startswith("text/html"):
                        raise BpsProtocolError(
                            "BPS report download unexpectedly returned HTML",
                            code=ErrorCode.BPS_REPORT_ERROR,
                        )
                    declared = response.headers.get("content-length")
                    if declared and declared.isdecimal() and int(declared) > max_bytes:
                        raise BpsProtocolError(
                            "BPS report exceeds configured size limit",
                            code=ErrorCode.BPS_REPORT_ERROR,
                        )
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
                                    raise BpsProtocolError(
                                        "BPS report exceeds configured size limit",
                                        code=ErrorCode.BPS_REPORT_ERROR,
                                    )
                                if len(prefix) < 1024:
                                    prefix.extend(chunk[: 1024 - len(prefix)])
                                handle.write(chunk)
                            if size == 0:
                                raise BpsProtocolError(
                                    "BPS report download was empty",
                                    code=ErrorCode.BPS_REPORT_ERROR,
                                )
                            if require_pdf and b"%PDF-" not in prefix:
                                raise BpsProtocolError(
                                    "BPS PDF export did not contain a PDF signature",
                                    code=ErrorCode.BPS_REPORT_ERROR,
                                )
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(temporary, destination)
                        temporary = None
                        return destination
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
            except httpx.TransportError as exc:
                raise BpsError(
                    "BPS report download endpoint is unreachable",
                    code=ErrorCode.BPS_UNREACHABLE,
                ) from exc
            except OSError as exc:
                raise BpsError(
                    "could not save the BPS report",
                    code=ErrorCode.BPS_REPORT_ERROR,
                    hint="Check the report destination and available disk space.",
                ) from exc
        raise BpsProtocolError(
            "too many BPS report redirects",
            code=ErrorCode.BPS_REPORT_ERROR,
        )

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
        try:
            payload = require_success_payload(
                self._request(
                    "POST",
                    "/bps/api/v2/core/reports/operations/exportReport",
                    json={
                        "filepath": str(destination),
                        "runid": run_id,
                        "reportType": report_type,
                        "sectionIds": ",".join(section_ids),
                        "includeSubsections": include_subsections,
                        "dataType": self._config.report_data_type,
                    },
                    timeout=timeout_seconds,
                )
            )
        except BpsProtocolError as exc:
            code = (
                ErrorCode.BPS_AUTH_FAILED
                if exc.code == ErrorCode.BPS_AUTH_FAILED.value
                else ErrorCode.BPS_REPORT_ERROR
            )
            raise BpsError(str(exc), code=code) from exc
        reference: str | None = None
        if isinstance(payload, str):
            reference = payload.strip().strip('"')
        elif isinstance(payload, dict):
            for key in ("url", "downloadUrl", "downloadURL", "download", "href", "path", "file"):
                if isinstance(payload.get(key), str) and payload[key].strip():
                    reference = payload[key].strip()
                    break
        if not reference:
            raise BpsProtocolError(
                "BPS export response omitted a download reference",
                code=ErrorCode.BPS_REPORT_ERROR,
            )
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
            report_type=self._config.report_type,
            max_bytes=self._config.max_report_bytes,
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
            max_bytes=self._config.max_pdf_report_bytes,
            timeout_seconds=self._config.pdf_report_timeout_seconds,
            include_subsections=True,
        )
