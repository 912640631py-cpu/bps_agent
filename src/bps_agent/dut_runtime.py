"""Single registry for DUT Collection Method requirements and adapter construction."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bps_agent.adapters.dut import DutClient
from bps_agent.adapters.dut_backend import DutBackendCollector
from bps_agent.credentials import CredentialRequirement
from bps_agent.models import DutCollectionMethod, DutConfig
from bps_agent.ports import DutPort

LOGGER = logging.getLogger(__name__)

CaptchaReader = Callable[[bytes, str], str]
DutAdapterBuilder = Callable[[DutConfig, dict[str, str], CaptchaReader], DutPort]


@dataclass(frozen=True)
class DutRuntimeSpec:
    method: DutCollectionMethod
    configuration_name: Literal["backend", "frontend"]
    credential_requirements: tuple[CredentialRequirement, ...]
    adapter_builder: DutAdapterBuilder
    startup_message: str

    def validate_config(self, config: DutConfig) -> None:
        if getattr(config, self.configuration_name) is None:
            raise ValueError(
                f"{self.method.value} DUT collection requires dut.{self.configuration_name}"
            )

    def build_adapter(
        self,
        config: DutConfig,
        credentials: dict[str, str],
        captcha_reader: CaptchaReader,
    ) -> DutPort:
        self.validate_config(config)
        return self.adapter_builder(config, credentials, captcha_reader)


def _build_backend(
    config: DutConfig,
    credentials: dict[str, str],
    _captcha_reader: CaptchaReader,
) -> DutPort:
    return DutBackendCollector(
        config,
        username=credentials["DUT_BACKEND_USERNAME"],
        password=credentials["DUT_BACKEND_PASSWORD"],
    )


def _build_frontend(
    config: DutConfig,
    credentials: dict[str, str],
    captcha_reader: CaptchaReader,
) -> DutPort:
    adapter = DutClient(
        config,
        username=credentials["DUT_FRONTEND_USERNAME"],
        password=credentials["DUT_FRONTEND_PASSWORD"],
        captcha_reader=captcha_reader,
    )
    adapter.authenticate()
    return adapter


_DUT_RUNTIME_SPECS = {
    DutCollectionMethod.BACKEND_SSH: DutRuntimeSpec(
        method=DutCollectionMethod.BACKEND_SSH,
        configuration_name="backend",
        credential_requirements=(
            CredentialRequirement("DUT_BACKEND_USERNAME", "DUT backend SSH username: "),
            CredentialRequirement(
                "DUT_BACKEND_PASSWORD", "DUT backend SSH password: ", secret=True
            ),
        ),
        adapter_builder=_build_backend,
        startup_message=(
            "DUT backend SSH collection enabled (host keys are not currently verified)."
        ),
    ),
    DutCollectionMethod.FRONTEND_API: DutRuntimeSpec(
        method=DutCollectionMethod.FRONTEND_API,
        configuration_name="frontend",
        credential_requirements=(
            CredentialRequirement("DUT_FRONTEND_USERNAME", "DUT frontend username: "),
            CredentialRequirement(
                "DUT_FRONTEND_PASSWORD", "DUT frontend password: ", secret=True
            ),
        ),
        adapter_builder=_build_frontend,
        startup_message="Authenticating to DUT (CAPTCHA required)...",
    ),
}


def dut_runtime_spec(method: DutCollectionMethod) -> DutRuntimeSpec:
    return _DUT_RUNTIME_SPECS[method]


def read_captcha(image: bytes, media_type: str) -> str:
    suffix = {
        "image/gif": ".gif",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(media_type.casefold(), ".img")
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="dut-captcha-", suffix=suffix, delete=False
        ) as handle:
            handle.write(image)
            path = Path(handle.name)
        print(f"DUT CAPTCHA image: {path}")
        try:
            subprocess.Popen(
                ["explorer.exe", f"/select,{path.resolve()}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            LOGGER.warning("Could not open CAPTCHA image automatically: %s", exc)
        value = input("DUT CAPTCHA: ").strip()
        if not value:
            raise ValueError("DUT CAPTCHA must not be empty")
        return value
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
