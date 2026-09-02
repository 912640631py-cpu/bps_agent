"""Single registry for DUT Collection Method requirements and adapter construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from bps_agent.adapters.dut import DutClient
from bps_agent.adapters.dut_backend import DutBackendCollector
from bps_agent.credentials import CredentialRequirement
from bps_agent.models.common import DutCollectionMethod
from bps_agent.models.config import DutConfig
from bps_agent.ports import DutPort

CaptchaReader = Callable[[bytes, str], str]
DutAdapterBuilder = Callable[[DutConfig, dict[str, str], CaptchaReader], DutPort]


@dataclass(frozen=True)
class DutRuntimeSpec:
    method: DutCollectionMethod
    credential_requirements: tuple[CredentialRequirement, ...]
    adapter_builder: DutAdapterBuilder

    def build_adapter(
        self,
        config: DutConfig,
        credentials: dict[str, str],
        captcha_reader: CaptchaReader,
    ) -> DutPort:
        return self.adapter_builder(config, credentials, captcha_reader)


def _build_backend(
    config: DutConfig,
    credentials: dict[str, str],
    _captcha_reader: CaptchaReader,
) -> DutPort:
    adapter = DutBackendCollector(
        config,
        username=credentials["DUT_BACKEND_USERNAME"],
        password=credentials["DUT_BACKEND_PASSWORD"],
    )
    adapter.preflight()
    return adapter


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
        credential_requirements=(
            CredentialRequirement("DUT_BACKEND_USERNAME", "DUT backend SSH username: "),
            CredentialRequirement(
                "DUT_BACKEND_PASSWORD", "DUT backend SSH password: ", secret=True
            ),
        ),
        adapter_builder=_build_backend,
    ),
    DutCollectionMethod.FRONTEND_API: DutRuntimeSpec(
        method=DutCollectionMethod.FRONTEND_API,
        credential_requirements=(
            CredentialRequirement("DUT_FRONTEND_USERNAME", "DUT frontend username: "),
            CredentialRequirement("DUT_FRONTEND_PASSWORD", "DUT frontend password: ", secret=True),
        ),
        adapter_builder=_build_frontend,
    ),
}


def dut_runtime_spec(method: DutCollectionMethod) -> DutRuntimeSpec:
    return _DUT_RUNTIME_SPECS[method]
