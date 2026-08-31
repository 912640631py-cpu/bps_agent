"""Construction and lifetime management for live Evaluation Run resources."""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from bps_agent.adapters.bps import BpsClient
from bps_agent.adapters.deepseek import DeepSeekJudge
from bps_agent.artifacts import ArtifactStore
from bps_agent.credentials import CredentialRequirement
from bps_agent.dut_runtime import dut_runtime_spec, read_captcha
from bps_agent.models import AppConfig, EvaluationMode, EvidenceBundle
from bps_agent.ports import BpsPort, Clock, DutPort, JudgePort

_BPS_CREDENTIALS = (
    CredentialRequirement("BPS_USERNAME", "BPS username: "),
    CredentialRequirement("BPS_PASSWORD", "BPS password: ", secret=True),
)


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class EvidenceOnlyJudge:
    provider_name = "not-called"
    model_name = "not-called"

    def adjudicate(self, evidence: EvidenceBundle) -> tuple[Any, dict[str, Any]]:
        raise RuntimeError("LLM adjudication is disabled for this evidence-only run")

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class RuntimeResources:
    config: AppConfig
    bps: BpsPort
    dut: DutPort | None
    judge: JudgePort
    clock: Clock
    artifacts: ArtifactStore

    def close(self) -> None:
        if self.dut is not None:
            with suppress(Exception):
                self.dut.close()
        with suppress(Exception):
            close_bps = getattr(self.bps, "close", None)
            if close_bps is not None:
                close_bps()
        with suppress(Exception):
            close_judge = getattr(self.judge, "close", None)
            if close_judge is not None:
                close_judge()


def run_credential_requirements(
    config: AppConfig,
    *,
    stop_before_llm: bool,
) -> tuple[CredentialRequirement, ...]:
    requirements = list(_BPS_CREDENTIALS)
    if config.evaluation.mode == EvaluationMode.BPS_AND_DUT:
        assert config.dut is not None
        requirements.extend(
            dut_runtime_spec(config.dut.collection_method).credential_requirements
        )
    if not stop_before_llm:
        requirements.append(
            CredentialRequirement(
                config.llm.selected.token_env,
                f"{config.llm.provider} DeepSeek Bearer token: ",
                secret=True,
            )
        )
    return tuple(requirements)


def build_judge(config: AppConfig, credentials: dict[str, str]) -> DeepSeekJudge:
    selected = config.llm.selected
    return DeepSeekJudge(
        config.llm.provider,
        selected,
        token=credentials[selected.token_env],
        reasoning_effort=config.llm.reasoning_effort,
    )


def build_runtime(
    config: AppConfig,
    credentials: dict[str, str],
    *,
    stop_before_llm: bool = False,
) -> RuntimeResources:
    judge: DeepSeekJudge | EvidenceOnlyJudge = EvidenceOnlyJudge()
    bps: BpsClient | None = None
    dut: DutPort | None = None
    try:
        if not stop_before_llm:
            configured_judge = build_judge(config, credentials)
            judge = configured_judge
            configured_judge.validate_compatibility()
        bps = BpsClient(
            config.bps,
            username=credentials["BPS_USERNAME"],
            password=credentials["BPS_PASSWORD"],
        )
        bps.authenticate()
        if config.evaluation.mode == EvaluationMode.BPS_AND_DUT:
            assert config.dut is not None
            dut = dut_runtime_spec(config.dut.collection_method).build_adapter(
                config.dut, credentials, read_captcha
            )
        return RuntimeResources(
            config=config,
            bps=bps,
            dut=dut,
            judge=judge,
            clock=SystemClock(),
            artifacts=ArtifactStore(config.storage.artifact_dir),
        )
    except BaseException:
        if dut is not None:
            with suppress(Exception):
                dut.close()
        if bps is not None:
            with suppress(Exception):
                bps.close()
        with suppress(Exception):
            judge.close()
        raise
