"""Construction and lifetime management for live Evaluation Run resources."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from bps_agent.adapters.bps import BpsClient
from bps_agent.adapters.deepseek import DeepSeekJudge
from bps_agent.artifacts import ArtifactStore
from bps_agent.credentials import CredentialRequirement
from bps_agent.dut_runtime import CaptchaReader, dut_runtime_spec
from bps_agent.errors import AgentError, ArtifactError, ErrorCode
from bps_agent.models.common import EvaluationMode
from bps_agent.models.config import AppConfig
from bps_agent.models.evaluation import EvidenceBundle
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
        raise AgentError(
            "LLM adjudication is disabled for this evidence-only run",
            code=ErrorCode.INTERNAL_ERROR,
        )

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
    download_full_pdf: bool = False

    def close(self) -> None:
        if self.dut is not None:
            with suppress(Exception):
                self.dut.close()
        with suppress(Exception):
            self.bps.close()
        with suppress(Exception):
            self.judge.close()


def run_credential_requirements(
    config: AppConfig,
    *,
    stop_before_llm: bool,
) -> tuple[CredentialRequirement, ...]:
    requirements = list(_BPS_CREDENTIALS)
    if config.evaluation.mode == EvaluationMode.BPS_AND_DUT:
        assert config.dut is not None
        requirements.extend(dut_runtime_spec(config.dut.collection_method).credential_requirements)
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
    captcha_reader: CaptchaReader,
    stop_before_llm: bool = False,
    download_full_pdf: bool = False,
    progress: Callable[[str], None] | None = None,
) -> RuntimeResources:
    judge: DeepSeekJudge | EvidenceOnlyJudge = EvidenceOnlyJudge()
    bps: BpsClient | None = None
    dut: DutPort | None = None
    announce = progress or (lambda _message: None)
    try:
        if not stop_before_llm:
            announce(
                f"Checking {config.llm.provider} provider compatibility with "
                f"reasoning_effort={config.llm.reasoning_effort}..."
            )
            configured_judge = build_judge(config, credentials)
            judge = configured_judge
            configured_judge.validate_compatibility()
        else:
            announce("DeepSeek compatibility check skipped (evidence-only mode).")
        bps = BpsClient(
            config.bps,
            username=credentials["BPS_USERNAME"],
            password=credentials["BPS_PASSWORD"],
        )
        announce("Authenticating to BPS...")
        bps.authenticate()
        announce("Validating BPS template and bandwidth...")
        preflight = getattr(bps, "preflight", None)
        if callable(preflight):
            preflight()
        if config.evaluation.mode == EvaluationMode.BPS_AND_DUT:
            assert config.dut is not None
            announce(f"Checking DUT ({config.dut.collection_method.value})...")
            dut = dut_runtime_spec(config.dut.collection_method).build_adapter(
                config.dut, credentials, captcha_reader
            )
        else:
            announce("BPS-only mode: DUT checks skipped.")
        announce("Checking local storage...")
        try:
            config.storage.artifact_dir.mkdir(parents=True, exist_ok=True)
            config.storage.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactError(
                "local storage is not writable",
                code=ErrorCode.ARTIFACT_IO_ERROR,
                hint="Check artifact and checkpoint paths and their permissions.",
            ) from exc
        return RuntimeResources(
            config=config,
            bps=bps,
            dut=dut,
            judge=judge,
            clock=SystemClock(),
            artifacts=ArtifactStore(config.storage.artifact_dir),
            download_full_pdf=download_full_pdf,
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
