"""Stable error contracts shared by the CLI and Agent domains."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    CLI_USAGE_ERROR = "CLI_USAGE_ERROR"
    CONFIG_NOT_FOUND = "CONFIG_NOT_FOUND"
    CONFIG_INVALID = "CONFIG_INVALID"
    RESUME_CONFLICT = "RESUME_CONFLICT"

    CREDENTIAL_MISSING = "CREDENTIAL_MISSING"
    CREDENTIAL_STORE_ERROR = "CREDENTIAL_STORE_ERROR"

    CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
    CHECKPOINT_INVALID = "CHECKPOINT_INVALID"
    CHECKPOINT_IO_ERROR = "CHECKPOINT_IO_ERROR"

    BPS_UNREACHABLE = "BPS_UNREACHABLE"
    BPS_AUTH_FAILED = "BPS_AUTH_FAILED"
    BPS_PROTOCOL_ERROR = "BPS_PROTOCOL_ERROR"
    BPS_TEMPLATE_NOT_FOUND = "BPS_TEMPLATE_NOT_FOUND"
    BPS_TEMPLATE_AMBIGUOUS = "BPS_TEMPLATE_AMBIGUOUS"
    BPS_BANDWIDTH_INVALID = "BPS_BANDWIDTH_INVALID"
    BPS_PORT_OCCUPIED = "BPS_PORT_OCCUPIED"
    BPS_RESERVATION_ERROR = "BPS_RESERVATION_ERROR"
    BPS_LAUNCH_ERROR = "BPS_LAUNCH_ERROR"
    BPS_RUN_TIMEOUT = "BPS_RUN_TIMEOUT"
    BPS_REPORT_ERROR = "BPS_REPORT_ERROR"

    DUT_UNREACHABLE = "DUT_UNREACHABLE"
    DUT_AUTH_FAILED = "DUT_AUTH_FAILED"
    DUT_CAPTCHA_FAILED = "DUT_CAPTCHA_FAILED"
    DUT_PROTOCOL_ERROR = "DUT_PROTOCOL_ERROR"
    DUT_COLLECTION_ERROR = "DUT_COLLECTION_ERROR"

    LLM_UNREACHABLE = "LLM_UNREACHABLE"
    LLM_AUTH_FAILED = "LLM_AUTH_FAILED"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    LLM_COMPATIBILITY_ERROR = "LLM_COMPATIBILITY_ERROR"
    LLM_REQUEST_ERROR = "LLM_REQUEST_ERROR"
    LLM_RESPONSE_INVALID = "LLM_RESPONSE_INVALID"

    ARTIFACT_IO_ERROR = "ARTIFACT_IO_ERROR"

    REPLAY_EVIDENCE_NOT_FOUND = "REPLAY_EVIDENCE_NOT_FOUND"
    REPLAY_EVIDENCE_INVALID = "REPLAY_EVIDENCE_INVALID"
    REPLAY_EVIDENCE_IO_ERROR = "REPLAY_EVIDENCE_IO_ERROR"

    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentError(RuntimeError):
    """Base class for failures that are safe to show to an operator."""

    default_code = ErrorCode.INTERNAL_ERROR.value
    code: str
    hint: str | None

    def __init__(
        self,
        message: str,
        *,
        code: str | ErrorCode | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code.value if isinstance(code, ErrorCode) else code or self.default_code
        self.hint = hint


class ConfigError(AgentError, ValueError):
    default_code = ErrorCode.CONFIG_INVALID.value


class CliUsageError(AgentError):
    default_code = ErrorCode.CLI_USAGE_ERROR.value


class CredentialError(AgentError, ValueError):
    default_code = ErrorCode.CREDENTIAL_STORE_ERROR.value


class BpsError(AgentError, ValueError):
    default_code = ErrorCode.BPS_PROTOCOL_ERROR.value


class DutError(AgentError):
    default_code = ErrorCode.DUT_COLLECTION_ERROR.value


class ProviderError(AgentError):
    default_code = ErrorCode.LLM_REQUEST_ERROR.value


class CheckpointError(AgentError, ValueError):
    default_code = ErrorCode.CHECKPOINT_INVALID.value


class ArtifactError(AgentError, ValueError):
    default_code = ErrorCode.ARTIFACT_IO_ERROR.value


class ReplayError(AgentError):
    default_code = ErrorCode.REPLAY_EVIDENCE_INVALID.value


class ProviderCompatibilityError(ProviderError):
    default_code = ErrorCode.LLM_COMPATIBILITY_ERROR.value


class ProviderRequestError(ProviderError):
    default_code = ErrorCode.LLM_REQUEST_ERROR.value


class ProviderResponseError(ProviderError):
    default_code = ErrorCode.LLM_RESPONSE_INVALID.value
