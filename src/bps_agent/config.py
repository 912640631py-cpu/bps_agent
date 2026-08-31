"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bps_agent.models.common import EvaluationMode
from bps_agent.models.config import AppConfig, RunOverrides


def load_config(
    path: Path,
    *,
    mode_override: EvaluationMode | None = None,
) -> AppConfig:
    config_path = path.resolve()
    try:
        document: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("configuration root must be a YAML mapping")
    if mode_override is not None:
        evaluation = document.setdefault("evaluation", {})
        if not isinstance(evaluation, dict):
            raise ValueError("evaluation configuration must be a YAML mapping")
        evaluation["mode"] = mode_override.value
    config = AppConfig.model_validate(document)
    storage = config.storage.resolved_relative_to(config_path.parent)
    return config.model_copy(update={"storage": storage})


def apply_overrides(
    config: AppConfig,
    overrides: RunOverrides,
) -> AppConfig:
    """Apply the complete CLI override group through one validated interface."""

    document = config.model_dump(mode="python")
    if overrides.template is not None:
        cleaned = overrides.template.strip()
        if not cleaned:
            raise ValueError("--template must not be empty")
        document["bps"]["template"] = cleaned
    if overrides.ports is not None:
        document["bps"]["ports"] = overrides.ports
    if overrides.total_bandwidth_mbps is not None:
        document["bps"]["total_bandwidth_mbps"] = overrides.total_bandwidth_mbps
    if overrides.evaluation_mode is not None:
        document["evaluation"]["mode"] = overrides.evaluation_mode.value

    dut_values = (
        overrides.dut_collection_method,
        overrides.dut_host,
        overrides.dut_port,
        overrides.dut_interfaces,
        overrides.dut_interval_seconds,
    )
    if any(value is not None for value in dut_values) and document.get("dut") is None:
        raise ValueError("DUT overrides require a dut section in the configuration")
    if overrides.dut_collection_method is not None:
        document["dut"]["collection_method"] = overrides.dut_collection_method.value
    if any(
        value is not None
        for value in (
            overrides.dut_host,
            overrides.dut_port,
            overrides.dut_interval_seconds,
        )
    ):
        document["dut"]["backend"] = document["dut"].get("backend") or {}
    if overrides.dut_host is not None:
        document["dut"]["backend"]["host"] = overrides.dut_host
    if overrides.dut_port is not None:
        document["dut"]["backend"]["port"] = overrides.dut_port
    if overrides.dut_interfaces is not None:
        document["dut"]["interfaces"] = overrides.dut_interfaces
    if overrides.dut_interval_seconds is not None:
        document["dut"]["backend"]["interval_seconds"] = overrides.dut_interval_seconds
    return AppConfig.model_validate(document)
