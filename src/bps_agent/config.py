"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bps_agent.models import AppConfig, EvaluationMode


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
