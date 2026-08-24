"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bps_agent.models import AppConfig


def load_config(path: Path) -> AppConfig:
    try:
        document: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read configuration {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("configuration root must be a YAML mapping")
    return AppConfig.model_validate(document)
