"""Compact, evidence-referencing adjudication audit artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from bps_agent.models.evaluation import VerdictDocument


def verdict_artifact(
    *,
    provider: str,
    model: str,
    reasoning_effort: str | None,
    verdict: VerdictDocument,
    provider_exchange: dict[str, Any],
    evidence_path: Path,
) -> dict[str, Any]:
    evidence_bytes = evidence_path.read_bytes()
    provider_response = provider_exchange.get("response", provider_exchange)
    model_parameters: dict[str, Any] = {}
    if reasoning_effort is not None:
        model_parameters["reasoning_effort"] = reasoning_effort
    return {
        "schema_version": "1",
        "provider": provider,
        "model": model,
        "model_parameters": model_parameters,
        "parsed": verdict.model_dump(mode="json"),
        "provider_response": provider_response,
        "evidence": {
            "path": str(evidence_path),
            "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        },
    }
