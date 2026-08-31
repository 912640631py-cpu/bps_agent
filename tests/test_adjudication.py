from __future__ import annotations

import hashlib
from pathlib import Path

from bps_agent.adjudication import verdict_artifact
from bps_agent.models.common import VerdictValue
from bps_agent.models.evaluation import VerdictDocument


def test_verdict_artifact_references_evidence_without_repeating_request(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text('{"evidence": true}\n', encoding="utf-8")

    artifact = verdict_artifact(
        provider="provider",
        model="model",
        reasoning_effort="high",
        verdict=VerdictDocument(verdict=VerdictValue.PASS, summary="fixture"),
        provider_exchange={"response": {"fixture": True}, "request": {"secret": True}},
        evidence_path=evidence_path,
    )

    assert artifact["evidence"]["path"] == str(evidence_path)
    assert artifact["evidence"]["sha256"] == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    assert artifact["provider_response"] == {"fixture": True}
    assert "raw_response" not in artifact
    assert "request" not in artifact
