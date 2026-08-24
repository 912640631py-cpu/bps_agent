"""Reserved extension boundary for future attack-test evidence."""

from __future__ import annotations

from typing import Any, Protocol


class SecurityEvidenceProvider(Protocol):
    """Future provider for attack, detection, blocking, and security-event evidence."""

    def collect(self, *, bps_run_id: str, started_at: str, finished_at: str) -> dict[str, Any]: ...
