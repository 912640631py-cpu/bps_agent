from __future__ import annotations

import pytest

from bps_agent.graph import _attempt_bandwidth_target


@pytest.mark.parametrize(
    ("configured_mbps", "template_mbps", "attempt_number", "expected"),
    [
        (300.0, 400.0, 2, (240.0, 60.0)),
        (400.0, 800.0, 6, (40.0, 5.0)),
    ],
)
def test_attempt_bandwidth_target_is_relative_to_config_and_template(
    configured_mbps: float,
    template_mbps: float,
    attempt_number: int,
    expected: tuple[float, float],
) -> None:
    assert _attempt_bandwidth_target(configured_mbps, template_mbps, attempt_number) == expected
