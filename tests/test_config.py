from __future__ import annotations

from pathlib import Path

from bps_agent.config import load_config


def test_demo_configuration_matches_real_lab_defaults() -> None:
    config = load_config(Path("config/demo.yaml"))

    assert config.bps.endpoint == "https://10.66.250.104"
    assert config.bps.template == "ai_bps_puyu"
    assert config.bps.ports == (4, 5)
    assert config.bps.total_bandwidth_mbps == 400.0
    assert "report_section_ids" not in config.bps.model_dump()
    assert config.dut.endpoint == "https://10.66.246.133"
    assert config.dut.interfaces == ("T1/1", "T1/2")
    assert config.dut.baseline_seconds == 600
    assert config.dut.cooldown_seconds == 10
    assert config.dut.keepalive_interval_seconds == 60
    assert config.llm.company.model == "deepseek-v4-flash-0731"
    assert config.llm.official.model == "deepseek-v4-flash"
    assert config.evaluation.max_attempts == 5


def test_serialized_configuration_contains_no_secret_values() -> None:
    serialized = load_config(Path("config/demo.yaml")).model_dump_json()

    assert "password" not in serialized.casefold()
    assert "bearer" not in serialized.casefold()
    assert "api_key" in serialized.casefold()  # Environment-variable names only.
