from __future__ import annotations

from pathlib import Path

from bps_agent.config import load_config
from bps_agent.models import DutCollectionMethod, DutConfig, EvaluationMode


def test_demo_configuration_matches_real_lab_defaults() -> None:
    config = load_config(Path("config/demo.yaml"))

    assert config.bps.endpoint == "https://10.66.250.104"
    assert config.bps.template == "ai_bps_puyu"
    assert config.bps.ports == (4, 5)
    assert config.bps.total_bandwidth_mbps == 400.0
    assert config.bps.port_release_attempts == 6
    assert config.bps.port_release_retry_backoff_seconds == 5
    assert "report_section_ids" not in config.bps.model_dump()
    assert config.dut.collection_method == DutCollectionMethod.FRONTEND_API
    assert config.dut.backend.host == "10.66.246.156"
    assert config.dut.backend.port == 50023
    assert config.dut.backend.interval_seconds == 10
    assert config.dut.frontend is not None
    assert config.dut.frontend.endpoint == "https://10.66.246.133"
    assert config.dut.interfaces == ("T1/1", "T1/2")
    assert config.dut.frontend.baseline_seconds == 600
    assert config.dut.frontend.cooldown_seconds == 10
    assert config.dut.frontend.keepalive_interval_seconds == 60
    assert config.llm.company.model == "deepseek-v4-flash-0731"
    assert config.llm.official.model == "deepseek-v4-flash"
    assert config.evaluation.max_attempts == 5
    assert config.evaluation.mode == EvaluationMode.BPS_AND_DUT
    assert "BPS" in config.bps_only_assessment.goal
    assert "lock_dir" not in config.storage.model_dump()


def test_legacy_flat_dut_configuration_selects_frontend_collection() -> None:
    config = DutConfig(
        endpoint="https://dut.example.test",
        interfaces=("T1/1",),
        baseline_seconds=300,
    )

    assert config.collection_method == DutCollectionMethod.FRONTEND_API
    assert config.frontend is not None
    assert config.frontend.endpoint == "https://dut.example.test"
    assert config.frontend.baseline_seconds == 300


def test_structured_dut_configuration_defaults_to_frontend_collection() -> None:
    config = DutConfig(
        interfaces=("T1/1",),
        frontend={"endpoint": "https://dut.example.test"},
    )

    assert config.collection_method == DutCollectionMethod.FRONTEND_API


def test_serialized_configuration_contains_no_secret_values() -> None:
    serialized = load_config(Path("config/demo.yaml")).model_dump_json()

    assert "password" not in serialized.casefold()
    assert "bearer" not in serialized.casefold()
    assert "api_key" in serialized.casefold()  # Environment-variable names only.
