from __future__ import annotations

from pathlib import Path

import pytest

from bps_agent.config import load_config
from bps_agent.models import AppConfig, DutCollectionMethod, DutConfig, EvaluationMode


def test_demo_configuration_matches_real_lab_defaults() -> None:
    config = load_config(Path("config/demo.yaml"))

    assert config.bps.endpoint == "https://10.66.250.104"
    assert config.bps.template == "ai_bps_puyu"
    assert config.bps.ports == (4, 5)
    assert config.bps.total_bandwidth_mbps == 400.0
    assert config.bps.port_release_attempts == 6
    assert config.bps.port_release_retry_backoff_seconds == 5
    assert "report_section_ids" not in config.bps.model_dump()
    assert config.bps.pdf_report_timeout_seconds == 1200
    assert config.bps.max_pdf_report_bytes == 512 * 1024 * 1024
    assert config.dut.collection_method == DutCollectionMethod.BACKEND_SSH
    assert config.dut.backend.host == "10.66.246.133"
    assert config.dut.backend.port == 50023
    assert config.dut.backend.interval_seconds == 10
    assert config.dut.backend.worker_stop_timeout_seconds == 5
    assert config.dut.frontend is not None
    assert config.dut.frontend.endpoint == "https://10.66.246.133"
    assert config.dut.interfaces == ("T1/1", "T1/2")
    assert config.dut.frontend.baseline_seconds == 600
    assert config.dut.frontend.cooldown_seconds == 10
    assert config.dut.frontend.keepalive_interval_seconds == 60
    assert config.llm.company.model == "deepseek-v4-flash-0731"
    assert config.llm.official.model == "deepseek-v4-flash"
    assert config.llm.reasoning_effort == "max"
    assert config.evaluation.max_attempts == 6
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


def test_structured_backend_configuration_requires_an_explicit_target() -> None:
    with pytest.raises(ValueError, match=r"requires dut\.backend"):
        DutConfig(
            interfaces=("T1/1",),
            frontend={"endpoint": "https://dut.example.test"},
        )


def test_serialized_configuration_contains_no_secret_values() -> None:
    serialized = load_config(Path("config/demo.yaml")).model_dump_json()

    assert "password" not in serialized.casefold()
    assert "bearer" not in serialized.casefold()
    assert "api_key" in serialized.casefold()  # Environment-variable names only.


def test_llm_reasoning_effort_is_configurable() -> None:
    config = load_config(Path("config/demo.yaml"))
    document = config.model_dump(mode="python")
    document["llm"]["reasoning_effort"] = "high"

    updated = type(config).model_validate(document)

    assert updated.llm.reasoning_effort == "high"


def test_bps_only_configuration_can_omit_dut_section(tmp_path: Path) -> None:
    config_path = tmp_path / "bps-only.yaml"
    config_path.write_text(
        """
bps:
  endpoint: https://bps.example.test
  template: performance-demo
  slot: 4
  ports: [4, 5]
  group: 10
evaluation:
  mode: bps_only
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.evaluation.mode == EvaluationMode.BPS_ONLY
    assert config.dut is None


def test_bps_only_cli_mode_override_is_applied_before_cross_field_validation(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bps-only-via-cli.yaml"
    config_path.write_text(
        """
bps:
  endpoint: https://bps.example.test
  template: performance-demo
  slot: 4
  ports: [4, 5]
  group: 10
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path, mode_override=EvaluationMode.BPS_ONLY)

    assert config.evaluation.mode == EvaluationMode.BPS_ONLY
    assert config.dut is None


def test_bps_and_dut_configuration_requires_dut_section() -> None:
    with pytest.raises(ValueError, match="bps_and_dut evaluation requires dut"):
        AppConfig.model_validate(
            {
                "bps": {
                    "endpoint": "https://bps.example.test",
                    "template": "performance-demo",
                    "slot": 4,
                    "ports": [4, 5],
                    "group": 10,
                }
            }
        )


def test_storage_paths_are_anchored_to_configuration_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    config_path = config_dir / "bps-only.yaml"
    config_path.write_text(
        """
bps:
  endpoint: https://bps.example.test
  template: performance-demo
  slot: 4
  ports: [4, 5]
  group: 10
evaluation:
  mode: bps_only
storage:
  artifact_dir: output/artifacts
  checkpoint_db: state/checkpoints.sqlite3
""".strip(),
        encoding="utf-8",
    )
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    config = load_config(config_path)

    assert config.storage.artifact_dir == (config_dir / "output/artifacts").resolve()
    assert config.storage.checkpoint_db == (config_dir / "state/checkpoints.sqlite3").resolve()
    assert config.storage.artifact_dir.is_absolute()
    assert config.storage.checkpoint_db.is_absolute()
