from __future__ import annotations

from typing import Any

import pytest

from bps_agent.models.config import AppConfig
from bps_agent.runtime import RuntimeResources, build_runtime


def test_build_runtime_constructs_and_closes_the_selected_resources(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Judge:
        provider_name = "provider"
        model_name = "model"

        def validate_compatibility(self) -> None:
            calls.append("judge.validate")

        def adjudicate(self, _evidence: object) -> tuple[Any, dict[str, Any]]:
            raise AssertionError("not used")

        def close(self) -> None:
            calls.append("judge.close")

    class Bps:
        def __init__(self, _config: object, *, username: str, password: str) -> None:
            assert (username, password) == ("bps-user", "bps-password")
            calls.append("bps.create")

        def authenticate(self) -> None:
            calls.append("bps.authenticate")

        def close(self) -> None:
            calls.append("bps.close")

    class Dut:
        def __init__(
            self,
            _config: object,
            *,
            username: str,
            password: str,
            captcha_reader: object,
        ) -> None:
            assert (username, password) == ("dut-user", "dut-password")
            assert captcha_reader is not None
            calls.append("dut.create")

        def authenticate(self) -> None:
            calls.append("dut.authenticate")

        def close(self) -> None:
            calls.append("dut.close")

    judge = Judge()
    monkeypatch.setattr("bps_agent.runtime.build_judge", lambda *_args: judge)
    monkeypatch.setattr("bps_agent.runtime.BpsClient", Bps)
    monkeypatch.setattr("bps_agent.dut_runtime.DutClient", Dut)
    resources = build_runtime(
        app_config,
        {
            "BPS_USERNAME": "bps-user",
            "BPS_PASSWORD": "bps-password",
            "DUT_FRONTEND_USERNAME": "dut-user",
            "DUT_FRONTEND_PASSWORD": "dut-password",
            app_config.llm.selected.token_env: "token",
        },
        captcha_reader=lambda _image, _media_type: "captcha",
    )

    assert isinstance(resources, RuntimeResources)
    assert calls == [
        "judge.validate",
        "bps.create",
        "bps.authenticate",
        "dut.create",
        "dut.authenticate",
    ]

    resources.close()

    assert calls[-3:] == ["dut.close", "bps.close", "judge.close"]
