from __future__ import annotations

from argparse import Namespace

import keyring
import pytest

from bps_agent.cli import _parser, manage_credentials
from bps_agent.credentials import (
    SUPPORTED_CREDENTIALS,
    CredentialRequirement,
    CredentialResolver,
    CredentialStore,
)
from bps_agent.models.common import DutCollectionMethod, EvaluationMode
from bps_agent.models.config import AppConfig
from bps_agent.runtime import run_credential_requirements


def test_credential_store_round_trip_without_exposing_values(monkeypatch: object) -> None:
    values: dict[tuple[str, str], str] = {}

    def get_password(service: str, name: str) -> str | None:
        return values.get((service, name))

    def set_password(service: str, name: str, value: str) -> None:
        values[(service, name)] = value

    def delete_password(service: str, name: str) -> None:
        del values[(service, name)]

    monkeypatch.setattr(keyring, "get_password", get_password)  # type: ignore[attr-defined]
    monkeypatch.setattr(keyring, "set_password", set_password)  # type: ignore[attr-defined]
    monkeypatch.setattr(keyring, "delete_password", delete_password)  # type: ignore[attr-defined]
    store = CredentialStore("test-service")

    assert store.get("BPS_USERNAME") is None
    store.set("BPS_USERNAME", "operator")
    assert store.get("BPS_USERNAME") == "operator"
    assert store.status(["BPS_USERNAME"]) == {"BPS_USERNAME": True}
    assert store.delete("BPS_USERNAME")
    assert store.get("BPS_USERNAME") is None


class MemoryStore:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = values or {}

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete(self, name: str) -> bool:
        return self.values.pop(name, None) is not None

    def status(self) -> dict[str, bool]:
        return {name: name in self.values for name in SUPPORTED_CREDENTIALS}


def test_environment_value_overrides_keyring(monkeypatch: object) -> None:
    store = MemoryStore({"BPS_USERNAME": "keyring-user"})

    values = CredentialResolver(  # type: ignore[arg-type]
        store,
        {"BPS_USERNAME": "environment-user"},
    ).resolve((CredentialRequirement("BPS_USERNAME", "BPS username: "),))

    assert values["BPS_USERNAME"] == "environment-user"


@pytest.mark.parametrize(("save_answer", "saved"), [("y", True), ("n", False)])
def test_new_credentials_use_one_save_confirmation(
    monkeypatch: object,
    save_answer: str,
    saved: bool,
) -> None:
    prompts: list[str] = []

    def read_text(prompt: str) -> str:
        prompts.append(prompt)
        return save_answer if prompt.startswith("Save all") else "prompted-user"

    monkeypatch.setattr("builtins.input", read_text)  # type: ignore[attr-defined]
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "prompted-password")  # type: ignore[attr-defined]
    store = MemoryStore()

    values = CredentialResolver(store, {}).resolve(  # type: ignore[arg-type]
        (
            CredentialRequirement("BPS_USERNAME", "BPS username: "),
            CredentialRequirement("BPS_PASSWORD", "BPS password: ", secret=True),
        )
    )

    assert values == {
        "BPS_USERNAME": "prompted-user",
        "BPS_PASSWORD": "prompted-password",
    }
    assert len([prompt for prompt in prompts if prompt.startswith("Save all")]) == 1
    assert (store.values == values) is saved


@pytest.mark.parametrize(
    ("mode", "method", "stop_before_llm", "expected_names"),
    [
        (
            EvaluationMode.BPS_ONLY,
            DutCollectionMethod.BACKEND_SSH,
            False,
            ("BPS_USERNAME", "BPS_PASSWORD", "COMPANY_DEEPSEEK_API_KEY"),
        ),
        (
            EvaluationMode.BPS_ONLY,
            DutCollectionMethod.BACKEND_SSH,
            True,
            ("BPS_USERNAME", "BPS_PASSWORD"),
        ),
        (
            EvaluationMode.BPS_AND_DUT,
            DutCollectionMethod.BACKEND_SSH,
            False,
            (
                "BPS_USERNAME",
                "BPS_PASSWORD",
                "DUT_BACKEND_USERNAME",
                "DUT_BACKEND_PASSWORD",
                "COMPANY_DEEPSEEK_API_KEY",
            ),
        ),
        (
            EvaluationMode.BPS_AND_DUT,
            DutCollectionMethod.BACKEND_SSH,
            True,
            (
                "BPS_USERNAME",
                "BPS_PASSWORD",
                "DUT_BACKEND_USERNAME",
                "DUT_BACKEND_PASSWORD",
            ),
        ),
        (
            EvaluationMode.BPS_AND_DUT,
            DutCollectionMethod.FRONTEND_API,
            False,
            (
                "BPS_USERNAME",
                "BPS_PASSWORD",
                "DUT_FRONTEND_USERNAME",
                "DUT_FRONTEND_PASSWORD",
                "COMPANY_DEEPSEEK_API_KEY",
            ),
        ),
        (
            EvaluationMode.BPS_AND_DUT,
            DutCollectionMethod.FRONTEND_API,
            True,
            (
                "BPS_USERNAME",
                "BPS_PASSWORD",
                "DUT_FRONTEND_USERNAME",
                "DUT_FRONTEND_PASSWORD",
            ),
        ),
    ],
)
def test_run_credentials_are_mode_aware(
    app_config: AppConfig,
    mode: EvaluationMode,
    method: DutCollectionMethod,
    stop_before_llm: bool,
    expected_names: tuple[str, ...],
) -> None:
    config = app_config.model_copy(
        update={
            "evaluation": app_config.evaluation.model_copy(update={"mode": mode}),
            "dut": app_config.dut.model_copy(update={"collection_method": method}),
        }
    )

    requirements = run_credential_requirements(
        config,
        stop_before_llm=stop_before_llm,
    )

    assert tuple(requirement.name for requirement in requirements) == expected_names


def test_status_prints_presence_but_not_secret(monkeypatch: object, capsys: object) -> None:
    monkeypatch.delenv("BPS_PASSWORD", raising=False)  # type: ignore[attr-defined]
    store = MemoryStore({"BPS_PASSWORD": "never-print-this"})

    result = manage_credentials(  # type: ignore[arg-type]
        Namespace(credential_command="status"), store
    )

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert result == 0
    assert "BPS_PASSWORD: keyring=stored, environment=unset" in output
    assert "never-print-this" not in output


def test_credentials_set_without_names_selects_all_at_management_boundary() -> None:
    arguments = _parser().parse_args(["credentials", "set"])

    assert arguments.names == []


def test_backend_dut_credentials_are_managed_separately() -> None:
    assert "DUT_BACKEND_USERNAME" in SUPPORTED_CREDENTIALS
    assert "DUT_BACKEND_PASSWORD" in SUPPORTED_CREDENTIALS
    assert "DUT_FRONTEND_USERNAME" in SUPPORTED_CREDENTIALS
    assert "DUT_FRONTEND_PASSWORD" in SUPPORTED_CREDENTIALS
    assert "DUT_USERNAME" not in SUPPORTED_CREDENTIALS
    assert "DUT_PASSWORD" not in SUPPORTED_CREDENTIALS


def test_credentials_management_rejects_unsupported_name() -> None:
    arguments = _parser().parse_args(["credentials", "delete", "UNKNOWN_CREDENTIAL"])

    with pytest.raises(ValueError, match="unsupported credential name: UNKNOWN_CREDENTIAL"):
        manage_credentials(arguments, MemoryStore())  # type: ignore[arg-type]
