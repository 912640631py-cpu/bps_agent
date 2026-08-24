from __future__ import annotations

from argparse import Namespace

import keyring
import pytest

from bps_agent.cli import _credential, _parser, manage_credentials
from bps_agent.credentials import SUPPORTED_CREDENTIALS, CredentialStore


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
    monkeypatch.setenv("BPS_USERNAME", "environment-user")  # type: ignore[attr-defined]
    store = MemoryStore({"BPS_USERNAME": "keyring-user"})

    value = _credential(  # type: ignore[arg-type]
        store, "BPS_USERNAME", "BPS username: ", secret=False
    )

    assert value == "environment-user"


def test_missing_prompted_value_is_saved(monkeypatch: object) -> None:
    monkeypatch.delenv("BPS_USERNAME", raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr("builtins.input", lambda _prompt: "prompted-user")  # type: ignore[attr-defined]
    store = MemoryStore()

    value = _credential(  # type: ignore[arg-type]
        store, "BPS_USERNAME", "BPS username: ", secret=False
    )

    assert value == "prompted-user"
    assert store.values["BPS_USERNAME"] == "prompted-user"


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


def test_credentials_management_rejects_unsupported_name() -> None:
    arguments = _parser().parse_args(["credentials", "delete", "UNKNOWN_CREDENTIAL"])

    with pytest.raises(ValueError, match="unsupported credential name: UNKNOWN_CREDENTIAL"):
        manage_credentials(arguments, MemoryStore())  # type: ignore[arg-type]
