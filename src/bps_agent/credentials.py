"""Credential storage backed by the operating-system keyring."""

from __future__ import annotations

import getpass
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

SERVICE_NAME = "nsfocus-bps-evaluation-agent"
SUPPORTED_CREDENTIALS = (
    "BPS_USERNAME",
    "BPS_PASSWORD",
    "DUT_FRONTEND_USERNAME",
    "DUT_FRONTEND_PASSWORD",
    "DUT_BACKEND_USERNAME",
    "DUT_BACKEND_PASSWORD",
    "COMPANY_DEEPSEEK_API_KEY",
    "DEEPSEEK_API_KEY",
)
SECRET_CREDENTIALS = frozenset(
    {
        "BPS_PASSWORD",
        "DUT_FRONTEND_PASSWORD",
        "DUT_BACKEND_PASSWORD",
        "COMPANY_DEEPSEEK_API_KEY",
        "DEEPSEEK_API_KEY",
    }
)


class CredentialStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class CredentialRequirement:
    """One credential required by the selected run configuration."""

    name: str
    prompt: str
    secret: bool = False


class CredentialStore:
    def __init__(self, service_name: str = SERVICE_NAME) -> None:
        self.service_name = service_name

    @staticmethod
    def validate_name(name: str) -> str:
        if name not in SUPPORTED_CREDENTIALS:
            raise ValueError(f"unsupported credential name: {name}")
        return name

    def get(self, name: str) -> str | None:
        name = self.validate_name(name)
        try:
            value = keyring.get_password(self.service_name, name)
        except KeyringError as exc:
            raise CredentialStoreError(
                f"cannot read {name} from the system keyring: {exc}"
            ) from exc
        return value if value else None

    def set(self, name: str, value: str) -> None:
        name = self.validate_name(name)
        if not value:
            raise ValueError(f"credential {name} must not be empty")
        try:
            keyring.set_password(self.service_name, name, value)
        except KeyringError as exc:
            raise CredentialStoreError(f"cannot save {name} to the system keyring: {exc}") from exc

    def delete(self, name: str) -> bool:
        name = self.validate_name(name)
        if self.get(name) is None:
            return False
        try:
            keyring.delete_password(self.service_name, name)
        except PasswordDeleteError:
            return False
        except KeyringError as exc:
            raise CredentialStoreError(
                f"cannot delete {name} from the system keyring: {exc}"
            ) from exc
        return True

    def status(self, names: Iterable[str] = SUPPORTED_CREDENTIALS) -> dict[str, bool]:
        return {name: self.get(name) is not None for name in names}


class CredentialResolver:
    """Resolve a run's credentials and optionally persist newly entered values."""

    def __init__(
        self,
        store: CredentialStore,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._store = store
        self._environment = os.environ if environment is None else environment

    def resolve(
        self,
        requirements: Iterable[CredentialRequirement],
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        entered: dict[str, str] = {}
        for requirement in requirements:
            name = CredentialStore.validate_name(requirement.name)
            if name in resolved:
                continue
            value = self._environment.get(name) or self._store.get(name)
            if not value:
                value = (
                    getpass.getpass(requirement.prompt)
                    if requirement.secret
                    else input(requirement.prompt).strip()
                )
                if not value:
                    raise ValueError(f"{name} is required")
                entered[name] = value
            resolved[name] = value

        if entered:
            save = input(
                "Save all newly entered credentials in the system keyring? [y/N]: "
            ).strip()
            if save.casefold() in {"y", "yes"}:
                for name, value in entered.items():
                    self._store.set(name, value)
                print(
                    "Saved newly entered credentials in the system keyring: " + ", ".join(entered)
                )
        return resolved
