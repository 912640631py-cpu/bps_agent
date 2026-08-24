"""Credential storage backed by the operating-system keyring."""

from __future__ import annotations

from collections.abc import Iterable

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

SERVICE_NAME = "nsfocus-bps-evaluation-agent"
SUPPORTED_CREDENTIALS = (
    "BPS_USERNAME",
    "BPS_PASSWORD",
    "DUT_USERNAME",
    "DUT_PASSWORD",
    "COMPANY_DEEPSEEK_API_KEY",
    "DEEPSEEK_API_KEY",
)
SECRET_CREDENTIALS = frozenset(
    {
        "BPS_PASSWORD",
        "DUT_PASSWORD",
        "COMPANY_DEEPSEEK_API_KEY",
        "DEEPSEEK_API_KEY",
    }
)


class CredentialStoreError(RuntimeError):
    pass


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
