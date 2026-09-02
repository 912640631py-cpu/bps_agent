"""Credential storage backed by the operating-system keyring."""

from __future__ import annotations

import builtins
import getpass
import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

from bps_agent.errors import CredentialError, ErrorCode

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

_ORIGINAL_INPUT = builtins.input
_ORIGINAL_GETPASS = getpass.getpass


class CredentialStoreError(CredentialError):
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
            raise CredentialStoreError(
                f"unsupported credential name: {name}",
                code=ErrorCode.CREDENTIAL_STORE_ERROR,
            )
        return name

    def get(self, name: str) -> str | None:
        name = self.validate_name(name)
        try:
            value = keyring.get_password(self.service_name, name)
        except KeyringError as exc:
            raise CredentialStoreError(
                f"cannot read {name} from the system keyring",
                code=ErrorCode.CREDENTIAL_STORE_ERROR,
            ) from exc
        except Exception as exc:
            raise CredentialStoreError(
                f"cannot read {name} from the system keyring",
                code=ErrorCode.CREDENTIAL_STORE_ERROR,
            ) from exc
        return value if value else None

    def set(self, name: str, value: str) -> None:
        name = self.validate_name(name)
        if not value:
            raise CredentialError(
                f"credential {name} must not be empty",
                code=ErrorCode.CREDENTIAL_MISSING,
            )
        try:
            keyring.set_password(self.service_name, name, value)
        except KeyringError as exc:
            raise CredentialStoreError(
                f"cannot save {name} to the system keyring",
                code=ErrorCode.CREDENTIAL_STORE_ERROR,
            ) from exc
        except Exception as exc:
            raise CredentialStoreError(
                f"cannot save {name} to the system keyring",
                code=ErrorCode.CREDENTIAL_STORE_ERROR,
            ) from exc

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
                f"cannot delete {name} from the system keyring",
                code=ErrorCode.CREDENTIAL_STORE_ERROR,
            ) from exc
        except Exception as exc:
            raise CredentialStoreError(
                f"cannot delete {name} from the system keyring",
                code=ErrorCode.CREDENTIAL_STORE_ERROR,
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
                if not _stdin_is_interactive():
                    raise CredentialError(
                        f"{name} is required",
                        code=ErrorCode.CREDENTIAL_MISSING,
                        hint=f"Set {name} in the environment or system keyring.",
                    )
                try:
                    value = (
                        getpass.getpass(requirement.prompt)
                        if requirement.secret
                        else input(requirement.prompt).strip()
                    )
                except (EOFError, OSError) as exc:
                    raise CredentialError(
                        f"{name} is required",
                        code=ErrorCode.CREDENTIAL_MISSING,
                        hint=f"Set {name} in the environment or system keyring.",
                    ) from exc
                if not value:
                    raise CredentialError(
                        f"{name} is required",
                        code=ErrorCode.CREDENTIAL_MISSING,
                        hint=f"Set {name} in the environment or system keyring.",
                    )
                entered[name] = value
            resolved[name] = value

        if entered:
            try:
                save = input(
                    "Save all newly entered credentials in the system keyring? [y/N]: "
                ).strip()
            except (EOFError, OSError):
                save = ""
            if save.casefold() in {"y", "yes"}:
                for name, value in entered.items():
                    self._store.set(name, value)
                print(
                    "Saved newly entered credentials in the system keyring: " + ", ".join(entered)
                )
        return resolved


def _stdin_is_interactive() -> bool:
    """Return whether prompting is safe for this process."""

    # Test callers and embedders may provide a prompt callback while stdout/stdin
    # itself is captured or redirected.  Treat an explicit callback as interactive.
    if builtins.input is not _ORIGINAL_INPUT or getpass.getpass is not _ORIGINAL_GETPASS:
        return True
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, OSError):
        return False
