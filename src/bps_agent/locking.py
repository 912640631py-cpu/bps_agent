"""Conservative local ownership lock for a BPS port group."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class PortGroupLockedError(RuntimeError):
    pass


class PortGroupLock:
    def __init__(
        self,
        lock_dir: Path,
        *,
        endpoint: str,
        slot: int,
        ports: tuple[int, ...],
        group: int,
        evaluation_id: str,
    ) -> None:
        identity = f"{endpoint}|{slot}|{','.join(map(str, ports))}|{group}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        self.path = lock_dir / f"bps-{digest}.lock"
        self.evaluation_id = evaluation_id
        self._acquired = False
        self._release_on_exit = True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {"evaluation_id": self.evaluation_id, "pid": os.getpid()}
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing: Any = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if existing.get("evaluation_id") == self.evaluation_id:
                owner_pid = existing.get("pid")
                if isinstance(owner_pid, int) and self._pid_is_alive(owner_pid):
                    raise PortGroupLockedError(
                        f"Evaluation Run {self.evaluation_id} is active in process {owner_pid}"
                    ) from None
                try:
                    self.path.unlink()
                except OSError as exc:
                    raise PortGroupLockedError(
                        f"cannot take over stale lock for Evaluation Run {self.evaluation_id}"
                    ) from exc
                self.acquire()
                return
            owner = existing.get("evaluation_id", "unknown")
            raise PortGroupLockedError(
                f"BPS port group is locked by Evaluation Run {owner}"
            ) from None
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        self._acquired = True

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True
        if os.name == "nt":
            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            open_process.restype = ctypes.c_void_p
            get_exit_code = kernel32.GetExitCodeProcess
            get_exit_code.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            get_exit_code.restype = ctypes.c_int
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [ctypes.c_void_p]
            close_handle.restype = ctypes.c_int
            handle = open_process(process_query_limited_information, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            try:
                if not get_exit_code(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == still_active
            finally:
                close_handle(handle)
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("evaluation_id") == self.evaluation_id:
            self.path.unlink(missing_ok=True)
        self._acquired = False

    def preserve(self) -> None:
        """Keep the lock file when live BPS state requires human recovery."""

        self._release_on_exit = False

    def __enter__(self) -> PortGroupLock:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        if self._release_on_exit:
            self.release()
