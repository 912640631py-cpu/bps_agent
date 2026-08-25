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
        canonical_ports = tuple(sorted(ports))
        if not canonical_ports or len(canonical_ports) != len(set(canonical_ports)):
            raise ValueError("BPS lock ports must be non-empty and unique")
        self.paths = tuple(
            lock_dir
            / (
                "bps-port-"
                + hashlib.sha256(f"{endpoint}|{slot}|{port}".encode()).hexdigest()[:20]
                + ".lock"
            )
            for port in canonical_ports
        )
        self.path = self.paths[0]
        self.evaluation_id = evaluation_id
        self._document = {
            "evaluation_id": evaluation_id,
            "pid": os.getpid(),
            "endpoint": endpoint,
            "slot": slot,
            "ports": canonical_ports,
            "group": group,
        }
        self._acquired_paths: list[Path] = []
        self._release_on_exit = True

    def acquire(self) -> None:
        acquired: list[Path] = []
        try:
            for path in self.paths:
                self._acquire_path(path)
                acquired.append(path)
        except Exception:
            for path in reversed(acquired):
                self._release_path(path)
            raise
        self._acquired_paths = acquired

    def _acquire_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    existing: Any = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                if existing.get("evaluation_id") == self.evaluation_id:
                    owner_pid = existing.get("pid")
                    if isinstance(owner_pid, int) and self._pid_is_alive(owner_pid):
                        raise PortGroupLockedError(
                            f"Evaluation Run {self.evaluation_id} is active in process {owner_pid}"
                        ) from None
                    try:
                        path.unlink()
                    except OSError as exc:
                        raise PortGroupLockedError(
                            f"cannot take over stale lock for Evaluation Run {self.evaluation_id}"
                        ) from exc
                    continue
                owner = existing.get("evaluation_id", "unknown")
                raise PortGroupLockedError(
                    f"BPS port is locked by Evaluation Run {owner}"
                ) from None
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self._document, handle)
            return

    def _release_path(self, path: Path) -> None:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("evaluation_id") == self.evaluation_id:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise PortGroupLockedError(
                    f"cannot release port lock for Evaluation Run {self.evaluation_id}"
                ) from exc

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
        paths = tuple(self._acquired_paths)
        try:
            for path in reversed(paths):
                self._release_path(path)
        finally:
            self._acquired_paths.clear()

    def preserve(self) -> None:
        """Keep the lock file when live BPS state requires human recovery."""

        self._release_on_exit = False

    def __enter__(self) -> PortGroupLock:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        if self._release_on_exit:
            self.release()
