"""Atomic audit-artifact persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def evaluation_dir(self, evaluation_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", evaluation_id):
            raise ValueError("Evaluation Run ID contains unsafe path characters")
        return self.root / evaluation_id

    def attempt_dir(self, evaluation_id: str, attempt_number: int) -> Path:
        return self.evaluation_dir(evaluation_id) / f"attempt-{attempt_number:02d}"

    def ensure_evaluation(self, evaluation_id: str) -> Path:
        directory = self.evaluation_dir(evaluation_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_evaluation_json(self, evaluation_id: str, name: str, value: Any) -> Path:
        return self.write_json(self.ensure_evaluation(evaluation_id) / name, value)

    def write_attempt_json(
        self, evaluation_id: str, attempt_number: int, name: str, value: Any
    ) -> Path:
        directory = self.attempt_dir(evaluation_id, attempt_number)
        directory.mkdir(parents=True, exist_ok=True)
        return self.write_json(directory / name, value)

    def write_attempt_text(
        self, evaluation_id: str, attempt_number: int, name: str, value: str
    ) -> Path:
        directory = self.attempt_dir(evaluation_id, attempt_number)
        directory.mkdir(parents=True, exist_ok=True)
        return self.write_text(directory / name, value)

    @staticmethod
    def _serializable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        return value

    @classmethod
    def write_json(cls, path: Path, value: Any) -> Path:
        encoded = (
            json.dumps(cls._serializable(value), ensure_ascii=False, indent=2, default=str) + "\n"
        )
        return cls.write_text(path, encoded)

    @staticmethod
    def write_text(path: Path, value: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, path)
            temporary = None
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            with suppress(OSError):
                directory_fd = os.open(path.parent, directory_flags)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            return path
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))
