"""Console UI for solving DUT frontend CAPTCHA challenges."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from bps_agent.errors import DutError, ErrorCode

LOGGER = logging.getLogger(__name__)


def read_captcha(image: bytes, media_type: str) -> str:
    suffix = {
        "image/gif": ".gif",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(media_type.casefold(), ".img")
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="dut-captcha-", suffix=suffix, delete=False
        ) as handle:
            handle.write(image)
            path = Path(handle.name)
        print(f"DUT CAPTCHA image: {path}")
        try:
            subprocess.Popen(
                ["explorer.exe", f"/select,{path.resolve()}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            LOGGER.warning("Could not open CAPTCHA image automatically: %s", exc)
        try:
            value = input("DUT CAPTCHA: ").strip()
        except EOFError as exc:
            raise DutError(
                "DUT CAPTCHA input was interrupted",
                code=ErrorCode.DUT_CAPTCHA_FAILED,
                hint="Run CAPTCHA authentication from an interactive terminal.",
            ) from exc
        if not value:
            raise DutError(
                "DUT CAPTCHA must not be empty",
                code=ErrorCode.DUT_CAPTCHA_FAILED,
            )
        return value
    except OSError as exc:
        raise DutError(
            "DUT CAPTCHA image could not be prepared",
            code=ErrorCode.DUT_CAPTCHA_FAILED,
        ) from exc
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
