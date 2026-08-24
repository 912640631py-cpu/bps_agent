"""Command-line interface for live Evaluation Runs and replay."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from bps_agent.adapters.bps import BpsClient
from bps_agent.adapters.deepseek import DeepSeekJudge
from bps_agent.adapters.dut import DutClient
from bps_agent.artifacts import ArtifactStore
from bps_agent.config import load_config
from bps_agent.graph import EvaluationServices, SystemClock, build_graph, initial_state
from bps_agent.locking import PortGroupLock
from bps_agent.models import AppConfig, AttemptRecord, EvaluationOutcome, EvidenceBundle

LOGGER = logging.getLogger(__name__)

EXIT_CODES = {
    EvaluationOutcome.PASSED.value: 0,
    EvaluationOutcome.NOT_PASSED.value: 2,
    EvaluationOutcome.INCONCLUSIVE.value: 3,
}


def _secret(env_name: str, prompt: str) -> str:
    value = os.environ.get(env_name) or getpass.getpass(prompt)
    if not value:
        raise ValueError(f"{env_name} is required")
    return value


def _username(env_name: str, prompt: str) -> str:
    value = os.environ.get(env_name) or input(prompt).strip()
    if not value:
        raise ValueError(f"{env_name} is required")
    return value


def _captcha_reader(image: bytes, media_type: str) -> str:
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
        value = input("DUT CAPTCHA: ").strip()
        if not value:
            raise ValueError("DUT CAPTCHA must not be empty")
        return value
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _provider(config: AppConfig) -> tuple[str, Any, str]:
    name = config.llm.provider
    selected = config.llm.selected
    token = _secret(selected.token_env, f"{name} DeepSeek Bearer token: ")
    return name, selected, token


def _has_unsafe_reservation(result: dict[str, Any]) -> bool:
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    try:
        return AttemptRecord.model_validate(attempts[-1]).ports_reserved
    except ValueError:
        return True


def run_live(config_path: Path, resume_id: str | None) -> int:
    config = load_config(config_path)
    evaluation_id = resume_id or str(uuid4())
    artifacts = ArtifactStore(config.storage.artifact_dir)
    lock = PortGroupLock(
        config.storage.lock_dir,
        endpoint=config.bps.endpoint,
        slot=config.bps.slot,
        ports=config.bps.ports,
        group=config.bps.group,
        evaluation_id=evaluation_id,
    )
    judge: DeepSeekJudge | None = None
    bps: BpsClient | None = None
    dut: DutClient | None = None
    result: dict[str, Any] | None = None
    try:
        provider_name, provider_config, token = _provider(config)
        judge = DeepSeekJudge(provider_name, provider_config, token=token)
        print(f"Checking {provider_name} provider compatibility with reasoning_effort=max...")
        judge.validate_compatibility()
        bps = BpsClient(
            config.bps,
            username=_username("BPS_USERNAME", "BPS username: "),
            password=_secret("BPS_PASSWORD", "BPS password: "),
        )
        print("Authenticating to BPS...")
        bps.authenticate()
        dut = DutClient(
            config.dut,
            username=_username("DUT_USERNAME", "DUT username: "),
            password=_secret("DUT_PASSWORD", "DUT password: "),
            captcha_reader=_captcha_reader,
        )
        print("Authenticating to DUT (CAPTCHA required)...")
        dut.authenticate()
        config.storage.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        with lock, SqliteSaver.from_conn_string(str(config.storage.checkpoint_db)) as saver:
            services = EvaluationServices(
                config=config,
                bps=bps,
                dut=dut,
                judge=judge,
                artifacts=artifacts,
                clock=SystemClock(),
            )
            graph = build_graph(services, checkpointer=saver)
            invocation_config = {"configurable": {"thread_id": evaluation_id}}
            if resume_id:
                snapshot = graph.get_state(invocation_config)
                if not snapshot.values:
                    raise ValueError(f"no checkpoint exists for Evaluation Run {evaluation_id}")
            try:
                if resume_id:
                    result = graph.invoke(None, config=invocation_config)
                else:
                    result = graph.invoke(
                        initial_state(evaluation_id, config), config=invocation_config
                    )
            except BaseException:
                lock.preserve()
                with suppress(Exception):
                    snapshot = graph.get_state(invocation_config)
                    attempts = snapshot.values.get("attempts", []) if snapshot.values else []
                    if attempts:
                        active = AttemptRecord.model_validate(attempts[-1])
                        if active.bps_run_id and not active.terminal_confirmed:
                            with suppress(Exception):
                                bps.stop_run(active.bps_run_id)
                            LOGGER.error(
                                "Stop requested for BPS run %s; terminal state remains unconfirmed",
                                active.bps_run_id,
                            )
                raise
            if _has_unsafe_reservation(result):
                lock.preserve()
        outcome = str(result["outcome"])
        print(json.dumps({"evaluation_id": evaluation_id, "outcome": outcome}, ensure_ascii=False))
        if result.get("final_artifact"):
            print(f"Audit result: {result['final_artifact']}")
        return EXIT_CODES.get(outcome, 4)
    except KeyboardInterrupt:
        lock.preserve()
        LOGGER.error(
            "Interrupted. The local port-group lock was preserved; "
            "inspect BPS state before cleanup."
        )
        return 130
    finally:
        if result is not None and _has_unsafe_reservation(result):
            LOGGER.error("BPS reservation state is ambiguous; local lock was preserved")
        if dut is not None:
            dut.close()
        if bps is not None:
            bps.close()
        if judge is not None:
            judge.close()


def replay(config_path: Path, evidence_path: Path) -> int:
    config = load_config(config_path)
    evidence = EvidenceBundle.model_validate(json.loads(evidence_path.read_text(encoding="utf-8")))
    provider_name, provider_config, token = _provider(config)
    judge = DeepSeekJudge(provider_name, provider_config, token=token)
    try:
        verdict, raw = judge.adjudicate(evidence)
        output_path = evidence_path.parent / "replay-verdict.json"
        ArtifactStore.write_json(
            output_path,
            {
                "provider": provider_name,
                "model": provider_config.model,
                "parsed": verdict.model_dump(mode="json"),
                "raw_response": raw,
            },
        )
        print(verdict.model_dump_json(indent=2))
        print(f"Replay audit: {output_path}")
        return 0
    finally:
        judge.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangGraph BPS performance-test agent")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    live = subparsers.add_parser("run", help="run against real BPS and DUT devices")
    live.add_argument("--config", type=Path, required=True)
    live.add_argument("--resume", metavar="EVALUATION_ID")
    replay_parser = subparsers.add_parser("replay", help="re-adjudicate saved evidence")
    replay_parser.add_argument("--config", type=Path, required=True)
    replay_parser.add_argument("--evidence", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        if arguments.command == "run":
            return run_live(arguments.config, arguments.resume)
        return replay(arguments.config, arguments.evidence)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 4
