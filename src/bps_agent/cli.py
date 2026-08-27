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

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

from bps_agent.adapters.bps import BpsClient
from bps_agent.adapters.deepseek import DeepSeekJudge
from bps_agent.adapters.dut import DutClient
from bps_agent.artifacts import ArtifactStore
from bps_agent.config import load_config
from bps_agent.credentials import (
    SECRET_CREDENTIALS,
    SUPPORTED_CREDENTIALS,
    CredentialStore,
)
from bps_agent.graph import EvaluationServices, SystemClock, build_graph, initial_state
from bps_agent.models import (
    AppConfig,
    AttemptRecord,
    EvaluationMode,
    EvaluationOutcome,
    EvidenceBundle,
)

LOGGER = logging.getLogger(__name__)

EXIT_CODES = {
    EvaluationOutcome.PASSED.value: 0,
    EvaluationOutcome.DEGRADED_PASS.value: 1,
    EvaluationOutcome.NOT_PASSED.value: 2,
    EvaluationOutcome.INCONCLUSIVE.value: 3,
}


class _EvidenceOnlyJudge:
    provider_name = "not-called"
    model_name = "not-called"

    def adjudicate(self, evidence: EvidenceBundle) -> tuple[Any, dict[str, Any]]:
        raise RuntimeError("LLM adjudication is disabled for this evidence-only run")

    def close(self) -> None:
        return None


def _credential(
    store: CredentialStore,
    name: str,
    prompt: str,
    *,
    secret: bool,
) -> str:
    value = os.environ.get(name)
    if value:
        return value
    value = store.get(name)
    if value:
        return value
    value = getpass.getpass(prompt) if secret else input(prompt).strip()
    if not value:
        raise ValueError(f"{name} is required")
    store.set(name, value)
    print(f"Saved {name} in the system keyring.")
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


def _provider(config: AppConfig, store: CredentialStore) -> tuple[str, Any, str]:
    name = config.llm.provider
    selected = config.llm.selected
    token = _credential(
        store,
        selected.token_env,
        f"{name} DeepSeek Bearer token: ",
        secret=True,
    )
    return name, selected, token


def _has_unsafe_reservation(result: dict[str, Any]) -> bool:
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    try:
        return AttemptRecord.model_validate(attempts[-1]).ports_reserved
    except ValueError:
        return True


def _verdict_console_fields(result: dict[str, Any]) -> dict[str, Any] | None:
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    try:
        verdict = AttemptRecord.model_validate(attempts[-1]).verdict
    except ValueError:
        return None
    if verdict is None:
        return None
    return {"parsed": verdict.model_dump(mode="json")}


def _apply_bps_overrides(
    config: AppConfig,
    *,
    template: str | None,
    ports: tuple[int, ...] | None,
    total_bandwidth_mbps: float | None,
    resume_id: str | None,
    bps_only: bool | None = None,
) -> AppConfig:
    if resume_id and (
        template is not None or ports is not None or total_bandwidth_mbps is not None or bps_only
    ):
        raise ValueError(
            "--template, --ports, --total-bandwidth-mbps, and --bps-only "
            "cannot be used with --resume"
        )
    document = config.model_dump(mode="python")
    if template is not None:
        cleaned = template.strip()
        if not cleaned:
            raise ValueError("--template must not be empty")
        document["bps"]["template"] = cleaned
    if ports is not None:
        document["bps"]["ports"] = ports
    if total_bandwidth_mbps is not None:
        document["bps"]["total_bandwidth_mbps"] = total_bandwidth_mbps
    if bps_only:
        document["evaluation"]["mode"] = EvaluationMode.BPS_ONLY.value
    return AppConfig.model_validate(document)


def _load_resume_config(current_config: AppConfig, resume_id: str) -> AppConfig:
    checkpoint_db = current_config.storage.checkpoint_db
    invocation_config: RunnableConfig = {"configurable": {"thread_id": resume_id}}
    with SqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
        checkpoint_tuple = saver.get_tuple(invocation_config)
    if checkpoint_tuple is None:
        raise ValueError(f"no checkpoint exists for Evaluation Run {resume_id}")
    channel_values = checkpoint_tuple.checkpoint.get("channel_values")
    if not isinstance(channel_values, dict) or not isinstance(channel_values.get("config"), dict):
        raise ValueError(f"checkpoint for Evaluation Run {resume_id} omitted its configuration")
    checkpoint_config = dict(channel_values["config"])
    storage = checkpoint_config.get("storage")
    if isinstance(storage, dict) and "lock_dir" in storage:
        checkpoint_config["storage"] = {
            name: value for name, value in storage.items() if name != "lock_dir"
        }
    try:
        restored = AppConfig.model_validate(checkpoint_config)
    except ValueError as exc:
        raise ValueError(
            f"checkpoint for Evaluation Run {resume_id} contains an invalid configuration"
        ) from exc
    if restored.storage.checkpoint_db.resolve() != checkpoint_db.resolve():
        raise ValueError(
            f"checkpoint for Evaluation Run {resume_id} refers to a different checkpoint database"
        )
    return restored


def run_live(
    config_path: Path,
    resume_id: str | None,
    credential_store: CredentialStore | None = None,
    *,
    stop_before_llm: bool = False,
    template: str | None = None,
    ports: tuple[int, ...] | None = None,
    total_bandwidth_mbps: float | None = None,
    bps_only: bool | None = None,
) -> int:
    config = _apply_bps_overrides(
        load_config(config_path),
        template=template,
        ports=ports,
        total_bandwidth_mbps=total_bandwidth_mbps,
        resume_id=resume_id,
        bps_only=bps_only,
    )
    if resume_id:
        config = _load_resume_config(config, resume_id)
    store = credential_store or CredentialStore()
    evaluation_id = resume_id or str(uuid4())
    artifacts = ArtifactStore(config.storage.artifact_dir)
    judge: DeepSeekJudge | _EvidenceOnlyJudge | None = None
    bps: BpsClient | None = None
    dut: DutClient | None = None
    result: dict[str, Any] | None = None
    try:
        if stop_before_llm:
            judge = _EvidenceOnlyJudge()
            print("Evidence-only mode: DeepSeek will not be contacted.")
        else:
            provider_name, provider_config, token = _provider(config, store)
            judge = DeepSeekJudge(provider_name, provider_config, token=token)
            print(f"Checking {provider_name} provider compatibility with reasoning_effort=max...")
            judge.validate_compatibility()
        bps = BpsClient(
            config.bps,
            username=_credential(store, "BPS_USERNAME", "BPS username: ", secret=False),
            password=_credential(store, "BPS_PASSWORD", "BPS password: ", secret=True),
        )
        print("Authenticating to BPS...")
        bps.authenticate()
        if config.evaluation.mode == EvaluationMode.BPS_ONLY:
            print("BPS-only mode: DUT authentication and monitoring are disabled.")
        else:
            dut = DutClient(
                config.dut,
                username=_credential(store, "DUT_USERNAME", "DUT username: ", secret=False),
                password=_credential(store, "DUT_PASSWORD", "DUT password: ", secret=True),
                captcha_reader=_captcha_reader,
            )
            print("Authenticating to DUT (CAPTCHA required)...")
            dut.authenticate()
        config.storage.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(config.storage.checkpoint_db)) as saver:
            services = EvaluationServices(
                config=config,
                bps=bps,
                dut=dut,
                judge=judge,
                artifacts=artifacts,
                clock=SystemClock(),
            )
            graph = build_graph(
                services,
                checkpointer=saver,
                interrupt_before=["adjudicate"] if stop_before_llm else None,
            )
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
            if stop_before_llm and result.get("outcome") is None:
                snapshot = graph.get_state(invocation_config)
                if snapshot.next != ("adjudicate",):
                    raise RuntimeError(
                        "evidence-only run stopped at an unexpected graph position: "
                        f"{snapshot.next!r}"
                    )
                attempts = result.get("attempts", [])
                if not attempts:
                    raise RuntimeError("evidence-only run stopped without an Attempt")
                attempt = AttemptRecord.model_validate(attempts[-1])
                if not attempt.evidence_complete or not attempt.evidence_path:
                    raise RuntimeError("evidence-only run stopped without complete Evidence")
                print(
                    json.dumps(
                        {
                            "evaluation_id": evaluation_id,
                            "status": "EVIDENCE_READY",
                            "attempt": attempt.number,
                        },
                        ensure_ascii=False,
                    )
                )
                print(f"Evidence: {attempt.evidence_path}")
                return 0
        outcome = str(result["outcome"])
        print(json.dumps({"evaluation_id": evaluation_id, "outcome": outcome}, ensure_ascii=False))
        verdict_fields = _verdict_console_fields(result)
        if verdict_fields is not None:
            print(json.dumps(verdict_fields, ensure_ascii=False, indent=2))
        if result.get("final_artifact"):
            print(f"Audit result: {result['final_artifact']}")
        return EXIT_CODES.get(outcome, 4)
    except KeyboardInterrupt:
        LOGGER.error(
            "Interrupted. Inspect BPS run and reservation state before starting another Evaluation."
        )
        return 130
    finally:
        if result is not None and _has_unsafe_reservation(result):
            LOGGER.error(
                "BPS reservation state is ambiguous; inspect BPS before starting another Evaluation"
            )
        if dut is not None:
            dut.close()
        if bps is not None:
            bps.close()
        if judge is not None:
            judge.close()


def replay(
    config_path: Path,
    evidence_path: Path,
    credential_store: CredentialStore | None = None,
) -> int:
    config = load_config(config_path)
    store = credential_store or CredentialStore()
    evidence = EvidenceBundle.model_validate(json.loads(evidence_path.read_text(encoding="utf-8")))
    provider_name, provider_config, token = _provider(config, store)
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


def manage_credentials(arguments: argparse.Namespace, store: CredentialStore) -> int:
    command = arguments.credential_command
    if command == "status":
        for name, saved in store.status().items():
            environment = "set" if os.environ.get(name) else "unset"
            keyring_status = "stored" if saved else "missing"
            print(f"{name}: keyring={keyring_status}, environment={environment}")
        return 0

    names = (
        tuple(CredentialStore.validate_name(name) for name in arguments.names)
        or SUPPORTED_CREDENTIALS
    )
    if command == "set":
        for name in names:
            prompt = f"{name}: "
            value = getpass.getpass(prompt) if name in SECRET_CREDENTIALS else input(prompt).strip()
            store.set(name, value)
            print(f"Saved {name} in the system keyring.")
        return 0

    if command == "delete":
        for name in names:
            deleted = store.delete(name)
            print(f"{name}: {'deleted' if deleted else 'not stored'}")
        return 0

    raise ValueError(f"unsupported credentials command: {command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangGraph BPS performance-test agent")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    live = subparsers.add_parser("run", help="run against real BPS with optional DUT monitoring")
    live.add_argument(
        "--config",
        type=Path,
        default=Path("config/demo.yaml"),
        help="configuration file (default: config/demo.yaml)",
    )
    live.add_argument("--resume", metavar="EVALUATION_ID")
    live.add_argument(
        "--template",
        help="override the exact BPS template name from the configuration",
    )
    live.add_argument(
        "--ports",
        type=int,
        nargs="+",
        metavar="PORT",
        help="override the BPS port numbers from the configuration",
    )
    live.add_argument(
        "--total-bandwidth-mbps",
        type=float,
        help="override the initial BPS Total Bandwidth target in Mbps",
    )
    live.add_argument(
        "--bps-only",
        action="store_true",
        default=None,
        help="skip DUT credentials, login, keepalive, and monitoring",
    )
    live.add_argument(
        "--stop-before-llm",
        action="store_true",
        help="collect complete live evidence, then stop before contacting the LLM",
    )
    replay_parser = subparsers.add_parser("replay", help="re-adjudicate saved evidence")
    replay_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/demo.yaml"),
        help="configuration file (default: config/demo.yaml)",
    )
    replay_parser.add_argument("--evidence", type=Path, required=True)
    credentials = subparsers.add_parser(
        "credentials", help="manage credentials in the operating-system keyring"
    )
    credential_commands = credentials.add_subparsers(dest="credential_command", required=True)
    credential_commands.add_parser("status", help="show presence without revealing values")
    set_credentials = credential_commands.add_parser("set", help="save credentials")
    set_credentials.add_argument("names", nargs="*")
    delete_credentials = credential_commands.add_parser("delete", help="delete saved credentials")
    delete_credentials.add_argument("names", nargs="*")
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    _configure_logging(arguments.verbose)
    try:
        if arguments.command == "credentials":
            return manage_credentials(arguments, CredentialStore())
        if arguments.command == "run":
            return run_live(
                arguments.config,
                arguments.resume,
                stop_before_llm=arguments.stop_before_llm,
                template=arguments.template,
                ports=tuple(arguments.ports) if arguments.ports is not None else None,
                total_bandwidth_mbps=arguments.total_bandwidth_mbps,
                bps_only=arguments.bps_only,
            )
        return replay(arguments.config, arguments.evidence)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 4
