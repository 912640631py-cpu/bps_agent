"""Command-line interface for live Evaluation Runs and replay."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, NoReturn
from uuid import uuid4

from pydantic import ValidationError

from bps_agent.adjudication import verdict_artifact
from bps_agent.artifacts import ArtifactStore
from bps_agent.captcha import read_captcha
from bps_agent.config import apply_overrides, load_config
from bps_agent.credentials import (
    SECRET_CREDENTIALS,
    SUPPORTED_CREDENTIALS,
    CredentialRequirement,
    CredentialResolver,
    CredentialStore,
)
from bps_agent.errors import (
    AgentError,
    CliUsageError,
    CredentialError,
    ErrorCode,
    ReplayError,
)
from bps_agent.graph import build_graph
from bps_agent.models.common import (
    DutCollectionMethod,
    EvaluationMode,
    EvaluationOutcome,
)
from bps_agent.models.config import RunOverrides
from bps_agent.models.evaluation import AttemptRecord, EvidenceBundle
from bps_agent.resume import (
    checkpoint_store,
    has_unsafe_reservation,
    invoke_evaluation,
    load_resume_config,
    validate_resume_request,
)
from bps_agent.runtime import (
    RuntimeResources,
    build_judge,
    build_runtime,
    run_credential_requirements,
)

LOGGER = logging.getLogger(__name__)

EXIT_CODES = {
    EvaluationOutcome.PASSED.value: 0,
    EvaluationOutcome.DEGRADED_PASS.value: 1,
    EvaluationOutcome.NOT_PASSED.value: 2,
    EvaluationOutcome.INCONCLUSIVE.value: 3,
}


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


def run_live(
    config_path: Path,
    resume_id: str | None,
    credential_store: CredentialStore | None = None,
    *,
    stop_before_llm: bool = False,
    download_full_pdf: bool = False,
    overrides: RunOverrides | None = None,
) -> int:
    overrides = overrides or RunOverrides()
    validate_resume_request(resume_id, overrides)
    config = apply_overrides(
        load_config(
            config_path,
            mode_override=overrides.evaluation_mode,
        ),
        overrides,
    )
    if resume_id:
        config = load_resume_config(config, resume_id)
    store = credential_store or CredentialStore()
    credentials = CredentialResolver(store).resolve(
        run_credential_requirements(config, stop_before_llm=stop_before_llm)
    )
    evaluation_id = resume_id or str(uuid4())
    runtime: RuntimeResources | None = None
    result: dict[str, Any] | None = None
    try:
        if stop_before_llm:
            print("Evidence-only mode: DeepSeek will not be contacted.")
        runtime = build_runtime(
            config,
            credentials,
            captcha_reader=read_captcha,
            stop_before_llm=stop_before_llm,
            download_full_pdf=download_full_pdf,
            progress=print,
        )
        with checkpoint_store(config.storage.checkpoint_db) as saver:
            graph = build_graph(
                runtime,
                checkpointer=saver,
                interrupt_before=["adjudicate"] if stop_before_llm else None,
            )
            result = invoke_evaluation(
                graph,
                evaluation_id,
                config,
                resume=resume_id is not None,
                bps=runtime.bps,
            )
            invocation_config = {"configurable": {"thread_id": evaluation_id}}
            if stop_before_llm and result.get("outcome") is None:
                snapshot = graph.get_state(invocation_config)
                if snapshot.next != ("adjudicate",):
                    raise AgentError(
                        "evidence-only run stopped at an unexpected graph position: "
                        f"{snapshot.next!r}",
                        code=ErrorCode.INTERNAL_ERROR,
                    )
                attempts = result.get("attempts", [])
                if not attempts:
                    raise AgentError(
                        "evidence-only run stopped without an Attempt",
                        code=ErrorCode.INTERNAL_ERROR,
                    )
                attempt = AttemptRecord.model_validate(attempts[-1])
                if not attempt.evidence_complete or not attempt.evidence_path:
                    raise AgentError(
                        "evidence-only run stopped without complete Evidence",
                        code=ErrorCode.INTERNAL_ERROR,
                    )
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
        if result.get("error"):
            error_code = result.get("error_code")
            error_message = (
                "Unexpected internal error."
                if error_code == ErrorCode.INTERNAL_ERROR.value
                else str(result["error"])
            )
            if error_code:
                print(f"[{error_code}] {error_message}", file=sys.stderr)
            else:
                print(f"Error: {error_message}", file=sys.stderr)
        verdict_fields = _verdict_console_fields(result)
        if verdict_fields is not None:
            print(json.dumps(verdict_fields, ensure_ascii=False, indent=2))
        if result.get("final_artifact"):
            print(f"Audit result: {result['final_artifact']}")
        return EXIT_CODES.get(outcome, 4)
    finally:
        if result is not None and has_unsafe_reservation(result):
            LOGGER.error(
                "BPS reservation state is ambiguous; inspect BPS before starting another Evaluation"
            )
        if runtime is not None:
            runtime.close()


def replay(
    config_path: Path,
    evidence_path: Path,
    credential_store: CredentialStore | None = None,
) -> int:
    config = load_config(config_path)
    store = credential_store or CredentialStore()
    try:
        evidence = EvidenceBundle.model_validate(ArtifactStore.read_json(evidence_path))
    except FileNotFoundError as exc:
        raise ReplayError(
            f"replay Evidence file was not found: {evidence_path}",
            code=ErrorCode.REPLAY_EVIDENCE_NOT_FOUND,
            hint="Check --evidence and ensure the file exists.",
        ) from exc
    except OSError as exc:
        raise ReplayError(
            f"could not read replay Evidence file: {evidence_path}",
            code=ErrorCode.REPLAY_EVIDENCE_IO_ERROR,
            hint="Check the Evidence file permissions.",
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
        raise ReplayError(
            f"replay Evidence file is invalid: {evidence_path}",
            code=ErrorCode.REPLAY_EVIDENCE_INVALID,
            hint="Provide a valid UTF-8 Evidence Bundle JSON file.",
        ) from exc
    token_name = config.llm.selected.token_env
    credentials = CredentialResolver(store).resolve(
        (
            CredentialRequirement(
                token_name,
                f"{config.llm.provider} DeepSeek Bearer token: ",
                secret=True,
            ),
        )
    )
    judge = build_judge(config, credentials)
    try:
        verdict, raw = judge.adjudicate(evidence)
        output_path = evidence_path.parent / "replay-verdict.json"
        ArtifactStore.write_json(
            output_path,
            verdict_artifact(
                provider=judge.provider_name,
                model=judge.model_name,
                reasoning_effort=judge.reasoning_effort,
                verdict=verdict,
                provider_exchange=raw,
                evidence_path=evidence_path,
            ),
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
            try:
                value = (
                    getpass.getpass(prompt) if name in SECRET_CREDENTIALS else input(prompt).strip()
                )
            except (EOFError, OSError) as exc:
                raise CredentialError(
                    f"{name} is required",
                    code=ErrorCode.CREDENTIAL_MISSING,
                    hint=f"Set {name} in the environment or use an interactive terminal.",
                ) from exc
            store.set(name, value)
            print(f"Saved {name} in the system keyring.")
        return 0

    if command == "delete":
        for name in names:
            deleted = store.delete(name)
            print(f"{name}: {'deleted' if deleted else 'not stored'}")
        return 0

    raise CliUsageError(f"unsupported credentials command: {command}")


class AgentArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliUsageError(message, hint="Run --help to see valid command-line options.")


def _parser() -> argparse.ArgumentParser:
    parser = AgentArgumentParser(description="LangGraph BPS performance-test agent")
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=AgentArgumentParser,
    )
    live = subparsers.add_parser("run", help="run against real BPS with optional DUT monitoring")
    live.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config/demo.yaml"),
        help="configuration file (default: config/demo.yaml)",
    )
    live.add_argument("-r", "--resume", metavar="EVALUATION_ID")
    live.add_argument(
        "-t",
        "--template",
        help="override the exact BPS template name from the configuration",
    )
    live.add_argument(
        "-p",
        "--ports",
        type=int,
        nargs="+",
        metavar="PORT",
        help="override the BPS port numbers from the configuration",
    )
    live.add_argument(
        "-b",
        "--total-bandwidth-mbps",
        type=float,
        help="override the initial BPS Total Bandwidth target in Mbps",
    )
    live.add_argument(
        "-bo",
        "--bps-only",
        action="store_true",
        default=None,
        help="skip DUT credentials, login, keepalive, and monitoring",
    )
    live.add_argument(
        "-m",
        "--dut-collection-method",
        type=DutCollectionMethod,
        choices=tuple(DutCollectionMethod),
        help="override the DUT Collection Method",
    )
    live.add_argument("-dh", "--dut-host", help="override the DUT backend SSH host")
    live.add_argument("-dp", "--dut-port", type=int, help="override the DUT backend SSH port")
    live.add_argument(
        "-i",
        "--dut-interface",
        action="append",
        dest="dut_interfaces",
        help="override a DUT interface; repeat for multiple interfaces",
    )
    live.add_argument(
        "-s",
        "--dut-interval-seconds",
        type=float,
        help="override the DUT backend sampling interval",
    )
    live.add_argument(
        "-sb",
        "--stop-before-llm",
        action="store_true",
        help="collect complete live evidence, then stop before contacting the LLM",
    )
    live.add_argument(
        "-f",
        "--full-pdf",
        action="store_true",
        help="download the complete BPS PDF report in the background",
    )
    replay_parser = subparsers.add_parser("replay", help="re-adjudicate saved evidence")
    replay_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config/demo.yaml"),
        help="configuration file (default: config/demo.yaml)",
    )
    replay_parser.add_argument("-e", "--evidence", type=Path, required=True)
    credentials = subparsers.add_parser(
        "credentials", help="manage credentials in the operating-system keyring"
    )
    credential_commands = credentials.add_subparsers(
        dest="credential_command",
        required=True,
        parser_class=AgentArgumentParser,
    )
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


def _print_agent_error(error: AgentError) -> None:
    print(f"[{error.code}] {error}", file=sys.stderr)
    if error.hint:
        print(f"\nAction: {error.hint}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    verbose_requested = any(
        option in (sys.argv[1:] if argv is None else argv) for option in ("-v", "--verbose")
    )
    try:
        arguments = _parser().parse_args(argv)
        _configure_logging(arguments.verbose)
        if arguments.command == "credentials":
            return manage_credentials(arguments, CredentialStore())
        if arguments.command == "run":
            return run_live(
                arguments.config,
                arguments.resume,
                stop_before_llm=arguments.stop_before_llm,
                download_full_pdf=arguments.full_pdf,
                overrides=RunOverrides(
                    template=arguments.template,
                    ports=tuple(arguments.ports) if arguments.ports is not None else None,
                    total_bandwidth_mbps=arguments.total_bandwidth_mbps,
                    evaluation_mode=(EvaluationMode.BPS_ONLY if arguments.bps_only else None),
                    dut_collection_method=arguments.dut_collection_method,
                    dut_host=arguments.dut_host,
                    dut_port=arguments.dut_port,
                    dut_interfaces=(
                        tuple(arguments.dut_interfaces)
                        if arguments.dut_interfaces is not None
                        else None
                    ),
                    dut_interval_seconds=arguments.dut_interval_seconds,
                ),
            )
        return replay(arguments.config, arguments.evidence)
    except KeyboardInterrupt:
        print(
            "Interrupted by user. Inspect active BPS runs and reservations before retrying.",
            file=sys.stderr,
        )
        return 130
    except CliUsageError as exc:
        _print_agent_error(exc)
        return 64
    except AgentError as exc:
        _print_agent_error(exc)
        return 4
    except Exception:
        if verbose_requested:
            LOGGER.exception("Unexpected internal error")
        _print_agent_error(
            AgentError(
                "Unexpected internal error.",
                code=ErrorCode.INTERNAL_ERROR,
                hint="Re-run with --verbose and inspect the traceback.",
            )
        )
        return 4
