from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO

from .adapters import AdapterError, execute_fake_run
from .config import ConfigError, load_settings
from .orchestrator import OrchestratorError, execute_execute_mode, execute_plan_mode
from .providers import ProviderRegistry
from .routing import AgentRun, RouteValidationError, Router
from .store import SessionStore, StoreError
from .workspace import WorkspaceError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentx", description="Provider-neutral agentic coding gateway")
    parser.add_argument("--json", action="store_true", help="write machine-readable JSON output")

    subparsers = parser.add_subparsers(dest="command")

    providers = subparsers.add_parser("providers", help="inspect provider availability")
    providers_sub = providers.add_subparsers(dest="providers_command")
    providers_sub.add_parser("list", help="list configured providers")

    route = subparsers.add_parser("route", help="explain routing without running a provider")
    route.add_argument("--explain", action="store_true", help="include routing explanation")
    route.add_argument("--provider", default="auto", help="provider id or auto")
    route.add_argument("--mode", default="plan", help="task mode")
    route.add_argument("--model-tier", default=None, help="model tier override")
    route.add_argument("prompt", nargs="?", default="", help="task prompt")

    run = subparsers.add_parser("run", help="execute a local agent run")
    run.add_argument("--fake", action="store_true", help="use the deterministic local fake adapter")
    run.add_argument("--session-id", default="local-fake-run", help="local session/run id")
    run.add_argument("--mode", default="plan", help="task mode")
    run.add_argument("--model-tier", default=None, help="model tier override")
    run.add_argument("prompt", nargs="?", default="", help="task prompt")

    plan = subparsers.add_parser("plan", help="run a read-only plan workflow")
    plan.add_argument("--fake", action="store_true", help="use the deterministic local fake adapter")
    plan.add_argument("--session-id", default="local-plan", help="local session/run id")
    plan.add_argument("--model-tier", default=None, help="model tier override")
    plan.add_argument("--context", action="append", default=None, help="context path to include")
    plan.add_argument("prompt", nargs="?", default="", help="task prompt")

    execute = subparsers.add_parser("execute", help="run a controlled execute workflow")
    execute.add_argument("--fake", action="store_true", help="use the deterministic local fake adapter")
    execute.add_argument("--session-id", default="local-execute", help="local session/run id")
    execute.add_argument("--model-tier", default=None, help="model tier override")
    execute.add_argument("--context", action="append", default=None, help="context path to include")
    execute.add_argument(
        "--allowed-patch",
        action="append",
        default=None,
        help="relative path an adapter patch may target",
    )
    execute.add_argument(
        "--denied-patch",
        action="append",
        default=None,
        help="relative path an adapter patch must not target",
    )
    execute.add_argument("prompt", nargs="?", default="", help="task prompt")

    config = subparsers.add_parser("config", help="inspect configuration")
    config_sub = config.add_subparsers(dest="config_command")
    config_sub.add_parser("show", help="show resolved settings")
    config_sub.add_parser("path", help="show resolved AgentX state paths")

    parser.add_argument(
        "prompt_shorthand",
        nargs="?",
        help="prompt shorthand for a plan-mode route explanation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv or sys.argv[1:], sys.stdout, sys.stderr)


def run(argv: Sequence[str], stdout: TextIO, stderr: TextIO) -> int:
    shorthand = _prompt_shorthand(argv)
    if shorthand is not None:
        json_output, prompt = shorthand
        try:
            settings = load_settings()
        except ConfigError as exc:
            stderr.write(f"agentx: {exc}\n")
            return 2
        return _route(prompt, "plan", "auto", None, settings, json_output, stdout, stderr)

    parser = build_parser()
    args = parser.parse_args(list(argv))

    try:
        settings = load_settings()
    except ConfigError as exc:
        stderr.write(f"agentx: {exc}\n")
        return 2

    if args.command is None and args.prompt_shorthand:
        return _route(
            args.prompt_shorthand,
            "plan",
            "auto",
            None,
            settings,
            args.json,
            stdout,
            stderr,
        )

    if args.command == "providers" and args.providers_command == "list":
        statuses = ProviderRegistry(settings=settings).list_statuses()
        return _write(
            [status.as_dict() for status in statuses],
            args.json,
            stdout,
            text_formatter=_format_providers,
        )

    if args.command == "route":
        return _route(
            args.prompt,
            args.mode,
            args.provider,
            args.model_tier,
            settings,
            args.json,
            stdout,
            stderr,
        )

    if args.command == "run":
        if not args.fake:
            stderr.write("agentx: run currently supports only --fake local execution.\n")
            return 2
        return _fake_run(
            args.prompt,
            args.mode,
            args.model_tier,
            args.session_id,
            settings,
            args.json,
            stdout,
            stderr,
        )

    if args.command == "plan":
        if not args.fake:
            stderr.write("agentx: plan currently supports only --fake local execution.\n")
            return 2
        return _plan(
            args.prompt,
            args.model_tier,
            args.session_id,
            tuple(args.context or ()),
            settings,
            args.json,
            stdout,
            stderr,
        )

    if args.command == "execute":
        if not args.fake:
            stderr.write("agentx: execute currently supports only --fake local execution.\n")
            return 2
        return _execute(
            args.prompt,
            args.model_tier,
            args.session_id,
            tuple(args.context or ()),
            tuple(args.allowed_patch or ()),
            tuple(args.denied_patch or ()),
            settings,
            args.json,
            stdout,
            stderr,
        )

    if args.command == "config" and args.config_command == "show":
        return _write(settings.as_dict(), args.json, stdout)

    if args.command == "config" and args.config_command == "path":
        return _write(settings.paths.as_dict(), args.json, stdout)

    parser.print_help(stdout)
    return 0


def _prompt_shorthand(argv: Sequence[str]) -> tuple[bool, str] | None:
    tokens = list(argv)
    json_output = False
    if tokens and tokens[0] == "--json":
        json_output = True
        tokens = tokens[1:]

    if not tokens or tokens[0] in {"providers", "route", "run", "plan", "execute", "config", "-h", "--help"}:
        return None

    if tokens[0].startswith("-"):
        return None

    return json_output, " ".join(tokens)


def _route(
    prompt: str,
    mode: str,
    provider: str,
    model_tier: str | None,
    settings,
    json_output: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    statuses = ProviderRegistry(settings=settings).list_statuses()
    try:
        decision = Router(settings, statuses).explain(
            AgentRun(prompt=prompt, mode=mode, provider=provider, model_tier=model_tier)
        )
    except RouteValidationError as exc:
        stderr.write(f"agentx: {exc}\n")
        return 2
    return _write(decision.as_dict(), json_output, stdout)


def _fake_run(
    prompt: str,
    mode: str,
    model_tier: str | None,
    session_id: str,
    settings,
    json_output: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        run = AgentRun(
            prompt=prompt,
            mode=mode,
            provider="fake-local",
            model_tier=model_tier,
        )
        stored = execute_fake_run(
            session_store=SessionStore(settings.paths),
            session_id=session_id,
            run=run,
        )
    except (AdapterError, RouteValidationError, StoreError, OSError) as exc:
        stderr.write(f"agentx: {exc}\n")
        return 2

    return _write(
        stored.as_dict(),
        json_output,
        stdout,
        text_formatter=_format_fake_run,
    )


def _plan(
    prompt: str,
    model_tier: str | None,
    session_id: str,
    context_paths: tuple[str, ...],
    settings,
    json_output: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        run = AgentRun(
            prompt=prompt,
            mode="plan",
            provider="auto",
            model_tier=model_tier,
            context_paths=context_paths,
        )
        result = execute_plan_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id=session_id,
            run=run,
        )
    except (
        AdapterError,
        OrchestratorError,
        RouteValidationError,
        StoreError,
        OSError,
    ) as exc:
        stderr.write(f"agentx: {exc}\n")
        return 2

    return _write(
        result.as_dict(),
        json_output,
        stdout,
        text_formatter=_format_plan,
    )


def _execute(
    prompt: str,
    model_tier: str | None,
    session_id: str,
    context_paths: tuple[str, ...],
    allowed_patch_paths: tuple[str, ...],
    denied_patch_paths: tuple[str, ...],
    settings,
    json_output: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        run = AgentRun(
            prompt=prompt,
            mode="execute",
            provider="auto",
            model_tier=model_tier,
            context_paths=context_paths,
        )
        result = execute_execute_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id=session_id,
            run=run,
            allowed_patch_paths=allowed_patch_paths,
            denied_patch_paths=denied_patch_paths,
        )
    except (
        AdapterError,
        OrchestratorError,
        RouteValidationError,
        StoreError,
        WorkspaceError,
        OSError,
    ) as exc:
        stderr.write(f"agentx: {exc}\n")
        return 2

    return _write(
        result.as_dict(),
        json_output,
        stdout,
        text_formatter=_format_execute,
    )


def _write(value, json_output: bool, stdout: TextIO, text_formatter=None) -> int:
    if json_output:
        stdout.write(json.dumps(value, indent=2, sort_keys=True))
        stdout.write("\n")
        return 0

    if text_formatter:
        stdout.write(text_formatter(value))
        return 0

    stdout.write(json.dumps(value, indent=2, sort_keys=True))
    stdout.write("\n")
    return 0


def _format_providers(statuses: list[dict[str, object]]) -> str:
    lines = []
    for status in statuses:
        enabled = "enabled" if status["enabled"] else "disabled"
        lines.append(f"{status['id']}\t{enabled}\t{status['reason']}")
    return "\n".join(lines) + "\n"


def _format_fake_run(payload: dict[str, object]) -> str:
    return f"wrote fake run artifacts to {payload['root']}\n"


def _format_plan(payload: dict[str, object]) -> str:
    route = payload["route"]
    return (
        f"wrote plan artifacts to {payload['root']}\n"
        f"{route['explanation']}\n"
    )


def _format_execute(payload: dict[str, object]) -> str:
    validation = payload["patch_validation"]
    status = "accepted" if validation["accepted"] else "rejected"
    return (
        f"wrote execute artifacts to {payload['root']}\n"
        f"patch validation: {status}; patch applied: false\n"
    )
