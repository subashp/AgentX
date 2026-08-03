from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
import threading
import uuid
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence, TextIO

try:
    import select
    import termios
    import tty
except ImportError:  # pragma: no cover - only used for interactive POSIX terminals.
    select = None
    termios = None
    tty = None

from .adapters import (
    AdapterError,
    AdapterRequest,
    CliPlanAdapter,
    CodexCliAdapter,
    execute_adapter_run,
    execute_fake_run,
)
from .config import ConfigError, ProviderSettings, Settings, load_settings
from .memory import (
    AgentXMemoryError,
    append_interaction_events,
    apply_memory_proposal,
    call_memory_tool,
    list_memory_proposals,
)
from .openai_compatible import OpenAICompatibleAdapter, RequestCancellation
from .orchestrator import OrchestratorError, execute_execute_mode, execute_plan_mode
from .subagents import SubagentManager, SubagentTask, SubagentTools
from .tools import ApprovalCallback, ToolError
from .tool_registry import build_private_tool_executor, tool_names
from .workspace import WorkspaceError, normalize_scoped_path
from .providers import ProviderRegistry
from .routing import AgentRun, RouteValidationError, Router
from .store import SessionStore, SettingsStore, StoreError


_INTERACTIVE_SLASH_COMMANDS = (
    "/provider",
    "/providers",
    "/context",
    "/memory",
    "/tools",
    "/execute",
    "/help",
    "/quit",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentx", description="Provider-neutral agentic coding gateway")
    parser.add_argument("--json", action="store_true", help="write machine-readable JSON output")
    parser.add_argument(
        "--provider",
        dest="global_provider",
        default=None,
        metavar="PROVIDER",
        help="provider to use when entering interactive mode (default: auto)",
    )

    subparsers = parser.add_subparsers(dest="command")

    init = subparsers.add_parser("init", help="initialize AgentX settings")
    init.add_argument(
        "--profile",
        choices=("agentx", "codex", "private-openai-compatible", "halo"),
        default="agentx",
        help="settings profile to write",
    )
    init.add_argument("--force", action="store_true", help="overwrite an existing settings file")
    init.add_argument("--codex-command", default="codex", help="Codex command for the codex profile")
    init.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL for the private provider profile")
    init.add_argument("--model", default=None, help="model ID for the private provider profile")
    init.add_argument("--api-key-env", default=None, help="environment variable containing the private provider API key")
    init.add_argument("--timeout", type=float, default=60.0, help="private provider request timeout in seconds")

    interactive = subparsers.add_parser(
        "interactive",
        aliases=("shell",),
        help="enter the provider-aware interactive CLI",
    )
    interactive.add_argument(
        "--provider",
        default=None,
        metavar="PROVIDER",
        help="provider to use for this session (default: prompt for auto)",
    )

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
    plan.add_argument("--provider", default="auto", help="provider id or auto")
    plan.add_argument("--session-id", default="local-plan", help="local session/run id")
    plan.add_argument("--source-root", default=".", help="source workspace root for scoped context")
    plan.add_argument("--workspace-id", default=None, help="scoped workspace id")
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

    memory = subparsers.add_parser("memory", help="inspect and manage AgentX memory")
    memory_sub = memory.add_subparsers(dest="memory_command")
    memory_remember = memory_sub.add_parser("remember", help="store an explicit memory")
    memory_remember.add_argument("--class", dest="privacy_class", choices=("generic", "team", "private"), required=True)
    memory_remember.add_argument("content", nargs="+")
    memory_search = memory_sub.add_parser("search", help="search memory")
    memory_search.add_argument("query", nargs="?", default="")
    memory_search.add_argument("--class", dest="privacy_class", choices=("generic", "team", "private"), default=None)
    memory_show = memory_sub.add_parser("show", help="show one memory")
    memory_show.add_argument("memory_id")
    memory_correct = memory_sub.add_parser("correct", help="correct one memory")
    memory_correct.add_argument("memory_id")
    memory_correct.add_argument("replacement", nargs="+")
    memory_forget = memory_sub.add_parser("forget", help="delete memory")
    memory_forget.add_argument("memory_id", nargs="?")
    memory_forget.add_argument("--all", action="store_true")
    memory_forget.add_argument("--soft", action="store_true")
    memory_proposals = memory_sub.add_parser("proposals", help="list memory proposals")
    memory_proposals.add_argument("--status", default=None)
    memory_apply = memory_sub.add_parser("apply", help="apply one memory proposal")
    memory_apply.add_argument("proposal_id")

    parser.add_argument(
        "prompt_shorthand",
        nargs="?",
        help="prompt shorthand for a plan-mode route explanation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(sys.argv[1:] if argv is None else argv, sys.stdout, sys.stderr)


def run(
    argv: Sequence[str],
    stdout: TextIO,
    stderr: TextIO,
    stdin: TextIO | None = None,
) -> int:
    input_stream = sys.stdin if stdin is None else stdin
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
            args.global_provider or "auto",
            None,
            settings,
            args.json,
            stdout,
            stderr,
        )

    if args.command is None:
        if args.json:
            stderr.write("agentx: interactive mode does not support --json; use a subcommand.\n")
            return 2
        return _interactive(
            settings,
            provider=args.global_provider,
            stdin=input_stream,
            stdout=stdout,
            stderr=stderr,
        )

    if args.command == "init":
        return _init(
            args.profile,
            args.force,
            args.codex_command,
            args.endpoint,
            args.model,
            args.api_key_env,
            args.timeout,
            settings,
            args.json,
            stdout,
            stderr,
        )

    if args.command in {"interactive", "shell"}:
        if args.json:
            stderr.write("agentx: interactive mode does not support --json; use a subcommand.\n")
            return 2
        return _interactive(
            settings,
            provider=args.provider,
            stdin=input_stream,
            stdout=stdout,
            stderr=stderr,
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
        provider = _resolve_plan_provider(
            fake=args.fake,
            requested_provider=args.provider,
            settings=settings,
        )
        if provider not in {
            "auto",
            "codex",
            "claude",
            "kiro",
            "private-openai-compatible",
            "fake-local",
        }:
            stderr.write(
                f"agentx: plan provider '{provider}' is not supported; use codex, claude, kiro, private-openai-compatible, or fake-local.\n"
            )
            return 2
        return _plan(
            args.prompt,
            provider,
            args.model_tier,
            args.session_id,
            args.source_root,
            args.workspace_id,
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

    if args.command == "memory":
        return _memory_command(args, settings, args.json, stdout, stderr)

    parser.print_help(stdout)
    return 0


def _prompt_shorthand(argv: Sequence[str]) -> tuple[bool, str] | None:
    tokens = list(argv)
    json_output = False
    if tokens and tokens[0] == "--json":
        json_output = True
        tokens = tokens[1:]

    if not tokens or tokens[0] in {
        "init",
        "interactive",
        "shell",
        "providers",
        "route",
        "run",
        "plan",
        "execute",
        "config",
        "memory",
        "-h",
        "--help",
    }:
        return None

    if tokens[0].startswith("-"):
        return None

    return json_output, " ".join(tokens)


def _interactive(
    settings,
    *,
    provider: str | None,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    statuses = ProviderRegistry(settings=settings).list_statuses()
    provider_ids = ("auto",) + tuple(status.id for status in statuses)
    selected_provider = provider.strip().lower() if provider else None

    stdout.write("AgentX interactive mode\n")
    stdout.write("Type /help for commands or /quit to exit.\n")

    _write_custom_provider_startup_warning(settings, statuses, stdout)

    if selected_provider is None:
        stdout.write("\nProviders:\n")
        stdout.write("  1. auto - route by policy and availability\n")
        for index, status in enumerate(statuses, start=2):
            availability = "available" if status.enabled else f"unavailable: {status.reason}"
            stdout.write(f"  {index}. {status.id} - {status.display_name} ({availability})\n")
        stdout.write("Select provider [auto]: ")
        stdout.flush()
        choice = stdin.readline().strip()
        if choice.lower() in {"/quit", "/exit", ":q"}:
            return 0
        if choice.isdigit():
            choice_index = int(choice)
            if choice_index == 1:
                selected_provider = "auto"
            elif 2 <= choice_index <= len(statuses) + 1:
                selected_provider = statuses[choice_index - 2].id
        elif choice:
            selected_provider = choice.lower()
        else:
            selected_provider = "auto"

    if selected_provider not in provider_ids:
        stderr.write(
            f"agentx: unknown provider '{selected_provider}'. Use /providers to list providers.\n"
        )
        return 2
    if selected_provider != "auto":
        selected_status = next(status for status in statuses if status.id == selected_provider)
        if not selected_status.enabled:
            stderr.write(
                f"agentx: provider '{selected_provider}' is unavailable: {selected_status.reason}.\n"
            )
            return 2

    context_paths: tuple[str, ...] = ()
    web_approval = _InteractiveApproval(stdin, stdout)
    request_cancellation = _InteractiveRequestCancellation(stdin, stdout)

    while True:
        line = _read_interactive_line(
            f"\nagentx[{selected_provider}]> ",
            stdin,
            stdout,
            provider_ids=provider_ids,
        )
        if line == "":
            stdout.write("\n")
            return 0
        prompt = line.strip()
        if not prompt:
            continue
        command = prompt.lower()
        if command in {"/quit", "/exit", ":q"}:
            return 0
        if command in {"/help", "help"}:
            stdout.write(
                "Commands: /provider [id|auto], /providers, /context [clear|path...], "
                "/memory [search|remember|show|correct|forget|proposals|apply], /tools, /execute <task>, /help, /quit. "
                "Any other input is treated as a coding task; press Esc while a private-model request is active to cancel it.\n"
            )
            continue
        if command == "/providers":
            for status in statuses:
                availability = "available" if status.enabled else f"unavailable: {status.reason}"
                stdout.write(f"  {status.id}: {availability}\n")
            continue
        if command.startswith("/provider"):
            parts = prompt.split(maxsplit=1)
            if len(parts) == 1:
                stdout.write(f"Current provider: {selected_provider}\n")
                continue
            requested = parts[1].strip().lower()
            if requested not in provider_ids:
                stderr.write(f"agentx: unknown provider '{requested}'.\n")
                continue
            if requested != "auto":
                requested_status = next(status for status in statuses if status.id == requested)
                if not requested_status.enabled:
                    stderr.write(
                        f"agentx: provider '{requested}' is unavailable: {requested_status.reason}.\n"
                    )
                    continue
            selected_provider = requested
            stdout.write(f"Provider changed to {selected_provider}.\n")
            continue
        if command == "/memory" or command.startswith("/memory "):
            _interactive_memory(prompt, settings, stdout, stderr)
            continue
        if command == "/tools":
            _interactive_tools(selected_provider, settings, context_paths, web_approval, stdout, stderr)
            continue
        if command == "/execute":
            stderr.write("agentx: /execute requires a task prompt.\n")
            continue
        if command.startswith("/execute "):
            execute_prompt = prompt.split(maxsplit=1)[1].strip()
            if not execute_prompt:
                stderr.write("agentx: /execute requires a task prompt.\n")
                continue
            session_id = f"interactive-{uuid.uuid4().hex[:12]}"
            task_provider = selected_provider
            if selected_provider == "auto":
                task_provider = _select_auto_interactive_provider(settings, statuses)
            if task_provider != "private-openai-compatible":
                stderr.write("agentx: /execute currently requires provider private-openai-compatible.\n")
                continue
            try:
                _plan(
                    execute_prompt,
                    task_provider,
                    None,
                    session_id,
                    ".",
                    None,
                    context_paths,
                    settings,
                    False,
                    stdout,
                    stderr,
                    interactive_output=True,
                    web_approval=web_approval,
                    request_cancellation=request_cancellation,
                    enable_patch_tool=True,
                    enable_shell_tool=True,
                    run_mode="execute",
                )
            except KeyboardInterrupt:
                stdout.write("\nRequest cancelled. Returning to AgentX.\n")
            continue
        if command == "/context" or command.startswith("/context "):
            parts = prompt.split(maxsplit=1)
            if len(parts) == 1:
                if context_paths:
                    stdout.write("Context paths:\n")
                    stdout.writelines(f"  {path}\n" for path in context_paths)
                else:
                    stdout.write("Context paths: none.\n")
                continue
            requested_paths = parts[1].split()
            if len(requested_paths) == 1 and requested_paths[0].lower() == "clear":
                context_paths = ()
                stdout.write("Context paths cleared.\n")
                continue
            try:
                normalized_paths = tuple(
                    dict.fromkeys(normalize_scoped_path(path) for path in requested_paths)
                )
            except WorkspaceError as exc:
                stderr.write(f"agentx: invalid context path: {exc}\n")
                continue
            context_paths = normalized_paths
            stdout.write("Context paths set:\n")
            stdout.writelines(f"  {path}\n" for path in context_paths)
            continue

        session_id = f"interactive-{uuid.uuid4().hex[:12]}"
        task_provider = selected_provider
        if selected_provider == "auto":
            task_provider = _select_auto_interactive_provider(settings, statuses)
        if task_provider in {"codex", "claude", "kiro", "private-openai-compatible"}:
            try:
                _plan(
                    prompt,
                    task_provider,
                    None,
                    session_id,
                    ".",
                    None,
                    context_paths,
                    settings,
                    False,
                    stdout,
                    stderr,
                    interactive_output=True,
                    web_approval=web_approval,
                    request_cancellation=request_cancellation,
                )
            except KeyboardInterrupt:
                stdout.write("\nRequest cancelled. Returning to AgentX.\n")
        elif selected_provider == "fake-local":
            _plan(
                prompt,
                "fake-local",
                None,
                session_id,
                ".",
                None,
                context_paths,
                settings,
                False,
                stdout,
                stderr,
                interactive_output=True,
                web_approval=web_approval,
            )
        else:
            if selected_provider != "auto":
                stdout.write(
                    f"{selected_provider} is currently available for routing explanations only.\n"
                )
            _route(prompt, "plan", selected_provider, None, settings, False, stdout, stderr)


def _select_auto_interactive_provider(settings, statuses) -> str:
    status_by_id = {status.id: status for status in statuses}
    candidates = list(settings.public_providers)
    if settings.private_provider:
        candidates.append(settings.private_provider)
    candidates.extend(status.id for status in statuses)
    for provider_id in candidates:
        status = status_by_id.get(provider_id)
        if status is not None and status.enabled:
            return provider_id
    return "auto"


def _write_custom_provider_startup_warning(settings, statuses, stdout) -> None:
    custom = next(
        (status for status in statuses if status.id == "private-openai-compatible"),
        None,
    )
    if custom is None or custom.enabled:
        return

    stdout.write(
        "\nWarning: custom model provider is unavailable "
        f"({custom.reason}).\n"
    )
    stdout.write(f"External settings file: {settings.paths.settings}\n")
    if custom.reason == "endpoint_not_configured":
        stdout.write(
            "Configure it before starting work, for example:\n"
            "  agentx init --profile private-openai-compatible "
            "--endpoint <local-or-ngrok-url>/v1 --model <model-id> --force\n"
        )
    elif custom.reason == "model_not_configured":
        stdout.write(
            "Add the custom model ID to that settings file or rerun the private-provider init profile.\n"
        )
    else:
        stdout.write("Check the endpoint and model settings before selecting the custom provider.\n")


def _slash_command_completion_tree(provider_ids: Sequence[str]) -> dict[str, object]:
    provider_tree = {provider_id: None for provider_id in provider_ids}
    memory_class_tree = {"generic": None, "team": None, "private": None}
    tree = {command: None for command in _INTERACTIVE_SLASH_COMMANDS}
    tree.update({
        "/provider": provider_tree,
        "/context": {"clear": None},
        "/memory": {
            "search": None,
            "remember": {"--class": memory_class_tree},
            "show": None,
            "correct": None,
            "forget": {"--all": None},
            "proposals": None,
            "apply": None,
        },
    })
    return tree


def _read_interactive_line(
    prompt_text: str,
    stdin: TextIO,
    stdout: TextIO,
    *,
    provider_ids: Sequence[str],
) -> str:
    if stdin is not sys.stdin or stdout is not sys.stdout:
        stdout.write(prompt_text)
        stdout.flush()
        return stdin.readline()
    stdin_isatty = getattr(stdin, "isatty", None)
    stdout_isatty = getattr(stdout, "isatty", None)
    if not callable(stdin_isatty) or not stdin_isatty():
        stdout.write(prompt_text)
        stdout.flush()
        return stdin.readline()
    if not callable(stdout_isatty) or not stdout_isatty():
        stdout.write(prompt_text)
        stdout.flush()
        return stdin.readline()

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import NestedCompleter
    except ImportError:
        stdout.write(prompt_text)
        stdout.flush()
        return stdin.readline()

    try:
        session = PromptSession(
            completer=NestedCompleter.from_nested_dict(
                _slash_command_completion_tree(provider_ids)
            ),
            complete_while_typing=True,
        )
        return session.prompt(prompt_text) + "\n"
    except EOFError:
        return ""


def _interactive_memory(prompt: str, settings, stdout: TextIO, stderr: TextIO) -> None:
    try:
        tokens = shlex.split(prompt)
    except ValueError as exc:
        stderr.write(f"agentx: invalid /memory command: {exc}\n")
        return
    args = tokens[1:]
    if not args:
        args = ["search", ""]
    command = args[0].lower()
    try:
        if command == "search":
            query = " ".join(args[1:]) if len(args) > 1 else ""
            payload = call_memory_tool(settings.paths, "memory_search", {"query": query})
            stdout.write(_format_memory_search(payload))
            return
        if command == "remember":
            privacy_class = None
            content_tokens: list[str] = []
            index = 1
            while index < len(args):
                token = args[index]
                if token == "--class" and index + 1 < len(args):
                    privacy_class = args[index + 1]
                    index += 2
                    continue
                content_tokens.append(token)
                index += 1
            if privacy_class not in {"generic", "team", "private"}:
                stderr.write("agentx: /memory remember requires --class generic|team|private.\n")
                return
            if not content_tokens:
                stderr.write("agentx: /memory remember requires text.\n")
                return
            payload = call_memory_tool(
                settings.paths,
                "memory_remember",
                {"content": " ".join(content_tokens), "privacy_class": privacy_class},
            )
            stdout.write(_format_memory_record(payload))
            return
        if command == "show" and len(args) == 2:
            payload = call_memory_tool(settings.paths, "memory_show", {"memory_id": args[1]})
            stdout.write(_format_memory_record(payload))
            return
        if command == "correct" and len(args) >= 3:
            payload = call_memory_tool(
                settings.paths,
                "memory_correct",
                {"memory_id": args[1], "replacement": " ".join(args[2:])},
            )
            stdout.write(_format_memory_record(payload))
            return
        if command == "forget":
            if len(args) == 2 and args[1] == "--all":
                payload = call_memory_tool(settings.paths, "memory_forget", {"all": True, "hard": True})
            elif len(args) == 2:
                payload = call_memory_tool(settings.paths, "memory_forget", {"memory_id": args[1], "hard": True})
            else:
                stderr.write("agentx: /memory forget requires <memory-id> or --all.\n")
                return
            stdout.write(_format_memory_delete(payload))
            return
        if command == "proposals":
            stdout.write(_format_memory_proposals({"proposals": list_memory_proposals(settings.paths)}))
            return
        if command == "apply" and len(args) == 2:
            stdout.write(_format_memory_record(apply_memory_proposal(settings.paths, args[1])))
            return
    except AgentXMemoryError as exc:
        stderr.write(f"agentx: {exc}\n")
        return
    stderr.write(
        "agentx: usage: /memory [search [query]|remember --class <class> <text>|"
        "show <id>|correct <id> <replacement>|forget <id|--all>|proposals|apply <proposal-id>]\n"
    )


def _interactive_tools(
    selected_provider: str,
    settings,
    context_paths: tuple[str, ...],
    web_approval: ApprovalCallback,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    if selected_provider not in {"private-openai-compatible", "auto"}:
        stdout.write(f"{selected_provider} uses its native CLI tool surface; AgentX does not inject tools.\n")
        return
    try:
        executor = build_private_tool_executor(
            mode="plan",
            workspace_root=".",
            context_paths=context_paths,
            paths=settings.paths,
            user_prompt="",
            approval_callback=web_approval,
            artifacts_dir=SessionStore(settings.paths).path_for_session("interactive-tools") / "artifacts",
        )
    except (ToolError, OSError) as exc:
        stderr.write(f"agentx: cannot list tools: {exc}\n")
        return
    stdout.write("Available AgentX tools:\n")
    stdout.writelines(f"  {name}\n" for name in tool_names(executor))
    stdout.write("Use /execute <task> to enable approval-gated workspace_patch and shell_exec.\n")


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


def _init(
    profile: str,
    force: bool,
    codex_command: str,
    endpoint: str | None,
    model: str | None,
    api_key_env: str | None,
    timeout: float,
    settings,
    json_output: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        initialized = _settings_for_profile(
            profile,
            base=settings,
            codex_command=codex_command,
            endpoint=endpoint,
            model=model,
            api_key_env=api_key_env,
            timeout=timeout,
        )
        store = SettingsStore(settings.paths)
        if store.path.exists() and not force:
            stderr.write(
                f"agentx: settings already exist at {store.path}; pass --force to overwrite.\n"
            )
            return 2
        written = store.write(initialized)
    except (ConfigError, StoreError, OSError) as exc:
        stderr.write(f"agentx: {exc}\n")
        return 2

    return _write(
        {
            "profile": profile,
            "settings_path": str(written),
            "settings": initialized.as_dict(),
        },
        json_output,
        stdout,
        text_formatter=_format_init,
    )


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


def _resolve_plan_provider(*, fake: bool, requested_provider: str, settings) -> str:
    if fake:
        return "fake-local"
    normalized = requested_provider.strip().lower()
    if normalized != "auto":
        return normalized
    public_providers = set(settings.public_providers)
    if "fake-local" in public_providers and "codex" not in public_providers:
        return "fake-local"
    if "codex" not in public_providers and settings.private_provider:
        return settings.private_provider
    return "codex"


def _settings_for_profile(
    profile: str,
    *,
    base,
    codex_command: str,
    endpoint: str | None,
    model: str | None,
    api_key_env: str | None,
    timeout: float,
) -> Settings:
    if profile == "agentx":
        return Settings(
            paths=base.paths,
            public_providers=("fake-local",),
            private_provider=None,
            external_max_classification=base.external_max_classification,
            providers={},
        )
    if profile == "codex":
        return Settings(
            paths=base.paths,
            public_providers=("codex",),
            private_provider=None,
            external_max_classification=base.external_max_classification,
            providers={"codex": ProviderSettings(command=codex_command)},
        )
    if profile in {"private-openai-compatible", "halo"}:
        if not endpoint:
            raise ConfigError("private-openai-compatible profile requires --endpoint.")
        if not model:
            raise ConfigError("private-openai-compatible profile requires --model.")
        providers = dict(base.providers)
        providers["private-openai-compatible"] = ProviderSettings(
            endpoint=endpoint,
            model=model,
            api_key_env=api_key_env,
            timeout=timeout,
        )
        return Settings(
            paths=base.paths,
            # Add Qwen without disabling existing CLI integrations. An empty
            # list intentionally means no provider allow-list.
            public_providers=base.public_providers,
            private_provider="private-openai-compatible",
            external_max_classification=base.external_max_classification,
            providers=providers,
        )
    raise ConfigError("init profile must be 'agentx', 'codex', 'private-openai-compatible', or 'halo'.")


def _plan(
    prompt: str,
    provider: str,
    model_tier: str | None,
    session_id: str,
    source_root: str,
    workspace_id: str | None,
    context_paths: tuple[str, ...],
    settings,
    json_output: bool,
    stdout: TextIO,
    stderr: TextIO,
    *,
    interactive_output: bool = False,
    web_approval: ApprovalCallback | None = None,
    request_cancellation: RequestCancellation | None = None,
    enable_patch_tool: bool = False,
    enable_shell_tool: bool = False,
    run_mode: str = "plan",
) -> int:
    try:
        if provider == "fake-local":
            run_provider = "fake-local"
            provider_statuses = None
            adapter = None
            scoped_source_root = None
        elif provider == "codex":
            run_provider = "codex"
            provider_statuses = ProviderRegistry(settings=settings).list_statuses()
            codex_status = next(
                (status for status in provider_statuses if status.id == "codex"),
                None,
            )
            if codex_status is None or not codex_status.enabled:
                reason = "not registered" if codex_status is None else codex_status.reason
                stderr.write(f"agentx: codex provider is not available: {reason}\n")
                return 2
            # The provider process starts in this directory, so pass an
            # absolute path to both subprocess cwd and Codex -C.  A relative
            # AGENTX_HOME would otherwise make Codex resolve -C against the
            # already-changed cwd and look for a duplicated nested path.
            scoped_workspace = (
                SessionStore(settings.paths).path_for_session(session_id) / "workspace"
            ).resolve()
            codex_settings = settings.providers.get("codex")
            command = codex_settings.command if codex_settings and codex_settings.command else "codex"
            adapter = CodexCliAdapter(
                command=command,
                cwd=scoped_workspace,
                extra_args=("-C", str(scoped_workspace)),
            )
            scoped_source_root = Path(source_root)
        elif provider in {"claude", "kiro"}:
            run_provider = provider
            provider_statuses = ProviderRegistry(settings=settings).list_statuses()
            provider_status = next(
                (status for status in provider_statuses if status.id == provider),
                None,
            )
            if provider_status is None or not provider_status.enabled:
                reason = "not registered" if provider_status is None else provider_status.reason
                stderr.write(f"agentx: provider '{provider}' is not available: {reason}\n")
                return 2
            provider_settings = settings.providers.get(provider)
            command = (
                provider_settings.command
                if provider_settings is not None and provider_settings.command
                else provider_status.command
            )
            if not command:
                stderr.write(f"agentx: provider '{provider}' has no executable command configured.\n")
                return 2
            scoped_workspace = (
                SessionStore(settings.paths).path_for_session(session_id) / "workspace"
            ).resolve()
            extra_args = (
                ("-p", "--permission-mode", "plan", "--no-session-persistence", "--output-format", "text")
                if provider == "claude"
                else ("chat", "--no-interactive", "--trust-tools=fs_read")
            )
            adapter = CliPlanAdapter(
                provider_id=provider,
                command=command,
                extra_args=extra_args,
                cwd=scoped_workspace,
            )
            scoped_source_root = Path(source_root)
        elif provider == "private-openai-compatible":
            run_provider = provider
            provider_statuses = ProviderRegistry(settings=settings).list_statuses()
            provider_status = next(
                (status for status in provider_statuses if status.id == provider),
                None,
            )
            if provider_status is None or not provider_status.enabled:
                reason = "not registered" if provider_status is None else provider_status.reason
                stderr.write(f"agentx: provider '{provider}' is not available: {reason}\n")
                return 2
            provider_settings = settings.providers.get(provider)
            endpoint = (
                provider_settings.endpoint
                if provider_settings is not None
                else None
            )
            if provider_settings is None or not endpoint:
                stderr.write(
                    f"agentx: provider '{provider}' requires an endpoint in the external settings file.\n"
                )
                return 2
            if not provider_settings.model:
                stderr.write(f"agentx: provider '{provider}' requires a model setting.\n")
                return 2
            api_key = (
                os.environ.get(provider_settings.api_key_env)
                if provider_settings.api_key_env
                else None
            )
            scoped_source_root = Path(source_root)
            subagent_manager = SubagentManager(
                parent_session_id=session_id,
                runner=_PrivateSubagentRunner(
                    base_url=endpoint,
                    model=provider_settings.model,
                    api_key=api_key,
                    timeout=provider_settings.timeout,
                    provider_id=provider,
                    source_root=scoped_source_root,
                    session_store=SessionStore(settings.paths),
                    web_approval=web_approval,
                    request_cancellation=request_cancellation,
                ),
            )
            tool_mode = "execute" if enable_patch_tool or enable_shell_tool else "plan"
            tool_executor = build_private_tool_executor(
                mode=tool_mode,
                workspace_root=scoped_source_root,
                context_paths=context_paths,
                paths=settings.paths,
                user_prompt=prompt,
                approval_callback=web_approval,
                artifacts_dir=SessionStore(settings.paths).path_for_session(session_id) / "artifacts",
                extra_executors=(SubagentTools(subagent_manager),),
            )
            adapter = OpenAICompatibleAdapter(
                base_url=endpoint,
                model=provider_settings.model,
                api_key=api_key,
                timeout=provider_settings.timeout,
                provider_id=provider,
                context_root=(
                    SessionStore(settings.paths).path_for_session(session_id) / "workspace"
                ).resolve(),
                stream=not json_output,
                stream_callback=(
                    _CliStreamRenderer(
                        stdout,
                        color=_supports_color(stdout),
                    )
                    if not json_output
                    else None
                ),
                tool_executor=tool_executor,
                request_cancellation=request_cancellation,
            )

        run = AgentRun(
            prompt=prompt,
            mode=run_mode,
            provider=run_provider,
            model_tier=model_tier,
            context_paths=context_paths,
        )
        result = execute_plan_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id=session_id,
            run=run,
            provider_statuses=provider_statuses,
            source_root=scoped_source_root,
            workspace_id=workspace_id,
            adapter=adapter,
        )
    except (
        AdapterError,
        ToolError,
        OrchestratorError,
        RouteValidationError,
        StoreError,
        OSError,
    ) as exc:
        stderr.write(f"agentx: {exc}\n")
        return 2

    code = 0 if result.stored_run.result.status == "success" else 2
    if code == 0:
        _record_plan_memory_events(settings, session_id, prompt, provider, result)
    _write(
        result.as_dict(),
        json_output,
        stdout,
        text_formatter=lambda payload: _format_plan(
            payload,
            color=_supports_color(stdout),
            show_metadata=not interactive_output,
        ),
    )
    return code


def _record_plan_memory_events(settings, session_id: str, prompt: str, provider: str, result) -> None:
    outcome = result.stored_run.result.outcome
    if not isinstance(outcome, Mapping):
        return
    summary = outcome.get("summary") or outcome.get("response") or ""
    tool_names = outcome.get("tools_used") or ()
    if isinstance(tool_names, (str, bytes)) or not isinstance(tool_names, Sequence):
        tool_names = ()
    try:
        append_interaction_events(
            settings.paths,
            session_id=session_id,
            user_prompt=prompt,
            assistant_summary=str(summary),
            tool_names=tuple(str(name) for name in tool_names),
            provider_id=provider,
        )
    except AgentXMemoryError:
        return


class _PrivateSubagentRunner:
    """Run child tasks through isolated private-provider adapter instances."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout: float,
        provider_id: str,
        source_root: Path,
        session_store: SessionStore,
        web_approval: ApprovalCallback | None = None,
        request_cancellation: RequestCancellation | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.provider_id = provider_id
        self.source_root = source_root
        self.session_store = session_store
        self.web_approval = web_approval
        self.request_cancellation = request_cancellation

    def run(
        self,
        task: SubagentTask,
        *,
        session_id: str,
        depth: int,
    ) -> dict[str, object]:
        if depth != 1:
            raise ToolError("private subagent runner only accepts depth-one children")
        provider = self.provider_id if task.provider == "auto" else task.provider
        if provider != self.provider_id:
            raise ToolError(
                f"provider '{provider}' is not available to this private subagent runner"
            )
        child_run = AgentRun(
            prompt=task.prompt,
            mode=task.mode,
            provider=provider,
            model_tier=task.model_tier,
            context_paths=task.context_paths,
            task_hints=task.task_hints,
        )
        child_adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            timeout=self.timeout,
            provider_id=self.provider_id,
            context_root=self.source_root,
            stream=False,
            tool_executor=build_private_tool_executor(
                mode="plan",
                workspace_root=self.source_root,
                context_paths=task.context_paths,
                paths=self.session_store.paths,
                user_prompt=task.prompt,
                approval_callback=self.web_approval,
                artifacts_dir=self.session_store.path_for_session(session_id) / "artifacts",
            ),
            request_cancellation=self.request_cancellation,
        )
        stored = execute_adapter_run(
            session_store=self.session_store,
            session_id=session_id,
            run=child_run,
            adapter=child_adapter,
            context_map={
                "schema_version": 1,
                "requested_paths": list(task.context_paths),
                "included_paths": list(task.context_paths),
                "excluded_paths": [],
            },
        )
        outcome = dict(stored.result.outcome)
        return {
            "status": stored.result.status,
            "summary": str(outcome.get("summary", "")),
            "response": outcome.get("response"),
            "provider_id": stored.result.provider_id,
            "model_id": stored.result.model_id,
            "artifact_root": str(stored.root),
            "result": stored.result.as_dict(),
        }


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


def _memory_command(args, settings, json_output: bool, stdout: TextIO, stderr: TextIO) -> int:
    if args.memory_command is None:
        stderr.write("agentx: memory requires a subcommand.\n")
        return 2
    try:
        if args.memory_command == "remember":
            payload = call_memory_tool(
                settings.paths,
                "memory_remember",
                {
                    "content": " ".join(args.content),
                    "privacy_class": args.privacy_class,
                },
            )
            return _write(payload, json_output, stdout, text_formatter=_format_memory_record)
        if args.memory_command == "search":
            request = {"query": args.query}
            if args.privacy_class:
                request["privacy_class"] = args.privacy_class
            payload = call_memory_tool(settings.paths, "memory_search", request)
            return _write(payload, json_output, stdout, text_formatter=_format_memory_search)
        if args.memory_command == "show":
            payload = call_memory_tool(settings.paths, "memory_show", {"memory_id": args.memory_id})
            return _write(payload, json_output, stdout, text_formatter=_format_memory_record)
        if args.memory_command == "correct":
            payload = call_memory_tool(
                settings.paths,
                "memory_correct",
                {"memory_id": args.memory_id, "replacement": " ".join(args.replacement)},
            )
            return _write(payload, json_output, stdout, text_formatter=_format_memory_record)
        if args.memory_command == "forget":
            if args.all:
                payload = call_memory_tool(settings.paths, "memory_forget", {"all": True, "hard": not args.soft})
            elif args.memory_id:
                payload = call_memory_tool(
                    settings.paths,
                    "memory_forget",
                    {"memory_id": args.memory_id, "hard": not args.soft},
                )
            else:
                stderr.write("agentx: memory forget requires <memory-id> or --all.\n")
                return 2
            return _write(payload, json_output, stdout, text_formatter=_format_memory_delete)
        if args.memory_command == "proposals":
            payload = {"proposals": list_memory_proposals(settings.paths, status=args.status)}
            return _write(payload, json_output, stdout, text_formatter=_format_memory_proposals)
        if args.memory_command == "apply":
            payload = apply_memory_proposal(settings.paths, args.proposal_id)
            return _write(payload, json_output, stdout, text_formatter=_format_memory_record)
    except AgentXMemoryError as exc:
        stderr.write(f"agentx: {exc}\n")
        return 2
    stderr.write(f"agentx: unknown memory subcommand '{args.memory_command}'.\n")
    return 2


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


def _format_init(payload: dict[str, object]) -> str:
    return (
        f"initialized AgentX profile '{payload['profile']}' at {payload['settings_path']}\n"
    )


def _format_fake_run(payload: dict[str, object]) -> str:
    return f"wrote fake run artifacts to {payload['root']}\n"


def _format_memory_record(payload: Mapping[str, object]) -> str:
    memory_id = payload.get("memory_id", payload.get("id", "<unknown>"))
    privacy = payload.get("privacy_class", payload.get("classification", "unknown"))
    kind = payload.get("memory_kind", "memory")
    text = payload.get("summary") or payload.get("content") or ""
    return f"{memory_id}\t{privacy}\t{kind}\t{text}\n"


def _format_memory_search(payload: Mapping[str, object]) -> str:
    memories = payload.get("memories", ())
    if not memories:
        return "No memory records found.\n"
    return "".join(_format_memory_record(memory) for memory in memories if isinstance(memory, Mapping))


def _format_memory_delete(payload: Mapping[str, object]) -> str:
    return f"deleted: {payload.get('deleted', False)}\n"


def _format_memory_proposals(payload: Mapping[str, object]) -> str:
    proposals = payload.get("proposals", ())
    if not proposals:
        return "No memory proposals found.\n"
    lines = []
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            continue
        lines.append(
            f"{proposal.get('proposal_id')}\t{proposal.get('status')}\t"
            f"{proposal.get('operation')}\t{proposal.get('reason')}"
        )
    return "\n".join(lines) + "\n"


def _format_plan(
    payload: dict[str, object],
    *,
    color: bool = False,
    show_metadata: bool = True,
) -> str:
    route = payload["route"]
    result = payload.get("result", {})
    lines = []
    if show_metadata:
        lines.append(
            f"wrote plan artifacts to {payload['root']}\n"
            f"{route['explanation']}"
        )
    if isinstance(result, dict):
        output_event = next(
            (
                event
                for event in result.get("transcript_events", ())
                if isinstance(event, dict) and event.get("event") == "process_output_captured"
            ),
            None,
        )
        if isinstance(output_event, dict):
            stdout_text = str(output_event.get("stdout") or "").strip()
            stderr_text = str(output_event.get("stderr") or "").strip()
            if stdout_text:
                lines.extend(("", stdout_text))
            if stderr_text and result.get("status") != "success":
                lines.extend(("", "provider error:", stderr_text))
        outcome = result.get("outcome", {})
        if (
            result.get("provider_id") == "private-openai-compatible"
            and result.get("status") == "success"
            and isinstance(outcome, dict)
            and not outcome.get("streamed", False)
        ):
            thinking = outcome.get("thinking")
            response = outcome.get("response") or outcome.get("summary")
            if isinstance(thinking, str) and thinking.strip():
                lines.extend(("", _render_thinking(thinking.strip(), color=color)))
            if isinstance(response, str) and response.strip():
                lines.extend(("", "Assistant:", response.strip()))
        if result.get("status") != "success":
            summary = outcome.get("summary") if isinstance(outcome, dict) else None
            if summary:
                lines.extend(("", f"provider status: {summary}"))
    return "\n".join(lines) + "\n"


def _render_thinking(thinking: str, *, color: bool) -> str:
    rendered = f"Thinking:\n{thinking}"
    if not color:
        return rendered
    return f"\x1b[90m{rendered}\x1b[0m"


def _supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _supports_terminal_key_input(stream: TextIO) -> bool:
    if select is None or termios is None or tty is None:
        return False
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty) or not isatty():
        return False
    try:
        stream.fileno()
    except (AttributeError, OSError):
        return False
    return True


def _read_terminal_line_or_escape(stdin: TextIO, stdout: TextIO) -> str:
    """Read a short terminal answer, letting Escape take effect immediately."""

    if not _supports_terminal_key_input(stdin):
        return stdin.readline()
    try:
        fd = stdin.fileno()
        attributes = termios.tcgetattr(fd)
    except (AttributeError, OSError, termios.error):
        return stdin.readline()

    try:
        tty.setcbreak(fd)
        no_echo_attributes = termios.tcgetattr(fd)
        no_echo_attributes[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, no_echo_attributes)
        characters: list[str] = []
        while True:
            key = os.read(fd, 1)
            if key == b"\x1b":
                stdout.write("\n")
                stdout.flush()
                return "\x1b"
            if not key:
                return ""
            if key in {b"\r", b"\n"}:
                stdout.write("\n")
                stdout.flush()
                return "".join(characters)
            if key in {b"\x08", b"\x7f"}:
                if characters:
                    characters.pop()
                    stdout.write("\b \b")
                    stdout.flush()
                continue
            character = key.decode("utf-8", errors="ignore")
            if character:
                characters.append(character)
                stdout.write(character)
                stdout.flush()
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, attributes)
        except (OSError, termios.error, UnboundLocalError):
            pass


class _InteractiveRequestCancellation:
    """Watch a POSIX terminal for Escape while a provider request is active."""

    def __init__(self, stdin: TextIO, stdout: TextIO) -> None:
        self.stdin = stdin
        self.stdout = stdout

    @contextlib.contextmanager
    def request(self, cancel_request: Callable[[], None]) -> Iterator[threading.Event]:
        cancelled = threading.Event()
        if not self._supports_escape_monitoring():
            yield cancelled
            return

        completed = threading.Event()
        ready = threading.Event()
        watcher = threading.Thread(
            target=self._watch_for_escape,
            args=(cancelled, completed, ready, cancel_request),
            daemon=True,
            name="agentx-escape-cancel",
        )
        watcher.start()
        ready.wait(timeout=0.25)
        if ready.is_set():
            self.stdout.write("\nPress Esc to cancel the active request.\n")
            self.stdout.flush()
        try:
            yield cancelled
        finally:
            completed.set()
            watcher.join(timeout=0.25)

    def _supports_escape_monitoring(self) -> bool:
        return _supports_terminal_key_input(self.stdin)

    def _watch_for_escape(
        self,
        cancelled: threading.Event,
        completed: threading.Event,
        ready: threading.Event,
        cancel_request: Callable[[], None],
    ) -> None:
        try:
            fd = self.stdin.fileno()
            attributes = termios.tcgetattr(fd)
        except (AttributeError, OSError, termios.error):
            ready.set()
            return

        try:
            tty.setcbreak(fd)
            ready.set()
            while not completed.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    continue
                key = os.read(fd, 1)
                if key != b"\x1b":
                    continue
                cancelled.set()
                cancel_request()
                self.stdout.write("\r\nRequest cancelled. Returning to AgentX.\n")
                self.stdout.flush()
                return
        except (OSError, termios.error):
            ready.set()
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, attributes)
            except (OSError, termios.error, UnboundLocalError):
                pass


class _InteractiveApproval:
    """Ask before a model-requested operation with side effects or external access."""

    def __init__(self, stdin: TextIO, stdout: TextIO) -> None:
        self.stdin = stdin
        self.stdout = stdout

    def __call__(self, operation: str, details: Mapping[str, object]) -> bool:
        if operation == "web.search":
            self.stdout.write("\nInternet search requested by the model.\n")
            self.stdout.write(f"Search provider: {details.get('source', 'web search')}\n")
            reason = details.get("reason")
            if isinstance(reason, str) and reason:
                self.stdout.write(f"Reason: {reason}\n")
            self.stdout.write(f"Query: {details.get('query', '')}\n")
            self.stdout.write(f"Maximum results: {details.get('max_results', '')}\n")
        elif operation in {"web.fetch", "web.fetch_document"}:
            self.stdout.write("\nInternet fetch requested by the model.\n")
            self.stdout.write(f"Fetch URL: {details.get('url', '')}\n")
            self.stdout.write(f"Maximum returned text: {details.get('max_chars', '')} characters\n")
            if operation == "web.fetch_document":
                self.stdout.write(f"Maximum pages: {details.get('max_pages', '')}\n")
        elif operation == "workspace.patch":
            self.stdout.write("\nWorkspace patch requested by the model.\n")
            paths = details.get("paths", ())
            if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
                self.stdout.write("Paths:\n")
                for path in paths:
                    self.stdout.write(f"  {path}\n")
            patch = details.get("patch")
            if isinstance(patch, str):
                preview = patch[:2_000]
                self.stdout.write("Patch preview:\n")
                self.stdout.write(preview)
                if len(patch) > len(preview):
                    self.stdout.write("\n... truncated ...")
                if not preview.endswith("\n"):
                    self.stdout.write("\n")
        elif operation == "shell.exec":
            self.stdout.write("\nShell command requested by the model.\n")
            argv = details.get("argv", ())
            if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
                self.stdout.write("Argv:\n")
                for item in argv:
                    self.stdout.write(f"  {item}\n")
            self.stdout.write(f"Working directory: {details.get('cwd', '.')}\n")
            self.stdout.write(f"Timeout: {details.get('timeout_seconds', '')} seconds\n")
        elif operation.startswith("browser."):
            self.stdout.write("\nBrowser action requested by the model.\n")
            self.stdout.write(f"Operation: {operation}\n")
            for key, value in details.items():
                self.stdout.write(f"{key}: {value}\n")
        elif operation.startswith("memory."):
            self.stdout.write("\nMemory operation requested by the model.\n")
            self.stdout.write(f"Operation: {operation}\n")
            for key, value in details.items():
                self.stdout.write(f"{key}: {value}\n")
        else:
            self.stdout.write("\nAgentX tool operation requested by the model.\n")
            self.stdout.write(f"Operation: {operation}\n")
        self.stdout.write("Allow this request? [y/N]: ")
        self.stdout.flush()
        answer = _read_terminal_line_or_escape(self.stdin, self.stdout).strip().lower()
        if answer == "\x1b":
            self.stdout.write("Cancelled.\n")
            return False
        allowed = answer in {"y", "yes"}
        self.stdout.write("Approved.\n" if allowed else "Denied.\n")
        return allowed


class _CliStreamRenderer:
    def __init__(self, stdout: TextIO, *, color: bool) -> None:
        self.stdout = stdout
        self.color = color
        self.section: str | None = None

    def __call__(self, kind: str, value: str) -> None:
        if kind == "thinking":
            if self.section != "thinking":
                if self.section is not None:
                    self.stdout.write("\n")
                prefix = "\x1b[90m" if self.color else ""
                self.stdout.write(f"{prefix}Thinking:\n")
                self.section = "thinking"
            self.stdout.write(value)
        elif kind == "content":
            if self.section != "content":
                if self.section == "thinking":
                    self.stdout.write("\x1b[0m" if self.color else "")
                    self.stdout.write("\n")
                self.stdout.write("Assistant:\n")
                self.section = "content"
            self.stdout.write(value)
        elif kind in {"complete", "error"}:
            if self.section is not None:
                if self.section == "thinking" and self.color:
                    self.stdout.write("\x1b[0m")
                self.stdout.write("\n")
                self.stdout.flush()
                self.section = None
        self.stdout.flush()


def _format_execute(payload: dict[str, object]) -> str:
    validation = payload["patch_validation"]
    status = "accepted" if validation["accepted"] else "rejected"
    return (
        f"wrote execute artifacts to {payload['root']}\n"
        f"patch validation: {status}; patch applied: false\n"
    )
