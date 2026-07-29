#!/usr/bin/env python3
"""Small, dependency-free helpers for the local Halo launcher scripts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-14B"
PRIVATE_PROVIDER = "private-openai-compatible"
PUBLIC_PROVIDERS = ["codex", "claude", "kiro"]


def resolve_settings_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    configured = os.environ.get("AGENTX_SETTINGS")
    if configured:
        return Path(configured).expanduser()
    home = os.environ.get("AGENTX_HOME")
    if home:
        return Path(home).expanduser() / "settings.json"
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home).expanduser() / "agentx" / "settings.json"
    return Path.home() / ".agentx" / "settings.json"


def merge_settings(path: Path, endpoint: str, model: str, timeout: float) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError("Halo setup can write JSON settings only; use a .json AgentX settings path.")

    created = not path.exists()
    if created:
        document: dict[str, Any] = {}
    else:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"AgentX settings are not valid JSON: {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"AgentX settings must contain a JSON object: {path}")
        document = loaded

    before = json.dumps(document, sort_keys=True)
    if "public_providers" not in document:
        document["public_providers"] = list(PUBLIC_PROVIDERS)
    document.setdefault("external_max_classification", "internal")

    existing_private = document.get("private_provider")
    private_provider_changed = existing_private in (None, PRIVATE_PROVIDER)
    if private_provider_changed:
        document["private_provider"] = PRIVATE_PROVIDER

    providers = document.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("AgentX settings field 'providers' must be a JSON object.")
    provider = providers.setdefault(PRIVATE_PROVIDER, {})
    if not isinstance(provider, dict):
        raise ValueError(f"AgentX provider '{PRIVATE_PROVIDER}' must be a JSON object.")
    provider.update(
        {
            "endpoint": endpoint,
            "model": model,
            "timeout": timeout,
            "enabled": True,
        }
    )

    after = json.dumps(document, sort_keys=True)
    changed = before != after
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    return {
        "path": str(path),
        "created": created,
        "changed": changed,
        "private_provider": document.get("private_provider"),
        "private_provider_changed": private_provider_changed,
        "public_providers": document.get("public_providers"),
        "endpoint": endpoint,
        "model": model,
        "timeout": timeout,
    }


def models_url(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"endpoint must be an HTTP(S) URL: {endpoint}")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path + "/models", "", ""))


def fetch_models(endpoint: str, timeout: float) -> list[str]:
    request = urllib.request.Request(
        models_url(endpoint),
        headers={
            "Accept": "application/json",
            "User-Agent": "agentx-halo-launcher/0.1",
            "ngrok-skip-browser-warning": "true",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("/v1/models returned an invalid OpenAI-compatible response")
    identifiers = [item.get("id") for item in payload["data"] if isinstance(item, dict)]
    return [identifier for identifier in identifiers if isinstance(identifier, str) and identifier]


def probe(endpoint: str, timeout: float, expected_model: str | None = None) -> dict[str, Any]:
    try:
        model_ids = fetch_models(endpoint, timeout)
    except (OSError, TimeoutError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"healthy": False, "endpoint": endpoint, "error": str(exc)}
    result: dict[str, Any] = {
        "healthy": True,
        "endpoint": endpoint,
        "models": model_ids,
    }
    if expected_model:
        result["model"] = expected_model
        result["model_available"] = expected_model in model_ids
    return result


def command_settings(args: argparse.Namespace) -> int:
    try:
        result = merge_settings(
            resolve_settings_path(args.settings),
            args.endpoint,
            args.model,
            args.timeout,
        )
    except (OSError, ValueError) as exc:
        print(f"halo: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    if not result["private_provider_changed"]:
        print(
            "halo: preserved an existing non-Qwen private_provider; select "
            "private-openai-compatible explicitly if needed.",
            file=sys.stderr,
        )
    return 0


def command_probe(args: argparse.Namespace) -> int:
    result = probe(args.endpoint, args.timeout, args.model)
    if args.format == "json":
        print(json.dumps(result, sort_keys=True))
    elif result["healthy"]:
        models = ", ".join(result["models"]) or "none"
        print(f"healthy; models: {models}")
    else:
        print(f"unhealthy; {result.get('error', 'unknown error')}")
    if not result["healthy"]:
        return 1
    if args.model and not result.get("model_available", False):
        return 2
    return 0


def command_wait(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout
    last_error = "not checked"
    while True:
        result = probe(args.endpoint, min(args.request_timeout, max(args.timeout, 1.0)), args.model)
        if result["healthy"] and (not args.model or result.get("model_available")):
            if args.format == "json":
                print(json.dumps(result, sort_keys=True))
            else:
                print(f"ready; models: {', '.join(result['models']) or 'none'}")
            return 0
        last_error = result.get("error", "configured model is not advertised")
        if time.monotonic() >= deadline:
            print(
                f"halo: timed out waiting for {args.endpoint}: {last_error}",
                file=sys.stderr,
            )
            return 1
        time.sleep(args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    settings = subparsers.add_parser("settings", help="merge local Halo provider settings")
    settings.add_argument("--settings")
    settings.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    settings.add_argument("--model", default=DEFAULT_MODEL)
    settings.add_argument("--timeout", type=float, default=900.0)
    settings.set_defaults(function=command_settings)

    for command_name in ("probe", "wait"):
        command = subparsers.add_parser(command_name, help=f"{command_name} an OpenAI-compatible endpoint")
        command.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
        command.add_argument("--model")
        command.add_argument("--timeout", type=float, default=30.0 if command_name == "probe" else 900.0)
        command.add_argument("--request-timeout", type=float, default=5.0)
        command.add_argument("--interval", type=float, default=2.0)
        command.add_argument("--format", choices=("json", "text"), default="text")
        command.set_defaults(function=command_probe if command_name == "probe" else command_wait)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
