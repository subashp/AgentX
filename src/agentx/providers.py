from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable

from .config import Settings


ExecutableSearch = Callable[[str], str | None]
ProviderCheck = Callable[[str], bool]


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    display_name: str
    kind: str
    command: str | None = None
    public: bool = True


@dataclass(frozen=True)
class ProviderStatus:
    id: str
    display_name: str
    kind: str
    enabled: bool
    reason: str
    command: str | None = None
    resolved_command: str | None = None
    endpoint: str | None = None
    checks: dict[str, bool] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "kind": self.kind,
            "enabled": self.enabled,
            "reason": self.reason,
            "command": self.command,
            "resolved_command": self.resolved_command,
            "endpoint": self.endpoint,
            "checks": dict(self.checks or {}),
        }


DEFAULT_PROVIDERS: tuple[ProviderDefinition, ...] = (
    ProviderDefinition("codex", "Codex CLI", "cli", "codex"),
    ProviderDefinition("claude", "Claude Code", "cli", "claude"),
    ProviderDefinition("kiro", "Kiro CLI", "cli", "kiro-cli"),
    ProviderDefinition(
        "private-openai-compatible",
        "Private OpenAI-Compatible Endpoint",
        "openai_compatible",
        None,
        public=False,
    ),
    ProviderDefinition("fake-local", "AgentX Fake Local", "builtin", None, public=False),
)


class ProviderRegistry:
    def __init__(
        self,
        providers: Iterable[ProviderDefinition] = DEFAULT_PROVIDERS,
        executable_search: ExecutableSearch | None = None,
        settings: Settings | None = None,
        auth_check: ProviderCheck | None = None,
        subscription_check: ProviderCheck | None = None,
        endpoint_check: ProviderCheck | None = None,
    ) -> None:
        self._providers = tuple(providers)
        self._executable_search = executable_search or shutil.which
        self._settings = settings
        self._auth_check = auth_check or (lambda check_id: True)
        self._subscription_check = subscription_check or (lambda check_id: True)
        self._endpoint_check = endpoint_check or _default_endpoint_check

    def list_statuses(self) -> tuple[ProviderStatus, ...]:
        return tuple(self._status_for(provider) for provider in self._providers)

    def _status_for(self, provider: ProviderDefinition) -> ProviderStatus:
        provider_settings = (self._settings.providers.get(provider.id) if self._settings else None)
        checks: dict[str, bool] = {}

        if provider_settings and not provider_settings.enabled:
            return ProviderStatus(
                id=provider.id,
                display_name=provider.display_name,
                kind=provider.kind,
                enabled=False,
                reason="disabled_by_settings",
                command=provider_settings.command or provider.command,
                endpoint=provider_settings.endpoint,
                checks=checks,
            )

        command = provider_settings.command if provider_settings and provider_settings.command else provider.command
        endpoint = _configured_endpoint(provider_settings)

        if provider.kind == "builtin":
            return self._checked_status(provider, command, endpoint, None, checks)

        if provider.kind == "openai_compatible":
            if endpoint:
                if provider_settings is not None and not provider_settings.model:
                    checks["model"] = False
                    return ProviderStatus(
                        id=provider.id,
                        display_name=provider.display_name,
                        kind=provider.kind,
                        enabled=False,
                        reason="model_not_configured",
                        command=command,
                        endpoint=endpoint,
                        checks=checks,
                    )
                checks["endpoint"] = self._endpoint_check(endpoint)
                if not checks["endpoint"]:
                    return ProviderStatus(
                        id=provider.id,
                        display_name=provider.display_name,
                        kind=provider.kind,
                        enabled=False,
                        reason="disabled_unhealthy",
                        command=command,
                        endpoint=endpoint,
                        checks=checks,
                    )
                return self._checked_status(provider, command, endpoint, None, checks)

            return ProviderStatus(
                id=provider.id,
                display_name=provider.display_name,
                kind=provider.kind,
                enabled=False,
                reason="endpoint_not_configured",
                command=command,
                endpoint=endpoint,
                checks=checks,
            )

        if not command:
            return ProviderStatus(
                id=provider.id,
                display_name=provider.display_name,
                kind=provider.kind,
                enabled=False,
                reason="missing_command",
                endpoint=endpoint,
                checks=checks,
            )

        resolved = self._executable_search(command)
        if not resolved:
            return ProviderStatus(
                id=provider.id,
                display_name=provider.display_name,
                kind=provider.kind,
                enabled=False,
                reason="disabled_missing_binary",
                command=command,
                endpoint=endpoint,
                checks=checks,
            )

        return self._checked_status(provider, command, endpoint, resolved, checks)

    def _checked_status(
        self,
        provider: ProviderDefinition,
        command: str | None,
        endpoint: str | None,
        resolved: str | None,
        checks: dict[str, bool],
    ) -> ProviderStatus:
        provider_settings = (self._settings.providers.get(provider.id) if self._settings else None)
        if provider_settings and provider_settings.auth_check:
            checks["auth"] = self._auth_check(provider_settings.auth_check)
            if not checks["auth"]:
                return ProviderStatus(
                    id=provider.id,
                    display_name=provider.display_name,
                    kind=provider.kind,
                    enabled=False,
                    reason="disabled_missing_auth",
                    command=command,
                    resolved_command=resolved,
                    endpoint=endpoint,
                    checks=checks,
                )

        if provider_settings and provider_settings.subscription_check:
            checks["subscription"] = self._subscription_check(provider_settings.subscription_check)
            if not checks["subscription"]:
                return ProviderStatus(
                    id=provider.id,
                    display_name=provider.display_name,
                    kind=provider.kind,
                    enabled=False,
                    reason="disabled_missing_subscription",
                    command=command,
                    resolved_command=resolved,
                    endpoint=endpoint,
                    checks=checks,
                )

        return ProviderStatus(
            id=provider.id,
            display_name=provider.display_name,
            kind=provider.kind,
            enabled=True,
            reason="available",
            command=command,
            resolved_command=resolved,
            endpoint=endpoint,
            checks=checks,
        )


def _configured_endpoint(provider_settings) -> str | None:
    if provider_settings is None:
        return None
    return provider_settings.endpoint


def _default_endpoint_check(endpoint: str) -> bool:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path + "/models", "", ""))
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "agentx-provider-check/0.1",
            "ngrok-skip-browser-warning": "true",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError, urllib.error.URLError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("data"), list)
