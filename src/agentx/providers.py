from __future__ import annotations

import shutil
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
        self._endpoint_check = endpoint_check or (lambda endpoint: True)

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
        endpoint = provider_settings.endpoint if provider_settings else None

        if provider.kind == "openai_compatible":
            if endpoint:
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
