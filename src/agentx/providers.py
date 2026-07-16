from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Callable, Iterable


ExecutableSearch = Callable[[str], str | None]


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

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "kind": self.kind,
            "enabled": self.enabled,
            "reason": self.reason,
            "command": self.command,
            "resolved_command": self.resolved_command,
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
    ) -> None:
        self._providers = tuple(providers)
        self._executable_search = executable_search or shutil.which

    def list_statuses(self) -> tuple[ProviderStatus, ...]:
        return tuple(self._status_for(provider) for provider in self._providers)

    def _status_for(self, provider: ProviderDefinition) -> ProviderStatus:
        if provider.kind == "openai_compatible":
            return ProviderStatus(
                id=provider.id,
                display_name=provider.display_name,
                kind=provider.kind,
                enabled=False,
                reason="endpoint_not_configured",
                command=provider.command,
            )

        if not provider.command:
            return ProviderStatus(
                id=provider.id,
                display_name=provider.display_name,
                kind=provider.kind,
                enabled=False,
                reason="missing_command",
            )

        resolved = self._executable_search(provider.command)
        return ProviderStatus(
            id=provider.id,
            display_name=provider.display_name,
            kind=provider.kind,
            enabled=resolved is not None,
            reason="available" if resolved else "disabled_missing_binary",
            command=provider.command,
            resolved_command=resolved,
        )
