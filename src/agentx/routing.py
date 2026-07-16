from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .providers import ProviderStatus


@dataclass(frozen=True)
class AgentRun:
    prompt: str
    mode: str = "plan"
    provider: str = "auto"
    model_tier: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "mode": self.mode,
            "provider": self.provider,
            "model_tier": self.model_tier,
        }


@dataclass(frozen=True)
class RouteDecision:
    run: AgentRun
    selected_provider: str | None
    selected_model_tier: str
    eligible_providers: tuple[str, ...]
    rejected_providers: dict[str, str]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "run": self.run.as_dict(),
            "selected_provider": self.selected_provider,
            "selected_model_tier": self.selected_model_tier,
            "eligible_providers": list(self.eligible_providers),
            "rejected_providers": dict(self.rejected_providers),
            "reason": self.reason,
        }


DEFAULT_MODEL_TIERS: dict[str, str] = {
    "plan": "high",
    "execute": "standard",
    "review": "standard",
    "explain": "economy",
    "tests": "economy",
    "docs": "economy",
}


class Router:
    def __init__(self, settings: Settings, providers: tuple[ProviderStatus, ...]) -> None:
        self._settings = settings
        self._providers = providers

    def explain(self, run: AgentRun) -> RouteDecision:
        selected_tier = run.model_tier or DEFAULT_MODEL_TIERS.get(run.mode, "standard")
        eligible: list[str] = []
        rejected: dict[str, str] = {}

        for provider in self._providers:
            if not provider.enabled:
                rejected[provider.id] = provider.reason
                continue
            if run.provider != "auto" and provider.id != run.provider:
                rejected[provider.id] = "not_requested_provider"
                continue
            if self._settings.public_providers and provider.id not in self._settings.public_providers:
                rejected[provider.id] = "not_in_public_provider_defaults"
                continue
            eligible.append(provider.id)

        selected = eligible[0] if eligible else None
        reason = "selected_first_eligible_provider" if selected else "no_eligible_provider"
        return RouteDecision(
            run=run,
            selected_provider=selected,
            selected_model_tier=selected_tier,
            eligible_providers=tuple(eligible),
            rejected_providers=rejected,
            reason=reason,
        )
