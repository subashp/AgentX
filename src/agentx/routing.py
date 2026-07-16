from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .config import Settings
from .providers import ProviderStatus


VALID_MODES: frozenset[str] = frozenset(
    {"plan", "execute", "review", "explain", "tests", "docs"}
)
VALID_MODEL_TIERS: frozenset[str] = frozenset(
    {"auto", "high", "standard", "economy"}
)


class RouteValidationError(ValueError):
    """Raised when an AgentRun cannot be normalized into a valid route contract."""


@dataclass(frozen=True)
class RunBudget:
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_input_tokens",
            _normalize_optional_positive_int(self.max_input_tokens, "budget.max_input_tokens"),
        )
        object.__setattr__(
            self,
            "max_output_tokens",
            _normalize_optional_positive_int(self.max_output_tokens, "budget.max_output_tokens"),
        )
        object.__setattr__(
            self,
            "max_cost_usd",
            _normalize_optional_positive_float(self.max_cost_usd, "budget.max_cost_usd"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_cost_usd": self.max_cost_usd,
        }

    @classmethod
    def from_value(cls, value: object) -> RunBudget:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise RouteValidationError("budget must be a mapping or RunBudget.")

        supported_fields = {"max_input_tokens", "max_output_tokens", "max_cost_usd"}
        unknown_fields = sorted(set(value) - supported_fields)
        if unknown_fields:
            raise RouteValidationError(
                "budget contains unsupported fields: " + ", ".join(unknown_fields)
            )

        return cls(
            max_input_tokens=value.get("max_input_tokens"),
            max_output_tokens=value.get("max_output_tokens"),
            max_cost_usd=value.get("max_cost_usd"),
        )


@dataclass(frozen=True)
class AgentRun:
    prompt: str
    mode: str = "plan"
    provider: str = "auto"
    model_tier: str | None = None
    budget: RunBudget = field(default_factory=RunBudget)
    required_tools: tuple[str, ...] = ()
    required_mcp_servers: tuple[str, ...] = ()
    required_mcp_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _normalize_prompt(self.prompt))
        object.__setattr__(self, "mode", _normalize_mode(self.mode))
        object.__setattr__(self, "provider", _normalize_provider(self.provider))
        object.__setattr__(self, "model_tier", _normalize_model_tier(self.model_tier))
        object.__setattr__(self, "budget", RunBudget.from_value(self.budget))
        object.__setattr__(
            self,
            "required_tools",
            _normalize_string_collection(self.required_tools, "required_tools"),
        )
        object.__setattr__(
            self,
            "required_mcp_servers",
            _normalize_string_collection(
                self.required_mcp_servers,
                "required_mcp_servers",
            ),
        )
        object.__setattr__(
            self,
            "required_mcp_tools",
            _normalize_string_collection(
                self.required_mcp_tools,
                "required_mcp_tools",
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "mode": self.mode,
            "provider": self.provider,
            "model_tier": self.model_tier,
            "budget": self.budget.as_dict(),
            "required_tools": list(self.required_tools),
            "required_mcp_servers": list(self.required_mcp_servers),
            "required_mcp_tools": list(self.required_mcp_tools),
        }


@dataclass(frozen=True)
class ProviderRouteReport:
    id: str
    display_name: str
    enabled: bool
    eligible: bool
    reason: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "enabled": self.enabled,
            "eligible": self.eligible,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RouteDecision:
    run: AgentRun
    selected_provider: str | None
    selected_model_tier: str
    mode_default_model_tier: str
    eligible_providers: tuple[str, ...]
    rejected_providers: dict[str, str]
    provider_reports: tuple[ProviderRouteReport, ...]
    reason: str
    explanation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "run": self.run.as_dict(),
            "selected_provider": self.selected_provider,
            "selected_model_tier": self.selected_model_tier,
            "mode_default_model_tier": self.mode_default_model_tier,
            "eligible_providers": list(self.eligible_providers),
            "rejected_providers": dict(self.rejected_providers),
            "provider_reports": [report.as_dict() for report in self.provider_reports],
            "reason": self.reason,
            "explanation": self.explanation,
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
        default_tier = DEFAULT_MODEL_TIERS.get(run.mode, "standard")
        selected_tier = default_tier if run.model_tier in {None, "auto"} else run.model_tier
        eligible: list[str] = []
        rejected: dict[str, str] = {}
        provider_reports: list[ProviderRouteReport] = []
        requested_provider_registered = (
            run.provider == "auto" or any(provider.id == run.provider for provider in self._providers)
        )

        for provider in self._providers:
            eligible_provider = False
            if not provider.enabled:
                reason = provider.reason
                detail = f"Provider is disabled: {provider.reason}."
            elif run.provider != "auto" and provider.id != run.provider:
                reason = "not_requested_provider"
                detail = f"Provider was not requested; requested provider is '{run.provider}'."
            elif self._settings.public_providers and provider.id not in self._settings.public_providers:
                reason = "not_in_public_provider_defaults"
                detail = "Provider is excluded by settings.public_providers."
            else:
                reason = "eligible"
                detail = "Provider passed current dry-run availability and settings filters."
                eligible_provider = True
                eligible.append(provider.id)

            if not eligible_provider:
                rejected[provider.id] = reason

            provider_reports.append(
                ProviderRouteReport(
                    id=provider.id,
                    display_name=provider.display_name,
                    enabled=provider.enabled,
                    eligible=eligible_provider,
                    reason=reason,
                    detail=detail,
                )
            )

        selected = eligible[0] if eligible else None
        if selected:
            reason = "selected_first_eligible_provider"
        elif run.provider != "auto" and not requested_provider_registered:
            reason = "requested_provider_not_registered"
        elif run.provider != "auto":
            reason = "requested_provider_not_eligible"
        else:
            reason = "no_eligible_provider"

        return RouteDecision(
            run=run,
            selected_provider=selected,
            selected_model_tier=selected_tier,
            mode_default_model_tier=default_tier,
            eligible_providers=tuple(eligible),
            rejected_providers=rejected,
            provider_reports=tuple(provider_reports),
            reason=reason,
            explanation=_build_explanation(
                selected_provider=selected,
                selected_model_tier=selected_tier,
                eligible_providers=tuple(eligible),
                rejected_providers=rejected,
                provider_reports=tuple(provider_reports),
            ),
        )


def _normalize_prompt(value: object) -> str:
    if not isinstance(value, str):
        raise RouteValidationError("prompt must be a string.")
    prompt = value.strip()
    if not prompt:
        raise RouteValidationError("prompt is required.")
    return prompt


def _normalize_mode(value: object) -> str:
    mode = _normalize_choice(value, "mode", VALID_MODES)
    if mode not in DEFAULT_MODEL_TIERS:
        raise RouteValidationError(f"mode '{mode}' does not have a default model tier.")
    return mode


def _normalize_provider(value: object) -> str:
    if not isinstance(value, str):
        raise RouteValidationError("provider must be a string.")
    provider = value.strip().lower()
    if not provider:
        raise RouteValidationError("provider must be 'auto' or a provider id.")
    return provider


def _normalize_model_tier(value: object) -> str | None:
    if value is None:
        return None
    return _normalize_choice(value, "model_tier", VALID_MODEL_TIERS)


def _normalize_choice(value: object, field_name: str, valid_values: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise RouteValidationError(f"{field_name} must be a string.")
    normalized = value.strip().lower()
    if not normalized:
        raise RouteValidationError(
            f"{field_name} must be one of: {', '.join(sorted(valid_values))}."
        )
    if normalized not in valid_values:
        raise RouteValidationError(
            f"invalid {field_name} '{normalized}'. Expected one of: "
            + ", ".join(sorted(valid_values))
            + "."
        )
    return normalized


def _normalize_string_collection(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise RouteValidationError(f"{field_name} must be a string or a sequence of strings.")

    normalized_items: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise RouteValidationError(f"{field_name} entries must be strings.")
        normalized = item.strip()
        if not normalized:
            raise RouteValidationError(f"{field_name} entries must be non-empty strings.")
        if normalized not in normalized_items:
            normalized_items.append(normalized)
    return tuple(normalized_items)


def _normalize_optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise RouteValidationError(f"{field_name} must be an integer.")
    if value <= 0:
        raise RouteValidationError(f"{field_name} must be greater than zero.")
    return value


def _normalize_optional_positive_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RouteValidationError(f"{field_name} must be a number.")
    number = float(value)
    if number <= 0:
        raise RouteValidationError(f"{field_name} must be greater than zero.")
    return number


def _build_explanation(
    *,
    selected_provider: str | None,
    selected_model_tier: str,
    eligible_providers: tuple[str, ...],
    rejected_providers: dict[str, str],
    provider_reports: tuple[ProviderRouteReport, ...],
) -> str:
    if selected_provider:
        selected_clause = (
            f"Selected provider '{selected_provider}' with model tier '{selected_model_tier}'."
        )
    else:
        selected_clause = (
            f"No provider was selected; requested model tier resolved to '{selected_model_tier}'."
        )

    if eligible_providers:
        eligibility_clause = "Eligible providers: " + ", ".join(eligible_providers) + "."
    else:
        eligibility_clause = "Eligible providers: none."

    if not rejected_providers:
        rejection_clause = "Rejected providers: none."
    else:
        rejection_parts = []
        for report in provider_reports:
            if report.id in rejected_providers:
                rejection_parts.append(f"{report.id} ({report.reason})")
        rejection_clause = "Rejected providers: " + ", ".join(rejection_parts) + "."

    return " ".join((selected_clause, eligibility_clause, rejection_clause))
