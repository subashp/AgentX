from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from .config import Settings
from .models import (
    DEFAULT_STALE_METADATA_DAYS,
    ModelCatalog,
    ModelSelection,
    classify_task_complexity,
)
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
    task_hints: tuple[str, ...] = ()
    require_structured_output: bool = False

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
        object.__setattr__(
            self,
            "task_hints",
            _normalize_string_collection(self.task_hints, "task_hints"),
        )
        object.__setattr__(
            self,
            "require_structured_output",
            _normalize_bool(self.require_structured_output, "require_structured_output"),
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
            "task_hints": list(self.task_hints),
            "require_structured_output": self.require_structured_output,
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
    selected_model_id: str | None
    selected_model_tier: str
    required_model_tier: str
    mode_default_model_tier: str
    task_complexity_tier: str
    task_complexity_reason: str
    eligible_providers: tuple[str, ...]
    rejected_providers: dict[str, str]
    provider_reports: tuple[ProviderRouteReport, ...]
    model_metadata_warnings: tuple[str, ...]
    reason: str
    explanation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "run": self.run.as_dict(),
            "selected_provider": self.selected_provider,
            "selected_model_id": self.selected_model_id,
            "selected_model_tier": self.selected_model_tier,
            "required_model_tier": self.required_model_tier,
            "mode_default_model_tier": self.mode_default_model_tier,
            "task_complexity_tier": self.task_complexity_tier,
            "task_complexity_reason": self.task_complexity_reason,
            "eligible_providers": list(self.eligible_providers),
            "rejected_providers": dict(self.rejected_providers),
            "provider_reports": [report.as_dict() for report in self.provider_reports],
            "model_metadata_warnings": list(self.model_metadata_warnings),
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
    def __init__(
        self,
        settings: Settings,
        providers: tuple[ProviderStatus, ...],
        model_catalog: ModelCatalog | None = None,
        *,
        today: date | None = None,
        stale_metadata_max_age_days: int = DEFAULT_STALE_METADATA_DAYS,
    ) -> None:
        self._settings = settings
        self._providers = providers
        self._model_catalog = model_catalog
        self._today = today
        self._stale_metadata_max_age_days = stale_metadata_max_age_days

    def explain(self, run: AgentRun) -> RouteDecision:
        default_tier = DEFAULT_MODEL_TIERS.get(run.mode, "standard")
        complexity = classify_task_complexity(run.mode, run.task_hints)
        required_tier = _resolve_required_tier(
            run,
            default_tier=default_tier,
            complexity_tier=complexity.tier,
            use_model_catalog=self._model_catalog is not None,
        )
        eligible: list[str] = []
        rejected: dict[str, str] = {}
        provider_reports: list[ProviderRouteReport] = []
        provider_model_selections: list[tuple[int, str, ModelSelection]] = []
        requested_provider_registered = (
            run.provider == "auto" or any(provider.id == run.provider for provider in self._providers)
        )
        require_tools = bool(
            run.required_tools or run.required_mcp_servers or run.required_mcp_tools
        )

        for index, provider in enumerate(self._providers):
            eligible_provider = False
            selected_model_detail = ""
            if not provider.enabled:
                reason = provider.reason
                detail = f"Provider is disabled: {provider.reason}."
            elif run.provider != "auto" and provider.id != run.provider:
                reason = "not_requested_provider"
                detail = f"Provider was not requested; requested provider is '{run.provider}'."
            elif self._settings.public_providers and provider.id not in self._settings.public_providers:
                reason = "not_in_public_provider_defaults"
                detail = "Provider is excluded by settings.public_providers."
            elif self._model_catalog is not None:
                selection = self._model_catalog.select_model(
                    provider.id,
                    min_tier=required_tier,
                    require_tools=require_tools,
                    require_structured_output=run.require_structured_output,
                    as_of=self._today,
                    stale_after_days=self._stale_metadata_max_age_days,
                )
                if selection is None:
                    reason = "no_matching_model"
                    detail = _build_no_model_detail(
                        required_tier=required_tier,
                        require_tools=require_tools,
                        require_structured_output=run.require_structured_output,
                    )
                else:
                    reason = "eligible"
                    eligible_provider = True
                    eligible.append(provider.id)
                    provider_model_selections.append((index, provider.id, selection))
                    selected_model_detail = _build_selected_model_detail(selection)
                    detail = (
                        "Provider passed current dry-run availability and settings filters. "
                        + selected_model_detail
                    )
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

        selected = None
        selected_model_id = None
        selected_model_tier = required_tier
        model_metadata_warnings: tuple[str, ...] = ()

        if provider_model_selections:
            selected_index, selected, selected_model = min(
                provider_model_selections,
                key=lambda item: item[2].sort_key + (item[0],),
            )
            del selected_index
            selected_model_id = selected_model.profile.model_id
            selected_model_tier = selected_model.profile.tier
            model_metadata_warnings = selected_model.stale_warnings
            reason = "selected_lowest_cost_eligible_model"
        elif eligible:
            selected = eligible[0]
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
            selected_model_id=selected_model_id,
            selected_model_tier=selected_model_tier,
            required_model_tier=required_tier,
            mode_default_model_tier=default_tier,
            task_complexity_tier=complexity.tier,
            task_complexity_reason=complexity.explanation,
            eligible_providers=tuple(eligible),
            rejected_providers=rejected,
            provider_reports=tuple(provider_reports),
            model_metadata_warnings=model_metadata_warnings,
            reason=reason,
            explanation=_build_explanation(
                selected_provider=selected,
                selected_model_id=selected_model_id,
                selected_model_tier=selected_model_tier,
                required_model_tier=required_tier,
                task_complexity_reason=complexity.explanation,
                eligible_providers=tuple(eligible),
                rejected_providers=rejected,
                provider_reports=tuple(provider_reports),
                model_metadata_warnings=model_metadata_warnings,
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


def _normalize_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RouteValidationError(f"{field_name} must be a boolean.")
    return value


def _resolve_required_tier(
    run: AgentRun,
    *,
    default_tier: str,
    complexity_tier: str,
    use_model_catalog: bool,
) -> str:
    if run.model_tier not in {None, "auto"}:
        return run.model_tier
    if use_model_catalog:
        return complexity_tier
    return default_tier


def _build_no_model_detail(
    *,
    required_tier: str,
    require_tools: bool,
    require_structured_output: bool,
) -> str:
    constraints = [f"tier '{required_tier}' or better"]
    if require_tools:
        constraints.append("tool support")
    if require_structured_output:
        constraints.append("structured output support")
    return "Provider passed current dry-run availability and settings filters, but no catalog model satisfied " + ", ".join(
        constraints
    ) + "."


def _build_selected_model_detail(selection: ModelSelection) -> str:
    detail = (
        f"Selected model '{selection.profile.model_id}' "
        f"(tier '{selection.profile.tier}', cost '{selection.profile.cost_profile}')."
    )
    if selection.stale_warnings:
        detail += " " + " ".join(selection.stale_warnings)
    return detail


def _build_explanation(
    *,
    selected_provider: str | None,
    selected_model_id: str | None,
    selected_model_tier: str,
    required_model_tier: str,
    task_complexity_reason: str,
    eligible_providers: tuple[str, ...],
    rejected_providers: dict[str, str],
    provider_reports: tuple[ProviderRouteReport, ...],
    model_metadata_warnings: tuple[str, ...],
) -> str:
    if selected_provider and selected_model_id:
        selected_clause = (
            f"Selected provider '{selected_provider}' with model '{selected_model_id}' "
            f"(tier '{selected_model_tier}') for required tier '{required_model_tier}'."
        )
    elif selected_provider:
        selected_clause = (
            f"Selected provider '{selected_provider}' with model tier '{selected_model_tier}'."
        )
    else:
        selected_clause = (
            f"No provider was selected; requested model tier resolved to '{required_model_tier}'."
        )

    complexity_clause = f"Task complexity: {task_complexity_reason}."

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

    warnings_clause = ""
    if model_metadata_warnings:
        warnings_clause = " Metadata warnings: " + " ".join(model_metadata_warnings)

    return " ".join((selected_clause, complexity_clause, eligibility_clause, rejection_clause)) + warnings_clause
