from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime


VALID_MODEL_TIERS: frozenset[str] = frozenset({"economy", "standard", "high"})
MODEL_TIER_RANKS: dict[str, int] = {
    "economy": 0,
    "standard": 1,
    "high": 2,
}
DEFAULT_STALE_METADATA_DAYS = 90

_MODE_COMPLEXITY_TIERS: dict[str, str] = {
    "plan": "high",
    "execute": "standard",
    "review": "standard",
    "explain": "economy",
    "tests": "economy",
    "docs": "economy",
}
_HINT_COMPLEXITY_TIERS: dict[str, str] = {
    "architecture": "high",
    "architecture_planning": "high",
    "planning": "high",
    "risky_refactor": "high",
    "review": "standard",
    "code_review": "standard",
    "execute": "standard",
    "execution": "standard",
    "bounded_execution": "standard",
    "tests": "economy",
    "test_generation": "economy",
    "docs": "economy",
    "documentation": "economy",
    "summarization": "economy",
    "summary": "economy",
    "log_triage": "economy",
    "logs": "economy",
}
_KNOWN_COST_RANKS: dict[str, int] = {
    "economy": 0,
    "standard": 1,
    "premium": 2,
}


class ModelCatalogError(ValueError):
    """Raised when model catalog metadata cannot be normalized."""


@dataclass(frozen=True)
class ModelProfile:
    provider_id: str
    model_id: str
    tier: str
    capability_score: int
    cost_profile: str
    latency_profile: str
    context_limit: int
    tool_support: bool
    structured_output_support: bool
    privacy_clearance: str
    best_for: tuple[str, ...]
    not_for: tuple[str, ...]
    metadata_source: str
    metadata_updated_at: date

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "tier": self.tier,
            "capability_score": self.capability_score,
            "cost_profile": self.cost_profile,
            "latency_profile": self.latency_profile,
            "context_limit": self.context_limit,
            "tool_support": self.tool_support,
            "structured_output_support": self.structured_output_support,
            "privacy_clearance": self.privacy_clearance,
            "best_for": list(self.best_for),
            "not_for": list(self.not_for),
            "metadata_source": self.metadata_source,
            "metadata_updated_at": self.metadata_updated_at.isoformat(),
        }

    def stale_warning(
        self,
        *,
        as_of: date | None = None,
        max_age_days: int = DEFAULT_STALE_METADATA_DAYS,
    ) -> str | None:
        stale_days = _normalize_positive_int(max_age_days, "max_age_days")
        reference_day = date.today() if as_of is None else _normalize_date(as_of, "as_of")
        age_days = (reference_day - self.metadata_updated_at).days
        if age_days > stale_days:
            return (
                f"Model metadata for '{self.provider_id}/{self.model_id}' is {age_days} days old "
                f"(updated {self.metadata_updated_at.isoformat()})."
            )
        return None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelProfile:
        required_fields = {
            "provider_id",
            "model_id",
            "tier",
            "capability_score",
            "cost_profile",
            "latency_profile",
            "context_limit",
            "tool_support",
            "structured_output_support",
            "privacy_clearance",
            "best_for",
            "not_for",
            "metadata_source",
            "metadata_updated_at",
        }
        unknown_fields = sorted(set(value) - required_fields)
        if unknown_fields:
            raise ModelCatalogError(
                "model profile contains unsupported fields: " + ", ".join(unknown_fields)
            )

        return cls(
            provider_id=_normalize_identifier(value.get("provider_id"), "provider_id"),
            model_id=_normalize_identifier(value.get("model_id"), "model_id"),
            tier=_normalize_tier(value.get("tier"), "tier"),
            capability_score=_normalize_positive_int(value.get("capability_score"), "capability_score"),
            cost_profile=_normalize_non_empty_string(value.get("cost_profile"), "cost_profile").lower(),
            latency_profile=_normalize_non_empty_string(value.get("latency_profile"), "latency_profile").lower(),
            context_limit=_normalize_positive_int(value.get("context_limit"), "context_limit"),
            tool_support=_normalize_bool(value.get("tool_support"), "tool_support"),
            structured_output_support=_normalize_bool(
                value.get("structured_output_support"),
                "structured_output_support",
            ),
            privacy_clearance=_normalize_non_empty_string(
                value.get("privacy_clearance"),
                "privacy_clearance",
            ).lower(),
            best_for=_normalize_string_collection(value.get("best_for"), "best_for"),
            not_for=_normalize_string_collection(value.get("not_for"), "not_for"),
            metadata_source=_normalize_non_empty_string(value.get("metadata_source"), "metadata_source"),
            metadata_updated_at=_normalize_date(
                value.get("metadata_updated_at"),
                "metadata_updated_at",
            ),
        )


@dataclass(frozen=True)
class ModelSelection:
    profile: ModelProfile
    required_tier: str
    stale_warnings: tuple[str, ...]

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            _cost_sort_key(self.profile.cost_profile),
            MODEL_TIER_RANKS[self.profile.tier],
            -self.profile.capability_score,
            self.profile.provider_id,
            self.profile.model_id,
        )


@dataclass(frozen=True)
class TaskComplexityAssessment:
    tier: str
    reasons: tuple[str, ...]

    @property
    def explanation(self) -> str:
        return "; ".join(self.reasons)

    def as_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ModelCatalog:
    profiles: tuple[ModelProfile, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[str, str]] = set()
        for profile in self.profiles:
            key = (profile.provider_id, profile.model_id)
            if key in seen:
                raise ModelCatalogError(
                    f"duplicate model profile '{profile.provider_id}/{profile.model_id}'."
                )
            seen.add(key)

    def as_dict(self) -> dict[str, object]:
        return {
            "profiles": [profile.as_dict() for profile in self.profiles],
        }

    def profiles_for_provider(self, provider_id: str) -> tuple[ModelProfile, ...]:
        normalized_provider = _normalize_identifier(provider_id, "provider_id")
        return tuple(
            profile for profile in self.profiles if profile.provider_id == normalized_provider
        )

    def stale_warnings(
        self,
        *,
        as_of: date | None = None,
        max_age_days: int = DEFAULT_STALE_METADATA_DAYS,
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        for profile in self.profiles:
            warning = profile.stale_warning(as_of=as_of, max_age_days=max_age_days)
            if warning:
                warnings.append(warning)
        return tuple(warnings)

    def select_model(
        self,
        provider_id: str,
        *,
        min_tier: str,
        require_tools: bool = False,
        require_structured_output: bool = False,
        as_of: date | None = None,
        stale_after_days: int = DEFAULT_STALE_METADATA_DAYS,
    ) -> ModelSelection | None:
        normalized_provider = _normalize_identifier(provider_id, "provider_id")
        required_tier = _normalize_tier(min_tier, "min_tier")
        eligible: list[ModelSelection] = []
        for profile in self.profiles_for_provider(normalized_provider):
            if MODEL_TIER_RANKS[profile.tier] < MODEL_TIER_RANKS[required_tier]:
                continue
            if require_tools and not profile.tool_support:
                continue
            if require_structured_output and not profile.structured_output_support:
                continue

            warning = profile.stale_warning(as_of=as_of, max_age_days=stale_after_days)
            eligible.append(
                ModelSelection(
                    profile=profile,
                    required_tier=required_tier,
                    stale_warnings=((warning,) if warning else ()),
                )
            )

        if not eligible:
            return None
        return min(eligible, key=lambda selection: selection.sort_key)

    @classmethod
    def from_dicts(cls, raw_profiles: Sequence[Mapping[str, object]]) -> ModelCatalog:
        if not isinstance(raw_profiles, Sequence) or isinstance(raw_profiles, (str, bytes)):
            raise ModelCatalogError("raw_profiles must be a sequence of mappings.")
        profiles: list[ModelProfile] = []
        for index, raw_profile in enumerate(raw_profiles):
            if not isinstance(raw_profile, Mapping):
                raise ModelCatalogError(f"model profile at index {index} must be a mapping.")
            profiles.append(ModelProfile.from_mapping(raw_profile))
        return cls(profiles=tuple(profiles))


def classify_task_complexity(
    mode: str,
    hints: Sequence[str] | str | None = None,
) -> TaskComplexityAssessment:
    normalized_mode = _normalize_identifier(mode, "mode")
    mode_tier = _MODE_COMPLEXITY_TIERS.get(normalized_mode)
    if mode_tier is None:
        raise ModelCatalogError(f"unsupported mode '{normalized_mode}' for complexity scoring.")

    normalized_hints = _normalize_hints(hints)
    highest_tier = mode_tier
    reasons = [f"mode '{normalized_mode}' maps to tier '{mode_tier}'"]
    matched_hints: list[str] = []

    for hint in normalized_hints:
        hint_tier = _HINT_COMPLEXITY_TIERS.get(hint)
        if hint_tier is None:
            continue
        matched_hints.append(hint)
        if MODEL_TIER_RANKS[hint_tier] > MODEL_TIER_RANKS[highest_tier]:
            highest_tier = hint_tier

    if matched_hints:
        reasons.append(
            "matched hints: "
            + ", ".join(f"'{hint}'" for hint in matched_hints)
            + f"; required tier resolved to '{highest_tier}'"
        )

    return TaskComplexityAssessment(
        tier=highest_tier,
        reasons=tuple(reasons),
    )


def _normalize_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ModelCatalogError(f"{field_name} must be a string.")
    normalized = value.strip().lower()
    if not normalized:
        raise ModelCatalogError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_tier(value: object, field_name: str) -> str:
    tier = _normalize_identifier(value, field_name)
    if tier not in VALID_MODEL_TIERS:
        raise ModelCatalogError(
            f"invalid {field_name} '{tier}'. Expected one of: "
            + ", ".join(sorted(VALID_MODEL_TIERS))
            + "."
        )
    return tier


def _normalize_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ModelCatalogError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ModelCatalogError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelCatalogError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ModelCatalogError(f"{field_name} must be greater than zero.")
    return value


def _normalize_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ModelCatalogError(f"{field_name} must be a boolean.")
    return value


def _normalize_string_collection(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise ModelCatalogError(f"{field_name} must be a string or a sequence of strings.")

    normalized_items: list[str] = []
    for item in items:
        normalized = _normalize_non_empty_string(item, field_name)
        if normalized not in normalized_items:
            normalized_items.append(normalized)
    return tuple(normalized_items)


def _normalize_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ModelCatalogError(f"{field_name} must be an ISO date string.")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ModelCatalogError(f"{field_name} must be an ISO date string.") from exc


def _normalize_hints(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise ModelCatalogError("hints must be a string or a sequence of strings.")

    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ModelCatalogError("hints entries must be strings.")
        canonical = item.strip().lower().replace("-", "_").replace(" ", "_")
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized)


def _cost_sort_key(cost_profile: str) -> tuple[object, ...]:
    if cost_profile in _KNOWN_COST_RANKS:
        return (0, _KNOWN_COST_RANKS[cost_profile])
    return (1, cost_profile)
