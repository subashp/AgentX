from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .policy import (
    Policy,
    PolicyError,
    VALID_CLASSIFICATIONS,
    compare_classification_levels,
    highest_classification,
    normalize_path,
)


VALID_PROVIDER_CLASSES: frozenset[str] = frozenset({"external", "private"})
VALID_MEMORY_ACTIONS: frozenset[str] = frozenset(
    {"include", "summarize", "redact", "exclude"}
)


class ContextError(ValueError):
    """Raised when context compilation inputs are invalid."""


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    classification: str
    content: str | None = None
    summary: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _normalize_non_empty_string(self.id, "memory.id"))
        object.__setattr__(
            self,
            "classification",
            _normalize_classification(self.classification, "memory.classification"),
        )
        object.__setattr__(self, "content", _normalize_optional_text(self.content, "memory.content"))
        object.__setattr__(self, "summary", _normalize_optional_text(self.summary, "memory.summary"))
        if self.content is None and self.summary is None:
            raise ContextError("memory must include content, summary, or both.")

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "classification": self.classification,
            "content": self.content,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class MemoryExposureDecision:
    memory_id: str
    classification: str
    action: str
    visible_text: str | None
    source: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "memory_id",
            _normalize_non_empty_string(self.memory_id, "memory_exposure.memory_id"),
        )
        object.__setattr__(
            self,
            "classification",
            _normalize_classification(
                self.classification,
                "memory_exposure.classification",
            ),
        )
        object.__setattr__(self, "action", _normalize_action(self.action))
        object.__setattr__(
            self,
            "source",
            _normalize_source(self.source, "memory_exposure.source"),
        )
        object.__setattr__(
            self,
            "reason",
            _normalize_non_empty_string(self.reason, "memory_exposure.reason"),
        )
        if self.action in {"include", "summarize"} and self.visible_text is None:
            raise ContextError(
                "memory exposure decisions that include provider-visible text require visible_text."
            )
        if self.action in {"redact", "exclude"} and self.visible_text is not None:
            raise ContextError(
                "memory exposure decisions that redact or exclude text cannot include visible_text."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "classification": self.classification,
            "action": self.action,
            "visible_text": self.visible_text,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RedactionEntry:
    target_type: str
    target_id: str
    classification: str | None
    action: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_type",
            _normalize_choice(self.target_type, "redaction.target_type", {"path", "memory"}),
        )
        object.__setattr__(
            self,
            "target_id",
            _normalize_non_empty_string(self.target_id, "redaction.target_id"),
        )
        if self.classification is not None:
            object.__setattr__(
                self,
                "classification",
                _normalize_classification(self.classification, "redaction.classification"),
            )
        object.__setattr__(self, "action", _normalize_action(self.action))
        object.__setattr__(
            self,
            "reason",
            _normalize_non_empty_string(self.reason, "redaction.reason"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "target_type": self.target_type,
            "target_id": self.target_id,
            "classification": self.classification,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SummarySubstitution:
    target_type: str
    target_id: str
    classification: str | None
    summary: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_type",
            _normalize_choice(
                self.target_type,
                "summary_substitution.target_type",
                {"path", "memory"},
            ),
        )
        object.__setattr__(
            self,
            "target_id",
            _normalize_non_empty_string(self.target_id, "summary_substitution.target_id"),
        )
        if self.classification is not None:
            object.__setattr__(
                self,
                "classification",
                _normalize_classification(
                    self.classification,
                    "summary_substitution.classification",
                ),
            )
        object.__setattr__(
            self,
            "summary",
            _normalize_non_empty_string(self.summary, "summary_substitution.summary"),
        )
        object.__setattr__(
            self,
            "reason",
            _normalize_non_empty_string(self.reason, "summary_substitution.reason"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "target_type": self.target_type,
            "target_id": self.target_id,
            "classification": self.classification,
            "summary": self.summary,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ProviderVisibleMemory:
    memory_id: str
    classification: str
    source: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "memory_id",
            _normalize_non_empty_string(self.memory_id, "provider_visible_memory.memory_id"),
        )
        object.__setattr__(
            self,
            "classification",
            _normalize_classification(
                self.classification,
                "provider_visible_memory.classification",
            ),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_source(self.source, "provider_visible_memory.source"),
        )
        object.__setattr__(
            self,
            "text",
            _normalize_non_empty_string(self.text, "provider_visible_memory.text"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "classification": self.classification,
            "source": self.source,
            "text": self.text,
        }


@dataclass(frozen=True)
class ProviderVisibleContext:
    provider_class: str
    visible_paths: tuple[str, ...]
    visible_memories: tuple[ProviderVisibleMemory, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_class", _normalize_provider_class(self.provider_class))
        object.__setattr__(
            self,
            "visible_paths",
            _normalize_path_collection(self.visible_paths, "provider_visible_context.visible_paths"),
        )
        object.__setattr__(
            self,
            "visible_memories",
            _normalize_memory_collection(self.visible_memories),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_class": self.provider_class,
            "visible_paths": list(self.visible_paths),
            "visible_memories": [memory.as_dict() for memory in self.visible_memories],
        }


@dataclass(frozen=True)
class PolicyDecision:
    provider_class: str
    eligible: bool
    reason: str
    external_max_classification: str
    effective_max_classification: str | None
    highest_requested_classification: str | None
    highest_included_classification: str | None
    has_unclassified_paths: bool
    secret_explicitly_allowed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_class", _normalize_provider_class(self.provider_class))
        object.__setattr__(self, "eligible", _normalize_bool(self.eligible, "policy_decision.eligible"))
        object.__setattr__(
            self,
            "reason",
            _normalize_non_empty_string(self.reason, "policy_decision.reason"),
        )
        object.__setattr__(
            self,
            "external_max_classification",
            _normalize_classification(
                self.external_max_classification,
                "policy_decision.external_max_classification",
            ),
        )
        if self.effective_max_classification is not None:
            object.__setattr__(
                self,
                "effective_max_classification",
                _normalize_classification(
                    self.effective_max_classification,
                    "policy_decision.effective_max_classification",
                ),
            )
        if self.highest_requested_classification is not None:
            object.__setattr__(
                self,
                "highest_requested_classification",
                _normalize_classification(
                    self.highest_requested_classification,
                    "policy_decision.highest_requested_classification",
                ),
            )
        if self.highest_included_classification is not None:
            object.__setattr__(
                self,
                "highest_included_classification",
                _normalize_classification(
                    self.highest_included_classification,
                    "policy_decision.highest_included_classification",
                ),
            )
        object.__setattr__(
            self,
            "has_unclassified_paths",
            _normalize_bool(
                self.has_unclassified_paths,
                "policy_decision.has_unclassified_paths",
            ),
        )
        object.__setattr__(
            self,
            "secret_explicitly_allowed",
            _normalize_bool(
                self.secret_explicitly_allowed,
                "policy_decision.secret_explicitly_allowed",
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_class": self.provider_class,
            "eligible": self.eligible,
            "reason": self.reason,
            "external_max_classification": self.external_max_classification,
            "effective_max_classification": self.effective_max_classification,
            "highest_requested_classification": self.highest_requested_classification,
            "highest_included_classification": self.highest_included_classification,
            "has_unclassified_paths": self.has_unclassified_paths,
            "secret_explicitly_allowed": self.secret_explicitly_allowed,
        }


@dataclass(frozen=True)
class ContextManifest:
    requested_paths: tuple[str, ...]
    inferred_paths: tuple[str, ...]
    included_paths: tuple[str, ...]
    excluded_paths: tuple[str, ...]
    classification_by_path: dict[str, str | None]
    provider_visible_context: ProviderVisibleContext
    redactions: tuple[RedactionEntry, ...]
    summary_substitutions: tuple[SummarySubstitution, ...]
    policy_decision: PolicyDecision
    memory_exposure: tuple[MemoryExposureDecision, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_paths",
            _normalize_path_collection(self.requested_paths, "context_manifest.requested_paths"),
        )
        object.__setattr__(
            self,
            "inferred_paths",
            _normalize_path_collection(self.inferred_paths, "context_manifest.inferred_paths"),
        )
        object.__setattr__(
            self,
            "included_paths",
            _normalize_path_collection(self.included_paths, "context_manifest.included_paths"),
        )
        object.__setattr__(
            self,
            "excluded_paths",
            _normalize_path_collection(self.excluded_paths, "context_manifest.excluded_paths"),
        )
        object.__setattr__(
            self,
            "classification_by_path",
            _normalize_classification_map(self.classification_by_path),
        )
        object.__setattr__(
            self,
            "provider_visible_context",
            _normalize_provider_visible_context(self.provider_visible_context),
        )
        object.__setattr__(self, "redactions", _normalize_redactions(self.redactions))
        object.__setattr__(
            self,
            "summary_substitutions",
            _normalize_summary_substitutions(self.summary_substitutions),
        )
        object.__setattr__(
            self,
            "policy_decision",
            _normalize_policy_decision(self.policy_decision),
        )
        object.__setattr__(
            self,
            "memory_exposure",
            _normalize_memory_exposure(self.memory_exposure),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_paths": list(self.requested_paths),
            "inferred_paths": list(self.inferred_paths),
            "included_paths": list(self.included_paths),
            "excluded_paths": list(self.excluded_paths),
            "classification_by_path": dict(self.classification_by_path),
            "provider_visible_context": self.provider_visible_context.as_dict(),
            "redactions": [entry.as_dict() for entry in self.redactions],
            "summary_substitutions": [entry.as_dict() for entry in self.summary_substitutions],
            "policy_decision": self.policy_decision.as_dict(),
            "memory_exposure": [decision.as_dict() for decision in self.memory_exposure],
        }


def compile_external_context_manifest(
    policy: Policy,
    *,
    requested_paths: Sequence[str],
    inferred_paths: Sequence[str] = (),
    memories: Sequence[MemoryRecord] = (),
) -> ContextManifest:
    return compile_context_manifest(
        policy,
        provider_class="external",
        requested_paths=requested_paths,
        inferred_paths=inferred_paths,
        memories=memories,
    )


def compile_private_context_manifest(
    policy: Policy,
    *,
    requested_paths: Sequence[str],
    inferred_paths: Sequence[str] = (),
    memories: Sequence[MemoryRecord] = (),
    allow_secret: bool = False,
) -> ContextManifest:
    return compile_context_manifest(
        policy,
        provider_class="private",
        requested_paths=requested_paths,
        inferred_paths=inferred_paths,
        memories=memories,
        allow_secret=allow_secret,
    )


def compile_context_manifest(
    policy: Policy,
    *,
    provider_class: str,
    requested_paths: Sequence[str],
    inferred_paths: Sequence[str] = (),
    memories: Sequence[MemoryRecord] = (),
    allow_secret: bool = False,
) -> ContextManifest:
    normalized_provider_class = _normalize_provider_class(provider_class)
    normalized_requested = _normalize_path_collection(
        requested_paths,
        "requested_paths",
    )
    normalized_inferred = _normalize_path_collection(
        inferred_paths,
        "inferred_paths",
    )
    path_order = _merge_paths(normalized_requested, normalized_inferred)
    path_classifications = {
        path: policy.classify_path(path) for path in path_order
    }

    classification_by_path = {
        path: path_classifications[path].classification for path in path_order
    }
    included_paths: list[str] = []
    excluded_paths: list[str] = []
    redactions: list[RedactionEntry] = []
    summary_substitutions: list[SummarySubstitution] = []

    for path in path_order:
        classification = path_classifications[path].classification
        include, reason = _should_include_path(
            policy=policy,
            provider_class=normalized_provider_class,
            classification=classification,
            allow_secret=allow_secret,
        )
        if include:
            included_paths.append(path)
            continue

        excluded_paths.append(path)
        redactions.append(
            RedactionEntry(
                target_type="path",
                target_id=path,
                classification=classification,
                action="exclude",
                reason=reason,
            )
        )
        if _should_add_path_summary(normalized_provider_class, classification):
            summary_substitutions.append(
                SummarySubstitution(
                    target_type="path",
                    target_id=path,
                    classification=classification,
                    summary=_build_path_summary(path, classification),
                    reason="summary_substitution_for_withheld_path",
                )
            )

    memory_exposure: list[MemoryExposureDecision] = []
    visible_memories: list[ProviderVisibleMemory] = []
    for memory in _normalize_memory_inputs(memories):
        decision = _compile_memory_exposure(
            memory=memory,
            provider_class=normalized_provider_class,
            allow_secret=allow_secret,
            policy=policy,
        )
        memory_exposure.append(decision)
        if decision.action in {"include", "summarize"}:
            visible_memories.append(
                ProviderVisibleMemory(
                    memory_id=decision.memory_id,
                    classification=decision.classification,
                    source=decision.source,
                    text=decision.visible_text or "",
                )
            )
        else:
            redactions.append(
                RedactionEntry(
                    target_type="memory",
                    target_id=decision.memory_id,
                    classification=decision.classification,
                    action=decision.action,
                    reason=decision.reason,
                )
            )

        if decision.action == "summarize":
            summary_substitutions.append(
                SummarySubstitution(
                    target_type="memory",
                    target_id=decision.memory_id,
                    classification=decision.classification,
                    summary=decision.visible_text or "",
                    reason="summary_substitution_for_memory",
                )
            )

    highest_requested = highest_classification(
        [
            classification
            for classification in classification_by_path.values()
            if classification is not None
        ]
    )
    highest_included = highest_classification(
        [
            classification_by_path[path]
            for path in included_paths
            if classification_by_path[path] is not None
        ]
    )
    has_unclassified_paths = any(
        classification is None for classification in classification_by_path.values()
    )
    eligible, reason = _policy_eligibility_reason(
        provider_class=normalized_provider_class,
        requested_paths=normalized_requested,
        included_paths=tuple(included_paths),
        classification_by_path=classification_by_path,
        require_private_for_unclassified=policy.require_private_for_unclassified,
        allow_secret=allow_secret,
    )

    return ContextManifest(
        requested_paths=normalized_requested,
        inferred_paths=normalized_inferred,
        included_paths=tuple(included_paths),
        excluded_paths=tuple(excluded_paths),
        classification_by_path=classification_by_path,
        provider_visible_context=ProviderVisibleContext(
            provider_class=normalized_provider_class,
            visible_paths=tuple(included_paths),
            visible_memories=tuple(visible_memories),
        ),
        redactions=tuple(redactions),
        summary_substitutions=tuple(summary_substitutions),
        policy_decision=PolicyDecision(
            provider_class=normalized_provider_class,
            eligible=eligible,
            reason=reason,
            external_max_classification=policy.external_max_classification,
            effective_max_classification=_effective_max_classification(
                normalized_provider_class,
                policy.external_max_classification,
            ),
            highest_requested_classification=highest_requested,
            highest_included_classification=highest_included,
            has_unclassified_paths=has_unclassified_paths,
            secret_explicitly_allowed=allow_secret,
        ),
        memory_exposure=tuple(memory_exposure),
    )


def _normalize_classification(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ContextError(f"{field_name} must be a string.")
    normalized = value.strip().lower()
    if normalized not in VALID_CLASSIFICATIONS:
        raise ContextError(
            f"{field_name} must be one of: {', '.join(VALID_CLASSIFICATIONS)}."
        )
    return normalized


def _normalize_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ContextError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ContextError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContextError(f"{field_name} must be a string when set.")
    if not value.strip():
        raise ContextError(f"{field_name} must be a non-empty string when set.")
    return value


def _normalize_action(value: object) -> str:
    return _normalize_choice(value, "memory action", VALID_MEMORY_ACTIONS)


def _normalize_source(value: object, field_name: str) -> str:
    return _normalize_choice(value, field_name, {"content", "summary", "metadata"})


def _normalize_choice(value: object, field_name: str, valid_values: set[str] | frozenset[str]) -> str:
    if not isinstance(value, str):
        raise ContextError(f"{field_name} must be a string.")
    normalized = value.strip().lower()
    if normalized not in valid_values:
        raise ContextError(
            f"{field_name} must be one of: {', '.join(sorted(valid_values))}."
        )
    return normalized


def _normalize_provider_class(value: object) -> str:
    return _normalize_choice(value, "provider_class", VALID_PROVIDER_CLASSES)


def _normalize_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContextError(f"{field_name} must be a boolean.")
    return value


def _normalize_path_collection(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, Sequence):
        raw_items = list(value)
    else:
        raise ContextError(f"{field_name} must be a string or a sequence of strings.")

    normalized: list[str] = []
    for item in raw_items:
        try:
            normalized_path = normalize_path(item)
        except PolicyError as exc:
            raise ContextError(f"{field_name}: {exc}") from exc
        if normalized_path not in normalized:
            normalized.append(normalized_path)
    return tuple(normalized)


def _normalize_memory_collection(value: object) -> tuple[ProviderVisibleMemory, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence):
        raise ContextError("visible memories must be a sequence.")
    memories: list[ProviderVisibleMemory] = []
    for item in value:
        if isinstance(item, ProviderVisibleMemory):
            memories.append(item)
            continue
        raise ContextError("visible memories must contain ProviderVisibleMemory entries.")
    return tuple(memories)


def _normalize_classification_map(value: object) -> dict[str, str | None]:
    if not isinstance(value, dict):
        raise ContextError("classification_by_path must be a dict.")
    normalized: dict[str, str | None] = {}
    for path, classification in value.items():
        normalized_path = normalize_path(path)
        if classification is not None:
            normalized_classification = _normalize_classification(
                classification,
                f"classification_by_path[{normalized_path}]",
            )
        else:
            normalized_classification = None
        normalized[normalized_path] = normalized_classification
    return normalized


def _normalize_provider_visible_context(value: object) -> ProviderVisibleContext:
    if not isinstance(value, ProviderVisibleContext):
        raise ContextError("provider_visible_context must be a ProviderVisibleContext.")
    return value


def _normalize_redactions(value: object) -> tuple[RedactionEntry, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence):
        raise ContextError("redactions must be a sequence.")
    entries: list[RedactionEntry] = []
    for item in value:
        if not isinstance(item, RedactionEntry):
            raise ContextError("redactions entries must be RedactionEntry objects.")
        entries.append(item)
    return tuple(entries)


def _normalize_summary_substitutions(value: object) -> tuple[SummarySubstitution, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence):
        raise ContextError("summary_substitutions must be a sequence.")
    entries: list[SummarySubstitution] = []
    for item in value:
        if not isinstance(item, SummarySubstitution):
            raise ContextError(
                "summary_substitutions entries must be SummarySubstitution objects."
            )
        entries.append(item)
    return tuple(entries)


def _normalize_policy_decision(value: object) -> PolicyDecision:
    if not isinstance(value, PolicyDecision):
        raise ContextError("policy_decision must be a PolicyDecision.")
    return value


def _normalize_memory_exposure(value: object) -> tuple[MemoryExposureDecision, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence):
        raise ContextError("memory_exposure must be a sequence.")
    decisions: list[MemoryExposureDecision] = []
    for item in value:
        if not isinstance(item, MemoryExposureDecision):
            raise ContextError(
                "memory_exposure entries must be MemoryExposureDecision objects."
            )
        decisions.append(item)
    return tuple(decisions)


def _normalize_memory_inputs(memories: Sequence[MemoryRecord]) -> tuple[MemoryRecord, ...]:
    if memories is None:
        return ()
    normalized: list[MemoryRecord] = []
    seen_ids: set[str] = set()
    for memory in memories:
        if not isinstance(memory, MemoryRecord):
            raise ContextError("memories must contain MemoryRecord entries.")
        if memory.id in seen_ids:
            raise ContextError(f"duplicate memory id '{memory.id}'.")
        seen_ids.add(memory.id)
        normalized.append(memory)
    return tuple(normalized)


def _merge_paths(requested_paths: Sequence[str], inferred_paths: Sequence[str]) -> tuple[str, ...]:
    merged = list(requested_paths)
    for path in inferred_paths:
        if path not in merged:
            merged.append(path)
    return tuple(merged)


def _effective_max_classification(provider_class: str, external_max: str) -> str | None:
    if provider_class == "private":
        return None
    if compare_classification_levels(external_max, "internal") < 0:
        return external_max
    return "internal"


def _should_include_path(
    *,
    policy: Policy,
    provider_class: str,
    classification: str | None,
    allow_secret: bool,
) -> tuple[bool, str]:
    if provider_class == "external":
        effective_max = _effective_max_classification(provider_class, policy.external_max_classification)
        if classification is None:
            if policy.require_private_for_unclassified:
                return False, "unclassified_path_requires_private_provider"
            return True, "included"
        if effective_max is not None and compare_classification_levels(classification, effective_max) <= 0:
            return True, "included"
        return False, "path_classification_not_visible_to_external_provider"

    if classification == "secret" and not allow_secret:
        return False, "secret_path_requires_explicit_private_opt_in"
    return True, "included"


def _should_add_path_summary(provider_class: str, classification: str | None) -> bool:
    if classification is None:
        return provider_class == "external"
    if classification == "secret":
        return False
    return provider_class == "external"


def _build_path_summary(path: str, classification: str | None) -> str:
    if classification is None:
        return f"Withheld unclassified path '{path}'. Provider received a summary placeholder only."
    return (
        f"Withheld {classification} path '{path}'. "
        "Provider received a summary placeholder only."
    )


def _compile_memory_exposure(
    *,
    memory: MemoryRecord,
    provider_class: str,
    allow_secret: bool,
    policy: Policy,
) -> MemoryExposureDecision:
    if provider_class == "external":
        effective_max = _effective_max_classification(provider_class, policy.external_max_classification)
        assert effective_max is not None
        if compare_classification_levels(memory.classification, effective_max) <= 0:
            text, source = _select_memory_text(memory)
            return MemoryExposureDecision(
                memory_id=memory.id,
                classification=memory.classification,
                action="include",
                visible_text=text,
                source=source,
                reason="memory_classification_visible_to_external_provider",
            )
        if memory.summary is not None:
            return MemoryExposureDecision(
                memory_id=memory.id,
                classification=memory.classification,
                action="summarize",
                visible_text=memory.summary,
                source="summary",
                reason="memory_summary_used_for_external_provider",
            )
        if memory.classification == "secret":
            return MemoryExposureDecision(
                memory_id=memory.id,
                classification=memory.classification,
                action="exclude",
                visible_text=None,
                source="metadata",
                reason="secret_memory_excluded_from_external_provider",
            )
        return MemoryExposureDecision(
            memory_id=memory.id,
            classification=memory.classification,
            action="redact",
            visible_text=None,
            source="metadata",
            reason="memory_redacted_from_external_provider",
        )

    if memory.classification == "secret" and not allow_secret:
        return MemoryExposureDecision(
            memory_id=memory.id,
            classification=memory.classification,
            action="exclude",
            visible_text=None,
            source="metadata",
            reason="secret_memory_requires_explicit_private_opt_in",
        )

    text, source = _select_memory_text(memory)
    return MemoryExposureDecision(
        memory_id=memory.id,
        classification=memory.classification,
        action="include",
        visible_text=text,
        source=source,
        reason="memory_visible_to_private_provider",
    )


def _select_memory_text(memory: MemoryRecord) -> tuple[str, str]:
    if memory.content is not None:
        return memory.content, "content"
    if memory.summary is not None:
        return memory.summary, "summary"
    raise ContextError(f"memory '{memory.id}' has no visible text.")


def _policy_eligibility_reason(
    *,
    provider_class: str,
    requested_paths: Sequence[str],
    included_paths: Sequence[str],
    classification_by_path: dict[str, str | None],
    require_private_for_unclassified: bool,
    allow_secret: bool,
) -> tuple[bool, str]:
    requested_set = set(requested_paths)
    included_requested = [path for path in included_paths if path in requested_set]
    if len(included_requested) == len(requested_paths):
        return True, "all_requested_paths_visible"

    blocked_requested = [
        path for path in requested_paths if path not in set(included_requested)
    ]
    blocked_classifications = {
        classification_by_path[path] for path in blocked_requested
    }

    if provider_class == "private":
        if "secret" in blocked_classifications and not allow_secret:
            return False, "requested_secret_paths_require_explicit_private_opt_in"
        return False, "requested_paths_blocked_by_private_context_policy"

    if None in blocked_classifications and require_private_for_unclassified:
        return False, "requested_unclassified_paths_require_private_provider"
    return False, "requested_paths_blocked_by_external_context_policy"
