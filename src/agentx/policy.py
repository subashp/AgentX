from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase


VALID_CLASSIFICATIONS: tuple[str, ...] = (
    "public",
    "internal",
    "confidential",
    "proprietary",
    "secret",
)
_CLASSIFICATION_ORDER: dict[str, int] = {
    classification: index for index, classification in enumerate(VALID_CLASSIFICATIONS)
}


class PolicyError(ValueError):
    """Raised when policy data is invalid."""


class PolicyParseError(PolicyError):
    """Raised when policy text cannot be parsed."""


@dataclass(frozen=True)
class ClassificationRule:
    pattern: str
    classification: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern", _normalize_pattern(self.pattern))
        object.__setattr__(
            self,
            "classification",
            _normalize_classification(self.classification, "classification rule"),
        )

    def matches(self, path: str) -> bool:
        normalized_path = normalize_path(path)
        if fnmatchcase(normalized_path, self.pattern):
            return True
        if self.pattern.endswith("/**"):
            prefix = self.pattern[:-3].rstrip("/")
            return normalized_path == prefix or normalized_path.startswith(prefix + "/")
        return False


@dataclass(frozen=True)
class PathClassification:
    path: str
    normalized_path: str
    classification: str | None
    matched_pattern: str | None
    source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "normalized_path": self.normalized_path,
            "classification": self.classification,
            "matched_pattern": self.matched_pattern,
            "source": self.source,
        }


@dataclass(frozen=True)
class ProviderEligibility:
    eligible_provider_ids: tuple[str, ...]
    rejected_providers: dict[str, str]
    classifications: tuple[PathClassification, ...]
    highest_classification: str | None
    has_unclassified: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "eligible_provider_ids": list(self.eligible_provider_ids),
            "rejected_providers": dict(self.rejected_providers),
            "classifications": [entry.as_dict() for entry in self.classifications],
            "highest_classification": self.highest_classification,
            "has_unclassified": self.has_unclassified,
        }


@dataclass(frozen=True)
class Policy:
    external_max_classification: str = "internal"
    private_provider: str | None = None
    require_private_for_unclassified: bool = False
    default_classification: str | None = "internal"
    classification_rules: tuple[ClassificationRule, ...] = ()
    routing: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "external_max_classification",
            _normalize_classification(
                self.external_max_classification,
                "external_max_classification",
            ),
        )
        object.__setattr__(
            self,
            "private_provider",
            _normalize_optional_string(self.private_provider, "private_provider"),
        )
        object.__setattr__(
            self,
            "require_private_for_unclassified",
            _normalize_bool(
                self.require_private_for_unclassified,
                "require_private_for_unclassified",
            ),
        )
        object.__setattr__(
            self,
            "default_classification",
            _normalize_optional_classification(
                self.default_classification,
                "default_classification",
            ),
        )
        object.__setattr__(
            self,
            "classification_rules",
            _normalize_rules(self.classification_rules),
        )
        object.__setattr__(self, "routing", _normalize_routing(self.routing))

    def as_dict(self) -> dict[str, object]:
        return {
            "external_max_classification": self.external_max_classification,
            "private_provider": self.private_provider,
            "require_private_for_unclassified": self.require_private_for_unclassified,
            "default_classification": self.default_classification,
            "classification": {
                rule.pattern: rule.classification for rule in self.classification_rules
            },
            "routing": {
                classification: list(provider_ids)
                for classification, provider_ids in sorted(self.routing.items())
            },
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Policy:
        if not isinstance(raw, Mapping):
            raise PolicyError("policy must be a mapping.")

        defaults_raw = raw.get("defaults", raw)
        if not isinstance(defaults_raw, Mapping):
            raise PolicyError("policy defaults must be a mapping.")

        classification_raw = raw.get("classification", {})
        if not isinstance(classification_raw, Mapping):
            raise PolicyError("policy classification rules must be a mapping.")

        routing_raw = raw.get("routing", {})
        if not isinstance(routing_raw, Mapping):
            raise PolicyError("policy routing rules must be a mapping.")

        rules = tuple(
            ClassificationRule(pattern=str(pattern), classification=_required_string(value))
            for pattern, value in classification_raw.items()
        )

        return cls(
            external_max_classification=_required_string(
                defaults_raw.get("external_max_classification", "internal")
            ),
            private_provider=_optional_string(defaults_raw.get("private_provider"), "private_provider"),
            require_private_for_unclassified=_optional_bool(
                defaults_raw.get("require_private_for_unclassified", False),
                "require_private_for_unclassified",
            ),
            default_classification=_optional_classification_value(
                defaults_raw.get("default_classification", "internal"),
                "default_classification",
            ),
            classification_rules=rules,
            routing={
                _normalize_classification(str(classification), "routing classification"): _string_tuple(
                    provider_ids,
                    f"routing.{classification}",
                )
                for classification, provider_ids in routing_raw.items()
            },
        )

    @classmethod
    def from_text(cls, text: str) -> Policy:
        return cls.from_mapping(parse_policy_text(text))

    def classify_path(self, path: str) -> PathClassification:
        normalized_path = normalize_path(path)
        for rule in self.classification_rules:
            if rule.matches(normalized_path):
                return PathClassification(
                    path=path,
                    normalized_path=normalized_path,
                    classification=rule.classification,
                    matched_pattern=rule.pattern,
                    source="rule",
                )
        if self.default_classification is not None:
            return PathClassification(
                path=path,
                normalized_path=normalized_path,
                classification=self.default_classification,
                matched_pattern=None,
                source="default",
            )
        return PathClassification(
            path=path,
            normalized_path=normalized_path,
            classification=None,
            matched_pattern=None,
            source="unclassified",
        )

    def classify_paths(self, paths: Sequence[str]) -> tuple[PathClassification, ...]:
        return tuple(self.classify_path(path) for path in paths)

    def evaluate_provider_eligibility(
        self,
        provider_ids: Sequence[str],
        paths: Sequence[str],
        *,
        public_provider_ids: Collection[str] = (),
    ) -> ProviderEligibility:
        classifications = self.classify_paths(paths)
        normalized_public_provider_ids = {provider_id for provider_id in public_provider_ids}
        highest = highest_classification(
            entry.classification for entry in classifications if entry.classification is not None
        )
        has_unclassified = any(entry.classification is None for entry in classifications)
        exceeds_external_max = (
            highest is not None
            and compare_classification_levels(highest, self.external_max_classification) > 0
        )

        allowed_provider_ids = set(provider_ids)
        explicit_secret_routing = "secret" in self.routing
        if has_unclassified and self.require_private_for_unclassified:
            allowed_provider_ids = (
                {self.private_provider} if self.private_provider is not None else set()
            )
        if exceeds_external_max:
            allowed_provider_ids -= normalized_public_provider_ids

        routing_allowlist = self._routing_allowlist(classifications)
        if routing_allowlist is not None:
            allowed_provider_ids &= routing_allowlist

        rejected: dict[str, str] = {}
        eligible: list[str] = []
        for provider_id in provider_ids:
            if provider_id in allowed_provider_ids:
                eligible.append(provider_id)
                continue
            rejected[provider_id] = _policy_rejection_reason(
                provider_id=provider_id,
                public_provider_ids=normalized_public_provider_ids,
                has_unclassified=has_unclassified,
                require_private_for_unclassified=self.require_private_for_unclassified,
                private_provider=self.private_provider,
                exceeds_external_max=exceeds_external_max,
                has_secret=any(
                    entry.classification == "secret" for entry in classifications
                ),
                explicit_secret_routing=explicit_secret_routing,
                routing_allowlist=routing_allowlist,
            )

        return ProviderEligibility(
            eligible_provider_ids=tuple(eligible),
            rejected_providers=rejected,
            classifications=classifications,
            highest_classification=highest,
            has_unclassified=has_unclassified,
        )

    def _routing_allowlist(
        self,
        classifications: Sequence[PathClassification],
    ) -> set[str] | None:
        allowlist: set[str] | None = None
        seen_classifications = {
            entry.classification for entry in classifications if entry.classification is not None
        }
        for classification in VALID_CLASSIFICATIONS:
            if classification not in seen_classifications:
                continue
            if classification == "secret" and classification not in self.routing:
                return set()
            if classification not in self.routing:
                continue
            providers = set(self.routing[classification])
            if allowlist is None:
                allowlist = providers
            else:
                allowlist &= providers
        return allowlist


def parse_policy_text(text: str) -> dict[str, object]:
    result: dict[str, object] = {}
    current_section: dict[str, object] | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip()
            if not section_name:
                raise PolicyParseError(f"Line {line_number}: empty section name.")
            section = result.setdefault(section_name, {})
            if not isinstance(section, dict):
                raise PolicyParseError(
                    f"Line {line_number}: section '{section_name}' conflicts with a scalar value."
                )
            current_section = section
            continue
        if "=" not in line:
            raise PolicyParseError(f"Line {line_number}: expected 'key = value'.")
        if current_section is None:
            raise PolicyParseError(
                f"Line {line_number}: keys must appear inside a section."
            )
        key, raw_value = line.split("=", 1)
        normalized_key = _parse_key(key.strip(), line_number)
        current_section[normalized_key] = _parse_value(raw_value.strip(), line_number)

    return result


def normalize_path(path: str) -> str:
    if not isinstance(path, str):
        raise PolicyError("path must be a string.")
    normalized = path.replace("\\", "/").strip()
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    normalized = normalized.lstrip("./")
    normalized = normalized.strip("/")
    if not normalized:
        raise PolicyError("path must be a non-empty string.")
    return normalized


def compare_classification_levels(left: str, right: str) -> int:
    left_level = _classification_level(left)
    right_level = _classification_level(right)
    if left_level < right_level:
        return -1
    if left_level > right_level:
        return 1
    return 0


def highest_classification(classifications: Sequence[str]) -> str | None:
    highest: str | None = None
    highest_level = -1
    for classification in classifications:
        level = _classification_level(classification)
        if level > highest_level:
            highest = classification
            highest_level = level
    return highest


def _classification_level(classification: str) -> int:
    normalized = _normalize_classification(classification, "classification")
    return _CLASSIFICATION_ORDER[normalized]


def _normalize_pattern(pattern: object) -> str:
    if not isinstance(pattern, str):
        raise PolicyError("classification rule patterns must be strings.")
    normalized = pattern.replace("\\", "/").strip()
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    normalized = normalized.lstrip("./")
    normalized = normalized.strip("/")
    if not normalized:
        raise PolicyError("classification rule patterns must be non-empty strings.")
    return normalized


def _normalize_classification(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise PolicyError(f"{field_name} must be a string.")
    normalized = value.strip().lower()
    if normalized not in _CLASSIFICATION_ORDER:
        raise PolicyError(
            f"{field_name} must be one of: {', '.join(VALID_CLASSIFICATIONS)}."
        )
    return normalized


def _normalize_optional_classification(
    value: object,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return _normalize_classification(value, field_name)


def _normalize_optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PolicyError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise PolicyError(f"{field_name} must be a non-empty string when set.")
    return normalized


def _normalize_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyError(f"{field_name} must be a boolean.")
    return value


def _normalize_rules(value: object) -> tuple[ClassificationRule, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(
            ClassificationRule(pattern=str(pattern), classification=_required_string(classification))
            for pattern, classification in value.items()
        )
    if not isinstance(value, Sequence):
        raise PolicyError("classification_rules must be a sequence of rules.")

    rules: list[ClassificationRule] = []
    for entry in value:
        if isinstance(entry, ClassificationRule):
            rules.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise PolicyError("classification_rules entries must be ClassificationRule objects.")
        rules.append(
            ClassificationRule(
                pattern=_required_string(entry.get("pattern")),
                classification=_required_string(entry.get("classification")),
            )
        )
    return tuple(rules)


def _normalize_routing(value: object) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PolicyError("routing must be a mapping keyed by classification.")
    normalized: dict[str, tuple[str, ...]] = {}
    for classification, provider_ids in value.items():
        normalized[_normalize_classification(classification, "routing classification")] = _string_tuple(
            provider_ids,
            f"routing.{classification}",
        )
    return normalized


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PolicyError(f"{field_name} must be a list of provider ids.")
    provider_ids: list[str] = []
    for provider_id in value:
        if not isinstance(provider_id, str):
            raise PolicyError(f"{field_name} must be a list of provider ids.")
        normalized = provider_id.strip()
        if not normalized:
            raise PolicyError(f"{field_name} entries must be non-empty strings.")
        if normalized not in provider_ids:
            provider_ids.append(normalized)
    return tuple(provider_ids)


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PolicyError(f"{field_name} must be a string.")
    return value


def _optional_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyError(f"{field_name} must be a boolean.")
    return value


def _optional_classification_value(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PolicyError(f"{field_name} must be a string or null.")
    return value


def _required_string(value: object) -> str:
    if not isinstance(value, str):
        raise PolicyError("expected a string value.")
    return value


def _strip_comment(line: str) -> str:
    in_quotes = False
    quote_char = ""
    for index, character in enumerate(line):
        if character in {'"', "'"}:
            if not in_quotes:
                in_quotes = True
                quote_char = character
            elif quote_char == character:
                in_quotes = False
                quote_char = ""
        elif character == "#" and not in_quotes:
            return line[:index]
    return line


def _parse_key(key: str, line_number: int) -> str:
    if not key:
        raise PolicyParseError(f"Line {line_number}: empty key.")
    if (key.startswith('"') and key.endswith('"')) or (
        key.startswith("'") and key.endswith("'")
    ):
        return key[1:-1]
    return key


def _parse_value(raw_value: str, line_number: int) -> object:
    if raw_value in {"true", "false"}:
        return raw_value == "true"
    if raw_value in {"null", "None"}:
        return None
    if raw_value.startswith("[") and raw_value.endswith("]"):
        inner = raw_value[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(item.strip(), line_number) for item in _split_list(inner)]
    if (raw_value.startswith('"') and raw_value.endswith('"')) or (
        raw_value.startswith("'") and raw_value.endswith("'")
    ):
        return raw_value[1:-1]
    if raw_value:
        return raw_value
    raise PolicyParseError(f"Line {line_number}: empty value.")


def _split_list(value: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    in_quotes = False
    quote_char = ""
    for character in value:
        if character in {'"', "'"}:
            if not in_quotes:
                in_quotes = True
                quote_char = character
            elif quote_char == character:
                in_quotes = False
                quote_char = ""
        if character == "," and not in_quotes:
            items.append("".join(current))
            current = []
            continue
        current.append(character)
    items.append("".join(current))
    return items


def _policy_rejection_reason(
    *,
    provider_id: str,
    public_provider_ids: Collection[str],
    has_unclassified: bool,
    require_private_for_unclassified: bool,
    private_provider: str | None,
    exceeds_external_max: bool,
    has_secret: bool,
    explicit_secret_routing: bool,
    routing_allowlist: set[str] | None,
) -> str:
    if (
        has_unclassified
        and require_private_for_unclassified
        and provider_id != private_provider
    ):
        return "unclassified_requires_private"
    if exceeds_external_max and provider_id in public_provider_ids:
        return "classification_exceeds_external_max"
    if has_secret and not explicit_secret_routing:
        return "secret_requires_explicit_routing"
    if routing_allowlist is not None and provider_id not in routing_allowlist:
        return "classification_routing_restricted"
    return "policy_restricted"
