from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .config import AgentXPaths
from .store import resolve_auth_service_path


_VALID_IDENTIFIER_CHARS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_SECRET_NAME_PARTS: tuple[str, ...] = (
    "auth",
    "credential",
    "key",
    "password",
    "secret",
    "token",
)
_REDACTED_VALUE = "[REDACTED]"


class MCPGatewayError(ValueError):
    """Raised when MCP policy or gateway inputs are invalid."""


@dataclass(frozen=True)
class MCPServicePolicy:
    service_id: str
    command: str | None = None
    args: tuple[str, ...] = ()
    endpoint: str | None = None
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    auth_service_id: str | None = None
    provider_visibility: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        service_id = _normalize_identifier(self.service_id, "service_id")
        command = _normalize_optional_string(self.command, "command")
        endpoint = _normalize_optional_string(self.endpoint, "endpoint")
        if command is None and endpoint is None:
            raise MCPGatewayError("MCP service policy requires command or endpoint.")

        object.__setattr__(self, "service_id", service_id)
        object.__setattr__(self, "command", command)
        object.__setattr__(
            self,
            "args",
            _normalize_string_tuple(self.args, "args"),
        )
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(
            self,
            "allowed_tools",
            _normalize_tool_tuple(self.allowed_tools, "allowed_tools"),
        )
        object.__setattr__(
            self,
            "denied_tools",
            _normalize_tool_tuple(self.denied_tools, "denied_tools"),
        )
        object.__setattr__(
            self,
            "auth_service_id",
            _normalize_optional_identifier(self.auth_service_id, "auth_service_id"),
        )
        object.__setattr__(
            self,
            "provider_visibility",
            _normalize_string_tuple(self.provider_visibility, "provider_visibility"),
        )

    def is_visible_to(self, provider_id: str) -> bool:
        normalized_provider_id = _normalize_identifier(provider_id, "provider_id")
        return (
            not self.provider_visibility
            or normalized_provider_id in self.provider_visibility
        )

    def sanitized_config_entry(
        self,
        *,
        included_tools: Sequence[str],
    ) -> dict[str, object]:
        entry: dict[str, object] = {
            "service_id": self.service_id,
            "tools": list(included_tools),
        }
        if self.command is not None:
            entry["command"] = self.command
        if self.args:
            entry["args"] = list(_sanitize_command_args(self.args))
        if self.endpoint is not None:
            entry["endpoint"] = _sanitize_endpoint(self.endpoint)
        if self.auth_service_id is not None:
            entry["auth"] = {
                "service_id": self.auth_service_id,
                "path_ref": f"agentx-auth://{self.auth_service_id}",
            }
        return entry


@dataclass(frozen=True)
class MCPAuthReference:
    service_id: str
    auth_service_id: str
    path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "service_id": self.service_id,
            "auth_service_id": self.auth_service_id,
            "path_ref": f"agentx-auth://{self.auth_service_id}",
            "resolved": True,
        }


@dataclass(frozen=True)
class MCPToolDecision:
    provider_id: str
    run_id: str
    service_id: str
    tool_name: str
    allowed: bool
    reason: str
    auth_service_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            _normalize_identifier(self.provider_id, "provider_id"),
        )
        object.__setattr__(self, "run_id", _normalize_identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "service_id",
            _normalize_identifier(self.service_id, "service_id"),
        )
        object.__setattr__(
            self,
            "tool_name",
            _normalize_tool_name(self.tool_name, "tool_name"),
        )
        object.__setattr__(self, "allowed", _normalize_bool(self.allowed, "allowed"))
        object.__setattr__(self, "reason", _normalize_reason(self.reason))
        object.__setattr__(
            self,
            "auth_service_id",
            _normalize_optional_identifier(self.auth_service_id, "auth_service_id"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "run_id": self.run_id,
            "service_id": self.service_id,
            "tool_name": self.tool_name,
            "allowed": self.allowed,
            "reason": self.reason,
            "auth_service_id": self.auth_service_id,
        }


@dataclass(frozen=True)
class MCPRunPolicyResult:
    provider_id: str
    run_id: str
    config: dict[str, object]
    decisions: tuple[MCPToolDecision, ...] = ()
    auth_references: tuple[MCPAuthReference, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "run_id": self.run_id,
            "config": _json_data(self.config),
            "decisions": [decision.as_dict() for decision in self.decisions],
            "auth_references": [
                reference.as_dict() for reference in self.auth_references
            ],
        }

    def auth_reference_for(self, service_id: str) -> MCPAuthReference | None:
        normalized = _normalize_identifier(service_id, "service_id")
        for reference in self.auth_references:
            if reference.service_id == normalized:
                return reference
        return None


@dataclass(frozen=True)
class RedactionEntry:
    path: str
    rule: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "rule": self.rule,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RedactionResult:
    payload: dict[str, object]
    redactions: tuple[RedactionEntry, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "payload": _json_data(self.payload),
            "redactions": [entry.as_dict() for entry in self.redactions],
        }


def generate_per_run_mcp_config(
    *,
    services: Sequence[MCPServicePolicy],
    paths: AgentXPaths,
    provider_id: str,
    run_id: str,
    required_services: Sequence[str] = (),
    required_tools: Sequence[str] = (),
) -> MCPRunPolicyResult:
    normalized_provider_id = _normalize_identifier(provider_id, "provider_id")
    normalized_run_id = _normalize_identifier(run_id, "run_id")
    service_by_id = _services_by_id(services)
    requested_services = set(_normalize_string_tuple(required_services, "required_services"))
    requested_tools = _requested_tools_by_service(required_tools)

    if not requested_services and not requested_tools:
        selected_service_ids = set(service_by_id)
    else:
        selected_service_ids = requested_services | set(requested_tools)

    config_services: dict[str, object] = {}
    decisions: list[MCPToolDecision] = []
    auth_references: list[MCPAuthReference] = []

    for service_id in sorted(selected_service_ids):
        service = service_by_id.get(service_id)
        requested_service_tools = requested_tools.get(service_id, ())
        if service is None:
            decisions.extend(
                _unknown_service_decisions(
                    provider_id=normalized_provider_id,
                    run_id=normalized_run_id,
                    service_id=service_id,
                    requested_tools=requested_service_tools,
                )
            )
            continue

        visible = service.is_visible_to(normalized_provider_id)
        candidate_tools = (
            requested_service_tools
            if requested_service_tools
            else service.allowed_tools
        )
        included_tools: list[str] = []

        for tool_name in candidate_tools:
            allowed, reason = _evaluate_tool(service, tool_name, visible=visible)
            decisions.append(
                MCPToolDecision(
                    provider_id=normalized_provider_id,
                    run_id=normalized_run_id,
                    service_id=service.service_id,
                    tool_name=tool_name,
                    allowed=allowed,
                    reason=reason,
                    auth_service_id=service.auth_service_id,
                )
            )
            if allowed:
                included_tools.append(tool_name)

        if not visible or not included_tools:
            continue

        config_services[service.service_id] = service.sanitized_config_entry(
            included_tools=included_tools,
        )
        if service.auth_service_id is not None:
            auth_references.append(
                MCPAuthReference(
                    service_id=service.service_id,
                    auth_service_id=service.auth_service_id,
                    path=resolve_auth_service_path(paths, service.auth_service_id),
                )
            )

    config = {
        "schema_version": 1,
        "provider_id": normalized_provider_id,
        "run_id": normalized_run_id,
        "mcp_services": config_services,
    }
    return MCPRunPolicyResult(
        provider_id=normalized_provider_id,
        run_id=normalized_run_id,
        config=config,
        decisions=tuple(decisions),
        auth_references=tuple(auth_references),
    )


def redact_mapping(
    payload: Mapping[str, object],
    *,
    redact_keys: Sequence[str] = (),
    redact_paths: Sequence[str] = (),
    replacement: str = _REDACTED_VALUE,
) -> RedactionResult:
    if not isinstance(payload, Mapping):
        raise MCPGatewayError("payload must be a mapping.")
    key_rules = frozenset(_normalize_string_tuple(redact_keys, "redact_keys"))
    path_rules = frozenset(_normalize_path_rule(rule) for rule in redact_paths)
    redactions: list[RedactionEntry] = []
    redacted = _redact_value(
        dict(payload),
        path=(),
        key_rules=key_rules,
        path_rules=path_rules,
        replacement=replacement,
        redactions=redactions,
    )
    if not isinstance(redacted, dict):
        raise MCPGatewayError("redacted payload must remain a mapping.")
    return RedactionResult(payload=redacted, redactions=tuple(redactions))


def mcp_tool_audit_event(
    decision: MCPToolDecision,
    *,
    sequence: int | None = None,
    argument_redactions: Sequence[RedactionEntry] = (),
    result_redactions: Sequence[RedactionEntry] = (),
) -> dict[str, object]:
    event: dict[str, object] = {
        "event": "mcp_tool_policy_decision",
        "provider_id": decision.provider_id,
        "run_id": decision.run_id,
        "service_id": decision.service_id,
        "tool_name": decision.tool_name,
        "allowed": decision.allowed,
        "reason": decision.reason,
        "auth_service_id": decision.auth_service_id,
        "argument_redactions": [
            redaction.as_dict() for redaction in argument_redactions
        ],
        "result_redactions": [redaction.as_dict() for redaction in result_redactions],
    }
    if sequence is not None:
        if not isinstance(sequence, int) or sequence < 1:
            raise MCPGatewayError("sequence must be a positive integer.")
        event["sequence"] = sequence
    return event


def _services_by_id(
    services: Sequence[MCPServicePolicy],
) -> dict[str, MCPServicePolicy]:
    if isinstance(services, (str, bytes)) or not isinstance(services, Sequence):
        raise MCPGatewayError("services must be a sequence of MCPServicePolicy objects.")
    result: dict[str, MCPServicePolicy] = {}
    for service in services:
        if not isinstance(service, MCPServicePolicy):
            raise MCPGatewayError("services must contain only MCPServicePolicy objects.")
        if service.service_id in result:
            raise MCPGatewayError(f"duplicate MCP service id '{service.service_id}'.")
        result[service.service_id] = service
    return result


def _requested_tools_by_service(
    required_tools: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    tool_references = _normalize_string_tuple(required_tools, "required_tools")
    result: dict[str, list[str]] = {}
    for tool_reference in tool_references:
        service_id, tool_name = _split_tool_reference(tool_reference)
        result.setdefault(service_id, []).append(tool_name)
    return {
        service_id: tuple(dict.fromkeys(tool_names))
        for service_id, tool_names in result.items()
    }


def _unknown_service_decisions(
    *,
    provider_id: str,
    run_id: str,
    service_id: str,
    requested_tools: Sequence[str],
) -> tuple[MCPToolDecision, ...]:
    tools = tuple(requested_tools) or ("*",)
    return tuple(
        MCPToolDecision(
            provider_id=provider_id,
            run_id=run_id,
            service_id=service_id,
            tool_name=tool_name,
            allowed=False,
            reason="service_not_configured",
        )
        for tool_name in tools
    )


def _evaluate_tool(
    service: MCPServicePolicy,
    tool_name: str,
    *,
    visible: bool,
) -> tuple[bool, str]:
    if not visible:
        return False, "service_not_visible_to_provider"
    if tool_name in service.denied_tools:
        return False, "tool_denied"
    if tool_name not in service.allowed_tools:
        return False, "tool_not_allowlisted"
    return True, "allowed"


def _redact_value(
    value: object,
    *,
    path: tuple[str, ...],
    key_rules: frozenset[str],
    path_rules: frozenset[tuple[str, ...]],
    replacement: str,
    redactions: list[RedactionEntry],
) -> object:
    if path and path in path_rules:
        redactions.append(
            RedactionEntry(
                path=_format_path(path),
                rule=_format_path(path),
                reason="path",
            )
        )
        return replacement

    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise MCPGatewayError("payload mapping keys must be strings.")
            item_path = path + (key,)
            if key in key_rules:
                redactions.append(
                    RedactionEntry(
                        path=_format_path(item_path),
                        rule=key,
                        reason="key",
                    )
                )
                result[key] = replacement
                continue
            result[key] = _redact_value(
                item,
                path=item_path,
                key_rules=key_rules,
                path_rules=path_rules,
                replacement=replacement,
                redactions=redactions,
            )
        return result

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _redact_value(
                item,
                path=path + (str(index),),
                key_rules=key_rules,
                path_rules=path_rules,
                replacement=replacement,
                redactions=redactions,
            )
            for index, item in enumerate(value)
        ]

    return value


def _split_tool_reference(tool_reference: str) -> tuple[str, str]:
    if "." not in tool_reference:
        raise MCPGatewayError(
            "required MCP tools must use '<service_id>.<tool_name>' references."
        )
    service_id, tool_name = tool_reference.split(".", 1)
    return (
        _normalize_identifier(service_id, "service_id"),
        _normalize_tool_name(tool_name, "tool_name"),
    )


def _sanitize_command_args(args: Sequence[str]) -> tuple[str, ...]:
    sanitized: list[str] = []
    redact_next = False
    for arg in args:
        if redact_next:
            sanitized.append(_REDACTED_VALUE)
            redact_next = False
            continue

        if "=" in arg:
            key, _separator, value = arg.partition("=")
            if _looks_secret_name(key):
                sanitized.append(f"{key}={_REDACTED_VALUE}")
            else:
                sanitized.append(arg)
            continue

        sanitized.append(arg)
        redact_next = _looks_secret_name(arg)
    return tuple(sanitized)


def _sanitize_endpoint(endpoint: str) -> str:
    split = urlsplit(endpoint)
    if not split.scheme or not split.netloc:
        return endpoint.split("?", 1)[0].split("#", 1)[0]

    host = split.hostname or ""
    if split.port is not None:
        host = f"{host}:{split.port}"
    return urlunsplit((split.scheme, host, split.path, "", ""))


def _looks_secret_name(value: str) -> bool:
    lowered = value.lower().lstrip("-_")
    return any(part in lowered for part in _SECRET_NAME_PARTS)


def _normalize_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise MCPGatewayError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise MCPGatewayError(f"{field_name} must be a non-empty string.")
    if normalized in {".", ".."}:
        raise MCPGatewayError(f"{field_name} must not be '.' or '..'.")
    if any(character not in _VALID_IDENTIFIER_CHARS for character in normalized):
        raise MCPGatewayError(
            f"{field_name} must use only letters, numbers, '.', '-', or '_'."
        )
    return normalized


def _normalize_tool_name(value: object, field_name: str) -> str:
    return _normalize_identifier(value, field_name)


def _normalize_reason(value: object) -> str:
    return _normalize_identifier(value, "reason")


def _normalize_optional_identifier(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalize_identifier(value, field_name)


def _normalize_optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MCPGatewayError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise MCPGatewayError(f"{field_name} must be non-empty when set.")
    return normalized


def _normalize_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise MCPGatewayError(f"{field_name} must be a string sequence.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise MCPGatewayError(f"{field_name} must contain only strings.")
        normalized = item.strip()
        if not normalized:
            raise MCPGatewayError(f"{field_name} cannot contain empty strings.")
        result.append(normalized)
    return tuple(dict.fromkeys(result))


def _normalize_tool_tuple(value: object, field_name: str) -> tuple[str, ...]:
    return tuple(
        _normalize_tool_name(tool_name, field_name)
        for tool_name in _normalize_string_tuple(value, field_name)
    )


def _normalize_path_rule(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise MCPGatewayError("redact_paths must contain only strings.")
    parts = tuple(part.strip() for part in value.split(".") if part.strip())
    if not parts:
        raise MCPGatewayError("redact_paths cannot contain empty paths.")
    return parts


def _normalize_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise MCPGatewayError(f"{field_name} must be a boolean.")
    return value


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _json_data(value: object) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _json_data(value.as_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_data(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_data(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise MCPGatewayError(f"value of type {type(value).__name__} is not JSON serializable.")


__all__ = [
    "MCPAuthReference",
    "MCPGatewayError",
    "MCPRunPolicyResult",
    "MCPServicePolicy",
    "MCPToolDecision",
    "RedactionEntry",
    "RedactionResult",
    "generate_per_run_mcp_config",
    "mcp_tool_audit_event",
    "redact_mapping",
]
