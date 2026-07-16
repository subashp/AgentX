from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from .adapters import (
    AdapterError,
    AdapterRequest,
    AdapterResult,
    _normalize_non_empty_string,
    _normalize_optional_timeout,
    _selected_model_id,
    _selected_model_tier,
)
from .routing import AgentRun


DEFAULT_PROVIDER_ID = "private-openai-compatible"


class OpenAICompatibleClientError(RuntimeError):
    """Raised for controlled private endpoint request/response failures."""

    def __init__(self, error_type: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


UrlOpen = Callable[..., Any]


@dataclass(frozen=True)
class OpenAICompatibleChatClient:
    """Small stdlib client for OpenAI-compatible chat-completions endpoints."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60.0
    opener: UrlOpen = urllib.request.urlopen

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _normalize_base_url(self.base_url))
        object.__setattr__(self, "model", _normalize_non_empty_string(self.model, "model"))
        if self.api_key is not None:
            object.__setattr__(self, "api_key", _normalize_optional_secret(self.api_key, "api_key"))
        object.__setattr__(self, "timeout", _normalize_optional_timeout(self.timeout))
        if not callable(self.opener):
            raise AdapterError("opener must be callable.")

    @property
    def chat_completions_url(self) -> str:
        return _chat_completions_url(self.base_url)

    @property
    def request_path(self) -> str:
        parsed = urllib.parse.urlsplit(self.chat_completions_url)
        path = parsed.path or "/"
        return path + (f"?{parsed.query}" if parsed.query else "")

    def build_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
    ) -> dict[str, object]:
        normalized_model = self.model if model is None else _normalize_non_empty_string(model, "model")
        return {
            "model": normalized_model,
            "messages": [_normalize_message(message) for message in messages],
            "stream": False,
        }

    def create_chat_completion(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
    ) -> Mapping[str, object]:
        payload = self.build_payload(messages, model=model)
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "agentx-openai-compatible/0.1",
        }
        if self.api_key is not None:
            headers["Authorization"] = "Bearer " + self.api_key

        request = urllib.request.Request(
            self.chat_completions_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with self.opener(request, timeout=self.timeout) as response:
                raw_body = response.read()
        except urllib.error.HTTPError as exc:
            raise OpenAICompatibleClientError(
                "http_error",
                f"OpenAI-compatible endpoint returned HTTP {exc.code}.",
                status_code=exc.code,
            ) from exc
        except TimeoutError as exc:
            raise OpenAICompatibleClientError(
                "timeout",
                f"OpenAI-compatible endpoint timed out after {self.timeout} seconds.",
            ) from exc
        except socket.timeout as exc:
            raise OpenAICompatibleClientError(
                "timeout",
                f"OpenAI-compatible endpoint timed out after {self.timeout} seconds.",
            ) from exc
        except urllib.error.URLError as exc:
            reason = _safe_url_error_reason(exc)
            raise OpenAICompatibleClientError(
                "url_error",
                f"OpenAI-compatible endpoint request failed: {reason}.",
            ) from exc
        except OSError as exc:
            raise OpenAICompatibleClientError(
                "connection_error",
                f"OpenAI-compatible endpoint request failed: {type(exc).__name__}.",
            ) from exc

        try:
            decoded = raw_body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OpenAICompatibleClientError(
                "malformed_response",
                "OpenAI-compatible endpoint returned non-UTF-8 response data.",
            ) from exc

        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleClientError(
                "malformed_json",
                "OpenAI-compatible endpoint returned malformed JSON.",
            ) from exc
        if not isinstance(payload, Mapping):
            raise OpenAICompatibleClientError(
                "malformed_json",
                "OpenAI-compatible endpoint response must be a JSON object.",
            )
        return dict(payload)


class OpenAICompatibleAdapter:
    """ProviderAdapter for local or private-cloud OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        provider_id: str = DEFAULT_PROVIDER_ID,
        client: OpenAICompatibleChatClient | None = None,
    ) -> None:
        self.provider_id = _normalize_non_empty_string(provider_id, "provider_id")
        if client is not None:
            if not isinstance(client, OpenAICompatibleChatClient):
                raise AdapterError("client must be an OpenAICompatibleChatClient.")
            self.client = client
        else:
            if base_url is None:
                raise AdapterError("base_url is required when client is not supplied.")
            if model is None:
                raise AdapterError("model is required when client is not supplied.")
            self.client = OpenAICompatibleChatClient(
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout=timeout,
            )

    def execute(self, request: AdapterRequest) -> AdapterResult:
        if not isinstance(request, AdapterRequest):
            raise AdapterError("request must be an AdapterRequest.")
        run = request.run
        model_tier = _selected_model_tier(run, request.route)
        model_id = _selected_model_id(self.client.model, request.route)
        messages = _build_agentx_messages(run)
        transcript_prefix = (
            {
                "sequence": 1,
                "event": "execution_started",
                "provider_id": self.provider_id,
                "model_id": model_id,
                "model_tier": model_tier,
                "mode": run.mode,
                "request_path": self.client.request_path,
                "timeout_seconds": self.client.timeout,
                "auth_configured": self.client.api_key is not None,
            },
            {
                "sequence": 2,
                "event": "request_prepared",
                "message_count": len(messages),
                "input_characters": sum(len(message["content"]) for message in messages),
                "context_path_count": len(run.context_paths),
            },
        )

        try:
            response = self.client.create_chat_completion(messages, model=model_id)
        except OpenAICompatibleClientError as exc:
            return self._failure_result(
                model_id=model_id,
                model_tier=model_tier,
                transcript_prefix=transcript_prefix,
                error_type=exc.error_type,
                message=str(exc),
                status_code=exc.status_code,
            )

        assistant_content = _extract_assistant_content(response)
        if assistant_content is None:
            return self._failure_result(
                model_id=model_id,
                model_tier=model_tier,
                transcript_prefix=transcript_prefix,
                error_type="missing_assistant_content",
                message="OpenAI-compatible endpoint response did not include assistant content.",
                status_code=None,
            )

        usage = _normalize_usage(response.get("usage"))
        transcript_events = transcript_prefix + (
            {
                "sequence": 3,
                "event": "response_received",
                "status": "success",
                "choice_count": _choice_count(response),
                "usage_present": usage is not None,
            },
            {
                "sequence": 4,
                "event": "execution_completed",
                "status": "success",
            },
        )
        outcome = {
            "status": "success",
            "outcome": "openai_compatible_completed",
            "summary": assistant_content,
            "patch_applied": False,
        }
        return AdapterResult(
            provider_id=self.provider_id,
            model_id=model_id,
            model_tier=model_tier,
            status="success",
            transcript_events=transcript_events,
            cost=_cost_from_usage(usage),
            outcome=outcome,
            patch="",
        )

    def _failure_result(
        self,
        *,
        model_id: str,
        model_tier: str,
        transcript_prefix: tuple[Mapping[str, object], ...],
        error_type: str,
        message: str,
        status_code: int | None,
    ) -> AdapterResult:
        event: dict[str, object] = {
            "sequence": 3,
            "event": "request_failed",
            "status": "failure",
            "error_type": error_type,
            "message": message,
        }
        if status_code is not None:
            event["http_status"] = status_code
        transcript_events = transcript_prefix + (
            event,
            {
                "sequence": 4,
                "event": "execution_completed",
                "status": "failure",
            },
        )
        outcome = {
            "status": "failure",
            "outcome": "openai_compatible_request_failed",
            "summary": message,
            "error_type": error_type,
            "patch_applied": False,
        }
        if status_code is not None:
            outcome["http_status"] = status_code
        return AdapterResult(
            provider_id=self.provider_id,
            model_id=model_id,
            model_tier=model_tier,
            status="failure",
            transcript_events=transcript_events,
            cost=_cost_from_usage(None),
            outcome=outcome,
            patch="",
        )


def _build_agentx_messages(run: AgentRun) -> tuple[Mapping[str, str], ...]:
    system = "\n".join(
        (
            "You are running behind AgentX as a private OpenAI-compatible provider adapter.",
            "Treat this as a plan-safe and execution-safe advisory run.",
            "Do not claim that files were edited, commands were run, patches were applied, or external systems were changed.",
            "Return a concise assistant response that AgentX can store as the run outcome summary.",
        )
    )
    lines = [
        f"Mode: {run.mode}",
        f"Requested provider: {run.provider}",
        f"Requested model tier: {run.model_tier or 'auto'}",
    ]
    if run.context_paths:
        lines.append("Context paths:")
        lines.extend(f"- {path}" for path in run.context_paths)
    if run.task_hints:
        lines.append("Task hints:")
        lines.extend(f"- {hint}" for hint in run.task_hints)
    if run.required_tools:
        lines.append("Required tools:")
        lines.extend(f"- {tool}" for tool in run.required_tools)
    if run.required_mcp_servers:
        lines.append("Required MCP servers:")
        lines.extend(f"- {server}" for server in run.required_mcp_servers)
    if run.required_mcp_tools:
        lines.append("Required MCP tools:")
        lines.extend(f"- {tool}" for tool in run.required_mcp_tools)
    lines.extend(("", "User prompt:", run.prompt))
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    )


def _chat_completions_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path + "/chat/completions"
    else:
        path = path + "/v1/chat/completions"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def _normalize_base_url(value: object) -> str:
    normalized = _normalize_non_empty_string(value, "base_url").rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AdapterError("base_url must be an absolute http or https URL.")
    if parsed.username is not None or parsed.password is not None:
        raise AdapterError("base_url must not include credentials.")
    if parsed.query or parsed.fragment:
        raise AdapterError("base_url must not include a query string or fragment.")
    return normalized


def _normalize_optional_secret(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise AdapterError(f"{field_name} must be a string or None.")
    normalized = value.strip()
    if not normalized:
        raise AdapterError(f"{field_name} must be non-empty when set.")
    return normalized


def _normalize_message(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise AdapterError("messages must contain mappings.")
    role = _normalize_non_empty_string(value.get("role"), "message.role")
    content = _normalize_non_empty_string(value.get("content"), "message.content")
    return {"role": role, "content": content}


def _extract_assistant_content(response: Mapping[str, object]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if isinstance(content, str):
        normalized = content.strip()
        return normalized or None
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        normalized = "".join(parts).strip()
        return normalized or None
    return None


def _normalize_usage(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    return dict(value)


def _cost_from_usage(usage: Mapping[str, object] | None) -> dict[str, object]:
    input_tokens = _integer_usage_field(usage, "prompt_tokens")
    output_tokens = _integer_usage_field(usage, "completion_tokens")
    total_tokens = _integer_usage_field(usage, "total_tokens")
    cost: dict[str, object] = {
        "currency": "USD",
        "estimated": False,
        "known": usage is not None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": 0.0,
    }
    if usage is not None:
        cost["usage"] = dict(usage)
    return cost


def _integer_usage_field(usage: Mapping[str, object] | None, field_name: str) -> int:
    if usage is None:
        return 0
    value = usage.get(field_name)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _choice_count(response: Mapping[str, object]) -> int:
    choices = response.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        return len(choices)
    return 0


def _safe_url_error_reason(exc: urllib.error.URLError) -> str:
    reason = exc.reason
    if isinstance(reason, BaseException):
        return type(reason).__name__
    if isinstance(reason, str):
        return reason
    return type(reason).__name__


__all__ = [
    "DEFAULT_PROVIDER_ID",
    "OpenAICompatibleAdapter",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleClientError",
]
