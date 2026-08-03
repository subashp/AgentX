from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Protocol

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
from .tools import ToolError, ToolExecutor, ToolResult, ToolSpec


DEFAULT_PROVIDER_ID = "private-openai-compatible"
_RAW_QWEN_TOOL_CALL = re.compile(
    r"<tool_call>\s*(?P<payload>\{.*?\})\s*</tool_call>", re.DOTALL
)


class OpenAICompatibleClientError(RuntimeError):
    """Raised for controlled private endpoint request/response failures."""

    def __init__(self, error_type: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


def _raise_if_cancelled(
    cancel_event: threading.Event | None,
    *,
    cause: BaseException | None = None,
) -> None:
    if cancel_event is None or not cancel_event.is_set():
        return
    error = OpenAICompatibleClientError("cancelled", "Request cancelled by user.")
    if cause is not None:
        raise error from cause
    raise error


UrlOpen = Callable[..., Any]


class RequestCancellation(Protocol):
    """Coordinates cancellation of one interactive provider request."""

    def request(
        self,
        cancel_request: Callable[[], None],
    ) -> ContextManager[threading.Event]: ...


@dataclass(frozen=True)
class OpenAICompatibleChatClient:
    """Small stdlib client for OpenAI-compatible chat-completions endpoints."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60.0
    opener: UrlOpen = urllib.request.urlopen
    _active_response: Any | None = field(default=None, init=False, repr=False, compare=False)
    _active_response_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

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
        messages: Sequence[Mapping[str, object]],
        *,
        model: str | None = None,
        stream: bool = False,
        tools: Sequence[Mapping[str, object]] | None = None,
        tool_choice: str | Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        normalized_model = self.model if model is None else _normalize_non_empty_string(model, "model")
        payload: dict[str, object] = {
            "model": normalized_model,
            "messages": [_normalize_message(message) for message in messages],
            "stream": stream,
        }
        if tools is not None:
            payload["tools"] = [dict(tool) for tool in tools]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return payload

    def create_chat_completion(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        model: str | None = None,
        tools: Sequence[Mapping[str, object]] | None = None,
        tool_choice: str | Mapping[str, object] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Mapping[str, object]:
        _raise_if_cancelled(cancel_event)
        payload = self.build_payload(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "agentx-openai-compatible/0.1",
            # ngrok's free browser interstitial otherwise intercepts API
            # clients before forwarding requests to the OpenAI-compatible
            # service. This is harmless for local and authenticated gateways.
            "ngrok-skip-browser-warning": "true",
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
                self._set_active_response(response)
                try:
                    raw_body = response.read()
                finally:
                    self._clear_active_response(response)
        except urllib.error.HTTPError as exc:
            _raise_if_cancelled(cancel_event, cause=exc)
            raise OpenAICompatibleClientError(
                "http_error",
                _http_error_message(exc),
                status_code=exc.code,
            ) from exc
        except TimeoutError as exc:
            _raise_if_cancelled(cancel_event, cause=exc)
            raise OpenAICompatibleClientError(
                "timeout",
                f"OpenAI-compatible endpoint timed out after {self.timeout} seconds.",
            ) from exc
        except socket.timeout as exc:
            _raise_if_cancelled(cancel_event, cause=exc)
            raise OpenAICompatibleClientError(
                "timeout",
                f"OpenAI-compatible endpoint timed out after {self.timeout} seconds.",
            ) from exc
        except urllib.error.URLError as exc:
            _raise_if_cancelled(cancel_event, cause=exc)
            reason = _safe_url_error_reason(exc)
            raise OpenAICompatibleClientError(
                "url_error",
                f"OpenAI-compatible endpoint request failed: {reason}.",
            ) from exc
        except (OSError, ValueError, http.client.HTTPException) as exc:
            _raise_if_cancelled(cancel_event, cause=exc)
            raise OpenAICompatibleClientError(
                "connection_error",
                f"OpenAI-compatible endpoint request failed: {type(exc).__name__}.",
            ) from exc

        _raise_if_cancelled(cancel_event)
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

    def stream_chat_completion(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        model: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[Mapping[str, object]]:
        _raise_if_cancelled(cancel_event)
        payload = self.build_payload(messages, model=model, stream=True)
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "User-Agent": "agentx-openai-compatible/0.1",
            "ngrok-skip-browser-warning": "true",
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
                self._set_active_response(response)
                try:
                    for event in _iter_sse_json(response):
                        _raise_if_cancelled(cancel_event)
                        yield event
                finally:
                    self._clear_active_response(response)
        except urllib.error.HTTPError as exc:
            _raise_if_cancelled(cancel_event, cause=exc)
            raise OpenAICompatibleClientError(
                "http_error",
                _http_error_message(exc),
                status_code=exc.code,
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            _raise_if_cancelled(cancel_event, cause=exc)
            raise OpenAICompatibleClientError(
                "timeout",
                f"OpenAI-compatible endpoint timed out after {self.timeout} seconds.",
            ) from exc
        except OpenAICompatibleClientError:
            raise
        except urllib.error.URLError as exc:
            _raise_if_cancelled(cancel_event, cause=exc)
            reason = _safe_url_error_reason(exc)
            raise OpenAICompatibleClientError(
                "url_error",
                f"OpenAI-compatible endpoint request failed: {reason}.",
            ) from exc
        except (OSError, ValueError, http.client.HTTPException) as exc:
            _raise_if_cancelled(cancel_event, cause=exc)
            raise OpenAICompatibleClientError(
                "connection_error",
                f"OpenAI-compatible endpoint request failed: {type(exc).__name__}.",
            ) from exc

    def cancel_active_request(self) -> None:
        """Close the active response so a blocking read returns promptly."""

        with self._active_response_lock:
            response = self._active_response
        if response is None:
            return
        try:
            response.close()
        except OSError:
            pass

    def _set_active_response(self, response: Any) -> None:
        with self._active_response_lock:
            object.__setattr__(self, "_active_response", response)

    def _clear_active_response(self, response: Any) -> None:
        with self._active_response_lock:
            if self._active_response is response:
                object.__setattr__(self, "_active_response", None)


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
        context_root: str | Path | None = None,
        client: OpenAICompatibleChatClient | None = None,
        stream: bool = False,
        stream_callback: Callable[[str, str], None] | None = None,
        tool_event_callback: Callable[[Mapping[str, object]], None] | None = None,
        tool_executor: ToolExecutor | None = None,
        max_tool_rounds: int = 8,
        request_cancellation: RequestCancellation | None = None,
    ) -> None:
        self.provider_id = _normalize_non_empty_string(provider_id, "provider_id")
        if not isinstance(stream, bool):
            raise AdapterError("stream must be a boolean.")
        if stream_callback is not None and not callable(stream_callback):
            raise AdapterError("stream_callback must be callable or None.")
        if tool_event_callback is not None and not callable(tool_event_callback):
            raise AdapterError("tool_event_callback must be callable or None.")
        self.stream = stream
        self.stream_callback = stream_callback
        self.tool_event_callback = tool_event_callback
        if request_cancellation is not None and not callable(
            getattr(request_cancellation, "request", None)
        ):
            raise AdapterError("request_cancellation must provide request().")
        self.request_cancellation = request_cancellation
        if tool_executor is not None:
            if not hasattr(tool_executor, "specs") or not callable(getattr(tool_executor, "call", None)):
                raise AdapterError("tool_executor must provide specs and call().")
            specs = getattr(tool_executor, "specs")
            if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)) or not all(
                isinstance(spec, ToolSpec) for spec in specs
            ):
                raise AdapterError("tool_executor.specs must contain ToolSpec values.")
        if isinstance(max_tool_rounds, bool) or not isinstance(max_tool_rounds, int) or not 1 <= max_tool_rounds <= 32:
            raise AdapterError("max_tool_rounds must be an integer from 1 to 32.")
        self.tool_executor = tool_executor
        self.max_tool_rounds = max_tool_rounds
        if context_root is not None and not isinstance(context_root, (str, Path)):
            raise AdapterError("context_root must be a string, Path, or None.")
        self.context_root = None if context_root is None else Path(context_root)
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
        messages = _build_agentx_messages(
            run,
            context_map=request.context_map,
            context_root=self.context_root,
            available_tool_names=(
                tuple(spec.name for spec in self.tool_executor.specs)
                if self.tool_executor is not None
                else ()
            ),
        )
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
            tool_calls: tuple[str, ...] = ()
            tool_trace: tuple[Mapping[str, object], ...] = ()
            if self.tool_executor is not None:
                response, assistant_content, thinking, usage, tool_calls, tool_trace = self._execute_with_tools(
                    messages,
                    model_id=model_id,
                )
            elif self.stream:
                assistant_content, thinking, usage = self._execute_stream(
                    messages,
                    model_id=model_id,
                )
                response = None
            else:
                response = self._create_chat_completion(messages, model=model_id)
                assistant_content, thinking = _extract_assistant_message(response)
                usage = _normalize_usage(response.get("usage"))
        except OpenAICompatibleClientError as exc:
            self._notify_stream("error", "")
            return self._failure_result(
                model_id=model_id,
                model_tier=model_tier,
                transcript_prefix=transcript_prefix,
                error_type=exc.error_type,
                message=str(exc),
                status_code=exc.status_code,
            )

        if assistant_content is None:
            self._notify_stream("error", "")
            return self._failure_result(
                model_id=model_id,
                model_tier=model_tier,
                transcript_prefix=transcript_prefix,
                error_type="missing_assistant_content",
                message="OpenAI-compatible endpoint response did not include assistant content.",
                status_code=None,
            )

        self._notify_stream("complete", "")
        transcript_events = transcript_prefix + (
            {
                "sequence": 3,
                "event": "response_received",
                "status": "success",
                "choice_count": 1 if self.stream and self.tool_executor is None else _choice_count(response),
                "usage_present": usage is not None,
                "tools_used": list(tool_calls),
                "tool_trace": [dict(entry) for entry in tool_trace],
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
            "response": assistant_content,
            "streamed": self.stream and self.tool_executor is None,
            "patch_applied": False,
        }
        if thinking:
            outcome["thinking"] = thinking
        if tool_calls:
            outcome["tools_used"] = list(tool_calls)
            outcome["tool_trace"] = [dict(entry) for entry in tool_trace]
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

    def _execute_with_tools(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        model_id: str,
    ) -> tuple[
        Mapping[str, object] | None,
        str | None,
        str | None,
        Mapping[str, object] | None,
        tuple[str, ...],
        tuple[Mapping[str, object], ...],
    ]:
        if self.tool_executor is None:
            raise AdapterError("tool executor is not configured.")
        conversation: list[Mapping[str, object]] = [dict(message) for message in messages]
        tool_specs = tuple(spec.as_dict() for spec in self.tool_executor.specs)
        used_tools: list[str] = []
        tool_trace: list[Mapping[str, object]] = []
        usage: Mapping[str, object] | None = None

        for _round in range(self.max_tool_rounds):
            round_number = _round + 1
            response = self._create_chat_completion(
                conversation,
                model=model_id,
                tools=tool_specs,
                tool_choice="auto",
            )
            event_usage = _normalize_usage(response.get("usage"))
            if event_usage is not None:
                usage = _merge_usage(usage, event_usage)
            message = _first_assistant_message(response)
            if message is None:
                raise OpenAICompatibleClientError(
                    "malformed_response",
                    "OpenAI-compatible endpoint response did not include an assistant message.",
                )
            tool_calls = _extract_tool_calls(message)
            raw_qwen_tool_calls = False
            if not tool_calls:
                tool_calls = _extract_raw_qwen_tool_calls(message)
                raw_qwen_tool_calls = bool(tool_calls)
            if not tool_calls:
                assistant_content, thinking = _extract_assistant_message(response)
                return response, assistant_content, thinking, usage, tuple(used_tools), tuple(tool_trace)

            conversation.append(
                _assistant_tool_message(
                    message,
                    tool_calls,
                    raw_qwen_tool_calls=raw_qwen_tool_calls,
                )
            )
            for tool_call in tool_calls:
                name = tool_call["name"]
                call_id = tool_call["id"]
                used_tools.append(name)
                arguments = tool_call["arguments"]
                self._notify_tool_event(
                    "requested",
                    name,
                    _tool_event_request_details(round_number, arguments),
                )
                self._notify_tool_event("started", name, {"round": round_number})
                result = _run_tool_call(self.tool_executor, name, arguments)
                self._notify_tool_event(
                    _tool_result_event_name(result),
                    name,
                    _tool_event_result_details(round_number, result),
                )
                tool_trace.append(_tool_trace_entry(name, result))
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result.as_json(),
                    }
                )

        raise OpenAICompatibleClientError(
            "tool_loop_limit",
            f"OpenAI-compatible provider exceeded the {self.max_tool_rounds}-round tool limit.",
        )

    def _execute_stream(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model_id: str,
    ) -> tuple[str | None, str | None, Mapping[str, object] | None]:
        accumulator = _StreamingAssistantAccumulator(self._notify_stream)
        usage: Mapping[str, object] | None = None
        with self._request_scope() as cancel_event:
            for event in self.client.stream_chat_completion(
                messages,
                model=model_id,
                cancel_event=cancel_event,
            ):
                event_usage = _normalize_usage(event.get("usage"))
                if event_usage is not None:
                    usage = event_usage
                choices = event.get("choices")
                if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
                    continue
                first = choices[0]
                if not isinstance(first, Mapping):
                    continue
                delta = first.get("delta") or first.get("message")
                if not isinstance(delta, Mapping):
                    continue
                accumulator.feed_reasoning(
                    _extract_stream_text_value(
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or delta.get("thinking")
                    )
                )
                accumulator.feed_content(_extract_stream_text_value(delta.get("content")))
        accumulator.finish()
        return accumulator.response, accumulator.thinking, usage

    def _create_chat_completion(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        model: str,
        tools: Sequence[Mapping[str, object]] | None = None,
        tool_choice: str | Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        with self._request_scope() as cancel_event:
            return self.client.create_chat_completion(
                messages,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                cancel_event=cancel_event,
            )

    def _request_scope(self) -> ContextManager[threading.Event | None]:
        if self.request_cancellation is None:
            return nullcontext(None)
        return self.request_cancellation.request(self.client.cancel_active_request)

    def _notify_stream(self, kind: str, value: str) -> None:
        if self.stream_callback is not None:
            self.stream_callback(kind, value)

    def _notify_tool_event(self, event: str, name: str, details: Mapping[str, object]) -> None:
        if self.tool_event_callback is None:
            return
        payload = {"event": event, "name": name}
        payload.update(details)
        self.tool_event_callback(payload)

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


_MAX_CONTEXT_FILE_CHARACTERS = 12_000
_MAX_CONTEXT_CHARACTERS = 28_000


def _build_agentx_messages(
    run: AgentRun,
    *,
    context_map: Mapping[str, object] | None = None,
    context_root: Path | None = None,
    available_tool_names: Sequence[str] = (),
) -> tuple[Mapping[str, str], ...]:
    system = _agentx_system_prompt(run.mode)
    lines = [
        f"Mode: {run.mode}",
        f"Requested provider: {run.provider}",
        f"Requested model tier: {run.model_tier or 'auto'}",
    ]
    normalized_tool_names = tuple(
        name.strip()
        for name in available_tool_names
        if isinstance(name, str) and name.strip()
    )
    if run.mode == "execute" and normalized_tool_names:
        lines.append("Available AgentX tools:")
        lines.extend(f"- {name}" for name in normalized_tool_names)
    if run.context_paths:
        lines.append("Context paths:")
        lines.extend(f"- {path}" for path in run.context_paths)
    context_text = _read_visible_context(
        context_map=context_map,
        context_root=context_root,
    )
    if context_text:
        lines.extend(("", "Policy-approved context contents:", context_text))
    memory_text = _read_visible_memories(context_map=context_map)
    if memory_text:
        lines.extend(("", "Policy-approved memory:", memory_text))
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


def _agentx_system_prompt(mode: str) -> str:
    common = (
        "You are running behind AgentX as a private OpenAI-compatible provider adapter.",
        "When web research tools are available, use web_fetch for a user-named website or URL; use web_search only when the user did not name a source.",
        "Return a concise assistant response that AgentX can store as the run outcome summary.",
    )
    if mode == "execute":
        return "\n".join(
            common
            + (
                "Treat this as a bounded execute-mode coding run.",
                "You may inspect files with workspace tools, request test commands with shell_exec, and request scoped patches with workspace_patch when those tools are available.",
                "Every shell command, patch, browser side effect, and internet request is approved by AgentX outside the prompt before it runs.",
                "Use tool results as ground truth: do not claim a command ran, a patch applied, or tests passed unless a returned tool result confirms it.",
                "After a failing test, inspect the failure, request a focused patch, and rerun the relevant focused test when possible.",
                "Stop and summarize when validation passes, approval is denied, safety limits are reached, or the same failure repeats.",
            )
        )
    return "\n".join(
        common
        + (
            "Treat this as a plan-safe advisory run.",
            "Do not edit files, run mutating commands, apply patches, or modify the workspace.",
            "Do not claim that files were edited, commands were run, patches were applied, or external systems were changed.",
        )
    )


def _read_visible_context(
    *,
    context_map: Mapping[str, object] | None,
    context_root: Path | None,
) -> str:
    if context_map is None or context_root is None:
        return ""
    included_paths = context_map.get("included_paths")
    if not isinstance(included_paths, Sequence) or isinstance(included_paths, (str, bytes)):
        return ""

    root = context_root.resolve()
    sections: list[str] = []
    total_characters = 0
    for raw_path in included_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        relative_path = Path(raw_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            continue
        path = (root / relative_path).resolve()
        if path != root and root not in path.parents:
            continue
        if not path.is_file():
            continue
        try:
            full_content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        remaining = _MAX_CONTEXT_CHARACTERS - total_characters
        if remaining <= 0:
            break
        content = full_content[: min(_MAX_CONTEXT_FILE_CHARACTERS, remaining)]
        if not content:
            continue
        suffix = "" if len(content) == len(full_content) else "\n[truncated]"
        sections.append(f"--- {raw_path} ---\n{content}{suffix}")
        total_characters += len(content) + len(raw_path) + 10
    return "\n\n".join(sections)


def _read_visible_memories(*, context_map: Mapping[str, object] | None) -> str:
    if context_map is None:
        return ""
    packet = context_map.get("agentmemory_prompt")
    if isinstance(packet, Mapping):
        rendered = packet.get("rendered_text")
        if isinstance(rendered, str) and rendered.strip():
            return rendered.strip()[:12_000]
    visible_context = context_map.get("provider_visible_context")
    if not isinstance(visible_context, Mapping):
        return ""
    memories = visible_context.get("visible_memories")
    if not isinstance(memories, Sequence) or isinstance(memories, (str, bytes)):
        return ""
    sections: list[str] = []
    for memory in memories:
        if not isinstance(memory, Mapping):
            continue
        memory_id = memory.get("memory_id")
        text = memory.get("text")
        classification = memory.get("classification")
        if isinstance(memory_id, str) and isinstance(text, str) and text.strip():
            sections.append(f"- [{classification}] {memory_id}: {text.strip()}")
    return "\n".join(sections)[:12_000]


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


def _normalize_message(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AdapterError("messages must contain mappings.")
    role = _normalize_non_empty_string(value.get("role"), "message.role")
    content = value.get("content")
    if content is None:
        if role != "assistant" or "tool_calls" not in value:
            raise AdapterError("message.content must be a non-empty string.")
    elif not isinstance(content, str) or not content.strip():
        raise AdapterError("message.content must be a non-empty string.")
    normalized: dict[str, object] = {"role": role, "content": content}
    if "tool_calls" in value:
        tool_calls = value["tool_calls"]
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
            raise AdapterError("message.tool_calls must be a list.")
        normalized["tool_calls"] = [dict(call) for call in tool_calls if isinstance(call, Mapping)]
    for field in ("tool_call_id", "name"):
        if field in value:
            normalized[field] = _normalize_non_empty_string(value[field], f"message.{field}")
    return normalized


def _first_assistant_message(response: Mapping[str, object]) -> Mapping[str, object] | None:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    return message if isinstance(message, Mapping) else None


def _extract_tool_calls(message: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return ()
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        raise OpenAICompatibleClientError(
            "malformed_response",
            "OpenAI-compatible assistant tool_calls must be a list.",
        )
    calls: list[dict[str, object]] = []
    for index, raw_call in enumerate(raw_calls, start=1):
        if not isinstance(raw_call, Mapping):
            raise OpenAICompatibleClientError(
                "malformed_response",
                "OpenAI-compatible assistant tool calls must be objects.",
            )
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            raise OpenAICompatibleClientError(
                "malformed_response",
                "OpenAI-compatible tool call is missing its function object.",
            )
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise OpenAICompatibleClientError(
                "malformed_response",
                "OpenAI-compatible tool call is missing a function name.",
            )
        call_id = raw_call.get("id") or f"agentx-tool-call-{index}"
        if not isinstance(call_id, str) or not call_id.strip():
            raise OpenAICompatibleClientError(
                "malformed_response",
                "OpenAI-compatible tool call is missing an id.",
            )
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"__agentx_invalid_arguments__": arguments}
        if not isinstance(arguments, Mapping):
            arguments = {"__agentx_invalid_arguments__": arguments}
        calls.append({"id": call_id, "name": name.strip(), "arguments": dict(arguments)})
    return tuple(calls)


def _extract_raw_qwen_tool_calls(message: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Decode Qwen's raw XML-wrapped JSON fallback when vLLM misses it.

    vLLM normally translates model output into OpenAI ``tool_calls``. Some
    Qwen/vLLM combinations leave complete ``<tool_call>{...}</tool_call>``
    blocks in ``message.content`` instead. Only an otherwise-empty sequence of
    complete blocks is accepted, so ordinary assistant prose is never executed
    as a tool request.
    """
    content = _extract_text_value(message.get("content"))
    if content is None:
        return ()
    matches = tuple(_RAW_QWEN_TOOL_CALL.finditer(content))
    if not matches:
        if "<tool_call>" in content or "</tool_call>" in content:
            raise OpenAICompatibleClientError(
                "malformed_tool_call",
                "Qwen returned an incomplete or malformed raw tool-call block.",
            )
        return ()
    remainder = _RAW_QWEN_TOOL_CALL.sub("", content).strip()
    if remainder:
        raise OpenAICompatibleClientError(
            "malformed_tool_call",
            "Qwen mixed raw tool-call blocks with assistant content.",
        )

    calls: list[dict[str, object]] = []
    for index, match in enumerate(matches, start=1):
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleClientError(
                "malformed_tool_call",
                "Qwen raw tool-call arguments must be valid JSON.",
            ) from exc
        if not isinstance(payload, Mapping):
            raise OpenAICompatibleClientError(
                "malformed_tool_call",
                "Qwen raw tool-call payload must be a JSON object.",
            )
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise OpenAICompatibleClientError(
                "malformed_tool_call",
                "Qwen raw tool-call payload is missing a function name.",
            )
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"__agentx_invalid_arguments__": arguments}
        if not isinstance(arguments, Mapping):
            arguments = {"__agentx_invalid_arguments__": arguments}
        calls.append(
            {
                "id": f"agentx-qwen-tool-call-{index}",
                "name": name.strip(),
                "arguments": dict(arguments),
            }
        )
    return tuple(calls)


def _assistant_tool_message(
    message: Mapping[str, object],
    tool_calls: Sequence[Mapping[str, object]],
    *,
    raw_qwen_tool_calls: bool,
) -> dict[str, object]:
    """Return the assistant turn in the OpenAI shape expected before tools."""
    if not raw_qwen_tool_calls:
        return dict(message)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": str(call["id"]),
                "type": "function",
                "function": {
                    "name": str(call["name"]),
                    "arguments": json.dumps(call["arguments"], separators=(",", ":")),
                },
            }
            for call in tool_calls
        ],
    }


def _run_tool_call(executor: ToolExecutor, name: str, arguments: Mapping[str, object]) -> ToolResult:
    if "__agentx_invalid_arguments__" in arguments:
        return ToolResult(
            name=name,
            ok=False,
            error="tool arguments must be a valid JSON object",
        )
    try:
        result = executor.call(name, arguments)
    except (ToolError, TypeError, ValueError, KeyError) as exc:
        return ToolResult(name=name, ok=False, error=str(exc) or "tool call failed")
    if not isinstance(result, ToolResult):
        return ToolResult(name=name, ok=False, error="tool executor returned an invalid result")
    return result


def _tool_trace_entry(name: str, result: ToolResult) -> Mapping[str, object]:
    payload = result.as_json()
    entry: dict[str, object] = {
        "name": name,
        "ok": result.ok,
        "result_characters": len(payload),
    }
    if not result.ok:
        entry["error"] = (result.error or "tool failed")[:500]
    return entry


def _tool_event_request_details(round_number: int, arguments: Mapping[str, object]) -> Mapping[str, object]:
    argument_keys = sorted(str(key) for key in arguments.keys())[:20]
    return {"round": round_number, "argument_keys": argument_keys}


def _tool_event_result_details(round_number: int, result: ToolResult) -> Mapping[str, object]:
    payload = result.as_json()
    details: dict[str, object] = {
        "round": round_number,
        "ok": result.ok,
        "result_characters": len(payload),
    }
    if not result.ok:
        details["error"] = (result.error or "tool failed")[:500]
    return details


def _tool_result_event_name(result: ToolResult) -> str:
    if result.ok:
        return "completed"
    if (result.error or "").lower() == "approval denied":
        return "denied"
    return "failed"


def _extract_assistant_message(response: Mapping[str, object]) -> tuple[str | None, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        return None, None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None, None
    message = first.get("message")
    if not isinstance(message, Mapping):
        return None, None

    thinking = _extract_text_value(
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("thinking")
        or first.get("reasoning_content")
    )
    content = _extract_text_value(message.get("content"))
    if content is None:
        return None, thinking
    content, inline_thinking = _split_inline_thinking(content)
    if thinking is None:
        thinking = inline_thinking
    return content, thinking


def _extract_text_value(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        normalized = "".join(parts).strip()
        return normalized or None
    return None


def _extract_stream_text_value(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        value = "".join(parts)
        return value or None
    return None


def _split_inline_thinking(content: str) -> tuple[str, str | None]:
    opening = content.find("<think>")
    if opening < 0:
        return content, None
    closing = content.find("</think>", opening + len("<think>"))
    if closing < 0:
        return content, None
    thinking = content[opening + len("<think>") : closing].strip()
    response = (content[:opening] + content[closing + len("</think>") :]).strip()
    return response or None, thinking or None


def _iter_sse_json(response: Any) -> Iterator[Mapping[str, object]]:
    data_lines: list[str] = []
    for raw_line in response:
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise OpenAICompatibleClientError(
                    "malformed_response",
                    "OpenAI-compatible stream returned non-UTF-8 data.",
                ) from exc
        elif isinstance(raw_line, str):
            line = raw_line
        else:
            raise OpenAICompatibleClientError(
                "malformed_stream",
                "OpenAI-compatible stream returned an invalid line.",
            )
        line = line.rstrip("\r\n")
        if not line:
            payload, done = _decode_sse_event(data_lines)
            data_lines = []
            if done:
                return
            if payload is not None:
                yield payload
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))

    payload, done = _decode_sse_event(data_lines)
    if not done and payload is not None:
        yield payload


def _decode_sse_event(data_lines: Sequence[str]) -> tuple[Mapping[str, object] | None, bool]:
    if not data_lines:
        return None, False
    raw_payload = "\n".join(data_lines).strip()
    if raw_payload == "[DONE]":
        return None, True
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise OpenAICompatibleClientError(
            "malformed_stream",
            "OpenAI-compatible endpoint returned malformed SSE JSON.",
        ) from exc
    if not isinstance(payload, Mapping):
        raise OpenAICompatibleClientError(
            "malformed_stream",
            "OpenAI-compatible stream events must be JSON objects.",
        )
    return dict(payload), False


class _StreamingAssistantAccumulator:
    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"

    def __init__(self, callback: Callable[[str, str], None]) -> None:
        self._callback = callback
        self._pending = ""
        self._inline_state = "undecided"
        self._response_parts: list[str] = []
        self._thinking_parts: list[str] = []

    @property
    def response(self) -> str | None:
        value = "".join(self._response_parts).strip()
        return value or None

    @property
    def thinking(self) -> str | None:
        value = "".join(self._thinking_parts).strip()
        return value or None

    def feed_reasoning(self, value: str | None) -> None:
        if value:
            self._emit_thinking(value)

    def feed_content(self, value: str | None) -> None:
        if value:
            self._pending += value
            self._drain()

    def finish(self) -> None:
        if self._pending:
            if self._inline_state == "thinking":
                self._emit_thinking(self._pending)
            else:
                self._emit_response(self._pending)
            self._pending = ""

    def _drain(self) -> None:
        while self._pending:
            if self._inline_state == "response":
                self._emit_response(self._pending)
                self._pending = ""
                return
            if self._inline_state == "undecided":
                opening = self._pending.find(self._OPEN_TAG)
                if opening >= 0:
                    if opening:
                        self._emit_response(self._pending[:opening])
                    self._pending = self._pending[opening + len(self._OPEN_TAG) :]
                    self._inline_state = "thinking"
                    continue
                safe_length = max(0, len(self._pending) - len(self._OPEN_TAG) + 1)
                if safe_length:
                    self._emit_response(self._pending[:safe_length])
                    self._pending = self._pending[safe_length:]
                return
            closing = self._pending.find(self._CLOSE_TAG)
            if closing >= 0:
                if closing:
                    self._emit_thinking(self._pending[:closing])
                self._pending = self._pending[closing + len(self._CLOSE_TAG) :]
                self._inline_state = "response"
                continue
            safe_length = max(0, len(self._pending) - len(self._CLOSE_TAG) + 1)
            if safe_length:
                self._emit_thinking(self._pending[:safe_length])
                self._pending = self._pending[safe_length:]
            return

    def _emit_response(self, value: str) -> None:
        self._response_parts.append(value)
        self._callback("content", value)

    def _emit_thinking(self, value: str) -> None:
        self._thinking_parts.append(value)
        self._callback("thinking", value)


def _normalize_usage(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    return dict(value)


def _merge_usage(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
) -> Mapping[str, object]:
    """Aggregate per-request usage from a multi-turn tool exchange."""

    if previous is None:
        return dict(current)
    merged = dict(previous)
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        old_value = previous.get(field)
        new_value = current.get(field)
        if isinstance(old_value, int) and not isinstance(old_value, bool) and isinstance(new_value, int) and not isinstance(new_value, bool):
            merged[field] = old_value + new_value
        elif field in current:
            merged[field] = new_value
    for key, value in current.items():
        if key not in {"prompt_tokens", "completion_tokens", "total_tokens"}:
            merged[key] = value
    return merged


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


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    """Preserve a short provider diagnostic without echoing the request."""

    detail = ""
    try:
        raw_body = exc.read(4096)
        decoded = raw_body.decode("utf-8", errors="replace")
        payload = json.loads(decoded)
        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, Mapping):
                detail = _extract_text_value(error.get("message") or error.get("detail")) or ""
            elif isinstance(error, str):
                detail = error.strip()
            if not detail:
                detail = _extract_text_value(payload.get("message") or payload.get("detail")) or ""
    except (OSError, UnicodeError, json.JSONDecodeError):
        detail = ""
    detail = " ".join(detail.split())[:400]
    if any(
        marker in detail.casefold()
        for marker in ("secret", "token", "api_key", "authorization", "password", "credential")
    ):
        detail = ""
    if detail:
        return f"OpenAI-compatible endpoint returned HTTP {exc.code}: {detail}."
    return f"OpenAI-compatible endpoint returned HTTP {exc.code}."


__all__ = [
    "DEFAULT_PROVIDER_ID",
    "OpenAICompatibleAdapter",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleClientError",
    "RequestCancellation",
]
