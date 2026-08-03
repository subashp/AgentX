"""Bounded autonomous execution helpers for private-provider coding runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic

from .adapters import AdapterRequest, AdapterResult, ProviderAdapter
from .tools import ToolExecutor, ToolResult, ToolSpec


class AutonomousWorkError(ValueError):
    """Raised when autonomous work controller inputs are invalid."""


@dataclass(frozen=True)
class AutonomousLimits:
    max_iterations: int = 6
    max_shell_calls: int = 8
    max_patch_attempts: int = 4
    max_wall_time_seconds: int = 1_800

    def __post_init__(self) -> None:
        _validate_limit(self.max_iterations, "max_iterations", 1, 32)
        _validate_limit(self.max_shell_calls, "max_shell_calls", 0, 64)
        _validate_limit(self.max_patch_attempts, "max_patch_attempts", 0, 32)
        _validate_limit(self.max_wall_time_seconds, "max_wall_time_seconds", 1, 86_400)

    def as_dict(self) -> dict[str, int]:
        return {
            "max_iterations": self.max_iterations,
            "max_shell_calls": self.max_shell_calls,
            "max_patch_attempts": self.max_patch_attempts,
            "max_wall_time_seconds": self.max_wall_time_seconds,
        }


class AutonomousWorkController:
    """Track and bound a model/tool execute-mode coding run."""

    def __init__(self, limits: AutonomousLimits | None = None) -> None:
        self.limits = limits or AutonomousLimits()
        self.started_at = monotonic()
        self.iteration_count = 0
        self.tool_call_count = 0
        self.shell_call_count = 0
        self.patch_attempt_count = 0
        self.changed_paths: list[str] = []
        self.last_test: dict[str, object] | None = None
        self.stop_reason: str | None = None

    def wrap_tool_executor(self, executor: ToolExecutor) -> ToolExecutor:
        return _AutonomousToolExecutor(executor, self)

    def wrap_adapter(self, adapter: ProviderAdapter) -> ProviderAdapter:
        return _AutonomousAdapter(adapter, self)

    def tool_event_callback(
        self,
        delegate: Callable[[Mapping[str, object]], None] | None = None,
    ) -> Callable[[Mapping[str, object]], None]:
        if delegate is not None and not callable(delegate):
            raise AutonomousWorkError("delegate must be callable or None")

        def record(event: Mapping[str, object]) -> None:
            round_number = event.get("round")
            if isinstance(round_number, int) and round_number > self.iteration_count:
                self.iteration_count = round_number
            if delegate is not None:
                delegate(event)

        return record

    def before_tool_call(self, name: str) -> ToolResult | None:
        self.tool_call_count += 1
        elapsed = monotonic() - self.started_at
        if elapsed > self.limits.max_wall_time_seconds:
            return self._limit_result(name, "max_wall_time_seconds")
        if self.iteration_count > self.limits.max_iterations:
            return self._limit_result(name, "max_iterations")
        if _is_shell_like_tool(name):
            self.shell_call_count += 1
            if self.shell_call_count > self.limits.max_shell_calls:
                return self._limit_result(name, "max_shell_calls")
        if _is_patch_tool(name):
            self.patch_attempt_count += 1
            if self.patch_attempt_count > self.limits.max_patch_attempts:
                return self._limit_result(name, "max_patch_attempts")
        return None

    def after_tool_call(self, name: str, result: ToolResult) -> None:
        if not result.ok and (result.error or "").lower() == "approval denied":
            self.stop_reason = self.stop_reason or "approval_denied"
        if _is_patch_tool(name) and result.ok and isinstance(result.output, Mapping):
            paths = result.output.get("paths")
            if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
                for path in paths:
                    if isinstance(path, str) and path not in self.changed_paths:
                        self.changed_paths.append(path)
        if _is_test_tool(name):
            self.last_test = _test_summary(result)

    def autonomous_summary(self, *, provider_status: str) -> dict[str, object]:
        stop_reason = self.stop_reason
        if stop_reason is None and self.last_test is not None:
            stop_reason = "validation_passed" if self.last_test.get("ok") is True else "validation_failed"
        if stop_reason is None:
            stop_reason = "completed" if provider_status == "success" else "provider_failed"
        return {
            "limits": self.limits.as_dict(),
            "iterations": self.iteration_count,
            "tool_calls": self.tool_call_count,
            "shell_calls": self.shell_call_count,
            "patch_attempts": self.patch_attempt_count,
            "changed_paths": list(self.changed_paths),
            "last_test": dict(self.last_test) if self.last_test is not None else None,
            "stop_reason": stop_reason,
        }

    def _limit_result(self, name: str, limit_name: str) -> ToolResult:
        self.stop_reason = f"limit_exceeded:{limit_name}"
        return ToolResult(
            name=name,
            ok=False,
            error=f"autonomous limit exceeded: {limit_name}",
            output={"stop_reason": self.stop_reason, "limits": self.limits.as_dict()},
        )


class _AutonomousToolExecutor:
    def __init__(self, inner: ToolExecutor, controller: AutonomousWorkController) -> None:
        self.inner = inner
        self.controller = controller

    @property
    def specs(self) -> Sequence[ToolSpec]:
        return self.inner.specs

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        limit_result = self.controller.before_tool_call(name)
        if limit_result is not None:
            return limit_result
        result = self.inner.call(name, arguments)
        self.controller.after_tool_call(name, result)
        return result


class _AutonomousAdapter:
    def __init__(self, inner: ProviderAdapter, controller: AutonomousWorkController) -> None:
        self.inner = inner
        self.controller = controller

    @property
    def provider_id(self) -> str:
        return self.inner.provider_id

    def execute(self, request: AdapterRequest) -> AdapterResult:
        result = self.inner.execute(request)
        outcome = dict(result.outcome)
        outcome["autonomous"] = self.controller.autonomous_summary(provider_status=result.status)
        return AdapterResult(
            provider_id=result.provider_id,
            model_id=result.model_id,
            model_tier=result.model_tier,
            status=result.status,
            transcript_events=result.transcript_events,
            cost=result.cost,
            outcome=outcome,
            patch=result.patch,
        )


def _validate_limit(value: int, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise AutonomousWorkError(f"{name} must be an integer from {minimum} to {maximum}")


def _is_shell_like_tool(name: str) -> bool:
    return name in {"shell.exec", "shell_exec", "test.run", "test_run"}


def _is_patch_tool(name: str) -> bool:
    return name in {"workspace.patch", "workspace_patch"}


def _is_test_tool(name: str) -> bool:
    return name in {"test.run", "test_run"}


def _test_summary(result: ToolResult) -> dict[str, object]:
    summary: dict[str, object] = {"ok": result.ok}
    if result.error:
        summary["error"] = result.error
    if isinstance(result.output, Mapping):
        for key in ("profile", "argv", "exit_code", "truncated"):
            if key in result.output:
                summary[key] = result.output[key]
    return summary


__all__ = [
    "AutonomousLimits",
    "AutonomousWorkController",
    "AutonomousWorkError",
]
