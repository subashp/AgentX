from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


RUNTIME_MODE_CONNECT = "connect"
RUNTIME_MODE_LAUNCH = "launch"

SHUTDOWN_NEVER = "never"
SHUTDOWN_ALWAYS = "always"

VALID_RUNTIME_MODES = frozenset({RUNTIME_MODE_CONNECT, RUNTIME_MODE_LAUNCH})
VALID_SHUTDOWN_POLICIES = frozenset({SHUTDOWN_NEVER, SHUTDOWN_ALWAYS})


class LifecycleError(ValueError):
    """Raised when private runtime lifecycle inputs are invalid."""


class LaunchedRuntimeHandle(Protocol):
    """Handle returned by a private runtime launcher."""

    runtime_id: str
    endpoint: str
    pid: int | None

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop the launched runtime."""


class RuntimeLauncher(Protocol):
    """Platform-neutral launch boundary for private model runtimes."""

    def launch(self, config: "RuntimeConfig") -> LaunchedRuntimeHandle:
        """Start a runtime and return a handle without assuming shell syntax."""


class HealthCheck(Protocol):
    """Runtime health check boundary."""

    def __call__(self, config: "RuntimeConfig") -> "HealthCheckResult":
        """Return the current health status for config.endpoint."""


@dataclass(frozen=True)
class RuntimeConfig:
    runtime_id: str
    endpoint: str
    mode: str = RUNTIME_MODE_CONNECT
    provider_id: str = "private-openai-compatible"
    command: str | None = None
    args: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | Path | None = None
    auth_fields: Mapping[str, str] = field(default_factory=dict)
    health_timeout_seconds: float = 30.0
    health_interval_seconds: float = 1.0
    shutdown_policy: str = SHUTDOWN_NEVER
    shutdown_timeout_seconds: float | None = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_id", _non_empty_string(self.runtime_id, "runtime_id"))
        object.__setattr__(self, "provider_id", _non_empty_string(self.provider_id, "provider_id"))
        object.__setattr__(self, "endpoint", _absolute_http_url(self.endpoint, "endpoint"))
        object.__setattr__(self, "mode", _choice(self.mode, "mode", VALID_RUNTIME_MODES))
        if self.command is not None:
            object.__setattr__(self, "command", _non_empty_string(self.command, "command"))
        object.__setattr__(self, "args", _string_tuple(self.args, "args"))
        object.__setattr__(self, "env", _string_mapping(self.env, "env"))
        object.__setattr__(self, "auth_fields", _string_mapping(self.auth_fields, "auth_fields"))
        object.__setattr__(
            self,
            "health_timeout_seconds",
            _non_negative_float(self.health_timeout_seconds, "health_timeout_seconds"),
        )
        object.__setattr__(
            self,
            "health_interval_seconds",
            _positive_float(self.health_interval_seconds, "health_interval_seconds"),
        )
        object.__setattr__(
            self,
            "shutdown_policy",
            _choice(self.shutdown_policy, "shutdown_policy", VALID_SHUTDOWN_POLICIES),
        )
        if self.shutdown_timeout_seconds is not None:
            object.__setattr__(
                self,
                "shutdown_timeout_seconds",
                _non_negative_float(self.shutdown_timeout_seconds, "shutdown_timeout_seconds"),
            )
        if self.mode == RUNTIME_MODE_LAUNCH and self.command is None:
            raise LifecycleError("command is required when mode is launch.")

    @property
    def argv(self) -> tuple[str, ...]:
        if self.command is None:
            return ()
        return (self.command,) + tuple(self.args)

    def safe_summary(self) -> dict[str, object]:
        return {
            "runtime_id": self.runtime_id,
            "provider_id": self.provider_id,
            "endpoint": self.endpoint,
            "mode": self.mode,
            "command_configured": self.command is not None,
            "arg_count": len(self.args),
            "cwd_configured": self.cwd is not None,
            "env_keys": sorted(self.env),
            "auth_configured": bool(self.auth_fields),
            "auth_field_count": len(self.auth_fields),
            "health_timeout_seconds": self.health_timeout_seconds,
            "health_interval_seconds": self.health_interval_seconds,
            "shutdown_policy": self.shutdown_policy,
        }


@dataclass(frozen=True)
class HealthCheckResult:
    healthy: bool
    status: str = "healthy"
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.healthy, bool):
            raise LifecycleError("healthy must be a boolean.")
        object.__setattr__(self, "status", _non_empty_string(self.status, "status"))
        if self.detail is not None:
            object.__setattr__(self, "detail", _non_empty_string(self.detail, "detail"))


@dataclass(frozen=True)
class RuntimeLifecycleResult:
    runtime_id: str
    provider_id: str
    endpoint: str
    mode: str
    status: str
    launched: bool
    shutdown_policy: str
    shutdown_timeout_seconds: float | None
    handle: LaunchedRuntimeHandle | None
    events: tuple[Mapping[str, object], ...]

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def as_dict(self) -> dict[str, object]:
        return {
            "runtime_id": self.runtime_id,
            "provider_id": self.provider_id,
            "endpoint": self.endpoint,
            "mode": self.mode,
            "status": self.status,
            "launched": self.launched,
            "shutdown_policy": self.shutdown_policy,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "pid": self.handle.pid if self.handle is not None else None,
            "events": [dict(event) for event in self.events],
        }


@dataclass(frozen=True)
class RuntimeShutdownResult:
    runtime_id: str
    provider_id: str
    endpoint: str
    mode: str
    status: str
    attempted: bool
    events: tuple[Mapping[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "runtime_id": self.runtime_id,
            "provider_id": self.provider_id,
            "endpoint": self.endpoint,
            "mode": self.mode,
            "status": self.status,
            "attempted": self.attempted,
            "events": [dict(event) for event in self.events],
        }


@dataclass
class SubprocessRuntimeHandle:
    runtime_id: str
    endpoint: str
    process: subprocess.Popen[Any]

    @property
    def pid(self) -> int | None:
        return self.process.pid

    def shutdown(self, timeout: float | None = None) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)


@dataclass(frozen=True)
class SubprocessRuntimeLauncher:
    """Stdlib launcher that starts argv without a shell."""

    inherit_environment: bool = True

    def launch(self, config: RuntimeConfig) -> SubprocessRuntimeHandle:
        if config.command is None:
            raise LifecycleError("command is required to launch a runtime.")
        env = dict(os.environ) if self.inherit_environment else {}
        env.update(config.env)
        try:
            process = subprocess.Popen(
                list(config.argv),
                cwd=None if config.cwd is None else str(config.cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError(f"runtime launch failed: {type(exc).__name__}") from exc
        return SubprocessRuntimeHandle(
            runtime_id=config.runtime_id,
            endpoint=config.endpoint,
            process=process,
        )


class RuntimeLifecycleManager:
    """Acquire, wait for, and release private OpenAI-compatible runtimes."""

    def __init__(
        self,
        *,
        launcher: RuntimeLauncher | None = None,
        health_check: HealthCheck | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._launcher = launcher or SubprocessRuntimeLauncher()
        self._health_check = health_check or openai_compatible_health_check
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep

    def acquire(self, config: RuntimeConfig) -> RuntimeLifecycleResult:
        if not isinstance(config, RuntimeConfig):
            raise LifecycleError("config must be a RuntimeConfig.")

        events: list[Mapping[str, object]] = [
            _event(1, "runtime_lifecycle_started", "started", config)
        ]
        handle: LaunchedRuntimeHandle | None = None
        launched = False

        if config.mode == RUNTIME_MODE_LAUNCH:
            events.append(_event(2, "runtime_launch_started", "started", config))
            try:
                handle = self._launcher.launch(config)
            except Exception as exc:
                events.append(
                    _event(
                        3,
                        "runtime_launch_failed",
                        "failure",
                        config,
                        error_type=type(exc).__name__,
                    )
                )
                return _lifecycle_result(config, "launch_failed", False, None, events)
            launched = True
            events.append(
                _event(
                    3,
                    "runtime_launch_completed",
                    "success",
                    config,
                    pid=handle.pid,
                )
            )
        else:
            events.append(_event(2, "runtime_connect_selected", "success", config))

        status = self._wait_for_health(config, events)
        return _lifecycle_result(config, status, launched, handle, events)

    def shutdown(
        self,
        result: RuntimeLifecycleResult,
        *,
        policy: str | None = None,
        reason: str = "manager_shutdown",
    ) -> RuntimeShutdownResult:
        if not isinstance(result, RuntimeLifecycleResult):
            raise LifecycleError("result must be a RuntimeLifecycleResult.")
        shutdown_policy = result.shutdown_policy if policy is None else _choice(
            policy,
            "policy",
            VALID_SHUTDOWN_POLICIES,
        )
        reason_code = reason if reason in {"manager_shutdown", "runtime_not_needed"} else "custom"
        event_config = _ResultEventConfig(result)
        sequence = len(result.events) + 1
        events: list[Mapping[str, object]] = []

        if shutdown_policy == SHUTDOWN_NEVER:
            events.append(
                _event(
                    sequence,
                    "runtime_shutdown_skipped",
                    "skipped",
                    event_config,
                    reason=reason_code,
                    policy=shutdown_policy,
                )
            )
            return _shutdown_result(result, "skipped", False, events)

        if not result.launched or result.handle is None:
            events.append(
                _event(
                    sequence,
                    "runtime_shutdown_skipped",
                    "skipped",
                    event_config,
                    reason="not_launched_by_manager",
                    policy=shutdown_policy,
                )
            )
            return _shutdown_result(result, "skipped", False, events)

        events.append(
            _event(
                sequence,
                "runtime_shutdown_started",
                "started",
                event_config,
                reason=reason_code,
                policy=shutdown_policy,
            )
        )
        try:
            result.handle.shutdown(timeout=result.shutdown_timeout_seconds)
        except Exception as exc:
            events.append(
                _event(
                    sequence + 1,
                    "runtime_shutdown_failed",
                    "failure",
                    event_config,
                    error_type=type(exc).__name__,
                    policy=shutdown_policy,
                )
            )
            return _shutdown_result(result, "shutdown_failed", True, events)

        events.append(
            _event(
                sequence + 1,
                "runtime_shutdown_completed",
                "success",
                event_config,
                policy=shutdown_policy,
            )
        )
        return _shutdown_result(result, "shutdown_complete", True, events)

    def _wait_for_health(
        self,
        config: RuntimeConfig,
        events: list[Mapping[str, object]],
    ) -> str:
        started_at = self._clock()
        deadline = started_at + config.health_timeout_seconds
        attempt = 0

        while True:
            attempt += 1
            sequence = len(events) + 1
            try:
                health = self._health_check(config)
                if not isinstance(health, HealthCheckResult):
                    raise LifecycleError("health_check must return a HealthCheckResult.")
            except Exception as exc:
                health = HealthCheckResult(False, "health_check_error")
                events.append(
                    _event(
                        sequence,
                        "runtime_health_check_error",
                        "failure",
                        config,
                        attempt=attempt,
                        error_type=type(exc).__name__,
                    )
                )
            else:
                if health.healthy:
                    events.append(
                        _event(
                            sequence,
                            "runtime_health_check_passed",
                            "success",
                            config,
                            attempt=attempt,
                            health_status=health.status,
                        )
                    )
                    events.append(
                        _event(
                            sequence + 1,
                            "runtime_lifecycle_ready",
                            "ready",
                            config,
                            attempt=attempt,
                        )
                    )
                    return "ready"
                events.append(
                    _event(
                        sequence,
                        "runtime_health_check_failed",
                        "retrying",
                        config,
                        attempt=attempt,
                        health_status=health.status,
                    )
                )

            now = self._clock()
            if now >= deadline:
                events.append(
                    _event(
                        len(events) + 1,
                        "runtime_health_timeout",
                        "timeout",
                        config,
                        attempts=attempt,
                    )
                )
                return "health_timeout"

            self._sleep(min(config.health_interval_seconds, max(0.0, deadline - now)))


def openai_compatible_health_check(config: RuntimeConfig) -> HealthCheckResult:
    url = _join_url(config.endpoint, "/v1/models")
    headers = {"Accept": "application/json", "User-Agent": "agentx-lifecycle/0.1"}
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            status_code = getattr(response, "status", None) or response.getcode()
    except urllib.error.HTTPError as exc:
        return HealthCheckResult(False, f"http_{exc.code}")
    except TimeoutError:
        return HealthCheckResult(False, "timeout")
    except OSError:
        return HealthCheckResult(False, "connection_error")
    return HealthCheckResult(200 <= int(status_code) < 300, f"http_{status_code}")


@dataclass(frozen=True)
class _ResultEventConfig:
    result: RuntimeLifecycleResult

    @property
    def runtime_id(self) -> str:
        return self.result.runtime_id

    @property
    def provider_id(self) -> str:
        return self.result.provider_id

    @property
    def endpoint(self) -> str:
        return self.result.endpoint

    @property
    def mode(self) -> str:
        return self.result.mode

    @property
    def shutdown_policy(self) -> str:
        return SHUTDOWN_NEVER

    def safe_summary(self) -> dict[str, object]:
        return {
            "runtime_id": self.runtime_id,
            "provider_id": self.provider_id,
            "endpoint": self.endpoint,
            "mode": self.mode,
        }


def _lifecycle_result(
    config: RuntimeConfig,
    status: str,
    launched: bool,
    handle: LaunchedRuntimeHandle | None,
    events: Sequence[Mapping[str, object]],
) -> RuntimeLifecycleResult:
    return RuntimeLifecycleResult(
        runtime_id=config.runtime_id,
        provider_id=config.provider_id,
        endpoint=config.endpoint,
        mode=config.mode,
        status=status,
        launched=launched,
        shutdown_policy=config.shutdown_policy,
        shutdown_timeout_seconds=config.shutdown_timeout_seconds,
        handle=handle,
        events=tuple(dict(event) for event in events),
    )


def _shutdown_result(
    result: RuntimeLifecycleResult,
    status: str,
    attempted: bool,
    events: Sequence[Mapping[str, object]],
) -> RuntimeShutdownResult:
    return RuntimeShutdownResult(
        runtime_id=result.runtime_id,
        provider_id=result.provider_id,
        endpoint=result.endpoint,
        mode=result.mode,
        status=status,
        attempted=attempted,
        events=tuple(dict(event) for event in events),
    )


def _event(
    sequence: int,
    name: str,
    status: str,
    config: RuntimeConfig | _ResultEventConfig,
    **details: object,
) -> dict[str, object]:
    event = {
        "sequence": sequence,
        "event": name,
        "status": status,
        "runtime_id": config.runtime_id,
        "provider_id": config.provider_id,
        "endpoint": config.endpoint,
        "mode": config.mode,
    }
    safe_details = {
        key: value
        for key, value in details.items()
        if value is not None
    }
    if safe_details:
        event.update(safe_details)
    if isinstance(config, RuntimeConfig):
        event.update(
            {
                "command_configured": config.command is not None,
                "arg_count": len(config.args),
                "cwd_configured": config.cwd is not None,
                "env_keys": sorted(config.env),
                "auth_configured": bool(config.auth_fields),
                "auth_field_count": len(config.auth_fields),
                "shutdown_policy": config.shutdown_policy,
            }
        )
    return event


def _join_url(base_url: str, suffix: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1") and suffix.startswith("/v1/"):
        suffix = suffix[3:]
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path + suffix, "", "")
    )


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise LifecycleError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise LifecycleError(f"{field_name} cannot be empty.")
    return normalized


def _absolute_http_url(value: object, field_name: str) -> str:
    normalized = _non_empty_string(value, field_name).rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LifecycleError(f"{field_name} must be an absolute http or https URL.")
    if parsed.username is not None or parsed.password is not None:
        raise LifecycleError(f"{field_name} must not include credentials.")
    if parsed.query or parsed.fragment:
        raise LifecycleError(f"{field_name} must not include a query string or fragment.")
    return normalized


def _choice(value: object, field_name: str, valid: frozenset[str]) -> str:
    normalized = _non_empty_string(value, field_name)
    if normalized not in valid:
        choices = ", ".join(sorted(valid))
        raise LifecycleError(f"{field_name} must be one of: {choices}.")
    return normalized


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise LifecycleError(f"{field_name} must be a sequence of strings.")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_non_empty_string(item, f"{field_name}[{index}]"))
    return tuple(result)


def _string_mapping(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise LifecycleError(f"{field_name} must be a mapping of strings.")
    result: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = _non_empty_string(key, f"{field_name} key")
        result[normalized_key] = _non_empty_string(item, f"{field_name}.{normalized_key}")
    return result


def _non_negative_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LifecycleError(f"{field_name} must be a number.")
    normalized = float(value)
    if normalized < 0:
        raise LifecycleError(f"{field_name} cannot be negative.")
    return normalized


def _positive_float(value: object, field_name: str) -> float:
    normalized = _non_negative_float(value, field_name)
    if normalized == 0:
        raise LifecycleError(f"{field_name} must be greater than zero.")
    return normalized
