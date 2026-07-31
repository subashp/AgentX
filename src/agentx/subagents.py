"""Bounded two-level subagent orchestration for AgentX tool callers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .tools import ToolError, ToolExecutor, ToolResult, ToolSpec
from .workspace import WorkspaceError, normalize_scoped_path


MAX_SUBAGENTS = 10
MAX_SUBAGENT_DEPTH = 1


class SubagentError(ValueError):
    """Raised when a subagent request violates the manager contract."""


@dataclass(frozen=True)
class SubagentTask:
    prompt: str
    context_paths: tuple[str, ...] = ()
    provider: str = "auto"
    model_tier: str | None = None
    mode: str = "plan"
    task_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise SubagentError("subagent prompt must be non-empty")
        object.__setattr__(self, "prompt", self.prompt.strip())
        object.__setattr__(self, "provider", _normalized_label(self.provider, "provider"))
        object.__setattr__(self, "mode", _normalized_label(self.mode, "mode"))
        if self.model_tier is not None:
            object.__setattr__(self, "model_tier", _normalized_label(self.model_tier, "model_tier"))
        object.__setattr__(self, "context_paths", _normalized_paths(self.context_paths))
        object.__setattr__(self, "task_hints", _normalized_strings(self.task_hints, "task_hints"))

    def as_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "context_paths": list(self.context_paths),
            "provider": self.provider,
            "model_tier": self.model_tier,
            "mode": self.mode,
            "task_hints": list(self.task_hints),
        }


@dataclass(frozen=True)
class SubagentRecord:
    id: str
    session_id: str
    task: SubagentTask
    status: str
    summary: str = ""
    result: Mapping[str, object] = field(default_factory=dict)
    artifact_root: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "session_id": self.session_id,
            "task": self.task.as_dict(),
            "status": self.status,
            "summary": self.summary,
            "result": dict(self.result),
            "artifact_root": self.artifact_root,
        }
        if self.error:
            payload["error"] = self.error
        return payload


class SubagentRunner(Protocol):
    """Run one complete child task and return a sanitized result mapping."""

    def run(
        self,
        task: SubagentTask,
        *,
        session_id: str,
        depth: int,
    ) -> Mapping[str, object]:
        """Execute the task with an isolated provider interaction."""


class SubagentManager:
    """Own child lifecycle, count limits, and the two-level depth boundary."""

    def __init__(
        self,
        *,
        parent_session_id: str,
        runner: SubagentRunner,
        depth: int = 0,
        max_subagents: int = MAX_SUBAGENTS,
    ) -> None:
        if not isinstance(parent_session_id, str) or not parent_session_id.strip():
            raise SubagentError("parent_session_id must be non-empty")
        if not callable(getattr(runner, "run", None)):
            raise SubagentError("runner must provide run()")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth not in {0, 1}:
            raise SubagentError("depth must be 0 or 1")
        if isinstance(max_subagents, bool) or not isinstance(max_subagents, int) or not 1 <= max_subagents <= MAX_SUBAGENTS:
            raise SubagentError(f"max_subagents must be an integer from 1 to {MAX_SUBAGENTS}")
        self.parent_session_id = parent_session_id.strip()
        self.runner = runner
        self.depth = depth
        self.max_subagents = max_subagents
        self._records: dict[str, SubagentRecord] = {}
        self._next_number = 1

    @property
    def can_spawn(self) -> bool:
        return self.depth < MAX_SUBAGENT_DEPTH and len(self._records) < self.max_subagents

    def spawn(self, task: SubagentTask | Mapping[str, object]) -> SubagentRecord:
        if self.depth >= MAX_SUBAGENT_DEPTH:
            raise SubagentError("subagents cannot create further subagents")
        if len(self._records) >= self.max_subagents:
            raise SubagentError(f"subagent limit reached ({self.max_subagents})")
        normalized_task = task if isinstance(task, SubagentTask) else _task_from_mapping(task)
        child_id = f"subagent-{self._next_number:02d}"
        self._next_number += 1
        session_id = f"{self.parent_session_id}-{child_id}"
        running = SubagentRecord(
            id=child_id,
            session_id=session_id,
            task=normalized_task,
            status="running",
        )
        self._records[child_id] = running
        try:
            raw_result = self.runner.run(
                normalized_task,
                session_id=session_id,
                depth=self.depth + 1,
            )
            if not isinstance(raw_result, Mapping):
                raise SubagentError("subagent runner must return a mapping")
            result = dict(raw_result)
            summary = _result_summary(result)
            completed = SubagentRecord(
                id=child_id,
                session_id=session_id,
                task=normalized_task,
                status="completed",
                summary=summary,
                result=result,
                artifact_root=_optional_string(result.get("artifact_root")),
            )
        except Exception as exc:
            completed = SubagentRecord(
                id=child_id,
                session_id=session_id,
                task=normalized_task,
                status="failed",
                summary="",
                result={},
                error=f"{type(exc).__name__}: {exc}",
            )
        self._records[child_id] = completed
        return completed

    def list(self) -> tuple[SubagentRecord, ...]:
        return tuple(self._records.values())

    def get(self, subagent_id: str) -> SubagentRecord:
        normalized = _normalized_label(subagent_id, "subagent_id")
        try:
            return self._records[normalized]
        except KeyError as exc:
            raise SubagentError(f"unknown subagent '{normalized}'") from exc


SUBAGENT_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "subagent_create",
        "Create and run one isolated child agent. Child agents cannot create children.",
        {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string", "minLength": 1},
                "context_paths": {"type": "array", "items": {"type": "string"}, "maxItems": 64},
                "provider": {"type": "string"},
                "model_tier": {"type": "string"},
                "mode": {"type": "string"},
                "task_hints": {"type": "array", "items": {"type": "string"}, "maxItems": 32},
            },
        },
    ),
    ToolSpec(
        "subagent_list",
        "List child agents created by this parent and their statuses.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "subagent_get",
        "Get the summary and result for one child agent by ID.",
        {
            "type": "object",
            "required": ["subagent_id"],
            "properties": {"subagent_id": {"type": "string"}},
        },
    ),
)


class SubagentTools:
    """Expose bounded child management as model-callable tools."""

    def __init__(self, manager: SubagentManager) -> None:
        if not isinstance(manager, SubagentManager):
            raise ToolError("manager must be a SubagentManager")
        self.manager = manager

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        if self.manager.depth >= MAX_SUBAGENT_DEPTH:
            return ()
        return SUBAGENT_TOOL_SPECS

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        if name not in {spec.name for spec in self.specs}:
            return ToolResult(name=name, ok=False, error="subagent tool is unavailable at this depth")
        args = dict(arguments or {})
        try:
            if name == "subagent_create":
                return ToolResult(name=name, ok=True, output=self.manager.spawn(_task_from_mapping(args)).as_dict())
            if name == "subagent_list":
                return ToolResult(name=name, ok=True, output={"subagents": [record.as_dict() for record in self.manager.list()]})
            record = self.manager.get(args.get("subagent_id"))
            return ToolResult(name=name, ok=True, output=record.as_dict())
        except (SubagentError, TypeError, ValueError) as exc:
            return ToolResult(name=name, ok=False, error=str(exc))


def _task_from_mapping(value: Mapping[str, object]) -> SubagentTask:
    if not isinstance(value, Mapping):
        raise SubagentError("subagent task must be a mapping")
    return SubagentTask(
        prompt=value.get("prompt"),
        context_paths=value.get("context_paths", ()),
        provider=value.get("provider", "auto"),
        model_tier=value.get("model_tier"),
        mode=value.get("mode", "plan"),
        task_hints=value.get("task_hints", ()),
    )


def _normalized_paths(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SubagentError("context_paths must be a sequence of relative paths")
    paths: list[str] = []
    for item in value:
        try:
            normalized = normalize_scoped_path(item, "context_path")
        except WorkspaceError as exc:
            raise SubagentError(str(exc)) from exc
        if normalized not in paths:
            paths.append(normalized)
    return tuple(paths)


def _normalized_strings(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SubagentError(f"{field_name} must be a sequence")
    values: list[str] = []
    for item in value:
        values.append(_normalized_label(item, field_name))
    return tuple(values)


def _normalized_label(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubagentError(f"{field_name} must be a non-empty string")
    return value.strip()


def _result_summary(result: Mapping[str, object]) -> str:
    for key in ("summary", "response", "outcome"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = value.get("summary")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return "Subagent completed without a textual summary."


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


__all__ = [
    "MAX_SUBAGENTS",
    "MAX_SUBAGENT_DEPTH",
    "SUBAGENT_TOOL_SPECS",
    "SubagentError",
    "SubagentManager",
    "SubagentRecord",
    "SubagentRunner",
    "SubagentTask",
    "SubagentTools",
]
