from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agentmemory_bridge import default_agentmemory_db_path
from .config import AgentXPaths
from .tools import ToolError, ToolExecutor, ToolResult, ToolSpec


class AgentXMemoryError(ValueError):
    """Raised when AgentX memory operations cannot be completed."""


def agentmemory_db_path(paths: AgentXPaths) -> Path:
    return default_agentmemory_db_path(paths.memories)


def agentmemory_install_hint() -> str:
    return (
        "AgentMemory is not importable. Run "
        "'git submodule update --init --recursive' and "
        "'python -m pip install -e third_party/AgentMemory'."
    )


def load_agentmemory_module() -> Any:
    try:
        import agentmemory

        return agentmemory
    except ImportError:
        candidate = Path(__file__).resolve().parents[2] / "third_party" / "AgentMemory" / "src"
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            import agentmemory

            return agentmemory
        except ImportError as exc:
            raise AgentXMemoryError(agentmemory_install_hint()) from exc


def open_memory_runtime(paths: AgentXPaths) -> tuple[Any, Any]:
    agentmemory = load_agentmemory_module()
    db_path = agentmemory_db_path(paths)
    store = agentmemory.SQLiteMemoryStore(db_path)
    return store, agentmemory.MemoryToolRuntime(store)


def call_memory_tool(paths: AgentXPaths, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    _store, runtime = open_memory_runtime(paths)
    try:
        result = runtime.call(name, arguments)
    except Exception as exc:  # AgentMemory owns exact exception classes.
        raise AgentXMemoryError(str(exc)) from exc
    if not isinstance(result, Mapping):
        raise AgentXMemoryError("AgentMemory tool returned a non-object result")
    return dict(result)


def list_memory_proposals(paths: AgentXPaths, *, status: str | None = None) -> list[dict[str, Any]]:
    store, _runtime = open_memory_runtime(paths)
    return [proposal.as_dict() for proposal in store.list_proposals(status=status)]


def apply_memory_proposal(paths: AgentXPaths, proposal_id: str) -> dict[str, Any]:
    agentmemory = load_agentmemory_module()
    store, _runtime = open_memory_runtime(paths)
    try:
        record = agentmemory.DeterministicMemoryDistiller(store).apply_proposal(proposal_id)
    except Exception as exc:
        raise AgentXMemoryError(str(exc)) from exc
    return record.as_dict()


def append_interaction_events(
    paths: AgentXPaths,
    *,
    session_id: str,
    user_prompt: str,
    assistant_summary: str,
    tool_names: Sequence[str] = (),
    provider_id: str = "private-openai-compatible",
) -> dict[str, Any]:
    agentmemory = load_agentmemory_module()
    store, _runtime = open_memory_runtime(paths)
    events = []
    if user_prompt.strip():
        events.append(
            store.append_event(
                agentmemory.MemoryEvent(
                    session_id=session_id,
                    agent_id="agentx",
                    event_type="user_prompt",
                    content=user_prompt[:12_000],
                    summary=user_prompt[:1_000],
                    source="agentx_cli",
                    actor="user",
                    privacy_class="private",
                    metadata={"provider_id": provider_id},
                )
            )
        )
    if assistant_summary.strip():
        events.append(
            store.append_event(
                agentmemory.MemoryEvent(
                    session_id=session_id,
                    agent_id="agentx",
                    event_type="assistant_response",
                    content=assistant_summary[:12_000],
                    summary=assistant_summary[:1_000],
                    source="agentx_cli",
                    actor="assistant",
                    privacy_class="private",
                    metadata={"provider_id": provider_id, "tool_names": list(tool_names)},
                )
            )
        )
    proposals = agentmemory.DeterministicMemoryDistiller(store).propose_from_events(events)
    return {
        "events": [event.as_dict() for event in events],
        "proposals": proposals.as_dict()["proposals"],
    }


class AgentMemoryTools:
    """Expose AgentMemory operations as model-callable tools with AgentX policy."""

    _MUTATING = frozenset({"memory_remember", "memory_correct", "memory_forget"})
    _READ_ONLY = frozenset({"memory_search", "memory_show", "memory_policy_explain"})

    def __init__(
        self,
        paths: AgentXPaths,
        *,
        user_prompt: str = "",
        approval_callback=None,
    ) -> None:
        self.paths = paths
        self.user_prompt = user_prompt
        self.approval_callback = approval_callback

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return (
            ToolSpec(
                "memory_remember",
                "Store an explicit user-approved memory. Use only when the user asked AgentX to remember something.",
                {
                    "type": "object",
                    "required": ["content"],
                    "properties": {
                        "content": {"type": "string", "minLength": 1},
                        "privacy_class": {"enum": ["generic", "team", "private"]},
                        "memory_kind": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                },
            ),
            ToolSpec(
                "memory_search",
                "Search AgentX memory records by query and optional privacy class.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "privacy_class": {"enum": ["generic", "team", "private"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                },
            ),
            ToolSpec(
                "memory_show",
                "Show one AgentX memory record by ID.",
                {"type": "object", "required": ["memory_id"], "properties": {"memory_id": {"type": "string"}}},
            ),
            ToolSpec(
                "memory_correct",
                "Correct an existing memory when the user explicitly says it is wrong.",
                {
                    "type": "object",
                    "required": ["memory_id", "replacement"],
                    "properties": {
                        "memory_id": {"type": "string"},
                        "replacement": {"type": "string", "minLength": 1},
                        "privacy_class": {"enum": ["generic", "team", "private"]},
                    },
                },
            ),
            ToolSpec(
                "memory_forget",
                "Forget one memory when the user explicitly asks. Deleting all memory requires direct CLI confirmation.",
                {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "all": {"type": "boolean"},
                        "hard": {"type": "boolean"},
                    },
                },
            ),
            ToolSpec(
                "memory_policy_explain",
                "Explain whether a memory may be exposed to a provider class.",
                {
                    "type": "object",
                    "required": ["memory_id", "provider_class"],
                    "properties": {
                        "memory_id": {"type": "string"},
                        "provider_class": {"enum": ["external_public", "external_team", "local_private"]},
                    },
                },
            ),
        )

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        args = dict(arguments or {})
        if name not in {spec.name for spec in self.specs}:
            return ToolResult(name=name, ok=False, error="unknown memory tool")
        try:
            if name in self._MUTATING:
                self._authorize_mutation(name, args)
            result = call_memory_tool(self.paths, name, args)
        except (AgentXMemoryError, ToolError, ValueError) as exc:
            return ToolResult(name=name, ok=False, error=str(exc))
        return ToolResult(name=name, ok=True, output=result)

    def _authorize_mutation(self, name: str, arguments: Mapping[str, object]) -> None:
        prompt = self.user_prompt.casefold()
        if name == "memory_remember" and "remember" in prompt:
            return
        if name == "memory_correct" and any(word in prompt for word in ("correct", "wrong", "fix that memory")):
            return
        if name == "memory_forget":
            if arguments.get("all"):
                raise ToolError("memory_forget all requires a direct CLI command")
            if any(word in prompt for word in ("forget", "delete", "remove")):
                return
        if self.approval_callback is not None and self.approval_callback(name, dict(arguments)):
            return
        raise ToolError(f"{name} requires explicit user intent or approval")
