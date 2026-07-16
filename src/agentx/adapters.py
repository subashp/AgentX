from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .routing import AgentRun, RouteDecision
from .store import RUN_ARTIFACT_FILES, SessionStore


class AdapterError(ValueError):
    """Raised when provider adapter execution inputs are invalid."""


class ProviderAdapter(Protocol):
    """Small execution boundary for provider-backed agent runs."""

    provider_id: str

    def execute(self, request: "AdapterRequest") -> "AdapterResult":
        """Execute the request and return structured local result data."""


@dataclass(frozen=True)
class AdapterRequest:
    run: AgentRun
    route: RouteDecision | None = None
    context_map: Mapping[str, object] | None = None
    memory_map: Mapping[str, object] | None = None
    redactions: Sequence[object] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run, AgentRun):
            raise AdapterError("request.run must be an AgentRun.")
        if self.route is not None and not isinstance(self.route, RouteDecision):
            raise AdapterError("request.route must be a RouteDecision when set.")
        object.__setattr__(
            self,
            "context_map",
            _normalize_optional_mapping(self.context_map, "context_map"),
        )
        object.__setattr__(
            self,
            "memory_map",
            _normalize_optional_mapping(self.memory_map, "memory_map"),
        )
        object.__setattr__(self, "redactions", _normalize_sequence(self.redactions, "redactions"))


@dataclass(frozen=True)
class AdapterResult:
    provider_id: str
    model_id: str | None
    model_tier: str
    status: str
    transcript_events: tuple[Mapping[str, object], ...]
    cost: Mapping[str, object]
    outcome: Mapping[str, object]
    patch: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _normalize_non_empty_string(self.provider_id, "provider_id"))
        if self.model_id is not None:
            object.__setattr__(self, "model_id", _normalize_non_empty_string(self.model_id, "model_id"))
        object.__setattr__(self, "model_tier", _normalize_non_empty_string(self.model_tier, "model_tier"))
        object.__setattr__(self, "status", _normalize_non_empty_string(self.status, "status"))
        object.__setattr__(
            self,
            "transcript_events",
            tuple(_normalize_mapping(event, "transcript_event") for event in self.transcript_events),
        )
        object.__setattr__(self, "cost", _normalize_mapping(self.cost, "cost"))
        object.__setattr__(self, "outcome", _normalize_mapping(self.outcome, "outcome"))
        if not isinstance(self.patch, str):
            raise AdapterError("patch must be a string.")

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_tier": self.model_tier,
            "status": self.status,
            "transcript_events": [dict(event) for event in self.transcript_events],
            "cost": dict(self.cost),
            "outcome": dict(self.outcome),
            "patch": self.patch,
        }


@dataclass(frozen=True)
class StoredAdapterRun:
    session_id: str
    root: Path
    artifact_paths: Mapping[str, Path]
    result: AdapterResult

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "root": str(self.root),
            "artifacts": {
                name: str(path)
                for name, path in sorted(self.artifact_paths.items())
            },
            "result": self.result.as_dict(),
        }


@dataclass(frozen=True)
class FakeLocalAdapter:
    provider_id: str = "fake-local"
    model_id: str = "fake-local-deterministic"

    def execute(self, request: AdapterRequest) -> AdapterResult:
        run = request.run
        model_tier = _selected_model_tier(run, request.route)
        model_id = _selected_model_id(self.model_id, request.route)
        transcript_events = (
            {
                "sequence": 1,
                "event": "execution_started",
                "provider_id": self.provider_id,
                "model_id": model_id,
                "model_tier": model_tier,
                "mode": run.mode,
            },
            {
                "sequence": 2,
                "event": "prompt_received",
                "input_characters": len(run.prompt),
                "context_path_count": len(run.context_paths),
            },
            {
                "sequence": 3,
                "event": "execution_completed",
                "status": "success",
                "outcome": "fake_dry_run_completed",
            },
        )
        cost = {
            "currency": "USD",
            "estimated": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost_usd": 0.0,
        }
        outcome = {
            "status": "success",
            "outcome": "fake_dry_run_completed",
            "summary": "Deterministic fake adapter completed without external execution.",
            "patch_applied": False,
        }
        return AdapterResult(
            provider_id=self.provider_id,
            model_id=model_id,
            model_tier=model_tier,
            status="success",
            transcript_events=transcript_events,
            cost=cost,
            outcome=outcome,
            patch="",
        )


def execute_adapter_run(
    *,
    session_store: SessionStore,
    session_id: str,
    run: AgentRun,
    adapter: ProviderAdapter | None = None,
    route: RouteDecision | None = None,
    context_map: Mapping[str, object] | None = None,
    memory_map: Mapping[str, object] | None = None,
    redactions: Sequence[object] = (),
) -> StoredAdapterRun:
    if not isinstance(session_store, SessionStore):
        raise AdapterError("session_store must be a SessionStore.")
    if not isinstance(session_id, str) or not session_id.strip():
        raise AdapterError("session_id must be a non-empty string.")
    if not isinstance(run, AgentRun):
        raise AdapterError("run must be an AgentRun.")

    request = AdapterRequest(
        run=run,
        route=route,
        context_map=context_map,
        memory_map=memory_map,
        redactions=redactions,
    )
    provider_adapter = adapter or FakeLocalAdapter()
    result = provider_adapter.execute(request)
    if not isinstance(result, AdapterResult):
        raise AdapterError("adapter.execute must return an AdapterResult.")
    run_store = session_store.open_run(session_id)
    resolved_context_map = _default_context_map(run) if request.context_map is None else dict(request.context_map)
    resolved_memory_map = _default_memory_map() if request.memory_map is None else dict(request.memory_map)
    resolved_redactions = list(request.redactions)
    provider = {
        "provider_id": result.provider_id,
        "model_id": result.model_id,
        "model_tier": result.model_tier,
        "status": result.status,
    }
    manifest = {
        "schema_version": 1,
        "session_id": session_id,
        "run": run.as_dict(),
        "route": route.as_dict() if route is not None else None,
        "provider": provider,
        "artifacts": list(RUN_ARTIFACT_FILES),
    }

    artifact_paths = run_store.write_artifacts(
        manifest=manifest,
        prompt=run.prompt + "\n",
        context_map=resolved_context_map,
        memory_map=resolved_memory_map,
        redactions=resolved_redactions,
        provider=provider,
        transcript=result.transcript_events,
        patch=result.patch,
        cost=result.cost,
        outcome=result.outcome,
    )
    return StoredAdapterRun(
        session_id=session_id,
        root=run_store.root,
        artifact_paths=artifact_paths,
        result=result,
    )


def execute_fake_run(
    *,
    session_store: SessionStore,
    session_id: str,
    run: AgentRun,
    route: RouteDecision | None = None,
    context_map: Mapping[str, object] | None = None,
    memory_map: Mapping[str, object] | None = None,
    redactions: Sequence[object] = (),
) -> StoredAdapterRun:
    return execute_adapter_run(
        session_store=session_store,
        session_id=session_id,
        run=run,
        adapter=FakeLocalAdapter(),
        route=route,
        context_map=context_map,
        memory_map=memory_map,
        redactions=redactions,
    )


def _default_context_map(run: AgentRun) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "agent_run.context_paths",
        "requested_paths": list(run.context_paths),
        "included_paths": list(run.context_paths),
        "excluded_paths": [],
    }


def _default_memory_map() -> dict[str, object]:
    return {
        "schema_version": 1,
        "memory_ids": [],
        "included_memories": [],
        "excluded_memories": [],
    }


def _selected_model_tier(run: AgentRun, route: RouteDecision | None) -> str:
    if route is not None:
        return route.selected_model_tier
    if run.model_tier and run.model_tier != "auto":
        return run.model_tier
    return "standard"


def _selected_model_id(default_model_id: str, route: RouteDecision | None) -> str:
    if route is not None and route.selected_model_id:
        return route.selected_model_id
    return default_model_id


def _normalize_optional_mapping(value: Mapping[str, object] | None, field_name: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _normalize_mapping(value, field_name)


def _normalize_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AdapterError(f"{field_name} must be a mapping.")
    return dict(value)


def _normalize_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AdapterError(f"{field_name} must be a sequence.")
    return tuple(value)


def _normalize_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise AdapterError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise AdapterError(f"{field_name} must be a non-empty string.")
    return normalized


__all__ = [
    "AdapterError",
    "AdapterRequest",
    "AdapterResult",
    "FakeLocalAdapter",
    "ProviderAdapter",
    "StoredAdapterRun",
    "execute_adapter_run",
    "execute_fake_run",
]
