from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .adapters import (
    AdapterRequest,
    AdapterResult,
    FakeLocalAdapter,
    ProviderAdapter,
    StoredAdapterRun,
    execute_adapter_run,
)
from .config import Settings
from .context import (
    ContextManifest,
    MemoryRecord,
    compile_external_context_manifest,
    compile_private_context_manifest,
)
from .models import ModelCatalog
from .policy import Policy
from .providers import ProviderStatus
from .routing import AgentRun, RouteDecision, Router
from .store import SessionStore


class OrchestratorError(ValueError):
    """Raised when an orchestrated agent workflow cannot be built safely."""


@dataclass(frozen=True)
class PlanModeResult:
    run: AgentRun
    route: RouteDecision
    provider_class: str
    context_manifest: ContextManifest
    stored_run: StoredAdapterRun

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.stored_run.session_id,
            "root": str(self.stored_run.root),
            "artifacts": {
                name: str(path)
                for name, path in sorted(self.stored_run.artifact_paths.items())
            },
            "run": self.run.as_dict(),
            "route": self.route.as_dict(),
            "provider_class": self.provider_class,
            "context_manifest": self.context_manifest.as_dict(),
            "result": self.stored_run.result.as_dict(),
        }


def execute_plan_mode(
    *,
    settings: Settings,
    session_store: SessionStore,
    session_id: str,
    prompt: str | None = None,
    run: AgentRun | None = None,
    provider_statuses: Sequence[ProviderStatus] | None = None,
    router: Router | None = None,
    model_catalog: ModelCatalog | None = None,
    policy: Policy | None = None,
    memories: Sequence[MemoryRecord] = (),
    context_paths: Sequence[str] | None = None,
    inferred_context_paths: Sequence[str] = (),
    adapter: ProviderAdapter | None = None,
) -> PlanModeResult:
    """Run a plan-only workflow and persist standard local artifacts.

    The default provider status and adapter are deterministic fakes. Callers that
    want live routing or execution must inject those dependencies explicitly.
    """

    if not isinstance(settings, Settings):
        raise OrchestratorError("settings must be a Settings object.")
    if not isinstance(session_store, SessionStore):
        raise OrchestratorError("session_store must be a SessionStore.")

    plan_run = _build_plan_run(prompt=prompt, run=run, context_paths=context_paths)
    statuses = _normalize_provider_statuses(provider_statuses)
    active_policy = policy or _default_policy(settings)
    active_router = router or Router(
        settings=settings,
        providers=statuses,
        model_catalog=model_catalog,
        policy=active_policy,
    )
    route = active_router.explain(plan_run)
    provider_class = _provider_class_for_route(
        route,
        statuses=statuses,
        settings=settings,
        policy=active_policy,
        run=plan_run,
    )
    context_manifest = _compile_manifest(
        policy=active_policy,
        provider_class=provider_class,
        requested_paths=plan_run.context_paths,
        inferred_paths=inferred_context_paths,
        memories=memories,
    )
    context_map = _context_map_from_manifest(context_manifest, route)
    memory_map = _memory_map_from_manifest(context_manifest)
    redactions = [entry.as_dict() for entry in context_manifest.redactions]
    read_only_adapter = _ReadOnlyPlanAdapter(adapter or FakeLocalAdapter())

    stored = execute_adapter_run(
        session_store=session_store,
        session_id=session_id,
        run=plan_run,
        adapter=read_only_adapter,
        route=route,
        context_map=context_map,
        memory_map=memory_map,
        redactions=redactions,
    )
    return PlanModeResult(
        run=plan_run,
        route=route,
        provider_class=provider_class,
        context_manifest=context_manifest,
        stored_run=stored,
    )


@dataclass(frozen=True)
class _ReadOnlyPlanAdapter:
    adapter: ProviderAdapter

    @property
    def provider_id(self) -> str:
        return self.adapter.provider_id

    def execute(self, request: AdapterRequest) -> AdapterResult:
        result = self.adapter.execute(request)
        if not isinstance(result, AdapterResult):
            raise OrchestratorError("adapter.execute must return an AdapterResult.")
        if result.patch == "":
            return result
        outcome = dict(result.outcome)
        outcome["patch_suppressed"] = True
        outcome["patch_applied"] = False
        return replace(result, patch="", outcome=outcome)


def _build_plan_run(
    *,
    prompt: str | None,
    run: AgentRun | None,
    context_paths: Sequence[str] | None,
) -> AgentRun:
    if run is not None:
        if prompt is not None:
            raise OrchestratorError("pass either run or prompt, not both.")
        if run.mode != "plan":
            raise OrchestratorError("plan mode requires run.mode to be 'plan'.")
        if context_paths is None:
            return run
        return replace(run, context_paths=_normalize_context_paths(context_paths))

    if prompt is None:
        raise OrchestratorError("prompt is required when run is not supplied.")
    return AgentRun(
        prompt=prompt,
        mode="plan",
        provider="auto",
        context_paths=_normalize_context_paths(context_paths or ()),
    )


def _normalize_context_paths(paths: Sequence[str]) -> tuple[str, ...]:
    if isinstance(paths, (str, bytes)):
        return (str(paths),)
    return tuple(paths)


def _normalize_provider_statuses(
    provider_statuses: Sequence[ProviderStatus] | None,
) -> tuple[ProviderStatus, ...]:
    if provider_statuses is None:
        return (
            ProviderStatus(
                id="fake-local",
                display_name="Fake Local Adapter",
                kind="local",
                enabled=True,
                reason="available",
            ),
        )
    if isinstance(provider_statuses, (str, bytes)):
        raise OrchestratorError("provider_statuses must be a sequence of ProviderStatus objects.")
    statuses = tuple(provider_statuses)
    for status in statuses:
        if not isinstance(status, ProviderStatus):
            raise OrchestratorError("provider_statuses must contain ProviderStatus objects.")
    return statuses


def _default_policy(settings: Settings) -> Policy:
    return Policy(
        external_max_classification=settings.external_max_classification,
        private_provider=settings.private_provider,
    )


def _provider_class_for_route(
    route: RouteDecision,
    *,
    statuses: Sequence[ProviderStatus],
    settings: Settings,
    policy: Policy,
    run: AgentRun,
) -> str:
    provider_id = route.selected_provider
    if provider_id is None and run.provider != "auto":
        provider_id = run.provider

    status_by_id = {status.id: status for status in statuses}
    status = status_by_id.get(provider_id or "")
    private_provider = settings.private_provider or policy.private_provider
    if provider_id and provider_id == private_provider:
        return "private"
    if provider_id and provider_id.startswith("private"):
        return "private"
    if status is not None and status.kind == "openai_compatible":
        return "private"
    return "external"


def _compile_manifest(
    *,
    policy: Policy,
    provider_class: str,
    requested_paths: Sequence[str],
    inferred_paths: Sequence[str],
    memories: Sequence[MemoryRecord],
) -> ContextManifest:
    if provider_class == "private":
        return compile_private_context_manifest(
            policy,
            requested_paths=requested_paths,
            inferred_paths=inferred_paths,
            memories=memories,
        )
    return compile_external_context_manifest(
        policy,
        requested_paths=requested_paths,
        inferred_paths=inferred_paths,
        memories=memories,
    )


def _context_map_from_manifest(
    manifest: ContextManifest,
    route: RouteDecision,
) -> dict[str, object]:
    manifest_dict = manifest.as_dict()
    return {
        "schema_version": 1,
        "source": "compiled_context_manifest",
        "route": {
            "selected_provider": route.selected_provider,
            "selected_model_id": route.selected_model_id,
            "selected_model_tier": route.selected_model_tier,
            "reason": route.reason,
            "explanation": route.explanation,
        },
        "requested_paths": manifest_dict["requested_paths"],
        "inferred_paths": manifest_dict["inferred_paths"],
        "included_paths": manifest_dict["included_paths"],
        "excluded_paths": manifest_dict["excluded_paths"],
        "classification_by_path": manifest_dict["classification_by_path"],
        "provider_visible_context": manifest_dict["provider_visible_context"],
        "policy_decision": manifest_dict["policy_decision"],
        "summary_substitutions": manifest_dict["summary_substitutions"],
    }


def _memory_map_from_manifest(manifest: ContextManifest) -> dict[str, object]:
    exposure = [decision.as_dict() for decision in manifest.memory_exposure]
    by_action: dict[str, list[str]] = {
        "include": [],
        "summarize": [],
        "redact": [],
        "exclude": [],
    }
    for decision in manifest.memory_exposure:
        by_action[decision.action].append(decision.memory_id)

    return {
        "schema_version": 1,
        "source": "compiled_context_manifest",
        "memory_ids": [decision["memory_id"] for decision in exposure],
        "included_memories": by_action["include"],
        "summarized_memories": by_action["summarize"],
        "redacted_memories": by_action["redact"],
        "excluded_memories": by_action["exclude"],
        "memory_exposure": exposure,
    }


__all__ = [
    "OrchestratorError",
    "PlanModeResult",
    "execute_plan_mode",
]
