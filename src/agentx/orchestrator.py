from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .agentmemory_bridge import merge_agentmemory_records
from .memory import AgentXMemoryError, assemble_memory_prompt_packet
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
from .workspace import (
    PatchValidationResult,
    ScopedWorkspaceConfig,
    SecretScanner,
    WithheldPathSummary,
    WorkspaceMaterializationResult,
    materialize_scoped_workspace,
    validate_patch_paths,
)


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


@dataclass(frozen=True)
class ExecuteModeResult:
    run: AgentRun
    route: RouteDecision
    provider_class: str
    context_manifest: ContextManifest
    patch_validation: PatchValidationResult
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
            "patch_validation": self.patch_validation.as_dict(),
            "patch_accepted": self.patch_validation.accepted,
            "patch_applied": False,
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
    source_root: str | Path | None = None,
    workspace_id: str | None = None,
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
    active_memories = merge_agentmemory_records(memories, settings.paths.memories)
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
        memories=active_memories,
    )
    context_map = _context_map_from_manifest(context_manifest, route)
    _attach_agentmemory_prompt_packet(
        context_map,
        settings=settings,
        provider_class=provider_class,
        query=plan_run.prompt,
    )
    memory_map = _memory_map_from_manifest(context_manifest)
    _attach_agentmemory_omissions(memory_map, context_map)
    redactions = [entry.as_dict() for entry in context_manifest.redactions]
    selected_adapter = adapter or FakeLocalAdapter()
    provider_selection_error = _provider_selection_error(selected_adapter, route)
    workspace_materialization: WorkspaceMaterializationResult | None = None
    workspace_metadata: dict[str, object] | None = None
    if source_root is not None and provider_selection_error is None:
        workspace_materialization = _materialize_plan_workspace(
            session_store=session_store,
            session_id=session_id,
            source_root=source_root,
            context_manifest=context_manifest,
        )
        workspace_metadata = _workspace_metadata(
            workspace_materialization,
            workspace_id=workspace_id,
            default_workspace_id=session_id,
        )
        context_map["scoped_workspace"] = workspace_metadata

    if provider_selection_error is not None:
        selected_adapter = _ProviderSelectionFailureAdapter(
            provider_id=selected_adapter.provider_id,
            reason=provider_selection_error,
        )
    elif workspace_materialization is not None and not workspace_materialization.ok:
        selected_adapter = _WorkspaceMaterializationFailureAdapter(
            provider_id=selected_adapter.provider_id,
            workspace_materialization=workspace_materialization,
        )
    read_only_adapter = _ReadOnlyPlanAdapter(
        selected_adapter,
        workspace_materialization=workspace_metadata,
    )

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


def execute_execute_mode(
    *,
    settings: Settings,
    session_store: SessionStore,
    session_id: str,
    allowed_patch_paths: Sequence[str],
    denied_patch_paths: Sequence[str] = (),
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
    secret_scanner: SecretScanner | None = None,
) -> ExecuteModeResult:
    """Run controlled execute mode and persist audit artifacts.

    AX-014 validates provider patch output and records accepted patches, but it
    never applies patches to source files.
    """

    if not isinstance(settings, Settings):
        raise OrchestratorError("settings must be a Settings object.")
    if not isinstance(session_store, SessionStore):
        raise OrchestratorError("session_store must be a SessionStore.")

    execute_run = _build_execute_run(prompt=prompt, run=run, context_paths=context_paths)
    active_memories = merge_agentmemory_records(memories, settings.paths.memories)
    statuses = _normalize_provider_statuses(provider_statuses)
    active_policy = policy or _default_policy(settings)
    active_router = router or Router(
        settings=settings,
        providers=statuses,
        model_catalog=model_catalog,
        policy=active_policy,
    )
    route = active_router.explain(execute_run)
    provider_class = _provider_class_for_route(
        route,
        statuses=statuses,
        settings=settings,
        policy=active_policy,
        run=execute_run,
    )
    context_manifest = _compile_manifest(
        policy=active_policy,
        provider_class=provider_class,
        requested_paths=execute_run.context_paths,
        inferred_paths=inferred_context_paths,
        memories=active_memories,
    )
    context_map = _context_map_from_manifest(context_manifest, route)
    _attach_agentmemory_prompt_packet(
        context_map,
        settings=settings,
        provider_class=provider_class,
        query=execute_run.prompt,
    )
    memory_map = _memory_map_from_manifest(context_manifest)
    _attach_agentmemory_omissions(memory_map, context_map)
    redactions = [entry.as_dict() for entry in context_manifest.redactions]
    validating_adapter = _PatchValidatingExecuteAdapter(
        adapter=adapter or FakeLocalAdapter(),
        allowed_patch_paths=allowed_patch_paths,
        denied_patch_paths=denied_patch_paths,
        secret_scanner=secret_scanner,
    )

    stored = execute_adapter_run(
        session_store=session_store,
        session_id=session_id,
        run=execute_run,
        adapter=validating_adapter,
        route=route,
        context_map=context_map,
        memory_map=memory_map,
        redactions=redactions,
    )
    return ExecuteModeResult(
        run=execute_run,
        route=route,
        provider_class=provider_class,
        context_manifest=context_manifest,
        patch_validation=validating_adapter.patch_validation,
        stored_run=stored,
    )


@dataclass(frozen=True)
class _ReadOnlyPlanAdapter:
    adapter: ProviderAdapter
    workspace_materialization: Mapping[str, object] | None = None

    @property
    def provider_id(self) -> str:
        return self.adapter.provider_id

    def execute(self, request: AdapterRequest) -> AdapterResult:
        result = self.adapter.execute(request)
        if not isinstance(result, AdapterResult):
            raise OrchestratorError("adapter.execute must return an AdapterResult.")
        outcome = dict(result.outcome)
        if self.workspace_materialization is not None:
            outcome["scoped_workspace"] = dict(self.workspace_materialization)
        if result.patch == "" and outcome == dict(result.outcome):
            return result
        if result.patch != "":
            outcome["patch_suppressed"] = True
            outcome["patch_applied"] = False
        return replace(result, patch="", outcome=outcome)


@dataclass(frozen=True)
class _WorkspaceMaterializationFailureAdapter:
    provider_id: str
    workspace_materialization: WorkspaceMaterializationResult

    def execute(self, request: AdapterRequest) -> AdapterResult:
        route = request.route
        model_tier = route.selected_model_tier if route is not None else "high"
        errors = [
            event.as_dict()
            for event in self.workspace_materialization.events
            if event.severity == "error"
        ]
        return AdapterResult(
            provider_id=self.provider_id,
            model_id=None,
            model_tier=model_tier,
            status="failure",
            transcript_events=(
                {
                    "sequence": 1,
                    "event": "workspace_materialization_failed",
                    "provider_id": self.provider_id,
                    "mode": request.run.mode,
                    "errors": errors,
                },
            ),
            cost={
                "currency": "USD",
                "estimated": False,
                "known": True,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": 0.0,
            },
            outcome={
                "status": "failure",
                "outcome": "scoped_workspace_materialization_failed",
                "summary": "AgentX did not invoke the provider because scoped workspace materialization failed.",
                "patch_applied": False,
            },
            patch="",
        )


@dataclass(frozen=True)
class _ProviderSelectionFailureAdapter:
    provider_id: str
    reason: str

    def execute(self, request: AdapterRequest) -> AdapterResult:
        route = request.route
        model_tier = route.selected_model_tier if route is not None else "high"
        return AdapterResult(
            provider_id=self.provider_id,
            model_id=None,
            model_tier=model_tier,
            status="failure",
            transcript_events=(
                {
                    "sequence": 1,
                    "event": "provider_not_invoked",
                    "provider_id": self.provider_id,
                    "mode": request.run.mode,
                    "reason": self.reason,
                },
            ),
            cost={
                "currency": "USD",
                "estimated": False,
                "known": True,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": 0.0,
            },
            outcome={
                "status": "failure",
                "outcome": "provider_not_invoked",
                "summary": self.reason,
                "patch_applied": False,
            },
            patch="",
        )


@dataclass
class _PatchValidatingExecuteAdapter:
    adapter: ProviderAdapter
    allowed_patch_paths: Sequence[str]
    denied_patch_paths: Sequence[str] = ()
    secret_scanner: SecretScanner | None = None
    patch_validation: PatchValidationResult | None = None

    @property
    def provider_id(self) -> str:
        return self.adapter.provider_id

    def execute(self, request: AdapterRequest) -> AdapterResult:
        result = self.adapter.execute(request)
        if not isinstance(result, AdapterResult):
            raise OrchestratorError("adapter.execute must return an AdapterResult.")

        validation = validate_patch_paths(
            result.patch,
            allowed_paths=self.allowed_patch_paths,
            denied_paths=self.denied_patch_paths,
            secret_scanner=self.secret_scanner,
        )
        self.patch_validation = validation

        patch_has_content = result.patch.strip() != ""
        accepted_patch = validation.accepted
        stored_patch = result.patch if accepted_patch else ""
        status = result.status if accepted_patch else "validation_failed"
        outcome = dict(result.outcome)
        outcome.update(
            {
                "status": status,
                "outcome": (
                    outcome.get("outcome", "execute_completed")
                    if accepted_patch
                    else "patch_validation_failed"
                ),
                "patch_accepted": accepted_patch,
                "patch_applied": False,
                "patch_application": {
                    "applied": False,
                    "supported": False,
                    "reason": "AX-014 validates adapter patches and stores accepted patch artifacts, but does not apply source mutations.",
                },
                "patch_present": patch_has_content,
                "patch_stored": bool(stored_patch),
                "patch_validation": validation.as_dict(),
            }
        )

        return replace(
            result,
            status=status,
            patch=stored_patch,
            outcome=outcome,
        )


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


def _build_execute_run(
    *,
    prompt: str | None,
    run: AgentRun | None,
    context_paths: Sequence[str] | None,
) -> AgentRun:
    if run is not None:
        if prompt is not None:
            raise OrchestratorError("pass either run or prompt, not both.")
        if run.mode != "execute":
            raise OrchestratorError("execute mode requires run.mode to be 'execute'.")
        if context_paths is None:
            return run
        return replace(run, context_paths=_normalize_context_paths(context_paths))

    if prompt is None:
        raise OrchestratorError("prompt is required when run is not supplied.")
    return AgentRun(
        prompt=prompt,
        mode="execute",
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


def _attach_agentmemory_prompt_packet(
    context_map: dict[str, object],
    *,
    settings: Settings,
    provider_class: str,
    query: str,
) -> None:
    try:
        packet = assemble_memory_prompt_packet(
        settings.paths,
        provider_class=provider_class,
        query="",
        current_prompt="",
    )
    except AgentXMemoryError:
        return
    if packet is not None:
        context_map["agentmemory_prompt"] = packet


def _attach_agentmemory_omissions(memory_map: dict[str, object], context_map: Mapping[str, object]) -> None:
    packet = context_map.get("agentmemory_prompt")
    if not isinstance(packet, Mapping):
        return
    omitted = packet.get("omitted_memory_ids")
    if isinstance(omitted, Sequence) and not isinstance(omitted, (str, bytes)):
        memory_map["agentmemory_omitted_memory_ids"] = [str(item) for item in omitted]


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


def _provider_selection_error(
    adapter: ProviderAdapter,
    route: RouteDecision,
) -> str | None:
    selected = route.selected_provider
    if selected == adapter.provider_id:
        return None
    if selected is None:
        report = next(
            (
                provider_report
                for provider_report in route.provider_reports
                if provider_report.id == adapter.provider_id
            ),
            None,
        )
        if report is not None and report.reason in _FILTERABLE_CONTEXT_REJECTION_REASONS:
            return None
        if report is not None:
            return (
                f"Provider '{adapter.provider_id}' was not invoked because routing rejected it: {report.reason}."
            )
        return (
            f"Provider '{adapter.provider_id}' was not invoked because routing did not select an eligible provider."
        )
    return (
        f"Provider '{adapter.provider_id}' was not invoked because routing selected provider '{selected}'."
    )


_FILTERABLE_CONTEXT_REJECTION_REASONS: frozenset[str] = frozenset(
    {
        "classification_exceeds_external_max",
        "classification_routing_restricted",
        "policy_restricted",
        "secret_requires_explicit_routing",
        "unclassified_requires_private",
    }
)


def _materialize_plan_workspace(
    *,
    session_store: SessionStore,
    session_id: str,
    source_root: str | Path,
    context_manifest: ContextManifest,
) -> WorkspaceMaterializationResult:
    session_root = session_store.path_for_session(session_id)
    workspace_root = session_root / "workspace"
    if workspace_root.exists():
        shutil.rmtree(workspace_root)

    summaries_by_path = {
        entry.target_id: entry
        for entry in context_manifest.summary_substitutions
        if entry.target_type == "path"
    }
    withheld_paths = []
    for path in context_manifest.excluded_paths:
        summary = summaries_by_path.get(path)
        withheld_paths.append(
            WithheldPathSummary(
                path=path,
                classification=context_manifest.classification_by_path.get(path),
                reason=(
                    summary.reason
                    if summary is not None
                    else "withheld_by_context_policy"
                ),
                summary=None if summary is None else summary.summary,
            )
        )

    return materialize_scoped_workspace(
        ScopedWorkspaceConfig.plan(
            source_root=Path(source_root),
            workspace_root=workspace_root,
            allowed_paths=context_manifest.included_paths,
            withheld_paths=withheld_paths,
        )
    )


def _workspace_metadata(
    materialization: WorkspaceMaterializationResult,
    *,
    workspace_id: str | None,
    default_workspace_id: str,
) -> dict[str, object]:
    metadata = materialization.as_dict()
    metadata["workspace_id"] = workspace_id or default_workspace_id
    return metadata


__all__ = [
    "ExecuteModeResult",
    "OrchestratorError",
    "PlanModeResult",
    "execute_execute_mode",
    "execute_plan_mode",
]
