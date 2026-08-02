from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .browser import BrowserToolExecutor
from .config import AgentXPaths
from .memory import AgentMemoryTools
from .tools import (
    ApprovalCallback,
    CompositeToolExecutor,
    ControlledWorkspaceTools,
    ReadOnlyWorkspaceTools,
    ToolError,
    ToolExecutor,
)
from .web import WebAccessService


VALID_TOOL_MODES = frozenset({"plan", "execute", "memory"})


def build_private_tool_executor(
    *,
    mode: str,
    workspace_root: str | Path,
    context_paths: Sequence[str],
    paths: AgentXPaths,
    user_prompt: str,
    approval_callback: ApprovalCallback | None = None,
    artifacts_dir: str | Path | None = None,
    extra_executors: Sequence[ToolExecutor] = (),
) -> CompositeToolExecutor:
    normalized_mode = _normalize_mode(mode)
    executors: list[ToolExecutor] = []
    if normalized_mode in {"plan", "execute"}:
        if normalized_mode == "execute":
            executors.append(
                ControlledWorkspaceTools(
                    workspace_root,
                    allowed_paths=context_paths,
                    approval_callback=approval_callback,
                    enable_patch=True,
                    enable_shell=True,
                )
            )
        else:
            executors.append(ReadOnlyWorkspaceTools(workspace_root, allowed_paths=context_paths))
    executors.append(AgentMemoryTools(paths, user_prompt=user_prompt, approval_callback=approval_callback))
    if normalized_mode in {"plan", "execute"} and approval_callback is not None:
        executors.append(WebAccessService(approval_callback=approval_callback))
        if artifacts_dir is not None:
            executors.append(
                BrowserToolExecutor(
                    artifacts_dir=Path(artifacts_dir),
                    approval_callback=approval_callback,
                )
            )
    executors.extend(extra_executors)
    return CompositeToolExecutor(*executors)


def tool_names(executor: ToolExecutor) -> tuple[str, ...]:
    return tuple(spec.name for spec in executor.specs)


def _normalize_mode(mode: str) -> str:
    if not isinstance(mode, str) or mode not in VALID_TOOL_MODES:
        raise ToolError(f"tool mode must be one of: {', '.join(sorted(VALID_TOOL_MODES))}")
    return mode
