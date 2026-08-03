from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol


PLAN_WORKSPACE_MODE = "plan-readonly"
EXECUTE_WORKSPACE_MODE = "execute-overlay"
VALID_WORKSPACE_MODES: frozenset[str] = frozenset(
    {PLAN_WORKSPACE_MODE, EXECUTE_WORKSPACE_MODE}
)


class WorkspaceError(ValueError):
    """Raised when scoped workspace inputs are invalid."""


@dataclass(frozen=True)
class WorkspaceEvent:
    code: str
    severity: str
    message: str
    path: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _normalize_non_empty_string(self.code, "event.code"))
        object.__setattr__(
            self,
            "severity",
            _normalize_choice(self.severity, "event.severity", {"info", "warning", "error"}),
        )
        object.__setattr__(
            self,
            "message",
            _normalize_non_empty_string(self.message, "event.message"),
        )
        if self.path is not None:
            object.__setattr__(self, "path", normalize_scoped_path(self.path, "event.path"))
        object.__setattr__(self, "details", dict(self.details))

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class WithheldPathSummary:
    path: str
    classification: str | None = None
    reason: str = "withheld_by_policy"
    summary: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_scoped_path(self.path, "withheld.path"))
        if self.classification is not None:
            object.__setattr__(
                self,
                "classification",
                _normalize_non_empty_string(
                    self.classification,
                    "withheld.classification",
                ),
            )
        object.__setattr__(
            self,
            "reason",
            _normalize_non_empty_string(self.reason, "withheld.reason"),
        )
        if self.summary is not None:
            object.__setattr__(
                self,
                "summary",
                _normalize_non_empty_string(self.summary, "withheld.summary"),
            )

    def placeholder_text(self) -> str:
        lines = [
            "# AgentX withheld file",
            "",
            f"Path: {self.path}",
            f"Reason: {self.reason}",
        ]
        if self.classification is not None:
            lines.append(f"Classification: {self.classification}")
        lines.append("")
        lines.append(self.summary or "Content withheld by AgentX policy.")
        lines.append("")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "classification": self.classification,
            "reason": self.reason,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ScopedWorkspaceConfig:
    source_root: Path
    workspace_root: Path
    mode: str
    allowed_paths: tuple[str, ...] = ()
    withheld_paths: tuple[WithheldPathSummary, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "workspace_root", Path(self.workspace_root))
        object.__setattr__(
            self,
            "mode",
            _normalize_choice(self.mode, "workspace.mode", VALID_WORKSPACE_MODES),
        )
        normalized_allowed = _normalize_path_sequence(self.allowed_paths, "allowed_paths")
        normalized_withheld = _normalize_withheld_paths(self.withheld_paths)
        _reject_duplicate_aliases(
            (*normalized_allowed, *(entry.path for entry in normalized_withheld)),
            "workspace paths",
        )
        object.__setattr__(self, "allowed_paths", normalized_allowed)
        object.__setattr__(self, "withheld_paths", normalized_withheld)

    @classmethod
    def plan(
        cls,
        *,
        source_root: Path,
        workspace_root: Path,
        allowed_paths: Sequence[str] = (),
        withheld_paths: Sequence[WithheldPathSummary | Mapping[str, object]] = (),
    ) -> "ScopedWorkspaceConfig":
        return cls(
            source_root=source_root,
            workspace_root=workspace_root,
            mode=PLAN_WORKSPACE_MODE,
            allowed_paths=tuple(allowed_paths),
            withheld_paths=_normalize_withheld_paths(withheld_paths),
        )

    @classmethod
    def execute(
        cls,
        *,
        source_root: Path,
        workspace_root: Path,
        allowed_paths: Sequence[str] = (),
        withheld_paths: Sequence[WithheldPathSummary | Mapping[str, object]] = (),
    ) -> "ScopedWorkspaceConfig":
        return cls(
            source_root=source_root,
            workspace_root=workspace_root,
            mode=EXECUTE_WORKSPACE_MODE,
            allowed_paths=tuple(allowed_paths),
            withheld_paths=_normalize_withheld_paths(withheld_paths),
        )

    @property
    def write_overlay(self) -> bool:
        return self.mode == EXECUTE_WORKSPACE_MODE

    def as_dict(self) -> dict[str, object]:
        return {
            "source_root": str(self.source_root),
            "workspace_root": str(self.workspace_root),
            "mode": self.mode,
            "write_overlay": self.write_overlay,
            "allowed_paths": list(self.allowed_paths),
            "withheld_paths": [entry.as_dict() for entry in self.withheld_paths],
        }


@dataclass(frozen=True)
class WorkspaceMaterializedEntry:
    path: str
    destination: str
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_scoped_path(self.path, "entry.path"))
        object.__setattr__(self, "destination", str(self.destination))
        object.__setattr__(
            self,
            "kind",
            _normalize_choice(self.kind, "entry.kind", {"copied_file", "summary_placeholder"}),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "destination": self.destination,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class WorkspaceMaterializationResult:
    ok: bool
    mode: str
    workspace_root: str
    entries: tuple[WorkspaceMaterializedEntry, ...]
    events: tuple[WorkspaceEvent, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "workspace_root": self.workspace_root,
            "entries": [entry.as_dict() for entry in self.entries],
            "events": [event.as_dict() for event in self.events],
        }


@dataclass(frozen=True)
class SecretFinding:
    marker_class: str
    line_number: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "marker_class",
            _normalize_non_empty_string(self.marker_class, "secret.marker_class"),
        )
        if not isinstance(self.line_number, int) or self.line_number < 1:
            raise WorkspaceError("secret.line_number must be a positive integer.")

    def as_dict(self) -> dict[str, object]:
        return {
            "marker_class": self.marker_class,
            "line_number": self.line_number,
        }


class SecretScanner(Protocol):
    def scan(self, text: str) -> Sequence[SecretFinding]:
        """Return non-secret metadata for secret-like content in text."""


@dataclass(frozen=True)
class MarkerSecretRule:
    marker: str
    marker_class: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "marker", _normalize_non_empty_string(self.marker, "marker"))
        object.__setattr__(
            self,
            "marker_class",
            _normalize_non_empty_string(self.marker_class, "marker_class"),
        )


@dataclass(frozen=True)
class MarkerSecretScanner:
    rules: tuple[MarkerSecretRule, ...] = ()

    def __post_init__(self) -> None:
        normalized: list[MarkerSecretRule] = []
        for rule in self.rules:
            if not isinstance(rule, MarkerSecretRule):
                raise WorkspaceError("secret scanner rules must be MarkerSecretRule entries.")
            normalized.append(rule)
        object.__setattr__(self, "rules", tuple(normalized))

    def scan(self, text: str) -> tuple[SecretFinding, ...]:
        if not isinstance(text, str):
            raise WorkspaceError("secret scanner text must be a string.")
        findings: list[SecretFinding] = []
        seen: set[tuple[str, int]] = set()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule in self.rules:
                if rule.marker not in line:
                    continue
                key = (rule.marker_class, line_number)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    SecretFinding(
                        marker_class=rule.marker_class,
                        line_number=line_number,
                    )
                )
        return tuple(findings)


@dataclass(frozen=True)
class PatchValidationResult:
    accepted: bool
    paths: tuple[str, ...]
    events: tuple[WorkspaceEvent, ...]
    secret_findings: tuple[SecretFinding, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "paths": list(self.paths),
            "events": [event.as_dict() for event in self.events],
            "secret_findings": [finding.as_dict() for finding in self.secret_findings],
        }


def normalize_scoped_path(path: object, field_name: str = "path") -> str:
    if not isinstance(path, str):
        raise WorkspaceError(f"{field_name} must be a string.")
    raw = path.strip()
    if not raw:
        raise WorkspaceError(f"{field_name} must be a non-empty relative path.")
    if "\x00" in raw:
        raise WorkspaceError(f"{field_name} must not contain NUL bytes.")
    if _is_absolute_or_drive_path(raw):
        raise WorkspaceError(f"{field_name} must be relative.")

    parts: list[str] = []
    for part in raw.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise WorkspaceError(f"{field_name} must not contain parent traversal.")
        parts.append(part)
    if not parts:
        raise WorkspaceError(f"{field_name} must be a non-empty relative path.")
    return "/".join(parts)


def materialize_scoped_workspace(
    config: ScopedWorkspaceConfig,
) -> WorkspaceMaterializationResult:
    if not isinstance(config, ScopedWorkspaceConfig):
        raise WorkspaceError("config must be a ScopedWorkspaceConfig.")

    events: list[WorkspaceEvent] = []
    entries: list[WorkspaceMaterializedEntry] = []
    config.workspace_root.mkdir(parents=True, exist_ok=True)

    for relative_path in config.allowed_paths:
        source_path = config.source_root / Path(*relative_path.split("/"))
        destination = _workspace_destination(config.workspace_root, relative_path)
        if source_path.is_symlink():
            events.append(
                WorkspaceEvent(
                    code="source_symlink_rejected",
                    severity="error",
                    path=relative_path,
                    message="Allowed path was not copied because symlinks are not materialized.",
                )
            )
            continue
        if not source_path.is_file():
            events.append(
                WorkspaceEvent(
                    code="source_file_missing",
                    severity="error",
                    path=relative_path,
                    message="Allowed path does not resolve to a source file.",
                )
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        entries.append(
            WorkspaceMaterializedEntry(
                path=relative_path,
                destination=str(destination),
                kind="copied_file",
            )
        )
        events.append(
            WorkspaceEvent(
                code="file_materialized",
                severity="info",
                path=relative_path,
                message="Allowed source file was copied into the scoped workspace.",
            )
        )

    for withheld in config.withheld_paths:
        destination = _workspace_destination(config.workspace_root, withheld.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(withheld.placeholder_text(), encoding="utf-8", newline="\n")
        entries.append(
            WorkspaceMaterializedEntry(
                path=withheld.path,
                destination=str(destination),
                kind="summary_placeholder",
            )
        )
        events.append(
            WorkspaceEvent(
                code="withheld_placeholder_materialized",
                severity="info",
                path=withheld.path,
                message="A summary placeholder was written for a withheld path.",
                details={
                    "classification": withheld.classification,
                    "reason": withheld.reason,
                },
            )
        )

    return WorkspaceMaterializationResult(
        ok=not any(event.severity == "error" for event in events),
        mode=config.mode,
        workspace_root=str(config.workspace_root),
        entries=tuple(entries),
        events=tuple(events),
    )


def validate_patch_paths(
    patch_text: str,
    *,
    allowed_paths: Sequence[str],
    denied_paths: Sequence[str] = (),
    secret_scanner: SecretScanner | None = None,
) -> PatchValidationResult:
    if not isinstance(patch_text, str):
        raise WorkspaceError("patch_text must be a string.")

    events: list[WorkspaceEvent] = []
    extracted_paths, path_events = _extract_patch_paths(patch_text)
    events.extend(path_events)
    events.extend(_patch_format_events(patch_text, extracted_paths))

    allowed = set(_normalize_path_sequence(allowed_paths, "allowed_paths"))
    denied = set(_normalize_path_sequence(denied_paths, "denied_paths"))
    _reject_duplicate_aliases(tuple(allowed), "allowed_paths")
    _reject_duplicate_aliases(tuple(denied), "denied_paths")
    paths: list[str] = []
    for path in extracted_paths:
        if path in paths:
            continue
        paths.append(path)
        if path in denied:
            events.append(
                WorkspaceEvent(
                    code="patch_path_denied",
                    severity="error",
                    path=path,
                    message="Patch targets a denied path.",
                )
            )
        if path not in allowed:
            events.append(
                WorkspaceEvent(
                    code="patch_path_out_of_scope",
                    severity="error",
                    path=path,
                    message="Patch targets a path outside the allowed workspace scope.",
                )
            )

    findings: tuple[SecretFinding, ...] = ()
    if secret_scanner is not None:
        findings = tuple(secret_scanner.scan(patch_text))
        for finding in findings:
            events.append(
                WorkspaceEvent(
                    code="patch_secret_marker_detected",
                    severity="error",
                    message="Patch text matched a configured secret marker class.",
                    details={
                        "marker_class": finding.marker_class,
                        "line_number": finding.line_number,
                    },
                )
            )

    return PatchValidationResult(
        accepted=not any(event.severity == "error" for event in events),
        paths=tuple(paths),
        events=tuple(events),
        secret_findings=findings,
    )


def _extract_patch_paths(patch_text: str) -> tuple[tuple[str, ...], tuple[WorkspaceEvent, ...]]:
    paths: list[str] = []
    events: list[WorkspaceEvent] = []
    for line_number, line in enumerate(patch_text.splitlines(), start=1):
        candidates: tuple[str, ...] = ()
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                candidates = (parts[2], parts[3])
        elif line.startswith("--- ") or line.startswith("+++ "):
            token = line[4:].split("\t", 1)[0].strip()
            if token:
                candidates = (token,)

        for candidate in candidates:
            if candidate == "/dev/null":
                continue
            patch_path = _strip_diff_prefix(candidate)
            try:
                normalized = normalize_scoped_path(patch_path, "patch.path")
            except WorkspaceError as exc:
                events.append(
                    WorkspaceEvent(
                        code="patch_path_invalid",
                        severity="error",
                        message=str(exc),
                        details={"line_number": line_number},
                    )
                )
                continue
            if normalized not in paths:
                paths.append(normalized)
    return tuple(paths), tuple(events)


def _patch_format_events(patch_text: str, paths: Sequence[str]) -> tuple[WorkspaceEvent, ...]:
    stripped = patch_text.strip()
    if not stripped:
        return ()
    if not paths:
        return (
            WorkspaceEvent(
                code="patch_no_target_paths",
                severity="error",
                message="Patch did not include any target paths.",
            ),
        )
    if not any(line.startswith("@@") for line in patch_text.splitlines()):
        return (
            WorkspaceEvent(
                code="patch_hunk_missing",
                severity="error",
                message="Patch did not include a unified diff hunk.",
            ),
        )
    return ()


def _strip_diff_prefix(path: str) -> str:
    if path.startswith(("a/", "b/")) and len(path) > 2:
        return path[2:]
    return path


def _workspace_destination(root: Path, relative_path: str) -> Path:
    return root / Path(*relative_path.split("/"))


def _normalize_path_sequence(value: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorkspaceError(f"{field_name} must be a sequence of paths.")
    paths = tuple(normalize_scoped_path(item, field_name) for item in value)
    _reject_duplicate_aliases(paths, field_name)
    return paths


def _normalize_withheld_paths(
    value: Sequence[WithheldPathSummary | Mapping[str, object]],
) -> tuple[WithheldPathSummary, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorkspaceError("withheld_paths must be a sequence.")
    entries: list[WithheldPathSummary] = []
    for item in value:
        if isinstance(item, WithheldPathSummary):
            entries.append(item)
            continue
        if isinstance(item, Mapping):
            entries.append(
                WithheldPathSummary(
                    path=item.get("path"),
                    classification=item.get("classification"),
                    reason=str(item.get("reason", "withheld_by_policy")),
                    summary=item.get("summary"),
                )
            )
            continue
        raise WorkspaceError("withheld_paths entries must be WithheldPathSummary objects.")
    return tuple(entries)


def _reject_duplicate_aliases(paths: Sequence[str], field_name: str) -> None:
    seen_exact: set[str] = set()
    seen_folded: dict[str, str] = {}
    for path in paths:
        normalized = normalize_scoped_path(path, field_name)
        if normalized in seen_exact:
            raise WorkspaceError(f"{field_name} contains duplicate path '{normalized}'.")
        seen_exact.add(normalized)
        folded = normalized.casefold()
        if folded in seen_folded and seen_folded[folded] != normalized:
            raise WorkspaceError(
                f"{field_name} contains case-ambiguous path aliases "
                f"'{seen_folded[folded]}' and '{normalized}'."
            )
        seen_folded[folded] = normalized


def _is_absolute_or_drive_path(path: str) -> bool:
    posix = PurePosixPath(path.replace("\\", "/"))
    windows = PureWindowsPath(path)
    return (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
    )


def _normalize_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise WorkspaceError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise WorkspaceError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_choice(value: object, field_name: str, valid_values: set[str] | frozenset[str]) -> str:
    normalized = _normalize_non_empty_string(value, field_name)
    if normalized not in valid_values:
        raise WorkspaceError(
            f"{field_name} must be one of: {', '.join(sorted(valid_values))}."
        )
    return normalized


__all__ = [
    "EXECUTE_WORKSPACE_MODE",
    "MarkerSecretRule",
    "MarkerSecretScanner",
    "PLAN_WORKSPACE_MODE",
    "PatchValidationResult",
    "ScopedWorkspaceConfig",
    "SecretFinding",
    "SecretScanner",
    "VALID_WORKSPACE_MODES",
    "WithheldPathSummary",
    "WorkspaceError",
    "WorkspaceEvent",
    "WorkspaceMaterializationResult",
    "WorkspaceMaterializedEntry",
    "materialize_scoped_workspace",
    "normalize_scoped_path",
    "validate_patch_paths",
]
