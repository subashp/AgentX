"""Policy-bounded read-only workspace tools for AgentX providers."""

from __future__ import annotations

import fnmatch
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .workspace import WorkspaceError, normalize_scoped_path, validate_patch_paths


class ToolError(ValueError):
    """Raised when a workspace tool request is invalid or out of scope."""


class ToolExecutor(Protocol):
    """Provider-neutral boundary for model-requested tools."""

    @property
    def specs(self) -> Sequence["ToolSpec"]:
        """Return the tool schemas exposed to a model."""

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> "ToolResult":
        """Execute one validated tool call and return a bounded result."""


ApprovalCallback = Callable[[str, Mapping[str, object]], bool]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    output: object = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"name": self.name, "ok": self.ok}
        if self.ok:
            result["output"] = self.output
        else:
            result["error"] = self.error or "tool failed"
        return result

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)


_DENIED_DIRECTORY_NAMES = frozenset({".git", ".agentx", ".codex", ".agents", "__pycache__"})
_DENIED_FILE_PATTERNS = (".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx")
_DENIED_FILE_NAMES = frozenset({"auth.json", "credentials.json", "secrets.json"})
_MAX_OUTPUT_CHARS = 24_000
_MAX_TREE_ENTRIES = 500
_MAX_SEARCH_RESULTS = 200


READ_ONLY_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "workspace.tree",
        "List safe files and directories in the scoped workspace.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory, or empty for the workspace root."},
                "max_entries": {"type": "integer", "minimum": 1, "maximum": _MAX_TREE_ENTRIES},
            },
        },
    ),
    ToolSpec(
        "workspace.read",
        "Read a bounded range from one safe UTF-8 text file.",
        {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "max_chars": {"type": "integer", "minimum": 1, "maximum": _MAX_OUTPUT_CHARS},
            },
        },
    ),
    ToolSpec(
        "workspace.search",
        "Search safe text files for a literal string.",
        {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "path": {"type": "string"},
                "case_sensitive": {"type": "boolean"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": _MAX_SEARCH_RESULTS},
            },
        },
    ),
    ToolSpec(
        "git.status",
        "Show safe, read-only Git status for the scoped workspace.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "git.diff",
        "Show a bounded read-only diff for explicitly scoped paths.",
        {
            "type": "object",
            "required": ["paths"],
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "staged": {"type": "boolean"},
            },
        },
    ),
)


CONTROLLED_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "workspace.patch",
        "Apply a unified diff to explicitly approved workspace paths after user approval.",
        {
            "type": "object",
            "required": ["patch"],
            "properties": {
                "patch": {"type": "string", "minLength": 1},
            },
        },
    ),
    ToolSpec(
        "shell.exec",
        "Run an explicitly approved executable argv in the scoped workspace without a shell.",
        {
            "type": "object",
            "required": ["argv"],
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 32},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
            },
        },
    ),
)


class ReadOnlyWorkspaceTools:
    """Execute safe workspace inspection tools without a shell."""

    def __init__(self, root: str | Path, *, allowed_paths: Sequence[str] = ()) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ToolError(f"workspace root is not a directory: {self.root}")
        try:
            self.allowed_paths = tuple(normalize_scoped_path(path, "allowed_path") for path in allowed_paths)
        except WorkspaceError as exc:
            raise ToolError(str(exc)) from exc

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return READ_ONLY_TOOL_SPECS

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        args = dict(arguments or {})
        try:
            output = {
                "workspace.tree": self.tree,
                "workspace.read": self.read,
                "workspace.search": self.search,
                "git.status": self.git_status,
                "git.diff": self.git_diff,
            }[name](args)
        except (KeyError, ToolError, OSError, ValueError) as exc:
            return ToolResult(name=name, ok=False, error=str(exc))
        return ToolResult(name=name, ok=True, output=output)

    def tree(self, args: Mapping[str, object]) -> dict[str, object]:
        path = self._optional_path(args.get("path"))
        max_entries = _bounded_int(args.get("max_entries", 200), "max_entries", 1, _MAX_TREE_ENTRIES)
        roots = self._roots_for(path)
        entries: list[dict[str, str]] = []
        for root in roots:
            if root.is_file():
                entries.append({"path": self._relative(root), "kind": "file"})
                continue
            for current, directories, files in self._walk(root):
                for directory in directories:
                    entries.append({"path": self._relative(current / directory), "kind": "directory"})
                for filename in files:
                    entries.append({"path": self._relative(current / filename), "kind": "file"})
                if len(entries) >= max_entries:
                    break
            if len(entries) >= max_entries:
                break
        entries = sorted(entries, key=lambda item: item["path"])[:max_entries]
        return {"root": path or ".", "entries": entries, "truncated": len(entries) >= max_entries}

    def read(self, args: Mapping[str, object]) -> dict[str, object]:
        path = self._required_path(args.get("path"))
        file_path = self._safe_path(path, must_exist=True)
        if not file_path.is_file():
            raise ToolError(f"not a file: {path}")
        max_chars = _bounded_int(args.get("max_chars", 12_000), "max_chars", 1, _MAX_OUTPUT_CHARS)
        start = _bounded_int(args.get("start_line", 1), "start_line", 1, 1_000_000)
        end = _bounded_int(args.get("end_line", start + 399), "end_line", start, 1_000_000)
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"file is not UTF-8 text: {path}") from exc
        lines = text.splitlines()
        selected = lines[start - 1 : end]
        content = "\n".join(f"{index}: {line}" for index, line in enumerate(selected, start=start))
        return {"path": path, "start_line": start, "end_line": min(end, len(lines)), "content": content[:max_chars]}

    def search(self, args: Mapping[str, object]) -> dict[str, object]:
        query = args.get("query")
        if not isinstance(query, str) or not query:
            raise ToolError("query must be a non-empty string")
        path = self._optional_path(args.get("path"))
        case_sensitive = bool(args.get("case_sensitive", False))
        max_results = _bounded_int(args.get("max_results", 100), "max_results", 1, _MAX_SEARCH_RESULTS)
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, object]] = []
        for file_path in self._files_for(path):
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle in haystack:
                    matches.append({"path": self._relative(file_path), "line": line_number, "text": line[:500]})
                    if len(matches) >= max_results:
                        return {"query": query, "matches": matches, "truncated": True}
        return {"query": query, "matches": matches, "truncated": False}

    def git_status(self, args: Mapping[str, object]) -> dict[str, object]:
        del args
        completed = self._git("status", "--short", "--untracked-files=all")
        lines = [line for line in completed.stdout.splitlines() if self._git_line_is_safe(line)]
        return {"status": "\n".join(lines), "exit_code": completed.returncode}

    def git_diff(self, args: Mapping[str, object]) -> dict[str, object]:
        paths = args.get("paths")
        if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)) or not paths:
            raise ToolError("paths must be a non-empty list")
        normalized = tuple(self._required_path(path) for path in paths)
        for path in normalized:
            self._safe_path(path, must_exist=False)
        command = ["diff"]
        if bool(args.get("staged", False)):
            command.append("--cached")
        command.extend(["--no-ext-diff", "--", *normalized])
        completed = self._git(*command)
        return {"diff": completed.stdout[:_MAX_OUTPUT_CHARS], "exit_code": completed.returncode, "truncated": len(completed.stdout) > _MAX_OUTPUT_CHARS}

    def _safe_path(self, path: str, *, must_exist: bool) -> Path:
        normalized = self._required_path(path)
        if self._is_denied(normalized) or not self._is_allowed(normalized):
            raise ToolError(f"path is outside the readable workspace scope: {normalized}")
        candidate = self.root.joinpath(*normalized.split("/"))
        current = self.root
        for part in normalized.split("/"):
            current = current / part
            if current.is_symlink():
                raise ToolError(f"symlink traversal is not allowed: {normalized}")
        resolved = candidate.resolve(strict=False)
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolError(f"path escapes the workspace root: {normalized}")
        if must_exist and not resolved.exists():
            raise ToolError(f"path does not exist: {normalized}")
        return resolved

    def _roots_for(self, path: str) -> tuple[Path, ...]:
        if path:
            return (self._safe_path(path, must_exist=True),)
        if self.allowed_paths:
            return tuple(self._safe_path(item, must_exist=True) for item in self.allowed_paths)
        return (self.root,)

    def _files_for(self, path: str) -> tuple[Path, ...]:
        files: list[Path] = []
        for root in self._roots_for(path):
            if root.is_file():
                files.append(root)
            else:
                files.extend(current / filename for current, _, filenames in self._walk(root) for filename in filenames)
        return tuple(files)

    def _walk(self, root: Path):
        import os

        for current, directories, files in os.walk(root, followlinks=False):
            directories[:] = [directory for directory in directories if not self._is_denied(Path(current) / directory)]
            files[:] = [filename for filename in files if not self._is_denied(Path(current) / filename)]
            yield Path(current), sorted(directories), sorted(files)

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ToolError("git executable is not available") from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolError("git command timed out") from exc

    def _git_line_is_safe(self, line: str) -> bool:
        path = line[3:].strip() if len(line) >= 3 else ""
        return not path or not self._is_denied(path.strip('"'))

    def _required_path(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ToolError("path must be a non-empty relative path")
        try:
            return normalize_scoped_path(value, "path")
        except WorkspaceError as exc:
            raise ToolError(str(exc)) from exc

    def _optional_path(self, value: object) -> str:
        if value is None or value == "":
            return ""
        return self._required_path(value)

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def _is_allowed(self, path: str) -> bool:
        return not self.allowed_paths or any(path == allowed or path.startswith(allowed + "/") for allowed in self.allowed_paths)

    def _is_denied(self, path: str | Path) -> bool:
        parts = Path(path).parts
        if any(part in _DENIED_DIRECTORY_NAMES for part in parts):
            return True
        name = parts[-1] if parts else ""
        return name in _DENIED_FILE_NAMES or any(fnmatch.fnmatch(name, pattern) for pattern in _DENIED_FILE_PATTERNS)


class ControlledWorkspaceTools(ReadOnlyWorkspaceTools):
    """Read-only tools plus explicitly approval-gated workspace mutations."""

    def __init__(
        self,
        root: str | Path,
        *,
        allowed_paths: Sequence[str] = (),
        approval: ApprovalCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        denied_paths: Sequence[str] = (),
    ) -> None:
        super().__init__(root, allowed_paths=allowed_paths)
        if approval is not None and approval_callback is not None and approval is not approval_callback:
            raise ToolError("set only one of approval or approval_callback")
        callback = approval_callback if approval_callback is not None else approval
        if callback is not None and not callable(callback):
            raise ToolError("approval_callback must be callable or None")
        self.approval_callback = callback
        try:
            self.denied_paths = tuple(normalize_scoped_path(path, "denied_path") for path in denied_paths)
        except WorkspaceError as exc:
            raise ToolError(str(exc)) from exc

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return READ_ONLY_TOOL_SPECS + CONTROLLED_TOOL_SPECS

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        if name == "workspace.patch":
            return self.apply_patch(dict(arguments or {}))
        if name == "shell.exec":
            return self.exec_shell(dict(arguments or {}))
        return super().call(name, arguments)

    def apply_patch(self, args: Mapping[str, object]) -> ToolResult:
        patch = args.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return ToolResult(name="workspace.patch", ok=False, error="patch must be a non-empty unified diff")
        if not self.allowed_paths:
            return ToolResult(name="workspace.patch", ok=False, error="patch requires explicit allowed paths")
        try:
            validation = validate_patch_paths(
                patch,
                allowed_paths=self.allowed_paths,
                denied_paths=self.denied_paths,
            )
            if not validation.accepted:
                return ToolResult(
                    name="workspace.patch",
                    ok=False,
                    error="patch failed path scope validation",
                    output={"validation": validation.as_dict()},
                )
            for path in validation.paths:
                self._safe_path(path, must_exist=False)
        except (ToolError, WorkspaceError, ValueError) as exc:
            return ToolResult(name="workspace.patch", ok=False, error=str(exc))

        if not self._approve(
            "workspace.patch",
            {"paths": list(validation.paths), "patch": patch, "validation": validation.as_dict()},
        ):
            return ToolResult(name="workspace.patch", ok=False, error="approval denied")

        try:
            check = subprocess.run(
                ["git", "apply", "--check", "--whitespace=nowarn", "-"],
                cwd=self.root,
                input=patch,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            if check.returncode != 0:
                return ToolResult(
                    name="workspace.patch",
                    ok=False,
                    error="git apply check failed",
                    output={"stderr": check.stderr[:_MAX_OUTPUT_CHARS]},
                )
            applied = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=self.root,
                input=patch,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(name="workspace.patch", ok=False, error=f"patch application failed: {type(exc).__name__}")
        if applied.returncode != 0:
            return ToolResult(
                name="workspace.patch",
                ok=False,
                error="git apply failed",
                output={"stderr": applied.stderr[:_MAX_OUTPUT_CHARS]},
            )
        return ToolResult(
            name="workspace.patch",
            ok=True,
            output={"applied": True, "paths": list(validation.paths)},
        )

    def exec_shell(self, args: Mapping[str, object]) -> ToolResult:
        argv = args.get("argv")
        try:
            normalized_argv = _normalize_argv(argv)
            timeout = _bounded_int(args.get("timeout_seconds", 30), "timeout_seconds", 1, 120)
        except ToolError as exc:
            return ToolResult(name="shell.exec", ok=False, error=str(exc))
        if not self._approve(
            "shell.exec",
            {"argv": list(normalized_argv), "cwd": ".", "timeout_seconds": timeout},
        ):
            return ToolResult(name="shell.exec", ok=False, error="approval denied")
        try:
            completed = subprocess.run(
                list(normalized_argv),
                cwd=self.root,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(name="shell.exec", ok=False, error=f"command failed: {type(exc).__name__}")
        return ToolResult(
            name="shell.exec",
            ok=completed.returncode == 0,
            output={
                "argv": list(normalized_argv),
                "exit_code": completed.returncode,
                "stdout": completed.stdout[:_MAX_OUTPUT_CHARS],
                "stderr": completed.stderr[:_MAX_OUTPUT_CHARS],
                "truncated": len(completed.stdout) > _MAX_OUTPUT_CHARS or len(completed.stderr) > _MAX_OUTPUT_CHARS,
            },
            error=None if completed.returncode == 0 else "command returned a non-zero exit code",
        )

    def _approve(self, operation: str, details: Mapping[str, object]) -> bool:
        if self.approval_callback is None:
            return False
        try:
            return self.approval_callback(operation, details) is True
        except Exception:
            return False


def _bounded_int(value: object, field_name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ToolError(f"{field_name} must be an integer from {minimum} to {maximum}")
    return value


def _normalize_argv(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ToolError("argv must be a non-empty list of strings")
    if len(value) > 32 or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ToolError("argv must contain 1 to 32 non-empty strings")
    return tuple(item.strip() for item in value)


__all__ = [
    "ApprovalCallback",
    "CONTROLLED_TOOL_SPECS",
    "ControlledWorkspaceTools",
    "READ_ONLY_TOOL_SPECS",
    "ReadOnlyWorkspaceTools",
    "ToolError",
    "ToolExecutor",
    "ToolResult",
    "ToolSpec",
]
