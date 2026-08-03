"""Policy-bounded read-only workspace tools for AgentX providers."""

from __future__ import annotations

import fnmatch
import html
import http.client
import ipaddress
import json
import shutil
import socket
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
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


class CompositeToolExecutor:
    """Compose disjoint tool providers behind one model-facing boundary."""

    def __init__(self, *executors: ToolExecutor) -> None:
        self._executors = tuple(executors)
        specs: list[ToolSpec] = []
        names: set[str] = set()
        for executor in self._executors:
            if not hasattr(executor, "specs") or not callable(getattr(executor, "call", None)):
                raise ToolError("composite entries must provide specs and call()")
            for spec in executor.specs:
                if not isinstance(spec, ToolSpec):
                    raise ToolError("composite entries must expose ToolSpec values")
                if spec.name in names:
                    raise ToolError(f"duplicate tool name: {spec.name}")
                names.add(spec.name)
                specs.append(spec)
        self._specs = tuple(specs)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return self._specs

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        for executor in self._executors:
            if any(spec.name == name for spec in executor.specs):
                return executor.call(name, arguments)
        return ToolResult(name=name, ok=False, error="unknown tool")


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
_MAX_WEB_QUERY_CHARS = 500
_MAX_WEB_RESULTS = 5
_MAX_WEB_RESULT_CHARS = 500
_MAX_WEB_FETCH_CHARS = 6_000
_MAX_WEB_RESPONSE_BYTES = 512_000
_MAX_WEB_REDIRECTS = 3
_WEB_TIMEOUT_SECONDS = 15
_WEB_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
_WEB_SEARCH_SOURCE = "DuckDuckGo Search"
_WEB_SEARCH_FALLBACK_ENDPOINT = "https://search.brave.com/search"
_WEB_SEARCH_FALLBACK_SOURCE = "Brave Search"


READ_ONLY_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "workspace_tree",
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
        "workspace_read",
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
        "workspace_search",
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
        "git_status",
        "Show safe, read-only Git status for the scoped workspace.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "git_diff",
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
        "workspace_patch",
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
        "shell_exec",
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


TEST_RUN_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "test_run",
        "Run an approved cross-platform test command profile without a shell.",
        {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "enum": ["auto", "python-unittest", "python-pytest", "npm-test"],
                },
                "target": {"type": "string", "description": "Optional test target, path, or module."},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
            },
        },
    ),
)


GIT_COMMIT_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "git_add",
        "Stage explicitly approved relative workspace paths for a local Git commit.",
        {
            "type": "object",
            "required": ["paths"],
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 64},
            },
        },
    ),
    ToolSpec(
        "git_commit",
        "Create an explicitly approved local Git commit. This tool never pushes.",
        {
            "type": "object",
            "required": ["message"],
            "properties": {
                "message": {"type": "string", "minLength": 1, "maxLength": 200},
            },
        },
    ),
)


WEB_RESEARCH_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "web_search",
        "Search the public web for current information when the user did not name a website or URL. Uses DuckDuckGo. Call this tool directly when needed: the client independently asks the user to approve the exact query after you request it, so do not ask for or wait for approval yourself.",
        {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": _MAX_WEB_QUERY_CHARS},
                "max_results": {"type": "integer", "minimum": 1, "maximum": _MAX_WEB_RESULTS},
            },
        },
    ),
    ToolSpec(
        "web_fetch",
        "Fetch a public HTTPS page as bounded plain text. Use this when the user names a specific website or URL. Call this tool directly when needed: the client independently asks the user to approve the exact URL after you request it, so do not ask for or wait for approval yourself.",
        {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string", "minLength": 1, "maxLength": 2048},
                "max_chars": {"type": "integer", "minimum": 500, "maximum": _MAX_WEB_FETCH_CHARS},
            },
        },
    ),
    ToolSpec(
        "web_fetch_document",
        "Fetch and extract a bounded public HTTPS text, JSON, HTML, or PDF document. PDF support is optional.",
        {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string", "minLength": 1, "maxLength": 2048},
                "max_chars": {"type": "integer", "minimum": 500, "maximum": 12000},
                "max_pages": {"type": "integer", "minimum": 1, "maximum": 32},
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
        name = {
            "workspace.tree": "workspace_tree",
            "workspace.read": "workspace_read",
            "workspace.search": "workspace_search",
            "git.status": "git_status",
            "git.diff": "git_diff",
        }.get(name, name)
        try:
            output = {
                "workspace_tree": self.tree,
                "workspace_read": self.read,
                "workspace_search": self.search,
                "git_status": self.git_status,
                "git_diff": self.git_diff,
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


class WebResearchTools:
    """Approval-gated, bounded public-web research tools.

    These tools are deliberately separate from workspace tools. They are only
    advertised when an approval callback is supplied, so non-interactive runs
    cannot accidentally send prompts or URLs to a third party.
    """

    def __init__(
        self,
        *,
        approval_callback: ApprovalCallback | None = None,
        opener: Callable[..., object] | None = None,
        resolver: Callable[..., object] | None = None,
        document_extractor: Any = None,
    ) -> None:
        if approval_callback is not None and not callable(approval_callback):
            raise ToolError("approval_callback must be callable or None")
        if opener is not None and not callable(opener):
            raise ToolError("opener must be callable or None")
        if resolver is not None and not callable(resolver):
            raise ToolError("resolver must be callable or None")
        self.approval_callback = approval_callback
        self._opener = opener
        self._resolver = resolver or socket.getaddrinfo
        self._document_extractor = document_extractor

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return WEB_RESEARCH_TOOL_SPECS if self.approval_callback is not None else ()

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        args = dict(arguments or {})
        name = {
            "web.search": "web_search",
            "web.fetch": "web_fetch",
            "web.fetch_document": "web_fetch_document",
        }.get(name, name)
        try:
            output = {
                "web_search": self.search,
                "web_fetch": self.fetch,
                "web_fetch_document": self.fetch_document,
            }[name](args)
        except (KeyError, ToolError, OSError, UnicodeError, ValueError) as exc:
            return ToolResult(name=name, ok=False, error=str(exc))
        return ToolResult(name=name, ok=True, output=output)

    def search(self, args: Mapping[str, object]) -> dict[str, object]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        query = query.strip()
        if len(query) > _MAX_WEB_QUERY_CHARS:
            raise ToolError(f"query must be at most {_MAX_WEB_QUERY_CHARS} characters")
        max_results = _bounded_int(args.get("max_results", 5), "max_results", 1, _MAX_WEB_RESULTS)
        if not self._approve(
            "web.search",
            {
                "query": query,
                "source": _WEB_SEARCH_SOURCE,
                "max_results": max_results,
            },
        ):
            raise ToolError("approval denied")

        search_result, duckduckgo_challenge = self._search_with_provider(
            endpoint=_WEB_SEARCH_ENDPOINT,
            source=_WEB_SEARCH_SOURCE,
            query=query,
            max_results=max_results,
        )
        if duckduckgo_challenge:
            if not self._approve(
                "web.search",
                {
                    "query": query,
                    "source": _WEB_SEARCH_FALLBACK_SOURCE,
                    "max_results": max_results,
                    "reason": "DuckDuckGo requested a bot challenge",
                },
            ):
                raise ToolError("DuckDuckGo requested a bot challenge and Brave Search fallback was not approved")
            search_result, _ = self._search_with_provider(
                endpoint=_WEB_SEARCH_FALLBACK_ENDPOINT,
                source=_WEB_SEARCH_FALLBACK_SOURCE,
                query=query,
                max_results=max_results,
            )
        return search_result

    def _search_with_provider(
        self,
        *,
        endpoint: str,
        source: str,
        query: str,
        max_results: int,
    ) -> tuple[dict[str, object], bool]:
        url = endpoint + "?" + urllib.parse.urlencode({"q": query})
        _, content_type, body, source_truncated = self._open_public_https(url)
        if content_type and not _is_text_content_type(content_type):
            raise ToolError("search service returned a non-text response")
        source_text = _decode_web_text(body)
        parser = _SearchResultParser()
        parser.feed(source_text)
        parser.close()

        results = []
        for result in parser.results[:max_results]:
            title = _clip_text(result["title"], _MAX_WEB_RESULT_CHARS)
            href = _unwrap_search_result_url(result["url"])
            if not title or not href:
                continue
            item: dict[str, str] = {"title": title, "url": href}
            snippet = _clip_text(result.get("snippet", ""), _MAX_WEB_RESULT_CHARS)
            if snippet:
                item["snippet"] = snippet
            results.append(item)

        return (
            {
                "query": query,
                "results": results,
                "source": source,
                "truncated": source_truncated or len(parser.results) > max_results,
            },
            _is_duckduckgo_challenge(source_text) if source == _WEB_SEARCH_SOURCE else False,
        )

    def fetch(self, args: Mapping[str, object]) -> dict[str, object]:
        url = _normalize_public_https_url(args.get("url"))
        max_chars = _bounded_int(args.get("max_chars", 4_000), "max_chars", 500, _MAX_WEB_FETCH_CHARS)
        if not self._approve(
            "web.fetch",
            {"url": url, "max_chars": max_chars},
        ):
            raise ToolError("approval denied")

        final_url, content_type, body, source_truncated = self._open_public_https(url)
        if content_type and not _is_text_content_type(content_type):
            raise ToolError("web_fetch only accepts text or JSON responses")
        source_text = _decode_web_text(body)
        if content_type.startswith("text/html") or content_type.startswith("application/xhtml") or not content_type:
            parser = _WebDocumentParser()
            parser.feed(source_text)
            parser.close()
            title = _clip_text(parser.title, 300)
            text = parser.text
        else:
            title = ""
            text = source_text
        text = _compact_text(text)
        truncated = source_truncated or len(text) > max_chars
        payload: dict[str, object] = {
            "url": final_url,
            "content": text[:max_chars],
            "truncated": truncated,
        }
        if title:
            payload["title"] = title
        return payload

    def fetch_document(self, args: Mapping[str, object]) -> dict[str, object]:
        from .documents import DocumentExtractor, DocumentExtractionError

        url = _normalize_public_https_url(args.get("url"))
        max_chars = _bounded_int(args.get("max_chars", 12_000), "max_chars", 500, 12_000)
        max_pages = _bounded_int(args.get("max_pages", 32), "max_pages", 1, 32)
        if not self._approve(
            "web.fetch_document",
            {"url": url, "max_chars": max_chars, "max_pages": max_pages},
        ):
            raise ToolError("approval denied")
        final_url, content_type, body, source_truncated = self._open_public_https(url)
        extractor = self._document_extractor or DocumentExtractor(max_chars=max_chars, max_pages=max_pages)
        try:
            document = extractor.extract(
                body,
                media_type=content_type or "application/octet-stream",
                filename=urllib.parse.urlsplit(final_url).path,
            )
        except DocumentExtractionError as exc:
            raise ToolError(str(exc)) from exc
        payload = document.as_dict()
        payload["url"] = final_url
        payload["truncated"] = bool(payload.get("truncated")) or source_truncated
        return payload

    def _approve(self, operation: str, details: Mapping[str, object]) -> bool:
        if self.approval_callback is None:
            return False
        try:
            return self.approval_callback(operation, details) is True
        except Exception:
            return False

    def _open_public_https(self, url: str) -> tuple[str, str, bytes, bool]:
        current_url = url
        for redirect_count in range(_MAX_WEB_REDIRECTS + 1):
            hostname, addresses = _validate_public_https_url(current_url, resolver=self._resolver)
            request = urllib.request.Request(
                current_url,
                headers={
                    "Accept": "text/html, text/plain, application/json;q=0.9",
                    "User-Agent": "AgentX-WebResearch/0.1",
                },
                method="GET",
            )
            try:
                response = (
                    self._opener(request, timeout=_WEB_TIMEOUT_SECONDS)
                    if self._opener is not None
                    else _open_pinned_https(request, hostname, addresses, timeout=_WEB_TIMEOUT_SECONDS)
                )
            except urllib.error.HTTPError as exc:
                response = exc
            except urllib.error.URLError as exc:
                raise ToolError(f"web request failed: {_safe_web_error_reason(exc)}") from exc
            except (socket.timeout, TimeoutError) as exc:
                raise ToolError("web request timed out") from exc

            status = _response_status(response)
            if status in {301, 302, 303, 307, 308}:
                location = _response_header(response, "Location")
                _close_response(response)
                if redirect_count >= _MAX_WEB_REDIRECTS:
                    raise ToolError(f"web request exceeded {_MAX_WEB_REDIRECTS} redirects")
                if not location:
                    raise ToolError("web redirect did not include a location")
                current_url = urllib.parse.urljoin(current_url, location)
                continue
            if not 200 <= status < 300:
                _close_response(response)
                raise ToolError(f"web request returned HTTP {status}")
            try:
                body, source_truncated = _read_web_response(response)
                content_type = _response_header(response, "Content-Type").split(";", 1)[0].strip().lower()
            finally:
                _close_response(response)
            return current_url, content_type, body, source_truncated
        raise ToolError(f"web request exceeded {_MAX_WEB_REDIRECTS} redirects")


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to a prevalidated address with hostname TLS."""

    def __init__(self, hostname: str, address: str, *, timeout: float) -> None:
        super().__init__(hostname, port=443, timeout=timeout, context=ssl.create_default_context())
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._address, self.port), self.timeout)
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


class _PinnedResponse:
    def __init__(self, response: http.client.HTTPResponse, connection: _PinnedHTTPSConnection) -> None:
        self._response = response
        self._connection = connection
        self.status = response.status
        self.headers = response.headers

    def read(self, amount: int = -1) -> bytes:
        return self._response.read(amount)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()


def _open_pinned_https(
    request: urllib.request.Request,
    hostname: str,
    addresses: Sequence[str],
    *,
    timeout: float,
) -> _PinnedResponse:
    parsed = urllib.parse.urlsplit(request.full_url)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    headers = dict(request.header_items())
    last_error: OSError | None = None
    for address in addresses:
        connection = _PinnedHTTPSConnection(hostname, address, timeout=timeout)
        try:
            connection.request(request.get_method(), target, headers=headers)
            return _PinnedResponse(connection.getresponse(), connection)
        except OSError as exc:
            last_error = exc
            connection.close()
    raise ToolError(f"could not connect to public hostname: {type(last_error).__name__}")


def _normalize_public_https_url(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError("url must be a non-empty public HTTPS URL")
    url = value.strip()
    if len(url) > 2048:
        raise ToolError("url must be at most 2048 characters")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ToolError("url must use public HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ToolError("url must not include credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ToolError("url has an invalid port") from exc
    if port not in {None, 443}:
        raise ToolError("url must use the default HTTPS port")
    if not parsed.hostname:
        raise ToolError("url must include a hostname")
    return urllib.parse.urlunsplit(("https", parsed.netloc, parsed.path or "/", parsed.query, ""))


def _validate_public_https_url(
    url: str,
    *,
    resolver: Callable[..., object],
) -> tuple[str, tuple[str, ...]]:
    normalized = _normalize_public_https_url(url)
    hostname = urllib.parse.urlsplit(normalized).hostname
    if not hostname:
        raise ToolError("url must include a hostname")
    hostname = hostname.rstrip(".")
    if hostname.lower() == "localhost" or hostname.lower().endswith(".localhost"):
        raise ToolError("localhost URLs are not allowed")
    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        if not literal_ip.is_global:
            raise ToolError("non-public IP addresses are not allowed")
        return hostname, (str(literal_ip),)
    try:
        addresses = resolver(hostname, 443, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror) as exc:
        raise ToolError("could not resolve public hostname") from exc
    resolved_ips = {entry[4][0].split("%", 1)[0] for entry in addresses if len(entry) >= 5 and entry[4]}
    if not resolved_ips:
        raise ToolError("could not resolve public hostname")
    for address in resolved_ips:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ToolError("hostname resolved to an invalid address") from exc
        if not ip.is_global:
            raise ToolError("hostname resolves to a non-public IP address")
    return hostname, tuple(sorted(resolved_ips))


def _response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if not isinstance(status, int):
        getcode = getattr(response, "getcode", None)
        status = getcode() if callable(getcode) else None
    if not isinstance(status, int):
        raise ToolError("web response did not include an HTTP status")
    return status


def _response_header(response: object, name: str) -> str:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    value = getter(name) if callable(getter) else None
    return value.strip() if isinstance(value, str) else ""


def _read_web_response(response: object) -> tuple[bytes, bool]:
    reader = getattr(response, "read", None)
    if not callable(reader):
        raise ToolError("web response is not readable")
    body = reader(_MAX_WEB_RESPONSE_BYTES + 1)
    if not isinstance(body, bytes):
        raise ToolError("web response body is not bytes")
    return body[:_MAX_WEB_RESPONSE_BYTES], len(body) > _MAX_WEB_RESPONSE_BYTES


def _close_response(response: object) -> None:
    closer = getattr(response, "close", None)
    if callable(closer):
        closer()


def _safe_web_error_reason(exc: urllib.error.URLError) -> str:
    reason = exc.reason
    if isinstance(reason, str):
        return reason[:300]
    return type(reason).__name__


def _is_text_content_type(content_type: str) -> bool:
    return content_type.startswith("text/") or content_type in {
        "application/json",
        "application/xhtml+xml",
    }


def _decode_web_text(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _compact_text(value: str) -> str:
    return " ".join(html.unescape(value).split())


def _clip_text(value: str, maximum: int) -> str:
    compact = _compact_text(value)
    return compact[:maximum].rstrip()


def _unwrap_search_result_url(value: str) -> str:
    href = html.unescape(value).strip()
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlsplit(href)
    if parsed.hostname and parsed.hostname.lower().endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        redirected = urllib.parse.parse_qs(parsed.query).get("uddg", [])
        if redirected and isinstance(redirected[0], str):
            return redirected[0]
    return href


def _is_duckduckgo_challenge(value: str) -> bool:
    lowered = value.casefold()
    return "anomaly-modal" in lowered or "bots use duckduckgo" in lowered


class _SearchResultParser(HTMLParser):
    _VOID_TAGS = frozenset(
        {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._anchor: dict[str, str] | None = None
        self._capture_snippet = False
        self._snippet_parts: list[str] = []
        self._depth = 0
        self._brave_record: dict[str, str] | None = None
        self._brave_block_depth: int | None = None
        self._brave_capture: str | None = None
        self._brave_capture_depth: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if (
            tag == "div"
            and self._brave_record is None
            and "snippet" in classes
            and attributes.get("data-type") == "web"
        ):
            self._brave_record = {"url": "", "title": "", "snippet": ""}
            self._brave_block_depth = self._depth
        if self._brave_record is not None:
            if tag == "a" and not self._brave_record["url"] and attributes.get("href"):
                self._brave_record["url"] = attributes["href"] or ""
            if "search-snippet-title" in classes:
                self._brave_capture = "title"
                self._brave_capture_depth = self._depth
            elif "generic-snippet" in classes:
                self._brave_capture = "snippet"
                self._brave_capture_depth = self._depth
        if tag == "a" and "result__a" in classes:
            self._anchor = {"url": attributes.get("href") or "", "title": ""}
        elif "result__snippet" in classes and self.results:
            self._capture_snippet = True
            self._snippet_parts = []
        if tag not in self._VOID_TAGS:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor is not None:
            self.results.append(self._anchor)
            self._anchor = None
        elif self._capture_snippet and tag in {"a", "div", "span"}:
            self._capture_snippet = False
            if self.results:
                self.results[-1]["snippet"] = " ".join(self._snippet_parts)
        if self._brave_capture is not None and self._brave_capture_depth is not None:
            if self._depth == self._brave_capture_depth + 1:
                self._brave_capture = None
                self._brave_capture_depth = None
        if tag not in self._VOID_TAGS:
            self._depth = max(0, self._depth - 1)
        if self._brave_record is not None and self._brave_block_depth == self._depth:
            if self._brave_record["title"] and self._brave_record["url"]:
                self.results.append(self._brave_record)
            self._brave_record = None
            self._brave_block_depth = None

    def handle_data(self, data: str) -> None:
        if self._brave_record is not None and self._brave_capture is not None:
            self._brave_record[self._brave_capture] += data
        elif self._anchor is not None:
            self._anchor["title"] += data
        elif self._capture_snippet:
            self._snippet_parts.append(data)


class _WebDocumentParser(HTMLParser):
    _SUPPRESSED_TAGS = frozenset({"script", "style", "noscript", "svg", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._suppressed_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._parts: list[str] = []

    @property
    def title(self) -> str:
        return " ".join(self._title_parts)

    @property
    def text(self) -> str:
        return " ".join(self._parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in self._SUPPRESSED_TAGS:
            self._suppressed_depth += 1
        elif tag == "title" and not self._suppressed_depth:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SUPPRESSED_TAGS and self._suppressed_depth:
            self._suppressed_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._suppressed_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._parts.append(data)


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
        enable_patch: bool = True,
        enable_shell: bool = True,
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
        self.enable_patch = bool(enable_patch)
        self.enable_shell = bool(enable_shell)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        specs = list(READ_ONLY_TOOL_SPECS)
        if self.enable_patch:
            specs.append(CONTROLLED_TOOL_SPECS[0])
        if self.enable_shell:
            specs.append(CONTROLLED_TOOL_SPECS[1])
        return tuple(specs)

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        if name in {"workspace.patch", "workspace_patch"}:
            if not self.enable_patch:
                return ToolResult(name="workspace.patch", ok=False, error="patch tool is disabled")
            return self.apply_patch(dict(arguments or {}))
        if name in {"shell.exec", "shell_exec"}:
            if not self.enable_shell:
                return ToolResult(name="shell.exec", ok=False, error="shell tool is disabled")
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


class TestRunTools:
    """Execute approved test command profiles through argv-only subprocess calls."""

    _PROFILES = frozenset({"auto", "python-unittest", "python-pytest", "npm-test"})

    def __init__(
        self,
        root: str | Path,
        *,
        approval_callback: ApprovalCallback | None = None,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if approval_callback is not None and not callable(approval_callback):
            raise ToolError("approval_callback must be callable or None")
        self.approval_callback = approval_callback
        self.runner = runner or subprocess.run

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return TEST_RUN_TOOL_SPECS if self.approval_callback is not None else ()

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        if name not in {"test.run", "test_run"}:
            return ToolResult(name=name, ok=False, error="unknown tool")
        return self.run_tests(dict(arguments or {}))

    def run_tests(self, args: Mapping[str, object]) -> ToolResult:
        try:
            profile = _test_profile(args.get("profile", "auto"))
            target = _optional_test_target(args.get("target"))
            timeout = _bounded_int(args.get("timeout_seconds", 60), "timeout_seconds", 1, 120)
            resolved_profile, argv = self._argv_for_profile(profile, target)
        except ToolError as exc:
            return ToolResult(name="test.run", ok=False, error=str(exc))

        details = {
            "profile": resolved_profile,
            "target": target or "",
            "argv": list(argv),
            "cwd": ".",
            "timeout_seconds": timeout,
        }
        if self.approval_callback is None or self.approval_callback("test.run", details) is not True:
            return ToolResult(name="test.run", ok=False, error="approval denied")

        try:
            completed = self.runner(
                list(argv),
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
            return ToolResult(name="test.run", ok=False, error=f"test command failed: {type(exc).__name__}")

        return ToolResult(
            name="test.run",
            ok=completed.returncode == 0,
            output={
                "profile": resolved_profile,
                "argv": list(argv),
                "exit_code": completed.returncode,
                "stdout": completed.stdout[:_MAX_OUTPUT_CHARS],
                "stderr": completed.stderr[:_MAX_OUTPUT_CHARS],
                "truncated": len(completed.stdout) > _MAX_OUTPUT_CHARS or len(completed.stderr) > _MAX_OUTPUT_CHARS,
            },
            error=None if completed.returncode == 0 else "test command returned a non-zero exit code",
        )

    def _argv_for_profile(self, profile: str, target: str | None) -> tuple[str, tuple[str, ...]]:
        resolved_profile = self._detect_profile() if profile == "auto" else profile
        if resolved_profile == "python-unittest":
            if target:
                return resolved_profile, (sys.executable, "-B", "-m", "unittest", target)
            if (self.root / "tests").is_dir():
                return resolved_profile, (sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests")
            return resolved_profile, (sys.executable, "-B", "-m", "unittest", "discover")
        if resolved_profile == "python-pytest":
            argv = [sys.executable, "-m", "pytest"]
            if target:
                argv.append(target)
            return resolved_profile, tuple(argv)
        if resolved_profile == "npm-test":
            npm = shutil.which("npm.cmd") or shutil.which("npm")
            if not npm:
                raise ToolError("npm executable was not found")
            argv = [npm, "test"]
            if target:
                argv.extend(("--", target))
            return resolved_profile, tuple(argv)
        raise ToolError(f"unsupported test profile '{resolved_profile}'")

    def _detect_profile(self) -> str:
        package_json = self.root / "package.json"
        if package_json.exists() and not (self.root / "tests").exists():
            return "npm-test"
        if (self.root / "pytest.ini").exists() or (self.root / "conftest.py").exists():
            return "python-pytest"
        pyproject = self.root / "pyproject.toml"
        if pyproject.exists():
            try:
                text = pyproject.read_text(encoding="utf-8", errors="replace")[:20_000]
            except OSError:
                text = ""
            if "pytest" in text or "[tool.pytest" in text:
                return "python-pytest"
            return "python-unittest"
        return "python-unittest"


class GitCommitTools:
    """Approval-gated local Git staging and commit tools. Never pushes."""

    def __init__(
        self,
        root: str | Path,
        *,
        approval_callback: ApprovalCallback | None = None,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if approval_callback is not None and not callable(approval_callback):
            raise ToolError("approval_callback must be callable or None")
        self.approval_callback = approval_callback
        self.runner = runner or subprocess.run

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return GIT_COMMIT_TOOL_SPECS if self.approval_callback is not None else ()

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        if name in {"git.add", "git_add"}:
            return self.add(dict(arguments or {}))
        if name in {"git.commit", "git_commit"}:
            return self.commit(dict(arguments or {}))
        return ToolResult(name=name, ok=False, error="unknown tool")

    def add(self, args: Mapping[str, object]) -> ToolResult:
        try:
            paths = _git_add_paths(args.get("paths"))
        except ToolError as exc:
            return ToolResult(name="git.add", ok=False, error=str(exc))
        details = {"paths": list(paths), "argv": ["git", "add", "--", *paths], "cwd": "."}
        if self.approval_callback is None or self.approval_callback("git.add", details) is not True:
            return ToolResult(name="git.add", ok=False, error="approval denied")
        return self._run_git("git.add", ("git", "add", "--", *paths), extra={"paths": list(paths)})

    def commit(self, args: Mapping[str, object]) -> ToolResult:
        try:
            message = _git_commit_message(args.get("message"))
        except ToolError as exc:
            return ToolResult(name="git.commit", ok=False, error=str(exc))
        details = {"message": message, "argv": ["git", "commit", "-m", message], "cwd": "."}
        if self.approval_callback is None or self.approval_callback("git.commit", details) is not True:
            return ToolResult(name="git.commit", ok=False, error="approval denied")
        return self._run_git("git.commit", ("git", "commit", "-m", message), extra={"message": message})

    def _run_git(self, name: str, argv: Sequence[str], *, extra: Mapping[str, object]) -> ToolResult:
        try:
            completed = self.runner(
                list(argv),
                cwd=self.root,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(name=name, ok=False, error=f"git command failed: {type(exc).__name__}")
        output = {
            **dict(extra),
            "argv": list(argv),
            "exit_code": completed.returncode,
            "stdout": completed.stdout[:_MAX_OUTPUT_CHARS],
            "stderr": completed.stderr[:_MAX_OUTPUT_CHARS],
            "truncated": len(completed.stdout) > _MAX_OUTPUT_CHARS or len(completed.stderr) > _MAX_OUTPUT_CHARS,
        }
        return ToolResult(
            name=name,
            ok=completed.returncode == 0,
            output=output,
            error=None if completed.returncode == 0 else "git command returned a non-zero exit code",
        )


def _bounded_int(value: object, field_name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ToolError(f"{field_name} must be an integer from {minimum} to {maximum}")
    return value


def _test_profile(value: object) -> str:
    if not isinstance(value, str) or value.strip() not in TestRunTools._PROFILES:
        raise ToolError("profile must be one of: auto, python-unittest, python-pytest, npm-test")
    return value.strip()


def _optional_test_target(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolError("target must be a string when provided")
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > 500:
        raise ToolError("target is too long")
    return normalized


def _git_add_paths(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ToolError("paths must be a non-empty list of relative paths")
    if len(value) > 64:
        raise ToolError("paths must contain at most 64 entries")
    paths: list[str] = []
    for item in value:
        try:
            path = normalize_scoped_path(item, "path")
        except WorkspaceError as exc:
            raise ToolError(str(exc)) from exc
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def _git_commit_message(value: object) -> str:
    if not isinstance(value, str):
        raise ToolError("message must be a string")
    message = " ".join(value.strip().split())
    if not message:
        raise ToolError("message must be non-empty")
    if len(message) > 200:
        raise ToolError("message must be at most 200 characters")
    return message


def _normalize_argv(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ToolError("argv must be a non-empty list of strings")
    if len(value) > 32 or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ToolError("argv must contain 1 to 32 non-empty strings")
    return tuple(item.strip() for item in value)


__all__ = [
    "ApprovalCallback",
    "CompositeToolExecutor",
    "CONTROLLED_TOOL_SPECS",
    "ControlledWorkspaceTools",
    "GIT_COMMIT_TOOL_SPECS",
    "GitCommitTools",
    "READ_ONLY_TOOL_SPECS",
    "ReadOnlyWorkspaceTools",
    "ToolError",
    "ToolExecutor",
    "ToolResult",
    "ToolSpec",
    "TEST_RUN_TOOL_SPECS",
    "TestRunTools",
    "WEB_RESEARCH_TOOL_SPECS",
    "WebResearchTools",
]
