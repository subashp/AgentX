"""Shared public-web access service for AgentX front ends.

The service is deliberately independent of a terminal, HTTP handler, or UI.
Those surfaces provide an approval callback and consume the same bounded tool
contract.  ``WebResearchTools`` remains the compatibility implementation for
existing integrations while callers migrate to this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import threading
import time
from typing import Any

from .tools import ToolError, ToolResult, ToolSpec, WebResearchTools


class WebAccessError(RuntimeError):
    """Raised when a shared web operation cannot produce a valid result."""


@dataclass(frozen=True)
class Citation:
    """A compact source reference that can be rendered by any front end."""

    title: str
    url: str
    snippet: str = ""
    retrieved_at: str = ""

    def as_dict(self) -> dict[str, str]:
        result = {"title": self.title, "url": self.url}
        if self.snippet:
            result["snippet"] = self.snippet
        if self.retrieved_at:
            result["retrieved_at"] = self.retrieved_at
        return result


@dataclass(frozen=True)
class _CacheEntry:
    expires_at: float
    value: dict[str, object]


class WebCache:
    """Small in-process TTL/LRU cache for bounded public-web responses."""

    def __init__(self, *, max_entries: int = 32, max_chars: int = 100_000, ttl_seconds: float = 300.0, clock: Any = None) -> None:
        if isinstance(max_entries, bool) or not 1 <= max_entries <= 256:
            raise ValueError("max_entries must be from 1 to 256")
        if isinstance(max_chars, bool) or not 1_000 <= max_chars <= 2_000_000:
            raise ValueError("max_chars must be from 1000 to 2000000")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_entries = max_entries
        self.max_chars = max_chars
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock or time.monotonic
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, object] | None:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return dict(entry.value)

    def put(self, key: str, value: Mapping[str, object]) -> bool:
        candidate = dict(value)
        try:
            size = len(json.dumps(candidate, ensure_ascii=True, separators=(",", ":")))
        except (TypeError, ValueError):
            return False
        if size > self.max_chars:
            return False
        with self._lock:
            self._entries[key] = _CacheEntry(self._clock() + self.ttl_seconds, candidate)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class WebAccessService:
    """Provider-neutral public-web service shared by CLI and UI adapters.

    ``research_tools`` is injectable so UI and test adapters can share this
    service without making live network requests.  The default preserves the
    existing AgentX public-HTTPS, approval, redirect, and response limits.
    """

    def __init__(
        self,
        *,
        approval_callback: Any = None,
        opener: Any = None,
        resolver: Any = None,
        research_tools: Any = None,
        cache: WebCache | None = None,
        clock: Any = None,
    ) -> None:
        if research_tools is None:
            research_tools = WebResearchTools(
                approval_callback=approval_callback,
                opener=opener,
                resolver=resolver,
            )
        if not hasattr(research_tools, "specs") or not callable(getattr(research_tools, "call", None)):
            raise ToolError("research_tools must provide specs and call()")
        self._research_tools = research_tools
        self._approval_callback = approval_callback
        self._cache = cache
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """Return the model-facing tools enabled for this service instance."""

        return tuple(self._research_tools.specs)

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        """Execute a model-facing web tool through the shared service."""

        canonical_name = {"web.search": "web_search", "web.fetch": "web_fetch"}.get(name, name)
        args = dict(arguments or {})
        if canonical_name == "web_search":
            try:
                return ToolResult(name=canonical_name, ok=True, output=self.search(
                    args.get("query", ""), max_results=args.get("max_results", 5)
                ))
            except (TypeError, ValueError, WebAccessError, ToolError) as exc:
                return ToolResult(name=canonical_name, ok=False, error=str(exc))
        if canonical_name == "web_fetch":
            try:
                return ToolResult(name=canonical_name, ok=True, output=self.fetch(
                    args.get("url", ""), max_chars=args.get("max_chars", 4_000)
                ))
            except (TypeError, ValueError, WebAccessError, ToolError) as exc:
                return ToolResult(name=canonical_name, ok=False, error=str(exc))
        if canonical_name == "web_fetch_document":
            try:
                return ToolResult(name=canonical_name, ok=True, output=self.fetch_document(
                    args.get("url", ""),
                    max_chars=args.get("max_chars", 12_000),
                    max_pages=args.get("max_pages", 32),
                ))
            except (TypeError, ValueError, WebAccessError, ToolError) as exc:
                return ToolResult(name=canonical_name, ok=False, error=str(exc))
        return self._research_tools.call(name, arguments)

    def search(self, query: str, *, max_results: int = 5) -> dict[str, object]:
        """Search the public web and return a normalized bounded payload."""

        key = f"search:{query}:{max_results}"
        details = {"query": query, "max_results": max_results}
        cached = self._cached(key, "web.search", details)
        if cached is not None:
            return cached
        result = self._require_mapping(self._research_tools.call("web_search", details))
        normalized = self._with_search_citations(result)
        self._store(key, normalized)
        return normalized

    def fetch(self, url: str, *, max_chars: int = 4_000) -> dict[str, object]:
        """Fetch a public HTTPS page and return bounded extracted text."""

        key = f"fetch:{url}:{max_chars}"
        details = {"url": url, "max_chars": max_chars}
        cached = self._cached(key, "web.fetch", details)
        if cached is not None:
            return cached
        result = self._require_mapping(self._research_tools.call("web_fetch", details))
        normalized = self._with_fetch_citations(result)
        self._store(key, normalized)
        return normalized

    def fetch_document(self, url: str, *, max_chars: int = 12_000, max_pages: int = 32) -> dict[str, object]:
        """Fetch and extract a bounded public document with source metadata."""

        key = f"document:{url}:{max_chars}:{max_pages}"
        details = {"url": url, "max_chars": max_chars, "max_pages": max_pages}
        cached = self._cached(key, "web.fetch_document", details)
        if cached is not None:
            return cached
        result = self._require_mapping(self._research_tools.call("web_fetch_document", details))
        normalized = self._with_document_citations(result)
        self._store(key, normalized)
        return normalized

    def _cached(self, key: str, operation: str, details: Mapping[str, object]) -> dict[str, object] | None:
        if self._cache is None or self._approval_callback is None:
            return None
        cached = self._cache.get(key)
        if cached is None:
            return None
        if self._approval_callback(operation, details) is not True:
            raise WebAccessError("approval denied")
        return cached

    def _store(self, key: str, value: Mapping[str, object]) -> None:
        if self._cache is not None:
            self._cache.put(key, value)

    def _with_search_citations(self, result: Mapping[str, object]) -> dict[str, object]:
        retrieved_at = self._retrieved_at()
        normalized = dict(result)
        citations = []
        for item in result.get("results", ()):
            if isinstance(item, Mapping) and isinstance(item.get("url"), str):
                citations.append(Citation(
                    title=str(item.get("title", item["url"])),
                    url=item["url"],
                    snippet=str(item.get("snippet", "")),
                    retrieved_at=retrieved_at,
                ).as_dict())
        normalized["citations"] = citations
        normalized["retrieved_at"] = retrieved_at
        return normalized

    def _with_fetch_citations(self, result: Mapping[str, object]) -> dict[str, object]:
        normalized = dict(result)
        retrieved_at = self._retrieved_at()
        url = str(result.get("url", ""))
        title = str(result.get("title", url))
        normalized["citations"] = [Citation(title=title, url=url, retrieved_at=retrieved_at).as_dict()] if url else []
        normalized["retrieved_at"] = retrieved_at
        return normalized

    def _with_document_citations(self, result: Mapping[str, object]) -> dict[str, object]:
        normalized = dict(result)
        retrieved_at = self._retrieved_at()
        url = str(result.get("url", ""))
        title = str(result.get("title", url))
        normalized["citations"] = [Citation(title=title, url=url, retrieved_at=retrieved_at).as_dict()] if url else []
        normalized["retrieved_at"] = retrieved_at
        return normalized

    def _retrieved_at(self) -> str:
        value = self._clock()
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _require_mapping(result: ToolResult) -> dict[str, object]:
        if not result.ok:
            raise WebAccessError(result.error or "web operation failed")
        if not isinstance(result.output, Mapping):
            raise WebAccessError("web operation returned an invalid result")
        return dict(result.output)


class WebUIAdapter:
    """Translate the shared service into a UI/gateway-friendly contract.

    The adapter intentionally returns only tool status for progress events;
    fetched content remains in the model conversation and is not broadcast to
    the browser event stream by default.
    """

    def __init__(self, service: WebAccessService, *, browser: Any = None) -> None:
        if not isinstance(service, WebAccessService):
            raise ToolError("service must be a WebAccessService")
        if browser is not None and (not hasattr(browser, "specs") or not callable(getattr(browser, "call", None))):
            raise ToolError("browser must provide specs and call()")
        self.service = service
        self.browser = browser

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        specs = list(self.service.specs)
        if self.browser is not None:
            names = {spec.name for spec in specs}
            for spec in self.browser.specs:
                if spec.name in names:
                    raise ToolError(f"duplicate web UI tool name: {spec.name}")
                specs.append(spec)
        return tuple(specs)

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        if self.browser is not None and any(spec.name == name for spec in self.browser.specs):
            return self.browser.call(name, arguments)
        return self.service.call(name, arguments)

    @staticmethod
    def event(result: ToolResult) -> dict[str, object]:
        """Return the stable progress-event shape shared by UI transports."""

        return {"name": result.name, "ok": result.ok}


__all__ = ["Citation", "WebAccessError", "WebAccessService", "WebCache", "WebUIAdapter"]
