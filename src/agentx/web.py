"""Shared public-web access service for AgentX front ends.

The service is deliberately independent of a terminal, HTTP handler, or UI.
Those surfaces provide an approval callback and consume the same bounded tool
contract.  ``WebResearchTools`` remains the compatibility implementation for
existing integrations while callers migrate to this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .tools import ToolError, ToolResult, ToolSpec, WebResearchTools


class WebAccessError(RuntimeError):
    """Raised when a shared web operation cannot produce a valid result."""


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

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """Return the model-facing tools enabled for this service instance."""

        return tuple(self._research_tools.specs)

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        """Execute a model-facing web tool through the shared service."""

        return self._research_tools.call(name, arguments)

    def search(self, query: str, *, max_results: int = 5) -> dict[str, object]:
        """Search the public web and return a normalized bounded payload."""

        return self._require_mapping(self.call("web_search", {"query": query, "max_results": max_results}))

    def fetch(self, url: str, *, max_chars: int = 4_000) -> dict[str, object]:
        """Fetch a public HTTPS page and return bounded extracted text."""

        return self._require_mapping(self.call("web_fetch", {"url": url, "max_chars": max_chars}))

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


__all__ = ["WebAccessError", "WebAccessService", "WebUIAdapter"]
