"""Optional, approval-gated browser tools for AgentX."""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .tools import ApprovalCallback, ToolError, ToolResult, ToolSpec


_MAX_BROWSER_TEXT = 12_000
_MAX_SELECTOR_CHARS = 500
_MAX_URL_CHARS = 2_048
_MAX_TIMEOUT_MS = 30_000
_BROWSER_TOOL_SPECS = (
    ToolSpec(
        "browser_open",
        "Start an optional Playwright browser session and open a public HTTP(S) URL.",
        {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string", "maxLength": _MAX_URL_CHARS},
                "headless": {"type": "boolean"},
                "channel": {"type": "string", "description": "Optional browser channel such as msedge or chrome."},
                "width": {"type": "integer", "minimum": 320, "maximum": 4_000},
                "height": {"type": "integer", "minimum": 240, "maximum": 4_000},
            },
        },
    ),
    ToolSpec(
        "browser_navigate",
        "Navigate the current browser session to a public HTTP(S) URL.",
        {"type": "object", "required": ["url"], "properties": {"url": {"type": "string", "maxLength": _MAX_URL_CHARS}}},
    ),
    ToolSpec(
        "browser_click",
        "Click a selector in the current browser page.",
        {
            "type": "object",
            "required": ["selector"],
            "properties": {
                "selector": {"type": "string", "maxLength": _MAX_SELECTOR_CHARS},
                "timeout_ms": {"type": "integer", "minimum": 1, "maximum": _MAX_TIMEOUT_MS},
            },
        },
    ),
    ToolSpec(
        "browser_fill",
        "Fill a selector in the current browser page and optionally press Enter.",
        {
            "type": "object",
            "required": ["selector", "text"],
            "properties": {
                "selector": {"type": "string", "maxLength": _MAX_SELECTOR_CHARS},
                "text": {"type": "string", "maxLength": 8_000},
                "submit": {"type": "boolean"},
                "timeout_ms": {"type": "integer", "minimum": 1, "maximum": _MAX_TIMEOUT_MS},
            },
        },
    ),
    ToolSpec(
        "browser_text",
        "Extract bounded visible text from the current browser page.",
        {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "maxLength": _MAX_SELECTOR_CHARS},
                "timeout_ms": {"type": "integer", "minimum": 1, "maximum": _MAX_TIMEOUT_MS},
            },
        },
    ),
    ToolSpec("browser_title", "Read the current browser page title.", {"type": "object", "properties": {}}),
    ToolSpec("browser_url", "Read the current browser page URL.", {"type": "object", "properties": {}}),
    ToolSpec(
        "browser_screenshot",
        "Save a bounded screenshot under the current AgentX session artifacts.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 120},
                "full_page": {"type": "boolean"},
            },
        },
    ),
    ToolSpec("browser_status", "Report whether the browser session is running.", {"type": "object", "properties": {}}),
    ToolSpec("browser_close", "Close the current browser session.", {"type": "object", "properties": {}}),
)


class BrowserError(RuntimeError):
    """Raised for browser setup, policy, or interaction failures."""


class BrowserController(Protocol):
    def open(self, url: str, *, headless: bool, channel: str, width: int, height: int) -> str: ...
    def navigate(self, url: str) -> str: ...
    def click(self, selector: str, *, timeout_ms: int) -> str: ...
    def fill(self, selector: str, text: str, *, submit: bool, timeout_ms: int) -> str: ...
    def text(self, selector: str, *, timeout_ms: int) -> str: ...
    def title(self) -> str: ...
    def url(self) -> str: ...
    def screenshot(self, name: str, *, full_page: bool) -> str: ...
    def status(self) -> str: ...
    def close(self) -> str: ...


class BrowserToolExecutor:
    """Expose the Ryzen-style browser controller through AgentX tools."""

    def __init__(
        self,
        *,
        artifacts_dir: str | Path,
        approval_callback: ApprovalCallback | None = None,
        controller: BrowserController | None = None,
        resolver: Any = None,
        allow_private_hosts: bool = False,
    ) -> None:
        if approval_callback is not None and not callable(approval_callback):
            raise ToolError("approval_callback must be callable or None")
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve()
        self.approval_callback = approval_callback
        self._controller = controller
        self._resolver = resolver or socket.getaddrinfo
        self.allow_private_hosts = allow_private_hosts

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return _BROWSER_TOOL_SPECS if self.approval_callback is not None else ()

    def call(self, name: str, arguments: Mapping[str, object] | None = None) -> ToolResult:
        args = dict(arguments or {})
        name = {"browser.open": "browser_open", "browser.navigate": "browser_navigate"}.get(name, name)
        try:
            output = {
                "browser_open": self.open,
                "browser_navigate": self.navigate,
                "browser_click": self.click,
                "browser_fill": self.fill,
                "browser_text": self.text,
                "browser_title": self.title,
                "browser_url": self.url,
                "browser_screenshot": self.screenshot,
                "browser_status": self.status,
                "browser_close": self.close,
            }[name](args)
        except (KeyError, BrowserError, ToolError, OSError, ValueError) as exc:
            return ToolResult(name=name, ok=False, error=str(exc))
        return ToolResult(name=name, ok=True, output=output)

    def open(self, args: Mapping[str, object]) -> str:
        url = self._url(args.get("url"))
        headless = bool(args.get("headless", True))
        channel = self._optional_string(args.get("channel", ""), "channel")
        width = self._bounded_int(args.get("width", 1_440), "width", 320, 4_000)
        height = self._bounded_int(args.get("height", 960), "height", 240, 4_000)
        self._approve("browser.open", {"url": url, "headless": headless})
        return self._browser().open(url, headless=headless, channel=channel, width=width, height=height)

    def navigate(self, args: Mapping[str, object]) -> str:
        url = self._url(args.get("url"))
        self._approve("browser.navigate", {"url": url})
        return self._browser().navigate(url)

    def click(self, args: Mapping[str, object]) -> str:
        selector = self._selector(args.get("selector"))
        timeout = self._bounded_int(args.get("timeout_ms", 10_000), "timeout_ms", 1, _MAX_TIMEOUT_MS)
        self._approve("browser.click", {"selector": selector})
        return self._browser().click(selector, timeout_ms=timeout)

    def fill(self, args: Mapping[str, object]) -> str:
        selector = self._selector(args.get("selector"))
        text = args.get("text")
        if not isinstance(text, str) or len(text) > 8_000:
            raise BrowserError("text must be a string of at most 8000 characters")
        submit = bool(args.get("submit", False))
        timeout = self._bounded_int(args.get("timeout_ms", 10_000), "timeout_ms", 1, _MAX_TIMEOUT_MS)
        self._approve("browser.fill", {"selector": selector, "text_length": len(text), "submit": submit})
        return self._browser().fill(selector, text, submit=submit, timeout_ms=timeout)

    def text(self, args: Mapping[str, object]) -> str:
        selector = self._selector(args.get("selector", "body"))
        timeout = self._bounded_int(args.get("timeout_ms", 10_000), "timeout_ms", 1, _MAX_TIMEOUT_MS)
        return self._browser().text(selector, timeout_ms=timeout)[:_MAX_BROWSER_TEXT]

    def title(self, args: Mapping[str, object]) -> str:
        del args
        return self._browser().title()[:500]

    def url(self, args: Mapping[str, object]) -> str:
        del args
        return self._browser().url()[:_MAX_URL_CHARS]

    def screenshot(self, args: Mapping[str, object]) -> str:
        name = self._optional_string(args.get("name", "browser"), "name")
        full_page = bool(args.get("full_page", True))
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._approve("browser.screenshot", {"name": name, "full_page": full_page})
        return self._browser().screenshot(name, full_page=full_page)

    def status(self, args: Mapping[str, object]) -> str:
        del args
        return self._browser().status()[:1_000]

    def close(self, args: Mapping[str, object]) -> str:
        del args
        return self._browser().close()

    def _browser(self) -> BrowserController:
        if self._controller is None:
            self._controller = _PlaywrightBrowser(self.artifacts_dir, resolver=self._resolver, allow_private_hosts=self.allow_private_hosts)
        return self._controller

    def _approve(self, operation: str, details: Mapping[str, object]) -> None:
        if self.approval_callback is None or self.approval_callback(operation, details) is not True:
            raise BrowserError("approval denied")

    def _url(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > _MAX_URL_CHARS:
            raise BrowserError("url must be an HTTP(S) URL of at most 2048 characters")
        url = value.strip()
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise BrowserError("url must use HTTP or HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise BrowserError("url must not include credentials")
        if not self.allow_private_hosts:
            _validate_public_host(parsed.hostname, resolver=self._resolver)
        return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, ""))

    @staticmethod
    def _selector(value: object) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > _MAX_SELECTOR_CHARS:
            raise BrowserError("selector must be a non-empty string of at most 500 characters")
        return value.strip()

    @staticmethod
    def _optional_string(value: object, field: str) -> str:
        if not isinstance(value, str):
            raise BrowserError(f"{field} must be a string")
        return value[:120]

    @staticmethod
    def _bounded_int(value: object, field: str, minimum: int, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise BrowserError(f"{field} must be an integer from {minimum} to {maximum}")
        return value


class _PlaywrightBrowser:
    def __init__(self, artifacts_dir: Path, *, resolver: Any, allow_private_hosts: bool) -> None:
        self.artifacts_dir = artifacts_dir
        self.resolver = resolver
        self.allow_private_hosts = allow_private_hosts
        self._playwright = None
        self._browser = None
        self._page = None

    def open(self, url: str, *, headless: bool, channel: str, width: int, height: int) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserError("Playwright is not installed. Install it with: python -m pip install playwright") from exc
        if self._page is None:
            self._playwright = sync_playwright().start()
            launch_args: dict[str, object] = {"headless": headless}
            if channel:
                launch_args["channel"] = channel
            self._browser = self._playwright.chromium.launch(**launch_args)
            context = self._browser.new_context(viewport={"width": width, "height": height})
            self._page = context.new_page()
        self._page.goto(url, wait_until="domcontentloaded")
        return f"Opened {url}"

    def _page_or_raise(self):
        if self._page is None:
            raise BrowserError("browser session is not open; call browser_open first")
        return self._page

    def navigate(self, url: str) -> str:
        self._page_or_raise().goto(url, wait_until="domcontentloaded")
        return f"Navigated to {url}"

    def click(self, selector: str, *, timeout_ms: int) -> str:
        self._page_or_raise().locator(selector).click(timeout=timeout_ms)
        return f"Clicked {selector}"

    def fill(self, selector: str, text: str, *, submit: bool, timeout_ms: int) -> str:
        locator = self._page_or_raise().locator(selector)
        locator.fill(text, timeout=timeout_ms)
        if submit:
            locator.press("Enter", timeout=timeout_ms)
        return f"Filled {selector}"

    def text(self, selector: str, *, timeout_ms: int) -> str:
        return self._page_or_raise().locator(selector).inner_text(timeout=timeout_ms)

    def title(self) -> str:
        return self._page_or_raise().title()

    def url(self) -> str:
        return self._page_or_raise().url

    def screenshot(self, name: str, *, full_page: bool) -> str:
        page = self._page_or_raise()
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-") or "browser"
        path = self.artifacts_dir / f"{safe_name[:100]}.png"
        page.screenshot(path=str(path), full_page=full_page)
        return str(path)

    def status(self) -> str:
        return "open: " + self.url() if self._page is not None else "closed"

    def close(self) -> str:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._browser = None
        self._page = None
        self._playwright = None
        return "Browser closed"


def _validate_public_host(hostname: str, *, resolver: Any) -> None:
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        raise BrowserError("private browser hosts are disabled")
    try:
        literal = ipaddress.ip_address(normalized)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise BrowserError("private browser hosts are disabled")
        return
    try:
        entries = resolver(normalized, 443, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror) as exc:
        raise BrowserError("could not resolve browser hostname") from exc
    addresses = {entry[4][0].split("%", 1)[0] for entry in entries if len(entry) >= 5 and entry[4]}
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise BrowserError("browser hostname resolves to a private address")


__all__ = ["BrowserError", "BrowserToolExecutor", "_BROWSER_TOOL_SPECS"]
