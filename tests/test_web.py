import unittest
from datetime import datetime, timezone

from agentx.tools import ToolResult, ToolSpec
from agentx.browser import BrowserToolExecutor
from agentx.web import WebAccessError, WebAccessService, WebCache, WebUIAdapter


class _FakeWebTools:
    specs = (
        ToolSpec("web_search", "search", {"type": "object"}),
        ToolSpec("web_fetch", "fetch", {"type": "object"}),
    )

    def __init__(self, result: ToolResult):
        self.result = result
        self.calls = []

    def call(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return self.result


class WebAccessServiceTests(unittest.TestCase):
    def test_service_exposes_injected_tools_without_network_access(self):
        tools = _FakeWebTools(ToolResult(name="web_search", ok=True, output={"results": []}))
        service = WebAccessService(research_tools=tools)

        result = service.search("AgentX", max_results=2)

        self.assertEqual([], result["results"])
        self.assertEqual([], result["citations"])
        self.assertEqual([("web_search", {"query": "AgentX", "max_results": 2})], tools.calls)
        self.assertEqual(("web_search", "web_fetch"), tuple(spec.name for spec in service.specs))

    def test_service_turns_failed_tool_result_into_shared_error(self):
        tools = _FakeWebTools(ToolResult(name="web_fetch", ok=False, error="approval denied"))
        service = WebAccessService(research_tools=tools)

        with self.assertRaisesRegex(WebAccessError, "approval denied"):
            service.fetch("https://example.com")

    def test_ui_adapter_preserves_specs_and_emits_status_only_events(self):
        tools = _FakeWebTools(ToolResult(name="web_fetch", ok=True, output={"content": "private page text"}))
        adapter = WebUIAdapter(WebAccessService(research_tools=tools))

        result = adapter.call("web_fetch", {"url": "https://example.com"})

        self.assertEqual("web_fetch", result.name)
        self.assertEqual({"name": "web_fetch", "ok": True}, adapter.event(result))
        self.assertNotIn("private page text", adapter.event(result))

    def test_ui_adapter_combines_web_and_browser_tools(self):
        tools = _FakeWebTools(ToolResult(name="web_search", ok=True, output={"results": []}))
        browser = BrowserToolExecutor(artifacts_dir=".", controller=object(), approval_callback=lambda operation, details: True)
        adapter = WebUIAdapter(WebAccessService(research_tools=tools), browser=browser)

        self.assertIn("web_search", {spec.name for spec in adapter.specs})
        self.assertIn("browser_open", {spec.name for spec in adapter.specs})

    def test_service_uses_approved_bounded_cache(self):
        tools = _FakeWebTools(ToolResult(name="web_fetch", ok=True, output={"url": "https://example.com", "content": "cached"}))
        cache = WebCache(clock=lambda: 100.0, ttl_seconds=30)
        approvals = []
        service = WebAccessService(
            research_tools=tools,
            approval_callback=lambda operation, details: approvals.append((operation, details)) or True,
            cache=cache,
            clock=lambda: datetime.fromtimestamp(100, timezone.utc),
        )

        first = service.fetch("https://example.com")
        second = service.fetch("https://example.com")

        self.assertEqual(first, second)
        self.assertEqual(1, len(tools.calls))
        self.assertEqual(1, len(approvals))

    def test_cache_expires_and_evicts_old_entries(self):
        current = [0.0]
        cache = WebCache(max_entries=1, ttl_seconds=10, clock=lambda: current[0])
        self.assertTrue(cache.put("a", {"value": "one"}))
        self.assertTrue(cache.put("b", {"value": "two"}))
        self.assertIsNone(cache.get("a"))
        self.assertEqual({"value": "two"}, cache.get("b"))
        current[0] = 11.0
        self.assertIsNone(cache.get("b"))


if __name__ == "__main__":
    unittest.main()
