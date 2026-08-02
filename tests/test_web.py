import unittest

from agentx.tools import ToolResult, ToolSpec
from agentx.web import WebAccessError, WebAccessService


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

        self.assertEqual({"results": []}, result)
        self.assertEqual([("web_search", {"query": "AgentX", "max_results": 2})], tools.calls)
        self.assertEqual(("web_search", "web_fetch"), tuple(spec.name for spec in service.specs))

    def test_service_turns_failed_tool_result_into_shared_error(self):
        tools = _FakeWebTools(ToolResult(name="web_fetch", ok=False, error="approval denied"))
        service = WebAccessService(research_tools=tools)

        with self.assertRaisesRegex(WebAccessError, "approval denied"):
            service.fetch("https://example.com")


if __name__ == "__main__":
    unittest.main()
