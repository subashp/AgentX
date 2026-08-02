import shutil
import unittest
from pathlib import Path

from agentx.browser import BrowserToolExecutor


def _public_resolver(host, port, type=None):
    del host, port, type
    return [(0, 0, 0, "", ("93.184.216.34", 443))]


class _FakeBrowser:
    def __init__(self):
        self.calls = []

    def open(self, url, *, headless, channel, width, height):
        self.calls.append(("open", url, headless, channel, width, height))
        return "opened"

    def navigate(self, url):
        self.calls.append(("navigate", url))
        return "navigated"

    def click(self, selector, *, timeout_ms):
        self.calls.append(("click", selector, timeout_ms))
        return "clicked"

    def fill(self, selector, text, *, submit, timeout_ms):
        self.calls.append(("fill", selector, text, submit, timeout_ms))
        return "filled"

    def text(self, selector, *, timeout_ms):
        self.calls.append(("text", selector, timeout_ms))
        return "page text"

    def title(self):
        return "title"

    def url(self):
        return "https://example.com/"

    def screenshot(self, name, *, full_page):
        self.calls.append(("screenshot", name, full_page))
        return "artifact.png"

    def status(self):
        return "open"

    def close(self):
        return "closed"


class BrowserToolExecutorTests(unittest.TestCase):
    def setUp(self):
        self.artifacts = Path("tests") / ".tmp_browser_tools"
        shutil.rmtree(self.artifacts, ignore_errors=True)
        self.artifacts.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.artifacts, ignore_errors=True)

    def test_tools_are_hidden_without_approval(self):
        executor = BrowserToolExecutor(artifacts_dir=self.artifacts, controller=_FakeBrowser())
        self.assertEqual((), executor.specs)
        result = executor.call("browser_open", {"url": "https://example.com"})
        self.assertFalse(result.ok)
        self.assertEqual("approval denied", result.error)

    def test_fake_controller_receives_approved_bounded_operations(self):
        browser = _FakeBrowser()
        approvals = []
        executor = BrowserToolExecutor(
            artifacts_dir=self.artifacts,
            controller=browser,
            approval_callback=lambda operation, details: approvals.append((operation, details)) or True,
            resolver=_public_resolver,
        )
        opened = executor.call("browser_open", {"url": "https://example.com", "headless": True})
        filled = executor.call("browser_fill", {"selector": "#q", "text": "AgentX", "submit": True})
        text = executor.call("browser_text", {})

        self.assertTrue(opened.ok)
        self.assertTrue(filled.ok)
        self.assertEqual("page text", text.output)
        self.assertEqual("open", approvals[0][0].split(".")[-1])
        self.assertEqual(("open", "https://example.com/", True, "", 1440, 960), browser.calls[0])

    def test_private_host_is_rejected_before_controller_use(self):
        browser = _FakeBrowser()
        executor = BrowserToolExecutor(
            artifacts_dir=self.artifacts,
            controller=browser,
            approval_callback=lambda operation, details: True,
            resolver=lambda *args, **kwargs: [(0, 0, 0, "", ("192.168.1.2", 80))],
        )
        result = executor.call("browser_open", {"url": "https://internal.example"})

        self.assertFalse(result.ok)
        self.assertIn("private", result.error)
        self.assertEqual([], browser.calls)


if __name__ == "__main__":
    unittest.main()
