import shutil
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

from agentx.tools import (
    ControlledWorkspaceTools,
    GitCommitTools,
    ReadOnlyWorkspaceTools,
    TestRunTools,
    ToolResult,
    WebResearchTools,
)


class ToolResultTests(unittest.TestCase):
    def test_failed_result_preserves_bounded_output_for_model_repair_loop(self):
        result = ToolResult(
            name="test.run",
            ok=False,
            error="test command returned a non-zero exit code",
            output={"exit_code": 1, "stdout": "AssertionError"},
        ).as_dict()

        self.assertEqual("test command returned a non-zero exit code", result["error"])
        self.assertEqual({"exit_code": 1, "stdout": "AssertionError"}, result["output"])


class ReadOnlyWorkspaceToolsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("tests") / ".tmp_workspace_tools"
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "src").mkdir(parents=True)
        (self.root / "src" / "main.py").write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
        (self.root / "README.md").write_text("AgentX workspace\n", encoding="utf-8")
        (self.root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_tree_and_read_are_bounded_and_hide_private_files(self):
        tools = ReadOnlyWorkspaceTools(self.root)

        tree = tools.call("workspace.tree", {"max_entries": 20})
        self.assertTrue(tree.ok)
        tree_paths = {entry["path"] for entry in tree.output["entries"]}
        self.assertIn("src", tree_paths)
        self.assertNotIn(".env", tree_paths)

        read = tools.call("workspace.read", {"path": "src/main.py", "start_line": 1, "end_line": 1})
        self.assertTrue(read.ok)
        self.assertIn("def hello", read.output["content"])

    def test_read_rejects_traversal_and_private_paths(self):
        tools = ReadOnlyWorkspaceTools(self.root)

        traversal = tools.call("workspace.read", {"path": "../secret.txt"})
        private = tools.call("workspace.read", {"path": ".env"})

        self.assertFalse(traversal.ok)
        self.assertFalse(private.ok)

    def test_search_returns_line_numbered_matches(self):
        tools = ReadOnlyWorkspaceTools(self.root)

        result = tools.call("workspace.search", {"query": "HELLO", "case_sensitive": False})

        self.assertTrue(result.ok)
        self.assertEqual("src/main.py", result.output["matches"][0]["path"])
        self.assertEqual(1, result.output["matches"][0]["line"])

    def test_allowed_paths_limit_all_read_tools(self):
        tools = ReadOnlyWorkspaceTools(self.root, allowed_paths=("src",))

        allowed = tools.call("workspace.read", {"path": "src/main.py"})
        denied = tools.call("workspace.read", {"path": "README.md"})

        self.assertTrue(allowed.ok)
        self.assertFalse(denied.ok)

    def test_git_commands_use_argument_arrays_and_bounded_output(self):
        tools = ReadOnlyWorkspaceTools(self.root)
        completed = mock.Mock(returncode=0, stdout=" M src/main.py\n", stderr="")

        with mock.patch("agentx.tools.subprocess.run", return_value=completed) as run:
            status = tools.call("git.status", {})
            diff = tools.call("git.diff", {"paths": ["src/main.py"]})

        self.assertTrue(status.ok)
        self.assertTrue(diff.ok)
        self.assertEqual(["git", "status", "--short", "--untracked-files=all"], run.call_args_list[0].args[0])
        self.assertIn("--no-ext-diff", run.call_args_list[1].args[0])


class _WebResponse:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict[str, str] | None = None):
        self.body = body
        self.status = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.closed = False

    def read(self, amount: int = -1) -> bytes:
        return self.body if amount < 0 else self.body[:amount]

    def close(self) -> None:
        self.closed = True


def _public_resolver(host, port, *, type):
    del host, port, type
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


class WebResearchToolsTests(unittest.TestCase):
    def test_tools_are_hidden_without_an_approval_callback(self):
        tools = WebResearchTools()

        self.assertEqual((), tools.specs)
        result = tools.call("web_search", {"query": "AgentX"})
        self.assertFalse(result.ok)
        self.assertEqual("approval denied", result.error)

    def test_search_requires_approval_before_sending_the_query(self):
        opener = mock.Mock()
        approvals = []
        tools = WebResearchTools(
            approval_callback=lambda operation, details: approvals.append((operation, details)) or False,
            opener=opener,
            resolver=_public_resolver,
        )

        result = tools.call("web_search", {"query": "current Halo ROCm support"})

        self.assertFalse(result.ok)
        self.assertEqual("approval denied", result.error)
        self.assertEqual("web.search", approvals[0][0])
        self.assertEqual("current Halo ROCm support", approvals[0][1]["query"])
        opener.assert_not_called()

    def test_search_parses_compact_results_and_unwraps_duckduckgo_links(self):
        body = b"""
            <a class='result__a' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fguide'>
              Example &amp; guide
            </a>
            <div class='result__snippet'>A current <b>reference</b> page.</div>
        """
        response = _WebResponse(body)
        opener = mock.Mock(return_value=response)
        tools = WebResearchTools(
            approval_callback=lambda operation, details: True,
            opener=opener,
            resolver=_public_resolver,
        )

        result = tools.call("web_search", {"query": "example guide", "max_results": 1})

        self.assertTrue(result.ok)
        self.assertEqual("example guide", result.output["query"])
        self.assertEqual("https://example.com/guide", result.output["results"][0]["url"])
        self.assertEqual("Example & guide", result.output["results"][0]["title"])
        self.assertEqual("A current reference page.", result.output["results"][0]["snippet"])
        self.assertIn("q=example+guide", opener.call_args.args[0].full_url)
        self.assertTrue(response.closed)

    def test_search_falls_back_to_brave_after_a_duckduckgo_bot_challenge(self):
        challenge = _WebResponse(b"<div class='anomaly-modal'>Unfortunately, bots use DuckDuckGo too.</div>")
        body = _WebResponse(b"""
            <div class='snippet' data-type='web'>
              <a href='https://example.com/current'><div class='title search-snippet-title'>
                Current example
              </div></a>
              <div class='generic-snippet'>A concise current reference.</div>
            </div>
        """)
        approvals = []
        tools = WebResearchTools(
            approval_callback=lambda operation, details: approvals.append(details) or True,
            opener=mock.Mock(side_effect=[challenge, body]),
            resolver=_public_resolver,
        )

        result = tools.call("web_search", {"query": "current example"})

        self.assertTrue(result.ok)
        self.assertEqual("Brave Search", result.output["source"])
        self.assertEqual(2, len(approvals))
        self.assertEqual("DuckDuckGo Search", approvals[0]["source"])
        self.assertEqual("Brave Search", approvals[1]["source"])
        self.assertIn("bot challenge", approvals[1]["reason"])
        self.assertEqual(
            {
                "title": "Current example",
                "url": "https://example.com/current",
                "snippet": "A concise current reference.",
            },
            result.output["results"][0],
        )

    def test_fetch_strips_markup_and_honors_the_model_context_cap(self):
        body = (
            b"<html><head><title>Example title</title><script>secret()</script></head>"
            b"<body><main>" + (b"Useful text " * 200) + b"</main></body></html>"
        )
        response = _WebResponse(body)
        tools = WebResearchTools(
            approval_callback=lambda operation, details: True,
            opener=mock.Mock(return_value=response),
            resolver=_public_resolver,
        )

        result = tools.call("web_fetch", {"url": "https://example.com/article", "max_chars": 500})

        self.assertTrue(result.ok)
        self.assertEqual("Example title", result.output["title"])
        self.assertNotIn("secret", result.output["content"])
        self.assertLessEqual(len(result.output["content"]), 500)
        self.assertTrue(result.output["truncated"])
        self.assertTrue(response.closed)

    def test_fetch_rejects_non_public_redirect_before_a_second_request(self):
        redirect = _WebResponse(
            b"",
            status=302,
            headers={"Location": "https://127.0.0.1/internal"},
        )
        opener = mock.Mock(return_value=redirect)
        tools = WebResearchTools(
            approval_callback=lambda operation, details: True,
            opener=opener,
            resolver=_public_resolver,
        )

        result = tools.call("web_fetch", {"url": "https://example.com/redirect"})

        self.assertFalse(result.ok)
        self.assertIn("non-public IP", result.error)
        self.assertEqual(1, opener.call_count)
        self.assertTrue(redirect.closed)

    def test_fetch_document_uses_bounded_extractor_for_pdf_response(self):
        class FakeExtractor:
            def extract(self, body, *, media_type, filename):
                self.args = (body, media_type, filename)
                from agentx.documents import ExtractedDocument

                return ExtractedDocument(media_type=media_type, text="document text", page_count=2)

        extractor = FakeExtractor()
        response = _WebResponse(b"pdf bytes", headers={"Content-Type": "application/pdf"})
        tools = WebResearchTools(
            approval_callback=lambda operation, details: True,
            opener=mock.Mock(return_value=response),
            resolver=_public_resolver,
            document_extractor=extractor,
        )

        result = tools.call("web_fetch_document", {"url": "https://example.com/report.pdf", "max_pages": 2})

        self.assertTrue(result.ok)
        self.assertEqual("document text", result.output["content"])
        self.assertEqual("application/pdf", extractor.args[1])
        self.assertEqual("/report.pdf", extractor.args[2])
        self.assertTrue(response.closed)


class ControlledWorkspaceToolsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("tests") / ".tmp_controlled_workspace_tools"
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "src").mkdir(parents=True)
        (self.root / "src" / "main.py").write_text("print('before')\n", encoding="utf-8")
        (self.root / "README.md").write_text("AgentX workspace\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _patch(self, path="src/main.py"):
        return f"""diff --git a/{path} b/{path}
--- a/{path}
+++ b/{path}
@@ -1 +1 @@
-print('before')
+print('after')
"""

    def _approval(self, decision):
        calls = []

        def approve(operation, details):
            calls.append((operation, details))
            return decision

        return approve, calls

    def test_patch_approval_denial_does_not_invoke_git_apply(self):
        approve, calls = self._approval(False)
        tools = ControlledWorkspaceTools(
            self.root,
            allowed_paths=("src/main.py",),
            approval_callback=approve,
        )

        with mock.patch("agentx.tools.subprocess.run") as run:
            result = tools.call("workspace.patch", {"patch": self._patch()})

        self.assertFalse(result.ok)
        self.assertEqual(1, len(calls))
        self.assertEqual("workspace.patch", calls[0][0])
        run.assert_not_called()
        self.assertEqual("print('before')\n", (self.root / "src" / "main.py").read_text(encoding="utf-8"))

    def test_controlled_tools_can_expose_patch_without_shell(self):
        approve, _calls = self._approval(True)
        tools = ControlledWorkspaceTools(
            self.root,
            allowed_paths=("src/main.py",),
            approval_callback=approve,
            enable_patch=True,
            enable_shell=False,
        )

        names = {spec.name for spec in tools.specs}
        self.assertIn("workspace_patch", names)
        self.assertNotIn("shell_exec", names)
        result = tools.call("shell_exec", {"argv": ["python", "--version"]})
        self.assertFalse(result.ok)
        self.assertIn("disabled", result.error)

    def test_patch_validation_rejects_out_of_scope_path_before_approval(self):
        approve, calls = self._approval(True)
        tools = ControlledWorkspaceTools(
            self.root,
            allowed_paths=("src/main.py",),
            approval_callback=approve,
        )

        with mock.patch("agentx.tools.subprocess.run") as run:
            result = tools.call("workspace.patch", {"patch": self._patch("README.md")})

        self.assertFalse(result.ok)
        self.assertEqual([], calls)
        run.assert_not_called()
        self.assertEqual("patch_path_out_of_scope", result.output["validation"]["events"][0]["code"])

    def test_malformed_patch_is_rejected_before_approval(self):
        approve, calls = self._approval(True)
        tools = ControlledWorkspaceTools(
            self.root,
            allowed_paths=("src/main.py",),
            approval_callback=approve,
        )

        with mock.patch("agentx.tools.subprocess.run") as run:
            result = tools.call("workspace.patch", {"patch": "not a unified diff"})

        self.assertFalse(result.ok)
        self.assertEqual([], calls)
        run.assert_not_called()
        self.assertEqual("patch_no_target_paths", result.output["validation"]["events"][0]["code"])

    def test_patch_approval_success_invokes_git_apply_without_shell(self):
        approve, calls = self._approval(True)
        tools = ControlledWorkspaceTools(
            self.root,
            allowed_paths=("src/main.py",),
            approval_callback=approve,
        )
        completed = mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("agentx.tools.subprocess.run", return_value=completed) as run:
            result = tools.call("workspace.patch", {"patch": self._patch()})

        self.assertTrue(result.ok)
        self.assertEqual("workspace.patch", calls[0][0])
        command = run.call_args.args[0]
        self.assertEqual(["git", "apply"], command[:2])
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(self.root.resolve(), run.call_args.kwargs["cwd"])
        self.assertEqual(self._patch(), run.call_args.kwargs["input"])

    def test_shell_approval_denial_does_not_invoke_subprocess(self):
        approve, calls = self._approval(False)
        tools = ControlledWorkspaceTools(self.root, allowed_paths=("src/main.py",), approval_callback=approve)

        with mock.patch("agentx.tools.subprocess.run") as run:
            result = tools.call("shell.exec", {"argv": ["python", "--version"]})

        self.assertFalse(result.ok)
        self.assertEqual(1, len(calls))
        self.assertEqual("shell.exec", calls[0][0])
        run.assert_not_called()

    def test_controlled_tools_can_expose_shell_without_patch(self):
        approve, _calls = self._approval(True)
        tools = ControlledWorkspaceTools(
            self.root,
            allowed_paths=("src/main.py",),
            approval_callback=approve,
            enable_patch=False,
            enable_shell=True,
        )

        names = {spec.name for spec in tools.specs}
        self.assertNotIn("workspace_patch", names)
        self.assertIn("shell_exec", names)
        result = tools.call("workspace_patch", {"patch": self._patch()})
        self.assertFalse(result.ok)
        self.assertIn("disabled", result.error)

    def test_shell_exec_uses_argv_without_shell(self):
        approve, calls = self._approval(True)
        tools = ControlledWorkspaceTools(self.root, allowed_paths=("src/main.py",), approval_callback=approve)
        completed = mock.Mock(returncode=0, stdout="ok\n", stderr="")
        argv = ["python", "-c", "print('ok')"]

        with mock.patch("agentx.tools.subprocess.run", return_value=completed) as run:
            result = tools.call("shell.exec", {"argv": argv})

        self.assertTrue(result.ok)
        self.assertEqual("shell.exec", calls[0][0])
        self.assertEqual(argv, run.call_args.args[0])
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(self.root.resolve(), run.call_args.kwargs["cwd"])


class TestRunToolsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("tests") / ".tmp_test_run_tools"
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "tests").mkdir(parents=True)
        (self.root / "tests" / "test_sample.py").write_text("import unittest\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _approval(self, decision):
        calls = []

        def approve(operation, details):
            calls.append((operation, details))
            return decision

        return approve, calls

    def test_tools_are_hidden_without_approval_callback(self):
        tools = TestRunTools(self.root)

        self.assertEqual((), tools.specs)

    def test_approval_denial_does_not_invoke_runner(self):
        approve, calls = self._approval(False)
        runner = mock.Mock()
        tools = TestRunTools(self.root, approval_callback=approve, runner=runner)

        result = tools.call("test_run", {"profile": "python-unittest"})

        self.assertFalse(result.ok)
        self.assertEqual("approval denied", result.error)
        self.assertEqual("test.run", calls[0][0])
        runner.assert_not_called()

    def test_python_unittest_profile_uses_current_python_without_shell(self):
        approve, calls = self._approval(True)
        completed = mock.Mock(returncode=0, stdout="ok\n", stderr="")
        runner = mock.Mock(return_value=completed)
        tools = TestRunTools(self.root, approval_callback=approve, runner=runner)

        result = tools.call(
            "test_run",
            {"profile": "python-unittest", "target": "tests.test_sample", "timeout_seconds": 30},
        )

        self.assertTrue(result.ok)
        argv = runner.call_args.args[0]
        self.assertEqual([sys.executable, "-B", "-m", "unittest", "tests.test_sample"], argv)
        self.assertFalse(runner.call_args.kwargs["shell"])
        self.assertEqual(self.root.resolve(), runner.call_args.kwargs["cwd"])
        self.assertEqual(argv, calls[0][1]["argv"])
        self.assertEqual("python-unittest", result.output["profile"])

    def test_auto_profile_uses_unittest_discover_for_tests_directory(self):
        approve, _calls = self._approval(True)
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        runner = mock.Mock(return_value=completed)
        tools = TestRunTools(self.root, approval_callback=approve, runner=runner)

        result = tools.call("test.run", {"profile": "auto"})

        self.assertTrue(result.ok)
        self.assertEqual(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
            runner.call_args.args[0],
        )
        self.assertEqual("python-unittest", result.output["profile"])

    def test_npm_profile_uses_resolved_executable_without_shell(self):
        approve, calls = self._approval(True)
        completed = mock.Mock(returncode=1, stdout="", stderr="failed\n")
        runner = mock.Mock(return_value=completed)
        tools = TestRunTools(self.root, approval_callback=approve, runner=runner)

        with mock.patch("agentx.tools.shutil.which", return_value="C:\\tools\\npm.cmd"):
            result = tools.call("test_run", {"profile": "npm-test", "target": "unit"})

        self.assertFalse(result.ok)
        self.assertEqual(["C:\\tools\\npm.cmd", "test", "--", "unit"], runner.call_args.args[0])
        self.assertFalse(runner.call_args.kwargs["shell"])
        self.assertEqual("npm-test", calls[0][1]["profile"])
        self.assertEqual("test command returned a non-zero exit code", result.error)


class GitCommitToolsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("tests") / ".tmp_git_commit_tools"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _approval(self, decision):
        calls = []

        def approve(operation, details):
            calls.append((operation, details))
            return decision

        return approve, calls

    def test_tools_are_hidden_without_approval_callback(self):
        tools = GitCommitTools(self.root)

        self.assertEqual((), tools.specs)

    def test_git_add_denial_does_not_invoke_runner(self):
        approve, calls = self._approval(False)
        runner = mock.Mock()
        tools = GitCommitTools(self.root, approval_callback=approve, runner=runner)

        result = tools.call("git_add", {"paths": ["src/agentx/cli.py"]})

        self.assertFalse(result.ok)
        self.assertEqual("approval denied", result.error)
        self.assertEqual("git.add", calls[0][0])
        self.assertEqual(["src/agentx/cli.py"], calls[0][1]["paths"])
        runner.assert_not_called()

    def test_git_add_uses_argv_without_shell(self):
        approve, calls = self._approval(True)
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        runner = mock.Mock(return_value=completed)
        tools = GitCommitTools(self.root, approval_callback=approve, runner=runner)

        result = tools.call("git.add", {"paths": ["src\\agentx\\cli.py", "tests/test_cli.py"]})

        self.assertTrue(result.ok)
        self.assertEqual(["git", "add", "--", "src/agentx/cli.py", "tests/test_cli.py"], runner.call_args.args[0])
        self.assertFalse(runner.call_args.kwargs["shell"])
        self.assertEqual(self.root.resolve(), runner.call_args.kwargs["cwd"])
        self.assertEqual(["git", "add", "--", "src/agentx/cli.py", "tests/test_cli.py"], calls[0][1]["argv"])

    def test_git_commit_requires_bounded_message(self):
        approve, _calls = self._approval(True)
        tools = GitCommitTools(self.root, approval_callback=approve, runner=mock.Mock())

        result = tools.call("git_commit", {"message": "   "})

        self.assertFalse(result.ok)
        self.assertIn("non-empty", result.error)

    def test_git_commit_uses_argv_and_never_pushes(self):
        approve, calls = self._approval(True)
        completed = mock.Mock(returncode=0, stdout="[main abc] msg\n", stderr="")
        runner = mock.Mock(return_value=completed)
        tools = GitCommitTools(self.root, approval_callback=approve, runner=runner)

        result = tools.call("git_commit", {"message": "  Fix tests  "})

        self.assertTrue(result.ok)
        self.assertEqual(["git", "commit", "-m", "Fix tests"], runner.call_args.args[0])
        self.assertNotIn("push", runner.call_args.args[0])
        self.assertFalse(runner.call_args.kwargs["shell"])
        self.assertEqual("git.commit", calls[0][0])
        self.assertEqual("Fix tests", calls[0][1]["message"])


if __name__ == "__main__":
    unittest.main()
