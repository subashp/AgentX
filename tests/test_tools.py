import shutil
import unittest
from pathlib import Path
from unittest import mock

from agentx.tools import ReadOnlyWorkspaceTools


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


if __name__ == "__main__":
    unittest.main()
