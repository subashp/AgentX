import shutil
import unittest
from pathlib import Path

from agentx.config import AgentXPaths
from agentx.tool_registry import build_private_tool_executor, tool_names


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("tests") / ".tmp_tool_registry"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        self.paths = AgentXPaths(
            root=self.root / "state",
            settings=self.root / "state" / "settings.json",
            sessions=self.root / "state" / "sessions",
            memories=self.root / "state" / "memories",
            auth=self.root / "state" / "auth",
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_plan_mode_excludes_mutation_tools(self):
        executor = build_private_tool_executor(
            mode="plan",
            workspace_root=self.root,
            context_paths=(),
            paths=self.paths,
            user_prompt="inspect",
        )

        names = set(tool_names(executor))
        self.assertIn("workspace_tree", names)
        self.assertIn("memory_search", names)
        self.assertNotIn("workspace_patch", names)
        self.assertNotIn("shell_exec", names)

    def test_execute_mode_includes_patch_and_shell_tools(self):
        executor = build_private_tool_executor(
            mode="execute",
            workspace_root=self.root,
            context_paths=("README.md",),
            paths=self.paths,
            user_prompt="patch README",
            approval_callback=lambda operation, details: False,
        )

        names = set(tool_names(executor))
        self.assertIn("workspace_patch", names)
        self.assertIn("shell_exec", names)
        self.assertEqual(len(names), len(tool_names(executor)))

    def test_memory_mode_exposes_memory_tools_only(self):
        executor = build_private_tool_executor(
            mode="memory",
            workspace_root=self.root,
            context_paths=(),
            paths=self.paths,
            user_prompt="search memory",
        )

        names = set(tool_names(executor))
        self.assertIn("memory_search", names)
        self.assertNotIn("workspace_tree", names)
        self.assertNotIn("workspace_patch", names)


if __name__ == "__main__":
    unittest.main()
