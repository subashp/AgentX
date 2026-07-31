import unittest

from agentx.subagents import (
    MAX_SUBAGENTS,
    SubagentError,
    SubagentManager,
    SubagentTools,
)
from agentx.tools import CompositeToolExecutor, ToolError, ToolResult, ToolSpec


class RecordingRunner:
    def __init__(self, *, failure=None):
        self.calls = []
        self.failure = failure

    def run(self, task, *, session_id, depth):
        self.calls.append((task, session_id, depth))
        if self.failure is not None:
            raise self.failure
        return {
            "summary": f"completed: {task.prompt}",
            "artifact_root": f"artifacts/{session_id}",
        }


class SubagentManagerTests(unittest.TestCase):
    def test_child_receives_isolated_normalized_context_and_depth_one(self):
        runner = RecordingRunner()
        manager = SubagentManager(parent_session_id="parent-1", runner=runner)

        record = manager.spawn(
            {
                "prompt": "Inspect the selected files",
                "context_paths": ["docs\\README.md", "./src/main.py", "src/main.py"],
            }
        )

        self.assertEqual("completed", record.status)
        self.assertEqual(("docs/README.md", "src/main.py"), record.task.context_paths)
        self.assertEqual(1, len(runner.calls))
        task, session_id, depth = runner.calls[0]
        self.assertEqual(record.task, task)
        self.assertEqual(("docs/README.md", "src/main.py"), task.context_paths)
        self.assertEqual("parent-1-subagent-01", session_id)
        self.assertEqual(1, depth)

    def test_max_subagent_limit_is_ten(self):
        runner = RecordingRunner()
        manager = SubagentManager(parent_session_id="parent-1", runner=runner)

        self.assertEqual(10, MAX_SUBAGENTS)
        for number in range(MAX_SUBAGENTS):
            manager.spawn({"prompt": f"task {number}"})

        self.assertFalse(manager.can_spawn)
        with self.assertRaisesRegex(SubagentError, "limit reached"):
            manager.spawn({"prompt": "one too many"})

        with self.assertRaises(SubagentError):
            SubagentManager(
                parent_session_id="parent-1",
                runner=runner,
                max_subagents=MAX_SUBAGENTS + 1,
            )

    def test_depth_one_cannot_spawn_and_exposes_no_subagent_tools(self):
        manager = SubagentManager(parent_session_id="child-1", runner=RecordingRunner(), depth=1)
        tools = SubagentTools(manager)

        self.assertFalse(manager.can_spawn)
        self.assertEqual((), tools.specs)
        with self.assertRaisesRegex(SubagentError, "cannot create further subagents"):
            manager.spawn({"prompt": "nested task"})

        unavailable = tools.call("subagent_list")
        self.assertFalse(unavailable.ok)
        self.assertIn("unavailable", unavailable.error)

    def test_subagent_create_list_and_get_tool_calls(self):
        manager = SubagentManager(parent_session_id="parent-1", runner=RecordingRunner())
        tools = SubagentTools(manager)

        created = tools.call(
            "subagent_create",
            {"prompt": "Summarize the module", "provider": "codex"},
        )
        self.assertTrue(created.ok)
        child_id = created.output["id"]
        self.assertEqual("subagent-01", child_id)

        listed = tools.call("subagent_list")
        self.assertTrue(listed.ok)
        self.assertEqual([child_id], [item["id"] for item in listed.output["subagents"]])

        fetched = tools.call("subagent_get", {"subagent_id": child_id})
        self.assertTrue(fetched.ok)
        self.assertEqual(child_id, fetched.output["id"])
        self.assertEqual("completed: Summarize the module", fetched.output["summary"])

    def test_failed_runner_returns_failed_record_without_raising(self):
        manager = SubagentManager(
            parent_session_id="parent-1",
            runner=RecordingRunner(failure=RuntimeError("model unavailable")),
        )

        record = manager.spawn({"prompt": "Use the remote model"})

        self.assertEqual("failed", record.status)
        self.assertEqual("RuntimeError: model unavailable", record.error)
        self.assertEqual({}, dict(record.result))
        self.assertEqual(record, manager.get(record.id))


class CompositeToolExecutorTests(unittest.TestCase):
    class FixtureTools:
        def __init__(self, name, value):
            self._specs = (ToolSpec(name, f"Fixture {name}", {"type": "object"}),)
            self.value = value
            self.calls = []

        @property
        def specs(self):
            return self._specs

        def call(self, name, arguments=None):
            self.calls.append((name, arguments))
            return ToolResult(name=name, ok=True, output=self.value)

    def test_dispatches_to_matching_executor(self):
        first = self.FixtureTools("first_tool", "one")
        second = self.FixtureTools("second_tool", "two")
        composite = CompositeToolExecutor(first, second)

        result = composite.call("second_tool", {"value": 2})

        self.assertEqual(("first_tool", "second_tool"), tuple(spec.name for spec in composite.specs))
        self.assertTrue(result.ok)
        self.assertEqual("two", result.output)
        self.assertEqual([], first.calls)
        self.assertEqual([("second_tool", {"value": 2})], second.calls)

    def test_rejects_duplicate_tool_names(self):
        first = self.FixtureTools("same_tool", "one")
        second = self.FixtureTools("same_tool", "two")

        with self.assertRaisesRegex(ToolError, "duplicate tool name: same_tool"):
            CompositeToolExecutor(first, second)


if __name__ == "__main__":
    unittest.main()
