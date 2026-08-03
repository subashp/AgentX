import unittest

from agentx.adapters import AdapterRequest, AdapterResult
from agentx.autonomy import AutonomousLimits, AutonomousWorkController
from agentx.routing import AgentRun
from agentx.tools import ToolResult, ToolSpec


class AutonomousWorkControllerTests(unittest.TestCase):
    def test_shell_like_limit_blocks_without_calling_inner_tool(self):
        inner = RecordingToolExecutor(
            ToolResult(
                name="test.run",
                ok=True,
                output={"profile": "python-unittest", "argv": ["python"], "exit_code": 0},
            )
        )
        controller = AutonomousWorkController(
            AutonomousLimits(max_iterations=6, max_shell_calls=1, max_patch_attempts=4)
        )
        tools = controller.wrap_tool_executor(inner)

        first = tools.call("test_run", {"profile": "python-unittest"})
        second = tools.call("shell_exec", {"argv": ["python", "--version"]})

        self.assertTrue(first.ok)
        self.assertFalse(second.ok)
        self.assertIn("max_shell_calls", second.error)
        self.assertEqual(["test_run"], inner.calls)
        self.assertEqual("limit_exceeded:max_shell_calls", controller.stop_reason)

    def test_patch_tracking_records_changed_paths_and_limit(self):
        inner = RecordingToolExecutor(
            ToolResult(name="workspace.patch", ok=True, output={"paths": ["src/agentx/cli.py"]})
        )
        controller = AutonomousWorkController(
            AutonomousLimits(max_iterations=6, max_shell_calls=8, max_patch_attempts=1)
        )
        tools = controller.wrap_tool_executor(inner)

        first = tools.call("workspace_patch", {"patch": "diff --git a/src/agentx/cli.py b/src/agentx/cli.py\n"})
        second = tools.call("workspace_patch", {"patch": "diff --git a/tests/test_cli.py b/tests/test_cli.py\n"})

        self.assertTrue(first.ok)
        self.assertFalse(second.ok)
        self.assertEqual(["src/agentx/cli.py"], controller.changed_paths)
        self.assertEqual("limit_exceeded:max_patch_attempts", controller.stop_reason)

    def test_tool_event_callback_tracks_iterations_and_delegates(self):
        events = []
        controller = AutonomousWorkController()
        callback = controller.tool_event_callback(events.append)

        callback({"event": "requested", "name": "workspace_read", "round": 3})

        self.assertEqual(3, controller.iteration_count)
        self.assertEqual([{"event": "requested", "name": "workspace_read", "round": 3}], events)

    def test_adapter_wrapper_adds_autonomous_summary(self):
        controller = AutonomousWorkController()
        controller.tool_event_callback()({"event": "requested", "name": "test_run", "round": 2})
        tools = controller.wrap_tool_executor(
            RecordingToolExecutor(
                ToolResult(
                    name="test.run",
                    ok=True,
                    output={
                        "profile": "python-unittest",
                        "argv": ["python", "-m", "unittest"],
                        "exit_code": 0,
                    },
                )
            )
        )
        tools.call("test_run", {"profile": "python-unittest"})
        adapter = controller.wrap_adapter(RecordingAdapter())

        result = adapter.execute(AdapterRequest(run=AgentRun(prompt="fix tests", mode="execute")))

        autonomous = result.outcome["autonomous"]
        self.assertEqual(2, autonomous["iterations"])
        self.assertEqual(1, autonomous["tool_calls"])
        self.assertEqual(1, autonomous["shell_calls"])
        self.assertEqual("validation_passed", autonomous["stop_reason"])
        self.assertEqual(0, autonomous["last_test"]["exit_code"])


class RecordingToolExecutor:
    def __init__(self, result: ToolResult):
        self.result = result
        self.calls = []

    @property
    def specs(self):
        return (
            ToolSpec("test_run", "Run tests.", {"type": "object", "properties": {}}),
            ToolSpec("shell_exec", "Run shell.", {"type": "object", "properties": {}}),
            ToolSpec("workspace_patch", "Patch.", {"type": "object", "properties": {}}),
        )

    def call(self, name, arguments=None):
        self.calls.append(name)
        return self.result


class RecordingAdapter:
    provider_id = "private-openai-compatible"

    def execute(self, request):
        return AdapterResult(
            provider_id=self.provider_id,
            model_id="fixture-model",
            model_tier="high",
            status="success",
            transcript_events=(),
            cost={"currency": "USD", "estimated": False, "total_cost_usd": 0.0},
            outcome={"status": "success", "summary": "done"},
            patch="",
        )


if __name__ == "__main__":
    unittest.main()
