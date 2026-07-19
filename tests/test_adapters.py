import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from agentx.adapters import (
    AdapterRequest,
    AdapterResult,
    CodexCliAdapter,
    ProcessResult,
    execute_adapter_run,
    execute_fake_run,
)
from agentx.config import AgentXPaths
from agentx.routing import AgentRun
from agentx.store import RUN_ARTIFACT_FILES, SessionStore


class AdapterFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self.fixture_root = Path("tests") / ".tmp_adapters"
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)
        self.fixture_root.mkdir(parents=True)

    def tearDown(self):
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)

    def make_paths(self):
        root = self.fixture_root / "state"
        return AgentXPaths(
            root=root,
            settings=root / "settings.json",
            sessions=root / "sessions",
            memories=root / "memories",
            auth=root / "auth",
        )


class FakeLocalAdapterTests(AdapterFixtureTestCase):
    def test_fake_execution_writes_standard_deterministic_artifact_bundle(self):
        paths = self.make_paths()
        run = AgentRun(
            prompt="Implement a focused test",
            mode="tests",
            provider="fake-local",
            model_tier="economy",
            context_paths=("src/agentx/adapters.py",),
        )

        stored = execute_fake_run(
            session_store=SessionStore(paths),
            session_id="ax-007",
            run=run,
        )
        first_snapshot = _read_artifacts(stored.root)
        second = execute_fake_run(
            session_store=SessionStore(paths),
            session_id="ax-007",
            run=run,
        )
        second_snapshot = _read_artifacts(second.root)

        self.assertEqual(paths.sessions / "ax-007", stored.root)
        self.assertEqual(set(RUN_ARTIFACT_FILES), set(stored.artifact_paths))
        self.assertEqual(first_snapshot, second_snapshot)
        self.assertEqual("Implement a focused test\n", first_snapshot["prompt.md"])
        self.assertEqual("", first_snapshot["patch.diff"])

        manifest = json.loads(first_snapshot["manifest.json"])
        self.assertEqual(1, manifest["schema_version"])
        self.assertEqual("ax-007", manifest["session_id"])
        self.assertEqual("Implement a focused test", manifest["run"]["prompt"])
        self.assertEqual(list(RUN_ARTIFACT_FILES), manifest["artifacts"])

        provider = json.loads(first_snapshot["provider.json"])
        self.assertEqual(
            {
                "model_id": "fake-local-deterministic",
                "model_tier": "economy",
                "provider_id": "fake-local",
                "status": "success",
            },
            provider,
        )
        self.assertEqual(
            {
                "excluded_paths": [],
                "included_paths": ["src/agentx/adapters.py"],
                "requested_paths": ["src/agentx/adapters.py"],
                "schema_version": 1,
                "source": "agent_run.context_paths",
            },
            json.loads(first_snapshot["context-map.json"]),
        )
        self.assertEqual(
            {
                "excluded_memories": [],
                "included_memories": [],
                "memory_ids": [],
                "schema_version": 1,
            },
            json.loads(first_snapshot["memory-map.json"]),
        )
        self.assertEqual([], json.loads(first_snapshot["redactions.json"]))
        self.assertEqual(
            {
                "currency": "USD",
                "estimated": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": 0.0,
            },
            json.loads(first_snapshot["cost.json"]),
        )
        self.assertEqual("success", json.loads(first_snapshot["outcome.json"])["status"])
        self.assertEqual(
            [
                "execution_started",
                "prompt_received",
                "execution_completed",
            ],
            [
                json.loads(line)["event"]
                for line in first_snapshot["transcript.jsonl"].splitlines()
            ],
        )

    def test_fake_adapter_does_not_require_path_or_provider_checks(self):
        paths = self.make_paths()
        run = AgentRun(prompt="No external provider", provider="fake-local")

        with mock.patch("shutil.which", side_effect=AssertionError("PATH was used")):
            stored = execute_fake_run(
                session_store=SessionStore(paths),
                session_id="no-path",
                run=run,
            )

        self.assertTrue((stored.root / "manifest.json").exists())
        self.assertEqual("success", stored.result.status)
        self.assertIsInstance(stored.result, AdapterResult)


class CodexCliAdapterTests(AdapterFixtureTestCase):
    def test_codex_cli_adapter_constructs_plan_command_and_forwards_process_options(self):
        runner = RecordingProcessRunner(ProcessResult(exit_code=0, stdout="plan", stderr="note"))
        adapter = CodexCliAdapter(
            command="codex-under-test",
            extra_args=("--model", "gpt-test"),
            cwd=Path("repo-root"),
            env={"AGENTX_TEST": "1"},
            timeout=12.5,
            process_runner=runner,
            model_id="codex-model",
        )
        run = AgentRun(
            prompt="Implement AX-008",
            mode="plan",
            provider="codex",
            model_tier="high",
            context_paths=("src/agentx/adapters.py",),
            task_hints=("plan only",),
        )

        result = adapter.execute(AdapterRequest(run=run))

        self.assertEqual("success", result.status)
        self.assertEqual("codex", result.provider_id)
        self.assertEqual("codex-model", result.model_id)
        self.assertEqual(1, len(runner.calls))
        call = runner.calls[0]
        self.assertEqual(Path("repo-root"), call["cwd"])
        self.assertEqual({"AGENTX_TEST": "1"}, call["env"])
        self.assertEqual(12.5, call["timeout"])
        argv = call["argv"]
        self.assertEqual("codex-under-test", argv[0])
        self.assertEqual(
            (
                "codex-under-test",
                "exec",
                "--sandbox",
                "read-only",
                "--ask-for-approval",
                "never",
                "--model",
                "gpt-test",
            ),
            argv[:-1],
        )
        self.assertIn("Do not edit files", argv[-1])
        self.assertIn("Context paths:\n- src/agentx/adapters.py", argv[-1])
        self.assertIn("User prompt:\nImplement AX-008", argv[-1])
        output_event = result.transcript_events[1]
        self.assertEqual("process_output_captured", output_event["event"])
        self.assertEqual("plan", output_event["stdout"])
        self.assertEqual("note", output_event["stderr"])

    def test_codex_cli_adapter_maps_nonzero_exit_to_failure(self):
        runner = RecordingProcessRunner(ProcessResult(exit_code=7, stdout="", stderr="denied"))
        adapter = CodexCliAdapter(command="codex-test", process_runner=runner)
        run = AgentRun(prompt="Review this", mode="review", provider="codex")

        result = adapter.execute(AdapterRequest(run=run))

        self.assertEqual("failure", result.status)
        self.assertEqual("failure", result.outcome["status"])
        self.assertEqual("codex_cli_failed", result.outcome["outcome"])
        self.assertEqual(7, result.outcome["exit_code"])
        self.assertEqual("", result.patch)

    def test_codex_cli_adapter_forwards_scoped_workspace_argument(self):
        runner = RecordingProcessRunner(ProcessResult(exit_code=0, stdout="plan", stderr=""))
        adapter = CodexCliAdapter(
            command="codex-test",
            cwd=Path("state/sessions/run/workspace"),
            extra_args=("-C", "state/sessions/run/workspace"),
            process_runner=runner,
        )

        adapter.execute(AdapterRequest(run=AgentRun(prompt="Plan this", provider="codex")))

        argv = runner.calls[0]["argv"]
        self.assertIn("-C", argv)
        self.assertEqual(
            "state/sessions/run/workspace",
            argv[argv.index("-C") + 1],
        )

    def test_execute_adapter_run_writes_codex_cli_artifacts_without_live_process(self):
        paths = self.make_paths()
        runner = RecordingProcessRunner(ProcessResult(exit_code=0, stdout="1. inspect\n", stderr=""))
        adapter = CodexCliAdapter(command="codex-test", process_runner=runner, timeout=9.0)
        run = AgentRun(prompt="Plan the adapter", mode="plan", provider="codex")

        stored = execute_adapter_run(
            session_store=SessionStore(paths),
            session_id="codex-adapter",
            run=run,
            adapter=adapter,
        )

        snapshot = _read_artifacts(stored.root)
        self.assertEqual("", snapshot["patch.diff"])
        provider = json.loads(snapshot["provider.json"])
        self.assertEqual("codex", provider["provider_id"])
        self.assertEqual("success", provider["status"])
        self.assertEqual("success", json.loads(snapshot["outcome.json"])["status"])
        self.assertEqual(
            {
                "currency": "USD",
                "estimated": False,
                "input_tokens": 0,
                "known": False,
                "output_tokens": 0,
                "total_cost_usd": 0.0,
            },
            json.loads(snapshot["cost.json"]),
        )
        transcript = [json.loads(line) for line in snapshot["transcript.jsonl"].splitlines()]
        self.assertEqual(["execution_started", "process_output_captured", "execution_completed"], [event["event"] for event in transcript])
        self.assertEqual("1. inspect\n", transcript[1]["stdout"])
        self.assertEqual(9.0, runner.calls[0]["timeout"])


class RecordingProcessRunner:
    def __init__(self, result: ProcessResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def run(self, argv, *, cwd=None, env=None, timeout=None):
        self.calls.append(
            {
                "argv": tuple(argv),
                "cwd": cwd,
                "env": None if env is None else dict(env),
                "timeout": timeout,
            }
        )
        return self.result


def _read_artifacts(root: Path) -> dict[str, str]:
    return {
        name: (root / name).read_text(encoding="utf-8")
        for name in RUN_ARTIFACT_FILES
    }


if __name__ == "__main__":
    unittest.main()
