import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from agentx.adapters import AdapterResult, execute_fake_run
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


def _read_artifacts(root: Path) -> dict[str, str]:
    return {
        name: (root / name).read_text(encoding="utf-8")
        for name in RUN_ARTIFACT_FILES
    }


if __name__ == "__main__":
    unittest.main()
