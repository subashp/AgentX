import json
import shutil
import unittest
from pathlib import Path

from agentx.config import AgentXPaths, ProviderSettings, Settings
from agentx.context import MemoryRecord
from agentx.store import (
    RUN_ARTIFACT_FILES,
    MemoryStore,
    SessionStore,
    SettingsStore,
    resolve_auth_service_path,
)


class StoreFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self.fixture_root = Path("tests") / ".tmp_store"
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)
        self.fixture_root.mkdir(parents=True)

    def tearDown(self):
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)

    def make_paths(self, **overrides):
        root = self.fixture_root / "state"
        return AgentXPaths(
            root=root,
            settings=overrides.get("settings", root / "settings.json"),
            sessions=overrides.get("sessions", root / "sessions"),
            memories=overrides.get("memories", root / "memories"),
            auth=overrides.get("auth", root / "auth"),
        )


class SettingsStoreTests(StoreFixtureTestCase):
    def test_settings_store_writes_json_to_resolved_override_path(self):
        paths = self.make_paths(settings=self.fixture_root / "config" / "settings.json")
        settings = Settings(
            paths=paths,
            public_providers=("claude", "codex"),
            private_provider="private-local",
            providers={
                "codex": ProviderSettings(
                    command="codex",
                    auth_check="codex-auth",
                )
            },
        )

        written = SettingsStore(paths).write(settings)

        self.assertEqual(paths.settings, written)
        self.assertTrue(written.exists())
        self.assertEqual(
            {
                "external_max_classification": "internal",
                "private_provider": "private-local",
                "providers": {
                    "codex": {
                        "auth_check": "codex-auth",
                        "command": "codex",
                        "enabled": True,
                        "endpoint": None,
                        "subscription_check": None,
                    }
                },
                "public_providers": ["claude", "codex"],
            },
            json.loads(written.read_text(encoding="utf-8")),
        )


class SessionAndRunStoreTests(StoreFixtureTestCase):
    def test_session_store_uses_supplied_session_root_and_writes_expected_artifacts(self):
        paths = self.make_paths(sessions=self.fixture_root / "custom-sessions")
        run = SessionStore(paths).open_run("run-001")

        artifact_paths = run.write_artifacts(
            manifest={"schema_version": 1, "session_id": "run-001"},
            prompt="# Fix the tests\n",
            context_map={"included_paths": ["src/agentx/store.py"]},
            memory_map={"memory_ids": ["memory-a"]},
            redactions=[],
            provider={"provider_id": "codex", "model_id": "gpt-5"},
            transcript=[
                {"event": "started", "sequence": 1},
                {"event": "completed", "sequence": 2},
            ],
            patch="diff --git a/file b/file\n",
            cost={"input_tokens": 12, "output_tokens": 34},
            outcome={"status": "success"},
        )

        self.assertEqual(paths.sessions / "run-001", run.root)
        self.assertTrue(run.root.is_dir())
        self.assertEqual(set(RUN_ARTIFACT_FILES), set(path.name for path in artifact_paths.values()))
        self.assertEqual(
            '{\n  "schema_version": 1,\n  "session_id": "run-001"\n}\n',
            (run.root / "manifest.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            '# Fix the tests\n',
            (run.root / "prompt.md").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            [
                {"event": "started", "sequence": 1},
                {"event": "completed", "sequence": 2},
            ],
            [
                json.loads(line)
                for line in (run.root / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
            ],
        )

    def test_run_store_can_append_transcript_events(self):
        run = SessionStore(self.make_paths()).open_run("run-002")

        run.append_transcript_event({"sequence": 1, "event": "started"})
        run.append_transcript_event({"sequence": 2, "event": "completed"})

        self.assertEqual(
            [
                {"event": "started", "sequence": 1},
                {"event": "completed", "sequence": 2},
            ],
            [
                json.loads(line)
                for line in (run.root / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
            ],
        )


class MemoryStoreTests(StoreFixtureTestCase):
    def test_memory_store_crud_round_trip_uses_local_json_files(self):
        store = MemoryStore(self.make_paths())
        alpha = MemoryRecord(
            id="alpha-note",
            classification="public",
            content="alpha content",
            summary="alpha summary",
        )
        beta = MemoryRecord(
            id="beta-note",
            classification="confidential",
            content="beta content",
        )

        alpha_path = store.write(alpha)
        store.write(beta)
        store.write(
            MemoryRecord(
                id="alpha-note",
                classification="public",
                content="updated alpha content",
                summary="alpha summary",
            )
        )

        self.assertEqual(store.root / "alpha-note.json", alpha_path)
        self.assertEqual(["alpha-note", "beta-note"], [memory.id for memory in store.list()])
        self.assertEqual("updated alpha content", store.read("alpha-note").content)
        self.assertTrue(store.delete("beta-note"))
        self.assertFalse(store.delete("beta-note"))
        self.assertEqual(["alpha-note"], [memory.id for memory in store.list()])
        self.assertEqual(
            {
                "id": "alpha-note",
                "classification": "public",
                "content": "updated alpha content",
                "summary": "alpha summary",
            },
            json.loads(alpha_path.read_text(encoding="utf-8")),
        )


class AuthPathResolutionTests(StoreFixtureTestCase):
    def test_auth_service_path_uses_supplied_auth_root(self):
        paths = self.make_paths(auth=self.fixture_root / "service-auth")

        resolved = resolve_auth_service_path(paths, "github")

        self.assertEqual(self.fixture_root / "service-auth" / "github", resolved)


if __name__ == "__main__":
    unittest.main()
