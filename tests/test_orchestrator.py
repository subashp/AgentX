import json
import shutil
import unittest
from pathlib import Path

from agentx.adapters import AdapterRequest, AdapterResult
from agentx.config import AgentXPaths, Settings
from agentx.context import MemoryRecord
from agentx.orchestrator import execute_plan_mode
from agentx.policy import Policy
from agentx.providers import ProviderStatus
from agentx.routing import AgentRun
from agentx.store import RUN_ARTIFACT_FILES, SessionStore


class OrchestratorFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self.fixture_root = Path("tests") / ".tmp_orchestrator"
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)
        self.fixture_root.mkdir(parents=True)

    def tearDown(self):
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)

    def make_settings(self):
        root = self.fixture_root / "state"
        return Settings(
            paths=AgentXPaths(
                root=root,
                settings=root / "settings.json",
                sessions=root / "sessions",
                memories=root / "memories",
                auth=root / "auth",
            ),
            public_providers=("fake-local",),
        )


class PlanModeOrchestratorTests(OrchestratorFixtureTestCase):
    maxDiff = None

    def test_fake_plan_mode_writes_route_context_and_run_artifacts(self):
        settings = self.make_settings()

        result = execute_plan_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id="plan-e2e",
            prompt="Plan AX-013",
            context_paths=("tests/test_orchestrator.py",),
            memories=(
                MemoryRecord(
                    id="public-note",
                    classification="public",
                    content="Use fake provider for default tests.",
                ),
            ),
        )

        snapshot = _read_artifacts(result.stored_run.root)
        self.assertEqual(set(RUN_ARTIFACT_FILES), set(result.stored_run.artifact_paths))
        self.assertEqual("success", result.stored_run.result.status)
        self.assertEqual("fake-local", result.route.selected_provider)
        self.assertIn("Selected provider 'fake-local'", result.route.explanation)
        self.assertEqual("", snapshot["patch.diff"])

        manifest = json.loads(snapshot["manifest.json"])
        self.assertEqual("plan-e2e", manifest["session_id"])
        self.assertEqual("fake-local", manifest["route"]["selected_provider"])
        self.assertIn("Selected provider 'fake-local'", manifest["route"]["explanation"])

        context_map = json.loads(snapshot["context-map.json"])
        self.assertEqual("compiled_context_manifest", context_map["source"])
        self.assertEqual(["tests/test_orchestrator.py"], context_map["requested_paths"])
        self.assertEqual(["tests/test_orchestrator.py"], context_map["included_paths"])
        self.assertEqual([], context_map["excluded_paths"])
        self.assertIn("Selected provider 'fake-local'", context_map["route"]["explanation"])

        memory_map = json.loads(snapshot["memory-map.json"])
        self.assertEqual(["public-note"], memory_map["memory_ids"])
        self.assertEqual(["public-note"], memory_map["included_memories"])
        self.assertEqual([], memory_map["redacted_memories"])
        self.assertEqual([], json.loads(snapshot["redactions.json"]))

    def test_external_plan_context_redacts_policy_blocked_paths_and_memories(self):
        settings = self.make_settings()
        policy = Policy(
            private_provider="private-local",
            classification_rules={
                "tests/**": "internal",
                "src/customer/**": "confidential",
                "src/core/**": "proprietary",
            },
            routing={
                "confidential": ("private-local",),
                "proprietary": ("private-local",),
            },
        )
        run = AgentRun(
            prompt="Plan customer scheduler refactor",
            mode="plan",
            provider="fake-local",
            context_paths=(
                "tests/test_orchestrator.py",
                "src/customer/account.py",
                "src/core/planner.py",
            ),
        )

        result = execute_plan_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id="policy-redactions",
            run=run,
            policy=policy,
            memories=(
                MemoryRecord(
                    id="customer-note",
                    classification="confidential",
                    content="Customer-specific escalation detail.",
                    summary="Customer escalation exists; keep account terms generic.",
                ),
                MemoryRecord(
                    id="planner-note",
                    classification="proprietary",
                    content="Private planner heuristic.",
                ),
            ),
        )

        snapshot = _read_artifacts(result.stored_run.root)
        context_map = json.loads(snapshot["context-map.json"])
        memory_map = json.loads(snapshot["memory-map.json"])
        redactions = json.loads(snapshot["redactions.json"])

        self.assertEqual("external", result.provider_class)
        self.assertEqual(["tests/test_orchestrator.py"], context_map["included_paths"])
        self.assertEqual(
            ["src/customer/account.py", "src/core/planner.py"],
            context_map["excluded_paths"],
        )
        self.assertFalse(context_map["policy_decision"]["eligible"])
        self.assertEqual(["customer-note"], memory_map["summarized_memories"])
        self.assertEqual(["planner-note"], memory_map["redacted_memories"])
        self.assertTrue(
            any(
                entry["target_type"] == "path"
                and entry["target_id"] == "src/core/planner.py"
                and entry["action"] == "exclude"
                for entry in redactions
            )
        )
        self.assertTrue(
            any(
                entry["target_type"] == "memory"
                and entry["target_id"] == "planner-note"
                and entry["action"] == "redact"
                for entry in redactions
            )
        )
        self.assertEqual("", snapshot["patch.diff"])

    def test_plan_mode_suppresses_patch_content_from_supplied_adapter(self):
        settings = self.make_settings()
        result = execute_plan_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id="patch-suppressed",
            prompt="Plan only",
            adapter=PatchyAdapter(),
        )

        snapshot = _read_artifacts(result.stored_run.root)
        outcome = json.loads(snapshot["outcome.json"])
        self.assertEqual("", snapshot["patch.diff"])
        self.assertEqual("", result.stored_run.result.patch)
        self.assertTrue(outcome["patch_suppressed"])
        self.assertFalse(outcome["patch_applied"])

    def test_injected_fake_dependencies_do_not_require_live_provider_registry(self):
        settings = self.make_settings()
        statuses = (
            ProviderStatus(
                id="fake-local",
                display_name="Fake Local",
                kind="local",
                enabled=True,
                reason="available",
            ),
        )

        result = execute_plan_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id="injected-fake",
            prompt="No live providers",
            provider_statuses=statuses,
        )

        self.assertEqual("fake-local", result.route.selected_provider)
        self.assertEqual("success", result.stored_run.result.status)


class PatchyAdapter:
    provider_id = "fake-local"

    def execute(self, request: AdapterRequest) -> AdapterResult:
        return AdapterResult(
            provider_id=self.provider_id,
            model_id="patchy",
            model_tier="high",
            status="success",
            transcript_events=(
                {"sequence": 1, "event": "execution_completed", "status": "success"},
            ),
            cost={"currency": "USD", "estimated": False, "total_cost_usd": 0.0},
            outcome={"status": "success", "patch_applied": True},
            patch="diff --git a/file b/file\n",
        )


def _read_artifacts(root: Path) -> dict[str, str]:
    return {
        name: (root / name).read_text(encoding="utf-8")
        for name in RUN_ARTIFACT_FILES
    }


if __name__ == "__main__":
    unittest.main()
