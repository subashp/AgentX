import json
import shutil
import unittest
from pathlib import Path

from agentx.adapters import AdapterRequest, AdapterResult
from agentx.config import AgentXPaths, Settings
from agentx.context import MemoryRecord
from agentx.orchestrator import execute_execute_mode, execute_plan_mode
from agentx.policy import Policy
from agentx.providers import ProviderStatus
from agentx.routing import AgentRun
from agentx.store import RUN_ARTIFACT_FILES, SessionStore
from agentx.workspace import MarkerSecretRule, MarkerSecretScanner


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


class ExecuteModeOrchestratorTests(OrchestratorFixtureTestCase):
    maxDiff = None

    def test_fake_execute_success_with_empty_patch_is_accepted(self):
        settings = self.make_settings()

        result = execute_execute_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id="execute-empty",
            prompt="Do nothing safely",
            allowed_patch_paths=("src/app.py",),
        )

        snapshot = _read_artifacts(result.stored_run.root)
        outcome = json.loads(snapshot["outcome.json"])
        self.assertEqual(set(RUN_ARTIFACT_FILES), set(result.stored_run.artifact_paths))
        self.assertEqual("execute", result.run.mode)
        self.assertEqual("success", result.stored_run.result.status)
        self.assertTrue(result.patch_validation.accepted)
        self.assertEqual("", snapshot["patch.diff"])
        self.assertFalse(outcome["patch_present"])
        self.assertFalse(outcome["patch_stored"])
        self.assertFalse(outcome["patch_applied"])
        self.assertFalse(outcome["patch_application"]["supported"])
        self.assertTrue(outcome["patch_validation"]["accepted"])

    def test_in_scope_fake_patch_is_accepted_but_not_applied(self):
        settings = self.make_settings()
        target = self.fixture_root / "src" / "app.py"
        target.parent.mkdir(parents=True)
        target.write_text("old\n", encoding="utf-8")
        patch = """diff --git a/tests/.tmp_orchestrator/src/app.py b/tests/.tmp_orchestrator/src/app.py
--- a/tests/.tmp_orchestrator/src/app.py
+++ b/tests/.tmp_orchestrator/src/app.py
@@ -1 +1 @@
-old
+new
"""

        result = execute_execute_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id="execute-accepted",
            prompt="Patch within scope",
            allowed_patch_paths=("tests/.tmp_orchestrator/src/app.py",),
            adapter=PatchTextAdapter(patch),
        )

        snapshot = _read_artifacts(result.stored_run.root)
        outcome = json.loads(snapshot["outcome.json"])
        self.assertEqual(patch, snapshot["patch.diff"])
        self.assertEqual("old\n", target.read_text(encoding="utf-8"))
        self.assertTrue(result.patch_validation.accepted)
        self.assertTrue(outcome["patch_accepted"])
        self.assertTrue(outcome["patch_stored"])
        self.assertFalse(outcome["patch_applied"])
        self.assertEqual(
            ["tests/.tmp_orchestrator/src/app.py"],
            outcome["patch_validation"]["paths"],
        )

    def test_out_of_scope_patch_is_rejected_and_not_stored(self):
        settings = self.make_settings()
        patch = _patch_for("src/app.py")

        result = execute_execute_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id="execute-out-of-scope",
            prompt="Patch outside scope",
            allowed_patch_paths=("tests/test_app.py",),
            adapter=PatchTextAdapter(patch),
        )

        snapshot = _read_artifacts(result.stored_run.root)
        outcome = json.loads(snapshot["outcome.json"])
        self.assertEqual("validation_failed", result.stored_run.result.status)
        self.assertEqual("", snapshot["patch.diff"])
        self.assertFalse(result.patch_validation.accepted)
        self.assertFalse(outcome["patch_accepted"])
        self.assertFalse(outcome["patch_stored"])
        self.assertEqual("patch_validation_failed", outcome["outcome"])
        self.assertEqual(
            ["patch_path_out_of_scope"],
            [
                event["code"]
                for event in outcome["patch_validation"]["events"]
                if event["severity"] == "error"
            ],
        )

    def test_denied_patch_path_is_rejected_and_not_stored(self):
        settings = self.make_settings()
        patch = _patch_for("src/app.py")

        result = execute_execute_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id="execute-denied",
            prompt="Patch denied path",
            allowed_patch_paths=("src/app.py",),
            denied_patch_paths=("src/app.py",),
            adapter=PatchTextAdapter(patch),
        )

        snapshot = _read_artifacts(result.stored_run.root)
        outcome = json.loads(snapshot["outcome.json"])
        self.assertEqual("validation_failed", result.stored_run.result.status)
        self.assertEqual("", snapshot["patch.diff"])
        self.assertFalse(result.patch_validation.accepted)
        self.assertEqual(
            ["patch_path_denied"],
            [
                event["code"]
                for event in outcome["patch_validation"]["events"]
                if event["severity"] == "error"
            ],
        )

    def test_secret_marker_patch_is_rejected_without_storing_marker_text(self):
        settings = self.make_settings()
        marker = "do-not-store-this-fixture-secret"
        patch = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1,2 @@
 old
+TOKEN = 'do-not-store-this-fixture-secret'
"""

        result = execute_execute_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id="execute-secret",
            prompt="Patch with secret marker",
            allowed_patch_paths=("src/app.py",),
            adapter=PatchTextAdapter(patch),
            secret_scanner=MarkerSecretScanner(
                rules=(
                    MarkerSecretRule(
                        marker=marker,
                        marker_class="fixture_secret",
                    ),
                )
            ),
        )

        snapshot = _read_artifacts(result.stored_run.root)
        outcome = json.loads(snapshot["outcome.json"])
        self.assertEqual("validation_failed", result.stored_run.result.status)
        self.assertEqual("", snapshot["patch.diff"])
        self.assertFalse(result.patch_validation.accepted)
        self.assertEqual(
            [{"line_number": 6, "marker_class": "fixture_secret"}],
            outcome["patch_validation"]["secret_findings"],
        )
        self.assertNotIn(marker, "".join(snapshot.values()))


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


class PatchTextAdapter:
    provider_id = "fake-local"

    def __init__(self, patch: str):
        self.patch = patch

    def execute(self, request: AdapterRequest) -> AdapterResult:
        return AdapterResult(
            provider_id=self.provider_id,
            model_id="patch-text",
            model_tier="standard",
            status="success",
            transcript_events=(
                {"sequence": 1, "event": "execution_completed", "status": "success"},
            ),
            cost={"currency": "USD", "estimated": False, "total_cost_usd": 0.0},
            outcome={"status": "success", "outcome": "patch_text_completed"},
            patch=self.patch,
        )


def _patch_for(path: str) -> str:
    return f"""diff --git a/{path} b/{path}
--- a/{path}
+++ b/{path}
@@ -1 +1 @@
-old
+new
"""


def _read_artifacts(root: Path) -> dict[str, str]:
    return {
        name: (root / name).read_text(encoding="utf-8")
        for name in RUN_ARTIFACT_FILES
    }


if __name__ == "__main__":
    unittest.main()
