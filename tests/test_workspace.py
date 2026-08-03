import shutil
import unittest
from pathlib import Path

from agentx.workspace import (
    MarkerSecretRule,
    MarkerSecretScanner,
    ScopedWorkspaceConfig,
    WithheldPathSummary,
    WorkspaceError,
    materialize_scoped_workspace,
    normalize_scoped_path,
    validate_patch_paths,
)


class WorkspacePathTests(unittest.TestCase):
    def test_normalize_scoped_path_rejects_absolute_and_traversal_paths(self):
        invalid_paths = [
            "../secret.txt",
            "src/../secret.txt",
            "/absolute/file.py",
            "\\absolute\\file.py",
            "C:\\absolute\\file.py",
            "C:relative\\file.py",
        ]

        for invalid_path in invalid_paths:
            with self.subTest(invalid_path=invalid_path):
                with self.assertRaises(WorkspaceError):
                    normalize_scoped_path(invalid_path)

    def test_duplicate_path_aliases_are_rejected(self):
        with self.assertRaises(WorkspaceError):
            ScopedWorkspaceConfig.plan(
                source_root=Path("source"),
                workspace_root=Path("workspace"),
                allowed_paths=["src/module.py", "./src//module.py"],
            )

    def test_case_ambiguous_path_aliases_are_rejected(self):
        with self.assertRaises(WorkspaceError):
            ScopedWorkspaceConfig.plan(
                source_root=Path("source"),
                workspace_root=Path("workspace"),
                allowed_paths=["src/Module.py", "src/module.py"],
            )


class WorkspaceMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.fixture_root = Path("tests") / ".tmp_workspace"
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)
        self.fixture_root.mkdir(parents=True)

    def tearDown(self):
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)

    def test_materializes_allowed_files_and_withheld_placeholders(self):
        source = self.fixture_root / "source"
        workspace = self.fixture_root / "workspace"
        (source / "src").mkdir(parents=True)
        (source / "secrets").mkdir()
        (source / "src" / "public.py").write_text("print('visible')\n", encoding="utf-8")
        (source / "secrets" / "token.txt").write_text(
            "raw-secret-value\n",
            encoding="utf-8",
        )

        result = materialize_scoped_workspace(
            ScopedWorkspaceConfig.plan(
                source_root=source,
                workspace_root=workspace,
                allowed_paths=["src/public.py"],
                withheld_paths=[
                    WithheldPathSummary(
                        path="secrets/token.txt",
                        classification="secret",
                        reason="external_provider_scope",
                        summary="A secret file exists but its content is withheld.",
                    )
                ],
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(
            "print('visible')\n",
            (workspace / "src" / "public.py").read_text(encoding="utf-8"),
        )
        placeholder = (workspace / "secrets" / "token.txt").read_text(encoding="utf-8")
        self.assertIn("A secret file exists but its content is withheld.", placeholder)
        self.assertIn("Classification: secret", placeholder)
        self.assertNotIn("raw-secret-value", placeholder)
        self.assertEqual(
            ["copied_file", "summary_placeholder"],
            [entry.kind for entry in result.entries],
        )


class PatchValidationTests(unittest.TestCase):
    def test_in_scope_patch_is_accepted(self):
        patch = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-old
+new
"""

        result = validate_patch_paths(
            patch,
            allowed_paths=["src/app.py"],
        )

        self.assertTrue(result.accepted)
        self.assertEqual(("src/app.py",), result.paths)

    def test_out_of_scope_patch_is_rejected(self):
        patch = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-old
+new
"""

        result = validate_patch_paths(
            patch,
            allowed_paths=["tests/test_app.py"],
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            ["patch_path_out_of_scope"],
            [event.code for event in result.events if event.severity == "error"],
        )

    def test_denied_patch_path_is_rejected(self):
        patch = """diff --git a/secrets/token.txt b/secrets/token.txt
--- a/secrets/token.txt
+++ b/secrets/token.txt
@@ -1 +1 @@
-old
+new
"""

        result = validate_patch_paths(
            patch,
            allowed_paths=["secrets/token.txt"],
            denied_paths=["secrets/token.txt"],
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            ["patch_path_denied"],
            [event.code for event in result.events if event.severity == "error"],
        )

    def test_patch_path_traversal_and_absolute_paths_are_rejected(self):
        patch = """diff --git a/src/app.py b/../private.txt
--- a/src/app.py
+++ /absolute/private.txt
@@ -1 +1 @@
-old
+new
"""

        result = validate_patch_paths(
            patch,
            allowed_paths=["src/app.py"],
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            ["patch_path_invalid", "patch_path_invalid"],
            [event.code for event in result.events if event.severity == "error"],
        )
        self.assertEqual(("src/app.py",), result.paths)

    def test_secret_marker_scan_rejects_patch_without_exposing_marker_value(self):
        patch = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1,2 @@
 print('visible')
+TOKEN = 'do-not-expose-this-marker'
"""

        result = validate_patch_paths(
            patch,
            allowed_paths=["src/app.py"],
            secret_scanner=MarkerSecretScanner(
                rules=(
                    MarkerSecretRule(
                        marker="do-not-expose-this-marker",
                        marker_class="fixture_secret",
                    ),
                )
            ),
        )

        self.assertFalse(result.accepted)
        payload = result.as_dict()
        self.assertEqual(
            [{"line_number": 6, "marker_class": "fixture_secret"}],
            payload["secret_findings"],
        )
        self.assertNotIn("do-not-expose-this-marker", repr(payload))

    def test_malformed_patch_without_target_paths_is_rejected(self):
        result = validate_patch_paths(
            "not a unified diff",
            allowed_paths=["src/app.py"],
        )

        self.assertFalse(result.accepted)
        self.assertEqual((), result.paths)
        self.assertIn("patch_no_target_paths", [event.code for event in result.events])

    def test_malformed_patch_without_hunk_is_rejected(self):
        patch = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
"""

        result = validate_patch_paths(
            patch,
            allowed_paths=["src/app.py"],
        )

        self.assertFalse(result.accepted)
        self.assertEqual(("src/app.py",), result.paths)
        self.assertIn("patch_hunk_missing", [event.code for event in result.events])


if __name__ == "__main__":
    unittest.main()
