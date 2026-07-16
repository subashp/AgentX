import io
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from agentx import cli
from agentx.config import AgentXPaths, Settings


class CliTests(unittest.TestCase):
    def test_config_path_outputs_json(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = cli.run(["--json", "config", "path"], stdout, stderr)

        self.assertEqual(0, code)
        self.assertEqual("", stderr.getvalue())
        self.assertIn("settings", json.loads(stdout.getvalue()))

    def test_prompt_shorthand_returns_route_decision(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("agentx.cli.ProviderRegistry") as registry:
            registry.return_value.list_statuses.return_value = ()
            code = cli.run(["--json", "fix tests"], stdout, stderr)

        self.assertEqual(0, code)
        payload = json.loads(stdout.getvalue())
        self.assertEqual("fix tests", payload["run"]["prompt"])
        self.assertEqual({"max_cost_usd": None, "max_input_tokens": None, "max_output_tokens": None}, payload["run"]["budget"])
        self.assertEqual("no_eligible_provider", payload["reason"])

    def test_route_rejects_invalid_mode(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("agentx.cli.ProviderRegistry") as registry:
            registry.return_value.list_statuses.return_value = ()
            code = cli.run(["route", "--mode", "ship", "fix tests"], stdout, stderr)

        self.assertEqual(2, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("invalid mode 'ship'", stderr.getvalue())

    def test_route_rejects_missing_prompt(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("agentx.cli.ProviderRegistry") as registry:
            registry.return_value.list_statuses.return_value = ()
            code = cli.run(["route", "--mode", "plan"], stdout, stderr)

        self.assertEqual(2, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("prompt is required", stderr.getvalue())

    def test_providers_list_text_output(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = cli.run(["providers", "list"], stdout, stderr)

        self.assertEqual(0, code)
        self.assertIn("codex", stdout.getvalue())

    def test_providers_list_uses_loaded_settings(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("agentx.cli.load_settings") as load_settings:
            settings = mock.Mock()
            settings.providers = {}
            load_settings.return_value = settings
            code = cli.run(["--json", "providers", "list"], stdout, stderr)

        self.assertEqual(0, code)
        load_settings.assert_called_once()
        payload = json.loads(stdout.getvalue())
        self.assertTrue(any(provider["id"] == "codex" for provider in payload))

    def test_run_fake_writes_artifacts_without_provider_registry(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_fake"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

        try:
            settings = Settings(
                paths=AgentXPaths(
                    root=root,
                    settings=root / "settings.json",
                    sessions=root / "sessions",
                    memories=root / "memories",
                    auth=root / "auth",
                )
            )
            with mock.patch("agentx.cli.load_settings", return_value=settings):
                with mock.patch("agentx.cli.ProviderRegistry") as registry:
                    registry.side_effect = AssertionError("provider registry was used")
                    code = cli.run(
                        [
                            "--json",
                            "run",
                            "--fake",
                            "--session-id",
                            "cli-fake",
                            "--mode",
                            "execute",
                            "local dry run",
                        ],
                        stdout,
                        stderr,
                    )

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual("cli-fake", payload["session_id"])
            self.assertEqual("success", payload["result"]["status"])
            self.assertEqual("fake-local", payload["result"]["provider_id"])
            self.assertTrue((root / "sessions" / "cli-fake" / "manifest.json").exists())
            self.assertEqual(
                "local dry run\n",
                (root / "sessions" / "cli-fake" / "prompt.md").read_text(encoding="utf-8"),
            )
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
