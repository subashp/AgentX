import io
import json
import unittest
from unittest import mock

from agentx import cli


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
        self.assertEqual("no_eligible_provider", payload["reason"])

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


if __name__ == "__main__":
    unittest.main()
