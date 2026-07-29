import unittest
from pathlib import Path
from unittest import mock

from agentx.config import AgentXPaths, ProviderSettings, Settings
from agentx.providers import ProviderDefinition, ProviderRegistry


def settings(providers):
    paths = AgentXPaths(
        root=Path("/state"),
        settings=Path("/state/settings.json"),
        sessions=Path("/state/sessions"),
        memories=Path("/state/memories"),
        auth=Path("/state/auth"),
    )
    return Settings(paths=paths, providers=providers)


class ProviderRegistryTests(unittest.TestCase):
    def test_builtin_fake_local_provider_is_always_available(self):
        registry = ProviderRegistry(
            [ProviderDefinition("fake-local", "AgentX Fake Local", "builtin", None)]
        )

        statuses = registry.list_statuses()

        self.assertTrue(statuses[0].enabled)
        self.assertEqual("available", statuses[0].reason)
        self.assertEqual("builtin", statuses[0].kind)
        self.assertIsNone(statuses[0].command)

    def test_cli_provider_is_enabled_when_command_resolves(self):
        registry = ProviderRegistry(
            [ProviderDefinition("codex", "Codex", "cli", "codex")],
            executable_search=lambda command: f"/bin/{command}",
        )

        statuses = registry.list_statuses()

        self.assertTrue(statuses[0].enabled)
        self.assertEqual("available", statuses[0].reason)

    def test_cli_provider_is_disabled_when_command_is_missing(self):
        registry = ProviderRegistry(
            [ProviderDefinition("claude", "Claude", "cli", "claude")],
            executable_search=lambda command: None,
        )

        statuses = registry.list_statuses()

        self.assertFalse(statuses[0].enabled)
        self.assertEqual("disabled_missing_binary", statuses[0].reason)

    def test_private_endpoint_is_disabled_until_configured(self):
        registry = ProviderRegistry(
            [ProviderDefinition("private", "Private", "openai_compatible", None, public=False)]
        )

        statuses = registry.list_statuses()

        self.assertFalse(statuses[0].enabled)
        self.assertEqual("endpoint_not_configured", statuses[0].reason)

    def test_configured_command_overrides_default_command(self):
        registry = ProviderRegistry(
            [ProviderDefinition("codex", "Codex", "cli", "codex")],
            settings=settings({"codex": ProviderSettings(command="/tools/codex")}),
            executable_search=lambda command: command if command == "/tools/codex" else None,
        )

        statuses = registry.list_statuses()

        self.assertTrue(statuses[0].enabled)
        self.assertEqual("/tools/codex", statuses[0].command)
        self.assertEqual("/tools/codex", statuses[0].resolved_command)

    def test_provider_can_be_disabled_by_settings(self):
        registry = ProviderRegistry(
            [ProviderDefinition("codex", "Codex", "cli", "codex")],
            settings=settings({"codex": ProviderSettings(enabled=False)}),
            executable_search=lambda command: f"/bin/{command}",
        )

        statuses = registry.list_statuses()

        self.assertFalse(statuses[0].enabled)
        self.assertEqual("disabled_by_settings", statuses[0].reason)

    def test_auth_failure_disables_provider(self):
        registry = ProviderRegistry(
            [ProviderDefinition("codex", "Codex", "cli", "codex")],
            settings=settings({"codex": ProviderSettings(auth_check="codex-auth")}),
            executable_search=lambda command: f"/bin/{command}",
            auth_check=lambda check_id: False,
        )

        statuses = registry.list_statuses()

        self.assertFalse(statuses[0].enabled)
        self.assertEqual("disabled_missing_auth", statuses[0].reason)
        self.assertEqual({"auth": False}, statuses[0].checks)

    def test_subscription_failure_disables_provider(self):
        registry = ProviderRegistry(
            [ProviderDefinition("claude", "Claude", "cli", "claude")],
            settings=settings({"claude": ProviderSettings(subscription_check="claude-subscription")}),
            executable_search=lambda command: f"/bin/{command}",
            subscription_check=lambda check_id: False,
        )

        statuses = registry.list_statuses()

        self.assertFalse(statuses[0].enabled)
        self.assertEqual("disabled_missing_subscription", statuses[0].reason)
        self.assertEqual({"subscription": False}, statuses[0].checks)

    def test_configured_private_endpoint_can_be_available(self):
        registry = ProviderRegistry(
            [ProviderDefinition("private", "Private", "openai_compatible", None, public=False)],
            settings=settings(
                {
                    "private": ProviderSettings(
                        endpoint="http://localhost:8000/v1",
                        model="local-coder",
                    )
                }
            ),
            endpoint_check=lambda endpoint: True,
        )

        statuses = registry.list_statuses()

        self.assertTrue(statuses[0].enabled)
        self.assertEqual("available", statuses[0].reason)
        self.assertEqual({"endpoint": True}, statuses[0].checks)

    def test_unhealthy_private_endpoint_is_disabled(self):
        registry = ProviderRegistry(
            [ProviderDefinition("private", "Private", "openai_compatible", None, public=False)],
            settings=settings(
                {
                    "private": ProviderSettings(
                        endpoint="http://localhost:8000/v1",
                        model="local-coder",
                    )
                }
            ),
            endpoint_check=lambda endpoint: False,
        )

        statuses = registry.list_statuses()

        self.assertFalse(statuses[0].enabled)
        self.assertEqual("disabled_unhealthy", statuses[0].reason)
        self.assertEqual({"endpoint": False}, statuses[0].checks)

    def test_private_endpoint_can_be_loaded_from_environment(self):
        registry = ProviderRegistry(
            [ProviderDefinition("private", "Private", "openai_compatible", None, public=False)],
            settings=settings(
                {
                    "private": ProviderSettings(
                        endpoint_env="AGENTX_TEST_ENDPOINT",
                        model="local-coder",
                    )
                }
            ),
            endpoint_check=lambda endpoint: endpoint == "http://127.0.0.1:8000/v1",
        )

        with mock.patch.dict("os.environ", {"AGENTX_TEST_ENDPOINT": "http://127.0.0.1:8000/v1"}):
            statuses = registry.list_statuses()

        self.assertTrue(statuses[0].enabled)
        self.assertEqual("http://127.0.0.1:8000/v1", statuses[0].endpoint)

    def test_default_private_endpoint_check_queries_models(self):
        registry = ProviderRegistry(
            [ProviderDefinition("private", "Private", "openai_compatible", None, public=False)],
            settings=settings(
                {
                    "private": ProviderSettings(
                        endpoint="https://model.example/v1",
                        model="local-coder",
                    )
                }
            ),
        )
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"data":[{"id":"local-coder"}]}'

        with mock.patch("agentx.providers.urllib.request.urlopen", return_value=response) as opener:
            statuses = registry.list_statuses()

        self.assertTrue(statuses[0].enabled)
        request = opener.call_args.args[0]
        self.assertEqual("https://model.example/v1/models", request.full_url)
        self.assertEqual("true", request.get_header("Ngrok-skip-browser-warning"))


if __name__ == "__main__":
    unittest.main()
