import unittest

from agentx.providers import ProviderDefinition, ProviderRegistry


class ProviderRegistryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
