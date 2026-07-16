import unittest
from pathlib import Path

from agentx.config import AgentXPaths, Settings
from agentx.providers import ProviderStatus
from agentx.routing import AgentRun, Router


def settings(public_providers=()):
    paths = AgentXPaths(
        root=Path("/state"),
        settings=Path("/state/settings.json"),
        sessions=Path("/state/sessions"),
        memories=Path("/state/memories"),
        auth=Path("/state/auth"),
    )
    return Settings(paths=paths, public_providers=tuple(public_providers))


class RouterTests(unittest.TestCase):
    def test_selects_first_enabled_provider(self):
        decision = Router(
            settings(),
            (
                ProviderStatus("codex", "Codex", "cli", True, "available"),
                ProviderStatus("claude", "Claude", "cli", True, "available"),
            ),
        ).explain(AgentRun(prompt="test"))

        self.assertEqual("codex", decision.selected_provider)
        self.assertEqual("high", decision.selected_model_tier)

    def test_model_tier_defaults_to_economy_for_docs(self):
        decision = Router(
            settings(),
            (ProviderStatus("codex", "Codex", "cli", True, "available"),),
        ).explain(AgentRun(prompt="write docs", mode="docs"))

        self.assertEqual("economy", decision.selected_model_tier)

    def test_public_provider_defaults_filter_providers(self):
        decision = Router(
            settings(public_providers=("claude",)),
            (
                ProviderStatus("codex", "Codex", "cli", True, "available"),
                ProviderStatus("claude", "Claude", "cli", True, "available"),
            ),
        ).explain(AgentRun(prompt="test"))

        self.assertEqual("claude", decision.selected_provider)
        self.assertEqual("not_in_public_provider_defaults", decision.rejected_providers["codex"])


if __name__ == "__main__":
    unittest.main()
