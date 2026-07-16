import unittest
from pathlib import Path

from agentx.config import AgentXPaths, Settings
from agentx.providers import ProviderStatus
from agentx.routing import AgentRun, RouteValidationError, Router


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
    def test_agent_run_normalizes_contract_fields(self):
        run = AgentRun(
            prompt="  Write docs  ",
            mode=" DoCs ",
            provider=" AUTO ",
            model_tier=" auto ",
            budget={
                "max_input_tokens": 1200,
                "max_output_tokens": 400,
                "max_cost_usd": 1.5,
            },
            required_tools=["git", " shell ", "git"],
            required_mcp_servers="github",
            required_mcp_tools=["browser.open"],
        )

        self.assertEqual("Write docs", run.prompt)
        self.assertEqual("docs", run.mode)
        self.assertEqual("auto", run.provider)
        self.assertEqual("auto", run.model_tier)
        self.assertEqual(1200, run.budget.max_input_tokens)
        self.assertEqual(400, run.budget.max_output_tokens)
        self.assertEqual(1.5, run.budget.max_cost_usd)
        self.assertEqual(("git", "shell"), run.required_tools)
        self.assertEqual(("github",), run.required_mcp_servers)
        self.assertEqual(("browser.open",), run.required_mcp_tools)

    def test_agent_run_rejects_invalid_mode(self):
        with self.assertRaisesRegex(RouteValidationError, "invalid mode 'ship'"):
            AgentRun(prompt="test", mode="ship")

    def test_agent_run_rejects_empty_prompt(self):
        with self.assertRaisesRegex(RouteValidationError, "prompt is required"):
            AgentRun(prompt="   ")

    def test_agent_run_rejects_invalid_budget_fields(self):
        with self.assertRaisesRegex(
            RouteValidationError,
            "budget.max_input_tokens must be greater than zero",
        ):
            AgentRun(prompt="test", budget={"max_input_tokens": 0})

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
        ).explain(AgentRun(prompt="write docs", mode="docs", model_tier="auto"))

        self.assertEqual("economy", decision.selected_model_tier)
        self.assertEqual("economy", decision.mode_default_model_tier)

    def test_route_explanation_reports_eligibility_and_rejections(self):
        decision = Router(
            settings(),
            (
                ProviderStatus("codex", "Codex", "cli", True, "available"),
                ProviderStatus("claude", "Claude", "cli", False, "disabled_missing_auth"),
            ),
        ).explain(
            AgentRun(
                prompt="review this change",
                mode="review",
                budget={"max_cost_usd": 2.0},
                required_tools=["git"],
                required_mcp_servers=["github"],
                required_mcp_tools=["browser.open"],
            )
        )

        self.assertEqual("codex", decision.selected_provider)
        self.assertEqual(("codex",), decision.eligible_providers)
        self.assertEqual("disabled_missing_auth", decision.rejected_providers["claude"])
        self.assertEqual("standard", decision.selected_model_tier)
        self.assertIn("Selected provider 'codex' with model tier 'standard'.", decision.explanation)
        self.assertIn("claude (disabled_missing_auth)", decision.explanation)
        self.assertEqual(["git"], decision.as_dict()["run"]["required_tools"])

    def test_requested_provider_filter_is_preserved(self):
        decision = Router(
            settings(),
            (
                ProviderStatus("codex", "Codex", "cli", True, "available"),
                ProviderStatus("claude", "Claude", "cli", True, "available"),
            ),
        ).explain(AgentRun(prompt="test", provider="claude"))

        self.assertEqual("claude", decision.selected_provider)
        self.assertEqual("not_requested_provider", decision.rejected_providers["codex"])

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
