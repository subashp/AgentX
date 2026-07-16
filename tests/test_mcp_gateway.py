import json
import shutil
import unittest
from pathlib import Path

from agentx.config import AgentXPaths
from agentx.mcp_gateway import (
    MCPServicePolicy,
    generate_per_run_mcp_config,
    mcp_tool_audit_event,
    redact_mapping,
)


class MCPGatewayFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self.fixture_root = Path("tests") / ".tmp_mcp_gateway"
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


class MCPGatewayConfigTests(MCPGatewayFixtureTestCase):
    def test_allowlisted_service_config_generation_excludes_denied_tools_and_secrets(self):
        paths = self.make_paths(auth=self.fixture_root / "custom-auth")
        services = (
            MCPServicePolicy(
                service_id="github",
                command="mcp-github",
                args=("--api-key", "ghp_secret_config_value", "--mode=readonly"),
                endpoint="https://token:secret@example.test/mcp?token=query-secret",
                allowed_tools=("issues.read", "repos.get"),
                denied_tools=("secrets.read",),
                auth_service_id="github-token",
                provider_visibility=("codex",),
            ),
        )

        result = generate_per_run_mcp_config(
            services=services,
            paths=paths,
            provider_id="codex",
            run_id="run-001",
            required_services=("github",),
            required_tools=("github.issues.read", "github.secrets.read"),
        )

        config_services = result.config["mcp_services"]
        self.assertEqual(["github"], list(config_services))
        self.assertEqual(["issues.read"], config_services["github"]["tools"])
        self.assertEqual(
            {
                "service_id": "github-token",
                "path_ref": "agentx-auth://github-token",
            },
            config_services["github"]["auth"],
        )
        self.assertEqual(
            self.fixture_root / "custom-auth" / "github-token",
            result.auth_reference_for("github").path,
        )

        encoded = json.dumps(result.as_dict(), sort_keys=True)
        self.assertNotIn("ghp_secret_config_value", encoded)
        self.assertNotIn("query-secret", encoded)
        self.assertNotIn(str(self.fixture_root / "custom-auth"), encoded)
        self.assertNotIn("secrets.read", json.dumps(result.config, sort_keys=True))

    def test_denied_service_and_tool_requests_are_excluded_from_config(self):
        paths = self.make_paths()
        services = (
            MCPServicePolicy(
                service_id="github",
                command="mcp-github",
                allowed_tools=("issues.read",),
                denied_tools=("secrets.read",),
            ),
            MCPServicePolicy(
                service_id="internal",
                command="mcp-internal",
                allowed_tools=("lookup",),
                provider_visibility=("private-local",),
            ),
        )

        result = generate_per_run_mcp_config(
            services=services,
            paths=paths,
            provider_id="codex",
            run_id="run-002",
            required_tools=(
                "github.issues.read",
                "github.secrets.read",
                "internal.lookup",
            ),
        )

        self.assertEqual(["github"], list(result.config["mcp_services"]))
        decisions = {
            (decision.service_id, decision.tool_name): decision
            for decision in result.decisions
        }
        self.assertTrue(decisions[("github", "issues.read")].allowed)
        self.assertEqual("tool_denied", decisions[("github", "secrets.read")].reason)
        self.assertEqual(
            "service_not_visible_to_provider",
            decisions[("internal", "lookup")].reason,
        )

    def test_provider_visibility_filtering_includes_matching_services_only(self):
        paths = self.make_paths()
        services = (
            MCPServicePolicy(
                service_id="public",
                command="mcp-public",
                allowed_tools=("search",),
            ),
            MCPServicePolicy(
                service_id="private",
                command="mcp-private",
                allowed_tools=("inspect",),
                provider_visibility=("private-local",),
            ),
        )

        codex_result = generate_per_run_mcp_config(
            services=services,
            paths=paths,
            provider_id="codex",
            run_id="run-003",
        )
        private_result = generate_per_run_mcp_config(
            services=services,
            paths=paths,
            provider_id="private-local",
            run_id="run-004",
        )

        self.assertEqual(["public"], list(codex_result.config["mcp_services"]))
        self.assertEqual(
            ["private", "public"],
            sorted(private_result.config["mcp_services"]),
        )


class MCPGatewayRedactionTests(unittest.TestCase):
    def test_argument_and_result_redaction_keys_and_paths_hide_values(self):
        payload = {
            "token": "ARGTOKEN-123",
            "nested": {
                "password": "PASSWORD-123",
                "items": [{"api_key": "APIKEY-123", "safe": "visible"}],
            },
            "result": {"secret": "RESULTSECRET-123", "safe": "ok"},
        }

        redacted = redact_mapping(
            payload,
            redact_keys=("token", "password", "api_key"),
            redact_paths=("result.secret",),
        )

        self.assertEqual("[REDACTED]", redacted.payload["token"])
        self.assertEqual("[REDACTED]", redacted.payload["nested"]["password"])
        self.assertEqual(
            "[REDACTED]",
            redacted.payload["nested"]["items"][0]["api_key"],
        )
        self.assertEqual("[REDACTED]", redacted.payload["result"]["secret"])
        self.assertEqual("visible", redacted.payload["nested"]["items"][0]["safe"])
        self.assertEqual(
            {
                "nested.items.0.api_key",
                "nested.password",
                "result.secret",
                "token",
            },
            {entry.path for entry in redacted.redactions},
        )

        encoded = json.dumps(redacted.as_dict(), sort_keys=True)
        self.assertNotIn("ARGTOKEN-123", encoded)
        self.assertNotIn("PASSWORD-123", encoded)
        self.assertNotIn("APIKEY-123", encoded)
        self.assertNotIn("RESULTSECRET-123", encoded)

    def test_audit_event_records_decision_and_redaction_metadata_without_values(self):
        payload = {"token": "ARGTOKEN-456", "result": {"secret": "RESULTSECRET-456"}}
        argument_redactions = redact_mapping(payload, redact_keys=("token",)).redactions
        result_redactions = redact_mapping(
            payload,
            redact_paths=("result.secret",),
        ).redactions
        decision = generate_per_run_mcp_config(
            services=(
                MCPServicePolicy(
                    service_id="github",
                    command="mcp-github",
                    allowed_tools=("issues.read",),
                    auth_service_id="github-token",
                ),
            ),
            paths=AgentXPaths(
                root=Path("state"),
                settings=Path("state") / "settings.json",
                sessions=Path("state") / "sessions",
                memories=Path("state") / "memories",
                auth=Path("state") / "auth",
            ),
            provider_id="codex",
            run_id="run-005",
            required_tools=("github.issues.read",),
        ).decisions[0]

        event = mcp_tool_audit_event(
            decision,
            sequence=1,
            argument_redactions=argument_redactions,
            result_redactions=result_redactions,
        )

        self.assertEqual("mcp_tool_policy_decision", event["event"])
        self.assertTrue(event["allowed"])
        self.assertEqual("github-token", event["auth_service_id"])
        self.assertEqual(["token"], [entry["path"] for entry in event["argument_redactions"]])
        self.assertEqual(
            ["result.secret"],
            [entry["path"] for entry in event["result_redactions"]],
        )

        encoded = json.dumps(event, sort_keys=True)
        self.assertNotIn("ARGTOKEN-456", encoded)
        self.assertNotIn("RESULTSECRET-456", encoded)


if __name__ == "__main__":
    unittest.main()
