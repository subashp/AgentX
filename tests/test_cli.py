import io
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from agentx.adapters import AdapterResult
from agentx import cli
from agentx.config import AgentXPaths, ProviderSettings, Settings
from agentx.providers import ProviderStatus


class CliTests(unittest.TestCase):
    def test_init_default_agentx_profile_writes_fake_local_settings(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_init_agentx"
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
                code = cli.run(["--json", "init"], stdout, stderr)

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual("agentx", payload["profile"])
            written = json.loads((root / "settings.json").read_text(encoding="utf-8"))
            self.assertEqual(["fake-local"], written["public_providers"])
            self.assertEqual({}, written["providers"])
            self.assertNotIn("paths", written)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_init_refuses_to_overwrite_without_force(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_init_exists"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        (root / "settings.json").write_text('{"public_providers":["codex"]}\n', encoding="utf-8")

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
                code = cli.run(["init"], stdout, stderr)

            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertIn("settings already exist", stderr.getvalue())
            self.assertEqual(
                {"public_providers": ["codex"]},
                json.loads((root / "settings.json").read_text(encoding="utf-8")),
            )
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_init_codex_profile_writes_configured_command(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_init_codex"
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
                code = cli.run(
                    [
                        "init",
                        "--profile",
                        "codex",
                        "--codex-command",
                        "codex-under-test",
                    ],
                    stdout,
                    stderr,
                )

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            written = json.loads((root / "settings.json").read_text(encoding="utf-8"))
            self.assertEqual(["codex"], written["public_providers"])
            self.assertEqual("codex-under-test", written["providers"]["codex"]["command"])
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_init_private_profile_writes_endpoint_model_and_secret_reference(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_init_private"
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
                code = cli.run(
                    [
                        "init",
                        "--profile",
                        "private-openai-compatible",
                        "--endpoint",
                        "http://127.0.0.1:8000/v1",
                        "--model",
                        "Qwen/Qwen3-14B",
                        "--api-key-env",
                        "AGENTX_QWEN_API_KEY",
                        "--timeout",
                        "180",
                    ],
                    stdout,
                    stderr,
                )

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            written = json.loads((root / "settings.json").read_text(encoding="utf-8"))
            self.assertEqual("private-openai-compatible", written["private_provider"])
            provider = written["providers"]["private-openai-compatible"]
            self.assertEqual("http://127.0.0.1:8000/v1", provider["endpoint"])
            self.assertEqual("Qwen/Qwen3-14B", provider["model"])
            self.assertEqual("AGENTX_QWEN_API_KEY", provider["api_key_env"])
            self.assertEqual(180.0, provider["timeout"])
            initialized = Settings(
                paths=settings.paths,
                private_provider="private-openai-compatible",
                providers={
                    "private-openai-compatible": ProviderSettings(
                        endpoint="http://127.0.0.1:8000/v1",
                        model="Qwen/Qwen3-14B",
                    )
                },
            )
            self.assertEqual(
                "private-openai-compatible",
                cli._resolve_plan_provider(
                    fake=False,
                    requested_provider="auto",
                    settings=initialized,
                ),
            )
        finally:
            if root.exists():
                shutil.rmtree(root)

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

    def test_no_arguments_enters_interactive_mode_and_can_quit(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        settings = Settings(
            paths=AgentXPaths(
                root=Path("tests") / ".tmp_cli_interactive",
                settings=Path("tests") / ".tmp_cli_interactive" / "settings.json",
                sessions=Path("tests") / ".tmp_cli_interactive" / "sessions",
                memories=Path("tests") / ".tmp_cli_interactive" / "memories",
                auth=Path("tests") / ".tmp_cli_interactive" / "auth",
            )
        )
        statuses = (
            ProviderStatus(
                id="fake-local",
                display_name="AgentX Fake Local",
                kind="builtin",
                enabled=True,
                reason="available",
            ),
        )

        with mock.patch("agentx.cli.load_settings", return_value=settings):
            with mock.patch("agentx.cli.ProviderRegistry") as registry:
                registry.return_value.list_statuses.return_value = statuses
                code = cli.run([], stdout, stderr, io.StringIO("1\n/quit\n"))

        self.assertEqual(0, code)
        self.assertEqual("", stderr.getvalue())
        self.assertIn("AgentX interactive mode", stdout.getvalue())
        self.assertIn("fake-local", stdout.getvalue())
        self.assertIn("agentx[auto]>", stdout.getvalue())

    def test_interactive_provider_option_runs_selected_codex_prompt(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        settings = Settings(
            paths=AgentXPaths(
                root=Path("tests") / ".tmp_cli_interactive_codex",
                settings=Path("tests") / ".tmp_cli_interactive_codex" / "settings.json",
                sessions=Path("tests") / ".tmp_cli_interactive_codex" / "sessions",
                memories=Path("tests") / ".tmp_cli_interactive_codex" / "memories",
                auth=Path("tests") / ".tmp_cli_interactive_codex" / "auth",
            )
        )
        statuses = (
            ProviderStatus(
                id="codex",
                display_name="Codex CLI",
                kind="cli",
                enabled=True,
                reason="available",
            ),
        )

        with mock.patch("agentx.cli.load_settings", return_value=settings):
            with mock.patch("agentx.cli.ProviderRegistry") as registry:
                registry.return_value.list_statuses.return_value = statuses
                with mock.patch("agentx.cli._plan") as plan:
                    code = cli.run(
                        ["interactive", "--provider", "codex"],
                        stdout,
                        stderr,
                        io.StringIO("plan this change\n/quit\n"),
                    )

        self.assertEqual(0, code)
        self.assertEqual("", stderr.getvalue())
        self.assertEqual("plan this change", plan.call_args.args[0])
        self.assertEqual("codex", plan.call_args.args[1])
        self.assertIn("agentx[codex]>", stdout.getvalue())

    def test_interactive_keeps_provider_until_provider_command_changes_it(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        settings = Settings(
            paths=AgentXPaths(
                root=Path("tests") / ".tmp_cli_interactive_switch",
                settings=Path("tests") / ".tmp_cli_interactive_switch" / "settings.json",
                sessions=Path("tests") / ".tmp_cli_interactive_switch" / "sessions",
                memories=Path("tests") / ".tmp_cli_interactive_switch" / "memories",
                auth=Path("tests") / ".tmp_cli_interactive_switch" / "auth",
            )
        )
        statuses = (
            ProviderStatus("codex", "Codex CLI", "cli", True, "available", command="codex"),
            ProviderStatus("claude", "Claude Code", "cli", True, "available", command="claude"),
            ProviderStatus("kiro", "Kiro CLI", "cli", True, "available", command="kiro-cli"),
            ProviderStatus(
                "private-openai-compatible",
                "Private OpenAI-Compatible Endpoint",
                "openai_compatible",
                False,
                "endpoint_not_configured",
            ),
        )
        with mock.patch("agentx.cli.load_settings", return_value=settings):
            with mock.patch("agentx.cli.ProviderRegistry") as registry:
                registry.return_value.list_statuses.return_value = statuses
                with mock.patch("agentx.cli._plan") as plan:
                    code = cli.run(
                        [],
                        stdout,
                        stderr,
                        io.StringIO("1\nfirst task\n/provider claude\nsecond task\n/quit\n"),
                    )

        self.assertEqual(0, code)
        self.assertEqual("", stderr.getvalue())
        self.assertEqual(["codex", "claude"], [call.args[1] for call in plan.call_args_list])
        self.assertIn("Warning: custom model provider is unavailable", stdout.getvalue())
        self.assertIn("External settings file:", stdout.getvalue())
        self.assertIn("agentx[auto]>", stdout.getvalue())
        self.assertIn("agentx[claude]>", stdout.getvalue())

    def test_interactive_rejects_switch_to_unavailable_provider(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        settings = Settings(
            paths=AgentXPaths(
                root=Path("tests") / ".tmp_cli_interactive_reject",
                settings=Path("tests") / ".tmp_cli_interactive_reject" / "settings.json",
                sessions=Path("tests") / ".tmp_cli_interactive_reject" / "sessions",
                memories=Path("tests") / ".tmp_cli_interactive_reject" / "memories",
                auth=Path("tests") / ".tmp_cli_interactive_reject" / "auth",
            )
        )
        statuses = (
            ProviderStatus("codex", "Codex CLI", "cli", True, "available", command="codex"),
            ProviderStatus("claude", "Claude Code", "cli", False, "disabled_missing_binary", command="claude"),
        )
        with mock.patch("agentx.cli.load_settings", return_value=settings):
            with mock.patch("agentx.cli.ProviderRegistry") as registry:
                registry.return_value.list_statuses.return_value = statuses
                code = cli.run(
                    ["interactive", "--provider", "codex"],
                    stdout,
                    stderr,
                    io.StringIO("/provider claude\n/quit\n"),
                )

        self.assertEqual(0, code)
        self.assertIn("provider 'claude' is unavailable", stderr.getvalue())
        self.assertIn("agentx[codex]>", stdout.getvalue())

    def test_plan_formatter_surfaces_provider_stdout_and_stderr(self):
        rendered = cli._format_plan(
            {
                "root": "state/session",
                "route": {"explanation": "Selected provider 'codex'."},
                "result": {
                    "status": "success",
                    "transcript_events": [
                        {
                            "event": "process_output_captured",
                            "stdout": "Codex response",
                            "stderr": "Codex warning",
                        }
                    ],
                    "outcome": {},
                },
            }
        )

        self.assertIn("Codex response", rendered)
        self.assertIn("provider stderr:", rendered)
        self.assertIn("Codex warning", rendered)

    def test_plan_formatter_surfaces_private_response_and_dimmed_thinking(self):
        rendered = cli._format_plan(
            {
                "root": "state/session",
                "route": {"explanation": "Selected provider 'private-openai-compatible'."},
                "result": {
                    "provider_id": "private-openai-compatible",
                    "status": "success",
                    "outcome": {
                        "response": "Hello from Qwen.",
                        "thinking": "The user offered a greeting.",
                    },
                },
            },
            color=True,
        )

        self.assertIn("Assistant:\nHello from Qwen.", rendered)
        self.assertIn(
            "\x1b[90mThinking:\nThe user offered a greeting.\x1b[0m",
            rendered,
        )

    def test_plan_formatter_surfaces_provider_failure(self):
        rendered = cli._format_plan(
            {
                "root": "state/session",
                "route": {"explanation": "Selected provider 'codex'."},
                "result": {
                    "status": "failure",
                    "transcript_events": [
                        {
                            "event": "process_output_captured",
                            "stdout": "",
                            "stderr": "unsupported option",
                        }
                    ],
                    "outcome": {
                        "summary": "Codex CLI exited with code 2.",
                    },
                },
            }
        )

        self.assertIn("provider error:", rendered)
        self.assertIn("unsupported option", rendered)
        self.assertIn("provider status: Codex CLI exited with code 2.", rendered)

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

    def test_plan_outputs_json_and_writes_artifacts_without_provider_registry(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_plan"
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
                ),
                public_providers=("fake-local",),
            )
            with mock.patch("agentx.cli.load_settings", return_value=settings):
                with mock.patch("agentx.cli.ProviderRegistry") as registry:
                    registry.side_effect = AssertionError("provider registry was used")
                    code = cli.run(
                        [
                            "--json",
                            "plan",
                            "--fake",
                            "--session-id",
                            "cli-plan",
                            "--context",
                            "tests/test_cli.py",
                            "local plan",
                        ],
                        stdout,
                        stderr,
                    )

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual("cli-plan", payload["session_id"])
            self.assertEqual("fake-local", payload["route"]["selected_provider"])
            self.assertEqual("success", payload["result"]["status"])
            self.assertEqual(["tests/test_cli.py"], payload["context_manifest"]["included_paths"])
            self.assertTrue((root / "sessions" / "cli-plan" / "manifest.json").exists())
            self.assertEqual(
                "",
                (root / "sessions" / "cli-plan" / "patch.diff").read_text(encoding="utf-8"),
            )
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_plan_reports_unavailable_codex_without_running_provider(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_codex_unavailable"
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
            statuses = (
                ProviderStatus(
                    id="codex",
                    display_name="Codex CLI",
                    kind="cli",
                    enabled=False,
                    reason="disabled_missing_binary",
                ),
            )
            with mock.patch("agentx.cli.load_settings", return_value=settings):
                with mock.patch("agentx.cli.ProviderRegistry") as registry:
                    registry.return_value.list_statuses.return_value = statuses
                    code = cli.run(["plan", "local plan"], stdout, stderr)

            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertIn("codex provider is not available", stderr.getvalue())
            self.assertFalse((root / "sessions").exists())
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_plan_rejects_unknown_live_provider(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = cli.run(["plan", "--provider", "unknown", "local plan"], stdout, stderr)

        self.assertEqual(2, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("provider 'unknown' is not supported", stderr.getvalue())

    def test_provider_fake_local_matches_fake_shorthand_without_registry(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_plan_alias"
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
                ),
                public_providers=("fake-local",),
            )
            with mock.patch("agentx.cli.load_settings", return_value=settings):
                with mock.patch("agentx.cli.ProviderRegistry") as registry:
                    registry.side_effect = AssertionError("provider registry was used")
                    code = cli.run(
                        [
                            "--json",
                            "plan",
                            "--provider",
                            "fake-local",
                            "--session-id",
                            "cli-plan-alias",
                            "local plan",
                        ],
                        stdout,
                        stderr,
                    )

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual("fake-local", payload["run"]["provider"])
            self.assertEqual("fake-local", payload["result"]["provider_id"])
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_auto_plan_uses_fake_local_for_agentx_only_profile(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_plan_agentx_only"
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
                ),
                public_providers=("fake-local",),
            )
            with mock.patch("agentx.cli.load_settings", return_value=settings):
                with mock.patch("agentx.cli.ProviderRegistry") as registry:
                    registry.side_effect = AssertionError("provider registry was used")
                    code = cli.run(
                        [
                            "--json",
                            "plan",
                            "--session-id",
                            "agentx-only-plan",
                            "local plan",
                        ],
                        stdout,
                        stderr,
                    )

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual("fake-local", payload["run"]["provider"])
            self.assertEqual("fake-local", payload["result"]["provider_id"])
            self.assertTrue((root / "sessions" / "agentx-only-plan" / "manifest.json").exists())
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_codex_plan_uses_registry_scoped_workspace_and_configured_command(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_codex"
        if root.exists():
            shutil.rmtree(root)
        source = root / "source"
        source.mkdir(parents=True)
        (source / "README.md").write_text("# Visible\n", encoding="utf-8")
        state = root / "state"

        settings = Settings(
            paths=AgentXPaths(
                root=state,
                settings=state / "settings.json",
                sessions=state / "sessions",
                memories=state / "memories",
                auth=state / "auth",
            ),
            public_providers=("codex",),
            providers={"codex": ProviderSettings(command="codex-under-test")},
        )
        statuses = (
            ProviderStatus(
                id="codex",
                display_name="Codex CLI",
                kind="cli",
                enabled=True,
                reason="available",
                command="codex-under-test",
                resolved_command="codex-under-test",
            ),
        )

        RecordingCodexAdapter.instances.clear()
        try:
            with mock.patch("agentx.cli.load_settings", return_value=settings):
                with mock.patch("agentx.cli.ProviderRegistry") as registry:
                    registry.return_value.list_statuses.return_value = statuses
                    with mock.patch("agentx.cli.CodexCliAdapter", RecordingCodexAdapter):
                        code = cli.run(
                            [
                                "--json",
                                "plan",
                                "--provider",
                                "codex",
                                "--session-id",
                                "cli-codex",
                                "--source-root",
                                str(source),
                                "--workspace-id",
                                "demo-workspace",
                                "--context",
                                "README.md",
                                "live plan",
                            ],
                            stdout,
                            stderr,
                        )

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            registry.return_value.list_statuses.assert_called_once()
            payload = json.loads(stdout.getvalue())
            workspace = state / "sessions" / "cli-codex" / "workspace"
            self.assertEqual("codex", payload["run"]["provider"])
            self.assertEqual("codex", payload["result"]["provider_id"])
            self.assertEqual("success", payload["result"]["status"])
            self.assertEqual("# Visible\n", (workspace / "README.md").read_text(encoding="utf-8"))
            adapter = RecordingCodexAdapter.instances[0]
            self.assertEqual("codex-under-test", adapter.command)
            absolute_workspace = workspace.resolve()
            self.assertEqual(absolute_workspace, adapter.cwd)
            self.assertEqual(("-C", str(absolute_workspace)), adapter.extra_args)
            scoped = payload["result"]["outcome"]["scoped_workspace"]
            self.assertTrue(scoped["ok"])
            self.assertEqual("demo-workspace", scoped["workspace_id"])
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_private_plan_constructs_configured_openai_compatible_adapter(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_private"
        if root.exists():
            shutil.rmtree(root)
        state = root / "state"
        settings = Settings(
            paths=AgentXPaths(
                root=state,
                settings=state / "settings.json",
                sessions=state / "sessions",
                memories=state / "memories",
                auth=state / "auth",
            ),
            private_provider="private-openai-compatible",
            providers={
                "private-openai-compatible": ProviderSettings(
                    endpoint="https://example.ngrok-free.app/v1",
                    model="Qwen/Qwen3-14B",
                    api_key_env="AGENTX_TEST_API_KEY",
                    timeout=180.0,
                )
            },
        )
        statuses = (
            ProviderStatus(
                id="private-openai-compatible",
                display_name="Private OpenAI-Compatible Endpoint",
                kind="openai_compatible",
                enabled=True,
                reason="available",
                endpoint="https://example.ngrok-free.app/v1",
            ),
        )

        RecordingPrivateAdapter.instances.clear()
        try:
            with mock.patch("agentx.cli.load_settings", return_value=settings):
                with mock.patch("agentx.cli.ProviderRegistry") as registry:
                    registry.return_value.list_statuses.return_value = statuses
                    with mock.patch("agentx.cli.OpenAICompatibleAdapter", RecordingPrivateAdapter):
                        with mock.patch.dict("os.environ", {"AGENTX_TEST_API_KEY": "test-key"}):
                            code = cli.run(
                                [
                                    "--json",
                                    "plan",
                                    "--provider",
                                    "private-openai-compatible",
                                    "--session-id",
                                    "private-plan",
                                    "hello from private model",
                                ],
                                stdout,
                                stderr,
                            )

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual("private-openai-compatible", payload["run"]["provider"])
            self.assertEqual("private-openai-compatible", payload["result"]["provider_id"])
            adapter = RecordingPrivateAdapter.instances[0]
            self.assertEqual("https://example.ngrok-free.app/v1", adapter.base_url)
            self.assertEqual("Qwen/Qwen3-14B", adapter.model)
            self.assertEqual("test-key", adapter.api_key)
            self.assertEqual(180.0, adapter.timeout)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_execute_outputs_json_and_writes_validation_artifacts_without_provider_registry(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        root = Path("tests") / ".tmp_cli_execute"
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
                ),
                public_providers=("fake-local",),
            )
            with mock.patch("agentx.cli.load_settings", return_value=settings):
                with mock.patch("agentx.cli.ProviderRegistry") as registry:
                    registry.side_effect = AssertionError("provider registry was used")
                    code = cli.run(
                        [
                            "--json",
                            "execute",
                            "--fake",
                            "--session-id",
                            "cli-execute",
                            "--allowed-patch",
                            "src/app.py",
                            "local execute",
                        ],
                        stdout,
                        stderr,
                    )

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual("cli-execute", payload["session_id"])
            self.assertEqual("execute", payload["run"]["mode"])
            self.assertEqual("success", payload["result"]["status"])
            self.assertTrue(payload["patch_validation"]["accepted"])
            self.assertFalse(payload["patch_applied"])
            self.assertTrue((root / "sessions" / "cli-execute" / "outcome.json").exists())
            outcome = json.loads(
                (root / "sessions" / "cli-execute" / "outcome.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(outcome["patch_validation"]["accepted"])
            self.assertFalse(outcome["patch_application"]["supported"])
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_execute_requires_fake_until_live_execution_is_supported(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = cli.run(["execute", "local execute"], stdout, stderr)

        self.assertEqual(2, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("execute currently supports only --fake", stderr.getvalue())

    def test_run_fake_reports_storage_failure_without_traceback(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("agentx.cli.execute_fake_run", side_effect=OSError("storage unavailable")):
            code = cli.run(["run", "--fake", "local dry run"], stdout, stderr)

        self.assertEqual(2, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("agentx: storage unavailable", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


class RecordingCodexAdapter:
    provider_id = "codex"
    instances = []

    def __init__(self, *, command, cwd, extra_args):
        self.command = command
        self.cwd = cwd
        self.extra_args = tuple(extra_args)
        self.requests = []
        self.instances.append(self)

    def execute(self, request):
        self.requests.append(request)
        return AdapterResult(
            provider_id=self.provider_id,
            model_id="codex-test",
            model_tier="high",
            status="success",
            transcript_events=(
                {"sequence": 1, "event": "execution_completed", "status": "success"},
            ),
            cost={"currency": "USD", "estimated": False, "total_cost_usd": 0.0},
            outcome={"status": "success", "outcome": "codex_test_completed"},
            patch="diff --git a/README.md b/README.md\n",
        )


class RecordingPrivateAdapter:
    provider_id = "private-openai-compatible"
    instances = []

    def __init__(self, *, base_url, model, api_key, timeout, provider_id, context_root):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.provider_id = provider_id
        self.context_root = context_root
        self.instances.append(self)

    def execute(self, request):
        return AdapterResult(
            provider_id=self.provider_id,
            model_id=self.model,
            model_tier="high",
            status="success",
            transcript_events=(
                {"sequence": 1, "event": "execution_completed", "status": "success"},
            ),
            cost={"currency": "USD", "estimated": False, "total_cost_usd": 0.0},
            outcome={"status": "success", "outcome": "private_test_completed", "summary": "private response"},
            patch="",
        )


if __name__ == "__main__":
    unittest.main()
