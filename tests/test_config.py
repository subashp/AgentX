import json
import unittest
from pathlib import Path
from unittest import mock

from agentx.config import ConfigError, PathResolver, SettingsLoader


class PathResolverTests(unittest.TestCase):
    def test_agentx_home_overrides_default_root(self):
        paths = PathResolver({"AGENTX_HOME": "/state"}).resolve()

        self.assertEqual(Path("/state"), paths.root)
        self.assertEqual(Path("/state") / "settings.json", paths.settings)
        self.assertEqual(Path("/state") / "sessions", paths.sessions)

    def test_specific_paths_override_agentx_home_children(self):
        paths = PathResolver(
            {
                "AGENTX_HOME": "/state",
                "AGENTX_SETTINGS": "/config/settings.json",
                "AGENTX_MEMORIES": "/memory",
            }
        ).resolve()

        self.assertEqual(Path("/config/settings.json"), paths.settings)
        self.assertEqual(Path("/memory"), paths.memories)
        self.assertEqual(Path("/state") / "auth", paths.auth)


class SettingsLoaderTests(unittest.TestCase):
    def test_missing_settings_uses_defaults(self):
        paths = PathResolver({"AGENTX_HOME": "/state"}).resolve()

        with mock.patch.object(Path, "exists", return_value=False):
            settings = SettingsLoader(paths).load()

        self.assertEqual((), settings.public_providers)
        self.assertEqual("internal", settings.external_max_classification)

    def test_json_settings_are_loaded(self):
        paths = PathResolver({"AGENTX_HOME": "/state"}).resolve()
        raw = json.dumps(
            {
                "public_providers": ["codex", "claude"],
                "private_provider": "private-local",
                "external_max_classification": "public",
                "providers": {
                    "codex": {
                        "command": "/tools/codex",
                        "auth_check": "codex-auth",
                        "subscription_check": "codex-subscription",
                    },
                    "private-openai-compatible": {
                        "endpoint": "http://127.0.0.1:8000/v1",
                        "enabled": False,
                    },
                },
            }
        )

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "read_text", return_value=raw),
        ):
            settings = SettingsLoader(paths).load()

        self.assertEqual(("codex", "claude"), settings.public_providers)
        self.assertEqual("private-local", settings.private_provider)
        self.assertEqual("public", settings.external_max_classification)
        self.assertEqual("/tools/codex", settings.providers["codex"].command)
        self.assertEqual("codex-auth", settings.providers["codex"].auth_check)
        self.assertFalse(settings.providers["private-openai-compatible"].enabled)

    def test_invalid_settings_shape_is_rejected(self):
        paths = PathResolver({"AGENTX_HOME": "/state"}).resolve()

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "read_text", return_value="[]"),
            self.assertRaises(ConfigError),
        ):
            SettingsLoader(paths).load()

    def test_yaml_settings_are_loaded(self):
        paths = PathResolver(
            {
                "AGENTX_HOME": "/state",
                "AGENTX_SETTINGS": "/state/settings.yaml",
            }
        ).resolve()
        raw = """
public_providers:
  - codex
  - claude
private_provider: private-local
external_max_classification: public
"""

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "read_text", return_value=raw),
        ):
            settings = SettingsLoader(paths).load()

        self.assertEqual(("codex", "claude"), settings.public_providers)
        self.assertEqual("private-local", settings.private_provider)
        self.assertEqual("public", settings.external_max_classification)

    def test_invalid_provider_settings_are_rejected(self):
        paths = PathResolver({"AGENTX_HOME": "/state"}).resolve()
        raw = json.dumps({"providers": {"codex": {"enabled": "yes"}}})

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "read_text", return_value=raw),
            self.assertRaises(ConfigError),
        ):
            SettingsLoader(paths).load()

if __name__ == "__main__":
    unittest.main()
