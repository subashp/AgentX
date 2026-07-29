import importlib.util
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock


HELPER_PATH = Path(__file__).parents[1] / "deploy" / "halo" / "halo_helper.py"
SPEC = importlib.util.spec_from_file_location("agentx_halo_helper", HELPER_PATH)
assert SPEC and SPEC.loader
halo_helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(halo_helper)


class HaloHelperTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("tests") / ".tmp_halo_helper"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_merge_settings_preserves_existing_provider_configuration(self):
        path = self.root / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "public_providers": ["codex"],
                    "providers": {"codex": {"command": "codex"}},
                }
            ),
            encoding="utf-8",
        )

        result = halo_helper.merge_settings(
            path,
            "http://127.0.0.1:8000/v1",
            "Qwen/Qwen3-14B",
            900,
        )

        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(result["changed"])
        self.assertEqual(["codex"], document["public_providers"])
        self.assertEqual("codex", document["providers"]["codex"]["command"])
        self.assertEqual(
            "Qwen/Qwen3-14B",
            document["providers"]["private-openai-compatible"]["model"],
        )
        self.assertEqual(900, document["providers"]["private-openai-compatible"]["timeout"])

    def test_merge_settings_adds_provider_defaults_without_existing_settings(self):
        path = self.root / "nested" / "settings.json"

        result = halo_helper.merge_settings(
            path,
            "http://127.0.0.1:8000/v1",
            "Qwen/Qwen3-14B",
            900,
        )

        self.assertTrue(result["created"])
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(["codex", "claude", "kiro"], document["public_providers"])
        self.assertEqual("private-openai-compatible", document["private_provider"])

    def test_probe_reports_model_availability(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"data":[{"id":"Qwen/Qwen3-14B"}]}'

        with mock.patch.object(halo_helper.urllib.request, "urlopen", return_value=response):
            result = halo_helper.probe(
                "http://127.0.0.1:8000/v1",
                5,
                "Qwen/Qwen3-14B",
            )

        self.assertTrue(result["healthy"])
        self.assertTrue(result["model_available"])
        self.assertEqual(["Qwen/Qwen3-14B"], result["models"])

    def test_probe_rejects_missing_model(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"data":[{"id":"other-model"}]}'

        with mock.patch.object(halo_helper.urllib.request, "urlopen", return_value=response):
            result = halo_helper.probe(
                "http://127.0.0.1:8000/v1",
                5,
                "Qwen/Qwen3-14B",
            )

        self.assertTrue(result["healthy"])
        self.assertFalse(result["model_available"])


if __name__ == "__main__":
    unittest.main()
