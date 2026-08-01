import unittest
from pathlib import Path


LAUNCHER = Path("deploy") / "halo" / "start-qwen3-vllm.sh"


class HaloVllmLauncherTests(unittest.TestCase):
    def test_launcher_enables_qwen3_tool_calling_by_default_and_allows_opt_out(self):
        text = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn('ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"', text)
        self.assertIn('TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"', text)
        self.assertIn('REASONING_PARSER="${REASONING_PARSER:-qwen3}"', text)
        self.assertIn("--enable-auto-tool-choice", text)
        self.assertIn('"${TOOL_CALLING_ARGS[@]}"', text)
        self.assertIn('if [[ "$ENABLE_TOOL_CALLING" == "1" ]]', text)


if __name__ == "__main__":
    unittest.main()
