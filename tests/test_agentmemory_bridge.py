import json
import shutil
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

from agentx.agentmemory_bridge import (
    agentmemory_record_to_agentx,
    default_agentmemory_db_path,
    load_agentmemory_records,
)
from agentx.config import AgentXPaths, Settings
from agentx.orchestrator import execute_plan_mode
from agentx.store import SessionStore


@dataclass(frozen=True)
class FakeAgentMemoryRecord:
    memory_id: str
    privacy_class: str
    content: str
    summary: str = ""


class AgentMemoryBridgeTests(unittest.TestCase):
    def setUp(self):
        self.fixture_root = Path("tests") / ".tmp_agentmemory_bridge"
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)
        self.fixture_root.mkdir(parents=True)

    def tearDown(self):
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)

    def test_maps_agentmemory_privacy_to_agentx_classification(self):
        generic = agentmemory_record_to_agentx(
            FakeAgentMemoryRecord("generic-note", "generic", "public preference")
        )
        team = agentmemory_record_to_agentx(
            FakeAgentMemoryRecord("team-note", "team", "team workflow")
        )
        private = agentmemory_record_to_agentx(
            FakeAgentMemoryRecord("private-note", "private", "local-only preference")
        )
        self.assertEqual("public", generic.classification)
        self.assertEqual("internal", team.classification)
        self.assertEqual("secret", private.classification)

    def test_missing_agentmemory_db_returns_empty_without_creating_file(self):
        memory_root = self.fixture_root / "memories"
        self.assertEqual((), load_agentmemory_records(memory_root))
        self.assertFalse(default_agentmemory_db_path(memory_root).exists())

    def test_orchestrator_loads_agentmemory_records_when_submodule_package_is_available(self):
        agentmemory_src = Path("third_party") / "AgentMemory" / "src"
        if not agentmemory_src.exists():
            self.skipTest("AgentMemory submodule is not initialized")
        sys.path.insert(0, str(agentmemory_src.resolve()))
        try:
            from agentmemory import MemoryRecord, SQLiteMemoryStore
        finally:
            sys.path.pop(0)

        settings = self.make_settings()
        db_path = default_agentmemory_db_path(settings.paths.memories)
        store = SQLiteMemoryStore(db_path)
        store.remember(
            MemoryRecord(
                memory_id="agentmemory-note",
                content="Use compact AgentX answers.",
                summary="compact AgentX answers",
                privacy_class="generic",
            )
        )
        store.remember(
            MemoryRecord(
                memory_id="private-agentmemory-note",
                content="Private local-only note.",
                summary="private local-only note",
                privacy_class="private",
            )
        )

        result = execute_plan_mode(
            settings=settings,
            session_store=SessionStore(settings.paths),
            session_id="agentmemory-load",
            prompt="Plan with memory",
        )
        context_map = json.loads((result.stored_run.root / "context-map.json").read_text(encoding="utf-8"))
        memory_map = json.loads((result.stored_run.root / "memory-map.json").read_text(encoding="utf-8"))
        self.assertEqual(["agentmemory-note"], memory_map["included_memories"])
        self.assertIn("agentmemory_prompt", context_map)
        self.assertIn("compact AgentX answers", context_map["agentmemory_prompt"]["rendered_text"])
        self.assertIn("private-agentmemory-note", memory_map["agentmemory_omitted_memory_ids"])

    def make_settings(self):
        root = self.fixture_root / "state"
        return Settings(
            paths=AgentXPaths(
                root=root,
                settings=root / "settings.json",
                sessions=root / "sessions",
                memories=root / "memories",
                auth=root / "auth",
            ),
            public_providers=("fake-local",),
        )


if __name__ == "__main__":
    unittest.main()
