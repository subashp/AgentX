from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .context import MemoryRecord


AGENTMEMORY_SQLITE_FILENAME = "agentmemory.sqlite3"
PRIVACY_TO_CLASSIFICATION: dict[str, str] = {
    "generic": "public",
    "team": "internal",
    "private": "secret",
}


class AgentMemoryBridgeError(ValueError):
    """Raised when AgentMemory data cannot be converted for AgentX."""


def default_agentmemory_db_path(memory_root: str | Path) -> Path:
    return Path(memory_root) / AGENTMEMORY_SQLITE_FILENAME


def load_agentmemory_records(
    memory_root: str | Path,
    *,
    subject_id: str | None = None,
    workspace_id: str | None = None,
    limit: int = 50,
) -> tuple[MemoryRecord, ...]:
    """Load AgentMemory records for AgentX context compilation.

    This returns an empty tuple when AgentMemory is not installed or no database
    exists. It does not create the memory directory or database as a side effect.
    """

    db_path = default_agentmemory_db_path(memory_root)
    if not db_path.exists():
        return ()
    try:
        from agentmemory import SQLiteMemoryStore
    except ImportError:
        return ()
    store = SQLiteMemoryStore(db_path)
    records = store.list_records(subject_id=subject_id, workspace_id=workspace_id)
    return tuple(agentmemory_record_to_agentx(record) for record in records[:limit])


def agentmemory_record_to_agentx(record: Any) -> MemoryRecord:
    privacy_class = str(getattr(record, "privacy_class", "private"))
    classification = PRIVACY_TO_CLASSIFICATION.get(privacy_class)
    if classification is None:
        raise AgentMemoryBridgeError(f"unsupported AgentMemory privacy class '{privacy_class}'")
    memory_id = str(getattr(record, "memory_id", "")).strip()
    content = _optional_text(getattr(record, "content", None))
    summary = _optional_text(getattr(record, "summary", None))
    return MemoryRecord(
        id=memory_id,
        classification=classification,
        content=content,
        summary=summary,
    )


def merge_agentmemory_records(
    explicit_memories: Sequence[MemoryRecord],
    memory_root: str | Path,
) -> tuple[MemoryRecord, ...]:
    if explicit_memories:
        return tuple(explicit_memories)
    return load_agentmemory_records(memory_root)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
