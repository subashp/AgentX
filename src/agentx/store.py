from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .config import AgentXPaths, Settings, SettingsLoader
from .context import MemoryRecord


RUN_ARTIFACT_FILES: tuple[str, ...] = (
    "manifest.json",
    "prompt.md",
    "context-map.json",
    "memory-map.json",
    "redactions.json",
    "provider.json",
    "transcript.jsonl",
    "patch.diff",
    "cost.json",
    "outcome.json",
)
RUN_JSON_ARTIFACTS: frozenset[str] = frozenset(
    {
        "manifest.json",
        "context-map.json",
        "memory-map.json",
        "redactions.json",
        "provider.json",
        "cost.json",
        "outcome.json",
    }
)
RUN_TEXT_ARTIFACTS: frozenset[str] = frozenset({"prompt.md", "patch.diff"})

_VALID_IDENTIFIER_CHARS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


class StoreError(ValueError):
    """Raised when local storage inputs or on-disk data are invalid."""


class SettingsStore:
    """Persist settings documents to the configured settings path."""

    def __init__(self, paths: AgentXPaths) -> None:
        self._paths = paths

    @property
    def path(self) -> Path:
        return self._paths.settings

    def load(self) -> Settings:
        return SettingsLoader(self._paths).load()

    def write(self, settings: Settings) -> Path:
        if not isinstance(settings, Settings):
            raise StoreError("settings must be a Settings object.")

        suffix = self.path.suffix.lower()
        if suffix not in {"", ".json"}:
            raise StoreError(
                "settings writing currently supports only JSON settings paths."
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(self.path, _dump_json(settings.as_dict()))
        return self.path


class SessionStore:
    """Resolve session directories and open run artifact stores."""

    def __init__(self, paths: AgentXPaths) -> None:
        self._paths = paths

    @property
    def root(self) -> Path:
        return self._paths.sessions

    def path_for_session(self, session_id: str) -> Path:
        return self.root / _normalize_identifier(session_id, "session_id")

    def open_run(self, session_id: str) -> "RunStore":
        return RunStore(self.path_for_session(session_id))


class RunStore:
    """Write deterministic local artifacts for a single run/session."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def ensure_exists(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def artifact_path(self, name: str) -> Path:
        normalized = _normalize_artifact_name(name)
        return self.root / normalized

    def write_json_artifact(self, name: str, payload: object) -> Path:
        normalized = _normalize_artifact_name(name)
        if normalized not in RUN_JSON_ARTIFACTS:
            raise StoreError(f"artifact '{normalized}' must be written as JSON.")
        self.ensure_exists()
        path = self.artifact_path(normalized)
        _write_text(path, _dump_json(payload))
        return path

    def write_text_artifact(self, name: str, text: str) -> Path:
        normalized = _normalize_artifact_name(name)
        if normalized not in RUN_TEXT_ARTIFACTS:
            raise StoreError(f"artifact '{normalized}' must be written as text.")
        if not isinstance(text, str):
            raise StoreError(f"artifact '{normalized}' text must be a string.")
        self.ensure_exists()
        path = self.artifact_path(normalized)
        _write_text(path, text)
        return path

    def write_transcript(self, events: Sequence[object]) -> Path:
        if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
            raise StoreError("transcript events must be a sequence of JSON objects.")

        self.ensure_exists()
        path = self.artifact_path("transcript.jsonl")
        lines = [
            json.dumps(_to_json_data(event), indent=None, sort_keys=True)
            for event in events
        ]
        text = ""
        if lines:
            text = "\n".join(lines) + "\n"
        _write_text(path, text)
        return path

    def append_transcript_event(self, event: object) -> Path:
        self.ensure_exists()
        path = self.artifact_path("transcript.jsonl")
        line = json.dumps(_to_json_data(event), indent=None, sort_keys=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{line}\n")
        return path

    def write_artifacts(
        self,
        *,
        manifest: object,
        prompt: str,
        context_map: object,
        memory_map: object,
        redactions: object,
        provider: object,
        transcript: Sequence[object] = (),
        patch: str = "",
        cost: object = None,
        outcome: object = None,
    ) -> dict[str, Path]:
        return {
            "manifest.json": self.write_json_artifact("manifest.json", manifest),
            "prompt.md": self.write_text_artifact("prompt.md", prompt),
            "context-map.json": self.write_json_artifact(
                "context-map.json", context_map
            ),
            "memory-map.json": self.write_json_artifact("memory-map.json", memory_map),
            "redactions.json": self.write_json_artifact("redactions.json", redactions),
            "provider.json": self.write_json_artifact("provider.json", provider),
            "transcript.jsonl": self.write_transcript(transcript),
            "patch.diff": self.write_text_artifact("patch.diff", patch),
            "cost.json": self.write_json_artifact("cost.json", cost),
            "outcome.json": self.write_json_artifact("outcome.json", outcome),
        }


class MemoryStore:
    """CRUD helpers for local JSON memory records."""

    def __init__(self, paths: AgentXPaths) -> None:
        self._paths = paths

    @property
    def root(self) -> Path:
        return self._paths.memories

    def list(self) -> tuple[MemoryRecord, ...]:
        if not self.root.exists():
            return ()

        records: list[MemoryRecord] = []
        for path in sorted(self.root.glob("*.json")):
            records.append(self._read_path(path))
        return tuple(records)

    def read(self, memory_id: str) -> MemoryRecord:
        path = self.path_for(memory_id)
        if not path.exists():
            raise StoreError(f"memory '{memory_id}' does not exist.")
        return self._read_path(path)

    def write(self, memory: MemoryRecord) -> Path:
        if not isinstance(memory, MemoryRecord):
            raise StoreError("memory must be a MemoryRecord.")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(memory.id)
        _write_text(path, _dump_json(memory.as_dict()))
        return path

    def delete(self, memory_id: str) -> bool:
        path = self.path_for(memory_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def path_for(self, memory_id: str) -> Path:
        normalized = _normalize_identifier(memory_id, "memory_id")
        return self.root / f"{normalized}.json"

    def _read_path(self, path: Path) -> MemoryRecord:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StoreError(f"memory file '{path.name}' is not valid JSON.") from exc

        if not isinstance(raw, dict):
            raise StoreError(f"memory file '{path.name}' must contain a JSON object.")

        try:
            record = MemoryRecord(
                id=raw.get("id"),
                classification=raw.get("classification"),
                content=raw.get("content"),
                summary=raw.get("summary"),
            )
        except ValueError as exc:
            raise StoreError(f"memory file '{path.name}' is invalid: {exc}") from exc

        if record.id != path.stem:
            raise StoreError(
                f"memory file '{path.name}' id does not match filename '{path.stem}'."
            )
        return record


def resolve_auth_service_path(paths: AgentXPaths, service: str) -> Path:
    return paths.auth / _normalize_identifier(service, "service")


def _normalize_artifact_name(value: object) -> str:
    if not isinstance(value, str):
        raise StoreError("artifact name must be a string.")
    normalized = value.strip()
    if normalized not in RUN_ARTIFACT_FILES:
        raise StoreError(
            f"artifact name must be one of: {', '.join(RUN_ARTIFACT_FILES)}."
        )
    return normalized


def _normalize_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise StoreError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise StoreError(f"{field_name} must be a non-empty string.")
    if normalized in {".", ".."}:
        raise StoreError(f"{field_name} must not be '.' or '..'.")
    if any(character not in _VALID_IDENTIFIER_CHARS for character in normalized):
        raise StoreError(
            f"{field_name} must use only letters, numbers, '.', '-', or '_'."
        )
    return normalized


def _dump_json(payload: object) -> str:
    return json.dumps(_to_json_data(payload), indent=2, sort_keys=True) + "\n"


def _to_json_data(value: object) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _to_json_data(value.as_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _to_json_data(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _to_json_data(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_data(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise StoreError(f"value of type {type(value).__name__} is not JSON serializable.")


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


__all__ = [
    "MemoryStore",
    "RUN_ARTIFACT_FILES",
    "RunStore",
    "SessionStore",
    "SettingsStore",
    "StoreError",
    "resolve_auth_service_path",
]
