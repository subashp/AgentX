from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """Raised when settings cannot be loaded or validated."""


@dataclass(frozen=True)
class AgentXPaths:
    root: Path
    settings: Path
    sessions: Path
    memories: Path
    auth: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "settings": str(self.settings),
            "sessions": str(self.sessions),
            "memories": str(self.memories),
            "auth": str(self.auth),
        }


@dataclass(frozen=True)
class Settings:
    paths: AgentXPaths
    public_providers: tuple[str, ...] = ()
    private_provider: str | None = None
    external_max_classification: str = "internal"

    def as_dict(self) -> dict[str, object]:
        return {
            "paths": self.paths.as_dict(),
            "public_providers": list(self.public_providers),
            "private_provider": self.private_provider,
            "external_max_classification": self.external_max_classification,
        }


class PathResolver:
    """Resolve AgentX state paths behind an injectable environment boundary."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = dict(os.environ if environ is None else environ)

    def resolve(self) -> AgentXPaths:
        root = self._resolve_root()
        settings = Path(self._environ.get("AGENTX_SETTINGS", root / "settings.json"))
        return AgentXPaths(
            root=root,
            settings=settings,
            sessions=Path(self._environ.get("AGENTX_SESSIONS", root / "sessions")),
            memories=Path(self._environ.get("AGENTX_MEMORIES", root / "memories")),
            auth=Path(self._environ.get("AGENTX_AUTH", root / "auth")),
        )

    def _resolve_root(self) -> Path:
        explicit = self._environ.get("AGENTX_HOME")
        if explicit:
            return Path(explicit)

        app_data = self._environ.get("APPDATA") or self._environ.get("XDG_DATA_HOME")
        if app_data:
            return Path(app_data) / "agentx"

        home = self._environ.get("HOME")
        if home:
            return Path(home) / ".agentx"

        return Path.home() / ".agentx"


class SettingsLoader:
    def __init__(self, paths: AgentXPaths) -> None:
        self._paths = paths

    def load(self) -> Settings:
        if not self._paths.settings.exists():
            return Settings(paths=self._paths)

        suffix = self._paths.settings.suffix.lower()

        try:
            text = self._paths.settings.read_text(encoding="utf-8")
            raw = _parse_settings_text(text, suffix)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid settings JSON: {exc}") from exc

        if not isinstance(raw, dict):
            raise ConfigError("Settings must be a JSON object.")

        return Settings(
            paths=self._paths,
            public_providers=_string_tuple(raw.get("public_providers", ())),
            private_provider=_optional_string(raw.get("private_provider")),
            external_max_classification=_required_string(
                raw.get("external_max_classification", "internal"),
                "external_max_classification",
            ),
        )


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    paths = PathResolver(environ).resolve()
    return SettingsLoader(paths).load()


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError("public_providers must be a list of strings.")
    if not all(isinstance(item, str) for item in value):
        raise ConfigError("public_providers must be a list of strings.")
    return tuple(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError("private_provider must be a string.")
    return value


def _parse_settings_text(text: str, suffix: str) -> object:
    if suffix in {"", ".json"}:
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        return _parse_simple_yaml(text)
    raise ConfigError(
        f"Unsupported settings format '{suffix}'. Use JSON or YAML settings."
    )


def _parse_simple_yaml(text: str) -> dict[str, object]:
    result: dict[str, object] = {}
    current_list_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            if current_list_key is None:
                raise ConfigError("YAML list item found without a key.")
            existing = result.setdefault(current_list_key, [])
            if not isinstance(existing, list):
                raise ConfigError(f"YAML key '{current_list_key}' is not a list.")
            existing.append(_parse_yaml_scalar(stripped[2:].strip()))
            continue

        current_list_key = None
        if ":" not in stripped:
            raise ConfigError(f"Invalid YAML settings line: {raw_line}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ConfigError("YAML settings keys cannot be empty.")

        if value == "":
            result[key] = []
            current_list_key = key
        else:
            result[key] = _parse_yaml_scalar(value)

    return result


def _parse_yaml_scalar(value: str) -> object:
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a string.")
    return value
