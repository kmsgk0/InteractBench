from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_SETTINGS = Path("settings/models.yaml")
EXAMPLE_MODEL_SETTINGS = Path("settings/models.yaml.example")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    enabled: bool
    provider: str
    model: str
    endpoint: str
    credential_env: str
    auth_mode: str = "bearer"
    auth_header: str | None = None
    auth_param: str | None = None
    max_tokens: int | None = None
    timeout_s: float | None = None
    disable_system_instruction: bool = False
    options: dict[str, Any] = field(default_factory=dict)


def _infer_provider(provider: str, endpoint: str) -> str:
    key = provider.strip().lower()
    if key in {"gemini", "google_gemini"}:
        return "gemini_v1beta"
    if key:
        return key
    endpoint = endpoint.lower()
    if "/v1beta" in endpoint and "/openai" not in endpoint:
        return "gemini_v1beta"
    return "openai_compatible"


def _profile_from_mapping(name: str, raw: dict[str, Any]) -> ModelProfile:
    options = raw.get("options") or {}
    auth = raw.get("credential") or raw.get("credentials") or {}
    return ModelProfile(
        name=name,
        enabled=raw.get("enabled") is True,
        provider=_infer_provider(str(raw.get("provider") or ""), str(raw.get("endpoint") or "")),
        model=str(raw.get("model") or name),
        endpoint=str(raw.get("endpoint") or ""),
        credential_env=str(auth.get("env") or ""),
        auth_mode=str(auth.get("mode") or "bearer").strip().lower(),
        auth_header=auth.get("header"),
        auth_param=auth.get("param"),
        max_tokens=options.get("max_tokens"),
        timeout_s=options.get("timeout_s"),
        disable_system_instruction=options.get("disable_system_instruction") is True,
        options=dict(options),
    )


def load_model_profiles(path: Path | None = None) -> tuple[Path, dict[str, ModelProfile]]:
    config_path = Path(path) if path is not None else DEFAULT_MODEL_SETTINGS
    if not config_path.exists() and path is None and EXAMPLE_MODEL_SETTINGS.exists():
        config_path = EXAMPLE_MODEL_SETTINGS
    if not config_path.exists():
        raise FileNotFoundError(f"model settings not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        root = yaml.load(f, Loader=yaml.FullLoader) or {}
    models = root.get("models") or {}
    profiles = {
        str(name): _profile_from_mapping(str(name), raw or {})
        for name, raw in models.items()
    }
    return config_path, profiles


def enabled_profile_names(profiles: dict[str, ModelProfile]) -> list[str]:
    return [name for name, profile in profiles.items() if profile.enabled]
