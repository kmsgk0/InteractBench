from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SAMPLES_DIR_NAME = "samples"
TRANSCRIPTS_DIR_NAME = "transcripts"
RESULT_FILENAME = "result.json"
_SAMPLE_ID_RE = re.compile(r"^sample-(\d{3})$")


@dataclass(frozen=True)
class LanguageSpec:
    display_name: str
    tag: str
    extension: str


_LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "cpp": LanguageSpec("C++", "cpp", ".cpp"),
    "c++": LanguageSpec("C++", "cpp", ".cpp"),
    "python": LanguageSpec("Python", "python", ".py"),
    "py": LanguageSpec("Python", "python", ".py"),
    "java": LanguageSpec("Java", "java", ".java"),
    "go": LanguageSpec("Go", "go", ".go"),
}

_EXTENSION_TO_SPEC: dict[str, LanguageSpec] = {
    spec.extension: spec for spec in {spec.tag: spec for spec in _LANGUAGE_SPECS.values()}.values()
}


def get_language_spec(language: str) -> LanguageSpec:
    key = str(language or "").strip().lower()
    spec = _LANGUAGE_SPECS.get(key)
    if spec is None:
        raise ValueError(f"unsupported language: {language}")
    return spec


def get_language_spec_from_path(code_path: Path) -> LanguageSpec:
    spec = _EXTENSION_TO_SPEC.get(Path(code_path).suffix.lower())
    if spec is None:
        raise ValueError(f"unsupported solution language: {Path(code_path).suffix}")
    return spec


def build_variant_name(model: str, language: str) -> str:
    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("model must be non-empty")
    return f"{model_name}__{get_language_spec(language).tag}"


def variant_root(problem_dir: Path, variant_name: str) -> Path:
    return Path(problem_dir) / "codes" / variant_name


def samples_dir(problem_dir: Path, variant_name: str) -> Path:
    return variant_root(problem_dir, variant_name) / SAMPLES_DIR_NAME


def transcripts_dir(problem_dir: Path, variant_name: str) -> Path:
    return variant_root(problem_dir, variant_name) / TRANSCRIPTS_DIR_NAME


def result_path(problem_dir: Path, variant_name: str) -> Path:
    return variant_root(problem_dir, variant_name) / RESULT_FILENAME


def format_sample_id(index: int) -> str:
    if int(index) < 1:
        raise ValueError(f"sample index must be >= 1: {index}")
    return f"sample-{int(index):03d}"


def parse_sample_id(code_id: str) -> int | None:
    match = _SAMPLE_ID_RE.match(str(code_id or ""))
    if not match:
        return None
    index = int(match.group(1))
    return index if index >= 1 else None


def is_sample_id(code_id: str) -> bool:
    return parse_sample_id(code_id) is not None


def sample_code_path(problem_dir: Path, variant_name: str, index: int, extension: str) -> Path:
    return samples_dir(problem_dir, variant_name) / f"{format_sample_id(index)}{extension}"


def sample_transcript_path(problem_dir: Path, variant_name: str, index: int) -> Path:
    return transcripts_dir(problem_dir, variant_name) / f"{format_sample_id(index)}.txt"


def resolve_result_variant_name(problem_dir: Path, model: str, language: str) -> str:
    """
    Resolve the concrete codes/<variant>/ directory used for aggregation.
    """
    problem_dir = Path(problem_dir)
    codes_dir = problem_dir / "codes"
    if not codes_dir.exists() or not codes_dir.is_dir():
        raise FileNotFoundError(f"codes dir not found: {codes_dir}")

    language_spec = get_language_spec(language)
    variant_name = build_variant_name(model, language_spec.tag)
    exact_dir = codes_dir / variant_name
    if exact_dir.exists() and exact_dir.is_dir():
        return variant_name

    raise FileNotFoundError(
        f"result variant not found for model={str(model or '').strip()!r}, "
        f"language={language_spec.tag}: expected {codes_dir / variant_name}"
    )


def infer_variant_name_from_code_path(code_path: Path) -> str:
    parts = Path(code_path).parts
    try:
        idx = parts.index("codes")
    except ValueError:
        return "unknown"
    return parts[idx + 1] if idx + 1 < len(parts) else "unknown"


def resolve_variant_name(
    *,
    model: str | None,
    language: str | None = None,
    code_path: Path | None = None,
) -> str:
    model_name = str(model or "").strip()
    if model_name:
        resolved_language = language
        if resolved_language is None:
            if code_path is None:
                raise ValueError("language is required when model is provided")
            resolved_language = get_language_spec_from_path(code_path).tag
        return build_variant_name(model_name, resolved_language)

    if code_path is None:
        raise ValueError("code_path is required when model is omitted")

    variant_name = infer_variant_name_from_code_path(code_path)
    if variant_name == "unknown":
        raise ValueError("model is required when --code-path is outside codes/<variant>/")
    return variant_name


def infer_code_id(code_path: Path) -> str:
    return Path(code_path).stem


def discover_code_paths(problem_dir: Path, model: str, language: str) -> tuple[str, list[Path]]:
    variant_name = build_variant_name(model, language)
    spec = get_language_spec(language)
    code_dir = samples_dir(problem_dir, variant_name)
    if not code_dir.exists() or not code_dir.is_dir():
        raise FileNotFoundError(f"sample directory not found: {code_dir}")

    code_paths = sorted(
        path
        for path in code_dir.glob(f"*{spec.extension}")
        if path.is_file() and is_sample_id(path.stem)
    )
    if not code_paths:
        raise FileNotFoundError(f"no code files matched: {code_dir / ('*' + spec.extension)}")

    return variant_name, code_paths
