#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from interactbench.code_layout import (
    build_variant_name,
    format_sample_id,
    get_language_spec,
    parse_sample_id,
    sample_code_path,
    sample_transcript_path,
    samples_dir,
    transcripts_dir,
)
from interactbench.model_profiles import (
    ModelProfile,
    enabled_profile_names,
    load_model_profiles,
)

_PROMPT_TEMPLATE = """You are a competitive programming expert solving an INTERACTIVE problem.
Implement a complete {LANGUAGE} solution that deduces hidden parameters strictly through the defined query protocol.
Your solution will be compiled and executed against an interactive judge; only the program's interactive behavior under the protocol is evaluated.

Requirements:
- If the statement defines a stdin/stdout interaction protocol, write a complete single-file program that follows that protocol. Java stdin/stdout solutions must use `public class Main`.
- If the statement defines an official grader or library interface, implement exactly the required functions, classes, or headers. Do not add a separate `main` or stdin/stdout protocol unless the statement requires it.
- For stdin/stdout interactions, flush stdout after every query using the idiomatic mechanism in {LANGUAGE} (examples):
  - C++: `std::endl`, `std::cout << std::flush`, or `std::cout.flush()`.
  - Python: `print(..., flush=True)` or `sys.stdout.flush()`.
  - Java: `System.out.flush()`.
  - Go: if using `bufio.Writer`, call `w.Flush()` after each query.
- Terminate immediately if the judge returns an invalid response (e.g., -1).
- No debug output to stdout.

Output format:
- Output exactly one fenced Markdown code block, and nothing else:
```{LANGUAGE_TAG}
// complete {LANGUAGE_TAG} program
```
"""
_CODE_BLOCK_RE = re.compile(
    r"```[a-zA-Z0-9_+-]*\r?\n(.*?)\r?\n\s*```",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class GenerationJob:
    profile: ModelProfile
    variant_name: str
    sample_index: int
    system_prompt: str
    user_msg: str
    max_tokens: int | None
    temperature: float
    max_retries: int
    code_path: Path
    transcript_path: Path


@dataclass(frozen=True)
class GenerationResult:
    sample_index: int
    ok: bool
    code_path: Path | None = None


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _extract_fenced_code(text: str) -> str:
    text = (text or "").strip()
    matches = _CODE_BLOCK_RE.findall(text)
    if not matches:
        return ""
    return matches[-1].strip()


def _default_problem_dir(problem_id: str, problems_dir: str = "data/problems") -> Path:
    return Path(problems_dir) / problem_id


def _load_prompt(language: str) -> tuple[str, str]:
    spec = get_language_spec(language)
    lang_name, lang_tag, ext = spec.display_name, spec.tag, spec.extension
    prompt = (
        _PROMPT_TEMPLATE.replace("{LANGUAGE}", lang_name)
        .replace("{LANGUAGE_TAG}", lang_tag)
    )
    return prompt, ext


def _load_problem(problem_dir: Path) -> str:
    desc_path = problem_dir / "desc.md"
    meta_path = problem_dir / "meta.json"
    if not desc_path.exists():
        raise FileNotFoundError(f"missing desc.md: {desc_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta.json: {meta_path}")
    desc = _read_text(desc_path)
    json.loads(_read_text(meta_path))
    return desc


def _format_user_message(desc: str) -> str:
    return f"# description\n{desc}\n\n"


def _existing_samples(code_dir: Path, ext: str) -> set[int]:
    if not code_dir.exists():
        return set()
    used: set[int] = set()
    for path in code_dir.glob(f"sample-*{ext}"):
        sample_index = parse_sample_id(path.stem)
        if sample_index is not None and path.stat().st_size > 0:
            used.add(sample_index)
    return used


def _selected_profile_names(profiles: dict[str, ModelProfile], models_arg: str | None) -> tuple[list[str], list[str]]:
    enabled = enabled_profile_names(profiles)
    if not models_arg:
        return enabled, enabled

    wanted = [x.strip() for x in models_arg.split(",") if x.strip()]
    unknown = [name for name in wanted if name not in profiles]
    if unknown:
        raise ValueError(f"unknown model profiles: {', '.join(unknown)}")
    return enabled, wanted


def _retry_delays(max_retries: int) -> list[float]:
    return [0.0] + [min(8.0, 1.5 * (i + 1)) for i in range(max(0, max_retries - 1))]


async def _call_until_code(job: GenerationJob) -> tuple[str | None, str | None]:
    from interactbench.llm import async_call_model

    raw: str | None = None
    for retry_number, delay_s in enumerate(_retry_delays(job.max_retries)):
        if delay_s:
            await asyncio.sleep(delay_s)
        try:
            raw = await async_call_model(
                job.profile,
                job.system_prompt,
                job.user_msg,
                max_tokens=job.max_tokens,
                temperature=job.temperature,
            )
        except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
            print(
                f"[generate.py] {job.variant_name}/{format_sample_id(job.sample_index)} "
                f"retry {retry_number}: {exc}",
                file=sys.stderr,
            )
            continue
        code = _extract_fenced_code(raw)
        if code:
            return raw, code
        print(
            f"[generate.py] {job.variant_name}/{format_sample_id(job.sample_index)} "
            f"retry {retry_number}: fenced code block missing",
            file=sys.stderr,
        )
    return raw, None


async def _write_generated_sample(job: GenerationJob, gate: asyncio.Semaphore) -> GenerationResult:
    async with gate:
        raw, code = await _call_until_code(job)
    if not code:
        print(
            f"[generate.py] {job.variant_name}/{format_sample_id(job.sample_index)}: no usable code",
            file=sys.stderr,
        )
        return GenerationResult(sample_index=job.sample_index, ok=False)

    job.transcript_path.parent.mkdir(parents=True, exist_ok=True)
    job.code_path.parent.mkdir(parents=True, exist_ok=True)
    job.transcript_path.write_text(raw or "", encoding="utf-8")
    job.code_path.write_text(code.rstrip() + "\n", encoding="utf-8")
    print(f"[generate.py] wrote: {job.code_path}")
    return GenerationResult(sample_index=job.sample_index, ok=True, code_path=job.code_path)


async def _run_generation_jobs(jobs: list[GenerationJob], max_parallel: int) -> list[GenerationResult]:
    gate = asyncio.Semaphore(max(1, max_parallel))
    return list(await asyncio.gather(*(_write_generated_sample(job, gate) for job in jobs)))


def _build_jobs(
    *,
    profiles: dict[str, ModelProfile],
    selected: list[str],
    problem_dir: Path,
    language: str,
    sample_indices: dict[str, list[int]],
    system_prompt: str,
    user_msg: str,
    max_tokens: int | None,
    temperature: float,
    max_retries: int,
    file_ext: str,
) -> list[GenerationJob]:
    lang_tag = get_language_spec(language).tag
    jobs: list[GenerationJob] = []
    for profile_name in selected:
        profile = profiles[profile_name]
        variant_name = build_variant_name(profile_name, lang_tag)
        for sample_index in sample_indices.get(profile_name, []):
            jobs.append(
                GenerationJob(
                    profile=profile,
                    variant_name=variant_name,
                    sample_index=sample_index,
                    system_prompt=system_prompt,
                    user_msg=user_msg,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_retries=max_retries,
                    code_path=sample_code_path(problem_dir, variant_name, sample_index, file_ext),
                    transcript_path=sample_transcript_path(problem_dir, variant_name, sample_index),
                )
            )
    return jobs


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate InteractBench solution samples")
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--problems-dir", default="data/problems", help="problems directory")
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model profile names (default: enabled in settings/models.yaml)",
    )
    parser.add_argument("--model-settings", default=None, help="model settings YAML path")
    parser.add_argument("--n", type=int, default=10, help="number of samples per model (default: 10)")
    parser.add_argument("--k", type=int, default=None, help="explicit sample index (only valid when --n=1)")
    parser.add_argument("--dry-run", action="store_true", help="print prompt and exit (no API call)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing samples (use k=1..n)")
    parser.add_argument("--max-retries", type=int, default=3, help="max extraction retries (default: 3)")
    parser.add_argument("--max-parallel", type=int, default=4, help="maximum concurrent API calls (default: 4)")
    parser.add_argument("--max-tokens", type=int, default=None, help="max output tokens (default: profile options)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--language", default="cpp", help="target language (cpp/python/java/go)")
    args = parser.parse_args(argv)

    if args.n < 1:
        print("[generate.py] --n must be >= 1", file=sys.stderr)
        return 2
    if args.max_retries < 1:
        print("[generate.py] --max-retries must be >= 1", file=sys.stderr)
        return 2
    if args.max_parallel < 1:
        print("[generate.py] --max-parallel must be >= 1", file=sys.stderr)
        return 2
    if args.k is not None and args.n != 1:
        print("[generate.py] --k can only be used with --n=1", file=sys.stderr)
        return 2
    if args.k is not None and args.k < 1:
        print("[generate.py] --k must be >= 1", file=sys.stderr)
        return 2
    if args.k is not None and args.overwrite:
        print("[generate.py] --k and --overwrite cannot be used together", file=sys.stderr)
        return 2

    problem_dir = _default_problem_dir(args.problem_id, args.problems_dir)
    if not problem_dir.exists():
        print(f"[generate.py] problem not found: {problem_dir}", file=sys.stderr)
        return 2

    load_dotenv(override=False)
    try:
        model_settings_path, profiles = load_model_profiles(Path(args.model_settings) if args.model_settings else None)
        enabled, selected = _selected_profile_names(profiles, args.models)
        system_prompt, file_ext = _load_prompt(args.language)
        desc = _load_problem(problem_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[generate.py] setup error: {exc}", file=sys.stderr)
        return 2

    user_msg = _format_user_message(desc)
    if args.dry_run:
        print(f"[generate.py] model_settings={model_settings_path}")
        print(f"[generate.py] enabled={enabled}")
        print(f"[generate.py] selected={selected}")
        print("----- system prompt -----")
        print(system_prompt)
        print("----- user message -----")
        print(user_msg)
        return 0

    if not selected:
        print(f"[generate.py] no model profiles selected (edit {model_settings_path})", file=sys.stderr)
        return 2

    lang_tag = get_language_spec(args.language).tag
    sample_indices: dict[str, list[int]] = {}
    any_ran = False
    for profile_name in selected:
        profile = profiles[profile_name]
        if not profile.enabled and args.models is None:
            continue
        any_ran = True
        variant_name = build_variant_name(profile_name, lang_tag)
        code_dir = samples_dir(problem_dir, variant_name)
        transcripts_dir(problem_dir, variant_name).mkdir(parents=True, exist_ok=True)
        code_dir.mkdir(parents=True, exist_ok=True)
        if args.k is not None:
            indices = [args.k]
        elif args.overwrite:
            indices = list(range(1, args.n + 1))
        else:
            existing = _existing_samples(code_dir, file_ext)
            indices = [idx for idx in range(1, args.n + 1) if idx not in existing]
            if not indices:
                print(f"[generate.py] {variant_name}: all {args.n} samples exist, skipping")
                continue
        sample_indices[profile_name] = indices

    if not any_ran:
        print("[generate.py] no valid model profiles found", file=sys.stderr)
        return 2

    jobs = _build_jobs(
        profiles=profiles,
        selected=selected,
        problem_dir=problem_dir,
        language=args.language,
        sample_indices=sample_indices,
        system_prompt=system_prompt,
        user_msg=user_msg,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        file_ext=file_ext,
    )
    if not jobs:
        return 0

    results = asyncio.run(_run_generation_jobs(jobs, args.max_parallel))
    return 1 if any(not item.ok for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
