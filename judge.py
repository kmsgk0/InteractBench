#!/usr/bin/env python3
"""Unified judging entry point for all problem families."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from interactbench.code_layout import (
    discover_code_paths,
    format_sample_id,
    get_language_spec,
    get_language_spec_from_path,
    infer_variant_name_from_code_path,
    is_sample_id,
    resolve_variant_name,
)


def _is_ioi_problem_id(problem_id: str) -> bool:
    return problem_id.upper().startswith("IOI")


def _load_runner(problem_id: str) -> Callable[[list[str]], int]:
    if _is_ioi_problem_id(problem_id):
        from interactbench.runners.run_ioi import main as runner_main
    else:
        from interactbench.runners.run_main import main as runner_main
    return runner_main


def _list_problem_ids(problem_root: Path) -> list[str]:
    if not problem_root.exists() or not problem_root.is_dir():
        raise FileNotFoundError(f"problems root not found: {problem_root}")
    return [
        path.name
        for path in sorted(problem_root.iterdir())
        if path.is_dir() and (path / "meta.json").exists()
    ]


def _default_code_id_for_code_path(code_path: Path) -> str:
    stem = Path(code_path).stem
    if is_sample_id(stem):
        return stem
    if infer_variant_name_from_code_path(code_path) != "unknown":
        return stem
    return format_sample_id(1)


def _build_backend_argv(
    *,
    problem_id: str,
    problem_root: Path,
    code_path: Path,
    variant_name: str | None,
    code_id: str | None,
    args: argparse.Namespace,
) -> list[str]:
    argv = [
        "--problem-id", problem_id,
        "--problem-dir", str(problem_root),
        "--code-path", str(code_path),
    ]
    if variant_name:
        argv.extend(["--model", variant_name])
    if code_id:
        argv.extend(["--code-id", code_id])
    if args.cases_glob:
        argv.extend(["--cases-glob", args.cases_glob])
    if args.cpu_ms is not None:
        argv.extend(["--cpu-ms", str(args.cpu_ms)])
    if not _is_ioi_problem_id(problem_id):
        if args.mem_mb is not None:
            argv.extend(["--mem-mb", str(args.mem_mb)])
        if args.gojudge_url:
            argv.extend(["--gojudge-url", args.gojudge_url])
        if args.interactor_path:
            argv.extend(["--interactor-path", args.interactor_path])
        if args.adaptive_interactor_path:
            argv.extend(["--adaptive-interactor-path", args.adaptive_interactor_path])
    return argv


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Unified judging entry point")
    parser.add_argument("--problem-id", default=None, help="judge a single problem (default: all problems)")
    parser.add_argument("--problem-dir", default="data/problems", help="problem root directory")
    parser.add_argument("--model", default=None, help="model profile name used by generate.py")
    parser.add_argument("--language", default=None, help="solution language (default: cpp)")
    parser.add_argument("--code-path", default=None, help="judge a single source file")
    parser.add_argument("--code-id", default=None, help="result entry id for --code-path")
    parser.add_argument("--cases-glob", default="cases/*.in", help="case glob relative to problem dir")
    parser.add_argument("--cpu-ms", type=int, default=None, help="override cpu time limit in ms")
    parser.add_argument("--mem-mb", type=int, default=None, help="override memory limit in MB for the main pipeline")
    parser.add_argument("--gojudge-url", default="http://127.0.0.1:5050", help="go-judge endpoint for the main pipeline")
    parser.add_argument("--interactor-path", default=None, help="override non-adaptive interactor path for the main pipeline")
    parser.add_argument("--adaptive-interactor-path", default=None, help="override adaptive interactor path for the main pipeline")
    args = parser.parse_args(argv)

    problem_root = Path(args.problem_dir)
    if args.code_path and not args.problem_id:
        print("[judge.py] --code-path requires --problem-id", file=sys.stderr)
        return 2
    if args.code_id and not args.code_path:
        print("[judge.py] --code-id requires --code-path", file=sys.stderr)
        return 2
    if not args.code_path and not args.model:
        print("[judge.py] --model is required unless --code-path is provided", file=sys.stderr)
        return 2

    try:
        problem_ids = [args.problem_id] if args.problem_id else _list_problem_ids(problem_root)
    except FileNotFoundError as exc:
        print(f"[judge.py] {exc}", file=sys.stderr)
        return 2
    if not problem_ids:
        print(f"[judge.py] no problems found under: {problem_root}", file=sys.stderr)
        return 2

    batch_language = args.language or "cpp"
    if args.language is not None:
        try:
            get_language_spec(args.language)
        except ValueError as exc:
            print(f"[judge.py] {exc}", file=sys.stderr)
            return 2

    any_failed = False
    for problem_id in problem_ids:
        problem_dir = problem_root / problem_id
        if not problem_dir.exists() or not problem_dir.is_dir():
            print(f"[judge.py] problem not found: {problem_dir}", file=sys.stderr)
            any_failed = True
            continue

        if args.code_path:
            code_paths = [Path(args.code_path)]
            if not code_paths[0].exists():
                print(f"[judge.py] code not found: {code_paths[0]}", file=sys.stderr)
                return 2
            try:
                code_language = get_language_spec_from_path(code_paths[0]).tag
            except ValueError as exc:
                print(f"[judge.py] {exc}", file=sys.stderr)
                return 2
            if args.language is not None and get_language_spec(args.language).tag != code_language:
                print(
                    f"[judge.py] --language={args.language} does not match code file extension for {code_paths[0]}",
                    file=sys.stderr,
                )
                return 2
            try:
                variant_name = resolve_variant_name(
                    model=args.model,
                    language=code_language,
                    code_path=code_paths[0],
                )
            except ValueError as exc:
                print(f"[judge.py] {exc}", file=sys.stderr)
                return 2
            code_id = args.code_id or _default_code_id_for_code_path(code_paths[0])
        else:
            try:
                variant_name, code_paths = discover_code_paths(problem_dir, args.model, batch_language)
            except FileNotFoundError as exc:
                print(f"[judge.py] {problem_id}: {exc}", file=sys.stderr)
                any_failed = True
                continue
            except ValueError as exc:
                print(f"[judge.py] {problem_id}: {exc}", file=sys.stderr)
                return 2
            code_id = None

        runner_main = _load_runner(problem_id)
        print(f"[judge.py] problem={problem_id} variant={variant_name} codes={len(code_paths)}")
        for code_path in code_paths:
            backend_argv = _build_backend_argv(
                problem_id=problem_id,
                problem_root=problem_root,
                code_path=code_path.resolve(),
                variant_name=variant_name,
                code_id=code_id,
                args=args,
            )
            ret = runner_main(backend_argv)
            if ret == 2:
                return 2
            if ret != 0:
                any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
