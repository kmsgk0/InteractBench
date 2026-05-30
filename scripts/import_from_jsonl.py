#!/usr/bin/env python3
"""Import problems from jsonl files to directory structure."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ANCILLARY_CASE_STEMS: frozenset[str] = frozenset({
    "groups",
    "license",
    "protocol",
    "readme",
    "notes",
})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _reset_managed_problem_paths(
    problem_dir: Path,
    *,
    managed_files: tuple[str, ...],
    managed_dirs: tuple[str, ...],
) -> None:
    problem_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in managed_files:
        _remove_path(problem_dir / rel_path)
    for rel_path in managed_dirs:
        _remove_path(problem_dir / rel_path)


def _read_int_tokens(path: Path) -> list[int]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError as exc:
        raise ValueError(f"failed to read case file: {path}") from exc
    if not raw:
        return []
    try:
        return [int(token) for token in raw.split()]
    except ValueError as exc:
        raise ValueError(f"case file does not contain plain integer tokens: {path}") from exc


def _load_materialization(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("materialization")
    return payload if isinstance(payload, dict) else {}


def _apply_exact_replacements(path: Path, replacements: list[dict[str, Any]]) -> None:
    if not path.exists():
        raise ValueError(f"materialization target not found: {path}")

    original = path.read_text(encoding="utf-8")
    updated = original
    for idx, replacement in enumerate(replacements):
        match = replacement.get("match")
        replace = replacement.get("replace")
        if not isinstance(match, str) or not isinstance(replace, str):
            raise ValueError(f"replacement #{idx} for {path} must contain string `match` and `replace`")
        occurrences = updated.count(match)
        if occurrences != 1:
            raise ValueError(
                f"replacement #{idx} for {path} expected exactly 1 match, found {occurrences}"
            )
        updated = updated.replace(match, replace, 1)

    if updated != original:
        path.write_text(updated, encoding="utf-8")


def _derive_output_path(input_path: Path, output_suffix: str) -> Path:
    if input_path.suffix:
        return input_path.with_suffix(output_suffix)
    return input_path.parent / f"{input_path.name}{output_suffix}"


def _materialize_case_outputs(problem_dir: Path, rule: dict[str, Any]) -> None:
    mode = rule.get("mode")
    if mode != "input_token_slice":
        raise ValueError(f"unsupported case_outputs mode: {mode!r}")

    input_glob = rule.get("input_glob", "cases/*")
    output_suffix = rule.get("output_suffix", ".out")
    if not isinstance(input_glob, str) or not isinstance(output_suffix, str):
        raise ValueError("case_outputs rules require string `input_glob` and `output_suffix`")

    try:
        length_index = int(rule["length_index"])
        start_index = int(rule["start_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("case_outputs `input_token_slice` requires integer `length_index` and `start_index`") from exc

    input_paths = [
        path
        for path in sorted(problem_dir.glob(input_glob))
        if path.is_file()
        and not path.name.endswith(output_suffix)
        and path.stem.lower() not in ANCILLARY_CASE_STEMS
    ]
    if not input_paths:
        raise ValueError(
            f"case_outputs rule matched no inputs under {problem_dir} with glob {input_glob!r}"
        )

    for input_path in input_paths:
        tokens = _read_int_tokens(input_path)
        if length_index < 0 or start_index < 0 or length_index >= len(tokens):
            raise ValueError(f"case_outputs indices out of range for {input_path}")
        expected_len = tokens[length_index]
        if expected_len < 0:
            raise ValueError(f"case_outputs length token must be non-negative for {input_path}")
        end_index = start_index + expected_len
        if end_index > len(tokens):
            raise ValueError(f"case_outputs slice overruns payload for {input_path}")
        output_path = _derive_output_path(input_path, output_suffix)
        output_text = " ".join(str(token) for token in tokens[start_index:end_index]) + "\n"
        output_path.write_text(output_text, encoding="utf-8")


def _apply_materialization(problem_dir: Path, record: dict[str, Any]) -> None:
    rules = _load_materialization(record)
    if not rules:
        return

    file_patches = rules.get("file_patches", [])
    if file_patches:
        if not isinstance(file_patches, list):
            raise ValueError("materialization `file_patches` must be a list")
        for idx, patch in enumerate(file_patches):
            if not isinstance(patch, dict):
                raise ValueError(f"file_patches[{idx}] must be an object")
            rel_path = patch.get("path")
            replacements = patch.get("replacements", [])
            if not isinstance(rel_path, str):
                raise ValueError(f"file_patches[{idx}] requires string `path`")
            if not isinstance(replacements, list):
                raise ValueError(f"file_patches[{idx}] requires list `replacements`")
            _apply_exact_replacements(problem_dir / rel_path, replacements)

    case_outputs = rules.get("case_outputs", [])
    if case_outputs:
        if not isinstance(case_outputs, list):
            raise ValueError("materialization `case_outputs` must be a list")
        for idx, rule in enumerate(case_outputs):
            if not isinstance(rule, dict):
                raise ValueError(f"case_outputs[{idx}] must be an object")
            _materialize_case_outputs(problem_dir, rule)


def import_interactbench(jsonl_path: Path, output_dir: Path, problem_ids: set[str] | None = None) -> int:
    """Import standard InteractBench problems."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pid = record["problem_id"]
            if problem_ids and pid not in problem_ids:
                continue
            pdir = output_dir / pid
            _reset_managed_problem_paths(
                pdir,
                managed_files=("meta.json", "desc.md", "problem.json"),
                managed_dirs=("interactor", "generator"),
            )

            meta = {
                "difficulty": record.get("difficulty", ""),
                "cate": record.get("cate", []),
                "cpu_time_limit_ms": record.get("cpu_time_limit_ms"),
                "memory_limit_mb": record.get("memory_limit_mb"),
                "interactor_mode": record.get("interactor_mode", ""),
            }
            _write_json(pdir / "meta.json", meta)

            if record.get("description"):
                (pdir / "desc.md").write_text(record["description"], encoding="utf-8")

            int_dir = pdir / "interactor"
            int_dir.mkdir(exist_ok=True)
            if record.get("interactor_non_adaptive"):
                (int_dir / "non_adaptive.cpp").write_text(
                    record["interactor_non_adaptive"],
                    encoding="utf-8",
                )
            if record.get("interactor_adaptive"):
                (int_dir / "adaptive.cpp").write_text(record["interactor_adaptive"], encoding="utf-8")

            if record.get("generator"):
                gen_dir = pdir / "generator"
                gen_dir.mkdir(exist_ok=True)
                (gen_dir / "gen_cases.cpp").write_text(record["generator"], encoding="utf-8")

            count += 1

    return count


def import_ioi(jsonl_path: Path, output_dir: Path, problem_ids: set[str] | None = None) -> int:
    """Import IOI problems."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pid = record["problem_id"]
            if problem_ids and pid not in problem_ids:
                continue
            pdir = output_dir / pid
            _reset_managed_problem_paths(
                pdir,
                managed_files=("meta.json", "desc.md", "problem.json"),
                managed_dirs=("interactor", "generator"),
            )

            meta = {
                "difficulty": record.get("difficulty", ""),
                "cate": record.get("cate", []),
                "cpu_time_limit_ms": record.get("cpu_time_limit_ms"),
                "memory_limit_mb": record.get("memory_limit_mb"),
            }
            _write_json(pdir / "meta.json", meta)

            if record.get("description"):
                (pdir / "desc.md").write_text(record["description"], encoding="utf-8")

            int_files = record.get("interactor_files", {})
            if int_files:
                int_dir = pdir / "interactor"
                int_dir.mkdir(exist_ok=True)
                for fname, content in int_files.items():
                    (int_dir / fname).write_text(content, encoding="utf-8")

            problem_json = record.get("problem_json")
            if isinstance(problem_json, dict):
                (pdir / "problem.json").write_text(
                    json.dumps(problem_json, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            elif isinstance(problem_json, str) and problem_json.strip():
                (pdir / "problem.json").write_text(problem_json.strip() + "\n", encoding="utf-8")

            try:
                _apply_materialization(pdir, record)
            except ValueError as exc:
                raise ValueError(f"{pid}: {exc}") from exc

            count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Import problems from jsonl")
    parser.add_argument(
        "--type",
        choices=["standard", "ioi"],
        required=True,
        help="Problem type to import",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for problems",
    )
    parser.add_argument(
        "--problem-ids",
        nargs="*",
        default=None,
        help="Only import specific problem IDs",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}")
        return 1

    pids = set(args.problem_ids) if args.problem_ids else None

    try:
        if args.type == "standard":
            count = import_interactbench(args.input, args.output_dir, pids)
        else:
            count = import_ioi(args.input, args.output_dir, pids)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Imported {count} problems to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
