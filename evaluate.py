#!/usr/bin/env python3
"""Compute InteractBench pass@k and primary failure aggregation."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from interactbench.code_layout import (
    get_language_spec,
    is_sample_id,
    result_path as build_result_path,
    resolve_result_variant_name,
)

PRIMARY_FAILURES: tuple[str, ...] = (
    "IDLE",
    "PE",
    "CE",
    "TLE",
    "MLE",
    "RE",
    "WA",
    "QLE",
)
MIN_SAMPLES_PER_PROBLEM = 5


class InsufficientSamplesError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        raise ValueError(f"n ({n}) must be >= k ({k})")
    if n - c < k and c > 0:
        return 1.0
    if c == 0:
        return 0.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _empty_failure_counts() -> dict[str, int]:
    return {label: 0 for label in PRIMARY_FAILURES}


def _rate_fail_fields(counts: dict[str, int], failed_total: int) -> dict[str, float]:
    if failed_total <= 0:
        return {f"{label}_rate_fail": 0.0 for label in PRIMARY_FAILURES}
    return {
        f"{label}_rate_fail": round(counts[label] / failed_total, 4)
        for label in PRIMARY_FAILURES
    }


def _effective_case_verdict(
    result: dict[str, Any],
    *,
    memory_limit_kb: int | None = None,
) -> str | None:
    verdict = str(result.get("verdict", "") or "")
    if verdict == "SKIP":
        return None

    queries = _to_int(result.get("queries"))
    query_limit = _to_int(result.get("query_limit"))
    rss_kb = _to_int(result.get("max_rss_solution_kb"))
    mem_over = (
        memory_limit_kb is not None
        and rss_kb is not None
        and rss_kb > memory_limit_kb
    )

    if verdict == "IDLE":
        return "IDLE"
    if verdict == "PE":
        return "PE"
    if verdict == "CE":
        return "CE"
    if verdict == "TLE":
        return "TLE"
    if verdict == "MLE":
        return "MLE"
    if mem_over:
        return "MLE"
    if verdict == "RE":
        return "RE"
    if verdict == "WA":
        return "WA"
    if verdict == "QLE":
        return "QLE"
    if verdict == "OK":
        if queries is not None and query_limit is not None and queries > query_limit:
            return "QLE"
        score = result.get("score")
        if score is not None:
            try:
                if float(score) < 0.9999:
                    return "WA"
            except (TypeError, ValueError):
                return "WA"
        return "OK"
    raise ValueError(
        f"unsupported verdict in result.json: {verdict!r}; "
        "judging failures must be resolved before aggregation"
    )


def categorize_sample(
    cases: dict[str, Any],
    *,
    memory_limit_kb: int | None = None,
) -> dict[str, Any]:
    if not cases:
        return {"passed": False, "primary": None}

    executed = 0

    for _, result in sorted(cases.items()):
        if not isinstance(result, dict):
            result = {}
        verdict = _effective_case_verdict(result, memory_limit_kb=memory_limit_kb)
        if verdict is None:
            continue
        executed += 1
        if verdict != "OK":
            return {"passed": False, "primary": verdict}

    if executed == 0:
        return {"passed": False, "primary": None}
    return {"passed": True, "primary": None}


def _select_code_ids(result_data: dict[str, Any]) -> list[str]:
    return sorted(str(code_id) for code_id in result_data if is_sample_id(str(code_id)))


def _require_minimum_sample_count(
    *,
    selected_code_ids: list[str],
    result_path: Path,
    variant_name: str,
) -> None:
    actual_n = len(selected_code_ids)
    if actual_n < MIN_SAMPLES_PER_PROBLEM:
        raise InsufficientSamplesError(
            f"insufficient samples for {variant_name}: need at least "
            f"{MIN_SAMPLES_PER_PROBLEM}, found {actual_n} in {result_path}"
        )


def _build_metrics(
    *,
    passed_count: int,
    failure_counts: dict[str, int],
    sample_count: int,
) -> dict[str, Any]:
    failed_total = sum(failure_counts.values())
    result: dict[str, Any] = {
        "pass@1": pass_at_k(sample_count, passed_count, 1),
        "pass@5": pass_at_k(sample_count, passed_count, 5),
    }
    result.update(failure_counts)
    result.update(_rate_fail_fields(failure_counts, failed_total))
    return result


def evaluate_problem(
    problem_dir: Path,
    *,
    model: str,
    language: str,
) -> dict[str, Any]:
    meta_path = problem_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")

    meta = load_json(meta_path)
    memory_limit_kb: int | None = None
    try:
        mem_mb = meta.get("memory_limit_mb")
        if mem_mb is not None:
            memory_limit_kb = int(mem_mb) * 1024
    except Exception:
        memory_limit_kb = None

    variant_name = resolve_result_variant_name(problem_dir, model, language)
    result_path = build_result_path(problem_dir, variant_name)
    if not result_path.exists():
        raise FileNotFoundError(f"result.json not found: {result_path}")

    result_data = load_json(result_path)
    if not isinstance(result_data, dict):
        raise ValueError(f"result.json must contain an object: {result_path}")

    selected_code_ids = _select_code_ids(result_data)
    _require_minimum_sample_count(
        selected_code_ids=selected_code_ids,
        result_path=result_path,
        variant_name=variant_name,
    )

    passed_count = 0
    failure_counts = _empty_failure_counts()
    for code_id in selected_code_ids:
        code_result = result_data.get(code_id)
        code_result = code_result if isinstance(code_result, dict) else {}
        cases = code_result.get("cases", {})
        sample = categorize_sample(cases, memory_limit_kb=memory_limit_kb)
        if sample["passed"]:
            passed_count += 1
            continue
        primary = sample.get("primary")
        if primary is None:
            raise ValueError(
                f"sample {code_id!r} has no executable case verdicts in {result_path}"
            )
        if primary in failure_counts:
            failure_counts[primary] += 1

    return _build_metrics(
        passed_count=passed_count,
        failure_counts=failure_counts,
        sample_count=len(selected_code_ids),
    )


def aggregate_problems(problem_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not problem_results:
        raise ValueError("no problem results to aggregate")

    sum_pass1 = 0.0
    sum_pass5 = 0.0
    failure_counts = _empty_failure_counts()

    for item in problem_results:
        sum_pass1 += float(item.get("pass@1") or 0.0)
        sum_pass5 += float(item.get("pass@5") or 0.0)
        for label in PRIMARY_FAILURES:
            failure_counts[label] += int(item.get(label, 0) or 0)

    result = {
        "pass@1": round(sum_pass1 / len(problem_results), 4),
        "pass@5": round(sum_pass5 / len(problem_results), 4),
    }
    result.update(failure_counts)
    result.update(_rate_fail_fields(failure_counts, sum(failure_counts.values())))
    return result


def _list_problem_dirs(problems_dir: Path) -> list[Path]:
    if not problems_dir.exists() or not problems_dir.is_dir():
        raise FileNotFoundError(f"problems dir not found: {problems_dir}")
    return [
        path
        for path in sorted(problems_dir.iterdir())
        if path.is_dir() and (path / "meta.json").exists()
    ]


def _evaluate_available_problems(
    problem_dirs: list[Path],
    *,
    model: str,
    language: str,
) -> list[dict[str, Any]]:
    problem_results: list[dict[str, Any]] = []
    for problem_dir in problem_dirs:
        try:
            result = evaluate_problem(
                problem_dir,
                model=model,
                language=language,
            )
        except (FileNotFoundError, InsufficientSamplesError):
            continue
        problem_results.append(result)
    return problem_results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="InteractBench evaluator")
    parser.add_argument("--problem-id", default=None, help="evaluate a single problem (default: all problems)")
    parser.add_argument("--model", required=True, help="model profile name used by generate.py")
    parser.add_argument("--language", default="cpp", help="solution language (default: cpp)")
    parser.add_argument("--out", default=None, help="output json path")
    parser.add_argument("--problems-dir", default="data/problems")
    args = parser.parse_args(argv)

    problems_dir = Path(args.problems_dir)

    try:
        spec = get_language_spec(args.language)
        if args.problem_id:
            problem_dir = problems_dir / args.problem_id
            if not problem_dir.exists():
                raise FileNotFoundError(f"problem not found: {problem_dir}")
            output = evaluate_problem(
                problem_dir,
                model=args.model,
                language=spec.tag,
            )
        else:
            problem_results = _evaluate_available_problems(
                _list_problem_dirs(problems_dir),
                model=args.model,
                language=spec.tag,
            )
            output = aggregate_problems(problem_results)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[evaluate.py] {exc}", file=sys.stderr)
        return 2

    if args.out:
        json_str = json.dumps(output, indent=2, ensure_ascii=False)
        Path(args.out).write_text(json_str + "\n", encoding="utf-8")
        print(f"[evaluate.py] wrote: {args.out}")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
