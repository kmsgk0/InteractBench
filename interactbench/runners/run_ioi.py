#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from interactbench.code_layout import (
    get_language_spec_from_path,
    infer_code_id,
    infer_variant_name_from_code_path,
    result_path as build_result_path,
)
from interactbench.result_store import update_result
from ioi.ioi_evaluator import (
    EvaluationType,
    compile_manager,
    compile_solver_with_grader,
    compile_solver_with_stub,
    compile_stdio_interactor,
    compile_stdio_solver,
    detect_evaluation_type,
    run_dual_process,
    run_grader_communication,
    run_grader_linked,
    run_stdio_interactive,
)


ANCILLARY_CASE_STEMS: frozenset[str] = frozenset({
    "groups",
    "license",
    "protocol",
    "readme",
    "notes",
})

_FREOPEN_RE = re.compile(r'freopen\("([^"]+)",\s*"([rw])",\s*(stdin|stdout)\s*\)')


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_meta(problem_dir: Path) -> dict[str, Any]:
    meta_path = problem_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return _load_json(meta_path)


def _case_key(case_path: Path) -> str:
    if case_path.suffix == ".in":
        return case_path.stem
    return case_path.name


def _list_case_inputs(
    *,
    problem_dir: Path,
    eval_type: EvaluationType,
    cases_glob: str | None,
) -> list[Path]:
    cases_dir = problem_dir / "cases"
    if not cases_dir.exists():
        return []

    if cases_glob:
        selected = sorted(problem_dir.glob(cases_glob))
        selected = [p for p in selected if p.is_file() and p.stem.lower() not in ANCILLARY_CASE_STEMS]
        if selected:
            return selected
        if cases_glob != "cases/*.in":
            return []

    def filter_ancillary(paths: list[Path]) -> list[Path]:
        return [p for p in paths if p.stem.lower() not in ANCILLARY_CASE_STEMS]

    inputs = filter_ancillary(sorted(cases_dir.glob("*.in")))
    if inputs:
        return inputs

    if eval_type is EvaluationType.GRADER_LINKED:
        inputs = filter_ancillary(sorted(cases_dir.glob("grader.in.*")))
        if inputs:
            return inputs
        inputs = filter_ancillary(sorted(cases_dir.glob("*.in.*")))
        if inputs:
            return inputs

    fallback: list[Path] = []
    for p in sorted(cases_dir.iterdir()):
        if p.is_file() and not p.name.endswith(".out") and p.stem.lower() not in ANCILLARY_CASE_STEMS:
            fallback.append(p)
    return fallback


def _read_int_tokens(text: str) -> list[int] | None:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return [int(tok) for tok in stripped.split()]
    except ValueError:
        return None


def _read_int_tokens_from_file(path: Path) -> list[int] | None:
    try:
        return _read_int_tokens(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return None


def _set_token_compare_verdict(rr: Any, actual: list[int], expected: list[int], *, abs_compare: bool = False) -> None:
    if len(actual) != len(expected):
        rr.verdict = "WA"
        rr.score = 0.0
        return
    if abs_compare:
        ok = all(abs(a) == abs(b) for a, b in zip(actual, expected))
    else:
        ok = actual == expected
    rr.verdict = "OK" if ok else "WA"
    rr.score = 1.0 if ok else 0.0


def _resolve_linked_file_io(graders_dir: Path) -> tuple[str, str] | None:
    for candidate in (graders_dir / "grader.cpp", graders_dir / "grader.c"):
        if not candidate.exists():
            continue
        try:
            content = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        input_name: str | None = None
        output_name: str | None = None
        for name, mode, stream in _FREOPEN_RE.findall(content):
            if mode == "r" and stream == "stdin":
                input_name = name
            elif mode == "w" and stream == "stdout":
                output_name = name
        if input_name and output_name:
            return (input_name, output_name)
    return None


def _apply_linked_expected_output_check(case_path: Path, rr: Any) -> None:
    exit_codes = rr.exit_codes or {}
    if exit_codes.get("solver") != 0:
        return

    actual_tokens = _read_int_tokens(rr.stdout_text or "")
    if actual_tokens is None:
        return

    expected_path = case_path.with_suffix(".out")
    if not expected_path.exists():
        return
    expected_tokens = _read_int_tokens_from_file(expected_path)
    if expected_tokens is None:
        return
    _set_token_compare_verdict(rr, actual_tokens, expected_tokens)


def _system_error_payload(*, eval_type: EvaluationType, message: str) -> dict[str, Any]:
    return {
        "queries": None,
        "query_limit": None,
        "cpu_ms": None,
        "wall_ms": None,
        "max_rss_solution_kb": None,
        "max_rss_interactor_kb": None,
        "exit_codes": None,
        "score": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "system_error": message,
        "eval_type": eval_type.value,
    }


def _write_code_result(result_path: Path, code_id: str, code_entry: dict[str, Any]) -> None:
    update_result(result_path, lambda d: d.__setitem__(str(code_id), code_entry))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run IOI evaluation for data/problems/<problem_id>")
    parser.add_argument("--problem-id", required=True)
    parser.add_argument(
        "--problem-dir",
        default=None,
        help="problem root dir containing <problem-id>/ (default: data/problems)",
    )
    parser.add_argument("--model", default=None, help="override model name (default: inferred from code path)")
    parser.add_argument("--code-id", default=None, help="override code_id (default: inferred from code path)")
    parser.add_argument("--code-path", default=None, help="path to solver source file")
    parser.add_argument("--cases-glob", default="cases/*.in", help="case glob relative to problem dir (default: cases/*.in)")
    parser.add_argument("--cpu-ms", type=int, default=None, help="CPU time limit hint (ms); used as wall budget")
    args = parser.parse_args(argv)

    problem_root = Path(args.problem_dir) if args.problem_dir else Path("data/problems")
    problem_dir = problem_root / args.problem_id
    if not problem_dir.exists() or not problem_dir.is_dir():
        print(f"[judge.py] problem not found: {problem_dir}", file=sys.stderr)
        return 2

    if not args.code_path:
        print("[judge.py] --code-path is required", file=sys.stderr)
        return 2

    code_path = Path(args.code_path)
    if not code_path.exists():
        print(f"[judge.py] code not found: {code_path}", file=sys.stderr)
        return 2
    try:
        code_language = get_language_spec_from_path(code_path).tag
    except ValueError as exc:
        print(f"[judge.py] {exc}", file=sys.stderr)
        return 2
    if code_language != "cpp":
        print(f"[judge.py] IOI judging requires C++ source, got: {code_path}", file=sys.stderr)
        return 2

    variant_name = args.model or infer_variant_name_from_code_path(code_path)
    code_id = args.code_id or infer_code_id(code_path)

    meta = _load_meta(problem_dir)
    cpu_ms = args.cpu_ms or int(meta.get("cpu_time_limit_ms") or 2000)
    time_limit_s = max(0.1, float(cpu_ms) / 1000.0)

    eval_info = detect_evaluation_type(problem_dir)
    case_paths = _list_case_inputs(
        problem_dir=problem_dir,
        eval_type=eval_info.eval_type,
        cases_glob=args.cases_glob,
    )
    if not case_paths:
        print(f"[judge.py] no cases found under: {problem_dir / 'cases'}", file=sys.stderr)
        return 2

    result_path = build_result_path(problem_dir, variant_name)
    code_entry: dict[str, Any] = {"cases": {}, "eval_type": eval_info.eval_type.value}
    had_system_error = False

    def record_all(verdict: str, payload: dict[str, Any]) -> None:
        for cp in case_paths:
            code_entry["cases"][_case_key(cp)] = {"verdict": verdict, **payload}

    with tempfile.TemporaryDirectory(prefix=f"ioi_{args.problem_id}_{variant_name}_{code_id}_") as tmp:
        work_dir = Path(tmp)

        manager_exe: Path | None = None
        solver_exe: Path | None = None
        interactor_exe: Path | None = None
        solver_uses_stdio = True

        graders_dir = eval_info.grader_files.manager.parent if (eval_info.grader_files and eval_info.grader_files.manager) else (problem_dir / "interactor")
        linked_file_io = _resolve_linked_file_io(graders_dir)

        try:
            if eval_info.eval_type == EvaluationType.STDIO_INTERACTIVE:
                interactor_src = problem_dir / "interactor" / "non_adaptive.cpp"
                if not interactor_src.exists():
                    record_all(
                        "SE",
                        _system_error_payload(
                            eval_type=eval_info.eval_type,
                            message=f"missing stdio interactor: {interactor_src}",
                        ),
                    )
                    _write_code_result(result_path, code_id, code_entry)
                    return 1

                solc = compile_stdio_solver(code_path, work_dir)
                if not solc.success or not solc.executable:
                    record_all("CE", {"compile_error": solc.message})
                    _write_code_result(result_path, code_id, code_entry)
                    return 0
                solver_exe = solc.executable

                intc = compile_stdio_interactor(interactor_src, work_dir)
                if not intc.success or not intc.executable:
                    record_all(
                        "SE",
                        _system_error_payload(
                            eval_type=eval_info.eval_type,
                            message=f"interactor compile error: {intc.message}",
                        ),
                    )
                    _write_code_result(result_path, code_id, code_entry)
                    return 1
                interactor_exe = intc.executable

            elif eval_info.eval_type in (EvaluationType.GRADER_COMMUNICATION, EvaluationType.DUAL_PROCESS):
                if eval_info.grader_files is None or eval_info.grader_files.manager is None:
                    record_all(
                        "SE",
                        _system_error_payload(
                            eval_type=eval_info.eval_type,
                            message="missing manager.cpp for grader_communication",
                        ),
                    )
                    _write_code_result(result_path, code_id, code_entry)
                    return 1

                mgrc = compile_manager(graders_dir, work_dir)
                if not mgrc.success or not mgrc.executable:
                    record_all(
                        "SE",
                        _system_error_payload(
                            eval_type=eval_info.eval_type,
                            message=f"manager compile error: {mgrc.message}",
                        ),
                    )
                    _write_code_result(result_path, code_id, code_entry)
                    return 1
                manager_exe = mgrc.executable

                if eval_info.grader_files.stub is not None:
                    solc = compile_solver_with_stub(code_path, graders_dir, work_dir)
                    solver_uses_stdio = True
                elif eval_info.grader_files.grader is not None:
                    solc = compile_solver_with_grader(code_path, graders_dir, work_dir)
                    solver_uses_stdio = False
                else:
                    record_all(
                        "SE",
                        _system_error_payload(
                            eval_type=eval_info.eval_type,
                            message="missing stub.cpp/grader.cpp for communication task",
                        ),
                    )
                    _write_code_result(result_path, code_id, code_entry)
                    return 1

                if not solc.success or not solc.executable:
                    record_all("CE", {"compile_error": solc.message})
                    _write_code_result(result_path, code_id, code_entry)
                    return 0
                solver_exe = solc.executable

            elif eval_info.eval_type == EvaluationType.GRADER_LINKED:
                solc = compile_solver_with_grader(code_path, graders_dir, work_dir)
                if not solc.success or not solc.executable:
                    record_all("CE", {"compile_error": solc.message})
                    _write_code_result(result_path, code_id, code_entry)
                    return 0
                solver_exe = solc.executable

            else:
                record_all(
                    "SE",
                    _system_error_payload(
                        eval_type=eval_info.eval_type,
                        message=f"unsupported eval_type: {eval_info.eval_type.value}",
                    ),
                )
                _write_code_result(result_path, code_id, code_entry)
                return 1
        except Exception as exc:
            record_all(
                "SE",
                _system_error_payload(
                    eval_type=eval_info.eval_type,
                    message=f"setup error: {exc}",
                ),
            )
            _write_code_result(result_path, code_id, code_entry)
            return 1
        for idx, case_path in enumerate(case_paths):
            case_id = _case_key(case_path)

            try:
                case_work = work_dir / f"case_{idx:04d}"
                case_work.mkdir(parents=True, exist_ok=True)
                if eval_info.eval_type == EvaluationType.STDIO_INTERACTIVE:
                    assert interactor_exe is not None and solver_exe is not None
                    rr = run_stdio_interactive(
                        interactor_exe,
                        solver_exe,
                        case_path,
                        case_work,
                        time_limit_s=time_limit_s,
                    )
                elif eval_info.eval_type == EvaluationType.GRADER_COMMUNICATION:
                    assert manager_exe is not None and solver_exe is not None
                    rr = run_grader_communication(
                        manager_exe,
                        solver_exe,
                        case_path,
                        case_work,
                        time_limit_s=time_limit_s,
                        solver_uses_stdio=solver_uses_stdio,
                    )
                elif eval_info.eval_type == EvaluationType.DUAL_PROCESS:
                    assert manager_exe is not None and solver_exe is not None
                    rr = run_dual_process(
                        manager_exe,
                        solver_exe,
                        solver_exe,
                        case_path,
                        case_work,
                        time_limit_s=time_limit_s,
                    )
                else:
                    assert solver_exe is not None
                    rr = run_grader_linked(
                        solver_exe,
                        case_path,
                        case_work,
                        time_limit_s=time_limit_s,
                        file_io=linked_file_io,
                    )
                    _apply_linked_expected_output_check(case_path, rr)

                case_result: dict[str, Any] = {
                    "verdict": rr.verdict,
                    "wall_ms": rr.wall_ms,
                    "cpu_ms": rr.cpu_ms,
                    "max_rss_solution_kb": rr.max_rss_solution_kb,
                    "max_rss_interactor_kb": rr.max_rss_interactor_kb,
                    "exit_codes": rr.exit_codes,
                    "queries": rr.queries,
                    "query_limit": rr.query_limit,
                    "score": rr.score,
                    "stdout_tail": rr.stdout_tail,
                    "stderr_tail": rr.stderr_tail,
                    "eval_type": eval_info.eval_type.value,
                }
                verdict_for_log = rr.verdict
                score_for_log = rr.score
            except Exception as exc:
                case_result = {
                    "verdict": "SE",
                    **_system_error_payload(
                        eval_type=eval_info.eval_type,
                        message=str(exc),
                    ),
                }
                verdict_for_log = "SE"
                score_for_log = None
                had_system_error = True
            code_entry["cases"][case_id] = case_result
            print(f"[judge.py] {case_id}: {verdict_for_log} (score={score_for_log})")

    _write_code_result(result_path, code_id, code_entry)
    print(f"[judge.py] wrote: {result_path}")
    return 1 if had_system_error else 0
