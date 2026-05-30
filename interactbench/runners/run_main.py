#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from interactbench.eval_defaults import BOTH_ADAPTIVE, BOTH_NON, LANGUAGE_TIME_MULTIPLIER
from interactbench.code_layout import (
    get_language_spec_from_path,
    infer_code_id,
    infer_variant_name_from_code_path,
    result_path as build_result_path,
)
from interactbench.gojudge_client import CompileError, GoJudgeClient
from interactbench.result_store import update_result

TESTLIB_DIR = Path("third_party/testlib")
_QUERIES_RE = re.compile(r"queries\s*=\s*(\d+)")
_QUERY_LIMIT_RE = re.compile(r"query_limit\s*=\s*(\d+)")
_DIAGNOSTIC_TAIL_BYTES = 2000
_IDLE_MIN_WALL_NS = 1_000_000_000
_IDLE_CPU_NEAR_ZERO_NS = 100_000_000
_IDLE_CPU_WALL_RATIO = 0.05


def _load_meta(problem_dir: Path) -> dict:
    meta_path = problem_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_interactor_mode(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"adaptive", "non_adaptive", "both"}:
        return key
    return None


def validate_ratio() -> None:
    assert 0 <= BOTH_NON <= 100, f"BOTH_NON must be in [0,100], got {BOTH_NON}"
    assert 0 <= BOTH_ADAPTIVE <= 100, (
        f"BOTH_ADAPTIVE must be in [0,100], got {BOTH_ADAPTIVE}"
    )
    assert BOTH_NON + BOTH_ADAPTIVE <= 100, (
        f"BOTH_NON + BOTH_ADAPTIVE must be <= 100, got {BOTH_NON + BOTH_ADAPTIVE}"
    )


def select_cases(
    problem_dir: Path,
    mode: str,
    cases_glob: str = "cases/*.in",
) -> tuple[list[Path], list[Path]]:
    """
    Return (non_adaptive_cases, adaptive_cases) selected by pool.

    Pool identity is encoded in the numeric case filename:
    001-100 -> non-adaptive pool, 101-200 -> adaptive pool.
    """
    all_cases = sorted(problem_dir.glob(cases_glob))
    by_num: dict[int, Path] = {}
    for cp in all_cases:
        try:
            by_num[int(cp.stem)] = cp
        except ValueError:
            continue

    def pick(lo: int, hi: int) -> list[Path]:
        return [by_num[n] for n in range(lo, hi + 1) if n in by_num]

    if mode == "non_adaptive":
        return (pick(1, 100), [])
    if mode == "adaptive":
        return ([], pick(101, 200))
    if mode == "both":
        non_cases = pick(1, BOTH_NON) if BOTH_NON > 0 else []
        adp_cases = pick(101, 100 + BOTH_ADAPTIVE) if BOTH_ADAPTIVE > 0 else []
        return (non_cases, adp_cases)
    raise ValueError(f"unknown interactor_mode: {mode!r}")


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _apply_qle_verdict(case_result: dict[str, object]) -> None:
    if case_result.get("verdict") != "OK":
        return
    qv = _safe_int(case_result.get("queries"))
    bv = _safe_int(case_result.get("query_limit"))
    if qv is None or bv is None:
        return
    if qv > bv:
        case_result["verdict"] = "QLE"
        case_result["qle"] = True


def _system_error_result(*, interactor_kind: str | None, message: str) -> dict[str, object]:
    result: dict[str, object] = {
        "verdict": "SE",
        "queries": None,
        "query_limit": None,
        "cpu_ms": None,
        "max_rss_solution_kb": None,
        "exit_codes": {"solution": None, "interactor": None},
        "system_error": message,
    }
    if interactor_kind is not None:
        result["interactor_kind"] = interactor_kind
    return result


def _write_code_result(result_path: Path, code_id: str, code_entry: dict[str, object]) -> None:
    update_result(result_path, lambda d: d.__setitem__(str(code_id), code_entry))


def _cleanup_uploaded_files(client: GoJudgeClient, cleanup_ids: list[str]) -> None:
    for fid in cleanup_ids:
        try:
            client.delete_file(fid)
        except Exception as exc:
            print(f"[judge.py] cleanup warning for file {fid}: {exc}", file=sys.stderr)


def _parse_tout(content: str) -> tuple[int | None, int | None]:
    """Parse queries and query_limit from tout.txt content (take last occurrence)."""
    queries = None
    query_limit = None
    for m in _QUERIES_RE.finditer(content):
        queries = int(m.group(1))
    for m in _QUERY_LIMIT_RE.finditer(content):
        query_limit = int(m.group(1))
    return queries, query_limit


def _tail_text(value: object, *, limit: int = _DIAGNOSTIC_TAIL_BYTES) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _safe_ns(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def _is_idle_like_timeout(sol_result: dict[str, object]) -> bool:
    cpu_ns = _safe_ns(sol_result.get("time"))
    wall_ns = _safe_ns(sol_result.get("runTime"))
    cpu_limit_ns = _safe_ns(sol_result.get("cpuLimit"))
    near_zero_ns = _IDLE_CPU_NEAR_ZERO_NS
    if cpu_limit_ns > 0:
        near_zero_ns = min(near_zero_ns, max(1, cpu_limit_ns // 10))
    if cpu_ns <= near_zero_ns:
        return True
    if wall_ns < _IDLE_MIN_WALL_NS:
        return False
    return (cpu_ns / wall_ns) <= _IDLE_CPU_WALL_RATIO


def _map_verdict(
    sol_result: dict[str, object],
    int_result: dict[str, object],
    queries: int | None,
    query_limit: int | None,
) -> str:
    """Map go-judge status to InteractBench verdicts."""
    sol_status = sol_result.get("status", "")
    int_status = int_result.get("status", "")
    int_exit = int_result.get("exitStatus", 0)

    if sol_status == "Time Limit Exceeded":
        if _is_idle_like_timeout(sol_result):
            return "IDLE"
        return "TLE"
    if sol_status == "Memory Limit Exceeded":
        return "MLE"
    if sol_status == "Signalled":
        return "RE"
    if sol_status not in ("Accepted", ""):
        return "RE"

    if int_status not in ("Accepted", ""):
        if int_status == "Time Limit Exceeded":
            if _is_idle_like_timeout(sol_result):
                return "IDLE"
            return "TLE"
        if int_exit == 1:
            return "WA"
        if int_exit == 2:
            return "PE"
        return "RE"

    if int_exit == 0:
        if queries is not None and query_limit is not None and queries > query_limit:
            return "QLE"
        return "OK"
    if int_exit == 1:
        return "WA"
    if int_exit == 2:
        return "PE"
    return "RE"


def run_interactive_case(
    client: GoJudgeClient,
    sol_cmd: dict[str, object],
    int_exe_id: str,
    case_input: str,
    cpu_limit_ns: int,
    mem_limit_bytes: int,
    interactor_kind: str = "non_adaptive",
) -> dict:
    """Run a single interactive case using go-judge pipeMapping."""
    request = {
        "cmd": [
            {
                "args": sol_cmd["args"],
                "env": ["PATH=/usr/bin:/bin"],
                "files": [None, None, {"name": "stderr", "max": 1 << 20}],
                "copyOut": ["stderr"],
                "cpuLimit": cpu_limit_ns,
                "clockLimit": cpu_limit_ns * 2,
                "memoryLimit": mem_limit_bytes,
                "stackLimit": mem_limit_bytes,
                "procLimit": 512,
                "copyIn": sol_cmd["copyIn"],
            },
            {
                "args": ["interactor", "in.txt", "tout.txt"],
                "env": ["PATH=/usr/bin:/bin"],
                "files": [None, None, {"name": "stderr", "max": 1 << 20}],
                "cpuLimit": cpu_limit_ns * 4,
                "clockLimit": cpu_limit_ns * 8,
                "memoryLimit": mem_limit_bytes * 4,
                "stackLimit": mem_limit_bytes * 4,
                "procLimit": 512,
                "copyIn": {
                    "interactor": {"fileId": int_exe_id},
                    "in.txt": {"content": case_input},
                },
                "copyOut": ["stderr", "tout.txt?"],
            },
        ],
        "pipeMapping": [
            {
                "in": {"index": 0, "fd": 1},
                "out": {"index": 1, "fd": 0},
                "proxy": True,
                "name": "pipe_s2i",
                "max": _DIAGNOSTIC_TAIL_BYTES,
            },
            {
                "in": {"index": 1, "fd": 1},
                "out": {"index": 0, "fd": 0},
                "proxy": True,
                "name": "pipe_i2s",
                "max": _DIAGNOSTIC_TAIL_BYTES,
            },
        ],
    }

    results = client.run(request)
    sol_res = results[0]
    int_res = results[1]

    tout_content = int_res.files.get("tout.txt", int_res.files.get("tout.txt?", ""))
    queries, query_limit = _parse_tout(tout_content)

    verdict = _map_verdict(
        {
            "status": sol_res.status,
            "exitStatus": sol_res.exit_status,
            "time": sol_res.time_ns,
            "runTime": sol_res.run_time_ns,
            "cpuLimit": cpu_limit_ns,
        },
        {
            "status": int_res.status,
            "exitStatus": int_res.exit_status,
            "time": int_res.time_ns,
            "runTime": int_res.run_time_ns,
            "cpuLimit": cpu_limit_ns,
        },
        queries,
        query_limit,
    )

    def copied_file(name: str) -> str:
        value = sol_res.files.get(name)
        if value is not None:
            return str(value)
        value = int_res.files.get(name)
        if value is not None:
            return str(value)
        return ""

    result = {
        "verdict": verdict,
        "queries": queries,
        "query_limit": query_limit,
        "cpu_ms": sol_res.time_ns // 1_000_000,
        "max_rss_solution_kb": sol_res.memory_bytes // 1024,
        "interactor_kind": interactor_kind,
        "exit_codes": {
            "solution": sol_res.exit_status,
            "interactor": int_res.exit_status,
        },
    }
    if verdict != "OK":
        result["stderr_solution_tail"] = _tail_text(sol_res.files.get("stderr"))
        result["stderr_interactor_tail"] = _tail_text(int_res.files.get("stderr"))
        result["tout_tail"] = _tail_text(tout_content)
        result["pipe_tail_s2i"] = _tail_text(copied_file("pipe_s2i"))
        result["pipe_tail_i2s"] = _tail_text(copied_file("pipe_i2s"))
        file_errors = [*sol_res.file_error, *int_res.file_error]
        if file_errors:
            result["file_error"] = file_errors
        if sol_res.error or int_res.error:
            result["system_error"] = "; ".join(
                x for x in [sol_res.error, int_res.error] if x
            )

    return result


def _build_sol_cmd(
    client: GoJudgeClient,
    code_path: Path,
    cleanup_ids: list[str],
    *,
    mem_limit_mb: int | None = None,
) -> dict[str, object]:
    """Compile or stage the solver and return go-judge args/copyIn."""
    source = code_path.read_text(encoding="utf-8")
    ext = code_path.suffix.lower()

    if ext == ".cpp":
        exe_id, _ = client.compile_cpp(source)
        cleanup_ids.append(exe_id)
        return {"args": ["a"], "copyIn": {"a": {"fileId": exe_id}}}

    if ext == ".py":
        return {
            "args": ["/usr/bin/python3", "main.py"],
            "copyIn": {"main.py": {"content": source}},
        }

    if ext == ".go":
        exe_id, _ = client.compile_go(source)
        cleanup_ids.append(exe_id)
        return {"args": ["a"], "copyIn": {"a": {"fileId": exe_id}}}

    if ext == ".java":
        class_files, _ = client.compile_java(source)
        cleanup_ids.extend(class_files.values())
        heap_mb = 64
        if mem_limit_mb is not None:
            heap_mb = max(16, min(256, int(mem_limit_mb) // 2))
        return {
            "args": ["/usr/bin/java", f"-Xmx{heap_mb}m", "-cp", "classes", "Main"],
            "copyIn": {
                f"classes/{name}": {"fileId": file_id}
                for name, file_id in class_files.items()
            },
        }

    raise ValueError(f"unsupported solution language: {ext}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Evaluate standard interactive problems")
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--problem-dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--code-id", default=None)
    parser.add_argument("--code-path", default=None)
    parser.add_argument("--interactor-path", default=None)
    parser.add_argument("--adaptive-interactor-path", default=None)
    parser.add_argument("--cases-glob", default="cases/*.in")
    parser.add_argument("--cpu-ms", type=int, default=None)
    parser.add_argument("--mem-mb", type=int, default=None)
    parser.add_argument("--gojudge-url", default="http://127.0.0.1:5050")
    args = parser.parse_args(argv)
    validate_ratio()

    problem_dir = (Path(args.problem_dir) if args.problem_dir else Path("data/problems")) / args.problem_id
    if not problem_dir.exists():
        print(f"[judge.py] problem not found: {problem_dir}", file=sys.stderr)
        return 2

    if not args.code_path:
        print("[judge.py] --code-path is required", file=sys.stderr)
        return 2

    code_path = Path(args.code_path)
    if not code_path.exists():
        print(f"[judge.py] code not found: {code_path}", file=sys.stderr)
        return 2

    variant_name = args.model or infer_variant_name_from_code_path(code_path)
    code_id = args.code_id or infer_code_id(code_path)
    meta = _load_meta(problem_dir)
    interactor_mode = _normalize_interactor_mode(meta.get("interactor_mode")) or "both"
    allow_adaptive = interactor_mode != "non_adaptive"
    lang_key = get_language_spec_from_path(code_path).tag
    non_cases, adp_cases = select_cases(problem_dir, interactor_mode, args.cases_glob)
    selected_case_paths = [*non_cases, *adp_cases]
    all_case_paths = sorted(problem_dir.glob(args.cases_glob))
    if not all_case_paths:
        print(f"[judge.py] no cases matched: {problem_dir / args.cases_glob}", file=sys.stderr)
        return 2
    if not selected_case_paths:
        print(f"[judge.py] no cases selected for mode={interactor_mode}: {problem_dir / args.cases_glob}", file=sys.stderr)
        return 2
    print(
        f"[judge.py] problem={args.problem_id} mode={interactor_mode} "
        f"BOTH_NON={BOTH_NON} BOTH_ADAPTIVE={BOTH_ADAPTIVE} "
        f"running {len(non_cases)} cases (non_adaptive pool) + "
        f"{len(adp_cases)} cases (adaptive pool)"
    )

    result_path = build_result_path(problem_dir, variant_name)
    code_entry: dict[str, object] = {"cases": {}}

    interactor_cpp = Path(args.interactor_path) if args.interactor_path else problem_dir / "interactor" / "non_adaptive.cpp"
    if non_cases and not interactor_cpp.exists():
        print(f"[judge.py] interactor not found: {interactor_cpp}", file=sys.stderr)
        for cp in selected_case_paths:
            code_entry["cases"][cp.stem] = _system_error_result(
                interactor_kind="non_adaptive",
                message=f"interactor not found: {interactor_cpp}",
            )
        _write_code_result(result_path, code_id, code_entry)
        return 1

    adaptive_cpp: Path | None = None
    if adp_cases and args.adaptive_interactor_path:
        adaptive_cpp = Path(args.adaptive_interactor_path)
        if not adaptive_cpp.exists():
            print(f"[judge.py] adaptive interactor not found: {adaptive_cpp}", file=sys.stderr)
            for cp in selected_case_paths:
                code_entry["cases"][cp.stem] = _system_error_result(
                    interactor_kind="adaptive",
                    message=f"adaptive interactor not found: {adaptive_cpp}",
                )
            _write_code_result(result_path, code_id, code_entry)
            return 1
    elif adp_cases and allow_adaptive:
        default_adaptive = problem_dir / "interactor" / "adaptive.cpp"
        if default_adaptive.exists():
            adaptive_cpp = default_adaptive
    if adp_cases and adaptive_cpp is None:
        print("[judge.py] adaptive cases selected but adaptive interactor not found", file=sys.stderr)
        for cp in selected_case_paths:
            code_entry["cases"][cp.stem] = _system_error_result(
                interactor_kind="adaptive",
                message="adaptive interactor not found",
            )
        _write_code_result(result_path, code_id, code_entry)
        return 1

    base_cpu_ms = args.cpu_ms or meta.get("cpu_time_limit_ms") or 2000
    cpu_ms = int(base_cpu_ms) * LANGUAGE_TIME_MULTIPLIER.get(lang_key, 1)
    cpu_limit_ns = int(cpu_ms) * 1_000_000
    mem_limit_mb = args.mem_mb if args.mem_mb is not None else (_safe_int(meta.get("memory_limit_mb")) or 256)
    mem_limit_bytes = int(mem_limit_mb) * 1024 * 1024

    client = GoJudgeClient(args.gojudge_url)
    cleanup_ids: list[str] = []
    had_system_error = False

    try:
        sol_cmd = _build_sol_cmd(client, code_path, cleanup_ids, mem_limit_mb=mem_limit_mb)
    except CompileError as e:
        print(f"[judge.py] compile error: {e}", file=sys.stderr)
        for cp in selected_case_paths:
            code_entry["cases"][cp.stem] = {"verdict": "CE", "compile_error": str(e)}
        _write_code_result(result_path, code_id, code_entry)
        return 0
    except ValueError as e:
        print(f"[judge.py] {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[judge.py] connection/compile error: {e}", file=sys.stderr)
        for cp in selected_case_paths:
            code_entry["cases"][cp.stem] = _system_error_result(
                interactor_kind=None,
                message=str(e),
            )
        _write_code_result(result_path, code_id, code_entry)
        return 1

    if selected_case_paths and not (TESTLIB_DIR / "testlib.h").exists():
        print(f"[judge.py] missing testlib.h at: {TESTLIB_DIR / 'testlib.h'}", file=sys.stderr)
        for cp in selected_case_paths:
            code_entry["cases"][cp.stem] = _system_error_result(
                interactor_kind=None,
                message="missing testlib.h",
            )
        _write_code_result(result_path, code_id, code_entry)
        _cleanup_uploaded_files(client, cleanup_ids)
        return 1

    try:
        testlib_content = (TESTLIB_DIR / "testlib.h").read_text(encoding="utf-8")
    except Exception as e:
        print(f"[judge.py] failed to read testlib.h: {e}", file=sys.stderr)
        for cp in selected_case_paths:
            code_entry["cases"][cp.stem] = _system_error_result(
                interactor_kind=None,
                message=f"failed to read testlib.h: {e}",
            )
        _write_code_result(result_path, code_id, code_entry)
        _cleanup_uploaded_files(client, cleanup_ids)
        return 1

    int_exe_id: str | None = None
    if non_cases:
        try:
            int_source = interactor_cpp.read_text(encoding="utf-8")
            int_exe_id, _ = client.compile_cpp(int_source, extra_files={"testlib.h": testlib_content})
            cleanup_ids.append(int_exe_id)
        except CompileError as e:
            print(f"[judge.py] interactor compile error: {e}", file=sys.stderr)
            for cp in selected_case_paths:
                code_entry["cases"][cp.stem] = _system_error_result(
                    interactor_kind="non_adaptive",
                    message=f"interactor compile: {e}",
                )
            _write_code_result(result_path, code_id, code_entry)
            _cleanup_uploaded_files(client, cleanup_ids)
            return 1
        except Exception as e:
            print(f"[judge.py] interactor setup error: {e}", file=sys.stderr)
            for cp in selected_case_paths:
                code_entry["cases"][cp.stem] = _system_error_result(
                    interactor_kind="non_adaptive",
                    message=f"interactor setup: {e}",
                )
            _write_code_result(result_path, code_id, code_entry)
            _cleanup_uploaded_files(client, cleanup_ids)
            return 1

    int_adp_exe_id: str | None = None
    if adp_cases and adaptive_cpp:
        try:
            adp_source = adaptive_cpp.read_text(encoding="utf-8")
            int_adp_exe_id, _ = client.compile_cpp(adp_source, extra_files={"testlib.h": testlib_content})
            cleanup_ids.append(int_adp_exe_id)
            print(f"[judge.py] compiled adaptive interactor: {adaptive_cpp}")
        except CompileError as e:
            print(f"[judge.py] adaptive interactor compile failed: {e}", file=sys.stderr)
            for cp in selected_case_paths:
                code_entry["cases"][cp.stem] = _system_error_result(
                    interactor_kind="adaptive",
                    message=f"adaptive interactor compile: {e}",
                )
            _write_code_result(result_path, code_id, code_entry)
            _cleanup_uploaded_files(client, cleanup_ids)
            return 1
        except Exception as e:
            print(f"[judge.py] adaptive interactor setup error: {e}", file=sys.stderr)
            for cp in selected_case_paths:
                code_entry["cases"][cp.stem] = _system_error_result(
                    interactor_kind="adaptive",
                    message=f"adaptive interactor setup: {e}",
                )
            _write_code_result(result_path, code_id, code_entry)
            _cleanup_uploaded_files(client, cleanup_ids)
            return 1

    tasks: list[tuple[Path, str, str]] = []
    if non_cases:
        if not int_exe_id:
            print("[judge.py] non_adaptive cases require non_adaptive interactor", file=sys.stderr)
            for cp in non_cases:
                code_entry["cases"][cp.stem] = _system_error_result(
                    interactor_kind="non_adaptive",
                    message="missing non_adaptive interactor",
                )
            _write_code_result(result_path, code_id, code_entry)
            _cleanup_uploaded_files(client, cleanup_ids)
            return 1
        tasks.extend((cp, int_exe_id, "non_adaptive") for cp in non_cases)
    if adp_cases:
        if not int_adp_exe_id:
            print("[judge.py] adaptive cases require adaptive interactor", file=sys.stderr)
            for cp in adp_cases:
                code_entry["cases"][cp.stem] = _system_error_result(
                    interactor_kind="adaptive",
                    message="missing adaptive interactor",
                )
            _write_code_result(result_path, code_id, code_entry)
            _cleanup_uploaded_files(client, cleanup_ids)
            return 1
        tasks.extend((cp, int_adp_exe_id, "adaptive") for cp in adp_cases)

    for idx, (case_path, current_int_exe_id, interactor_kind) in enumerate(tasks):
        case_id = case_path.stem
        try:
            case_input = case_path.read_text(encoding="utf-8")
            result = run_interactive_case(
                client, sol_cmd, current_int_exe_id, case_input,
                cpu_limit_ns, mem_limit_bytes, interactor_kind
            )
            _apply_qle_verdict(result)
        except Exception as e:
            result = _system_error_result(interactor_kind=interactor_kind, message=str(e))
            had_system_error = True
        code_entry["cases"][case_id] = result
        print(f"[judge.py] {case_id}: {result.get('verdict')} (queries={result.get('queries')})")

    _write_code_result(result_path, code_id, code_entry)
    _cleanup_uploaded_files(client, cleanup_ids)
    print(f"[judge.py] wrote: {result_path}")
    return 1 if had_system_error else 0
