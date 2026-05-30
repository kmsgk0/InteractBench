"""IOI evaluation adapter with feature-based type detection."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class EvaluationType(Enum):
    GRADER_COMMUNICATION = "grader_communication"
    GRADER_LINKED = "grader_linked"
    STDIO_INTERACTIVE = "stdio_interactive"
    DUAL_PROCESS = "dual_process"


@dataclass
class GraderFiles:
    manager: Path | None = None
    stub: Path | None = None
    header: Path | None = None
    grader: Path | None = None
    testlib: Path | None = None


@dataclass
class EvaluationInfo:
    eval_type: EvaluationType
    grader_files: GraderFiles | None = None
    num_processes: int = 1


def find_graders_dir(task_dir: Path) -> Path | None:
    """Return the first grader directory found under a known task layout."""
    candidates = [
        task_dir / "interactor",
        task_dir / "graders",
        task_dir / "grader" / "cpp",
        task_dir / "grader",
    ]
    for d in candidates:
        if d.exists() and d.is_dir():
            return d
    return None


def find_file_by_pattern(directory: Path, pattern: str, exclude: str | None = None) -> Path | None:
    """Find first file matching pattern in directory."""
    if not directory.exists():
        return None
    import fnmatch
    for f in directory.iterdir():
        if f.is_file() and fnmatch.fnmatch(f.name, pattern):
            if exclude and fnmatch.fnmatch(f.name, exclude):
                continue
            return f
    return None


def load_problem_json(task_dir: Path) -> dict[str, Any] | None:
    """Load problem.json if exists."""
    problem_json = task_dir / "problem.json"
    if problem_json.exists():
        try:
            return json.loads(problem_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def parse_task_type_params(params_str: str) -> dict[str, Any]:
    """Parse task_type_params JSON string."""
    if not params_str:
        return {}
    try:
        return json.loads(params_str)
    except json.JSONDecodeError:
        return {}


def detect_grader_files(graders_dir: Path) -> GraderFiles:
    """Detect available grader files in directory."""
    files = GraderFiles()
    if not graders_dir or not graders_dir.exists():
        return files

    files.manager = find_file_by_pattern(graders_dir, "manager.cpp")
    files.stub = find_file_by_pattern(graders_dir, "stub.cpp")
    files.header = find_file_by_pattern(graders_dir, "*.h", exclude="testlib.h")
    files.grader = find_file_by_pattern(graders_dir, "grader.cpp")
    files.testlib = find_file_by_pattern(graders_dir, "testlib.h")

    return files


def detect_evaluation_type(task_dir: Path) -> EvaluationInfo:
    """Detect the evaluation mode from task files."""
    config = load_problem_json(task_dir)
    if config:
        task_type = config.get("task_type") or config.get("type")
        if task_type == "Communication":
            params_str = config.get("task_type_params", "")
            params = parse_task_type_params(params_str)
            num_processes = params.get(
                "task_type_parameters_Communication_num_processes", 1
            )

            if num_processes == 2:
                graders_dir = find_graders_dir(task_dir)
                grader_files = detect_grader_files(graders_dir) if graders_dir else None
                return EvaluationInfo(
                    eval_type=EvaluationType.DUAL_PROCESS,
                    grader_files=grader_files,
                    num_processes=2,
                )

    graders_dir = find_graders_dir(task_dir)
    if graders_dir:
        grader_files = detect_grader_files(graders_dir)

        if grader_files.manager and grader_files.stub:
            return EvaluationInfo(
                eval_type=EvaluationType.GRADER_COMMUNICATION,
                grader_files=grader_files,
            )

        if grader_files.manager and grader_files.grader:
            return EvaluationInfo(
                eval_type=EvaluationType.GRADER_COMMUNICATION,
                grader_files=grader_files,
            )

        if grader_files.grader or grader_files.header:
            return EvaluationInfo(
                eval_type=EvaluationType.GRADER_LINKED,
                grader_files=grader_files,
            )

    return EvaluationInfo(eval_type=EvaluationType.STDIO_INTERACTIVE)


@dataclass
class CompileResult:
    success: bool
    message: str
    executable: Path | None = None


def compile_manager(grader_dir: Path, work_dir: Path) -> CompileResult:
    """Compile manager.cpp for grader_communication type."""
    manager_src = grader_dir / "manager.cpp"
    if not manager_src.exists():
        return CompileResult(False, "manager.cpp not found")

    manager_exe = work_dir / "manager"
    cmd = ["g++", "-std=c++17", "-O2", "-o", str(manager_exe), str(manager_src)]

    cmd.extend(["-I", str(grader_dir)])
    cmd.extend(["-I", str(_default_testlib_dir())])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return CompileResult(False, f"Compile error: {result.stderr}")

    return CompileResult(True, "OK", manager_exe)


def compile_solver_with_stub(
    solver_src: Path,
    grader_dir: Path,
    work_dir: Path,
) -> CompileResult:
    """Compile solver.cpp with stub.cpp for grader_communication type."""
    stub_src = grader_dir / "stub.cpp"
    if not stub_src.exists():
        return CompileResult(False, "stub.cpp not found")

    solver_exe = work_dir / "solver"
    cmd = [
        "g++", "-std=c++17", "-O2",
        "-DONLINE_JUDGE",
        "-o", str(solver_exe),
        str(solver_src), str(stub_src),
        "-I", str(grader_dir),
    ]
    cmd.extend(["-I", str(_default_testlib_dir())])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return CompileResult(False, f"Compile error: {result.stderr}")

    return CompileResult(True, "OK", solver_exe)


@dataclass
class RunResult:
    verdict: str
    wall_ms: int
    cpu_ms: int | None = None
    max_rss_solution_kb: int | None = None
    max_rss_interactor_kb: int | None = None
    exit_codes: dict | None = None
    queries: int | None = None
    query_limit: int | None = None
    score: float | None = None
    stdout_text: str | None = None
    stdout_tail: str | None = None
    stderr_tail: str | None = None


def _tail(text: str, *, limit: int = 2000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _read_vmrss_kb(pid: int) -> int | None:
    """Return VmRSS in KB for a running process when /proc is available."""
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def _parse_float_token(token: str) -> float | None:
    token = (token or "").strip()
    if not token:
        return None
    if token.endswith("%"):
        try:
            value = float(token[:-1]) / 100.0
        except ValueError:
            return None
    else:
        try:
            value = float(token)
        except ValueError:
            return None
    if value < 0.0 or value > 1.0:
        return None
    return value


def _score_from_ok_line(line: str) -> float:
    tokens = line.split()
    if len(tokens) <= 1:
        return 1.0

    tail_tokens = {tok.lower() for tok in tokens[3:]}
    if len(tokens) >= 4 and tokens[1].isdigit() and tokens[2].lower() == "cells" and "not" in tail_tokens:
        return 1.0 if int(tokens[1]) == 0 else 0.0

    first_score = _parse_float_token(tokens[1])
    if first_score is not None:
        return first_score

    return 0.0


def _interpret_verdict_and_score(stdout: str) -> tuple[str | None, float | None]:
    """Parse checker or manager output into a verdict and optional score."""
    stdout = stdout or ""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None, None

    for ln in lines:
        up = ln.strip().upper()
        if up.startswith("OK") or up.startswith("ACCEPTED"):
            points = _score_from_ok_line(ln)
            if points >= 0.9999:
                return "OK", points
            return "WA", points
        if up == "AC":
            return "OK", 1.0
        if up == "CORRECT":
            return "OK", 1.0
        if up == "INCORRECT":
            return "WA", 0.0
        if up == "NO" or up in {"WA", "WRONG ANSWER"} or "WRONG ANSWER" in up:
            return "WA", 0.0
        if up in {"PE"} or "PROTOCOL" in up:
            return "PE", 0.0
        if up in {"SV"} or "SECURITY" in up:
            return "PE", 0.0

    tokens = lines[0].split()
    if len(tokens) == 1:
        points = _parse_float_token(tokens[0])
        if points is not None:
            if points >= 0.9999:
                return "OK", points
            return "WA", points

    return None, None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_testlib_dir() -> Path:
    return _repo_root() / "third_party" / "testlib"


def compile_stdio_interactor(
    interactor_src: Path,
    work_dir: Path,
    *,
    testlib_dir: Path | None = None,
) -> CompileResult:
    """Compile a stdio interactor (testlib-based) into `work_dir/interactor`."""
    testlib_dir = Path(testlib_dir) if testlib_dir is not None else _default_testlib_dir()
    if not (testlib_dir / "testlib.h").exists():
        return CompileResult(False, f"testlib.h not found under: {testlib_dir}")

    interactor_exe = work_dir / "interactor"
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        "-pipe",
        "-o",
        str(interactor_exe),
        str(interactor_src),
        "-I",
        str(testlib_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return CompileResult(False, f"Compile error: {result.stderr}")
    return CompileResult(True, "OK", interactor_exe)


def compile_stdio_solver(solver_src: Path, work_dir: Path) -> CompileResult:
    """Compile a stdio solver into `work_dir/solver`."""
    solver_exe = work_dir / "solver"
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        "-pipe",
        "-o",
        str(solver_exe),
        str(solver_src),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return CompileResult(False, f"Compile error: {result.stderr}")
    return CompileResult(True, "OK", solver_exe)


def run_stdio_interactive(
    interactor_exe: Path,
    solver_exe: Path,
    input_file: Path,
    work_dir: Path,
    *,
    time_limit_s: float = 2.0,
    wall_time_limit_s: float | None = None,
    idle_time_limit_s: float | None = None,
) -> RunResult:
    """Run a stdio-interactive task."""
    from interactbench.local_interactive import InteractiveConfig, run_interactive_case

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_file = Path(input_file).resolve()

    wall_limit = float(wall_time_limit_s) if wall_time_limit_s is not None else float(time_limit_s) * 2.0
    idle_limit = idle_time_limit_s
    if idle_limit is None:
        idle_limit = min(5.0, wall_limit) if wall_limit > 0 else 5.0

    cfg = InteractiveConfig(
        cpu_time_limit_s=float(time_limit_s),
        wall_time_limit_s=wall_limit,
        idle_time_limit_s=float(idle_limit) if idle_limit is not None else None,
        rlimits=None,
    )

    result = run_interactive_case(
        solution_cmd=[str(solver_exe)],
        interactor_cmd=[str(interactor_exe), str(input_file), "tout.txt"],
        work_dir=work_dir,
        tout_path=work_dir / "tout.txt",
        stderr_solution_path=work_dir / "stderr_solution.txt",
        stderr_interactor_path=work_dir / "stderr_interactor.txt",
        config=cfg,
    )

    verdict = str(result.get("verdict", "RE"))
    score = 1.0 if verdict == "OK" else 0.0
    return RunResult(
        verdict=verdict,
        wall_ms=int(result.get("wall_ms") or 0),
        cpu_ms=result.get("cpu_ms"),
        max_rss_solution_kb=result.get("max_rss_solution_kb"),
        max_rss_interactor_kb=result.get("max_rss_interactor_kb"),
        exit_codes=result.get("exit_codes"),
        queries=result.get("queries"),
        query_limit=result.get("query_limit"),
        score=score,
        stdout_text=None,
    )


def run_grader_communication(
    manager_exe: Path,
    solver_exe: Path,
    input_file: Path,
    work_dir: Path,
    time_limit_s: float = 2.0,
    *,
    solver_uses_stdio: bool = True,
    solver_args: list[str] | None = None,
) -> RunResult:
    """Run a communication task with manager and solver FIFOs."""
    fifo_s2m = work_dir / "sol_to_mgr"
    fifo_m2s = work_dir / "mgr_to_sol"

    for fifo in [fifo_s2m, fifo_m2s]:
        if fifo.exists():
            fifo.unlink()
        os.mkfifo(fifo)

    wall_start = time.monotonic()
    manager_proc = None
    solver_proc = None
    solver_stdin = None
    solver_stdout = None
    timed_out = False
    manager_out = ""
    manager_err = ""
    solver_out = ""
    solver_err = ""
    max_rss: dict[str, int] = {}
    rss_stop = threading.Event()
    rss_thread: threading.Thread | None = None

    def poll_rss() -> None:
        interval_s = 0.02
        while not rss_stop.is_set():
            if solver_proc is not None and solver_proc.poll() is None:
                rss = _read_vmrss_kb(solver_proc.pid)
                if rss is not None:
                    prev = max_rss.get("solver")
                    max_rss["solver"] = rss if prev is None else max(prev, rss)
            if manager_proc is not None and manager_proc.poll() is None:
                rss = _read_vmrss_kb(manager_proc.pid)
                if rss is not None:
                    prev = max_rss.get("manager")
                    max_rss["manager"] = rss if prev is None else max(prev, rss)
            time.sleep(interval_s)

    try:
        def start_manager():
            nonlocal manager_proc
            with open(input_file, "r") as inf:
                manager_proc = subprocess.Popen(
                    [str(manager_exe), str(fifo_s2m), str(fifo_m2s)],
                    stdin=inf,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(work_dir),
                    text=True,
                    errors="replace",
                )

        def start_solver():
            nonlocal solver_proc, solver_stdin, solver_stdout
            if solver_uses_stdio:
                solver_stdin = open(fifo_m2s, "r")
                solver_stdout = open(fifo_s2m, "w")
                argv = [str(solver_exe)]
                if solver_args:
                    argv.extend(solver_args)
                elif solver_args is None:
                    argv.extend([str(fifo_m2s), str(fifo_s2m)])
                solver_proc = subprocess.Popen(
                    argv,
                    stdin=solver_stdin,
                    stdout=solver_stdout,
                    stderr=subprocess.PIPE,
                    cwd=str(work_dir),
                    text=True,
                    errors="replace",
                )
                return

            argv = [str(solver_exe)]
            if solver_args is not None:
                argv.extend(solver_args)
            else:
                argv.extend([str(fifo_m2s), str(fifo_s2m)])
            solver_proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(work_dir),
                text=True,
                errors="replace",
            )

        mgr_thread = threading.Thread(target=start_manager)
        sol_thread = threading.Thread(target=start_solver)
        mgr_thread.start()
        sol_thread.start()
        rss_thread = threading.Thread(target=poll_rss, daemon=True)
        rss_thread.start()
        mgr_thread.join(timeout=time_limit_s)
        sol_thread.join(timeout=time_limit_s)

        wall_limit = max(0.1, float(time_limit_s) * 2.0)
        deadline = wall_start + wall_limit

        def wait_until_deadline(proc: subprocess.Popen) -> None:
            nonlocal timed_out
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                return
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True

        if solver_proc is not None:
            wait_until_deadline(solver_proc)
        else:
            timed_out = True
        if manager_proc is not None:
            wait_until_deadline(manager_proc)
        else:
            timed_out = True

    finally:
        rss_stop.set()
        if rss_thread is not None:
            try:
                rss_thread.join(timeout=0.2)
            except Exception:
                pass

        for fh in [solver_stdin, solver_stdout]:
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass

        if solver_proc is not None:
            try:
                out_s, err_s = solver_proc.communicate(timeout=0.2)
                solver_out = out_s or ""
                solver_err = err_s or ""
            except Exception:
                pass
        if manager_proc is not None:
            try:
                out_m, err_m = manager_proc.communicate(timeout=0.2)
                manager_out = out_m or ""
                manager_err = err_m or ""
            except Exception:
                pass

        for proc in [solver_proc, manager_proc]:
            if proc and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    pass

        for fifo in [fifo_m2s, fifo_s2m]:
            try:
                if fifo.exists():
                    fifo.unlink()
            except Exception:
                pass

    wall_ms = int((time.monotonic() - wall_start) * 1000)

    manager_exit = manager_proc.returncode if manager_proc else -1
    solver_exit = solver_proc.returncode if solver_proc else -1

    score: float | None = None
    parsed_verdict, parsed_score = _interpret_verdict_and_score(manager_out)
    manager_err_verdict, manager_err_score = _interpret_verdict_and_score(manager_err)
    if manager_err_verdict is not None and (parsed_verdict is None or manager_err_verdict != "OK"):
        parsed_verdict, parsed_score = manager_err_verdict, manager_err_score

    if parsed_verdict is None:
        parsed_verdict, parsed_score = _interpret_verdict_and_score(solver_out)
    if parsed_verdict is None:
        parsed_verdict, parsed_score = _interpret_verdict_and_score(solver_err)

    if parsed_verdict is not None:
        verdict = parsed_verdict
        score = parsed_score
        if verdict == "OK" and score is not None and score < 0.9999:
            verdict = "WA"
    elif timed_out:
        verdict = "TLE"
    else:
        if manager_exit == 0:
            verdict = "OK"
        elif manager_exit == 1:
            verdict = "WA"
        elif manager_exit == 2:
            verdict = "PE"
        else:
            verdict = "RE"

    return RunResult(
        verdict=verdict,
        wall_ms=wall_ms,
        score=score,
        max_rss_solution_kb=max_rss.get("solver"),
        max_rss_interactor_kb=max_rss.get("manager"),
        exit_codes={"manager": manager_exit, "solver": solver_exit},
        stdout_text=manager_out,
        stdout_tail=_tail(manager_out),
        stderr_tail=_tail(manager_err or solver_err),
    )


def compile_solver_with_grader(
    solver_src: Path,
    grader_dir: Path,
    work_dir: Path,
) -> CompileResult:
    """Compile solver.cpp with grader.cpp for grader_linked type."""
    grader_src = grader_dir / "grader.cpp"
    extra_c_sources: list[Path] = []
    if not grader_src.exists():
        grader_src = grader_dir / "grader.c"
        if not grader_src.exists():
            return CompileResult(False, "grader.cpp/grader.c not found")
        included_c_sources: set[str] = set()
        try:
            for raw_line in grader_src.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if line.startswith('#include "') and line.endswith('.c"'):
                    included_c_sources.add(line[len('#include "') : -1])
        except OSError:
            pass
        for extra_src in sorted(grader_dir.glob("grader*.c")):
            if extra_src.name == grader_src.name or extra_src.name in included_c_sources:
                continue
            extra_c_sources.append(extra_src)

    solver_exe = work_dir / "solver"
    cmd = ["g++", "-std=c++17", "-O2", "-DONLINE_JUDGE", "-o", str(solver_exe), str(solver_src)]
    if grader_src.suffix == ".c":
        cmd.extend(["-x", "c++", str(grader_src)])
        for extra_src in extra_c_sources:
            cmd.extend(["-x", "c++", str(extra_src)])
    else:
        cmd.append(str(grader_src))
    cmd.extend(["-I", str(grader_dir)])
    cmd.extend(["-I", str(_default_testlib_dir())])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return CompileResult(False, f"Compile error: {result.stderr}")

    return CompileResult(True, "OK", solver_exe)


def run_grader_linked(
    solver_exe: Path,
    input_file: Path,
    work_dir: Path,
    time_limit_s: float = 2.0,
    *,
    file_io: tuple[str, str] | None = None,
) -> RunResult:
    """Run a linked grader task, optionally staging fixed file-IO names."""
    wall_start = time.monotonic()
    timed_out = False
    stdout = ""
    stderr = ""
    exit_code = -1
    max_rss_solution_kb: int | None = None

    def popen_and_track(*, stdin: Any) -> tuple[str, str, int, bool]:
        nonlocal max_rss_solution_kb

        proc = subprocess.Popen(
            [str(solver_exe)],
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            cwd=str(work_dir),
        )

        rss_stop = threading.Event()
        rss_val: int | None = None

        def poll_rss() -> None:
            nonlocal rss_val
            interval_s = 0.02
            while not rss_stop.is_set():
                rss = _read_vmrss_kb(proc.pid)
                if rss is not None:
                    rss_val = rss if rss_val is None else max(rss_val, rss)
                time.sleep(interval_s)

        rss_thread = threading.Thread(target=poll_rss, daemon=True)
        rss_thread.start()

        out = ""
        err = ""
        local_timed_out = False
        try:
            out, err = proc.communicate(timeout=time_limit_s * 2)
        except subprocess.TimeoutExpired:
            local_timed_out = True
            proc.kill()
            try:
                out, err = proc.communicate(timeout=0.2)
            except Exception:
                pass
        finally:
            rss_stop.set()
            try:
                rss_thread.join(timeout=0.2)
            except Exception:
                pass

        if rss_val is not None:
            max_rss_solution_kb = rss_val
        rc = proc.returncode if proc.returncode is not None else -1
        return (out or "", err or "", rc, local_timed_out)

    if file_io:
        in_name, out_name = file_io
        shutil.copy(input_file, work_dir / in_name)
        out, err, rc, local_timed_out = popen_and_track(stdin=subprocess.DEVNULL)
        timed_out = timed_out or local_timed_out
        stderr = err
        exit_code = rc
        out_path = work_dir / out_name
        if out_path.exists():
            stdout = out_path.read_text(errors="replace")
        else:
            stdout = out
    else:
        with open(input_file, "r") as inf:
            out, err, rc, local_timed_out = popen_and_track(stdin=inf)
            timed_out = timed_out or local_timed_out
            stdout = out
            stderr = err
            exit_code = rc

    wall_ms = int((time.monotonic() - wall_start) * 1000)

    if timed_out:
        return RunResult(
            verdict="TLE",
            wall_ms=wall_ms,
            max_rss_solution_kb=max_rss_solution_kb,
        )

    parsed_verdict, parsed_score = _interpret_verdict_and_score(stdout)
    verdict = parsed_verdict or ("RE" if exit_code != 0 else "WA")
    score = parsed_score
    if verdict == "OK" and score is not None and score < 0.9999:
        verdict = "WA"

    return RunResult(
        verdict=verdict,
        wall_ms=wall_ms,
        score=score,
        max_rss_solution_kb=max_rss_solution_kb,
        exit_codes={"solver": exit_code},
        stdout_text=stdout,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )


def run_dual_process(
    manager_exe: Path,
    solver0_exe: Path,
    solver1_exe: Path,
    input_file: Path,
    work_dir: Path,
    time_limit_s: float = 2.0,
) -> RunResult:
    """Run a two-solver communication task."""
    fifo_s0_to_mgr = work_dir / "sol0_to_mgr"
    fifo_mgr_to_s0 = work_dir / "mgr_to_sol0"
    fifo_s1_to_mgr = work_dir / "sol1_to_mgr"
    fifo_mgr_to_s1 = work_dir / "mgr_to_sol1"

    fifos = [fifo_s0_to_mgr, fifo_mgr_to_s0, fifo_s1_to_mgr, fifo_mgr_to_s1]
    for fifo in fifos:
        if fifo.exists():
            fifo.unlink()
        os.mkfifo(fifo)

    wall_start = time.monotonic()
    manager_proc = None
    solver0_proc = None
    solver1_proc = None
    fhs = []
    timed_out = False
    manager_out = ""
    manager_err = ""
    max_rss: dict[str, int] = {}
    rss_stop = threading.Event()
    rss_thread: threading.Thread | None = None

    def poll_rss() -> None:
        interval_s = 0.02
        while not rss_stop.is_set():
            if solver0_proc is not None and solver0_proc.poll() is None:
                rss = _read_vmrss_kb(solver0_proc.pid)
                if rss is not None:
                    prev = max_rss.get("solver0")
                    max_rss["solver0"] = rss if prev is None else max(prev, rss)
            if solver1_proc is not None and solver1_proc.poll() is None:
                rss = _read_vmrss_kb(solver1_proc.pid)
                if rss is not None:
                    prev = max_rss.get("solver1")
                    max_rss["solver1"] = rss if prev is None else max(prev, rss)
            if manager_proc is not None and manager_proc.poll() is None:
                rss = _read_vmrss_kb(manager_proc.pid)
                if rss is not None:
                    prev = max_rss.get("manager")
                    max_rss["manager"] = rss if prev is None else max(prev, rss)
            time.sleep(interval_s)

    try:
        def start_manager():
            nonlocal manager_proc
            with open(input_file, "r") as inf:
                manager_proc = subprocess.Popen(
                    [
                        str(manager_exe),
                        str(fifo_s0_to_mgr), str(fifo_mgr_to_s0),
                        str(fifo_s1_to_mgr), str(fifo_mgr_to_s1),
                    ],
                    stdin=inf,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(work_dir),
                    text=True,
                    errors="replace",
                )

        def start_solver0():
            nonlocal solver0_proc
            stdin_fh = open(fifo_mgr_to_s0, "r")
            stdout_fh = open(fifo_s0_to_mgr, "w")
            fhs.extend([stdin_fh, stdout_fh])
            solver0_proc = subprocess.Popen(
                [str(solver0_exe), "0"],
                stdin=stdin_fh, stdout=stdout_fh,
                stderr=subprocess.PIPE, cwd=str(work_dir),
                text=True, errors="replace",
            )

        def start_solver1():
            nonlocal solver1_proc
            stdin_fh = open(fifo_mgr_to_s1, "r")
            stdout_fh = open(fifo_s1_to_mgr, "w")
            fhs.extend([stdin_fh, stdout_fh])
            solver1_proc = subprocess.Popen(
                [str(solver1_exe), "1"],
                stdin=stdin_fh, stdout=stdout_fh,
                stderr=subprocess.PIPE, cwd=str(work_dir),
                text=True, errors="replace",
            )

        threads = [
            threading.Thread(target=start_manager),
            threading.Thread(target=start_solver0),
            threading.Thread(target=start_solver1),
        ]
        for t in threads:
            t.start()
        rss_thread = threading.Thread(target=poll_rss, daemon=True)
        rss_thread.start()
        for t in threads:
            t.join(timeout=time_limit_s)

        wall_limit = max(0.1, float(time_limit_s) * 2.0)
        for proc in [solver0_proc, solver1_proc]:
            if proc:
                try:
                    proc.wait(timeout=wall_limit)
                except subprocess.TimeoutExpired:
                    timed_out = True
        if manager_proc:
            try:
                manager_proc.wait(timeout=wall_limit)
            except subprocess.TimeoutExpired:
                timed_out = True

    finally:
        rss_stop.set()
        if rss_thread is not None:
            try:
                rss_thread.join(timeout=0.2)
            except Exception:
                pass

        for fh in fhs:
            try:
                fh.close()
            except Exception:
                pass
        if manager_proc is not None:
            try:
                out_m, err_m = manager_proc.communicate(timeout=0.2)
                manager_out = out_m or ""
                manager_err = err_m or ""
            except Exception:
                pass
        for proc in [solver0_proc, solver1_proc, manager_proc]:
            if proc and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
        for fifo in fifos:
            try:
                if fifo.exists():
                    fifo.unlink()
            except Exception:
                pass

    wall_ms = int((time.monotonic() - wall_start) * 1000)
    manager_exit = manager_proc.returncode if manager_proc else -1

    score: float | None = None
    if timed_out:
        verdict = "TLE"
    else:
        parsed_verdict, parsed_score = _interpret_verdict_and_score(manager_out)
        if parsed_verdict is not None:
            verdict = parsed_verdict
            score = parsed_score
            if verdict == "OK" and score is not None and score < 0.9999:
                verdict = "WA"
        else:
            if manager_exit == 0:
                verdict = "OK"
            elif manager_exit == 1:
                verdict = "WA"
            elif manager_exit == 2:
                verdict = "PE"
            else:
                verdict = "RE"

    return RunResult(
        verdict=verdict,
        wall_ms=wall_ms,
        score=score,
        max_rss_solution_kb=max(
            max_rss.get("solver0") or 0,
            max_rss.get("solver1") or 0,
        )
        or None,
        max_rss_interactor_kb=max_rss.get("manager"),
        exit_codes={
            "manager": manager_exit,
            "solver0": solver0_proc.returncode if solver0_proc else -1,
            "solver1": solver1_proc.returncode if solver1_proc else -1,
        },
        stdout_text=manager_out,
        stdout_tail=_tail(manager_out),
        stderr_tail=_tail(manager_err),
    )
