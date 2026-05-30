from __future__ import annotations

import os
import re
import resource
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_QUERIES_RE = re.compile(r"queries\s*=\s*(\d+)")
_QUERY_LIMIT_RE = re.compile(r"query_limit\s*=\s*(\d+)")
_CLK_TCK_FALLBACK = 100
_DEFAULT_STACK_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class ResourceLimits:
    cpu_seconds: int | None = None
    address_space_bytes: int | None = None


@dataclass(frozen=True)
class InteractiveConfig:
    cpu_time_limit_s: float | None = None
    wall_time_limit_s: float = 5.0
    idle_time_limit_s: float | None = None
    rlimits: ResourceLimits | None = None
    pipe_mode: str = "pump"


def _clk_tck() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except Exception:
        return _CLK_TCK_FALLBACK


def _read_proc_cpu_ticks(pid: int) -> int | None:
    """Return utime+stime clock ticks from /proc/<pid>/stat."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore").strip()
    except FileNotFoundError:
        return None
    rparen = stat.rfind(")")
    if rparen < 0:
        return None
    after = stat[rparen + 2 :]
    fields = after.split()
    if len(fields) <= 12:
        return None
    try:
        utime = int(fields[11])
        stime = int(fields[12])
    except ValueError:
        return None
    return utime + stime


def _read_vmrss_kb(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def _make_preexec_fn(limits: ResourceLimits | None) -> Any:
    def _fn() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:
            pass

        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
            target = _DEFAULT_STACK_BYTES
            hard_cap = target if hard == resource.RLIM_INFINITY else min(int(hard), target)
            if soft != resource.RLIM_INFINITY and int(soft) < hard_cap:
                resource.setrlimit(resource.RLIMIT_STACK, (hard_cap, hard))
        except Exception:
            pass

        if limits is None:
            return

        if limits.cpu_seconds is not None:
            cpu = int(limits.cpu_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        if limits.address_space_bytes is not None:
            mem = int(limits.address_space_bytes)
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    return _fn


def _kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _parse_tout_metrics(tout_path: Path) -> dict[str, int | None]:
    """Parse queries and query_limit from tout.txt."""
    try:
        text = Path(tout_path).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return {"queries": None, "query_limit": None}

    def _last_int(pattern: re.Pattern[str]) -> int | None:
        matches = pattern.findall(text)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except Exception:
            return None

    return {
        "queries": _last_int(_QUERIES_RE),
        "query_limit": _last_int(_QUERY_LIMIT_RE),
    }


def _read_tail_text(path: Path, *, limit_bytes: int) -> str:
    """Read the last bytes of a UTF-8 text file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - int(limit_bytes))
            f.seek(start, os.SEEK_SET)
            data = f.read()
    except Exception:
        return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def run_interactive_case(
    *,
    solution_cmd: list[str],
    interactor_cmd: list[str],
    work_dir: Path,
    solution_work_dir: Path | None = None,
    interactor_work_dir: Path | None = None,
    tout_path: Path,
    stderr_solution_path: Path,
    stderr_interactor_path: Path,
    config: InteractiveConfig,
) -> dict[str, Any]:
    """Run one local interactive case and return a result dict."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    sol_cwd = Path(solution_work_dir) if solution_work_dir is not None else work_dir
    int_cwd = Path(interactor_work_dir) if interactor_work_dir is not None else work_dir
    sol_cwd.mkdir(parents=True, exist_ok=True)
    int_cwd.mkdir(parents=True, exist_ok=True)

    tout_path = Path(tout_path)
    if not tout_path.is_absolute():
        tout_path = int_cwd / tout_path
    tout_path.write_text("", encoding="utf-8")

    pipe_mode = (config.pipe_mode or "").strip().lower()
    if pipe_mode == "direct":
        return _run_interactive_case_direct(
            solution_cmd=solution_cmd,
            interactor_cmd=interactor_cmd,
            sol_cwd=sol_cwd,
            int_cwd=int_cwd,
            tout_path=tout_path,
            stderr_solution_path=stderr_solution_path,
            stderr_interactor_path=stderr_interactor_path,
            config=config,
        )

    clk_tck = _clk_tck()
    wall_start = time.monotonic()

    lock = threading.Lock()
    last_activity = {"t": wall_start}
    bytes_forwarded = {"s2i": 0, "i2s": 0}
    _TAIL_LIMIT = 2000
    pipe_tail: dict[str, bytearray] = {"s2i": bytearray(), "i2s": bytearray()}

    def touch(direction: str, n: int) -> None:
        now = time.monotonic()
        with lock:
            last_activity["t"] = now
            bytes_forwarded[direction] += n

    def _append_tail(direction: str, data: bytes) -> None:
        if not data:
            return
        with lock:
            buf = pipe_tail[direction]
            buf.extend(data)
            if len(buf) > _TAIL_LIMIT:
                del buf[:-_TAIL_LIMIT]

    def pump(src: Any, dst: Any, direction: str) -> None:
        try:
            while True:
                data = src.read(4096)
                if not data:
                    break
                try:
                    dst.write(data)
                    dst.flush()
                except BrokenPipeError:
                    break
                _append_tail(direction, data)
                touch(direction, len(data))
        finally:
            try:
                dst.close()
            except Exception:
                pass

    preexec = _make_preexec_fn(config.rlimits)
    cpu_ticks_start_solution: int | None = None
    cpu_ticks_last_solution: int | None = None
    cpu_ticks_prev_solution: int | None = None
    last_cpu_activity_t: float | None = None

    stderr_solution_path = Path(stderr_solution_path)
    stderr_interactor_path = Path(stderr_interactor_path)
    if not stderr_solution_path.is_absolute():
        stderr_solution_path = sol_cwd / stderr_solution_path
    if not stderr_interactor_path.is_absolute():
        stderr_interactor_path = int_cwd / stderr_interactor_path

    with open(stderr_solution_path, "wb") as sol_err, open(stderr_interactor_path, "wb") as int_err:
        interactor = subprocess.Popen(
            interactor_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=int_err,
            cwd=str(int_cwd),
            bufsize=0,
            start_new_session=True,
            preexec_fn=preexec,
        )
        solution = subprocess.Popen(
            solution_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sol_err,
            cwd=str(sol_cwd),
            bufsize=0,
            start_new_session=True,
            preexec_fn=preexec,
        )

        assert interactor.stdin and interactor.stdout and solution.stdin and solution.stdout

        cpu_ticks_start_solution = _read_proc_cpu_ticks(solution.pid)
        cpu_ticks_last_solution = cpu_ticks_start_solution
        cpu_ticks_prev_solution = cpu_ticks_start_solution
        if cpu_ticks_start_solution is not None:
            last_cpu_activity_t = wall_start

        t1 = threading.Thread(
            target=pump, args=(solution.stdout, interactor.stdin, "s2i"), daemon=True
        )
        t2 = threading.Thread(
            target=pump, args=(interactor.stdout, solution.stdin, "i2s"), daemon=True
        )
        t1.start()
        t2.start()

        max_rss_solution_kb = 0
        max_rss_interactor_kb = 0
        timeout_reason: str | None = None

        while True:
            now = time.monotonic()
            ticks = _read_proc_cpu_ticks(solution.pid)
            if ticks is not None:
                if cpu_ticks_start_solution is None:
                    cpu_ticks_start_solution = ticks
                cpu_ticks_last_solution = ticks
                if cpu_ticks_prev_solution is not None and ticks != cpu_ticks_prev_solution:
                    last_cpu_activity_t = now
                if cpu_ticks_prev_solution is None:
                    last_cpu_activity_t = now
                cpu_ticks_prev_solution = ticks

            if (
                config.cpu_time_limit_s is not None
                and cpu_ticks_start_solution is not None
                and cpu_ticks_last_solution is not None
            ):
                cpu_used_s = max(0.0, (cpu_ticks_last_solution - cpu_ticks_start_solution) / clk_tck)
                if cpu_used_s > config.cpu_time_limit_s:
                    timeout_reason = "TLE"
                    break

            if solution.poll() is not None and interactor.poll() is not None:
                break

            if config.idle_time_limit_s is not None:
                with lock:
                    idle_for_io = now - last_activity["t"]
                idle_for_cpu = (now - last_cpu_activity_t) if last_cpu_activity_t is not None else idle_for_io
                if idle_for_io > config.idle_time_limit_s and idle_for_cpu > config.idle_time_limit_s:
                    timeout_reason = "IDLE"
                    break

            if (now - wall_start) > config.wall_time_limit_s:
                timeout_reason = "TLE"
                break

            rss_s = _read_vmrss_kb(solution.pid)
            rss_i = _read_vmrss_kb(interactor.pid)
            if rss_s is not None:
                max_rss_solution_kb = max(max_rss_solution_kb, rss_s)
            if rss_i is not None:
                max_rss_interactor_kb = max(max_rss_interactor_kb, rss_i)

            time.sleep(0.01)

        if timeout_reason is not None:
            _kill_process_tree(solution)
            _kill_process_tree(interactor)

        try:
            solution.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            _kill_process_tree(solution)
            solution.wait(timeout=1.0)

        try:
            interactor.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            _kill_process_tree(interactor)
            interactor.wait(timeout=1.0)

        t1.join(timeout=0.2)
        t2.join(timeout=0.2)

        wall_ms = int((time.monotonic() - wall_start) * 1000)

    cpu_ms = None
    if (
        cpu_ticks_start_solution is not None
        and cpu_ticks_last_solution is not None
        and clk_tck > 0
        and cpu_ticks_last_solution >= cpu_ticks_start_solution
    ):
        cpu_ms = int((cpu_ticks_last_solution - cpu_ticks_start_solution) * 1000 / clk_tck)

    interactor_exit = int(interactor.returncode) if interactor.returncode is not None else None
    solution_exit = int(solution.returncode) if solution.returncode is not None else None

    if timeout_reason is not None:
        verdict = timeout_reason
    else:
        if interactor_exit == 0:
            verdict = "OK"
        elif interactor_exit == 1:
            verdict = "WA"
        elif interactor_exit == 2:
            verdict = "PE"
        else:
            verdict = "RE"

    tout_metrics = _parse_tout_metrics(tout_path)

    result: dict[str, Any] = {
        "verdict": verdict,
        "queries": tout_metrics["queries"],
        "query_limit": tout_metrics["query_limit"],
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "max_rss_solution_kb": max_rss_solution_kb,
        "max_rss_interactor_kb": max_rss_interactor_kb,
        "exit_codes": {"solution": solution_exit, "interactor": interactor_exit},
        "bytes_forwarded": bytes_forwarded,
    }

    if verdict != "OK":
        result["stderr_solution_tail"] = _read_tail_text(stderr_solution_path, limit_bytes=2000)
        result["stderr_interactor_tail"] = _read_tail_text(stderr_interactor_path, limit_bytes=2000)
        result["tout_tail"] = _read_tail_text(tout_path, limit_bytes=2000)
        with lock:
            try:
                result["pipe_tail_s2i"] = bytes(pipe_tail["s2i"]).decode("utf-8", errors="ignore")
            except Exception:
                result["pipe_tail_s2i"] = ""
            try:
                result["pipe_tail_i2s"] = bytes(pipe_tail["i2s"]).decode("utf-8", errors="ignore")
            except Exception:
                result["pipe_tail_i2s"] = ""

    return result


def _run_interactive_case_direct(
    *,
    solution_cmd: list[str],
    interactor_cmd: list[str],
    sol_cwd: Path,
    int_cwd: Path,
    tout_path: Path,
    stderr_solution_path: Path,
    stderr_interactor_path: Path,
    config: InteractiveConfig,
) -> dict[str, Any]:
    """Run one interactive case with direct solution/interactor pipes."""
    clk_tck = _clk_tck()
    wall_start = time.monotonic()
    preexec = _make_preexec_fn(config.rlimits)

    cpu_ticks_start_solution: int | None = None
    cpu_ticks_last_solution: int | None = None
    cpu_ticks_prev_solution: int | None = None
    last_cpu_activity_solution_t: float | None = None

    cpu_ticks_prev_interactor: int | None = None
    last_cpu_activity_interactor_t: float | None = None

    max_rss_solution_kb = 0
    max_rss_interactor_kb = 0
    timeout_reason: str | None = None

    stderr_solution_path = Path(stderr_solution_path)
    stderr_interactor_path = Path(stderr_interactor_path)
    if not stderr_solution_path.is_absolute():
        stderr_solution_path = sol_cwd / stderr_solution_path
    if not stderr_interactor_path.is_absolute():
        stderr_interactor_path = int_cwd / stderr_interactor_path

    s2i_r, s2i_w = os.pipe()
    i2s_r, i2s_w = os.pipe()
    try:
        with open(stderr_solution_path, "wb") as sol_err, open(stderr_interactor_path, "wb") as int_err:
            interactor = subprocess.Popen(
                interactor_cmd,
                stdin=s2i_r,
                stdout=i2s_w,
                stderr=int_err,
                cwd=str(int_cwd),
                start_new_session=True,
                preexec_fn=preexec,
            )
            solution = subprocess.Popen(
                solution_cmd,
                stdin=i2s_r,
                stdout=s2i_w,
                stderr=sol_err,
                cwd=str(sol_cwd),
                start_new_session=True,
                preexec_fn=preexec,
            )

            for fd in (s2i_r, s2i_w, i2s_r, i2s_w):
                try:
                    os.close(fd)
                except Exception:
                    pass
            s2i_r = s2i_w = i2s_r = i2s_w = -1

            cpu_ticks_start_solution = _read_proc_cpu_ticks(solution.pid)
            cpu_ticks_last_solution = cpu_ticks_start_solution
            cpu_ticks_prev_solution = cpu_ticks_start_solution
            if cpu_ticks_start_solution is not None:
                last_cpu_activity_solution_t = wall_start

            cpu_ticks_prev_interactor = _read_proc_cpu_ticks(interactor.pid)
            if cpu_ticks_prev_interactor is not None:
                last_cpu_activity_interactor_t = wall_start

            while True:
                now = time.monotonic()

                sol_ticks = _read_proc_cpu_ticks(solution.pid)
                if sol_ticks is not None:
                    if cpu_ticks_start_solution is None:
                        cpu_ticks_start_solution = sol_ticks
                    cpu_ticks_last_solution = sol_ticks
                    if cpu_ticks_prev_solution is not None and sol_ticks != cpu_ticks_prev_solution:
                        last_cpu_activity_solution_t = now
                    if cpu_ticks_prev_solution is None:
                        last_cpu_activity_solution_t = now
                    cpu_ticks_prev_solution = sol_ticks

                int_ticks = _read_proc_cpu_ticks(interactor.pid)
                if int_ticks is not None:
                    if cpu_ticks_prev_interactor is not None and int_ticks != cpu_ticks_prev_interactor:
                        last_cpu_activity_interactor_t = now
                    if cpu_ticks_prev_interactor is None:
                        last_cpu_activity_interactor_t = now
                    cpu_ticks_prev_interactor = int_ticks

                if (
                    config.cpu_time_limit_s is not None
                    and cpu_ticks_start_solution is not None
                    and cpu_ticks_last_solution is not None
                ):
                    cpu_used_s = max(
                        0.0, (cpu_ticks_last_solution - cpu_ticks_start_solution) / clk_tck
                    )
                    if cpu_used_s > config.cpu_time_limit_s:
                        timeout_reason = "TLE"
                        break

                if solution.poll() is not None and interactor.poll() is not None:
                    break

                if config.idle_time_limit_s is not None:
                    idle_for_sol = (
                        now - last_cpu_activity_solution_t
                        if last_cpu_activity_solution_t is not None
                        else now - wall_start
                    )
                    idle_for_int = (
                        now - last_cpu_activity_interactor_t
                        if last_cpu_activity_interactor_t is not None
                        else now - wall_start
                    )
                    if idle_for_sol > config.idle_time_limit_s and idle_for_int > config.idle_time_limit_s:
                        timeout_reason = "IDLE"
                        break

                if (now - wall_start) > config.wall_time_limit_s:
                    timeout_reason = "TLE"
                    break

                rss_s = _read_vmrss_kb(solution.pid)
                rss_i = _read_vmrss_kb(interactor.pid)
                if rss_s is not None:
                    max_rss_solution_kb = max(max_rss_solution_kb, rss_s)
                if rss_i is not None:
                    max_rss_interactor_kb = max(max_rss_interactor_kb, rss_i)

                time.sleep(0.01)

            if timeout_reason is not None:
                _kill_process_tree(solution)
                _kill_process_tree(interactor)

            try:
                solution.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _kill_process_tree(solution)
                solution.wait(timeout=1.0)

            try:
                interactor.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _kill_process_tree(interactor)
                interactor.wait(timeout=1.0)

            wall_ms = int((time.monotonic() - wall_start) * 1000)
    finally:
        for fd in (s2i_r, s2i_w, i2s_r, i2s_w):
            if isinstance(fd, int) and fd >= 0:
                try:
                    os.close(fd)
                except Exception:
                    pass

    cpu_ms = None
    if (
        cpu_ticks_start_solution is not None
        and cpu_ticks_last_solution is not None
        and clk_tck > 0
        and cpu_ticks_last_solution >= cpu_ticks_start_solution
    ):
        cpu_ms = int((cpu_ticks_last_solution - cpu_ticks_start_solution) * 1000 / clk_tck)

    interactor_exit = int(interactor.returncode) if interactor.returncode is not None else None
    solution_exit = int(solution.returncode) if solution.returncode is not None else None

    if timeout_reason is not None:
        verdict = timeout_reason
    else:
        if interactor_exit == 0:
            verdict = "OK"
        elif interactor_exit == 1:
            verdict = "WA"
        elif interactor_exit == 2:
            verdict = "PE"
        else:
            verdict = "RE"

    tout_metrics = _parse_tout_metrics(tout_path)
    result: dict[str, Any] = {
        "verdict": verdict,
        "queries": tout_metrics["queries"],
        "query_limit": tout_metrics["query_limit"],
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "max_rss_solution_kb": max_rss_solution_kb,
        "max_rss_interactor_kb": max_rss_interactor_kb,
        "exit_codes": {"solution": solution_exit, "interactor": interactor_exit},
        "bytes_forwarded": {"s2i": None, "i2s": None},
    }

    if verdict != "OK":
        result["stderr_solution_tail"] = _read_tail_text(stderr_solution_path, limit_bytes=2000)
        result["stderr_interactor_tail"] = _read_tail_text(stderr_interactor_path, limit_bytes=2000)
        result["tout_tail"] = _read_tail_text(tout_path, limit_bytes=2000)
        result["pipe_tail_s2i"] = ""
        result["pipe_tail_i2s"] = ""

    return result
