"""go-judge HTTP API client for sandboxed code execution."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


def _resolve_go_binary() -> str:
    override = os.environ.get("GO_BIN", "").strip()
    if override:
        if "/" in override:
            override_path = Path(override).expanduser()
            if override_path.is_file() and os.access(override_path, os.X_OK):
                return str(override_path)
        else:
            resolved = shutil.which(override)
            if resolved:
                return resolved
        raise CompileError(f"Configured GO compiler not found: {override}")

    resolved = shutil.which("go")
    if resolved:
        return resolved

    raise CompileError("Go compiler not found in PATH; install `go` or set GO_BIN")


@dataclass
class RunResult:
    status: str = ""
    exit_status: int = 0
    time_ns: int = 0
    run_time_ns: int = 0
    memory_bytes: int = 0
    files: dict[str, str] = field(default_factory=dict)
    file_ids: dict[str, str] = field(default_factory=dict)
    file_error: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class GoJudgeClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5050", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def run_one(self, cmd: dict[str, Any]) -> RunResult:
        resp = self.session.post(
            f"{self.base_url}/run",
            json={"cmd": [cmd]},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()[0]
        return self._parse_result(data)

    def run(self, request: dict[str, Any]) -> list[RunResult]:
        resp = self.session.post(
            f"{self.base_url}/run",
            json=request,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return [self._parse_result(r) for r in resp.json()]

    def _parse_result(self, data: dict[str, Any]) -> RunResult:
        files = data.get("files", {})
        file_ids = data.get("fileIds", {})
        file_error = data.get("fileError", [])
        return RunResult(
            status=str(data.get("status", "") or ""),
            exit_status=int(data.get("exitStatus", 0) or 0),
            time_ns=int(data.get("time", 0) or 0),
            run_time_ns=int(data.get("runTime", 0) or 0),
            memory_bytes=int(data.get("memory", 0) or 0),
            files=files if isinstance(files, dict) else {},
            file_ids=file_ids if isinstance(file_ids, dict) else {},
            file_error=file_error if isinstance(file_error, list) else [],
            error=data.get("error"),
        )

    def delete_file(self, file_id: str) -> None:
        try:
            self.session.delete(f"{self.base_url}/file/{file_id}", timeout=10)
        except Exception:
            pass

    def upload_file(self, file_path: Path) -> str:
        with open(file_path, "rb") as f:
            resp = self.session.post(
                f"{self.base_url}/file",
                files={"file": f},
                timeout=self.timeout,
            )
        resp.raise_for_status()
        return resp.json().get("fileId", "")

    def compile_cpp(
        self,
        source: str,
        *,
        src_name: str = "main.cpp",
        out_name: str = "a",
        extra_files: dict[str, str] | None = None,
        std: str = "c++17",
    ) -> tuple[str, str]:
        """Compile C++ source, return (exe_file_id, stderr).

        Args:
            extra_files: Additional files to copy into sandbox (e.g. {"testlib.h": content})
        """
        args = ["/usr/bin/g++", src_name, "-O2", "-pipe", f"-std={std}", "-o", out_name]
        if extra_files:
            args.extend(["-I", "."])

        copy_in = {src_name: {"content": source}}
        if extra_files:
            for name, content in extra_files.items():
                copy_in[name] = {"content": content}

        result = self.run_one({
            "args": args,
            "env": ["PATH=/usr/bin:/bin"],
            "files": [
                {"content": ""},
                {"name": "stdout", "max": 1 << 20},
                {"name": "stderr", "max": 1 << 20},
            ],
            "copyIn": copy_in,
            "copyOut": ["stdout", "stderr"],
            "copyOutCached": [out_name],
            "cpuLimit": 30_000_000_000,
            "memoryLimit": 512 << 20,
            "procLimit": 50,
        })

        if result.status != "Accepted":
            raise CompileError(result.files.get("stderr", result.status))

        exe_id = result.file_ids.get(out_name, "")
        if not exe_id:
            raise CompileError("No executable produced")
        return exe_id, result.files.get("stderr", "")

    def compile_go(self, source: str, *, src_name: str = "main.go") -> tuple[str, str]:
        """Compile Go source, return (exe_file_id, stderr)."""
        go_bin = _resolve_go_binary()
        go_bin_dir = str(Path(go_bin).resolve().parent)
        result = self.run_one({
            "args": [go_bin, "build", "-o", "a", src_name],
            "env": [
                f"PATH={go_bin_dir}:/usr/bin:/bin",
                "HOME=/w",
                "GOCACHE=/w/.gocache",
                "GOMODCACHE=/w/.gomodcache",
                "GOPATH=/w/.gopath",
            ],
            "files": [
                {"content": ""},
                {"name": "stdout", "max": 1 << 20},
                {"name": "stderr", "max": 1 << 20},
            ],
            "copyIn": {src_name: {"content": source}},
            "copyOut": ["stdout", "stderr"],
            "copyOutCached": ["a"],
            "cpuLimit": 30_000_000_000,
            "memoryLimit": 512 << 20,
            "procLimit": 128,
        })

        if result.status != "Accepted":
            raise CompileError(result.files.get("stderr", result.status))

        exe_id = result.file_ids.get("a", "")
        if not exe_id:
            raise CompileError("No Go executable produced")
        return exe_id, result.files.get("stderr", "")

    def compile_java(self, source: str) -> tuple[dict[str, str], str]:
        """Compile Java source as Main.java, return ({class_name: file_id}, stderr)."""
        common_request = {
            "env": ["PATH=/usr/bin:/bin"],
            "files": [
                {"content": ""},
                {"name": "stdout", "max": 1 << 20},
                {"name": "stderr", "max": 1 << 20},
            ],
            "copyIn": {"Main.java": {"content": source}},
            "copyOut": ["stdout", "stderr"],
            "cpuLimit": 30_000_000_000,
            "memoryLimit": 512 << 20,
            "procLimit": 50,
        }

        listing = self.run_one({
            **common_request,
            "args": [
                "/bin/sh",
                "-lc",
                (
                    "mkdir -p classes "
                    "&& /usr/bin/javac -encoding UTF-8 -d classes Main.java "
                    "&& /usr/bin/find classes -maxdepth 1 -type f -name '*.class' "
                    "-printf '%f\\n' | /usr/bin/sort > classlist.txt"
                ),
            ],
            "copyOut": ["stdout", "stderr", "classlist.txt"],
        })
        if listing.status != "Accepted":
            raise CompileError(listing.files.get("stderr", listing.status))

        class_names = [
            line.strip()
            for line in listing.files.get("classlist.txt", "").splitlines()
            if line.strip()
        ]
        if not class_names:
            raise CompileError("No Java class files produced")

        result = self.run_one({
            **common_request,
            "args": ["/usr/bin/javac", "-encoding", "UTF-8", "-d", "classes", "Main.java"],
            "copyOutCached": [f"classes/{name}" for name in class_names],
        })
        if result.status != "Accepted":
            raise CompileError(result.files.get("stderr", result.status))

        class_files: dict[str, str] = {}
        missing: list[str] = []
        for name in class_names:
            file_id = result.file_ids.get(f"classes/{name}", "")
            if not file_id:
                missing.append(name)
                continue
            class_files[name] = file_id
        if missing:
            raise CompileError(f"Missing Java class outputs: {', '.join(missing)}")

        return class_files, result.files.get("stderr", "")


class CompileError(Exception):
    pass
