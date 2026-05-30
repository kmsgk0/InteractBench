from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

@contextlib.contextmanager
def _locked_json_file(path: Path) -> Any:
    """Open `path` for r/w under an exclusive advisory lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        f = os.fdopen(fd, "r+", encoding="utf-8")
        fd = -1
        with f:
            yield f
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def update_result(path: Path, updater: Any) -> dict[str, Any]:
    """Atomically update a result.json file."""
    with _locked_json_file(path) as f:
        f.seek(0)
        raw = f.read()
        try:
            data = json.loads(raw) if raw.strip() else {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}

        updater(data)

        f.seek(0)
        f.truncate()
        f.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

        return data
