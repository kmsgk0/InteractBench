#!/usr/bin/env python3
"""Generate test cases for InteractBench problems using their generators."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Batch:
    mode: str
    count: int
    seed_start: int


def _load_interactor_mode(problem_dir: Path) -> str:
    meta_path = problem_dir / "meta.json"
    if not meta_path.exists():
        return "non_adaptive"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return "non_adaptive"
    mode = meta.get("interactor_mode", "non_adaptive")
    if not isinstance(mode, str):
        return "non_adaptive"
    return mode


def _pick_batches(
    *,
    meta_interactor_mode: str,
    count_per_mode: int,
    seed_start: int,
    forced_mode: str,
) -> list[Batch]:
    """Pick generation batches from the problem interactor mode."""
    if forced_mode != "auto":
        return [Batch(mode=forced_mode, count=count_per_mode, seed_start=seed_start)]

    if meta_interactor_mode == "adaptive":
        return [Batch(mode="adp", count=count_per_mode, seed_start=seed_start)]

    if meta_interactor_mode == "both":
        return [
            Batch(mode="non", count=count_per_mode, seed_start=seed_start),
            Batch(mode="adp", count=count_per_mode, seed_start=seed_start + count_per_mode),
        ]

    return [Batch(mode="non", count=count_per_mode, seed_start=seed_start)]


def _detect_problem_dirs(root: Path, problem_ids: Sequence[str] | None) -> list[Path]:
    skip_names = {"third_party", "__pycache__", ".git"}
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"problems root not found: {root}. Run scripts/import_from_jsonl.py first."
        )

    if problem_ids:
        dirs = [root / pid for pid in problem_ids]
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]

    out: list[Path] = []
    for d in dirs:
        if d.name in skip_names:
            continue
        if not d.is_dir():
            continue
        out.append(d)
    return sorted(out, key=lambda p: p.name)


def _ensure_compiler(cxx: str) -> str:
    resolved = shutil.which(cxx)
    if resolved:
        return resolved
    if Path(cxx).exists():
        return cxx
    raise FileNotFoundError(f"Compiler not found: {cxx}")


def _compile_generator(
    *,
    compiler: str,
    generator_dir: Path,
    testlib_include_dir: Path,
    rebuild: bool,
    verbose: bool,
) -> Path:
    src = (generator_dir / "gen_cases.cpp").resolve()
    if not src.exists():
        raise FileNotFoundError(f"Missing generator source: {src}")

    bin_path = generator_dir / ("gen_cases.exe" if os.name == "nt" else "gen_cases")
    bin_path = bin_path.resolve()

    if (
        not rebuild
        and bin_path.exists()
        and bin_path.stat().st_mtime >= src.stat().st_mtime
    ):
        if os.name != "nt":
            try:
                st = bin_path.stat()
                if (st.st_mode & 0o111) == 0:
                    bin_path.chmod(st.st_mode | 0o111)
            except Exception:
                pass
        return bin_path

    cmd = [
        compiler,
        "-std=c++17",
        "-O2",
        "-pipe",
        "-o",
        str(bin_path),
        str(src),
        f"-I{testlib_include_dir.resolve()}",
    ]
    if verbose:
        print("[CXX]", " ".join(cmd))

    subprocess.run(cmd, check=True)
    if os.name != "nt":
        try:
            st = bin_path.stat()
            if (st.st_mode & 0o111) == 0:
                bin_path.chmod(st.st_mode | 0o111)
        except Exception:
            pass
    return bin_path


def _write_case(
    *,
    generator_bin: Path,
    generator_cwd: Path,
    out_path: Path,
    seed: int,
    mode: str,
    pass_mode_arg: bool,
    gen_q: int | None,
    extra_gen_args: Sequence[str],
    verbose: bool,
) -> None:
    cmd: list[str] = [str(generator_bin), str(seed)]

    if gen_q is not None:
        cmd.append(f"-q={gen_q}")

    if pass_mode_arg:
        cmd.append(f"-mode={mode}")

    cmd.extend(extra_gen_args)

    if verbose:
        print(f"[GEN] {generator_cwd.parent.name} -> {out_path.name}: {' '.join(cmd)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, cwd=str(generator_cwd), stdout=f)


def _has_mode_arg(extra_args: Sequence[str]) -> bool:
    for a in extra_args:
        if a.startswith("-mode=") or a.startswith("--mode="):
            return True
        if a in {"-mode", "--mode"}:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate test cases (cases/*.in) for InteractBench problems.\n\n"
            "Tip: pass generator-specific flags after '--', e.g.\n"
            "  python scripts/gen_cases.py --problem-ids cf1999_g1 --count 20 -- -q=1000\n"
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/problems"),
        help="Problems root directory (default: data/problems)",
    )
    parser.add_argument(
        "--problem-ids",
        nargs="*",
        default=None,
        help="Generate for specific problem IDs (default: all)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Cases to generate per problem (default: 100)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Starting seed (default: 1)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "non", "adp"],
        default="auto",
        help="Generation mode: auto (from meta.json), non, or adp (default: auto)",
    )
    parser.add_argument(
        "--gen-q",
        type=int,
        default=None,
        help="Pass '-q=<N>' to generator",
    )
    parser.add_argument(
        "--cxx",
        default=os.environ.get("CXX", "g++"),
        help="C++ compiler (default: $CXX or g++)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing cases/*.in before generating",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force recompile generator",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "gen_args",
        nargs=argparse.REMAINDER,
        help="Extra args for generator (after '--')",
    )

    args = parser.parse_args(argv)

    root: Path = args.root
    testlib_include_dir = Path("third_party/testlib")
    testlib_header = testlib_include_dir / "testlib.h"
    if not args.dry_run and not testlib_header.exists():
        print(f"[ERROR] Missing {testlib_header}", file=sys.stderr)
        return 2

    extra_gen_args: list[str] = list(args.gen_args)
    if extra_gen_args and extra_gen_args[0] == "--":
        extra_gen_args = extra_gen_args[1:]

    try:
        problem_dirs = _detect_problem_dirs(root, args.problem_ids)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    if not problem_dirs:
        print("[ERROR] No problem directories found.", file=sys.stderr)
        return 2

    compiler: str | None = None
    if not args.dry_run:
        try:
            compiler = _ensure_compiler(args.cxx)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 2

    width = max(3, len(str(args.count)))
    pass_mode_arg = not _has_mode_arg(extra_gen_args)

    ok = 0
    skipped = 0
    failed = 0

    for problem_dir in problem_dirs:
        generator_dir = problem_dir / "generator"
        gen_cpp = generator_dir / "gen_cases.cpp"
        if not gen_cpp.exists():
            skipped += 1
            if args.verbose:
                print(f"[SKIP] {problem_dir.name}: no generator")
            continue

        cases_dir = problem_dir / "cases"
        meta_mode = _load_interactor_mode(problem_dir)
        batches = _pick_batches(
            meta_interactor_mode=meta_mode,
            count_per_mode=args.count,
            seed_start=args.seed_start,
            forced_mode=args.mode,
        )

        if args.dry_run:
            print(f"[DRY] {problem_dir.name}: batches={batches}")
            ok += 1
            continue

        if args.clean and cases_dir.exists():
            for p in cases_dir.glob("*.in"):
                try:
                    p.unlink()
                except Exception:
                    pass

        try:
            assert compiler is not None
            generator_bin = _compile_generator(
                compiler=compiler,
                generator_dir=generator_dir,
                testlib_include_dir=testlib_include_dir,
                rebuild=args.rebuild,
                verbose=args.verbose,
            )

            total_cases = 0
            for batch in batches:
                for i in range(batch.count):
                    seed = batch.seed_start + i
                    case_num = 101 + i if batch.mode == "adp" else 1 + i
                    out_path = cases_dir / f"{case_num:0{width}d}.in"
                    total_cases += 1
                    _write_case(
                        generator_bin=generator_bin,
                        generator_cwd=generator_dir,
                        out_path=out_path,
                        seed=seed,
                        mode=batch.mode,
                        pass_mode_arg=pass_mode_arg,
                        gen_q=args.gen_q,
                        extra_gen_args=extra_gen_args,
                        verbose=args.verbose,
                    )

            print(f"[OK] {problem_dir.name}: {total_cases} cases")
            ok += 1
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"[FAIL] {problem_dir.name}: command failed ({e.returncode})", file=sys.stderr)
        except Exception as e:
            failed += 1
            print(f"[FAIL] {problem_dir.name}: {e}", file=sys.stderr)

    print(f"Done. ok={ok}, skipped={skipped}, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
