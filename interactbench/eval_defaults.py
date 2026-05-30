from __future__ import annotations

"""Global defaults for InteractBench evaluation."""

# For `both` mode tasks: how many cases to run from each pool.
# non-adaptive pool: cases/001.in .. cases/100.in
# adaptive pool:     cases/101.in .. cases/200.in
BOTH_NON = 100
BOTH_ADAPTIVE = 0

# Per-language CPU time multiplier applied to the base time limit.
LANGUAGE_TIME_MULTIPLIER = {
    "cpp": 1,
    "go": 1,
    "python": 2,
    "java": 2,
}

CASE_NAME_WIDTH = 3
