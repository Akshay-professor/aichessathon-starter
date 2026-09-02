"""Measure cold import cost against the platform's 60 second init budget.

numba compiles at import, the platform's filesystem is read-only, and each game starts a fresh
container, so a compile cache cannot persist. Every game pays this in full, and a build that
overruns loses every game to `init` while looking perfect locally on a warm cache. This is the
number to watch on every commit.

    python -m tools.import_time
"""

import argparse
import subprocess
import sys
import time

BUDGET_S = 60.0
CEILING_S = 40.0

SCRIPT = """
import time
started = time.perf_counter()
import agent
print(f"{time.perf_counter() - started:.2f}")
"""


def measure(runs: int) -> list[float]:
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", SCRIPT],
            check=True,
            capture_output=True,
            env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
        )
        timings.append(time.perf_counter() - started)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Time a cold import of the agent.")
    parser.add_argument("--runs", type=int, default=3)
    arguments = parser.parse_args()

    timings = measure(arguments.runs)
    worst = max(timings)
    print("  ".join(f"{value:.2f}s" for value in timings))
    print(f"worst {worst:.2f}s of a {BUDGET_S:.0f}s budget, ceiling {CEILING_S:.0f}s")
    if worst > CEILING_S:
        raise SystemExit(
            f"cold import takes {worst:.2f}s, over the {CEILING_S:.0f}s ceiling. "
            "Every game pays this before the clock starts."
        )


if __name__ == "__main__":
    main()
