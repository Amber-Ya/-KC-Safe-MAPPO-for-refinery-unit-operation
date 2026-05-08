"""Convenience entry point for the IEEE TII-oriented baseline suite."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.run_comparison import main as run_comparison_main


def main() -> int:
    args = sys.argv[1:]
    sys.argv = [
        "baselines/run_comparison.py",
        "--include_metaheuristics",
        "--include_rolling",
        *args,
    ]
    return run_comparison_main()


if __name__ == "__main__":
    raise SystemExit(main())
