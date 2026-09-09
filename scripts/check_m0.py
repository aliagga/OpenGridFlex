#!/usr/bin/env python3
"""Milestone-0 acceptance gate.

Exit codes:
  0: all required checks passed
  1: a required check failed
  2: a required tool is unavailable (gate remains blocked)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, cmd: list[str], *, env: dict[str, str] | None = None) -> bool:
    print(f"\n=== {label} ===")
    p = subprocess.run(cmd, cwd=ROOT, env=env)
    if p.returncode != 0:
        print(f"FAIL: {label} (exit {p.returncode})")
        return False
    print(f"PASS: {label}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-missing-lint",
        action="store_true",
        help="Development-only escape hatch. Never use to certify M0 green.",
    )
    args = parser.parse_args()

    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    checks = [
        ("Python compile", [sys.executable, "-m", "compileall", "-q", "src"]),
        (
            "Config validation",
            [
                sys.executable,
                "-m",
                "opengridflex.cli",
                "validate-config",
                "configs/paper1/mvp.yaml",
                "--repo-root",
                ".",
            ],
        ),
        ("Unit tests", [sys.executable, "-m", "pytest", "-q"]),
        (
            "Reproducibility self-check",
            [
                sys.executable,
                "-m",
                "opengridflex.cli",
                "self-check",
                "configs/paper1/mvp.yaml",
                "--repo-root",
                ".",
                "--seed",
                "42",
                "--output",
                "artifacts/self_check",
            ],
        ),
    ]

    for label, cmd in checks:
        if not run(label, cmd, env=env):
            return 1

    ruff = shutil.which("ruff")
    if ruff is None:
        print("\n=== Static lint ===")
        print(
            "BLOCKED: ruff is not installed. M0 must remain non-green "
            "until lint runs in CI or a dev environment."
        )
        return 0 if args.allow_missing_lint else 2
    if not run("Static lint", [ruff, "check", "src", "tests"], env=env):
        return 1

    print("\nM0 GATE: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
