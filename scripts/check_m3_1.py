from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> bool:
    print(f"\n--- {label} ---")
    print(" ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=False)

    if result.returncode == 0:
        print(f"{label}: PASS")
        return True

    print(f"{label}: FAIL (exit {result.returncode})")
    return False


def main() -> int:
    if shutil.which("ruff") is None:
        print("M3.1 GATE: BLOCKED — ruff is unavailable.")
        return 2

    checks = [
        (
            "M2.2 regression gate",
            [sys.executable, "scripts/check_m2_2.py"],
        ),
        (
            "Static quality",
            ["ruff", "check", "src", "tests", "scripts"],
        ),
        (
            "M3.1 focused tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_sanity_baselines_unit.py",
                "tests/integration/test_sanity_baselines_integration.py",
            ],
        ),
        (
            "M3.1 full validation baselines",
            [sys.executable, "scripts/run_m3_1_baselines.py"],
        ),
    ]

    passed = all(run(label, command) for label, command in checks)

    if passed:
        print("\nM3.1 GATE: GREEN")
        return 0

    print("\nM3.1 GATE: RED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
