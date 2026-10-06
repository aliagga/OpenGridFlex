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
        print("M3.2 GATE: BLOCKED — ruff is unavailable.")
        return 2

    checks = [
        (
            "M3.1 regression gate",
            [sys.executable, "scripts/check_m3_1.py"],
        ),
        (
            "LightGBM version",
            [sys.executable, "scripts/check_lightgbm_version.py"],
        ),
        (
            "Static quality",
            ["ruff", "check", "src", "tests", "scripts"],
        ),
        (
            "M3.2 focused tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_lightgbm_features_unit.py",
                "tests/integration/test_lightgbm_baseline_integration.py",
            ],
        ),
        (
            "M3.2 full primary validation",
            [sys.executable, "scripts/run_m3_2_lightgbm.py"],
        ),
    ]

    passed = all(run(label, command) for label, command in checks)

    if passed:
        print("\nM3.2 GATE: GREEN")
        return 0

    print("\nM3.2 GATE: RED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
