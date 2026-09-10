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
        print("M1.3 GATE: BLOCKED — ruff is unavailable.")
        return 2

    checks = [
        (
            "M1.2 regression gate",
            [sys.executable, "scripts/check_m1_2.py"],
        ),
        (
            "Static quality",
            ["ruff", "check", "src", "tests", "scripts"],
        ),
        (
            "M1.3 focused tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_physical_reconstruction_unit.py",
                "tests/integration/test_physical_reconstruction.py",
            ],
        ),
        (
            "M1.3 multi-snapshot physics validation",
            [sys.executable, "scripts/validate_m1_3_physics.py"],
        ),
    ]

    passed = all(run(label, command) for label, command in checks)

    if passed:
        print("\nM1.3 GATE: GREEN")
        return 0

    print("\nM1.3 GATE: RED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
