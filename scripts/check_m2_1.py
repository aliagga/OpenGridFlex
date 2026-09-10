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
        print("M2.1 GATE: BLOCKED — ruff is unavailable.")
        return 2

    checks = [
        (
            "M1.3 regression gate",
            [sys.executable, "scripts/check_m1_3.py"],
        ),
        (
            "Static quality",
            ["ruff", "check", "src", "tests", "scripts"],
        ),
        (
            "M2.1 focused tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_leakproof_dataset_unit.py",
                "tests/integration/test_leakproof_dataset_integration.py",
            ],
        ),
        (
            "M2.1 leakage audit",
            [sys.executable, "scripts/audit_m2_dataset.py"],
        ),
    ]

    passed = all(run(label, command) for label, command in checks)

    if passed:
        print("\nM2.1 GATE: GREEN")
        return 0

    print("\nM2.1 GATE: RED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
