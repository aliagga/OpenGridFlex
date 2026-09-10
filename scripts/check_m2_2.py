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
        print("M2.2 GATE: BLOCKED — ruff is unavailable.")
        return 2

    checks = [
        (
            "M2.1 regression gate",
            [sys.executable, "scripts/check_m2_1.py"],
        ),
        (
            "Static quality",
            ["ruff", "check", "src", "tests", "scripts"],
        ),
        (
            "M2.2 focused tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_benchmark_contract_unit.py",
                "tests/integration/test_benchmark_contract_integration.py",
            ],
        ),
        (
            "M2.2 contract + serialization audit",
            [sys.executable, "scripts/audit_m2_2_contract.py"],
        ),
    ]

    passed = all(run(label, command) for label, command in checks)

    if passed:
        print("\nM2.2 GATE: GREEN")
        return 0

    print("\nM2.2 GATE: RED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
