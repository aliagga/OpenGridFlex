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
        print("M3.3 DEVELOPMENT GATE: BLOCKED — ruff is unavailable.")
        return 2

    checks = [
        (
            "M3.2 regression gate",
            [sys.executable, "scripts/check_m3_2.py"],
        ),
        (
            "NeuralForecast version",
            [sys.executable, "scripts/check_neuralforecast_version.py"],
        ),
        (
            "Static quality",
            ["ruff", "check", "src", "tests", "scripts"],
        ),
        (
            "M3.3 focused tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_patchtst_baseline_unit.py",
                "tests/integration/test_patchtst_baseline_integration.py",
            ],
        ),
        (
            "M3.3 frozen-contract audit",
            [sys.executable, "scripts/audit_m3_3_patchtst.py"],
        ),
    ]

    passed = all(run(label, command) for label, command in checks)

    if passed:
        print("\nM3.3 DEVELOPMENT GATE: GREEN")
        print("Full PatchTST validation is still required before M3.3 can be frozen.")
        return 0

    print("\nM3.3 DEVELOPMENT GATE: RED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
