from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "neuralforecast",
    "pytorch-lightning",
    "torch",
    "torch-geometric",
    "pandapower",
    "simbench",
    "pyyaml",
)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _git_value(args: list[str], repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _package_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    created_utc: str
    experiment: str
    seed: int
    config_sha256: str
    config_path: str
    git_commit: str | None
    git_dirty: bool | None
    python_version: str
    platform: str
    machine: str
    processor: str
    cpu_count: int | None
    packages: dict[str, str | None]
    env: dict[str, str | None]


def build_run_manifest(
    *, experiment: str, seed: int, config_path: str | Path, repo_root: str | Path
) -> RunManifest:
    config_path = Path(config_path).resolve()
    root = Path(repo_root).resolve()
    status = _git_value(["status", "--porcelain"], root)
    return RunManifest(
        schema_version="1.0",
        created_utc=datetime.now(UTC).isoformat(),
        experiment=experiment,
        seed=seed,
        config_sha256=sha256_file(config_path),
        config_path=str(config_path),
        git_commit=_git_value(["rev-parse", "HEAD"], root),
        git_dirty=None if status is None else bool(status),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        packages=_package_versions(),
        env={
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
        },
    )


def write_run_manifest(manifest: RunManifest, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
