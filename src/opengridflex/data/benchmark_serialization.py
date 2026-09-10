from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from opengridflex.data.benchmark_contract import (
    BenchmarkContract,
    build_contract_window_plans,
    default_benchmark_contract,
    fit_contract_standardizer,
    make_contract_split,
    validate_contract,
)
from opengridflex.data.leakproof_dataset import (
    CanonicalGridSeries,
    known_future_calendar_features,
)
from opengridflex.grids.simbench_adapter import GridIntegrityError

MANIFEST_NAME = "manifest.json"
SERIALIZATION_VERSION = 1

SPLIT_CODE = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

WINDOW_COLUMNS = (
    "split_code",
    "history_start",
    "history_end",
    "origin",
    "target_start",
    "target_end",
)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def _save_npy(path: Path, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    with path.open("wb") as handle:
        np.save(
            handle,
            contiguous,
            allow_pickle=False,
        )


def _window_array(plan) -> np.ndarray:
    rows = [
        (
            SPLIT_CODE[window.split],
            window.history_start,
            window.history_end,
            window.origin,
            window.target_start,
            window.target_end,
        )
        for window in plan.windows
    ]
    return np.asarray(rows, dtype=np.int64)


def write_benchmark_artifact(
    series: CanonicalGridSeries,
    output_dir: str | Path,
    *,
    contract: BenchmarkContract | None = None,
    overwrite: bool = False,
) -> Path:
    benchmark = contract or default_benchmark_contract()
    contract_audit = validate_contract(series, benchmark)

    root = Path(output_dir)
    if root.exists():
        if not overwrite:
            if any(root.iterdir()):
                raise GridIntegrityError(f"Artifact directory {root} is not empty.")
        else:
            shutil.rmtree(root)

    root.mkdir(parents=True, exist_ok=True)
    channels_dir = root / "channels"
    windows_dir = root / "windows"
    scaling_dir = root / "scaling"
    channels_dir.mkdir()
    windows_dir.mkdir()
    scaling_dir.mkdir()

    payload_files: list[Path] = []

    time_path = root / "time_utc_ns.npy"
    _save_npy(
        time_path,
        np.asarray(series.time_utc.asi8, dtype=np.int64),
    )
    payload_files.append(time_path)

    bus_path = root / "bus_index.npy"
    try:
        bus_values = np.asarray(series.bus_index, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise GridIntegrityError("Benchmark v1 requires integer bus indices.") from exc
    _save_npy(bus_path, bus_values)
    payload_files.append(bus_path)

    for channel, values in series.values.items():
        path = channels_dir / f"{channel}.npy"
        _save_npy(path, values.astype(np.float32, copy=False))
        payload_files.append(path)

    calendar = known_future_calendar_features(series.time_utc).to_numpy(dtype=np.float32)
    calendar_path = root / "calendar_features.npy"
    _save_npy(calendar_path, calendar)
    payload_files.append(calendar_path)

    plans = build_contract_window_plans(series, benchmark)
    for task_id, plan in plans.items():
        path = windows_dir / f"{task_id}.npy"
        _save_npy(path, _window_array(plan))
        payload_files.append(path)

    scaler = fit_contract_standardizer(series, benchmark)
    for channel in scaler.channels:
        for suffix, array in (
            ("mean", scaler.means[channel]),
            ("scale", scaler.scales[channel]),
            (
                "constant_mask",
                scaler.constant_mask[channel].astype(np.uint8),
            ),
        ):
            path = scaling_dir / f"{channel}_{suffix}.npy"
            _save_npy(path, array)
            payload_files.append(path)

    split = make_contract_split(series, benchmark)

    file_records = {}
    for path in sorted(payload_files):
        relative = path.relative_to(root).as_posix()
        file_records[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    manifest = {
        "serialization_version": SERIALIZATION_VERSION,
        "contract": benchmark.to_dict(),
        "contract_audit": contract_audit,
        "source_fingerprint": series.fingerprint,
        "grid_code": series.grid_code,
        "n_steps": series.n_steps,
        "n_buses": series.n_buses,
        "time": {
            "timezone": "UTC",
            "start": series.time_utc[0].isoformat(),
            "end_inclusive": series.time_utc[-1].isoformat(),
            "resolution_minutes": 15,
        },
        "split": split.to_dict(),
        "channel_order": list(series.values),
        "channel_dtype": "float32",
        "calendar_feature_order": list(benchmark.tasks[0].known_future_features),
        "window_columns": list(WINDOW_COLUMNS),
        "split_code": SPLIT_CODE,
        "scaler": {
            "channels": list(scaler.channels),
            "fit_start": scaler.fit_start,
            "fit_end": scaler.fit_end,
            "source_fingerprint": scaler.source_fingerprint,
        },
        "files": file_records,
    }

    manifest_path = root / MANIFEST_NAME
    _write_json(manifest_path, manifest)

    verify_benchmark_artifact(root)
    return root


def verify_benchmark_artifact(
    artifact_dir: str | Path,
) -> dict[str, Any]:
    root = Path(artifact_dir)
    manifest_path = root / MANIFEST_NAME

    if not manifest_path.is_file():
        raise GridIntegrityError(f"Benchmark artifact has no {MANIFEST_NAME}.")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GridIntegrityError("Benchmark manifest is invalid JSON.") from exc

    if manifest.get("serialization_version") != SERIALIZATION_VERSION:
        raise GridIntegrityError("Unexpected benchmark serialization version.")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise GridIntegrityError("Benchmark manifest contains no payload-file records.")

    for relative, record in files.items():
        path = root / relative
        if not path.is_file():
            raise GridIntegrityError(f"Serialized payload is missing: {relative}")

        observed = _sha256_file(path)
        expected = record.get("sha256")
        if observed != expected:
            raise GridIntegrityError(
                f"SHA-256 mismatch for {relative}: expected {expected}, observed {observed}."
            )

        if path.stat().st_size != record.get("size_bytes"):
            raise GridIntegrityError(f"File-size mismatch for {relative}.")

    return manifest
