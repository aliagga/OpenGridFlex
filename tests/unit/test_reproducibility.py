from pathlib import Path

import json

import numpy as np
import pytest

from opengridflex.reproducibility import (
    build_run_manifest,
    seed_everything,
    sha256_file,
    write_run_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def test_seed_everything_repeats_numpy_stream():
    seed_everything(123, deterministic_torch=False)
    a = np.random.random(5)
    seed_everything(123, deterministic_torch=False)
    b = np.random.random(5)
    np.testing.assert_allclose(a, b)


def test_manifest_hashes_config():
    cfg = ROOT / "configs/paper1/mvp.yaml"
    m = build_run_manifest(experiment="test", seed=1, config_path=cfg, repo_root=ROOT)
    assert m.config_sha256 == sha256_file(cfg)
    assert m.python_version


def test_seed_everything_rejects_negative_seed():
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)


def test_seed_everything_enables_torch_determinism():
    state = seed_everything(321)
    assert state["seed"] == 321
    assert state["python_hash_seed"] == "321"
    assert state["numpy_seeded"] is True
    assert state["torch_available"] is True
    assert state["torch_deterministic"] is True


def test_manifest_can_be_written_and_read(tmp_path):
    cfg = ROOT / "configs/paper1/mvp.yaml"
    manifest = build_run_manifest(
        experiment="write-test",
        seed=7,
        config_path=cfg,
        repo_root=ROOT,
    )
    out = write_run_manifest(manifest, tmp_path / "nested" / "run_manifest.json")

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["experiment"] == "write-test"
    assert payload["seed"] == 7
    assert payload["config_sha256"] == sha256_file(cfg)


def test_manifest_handles_non_git_directory(tmp_path):
    cfg = ROOT / "configs/paper1/mvp.yaml"
    manifest = build_run_manifest(
        experiment="no-git",
        seed=9,
        config_path=cfg,
        repo_root=tmp_path,
    )

    assert manifest.git_commit is None
    assert manifest.git_dirty is None
