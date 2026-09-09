from pathlib import Path

import numpy as np

from opengridflex.reproducibility import build_run_manifest, seed_everything, sha256_file

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
