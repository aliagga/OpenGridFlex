from pathlib import Path

import pytest

from opengridflex.config.schema import ConfigError, DataConfig, load_experiment_config

ROOT = Path(__file__).resolve().parents[2]


def test_repo_paper1_config_is_valid():
    exp, data = load_experiment_config(ROOT / "configs/paper1/mvp.yaml", repo_root=ROOT)
    assert exp.experiment == "gridshiftbench_mvp"
    assert data.dataset == "simbench"
    assert len(data.grids) == 3


def test_data_fractions_must_sum_to_one():
    with pytest.raises(ConfigError):
        DataConfig(
            dataset="x",
            grids=("g",),
            resolution_minutes=60,
            train_fraction=0.6,
            val_fraction=0.3,
            test_fraction=0.2,
            seed=1,
        ).validate()
