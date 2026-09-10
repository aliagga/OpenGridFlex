from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from opengridflex.grids.physical_reconstruction import (
    AbsoluteProfileBundle,
    GridIntegrityError,
    _normalize_absolute_key,
    _resolve_profile_factor,
    apply_absolute_profile_row,
    manual_absolute_row,
)


def test_absolute_key_normalization_supports_both_simbench_forms() -> None:
    assert _normalize_absolute_key(("load", "p_mw")) == ("load", "p_mw")
    assert _normalize_absolute_key("load.p_mw") == ("load", "p_mw")


def test_absolute_key_normalization_fails_closed() -> None:
    with pytest.raises(GridIntegrityError, match="Unexpected"):
        _normalize_absolute_key("load")


def test_manual_load_p_and_q_formula_uses_simbench_suffixes() -> None:
    net = SimpleNamespace(
        load=pd.DataFrame(
            {
                "p_mw": [2.0],
                "q_mvar": [0.5],
                "profile": ["H0"],
            },
            index=[7],
        ),
        profiles={
            "load": pd.DataFrame(
                {
                    "time": ["01.01.2016 00:00"],
                    "H0_pload": [0.4],
                    "H0_qload": [0.6],
                }
            ),
        },
    )

    p = manual_absolute_row(net, "load", "p_mw", 0)
    q = manual_absolute_row(net, "load", "q_mvar", 0)

    assert p.at[7] == pytest.approx(0.8)
    assert q.at[7] == pytest.approx(0.3)


def test_storage_formula_preserves_consumer_sign() -> None:
    net = SimpleNamespace(
        storage=pd.DataFrame(
            {
                "p_mw": [-2.0],
                "profile": ["battery"],
            },
            index=[3],
        ),
        profiles={
            "storage": pd.DataFrame(
                {
                    "time": ["01.01.2016 00:00"],
                    "battery": [0.25],
                }
            ),
        },
    )

    p = manual_absolute_row(net, "storage", "p_mw", 0)

    assert p.at[3] == pytest.approx(-0.5)


def test_generator_profile_lookup_does_not_merge_on_duplicate_local_time() -> None:
    times = [
        "30.10.2016 02:00",
        "30.10.2016 02:00",
    ]
    net = SimpleNamespace(
        profiles={
            "powerplants": pd.DataFrame({"time": times}),
            "renewables": pd.DataFrame(
                {
                    "time": times,
                    "PV-A": [0.25, 0.75],
                }
            ),
        }
    )

    first = _resolve_profile_factor(net, "sgen", "p_mw", "PV-A", 0)
    second = _resolve_profile_factor(net, "sgen", "p_mw", "PV-A", 1)

    assert first == pytest.approx(0.25)
    assert second == pytest.approx(0.75)


def test_generator_profile_lookup_rejects_ambiguous_profile_name() -> None:
    net = SimpleNamespace(
        profiles={
            "powerplants": pd.DataFrame(
                {
                    "time": ["01.01.2016 00:00"],
                    "same": [0.5],
                }
            ),
            "renewables": pd.DataFrame(
                {
                    "time": ["01.01.2016 00:00"],
                    "same": [0.7],
                }
            ),
        }
    )

    with pytest.raises(GridIntegrityError, match="ambiguous"):
        _resolve_profile_factor(net, "sgen", "p_mw", "same", 0)


def test_apply_absolute_profile_row_is_exact() -> None:
    net = {
        "load": pd.DataFrame(
            {
                "p_mw": [10.0, 20.0],
                "q_mvar": [1.0, 2.0],
            },
            index=[4, 9],
        )
    }
    bundle = AbsoluteProfileBundle(
        values={
            ("load", "p_mw"): pd.DataFrame(
                [[3.0, 5.0]],
                columns=[4, 9],
            ),
        },
        row_count=1,
    )

    apply_absolute_profile_row(net, bundle, 0)

    assert np.array_equal(net["load"]["p_mw"].to_numpy(), np.array([3.0, 5.0]))
