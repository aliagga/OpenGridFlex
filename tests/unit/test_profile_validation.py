from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.api.types import is_object_dtype

import opengridflex.grids.simbench_adapter as adapter
from opengridflex.grids.simbench_adapter import (
    GridIntegrityError,
    _numeric_profile_view,
    _parse_profile_time,
    _profile_group_summary,
    _validate_profile_mapping_integrity,
)

NORMAL_TIMES = ["01.01.2016 00:00", "01.01.2016 00:15"]


def _profile(times: list[str], values: list[object] | None = None) -> pd.DataFrame:
    data: dict[str, list[object] | list[str]] = {"time": times}
    if values is not None:
        data["signal"] = values
    return pd.DataFrame(data)


def test_numeric_profile_view_accepts_numeric_strings_without_mutation() -> None:
    frame = pd.DataFrame(
        {
            "time": NORMAL_TIMES,
            "plant": ["1.25", "2.50"],
        }
    )
    before = frame.copy(deep=True)

    view = _numeric_profile_view("powerplants", frame)

    assert view.shape == (2, 1)
    assert is_object_dtype(frame["plant"].dtype)
    assert frame.equals(before)
    assert view["plant"].dtype == float


def test_profile_validator_rejects_truly_non_numeric_values() -> None:
    frame = pd.DataFrame(
        {
            "time": NORMAL_TIMES,
            "plant": ["1.25", "not-a-number"],
        }
    )

    with pytest.raises(GridIntegrityError, match="cannot be interpreted as numeric"):
        _numeric_profile_view("powerplants", frame)


def test_profile_validator_rejects_bad_timestamp_format() -> None:
    frame = pd.DataFrame(
        {
            "time": ["2016-01-01 00:00", "2016-01-01 00:15"],
            "plant": [1.0, 2.0],
        }
    )

    with pytest.raises(GridIntegrityError, match="timestamps that do not match"):
        _parse_profile_time("powerplants", frame)


def test_dst_fallback_repeated_local_hour_becomes_unique_utc() -> None:
    times = [
        "30.10.2016 01:45",
        "30.10.2016 02:00",
        "30.10.2016 02:15",
        "30.10.2016 02:30",
        "30.10.2016 02:45",
        "30.10.2016 02:00",
        "30.10.2016 02:15",
        "30.10.2016 02:30",
        "30.10.2016 02:45",
        "30.10.2016 03:00",
    ]
    frame = _profile(times, [1.0] * len(times))

    utc_time = _parse_profile_time("load", frame)

    assert not utc_time.has_duplicates
    assert utc_time.is_monotonic_increasing
    assert set(utc_time[1:] - utc_time[:-1]) == {pd.Timedelta(minutes=15)}


def test_dst_spring_forward_keeps_constant_physical_cadence() -> None:
    times = [
        "27.03.2016 01:30",
        "27.03.2016 01:45",
        "27.03.2016 03:00",
        "27.03.2016 03:15",
    ]
    frame = _profile(times, [1.0] * len(times))

    utc_time = _parse_profile_time("load", frame)

    assert set(utc_time[1:] - utc_time[:-1]) == {pd.Timedelta(minutes=15)}


def test_profile_validator_rejects_wrong_physical_cadence() -> None:
    frame = pd.DataFrame(
        {
            "time": ["01.01.2016 00:00", "01.01.2016 00:30"],
            "plant": [1.0, 2.0],
        }
    )

    with pytest.raises(GridIntegrityError, match="constant physical"):
        _parse_profile_time("powerplants", frame)


def test_unused_time_only_profile_group_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _profile(NORMAL_TIMES)
    numeric_view = _numeric_profile_view("powerplants", frame)

    monkeypatch.setattr(adapter.sb, "get_applied_profiles", lambda *_: set())
    monkeypatch.setattr(
        adapter.sb,
        "get_available_profiles",
        lambda *_, **__: set(),
    )

    summary = _profile_group_summary(
        SimpleNamespace(),
        "powerplants",
        numeric_view,
    )

    assert summary.signal_columns == 0
    assert summary.applied_profiles == ()


def test_used_time_only_profile_group_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _profile(NORMAL_TIMES)
    numeric_view = _numeric_profile_view("powerplants", frame)

    monkeypatch.setattr(
        adapter.sb,
        "get_applied_profiles",
        lambda *_: {"PowerPlantProfileA"},
    )
    monkeypatch.setattr(
        adapter.sb,
        "get_available_profiles",
        lambda *_, **__: set(),
    )

    with pytest.raises(GridIntegrityError, match="contains no signal columns"):
        _profile_group_summary(
            SimpleNamespace(),
            "powerplants",
            numeric_view,
        )


def test_authoritative_profile_mapping_check_passes_when_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter.sb,
        "profiles_are_missing",
        lambda *_, **__: False,
    )

    _validate_profile_mapping_integrity(SimpleNamespace())


def test_authoritative_profile_mapping_check_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter.sb,
        "profiles_are_missing",
        lambda *_, **__: True,
    )

    with pytest.raises(GridIntegrityError, match="at least one profile"):
        _validate_profile_mapping_integrity(SimpleNamespace())
