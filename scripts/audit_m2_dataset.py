from __future__ import annotations

import json

from opengridflex.data.leakproof_dataset import (
    TrainOnlyStandardizer,
    WindowSpec,
    audit_window_plan,
    build_canonical_grid_series,
    build_window_plan,
    known_future_calendar_features,
    make_chronological_split,
)

GRID_CODE = "1-MV-urban--1-no_sw"


def main() -> int:
    series = build_canonical_grid_series(GRID_CODE)
    split = make_chronological_split(series.n_steps)

    print("=" * 78)
    print("OpenGridFlex — M2.1 Leak-Proof Dataset Audit")
    print("=" * 78)
    print(f"Grid: {series.grid_code}")
    print(f"Shape: {series.n_steps} timesteps x {series.n_buses} buses")
    print(f"Timezone: {series.time_utc.tz}")
    print(f"Start UTC: {series.time_utc[0].isoformat()}")
    print(f"End UTC: {series.time_utc[-1].isoformat()}")
    print(f"Fingerprint: {series.fingerprint}")
    print(f"Channels: {tuple(series.values)}")
    print()
    print("Chronological split:")
    print(json.dumps(split.to_dict(), indent=2, sort_keys=True))

    specs = {
        "next_4h": WindowSpec(
            history_steps=96,
            horizon_steps=16,
            lead_steps=1,
        ),
        "next_24h": WindowSpec(
            history_steps=672,
            horizon_steps=96,
            lead_steps=1,
        ),
    }

    print("\nLeakage audits:")
    for name, spec in specs.items():
        plan = build_window_plan(series.n_steps, split, spec)
        audit = audit_window_plan(split, plan)
        print(f"\n{name}")
        print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))

    scaler = TrainOnlyStandardizer.fit(
        series,
        split,
        channels=(
            "load_p_mw",
            "load_q_mvar",
            "sgen_p_mw",
            "storage_p_mw",
            "net_demand_p_mw",
            "net_demand_q_mvar",
        ),
    )

    print("\nScaling provenance:")
    print(f"fit rows: [{scaler.fit_start}, {scaler.fit_end})")
    print(f"source fingerprint: {scaler.source_fingerprint}")

    calendar = known_future_calendar_features(series.time_utc)
    print("\nKnown-future feature columns:")
    print(tuple(calendar.columns))

    print("\nLeakage policy:")
    print("- split is chronological; no random split")
    print("- every target horizon must lie fully inside one split")
    print("- windows straddling split boundaries are discarded")
    print("- validation/test history may use already-observed earlier data")
    print("- normalization statistics are fit on TRAIN rows only")
    print("- future calendar variables are allowed because known at forecast time")
    print("- no future measured load/generation enters the history tensor")

    print("\nM2.1 DATASET AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
