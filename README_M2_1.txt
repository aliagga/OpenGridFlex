OpenGridFlex M2.1 — Leak-proof canonical forecasting dataset

PREREQUISITE:
  M1.3 GATE: GREEN

ADD:
  src/opengridflex/data/leakproof_dataset.py
  tests/unit/test_leakproof_dataset_unit.py
  tests/integration/test_leakproof_dataset_integration.py
  scripts/audit_m2_dataset.py
  scripts/check_m2_1.py

Before adding M2, freeze M1.3:
  git add .
  git commit -m "Complete M1.3 physical profile reconstruction and AC validation"
  git tag m1.3-green

Canonical bus-level channels:
  load_p_mw
  load_q_mvar
  sgen_p_mw
  gen_p_mw
  storage_p_mw
  net_demand_p_mw
  net_demand_q_mvar

Sign convention:
  net_demand_p_mw = load + storage - sgen - gen
  Positive net demand means net consumption from the grid.

Leakage contract:
1. UTC, unique, monotonic 15-minute physical timeline.
2. Chronological train / validation / test split.
3. No random splitting.
4. Forecast windows are assigned by their TARGET interval.
5. A target horizon that crosses a split boundary is discarded.
6. History always ends strictly before target start.
7. Validation/test history may include earlier observations that would genuinely
   be available at inference time.
8. Standardization statistics are fit ONLY on training rows.
9. Calendar features are future-known only (clock/calendar/DST/UTC offset).
10. No future measured power is permitted in the history tensor.
11. Canonical dataset is SHA-256 fingerprinted.

Default audit:
  train      70%
  validation 15%
  test       15%

Window configurations audited:
  next 4h:
    24h history, 4h horizon, 15-min resolution
  next 24h:
    7-day history, 24h horizon, 15-min resolution

Run:
  ruff check src tests scripts --fix
  ruff format src tests scripts
  ruff check src tests scripts
  pytest -q
  python scripts/check_m2_1.py

Required final line:
  M2.1 GATE: GREEN

After GREEN:
  git add .
  git commit -m "Complete M2.1 leak-proof canonical forecasting dataset"
  git tag m2.1-green

Next:
  M2.2 will freeze task definitions, target channels, serialization, and
  experiment manifests before any baseline model is trained.
