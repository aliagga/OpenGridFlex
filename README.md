# OpenGridFlex

**Trustworthy AI benchmarking for active distribution grids under distribution shift.**

OpenGridFlex is being built as reusable research infrastructure for GridShiftBench rather than a one-off manuscript repository. Its central scientific question is whether forecasting and uncertainty methods that look strong under IID evaluation remain accurate, calibrated and transferable under realistic distribution-grid shifts.

## Paper 1: GridShiftBench
Planned shift families:
- seasonal/weather regime shift;
- DER penetration shift;
- topology/reconfiguration shift;
- sensor dropout and structured outage shift;
- cross-feeder and cross-dataset transfer.

Planned model ladder:
1. persistence / seasonal naive;
2. classical linear baseline where appropriate;
3. LightGBM point and quantile models;
4. modern temporal deep baseline;
5. graph-temporal baseline;
6. ShiftGuard-GT candidate.

## Quality contract
Every manuscript table/figure must originate from a versioned config and result artifact. Raw data are immutable, split/calibration leakage is prohibited, simulator compatibility is validated explicitly, and every run records provenance (config hash, Git state, seed, packages and platform).

See:
- `docs/MILESTONES.md` for the gated build plan;
- `docs/QUALITY_POLICY.md` for non-negotiable scientific/software rules.

## Core checks

```bash
PYTHONPATH=src python -m opengridflex.cli validate-config configs/paper1/mvp.yaml --repo-root .
PYTHONPATH=src python -m opengridflex.cli self-check configs/paper1/mvp.yaml --repo-root .
PYTHONPATH=src pytest -q
```

## Current status
**Milestone 0 — engineering and reproducibility foundation.** Grid/data integration is intentionally blocked until this gate is green.
