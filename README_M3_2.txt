OpenGridFlex M3.2 — Global LightGBM deterministic + quantile baseline

PREREQUISITES
  M3.1 GATE: GREEN
  M2.2 benchmark contract frozen

INSTALL
  pip install -r requirements-m3_2.txt

Pinned dependency:
  lightgbm==4.7.0

M3.2 SCOPE
This milestone trains the first learned baseline on the PRIMARY task only:
  gsb_bus_pq_4h_v1

The secondary 24-hour task remains frozen in M2.2 but is not opened in this
milestone. We first validate learned-model design, leakage safety, metrics,
reproducibility and computational behavior on the primary benchmark.

GLOBAL DIRECT MODEL
Each training row represents:
  (forecast origin, bus, future horizon step)

One global LightGBM model therefore shares information across buses and all
16 forecast steps instead of fitting thousands of separate models.

TARGET CHANNELS
Models remain separate for:
  net_demand_p_mw
  net_demand_q_mvar

For each channel:
  1 point-regression model
  7 quantile-regression models:
    0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95

Total:
  16 fitted LightGBM models.

FEATURES
41 features, including:
  bus identity
  forecast horizon step
  observed P/Q at the forecast origin
  1 h / 4 h / 24 h / 7 d causal lags
  target-aligned 24 h and 7 d seasonal lags
  causal 4 h / 24 h / 7 d rolling means
  recent trend
  bus voltage level
  line / transformer degree
  counts of load / sgen / storage elements
  installed load / sgen / storage capacity
  future-known calendar and DST variables

ANTI-LEAKAGE RULE
Measured features must always satisfy:
  source_timestamp <= forecast_origin

For the frozen primary 4-hour task, target-aligned 24 h and 7 d lags are
provably historical. The code fails closed if the horizon is changed so that
the 24-hour lag could enter the future.

Future measured load/generation/storage are never features.
Future calendar variables are allowed because they are known at issuance.

TRAINING
Default development configuration:
  seed=42
  maximum sampled training rows=150,000
  trees=300
  learning_rate=0.05
  num_leaves=63
  min_child_samples=100
  row subsample=0.80
  column subsample=0.90

The Cartesian train set is sampled reproducibly to keep this baseline usable
on a research workstation. The full validation split is still evaluated.

REPRODUCIBILITY
LightGBM:
  deterministic=true
  force_col_wise=true
  random_state=42
  exact LightGBM version gate

TEST SET
SEALED.
M3.2 evaluation accepts validation only. Any request for test evaluation fails.

METRICS
Point:
  MAE
  RMSE
  WAPE
  bus micro / active-bus macro / system / horizon step

Probabilistic:
  mean pinball loss
  pinball per quantile
  90% empirical coverage
  90% interval width
  absolute coverage error
  adjacent quantile crossing rate

Quantile crossing is REPORTED, not silently repaired. Calibration and monotone
uncertainty handling belong to later trustworthy-uncertainty milestones.

MODEL ARTIFACTS
Persisted output contains:
  LightGBM text models
  validation metrics
  feature importance
  model-file SHA-256 hashes
  source dataset fingerprint
  exact effective run configuration
  repository-wide run_manifest.json
  Git commit / dirty state
  Python, platform and tracked package versions
  LightGBM version

RUN
  pip install -r requirements-m3_2.txt

  ruff check src tests scripts --fix
  ruff format src tests scripts
  ruff check src tests scripts

  pytest -q

  python scripts/check_m3_2.py

Required final line:
  M3.2 GATE: GREEN

To persist the full validation artifact after GREEN:
  python scripts/run_m3_2_lightgbm.py --output-dir artifacts/baselines/m3_2_lightgbm --overwrite

After reviewing the validation metrics:
  git add .
  git commit -m "Complete M3.2 global LightGBM baseline"
  git tag m3.2-green

IMPORTANT
Do not move on only because the gate is technically green. Before M3.3 we
inspect:
  - learned MAE/RMSE/WAPE vs best M3.1 naive baseline
  - P vs Q behavior
  - interval coverage and width
  - quantile crossing
  - dominant feature importances

NEXT
If M3.2 is both technically valid and competitive:
  M3.3 = deep temporal baseline (PatchTST/TSMixer family), still validation-only.
