OpenGridFlex M3.3 — PatchTST modern temporal baseline

PREREQUISITES
  M3.2 GATE: GREEN
  gridshiftbench-v1 benchmark contract frozen
  primary task only: gsb_bus_pq_4h_v1
  test split remains sealed

DEPENDENCY
  neuralforecast==3.2.2

Install:
  pip install -r requirements-m3_3.txt

MODEL
  PatchTST from NeuralForecast.

Why this baseline:
  - modern Transformer-based time-series reference model;
  - maintained open-source implementation;
  - channel-independent architecture suited to many bus series;
  - CPU execution is kept as the frozen default for reproducibility.

M3.3 V1 SCOPE
  Deterministic point forecasting only.

Channels are evaluated separately:
  net_demand_p_mw
  net_demand_q_mvar

Uncertainty/calibration is intentionally not added here. That belongs to M5.

FROZEN PRIMARY TASK
  history: 96 x 15 min = 24 h
  horizon: 16 x 15 min = 4 h
  validation rolling origins: 5,269
  stride: 1
  test split: SEALED

ANTI-LEAKAGE DESIGN
NeuralForecast receives only data ending at the frozen validation end.
Its no-refit cross-validation holdout starts exactly at the frozen
validation boundary.

Full validation geometry:
  validation_start = 24572
  validation_end   = 29856
  test_size        = 5284
  n_windows        = 5269

For development smoke runs, the panel is truncated inside validation so
the training boundary remains unchanged. Limiting validation windows never
allows validation observations to become training observations.

FROZEN DEFAULT CONFIGURATION
  seed=42
  max_steps=500
  encoder_layers=3
  n_heads=8
  hidden_size=128
  linear_hidden_size=256
  patch_len=16
  stride=8
  dropout=0.20
  fc_dropout=0.20
  learning_rate=1e-4
  batch_size=32
  windows_batch_size=512
  inference_windows_batch_size=1024
  scaler_type=identity
  RevIN=true
  accelerator=cpu
  devices=1

DEVELOPMENT GATE
  ruff check src tests scripts
  pytest -q
  python scripts/check_m3_3.py

Required final line:
  M3.3 DEVELOPMENT GATE: GREEN

This does NOT freeze M3.3. It verifies the dependency, API integration,
benchmark geometry, leakage safety and a tiny end-to-end PatchTST fit.

FULL VALIDATION
Run both P and Q on all 5,269 validation origins:

  python scripts/run_m3_3_patchtst.py \
    --output-dir artifacts/gates/m3_3_patchtst \
    --overwrite

Required final line:
  M3.3 PATCHTST FULL VALIDATION: PASS

Full artifacts:
  effective_run_config.json
  run_manifest.json
  validation_manifest.json

RELEASE-GREEN REVIEW
Before freezing M3.3 inspect:
  - P and Q MAE/RMSE/WAPE;
  - PatchTST versus the strongest M3.1 naive baseline;
  - PatchTST versus M3.2 LightGBM;
  - runtime and parameter count;
  - repeatability under the same seed;
  - run_manifest Git state must be clean.

Do not open M4 until M3.3 full validation is reviewed.
