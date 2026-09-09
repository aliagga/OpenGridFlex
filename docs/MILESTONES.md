# GridShiftBench gated execution plan

The project advances only when the current milestone's acceptance gate passes. A green gate means:
1. automated tests pass;
2. scientific sanity checks pass;
3. results are reproducible from a committed configuration;
4. artifacts and limitations are documented;
5. no known correctness issue is being silently ignored.

## M0 — Engineering and reproducibility foundation
**Goal:** make every later experiment traceable, testable and fail-fast.

Deliverables:
- validated YAML experiment/data schemas;
- deterministic seed utility with explicit limitations;
- run manifest: config hash, seed, Git state, Python/platform/package versions;
- unit-tested core shifts, metrics and split conformal utilities;
- CI, linting, coverage, contribution and release rules;
- simulator compatibility policy: known SimBench/pandapower issues must trigger an explicit validation failure or documented repair.

**Gate M0:** core unit tests pass; config self-check writes a valid manifest; no silent numerical NaN handling; repo has a documented quality policy.

## M1 — SimBench adapter and electrical integrity
**Goal:** load selected LV/MV networks and profiles without ambiguity and prove electrical consistency before ML.

Deliverables:
- exact supported SimBench codes and dataset/license metadata;
- network/profile loader with immutable raw-cache hashes;
- known transformer-tap compatibility validator;
- time-index validation, unit conversion audit, missing-profile audit;
- AC power-flow snapshots and time-series sanity checks;
- grid summary report (buses, lines, trafos, loads, DER, voltage levels, peak powers).

**Gate M1:** all configured grids load; raw/profile hashes recorded; no unexplained missing values; representative AC power flows converge; transformer tap behavior verified; engineering units documented.

## M2 — Leak-proof forecasting dataset contract
**Goal:** build forecasting samples with zero temporal/topological leakage.

Deliverables:
- target definition and horizon contract;
- feature availability matrix (what is known at prediction time);
- chronological train/validation/test policy;
- window generator tested at split boundaries;
- scaling fitted on train only;
- cross-feeder split protocol;
- dataset cards and sample-count checks.

**Gate M2:** automated leakage tests pass; boundary examples manually audited; independent rebuild yields identical sample hashes.

## M3 — Strong baseline suite
**Goal:** establish a baseline floor strong enough that later claims are credible.

Models:
- persistence/seasonal naive;
- linear/ridge where relevant;
- LightGBM point + quantile;
- one modern deep time-series baseline;
- one graph-temporal baseline.

**Gate M3:** all baselines run from the same interface/config; results repeat across seeds; naive and LightGBM sanity expectations hold; runtime/memory recorded; no novel model work begins before this gate is green.

## M4 — GridShiftBench shift engine
**Goal:** create deterministic, interpretable OOD scenarios rather than arbitrary noise.

Shift families:
- season/weather regime;
- DER penetration;
- topology/reconfiguration;
- sensor MCAR dropout plus structured outages;
- cross-feeder/domain transfer.

Each shift has severity, seed, metadata, invariants, and a physical plausibility test.

**Gate M4:** every shift is deterministic from config; original data remain immutable; shift magnitude is quantified; physical plausibility checks pass.

## M5 — Probabilistic forecasting and calibration
**Goal:** measure not only error but trustworthy uncertainty under shift.

Deliverables:
- quantile and/or distributional predictions;
- split conformal baseline;
- adaptive/shift-aware conformal candidate;
- coverage, interval width, interval score/CRPS where applicable;
- reliability curves stratified by shift severity.

**Gate M5:** IID coverage behaves as expected; OOD miscalibration is measurable; calibration code passes synthetic coverage tests; no test data used to calibrate.

## M6 — ShiftGuard-GT
**Goal:** introduce novelty only after the benchmark can falsify it.

Candidate components:
- graph-temporal encoder;
- multi-feeder training;
- masked/self-supervised pretraining;
- optional physics-consistency regularization;
- shift-aware/adaptive calibration outside the prediction backbone.

**Gate M6:** improvement is statistically and practically meaningful on predefined OOD metrics; every claimed component survives ablation; no performance claim depends on a favorable single seed or feeder.

## M7 — External validation, robustness and ablation freeze
**Goal:** prove conclusions are not SimBench-specific.

Deliverables:
- SMART-DS external subset;
- cross-dataset validation protocol;
- sensitivity to seeds, shift severity and training-data volume;
- calibration and architecture ablations;
- computational cost and parameter counts;
- failure-case analysis.

**Gate M7:** main qualitative conclusions reproduce externally; negative/failure cases are documented; final tables are generated automatically.

## M8 — Public release and manuscript freeze
**Goal:** make the work worthy of public scrutiny.

Deliverables:
- tagged release, CITATION.cff, license, changelog;
- data provenance and licenses;
- reproducible commands for every table/figure;
- CI passing from a clean checkout;
- archival DOI plan;
- manuscript, supplement and benchmark model card.

**Gate M8:** a second machine/environment can reproduce the headline result from the released instructions without manual intervention.
