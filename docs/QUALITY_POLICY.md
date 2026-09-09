# Scientific and software quality policy

OpenGridFlex is designed as research infrastructure, not disposable manuscript code.

## Non-negotiable rules
- **Fail loudly:** invalid units, shapes, NaNs, missing profiles, non-convergent power flow, unknown simulator compatibility, and data leakage are errors unless explicitly handled by a documented method.
- **Immutable raw data:** raw downloaded/generated source data are never overwritten. Derived data carry provenance and hashes.
- **No test-set tuning:** model selection, preprocessing, calibration and threshold selection use training/validation data only.
- **Configuration is the experiment:** every result must be generated from a versioned configuration.
- **Trace every result:** every run stores seed, config hash, Git commit/dirty state, package versions and platform information.
- **Strong baselines first:** novel models are not evaluated without naive, classical ML and representative modern baselines.
- **Report failures:** negative cases, non-convergence, calibration failure and sensitivity are part of the scientific result.
- **Physics before aesthetics:** no model result is accepted if the underlying grid/data pipeline fails electrical sanity checks.

## Simulator compatibility
Known upstream issues are never silently worked around. A workaround must:
1. be triggered by an explicit validator;
2. be documented with the upstream issue/reference;
3. change only the affected field(s);
4. be covered by a regression test;
5. be recorded in the run manifest or dataset manifest.
