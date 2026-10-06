# Changelog

## Unreleased
- Integrated M3.2 LightGBM artifacts with the existing repository-wide reproducibility manifest.
- Made the M3.2 gate persist effective configuration, run provenance and validation artifacts automatically.
- Fixed GitHub Actions to install the optional grid stack required by the test suite.
- Updated the repository status documentation to reflect progress through M3.2.

## 0.2.0.dev0 — Milestone 0 foundation
- Added validated experiment/data configuration schemas.
- Added deterministic seeding with explicit reproducibility limitations.
- Added run provenance manifests with config hash, environment and package versions.
- Hardened chronological splitting, controlled shift primitives, metrics and split conformal utilities.
- Added 15 unit tests and CI definition.
- Added gated research milestones and scientific/software quality policy.
- Added a fail-closed Milestone-0 acceptance script.
