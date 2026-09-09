# Contributing to OpenGridFlex

All scientific changes must include tests and a reproducible configuration where applicable.

Before opening a pull request:

```bash
ruff check src tests
pytest --cov=opengridflex --cov-fail-under=85
python -m opengridflex.cli validate-config configs/paper1/mvp.yaml --repo-root .
```

Changes affecting datasets, grid conversion, simulator behavior, split logic, forecasting targets, metrics or calibration require a regression test because these can alter scientific conclusions without obvious runtime failures.
