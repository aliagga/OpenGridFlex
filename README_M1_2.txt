OpenGridFlex M1.2 patch

Copy the three files into the same relative paths in your local repository:

src/opengridflex/grids/simbench_adapter.py
tests/integration/test_simbench_adapter.py
scripts/check_m1_2.py

Then update pyproject.toml in BOTH [project.optional-dependencies] grid and all:
  pandapower==3.5.4
  simbench==1.6.2
  numba==0.67.0

Install:
  pip install -e ".[grid,dev]"

Run:
  ruff format src tests scripts
  ruff check src tests scripts
  pytest -q
  python scripts/check_m1_2.py

Do not proceed unless the final line is:
  M1.2 GATE: GREEN
