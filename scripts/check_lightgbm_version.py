from importlib.metadata import version

EXPECTED = "4.7.0"
observed = version("lightgbm")

if observed != EXPECTED:
    raise SystemExit(f"BLOCKED: lightgbm=={EXPECTED} required, found {observed}")

print(f"PASS: lightgbm=={observed}")
