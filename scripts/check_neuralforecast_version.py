from importlib.metadata import version

from opengridflex.baselines.patchtst_global import (
    NEURALFORECAST_VERSION,
    require_neuralforecast_version,
)


def main() -> int:
    observed = require_neuralforecast_version()
    assert observed == version("neuralforecast")
    print(f"PASS: neuralforecast=={NEURALFORECAST_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
