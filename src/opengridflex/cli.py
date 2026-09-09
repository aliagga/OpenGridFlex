from __future__ import annotations

import argparse
import json
from pathlib import Path

from opengridflex.config import load_experiment_config
from opengridflex.reproducibility import build_run_manifest, seed_everything, write_run_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cmd_validate_config(args: argparse.Namespace) -> int:
    exp, data = load_experiment_config(args.config, repo_root=args.repo_root)
    print(json.dumps({"experiment": exp.experiment, "data": data.dataset, "grids": data.grids}, indent=2))
    return 0


def cmd_self_check(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    exp, _ = load_experiment_config(args.config, repo_root=root)
    state = seed_everything(args.seed, deterministic_torch=not args.allow_nondeterminism)
    manifest = build_run_manifest(
        experiment=exp.experiment, seed=args.seed, config_path=args.config, repo_root=root
    )
    out = write_run_manifest(manifest, Path(args.output) / "run_manifest.json")
    print(json.dumps({"seed_state": state, "manifest": str(out)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opengridflex", description="OpenGridFlex research benchmark tooling")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-config", help="Validate experiment and linked data configuration")
    v.add_argument("config")
    v.add_argument("--repo-root", default=str(Path.cwd()))
    v.set_defaults(func=cmd_validate_config)

    s = sub.add_parser("self-check", help="Check reproducibility setup and write a run manifest")
    s.add_argument("config")
    s.add_argument("--repo-root", default=str(Path.cwd()))
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--output", default="artifacts/self_check")
    s.add_argument("--allow-nondeterminism", action="store_true")
    s.set_defaults(func=cmd_self_check)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
