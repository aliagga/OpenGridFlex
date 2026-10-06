from __future__ import annotations

import json
from pathlib import Path

from opengridflex import cli

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "paper1" / "mvp.yaml"


def test_repo_root_points_to_repository() -> None:
    assert (cli._repo_root() / "pyproject.toml").exists()


def test_validate_config_command(capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "validate-config",
            str(CONFIG),
            "--repo-root",
            str(ROOT),
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment"]
    assert payload["data"]
    assert payload["grids"]


def test_self_check_command_writes_manifest(tmp_path, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "self-check",
            str(CONFIG),
            "--repo-root",
            str(ROOT),
            "--seed",
            "123",
            "--output",
            str(tmp_path),
            "--allow-nondeterminism",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    manifest = Path(payload["manifest"])

    assert manifest.exists()
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert stored["seed"] == 123
    assert stored["experiment"] == payload["seed_state"].get("experiment", stored["experiment"])
    assert stored["config_sha256"]


def test_main_dispatches_validate_config(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "opengridflex",
            "validate-config",
            str(CONFIG),
            "--repo-root",
            str(ROOT),
        ],
    )

    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment"]
