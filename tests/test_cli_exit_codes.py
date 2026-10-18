"""Exit-code contract: 0 = success, 1 = preflight/preparation ERROR, 2 = configuration problem.

Argparse usage errors also exit with 2 (via SystemExit), which is the same contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trapsig.cli import main
from trapsig.synthetic import SyntheticSpec, write_synthetic_dataset

REPO = Path(__file__).resolve().parents[1]


def _exit_code(argv: list[str]) -> int:
    try:
        return main(argv)
    except SystemExit as exc:  # argparse raises SystemExit on usage errors
        return int(exc.code or 0)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    data = tmp_path / "data" / "SYN"
    meta = write_synthetic_dataset(data, SyntheticSpec(replicates_per_class=6, duration_s=0.5, seed=3))
    cfg = {
        "run": {"name": "exit", "seed": 1},
        "data": {
            "source_dir": str(data), "file_glob": "*.csv", "label_level": "file",
            "label_from": "metadata", "metadata_csv": str(meta),
            "expected_duration_s": 0.5, "duration_tolerance_s": 0.001,
        },
        "sampling": {"original_hz": 100000, "target_rates_hz": [10000, 1000, 100, 10]},
        "filters": {"mode": "lpf", "lpf_cutoff_hz": 200.0, "notch_hz": [50]},
        "events": {"baseline_window_s": 0.1, "pre_event_window_s": 0.02, "min_duration_s": 0.001},
        "splitting": {"n_splits": 5, "n_repeats": 2, "min_groups_per_class": 5},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_dry_run_exit_0_when_no_errors(project: Path):
    assert _exit_code(["dry-run", "--config", str(project)]) == 0


def test_dry_run_exit_1_on_preflight_error():
    # configs/default.yaml leaves source_dir unset, which is a deliberate preflight ERROR (P05).
    assert _exit_code(["dry-run", "--config", str(REPO / "configs" / "default.yaml")]) == 1


def test_dry_run_exit_2_for_missing_config(tmp_path: Path):
    assert _exit_code(["dry-run", "--config", str(tmp_path / "nope.yaml")]) == 2


def test_dry_run_exit_2_for_invalid_yaml(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: [unclosed\n", encoding="utf-8")
    assert _exit_code(["dry-run", "--config", str(bad)]) == 2


def test_dry_run_exit_2_for_unknown_key(tmp_path: Path):
    bad = tmp_path / "typo.yaml"
    bad.write_text("run:\n  nmae: typo\n", encoding="utf-8")
    assert _exit_code(["dry-run", "--config", str(bad)]) == 2


def test_prepare_exit_1_creates_no_run_directory(tmp_path: Path):
    runs = tmp_path / "runs"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"output": {"runs_dir": str(runs)}}), encoding="utf-8")
    assert _exit_code(["prepare", "--config", str(cfg)]) == 1
    assert not runs.exists() or not any(runs.iterdir())


def test_prepare_exit_2_for_missing_config(tmp_path: Path):
    assert _exit_code(["prepare", "--config", str(tmp_path / "missing.yaml")]) == 2


def test_events_exit_2_for_rate_above_original(project: Path):
    assert _exit_code(["events", "--config", str(project), "--rate-hz", "1000000"]) == 2


def test_unknown_subcommand_is_usage_error_2():
    assert _exit_code(["not-a-command"]) == 2


def test_inspect_dat_exit_0_and_read_only(tmp_path: Path, capsys):
    p = tmp_path / "x.dat"
    p.write_bytes(b"\x00\x00\x80\x3f" * 500)  # 500 float32 values equal to 1.0
    before = p.read_bytes()
    assert _exit_code(["inspect-dat", "--file", str(p), "--rate-hz", "1000", "--duration-s", "0.5"]) == 0
    assert p.read_bytes() == before
    assert "float32" in capsys.readouterr().out


def test_inspect_dat_exit_2_for_missing_file(tmp_path: Path):
    assert _exit_code(["inspect-dat", "--file", str(tmp_path / "nope.dat"), "--rate-hz", "1000"]) == 2
