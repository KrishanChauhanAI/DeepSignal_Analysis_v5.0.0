"""Preflight report contract and run-directory safety (overwrite refusal, environment capture)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
import yaml

from trapsig.config import config_from_dict
from trapsig.preflight import Status, run_preflight
from trapsig.runs import (
    OverwriteError,
    capture_environment,
    create_run_dir,
    write_json_exclusive,
    write_text_exclusive,
)
from trapsig.synthetic import SyntheticSpec, write_synthetic_dataset

REPO = Path(__file__).resolve().parents[1]
ALL_IDS = {f"P{i:02d}" for i in range(1, 27)}


def _base(runs_dir: Path) -> dict:
    return {
        "run": {"name": "pf", "seed": 1},
        "data": {"file_glob": "*.csv", "label_level": "file", "label_from": "metadata"},
        "sampling": {"original_hz": 100000, "target_rates_hz": [10000, 1000, 100, 10]},
        "filters": {"mode": "lpf", "lpf_cutoff_hz": 200.0, "notch_hz": [50]},
        "events": {"baseline_window_s": 0.1, "pre_event_window_s": 0.02, "min_duration_s": 0.001},
        "splitting": {"n_splits": 5, "n_repeats": 2, "min_groups_per_class": 5},
        "output": {"runs_dir": str(runs_dir)},
    }


def _status_map(rep) -> dict[str, Status]:
    return {c.check_id: c.status for c in rep.checks}


# ---- preflight -----------------------------------------------------------------------------

def test_default_config_fails_only_on_missing_inputs(tmp_path: Path):
    raw = yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg = config_from_dict(raw)
    rep = run_preflight(cfg)
    st = _status_map(rep)
    assert st["P05"] is Status.ERROR          # source_dir unset
    assert st["P07"] is Status.ERROR          # label level unknown
    assert st["P08"] is Status.ERROR          # label source none
    assert rep.errors, "default config must not pass"
    assert {c.check_id for c in rep.checks} <= ALL_IDS


def test_every_status_is_from_the_allowed_set(tmp_path: Path):
    cfg = config_from_dict(_base(tmp_path / "runs"))
    rep = run_preflight(cfg)
    assert all(isinstance(c.status, Status) for c in rep.checks)
    assert {c.check_id for c in rep.checks} <= ALL_IDS


def test_missing_source_dir_is_error_and_data_checks_are_skipped(tmp_path: Path):
    raw = _base(tmp_path / "runs")
    rep = run_preflight(config_from_dict(raw))
    st = _status_map(rep)
    assert st["P05"] is Status.ERROR
    for cid in ("P06", "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17"):
        assert st[cid] is Status.SKIPPED, cid


def test_consistent_synthetic_project_has_no_errors(tmp_path: Path):
    data = tmp_path / "data"
    meta = write_synthetic_dataset(data, SyntheticSpec(replicates_per_class=6, duration_s=0.5, seed=3))
    raw = _base(tmp_path / "runs")
    raw["data"].update({
        "source_dir": str(data), "metadata_csv": str(meta),
        "expected_duration_s": 0.5, "duration_tolerance_s": 0.001,
    })
    rep = run_preflight(config_from_dict(raw))
    assert not rep.errors, [c.message for c in rep.errors]
    st = _status_map(rep)
    assert st["P06"] is Status.PASS
    assert st["P11"] is Status.PASS


def test_preflight_writes_nothing(tmp_path: Path):
    raw = _base(tmp_path / "runs")
    run_preflight(config_from_dict(raw))
    assert not (tmp_path / "runs").exists()


# ---- run directories and exclusive writes -------------------------------------------------

def test_run_dir_name_is_timestamped_and_never_reused(tmp_path: Path):
    now = dt.datetime(2026, 10, 8, 7, 30, 0)
    a = create_run_dir(tmp_path, kind="training", now=now)
    b = create_run_dir(tmp_path, kind="training", now=now)
    assert a.name == "training_2026-10-08_07-30-00"
    assert b != a and b.exists() and a.exists()


def test_run_stamp_extracts_the_timestamp_and_drops_collision_suffixes():
    from trapsig.runs import run_stamp

    assert run_stamp("train_2026-10-09_06-10-08") == "2026-10-09_06-10-08"
    assert run_stamp("train_2026-10-09_06-10-08_001") == "2026-10-09_06-10-08"
    assert run_stamp(Path("runs/dryrun_2026-10-09_06-10-08")) == "2026-10-09_06-10-08"


def test_prepared_dataset_lives_inside_its_run_folder(tmp_path: Path):
    from trapsig.runs import PREPARED_SUBDIRS, RUN_SUBDIRS, create_prepared_dir, create_run_dir

    run = create_run_dir(tmp_path, kind="train")
    prepared = create_prepared_dir(run)
    assert prepared.parent == run, "the dataset is nested, never a parallel folder"
    assert prepared.name == "training_" + run.name.split("train_", 1)[1]
    for sub in PREPARED_SUBDIRS:
        assert (prepared / sub).is_dir()
    for sub in RUN_SUBDIRS:
        assert (run / sub).is_dir()
    assert not (run / "data").exists(), "the run folder holds no second dataset skeleton"


def test_prepared_dataset_folder_is_never_reused(tmp_path: Path):
    from trapsig.runs import OverwriteError, create_prepared_dir, create_run_dir

    run = create_run_dir(tmp_path, kind="train")
    create_prepared_dir(run)
    with pytest.raises(OverwriteError):
        create_prepared_dir(run)


def test_exclusive_text_write_refuses_overwrite(tmp_path: Path):
    p = tmp_path / "x.txt"
    write_text_exclusive(p, "first")
    with pytest.raises((OverwriteError, FileExistsError)):
        write_text_exclusive(p, "second")
    assert p.read_text(encoding="utf-8") == "first"


def test_exclusive_json_write_refuses_overwrite(tmp_path: Path):
    p = tmp_path / "x.json"
    write_json_exclusive(p, {"a": 1})
    with pytest.raises((OverwriteError, FileExistsError)):
        write_json_exclusive(p, {"a": 2})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}


# ---- environment capture ------------------------------------------------------------------

def test_capture_environment_has_required_fields(tmp_path: Path):
    env = capture_environment(tmp_path)
    for key in ("timestamp_local", "python", "python_executable", "platform", "cpu_count",
                "packages", "gpu", "git"):
        assert key in env, key
    assert set(env["packages"]) >= {"numpy", "scipy", "scikit-learn", "PyYAML", "trapsig"}
    assert env["gpu"]["torch_installed"] in (True, False)
    json.dumps(env, default=str)  # must be serialisable
