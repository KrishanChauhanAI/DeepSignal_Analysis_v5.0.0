"""End-to-end checks on SYNTHETIC data. These verify plumbing, not science."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from trapsig.cli import main
from trapsig.synthetic import SyntheticSpec, write_synthetic_dataset


@pytest.fixture
def synthetic_project(tmp_path: Path) -> dict[str, Path]:
    data = tmp_path / "data" / "SYN"
    meta = write_synthetic_dataset(data, SyntheticSpec(replicates_per_class=6, duration_s=0.5, seed=3))
    runs = tmp_path / "runs"
    cfg = {
        "run": {"name": "e2e", "seed": 7},
        "data": {
            "source_dir": str(data),
            "file_glob": "*.csv",
            "label_level": "file",
            "label_from": "metadata",
            "metadata_csv": str(meta),
            "expected_duration_s": 0.5,
            "duration_tolerance_s": 0.001,
        },
        "sampling": {"original_hz": 100000, "target_rates_hz": [10000, 1000, 100, 10]},
        "filters": {"mode": "lpf", "lpf_cutoff_hz": 200.0, "notch_hz": [50]},
        "events": {"baseline_window_s": 0.1, "pre_event_window_s": 0.02, "min_duration_s": 0.001},
        "splitting": {"n_splits": 5, "n_repeats": 2, "min_groups_per_class": 5},
        "output": {"runs_dir": str(runs)},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return {"config": cfg_path, "data": data, "runs": runs, "meta": meta}


def _only_run(runs: Path, prefix: str) -> Path:
    found = sorted(p for p in runs.iterdir() if p.name.startswith(prefix))
    assert len(found) == 1, found
    return found[0]


def test_dry_run_on_synthetic_data_passes_and_writes_report(synthetic_project, capsys):
    code = main(["dry-run", "--config", str(synthetic_project["config"])])
    assert code == 0
    run = _only_run(synthetic_project["runs"], "dryrun_")
    report = json.loads((run / "reports" / "preflight_report.json").read_text())
    assert report["exit_code"] == 0
    ids = {c["id"]: c["status"] for c in report["checks"]}
    assert ids["P16"] == "PASS"
    assert ids["P14"] == "PASS"
    assert "RESULT: PASSED" in capsys.readouterr().out


def test_dry_run_default_config_fails_safely_without_data(tmp_path, capsys):
    repo_default = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = yaml.safe_load(repo_default.read_text(encoding="utf-8"))
    cfg["output"]["runs_dir"] = str(tmp_path / "runs")
    p = tmp_path / "default_copy.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    code = main(["dry-run", "--config", str(p)])
    assert code == 1
    out = capsys.readouterr().out
    assert "RESULT: FAILED" in out
    assert "data.source_dir is not set" in out


def test_prepare_writes_manifest_folds_and_links_without_touching_sources(synthetic_project):
    before = {p.name: p.read_bytes() for p in synthetic_project["data"].iterdir()}
    code = main(["prepare", "--config", str(synthetic_project["config"])])
    assert code == 0
    # one run folder per run, with the prepared dataset nested inside it
    run = _only_run(synthetic_project["runs"], "train_")
    prepared = _only_run(run, "training_")

    manifest = list(csv.DictReader((prepared / "data/manifests/master_manifest.csv").open(encoding="utf-8")))
    assert len(manifest) == 12
    assert all(r["split"] == "cv_pool" for r in manifest)
    assert all(r["link_status"] == "linked" for r in manifest)

    folds = list(csv.DictReader((prepared / "data/manifests/fold_assignments.csv").open(encoding="utf-8")))
    assert {f["fold_r0"] for f in folds} <= {"0", "1", "2", "3", "4"}
    # group disjointness across folds for each repeat
    for rep in ("fold_r0", "fold_r1"):
        by_group: dict[str, set[str]] = {}
        for f in folds:
            by_group.setdefault(f["group_id"], set()).add(f[rep])
        assert all(len(v) == 1 for v in by_group.values())

    after = {p.name: p.read_bytes() for p in synthetic_project["data"].iterdir()}
    assert before == after
    summary = json.loads((prepared / "artifacts/run_summary.json").read_text())
    assert summary["n_cv_pool"] == 12
    assert summary["run_dir"] == str(run) and summary["prepared_dir"] == str(prepared)
    # the split record of the prepared dataset lives with the dataset
    assert (prepared / "metadata.csv").is_file() and (prepared / "metadata.json").is_file()
    # no parallel dataset folder next to the run folder
    assert not list(synthetic_project["runs"].glob("training_*"))


def test_prepare_refuses_when_preflight_fails(tmp_path, capsys):
    cfg = {"data": {"source_dir": str(tmp_path / "missing")}, "output": {"runs_dir": str(tmp_path / "r")}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert main(["prepare", "--config", str(p)]) == 1
    assert not (tmp_path / "r").exists() or not any((tmp_path / "r").iterdir())


def test_events_command_detects_synthetic_events_and_records_filter(synthetic_project, capsys):
    code = main(["events", "--config", str(synthetic_project["config"]), "--rate-hz", "1000"])
    assert code == 0
    run = _only_run(synthetic_project["runs"], "events_")
    rows = list(csv.DictReader((run / "metrics/events.csv").open(encoding="utf-8")))
    assert len(rows) >= 12  # at least one event per recording (1-3 injected)
    assert all(float(r["onset_s"]) >= 0.01 for r in rows)  # start guard respected
    assert all(r["filter"] == "Filter: LPF (LPF Butterworth order 4, fc=200 Hz, zero-phase)" for r in rows)
    depths = [float(r["delta_i_over_i0"]) for r in rows]
    assert all(d < 0 for d in depths)  # blockades are negative (assumption A2)
    summary = json.loads((run / "metrics/events_summary.json").read_text())
    assert summary["event_detection_rate_hz"] == 1000
    assert "not classified" in summary["note"]


def test_events_rejects_rate_above_original(synthetic_project):
    assert main(["events", "--config", str(synthetic_project["config"]), "--rate-hz", "1e9"]) == 2


def test_configuration_error_returns_exit_code_2(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text("events:\n  threshold_sigmaa: 5\n", encoding="utf-8")
    assert main(["dry-run", "--config", str(p)]) == 2
    assert "CONFIGURATION ERROR" in capsys.readouterr().err


def test_run_creates_one_folder_holding_both_the_dataset_and_the_results(synthetic_project):
    """`trapsig run --quick` must not leave a train_ folder next to a training_ folder."""
    code = main(["run", "--config", str(synthetic_project["config"]), "--quick"])
    assert code == 0
    runs = synthetic_project["runs"]
    tops = sorted(p for p in runs.iterdir() if p.is_dir())
    assert len(tops) == 1, tops
    run = tops[0]
    assert run.name.startswith("train_")
    assert not list(runs.glob("training_*")), "no parallel dataset folder"
    prepared = _only_run(run, "training_")
    assert (prepared / "data/manifests/master_manifest.csv").is_file()
    assert (prepared / "metadata.csv").is_file()
    assert (run / "summary.json").is_file() and (run / "metadata.csv").is_file()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["prepared_run"] == str(prepared)


def test_train_reuses_the_run_folder_of_the_prepared_dataset(synthetic_project):
    runs = synthetic_project["runs"]
    assert main(["prepare", "--config", str(synthetic_project["config"])]) == 0
    prepared = _only_run(_only_run(runs, "train_"), "training_")
    assert main(["train", "--config", str(synthetic_project["config"]),
                 "--prepared-run", str(prepared)]) == 0
    assert len(sorted(p for p in runs.iterdir() if p.is_dir())) == 1, "no second run folder"
    assert (prepared.parent / "summary.json").is_file()
    # training again into the same folder is refused instead of overwriting results
    out = main(["train", "--config", str(synthetic_project["config"]),
                "--prepared-run", str(prepared)])
    assert out == 2
