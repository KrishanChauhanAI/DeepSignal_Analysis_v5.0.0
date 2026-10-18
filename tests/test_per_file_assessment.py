"""Per-file assessment: nested class folders, raw annotations, ignored files, skips, duplicates.

The dataset layout under test mirrors the laboratory's: the top-level folders under source_dir are
the classes, a class folder may contain a nested subfolder, the folder also holds non-data files,
and file names carry free-text annotations after " - " that are kept verbatim.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from trapsig.cli import main
from trapsig.config import config_from_dict
from trapsig.manifest import (
    build_manifest,
    discover_sources,
    list_ignored_files,
    parse_name_annotation,
    parse_source_subdir,
)
from trapsig.pipeline import RecordingSkipped, process_recording

FS = 1000.0
N = 1000  # one second at 1 kHz


def _cfg_dict(source: Path, runs: Path) -> dict:
    return {
        "run": {"name": "assess", "seed": 1},
        "data": {
            "source_dir": str(source), "file_glob": "*.dat", "label_level": "file",
            "label_from": "folder", "allow_file_level_fallback": True,
            "expected_duration_s": 1.0, "duration_tolerance_s": 0.001,
            "dat": {"encoding": "binary", "dtype": "float32", "byteorder": "little",
                    "header_bytes": 0, "n_columns": 1, "column": 0, "scale": 1.0, "units": "A"},
        },
        "sampling": {"original_hz": FS, "target_rates_hz": [500.0]},
        "filters": {"mode": "none"},
        "events": {"baseline_window_s": 0.1, "pre_event_window_s": 0.02, "min_duration_s": 0.002},
        "splitting": {"n_splits": 2, "n_repeats": 1, "min_groups_per_class": 2},
        "output": {"runs_dir": str(runs)},
    }


def _write_dat(path: Path, x: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.asarray(x, dtype="<f4").tobytes())
    return path


def _noise(seed: int) -> np.ndarray:
    return 1e-9 * np.random.default_rng(seed).normal(size=N) + 2e-9


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    src = tmp_path / "dataset"
    _write_dat(src / "ClassA" / "rec_01 - trap.dat", _noise(1))
    _write_dat(src / "ClassA" / "rec_02.dat", _noise(2))
    _write_dat(src / "ClassA" / "Connected trace" / "rec_03 - reduced to 300 mA.dat", _noise(3))
    (src / "ClassA" / "plot_01 5000Hz.png").write_bytes(b"\x89PNG-not-really")
    _write_dat(src / "ClassB" / "rec_04.dat", _noise(4))
    _write_dat(src / "ClassB" / "rec_05.dat", _noise(5))
    return {"src": src, "runs": tmp_path / "runs", "cfg": tmp_path / "cfg.yaml"}


def test_path_helpers_keep_annotations_verbatim():
    assert parse_source_subdir("ClassA/Connected trace/x.dat") == "ClassA/Connected trace"
    assert parse_source_subdir("ClassB/x.dat") == "ClassB"
    assert parse_name_annotation("Ntau_XL07 @14_41_06 - reduced to 300 mA") == "reduced to 300 mA"
    assert parse_name_annotation("rec_02") == ""


def test_class_is_top_folder_and_nested_subfolder_is_kept(tree):
    cfg = config_from_dict(_cfg_dict(tree["src"], tree["runs"]))
    records = discover_sources(cfg.data, "t")
    by_name = {r.path.name: r for r in records}
    assert len(records) == 5
    assert by_name["rec_03 - reduced to 300 mA.dat"].label == "ClassA"
    rows = build_manifest(records, default_fs=FS, allow_file_fallback=True,
                          label_level="file", dat=cfg.data.dat)
    row = {r["filename"]: r for r in rows}
    assert row["rec_03 - reduced to 300 mA.dat"]["source_subdir"] == "ClassA/Connected trace"
    assert row["rec_03 - reduced to 300 mA.dat"]["name_annotation"] == "reduced to 300 mA"
    assert row["rec_02.dat"]["source_subdir"] == "ClassA"
    assert row["rec_02.dat"]["name_annotation"] == ""
    assert row["rec_04.dat"]["class"] == "ClassB"


def test_non_matching_files_are_listed_not_silently_dropped(tree):
    cfg = config_from_dict(_cfg_dict(tree["src"], tree["runs"]))
    ignored = list_ignored_files(cfg.data)
    assert [(rel, ext) for rel, ext, _ in ignored] == [("ClassA/plot_01 5000Hz.png", ".png")]


def test_identical_files_in_different_classes_share_a_group_and_are_flagged(tree):
    shared = _noise(99)
    _write_dat(tree["src"] / "ClassA" / "twin_a.dat", shared)
    _write_dat(tree["src"] / "ClassB" / "twin_b.dat", shared)
    cfg = config_from_dict(_cfg_dict(tree["src"], tree["runs"]))
    rows = build_manifest(discover_sources(cfg.data, "t"), default_fs=FS,
                          allow_file_fallback=True, label_level="file", dat=cfg.data.dat)
    twins = [r for r in rows if r["filename"].startswith("twin_")]
    assert len({r["group_id"] for r in twins}) == 1
    assert twins[0]["group_id"].startswith("dup:")
    assert all("exact_duplicate_across_classes" in r["quality_flags"] for r in twins)


def test_non_finite_recording_is_skipped_with_reason(tree):
    bad = _noise(7).copy()
    bad[10] = np.nan
    f = _write_dat(tree["src"] / "ClassB" / "bad.dat", bad)
    cfg = config_from_dict(_cfg_dict(tree["src"], tree["runs"]))
    with pytest.raises(RecordingSkipped) as info:
        process_recording(f, cfg, FS)
    assert "non-finite" in info.value.reason
    assert info.value.stats["n_nonfinite"] == 1


def test_flat_recording_is_skipped_not_analysed(tree):
    f = _write_dat(tree["src"] / "ClassB" / "flat.dat", np.full(N, 2e-9))
    cfg = config_from_dict(_cfg_dict(tree["src"], tree["runs"]))
    with pytest.raises(RecordingSkipped) as info:
        process_recording(f, cfg, FS)
    assert "flat" in info.value.reason


def test_events_writes_one_row_per_source_file_including_skipped(tree):
    _write_dat(tree["src"] / "ClassB" / "flat.dat", np.full(N, 2e-9))
    bad = _noise(8).copy()
    bad[5] = np.inf
    _write_dat(tree["src"] / "ClassA" / "bad.dat", bad)
    tree["cfg"].write_text(yaml.safe_dump(_cfg_dict(tree["src"], tree["runs"])), encoding="utf-8")

    assert main(["events", "--config", str(tree["cfg"]), "--rate-hz", str(FS)]) == 0
    run = next(p for p in tree["runs"].iterdir() if p.name.startswith("events_"))
    with (run / "metrics" / "per_file_summary.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 7  # every source .dat file, nothing dropped (the .png is not a recording)
    status = {r["filename"]: r["status"] for r in rows}
    assert status["bad.dat"] == "skipped" and status["flat.dat"] == "skipped"
    assert status["rec_01 - trap.dat"] == "ok"
    reasons = {r["filename"]: r["reason"] for r in rows}
    assert "non-finite" in reasons["bad.dat"]
    ok = next(r for r in rows if r["filename"] == "rec_02.dat")
    assert float(ok["n_samples"]) == N and float(ok["duration_s"]) == pytest.approx(1.0)
    assert ok["class"] == "ClassA" and ok["source_subdir"] == "ClassA"
    assert ok["sample_id"] and len(ok["sample_id"]) == 16

    import json
    summary = json.loads((run / "metrics" / "events_summary.json").read_text(encoding="utf-8"))
    assert summary["n_skipped"] == 2
    assert summary["grouping"]["policy"] == "file_level_fallback"
    assert "optimistic" in summary["grouping"]["note"]


def test_prepare_writes_ignored_files_and_links_every_pool_file(tree):
    tree["cfg"].write_text(yaml.safe_dump(_cfg_dict(tree["src"], tree["runs"])), encoding="utf-8")
    assert main(["prepare", "--config", str(tree["cfg"])]) == 0
    run = next(tree["runs"].glob("train_*/training_*"))
    with (run / "data" / "manifests" / "ignored_files.csv").open(encoding="utf-8") as fh:
        ignored = list(csv.DictReader(fh))
    assert [r["extension"] for r in ignored] == [".png"]
    with (run / "data" / "manifests" / "master_manifest.csv").open(encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))
    assert len(manifest) == 5
    assert {"source_subdir", "name_annotation", "group_level"} <= set(manifest[0])
    nested = next(r for r in manifest if r["filename"].startswith("rec_03"))
    assert nested["source_subdir"] == "ClassA/Connected trace"
    assert nested["name_annotation"] == "reduced to 300 mA"
    assert all(r["link_status"] in ("linked", "created", "exists") or r["symlink_path"] for r in manifest)
