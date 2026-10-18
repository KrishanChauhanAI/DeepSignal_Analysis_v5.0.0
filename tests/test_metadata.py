"""Per-run metadata and the CSV twins of the PDF report.

The split is re-randomised on every run, so the metadata record has to be rewritten into each
run directory and has to describe the folds of THAT run. These tests pin the column layout, the
fold/role semantics (including the (n_repeats, n_pool) orientation of ``fold_ids``), the
per-file run facts, and the section CSVs that carry the same content as ``report.pdf``.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trapsig.config import TrapsigConfig
from trapsig.manifest import MANIFEST_FIELDS, write_manifest_csv
from trapsig.metadata import (
    BASE_COLUMNS,
    PER_REPEAT_COLUMNS,
    build_metadata_rows,
    read_metadata_csv,
    run_metadata_block,
    summary_line,
    write_run_metadata,
)
from trapsig.train import write_train_metadata

N_REPEATS = 2
N_SPLITS = 3


def _cfg(n_repeats: int = N_REPEATS, n_splits: int = N_SPLITS) -> TrapsigConfig:
    return dataclasses.replace(
        TrapsigConfig(),
        splitting=dataclasses.replace(TrapsigConfig().splitting,
                                      n_repeats=n_repeats, n_splits=n_splits),
    )


def _row(sid: str, cls: str = "A", group: str | None = None, **over: Any) -> dict[str, Any]:
    """A manifest-shaped row. Only the fields the metadata record reads need real values."""
    base: dict[str, Any] = {f: "" for f in MANIFEST_FIELDS}
    base.update({
        "sample_id": sid, "filename": f"{sid}.dat", "original_path": f"/data/{cls}/{sid}.dat",
        "source_subdir": cls, "class": cls, "label_source": "folder",
        "group_id": group if group is not None else sid, "group_level": "file",
        "file_sha256": "0" * 64, "size_bytes": 80000, "n_samples": 20000,
        "duration_s": 20.0, "sampling_rate_hz": 1000.0, "split": "cv",
    })
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# build_metadata_rows
# --------------------------------------------------------------------------- #

def test_every_base_column_is_present_and_per_repeat_columns_are_generated():
    rows = [_row("S1"), _row("S2")]
    fold_ids = np.array([[0, 1], [2, 0]])
    out = build_metadata_rows(rows, [0, 1], fold_ids, N_REPEATS, None, N_SPLITS)
    fields = list(out[0])
    for col in BASE_COLUMNS:
        assert col in fields, col
    for r in range(N_REPEATS):
        for pattern in PER_REPEAT_COLUMNS:
            assert pattern.replace("{N}", str(r)) in fields


def test_fold_ids_are_read_as_repeats_by_pool_not_pool_by_repeats():
    """Regression: a (n_pool, n_repeats) array silently swaps repeats and files."""
    rows = [_row("S1"), _row("S2"), _row("S3")]
    # Repeat 0 puts S1/S2/S3 in folds 0/1/2; repeat 1 shifts everyone by one.
    fold_ids = np.array([[0, 1, 2], [1, 2, 0]])
    out = build_metadata_rows(rows, [0, 1, 2], fold_ids, N_REPEATS, None, N_SPLITS)
    assert [r["fold_r0"] for r in out] == [0, 1, 2]
    assert [r["fold_r1"] for r in out] == [1, 2, 0]


def test_role_and_training_folds_follow_the_fold_assignment():
    rows = [_row("S1"), _row("S2")]
    fold_ids = np.array([[1, 2], [0, 1]])
    out = build_metadata_rows(rows, [0, 1], fold_ids, N_REPEATS, None, N_SPLITS)
    # S1 belongs to fold 1 in repeat 0: it is the test file there and trains in folds 0 and 2.
    assert (out[0]["role_r0"], out[0]["fold_r0"]) == ("test", 1)
    assert out[0]["trains_in_folds_r0"] == "0|2"
    assert out[1]["trains_in_folds_r0"] == "0|1"
    assert out[0]["trains_in_folds_r1"] == "1|2"


def test_files_outside_the_pool_are_recorded_but_never_assigned_a_fold():
    rows = [_row("S1"), _row("S2", cls="NA", group="NA", quality_flags="unreadable")]
    fold_ids = np.array([[0], [1]])
    out = build_metadata_rows(rows, [0], fold_ids, N_REPEATS, None, N_SPLITS)
    assert out[0]["in_cv_pool"] is True
    assert out[1]["in_cv_pool"] is False
    for r in range(N_REPEATS):
        assert out[1][f"fold_r{r}"] == ""
        assert out[1][f"role_r{r}"] == "not_in_pool"
        assert out[1][f"trains_in_folds_r{r}"] == ""
        assert out[1][f"cnn_inner_val_r{r}"] == ""


def test_no_fold_plan_leaves_the_columns_empty_instead_of_crashing():
    out = build_metadata_rows([_row("S1")], [0], None, N_REPEATS, None, N_SPLITS)
    assert out[0]["fold_r0"] == "" and out[0]["role_r0"] == "not_in_pool"


def test_extras_carry_the_per_file_outcome_of_this_run():
    rows = [_row("S1"), _row("S2")]
    extras = {
        "S1": {"status": "ok", "reason": "", "n_events": 4, "n_windows": 3,
               "model_b_scored": True, "cnn_inner_val_folds": ["r0f1", "r1f2"]},
        "S2": {"status": "skipped", "reason": "no event passed the detection threshold",
               "n_events": 0, "n_windows": None, "model_b_scored": False,
               "cnn_inner_val_folds": []},
    }
    out = build_metadata_rows(rows, [0, 1], np.array([[0, 1], [1, 0]]), N_REPEATS, extras,
                              N_SPLITS)
    assert out[0]["status"] == "ok" and out[0]["n_events"] == 4 and out[0]["n_windows"] == 3
    assert out[0]["model_b_scored"] is True
    assert out[0]["cnn_inner_val_r0"] == "1" and out[0]["cnn_inner_val_r1"] == "2"
    assert out[1]["status"] == "skipped"
    assert out[1]["reason"] == "no event passed the detection threshold"
    assert out[1]["n_windows"] == "" and out[1]["model_b_scored"] is False
    assert out[1]["cnn_inner_val_r0"] == ""


def test_malformed_inner_validation_tags_are_ignored_not_fatal():
    out = build_metadata_rows([_row("S1")], [0], np.array([[0], [1]]), N_REPEATS,
                              {"S1": {"cnn_inner_val_folds": ["r0fx", "junk", "r0f2"]}}, N_SPLITS)
    assert out[0]["cnn_inner_val_r0"] == "2"
    assert out[0]["cnn_inner_val_r1"] == ""


def test_list_valued_manifest_fields_are_joined_not_repr_ed():
    out = build_metadata_rows([_row("S1", quality_flags=["short", "clipped"])], [0],
                              np.array([[0], [0]]), N_REPEATS, None, N_SPLITS)
    assert out[0]["quality_flags"] == "short|clipped"


# --------------------------------------------------------------------------- #
# write_run_metadata / read_metadata_csv / summary_line
# --------------------------------------------------------------------------- #

def test_write_run_metadata_creates_both_files_in_the_run_dir(tmp_path: Path):
    run_dir = tmp_path / "runs" / "training_x"
    run_dir.mkdir(parents=True)
    rows = [_row("S1"), _row("S2", cls="B"), _row("S3", cls="NA", group="NA")]
    csv_path, json_path = write_run_metadata(run_dir, _cfg(), rows, [0, 1],
                                             np.array([[0, 1], [1, 0]]), 7, "prepare")
    assert csv_path == run_dir / "metadata.csv" and csv_path.is_file()
    assert json_path == run_dir / "metadata.json" and json_path.is_file()
    got = read_metadata_csv(csv_path)
    assert len(got) == 3, "every source file gets a row, including files outside the pool"
    assert {g["sample_id"] for g in got} == {"S1", "S2", "S3"}
    block = json.loads(json_path.read_text(encoding="utf-8"))
    assert block["stage"] == "prepare"
    assert block["n_source_files"] == 3 and block["n_cv_pool"] == 2
    assert block["n_ignored_by_glob"] == 7
    assert block["class_counts"] == {"A": 1, "B": 1}
    assert block["n_repeats"] == N_REPEATS and block["n_splits"] == N_SPLITS
    assert block["seed"] == _cfg().run.seed
    assert len(block["config_fingerprint"]) == 64
    assert "re-randomised" in block["split_randomisation"]
    assert "test" in block["role_convention"]


def test_metadata_is_never_silently_overwritten(tmp_path: Path):
    from trapsig.runs import OverwriteError

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_run_metadata(run_dir, _cfg(), [_row("S1")], [0], np.array([[0], [0]]), 0, "dry-run")
    with pytest.raises(OverwriteError):
        write_run_metadata(run_dir, _cfg(), [_row("S1")], [0], np.array([[0], [0]]), 0, "dry-run")


def test_run_metadata_block_records_the_grouping_policy(tmp_path: Path):
    cfg = dataclasses.replace(_cfg(), data=dataclasses.replace(_cfg().data,
                                                               allow_file_level_fallback=True))
    block = run_metadata_block(cfg, tmp_path, 5, 4, 1, {"A": 4}, "train")
    assert block["grouping_level"] == "file"
    assert "optimistic" in block["grouping_note"]
    assert block["holdout"].startswith("none")
    assert block["written_at_utc"].endswith("+00:00")


def test_summary_line_reads_the_written_record(tmp_path: Path):
    run_dir = tmp_path / "train_2026-01-01_00-00-00"
    run_dir.mkdir()
    write_run_metadata(run_dir, _cfg(), [_row("S1"), _row("S2", cls="B")], [0, 1],
                       np.array([[0, 1], [1, 0]]), 3, "train")
    line = summary_line(run_dir)
    assert "2 of 2 files in the CV pool" in line
    assert f"{N_REPEATS} repeats x {N_SPLITS} folds" in line
    assert "metadata.csv" in line and run_dir.name in line


def test_summary_line_says_so_when_no_record_exists(tmp_path: Path):
    assert "no record written" in summary_line(tmp_path)


# --------------------------------------------------------------------------- #
# write_train_metadata: reading the prepared run back
# --------------------------------------------------------------------------- #

def _prepared_dir(tmp_path: Path, rows: list[dict[str, Any]], pool_sids: list[str],
                  folds: list[list[int]]) -> Path:
    """A prepared run directory with the manifests the training stage reads back."""
    prepared = tmp_path / "prepared"
    mdir = prepared / "data" / "manifests"
    mdir.mkdir(parents=True)
    write_manifest_csv(rows, mdir / "master_manifest.csv")
    by_sid = {r["sample_id"]: r for r in rows}
    header = "sample_id,class,group_id," + ",".join(f"fold_r{r}" for r in range(N_REPEATS))
    lines = [header]
    for sid, fold_row in zip(pool_sids, folds, strict=True):
        lines.append(f"{sid},{by_sid[sid]['class']},{by_sid[sid]['group_id']},"
                     + ",".join(str(f) for f in fold_row))
    (mdir / "fold_assignments.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (mdir / "ignored_files.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["relative_path", "extension", "size_bytes", "reason"])
        w.writerow(["a/A1.png", ".png", "10", "not matched by data.file_glob"])
        w.writerow(["a/noext", "", "10", "not matched by data.file_glob"])
    return prepared


def test_train_metadata_reproduces_the_prepared_split_and_adds_run_facts(tmp_path: Path):
    rows = [_row("S1"), _row("S2", cls="B"), _row("S3", cls="B")]
    # fold_assignments.csv is file-major: one row per pool file, one column per repeat
    prepared = _prepared_dir(tmp_path, rows, ["S1", "S2", "S3"], [[0, 1], [1, 2], [2, 0]])
    run_dir = tmp_path / "train_run"
    run_dir.mkdir()
    table = [
        {"sample_id": "S1", "status": "ok", "reason": "", "event_n": 5.0},
        {"sample_id": "S2", "status": "ok", "reason": "", "event_n": 2.0},
        {"sample_id": "S3", "status": "skipped", "reason": "no event", "event_n": 0.0},
    ]
    branch = SimpleNamespace(counts=np.array([4, 2, 0]),
                             has_windows=np.array([True, True, False]),
                             inner_val={"r0f1": ["S2"], "r1f0": ["S1", "S2"]})
    msgs: list[str] = []
    write_train_metadata(_cfg(), run_dir, prepared, table, ["S1", "S2", "S3"], branch,
                         msgs.append)

    got = {r["sample_id"]: r for r in read_metadata_csv(run_dir / "metadata.csv")}
    assert [r["fold_r0"] for r in got.values()] == ["0", "1", "2"]
    assert [r["fold_r1"] for r in got.values()] == ["1", "2", "0"]
    assert got["S3"]["status"] == "skipped" and got["S3"]["reason"] == "no event"
    assert got["S1"]["n_events"] == "5" and got["S1"]["n_windows"] == "4"
    assert got["S3"]["n_windows"] == "0" and got["S3"]["model_b_scored"] == "False"
    assert got["S1"]["cnn_inner_val_r0"] == "" and got["S1"]["cnn_inner_val_r1"] == "0"
    assert got["S2"]["cnn_inner_val_r0"] == "1" and got["S2"]["cnn_inner_val_r1"] == "0"
    block = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert block["stage"] == "train"
    assert block["prepared_run"] == str(prepared)
    assert block["n_ignored_by_glob"] == 2
    assert block["class_counts"] == {"A": 1, "B": 2}
    assert any(m.startswith("metadata:") for m in msgs)


def test_train_metadata_tolerates_a_missing_manifest(tmp_path: Path):
    run_dir = tmp_path / "train_run"
    run_dir.mkdir()
    msgs: list[str] = []
    write_train_metadata(_cfg(), run_dir, tmp_path / "nothing", [], [], None, msgs.append)
    assert not (run_dir / "metadata.csv").exists()
    assert any("not written" in m for m in msgs)


def test_train_metadata_without_a_deep_branch_leaves_model_b_columns_empty(tmp_path: Path):
    rows = [_row("S1"), _row("S2", cls="B")]
    prepared = _prepared_dir(tmp_path, rows, ["S1", "S2"], [[0, 1], [1, 0]])  # file-major
    run_dir = tmp_path / "train_run"
    run_dir.mkdir()
    write_train_metadata(_cfg(), run_dir, prepared,
                         [{"sample_id": "S1", "status": "ok", "event_n": 1.0},
                          {"sample_id": "S2", "status": "ok", "event_n": 1.0}],
                         ["S1", "S2"], None, None)
    got = read_metadata_csv(run_dir / "metadata.csv")
    assert all(r["n_windows"] == "" and r["model_b_scored"] == "" for r in got)
    assert all(r["cnn_inner_val_r0"] == "" for r in got)


# --------------------------------------------------------------------------- #
# CSV twins of the PDF report
# --------------------------------------------------------------------------- #

def _summary() -> dict[str, Any]:
    return {
        "n_pool_files": 12, "n_files_used": 11, "n_files_skipped_or_failed": 1,
        "class_counts_used": {"A": 6, "B": 5}, "imbalance_ratio": 1.2, "class_weight": None,
        "analysis_rate_hz": 10000.0, "filter_mode": "none",
        "skipped_or_failed": [{"sample_id": "S9", "status": "skipped", "reason": "no event"}],
        "design_notes": {"grouping": "file", "holdout": "none"},
        "models": {
            "logreg": {"kind": "feature", "n_files_evaluated": 11, "evaluated_on": "all usable files",
                       "per_repeat": {"balanced_accuracy": {"mean": 0.9, "sd": 0.05},
                                      "macro_f1": {"mean": 0.89}},
                       "aggregated": {"balanced_accuracy": {"value": 0.91},
                                      "balanced_accuracy_bootstrap_ci95": {"lo": 0.7, "hi": 1.0}},
                       "permutation": {"p_value": 0.04, "status": "computed"}},
            "cnn1d": {"kind": "deep", "n_files_evaluated": 10,
                      "evaluated_on": "files with at least one usable event window",
                      "per_repeat": {"balanced_accuracy": {"mean": 0.8, "sd": 0.1}},
                      "aggregated": {}, "permutation": {}},
        },
        "deep": {"n_windows": 16, "window_ms": 200.0, "architecture": "cnn1d"},
        "features": {"requested": ["all"], "n_features": 5,
                     "computed": {"noise": 2, "spec": 2, "ctx": 1},
                     "descriptions": {"noise": "residual noise level",
                                      "spec": "spectral content",
                                      "ctx": "event context and timing"}},
        "features_dropped_all_nan": ["tf_stft_entropy"],
        "ablation": {"mode": "both", "model": "logreg", "status": "computed",
                     "protocol": "same folds, same files",
                     "interpretation": "paired comparisons, not independent tests",
                     "rows": [
                         {"variant": "all_groups", "kind": "baseline", "groups_used": "ctx|noise|spec",
                          "n_features": 5, "balanced_accuracy_mean": 0.9, "balanced_accuracy_sd": 0.05,
                          "delta_vs_all": None, "status": "computed", "reason": ""},
                         {"variant": "only:spec", "kind": "group_only", "groups_used": "spec",
                          "n_features": 2, "balanced_accuracy_mean": 0.85, "balanced_accuracy_sd": 0.1,
                          "delta_vs_all": -0.05, "status": "computed", "reason": ""},
                         {"variant": "without:tf", "kind": "leave_one_out", "groups_used": "ctx|noise|spec",
                          "n_features": 5, "balanced_accuracy_mean": None, "balanced_accuracy_sd": None,
                          "delta_vs_all": None, "status": "skipped",
                          "reason": "this variant has no feature columns left"}]},
        "feature_redundancy": {"threshold": 0.98, "n_pairs_flagged": 1,
                               "policy": "reported, never dropped",
                               "pairs": [{"feature_a": "noise_log10_sigma", "feature_b": "ctx_noise_over_i0",
                                          "spearman": 0.991, "n_files_used": 11,
                                          "verdict": "redundant at this threshold"}]},
        "comparison": {"matched_subset": {"gate_G7": {
            "verdict": "indistinguishable at this sample size",
            "reason": "bootstrap intervals overlap",
            "n_files_matched_subset": 10, "ci_overlap": True,
            "model_a": {"name": "logreg", "value": 0.91, "ci95": [0.7, 1.0]},
            "model_b": {"name": "cnn1d", "value": 0.8, "ci95": [0.5, 1.0]},
        }}},
        "config_fingerprint": "abc123", "seed": 7, "model_set": "hybrid",
        "n_repeats": 2, "n_splits": 5, "trapsig_version": "0.0.0-test",
    }


def _explain() -> dict[str, Any]:
    return {
        "architecture_condition": "Grad-CAM is valid: the explanation model is the evaluated CNN",
        "methods": {
            "temporal": {"n_windows_explained": 4, "computed": True,
                         "window_length_samples": 2000, "explanation_model": "cnn1d"},
            "baseline_only_probe": {"status": "computed", "chance": 0.5,
                                    "balanced_accuracy_mean_over_repeats": 0.62,
                                    "interpretation": "timing matters"},
            "time_shift_null": {"shift_samples": 500, "moves_with_event_fraction": 0.9,
                                "interpretation": "attributions follow the event"},
            "features": {"models": {"logreg": {
                "status": "computed",
                "top_features_by_class": {"A": [["rms", 0.4], ["area", 0.2]]}}}},
        },
    }


def _advanced() -> dict[str, Any]:
    return {
        "few_shot_results": {"setting": "k=1 episodes=40",
                             "results": {"k1": {"status": "computed", "accuracy_episode_mean": 0.67,
                                                "accuracy_episode_sd": 0.1, "n_episodes": 40}}},
        "novelty_results": {"status": "not evaluable", "reason": "no out-of-class set exists",
                            "leave_one_class_out_diagnostic": {"A": {"auroc_distance": 0.5,
                                                                     "held_out_flagged_fraction": 0.4}}},
        "semi_supervised_results": {"summary": {"supervised_only_mean": 0.8,
                                                "with_pseudo_labels_mean": 0.83, "delta_mean": 0.03,
                                                "n_accepted_pseudo_labels": 12,
                                                "n_unlabelled_pool_windows": 40}},
        "clustering_results": {"results": {"k2": {"status": "computed", "ari_train": -0.1,
                                                  "nmi_train": 0.02,
                                                  "bootstrap_jaccard_mean": 0.5,
                                                  "bootstrap_jaccard_sd": 0.1,
                                                  "note": "unstable"}}},
    }


def test_section_csvs_cover_every_pdf_section(tmp_path: Path):
    from trapsig.report import write_section_csvs

    written = write_section_csvs(_summary(), tmp_path, _explain(), _advanced(),
                                 {"model_comparison": "figures/model_comparison.png"})
    names = sorted(Path(w).name for w in written)
    assert names == [
        "report_ablation.csv", "report_advanced.csv", "report_data_landscape.csv",
        "report_explainability.csv", "report_extended_metrics.csv", "report_figures.csv",
        "report_information_gaps.csv", "report_limitations.csv", "report_methods.csv",
        "report_quad.csv", "report_reproducibility.csv", "report_results.csv",
        "report_sampling.csv", "report_viz_registry.csv",
    ]


def test_section_csv_content_matches_the_pdf_sections(tmp_path: Path):
    from trapsig.report import information_gaps, limitations, write_csv_rollups

    summary = _summary()
    write_csv_rollups(summary, tmp_path, _explain(), _advanced(),
                      {"model_comparison": "figures/model_comparison.png"})

    with (tmp_path / "report_results.csv").open(encoding="utf-8", newline="") as fh:
        results = list(csv.DictReader(fh))
    models = [r for r in results if r["row_type"] == "model"]
    assert [r["model"] for r in models] == ["logreg", "cnn1d"]
    assert models[0]["balanced_accuracy_mean"] == "0.9"
    assert models[0]["ci95_lo"] == "0.7"
    assert models[0]["permutation_p_value"] == "0.04"
    assert any(r["row_type"] == "gate_G7" for r in results)

    lim = list(csv.DictReader((tmp_path / "report_limitations.csv").open(encoding="utf-8",
                                                                        newline="")))
    assert len(lim) == len(limitations(summary))
    assert [r["limitation"] for r in lim] == [t for t, _, _ in limitations(summary)]
    assert all({r["consequence"], r["safeguard"]} - {""} for r in lim)

    gaps = list(csv.DictReader((tmp_path / "report_information_gaps.csv").open(encoding="utf-8",
                                                                              newline="")))
    assert len(gaps) == len(information_gaps(summary))
    assert [r["gap"] for r in gaps] == [g for g, _, _ in information_gaps(summary)]

    landscape = dict(csv.reader((tmp_path / "report_data_landscape.csv").open(encoding="utf-8",
                                                                              newline="")))
    assert landscape["files_in_pool"] == "12"
    assert "skipped[0]" in landscape

    explain_csv = dict(csv.reader((tmp_path / "report_explainability.csv").open(encoding="utf-8",
                                                                               newline="")))
    assert explain_csv["architecture_condition"].startswith("Grad-CAM is valid")
    assert "features.logreg.class_A" in explain_csv

    advanced_csv = dict(csv.reader((tmp_path / "report_advanced.csv").open(encoding="utf-8",
                                                                          newline="")))
    assert advanced_csv["few_shot.k1"].startswith("accuracy 0.670")
    assert advanced_csv["novelty.status"].startswith("not evaluable")
    assert "semi_supervised" in advanced_csv and "clustering.k2" in advanced_csv

    figures = list(csv.DictReader((tmp_path / "report_figures.csv").open(encoding="utf-8",
                                                                        newline="")))
    assert figures == [{"figure": "model_comparison", "file": "model_comparison.png"}]

    repro = dict(csv.reader((tmp_path / "report_reproducibility.csv").open(encoding="utf-8",
                                                                          newline="")))
    assert repro["config_fingerprint"] == "abc123"
    assert repro["seed"] == "7" and repro["model_set"] == "hybrid"
    assert repro["trapsig_version"] == "0.0.0-test"
    env = json.loads(repro["environment"])
    assert env["python"].startswith("3") and "numpy" in env["packages"]

    models_csv = list(csv.DictReader((tmp_path / "report_models.csv").open(encoding="utf-8",
                                                                          newline="")))
    assert [r["model"] for r in models_csv] == ["logreg", "cnn1d"]
    assert models_csv[0]["permutation_status"] == "computed"


def test_csv_cells_keep_the_full_unwrapped_text(tmp_path: Path):
    """What wraps across lines in the PDF must not be truncated in the CSV twin."""
    from trapsig.report import write_section_csvs

    write_section_csvs(_summary(), tmp_path, _explain(), _advanced(), None)
    text = (tmp_path / "report_limitations.csv").read_text(encoding="utf-8")
    long_cell = max((len(line) for line in text.splitlines()), default=0)
    assert long_cell > 120, "the limitation text is written out in full, not clipped"


def test_sections_seven_and_eight_use_fixed_width_wrapping_cells():
    from reportlab.platypus import Paragraph, Table

    from trapsig import report

    style = report.ParagraphStyle("celltest", fontName="Helvetica", fontSize=7, leading=9)
    saved = report._CELL
    report._CELL = style
    try:
        table = report._three_col([("a" * 400, "b" * 400, "c" * 400)],
                                  ["Limitation", "Consequence", "Safeguard applied"])
    finally:
        report._CELL = saved
    assert isinstance(table, Table)
    widths = table._argW  # noqa: SLF001 - the fixed boundaries are the point of the test
    assert len(widths) == 3 and all(w is not None and w > 0 for w in widths)
    assert sum(widths) <= 172 * report.mm, "the three columns fit inside the text frame"
    assert table.splitByRow == 1, "a row that does not fit continues on the next page"
    for cell in table._cellvalues[1]:  # noqa: SLF001
        assert isinstance(cell, Paragraph)
        assert cell.style is style
        assert cell.style.splitLongWords, "unbreakable tokens must stay inside the boundary"


def test_comparison_table_cells_are_paragraphs_too():
    from reportlab.platypus import Paragraph

    from trapsig import report

    style = report.ParagraphStyle("celltest2", fontName="Helvetica", fontSize=7, leading=9)
    saved = report._CELL
    report._CELL = style
    try:
        table = report._comparison_table(_summary(), style)
    finally:
        report._CELL = saved
    body = table._cellvalues[1]  # noqa: SLF001
    assert all(isinstance(c, Paragraph) for c in body)
    assert body[0].text == "logreg"


def test_section_10_csv_carries_groups_ablation_and_redundancy(tmp_path: Path):
    """The CSV twin of section 10 holds every row the PDF shows, in the same order."""
    from trapsig.report import ABLATION_FIELDS, _ablation_section, ablation_rows, write_section_csvs

    summary = _summary()
    write_section_csvs(summary, tmp_path, _explain(), _advanced(), None)
    with (tmp_path / "report_ablation.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    expected = [{k: ("" if r.get(k) is None else str(r.get(k))) for k in ABLATION_FIELDS}
                for r in ablation_rows(summary)]
    assert rows == expected                                   # same rows, same order, same content
    kinds = [r["row_type"] for r in rows]
    assert kinds[:3] == ["feature_group"] * 3
    assert "ablation" in kinds and "redundant_pair" in kinds and "note" in kinds
    by_name = {r["name"]: r for r in rows if r["row_type"] == "ablation"}
    assert by_name["all_groups"]["balanced_accuracy_mean"] == "0.9"
    assert by_name["all_groups"]["delta_vs_all"] == ""          # the baseline has no delta
    assert by_name["only:spec"]["delta_vs_all"] == "-0.05"
    assert by_name["without:tf"]["status"] == "skipped"
    assert "no feature columns left" in by_name["without:tf"]["reason"]
    pair = [r for r in rows if r["row_type"] == "redundant_pair"][0]
    assert pair["name"] == "noise_log10_sigma <-> ctx_noise_over_i0"
    assert pair["spearman"] == "0.991" and pair["n_files_used"] == "11"
    # Section 2 also records which features were used and how they were chosen.
    methods = dict(csv.reader((tmp_path / "report_methods.csv").open(encoding="utf-8", newline="")))
    assert methods["features.n_features_used"] == "5"
    assert "noise" in methods["features.per_group"]
    assert "section 10" in methods["features.selection_rule"]
    assert methods["ablation.mode"] == "both"

    # The PDF section renders the same rows, including a variant that was not computed.
    from trapsig import report

    style = report.ParagraphStyle("celltest", fontName="Helvetica", fontSize=7, leading=9)
    saved = report._CELL
    report._CELL = style
    try:
        flow = _ablation_section(summary, style, style, style)
    finally:
        report._CELL = saved
    tables = [f for f in flow if type(f).__name__ == "Table"]
    assert len(tables) == 3                                    # groups, ablation, redundancy
    ablation_table = tables[1]
    assert ablation_table._cellvalues[0][0].text == "Feature subset"  # noqa: SLF001
    variants = [row[0].text for row in ablation_table._cellvalues[1:]]  # noqa: SLF001
    assert variants[0].startswith("all_groups (5 features)")
    assert any("not computed" in row[1].text for row in ablation_table._cellvalues[1:])


def test_pdf_sections_announce_their_csv_twin(tmp_path: Path):
    """The reader of the PDF is told which CSV carries the same content."""
    source = Path(report_module_path()).read_text(encoding="utf-8")
    for name in ("report_data_landscape.csv", "report_methods.csv", "report_results.csv",
                 "report_figures.csv", "report_explainability.csv", "report_advanced.csv",
                 "report_limitations.csv", "report_information_gaps.csv",
                 "report_reproducibility.csv", "report_ablation.csv"):
        assert f"CSV twin: <font face='Courier'>{name}</font>" in source, name


def report_module_path() -> str:
    import trapsig.report as mod

    return mod.__file__ or ""
