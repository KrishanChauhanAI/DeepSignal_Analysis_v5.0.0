"""Tests for the figure subsystem and its machine-readable twins (Phase VIII).

Every category the run can request is either drawn or recorded with a reason in
``figures_index.csv``; a figure never silently disappears. These tests also pin the behaviour on
degenerate inputs (empty tables, all-missing features, CSV-round-tripped string numbers), because
a visualization that crashes on a small dataset would take the training run with it.
"""

from __future__ import annotations

import csv

import numpy as np

from trapsig import viz


def _rows(n_per_class: int = 4):
    rng = np.random.default_rng(3)
    rows = []
    for ci, c in enumerate(("A", "B")):
        for i in range(n_per_class):
            rows.append({"sample_id": f"{c}{i}", "class": c, "group_id": f"{c}{i}",
                         "filename": f"{c}{i}.dat", "status": "ok", "reason": "",
                         "duration_s": 20.0, "event_n": float(1 + (i % 3)),
                         "noise_log10_sigma": rng.normal(ci), "amp_log10_rms": rng.normal(ci),
                         "ctx_baseline_drift_rel": rng.normal(), "event_dip_mean": rng.normal(ci),
                         "spec_log10_total": np.nan if i == 0 else rng.normal(ci)})
    return rows


def _matrix(rows, names):
    return np.array([[r.get(n, np.nan) for n in names] for r in rows], dtype=float)


def test_dataset_composition_draws_and_reports(tmp_path) -> None:
    rows = _rows()
    recs = viz.dataset_composition_rows(rows, ["A", "B"])
    assert {r["class"] for r in recs} == {"A", "B"}
    assert recs[0]["n_files"] == 4 and recs[0]["events_total"] > 0
    out = viz.plot_dataset_composition(rows, ["A", "B"], 1.0, tmp_path / "c.png")
    assert out and (tmp_path / "c.png").is_file()
    assert viz.plot_dataset_composition([], ["A"], 1.0, tmp_path / "no.png") is None


def test_feature_distributions_and_missing_pattern(tmp_path) -> None:
    rows = _rows()
    names = ["noise_log10_sigma", "amp_log10_rms", "ctx_baseline_drift_rel", "event_dip_mean",
             "spec_log10_total"]
    x = _matrix(rows, names)
    y = np.array([r["class"] for r in rows], dtype=object)
    dist = viz.feature_distribution_rows(x, names, y, ["A", "B"])
    assert len(dist) == len(names) * 2
    assert any(r["n_missing"] == 1 for r in dist), "the NaN entry must be counted"
    p = viz.plot_feature_distributions(x, names, y, ["A", "B"], tmp_path / "d.png")
    assert p and (tmp_path / "d.png").is_file()

    miss = viz.missing_value_rows(x, names, [r["sample_id"] for r in rows])
    by_key = {r["key"]: r for r in miss if r["row_type"] == "feature"}
    assert by_key["spec_log10_total"]["n_missing"] == 2, "one NaN per class (i == 0)"
    p = viz.plot_missing_pattern(x, names, [r["sample_id"] for r in rows], tmp_path / "m.png")
    assert p and (tmp_path / "m.png").is_file()
    # A table with nothing missing must still produce the figure (the check was performed).
    xf = np.where(np.isfinite(x), x, 0.0)
    p = viz.plot_missing_pattern(xf, names, [r["sample_id"] for r in rows], tmp_path / "m0.png")
    assert p and (tmp_path / "m0.png").is_file()


def test_correlation_flags_redundant_pairs(tmp_path) -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=30)
    x = np.column_stack([a, a + rng.normal(scale=0.01, size=30), rng.normal(size=30)])
    names = ["a", "a_copy", "other"]
    pairs = viz.feature_correlation_rows(x, names, threshold=0.98, report_at=0.8)
    top = pairs[0]
    assert {top["feature_a"], top["feature_b"]} == {"a", "a_copy"}
    assert top["redundant_at_threshold"] is True
    p = viz.plot_feature_correlation(x, names, np.array(["A"] * 15 + ["B"] * 15, dtype=object),
                                     ["A", "B"], tmp_path / "r.png", threshold=0.98)
    assert p and (tmp_path / "r.png").is_file()


def test_ablation_figure_draws_the_baseline_line(tmp_path) -> None:
    rows = [{"variant": "all_groups", "kind": "baseline", "n_features": 83,
             "balanced_accuracy_mean": 0.9, "balanced_accuracy_sd": 0.05},
            {"variant": "only:tf", "kind": "only", "n_features": 12,
             "balanced_accuracy_mean": 0.6, "balanced_accuracy_sd": 0.1},
            {"variant": "without:tf", "kind": "without", "n_features": 71,
             "balanced_accuracy_mean": 0.88, "balanced_accuracy_sd": 0.04}]
    p = viz.plot_ablation(rows, tmp_path / "a.png")
    assert p and (tmp_path / "a.png").is_file()
    assert viz.plot_ablation([], tmp_path / "no.png") is None


def test_hpo_figure_reads_csv_strings_as_numbers(tmp_path) -> None:
    rows = [{"model": "logreg", "repeat": 0, "fold": 0, "fold_tag": "r0f0", "method": "random",
             "metric": "balanced_accuracy", "trial": str(i), "rung": "0", "n_train_files": "9",
             "score": ("0.9" if i else ""), "n_inner_splits": "3",
             "status": "computed" if i else "failed", "reason": "" if i else "blew up",
             "params": "{}"} for i in range(4)]
    p = viz.plot_hpo(rows, tmp_path / "h.png")
    assert p and (tmp_path / "h.png").is_file()
    assert viz.plot_hpo([], tmp_path / "no.png") is None


def test_error_analysis_separates_files_and_calls(tmp_path) -> None:
    y = np.array(["A", "B", "A", "B"], dtype=object)
    pred = np.array([["A", "B", "A", "A"], ["A", "B", "B", "B"]], dtype=object)
    proba = np.array([[[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4]],
                      [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]]])
    sids = ["f1", "f2", "f3", "f4"]
    rows = viz.error_analysis_rows(y, pred, proba, sids, ["A", "B"], "logreg")
    files = {r["sample_id"]: r for r in rows if r["row_type"] == "file"}
    assert files["f3"]["error_rate"] == 0.5 and files["f3"]["failure_case"] is True
    assert files["f1"]["error_rate"] == 0.0 and files["f1"]["failure_case"] is False
    assert files["f3"]["modal_wrong_class"] == "B"
    p = viz.plot_error_analysis(y, pred, proba, sids, ["A", "B"], "logreg", tmp_path / "e.png")
    assert p and (tmp_path / "e.png").is_file()


def test_figures_index_records_every_category(tmp_path) -> None:
    statuses = [{"category": c, "figure": "", "file": "", "status": "not requested",
                 "reason": "because", "csv_twin": ""} for c in viz.FIGURE_CATEGORIES]
    path = viz.write_figures_index(statuses, tmp_path / "figures_index.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert {r["category"] for r in got} == set(viz.FIGURE_CATEGORIES)
    assert all(r["description"] for r in got), "every category must say what it shows"
    assert all(r["status"] == "not requested" and r["reason"] == "because" for r in got)
