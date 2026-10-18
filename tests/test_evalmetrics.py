"""Tests for the extended evaluation surface (design section 9)."""

from __future__ import annotations

import numpy as np

from trapsig.evalmetrics import DEFINITIONS, extended_metrics, write_extended_csv


def _rows_dict(rows, metric, cls=""):
    return {r["class"]: r["value"] for r in rows if r["metric"] == metric and
            (cls == "" or r["class"] == cls)}


def test_perfect_predictions() -> None:
    classes = ["A", "B"]
    y = np.array(["A", "B", "A", "B"], dtype=object)
    pred = y.copy()
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    rows, summary = extended_metrics("m", y, pred, proba, classes, top_k=2, bins=5)
    d = {r["metric"]: r["value"] for r in rows if r["class"] == ""}
    assert d["mcc_multiclass"] == 1.0
    assert abs(d["brier_multiclass"] - 0.05) < 1e-6      # mean of 0.02, 0.08, 0.08, 0.02
    assert d["ece"] <= 0.2 and d["mce"] <= 0.2
    assert summary["status"] == "computed" and summary["n_files"] == 4
    assert d["top2_accuracy"] == 1.0
    assert all(v is not None for v in DEFINITIONS.values())


def test_per_class_specificity_and_support() -> None:
    classes = ["A", "B"]
    y = np.array(["A", "A", "B", "B"], dtype=object)
    pred = np.array(["A", "B", "B", "B"], dtype=object)
    proba = np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    rows, _ = extended_metrics("m", y, pred, proba, classes)
    sup = _rows_dict(rows, "support")
    assert sup == {"A": 2, "B": 2}
    err = _rows_dict(rows, "errors")
    assert err["A"] == 1 and err["B"] == 0
    spec = _rows_dict(rows, "specificity")
    assert spec["A"] == 1.0 and spec["B"] == 0.5
    pr = _rows_dict(rows, "pr_auc_ovr")
    assert all(v is not None for v in pr.values())


def test_absent_class_reports_undefined_not_fabricated() -> None:
    classes = ["A", "B", "C"]
    y = np.array(["A", "B", "A", "B"], dtype=object)
    pred = y.copy()
    proba = np.array([[0.6, 0.4, 0.0], [0.3, 0.7, 0.0], [0.8, 0.2, 0.0], [0.2, 0.8, 0.0]])
    rows, _ = extended_metrics("m", y, pred, proba, classes, pr_curves=True)
    pr = [r for r in rows if r["metric"] == "pr_auc_ovr" and r["class"] == "C"]
    assert pr and pr[0]["value"] is None and "undefined" in pr[0]["note"]
    spec = [r for r in rows if r["metric"] == "specificity" and r["class"] == "C"]
    assert spec and spec[0]["value"] == 1.0      # every file is a true negative for C


def test_degenerate_input_is_not_applicable() -> None:
    rows, summary = extended_metrics("m", np.array(["A"], dtype=object),
                                     np.array(["A"], dtype=object),
                                     np.array([[np.nan, np.nan]]), ["A", "B"])
    assert rows == []
    assert summary["status"] == "not_applicable"


def test_write_extended_csv(tmp_path):
    classes = ["A", "B"]
    y = np.array(["A", "B", "A", "B"], dtype=object)
    rows, _ = extended_metrics("m", y, y.copy(), np.full((4, 2), 0.5), classes, top_k=2)
    path = write_extended_csv(rows, tmp_path / "metrics_extended.csv")
    text = (tmp_path / "metrics_extended.csv").read_text()
    assert text.splitlines()[0] == "model,metric,class,value,n,note"
    assert "specificity" in text and "brier_multiclass" in text
    assert str(path).endswith("metrics_extended.csv")
