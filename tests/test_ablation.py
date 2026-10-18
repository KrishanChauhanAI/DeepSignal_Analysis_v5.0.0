"""Feature ablation and redundancy: the evidence behind "which features earn their place".

The ablation refits the same model on the same folds with a different feature subset, so the
numbers are paired comparisons, not independent tests. These tests check the plumbing and the
direction of the effect on data built to have one: two columns carry the class signal, one does
not, and one is an exact duplicate of another.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from trapsig.ablation import (
    ABLATION_MODES,
    feature_group_block,
    redundancy_report,
    run_ablation,
    write_ablation_csv,
    write_redundancy_csv,
)
from trapsig.config import config_from_dict, validate_training
from trapsig.features import FEATURE_GROUPS, group_of_features
from trapsig.runs import OverwriteError


def _cfg(**training_over: object):
    return config_from_dict({
        "data": {"source_dir": "/tmp", "label_from": "folder", "label_level": "file"},
        "splitting": {"n_splits": 3, "n_repeats": 2, "min_groups_per_class": 3},
        "training": {"analysis_rate_hz": 10_000.0, "models": ["logreg"], "lr_c_grid": [1.0],
                     "inner_splits": 2, "n_permutations": 0, "bootstrap_resamples": 0,
                     **training_over},
    })


COLS = ["noise_log10_sigma", "spec_log10_total", "spec_spectral_centroid_hz",
        "spec_log10_band_1_10Hz"]


def _dataset(n_per_class: int = 9, seed: int = 0):
    """Two classes separated on the 'spec' columns; 'noise' is unrelated; column 3 duplicates 1."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    y = np.array(["A"] * n_per_class + ["B"] * n_per_class, dtype=object)
    shift = np.where(y == "B", 1.0, 0.0)
    x = np.zeros((n, len(COLS)))
    x[:, 0] = rng.normal(0.0, 1.0, n)                     # no class information
    x[:, 1] = rng.normal(0.0, 0.5, n) + 4.0 * shift       # informative
    x[:, 2] = rng.normal(0.0, 0.5, n) + 3.0 * shift       # informative
    x[:, 3] = x[:, 1]                                     # exact duplicate of column 1
    groups = np.array([f"g{i}" for i in range(n)], dtype=object)
    fold_ids = np.array([[i % 3 for i in range(n)], [(i + 1) % 3 for i in range(n)]])
    return x, y, groups, fold_ids, ["A", "B"]


def _rows(block: dict) -> dict[str, dict]:
    return {r["variant"]: r for r in block["rows"]}


def test_duplicate_column_names_are_ablated_as_two_columns():
    # A name collision must not make a group lose a column it owns.
    x, y, groups, fold_ids, classes = _dataset()
    cols = ["noise_log10_sigma", "spec_log10_total", "spec_log10_total"]
    xx = x[:, :3]
    block = run_ablation(_cfg(ablation="both"), xx, y, groups, fold_ids, classes, cols, None)
    rows = _rows(block)
    assert rows["all_groups"]["n_features"] == 3
    assert rows["only:spec"]["n_features"] == 2
    assert rows["without:spec"]["n_features"] == 1
    assert rows["without:spec"]["balanced_accuracy_mean"] < 0.8


def test_run_ablation_scores_every_variant_on_the_same_folds():
    x, y, groups, fold_ids, classes = _dataset()
    cfg = _cfg(ablation="both", ablation_model="logreg")
    block = run_ablation(cfg, x, y, groups, fold_ids, classes, COLS, None)
    assert block["status"] == "computed"
    present = set(group_of_features(COLS))                # noise, spec
    rows = _rows(block)
    assert set(rows) == {"all_groups"} | {f"only:{g}" for g in present} \
        | {f"without:{g}" for g in present}
    assert rows["all_groups"]["kind"] == "baseline"
    assert rows["all_groups"]["delta_vs_all"] is None
    assert rows["all_groups"]["n_features"] == len(COLS)
    assert rows["only:spec"]["n_features"] == 3 and rows["only:noise"]["n_features"] == 1
    # The direction the data was built to have: the informative group alone does the job, and
    # removing it costs accuracy.
    assert rows["only:spec"]["balanced_accuracy_mean"] > 0.9
    assert rows["only:noise"]["balanced_accuracy_mean"] < rows["only:spec"]["balanced_accuracy_mean"]
    assert rows["without:spec"]["balanced_accuracy_mean"] < rows["all_groups"]["balanced_accuracy_mean"]
    assert rows["without:spec"]["delta_vs_all"] < 0
    assert rows["without:noise"]["delta_vs_all"] == pytest.approx(0.0, abs=0.05)
    for r in block["rows"]:
        assert r["status"] in ("computed", "skipped")
        assert set(r) == {"variant", "kind", "groups_used", "n_features", "balanced_accuracy_mean",
                          "balanced_accuracy_sd", "delta_vs_all", "status", "reason"}


def test_ablation_group_only_mode_lists_one_variant_per_group():
    x, y, groups, fold_ids, classes = _dataset()
    block = run_ablation(_cfg(ablation="groups"), x, y, groups, fold_ids, classes, COLS, None)
    kinds = {r["kind"] for r in block["rows"]}
    assert kinds == {"baseline", "group_only"}
    loo = run_ablation(_cfg(ablation="leave_one_out"), x, y, groups, fold_ids, classes, COLS, None)
    assert {r["kind"] for r in loo["rows"]} == {"baseline", "leave_one_out"}
    assert set(ABLATION_MODES) == {"none", "groups", "leave_one_out", "both"}


def test_ablation_is_skipped_with_a_reason_instead_of_running_half_configured():
    x, y, groups, fold_ids, classes = _dataset()
    off = run_ablation(_cfg(ablation="none"), x, y, groups, fold_ids, classes, COLS, None)
    assert off["status"] == "skipped" and off["rows"] == []
    assert "training.ablation is 'none'" in off["reason"]
    bad = run_ablation(_cfg(ablation="groups", ablation_model="rf"), x, y, groups, fold_ids,
                       classes, COLS, None)
    assert bad["status"] == "skipped" and "not in" in bad["reason"]


def test_ablation_respects_the_evaluation_mask():
    x, y, groups, fold_ids, classes = _dataset()
    mask = np.ones(len(y), dtype=bool)
    mask[-3:] = False                                     # three files never evaluated
    cfg = _cfg(ablation="groups")
    full = run_ablation(cfg, x, y, groups, fold_ids, classes, COLS, None)
    part = run_ablation(cfg, x, y, groups, fold_ids, classes, COLS, None, mask=mask)
    assert _rows(full)["all_groups"]["n_features"] == _rows(part)["all_groups"]["n_features"]
    assert math.isfinite(float(_rows(part)["all_groups"]["balanced_accuracy_mean"]))


def test_redundancy_flags_a_duplicate_and_leaves_independent_columns_alone():
    x, _, _, _, _ = _dataset()
    pairs = redundancy_report(x, COLS, threshold=0.98)
    assert {(p["feature_a"], p["feature_b"]) for p in pairs} == {
        ("spec_log10_total", "spec_log10_band_1_10Hz")}
    for p in pairs:
        assert abs(p["spearman"]) >= 0.98
        assert p["verdict"] == "redundant at this threshold"
        assert p["n_files_used"] == x.shape[0]
    # A looser threshold also catches the monotone relationship between the two informative
    # columns, which is exactly what the report is for.
    loose = redundancy_report(x, COLS, threshold=0.5)
    assert len(loose) > len(pairs)
    assert loose == sorted(loose, key=lambda r: -abs(r["spearman"]))


def test_redundancy_needs_enough_files_with_both_values():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(18, 3))
    x[4:, 1] = np.nan                                     # only 4 files have both columns
    cols = ["noise_a", "noise_b", "spec_c"]
    assert redundancy_report(x, cols, threshold=0.5, min_files=5) == []
    x2 = np.column_stack([x[:, 0], x[:, 0] * 2 + 1, x[:, 2]])
    assert redundancy_report(x2[:4], cols, threshold=0.5, min_files=5) == []
    flat = np.column_stack([np.ones(10), rng.normal(size=10), rng.normal(size=10)])
    assert redundancy_report(flat, cols, threshold=0.5, min_files=5) == []   # zero variance


def test_feature_group_block_reports_what_the_run_computed():
    block = feature_group_block(COLS, _cfg(feature_groups=["noise", "spec"]))
    assert block["requested"] == ["noise", "spec"]
    assert block["computed"] == {"noise": 1, "spec": 3}
    assert block["n_features"] == len(COLS)
    assert set(block["descriptions"]) == {"noise", "spec"}
    assert all(block["descriptions"].values())


def test_ablation_and_redundancy_csvs_are_written_once_and_are_readable():
    x, y, groups, fold_ids, classes = _dataset()
    block = run_ablation(_cfg(ablation="both"), x, y, groups, fold_ids, classes, COLS, None)
    pairs = redundancy_report(x, COLS, threshold=0.9)
    path = write_ablation_csv(block, _tmp_csv("ablation.csv"))
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(block["rows"])
    assert rows[0]["variant"] == "all_groups"
    rpath = write_redundancy_csv(pairs, _tmp_csv("feature_redundancy.csv"), 0.9)
    with rpath.open(encoding="utf-8", newline="") as fh:
        rrows = list(csv.DictReader(fh))
    assert len(rrows) == len(pairs)
    assert all(r["threshold"] == "0.9" for r in rrows)
    with pytest.raises(OverwriteError):
        write_ablation_csv(block, path)


def _tmp_csv(name: str):
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp()) / name


def test_configuration_rejects_unknown_groups_and_ablation_models():
    assert any("feature_groups" in e for e in validate_training(_cfg(feature_groups=["nope"])))
    assert any("ablation" in e for e in validate_training(_cfg(ablation="everything")))
    assert any("ablation_model" in e for e in validate_training(_cfg(ablation_model="rf")))
    assert any("redundancy_threshold" in e
               for e in validate_training(_cfg(redundancy_threshold=0.0)))
    assert any("redundancy_threshold" in e
               for e in validate_training(_cfg(redundancy_threshold=1.5)))
    assert validate_training(_cfg(ablation="both", feature_groups=list(FEATURE_GROUPS))) == []
