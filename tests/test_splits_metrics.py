from __future__ import annotations

import numpy as np
import pytest

from trapsig.metrics import EXCLUDED_METRICS, classification_metrics
from trapsig.splits import (
    FoldPlan,
    assert_plan_valid,
    feasibility_problems,
    grouped_stratified_folds,
    train_test_indices,
)


def _toy(n_groups_per_class: int = 6, rows_per_group: int = 4):
    groups, labels = [], []
    for cls in ("A", "B"):
        for g in range(n_groups_per_class):
            for _ in range(rows_per_group):
                groups.append(f"{cls}-rep{g}")
                labels.append(cls)
    return np.array(groups), np.array(labels)


def test_no_group_is_split_across_folds():
    groups, labels = _toy()
    plan = grouped_stratified_folds(groups, labels, n_splits=3, n_repeats=4, seed=0)
    assert_plan_valid(plan, groups, labels)  # raises on leakage


def test_every_row_is_test_exactly_once_per_repeat():
    groups, labels = _toy()
    plan = grouped_stratified_folds(groups, labels, n_splits=3, n_repeats=2, seed=0)
    for r in range(plan.n_repeats):
        seen = np.zeros(groups.size, dtype=int)
        for k in range(plan.n_splits):
            _, test = train_test_indices(plan, r, k)
            seen[test] += 1
        assert np.all(seen == 1)


def test_train_and_test_are_disjoint_by_group():
    groups, labels = _toy()
    plan = grouped_stratified_folds(groups, labels, n_splits=3, n_repeats=1, seed=5)
    for k in range(3):
        train, test = train_test_indices(plan, 0, k)
        assert set(groups[train]).isdisjoint(set(groups[test]))


def test_every_class_appears_in_every_test_fold_when_feasible():
    groups, labels = _toy(n_groups_per_class=6)
    plan = grouped_stratified_folds(groups, labels, n_splits=3, n_repeats=1, seed=1)
    for k in range(3):
        _, test = train_test_indices(plan, 0, k)
        assert set(labels[test]) == {"A", "B"}


def test_plans_are_reproducible_from_seed():
    groups, labels = _toy()
    a = grouped_stratified_folds(groups, labels, 3, 2, seed=7).fold_ids
    b = grouped_stratified_folds(groups, labels, 3, 2, seed=7).fold_ids
    assert np.array_equal(a, b)


def test_assert_plan_valid_detects_manual_leakage():
    groups = np.array(["g1", "g1", "g2", "g2"])
    labels = np.array(["A", "A", "B", "B"])
    bad = FoldPlan(fold_ids=np.array([[0, 1, 1, 1]]), n_splits=2, n_repeats=1, seed=0)
    with pytest.raises(ValueError, match="split across folds"):
        assert_plan_valid(bad, groups, labels)


def test_infeasible_plan_is_reported_before_fitting():
    groups, labels = _toy(n_groups_per_class=2)
    problems = feasibility_problems(groups, labels, n_splits=3, min_groups_per_class=5)
    assert any("independent group" in p for p in problems)


def test_single_class_is_infeasible():
    groups = np.array(["a", "b", "c", "d"])
    labels = np.array(["A"] * 4)
    assert any("fewer than 2 classes" in p for p in feasibility_problems(groups, labels, 2, 2))


# --- metrics ---------------------------------------------------------------

def test_perfect_predictions_give_unit_scores():
    y = np.array(["A", "B", "A", "B", "C", "C"])
    m = classification_metrics(y, y, labels=["A", "B", "C"])
    assert m["accuracy"].value == pytest.approx(1.0)
    assert m["balanced_accuracy"].value == pytest.approx(1.0)
    assert m["macro_f1"].value == pytest.approx(1.0)
    assert m["mcc"].value == pytest.approx(1.0)
    assert m["micro_f1"].value == pytest.approx(m["accuracy"].value)  # identity for single-label


def test_probability_metrics_undefined_without_probabilities():
    y = np.array(["A", "B"])
    m = classification_metrics(y, y)
    assert m["log_loss"].status == "undefined"
    assert m["roc_auc_ovr_macro"].status == "undefined"


def test_roc_auc_undefined_when_a_class_is_absent_from_truth():
    y_true = np.array(["A", "A", "A"])
    y_pred = np.array(["A", "A", "B"])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
    m = classification_metrics(y_true, y_pred, labels=["A", "B"], y_proba=proba)
    assert m["roc_auc_ovr_macro"].status == "undefined"
    assert m["log_loss"].status == "computed"


def test_probabilities_must_sum_to_one():
    y = np.array(["A", "B"])
    with pytest.raises(ValueError, match="sum to 1"):
        classification_metrics(y, y, labels=["A", "B"], y_proba=np.array([[0.5, 0.6], [0.5, 0.5]]))


def test_empty_evaluation_is_undefined_not_zero():
    m = classification_metrics(np.array([], dtype=str), np.array([], dtype=str))
    assert m["accuracy"].status == "undefined"
    assert m["accuracy"].value is None


def test_inapplicable_metrics_are_documented_as_excluded():
    assert "hamming_loss" in EXCLUDED_METRICS
    assert "jaccard" in EXCLUDED_METRICS
    assert any("MSE" in k for k in EXCLUDED_METRICS)


def test_balanced_accuracy_matches_sklearn_when_all_classes_have_support():
    y_true = np.array(["A", "A", "B", "B", "B"])
    y_pred = np.array(["A", "B", "B", "B", "A"])
    m = classification_metrics(y_true, y_pred, labels=["A", "B"])
    expected = (0.5 + 2 / 3) / 2
    assert m["balanced_accuracy"].status == "computed"
    assert abs(m["balanced_accuracy"].value - expected) < 1e-12


def test_balanced_accuracy_names_predicted_class_without_truth_support():
    # Class C is predicted but never true: excluded from the mean, reported, and no sklearn warning.
    import warnings

    y_true = np.array(["A", "A", "B", "B"])
    y_pred = np.array(["A", "C", "B", "B"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = classification_metrics(y_true, y_pred, labels=["A", "B", "C"])
    bal = m["balanced_accuracy"]
    assert bal.status == "computed"
    assert abs(bal.value - (0.5 + 1.0) / 2) < 1e-12
    assert "C" in bal.reason and "no true support" in bal.reason


def test_binary_roc_auc_uses_positive_class_column():
    # Two classes: AUC must be computed from the second label's probability column.
    y = np.array(["A", "A", "B", "B"])
    pred = np.array(["A", "A", "B", "B"])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
    m = classification_metrics(y, pred, labels=["A", "B"], y_proba=proba)
    assert m["roc_auc_ovr_macro"].status == "computed"
    assert m["roc_auc_ovr_macro"].value == 1.0
