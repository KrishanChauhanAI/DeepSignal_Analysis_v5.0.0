"""Leakage-safe, stratified, group-aware, repeated cross-validation folds.

Every row of the dataset carries a ``group`` (replicate, session, or file).
All rows with the same group are assigned to the same fold, so no replicate or
session can appear on both sides of a split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class FoldPlan:
    """``fold_ids[r][i]`` is the test-fold index of row ``i`` in repeat ``r``."""

    fold_ids: np.ndarray  # shape (n_repeats, n_rows), dtype int
    n_splits: int
    n_repeats: int
    seed: int


def group_class_counts(groups: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    """Number of distinct groups per class."""
    counts: dict[str, int] = {}
    for c in np.unique(labels):
        counts[str(c)] = int(np.unique(groups[labels == c]).size)
    return counts


def feasibility_problems(
    groups: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    min_groups_per_class: int,
) -> list[str]:
    """Human-readable reasons the requested split cannot be built. Empty means feasible."""
    problems: list[str] = []
    if groups.size == 0:
        return ["no rows to split"]
    if np.unique(labels).size < 2:
        problems.append("fewer than 2 classes present")
    for cls, n_groups in group_class_counts(groups, labels).items():
        if n_groups < min_groups_per_class:
            problems.append(
                f"class '{cls}' has {n_groups} independent group(s); "
                f"min_groups_per_class={min_groups_per_class}"
            )
        if n_groups < n_splits:
            problems.append(
                f"class '{cls}' has {n_groups} group(s) < n_splits={n_splits}; "
                "some test folds would lack this class"
            )
    n_total_groups = np.unique(groups).size
    if n_total_groups < n_splits:
        problems.append(f"only {n_total_groups} groups for {n_splits} folds")
    return problems


def grouped_stratified_folds(
    groups: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> FoldPlan:
    """Build ``n_repeats`` stratified group K-fold assignments.

    Each repeat uses ``random_state = seed + repeat`` so repeats differ in group
    allocation. Raises ``ValueError`` if the plan is infeasible or leaks groups.
    """
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    if groups.shape != labels.shape:
        raise ValueError("groups and labels must have the same shape")
    fold_ids = np.full((n_repeats, groups.size), -1, dtype=int)
    X = np.zeros((groups.size, 1))
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (_, test_idx) in enumerate(cv.split(X, labels, groups)):
            fold_ids[r, test_idx] = k
    plan = FoldPlan(fold_ids=fold_ids, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    assert_plan_valid(plan, groups, labels)
    return plan


def assert_plan_valid(plan: FoldPlan, groups: np.ndarray, labels: np.ndarray) -> None:
    """Check: every row assigned once per repeat, no group split across folds."""
    groups = np.asarray(groups)
    for r in range(plan.n_repeats):
        ids = plan.fold_ids[r]
        if (ids < 0).any():
            raise ValueError(f"repeat {r}: {int((ids < 0).sum())} rows unassigned")
        for g in np.unique(groups):
            if np.unique(ids[groups == g]).size != 1:
                raise ValueError(f"repeat {r}: group {g!r} is split across folds (leakage)")


def train_test_indices(plan: FoldPlan, repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays ``(train, test)`` for one repeat and fold."""
    ids = plan.fold_ids[repeat]
    test = np.flatnonzero(ids == fold)
    train = np.flatnonzero(ids != fold)
    return train, test
