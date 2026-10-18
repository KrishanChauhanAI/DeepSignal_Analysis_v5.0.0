"""Classification metric engine with explicit applicability.

Every metric returns a :class:`MetricResult` with a status:

* ``computed``        a value was calculated
* ``undefined``       mathematically undefined for this data (e.g. ROC-AUC with a missing class)
* ``not_applicable``  the metric does not apply to single-label multiclass classification

Metrics that are excluded on purpose are listed in :data:`EXCLUDED_METRICS`
with reasons, so a report can show *why* they are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn import metrics as skm

Status = Literal["computed", "undefined", "not_applicable"]


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float | None
    status: Status
    reason: str = ""


EXCLUDED_METRICS: dict[str, str] = {
    "hamming_loss": "single-label multiclass: equals 1 - accuracy, adds no information",
    "jaccard": "multi-label set metric; not used for single-label classification",
    "gini": "not a classification-performance metric for this task",
    "mAP / MRR / NDCG": "ranking metrics; only valid for embedding retrieval evaluation",
    "MSE / RMSE / MAE / R2 / MAPE / EVS / Spearman": "regression metrics; reported only in a named regression sub-analysis",
}


def _computed(name: str, value: float) -> MetricResult:
    return MetricResult(name, float(value), "computed")


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> MetricResult:
    """Mean per-class recall over classes that have true support.

    A class that is predicted but absent from ``y_true`` has no defined recall. It is excluded
    from the mean and named in the result, instead of relying on sklearn's silent drop.
    """
    truth = np.asarray([str(v) for v in y_true])
    pred = np.asarray([str(v) for v in y_pred])
    present = [c for c in labels if np.any(truth == c)]
    if not present:
        return MetricResult("balanced_accuracy", None, "undefined", "no true labels")
    recalls = [float(np.mean(pred[truth == c] == c)) for c in present]
    unsupported = sorted(set(labels) - set(present))
    extra = sorted(set(np.unique(pred).tolist()) - set(labels) - set(present))
    notes = []
    if unsupported:
        notes.append(f"classes with no true support excluded: {unsupported}")
    if extra:
        notes.append(f"predicted labels outside the label set: {extra}")
    return MetricResult("balanced_accuracy", float(np.mean(recalls)), "computed", "; ".join(notes))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[str] | None = None,
    y_proba: np.ndarray | None = None,
) -> dict[str, MetricResult]:
    """Compute applicable metrics for single-label multiclass predictions.

    ``y_proba`` columns must follow ``labels``. Probability-based metrics are
    marked ``undefined`` when it is absent.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, MetricResult] = {}
    if y_true.size == 0:
        for name in ("accuracy", "balanced_accuracy", "macro_f1"):
            out[name] = MetricResult(name, None, "undefined", "empty evaluation set")
        return out
    if labels is None:
        labels = [str(c) for c in np.unique(np.concatenate([y_true, y_pred]))]

    out["accuracy"] = _computed("accuracy", skm.accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = _balanced_accuracy(y_true, y_pred, labels)
    out["macro_f1"] = _computed(
        "macro_f1", skm.f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    # Single-label multiclass: micro-F1 is identical to accuracy. Kept as an identity check.
    out["micro_f1"] = _computed(
        "micro_f1", skm.f1_score(y_true, y_pred, average="micro", zero_division=0)
    )
    out["mcc"] = _computed("mcc", skm.matthews_corrcoef(y_true, y_pred))
    out["cohen_kappa"] = _computed("cohen_kappa", skm.cohen_kappa_score(y_true, y_pred))

    if y_proba is None:
        out["log_loss"] = MetricResult("log_loss", None, "undefined", "no probabilities supplied")
        out["roc_auc_ovr_macro"] = MetricResult(
            "roc_auc_ovr_macro", None, "undefined", "no probabilities supplied"
        )
        return out

    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.shape != (y_true.size, len(labels)):
        raise ValueError(f"y_proba shape {y_proba.shape} != ({y_true.size}, {len(labels)})")
    row_sums = y_proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("y_proba rows must sum to 1")
    # Renormalise exactly. Probabilities that came out of a float32 network or an average over
    # repeats can sit within 1e-6 of 1 but outside the 1e-8 tolerance sklearn's roc_auc_score
    # applies, which would emit a spurious "y_pred values do not sum to one" warning.
    y_proba = y_proba / row_sums[:, None]

    out["log_loss"] = _computed(
        "log_loss", skm.log_loss(y_true, y_proba, labels=labels)
    )
    present = np.unique(y_true)
    if len(labels) < 2:
        out["roc_auc_ovr_macro"] = MetricResult(
            "roc_auc_ovr_macro", None, "undefined", "fewer than 2 classes"
        )
    elif set(map(str, present)) != set(labels):
        missing = sorted(set(labels) - set(map(str, present)))
        out["roc_auc_ovr_macro"] = MetricResult(
            "roc_auc_ovr_macro", None, "undefined", f"classes absent from y_true: {missing}"
        )
    else:
        if len(labels) == 2:
            # Binary: sklearn needs a 1-D score for the positive class (labels[1]) and a 0/1 truth.
            pos = (np.asarray(y_true).astype(str) == str(labels[1])).astype(int)
            auc = skm.roc_auc_score(pos, y_proba[:, 1])
        else:
            auc = skm.roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
        out["roc_auc_ovr_macro"] = _computed("roc_auc_ovr_macro", auc)
    return out
