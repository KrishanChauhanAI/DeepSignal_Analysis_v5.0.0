"""The extended evaluation surface (design section 9), computed from real predictions only.

Everything here consumes the same out-of-fold label/probability arrays the headline metrics use;
nothing is recomputed from a different partition. Definitions are stated where they matter:

* **specificity** (per class): TN / (TN + FP) in the one-vs-rest reading; undefined when no file
  lacks the class, and reported as ``None`` rather than 1.0.
* **MCC**: the multiclass Matthews correlation from the confusion matrix (the standard
  generalisation, ``sklearn.matthews_corrcoef``).
* **Brier** (multiclass): mean over files of the squared distance between the probability vector
  and the one-hot label, i.e. ``mean(sum_c (p_c - 1[y=c])^2)``; 0 is perfect, 2 is worst.
* **ECE** (expected calibration error): ``sum_b (n_b / n) * |acc_b - conf_b|`` over ``bins``
  equal-width bins of the maximum probability; **MCE** is the maximum of the same gap. Both are
  documented in the CSV header row through ``summary["evaluation"]["definitions"]``.
* **PR-AUC**: one-vs-rest average precision; undefined for a class absent from, or universal in,
  the evaluated subset, and recorded as ``None`` with the reason.
* **top-k accuracy**: fraction of files whose true label lies in the ``k`` highest probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

EXT_FIELDS = ["model", "metric", "class", "value", "n", "note"]


def _onehot(y: np.ndarray, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    out = np.zeros((y.size, len(classes)))
    for i, v in enumerate(y.tolist()):
        j = idx.get(str(v))
        if j is not None:
            out[i, j] = 1.0
    return out


def extended_metrics(name: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray,
                     classes: list[str], top_k: int = 0, bins: int = 10,
                     pr_curves: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows for ``metrics_extended.csv`` plus a summary block for ``summary.json``."""
    rows: list[dict[str, Any]] = []
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=object)
    pred = np.asarray(pred, dtype=object)
    fin = np.all(np.isfinite(proba), axis=1) if proba.ndim == 2 else np.zeros(y.size, bool)
    n = int(fin.sum())
    if n < 2:
        return rows, {"status": "not_applicable",
                      "reason": "fewer than two files with finite probabilities"}
    yy, pp, pr = y[fin], pred[fin], proba[fin]
    summary: dict[str, Any] = {"n_files": n, "status": "computed"}

    cm_classes = classes
    for c in cm_classes:
        truth = np.array([str(v) == c for v in yy.tolist()])
        predd = np.array([str(v) == c for v in pp.tolist()])
        tn = int((~truth & ~predd).sum())
        fp = int((~truth & predd).sum())
        fn = int((truth & ~predd).sum())
        rows.append({"model": name, "metric": "support", "class": c, "value": int(truth.sum()),
                     "n": n, "note": ""})
        rows.append({"model": name, "metric": "errors", "class": c, "value": int(fn),
                     "n": n, "note": "false negatives in the one-vs-rest reading"})
        spec = tn / (tn + fp) if (tn + fp) > 0 else None
        rows.append({"model": name, "metric": "specificity", "class": c,
                     "value": None if spec is None else round(spec, 6), "n": n,
                     "note": "" if spec is not None else "no negative file: undefined"})
        if pr_curves and 0 < int(truth.sum()) < n and proba.shape[1] == len(classes):
            j = classes.index(c)
            ap = float(average_precision_score(truth.astype(int), pr[:, j]))
            prec, rec, _ = precision_recall_curve(truth.astype(int), pr[:, j])
            rows.append({"model": name, "metric": "pr_auc_ovr", "class": c,
                         "value": round(ap, 6), "n": n,
                         "note": f"average precision; curve area by trapezoid over {prec.size} "
                                 "operating points"})
        else:
            rows.append({"model": name, "metric": "pr_auc_ovr", "class": c, "value": None,
                         "n": n, "note": "class absent or universal in this subset: undefined"})

    try:
        mcc = float(matthews_corrcoef(yy, pp))
    except ValueError:
        mcc = None
    rows.append({"model": name, "metric": "mcc_multiclass", "class": "",
                 "value": None if mcc is None else round(mcc, 6), "n": n, "note": ""})
    oh = _onehot(yy, classes)
    brier = float(np.mean(np.sum((pr - oh) ** 2, axis=1)))
    rows.append({"model": name, "metric": "brier_multiclass", "class": "",
                 "value": round(brier, 6), "n": n,
                 "note": "mean squared distance to the one-hot label (0 perfect, 2 worst)"})
    conf = np.max(pr, axis=1)
    acc = np.array([str(p) == str(t) for p, t in zip(pp.tolist(), yy.tolist(), strict=True)],
                   dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    for b in range(bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1]) if b else (conf >= 0) & (conf <= edges[1])
        if m.sum() == 0:
            continue
        gap = abs(float(acc[m].mean()) - float(conf[m].mean()))
        ece += gap * m.sum() / n
        mce = max(mce, gap)
    rows.append({"model": name, "metric": "ece", "class": "", "value": round(ece, 6), "n": n,
                 "note": f"expected calibration error, {bins} equal-width max-probability bins, "
                         "weighted by bin count"})
    rows.append({"model": name, "metric": "mce", "class": "", "value": round(mce, 6), "n": n,
                 "note": "maximum calibration error over the same bins"})
    if top_k and top_k > 1 and proba.shape[1] >= top_k:
        topk = np.argsort(-pr, axis=1)[:, :top_k]
        truth_idx = np.array([classes.index(str(v)) for v in yy.tolist()])
        hit = float(np.mean([t in row for row, t in zip(topk, truth_idx, strict=True)]))
        rows.append({"model": name, "metric": f"top{top_k}_accuracy", "class": "",
                     "value": round(hit, 6), "n": n, "note": ""})
    if proba.shape[1] >= 2 and len({str(v) for v in yy.tolist()}) >= 2:
        try:
            auc = float(roc_auc_score(yy, pr, multi_class="ovr", labels=classes))
            rows.append({"model": name, "metric": "roc_auc_ovr", "class": "",
                         "value": round(auc, 6), "n": n, "note": ""})
        except ValueError as exc:
            rows.append({"model": name, "metric": "roc_auc_ovr", "class": "", "value": None,
                         "n": n, "note": f"undefined here: {exc}"})
    rep = classification_report(yy, pp, labels=classes, output_dict=True, zero_division=0)
    for c in classes:
        for met in ("precision", "recall", "f1-score"):
            rows.append({"model": name, "metric": met, "class": c,
                         "value": round(float(rep[c][met]), 6), "n": n, "note": ""})
    try:
        kappa = float(cohen_kappa_score(yy, pp))
        rows.append({"model": name, "metric": "cohens_kappa", "class": "",
                     "value": round(kappa, 6), "n": n, "note": ""})
        summary["cohens_kappa"] = round(kappa, 6)
    except ValueError:
        rows.append({"model": name, "metric": "cohens_kappa", "class": "", "value": None,
                     "n": n, "note": "undefined: degenerate prediction set"})
    for avg in ("macro avg", "weighted avg"):
        tag = avg.split()[0]
        for met in ("precision", "recall", "f1-score"):
            rows.append({"model": name, "metric": f"{met}_{tag}", "class": "",
                         "value": round(float(rep[avg][met]), 6), "n": n, "note": ""})
    micro = precision_recall_fscore_support(yy, pp, average="micro", zero_division=0)[:3]
    for met, val in zip(("precision", "recall", "f1-score"), micro, strict=True):
        rows.append({"model": name, "metric": f"{met}_micro", "class": "",
                     "value": round(float(val), 6), "n": n,
                     "note": "multiclass micro averages equal accuracy"})
    summary["brier"] = round(brier, 6)
    summary["ece"] = round(ece, 6)
    summary["mce"] = round(mce, 6)
    summary["mcc"] = None if mcc is None else round(mcc, 6)
    return rows, summary


def write_extended_csv(rows: list[dict[str, Any]], path: Any) -> Any:
    import csv
    import io

    from .runs import write_text_exclusive

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=EXT_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in EXT_FIELDS})
    return write_text_exclusive(path, buf.getvalue())


DEFINITIONS = {
    "cohens_kappa": "chance-corrected agreement (sklearn cohen_kappa_score); empty when the "
                    "prediction set is degenerate",
    "precision_micro/recall_micro/f1_micro": "multiclass micro averages; identical to accuracy",
    "precision_macro/recall_macro/f1_macro": "unweighted mean of the per-class scores",
    "precision_weighted/recall_weighted/f1_weighted": "per-class scores weighted by support",
    "specificity": "TN/(TN+FP) one-vs-rest; None when the subset holds no negative file",
    "mcc_multiclass": "Matthews correlation from the multiclass confusion matrix",
    "brier_multiclass": "mean over files of sum_c (p_c - 1[y=c])^2",
    "ece": "sum_b (n_b/n)|acc_b - conf_b| over equal-width bins of the max probability",
    "mce": "max_b |acc_b - conf_b| over the same bins",
    "pr_auc_ovr": "one-vs-rest average precision; None when the class is absent or universal",
    "topk_accuracy": "fraction of files whose true label is among the k largest probabilities",
}
