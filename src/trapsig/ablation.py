"""Feature-group ablation and redundancy reporting.

Adding features is only an improvement if the run can show it, and a feature set is only honest
if near-duplicates are visible. This module answers both questions without touching the
evaluation protocol:

* **Ablation.** The declared ablation model is refitted on each feature group alone and on
  everything minus one group, using *the same folds* as the main cross-validation, so every
  variant is compared on identical splits and identical files. The difference against the
  all-groups run is reported per variant with the number of features it used. Nothing here
  selects features: the full declared set is what the main results use, and the ablation is
  evidence about which groups carry the signal.
* **Redundancy.** Pairwise |Spearman| over the pooled feature matrix flags groups of features
  that say the same thing. Flagged pairs are *reported, never dropped*: dropping columns on the
  pooled matrix would use information from every file, including test-fold files.

Both analyses are label-blind in the sense that matters here: the ablation refits inside the
existing folds, and redundancy uses the features only, never the labels.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sst

from .config import TrapsigConfig
from .features import GROUP_DESCRIPTIONS, feature_group, group_of_features
from .runs import write_text_exclusive

Say = Any  # callable(str) -> None

ABLATION_MODES = ("none", "groups", "leave_one_out", "both")


@dataclass
class AblationRow:
    variant: str                # "all_groups" | "only:<group>" | "without:<group>"
    kind: str                   # baseline | group_only | leave_one_out
    groups_used: str            # "|"-joined group names
    n_features: int
    balanced_accuracy_mean: float | None
    balanced_accuracy_sd: float | None
    delta_vs_all: float | None
    status: str                 # computed | skipped
    reason: str = ""


def _balanced_accuracies(pred: np.ndarray, y: np.ndarray, classes: list[str], mask: np.ndarray
                         ) -> tuple[float | None, float | None]:
    """Mean and SD of per-repeat balanced accuracy over the evaluated files."""
    from .train import balanced_accuracy

    vals = []
    for r in range(pred.shape[0]):
        v = balanced_accuracy(y[mask], pred[r][mask], classes)
        if v is not None and math.isfinite(v):
            vals.append(float(v))
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def run_ablation(cfg: TrapsigConfig, x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 fold_ids: np.ndarray, classes: list[str], feat_cols: list[str],
                 class_weight: str | None, mask: np.ndarray | None = None,
                 say: Say = None) -> dict[str, Any]:
    """Refit the ablation model per feature-group variant on the main CV folds.

    Returns a block for the summary with one row per variant. ``mask`` restricts evaluation to
    the files the main run evaluated, so the ablation and the headline numbers describe the same
    files.
    """
    from .train import cv_predictions

    tell = say or (lambda m: None)
    t = cfg.training
    out: dict[str, Any] = {
        "mode": t.ablation, "model": t.ablation_model,
        "feature_groups_available": sorted({feature_group(c) for c in feat_cols}),
        "n_features_total": len(feat_cols),
        "protocol": ("same folds, same files, same model settings as the main cross-validation; "
                     "only the feature columns change"),
        "tuning": ("declared hyperparameters, no nested search: a feature-subset difference must "
                   "not be confounded by tuning noise, and 21 variants x a search budget would "
                   "dominate the run"),
    }
    if t.ablation == "none":
        out.update({"status": "skipped", "reason": "training.ablation is 'none'", "rows": []})
        return out
    if t.ablation_model not in t.models:
        out.update({"status": "skipped",
                    "reason": f"training.ablation_model {t.ablation_model!r} is not in "
                              f"training.models {t.models}"})
        return out
    if mask is None:
        mask = np.ones(len(y), dtype=bool)

    # Positions, not names: two columns that happen to share a name must still be ablated as two
    # columns, and a group must lose every column it owns.
    all_idx = list(range(len(feat_cols)))
    pos_of_group: dict[str, list[int]] = {}
    for i, c in enumerate(feat_cols):
        pos_of_group.setdefault(feature_group(c), []).append(i)
    variants: list[tuple[str, str, list[int]]] = [("all_groups", "baseline", all_idx)]
    if t.ablation in ("groups", "both"):
        for g, idx in pos_of_group.items():
            variants.append((f"only:{g}", "group_only", list(idx)))
    if t.ablation in ("leave_one_out", "both"):
        for g, idx in pos_of_group.items():
            drop = set(idx)
            variants.append((f"without:{g}", "leave_one_out", [i for i in all_idx if i not in drop]))

    rows: list[AblationRow] = []
    base_mean: float | None = None
    for i, (variant, kind, idx) in enumerate(variants):
        if not idx:
            rows.append(AblationRow(variant, kind, "", 0, None, None, None, "skipped",
                                    "this variant has no feature columns left"))
            continue
        used = "|".join(sorted({feature_group(feat_cols[j]) for j in idx}))
        tell(f"ablation {i + 1}/{len(variants)}: {variant} ({len(idx)} features)")
        try:
            pred, _, _ = cv_predictions(t.ablation_model, x[:, idx], y, groups, fold_ids, classes,
                                        cfg, class_weight, tune=False)
        except Exception as exc:                       # a degenerate variant must not stop the run
            rows.append(AblationRow(variant, kind, used, len(idx), None, None, None, "skipped",
                                    f"{type(exc).__name__}: {exc}"))
            continue
        mean, sd = _balanced_accuracies(pred, y, classes, mask)
        if kind == "baseline":
            base_mean = mean
        rows.append(AblationRow(variant, kind, used, len(idx), mean, sd,
                                (mean - base_mean) if (mean is not None and base_mean is not None
                                                       and kind != "baseline") else None,
                                "computed" if mean is not None else "skipped",
                                "" if mean is not None else "no repeat produced a computable score"))
    out.update({"status": "computed", "rows": [r.__dict__ for r in rows],
                "baseline_balanced_accuracy_mean": base_mean,
                "interpretation": (
                    "a group whose 'only:' score is near the all-groups score carries most of the "
                    "signal; a group whose 'without:' score is near the all-groups score adds "
                    "little. Differences are on the same folds, so they are paired comparisons, "
                    "but they are not independent tests: no p-value is claimed here")})
    return out


def redundancy_report(x: np.ndarray, feat_cols: list[str], threshold: float,
                      min_files: int = 5) -> list[dict[str, Any]]:
    """Pairwise |Spearman| over the pooled feature matrix, flagging near-duplicates.

    A pair is scored only on the files where both features are finite, and a pair with fewer than
    ``min_files`` such files is left out of the report entirely rather than being reported as
    uncorrelated: too few files means the correlation is not estimated, not that it is zero.
    """
    n, p = x.shape
    if p < 2 or n < min_files:
        return []
    # Rank each column on its finite values only; NaN stays NaN so a pair is scored on the files
    # where both features exist.
    ranks = np.full((n, p), np.nan)
    for j in range(p):
        col = x[:, j]
        fin = np.isfinite(col)
        if np.count_nonzero(fin) >= 2:
            ranks[fin, j] = sst.rankdata(col[fin])
    out: list[dict[str, Any]] = []
    for a in range(p):
        for b in range(a + 1, p):
            both = np.isfinite(ranks[:, a]) & np.isfinite(ranks[:, b])
            k = int(np.count_nonzero(both))
            if k < min_files:
                continue
            ra, rb = ranks[both, a], ranks[both, b]
            if float(np.std(ra)) == 0 or float(np.std(rb)) == 0:
                continue
            rho = float(np.corrcoef(ra, rb)[0, 1])
            if not math.isfinite(rho):
                continue
            if abs(rho) >= threshold:
                out.append({"feature_a": feat_cols[a], "feature_b": feat_cols[b],
                            "spearman": round(rho, 6), "n_files_used": k,
                            "verdict": "redundant at this threshold"})
    out.sort(key=lambda r: -abs(r["spearman"]))
    return out


def write_ablation_csv(block: dict[str, Any], path: Path) -> Path:
    fields = ["variant", "kind", "groups_used", "n_features", "balanced_accuracy_mean",
              "balanced_accuracy_sd", "delta_vs_all", "status", "reason"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in block.get("rows") or []:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return write_text_exclusive(path, buf.getvalue())


def write_redundancy_csv(pairs: list[dict[str, Any]], path: Path, threshold: float) -> Path:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["feature_a", "feature_b", "spearman", "n_files_used",
                                        "verdict", "threshold"])
    w.writeheader()
    for r in pairs:
        w.writerow({**r, "threshold": threshold})
    return write_text_exclusive(path, buf.getvalue())


def feature_group_block(feat_cols: list[str], cfg: TrapsigConfig) -> dict[str, Any]:
    """What the run actually computed: groups, counts, and what each group means."""
    by_group = group_of_features(feat_cols)
    return {
        "requested": list(cfg.training.feature_groups),
        "computed": {g: len(cols) for g, cols in by_group.items()},
        "n_features": len(feat_cols),
        "descriptions": {g: GROUP_DESCRIPTIONS.get(g, "") for g in by_group},
    }
