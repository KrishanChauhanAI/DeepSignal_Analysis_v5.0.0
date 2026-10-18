"""The hybrid branch: Model B (1D CNN) plus the pre-declared combination with Model A.

Design section 11.1 fixes how window-level predictions become a file-level decision:
the aggregation rule is declared in config before evaluation (``deep.aggregation``) and is
never chosen by looking at test performance.

Fairness rule for the comparison
--------------------------------
Model B can only score files that produced at least one detected event. Model A scores every
readable file. Comparing Model A on all files with Model B on a subset is not like for like, so
this module also scores Model A **restricted to the Model B subset** and reports that alongside.
The gate in design section 15 (G7: Model B beats or matches Model A) is evaluated on the matched
subset, and the subset size is reported with it.
"""

from __future__ import annotations

import csv
import io
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrapsigConfig
from .runs import write_text_exclusive
from .windows import WindowSet, aggregate_to_file

Say = Any  # callable(str) -> None


class HybridError(RuntimeError):
    """A condition that stops the hybrid branch."""


@dataclass
class DeepBranch:
    """File-level results of Model B, aligned with the Model A file order."""
    proba: np.ndarray                 # (n_repeats, n_files, k); NaN rows for files without windows
    pred: np.ndarray                  # (n_repeats, n_files) object; "" where not applicable
    counts: np.ndarray                # (n_files,) windows per file, from the last repeat
    has_windows: np.ndarray           # (n_files,) bool
    cv: Any = None
    notes: list[str] = field(default_factory=list)
    inner_val: dict[str, list[str]] = field(default_factory=dict)
    # Nearest-prototype classifier on the same frozen embedding (metric learning at inference).
    proba_proto: np.ndarray | None = None
    pred_proto: np.ndarray | None = None
    has_proto: np.ndarray | None = None
    val_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    val_files: dict[str, list[str]] = field(default_factory=dict)
    # Model B's own hyperparameter search, when deep.hpo_trials > 0: one row per trial plus one
    # summary per searched fold, so the run reports the search instead of only its winner.
    hpo_rows: list[dict[str, Any]] = field(default_factory=list)
    hpo_summaries: list[dict[str, Any]] = field(default_factory=list)


def run_deep_branch(
    ws: WindowSet, fold_ids: np.ndarray, file_order: list[str], classes: list[str],
    cfg: TrapsigConfig, imbalance_ratio: float, run_dir: Path | None = None, say: Say = None,
) -> DeepBranch:
    """Train Model B over all repeats and aggregate window predictions to file level."""
    tell = say or (lambda m: None)
    k = len(classes)
    n_rep, n_files = fold_ids.shape
    notes: list[str] = []

    if ws.x.shape[0] == 0:
        notes.append("no event windows exist in the pool; Model B is not evaluable")
        tell("deep: no windows; skipping Model B")
        return DeepBranch(
            proba=np.full((n_rep, n_files, k), np.nan), pred=np.full((n_rep, n_files), "", dtype=object),
            counts=np.zeros(n_files, dtype=np.int64), has_windows=np.zeros(n_files, dtype=bool),
            notes=notes, has_proto=np.zeros(n_files, dtype=bool))

    from .deep import cv_windows

    cv = cv_windows(ws, fold_ids, file_order, k, classes, cfg, imbalance_ratio, run_dir, tell)

    proba = np.full((n_rep, n_files, k), np.nan)
    pred = np.full((n_rep, n_files), "", dtype=object)
    proba_proto = np.full((n_rep, n_files, k), np.nan)
    pred_proto = np.full((n_rep, n_files), "", dtype=object)
    counts = np.zeros(n_files, dtype=np.int64)
    cls_arr = np.array(classes, dtype=object)
    proto_src = getattr(cv, "proba_proto", None)
    for r in range(n_rep):
        p_file, c_file = aggregate_to_file(cv.proba[r], ws.sample_id, file_order, cfg.deep.aggregation)
        proba[r] = p_file
        counts = c_file
        ok = ~np.isnan(p_file[:, 0])
        pred[r, ok] = cls_arr[np.argmax(p_file[ok], axis=1)]
        if proto_src is not None:
            pp, _ = aggregate_to_file(proto_src[r], ws.sample_id, file_order,
                                      cfg.deep.aggregation)
            proba_proto[r] = pp
            okp = ~np.isnan(pp[:, 0])
            pred_proto[r, okp] = cls_arr[np.argmax(pp[okp], axis=1)]
    has_windows = counts > 0
    # A file counts as prototyped only where every repeat scored it: a metric computed on a
    # mask that is partial for some repeats would average over files the model never saw.
    has_proto = (~np.isnan(proba_proto[:, :, 0]).any(axis=0) if proto_src is not None
                 else np.zeros(n_files, dtype=bool))
    if proto_src is not None:
        partial = (~np.isnan(proba_proto[:, :, 0]).all(axis=0)) & ~has_proto
        if partial.any():
            notes.append(f"{int(partial.sum())} file(s) were prototyped in only some repeats and "
                         "are excluded from the prototype head's metrics")
    if not has_windows.any():
        notes.append("every file lost its windows to the edge rule; Model B is not evaluable")
    n_missing = int(np.sum(~has_windows))
    if n_missing:
        notes.append(
            f"{n_missing} of {n_files} files produced no usable event window; Model B does not score "
            "them and the hybrid falls back to Model A for those files")
        # Design 11.1: with file-level labels, report the label noise rate as the share of
        # files with zero detected events under a label that implies events.
        notes.append(f"label-noise rate (files with zero usable events): {n_missing / n_files:.3f}")
    notes.extend(cv.notes)
    return DeepBranch(proba=proba, pred=pred, counts=counts, has_windows=has_windows, cv=cv,
                      notes=notes, inner_val=dict(getattr(cv, "inner_val", {}) or {}),
                      proba_proto=proba_proto if has_proto.any() else None,
                      pred_proto=pred_proto if has_proto.any() else None,
                      has_proto=has_proto,
                      val_embeddings=dict(getattr(cv, "val_embeddings", {}) or {}),
                      val_files=dict(getattr(cv, "val_files", {}) or {}),
                      hpo_rows=list(getattr(cv, "hpo_rows", []) or []),
                      hpo_summaries=list(getattr(cv, "hpo_summaries", []) or []))


@dataclass
class FusionStack:
    """File-level probabilities of the fused model (handcrafted features + learned embedding)."""
    proba: np.ndarray                   # (n_repeats, n_files, k); NaN where a fold could not fuse
    pred: np.ndarray                    # (n_repeats, n_files) object
    ok: np.ndarray                      # (n_files,) bool: fused in at least one repeat
    n_val_files: int = 0                # files the head was fitted on, summed over folds
    notes: list[str] = field(default_factory=list)


def _robust_standardise(fit_x: np.ndarray, apply_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median/IQR standardisation fitted on the fusion head's own training rows only."""
    med = np.nanmedian(fit_x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    q1, q3 = np.nanpercentile(fit_x, 25, axis=0), np.nanpercentile(fit_x, 75, axis=0)
    iqr = np.where(np.isfinite(q3 - q1) & ((q3 - q1) > 0), q3 - q1, 1.0)
    a = (np.nan_to_num(apply_x, nan=0.0) - med) / iqr
    f = (np.nan_to_num(fit_x, nan=0.0) - med) / iqr
    return np.clip(f, -8, 8), np.clip(a, -8, 8)


def _mean_by_file(emb: np.ndarray, sids: Sequence[str], order: list[str]) -> tuple[np.ndarray, list[int]]:
    """Average window embeddings per file, returning the rows and the file positions used."""
    pos = {s: i for i, s in enumerate(order)}
    acc: dict[int, list[np.ndarray]] = {}
    for i, sid in enumerate(sids):
        j = pos.get(str(sid))
        if j is None or i >= emb.shape[0] or not np.all(np.isfinite(emb[i])):
            continue
        acc.setdefault(j, []).append(emb[i])
    idx = sorted(acc)
    out = np.stack([np.mean(acc[j], axis=0) for j in idx]) if idx else np.empty((0, emb.shape[1]
                                                                                if emb.ndim > 1 else 0))
    return out, idx


def fusion_stack(
    deep_branch: DeepBranch, ws: WindowSet, fold_ids: np.ndarray, file_order: list[str],
    x_files: np.ndarray, y_files: np.ndarray, classes: list[str], cfg: TrapsigConfig,
    say: Say = None,
) -> FusionStack:
    """Fuse the handcrafted features with the learned representation, one head per outer fold.

    Protocol (fixed before evaluation, and the reason it is leakage-safe):

    * the encoder of fold ``f`` in repeat ``r`` was trained on the outer-training files **minus**
      the inner-validation files, so the inner-validation embeddings are out-of-sample for it;
    * the fusion head is fitted on those inner-validation files only - their handcrafted features
      (label-blind, per file) concatenated with their mean embedding - and predicts the outer test
      files of the same fold;
    * no test file contributes to the standardisation, to the head, or to the prototypes.

    The cost is one logistic regression per (repeat, fold), because the embeddings already exist.
    """
    from sklearn.linear_model import LogisticRegression

    tell = say or (lambda m: None)
    k = len(classes)
    n_rep, n_files = fold_ids.shape
    proba = np.full((n_rep, n_files, k), np.nan)
    pred = np.full((n_rep, n_files), "", dtype=object)
    cls_arr = np.array(classes, dtype=object)
    notes: list[str] = []
    n_val_total = 0
    n_fitted = 0
    n_single_class = 0
    n_pairs = 0
    val_emb = deep_branch.val_embeddings
    val_sids = deep_branch.val_files
    emb_all = getattr(deep_branch.cv, "embeddings", None)
    if emb_all is None or not val_emb:
        notes.append("no fold produced validation embeddings; the fusion head cannot be fitted")
        return FusionStack(proba=proba, pred=pred, ok=np.zeros(n_files, dtype=bool), notes=notes)

    sid_pos = {s: i for i, s in enumerate(file_order)}
    w_file = np.array([sid_pos.get(str(s), -1) for s in ws.sample_id.tolist()])
    for r in range(n_rep):
        w_fold = fold_ids[r][np.clip(w_file, 0, max(n_files - 1, 0))]
        w_fold = np.where(w_file >= 0, w_fold, -1)
        for f in range(cfg.splitting.n_splits):
            tag = f"r{r}f{f}"
            n_pairs += 1
            te_w = w_fold == f
            va_emb, va_idx = _mean_by_file(val_emb.get(tag, np.empty((0, 0))),
                                           val_sids.get(tag, []), file_order)
            te_emb, te_idx = _mean_by_file(emb_all[r][te_w], ws.sample_id[te_w].tolist(), file_order)
            if va_emb.shape[0] == 0 or te_emb.shape[0] == 0:
                continue
            y_va = y_files[va_idx]
            if len(set(y_va.tolist())) < 2:
                n_single_class += 1          # one logistic regression needs at least two classes
                continue
            zf, za = _robust_standardise(x_files[va_idx], x_files[te_idx])
            ef, ea = _robust_standardise(va_emb, te_emb)
            fit_x = np.hstack([zf, ef])
            app_x = np.hstack([za, ea])
            head = LogisticRegression(C=1.0, max_iter=2000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                head.fit(fit_x, y_va)
            p = head.predict_proba(app_x)
            cols = np.zeros(k)
            full = np.zeros((app_x.shape[0], k))
            for j, c in enumerate(head.classes_):
                if str(c) in classes:
                    full[:, classes.index(str(c))] = p[:, j]
            tot = full.sum(axis=1, keepdims=True)
            full = np.where(tot > 0, full / np.maximum(tot, 1e-12), 1.0 / k)
            proba[r, te_idx] = full
            pred[r, te_idx] = cls_arr[np.argmax(full, axis=1)]
            n_val_total += len(va_idx)
            n_fitted += 1
            del cols
    # Strict mask: a file is fused only where every repeat produced a fusion head for its fold.
    if n_single_class:
        notes.append(f"{n_single_class} of {n_pairs} (repeat, fold) pairs had inner-validation "
                     "files from one class only; a logistic regression cannot be fitted there, so "
                     "no head was produced for those folds")
    ok = ~np.isnan(proba[:, :, 0]).any(axis=0)
    partial = (~np.isnan(proba[:, :, 0]).all(axis=0)) & ~ok
    if partial.any():
        notes.append(f"{int(partial.sum())} file(s) were fused in only some repeats and are "
                     "excluded from the fusion head's metrics, because a metric averaged over "
                     "different file sets per repeat would not describe one evaluation")
    if not ok.any():
        if n_fitted == 0:
            notes.append("no (repeat, fold) pair produced a fusion head; the fused model is not "
                         "evaluable on this dataset")
        else:
            notes.append(f"{n_fitted} of {n_pairs} (repeat, fold) pairs fitted a head, but no file "
                         "was fused in every repeat, so the fused model has no complete "
                         "out-of-fold prediction here and is reported as not evaluable rather than "
                         "scored on the repeats that happened to work")
    tell(f"fusion: {int(ok.sum())} of {n_files} files scored by the fusion head "
         f"({n_val_total} validation-file fits)")
    return FusionStack(proba=proba, pred=pred, ok=ok, n_val_files=n_val_total, notes=notes)


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------

def combine_probabilities(
    proba_a: np.ndarray, proba_b: np.ndarray, weight_a: float, has_windows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Blend Model A and Model B file-level probabilities with the pre-declared rule.

    ``proba_a`` is (n_repeats, n_files, k) and must be finite everywhere. ``proba_b`` has NaN
    rows for files Model B cannot score; those rows fall back to Model A alone and are flagged
    in the returned ``source`` array so the fallback is visible in the outputs rather than
    hidden inside an averaged number.
    """
    k = proba_a.shape[2]
    out = np.array(proba_a, dtype=float, copy=True)
    source = np.full(proba_a.shape[:2], "model_a_only", dtype=object)
    b_ok = ~np.isnan(proba_b[:, :, 0]) & has_windows[None, :]
    w_a = float(min(max(weight_a, 0.0), 1.0))
    out[b_ok] = w_a * proba_a[b_ok] + (1.0 - w_a) * np.nan_to_num(proba_b[b_ok], nan=0.0)
    source[b_ok] = "hybrid"
    # Renormalise: both inputs are probability vectors, so the blend already sums to 1 up to
    # floating point, but the fallback rows must stay exactly normalised.
    tot = out.sum(axis=2, keepdims=True)
    out = np.where(tot > 0, out / np.maximum(tot, 1e-12), np.full_like(out, 1.0 / k))
    return out, source


def hybrid_predictions(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    cls = np.array(classes, dtype=object)
    out: np.ndarray = cls[np.argmax(proba, axis=2)]
    return out


# ---------------------------------------------------------------------------
# Model comparison framework
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRow:
    model: str
    metric: str
    mean: float | None
    sd: float | None
    ci_lo: float | None
    ci_hi: float | None
    n_files: int
    status: str
    reason: str = ""


def _mean_sd(vals: list[float]) -> tuple[float | None, float | None]:
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if not v:
        return None, None
    if len(v) == 1:
        return float(v[0]), 0.0
    return float(np.mean(v)), float(np.std(v, ddof=1))


def build_comparison(
    per_repeat_metrics: dict[str, list[dict[str, Any]]],
    n_files_by_model: dict[str, int],
    focus: tuple[str, ...] = ("balanced_accuracy", "macro_f1", "accuracy", "log_loss"),
) -> list[ComparisonRow]:
    """One row per model and metric, with mean and SD across repeats.

    ``per_repeat_metrics[model]`` is a list of ``{metric: {value, status, reason}}`` dicts, one
    per repeat. Metrics that are not applicable stay not applicable, with the reason carried
    through, rather than being written as numbers (design 11.11).
    """
    rows: list[ComparisonRow] = []
    for model, reps in per_repeat_metrics.items():
        if not reps:
            continue
        names = list(reps[0].keys())
        ordered = [m for m in focus if m in names] + [m for m in names if m not in focus]
        for mname in ordered:
            vals = [r[mname].get("value") for r in reps if r.get(mname, {}).get("status") == "computed"]
            mean, sd = _mean_sd([float(v) for v in vals if v is not None])
            status = "computed" if mean is not None else reps[0][mname].get("status", "not_applicable")
            reason = "" if mean is not None else str(reps[0][mname].get("reason", ""))
            rows.append(ComparisonRow(model=model, metric=mname, mean=mean, sd=sd, ci_lo=None, ci_hi=None,
                                      n_files=n_files_by_model.get(model, 0), status=status, reason=reason))
    return rows


def attach_cis(rows: list[ComparisonRow], cis: dict[tuple[str, str], tuple[float, float]]) -> None:
    for r in rows:
        ci = cis.get((r.model, r.metric))
        if ci:
            r.ci_lo, r.ci_hi = ci


def gate_g7(
    best_a: float | None, best_a_name: str, a_ci: tuple[float, float] | None,
    b: float | None, b_name: str, b_ci: tuple[float, float] | None, n_subset: int,
) -> dict[str, Any]:
    """Design gate G7: does Model B beat or match Model A, with CI overlap assessed?

    The verdict is deliberately conservative. With this sample size the intervals are wide, so
    "beats" is only claimed when the intervals do not overlap at all. Otherwise the honest
    answer is that the comparison cannot separate the models, and Model A (cheaper, interpretable)
    is the one to prefer, as design 11.2 states.
    """
    out: dict[str, Any] = {
        "gate": "G7", "metric": "balanced_accuracy", "n_files_matched_subset": n_subset,
        "model_a": {"name": best_a_name, "value": best_a, "ci95": list(a_ci) if a_ci else None},
        "model_b": {"name": b_name, "value": b, "ci95": list(b_ci) if b_ci else None},
    }
    if best_a is None or b is None:
        out.update({"verdict": "not evaluable",
                    "reason": "one side has no computable balanced accuracy on the matched subset"})
        return out
    overlap = None
    if a_ci and b_ci:
        overlap = not (a_ci[1] < b_ci[0] or b_ci[1] < a_ci[0])
    out["ci_overlap"] = overlap
    if overlap is False and b > best_a:
        out.update({"verdict": "Model B beats Model A",
                    "reason": "95% intervals do not overlap and Model B is higher"})
    elif overlap is False and b < best_a:
        out.update({"verdict": "Model A beats Model B",
                    "reason": "95% intervals do not overlap and Model A is higher"})
    else:
        out.update({"verdict": "indistinguishable at this sample size",
                    "reason": "the 95% intervals overlap, so the data cannot separate the models; "
                              "design 11.2 prefers Model A when a CNN does not clearly win"})
    return out


def write_comparison_csv(rows: list[ComparisonRow], path: Path) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["model", "metric", "mean", "sd", "ci95_lo", "ci95_hi",
                                        "n_files", "status", "reason"])
    w.writeheader()
    for r in rows:
        w.writerow({"model": r.model, "metric": r.metric,
                    "mean": "" if r.mean is None else round(r.mean, 6),
                    "sd": "" if r.sd is None else round(r.sd, 6),
                    "ci95_lo": "" if r.ci_lo is None else round(r.ci_lo, 6),
                    "ci95_hi": "" if r.ci_hi is None else round(r.ci_hi, 6),
                    "n_files": r.n_files, "status": r.status, "reason": r.reason})
    write_text_exclusive(path, buf.getvalue())


def write_deep_windows_csv(ws: WindowSet, path: Path) -> None:
    """Provenance for every window: which file, which event, which onset."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["window_index", "sample_id", "class", "group_id",
                                        "event_index", "onset_s"])
    w.writeheader()
    for i in range(ws.x.shape[0]):
        w.writerow({"window_index": i, "sample_id": ws.sample_id[i], "class": ws.class_label[i],
                    "group_id": ws.group_id[i], "event_index": int(ws.event_index[i]),
                    "onset_s": round(float(ws.onset_s[i]), 6)})
    write_text_exclusive(path, buf.getvalue())
