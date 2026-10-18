"""Visualization (design section 12).

Two families of figure, kept separate on purpose:

* **Scientific signal figures** - the analysed trace with detected events, the class-mean event
  window, and the class-mean power spectral density. These are about the data.
* **Model figures** - learning curves, confusion matrices, ROC, calibration, the model comparison
  with bootstrap intervals, embedding scatter, and temporal attributions. These are about the fit.

Every figure is written with the Agg backend at the configured DPI and never depends on a display.
Axes are labelled with units, and no figure is generated from data the model was evaluated on
without saying so in the caption text.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

# Headless backend, set before pyplot is imported so no display is ever required.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

Say = Any

# One colour per class, cycled. Colour-blind-safe enough for two or three classes.
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _colors(classes: list[str]) -> dict[str, str]:
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(classes)}


def _save(fig: Any, path: Path, dpi: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Scientific signal figures
# ---------------------------------------------------------------------------

def plot_recording(signal: np.ndarray, baseline: np.ndarray, fs_hz: float, events: list[Any],
                   title: str, path: Path, dpi: int = 150) -> str:
    """Analysed trace with the rolling-median baseline and every detected event marked."""
    t = np.arange(signal.size) / fs_hz
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(t, signal, lw=0.6, color="#333333", label="analysed signal")
    ax.plot(t, baseline, lw=0.9, color="#D55E00", alpha=0.85, label="rolling-median baseline")
    for i, ev in enumerate(events):
        ax.axvspan(ev.onset_s, ev.offset_s, color="#0072B2", alpha=0.16,
                   label="detected event" if i == 0 else None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("current (configured units)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=3, framealpha=0.85)
    return _save(fig, path, dpi)


def plot_class_mean_windows(ws: Any, classes: list[str], path: Path, dpi: int = 150) -> str | None:
    """Mean event window per class, aggregated over files first.

    Averaging over files rather than over windows stops a file with many events from dominating the
    class mean, which would make the picture a statement about one recording.
    """
    if ws is None or ws.x.shape[0] == 0:
        return None
    fs = ws.rate_hz
    t = np.arange(ws.x.shape[1]) / fs * 1000.0
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    cols = _colors(classes)
    for c in classes:
        m = ws.class_label == c
        if not np.any(m):
            continue
        per_file = []
        for sid in set(ws.sample_id[m].tolist()):
            sel = m & (ws.sample_id == sid)
            if np.any(sel):
                per_file.append(ws.x[sel].mean(axis=0))
        arr = np.stack(per_file)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(t, mean, color=cols[c], lw=1.4, label=f"{c} (n={arr.shape[0]} files)")
        ax.fill_between(t, mean - sd, mean + sd, color=cols[c], alpha=0.18)
    ax.axvline(ws.length and float(np.mean(ws.onset_s)) * 0 + 0, color="none")
    ax.set_xlabel("time within window (ms)")
    ax.set_ylabel("baseline-subtracted current")
    ax.set_title("Class-mean event window (mean over files, SD band)", fontsize=10)
    ax.legend(fontsize=7)
    return _save(fig, path, dpi)


def plot_psd_by_class(features: list[dict[str, Any]], classes: list[str], bands: list[list[float]],
                      path: Path, dpi: int = 150) -> str | None:
    """Mean band power per class, from the already-computed per-file features.

    The column names are built with the same helper ``features.py`` uses, so this figure cannot
    silently look for a name that does not exist (it did, once, and the run reported no figure
    without saying why -- ``figures_index.csv`` now records the skip).
    """
    from .features import _band_tag

    cols = [f"spec_log10_band_{_band_tag(float(b[0]), float(b[1]))}" for b in bands]
    if not features or not bands or cols[0] not in features[0]:
        return None
    labels = [f"{b[0]:g}-{b[1]:g}" for b in bands]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.8 / max(len(classes), 1)
    cc = _colors(classes)
    for i, c in enumerate(classes):
        rows = [r for r in features if r.get("class") == c]
        vals, errs = [], []
        for col in cols:
            v = np.array([float(r[col]) for r in rows
                          if col in r and _finite(r.get(col))], dtype=float)
            vals.append(float(np.mean(v)) if v.size else 0.0)
            errs.append(float(np.std(v, ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0)
        ax.bar(np.arange(len(cols)) + i * width, vals, width=width, yerr=errs,
               color=cc[c], label=f"{c} (n={len(rows)})", capsize=2)
    ax.set_xticks(np.arange(len(cols)) + width * (len(classes) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("frequency band (Hz)")
    ax.set_ylabel("log10 band power")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_title("Mean log10 band power per class (SEM)", fontsize=10)
    ax.legend(fontsize=7)
    return _save(fig, path, dpi)


def _finite(v: Any) -> bool:
    try:
        return bool(math.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Model figures
# ---------------------------------------------------------------------------

def plot_learning_curves(logs: list[dict[str, Any]], path: Path, dpi: int = 150) -> str | None:
    """Train and validation loss, and the checkpoint-selection metric, for every fold."""
    if not logs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for lg in logs:
        h = lg.get("history") or []
        if not h:
            continue
        ep = [r["epoch"] for r in h]
        lab = f"r{lg['repeat']}f{lg['fold']}"
        axes[0].plot(ep, [r["train_loss"] for r in h], lw=0.8, alpha=0.75, label=lab)
        axes[0].plot(ep, [r["val_loss"] for r in h], lw=0.8, alpha=0.75, ls="--")
        axes[1].plot(ep, [r["select_value"] for r in h], lw=0.8, alpha=0.75, label=lab)
        be = lg.get("best_epoch")
        if be is not None and be >= 0:
            axes[1].axvline(be, color="grey", lw=0.5, alpha=0.6)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Training (solid) and validation (dashed) loss", fontsize=10)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(str(logs[0].get("select_metric", "select metric")))
    axes[1].set_title("Checkpoint-selection metric on the validation split", fontsize=10)
    axes[1].legend(fontsize=5, ncol=3, loc="lower right")
    return _save(fig, path, dpi)


def plot_confusion(y: np.ndarray, pred: np.ndarray, classes: list[str], title: str,
                   path: Path, dpi: int = 150) -> str:
    """Row-normalised confusion matrix, so a small class is visible next to a large one."""
    k = len(classes)
    m = np.zeros((k, k))
    for i, c in enumerate(classes):
        sel = y == c
        if not np.any(sel):
            continue
        for j, p in enumerate(classes):
            m[i, j] = float(np.mean(pred[sel] == p))
    fig, ax = plt.subplots(figsize=(3.6 + 0.5 * k, 3.2 + 0.5 * k))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), classes, fontsize=8)
    ax.set_yticks(range(k), classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 0.55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path, dpi)


def plot_roc(y: np.ndarray, proba: np.ndarray, classes: list[str], title: str, path: Path,
             dpi: int = 150) -> str | None:
    """One-vs-rest ROC per class, plus the macro average."""
    from sklearn.metrics import auc, roc_curve

    if proba is None or proba.size == 0 or len(classes) < 2:
        return None
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    cols = _colors(classes)
    aucs = []
    for j, c in enumerate(classes):
        truth = (np.asarray(y).astype(str) == str(c)).astype(int)
        if truth.sum() == 0 or truth.sum() == truth.size:
            continue
        fpr, tpr, _ = roc_curve(truth, proba[:, j])
        a = float(auc(fpr, tpr))
        aucs.append(a)
        ax.plot(fpr, tpr, lw=1.4, color=cols[c], label=f"{c} (AUC {a:.3f})")
    ax.plot([0, 1], [0, 1], lw=0.8, ls=":", color="grey")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{title}\nmacro AUC {np.mean(aucs):.3f}" if aucs else title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, path, dpi)


def plot_calibration(y: np.ndarray, proba: np.ndarray, title: str, path: Path, dpi: int = 150,
                     n_bins: int = 10) -> str | None:
    """Reliability diagram. A small dataset gives few points per bin, so the counts are printed."""
    if proba is None or proba.size == 0:
        return None
    classes = sorted({str(v) for v in y})
    yi = np.array([classes.index(str(v)) for v in y])
    conf = proba.max(axis=1)
    correct = (np.argmax(proba, axis=1) == yi).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    xs, ys, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        sel = (conf >= a) & (conf < b if b < 1 else conf <= b)
        if sel.sum() == 0:
            continue
        xs.append(float(conf[sel].mean()))
        ys.append(float(correct[sel].mean()))
        ns.append(int(sel.sum()))
    ax.plot(xs, ys, "o-", lw=1.2, ms=4, color="#0072B2")
    ax.plot([0, 1], [0, 1], ls=":", lw=0.9, color="grey")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\nbin counts: {ns}", fontsize=9)
    return _save(fig, path, dpi)


def plot_model_comparison(summary: dict[str, Any], path: Path, dpi: int = 150) -> str | None:
    """Balanced accuracy per model with 95% bootstrap intervals, on the files each model scored."""
    models = summary.get("models") or {}
    names, vals, los, his, ns = [], [], [], [], []
    for name, ent in models.items():
        if name.startswith("dummy"):
            continue
        agg = (ent.get("aggregated") or {})
        ba = agg.get("balanced_accuracy") or {}
        ci = agg.get("balanced_accuracy_bootstrap_ci95") or {}
        if ba.get("value") is None:
            continue
        names.append(name)
        vals.append(float(ba["value"]))
        los.append(float(ci.get("lo", ba["value"])) if ci.get("lo") is not None else float(ba["value"]))
        his.append(float(ci.get("hi", ba["value"])) if ci.get("hi") is not None else float(ba["value"]))
        ns.append(int(ent.get("n_files_evaluated", 0)))
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ypos = np.arange(len(names))
    err = np.array([[v - lo for v, lo in zip(vals, los, strict=True)],
                    [hi - v for v, hi in zip(vals, his, strict=True)]])
    ax.errorbar(vals, ypos, xerr=err, fmt="o", color="#0072B2", ecolor="#6699CC",
                capsize=3, lw=1.3, ms=5)
    ax.set_yticks(ypos, [f"{n} (n={k})" for n, k in zip(names, ns, strict=True)], fontsize=8)
    ax.axvline(1.0 / 2, ls=":", color="grey", lw=0.9)
    ax.set_xlabel("balanced accuracy (aggregated over repeats)")
    ax.set_xlim(0, 1.02)
    ax.set_title("Model comparison with 95% file-level bootstrap intervals", fontsize=10)
    ax.text(0.5, -0.28, "dotted line: two-class chance. Intervals are file-level; there are no "
            "replicate groups.", transform=ax.transAxes, fontsize=6.5, ha="center", color="#555555")
    return _save(fig, path, dpi)


def plot_embedding(rep: Any, classes: list[str], path: Path, dpi: int = 150) -> str | None:
    """PCA of the out-of-fold file embeddings. PCA, not t-SNE or UMAP: it is deterministic."""
    if rep is None or rep.emb.shape[0] < 4:
        return None
    from sklearn.decomposition import PCA

    z = PCA(n_components=min(2, rep.emb.shape[0] - 1, rep.emb.shape[1])).fit_transform(rep.emb)
    if z.shape[1] < 2:
        z = np.hstack([z, np.zeros((z.shape[0], 1))])
    cols = _colors(classes)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for c in classes:
        m = rep.labels == c
        if np.any(m):
            ax.scatter(z[m, 0], z[m, 1], s=22, color=cols[c], label=f"{c} (n={int(m.sum())})",
                       edgecolor="white", linewidth=0.4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Out-of-fold CNN embeddings, one point per file", fontsize=10)
    ax.legend(fontsize=7)
    return _save(fig, path, dpi)


def plot_attribution(ws: Any, window: np.ndarray, attrs: dict[str, np.ndarray], w_i: int,
                     path: Path, dpi: int = 150) -> str | None:
    """One window: the trace, then one attribution panel per method, on a shared time axis.

    ``attrs`` maps a method name to its attribution array for this single window.
    """
    if ws is None or ws.x.shape[0] == 0:
        return None
    methods = list(attrs)
    n_pan = 1 + len(methods)
    fig, axes = plt.subplots(n_pan, 1, figsize=(7.4, 1.7 * n_pan), sharex=True)
    if n_pan == 1:
        axes = [axes]
    t = np.arange(ws.x.shape[1]) / ws.rate_hz * 1000.0
    axes[0].plot(t, window, lw=0.8, color="#333333")
    axes[0].set_ylabel("window")
    axes[0].set_title(f"window {w_i} (file {ws.sample_id[w_i]}, class {ws.class_label[w_i]})",
                      fontsize=9)
    for ax, m in zip(axes[1:], methods, strict=False):
        a = np.asarray(attrs[m], dtype=float).reshape(-1)
        ax.plot(t[:a.size], a, lw=0.9, color="#D55E00")
        ax.fill_between(t[:a.size], 0, a, color="#D55E00", alpha=0.2)
        ax.set_ylabel(m, fontsize=8)
    axes[-1].set_xlabel("time within window (ms)")
    return _save(fig, path, dpi)


def plot_feature_importance(expl: dict[str, Any], path: Path, dpi: int = 150) -> str | None:
    """Top mean |SHAP| features per class for the Model A classifiers."""
    feats = ((expl.get("methods") or {}).get("features") or {}).get("models") or {}
    panels = [(n, d.get("top_features_by_class") or {}) for n, d in feats.items()
              if d.get("top_features_by_class")]
    if not panels:
        return None
    name, top = panels[0]
    cls = list(top)
    fig, axes = plt.subplots(1, len(cls), figsize=(4.6 * len(cls), 4.2), squeeze=False)
    for ax, c in zip(axes[0], cls, strict=False):
        items = top[c][:12][::-1]
        ax.barh([i[0] for i in items], [i[1] for i in items], color="#0072B2")
        ax.set_title(f"{name}: class {c}", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("mean |SHAP|", fontsize=8)
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# Dataset, feature, tuning and error figures
#
# The families above are about the signal and about the fit. These are about the *inputs* and
# about *where the models fail*, and every one of them has a machine-readable twin: the numbers a
# figure draws are written to a CSV by the matching ``*_rows`` function, so a reader who cannot see
# the PNG can still audit it. A figure is never the only record of a claim.
# ---------------------------------------------------------------------------

# Every category the run can ask for, and what it shows. ``figures_index.csv`` reports one row per
# category with its generation status, so "we did not draw it" is distinguishable from "we forgot".
FIGURE_CATEGORIES: dict[str, str] = {
    "dataset_composition": "files and detected events per class, and the imbalance ratio that "
                           "decides whether class weights are used",
    "signal_examples": "analysed trace, rolling-median baseline and detected events, per example file",
    "class_mean_window": "mean event window per class at the analysis rate",
    "psd_by_class": "mean Welch band power per class",
    "feature_distributions": "per-class distribution of the most separated features, by domain",
    "missing_values": "which features and which files carry NaN, i.e. where a domain was undefined",
    "feature_correlation": "|Spearman| between features, with the declared redundancy pairs marked",
    "deep_learning_curves": "training and validation loss per fold for the deep encoder",
    "confusion": "confusion matrix per model, on that model's evaluation mask",
    "roc": "one-vs-rest ROC per model from out-of-fold probabilities",
    "calibration": "reliability diagram per model",
    "model_comparison": "primary metric per model with bootstrap intervals",
    "embedding": "PCA of the out-of-fold learned representation",
    "attribution": "temporal attribution of explained windows (Grad-CAM, integrated gradients, "
                   "occlusion)",
    "feature_importance": "mean |SHAP| per class for the Model A classifiers",
    "ablation": "primary metric per feature-subset variant against the all-groups baseline",
    "hpo": "every trial of every nested search, failures included",
    "error_analysis": "per-file error rate across repeats, the errors only, and confidence of "
                      "correct versus wrong calls",
    "pr_curves": "one-vs-rest precision-recall curves with average precision per class",
    "quad_gates": "mean cross-branch fusion gate weight: which representation carried the call",
    "shot_curve": "one-shot/few-shot episode accuracy versus support shots, with intervals",
    "branch_ablation": "fused four-branch model versus each branch-ablated variant",
    "quad_embedding": "PCA and t-SNE of the shared four-branch embedding, coloured by class",
}

# Columns of a feature row that describe the file rather than the signal.
_META_KEYS = frozenset({"sample_id", "class", "group_id", "filename", "status", "reason",
                        "duration_s", "rate_hz", "n_samples", "original_path", "source_path",
                        "size_bytes", "sha256", "skipped"})


def _group_of(name: str) -> str:
    from .features import feature_group

    try:
        return str(feature_group(name))
    except Exception:
        return "other"


def _fnum(v: Any) -> float | None:
    """A float, or None for anything that is not a finite number.

    Numeric strings count: ``hpo_trials.csv`` and ``ablation.csv`` are read back as text, and a
    trial whose score came from a CSV must not be drawn as a failed trial.
    """
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def dataset_composition_rows(rows: list[dict[str, Any]], classes: list[str]) -> list[dict[str, Any]]:
    """Per-class file and event counts, plus how much of each class is missing by construction."""
    out: list[dict[str, Any]] = []
    for c in classes:
        sub = [r for r in rows if str(r.get("class")) == c]
        ev = [v for v in (_fnum(r.get("event_n")) for r in sub) if v is not None]
        dur = [v for v in (_fnum(r.get("duration_s")) for r in sub) if v is not None]
        miss = []
        for r in sub:
            vals = [_fnum(v) for k, v in r.items() if k not in _META_KEYS]
            if vals:
                miss.append(sum(1 for v in vals if v is None) / len(vals))
        out.append({
            "class": c, "n_files": len(sub), "n_files_with_events": len(ev),
            "n_files_without_events": sum(1 for v in ev if v == 0),
            "events_total": int(sum(ev)),
            "events_per_file_mean": round(float(np.mean(ev)), 4) if ev else None,
            "events_per_file_sd": round(float(np.std(ev, ddof=1)), 4) if len(ev) > 1 else None,
            "events_per_file_min": min(ev) if ev else None,
            "events_per_file_max": max(ev) if ev else None,
            "duration_s_total": round(float(sum(dur)), 3) if dur else None,
            "mean_missing_feature_fraction": round(float(np.mean(miss)), 4) if miss else None,
        })
    return out


def plot_dataset_composition(rows: list[dict[str, Any]], classes: list[str], imbalance: float,
                             path: Path, dpi: int = 150) -> str | None:
    """What the dataset actually contains, before any model sees it."""
    recs = dataset_composition_rows(rows, classes)
    if not recs or not any(r["n_files"] for r in recs):
        return None
    cols = _colors(classes)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax = axes[0]
    ax.bar([r["class"] for r in recs], [r["n_files"] for r in recs],
           color=[cols.get(r["class"], "#888888") for r in recs])
    for i, r in enumerate(recs):
        ax.text(i, float(r["n_files"]), str(r["n_files"]), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("files")
    ax.set_title(f"files per class\nimbalance ratio {imbalance:.3f} "
                 f"({'no class weights' if imbalance < 1.5 else 'class weights would apply'})",
                 fontsize=9)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ev_by_class: dict[str, list[float]] = {c: [] for c in classes}
    for r in rows:
        v = _fnum(r.get("event_n"))
        c = str(r.get("class"))
        if v is not None and c in ev_by_class:
            ev_by_class[c].append(v)
    data = [ev_by_class[c] for c in classes]
    keep = [(c, d) for c, d in zip(classes, data, strict=True) if d]
    if keep:
        bp = ax.boxplot([d for _, d in keep], tick_labels=[c for c, _ in keep], widths=0.55,
                        patch_artist=True)
        for patch, (c, _) in zip(bp["boxes"], keep, strict=True):
            patch.set_facecolor(cols.get(c, "#888888"))
            patch.set_alpha(0.7)
        for i, (_, d) in enumerate(keep):
            ax.scatter(np.full(len(d), i + 1) + np.linspace(-0.12, 0.12, len(d)), d, s=9,
                       color="#333333", zorder=3)
    ax.set_ylabel("detected events per 20 s file")
    ax.set_title("event count per file, by class", fontsize=9)
    ax.tick_params(labelsize=8)

    ax = axes[2]
    width = 0.38
    idx = np.arange(len(recs))
    no_ev = [100.0 * float(r["n_files_without_events"] or 0) / max(1, int(r["n_files"]))
             for r in recs]
    miss = [100.0 * float(r["mean_missing_feature_fraction"] or 0.0) for r in recs]
    ax.bar(idx - width / 2, no_ev, width, label="files with no detected event", color="#D55E00")
    ax.bar(idx + width / 2, miss, width, label="mean missing feature fraction", color="#0072B2")
    ax.set_xticks(idx, [r["class"] for r in recs])
    ax.set_ylabel("% of files / % of features")
    ax.set_title("data quality per class", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.85)
    ax.tick_params(labelsize=8)
    return _save(fig, path, dpi)


def _separation(x: np.ndarray, y: np.ndarray, classes: list[str]) -> np.ndarray:
    """Standardised mean difference across classes, per column. Display ordering only."""
    out = np.zeros(x.shape[1], dtype=float)
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx.get(str(v), -1) for v in y.tolist()], dtype=int)
    for j in range(x.shape[1]):
        col = np.asarray(x[:, j], dtype=float)
        ok = np.isfinite(col)
        if ok.sum() < 3:
            continue
        sd = float(np.std(col[ok], ddof=1)) if ok.sum() > 1 else 0.0
        if not math.isfinite(sd) or sd <= 0:
            continue
        means = [float(np.mean(col[ok & (yi == i)])) for i in range(len(classes))
                 if (ok & (yi == i)).sum() > 0]
        if len(means) >= 2:
            out[j] = (max(means) - min(means)) / sd
    return out


def display_feature_order(x: np.ndarray, names: list[str], y: np.ndarray, classes: list[str],
                          max_features: int = 24, per_group: int = 4) -> list[int]:
    """Column indices to draw, most separated first, capped per feature domain.

    This orders a *figure*. It is computed over all files and is never fed back into a model: the
    models use the declared feature groups, and the ablation is what tests whether a domain earns
    its place. The cap per domain keeps one large domain from hiding the others.
    """
    sep = _separation(x, y, classes)
    order = [int(j) for j in np.argsort(-sep)]
    chosen: list[int] = []
    per: dict[str, int] = {}
    for j in order:
        g = _group_of(str(names[j]))
        if per.get(g, 0) >= per_group:
            continue
        per[g] = per.get(g, 0) + 1
        chosen.append(j)
        if len(chosen) >= max_features:
            break
    return sorted(chosen, key=lambda j: (-sep[j], str(names[j])))


def feature_distribution_rows(x: np.ndarray, names: list[str], y: np.ndarray,
                              classes: list[str]) -> list[dict[str, Any]]:
    """Per feature and class: n, mean, sd, quartiles and missing count. The CSV twin."""
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx.get(str(v), -1) for v in y.tolist()], dtype=int)
    out: list[dict[str, Any]] = []
    for j, nm in enumerate(names):
        col = np.asarray(x[:, j], dtype=float)
        for c in classes:
            v = col[yi == idx[c]]
            fin = v[np.isfinite(v)]
            out.append({
                "feature": str(nm), "group": _group_of(str(nm)), "class": c, "n": int(v.size),
                "n_missing": int(v.size - fin.size),
                "mean": round(float(np.mean(fin)), 6) if fin.size else None,
                "sd": round(float(np.std(fin, ddof=1)), 6) if fin.size > 1 else None,
                "min": round(float(np.min(fin)), 6) if fin.size else None,
                "q25": round(float(np.percentile(fin, 25)), 6) if fin.size else None,
                "median": round(float(np.median(fin)), 6) if fin.size else None,
                "q75": round(float(np.percentile(fin, 75)), 6) if fin.size else None,
                "max": round(float(np.max(fin)), 6) if fin.size else None,
            })
    return out


def plot_feature_distributions(x: np.ndarray, names: list[str], y: np.ndarray, classes: list[str],
                               path: Path, dpi: int = 150, max_features: int = 24) -> str | None:
    """Boxplots per class for the most separated features, at most four per domain."""
    if x.size == 0 or not classes:
        return None
    chosen = display_feature_order(x, names, y, classes, max_features=max_features)
    if not chosen:
        return None
    cols = _colors(classes)
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx.get(str(v), -1) for v in y.tolist()], dtype=int)
    ncol = 4
    nrow = math.ceil(len(chosen) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow), squeeze=False)
    for p, j in enumerate(chosen):
        ax = axes[p // ncol][p % ncol]
        col = np.asarray(x[:, j], dtype=float)
        data, ticks, colors = [], [], []
        for c in classes:
            v = col[yi == idx[c]]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            data.append(v)
            ticks.append(c)
            colors.append(cols.get(c, "#888888"))
        if not data:
            ax.set_title(f"{names[j]}\n(all missing)", fontsize=7)
            ax.axis("off")
            continue
        bp = ax.boxplot(data, tick_labels=ticks, widths=0.6, patch_artist=True)
        for patch, cc in zip(bp["boxes"], colors, strict=True):
            patch.set_facecolor(cc)
            patch.set_alpha(0.7)
        ax.set_title(f"{names[j]}  [{_group_of(str(names[j]))}]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylabel("value", fontsize=7)
    for p in range(len(chosen), nrow * ncol):
        axes[p // ncol][p % ncol].axis("off")
    fig.suptitle("feature distributions by class (most separated first, <=4 per domain; "
                 "display order only, not a model input selection)", fontsize=9)
    return _save(fig, path, dpi)


def missing_value_rows(x: np.ndarray, names: list[str], sids: list[str]
                       ) -> list[dict[str, Any]]:
    """Per-feature and per-file missingness, in one long table. The CSV twin."""
    out: list[dict[str, Any]] = []
    n = x.shape[0]
    for j, nm in enumerate(names):
        col = np.asarray(x[:, j], dtype=float)
        bad = ~np.isfinite(col)
        out.append({"row_type": "feature", "key": str(nm), "group": _group_of(str(nm)),
                    "n_rows": int(n), "n_missing": int(bad.sum()),
                    "fraction_missing": round(float(bad.sum()) / n, 6) if n else None,
                    "all_missing": bool(n > 0 and bad.all())})
    for i, sid in enumerate(sids):
        row = np.asarray(x[i], dtype=float)
        bad = ~np.isfinite(row)
        out.append({"row_type": "file", "key": str(sid), "group": "",
                    "n_rows": int(row.size), "n_missing": int(bad.sum()),
                    "fraction_missing": round(float(bad.sum()) / row.size, 6) if row.size else None,
                    "all_missing": bool(row.size > 0 and bad.all())})
    return out


def plot_missing_pattern(x: np.ndarray, names: list[str], sids: list[str], path: Path,
                         dpi: int = 150) -> str | None:
    """Where the feature table is NaN, per feature and per file."""
    if x.size == 0:
        return None
    n = x.shape[0]
    bad = ~np.isfinite(np.asarray(x, dtype=float))
    per_feat = bad.mean(axis=0) if n else np.zeros(x.shape[1])
    per_file = bad.mean(axis=1) if x.shape[1] else np.zeros(n)
    # Cap both panels: 90 features or 135 files would make an unreadable 30-inch figure, so the
    # panel shows the worst offenders and the CSV twin carries every feature and every file.
    shown = [int(j) for j in np.argsort(-per_feat) if per_feat[j] > 0][:40]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.4, 0.2 * len(shown) + 1.8)))

    ax = axes[0]
    if shown:
        ax.barh([str(names[j]) for j in shown][::-1], [per_feat[j] for j in shown][::-1],
                color=["#D55E00" if per_feat[j] >= 1.0 else "#0072B2" for j in shown][::-1])
        ax.set_xlim(0, max(1.0, float(per_feat.max())))
        ax.tick_params(labelsize=6)
    else:
        ax.text(0.5, 0.5, "no feature is missing in any file", ha="center", va="center",
                fontsize=9, transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_xlabel("fraction of files where the feature is NaN")
    n_ever = int((per_feat > 0).sum())
    ax.set_title(f"per feature ({n_ever} of {x.shape[1]} ever missing, "
                 f"{int((per_feat >= 1.0).sum())} always missing"
                 + (f"; {min(len(shown), n_ever)} shown" if n_ever > len(shown) else "") + ")",
                 fontsize=9)

    ax = axes[1]
    if n <= 40:
        ax.barh([str(v) for v in sids][::-1], per_file[::-1], color="#009E73")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("fraction of features that are NaN for this file")
    else:
        ax.hist(per_file, bins=25, color="#009E73")
        ax.set_xlabel("fraction of features that are NaN")
        ax.set_ylabel("files")
    ax.set_title(f"per file (n={n}, mean {float(per_file.mean()) if per_file.size else 0.0:.3f}, "
                 f"max {float(per_file.max()) if per_file.size else 0.0:.3f})", fontsize=9)
    return _save(fig, path, dpi)


def feature_correlation_rows(x: np.ndarray, names: list[str], threshold: float = 0.98,
                             report_at: float = 0.80) -> list[dict[str, Any]]:
    """Spearman pairs above ``report_at``, flagged when they reach the redundancy ``threshold``."""
    from scipy.stats import spearmanr

    fin = np.asarray(x, dtype=float)
    keep = [j for j in range(fin.shape[1])
            if np.isfinite(fin[:, j]).sum() >= 3 and np.nanstd(fin[:, j]) > 0]
    if len(keep) < 2:
        return []
    rho = spearmanr(fin[:, keep]).statistic
    rho = np.atleast_2d(np.asarray(rho, dtype=float))
    out: list[dict[str, Any]] = []
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            v = float(rho[a, b])
            if not math.isfinite(v) or abs(v) < report_at:
                continue
            out.append({"feature_a": str(names[keep[a]]), "feature_b": str(names[keep[b]]),
                        "group_a": _group_of(str(names[keep[a]])),
                        "group_b": _group_of(str(names[keep[b]])),
                        "spearman": round(v, 6), "abs_spearman": round(abs(v), 6),
                        "redundant_at_threshold": bool(abs(v) >= threshold)})
    out.sort(key=lambda r: -float(r["abs_spearman"]))
    return out


def plot_feature_correlation(x: np.ndarray, names: list[str], y: np.ndarray, classes: list[str],
                             path: Path, dpi: int = 150, threshold: float = 0.98,
                             max_features: int = 30) -> str | None:
    """|Spearman| heatmap over the drawn features, plus the strongest pairs as a bar."""
    if x.size == 0:
        return None
    from scipy.stats import spearmanr

    chosen = display_feature_order(x, names, y, classes, max_features=max_features, per_group=10**6)
    if len(chosen) < 2:
        return None
    sub = np.asarray(x[:, chosen], dtype=float)
    with np.errstate(invalid="ignore"):
        rho = np.atleast_2d(np.asarray(spearmanr(sub).statistic, dtype=float))
    rho = np.where(np.isfinite(rho), rho, 0.0)
    labels = [str(names[j]) for j in chosen]

    fig = plt.figure(figsize=(14, 5.6))
    ax = fig.add_subplot(1, 2, 1)
    im = ax.imshow(np.abs(rho), cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=5)
    ax.set_yticks(range(len(labels)), labels, fontsize=5)
    ax.set_title(f"|Spearman| over {len(labels)} drawn features\n"
                 f"(pairs at |rho| >= {threshold:g} are reported, not dropped)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = fig.add_subplot(1, 2, 2)
    pairs = feature_correlation_rows(x, names, threshold=threshold, report_at=threshold)[:14]
    if pairs:
        lab = [f"{p['feature_a']} | {p['feature_b']}" for p in pairs][::-1]
        val = [float(p["abs_spearman"]) for p in pairs][::-1]
        ax.barh(lab, val, color="#D55E00")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("|Spearman|")
        ax.set_title(f"{len(pairs)} pair(s) at or above the redundancy threshold", fontsize=9)
        ax.tick_params(labelsize=5)
    else:
        ax.text(0.5, 0.5, f"no pair reaches |rho| >= {threshold:g}", ha="center", va="center",
                fontsize=9, transform=ax.transAxes)
        ax.set_axis_off()
    return _save(fig, path, dpi)


def plot_ablation(rows: list[dict[str, Any]], path: Path, dpi: int = 150) -> str | None:
    """Every ablation variant against the all-groups baseline, from the recorded rows."""
    vals: list[tuple[dict[str, Any], float]] = []
    for r in rows:
        v = _fnum(r.get("balanced_accuracy_mean"))
        if v is not None:
            vals.append((r, v))
    if not vals:
        return None
    base = next((v for r, v in vals if str(r.get("variant")) == "all_groups"), None)
    kind_color = {"baseline": "#333333", "only": "#0072B2", "without": "#D55E00",
                  "both": "#009E73"}
    order = sorted(vals, key=lambda t: -float(t[1]))
    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(order) + 1.6))
    labels = [f"{r.get('variant')}  ({r.get('n_features')} features)" for r, _ in order]
    bars = ax.barh(labels[::-1], [float(v) for _, v in order][::-1],
                   color=[kind_color.get(str(r.get("kind")), "#888888") for r, _ in order][::-1])
    if base is not None:
        ax.axvline(float(base), ls="--", lw=1.0, color="#333333",
                   label=f"all_groups baseline = {float(base):.3f}")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
    for bar, (r, v) in zip(bars, list(order)[::-1], strict=True):
        sd = _fnum(r.get("balanced_accuracy_sd"))
        note = f"{float(v):.3f}" + (f" ±{sd:.3f}" if sd is not None else "")
        if base is not None and str(r.get("variant")) != "all_groups":
            note += f"  (Δ {float(v) - float(base):+.3f})"
        ax.text(float(v) + 0.004, bar.get_y() + bar.get_height() / 2, note, va="center",
                fontsize=6)
    ax.set_xlabel("balanced accuracy (mean over repeats, out-of-fold)")
    ax.set_xlim(0, 1.12)
    ax.set_title("feature-domain ablation, refitted with the declared hyperparameters "
                 "(no tuning, so the variants are comparable)", fontsize=9)
    ax.tick_params(labelsize=6)
    return _save(fig, path, dpi)


def plot_hpo(rows: list[dict[str, Any]], path: Path, dpi: int = 150) -> str | None:
    """Every trial of every search: the spread, the failures, and the setting each fold chose."""
    if not rows:
        return None
    models = sorted({str(r.get("model")) for r in rows})[:4]
    fig, axes = plt.subplots(len(models), 1, figsize=(11.5, 2.6 * len(models)), squeeze=False,
                             sharex=False)
    for ax, name in zip(axes[:, 0], models, strict=True):
        sub = [r for r in rows if str(r.get("model")) == name]
        tags = sorted({str(r.get("fold_tag")) for r in sub})
        pos = {t: i for i, t in enumerate(tags)}
        metric = str(sub[0].get("metric"))
        seen_sc = seen_fail = seen_rung = False
        for r in sub:
            i = pos.get(str(r.get("fold_tag")))
            if i is None:
                continue
            sc = _fnum(r.get("score"))
            rung = int(_fnum(r.get("rung")) or 0)
            jit = ((int(_fnum(r.get("trial")) or 0) % 5) - 2) * 0.045
            if sc is None:
                ax.scatter(i + jit, 0.0, marker="x", color="#D55E00", s=34, zorder=3,
                           label=None if seen_fail else "failed trial (no score)")
                seen_fail = True
                continue
            if rung > 0:
                lab = None if seen_rung else "promoted trial (halving rung)"
                seen_rung = True
            else:
                lab = None if seen_sc else "scored trial"
                seen_sc = True
            ax.scatter(i + jit, sc, s=18 + 12 * rung, color="#0072B2", alpha=0.75, zorder=3,
                       marker="o" if rung == 0 else "s", label=lab)
        best: dict[str, float] = {}
        for r in sub:
            sc = _fnum(r.get("score"))
            if sc is None:
                continue
            t = str(r.get("fold_tag"))
            best[t] = max(best.get(t, -math.inf), sc)
        if best:
            ax.scatter([pos[t] for t in best], [best[t] for t in best], marker="*", s=150,
                       color="#009E73", zorder=4, label="chosen for that fold")
        ax.set_xticks(list(pos.values()), list(pos), fontsize=6, rotation=90)
        ax.set_ylabel(metric, fontsize=8)
        n_fail = sum(1 for r in sub if _fnum(r.get("score")) is None)
        ax.set_title(f"{name}: {len(sub)} trials over {len(tags)} folds "
                     f"({str(sub[0].get('method'))} search, objective {metric}, "
                     f"{n_fail} failed and dropped not zeroed)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-0.06, 1.06)
        ax.legend(fontsize=6, loc="lower right", ncol=3, framealpha=0.85)
    return _save(fig, path, dpi)


def error_analysis_rows(y: np.ndarray, pred: np.ndarray, proba: np.ndarray, sids: list[str],
                        classes: list[str], name: str) -> list[dict[str, Any]]:
    """Per file: how often each repeat got it wrong, and how confident the model was."""
    idx = {c: i for i, c in enumerate(classes)}
    n_rep = int(np.asarray(pred).shape[0]) if np.ndim(pred) else 1
    out: list[dict[str, Any]] = []
    for i, sid in enumerate(sids):
        truth = str(y[i])
        labs = [str(pred[r, i]) for r in range(n_rep) if str(pred[r, i]) != ""]
        if not labs:
            continue
        wrong = [p for p in labs if p != truth]
        conf_c, conf_w = [], []
        for r in range(n_rep):
            p = np.asarray(proba[r, i], dtype=float) if proba is not None else None
            if p is None or p.size != len(classes) or not np.all(np.isfinite(p)):
                continue
            j = idx.get(truth)
            mx = float(np.max(p))
            if j is not None and str(pred[r, i]) == truth:
                conf_c.append(float(p[j]))
            elif j is not None:
                conf_w.append(float(p[j]))
            out.append({"row_type": "call", "model": name, "sample_id": str(sid), "class": truth,
                        "repeat": r, "predicted": str(pred[r, i]),
                        "correct": bool(str(pred[r, i]) == truth),
                        "prob_of_true_class": round(float(p[j]), 6) if j is not None else None,
                        "max_prob": round(mx, 6),
                        "margin_top2": round(mx - float(np.sort(p)[-2]), 6) if p.size > 1 else None})
        counts = {c: wrong.count(c) for c in classes if wrong.count(c)}
        modal = max(counts, key=lambda c: counts[c]) if counts else None
        out.append({"row_type": "file", "model": name, "sample_id": str(sid), "class": truth,
                    "n_repeats_scored": len(labs), "n_wrong": len(wrong),
                    "error_rate": round(len(wrong) / len(labs), 6),
                    "modal_wrong_class": modal,
                    "wrong_class_counts": "|".join(f"{c}:{counts[c]}" for c in sorted(counts)),
                    "mean_prob_of_true_class_when_correct":
                        round(float(np.mean(conf_c)), 6) if conf_c else None,
                    "mean_prob_of_true_class_when_wrong":
                        round(float(np.mean(conf_w)), 6) if conf_w else None,
                    "failure_case": bool(len(wrong) * 2 >= len(labs))})
    return out


def plot_error_analysis(y: np.ndarray, pred: np.ndarray, proba: np.ndarray, sids: list[str],
                        classes: list[str], name: str, path: Path, dpi: int = 150) -> str | None:
    """Where one model fails: which files, into which class, and with what confidence."""
    rows = [r for r in error_analysis_rows(y, pred, proba, sids, classes, name)
            if r.get("row_type") == "file"]
    if not rows:
        return None
    cols = _colors(classes)
    rows.sort(key=lambda r: (-float(r.get("error_rate") or 0.0), str(r.get("sample_id"))))
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.3 * min(len(rows), 40) + 2.4))

    ax = axes[0]
    shown = rows[:40]
    ax.barh([f"{r['sample_id']} ({r['class']})" for r in shown][::-1],
            [float(r.get("error_rate") or 0.0) for r in shown][::-1],
            color=[cols.get(str(r.get("class")), "#888888") for r in shown][::-1])
    ax.set_xlabel("fraction of repeats that misclassified this file")
    ax.set_title(f"per-file error rate ({len(rows)} files"
                 + (f", {len(shown)} hardest shown" if len(rows) > len(shown) else "")
                 + ")", fontsize=9)
    ax.tick_params(labelsize=6)
    ax.set_xlim(0, 1.02)

    ax = axes[1]
    pairs: dict[tuple[str, str], int] = {}
    for r in rows:
        for part in str(r.get("wrong_class_counts") or "").split("|"):
            if not part:
                continue
            c, n = part.rsplit(":", 1)
            pairs[(str(r.get("class")), c)] = pairs.get((str(r.get("class")), c), 0) + int(n)
    if pairs:
        items = sorted(pairs.items(), key=lambda kv: -kv[1])[:12]
        ax.barh([f"{a} → {b}" for (a, b), _ in items][::-1], [n for _, n in items][::-1],
                color="#D55E00")
        ax.set_xlabel("misclassifications (summed over repeats)")
        ax.set_title("the errors only: true class → predicted class", fontsize=9)
        ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "no misclassification in any repeat", ha="center", va="center",
                fontsize=9, transform=ax.transAxes)
        ax.set_axis_off()

    ax = axes[2]
    calls = [r for r in error_analysis_rows(y, pred, proba, sids, classes, name)
             if r.get("row_type") == "call" and _fnum(r.get("prob_of_true_class")) is not None]
    c_ok = [float(r["prob_of_true_class"]) for r in calls if r.get("correct")]
    c_no = [float(r["prob_of_true_class"]) for r in calls if not r.get("correct")]
    bins = np.linspace(0, 1, 21)
    if c_ok:
        ax.hist(c_ok, bins=bins, alpha=0.75, color="#009E73", label=f"correct ({len(c_ok)})")
    if c_no:
        ax.hist(c_no, bins=bins, alpha=0.75, color="#D55E00", label=f"wrong ({len(c_no)})")
    ax.set_xlabel("probability assigned to the true class")
    ax.set_ylabel("calls")
    ax.set_title("confidence of correct versus wrong calls", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.85)
    ax.tick_params(labelsize=7)
    return _save(fig, path, dpi)


def write_figures_index(statuses: list[dict[str, Any]], path: Path) -> str:
    """``figures_index.csv``: one row per requested category, with its generation status.

    A category that was skipped says why, so the reader can tell "not applicable to this run" from
    "requested and failed". Written even when nothing was drawn.
    """
    import csv
    import io

    from .runs import write_text_exclusive

    fields = ["category", "figure", "file", "status", "reason", "csv_twin", "description"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for s in statuses:
        row = {k: ("" if s.get(k) is None else s.get(k)) for k in fields}
        row.setdefault("description", FIGURE_CATEGORIES.get(str(s.get("category")), ""))
        if not row.get("description"):
            row["description"] = FIGURE_CATEGORIES.get(str(s.get("category")), "")
        w.writerow(row)
    return str(write_text_exclusive(path, buf.getvalue()))


# ---------------------------------------------------------------------------
# Four-branch diagnostics and the extended evaluation figures
# ---------------------------------------------------------------------------

def plot_quad_gates(gates: dict[str, float], path: Path, dpi: int = 150) -> str | None:
    """Mean fusion gate weight per branch: which representation the fusion actually used."""
    if not gates:
        return None
    names = list(gates)
    vals = [float(gates[n]) for n in names]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(names, vals, color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"][:len(names)])
    ax.axhline(1.0 / len(names), ls="--", lw=0.9, color="#333333",
               label=f"uniform = {1.0 / len(names):.3f}")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mean gate weight (softmax over branch tokens)")
    ax.set_title("cross-branch attention fusion: contribution per branch", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.85)
    ax.tick_params(labelsize=8)
    return _save(fig, path, dpi)


def plot_shot_curve(rows: list[dict[str, Any]], path: Path, dpi: int = 150) -> str | None:
    """Episode accuracy versus support shots, with the Wilson interval over pooled queries."""
    rows = [r for r in rows if _fnum(r.get("episode_accuracy_mean")) is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: float(r["shot"]))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = [float(r["shot"]) for r in rows]
    y = [float(r["episode_accuracy_mean"]) for r in rows]
    sd = [float(r["episode_accuracy_sd"]) if _fnum(r.get("episode_accuracy_sd")) is not None
          else 0.0 for r in rows]
    lo = [float(y[i] - sd[i]) for i in range(len(rows))]
    hi = [float(y[i] + sd[i]) for i in range(len(rows))]
    ax.errorbar(x, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)],
                marker="o", capsize=3, color="#0072B2", label="episode accuracy ± sd")
    if all(_fnum(r.get("wilson_ci95_lo")) is not None for r in rows):
        ax.fill_between(x, [float(r["wilson_ci95_lo"]) for r in rows],
                        [float(r["wilson_ci95_hi"]) for r in rows], alpha=0.18, color="#0072B2",
                        label="Wilson 95% CI over pooled queries")
    ax.set_xlabel("support examples per class (shot)")
    ax.set_ylabel("episode accuracy")
    ax.set_title("one-shot / few-shot performance of the metric embedding "
                 "(closed-set episodes, disjoint support and query)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, framealpha=0.85)
    ax.tick_params(labelsize=8)
    return _save(fig, path, dpi)


def plot_branch_ablation(summary: dict[str, Any], path: Path, dpi: int = 150) -> str | None:
    """Fused model versus each branch-ablated variant, from the recorded run metrics."""
    models = summary.get("models") or {}
    names = [n for n in models if n == "quad" or str(n).startswith("quad_minus_")]
    vals = []
    for n in names:
        ba = ((models[n].get("aggregated") or {}).get("balanced_accuracy") or {})
        v = _fnum(ba.get("value"))
        if v is not None:
            vals.append((n, v))
    if len(vals) < 2:
        return None
    vals.sort(key=lambda t: -t[1])
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(vals) + 1.6))
    base = next((v for n, v in vals if n == "quad"), None)
    ax.barh([n for n, _ in vals][::-1], [v for _, v in vals][::-1],
            color=["#009E73" if n == "quad" else "#D55E00" for n, _ in vals][::-1])
    if base is not None:
        ax.axvline(base, ls="--", lw=0.9, color="#333333", label=f"fused = {base:.3f}")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
    for i, (n, v) in enumerate(list(vals)[::-1]):
        note = f"{v:.3f}"
        if base is not None and n != "quad":
            note += f"  (Δ {v - base:+.3f})"
        ax.text(v + 0.004, i, note, va="center", fontsize=7)
    ax.set_xlabel("balanced accuracy (aggregated over repeats, out-of-fold)")
    ax.set_xlim(0, 1.12)
    ax.set_title("branch ablation of the four-branch fusion (same folds, same files)", fontsize=9)
    ax.tick_params(labelsize=7)
    return _save(fig, path, dpi)


def plot_embedding_xy(emb: np.ndarray, y: np.ndarray, classes: list[str], path: Path,
                      dpi: int = 150, method: str = "pca") -> str | None:
    """PCA or t-SNE of any embedding matrix, coloured by class."""
    emb = np.asarray(emb, dtype=float)
    fin = np.all(np.isfinite(emb), axis=1)
    if fin.sum() < 4 or emb.shape[1] < 2:
        return None
    e = emb[fin]
    lab = np.asarray(y, dtype=object)[fin]
    if method == "tsne":
        from sklearn.manifold import TSNE

        if e.shape[0] < 5:
            return None
        z = TSNE(n_components=2, random_state=0, perplexity=min(30.0, max(5.0, e.shape[0] / 4)),
                 init="pca", learning_rate="auto").fit_transform(e)
    else:
        from sklearn.decomposition import PCA

        z = PCA(n_components=2, random_state=0).fit_transform(e)
    cols = _colors(classes)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for c in classes:
        m = np.array([str(v) == c for v in lab.tolist()])
        if m.any():
            ax.scatter(z[m, 0], z[m, 1], s=22, alpha=0.8, color=cols.get(c, "#888888"),
                       label=f"{c} ({int(m.sum())})")
    ax.set_title(f"{method.upper()} of the shared embedding ({int(fin.sum())} files)", fontsize=9)
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.legend(fontsize=7, framealpha=0.85)
    ax.tick_params(labelsize=7)
    return _save(fig, path, dpi)


def plot_pr_curve(y: np.ndarray, proba: np.ndarray, classes: list[str], title: str, path: Path,
                  dpi: int = 150) -> str | None:
    """One-vs-rest precision-recall curves with the average-precision score in the legend."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] < 2 or y.size < 4:
        return None
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx.get(str(v), -1) for v in y.tolist()], dtype=int)
    keep = yi >= 0
    if keep.sum() < 4:
        return None
    cols = _colors(classes)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for c in classes:
        j = idx[c]
        if (yi[keep] == j).sum() == 0 or (yi[keep] == j).sum() == keep.sum():
            continue    # PR-AUC is undefined for an absent or universal class: say nothing drawn
        ap = average_precision_score((yi[keep] == j).astype(int), proba[keep, j])
        prec, rec, _ = precision_recall_curve((yi[keep] == j).astype(int), proba[keep, j])
        ax.plot(rec, prec, color=cols.get(c, "#888888"), lw=1.1, label=f"{c} (AP {ap:.3f})")
    ax.plot([0, 1], [1, 0], ls="--", lw=0.8, color="#999999")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"one-vs-rest precision-recall: {title}", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.85)
    ax.tick_params(labelsize=7)
    return _save(fig, path, dpi)
