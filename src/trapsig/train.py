"""Model A training: label-blind features, repeated grouped CV, metrics, label-shuffle null.

Inputs come from a completed ``trapsig prepare`` run: its ``fold_assignments.csv`` (one
test-fold column per repeat, keyed by sample_id) and ``master_manifest.csv`` (source paths).
Labels are used only for fitting and scoring, never for feature extraction.

Evaluation rules (design sections 6 and 11):
  * Splits are file-level (each file is its own group). No replicates exist in this data.
  * Hyperparameters (logreg C) are tuned only inside each outer training fold, by inner
    grouped CV. The outer test fold is never used for tuning.
  * Imputation and scaling are fitted inside each training fold only.
  * Repeats are reported as mean and SD across repeats. The aggregated (file-level)
    prediction is the mean probability over repeats; each file is out-of-fold in every repeat.
  * Bootstrap is a file-level bootstrap of the aggregated predictions, because there are
    no replicate or session groups for a hierarchical bootstrap. It is labelled as such.
  * Permutation null: class labels are shuffled across files and the full CV is rerun.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from . import evalmetrics as extmetrics
from .config import TrapsigConfig
from .features import extract_features
from .manifest import read_manifest_csv
from .metrics import MetricResult, classification_metrics
from .pipeline import RecordingSkipped, process_recording
from .resample import decimate_staged
from .runs import write_json_exclusive, write_text_exclusive
from .windows import WindowSet, build_window_set, cut_windows, resolve_window_length_ms, window_samples

BASELINE_MODELS = ("dummy_majority", "dummy_stratified")
# Design section 11.4: no imbalance correction below 1.5; class weights for linear and tree models at or above it.
IMBALANCE_WEIGHT_THRESHOLD = 1.5
META_FIELDS = ["sample_id", "class", "group_id", "filename", "status", "reason"]


class TrainError(RuntimeError):
    """A condition that stops training (missing inputs, inconsistent prepared run, too few files)."""


# ---------------------------------------------------------------------------
# Prepared-run inputs
# ---------------------------------------------------------------------------


def load_prepared(
    prepared_dir: Path, cfg: TrapsigConfig
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    """Read fold assignments and manifest rows from a prepare run."""
    mdir = prepared_dir / "data" / "manifests"
    fold_path = mdir / "fold_assignments.csv"
    manifest_path = mdir / "master_manifest.csv"
    if not fold_path.is_file():
        raise TrainError(f"prepared run is missing {fold_path}; run `trapsig prepare` first")
    with fold_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fold_cols = [c for c in (reader.fieldnames or []) if c.startswith("fold_r")]
        rows = list(reader)
    n_file_repeats = len(fold_cols)
    if n_file_repeats != cfg.splitting.n_repeats:
        raise TrainError(
            f"prepared run has {n_file_repeats} fold repeats but the configuration says "
            f"splitting.n_repeats={cfg.splitting.n_repeats}; re-run `trapsig prepare` with this configuration"
        )
    if not rows:
        raise TrainError("fold_assignments.csv has no rows")
    if not manifest_path.is_file():
        raise TrainError(f"prepared run is missing {manifest_path}; run `trapsig prepare` first")
    manifest = {r["sample_id"]: r for r in read_manifest_csv(manifest_path)}
    missing = [r["sample_id"] for r in rows if r["sample_id"] not in manifest]
    if missing:
        raise TrainError(f"{len(missing)} fold rows are absent from master_manifest.csv (e.g. {missing[0]})")
    return rows, manifest


def build_feature_table(
    cfg: TrapsigConfig, fold_rows: list[dict[str, str]], manifest: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    """One row per pool file. Skipped and failed files are kept with a status and reason."""
    t = cfg.training
    out: list[dict[str, Any]] = []
    for row in fold_rows:
        sid = row["sample_id"]
        path = manifest[sid]["original_path"]
        base = {
            "sample_id": sid,
            "class": row["class"],
            "group_id": row["group_id"],
            "filename": Path(path).name,
        }
        try:
            res = process_recording(path, cfg, t.analysis_rate_hz, keep_signal=True)
        except RecordingSkipped as exc:
            out.append({**base, "status": "skipped", "reason": exc.reason})
            continue
        except Exception as exc:  # a broken file must not stop the run; it is reported
            out.append({**base, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if res.signal is None or res.baseline is None:  # keep_signal=True guarantees these
            raise TrainError("internal error: processed signal not returned")
        feats = extract_features(
            res.signal,
            res.baseline,
            res.fs_hz,
            res.events,
            raw_std=float(res.stats.get("raw_std", float("nan"))),
            raw_level=float(res.stats.get("raw_mean", float("nan"))),
            bands_hz=t.psd_bands_hz,
            acf_lags_ms=t.acf_lags_ms,
            groups=t.feature_groups,
        )
        out.append({**base, "status": "ok", "reason": "", **feats})
    return out


@dataclass
class PoolData:
    """Everything one pass over the pool produces: features, and optionally CNN windows."""

    table: list[dict[str, Any]]
    windows: WindowSet | None = None
    event_table: list[dict[str, Any]] = field(default_factory=list)
    duration_stats: dict[str, float] = field(default_factory=dict)
    window_ms: float = float("nan")
    window_ms_source: str = ""
    window_length_samples: int = 0
    window_rate_hz: float = float("nan")
    cache_dir: str = ""
    cache_bytes: int = 0


def collect_pool(
    cfg: TrapsigConfig,
    fold_rows: list[dict[str, str]],
    manifest: dict[str, dict[str, str]],
    run_dir: Path,
    want_windows: bool,
    say: Any = None,
) -> PoolData:
    """One pass over the pool producing Model A features and, if asked, Model B windows.

    Doing both in a single pass matters: loading, notch filtering, decimating from 1 MHz, and
    event detection dominate the cost (about 3 s per 20 s file), and feature extraction on top
    of that is about 0.05 s. Running the two models as separate passes would double that cost.

    The window length for ``deep.window_ms: auto`` depends on the event-duration distribution of
    the whole pool, which is only known after every file has been read. The analysed signal is
    therefore cached to disk during the pass and the windows are cut afterwards, one file at a
    time, so peak memory stays at roughly one recording. The cache size is reported.
    """
    tell = say or (lambda msg: None)
    t, d = cfg.training, cfg.deep
    table: list[dict[str, Any]] = []
    event_table: list[dict[str, Any]] = []
    durations: list[float] = []
    cache = (run_dir / "cache" / "analysis") if want_windows else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, Any]] = []
    cache_bytes = 0

    for i, row in enumerate(fold_rows):
        sid = row["sample_id"]
        path = manifest[sid]["original_path"]
        base = {
            "sample_id": sid,
            "class": row["class"],
            "group_id": row["group_id"],
            "filename": Path(path).name,
        }
        try:
            res = process_recording(path, cfg, t.analysis_rate_hz, keep_signal=True)
        except RecordingSkipped as exc:
            table.append({**base, "status": "skipped", "reason": exc.reason})
            continue
        except Exception as exc:  # a broken file must not stop the run; it is reported
            table.append({**base, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if res.signal is None or res.baseline is None:  # keep_signal=True guarantees these
            raise TrainError("internal error: processed signal not returned")
        feats = extract_features(
            res.signal,
            res.baseline,
            res.fs_hz,
            res.events,
            raw_std=float(res.stats.get("raw_std", float("nan"))),
            raw_level=float(res.stats.get("raw_mean", float("nan"))),
            bands_hz=t.psd_bands_hz,
            acf_lags_ms=t.acf_lags_ms,
            groups=t.feature_groups,
        )
        table.append({**base, "status": "ok", "reason": "", **feats})
        for ev in res.events:
            event_table.append(
                {
                    "sample_id": sid,
                    "class": row["class"],
                    "group_id": row["group_id"],
                    "filename": Path(path).name,
                    "onset_s": ev.onset_s,
                    "offset_s": ev.offset_s,
                    "duration_s": ev.duration_s,
                    "i0": ev.i0,
                    "i0_source": ev.i0_source,
                    "delta_i_mean": ev.delta_i_mean,
                    "delta_i_over_i0": ev.delta_i_over_i0,
                    "amplitude": ev.amplitude,
                    "polarity": "down" if ev.delta_i_mean < 0 else "up",
                    "censored_start": ev.censored_start,
                    "censored_end": ev.censored_end,
                }
            )
        if cache is None:
            continue
        durations.extend(ev.offset_s - ev.onset_s for ev in res.events)
        resid = (res.signal - res.baseline).astype(np.float32, copy=False)
        cpath = cache / f"{sid}.npy"
        np.save(cpath, resid)
        cache_bytes += cpath.stat().st_size
        pending.append(
            {
                "sample_id": sid,
                "class": row["class"],
                "group_id": row["group_id"],
                "path": cpath,
                "fs_hz": res.fs_hz,
                "onsets_s": np.array([ev.onset_s for ev in res.events], dtype=np.float64),
            }
        )
        if (i + 1) % 25 == 0:
            tell(f"collect: {i + 1}/{len(fold_rows)} files")

    if cache is None:
        return PoolData(table=table, event_table=event_table)

    wms, wsrc, dstats = resolve_window_length_ms(
        durations, d.window_ms, d.window_ms_auto_multiplier, d.window_ms_min, d.window_ms_max
    )
    rate = d.window_rate_hz if d.window_rate_hz else t.analysis_rate_hz
    length = window_samples(wms, rate)
    tell(
        f"windows: {wms:g} ms ({wsrc}) = {length} samples at {rate:g} Hz; {len(durations)} events in the pool"
    )

    per_file: list[dict[str, Any]] = []
    n_edge = 0
    for rec in pending:
        x = np.load(rec["path"])
        fs = float(rec["fs_hz"])
        onsets = rec["onsets_s"]
        if abs(fs - rate) > 1e-9 * max(fs, rate):
            x, fs = decimate_staged(x, fs, rate)
        # Rebuild lightweight event records at the window rate from onset times alone.
        evs = [_OnsetOnly(int(round(o * fs)), o) for o in onsets]
        # The cached array is already baseline-subtracted, so the baseline argument is zeros.
        w, eidx, ons, ne = cut_windows(
            x, np.zeros_like(x), fs, evs, length, d.pre_event_fraction, d.max_windows_per_file
        )
        n_edge += ne
        per_file.append(
            {
                "sample_id": rec["sample_id"],
                "class": rec["class"],
                "group_id": rec["group_id"],
                "windows": w,
                "event_indices": eidx,
                "onsets_s": ons,
                "n_edge_excluded": ne,
            }
        )
    ws = build_window_set(per_file, length, rate, wms, wsrc, dstats)
    tell(
        f"windows built: {ws.x.shape[0]} windows from {len(per_file)} files "
        f"({n_edge} events excluded by the edge rule)"
    )
    return PoolData(
        table=table,
        windows=ws,
        duration_stats=dstats,
        window_ms=wms,
        window_ms_source=wsrc,
        window_length_samples=length,
        window_rate_hz=rate,
        cache_dir=str(cache),
        cache_bytes=cache_bytes,
        event_table=event_table,
    )


@dataclass
class _OnsetOnly:
    """Minimal stand-in for an ``Event`` when cutting windows from cached signals."""

    onset_idx: int
    onset_s: float
    offset_idx: int = 0
    offset_s: float = 0.0


def write_features_csv(table: list[dict[str, Any]], feature_cols: list[str], path: Path) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=META_FIELDS + feature_cols, extrasaction="ignore")
    w.writeheader()
    for row in table:
        out = {k: row.get(k, "") for k in META_FIELDS}
        for c in feature_cols:
            v = row.get(c, "")
            out[c] = "" if (isinstance(v, float) and not math.isfinite(v)) or v is None else v
        w.writerow(out)
    write_text_exclusive(path, buf.getvalue())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _full_proba(model: Any, x: np.ndarray, classes: list[str]) -> np.ndarray:
    """Probability matrix with one column per global class (zeros for classes unseen in training)."""
    p = model.predict_proba(x)
    out = np.zeros((x.shape[0], len(classes)))
    idx = {c: j for j, c in enumerate(classes)}
    for j, c in enumerate(model.classes_):
        out[:, idx[str(c)]] = p[:, j]
    return out


def _base_model(
    name: str,
    cfg: TrapsigConfig,
    seed: int,
    c_value: float | None = None,
    class_weight: str | None = None,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """The pipeline that is cross-validated, saved, and used for prediction.

    ``params`` carries the hyperparameters a nested search chose for this fold (keys are
    ``clf__<argument>``, as in ``optimize.search_space``). Without a search the declared values
    are used: logreg C from ``_tune_logreg_c``, rf/hgb from the configuration.
    """
    t = cfg.training
    pr = dict(params or {})
    impute = ("impute", SimpleImputer(strategy="median"))
    if name == "logreg":
        c = float(pr.get("clf__C", c_value or 1.0))
        return Pipeline(
            [
                impute,
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=c, max_iter=5000, class_weight=class_weight)),
            ]
        )
    if name == "rf":
        return Pipeline(
            [
                impute,
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=int(pr.get("clf__n_estimators", t.rf_n_estimators)),
                        max_depth=pr.get("clf__max_depth"),
                        min_samples_leaf=int(pr.get("clf__min_samples_leaf", 1)),
                        max_features=pr.get("clf__max_features", "sqrt"),
                        random_state=seed,
                        n_jobs=1,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
    if name == "hgb":
        return Pipeline(
            [
                impute,
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=int(pr.get("clf__max_iter", t.hgb_max_iter)),
                        learning_rate=float(pr.get("clf__learning_rate", 0.1)),
                        max_leaf_nodes=int(pr.get("clf__max_leaf_nodes", 31)),
                        min_samples_leaf=int(pr.get("clf__min_samples_leaf", 20)),
                        l2_regularization=float(pr.get("clf__l2_regularization", 0.0)),
                        early_stopping=False,
                        random_state=seed,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
    raise TrainError(f"unknown model {name!r}")


def _tune_logreg_c(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    cfg: TrapsigConfig,
    seed: int,
    class_weight: str | None,
) -> float:
    """Choose C by inner grouped CV, using mean held-out log loss. Training fold only."""
    inner = StratifiedGroupKFold(n_splits=cfg.training.inner_splits, shuffle=True, random_state=seed)
    grid = list(cfg.training.lr_c_grid)
    best_c, best_loss = grid[0], math.inf
    import warnings

    with warnings.catch_warnings():
        # A small outer training fold can leave a class with fewer members than inner_splits.
        # That is expected on small data; the inner split still trains, and the fold is reported.
        warnings.simplefilter("ignore", UserWarning)
        splits = list(inner.split(x, y, groups))
    # An inner fold whose training part holds a single class cannot fit a discriminator. This
    # happens on small data and routinely under a label-shuffle null; such a fold is skipped and
    # the skip is counted, rather than letting the whole cross-validation crash.
    usable = [(itr, iva) for itr, iva in splits if len(set(y[itr].tolist())) >= 2]
    if not usable:
        # Nothing to tune on: fall back to the pre-declared C=1.0 (or the closest grid value).
        return 1.0 if 1.0 in grid else grid[0]
    for c in grid:
        losses = []
        for itr, iva in usable:
            m = _base_model("logreg", cfg, seed, c, class_weight).fit(x[itr], y[itr])
            p = _full_proba(m, x[iva], classes)
            losses.append(log_loss(y[iva], np.clip(p, 1e-12, 1.0), labels=classes))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:  # ties keep the smaller C in the grid order
            best_c, best_loss = c, mean_loss
    return best_c


def _file_pred(proba: np.ndarray, mask: np.ndarray, classes: list[str]) -> np.ndarray:
    """Per-repeat labels from file-level probabilities; '' outside the evaluation mask."""
    pred = np.full(proba.shape[:2], "", dtype=object)
    arr = np.array(classes, dtype=object)
    for r in range(proba.shape[0]):
        fin = mask & np.isfinite(proba[r]).all(axis=1)
        pred[r, fin] = arr[np.argmax(proba[r][fin], axis=1)]
    return pred


def _write_kv_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> Path:
    import csv
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return write_text_exclusive(path, buf.getvalue())


def _write_fewshot_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    import csv
    import io

    fields = [
        "fold_tag",
        "shot",
        "episode",
        "n_way",
        "n_query",
        "accuracy",
        "n_correct",
        "mean_margin",
        "support_classes",
        "pool",
        "protocol",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return write_text_exclusive(path, buf.getvalue())


def fewshot_summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .fewshot import summarize_episodes

    return summarize_episodes(rows)


def _index_saved_models(save_dir: Path | None) -> dict[str, Any] | None:
    """Write and return ``models/INDEX.json``: one entry per saved fold pipeline."""
    if save_dir is None:
        return None
    entries: list[dict[str, Any]] = []
    for side in sorted(save_dir.glob("*.joblib.json")):
        meta = json.loads(side.read_text(encoding="utf-8"))
        entries.append(
            {
                "file": side.name.replace(".json", ""),
                "model": meta.get("model"),
                "fold_tag": meta.get("fold_tag"),
                "repeat": meta.get("repeat"),
                "fold": meta.get("fold"),
                "n_feature_columns": meta.get("n_feature_columns"),
                "n_train_files": meta.get("n_train_files"),
                "trapsig_version": meta.get("trapsig_version"),
                "config_fingerprint": meta.get("config_fingerprint"),
            }
        )
    (save_dir / "INDEX.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return {
        "dir": "models",
        "n": len(entries),
        "models": sorted({str(e["model"]) for e in entries}),
        "fold_tags": sorted({str(e["fold_tag"]) for e in entries}),
        "note": "one fitted pipeline per (repeat, outer fold); `trapsig predict` averages "
        "their probabilities over new files",
    }


def _save_fold_model(
    model: Any,
    name: str,
    cfg: TrapsigConfig,
    classes: list[str],
    feature_columns: list[str] | None,
    hyper: dict[str, Any],
    n_train_files: int,
    seed: int,
    class_weight: str | None,
    save_dir: Path,
    fold_tag: str,
    repeat: int,
    fold: int,
) -> None:
    """Persist one fitted fold pipeline with the sidecar that makes it reusable."""
    from . import __version__
    from .predict import save_model

    meta = {
        "model": name,
        "fold_tag": fold_tag,
        "repeat": repeat,
        "fold": fold,
        "classes": list(classes),
        "feature_columns": list(feature_columns or []),
        "n_feature_columns": len(feature_columns or []),
        "hyperparameters": {k: v for k, v in sorted(hyper.items())},
        "n_train_files": int(n_train_files),
        "seed": int(seed),
        "class_weight": class_weight,
        "trapsig_version": __version__,
        "config_fingerprint": cfg.fingerprint(),
    }
    save_model(model, meta, save_dir / f"{name}_{fold_tag}.joblib")


def fit_predict(
    name: str,
    xtr: np.ndarray,
    ytr: np.ndarray,
    gtr: np.ndarray,
    xte: np.ndarray,
    classes: list[str],
    cfg: TrapsigConfig,
    seed: int,
    class_weight: str | None = None,
    params: dict[str, Any] | None = None,
    save_dir: Path | None = None,
    feature_columns: list[str] | None = None,
    fold_tag: str = "",
    repeat: int = 0,
    fold: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train on one outer training fold and predict the test fold.

    Returns (predicted labels, probabilities with columns in ``classes`` order, chosen C or NaN).
    ``params`` are the hyperparameters a nested search chose for this fold; when it is absent the
    logreg C grid search runs as before and rf/hgb keep their declared settings.
    """
    k = len(classes)
    counts = np.array([np.count_nonzero(ytr == c) for c in classes], dtype=float)
    prior = counts / counts.sum()
    if int(np.count_nonzero(counts > 0)) < 2 and name not in BASELINE_MODELS:
        # Degenerate training fold: one class only (possible on small data, and routinely under a
        # label-shuffle null). No discriminator can be fitted, so the fold predicts the training
        # prior and cv_predictions counts it; the count is reported next to the model.
        proba = np.tile(prior, (xte.shape[0], 1))
        pred = np.array([classes[int(np.argmax(prior))]] * xte.shape[0], dtype=object)
        return pred, proba, float("nan")
    if name == "dummy_majority":
        proba = np.tile(prior, (xte.shape[0], 1))
        pred = np.array([classes[int(np.argmax(prior))]] * xte.shape[0], dtype=object)
        return pred, proba, float("nan")
    if name == "dummy_stratified":
        rng = np.random.default_rng(seed)
        proba = np.tile(prior, (xte.shape[0], 1))
        pred = rng.choice(np.array(classes, dtype=object), size=xte.shape[0], p=prior)
        return pred, proba, float("nan")
    chosen = float("nan")
    if params:
        # A nested search already chose this fold's hyperparameters; do not tune a second time.
        chosen = float(params["clf__C"]) if "clf__C" in params else float("nan")
        model = _base_model(name, cfg, seed, None, class_weight, params).fit(xtr, ytr)
    elif name == "logreg":
        chosen = _tune_logreg_c(xtr, ytr, gtr, classes, cfg, seed, class_weight)
        model = _base_model("logreg", cfg, seed, chosen, class_weight).fit(xtr, ytr)
    else:
        model = _base_model(name, cfg, seed, None, class_weight).fit(xtr, ytr)
    proba = _full_proba(model, xte, classes)
    if proba.shape[1] != k:  # pragma: no cover - defensive
        raise TrainError("probability columns do not match classes")
    pred = np.array(classes, dtype=object)[np.argmax(proba, axis=1)]
    if save_dir is not None and feature_columns:
        hyper = dict(params or {})
        if math.isfinite(chosen):
            hyper.setdefault("clf__C", chosen)
        _save_fold_model(
            model,
            name,
            cfg,
            classes,
            feature_columns,
            hyper,
            xtr.shape[0],
            seed,
            class_weight,
            save_dir,
            fold_tag or "fold",
            repeat,
            fold,
        )
    return pred, proba, chosen


def cv_predictions(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fold_ids: np.ndarray,
    classes: list[str],
    cfg: TrapsigConfig,
    class_weight: str | None = None,
    diag: dict[str, Any] | None = None,
    tune: bool = True,
    hpo_rows: list[dict[str, Any]] | None = None,
    say: Any = None,
    save_dir: Path | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold predictions for every repeat. ``fold_ids`` has shape (n_repeats, n_files).

    ``diag``, when given, receives ``degenerate_training_folds``: the number of (repeat, fold)
    pairs whose training part held a single class and therefore predicted the training prior, and
    — when a search ran — ``hpo`` with the per-fold chosen settings and the search cost.

    ``tune=False`` runs the fold with the declared hyperparameters and no search. The ablation uses
    it (so a feature-subset difference is not confounded by tuning noise) and says so; the headline
    models and the permutation nulls use the same setting as each other, because a null fitted with
    weaker hyperparameters than the real model would bias the p-value.

    ``hpo_rows``, when given, receives one dict per trial for ``hpo_trials.csv``.
    """
    n_rep, n = fold_ids.shape
    k = len(classes)
    t = cfg.training
    pred = np.empty((n_rep, n), dtype=object)
    proba = np.zeros((n_rep, n, k))
    chosen = np.full((n_rep, n), np.nan)
    base_seed = cfg.run.seed
    for r in range(n_rep):
        for f in range(cfg.splitting.n_splits):
            te = fold_ids[r] == f
            tr = ~te
            if not te.any() or not tr.any():
                raise TrainError(f"repeat {r} fold {f} is empty after skipping unreadable files")
            seed = base_seed + 1000 * r + f
            if diag is not None and len(set(y[tr].tolist())) < 2:
                diag["degenerate_training_folds"] = int(diag.get("degenerate_training_folds", 0)) + 1
            params: dict[str, Any] | None = None
            if tune and t.hpo != "none" and name not in BASELINE_MODELS:
                from . import optimize

                tag = f"r{r}f{f}"
                res = optimize.tune_model(
                    name, x[tr], y[tr], groups[tr], classes, cfg, seed, class_weight, tag, say
                )
                if diag is not None:
                    h = diag.setdefault(
                        "hpo",
                        {
                            "method": res.method,
                            "metric": res.metric,
                            "space": res.space,
                            "folds": [],
                            "seconds": 0.0,
                            "n_trials": 0,
                            "failures": [],
                        },
                    )
                    h["folds"].append(
                        {
                            "fold_tag": tag,
                            "status": res.status,
                            "best_params": res.best_params,
                            "best_score": res.best_score,
                            "n_trials": res.n_trials,
                            "reason": res.reason,
                        }
                    )
                    h["seconds"] = round(float(h["seconds"]) + res.seconds, 3)
                    h["n_trials"] += res.n_trials
                    if res.status != "computed" and res.reason:
                        h["failures"].append(res.reason)
                if hpo_rows is not None:
                    for tl in res.history:
                        hpo_rows.append(
                            {
                                "model": name,
                                "repeat": r,
                                "fold": f,
                                "fold_tag": tag,
                                "method": res.method,
                                "metric": res.metric,
                                "trial": tl.trial,
                                "rung": tl.rung,
                                "n_train_files": tl.n_train_files,
                                "score": tl.score,
                                "n_inner_splits": tl.n_inner_splits,
                                "status": tl.status,
                                "reason": tl.reason,
                                "params": json.dumps(tl.params, sort_keys=True, default=str),
                            }
                        )
                if res.status == "computed" and res.best_params:
                    params = res.best_params
            p_lab, p_prob, c = fit_predict(
                name,
                x[tr],
                y[tr],
                groups[tr],
                x[te],
                classes,
                cfg,
                seed,
                class_weight,
                params,
                save_dir,
                feature_columns,
                f"r{r}f{f}",
                r,
                f,
            )
            pred[r, te] = p_lab
            proba[r, te] = p_prob
            chosen[r, te] = c
    return pred, proba, chosen


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> float:
    """Mean per-class recall over classes present in ``y``. Used for bootstrap and permutation."""
    recalls = [float(np.mean(pred[y == c] == c)) for c in classes if np.any(y == c)]
    return float(np.mean(recalls)) if recalls else float("nan")


def _metric_value(res: MetricResult) -> float | None:
    return res.value if res.status == "computed" else None


def bootstrap_ci(
    y: np.ndarray, pred: np.ndarray, classes: list[str], n_resamples: int, seed: int
) -> tuple[float, float] | None:
    """95% percentile interval from a file-level bootstrap of balanced accuracy."""
    if n_resamples <= 0:
        return None
    rng = np.random.default_rng(seed)
    n = y.size
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals.append(balanced_accuracy(y[idx], pred[idx], classes))
    arr = np.array([v for v in vals if math.isfinite(v)])
    if arr.size == 0:
        return None
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

TORCH_INSTALL_HINT = (
    "pip install torch --index-url https://download.pytorch.org/whl/cpu   "
    "(CPU-only build; use a plain `pip install torch` on a GPU machine)"
)


def deep_dependency_status() -> dict[str, Any]:
    """Report whether Model B can run here, and if not, exactly what to install.

    PyTorch is a large optional dependency. If it is missing the pipeline must still produce
    Model A results and say why Model B is absent, rather than failing at import time.
    """
    try:
        import torch
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "install_hint": TORCH_INSTALL_HINT,
        }
    return {
        "available": True,
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@dataclass
class ModelEntry:
    """One scored model, on the common footing needed to compare Model A, Model B, and the blend.

    ``mask`` says which files this model can be evaluated on. Model B cannot score a file that
    produced no usable event window, so its mask is narrower than Model A's; every metric,
    interval, and confusion matrix for that model is computed on the mask, and the count is
    reported next to it. Nothing is averaged over files the model never saw.
    """

    proba: np.ndarray  # (n_repeats, n_files, k); NaN outside the mask
    pred: np.ndarray  # (n_repeats, n_files) object; "" outside the mask
    mask: np.ndarray  # (n_files,) bool
    kind: str = "feature"  # feature | deep | prototype | fusion | ensemble | baseline
    chosen: np.ndarray | None = None  # per-fold tuned hyperparameter (logreg C)
    source: np.ndarray | None = None  # per-file hybrid provenance ("hybrid"/"model_a_only")
    extra: dict[str, Any] = field(default_factory=dict)


def run_training(
    cfg: TrapsigConfig, prepared_dir: Path, run_dir: Path, progress: Any = None, stages: bool = False
) -> dict[str, Any]:
    """Run the configured model set end to end and write all outputs into ``run_dir``.

    ``stages=True`` also runs explainability, figures, conditional analyses, and reporting inside
    the same process, because they need the window set and feature table this function already
    built. Rebuilding them would mean reading every recording a second time.
    """
    say = progress or (lambda msg: None)
    t, d = cfg.training, cfg.deep
    model_set = d.model_set if d.enabled else "features"
    dep = deep_dependency_status()
    deep_unavailable: dict[str, Any] | None = None
    if model_set in ("deep", "hybrid") and not dep["available"]:
        deep_unavailable = {
            "requested_model_set": model_set,
            "reason": f"Model B needs PyTorch, which is not importable here: {dep.get('reason')}",
            "install_hint": dep.get("install_hint", ""),
            "action_taken": "ran Model A only; no Model B or hybrid number is reported",
        }
        say(f"deep: PyTorch unavailable ({dep.get('reason')}); continuing with Model A only")
        model_set = "features"
    want_windows = model_set in ("deep", "hybrid") or cfg.quad.enabled
    want_features = model_set in ("features", "hybrid")
    from . import hybrid as hyb  # imported lazily: it pulls in PyTorch

    fold_rows, manifest = load_prepared(prepared_dir, cfg)
    say(f"collect: {len(fold_rows)} pool files at {t.analysis_rate_hz:g} Hz (model set: {model_set})")
    pool = collect_pool(cfg, fold_rows, manifest, run_dir, want_windows, say)
    table = pool.table
    ok = [row for row in table if row["status"] == "ok"]
    skipped = [row for row in table if row["status"] != "ok"]
    say(f"collect done: {len(ok)} ok, {len(skipped)} skipped or failed")

    if not ok:
        raise TrainError("no recording produced features; see features.csv for reasons")
    feat_cols: list[str] = []
    for row in ok:
        for kcol in row:
            if kcol not in META_FIELDS and kcol not in feat_cols:
                feat_cols.append(kcol)
    all_nan = [c for c in feat_cols if all(not math.isfinite(float(row.get(c, float("nan")))) for row in ok)]
    feat_cols = [c for c in feat_cols if c not in all_nan]
    write_features_csv(table, feat_cols, run_dir / "features.csv")

    save_dir: Path | None = None
    if cfg.training.save_models:
        save_dir = run_dir / "models"
        save_dir.mkdir(parents=True, exist_ok=True)

    sids = [row["sample_id"] for row in ok]
    fold_map = {row["sample_id"]: row for row in fold_rows}
    fold_ids = np.array(
        [[int(fold_map[s][f"fold_r{rep}"]) for s in sids] for rep in range(cfg.splitting.n_repeats)]
    )
    x = np.array([[float(row[c]) for c in feat_cols] for row in ok], dtype=float)
    y = np.array([row["class"] for row in ok], dtype=object)
    groups = np.array([row["group_id"] for row in ok], dtype=object)
    classes = sorted(set(y.tolist()))
    class_counts = {c: int(np.sum(y == c)) for c in classes}
    if len(classes) < 2:
        raise TrainError(f"only one class has usable files ({class_counts}); nothing to classify")
    if min(class_counts.values()) < cfg.splitting.n_splits:
        raise TrainError(
            f"a class has fewer usable files ({class_counts}) than n_splits={cfg.splitting.n_splits}"
        )
    imbalance = max(class_counts.values()) / min(class_counts.values())
    class_weight = "balanced" if imbalance >= IMBALANCE_WEIGHT_THRESHOLD else None
    all_mask = np.ones(len(ok), dtype=bool)

    entries: dict[str, ModelEntry] = {}
    proba_a_mean: np.ndarray | None = None
    hpo_rows: list[dict[str, Any]] = []

    if want_features:
        for name in list(BASELINE_MODELS) + list(t.models):
            say(
                f"CV: {name}"
                + (
                    f" (hyperparameter search: {t.hpo}, objective {t.hpo_metric})"
                    if t.hpo != "none" and name not in BASELINE_MODELS
                    else ""
                )
            )
            diag: dict[str, Any] = {}
            pred, proba, chosen = cv_predictions(
                name,
                x,
                y,
                groups,
                fold_ids,
                classes,
                cfg,
                class_weight,
                diag,
                hpo_rows=hpo_rows,
                say=say,
                save_dir=save_dir if name not in BASELINE_MODELS else None,
                feature_columns=list(feat_cols),
            )
            entries[name] = ModelEntry(
                proba=proba,
                pred=pred,
                mask=all_mask,
                chosen=chosen,
                kind="baseline" if name in BASELINE_MODELS else "feature",
                extra=diag,
            )
        proba_a_mean = np.mean([entries[n].proba for n in t.models], axis=0)

    deep_branch = None
    fusion_note = "deep branch disabled"
    if want_windows:
        if pool.windows is None:
            say("deep: no window set was produced; Model B skipped")
        else:
            deep_branch = hyb.run_deep_branch(
                pool.windows, fold_ids, sids, classes, cfg, imbalance, run_dir, say
            )
            if deep_branch.has_windows.any():
                entries["cnn1d"] = ModelEntry(
                    proba=deep_branch.proba,
                    pred=deep_branch.pred,
                    mask=deep_branch.has_windows,
                    kind="deep",
                    extra={
                        "n_windows": int(pool.windows.x.shape[0]),
                        "windows_per_file_mean": float(np.mean(deep_branch.counts[deep_branch.has_windows])),
                        "temperature_mean": float(np.mean(deep_branch.cv.temperature))
                        if deep_branch.cv
                        else None,
                    },
                )
            else:
                say("deep: no file produced a usable window; Model B is not evaluable")

            # ---- the same frozen embedding, read three more ways ----
            # One encoder per outer fold, four readings of it. This is what makes the learning
            # strategies one architecture: the prototype head, the fusion head, few-shot, novelty
            # detection, clustering, and the conditional analyses all consume the embedding this
            # fold produced instead of each learning their own.
            if (
                deep_branch.has_proto is not None
                and bool(deep_branch.has_proto.any())
                and deep_branch.proba_proto is not None
                and deep_branch.pred_proto is not None
            ):
                entries["proto_cnn"] = ModelEntry(
                    proba=deep_branch.proba_proto,
                    pred=deep_branch.pred_proto,
                    mask=deep_branch.has_proto,
                    kind="prototype",
                    extra={
                        "head": "nearest class prototype on the L2-normalised GAP embedding",
                        "prototypes_from": "inner-validation files of the same fold",
                        "proto_distance": str(d.proto_distance),
                        "metric_loss": str(d.metric_loss),
                        "metric_loss_weight": float(d.metric_loss_weight),
                        "metric_temperature": float(d.metric_temperature),
                        "n_files": int(deep_branch.has_proto.sum()),
                    },
                )
                say(
                    f"prototype head: {int(deep_branch.has_proto.sum())} of {len(ok)} files "
                    f"({d.proto_distance} distance, {d.metric_loss} auxiliary loss)"
                )
            elif d.prototypical_head:
                say("prototype head: no fold produced prototypes (empty inner-validation split)")

            fusion_note = (
                f"fusion rule '{d.fusion}': the stacked fusion head is disabled by configuration"
                if d.fusion not in ("stack", "both")
                else "not used"
            )
            if d.fusion in ("stack", "both") and pool.windows is not None and deep_branch is not None:
                fs = hyb.fusion_stack(deep_branch, pool.windows, fold_ids, sids, x, y, classes, cfg, say)
                fusion_note = "; ".join(fs.notes) if fs.notes else "fitted per repeat and fold"
                if fs.ok.any():
                    entries["fusion"] = ModelEntry(
                        proba=fs.proba,
                        pred=fs.pred,
                        mask=fs.ok,
                        kind="fusion",
                        extra={
                            "head": "logistic regression on handcrafted features + mean embedding",
                            "n_handcrafted": len(feat_cols),
                            "n_embedding": int(d.channels),
                            "n_inputs": len(feat_cols) + int(d.channels),
                            "fitted_on": "inner-validation files of the same fold only",
                            "n_val_file_fits": int(fs.n_val_files),
                            "n_files": int(fs.ok.sum()),
                        },
                    )
                    say(
                        f"fusion head: {int(fs.ok.sum())} of {len(ok)} files "
                        f"({len(feat_cols)} handcrafted + {int(d.channels)} learned inputs)"
                    )
                else:
                    say(f"fusion head: no fold could be fitted ({fusion_note})")
            if d.fusion in ("blend", "both") and model_set != "hybrid":
                extra_note = (
                    f"fusion rule '{d.fusion}' includes the probability blend, which "
                    f"needs Model A probabilities; model_set is '{model_set}', so only "
                    "the stacked head was fitted"
                )
                fusion_note = f"{fusion_note}; {extra_note}"
                say(f"fusion: {extra_note}")

        if (
            model_set == "hybrid"
            and d.fusion in ("blend", "both")
            and "cnn1d" in entries
            and proba_a_mean is not None
        ):
            sub = entries["cnn1d"].mask
            ph, src = hyb.combine_probabilities(proba_a_mean, entries["cnn1d"].proba, d.model_a_weight, sub)
            entries["hybrid"] = ModelEntry(
                proba=ph,
                pred=hyb.hybrid_predictions(ph, classes),
                mask=all_mask,
                kind="ensemble",
                source=src,
                extra={"fusion_rule": "blend", "model_a_weight": float(d.model_a_weight)},
            )
            say(
                f"hybrid: {int(sub.sum())} of {len(ok)} files blended; "
                f"{int(len(ok) - sub.sum())} fell back to Model A"
            )

    # ---- the four-branch deep architecture: additive, opt-in, same leakage contract ----
    quad_branch: Any = None
    quad_block: dict[str, Any] | None = None
    if cfg.quad.enabled:
        if not pool.cache_dir:
            say("quad: enabled but no analysed-signal cache exists; branch not evaluated")
        else:
            from . import quad as quadmod

            tensors = quadmod.build_quad_tensors(pool.cache_dir, ok, cfg, say)
            if tensors is None:
                say("quad: no recording produced a segmentable window; branch not evaluated")
            else:
                qcv = quadmod.cv_quad(tensors, sids, y, fold_ids, classes, cfg, run_dir, say)
                quad_branch = qcv
                entries["quad"] = ModelEntry(
                    proba=qcv.proba,
                    pred=_file_pred(qcv.proba, qcv.mask, classes),
                    mask=qcv.mask,
                    kind="deep",
                    extra={
                        "architecture": "four-branch (temporal, spectral, self-supervised "
                        "pretrain, engineered) with cross-branch attention "
                        "fusion, softmax head and prototype head",
                        "branches": qcv.branch_names,
                        "gate_mean": qcv.gates_mean,
                        "stages": "2 pretrain -> 3 heads/fusion -> 4 joint -> 5 episodic",
                        "n_windows": int(tensors["x"].shape[0]),
                        "n_engineered_window_features": len(tensors["feat_names"]),
                    },
                )
                say(
                    f"quad: {int(qcv.mask.sum())} of {len(ok)} files scored by the fused "
                    f"four-branch model; gates {qcv.gates_mean}"
                )
                for ab in cfg.quad.branch_ablation:
                    qab = quadmod.cv_quad(tensors, sids, y, fold_ids, classes, cfg, run_dir, say, ablate=[ab])
                    entries[f"quad_minus_{ab}"] = ModelEntry(
                        proba=qab.proba,
                        pred=_file_pred(qab.proba, qab.mask, classes),
                        mask=qab.mask,
                        kind="deep",
                        extra={
                            "branch_ablation": ab,
                            "note": f"the {ab} branch token is removed from the fusion",
                        },
                    )
                    say(f"quad ablation: branch '{ab}' removed -> {int(qab.mask.sum())} files scored")
                if qcv.fewshot_rows:
                    _write_fewshot_csv(qcv.fewshot_rows, run_dir / "fewshot_episodes.csv")
                    say(f"quad: {len(qcv.fewshot_rows)} few-shot episodes recorded in fewshot_episodes.csv")
                from . import featreg as _featreg

                _write_kv_csv(
                    _featreg.registry_rows(
                        [list(b) for b in cfg.training.psd_bands_hz],
                        list(cfg.quad.feature_tiers) or [1, 2],
                        list(cfg.quad.feature_families) or None,
                    ),
                    run_dir / "feature_registry.csv",
                    ["id", "family", "family_description", "tier", "description", "unit", "parameters"],
                )
                rec_rows = tensors.get("rec_rows") or []
                if rec_rows:
                    fields = ["sample_id", "class"] + [
                        k for k in rec_rows[0] if k not in ("sample_id", "class")
                    ]
                    _write_kv_csv(rec_rows, run_dir / "recording_feature_summary.csv", fields)
                    say(
                        f"quad: recording-level feature summary for {len(rec_rows)} files in "
                        "recording_feature_summary.csv"
                    )
                open_set_block: dict[str, Any] = {}
                thr = cfg.quad.open_set_msp_threshold
                if thr is not None and qcv.msp is not None:
                    msp = qcv.msp
                    flagged = int(np.sum(np.nan_to_num(msp, nan=0.0) < thr))
                    _write_kv_csv(
                        [
                            {
                                "sample_id": sid,
                                "max_softmax_prob": (
                                    None if not np.isfinite(msp[i]) else round(float(msp[i]), 6)
                                ),
                                "flagged_unknown": bool(np.isfinite(msp[i]) and msp[i] < thr),
                            }
                            for i, sid in enumerate(sids)
                        ],
                        run_dir / "quad_open_set_flags.csv",
                        ["sample_id", "max_softmax_prob", "flagged_unknown"],
                    )
                    open_set_block = {
                        "threshold": float(thr),
                        "n_flagged": flagged,
                        "fraction_flagged": round(flagged / max(1, len(sids)), 4),
                        "note": (
                            "maximum-softmax-probability rejection is a DIAGNOSTIC only: "
                            "open-set recognition needs held-out unknown-class data and is "
                            "not validated by this run"
                        ),
                    }
                    say(f"quad: open-set diagnostic flagged {flagged} of {len(sids)} files below MSP {thr}")
                else:
                    open_set_block = {
                        "threshold": None,
                        "note": "quad.open_set_msp_threshold is null: no unknown-class diagnostic computed",
                    }
                logs0 = qcv.logs[0] if qcv.logs else {}
                quad_block = {
                    "branches": qcv.branch_names,
                    "gate_mean": qcv.gates_mean,
                    "n_windows": int(tensors["x"].shape[0]),
                    "n_engineered_window_features": len(tensors["feat_names"]),
                    "n_parameters": logs0.get("n_parameters"),
                    "receptive_field_samples": logs0.get("receptive_field_samples"),
                    "pretrain": {
                        "epochs_run": logs0.get("pretrain_epochs_run"),
                        "seconds": logs0.get("pretrain_seconds"),
                        "objective": "NT-Xent over augmentation views + masked-span "
                        "reconstruction, training fold only",
                    },
                    "fewshot_summary": fewshot_summarize(qcv.fewshot_rows),
                    "open_set": open_set_block,
                    "feature_registry": {
                        "tiers": list(cfg.quad.feature_tiers),
                        "families": list(cfg.quad.feature_families) or "all",
                        "n_columns": len(tensors["feat_names"]),
                        "csv": "feature_registry.csv",
                        "selection": str(cfg.quad.feature_select),
                    },
                    "fold_notes": qcv.notes[:8],
                    "seconds": qcv.seconds,
                    "checkpoint_selection": "inner-validation windows of each outer fold only",
                }

    write_train_metadata(cfg, run_dir, prepared_dir, table, sids, deep_branch, say)

    # ---- metrics, aggregation, bootstrap intervals ----
    metric_rows: list[dict[str, Any]] = []
    model_summary: dict[str, Any] = {}
    agg_pred: dict[str, np.ndarray] = {}
    agg_ci: dict[str, tuple[float, float] | None] = {}
    for name, ent in entries.items():
        n_sub = int(ent.mask.sum())
        summary_entry: dict[str, Any] = {
            "kind": ent.kind,
            "n_files_evaluated": n_sub,
            "evaluated_on": (
                "all usable files" if ent.mask.all() else "files with at least one usable event window"
            ),
            "per_repeat": {},
        }
        per_repeat: list[dict[str, MetricResult]] = []
        for rep_i in range(cfg.splitting.n_repeats):
            m = ent.mask
            mets = classification_metrics(
                y[m], ent.pred[rep_i][m], labels=classes, y_proba=_rows(ent.proba[rep_i], m)
            )
            per_repeat.append(mets)
            for met in mets.values():
                metric_rows.append(
                    {
                        "scope": f"repeat_{rep_i}",
                        "model": name,
                        "metric": met.name,
                        "value": met.value if met.value is not None else "",
                        "status": met.status,
                        "reason": met.reason,
                        "n_files": n_sub,
                    }
                )
        for mname in per_repeat[0]:
            vals = [v for v in (_metric_value(p[mname]) for p in per_repeat) if v is not None]
            if vals:
                summary_entry["per_repeat"][mname] = {
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n_repeats": len(vals),
                }
            else:
                summary_entry["per_repeat"][mname] = {
                    "status": per_repeat[0][mname].status,
                    "reason": per_repeat[0][mname].reason,
                }
        if name == "dummy_stratified":
            # A random baseline has no single aggregated prediction; report per repeat only.
            summary_entry["aggregated"] = {
                "status": "not_applicable",
                "reason": "random baseline; reported per repeat only",
            }
            agg_ci[name] = None
        else:
            mean_proba = _nanmean_repeats(ent.proba)
            m = ent.mask
            agg = np.full(len(ok), "", dtype=object)
            agg[m] = np.array(classes, dtype=object)[np.argmax(mean_proba[m], axis=1)]
            agg_pred[name] = agg
            agg_mets = classification_metrics(y[m], agg[m], labels=classes, y_proba=mean_proba[m])
            summary_entry["aggregated"] = {}
            for met in agg_mets.values():
                metric_rows.append(
                    {
                        "scope": "aggregated",
                        "model": name,
                        "metric": met.name,
                        "value": met.value if met.value is not None else "",
                        "status": met.status,
                        "reason": met.reason,
                        "n_files": n_sub,
                    }
                )
                summary_entry["aggregated"][met.name] = {
                    "value": met.value,
                    "status": met.status,
                    "reason": met.reason,
                }
            ci = bootstrap_ci(y[m], agg[m], classes, t.bootstrap_resamples, cfg.run.seed)
            agg_ci[name] = ci
            summary_entry["aggregated"]["balanced_accuracy_bootstrap_ci95"] = (
                {
                    "method": "file-level bootstrap of aggregated predictions (no replicate groups)",
                    "lo": ci[0] if ci else None,
                    "hi": ci[1] if ci else None,
                    "n_resamples": t.bootstrap_resamples,
                }
                if ci
                else None
            )
        if ent.extra:
            summary_entry.update(ent.extra)
        model_summary[name] = summary_entry

    # ---- label-shuffle nulls ----
    rng = np.random.default_rng(cfg.run.seed + 7)
    for name in t.models:
        if name not in entries:
            continue
        model_summary[name]["permutation"] = _permutation_null_feature(
            name, x, y, groups, fold_ids, classes, cfg, class_weight, entries[name], rng, say
        )
    if "cnn1d" in entries:
        model_summary["cnn1d"]["permutation"] = _permutation_null_deep(
            pool.windows, y, sids, fold_ids, classes, cfg, imbalance, entries["cnn1d"], rng, say
        )
    if "hybrid" in entries:
        model_summary["hybrid"]["permutation"] = {
            "status": "not run",
            "reason": "the ensemble has no free parameters, so its null is implied by the nulls of "
            "Model A and Model B; running it would double the cost without adding evidence",
        }

    # ---- fair comparison on the matched subset, and gate G7 ----
    comparison = _comparison_block(entries, y, classes, cfg, agg_ci, run_dir, t, pool, model_set, say)

    # ---- feature ablation and redundancy: evidence about the feature set, on the same folds ----
    from .ablation import (
        feature_group_block,
        redundancy_report,
        run_ablation,
        write_ablation_csv,
        write_redundancy_csv,
    )

    say("stage: feature ablation and redundancy")
    ablation = run_ablation(cfg, x, y, groups, fold_ids, classes, feat_cols, class_weight, all_mask, say)
    redundant = redundancy_report(x, feat_cols, t.redundancy_threshold)
    write_ablation_csv(ablation, run_dir / "ablation.csv")
    write_redundancy_csv(redundant, run_dir / "feature_redundancy.csv", t.redundancy_threshold)
    if ablation.get("status") == "computed":
        say(
            f"ablation: {len(ablation['rows'])} feature variants refitted on the same folds "
            f"with {ablation['model']}"
        )
    else:
        say(f"ablation: {ablation.get('status')} - {ablation.get('reason')}")
    say(
        f"feature redundancy: {len(redundant)} pair(s) at |Spearman| >= "
        f"{t.redundancy_threshold:g} (reported, never dropped)"
    )

    # ---- the hyperparameter search, if one ran: every trial is on the record ----
    hpo_block: dict[str, Any] = {
        "method": t.hpo,
        "metric": t.hpo_metric,
        "higher_is_better": True,
        "objective": (
            "mean inner grouped score inside each outer training fold; the outer test "
            "fold never influences the setting used to predict it"
        ),
        "status": (
            "not run"
            if t.hpo == "none" and not d.hpo_trials
            else (
                "computed"
                if (hpo_rows or (deep_branch is not None and getattr(deep_branch, "hpo_rows", None)))
                else "no trial ran"
            )
        ),
        "n_trials": len(hpo_rows),
        "per_model": {},
    }
    if t.hpo == "none":
        hpo_block["reason"] = (
            "training.hpo is 'none': logreg C comes from its exact grid over "
            "training.lr_c_grid inside each training fold, and rf/hgb keep the "
            "declared n_estimators and max_iter"
            + (
                ""
                if d.hpo_trials
                else "; deep.hpo_trials is 0, so Model B keeps its declared "
                "learning rate, dropout and weight decay (each trial would be a "
                "full training run, so the budget is the user's decision)"
            )
        )
    else:
        from .optimize import write_hpo_csv

        if deep_branch is not None and getattr(deep_branch, "hpo_rows", None):
            hpo_rows.extend(deep_branch.hpo_rows)
            scores: list[float] = [
                float(sm["best_score"])
                for sm in deep_branch.hpo_summaries
                if isinstance(sm.get("best_score"), float)
            ]
            hpo_block["per_model"]["cnn1d"] = {
                "method": deep_branch.hpo_summaries[0].get("method"),
                "metric": deep_branch.hpo_summaries[0].get("metric"),
                "space": deep_branch.hpo_summaries[0].get("space"),
                "protocol": deep_branch.hpo_summaries[0].get("protocol"),
                "not_searched": deep_branch.hpo_summaries[0].get("not_searched"),
                "n_searches": len(deep_branch.hpo_summaries),
                "n_trials": len(deep_branch.hpo_rows),
                "seconds": round(sum(float(sm.get("seconds") or 0.0) for sm in deep_branch.hpo_summaries), 3),
                "best_score_mean": float(np.mean(scores)) if scores else None,
                "best_per_fold": [
                    {
                        "fold": sm.get("fold_tag"),
                        "score": sm.get("best_score"),
                        "params": sm.get("best_params"),
                    }
                    for sm in deep_branch.hpo_summaries
                ],
                "n_folds_failed": sum(int(sm.get("n_folds_failed") or 0) for sm in deep_branch.hpo_summaries),
            }
        if hpo_rows:
            write_hpo_csv(hpo_rows, run_dir / "hpo_trials.csv")
        for name, ent in entries.items():
            h = (ent.extra or {}).get("hpo")
            if not h:
                continue
            scores = [float(f["best_score"]) for f in h["folds"] if isinstance(f.get("best_score"), float)]
            hpo_block["per_model"][name] = {
                "method": h["method"],
                "metric": h["metric"],
                "space": h["space"],
                "n_searches": len(h["folds"]),
                "n_trials": h["n_trials"],
                "seconds": h["seconds"],
                "best_score_mean": float(np.mean(scores)) if scores else None,
                "best_score_per_fold": [
                    {
                        "fold": f["fold_tag"],
                        "score": f["best_score"],
                        "params": f["best_params"],
                        "status": f["status"],
                    }
                    for f in h["folds"]
                ],
                "n_folds_failed": sum(1 for f in h["folds"] if f["status"] != "computed"),
                "failure_reasons": sorted(set(h["failures"]))[:5],
            }
        say(
            f"hpo: {len(hpo_rows)} trial(s) recorded in hpo_trials.csv "
            f"({t.hpo} search, objective {t.hpo_metric})"
        )

    saved_index = _index_saved_models(save_dir)

    _write_metrics(metric_rows, run_dir / "metrics.csv")
    _write_predictions(ok, entries, classes, fold_ids, agg_pred, run_dir / "predictions.csv")
    ext_rows: list[dict[str, Any]] = []
    ext_summary: dict[str, Any] = {}
    for name in entries:
        if name in agg_pred:
            m = entries[name].mask
            _write_confusion(y[m], agg_pred[name][m], classes, run_dir / f"confusion_{name}.csv")
            proba_mean = entries[name].proba[:, m, :].mean(axis=0)
            rows_i, summ_i = extmetrics.extended_metrics(
                name,
                y[m],
                agg_pred[name][m],
                proba_mean,
                classes,
                top_k=int(t.report_top_k),
                bins=int(t.calibration_bins),
                pr_curves=bool(t.pr_curves),
            )
            ext_rows.extend(rows_i)
            ext_summary[name] = summ_i
    if ext_rows:
        extmetrics.write_extended_csv(ext_rows, run_dir / "metrics_extended.csv")
        say(f"extended metrics: {len(ext_rows)} rows in metrics_extended.csv ({len(ext_summary)} models)")
    if pool.windows is not None and pool.windows.x.shape[0]:
        from .hybrid import write_deep_windows_csv

        write_deep_windows_csv(pool.windows, run_dir / "windows.csv")
    if pool.event_table:
        import csv as _csv
        import io as _io

        _fields = [
            "sample_id",
            "class",
            "group_id",
            "filename",
            "onset_s",
            "offset_s",
            "duration_s",
            "i0",
            "i0_source",
            "delta_i_mean",
            "delta_i_over_i0",
            "amplitude",
            "polarity",
            "censored_start",
            "censored_end",
        ]
        _buf = _io.StringIO()
        _w = _csv.DictWriter(_buf, fieldnames=_fields, extrasaction="ignore")
        _w.writeheader()
        for _r in pool.event_table:
            _w.writerow({k: ("" if _r.get(k) is None else _r.get(k)) for k in _fields})
        write_text_exclusive(run_dir / "events_table.csv", _buf.getvalue())
        say(f"events: per-event table for {len(pool.event_table)} events in events_table.csv")
    write_text_exclusive(
        run_dir / "config_used.yaml", yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True)
    )

    skipped_summary = [
        {
            "sample_id": row["sample_id"],
            "class": row["class"],
            "status": row["status"],
            "reason": row["reason"],
        }
        for row in skipped
    ]
    summary = {
        "trapsig_version": __version__,
        "config_fingerprint": cfg.fingerprint(),
        "prepared_run": str(prepared_dir),
        "seed": cfg.run.seed,
        "model_set": model_set,
        "analysis_rate_hz": t.analysis_rate_hz,
        "filter_mode": cfg.filters.mode,
        "n_pool_files": len(fold_rows),
        "n_files_used": len(ok),
        "n_files_skipped_or_failed": len(skipped),
        "skipped_or_failed": skipped_summary,
        "class_counts_used": class_counts,
        "imbalance_ratio": imbalance,
        "class_weight": class_weight,
        "n_features": len(feat_cols),
        "features": feature_group_block(feat_cols, cfg),
        "saved_models": saved_index
        if saved_index is not None
        else {"dir": "models", "n": 0, "reason": "training.save_models is false"},
        "quad": quad_block,
        "evaluation": {"definitions": extmetrics.DEFINITIONS, "per_model": ext_summary},
        "features_dropped_all_nan": all_nan,
        "hpo": hpo_block,
        "ablation": ablation,
        "feature_redundancy": {
            "threshold": t.redundancy_threshold,
            "n_pairs_flagged": len(redundant),
            "pairs": redundant[:40],
            "policy": (
                "flagged pairs are reported in feature_redundancy.csv and never dropped: "
                "dropping columns on the pooled matrix would use test-fold information"
            ),
        },
        "n_repeats": cfg.splitting.n_repeats,
        "n_splits": cfg.splitting.n_splits,
        "inner_splits": t.inner_splits,
        "models": model_summary,
        "comparison": comparison,
        "deep": {
            **_deep_block(pool, deep_branch, cfg, run_dir),
            "dependency": dep,
            **({"unavailable": deep_unavailable} if deep_unavailable else {}),
        },
        "design_notes": {
            "split_unit": "file (each file is its own group; no replicate or session metadata)",
            "holdout": "none (no replicates to hold out)",
            "tuning": (
                "logreg C chosen by inner grouped CV within each outer training fold; the CNN "
                "early-stops and temperature-scales on a validation split carved from the "
                "outer training fold by file group"
            ),
            "imputation": "median, fitted within each training fold",
            "window_scaling": "robust median/IQR fitted on each fold's training windows only",
            "aggregation": f"window -> file by {cfg.deep.aggregation}, declared before evaluation",
            "bootstrap": "file-level only; a hierarchical bootstrap needs replicate groups",
            "class_weighting": (
                "balanced (imbalance at or above 1.5)"
                if class_weight
                else "none (imbalance below the design threshold of 1.5)"
            ),
            "hybrid_combination": (
                f"fixed convex blend, Model A weight {cfg.deep.model_a_weight:g}"
                if "hybrid" in entries
                else "not used"
            ),
            "fusion_rule": str(cfg.deep.fusion),
            "fusion_note": fusion_note,
            "representation_flow": (
                "one 1D ConvNet encoder is fitted per outer fold; its global-average-pooled "
                "embedding is the single learned representation of the run and is consumed by "
                "(1) the softmax head -> cnn1d, (2) the nearest-prototype head -> proto_cnn, "
                "(3) the per-fold linear fusion head on handcrafted features + mean embedding -> "
                "fusion, (4) the probability blend with the best Model A -> hybrid, and (5) the "
                "conditional analyses: few-shot prototypes, semi-supervised pseudo-labels, "
                "novelty detection, clustering, and the file-level explanation map. Nothing "
                "downstream re-learns a representation, and no strategy is evaluated on files "
                "that contributed to the encoder, the prototypes, the fusion head, or the scaler."
                if deep_branch is not None
                else "the deep branch did not run, so no learned representation exists; every reported "
                "model uses the handcrafted features only"
            ),
            "matched_subset": (
                "Model A is also scored on the Model B subset so the comparison is "
                "like for like; see comparison.model_a_on_model_b_subset"
                if comparison.get("model_a_on_model_b_subset")
                else "not applicable"
            ),
        },
    }
    if stages:
        from .stages import TrainContext, run_downstream

        ctx = TrainContext(
            y=y,
            classes=classes,
            x=x,
            feat_cols=feat_cols,
            groups=groups,
            fold_ids=fold_ids,
            sids=sids,
            ok_rows=ok,
            manifest=manifest,
            quad_branch=quad_branch,
            entries=entries,
            agg_pred=agg_pred,
            imbalance=imbalance,
            pool=pool,
            deep_branch=deep_branch,
            summary=summary,
        )
        summary["downstream"] = run_downstream(cfg, run_dir, ctx, say)
    write_json_exclusive(run_dir / "summary.json", summary)
    return summary


def _opt_int(v: Any) -> int | None:
    """Event counts come from features as floats; anything absent or non-finite stays absent."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) else None


def write_train_metadata(
    cfg: TrapsigConfig,
    run_dir: Path,
    prepared_dir: Path,
    table: list[dict[str, Any]],
    sids: list[str],
    deep_branch: Any,
    say: Any = None,
) -> None:
    """Write this training run's own ``metadata.csv`` / ``metadata.json``.

    The train run gets its own copy rather than pointing at the prepare run, because the run
    folder must be self-contained: it records not only the split but also what happened to
    each file in THIS run (status, event count, window count, whether Model B could score it).
    """
    from .metadata import write_run_metadata

    tell = say or (lambda m: None)
    mdir = prepared_dir / "data" / "manifests"
    manifest_path = mdir / "master_manifest.csv"
    fold_path = mdir / "fold_assignments.csv"
    if not manifest_path.is_file() or not fold_path.is_file():
        tell("metadata: prepared run has no manifest or folds; metadata not written")
        return
    rows = read_manifest_csv(manifest_path)
    idx_of = {r["sample_id"]: i for i, r in enumerate(rows)}
    with fold_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fold_cols = [c for c in (reader.fieldnames or []) if c.startswith("fold_r")]
        fold_rows = list(reader)
    pool_positions = [idx_of[r["sample_id"]] for r in fold_rows if r["sample_id"] in idx_of]
    # The CSV stores one row per file with the repeats as columns; build_metadata_rows wants the
    # (n_repeats, n_pool) layout that grouped_stratified_folds returns, so transpose.
    per_file = np.array([[int(r[c]) for c in fold_cols] for r in fold_rows], dtype=np.int64)
    fold_ids = per_file.T if per_file.size else None
    ignored_path = mdir / "ignored_files.csv"
    n_ignored = 0
    if ignored_path.is_file():
        with ignored_path.open(encoding="utf-8", newline="") as fh:
            n_ignored = max(0, sum(1 for _ in csv.DictReader(fh)))

    win_by_sid: dict[str, int] = {}
    scored_by_sid: dict[str, bool] = {}
    val_tags_by_sid: dict[str, list[str]] = {}
    if deep_branch is not None:
        for i, sid in enumerate(sids):
            win_by_sid[sid] = int(deep_branch.counts[i])
            scored_by_sid[sid] = bool(deep_branch.has_windows[i])
        for tag, tag_sids in (getattr(deep_branch, "inner_val", {}) or {}).items():
            for sid in tag_sids:
                val_tags_by_sid.setdefault(str(sid), []).append(str(tag))
    extras: dict[str, dict[str, Any]] = {}
    for row in table:
        sid = row["sample_id"]
        extras[sid] = {
            "status": row.get("status", ""),
            "reason": row.get("reason", ""),
            "n_events": _opt_int(row.get("event_n")),
            "n_windows": win_by_sid.get(sid),
            "model_b_scored": scored_by_sid.get(sid),
            "cnn_inner_val_folds": sorted(val_tags_by_sid.get(str(sid), [])),
        }
    write_run_metadata(
        run_dir,
        cfg,
        rows,
        pool_positions,
        fold_ids,
        n_ignored,
        "train",
        extras,
        extra_block={"prepared_run": str(prepared_dir)},
    )
    tell(f"metadata: split record for this run written to {run_dir.name}/metadata.csv")


def _rows(proba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Probability rows for the evaluated subset, with NaN replaced by a uniform vector.

    A NaN row means the model produced nothing for that file. Uniform probabilities make the
    metric engine report a defined value instead of crashing, and the file count next to the
    metric says how many files were really scored.
    """
    out = np.array(proba[mask], dtype=float, copy=True)
    bad = np.isnan(out[:, 0])
    if bad.any():
        out[bad] = 1.0 / out.shape[1]
    return out


def _nanmean_repeats(proba: np.ndarray) -> np.ndarray:
    """Mean probability across repeats, ignoring files a model could not score."""
    import warnings

    with warnings.catch_warnings():
        # A file no repeat could score yields an all-NaN slice; that is expected and handled below.
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(proba, axis=0)
    out = np.asarray(m, dtype=float)
    return out


def _permutation_null_feature(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fold_ids: np.ndarray,
    classes: list[str],
    cfg: TrapsigConfig,
    class_weight: str | None,
    ent: ModelEntry,
    rng: np.random.Generator,
    say: Any,
) -> dict[str, Any]:
    """Label-shuffle null for a feature model: shuffle labels across files, rerun the full CV."""
    t = cfg.training
    observed = float(
        np.nanmean(
            [
                balanced_accuracy(y[ent.mask], ent.pred[r][ent.mask], classes)
                for r in range(cfg.splitting.n_repeats)
            ]
        )
    )
    null: list[float] = []
    for p in range(t.n_permutations):
        yp = rng.permutation(y)
        pred_p, _, _ = cv_predictions(name, x, yp, groups, fold_ids, classes, cfg, class_weight)
        null.append(
            float(
                np.nanmean(
                    [
                        balanced_accuracy(yp[ent.mask], pred_p[r][ent.mask], classes)
                        for r in range(cfg.splitting.n_repeats)
                    ]
                )
            )
        )
        say(f"permutation {name} {p + 1}/{t.n_permutations}")
    return _summarise_null(observed, null, t.n_permutations)


def _permutation_null_deep(
    ws: WindowSet | None,
    y: np.ndarray,
    sids: list[str],
    fold_ids: np.ndarray,
    classes: list[str],
    cfg: TrapsigConfig,
    imbalance: float,
    ent: ModelEntry,
    rng: np.random.Generator,
    say: Any,
) -> dict[str, Any]:
    """Label-shuffle null for the CNN: shuffle labels across files, rerun the whole deep CV.

    Off by default. Each permutation costs one full repeated CNN cross-validation, which on CPU
    is minutes to hours depending on the window length. The reason is recorded in the summary
    rather than the number being silently omitted.
    """
    n_perm = cfg.deep.n_permutations
    mask = ent.mask
    observed = float(
        np.nanmean(
            [balanced_accuracy(y[mask], ent.pred[r][mask], classes) for r in range(cfg.splitting.n_repeats)]
        )
    )
    if n_perm <= 0:
        return {
            "status": "not run",
            "n_permutations": 0,
            "observed_balanced_accuracy_mean_over_repeats": observed,
            "reason": "deep.n_permutations is 0; each CNN permutation costs one full repeated "
            "cross-validation on CPU. Raise it if a Model B null is required.",
        }
    if ws is None or ws.x.shape[0] == 0:
        return {
            "status": "not applicable",
            "n_permutations": n_perm,
            "observed_balanced_accuracy_mean_over_repeats": observed,
            "reason": "no event windows",
        }
    null: list[float] = []
    for p in range(n_perm):
        yp = rng.permutation(y)
        new_label = dict(zip(sids, yp.tolist(), strict=True))
        ws_p = dataclasses.replace(
            ws, class_label=np.array([new_label[s] for s in ws.sample_id.tolist()], dtype=object)
        )
        from .hybrid import run_deep_branch

        br = run_deep_branch(ws_p, fold_ids, sids, classes, cfg, imbalance, None, say)
        sub = br.has_windows
        null.append(
            float(
                np.nanmean(
                    [
                        balanced_accuracy(yp[sub], br.pred[r][sub], classes)
                        for r in range(cfg.splitting.n_repeats)
                    ]
                )
            )
        )
        say(f"permutation cnn1d {p + 1}/{n_perm}")
    return _summarise_null(observed, null, n_perm)


def _summarise_null(observed: float | None, null: list[float], n_perm: int) -> dict[str, Any]:
    out: dict[str, Any] = {"observed_balanced_accuracy_mean_over_repeats": observed, "n_permutations": n_perm}
    if null:
        arr = np.array(null)
        out.update(
            {
                "null_mean": float(arr.mean()),
                "null_p95": float(np.percentile(arr, 95)),
                "p_value": float((1 + np.sum(arr >= (observed or 0.0))) / (1 + arr.size)),
                "smallest_attainable_p": float(1.0 / (1.0 + arr.size)),
            }
        )
    else:
        out["p_value"] = None
        out["status"] = "not run (n_permutations = 0)"
    return out


def _comparison_block(
    entries: dict[str, ModelEntry],
    y: np.ndarray,
    classes: list[str],
    cfg: TrapsigConfig,
    agg_ci: dict[str, tuple[float, float] | None],
    run_dir: Path,
    t: Any,
    pool: PoolData,
    model_set: str,
    say: Any,
) -> dict[str, Any]:
    """Model-comparison framework: one table, plus the design's G7 gate on a matched subset."""
    per_rep: dict[str, list[dict[str, Any]]] = {}
    n_files_by: dict[str, int] = {}
    for name, ent in entries.items():
        m = ent.mask
        n_files_by[name] = int(m.sum())
        lst = []
        for r in range(cfg.splitting.n_repeats):
            mets = classification_metrics(
                y[m], ent.pred[r][m], labels=classes, y_proba=_rows(ent.proba[r], m)
            )
            lst.append(
                {k: {"value": v.value, "status": v.status, "reason": v.reason} for k, v in mets.items()}
            )
        per_rep[name] = lst

    matched: dict[str, Any] = {}
    # Model A restricted to the Model B subset: the only fair head-to-head.
    if "cnn1d" in entries and any(e.kind == "feature" for e in entries.values()):
        sub = entries["cnn1d"].mask
        a_names = [n for n, e in entries.items() if e.kind == "feature"]
        best_a, best_a_val = None, None
        sub_vals: dict[str, list[float]] = {}
        for n in a_names:
            vals = []
            for r in range(cfg.splitting.n_repeats):
                mets = classification_metrics(
                    y[sub], entries[n].pred[r][sub], labels=classes, y_proba=_rows(entries[n].proba[r], sub)
                )
                v = mets["balanced_accuracy"].value
                if v is not None:
                    vals.append(float(v))
            if vals:
                sub_vals[n] = vals
                mv = float(np.mean(vals))
                if best_a_val is None or mv > best_a_val:
                    best_a, best_a_val = n, mv
        if best_a is not None:
            agg_a = _nanmean_repeats(entries[best_a].proba)
            agg_a_pred = np.array(classes, dtype=object)[np.argmax(agg_a[sub], axis=1)]
            ci_a = bootstrap_ci(y[sub], agg_a_pred, classes, t.bootstrap_resamples, cfg.run.seed)
            ent_b = entries["cnn1d"]
            agg_b = _nanmean_repeats(ent_b.proba)
            agg_b_pred = np.array(classes, dtype=object)[np.argmax(agg_b[sub], axis=1)]
            ci_b = bootstrap_ci(y[sub], agg_b_pred, classes, t.bootstrap_resamples, cfg.run.seed)
            matched = {
                "n_files": int(sub.sum()),
                "model_a_on_model_b_subset": {
                    "per_model_balanced_accuracy": {
                        n: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                        for n, v in sub_vals.items()
                    },
                    "best_model_a": best_a,
                    "best_value": best_a_val,
                    "bootstrap_ci95": list(ci_a) if ci_a else None,
                },
                "model_b": {
                    "value": float(
                        np.mean(
                            [
                                v
                                for r in range(cfg.splitting.n_repeats)
                                for v in [
                                    classification_metrics(y[sub], ent_b.pred[r][sub], labels=classes)[
                                        "balanced_accuracy"
                                    ].value
                                ]
                                if v is not None
                            ]
                        )
                        or float("nan")
                    ),
                    "bootstrap_ci95": list(ci_b) if ci_b else None,
                },
            }
            from .hybrid import gate_g7

            matched["gate_G7"] = gate_g7(
                best_a_val, best_a, ci_a, matched["model_b"]["value"], "cnn1d", ci_b, int(sub.sum())
            )

    from .hybrid import attach_cis, build_comparison, write_comparison_csv

    rows = build_comparison(per_rep, n_files_by)
    cis: dict[tuple[str, str], tuple[float, float]] = {}
    for n, ci in agg_ci.items():
        if ci is not None:
            cis[(n, "balanced_accuracy")] = ci
    attach_cis(rows, cis)
    write_comparison_csv(rows, run_dir / "model_comparison.csv")
    say(f"comparison: {len(rows)} rows over {len(entries)} models")
    return {
        "models_evaluated": list(entries),
        "n_files_by_model": n_files_by,
        "table": [
            {
                "model": r.model,
                "metric": r.metric,
                "mean": r.mean,
                "sd": r.sd,
                "ci95_lo": r.ci_lo,
                "ci95_hi": r.ci_hi,
                "n_files": r.n_files,
                "status": r.status,
                "reason": r.reason,
            }
            for r in rows
        ],
        "matched_subset": matched,
        "model_set": model_set,
        "window_ms": pool.window_ms if math.isfinite(pool.window_ms) else None,
        "window_ms_source": pool.window_ms_source or None,
    }


def _deep_block(pool: PoolData, deep_branch: Any, cfg: TrapsigConfig, run_dir: Path) -> dict[str, Any]:
    """Reproducibility metadata for Model B: windows, architecture, training cost, checkpoints."""
    ws = pool.windows
    out: dict[str, Any] = {
        "enabled": cfg.deep.enabled and cfg.deep.model_set in ("deep", "hybrid"),
        "window_ms": pool.window_ms if math.isfinite(pool.window_ms) else None,
        "window_ms_source": pool.window_ms_source or None,
        "window_rate_hz": pool.window_rate_hz if math.isfinite(pool.window_rate_hz) else None,
        "window_length_samples": pool.window_length_samples,
        "event_duration_stats_ms": pool.duration_stats,
        "aggregation": cfg.deep.aggregation,
        "signal_cache_dir": pool.cache_dir,
        "signal_cache_bytes": pool.cache_bytes,
    }
    if ws is not None:
        out.update(
            {
                "n_windows": int(ws.x.shape[0]),
                "n_files_with_windows": int(sum(1 for v in ws.per_file_counts.values() if v > 0)),
                "n_files_without_windows": int(sum(1 for v in ws.per_file_counts.values() if v == 0)),
                "n_events_edge_excluded": int(ws.n_edge_excluded),
            }
        )
    if deep_branch is not None and deep_branch.cv is not None:
        cv = deep_branch.cv
        out.update(
            {
                "architecture": "stem(strided conv) -> residual dilated conv blocks -> GAP -> linear",
                "n_parameters": cv.n_parameters,
                "receptive_field_samples": cv.receptive_field_samples,
                "receptive_field_ms": (
                    round(
                        cv.receptive_field_samples
                        / float(cfg.deep.window_rate_hz or cfg.training.analysis_rate_hz)
                        * 1000.0,
                        3,
                    )
                    if cv.receptive_field_samples
                    else None
                ),
                "select_metric": cfg.deep.select_metric,
                "representation_learning": (
                    "supervised contrastive auxiliary loss on the GAP embedding"
                    if cfg.deep.metric_loss == "supcon"
                    else "batch-prototype auxiliary loss on the GAP embedding"
                    if cfg.deep.metric_loss == "prototypical"
                    else "contrastive objective disabled; cross-entropy only"
                ),
                "metric_loss": str(cfg.deep.metric_loss),
                "metric_loss_weight": float(cfg.deep.metric_loss_weight),
                "metric_temperature": float(cfg.deep.metric_temperature),
                "prototypical_head": bool(cfg.deep.prototypical_head),
                "proto_distance": str(cfg.deep.proto_distance),
                "fusion_rule": str(cfg.deep.fusion),
                "fusion_explanation": {
                    "blend": "probability blend of Model A and Model B",
                    "stack": "linear head on handcrafted features + learned embedding",
                    "both": "probability blend and stacked head, both reported",
                    "none": "no fusion; Model A and Model B are reported separately",
                }[str(cfg.deep.fusion)],
                "temperature_mean": float(np.mean(cv.temperature)),
                "n_prototyped_files": (
                    int(deep_branch.has_proto.sum())
                    if getattr(deep_branch, "has_proto", None) is not None
                    else 0
                ),
                "total_train_seconds": round(cv.seconds, 1),
                "mean_fold_seconds": round(cv.seconds / max(len(cv.logs), 1), 1),
                "n_folds_trained": len(cv.logs),
                "checkpoints_dir": str(run_dir / "checkpoints"),
                "training_log": str(run_dir / "logs" / "deep_training_log.jsonl"),
                "notes": deep_branch.notes + cv.notes,
                "epochs_run_per_fold": [lg["epochs_run"] for lg in cv.logs],
                "best_epoch_per_fold": [lg["best_epoch"] for lg in cv.logs],
            }
        )
    elif deep_branch is not None:
        out["notes"] = deep_branch.notes
    return out


def _write_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["scope", "model", "metric", "value", "status", "reason", "n_files"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
    write_text_exclusive(path, buf.getvalue())


def _write_predictions(
    ok: list[dict[str, Any]],
    entries: dict[str, ModelEntry],
    classes: list[str],
    fold_ids: np.ndarray,
    agg_pred: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Every out-of-fold prediction, one row per model, repeat, and file.

    ``evaluated`` is false for a file a model cannot score (Model B on a file with no usable
    event window). ``hybrid_source`` says whether a hybrid row was a real blend or a Model A
    fallback, so the blend is never silently an average over one model.
    """
    buf = io.StringIO()
    proba_cols = [f"p_{c}" for c in classes]
    w = csv.DictWriter(
        buf,
        fieldnames=[
            "sample_id",
            "class",
            "group_id",
            "model",
            "kind",
            "repeat",
            "fold",
            "evaluated",
            "hybrid_source",
            "pred",
            "chosen_C",
        ]
        + proba_cols,
    )
    w.writeheader()
    for name, ent in entries.items():
        for r in range(fold_ids.shape[0]):
            for i, row in enumerate(ok):
                in_mask = bool(ent.mask[i])
                p = ent.proba[r, i]
                rec = {
                    "sample_id": row["sample_id"],
                    "class": row["class"],
                    "group_id": row["group_id"],
                    "model": name,
                    "kind": ent.kind,
                    "repeat": r,
                    "fold": int(fold_ids[r, i]),
                    "evaluated": in_mask,
                    "hybrid_source": ent.source[r, i] if ent.source is not None else "",
                    "pred": ent.pred[r, i] if in_mask else "",
                    "chosen_C": (
                        "" if ent.chosen is None or math.isnan(ent.chosen[r, i]) else ent.chosen[r, i]
                    ),
                }
                for j, c in enumerate(classes):
                    rec[f"p_{c}"] = "" if not in_mask or not np.isfinite(p[j]) else float(p[j])
                w.writerow(rec)
        if name in agg_pred:
            mean_proba = _nanmean_repeats(ent.proba)
            for i, row in enumerate(ok):
                in_mask = bool(ent.mask[i])
                rec = {
                    "sample_id": row["sample_id"],
                    "class": row["class"],
                    "group_id": row["group_id"],
                    "model": name,
                    "kind": ent.kind,
                    "repeat": "aggregated",
                    "fold": "",
                    "evaluated": in_mask,
                    "hybrid_source": "",
                    "pred": agg_pred[name][i],
                    "chosen_C": "",
                }
                for j, c in enumerate(classes):
                    v = mean_proba[i, j]
                    rec[f"p_{c}"] = "" if not in_mask or not np.isfinite(v) else float(v)
                w.writerow(rec)
    write_text_exclusive(path, buf.getvalue())


def _write_confusion(y: np.ndarray, pred: np.ndarray, classes: list[str], path: Path) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["true\\pred"] + classes)
    for c in classes:
        w.writerow([c] + [int(np.sum((y == c) & (pred == p))) for p in classes])
    write_text_exclusive(path, buf.getvalue())
