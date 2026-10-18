"""Inference from saved artifacts (design section 13: a run must be reusable after it finishes).

Training saves the fitted Model A pipeline of every (repeat, outer fold) together with a JSON
sidecar that records exactly which feature columns, classes and hyperparameters it was fitted
with. Nothing here re-derives any of that: the sidecar is the contract, and a prediction is only
made with the columns the pipeline was trained on, in the order it was trained on them.

Two reasons the sidecar exists rather than a pickled feature list:

* A feature matrix without its column order is a silent bug. The pipeline would accept any array
  of the right width and produce confident nonsense.
* A model saved by one code version must say what it is. ``trapsig_version`` and the config
  fingerprint of the training run are recorded so a stale model is refused, not guessed at.

Predictions for a file are the mean probability over the saved fold models of the requested model
name (each was fitted on a different outer training fold, all of them leakage-free with respect to
new files). ``--fold-tag`` restricts the ensemble to one saved model.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrapsigConfig, load_config
from .features import extract_features
from .pipeline import RecordingSkipped, process_recording

MODEL_DIRNAME = "models"
INDEX_NAME = "INDEX.json"


class PredictError(RuntimeError):
    """Raised when inference cannot proceed for a stated reason."""


@dataclass
class SavedModel:
    """A fitted pipeline plus the record of what it was fitted with."""

    path: Path
    pipeline: Any
    meta: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.meta.get("model", ""))

    @property
    def feature_columns(self) -> list[str]:
        return [str(c) for c in self.meta.get("feature_columns", [])]

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.meta.get("classes", [])]


def save_model(pipeline: Any, meta: dict[str, Any], path: Path) -> Path:
    """Persist one fitted pipeline next to its JSON sidecar."""
    import joblib  # type: ignore[import-untyped]

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(meta, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return path


def load_saved_model(path: Path, want_version: str | None = None) -> SavedModel:
    """Load a saved pipeline and refuse it if its sidecar disagrees with this code."""
    import joblib

    path = Path(path)
    if not path.is_file():
        raise PredictError(f"saved model not found: {path}")
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.is_file():
        raise PredictError(f"saved model has no sidecar, so its columns are unknown: {path}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if want_version and str(meta.get("trapsig_version", "")) != want_version:
        raise PredictError(
            f"{path.name} was saved by trapsig {meta.get('trapsig_version')!r} but this is "
            f"{want_version!r}; re-run training instead of loading a stale model")
    return SavedModel(path=path, pipeline=joblib.load(path), meta=meta)


def list_saved_models(run_dir: Path) -> list[dict[str, Any]]:
    """The index of a run's ``models/`` directory, or [] when nothing was saved."""
    idx = Path(run_dir) / MODEL_DIRNAME / INDEX_NAME
    if not idx.is_file():
        return []
    return list(json.loads(idx.read_text(encoding="utf-8")))


def feature_row_for(path: Path, cfg: TrapsigConfig) -> dict[str, float]:
    """The feature vector of one recording, built exactly the way training builds it."""
    t = cfg.training
    res = process_recording(path, cfg, t.analysis_rate_hz, keep_signal=True)
    if res.signal is None or res.baseline is None:
        raise PredictError(f"{path.name}: processed signal not returned")
    return extract_features(
        res.signal, res.baseline, res.fs_hz, res.events,
        raw_std=float(res.stats.get("raw_std", float("nan"))),
        raw_level=float(res.stats.get("raw_mean", float("nan"))),
        bands_hz=t.psd_bands_hz,
        acf_lags_ms=t.acf_lags_ms,
        groups=t.feature_groups,
    )


def _matrix(rows: list[dict[str, float]], columns: list[str]) -> np.ndarray:
    """The feature matrix in the saved column order; a missing column is a stated failure."""
    missing = sorted({c for c in columns for r in rows if c not in r})
    if missing:
        raise PredictError(
            "the saved model expects feature columns this code does not produce: "
            + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    out = np.full((len(rows), len(columns)), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for j, c in enumerate(columns):
            v = r.get(c)
            try:
                out[i, j] = float(v) if v is not None else np.nan
            except (TypeError, ValueError):
                out[i, j] = np.nan
    return out


def predict_files(paths: list[Path], models: list[SavedModel],
                  cfg: TrapsigConfig) -> list[dict[str, Any]]:
    """Predict every file with every saved model; probabilities are averaged over models."""
    if not models:
        raise PredictError("no saved model was given to predict with")
    columns = models[0].feature_columns
    classes = models[0].classes
    for m in models[1:]:
        if m.feature_columns != columns or m.classes != classes:
            raise PredictError(
                f"{m.path.name} was fitted on different columns or classes than "
                f"{models[0].path.name}; they cannot vote on the same file")
    rows: list[dict[str, Any]] = []
    for p in paths:
        rec: dict[str, Any] = {"file": str(p), "filename": Path(p).name, "status": "ok",
                               "reason": "", "n_models": len(models)}
        try:
            feats = feature_row_for(p, cfg)
        except RecordingSkipped as exc:
            rec.update({"status": "skipped", "reason": exc.reason, "predicted": ""})
            rec.update({f"prob_{c}": "" for c in classes})
            rows.append(rec)
            continue
        except Exception as exc:
            rec.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}",
                        "predicted": ""})
            rows.append(rec)
            continue
        x = _matrix([feats], columns)
        acc = np.zeros((1, len(classes)), dtype=float)
        used = 0
        for m in models:
            with warnings.catch_warnings():
                # A new file can leave a whole feature column NaN (a domain that is undefined for
                # it); the imputer then announces that it skips the column. Expected, not an error.
                warnings.simplefilter("ignore", UserWarning)
                proba = np.asarray(m.pipeline.predict_proba(x), dtype=float)
            if proba.shape[1] != len(classes):
                raise PredictError(f"{m.path.name} returns {proba.shape[1]} probability columns, "
                                   f"the sidecar promises {len(classes)}")
            acc += proba
            used += 1
        acc /= max(1, used)
        rec["predicted"] = classes[int(np.argmax(acc[0]))]
        for c, v in zip(classes, acc[0], strict=True):
            rec[f"prob_{c}"] = round(float(v), 6)
        rec["margin_top2"] = round(float(np.sort(acc[0])[-1] - np.sort(acc[0])[-2]), 6) \
            if len(classes) > 1 else None
        rows.append(rec)
    return rows


def load_run_config(run_dir: Path) -> TrapsigConfig:
    """The configuration the run itself used, so inference repeats training's feature settings."""
    cfg_path = Path(run_dir) / "config_used.yaml"
    if not cfg_path.is_file():
        raise PredictError(f"{run_dir} holds no config_used.yaml; is it a finished run folder?")
    return load_config(cfg_path)


def models_for(run_dir: Path, name: str, fold_tag: str | None = None,
               check_version: str | None = None) -> list[SavedModel]:
    """The saved models of one model name, optionally restricted to one fold tag."""
    entries = [e for e in list_saved_models(run_dir) if e.get("model") == name]
    if fold_tag:
        entries = [e for e in entries if e.get("fold_tag") == fold_tag]
    if not entries:
        have = sorted({str(e.get("model")) for e in list_saved_models(run_dir)})
        raise PredictError(
            f"run {run_dir} saved no {name!r} model"
            + (f" for fold {fold_tag!r}" if fold_tag else "")
            + (f"; available: {', '.join(have)}" if have else " (models/ is empty or missing)"))
    return [load_saved_model(Path(run_dir) / MODEL_DIRNAME / str(e["file"]), check_version)
            for e in entries]


def write_inference_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """The inference table; skipped files are kept and reported, exactly as in training."""
    import csv
    import io

    from .runs import write_text_exclusive

    fields = ["file", "filename", "status", "reason", "predicted", "margin_top2", "n_models"]
    classes: list[str] = []
    for r in rows:
        for k in r:
            if k.startswith("prob_") and k[5:] not in classes:
                classes.append(k[5:])
    fields += [f"prob_{c}" for c in classes]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return Path(write_text_exclusive(path, buf.getvalue()))
