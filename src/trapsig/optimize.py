"""Model-specific hyperparameter search, nested inside the outer cross-validation.

Every search here runs **inside one outer training fold**: it sees only the files that fold trains
on, scored by grouped inner splits of those same files. The outer test fold never influences a
hyperparameter that is later used to predict it, which is the whole point of nesting.

Two methods are offered because they answer to different budgets:

* ``random`` — uniform sampling without replacement over the declared per-model space, scored on
  every usable inner split. Simple, reproducible, and the right default for tens of files.
* ``halving`` — successive halving (Jamieson & Talwalkar 2016): all trials start on a grouped
  subsample of the training files, the best third is promoted to a larger subsample, and so on.
  Cheaper per trial, so it can afford a wider space; it falls back to ``random`` with a recorded
  reason when the fold is too small to subsample meaningfully.

Spaces are model-specific and deliberately exclude parameters that do not apply to a model: a
linear model gets regularisation strength, trees get depth and leaf size, boosting gets learning
rate and leaf count. Nothing here tunes the number of classes or the metric.

``logreg`` keeps its exact grid search over ``lr_c_grid`` when the search is off, because that grid
is small enough to enumerate and it is what the shipped results were produced with.
"""

from __future__ import annotations

import csv
import io
import itertools
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .config import TrapsigConfig
from .runs import write_text_exclusive

Say = Any  # callable(str) -> None

HPO_METHODS = ("none", "random", "halving")
HPO_METRICS = ("balanced_accuracy", "macro_f1", "roc_auc_ovr", "log_loss")
# A metric is only usable when the inner validation split holds at least two classes; roc_auc_ovr
# additionally needs probabilities for every class, so it is refused rather than silently averaged.
HALVING_ETA = 3


@dataclass
class Trial:
    """One evaluated hyperparameter setting, with the rung it was evaluated at."""
    model: str
    trial: int
    rung: int
    n_train_files: int
    params: dict[str, Any]
    score: float | None
    n_inner_splits: int
    status: str                    # computed | failed
    reason: str = ""


@dataclass
class TuneResult:
    """What one (model, repeat, outer fold) search produced."""
    model: str
    method: str
    metric: str
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    n_trials: int = 0
    n_promoted: int = 0
    status: str = "not_run"        # computed | not_run | failed
    reason: str = ""
    seconds: float = 0.0
    history: list[Trial] = field(default_factory=list)
    space: dict[str, list[Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def search_space(name: str, cfg: TrapsigConfig) -> dict[str, list[Any]]:
    """The declared space of one model. Values are the candidates; one is drawn per trial.

    The upper bounds follow the configuration, so a user who declares ``rf_n_estimators: 100``
    searches up to 100 and not up to a hard-coded 500: the budget stays the user's decision.
    """
    t = cfg.training
    if name == "logreg":
        return {"clf__C": [float(c) for c in t.lr_c_grid]}
    if name == "rf":
        top = max(50, int(t.rf_n_estimators))
        return {
            "clf__n_estimators": sorted({50, 100, 200, top}),
            "clf__max_depth": [None, 4, 8, 16],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", 0.5, 1.0],
        }
    if name == "hgb":
        top = max(50, int(t.hgb_max_iter))
        return {
            "clf__max_iter": sorted({50, 100, top}),
            "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "clf__max_leaf_nodes": [7, 15, 31],
            "clf__min_samples_leaf": [5, 10, 20],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
        }
    raise KeyError(f"no hyperparameter space is defined for model {name!r}")


def build_estimator(name: str, cfg: TrapsigConfig, seed: int, class_weight: str | None,
                    params: dict[str, Any] | None = None) -> Any:
    """The object one trial fits: the very pipeline ``train._base_model`` builds for the fold.

    Delegating instead of constructing an estimator here matters twice over. The feature matrix
    carries NaNs by design (a feature domain can be undefined for a file, e.g. no detected event),
    and the shipped pipeline imputes with the median and standardises inside the fold; a bare
    LogisticRegression would fail or be judged on different numbers. It also means a hyperparameter
    can never be selected by a model that differs from the one reported.

    The import is local because ``train.py`` imports this module inside ``cv_predictions``.
    """
    from .train import _base_model

    return _base_model(name, cfg, seed, None, class_weight, params)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_predictions(y_true: np.ndarray, proba: np.ndarray, classes: list[str], metric: str
                      ) -> float | None:
    """One number, oriented so that higher is always better (log loss is negated)."""
    pred = np.array(classes, dtype=object)[np.argmax(proba, axis=1)]
    present = sorted({str(v) for v in y_true.tolist()})
    try:
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, pred))
        if metric == "macro_f1":
            return float(f1_score(y_true, pred, labels=classes, average="macro", zero_division=0))
        if metric == "log_loss":
            return -float(log_loss(y_true, np.clip(proba, 1e-12, 1.0), labels=classes))
        if metric == "roc_auc_ovr":
            if len(present) < 2 or proba.shape[1] < 2:
                return None                     # undefined, not 0.5: reported as a failed trial
            return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=classes))
    except (ValueError, ZeroDivisionError):
        return None
    raise KeyError(f"unknown hpo metric {metric!r}")


def _proba_in_class_order(model: Any, x: np.ndarray, classes: list[str]) -> np.ndarray:
    p = model.predict_proba(x)
    out = np.zeros((x.shape[0], len(classes)))
    idx = {c: j for j, c in enumerate(classes)}
    for j, c in enumerate(model.classes_):
        out[:, idx[str(c)]] = p[:, j]
    return out


def inner_splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped, class-stratified inner splits, keeping only the usable ones.

    A split whose validation side holds one class cannot score a discriminator, and a split whose
    training side holds one class cannot fit one; both are dropped and the drop is counted by the
    caller. On very small folds nothing may survive, and the search reports that instead of
    scoring on a degenerate split.
    """
    if n_splits < 2:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            raw = list(sg.split(x, y, groups))
        except ValueError:
            raw = []
    if not raw:
        # Too few groups per class for stratification: fall back to a plain grouped split, still
        # file-level, still inside the outer training fold.
        from sklearn.model_selection import GroupShuffleSplit

        out = []
        for i in range(n_splits):
            gss = GroupShuffleSplit(n_splits=2, test_size=1.0 / n_splits, random_state=seed + i)
            for itr, iva in gss.split(x, y, groups):
                out.append((itr, iva))
        raw = out
    usable = [(itr, iva) for itr, iva in raw
              if len({str(v) for v in y[itr].tolist()}) >= 2
              and len({str(v) for v in y[iva].tolist()}) >= 2
              and itr.size and iva.size]
    return usable


def _subsample_by_group(x: np.ndarray, y: np.ndarray, groups: np.ndarray, fraction: float,
                        seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep a fraction of the *files*, never part of a file: the group is the sampling unit."""
    uniq = np.array(sorted({str(g) for g in groups.tolist()}), dtype=object)
    if uniq.size <= 1 or fraction >= 1.0:
        return x, y, groups
    rng = np.random.default_rng(seed)
    keep = uniq[rng.random(uniq.size) < max(fraction, 1.0 / uniq.size)]
    if keep.size == 0:
        keep = uniq[: max(1, int(uniq.size * fraction))]
    keep_set = {str(g) for g in keep.tolist()}
    m = np.array([str(g) in keep_set for g in groups.tolist()])
    if not m.any():
        return x, y, groups
    return x[m], y[m], groups[m]


# ---------------------------------------------------------------------------
# The searches
# ---------------------------------------------------------------------------

def _draw_params(space: dict[str, list[Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Uniform sampling without replacement, falling back to with-replacement for big spaces."""
    keys = sorted(space)
    combos = list(itertools.product(*(space[k] for k in keys)))
    rng = np.random.default_rng(seed)
    if len(combos) <= n:
        draws = list(combos)
        rng.shuffle(draws)
        return [dict(zip(keys, c, strict=True)) for c in draws]
    idx = rng.choice(len(combos), size=n, replace=False)
    return [dict(zip(keys, combos[int(i)], strict=True)) for i in idx]


def _evaluate(name: str, params: dict[str, Any], x: np.ndarray, y: np.ndarray,
              groups: np.ndarray, classes: list[str], cfg: TrapsigConfig, seed: int,
              class_weight: str | None, splits: list[tuple[np.ndarray, np.ndarray]],
              metric: str) -> tuple[float | None, int, str]:
    """Mean inner score of one setting, plus how many splits contributed and why not more."""
    scores: list[float] = []
    last_error = ""
    for itr, iva in splits:
        try:
            m = build_estimator(name, cfg, seed, class_weight, params)
            m.fit(x[itr], y[itr])
            p = _proba_in_class_order(m, x[iva], classes)
        except Exception as exc:                       # a failed trial must not stop the search
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        v = score_predictions(y[iva], p, classes, metric)
        if v is None:
            last_error = f"metric {metric} is undefined on this inner split"
            continue
        scores.append(v)
    if not scores:
        return None, 0, last_error or "no usable inner split"
    return float(np.mean(scores)), len(scores), ""


def tune_model(name: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, classes: list[str],
               cfg: TrapsigConfig, seed: int, class_weight: str | None = None,
               fold_tag: str = "", say: Say = None) -> TuneResult:
    """Search one model's hyperparameters inside one outer training fold.

    Returns the best setting, every trial, and the reason if the search could not run. The caller
    is responsible for using ``best_params`` only on this fold's test files.
    """
    tell = say or (lambda m: None)
    t = cfg.training
    metric = t.hpo_metric
    space = search_space(name, cfg)
    out = TuneResult(model=name, method=t.hpo, metric=metric, space=space)
    if t.hpo == "none":
        out.status, out.reason = "not_run", "training.hpo is 'none'"
        return out
    t0 = time.perf_counter()
    splits = inner_splits(x, y, groups, t.inner_splits, seed)
    n_files = len({str(g) for g in groups.tolist()})
    if not splits:
        out.status = "failed"
        out.reason = (f"no usable inner split: {n_files} file(s) in this training fold cannot be "
                      f"divided into {t.inner_splits} class-stratified grouped splits")
        out.seconds = time.perf_counter() - t0
        return out

    n_iter = max(1, int(t.hpo_n_iter))
    draws = _draw_params(space, n_iter, seed + t.hpo_seed_offset)
    history: list[Trial] = []
    best_params, best_score = dict(draws[0]) if draws else {}, None

    if t.hpo == "random" or n_files < 2 * HALVING_ETA:
        if t.hpo == "halving":
            out.reason = (f"successive halving needs at least {2 * HALVING_ETA} files per rung; "
                          f"this fold has {n_files}, so plain random search was used")
        for i, params in enumerate(draws):
            score, n_used, why = _evaluate(name, params, x, y, groups, classes, cfg, seed,
                                           class_weight, splits, metric)
            history.append(Trial(name, i, 0, n_files, params, score, n_used,
                                 "computed" if score is not None else "failed", why))
            if score is not None and (best_score is None or score > best_score):
                best_score, best_params = score, dict(params)
    else:
        # Successive halving: start every trial on 1/eta of the files, promote the best third.
        rung = 0
        alive = list(enumerate(draws))
        fraction = 1.0 / (HALVING_ETA ** max(1, int(math.log(len(alive), HALVING_ETA))))
        while alive:
            xs, ys, gs = _subsample_by_group(x, y, groups, fraction, seed + 17 * rung)
            sub_splits = inner_splits(xs, ys, gs, t.inner_splits, seed + rung) or splits
            scored: list[tuple[float, int, dict[str, Any]]] = []
            for i, params in alive:
                score, n_used, why = _evaluate(name, params, xs, ys, gs, classes, cfg, seed,
                                               class_weight, sub_splits, metric)
                history.append(Trial(name, i, rung, len({str(g) for g in gs.tolist()}),
                                     params, score, n_used,
                                     "computed" if score is not None else "failed", why))
                if score is not None:
                    scored.append((score, i, params))
            if not scored:
                out.reason = f"every trial failed at rung {rung}; see the trial history"
                break
            scored.sort(key=lambda r: -r[0])
            keep = max(1, len(scored) // HALVING_ETA)
            best_score, _, best_params = scored[0]
            alive = [(i, p) for _, i, p in scored[:keep]]
            out.n_promoted = len(alive)
            if fraction >= 1.0 or len(alive) <= 1:
                break
            fraction = min(1.0, fraction * HALVING_ETA)
            rung += 1

    usable = [h for h in history if h.status == "computed"]
    out.history = history
    out.n_trials = len(history)
    out.seconds = time.perf_counter() - t0
    if best_score is None:
        out.status = "failed"
        out.reason = out.reason or "no trial produced a score on this fold"
        out.best_params = {}
        tell(f"hpo {name} {fold_tag}: FAILED - {out.reason}")
        return out
    out.status = "computed"
    out.best_params = best_params
    out.best_score = best_score
    tell(f"hpo {name} {fold_tag}: {len(usable)}/{len(history)} trials scored, best {metric} "
         f"{best_score:.4f} at {best_params}")
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

HPO_FIELDS = ["model", "repeat", "fold", "fold_tag", "method", "metric", "trial", "rung",
              "n_train_files", "score", "n_inner_splits", "status", "reason", "params"]


def write_hpo_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """One row per trial, so a search can be audited rather than trusted."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=HPO_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in HPO_FIELDS})
    return write_text_exclusive(path, buf.getvalue())


def hpo_summary(results: list[TuneResult], method: str, metric: str) -> dict[str, Any]:
    """The per-model outcome of the search: best setting, score, budget, and failures."""
    by_model: dict[str, Any] = {}
    for r in results:
        e = by_model.setdefault(r.model, {"method": r.method, "metric": r.metric,
                                          "space": r.space, "folds": [], "n_trials": 0,
                                          "seconds": 0.0, "failures": []})
        e["folds"].append({"best_params": r.best_params, "best_score": r.best_score,
                           "n_trials": r.n_trials, "status": r.status,
                           "seconds": round(r.seconds, 3)})
        e["n_trials"] += r.n_trials
        e["seconds"] = round(float(e["seconds"]) + r.seconds, 3)
        if r.status != "computed" and r.reason:
            e["failures"].append(r.reason)
    for e in by_model.values():
        scores = [f["best_score"] for f in e["folds"] if isinstance(f["best_score"], float)]
        e["best_score_mean"] = float(np.mean(scores)) if scores else None
        e["n_folds_failed"] = len(e["failures"])
        e["failures"] = sorted(set(e["failures"]))[:5]
    return {"method": method, "metric": metric, "higher_is_better": True,
            "objective": (f"mean inner grouped {metric} inside each outer training fold; the "
                          "outer test fold never influences the setting used to predict it"),
            "budget": {"trials_per_model_per_fold": len(results[0].history) if results else 0,
                       "n_model_fold_searches": len(results)},
            "per_model": by_model,
            "note": ("log loss is negated so that every metric is maximised; a trial whose metric "
                     "is undefined on an inner split is recorded as failed, never as 0.5")}
