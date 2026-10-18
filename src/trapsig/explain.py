"""Explainability (design section 11.12). Runs after the model is frozen.

Two things are deliberately separated here:

1. **Model A attributions** are feature level. Logistic regression gets exact linear SHAP
   (coefficients times centred features). The tree models get TreeSHAP when the optional ``shap``
   package is installed, and are reported as "not applicable" with the reason when it is not.
   Nothing invents a number for a method that could not run.

2. **Model B attributions** are temporal. The head is a single linear layer on the global average
   pool of the final convolutional map, which is the condition under which Grad-CAM is valid, so
   the output is labelled ``gradcam`` rather than something looser. Integrated gradients and
   occlusion are also computed; they make no architectural assumption.

Design R21 forbids SHAP on raw 1 MHz samples (20 million features per file). Attributions here are
computed on the window length actually fed to the model, and segment-level summaries are what get
written to CSV.

The **shortcut probes** are mandatory and are the most important output of this module: if a
baseline-only classifier can predict the class, the model is reading an acquisition artefact, not
the event.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrapsigConfig
from .runs import write_json_exclusive, write_text_exclusive
from .windows import WindowSet, apply_robust_scale, fit_robust_scale

try:  # optional: only needed for TreeSHAP on the tree models
    import shap as _shap
except Exception:  # pragma: no cover - depends on the user's environment
    _shap = None

try:  # optional: the whole Model B branch degrades to Model A without it
    import torch
    from torch import nn

    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False

Say = Any


class ExplainError(RuntimeError):
    """A condition that stops explanation."""


def shap_available() -> bool:
    return _shap is not None


def torch_available() -> bool:
    return _TORCH


# ---------------------------------------------------------------------------
# The model used for explanation
# ---------------------------------------------------------------------------

@dataclass
class ExplainerModel:
    """One CNN fitted on the whole pool, used only to produce attributions.

    This model is *not* used for any performance number. Every reported metric comes from the
    grouped cross-validation in :mod:`trapsig.train`. Fitting on the full pool is what makes the
    attributions describe the model class rather than one fold's accident, and it is labelled as
    such wherever it appears.
    """
    model: Any
    med: float
    iqr: float
    length: int
    classes: list[str]
    n_parameters: int
    note: str


def fit_explainer_model(ws: WindowSet, classes: list[str], cfg: TrapsigConfig,
                        say: Say = None) -> ExplainerModel | None:
    """Fit the explanation model. Returns None (with the reason logged) when it cannot be fitted."""
    from .deep import build_model, count_parameters, validation_split

    tell = say or (lambda m: None)
    if not torch_available():
        tell("explain: PyTorch is not installed; Model B attributions are not available")
        return None
    if ws is None or ws.x.shape[0] == 0:
        tell("explain: no event windows; Model B attributions are not applicable")
        return None
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[c] for c in ws.class_label.tolist()], dtype=np.int64)
    med, iqr = fit_robust_scale(ws.x)
    xs = apply_robust_scale(ws.x, med, iqr)

    va = validation_split(y, ws.group_id, cfg.deep.validation_fraction, cfg.run.seed, k)
    if not va.any() or va.all():
        va = np.zeros(y.shape[0], dtype=bool)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.run.seed + 991)
    model = build_model(ws.x.shape[1], k, cfg).to(dev)
    n_par = count_parameters(model)
    xt = torch.as_tensor(xs, dtype=torch.float32, device=dev).unsqueeze(1)
    yt = torch.as_tensor(y, dtype=torch.long, device=dev)
    counts = np.array([max(np.count_nonzero(y == c), 1) for c in range(k)], dtype=float)
    ratio = counts.max() / counts.min()
    from .deep import IMBALANCE_WEIGHT_THRESHOLD
    w = None
    if ratio >= IMBALANCE_WEIGHT_THRESHOLD:
        w = torch.as_tensor(counts.sum() / (k * counts), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.deep.learning_rate,
                            weight_decay=cfg.deep.weight_decay)
    n = xt.shape[0]
    bs = cfg.deep.batch_size
    steps = max(1, math.ceil(n / bs))
    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.run.seed)
    best, best_state, patience = -math.inf, None, 0
    for _epoch in range(cfg.deep.epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for b in range(steps):
            sel = perm[b * bs:(b + 1) * bs]
            if sel.numel() == 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[sel]), yt[sel])
            loss.backward()
            if cfg.deep.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.deep.gradient_clip)
            opt.step()
        if va.any():
            model.eval()
            with torch.no_grad():
                p = torch.softmax(model(xt[va]), dim=1).cpu().numpy()
            pred = np.argmax(p, axis=1)
            score = float(np.mean([np.mean(pred[y[va] == c] == c) for c in range(k)
                                   if np.any(y[va] == c)]))
            if score > best + cfg.deep.early_stopping_min_delta:
                best, patience = score, 0
                best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg.deep.early_stopping_patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    tell(f"explain: fitted the explanation model on {n} windows ({n_par} parameters)")
    return ExplainerModel(model=model, med=med, iqr=iqr, length=ws.x.shape[1], classes=classes,
                          n_parameters=n_par,
                          note=("fitted on the full pool for attribution only; no performance number "
                                "in this report comes from it"))


# ---------------------------------------------------------------------------
# Temporal attribution methods
# ---------------------------------------------------------------------------

def gradcam_1d(em: ExplainerModel, x_raw: np.ndarray, class_idx: int) -> np.ndarray:
    """Grad-CAM over one window, upsampled to the window length.

    Valid here because the class score is a linear function of the global average pool of the
    final convolutional map. Weights are the gradient of the score with respect to each channel's
    pooled activation; the map is their weighted sum, passed through a ReLU.
    """
    dev = next(em.model.parameters()).device
    x = apply_robust_scale(x_raw.reshape(1, -1), em.med, em.iqr)
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev).unsqueeze(1)
    em.model.zero_grad(set_to_none=True)
    feats = em.model.features(xt)                 # (1, C, T')
    score = em.model.head(feats.mean(dim=-1))[0, class_idx]
    grads = torch.autograd.grad(score, feats, retain_graph=False)[0]   # (1, C, T')
    weights = grads.mean(dim=(0, 2))              # (C,) global average pooled gradient per channel
    # feats[0] is (C, T'). Weight each channel map and sum over channels to get (T',).
    cam = (weights[:, None] * feats[0]).sum(dim=0)
    cam = torch.relu(cam).detach().cpu().numpy().astype(float).reshape(-1)
    up: np.ndarray = _upsample(cam, x_raw.size)
    return up


def integrated_gradients_1d(em: ExplainerModel, x_raw: np.ndarray, class_idx: int, steps: int,
                            baseline: np.ndarray | None = None) -> np.ndarray:
    """Integrated gradients from a baseline window to the input, along a straight path."""
    dev = next(em.model.parameters()).device
    x = apply_robust_scale(x_raw.reshape(1, -1), em.med, em.iqr)[0]
    b = (np.zeros_like(x) if baseline is None
         else apply_robust_scale(baseline.reshape(1, -1), em.med, em.iqr)[0])
    acc = np.zeros_like(x, dtype=float)
    for i in range(1, steps + 1):
        alpha = i / steps
        pt = b + alpha * (x - b)
        xt = torch.as_tensor(pt.reshape(1, 1, -1), dtype=torch.float32, device=dev).requires_grad_(True)
        sc = em.model.head(em.model.features(xt).mean(dim=-1))[0, class_idx]
        gr = torch.autograd.grad(sc, xt)[0].detach().cpu().numpy().reshape(-1)
        acc += gr / steps
    result: np.ndarray = np.asarray((x - b) * acc, dtype=float)
    return result


def occlusion_1d(em: ExplainerModel, x_raw: np.ndarray, class_idx: int, n_segments: int,
                 background_value: float = 0.0) -> np.ndarray:
    """Score drop when each segment is replaced by a flat background. Per-sample attribution."""
    dev = next(em.model.parameters()).device
    x = apply_robust_scale(x_raw.reshape(1, -1), em.med, em.iqr)[0]

    def score(v: np.ndarray) -> float:
        xt = torch.as_tensor(v.reshape(1, 1, -1), dtype=torch.float32, device=dev)
        with torch.no_grad():
            return float(torch.softmax(em.model.head(em.model.features(xt).mean(dim=-1)),
                                       dim=1)[0, class_idx])

    base = score(x)
    n = x.size
    edges = np.linspace(0, n, n_segments + 1).astype(int)
    attr: np.ndarray = np.zeros(n, dtype=float)
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        if b <= a:
            continue
        v = x.copy()
        v[a:b] = background_value
        drop = base - score(v)
        attr[a:b] = drop / max(b - a, 1)
    return attr


def _upsample(a: np.ndarray, n: int) -> np.ndarray:
    """Nearest-neighbour stretch of an attribution map to ``n`` samples."""
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size == 0:
        return np.zeros(n, dtype=float)
    if a.size == n:
        same: np.ndarray = np.array(a, dtype=float)
        return same
    idx = np.clip((np.arange(n) * a.size / n).astype(int), 0, a.size - 1)
    return np.asarray(a[idx], dtype=float)


def time_shift_null(em: ExplainerModel, x_raw: np.ndarray, class_idx: int, shift: int) -> dict[str, Any]:
    """Does the attribution move when the window is shifted?

    If a model reads the event, shifting the signal by ``shift`` samples should shift the
    attribution by about the same amount. If the attribution stays put, the model is keyed to
    position within the window, which is an artefact of where events happen to sit.
    """
    a0 = gradcam_1d(em, x_raw, class_idx)
    xs = np.roll(x_raw, shift)
    a1 = gradcam_1d(em, xs, class_idx)
    peak0 = int(np.argmax(a0))
    peak1 = int(np.argmax(a1))
    observed = peak1 - peak0
    return {"shift_samples": int(shift), "peak_before": peak0, "peak_after": peak1,
            "peak_moved": int(observed),
            "moves_with_event": bool(abs(observed - shift) <= max(2, abs(shift) // 5)),
            "correlation_before_after": float(_corr(a0, np.roll(a1, -shift)))}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def baseline_only_probe(ws: WindowSet, cfg: TrapsigConfig, classes: list[str],
                        fold_ids: np.ndarray, file_order: list[str], say: Say = None) -> dict[str, Any]:
    """Mandatory shortcut probe: can pre-event baseline alone predict the class?

    The pre-event segment of every window is kept and the event part is replaced by that
    window's own pre-event statistics, so no event shape survives. If the CNN still classifies
    above chance, the signal carrying the label is the baseline, which means session, file, or
    instrument differences rather than trapping behaviour.
    """
    tell = say or (lambda m: None)
    if not torch_available():
        return {"status": "not_applicable", "reason": "PyTorch is not installed"}
    if ws is None or ws.x.shape[0] == 0:
        return {"status": "not_applicable", "reason": "no event windows"}
    from .hybrid import run_deep_branch

    pre = max(2, int(round(cfg.deep.pre_event_fraction * ws.x.shape[1])))
    probe = ws.x.copy()
    for i in range(probe.shape[0]):
        probe[i, pre:] = float(np.mean(probe[i, :pre]))   # flatten everything after the pre-event part
    ws_probe = dataclasses.replace(ws, x=probe.astype(np.float32, copy=False))
    k = len(classes)
    br = run_deep_branch(ws_probe, fold_ids, file_order, classes, cfg, 1.0, None, tell)
    first_label: dict[str, str] = {}
    for sid, lab in zip(ws.sample_id.tolist(), ws.class_label.tolist(), strict=True):
        first_label.setdefault(sid, lab)
    y_files = np.array([first_label.get(s, "") for s in file_order], dtype=object)
    sub = br.has_windows
    vals = []
    for r in range(cfg.splitting.n_repeats):
        p = br.pred[r][sub]
        t = y_files[sub]
        rec = [float(np.mean(p[t == c] == c)) for c in classes if np.any(t == c)]
        if rec:
            vals.append(float(np.mean(rec)))
    return {
        "status": "computed",
        "probe": "event part of each window replaced by its own pre-event mean; baseline shape only",
        "pre_event_samples": pre,
        "balanced_accuracy_mean_over_repeats": float(np.mean(vals)) if vals else None,
        "n_files": int(sub.sum()),
        "chance": 1.0 / k,
        "interpretation": ("baseline alone predicts above chance: investigate session or instrument "
                           "artefacts before reading the model as biology"
                           if vals and float(np.mean(vals)) > 1.0 / k + 0.05 else
                           "baseline alone does not predict above chance"),
    }


# ---------------------------------------------------------------------------
# Model A attributions
# ---------------------------------------------------------------------------

def feature_attributions(x: np.ndarray, y: np.ndarray, groups: np.ndarray, feat_cols: list[str],
                         classes: list[str], cfg: TrapsigConfig, seed: int) -> dict[str, Any]:
    """Feature-level attributions for Model A, fitted on the pool for explanation only.

    Logistic regression gets exact linear SHAP. The tree models get TreeSHAP when ``shap`` is
    installed; otherwise the entry says so rather than substituting a different method silently.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    out: dict[str, Any] = {"note": "fitted on the full pool for attribution only",
                           "shap_installed": shap_available(), "models": {}}
    xi = SimpleImputer(strategy="median").fit_transform(x)
    n_bg = min(cfg.explain.shap_background, xi.shape[0])
    bg_idx = np.linspace(0, xi.shape[0] - 1, n_bg).astype(int)

    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, C=1.0))]).fit(xi, y)
    scaled = pipe.named_steps["scale"].transform(xi)
    clf = pipe.named_steps["clf"]
    clf_classes = [str(v) for v in getattr(clf, "classes_", classes)]
    coef = np.asarray(clf.coef_, dtype=float)
    if len(clf_classes) == 2 and coef.shape[0] == 1:
        # Binary logistic regression stores ONE coefficient row: the log-odds of class 1 over
        # class 0. The per-class SHAP values are that row and its negation, which is exact, not
        # an approximation. Indexing coef by class position would go out of bounds here.
        coef = np.stack([-coef[0], coef[0]])
    # Exact linear SHAP: coefficient times the centred feature value, per class column.
    centre = scaled[bg_idx].mean(axis=0)
    # (n_samples, 1, n_features) x (1, n_classes, n_features) -> (n_samples, n_classes, n_features)
    phi = (scaled - centre)[:, None, :] * coef[None, :, :]
    out["models"]["logreg"] = {
        "method": "exact linear SHAP (coefficients x centred, standardised features)",
        "binary_coefficient_expanded": bool(len(clf_classes) == 2
                                            and np.asarray(clf.coef_).shape[0] == 1),
        "base_value": np.asarray(clf.intercept_, dtype=float).tolist(),
        "mean_abs_phi": {c: float(np.mean(np.abs(phi[:, j, :])))
                         for j, c in enumerate(clf_classes) if j < phi.shape[1]},
        "phi_shape": list(phi.shape),
    }
    top: dict[str, list[tuple[str, float]]] = {}
    for j, c in enumerate(clf_classes):
        if j >= phi.shape[1]:
            break
        m = np.mean(np.abs(phi[:, j, :]), axis=0)
        order = np.argsort(m)[::-1][:15]
        top[c] = [(feat_cols[i], float(m[i])) for i in order]
    out["models"]["logreg"]["top_features_by_class"] = top

    for name, make in (("rf", lambda: __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"])
                        .RandomForestClassifier(n_estimators=cfg.training.rf_n_estimators,
                                                random_state=seed, n_jobs=1)),
                       ("hgb", lambda: __import__("sklearn.ensemble",
                                                 fromlist=["HistGradientBoostingClassifier"])
                        .HistGradientBoostingClassifier(max_iter=cfg.training.hgb_max_iter,
                                                        random_state=seed))):
        if not shap_available():
            out["models"][name] = {"method": "TreeSHAP", "status": "not_applicable",
                                   "reason": "the optional 'shap' package is not installed; "
                                             "install it with: pip install shap"}
            continue
        m = make().fit(xi, y)
        ex = _shap.TreeExplainer(m, data=xi[bg_idx], feature_perturbation="interventional")
        sv = ex.shap_values(xi[:min(cfg.explain.shap_background * 4, xi.shape[0])])
        arr = np.asarray(sv)
        if arr.ndim == 2:                       # binary: one array, sign separates the classes
            arr = np.stack([-arr, arr], axis=-1)
        mean_abs = np.mean(np.abs(arr), axis=0)
        m_classes = [str(v) for v in getattr(m, "classes_", classes)]
        if mean_abs.ndim == 1:
            mean_abs = np.stack([mean_abs] * len(m_classes))
        out["models"][name] = {
            "method": "TreeSHAP (interventional, background = a spread of pool rows)",
            "n_background": n_bg,
            "top_features_by_class": {
                c: [(feat_cols[i], float(mean_abs[j, i]))
                    for i in np.argsort(mean_abs[j])[::-1][:15]]
                for j, c in enumerate(m_classes) if j < mean_abs.shape[0]},
        }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_explainability(cfg: TrapsigConfig, run_dir: Path, train_summary: dict[str, Any],
                       ws: WindowSet | None, x: np.ndarray | None, y: np.ndarray | None,
                       groups: np.ndarray | None, feat_cols: list[str], classes: list[str],
                       fold_ids: np.ndarray, file_order: list[str], say: Say = None) -> dict[str, Any]:
    """Produce every attribution the configuration asks for and write it into ``run_dir``."""
    tell = say or (lambda m: None)
    e = cfg.explain
    out: dict[str, Any] = {"enabled": e.enabled, "torch_available": torch_available(),
                           "shap_available": shap_available(), "methods": {}}
    if not e.enabled:
        out["status"] = "disabled in configuration"
        return out
    out["architecture_condition"] = (
        "conv -> global average pooling -> single linear head, so Grad-CAM is mathematically "
        "valid (design R22); attributions are labelled gradcam, not CAM")

    have_ws = ws is not None and ws.x.shape[0] > 0
    em = (fit_explainer_model(ws, classes, cfg, tell)
          if (e.enabled and ws is not None and ws.x.shape[0] > 0) else None)
    n_expl = min(e.n_explained_windows, ws.x.shape[0]) if have_ws and ws is not None else 0

    if em is not None and have_ws and ws is not None and n_expl:
        rng = np.random.default_rng(cfg.run.seed + 5)
        pick = rng.choice(ws.x.shape[0], size=n_expl, replace=False)
        idx = {c: i for i, c in enumerate(classes)}
        rows: list[dict[str, Any]] = []
        for w_i in pick:
            lab = str(ws.class_label[w_i])
            ci = idx.get(lab, 0)
            xw = ws.x[w_i]
            rec: dict[str, Any] = {"window_index": int(w_i), "sample_id": str(ws.sample_id[w_i]),
                                   "class": lab, "predicted_class_index": ci,
                                   "onset_s": float(ws.onset_s[w_i])}
            if e.grad_cam_1d:
                rec["gradcam"] = gradcam_1d(em, xw, ci)
            if e.integrated_gradients:
                rec["integrated_gradients"] = integrated_gradients_1d(
                    em, xw, ci, e.integrated_gradients_steps)
            if e.occlusion:
                rec["occlusion"] = occlusion_1d(em, xw, ci, e.occlusion_segments)
            rows.append(rec)
        arrays: dict[str, Any] = {
            "window_index": np.array([r["window_index"] for r in rows]),
            "sample_id": np.array([r["sample_id"] for r in rows], dtype=object)}
        for m in ("gradcam", "integrated_gradients", "occlusion"):
            if m in rows[0]:
                arrays[m] = np.stack([np.asarray(r[m], dtype=float) for r in rows])
        np.savez_compressed(run_dir / "attributions.npz", **arrays)
        _write_attribution_summary(rows, run_dir / "attributions_summary.csv", em.length,
                                   e.occlusion_segments)
        out["methods"]["temporal"] = {
            "n_windows_explained": n_expl,
            "computed": [m for m in ("gradcam", "integrated_gradients", "occlusion") if m in rows[0]],
            "window_length_samples": em.length,
            "explanation_model": em.note,
            "n_parameters": em.n_parameters,
            "files": {"attributions": "attributions.npz",
                      "summary": "attributions_summary.csv"},
        }
        tell(f"explain: {n_expl} windows attributed")

        if e.shortcut_probes:
            shift = max(1, em.length // 8)
            ts = [time_shift_null(em, ws.x[int(w)], idx.get(str(ws.class_label[int(w)]), 0), shift)
                  for w in pick[:min(5, n_expl)]]
            out["methods"]["time_shift_null"] = {
                "shift_samples": shift, "per_window": ts,
                "moves_with_event_fraction": float(np.mean([t["moves_with_event"] for t in ts]))
                if ts else None,
                "interpretation": ("attribution tracks the shifted event" if ts and
                                   np.mean([t["moves_with_event"] for t in ts]) >= 0.6 else
                                   "attribution does not reliably track the shifted event: the model "
                                   "may be keyed to window position"),
            }

    if e.shortcut_probes and have_ws and ws is not None:
        tell("explain: running the baseline-only shortcut probe (this is a full CNN CV)")
        out["methods"]["baseline_only_probe"] = baseline_only_probe(
            ws, cfg, classes, fold_ids, file_order, tell)

    if e.shap_features and x is not None and y is not None and groups is not None and feat_cols:
        out["methods"]["features"] = feature_attributions(
            x, y, groups, feat_cols, classes, cfg, cfg.run.seed)
        tell("explain: feature attributions computed")

    write_json_exclusive(run_dir / "explainability.json", _trim(out))
    return out


def _trim(obj: Any, depth: int = 0) -> Any:
    """Drop large arrays before writing JSON; the .npz carries them."""
    if isinstance(obj, dict):
        return {k: ("[omitted: see attributions.npz]" if k in ("gradcam", "integrated_gradients",
                                                               "occlusion", "phi") else _trim(v, depth + 1))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if depth >= 3 and len(obj) > 20:
            return f"[{len(obj)} values]"
        return [_trim(v, depth + 1) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _write_attribution_summary(rows: list[dict[str, Any]], path: Path, length: int,
                               n_segments: int) -> None:
    """Segment-level means, so the CSV stays readable at any window length."""
    edges = np.linspace(0, length, n_segments + 1).astype(int)
    methods = [m for m in ("gradcam", "integrated_gradients", "occlusion") if m in rows[0]]
    buf = io.StringIO()
    fields = ["window_index", "sample_id", "class", "method", "segment", "t_start_frac", "t_end_frac",
              "mean_attribution"]
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for r in rows:
        for m in methods:
            a = np.asarray(r[m], dtype=float)
            for si, (a0, b0) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
                if b0 <= a0:
                    continue
                w.writerow({"window_index": r["window_index"], "sample_id": r["sample_id"],
                            "class": r["class"], "method": m, "segment": si,
                            "t_start_frac": round(a0 / length, 4), "t_end_frac": round(b0 / length, 4),
                            "mean_attribution": float(np.mean(a[a0:b0]))})
    write_text_exclusive(path, buf.getvalue())
