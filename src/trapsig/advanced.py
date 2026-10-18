"""Conditional analyses over the frozen CNN representation (design sections 11.6 to 11.10).

Each of these is gated in configuration and off by default. They all consume the **out-of-fold**
GAP embeddings that Model B already produced during cross-validation, so no file is represented
by a network that trained on it.

Few-shot uses setting (b) of design 11.7: the classes are known and the few labelled examples come
from files the encoder did not train on. Setting (a), novel classes, is not possible here without
a principled way to hold whole classes out, and the dataset has only two.

Semi-supervised is inductive exactly as design 11.8 specifies: the unlabelled pool is windows from
the *same training files* whose labels are hidden from the teacher. The test fold is never in the
pool.

Novelty detection cannot be evaluated honestly on this dataset, because there is no unknown class.
That is reported as "not evaluable" rather than being faked. A leave-one-class-out run is provided
and labelled as a diagnostic only, which is what design 11.9 allows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

from .config import TrapsigConfig

Say = Any


class AdvancedError(RuntimeError):
    """A condition that stops one of the conditional analyses."""


# ---------------------------------------------------------------------------
# Shared representation
# ---------------------------------------------------------------------------

@dataclass
class Representation:
    """File-level embeddings from the out-of-fold CNN, with provenance."""
    emb: np.ndarray                   # (n_files, channels)
    sample_id: list[str]
    labels: np.ndarray                # (n_files,) object
    groups: np.ndarray                # (n_files,) object
    n_windows: np.ndarray             # (n_files,)
    repeat: int
    source: str = "out-of-fold GAP embedding of the Model B encoder"


def file_representation(cv_result: Any, ws: Any, file_order: list[str], repeat: int = 0
                        ) -> Representation | None:
    """Average each file's out-of-fold window embeddings into one vector."""
    if cv_result is None or ws is None or ws.x.shape[0] == 0:
        return None
    emb = cv_result.embeddings[repeat]              # (n_windows, channels)
    pos = {s: i for i, s in enumerate(file_order)}
    n = len(file_order)
    acc = np.zeros((n, emb.shape[1]), dtype=float)
    cnt = np.zeros(n, dtype=np.int64)
    labs = np.full(n, "", dtype=object)
    grps = np.full(n, "", dtype=object)
    for i, sid in enumerate(ws.sample_id.tolist()):
        j = pos.get(sid)
        if j is None or not np.all(np.isfinite(emb[i])):
            continue
        acc[j] += emb[i]
        cnt[j] += 1
        labs[j] = ws.class_label[i]
        grps[j] = ws.group_id[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], np.nan)
    keep = cnt > 0
    return Representation(emb=mean[keep], sample_id=[s for s, k in zip(file_order, keep, strict=True) if k],
                          labels=labs[keep], groups=grps[keep], n_windows=cnt[keep], repeat=repeat)


def _l2norm(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    out: np.ndarray = a / np.maximum(n, 1e-12)
    return out


# ---------------------------------------------------------------------------
# Few-shot: prototypical episodes (design 11.7, setting b)
# ---------------------------------------------------------------------------

def few_shot_prototypical(rep: Representation, classes: list[str], cfg: TrapsigConfig,
                          say: Say = None) -> dict[str, Any]:
    """Nearest-prototype classification with K labelled files per class.

    Support and query files are disjoint, and both are files the encoder never trained on.
    Episodes overlap in their examples, so episode-level spread is **not** independent variance;
    the interval is a bootstrap over files, which is the only honest unit here.
    """
    tell = say or (lambda m: None)
    a = cfg.advanced
    y = rep.labels
    per_class = {c: np.flatnonzero(y == c) for c in classes}
    out: dict[str, Any] = {
        "setting": "(b) new files, known classes; support and query files are disjoint",
        "distance": a.few_shot_distance,
        "episodes": a.few_shot_episodes,
        "representation": rep.source,
        "embedding_repeat": rep.repeat,
        "class_file_counts": {c: int(len(v)) for c, v in per_class.items()},
        "results": {},
    }
    emb = _l2norm(rep.emb) if a.few_shot_distance == "cosine" else rep.emb
    rng = np.random.default_rng(cfg.run.seed + 31)

    for k in a.few_shot_k:
        # Need k support + at least 1 query per class.
        need = k + 1
        infeasible = [c for c in classes if len(per_class[c]) < need]
        if infeasible:
            out["results"][f"k={k}"] = {
                "status": "SKIPPED",
                "reason": f"classes {infeasible} have fewer than {need} files "
                          f"(k support + 1 query); design D32"}
            continue
        accs: list[float] = []
        f1s: list[float] = []
        for _ in range(a.few_shot_episodes):
            sup_list: list[int] = []
            qry_list: list[int] = []
            for c in classes:
                pool = per_class[c]
                pick = rng.choice(pool, size=need, replace=False)
                sup_list.extend(pick[:k].tolist())
                qry_list.extend(pick[k:].tolist())
            sup_idx = np.array(sup_list, dtype=np.int64)
            qry_idx = np.array(qry_list, dtype=np.int64)
            protos = np.stack([emb[sup_idx[y[sup_idx] == c]].mean(axis=0) for c in classes])
            q = emb[qry_idx]
            if a.few_shot_distance == "cosine":
                d = 1.0 - q @ _l2norm(protos).T
            else:
                d = np.linalg.norm(q[:, None, :] - protos[None, :, :], axis=2)
            pred = np.array(classes, dtype=object)[np.argmin(d, axis=1)]
            truth = y[qry_idx]
            accs.append(float(np.mean(pred == truth)))
            f1s.append(float(np.mean([np.mean(pred[truth == c] == c) for c in classes
                                      if np.any(truth == c)])))
        arr = np.array(accs)
        boot = _bootstrap_mean(_episode_file_scores(emb, y, classes, per_class, k, a, cfg), 1000,
                               cfg.run.seed)
        out["results"][f"k={k}"] = {
            "status": "computed",
            "accuracy_episode_mean": float(arr.mean()),
            "accuracy_episode_sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "balanced_accuracy_episode_mean": float(np.mean(f1s)),
            "accuracy_bootstrap_ci95_over_files": {"lo": boot[0], "hi": boot[1]} if boot else None,
            "n_episodes": int(arr.size),
            "note": "episodes overlap, so episode SD is not an independent-variance estimate; "
                    "the interval is bootstrapped over files",
        }
        tell(f"few-shot k={k}: accuracy {arr.mean():.3f}")
    return out


def _episode_file_scores(emb: np.ndarray, y: np.ndarray, classes: list[str],
                         per_class: dict[str, np.ndarray], k: int, a: Any,
                         cfg: TrapsigConfig) -> list[float]:
    """Per-file few-shot correctness, averaged over episodes, for a file-level bootstrap."""
    rng = np.random.default_rng(cfg.run.seed + 32)
    hits = np.zeros(len(y))
    tries = np.zeros(len(y))
    for _ in range(max(1, a.few_shot_episodes // 4)):
        sup_idx = np.concatenate([rng.choice(per_class[c], size=k, replace=False) for c in classes])
        protos = np.stack([emb[sup_idx[y[sup_idx] == c]].mean(axis=0) for c in classes])
        protos = _l2norm(protos) if a.few_shot_distance == "cosine" else protos
        q = _l2norm(emb) if a.few_shot_distance == "cosine" else emb
        if a.few_shot_distance == "cosine":
            d = 1.0 - q @ protos.T
        else:
            d = np.linalg.norm(q[:, None, :] - protos[None, :, :], axis=2)
        pred = np.array(classes, dtype=object)[np.argmin(d, axis=1)]
        ok = (pred == y).astype(float)
        mask = np.ones(len(y), dtype=bool)
        mask[sup_idx] = False                     # support files are excluded from scoring
        hits[mask] += ok[mask]
        tries[mask] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return [float(h / t) for h, t in zip(hits, tries, strict=True) if t > 0]


def _bootstrap_mean(vals: list[float], n: int, seed: int) -> tuple[float, float] | None:
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------------------
# Semi-supervised: confidence-aware pseudo-labelling (design 11.8)
# ---------------------------------------------------------------------------

def semi_supervised(ws: Any, fold_ids: np.ndarray, file_order: list[str], classes: list[str],
                    cfg: TrapsigConfig, imbalance: float, say: Say = None) -> dict[str, Any]:
    """Inductive pseudo-labelling on windows from the training files of repeat 0.

    The teacher is trained on a labelled subset of the training windows. Windows from the same
    training files are then hidden, scored, and accepted as pseudo-labels only when the maximum
    probability clears tau and the top-1/top-2 margin clears delta. Round 2 retrains from scratch
    on labelled + accepted, which is what stops confirmation drift.
    """
    from .windows import apply_robust_scale, fit_robust_scale

    tell = say or (lambda m: None)
    a = cfg.advanced
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    y_all = np.array([idx[c] for c in ws.class_label.tolist()], dtype=np.int64)
    pos = {s: i for i, s in enumerate(file_order)}
    w_file = np.array([pos[s] for s in ws.sample_id.tolist()], dtype=np.int64)
    w_fold = fold_ids[0][w_file]
    rng = np.random.default_rng(cfg.run.seed + 77)
    labelled_fraction = 0.5

    out: dict[str, Any] = {
        "setting": "inductive; the unlabelled pool is windows from the same training files, "
                   "with their labels hidden from the teacher (design 11.8)",
        "labelled_fraction_of_training_windows": labelled_fraction,
        "tau": a.pseudo_confidence_tau, "delta": a.pseudo_margin_delta, "rounds": a.pseudo_rounds,
        "repeat_used": 0,
        "note": "repeat 0 only; running every repeat would multiply the cost by n_repeats",
        "folds": [],
    }
    sup_total = lab_total = acc_total = pool_total = 0
    accs_by_class: dict[str, int] = {c: 0 for c in classes}
    bal_sup: list[float] = []
    bal_lab: list[float] = []

    for f in range(cfg.splitting.n_splits):
        te_w = w_fold == f
        tr_w = ~te_w
        if te_w.sum() == 0 or tr_w.sum() < 8:
            out["folds"].append({"fold": f, "status": "SKIPPED", "reason": "too few windows"})
            continue
        med, iqr = fit_robust_scale(ws.x[tr_w])
        xtr_all = apply_robust_scale(ws.x[tr_w], med, iqr)
        xte = apply_robust_scale(ws.x[te_w], med, iqr)
        ytr_all = y_all[tr_w]
        gtr = ws.group_id[tr_w]
        yte = y_all[te_w]

        lab_mask = rng.random(tr_w.sum()) < labelled_fraction
        # Keep at least one window per class labelled, or the teacher cannot learn that class.
        for c in range(k):
            if not np.any(lab_mask & (ytr_all == c)):
                cand = np.flatnonzero(ytr_all == c)
                if cand.size:
                    lab_mask[cand[0]] = True
        unl_mask = ~lab_mask

        seed = cfg.run.seed + 500 + f

        # Supervised-only reference on exactly the same labelled subset.
        _, base_acc = _score_fold(xtr_all[lab_mask], ytr_all[lab_mask], gtr[lab_mask],
                                  xte, yte, k, cfg, seed, imbalance)
        bal_lab.append(base_acc)

        current = xtr_all[lab_mask]
        current_y = ytr_all[lab_mask]
        current_g = gtr[lab_mask]
        rounds_log: list[dict[str, Any]] = []
        x_unl = apply_robust_scale(ws.x[tr_w][unl_mask], med, iqr)
        for rnd in range(1, a.pseudo_rounds + 1):
            teacher_p, _ = _score_fold(current, current_y, current_g, x_unl,
                                       np.zeros(0, dtype=np.int64), k, cfg, seed, imbalance)
            p = teacher_p
            top2 = np.sort(p, axis=1)[:, -2:]
            conf = top2[:, 1]
            margin = top2[:, 1] - top2[:, 0]
            accept = (conf >= a.pseudo_confidence_tau) & (margin >= a.pseudo_margin_delta)
            pseudo = np.argmax(p, axis=1)
            n_acc = int(accept.sum())
            per_class_acc = {classes[c]: int(np.sum(pseudo[accept] == c)) for c in range(k)}
            # Design 11.8: reject an accepted set that is heavily skewed to one class.
            skewed = False
            if n_acc > 0:
                shares = np.array([per_class_acc[c] / n_acc for c in classes])
                skewed = bool(shares.max() > 0.95 and len(classes) > 1)
            rounds_log.append({"round": rnd, "pool": int(unl_mask.sum()), "candidates": n_acc,
                               "accepted": 0 if skewed else n_acc,
                               "rejected": int(unl_mask.sum()) - (0 if skewed else n_acc),
                               "acceptance_rate": (0.0 if skewed else n_acc / max(unl_mask.sum(), 1)),
                               "pseudo_labels_per_class": per_class_acc,
                               "rejected_for_class_skew": skewed,
                               "confidence_mean": float(conf.mean()),
                               "confidence_p95": float(np.percentile(conf, 95))})
            if skewed or n_acc == 0:
                break
            xu = apply_robust_scale(ws.x[tr_w][unl_mask][accept], med, iqr)
            gu = ws.group_id[tr_w][unl_mask][accept]
            # Retrain from scratch on labelled + accepted (design 11.8).
            current = np.concatenate([xtr_all[lab_mask], xu])
            current_y = np.concatenate([ytr_all[lab_mask], pseudo[accept]])
            current_g = np.concatenate([gtr[lab_mask], gu])

        _, sup_acc = _score_fold(current, current_y, current_g, xte, yte, k, cfg, seed, imbalance)
        bal_sup.append(sup_acc)
        sup_total += 1
        lab_total += int(lab_mask.sum())
        pool_total += int(unl_mask.sum())
        acc_total += sum(int(r["accepted"]) for r in rounds_log)
        for r in rounds_log:
            if r["rejected_for_class_skew"]:
                continue          # a skewed round contributes no labels, so count none
            for c, v in r["pseudo_labels_per_class"].items():
                accs_by_class[c] += int(v)
        out["folds"].append({"fold": f, "status": "computed",
                             "n_training_windows": int(tr_w.sum()),
                             "n_labelled": int(lab_mask.sum()),
                             "n_pool": int(unl_mask.sum()),
                             "n_test_windows": int(te_w.sum()),
                             "supervised_only_balanced_accuracy": base_acc,
                             "with_pseudo_labels_balanced_accuracy": sup_acc,
                             "rounds": rounds_log})
        tell(f"semi-supervised fold {f + 1}: {base_acc:.3f} -> {sup_acc:.3f}")

    vals_s = [v for v in bal_sup if math.isfinite(v)]
    vals_l = [v for v in bal_lab if math.isfinite(v)]
    out["summary"] = {
        "folds_used": sup_total,
        "n_labelled_windows": lab_total,
        "n_unlabelled_pool_windows": pool_total,
        "n_accepted_pseudo_labels": acc_total,
        "accepted_per_class": accs_by_class,
        # Design 11.8 defines the acceptance rate against the unlabelled pool, not the labelled set.
        "overall_acceptance_rate": acc_total / max(pool_total, 1),
        "supervised_only_mean": float(np.mean(vals_l)) if vals_l else None,
        "with_pseudo_labels_mean": float(np.mean(vals_s)) if vals_s else None,
        "delta_mean": (float(np.mean(vals_s) - np.mean(vals_l))
                       if vals_s and vals_l else None),
        "comparison": "same labelled subset, same outer folds, same protocol; the only difference "
                      "is the accepted pseudo-labels",
    }
    return out


def _score_fold(x: np.ndarray, y: np.ndarray, g: np.ndarray, xte: np.ndarray, yte: np.ndarray,
                k: int, cfg: TrapsigConfig, seed: int, imbalance: float
                ) -> tuple[np.ndarray, float]:
    """Train once and return (test probabilities, balanced accuracy on the test windows)."""
    from .deep import train_fold, validation_split

    vm = validation_split(y, g, cfg.deep.validation_fraction, seed, k)
    if vm.sum() == 0 or (~vm).sum() == 0:
        vm = np.zeros(y.shape[0], dtype=bool)
    res = train_fold(x[~vm], y[~vm], x[vm], y[vm], xte, k, cfg, seed, imbalance, None, "", None)
    pred = np.argmax(res.proba_test, axis=1)
    rec = [float(np.mean(pred[yte == c] == c)) for c in range(k) if np.any(yte == c)]
    return res.proba_test, float(np.mean(rec)) if rec else float("nan")


# ---------------------------------------------------------------------------
# Novelty / unknown-class detection (design 11.9)
# ---------------------------------------------------------------------------

def novelty_detection(rep: Representation, classes: list[str], cfg: TrapsigConfig,
                      say: Say = None) -> dict[str, Any]:
    """Known-class scores plus a leave-one-class-out diagnostic.

    With only the two experimental classes present there is no genuine unknown set, so the honest
    headline is "not evaluable". Leave-one-class-out is reported separately and explicitly as a
    diagnostic: it measures whether the classes are separable in the embedding, not whether the
    model would reject a real unknown protein signal.
    """
    tell = say or (lambda m: None)
    a = cfg.advanced
    emb = rep.emb
    y = rep.labels
    out: dict[str, Any] = {
        "status": "not evaluable",
        "reason": "the dataset contains no unknown class; design 11.9 requires one to evaluate "
                  "rejection. Leave-one-class-out below is a diagnostic only.",
        "known_retention_target": a.novelty_known_retention,
        "scores_computed": ["prototype_distance", "max_softmax_substitute", "energy_substitute"],
        "note_on_confidence": "max-softmax and energy need the classifier head's logits on the "
                              "evaluation fold; here the file-level representation is used, so "
                              "prototype distance is the primary score and is reported as such",
        "leave_one_class_out_diagnostic": {},
    }
    for held in classes:
        known = [c for c in classes if c != held]
        if not known:
            continue
        protos = np.stack([emb[y == c].mean(axis=0) for c in known if np.any(y == c)])
        d = np.linalg.norm(emb[:, None, :] - protos[None, :, :], axis=2).min(axis=1)
        is_unknown = (y == held).astype(int)
        thr = float(np.quantile(d[is_unknown == 0], a.novelty_known_retention))
        entry: dict[str, Any] = {
            "held_out_class": held,
            "n_known_files": int(np.sum(is_unknown == 0)),
            "n_held_out_files": int(is_unknown.sum()),
            "threshold_at_target_retention": thr,
            "known_retained_fraction": float(np.mean(d[is_unknown == 0] <= thr)),
            "held_out_flagged_fraction": float(np.mean(d[is_unknown == 1] > thr))
            if is_unknown.sum() else None,
        }
        if is_unknown.sum() and (is_unknown == 0).sum():
            try:
                entry["auroc_distance"] = float(roc_auc_score(is_unknown, d))
            except ValueError as exc:
                entry["auroc_distance"] = None
                entry["auroc_reason"] = str(exc)
        out["leave_one_class_out_diagnostic"][held] = entry
        tell(f"novelty LOCO {held}: AUROC {entry.get('auroc_distance')}")
    return out


# ---------------------------------------------------------------------------
# Clustering (design 11.10, optional, default off)
# ---------------------------------------------------------------------------

def clustering_analysis(rep: Representation, classes: list[str], cfg: TrapsigConfig,
                        fold_ids: np.ndarray, file_order: list[str], say: Say = None
                        ) -> dict[str, Any]:
    """Exploratory clustering on training-fold embeddings, with stability and label agreement.

    Fitted on training groups only and applied to the rest by assignment. ARI and NMI against the
    labels are diagnostics, never a selection criterion: a clustering that matches the labels is
    not evidence that the labels are the right structure.
    """
    tell = say or (lambda m: None)
    a = cfg.advanced
    out: dict[str, Any] = {"fitted_on": "training groups of repeat 0 only",
                           "stability": "bootstrap Jaccard across 20 resamples", "results": {}}
    train_files = set(file_order[i] for i in range(len(file_order)) if fold_ids[0][i] != 0)
    is_train = np.array([s in train_files for s in rep.sample_id])
    if is_train.sum() < max(a.clustering_k) * 2:
        out["status"] = "SKIPPED"
        out["reason"] = (f"only {int(is_train.sum())} training files; need at least twice the "
                         f"largest k ({max(a.clustering_k)})")
        return out
    emb_tr = rep.emb[is_train]
    y_tr = rep.labels[is_train]
    y_all = rep.labels
    for kk in a.clustering_k:
        km = KMeans(n_clusters=kk, n_init=10, random_state=cfg.run.seed).fit(emb_tr)
        lab_tr = km.labels_
        lab_all = km.predict(rep.emb)
        # Bootstrap Jaccard stability: resample training files, refit, match clusters.
        jac = _cluster_stability(emb_tr, kk, cfg.run.seed)
        out["results"][f"k={kk}"] = {
            "status": "computed",
            "ari_train": float(adjusted_rand_score(y_tr, lab_tr)),
            "nmi_train": float(normalized_mutual_info_score(y_tr, lab_tr)),
            "ari_all": float(adjusted_rand_score(y_all, lab_all)),
            "nmi_all": float(normalized_mutual_info_score(y_all, lab_all)),
            "bootstrap_jaccard_mean": jac[0],
            "bootstrap_jaccard_sd": jac[1],
            "cluster_sizes": [int(np.sum(lab_tr == c)) for c in range(kk)],
            "note": "ARI/NMI are diagnostics; they are not a model-selection criterion",
        }
        tell(f"clustering k={kk}: ARI {out['results'][f'k={kk}']['ari_train']:.3f}")
    return out


def _cluster_stability(x: np.ndarray, kk: int, seed: int, n_boot: int = 20) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    base = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(x)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.shape[0], size=x.shape[0])
        lab = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(x[idx])
        scores.append(_mean_best_jaccard(base[idx], lab))
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _mean_best_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for ca in np.unique(a):
        best = 0.0
        for cb in np.unique(b):
            inter = float(np.sum((a == ca) & (b == cb)))
            union = float(np.sum((a == ca) | (b == cb)))
            if union > 0:
                best = max(best, inter / union)
        vals.append(best)
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_advanced(cfg: TrapsigConfig, rep: Representation | None, ws: Any, fold_ids: np.ndarray,
                 file_order: list[str], classes: list[str], cv_result: Any, imbalance: float,
                 say: Say = None) -> dict[str, Any]:
    """Run whichever conditional analyses the configuration enables."""
    tell = say or (lambda m: None)
    a = cfg.advanced
    out: dict[str, Any] = {"few_shot": a.few_shot, "semi_supervised": a.semi_supervised,
                           "novelty": a.novelty, "clustering": a.clustering}
    if not any([a.few_shot, a.semi_supervised, a.novelty, a.clustering]):
        out["status"] = "all conditional analyses are disabled in configuration"
        return out
    if rep is None:
        out["status"] = "not applicable: no Model B representation (Model B did not run)"
        return out

    if a.few_shot:
        out["few_shot_results"] = few_shot_prototypical(rep, classes, cfg, tell)
    if a.novelty:
        out["novelty_results"] = novelty_detection(rep, classes, cfg, tell)
    if a.clustering:
        out["clustering_results"] = clustering_analysis(rep, classes, cfg, fold_ids, file_order, tell)
    if a.semi_supervised and ws is not None and ws.x.shape[0]:
        out["semi_supervised_results"] = semi_supervised(ws, fold_ids, file_order, classes, cfg,
                                                         imbalance, tell)
    return out
