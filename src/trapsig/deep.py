"""Model B: compact 1D residual CNN with dilated convolutions (design sections 11.3 and 11.5).

Architecture
------------
``stem (strided conv) -> n_blocks residual dilated conv blocks -> global average pooling -> linear``.
The head is a single linear layer on the GAP of the final convolutional map, which is the
condition under which Grad-CAM attribution is mathematically valid (design R22 and 11.12).
Parameter count is logged per fold and targets 1e4-1e5, as the design requires; a larger
model is not used without evidence.

Training protocol
-----------------
AdamW with linear warmup then cosine decay. Early stopping on the **validation** split,
which is carved from the outer training fold by file group, never from the test fold.
Checkpoint selection on a metric fixed in config before running. Temperature scaling is
fitted on the validation split only and then applied to the test fold. Seeds are set for
Python, NumPy, and PyTorch, and deterministic algorithms are enabled where they are
available on CPU.

Class imbalance follows design 11.4: cross-entropy is class-weighted only when the
imbalance ratio is at or above the threshold; below it, no correction is applied.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from .config import TrapsigConfig
from .windows import WindowSet, apply_robust_scale, fit_robust_scale

# Design section 11.4: the same threshold as Model A, so exactly one imbalance rule is active.
IMBALANCE_WEIGHT_THRESHOLD = 1.5

Say = Callable[[str], None]


class DeepError(RuntimeError):
    """A condition that stops deep training (no windows, degenerate fold, missing dependency)."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResidualDilatedBlock(nn.Module):
    """Two dilated convolutions with a residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.bn1(self.conv1(x))))
        out: torch.Tensor = self.act(x + self.bn2(self.conv2(h)))
        return out


class CNN1D(nn.Module):
    """Compact 1D CNN for baseline-subtracted event windows."""

    def __init__(self, length: int, n_classes: int, cfg: TrapsigConfig) -> None:
        super().__init__()
        d = cfg.deep
        self.stem = nn.Sequential(
            nn.Conv1d(1, d.channels, kernel_size=7, stride=d.stem_stride, padding=3),
            nn.BatchNorm1d(d.channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            ResidualDilatedBlock(d.channels, d.kernel_size, dilation=2 ** i, dropout=d.dropout)
            for i in range(d.n_blocks)
        ])
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Linear(d.channels, n_classes)
        self.length = length
        self.n_classes = n_classes
        # Receptive field in window samples, for the report.
        self.receptive_field = self._receptive_field(cfg)

    def _receptive_field(self, cfg: TrapsigConfig) -> int:
        d = cfg.deep
        rf, stride = 7, d.stem_stride
        for i in range(d.n_blocks):
            dil = 2 ** i
            for _ in range(2):
                rf += (d.kernel_size - 1) * dil * stride
        return int(rf)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Final convolutional map before pooling: (batch, channels, time)."""
        h: torch.Tensor = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = h.mean(dim=-1)              # global average pooling (the CAM condition)
        out: torch.Tensor = self.head(pooled)
        return out

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """GAP representation, used by few-shot, novelty, and clustering analyses."""
        out: torch.Tensor = self.features(x).mean(dim=-1)
        return out


def build_model(length: int, n_classes: int, cfg: TrapsigConfig) -> CNN1D:
    return CNN1D(length, n_classes, cfg)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# Determinism and device
# ---------------------------------------------------------------------------

def seed_everything(seed: int, threads: int = 0) -> str:
    """Seed Python-independent RNGs used here and return a note on determinism."""
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    note = "torch.manual_seed set"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        note += "; deterministic algorithms enabled (warn_only)"
    except Exception as exc:  # pragma: no cover - depends on the torch build
        note += f"; deterministic algorithms unavailable ({type(exc).__name__})"
    if threads > 0:
        torch.set_num_threads(threads)
        note += f"; threads={threads}"
    return note


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Metrics used inside training
# ---------------------------------------------------------------------------

def _macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    f1s = []
    for c in range(k):
        tp = float(np.sum((pred == c) & (y == c)))
        fp = float(np.sum((pred == c) & (y != c)))
        fn = float(np.sum((pred != c) & (y == c)))
        if tp == 0 and (fp == 0 or fn == 0):
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _balanced_acc(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    recs = [float(np.mean(pred[y == c] == c)) for c in range(k) if np.any(y == c)]
    return float(np.mean(recs)) if recs else 0.0


def select_value(metric: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray, k: int) -> float:
    """Value of the checkpoint-selection metric. Higher is better for all three."""
    if metric == "macro_f1":
        return _macro_f1(y, pred, k)
    if metric == "balanced_accuracy":
        return _balanced_acc(y, pred, k)
    if metric == "log_loss":
        p = np.clip(proba, 1e-12, 1.0)
        return float(-np.mean(np.log(p[np.arange(y.size), y])))
    raise DeepError(f"unknown select_metric {metric!r}")


# ---------------------------------------------------------------------------
# Metric learning: the auxiliary loss and the prototype classifier
# ---------------------------------------------------------------------------

def metric_loss(emb: torch.Tensor, y: torch.Tensor, kind: str, temperature: float) -> torch.Tensor:
    """Auxiliary loss that shapes the embedding, not just the decision boundary.

    ``supcon`` is supervised contrastive (Khosla et al., 2020): every embedding is pulled towards
    the other embeddings of its own class and pushed away from the rest of the batch.
    ``prototypical`` computes one prototype per class inside the batch and asks each embedding to
    be closest to its own. Both act on the L2-normalised GAP embedding, which is the same vector
    the prototypical head, few-shot, novelty, and clustering consume later, so one representation
    serves every strategy instead of each of them re-learning one.
    """
    b = emb.shape[0]
    if kind == "none" or b < 2:
        return torch.zeros((), dtype=emb.dtype, device=emb.device)
    z = nn.functional.normalize(emb, dim=1)
    if kind == "prototypical":
        classes = torch.unique(y)
        if classes.numel() < 2:
            return torch.zeros((), dtype=emb.dtype, device=emb.device)
        protos = torch.stack([z[y == c].mean(dim=0) for c in classes])
        protos = nn.functional.normalize(protos, dim=1)
        d = 1.0 - z @ protos.T                       # cosine distance to each class prototype
        target = torch.searchsorted(classes.contiguous(), y.contiguous())
        return nn.functional.cross_entropy(-d / max(temperature, 1e-6), target)
    # supervised contrastive
    sim = z @ z.T / max(temperature, 1e-6)
    eye = torch.eye(b, dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(eye, float("-inf"))        # a sample is not its own positive
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    same = (y[:, None] == y[None, :]) & ~eye
    n_pos = same.sum(dim=1)
    valid = n_pos > 0
    if not bool(valid.any()):                        # a batch of singletons carries no contrast
        return torch.zeros((), dtype=emb.dtype, device=emb.device)
    # torch.where, not log_prob * same: the diagonal is -inf, and -inf * False is NaN.
    kept = torch.where(same, log_prob, torch.zeros_like(log_prob))
    mean_pos = kept.sum(dim=1)[valid] / n_pos[valid]
    return -mean_pos.mean()


def build_prototypes(emb: np.ndarray, y: np.ndarray, classes: list[str], distance: str
                     ) -> tuple[np.ndarray, list[str]]:
    """One prototype per class from the given embeddings (mean of L2-normalised vectors).

    Built from the inner-validation files of the fold only, so the outer test fold never
    contributes to a prototype that classifies it.
    """
    protos: list[np.ndarray] = []
    kept: list[str] = []
    for c in classes:
        m = y == c
        if not np.any(m):
            continue
        e = np.asarray(emb[m], dtype=float)
        if distance == "cosine":
            n = np.linalg.norm(e, axis=1, keepdims=True)
            e = e / np.maximum(n, 1e-12)
        protos.append(e.mean(axis=0))
        kept.append(str(c))
    out = np.stack(protos) if protos else np.empty((0, np.shape(emb)[1] if np.ndim(emb) else 0))
    if distance == "cosine" and out.size:
        n = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(n, 1e-12)
    return out, kept


def prototype_probabilities(emb: np.ndarray, protos: np.ndarray, proto_classes: list[str],
                            classes: list[str], distance: str, temperature: float) -> np.ndarray:
    """Softmax over negative distances to the prototypes, in ``classes`` column order."""
    k = len(classes)
    if emb.shape[0] == 0 or protos.shape[0] == 0:
        return np.empty((0, k))
    e = np.asarray(emb, dtype=float)
    if distance == "cosine":
        n = np.linalg.norm(e, axis=1, keepdims=True)
        e = e / np.maximum(n, 1e-12)
        d = 1.0 - e @ protos.T
    else:
        d = np.linalg.norm(e[:, None, :] - protos[None, :, :], axis=2)
    logits = -d / max(temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros((p.shape[0], k))
    for j, c in enumerate(proto_classes):
        if c in classes:
            out[:, classes.index(c)] = p[:, j]
    tot = out.sum(axis=1, keepdims=True)
    return np.where(tot > 0, out / np.maximum(tot, 1e-12), 1.0 / k)


# ---------------------------------------------------------------------------
# Training one fold
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    proba_test: np.ndarray                  # (n_test_windows, k), temperature-scaled
    proba_val: np.ndarray                   # (n_val_windows, k), temperature-scaled
    embedding_test: np.ndarray              # (n_test_windows, channels) GAP features
    embedding_val: np.ndarray               # (n_val_windows, channels); out-of-sample for the
                                            # encoder, and the only fitting data of the fusion head
    proba_test_proto: np.ndarray            # (n_test_windows, k) nearest-prototype probabilities
    temperature: float
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = -1
    n_parameters: int = 0
    seconds: float = 0.0
    state: dict[str, Any] = field(default_factory=dict)


def _class_weights(y: np.ndarray, k: int, ratio: float) -> torch.Tensor | None:
    """Class-weighted cross-entropy only at or above the design threshold (11.4)."""
    if ratio < IMBALANCE_WEIGHT_THRESHOLD:
        return None
    counts = np.array([max(np.count_nonzero(y == c), 1) for c in range(k)], dtype=float)
    w = counts.sum() / (k * counts)
    return torch.as_tensor(w, dtype=torch.float32)


def _lr_at(epoch: int, step_in_epoch: int, steps_per_epoch: int, cfg: TrapsigConfig) -> float:
    """Linear warmup then cosine decay to zero."""
    d = cfg.deep
    total = d.epochs * steps_per_epoch
    t = epoch * steps_per_epoch + step_in_epoch
    warm = max(1, d.warmup_epochs * steps_per_epoch)
    if t < warm:
        return d.learning_rate * (t + 1) / warm
    prog = (t - warm) / max(1, total - warm)
    return d.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def _predict(model: CNN1D, x: torch.Tensor, batch: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, x.shape[0], batch):
        out.append(torch.softmax(model(x[i:i + batch]), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, model.n_classes), dtype=float)


@torch.no_grad()
def _embed(model: CNN1D, x: torch.Tensor, batch: int) -> np.ndarray:
    model.eval()
    out = [model.embedding(x[i:i + batch]).cpu().numpy() for i in range(0, x.shape[0], batch)]
    return np.concatenate(out) if out else np.empty((0, 0), dtype=float)


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray, k: int, max_iter: int = 100) -> float:
    """One-parameter temperature scaling fitted on the validation split only.

    Minimises negative log-likelihood over T. T > 1 softens over-confident probabilities,
    which is the usual case for a small CNN. Fitted on validation data, applied to test.
    """
    if logits_val.size == 0:
        return 1.0
    z = torch.as_tensor(logits_val, dtype=torch.float64)
    yy = torch.as_tensor(y_val, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(z / torch.exp(log_t), yy)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(math.exp(log_t.item()))
    return float(min(max(t, 0.05), 20.0))   # clamp so a degenerate fit cannot divide by ~0


def train_fold(
    xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, xte: np.ndarray,
    k: int, cfg: TrapsigConfig, seed: int, imbalance_ratio: float, run_dir: Path | None = None,
    fold_tag: str = "", say: Say | None = None, classes_order: list[str] | None = None,
) -> FoldResult:
    """Train one outer fold: fit, early-stop on validation, temperature-scale, predict test."""
    d = cfg.deep
    classes_order = list(classes_order) if classes_order else [str(i) for i in range(k)]
    dev = torch.device(device_name())
    torch.manual_seed(seed)
    length = xtr.shape[1]
    model = build_model(length, k, cfg).to(dev)
    n_par = count_parameters(model)

    # Conv1d expects (batch, channels, length); a window is a single-channel trace.
    xtr_t = torch.as_tensor(xtr, dtype=torch.float32, device=dev).unsqueeze(1)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    xva_t = torch.as_tensor(xva, dtype=torch.float32, device=dev).unsqueeze(1)
    xte_t = torch.as_tensor(xte, dtype=torch.float32, device=dev).unsqueeze(1)

    weights = _class_weights(ytr, k, imbalance_ratio)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(dev) if weights is not None else None)
    yva_t = torch.as_tensor(yva, dtype=torch.long, device=dev) if len(yva) else None
    opt = torch.optim.AdamW(model.parameters(), lr=d.learning_rate, weight_decay=d.weight_decay)

    n = xtr_t.shape[0]
    steps_per_epoch = max(1, math.ceil(n / d.batch_size))
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_logits: np.ndarray | None = None
    epochs_no_gain = 0
    t_start = time.perf_counter()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    for epoch in range(d.epochs):
        model.train()
        t_ep = time.perf_counter()
        perm = torch.randperm(n, generator=g).to(dev)
        tot_loss, tot_n = 0.0, 0
        metric_tot, metric_n = 0.0, 0
        for b in range(steps_per_epoch):
            idx = perm[b * d.batch_size:(b + 1) * d.batch_size]
            if idx.numel() == 0:
                continue
            lr = _lr_at(epoch, b, steps_per_epoch, cfg)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad(set_to_none=True)
            xb = xtr_t[idx]
            logits = model(xb)
            loss = loss_fn(logits, ytr_t[idx])
            ce_loss = float(loss.item())
            m_loss = float("nan")
            if d.metric_loss != "none" and d.metric_loss_weight > 0 and xb.shape[0] >= 2:
                # The embedding is the same GAP vector the head consumes, so shaping it shapes
                # every downstream strategy at once.
                m = metric_loss(model.embedding(xb), ytr_t[idx], d.metric_loss,
                                d.metric_temperature)
                loss = loss + d.metric_loss_weight * m
                m_loss = float(m.item())
                metric_tot += m_loss * int(idx.numel())
                metric_n += int(idx.numel())
            loss.backward()
            if d.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), d.gradient_clip)
            opt.step()
            tot_loss += float(loss.item()) * int(idx.numel())
            tot_n += int(idx.numel())

        model.eval()
        with torch.no_grad():
            va_logits = torch.cat([model(xva_t[i:i + d.batch_size])
                                   for i in range(0, xva_t.shape[0], d.batch_size)]).cpu().numpy() \
                if xva_t.shape[0] else np.empty((0, k))
        va_proba = np.exp(va_logits - va_logits.max(axis=1, keepdims=True)) if va_logits.size else va_logits
        if va_proba.size:
            va_proba = va_proba / va_proba.sum(axis=1, keepdims=True)
        va_pred = np.argmax(va_proba, axis=1) if va_proba.size else np.empty(0, dtype=np.int64)
        tr_proba = _predict(model, xtr_t, d.batch_size)
        tr_pred = np.argmax(tr_proba, axis=1)

        score = select_value(d.select_metric, yva, va_pred, va_proba, k) if va_proba.size else -math.inf
        rec = {
            "epoch": epoch,
            "train_loss": tot_loss / max(tot_n, 1),
            "train_ce_loss": ce_loss,
            "train_metric_loss": (metric_tot / metric_n) if metric_n else float("nan"),
            "metric_loss_kind": d.metric_loss,
            "metric_loss_weight": d.metric_loss_weight,
            "val_loss": float(-np.mean(np.log(np.clip(va_proba[np.arange(yva.size), yva], 1e-12, 1.0))))
                        if va_proba.size else float("nan"),
            "val_macro_f1": _macro_f1(yva, va_pred, k) if va_proba.size else float("nan"),
            "val_balanced_accuracy": _balanced_acc(yva, va_pred, k) if va_proba.size else float("nan"),
            "train_balanced_accuracy": _balanced_acc(ytr, tr_pred, k),
            "select_metric": d.select_metric,
            "select_value": float(score),
            "learning_rate": float(opt.param_groups[0]["lr"]),
            "epoch_seconds": round(time.perf_counter() - t_ep, 3),
            "cumulative_seconds": round(time.perf_counter() - t_start, 3),
            "batch_size": d.batch_size,
            "window_length_samples": length,
            "n_parameters": n_par,
            "device": str(dev),
        }
        history.append(rec)

        if math.isfinite(score) and score > best_score + d.early_stopping_min_delta:
            best_score, best_epoch, epochs_no_gain = float(score), epoch, 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            best_val_logits = va_logits.copy() if va_logits.size else None
        else:
            epochs_no_gain += 1
        if epochs_no_gain >= d.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    temperature = 1.0
    if d.temperature_scaling and best_val_logits is not None and best_val_logits.size:
        temperature = fit_temperature(best_val_logits, yva, k)

    def scaled(x: torch.Tensor) -> np.ndarray:
        if x.shape[0] == 0:
            return np.empty((0, k))
        model.eval()
        with torch.no_grad():
            parts = [torch.softmax(model(x[i:i + d.batch_size]) / temperature, dim=1).cpu().numpy()
                     for i in range(0, x.shape[0], d.batch_size)]
        return np.concatenate(parts)

    proba_te = scaled(xte_t)
    proba_va = scaled(xva_t)
    emb_te = _embed(model, xte_t, d.batch_size)
    emb_va = _embed(model, xva_t, d.batch_size)

    # Prototypical head: prototypes come from the inner-validation windows, which the encoder was
    # early-stopped on but never trained on, so they are out-of-sample for it and the outer test
    # fold contributes nothing.
    proto_note = ""
    if d.prototypical_head and emb_va.shape[0] and emb_te.shape[0] and yva_t is not None:
        # yva holds integer class indices; the prototypes are keyed by class name so that the
        # probability columns can be put back in the declared class order.
        yva_names = np.array([classes_order[int(i)] if 0 <= int(i) < len(classes_order)
                              else str(int(i)) for i in np.asarray(yva).ravel()], dtype=object)
        protos, proto_classes = build_prototypes(emb_va, yva_names, classes_order, d.proto_distance)
        proba_te_proto = prototype_probabilities(emb_te, protos, proto_classes, classes_order,
                                                 d.proto_distance, d.metric_temperature)
        proto_note = (f"{len(proto_classes)} prototypes from {emb_va.shape[0]} validation windows "
                      f"({d.proto_distance} distance)")
    else:
        proba_te_proto = np.empty((0, k))
        proto_note = ("no prototypes: the validation split is empty" if d.prototypical_head
                      else "prototypical head disabled")
    seconds = time.perf_counter() - t_start

    ckpt_note = ""
    if run_dir is not None and best_state is not None:
        ck_dir = run_dir / "checkpoints"
        ck_dir.mkdir(parents=True, exist_ok=True)
        name = f"cnn1d_{fold_tag}_e{best_epoch}_val{best_score:.4f}.pt"
        torch.save({"state_dict": best_state, "epoch": best_epoch, "val_score": best_score,
                    "select_metric": d.select_metric, "temperature": temperature,
                    "n_parameters": n_par, "window_length_samples": length,
                    "config_fingerprint": cfg.fingerprint(), "seed": seed}, ck_dir / name)
        ckpt_note = name

    return FoldResult(
        proba_test=proba_te, proba_val=proba_va, embedding_test=emb_te, embedding_val=emb_va,
        proba_test_proto=proba_te_proto, temperature=temperature,
        history=history, best_epoch=best_epoch, n_parameters=n_par, seconds=seconds,
        state={"best_val_score": best_score, "best_epoch": best_epoch,
               "select_metric": d.select_metric,
               "epochs_run": len(history), "checkpoint": ckpt_note,
               "class_weighted_loss": weights is not None,
               "receptive_field_samples": int(model.receptive_field),
               "metric_loss": d.metric_loss, "metric_loss_weight": d.metric_loss_weight,
               "metric_temperature": d.metric_temperature,
               "prototypical_head": bool(d.prototypical_head),
               "proto_distance": d.proto_distance,
               "prototypes": proto_note},
    )


# ---------------------------------------------------------------------------
# Validation split inside an outer training fold
# ---------------------------------------------------------------------------

def validation_split(
    y: np.ndarray, groups: np.ndarray, fraction: float, seed: int, k: int
) -> np.ndarray:
    """Boolean mask of the validation windows, chosen by file group.

    Carved from the outer training fold only. Grouped so that all windows of one file
    land on the same side, which is what keeps early stopping from seeing test windows.

    Class-stratified whenever the fold allows it, because four things downstream need both
    classes in the validation split: the selection metric (macro F1 or balanced accuracy is
    degenerate on one class), temperature scaling, the class prototypes of the prototypical head,
    and the fusion head (a logistic regression cannot be fitted on one class). When a class has
    fewer files than the requested split count, ``StratifiedGroupKFold`` refuses; the fallback
    below stays group-aware and takes one file per class first, then fills the budget at random.
    """
    n_splits = max(2, int(round(1.0 / max(fraction, 1e-3))))
    y_int = np.asarray([int(v) for v in y])
    try:
        import warnings

        with warnings.catch_warnings():
            # With few windows per class sklearn warns that a class has fewer members than
            # n_splits. That is expected on small folds and is handled by the fallback below.
            warnings.simplefilter("ignore", UserWarning)
            sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for _, va in sg.split(np.zeros(y_int.shape[0]), y_int, groups):
                return _complete_classes(_index_mask(y_int.shape[0], va), y_int, groups, seed)
    except ValueError:
        pass
    # Too few groups per class for the requested split count: fall back to a by-file split that
    # is still group-aware, and still tries to hold every class present in the training fold.
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    n_val_groups = max(1, int(round(len(uniq) * fraction)))
    label_of: dict[Any, int] = {}
    for g, lab in zip(groups.tolist(), y_int.tolist(), strict=True):
        label_of.setdefault(g, int(lab))
    by_class: dict[int, list[Any]] = {}
    for g in uniq.tolist():
        by_class.setdefault(label_of[g], []).append(g)
    for members in by_class.values():
        rng.shuffle(members)
    order = sorted(by_class)
    chosen: list[Any] = []
    i = 0
    while len(chosen) < n_val_groups and any(by_class[c] for c in order):
        c = order[i % len(order)]
        if by_class[c]:
            chosen.append(by_class[c].pop())
        i += 1
    val_groups = set(chosen)
    return _complete_classes(np.array([g in val_groups for g in groups], dtype=bool), y_int,
                             groups, seed)


def _complete_classes(mask: np.ndarray, y_int: np.ndarray, groups: np.ndarray, seed: int
                      ) -> np.ndarray:
    """Repair a validation split that is missing a class, without breaking group disjointness.

    ``StratifiedGroupKFold`` stratifies best-effort: with few files per class the split it returns
    can hold one class only, and then the selection metric (macro F1, balanced accuracy), the
    temperature fit, the class prototypes and the fusion head are all fitted on a degenerate
    sample. This moves **one file per missing class** from the training side to the validation
    side, and only when that class has at least two files on the training side, so the repair can
    never remove a class from the data the encoder is trained on.
    """
    classes = sorted({int(v) for v in y_int.tolist()})
    if len(classes) < 2 or not mask.any() or mask.all():
        return mask
    out: np.ndarray = np.array(mask, dtype=bool)
    g_list = [str(g) for g in np.asarray(groups).tolist()]
    y_list = [int(v) for v in y_int.tolist()]
    rng = np.random.default_rng(seed + 77)
    for c in classes:
        if any(v == c and m for v, m in zip(y_list, out.tolist(), strict=True)):
            continue                                   # this class is already represented
        train_files = sorted({g for g, v, m in zip(g_list, y_list, out.tolist(), strict=True)
                              if v == c and not m})
        if len(train_files) < 2:
            continue                                   # the class cannot spare a file
        pick = train_files[int(rng.integers(0, len(train_files)))]
        for i, g in enumerate(g_list):
            if g == pick:
                out[i] = True
    return out


def _index_mask(n: int, idx: np.ndarray) -> np.ndarray:
    m: np.ndarray = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


# ---------------------------------------------------------------------------
# Repeated grouped CV over windows
# ---------------------------------------------------------------------------

@dataclass
class DeepCVResult:
    """Window-level out-of-fold probabilities per repeat, plus training logs."""
    proba: np.ndarray                       # (n_repeats, n_windows, k)
    embeddings: np.ndarray                  # (n_repeats, n_windows, channels)
    temperature: np.ndarray                 # (n_repeats,) mean over folds
    proba_proto: np.ndarray | None = None    # (n_repeats, n_windows, k) nearest-prototype
                                             # probabilities; NaN where a fold had no prototype
    val_embeddings: dict[str, np.ndarray] = field(default_factory=dict)   # "r{r}f{f}" -> emb
    val_files: dict[str, list[str]] = field(default_factory=dict)         # window -> sample_id
    hpo_rows: list[dict[str, Any]] = field(default_factory=list)          # one row per trial
    hpo_summaries: list[dict[str, Any]] = field(default_factory=list)     # one per searched fold
    logs: list[dict[str, Any]] = field(default_factory=list)
    # "r{repeat}f{fold}" -> sample_ids of the files whose windows formed the inner validation
    # split of that outer fold. The inner split is drawn at train time, so this is the only
    # record of it; it is copied into the run's metadata.csv.
    inner_val: dict[str, list[str]] = field(default_factory=dict)
    n_parameters: int = 0
    receptive_field_samples: int = 0
    window_length_samples: int = 0
    scale: tuple[float, float] = (0.0, 1.0)
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


def deep_search_space(d: Any) -> dict[str, list[float]]:
    """Model B's declared search space. Width is excluded on purpose (see DeepConfig)."""
    return {"learning_rate": [float(v) for v in d.hpo_space_learning_rate],
            "dropout": [float(v) for v in d.hpo_space_dropout],
            "weight_decay": [float(v) for v in d.hpo_space_weight_decay]}


def search_fold_params(
    xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, xte: np.ndarray, k: int,
    cfg: TrapsigConfig, seed: int, imbalance_ratio: float, classes: list[str], tag: str,
    repeat: int, fold: int, n_train_files: int, say: Say | None = None,
) -> tuple[dict[str, float] | None, list[dict[str, Any]], dict[str, Any]]:
    """Random search over Model B's space, scored on this fold's inner-validation files.

    Returns (best setting or None, one trial row per draw, a summary dict). Trials are run with
    ``run_dir=None`` so a rejected setting leaves no checkpoint behind; the caller retrains once
    with the winner, and that run is the one whose weights and log are kept.
    """
    import dataclasses as _dc

    tell = say or (lambda m: None)
    d = cfg.deep
    space = deep_search_space(d)
    keys = sorted(space)
    combos = list(itertools.product(*(space[k] for k in keys)))
    rng = np.random.default_rng(seed + 991)
    n = max(1, int(d.hpo_trials))
    if len(combos) <= n:
        rng.shuffle(combos)
        draws = list(combos)
    else:
        draws = [combos[int(i)] for i in rng.choice(len(combos), size=n, replace=False)]

    rows: list[dict[str, Any]] = []
    best_params: dict[str, float] | None = None
    best_score: float | None = None
    t0 = time.perf_counter()
    for i, combo in enumerate(draws):
        params = {k: float(v) for k, v in zip(keys, combo, strict=True)}
        cfg_i = _dc.replace(cfg, deep=_dc.replace(
            d, learning_rate=params["learning_rate"], dropout=params["dropout"],
            weight_decay=params["weight_decay"]))
        status, reason, score = "computed", "", float("nan")
        try:
            res = train_fold(xtr, ytr, xva, yva, xte, k, cfg_i, seed + i, imbalance_ratio, None,
                             f"{tag}t{i}", None, classes_order=classes)
            raw = res.state.get("best_val_score")
            score = float(raw) if raw is not None else float("nan")
            if not math.isfinite(score):
                status, reason = "failed", "no computable validation score for this trial"
        except Exception as exc:                       # one bad trial must not stop the fold
            status, reason = "failed", f"{type(exc).__name__}: {exc}"
        rows.append({"model": "cnn1d", "repeat": repeat, "fold": fold, "fold_tag": tag,
                     "method": "random", "metric": d.select_metric, "trial": i, "rung": 0,
                     "n_train_files": n_train_files,
                     "score": None if not math.isfinite(score) else round(score, 6),
                     "n_inner_splits": 1, "status": status, "reason": reason,
                     "params": json.dumps(params, sort_keys=True)})
        if status == "computed" and (best_score is None or score > best_score):
            best_score, best_params = score, dict(params)
        tell(f"deep hpo {tag} trial {i + 1}/{len(draws)}: {params} -> "
             f"{d.select_metric} {score:.4f}" if status == "computed"
             else f"deep hpo {tag} trial {i + 1}/{len(draws)}: FAILED {reason}")
    summary = {"method": "random", "metric": d.select_metric, "space": space,
               "n_searches": 1, "n_trials": len(rows), "best_score": best_score,
               "best_params": best_params or {}, "seconds": round(time.perf_counter() - t0, 3),
               "n_folds_failed": sum(1 for r in rows if r["status"] != "computed"),
               "protocol": ("each trial is a full train_fold scored by deep.select_metric on the "
                            "fold's inner-validation files; the fold is then retrained once with "
                            "the winning setting, and that is the checkpoint and log kept"),
               "not_searched": ("channels, n_blocks, kernel_size and the window length: the "
                                "embedding width is the representation contract shared with the "
                                "prototype head, the fusion head and the conditional analyses, so "
                                "it is declared before the run rather than tuned per fold")}
    return best_params, rows, summary


def cv_windows(
    ws: WindowSet, fold_ids_files: np.ndarray, file_order: list[str], k: int, classes: list[str],
    cfg: TrapsigConfig, imbalance_ratio: float, run_dir: Path | None = None, say: Say | None = None,
) -> DeepCVResult:
    """Out-of-fold window predictions for every repeat.

    ``fold_ids_files`` is (n_repeats, n_files) and ``file_order`` maps its columns to file ids.
    Windows inherit their parent file's fold, so no file appears on both sides of a split.
    The robust scale is fitted on each fold's training windows only.
    """
    tell = say or (lambda m: None)
    d = cfg.deep
    n_rep, n_files = fold_ids_files.shape
    cls_index = {c: i for i, c in enumerate(classes)}
    y_all = np.array([cls_index[c] for c in ws.class_label.tolist()], dtype=np.int64)
    sid_pos = {s: i for i, s in enumerate(file_order)}
    w_file_idx = np.array([sid_pos[s] for s in ws.sample_id.tolist()], dtype=np.int64)
    w_fold = np.zeros((n_rep, ws.x.shape[0]), dtype=np.int64)
    for r in range(n_rep):
        w_fold[r] = fold_ids_files[r][w_file_idx]

    # NaN, not zeros: a window that is never predicted must stay visibly unpredicted, so that
    # aggregate_to_file excludes it rather than averaging in an implicit zero probability.
    proba = np.full((n_rep, ws.x.shape[0], k), np.nan, dtype=float)
    proba_proto = np.full((n_rep, ws.x.shape[0], k), np.nan, dtype=float)
    embs = np.zeros((n_rep, ws.x.shape[0], d.channels), dtype=float)
    temps = np.zeros((n_rep,))
    logs: list[dict[str, Any]] = []
    inner_val: dict[str, list[str]] = {}
    val_embeddings: dict[str, np.ndarray] = {}
    val_files: dict[str, list[str]] = {}
    hpo_rows: list[dict[str, Any]] = []
    hpo_summaries: list[dict[str, Any]] = []
    notes: list[str] = []
    n_par = rf = wlen = 0
    scale_last = (0.0, 1.0)
    t0 = time.perf_counter()
    det_note = seed_everything(cfg.run.seed + 11, d.threads)
    notes.append(det_note)

    for r in range(n_rep):
        for f in range(cfg.splitting.n_splits):
            te_w = w_fold[r] == f
            tr_w = ~te_w
            if not te_w.any():
                notes.append(f"repeat {r} fold {f}: no test windows (skipped)")
                continue
            if not tr_w.any():
                raise DeepError(f"repeat {r} fold {f}: no training windows")
            med, iqr = fit_robust_scale(ws.x[tr_w])
            scale_last = (med, iqr)
            xtr = apply_robust_scale(ws.x[tr_w], med, iqr)
            xte = apply_robust_scale(ws.x[te_w], med, iqr)
            ytr = y_all[tr_w]
            gtr = ws.group_id[tr_w]

            va_mask = validation_split(ytr, gtr, d.validation_fraction, cfg.run.seed + 1000 * r + f, k)
            n_va_groups = len(set(gtr[va_mask].tolist())) if va_mask.any() else 0
            if va_mask.sum() == 0 or (~va_mask).sum() == 0 or n_va_groups < 1:
                notes.append(f"repeat {r} fold {f}: validation split empty; trained without early stopping signal")
                va_mask = np.zeros(va_mask.shape, dtype=bool)
            va_lab = ytr[va_mask]
            # If the validation split lost a class, early stopping degrades; record it.
            if va_mask.any() and len(set(va_lab.tolist())) < k:
                notes.append(f"repeat {r} fold {f}: validation split is missing a class")

            tag = f"r{r}f{f}"
            inner_val[tag] = sorted({str(v) for v in ws.sample_id[tr_w][va_mask].tolist()}) \
                if va_mask.any() else []
            tell(f"deep CV: repeat {r + 1}/{n_rep} fold {f + 1}/{cfg.splitting.n_splits} "
                 f"({int(tr_w.sum())} train, {int(te_w.sum())} test windows)")
            cfg_fold = cfg
            if d.hpo_trials > 0:
                import dataclasses as _dc

                best_params, trial_rows, trial_summary = search_fold_params(
                    xtr[~va_mask], ytr[~va_mask], xtr[va_mask], va_lab, xte, k, cfg,
                    cfg.run.seed + 1000 * r + f, imbalance_ratio, classes, tag, r, f,
                    len({str(v) for v in ws.sample_id[tr_w][~va_mask].tolist()}), tell)
                hpo_rows.extend(trial_rows)
                hpo_summaries.append({**trial_summary, "fold_tag": tag})
                if best_params:
                    cfg_fold = _dc.replace(cfg, deep=_dc.replace(
                        d, learning_rate=best_params["learning_rate"],
                        dropout=best_params["dropout"],
                        weight_decay=best_params["weight_decay"]))
                    notes.append(f"repeat {r} fold {f}: deep hyperparameters chosen by search: "
                                 f"{best_params} ({d.select_metric} "
                                 f"{trial_summary['best_score']:.4f} on the validation files)")
            res = train_fold(xtr[~va_mask], ytr[~va_mask], xtr[va_mask], va_lab, xte, k, cfg_fold,
                             cfg.run.seed + 1000 * r + f, imbalance_ratio, run_dir, tag, tell,
                             classes_order=classes)
            proba[r, te_w] = res.proba_test
            if res.proba_test_proto.shape[0] == int(te_w.sum()):
                proba_proto[r, te_w] = res.proba_test_proto
            # The validation side of this fold: out-of-sample embeddings of files the encoder
            # never trained on. This is the only leakage-safe fitting data a fusion head can have.
            tr_sids = ws.sample_id[tr_w].tolist()
            va_sids = [str(v) for v in np.asarray(tr_sids, dtype=object)[va_mask].tolist()]
            val_embeddings[tag] = np.asarray(res.embedding_val, dtype=float)
            val_files[tag] = va_sids
            embs[r, te_w] = res.embedding_test[:, :d.channels]
            temps[r] += res.temperature / max(1, cfg.splitting.n_splits)
            n_par, rf, wlen = res.n_parameters, res.state["receptive_field_samples"], xtr.shape[1]
            logs.append({"repeat": r, "fold": f, "n_train_windows": int((~va_mask).sum()),
                         "n_val_windows": int(va_mask.sum()), "n_test_windows": int(te_w.sum()),
                         "robust_scale_median": med, "robust_scale_iqr": iqr,
                         "temperature": res.temperature, **res.state, "seconds": round(res.seconds, 2),
                         "history": res.history})
            _append_fold_log(run_dir, tag, res, med, iqr, r, f)

    return DeepCVResult(proba=proba, embeddings=embs, temperature=temps,
                        proba_proto=proba_proto if d.prototypical_head else None,
                        val_embeddings=val_embeddings, val_files=val_files, logs=logs,
                        inner_val=inner_val, hpo_rows=hpo_rows, hpo_summaries=hpo_summaries,
                        n_parameters=n_par, receptive_field_samples=rf,
                        window_length_samples=wlen, scale=scale_last,
                        seconds=time.perf_counter() - t0, notes=notes)


def _append_fold_log(run_dir: Path | None, tag: str, res: FoldResult, med: float, iqr: float,
                     r: int, f: int) -> None:
    """Automated training log: one JSON-lines record per epoch, per fold."""
    if run_dir is None:
        return
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "deep_training_log.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for h in res.history:
            fh.write(json.dumps({"repeat": r, "fold": f, "fold_tag": tag, "temperature": res.temperature,
                                 "robust_scale_median": med, "robust_scale_iqr": iqr,
                                 **res.state, **h}, default=float) + "\n")
