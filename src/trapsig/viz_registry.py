"""The visualization registry (design section 10): 60+ distinct plot types, six categories.

Every entry states its category, what it shows, its prerequisites, its parameters and whether a
CSV twin carries the numbers it draws. ``render`` attempts every entry against the run context;
an entry whose prerequisites are absent is recorded as ``skipped`` with the reason, an entry whose
optional dependency is not shipped (EMD/VMD) is recorded as ``not_available``, and a builder that
raises is recorded as ``failed`` with the exception. Nothing is fabricated: a plot is only written
from arrays the run actually produced, and the index CSV is the audit trail.

Categories follow the design: A time domain, B frequency domain, C time-frequency, D nonlinear
and complexity, E dataset and class analysis, F deep-learning diagnostics.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import viz
from .runs import write_text_exclusive

# ---------------------------------------------------------------------------
# Context handed to every builder
# ---------------------------------------------------------------------------


@dataclass
class RegistryCtx:
    classes: list[str]
    fs: float
    signals: dict[str, list[np.ndarray]] = field(default_factory=dict)  # class -> analysed signals
    sids: dict[str, list[str]] = field(default_factory=dict)
    x: np.ndarray | None = None  # file-level engineered features
    feat_names: list[str] = field(default_factory=list)
    y: np.ndarray | None = None
    ok_rows: list[dict[str, Any]] = field(default_factory=list)
    probas: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # model -> (y, proba)
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    histories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # model -> logs
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    fewshot: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, float] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    explain: dict[str, Any] = field(default_factory=dict)
    redundancy: list[dict[str, Any]] = field(default_factory=list)
    # ---- round-3 additions (spec-168 extension; all optional, defaults keep old behaviour) ----
    events: list[dict[str, Any]] = field(default_factory=list)  # per-event rows (events_table)
    fold_ids: np.ndarray | None = None  # (n_repeats, n_files) fold ids
    groups: np.ndarray | None = None  # per-file group ids
    cfg_snapshot: dict[str, Any] = field(default_factory=dict)  # documented analysis settings
    truth: list[dict[str, Any]] = field(default_factory=list)  # ground-truth events, if any
    meta_rows: list[dict[str, Any]] = field(default_factory=list)  # metadata.csv rows (all files)


def _sig(ctx: RegistryCtx, ci: int = 0, si: int = 0) -> np.ndarray | None:
    c = ctx.classes[ci % max(1, len(ctx.classes))]
    lst = ctx.signals.get(c) or []
    return lst[si % len(lst)] if lst else None


def _any_sig(ctx: RegistryCtx) -> np.ndarray | None:
    for c in ctx.classes:
        if ctx.signals.get(c):
            return ctx.signals[c][0]
    return None


# ---------------------------------------------------------------------------
# Builders (A) time domain
# ---------------------------------------------------------------------------


def b_raw_waveform(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None:
        return None
    t = np.arange(x.size) / ctx.fs
    fig, ax = viz.plt.subplots(figsize=(10, 3))
    ax.plot(t, x, lw=0.6, color="#333333")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("analysed amplitude (configured units)")
    ax.set_title(f"raw analysed waveform at {ctx.fs:g} Hz ({x.size} samples)", fontsize=9)
    return viz._save(fig, p, dpi)


def b_segmented_windows(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 64:
        return None
    n = 6
    seg = x.size // n
    fig, axes = viz.plt.subplots(n, 1, figsize=(9, 1.5 * n), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(np.arange(seg) / ctx.fs, x[i * seg : (i + 1) * seg], lw=0.7)
        ax.set_ylabel(f"seg {i}", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("time within segment (s)")
    fig.suptitle("consecutive non-overlapping segments of one recording", fontsize=9)
    return viz._save(fig, p, dpi)


def b_class_overlay(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    if not any(ctx.signals.get(c) for c in ctx.classes):
        return None
    fig, ax = viz.plt.subplots(figsize=(10, 3.4))
    cols = viz._colors(ctx.classes)
    for c in ctx.classes:
        for s in (ctx.signals.get(c) or [])[:3]:
            m = min(len(s), 4000)
            ax.plot(np.arange(m) / ctx.fs, s[:m], lw=0.7, color=cols[c], alpha=0.6)
    ax.set_xlabel("time (s)")
    ax.set_title("overlaid recordings per class (first 3 per class)", fontsize=9)
    ax.tick_params(labelsize=7)
    return viz._save(fig, p, dpi)


def b_rolling_mean_var(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 200:
        return None
    w = max(16, x.size // 200)
    pad = x[: x.size - x.size % w].reshape(-1, w)
    fig, axes = viz.plt.subplots(2, 1, figsize=(10, 4.4), sharex=True)
    t = (np.arange(pad.shape[0]) + 0.5) * w / ctx.fs
    axes[0].plot(t, pad.mean(axis=1), lw=0.9, color="#0072B2")
    axes[0].set_ylabel("rolling mean")
    axes[1].plot(t, pad.var(axis=1), lw=0.9, color="#D55E00")
    axes[1].set_ylabel("rolling variance")
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"rolling mean and variance (window {w} samples = {w / ctx.fs * 1000:g} ms)", fontsize=9)
    return viz._save(fig, p, dpi)


def b_amplitude_histogram(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None:
        return None
    fig, ax = viz.plt.subplots(figsize=(6, 3.6))
    ax.hist(x, bins=80, color="#0072B2", alpha=0.8)
    ax.set_xlabel("amplitude")
    ax.set_ylabel("samples")
    ax.set_title("amplitude distribution of one recording", fontsize=9)
    return viz._save(fig, p, dpi)


def b_amplitude_boxplot(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    if not any(ctx.signals.get(c) for c in ctx.classes):
        return None
    data = [np.concatenate([s[:4000] for s in (ctx.signals.get(c) or [])[:3]]) for c in ctx.classes]
    fig, ax = viz.plt.subplots(figsize=(6, 3.6))
    bp = ax.boxplot(
        [d for d in data if d.size],
        tick_labels=[c for c, d in zip(ctx.classes, data, strict=True) if d.size],
        patch_artist=True,
    )
    for patch, c in zip(bp["boxes"], ctx.classes, strict=True):
        patch.set_facecolor(viz._colors(ctx.classes)[c])
        patch.set_alpha(0.7)
    ax.set_ylabel("amplitude")
    ax.set_title("amplitude spread per class", fontsize=9)
    return viz._save(fig, p, dpi)


def b_amplitude_qq(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 32:
        return None
    from scipy import stats

    fig, ax = viz.plt.subplots(figsize=(5, 4.4))
    stats.probplot(x[:20000], dist="norm", plot=ax)
    ax.set_title("Q-Q plot of amplitudes against a normal", fontsize=9)
    ax.tick_params(labelsize=7)
    return viz._save(fig, p, dpi)


def _diff_plot(ctx: RegistryCtx, p: Path, dpi: int, n: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < n + 8:
        return None
    d = np.diff(x, n=n)
    fig, ax = viz.plt.subplots(figsize=(10, 3))
    ax.plot(np.arange(d.size) / ctx.fs, d, lw=0.6, color="#009E73")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"difference order {n}")
    ax.set_title(f"{['first', 'second'][n - 1]} difference of the recording", fontsize=9)
    return viz._save(fig, p, dpi)


def b_first_difference(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    return _diff_plot(ctx, p, dpi, 1)


def b_second_difference(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    return _diff_plot(ctx, p, dpi, 2)


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    d = float(np.dot(x, x))
    if d <= 0:
        return np.zeros(nlags + 1)
    return np.array([np.dot(x[: x.size - k], x[k:]) / d for k in range(nlags + 1)])


def b_autocorrelation(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 64:
        return None
    nl = min(200, x.size // 4)
    a = _acf(x, nl)
    fig, ax = viz.plt.subplots(figsize=(8, 3.2))
    ax.vlines(np.arange(nl + 1) / ctx.fs * 1000, 0, a, color="#0072B2")
    ci = 1.96 / math.sqrt(x.size)
    ax.axhline(ci, ls="--", lw=0.8, color="#D55E00")
    ax.axhline(-ci, ls="--", lw=0.8, color="#D55E00")
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("autocorrelation with 95% white-noise bounds", fontsize=9)
    return viz._save(fig, p, dpi)


def b_partial_autocorrelation(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 64:
        return None
    try:
        from statsmodels.tsa.stattools import pacf  # type: ignore[import-not-found]

        a = pacf(x[:4000], nlags=40)
    except Exception:
        nl = 40
        a = _acf(x, nl)  # documented fallback: partial ACF needs statsmodels, ACF shown instead
    fig, ax = viz.plt.subplots(figsize=(8, 3.2))
    ax.vlines(np.arange(len(a)) / ctx.fs * 1000, 0, a, color="#CC79A7")
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("partial autocorrelation (or ACF fallback)")
    ax.set_title("partial autocorrelation where statsmodels is available", fontsize=9)
    return viz._save(fig, p, dpi)


def b_lag_plot(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 8:
        return None
    fig, ax = viz.plt.subplots(figsize=(5, 4.6))
    ax.scatter(x[:-1][:8000], x[1:][:8000], s=4, alpha=0.5, color="#0072B2")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t+1)")
    ax.set_title("lag-1 plot", fontsize=9)
    return viz._save(fig, p, dpi)


def b_peak_locations(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    from scipy import signal as sps

    x = _any_sig(ctx)
    if x is None:
        return None
    pk, _ = sps.find_peaks(np.abs(x), height=0.6 * np.max(np.abs(x)))
    fig, ax = viz.plt.subplots(figsize=(10, 3))
    ax.plot(np.arange(x.size) / ctx.fs, x, lw=0.6, color="#999999")
    ax.plot(pk / ctx.fs, x[pk], "v", color="#D55E00", ms=5, label=f"{pk.size} peaks")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=7)
    ax.set_title("detected peak locations", fontsize=9)
    return viz._save(fig, p, dpi)


def b_hilbert_envelope(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    from scipy import signal as sps

    x = _any_sig(ctx)
    if x is None or x.size < 64:
        return None
    env = np.abs(sps.hilbert(x - x.mean()))
    fig, ax = viz.plt.subplots(figsize=(10, 3))
    ax.plot(np.arange(x.size) / ctx.fs, x, lw=0.5, color="#BBBBBB")
    ax.plot(np.arange(x.size) / ctx.fs, env, lw=0.9, color="#D55E00", label="Hilbert envelope")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=7)
    ax.set_title("analytic envelope", fontsize=9)
    return viz._save(fig, p, dpi)


def b_changepoints(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 200:
        return None
    w = max(16, x.size // 100)
    mu = np.array([x[i : i + w].mean() for i in range(0, x.size - w, w)])
    d = np.abs(np.diff(mu))
    thr = np.median(d) + 3 * (np.median(np.abs(d - np.median(d))) or 1e-9)
    cp = np.flatnonzero(d > thr)
    fig, ax = viz.plt.subplots(figsize=(10, 3))
    ax.plot(np.arange(x.size) / ctx.fs, x, lw=0.5, color="#BBBBBB")
    for i in cp[:20]:
        ax.axvline(i * w / ctx.fs, color="#D55E00", lw=0.8, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_title(f"rolling-mean change points ({min(len(cp), 20)} shown, 3x MAD rule)", fontsize=9)
    return viz._save(fig, p, dpi)


# ---------------------------------------------------------------------------
# Builders (B) frequency domain
# ---------------------------------------------------------------------------


def _spectra(ctx: RegistryCtx) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 64:
        return None
    f = np.fft.rfftfreq(x.size, 1 / ctx.fs)
    X = np.fft.rfft(x - x.mean())
    return f, X


def b_fft_magnitude(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    s = _spectra(ctx)
    if not s:
        return None
    f, X = s
    fig, ax = viz.plt.subplots(figsize=(9, 3.4))
    ax.plot(f, np.abs(X), lw=0.7, color="#0072B2")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|FFT|")
    ax.set_title("FFT magnitude", fontsize=9)
    return viz._save(fig, p, dpi)


def b_fft_phase(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    s = _spectra(ctx)
    if not s:
        return None
    f, X = s
    m = np.abs(X) > np.max(np.abs(X)) * 1e-3
    fig, ax = viz.plt.subplots(figsize=(9, 3.4))
    ax.scatter(f[m], np.angle(X[m]), s=4, color="#CC79A7")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("phase (rad)")
    ax.set_title("FFT phase at significant bins", fontsize=9)
    return viz._save(fig, p, dpi)


def b_psd_periodogram(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    from scipy import signal as sps

    x = _any_sig(ctx)
    if x is None or x.size < 64:
        return None
    f, P = sps.periodogram(x, fs=ctx.fs)
    fig, ax = viz.plt.subplots(figsize=(9, 3.4))
    ax.semilogy(f, P + 1e-30, lw=0.7, color="#009E73")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("periodogram", fontsize=9)
    return viz._save(fig, p, dpi)


def b_welch_psd(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    from scipy import signal as sps

    x = _any_sig(ctx)
    if x is None or x.size < 256:
        return None
    f, P = sps.welch(x, fs=ctx.fs, nperseg=min(1024, x.size // 4))
    fig, ax = viz.plt.subplots(figsize=(9, 3.4))
    ax.semilogy(f, P + 1e-30, lw=0.9, color="#0072B2")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Welch PSD (segmented, averaged)", fontsize=9)
    return viz._save(fig, p, dpi)


def b_multitaper_psd(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    from scipy import signal as sps

    x = _any_sig(ctx)
    if x is None or x.size < 256:
        return None
    from scipy.signal.windows import dpss

    nw, k = 3, 5
    tapers = dpss(min(512, x.size), nw, Kmax=k)
    est = []
    for t in tapers:
        seg = x[: t.size] * t
        f, P = sps.periodogram(seg, fs=ctx.fs)
        est.append(P)
    fig, ax = viz.plt.subplots(figsize=(9, 3.4))
    ax.semilogy(
        f, np.mean(est, axis=0) + 1e-30, lw=0.9, color="#E69F00", label=f"multitaper (NW={nw}, K={k})"
    )
    ax.set_xlabel("frequency (Hz)")
    ax.legend(fontsize=7)
    ax.set_title("multitaper spectral estimate", fontsize=9)
    return viz._save(fig, p, dpi)


def b_dominant_frequency(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    s = _spectra(ctx)
    if not s:
        return None
    f, X = s
    top = np.argsort(-np.abs(X))[:8]
    fig, ax = viz.plt.subplots(figsize=(7, 3.4))
    ax.bar(f[top], np.abs(X[top]), color="#0072B2")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|FFT|")
    ax.set_title("dominant frequencies", fontsize=9)
    ax.tick_params(labelsize=7)
    return viz._save(fig, p, dpi)


def b_harmonics(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    s = _spectra(ctx)
    if not s:
        return None
    f, X = s
    f0 = f[int(np.argmax(np.abs(X)))]
    if f0 <= 0:
        return None
    hs = [f0 * i for i in range(1, 9) if f0 * i < f[-1]]
    m = np.abs(X)
    vals = [float(m[int(np.argmin(np.abs(f - h)))]) for h in hs]
    fig, ax = viz.plt.subplots(figsize=(7, 3.4))
    ax.bar([f"{h:g}" for h in hs], vals, color="#56B4E9")
    ax.set_xlabel("harmonic of f0")
    ax.set_ylabel("|FFT|")
    ax.set_title(f"harmonic series at f0 = {f0:g} Hz", fontsize=9)
    ax.tick_params(labelsize=7)
    return viz._save(fig, p, dpi)


def b_spectral_centroid_bw(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    rows: list[dict[str, Any]] = []
    for c in ctx.classes:
        for s in (ctx.signals.get(c) or [])[:3]:
            f = np.fft.rfftfreq(s.size, 1 / ctx.fs)
            P = np.abs(np.fft.rfft(s - s.mean())) ** 2
            tot = P.sum()
            if tot <= 0:
                continue
            pn = P / tot
            cen = float(np.sum(f * pn))
            bw = float(np.sqrt(np.sum(pn * (f - cen) ** 2)))
            rows.append({"class": c, "centroid_hz": round(cen, 3), "bandwidth_hz": round(bw, 3)})
    if not rows:
        return None
    fig, ax = viz.plt.subplots(figsize=(7, 3.6))
    for c in ctx.classes:
        sub = [r for r in rows if r["class"] == c]
        ax.scatter(
            np.asarray([float(r["centroid_hz"]) for r in sub], dtype=float),
            np.asarray([float(r["bandwidth_hz"]) for r in sub], dtype=float),
            s=26,
            color=viz._colors(ctx.classes)[c],
            label=c,
        )
    ax.set_xlabel("spectral centroid (Hz)")
    ax.set_ylabel("spectral bandwidth (Hz)")
    ax.set_title("centroid vs bandwidth per recording", fontsize=9)
    ax.legend(fontsize=7)
    return viz._save(fig, p, dpi), rows, ["class", "centroid_hz", "bandwidth_hz"]


def b_spectral_rolloff(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    vals = []
    for c in ctx.classes:
        for s in (ctx.signals.get(c) or [])[:3]:
            f = np.fft.rfftfreq(s.size, 1 / ctx.fs)
            P = np.abs(np.fft.rfft(s - s.mean())) ** 2
            cum = np.cumsum(P)
            if cum[-1] <= 0:
                continue
            vals.append(
                {"class": c, "rolloff85_hz": round(float(f[np.searchsorted(cum, 0.85 * cum[-1])]), 2)}
            )
    if not vals:
        return None
    fig, ax = viz.plt.subplots(figsize=(6, 3.6))
    data = [[v["rolloff85_hz"] for v in vals if v["class"] == c] for c in ctx.classes]
    ax.boxplot(
        [np.asarray(d, dtype=float) for d in data if d],
        tick_labels=[c for c, d in zip(ctx.classes, data, strict=True) if d],
    )
    ax.set_ylabel("85% spectral roll-off (Hz)")
    ax.set_title("spectral roll-off per class", fontsize=9)
    return viz._save(fig, p, dpi)


def b_spectral_flatness(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    vals = []
    for c in ctx.classes:
        for s in (ctx.signals.get(c) or [])[:3]:
            P = np.abs(np.fft.rfft(s - s.mean())) ** 2
            P = P[P > 0]
            if P.size == 0:
                continue
            vals.append({"class": c, "flatness": round(float(np.exp(np.mean(np.log(P))) / np.mean(P)), 6)})
    if not vals:
        return None
    fig, ax = viz.plt.subplots(figsize=(6, 3.6))
    data = [[v["flatness"] for v in vals if v["class"] == c] for c in ctx.classes]
    ax.boxplot(
        [np.asarray(d, dtype=float) for d in data if d],
        tick_labels=[c for c, d in zip(ctx.classes, data, strict=True) if d],
    )
    ax.set_ylabel("spectral flatness")
    ax.set_title("spectral flatness per class (1 = noise-like)", fontsize=9)
    return viz._save(fig, p, dpi)


def b_spectral_entropy(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    vals = []
    for c in ctx.classes:
        for s in (ctx.signals.get(c) or [])[:3]:
            P = np.abs(np.fft.rfft(s - s.mean())) ** 2
            tot = P.sum()
            if tot <= 0:
                continue
            pn = P / tot
            vals.append({"class": c, "entropy": round(float(-np.sum(pn[pn > 0] * np.log(pn[pn > 0]))), 6)})
    if not vals:
        return None
    fig, ax = viz.plt.subplots(figsize=(6, 3.6))
    data = [[v["entropy"] for v in vals if v["class"] == c] for c in ctx.classes]
    ax.boxplot(
        [np.asarray(d, dtype=float) for d in data if d],
        tick_labels=[c for c, d in zip(ctx.classes, data, strict=True) if d],
    )
    ax.set_ylabel("spectral entropy (nats)")
    ax.set_title("spectral entropy per class", fontsize=9)
    return viz._save(fig, p, dpi)


def b_band_power_comparison(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    bands = [(1, 10), (10, 100), (100, 1000), (1000, 5000)]
    rows = []
    for c in ctx.classes:
        for lo, hi in bands:
            vals = []
            for s in (ctx.signals.get(c) or [])[:3]:
                f = np.fft.rfftfreq(s.size, 1 / ctx.fs)
                P = np.abs(np.fft.rfft(s - s.mean())) ** 2
                m = (f >= lo) & (f < hi)
                if m.any():
                    vals.append(float(np.log10(P[m].sum() + 1e-12)))
            rows.append(
                {
                    "class": c,
                    "band": f"{lo}-{hi}",
                    "log10_power": round(float(np.mean(vals)), 4) if vals else None,
                }
            )
    if not rows:
        return None
    fig, ax = viz.plt.subplots(figsize=(8, 3.8))
    width = 0.8 / max(len(ctx.classes), 1)
    for i, c in enumerate(ctx.classes):
        ys = []
        for lo, hi in bands:
            b = f"{lo}-{hi}"
            v = next((r["log10_power"] for r in rows if r["class"] == c and r["band"] == b), None)
            ys.append(0.0 if v is None else float(v))
        ax.bar(
            np.arange(len(bands)) + i * width,
            np.asarray(ys, dtype=float),
            width,
            color=viz._colors(ctx.classes)[c],
            label=c,
        )
    ax.set_xticks(np.arange(len(bands)) + width * (len(ctx.classes) - 1) / 2)
    ax.set_xticklabels([f"{lo}-{hi} Hz" for lo, hi in bands], fontsize=7)
    ax.set_ylabel("log10 band power")
    ax.set_title("band power per class", fontsize=9)
    ax.legend(fontsize=7)
    return viz._save(fig, p, dpi), rows, ["class", "band", "log10_power"]


# ---------------------------------------------------------------------------
# Builders (C) time-frequency
# ---------------------------------------------------------------------------


def _stft(ctx: RegistryCtx) -> Any:
    from scipy import signal as sps

    x = _any_sig(ctx)
    if x is None or x.size < 256:
        return None
    nper = min(256, x.size // 4)
    f, t, Z = sps.spectrogram(x, fs=ctx.fs, nperseg=nper, noverlap=nper // 2, mode="complex")
    return f, t, Z


def b_stft_spectrogram(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    s = _stft(ctx)
    if not s:
        return None
    f, t, Z = s
    fig, ax = viz.plt.subplots(figsize=(9, 4))
    ax.pcolormesh(t, f, np.abs(Z), shading="auto", cmap="viridis")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("STFT magnitude spectrogram", fontsize=9)
    return viz._save(fig, p, dpi)


def b_log_spectrogram(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    s = _stft(ctx)
    if not s:
        return None
    f, t, Z = s
    fig, ax = viz.plt.subplots(figsize=(9, 4))
    ax.pcolormesh(t, f, np.log1p(np.abs(Z)), shading="auto", cmap="magma")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("log-magnitude spectrogram", fontsize=9)
    return viz._save(fig, p, dpi)


def _cwt_morlet(x: np.ndarray, widths: np.ndarray, w0: float = 6.0) -> np.ndarray:
    """Morlet CWT computed directly.

    ``scipy.signal.cwt`` and ``scipy.signal.morlet2`` were removed in SciPy 1.15+, so the
    transform is implemented here: complex Morlet kernel per width, same-mode convolution,
    energy normalised by 1/sqrt(width). The result is a valid scalogram for visualisation;
    it is not a reimplementation of any specific library's boundary handling.
    """
    out = np.empty((widths.size, x.size))
    for i, wdt in enumerate(widths):
        m = max(9, int(wdt * 8) | 1)  # odd kernel length
        t = (np.arange(m) - m // 2) / float(wdt)
        psi = math.pi**-0.25 * np.exp(1j * w0 * t) * np.exp(-0.5 * t**2)
        c = np.convolve(x.astype(float), psi[::-1].conj(), mode="same")
        out[i] = np.abs(c) / math.sqrt(float(wdt))
    return out


def b_wavelet_scalogram(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 128:
        return None
    widths = np.arange(2, 60, 3)
    cwt = _cwt_morlet(x[:8192], widths)
    fig, ax = viz.plt.subplots(figsize=(9, 4))
    ax.pcolormesh(np.arange(cwt.shape[1]) / ctx.fs, widths, np.abs(cwt), shading="auto", cmap="viridis")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wavelet width (samples)")
    ax.set_title("continuous wavelet scalogram (Morlet)", fontsize=9)
    return viz._save(fig, p, dpi)


def b_wavelet_coefficients(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 128:
        return None
    widths = np.array([4, 12, 32])
    coeffs = _cwt_morlet(x[:8192], widths)
    fig, axes = viz.plt.subplots(len(widths), 1, figsize=(9, 2.2 * len(widths)), sharex=True)
    for ax, c, w in zip(axes, coeffs, widths, strict=True):
        ax.plot(np.arange(c.size) / ctx.fs, np.abs(c), lw=0.6)
        ax.set_ylabel(f"w={int(w)}", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("wavelet coefficient magnitude at three scales", fontsize=9)
    return viz._save(fig, p, dpi)


def b_wavelet_packet_energy(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 256:
        return None
    ws = np.arange(2, 34, 4)
    coeffs = _cwt_morlet(x[:4096], ws)
    energies = [
        {"width": int(w), "energy": round(float(np.sum(c**2)), 4)} for w, c in zip(ws, coeffs, strict=True)
    ]
    fig, ax = viz.plt.subplots(figsize=(7, 3.4))
    ax.bar([e["width"] for e in energies], [e["energy"] for e in energies], color="#009E73")
    ax.set_xlabel("wavelet width (samples)")
    ax.set_ylabel("coefficient energy")
    ax.set_title("wavelet energy per scale", fontsize=9)
    return viz._save(fig, p, dpi), energies, ["width", "energy"]


def b_time_varying_bandpower(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    from scipy import signal as sps

    x = _any_sig(ctx)
    if x is None or x.size < 512:
        return None
    f, t, Z = sps.spectrogram(x, fs=ctx.fs, nperseg=256, noverlap=192, mode="magnitude")
    bands = [(1, 50), (50, 500), (500, 5000)]
    fig, ax = viz.plt.subplots(figsize=(9, 3.6))
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        if m.any():
            ax.plot(t, np.log10(Z[m].sum(axis=0) + 1e-12), lw=1.0, label=f"{lo}-{hi} Hz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("log10 band power")
    ax.set_title("time-varying band power", fontsize=9)
    ax.legend(fontsize=7)
    return viz._save(fig, p, dpi)


def b_spectral_flux(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    s = _stft(ctx)
    if not s:
        return None
    f, t, Z = s
    mag = np.abs(Z)
    flux = np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0))
    fig, ax = viz.plt.subplots(figsize=(9, 3))
    ax.plot(t[1:], flux, lw=0.9, color="#D55E00")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spectral flux")
    ax.set_title("spectral flux (onset-sensitive)", fontsize=9)
    return viz._save(fig, p, dpi)


def b_instantaneous_frequency(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    from scipy import signal as sps

    x = _any_sig(ctx)
    if x is None or x.size < 128:
        return None
    ana = sps.hilbert(x - x.mean())
    inst = np.diff(np.unwrap(np.angle(ana))) * ctx.fs / (2 * math.pi)
    fig, ax = viz.plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(inst.size) / ctx.fs, inst, lw=0.6, color="#56B4E9")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("instantaneous frequency (Hz)")
    ax.set_title("Hilbert instantaneous frequency", fontsize=9)
    return viz._save(fig, p, dpi)


def b_emd_components(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    return None  # optional dependency not shipped; registry records not_available


def b_vmd_components(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    return None  # optional dependency not shipped; registry records not_available


# ---------------------------------------------------------------------------
# Builders (D) nonlinear and complexity
# ---------------------------------------------------------------------------


def _entropies(ctx: RegistryCtx) -> Any:
    from .quad import _perm_entropy, _sample_entropy

    rows = []
    for c in ctx.classes:
        for i, s in enumerate((ctx.signals.get(c) or [])[:3]):
            rows.append(
                {
                    "class": c,
                    "signal": i,
                    "sample_entropy": round(_sample_entropy(s[:2000]), 4),
                    "perm_entropy3": round(_perm_entropy(s[:4000], 3), 4),
                }
            )
    return rows or None


def b_sample_entropy_summary(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    rows = _entropies(ctx)
    if not rows:
        return None
    fig, ax = viz.plt.subplots(figsize=(6, 3.6))
    data = [
        [r["sample_entropy"] for r in rows if r["class"] == c and np.isfinite(r["sample_entropy"])]
        for c in ctx.classes
    ]
    ax.boxplot(
        [np.asarray(d, dtype=float) for d in data if d],
        tick_labels=[c for c, d in zip(ctx.classes, data, strict=True) if d],
    )
    ax.set_ylabel("sample entropy (m=2, r=0.2 sd)")
    ax.set_title("sample entropy per class", fontsize=9)
    return viz._save(fig, p, dpi), rows, ["class", "signal", "sample_entropy", "perm_entropy3"]


def b_permutation_entropy(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    rows = _entropies(ctx)
    if not rows:
        return None
    fig, ax = viz.plt.subplots(figsize=(6, 3.6))
    data = [
        [r["perm_entropy3"] for r in rows if r["class"] == c and np.isfinite(r["perm_entropy3"])]
        for c in ctx.classes
    ]
    ax.boxplot(
        [np.asarray(d, dtype=float) for d in data if d],
        tick_labels=[c for c, d in zip(ctx.classes, data, strict=True) if d],
    )
    ax.set_ylabel("permutation entropy (order 3, normalised)")
    ax.set_title("permutation entropy per class", fontsize=9)
    return viz._save(fig, p, dpi)


def b_multiscale_entropy(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    from .quad import _sample_entropy

    rows: list[dict[str, Any]] = []
    for c in ctx.classes:
        for scale in (1, 2, 4, 8, 16):
            vals = []
            for s in (ctx.signals.get(c) or [])[:2]:
                n = s.size // scale
                coarse = s[: n * scale].reshape(n, scale).mean(axis=1)
                v = _sample_entropy(coarse[:2000])
                if np.isfinite(v):
                    vals.append(v)
            rows.append({"class": c, "scale": scale, "mse": round(float(np.mean(vals)), 4) if vals else None})
    if not rows:
        return None
    fig, ax = viz.plt.subplots(figsize=(7, 3.8))
    for c in ctx.classes:
        sub = [(r["scale"], r["mse"]) for r in rows if r["class"] == c and r["mse"] is not None]
        if sub:
            ax.plot(
                np.asarray([float(s) for s, _ in sub], dtype=float),
                np.asarray([float(m) for _, m in sub], dtype=float),
                marker="o",
                color=viz._colors(ctx.classes)[c],
                label=c,
            )
    ax.set_xlabel("coarse-graining scale")
    ax.set_ylabel("sample entropy")
    ax.set_title("multiscale entropy", fontsize=9)
    ax.legend(fontsize=7)
    return viz._save(fig, p, dpi), rows, ["class", "scale", "mse"]


def _recurrence(ctx: RegistryCtx) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 200:
        return None
    m = x[:1500]
    d = np.abs(m[:, None] - m[None, :])
    eps = np.percentile(d, 90)
    return d <= eps, eps


def b_recurrence_plot(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    r = _recurrence(ctx)
    if not r:
        return None
    R, eps = r
    fig, ax = viz.plt.subplots(figsize=(6, 6))
    ax.imshow(R, cmap="gray_r", interpolation="nearest")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_title(f"recurrence plot (eps = 90th percentile = {eps:.3g})", fontsize=9)
    return viz._save(fig, p, dpi)


def b_recurrence_quantification(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    r = _recurrence(ctx)
    if not r:
        return None
    R, eps = r
    n = R.shape[0]
    rr = float(R.mean())
    diag = []
    for k in range(-n + 1, n):
        line = np.diagonal(R, offset=k)
        run = 0
        best = 0
        for v in line:
            run = run + 1 if v else 0
            best = max(best, run)
        diag.append(best)
    det = float(np.mean([1 if R[i, i + 1] else 0 for i in range(n - 1)]))
    rows: list[dict[str, Any]] = [
        {"measure": "recurrence_rate", "value": round(rr, 6)},
        {"measure": "determinism_l2", "value": round(det, 6)},
        {"measure": "max_diagonal_length", "value": float(max(diag))},
        {"measure": "epsilon", "value": round(float(eps), 6)},
    ]
    fig, ax = viz.plt.subplots(figsize=(6, 3.2))
    ax.barh(
        np.asarray([str(x["measure"]) for x in rows][::-1], dtype=object),
        np.asarray([float(x["value"]) for x in rows][::-1], dtype=float),
        color="#CC79A7",
    )
    ax.set_title("recurrence quantification of one recording", fontsize=9)
    ax.tick_params(labelsize=7)
    return viz._save(fig, p, dpi), rows, ["measure", "value"]


def b_phase_space(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 60:
        return None
    m = x[:3000]
    fig, ax = viz.plt.subplots(figsize=(5.4, 5))
    ax.plot(m[:-10], m[10:], lw=0.5, color="#0072B2")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t+10)")
    ax.set_title("phase-space reconstruction (lag 10)", fontsize=9)
    return viz._save(fig, p, dpi)


def b_fractal_dimension(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    rows = []
    for c in ctx.classes:
        for s in (ctx.signals.get(c) or [])[:3]:
            m = s[:4000]
            L = np.array([np.abs(np.diff(m, 2**k)).sum() if 2**k < m.size else np.nan for k in range(1, 6)])
            fd = (
                float(np.polyfit(np.log2(np.arange(1, 6)), np.log2(np.where(L > 0, L, np.nan)), 1)[0])
                if np.all(np.isfinite(L)) and np.all(L > 0)
                else np.nan
            )
            rows.append({"class": c, "higuchi_like_fd": round(fd, 4) if np.isfinite(fd) else None})
    if not rows or all(r["higuchi_like_fd"] is None for r in rows):
        return None
    fig, ax = viz.plt.subplots(figsize=(6, 3.6))
    data = [
        [r["higuchi_like_fd"] for r in rows if r["class"] == c and r["higuchi_like_fd"] is not None]
        for c in ctx.classes
    ]
    ax.boxplot(
        [np.asarray(d, dtype=float) for d in data if d],
        tick_labels=[c for c, d in zip(ctx.classes, data, strict=True) if d],
    )
    ax.set_ylabel("fractal dimension estimate")
    ax.set_title("fractal dimension per class", fontsize=9)
    return viz._save(fig, p, dpi)


def b_dfa_plot(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    x = _any_sig(ctx)
    if x is None or x.size < 512:
        return None
    y = np.cumsum(x[:8192] - x[:8192].mean())
    ns = [2**k for k in range(5, int(np.log2(y.size)) - 1)]
    fluc = []
    for n in ns:
        seg = y[: y.size - y.size % n].reshape(-1, n)
        t = np.arange(n)
        fit = np.empty_like(seg)
        for i in range(seg.shape[0]):
            coef = np.polyfit(t, seg[i], 1)
            fit[i] = coef[0] * t + coef[1]
        fluc.append(float(np.sqrt(np.mean((seg - fit) ** 2))))
    rows = [{"n": n, "fluctuation": round(f, 6)} for n, f in zip(ns, fluc, strict=True)]
    fig, ax = viz.plt.subplots(figsize=(6, 4))
    ax.loglog(ns, fluc, marker="o", color="#009E73")
    if len(ns) > 2:
        alpha = np.polyfit(np.log(ns), np.log(fluc), 1)[0]
        ax.set_title(f"detrended fluctuation analysis (alpha = {alpha:.3f})", fontsize=9)
    ax.set_xlabel("window size (samples)")
    ax.set_ylabel("fluctuation")
    return viz._save(fig, p, dpi), rows, ["n", "fluctuation"]


def b_symbolic_complexity(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    rows = []
    for c in ctx.classes:
        for s in (ctx.signals.get(c) or [])[:3]:
            m = s[:4000]
            q = np.digitize(m, np.percentile(m, [25, 50, 75]))
            words = [tuple(q[i : i + 3]) for i in range(0, q.size - 3, 3)]
            uniq = len(set(words)) / max(1, len(words))
            rows.append({"class": c, "symbolic_diversity": round(uniq, 4)})
    if not rows:
        return None
    fig, ax = viz.plt.subplots(figsize=(6, 3.6))
    data = [[r["symbolic_diversity"] for r in rows if r["class"] == c] for c in ctx.classes]
    ax.boxplot(
        [np.asarray(d, dtype=float) for d in data if d],
        tick_labels=[c for c, d in zip(ctx.classes, data, strict=True) if d],
    )
    ax.set_ylabel("distinct 3-symbol word ratio")
    ax.set_title("symbolic complexity per class", fontsize=9)
    return viz._save(fig, p, dpi)


# ---------------------------------------------------------------------------
# Builders (E) dataset and class analysis
# ---------------------------------------------------------------------------


def b_class_counts(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    if not ctx.ok_rows:
        return None
    rows: list[dict[str, Any]] = [
        {"class": c, "n_files": sum(1 for r in ctx.ok_rows if str(r.get("class")) == c)} for c in ctx.classes
    ]
    fig, ax = viz.plt.subplots(figsize=(6, 3.4))
    ax.bar(
        np.asarray([str(r["class"]) for r in rows], dtype=object),
        np.asarray([float(r["n_files"]) for r in rows], dtype=float),
        color=[viz._colors(ctx.classes)[str(r["class"])] for r in rows],
    )
    for i, r in enumerate(rows):
        ax.text(i, float(r["n_files"]), str(r["n_files"]), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("files")
    ax.set_title("class counts (discovered from folders)", fontsize=9)
    return viz._save(fig, p, dpi), rows, ["class", "n_files"]


def b_class_duration(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    rows = []
    for c in ctx.classes:
        for s in ctx.signals.get(c) or []:
            rows.append({"class": c, "duration_s": round(s.size / ctx.fs, 3)})
    if not rows:
        return None
    fig, ax = viz.plt.subplots(figsize=(6, 3.6))
    data = [[r["duration_s"] for r in rows if r["class"] == c] for c in ctx.classes]
    ax.boxplot(
        [np.asarray(d, dtype=float) for d in data if d],
        tick_labels=[c for c, d in zip(ctx.classes, data, strict=True) if d],
    )
    ax.set_ylabel("duration (s)")
    ax.set_title("recording duration per class", fontsize=9)
    return viz._save(fig, p, dpi), rows, ["class", "duration_s"]


def b_signal_quality(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    rows = []
    for c in ctx.classes:
        for i, s in enumerate((ctx.signals.get(c) or [])[:4]):
            rows.append(
                {
                    "class": c,
                    "signal": i,
                    "clip_fraction": round(
                        float(np.mean(np.abs(s) >= 0.999 * np.max(np.abs(s))))
                        if np.max(np.abs(s)) > 0
                        else 0.0,
                        6,
                    ),
                    "flatline_fraction": round(float(np.mean(np.diff(s) == 0)), 6),
                    "nan_fraction": 0.0,
                    "outlier_rate": round(
                        float(np.mean(np.abs(s - np.median(s)) > 5 * np.median(np.abs(s - np.median(s))))), 6
                    ),
                }
            )
    if not rows:
        return None
    fig, axes = viz.plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, key in zip(axes, ("clip_fraction", "flatline_fraction", "outlier_rate"), strict=True):
        data = [[r[key] for r in rows if r["class"] == c] for c in ctx.classes]
        ax.boxplot(
            [np.asarray(d, dtype=float) for d in data if d],
            tick_labels=[c for c, d in zip(ctx.classes, data, strict=True) if d],
        )
        ax.set_title(key, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("signal-quality diagnostics per class", fontsize=9)
    return (
        viz._save(fig, p, dpi),
        rows,
        ["class", "signal", "clip_fraction", "flatline_fraction", "nan_fraction", "outlier_rate"],
    )


def b_classwise_waveforms(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    if not any(ctx.signals.get(c) for c in ctx.classes):
        return None
    n = len(ctx.classes)
    fig, axes = viz.plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, ctx.classes, strict=True):
        lst = ctx.signals.get(c) or []
        if lst:
            m = min(len(lst[0]), 6000)
            ax.plot(np.arange(m) / ctx.fs, lst[0][:m], lw=0.7, color=viz._colors(ctx.classes)[c])
        ax.set_ylabel(c, fontsize=8)
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("one representative waveform per class", fontsize=9)
    return viz._save(fig, p, dpi)


def b_classwise_psd(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    from scipy import signal as sps

    fig, ax = viz.plt.subplots(figsize=(9, 3.6))
    drawn = False
    for c in ctx.classes:
        sigs = [s for s in (ctx.signals.get(c) or [])[:3] if s.size >= 256]
        ps = []
        if sigs:
            nmin = min(s.size for s in sigs)
            nper = min(1024, nmin // 4)
            for s in sigs:
                f, P = sps.welch(s[:nmin], fs=ctx.fs, nperseg=nper)
                ps.append(P)
        if ps:
            ax.semilogy(f, np.mean(ps, axis=0) + 1e-30, lw=1.0, color=viz._colors(ctx.classes)[c], label=c)
            drawn = True
    if not drawn:
        return None
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("class-wise Welch PSD", fontsize=9)
    ax.legend(fontsize=7)
    return viz._save(fig, p, dpi)


def b_feature_correlation_matrix(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    if ctx.x is None or ctx.x.shape[1] < 3:
        return None
    from scipy.stats import spearmanr

    keep = [
        j for j in range(ctx.x.shape[1]) if np.isfinite(ctx.x[:, j]).sum() >= 3 and np.nanstd(ctx.x[:, j]) > 0
    ][:40]
    if len(keep) < 3:
        return None
    rho = np.atleast_2d(np.asarray(spearmanr(ctx.x[:, keep]).statistic, dtype=float))
    rho = np.where(np.isfinite(rho), rho, 0.0)
    fig, ax = viz.plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.abs(rho), cmap="viridis", vmin=0, vmax=1)
    labs = [ctx.feat_names[j] for j in keep]
    ax.set_xticks(range(len(labs)), labs, rotation=90, fontsize=5)
    ax.set_yticks(range(len(labs)), labs, fontsize=5)
    ax.set_title("|Spearman| between engineered features", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return viz._save(fig, p, dpi)


def b_feature_importance_plot(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    models = ((ctx.explain.get("methods") or {}).get("features") or {}).get("models") or {}
    top = next(
        (d.get("top_features_by_class") for d in models.values() if d.get("top_features_by_class")), None
    )
    if not top:
        return None
    cls = list(top)[0]
    items = top[cls][:15][::-1]
    fig, ax = viz.plt.subplots(figsize=(8, 0.4 * len(items) + 1.4))
    ax.barh([i[0] for i in items], [i[1] for i in items], color="#0072B2")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"feature importance, class {cls}", fontsize=9)
    ax.tick_params(labelsize=6)
    return viz._save(fig, p, dpi)


def b_feature_redundancy(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    if not ctx.redundancy:
        return None
    rows = ctx.redundancy[:20]
    fig, ax = viz.plt.subplots(figsize=(9, 0.35 * len(rows) + 1.4))
    ax.barh(
        [f"{r['feature_a']} | {r['feature_b']}" for r in rows][::-1],
        [abs(float(r.get("abs_spearman", r.get("spearman", 0.0)))) for r in rows][::-1],
        color="#D55E00",
    )
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("|Spearman|")
    ax.set_title("redundant feature pairs (reported, never dropped)", fontsize=9)
    ax.tick_params(labelsize=5)
    twin = [
        {
            "feature_a": r["feature_a"],
            "feature_b": r["feature_b"],
            "spearman": r.get("spearman"),
            "abs_spearman": abs(float(r.get("abs_spearman", r.get("spearman", 0.0)))),
        }
        for r in rows
    ]
    return viz._save(fig, p, dpi), twin, ["feature_a", "feature_b", "spearman", "abs_spearman"]


def b_feature_distributions(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    if ctx.x is None or ctx.y is None or not ctx.feat_names:
        return None
    return viz.plot_feature_distributions(ctx.x, ctx.feat_names, ctx.y, ctx.classes, p, dpi)


# ---------------------------------------------------------------------------
# Builders (F) deep-learning diagnostics
# ---------------------------------------------------------------------------


def _hist_rows(ctx: RegistryCtx) -> list[dict[str, Any]]:
    rows = []
    for model, logs in ctx.histories.items():
        for lg in logs:
            for h in lg.get("history") or []:
                if isinstance(h, dict) and "epoch" in h:
                    rows.append({"model": model, **h})
    return rows


def b_train_val_loss(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    rows = [r for r in _hist_rows(ctx) if r.get("train_loss") is not None or r.get("val_loss") is not None]
    if not rows:
        return None
    fig, ax = viz.plt.subplots(figsize=(8, 4))
    for model in {r["model"] for r in rows}:
        sub = [r for r in rows if r["model"] == model]
        ep = [int(r["epoch"]) for r in sub]
        tl = [float("nan") if r.get("train_loss") is None else float(r["train_loss"]) for r in sub]
        vl = [float("nan") if r.get("val_loss") is None else float(r["val_loss"]) for r in sub]
        ax.plot(ep, tl, marker=".", lw=0.9, label=f"{model} train")
        if any(v is not None for v in vl):
            ax.plot(ep, vl, marker=".", lw=0.9, ls="--", label=f"{model} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("training and validation loss", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    return viz._save(fig, p, dpi), rows, ["model", "epoch", "train_loss", "val_loss"]


def b_metric_curves(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    rows = [r for r in _hist_rows(ctx) if r.get("val_score") is not None]
    if not rows:
        return None
    fig, ax = viz.plt.subplots(figsize=(8, 4))
    for model in {r["model"] for r in rows}:
        sub = [r for r in rows if r["model"] == model]
        ax.plot([r["epoch"] for r in sub], [r["val_score"] for r in sub], marker=".", lw=0.9, label=model)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation score (selection metric)")
    ax.set_title("validation metric curves (checkpoint selection)", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    return viz._save(fig, p, dpi), rows, ["model", "epoch", "val_score"]


def b_confusion_matrix(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    name = next((n for n in ctx.preds), None)
    if not name:
        return None
    y, pr = ctx.preds[name]
    return viz.plot_confusion(y, pr, ctx.classes, f"{name} (aggregated)", p, dpi)


def b_roc_curve(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    name = next((n for n, (y, pb) in ctx.probas.items() if pb.shape[1] >= 2), None)
    if not name:
        return None
    y, pb = ctx.probas[name]
    return viz.plot_roc(y, pb, ctx.classes, f"{name} (out-of-fold)", p, dpi)


def b_pr_curve(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    name = next((n for n, (y, pb) in ctx.probas.items() if pb.shape[1] >= 2), None)
    if not name:
        return None
    y, pb = ctx.probas[name]
    return viz.plot_pr_curve(y, pb, ctx.classes, f"{name} (out-of-fold)", p, dpi)


def b_calibration_plot(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    name = next((n for n, (y, pb) in ctx.probas.items() if pb.shape[1] >= 2), None)
    if not name:
        return None
    y, pb = ctx.probas[name]
    return viz.plot_calibration(y, pb, name, p, dpi)


def b_embedding_umap_tsne(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    key = next(iter(ctx.embeddings), None)
    if not key:
        return None
    emb, y = ctx.embeddings[key]
    return viz.plot_embedding_xy(emb, y, ctx.classes, p, dpi, "tsne")


def b_embedding_pca(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    key = next(iter(ctx.embeddings), None)
    if not key:
        return None
    emb, y = ctx.embeddings[key]
    return viz.plot_embedding_xy(emb, y, ctx.classes, p, dpi, "pca")


def b_branch_ablation(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    return viz.plot_branch_ablation(ctx.summary, p, dpi)


def b_fusion_contribution(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    if not ctx.gates:
        return None
    return viz.plot_quad_gates(ctx.gates, p, dpi)


def b_fewshot_episode_curve(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    if not ctx.fewshot:
        return None
    accs: dict[int, list[float]] = {}
    for r in ctx.fewshot:
        accs.setdefault(int(r["shot"]), []).append(float(r["accuracy"]))
    rows = [
        {"shot": k, "mean": round(float(np.mean(v)), 4), "n_episodes": len(v)}
        for k, v in sorted(accs.items())
    ]
    fig, ax = viz.plt.subplots(figsize=(7, 3.8))
    ax.plot([r["shot"] for r in rows], [r["mean"] for r in rows], marker="o", color="#0072B2")
    ax.set_xlabel("shot")
    ax.set_ylabel("mean episode accuracy")
    ax.set_title("episode accuracy across episodes per shot", fontsize=9)
    return viz._save(fig, p, dpi), rows, ["shot", "mean", "n_episodes"]


def b_shot_performance(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    from .fewshot import summarize_episodes

    rows = summarize_episodes(ctx.fewshot)
    if not rows:
        return None
    return (
        viz.plot_shot_curve(rows, p, dpi),
        rows,
        [
            "shot",
            "n_episodes",
            "episode_accuracy_mean",
            "episode_accuracy_sd",
            "pooled_query_accuracy",
            "wilson_ci95_lo",
            "wilson_ci95_hi",
        ],
    )


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

Entry = dict[str, Any]
REGISTRY: list[Entry] = [
    # --- A time domain ---
    {
        "id": "raw_waveform",
        "cat": "A",
        "title": "raw analysed waveform",
        "b": b_raw_waveform,
        "prereq": "one analysed recording",
        "params": "fs from configuration",
        "csv": False,
    },
    {
        "id": "segmented_windows",
        "cat": "A",
        "title": "consecutive segments of a recording",
        "b": b_segmented_windows,
        "prereq": "recording >= 64 samples",
        "params": "6 segments",
        "csv": False,
    },
    {
        "id": "class_overlay",
        "cat": "A",
        "title": "overlaid recordings per class",
        "b": b_class_overlay,
        "prereq": "signals per class",
        "params": "3 per class",
        "csv": False,
    },
    {
        "id": "rolling_mean_var",
        "cat": "A",
        "title": "rolling mean and variance",
        "b": b_rolling_mean_var,
        "prereq": "recording >= 200 samples",
        "params": "auto window",
        "csv": False,
    },
    {
        "id": "amplitude_histogram",
        "cat": "A",
        "title": "amplitude distribution",
        "b": b_amplitude_histogram,
        "prereq": "one recording",
        "params": "80 bins",
        "csv": False,
    },
    {
        "id": "amplitude_boxplot",
        "cat": "A",
        "title": "amplitude spread per class",
        "b": b_amplitude_boxplot,
        "prereq": "signals per class",
        "params": "",
        "csv": False,
    },
    {
        "id": "amplitude_qq",
        "cat": "A",
        "title": "Q-Q plot against normal",
        "b": b_amplitude_qq,
        "prereq": "recording >= 32 samples",
        "params": "scipy.probplot",
        "csv": False,
    },
    {
        "id": "first_difference",
        "cat": "A",
        "title": "first difference",
        "b": b_first_difference,
        "prereq": "recording",
        "params": "",
        "csv": False,
    },
    {
        "id": "second_difference",
        "cat": "A",
        "title": "second difference",
        "b": b_second_difference,
        "prereq": "recording",
        "params": "",
        "csv": False,
    },
    {
        "id": "autocorrelation",
        "cat": "A",
        "title": "autocorrelation with CI bounds",
        "b": b_autocorrelation,
        "prereq": "recording >= 64 samples",
        "params": "lags <= 200",
        "csv": False,
    },
    {
        "id": "partial_autocorrelation",
        "cat": "A",
        "title": "partial autocorrelation",
        "b": b_partial_autocorrelation,
        "prereq": "recording >= 64 samples",
        "params": "statsmodels optional, ACF fallback documented",
        "csv": False,
    },
    {
        "id": "lag_plot",
        "cat": "A",
        "title": "lag-1 scatter",
        "b": b_lag_plot,
        "prereq": "recording",
        "params": "",
        "csv": False,
    },
    {
        "id": "peak_locations",
        "cat": "A",
        "title": "detected peaks",
        "b": b_peak_locations,
        "prereq": "recording",
        "params": "height 0.6 max",
        "csv": False,
    },
    {
        "id": "hilbert_envelope",
        "cat": "A",
        "title": "Hilbert envelope",
        "b": b_hilbert_envelope,
        "prereq": "recording >= 64 samples",
        "params": "",
        "csv": False,
    },
    {
        "id": "changepoints",
        "cat": "A",
        "title": "rolling-mean change points",
        "b": b_changepoints,
        "prereq": "recording >= 200 samples",
        "params": "3x MAD rule",
        "csv": False,
    },
    # --- B frequency ---
    {
        "id": "fft_magnitude",
        "cat": "B",
        "title": "FFT magnitude",
        "b": b_fft_magnitude,
        "prereq": "recording >= 64",
        "params": "",
        "csv": False,
    },
    {
        "id": "fft_phase",
        "cat": "B",
        "title": "FFT phase at significant bins",
        "b": b_fft_phase,
        "prereq": "recording >= 64",
        "params": "1e-3 max threshold",
        "csv": False,
    },
    {
        "id": "psd_periodogram",
        "cat": "B",
        "title": "periodogram",
        "b": b_psd_periodogram,
        "prereq": "recording >= 64",
        "params": "scipy periodogram",
        "csv": False,
    },
    {
        "id": "welch_psd",
        "cat": "B",
        "title": "Welch PSD",
        "b": b_welch_psd,
        "prereq": "recording >= 256",
        "params": "nperseg <= 1024",
        "csv": False,
    },
    {
        "id": "multitaper_psd",
        "cat": "B",
        "title": "multitaper PSD",
        "b": b_multitaper_psd,
        "prereq": "recording >= 256",
        "params": "NW=3, K=5 dpss",
        "csv": False,
    },
    {
        "id": "dominant_frequency",
        "cat": "B",
        "title": "dominant frequencies",
        "b": b_dominant_frequency,
        "prereq": "recording >= 64",
        "params": "top 8 bins",
        "csv": False,
    },
    {
        "id": "harmonics",
        "cat": "B",
        "title": "harmonic series at f0",
        "b": b_harmonics,
        "prereq": "f0 > 0",
        "params": "8 harmonics",
        "csv": False,
    },
    {
        "id": "spectral_centroid_bw",
        "cat": "B",
        "title": "centroid vs bandwidth",
        "b": b_spectral_centroid_bw,
        "prereq": "signals per class",
        "params": "",
        "csv": True,
    },
    {
        "id": "spectral_rolloff",
        "cat": "B",
        "title": "85% spectral roll-off",
        "b": b_spectral_rolloff,
        "prereq": "signals per class",
        "params": "",
        "csv": False,
    },
    {
        "id": "spectral_flatness",
        "cat": "B",
        "title": "spectral flatness",
        "b": b_spectral_flatness,
        "prereq": "signals per class",
        "params": "",
        "csv": False,
    },
    {
        "id": "spectral_entropy",
        "cat": "B",
        "title": "spectral entropy",
        "b": b_spectral_entropy,
        "prereq": "signals per class",
        "params": "",
        "csv": False,
    },
    {
        "id": "band_power_comparison",
        "cat": "B",
        "title": "band power per class",
        "b": b_band_power_comparison,
        "prereq": "signals per class",
        "params": "1-10,10-100,100-1k,1k-5k Hz",
        "csv": True,
    },
    # --- C time-frequency ---
    {
        "id": "stft_spectrogram",
        "cat": "C",
        "title": "STFT magnitude spectrogram",
        "b": b_stft_spectrogram,
        "prereq": "recording >= 256",
        "params": "nperseg <= 256",
        "csv": False,
    },
    {
        "id": "log_spectrogram",
        "cat": "C",
        "title": "log-magnitude spectrogram",
        "b": b_log_spectrogram,
        "prereq": "recording >= 256",
        "params": "",
        "csv": False,
    },
    {
        "id": "wavelet_scalogram",
        "cat": "C",
        "title": "Morlet scalogram",
        "b": b_wavelet_scalogram,
        "prereq": "recording >= 128",
        "params": "widths 2..60",
        "csv": False,
    },
    {
        "id": "wavelet_coefficients",
        "cat": "C",
        "title": "wavelet coefficients at 3 scales",
        "b": b_wavelet_coefficients,
        "prereq": "recording >= 128",
        "params": "w=4,12,32",
        "csv": False,
    },
    {
        "id": "wavelet_packet_energy",
        "cat": "C",
        "title": "wavelet energy per scale",
        "b": b_wavelet_packet_energy,
        "prereq": "recording >= 256",
        "params": "",
        "csv": True,
    },
    {
        "id": "time_varying_bandpower",
        "cat": "C",
        "title": "time-varying band power",
        "b": b_time_varying_bandpower,
        "prereq": "recording >= 512",
        "params": "3 bands",
        "csv": False,
    },
    {
        "id": "spectral_flux",
        "cat": "C",
        "title": "spectral flux",
        "b": b_spectral_flux,
        "prereq": "recording >= 256",
        "params": "",
        "csv": False,
    },
    {
        "id": "instantaneous_frequency",
        "cat": "C",
        "title": "Hilbert instantaneous frequency",
        "b": b_instantaneous_frequency,
        "prereq": "recording >= 128",
        "params": "",
        "csv": False,
    },
    {
        "id": "emd_components",
        "cat": "C",
        "title": "EMD intrinsic modes",
        "b": b_emd_components,
        "prereq": "optional EMD dependency (not shipped)",
        "params": "",
        "csv": False,
        "available": False,
    },
    {
        "id": "vmd_components",
        "cat": "C",
        "title": "VMD components",
        "b": b_vmd_components,
        "prereq": "optional VMD dependency (not shipped)",
        "params": "",
        "csv": False,
        "available": False,
    },
    # --- D nonlinear ---
    {
        "id": "sample_entropy_summary",
        "cat": "D",
        "title": "sample entropy per class",
        "b": b_sample_entropy_summary,
        "prereq": "signals per class",
        "params": "m=2, r=0.2 sd",
        "csv": True,
    },
    {
        "id": "permutation_entropy",
        "cat": "D",
        "title": "permutation entropy per class",
        "b": b_permutation_entropy,
        "prereq": "signals per class",
        "params": "order 3",
        "csv": False,
    },
    {
        "id": "multiscale_entropy",
        "cat": "D",
        "title": "multiscale entropy",
        "b": b_multiscale_entropy,
        "prereq": "signals per class",
        "params": "scales 1..16",
        "csv": True,
    },
    {
        "id": "recurrence_plot",
        "cat": "D",
        "title": "recurrence plot",
        "b": b_recurrence_plot,
        "prereq": "recording >= 200",
        "params": "eps = 90th percentile",
        "csv": False,
    },
    {
        "id": "recurrence_quantification",
        "cat": "D",
        "title": "recurrence quantification",
        "b": b_recurrence_quantification,
        "prereq": "recording >= 200",
        "params": "RR, DET, Lmax",
        "csv": True,
    },
    {
        "id": "phase_space",
        "cat": "D",
        "title": "phase-space reconstruction",
        "b": b_phase_space,
        "prereq": "recording >= 60",
        "params": "lag 10",
        "csv": False,
    },
    {
        "id": "fractal_dimension",
        "cat": "D",
        "title": "fractal dimension per class",
        "b": b_fractal_dimension,
        "prereq": "signals per class",
        "params": "log-log slope",
        "csv": False,
    },
    {
        "id": "dfa_plot",
        "cat": "D",
        "title": "detrended fluctuation analysis",
        "b": b_dfa_plot,
        "prereq": "recording >= 512",
        "params": "dyadic windows",
        "csv": True,
    },
    {
        "id": "symbolic_complexity",
        "cat": "D",
        "title": "symbolic word diversity",
        "b": b_symbolic_complexity,
        "prereq": "signals per class",
        "params": "quartile alphabet",
        "csv": False,
    },
    # --- E dataset ---
    {
        "id": "class_counts",
        "cat": "E",
        "title": "class counts",
        "b": b_class_counts,
        "prereq": "manifest rows",
        "params": "",
        "csv": True,
    },
    {
        "id": "class_duration",
        "cat": "E",
        "title": "recording duration per class",
        "b": b_class_duration,
        "prereq": "signals per class",
        "params": "",
        "csv": True,
    },
    {
        "id": "signal_quality",
        "cat": "E",
        "title": "clipping / flat-line / outlier diagnostics",
        "b": b_signal_quality,
        "prereq": "signals per class",
        "params": "",
        "csv": True,
    },
    {
        "id": "classwise_waveforms",
        "cat": "E",
        "title": "representative waveform per class",
        "b": b_classwise_waveforms,
        "prereq": "signals per class",
        "params": "",
        "csv": False,
    },
    {
        "id": "classwise_psd",
        "cat": "E",
        "title": "class-wise Welch PSD",
        "b": b_classwise_psd,
        "prereq": "signals per class",
        "params": "",
        "csv": False,
    },
    {
        "id": "feature_correlation_matrix",
        "cat": "E",
        "title": "|Spearman| feature matrix",
        "b": b_feature_correlation_matrix,
        "prereq": "feature matrix",
        "params": "top 40 features",
        "csv": False,
    },
    {
        "id": "feature_importance_plot",
        "cat": "E",
        "title": "feature importance (SHAP)",
        "b": b_feature_importance_plot,
        "prereq": "explainability enabled",
        "params": "top 15",
        "csv": False,
    },
    {
        "id": "feature_redundancy",
        "cat": "E",
        "title": "redundant pairs",
        "b": b_feature_redundancy,
        "prereq": "redundancy pairs computed",
        "params": "top 20",
        "csv": True,
    },
    {
        "id": "feature_distributions",
        "cat": "E",
        "title": "feature distributions by class",
        "b": b_feature_distributions,
        "prereq": "feature matrix",
        "params": "<=4 per domain",
        "csv": False,
    },
    # --- F deep diagnostics ---
    {
        "id": "train_val_loss",
        "cat": "F",
        "title": "training/validation loss curves",
        "b": b_train_val_loss,
        "prereq": "deep or quad logs",
        "params": "",
        "csv": True,
    },
    {
        "id": "metric_curves",
        "cat": "F",
        "title": "validation metric curves",
        "b": b_metric_curves,
        "prereq": "deep or quad logs",
        "params": "",
        "csv": True,
    },
    {
        "id": "confusion_matrix",
        "cat": "F",
        "title": "confusion matrix",
        "b": b_confusion_matrix,
        "prereq": "aggregated predictions",
        "params": "",
        "csv": False,
    },
    {
        "id": "roc_curve",
        "cat": "F",
        "title": "one-vs-rest ROC",
        "b": b_roc_curve,
        "prereq": "finite probabilities",
        "params": "",
        "csv": False,
    },
    {
        "id": "pr_curve",
        "cat": "F",
        "title": "one-vs-rest precision-recall",
        "b": b_pr_curve,
        "prereq": "finite probabilities",
        "params": "",
        "csv": False,
    },
    {
        "id": "calibration_plot",
        "cat": "F",
        "title": "reliability diagram",
        "b": b_calibration_plot,
        "prereq": "finite probabilities",
        "params": "",
        "csv": False,
    },
    {
        "id": "embedding_umap_tsne",
        "cat": "F",
        "title": "t-SNE of the shared embedding",
        "b": b_embedding_umap_tsne,
        "prereq": "embeddings >= 5 files",
        "params": "sklearn TSNE",
        "csv": False,
    },
    {
        "id": "embedding_pca",
        "cat": "F",
        "title": "PCA of the shared embedding",
        "b": b_embedding_pca,
        "prereq": "embeddings >= 4 files",
        "params": "sklearn PCA",
        "csv": False,
    },
    {
        "id": "branch_ablation",
        "cat": "F",
        "title": "branch ablation of the fusion",
        "b": b_branch_ablation,
        "prereq": "quad + ablation variants scored",
        "params": "",
        "csv": False,
    },
    {
        "id": "fusion_contribution",
        "cat": "F",
        "title": "fusion gate contribution per branch",
        "b": b_fusion_contribution,
        "prereq": "quad gates",
        "params": "",
        "csv": False,
    },
    {
        "id": "fewshot_episode_curve",
        "cat": "F",
        "title": "episode accuracy per shot",
        "b": b_fewshot_episode_curve,
        "prereq": "few-shot episodes",
        "params": "",
        "csv": True,
    },
    {
        "id": "shot_performance",
        "cat": "F",
        "title": "shot curve with Wilson intervals",
        "b": b_shot_performance,
        "prereq": "few-shot episodes",
        "params": "",
        "csv": True,
    },
]


def render(
    ctx: RegistryCtx, out_dir: Path, csv_dir: Path, dpi: int = 150, say: Callable[[str], None] | None = None
) -> list[dict[str, Any]]:
    """Draw every applicable registry entry; record the fate of each one."""
    tell = say or (lambda m: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, Any]] = []
    for e in REGISTRY:
        row = {
            "id": e["id"],
            "spec": e.get("spec", ""),
            "category": e["cat"],
            "title": e["title"],
            "prereq": e["prereq"],
            "params": e["params"],
            "csv_twin": "",
            "status": "",
            "reason": "",
            "file": "",
        }
        if e.get("available") is False:
            row.update(
                {"status": "not_available", "reason": f"optional dependency not shipped: {e['prereq']}"}
            )
            statuses.append(row)
            continue
        try:
            out = e["b"](ctx, out_dir / f"{e['id']}.png", dpi)
        except ModuleNotFoundError as exc:
            row.update(
                {"status": "not_available", "reason": f"optional plotting dependency unavailable: {exc.name}"}
            )
            tell(f"registry plot {e['id']} unavailable: {exc}")
            statuses.append(row)
            continue
        except Exception as exc:
            row.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            tell(f"registry plot {e['id']} failed: {exc}")
            statuses.append(row)
            continue
        rows = None
        if isinstance(out, tuple):
            out, rows, fields = out
            if rows:
                buf = io.StringIO()
                w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
                twin = csv_dir / f"{e['id']}.csv"
                write_text_exclusive(twin, buf.getvalue())
                row["csv_twin"] = str(twin)
        if out:
            row.update({"status": "generated", "file": str(out)})
        else:
            row.update({"status": "skipped", "reason": f"prerequisite not met on this run: {e['prereq']}"})
        statuses.append(row)
    return statuses


def write_registry_index(statuses: list[dict[str, Any]], path: Path) -> str:
    fields = ["id", "category", "title", "status", "spec", "reason", "file", "csv_twin", "prereq", "params"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for s in statuses:
        w.writerow({k: ("" if s.get(k) is None else s.get(k)) for k in fields})
    return str(write_text_exclusive(path, buf.getvalue()))


def registry_count() -> int:
    return len(REGISTRY)


# ---------------------------------------------------------------------------
# Round-3 extension: the 168-item visualization specification
# ---------------------------------------------------------------------------
# The extension module is imported LAST so it can reference RegistryCtx, and it
# only ever APPENDS entries - the original 67 stay exactly as they were.

from .viz_registry_ext import EXT_ENTRIES  # noqa: E402

REGISTRY.extend(EXT_ENTRIES)
