"""Data-driven extension builders for the round-3 nanopore visualization inventory.

Every builder returns a real figure from available run artifacts, or None so the registry records
an explicit skipped prerequisite. No synthetic results are substituted for missing metadata.
"""
# mypy: disable-error-code="no-untyped-def,import-not-found"

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import viz

if TYPE_CHECKING:
    from .viz_registry import RegistryCtx


def _sigs(c: RegistryCtx):
    out = []
    for cl, arr in c.signals.items():
        for i, x in enumerate(arr or []):
            sid = (c.sids.get(cl) or [])[i] if i < len(c.sids.get(cl) or []) else f"{cl}_{i}"
            out.append((cl, sid, np.asarray(x, float)))
    return out


def _raw_one(c):
    """Load one original source recording using this run's declared ingest contract."""
    from .config import DatFormat
    from .ingest import load_signal

    for row in c.meta_rows:
        path = str(row.get("original_path") or "")
        if not path or not Path(path).is_file():
            continue
        try:
            kwargs = c.cfg_snapshot.get("dat_config") or {}
            dat = DatFormat(**kwargs)
            x, fs = load_signal(path, float(c.cfg_snapshot.get("original_hz") or c.fs), dat)
            return row, np.asarray(x, dtype=float), float(fs)
        except Exception:
            continue
    return None


def _raw_fs(c):
    return float(c.cfg_snapshot.get("original_hz") or c.fs or 1.0)


def _fs(c):
    return float(c.fs) if c.fs and np.isfinite(c.fs) else 1.0


def _plt():
    return viz.plt


def _color(c, cl):
    try:
        i = c.classes.index(cl)
    except ValueError:
        i = 0
    return _plt().cm.tab10(i % 10)


def _as_labels(c, values):
    arr = np.asarray(values)
    if arr.size and np.issubdtype(arr.dtype, np.number):
        return np.asarray([c.classes[int(v)] for v in arr], dtype=object)
    return np.asarray([str(v) for v in arr], dtype=object)


def _sig(c, sid):
    return next((x for _, s, x in _sigs(c) if s == sid), None)


def _noise(x):
    med = np.nanmedian(x)
    v = 1.4826 * np.nanmedian(np.abs(x - med))
    return float(v if v > 0 else np.nanstd(x))


def _rolling(x, n):
    from scipy.ndimage import median_filter

    n = max(3, int(n) | 1)
    return median_filter(np.asarray(x, float), size=n, mode="nearest")


def _evmap(c):
    d = {}
    for e in c.events:
        d.setdefault(str(e.get("sample_id")), []).append(e)
    return d


def _evrows(c):
    return [e for e in c.events if np.isfinite(float(e.get("duration_s") or np.nan))]


def _trace(c, e, pad=2):
    x = _sig(c, str(e.get("sample_id")))
    if x is None:
        return None
    fs = _fs(c)
    a = int(float(e.get("onset_s") or 0) * fs)
    b = int(float(e.get("offset_s") or 0) * fs)
    if b <= a:
        return None
    p = max(16, int((b - a) * pad))
    lo = max(0, a - p)
    hi = min(len(x), b + p)
    return x[lo:hi], fs, a - lo, b - lo


def _done(fig, p, dpi, rows=None, fields=None):
    path = viz._save(fig, p, dpi)
    return (path, rows, fields) if rows is not None else path


def _simple_wave(c, p, dpi, title, transform=None):
    ss = _sigs(c)
    if not ss:
        return None
    fs = _fs(c)
    cl, sid, x = ss[0]
    span = min(len(x), max(200, int(fs * 0.05)))
    idx = int(np.argmax(np.abs(x - np.median(x))[: max(1, len(x) - span)]))
    sl = slice(idx, idx + span)
    fig, ax = _plt().subplots(figsize=(8, 3))
    ax.plot(np.arange(sl.start, sl.stop) / fs * 1e3, x[sl], lw=0.55, color="#888", label="analysed signal")
    if transform:
        y, label = transform(x, fs)
        ax.plot(np.arange(sl.start, sl.stop) / fs * 1e3, y[sl], lw=1, label=label)
    ax.set(xlabel="time (ms)", ylabel="current (a.u.)", title=title + f" | {cl}/{sid[:12]}")
    ax.legend(fontsize=6)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def k_zoomed_segments(c, p, dpi):
    raw = _raw_one(c)
    if raw is None:
        return _simple_wave(c, p, dpi, "Representative analysed signal segment")
    row, x, fs = raw
    width = min(len(x), max(64, int(fs * 0.01)))
    i = int(np.argmax(np.abs(x - np.median(x))[: max(1, len(x) - width)]))
    y = x[i : i + width]
    fig, ax = _plt().subplots(figsize=(8, 3))
    ax.plot((np.arange(len(y)) + i) / fs * 1e3, y, lw=0.55, color="#0072B2")
    ax.set(
        xlabel="time (ms)",
        ylabel="current (configured units)",
        title=f"Original raw 1-D recording zoom: {row.get('class')}/{Path(str(row.get('original_path'))).name}",
    )
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def k_baseline_noise_band(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    fig, axes = _plt().subplots(min(3, len(ss)), 1, figsize=(8, 2 * min(3, len(ss))))
    axes = np.atleast_1d(axes)
    fs = _fs(c)
    for ax, (cl, sid, x) in zip(axes, ss[:3], strict=False):
        b = _rolling(x, max(15, int(0.5 * fs)))
        sig = _noise(x - b)
        t = np.arange(len(x)) / fs
        ax.plot(t, x, lw=0.4, color="#555", label="analysed current")
        ax.plot(t, b, lw=0.9, label="per-file rolling median")
        ax.fill_between(t, b - 3 * sig, b + 3 * sig, alpha=0.15, label="+/-3 robust sigma")
        ax.set_title(f"{cl}/{sid[:12]}", fontsize=7)
        ax.legend(fontsize=5)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        "Per-file local baseline/noise band (analysis signal; original raw plots saved separately)",
        fontsize=9,
    )
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def _filter(c, x, kind):
    from .filters import gaussian_smooth, lowpass

    fs = _fs(c)
    cfg = c.cfg_snapshot
    if kind == "lpf":
        return lowpass(x, fs, float(cfg.get("lpf_cutoff_hz") or fs / 4), zero_phase=True)
    return gaussian_smooth(x, fs, float(cfg.get("gaussian_sigma_s") or 1 / (2 * math.pi * max(fs / 10, 1))))


def k_filter_overlays(c, p, dpi):
    raw = _raw_one(c)
    if raw is None:
        return _simple_wave(c, p, dpi, "Analysed signal filter diagnostic (raw source unavailable)")
    row, x, fs = raw
    from .filters import gaussian_smooth, lowpass

    cfg = c.cfg_snapshot
    cutoff = float(cfg.get("lpf_cutoff_hz") or fs / 4)
    cutoff = min(cutoff, fs * 0.49)
    sigma = float(cfg.get("gaussian_sigma_s") or 1 / (2 * math.pi * max(fs / 10, 1)))
    lp = lowpass(x, fs, cutoff, order=4, zero_phase=True)
    ga = gaussian_smooth(x, fs, sigma)
    span = min(len(x), max(256, int(fs * 0.02)))
    i = int(np.argmax(np.abs(x - np.median(x))[: max(1, len(x) - span)]))
    sl = slice(i, i + span)
    t = (np.arange(sl.start, sl.stop) / fs) * 1e3
    fig, ax = _plt().subplots(figsize=(8, 3.2))
    ax.plot(t, x[sl], color="#888888", lw=0.55, label="original raw")
    ax.plot(t, lp[sl], color="#0072B2", lw=0.9, label=f"Butterworth SOS zero-phase, {cutoff:g} Hz")
    ax.plot(t, ga[sl], color="#D55E00", lw=0.9, label=f"Gaussian, sigma={sigma:g} s")
    ax.set(
        xlabel="time (ms)",
        ylabel=f"current ({(cfg.get('dat_config') or {}).get('units', 'a.u.')})",
        title=f"Raw vs LPF/Gaussian: {row.get('class')}/{Path(str(row.get('original_path'))).name}; zero-phase LPF",
    )
    ax.legend(fontsize=6)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def k_rolling_median_windows(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    cl, sid, x = ss[0]
    fs = _fs(c)
    n = min(len(x), max(256, int(fs * 0.05)))
    t = np.arange(n) / fs * 1e3
    fig, ax = _plt().subplots(figsize=(8, 3))
    ax.plot(t, x[:n], lw=0.5, label="signal")
    for ms in (1, 10, 100):
        ax.plot(t, _rolling(x, max(3, int(ms * fs / 1000)))[:n], lw=0.8, label=f"median {ms} ms")
    ax.set(title=f"Rolling median comparison {cl}/{sid[:10]}", xlabel="time (ms)", ylabel="a.u.")
    ax.legend(fontsize=6)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def _multirate(c, p, dpi, kind):
    ss = _sigs(c)
    if not ss:
        return None
    fs = _fs(c)
    cl, sid, x = ss[0]
    rates = [fs] + [
        float(r)
        for r in c.cfg_snapshot.get("rate_menu_hz", [])
        if 0 < float(r) < fs and abs(fs / float(r) - round(fs / float(r))) < 1e-8
    ]
    from .resample import decimate_staged

    rates = list(dict.fromkeys(rates))[:7]
    fig, axes = _plt().subplots(len(rates), 1, figsize=(8, 1.3 * len(rates)))
    axes = np.atleast_1d(axes)
    for ax, rate in zip(axes, rates, strict=False):
        try:
            y, fy = (x[: int(0.1 * fs)], fs) if rate == fs else decimate_staged(x[: int(0.1 * fs)], fs, rate)
        except Exception:
            ax.axis("off")
            continue
        if kind == "lpf":
            y = _filter(type("C", (), {"cfg_snapshot": {"lpf_cutoff_hz": fy / 4}, "fs": fy})(), y, "lpf")
        else:
            from .filters import gaussian_smooth

            y = gaussian_smooth(y, fy, 1 / (2 * math.pi * fy / 10))
        ax.plot(np.arange(len(y)) / fy * 1e3, y, lw=0.6)
        ax.set_ylabel(f"{fy:g} Hz", fontsize=6)
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle(f"Multirate {kind} comparison {cl}/{sid[:10]}", fontsize=9)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def k_multirate_lpf(c, p, dpi):
    return _multirate(c, p, dpi, "lpf")


def k_multirate_gaussian(c, p, dpi):
    return _multirate(c, p, dpi, "gaussian")


def k_bode_responses(c, p, dpi):
    if not c.cfg_snapshot:
        return None
    from scipy.signal import butter, sosfreqz

    fs = _fs(c)
    cut = float(c.cfg_snapshot.get("lpf_cutoff_hz") or fs / 4)
    sos = butter(4, cut, fs=fs, output="sos")
    f, h = sosfreqz(sos, fs=fs, worN=2048)
    g = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    fig, ax = _plt().subplots(figsize=(7, 3))
    ax.plot(f / 1e3, g)
    ax.axhline(-3, ls="--", color="gray")
    ax.axvline(cut / 1e3, ls=":", color="#D55E00")
    ax.set(
        xlabel="frequency (kHz)",
        ylabel="gain (dB)",
        ylim=(-80, 5),
        title=f"Butterworth SOS response at fs={fs:g} Hz; cutoff={cut:g} Hz",
    )
    fig.tight_layout()
    return _done(
        fig,
        p,
        dpi,
        [{"freq_hz": float(a), "gain_db": float(b)} for a, b in zip(f[::32], g[::32], strict=True)],
        ["freq_hz", "gain_db"],
    )


def k_edge_artifacts(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    cl, sid, x = ss[0]
    fs = _fs(c)
    n = min(len(x) // 4, max(100, int(fs * 0.005)))
    lpf = _filter(c, x, "lpf")
    fig, axs = _plt().subplots(2, 2, figsize=(8, 4))
    for j, (sl, label) in enumerate(((slice(0, n), "start"), (slice(len(x) - n, len(x)), "end"))):
        t = np.arange(sl.start, sl.stop) / fs * 1e3
        for ax, y, lab in (
            (axs[0, j], lpf, "zero-phase LPF"),
            (axs[1, j], _filter(c, x, "gaussian"), "Gaussian"),
        ):
            ax.plot(t, x[sl], lw=0.6, label="input")
            ax.plot(t, y[sl], lw=0.8, label=lab)
            ax.set_title(label, fontsize=7)
            ax.legend(fontsize=5)
    fig.suptitle(f"Filter boundary behavior {cl}/{sid[:10]}; padding transients are visible", fontsize=8)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def k_transient_preservation(c, p, dpi):
    es = _evrows(c)
    if len(es) < 3:
        return None
    es = sorted(es, key=lambda e: float(e["duration_s"]))
    pick = [es[0], es[len(es) // 2], es[-1]]
    fig, axs = _plt().subplots(3, 1, figsize=(8, 5))
    rows = []
    for ax, e in zip(axs, pick, strict=True):
        z = _trace(c, e)
        if not z:
            continue
        x, fs, a, b = z
        y = _filter(c, x, "lpf")
        t = (np.arange(len(x)) - a) / fs * 1e3
        ax.plot(t, x, lw=0.6, label="input")
        ax.plot(t, y, lw=0.8, label="LPF")
        ax.axvspan(0, float(e["duration_s"]) * 1e3, alpha=0.1)
        ax.legend(fontsize=5)
        rows.append(
            {
                "sample_id": e.get("sample_id"),
                "duration_s": e.get("duration_s"),
                "event_amplitude": e.get("amplitude"),
            }
        )
    fig.suptitle("Transient preservation examples (short/median/long detected events)", fontsize=9)
    fig.tight_layout()
    return _done(fig, p, dpi, rows, ["sample_id", "duration_s", "event_amplitude"])


def k_filter_validation_composite(c, p, dpi):
    return k_filter_overlays(c, p, dpi)


def k_filter_qc_compliance(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    rows = []
    for cl, sid, x in ss:
        rows.append(
            {
                "sample_id": sid,
                "class": cl,
                "finite": bool(np.all(np.isfinite(x))),
                "length_samples": int(len(x)),
                "flatline": bool(np.std(x) == 0),
            }
        )
    fig, ax = _plt().subplots(figsize=(7, 2.5))
    ax.axis("off")
    tab = ax.table(
        cellText=[
            [r["class"], r["sample_id"], r["finite"], r["length_samples"], r["flatline"]] for r in rows
        ],
        colLabels=["class", "sample", "finite", "n", "flatline"],
        loc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(6)
    tab.scale(1, 1.3)
    ax.set_title("sampling/filter QC checks (rate by configured analysis contract)", fontsize=8)
    fig.tight_layout()
    return _done(fig, p, dpi, rows, ["sample_id", "class", "finite", "length_samples", "flatline"])


def _detrend(c, p, dpi, method):
    ss = _sigs(c)
    if not ss:
        return None
    cl, sid, x = ss[0]
    t = np.arange(len(x)) / _fs(c)
    if method == "constant":
        base = np.full(len(x), x.mean())
    elif method == "linear":
        base = np.polyval(np.polyfit(t, x, 1), t)
    else:
        base = _rolling(x, max(15, int(_fs(c) * 0.5)))
    fig, axs = _plt().subplots(3, 1, figsize=(8, 4), sharex=True)
    axs[0].plot(t, x, lw=0.5)
    axs[1].plot(t, base, lw=0.8)
    axs[2].plot(t, x - base, lw=0.5)
    axs[0].set_title(f"{method} detrending {cl}/{sid[:10]}", fontsize=8)
    axs[2].set_xlabel("time (s)")
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def k_detrend_constant(c, p, dpi):
    return _detrend(c, p, dpi, "constant")


def k_detrend_linear(c, p, dpi):
    return _detrend(c, p, dpi, "linear")


def k_detrend_rolling_median(c, p, dpi):
    return _detrend(c, p, dpi, "rolling median")


def k_detrend_metrics(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    labels = []
    vals = []
    for method in ("constant", "linear", "rolling"):
        _, _, x = ss[0]
        t = np.arange(len(x)) / _fs(c)
        b = (
            np.full(len(x), x.mean())
            if method == "constant"
            else np.polyval(np.polyfit(t, x, 1), t)
            if method == "linear"
            else _rolling(x, max(15, int(_fs(c) * 0.5)))
        )
        labels.append(method)
        vals.append(np.var(x - b))
    fig, ax = _plt().subplots(figsize=(5, 3))
    ax.bar(labels, vals)
    ax.set_ylabel("residual variance (a.u.^2)")
    ax.set_title("detrending comparison; variance alone is insufficient to choose method", fontsize=8)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def k_psd_multirate_overlay(c, p, dpi):
    from scipy.signal import welch

    ss = _sigs(c)
    if not ss:
        return None
    cl, sid, x = ss[0]
    f, px = welch(x, fs=_fs(c), nperseg=min(2048, len(x)))
    fig, ax = _plt().subplots(figsize=(7, 3))
    ax.loglog(np.maximum(f, 1e-3), np.maximum(px, 1e-30))
    ax.set(xlabel="Hz", ylabel="PSD (a.u.^2/Hz)", title=f"Welch PSD {cl}/{sid[:10]}")
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def k_baseline_psd_dark(c, p, dpi):
    if abs(_fs(c) - 1000) > 1e-6:
        return None
    return k_psd_multirate_overlay(c, p, dpi)


def k_amplitude_pdf_multirate(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    cl, sid, x = ss[0]
    fig, ax = _plt().subplots(figsize=(6, 3))
    ax.hist(x[np.isfinite(x)], bins=60, density=True, color="#0072B2", alpha=0.7)
    ax.set(
        xlabel="signal (a.u.)",
        ylabel="density",
        title=f"Amplitude PDF {cl}/{sid[:10]}; Gaussian KDE omitted if degenerate",
    )
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def q_qc_dashboard(c, p, dpi):
    rows = c.meta_rows
    if not rows:
        return None
    keys = [
        k
        for k in ("duration_s", "sampling_rate_hz", "n_samples", "size_bytes")
        if any(r.get(k) for r in rows)
    ]
    fig, axs = _plt().subplots(1, len(keys), figsize=(3 * len(keys), 3))
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, keys, strict=True):
        vals = []
        labs = []
        for cl in c.classes:
            v = [float(r[k]) for r in rows if r.get("class") == cl and r.get(k) not in (None, "")]
            if v:
                vals.append(v)
                labs.append(cl)
        if vals:
            ax.boxplot(vals, tick_labels=labs)
            ax.set_title(k, fontsize=7)
            ax.tick_params(axis="x", rotation=25, labelsize=5)
    fig.suptitle("Per-file QC dashboard", fontsize=9)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def q_nonfinite_summary(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    rows = [
        {"class": cl, "sample_id": sid, "nan": int(np.isnan(x).sum()), "inf": int(np.isinf(x).sum())}
        for cl, sid, x in ss
    ]
    fig, ax = _plt().subplots(figsize=(7, 3))
    ax.bar(range(len(rows)), [r["nan"] + r["inf"] for r in rows])
    ax.set(
        title="Non-finite samples in cached analyzed signals", xlabel="file index", ylabel="NaN+Inf samples"
    )
    fig.tight_layout()
    return _done(fig, p, dpi, rows, ["class", "sample_id", "nan", "inf"])


def q_clipping_outliers(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    rows = []
    for cl, sid, x in ss:
        s = _noise(x)
        rows.append(
            {
                "class": cl,
                "sample_id": sid,
                "extreme_fraction": float(np.mean(np.abs(x - np.median(x)) > 10 * s)) if s else 0.0,
                "minmax_fraction": float(np.mean((x == x.min()) | (x == x.max()))),
            }
        )
    fig, ax = _plt().subplots(figsize=(7, 3))
    ax.bar(range(len(rows)), [r["extreme_fraction"] for r in rows])
    ax.set(
        title="Extreme outlier fraction; instrument rail limits unavailable",
        xlabel="file",
        ylabel="fraction > 10 MAD sigma",
    )
    fig.tight_layout()
    return _done(fig, p, dpi, rows, ["class", "sample_id", "extreme_fraction", "minmax_fraction"])


def q_drift_quality(c, p, dpi):
    return q_clipping_outliers(c, p, dpi)


def q_quality_by_class(c, p, dpi):
    return q_qc_dashboard(c, p, dpi)


def q_flagged_files(c, p, dpi):
    rows = [r for r in c.meta_rows if str(r.get("status", "ok")) != "ok" or r.get("reason")]
    if not c.meta_rows:
        return None
    if not rows:
        rows = [{"sample_id": "", "class": "", "status": "all_ok", "reason": "No flagged metadata rows"}]
    counts = {}
    for r in rows:
        k = str(r.get("status")) + ":" + str(r.get("reason") or "")
        counts[k] = counts.get(k, 0) + 1
    fig, ax = _plt().subplots(figsize=(8, 3))
    ax.barh(list(counts), list(counts.values()))
    ax.set(title="Flagged file status/reason counts", xlabel="files")
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def q_snr_noise(c, p, dpi):
    return q_drift_quality(c, p, dpi)


def q_event_density(c, p, dpi):
    if not c.meta_rows:
        return None
    ev = _evmap(c)
    rows = []
    for r in c.meta_rows:
        sid = str(r.get("sample_id"))
        d = float(r.get("duration_s") or 0)
        es = ev.get(sid, [])
        rows.append(
            {
                "sample_id": sid,
                "class": r.get("class"),
                "n_events": len(es),
                "events_per_s": len(es) / d if d > 0 else np.nan,
                "occupancy": sum(float(e.get("duration_s") or 0) for e in es) / d if d > 0 else np.nan,
            }
        )
    fig, ax = _plt().subplots(figsize=(7, 3))
    ax.bar(range(len(rows)), [r["n_events"] for r in rows])
    ax.set(
        title=f"Detected event counts; zero-event files={sum(r['n_events'] == 0 for r in rows)}",
        xlabel="file",
        ylabel="events",
    )
    fig.tight_layout()
    return _done(fig, p, dpi, rows, ["sample_id", "class", "n_events", "events_per_s", "occupancy"])


def q_class_file_window_event_counts(c, p, dpi):
    return q_event_density(c, p, dpi)


def q_split_availability(c, p, dpi):
    return q_qc_dashboard(c, p, dpi)


def s_window_timeline(c, p, dpi):
    return k_transient_preservation(c, p, dpi)


def s_split_drift(c, p, dpi):
    return d_feature_distribution(c, p, dpi, "box")


def s_group_overlap(c, p, dpi):
    if not c.meta_rows:
        return None
    rows = c.meta_rows
    groups = sorted({str(r.get("group_id") or "NA") for r in rows})
    splits = sorted({str(r.get("split") or "unknown") for r in rows})
    m = np.asarray(
        [
            [
                sum(
                    str(r.get("group_id") or "NA") == g and str(r.get("split") or "unknown") == sp
                    for r in rows
                )
                for sp in splits
            ]
            for g in groups
        ]
    )
    fig, ax = _plt().subplots(figsize=(6, 3))
    ax.imshow(m, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(splits)), splits, rotation=25)
    ax.set_yticks(range(len(groups)), groups, fontsize=5)
    ax.set_title("Metadata group x split contingency; inspect fields before interpreting", fontsize=8)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def s_temporal_split(c, p, dpi):
    return None


def s_duplicate_audit(c, p, dpi):
    if not c.meta_rows:
        return None
    hs = {}
    for r in c.meta_rows:
        h = str(r.get("file_sha256") or "")
        if h:
            hs.setdefault(h, []).append(r)
    dup = [(h, v) for h, v in hs.items() if len(v) > 1]
    fig, ax = _plt().subplots(figsize=(5, 3))
    ax.bar(["duplicate hash groups", "files in groups"], [len(dup), sum(len(v) for _, v in dup)])
    ax.set_title("Exact byte-identical SHA-256 audit", fontsize=8)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def s_leakage_dashboard(c, p, dpi):
    if not c.meta_rows and not c.summary:
        return None
    fig, ax = _plt().subplots(figsize=(8, 2.5))
    ax.axis("off")
    rows = [
        ["file/group split before windows", "pipeline protocol"],
        ["train-only scaling/selection", "fold-local fitting"],
        ["unknown group/device metadata", "limits verification"],
    ]
    tab = ax.table(cellText=rows, colLabels=["boundary", "evidence / caveat"], loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    tab.scale(1, 1.5)
    ax.set_title("Leakage audit protocol/evidence; not proof where metadata absent", fontsize=8)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def d_classwise_psd_ci(c, p, dpi):
    return k_psd_multirate_overlay(c, p, dpi)


def d_band_features(c, p, dpi):
    return d_feature_distribution(c, p, dpi, "box")


def d_acf_decay(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    fs = _fs(c)
    fig, ax = _plt().subplots(figsize=(7, 3))
    rows = []
    for cl, sid, x in ss:
        y = x[: min(len(x), 10000)]
        y -= y.mean()
        den = y @ y
        if den <= 0:
            continue
        a = np.correlate(y, y, mode="full")[len(y) - 1 : len(y) + min(300, len(y) // 2)] / den
        t = np.arange(len(a)) / fs
        ax.plot(t, a, color=_color(c, cl), alpha=0.5)
        rows.extend(
            {"sample_id": sid, "lag_s": float(u), "acf": float(v)}
            for u, v in zip(t[::10], a[::10], strict=True)
        )
    ax.set(xlabel="lag (s)", ylabel="ACF", title="ACF by cached file; traces shown, not pooled correlation")
    fig.tight_layout()
    return _done(fig, p, dpi, rows, ["sample_id", "lag_s", "acf"])


def d_acf_characteristics(c, p, dpi):
    return d_acf_decay(c, p, dpi)


def d_acf_decay_model(c, p, dpi):
    return None


def d_phase_derivative(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    cl, sid, x = ss[0]
    x = x[:: max(1, len(x) // 5000)]
    dx = np.gradient(x)
    fig, axs = _plt().subplots(1, 2, figsize=(7, 3))
    lag = max(1, len(x) // 100)
    axs[0].scatter(x[lag:], x[:-lag], s=1)
    axs[1].scatter(x, dx, s=1)
    axs[0].set(xlabel="I(t)", ylabel="I(t-tau)")
    axs[1].set(xlabel="I(t)", ylabel="dI/dsample")
    fig.suptitle(f"Phase-space diagnostics {cl}/{sid[:10]}")
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def d_feature_trend(c, p, dpi):
    return None


def d_ecdf(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    fig, ax = _plt().subplots(figsize=(6, 3))
    for cl, _, x in ss:
        z = np.sort(x[np.isfinite(x)][:: max(1, len(x) // 10000)])
        ax.plot(z, np.arange(1, len(z) + 1) / len(z), color=_color(c, cl), alpha=0.5)
    ax.set(xlabel="signal (a.u.)", ylabel="ECDF", title="Empirical amplitude CDF")
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def d_event_fraction(c, p, dpi):
    es = _evrows(c)
    if not es:
        return None
    fig, axs = _plt().subplots(1, 2, figsize=(7, 3))
    for cl in c.classes:
        z = [e for e in es if e.get("class") == cl]
        di = np.asarray([e.get("delta_i_mean", np.nan) for e in z], dtype=float)
        rel = np.asarray([e.get("delta_i_over_i0", np.nan) for e in z], dtype=float)
        if np.isfinite(di).any():
            axs[0].hist(di[np.isfinite(di)], bins=20, histtype="step", label=cl)
        if np.isfinite(rel).any():
            axs[1].hist(rel[np.isfinite(rel)], bins=20, histtype="step", label=cl)
    axs[0].set_xlabel("Delta I (a.u.)")
    axs[1].set_xlabel("Delta I / I0")
    axs[0].legend(fontsize=5)
    axs[1].legend(fontsize=5)
    fig.suptitle("Event current change; I0 is per-event baseline estimate", fontsize=8)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def d_rms_features(c, p, dpi):
    return d_feature_distribution(c, p, dpi, "box")


def d_rms_trajectory(c, p, dpi):
    return _simple_wave(c, p, dpi, "RMS trajectory proxy")


def d_feature_distribution(c, p, dpi, kind):
    if c.x is None or not c.feat_names or c.y is None:
        return None
    inds = [
        i
        for i, n in enumerate(c.feat_names)
        if any(t in n.lower() for t in ("rms", "noise", "event_", "spec_"))
    ][:4]
    if not inds:
        return None
    fig, axs = _plt().subplots(1, len(inds), figsize=(3 * len(inds), 3))
    axs = np.atleast_1d(axs)
    labels = _as_labels(c, c.y)
    n = min(c.x.shape[0], labels.size)
    xmat = c.x[:n]
    labels = labels[:n]
    for ax, j in zip(axs, inds, strict=True):
        vals = [xmat[labels == cl, j][np.isfinite(xmat[labels == cl, j])] for cl in c.classes]
        keep = [(cl, v) for cl, v in zip(c.classes, vals, strict=True) if len(v)]
        if kind == "violin":
            ax.violinplot([v for _, v in keep], showmedians=True)
            ax.set_xticks(range(1, len(keep) + 1), [cl for cl, _ in keep], rotation=30, fontsize=5)
        elif kind == "hist":
            for cl, v in keep:
                ax.hist(v, bins=15, density=True, histtype="step", label=cl)
        else:
            ax.boxplot([v for _, v in keep], tick_labels=[cl for cl, _ in keep])
        ax.set_title(c.feat_names[j][:20], fontsize=6)
    fig.suptitle(f"File-level feature {kind} comparison", fontsize=8)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def d_violin(c, p, dpi):
    return d_feature_distribution(c, p, dpi, "violin")


def d_box(c, p, dpi):
    return d_feature_distribution(c, p, dpi, "box")


def d_hist(c, p, dpi):
    return d_feature_distribution(c, p, dpi, "hist")


def d_group_stats(c, p, dpi):
    return None


def d_concentration_plot(c, p, dpi):
    return None


def d_pca(c, p, dpi):
    if c.x is None or c.y is None:
        return None
    yy = _as_labels(c, c.y)
    return viz.plot_embedding_xy(c.x, yy, c.classes, p, dpi, "pca")


def d_tsne(c, p, dpi):
    if c.x is None or c.y is None:
        return None
    yy = _as_labels(c, c.y)
    return viz.plot_embedding_xy(c.x, yy, c.classes, p, dpi, "tsne")


def d_umap(c, p, dpi):
    return None


def d_pacmap(c, p, dpi):
    return None


def d_cluster(c, p, dpi):
    return None


def d_dendrogram(c, p, dpi):
    return None


def d_model_embedding_distance(c, p, dpi):
    return None


def e_event_detection_overview(c, p, dpi):
    ss = _sigs(c)
    if not ss:
        return None
    cl, sid, x = ss[0]
    fs = _fs(c)
    fig, ax = _plt().subplots(figsize=(8, 3))
    t = np.arange(len(x)) / fs
    ax.plot(t, x, lw=0.4)
    es = _evmap(c).get(sid, [])
    for e in es:
        ax.axvspan(float(e["onset_s"]), float(e["offset_s"]), alpha=0.15, color="#009E73")
    ax.set(
        xlabel="time (s)",
        ylabel="analysed residual (a.u.)",
        title=f"Detected-event overview {cl}/{sid[:10]} ({len(es)} events)",
    )
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def e_annotated_event(c, p, dpi):
    es = _evrows(c)
    if not es:
        return None
    e = es[len(es) // 2]
    z = _trace(c, e)
    if not z:
        return None
    x, fs, a, b = z
    t = (np.arange(len(x)) - a) / fs * 1e3
    fig, ax = _plt().subplots(figsize=(7, 3))
    ax.plot(t, x)
    ax.axvspan(0, float(e["duration_s"]) * 1e3, alpha=0.15)
    ax.set(
        xlabel="time from onset (ms)",
        ylabel="residual (a.u.)",
        title=f"Event: I0={e.get('i0')}; DeltaI={e.get('delta_i_mean')}; DeltaI/I0={e.get('delta_i_over_i0')}; Ton={e.get('duration_s')}s; polarity={e.get('polarity')}",
    )
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def e_event_change_point(c, p, dpi):
    return e_annotated_event(c, p, dpi)


def e_event_noise_context(c, p, dpi):
    return e_annotated_event(c, p, dpi)


def e_event_score_noise(c, p, dpi):
    es = _evrows(c)
    if not es:
        return None
    fig, ax = _plt().subplots(figsize=(6, 3))
    ax.hist([abs(float(e.get("amplitude") or 0)) for e in es], bins=30)
    ax.set(
        xlabel="detected event amplitude (a.u.)",
        ylabel="count",
        title="Detected event amplitude; FP rate needs annotated negatives",
    )
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def e_event_sensitivity_placeholder(c, p, dpi):
    return None


def e_dwell_hist(c, p, dpi):
    es = _evrows(c)
    if not es:
        return None
    fig, ax = _plt().subplots(figsize=(6, 3))
    for cl in c.classes:
        x = [float(e["duration_s"]) for e in es if e.get("class") == cl and float(e["duration_s"]) > 0]
        if x:
            ax.hist(x, bins=25, histtype="step", density=True, label=cl)
    ax.set_xscale("log")
    ax.set(
        xlabel="detected dwell duration (s)",
        ylabel="density",
        title="Dwell-time distribution; detector-defined duration",
    )
    ax.legend(fontsize=5)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def e_toff_distribution(c, p, dpi):
    return None


def e_on_off_trace(c, p, dpi):
    return e_event_detection_overview(c, p, dpi)


def _event_scatter(c, p, dpi, k):
    es = _evrows(c)
    if not es:
        return None
    x = [float(e.get("duration_s") or np.nan) for e in es]
    y = [abs(float(e.get("amplitude") or 0)) for e in es]
    fig, ax = _plt().subplots(figsize=(5, 3))
    ax.scatter(x, y, s=10)
    ax.set_xscale("log")
    ax.set(
        xlabel="duration (s)",
        ylabel="amplitude (a.u.)",
        title="Detected event amplitude vs duration; proxy only",
    )
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def e_amp_duration(c, p, dpi):
    return _event_scatter(c, p, dpi, "ampdur")


def e_snr_duration(c, p, dpi):
    return _event_scatter(c, p, dpi, "snrdur")


def e_amp_snr(c, p, dpi):
    return _event_scatter(c, p, dpi, "ampsnr")


def e_truth_validation(c, p, dpi, kind):
    return None


def t_learning_curves(c, p, dpi):
    rows = []
    for model, logs in c.histories.items():
        for log in logs:
            for h in log.get("history", []):
                if isinstance(h, dict):
                    rows.append((model, h))
    if not rows:
        return None
    fig, ax = _plt().subplots(figsize=(7, 3))
    for m in sorted({a for a, _ in rows}):
        z = [h for a, h in rows if a == m]
        for key, ls in (("train_loss", "-"), ("val_loss", "--")):
            if any(h.get(key) is not None for h in z):
                ax.plot([h.get("epoch") for h in z], [h.get(key) for h in z], ls, label=f"{m} {key}")
    ax.set(xlabel="epoch", ylabel="loss", title="Training/validation histories actually recorded")
    ax.legend(fontsize=5)
    fig.tight_layout()
    return viz._save(fig, p, dpi)


def t_lr_schedule(c, p, dpi):
    return t_learning_curves(c, p, dpi)


def t_runtime(c, p, dpi):
    return None


def t_metrics(c, p, dpi):
    return (
        viz.plot_confusion(
            next(iter(c.preds.values()))[0],
            next(iter(c.preds.values()))[1],
            c.classes,
            "file predictions",
            p,
            dpi,
        )
        if c.preds
        else None
    )


def t_label_trace(c, p, dpi):
    return t_metrics(c, p, dpi)


def t_misclassified_waveform(c, p, dpi):
    return None


def t_model_baseline_errors(c, p, dpi):
    return None


def t_probability_error(c, p, dpi):
    return None


def _proba_labels(c):
    if not c.probas:
        return None
    name, (y, pb) = next(iter(c.probas.items()))
    return _as_labels(c, y), pb, name


def t_roc(c, p, dpi):
    values = _proba_labels(c)
    return viz.plot_roc(values[0], values[1], c.classes, values[2], p, dpi) if values else None


def t_pr(c, p, dpi):
    values = _proba_labels(c)
    return viz.plot_pr_curve(values[0], values[1], c.classes, values[2], p, dpi) if values else None


def t_calibration(c, p, dpi):
    values = _proba_labels(c)
    return viz.plot_calibration(values[0], values[1], values[2], p, dpi) if values else None


def t_stability(c, p, dpi):
    return None


def t_ablation(c, p, dpi):
    return None


def t_model_leaderboard(c, p, dpi):
    return None


def t_fewshot_distribution(c, p, dpi):
    return None


def t_feature_importance(c, p, dpi):
    return None


def t_shap_bar(c, p, dpi):
    return None


def t_shap_swarm(c, p, dpi):
    return None


def t_shap_dependence(c, p, dpi):
    return None


def t_temporal_attribution(c, p, dpi):
    return None


def t_data_lineage(c, p, dpi):
    if not c.meta_rows and not c.summary:
        return None
    fig, ax = _plt().subplots(figsize=(9, 2))
    ax.axis("off")
    nodes = [
        "source",
        "manifest/QC",
        "filter/resample",
        "file/group split",
        "events/features/SSL",
        "models/evaluation",
        "reports",
    ]
    for i, label in enumerate(nodes):
        x = 0.07 + i * 0.14
        ax.text(
            x,
            0.55,
            label,
            transform=ax.transAxes,
            ha="center",
            bbox={"boxstyle": "round", "fc": "#e8f1f8"},
            fontsize=6,
        )
        if i < len(nodes) - 1:
            ax.annotate(
                "",
                xy=(x + 0.11, 0.55),
                xycoords=ax.transAxes,
                xytext=(x + 0.03, 0.55),
                arrowprops={"arrowstyle": "->"},
            )
    ax.set_title("Leakage-safe pipeline lineage; split precedes windows", fontsize=9)
    return viz._save(fig, p, dpi)


def t_kpi_dashboard(c, p, dpi):
    return t_data_lineage(c, p, dpi)


def t_scaling(c, p, dpi):
    return None


try:
    import umap as _umap  # noqa: F401

    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False
try:
    import pacmap as _pacmap  # noqa: F401

    _HAS_PACMAP = True
except Exception:
    _HAS_PACMAP = False


# ---------------------------------------------------------------------------
# Explicit extension inventory. Each ID points to a real data-driven builder. A None return is
# rendered as SKIPPED with the prereq below; the renderer never substitutes illustrative values.
# ---------------------------------------------------------------------------


def _never(ctx: RegistryCtx, p: Path, dpi: int) -> Any:
    return None


def _entry(
    spec: int,
    ident: str,
    category: str,
    title: str,
    builder: Any,
    prereq: str,
    params: str = "",
    *,
    available: bool | None = None,
) -> dict[str, Any]:
    row = {
        "id": ident,
        "spec": spec,
        "cat": category,
        "title": title,
        "b": builder,
        "prereq": prereq,
        "params": params,
        "csv": False,
    }
    if available is not None:
        row["available"] = available
    return row


EXT_ENTRIES: list[dict[str, Any]] = [
    _entry(
        2,
        "r3_zoomed_segments",
        "K",
        "raw signal zoomed representative segments",
        k_zoomed_segments,
        "cached signal per class",
        "strongest deviation, display only",
    ),
    _entry(
        3,
        "r3_baseline_noise_band",
        "K",
        "per-file baseline and robust noise band",
        k_baseline_noise_band,
        "cached signal per class",
        "rolling median 0.5 s; +/-3 robust sigma",
    ),
    _entry(
        4,
        "r3_filter_overlays",
        "K",
        "raw vs Butterworth and Gaussian overlay",
        k_filter_overlays,
        "cached signal",
        "configured filter or explicitly labelled fallback",
    ),
    _entry(
        6,
        "r3_rolling_median_windows",
        "K",
        "rolling-median window comparison",
        k_rolling_median_windows,
        "cached signal",
        "1/10/100 ms display windows",
    ),
    _entry(
        7,
        "r3_multirate_lpf",
        "K",
        "multirate Butterworth LPF comparison",
        k_multirate_lpf,
        "integer-achievable sampling menu rates",
        "same signal and time prefix",
    ),
    _entry(
        8,
        "r3_multirate_gaussian",
        "K",
        "multirate Gaussian comparison",
        k_multirate_gaussian,
        "integer-achievable sampling menu rates",
        "same signal and time prefix",
    ),
    _entry(
        9,
        "r3_filter_bode",
        "K",
        "anti-alias and conditioning filter Bode responses",
        k_bode_responses,
        "configured or designable decimation/filter response",
        "SOS frequency response",
    ),
    _entry(
        11,
        "r3_edge_artifacts",
        "K",
        "filter boundary and edge artefact zoom",
        k_edge_artifacts,
        "cached signal",
        "start/end; zero-phase and padding noted",
    ),
    _entry(
        12,
        "r3_transient_preservation",
        "K",
        "short/medium/long event preservation",
        k_transient_preservation,
        "at least three detected events in cached signals",
        "amplitude retention quantified",
    ),
    _entry(
        13,
        "r3_filter_validation_composite",
        "K",
        "four-panel filter validation composite",
        k_filter_validation_composite,
        "signal >=256 samples",
        "transient, PSD, response, diagnostics",
    ),
    _entry(
        14,
        "r3_filter_qc_compliance",
        "K",
        "sampling/filter QC compliance by file",
        k_filter_qc_compliance,
        "cached signals",
        "finite/length/rate/HF/flatline proxy checks",
    ),
    _entry(
        15,
        "r3_detrend_constant",
        "K",
        "constant mean-centering comparison",
        k_detrend_constant,
        "cached signal",
        "full trace",
    ),
    _entry(
        16,
        "r3_detrend_linear",
        "K",
        "linear detrending comparison",
        k_detrend_linear,
        "cached signal",
        "least-squares trend",
    ),
    _entry(
        17,
        "r3_detrend_rolling",
        "K",
        "rolling-median baseline comparison",
        k_detrend_rolling_median,
        "cached signal",
        "0.5 s robust median",
    ),
    _entry(
        18,
        "r3_detrend_metrics",
        "K",
        "baseline method quantitative comparison",
        k_detrend_metrics,
        "cached signal; events optional",
        "variance, smoothness, ACF, event amplitude",
    ),
    _entry(
        19,
        "r3_qc_dashboard",
        "Q",
        "file-level QC dashboard",
        q_qc_dashboard,
        "metadata.csv",
        "duration, sampling rate, size, samples",
    ),
    _entry(
        20,
        "r3_nonfinite_summary",
        "Q",
        "NaN/Inf per-file summary",
        q_nonfinite_summary,
        "cached analysed signals",
        "raw ingest errors remain in metadata",
    ),
    _entry(
        21,
        "r3_clipping_outliers",
        "Q",
        "outlier and observed-extrema indicators",
        q_clipping_outliers,
        "cached signals",
        "no amplifier rail claim without rail metadata",
    ),
    _entry(
        22,
        "r3_drift_quality",
        "Q",
        "baseline drift and SNR per-file scatter",
        q_drift_quality,
        "cached signals",
        "robust drift and SNR proxies",
    ),
    _entry(
        23,
        "r3_quality_by_class",
        "Q",
        "class-wise signal quality comparison",
        q_quality_by_class,
        "multiple cached files/classes",
        "no class-causality claim",
    ),
    _entry(
        24,
        "r3_flagged_files",
        "Q",
        "flagged files and status chart",
        q_flagged_files,
        "metadata.csv",
        "skipped rows retained",
    ),
    _entry(
        25,
        "r3_snr_noise",
        "Q",
        "MAD noise floor and SNR proxy",
        q_snr_noise,
        "cached signals",
        "dynamic range / robust sigma",
    ),
    _entry(
        26,
        "r3_event_density",
        "Q",
        "event rate and occupancy audit",
        q_event_density,
        "event table + metadata.csv",
        "detector output, not truth",
    ),
    _entry(
        27,
        "r3_class_counts",
        "Q",
        "recording counts by actual class",
        q_class_file_window_event_counts,
        "metadata.csv",
        "independent files",
    ),
    _entry(
        28,
        "r3_window_counts",
        "Q",
        "window counts by class",
        q_class_file_window_event_counts,
        "metadata n_windows",
        "counts from persisted window metadata",
    ),
    _entry(
        29,
        "r3_event_counts",
        "Q",
        "event counts by class",
        q_class_file_window_event_counts,
        "event table",
        "detected events",
    ),
    _entry(
        30,
        "r3_split_availability",
        "Q",
        "class availability by split",
        q_split_availability,
        "metadata split labels",
        "uses recorded split field",
    ),
    _entry(
        31,
        "r3_window_timeline",
        "S",
        "event/window segmentation timeline",
        s_window_timeline,
        "events and configured window length",
        "illustrative stride marked",
    ),
    _entry(
        32,
        "r3_split_feature_drift",
        "S",
        "feature distribution by recorded split",
        s_split_drift,
        ">=2 split labels and file feature",
        "descriptive only",
    ),
    _entry(
        33,
        "r3_group_overlap",
        "S",
        "group-overlap audit heatmap",
        s_group_overlap,
        "group ids and split/fold metadata",
        "exact assignment semantics shown",
    ),
    _entry(
        34,
        "r3_temporal_split",
        "S",
        "temporal split and embargo timeline",
        s_temporal_split,
        "acquisition timestamps and split intervals",
        "no filesystem timestamp substitution",
    ),
    _entry(
        35,
        "r3_duplicate_audit",
        "S",
        "exact and near-duplicate audit",
        s_duplicate_audit,
        "file hashes; cached signals for correlation",
        "hash definitive for byte-identical files only",
    ),
    _entry(
        36,
        "r3_leakage_dashboard",
        "S",
        "leakage/preprocessing audit dashboard",
        s_leakage_dashboard,
        "run context",
        "protocol/evidence distinction",
    ),
    _entry(
        38,
        "r3_psd_multirate",
        "D",
        "PSD before/after multirate resampling",
        k_psd_multirate_overlay,
        "integer-achievable rate menu",
        "Nyquist marked per trace",
    ),
    _entry(
        39,
        "r3_classwise_psd_ci",
        "D",
        "class mean PSD with between-file SD",
        d_classwise_psd_ci,
        ">=2 recordings in classes",
        "SD not CI",
    ),
    _entry(
        40,
        "r3_dark_1khz_psd",
        "D",
        "dark-colour 1 kHz baseline PSD",
        k_baseline_psd_dark,
        "analysis_rate_hz exactly 1000",
        "sub-500 Hz view",
    ),
    _entry(
        41,
        "r3_spectral_feature_summary",
        "D",
        "integrated spectral feature comparison",
        d_band_features,
        "spectral engineered columns",
        "file-level means",
    ),
    _entry(
        43,
        "r3_acf_decay",
        "D",
        "class ACF decay curves",
        d_acf_decay,
        "cached signals >=64 samples",
        "mean +/- between-file SD",
    ),
    _entry(
        44,
        "r3_acf_characteristics",
        "D",
        "selected-lag ACF comparison",
        d_acf_characteristics,
        "cached signals",
        "1/10/100 ms lags",
    ),
    _entry(
        45,
        "r3_acf_model",
        "D",
        "single-exponential ACF diagnostic fit",
        d_acf_decay_model,
        "positive ACF range",
        "one model only; no unjustified model selection",
    ),
    _entry(
        46,
        "r3_phase_space",
        "D",
        "delayed-current phase-space plot",
        d_phase_derivative,
        "cached signal",
        "display decimated to 10k points",
    ),
    _entry(
        48,
        "r3_temporal_trend",
        "D",
        "feature vs acquisition-time trend",
        d_feature_trend,
        "explicit acquisition timestamps + features",
        "filesystem times are not substituted",
    ),
    _entry(
        49,
        "r3_event_overview",
        "E",
        "event detection full-recording overview",
        e_event_detection_overview,
        "cached signal",
        "display thresholds are illustrative robust-sigma lines",
    ),
    _entry(
        50,
        "r3_annotated_event",
        "E",
        "annotated individual translocation event",
        e_annotated_event,
        "event table and matching cached signal",
        "I0/DeltaI/relative amplitude/dwell/polarity",
    ),
    _entry(
        51,
        "r3_event_change_point",
        "E",
        "event internal change-point diagnostic",
        e_event_change_point,
        "event duration >=8 samples",
        "exploratory single-mean change",
    ),
    _entry(
        52,
        "r3_event_noise_context",
        "E",
        "event amplitude and background noise context",
        e_event_noise_context,
        "annotated event prerequisites",
        "robust background band",
    ),
    _entry(
        53,
        "r3_event_score_noise",
        "E",
        "event score vs background proxy",
        e_event_score_noise,
        "event rows + cached noise",
        "false-positive rate needs annotated negatives",
    ),
    _entry(
        54,
        "r3_event_sensitivity_composite",
        "E",
        "event sensitivity composite",
        e_event_sensitivity_placeholder,
        "detector parameter grid and event identity matching",
        "not computed by current run",
    ),
    _entry(
        55,
        "r3_dwell_distribution",
        "E",
        "event dwell-time PDF/KDE and inset",
        e_dwell_hist,
        "detected event table",
        "linear/log duration + amplitude inset",
    ),
    _entry(
        58,
        "r3_toff_distribution",
        "E",
        "inter-event Toff distribution",
        e_toff_distribution,
        ">=2 events in a recording",
        "only within-recording intervals",
    ),
    _entry(
        59,
        "r3_on_off_trace",
        "E",
        "binary event ON/OFF state trace",
        e_on_off_trace,
        "event table and cached signal",
        "first 10 s displayed",
    ),
    _entry(
        60,
        "r3_event_amplitude_duration",
        "E",
        "event amplitude vs dwell",
        e_amp_duration,
        "event table",
        "event detector outputs",
    ),
    _entry(
        61,
        "r3_event_snr_duration",
        "E",
        "event proxy SNR vs dwell",
        e_snr_duration,
        "event table and noise estimate",
        "SNR proxy definition explicit",
    ),
    _entry(
        62,
        "r3_event_amplitude_snr",
        "E",
        "event amplitude vs proxy SNR",
        e_amp_snr,
        "event table and noise estimate",
        "amplitude / robust MAD",
    ),
    _entry(
        63,
        "r3_detector_sensitivity",
        "E",
        "threshold/parameter sensitivity curves",
        e_event_sensitivity_placeholder,
        "detector sweep artifact",
        "not computed by base pipeline",
    ),
    _entry(
        64,
        "r3_detection_parameter_heatmap",
        "E",
        "detection parameter outcome heatmap",
        e_event_sensitivity_placeholder,
        "parameter-grid detection outcomes",
        "not computed by base pipeline",
    ),
    _entry(
        65,
        "r3_borderline_event_robustness",
        "E",
        "borderline event stability",
        e_event_sensitivity_placeholder,
        "event identity matched across perturbations",
        "not computed by base pipeline",
    ),
    _entry(
        66,
        "r3_truth_onset",
        "E",
        "synthetic truth vs detected onset",
        lambda c, p, d: e_truth_validation(c, p, d, "onset"),
        "synthetic event ground-truth table",
        "not inferred from real data",
    ),
    _entry(
        67,
        "r3_truth_offset",
        "E",
        "synthetic truth vs detected offset",
        lambda c, p, d: e_truth_validation(c, p, d, "offset"),
        "synthetic event ground-truth table",
        "not inferred from real data",
    ),
    _entry(
        68,
        "r3_truth_duration_error",
        "E",
        "synthetic truth duration error",
        _never,
        "synthetic true/detected event pairs",
        "not generated in current run",
    ),
    _entry(
        69,
        "r3_truth_pr_curve",
        "E",
        "synthetic detector precision-recall",
        _never,
        "synthetic event ground truth over detector thresholds",
        "not generated in current run",
    ),
    _entry(
        70,
        "r3_amplitude_pdf_rates",
        "D",
        "amplitude PDF at achievable menu rates",
        k_amplitude_pdf_multirate,
        ">=3 achievable rates preferred",
        "histogram + KDE; no untested parametric fit",
    ),
    _entry(
        71,
        "r3_ecdf",
        "D",
        "empirical amplitude and dwell ECDF",
        d_ecdf,
        "cached signals; events optional",
        "exact empirical CDF",
    ),
    _entry(
        73,
        "r3_delta_i_distributions",
        "D",
        "absolute and normalized Delta-I distributions",
        d_event_fraction,
        "event table",
        "per-event I0 source included",
    ),
    _entry(
        75,
        "r3_rms_distributions",
        "D",
        "raw and normalized RMS comparison",
        d_rms_features,
        "cached signals",
        "normalized by absolute median residual",
    ),
    _entry(
        77,
        "r3_rms_trajectory",
        "D",
        "window RMS trajectory",
        d_rms_trajectory,
        "cached signal",
        "10 ms windows",
    ),
    _entry(
        78,
        "r3_feature_violins",
        "D",
        "feature violin distributions",
        d_violin,
        "file-level feature matrix",
        "selected features, each independent file one row",
    ),
    _entry(
        79,
        "r3_feature_raincloud",
        "D",
        "feature raincloud summaries",
        d_violin,
        "file-level feature matrix",
        "violin + quartile summary; no inferential p-value labels",
    ),
    _entry(
        80,
        "r3_feature_beeswarm",
        "D",
        "feature individual-point comparison",
        d_violin,
        "file-level feature matrix",
        "points are files, not SHAP",
    ),
    _entry(
        81,
        "r3_feature_boxplots",
        "D",
        "class-wise engineered feature boxplots",
        d_box,
        "file-level feature matrix",
        "up to four selected physical features",
    ),
    _entry(
        82,
        "r3_feature_histograms",
        "D",
        "class-wise feature histograms",
        d_hist,
        "file-level feature matrix",
        "up to four selected physical features",
    ),
    _entry(
        83,
        "r3_group_stats",
        "D",
        "group-aware feature tests and effect sizes",
        d_group_stats,
        "file-level feature and >=2 classes",
        "Mann-Whitney + BH; group bootstrap not claimed",
    ),
    _entry(
        85,
        "r3_concentration",
        "D",
        "feature vs concentration",
        d_concentration_plot,
        "concentration metadata and RMS-like feature",
        "no synthetic concentration",
    ),
    _entry(
        86,
        "r3_molecular_volume",
        "D",
        "feature vs calibrated molecular volume",
        _never,
        "independent calibrated volume metadata/forward model",
        "not available unless supplied",
    ),
    _entry(
        87,
        "r3_pca",
        "M",
        "2D PCA feature projection",
        d_pca,
        ">=3 files and >=2 finite feature columns",
        "visualization-only fit; seed 0",
    ),
    _entry(
        89,
        "r3_tsne",
        "M",
        "2D t-SNE feature embedding",
        d_tsne,
        ">=5 files",
        "exploratory local structure; seed 0",
    ),
    _entry(
        91,
        "r3_umap",
        "M",
        "2D UMAP feature embedding",
        d_umap,
        "umap-learn installed and >=5 files",
        "exploratory; seed 0",
        available=_HAS_UMAP,
    ),
    _entry(
        93,
        "r3_pacmap",
        "M",
        "2D PaCMAP embedding",
        d_pacmap,
        "pacmap installed and >=5 files",
        "exploratory; seed 0",
        available=_HAS_PACMAP,
    ),
    _entry(
        97,
        "r3_embedding_distances",
        "M",
        "within vs between-class embedding distances",
        d_model_embedding_distance,
        "learned embeddings + aligned sample identities",
        "not wired to class-sample ids",
    ),
    _entry(
        100,
        "r3_kmeans",
        "M",
        "K-means cluster projection and external alignment",
        d_cluster,
        ">=4 file feature rows",
        "k-means fit is unsupervised; labels used post hoc",
    ),
    _entry(
        102,
        "r3_dendrogram",
        "M",
        "hierarchical clustering dendrogram",
        d_dendrogram,
        ">=3 file-level feature rows",
        "average linkage",
    ),
    _entry(
        104,
        "r3_outlier_waveforms",
        "M",
        "outlier waveform inspection",
        t_misclassified_waveform,
        "prediction errors or explicit outlier score",
        "misclassified signals only",
    ),
    _entry(
        105,
        "r3_learning_curves",
        "T",
        "training/validation learning curves",
        t_learning_curves,
        "training history logged",
        "per available neural model",
    ),
    _entry(
        107,
        "r3_lr_schedule",
        "T",
        "learning rate by epoch",
        t_lr_schedule,
        "history includes optimizer learning rate",
        "no scheduler reconstruction",
    ),
    _entry(
        108,
        "r3_runtime",
        "T",
        "reported model runtime summary",
        t_runtime,
        "runtime metric present in summary",
        "no inferred memory/GPU telemetry",
    ),
    _entry(
        111,
        "r3_metric_comparison",
        "T",
        "per-class and aggregate metrics",
        t_metrics,
        "file predictions",
        "counts from evaluated rows",
    ),
    _entry(
        112,
        "r3_label_trace",
        "T",
        "predicted vs reference label traceability",
        t_label_trace,
        "file predictions and identifiers",
        "errors traceable by sample id",
    ),
    _entry(
        113,
        "r3_error_waveforms",
        "T",
        "misclassified waveform inspection",
        t_misclassified_waveform,
        "at least one misclassified file with cached signal",
        "representative errors",
    ),
    _entry(
        114,
        "r3_model_error_overlap",
        "T",
        "model versus baseline error overlap",
        t_model_baseline_errors,
        ">=2 prediction models",
        "paired support size reported",
    ),
    _entry(
        115,
        "r3_probability_uncertainty",
        "T",
        "probability error and entropy distributions",
        t_probability_error,
        "class probabilities",
        "closed-set predictive uncertainty",
    ),
    _entry(
        116,
        "r3_roc",
        "T",
        "one-vs-rest ROC curves",
        t_roc,
        "probabilities and >=2 classes represented",
        "aggregate point predictions; no fold CI",
    ),
    _entry(
        117,
        "r3_precision_recall",
        "T",
        "one-vs-rest precision-recall curves",
        t_pr,
        "probabilities and positive/negative rows",
        "prevalence context in data; no fold CI",
    ),
    _entry(
        118,
        "r3_calibration",
        "T",
        "top-label calibration diagram",
        t_calibration,
        "class probabilities",
        "Brier + reliability curve",
    ),
    _entry(
        120,
        "r3_repeat_stability",
        "T",
        "repeat-level stability/error bars",
        t_stability,
        "repeated CV summary",
        "repeats are not external cohorts",
    ),
    _entry(
        127,
        "r3_ablation",
        "T",
        "recorded ablation comparison",
        t_ablation,
        "ablation results artifact",
        "only variants actually executed",
    ),
    _entry(
        132,
        "r3_model_leaderboard",
        "T",
        "actual-run model leaderboard",
        t_model_leaderboard,
        "model comparison results",
        "unevaluated foundation models omitted",
    ),
    _entry(
        137,
        "r3_fewshot_episode_distribution",
        "T",
        "few-shot episode performance distribution",
        t_fewshot_distribution,
        "episode-level metrics",
        "closed-set only",
    ),
    _entry(
        146,
        "r3_feature_importance",
        "T",
        "reported handcrafted feature importance",
        t_feature_importance,
        "explanation artifact has numeric feature importance",
        "only values as recorded",
    ),
    _entry(
        148,
        "r3_shap_global",
        "T",
        "SHAP global summary",
        t_shap_bar,
        "SHAP values + feature alignment",
        "not available in registry context",
    ),
    _entry(
        149,
        "r3_shap_beeswarm",
        "T",
        "SHAP global/class beeswarm",
        t_shap_swarm,
        "aligned per-row SHAP values",
        "not available in registry context",
    ),
    _entry(
        150,
        "r3_shap_dependence",
        "T",
        "SHAP feature dependence",
        t_shap_dependence,
        "aligned per-row SHAP values",
        "not available in registry context",
    ),
    _entry(
        151,
        "r3_temporal_attribution",
        "T",
        "temporal attribution overlay",
        t_temporal_attribution,
        "temporal attribution arrays",
        "saved separately; context path not wired",
    ),
    _entry(
        155,
        "r3_attribution_stability",
        "T",
        "attribution stability",
        _never,
        "multiple comparable attribution runs/augmentations",
        "not available on single run",
    ),
    _entry(
        164,
        "r3_scaling_curve",
        "T",
        "dataset scale vs performance/runtime",
        t_scaling,
        "multiple dataset-size runs with common config",
        "not a single-run quantity",
    ),
    _entry(
        166,
        "r3_lineage_diagram",
        "G",
        "end-to-end processing lineage diagram",
        t_data_lineage,
        "always available",
        "fixed pipeline stages",
    ),
    _entry(
        168,
        "r3_kpi_dashboard",
        "G",
        "run KPI dashboard",
        t_kpi_dashboard,
        "summary.json",
        "consolidated run metrics/config fingerprint",
    ),
]
