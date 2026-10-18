"""Label-blind per-recording features for Model A.

Every function here sees only the signal, its baseline, the detected events, and the
configuration. Labels never enter this module. Missing values (for example event
statistics on a recording with no events) are returned as NaN; imputation happens
inside the modelling pipeline, fitted on training folds only.

Feature groups
--------------
The group of a feature is its name prefix, so a group can be switched off or ablated without
touching any other code (``training.feature_groups``).

    noise   robust residual sigma, raw level and spread
    time    temporal pattern: zero crossings, Hjorth parameters, line length, Teager energy,
            permutation entropy, segment-to-segment variability, drift
    amp     amplitude domain: RMS, peak, peak-to-peak, crest factor, percentiles, dynamic range
    event   event statistics: count, rate, duration, dip depth, amplitude
    ctx     context and temporal dependencies: inter-event intervals, burstiness, onset
            distribution, rate drift, amplitude-duration coupling, baseline stability, I0
    spec    frequency domain: Welch band powers, spectral slope, peak, centroid, spread,
            entropy, roll-off, peak-to-median (harmonicity)
    tf      time-frequency: STFT flux, per-band energy drift, frame entropy, and dyadic (db2)
            wavelet level energies
    acf     normalised autocorrelation of the residual at the configured lags
    shape   distribution shape: skewness, excess kurtosis, tail fraction
    morph   waveform morphology: turning-point rate, event rise/fall times, area-to-peak,
            sharpness, self-similarity to the recording's own mean event waveform

Design rules kept here
----------------------
* Scale-sensitive quantities are expressed as ratios to the recording's own robust sigma or to
  ``|I0|``, so files recorded at different gains stay comparable.
* Anything derived from a *class* mean, another file, or the whole pool would leak; the only
  cross-event reference used is the mean event waveform **of the same recording**.
* A degenerate input (no events, flat signal, two samples) yields NaN, never an exception: the
  file is still assessed and the reason is reported elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import signal as sps
from scipy import stats as sst

from .events import Event

WELCH_NPERSEG = 8192  # 0.82 s at 10 kHz; about 1.2 Hz resolution
STFT_NPERSEG = 4096   # 0.41 s at 10 kHz; time-frequency resolution trade-off
WAVELET_LEVELS = 4
# PyWavelets 'db2' decomposition filters, inlined so no new dependency is needed.
DB2_DEC_LO = np.array([-0.12940952, 0.22414387, 0.83651630, 0.48296291])
DB2_DEC_HI = np.array([0.48296291, -0.83651630, 0.22414387, 0.12940952])

FEATURE_GROUPS: tuple[str, ...] = (
    "noise", "time", "amp", "event", "ctx", "spec", "tf", "acf", "shape", "morph",
)

GROUP_DESCRIPTIONS: dict[str, str] = {
    "noise": "robust residual sigma, raw level and spread",
    "time": "temporal pattern: zero crossings, Hjorth, line length, Teager energy, permutation "
            "entropy, segment variability, drift",
    "amp": "amplitude domain: RMS, peak, peak-to-peak, crest factor, percentiles, dynamic range",
    "event": "event statistics: count, rate, duration, dip depth, amplitude",
    "ctx": "context: inter-event intervals, burstiness, onset distribution, rate drift, "
           "amplitude-duration coupling, baseline stability, reference level",
    "spec": "frequency domain: Welch band powers, slope, peak, centroid, spread, entropy, "
            "roll-off, harmonicity",
    "tf": "time-frequency: STFT flux, per-band energy drift, frame entropy, db2 wavelet levels",
    "acf": "normalised autocorrelation of the residual at the configured lags",
    "shape": "distribution shape: skewness, excess kurtosis, tail fraction",
    "morph": "morphology: turning points, event rise/fall, area-to-peak, sharpness, "
             "self-similarity to this recording's mean event waveform",
}


def _log10(v: float) -> float:
    return math.log10(v) if (v is not None and np.isfinite(v) and v > 0) else float("nan")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if (b not in (0, 0.0) and np.isfinite(a) and np.isfinite(b)) else float("nan")


def _nan_stats(prefix: str, values: np.ndarray, out: dict[str, float]) -> None:
    if values.size == 0:
        for k in ("mean", "median", "sd", "max"):
            out[f"{prefix}_{k}"] = float("nan")
        return
    out[f"{prefix}_mean"] = float(np.mean(values))
    out[f"{prefix}_median"] = float(np.median(values))
    out[f"{prefix}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    out[f"{prefix}_max"] = float(np.max(values))


# --------------------------------------------------------------------------- #
# Event statistics and event context
# --------------------------------------------------------------------------- #

def event_features(events: Sequence[Event], n_samples: int, fs: float, i0_level: float) -> dict[str, float]:
    """Event statistics. Duration and dip features are NaN when there are no events."""
    total_s = n_samples / fs if fs > 0 else float("nan")
    n = len(events)
    out: dict[str, float] = {
        "event_n": float(n),
        "event_has_any": float(n > 0),
        "event_rate_per_min": (n / (total_s / 60.0)) if total_s and total_s > 0 else float("nan"),
    }
    dur = np.array([e.duration_s for e in events], dtype=float)
    dip = np.array([e.delta_i_over_i0 for e in events], dtype=float)
    dip = dip[np.isfinite(dip)]
    amp = np.array([e.amplitude for e in events], dtype=float)
    # Relative amplitude uses the recording's baseline level, so it is comparable across files.
    amp_rel = amp / abs(i0_level) if (i0_level != 0.0 and np.isfinite(i0_level)) else np.full(amp.shape, np.nan)
    if n:
        out["event_log10_dur_median"] = float(np.log10(max(np.median(dur), 1e-9)))
        out["event_log10_dur_mean"] = float(np.log10(max(np.mean(dur), 1e-9)))
        out["event_dur_frac_of_time"] = float(np.sum(dur) / total_s) if total_s else float("nan")
        out["event_dur_cv"] = _safe_div(float(np.std(dur, ddof=1)) if n > 1 else 0.0,
                                        float(np.mean(dur)))
    else:
        out["event_log10_dur_median"] = float("nan")
        out["event_log10_dur_mean"] = float("nan")
        out["event_dur_frac_of_time"] = 0.0
        out["event_dur_cv"] = float("nan")
    _nan_stats("event_dip", dip, out)
    _nan_stats("event_amp_rel", amp_rel[np.isfinite(amp_rel)], out)
    return out


def event_context_features(events: Sequence[Event], n_samples: int, fs: float) -> dict[str, float]:
    """Timing, co-occurrence, and temporal dependencies between events.

    These are the features that separate a recording with regularly spaced events from one with
    bursts, which the plain count and rate cannot express. All of them are NaN with fewer than
    two events, and the interval features need three events for a spread.
    """
    total_s = n_samples / fs if fs > 0 else float("nan")
    n = len(events)
    out: dict[str, float] = {}
    onsets = np.array([e.onset_s for e in events], dtype=float)
    dur = np.array([e.duration_s for e in events], dtype=float)
    amp = np.array([e.amplitude for e in events], dtype=float)

    if n == 0 or not total_s or not np.isfinite(total_s):
        for k in ("isi_log10_median_s", "isi_cv", "burst_frac", "max_burst_len",
                  "first_onset_frac", "last_onset_frac", "onset_sd_frac", "onset_rayleigh_r",
                  "rate_half_ratio", "amp_dur_spearman"):
            out[f"ctx_{k}"] = float("nan")
        return out

    out["ctx_first_onset_frac"] = float(onsets.min() / total_s)
    out["ctx_last_onset_frac"] = float(onsets.max() / total_s)
    out["ctx_onset_sd_frac"] = _safe_div(float(np.std(onsets, ddof=1)) if n > 1 else 0.0, total_s)
    # Rayleigh's R for uniformity of onset phase over the recording: near 0 = evenly spread,
    # near 1 = clustered in time. Label-blind and bounded, so it is safe as a feature.
    if n > 1:
        phase = 2.0 * np.pi * (onsets / total_s)
        out["ctx_onset_rayleigh_r"] = float(
            abs(np.sum(np.exp(1j * phase))) / n)
    else:
        out["ctx_onset_rayleigh_r"] = float("nan")

    half = total_s / 2.0
    n_first = int(np.count_nonzero(onsets < half))
    n_second = n - n_first
    out["ctx_rate_half_ratio"] = _safe_div(float(n_second), float(n_first)) if n_first else float("nan")

    if n >= 2:
        isi = np.diff(np.sort(onsets))
        med = float(np.median(isi))
        out["ctx_isi_log10_median_s"] = _log10(med)
        out["ctx_isi_cv"] = _safe_div(float(np.std(isi, ddof=1)) if isi.size > 1 else 0.0,
                                        float(np.mean(isi))) if isi.size and np.mean(isi) > 0 else float("nan")
        # A burst is an interval below half the typical interval; the longest run of them says how
        # much of the recording is spent in clustered activity.
        burst = isi < 0.5 * med if med > 0 else np.zeros(isi.shape, dtype=bool)
        out["ctx_burst_frac"] = float(np.mean(burst)) if burst.size else float("nan")
        best = run = 0
        for b in burst.tolist():
            run = run + 1 if b else 0
            best = max(best, run)
        out["ctx_max_burst_len"] = float(best)
    else:
        for k in ("isi_log10_median_s", "isi_cv", "burst_frac", "max_burst_len"):
            out[f"ctx_{k}"] = float("nan")

    if n >= 3 and np.std(amp) > 0 and np.std(dur) > 0:
        rho = sst.spearmanr(amp, dur).statistic
        out["ctx_amp_dur_spearman"] = float(rho) if np.isfinite(rho) else float("nan")
    else:
        out["ctx_amp_dur_spearman"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Noise, time domain, amplitude domain
# --------------------------------------------------------------------------- #

def residual_sigma(residual: np.ndarray) -> float:
    """Robust (MAD-based) standard deviation of the residual. Same definition as the detector."""
    med = np.median(residual)
    mad = np.median(np.abs(residual - med))
    return float(1.4826 * mad)


def _hjorth(x: np.ndarray) -> tuple[float, float, float]:
    """Activity, mobility, complexity. Defined on the residual; scale-free after normalisation."""
    d1 = np.diff(x)
    activity = float(np.var(x))
    if activity <= 0:
        return activity, float("nan"), float("nan")
    mob = math.sqrt(float(np.var(d1)) / activity) if d1.size else float("nan")
    if d1.size < 2 or not np.isfinite(mob) or mob <= 0:
        return activity, mob, float("nan")
    d2 = np.diff(d1)
    mob_d = math.sqrt(float(np.var(d2)) / float(np.var(d1))) if np.var(d1) > 0 else float("nan")
    return activity, mob, _safe_div(mob_d, mob)


def _permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1, max_points: int = 20_000) -> float:
    """Normalised permutation entropy: 1 = as random as a shuffle, near 0 = a fixed pattern."""
    if x.size < order * delay + 1:
        return float("nan")
    step = max(1, x.size // max_points)
    y = x[::step]
    n_pat = y.size - (order - 1) * delay
    if n_pat <= 1:
        return float("nan")
    idx = np.arange(order) * delay
    patterns = np.lib.stride_tricks.sliding_window_view(y, (order - 1) * delay + 1)[:, idx]
    ranks = np.argsort(patterns, axis=1, kind="stable")
    keys = np.array([hash(tuple(r.tolist())) for r in ranks])
    _, counts = np.unique(keys, return_counts=True)
    p = counts / counts.sum()
    h = float(-np.sum(p * np.log2(p)))
    return h / math.log2(math.factorial(order))


def time_domain_features(residual: np.ndarray, fs: float, sigma: float) -> dict[str, float]:
    """Temporal structure of the residual, independent of any event detection."""
    n = residual.size
    out: dict[str, float] = {}
    if n < 4:
        for k in ("zcr_per_sample", "hjorth_mobility", "hjorth_complexity", "line_length_norm",
                  "teager_log10", "perm_entropy_norm", "seg_sd_ratio", "trend_slope_norm",
                  "frac_above_mean"):
            out[f"time_{k}"] = float("nan")
        return out
    d1 = np.diff(residual)
    signs = np.sign(residual)
    out["time_zcr_per_sample"] = float(np.count_nonzero(signs[1:] != signs[:-1]) / max(n - 1, 1))
    _, mob, comp = _hjorth(residual)
    out["time_hjorth_mobility"] = mob
    out["time_hjorth_complexity"] = comp
    denom = sigma if sigma > 0 else float("nan")
    out["time_line_length_norm"] = _safe_div(float(np.mean(np.abs(d1))), denom)
    if n >= 3:
        tk = residual[1:-1] ** 2 - residual[:-2] * residual[2:]
        out["time_teager_log10"] = _log10(float(np.mean(np.abs(tk))) / (denom ** 2)) \
            if np.isfinite(denom) else float("nan")
    else:
        out["time_teager_log10"] = float("nan")
    out["time_perm_entropy_norm"] = _permutation_entropy(residual)

    n_seg = 10
    bounds = np.linspace(0, n, n_seg + 1).astype(int)
    seg_sd = np.array([float(np.std(residual[a:b], ddof=1)) for a, b in zip(bounds[:-1], bounds[1:],
                                                                             strict=True) if b - a > 2])
    if seg_sd.size >= 2 and np.mean(seg_sd) > 0:
        out["time_seg_sd_ratio"] = float(np.std(seg_sd, ddof=1) / np.mean(seg_sd))
    else:
        out["time_seg_sd_ratio"] = float("nan")

    t = np.arange(n, dtype=float) / fs if fs > 0 else np.arange(n, dtype=float)
    slope = float(np.polyfit(t, residual, 1)[0]) if n > 2 else float("nan")
    out["time_trend_slope_norm"] = _safe_div(slope, denom) if np.isfinite(slope) else float("nan")
    out["time_frac_above_mean"] = float(np.mean(residual > float(np.mean(residual))))
    return out


def amplitude_features(residual: np.ndarray, sigma: float) -> dict[str, float]:
    """Amplitude-domain descriptors, normalised by the recording's own robust sigma."""
    out: dict[str, float] = {}
    a = np.abs(residual)
    if a.size == 0:
        for k in ("log10_rms", "log10_peak", "log10_p2p", "crest_factor", "abs_p50_over_sigma",
                  "abs_p95_over_sigma", "abs_p99_over_sigma", "iqr_over_sigma",
                  "dynamic_range_db"):
            out[f"amp_{k}"] = float("nan")
        return out
    rms = float(np.sqrt(np.mean(residual ** 2)))
    peak = float(a.max())
    p2p = float(residual.max() - residual.min())
    out["amp_log10_rms"] = _log10(rms)
    out["amp_log10_peak"] = _log10(peak)
    out["amp_log10_p2p"] = _log10(p2p)
    out["amp_crest_factor"] = _safe_div(peak, rms)
    q = np.percentile(a, [25, 50, 75, 95, 99]) if a.size >= 5 else None
    if q is not None and sigma > 0:
        out["amp_abs_p50_over_sigma"] = float(q[1] / sigma)
        out["amp_abs_p95_over_sigma"] = float(q[3] / sigma)
        out["amp_abs_p99_over_sigma"] = float(q[4] / sigma)
        out["amp_iqr_over_sigma"] = float((q[2] - q[0]) / sigma)
        out["amp_dynamic_range_db"] = float(20.0 * math.log10(q[4] / q[1])) if q[1] > 0 and q[4] > 0 \
            else float("nan")
    else:
        for k in ("abs_p50_over_sigma", "abs_p95_over_sigma", "abs_p99_over_sigma",
                  "iqr_over_sigma", "dynamic_range_db"):
            out[f"amp_{k}"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Frequency domain and time-frequency
# --------------------------------------------------------------------------- #

def welch_psd(x: np.ndarray, fs: float, nperseg: int = WELCH_NPERSEG) -> tuple[np.ndarray, np.ndarray]:
    """One Welch estimate, shared by the band powers and the spectral-shape features."""
    n = min(nperseg, x.size)
    f, pxx = sps.welch(x, fs=fs, nperseg=max(n, 8), detrend="constant")
    return np.asarray(f, dtype=float), np.asarray(pxx, dtype=float)


def welch_band_powers(
    x: np.ndarray, fs: float, bands_hz: Sequence[Sequence[float]],
    psd: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """log10 of the power in each band (integrated PSD), plus total power and log-log slope."""
    f, pxx = psd if psd is not None else welch_psd(x, fs)
    out: dict[str, float] = {}
    for lo, hi in bands_hz:
        m = (f >= lo) & (f <= hi)
        p = float(np.trapezoid(pxx[m], f[m])) if np.count_nonzero(m) >= 2 else 0.0
        out[f"spec_log10_band_{_band_tag(lo, hi)}"] = math.log10(p) if p > 0 else float("nan")
    total = float(np.trapezoid(pxx[f > 0], f[f > 0]))
    out["spec_log10_total"] = math.log10(total) if total > 0 else float("nan")
    fit = (f >= 1.0) & (f <= 1000.0) & (pxx > 0)
    if np.count_nonzero(fit) >= 3:
        slope = np.polyfit(np.log10(f[fit]), np.log10(pxx[fit]), 1)[0]
        out["spec_loglog_slope_1_1000"] = float(slope)
    else:
        out["spec_loglog_slope_1_1000"] = float("nan")
    return out


def spectral_shape_features(f: np.ndarray, pxx: np.ndarray) -> dict[str, float]:
    """Distribution of power over frequency: where it peaks, how wide it is, how ordered."""
    out: dict[str, float] = {}
    keys = ("log10_peak_hz", "log10_centroid_hz", "log10_spread_hz", "entropy_norm",
            "log10_rolloff85_hz", "log10_peak_to_median")
    m = (f > 0) & (pxx > 0) & np.isfinite(pxx)
    if np.count_nonzero(m) < 3:
        for k in keys:
            out[f"spec_{k}"] = float("nan")
        return out
    ff, pp = f[m], pxx[m]
    out["spec_log10_peak_hz"] = _log10(float(ff[int(np.argmax(pp))]))
    total = float(np.sum(pp))
    centroid = float(np.sum(ff * pp) / total) if total > 0 else float("nan")
    out["spec_log10_centroid_hz"] = _log10(centroid)
    if np.isfinite(centroid) and total > 0:
        spread = math.sqrt(float(np.sum(pp * (ff - centroid) ** 2) / total))
        out["spec_log10_spread_hz"] = _log10(spread)
    else:
        out["spec_log10_spread_hz"] = float("nan")
    p = pp / total if total > 0 else pp
    nz = p > 0
    h = float(-np.sum(p[nz] * np.log2(p[nz])))
    out["spec_entropy_norm"] = h / math.log2(p.size) if p.size > 1 else float("nan")
    cum = np.cumsum(pp) / total if total > 0 else np.cumsum(pp)
    idx = int(np.searchsorted(cum, 0.85))
    out["spec_log10_rolloff85_hz"] = _log10(float(ff[min(idx, ff.size - 1)]))
    med = float(np.median(pp))
    out["spec_log10_peak_to_median"] = _log10(float(pp.max()) / med) if med > 0 else float("nan")
    return out


def _band_tag(lo: float, hi: float) -> str:
    def fmt(v: float) -> str:
        return f"{v:g}".replace(".", "p").replace("-", "m")
    return f"{fmt(lo)}_{fmt(hi)}Hz"


def _dyadic_wavelet_energies(x: np.ndarray, levels: int = WAVELET_LEVELS) -> list[float]:
    """Energy of each db2 detail level, plus the approximation, from a manual filter bank."""
    energies: list[float] = []
    a = np.asarray(x, dtype=float)
    for _ in range(levels):
        if a.size < 8:
            energies.append(float("nan"))
            continue
        d = np.convolve(a, DB2_DEC_HI, mode="same")[::2]
        a = np.convolve(a, DB2_DEC_LO, mode="same")[::2]
        energies.append(float(np.mean(d ** 2)))
    return energies


def time_frequency_features(residual: np.ndarray, fs: float,
                            bands_hz: Sequence[Sequence[float]]) -> dict[str, float]:
    """How the spectrum moves over the recording: STFT flux, band drift, and wavelet levels.

    The STFT is computed once with a fixed window; the drift features compare the last third of
    the recording with the first, which is what distinguishes a stationary noise floor from a
    recording whose spectral content changes as events accumulate.
    """
    out: dict[str, float] = {}
    n = residual.size
    nperseg = min(STFT_NPERSEG, n)
    if n < 64 or fs <= 0 or nperseg < 32:
        for k in ("flux_mean", "flux_sd", "frame_entropy_mean", "peak_energy_time_frac"):
            out[f"tf_{k}"] = float("nan")
        for lo, hi in list(bands_hz)[:3]:
            out[f"tf_log10_drift_{_band_tag(lo, hi)}"] = float("nan")
        for lv in range(1, WAVELET_LEVELS + 1):
            out[f"tf_wav_log10_l{lv}"] = float("nan")
        out["tf_wav_l1_frac"] = float("nan")
        return out

    f, t, sxx = sps.spectrogram(residual, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                                detrend="constant", mode="psd")
    frame_tot = sxx.sum(axis=0)
    if sxx.shape[1] >= 2 and float(np.sum(frame_tot)) > 0:
        spec_n = sxx / np.maximum(frame_tot[None, :], 1e-300)
        flux = np.abs(np.diff(spec_n, axis=1)).sum(axis=0)
        out["tf_flux_mean"] = float(np.mean(flux))
        out["tf_flux_sd"] = float(np.std(flux, ddof=1)) if flux.size > 1 else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(np.where(spec_n > 0, spec_n * np.log2(spec_n), 0.0), axis=0)
        denom = math.log2(max(sxx.shape[0], 2))
        out["tf_frame_entropy_mean"] = float(np.mean(ent) / denom) if denom > 0 else float("nan")
        out["tf_peak_energy_time_frac"] = float(t[int(np.argmax(frame_tot))] / t[-1]) if t[-1] > 0 \
            else float("nan")
        thirds = max(1, sxx.shape[1] // 3)
        for lo, hi in list(bands_hz)[:3]:
            bm = (f >= lo) & (f <= hi)
            if np.count_nonzero(bm) < 2:
                out[f"tf_log10_drift_{_band_tag(lo, hi)}"] = float("nan")
                continue
            early = float(np.sum(sxx[bm, :thirds]))
            late = float(np.sum(sxx[bm, -thirds:]))
            out[f"tf_log10_drift_{_band_tag(lo, hi)}"] = _log10(late) - _log10(early) \
                if early > 0 and late > 0 else float("nan")
    else:
        for k in ("flux_mean", "flux_sd", "frame_entropy_mean", "peak_energy_time_frac"):
            out[f"tf_{k}"] = float("nan")
        for lo, hi in list(bands_hz)[:3]:
            out[f"tf_log10_drift_{_band_tag(lo, hi)}"] = float("nan")

    energies = _dyadic_wavelet_energies(residual)
    finite = [e for e in energies if np.isfinite(e) and e > 0]
    for lv, e in enumerate(energies, start=1):
        out[f"tf_wav_log10_l{lv}"] = _log10(e)
    out["tf_wav_l1_frac"] = _safe_div(energies[0], float(sum(finite))) if finite and len(energies) \
        and np.isfinite(energies[0]) else float("nan")
    return out


def acf_at_lags(residual: np.ndarray, fs: float, lags_ms: Sequence[float]) -> dict[str, float]:
    """Normalised autocorrelation (lag 0 = 1) at each lag in milliseconds, via FFT."""
    x = residual - residual.mean()
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    out: dict[str, float] = {}
    denom = acov[0]
    for lag_ms in lags_ms:
        k = max(1, int(round(lag_ms * 1e-3 * fs)))
        tag = f"{lag_ms:g}".replace(".", "p")
        if k >= n or denom <= 0:
            out[f"acf_{tag}ms"] = float("nan")
        else:
            out[f"acf_{tag}ms"] = float(acov[k] / denom)
    return out


# --------------------------------------------------------------------------- #
# Shape, morphology, recording context
# --------------------------------------------------------------------------- #

def shape_features(residual: np.ndarray, sigma: float) -> dict[str, float]:
    out: dict[str, float] = {
        "shape_skew": float(sst.skew(residual, bias=False)),
        "shape_excess_kurtosis": float(sst.kurtosis(residual, fisher=True, bias=False)),
    }
    if sigma > 0:
        out["shape_frac_beyond_3sigma"] = float(np.mean(np.abs(residual) > 3.0 * sigma))
    else:
        out["shape_frac_beyond_3sigma"] = float("nan")
    return out


def _event_slices(events: Sequence[Event], n: int, max_events: int = 40) -> list[tuple[int, int]]:
    spans = [(int(e.onset_idx), min(int(e.offset_idx), n)) for e in events[:max_events]]
    return [(a, b) for a, b in spans if 0 <= a < b <= n and b - a >= 3]


def morphology_features(residual: np.ndarray, fs: float, events: Sequence[Event],
                        sigma: float) -> dict[str, float]:
    """Waveform shape: how sharp, how symmetric, and how stereotyped the deflections are.

    The only cross-event reference is the mean event waveform **of the same recording**, so no
    information crosses files and no label is involved.
    """
    out: dict[str, float] = {}
    n = residual.size
    if n < 8:
        for k in ("turning_point_rate", "log10_rise_median_s", "log10_fall_median_s",
                  "rise_fall_ratio_median", "area_over_peak_median", "max_slope_norm_median",
                  "event_self_similarity"):
            out[f"morph_{k}"] = float("nan")
        return out

    d1 = np.diff(residual)
    sgn = np.sign(d1)
    out["morph_turning_point_rate"] = float(np.count_nonzero(sgn[1:] != sgn[:-1]) / max(n - 2, 1))

    spans = _event_slices(events, n)
    if not spans:
        for k in ("log10_rise_median_s", "log10_fall_median_s", "rise_fall_ratio_median",
                  "area_over_peak_median", "max_slope_norm_median", "event_self_similarity"):
            out[f"morph_{k}"] = float("nan")
        return out

    rises, falls, ratios, areas, slopes = [], [], [], [], []
    segs = []
    for a, b in spans:
        seg = residual[a:b]
        segs.append(seg)
        peak = int(np.argmax(np.abs(seg)))
        amp = float(abs(seg[peak]))
        rise_s = peak / fs if fs > 0 else float("nan")
        fall_s = (b - a - peak) / fs if fs > 0 else float("nan")
        rises.append(rise_s)
        falls.append(fall_s)
        if rise_s > 0 and np.isfinite(fall_s):
            ratios.append(fall_s / rise_s)
        if amp > 0:
            areas.append(float(np.sum(np.abs(seg))) / (b - a) / amp)
            if sigma > 0:
                slopes.append(float(np.max(np.abs(np.diff(seg)))) / amp)
    out["morph_log10_rise_median_s"] = _log10(float(np.median(rises))) if rises else float("nan")
    out["morph_log10_fall_median_s"] = _log10(float(np.median(falls))) if falls else float("nan")
    out["morph_rise_fall_ratio_median"] = float(np.median(ratios)) if ratios else float("nan")
    out["morph_area_over_peak_median"] = float(np.median(areas)) if areas else float("nan")
    out["morph_max_slope_norm_median"] = float(np.median(slopes)) if slopes else float("nan")

    # Self-similarity: correlation of each event with this recording's own mean event waveform.
    min_len = min(s.size for s in segs)
    if len(segs) >= 2 and min_len >= 4:
        mat = np.stack([s[:min_len] for s in segs])
        template = mat.mean(axis=0)
        if float(np.std(template)) > 0:
            cors = [float(np.corrcoef(row, template)[0, 1]) for row in mat
                    if float(np.std(row)) > 0]
            out["morph_event_self_similarity"] = float(np.mean(cors)) if cors else float("nan")
        else:
            out["morph_event_self_similarity"] = float("nan")
    else:
        out["morph_event_self_similarity"] = float("nan")
    return out


def context_features(baseline: np.ndarray, residual: np.ndarray, raw_level: float,
                     sigma: float, fs: float) -> dict[str, float]:
    """Recording-level context: baseline stability, reference level, and clipping indicators.

    These describe the acquisition conditions of the file. They are included because a class that
    was recorded on a different day or gain shows up here first - which is exactly the shortcut
    the explainability stage is required to look for.
    """
    out: dict[str, float] = {}
    n = baseline.size
    i0 = float(np.median(baseline)) if n else float("nan")
    out["ctx_log10_abs_i0"] = _log10(abs(i0)) if np.isfinite(i0) else float("nan")
    out["ctx_noise_over_i0"] = _safe_div(sigma, abs(i0)) if np.isfinite(i0) else float("nan")
    if n >= 10:
        out["ctx_baseline_drift_rel"] = _safe_div(float(np.std(baseline, ddof=1)), abs(i0))
        t = np.arange(n, dtype=float) / fs if fs > 0 else np.arange(n, dtype=float)
        slope = float(np.polyfit(t, baseline, 1)[0])
        out["ctx_baseline_slope_norm"] = _safe_div(slope, sigma) if sigma > 0 else float("nan")
        bounds = np.linspace(0, n, 11).astype(int)
        means = np.array([float(np.mean(residual[a:b])) for a, b in zip(bounds[:-1], bounds[1:],
                                                                         strict=True) if b - a > 1])
        out["ctx_seg_mean_range_norm"] = _safe_div(float(means.max() - means.min()), sigma) \
            if means.size >= 2 and sigma > 0 else float("nan")
    else:
        for k in ("baseline_drift_rel", "baseline_slope_norm", "seg_mean_range_norm"):
            out[f"ctx_{k}"] = float("nan")
    if n >= 4 and np.isfinite(raw_level):
        hi = float(np.max(residual))
        lo = float(np.min(residual))
        out["ctx_frac_at_extreme"] = float((np.count_nonzero(residual >= hi)
                                            + np.count_nonzero(residual <= lo)) / n)
    else:
        out["ctx_frac_at_extreme"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def resolve_groups(groups: Sequence[str] | None) -> tuple[str, ...]:
    """Expand ``all`` and validate group names against :data:`FEATURE_GROUPS`."""
    if groups is None:
        return FEATURE_GROUPS
    names = [str(g).strip().lower() for g in groups]
    if not names or "all" in names:
        return FEATURE_GROUPS
    unknown = [g for g in names if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature group(s): {', '.join(unknown)}; "
                         f"known groups: {', '.join(FEATURE_GROUPS)}")
    # Keep the canonical order so the feature matrix columns are stable across configurations.
    return tuple(g for g in FEATURE_GROUPS if g in set(names))


def extract_features(
    signal: np.ndarray,
    baseline: np.ndarray,
    fs: float,
    events: Sequence[Event],
    *,
    raw_std: float,
    raw_level: float,
    bands_hz: Sequence[Sequence[float]],
    acf_lags_ms: Sequence[float],
    groups: Sequence[str] | None = None,
) -> dict[str, float]:
    """All label-blind features for one recording at analysis rate ``fs``.

    ``signal`` is the processed (filtered, decimated) recording; ``baseline`` its rolling
    median. ``raw_std`` and ``raw_level`` (raw mean) come from the unfiltered recording.
    ``groups`` selects which feature groups are computed (``None`` or ``["all"]`` = every group);
    the order of the returned dictionary follows :data:`FEATURE_GROUPS`, so the feature matrix has
    a stable column order for a given configuration.
    """
    want = set(resolve_groups(groups))
    residual = np.asarray(signal, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    sigma = residual_sigma(residual)
    i0_level = float(np.median(baseline)) if baseline.size else float("nan")

    out: dict[str, float] = {}
    if "noise" in want:
        out.update({
            "noise_log10_sigma": math.log10(sigma) if sigma > 0 else float("nan"),
            "noise_log10_raw_std": math.log10(raw_std) if raw_std > 0 else float("nan"),
            "noise_raw_level": float(raw_level),
        })
    if "time" in want:
        out.update(time_domain_features(residual, fs, sigma))
    if "amp" in want:
        out.update(amplitude_features(residual, sigma))
    if "event" in want:
        out.update(event_features(events, signal.size, fs, i0_level))
    if "ctx" in want:
        out.update(context_features(np.asarray(baseline, dtype=np.float64), residual, raw_level,
                                    sigma, fs))
        out.update(event_context_features(events, signal.size, fs))
    psd: tuple[np.ndarray, np.ndarray] | None = None
    if "spec" in want or "tf" in want:
        psd = welch_psd(residual, fs)
    if "spec" in want and psd is not None:
        out.update(welch_band_powers(residual, fs, bands_hz, psd))
        out.update(spectral_shape_features(psd[0], psd[1]))
    if "tf" in want:
        out.update(time_frequency_features(residual, fs, bands_hz))
    if "acf" in want:
        out.update(acf_at_lags(residual, fs, acf_lags_ms))
    if "shape" in want:
        out.update(shape_features(residual, sigma))
    if "morph" in want:
        out.update(morphology_features(residual, fs, events, sigma))
    return out


def feature_names(features: dict[str, Any]) -> list[str]:
    """Stable column order: insertion order of the first recording."""
    return list(features.keys())


def feature_group(name: str) -> str:
    return name.split("_", 1)[0]


def group_of_features(names: Sequence[str]) -> dict[str, list[str]]:
    """Group the feature columns by prefix, in canonical group order."""
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(feature_group(n), []).append(n)
    return {g: out[g] for g in FEATURE_GROUPS if g in out} | \
        {g: v for g, v in out.items() if g not in FEATURE_GROUPS}
