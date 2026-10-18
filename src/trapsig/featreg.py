"""The engineered-feature registry (Branch D): 150+ candidates in three tiers, twelve families.

Every feature is a row in :data:`REGISTRY` with an id, family, tier, plain-language description,
unit and parameter set, so the CSV sidecar of a run and ``feature_registry.csv`` document exactly
what was computed and how. Nothing here looks at labels; everything is a deterministic function of
one window and the analysis sampling rate, which is why the same registry can serve window-level
and recording-level extraction.

Tiers follow the design rules:

* **Tier 1 (mandatory baseline)** - time statistics, amplitude/energy, differences,
  autocorrelation, Hjorth, FFT/PSD descriptors, band powers, STFT summaries, signal quality.
* **Tier 2 (strongly recommended)** - envelope/Hilbert, peak and event structure, wavelet and
  time-frequency summaries, entropies, change points, AR/cepstral/filter-bank descriptors.
* **Tier 3 (experimental)** - fractal dimensions, recurrence quantification, DFA/Hurst, Lyapunov,
  Lempel-Ziv, multiscale entropy, approximate entropy. These need long, fairly stationary
  segments; each is computed on a capped prefix and reported NaN when the window is too short,
  and none is claimed to help classification until the ablation says so.

Physical note: amplitudes are in the configured signal unit (raw ADC or arbitrary), so "energy"
features are signal-energy indicators, not joules; frequency features always use the analysis
sampling rate passed in, never a hardcoded one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# A hard cap for the quadratic / iterative estimators. Documented approximation, applied
# identically in every fold, so comparisons stay fair even on 20 s x 1 MHz recordings.
_CAP = 2048      # cap for the quadratic estimators; documented approximation, fold-invariant


@dataclass(frozen=True)
class FeatureSpec:
    id: str
    family: str
    tier: int
    description: str
    unit: str
    params: str = ""


FAMILIES: dict[str, str] = {
    "time_stats": "time-domain statistical features",
    "energy": "signal energy and amplitude features",
    "envelope": "envelope and instantaneous-frequency features (Hilbert)",
    "dynamics": "temporal dynamics, shape and autocorrelation features",
    "peaks_events": "peak, event, burst and change-point features",
    "quality": "signal-quality diagnostics",
    "spectral": "frequency-domain features (FFT/PSD)",
    "time_frequency": "time-frequency features (STFT and wavelet summaries)",
    "entropy_nonlinear": "entropy, complexity and nonlinear-dynamics features",
    "ar_cepstral": "autoregressive, cepstral, filter-bank and multitaper features",
    "decomposition": "signal-decomposition features (SSA always; EMD/VMD when the optional "
                     "packages are installed, NaN otherwise)",
}

_T1, _T2, _T3 = 1, 2, 3


def _spec(fid: str, family: str, tier: int, description: str, unit: str, params: str = ""
          ) -> FeatureSpec:
    return FeatureSpec(fid, family, tier, description, unit, params)


_STATIC: list[FeatureSpec] = [
    # ---- A time-domain statistics ----
    _spec("w_mean", "time_stats", _T1, "arithmetic mean of the window", "unit"),
    _spec("w_median", "time_stats", _T1, "median, robust central tendency", "unit"),
    _spec("w_sd", "time_stats", _T1, "standard deviation", "unit"),
    _spec("w_var", "time_stats", _T1, "variance", "unit^2"),
    _spec("w_min", "time_stats", _T1, "minimum amplitude", "unit"),
    _spec("w_max", "time_stats", _T1, "maximum amplitude", "unit"),
    _spec("w_p2p", "time_stats", _T1, "peak-to-peak amplitude", "unit"),
    _spec("w_rms", "time_stats", _T1, "root mean square", "unit"),
    _spec("w_mean_abs", "time_stats", _T1, "mean absolute amplitude", "unit"),
    _spec("w_mad", "time_stats", _T1, "median absolute deviation", "unit"),
    _spec("w_iqr", "time_stats", _T1, "interquartile range (q75-q25)", "unit"),
    _spec("w_skew", "time_stats", _T1, "skewness of the amplitude distribution", "-"),
    _spec("w_kurt", "time_stats", _T1, "excess kurtosis (tail weight)", "-"),
    _spec("w_q05", "time_stats", _T1, "5th percentile", "unit"),
    _spec("w_q25", "time_stats", _T1, "25th percentile", "unit"),
    _spec("w_q75", "time_stats", _T1, "75th percentile", "unit"),
    _spec("w_q95", "time_stats", _T1, "95th percentile", "unit"),
    _spec("w_cov", "time_stats", _T1, "coefficient of variation sd/|mean|", "-"),
    _spec("w_crest", "time_stats", _T1, "crest factor peak/RMS", "-"),
    _spec("w_shape", "time_stats", _T1, "shape factor RMS/mean|x|", "-"),
    _spec("w_impulse", "time_stats", _T1, "impulse factor peak/mean|x|", "-"),
    _spec("w_clearance", "time_stats", _T1, "clearance factor peak/mean(sqrt|x|)^2", "-"),
    _spec("w_zcr", "time_stats", _T1, "zero-crossing rate of the mean-removed window", "1/s"),
    # ---- B energy ----
    _spec("w_energy", "energy", _T1, "sum of squares (signal-energy indicator)", "unit^2"),
    _spec("w_mean_square", "energy", _T1, "mean squared amplitude", "unit^2"),
    _spec("w_log_energy", "energy", _T1, "log(energy+eps), dynamic-range compressed", "log unit^2"),
    _spec("w_ste_mean", "energy", _T2, "mean short-time energy over 64-sample frames", "unit^2"),
    _spec("w_ste_sd", "energy", _T2, "sd of short-time energy across frames", "unit^2"),
    _spec("w_energy_entropy", "energy", _T2, "entropy of the normalised frame-energy profile",
          "nats"),
    _spec("w_abs_integral", "energy", _T1, "sum of |x| (amplitude integral)", "unit*samples"),
    # ---- envelope ----
    _spec("w_env_mean", "envelope", _T2, "mean Hilbert envelope", "unit"),
    _spec("w_env_sd", "envelope", _T2, "Hilbert envelope sd", "unit"),
    _spec("w_env_peak", "envelope", _T2, "largest envelope amplitude", "unit"),
    _spec("w_env_mod_depth", "envelope", _T2, "(env peak - env mean)/env mean modulation depth",
          "-"),
    _spec("w_inst_freq_mean", "envelope", _T2, "mean Hilbert instantaneous frequency", "Hz"),
    # ---- C dynamics ----
    _spec("w_d1_mean", "dynamics", _T1, "mean first difference", "unit/sample"),
    _spec("w_d1_sd", "dynamics", _T1, "sd of first differences", "unit/sample"),
    _spec("w_d2_mean", "dynamics", _T2, "mean second difference", "unit/sample^2"),
    _spec("w_d2_sd", "dynamics", _T1, "sd of second differences", "unit/sample^2"),
    _spec("w_slope", "dynamics", _T1, "least-squares linear trend slope", "unit/s"),
    _spec("w_trend_r2", "dynamics", _T2, "goodness of fit (R^2) of the linear trend", "-"),
    _spec("w_acf1", "dynamics", _T1, "autocorrelation at lag 1", "-"),
    _spec("w_acf2", "dynamics", _T1, "autocorrelation at lag 2", "-"),
    _spec("w_acf3", "dynamics", _T2, "autocorrelation at lag 3", "-"),
    _spec("w_acf5", "dynamics", _T2, "autocorrelation at lag 5", "-"),
    _spec("w_acf10", "dynamics", _T2, "autocorrelation at lag 10", "-"),
    _spec("w_acf_decay_lag", "dynamics", _T2, "first lag where |ACF| < 0.5 (persistence)",
          "samples"),
    _spec("w_run_mean", "dynamics", _T2, "mean run length above/below the window median",
          "samples"),
    _spec("w_run_max", "dynamics", _T2, "longest run above/below the window median", "samples"),
    _spec("w_turning_rate", "dynamics", _T1, "turning-point rate of the first difference", "1/s"),
    _spec("w_hjorth_activity", "dynamics", _T1, "Hjorth activity (variance)", "unit^2"),
    _spec("w_hjorth_mobility", "dynamics", _T1, "Hjorth mobility sqrt(var(d1)/var(x))", "1/s"),
    _spec("w_hjorth_complexity", "dynamics", _T1, "Hjorth complexity (mobility ratio)", "-"),
    # ---- G peaks / events / change points ----
    _spec("w_n_peaks", "peaks_events", _T2, "count of prominent local maxima", "count"),
    _spec("w_peak_interval_mean", "peaks_events", _T2, "mean inter-peak interval", "s"),
    _spec("w_peak_interval_sd", "peaks_events", _T2, "sd of inter-peak intervals (regularity)",
          "s"),
    _spec("w_prominence_mean", "peaks_events", _T2, "mean peak prominence", "unit"),
    _spec("w_prominence_sd", "peaks_events", _T2, "sd of peak prominences", "unit"),
    _spec("w_burst_count", "peaks_events", _T2, "bursts above 3x robust sd, min 32 samples",
          "count"),
    _spec("w_burst_mean_dur", "peaks_events", _T2, "mean burst duration", "s"),
    _spec("w_transient_density", "peaks_events", _T2, "peaks per second (transient rate)", "1/s"),
    _spec("w_thresh_cross_interval", "peaks_events", _T2,
          "mean interval between +2-sd threshold crossings", "s"),
    _spec("w_changepoint_count", "peaks_events", _T2,
          "rolling-mean change points (3x MAD rule)", "count"),
    _spec("w_changepoint_mag", "peaks_events", _T2, "mean |jump| at detected change points",
          "unit"),
    _spec("w_local_var_sd", "peaks_events", _T2, "sd of the rolling-variance profile", "unit^2"),
    # ---- quality ----
    _spec("w_clip_pct", "quality", _T1, "fraction of samples at >=99.9% of the peak (clipping)",
          "%"),
    _spec("w_flatline_pct", "quality", _T1, "fraction of zero first-differences (flat-line)", "%"),
    _spec("w_outlier_rate", "quality", _T1, "fraction beyond 5x MAD from the median", "%"),
    _spec("w_saturation_pct", "quality", _T1, "fraction in the top/bottom 0.1% amplitude bins",
          "%"),
    _spec("w_nan_pct", "quality", _T1, "fraction of non-finite samples in the window", "%"),
    # ---- D spectral (band-independent part) ----
    _spec("w_dom_freq", "spectral", _T1, "dominant (largest-magnitude) FFT frequency", "Hz"),
    _spec("w_peak2_freq", "spectral", _T2, "second-largest separated spectral peak", "Hz"),
    _spec("w_peak3_freq", "spectral", _T2, "third-largest separated spectral peak", "Hz"),
    _spec("w_peak_amp", "spectral", _T1, "magnitude at the dominant frequency", "unit"),
    _spec("w_spec_centroid", "spectral", _T1, "spectral centroid (centre of mass)", "Hz"),
    _spec("w_spec_bandwidth", "spectral", _T1, "spectral bandwidth (sd about the centroid)", "Hz"),
    _spec("w_spec_rolloff85", "spectral", _T1, "frequency below 85% of cumulative power", "Hz"),
    _spec("w_spec_flatness", "spectral", _T1, "geometric/arithmetic power ratio (noise-like=1)",
          "-"),
    _spec("w_spec_entropy", "spectral", _T1, "entropy of the normalised power spectrum", "nats"),
    _spec("w_spec_skew", "spectral", _T2, "skewness of the power-weighted frequency distribution",
          "-"),
    _spec("w_spec_kurt", "spectral", _T2, "kurtosis of the power-weighted frequency distribution",
          "-"),
    _spec("w_total_psd", "spectral", _T1, "integrated power spectral density", "unit^2/Hz"),
    _spec("w_power_ratio_hl", "spectral", _T2, "power above Nyquist/2 divided by power below", "-"),
    _spec("w_harmonic_ratio", "spectral", _T2, "power at f0 harmonics / total power", "-"),
    _spec("w_f0_est", "spectral", _T2, "fundamental estimate: strongest low-half peak", "Hz"),
    _spec("w_harmonicity", "spectral", _T2, "mean normalised magnitude at 2f0..5f0", "-"),
    _spec("w_peak_spacing", "spectral", _T2, "median spacing of the top separated peaks", "Hz"),
    _spec("w_spec_slope", "spectral", _T2, "log-log slope of the power spectrum (spectral decay)",
          "-"),
    _spec("w_spec_flux", "spectral", _T2, "mean frame-to-frame spectral flux", "unit/frame"),
    # ---- E time-frequency ----
    _spec("w_stft_energy_mean", "time_frequency", _T1, "mean STFT frame energy", "unit^2"),
    _spec("w_stft_energy_sd", "time_frequency", _T1, "sd of STFT frame energy", "unit^2"),
    _spec("w_stft_centroid_mean", "time_frequency", _T2, "mean per-frame spectral centroid", "Hz"),
    _spec("w_stft_centroid_sd", "time_frequency", _T2, "sd of per-frame spectral centroid", "Hz"),
    _spec("w_stft_entropy_mean", "time_frequency", _T2, "mean per-frame spectral entropy", "nats"),
    _spec("w_stft_entropy_sd", "time_frequency", _T2, "sd of per-frame spectral entropy", "nats"),
    _spec("w_stft_flux_mean", "time_frequency", _T2, "mean STFT spectral flux", "unit/frame"),
    _spec("w_stft_flux_sd", "time_frequency", _T2, "sd of STFT spectral flux", "unit/frame"),
    _spec("w_wav_entropy", "time_frequency", _T2, "entropy across Morlet scale energies", "nats"),
    _spec("w_wp_energy_entropy", "time_frequency", _T3, "entropy across wavelet-packet bands",
          "nats"),
    _spec("w_cwt_ridge_mean", "time_frequency", _T2, "mean instantaneous ridge scale (argmax per "
          "column) converted to Hz", "Hz"),
    _spec("w_cwt_ridge_sd", "time_frequency", _T2, "sd of the instantaneous ridge frequency",
          "Hz"),
    _spec("w_tf_concentration", "time_frequency", _T3,
          "fraction of spectrogram energy in the top 10% of bins", "-"),
    # ---- F entropy / nonlinear ----
    _spec("w_perm_entropy3", "entropy_nonlinear", _T2, "normalised permutation entropy, order 3",
          "-", "order=3"),
    _spec("w_sampen_approx", "entropy_nonlinear", _T2, "sample entropy (capped, blocked)", "-",
          "m=2, r=0.2*sd, cap=4096"),
    _spec("w_approx_entropy", "entropy_nonlinear", _T3, "approximate entropy (self-matches)",
          "-", "m=2, r=0.2*sd, cap=2048"),
    _spec("w_mse_s2", "entropy_nonlinear", _T3, "sample entropy at coarse-graining scale 2", "-"),
    _spec("w_mse_s4", "entropy_nonlinear", _T3, "sample entropy at coarse-graining scale 4", "-"),
    _spec("w_mse_s8", "entropy_nonlinear", _T3, "sample entropy at coarse-graining scale 8", "-"),
    _spec("w_dfa_alpha", "entropy_nonlinear", _T3, "DFA scaling exponent (long-range dependence)",
          "-"),
    _spec("w_higuchi_fd", "entropy_nonlinear", _T3, "Higuchi fractal dimension", "-"),
    _spec("w_katz_fd", "entropy_nonlinear", _T3, "Katz fractal dimension", "-"),
    _spec("w_petrosian_fd", "entropy_nonlinear", _T3, "Petrosian fractal dimension (sign changes)",
          "-"),
    _spec("w_rec_rate", "entropy_nonlinear", _T3, "recurrence-plot density", "-",
          "eps=90th percentile, cap=2048"),
    _spec("w_determinism", "entropy_nonlinear", _T3, "REC recurrence quantification (diag-2)",
          "-", "cap=2048"),
    _spec("w_laminarity", "entropy_nonlinear", _T3, "vertical-line laminarity of the REC plot",
          "-", "cap=2048"),
    _spec("w_lz_complexity", "entropy_nonlinear", _T3, "normalised Lempel-Ziv complexity of the "
          "quartile-symbolised window", "-", "alphabet=4"),
    _spec("w_hurst", "entropy_nonlinear", _T3, "Hurst exponent from rescaled-range analysis", "-"),
    _spec("w_lyap_est", "entropy_nonlinear", _T3, "largest-Lyapunov-style divergence slope",
          "1/s", "Rosenstein-style, cap=1024"),
    # ---- H AR / cepstral / filter bank / multitaper ----
    _spec("w_ar_c1", "ar_cepstral", _T2, "autoregressive coefficient 1 (Yule-Walker)", "-"),
    _spec("w_ar_c2", "ar_cepstral", _T2, "autoregressive coefficient 2", "-"),
    _spec("w_ar_c3", "ar_cepstral", _T2, "autoregressive coefficient 3", "-"),
    _spec("w_ar_c4", "ar_cepstral", _T2, "autoregressive coefficient 4", "-"),
    _spec("w_ar_res_var", "ar_cepstral", _T2, "AR(4) residual variance", "unit^2"),
    _spec("w_mfcc1", "ar_cepstral", _T2, "first mel-cepstral coefficient (20 mel bands)", "-"),
    _spec("w_mfcc2", "ar_cepstral", _T2, "second mel-cepstral coefficient", "-"),
    _spec("w_mfcc3", "ar_cepstral", _T2, "third mel-cepstral coefficient", "-"),
    _spec("w_mfcc4", "ar_cepstral", _T2, "fourth mel-cepstral coefficient", "-"),
    _spec("w_mt_dom_freq", "ar_cepstral", _T3, "dominant frequency of a 5-taper multitaper "
          "estimate", "Hz", "NW=3, K=5"),
    # ---- I decomposition (SSA always; EMD/VMD optional dependencies) ----
    _spec("w_ssa_sv1_energy_frac", "decomposition", _T3,
          "energy fraction carried by the leading singular value of the Hankel (SSA) embedding; "
          "close to 1 means a near-deterministic/trend-dominated window",
          "fraction", "lag=min(64, n//4)"),
    _spec("w_ssa_sv_entropy", "decomposition", _T3,
          "Shannon entropy of the normalised singular-value spectrum of the Hankel embedding; "
          "high = many comparable modes", "nats", "lag=min(64, n//4)"),
    _spec("w_emd_imf1_energy_frac", "decomposition", _T3,
          "energy fraction of the first intrinsic mode function from empirical mode "
          "decomposition; NaN unless the optional `emd` package is installed",
          "fraction", "emd.sift defaults; cap 4096 samples"),
    _spec("w_vmd_mode1_energy_frac", "decomposition", _T3,
          "energy fraction of the first variational mode decomposition mode; NaN unless the "
          "optional `vmdpy` package is installed", "fraction", "alpha=2000, tau=0, K=3, DC=1, "
          "tol=1e-7; cap 4096 samples"),
]

_WAV_SCALES = (4, 8, 16, 32)
_FB_BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.4), (0.4, 0.5))   # fractions of Nyquist


def registry_specs(bands: list[list[float]], tiers: tuple[int, ...] | list[int] = (1, 2, 3),
                   families: list[str] | None = None) -> list[FeatureSpec]:
    """The ordered registry for a band configuration and tier/family selection."""
    tsel = {int(t) for t in tiers}
    specs: list[FeatureSpec] = []
    for sp in _STATIC:
        if sp.tier in tsel and (families is None or sp.family in families):
            specs.append(sp)
    if families is None or "spectral" in families:
        for lo, hi in bands:
            if 1 in tsel:
                specs.append(_spec(f"w_band_{lo:g}_{hi:g}", "spectral", 1,
                                   f"log10 power in [{lo:g},{hi:g}) Hz", "log10 unit^2"))
            if 2 in tsel:
                specs.append(_spec(f"w_relband_{lo:g}_{hi:g}", "spectral", 2,
                                   f"fraction of total power in [{lo:g},{hi:g}) Hz", "-"))
    if families is None or "time_frequency" in families:
        for wdt in _WAV_SCALES:
            if 2 in tsel:
                specs.append(_spec(f"w_wav_energy_{wdt}", "time_frequency", 2,
                                   f"Morlet CWT energy at width {wdt} samples", "unit^2"))
        for i, (lo, hi) in enumerate(_FB_BANDS):
            if 2 in tsel:
                specs.append(_spec(f"w_wp_energy_{i}", "time_frequency", 2,
                                   f"wavelet-packet-style band energy, {lo:.2f}-{hi:.2f} of "
                                   "Nyquist", "unit^2"))
    if families is None or "ar_cepstral" in families:
        for i, (lo, hi) in enumerate(_FB_BANDS):
            if 2 in tsel:
                specs.append(_spec(f"w_fb_energy_{i}", "ar_cepstral", 2,
                                   f"filter-bank energy, {lo:.2f}-{hi:.2f} of Nyquist", "unit^2"))
    return specs


def feature_names(bands: list[list[float]], tiers: tuple[int, ...] | list[int] = (1, 2, 3),
                  families: list[str] | None = None) -> list[str]:
    return [sp.id for sp in registry_specs(bands, tiers, families)]


def registry_rows(bands: list[list[float]], tiers: tuple[int, ...] | list[int] = (1, 2, 3),
                  families: list[str] | None = None) -> list[dict[str, Any]]:
    return [{"id": sp.id, "family": sp.family, "family_description": FAMILIES[sp.family],
             "tier": sp.tier, "description": sp.description, "unit": sp.unit,
             "parameters": sp.params}
            for sp in registry_specs(bands, tiers, families)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _finite(v: float) -> float:
    return float(v) if np.isfinite(v) else float("nan")


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    xc = x - x.mean()
    d = float(np.dot(xc, xc))
    if d <= 0:
        return np.zeros(nlags + 1)
    return np.array([np.dot(xc[: xc.size - k], xc[k:]) / d for k in range(nlags + 1)])


def _morlet(x: np.ndarray, widths: tuple[int, ...] | np.ndarray, w0: float = 6.0) -> np.ndarray:
    out = np.empty((len(widths), x.size))
    for i, wdt in enumerate(widths):
        m = max(9, int(wdt * 8) | 1)
        t = (np.arange(m) - m // 2) / float(wdt)
        psi = math.pi ** -0.25 * np.exp(1j * w0 * t) * np.exp(-0.5 * t ** 2)
        c = np.abs(np.convolve(x.astype(float), psi[::-1].conj(), mode="full")) \
            / math.sqrt(float(wdt))
        if c.size > x.size:                      # centre-crop when the kernel exceeds the window
            st = (c.size - x.size) // 2
            c = c[st:st + x.size]
        elif c.size < x.size:
            c = np.pad(c, (0, x.size - c.size))
        out[i] = c
    return out


def _near_counts(tmpl: np.ndarray, r: float, step: int = 512) -> np.ndarray:
    n = tmpl.shape[0]
    out = np.empty(n, dtype=np.int64)
    for i in range(0, n, step):
        blk = tmpl[i:i + step]
        d = np.max(np.abs(blk[:, None, :] - tmpl[None, :, :]), axis=-1)
        out[i:i + step] = (d <= r).sum(axis=1) - 1
    return out


def _sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)[:_CAP]
    sd = float(np.std(x))
    if x.size < m + 2 or sd <= 0:
        return float("nan")
    r = r_frac * sd
    n = x.size - m
    if n < 8:
        return float("nan")
    tmpl = np.stack([x[i:i + m] for i in range(n)])
    near = _near_counts(tmpl, r)
    tmpl2 = np.stack([x[i:i + m + 1] for i in range(n - 1)])
    near2 = _near_counts(tmpl2, r)
    a = float(np.sum(near2 > 0))
    b = float(np.sum(near > 0))
    if a == 0 or b == 0:
        return float("nan")
    return float(-math.log(a / b))


def perm_entropy(x: np.ndarray, order: int = 3) -> float:
    """Normalised permutation entropy of ordinal patterns (public, standalone)."""
    import collections

    x = np.asarray(x, dtype=float)[:_CAP]
    if x.size < order + 1:
        return float("nan")
    mat = np.lib.stride_tricks.sliding_window_view(x, order)
    ranks = np.argsort(np.argsort(mat, axis=1), axis=1)
    cnt: dict[tuple[int, ...], int] = collections.Counter(tuple(r) for r in ranks.tolist())
    ps = np.array(list(cnt.values()), dtype=float) / max(1, len(ranks))
    denom = math.log(math.factorial(order))
    return float(-np.sum(ps * np.log(ps)) / denom) if denom > 0 else float("nan")


def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """Public alias of the blocked, capped sample-entropy estimator."""
    return _sample_entropy(x, m, r_frac)


def _approx_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)[:2048]
    sd = float(np.std(x))
    n = x.size - m
    if n < 8 or sd <= 0:
        return float("nan")
    r = r_frac * sd
    tmpl = np.stack([x[i:i + m] for i in range(n)])
    cnt = _near_counts(tmpl, r) + 1          # approximate entropy includes self-matches
    phi_m = float(np.mean(np.log(cnt / n)))
    tmpl2 = np.stack([x[i:i + m + 1] for i in range(x.size - m - 1)])
    cnt2 = _near_counts(tmpl2, r) + 1
    phi_m1 = float(np.mean(np.log(cnt2 / max(1, tmpl2.shape[0]))))
    return _finite(phi_m - phi_m1)


# ---------------------------------------------------------------------------
# family computers: each returns {feature id: value}
# ---------------------------------------------------------------------------

def _f_time_stats(x: np.ndarray, fs: float) -> dict[str, float]:
    ma = float(np.mean(np.abs(x)))
    mean = float(np.mean(x))
    sd = float(np.std(x))
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    q = np.percentile(x, [5, 25, 50, 75, 95])
    sr = np.sqrt(np.mean(np.sqrt(np.abs(x))))
    return {
        "w_mean": mean, "w_median": float(q[2]), "w_sd": sd, "w_var": sd ** 2,
        "w_min": float(np.min(x)), "w_max": float(np.max(x)),
        "w_p2p": float(np.max(x) - np.min(x)), "w_rms": rms, "w_mean_abs": ma,
        "w_mad": float(np.median(np.abs(x - q[2]))), "w_iqr": float(q[3] - q[1]),
        "w_skew": float(((x - mean) ** 3).mean() / sd ** 3) if sd > 0 else float("nan"),
        "w_kurt": float(((x - mean) ** 4).mean() / sd ** 4 - 3.0) if sd > 0 else float("nan"),
        "w_q05": float(q[0]), "w_q25": float(q[1]), "w_q75": float(q[3]), "w_q95": float(q[4]),
        "w_cov": sd / abs(mean) if abs(mean) > 0 else float("nan"),
        "w_crest": peak / rms if rms > 0 else float("nan"),
        "w_shape": rms / ma if ma > 0 else float("nan"),
        "w_impulse": peak / ma if ma > 0 else float("nan"),
        "w_clearance": peak / sr ** 2 if sr > 0 else float("nan"),
        "w_zcr": float(np.mean(np.diff(np.signbit(x - mean)))) * fs,
    }


def _f_energy(x: np.ndarray, fs: float) -> dict[str, float]:
    e = float(np.sum(x ** 2))
    fr = 64
    if x.size >= fr:
        frames = x[: x.size - x.size % fr].reshape(-1, fr)
        ste = np.sum(frames ** 2, axis=1)
        pn = ste / ste.sum() if ste.sum() > 0 else ste
        ent = float(-np.sum(pn[pn > 0] * np.log(pn[pn > 0]))) if np.any(pn > 0) else float("nan")
        ste_mean, ste_sd = float(np.mean(ste)), float(np.std(ste))
    else:
        ste_mean = ste_sd = ent = float("nan")
    return {"w_energy": e, "w_mean_square": float(np.mean(x ** 2)),
            "w_log_energy": math.log(e + 1e-12), "w_ste_mean": ste_mean, "w_ste_sd": ste_sd,
            "w_energy_entropy": ent, "w_abs_integral": float(np.sum(np.abs(x)))}


def _f_envelope(x: np.ndarray, fs: float) -> dict[str, float]:
    from scipy import signal as sps

    if x.size < 16:
        return {k: float("nan") for k in ("w_env_mean", "w_env_sd", "w_env_peak",
                                          "w_env_mod_depth", "w_inst_freq_mean")}
    ana = sps.hilbert(x - x.mean())
    env = np.abs(ana)
    inst = np.diff(np.unwrap(np.angle(ana))) * fs / (2 * math.pi)
    em = float(np.mean(env))
    return {"w_env_mean": em, "w_env_sd": float(np.std(env)), "w_env_peak": float(np.max(env)),
            "w_env_mod_depth": (float(np.max(env)) - em) / em if em > 0 else float("nan"),
            "w_inst_freq_mean": float(np.mean(inst))}


def _f_dynamics(x: np.ndarray, fs: float) -> dict[str, float]:
    d1 = np.diff(x)
    d2 = np.diff(x, 2)
    t = np.arange(x.size) / fs
    v1 = float(np.var(d1)) if d1.size else float("nan")
    v2 = float(np.var(d2)) if d2.size else float("nan")
    vx = float(np.var(x))
    if x.size > 4 and vx > 0:
        slope, intercept = np.polyfit(t, x, 1)[:2]
        pred = slope * t + intercept
        ss_res = float(np.sum((x - pred) ** 2))
        r2 = 1.0 - ss_res / float(np.sum((x - x.mean()) ** 2))
    else:
        slope, r2 = float("nan"), float("nan")
        intercept = 0.0
    a = _acf(x, 10)
    decay = next((k for k in range(1, 11) if abs(a[k]) < 0.5), np.nan)
    med = np.median(x)
    above = x > med
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate(([0], above.view(np.int8), [0])))) )
    run_len = runs[::2] if runs.size else np.array([0])
    mob = math.sqrt(v1 / vx) if vx > 0 else float("nan")
    mob2 = math.sqrt(v2 / v1) if v1 and v1 > 0 else float("nan")
    return {"w_d1_mean": float(np.mean(d1)) if d1.size else float("nan"),
            "w_d1_sd": float(np.std(d1)) if d1.size else float("nan"),
            "w_d2_mean": float(np.mean(d2)) if d2.size else float("nan"),
            "w_d2_sd": float(np.std(d2)) if d2.size else float("nan"),
            "w_slope": float(slope), "w_trend_r2": _finite(r2),
            "w_acf1": float(a[1]), "w_acf2": float(a[2]), "w_acf3": float(a[3]),
            "w_acf5": float(a[5]), "w_acf10": float(a[10]), "w_acf_decay_lag": float(decay),
            "w_run_mean": float(np.mean(run_len)), "w_run_max": float(np.max(run_len)),
            "w_turning_rate": float(np.mean(np.diff(np.sign(d1)) != 0)) * fs,
            "w_hjorth_activity": vx, "w_hjorth_mobility": mob,
            "w_hjorth_complexity": (mob2 / mob) if mob and mob > 0 else float("nan")}


def _f_peaks_events(x: np.ndarray, fs: float) -> dict[str, float]:
    from scipy import signal as sps

    out = {k: float("nan") for k in
           ("w_n_peaks", "w_peak_interval_mean", "w_peak_interval_sd", "w_prominence_mean",
            "w_prominence_sd", "w_burst_count", "w_burst_mean_dur", "w_transient_density",
            "w_thresh_cross_interval", "w_changepoint_count", "w_changepoint_mag",
            "w_local_var_sd")}
    if x.size < 32:
        return out
    pk, props = sps.find_peaks(np.abs(x), height=0.6 * np.max(np.abs(x)), prominence=0.0)
    out["w_n_peaks"] = float(pk.size)
    if pk.size >= 2:
        iv = np.diff(pk) / fs
        out["w_peak_interval_mean"] = float(np.mean(iv))
        out["w_peak_interval_sd"] = float(np.std(iv))
    if pk.size:
        pr = props.get("prominences")
        if pr is not None and pr.size:
            out["w_prominence_mean"] = float(np.mean(pr))
            out["w_prominence_sd"] = float(np.std(pr))
    out["w_transient_density"] = pk.size / (x.size / fs)
    thr = np.median(x) + 2 * 1.4826 * np.median(np.abs(x - np.median(x)))
    cross = np.flatnonzero(np.diff((x > thr).astype(np.int8)) == 1)
    if cross.size >= 2:
        out["w_thresh_cross_interval"] = float(np.mean(np.diff(cross))) / fs
    sd3 = 1.4826 * np.median(np.abs(x - np.median(x))) or float(np.std(x)) or 1.0
    burst = (np.abs(x - np.median(x)) > 3 * sd3).astype(np.int8)
    edges = np.diff(np.concatenate(([0], burst, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if starts.size:
        durs = (ends - starts) / fs
        keep = durs >= 32 / fs
        out["w_burst_count"] = float(int(keep.sum()))
        out["w_burst_mean_dur"] = float(np.mean(durs[keep])) if keep.any() else float("nan")
    w = max(16, x.size // 100)
    if w >= 4 and x.size >= 2 * w:
        mu = np.array([x[i:i + w].mean() for i in range(0, x.size - w, w)])
        va = np.array([x[i:i + w].var() for i in range(0, x.size - w, w)])
        d = np.abs(np.diff(mu))
        mad = np.median(np.abs(d - np.median(d))) or 1e-9
        cp = np.flatnonzero(d > np.median(d) + 3 * mad)
        out["w_changepoint_count"] = float(cp.size)
        out["w_changepoint_mag"] = float(np.mean(d[cp])) if cp.size else 0.0
        out["w_local_var_sd"] = float(np.std(va))
    return out


def _f_quality(x: np.ndarray, fs: float) -> dict[str, float]:
    peak = float(np.max(np.abs(x)))
    d1z = float(np.mean(np.diff(x) == 0)) if x.size > 1 else float("nan")
    med = np.median(x)
    madv = np.median(np.abs(x - med)) or 1e-12
    lo, hi = np.percentile(x, [0.1, 99.9])
    return {"w_clip_pct": 100.0 * float(np.mean(np.abs(x) >= 0.999 * peak)) if peak > 0 else 0.0,
            "w_flatline_pct": 100.0 * d1z,
            "w_outlier_rate": 100.0 * float(np.mean(np.abs(x - med) > 5 * madv)),
            "w_saturation_pct": 100.0 * float(np.mean((x <= lo) | (x >= hi))),
            "w_nan_pct": 100.0 * float(np.mean(~np.isfinite(x)))}


def _spectra(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xc = x - x.mean()
    f = np.fft.rfftfreq(xc.size, 1 / fs)
    X = np.fft.rfft(xc)
    P = np.abs(X) ** 2
    return f, np.abs(X), P


def _f_spectral(x: np.ndarray, fs: float, bands: list[list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    if x.size < 32:
        keys = ["w_dom_freq", "w_peak2_freq", "w_peak3_freq", "w_peak_amp", "w_spec_centroid",
                "w_spec_bandwidth", "w_spec_rolloff85", "w_spec_flatness", "w_spec_entropy",
                "w_spec_skew", "w_spec_kurt", "w_total_psd", "w_power_ratio_hl",
                "w_harmonic_ratio", "w_f0_est", "w_harmonicity", "w_peak_spacing",
                "w_spec_slope", "w_spec_flux"]
        for lo, hi in bands:
            keys += [f"w_band_{lo:g}_{hi:g}", f"w_relband_{lo:g}_{hi:g}"]
        return {k: float("nan") for k in keys}
    f, M, P = _spectra(x, fs)
    tot = float(P.sum())
    order = np.argsort(-M)
    peaks: list[int] = []
    for j in order:
        if all(abs(int(j) - p) > max(2, x.size // 200) for p in peaks):
            peaks.append(int(j))
        if len(peaks) == 3:
            break
    while len(peaks) < 3:
        peaks.append(peaks[-1] if peaks else 0)
    pn = P / tot if tot > 0 else P
    cen = float(np.sum(f * pn))
    bw = float(np.sqrt(np.sum(pn * (f - cen) ** 2)))
    cum = np.cumsum(P)
    rolloff = float(f[min(f.size - 1, int(np.searchsorted(cum, 0.85 * cum[-1])))]) if tot > 0 \
        else float("nan")
    Ppos = P[P > 0]
    flat = float(np.exp(np.mean(np.log(Ppos))) / np.mean(Ppos)) if Ppos.size else float("nan")
    ent = float(-np.sum(pn[pn > 0] * np.log(pn[pn > 0]))) if tot > 0 else float("nan")
    half = f[-1] / 2
    hi_m = f >= half
    ratio = float(P[hi_m].sum() / P[~hi_m].sum()) if P[~hi_m].sum() > 0 else float("nan")
    low = (f > 0) & (f < half)
    f0 = float(f[peaks[0]]) if low.any() else float("nan")
    harm_m = np.zeros(f.size, dtype=bool)
    for h in range(1, 6):
        target = f0 * h
        if 0 < target < f[-1]:
            harm_m[int(np.argmin(np.abs(f - target)))] = True
    hratio = float(P[harm_m].sum() / tot) if tot > 0 else float("nan")
    hmag = [float(M[int(np.argmin(np.abs(f - f0 * h)))]) / (M[peaks[0]] + 1e-12)
            for h in range(2, 6) if 0 < f0 * h < f[-1]]
    pos_f = f[f > 0]
    pos_P = P[f > 0]
    if pos_f.size > 2 and np.all(pos_P > 0):
        slope = float(np.polyfit(np.log(pos_f), np.log(pos_P + 1e-30), 1)[0])
    else:
        slope = float("nan")
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        bp = float(np.log10(P[m].sum() + 1e-12)) if m.any() else float("nan")
        out[f"w_band_{lo:g}_{hi:g}"] = bp
        out[f"w_relband_{lo:g}_{hi:g}"] = float(P[m].sum() / tot) if m.any() and tot > 0 \
            else float("nan")
    spaced = [f[p] for p in peaks]
    out.update({
        "w_dom_freq": float(f[peaks[0]]), "w_peak2_freq": float(f[peaks[1]]),
        "w_peak3_freq": float(f[peaks[2]]), "w_peak_amp": float(M[peaks[0]]),
        "w_spec_centroid": cen, "w_spec_bandwidth": bw, "w_spec_rolloff85": rolloff,
        "w_spec_flatness": flat, "w_spec_entropy": ent,
        "w_spec_skew": float(np.sum(pn * ((f - cen) / bw) ** 3)) if bw > 0 else float("nan"),
        "w_spec_kurt": float(np.sum(pn * ((f - cen) / bw) ** 4) - 3) if bw > 0 else float("nan"),
        "w_total_psd": tot / fs * 2.0 / max(1, x.size), "w_power_ratio_hl": ratio,
        "w_harmonic_ratio": hratio, "w_f0_est": f0,
        "w_harmonicity": float(np.mean(hmag)) if hmag else float("nan"),
        "w_peak_spacing": float(np.median(np.diff(spaced))) if len(spaced) >= 2 else float("nan"),
        "w_spec_slope": slope,
        "w_spec_flux": float("nan")})     # filled by the time-frequency family
    return out


def _f_time_frequency(x: np.ndarray, fs: float) -> dict[str, float]:
    from scipy import signal as sps

    keys = ["w_stft_energy_mean", "w_stft_energy_sd", "w_stft_centroid_mean", "w_stft_centroid_sd",
            "w_stft_entropy_mean", "w_stft_entropy_sd", "w_stft_flux_mean", "w_stft_flux_sd",
            "w_wav_entropy", "w_wp_energy_entropy", "w_cwt_ridge_mean", "w_cwt_ridge_sd",
            "w_tf_concentration"] + [f"w_wav_energy_{w}" for w in _WAV_SCALES] \
        + [f"w_wp_energy_{i}" for i in range(len(_FB_BANDS))]
    out = {k: float("nan") for k in keys}
    if x.size < 128:
        return out
    nper = min(256, max(64, x.size // 8))
    f, t, Z = sps.spectrogram(x, fs=fs, nperseg=nper, noverlap=nper // 2, mode="magnitude")
    E = (Z ** 2).sum(axis=0)
    out["w_stft_energy_mean"] = float(np.mean(E))
    out["w_stft_energy_sd"] = float(np.std(E))
    tot_f = Z.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cen = np.where(tot_f > 0, (f[:, None] * Z).sum(axis=0) / np.where(tot_f > 0, tot_f, 1),
                       np.nan)
        Zp = Z / np.where(tot_f > 0, tot_f, 1)[None, :]
        ent = np.where(tot_f > 0, -np.nansum(np.where(Zp > 0, Zp * np.log(np.where(Zp > 0, Zp, 1)),
                                                      0.0), axis=0), np.nan)
    out["w_stft_centroid_mean"] = float(np.nanmean(cen))
    out["w_stft_centroid_sd"] = float(np.nanstd(cen))
    out["w_stft_entropy_mean"] = float(np.nanmean(ent))
    out["w_stft_entropy_sd"] = float(np.nanstd(ent))
    flux = np.sqrt(np.sum(np.diff(Z, axis=1) ** 2, axis=0))
    out["w_stft_flux_mean"] = float(np.mean(flux))
    out["w_stft_flux_sd"] = float(np.std(flux))
    flat = Z.ravel() ** 2
    srt = np.sort(flat)[::-1]
    k = max(1, srt.size // 10)
    out["w_tf_concentration"] = float(srt[:k].sum() / srt.sum()) if srt.sum() > 0 else float("nan")
    wav = _morlet(x[:8192], _WAV_SCALES)
    en = np.sum(wav ** 2, axis=1)
    for wdt, e in zip(_WAV_SCALES, en, strict=True):
        out[f"w_wav_energy_{wdt}"] = float(e)
    tot = float(en.sum())
    if tot > 0:
        pn = en / tot
        out["w_wav_entropy"] = float(-np.sum(pn[pn > 0] * np.log(pn[pn > 0])))
    # wavelet-packet-style bands: split the positive spectrum into fractional Nyquist bands
    fc = np.fft.rfftfreq(x.size, 1 / fs)
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    wp = []
    for lo, hi in _FB_BANDS:
        m = (fc >= lo * fs / 2) & (fc < hi * fs / 2)
        wp.append(float(P[m].sum()) if m.any() else float("nan"))
    for i, e in enumerate(wp):
        out[f"w_wp_energy_{i}"] = e
    wtot = np.nansum(np.array(wp))
    if wtot > 0:
        pn = np.nan_to_num(np.array(wp)) / wtot
        out["w_wp_energy_entropy"] = float(-np.sum(pn[pn > 0] * np.log(pn[pn > 0])))
    # ridge: per STFT column argmax frequency
    ridge = f[np.argmax(Z, axis=0)]
    out["w_cwt_ridge_mean"] = float(np.mean(ridge))
    out["w_cwt_ridge_sd"] = float(np.std(ridge))
    return out


def _f_entropy_nonlinear(x: np.ndarray, fs: float) -> dict[str, float]:
    out: dict[str, float] = {k: float("nan") for k in
                             ("w_perm_entropy3", "w_sampen_approx", "w_approx_entropy",
                              "w_mse_s2", "w_mse_s4", "w_mse_s8", "w_dfa_alpha", "w_higuchi_fd",
                              "w_katz_fd", "w_petrosian_fd", "w_rec_rate", "w_determinism",
                              "w_laminarity", "w_lz_complexity", "w_hurst", "w_lyap_est")}
    if x.size < 32:
        return out
    d = np.diff(x)
    q = np.digitize(x[:_CAP], np.percentile(x[:_CAP], [25, 50, 75]))
    words = [tuple(q[i:i + 3]) for i in range(0, q.size - 3)]
    if words:
        cnt: dict[tuple[int, ...], int] = {}
        for wd in words:
            cnt[wd] = cnt.get(wd, 0) + 1
        n = len(words)
        ps = np.array(list(cnt.values()), dtype=float) / n
        out["w_perm_entropy3"] = float(-np.sum(ps * np.log(ps)) / math.log(27))
    out["w_sampen_approx"] = _sample_entropy(x)
    out["w_approx_entropy"] = _approx_entropy(x)
    for scale, key in ((2, "w_mse_s2"), (4, "w_mse_s4"), (8, "w_mse_s8")):
        n = x.size // scale
        if n >= 32:
            out[key] = _sample_entropy(x[: n * scale].reshape(n, scale).mean(axis=1))
    y = np.cumsum(x[:8192] - x[:8192].mean())
    ns = [2 ** k for k in range(5, max(6, int(np.log2(max(2, y.size))) - 1))]
    fl = []
    for n in ns:
        if n < 8 or y.size < 2 * n:
            continue
        seg = y[: y.size - y.size % n].reshape(-1, n)
        tt = np.arange(n)
        fit = np.empty_like(seg)
        for i in range(seg.shape[0]):
            c = np.polyfit(tt, seg[i], 1)
            fit[i] = c[0] * tt + c[1]
        fl.append(float(np.sqrt(np.mean((seg - fit) ** 2))))
    if len(ns) >= 3 and len(fl) == len(ns) and np.all(np.array(fl) > 0):
        out["w_dfa_alpha"] = float(np.polyfit(np.log(ns), np.log(fl), 1)[0])
    # Higuchi
    ks = range(1, 9)
    lk = []
    ok = True
    for k in ks:
        lm = []
        for m0 in range(1, k + 1):
            idx = np.arange(m0 - 1, x.size, k)
            if idx.size < 3:
                ok = False
                break
            nm = (idx.size - 1) * k
            lm.append(np.sum(np.abs(np.diff(x[idx]))) * (x.size - 1) / (nm * k))
        if not ok or not lm:
            ok = False
            break
        lk.append(float(np.mean(lm)))
    if ok and lk and np.all(np.array(lk) > 0):
        out["w_higuchi_fd"] = float(-np.polyfit(np.log(list(ks)), np.log(lk), 1)[0])
    L = float(np.sum(np.abs(d)))
    a = float(np.max(x) - np.min(x))
    if L > 0 and a > 0 and d.size > 1:
        out["w_katz_fd"] = float(math.log10(d.size) /
                                 (math.log10(d.size) + math.log10(a / L)))
    nsc = int(np.sum(np.diff(np.sign(d)) != 0)) if d.size else 0
    out["w_petrosian_fd"] = float(math.log10(d.size) /
                                  (math.log10(d.size) + math.log10(d.size / (d.size + 0.4 * nsc)))
                                  ) if d.size else float("nan")
    m = x[:2048]
    dm = np.abs(m[:, None] - m[None, :])
    eps = float(np.percentile(dm, 90))
    R = dm <= eps
    out["w_rec_rate"] = float(R.mean())
    out["w_determinism"] = float(np.mean([R[i, i + 1] for i in range(m.size - 1)]))
    vert = float(np.mean([R[i, i + 2] & R[i + 1, i + 2] for i in range(m.size - 2)])) \
        if m.size > 2 else float("nan")
    out["w_laminarity"] = vert
    # Lempel-Ziv on quartile symbols
    s = np.digitize(m, np.percentile(m, [25, 50, 75])).astype(np.int8)
    seen = set()
    c, cur = 1, ""
    for ch in s.tolist():
        cur += str(ch)
        if cur not in seen:
            seen.add(cur)
            c += 1
            cur = ""
    n = s.size
    out["w_lz_complexity"] = float(c * math.log(max(2, n)) / max(1, n)) if n else float("nan")
    # Hurst R/S
    hs = []
    ns_h = [2 ** k for k in range(6, max(7, int(np.log2(max(2, m.size))) - 1))]
    for n in ns_h:
        if m.size < 2 * n:
            continue
        seg = m[: m.size - m.size % n].reshape(-1, n)
        rs = []
        for row in seg:
            mm = row.mean()
            dev = np.cumsum(row - mm)
            Rr = float(dev.max() - dev.min())
            Ss = float(row.std())
            if Ss > 0:
                rs.append(Rr / Ss)
        if rs:
            hs.append((math.log(n), math.log(float(np.mean(rs)))))
    if len(hs) >= 3:
        out["w_hurst"] = float(np.polyfit([h[0] for h in hs], [h[1] for h in hs], 1)[0])
    # Lyapunov-style divergence
    z = x[:1024]
    emb, lag = 4, 1
    if z.size > 64:
        T = np.stack([z[i:i + emb] for i in range(0, z.size - emb - 8, lag)])
        if T.shape[0] > 16:
            div = []
            for step in range(1, 9):
                ds = []
                for i in range(0, T.shape[0] - step - 1, max(1, T.shape[0] // 64)):
                    j = i + 10
                    if j + step >= T.shape[0]:
                        continue
                    d0 = np.max(np.abs(T[i] - T[j]))
                    d1 = np.max(np.abs(T[i + step] - T[j + step]))
                    if d0 > 0 and d1 > 0:
                        ds.append(math.log(d1 / d0))
                if ds:
                    div.append((step / fs, float(np.mean(ds))))
            if len(div) >= 3:
                out["w_lyap_est"] = float(np.polyfit([a for a, _ in div], [b for _, b in div],
                                                     1)[0])
    return out


def _f_ar_cepstral(x: np.ndarray, fs: float) -> dict[str, float]:
    keys = ["w_ar_c1", "w_ar_c2", "w_ar_c3", "w_ar_c4", "w_ar_res_var", "w_mfcc1", "w_mfcc2",
            "w_mfcc3", "w_mfcc4", "w_mt_dom_freq"] + [f"w_fb_energy_{i}"
                                                      for i in range(len(_FB_BANDS))]
    out = {k: float("nan") for k in keys}
    if x.size < 64:
        return out
    p = 4
    r = _acf(x, p)
    R = np.array([[r[abs(i - j)] for j in range(1, p + 1)] for i in range(1, p + 1)])
    try:
        coef = np.linalg.solve(R + 1e-9 * np.eye(p), r[1:p + 1])
        res = x[p:] - np.stack([x[p - k: x.size - k] for k in range(1, p + 1)]).T @ coef
        for i, c in enumerate(coef):
            out[f"w_ar_c{i + 1}"] = float(c)
        out["w_ar_res_var"] = float(np.var(res)) if res.size else float("nan")
    except np.linalg.LinAlgError:
        pass
    f, M, P = _spectra(x, fs)
    nyq = fs / 2
    for i, (lo, hi) in enumerate(_FB_BANDS):
        m = (f >= lo * nyq) & (f < hi * nyq)
        out[f"w_fb_energy_{i}"] = float(P[m].sum()) if m.any() else float("nan")
    # mel filter bank (20 triangular bands) + DCT-II -> first 4 cepstral coefficients
    n_mel, fmax = 20, min(nyq, 20000.0)
    mel_hi = 2595.0 * math.log10(1 + fmax / 700.0)
    centres = 700.0 * (10 ** (np.linspace(0, mel_hi, n_mel + 2) / 2595.0) - 1)
    loge = []
    for i in range(n_mel):
        lo, ce, hi = centres[i], centres[i + 1], centres[i + 2]
        m = (f >= lo) & (f <= hi)
        if not m.any() or hi <= lo:
            loge.append(-12.0)
            continue
        wts = 1.0 - np.abs(f[m] - ce) / max(1e-9, (hi - lo) / 2)
        v = float(np.sum(P[m] * np.maximum(wts, 0)))
        loge.append(math.log(v + 1e-12))
    le = np.array(loge)
    for k in range(1, 5):
        out[f"w_mfcc{k}"] = float(np.sum(le * np.cos(math.pi * k * (np.arange(n_mel) + 0.5)
                                                     / n_mel)))
    from scipy import signal as sps
    from scipy.signal.windows import dpss

    ntap = min(256, x.size // 2)
    if ntap >= 32:
        tapers = dpss(ntap, 3, Kmax=5)
        est = []
        for tp in tapers:
            ff, PP = sps.periodogram(x[:ntap] * tp, fs=fs)
            est.append(PP)
        out["w_mt_dom_freq"] = float(ff[int(np.argmax(np.mean(est, axis=0)))])
    return out


def _f_decomposition(x: np.ndarray, fs: float) -> dict[str, float]:
    """Tier-3 decomposition features. SSA (singular-spectrum analysis of the Hankel embedding)
    is pure numpy; EMD and VMD need optional packages (`emd`, `vmdpy`) and yield NaN when they
    are absent - the missing-value policy is documented per feature and enforced by the
    train-fold-only column filter."""
    out: dict[str, float] = {k: float("nan") for k in
                             ("w_ssa_sv1_energy_frac", "w_ssa_sv_entropy",
                              "w_emd_imf1_energy_frac", "w_vmd_mode1_energy_frac")}
    n = x.size
    lag = int(min(64, max(4, n // 4)))
    if n >= 2 * lag:
        H = np.lib.stride_tricks.sliding_window_view(x[:min(n, _CAP)], lag)
        try:
            sv = np.linalg.svd(H, compute_uv=False)
            e = sv ** 2
            tot = float(e.sum())
            if tot > 0:
                out["w_ssa_sv1_energy_frac"] = float(e[0] / tot)
                ps = e[e > 0] / tot
                out["w_ssa_sv_entropy"] = float(-np.sum(ps * np.log(ps)))
        except np.linalg.LinAlgError:
            pass
    try:                                             # optional dependency: EMD
        import emd  # type: ignore[import-not-found]

        imf = emd.sift.sift(x[:min(n, 4096)])
        e = np.sum(np.asarray(imf, dtype=float) ** 2, axis=0)
        tot = float(e.sum())
        if tot > 0 and e.size:
            out["w_emd_imf1_energy_frac"] = float(e[0] / tot)
    except Exception:
        pass
    try:                                             # optional dependency: VMD
        from vmdpy import VMD  # type: ignore[import-not-found]

        u, _, _ = VMD(x[:min(n, 4096)], 2000.0, 0.2, 3, 1, 1e-7, 0)
        e = np.sum(np.asarray(u, dtype=float) ** 2, axis=1)
        tot = float(e.sum())
        if tot > 0 and e.size:
            out["w_vmd_mode1_energy_frac"] = float(e[0] / tot)
    except Exception:
        pass
    return out


_COMPUTERS: dict[str, Any] = {
    "time_stats": _f_time_stats,
    "energy": _f_energy,
    "envelope": _f_envelope,
    "dynamics": _f_dynamics,
    "peaks_events": _f_peaks_events,
    "quality": _f_quality,
    "time_frequency": _f_time_frequency,
    "entropy_nonlinear": _f_entropy_nonlinear,
    "ar_cepstral": _f_ar_cepstral,
    "decomposition": _f_decomposition,
}


def compute_features(x: np.ndarray, fs: float, bands: list[list[float]],
                     tiers: tuple[int, ...] | list[int] = (1, 2, 3),
                     families: list[str] | None = None) -> dict[str, float]:
    """Evaluate every selected family once and return exactly the registered ids."""
    x = np.asarray(x, dtype=float)
    specs = registry_specs(bands, tiers, families)
    wanted = {sp.id for sp in specs}
    fams = {sp.family for sp in specs}
    values: dict[str, float] = {}
    for fam in sorted(fams):
        if fam == "spectral":
            values.update(_f_spectral(x, fs, bands))
        elif fam in _COMPUTERS:
            values.update(_COMPUTERS[fam](x, fs))
    if "spectral" in fams and "time_frequency" in fams:
        values["w_spec_flux"] = values.get("w_stft_flux_mean", float("nan"))
    out: dict[str, float] = {}
    for sp in specs:
        v = values.get(sp.id, float("nan"))
        out[sp.id] = _finite(float(v)) if v is not None else float("nan")
    assert set(out) == wanted
    return out


def recording_summary(win_feats: list[dict[str, float]]) -> dict[str, float]:
    """Recording-level summary of window-level features: count, mean and sd per column."""
    if not win_feats:
        return {}
    names = list(win_feats[0])
    mat = np.full((len(win_feats), len(names)), np.nan)
    for i, f in enumerate(win_feats):
        for j, nm in enumerate(names):
            mat[i, j] = float(f.get(nm, float("nan")))
    out: dict[str, float] = {"rec_n_windows": float(len(win_feats))}
    for j, nm in enumerate(names):
        col = mat[:, j]
        out[f"{nm}_rec_mean"] = float(np.nanmean(col)) if np.any(np.isfinite(col)) \
            else float("nan")
        out[f"{nm}_rec_sd"] = float(np.nanstd(col)) if np.count_nonzero(np.isfinite(col)) > 1 \
            else 0.0
    return out
