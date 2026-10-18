from __future__ import annotations

import numpy as np
import pytest

from trapsig.filters import (
    FilterSpec,
    gaussian_effective_cutoff_hz,
    gaussian_smooth,
    lowpass,
    notch,
)


def test_gaussian_effective_cutoff_constant():
    # f_c = sqrt(ln(sqrt 2) / (2 pi^2)) / sigma  ~= 0.13251 / sigma
    assert gaussian_effective_cutoff_hz(1.0) == pytest.approx(0.13251, rel=1e-4)
    assert gaussian_effective_cutoff_hz(0.002) == pytest.approx(0.13251 / 0.002, rel=1e-4)


def test_gaussian_effective_cutoff_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        gaussian_effective_cutoff_hz(0.0)


def _magnitude_at(y_impulse_response: np.ndarray, fs: float, f: float) -> float:
    spec = np.fft.rfft(y_impulse_response)
    freqs = np.fft.rfftfreq(y_impulse_response.size, d=1.0 / fs)
    idx = int(np.argmin(np.abs(freqs - f)))
    return float(np.abs(spec[idx]) / np.abs(spec[0]))


def test_gaussian_smoothing_matches_predicted_minus3db_point():
    """The measured -3 dB point of the implemented kernel equals the analytic formula."""
    fs = 100_000.0
    sigma_s = 0.5e-3
    n = 2**18
    impulse = np.zeros(n)
    impulse[n // 2] = 1.0
    h = gaussian_smooth(impulse, fs, sigma_s, truncate=8.0)
    fc = gaussian_effective_cutoff_hz(sigma_s)
    mag = _magnitude_at(h, fs, fc)
    assert mag == pytest.approx(1 / np.sqrt(2), rel=0.02)


def test_lowpass_passes_slow_and_attenuates_fast_tones():
    fs = 100_000.0
    t = np.arange(int(0.5 * fs)) / fs
    slow = np.sin(2 * np.pi * 50.0 * t)
    fast = np.sin(2 * np.pi * 5_000.0 * t)
    y_slow = lowpass(slow, fs, cutoff_hz=500.0, order=4, zero_phase=True)
    y_fast = lowpass(fast, fs, cutoff_hz=500.0, order=4, zero_phase=True)
    core = slice(len(t) // 4, 3 * len(t) // 4)
    assert np.std(y_slow[core]) == pytest.approx(np.std(slow[core]), rel=0.01)
    assert 20 * np.log10(np.std(y_fast[core]) / np.std(fast[core])) < -60


def test_zero_phase_lowpass_has_no_group_delay_shift():
    fs = 10_000.0
    t = np.arange(int(1.0 * fs)) / fs
    x = np.sin(2 * np.pi * 10.0 * t)
    y = lowpass(x, fs, cutoff_hz=100.0, order=4, zero_phase=True)
    core = slice(len(t) // 4, 3 * len(t) // 4)
    lag_samples = np.argmax(np.correlate(y[core] - y[core].mean(), x[core] - x[core].mean(), "full")) - (len(x[core]) - 1)
    assert abs(lag_samples) <= 1


def _tone_amplitude(y: np.ndarray, fs: float, f: float) -> float:
    """Least-squares amplitude of a sinusoid at exactly frequency f."""
    t = np.arange(y.size) / fs
    basis = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return float(np.hypot(coef[0], coef[1]))


def test_notch_removes_mains_tone_and_keeps_other_content():
    fs = 10_000.0
    t = np.arange(int(2.0 * fs)) / fs
    x = np.sin(2 * np.pi * 50.0 * t) + 0.1 * np.sin(2 * np.pi * 300.0 * t)
    y = notch(x, fs, 50.0, q=30.0)
    core = slice(len(t) // 4, 3 * len(t) // 4)
    assert _tone_amplitude(y[core], fs, 50.0) < 0.01
    assert _tone_amplitude(y[core], fs, 300.0) == pytest.approx(0.1, rel=0.02)


def test_lowpass_rejects_cutoff_outside_band():
    with pytest.raises(ValueError):
        lowpass(np.zeros(100), 1000.0, cutoff_hz=600.0)


# --- FilterSpec: the three modes cannot be confused -------------------------

def test_filterspec_rejects_parameters_inconsistent_with_mode():
    with pytest.raises(ValueError):
        FilterSpec(mode="lpf")  # missing cutoff
    with pytest.raises(ValueError):
        FilterSpec(mode="gaussian", lpf_cutoff_hz=100.0, gaussian_sigma_s=1e-3)
    with pytest.raises(ValueError):
        FilterSpec(mode="none", gaussian_sigma_s=1e-3)
    with pytest.raises(ValueError):
        FilterSpec(mode="sinc")  # type: ignore[arg-type]


def test_labels_identify_the_filter_mode_unambiguously():
    lpf = FilterSpec(mode="lpf", lpf_cutoff_hz=4000.0).describe()
    gau = FilterSpec(mode="gaussian", gaussian_sigma_s=1e-3).describe()
    both = FilterSpec(mode="lpf+gaussian", lpf_cutoff_hz=4000.0, gaussian_sigma_s=1e-3).describe()
    assert lpf.startswith("Filter: LPF") and "Gaussian" not in lpf
    assert gau.startswith("Filter: Gaussian") and "LPF" not in gau
    assert both.startswith("Filter: LPF + Gaussian")
    assert "fc=4000 Hz" in lpf
    assert "sigma=1 ms" in gau


def test_combined_mode_applies_both_stages():
    fs = 10_000.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20_000)
    spec_both = FilterSpec(mode="lpf+gaussian", lpf_cutoff_hz=500.0, gaussian_sigma_s=1e-3)
    spec_lpf = FilterSpec(mode="lpf", lpf_cutoff_hz=500.0)
    y_both = spec_both.apply(x, fs)
    y_lpf = spec_lpf.apply(x, fs)
    # The extra Gaussian smoothing must reduce high-frequency variance further.
    assert np.std(np.diff(y_both)) < np.std(np.diff(y_lpf))
