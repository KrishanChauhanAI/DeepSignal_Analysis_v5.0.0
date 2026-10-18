"""Filtering primitives with an explicit, non-confusable filter-mode model.

Three modes exist and never share parameters implicitly:

* ``lpf``            Butterworth SOS low-pass (zero-phase offline by default)
* ``gaussian``       Gaussian smoothing defined by sigma in seconds
* ``lpf+gaussian``   LPF followed by Gaussian; an explicit ablation, not a new channel

:class:`FilterSpec` refuses to build when the parameters do not match the mode,
and :meth:`FilterSpec.describe` produces the label used in every figure title.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, iirnotch, sosfilt, sosfiltfilt

FilterMode = Literal["none", "lpf", "gaussian", "lpf+gaussian"]

_LABELS: dict[str, str] = {
    "none": "Filter: none",
    "lpf": "Filter: LPF",
    "gaussian": "Filter: Gaussian",
    "lpf+gaussian": "Filter: LPF + Gaussian",
}


def gaussian_effective_cutoff_hz(sigma_s: float) -> float:
    """-3 dB (amplitude) frequency of a Gaussian kernel with std ``sigma_s`` seconds.

    The kernel's frequency response is ``|H(f)| = exp(-2 pi^2 sigma^2 f^2)``.
    Solving ``|H| = 1/sqrt(2)`` gives ``f_c = sqrt(ln(sqrt(2)) / (2 pi^2)) / sigma``
    which evaluates to about ``0.13251 / sigma``.
    """
    if sigma_s <= 0:
        raise ValueError("sigma_s must be positive")
    return math.sqrt(math.log(math.sqrt(2.0)) / (2.0 * math.pi**2)) / sigma_s


def lowpass(
    x: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Butterworth SOS low-pass filter.

    ``zero_phase=True`` uses ``sosfiltfilt`` (offline only): the magnitude response
    is squared (effective order doubles) and the filter is non-causal.
    """
    if not (0 < cutoff_hz < fs / 2):
        raise ValueError(f"cutoff {cutoff_hz} Hz must be in (0, {fs / 2}) Hz")
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    x = np.asarray(x, dtype=float)
    if zero_phase:
        return np.asarray(sosfiltfilt(sos, x, padtype="odd"), dtype=float)
    return np.asarray(sosfilt(sos, x), dtype=float)


def gaussian_smooth(x: np.ndarray, fs: float, sigma_s: float, truncate: float = 4.0) -> np.ndarray:
    """Gaussian smoothing with sigma converted from seconds to samples."""
    if sigma_s <= 0:
        raise ValueError("sigma_s must be positive")
    sigma_samples = sigma_s * fs
    out = gaussian_filter1d(np.asarray(x, dtype=float), sigma=sigma_samples,
                            mode="reflect", truncate=truncate)
    return np.asarray(out, dtype=float)


def notch(x: np.ndarray, fs: float, freq_hz: float, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch at ``freq_hz`` (used before any decimation, see docs)."""
    if not (0 < freq_hz < fs / 2):
        raise ValueError(f"notch {freq_hz} Hz must be in (0, {fs / 2}) Hz")
    b, a = iirnotch(freq_hz, q, fs=fs)
    return np.asarray(filtfilt(b, a, np.asarray(x, dtype=float), padtype="odd"), dtype=float)


@dataclass(frozen=True)
class FilterSpec:
    """A single, unambiguous filter configuration."""

    mode: FilterMode
    lpf_cutoff_hz: float | None = None
    lpf_order: int = 4
    gaussian_sigma_s: float | None = None
    gaussian_truncate: float = 4.0
    lpf_zero_phase: bool = True
    notches_hz: tuple[float, ...] = field(default_factory=tuple)
    notch_q: float = 30.0

    def __post_init__(self) -> None:
        uses_lpf = self.mode in ("lpf", "lpf+gaussian")
        uses_gauss = self.mode in ("gaussian", "lpf+gaussian")
        if self.mode not in _LABELS:
            raise ValueError(f"unknown filter mode {self.mode!r}")
        if uses_lpf != (self.lpf_cutoff_hz is not None):
            raise ValueError(f"mode '{self.mode}' and lpf_cutoff_hz are inconsistent")
        if uses_gauss != (self.gaussian_sigma_s is not None):
            raise ValueError(f"mode '{self.mode}' and gaussian_sigma_s are inconsistent")

    def apply(self, x: np.ndarray, fs: float) -> np.ndarray:
        """Apply the notches (if any), then the mode's filters, in a fixed order."""
        y = np.asarray(x, dtype=float)
        for nf in self.notches_hz:
            y = notch(y, fs, nf, self.notch_q)
        if self.mode in ("lpf", "lpf+gaussian"):
            assert self.lpf_cutoff_hz is not None
            y = lowpass(y, fs, self.lpf_cutoff_hz, self.lpf_order, self.lpf_zero_phase)
        if self.mode in ("gaussian", "lpf+gaussian"):
            assert self.gaussian_sigma_s is not None
            y = gaussian_smooth(y, fs, self.gaussian_sigma_s, self.gaussian_truncate)
        return y

    def describe(self) -> str:
        """Label for figure titles, e.g. ``Filter: LPF (Butterworth order 4, fc=4000 Hz, zero-phase)``."""
        parts: list[str] = []
        if self.mode in ("lpf", "lpf+gaussian"):
            phase = "zero-phase" if self.lpf_zero_phase else "causal"
            parts.append(f"LPF Butterworth order {self.lpf_order}, fc={self.lpf_cutoff_hz:g} Hz, {phase}")
        if self.mode in ("gaussian", "lpf+gaussian"):
            assert self.gaussian_sigma_s is not None
            fc = gaussian_effective_cutoff_hz(self.gaussian_sigma_s)
            parts.append(f"Gaussian sigma={self.gaussian_sigma_s * 1e3:g} ms, fc,eff~{fc:.3g} Hz")
        base = _LABELS[self.mode]
        if not parts:
            return base
        return f"{base} ({'; '.join(parts)})"
