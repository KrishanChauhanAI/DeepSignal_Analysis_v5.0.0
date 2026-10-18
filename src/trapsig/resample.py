"""Anti-aliased integer-ratio decimation.

Decimating without an anti-alias filter folds energy above the output Nyquist
into the output band. This module:

1. Splits each integer decimation ratio into stages of at most ``max_factor``
   (keeps every IIR design well-conditioned; a single 500 000:1 filter is not).
2. Designs a Butterworth low-pass per stage with a passband edge and a stopband
   edge defined relative to that stage's output Nyquist.
3. Filters, then keeps every ``factor``-th sample.

:func:`alias_suppression_db` quantifies the result with a test tone above the
output Nyquist, so the claim "aliasing is suppressed" is measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, buttord, sosfilt, sosfiltfilt


@dataclass(frozen=True)
class AntiAliasSpec:
    passband_fraction: float = 0.8    # passband edge as a fraction of output Nyquist
    stopband_fraction: float = 1.0    # stopband edge as a fraction of output Nyquist
    gpass_db: float = 0.5
    gstop_db: float = 60.0
    zero_phase: bool = True

    def __post_init__(self) -> None:
        if not (0 < self.passband_fraction < self.stopband_fraction <= 1.0):
            raise ValueError("require 0 < passband_fraction < stopband_fraction <= 1")


def integer_ratio(fs_in: float, fs_out: float) -> int:
    ratio = fs_in / fs_out
    if fs_out <= 0 or fs_out >= fs_in:
        raise ValueError("require 0 < fs_out < fs_in")
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"{fs_in:g} / {fs_out:g} = {ratio:g} is not an integer")
    return int(round(ratio))


def factorize_stages(ratio: int, max_factor: int = 10) -> list[int]:
    """Split ``ratio`` into factors, each <= ``max_factor``, multiplying back to ``ratio``.

    Prime factors are packed greedily, largest first. Raises if a prime exceeds
    ``max_factor`` (such a ratio cannot be decimated in integer stages).
    """
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    if ratio == 1:
        return []
    primes: list[int] = []
    n = ratio
    p = 2
    while p * p <= n:
        while n % p == 0:
            primes.append(p)
            n //= p
        p += 1
    if n > 1:
        primes.append(n)
    if max(primes) > max_factor:
        raise ValueError(f"ratio {ratio} has prime factor {max(primes)} > max_factor {max_factor}")
    stages: list[int] = []
    for prime in sorted(primes, reverse=True):
        if stages and stages[-1] * prime <= max_factor:
            stages[-1] *= prime
        else:
            stages.append(prime)
    return stages


def design_stage(fs_in: float, factor: int, spec: AntiAliasSpec) -> np.ndarray:
    """Return an SOS low-pass for one decimation stage (``fs_in -> fs_in/factor``)."""
    nyq_out = fs_in / factor / 2.0
    wp = spec.passband_fraction * nyq_out
    ws = spec.stopband_fraction * nyq_out
    order, wn = buttord(wp, ws, spec.gpass_db, spec.gstop_db, fs=fs_in)
    sos: np.ndarray = butter(order, wn, btype="low", fs=fs_in, output="sos")
    return sos


def decimate_staged(
    x: np.ndarray,
    fs_in: float,
    fs_out: float,
    spec: AntiAliasSpec | None = None,
) -> tuple[np.ndarray, float]:
    """Anti-aliased decimation from ``fs_in`` to ``fs_out``. Returns ``(y, fs_out)``.

    Offline zero-phase by default. ``sosfiltfilt`` needs a signal longer than its
    padding; very short inputs raise ``ValueError``.
    """
    spec = spec or AntiAliasSpec()
    ratio = integer_ratio(fs_in, fs_out)
    y = np.asarray(x, dtype=float)
    fs = float(fs_in)
    for factor in factorize_stages(ratio):
        sos = design_stage(fs, factor, spec)
        if spec.zero_phase:
            try:
                y = np.asarray(sosfiltfilt(sos, y, padtype="odd"), dtype=float)
            except ValueError as exc:
                raise ValueError(f"signal too short for stage factor {factor}: {exc}") from exc
        else:
            y = sosfilt(sos, y)
        y = y[::factor]
        fs = fs / factor
    return y, fs


def alias_suppression_db(
    fs_in: float,
    fs_out: float,
    spec: AntiAliasSpec | None = None,
    duration_s: float = 0.5,
) -> dict[str, float]:
    """Measure passband gain and alias attenuation with pure tones.

    * ``passband_gain_db``: tone at 0.25 * fs_out (should be near 0 dB)
    * ``alias_attenuation_db``: tone at 0.75 * fs_out, which would fold onto
      0.25 * fs_out without anti-aliasing; reported as a positive attenuation.
    """
    spec = spec or AntiAliasSpec()
    n = int(round(duration_s * fs_in))
    t = np.arange(n) / fs_in

    def amp_after(freq: float) -> float:
        x = np.sin(2 * np.pi * freq * t)
        y, _ = decimate_staged(x, fs_in, fs_out, spec)
        core = y[len(y) // 4: 3 * len(y) // 4]
        return float(np.sqrt(2.0) * np.sqrt(np.mean(core**2)))

    pass_amp = amp_after(0.25 * fs_out)
    alias_amp = amp_after(0.75 * fs_out)
    return {
        "passband_gain_db": 20 * np.log10(max(pass_amp, 1e-300)),
        "alias_attenuation_db": -20 * np.log10(max(alias_amp, 1e-300)),
    }
