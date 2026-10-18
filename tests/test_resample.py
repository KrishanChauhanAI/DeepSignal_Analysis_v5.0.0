from __future__ import annotations

import math

import numpy as np
import pytest

from trapsig.filters import notch
from trapsig.resample import (
    AntiAliasSpec,
    alias_suppression_db,
    decimate_staged,
    factorize_stages,
    integer_ratio,
)


@pytest.mark.parametrize("ratio", [2, 20, 100, 1_000, 10_000, 100_000, 500_000])
def test_stage_factors_multiply_back_and_respect_max(ratio):
    stages = factorize_stages(ratio, max_factor=10)
    assert math.prod(stages) == ratio
    assert all(1 < s <= 10 for s in stages)


def test_prime_factor_larger_than_max_is_rejected():
    with pytest.raises(ValueError, match="prime factor"):
        factorize_stages(13, max_factor=10)


def test_integer_ratio_detects_non_integer_ratio():
    assert integer_ratio(1e6, 50e3) == 20
    with pytest.raises(ValueError):
        integer_ratio(1e6, 30e3)


def test_decimated_length_and_rate_are_exact():
    fs_in, fs_out = 100_000.0, 1_000.0
    x = np.zeros(100_000)
    y, fs = decimate_staged(x, fs_in, fs_out)
    assert fs == fs_out
    assert y.size == 1_000


def test_anti_alias_attenuates_tone_above_output_nyquist():
    m = alias_suppression_db(100_000.0, 1_000.0, duration_s=0.5)
    # A tone at 0.75 * fs_out would fold to 0.25 * fs_out without an anti-alias filter.
    assert m["alias_attenuation_db"] > 40.0


def test_passband_is_preserved_within_half_a_decibel():
    m = alias_suppression_db(100_000.0, 1_000.0, duration_s=0.5)
    assert abs(m["passband_gain_db"]) < 0.5


def test_mains_50hz_does_not_survive_decimation_to_10hz():
    """Mains would alias to 0 Hz at 10 Hz output if it were not removed by the cascade."""
    fs_in = 100_000.0
    t = np.arange(int(20 * fs_in)) / fs_in
    x = 1e-12 * np.sin(2 * np.pi * 50.0 * t)
    y, fs = decimate_staged(x, fs_in, 10.0)
    core = y[len(y) // 4: 3 * len(y) // 4]
    residual_rms = np.sqrt(np.mean(core**2))
    input_rms = 1e-12 / np.sqrt(2)
    assert 20 * np.log10(residual_rms / input_rms) < -60.0


def test_notch_before_decimation_is_the_documented_order():
    """Notching at the original rate removes 50 Hz before any downsampling step."""
    fs_in = 100_000.0
    t = np.arange(int(2 * fs_in)) / fs_in
    x = np.sin(2 * np.pi * 50.0 * t)
    x_notched = notch(x, fs_in, 50.0, q=30.0)
    y, _ = decimate_staged(x_notched, fs_in, 1_000.0)
    assert np.sqrt(np.mean(y[len(y) // 4: 3 * len(y) // 4] ** 2)) < 0.01


def test_invalid_anti_alias_spec_rejected():
    with pytest.raises(ValueError):
        AntiAliasSpec(passband_fraction=0.9, stopband_fraction=0.8)
