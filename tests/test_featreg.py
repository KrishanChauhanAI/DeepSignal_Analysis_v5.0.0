"""Feature-registry tests: counts, tiers, determinism, NaN semantics, recording level."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trapsig import featreg
from trapsig.config import load_config
from trapsig.quad import window_feature_names, window_features

BANDS = [[1.0, 10.0], [10.0, 100.0], [100.0, 1000.0], [1000.0, 5000.0], [45.0, 55.0]]


def test_registry_has_more_than_100_features_with_metadata() -> None:
    specs = featreg.registry_specs(BANDS, tiers=(1, 2, 3))
    assert len(specs) > 100
    for sp in specs:
        assert sp.description and sp.unit          # params may be empty (no tunable knobs)
        assert sp.tier in (1, 2, 3)
        assert sp.family in featreg.FAMILIES
    assert len({sp.id for sp in specs}) == len(specs)


def test_tiers_are_nested_subsets() -> None:
    t1 = set(featreg.feature_names(BANDS, tiers=(1,)))
    t12 = set(featreg.feature_names(BANDS, tiers=(1, 2)))
    t123 = set(featreg.feature_names(BANDS, tiers=(1, 2, 3)))
    assert t1 < t12 < t123
    assert len(t1) > 20


def test_legacy_engineered_set_is_a_subset() -> None:
    legacy = {"w_mean", "w_sd", "w_rms", "w_energy", "w_zcr", "w_d1_mean", "w_d1_sd",
              "w_hjorth_mobility", "w_hjorth_complexity", "w_spec_centroid", "w_spec_entropy",
              "w_band_1000_5000", "w_env_mean", "w_inst_freq_mean", "w_n_peaks",
              "w_perm_entropy3", "w_sampen_approx"}
    assert legacy <= set(featreg.feature_names(BANDS, tiers=(1, 2)))


def test_family_filter_works() -> None:
    names = featreg.feature_names(BANDS, tiers=(1, 2, 3), families=["time_stats"])
    specs = [sp.id for sp in featreg.registry_specs(BANDS, tiers=(1, 2, 3))
             if sp.family == "time_stats"]
    assert names == specs and len(names) >= 15
    assert "w_rms" in names and "w_zcr" in names


def test_compute_features_matches_names_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(size=1000)
    a = featreg.compute_features(w, 10000.0, BANDS)
    b = featreg.compute_features(w.copy(), 10000.0, BANDS)
    assert list(a) == featreg.feature_names(BANDS)
    for k in a:
        assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k]))


def test_missing_values_are_nan_and_bounded() -> None:
    w = np.zeros(500)
    vals = featreg.compute_features(w, 10000.0, BANDS)
    assert np.isnan(vals["w_burst_mean_dur"])          # no burst in a silent window
    assert np.isnan(vals["w_peak_interval_mean"])      # no peaks
    finite = [v for v in vals.values() if np.isfinite(v)]
    assert all(np.isfinite(np.abs(np.asarray(finite))))


def test_recording_summary_reports_windows_and_moments() -> None:
    rng = np.random.default_rng(0)
    feats = [featreg.compute_features(rng.normal(size=800), 10000.0, BANDS) for _ in range(3)]
    rec = featreg.recording_summary(feats)
    assert rec["rec_n_windows"] == 3
    assert "w_rms_rec_mean" in rec and "w_rms_rec_sd" in rec


def test_quad_delegation_and_short_windows() -> None:
    names = window_feature_names(BANDS)
    rng = np.random.default_rng(0)
    w = rng.normal(size=128)
    vals = window_features(w, 10000.0, BANDS)
    assert list(vals) == names
    assert vals["w_rms"] == pytest.approx(float(np.sqrt(np.mean(w ** 2))), rel=1e-9)



def test_new_config_keys_validate(write_yaml: Any, base_config_dict: dict[str, Any]) -> None:
    from trapsig.config import validate_quad

    d = dict(base_config_dict)
    d["splitting"] = {"n_splits": 2, "n_repeats": 1, "min_groups_per_class": 1}
    d["quad"] = {"enabled": True, "embed_dim": 16, "feature_tiers": [1, 2, 3],
                 "feature_families": ["time_stats", "entropy_nonlinear"],
                 "feature_select": "variance", "triplet_weight": 0.5,
                 "lr_schedule": "cosine", "early_stopping_patience": 2,
                 "open_set_msp_threshold": 0.4}
    cfg = load_config(write_yaml(d, "featreg.yaml"))
    assert validate_quad(cfg) == []
    assert len(set(cfg.quad.feature_tiers)) == 3
    bad = dict(d)
    bad["quad"] = {**d["quad"], "feature_tiers": [4], "lr_schedule": "exp",
                   "feature_families": ["nope"], "open_set_msp_threshold": 1.5,
                   "feature_select": "lasso", "triplet_weight": -1.0}
    errs = " ".join(validate_quad(load_config(write_yaml(bad, "featreg_bad.yaml"))))
    for frag in ("feature_tiers", "lr_schedule", "nope", "open_set_msp_threshold",
                 "feature_select", "triplet_weight"):
        assert frag in errs
