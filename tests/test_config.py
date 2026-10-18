from __future__ import annotations

from pathlib import Path

import pytest

from trapsig.config import ConfigError, config_from_dict, load_config, validate_config


def test_default_config_is_constructible_and_fingerprint_is_stable(base_config_dict):
    a = config_from_dict(base_config_dict)
    b = config_from_dict(base_config_dict)
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 64


def test_fingerprint_changes_with_any_parameter(base_config_dict):
    a = config_from_dict(base_config_dict)
    base_config_dict["events"]["threshold_sigma"] = 6.0
    b = config_from_dict(base_config_dict)
    assert a.fingerprint() != b.fingerprint()


def test_unknown_key_is_rejected_with_its_location(base_config_dict):
    base_config_dict["events"]["threshold_sigmaa"] = 5.0  # typo
    with pytest.raises(ConfigError, match="events.*threshold_sigmaa"):
        config_from_dict(base_config_dict)


def test_unknown_top_level_section_is_rejected(base_config_dict):
    base_config_dict["modelz"] = {}
    with pytest.raises(ConfigError, match="modelz"):
        config_from_dict(base_config_dict)


def test_invalid_literal_is_rejected(base_config_dict):
    base_config_dict["filters"]["mode"] = "butterworth"
    with pytest.raises(ConfigError, match="not in"):
        config_from_dict(base_config_dict)


def test_wrong_scalar_type_is_rejected(base_config_dict):
    base_config_dict["splitting"]["n_splits"] = "five"
    with pytest.raises(ConfigError, match="expected int"):
        config_from_dict(base_config_dict)


def test_bool_is_not_accepted_as_int(base_config_dict):
    base_config_dict["splitting"]["n_splits"] = True
    with pytest.raises(ConfigError):
        config_from_dict(base_config_dict)


def test_load_config_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_load_config_invalid_yaml(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("run: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(p)


def test_consistent_config_has_no_validation_errors(base_config_dict):
    assert validate_config(config_from_dict(base_config_dict)) == []


def test_non_integer_decimation_ratio_is_an_error(base_config_dict):
    base_config_dict["sampling"]["target_rates_hz"] = [30000]
    errs = validate_config(config_from_dict(base_config_dict))
    assert any("integer divisor" in e for e in errs)


def test_lpf_mode_requires_cutoff(base_config_dict):
    base_config_dict["filters"] = {"mode": "lpf"}
    errs = validate_config(config_from_dict(base_config_dict))
    assert any("requires filters.lpf_cutoff_hz" in e for e in errs)


def test_parameter_set_for_unused_mode_is_an_error(base_config_dict):
    base_config_dict["filters"] = {"mode": "lpf", "lpf_cutoff_hz": 100.0, "gaussian_sigma_s": 0.001}
    errs = validate_config(config_from_dict(base_config_dict))
    assert any("does not use it" in e for e in errs)


def test_lpf_cutoff_above_nyquist_is_an_error(base_config_dict):
    base_config_dict["filters"] = {"mode": "lpf", "lpf_cutoff_hz": 60000.0}
    errs = validate_config(config_from_dict(base_config_dict))
    assert any("must be in (0," in e for e in errs)


def test_hysteresis_requires_exit_below_threshold(base_config_dict):
    base_config_dict["events"]["exit_sigma"] = 6.0
    errs = validate_config(config_from_dict(base_config_dict))
    assert any("hysteresis" in e for e in errs)


def test_min_groups_must_cover_folds(base_config_dict):
    base_config_dict["splitting"] = {"n_splits": 5, "min_groups_per_class": 3}
    errs = validate_config(config_from_dict(base_config_dict))
    assert any("min_groups_per_class must be >= n_splits" in e for e in errs)


def test_metadata_label_source_requires_metadata_file(base_config_dict):
    base_config_dict["data"] = {"label_from": "metadata"}
    errs = validate_config(config_from_dict(base_config_dict))
    assert any("requires data.metadata_csv" in e for e in errs)
