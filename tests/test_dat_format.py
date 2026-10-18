"""Laboratory ``.dat`` reader: layouts, size consistency, text rows, probe, and config checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trapsig.config import ConfigError, DatFormat, config_from_dict, load_config, validate_config
from trapsig.ingest import IngestError, load_signal, probe_dat, read_header
from trapsig.synthetic import SyntheticSpec, write_synthetic_dataset

REPO = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_binary_float32_roundtrip_with_header_and_scale(tmp_path: Path):
    x = np.linspace(-1e-9, 1e-9, 1000).astype("<f4")
    p = _write(tmp_path / "a.dat", b"HDR0" * 4 + x.tobytes())  # 16-byte header
    fmt = DatFormat(dtype="float32", header_bytes=16, scale=2.0, units="nA")
    hdr = read_header(p, 1_000_000.0, fmt)
    assert hdr.n_samples == x.size
    assert hdr.sampling_rate_source == "config"
    assert hdr.units == "nA"
    y, fs = load_signal(p, 1_000_000.0, fmt)
    assert fs == 1_000_000.0
    np.testing.assert_allclose(y, x.astype(float) * 2.0, rtol=0, atol=0)


def test_binary_interleaved_big_endian_int16_selects_column(tmp_path: Path):
    ch0 = np.arange(0, 50, dtype=">i2")
    ch1 = -np.arange(50, dtype=">i2")
    inter = np.empty(100, dtype=">i2")
    inter[0::2], inter[1::2] = ch0, ch1
    p = _write(tmp_path / "b.dat", inter.tobytes())
    fmt = DatFormat(dtype="int16", byteorder="big", n_columns=2, column=1)
    y, _ = load_signal(p, 1000.0, fmt)
    np.testing.assert_array_equal(y, ch1.astype(float))


def test_size_not_a_whole_number_of_frames_is_refused(tmp_path: Path):
    p = _write(tmp_path / "c.dat", b"\x00" * 10)  # float32 needs multiples of 4
    with pytest.raises(IngestError, match="whole number of 4-byte frames"):
        read_header(p, 1000.0, DatFormat(dtype="float32"))


def test_header_larger_than_file_is_refused(tmp_path: Path):
    p = _write(tmp_path / "d.dat", b"\x00" * 8)
    with pytest.raises(IngestError, match="smaller than"):
        read_header(p, 1000.0, DatFormat(dtype="float32", header_bytes=64))


def test_empty_payload_is_refused(tmp_path: Path):
    p = _write(tmp_path / "e.dat", b"\x00" * 16)
    with pytest.raises(IngestError):
        read_header(p, 1000.0, DatFormat(dtype="float32", header_bytes=16))


def test_text_dat_with_header_lines_and_column_selection(tmp_path: Path):
    p = tmp_path / "t.dat"
    p.write_text("instrument: test\nunits: nA\n1.0, 10.0\n2.0, 20.0\n3.0, 30.0\n", encoding="utf-8")
    fmt = DatFormat(encoding="text", header_lines=2, column=1, n_columns=2)
    hdr = read_header(p, 500.0, fmt)
    assert hdr.n_samples == 3
    y, fs = load_signal(p, 500.0, fmt)
    assert fs == 500.0
    np.testing.assert_array_equal(y, [10.0, 20.0, 30.0])


def test_text_dat_non_numeric_row_is_refused(tmp_path: Path):
    p = tmp_path / "u.dat"
    p.write_text("1.0\nnot-a-number\n2.0\n", encoding="utf-8")
    # The header check samples the first and last rows; a bad middle row is caught on full load.
    with pytest.raises(IngestError, match="line 2"):
        load_signal(p, 500.0, DatFormat(encoding="text"))


def test_no_sampling_rate_is_reported_as_missing(tmp_path: Path):
    p = _write(tmp_path / "f.dat", np.zeros(4, dtype="<f4").tobytes())
    hdr = read_header(p, None, DatFormat(dtype="float32"))
    assert hdr.sampling_rate_hz is None
    assert hdr.sampling_rate_source == "missing"
    with pytest.raises(IngestError, match="sampling rate"):
        load_signal(p, None, DatFormat(dtype="float32"))


def test_probe_identifies_float32_from_duration_and_size(tmp_path: Path):
    x = np.full(1000, 1.5e-10, dtype="<f4")
    p = _write(tmp_path / "g.dat", x.tobytes())
    lines = probe_dat(p, rate_hz=1000.0, duration_s=1.0)
    joined = "\n".join(lines)
    assert "float32  little columns=1  header_bytes = 0" in joined
    assert "1.5e-10" in joined
    assert "no layout matches" not in joined


def test_probe_reports_no_match_when_nothing_fits(tmp_path: Path):
    p = _write(tmp_path / "h.dat", b"\x00" * 7)  # 7 bytes: no frame size except 1 byte types
    lines = probe_dat(p, rate_hz=1000.0, duration_s=1.0)
    assert any("no layout matches" in line for line in lines)


def test_probe_is_read_only(tmp_path: Path):
    p = _write(tmp_path / "i.dat", np.zeros(100, dtype="<f4").tobytes())
    before = p.read_bytes()
    probe_dat(p, rate_hz=1000.0)
    assert p.read_bytes() == before
    assert sorted(x.name for x in tmp_path.iterdir()) == ["i.dat"]


def test_dat_config_rejects_bad_values():
    base = {"data": {"dat": {}}}
    bad = {
        "column out of range": {"n_columns": 2, "column": 2},
        "zero scale": {"scale": 0.0},
        "empty units": {"units": "  "},
        "header_lines on binary": {"encoding": "binary", "header_lines": 2},
        "header_bytes on text": {"encoding": "text", "header_bytes": 4},
    }
    for label, dat in bad.items():
        cfg = config_from_dict({"data": {"dat": dat}})
        assert validate_config(cfg), label
    cfg = config_from_dict(base)
    assert validate_config(cfg) == []


def test_unknown_dat_key_is_rejected_with_location():
    with pytest.raises(ConfigError, match="dtpye"):
        config_from_dict({"data": {"dat": {"dtpye": "float32"}}})


def test_demo_custom_config_is_valid():
    cfg = load_config(REPO / "configs" / "demo_custom_dat.yaml")
    assert validate_config(cfg) == []


def test_template_config_parses_and_is_internally_consistent():
    cfg = load_config(REPO / "configs" / "custom_dat.yaml")
    assert cfg.data.file_glob == "*.dat"
    assert cfg.data.label_from == "folder"
    assert cfg.data.allow_file_level_fallback is True  # no replicate information for this dataset
    # Filter stays 'none' until decided, so no LPF cutoff is required. Cross-field rules must pass.
    assert cfg.filters.mode == "none"
    assert validate_config(cfg) == []


def test_dat_synthetic_matches_csv_synthetic_signal(tmp_path: Path):
    spec_csv = SyntheticSpec(replicates_per_class=2, duration_s=0.2, seed=11, file_format="csv")
    spec_dat = SyntheticSpec(replicates_per_class=2, duration_s=0.2, seed=11, file_format="dat")
    write_synthetic_dataset(tmp_path / "csv", spec_csv)
    write_synthetic_dataset(tmp_path / "dat" / "x", spec_dat, class_folders=True)
    csv_file = tmp_path / "csv" / "SYN_A_R00_F0.csv"
    dat_file = tmp_path / "dat" / "x" / "A" / "SYN_A_R00_F0.dat"
    x_csv, _ = load_signal(csv_file, 100_000.0)
    x_dat, _ = load_signal(dat_file, 100_000.0, DatFormat(dtype="float32"))
    assert x_csv.size == x_dat.size
    np.testing.assert_allclose(x_dat, x_csv, rtol=1e-6, atol=0)
    assert (tmp_path / "dat" / "x" / "SYNTHETIC_NOTICE.txt").exists()
