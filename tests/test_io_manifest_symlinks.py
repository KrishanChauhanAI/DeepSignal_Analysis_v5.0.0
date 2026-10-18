from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from trapsig.config import DataConfig
from trapsig.ingest import IngestError, load_signal, read_header
from trapsig.manifest import (
    MANIFEST_FIELDS,
    build_manifest,
    discover_sources,
    load_metadata,
    sha256_file,
    write_manifest_csv,
)
from trapsig.runs import OverwriteError, write_text_exclusive
from trapsig.symlinks import create_split_links, verify_link, write_link_report


def _csv_signal(path: Path, values, fs: float | None = 1000.0) -> None:
    lines = []
    if fs is not None:
        lines.append(f"# sampling_rate_hz: {fs:g}")
    lines += [f"{v:.6e}" for v in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- ingest ----------------------------------------------------------------

def test_csv_header_supplies_sampling_rate(tmp_path):
    p = tmp_path / "a.csv"
    _csv_signal(p, np.arange(50, dtype=float), fs=1000.0)
    hdr = read_header(p)
    assert hdr.sampling_rate_hz == 1000.0
    assert hdr.sampling_rate_source == "header"
    assert hdr.n_samples == 50
    x, fs = load_signal(p)
    assert fs == 1000.0 and x.size == 50


def test_npy_without_rate_uses_configuration_and_says_so(tmp_path):
    p = tmp_path / "a.npy"
    np.save(p, np.ones(20))
    hdr = read_header(p, default_fs=5000.0)
    assert hdr.sampling_rate_source == "config"
    assert read_header(p).sampling_rate_source == "missing"


def test_unsupported_extension_is_refused(tmp_path):
    p = tmp_path / "a.abf"
    p.write_bytes(b"\x00")
    with pytest.raises(IngestError, match="unsupported format"):
        read_header(p, default_fs=1.0)


def test_two_column_nonuniform_timestamps_are_refused(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("# sampling_rate_hz: 10\n0.0,1\n0.1,2\n0.35,3\n", encoding="utf-8")
    with pytest.raises(IngestError, match="not uniform"):
        load_signal(p)


def test_two_column_uniform_timestamps_recover_rate(tmp_path):
    p = tmp_path / "t.csv"
    t = np.arange(20) / 100.0
    p.write_text("\n".join(f"{ti:.6f},{ti * 2:.6f}" for ti in t) + "\n", encoding="utf-8")
    x, fs = load_signal(p)
    assert fs == pytest.approx(100.0)
    assert x.size == 20


# --- metadata, grouping, manifest -----------------------------------------

def _make_dataset(root: Path) -> tuple[Path, Path]:
    src = root / "src"
    src.mkdir()
    rows = ["file,class,replicate_id,session_id"]
    for i, (cls, rep, ses) in enumerate([("A", "R1", "S1"), ("A", "R2", "S2"),
                                          ("B", "R3", "S1"), ("B", "R4", "S2")]):
        name = f"f{i}.csv"
        _csv_signal(src / name, np.sin(np.arange(100) * (i + 1)), fs=1000.0)
        rows.append(f"{name},{cls},{rep},{ses}")
    meta = root / "meta.csv"
    meta.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return src, meta


def test_metadata_duplicate_rows_are_refused(tmp_path):
    meta = tmp_path / "m.csv"
    meta.write_text("file,class\na.csv,A\na.csv,B\n", encoding="utf-8")
    with pytest.raises(IngestError, match="duplicate"):
        load_metadata(meta)


def test_metadata_missing_required_columns_is_refused(tmp_path):
    meta = tmp_path / "m.csv"
    meta.write_text("filename,label\na.csv,A\n", encoding="utf-8")
    with pytest.raises(IngestError, match="needs at least"):
        load_metadata(meta)


def test_manifest_contains_every_source_file_with_groups_and_hashes(tmp_path):
    src, meta = _make_dataset(tmp_path)
    cfg = DataConfig(source_dir=src, file_glob="*.csv", label_from="metadata",
                     metadata_csv=meta, label_level="file")
    recs = discover_sources(cfg, "exp")
    rows = build_manifest(recs, default_fs=1000.0, allow_file_fallback=False, label_level="file")
    assert len(rows) == 4
    assert {r["group_level"] for r in rows} == {"replicate"}
    assert all(r["file_sha256"] == sha256_file(r["original_path"]) for r in rows)
    assert all(r["duration_s"] == pytest.approx(0.1) for r in rows)
    out = write_manifest_csv(rows, tmp_path / "manifest.csv")
    with out.open(encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == MANIFEST_FIELDS


def test_exact_duplicates_share_a_duplicate_group(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _csv_signal(src / "a.csv", np.ones(10))
    (src / "b.csv").write_bytes((src / "a.csv").read_bytes())
    cfg = DataConfig(source_dir=src, file_glob="*.csv", label_from="folder", label_level="file")
    rows = build_manifest(discover_sources(cfg, "e"), default_fs=1000.0,
                          allow_file_fallback=True, label_level="file")
    flagged = [r for r in rows if "exact_duplicate" in r["quality_flags"]]
    assert len(flagged) == 2
    assert flagged[0]["dup_group_id"] == flagged[1]["dup_group_id"] != ""


def test_unreadable_file_is_kept_and_flagged(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "bad.csv").write_text("not,a,number\nx,y,z\n", encoding="utf-8")
    cfg = DataConfig(source_dir=src, file_glob="*.csv", label_from="folder", label_level="file")
    rows = build_manifest(discover_sources(cfg, "e"), default_fs=1000.0,
                          allow_file_fallback=True, label_level="file")
    assert len(rows) == 1
    assert "unreadable" in rows[0]["quality_flags"]


def test_manifest_is_never_overwritten(tmp_path):
    p = tmp_path / "m.csv"
    write_manifest_csv([], p)
    with pytest.raises(OverwriteError):
        write_manifest_csv([], p)


# --- symlinks ---------------------------------------------------------------

def test_symlinks_point_to_originals_and_sources_are_untouched(tmp_path):
    src, meta = _make_dataset(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
    cfg = DataConfig(source_dir=src, file_glob="*.csv", label_from="metadata",
                     metadata_csv=meta, label_level="file")
    rows = build_manifest(discover_sources(cfg, "e"), default_fs=1000.0,
                          allow_file_fallback=False, label_level="file")
    for r in rows:
        r["split"] = "cv_pool"
    links = create_split_links(rows, tmp_path / "links")
    assert all(res.status == "linked" for res in links)
    for res in links:
        link = Path(res.link_path)
        assert link.is_symlink()
        assert Path(os.path.realpath(link)) == Path(res.target_path)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
    assert before == after
    # No copies: the link directory contains only links.
    assert all(p.is_symlink() for p in (tmp_path / "links").rglob("*") if p.is_file() or p.is_symlink())


def test_manifest_only_strategy_creates_no_links(tmp_path):
    src, meta = _make_dataset(tmp_path)
    cfg = DataConfig(source_dir=src, file_glob="*.csv", label_from="metadata",
                     metadata_csv=meta, label_level="file")
    rows = build_manifest(discover_sources(cfg, "e"), default_fs=1000.0,
                          allow_file_fallback=False, label_level="file")
    for r in rows:
        r["split"] = "cv_pool"
    links = create_split_links(rows, tmp_path / "links", strategy="manifest_only")
    assert all(res.status == "manifest_only" for res in links)
    assert not (tmp_path / "links").exists()


def test_unassigned_rows_get_no_links(tmp_path):
    src, meta = _make_dataset(tmp_path)
    cfg = DataConfig(source_dir=src, file_glob="*.csv", label_from="metadata",
                     metadata_csv=meta, label_level="file")
    rows = build_manifest(discover_sources(cfg, "e"), default_fs=1000.0,
                          allow_file_fallback=False, label_level="file")
    links = create_split_links(rows, tmp_path / "links")  # split == 'unassigned'
    assert all(res.status == "manifest_only" for res in links)


def test_existing_link_path_is_not_overwritten(tmp_path):
    src, meta = _make_dataset(tmp_path)
    cfg = DataConfig(source_dir=src, file_glob="*.csv", label_from="metadata",
                     metadata_csv=meta, label_level="file")
    rows = build_manifest(discover_sources(cfg, "e"), default_fs=1000.0,
                          allow_file_fallback=False, label_level="file")
    for r in rows:
        r["split"] = "cv_pool"
    create_split_links(rows, tmp_path / "links")
    second = create_split_links(rows, tmp_path / "links")
    assert all(res.status == "verify_failed" for res in second)


def test_broken_link_is_detected(tmp_path):
    target = tmp_path / "t.csv"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "l.csv"
    os.symlink(target, link)
    target.unlink()
    res = verify_link("sid", link, target)
    assert res.status == "broken"


def test_link_report_is_exclusive(tmp_path):
    p = write_link_report([], tmp_path / "r.csv")
    with pytest.raises(OverwriteError):
        write_link_report([], p)


def test_text_writes_are_exclusive(tmp_path):
    p = write_text_exclusive(tmp_path / "x.txt", "a")
    with pytest.raises(OverwriteError):
        write_text_exclusive(p, "b")
