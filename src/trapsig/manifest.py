"""Source discovery, grouping, and the master dataset manifest.

The manifest is the source of truth for a run: one row per source file with its
path, hash, label, group identifiers, split, and quality flags. Raw files are
referenced, never copied.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DataConfig, DatFormat
from .ingest import IngestError, read_header

MANIFEST_FIELDS = [
    "sample_id", "original_path", "filename", "source_subdir", "name_annotation", "file_sha256", "size_bytes",
    "file_format", "class", "label_level", "label_source", "experiment_id",
    "replicate_id", "session_id", "group_id", "group_level", "dup_group_id",
    "split", "sampling_rate_hz", "sampling_rate_source", "n_samples", "duration_s",
    "units", "symlink_path", "link_status", "quality_flags",
]

META_COLUMNS = {"file", "class", "replicate_id", "session_id"}


@dataclass(frozen=True)
class SourceRecord:
    path: Path
    rel_path: str
    label: str | None
    replicate_id: str | None
    session_id: str | None
    experiment_id: str
    label_source: str = "none"
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    """Streaming SHA-256 so large files never sit in memory at once."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


def sample_id_for(path: Path, size: int) -> str:
    """Stable short identifier for a source file (path and size)."""
    return hashlib.sha256(f"{path}|{size}".encode()).hexdigest()[:16]


def parse_source_subdir(rel_path: str) -> str:
    """Directory of a file relative to the source root (e.g. ``ClassA/Connected trace``)."""
    return "/".join(rel_path.split("/")[:-1])


def parse_name_annotation(stem: str) -> str:
    """Free text after the first `` - `` in a file stem, kept verbatim and never interpreted."""
    if " - " not in stem:
        return ""
    return stem.split(" - ", 1)[1].strip()


def list_ignored_files(cfg: DataConfig) -> list[tuple[str, str, int]]:
    """Every file under source_dir that the file glob does not match: (relative path, extension, bytes)."""
    if cfg.source_dir is None or not Path(cfg.source_dir).is_dir():
        return []
    root = Path(cfg.source_dir)
    out: list[tuple[str, str, int]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.match(cfg.file_glob):
            out.append((p.relative_to(root).as_posix(), p.suffix.lower() or "(none)", p.stat().st_size))
    return out


def discover_sources(cfg: DataConfig, experiment_id: str) -> list[SourceRecord]:
    """Find data files and attach labels/groups according to configuration.

    Label and metadata resolution never guesses: a file without a metadata row
    gets ``label=None``, and preflight reports it.
    """
    if cfg.source_dir is None or not Path(cfg.source_dir).is_dir():
        return []
    root = Path(cfg.source_dir)
    paths = sorted(p for p in root.rglob(cfg.file_glob) if p.is_file())

    meta: dict[str, dict[str, str]] = {}
    if cfg.label_from == "metadata" and cfg.metadata_csv is not None:
        meta = load_metadata(cfg.metadata_csv)

    records: list[SourceRecord] = []
    for p in paths:
        rel = p.relative_to(root).as_posix()
        label: str | None = None
        rep: str | None = None
        ses: str | None = None
        if cfg.label_from == "folder":
            parts = p.relative_to(root).parts
            label = parts[0] if len(parts) > 1 else None
        elif cfg.label_from == "metadata":
            row = meta.get(rel) or meta.get(p.name)
            if row is not None:
                label = row.get("class") or None
                rep = row.get("replicate_id") or None
                ses = row.get("session_id") or None
        records.append(SourceRecord(p, rel, label, rep, ses, experiment_id,
                                    label_source=cfg.label_from))
    return records


def load_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    """Read the metadata CSV keyed by ``file`` (relative path or basename)."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        missing = META_COLUMNS - cols
        if "file" not in cols or "class" not in cols:
            raise IngestError(f"{path.name}: metadata needs at least columns 'file' and 'class' "
                              f"(missing: {sorted(missing)})")
        out: dict[str, dict[str, str]] = {}
        for row in reader:
            key = (row.get("file") or "").strip()
            if not key:
                continue
            if key in out:
                raise IngestError(f"{path.name}: duplicate metadata row for '{key}'")
            out[key] = {k: (row.get(k) or "").strip() for k in cols}
    return out


def assign_groups(records: list[SourceRecord], allow_file_fallback: bool) -> list[dict[str, Any]]:
    """Choose the strongest available grouping key per record.

    Order: replicate, then session, then file (only if explicitly allowed).
    Returns one dict per record with ``group_id`` and ``group_level`` (or None).
    """
    out: list[dict[str, Any]] = []
    for r in records:
        if r.replicate_id:
            gid, level = f"rep:{r.replicate_id}", "replicate"
        elif r.session_id:
            gid, level = f"ses:{r.session_id}", "session"
        elif allow_file_fallback:
            gid, level = f"file:{r.rel_path}", "file"
        else:
            gid, level = None, None
        out.append({"group_id": gid, "group_level": level})
    return out


def build_manifest(
    records: list[SourceRecord],
    *,
    default_fs: float | None,
    allow_file_fallback: bool,
    label_level: str = "unknown",
    dat: DatFormat | None = None,
) -> list[dict[str, Any]]:
    """Read headers, hash files, detect duplicates, and produce manifest rows.

    Unreadable files are kept with ``quality_flags`` containing ``unreadable`` so
    the manifest still accounts for every source file.
    """
    groups = assign_groups(records, allow_file_fallback)
    rows: list[dict[str, Any]] = []
    for rec, grp in zip(records, groups, strict=True):
        flags = list(rec.quality_flags)
        size = rec.path.stat().st_size
        sha = sha256_file(rec.path)
        row: dict[str, Any] = {
            "sample_id": sample_id_for(rec.path, size),
            "original_path": str(rec.path),
            "filename": rec.path.name,
            "source_subdir": parse_source_subdir(rec.rel_path),
            "name_annotation": parse_name_annotation(rec.path.stem),
            "file_sha256": sha,
            "size_bytes": size,
            "file_format": rec.path.suffix.lower().lstrip("."),
            "class": rec.label or "NA",
            "label_level": label_level,
            "label_source": rec.label_source,
            "experiment_id": rec.experiment_id,
            "replicate_id": rec.replicate_id or "NA",
            "session_id": rec.session_id or "NA",
            "group_id": grp["group_id"] or "NA",
            "group_level": grp["group_level"] or "NA",
            "dup_group_id": "",
            "split": "unassigned",
            "sampling_rate_hz": "",
            "sampling_rate_source": "missing",
            "n_samples": "",
            "duration_s": "",
            "units": "NA",
            "symlink_path": "",
            "link_status": "not_created",
        }
        try:
            hdr = read_header(rec.path, default_fs, dat)
            row["sampling_rate_hz"] = hdr.sampling_rate_hz if hdr.sampling_rate_hz is not None else ""
            row["sampling_rate_source"] = hdr.sampling_rate_source
            row["n_samples"] = hdr.n_samples
            row["units"] = hdr.units or "NA"
            if hdr.sampling_rate_hz:
                row["duration_s"] = hdr.n_samples / hdr.sampling_rate_hz
        except IngestError as exc:
            flags.append("unreadable")
            row["quality_flags"] = ";".join(flags) + f" ({exc})"
        row.setdefault("quality_flags", ";".join(flags))
        rows.append(row)

    # Exact duplicates share a content hash -> same dup_group_id.
    by_sha: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sha.setdefault(row["file_sha256"], []).append(row)
    for sha, members in by_sha.items():
        if len(members) > 1:
            classes = {m["class"] for m in members}
            for m in members:
                m["dup_group_id"] = f"dup:{sha[:12]}"
                # Identical content must never be split across CV folds, so it shares one group.
                m["group_id"] = f"dup:{sha[:12]}"
                m["group_level"] = "duplicate"
                flags = [f for f in m["quality_flags"].split(";") if f]
                flags.append("exact_duplicate")
                if len(classes) > 1:
                    flags.append("exact_duplicate_across_classes")
                m["quality_flags"] = ";".join(flags)
    return rows


def write_manifest_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write the manifest with exclusive-create semantics (never overwrites)."""
    from .runs import OverwriteError

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    except FileExistsError as exc:
        raise OverwriteError(f"refusing to overwrite manifest: {path}") from exc
    return path


def read_manifest_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))
