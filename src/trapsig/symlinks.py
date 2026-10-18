"""Logical split directories built from symbolic links.

Rules (design document, Section 7):

* Original files are never copied, moved, or modified.
* Links are relative where possible and are verified after creation.
* If the platform refuses symlinks, the module records ``manifest_only`` for
  each row and creates nothing. It never falls back to copying.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LinkStatus = Literal["linked", "manifest_only", "broken", "verify_failed"]


@dataclass(frozen=True)
class LinkResult:
    sample_id: str
    link_path: str
    target_path: str
    status: LinkStatus
    message: str


def link_name(sample_id: str, filename: str) -> str:
    """Stable, collision-free name: the sample ID prefixes the original filename."""
    return f"{sample_id}__{filename}"


def create_split_links(
    rows: list[dict[str, Any]],
    link_root: str | Path,
    *,
    strategy: Literal["symlink", "manifest_only"] = "symlink",
) -> list[LinkResult]:
    """Create ``link_root/<split>/<class>/<sample_id>__<filename>`` links.

    ``rows`` need keys: ``sample_id``, ``original_path``, ``filename``,
    ``split``, ``class``. Rows whose split is not a real split are skipped.
    """
    root = Path(link_root)
    results: list[LinkResult] = []
    for row in rows:
        sid = str(row["sample_id"])
        target = Path(row["original_path"]).resolve()
        if row.get("split") in (None, "", "unassigned", "excluded"):
            results.append(LinkResult(sid, "", str(target), "manifest_only", "not assigned to a split"))
            continue
        if strategy == "manifest_only":
            results.append(LinkResult(sid, "", str(target), "manifest_only", "strategy=manifest_only"))
            continue
        link = root / str(row["split"]) / str(row.get("class", "NA")) / link_name(sid, str(row["filename"]))
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            results.append(LinkResult(sid, str(link), str(target), "verify_failed",
                                      "link path already exists; not overwritten"))
            continue
        try:
            rel = os.path.relpath(target, link.parent)
            os.symlink(rel, link)
        except (OSError, NotImplementedError) as exc:
            results.append(LinkResult(sid, "", str(target), "manifest_only",
                                      f"symlink unavailable ({exc.__class__.__name__}); manifest-only"))
            continue
        results.append(verify_link(sid, link, target))
    return results


def verify_link(sample_id: str, link: Path, target: Path) -> LinkResult:
    """Check the link resolves to the original target and has the same size."""
    if not link.is_symlink():
        return LinkResult(sample_id, str(link), str(target), "verify_failed", "not a symlink")
    if not link.exists():
        return LinkResult(sample_id, str(link), str(target), "broken", "link target does not exist")
    if Path(os.path.realpath(link)) != Path(os.path.realpath(target)):
        return LinkResult(sample_id, str(link), str(target), "verify_failed", "resolves to a different file")
    if link.stat().st_size != target.stat().st_size:
        return LinkResult(sample_id, str(link), str(target), "verify_failed", "size mismatch")
    return LinkResult(sample_id, str(link), str(target), "linked", "ok")


def write_link_report(results: list[LinkResult], path: str | Path) -> Path:
    from .runs import OverwriteError

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sample_id", "link_path", "target_path", "status", "message"])
            for r in results:
                w.writerow([r.sample_id, r.link_path, r.target_path, r.status, r.message])
    except FileExistsError as exc:
        raise OverwriteError(f"refusing to overwrite {path}") from exc
    return path
