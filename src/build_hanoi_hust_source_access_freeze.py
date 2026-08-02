"""Bind the Hanoi HUST source-only implementation before MAT numeric access."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

from download_hanoi_hust import ARCHIVE, MANIFEST, hash_file
from electrical_fm.hanoi_hust import ARCHIVE_BYTES, ARCHIVE_ETAG


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "HANOI_HUST_NOACE_PREREGISTRATION.md"
PREACCESS_FREEZE = ROOT / "research" / "HANOI_HUST_V3_PREACCESS_FREEZE.json"
ARCHIVE_FREEZE = ROOT / "research" / "HANOI_HUST_V3_ARCHIVE_FREEZE.json"
OUTPUT = ROOT / "research" / "HANOI_HUST_SOURCE_ACCESS_FREEZE.json"
SOURCE_PATHS = (
    PROTOCOL,
    PREACCESS_FREEZE,
    ARCHIVE_FREEZE,
    ROOT / "src" / "electrical_fm" / "hanoi_hust.py",
    ROOT / "src" / "download_hanoi_hust.py",
    ROOT / "src" / "build_hanoi_hust_preaccess_freeze.py",
    ROOT / "src" / "build_hanoi_hust_archive_freeze.py",
    ROOT / "tests" / "test_hanoi_hust.py",
    ROOT / "tests" / "test_download_hanoi_hust.py",
)


def _git(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _validate_bound_sources() -> None:
    missing = [_relative(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise RuntimeError(f"Hanoi HUST source-access files are absent: {missing}")
    relative = [_relative(path) for path in SOURCE_PATHS]
    for path in relative:
        _git(("ls-files", "--error-unmatch", "--", path))
    changed = _git(("status", "--porcelain=v1", "--", *relative))
    if changed:
        raise RuntimeError(
            "Hanoi HUST source-access files must be committed and unchanged: "
            f"{changed}"
        )


def build_freeze() -> dict[str, Any]:
    """Authorize source numeric parsing without opening any MAT file."""
    _validate_bound_sources()
    preaccess = json.loads(PREACCESS_FREEZE.read_text(encoding="utf-8"))
    archive_freeze = json.loads(ARCHIVE_FREEZE.read_text(encoding="utf-8"))
    archive_bytes, archive_sha256 = hash_file(ARCHIVE)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        preaccess.get("status")
        != "archive_byte_acquisition_authorized_mat_numeric_sealed"
        or preaccess.get("partition", {}).get("source_bearings") != 19
        or preaccess.get("partition", {}).get("source_mat_files") != 57
        or preaccess.get("partition", {}).get("compound_bearings_sealed")
        != 14
        or preaccess.get("partition", {}).get("compound_mat_files_sealed")
        != 42
        or archive_freeze.get("status")
        != "opaque_archive_frozen_source_access_not_yet_authorized"
        or archive_freeze.get("archive", {}).get("bytes") != ARCHIVE_BYTES
        or archive_freeze.get("archive", {}).get("sha256") != archive_sha256
        or manifest.get("status") != "complete_numeric_members_unopened"
        or manifest.get("bytes") != archive_bytes
        or manifest.get("sha256") != archive_sha256
        or manifest.get("mat_members_opened") != 0
        or manifest.get("mat_numeric_values_parsed") != 0
        or archive_bytes != ARCHIVE_BYTES
        or preaccess.get("archive", {}).get("etag") != ARCHIVE_ETAG
    ):
        raise RuntimeError("Hanoi HUST source access prerequisites failed")
    source_contracts = [
        row
        for row in preaccess.get("file_contracts", [])
        if row.get("access_role") == "source"
    ]
    compound_contracts = [
        row
        for row in preaccess.get("file_contracts", [])
        if row.get("access_role") == "compound"
    ]
    if len(source_contracts) != 57 or len(compound_contracts) != 42:
        raise RuntimeError("Hanoi HUST partition changed")
    git_status = _git(
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
    )
    return {
        "stage": "hanoi_hust_source_access_freeze",
        "schema_version": 1,
        "status": "source_numeric_access_frozen",
        "record": preaccess.get("record"),
        "version": preaccess.get("version"),
        "doi": preaccess.get("doi"),
        "title": preaccess.get("title"),
        "license": preaccess.get("license"),
        "archive": {
            "path": _relative(ARCHIVE),
            "bytes": archive_bytes,
            "sha256": archive_sha256,
            "etag": ARCHIVE_ETAG,
        },
        "preaccess_freeze": {
            "path": _relative(PREACCESS_FREEZE),
            "sha256": _sha256(PREACCESS_FREEZE),
        },
        "archive_freeze": {
            "path": _relative(ARCHIVE_FREEZE),
            "sha256": _sha256(ARCHIVE_FREEZE),
        },
        "record_partition": {
            "physical_bearings": preaccess.get("partition", {}).get(
                "physical_bearings"
            ),
            "source_bearings": preaccess.get("partition", {}).get(
                "source_bearings"
            ),
            "source_mat_files": preaccess.get("partition", {}).get(
                "source_mat_files"
            ),
            "compound_bearings_sealed": preaccess.get("partition", {}).get(
                "compound_bearings_sealed"
            ),
            "compound_mat_files_sealed": preaccess.get("partition", {}).get(
                "compound_mat_files_sealed"
            ),
        },
        "record_contracts": [
            {
                "archive_index": row.get("archive_index"),
                "path": row.get("path"),
                "filename": row.get("filename"),
                "access_role": row.get("access_role"),
            }
            for row in preaccess.get("file_contracts", [])
        ],
        "information_boundary": {
            "source_numeric_files_opened": 0,
            "compound_numeric_files_opened": 0,
            "transient_numeric_files_opened": 0,
            "mat_numeric_values_parsed": 0,
            "archive_payload_bytes_read": 0,
            "archive_container_reopened": 0,
        },
        "source_sha256": {
            _relative(path): _sha256(path) for path in SOURCE_PATHS
        },
        "authorization": {
            "source_numeric_parse": True,
            "compound_numeric_parse": False,
            "compound_requires_separate_committed_freeze": True,
        },
        "provenance": {
            "git_commit": _git(("rev-parse", "HEAD")),
            "bound_files_clean": True,
            "whole_worktree_dirty": bool(git_status),
            "whole_worktree_status_sha256": hashlib.sha256(
                git_status.encode("utf-8")
            ).hexdigest(),
            "builder": "src/build_hanoi_hust_source_access_freeze.py",
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def main() -> None:
    payload = build_freeze()
    _atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
