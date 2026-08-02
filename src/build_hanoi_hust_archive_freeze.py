"""Bind the opaque Hanoi HUST archive bytes before source MAT access."""

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
PREACCESS_FREEZE = (
    ROOT / "research" / "HANOI_HUST_V3_PREACCESS_FREEZE.json"
)
OUTPUT = ROOT / "research" / "HANOI_HUST_V3_ARCHIVE_FREEZE.json"
PROTOCOL = ROOT / "research" / "HANOI_HUST_NOACE_PREREGISTRATION.md"
SOURCE_PATHS = (
    PROTOCOL,
    PREACCESS_FREEZE,
    ROOT / "src" / "electrical_fm" / "hanoi_hust.py",
    ROOT / "src" / "download_hanoi_hust.py",
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


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _validate_sources() -> None:
    relative = [_relative(path) for path in SOURCE_PATHS]
    if missing := [path for path in relative if not (ROOT / path).is_file()]:
        raise RuntimeError(f"Hanoi HUST archive sources are absent: {missing}")
    for path in relative:
        _git(("ls-files", "--error-unmatch", "--", path))
    if changed := _git(("status", "--porcelain=v1", "--", *relative)):
        raise RuntimeError(
            f"Hanoi HUST archive sources must be committed: {changed}"
        )


def build_freeze() -> dict[str, Any]:
    """Return an archive identity freeze without opening the ZIP container."""
    _validate_sources()
    preaccess = json.loads(PREACCESS_FREEZE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    archive_bytes, archive_sha256 = hash_file(ARCHIVE)
    if (
        preaccess.get("status")
        != "archive_byte_acquisition_authorized_mat_numeric_sealed"
        or archive_bytes != ARCHIVE_BYTES
        or manifest.get("status") != "complete_numeric_members_unopened"
        or manifest.get("bytes") != archive_bytes
        or manifest.get("sha256") != archive_sha256
        or manifest.get("etag") != ARCHIVE_ETAG
        or manifest.get("mat_members_opened") != 0
        or manifest.get("mat_numeric_values_parsed") != 0
    ):
        raise RuntimeError("Hanoi HUST opaque archive audit failed")
    git_status = _git(
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
    )
    return {
        "stage": "hanoi_hust_v3_archive_freeze",
        "schema_version": 1,
        "status": "opaque_archive_frozen_source_access_not_yet_authorized",
        "archive": {
            "path": _relative(ARCHIVE),
            "bytes": archive_bytes,
            "sha256": archive_sha256,
            "etag": ARCHIVE_ETAG,
        },
        "information_boundary": {
            "archive_bytes_hashed": archive_bytes,
            "zip_container_opened": 0,
            "mat_members_opened": 0,
            "mat_numeric_values_parsed": 0,
        },
        "source_sha256": {
            _relative(path): hash_file(path)[1] for path in SOURCE_PATHS
        },
        "download_manifest": {
            "path": _relative(MANIFEST),
            "sha256": hash_file(MANIFEST)[1],
        },
        "authorization": {
            "source_mat_numeric_parse": False,
            "source_access_requires_separate_committed_freeze": True,
            "compound_mat_numeric_parse": False,
        },
        "provenance": {
            "git_commit": _git(("rev-parse", "HEAD")),
            "bound_files_clean": True,
            "whole_worktree_dirty": bool(git_status),
            "whole_worktree_status_sha256": hashlib.sha256(
                git_status.encode("utf-8")
            ).hexdigest(),
            "builder": "src/build_hanoi_hust_archive_freeze.py",
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
