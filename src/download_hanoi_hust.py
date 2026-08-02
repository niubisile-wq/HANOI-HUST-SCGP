"""Download the frozen Hanoi HUST archive without opening any MAT member."""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import certifi

from electrical_fm.hanoi_hust import (
    ARCHIVE_BYTES,
    ARCHIVE_ETAG,
    ARCHIVE_LAST_MODIFIED,
    ARCHIVE_URL,
)


ROOT = Path(__file__).resolve().parents[1]
PREACCESS_FREEZE = (
    ROOT / "research" / "HANOI_HUST_V3_PREACCESS_FREEZE.json"
)
OUTPUT_ROOT = ROOT / "data" / "downloads" / "hanoi_hust_v3"
ARCHIVE = OUTPUT_ROOT / "cbv7jyx4p9-3.zip"
MANIFEST = OUTPUT_ROOT / "archive_download_manifest.json"
CHUNK_BYTES = 8 * 1024 * 1024


def _git(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def hash_file(path: Path) -> tuple[int, str]:
    """Return byte count and SHA-256 without interpreting archive content."""
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            observed += len(block)
            digest.update(block)
    return observed, digest.hexdigest()


def _trusted_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context(cafile=certifi.where())
    if context.cert_store_stats()["x509_ca"] <= 0:
        raise RuntimeError("Mendeley trusted CA bundle is empty")
    return context


def _validate_preaccess_freeze() -> dict[str, Any]:
    relative = PREACCESS_FREEZE.relative_to(ROOT).as_posix()
    _git(("ls-files", "--error-unmatch", "--", relative))
    if _git(("status", "--porcelain=v1", "--", relative)):
        raise RuntimeError("Hanoi HUST preaccess freeze has worktree changes")
    freeze = json.loads(PREACCESS_FREEZE.read_text(encoding="utf-8"))
    if (
        freeze.get("status")
        != "archive_byte_acquisition_authorized_mat_numeric_sealed"
        or freeze.get("authorization", {}).get(
            "complete_archive_byte_download"
        )
        is not True
        or freeze.get("information_boundary", {}).get("mat_files_opened") != 0
    ):
        raise RuntimeError("Hanoi HUST byte download is not authorized")
    source_hashes = freeze.get("source_sha256", {})
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("Hanoi HUST source bindings are absent")
    for relative_source, expected_sha256 in source_hashes.items():
        path = ROOT / relative_source
        if (
            not path.is_file()
            or hash_file(path)[1] != expected_sha256
            or _git(
                ("status", "--porcelain=v1", "--", relative_source)
            )
        ):
            raise RuntimeError(
                f"Frozen Hanoi HUST source changed: {relative_source}"
            )
    return freeze


def _validate_headers(headers: Any) -> None:
    if (
        headers.get("Content-Length") != str(ARCHIVE_BYTES)
        or headers.get("ETag") != ARCHIVE_ETAG
        or headers.get("Last-Modified") != ARCHIVE_LAST_MODIFIED
        or headers.get("Content-Type") != "application/zip"
    ):
        raise RuntimeError("Hanoi HUST archive response identity changed")


def _stream_opaque_archive(
    response: BinaryIO,
    destination: Path,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    observed = 0
    with destination.open("xb") as handle:
        while True:
            block = response.read(CHUNK_BYTES)
            if not block:
                break
            observed += len(block)
            if observed > ARCHIVE_BYTES:
                raise RuntimeError("Hanoi HUST archive response overflow")
            digest.update(block)
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    if observed != ARCHIVE_BYTES:
        raise RuntimeError(
            f"Hanoi HUST archive ended early: {observed}/{ARCHIVE_BYTES}"
        )
    return observed, digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    staging = path.with_suffix(".writing")
    staging.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def main() -> None:
    freeze = _validate_preaccess_freeze()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE.with_suffix(".zip.part")
    if partial.exists():
        raise RuntimeError(
            "Partial Hanoi HUST archive exists; remove only after audit"
        )
    if ARCHIVE.exists():
        observed, sha256 = hash_file(ARCHIVE)
        if observed != ARCHIVE_BYTES:
            raise RuntimeError("Existing Hanoi HUST archive length changed")
        print(
            json.dumps(
                {
                    "status": "verified_existing_opaque_archive",
                    "bytes": observed,
                    "sha256": sha256,
                },
                indent=2,
            )
        )
        return

    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "electrical-fm-opaque-download/1"},
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=1800,
            context=_trusted_ssl_context(),
        ) as response:
            _validate_headers(response.headers)
            observed, sha256 = _stream_opaque_archive(response, partial)
        os.replace(partial, ARCHIVE)
    except BaseException:
        # The partial file is deliberately preserved for forensic inspection.
        raise

    manifest = {
        "stage": "hanoi_hust_v3_opaque_archive_download",
        "status": "complete_numeric_members_unopened",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": ARCHIVE.relative_to(ROOT).as_posix(),
        "bytes": observed,
        "sha256": sha256,
        "etag": ARCHIVE_ETAG,
        "last_modified": ARCHIVE_LAST_MODIFIED,
        "preaccess_freeze_sha256": hash_file(PREACCESS_FREEZE)[1],
        "preaccess_git_commit": freeze["provenance"]["git_commit"],
        "mat_members_opened": 0,
        "mat_numeric_values_parsed": 0,
    }
    _atomic_json(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
