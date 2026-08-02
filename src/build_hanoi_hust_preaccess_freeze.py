"""Freeze Hanoi HUST v3 metadata before any MAT numeric access."""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import certifi

from electrical_fm.hanoi_hust import (
    ARCHIVE_BYTES,
    ARCHIVE_ETAG,
    ARCHIVE_LAST_MODIFIED,
    ARCHIVE_URL,
    DATASET_ID,
    DATASET_VERSION,
    DOI,
    LICENSE,
    METADATA_URL,
    TAIL_BYTES,
    TITLE,
    UPSTREAM_HUST_FEATURE_BLOB,
    UPSTREAM_HUST_FEATURE_BYTES,
    UPSTREAM_REPLICATION_COMMIT,
    build_contracts,
    contract_to_json,
    parse_central_directory_tail,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research" / "HANOI_HUST_V3_PREACCESS_FREEZE.json"
PROTOCOL = ROOT / "research" / "HANOI_HUST_NOACE_PREREGISTRATION.md"
ARCHIVE = (
    ROOT / "data" / "downloads" / "hanoi_hust_v3" / "cbv7jyx4p9-3.zip"
)
SOURCE_PATHS = (
    PROTOCOL,
    ROOT / "src" / "electrical_fm" / "hanoi_hust.py",
    ROOT / "src" / "build_hanoi_hust_preaccess_freeze.py",
    ROOT / "tests" / "test_hanoi_hust.py",
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


def _validate_sources() -> None:
    relative = [_relative(path) for path in SOURCE_PATHS]
    if missing := [path for path in relative if not (ROOT / path).is_file()]:
        raise RuntimeError(f"Hanoi HUST preaccess sources are absent: {missing}")
    for path in relative:
        _git(("ls-files", "--error-unmatch", "--", path))
    if changed := _git(("status", "--porcelain=v1", "--", *relative)):
        raise RuntimeError(
            f"Hanoi HUST preaccess sources must be committed: {changed}"
        )


def _trusted_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context(cafile=certifi.where())
    if context.cert_store_stats()["x509_ca"] <= 0:
        raise RuntimeError("Mendeley trusted CA bundle is empty")
    return context


def _request_json() -> dict[str, Any]:
    request = urllib.request.Request(
        METADATA_URL,
        headers={
            "Accept": "application/vnd.mendeley-public-dataset.1+json",
            "User-Agent": "electrical-fm-metadata-freeze/1",
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=60,
        context=_trusted_ssl_context(),
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_tail() -> tuple[bytes, dict[str, str]]:
    range_start = ARCHIVE_BYTES - TAIL_BYTES
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={
            "Range": f"bytes={range_start}-{ARCHIVE_BYTES - 1}",
            "User-Agent": "electrical-fm-metadata-freeze/1",
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=120,
        context=_trusted_ssl_context(),
    ) as response:
        payload = response.read(TAIL_BYTES + 1)
        headers = {
            "content_range": response.headers.get("Content-Range", ""),
            "content_length": response.headers.get("Content-Length", ""),
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
            "accept_ranges": response.headers.get("Accept-Ranges", ""),
            "final_url": response.geturl(),
        }
    if len(payload) != TAIL_BYTES:
        raise RuntimeError("Hanoi HUST metadata range length changed")
    return payload, headers


def _validate_metadata(metadata: dict[str, Any]) -> None:
    licence = metadata.get("data_licence", {})
    doi = metadata.get("doi", {})
    if (
        metadata.get("id") != DATASET_ID
        or metadata.get("version") != DATASET_VERSION
        or metadata.get("name") != TITLE
        or doi.get("id") != DOI
        or licence.get("short_name") != LICENSE
        or metadata.get("publish_date") != "2023-08-13T06:32:44.009Z"
    ):
        raise RuntimeError("Hanoi HUST public metadata changed")


def build_freeze() -> dict[str, Any]:
    """Return a metadata freeze while keeping every MAT numeric value sealed."""
    _validate_sources()
    if ARCHIVE.exists() or ARCHIVE.with_suffix(".zip.part").exists():
        raise RuntimeError("Hanoi HUST archive exists before preaccess freeze")
    metadata = _request_json()
    _validate_metadata(metadata)
    tail, headers = _request_tail()
    expected_range = (
        f"bytes {ARCHIVE_BYTES - TAIL_BYTES}-{ARCHIVE_BYTES - 1}/"
        f"{ARCHIVE_BYTES}"
    )
    if (
        headers["content_range"] != expected_range
        or headers["content_length"] != str(TAIL_BYTES)
        or headers["etag"] != ARCHIVE_ETAG
        or headers["last_modified"] != ARCHIVE_LAST_MODIFIED
        or headers["accept_ranges"] != "bytes"
    ):
        raise RuntimeError("Hanoi HUST remote archive identity changed")
    entries = parse_central_directory_tail(
        tail,
        range_start=ARCHIVE_BYTES - TAIL_BYTES,
    )
    contracts = build_contracts(entries)
    git_status = _git(
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
    )
    description = str(metadata.get("description", ""))
    return {
        "stage": "hanoi_hust_v3_preaccess_freeze",
        "schema_version": 1,
        "status": "archive_byte_acquisition_authorized_mat_numeric_sealed",
        "record": DATASET_ID,
        "version": DATASET_VERSION,
        "doi": DOI,
        "title": TITLE,
        "license": LICENSE,
        "archive": {
            "url": ARCHIVE_URL,
            "bytes": ARCHIVE_BYTES,
            "etag": ARCHIVE_ETAG,
            "last_modified": ARCHIVE_LAST_MODIFIED,
            "central_directory_tail_bytes_read": TAIL_BYTES,
            "central_directory_tail_sha256": hashlib.sha256(tail).hexdigest(),
            "mat_payload_bytes_read": 0,
        },
        "publisher_metadata": {
            "publish_date": metadata.get("publish_date"),
            "modified_on": metadata.get("modified_on"),
            "created_on": metadata.get("created_on"),
            "description_sha256": hashlib.sha256(
                description.encode("utf-8")
            ).hexdigest(),
        },
        "file_contracts": [contract_to_json(row) for row in contracts],
        "partition": {
            "physical_bearings": 33,
            "source_bearings": 19,
            "source_mat_files": 57,
            "compound_bearings_sealed": 14,
            "compound_mat_files_sealed": 42,
            "loads_w": [0, 200, 400],
        },
        "information_boundary": {
            "dataset_metadata_requests": 1,
            "zip_central_directory_requests": 1,
            "zip_central_directory_bytes_read": TAIL_BYTES,
            "archive_payload_download_requests": 0,
            "mat_files_opened": 0,
            "mat_numeric_values_parsed": 0,
            "upstream_feature_dataframe_reads": 0,
        },
        "pre_freeze_incident": {
            "official_replication_repository_cloned_for_code_review": True,
            "repository_commit": UPSTREAM_REPLICATION_COMMIT,
            "tracked_hust_feature_blob_was_acquired_but_not_opened": True,
            "tracked_hust_feature_blob_sha1": UPSTREAM_HUST_FEATURE_BLOB,
            "tracked_hust_feature_blob_bytes": UPSTREAM_HUST_FEATURE_BYTES,
            "blob_permanently_excluded_from_all_experiments": True,
        },
        "source_sha256": {
            _relative(path): _sha256(path) for path in SOURCE_PATHS
        },
        "authorization": {
            "complete_archive_byte_download": True,
            "archive_sha256_and_crc_verification": True,
            "source_mat_numeric_parse": False,
            "source_parse_requires_committed_access_freeze": True,
            "compound_mat_numeric_parse": False,
            "compound_parse_requires_later_committed_confirmation_freeze": True,
        },
        "provenance": {
            "git_commit": _git(("rev-parse", "HEAD")),
            "bound_files_clean": True,
            "whole_worktree_dirty": bool(git_status),
            "whole_worktree_status_sha256": hashlib.sha256(
                git_status.encode("utf-8")
            ).hexdigest(),
            "builder": "src/build_hanoi_hust_preaccess_freeze.py",
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
