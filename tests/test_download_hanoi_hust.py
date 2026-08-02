from __future__ import annotations

import hashlib
from io import BytesIO

import pytest

import download_hanoi_hust


def test_stream_opaque_archive_hashes_without_parsing(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"opaque-target-and-source-bytes"
    monkeypatch.setattr(download_hanoi_hust, "ARCHIVE_BYTES", len(payload))
    destination = tmp_path / "archive.part"
    observed, sha256 = download_hanoi_hust._stream_opaque_archive(
        BytesIO(payload),
        destination,
    )
    assert observed == len(payload)
    assert sha256 == hashlib.sha256(payload).hexdigest()
    assert destination.read_bytes() == payload


def test_stream_opaque_archive_rejects_short_and_long_responses(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(download_hanoi_hust, "ARCHIVE_BYTES", 4)
    with pytest.raises(RuntimeError, match="ended early"):
        download_hanoi_hust._stream_opaque_archive(
            BytesIO(b"abc"),
            tmp_path / "short.part",
        )
    with pytest.raises(RuntimeError, match="overflow"):
        download_hanoi_hust._stream_opaque_archive(
            BytesIO(b"abcde"),
            tmp_path / "long.part",
        )


def test_validate_headers_requires_frozen_remote_identity(
    monkeypatch,
) -> None:
    headers = {
        "Content-Length": "10",
        "ETag": '"etag"',
        "Last-Modified": "timestamp",
        "Content-Type": "application/zip",
    }
    monkeypatch.setattr(download_hanoi_hust, "ARCHIVE_BYTES", 10)
    monkeypatch.setattr(download_hanoi_hust, "ARCHIVE_ETAG", '"etag"')
    monkeypatch.setattr(
        download_hanoi_hust,
        "ARCHIVE_LAST_MODIFIED",
        "timestamp",
    )
    download_hanoi_hust._validate_headers(headers)
    headers["Content-Type"] = "text/html"
    with pytest.raises(RuntimeError, match="identity changed"):
        download_hanoi_hust._validate_headers(headers)
