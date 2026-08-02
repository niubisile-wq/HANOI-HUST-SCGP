from __future__ import annotations

import io
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io
import pytest

from electrical_fm.hanoi_hust import HanoiHustContract
from electrical_fm.hanoi_hust_io import parse_compound_record, parse_source_record


def _contract() -> HanoiHustContract:
    return HanoiHustContract(
        archive_index=0,
        path="demo/B500.mat",
        filename="B500.mat",
        uncompressed_bytes=0,
        compressed_bytes=0,
        crc32=0,
        compression_method=8,
        local_header_offset=0,
        state="B",
        components=("ball",),
        bearing_type=5,
        bearing_id="B5",
        load_w=0,
        access_role="source",
    )


def _freeze() -> dict[str, object]:
    return {
        "status": "source_numeric_access_frozen",
        "record_contracts": [{"archive_index": 0, "access_role": "source"}],
    }


def _compound_freeze() -> dict[str, object]:
    return {
        "status": "source_numeric_access_frozen",
        "record_contracts": [{"archive_index": 0, "access_role": "compound"}],
    }


def _zip_payload(*, with_ru_raw: bool) -> tuple[bytes, int, int, int]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        payload = {
            "data": np.arange(16, dtype=np.float64).reshape(-1, 1),
            "fs": np.asarray([[24.93]], dtype=np.float64),
        }
        if with_ru_raw:
            payload["ru_raw"] = np.arange(8, dtype=np.float64).reshape(-1, 1)
        else:
            payload["rpm"] = np.arange(8, dtype=np.float64).reshape(-1, 1)
            payload["ru"] = np.arange(8, dtype=np.float64).reshape(-1, 1)
        sio = io.BytesIO()
        scipy.io.savemat(sio, payload, do_compression=False)
        zf.writestr("demo/B500.mat", sio.getvalue())
    buffer.seek(0)
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as zf:
        info = zf.getinfo("demo/B500.mat")
    return buffer.getvalue(), info.file_size, info.compress_size, info.CRC


def test_parse_source_record_accepts_both_key_sets(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    payload, file_size, compress_size, crc32 = _zip_payload(with_ru_raw=True)
    archive.write_bytes(payload)
    contract = HanoiHustContract(
        archive_index=0,
        path="demo/B500.mat",
        filename="B500.mat",
        uncompressed_bytes=file_size,
        compressed_bytes=compress_size,
        crc32=crc32,
        compression_method=8,
        local_header_offset=0,
        state="B",
        components=("ball",),
        bearing_type=5,
        bearing_id="B5",
        load_w=0,
        access_role="source",
    )
    signal, schema = parse_source_record(archive, contract, _freeze())
    assert signal.shape == (16,)
    assert schema["mat_keys"] == ["data", "fs", "ru_raw"]
    assert schema["finite"] is True


def test_parse_source_record_rejects_wrong_role(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    payload, file_size, compress_size, crc32 = _zip_payload(with_ru_raw=False)
    archive.write_bytes(payload)
    contract = replace(_contract(), access_role="compound")
    with pytest.raises(RuntimeError, match="Only source"):
        parse_source_record(archive, contract, _freeze())


def test_parse_compound_record_accepts_compound_payload(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    payload, file_size, compress_size, crc32 = _zip_payload(with_ru_raw=False)
    archive.write_bytes(payload)
    contract = replace(
        _contract(),
        uncompressed_bytes=file_size,
        compressed_bytes=compress_size,
        crc32=crc32,
        access_role="compound",
    )
    signal, schema = parse_compound_record(
        archive,
        contract,
        _compound_freeze(),
    )
    assert signal.shape == (16,)
    assert schema["mat_keys"] == ["data", "fs", "rpm", "ru"]
    assert schema["source_role_verified"] is True
