"""Strict source-only parser for HANOI HUST MAT exports."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from electrical_fm.hanoi_hust import (
    HanoiHustContract,
)


EXPECTED_KEY_SETS = {
    ("data", "fs", "rpm", "ru"),
    ("data", "fs", "ru_raw"),
}
COMPOUND_KEY_SETS = {
    ("data", "fs"),
    ("data", "fs", "ru_raw"),
    ("data", "fs", "rpm", "ru"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_authorization(
    source_freeze_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Verify the committed implementation hashes authorizing source parse."""
    freeze = json.loads(source_freeze_path.read_text(encoding="utf-8"))
    authorization = freeze.get("authorization", {})
    if (
        freeze.get("status") != "source_numeric_access_frozen"
        or authorization.get("source_numeric_parse") is not True
        or authorization.get("compound_numeric_parse") is not False
    ):
        raise RuntimeError("HANOI HUST source numeric access is not authorized")
    changed = []
    for relative, expected in freeze.get("source_sha256", {}).items():
        path = repo_root / relative
        if not path.is_file() or _sha256(path) != expected:
            changed.append(relative)
    if changed:
        raise RuntimeError(
            f"HANOI HUST source-access implementation changed: {changed}"
        )
    return freeze


def parse_record(
    archive_path: Path,
    contract: HanoiHustContract,
    freeze: dict[str, Any],
    *,
    expected_access_role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one authorized MAT member and return its raw waveform."""
    if contract.access_role != expected_access_role:
        raise RuntimeError(
            f"Only {expected_access_role} HANOI HUST records are authorized"
        )
    source_indices = {
        row["archive_index"]
        for row in freeze.get("record_contracts", [])
        if row.get("access_role") == expected_access_role
    }
    if source_indices and contract.archive_index not in source_indices:
        raise RuntimeError(
            f"HANOI HUST contract is not in the {expected_access_role} partition"
        )
    with zipfile.ZipFile(archive_path) as archive:
        try:
            info = archive.getinfo(contract.path)
        except KeyError as exc:
            raise RuntimeError("HANOI HUST archive member is absent") from exc
        if (
            info.file_size != contract.uncompressed_bytes
            or info.compress_size != contract.compressed_bytes
            or info.CRC != contract.crc32
        ):
            raise RuntimeError("HANOI HUST archive member metadata changed")
        with archive.open(info) as handle:
            payload = handle.read()

    mat = scipy.io.loadmat(
        io.BytesIO(payload),
        squeeze_me=True,
        struct_as_record=False,
    )
    keys = sorted(key for key in mat if not key.startswith("__"))
    expected_key_sets = (
        COMPOUND_KEY_SETS if expected_access_role == "compound" else EXPECTED_KEY_SETS
    )
    if tuple(keys) not in expected_key_sets:
        raise RuntimeError(f"HANOI HUST MAT keys changed: {keys}")
    data = np.asarray(mat["data"], dtype=np.float64).reshape(-1)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise RuntimeError("HANOI HUST source waveform is invalid")
    fs = float(np.asarray(mat["fs"], dtype=np.float64).reshape(-1)[0])
    auxiliary_shapes = {
        key: list(np.asarray(mat[key]).shape)
        for key in keys
        if key not in {"data", "fs"}
    }
    schema = {
        "archive_index": contract.archive_index,
        "path": contract.path,
        "filename": contract.filename,
        "bearing_id": contract.bearing_id,
        "load_w": contract.load_w,
        "components": list(contract.components),
        "state": contract.state,
        "data_points": int(data.size),
        "finite": True,
        "fs_value": fs,
        "mat_keys": keys,
        "auxiliary_shapes": auxiliary_shapes,
        "archive_member_size_verified": True,
        "archive_member_crc32_verified": True,
        "source_role_verified": True,
    }
    return data.astype(np.float32), schema


def parse_source_record(
    archive_path: Path,
    contract: HanoiHustContract,
    source_freeze: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one authorized source MAT member and return its raw waveform."""
    return parse_record(
        archive_path,
        contract,
        source_freeze,
        expected_access_role="source",
    )


def parse_compound_record(
    archive_path: Path,
    contract: HanoiHustContract,
    source_freeze: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one authorized compound MAT member and return its raw waveform."""
    return parse_record(
        archive_path,
        contract,
        source_freeze,
        expected_access_role="compound",
    )
