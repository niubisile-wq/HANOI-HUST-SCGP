"""Build the authorized HANOI HUST source feature cache."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from download_hanoi_hust import ARCHIVE
from electrical_fm.hanoi_hust import HanoiHustContract, contract_to_json
from electrical_fm.hanoi_hust_features import (
    BLOCK_DIMENSIONS,
    WINDOW_OFFSETS,
    record_feature_blocks,
)
from electrical_fm.hanoi_hust_io import (
    parse_source_record,
    verify_source_authorization,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FREEZE = ROOT / "research" / "HANOI_HUST_SOURCE_ACCESS_FREEZE.json"
OUTPUT_DIR = ROOT / "artifacts" / "hanoi_hust"
OUTPUT_NPZ = OUTPUT_DIR / "source_features.npz"
OUTPUT_JSON = OUTPUT_DIR / "source_features_metadata.json"


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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    staging = path.with_suffix(".writing")
    staging.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def build_cache() -> dict[str, Any]:
    """Parse source records only and aggregate registered feature views."""
    if not SOURCE_FREEZE.is_file():
        raise RuntimeError("HANOI HUST source-access freeze is absent")
    source_freeze = verify_source_authorization(
        SOURCE_FREEZE,
        repo_root=ROOT,
    )
    preaccess = json.loads(
        (ROOT / "research" / "HANOI_HUST_V3_PREACCESS_FREEZE.json").read_text(
            encoding="utf-8"
        )
    )
    contracts = [
        HanoiHustContract(
            archive_index=row["archive_index"],
            path=row["path"],
            filename=row["filename"],
            uncompressed_bytes=row["uncompressed_bytes"],
            compressed_bytes=row["compressed_bytes"],
            crc32=int(row["crc32"], 16),
            compression_method=row["compression_method"],
            local_header_offset=row["local_header_offset"],
            state=row["state"],
            components=tuple(row["components"]),
            bearing_type=row["bearing_type"],
            bearing_id=row["bearing_id"],
            load_w=row["load_w"],
            access_role=row["access_role"],
        )
        for row in preaccess.get("file_contracts", [])
        if row.get("access_role") == "source"
    ]
    blocks: dict[str, list[list[np.ndarray]]] = {
        name: [] for name in BLOCK_DIMENSIONS
    }
    schemas = []
    window_starts = []
    archive_hash = _sha256(ARCHIVE)
    for contract in contracts:
        signal, schema = parse_source_record(
            ARCHIVE,
            contract,
            source_freeze,
        )
        record_views = {name: [] for name in BLOCK_DIMENSIONS}
        starts_by_offset = []
        for offset in WINDOW_OFFSETS:
            features, starts = record_feature_blocks(
                signal,
                offset=offset,
            )
            for name, values in features.items():
                record_views[name].append(values)
            starts_by_offset.append(starts.tolist())
        for name in BLOCK_DIMENSIONS:
            blocks[name].append(record_views[name])
        schemas.append(schema)
        window_starts.append(starts_by_offset)

    arrays = {
        name: np.asarray(rows, dtype=np.float32) for name, rows in blocks.items()
    }
    expected_records = len(contracts)
    for name, values in arrays.items():
        expected = (
            expected_records,
            len(WINDOW_OFFSETS),
            BLOCK_DIMENSIONS[name],
        )
        if values.shape != expected or not np.isfinite(values).all():
            raise RuntimeError(f"HANOI HUST cache block changed: {name}")
    labels = {
        "contract_index": np.asarray([row.archive_index for row in contracts], dtype=np.int16),
        "bearing_type": np.asarray([row.bearing_type for row in contracts], dtype=np.int16),
        "load_w": np.asarray([row.load_w for row in contracts], dtype=np.int16),
        "state": np.asarray([row.state for row in contracts]),
        "components": np.asarray(["+".join(row.components) for row in contracts]),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    staging = OUTPUT_NPZ.with_suffix(".writing.npz")
    np.savez_compressed(staging, **arrays, **labels)
    os.replace(staging, OUTPUT_NPZ)
    payload = {
        "stage": "hanoi_hust_source_feature_cache",
        "schema_version": 1,
        "status": "source_features_complete_compound_sealed",
        "source_freeze": {
            "path": SOURCE_FREEZE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(SOURCE_FREEZE),
        },
        "archive": {
            "path": ARCHIVE.relative_to(ROOT).as_posix(),
            "sha256": archive_hash,
        },
        "record_count": expected_records,
        "contract_indices": [row.archive_index for row in contracts],
        "contracts": [contract_to_json(row) for row in contracts],
        "window_offsets": list(WINDOW_OFFSETS),
        "window_starts": window_starts,
        "block_shapes": {name: list(values.shape) for name, values in arrays.items()},
        "parser_schemas": schemas,
        "cache": {
            "path": OUTPUT_NPZ.relative_to(ROOT).as_posix(),
            "sha256": _sha256(OUTPUT_NPZ),
        },
        "information_boundary": {
            "source_numeric_files_opened": expected_records,
            "compound_numeric_files_opened": 0,
            "transient_numeric_files_opened": 0,
            "window_level_independent_samples": 0,
        },
        "provenance": {
            "git_commit": _git(("rev-parse", "HEAD")),
            "builder": "src/build_hanoi_hust_source_features.py",
        },
    }
    _atomic_json(OUTPUT_JSON, payload)
    return payload


def main() -> None:
    print(json.dumps(build_cache(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
