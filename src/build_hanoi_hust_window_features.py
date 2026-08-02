"""Build a source-only window-level cache for the G2 window-random cell."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from electrical_fm.hanoi_hust import HanoiHustContract, contract_to_json
from electrical_fm.hanoi_hust_features import BLOCK_DIMENSIONS, WINDOW_OFFSETS, window_feature_blocks
from electrical_fm.hanoi_hust_io import parse_source_record, verify_source_authorization
from download_hanoi_hust import ARCHIVE


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FREEZE = ROOT / "research" / "HANOI_HUST_SOURCE_ACCESS_FREEZE.json"
PREACCESS = ROOT / "research" / "HANOI_HUST_V3_PREACCESS_FREEZE.json"
OUTPUT_DIR = ROOT / "artifacts" / "hanoi_hust_window"
OUTPUT_NPZ = OUTPUT_DIR / "source_window_features.npz"
OUTPUT_JSON = OUTPUT_DIR / "source_window_features_metadata.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    staging = path.with_suffix(".writing")
    staging.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(staging, path)


def build_cache(*, output_npz: Path = OUTPUT_NPZ, output_json: Path = OUTPUT_JSON) -> dict[str, Any]:
    freeze = verify_source_authorization(SOURCE_FREEZE, repo_root=ROOT)
    preaccess = json.loads(PREACCESS.read_text(encoding="utf-8"))
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
        for row in preaccess["file_contracts"]
        if row["access_role"] == "source"
    ]
    blocks: dict[str, list[np.ndarray]] = {name: [] for name in BLOCK_DIMENSIONS}
    metadata: dict[str, list[Any]] = {
        "contract_index": [],
        "bearing_id": [],
        "state": [],
        "components": [],
        "bearing_type": [],
        "load_w": [],
        "window_offset": [],
        "window_start": [],
        "record_window_index": [],
    }
    for contract in contracts:
        signal, _ = parse_source_record(ARCHIVE, contract, freeze)
        for offset in WINDOW_OFFSETS:
            window_blocks, starts = window_feature_blocks(signal, offset=offset)
            for row_index, start in enumerate(starts.tolist()):
                for name in BLOCK_DIMENSIONS:
                    blocks[name].append(window_blocks[name][row_index])
                metadata["contract_index"].append(contract.archive_index)
                metadata["bearing_id"].append(contract.bearing_id)
                metadata["state"].append(contract.state)
                metadata["components"].append("+".join(contract.components))
                metadata["bearing_type"].append(contract.bearing_type)
                metadata["load_w"].append(contract.load_w)
                metadata["window_offset"].append(offset)
                metadata["window_start"].append(start)
                metadata["record_window_index"].append(row_index)

    arrays = {name: np.asarray(values, dtype=np.float32) for name, values in blocks.items()}
    count = len(metadata["contract_index"])
    if count == 0 or any(values.shape != (count, BLOCK_DIMENSIONS[name] // 2) for name, values in arrays.items()):
        raise RuntimeError("window cache shape contract failed")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise RuntimeError("window cache contains non-finite values")
    labels = {
        "contract_index": np.asarray(metadata["contract_index"], dtype=np.int16),
        "bearing_id": np.asarray(metadata["bearing_id"]),
        "state": np.asarray(metadata["state"]),
        "components": np.asarray(metadata["components"]),
        "bearing_type": np.asarray(metadata["bearing_type"], dtype=np.int16),
        "load_w": np.asarray(metadata["load_w"], dtype=np.int16),
        "window_offset": np.asarray(metadata["window_offset"], dtype=np.int32),
        "window_start": np.asarray(metadata["window_start"], dtype=np.int32),
        "record_window_index": np.asarray(metadata["record_window_index"], dtype=np.int16),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    staging = output_npz.with_suffix(".writing.npz")
    np.savez_compressed(staging, **arrays, **labels)
    os.replace(staging, output_npz)
    result = {
        "stage": "hanoi_hust_source_window_feature_cache",
        "schema_version": 1,
        "status": "source_window_features_complete_compound_sealed",
        "source_freeze": {"path": SOURCE_FREEZE.relative_to(ROOT).as_posix(), "sha256": _sha256(SOURCE_FREEZE)},
        "archive": {"path": ARCHIVE.relative_to(ROOT).as_posix(), "sha256": _sha256(ARCHIVE)},
        "record_count": len(contracts),
        "window_count": count,
        "bearing_count": len(set(metadata["bearing_id"])),
        "window_offsets": list(WINDOW_OFFSETS),
        "block_shapes": {name: list(values.shape) for name, values in arrays.items()},
        "cache": {"path": output_npz.relative_to(ROOT).as_posix(), "sha256": _sha256(output_npz)},
        "information_boundary": {
            "source_numeric_files_opened": len(contracts),
            "window_level_independent_samples": count,
            "compound_numeric_files_opened": 0,
        },
        "provenance": {"git_commit": _git_commit(), "builder": "src/build_hanoi_hust_window_features.py"},
        "contracts": [contract_to_json(contract) for contract in contracts],
    }
    _atomic_json(output_json, result)
    return result


def main() -> None:
    print(json.dumps(build_cache(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
