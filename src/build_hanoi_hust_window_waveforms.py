"""Build the source-only raw-window cache required by neural G3 baselines."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from download_hanoi_hust import ARCHIVE
from electrical_fm.hanoi_hust import HanoiHustContract
from electrical_fm.hanoi_hust_features import WINDOW_OFFSETS, WINDOW_SAMPLES, window_starts
from electrical_fm.hanoi_hust_io import parse_source_record, verify_source_authorization


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FREEZE = ROOT / "research" / "HANOI_HUST_SOURCE_ACCESS_FREEZE.json"
PREACCESS = ROOT / "research" / "HANOI_HUST_V3_PREACCESS_FREEZE.json"
OUTPUT_DIR = ROOT / "artifacts" / "hanoi_hust_window"
OUTPUT_NPY = OUTPUT_DIR / "source_window_waveforms.npy"
OUTPUT_JSON = OUTPUT_DIR / "source_window_waveforms_metadata.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    staging = path.with_suffix(".writing")
    staging.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(staging, path)


def _contracts() -> list[HanoiHustContract]:
    preaccess = json.loads(PREACCESS.read_text(encoding="utf-8"))
    return [
        HanoiHustContract(
            archive_index=row["archive_index"], path=row["path"], filename=row["filename"],
            uncompressed_bytes=row["uncompressed_bytes"], compressed_bytes=row["compressed_bytes"],
            crc32=int(row["crc32"], 16), compression_method=row["compression_method"],
            local_header_offset=row["local_header_offset"], state=row["state"],
            components=tuple(row["components"]), bearing_type=row["bearing_type"],
            bearing_id=row["bearing_id"], load_w=row["load_w"], access_role=row["access_role"],
        )
        for row in preaccess["file_contracts"] if row["access_role"] == "source"
    ]


def build_cache(*, output_npy: Path = OUTPUT_NPY, output_json: Path = OUTPUT_JSON) -> dict[str, Any]:
    freeze = verify_source_authorization(SOURCE_FREEZE, repo_root=ROOT)
    contracts = _contracts()
    windows: list[np.ndarray] = []
    metadata: dict[str, list[Any]] = {
        "contract_index": [], "bearing_id": [], "state": [], "bearing_type": [],
        "load_w": [], "window_offset": [], "window_start": [],
    }
    for contract in contracts:
        signal, _ = parse_source_record(ARCHIVE, contract, freeze)
        values = np.asarray(signal, dtype=np.float32).reshape(-1)
        for offset in WINDOW_OFFSETS:
            starts = window_starts(values.size, offset=offset)
            for start in starts.tolist():
                windows.append(np.array(values[start:start + WINDOW_SAMPLES], dtype=np.float32, copy=True))
                metadata["contract_index"].append(contract.archive_index)
                metadata["bearing_id"].append(contract.bearing_id)
                metadata["state"].append(contract.state)
                metadata["bearing_type"].append(contract.bearing_type)
                metadata["load_w"].append(contract.load_w)
                metadata["window_offset"].append(offset)
                metadata["window_start"].append(start)
    waveform_array = np.stack(windows, axis=0).astype(np.float32)
    if waveform_array.ndim != 2 or waveform_array.shape[1] != WINDOW_SAMPLES or not np.isfinite(waveform_array).all():
        raise RuntimeError("raw window waveform cache contract failed")
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    staging = output_npy.with_suffix(".writing.npy")
    np.save(staging, waveform_array, allow_pickle=False)
    os.replace(staging, output_npy)
    payload = {
        "stage": "hanoi_hust_source_window_waveform_cache",
        "schema_version": 1,
        "status": "source_window_waveforms_complete_compound_sealed",
        "source_freeze": {"path": SOURCE_FREEZE.relative_to(ROOT).as_posix(), "sha256": _sha256(SOURCE_FREEZE)},
        "archive": {"path": ARCHIVE.relative_to(ROOT).as_posix(), "sha256": _sha256(ARCHIVE)},
        "window_count": int(len(waveform_array)),
        "window_samples": int(WINDOW_SAMPLES),
        "sample_rate_hz": 51200,
        "window_offsets": list(WINDOW_OFFSETS),
        "bearing_count": len(set(metadata["bearing_id"])),
        "cache": {"path": output_npy.relative_to(ROOT).as_posix(), "sha256": _sha256(output_npy), "bytes": int(output_npy.stat().st_size)},
        "metadata": {key: value for key, value in metadata.items()},
        "information_boundary": {"source_numeric_files_opened": len(contracts), "compound_numeric_files_opened": 0},
        "provenance": {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "builder": "src/build_hanoi_hust_window_waveforms.py"},
    }
    _atomic_json(output_json, payload)
    return payload


def main() -> None:
    result = build_cache()
    print(json.dumps({"stage": result["stage"], "window_count": result["window_count"], "cache": result["cache"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
