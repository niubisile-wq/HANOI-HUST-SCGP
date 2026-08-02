"""Immutable metadata contracts for the Hanoi HUST bearing dataset."""

from __future__ import annotations

import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal


DATASET_ID = "cbv7jyx4p9"
DATASET_VERSION = 3
DOI = "10.17632/cbv7jyx4p9.3"
TITLE = "HUST bearing: a practical dataset for ball bearing fault diagnosis"
LICENSE = "CC BY 4.0"
ARCHIVE_URL = (
    "https://data.mendeley.com/public-api/zip/"
    f"{DATASET_ID}/download/{DATASET_VERSION}"
)
METADATA_URL = (
    f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
    f"?version={DATASET_VERSION}"
)
ARCHIVE_BYTES = 696_985_660
ARCHIVE_ETAG = '"9a68020cb207941ced9f6d1af1484ad2-14"'
ARCHIVE_LAST_MODIFIED = "Fri, 17 Jan 2025 19:08:06 GMT"
CENTRAL_DIRECTORY_OFFSET = 696_971_389
CENTRAL_DIRECTORY_BYTES = 14_249
CENTRAL_DIRECTORY_ENTRIES = 101
TAIL_BYTES = 65_660
UPSTREAM_REPLICATION_COMMIT = "4c9fd8cb3cf9d5aa4ef2b653cb314a0727e12f3d"
UPSTREAM_HUST_FEATURE_BLOB = "6cf9e9179c9a0ce9dc31819fd2e1f341f05e45d4"
UPSTREAM_HUST_FEATURE_BYTES = 246_992

AccessRole = Literal["source", "compound"]
_MAT_NAME = re.compile(r"^(IB|IO|OB|I|O|B|N)([4-8])(00|02|04)\.mat$")
_COMPONENTS = {
    "N": (),
    "I": ("inner",),
    "O": ("outer",),
    "B": ("ball",),
    "IB": ("inner", "ball"),
    "IO": ("inner", "outer"),
    "OB": ("outer", "ball"),
}
_LOAD_W = {"00": 0, "02": 200, "04": 400}
_EXPECTED_BEARINGS = {
    "N": {f"N{index}" for index in range(4, 9)},
    "I": {f"I{index}" for index in range(4, 9)},
    "O": {f"O{index}" for index in range(4, 9)},
    "B": {f"B{index}" for index in range(5, 9)},
    "IB": {f"IB{index}" for index in range(5, 9)},
    "IO": {f"IO{index}" for index in range(4, 9)},
    "OB": {f"OB{index}" for index in range(4, 9)},
}
_DOCUMENTATION_FILES = {"defects.png", "testbench.png"}


@dataclass(frozen=True)
class ZipEntry:
    """One entry decoded from the ZIP central directory."""

    archive_index: int
    path: str
    uncompressed_bytes: int
    compressed_bytes: int
    crc32: int
    compression_method: int
    local_header_offset: int


@dataclass(frozen=True)
class HanoiHustContract:
    """One immutable MAT file and its filename-derived experimental identity."""

    archive_index: int
    path: str
    filename: str
    uncompressed_bytes: int
    compressed_bytes: int
    crc32: int
    compression_method: int
    local_header_offset: int
    state: str
    components: tuple[str, ...]
    bearing_type: int
    bearing_id: str
    load_w: int
    access_role: AccessRole


def parse_central_directory_tail(
    payload: bytes,
    *,
    range_start: int,
    archive_bytes: int = ARCHIVE_BYTES,
) -> tuple[ZipEntry, ...]:
    """Decode a ZIP central directory from a metadata-only trailing range."""
    eocd_signature = b"PK\x05\x06"
    central_signature = b"PK\x01\x02"
    eocd_position = payload.rfind(eocd_signature)
    if eocd_position < 0 or len(payload) - eocd_position < 22:
        raise ValueError("ZIP end-of-central-directory record is absent")
    (
        _,
        disk_number,
        central_disk,
        entries_on_disk,
        total_entries,
        central_bytes,
        central_offset,
        comment_bytes,
    ) = struct.unpack_from("<4s4H2LH", payload, eocd_position)
    if (
        disk_number != 0
        or central_disk != 0
        or entries_on_disk != CENTRAL_DIRECTORY_ENTRIES
        or total_entries != CENTRAL_DIRECTORY_ENTRIES
        or central_bytes != CENTRAL_DIRECTORY_BYTES
        or central_offset != CENTRAL_DIRECTORY_OFFSET
        or comment_bytes != 0
        or range_start + eocd_position + 22 != archive_bytes
    ):
        raise ValueError("ZIP central-directory identity changed")

    position = central_offset - range_start
    if position < 0:
        raise ValueError("Trailing range does not contain the central directory")
    entries: list[ZipEntry] = []
    for archive_index in range(total_entries):
        if payload[position : position + 4] != central_signature:
            raise ValueError("Malformed ZIP central-directory entry")
        (
            _,
            _version_made,
            _version_needed,
            flags,
            compression_method,
            _modified_time,
            _modified_date,
            crc32,
            compressed_bytes,
            uncompressed_bytes,
            name_bytes,
            extra_bytes,
            entry_comment_bytes,
            disk_start,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = struct.unpack_from("<4s6H3L5H2L", payload, position)
        if disk_start != 0:
            raise ValueError("Multi-disk ZIP entries are forbidden")
        name_start = position + 46
        raw_name = payload[name_start : name_start + name_bytes]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        path = raw_name.decode(encoding)
        entries.append(
            ZipEntry(
                archive_index=archive_index,
                path=path,
                uncompressed_bytes=uncompressed_bytes,
                compressed_bytes=compressed_bytes,
                crc32=crc32,
                compression_method=compression_method,
                local_header_offset=local_header_offset,
            )
        )
        position = (
            name_start + name_bytes + extra_bytes + entry_comment_bytes
        )
    if range_start + position != central_offset + central_bytes:
        raise ValueError("ZIP central-directory length changed")
    return tuple(entries)


def build_contracts(
    entries: tuple[ZipEntry, ...],
) -> tuple[HanoiHustContract, ...]:
    """Validate all archive names and derive the fixed source/target partition."""
    if len(entries) != CENTRAL_DIRECTORY_ENTRIES:
        raise ValueError("Hanoi HUST archive entry count changed")
    contracts: list[HanoiHustContract] = []
    documentation: set[str] = set()
    for entry in entries:
        pure_path = PurePosixPath(entry.path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ValueError("Unsafe Hanoi HUST archive path")
        if entry.compression_method != 8:
            raise ValueError("Hanoi HUST compression method changed")
        filename = pure_path.name
        if filename in _DOCUMENTATION_FILES:
            documentation.add(filename)
            continue
        match = _MAT_NAME.fullmatch(filename)
        if match is None:
            raise ValueError(f"Unexpected Hanoi HUST archive member: {filename}")
        state, bearing_type_text, load_code = match.groups()
        bearing_type = int(bearing_type_text)
        bearing_id = f"{state}{bearing_type}"
        if bearing_id not in _EXPECTED_BEARINGS[state]:
            raise ValueError(f"Unexpected Hanoi HUST bearing: {bearing_id}")
        role: AccessRole = (
            "compound" if len(_COMPONENTS[state]) == 2 else "source"
        )
        contracts.append(
            HanoiHustContract(
                archive_index=entry.archive_index,
                path=entry.path,
                filename=filename,
                uncompressed_bytes=entry.uncompressed_bytes,
                compressed_bytes=entry.compressed_bytes,
                crc32=entry.crc32,
                compression_method=entry.compression_method,
                local_header_offset=entry.local_header_offset,
                state=state,
                components=_COMPONENTS[state],
                bearing_type=bearing_type,
                bearing_id=bearing_id,
                load_w=_LOAD_W[load_code],
                access_role=role,
            )
        )
    if documentation != _DOCUMENTATION_FILES:
        raise ValueError("Hanoi HUST documentation members changed")
    if len(contracts) != 99:
        raise ValueError("Hanoi HUST MAT file count changed")
    identity_counts = Counter(row.bearing_id for row in contracts)
    expected_ids = set().union(*_EXPECTED_BEARINGS.values())
    if set(identity_counts) != expected_ids or set(identity_counts.values()) != {
        3
    }:
        raise ValueError("Hanoi HUST bearing replication changed")
    for bearing_id in expected_ids:
        loads = {
            row.load_w for row in contracts if row.bearing_id == bearing_id
        }
        if loads != {0, 200, 400}:
            raise ValueError(f"Hanoi HUST load grid changed for {bearing_id}")
    state_bearings = {
        state: {row.bearing_id for row in contracts if row.state == state}
        for state in _EXPECTED_BEARINGS
    }
    if state_bearings != _EXPECTED_BEARINGS:
        raise ValueError("Hanoi HUST state-to-bearing map changed")
    if Counter(row.access_role for row in contracts) != {
        "source": 57,
        "compound": 42,
    }:
        raise ValueError("Hanoi HUST access partition changed")
    contracts.sort(key=lambda row: (row.bearing_id, row.load_w))
    return tuple(contracts)


def contract_to_json(contract: HanoiHustContract) -> dict[str, Any]:
    """Serialize one immutable contract without exposing numeric content."""
    return {
        "archive_index": contract.archive_index,
        "path": contract.path,
        "filename": contract.filename,
        "uncompressed_bytes": contract.uncompressed_bytes,
        "compressed_bytes": contract.compressed_bytes,
        "crc32": f"{contract.crc32:08x}",
        "compression_method": contract.compression_method,
        "local_header_offset": contract.local_header_offset,
        "state": contract.state,
        "components": list(contract.components),
        "bearing_type": contract.bearing_type,
        "bearing_id": contract.bearing_id,
        "load_w": contract.load_w,
        "access_role": contract.access_role,
    }
