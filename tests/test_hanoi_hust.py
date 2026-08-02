from __future__ import annotations

import struct
from collections import Counter

import pytest

from electrical_fm.hanoi_hust import (
    ARCHIVE_BYTES,
    CENTRAL_DIRECTORY_BYTES,
    CENTRAL_DIRECTORY_ENTRIES,
    CENTRAL_DIRECTORY_OFFSET,
    HanoiHustContract,
    ZipEntry,
    build_contracts,
    parse_central_directory_tail,
)


def _entry(index: int, filename: str) -> ZipEntry:
    prefix = (
        "HUST bearing a practical dataset for ball bearing fault diagnosis/"
        "HUST bearing dataset/"
    )
    return ZipEntry(
        archive_index=index,
        path=prefix + filename,
        uncompressed_bytes=1000 + index,
        compressed_bytes=900 + index,
        crc32=100_000 + index,
        compression_method=8,
        local_header_offset=index * 2000,
    )


def _valid_entries() -> tuple[ZipEntry, ...]:
    filenames = ["defects.png", "testbench.png"]
    state_types = {
        "N": range(4, 9),
        "I": range(4, 9),
        "O": range(4, 9),
        "B": range(5, 9),
        "IB": range(5, 9),
        "IO": range(4, 9),
        "OB": range(4, 9),
    }
    for state, bearing_types in state_types.items():
        for bearing_type in bearing_types:
            for load_code in ("00", "02", "04"):
                filenames.append(f"{state}{bearing_type}{load_code}.mat")
    rows = []
    for index, filename in enumerate(filenames):
        if filename.endswith(".png"):
            rows.append(
                ZipEntry(
                    archive_index=index,
                    path=(
                        "HUST bearing a practical dataset for ball bearing "
                        f"fault diagnosis/{filename}"
                    ),
                    uncompressed_bytes=1000 + index,
                    compressed_bytes=900 + index,
                    crc32=100_000 + index,
                    compression_method=8,
                    local_header_offset=index * 2000,
                )
            )
        else:
            rows.append(_entry(index, filename))
    return tuple(rows)


def _synthetic_tail() -> tuple[bytes, int]:
    central = bytearray()
    for index in range(CENTRAL_DIRECTORY_ENTRIES):
        target_length = 103 if index == 100 else 95
        stem = f"entry_{index:03d}_"
        name = (
            stem + ("x" * (target_length - len(stem) - 4)) + ".mat"
        ).encode("ascii")
        central.extend(
            struct.pack(
                "<4s6H3L5H2L",
                b"PK\x01\x02",
                20,
                20,
                0,
                8,
                0,
                0,
                index,
                100 + index,
                120 + index,
                len(name),
                0,
                0,
                0,
                0,
                0,
                index * 1000,
            )
        )
        central.extend(name)
    assert len(central) == CENTRAL_DIRECTORY_BYTES
    eocd = struct.pack(
        "<4s4H2LH",
        b"PK\x05\x06",
        0,
        0,
        CENTRAL_DIRECTORY_ENTRIES,
        CENTRAL_DIRECTORY_ENTRIES,
        CENTRAL_DIRECTORY_BYTES,
        CENTRAL_DIRECTORY_OFFSET,
        0,
    )
    range_start = CENTRAL_DIRECTORY_OFFSET
    payload = bytes(central) + eocd
    assert range_start + len(payload) == ARCHIVE_BYTES
    return payload, range_start


def test_build_contracts_recovers_strict_bearing_partition() -> None:
    contracts = build_contracts(_valid_entries())
    assert len(contracts) == 99
    assert Counter(row.access_role for row in contracts) == {
        "source": 57,
        "compound": 42,
    }
    assert len({row.bearing_id for row in contracts}) == 33
    assert {
        row.load_w for row in contracts if row.bearing_id == "IB5"
    } == {0, 200, 400}
    assert next(
        row for row in contracts if row.filename == "IO404.mat"
    ).components == ("inner", "outer")


def test_build_contracts_rejects_missing_and_unexpected_bearings() -> None:
    entries = list(_valid_entries())
    changed = entries[-1]
    entries[-1] = _entry(changed.archive_index, "B400.mat")
    with pytest.raises(ValueError, match="Unexpected Hanoi HUST bearing"):
        build_contracts(tuple(entries))


def test_parse_central_directory_tail_checks_archive_identity() -> None:
    payload, range_start = _synthetic_tail()
    entries = parse_central_directory_tail(
        payload,
        range_start=range_start,
    )
    assert len(entries) == CENTRAL_DIRECTORY_ENTRIES
    assert entries[0].path.startswith("entry_000_")
    assert entries[0].path.endswith(".mat")
    assert entries[-1].local_header_offset == 100_000
    with pytest.raises(ValueError, match="identity changed"):
        parse_central_directory_tail(
            payload,
            range_start=range_start,
            archive_bytes=ARCHIVE_BYTES + 1,
        )


def test_contract_is_immutable() -> None:
    contract = build_contracts(_valid_entries())[0]
    assert isinstance(contract, HanoiHustContract)
    with pytest.raises((AttributeError, TypeError)):
        contract.load_w = 999  # type: ignore[misc]
