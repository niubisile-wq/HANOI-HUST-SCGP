from __future__ import annotations

import audit_repository_cross_hashes as cross_hashes


def test_public_repository_cross_hashes_are_consistent() -> None:
    audit = cross_hashes.build_audit()
    assert audit["status"] == "passed"
    assert audit["checked_reference_count"] >= 120
    assert audit["mismatch_count"] == 0
    assert audit["mismatches"] == []
