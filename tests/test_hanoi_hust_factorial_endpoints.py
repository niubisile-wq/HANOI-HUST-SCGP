from __future__ import annotations

import audit_hanoi_hust_factorial_endpoints as factorial_endpoints


def test_factorial_endpoints_rebuild_from_record_predictions() -> None:
    audit = factorial_endpoints.build_audit()
    assert audit["status"] == "passed"
    assert audit["cell_count"] == 3
    assert audit["split_count_per_cell"] == 100
    assert audit["failures"] == []
