from __future__ import annotations

import json
from pathlib import Path

import analyze_hanoi_hust_source_ablation as ablation


def test_ablation_report_uses_frozen_baseline_table(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(ablation, "OUTPUT", tmp_path / "report.json")
    monkeypatch.setattr(ablation, "OUTPUT_MD", tmp_path / "report.md")
    report = ablation.build_report()
    assert report["status"] == "completed"
    assert report["dominant_view"] in {"all", "envelope_log_power", "fixed_log_power", "statistics"}
    assert report["dominant_family"] == "logistic_l2"
    assert report["key_findings"]["best_view_mean_auroc"] >= report["key_findings"]["view_gap_vs_fixed"]


def test_ablation_markdown_is_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ablation, "OUTPUT", tmp_path / "out.json")
    monkeypatch.setattr(ablation, "OUTPUT_MD", tmp_path / "out.md")
    report = ablation.build_report()
    assert (tmp_path / "out.json").is_file()
    assert (tmp_path / "out.md").is_file()
    data = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert report["dominant_family"] == data["dominant_family"]
