"""Build a machine-readable multi-seed registry summary for HANOI HUST."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REGISTRY = ROOT / "results" / "audits" / "hanoi_hust_source_registry.json"
SOURCE_BASELINES = (
    ROOT / "results" / "development" / "hanoi_hust_source_baselines.json"
)
MULTISEED = ROOT / "results" / "development" / "hanoi_hust_source_multiseed.json"
OUTPUT_JSON = ROOT / "results" / "audits" / "hanoi_hust_seed_registry_summary.json"
OUTPUT_MD = ROOT / "results" / "audits" / "hanoi_hust_seed_registry_summary.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(staging, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def build_summary() -> dict[str, Any]:
    source_registry = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    baselines = json.loads(SOURCE_BASELINES.read_text(encoding="utf-8"))
    multiseed = json.loads(MULTISEED.read_text(encoding="utf-8"))

    runs = multiseed["runs"]
    seed_list = [int(row["seed"]) for row in runs]
    repeat_count = len(runs)
    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "node": platform.node(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    metrics = multiseed["aggregates"]
    summary = {
        "stage": "hanoi_hust_seed_registry_summary",
        "schema_version": 1,
        "status": "completed",
        "project": "HANOI_HUST",
        "dataset_record": source_registry["dataset"],
        "seed_list": seed_list,
        "repeat_count": repeat_count,
        "hardware": hardware,
        "commit_hash": _git("rev-parse", "HEAD"),
        "config_hash": _sha256(SOURCE_BASELINES),
        "split_hash": _sha256(SOURCE_REGISTRY),
        "metric_bundle": metrics,
        "artifact_path": MULTISEED.relative_to(ROOT).as_posix(),
        "selected_family": multiseed["champion"]["selected_family"],
        "selected_representation": multiseed["champion"]["selected_representation"],
        "selected_hyperparameter": multiseed["champion"]["selected_hyperparameter"],
        "allowed_current_statement": (
            "The HANOI HUST main method and main results are frozen, and a "
            "dedicated multi-seed registry is now summarized from completed "
            "seed repeats."
        ),
        "source_registry_sha256": _sha256(SOURCE_REGISTRY),
        "source_baselines_sha256": _sha256(SOURCE_BASELINES),
        "multiseed_sha256": _sha256(MULTISEED),
        "provenance": {
            "builder": "src/build_hanoi_hust_seed_registry_summary.py",
            "git_commit": _git("rev-parse", "HEAD"),
        },
    }
    _atomic_json(OUTPUT_JSON, summary)

    markdown = [
        "# HANOI HUST Seed Registry Summary",
        "",
        f"- project: `HANOI_HUST`",
        f"- dataset_record: `{summary['dataset_record']['record']}`",
        f"- dataset_version: `{summary['dataset_record']['version']}`",
        f"- dataset_doi: `{summary['dataset_record']['doi']}`",
        f"- dataset_title: `{summary['dataset_record']['title']}`",
        f"- dataset_license: `{summary['dataset_record']['license']}`",
        f"- sample_rate_hz: `{summary['dataset_record']['sample_rate_hz']}`",
        f"- record_count: `{summary['dataset_record']['record_count']}`",
        f"- physical_bearings: `{summary['dataset_record']['physical_bearings']}`",
        f"- source_bearings: `{summary['dataset_record']['source_bearings']}`",
        f"- compound_bearings: `{summary['dataset_record']['compound_bearings']}`",
        f"- seed_list: `{seed_list}`",
        f"- repeat_count: `{repeat_count}`",
        f"- hardware: `{hardware['node']} / {hardware['machine']} / {hardware['logical_cpus']} CPUs`",
        f"- commit_hash: `{summary['commit_hash']}`",
        f"- config_hash: `{summary['config_hash']}`",
        f"- split_hash: `{summary['split_hash']}`",
        f"- artifact_path: `{summary['artifact_path']}`",
        "",
        "## Metric Bundle",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric_name, stats in metrics.items():
        markdown.append(
            f"| {metric_name} | {stats['mean']:.15f} | {stats['std']:.15f} | {stats['min']:.15f} | {stats['max']:.15f} |"
        )
    markdown.extend(
        [
            "",
            "## Selected Model",
            "",
            f"- family: `{summary['selected_family']}`",
            f"- representation: `{summary['selected_representation']}`",
            f"- hyperparameter: `{summary['selected_hyperparameter']}`",
            "",
            "## Current Statement",
            "",
            summary["allowed_current_statement"],
            "",
        ]
    )
    _atomic_text(OUTPUT_MD, "\n".join(markdown))
    return summary


def main() -> None:
    print(json.dumps(build_summary(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
