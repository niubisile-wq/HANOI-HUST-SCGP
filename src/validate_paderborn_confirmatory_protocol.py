"""Validate the frozen Paderborn confirmatory protocol against a third-party split."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "research" / "PADERBORN_CONFIRMATORY_PROTOCOL_20260727.json"
DEFAULT_SPLIT = (
    ROOT
    / "third_party"
    / "bearing-data-leakage"
    / "configs"
    / "dataset"
    / "split_function"
    / "paderborn.yaml"
)
DEFAULT_REPORT = (
    ROOT
    / "results"
    / "analysis"
    / "paderborn_confirmatory_protocol_validation_report.json"
)


@dataclass(frozen=True)
class DriftSummary:
    protocol_version: str
    status: str
    protocol_groups: dict[str, list[str]]
    split_groups: dict[str, list[str]]
    missing_from_split: dict[str, list[str]]
    extra_in_split: dict[str, list[str]]
    excluded_codes_seen_in_split: list[str]
    sampled_pairs_touching_excluded_codes: dict[str, list[list[str]]]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _as_sorted_list(values: Any) -> list[str]:
    return sorted({str(value) for value in values or []})


def _pair_has_excluded(pair: list[str], excluded: set[str]) -> bool:
    return any(code in excluded for code in pair)


def validate_protocol(protocol: dict[str, Any], split: dict[str, Any]) -> DriftSummary:
    protocol_groups = {
        key: _as_sorted_list(values)
        for key, values in protocol["selected_groups"].items()
    }

    split_groups = {
        "healthy": _as_sorted_list(split.get("healthy_bearing_ids")),
        "outer_ring_damage": _as_sorted_list(split.get("outer_bearing_ids")),
        "inner_ring_damage": _as_sorted_list(split.get("inner_bearing_ids")),
    }

    missing_from_split = {
        key: sorted(set(protocol_groups[key]) - set(split_groups[key]))
        for key in protocol_groups
    }
    extra_in_split = {
        key: sorted(set(split_groups[key]) - set(protocol_groups[key]))
        for key in protocol_groups
    }

    excluded_codes = set(protocol["excluded_codes"])
    sampler = split.get("sampler", {})
    sampled_pairs_touching_excluded_codes: dict[str, list[list[str]]] = {}
    for list_name in ("list1", "list2", "list3"):
        pairs = sampler.get(list_name, []) or []
        violating_pairs = [list(pair) for pair in pairs if _pair_has_excluded(pair, excluded_codes)]
        sampled_pairs_touching_excluded_codes[list_name] = violating_pairs

    excluded_codes_seen_in_split = sorted(
        excluded_codes
        & set(chain.from_iterable(split_groups.values()))
    )

    status = "protocol_mismatch_detected"
    if not any(missing_from_split.values()) and not any(extra_in_split.values()):
        status = "protocol_aligned"

    return DriftSummary(
        protocol_version=str(protocol["protocol_version"]),
        status=status,
        protocol_groups=protocol_groups,
        split_groups=split_groups,
        missing_from_split=missing_from_split,
        extra_in_split=extra_in_split,
        excluded_codes_seen_in_split=excluded_codes_seen_in_split,
        sampled_pairs_touching_excluded_codes=sampled_pairs_touching_excluded_codes,
    )


def summarize(summary: DriftSummary) -> dict[str, Any]:
    return {
        "protocol_version": summary.protocol_version,
        "status": summary.status,
        "protocol_groups": summary.protocol_groups,
        "split_groups": summary.split_groups,
        "missing_from_split": summary.missing_from_split,
        "extra_in_split": summary.extra_in_split,
        "excluded_codes_seen_in_split": summary.excluded_codes_seen_in_split,
        "sampled_pairs_touching_excluded_codes": summary.sampled_pairs_touching_excluded_codes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Paderborn confirmatory protocol."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    protocol = load_json(args.protocol)
    split = load_yaml(args.split)
    summary = validate_protocol(protocol, split)
    report = summarize(summary)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
