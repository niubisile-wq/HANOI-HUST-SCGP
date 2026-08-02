"""Print the frozen HANOI HUST source registry summary as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hanoi_hust_registry import REGISTRY_PATH, registry_summary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "audits" / "hanoi_hust_source_registry_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    summary = registry_summary(REGISTRY_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
