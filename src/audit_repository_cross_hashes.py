"""Audit path/SHA-256 pairs against the canonical public-repository bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "tmp", "release"}
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".tex",
    ".yaml",
    ".yml",
    ".csv",
    ".cff",
    ".txt",
    ".ini",
    ".cls",
    ".sty",
    ".bst",
}


def _canonical_bytes(path: Path) -> bytes:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return path.read_bytes()
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(_canonical_bytes(path)).hexdigest()


def build_audit() -> dict[str, Any]:
    checked: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []

    for metadata_path in ROOT.rglob("*.json"):
        relative_metadata = metadata_path.relative_to(ROOT)
        if SKIP_PARTS.intersection(relative_metadata.parts):
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))

        def visit(node: Any, key_path: str = "$") -> None:
            if isinstance(node, dict):
                relative_target = node.get("path")
                expected = node.get("sha256")
                if (
                    isinstance(relative_target, str)
                    and isinstance(expected, str)
                    and len(expected) == 64
                ):
                    target = ROOT / relative_target
                    if target.is_file():
                        actual = _sha256(target)
                        row = {
                            "metadata": relative_metadata.as_posix(),
                            "key_path": key_path,
                            "target": relative_target,
                            "expected": expected.lower(),
                            "actual": actual,
                        }
                        checked.append(row)
                        if actual != expected.lower():
                            mismatches.append(row)
                for key, value in node.items():
                    visit(value, f"{key_path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    visit(value, f"{key_path}[{index}]")

        visit(payload)

    return {
        "stage": "public_repository_cross_hash_audit",
        "status": "passed" if not mismatches else "failed",
        "checked_reference_count": len(checked),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    result = build_audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
