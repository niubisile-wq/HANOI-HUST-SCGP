"""Finalize the HANOI HUST Applied Sciences release after a DOI is issued.

The default mode is a dry run. Use --apply only after checking the DOI and
version URL. The script does not publish files or contact an external service.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "research/HANOI_HUST_20260727/HANOI_HUST_DSP_doublecolumn_draft_20260728.tex"
METADATA = ROOT / "research/HANOI_HUST_AS_APPLIED_SCIENCES_SUBMISSION_METADATA_20260730.yaml"

OLD_DATA_SENTENCE = (
    "A public repository/archive release containing the same versioned artifacts will be deposited "
    "before acceptance and its persistent identifier will be added to the final version."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_doi(doi: str) -> bool:
    return bool(re.fullmatch(r"(?:https://doi\.org/)?10\.\d{4,9}/[-._;()/:A-Z0-9]+", doi, re.I))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--doi", required=True, help="DOI or https://doi.org/... URL")
    p.add_argument("--version-url", default="", help="Optional public archive version URL")
    p.add_argument("--apply", action="store_true", help="Actually edit files; default is dry-run")
    args = p.parse_args()

    if not validate_doi(args.doi):
        raise SystemExit("Invalid DOI format")
    doi_url = args.doi if args.doi.startswith("http") else f"https://doi.org/{args.doi}"
    replacement = (
        f"The versioned reproducibility package is publicly archived at {doi_url}. "
        "Raw benchmark waveforms are not redistributed; derived artifacts and source code are available "
        "from the archive and corresponding author."
    )

    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
    metadata_text = METADATA.read_text(encoding="utf-8")
    if OLD_DATA_SENTENCE not in manuscript_text:
        raise SystemExit("Expected Data Availability placeholder was not found; aborting")
    if "doi: null" not in metadata_text:
        raise SystemExit("Expected metadata DOI placeholder was not found; aborting")

    new_metadata = metadata_text.replace("doi: null", f'doi: "{doi_url}"', 1)
    if args.version_url:
        new_metadata = new_metadata.replace(
            'doi_status: "pending author-controlled public deposit"',
            f'doi_status: "published; version_url: {args.version_url}"',
            1,
        )
    new_manuscript = manuscript_text.replace(OLD_DATA_SENTENCE, replacement, 1)

    print(f"DOI: {doi_url}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Manuscript changed: {new_manuscript != manuscript_text}")
    print(f"Metadata changed: {new_metadata != metadata_text}")
    if not args.apply:
        print("No files changed. Re-run with --apply after checking the DOI.")
        return 0

    backup_dir = ROOT / "research/HANOI_HUST_public_release_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in (MANUSCRIPT, METADATA):
        shutil.copy2(path, backup_dir / f"{path.name}.before_doi")
    MANUSCRIPT.write_text(new_manuscript, encoding="utf-8")
    METADATA.write_text(new_metadata, encoding="utf-8")
    print(f"Updated: {MANUSCRIPT}")
    print(f"Updated: {METADATA}")
    print(f"Manuscript SHA-256: {sha256(MANUSCRIPT)}")
    print(f"Metadata SHA-256: {sha256(METADATA)}")
    print("Next step: compile the manuscript twice, inspect the PDF, and regenerate the release ZIP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
