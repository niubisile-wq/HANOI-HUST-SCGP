# Submission QA Report

Date: 2026-08-02

Target journal: *Machines*

## Outcome

The manuscript and Supplementary Information compile successfully and the paper's primary argument is internally consistent:

1. deployment-oriented diagnosis requires evidence from independent physical units;
2. SCGP specifies the partition, nested-selection, aggregation, and label-access contract;
3. the matched G2 audit quantifies how those protocol choices change the reported score;
4. the HANOI held-out and the separately reported three-code and 12-code Paderborn checks test bounded confirmation and cross-rig behavior; and
5. the Discussion and Conclusion interpret all scores as protocol-bound evidence rather than deployment guarantees.

The matched G2 table and figure are now the visual center of the Results. Legacy source-ranking and protocol-sensitivity graphics are retained as Supplementary Figures S1--S2.

Repeated evidential caveats were consolidated during the final language pass. The Abstract, Results, and Conclusion now lead with the supported findings, while access-history facts remain in Methods and the full scope conditions are concentrated in the Limitations subsection.

## Evidence and reproducibility checks

- Primary G2 reconstruction audit: passed.
- Six registered G2 cells present: passed.
- 100 splits per G2 cell: passed.
- Prediction-array SHA-256 hashes: passed.
- Paired split manifest: passed.
- Main table reconstruction from retained JSON/predictions: passed.
- Pre-access-frozen 12-code Paderborn extension audit: passed.
- Extension registry, frozen runner, HANOI source, and feature-cache SHA-256 checks: passed.
- Twelve unique codes, 960 records, and 80 records per code: passed.
- Unit probabilities, thresholded predictions, and headline metrics independently rebuilt: passed.
- All 12 official archive hashes matched the frozen registry during the local audit; the raw archives remain outside the repository and release packages.
- Canonical repository cross-hash audit: 128 existing path/SHA-256 references checked; mismatches: 0.
- Automated tests: 29 passed.
- JSON files parsed: 83; failures: 0.
- `CITATION.cff` parsed as CFF 1.2.0.
- Citations: 46 unique citation keys and 46 bibliography entries; no missing or unused entries; bibliography follows first-appearance order.
- Placeholder scan: no TODO, TBD, FIXME, XXX, placeholder, or insertion markers in the manuscript, Supplementary Information, or README.
- Authored-code/manuscript whitespace check: passed; publisher-supplied template files and intentional Markdown hard line breaks were excluded.

## Compilation and visual QA

- `manuscript.pdf`: 19 A4 pages; successful `latexmk` build.
- `supplementary_information.pdf`: 6 US-Letter pages; successful `latexmk` build.
- No undefined references, overfull boxes, `Float too large` warnings, LaTeX errors, or fatal errors.
- All pages were rasterized and visually inspected. Key checks covered the title/abstract page, access ladder, primary G2 table and Figure 2, the expanded boundary Table 4 and Figure 3, G3 table, declarations, bibliography ending, the new extension table, and all Supplementary Information pages.
- Remaining underfull-box and `fancyhdr` head-height messages are template/layout warnings; the rendered pages show no clipping or header collision.

## Evidence boundaries retained in the text

- Bearing-wise partitioning is acknowledged as established prior practice; novelty is limited to the integrated, machine-auditable access contract and crossed audit.
- HANOI compound evidence is described as a historical held-out confirmation under a documented local freeze, not as a publicly time-stamped prospective lockbox.
- The one-shot three-code Paderborn pilot and the later pre-access-frozen 12-code extension are reported separately; the historical 15-code registry was not scored as that protocol.
- Repeated splits reuse 19 bearings and are not treated as independent experiments.
- The 12-code extension is disjoint from the pilot, but it still contains only four codes per state from one rig and no Paderborn ball-fault state.
- The HANOI source (19 bearings), HANOI held-out boundary (14 bearings), Paderborn pilot (3 codes), and Paderborn extension (12 new codes) do not support population-level deployment guarantees.

## Final PDF hashes

```text
61332c342465b9b5bd0ec051986c54bd9b6a1356ff2f84f1cfe32156815ab3c1  manuscript.pdf
37b37ef4a411c86d5560d80da357b6ca7be6fce1e5e4f8916169dddd6226b95a  supplementary_information.pdf
```

Public reproducibility release 2.1.0 is maintained at <https://github.com/niubisile-wq/HANOI-HUST-SCGP> and archived at <https://doi.org/10.5281/zenodo.21754415>. The project concept DOI is <https://doi.org/10.5281/zenodo.21695842>.
