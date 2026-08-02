# G3 external lockbox qualification

Date: 2026-07-30  
Status: metadata-only qualification; no external waveform has been loaded into the HANOI-HUST experiments.

## Red lines

1. No external data may enter model selection, threshold tuning, feature design, or manuscript wording before the candidate is frozen in this file.
2. A candidate is eligible only if its public source provides a stable identifier, acquisition description, labels/operating conditions, and a redistribution/access statement that can be archived.
3. The external set must remain a final locked test: one preprocessing recipe fixed from HANOI-HUST, no per-dataset normalization choice, no class redefinition after inspection, and no repeated selection on the external score.
4. All external results must be reported separately from the HANOI-HUST primary table and must include sample/unit counts, missingness, label mapping, and exact exclusion rules.
5. If the labels do not support the three HANOI-HUST component targets, the set may be used only for transfer/representation analysis, not pooled classification accuracy.

## Candidate registry

| Candidate | Stable identifier | Current status | Next qualification action |
|---|---|---|---|
| PHM Beijing 2024 multi-component transmission dataset | public challenge/arXiv record | metadata candidate; not frozen | archive landing page, version/date, label schema, and license |
| Politecnico di Torino spherical-roller-bearing dataset | Institutional record `11583/2997515`; concept DOI `10.5281/zenodo.14856937`; version DOI `10.5281/zenodo.14856938` | **metadata-qualified; numeric access pending** | freeze a file manifest if/when the public record exposes files |

Neither candidate is permitted in the manuscript or in model development until the qualification actions are completed and the registry is amended to `frozen` with a SHA-256 manifest.

## Metadata verification log

- The PHM Society public repository confirms that PHM-Beijing 2024 challenge data are publicly listed, but this page alone does not establish a frozen downloadable version or the physical-unit split required here.
- The official Politecnico di Torino IRIS record resolves the concept DOI and documents the bearing unit, operating conditions, and fault configurations. The public Zenodo API for version `14856938` currently exposes zero files, so no waveform or label has been loaded and numeric evaluation remains on hold.

Checked public metadata pages: `https://data.phmsociety.org/` and `https://doi.org/10.5281/zenodo.14856937` (the latter unresolved in this check).

## Pass criteria

- Public source and version captured in an immutable text record.
- Unit identity is independent of windows and can be held out as a physical unit.
- At least two classes or a defensible continuous transfer target are available.
- Sampling rate, channel semantics, and calibration are documented well enough to define a fixed preprocessing map.
- Any domain mismatch is declared; negative transfer is a valid result.

## Planned analysis after freeze

1. Freeze source URL/DOI metadata and archive a file manifest without opening waveforms.
2. Define the HANOI-HUST-trained preprocessing and classifier interface.
3. Open external waveforms only after the lockbox record is committed.
4. Run one-shot transfer evaluation and report failures, exclusions, and uncertainty.

Until these gates pass, external validation remains pending and the current paper claims remain HANOI-HUST-only.
