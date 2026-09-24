# DeepSeekPathrep

This repository contains code and reproducible analysis files for the revised
PathRep-Bench leakage audit and internal survival-model sensitivity analysis.
The current manuscript is:

`deepseek_pathrep/manuscript_revised.md`

It evaluates TCGA-derived pathology report tasks:

- cancer type identification
- high-level AJCC stage prediction
- prognosis classification
- DeepSeek-assisted structured variable extraction for supervised prognosis modeling
- patient/report-text overlap and deduplicated Cox disease-specific survival modeling

The 173-case patient-ID-and-text-disjoint Cox test is a nested internal
sensitivity subset, **not external validation**. The original 887-case test
overlaps training; its scores are descriptive.

## Repository layout

- `deepseek_pathrep/manuscript_revised.md`: current corrected manuscript.
- `deepseek_pathrep/run_cox_deduplicated.py`: fit and export the patient- and
  report-text-deduplicated Cox analyses.
- `deepseek_pathrep/test_cox_deduplicated.py`: synthetic deduplication and
  paired-bootstrap safeguards.
- `deepseek_pathrep/package_verified_results.py`: score saved prediction files
  with fixed task labels and create a checksum-manifest audit archive.
- `deepseek_pathrep/manuscript_draft.md`: earlier manuscript, retained for
  history; do not use its superseded tables as the current results.

The BMC/Springer Nature LaTeX submission folder is maintained locally and is
not tracked in this repository.

## Reproduce the deduplicated Cox fit

Run from the repository root with Python 3 and the public input data. The
DeepSeek runner's `deepseek_pathrep/requirements.txt` does **not** include
dependencies for the Cox analysis. For the environment used to generate the
revised results, install:

```bash
python -m pip install "numpy==2.5.3" "pandas==2.3.3" "scipy==1.18.1" "scikit-learn==1.9.1" "lifelines==0.30.3"
```

Prepare the public [PathRep-Bench](https://github.com/rachitsaluja/PathRep-Bench)
train, validation, and test CSVs with the repository script:

```bash
python deepseek_pathrep/prepare_pathrep_data.py --output-dir data
```

This downloader uses the upstream `main` branch, which can change. For an
**exact** rerun, check the locally prepared CSVs against the input SHA-256
values below before fitting. If they differ, use the archived matching input
version rather than treating a new upstream revision as the same experiment.

| Prepared input | SHA-256 for the revised fit |
| --- | --- |
| `data/tcga_pathology_train.csv` | `52c05c8f66fca331e7c65f55a19c0b160bf12518dd1789cdbdda7c00a2062778` |
| `data/tcga_pathology_val.csv` | `394d9e3aae7a7a9779fe6376e374f0a5dcd57ae01fb56ffc9089ddc78166bbd1` |
| `data/tcga_pathology_test.csv` | `ffafc6f3c682051bf3b439bb9f30b54a754e0388bd7c7192cadfb1dc8104a09a` |

Save the raw TSV from [GerkeLab's TCGAclinical revision
8a95a11](https://github.com/GerkeLab/TCGAclinical/blob/8a95a11f763c0f6533b32d6076632db27169e693/data/cBioportal_data.tsv)
as `data/tcga_cdr_clinical.tsv`. Its SHA-256 must be
`e7f30c1ce2ac05c971c6a0c2dc068946b18315806c9f44584a51d343037c269a`.
This source supplies DSS event status; the fitted durations come from the
prepared benchmark CSVs' `DSS.time` in days, not a conversion of clinical
months. On Unix-like systems, check files with `sha256sum data/tcga_pathology_*.csv data/tcga_cdr_clinical.tsv`;
on Windows PowerShell use `Get-FileHash <path> -Algorithm SHA256`.

Fit all three Cox feature sets and export the disjointness manifest, per-case
risk scores, point estimates, and paired 2,000-resample percentile intervals:

```bash
python deepseek_pathrep/run_cox_deduplicated.py \
  --data-dir data \
  --cdr-clinical data/tcga_cdr_clinical.tsv \
  --output-dir outputs/deduplicated_cox \
  --bootstrap-replicates 2000
```

No DeepSeek or OpenAI API key or paid inference is needed for this Cox run. It
retains the first eligible patient record (train before validation), removes
additional identical normalized texts, and fits preprocessing on retained
train/validation rows only. Expected counts are 7,962 initially eligible
train-plus-validation rows, 703 repeated patient rows removed, 12 further
same-text rows removed, **7,247 fitted patients and reports**, and 173
patient-ID-and-text-disjoint test cases (48 observed events). The new
clinical-plus-text C-index on those 173 cases is approximately **0.693971**.
The 95% paired-bootstrap interval for its C-index difference from clinical-only
includes zero. The full 887-case set still contains training overlap.

Run the safeguards with:

```bash
python -m unittest discover -s deepseek_pathrep -p "test_cox_deduplicated.py" -v
```

## Rebuild the verification package

The five JSONL prediction runs and the older Cox/supervised exports live under
the **ignored** `outputs/` directory, not in this Git repository. If you have
the saved outputs and the independently prepared `VERIFICATION_REPORT.md`,
rebuild the minimal, checksum-manifest bundle without making API calls:

```bash
python deepseek_pathrep/package_verified_results.py \
  --repository . \
  --verification-report path/to/VERIFICATION_REPORT.md \
  --archive outputs/DeepSeekPathrep-verified-results-20260924.zip
```

The packager requires the five saved prediction JSONLs named in its
`PREDICTIONS` map, the prior `cox_test_risk_scores.csv`,
`cox_risk_score_metrics.json`, `supervised_sensitivity_predictions.csv`,
`supervised_sensitivity_metrics.json`, `leakage_audit.json`, and the new
`outputs/deduplicated_cox/` files. It checks corrected fixed-label macro-F1
against the verification report and strips raw model reasoning/content from
the packaged prediction rows. The private project archive contains the
verified outputs; it is not published by this repository or by the older
[Zenodo archive](https://doi.org/10.5281/zenodo.20754374).
Historical original Flash predictions used for some full-set paired tests
were checked by the external verifier but are absent from this bundle, so
those tests cannot be independently rerun from it. Neither the PathRep-Bench
data nor TCGA-CDR linkage supplies an external patient cohort.

For the older API experiments, see `deepseek_pathrep/README.md`. Do not commit
API keys, raw model reasoning, or de-identified report-containing outputs.
