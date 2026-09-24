# Historical Revision Plan (Superseded)

**This is an earlier planning note, not the current evidence record.** Its 0.7210 clean-subset Cox C-index was not reproduced by the final fitted model; the corrected clinical-plus-text value is **0.6818** on 174 barcode-disjoint cases. The sensitivity subset is not external validation, and not all original model prediction files are available for paired analysis. Refer to `deepseek_pathrep/manuscript_revised.md` and the project verification archive before using any numbers below.

## Situation

**Manuscript**: "DeepSeek for Pathology Report Understanding: A Benchmark Study of Cancer Type Extraction, AJCC Staging, and Prognosis Prediction"

**Rejection**: Scientific Reports rejected after both reviewers recommended acceptance. Editor's concerns:
1. Insufficient original scientific contribution beyond existing literature
2. Similar TCGA/PathRep-Bench analyses already reported multiple times
3. Single dataset — does not advance generalizability understanding

**Target venue**: BMC Bioinformatics or JAMIA Open

---

## New Experiments Completed

### 1. Data Leakage Assessment (CRITICAL NEW FINDING)

**Discovery**: The PathRep-Bench dataset has severe patient-level data leakage.

| Metric | Value |
|--------|-------|
| Test barcodes overlapping with training | 770 / 952 (80.9%) |
| Val barcodes overlapping with training | 765 / 953 (80.3%) |
| Overlapping reports with identical text | 770 / 770 (100%) |
| Clean test set (no overlap) | 182 rows (was 952) |
| Clean val set (no overlap) | 188 rows (was 953) |

**Impact on metrics** (TF-IDF LogReg, full TF-IDF features):

| Task | Original (leaked) | Clean (no leak) | Change |
|------|-------------------|-----------------|--------|
| Cancer type accuracy | 0.9910 | 0.9713 | -1.97 pp |
| Cancer type macro F1 | 0.9899 | 0.9412 | -4.87 pp |
| Prognosis accuracy | 0.8647 | 0.7644 | -10.03 pp |
| Prognosis macro F1 | 0.8617 | 0.7586 | -10.31 pp |

This finding affects not only this study but all prior PathRep-Bench evaluations.

### 2. Cox Proportional Hazards Survival Model

Replaces the binary threshold prognosis approach with proper time-to-event modeling using TCGA-CDR censoring data.

**Data**: Merged PathRep-Bench reports with TCGA-CDR clinical data (GerkeLab PanCanAtlas) to obtain DSS event/censoring status.

| Split | Rows | Events | Censored |
|-------|------|--------|----------|
| Train+Val | 7,962 | 1,658 | 6,304 |
| Test (original) | 887 | 183 | 704 |
| Test (clean) | 174 | 49 | 125 |

**Results** (TF-IDF SVD-100 features):

| Model | C-index (original) | C-index (clean) |
|-------|--------------------|--------------------|
| Clinical only | 0.7648 [0.7296, 0.7975] | — |
| Clinical + TF-IDF | 0.7875 [0.7567, 0.8193] | 0.7210 |
| TF-IDF only | 0.7780 [0.7480, 0.8109] | — |

Bootstrap 95% CIs computed with 500 resamples.

### 3. Patient Barcode Overlap Check

| Comparison | Overlap |
|------------|---------|
| Train-Val | 765 barcodes |
| Train-Test | 770 barcodes |
| Val-Test | 0 barcodes |

All overlapping barcodes have identical report text — this is row-level duplication, not different samples from the same patient.

---

## Experiments In Progress

### 4. Full-Test DeepSeek Prognosis Prompting (952 reports)

Running DeepSeek V4 Flash with reasoning mode on all 952 test reports (previously only 100).
- Status: In progress (background process)
- Addresses: "subset-based evaluation" criticism

### 5. Cross-Model Comparison (GPT-4o-mini)

OpenAI credential available. Script written (`run_cross_model.py`).
- Planned: GPT-4o-mini on cancer type (952 reports) and AJCC staging (594 reports)
- Addresses: "original contribution" and "generalizability" concerns

---

## Revised Manuscript Contributions

1. **Identification of patient-level data leakage in PathRep-Bench** (80.9% test-train overlap with identical text) — a critical methodological finding that affects all prior benchmark evaluations
2. **Proper time-to-event survival modeling** (Cox PH) replacing the binary DSS-threshold approach, with censoring from TCGA-CDR
3. **Clean evaluation on deduplicated splits** showing true generalization performance is substantially lower than leaked estimates
4. **Full-test DeepSeek prognosis prompting** (all 952 reports, not just 100)
5. **Cross-model comparison** with GPT-4o-mini (if completed)
6. **Cost-performance analysis** of DeepSeek Flash vs Pro (from original manuscript)
7. **Hybrid architecture recommendation** (from original manuscript)

---

## How Each Editor Concern Is Addressed

### "Insufficient original scientific contribution"
- Data leakage discovery is a genuine new finding
- Cox survival model is a methodological advance
- Clean-split evaluation provides the first unbiased estimate
- Cross-model comparison adds breadth

### "Similar TCGA analyses already reported"
- This study now identifies a flaw in the benchmark itself
- Time-to-event modeling has not been done in prior PathRep-Bench work
- Clean-split results question the validity of prior reported metrics

### "Single dataset / generalizability"
- Internal robustness testing: leave-one-cancer-type-out, clean-split evaluation
- Cross-model comparison tests model generalizability
- Data leakage finding suggests the benchmark needs revision before further use
- Acknowledge limitation honestly; frame as "generalization stress testing"

---

## Files Created

| File | Purpose |
|------|---------|
| `deepseek_pathrep/run_cox_survival.py` | Cox PH survival model with multiple feature sets |
| `deepseek_pathrep/run_cox_robustness.py` | Bootstrap CIs, paired tests, LOCTO, barcode checks |
| `deepseek_pathrep/run_cross_model.py` | Cross-model comparison (OpenAI, Anthropic, DeepSeek) |
| `data/tcga_cdr_clinical.tsv` | TCGA-CDR clinical data with censoring info |
| `data/tcga_pathology_test_clean.csv` | Deduplicated test set (182 rows) |
| `data/tcga_pathology_val_clean.csv` | Deduplicated validation set (188 rows) |
| `outputs/cox_survival_results.json` | Cox model results |
| `outputs/cox_robustness_results.json` | Robustness validation results |

---

## Next Steps

1. Wait for full-test DeepSeek prognosis run to complete
2. Run GPT-4o-mini on cancer type and AJCC staging (cross-model comparison)
3. Run DeepSeek on clean test set to show leakage impact on LLM metrics
4. Draft revised manuscript incorporating all new findings
5. Prepare cover letter for BMC Bioinformatics / JAMIA Open
