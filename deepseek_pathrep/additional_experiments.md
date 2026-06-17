# Additional Experiments

## 1. DeepSeek V4 Pro Benchmark

Purpose:

Test whether `deepseek-v4-pro` improves accuracy enough to justify the higher cost compared with `deepseek-v4-flash`.

Status:

Completed for cancer type and AJCC stage. In both cases, `deepseek-v4-pro` did not improve over `deepseek-v4-flash`.

Cancer type:

```powershell
python .\run_deepseek_pathology.py `
  --input ..\data\tcga_pathology_test.csv `
  --task cancer_type `
  --model deepseek-v4-pro `
  --output ..\outputs\deepseek_v4_pro_cancer_type_full.jsonl
```

AJCC stage:

```powershell
python .\run_deepseek_pathology.py `
  --input ..\data\tcga_pathology_test.csv `
  --task ajcc_stage `
  --model deepseek-v4-pro `
  --output ..\outputs\deepseek_v4_pro_ajcc_stage_full_max4096.jsonl `
  --skip-unlabeled `
  --max-tokens 4096
```

Analyze:

```powershell
python .\analyze_predictions.py `
  --predictions ..\outputs\deepseek_v4_pro_cancer_type_full.jsonl `
  --data ..\data\tcga_pathology_test.csv

python .\analyze_predictions.py `
  --predictions ..\outputs\deepseek_v4_pro_ajcc_stage_full_max4096.jsonl `
  --data ..\data\tcga_pathology_test.csv
```

## 2. Cost Analysis

Purpose:

Report practical deployment costs: cost per report, cost per 1000 reports, and cost per correct prediction.

Current implementation:

```powershell
python .\cost_analysis.py `
  ..\outputs\deepseek_cancer_type_full.jsonl.metrics.json `
  ..\outputs\deepseek_ajcc_stage_full_v2_max4096.jsonl.metrics.json `
  ..\outputs\deepseek_prognosis_100_max4096.jsonl.metrics.json `
  ..\outputs\deepseek_prognosis_100_fewshot8_max4096.jsonl.metrics.json `
  --markdown .\cost_analysis.md
```

Pricing note:

The script uses DeepSeek's official per-1M-token pricing table and conservatively treats input tokens as cache misses when cache-hit/cache-miss token counts are not available.

## 3. Clinical Error Analysis

Purpose:

Create discussion-ready tables for clinically plausible error modes.

```powershell
python .\make_error_tables.py
```

This generates:

```text
.\error_analysis_tables.md
```

Current tables:

- Rectum vs colon confusion
- Kidney RCC subtype confusion
- Stage III vs Stage IV confusion

## 4. DeepSeek-Assisted ML Prognosis

Purpose:

Test whether DeepSeek-extracted structured pathology variables improve the local prognosis classifier beyond the current TF-IDF + structured-field baseline.

Status:

Complete. Although the TF-IDF text-only model achieved the highest prognosis point estimate, adding DeepSeek-extracted variables did not produce a statistically significant improvement.

Original baseline to beat:

- TF-IDF + original structured fields logistic regression
- Accuracy: 0.8529
- Macro F1: 0.8501

Strongest final baseline:

- TF-IDF text-only logistic regression
- Accuracy: 0.8571
- Macro F1: 0.8543

Smoke test variable extraction:

```powershell
python .\extract_deepseek_variables.py `
  --input ..\data\tcga_pathology_test.csv `
  --output ..\outputs\deepseek_variables_test_100.jsonl `
  --limit 100 `
  --resume
```

Smoke test result:

- Rows: 100
- Model errors: 0
- Total tokens: 153,896
- Estimated cost: about $0.0235, or $0.2349 per 1000 reports
- Schema quality: all 100 rows returned the expected variable keys

Projected full extraction for train + validation + test:

- Rows: 9,523
- Estimated total tokens: about 14.66M
- Estimated cost: about $2.24 with `deepseek-v4-flash`

This estimate is based on the first 100 test reports and should be treated as approximate.

Full extraction needed for the complete hybrid experiment:

```powershell
python .\extract_deepseek_variables.py `
  --input ..\data\tcga_pathology_train.csv `
  --output ..\outputs\deepseek_variables_train_full.jsonl `
  --resume

python .\extract_deepseek_variables.py `
  --input ..\data\tcga_pathology_val.csv `
  --output ..\outputs\deepseek_variables_val_full.jsonl `
  --resume

python .\extract_deepseek_variables.py `
  --input ..\data\tcga_pathology_test.csv `
  --output ..\outputs\deepseek_variables_test_full.jsonl `
  --resume
```

Train prognosis model with extracted DeepSeek variables:

```powershell
python .\run_prognosis_ml_baseline.py `
  --train-on-val `
  --deepseek-features-train ..\outputs\deepseek_variables_train_full.jsonl `
  --deepseek-features-val ..\outputs\deepseek_variables_val_full.jsonl `
  --deepseek-features-test ..\outputs\deepseek_variables_test_full.jsonl `
  --output ..\outputs\ml_prognosis_deepseek_assisted_test.jsonl
```

Run ablations:

```powershell
# DeepSeek structured variables + original structured clinical fields, no report TF-IDF
python .\run_prognosis_ml_baseline.py `
  --train-on-val `
  --no-text `
  --deepseek-features-train ..\outputs\deepseek_variables_train_full.jsonl `
  --deepseek-features-val ..\outputs\deepseek_variables_val_full.jsonl `
  --deepseek-features-test ..\outputs\deepseek_variables_test_full.jsonl `
  --output ..\outputs\ml_prognosis_deepseek_structured_only_test.jsonl

# DeepSeek variables only, no raw text and no original structured fields
python .\run_prognosis_ml_baseline.py `
  --train-on-val `
  --no-text `
  --no-base-structured `
  --deepseek-features-train ..\outputs\deepseek_variables_train_full.jsonl `
  --deepseek-features-val ..\outputs\deepseek_variables_val_full.jsonl `
  --deepseek-features-test ..\outputs\deepseek_variables_test_full.jsonl `
  --output ..\outputs\ml_prognosis_deepseek_only_test.jsonl
```

Compare results:

```powershell
python .\compare_prognosis_models.py `
  ..\outputs\ml_prognosis_tfidf_logreg_test.jsonl.metrics.json `
  ..\outputs\ml_prognosis_deepseek_assisted_test.jsonl.metrics.json `
  ..\outputs\ml_prognosis_deepseek_structured_only_test.jsonl.metrics.json `
  ..\outputs\ml_prognosis_deepseek_only_test.jsonl.metrics.json `
  --markdown .\prognosis_model_comparison.md
```

Important:

DeepSeek variables must be available for the training rows and the test rows. Test-only DeepSeek features are useful only as a plumbing smoke test; they cannot improve the trained model because the classifier has not seen those feature names during training.

Decision rule:

If DeepSeek-assisted ML improves beyond the strongest `0.8543` macro F1 text-only baseline, it becomes a novel contribution. If not, the result still supports a valuable conclusion: LLM extraction is not automatically useful for prognosis unless the extracted variables add information beyond text.

Observed result:

- TF-IDF text-only: 0.8543 macro F1
- TF-IDF + original structured fields: 0.8501 macro F1
- TF-IDF + original structured fields + DeepSeek variables: 0.8394 macro F1
- DeepSeek variables only: 0.6407 macro F1

Conclusion:

Although the TF-IDF text-only model achieved the highest prognosis point estimate, adding DeepSeek-extracted variables did not produce a statistically significant improvement. These results suggest that the raw pathology report text already captures much of the prognostic information available in this benchmark dataset under the evaluated modeling framework. This strengthens the manuscript's claim that prognosis should be treated as a supervised text/outcome modeling problem rather than a pure LLM-prompting or LLM-extraction task.

Suggested interpretation language:

- Improvement over 0.8543 macro F1: DeepSeek variables add clinically useful structure beyond raw text.
- No improvement: raw report text plus supervised learning already captures much of the available prognostic signal, and LLM extraction does not automatically improve prognosis modeling.
