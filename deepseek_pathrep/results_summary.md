# DeepSeek Pathology Report Benchmark Results

## Manuscript Positioning

Recommended title:

DeepSeek for Pathology Report Understanding: Strengths and Limitations Across Extraction, Staging, and Prognosis Tasks

Core message:

DeepSeek achieves excellent performance for pathology information extraction and strong performance for AJCC staging, but dedicated machine-learning models substantially outperform LLM prompting for prognosis prediction.

Recommended architecture:

- DeepSeek -> cancer-type information extraction
- DeepSeek -> AJCC staging support
- Classical ML -> prognosis prediction

Additional reviewer-strengthening experiments:

- Compare `deepseek-v4-flash` and `deepseek-v4-pro` on cancer type and AJCC staging.
- Report cost per report, cost per 1000 reports, and cost per correct prediction.
- Include clinical error tables for rectum/colon, kidney RCC subtypes, and Stage III/IV confusion.
- Test DeepSeek-assisted ML prognosis using extracted structured pathology variables.

Current status:

The Flash-vs-Pro comparison, cost analysis, clinical error tables, and DeepSeek-assisted ML prognosis experiment are complete.

Decision rule:

- If DeepSeek-assisted ML exceeds the strongest prognosis baseline of 0.8543 macro F1, it becomes a novel hybrid-model contribution.
- If it does not exceed 0.8543 macro F1, it supports the conclusion that LLM extraction does not automatically improve prognosis prediction beyond supervised learning on report text.

Final decision:

DeepSeek-assisted ML did not exceed the strongest non-LLM prognosis baseline. The best prognosis point estimate was TF-IDF text-only logistic regression with 0.8543 macro F1. Adding DeepSeek-extracted structured variables reduced macro F1 to 0.8394, but the paired difference was not statistically significant. These results suggest that the raw pathology report text already captures much of the prognostic information available in this benchmark dataset under the evaluated modeling framework.

DeepSeek variable extraction smoke test:

- File: `..\outputs\deepseek_variables_test_100.jsonl`
- Rows: 100
- Model errors: 0
- Total tokens: 153,896
- Estimated cost: about $0.0235
- Projected train + validation + test extraction cost: about $2.24 for 9,523 reports

The smoke test confirms that structured variable extraction is parseable and resumable. The next required step is full train, validation, and test extraction so the prognosis classifier can learn from DeepSeek-derived variables.

Full DeepSeek variable extraction:

- Train rows: 7,618, model errors: 7
- Validation rows: 953, model errors: 1
- Test rows: 952, model errors: 1
- Total extracted rows: 9,523
- Total extraction cost: about $1.77

Current generated artifacts:

- `cost_analysis.md`
- `error_analysis_tables.md`
- `additional_experiments.md`

## Cost Analysis

Pricing source:

https://api-docs.deepseek.com/quick_start/pricing

Assumption:

When cache-hit/cache-miss token counts are unavailable, all prompt tokens are conservatively priced as cache misses.

| Run | Reports | Correct | Tokens | Estimated cost | Cost/report | Cost/1000 reports | Cost/correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cancer type, DeepSeek V4 Flash | 952 | 933 | 1,292,828 | $0.182611 | $0.000192 | $0.1918 | $0.000196 |
| Cancer type, DeepSeek V4 Pro | 952 | 930 | 1,294,297 | $0.463766 | $0.000487 | $0.4871 | $0.000499 |
| AJCC stage, DeepSeek V4 Flash | 594 | 485 | 945,862 | $0.161103 | $0.000271 | $0.2712 | $0.000332 |
| AJCC stage, DeepSeek V4 Pro | 594 | 481 | 1,170,725 | $0.695980 | $0.001172 | $1.1717 | $0.001447 |
| Prognosis zero-shot, DeepSeek V4 Flash | 100 | 53 | 151,124 | $0.024516 | $0.000245 | $0.2452 | $0.000463 |
| Prognosis 8-shot, DeepSeek V4 Flash | 100 | 57 | 386,105 | $0.058671 | $0.000587 | $0.5867 | $0.001029 |

## AJCC Stage Identification

Run:

```powershell
python .\run_deepseek_pathology.py `
  --input ..\data\tcga_pathology_test.csv `
  --task ajcc_stage `
  --output ..\outputs\deepseek_ajcc_stage_full_v2_max4096.jsonl `
  --skip-unlabeled `
  --max-tokens 4096
```

Result:

- Model: `deepseek-v4-flash`
- Prompt version: `stage_explicit_substage_v2`
- Input rows: 952
- Processed labeled rows: 594
- Skipped unlabeled rows: 358
- Model output errors: 0
- Correct: 485 / 594
- Accuracy: 0.8165
- Macro F1: 0.7841
- Token usage: 740,988 prompt + 204,874 completion = 945,862 total

Confusion summary:

| Gold | Predicted | Count |
| --- | --- | ---: |
| Stage I | Stage I | 161 |
| Stage I | Stage II | 18 |
| Stage I | Stage III | 9 |
| Stage I | Stage IV | 1 |
| Stage II | Stage I | 9 |
| Stage II | Stage II | 170 |
| Stage II | Stage III | 19 |
| Stage II | Stage IV | 5 |
| Stage III | Stage I | 4 |
| Stage III | Stage II | 13 |
| Stage III | Stage III | 120 |
| Stage III | Stage IV | 5 |
| Stage IV | Stage II | 2 |
| Stage IV | Stage III | 24 |
| Stage IV | Stage IV | 34 |

Most common error groups:

- Thyroid carcinoma: 19
- Bladder urothelial carcinoma: 13
- Stomach adenocarcinoma: 13
- Head and neck squamous cell carcinoma: 11
- Lung squamous cell carcinoma: 8

Interpretation:

DeepSeek produced a strong zero-shot/thinking-mode AJCC baseline with no JSON or empty-output failures. The main weakness is under-calling Stage IV as Stage III, plus adjacent-stage confusion among Stage I, II, and III.

### AJCC Flash vs Pro Comparison

DeepSeek V4 Pro run:

```powershell
python .\run_deepseek_pathology.py `
  --input ..\data\tcga_pathology_test.csv `
  --task ajcc_stage `
  --model deepseek-v4-pro `
  --output ..\outputs\deepseek_v4_pro_ajcc_stage_full_max4096.jsonl `
  --skip-unlabeled `
  --max-tokens 4096
```

| Model | Rows | Correct | Accuracy | Macro F1 | Model errors | Total tokens | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4 Flash | 594 | 485 | 0.8165 | 0.7841 | 0 | 945,862 | $0.161103 |
| DeepSeek V4 Pro | 594 | 481 | 0.8098 | 0.6288 | 14 | 1,170,725 | $0.695980 |

Interpretation:

For AJCC staging, `deepseek-v4-pro` did not improve the primary full-run result over `deepseek-v4-flash`. It produced 14 empty final responses in thinking mode, used more output tokens, and cost approximately 4.32x more for this run. On the 580 rows where Pro returned valid JSON, it achieved 481 / 580 accuracy (0.8293) and 0.7962 macro F1, so the model may have slightly stronger valid-answer performance. However, the empty-response reliability issue and higher cost still support using Flash as the better cost-performance model unless Pro is rerun with a larger completion budget.

## Cancer Type Identification

Run:

```powershell
python .\run_deepseek_pathology.py `
  --input ..\data\tcga_pathology_test.csv `
  --task cancer_type `
  --output ..\outputs\deepseek_cancer_type_full.jsonl
```

Result:

- Model: `deepseek-v4-flash`
- Input rows: 952
- Processed rows: 952
- Model output errors: 0
- Correct: 933 / 952
- Accuracy: 0.9800
- Macro F1: 0.9778
- Token usage: 1,281,292 prompt + 11,536 completion = 1,292,828 total

Main true error groups after case-insensitive label normalization:

- Rectum adenocarcinoma: 6
- Lung squamous cell carcinoma: 3
- Stomach adenocarcinoma: 3
- Kidney renal papillary cell carcinoma: 3
- Cervical squamous cell carcinoma and endocervical adenocarcinoma: 1
- Ovarian serous cystadenocarcinoma: 1
- Kidney renal clear cell carcinoma: 1
- Brain lower grade glioma: 1

Interpretation:

DeepSeek matches the strongest cancer-type identification results reported in the paper. Most remaining errors are clinically plausible neighboring-label confusions, especially rectum versus colon, stomach versus esophageal, and renal cancer subtypes.

### Cancer Type Flash vs Pro Comparison

DeepSeek V4 Pro run:

```powershell
python .\run_deepseek_pathology.py `
  --input ..\data\tcga_pathology_test.csv `
  --task cancer_type `
  --model deepseek-v4-pro `
  --output ..\outputs\deepseek_v4_pro_cancer_type_full.jsonl
```

| Model | Rows | Correct | Accuracy | Macro F1 | Model errors | Total tokens | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4 Flash | 952 | 933 | 0.9800 | 0.9778 | 0 | 1,292,828 | $0.182611 |
| DeepSeek V4 Pro | 952 | 930 | 0.9769 | 0.9685 | 0 | 1,294,297 | $0.463766 |

Interpretation:

For cancer type identification, `deepseek-v4-pro` did not improve over `deepseek-v4-flash`. It was slightly less accurate, had lower macro F1, and cost approximately 2.54x more despite cache-hit tokens. This further supports Flash as the better cost-performance model for pathology information extraction.

## Prognosis Assessment

Initial zero-shot run:

```powershell
python .\run_deepseek_pathology.py `
  --input ..\data\tcga_pathology_test.csv `
  --task prognosis `
  --output ..\outputs\deepseek_prognosis_100_max4096.jsonl `
  --limit 100 `
  --max-tokens 4096
```

Result:

- Model: `deepseek-v4-flash`
- Input rows: 100
- Model output errors: 0
- Correct: 53 / 100
- Accuracy: 0.5300
- Macro F1: 0.5083
- Token usage: 127,135 prompt + 23,989 completion = 151,124 total

Confusion summary:

| Gold | Predicted | Count |
| --- | --- | ---: |
| good | good | 16 |
| good | poor | 26 |
| poor | good | 21 |
| poor | poor | 37 |

Interpretation:

The pure DeepSeek prognosis prompt is close to chance, matching the paper's observation that prognosis is much harder than extraction or staging. The next planned run should use few-shot examples from the training split.

Few-shot run with 8 training examples:

```powershell
python .\run_deepseek_pathology.py `
  --input ..\data\tcga_pathology_test.csv `
  --task prognosis `
  --output ..\outputs\deepseek_prognosis_100_fewshot8_max4096.jsonl `
  --limit 100 `
  --examples ..\data\prognosis_examples_8.jsonl `
  --examples-per-call 8 `
  --max-tokens 4096
```

Result:

- Model: `deepseek-v4-flash`
- Input rows: 100
- Model output errors: 0
- Correct: 57 / 100
- Accuracy: 0.5700
- Macro F1: 0.5647
- Token usage: 353,135 prompt + 32,970 completion = 386,105 total

Few-shot confusion summary:

| Gold | Predicted | Count |
| --- | --- | ---: |
| good | good | 23 |
| good | poor | 19 |
| poor | good | 24 |
| poor | poor | 34 |

Interpretation:

Few-shot prompting improved the zero-shot baseline from 0.5300 to 0.5700 accuracy and from 0.5083 to 0.5647 macro F1. The improvement is real but too small to make prognosis reliable. A hybrid or classical survival/classification baseline is the better next experiment for this task.

## Prognosis Alternative: Local TF-IDF Logistic Regression

Run:

```powershell
python .\run_prognosis_ml_baseline.py `
  --train-on-val `
  --output ..\outputs\ml_prognosis_tfidf_logreg_test.jsonl
```

Result:

- Model: TF-IDF report text + structured clinical fields + logistic regression
- Training rows: 8,566
- Test rows: 952
- Accuracy: 0.8529
- Macro F1: 0.8501

Confusion summary:

| Gold | Predicted | Count |
| --- | --- | ---: |
| poor | poor | 471 |
| poor | good | 101 |
| good | poor | 39 |
| good | good | 341 |

Class-level performance:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| poor | 0.9235 | 0.8234 | 0.8706 |
| good | 0.7715 | 0.8974 | 0.8297 |

Interpretation:

For prognosis, the local classifier is much stronger than DeepSeek prompting. This supports a hybrid article direction: use DeepSeek for cancer-type extraction and AJCC staging, but use a dedicated survival/classification model for prognosis.

## Final Prognosis Ablation

After full DeepSeek variable extraction, prognosis models were compared against the strongest non-LLM baseline.

| Method | Accuracy | Macro F1 | Delta vs best |
| --- | ---: | ---: | ---: |
| TF-IDF text only | 0.8571 | 0.8543 | 0.0000 |
| TF-IDF text + original structured fields | 0.8529 | 0.8501 | -0.0042 |
| TF-IDF text + original structured fields + DeepSeek variables | 0.8435 | 0.8394 | -0.0149 |
| Original structured fields + DeepSeek variables | 0.6607 | 0.6512 | -0.2031 |
| DeepSeek variables only | 0.6492 | 0.6407 | -0.2136 |
| Original structured fields only | 0.5578 | 0.5540 | -0.3003 |

Best model confusion summary:

| Gold | Predicted | Count |
| --- | --- | ---: |
| poor | poor | 474 |
| poor | good | 98 |
| good | poor | 38 |
| good | good | 342 |

Final interpretation:

Although the TF-IDF text-only model achieved the highest prognosis point estimate, adding DeepSeek-extracted variables did not produce a statistically significant improvement. These results suggest that the raw pathology report text already captures much of the prognostic information available in this benchmark dataset under the evaluated modeling framework. This supports the manuscript's role-separation argument: use DeepSeek for extraction and AJCC staging support, but use supervised text-based modeling for prognosis.
