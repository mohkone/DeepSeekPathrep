# Data Leakage and Time-to-Event Evaluation in Pathology Report Benchmarks: A DeepSeek and GPT-4o-mini Study on PathRep-Bench

**Short title:** Pathology report benchmark audit with DeepSeek and GPT-4o-mini

**Authors:** Mohamed Kone¹, Gaoussou Haidara¹, Chao Hou¹, Shulin Wang¹*

¹School of Computer Science and Engineering, Hunan University, Changsha, China.

*Corresponding author: Shulin Wang. E-mail: books@hnu.edu.cn

---

## Abstract

### Background

Pathology report benchmarks derived from The Cancer Genome Atlas (TCGA) are increasingly used to evaluate large language models (LLMs) for clinical information extraction, staging, and prognosis prediction. However, the reliability of these benchmarks depends on proper data splitting and clinically meaningful outcome endpoints. We assessed PathRep-Bench for cross-split report duplication, evaluated DeepSeek and GPT-4o-mini on the original and deduplicated splits, and introduced proper time-to-event survival modeling to replace binary threshold-based prognosis evaluation.

### Methods

We audited the PathRep-Bench splits using patient barcodes and normalized report-text hashes. Saved DeepSeek V4 Flash and GPT-4o-mini predictions were scored over fixed task label sets, retaining invalid responses as errors. Patient-paired comparisons used exact McNemar tests and 5,000 bootstrap resamples. Cox models used benchmark disease-specific survival (DSS) time and TCGA Clinical Data Resource event status. After detecting 703 repeated patient rows and 12 additional same-text rows in the eligible train-plus-validation set, we refitted three Cox feature sets on 7,247 unique patients and texts, fitting preprocessing on these records alone. The primary survival sensitivity assessment used 173 test patients with neither barcode nor normalized text overlapping the original training or validation splits.

### Results

Of 952 test reports, 770 (80.9%) shared a training barcode and identical text; one further report had the same normalized text under a different barcode. Original-run Flash versus GPT-4o-mini cancer-type accuracy was 0.9800 versus 0.9727 (paired McNemar p = 0.0923), while stage accuracy was 0.8165 versus 0.5993 (p = 6.795 × 10⁻²²). On the 181 patient-and-text-disjoint classification cases, new Flash and restricted GPT cancer-type accuracy both equaled 0.9890; for 115 stage-labeled cases, they were 0.8174 and 0.5913. Full-test Flash prognosis prompting achieved accuracy 0.4716 and fixed-two-label macro-F1 0.4792, with 31 invalid outputs. The refitted, deduplicated Cox clinical-plus-text model scored C-index **0.6940** (patient-bootstrap 95% interval 0.6271–0.7629) on 173 disjoint cases; clinical-only scored 0.6551 (0.5822–0.7282). The paired C-index difference interval (−0.0041 to 0.0833) spans zero.

### Conclusions

The released splits contain substantial duplication and may contaminate supervised evaluation. Deduplicated Cox fitting addresses repeated training records, but the small nested disjoint test subset and unmatched case mix do not identify a causal leakage effect. The stage comparison supports a Flash advantage under the saved prompting configurations; the cancer-type comparison does not establish one. This single TCGA-derived cohort does **not** constitute external validation.

**Keywords:** pathology reports, data leakage, benchmark evaluation, large language models, DeepSeek, GPT-4o-mini, cancer staging, survival analysis, Cox proportional hazards, TCGA

---

## 1 Introduction

Pathology reports are among the most information-rich documents in oncology. They summarize tumor histology, tissue site, grade, resection status, nodal involvement, metastatic findings, biomarker context, and stage group. These reports are central to cancer diagnosis and treatment planning, but their free-text format limits large-scale computational reuse for cohort construction, registry enrichment, and outcomes research.

Recent progress in large language models (LLMs) has made pathology report understanding more feasible. Unlike earlier rule-based systems or narrow named-entity recognition pipelines, LLMs can interpret varied report formats, follow label-constrained instructions, and produce structured outputs from heterogeneous text. PathRep-Bench established a benchmark for evaluating this capability across cancer type identification, AJCC stage identification, and prognosis assessment from pathology reports [1]. The benchmark uses data derived from The Cancer Genome Atlas (TCGA) pathology reports and clinical outcome resources [2, 3, 4].

A fundamental assumption of any benchmark is that the training, validation, and test splits are disjoint. When the same reports appear in both training and test sets, supervised models can memorize specific examples, inflating apparent performance. This concern is particularly relevant for TCGA-derived data, where multiple data files may be merged using patient identifiers, and split procedures may operate at the row level rather than the patient level. However, the extent of cross-split duplication in PathRep-Bench has not been independently audited.

A second concern is the choice of prognosis endpoint. PathRep-Bench uses a binary label derived from cancer-type mean disease-specific survival (DSS) time: patients surviving beyond the mean are labeled "good" and those below are labeled "poor." This thresholding discards time-to-event and censoring information, creates ambiguity for cases near the threshold, and does not correspond to a clinically validated prognostic category. Proper survival analysis methods, such as Cox proportional hazards modeling, can use the actual survival time and censoring status to provide a more clinically meaningful evaluation.

This study makes three contributions. First, we audit released split identities and report text, identifying a nested patient-and-text-disjoint sensitivity subset. Second, we correct task-label macro-F1 scoring and compare saved Flash and GPT-4o-mini predictions with paired tests, while extending prognosis prompting to all 952 test reports. Third, we repeat DSS time-to-event Cox fitting after patient and report deduplication and train-only preprocessing. These analyses examine internal robustness, not generalization to an external cohort.

## 2 Methods

### 2.1 Dataset

We used the public PathRep-Bench data derived from TCGA pathology reports and TCGA clinical outcome resources [1, 2, 3, 4]. The prepared dataset contained 9,523 pathology reports with fixed train, validation, and test splits: 7,618 training reports, 953 validation reports, and 952 test reports. Each row included pathology report text, cancer type, AJCC stage fields where available, disease-specific survival time, and derived prognosis labels.

For cancer type identification, all 952 test reports had labels across 32 TCGA cancer types. For AJCC staging, 594 of the 952 test reports had usable high-level stage labels (Stage I–IV). For prognosis, all 952 test reports had derived good or poor labels based on whether disease-specific survival exceeded the cancer-type mean DSS threshold.

### 2.2 Split leakage audit

We audited the PathRep-Bench train, validation, and test splits for cross-split report duplication using two methods:

1. **Patient barcode overlap**: We compared the set of TCGA patient barcodes (`bcr_patient_barcode`) across all splits. Barcode overlap indicates that the same patient appears in multiple splits.

2. **Report text hash verification**: We computed SHA-256 hashes of normalized report text (lowercased, whitespace-collapsed) for all reports in each split. Text hash overlap confirms that the same report content appears in multiple splits, not just different samples from the same patient.

For overlapping barcodes between train and test, we also checked exact text equality and examined normalized-text matches under different barcodes. We checked for within-split duplicate barcodes. The audit script and outputs are provided in the verification archive.

### 2.3 Deduplicated test set

We excluded test reports whose patient barcodes appeared in training, leaving a 182-report barcode-disjoint sensitivity subset. Removing the one remaining cross-barcode normalized-text match left 181 patient-and-text-disjoint reports, 115 with AJCC stage labels. The corresponding DSS-eligible subsets included 174 barcode-disjoint and 173 barcode-and-text-disjoint cases. Selection depended on identifiers and text, not outcomes or prediction correctness; these are nested internal subsets, not an independently sampled cohort. The val–test split had no matching IDs or report hashes in the audit.

### 2.4 Models

We evaluated the following models:

- **DeepSeek V4 Flash**: A cost-effective reasoning model with JSON output support [5, 6].
- **DeepSeek V4 Pro**: A larger DeepSeek model with higher per-token pricing [5, 6].
- **GPT-4o-mini**: OpenAI's cost-efficient multimodal model, evaluated via the OpenAI chat completions API.
- **TF-IDF logistic regression**: A supervised sensitivity analysis on the DSS-observed subset used lowercasing, English stop-word removal, unigrams and bigrams, sublinear term frequency, a minimum document frequency of 2, and at most 50,000 features. Models trained on 7,962 train-plus-validation cases used balanced class weights, C = 2.0, maximum 3,000 iterations, and random seed 2026. The binary prognosis classifier used `liblinear`; the 32-class cancer-type classifier used `lbfgs` [7]. These sensitivity runs differ from the original manuscript's 952-case prognosis baseline.

### 2.5 Prompting framework

The prompts required JSON-only answers. The cancer type prompt supplied the 32 allowed labels, and the stage prompt supplied Stage I–IV; prognosis prompting supplied the cancer-type mean DSS threshold. DeepSeek cancer type calls used non-reasoning mode, while DeepSeek stage and prognosis calls used reasoning mode. GPT-4o-mini used the shared task prompts through a separate API without an equivalent DeepSeek reasoning-mode setting. Therefore, the cross-model contrast compares the tested configurations, not isolated model architectures. The original 100-report few-shot prognosis run supplied eight training examples; the new 952-report run was zero-shot.

Model output failures, empty responses, and out-of-vocabulary values were counted as incorrect. For the present analyses, macro-F1 averages over the prespecified 32 cancer types, four stages, or two prognosis labels (`fixed_task_labels_v1`): invalid outputs contribute false negatives rather than an extra class, and labels absent in a subset have zero F1. This corrects earlier runner-generated union-of-observed-and-predicted-label F1 values without changing accuracy or normalization. The two `unknown` GPT cancer predictions are invalid even though the API error field is zero; 29-present-class subset F1, when shown, is explicitly secondary.

### 2.6 Cox proportional hazards survival modeling

We obtained disease-specific survival event and censoring status from the TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR) [8] via the GerkeLab clinical data file [9]. The event indicator was set to 1 for patients who died with tumor and 0 for patients coded as alive or dead tumor-free. Disease-specific survival time was taken from the PathRep-Bench data (in days). We retained only cases with a positive DSS time and known event status.

We fitted Cox proportional hazards models using the following feature sets:

1. **Clinical only**: Age at diagnosis, gender, cancer type (one-hot encoded).
2. **Clinical + TF-IDF**: Clinical features plus 100 SVD-reduced TF-IDF components from pathology report text.
3. **TF-IDF only**: 100 SVD-reduced TF-IDF components only.

For historical comparison, the archived Cox models fitted 7,962 eligible train-plus-validation rows: 7,079 training and 883 validation, but only 7,259 distinct patient IDs. In the new fit we retain the first eligible row per patient (training precedes validation), then the first row per lowercased, whitespace-collapsed SHA-256 report hash. This removes 703 repeated patient rows and 12 further same-text rows under different barcodes, leaving 7,247 distinct fitted patients and texts. Age median imputation, cancer-type categories, TF-IDF vocabulary, 100-component truncated SVD, and scaling were fitted only on those retained records. Cox penalizer was fixed at 0.1; no validation-set tuning was performed after combining the splits. Models produced partial-hazard risk scores on all 887 test records, but the **primary sensitivity endpoint** is the 173-case patient-and-text-disjoint subset, chosen against the original eligible train-plus-validation inputs. Concordance uses negative risk; 2,000 patient-paired percentile bootstrap resamples (seed 20260924) quantify conditional test-sample uncertainty and the clinical-plus-text minus clinical-only difference. The original 887-case scores remain descriptive because 713 patients overlap training. No integrated Brier score or successful leave-one-cancer-type-out fit is claimed.

### 2.7 Statistical analysis

We report fixed-label macro-F1 and accuracy for classification, with invalid predictions in the denominator. The supplied verification report documents patient-ID and gold-label checks against historical original Flash prediction files held locally by the verifier, enabling four exploratory Flash–GPT comparisons. Exact two-sided McNemar tests assess paired correctness; 5,000 patient-paired percentile bootstrap resamples (seed 20260924) give unadjusted macro-F1 difference intervals. Holm-adjusted p values cover the four McNemar tests. These comparisons are conditional on saved runs and do not estimate API-run variability. The historical original Flash row files and verifier's machine-readable paired results are **not** in this distribution, so those particular tests cannot be independently reproduced from this package; we distinguish them from comparisons reconstructible using its five new JSONLs. The prior archived paired Cox bootstrap and all 20 leave-one-cancer-type-out fits failed; only the separately run deduplicated Cox bootstrap is reported as successful.

## 3 Results

### 3.1 Cross-split report duplication audit

The audit revealed substantial cross-split report duplication in PathRep-Bench (Table 1).

**Table 1.** Cross-split report duplication audit. Text-hash matches use lowercased, whitespace-collapsed report text; barcode overlaps and text-hash overlaps are distinct counts.

| Comparison | Shared barcodes | Shared normalized-text hashes | Exact text among train–test shared barcodes |
|-----------|----------------:|------------------------------:|-------------------------------------------:|
| Train–Val | 765 / 953 (80.3%) | 765 | Not assessed here |
| Train–Test | 770 / 952 (80.9%) | 771 | 770 / 770 |
| Val–Test | 0 | 0 | Not applicable |
| Within-split duplicate barcodes | 0 in each split | Not assessed | Not applicable |

All 770 test reports sharing training barcodes had byte-identical report text. The 771st train–test normalized-text hash match occurred under two different barcodes. Thus, the 182-report barcode-disjoint subset still contains one normalized-text match to training and is not completely report-text-disjoint.

### 3.2 Barcode-disjoint sensitivity analysis

The historical TF-IDF sensitivity runs trained on 7,962 DSS-observed train-plus-validation rows (before the Cox-specific deduplication) scored 0.9910 and 0.9713 cancer-type accuracy on all 887 and the nested 174 barcode-disjoint cases, respectively; prognosis accuracy was 0.8647 and 0.7644. These are retained as historical descriptive results, **not** outcomes of the newly deduplicated Cox fit. They differ from the original manuscript's separate 952-case supervised prognosis baseline.

Flash cancer type accuracy was 0.9800 in the original 952-case test and 0.9835 in the separately prompted 182-case barcode-disjoint run. Corresponding stage accuracies were 0.8165/594 and 0.8103/116; seven of the new stage outputs were invalid. The new subset run is **not** a restriction of the original run: repeated prompts may change responses. Table 2 marks run identity and the 173-case text-disjoint survival endpoint.

**Table 2.** Original test versus nested internal sensitivity subsets. Cox numbers here are from the new deduplicated fit, not the original Cox fit; the older TF-IDF logistic regression and original/new Flash prediction runs are separate experiments. The original cohort overlaps training.

| Task and method | Original test, n; score | Barcode-disjoint subset, n; score | Metric |
|-----------------|------------------------:|----------------------------------:|--------|
| Cancer type, TF-IDF logistic regression | 887; 0.9910 | 174; 0.9713 | Accuracy |
| Prognosis, TF-IDF logistic regression | 887; 0.8647 | 174; 0.7644 | Accuracy |
| Prognosis, same classifier | 887; 0.8617 | 174; 0.7586 | Macro F1 |
| Cancer type, DeepSeek Flash zero-shot | 952; 0.9800 | 182; 0.9835 | Accuracy |
| AJCC stage, DeepSeek Flash zero-shot | 594; 0.8165 | 116; 0.8103 | Accuracy |
| Cox clinical plus text, **new deduplicated fit** | 887; 0.7926 | 174; 0.6899 (173 ID/text-disjoint; **0.6940**) | C-index |

### 3.3 Cross-model comparison

On the original test, Flash cancer type accuracy/macro-F1 were 0.9800/0.9778 versus 0.9727/**0.9687** for GPT-4o-mini. The paired cancer accuracy difference was +0.00735, exact McNemar p = 0.0923 (Holm-adjusted 0.1846); the paired F1 difference was +0.00912 (95% percentile interval −0.00066 to 0.02224). AJCC accuracy/macro-F1 were 0.8165/0.7841 versus 0.5993/0.5904; paired accuracy difference was +0.21717, p = 6.795 × 10⁻²² (Holm-adjusted 2.718 × 10⁻²¹), and F1 difference was +0.19371 (0.14844 to 0.23900). These results support an AJCC advantage, not statistically demonstrated cancer-type superiority. The two APIs used different prompting/reasoning configurations, so they do not isolate architecture.

**Table 3.** Fixed-task-label scoring of saved original runs. Flash original rows and the paired verifier's machine-readable output remain outside the distributed bundle; its seven row-level checks and paired estimates are documented in the attached verification report. GPT rows are archived here. “Invalid” is distinct from model/API errors. Historical DeepSeek Pro results are omitted because fixed-label rescoring from row-level records is unavailable.

| Task | Model | n | Accuracy | Fixed-label macro-F1 | Invalid predictions |
|------|-------|---:|---------:|---------------------:|--------------------:|
| Cancer type | DeepSeek V4 Flash, original | 952 | 0.980042 | 0.977832 | 0 |
| Cancer type | GPT-4o-mini, original | 952 | 0.972689 | 0.968712 | 2 |
| AJCC stage | DeepSeek V4 Flash, original | 594 | 0.816498 | 0.784110 | 0 |
| AJCC stage | GPT-4o-mini, original | 594 | 0.599327 | 0.590398 | 0 |

Table 3's Flash original values and original paired tests are attributed to the user-supplied verification report, whose local source files are not included in the package. No revised cost claim or Pro-versus-GPT inference is made.

**Table 3a.** Matched ID/text-disjoint subset. New Flash predictions were generated separately for the barcode-disjoint subset, then restricted to text-disjoint records; GPT predictions are restrictions of its original full-test run. The 32-class cancer F1 includes three absent categories as zeros; on 29 present classes, Flash/GPT F1 is 0.994158/0.983431.

| Task | Flash new run, n; accuracy; fixed-label F1 | GPT original restricted, n; accuracy; fixed-label F1 | Paired McNemar p (Holm p) | Paired F1 difference, 95% CI |
|------|-------------------------------------------|-----------------------------------------------------|----------------------------|------------------------------|
| Cancer type | 181; 0.988950; 0.900956 | 181; 0.988950; 0.891234 | 1.000000 (1.000000) | +0.009722 (−0.001838, 0.016667) |
| AJCC stage | 115; 0.817391; 0.830834 | 115; 0.591304; 0.597037 | 2.434 × 10⁻⁵ (7.303 × 10⁻⁵) | +0.233797 (0.144359, 0.331858) |

Restricted original Flash AJCC predictions, a *different* run from new Flash, scored 0.843478 accuracy and 0.835582 F1 on these same 115 cases. No change across runs is ascribed to leakage alone.

### 3.4 Full-test DeepSeek prognosis prompting

DeepSeek V4 Flash prognosis prompting on all 952 reports returned 449 correct and 31 invalid outputs: accuracy 0.471639 and **fixed-two-label macro-F1 0.479198**. The previously quoted runner F1 of 0.319466 incorrectly averaged an empty-output value as a third class. On the 181 ID/text-disjoint reports (restriction of the original run), accuracy/F1 were 0.513812/0.517397, with three invalid outputs. The original 100-case zero-/eight-shot results remain historical and were not rerun here.

**Table 4.** DeepSeek prognosis prompting. Fixed-label F1 is used for the two verified 952/181-case rows. The older 100-case values and the 952-case supervised baseline are historical aggregates with different cohorts/scoring provenance; they are not used for quantitative model comparison.

| Condition | n | Accuracy | Macro F1 | Errors |
|-----------|---|----------|----------|--------|
| Zero-shot (100-report subset) | 100 | 0.5300 | 0.5083 | 0 |
| 8-shot (100-report subset) | 100 | 0.5700 | 0.5647 | 0 |
| Zero-shot (full test) | 952 | 0.4716 | **0.4792** | 31 |
| Zero-shot (original run restricted to ID/text-disjoint) | 181 | 0.5138 | 0.5174 | 3 |
| TF-IDF LogReg (original supervised baseline) | 952 | 0.8571 | 0.8543 | — |

### 3.5 Cox proportional hazards survival model

Table 5 contrasts the archived duplicate-training fit (7,962 rows) with the newly fitted patient-and-report-deduplicated Cox models (7,247 rows). The newly fitted clinical-plus-text C-index was 0.6940 on the locked 173-case ID/text-disjoint subset versus 0.6551 for clinical-only and 0.7181 for text-only. Its paired difference from clinical-only was +0.0389, bootstrap 95% interval −0.0041 to 0.0833, so a text-feature benefit is not established. The full 887-case test remains overlapping and is not the primary endpoint.

**Table 5.** Cox DSS concordance. The prior and new fits must not be pooled; each cell is evaluated on the stated test subset. The new 173-case intervals use 2,000 patient-paired percentile resamples conditional on fixed fitted models. No external validation or leave-one-cancer-type-out performance is implied.

| Fit / model | Original 887 (overlap) | ID-disjoint 174 | ID/text-disjoint 173 (95% CI for new fit) |
|-------------|-----------------------:|----------------:|---------------------------------------------:|
| Prior fit (7,962 rows), clinical only | 0.764758 | 0.655051 | 0.655925 |
| Prior fit (7,962 rows), clinical + text | 0.787460 | 0.681818 | 0.688773 |
| Prior fit (7,962 rows), text only | 0.777967 | 0.693333 | 0.705821 |
| **Deduplicated fit (7,247 rows), clinical only** | 0.764896 | 0.654040 | 0.655094 (0.582189–0.728202) |
| **Deduplicated fit (7,247 rows), clinical + text** | 0.792612 | 0.689899 | **0.693971 (0.627109–0.762943)** |
| **Deduplicated fit (7,247 rows), text only** | 0.787038 | 0.709899 | 0.718087 (0.651700–0.787822) |

The original eligible training-plus-validation set contained 1,658 events and 6,304 censored records. The 887/174/173 test subsets had 183/49/48 events, respectively. Archived risk scores reproduce the earlier clinical-plus-text C-index **0.6818** for 174 ID-disjoint cases and **0.6888** for 173 ID/text-disjoint cases, not the previously quoted 0.7210. The newly deduplicated fit is a different model, not a correction made by simply changing a test mask.

### 3.6 Error analysis

For cancer type identification, the main DeepSeek errors were clinically plausible neighboring-label confusions: rectum adenocarcinoma predicted as colon adenocarcinoma (5 errors), kidney renal papillary cell carcinoma predicted as clear cell carcinoma (3 errors), and stomach versus esophageal confusion (3 errors).

Earlier aggregate AJCC results reported Stage IV recall of 0.567 (34/60) for DeepSeek V4 Flash, with 24 Stage IV reports predicted as Stage III. In the newly archived 116-case barcode-disjoint stage run, Stage IV recall among the 18 Stage IV reports was 13/18 (0.722); seven overall stage outputs were empty. These different samples do not support attributing the recall difference to duplication.

GPT-4o-mini showed more diffuse stage confusion, with substantial errors across all adjacent stage pairs. Stage I→II confusion (63 errors) and Stage III→IV confusion (32 errors) were the dominant error modes.

## 4 Discussion

This audit found substantial cross-split duplication in the released PathRep-Bench splits: 80.9% of test reports shared a training barcode and exact text. This creates contamination risk for split-trained models, although nested subset contrasts do not measure a causal leakage effect. Paired saved-run analyses support Flash's advantage over GPT-4o-mini for AJCC under the tested configurations, not for cancer type. The new Cox fit resolves within-training patient/text repetition and uses DSS time and event status instead of the binary surrogate; its small disjoint-cohort scores remain exploratory.

### 4.1 Implications of data leakage

Supervised models trained on released reports may encounter identical text at evaluation, whereas these zero-shot API runs did not fit on those splits. The historical DSS-observed TF-IDF prognosis classifier scored 0.8647 on all 887 and 0.7644 on 174 barcode-disjoint cases. After training deduplication, the clinical-plus-text Cox model scored 0.7926 on all 887 overlapping cases and 0.6940 on the 173 ID/text-disjoint cases. The cohorts differ markedly in size and case mix. Neither this contrast nor the prior fit's 0.7875 versus 0.6888 contrast quantifies memorization or causal score inflation.

Other report benchmarks may also require cross-split identity and text checks. We recommend patient-level grouping before splitting, a report-text overlap audit, and a newly constructed held-out split rather than simply deleting overlapping test rows after model development.

### 4.2 Cross-model comparison

DeepSeek V4 Flash had a numerically higher original-run cancer-type accuracy (0.9800 versus 0.9727) but the paired exact test was not significant (p = 0.0923); AJCC staging showed a larger paired difference (0.8165 versus 0.5993, p = 6.795 × 10⁻²²). The 181-case text-disjoint cancer-type comparison had identical accuracy. The tested prompting and reasoning configurations differed, and paired estimates are conditional on saved API responses rather than repeated model draws. The original Flash row files used by the verifier are not in this distributed package, so the full original-run tests are reported from the verifier's checked results rather than independently reproducible here.

### 4.3 Prognosis: prompting versus survival modeling

Full-test DeepSeek prognosis prompting scored 0.4716 accuracy (449/952) and 0.4792 fixed-label macro-F1, with 31 invalid responses. The old 0.3195 union-label F1 was a scoring artifact, not evidence of a still lower two-class F1. These findings indicate weak performance in this particular binary surrogate task and prompting configuration, not a general inability of LLMs to support outcomes research.

The deduplicated Cox models use survival time and event status rather than a dichotomized label. On the 173 ID/text-disjoint cases, the clinical-plus-text versus clinical-only C-index difference was 0.0389, but its paired interval crossed zero. The prior archived paired bootstrap failed; that failure should not be confused with the successful, separately computed bootstrap of the new fit. The 48-event disjoint cohort is too limited to establish a clinical survival tool.

### 4.4 Limitations

Several limitations should be noted. First, all analyses use the same TCGA-derived cohort. The 182 barcode-disjoint reports still include one train-matching text under a different barcode; removing it leaves 181 reports, 115 stage-labeled and 173 survival-eligible (48 events). This is an internal sensitivity check, not external validation. An independently sourced corpus and a prospectively specified patient-grouped, text-audited resplit remain necessary.

Second, Cox used SVD-reduced TF-IDF features rather than full TF-IDF or transformer embeddings. Although the new fit removes repeated patients and report texts from the training inputs and fits transformations only there, it uses the predefined train-plus-validation combination without a separate model-selection validation fold. The overlapping full test remains contaminated; only its locked 173-case subset is disjoint from original eligible training inputs. All 20 prior leave-one-cancer-type-out fits failed and integrated Brier scores were not calculated. Bootstrap uncertainty is conditional on the fitted model and does not capture training-set or API-run variation.

Third, the full-set DeepSeek prognosis run used zero-shot reasoning-mode prompting and had 31 empty outputs. The eight-shot result applies only to an earlier 100-case subset. Prompt-budget and few-shot sensitivity on the full test remain untested.

Fourth, the verifier checked historical original Flash row-level predictions against GPT but only supplied the human-readable verification report, not those source files or the machine-readable paired bootstrap output. Therefore, the original-run paired results cannot be independently rerun from the package, whereas the new-run subset and deduplicated Cox records can. Original Flash/Pro token logs remain unavailable for cost recalculation. The GPT-4o-mini request did not activate a corresponding reasoning mode, regardless of a `thinking=enabled` metadata field in its saved records.

Finally, zero-shot LLMs were not trained on these released splits, but differences between full and barcode-disjoint LLM scores remain subject to sample selection and output failures. No independent institutional pathology-report cohort was evaluated.

### 4.5 Clinical implications

The results motivate evaluating LLMs for extraction and staging support separately from censored outcome modeling. Any supervised model fitted on PathRep-Bench should be assessed on a newly designed patient-level split and, ultimately, independent reports. The Cox experiment illustrates an analysis path beyond binary thresholds, not a deployable prognostic tool.

For deployment, model outputs should include evidence spans, uncertainty estimates, and audit logs. Performance should be validated across institutions, report formats, cancer types, and time periods. Clinical integration should prioritize human oversight and clear boundaries between retrospective research classification and patient-specific care.

## 5 Conclusions

The released PathRep-Bench test split has substantial train-test barcode and report-text overlap, creating contamination risk for split-trained models. Fixed-label rescoring corrects three previously misstated F1 estimates; patient-paired tests distinguish supported AJCC from unsupported cancer-type superiority in these saved configurations. Refitting Cox after patient/text deduplication yields 0.6940 clinical-plus-text concordance on a 173-case disjoint subset, without establishing an improvement over clinical-only or external validity. Full-test DeepSeek prognosis prompting remains weak despite corrected F1. A patient-grouped, text-audited new split and independent cohort are needed before broad generalizability or clinical claims.

## Declarations

### Ethics approval and consent to participate
This study used publicly available, de-identified data derived from TCGA and the PathRep-Bench benchmark. No identifiable human participant data were collected.

### Consent for publication
Not applicable.

### Availability of data and materials
The public benchmark data are available from the PathRep-Bench project repository and TCGA pathology report resources. Source code and a private reproducibility bundle with corrected fixed-label scores, predictions, risk scores, split manifests, and the independent verification report are provided alongside this draft. The original historical Flash paired-source files and associated machine-readable verifier results are not included in that bundle; the corresponding paired estimates require those files for independent reproduction. No patient-external cohort was analyzed. Code is available at https://github.com/mohkone/DeepSeekPathrep.

### Competing interests
The authors declare no competing interests.

### Funding
The authors received no specific funding for this work.

### Authors' contributions
Mohamed Kone: Conceptualization, Methodology, Software, Formal Analysis, Investigation, Data Curation, Visualization, Writing – Original Draft. Gaoussou Haidara: Data Curation, Validation, Writing – Review & Editing. Chao Hou: Validation, Writing – Review & Editing. Shulin Wang: Supervision, Methodology, Writing – Review & Editing.

## References

1. Saluja R, Rosenthal J, Windon A, et al. Cancer type, stage and prognosis assessment from pathology reports using LLMs. Scientific Reports. 2025;15:27300. doi:10.1038/s41598-025-10709-4.
2. PathRep-Bench contributors. Pathrep-bench project repository. https://github.com/rachitsaluja/PathRep-Bench. Accessed 19 Jun 2026.
3. Rosenthal Laboratory. TCGA pathology notes dataset. https://huggingface.co/datasets/rosenthal/tcga-path-notes. Accessed 19 Jun 2026.
4. Weinstein JN, Collisson EA, Mills GB, et al. The Cancer Genome Atlas pan-cancer analysis project. Nature Genetics. 2013;45:1113-1120. doi:10.1038/ng.2764.
5. DeepSeek. DeepSeek API documentation. https://api-docs.deepseek.com/. Accessed 19 Jun 2026.
6. DeepSeek. DeepSeek API pricing documentation. https://api-docs.deepseek.com/quick_start/pricing. Accessed 19 Jun 2026.
7. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research. 2011;12:2825-2830.
8. Liu J, Lichtenberg T, Hoadley KA, et al. An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics. Cell. 2018;173:400-416.e11. doi:10.1016/j.cell.2018.02.052.
9. GerkeLab. TCGA PanCanAtlas clinical data. https://github.com/GerkeLab/TCGAclinical. Accessed 21 Sep 2026.
