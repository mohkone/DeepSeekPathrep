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

We audited the PathRep-Bench train, validation, and test splits for patient-level overlap using TCGA barcodes and SHA-256 report text hashes. DeepSeek V4 Flash and GPT-4o-mini predictions were analyzed for cancer type identification and AJCC stage prediction; historical DeepSeek V4 Pro aggregate results were retained for context. DeepSeek V4 Flash prognosis prompting was evaluated on 952 reports. A barcode-disjoint test subset was formed by excluding test reports whose patient barcodes appeared in training. Supervised sensitivity analyses used the 887 test cases with usable disease-specific survival (DSS) time and event status, of which 174 were barcode-disjoint. Cox proportional hazards models used DSS time and TCGA Clinical Data Resource event status. Five hundred report-level bootstrap resamples were used for Cox C-index confidence intervals.

### Results

The audit found that 770 of 952 test reports (80.9%) shared barcodes with training and that all 770 had byte-identical text. One additional test report had a matching normalized-text hash under a different barcode. In the DSS-observed subset, TF-IDF logistic regression cancer type accuracy was 0.9910 on 887 cases and 0.9713 on the nested 174-case barcode-disjoint subset; prognosis accuracy was 0.8647 and 0.7644, respectively. DeepSeek V4 Flash cancer type accuracy was 0.9800 on the original 952-case test and 0.9835 on the 182-case barcode-disjoint subset. Its original-test cancer type and AJCC accuracy point estimates exceeded GPT-4o-mini's (0.9800 vs. 0.9727 and 0.8165 vs. 0.5993, respectively); the original DeepSeek row-level predictions were unavailable for paired tests. Full-test DeepSeek prognosis prompting achieved 0.4716 accuracy with 31 empty outputs. The clinical-plus-text Cox model achieved a C-index of 0.7875 (95% bootstrap interval, 0.7567–0.8193) on 887 cases and 0.6818 on the nested 174-case barcode-disjoint subset.

### Conclusions

The released splits contain substantial cross-split report duplication, making supervised and fine-tuned model evaluations vulnerable to train-test contamination. The nested barcode-disjoint sensitivity results differ from full-test results, but their smaller size and different case mix prevent attributing the whole difference to duplication. Zero-shot evaluations do not train on these splits, although their subset metrics can still vary with case mix. This single TCGA-derived cohort does not provide external validation; independent data and patient-level splitting remain necessary.

**Keywords:** pathology reports, data leakage, benchmark evaluation, large language models, DeepSeek, GPT-4o-mini, cancer staging, survival analysis, Cox proportional hazards, TCGA

---

## 1 Introduction

Pathology reports are among the most information-rich documents in oncology. They summarize tumor histology, tissue site, grade, resection status, nodal involvement, metastatic findings, biomarker context, and stage group. These reports are central to cancer diagnosis and treatment planning, but their free-text format limits large-scale computational reuse for cohort construction, registry enrichment, and outcomes research.

Recent progress in large language models (LLMs) has made pathology report understanding more feasible. Unlike earlier rule-based systems or narrow named-entity recognition pipelines, LLMs can interpret varied report formats, follow label-constrained instructions, and produce structured outputs from heterogeneous text. PathRep-Bench established a benchmark for evaluating this capability across cancer type identification, AJCC stage identification, and prognosis assessment from pathology reports [1]. The benchmark uses data derived from The Cancer Genome Atlas (TCGA) pathology reports and clinical outcome resources [2, 3, 4].

A fundamental assumption of any benchmark is that the training, validation, and test splits are disjoint. When the same reports appear in both training and test sets, supervised models can memorize specific examples, inflating apparent performance. This concern is particularly relevant for TCGA-derived data, where multiple data files may be merged using patient identifiers, and split procedures may operate at the row level rather than the patient level. However, the extent of cross-split duplication in PathRep-Bench has not been independently audited.

A second concern is the choice of prognosis endpoint. PathRep-Bench uses a binary label derived from cancer-type mean disease-specific survival (DSS) time: patients surviving beyond the mean are labeled "good" and those below are labeled "poor." This thresholding discards time-to-event and censoring information, creates ambiguity for cases near the threshold, and does not correspond to a clinically validated prognostic category. Proper survival analysis methods, such as Cox proportional hazards modeling, can use the actual survival time and censoring status to provide a more clinically meaningful evaluation.

This study makes three contributions. First, we audit the released splits for cross-split duplication and perform a nested barcode-disjoint sensitivity analysis. Second, we report DeepSeek and GPT-4o-mini classification point estimates and extend DeepSeek prognosis prompting to the full test set; comparisons with earlier DeepSeek Flash and Pro runs are descriptive because their row-level files were not available for this revision. Third, we evaluate Cox proportional hazards models using TCGA-CDR censoring information instead of relying exclusively on thresholded DSS labels. These analyses examine internal robustness, not generalization to an external cohort.

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

We excluded test reports whose patient barcodes appeared in training, leaving a 182-report **barcode-disjoint sensitivity subset** of the original 952-case test. This is a nested subset, not a new patient-level train/test resplit and not an external validation cohort. One of its reports still has a normalized-text match to a different-barcode training report. Among test cases eligible for DSS modeling, 174 of 887 were barcode-disjoint. AJCC staging had 116 labeled reports in the 182-case subset.

### 2.4 Models

We evaluated the following models:

- **DeepSeek V4 Flash**: A cost-effective reasoning model with JSON output support [5, 6].
- **DeepSeek V4 Pro**: A larger DeepSeek model with higher per-token pricing [5, 6].
- **GPT-4o-mini**: OpenAI's cost-efficient multimodal model, evaluated via the OpenAI chat completions API.
- **TF-IDF logistic regression**: A supervised sensitivity analysis on the DSS-observed subset used lowercasing, English stop-word removal, unigrams and bigrams, sublinear term frequency, a minimum document frequency of 2, and at most 50,000 features. Models trained on 7,962 train-plus-validation cases used balanced class weights, C = 2.0, maximum 3,000 iterations, and random seed 2026. The binary prognosis classifier used `liblinear`; the 32-class cancer-type classifier used `lbfgs` [7]. These sensitivity runs differ from the original manuscript's 952-case prognosis baseline.

### 2.5 Prompting framework

The prompts required JSON-only answers. The cancer type prompt supplied the 32 allowed labels, and the stage prompt supplied Stage I–IV; prognosis prompting supplied the cancer-type mean DSS threshold. DeepSeek cancer type calls used non-reasoning mode, while DeepSeek stage and prognosis calls used reasoning mode. GPT-4o-mini used the shared task prompts through a separate API without an equivalent DeepSeek reasoning-mode setting. Therefore, the cross-model contrast compares the tested configurations, not isolated model architectures. The original 100-report few-shot prognosis run supplied eight training examples; the new 952-report run was zero-shot.

Model output failures, including empty final responses, were counted as incorrect. The existing runner computes macro F1 over the union of observed gold labels and normalized prediction values, treating an empty prediction as an additional zero-F1 label. This nonstandard failure-label convention makes the reported macro F1 sensitive to output failures; we retain it to match the saved metrics.

### 2.6 Cox proportional hazards survival modeling

We obtained disease-specific survival event and censoring status from the TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR) [8] via the GerkeLab clinical data file [9]. The event indicator was set to 1 for patients who died with tumor and 0 for patients coded as alive or dead tumor-free. Disease-specific survival time was taken from the PathRep-Bench data (in days). We retained only cases with a positive DSS time and known event status.

We fitted Cox proportional hazards models using the following feature sets:

1. **Clinical only**: Age at diagnosis, gender, cancer type (one-hot encoded).
2. **Clinical + TF-IDF**: Clinical features plus 100 SVD-reduced TF-IDF components from pathology report text.
3. **TF-IDF only**: 100 SVD-reduced TF-IDF components only.

Models were trained on the combined training and validation splits (7,962 records with usable survival data) with Cox penalizer 0.1 and evaluated on 887 test records. A held-out C-index is the primary metric; 95% percentile intervals were computed using 500 report-level resamples of the original test set. Saved per-record partial-hazard scores reproduce the full-set C-indices and permit recalculation on the nested 174-case barcode-disjoint subset. The attempted paired bootstrap comparison failed, so no paired Cox difference interval or p value is reported.

### 2.7 Statistical analysis

We report accuracy and runner-defined macro F1 for classification tasks, counting empty outputs in the denominator and as an additional prediction value for macro F1. Confidence intervals for original DeepSeek cancer type and stage results are carried over from the earlier manuscript. No confidence interval was saved for the new GPT-4o-mini results or the 952-case DeepSeek prognosis run. Original DeepSeek Flash/Pro row-level predictions were not available in this workspace, so no paired DeepSeek-versus-GPT-4o-mini accuracy or F1 test was possible. Cox intervals used 500 test-report bootstrap resamples; the saved paired Cox bootstrap and leave-one-cancer-type-out runs failed and are not used as evidence.

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

For the TF-IDF sensitivity runs trained on the 7,962 DSS-observed train-plus-validation records, cancer type accuracy was 0.9910 on 887 test cases and 0.9713 on the nested 174-case barcode-disjoint subset. Prognosis accuracy was 0.8647 and 0.7644, respectively (Table 2). These runs must not be conflated with the original manuscript's 952-case TF-IDF prognosis result (0.8571 accuracy). The barcode-disjoint subset is much smaller and has a different case mix; differences between its score and the full-set score are descriptive, not an identified causal effect of duplication.

DeepSeek V4 Flash cancer type accuracy was 0.9800 on the original 952-case test and 0.9835 on the 182-case barcode-disjoint subset. Its stage accuracy was 0.8165 on 594 labeled original-test cases and 0.8103 on 116 labeled barcode-disjoint cases; seven of the latter produced empty responses. These zero-shot results do not establish equivalent performance across populations. Original-run DeepSeek metrics in this table come from the earlier manuscript's aggregate results, whose row-level records were not in the verification bundle.

**Table 2.** Original test versus nested barcode-disjoint sensitivity subset. The Cox scores are from the same fitted model; the supervised logistic regression runs in the first two rows were separately fitted on the DSS-observed subset. A one-report cross-barcode text match remains in the subset.

| Task and method | Original test, n; score | Barcode-disjoint subset, n; score | Metric |
|-----------------|------------------------:|----------------------------------:|--------|
| Cancer type, TF-IDF logistic regression | 887; 0.9910 | 174; 0.9713 | Accuracy |
| Prognosis, TF-IDF logistic regression | 887; 0.8647 | 174; 0.7644 | Accuracy |
| Prognosis, same classifier | 887; 0.8617 | 174; 0.7586 | Macro F1 |
| Cancer type, DeepSeek Flash zero-shot | 952; 0.9800 | 182; 0.9835 | Accuracy |
| AJCC stage, DeepSeek Flash zero-shot | 594; 0.8165 | 116; 0.8103 | Accuracy |
| Cox clinical plus text | 887; 0.7875 | 174; **0.6818** | C-index |

### 3.3 Cross-model comparison

DeepSeek V4 Flash had higher reported point estimates than GPT-4o-mini on both cancer type identification and AJCC stage prediction (Table 3). For cancer type, Flash's historical aggregate result was 0.9800 accuracy and 0.9778 macro F1, compared with 0.9727 and 0.9394 for the archived GPT-4o-mini run. For AJCC stage, the corresponding values were 0.8165 and 0.7841 versus 0.5993 and 0.5904. GPT-4o-mini predicted Stage II for 63 Stage I reports and Stage IV for 32 Stage III reports. No paired significance test was possible without the original DeepSeek row-level predictions, and the tested DeepSeek reasoning-mode configuration was not matched by an equivalent GPT-4o-mini configuration.

**Table 3.** Descriptive model-configuration comparison on the original test set. DeepSeek Flash/Pro values and their confidence intervals are historical aggregate results from the earlier manuscript, not recalculated from row-level files in the verification bundle; the GPT-4o-mini results are backed by archived prediction JSONLs. No paired cross-model test is reported.

| Task | Model | n | Accuracy (95% CI) | Macro F1 | Errors |
|------|-------|---|------------------|----------|--------|
| Cancer type | DeepSeek V4 Flash | 952 | 0.9800 (0.9695–0.9884) | 0.9778 | 0 |
| Cancer type | GPT-4o-mini | 952 | 0.9727 | 0.9394 | 0 |
| Cancer type | DeepSeek V4 Pro | 952 | 0.9769 (0.9664–0.9863) | 0.9685 | 0 |
| AJCC stage | DeepSeek V4 Flash | 594 | 0.8165 (0.7845–0.8468) | 0.7841 | 0 |
| AJCC stage | GPT-4o-mini | 594 | 0.5993 | 0.5904 | 0 |
| AJCC stage | DeepSeek V4 Pro | 594 | 0.8098 (0.7761–0.8401) | 0.6288 | 14 |

Earlier aggregate analyses reported 14 empty DeepSeek V4 Pro stage outputs with a 4,096-token completion budget. The original token-level cost logs and row-level Flash/Pro predictions were unavailable in this verification bundle, so revised cost and paired-comparison claims are deferred pending recovery of those records.

### 3.4 Full-test DeepSeek prognosis prompting

DeepSeek V4 Flash prognosis prompting was evaluated on all 952 test reports (Table 4), extending the original 100-report subset. It returned 449 correct classifications, 31 empty outputs, 0.4716 accuracy, and 0.3195 runner-defined macro F1. The latter averages over the two gold classes and an additional empty-output prediction label. Among 921 valid outputs, accuracy was 449/921 (0.4875) and the two-class macro F1 was approximately 0.4872; these are secondary valid-output-only values. The original 100-report subset scores are historical aggregate results, not rerun in the verification archive. No new full-test confidence interval was calculated.

**Table 4.** DeepSeek prognosis prompting. The full-test results have saved per-row predictions. The 100-case zero- and eight-shot values, and the 952-case TF-IDF baseline, are earlier manuscript aggregates and are not reproduced by the new prediction archive.

| Condition | n | Accuracy | Macro F1 | Errors |
|-----------|---|----------|----------|--------|
| Zero-shot (100-report subset) | 100 | 0.5300 | 0.5083 | 0 |
| 8-shot (100-report subset) | 100 | 0.5700 | 0.5647 | 0 |
| Zero-shot (full test) | 952 | 0.4716 | 0.3195 | 31 |
| TF-IDF LogReg (original supervised baseline) | 952 | 0.8571 | 0.8543 | — |

### 3.5 Cox proportional hazards survival model

The Cox clinical-plus-text model had a C-index of 0.7875 (95% bootstrap interval, 0.7567–0.8193) on the original 887-case test set (Table 5). The clinical-only and text-only models scored 0.7648 and 0.7780, respectively. The clinical-plus-text point estimate was 0.0227 above clinical-only on that set, but the paired bootstrap run failed; no paired difference interval or significance conclusion is available.

**Table 5.** Cox proportional hazards results. Intervals use 500 test-report bootstrap resamples on the original test set. No interval was calculated for the nested barcode-disjoint subset. All C-indices below were reproduced from the archived per-record risk scores.

| Model | C-index | 95% CI | n_train | n_test |
|-------|---------|--------|---------|--------|
| Clinical only | 0.7648 | 0.7296–0.7975 | 7,962 | 887 |
| Clinical + TF-IDF | 0.7875 | 0.7567–0.8193 | 7,962 | 887 |
| TF-IDF only | 0.7780 | 0.7480–0.8109 | 7,962 | 887 |
| Clinical only (barcode-disjoint subset) | 0.6551 | Not calculated | 7,962 | 174 |
| Clinical + TF-IDF (barcode-disjoint subset) | **0.6818** | Not calculated | 7,962 | 174 |
| TF-IDF only (barcode-disjoint subset) | 0.6933 | Not calculated | 7,962 | 174 |

The train-plus-validation set contained 1,658 events and 6,304 censored records; the 887-case test contained 183 events and 704 censored records. The nested barcode-disjoint survival subset contained 174 records and 49 events. On that subset, the same fitted clinical-plus-text model scored **0.6818**, not the previously quoted 0.7210. This smaller-subset score is an internal sensitivity result and cannot isolate the causal effect of duplicate training reports from differences in case mix.

### 3.6 Error analysis

For cancer type identification, the main DeepSeek errors were clinically plausible neighboring-label confusions: rectum adenocarcinoma predicted as colon adenocarcinoma (5 errors), kidney renal papillary cell carcinoma predicted as clear cell carcinoma (3 errors), and stomach versus esophageal confusion (3 errors).

Earlier aggregate AJCC results reported Stage IV recall of 0.567 (34/60) for DeepSeek V4 Flash, with 24 Stage IV reports predicted as Stage III. In the newly archived 116-case barcode-disjoint stage run, Stage IV recall among the 18 Stage IV reports was 13/18 (0.722); seven overall stage outputs were empty. These different samples do not support attributing the recall difference to duplication.

GPT-4o-mini showed more diffuse stage confusion, with substantial errors across all adjacent stage pairs. Stage I→II confusion (63 errors) and Stage III→IV confusion (32 errors) were the dominant error modes.

## 4 Discussion

This audit found substantial cross-split report duplication in the released PathRep-Bench splits: 80.9% of test reports shared a training barcode and exact report text. This creates a risk of contamination for models trained on the released splits, although the different scores on the nested barcode-disjoint subset do not by themselves measure a causal inflation effect. Original-test DeepSeek Flash accuracy point estimates exceeded those of the tested GPT-4o-mini configuration, without a recoverable paired comparison. Full-test DeepSeek prognosis prompting was weak under its tested setup, and the Cox analysis used DSS times and event indicators instead of only thresholded prognosis labels.

### 4.1 Implications of data leakage

Supervised models trained on released train-split reports may encounter identical text at evaluation, whereas our zero-shot API runs did not fit on that train split. The DSS-observed TF-IDF prognosis score was 0.8647 on all 887 test cases and 0.7644 on the 174-case barcode-disjoint subset; the same fitted clinical-plus-text Cox model scored 0.7875 and 0.6818 on those respective groups. These contrasts raise concern about apparent supervised performance, but the subset is smaller and may be more difficult for reasons unrelated to duplication. We therefore cannot quantify the fraction of any score difference attributable to memorization or infer that all earlier PathRep-Bench comparisons were biased in one direction.

Other report benchmarks may also require cross-split identity and text checks. We recommend patient-level grouping before splitting, a report-text overlap audit, and a newly constructed held-out split rather than simply deleting overlapping test rows after model development.

### 4.2 Cross-model comparison

DeepSeek V4 Flash had higher historical accuracy point estimates than the newly run GPT-4o-mini configuration for cancer type (0.9800 vs. 0.9727) and AJCC stage (0.8165 vs. 0.5993). The tested prompting and reasoning configurations differed; the data do not isolate model architecture or reasoning mode as the cause. The missing original DeepSeek row-level predictions preclude paired statistical testing, and historical Flash/Pro cost estimates cannot be rechecked without the corresponding token records.

### 4.3 Prognosis: prompting versus survival modeling

Full-test DeepSeek prognosis prompting scored 0.4716 accuracy (449/952), including 31 empty responses under the tested completion budget. Its runner-defined macro F1 of 0.3195 includes an empty-output label; among valid responses, accuracy was 0.4875. These findings indicate weak performance in this particular binary surrogate task and prompting configuration, not a general inability of LLMs to support outcome research.

The Cox model uses survival time and event status rather than a dichotomized label. Clinical-plus-text had a higher C-index point estimate than clinical-only (0.7875 vs. 0.7648) on the original test set, but the attempted paired comparison failed. The original set contains train-overlapping reports, and the barcode-disjoint subset has only 49 events; these results are exploratory rather than evidence of a validated clinical survival model.

### 4.4 Limitations

Several limitations should be noted. First, all analyses use the same TCGA-derived cohort. The barcode-disjoint subset contains only 182 reports (174 with usable survival data and 116 with stage labels), has different case mix, and contains one normalized-text match to a training report with a different barcode. It is an internal sensitivity check, not external validation. An independently sourced corpus and a new patient-level, text-audited split remain necessary.

Second, the Cox model used SVD-reduced TF-IDF features rather than full TF-IDF or transformer-based embeddings. The saved robustness run did not produce paired C-index comparisons or successful leave-one-cancer-type-out results, and integrated Brier scores were not calculated. Richer text representations and broader clinical features could produce different rankings.

Third, the full-set DeepSeek prognosis run used zero-shot reasoning-mode prompting and had 31 empty outputs. The eight-shot result applies only to an earlier 100-case subset. Prompt-budget and few-shot sensitivity on the full test remain untested.

Fourth, the original DeepSeek Flash/Pro row-level prediction files and token-level logs were unavailable for this revision. Their historical aggregate results can be described, but paired cross-model tests and revised cost calculations cannot be verified from the bundled artifacts. The GPT-4o-mini API setup did not mirror DeepSeek's reasoning-mode configuration.

Finally, zero-shot LLMs were not trained on these released splits, but differences between full and barcode-disjoint LLM scores remain subject to sample selection and output failures. No independent institutional pathology-report cohort was evaluated.

### 4.5 Clinical implications

The results motivate evaluating LLMs for extraction and staging support separately from censored outcome modeling. Any supervised model fitted on PathRep-Bench should be assessed on a newly designed patient-level split and, ultimately, independent reports. The Cox experiment illustrates an analysis path beyond binary thresholds, not a deployable prognostic tool.

For deployment, model outputs should include evidence spans, uncertainty estimates, and audit logs. Performance should be validated across institutions, report formats, cancer types, and time periods. Clinical integration should prioritize human oversight and clear boundaries between retrospective research classification and patient-specific care.

## 5 Conclusions

The released PathRep-Bench test split has substantial train-test barcode and report-text overlap, creating a contamination risk for split-trained models. Nested barcode-disjoint sensitivity analyses and a Cox survival analysis clarify, but do not causally quantify, how evaluation results may change with the sample. The tested DeepSeek Flash configuration had higher cancer-type and AJCC accuracy point estimates than GPT-4o-mini; full-test DeepSeek prognosis prompting was weak with 31 empty outputs. A patient-level, report-text-audited resplit and an independent external cohort are needed before making broad generalizability or clinical claims.

## Declarations

### Ethics approval and consent to participate
This study used publicly available, de-identified data derived from TCGA and the PathRep-Bench benchmark. No identifiable human participant data were collected.

### Consent for publication
Not applicable.

### Availability of data and materials
The public benchmark data are available from the PathRep-Bench project repository and TCGA pathology report resources. Source code and reproducible analysis scripts are available at https://github.com/mohkone/DeepSeekPathrep.

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
