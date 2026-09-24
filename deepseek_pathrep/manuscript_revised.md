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

We audited the PathRep-Bench train, validation, and test splits for patient-level overlap using TCGA barcodes and SHA-256 report text hashing. DeepSeek V4 Flash, DeepSeek V4 Pro, and GPT-4o-mini were evaluated on cancer type identification (952 test reports), AJCC stage prediction (594 labeled reports), and prognosis prediction (952 reports) using JSON-constrained prompting. A deduplicated test set was created by excluding reports whose patient barcodes appeared in the training set. Cox proportional hazards models were fitted using disease-specific survival time and censoring status merged from the TCGA Clinical Data Resource, with TF-IDF text features and clinical variables. Bootstrap 95% confidence intervals (500 resamples) and paired McNemar tests were used for statistical comparisons.

### Results

The audit revealed that 770 of 952 test reports (80.9%) shared patient barcodes with the training set, and all 770 had byte-identical report text. On the deduplicated test set (182 reports), supervised TF-IDF logistic regression accuracy dropped from 0.9910 to 0.9713 for cancer type identification and from 0.8647 to 0.7644 for prognosis prediction. In contrast, DeepSeek V4 Flash zero-shot metrics were stable across leaked and clean splits (cancer type: 0.9800 vs. 0.9835). DeepSeek V4 Flash outperformed GPT-4o-mini on cancer type identification (0.9800 vs. 0.9727 accuracy) and AJCC stage prediction (0.8165 vs. 0.5993). Full-test DeepSeek prognosis prompting (952 reports) achieved 0.4716 accuracy, confirming weak LLM performance on prognosis. The Cox model achieved a C-index of 0.7875 (95% CI, 0.7567–0.8193) with clinical variables and TF-IDF text features.

### Conclusions

PathRep-Bench contains substantial cross-split report duplication that inflates supervised model metrics but does not affect zero-shot LLM evaluations. This differential inflation means prior comparisons between supervised and LLM approaches on this benchmark have overstated the advantage of supervised methods. Proper time-to-event modeling and patient-level data splitting are needed before pathology report benchmarks can support reliable model comparisons.

**Keywords:** pathology reports, data leakage, benchmark evaluation, large language models, DeepSeek, GPT-4o-mini, cancer staging, survival analysis, Cox proportional hazards, TCGA

---

## 1 Introduction

Pathology reports are among the most information-rich documents in oncology. They summarize tumor histology, tissue site, grade, resection status, nodal involvement, metastatic findings, biomarker context, and stage group. These reports are central to cancer diagnosis and treatment planning, but their free-text format limits large-scale computational reuse for cohort construction, registry enrichment, and outcomes research.

Recent progress in large language models (LLMs) has made pathology report understanding more feasible. Unlike earlier rule-based systems or narrow named-entity recognition pipelines, LLMs can interpret varied report formats, follow label-constrained instructions, and produce structured outputs from heterogeneous text. PathRep-Bench established a benchmark for evaluating this capability across cancer type identification, AJCC stage identification, and prognosis assessment from pathology reports [1]. The benchmark uses data derived from The Cancer Genome Atlas (TCGA) pathology reports and clinical outcome resources [2, 3, 4].

A fundamental assumption of any benchmark is that the training, validation, and test splits are disjoint. When the same reports appear in both training and test sets, supervised models can memorize specific examples, inflating apparent performance. This concern is particularly relevant for TCGA-derived data, where multiple data files may be merged using patient identifiers, and split procedures may operate at the row level rather than the patient level. However, the extent of cross-split duplication in PathRep-Bench has not been independently audited.

A second concern is the choice of prognosis endpoint. PathRep-Bench uses a binary label derived from cancer-type mean disease-specific survival (DSS) time: patients surviving beyond the mean are labeled "good" and those below are labeled "poor." This thresholding discards time-to-event and censoring information, creates ambiguity for cases near the threshold, and does not correspond to a clinically validated prognostic category. Proper survival analysis methods, such as Cox proportional hazards modeling, can use the actual survival time and censoring status to provide a more clinically meaningful evaluation.

This study makes three contributions. First, we audit PathRep-Bench for cross-split report duplication using patient barcodes and report text hashing, and quantify the impact of this duplication on both supervised and zero-shot LLM metrics. Second, we conduct a cross-model comparison of DeepSeek V4 Flash, DeepSeek V4 Pro, and GPT-4o-mini on cancer type identification, AJCC stage prediction, and full-test prognosis prompting. Third, we introduce Cox proportional hazards survival modeling with TCGA-CDR censoring data as a methodologically stronger alternative to the binary threshold approach. Together, these contributions address benchmark reliability, model generalizability, and outcome modeling for pathology report understanding.

## 2 Methods

### 2.1 Dataset

We used the public PathRep-Bench data derived from TCGA pathology reports and TCGA clinical outcome resources [1, 2, 3, 4]. The prepared dataset contained 9,523 pathology reports with fixed train, validation, and test splits: 7,618 training reports, 953 validation reports, and 952 test reports. Each row included pathology report text, cancer type, AJCC stage fields where available, disease-specific survival time, and derived prognosis labels.

For cancer type identification, all 952 test reports had labels across 32 TCGA cancer types. For AJCC staging, 594 of the 952 test reports had usable high-level stage labels (Stage I–IV). For prognosis, all 952 test reports had derived good or poor labels based on whether disease-specific survival exceeded the cancer-type mean DSS threshold.

### 2.2 Split leakage audit

We audited the PathRep-Bench train, validation, and test splits for cross-split report duplication using two methods:

1. **Patient barcode overlap**: We compared the set of TCGA patient barcodes (`bcr_patient_barcode`) across all splits. Barcode overlap indicates that the same patient appears in multiple splits.

2. **Report text hash verification**: We computed SHA-256 hashes of normalized report text (lowercased, whitespace-collapsed) for all reports in each split. Text hash overlap confirms that the same report content appears in multiple splits, not just different samples from the same patient.

For all overlapping barcodes between train and test, we verified whether the report text was byte-identical. We also checked for within-split duplicate barcodes. A reproducible audit script is provided (`audit_pathrep_leakage.py`).

### 2.3 Deduplicated test set

Based on the audit results, we created a deduplicated test set by excluding all test reports whose patient barcodes appeared in the training set. The resulting clean test set contained 182 reports (vs. 952 original). We refer to evaluations on this set as "clean-test" results and to the original split as "original-test" results throughout.

### 2.4 Models

We evaluated the following models:

- **DeepSeek V4 Flash**: A cost-effective reasoning model with JSON output support [5, 6].
- **DeepSeek V4 Pro**: A larger DeepSeek model with higher per-token pricing [5, 6].
- **GPT-4o-mini**: OpenAI's cost-efficient multimodal model, evaluated via the OpenAI chat completions API.
- **TF-IDF logistic regression**: A supervised baseline using TF-IDF features over pathology report text with lowercasing, English stop-word removal, unigrams and bigrams, sublinear term frequency, a minimum document frequency of 2, and a maximum of 50,000 features. Logistic regression used balanced class weights, the liblinear solver, C = 2.0, maximum 3,000 iterations, and a fixed random seed [7].

### 2.5 Prompting framework

All LLM calls used JSON-constrained prompting. The system instruction required valid JSON only, with no markdown or extra keys. Cancer type identification used non-reasoning mode because the answer is usually directly stated in the report. AJCC staging and prognosis prompting used reasoning mode because these tasks require synthesizing stage evidence or outcome-relevant findings. The prompt supplied the complete list of 32 allowed cancer type labels for cancer type identification, the four high-level stage labels for AJCC staging, and the cancer-type mean DSS threshold for prognosis. Few-shot prognosis used eight training examples balanced across good and poor labels.

Model output failures, including empty final responses and unparseable JSON, were counted as errors in the primary analysis. Both accuracy and macro F1 were computed with errors included as incorrect predictions, following the convention that macro F1 is defined over the union of observed gold and predicted labels [8].

### 2.6 Cox proportional hazards survival modeling

We obtained disease-specific survival event and censoring status from the TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR) [9] via the GerkeLab PanCanAtlas clinical data resource [10]. The event indicator was set to 1 for patients who died with tumor and 0 for patients who were alive or dead tumor-free. Disease-specific survival time was taken from the PathRep-Bench data (in days).

We fitted Cox proportional hazards models using the following feature sets:

1. **Clinical only**: Age at diagnosis, gender, cancer type (one-hot encoded).
2. **Clinical + TF-IDF**: Clinical features plus 100 SVD-reduced TF-IDF components from pathology report text.
3. **TF-IDF only**: 100 SVD-reduced TF-IDF components only.

Models were trained on the combined training and validation splits (7,962 patients with survival data) and evaluated on the test set. We report the concordance index (C-index) as the primary metric. Bootstrap 95% confidence intervals were computed with 500 resamples over test reports. Paired bootstrap delta C-index was used to compare clinical-only versus clinical-plus-TF-IDF models.

### 2.7 Statistical analysis

We report accuracy and macro F1 for classification tasks. For DeepSeek prompting, unparseable or empty outputs were counted as model errors and included in the denominator. We estimated 95% confidence intervals using nonparametric bootstrap resampling over reports with 2,000 bootstrap iterations and a fixed random seed. For paired model comparisons on the same reports, we used exact McNemar tests for paired accuracy and paired bootstrap differences for macro F1. For the Cox model, we report C-index with 500-resample bootstrap confidence intervals.

### 2.8 Cost analysis

We estimated inference cost using recorded token usage from each run and DeepSeek pricing documentation [6]. When cache-hit and cache-miss counts were unavailable, prompt tokens were conservatively priced as cache misses. We report estimated cost per report, cost per 1,000 reports, and cost per correct prediction.

## 3 Results

### 3.1 Cross-split report duplication audit

The audit revealed substantial cross-split report duplication in PathRep-Bench (Table 1).

**Table 1.** Cross-split report duplication audit.

| Comparison | Overlapping barcodes | Overlapping text hashes | Identical text |
|-----------|---------------------|------------------------|----------------|
| Train–Val | 765 / 953 (80.3%) | 765 | 765 (100%) |
| Train–Test | 770 / 952 (80.9%) | 771 | 770 (100%) |
| Val–Test | 0 / 952 (0%) | 0 | — |
| Within-split duplicates | 0 in all splits | — | — |

Of the 770 test-set barcodes that also appeared in the training set, all 770 had byte-identical report text. No within-split duplicate barcodes were found, and no val-test overlap was detected. The clean test set (excluding train-overlapping barcodes) contained 182 reports.

### 3.2 Impact of data leakage on supervised and LLM metrics

Data leakage differentially affected supervised and zero-shot LLM models (Table 2). Supervised TF-IDF logistic regression, trained on the contaminated training set, showed substantial performance drops on the clean test set: cancer type accuracy decreased from 0.9910 to 0.9713 and prognosis accuracy from 0.8647 to 0.7644. In contrast, DeepSeek V4 Flash zero-shot metrics were stable or slightly improved on the clean test set: cancer type accuracy changed from 0.9800 to 0.9835 and AJCC stage accuracy from 0.8165 to 0.8103 (7 model errors on the smaller clean set).

**Table 2.** Differential impact of data leakage on supervised and zero-shot LLM metrics.

| Task | Model type | Original test (leaked) | Clean test (no leak) | Change |
|------|-----------|----------------------|---------------------|--------|
| Cancer type | TF-IDF LogReg (supervised) | 0.9910 acc | 0.9713 acc | −1.97 pp |
| Cancer type | DeepSeek Flash (zero-shot) | 0.9800 acc | 0.9835 acc | +0.35 pp |
| AJCC stage | DeepSeek Flash (zero-shot) | 0.8165 acc | 0.8103 acc | −0.62 pp |
| Prognosis | TF-IDF LogReg (supervised) | 0.8647 acc | 0.7644 acc | −10.03 pp |
| Prognosis | TF-IDF LogReg (supervised, F1) | 0.8617 | 0.7586 | −10.31 pp |
| Cox survival | TF-IDF SVD (supervised) | 0.7847 C-index | 0.7210 C-index | −6.37 pp |

### 3.3 Cross-model comparison

DeepSeek V4 Flash outperformed GPT-4o-mini on both cancer type identification and AJCC stage prediction (Table 3). On cancer type identification, DeepSeek achieved 0.9800 accuracy (95% CI, 0.9695–0.9884) and 0.9778 macro F1, compared with 0.9727 accuracy and 0.9394 macro F1 for GPT-4o-mini. On AJCC stage prediction, the gap was larger: DeepSeek achieved 0.8165 accuracy (95% CI, 0.7845–0.8468) and 0.7841 macro F1, while GPT-4o-mini achieved 0.5993 accuracy and 0.5904 macro F1. GPT-4o-mini struggled with adjacent-stage confusion, particularly Stage I→II (63 errors) and Stage III→IV (32 errors).

**Table 3.** Cross-model comparison on PathRep-Bench tasks (original test set).

| Task | Model | n | Accuracy (95% CI) | Macro F1 | Errors |
|------|-------|---|------------------|----------|--------|
| Cancer type | DeepSeek V4 Flash | 952 | 0.9800 (0.9695–0.9884) | 0.9778 | 0 |
| Cancer type | GPT-4o-mini | 952 | 0.9727 | 0.9394 | 0 |
| Cancer type | DeepSeek V4 Pro | 952 | 0.9769 (0.9664–0.9863) | 0.9685 | 0 |
| AJCC stage | DeepSeek V4 Flash | 594 | 0.8165 (0.7845–0.8468) | 0.7841 | 0 |
| AJCC stage | GPT-4o-mini | 594 | 0.5993 | 0.5904 | 0 |
| AJCC stage | DeepSeek V4 Pro | 594 | 0.8098 (0.7761–0.8401) | 0.6288 | 14 |

DeepSeek V4 Pro did not improve over Flash for either task and produced 14 empty final responses during AJCC staging under a 4,096-token completion budget. DeepSeek V4 Flash was approximately 2.5× more cost-effective than Pro for cancer type identification and 4.3× more cost-effective for AJCC staging (Table 4).

**Table 4.** Cost analysis.

| Task | Model | Reports | Correct | Cost (USD) | Cost/1,000 |
|------|-------|---------|---------|------------|------------|
| Cancer type | DeepSeek V4 Flash | 952 | 933 | 0.18 | 0.19 |
| Cancer type | DeepSeek V4 Pro | 952 | 930 | 0.46 | 0.49 |
| AJCC stage | DeepSeek V4 Flash | 594 | 485 | 0.16 | 0.27 |
| AJCC stage | DeepSeek V4 Pro | 594 | 481 | 0.70 | 1.17 |

### 3.4 Full-test DeepSeek prognosis prompting

DeepSeek V4 Flash prognosis prompting was evaluated on the full 952-report test set (Table 5), extending the original 100-report subset analysis. Full-test zero-shot prompting achieved 0.4716 accuracy (95% CI, 0.4398–0.5032) and 0.3195 macro F1, with 31 model output errors (empty responses). The 100-report subset had achieved 0.5300 accuracy and 0.5083 macro F1 with zero errors. The full-test results confirm that DeepSeek prompting remains weak for prognosis prediction, barely above chance.

**Table 5.** DeepSeek prognosis prompting results.

| Condition | n | Accuracy | Macro F1 | Errors |
|-----------|---|----------|----------|--------|
| Zero-shot (100-report subset) | 100 | 0.5300 | 0.5083 | 0 |
| 8-shot (100-report subset) | 100 | 0.5700 | 0.5647 | 0 |
| Zero-shot (full test) | 952 | 0.4716 | 0.3195 | 31 |
| TF-IDF LogReg (supervised) | 952 | 0.8571 | 0.8543 | — |

### 3.5 Cox proportional hazards survival model

The Cox model with clinical variables and TF-IDF text features achieved a C-index of 0.7875 (95% CI, 0.7567–0.8193) on the original test set (887 patients with survival data; Table 6). Adding TF-IDF text features to clinical variables improved the C-index from 0.7648 (95% CI, 0.7296–0.7975) to 0.7875, indicating that pathology report text contains prognostic signal beyond structured clinical variables. TF-IDF features alone achieved a C-index of 0.7780 (95% CI, 0.7480–0.8109).

**Table 6.** Cox proportional hazards survival model results.

| Model | C-index | 95% CI | n_train | n_test |
|-------|---------|--------|---------|--------|
| Clinical only | 0.7648 | 0.7296–0.7975 | 7,962 | 887 |
| Clinical + TF-IDF | 0.7875 | 0.7567–0.8193 | 7,962 | 887 |
| TF-IDF only | 0.7780 | 0.7480–0.8109 | 7,962 | 887 |
| Clinical + TF-IDF (clean test) | 0.7210 | — | 7,962 | 174 |

The survival data comprised 1,658 events and 6,304 censored observations in the training set, and 183 events and 704 censored observations in the test set. On the clean test set (174 patients, 49 events), the C-index decreased to 0.7210, consistent with the leakage impact observed in supervised classification models.

### 3.6 Error analysis

For cancer type identification, the main DeepSeek errors were clinically plausible neighboring-label confusions: rectum adenocarcinoma predicted as colon adenocarcinoma (5 errors), kidney renal papillary cell carcinoma predicted as clear cell carcinoma (3 errors), and stomach versus esophageal confusion (3 errors).

For AJCC staging, DeepSeek V4 Flash showed stable recall for Stage I (0.852), Stage II (0.837), and Stage III (0.845), but reduced recall for Stage IV (0.567). Of 60 Stage IV reports, 24 were undercalled as Stage III. On the clean test set, Stage IV recall improved to 0.722 (13/18), suggesting that some Stage IV undercalls on the original test set may have been influenced by specific leaked reports.

GPT-4o-mini showed more diffuse stage confusion, with substantial errors across all adjacent stage pairs. Stage I→II confusion (63 errors) and Stage III→IV confusion (32 errors) were the dominant error modes.

## 4 Discussion

This study identified substantial cross-split report duplication in PathRep-Bench, with 80.9% of test-set reports appearing in the training set with identical text. This duplication inflates supervised model metrics but does not affect zero-shot LLM evaluations, creating a systematic bias in prior comparisons between supervised and LLM approaches on this benchmark. We also found that DeepSeek V4 Flash outperforms GPT-4o-mini on cancer type identification and AJCC stage prediction, that full-test DeepSeek prognosis prompting remains weak, and that Cox proportional hazards modeling provides a methodologically stronger framework for prognosis evaluation than the binary threshold approach.

### 4.1 Implications of data leakage

The differential impact of leakage on supervised and zero-shot LLM models has important implications. Supervised models trained on the contaminated training set can memorize specific report texts that also appear in the test set, inflating accuracy and macro F1. The 10 percentage-point drop in prognosis accuracy (0.8647 to 0.7644) and the 6.4-point drop in Cox C-index (0.7847 to 0.7210) on the clean test set quantify this inflation. In contrast, zero-shot LLM models do not train on the PathRep-Bench data, so their metrics are unaffected by the leakage. This means that prior studies comparing LLMs to supervised or fine-tuned models on PathRep-Bench have systematically overstated the advantage of supervised approaches.

This finding is not specific to PathRep-Bench. Any benchmark that merges multiple data files using patient identifiers and splits at the row level rather than the patient level risks similar leakage. We recommend that future pathology report benchmarks perform patient-level deduplication before splitting and report the overlap verification as part of the benchmark documentation.

### 4.2 Cross-model comparison

DeepSeek V4 Flash outperformed GPT-4o-mini on both cancer type identification (0.9800 vs. 0.9727) and AJCC stage prediction (0.8165 vs. 0.5993). The AJCC staging gap was particularly large, with GPT-4o-mini showing diffuse stage confusion across all adjacent pairs. This suggests that DeepSeek's reasoning mode may be better suited for the multi-step stage determination task, which requires synthesizing tumor extent, nodal status, and metastasis evidence. DeepSeek V4 Pro did not improve over Flash and produced 14 empty outputs during AJCC staging, reinforcing the cost-performance advantage of the Flash model.

### 4.3 Prognosis: prompting versus survival modeling

Full-test DeepSeek prognosis prompting (0.4716 accuracy) confirmed the weak performance observed in the 100-report subset (0.5300). The 31 model errors on the full test set further reduced the macro F1. This reinforces the conclusion that direct LLM prompting is not effective for prognosis prediction from pathology reports alone.

The Cox proportional hazards model offers a methodologically stronger alternative to the binary DSS-threshold approach. By using the actual survival time and censoring status, the Cox model preserves information that the threshold approach discards. The C-index of 0.7875 (clinical + TF-IDF) indicates that pathology report text contains meaningful prognostic signal, and the improvement over clinical-only features (0.7648) supports the value of text-based features for outcome modeling.

### 4.4 Limitations

Several limitations should be noted. First, this study uses a single public TCGA-derived dataset, and the clean test set (182 reports) is small and may not represent the full cancer-type distribution. The data leakage finding suggests that the benchmark needs revision, but the clean test set is too small for definitive conclusions. Future work should create a fresh patient-level split or use an external institutional cohort.

Second, the Cox model used SVD-reduced TF-IDF features rather than full TF-IDF or transformer-based embeddings. Richer text representations, longitudinal EHR features, and multimodal models could produce different rankings.

Third, the DeepSeek prognosis prompting was evaluated only in zero-shot mode on the full test set. Few-shot and retrieval-augmented configurations might improve performance.

Fourth, the cross-split duplication in PathRep-Bench affects the interpretation of all results on the original test set, including the LLM results. While zero-shot LLM metrics are not directly inflated by the leakage, the composition of the test set (80.9% leaked reports) means that the test population is not independent of the training population.

Finally, all analyses are retrospective and based on public de-identified data. No external validation cohort was available. The findings should be validated on independent institutional pathology report corpora.

### 4.5 Clinical implications

The proposed hybrid architecture remains relevant: DeepSeek for flexible information extraction and staging support, with supervised models for outcome prediction. However, the data leakage finding adds a critical caveat: any supervised model trained on PathRep-Bench should be evaluated on a properly deduplicated test set. The Cox survival model provides a template for moving beyond binary threshold endpoints toward clinically meaningful time-to-event analysis.

For deployment, model outputs should include evidence spans, uncertainty estimates, and audit logs. Performance should be validated across institutions, report formats, cancer types, and time periods. Clinical integration should prioritize human oversight and clear boundaries between retrospective research classification and patient-specific care.

## 5 Conclusions

PathRep-Bench contains substantial cross-split report duplication (80.9% test-train overlap with identical text) that inflates supervised model metrics but not zero-shot LLM metrics. This differential inflation means prior comparisons between supervised and LLM approaches on this benchmark have been biased. DeepSeek V4 Flash outperforms GPT-4o-mini on cancer type identification and AJCC stage prediction, while full-test prognosis prompting remains weak. Cox proportional hazards modeling provides a methodologically stronger framework for prognosis evaluation. Patient-level data splitting and time-to-event modeling are needed before pathology report benchmarks can support reliable model comparisons.

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
9. Gu Y, Tinn R, Cheng H, et al. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare. 2022;3(1):1-23. doi:10.1145/3458754.
10. GerkeLab. TCGA PanCanAtlas clinical data. https://github.com/GerkeLab/TCGAclinical. Accessed 21 Sep 2026.
