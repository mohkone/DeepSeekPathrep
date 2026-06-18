# DeepSeek Article Blueprint

Working title:

DeepSeek for Pathology Report Understanding: A Benchmark Study of Cancer Type Extraction, AJCC Staging, and Prognosis Prediction

## Research Question

How well does DeepSeek perform across pathology information extraction, AJCC staging, and prognosis prediction, and where does classical supervised learning outperform LLM prompting?

## Core Message

DeepSeek achieves state-of-the-art or near-state-of-the-art performance for pathology information extraction and strong performance for AJCC staging, but dedicated machine-learning models substantially outperform LLM prompting for prognosis prediction.

The resulting architecture is:

- DeepSeek for information extraction.
- DeepSeek for staging support.
- Classical ML for prognosis prediction.

## Manuscript Contributions

1. First independent benchmark of DeepSeek on the PathRep-Bench pathology tasks.
2. Demonstration that DeepSeek matches leading proprietary models for cancer-type identification.
3. Demonstration that DeepSeek provides strong AJCC stage prediction using reasoning-mode prompting.
4. Evidence that prognosis prediction remains challenging for LLM prompting, even with few-shot examples.
5. A hybrid architecture recommendation combining DeepSeek-based report understanding with classical ML prognosis modeling.

## Main Hypotheses

1. DeepSeek will perform strongly on cancer type identification because it is mainly an information extraction task.
2. DeepSeek thinking mode will improve AJCC stage identification because staging requires multi-step reasoning over tumor extent, nodal involvement, and metastasis evidence.
3. Prognosis will remain the hardest task for prompting, and a supervised text/clinical classifier will outperform pure LLM prompting.

## Dataset

Use the same public data sources as the paper:

- TCGA-Reports pathology text
- TCGA Pan-Cancer Clinical Data Resource survival and clinical labels

Reproduce the 80:10:10 stratified split if possible. Keep the final test set untouched until all prompt and model choices are frozen.

## Methods

### DeepSeek Prompting Baseline

Evaluate `deepseek-v4-flash` and `deepseek-v4-pro`.

- Cancer type: zero-shot, thinking disabled, strict JSON output.
- AJCC stage: zero-shot multiple choice, thinking enabled, strict JSON final answer.
- Prognosis: mean DSS threshold included in prompt, thinking enabled, optional few-shot examples from training data.

### Hybrid Prognosis Alternative

Use DeepSeek to extract structured pathology variables into JSON. Then apply:

- DeepSeek or rule/retrieval-supported AJCC staging
- a supervised prognosis classifier or survival model

This alternative is scientifically attractive because it separates language understanding from outcome prediction.

## Evaluation

Report:

- accuracy
- macro F1
- per-cancer-type confusion matrix
- error analysis for Stage II versus Stage III, Stage III versus Stage IV, and similar gastrointestinal cancers
- cost and latency per report

## Expected Tables

1. Overall performance by task and model.
2. Per-cancer-type performance for cancer type identification.
3. Per-cancer-type performance for AJCC stage.
4. Prognosis performance by cancer type.
5. Ablation of thinking mode, few-shot examples, and hybrid rules.

## Discussion Points

- Whether thinking mode helps staging more than cancer type identification.
- Whether JSON output reduces parsing failures.
- Whether DeepSeek confuses adjacent stages or similar cancer types.
- Why prognosis is less suitable as a pure LLM prompting task.
- Why classical supervised models can outperform LLM prompting when labeled outcome data are available.
- Privacy implications of API use, local open-weight deployment, and local classical prognosis modeling.

## Clinical Safety Note

This workflow is for retrospective research only. It should not be used to make patient-care decisions without clinical validation, institutional review, and prospective evaluation.
