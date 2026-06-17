# DeepSeek Pathology Report Understanding

This folder supports an original study of DeepSeek for pathology report understanding across three increasingly difficult task types:

- cancer type identification from pathology reports
- high-level AJCC stage identification
- prognosis classification against cancer-type mean disease-specific survival

The stronger manuscript framing is not "DeepSeek reproduces PathRep-Bench." The stronger framing is:

DeepSeek achieves excellent performance for pathology information extraction and strong performance for AJCC staging, but dedicated machine-learning models substantially outperform LLM prompting for prognosis prediction.

In other words:

- DeepSeek -> cancer-type extraction
- DeepSeek -> AJCC staging support
- Classical ML -> prognosis prediction

## What The Article Did

The article combined TCGA-Reports pathology text with TCGA Pan-Cancer Clinical Data Resource labels. The final merged dataset had 9,523 samples, 32 cancer types, and an 80:10:10 train/validation/test split stratified by cancer type.

The paper evaluated GPT, Mistral, and Llama models on:

- cancer type identification: mostly direct information extraction
- AJCC stage identification: harder reasoning over pathology evidence
- prognosis assessment: binary classification using disease-specific survival thresholds

The strongest results came from instruction-tuned models. The paper reports Path-GPT-4o-mini-FT and Path-llama3.1-8B outperforming the base models on staging and prognosis, with the 8B LoRA model trained for 6,000 steps using rank 16, alpha 16, 4-bit quantization, max length 4,096, learning rate 3e-4, and adamw_8bit.

Useful links:

- Article DOI: https://doi.org/10.1038/s41598-025-10709-4
- Article code: https://github.com/rachitsaluja/PathRep-Bench
- Article data: https://huggingface.co/datasets/rosenthal/tcga-path-notes
- DeepSeek API docs: https://api-docs.deepseek.com/

## Study Contributions

1. First independent benchmark of DeepSeek on the PathRep-Bench pathology tasks.
2. Evidence that DeepSeek matches leading proprietary models for cancer-type identification.
3. Evidence that DeepSeek provides strong AJCC stage prediction using reasoning-mode prompting.
4. Evidence that prognosis prediction remains challenging for LLM prompting.
5. A hybrid architecture recommendation: DeepSeek for extraction and staging support, and classical ML for prognosis.

## Why DeepSeek Is Feasible

DeepSeek's API is OpenAI-compatible and supports strict JSON output. Current docs list `deepseek-v4-flash` and `deepseek-v4-pro`, both with thinking and non-thinking modes, 1M context length, and JSON output support. The runner uses:

- non-thinking mode for cancer type identification
- thinking mode for AJCC stage and prognosis
- `response_format={"type": "json_object"}` for parseable predictions

## Setup

Run the commands from the project root:

```powershell
cd "C:\Users\Mohamed KONE\Desktop\project\DeepSeekCancer"
python -m pip install -r .\deepseek_pathrep\requirements.txt
$env:DEEPSEEK_API_KEY = "your_deepseek_api_key"
```

Prepare the public PathRep-Bench CSV files:

```powershell
python .\deepseek_pathrep\prepare_pathrep_data.py
```

This creates:

- `.\data\tcga_pathology_train.csv`
- `.\data\tcga_pathology_val.csv`
- `.\data\tcga_pathology_test.csv`

Your input can be CSV or JSONL. At minimum, include a pathology report column named one of:

- `report_text`
- `pathology_report`
- `pathology_text`
- `report`
- `note_text`
- `text`

For metrics, include the relevant gold label column:

- cancer type: `cancer_type`
- AJCC stage: `ajcc_stage`, `stage`, or `pathologic_stage`
- prognosis: `prognosis`, `dss_label`, or `survival_label`

For prognosis, also include `mean_dss` if available.

## Example Runs

Cancer type:

```powershell
cd "C:\Users\Mohamed KONE\Desktop\project\DeepSeekCancer"
python .\deepseek_pathrep\run_deepseek_pathology.py `
  --input .\data\tcga_pathology_test.csv `
  --task cancer_type `
  --output .\outputs\deepseek_cancer_type.jsonl
```

AJCC stage:

```powershell
cd "C:\Users\Mohamed KONE\Desktop\project\DeepSeekCancer"
python .\deepseek_pathrep\run_deepseek_pathology.py `
  --input .\data\tcga_pathology_test.csv `
  --task ajcc_stage `
  --output .\outputs\deepseek_ajcc_stage.jsonl `
  --thinking enabled `
  --reasoning-effort high
```

Prognosis:

```powershell
cd "C:\Users\Mohamed KONE\Desktop\project\DeepSeekCancer"
python .\deepseek_pathrep\run_deepseek_pathology.py `
  --input .\data\tcga_pathology_test.csv `
  --task prognosis `
  --output .\outputs\deepseek_prognosis.jsonl `
  --thinking enabled `
  --reasoning-effort high
```

Optional few-shot prognosis examples:

```powershell
python .\deepseek_pathrep\make_prognosis_examples.py

python .\deepseek_pathrep\run_deepseek_pathology.py `
  --input .\data\tcga_pathology_test.csv `
  --task prognosis `
  --output .\outputs\deepseek_prognosis_100_fewshot8_max4096.jsonl `
  --limit 100 `
  --examples .\data\prognosis_examples_8.jsonl `
  --examples-per-call 8 `
  --max-tokens 4096
```

To do a cheap smoke test first, add `--limit 20`.

If you are already inside `C:\Users\Mohamed KONE\Desktop\project\DeepSeekCancer\deepseek_pathrep`, call the script directly:

```powershell
python .\run_deepseek_pathology.py `
  --input ..\data\tcga_pathology_test.csv `
  --task ajcc_stage `
  --output ..\outputs\deepseek_ajcc_stage.jsonl `
  --limit 20
```

If the data files are missing from that location, run this first:

```powershell
python .\prepare_pathrep_data.py
```

Each run writes:

- predictions to the requested `.jsonl` file
- metrics to a sidecar `.metrics.json` file

## Recommended DeepSeek Study Design

Run at least these conditions:

1. `deepseek-v4-flash`, cancer type, thinking disabled.
2. `deepseek-v4-flash`, AJCC stage, thinking enabled.
3. `deepseek-v4-pro`, AJCC stage, thinking enabled.
4. `deepseek-v4-flash`, prognosis, thinking enabled, with mean DSS.
5. Optional few-shot prognosis: pass training examples with `--examples` and `--examples-per-call 8`.

Report accuracy and macro F1 for all tasks. For prognosis, macro F1 is more important than accuracy because the labels may be imbalanced.

## Better Alternative Method

For prognosis, do not rely only on prompting. The current project results already support a hybrid pipeline:

1. Use DeepSeek to extract structured variables: cancer type, tumor size, lymph node involvement, margin status, metastasis evidence, grade, invasion, and key pathology phrases.
2. Use DeepSeek reasoning-mode prompting for high-level AJCC staging, optionally later validated with deterministic AJCC rules or a retrieval layer.
3. Train a local prognosis model, such as TF-IDF logistic regression, Cox proportional hazards, random survival forests, or XGBoost-style classifiers over extracted variables.
4. Compare against the pure LLM pipeline.

That gives the article a clearer scientific contribution: LLMs are excellent language-understanding components, but prognosis is better treated as a supervised prediction problem.

## Reviewer-Oriented Additions

The project includes utilities for the next four experiments:

- `cost_analysis.py`: computes cost per report, cost per 1000 reports, and cost per correct prediction.
- `make_error_tables.py`: creates tables for rectum/colon, kidney subtype, and Stage III/IV confusions.
- `extract_deepseek_variables.py`: extracts structured pathology variables for DeepSeek-assisted prognosis modeling, with resumable output for long API runs.
- `run_prognosis_ml_baseline.py`: can train prognosis models with or without DeepSeek-extracted variables and can run ablations without text or without base structured fields.
- `compare_prognosis_models.py`: compares prognosis model metrics against the current 0.8543 macro-F1 text-only baseline.

See `additional_experiments.md` for exact commands.

Final prognosis result:

Although the TF-IDF text-only model achieved the highest prognosis point estimate, adding DeepSeek-extracted variables did not produce a statistically significant improvement. The TF-IDF text-only model achieved 0.8571 accuracy and 0.8543 macro F1, suggesting that the raw pathology report text already captures much of the prognostic information available in this benchmark dataset under the evaluated modeling framework.
