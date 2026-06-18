# DeepSeekPathrep

This repository contains code and reproducible analysis files for:

**DeepSeek for pathology report understanding: strengths and limitations across extraction, staging, and prognosis tasks**

The project benchmarks DeepSeek on PathRep-Bench style TCGA pathology report tasks:

- cancer type identification
- high-level AJCC stage prediction
- prognosis classification
- DeepSeek-assisted structured variable extraction for supervised prognosis modeling

## Repository Layout

- `deepseek_pathrep/`: Python scripts, prompts, result summaries, statistical analysis, and manuscript draft.

The BMC/Springer Nature LaTeX submission folder is maintained locally and is not tracked in this repository.

## Archive

GitHub repository:

```text
https://github.com/mohkone/DeepSeekPathrep
```

Zenodo archive:

```text
https://doi.org/10.5281/zenodo.20753000
```

Raw benchmark CSV files and raw model-output JSONL files are intentionally not tracked. The data can be prepared from the public PathRep-Bench repository:

```powershell
python .\deepseek_pathrep\prepare_pathrep_data.py
```

## Manuscript

Main Markdown manuscript draft:

```text
deepseek_pathrep/manuscript_draft.md
```

## Reproducibility

Set a DeepSeek API key through the environment before running API experiments:

```powershell
$env:DEEPSEEK_API_KEY = "your_deepseek_api_key"
```

See `deepseek_pathrep/README.md` for detailed commands and experimental notes.
