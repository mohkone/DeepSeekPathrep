# DeepSeekPathrep

This repository contains code and manuscript files for:

**DeepSeek for pathology report understanding: strengths and limitations across extraction, staging, and prognosis tasks**

The project benchmarks DeepSeek on PathRep-Bench style TCGA pathology report tasks:

- cancer type identification
- high-level AJCC stage prediction
- prognosis classification
- DeepSeek-assisted structured variable extraction for supervised prognosis modeling

## Repository Layout

- `deepseek_pathrep/`: Python scripts, prompts, result summaries, statistical analysis, and manuscript draft.
- `sn-article-template/`: BMC/Springer Nature LaTeX manuscript source, bibliography, class file, bibliography styles, and submission figures.

Raw benchmark CSV files and raw model-output JSONL files are intentionally not tracked. The data can be prepared from the public PathRep-Bench repository:

```powershell
python .\deepseek_pathrep\prepare_pathrep_data.py
```

## Manuscript

Main BMC-formatted LaTeX source:

```text
sn-article-template/deepseek_pathrep_bmc.tex
```

Figures are stored in:

```text
sn-article-template/figures/
```

## Reproducibility

Set a DeepSeek API key through the environment before running API experiments:

```powershell
$env:DEEPSEEK_API_KEY = "your_deepseek_api_key"
```

See `deepseek_pathrep/README.md` for detailed commands and experimental notes.
