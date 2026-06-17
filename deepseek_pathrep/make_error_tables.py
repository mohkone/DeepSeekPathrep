#!/usr/bin/env python
"""Create clinically focused error-analysis tables from prediction JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_data(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def stage_pair(row: dict) -> tuple[str, str]:
    return row.get("gold") or "", row.get("prediction") or ""


def cancer_pair(row: dict) -> tuple[str, str]:
    return row.get("gold") or "", row.get("prediction") or ""


def source(data_rows: list[dict], row: dict) -> dict:
    idx = int(row["index"])
    return data_rows[idx] if idx < len(data_rows) else {}


def row_id(data_rows: list[dict], row: dict) -> str:
    return row.get("id") or source(data_rows, row).get("bcr_patient_barcode") or ""


def category_examples(rows: list[dict], data_rows: list[dict], limit: int) -> list[str]:
    lines = [
        "| Index | ID | Cancer type | Raw stage | Gold | Predicted |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows[:limit]:
        src = source(data_rows, row)
        lines.append(
            f"| {row['index']} | {row_id(data_rows, row)} | {src.get('type_name', '')} | "
            f"{src.get('stage', '')} | {row.get('gold', '')} | {row.get('prediction', '')} |"
        )
    return lines


def count_table(title: str, rows: list[dict]) -> list[str]:
    counts = Counter((row.get("gold") or "", row.get("prediction") or "") for row in rows)
    lines = [f"## {title}", "", "| Gold | Predicted | Count |", "| --- | --- | ---: |"]
    for (gold, pred), count in sorted(counts.items()):
        lines.append(f"| {gold} | {pred} | {count} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancer-predictions", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "deepseek_cancer_type_full.jsonl")
    parser.add_argument("--stage-predictions", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "deepseek_ajcc_stage_full_v2_max4096.jsonl")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "tcga_pathology_test.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[0] / "error_analysis_tables.md")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    cancer_rows = load_jsonl(args.cancer_predictions)
    stage_rows = load_jsonl(args.stage_predictions)
    data_rows = load_data(args.data)

    colon_rectum = [
        row
        for row in cancer_rows
        if norm(row.get("gold")) in {"colon adenocarcinoma", "rectum adenocarcinoma"}
        and norm(row.get("prediction")) in {"colon adenocarcinoma", "rectum adenocarcinoma"}
        and norm(row.get("gold")) != norm(row.get("prediction"))
    ]
    kidney_subtypes = [
        row
        for row in cancer_rows
        if norm(row.get("gold")) in {
            "kidney chromophobe",
            "kidney renal clear cell carcinoma",
            "kidney renal papillary cell carcinoma",
        }
        and norm(row.get("prediction")) in {
            "kidney chromophobe",
            "kidney renal clear cell carcinoma",
            "kidney renal papillary cell carcinoma",
        }
        and norm(row.get("gold")) != norm(row.get("prediction"))
    ]
    stage_iii_iv = [
        row
        for row in stage_rows
        if norm(row.get("gold")) in {"stage iii", "stage iv"}
        and norm(row.get("prediction")) in {"stage iii", "stage iv"}
        and norm(row.get("gold")) != norm(row.get("prediction"))
    ]

    sections = [
        "# Clinically Plausible Error Tables",
        "",
        "These tables summarize recurring errors that are useful for the manuscript discussion.",
        "",
    ]
    for title, rows in [
        ("Rectum vs Colon Confusion", colon_rectum),
        ("Kidney RCC Subtype Confusion", kidney_subtypes),
        ("Stage III vs Stage IV Confusion", stage_iii_iv),
    ]:
        sections.extend(count_table(title, rows))
        sections.append("")
        sections.extend(category_examples(rows, data_rows, args.examples))
        sections.append("")

    args.output.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
