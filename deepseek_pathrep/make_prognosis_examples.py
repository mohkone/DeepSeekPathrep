#!/usr/bin/env python
"""Build a compact balanced few-shot example file for prognosis prompting."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


IMPORTANT_TERMS = [
    "stage",
    "grade",
    "metasta",
    "lymph",
    "node",
    "margin",
    "invasion",
    "necrosis",
    "differentiated",
    "tumor",
]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def summarize_report(text: str, max_chars: int) -> str:
    text = clean_text(text)
    snippets = []
    for term in IMPORTANT_TERMS:
        match = re.search(term, text, flags=re.I)
        if not match:
            continue
        start = max(0, match.start() - 160)
        end = min(len(text), match.end() + 260)
        snippets.append(text[start:end])
        if len(" ".join(snippets)) >= max_chars:
            break
    prefix = text[:350]
    summary = clean_text(prefix + " " + " ".join(snippets)) or text
    return summary[:max_chars].rstrip()


def select_examples(rows: list[dict], per_label: int, max_summary_chars: int) -> list[dict]:
    examples = []
    for label in ["good", "poor"]:
        candidates = [
            row
            for row in rows
            if clean_text(row.get("prognosis", "")).lower() == label
            and row.get("report_text")
            and row.get("mean_dss")
        ]
        scored = []
        for row in candidates:
            summary = summarize_report(row.get("report_text", ""), max_summary_chars)
            score = 0
            score += 2 if row.get("ajcc_stage") else 0
            score += 1 if len(summary) >= 450 else 0
            score += 1 if any(term in summary.lower() for term in ["lymph", "metasta", "margin", "invasion"]) else 0
            scored.append((row.get("cancer_type", ""), -score, row.get("bcr_patient_barcode", ""), row))
        scored.sort(key=lambda item: (item[1], item[0], item[2]))

        selected = []
        seen_types = set()
        for cancer_type, _score, _barcode, row in scored:
            if cancer_type in seen_types:
                continue
            selected.append(row)
            seen_types.add(cancer_type)
            if len(selected) == per_label:
                break
        if len(selected) < per_label:
            for _cancer_type, _score, _barcode, row in scored:
                if row in selected:
                    continue
                selected.append(row)
                if len(selected) == per_label:
                    break
        examples.extend(selected)
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "tcga_pathology_train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "prognosis_examples_8.jsonl",
    )
    parser.add_argument("--per-label", type=int, default=4)
    parser.add_argument("--max-summary-chars", type=int, default=900)
    args = parser.parse_args()

    rows = read_rows(args.input)
    examples = select_examples(rows, args.per_label, args.max_summary_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in examples:
            summary = (
                f"Cancer type: {row.get('cancer_type')}. "
                f"AJCC stage: {row.get('ajcc_stage') or 'not provided'}. "
                f"Mean DSS threshold: {row.get('mean_dss')}. "
                f"Report evidence: {summarize_report(row.get('report_text', ''), args.max_summary_chars)}"
            )
            output = {
                "label": clean_text(row.get("prognosis", "")).lower(),
                "summary": summary,
                "cancer_type": row.get("cancer_type"),
                "id": row.get("bcr_patient_barcode"),
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
