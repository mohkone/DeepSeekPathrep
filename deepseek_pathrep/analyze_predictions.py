#!/usr/bin/env python
"""Summarize DeepSeek pathology JSONL predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_data(path: Path | None) -> list[dict]:
    if not path:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize(value) -> str:
    return "" if value is None else str(value).strip()


def normalize_for_match(value) -> str:
    text = normalize(value).lower().replace("_", " ")
    return " ".join(text.split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--data", type=Path, help="Optional source CSV for cancer types and raw stage labels.")
    parser.add_argument("--show-errors", type=int, default=20)
    args = parser.parse_args()

    rows = load_jsonl(args.predictions)
    data_rows = load_data(args.data)
    metrics_path = args.predictions.with_suffix(args.predictions.suffix + ".metrics.json")
    if metrics_path.exists():
        print(metrics_path.read_text(encoding="utf-8").strip())
        print()

    confusion = Counter()
    by_type = Counter()
    errors = []
    error_count = 0

    for row in rows:
        if row.get("error"):
            error_count += 1
        gold = normalize(row.get("gold"))
        pred = normalize(row.get("prediction"))
        if not gold:
            continue
        confusion[(gold, pred)] += 1
        if normalize_for_match(gold) != normalize_for_match(pred):
            source = data_rows[row["index"]] if data_rows and row["index"] < len(data_rows) else {}
            cancer_type = source.get("type_name") or source.get("cancer_type") or ""
            by_type[cancer_type] += 1
            errors.append(
                {
                    "index": row["index"],
                    "id": row.get("id") or source.get("bcr_patient_barcode"),
                    "cancer_type": cancer_type,
                    "raw_stage": source.get("stage", ""),
                    "gold": gold,
                    "prediction": pred,
                    "error": row.get("error"),
                }
            )

    print(f"rows: {len(rows)}")
    print(f"model_output_errors: {error_count}")
    print(f"labeled_rows: {sum(confusion.values())}")

    print("\nConfusion:")
    for (gold, pred), count in sorted(confusion.items()):
        marker = "OK" if normalize_for_match(gold) == normalize_for_match(pred) else "MISS"
        print(f"{marker:4} gold={gold:9} pred={pred or '<blank>':9} n={count}")

    if by_type:
        print("\nErrors by cancer type:")
        for cancer_type, count in by_type.most_common():
            print(f"{count:3} {cancer_type}")

    if errors and args.show_errors:
        print("\nErrors:")
        for error in errors[: args.show_errors]:
            print(
                f"idx={error['index']} id={error['id']} type={error['cancer_type']} "
                f"raw_stage={error['raw_stage']} gold={error['gold']} "
                f"pred={error['prediction']} error={error['error']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
