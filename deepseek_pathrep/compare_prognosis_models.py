#!/usr/bin/env python
"""Compare prognosis model metric JSON files against a baseline macro F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--baseline-macro-f1", type=float, default=0.8501468399062744)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.metrics:
        metrics = load_metrics(path)
        macro_f1 = metric_value(metrics, "macro_f1")
        accuracy = metric_value(metrics, "accuracy")
        rows.append(
            {
                "file": path.name,
                "model": metrics.get("model", path.stem),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "delta_macro_f1": None if macro_f1 is None else macro_f1 - args.baseline_macro_f1,
                "train_rows": metrics.get("train_rows"),
                "test_rows": metrics.get("test_rows"),
                "deepseek_features_train_rows": metrics.get("deepseek_features_train_rows"),
                "deepseek_features_test_rows": metrics.get("deepseek_features_test_rows"),
                "warning": metrics.get("deepseek_feature_warning"),
            }
        )

    output = {
        "baseline_macro_f1": args.baseline_macro_f1,
        "runs": rows,
        "best": max(
            (row for row in rows if row["macro_f1"] is not None),
            key=lambda row: row["macro_f1"],
            default=None,
        ),
    }
    print(json.dumps(output, indent=2))

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Prognosis Model Comparison",
            "",
            f"Baseline macro F1: `{args.baseline_macro_f1:.4f}`",
            "",
            "| Run | Model | Accuracy | Macro F1 | Delta vs baseline | Train rows | Test rows | DeepSeek train features | DeepSeek test features | Warning |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                "| {file} | {model} | {acc} | {f1} | {delta} | {train} | {test} | {ds_train} | {ds_test} | {warning} |".format(
                    file=row["file"],
                    model=row["model"],
                    acc="" if row["accuracy"] is None else f"{row['accuracy']:.4f}",
                    f1="" if row["macro_f1"] is None else f"{row['macro_f1']:.4f}",
                    delta="" if row["delta_macro_f1"] is None else f"{row['delta_macro_f1']:+.4f}",
                    train=row["train_rows"] or "",
                    test=row["test_rows"] or "",
                    ds_train=row["deepseek_features_train_rows"] or "",
                    ds_test=row["deepseek_features_test_rows"] or "",
                    warning=row["warning"] or "",
                )
            )
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
