#!/usr/bin/env python
"""Compute DeepSeek benchmark costs from metrics JSON files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


OFFICIAL_DEEPSEEK_PRICES_PER_1M = {
    "deepseek-v4-flash": {
        "input_cache_hit": 0.0028,
        "input_cache_miss": 0.14,
        "output": 0.28,
    },
    "deepseek-v4-pro": {
        "input_cache_hit": 0.003625,
        "input_cache_miss": 0.435,
        "output": 0.87,
    },
}


def expand_paths(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def infer_model(metrics: dict, default_model: str) -> str:
    return metrics.get("model") or default_model


def report_count(metrics: dict) -> int:
    return int(metrics.get("processed_n") or metrics.get("n") or metrics.get("test_rows") or metrics.get("input_n") or 0)


def input_token_split(usage: dict) -> tuple[float, float]:
    hit = float(usage.get("prompt_cache_hit_tokens") or usage.get("cache_hit_tokens") or 0)
    miss = float(usage.get("prompt_cache_miss_tokens") or usage.get("cache_miss_tokens") or 0)
    prompt = float(usage.get("prompt_tokens") or 0)
    if hit or miss:
        return hit, miss
    return 0.0, prompt


def calculate_cost(metrics: dict, model: str) -> dict:
    usage = metrics.get("usage") or {}
    prices = OFFICIAL_DEEPSEEK_PRICES_PER_1M[model]
    hit_tokens, miss_tokens = input_token_split(usage)
    output_tokens = float(usage.get("completion_tokens") or 0)
    total_cost = (
        hit_tokens * prices["input_cache_hit"]
        + miss_tokens * prices["input_cache_miss"]
        + output_tokens * prices["output"]
    ) / 1_000_000
    reports = report_count(metrics)
    correct = metrics.get("correct")
    return {
        "model": model,
        "reports": reports,
        "correct": correct,
        "input_cache_hit_tokens": int(hit_tokens),
        "input_cache_miss_tokens": int(miss_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(usage.get("total_tokens") or hit_tokens + miss_tokens + output_tokens),
        "estimated_cost_usd": total_cost,
        "cost_per_report_usd": total_cost / reports if reports else None,
        "cost_per_1000_reports_usd": total_cost * 1000 / reports if reports else None,
        "cost_per_correct_usd": total_cost / correct if correct else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", help="Metrics JSON paths or glob patterns.")
    parser.add_argument("--default-model", default="deepseek-v4-flash", choices=sorted(OFFICIAL_DEEPSEEK_PRICES_PER_1M))
    parser.add_argument("--markdown", type=Path, help="Optional Markdown output path.")
    args = parser.parse_args()

    rows = []
    for path in expand_paths(args.metrics):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        model = infer_model(metrics, args.default_model)
        if model not in OFFICIAL_DEEPSEEK_PRICES_PER_1M:
            raise ValueError(f"No default price for model {model!r}; pass a supported model or edit the price table.")
        row = {"file": str(path)}
        row.update(calculate_cost(metrics, model))
        rows.append(row)

    output = {
        "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing",
        "pricing_note": "If cache-hit token counts are absent, all prompt tokens are priced as cache misses.",
        "prices_per_1m_tokens": OFFICIAL_DEEPSEEK_PRICES_PER_1M,
        "runs": rows,
    }
    print(json.dumps(output, indent=2))

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Cost Analysis",
            "",
            "Pricing source: https://api-docs.deepseek.com/quick_start/pricing",
            "",
            "If cache-hit token counts are absent, all prompt tokens are priced as cache misses.",
            "",
            "| Run | Model | Reports | Correct | Tokens | Cost USD | Cost/report | Cost/1000 reports | Cost/correct |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                "| {file} | {model} | {reports} | {correct} | {total_tokens} | ${cost:.6f} | ${cpr:.6f} | ${cpk:.4f} | {cpc} |".format(
                    file=Path(row["file"]).name,
                    model=row["model"],
                    reports=row["reports"],
                    correct="" if row["correct"] is None else row["correct"],
                    total_tokens=row["total_tokens"],
                    cost=row["estimated_cost_usd"],
                    cpr=row["cost_per_report_usd"] or 0,
                    cpk=row["cost_per_1000_reports_usd"] or 0,
                    cpc="" if row["cost_per_correct_usd"] is None else f"${row['cost_per_correct_usd']:.6f}",
                )
            )
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
