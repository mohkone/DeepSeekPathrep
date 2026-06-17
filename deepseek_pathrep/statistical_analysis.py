#!/usr/bin/env python
"""Bootstrap confidence intervals, paired tests, and result figures."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "outputs"


RUNS = [
    {
        "key": "cancer_flash",
        "task": "Cancer type",
        "method": "DeepSeek V4 Flash",
        "path": "deepseek_cancer_type_full.jsonl",
    },
    {
        "key": "cancer_pro",
        "task": "Cancer type",
        "method": "DeepSeek V4 Pro",
        "path": "deepseek_v4_pro_cancer_type_full.jsonl",
    },
    {
        "key": "stage_flash",
        "task": "AJCC stage",
        "method": "DeepSeek V4 Flash",
        "path": "deepseek_ajcc_stage_full_v2_max4096.jsonl",
    },
    {
        "key": "stage_pro",
        "task": "AJCC stage",
        "method": "DeepSeek V4 Pro",
        "path": "deepseek_v4_pro_ajcc_stage_full_max4096.jsonl",
    },
    {
        "key": "prog_zero",
        "task": "Prognosis",
        "method": "DeepSeek zero-shot",
        "path": "deepseek_prognosis_100_max4096.jsonl",
    },
    {
        "key": "prog_fewshot",
        "task": "Prognosis",
        "method": "DeepSeek 8-shot",
        "path": "deepseek_prognosis_100_fewshot8_max4096.jsonl",
    },
    {
        "key": "prog_tfidf_text",
        "task": "Prognosis",
        "method": "TF-IDF text-only logistic regression",
        "path": "ml_prognosis_text_only_test.jsonl",
    },
    {
        "key": "prog_deepseek_assisted",
        "task": "Prognosis",
        "method": "TF-IDF + structured fields + DeepSeek variables",
        "path": "ml_prognosis_deepseek_assisted_test.jsonl",
    },
]


COMPARISONS = [
    {
        "name": "Cancer type: Flash vs Pro",
        "a": "cancer_flash",
        "b": "cancer_pro",
        "interpretation": "Positive delta favors Flash.",
    },
    {
        "name": "AJCC stage: Flash vs Pro",
        "a": "stage_flash",
        "b": "stage_pro",
        "interpretation": "Positive delta favors Flash.",
    },
    {
        "name": "Prognosis: TF-IDF text-only vs DeepSeek-assisted ML",
        "a": "prog_tfidf_text",
        "b": "prog_deepseek_assisted",
        "interpretation": "Positive delta favors TF-IDF text-only.",
    },
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ivx]+", " ", text).strip()
    stage_map = {
        "i": "stage i",
        "1": "stage i",
        "stage 1": "stage i",
        "stage i": "stage i",
        "ii": "stage ii",
        "2": "stage ii",
        "stage 2": "stage ii",
        "stage ii": "stage ii",
        "iii": "stage iii",
        "3": "stage iii",
        "stage 3": "stage iii",
        "stage iii": "stage iii",
        "iv": "stage iv",
        "4": "stage iv",
        "stage 4": "stage iv",
        "stage iv": "stage iv",
    }
    prognosis_map = {
        "good prognosis": "good",
        "favorable": "good",
        "positive": "good",
        "survive": "good",
        "poor prognosis": "poor",
        "bad": "poor",
        "unfavorable": "poor",
        "negative": "poor",
    }
    return stage_map.get(text, prognosis_map.get(text, text))


def accuracy(golds: list[str], preds: list[str]) -> float:
    if not golds:
        return 0.0
    return sum(g == p for g, p in zip(golds, preds)) / len(golds)


def macro_f1(golds: list[str], preds: list[str]) -> float:
    labels = sorted(set(golds) | set(preds))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def bootstrap_ci(
    golds: list[str],
    preds: list[str],
    metric_fn,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(golds)
    values = []
    for _ in range(iterations):
        sample_golds = []
        sample_preds = []
        for _ in range(n):
            index = rng.randrange(n)
            sample_golds.append(golds[index])
            sample_preds.append(preds[index])
        values.append(metric_fn(sample_golds, sample_preds))
    return percentile(values, 0.025), percentile(values, 0.975)


def bootstrap_delta_ci(
    golds: list[str],
    preds_a: list[str],
    preds_b: list[str],
    metric_fn,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(golds)
    values = []
    for _ in range(iterations):
        sample_golds = []
        sample_a = []
        sample_b = []
        for _ in range(n):
            index = rng.randrange(n)
            sample_golds.append(golds[index])
            sample_a.append(preds_a[index])
            sample_b.append(preds_b[index])
        values.append(metric_fn(sample_golds, sample_a) - metric_fn(sample_golds, sample_b))
    lower = percentile(values, 0.025)
    upper = percentile(values, 0.975)
    less_or_equal_zero = sum(value <= 0 for value in values)
    greater_or_equal_zero = sum(value >= 0 for value in values)
    p_value = 2 * min(less_or_equal_zero, greater_or_equal_zero) / iterations
    return lower, upper, min(1.0, p_value)


def exact_mcnemar(correct_a: list[bool], correct_b: list[bool]) -> dict:
    a_right_b_wrong = sum(a and not b for a, b in zip(correct_a, correct_b))
    a_wrong_b_right = sum(not a and b for a, b in zip(correct_a, correct_b))
    discordant = a_right_b_wrong + a_wrong_b_right
    if discordant == 0:
        p_value = 1.0
    else:
        try:
            from scipy.stats import binomtest

            p_value = float(binomtest(min(a_right_b_wrong, a_wrong_b_right), discordant, 0.5).pvalue)
        except Exception:
            p_value = exact_binomial_two_sided(min(a_right_b_wrong, a_wrong_b_right), discordant)
    return {
        "a_correct_b_wrong": a_right_b_wrong,
        "a_wrong_b_correct": a_wrong_b_right,
        "discordant": discordant,
        "p_value": p_value,
    }


def exact_binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    log_two = math.log(2)
    total = 0.0
    for i in range(0, k + 1):
        log_prob = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * log_two
        total += math.exp(log_prob)
    return min(1.0, 2 * total)


def paired_records(rows_a: list[dict], rows_b: list[dict]) -> tuple[list[str], list[str], list[str]]:
    by_index_b = {row.get("index"): row for row in rows_b}
    golds = []
    preds_a = []
    preds_b = []
    for row_a in rows_a:
        index = row_a.get("index")
        row_b = by_index_b.get(index)
        if row_b is None:
            continue
        gold_a = normalize(row_a.get("gold"))
        gold_b = normalize(row_b.get("gold"))
        if gold_a != gold_b:
            raise ValueError(f"Gold label mismatch at index {index}: {gold_a!r} vs {gold_b!r}")
        golds.append(gold_a)
        preds_a.append(normalize(row_a.get("prediction")))
        preds_b.append(normalize(row_b.get("prediction")))
    return golds, preds_a, preds_b


def summarize_run(run: dict, rows: list[dict], iterations: int, seed: int) -> dict:
    golds = [normalize(row.get("gold")) for row in rows]
    preds = [normalize(row.get("prediction")) for row in rows]
    acc = accuracy(golds, preds)
    f1 = macro_f1(golds, preds)
    acc_low, acc_high = bootstrap_ci(golds, preds, accuracy, iterations, seed)
    f1_low, f1_high = bootstrap_ci(golds, preds, macro_f1, iterations, seed + 101)
    return {
        "key": run["key"],
        "task": run["task"],
        "method": run["method"],
        "n": len(rows),
        "accuracy": acc,
        "accuracy_ci_low": acc_low,
        "accuracy_ci_high": acc_high,
        "macro_f1": f1,
        "macro_f1_ci_low": f1_low,
        "macro_f1_ci_high": f1_high,
        "model_errors": sum(1 for row in rows if row.get("error")),
    }


def summarize_comparison(comparison: dict, rows_by_key: dict, iterations: int, seed: int) -> dict:
    rows_a = rows_by_key[comparison["a"]]
    rows_b = rows_by_key[comparison["b"]]
    golds, preds_a, preds_b = paired_records(rows_a, rows_b)
    correct_a = [g == p for g, p in zip(golds, preds_a)]
    correct_b = [g == p for g, p in zip(golds, preds_b)]
    acc_delta = accuracy(golds, preds_a) - accuracy(golds, preds_b)
    f1_delta = macro_f1(golds, preds_a) - macro_f1(golds, preds_b)
    acc_low, acc_high, acc_boot_p = bootstrap_delta_ci(
        golds, preds_a, preds_b, accuracy, iterations, seed
    )
    f1_low, f1_high, f1_boot_p = bootstrap_delta_ci(
        golds, preds_a, preds_b, macro_f1, iterations, seed + 211
    )
    return {
        "name": comparison["name"],
        "a": comparison["a"],
        "b": comparison["b"],
        "n": len(golds),
        "accuracy_delta": acc_delta,
        "accuracy_delta_ci_low": acc_low,
        "accuracy_delta_ci_high": acc_high,
        "accuracy_bootstrap_p": acc_boot_p,
        "macro_f1_delta": f1_delta,
        "macro_f1_delta_ci_low": f1_low,
        "macro_f1_delta_ci_high": f1_high,
        "macro_f1_bootstrap_p": f1_boot_p,
        "mcnemar": exact_mcnemar(correct_a, correct_b),
        "interpretation": comparison["interpretation"],
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_p(value: float) -> str:
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def write_markdown(path: Path, run_summaries: list[dict], comparisons: list[dict], figure_path: Path | None) -> None:
    lines = [
        "# Statistical Analysis",
        "",
        "Confidence intervals were estimated with paired nonparametric bootstrap resampling over reports.",
        "Macro F1 confidence intervals use the same bootstrap procedure as accuracy confidence intervals.",
        "Paired model comparisons use exact McNemar tests for accuracy and bootstrap differences for macro F1.",
        "",
        "## Metric Confidence Intervals",
        "",
        "| Task | Method | n | Accuracy (95% CI) | Macro F1 (95% CI) | Errors |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for item in run_summaries:
        lines.append(
            "| {task} | {method} | {n} | {acc} ({acc_low}-{acc_high}) | {f1} ({f1_low}-{f1_high}) | {errors} |".format(
                task=item["task"],
                method=item["method"],
                n=item["n"],
                acc=fmt(item["accuracy"]),
                acc_low=fmt(item["accuracy_ci_low"]),
                acc_high=fmt(item["accuracy_ci_high"]),
                f1=fmt(item["macro_f1"]),
                f1_low=fmt(item["macro_f1_ci_low"]),
                f1_high=fmt(item["macro_f1_ci_high"]),
                errors=item["model_errors"],
            )
        )
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "| Comparison | n | Delta accuracy | McNemar b/c | McNemar p | Delta macro F1 (95% CI) | Bootstrap p |",
            "| --- | ---: | ---: | --- | ---: | --- | ---: |",
        ]
    )
    for item in comparisons:
        mcnemar = item["mcnemar"]
        lines.append(
            "| {name} | {n} | {acc_delta} | {bc} | {mcnemar_p} | {f1_delta} ({f1_low}-{f1_high}) | {boot_p} |".format(
                name=item["name"],
                n=item["n"],
                acc_delta=fmt(item["accuracy_delta"]),
                bc=f'{mcnemar["a_correct_b_wrong"]}/{mcnemar["a_wrong_b_correct"]}',
                mcnemar_p=fmt_p(mcnemar["p_value"]),
                f1_delta=fmt(item["macro_f1_delta"]),
                f1_low=fmt(item["macro_f1_delta_ci_low"]),
                f1_high=fmt(item["macro_f1_delta_ci_high"]),
                boot_p=fmt_p(item["macro_f1_bootstrap_p"]),
            )
        )
    if figure_path:
        lines.extend(["", "## Figure", "", f"![Macro F1 results]({figure_path.as_posix()})"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, run_summaries: list[dict]) -> None:
    import matplotlib.pyplot as plt

    selected = [
        item
        for item in run_summaries
        if item["key"]
        in {
            "cancer_flash",
            "cancer_pro",
            "stage_flash",
            "stage_pro",
            "prog_zero",
            "prog_fewshot",
            "prog_tfidf_text",
            "prog_deepseek_assisted",
        }
    ]
    labels = [f'{item["task"]}\n{item["method"]}' for item in selected]
    values = [item["macro_f1"] for item in selected]
    lows = [item["macro_f1"] - item["macro_f1_ci_low"] for item in selected]
    highs = [item["macro_f1_ci_high"] - item["macro_f1"] for item in selected]
    colors = []
    for item in selected:
        if item["task"] == "Cancer type":
            colors.append("#2f6f9f")
        elif item["task"] == "AJCC stage":
            colors.append("#40916c")
        else:
            colors.append("#b56576")

    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = range(len(selected))
    ax.bar(x, values, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.errorbar(x, values, yerr=[lows, highs], fmt="none", ecolor="#1f1f1f", capsize=4, linewidth=1)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("DeepSeek and Supervised Baselines Across Pathology Report Tasks")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for index, value in enumerate(values):
        ax.text(index, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--markdown", type=Path, default=Path(__file__).resolve().parent / "statistical_analysis.md")
    parser.add_argument("--json", type=Path, default=Path(__file__).resolve().parent / "statistical_analysis.json")
    parser.add_argument("--figure", type=Path, default=Path(__file__).resolve().parent / "figures" / "macro_f1_results.png")
    args = parser.parse_args()

    rows_by_key = {}
    run_summaries = []
    for run in RUNS:
        rows = read_jsonl(args.outputs_dir / run["path"])
        rows_by_key[run["key"]] = rows
        run_summaries.append(summarize_run(run, rows, args.iterations, args.seed))

    comparison_summaries = [
        summarize_comparison(comparison, rows_by_key, args.iterations, args.seed)
        for comparison in COMPARISONS
    ]

    result = {
        "bootstrap_iterations": args.iterations,
        "seed": args.seed,
        "runs": run_summaries,
        "comparisons": comparison_summaries,
    }
    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure_path = None
    try:
        write_figure(args.figure, run_summaries)
        figure_path = args.figure
    except Exception as exc:
        print(f"warning: figure generation failed: {exc}")

    write_markdown(args.markdown, run_summaries, comparison_summaries, figure_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
