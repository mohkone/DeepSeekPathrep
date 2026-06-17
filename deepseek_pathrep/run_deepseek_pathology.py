#!/usr/bin/env python
"""Run DeepSeek on pathology-report extraction tasks and write JSONL predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from prompts import PROMPT_VERSION, build_messages


REPORT_COLUMNS = [
    "report_text",
    "pathology_report",
    "pathology_text",
    "report",
    "note_text",
    "text",
]

LABEL_COLUMNS = {
    "cancer_type": ["cancer_type", "type_name", "label", "answer", "diagnosis"],
    "ajcc_stage": ["ajcc_stage", "stage_overall", "stage", "pathologic_stage", "label", "answer"],
    "prognosis": ["prognosis", "dss_label", "survival_label", "label", "answer"],
}

PREDICTION_KEYS = {
    "cancer_type": "cancer_type",
    "ajcc_stage": "ajcc_stage",
    "prognosis": "prognosis",
}


def load_table(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Create it first with:\n"
            "  python .\\deepseek_pathrep\\prepare_pathrep_data.py\n"
            "or pass --input to an existing CSV/JSONL file."
        )

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def get_first(record: dict, names: list[str], default=None):
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return default


def pick_examples(examples: list[dict], limit: int, seed: int) -> list[dict]:
    if not examples or limit <= 0:
        return []
    rng = random.Random(seed)
    if len(examples) <= limit:
        return list(examples)
    return rng.sample(examples, limit)


def parse_jsonish(text: str) -> dict:
    if not text:
        return {}
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


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


def call_deepseek(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    thinking: str,
    reasoning_effort: str,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> dict:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
    }
    if thinking in {"enabled", "disabled"}:
        payload["thinking"] = {"type": thinking}
        if thinking == "enabled":
            payload["reasoning_effort"] = reasoning_effort

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            choice = data["choices"][0]["message"]
            return {
                "content": choice.get("content") or "",
                "reasoning_content": choice.get("reasoning_content") or "",
                "usage": data.get("usage", {}),
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:1000]}"
        except Exception as exc:  # noqa: BLE001 - retry and record API errors
            last_error = str(exc)
        if attempt < retries:
            time.sleep(min(2**attempt, 8))
    return {"content": "", "reasoning_content": "", "usage": {}, "error": last_error}


def task_thinking(task: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "disabled" if task == "cancer_type" else "enabled"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV or JSONL with pathology reports.")
    parser.add_argument("--task", required=True, choices=["cancer_type", "ajcc_stage", "prognosis"])
    parser.add_argument("--output", type=Path, required=True, help="JSONL predictions path.")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--report-column", help="Input column containing the pathology report.")
    parser.add_argument("--label-column", help="Gold label column for metrics.")
    parser.add_argument("--cancer-type-column", default="cancer_type")
    parser.add_argument("--mean-dss-column", default="mean_dss")
    parser.add_argument("--examples", type=Path, help="Optional CSV/JSONL few-shot examples.")
    parser.add_argument("--examples-per-call", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Maximum records to process.")
    parser.add_argument(
        "--skip-unlabeled",
        action="store_true",
        help="Skip records without a gold label before making API calls.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument("--thinking", choices=["auto", "enabled", "disabled"], default="auto")
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum completion tokens. Defaults to 2048 with thinking enabled, otherwise 512.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-reasoning", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("Set DEEPSEEK_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    records = load_table(args.input)
    if args.limit:
        records = records[: args.limit]
    examples = load_table(args.examples) if args.examples else []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    report_columns = [args.report_column] if args.report_column else REPORT_COLUMNS
    label_columns = [args.label_column] if args.label_column else LABEL_COLUMNS[args.task]
    prediction_key = PREDICTION_KEYS[args.task]
    thinking = task_thinking(args.task, args.thinking)
    max_tokens = args.max_tokens or (2048 if thinking == "enabled" else 512)

    golds = []
    preds = []
    error_count = 0
    skipped_unlabeled = 0
    processed_count = 0
    total_usage = {}

    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            gold = get_first(record, label_columns)
            if args.skip_unlabeled and gold in (None, ""):
                skipped_unlabeled += 1
                continue

            report_text = get_first(record, report_columns, "")
            if not report_text:
                result = {"index": index, "error": "missing report text"}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed_count += 1
                continue

            messages = build_messages(
                args.task,
                report_text,
                cancer_type=record.get(args.cancer_type_column),
                mean_dss=record.get(args.mean_dss_column),
                examples=pick_examples(examples, args.examples_per_call, args.seed + index),
            )
            response = call_deepseek(
                args.api_base,
                args.api_key,
                args.model,
                messages,
                thinking,
                args.reasoning_effort,
                max_tokens,
                args.timeout,
                args.retries,
            )
            parsed = parse_jsonish(response.get("content", ""))
            prediction = parsed.get(prediction_key)
            model_error = response.get("error")
            if not model_error and not response.get("content"):
                model_error = (
                    "empty model content; for thinking mode, rerun with a larger --max-tokens value"
                )
            elif not model_error and not parsed:
                model_error = "model content was not parseable as JSON"
            if model_error:
                error_count += 1

            if gold is not None:
                golds.append(normalize(gold))
                preds.append(normalize(prediction))

            for key, value in response.get("usage", {}).items():
                if isinstance(value, (int, float)):
                    total_usage[key] = total_usage.get(key, 0) + value

            output_record = {
                "index": index,
                "id": (
                    record.get("id")
                    or record.get("case_id")
                    or record.get("tcga_barcode")
                    or record.get("bcr_patient_barcode")
                ),
                "task": args.task,
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "thinking": thinking,
                "gold": gold,
                "prediction": prediction,
                "parsed": parsed,
                "raw_content": response.get("content", ""),
                "error": model_error,
            }
            if args.save_reasoning:
                output_record["reasoning_content"] = response.get("reasoning_content", "")
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            handle.flush()
            processed_count += 1

            if args.sleep:
                time.sleep(args.sleep)

    metrics = {
        "model": args.model,
        "input_n": len(records),
        "processed_n": processed_count,
        "skipped_unlabeled": skipped_unlabeled,
        "labeled_n": len(golds),
        "model_error_count": error_count,
        "prompt_version": PROMPT_VERSION,
        "usage": total_usage,
    }
    if golds:
        correct = sum(g == p for g, p in zip(golds, preds))
        metrics["correct"] = correct
        metrics["accuracy"] = correct / len(golds)
        metrics["macro_f1"] = macro_f1(golds, preds)

    metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
