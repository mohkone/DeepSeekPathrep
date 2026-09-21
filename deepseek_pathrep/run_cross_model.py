#!/usr/bin/env python
"""Cross-model comparison: run multiple LLMs on the same PathRep-Bench tasks.

Supports OpenAI (GPT-4o, GPT-4o-mini), Anthropic (Claude), and DeepSeek models.
Auth is handled via platform credential proxy (HTTPS_PROXY) when api_credentials
is set on the bash call, or via explicit --api-key / env var.

Usage:
    # With platform credentials (recommended):
    python run_cross_model.py --input ../data/tcga_pathology_test.csv \\
        --task cancer_type --provider openai --model gpt-4o-mini \\
        --output ../outputs/gpt4o_mini_cancer_type.jsonl

    # With explicit key:
    python run_cross_model.py --input ../data/tcga_pathology_test.csv \\
        --task cancer_type --provider openai --model gpt-4o-mini \\
        --api-key sk-xxx --output ../outputs/gpt4o_mini_cancer_type.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PROMPT_VERSION, build_messages
from run_deepseek_pathology import (
    REPORT_COLUMNS,
    LABEL_COLUMNS,
    PREDICTION_KEYS,
    load_table,
    get_first,
    pick_examples,
    parse_jsonish,
    normalize,
    macro_f1,
    task_thinking,
)


def _post_json(url: str, payload: dict, headers: dict, timeout: int, retries: int) -> dict:
    """POST JSON with retries. Returns parsed response or error dict."""
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
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:500]}"
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries:
            time.sleep(min(2**attempt, 8))
    return {"_error": last_error}


def call_openai(api_key: str, model: str, messages: list[dict],
                max_tokens: int, timeout: int, retries: int) -> dict:
    """Call OpenAI chat completions API."""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = _post_json(url, payload, headers, timeout, retries)
    if "_error" in data:
        return {"content": "", "reasoning_content": "", "usage": {}, "error": data["_error"]}
    choice = data["choices"][0]["message"]
    return {
        "content": choice.get("content") or "",
        "reasoning_content": "",
        "usage": data.get("usage", {}),
    }


def call_anthropic(api_key: str, model: str, messages: list[dict],
                   max_tokens: int, timeout: int, retries: int) -> dict:
    """Call Anthropic Messages API."""
    url = "https://api.anthropic.com/v1/messages"

    system_content = ""
    conv_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content += msg["content"] + "\n"
        else:
            conv_messages.append(msg)

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": conv_messages,
        "stream": False,
    }
    if system_content:
        payload["system"] = system_content.strip()

    headers = {"anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    data = _post_json(url, payload, headers, timeout, retries)
    if "_error" in data:
        return {"content": "", "reasoning_content": "", "usage": {}, "error": data["_error"]}

    content = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            content += block.get("text", "")

    return {
        "content": content,
        "reasoning_content": "",
        "usage": {
            "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
            "total_tokens": data.get("usage", {}).get("input_tokens", 0)
            + data.get("usage", {}).get("output_tokens", 0),
        },
    }


def call_deepseek(api_key: str, api_base: str, model: str, messages: list[dict],
                  thinking: str, reasoning_effort: str, max_tokens: int,
                  timeout: int, retries: int) -> dict:
    """Call DeepSeek API."""
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

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = _post_json(url, payload, headers, timeout, retries)
    if "_error" in data:
        return {"content": "", "reasoning_content": "", "usage": {}, "error": data["_error"]}
    choice = data["choices"][0]["message"]
    return {
        "content": choice.get("content") or "",
        "reasoning_content": choice.get("reasoning_content") or "",
        "usage": data.get("usage", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["cancer_type", "ajcc_stage", "prognosis"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "deepseek"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--report-column")
    parser.add_argument("--label-column")
    parser.add_argument("--cancer-type-column", default="cancer_type")
    parser.add_argument("--mean-dss-column", default="mean_dss")
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--examples-per-call", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-unlabeled", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["auto", "enabled", "disabled"], default="auto")
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    api_key = args.api_key or os.environ.get(env_var_map[args.provider], "")
    if not api_key:
        print(f"No {env_var_map[args.provider]} set; relying on platform credential proxy.",
              file=sys.stderr)

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
    processed_count = 0
    total_usage = {}

    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            gold = get_first(record, label_columns)
            if args.skip_unlabeled and gold in (None, ""):
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

            if args.provider == "openai":
                response = call_openai(api_key, args.model, messages, max_tokens,
                                       args.timeout, args.retries)
            elif args.provider == "anthropic":
                response = call_anthropic(api_key, args.model, messages, max_tokens,
                                          args.timeout, args.retries)
            else:
                response = call_deepseek(api_key, args.api_base, args.model, messages,
                                         thinking, args.reasoning_effort,
                                         max_tokens, args.timeout, args.retries)

            parsed = parse_jsonish(response.get("content", ""))
            prediction = parsed.get(prediction_key)
            model_error = response.get("error")
            if not model_error and not response.get("content"):
                model_error = "empty model content"
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
                "id": record.get("id") or record.get("case_id")
                      or record.get("tcga_barcode")
                      or record.get("bcr_patient_barcode"),
                "task": args.task,
                "provider": args.provider,
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "thinking": thinking,
                "gold": gold,
                "prediction": prediction,
                "parsed": parsed,
                "raw_content": response.get("content", ""),
                "error": model_error,
            }
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            handle.flush()
            processed_count += 1

            if processed_count % 100 == 0:
                acc = sum(g == p for g, p in zip(golds, preds)) / max(len(golds), 1)
                print(f"  [{processed_count}] accuracy so far: {acc:.4f} ({len(golds)} labeled)")

            if args.sleep:
                time.sleep(args.sleep)

    metrics = {
        "provider": args.provider,
        "model": args.model,
        "input_n": len(records),
        "processed_n": processed_count,
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
