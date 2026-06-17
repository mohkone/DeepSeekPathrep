#!/usr/bin/env python
"""Extract structured pathology variables with DeepSeek for hybrid prognosis modeling."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPORT_COLUMNS = ["report_text", "pathology_report", "pathology_text", "report", "note_text", "text"]
ID_COLUMNS = ["bcr_patient_barcode", "id", "case_id", "tcga_barcode"]


def load_table(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def get_first(record: dict, names: list[str], default=""):
    for name in names:
        if record.get(name) not in (None, ""):
            return record[name]
    return default


def record_id(record: dict, index: int) -> str:
    return get_first(record, ID_COLUMNS, str(index))


def load_existing(path: Path) -> tuple[set[str], set[int]]:
    if not path.exists():
        return set(), set()
    seen_ids = set()
    seen_indexes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("id") not in (None, ""):
            seen_ids.add(str(row["id"]))
        if row.get("index") is not None:
            seen_indexes.add(int(row["index"]))
    return seen_ids, seen_indexes


def summarize_output(path: Path, model: str, input_n: int, skipped_existing: int, run_usage: dict) -> dict:
    rows = []
    if path.exists():
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    total_usage = {}
    for row in rows:
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                total_usage[key] = total_usage.get(key, 0) + value
    return {
        "model": model,
        "input_n": input_n,
        "processed_n": len(rows),
        "skipped_existing": skipped_existing,
        "model_error_count": sum(1 for row in rows if row.get("error")),
        "usage": total_usage or run_usage,
        "run_usage": run_usage,
    }


def parse_jsonish(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def build_messages(report_text: str) -> list[dict]:
    schema = {
        "cancer_type": "string or unknown",
        "ajcc_stage": "Stage I, Stage II, Stage III, Stage IV, or unknown",
        "tumor_size_cm": "number or null",
        "positive_nodes": "integer or null",
        "nodes_examined": "integer or null",
        "distant_metastasis": "present, absent, or unknown",
        "margin_status": "positive, negative, close, or unknown",
        "lymphovascular_invasion": "present, absent, or unknown",
        "grade": "string or unknown",
        "necrosis": "present, absent, or unknown",
        "key_evidence": ["short quoted or paraphrased evidence snippets"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a pathology information extraction assistant for retrospective research. "
                "Use only the provided report. Return valid json only, no markdown and no extra keys. "
                "Use null or unknown when a variable is not explicitly supported."
            ),
        },
        {
            "role": "user",
            "content": (
                "Extract structured variables relevant to cancer prognosis. "
                "Return exactly this JSON schema:\n"
                f"{json.dumps(schema)}\n\n"
                "Pathology report:\n"
                + report_text
            ),
        },
    ]


def call_deepseek(api_base, api_key, model, messages, max_tokens, timeout, retries):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
            message = data["choices"][0]["message"]
            return {"content": message.get("content") or "", "usage": data.get("usage", {})}
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:1000]}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        if attempt < retries:
            time.sleep(min(2**attempt, 8))
    return {"content": "", "usage": {}, "error": last_error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--start", type=int, default=0, help="Start index in the input CSV.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Append to output and skip rows already present.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    if not args.api_key:
        print("Set DEEPSEEK_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    rows = load_table(args.input)
    indexed_rows = list(enumerate(rows))
    if args.start:
        indexed_rows = indexed_rows[args.start :]
    if args.limit:
        indexed_rows = indexed_rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_usage = {}
    error_count = 0
    skipped_existing = 0
    seen_ids, seen_indexes = load_existing(args.output) if args.resume else (set(), set())
    mode = "a" if args.resume else "w"

    with args.output.open(mode, encoding="utf-8") as handle:
        for index, row in indexed_rows:
            row_id = record_id(row, index)
            if args.resume and (row_id in seen_ids or index in seen_indexes):
                skipped_existing += 1
                continue
            report_text = get_first(row, REPORT_COLUMNS)
            response = call_deepseek(
                args.api_base,
                args.api_key,
                args.model,
                build_messages(report_text),
                args.max_tokens,
                args.timeout,
                args.retries,
            )
            variables = parse_jsonish(response.get("content", ""))
            error = response.get("error")
            if not error and not variables:
                error = "empty or unparseable structured variable JSON"
            if error:
                error_count += 1
            for key, value in response.get("usage", {}).items():
                if isinstance(value, (int, float)):
                    total_usage[key] = total_usage.get(key, 0) + value
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "id": row_id,
                        "model": args.model,
                        "variables": variables,
                        "raw_content": response.get("content", ""),
                        "usage": response.get("usage", {}),
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if args.sleep:
                time.sleep(args.sleep)

    metrics = summarize_output(args.output, args.model, len(indexed_rows), skipped_existing, total_usage)
    metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
