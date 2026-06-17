#!/usr/bin/env python
"""Download PathRep-Bench CSV splits and add convenience benchmark columns."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import urllib.request
from pathlib import Path


RAW_BASE_URL = "https://raw.githubusercontent.com/rachitsaluja/PathRep-Bench/main/data"
SPLITS = ["train", "val", "test"]


def download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Codex"})
    print(f"download: {url}")
    with urllib.request.urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())
    print(f"wrote: {destination} ({destination.stat().st_size} bytes)")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except ValueError:
        return None


def build_mean_dss(train_rows: list[dict], fallback_rows: list[dict]) -> dict[str, float]:
    values_by_type: dict[str, list[float]] = {}
    for row in train_rows:
        cancer_type = (row.get("type_name") or "").strip()
        dss = as_float(row.get("DSS.time"))
        if cancer_type and dss is not None:
            values_by_type.setdefault(cancer_type, []).append(dss)

    fallback_by_type: dict[str, list[float]] = {}
    for row in fallback_rows:
        cancer_type = (row.get("type_name") or "").strip()
        dss = as_float(row.get("DSS.time"))
        if cancer_type and dss is not None:
            fallback_by_type.setdefault(cancer_type, []).append(dss)

    means = {}
    for cancer_type, values in fallback_by_type.items():
        source_values = values_by_type.get(cancer_type) or values
        means[cancer_type] = statistics.mean(source_values)
    return means


def enrich_rows(rows: list[dict], mean_dss_by_type: dict[str, float]) -> list[dict]:
    enriched = []
    for row in rows:
        out = dict(row)
        cancer_type = (out.get("type_name") or "").strip()
        dss = as_float(out.get("DSS.time"))
        mean_dss = mean_dss_by_type.get(cancer_type)
        out["cancer_type"] = cancer_type
        out["report_text"] = out.get("text", "")
        out["ajcc_stage"] = out.get("stage_overall") or out.get("stage") or ""
        out["mean_dss"] = f"{mean_dss:.1f} days" if mean_dss is not None else ""
        if dss is not None and mean_dss is not None:
            out["prognosis"] = "good" if dss > mean_dss else "poor"
        else:
            out["prognosis"] = ""
        enriched.append(out)
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory for local CSV files.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload existing raw files.")
    args = parser.parse_args()

    raw_paths = {}
    for split in SPLITS:
        path = args.output_dir / f"pathrep_{split}.csv"
        raw_paths[split] = path
        download(f"{RAW_BASE_URL}/{split}.csv", path, args.force)

    raw_rows = {split: read_csv(path) for split, path in raw_paths.items()}
    all_rows = [row for split in SPLITS for row in raw_rows[split]]
    mean_dss_by_type = build_mean_dss(raw_rows["train"], all_rows)

    manifest = {
        "source": "https://github.com/rachitsaluja/PathRep-Bench/tree/main/data",
        "mean_dss_policy": "train split mean by cancer type, falling back to all splits if needed",
        "splits": {},
    }

    for split in SPLITS:
        enriched = enrich_rows(raw_rows[split], mean_dss_by_type)
        fieldnames = list(raw_rows[split][0].keys())
        for extra in ["cancer_type", "report_text", "ajcc_stage", "mean_dss", "prognosis"]:
            if extra not in fieldnames:
                fieldnames.append(extra)
        out_path = args.output_dir / f"tcga_pathology_{split}.csv"
        write_csv(out_path, enriched, fieldnames)
        manifest["splits"][split] = {
            "raw": str(raw_paths[split]),
            "enriched": str(out_path),
            "rows": len(enriched),
        }
        print(f"prepared: {out_path} ({len(enriched)} rows)")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
