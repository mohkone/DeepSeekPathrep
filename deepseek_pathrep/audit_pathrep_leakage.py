#!/usr/bin/env python
"""Audit PathRep-Bench for cross-split report duplication.

Checks for patient-level data leakage by comparing:
1. Patient barcodes across train/val/test splits
2. Report text content (hash-based) across splits
3. Produces a reproducible audit report

Usage:
    python audit_pathrep_leakage.py --data-dir ../data --output ../outputs/leakage_audit.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def text_hash(text: str) -> str:
    """SHA-256 hash of normalized report text."""
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit_splits(data_dir: Path) -> dict:
    """Perform comprehensive leakage audit."""
    splits = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"tcga_pathology_{split}.csv"
        rows = load_csv(path)
        splits[split] = rows

    # Extract barcodes and text hashes
    barcodes = {}
    text_hashes = {}
    for split, rows in splits.items():
        bc = set()
        th = set()
        for row in rows:
            bc.add(row.get("bcr_patient_barcode", ""))
            report_text = row.get("report_text", "") or row.get("text", "")
            if report_text:
                th.add(text_hash(report_text))
        barcodes[split] = bc
        text_hashes[split] = th

    # Barcode overlap
    barcode_overlap = {
        "train_val": len(barcodes["train"] & barcodes["val"]),
        "train_test": len(barcodes["train"] & barcodes["test"]),
        "val_test": len(barcodes["val"] & barcodes["test"]),
    }

    # Text hash overlap
    text_overlap = {
        "train_val": len(text_hashes["train"] & text_hashes["val"]),
        "train_test": len(text_hashes["train"] & text_hashes["test"]),
        "val_test": len(text_hashes["val"] & text_hashes["test"]),
    }

    # Verify: do overlapping barcodes have identical text?
    train_by_bc = {r.get("bcr_patient_barcode", ""): r for r in splits["train"]}
    test_by_bc = {r.get("bcr_patient_barcode", ""): r for r in splits["test"]}
    val_by_bc = {r.get("bcr_patient_barcode", ""): r for r in splits["val"]}

    train_test_overlap_barcodes = barcodes["train"] & barcodes["test"]
    identical_text_count = 0
    text_diff_count = 0
    for bc in train_test_overlap_barcodes:
        train_text = train_by_bc.get(bc, {}).get("report_text", "") or train_by_bc.get(bc, {}).get("text", "")
        test_text = test_by_bc.get(bc, {}).get("report_text", "") or test_by_bc.get(bc, {}).get("text", "")
        if text_hash(train_text) == text_hash(test_text):
            identical_text_count += 1
        else:
            text_diff_count += 1

    # Within-split duplicates
    within_split_dups = {}
    for split, rows in splits.items():
        bcs = [r.get("bcr_patient_barcode", "") for r in rows]
        within_split_dups[split] = len(bcs) - len(set(bcs))

    # Clean split sizes
    clean_test = splits["test"][:0]  # empty DataFrame-like
    clean_test_count = sum(
        1 for r in splits["test"]
        if r.get("bcr_patient_barcode", "") not in barcodes["train"]
    )
    clean_val_count = sum(
        1 for r in splits["val"]
        if r.get("bcr_patient_barcode", "") not in barcodes["train"]
    )

    result = {
        "source": "https://github.com/rachitsaluja/PathRep-Bench",
        "data_files": {
            "train": str(data_dir / "tcga_pathology_train.csv"),
            "val": str(data_dir / "tcga_pathology_val.csv"),
            "test": str(data_dir / "tcga_pathology_test.csv"),
        },
        "split_sizes": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "unique_barcodes": {
            "train": len(barcodes["train"]),
            "val": len(barcodes["val"]),
            "test": len(barcodes["test"]),
        },
        "barcode_overlap": barcode_overlap,
        "text_hash_overlap": text_overlap,
        "train_test_overlap_verification": {
            "total_overlapping_barcodes": len(train_test_overlap_barcodes),
            "identical_text": identical_text_count,
            "different_text": text_diff_count,
            "identical_percentage": f"{100 * identical_text_count / max(len(train_test_overlap_barcodes), 1):.1f}%",
        },
        "within_split_duplicate_barcodes": within_split_dups,
        "clean_split_sizes": {
            "test_no_train_overlap": clean_test_count,
            "val_no_train_overlap": clean_val_count,
            "test_leakage_rate": f"{100 * (len(splits['test']) - clean_test_count) / len(splits['test']):.1f}%",
            "val_leakage_rate": f"{100 * (len(splits['val']) - clean_val_count) / len(splits['val']):.1f}%",
        },
    }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "leakage_audit.json",
    )
    args = parser.parse_args()

    print("Auditing PathRep-Bench splits for cross-split report duplication...")
    print(f"Data directory: {args.data_dir}")
    result = audit_splits(args.data_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("LEAKAGE AUDIT REPORT")
    print("=" * 60)
    print(f"Source: {result['source']}")
    print(f"\nSplit sizes: train={result['split_sizes']['train']}, "
          f"val={result['split_sizes']['val']}, test={result['split_sizes']['test']}")
    print(f"\nBarcode overlap:")
    print(f"  Train-Val:  {result['barcode_overlap']['train_val']}")
    print(f"  Train-Test: {result['barcode_overlap']['train_test']}")
    print(f"  Val-Test:   {result['barcode_overlap']['val_test']}")
    print(f"\nText hash overlap:")
    print(f"  Train-Val:  {result['text_hash_overlap']['train_val']}")
    print(f"  Train-Test: {result['text_hash_overlap']['train_test']}")
    print(f"  Val-Test:   {result['text_hash_overlap']['val_test']}")
    print(f"\nTrain-Test overlap verification:")
    v = result["train_test_overlap_verification"]
    print(f"  Overlapping barcodes: {v['total_overlapping_barcodes']}")
    print(f"  Identical text: {v['identical_text']} ({v['identical_percentage']})")
    print(f"  Different text: {v['different_text']}")
    print(f"\nClean split sizes:")
    c = result["clean_split_sizes"]
    print(f"  Test (no train overlap): {c['test_no_train_overlap']} "
          f"(leakage rate: {c['test_leakage_rate']})")
    print(f"  Val (no train overlap): {c['val_no_train_overlap']} "
          f"(leakage rate: {c['val_leakage_rate']})")
    print(f"\nWithin-split duplicate barcodes: {result['within_split_duplicate_barcodes']}")
    print(f"\nFull report saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
