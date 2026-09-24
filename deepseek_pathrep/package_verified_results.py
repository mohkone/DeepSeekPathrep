#!/usr/bin/env python3
"""Bundle checked, minimal research records without raw model reasoning."""

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

from scipy.stats import binomtest

from prompts import AJCC_STAGE_OPTIONS, CANCER_TYPES
from run_deepseek_pathology import normalize


PREDICTIONS = {
    "gpt4o_mini_cancer_type_full": "cancer_type",
    "gpt4o_mini_ajcc_stage_full": "ajcc_stage",
    "deepseek_cancer_type_clean_test": "cancer_type",
    "deepseek_ajcc_stage_clean_test": "ajcc_stage",
    "deepseek_prognosis_full_test": "prognosis",
}


def labels_for(task):
    return {
        "cancer_type": CANCER_TYPES,
        "ajcc_stage": AJCC_STAGE_OPTIONS,
        "prognosis": ["good", "poor"],
    }[task]


def score(rows, task):
    labels = [normalize(label) for label in labels_for(task)]
    gold = [normalize(row["gold"]) for row in rows]
    pred = [normalize(row["prediction"]) for row in rows]
    if any(label not in labels for label in gold):
        raise ValueError(f"Unrecognized gold label for {task}")
    per_class = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        per_class.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.)
    return {
        "n": len(rows),
        "correct": sum(g == p for g, p in zip(gold, pred)),
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(rows),
        "fixed_task_labels_macro_f1": sum(per_class) / len(labels),
        "invalid_prediction_count": sum(p not in labels for p in pred),
        "model_error_count": sum(bool(row.get("error")) for row in rows),
        "label_count": len(labels),
    }


def paired_correctness(flash, gpt):
    by_id = {row["id"]: row for row in gpt}
    if len(by_id) != len(gpt):
        raise ValueError("Duplicated GPT patient ID")
    flash_only = gpt_only = 0
    for left in flash:
        right = by_id[left["id"]]
        if normalize(left["gold"]) != normalize(right["gold"]):
            raise ValueError(f"Gold mismatch: {left['id']}")
        a = normalize(left["prediction"]) == normalize(left["gold"])
        b = normalize(right["prediction"]) == normalize(right["gold"])
        flash_only += bool(a and not b)
        gpt_only += bool(b and not a)
    return {
        "flash_only_correct": flash_only, "gpt_only_correct": gpt_only,
        "exact_mcnemar_two_sided_p": float(
            binomtest(flash_only, flash_only + gpt_only).pvalue
        ) if flash_only + gpt_only else 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--verification-report", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    root = args.repository
    bundle = {}
    checks = {}
    records_by_run = {}
    for run, task in PREDICTIONS.items():
        records = [
            json.loads(line)
            for line in (root / "outputs" / f"{run}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        minimal = [
            {
                "index": row["index"], "id": row["id"],
                "gold": row["gold"], "prediction": row["prediction"],
                "error": row.get("error"),
            }
            for row in records
        ]
        records_by_run[run] = minimal
        checks[run] = score(minimal, task)
        bundle[f"predictions/{run}.jsonl"] = (
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in minimal)
        ).encode()
    # Freeze expected counts and scores from the independently supplied report.
    expected = {
        "gpt4o_mini_cancer_type_full": (952, 926, .968712, 2),
        "gpt4o_mini_ajcc_stage_full": (594, 356, .590398, 0),
        "deepseek_cancer_type_clean_test": (182, 179, .899217, 0),
        "deepseek_ajcc_stage_clean_test": (116, 94, .827145, 7),
        "deepseek_prognosis_full_test": (952, 449, .479198, 31),
    }
    for run, (n, correct, f1, invalid) in expected.items():
        result = checks[run]
        if (result["n"], result["correct"], result["invalid_prediction_count"]) != (n, correct, invalid):
            raise AssertionError(f"Count mismatch: {run}: {result}")
        if abs(result["fixed_task_labels_macro_f1"] - f1) > 0.0000005:
            raise AssertionError(f"Fixed-label F1 mismatch: {run}: {result}")
    bundle["verified_fixed_label_scores.json"] = (
        json.dumps({"scoring": "fixed_task_labels_v1", "runs": checks}, indent=2) + "\n"
    ).encode()
    def read_csv(path):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    original_train = read_csv(root / "data" / "tcga_pathology_train.csv")
    original_val = read_csv(root / "data" / "tcga_pathology_val.csv")
    original_test = read_csv(root / "data" / "tcga_pathology_test.csv")
    original_inputs = original_train + original_val
    training_ids = {row["bcr_patient_barcode"] for row in original_inputs}
    def hash_report(text):
        return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
    training_hashes = {hash_report(row["report_text"]) for row in original_inputs}
    clean_ids = {
        row["bcr_patient_barcode"] for row in original_test
        if row["bcr_patient_barcode"] not in training_ids
        and hash_report(row["report_text"]) not in training_hashes
    }
    subset = {
        run: [row for row in records if row["id"] in clean_ids]
        for run, records in records_by_run.items()
    }
    subset_scores = {
        run: score(rows, PREDICTIONS[run])
        for run, rows in subset.items()
    }
    if len(clean_ids) != 181:
        raise AssertionError(f"Unexpected ID/text-disjoint cohort: {len(clean_ids)}")
    for run, n in (
        ("deepseek_cancer_type_clean_test", 181),
        ("deepseek_ajcc_stage_clean_test", 115),
        ("deepseek_prognosis_full_test", 181),
        ("gpt4o_mini_cancer_type_full", 181),
        ("gpt4o_mini_ajcc_stage_full", 115),
    ):
        if subset_scores[run]["n"] != n:
            raise AssertionError(f"Unexpected subset size: {run}")
    expected_subset_f1 = {
        "deepseek_cancer_type_clean_test": .900956,
        "deepseek_ajcc_stage_clean_test": .830834,
        "deepseek_prognosis_full_test": .517397,
        "gpt4o_mini_cancer_type_full": .891234,
        "gpt4o_mini_ajcc_stage_full": .597037,
    }
    for run, expected_f1 in expected_subset_f1.items():
        if abs(subset_scores[run]["fixed_task_labels_macro_f1"] - expected_f1) > .0000005:
            raise AssertionError(f"Subset fixed-label F1 mismatch: {run}")
    comparisons = {
        task: paired_correctness(
            subset[f"deepseek_{task}_clean_test"],
            subset[f"gpt4o_mini_{task}_full"],
        )
        for task in ("cancer_type", "ajcc_stage")
    }
    bundle["verified_text_disjoint_subset_scores.json"] = (
        json.dumps(
            {"selection": "patient and normalized report text disjoint from original train+val",
             "n_test_records": len(clean_ids), "runs": subset_scores,
             "paired_new_flash_vs_original_gpt": comparisons}, indent=2
        ) + "\n"
    ).encode()
    for filename in (
        "cox_test_risk_scores.csv",
        "cox_risk_score_metrics.json",
        "supervised_sensitivity_predictions.csv",
        "supervised_sensitivity_metrics.json",
        "leakage_audit.json",
    ):
        bundle[f"prior_outputs/{filename}"] = (root / "outputs" / filename).read_bytes()
    for path in sorted((root / "outputs" / "deduplicated_cox").glob("*")):
        bundle[f"deduplicated_cox/{path.name}"] = path.read_bytes()
    for name in (
        "run_cox_deduplicated.py", "test_cox_deduplicated.py",
        "package_verified_results.py", "run_cox_survival.py",
        "prompts.py", "run_deepseek_pathology.py",
        "requirements-analysis.txt", "requirements-analysis.lock.txt",
    ):
        bundle[f"code/{name}"] = (root / "deepseek_pathrep" / name).read_bytes()
    bundle["manuscript/manuscript_revised.md"] = (
        root / "deepseek_pathrep" / "manuscript_revised.md"
    ).read_bytes()
    bundle["VERIFICATION_REPORT.md"] = args.verification_report.read_bytes()
    bundle["README.md"] = (
        "# Verified PathRep revision package\n\n"
        "This is a single-cohort internal sensitivity study, not external validation. "
        "No API keys or raw model reasoning are included. The JSONL records retain "
        "only index, pseudonymous patient ID, gold label, prediction, and error.\n\n"
        "The `verified_fixed_label_scores.json` values are reproduced from the five "
        "included minimal prediction files by `code/package_verified_results.py`. "
        "The text-disjoint subset scores and exact paired correctness tests are "
        "also reproduced in `verified_text_disjoint_subset_scores.json`, using "
        "the original local benchmark CSVs. "
        "The `deduplicated_cox/` directory holds the new 7,247-record training manifest, "
        "887-row predictions, test manifest, and metrics including 2,000-resample "
        "paired disjoint-cohort bootstrap. The `code/` folder includes the "
        "27-package version lock from a clean Python 3.14.3 Linux x86_64 "
        "environment (versions pinned, wheel hashes not bundled). "
        "In the source repository, refit with:\n\n"
        "```bash\npython deepseek_pathrep/run_cox_deduplicated.py "
        "--data-dir data --cdr-clinical data/tcga_cdr_clinical.tsv "
        "--output-dir outputs/deduplicated_cox --bootstrap-replicates 2000\n```\n\n"
        "The source CSVs/TSV are not bundled. Their public preparation and clinical "
        "linkage provenance is described in the manuscript and verification report. "
        "`prior_outputs/` contains the earlier duplicated-training Cox scores for "
        "audit, not as leakage-free evidence. The supplied verification report cites "
        "historical original Flash predictions and machine-readable paired checks "
        "that were not provided here: original-run paired inference cannot be "
        "independently reproduced from this package. The archived failed "
        "leave-one-cancer-out fits and Brier scores are not successful analyses.\n"
    ).encode()
    manifest = {
        path: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in sorted(bundle.items())
    }
    bundle["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in sorted(bundle.items()):
            archive.writestr(path, data)
    print(json.dumps({"archive": str(args.archive), "files": len(bundle), "scores": checks}, indent=2))


if __name__ == "__main__":
    main()
