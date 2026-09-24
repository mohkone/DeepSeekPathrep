#!/usr/bin/env python3
"""Reproduce TF-IDF supervised sensitivity results on the DSS-observed subset.

These runs are separate from the manuscript's original 952-case prognosis model.
The 887-row test set here requires DSS time and event status; its 174-row
barcode-disjoint subset is a nested sensitivity subset, not external validation.
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from run_cox_survival import load_cdr_clinical, load_pathrep_data, merge_survival_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cdr-clinical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = merge_survival_info(
        load_pathrep_data(args.data_dir), load_cdr_clinical(args.cdr_clinical)
    )
    train = pd.concat([split["train"], split["val"]], ignore_index=True)
    test = split["test"].reset_index(drop=True)
    assert len(train) == 7962 and len(test) == 887
    barcode_disjoint = set(
        pd.read_csv(
            args.data_dir / "tcga_pathology_test_clean.csv",
            usecols=["bcr_patient_barcode"],
        )["bcr_patient_barcode"]
    )
    clean = test["bcr_patient_barcode"].isin(barcode_disjoint).to_numpy()
    assert sum(clean) == 174

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=50000,
    )
    x_train = vectorizer.fit_transform(train["report_text"].fillna(""))
    x_test = vectorizer.transform(test["report_text"].fillna(""))
    runs = [
        ("prognosis", "prognosis", "liblinear"),
        ("cancer_type", "cancer_type", "lbfgs"),
    ]
    rows = [
        {
            "test_row_index": i,
            "bcr_patient_barcode": r["bcr_patient_barcode"],
            "barcode_disjoint": bool(clean[i]),
        }
        for i, r in test.iterrows()
    ]
    metrics = {"n_train": len(train), "n_test": len(test), "n_barcode_disjoint": int(sum(clean))}
    for name, label, solver in runs:
        y_train = train[label].astype(str)
        y_test = test[label].astype(str)
        model = LogisticRegression(
            class_weight="balanced",
            solver=solver,
            C=2.0,
            max_iter=3000,
            random_state=2026,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        for i, row in enumerate(rows):
            row[f"{name}_gold"] = y_test.iloc[i]
            row[f"{name}_prediction"] = str(predictions[i])
        metrics[name] = {}
        for subset_name, subset in [
            ("full", slice(None)),
            ("barcode_disjoint", clean),
        ]:
            metrics[name][subset_name] = {
                "n": len(y_test[subset]),
                "accuracy": float(accuracy_score(y_test[subset], predictions[subset])),
                "macro_f1": float(
                    f1_score(y_test[subset], predictions[subset], average="macro")
                ),
            }

    out = args.output_dir / "supervised_sensitivity_predictions.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "supervised_sensitivity_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
