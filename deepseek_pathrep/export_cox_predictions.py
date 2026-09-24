#!/usr/bin/env python3
"""Export held-out risk scores for the published Cox configurations.

Uses the same data merge, feature preparation, and penalizer as
run_cox_survival.py. This is an audit artifact, not external validation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from run_cox_survival import (
    build_clinical_features,
    build_tfidf_features,
    load_cdr_clinical,
    load_pathrep_data,
    merge_survival_info,
)


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
    clinical = build_clinical_features(split["train"], split["val"], split["test"])
    tfidf = build_tfidf_features(split["train"], split["val"], split["test"], 100)

    clinical_train = pd.concat(
        [clinical["train"], clinical["val"]], ignore_index=True
    )
    clinical_test = clinical["test"].reset_index(drop=True)
    svd_train = np.vstack([tfidf["train"], tfidf["val"]])
    svd_test = tfidf["test"]
    svd_cols = [f"svd_{i}" for i in range(svd_train.shape[1])]
    clinical_cols = list(clinical_train.columns)
    assert len(train) == len(clinical_train) == len(svd_train)
    assert len(test) == len(clinical_test) == len(svd_test)

    x_train = pd.concat(
        [
            clinical_train,
            pd.DataFrame(svd_train, columns=svd_cols),
        ],
        axis=1,
    )
    x_test = pd.concat(
        [
            clinical_test,
            pd.DataFrame(svd_test, columns=svd_cols),
        ],
        axis=1,
    )
    target_cols = ["DSS.time", "dss_event"]
    y_train = train[target_cols].reset_index(drop=True)
    y_test = test[target_cols]

    scores = {}
    for label, columns in [
        ("clinical_only", clinical_cols),
        ("clinical_plus_text", clinical_cols + svd_cols),
        ("text_only", svd_cols),
    ]:
        model = CoxPHFitter(penalizer=0.1)
        model.fit(
            pd.concat([y_train, x_train[columns]], axis=1),
            duration_col="DSS.time",
            event_col="dss_event",
        )
        score = model.predict_partial_hazard(x_test[columns]).to_numpy().ravel()
        scores[label] = score

    clean_barcodes = set(
        pd.read_csv(
            args.data_dir / "tcga_pathology_test_clean.csv",
            usecols=["bcr_patient_barcode"],
        )["bcr_patient_barcode"]
    )
    clean_mask = test["bcr_patient_barcode"].isin(clean_barcodes).to_numpy()
    rows = []
    for idx, row in test.iterrows():
        rows.append(
            {
                "test_row_index": int(idx),
                "bcr_patient_barcode": row["bcr_patient_barcode"],
                "dss_time_days": float(row["DSS.time"]),
                "dss_event": int(row["dss_event"]),
                "clean_test": bool(clean_mask[idx]),
                **{f"risk_{label}": float(score[idx]) for label, score in scores.items()},
            }
        )
    path = args.output_dir / "cox_test_risk_scores.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {"n_train": len(train), "n_test": len(test), "n_clean_test": int(sum(clean_mask))}
    for label, score in scores.items():
        metrics[label] = {}
        for subset, mask in [
            ("full", np.ones(len(test), dtype=bool)),
            ("clean", clean_mask),
        ]:
            c = concordance_index(
                test.loc[mask, "DSS.time"].to_numpy(),
                -score[mask],
                test.loc[mask, "dss_event"].to_numpy(),
            )
            metrics[label][subset] = {"n": int(sum(mask)), "c_index": float(c)}
    (args.output_dir / "cox_risk_score_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
