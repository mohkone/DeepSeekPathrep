#!/usr/bin/env python3
"""Patient- and text-deduplicated Cox sensitivity experiment.

The original test split is not a leakage-free evaluation set. The primary
evaluation here is the locked subset with no patient ID or normalized report
text shared with either original training split. No test outcome is used for
selection, fitting, preprocessing, or hyperparameter choice.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from run_cox_survival import load_cdr_clinical, load_pathrep_data, merge_survival_info


def text_hash(value):
    text = " ".join(str(value if pd.notna(value) else "").lower().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_cohorts(split):
    train = pd.concat(
        [split["train"].assign(source_split="train"),
         split["val"].assign(source_split="val")],
        ignore_index=True,
    )
    test = split["test"].reset_index(drop=True).copy()
    for frame in (train, test):
        frame["report_sha256"] = frame["report_text"].map(text_hash)
    original_ids = set(train["bcr_patient_barcode"])
    original_hashes = set(train["report_sha256"])
    test["id_disjoint"] = ~test["bcr_patient_barcode"].isin(original_ids)
    test["text_disjoint"] = ~test["report_sha256"].isin(original_hashes)
    test["id_text_disjoint"] = test["id_disjoint"] & test["text_disjoint"]
    before = len(train)
    train = train.drop_duplicates("bcr_patient_barcode", keep="first")
    after_id = len(train)
    train = train.drop_duplicates("report_sha256", keep="first").reset_index(drop=True)
    assert train["bcr_patient_barcode"].is_unique
    assert train["report_sha256"].is_unique
    assert not set(test.loc[test["id_text_disjoint"], "bcr_patient_barcode"]) & set(train["bcr_patient_barcode"])
    assert not set(test.loc[test["id_text_disjoint"], "report_sha256"]) & set(train["report_sha256"])
    counts = {
        "eligible_train": len(split["train"]),
        "eligible_val": len(split["val"]),
        "combined_rows": before,
        "patient_duplicates_removed": before - after_id,
        "same_text_different_patient_removed": after_id - len(train),
        "fitted_unique_patients_and_reports": len(train),
        "test_all": len(test),
        "test_id_disjoint": int(test["id_disjoint"].sum()),
        "test_id_text_disjoint": int(test["id_text_disjoint"].sum()),
    }
    return train, test, counts


def features(train, test):
    """Fit imputation, vocabulary, SVD, scaling, and categories on train only."""
    age = pd.to_numeric(train["age_at_initial_pathologic_diagnosis"], errors="coerce")
    age_median = float(age.median()) if age.notna().any() else 60.0
    types = sorted(train["cancer_type"].dropna().unique())

    def clinical(frame):
        result = pd.DataFrame(index=range(len(frame)))
        result["age"] = pd.to_numeric(
            frame["age_at_initial_pathologic_diagnosis"], errors="coerce"
        ).reset_index(drop=True).fillna(age_median).astype(float)
        result["is_male"] = (
            frame["gender"].fillna("").str.upper().eq("MALE").astype(float)
            .reset_index(drop=True)
        )
        for number, cancer in enumerate(types):
            result[f"ct_{number:02d}"] = (
                frame["cancer_type"].eq(cancer).astype(float).reset_index(drop=True)
            )
        return result

    clinical_train, clinical_test = clinical(train), clinical(test)
    clinical_cols = [col for col in clinical_train if clinical_train[col].nunique() > 1]
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_features=50000,
    )
    text_train = vectorizer.fit_transform(train["report_text"].fillna("").astype(str))
    text_test = vectorizer.transform(test["report_text"].fillna("").astype(str))
    n_components = min(100, min(text_train.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=2026)
    svd_train = svd.fit_transform(text_train)
    svd_test = svd.transform(text_test)
    scaler = StandardScaler()
    svd_train = scaler.fit_transform(svd_train)
    svd_test = scaler.transform(svd_test)
    text_cols = [f"svd_{i}" for i in range(n_components)]
    x_train = pd.concat(
        [clinical_train, pd.DataFrame(svd_train, columns=text_cols)], axis=1
    )
    x_test = pd.concat(
        [clinical_test, pd.DataFrame(svd_test, columns=text_cols)], axis=1
    )
    return x_train, x_test, clinical_cols, text_cols, age_median, types


def c_index(frame, scores, mask):
    return float(concordance_index(
        frame.loc[mask, "DSS.time"].to_numpy(),
        -scores[mask],
        frame.loc[mask, "dss_event"].to_numpy(),
    ))


def bootstrap(frame, predictions, mask, replicates, seed):
    indices = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    scores = {key: values[indices] for key, values in predictions.items()}
    durations = frame.loc[mask, "DSS.time"].to_numpy()
    events = frame.loc[mask, "dss_event"].to_numpy()
    draws = {key: [] for key in [*scores, "clinical_plus_text_minus_clinical_only"]}
    skipped = 0
    for _ in range(replicates):
        draw = rng.integers(0, len(indices), size=len(indices))
        try:
            values = {
                name: float(concordance_index(durations[draw], -risk[draw], events[draw]))
                for name, risk in scores.items()
            }
        except ZeroDivisionError:
            skipped += 1
            continue
        for name, value in values.items():
            draws[name].append(value)
        draws["clinical_plus_text_minus_clinical_only"].append(
            values["clinical_plus_text"] - values["clinical_only"]
        )
    return {
        "resamples_requested": replicates, "resamples_valid": replicates - skipped,
        "seed": seed, "unit": "test patient, paired across feature sets",
        "conditional_on_fitted_models": True,
        "percentile_95_ci": {
            name: [float(v) for v in np.percentile(values, [2.5, 97.5])]
            for name, values in draws.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--cdr-clinical", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = merge_survival_info(
        load_pathrep_data(args.data_dir), load_cdr_clinical(args.cdr_clinical)
    )
    train, test, counts = prepare_cohorts(split)
    x_train, x_test, clinical, text, median, types = features(train, test)
    predictions = {}
    for label, columns in (
        ("clinical_only", clinical),
        ("clinical_plus_text", clinical + text),
        ("text_only", text),
    ):
        model = CoxPHFitter(penalizer=0.1)
        model.fit(
            pd.concat([train[["DSS.time", "dss_event"]], x_train[columns]], axis=1),
            duration_col="DSS.time", event_col="dss_event",
        )
        predictions[label] = model.predict_partial_hazard(x_test[columns]).to_numpy().ravel()

    output = pd.DataFrame({
        "test_row_index": np.arange(len(test)),
        "bcr_patient_barcode": test["bcr_patient_barcode"],
        "report_sha256": test["report_sha256"],
        "dss_time_days": test["DSS.time"],
        "dss_event": test["dss_event"],
        "id_disjoint": test["id_disjoint"],
        "text_disjoint": test["text_disjoint"],
        "id_text_disjoint": test["id_text_disjoint"],
    })
    for name, values in predictions.items():
        output[f"risk_{name}"] = values
    output.to_csv(args.output_dir / "cox_deduplicated_test_risk_scores.csv", index=False)
    train[["source_split", "bcr_patient_barcode", "report_sha256"]].to_csv(
        args.output_dir / "cox_deduplicated_training_manifest.csv", index=False
    )
    test[["bcr_patient_barcode", "report_sha256", "id_disjoint",
          "text_disjoint", "id_text_disjoint"]].to_csv(
        args.output_dir / "cox_deduplicated_test_manifest.csv", index=False
    )
    masks = {
        "original_test_overlapping": np.ones(len(test), dtype=bool),
        "patient_id_disjoint": test["id_disjoint"].to_numpy(),
        "patient_id_and_text_disjoint": test["id_text_disjoint"].to_numpy(),
    }
    result = {
        "design": "train+val patient-ID then normalized-text deduplication; retained first train then val record",
        "cohorts": counts,
        "feature_fit": "deduplicated training only; no test fit or test outcome-based selection",
        "age_imputation_median_train": median,
        "cancer_type_categories_train": types,
        "tfidf_svd_components": len(text),
        "penalizer": 0.1,
        "endpoints": {
            name: {
                "n": int(sum(mask)),
                "events": int(test.loc[mask, "dss_event"].sum()),
                "c_index": {
                    label: c_index(test, score, mask)
                    for label, score in predictions.items()
                },
            }
            for name, mask in masks.items()
        },
        "disjoint_test_bootstrap": bootstrap(
            test, predictions, masks["patient_id_and_text_disjoint"],
            args.bootstrap_replicates, 20260924,
        ),
        "limitations": [
            "Same TCGA-derived cohort; no external validation.",
            "Original 887-case test includes training overlap and is descriptive only.",
            "Training combines predefined train and validation splits without held-out model selection.",
            "No leave-one-cancer-type-out or integrated Brier score evaluation.",
        ],
    }
    (args.output_dir / "cox_deduplicated_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
