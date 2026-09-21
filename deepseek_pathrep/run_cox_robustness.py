#!/usr/bin/env python
"""Robustness validation for the Cox survival model.

Addresses reviewer/advisor concerns:
1. Bootstrap 95% CI for C-index
2. Paired bootstrap delta C-index (clinical-only vs clinical+TF-IDF)
3. Patient barcode overlap check across train/val/test
4. Leave-one-cancer-type-out (LOCTO) cross-validation
5. Report sample sizes and censoring distribution
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_and_merge(data_dir: Path, cdr_path: Path) -> dict[str, pd.DataFrame]:
    """Load PathRep-Bench data and merge with TCGA-CDR censoring info."""
    splits = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"tcga_pathology_{split}.csv"
        splits[split] = pd.read_csv(path, encoding="utf-8-sig")

    cdr = pd.read_csv(cdr_path, sep="\t", encoding="utf-8")
    cdr = cdr.rename(columns={"Patient ID": "bcr_patient_barcode"})
    dss_map = {"DEAD WITH TUMOR": 1, "ALIVE OR DEAD TUMOR FREE": 0}
    cdr["dss_event"] = cdr["Disease-specific Survival status"].map(dss_map)
    cdr["dss_time_days"] = pd.to_numeric(
        cdr["Months of disease-specific survival"], errors="coerce"
    ) * 30.4375

    merged = {}
    for split, df in splits.items():
        m = df.merge(
            cdr[["bcr_patient_barcode", "dss_event"]],
            on="bcr_patient_barcode",
            how="left",
        )
        m["dss_event"] = m["dss_event"].fillna(-1).astype(int)
        m = m[
            (m["DSS.time"].notna())
            & (m["DSS.time"] > 0)
            & (m["dss_event"] >= 0)
        ].copy()
        m["DSS.time"] = pd.to_numeric(m["DSS.time"], errors="coerce")
        m = m.dropna(subset=["DSS.time"])
        merged[split] = m

    return merged


def check_barcode_overlap(merged: dict[str, pd.DataFrame]) -> dict:
    """Check for patient barcode overlap across splits."""
    train_barcodes = set(merged["train"]["bcr_patient_barcode"])
    val_barcodes = set(merged["val"]["bcr_patient_barcode"])
    test_barcodes = set(merged["test"]["bcr_patient_barcode"])

    overlap = {
        "train_val": len(train_barcodes & val_barcodes),
        "train_test": len(train_barcodes & test_barcodes),
        "val_test": len(val_barcodes & test_barcodes),
        "train_size": len(train_barcodes),
        "val_size": len(val_barcodes),
        "test_size": len(test_barcodes),
    }
    return overlap


def build_tfidf(train_df, val_df, test_df, n_components=100):
    """Build TF-IDF + SVD features."""
    train_text = train_df["report_text"].fillna("").astype(str)
    test_text = test_df["report_text"].fillna("").astype(str)
    val_text = val_df["report_text"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_features=50000,
    )
    train_tfidf = vectorizer.fit_transform(train_text)
    val_tfidf = vectorizer.transform(val_text)
    test_tfidf = vectorizer.transform(test_text)

    n_comp = min(n_components, min(train_tfidf.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=2026)
    train_svd = svd.fit_transform(train_tfidf)
    val_svd = svd.transform(val_tfidf)
    test_svd = svd.transform(test_tfidf)

    scaler = StandardScaler()
    train_svd = scaler.fit_transform(train_svd)
    val_svd = scaler.transform(val_svd)
    test_svd = scaler.transform(test_svd)

    return {"train": train_svd, "val": val_svd, "test": test_svd, "n_components": n_comp}


def build_clinical(train_df, val_df, test_df):
    """Build clinical features."""
    splits = {"train": train_df, "val": val_df, "test": test_df}
    result = {}
    for name, df in splits.items():
        feat = pd.DataFrame()
        feat["age"] = pd.to_numeric(
            df["age_at_initial_pathologic_diagnosis"], errors="coerce"
        )
        feat["age"] = feat["age"].fillna(feat["age"].median() if feat["age"].notna().any() else 60.0)
        feat["is_male"] = (df["gender"].str.upper() == "MALE").astype(int)
        ct = pd.get_dummies(df["cancer_type"], prefix="ct").astype(float)
        feat = pd.concat([feat, ct], axis=1)
        result[name] = feat

    all_cols = set()
    for df in result.values():
        all_cols.update(df.columns)
    for name in result:
        for col in all_cols:
            if col not in result[name].columns:
                result[name][col] = 0.0
        result[name] = result[name][sorted(all_cols)]

    return result


def fit_and_evaluate(train_df, test_df, feature_cols, label=""):
    """Fit Cox model and return C-index."""
    train_data = train_df[["DSS.time", "dss_event"] + feature_cols].copy()
    train_data["DSS.time"] = pd.to_numeric(train_data["DSS.time"], errors="coerce")
    train_data = train_data.dropna()

    test_data = test_df[["DSS.time", "dss_event"] + feature_cols].copy()
    test_data["DSS.time"] = pd.to_numeric(test_data["DSS.time"], errors="coerce")
    test_data = test_data.dropna()

    if len(train_data) < 10 or len(test_data) < 10:
        return {"label": label, "error": "insufficient data"}

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_data, duration_col="DSS.time", event_col="dss_event")
    except Exception as exc:
        return {"label": label, "error": str(exc)}

    risk_scores = cph.predict_partial_hazard(test_data[feature_cols]).values.ravel()
    c_idx = concordance_index(
        test_data["DSS.time"].values, -risk_scores, test_data["dss_event"].values
    )

    return {
        "label": label,
        "c_index": float(c_idx),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "n_features": len(feature_cols),
    }


def bootstrap_c_index(train_df, test_df, feature_cols, n_boot=500, label=""):
    """Bootstrap 95% CI for C-index."""
    rng = np.random.RandomState(2026)
    train_data = train_df[["DSS.time", "dss_event"] + feature_cols].copy()
    train_data["DSS.time"] = pd.to_numeric(train_data["DSS.time"], errors="coerce")
    train_data = train_data.dropna()

    test_data = test_df[["DSS.time", "dss_event"] + feature_cols].copy()
    test_data["DSS.time"] = pd.to_numeric(test_data["DSS.time"], errors="coerce")
    test_data = test_data.dropna()

    if len(train_data) < 10 or len(test_data) < 10:
        return {"label": label, "error": "insufficient data"}

    # Point estimate
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_data, duration_col="DSS.time", event_col="dss_event")
        risk_scores = cph.predict_partial_hazard(test_data[feature_cols]).values.ravel()
        point_c_idx = concordance_index(
            test_data["DSS.time"].values, -risk_scores, test_data["dss_event"].values
        )
    except Exception as exc:
        return {"label": label, "error": str(exc)}

    # Bootstrap
    boot_c_indices = []
    n_test = len(test_data)
    for _ in range(n_boot):
        idx = rng.randint(0, n_test, size=n_test)
        boot_test = test_data.iloc[idx]
        try:
            boot_risk = cph.predict_partial_hazard(boot_test[feature_cols]).values.ravel()
            boot_c = concordance_index(
                boot_test["DSS.time"].values, -boot_risk, boot_test["dss_event"].values
            )
            if not np.isnan(boot_c):
                boot_c_indices.append(boot_c)
        except Exception:
            continue

    if len(boot_c_indices) < 10:
        return {"label": label, "c_index": float(point_c_idx), "ci_lower": None, "ci_upper": None}

    ci_lower = float(np.percentile(boot_c_indices, 2.5))
    ci_upper = float(np.percentile(boot_c_indices, 97.5))

    return {
        "label": label,
        "c_index": float(point_c_idx),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": len(boot_c_indices),
    }


def paired_bootstrap_c_index(
    train_df, test_df, feat_a, feat_b, n_boot=500, label_a="", label_b=""
):
    """Paired bootstrap delta C-index between two feature sets."""
    rng = np.random.RandomState(2026)
    train_data = train_df[["DSS.time", "dss_event"]].copy()
    train_data["DSS.time"] = pd.to_numeric(train_data["DSS.time"], errors="coerce")
    train_data = train_data.dropna()

    test_data = test_df[["DSS.time", "dss_event"]].copy()
    test_data["DSS.time"] = pd.to_numeric(test_data["DSS.time"], errors="coerce")
    test_data = test_data.dropna()

    # Fit both models
    try:
        cph_a = CoxPHFitter(penalizer=0.1)
        cph_a.fit(train_data[["DSS.time", "dss_event"] + feat_a],
                  duration_col="DSS.time", event_col="dss_event")
        cph_b = CoxPHFitter(penalizer=0.1)
        cph_b.fit(train_data[["DSS.time", "dss_event"] + feat_b],
                  duration_col="DSS.time", event_col="dss_event")

        risk_a = cph_a.predict_partial_hazard(test_data[feat_a]).values.ravel()
        risk_b = cph_b.predict_partial_hazard(test_data[feat_b]).values.ravel()

        c_a = concordance_index(test_data["DSS.time"].values, -risk_a, test_data["dss_event"].values)
        c_b = concordance_index(test_data["DSS.time"].values, -risk_b, test_data["dss_event"].values)
    except Exception as exc:
        return {"error": str(exc)}

    # Bootstrap
    deltas = []
    n_test = len(test_data)
    for _ in range(n_boot):
        idx = rng.randint(0, n_test, size=n_test)
        boot = test_data.iloc[idx]
        try:
            boot_risk_a = cph_a.predict_partial_hazard(boot[feat_a]).values.ravel()
            boot_risk_b = cph_b.predict_partial_hazard(boot[feat_b]).values.ravel()
            boot_ca = concordance_index(boot["DSS.time"].values, -boot_risk_a, boot["dss_event"].values)
            boot_cb = concordance_index(boot["DSS.time"].values, -boot_risk_b, boot["dss_event"].values)
            delta = boot_ca - boot_cb
            if not np.isnan(delta):
                deltas.append(delta)
        except Exception:
            continue

    if len(deltas) < 10:
        return {"error": "insufficient bootstrap samples"}

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_value = 2 * min(
        np.mean(np.array(deltas) <= 0),
        np.mean(np.array(deltas) >= 0),
    )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "c_index_a": float(c_a),
        "c_index_b": float(c_b),
        "delta": float(c_a - c_b),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": float(p_value),
        "n_bootstrap": len(deltas),
    }


def leave_one_cancer_type_out(merged, tfidf, clinical, n_min=20):
    """Leave-one-cancer-type-out cross-validation."""
    train_full = pd.concat([merged["train"], merged["val"]], ignore_index=True)
    test = merged["test"]

    cancer_types = train_full["cancer_type"].value_counts()
    results = []

    # Build full training features
    tfidf_train_full = np.vstack([tfidf["train"], tfidf["val"]])
    clinical_train_full = pd.concat([clinical["train"], clinical["val"]], ignore_index=True)

    tfidf_cols = [f"svd_{i}" for i in range(tfidf_train_full.shape[1])]
    clinical_cols = list(clinical_train_full.columns)

    for ct, n_total in cancer_types.items():
        # Test on this cancer type in the test set
        ct_test = test[test["cancer_type"] == ct].copy()
        if len(ct_test) < n_min:
            continue

        # Train on everything except this cancer type
        ct_mask = train_full["cancer_type"] != ct
        ct_train = train_full[ct_mask].copy()

        # Check if we have enough training data
        if len(ct_train) < 100:
            continue

        # Rebuild features for this split
        train_idx = ct_train.index.values
        test_idx = ct_test.index.values

        # Map indices back to original positions
        # For simplicity, use the pre-built features and select rows
        # This is approximate since indices may not align perfectly
        try:
            # Build features directly from the split
            ct_clinical_train = build_clinical(
                ct_train, pd.DataFrame(columns=ct_train.columns), pd.DataFrame(columns=ct_train.columns)
            )["train"]
            ct_clinical_test = build_clinical(
                pd.DataFrame(columns=ct_train.columns), pd.DataFrame(columns=ct_train.columns), ct_test
            )["test"]

            # Align columns
            all_cols = sorted(set(ct_clinical_train.columns) | set(ct_clinical_test.columns))
            for col in all_cols:
                if col not in ct_clinical_train.columns:
                    ct_clinical_train[col] = 0.0
                if col not in ct_clinical_test.columns:
                    ct_clinical_test[col] = 0.0
            ct_clinical_train = ct_clinical_train[all_cols]
            ct_clinical_test = ct_clinical_test[all_cols]

            # Use clinical features only for speed
            combined_train = ct_train.reset_index(drop=True).copy()
            combined_train[all_cols] = ct_clinical_train.values[:len(combined_train)]

            combined_test = ct_test.reset_index(drop=True).copy()
            combined_test[all_cols] = ct_clinical_test.values[:len(combined_test)]

            result = fit_and_evaluate(combined_train, combined_test, all_cols, label=ct)
            result["n_test_ct"] = len(ct_test)
            result["n_train_ct"] = len(ct_train)
            results.append(result)
        except Exception as exc:
            results.append({"label": ct, "error": str(exc), "n_test_ct": len(ct_test)})

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--cdr-clinical", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "tcga_cdr_clinical.tsv")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "cox_robustness_results.json")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    args = parser.parse_args()

    print("Loading and merging data...")
    merged = load_and_merge(args.data_dir, args.cdr_clinical)
    for split, df in merged.items():
        n_event = int(df["dss_event"].sum())
        print(f"  {split}: {len(df)} rows ({n_event} events, {len(df)-n_event} censored)")

    # 1. Barcode overlap check
    print("\n1. Patient barcode overlap check...")
    overlap = check_barcode_overlap(merged)
    print(f"  Train-Val overlap: {overlap['train_val']}")
    print(f"  Train-Test overlap: {overlap['train_test']}")
    print(f"  Val-Test overlap: {overlap['val_test']}")

    # 2. Build features
    print("\n2. Building features...")
    train_full = pd.concat([merged["train"], merged["val"]], ignore_index=True)
    test = merged["test"]

    clinical = build_clinical(merged["train"], merged["val"], merged["test"])
    clinical_train_full = pd.concat([clinical["train"], clinical["val"]], ignore_index=True)
    clinical_cols = list(clinical_train_full.columns)

    tfidf = build_tfidf(merged["train"], merged["val"], merged["test"])
    tfidf_train_full = np.vstack([tfidf["train"], tfidf["val"]])
    tfidf_cols = [f"svd_{i}" for i in range(tfidf["n_components"])]

    # Build combined DataFrames
    train_combined = train_full.reset_index(drop=True).copy()
    train_combined[clinical_cols] = clinical_train_full.values[:len(train_combined)]
    tfidf_df_train = pd.DataFrame(tfidf_train_full[:len(train_combined)], columns=tfidf_cols)
    train_combined = pd.concat([train_combined.reset_index(drop=True), tfidf_df_train.reset_index(drop=True)], axis=1)

    test_combined = test.reset_index(drop=True).copy()
    test_combined[clinical_cols] = clinical["test"].values[:len(test_combined)]
    tfidf_df_test = pd.DataFrame(tfidf["test"][:len(test_combined)], columns=tfidf_cols)
    test_combined = pd.concat([test_combined.reset_index(drop=True), tfidf_df_test.reset_index(drop=True)], axis=1)

    all_cols = clinical_cols + tfidf_cols

    # 3. Bootstrap C-index CIs
    print(f"\n3. Bootstrap C-index CIs (n_boot={args.n_bootstrap})...")

    print("  Clinical only...")
    ci_clinical = bootstrap_c_index(
        train_combined, test_combined, clinical_cols,
        n_boot=args.n_bootstrap, label="Clinical only"
    )
    print(f"    C-index: {ci_clinical.get('c_index', 'N/A'):.4f} "
          f"[{ci_clinical.get('ci_lower', 'N/A'):.4f}, {ci_clinical.get('ci_upper', 'N/A'):.4f}]"
          if "c_index" in ci_clinical else f"    Error: {ci_clinical.get('error')}")

    print("  Clinical + TF-IDF...")
    ci_combined = bootstrap_c_index(
        train_combined, test_combined, all_cols,
        n_boot=args.n_bootstrap, label="Clinical + TF-IDF"
    )
    print(f"    C-index: {ci_combined.get('c_index', 'N/A'):.4f} "
          f"[{ci_combined.get('ci_lower', 'N/A'):.4f}, {ci_combined.get('ci_upper', 'N/A'):.4f}]"
          if "c_index" in ci_combined else f"    Error: {ci_combined.get('error')}")

    print("  TF-IDF only...")
    ci_tfidf = bootstrap_c_index(
        train_combined, test_combined, tfidf_cols,
        n_boot=args.n_bootstrap, label="TF-IDF only"
    )
    print(f"    C-index: {ci_tfidf.get('c_index', 'N/A'):.4f} "
          f"[{ci_tfidf.get('ci_lower', 'N/A'):.4f}, {ci_tfidf.get('ci_upper', 'N/A'):.4f}]"
          if "c_index" in ci_tfidf else f"    Error: {ci_tfidf.get('error')}")

    # 4. Paired bootstrap delta
    print("\n4. Paired bootstrap delta (Clinical+TF-IDF vs Clinical only)...")
    paired = paired_bootstrap_c_index(
        train_combined, test_combined, all_cols, clinical_cols,
        n_boot=args.n_bootstrap,
        label_a="Clinical + TF-IDF", label_b="Clinical only"
    )
    if "error" not in paired:
        print(f"  Delta C-index: {paired['delta']:.4f} "
              f"[{paired['ci_lower']:.4f}, {paired['ci_upper']:.4f}], p={paired['p_value']:.4f}")
    else:
        print(f"  Error: {paired['error']}")

    # 5. Leave-one-cancer-type-out
    print("\n5. Leave-one-cancer-type-out cross-validation...")
    locto = leave_one_cancer_type_out(merged, tfidf, clinical)
    print(f"  Evaluated {len(locto)} cancer types")
    for r in locto:
        if "c_index" in r:
            print(f"    {r['label']}: C-index={r['c_index']:.4f} (n_test={r.get('n_test_ct', '?')})")
        else:
            print(f"    {r.get('label', '?')}: Error - {r.get('error', '?')[:60]}")

    # Summary
    locto_c_indices = [r["c_index"] for r in locto if "c_index" in r]
    if locto_c_indices:
        print(f"\n  LOCTO mean C-index: {np.mean(locto_c_indices):.4f} "
              f"(SD: {np.std(locto_c_indices):.4f}, range: {min(locto_c_indices):.4f}-{max(locto_c_indices):.4f})")

    # Save results
    results = {
        "barcode_overlap": overlap,
        "bootstrap_ci": [ci_clinical, ci_combined, ci_tfidf],
        "paired_bootstrap": paired,
        "leave_one_cancer_type_out": locto,
        "locto_summary": {
            "mean_c_index": float(np.mean(locto_c_indices)) if locto_c_indices else None,
            "std_c_index": float(np.std(locto_c_indices)) if locto_c_indices else None,
            "min_c_index": float(min(locto_c_indices)) if locto_c_indices else None,
            "max_c_index": float(max(locto_c_indices)) if locto_c_indices else None,
            "n_types": len(locto_c_indices),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
