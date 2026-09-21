#!/usr/bin/env python
"""Cox proportional hazards survival modeling for pathology report prognosis.

This script replaces the binary threshold prognosis approach with proper
time-to-event modeling using DSS time and censoring information from TCGA-CDR.

Evaluates multiple feature sets:
1. Clinical variables only (age, gender, cancer type)
2. Clinical + DeepSeek-extracted structured variables
3. Clinical + TF-IDF text features (dimensionality reduced)
4. Clinical + DeepSeek variables + TF-IDF text features

Reports C-index, integrated Brier score, and hazard ratios for key variables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
try:
    from lifelines.metrics import integrated_brier_score as _ibs
except ImportError:
    _ibs = None
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data loading and merging
# ---------------------------------------------------------------------------

def load_pathrep_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load PathRep-Bench enriched CSVs."""
    splits = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"tcga_pathology_{split}.csv"
        splits[split] = pd.read_csv(path, encoding="utf-8-sig")
    return splits


def load_cdr_clinical(path: Path) -> pd.DataFrame:
    """Load TCGA-CDR clinical data with censoring info."""
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    # Normalize patient ID for merging
    df = df.rename(columns={"Patient ID": "bcr_patient_barcode"})
    # Extract DSS status (event indicator)
    # 'DEAD WITH TUMOR' = event (1), 'ALIVE OR DEAD TUMOR FREE' = censored (0)
    dss_map = {
        "DEAD WITH TUMOR": 1,
        "ALIVE OR DEAD TUMOR FREE": 0,
    }
    df["dss_event"] = df["Disease-specific Survival status"].map(dss_map)
    # Convert DSS time from months to days for consistency with PathRep-Bench
    df["dss_time_months"] = pd.to_numeric(
        df["Months of disease-specific survival"], errors="coerce"
    )
    df["dss_time_days"] = df["dss_time_months"] * 30.4375
    # Keep only relevant columns
    cols = ["bcr_patient_barcode", "dss_event", "dss_time_days", "dss_time_months"]
    extra = ["Diagnosis Age", "Sex", "Cancer Type"]
    for c in extra:
        if c in df.columns:
            cols.append(c)
    return df[[c for c in cols if c in df.columns]].copy()


def merge_survival_info(
    pathrep: dict[str, pd.DataFrame], cdr: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Merge PathRep-Bench data with CDR censoring info."""
    merged = {}
    for split, df in pathrep.items():
        # Merge on patient barcode
        merged_df = df.merge(
            cdr[["bcr_patient_barcode", "dss_event"]],
            on="bcr_patient_barcode",
            how="left",
        )
        # Use CDR event if available, otherwise infer from DSS.time presence
        # If DSS.time is present and we can't determine censoring, assume event=1
        # (conservative: treats all as events, which biases toward event)
        # Better: only keep rows with known censoring status
        merged_df["dss_event"] = merged_df["dss_event"].fillna(-1).astype(int)
        # Keep only rows with known DSS time and known event status
        merged_df = merged_df[
            (merged_df["DSS.time"].notna())
            & (merged_df["DSS.time"] > 0)
            & (merged_df["dss_event"] >= 0)
        ].copy()
        merged_df["DSS.time"] = pd.to_numeric(merged_df["DSS.time"], errors="coerce")
        merged_df = merged_df.dropna(subset=["DSS.time"])
        merged[split] = merged_df
    return merged


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_tfidf_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_components: int = 100,
) -> dict[str, np.ndarray]:
    """Build TF-IDF + SVD features from pathology report text."""
    text_col = "report_text"
    # Handle missing text
    train_text = train_df[text_col].fillna("").astype(str)
    test_text = test_df[text_col].fillna("").astype(str)
    val_text = val_df[text_col].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=50000,
    )
    train_tfidf = vectorizer.fit_transform(train_text)
    val_tfidf = vectorizer.transform(val_text)
    test_tfidf = vectorizer.transform(test_text)

    # Dimensionality reduction with TruncatedSVD
    n_comp = min(n_components, min(train_tfidf.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=2026)
    train_svd = svd.fit_transform(train_tfidf)
    val_svd = svd.transform(val_tfidf)
    test_svd = svd.transform(test_tfidf)

    # Scale features
    scaler = StandardScaler()
    train_svd = scaler.fit_transform(train_svd)
    val_svd = scaler.transform(val_svd)
    test_svd = scaler.transform(test_svd)

    return {
        "train": train_svd,
        "val": val_svd,
        "test": test_svd,
        "n_components": n_comp,
    }


def load_deepseek_variables(jsonl_path: Path) -> pd.DataFrame | None:
    """Load DeepSeek-extracted structured variables from JSONL."""
    if not jsonl_path or not jsonl_path.exists():
        return None
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "parsed" in rec and rec["parsed"]:
                    records.append(rec["parsed"])
            except json.JSONDecodeError:
                continue
    if not records:
        return None
    df = pd.DataFrame(records)
    return df


def build_clinical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build clinical feature matrices (age, gender, cancer type one-hot)."""
    splits = {"train": train_df, "val": val_df, "test": test_df}
    result = {}

    # Age
    for name, df in splits.items():
        feat = pd.DataFrame()
        feat["age"] = pd.to_numeric(
            df["age_at_initial_pathologic_diagnosis"], errors="coerce"
        ).fillna(df["age_at_initial_pathologic_diagnosis"].apply(
            lambda x: float(x) if pd.notna(x) else np.nan
        ))
        feat["age"] = feat["age"].fillna(feat["age"].median() if feat["age"].notna().any() else 60.0)

        # Gender (binary)
        feat["is_male"] = (df["gender"].str.upper() == "MALE").astype(int)

        # Cancer type one-hot
        ct = pd.get_dummies(df["cancer_type"], prefix="ct").astype(float)
        feat = pd.concat([feat, ct], axis=1)

        result[name] = feat

    # Ensure all splits have same columns
    all_cols = set()
    for df in result.values():
        all_cols.update(df.columns)
    for name in result:
        for col in all_cols:
            if col not in result[name].columns:
                result[name][col] = 0.0
        result[name] = result[name][sorted(all_cols)]

    return result


# ---------------------------------------------------------------------------
# Cox model fitting and evaluation
# ---------------------------------------------------------------------------

def fit_cox_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    duration_col: str = "DSS.time",
    event_col: str = "dss_event",
    label: str = "",
) -> dict:
    """Fit Cox PH model and evaluate on test set."""
    # Prepare training data
    train_data = train_df[[duration_col, event_col] + feature_cols].copy()
    train_data[duration_col] = pd.to_numeric(train_data[duration_col], errors="coerce")
    train_data = train_data.dropna()

    # Prepare test data
    test_data = test_df[[duration_col, event_col] + feature_cols].copy()
    test_data[duration_col] = pd.to_numeric(test_data[duration_col], errors="coerce")
    test_data = test_data.dropna()

    if len(train_data) < 10 or len(test_data) < 10:
        return {"error": "insufficient data", "label": label}

    # Fit Cox model
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_data, duration_col=duration_col, event_col=event_col)
    except Exception as exc:
        return {"error": str(exc), "label": label}

    # Evaluate on test set
    test_features = test_data[feature_cols].copy()
    test_durations = test_data[duration_col].values
    test_events = test_data[event_col].values

    # Predict partial hazards (risk scores)
    risk_scores = cph.predict_partial_hazard(test_features).values.ravel()

    # C-index
    c_idx = concordance_index(
        test_durations, -risk_scores, test_events
    )

    # Integrated Brier Score (requires survival function predictions)
    try:
        # Get survival function predictions at multiple time points
        survival_funcs = cph.predict_survival_function(test_features)
        # Use a subset of time points for IBS
        time_grid = np.linspace(
            test_durations.min(),
            test_durations.max(),
            min(50, len(test_durations)),
        )
        surv_preds = np.zeros((len(test_data), len(time_grid)))
        for i, (_, sf) in enumerate(survival_funcs.iterrows()):
            for j, t in enumerate(time_grid):
                surv_preds[i, j] = sf.loc[t] if t in sf.index else sf.iloc[
                    np.searchsorted(sf.index.values, t, side="right") - 1
                ] if np.searchsorted(sf.index.values, t, side="right") > 0 else 1.0

        # Compute IBS using lifelines
        # Train baseline survival for IBS
        train_baseline = cph.baseline_survival_
        ibs = None
        if _ibs is not None:
            try:
                ibs = _ibs(
                    test_durations,
                    test_events,
                    surv_preds,
                    time_grid,
                )
            except Exception:
                ibs = None
    except Exception:
        ibs = None

    # Get hazard ratios for top features
    hazard_ratios = {}
    summary = cph.summary
    for feat in feature_cols[:10]:  # Top 10 features
        if feat in summary.index:
            hazard_ratios[feat] = {
                "hr": float(summary.loc[feat, "exp(coef)"]),
                "p": float(summary.loc[feat, "p"]),
                "ci_lower": float(summary.loc[feat, "exp(coef) lower 95%"]),
                "ci_upper": float(summary.loc[feat, "exp(coef) upper 95%"]),
            }

    return {
        "label": label,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "n_features": len(feature_cols),
        "c_index": float(c_idx),
        "ibs": float(ibs) if ibs is not None else None,
        "hazard_ratios": hazard_ratios,
        "concordance": float(c_idx),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory with PathRep-Bench enriched CSVs.",
    )
    parser.add_argument(
        "--cdr-clinical",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "tcga_cdr_clinical.tsv",
        help="TCGA-CDR clinical data file with censoring info.",
    )
    parser.add_argument(
        "--deepseek-features-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Directory with DeepSeek-extracted variable JSONL files.",
    )
    parser.add_argument(
        "--n-svd-components",
        type=int,
        default=100,
        help="Number of SVD components for TF-IDF features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "cox_survival_results.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    print("Loading PathRep-Bench data...")
    pathrep = load_pathrep_data(args.data_dir)
    for split, df in pathrep.items():
        print(f"  {split}: {len(df)} rows")

    print("Loading TCGA-CDR clinical data...")
    cdr = load_cdr_clinical(args.cdr_clinical)
    print(f"  CDR rows: {len(cdr)}")
    print(f"  With DSS event: {cdr['dss_event'].sum()}/{len(cdr)}")

    print("Merging survival info...")
    merged = merge_survival_info(pathrep, cdr)
    for split, df in merged.items():
        n_event = int(df["dss_event"].sum())
        n_censored = len(df) - n_event
        print(f"  {split}: {len(df)} rows ({n_event} events, {n_censored} censored)")

    # Combine train + val for training
    train_df = pd.concat([merged["train"], merged["val"]], ignore_index=True)
    test_df = merged["test"]

    print(f"\nTraining set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")

    # Build features
    print("\nBuilding clinical features...")
    clinical = build_clinical_features(
        merged["train"], merged["val"], merged["test"]
    )
    clinical_train = clinical["train"]
    clinical_val = clinical["val"]
    clinical_test = clinical["test"]

    # Reindex to match merged train/test
    clinical_train_full = pd.concat([clinical_train, clinical_val], ignore_index=True)

    print(f"  Clinical features: {clinical_train_full.shape[1]} columns")

    print("\nBuilding TF-IDF + SVD features...")
    tfidf = build_tfidf_features(
        merged["train"], merged["val"], merged["test"],
        n_components=args.n_svd_components,
    )
    tfidf_train_full = np.vstack([tfidf["train"], tfidf["val"]])
    print(f"  TF-IDF SVD components: {tfidf['n_components']}")

    # Load DeepSeek variables if available
    deepseek_dfs = {}
    for split in ["train", "val", "test"]:
        jsonl_path = args.deepseek_features_dir / f"deepseek_variables_{split}_full.jsonl"
        ds_df = load_deepseek_variables(jsonl_path)
        if ds_df is not None:
            print(f"  DeepSeek variables ({split}): {ds_df.shape}")
            deepseek_dfs[split] = ds_df
        else:
            print(f"  DeepSeek variables ({split}): not found at {jsonl_path}")

    # Build combined feature matrices for each ablation
    results = []

    # 1. Clinical only
    print("\n--- Model 1: Clinical variables only ---")
    feat_cols_clinical = list(clinical_train_full.columns)
    train_combined = train_df.reset_index(drop=True).copy()
    train_combined[feat_cols_clinical] = clinical_train_full.values
    test_combined = test_df.reset_index(drop=True).copy()
    test_combined[feat_cols_clinical] = clinical_test.values[:len(test_combined)]
    result = fit_cox_model(
        train_combined, test_combined, feat_cols_clinical, label="Clinical only"
    )
    print(f"  C-index: {result.get('c_index', 'N/A'):.4f}" if "c_index" in result else f"  Error: {result.get('error')}")
    results.append(result)

    # 2. Clinical + TF-IDF
    print("\n--- Model 2: Clinical + TF-IDF text features ---")
    tfidf_cols = [f"svd_{i}" for i in range(tfidf_train_full.shape[1])]
    train_combined2 = train_df.reset_index(drop=True).copy()
    train_combined2[feat_cols_clinical] = clinical_train_full.values[:len(train_combined2)]
    for i, col in enumerate(tfidf_cols):
        train_combined2[col] = tfidf_train_full[:len(train_combined2), i]
    test_combined2 = test_df.reset_index(drop=True).copy()
    test_combined2[feat_cols_clinical] = clinical_test.values[:len(test_combined2)]
    for i, col in enumerate(tfidf_cols):
        test_combined2[col] = tfidf["test"][:len(test_combined2), i]

    feat_cols_2 = feat_cols_clinical + tfidf_cols
    result = fit_cox_model(
        train_combined2, test_combined2, feat_cols_2, label="Clinical + TF-IDF"
    )
    print(f"  C-index: {result.get('c_index', 'N/A'):.4f}" if "c_index" in result else f"  Error: {result.get('error')}")
    results.append(result)

    # 3. Clinical + DeepSeek variables (if available)
    if deepseek_dfs:
        print("\n--- Model 3: Clinical + DeepSeek variables ---")
        # Process DeepSeek variables into features
        ds_train = pd.concat([
            deepseek_dfs.get("train", pd.DataFrame()),
            deepseek_dfs.get("val", pd.DataFrame()),
        ], ignore_index=True)
        ds_test = deepseek_dfs.get("test", pd.DataFrame())

        # Select numeric/categorical columns that can be used as features
        ds_feature_cols = []
        for col in ds_train.columns:
            if col in ["index", "id", "error", "raw_content"]:
                continue
            # Try to convert to numeric
            try:
                ds_train[col] = pd.to_numeric(ds_train[col], errors="coerce")
                ds_test[col] = pd.to_numeric(ds_test[col], errors="coerce")
                if ds_train[col].notna().sum() > 10:
                    ds_feature_cols.append(col)
            except Exception:
                pass

        if ds_feature_cols:
            # Fill missing values
            for col in ds_feature_cols:
                median_val = ds_train[col].median()
                if pd.isna(median_val):
                    median_val = 0
                ds_train[col] = ds_train[col].fillna(median_val)
                ds_test[col] = ds_test[col].fillna(median_val)

            train_combined3 = train_df.reset_index(drop=True).copy()
            train_combined3[feat_cols_clinical] = clinical_train_full.values[:len(train_combined3)]
            for col in ds_feature_cols:
                train_combined3[col] = ds_train[col].values[:len(train_combined3)]
            test_combined3 = test_df.reset_index(drop=True).copy()
            test_combined3[feat_cols_clinical] = clinical_test.values[:len(test_combined3)]
            for col in ds_feature_cols:
                test_combined3[col] = ds_test[col].values[:len(test_combined3)]

            feat_cols_3 = feat_cols_clinical + ds_feature_cols
            result = fit_cox_model(
                train_combined3, test_combined3, feat_cols_3,
                label="Clinical + DeepSeek variables"
            )
            print(f"  C-index: {result.get('c_index', 'N/A'):.4f}" if "c_index" in result else f"  Error: {result.get('error')}")
            results.append(result)

            # 4. Clinical + DeepSeek + TF-IDF
            print("\n--- Model 4: Clinical + DeepSeek + TF-IDF ---")
            train_combined4 = train_combined3.copy()
            test_combined4 = test_combined3.copy()
            for i, col in enumerate(tfidf_cols):
                train_combined4[col] = tfidf_train_full[:len(train_combined4), i]
                test_combined4[col] = tfidf["test"][:len(test_combined4), i]
            feat_cols_4 = feat_cols_3 + tfidf_cols
            result = fit_cox_model(
                train_combined4, test_combined4, feat_cols_4,
                label="Clinical + DeepSeek + TF-IDF"
            )
            print(f"  C-index: {result.get('c_index', 'N/A'):.4f}" if "c_index" in result else f"  Error: {result.get('error')}")
            results.append(result)
        else:
            print("  No usable DeepSeek variable features found.")
    else:
        print("\n  DeepSeek variables not available. Skipping Models 3 and 4.")
        print("  Run extract_deepseek_variables.py first to generate them.")

    # 5. TF-IDF only (no clinical)
    print("\n--- Model 5: TF-IDF text features only ---")
    train_combined5 = train_df.reset_index(drop=True).copy()
    for i, col in enumerate(tfidf_cols):
        train_combined5[col] = tfidf_train_full[:len(train_combined5), i]
    test_combined5 = test_df.reset_index(drop=True).copy()
    for i, col in enumerate(tfidf_cols):
        test_combined5[col] = tfidf["test"][:len(test_combined5), i]
    result = fit_cox_model(
        train_combined5, test_combined5, tfidf_cols, label="TF-IDF only"
    )
    print(f"  C-index: {result.get('c_index', 'N/A'):.4f}" if "c_index" in result else f"  Error: {result.get('error')}")
    results.append(result)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Print summary table
    print("\n" + "=" * 70)
    print("COX PROPORTIONAL HAZARDS SURVIVAL MODEL RESULTS")
    print("=" * 70)
    print(f"{'Model':<40} {'C-index':>10} {'IBS':>10} {'N_train':>8} {'N_test':>8}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<40} {'ERROR':>10}")
        else:
            ibs_str = f"{r['ibs']:.4f}" if r.get("ibs") is not None else "N/A"
            print(f"{r['label']:<40} {r['c_index']:>10.4f} {ibs_str:>10} {r['n_train']:>8} {r['n_test']:>8}")

    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
