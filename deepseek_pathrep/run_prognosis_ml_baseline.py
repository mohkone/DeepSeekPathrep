#!/usr/bin/env python
"""Train a local text+clinical prognosis classifier on PathRep-Bench splits."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


LABELS = ["poor", "good"]


def read_csv(path: Path, source: str) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for index, row in enumerate(csv.DictReader(handle)):
            row["__feature_source"] = source
            row["__row_index"] = str(index)
            rows.append(row)
        return rows


def read_jsonl(path: Path | None, source: str) -> dict:
    if not path:
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("index") is not None:
            rows[f"{source}:index:{row['index']}"] = row.get("variables") or {}
        if row.get("id") not in (None, ""):
            rows[f"id:{row['id']}"] = row.get("variables") or {}
    return rows


def clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_days(value):
    if value in (None, ""):
        return 0.0
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else 0.0


def label_value(row: dict) -> str:
    return clean_text(row.get("prognosis")).lower()


def keep_labeled(rows: list[dict]) -> list[dict]:
    return [row for row in rows if label_value(row) in LABELS]


def extracted_for_row(row: dict, extracted: dict) -> dict | None:
    source = row.get("__feature_source", "")
    index = row.get("__row_index", "")
    return extracted.get(f"{source}:index:{index}") or extracted.get(f"id:{row.get('bcr_patient_barcode')}")


def structured_features(
    row: dict,
    extracted: dict | None = None,
    include_base_structured: bool = True,
    include_deepseek: bool = True,
) -> dict:
    features = {}
    if include_base_structured:
        features.update(
            {
                "cancer_type": clean_text(row.get("cancer_type") or row.get("type_name") or "unknown"),
                "ajcc_stage": clean_text(row.get("ajcc_stage") or row.get("stage_overall") or "unknown"),
                "gender": clean_text(row.get("gender") or "unknown").lower(),
                "race": clean_text(row.get("race") or "unknown").lower(),
                "age": float(row.get("age_at_initial_pathologic_diagnosis") or 0.0),
                "mean_dss_days": parse_days(row.get("mean_dss")),
            }
        )
    if extracted and include_deepseek:
        for key, value in extracted.items():
            if key == "key_evidence":
                continue
            if isinstance(value, (int, float)):
                features[f"deepseek_{key}"] = float(value)
            else:
                features[f"deepseek_{key}"] = clean_text(value or "unknown").lower()
    return features


def report_text(row: dict) -> str:
    return clean_text(row.get("report_text") or row.get("text") or "")


def transform_features(
    train_rows: list[dict],
    test_rows: list[dict],
    max_features: int,
    min_df: int,
    train_extracted: dict | None = None,
    test_extracted: dict | None = None,
    include_text: bool = True,
    include_base_structured: bool = True,
    include_deepseek: bool = True,
):
    matrices_train = []
    matrices_test = []
    if include_text:
        text_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )
        matrices_train.append(text_vectorizer.fit_transform(report_text(row) for row in train_rows))
        matrices_test.append(text_vectorizer.transform(report_text(row) for row in test_rows))

    train_extracted = train_extracted or {}
    test_extracted = test_extracted or {}
    if include_base_structured or include_deepseek:
        dict_vectorizer = DictVectorizer(sparse=True)
        x_train_structured = dict_vectorizer.fit_transform(
            structured_features(
                row,
                extracted_for_row(row, train_extracted),
                include_base_structured=include_base_structured,
                include_deepseek=include_deepseek,
            )
            for row in train_rows
        )
        x_test_structured = dict_vectorizer.transform(
            structured_features(
                row,
                extracted_for_row(row, test_extracted),
                include_base_structured=include_base_structured,
                include_deepseek=include_deepseek,
            )
            for row in test_rows
        )
        matrices_train.append(x_train_structured)
        matrices_test.append(x_test_structured)

    if not matrices_train:
        return csr_matrix((len(train_rows), 0)), csr_matrix((len(test_rows), 0))
    if len(matrices_train) == 1:
        return matrices_train[0], matrices_test[0]
    return hstack(matrices_train), hstack(matrices_test)


def write_predictions(path: Path, rows: list[dict], golds: list[str], preds: list[str], probs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, (row, gold, pred) in enumerate(zip(rows, golds, preds)):
            record = {
                "index": index,
                "id": row.get("bcr_patient_barcode"),
                "task": "prognosis",
                "model": "tfidf_logistic_regression",
                "gold": gold,
                "prediction": pred,
                "prob_good": float(probs[index, LABELS.index("good")]),
                "prob_poor": float(probs[index, LABELS.index("poor")]),
                "error": None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "tcga_pathology_train.csv")
    parser.add_argument("--val", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "tcga_pathology_val.csv")
    parser.add_argument("--test", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "tcga_pathology_test.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "ml_prognosis_tfidf_logreg_test.jsonl")
    parser.add_argument("--train-on-val", action="store_true", help="Include validation rows in training.")
    parser.add_argument("--deepseek-features-train", type=Path, help="Optional DeepSeek structured variables for train rows.")
    parser.add_argument("--deepseek-features-val", type=Path, help="Optional DeepSeek structured variables for validation rows.")
    parser.add_argument("--deepseek-features-test", type=Path, help="Optional DeepSeek structured variables for test rows.")
    parser.add_argument("--no-text", action="store_true", help="Disable TF-IDF report text features.")
    parser.add_argument("--no-base-structured", action="store_true", help="Disable original structured clinical fields.")
    parser.add_argument("--no-deepseek-features", action="store_true", help="Ignore DeepSeek extracted variables even if files are passed.")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c", type=float, default=2.0)
    args = parser.parse_args()

    train_rows = keep_labeled(read_csv(args.train, "train"))
    if args.train_on_val:
        train_rows += keep_labeled(read_csv(args.val, "val"))
    test_rows = keep_labeled(read_csv(args.test, "test"))

    y_train = [label_value(row) for row in train_rows]
    y_test = [label_value(row) for row in test_rows]
    train_extracted = read_jsonl(args.deepseek_features_train, "train")
    if args.train_on_val and args.deepseek_features_val:
        train_extracted.update(read_jsonl(args.deepseek_features_val, "val"))
    test_extracted = read_jsonl(args.deepseek_features_test, "test")
    deepseek_feature_warning = None
    if not args.no_deepseek_features and (train_extracted or test_extracted):
        if not train_extracted:
            deepseek_feature_warning = (
                "DeepSeek features were provided only for test rows; they cannot affect the trained model."
            )
        elif not test_extracted:
            deepseek_feature_warning = (
                "DeepSeek features were provided only for train rows; test rows will use missing/unknown values."
            )
    x_train, x_test = transform_features(
        train_rows,
        test_rows,
        args.max_features,
        args.min_df,
        train_extracted=train_extracted,
        test_extracted=test_extracted,
        include_text=not args.no_text,
        include_base_structured=not args.no_base_structured,
        include_deepseek=not args.no_deepseek_features,
    )

    model = LogisticRegression(
        C=args.c,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=2026,
    )
    model.fit(x_train, y_train)
    preds = list(model.predict(x_test))
    probs = model.predict_proba(x_test)

    metrics = {
        "model": "tfidf_logistic_regression",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "deepseek_features_train_rows": sum(1 for row in train_rows if extracted_for_row(row, train_extracted)),
        "deepseek_features_test_rows": sum(1 for row in test_rows if extracted_for_row(row, test_extracted)),
        "include_text": not args.no_text,
        "include_base_structured": not args.no_base_structured,
        "include_deepseek_features": not args.no_deepseek_features,
        "deepseek_feature_warning": deepseek_feature_warning,
        "train_label_counts": dict(Counter(y_train)),
        "test_label_counts": dict(Counter(y_test)),
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "confusion_labels": LABELS,
        "confusion_matrix": confusion_matrix(y_test, preds, labels=LABELS).tolist(),
        "classification_report": classification_report(y_test, preds, labels=LABELS, output_dict=True),
    }

    write_predictions(args.output, test_rows, y_test, preds, probs)
    metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
