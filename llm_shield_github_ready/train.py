from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DEFAULT_DATA = "sample_prompts.csv"
DEFAULT_MODEL = "model.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )



def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"prompt", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must contain columns: prompt,label")

    cleaned = df[["prompt", "label"]].copy()
    cleaned["prompt"] = cleaned["prompt"].astype(str).str.strip()
    cleaned = cleaned[cleaned["prompt"].ne("")]
    cleaned["label"] = cleaned["label"].astype(int)

    invalid_labels = set(cleaned["label"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Labels must be only 0 or 1. Found: {sorted(invalid_labels)}")
    if cleaned.empty:
        raise ValueError("Dataset is empty after cleaning.")
    return cleaned



def main() -> None:
    parser = argparse.ArgumentParser(description="Train the adversarial prompt detector.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to CSV with columns: prompt,label")
    parser.add_argument("--model-out", default=DEFAULT_MODEL, help="Output path for trained model bundle")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = validate_dataframe(pd.read_csv(data_path))
    X = df["prompt"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    report_text = classification_report(y_test, predictions, digits=4)
    matrix = confusion_matrix(y_test, predictions)

    payload = {
        "model": pipeline,
        "labels": {0: "safe", 1: "adversarial"},
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report_text,
            "confusion_matrix": matrix.tolist(),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "dataset_name": data_path.name,
        },
    }

    model_out = Path(args.model_out)
    joblib.dump(payload, model_out)

    print("Training complete")
    print(f"Saved model to: {model_out.resolve()}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Classification report:")
    print(report_text)
    print("Confusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
