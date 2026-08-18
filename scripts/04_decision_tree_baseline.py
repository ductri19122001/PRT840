from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


INPUT_FILE = Path("outputs/clean_network_flows.csv")
FEATURE_RECOMMENDATIONS_FILE = Path("outputs/feature_analysis/feature_relevance_recommendations.csv")
METRICS_OUTPUT_FILE = Path("outputs/decision_tree_baseline_metrics.csv")
PREDICTIONS_OUTPUT_FILE = Path("outputs/decision_tree_baseline_predictions.csv")

DEFAULT_NUMERICAL_FEATURES = [
    "id.orig_p",
    "id.resp_p",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]
DEFAULT_CATEGORICAL_FEATURES = ["proto", "service", "conn_state", "history"]
TARGET_COLUMN = "label"
ROW_ID_COLUMN = "flow_row_id"


def load_approved_features() -> tuple[list[str], list[str]]:
    if not FEATURE_RECOMMENDATIONS_FILE.exists():
        return DEFAULT_NUMERICAL_FEATURES, DEFAULT_CATEGORICAL_FEATURES

    recommendations = pd.read_csv(FEATURE_RECOMMENDATIONS_FILE)
    approved = recommendations.loc[recommendations["recommendation"] != "DROP", "feature"].tolist()
    numerical = [feature for feature in DEFAULT_NUMERICAL_FEATURES if feature in approved]
    categorical = [feature for feature in DEFAULT_CATEGORICAL_FEATURES if feature in approved]
    return numerical, categorical


def build_pipeline(numerical_features: list[str], categorical_features: list[str]) -> Pipeline:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": false_positive_rate,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }
    return metrics


def main() -> None:
    dataframe = pd.read_csv(INPUT_FILE)
    numerical_features, categorical_features = load_approved_features()
    final_features = numerical_features + categorical_features

    X = dataframe.loc[:, final_features].copy()
    y = dataframe[TARGET_COLUMN].astype(int)
    row_ids = dataframe[ROW_ID_COLUMN].copy()

    X_train, X_test, y_train, y_test, row_ids_train, row_ids_test = train_test_split(
        X,
        y,
        row_ids,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline(numerical_features, categorical_features)
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        y_train_prob = pipeline.predict_proba(X_train)[:, 1]
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_train_prob = pd.Series(y_train_pred, index=y_train.index, dtype="float64")
        y_test_prob = pd.Series(y_test_pred, index=y_test.index, dtype="float64")

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

    metrics_rows = [
        {"metric": "train_accuracy", "value": train_accuracy},
        {"metric": "train_f1_score", "value": train_f1},
        {"metric": "test_accuracy", "value": test_metrics["accuracy"]},
        {"metric": "test_precision", "value": test_metrics["precision"]},
        {"metric": "test_recall", "value": test_metrics["recall"]},
        {"metric": "test_f1_score", "value": test_metrics["f1_score"]},
        {"metric": "test_false_positive_rate", "value": test_metrics["false_positive_rate"]},
        {"metric": "test_roc_auc", "value": test_metrics["roc_auc"]},
        {"metric": "true_negatives", "value": test_metrics["true_negatives"]},
        {"metric": "false_positives", "value": test_metrics["false_positives"]},
        {"metric": "false_negatives", "value": test_metrics["false_negatives"]},
        {"metric": "true_positives", "value": test_metrics["true_positives"]},
        {"metric": "train_samples", "value": len(X_train)},
        {"metric": "test_samples", "value": len(X_test)},
    ]
    pd.DataFrame(metrics_rows).to_csv(METRICS_OUTPUT_FILE, index=False)

    predictions = pd.DataFrame(
        {
            "flow_row_id": row_ids_test.to_numpy(),
            "true_label": y_test.to_numpy(),
            "predicted_label": y_test_pred,
            "predicted_probability": y_test_prob,
        }
    ).sort_values("flow_row_id")
    predictions.to_csv(PREDICTIONS_OUTPUT_FILE, index=False)

    train_distribution = y_train.value_counts().sort_index()
    test_distribution = y_test.value_counts().sort_index()

    print("Decision Tree binary baseline")
    print(f"Input dataset: {INPUT_FILE}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("\nTrain class distribution:")
    print(train_distribution.to_string())
    print("\nTest class distribution:")
    print(test_distribution.to_string())
    print("\nFinal feature list used:")
    print(final_features)
    print("\nEvaluation metrics:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train F1-score: {train_f1:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-score: {test_metrics['f1_score']:.4f}")
    print(f"Test False Positive Rate: {test_metrics['false_positive_rate']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print("\nConfusion matrix [TN FP; FN TP]:")
    print(
        f"{test_metrics['true_negatives']} {test_metrics['false_positives']}\n"
        f"{test_metrics['false_negatives']} {test_metrics['true_positives']}"
    )
    print("\nClassification report:")
    print(test_metrics["classification_report"])
    print(f"Saved metrics to: {METRICS_OUTPUT_FILE}")
    print(f"Saved predictions to: {PREDICTIONS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
