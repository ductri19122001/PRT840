from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


INPUT_FILE = Path("outputs/clean_network_flows.csv")
METADATA_FILE = Path("outputs/cti_mapping_metadata.csv")
FEATURE_RECOMMENDATIONS_FILE = Path("outputs/feature_analysis/feature_relevance_recommendations.csv")
METRICS_OUTPUT_FILE = Path("outputs/isolation_forest_chronological_metrics.csv")
PREDICTIONS_OUTPUT_FILE = Path("outputs/isolation_forest_chronological_predictions.csv")

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
CAPTURE_DATE_COLUMN = "capture_date"

TRAIN_DATES = [
    "2023-02-20",
    "2023-02-21",
    "2023-02-22",
    "2023-02-23",
]

TEST_DATES = [
    "2023-02-24",
]


def load_approved_features() -> tuple[list[str], list[str]]:
    if not FEATURE_RECOMMENDATIONS_FILE.exists():
        return DEFAULT_NUMERICAL_FEATURES, DEFAULT_CATEGORICAL_FEATURES

    recommendations = pd.read_csv(FEATURE_RECOMMENDATIONS_FILE)
    approved = recommendations.loc[recommendations["recommendation"] != "DROP", "feature"].tolist()
    numerical = [feature for feature in DEFAULT_NUMERICAL_FEATURES if feature in approved]
    categorical = [feature for feature in DEFAULT_CATEGORICAL_FEATURES if feature in approved]
    return numerical, categorical


def build_preprocessor(numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
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

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, anomaly_scores: pd.Series) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    predicted_anomalies = int((y_pred == 1).sum())
    predicted_anomaly_rate = predicted_anomalies / len(y_pred) if len(y_pred) else 0.0
    benign_flagged = int(((y_true == 0) & (y_pred == 1)).sum())
    malicious_detected = int(((y_true == 1) & (y_pred == 1)).sum())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": false_positive_rate,
        "roc_auc": roc_auc_score(y_true, anomaly_scores),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_anomalies": predicted_anomalies,
        "predicted_anomaly_rate": predicted_anomaly_rate,
        "benign_test_flagged_as_anomaly": benign_flagged,
        "malicious_test_detected": malicious_detected,
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }


def verify_unique_row_ids(dataframe: pd.DataFrame, source_name: str) -> None:
    if dataframe[ROW_ID_COLUMN].duplicated().any():
        duplicate_count = int(dataframe[ROW_ID_COLUMN].duplicated().sum())
        raise ValueError(f"{source_name} contains {duplicate_count} duplicated {ROW_ID_COLUMN} values.")


def load_modeling_dataframe_with_capture_dates() -> pd.DataFrame:
    modeling_dataframe = pd.read_csv(INPUT_FILE)
    metadata_dataframe = pd.read_csv(METADATA_FILE, usecols=[ROW_ID_COLUMN, CAPTURE_DATE_COLUMN])

    verify_unique_row_ids(modeling_dataframe, str(INPUT_FILE))
    verify_unique_row_ids(metadata_dataframe, str(METADATA_FILE))

    merged_dataframe = modeling_dataframe.merge(
        metadata_dataframe,
        on=ROW_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if len(merged_dataframe) != len(modeling_dataframe):
        raise ValueError("Joining capture_date changed the number of modelling records.")
    if merged_dataframe[CAPTURE_DATE_COLUMN].isna().any():
        missing_count = int(merged_dataframe[CAPTURE_DATE_COLUMN].isna().sum())
        raise ValueError(f"Join left {missing_count} modelling records without capture_date.")

    return merged_dataframe


def print_capture_date_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        dataframe.assign(
            class_name=dataframe[TARGET_COLUMN].map({0: "Benign", 1: "Malicious"}).fillna("Unknown")
        )
        .groupby([CAPTURE_DATE_COLUMN, "class_name"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    for class_name in ["Benign", "Malicious"]:
        if class_name not in distribution.columns:
            distribution[class_name] = 0
    distribution = distribution[["Benign", "Malicious"]]
    distribution["Total"] = distribution["Benign"] + distribution["Malicious"]

    print("\nBenign/Malicious distribution for every capture date:")
    print(distribution.to_string())
    return distribution


def validate_chronological_split(dataframe: pd.DataFrame) -> None:
    available_dates = set(dataframe[CAPTURE_DATE_COLUMN].astype(str).unique())
    missing_train_dates = [date for date in TRAIN_DATES if date not in available_dates]
    missing_test_dates = [date for date in TEST_DATES if date not in available_dates]
    overlap = sorted(set(TRAIN_DATES) & set(TEST_DATES))

    if not TRAIN_DATES or not TEST_DATES:
        raise ValueError("TRAIN_DATES and TEST_DATES must both be non-empty.")
    if missing_train_dates:
        raise ValueError(f"TRAIN_DATES contain dates not present in the dataset: {missing_train_dates}")
    if missing_test_dates:
        raise ValueError(f"TEST_DATES contain dates not present in the dataset: {missing_test_dates}")
    if overlap:
        raise ValueError(f"TRAIN_DATES and TEST_DATES overlap: {overlap}")


def save_metrics(
    total_train_records: int,
    benign_train_records: int,
    malicious_train_excluded: int,
    test_benign_count: int,
    test_malicious_count: int,
    metrics: dict[str, object],
) -> None:
    metrics_rows = [
        {"metric": "total_chronological_train_records", "value": total_train_records},
        {"metric": "benign_train_records_used_for_fit", "value": benign_train_records},
        {"metric": "malicious_train_records_excluded", "value": malicious_train_excluded},
        {"metric": "test_benign_count", "value": test_benign_count},
        {"metric": "test_malicious_count", "value": test_malicious_count},
        {"metric": "test_accuracy", "value": metrics["accuracy"]},
        {"metric": "test_precision", "value": metrics["precision"]},
        {"metric": "test_recall", "value": metrics["recall"]},
        {"metric": "test_f1_score", "value": metrics["f1_score"]},
        {"metric": "test_false_positive_rate", "value": metrics["false_positive_rate"]},
        {"metric": "test_roc_auc", "value": metrics["roc_auc"]},
        {"metric": "true_negatives", "value": metrics["true_negatives"]},
        {"metric": "false_positives", "value": metrics["false_positives"]},
        {"metric": "false_negatives", "value": metrics["false_negatives"]},
        {"metric": "true_positives", "value": metrics["true_positives"]},
        {"metric": "predicted_anomalies", "value": metrics["predicted_anomalies"]},
        {"metric": "predicted_anomaly_rate", "value": metrics["predicted_anomaly_rate"]},
        {"metric": "benign_test_flagged_as_anomaly", "value": metrics["benign_test_flagged_as_anomaly"]},
        {"metric": "malicious_test_detected", "value": metrics["malicious_test_detected"]},
    ]
    pd.DataFrame(metrics_rows).to_csv(METRICS_OUTPUT_FILE, index=False)


def save_predictions(
    row_ids: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    anomaly_scores: pd.Series,
) -> None:
    predictions = pd.DataFrame(
        {
            "flow_row_id": row_ids.to_numpy(),
            "true_label": y_true.to_numpy(),
            "predicted_label": y_pred,
            "anomaly_score": anomaly_scores,
        }
    ).sort_values("flow_row_id")
    predictions.to_csv(PREDICTIONS_OUTPUT_FILE, index=False)


def main() -> None:
    dataframe = load_modeling_dataframe_with_capture_dates()
    numerical_features, categorical_features = load_approved_features()
    final_features = numerical_features + categorical_features

    validate_chronological_split(dataframe)
    print("Isolation Forest chronological binary baseline")
    print(f"Input dataset: {INPUT_FILE}")
    print(f"Metadata dataset: {METADATA_FILE}")
    print("Default Isolation Forest decision rule: sklearn predicts +1=inlier, -1=anomaly.")
    print("Binary conversion used here: +1 -> 0 (Benign), -1 -> 1 (Malicious/anomaly).")
    print_capture_date_distribution(dataframe)
    print(f"\nTraining dates: {TRAIN_DATES}")
    print(f"Testing dates: {TEST_DATES}")

    train_mask = dataframe[CAPTURE_DATE_COLUMN].isin(TRAIN_DATES)
    test_mask = dataframe[CAPTURE_DATE_COLUMN].isin(TEST_DATES)

    chronological_train = dataframe.loc[train_mask].copy()
    chronological_test = dataframe.loc[test_mask].copy()
    benign_train = chronological_train.loc[chronological_train[TARGET_COLUMN] == 0].copy()
    malicious_train_excluded = chronological_train.loc[chronological_train[TARGET_COLUMN] == 1].copy()

    if benign_train.empty:
        raise ValueError("No benign training samples exist for Isolation Forest fitting.")

    y_test = chronological_test[TARGET_COLUMN].astype(int)
    if y_test.nunique() < 2:
        raise ValueError("Chronological test set must contain both Benign and Malicious records.")

    test_benign_count = int((y_test == 0).sum())
    test_malicious_count = int((y_test == 1).sum())
    if test_benign_count == 0 or test_malicious_count == 0:
        raise ValueError("Chronological test set must contain both Benign and Malicious records.")

    print(f"\nTotal chronological training-period records: {len(chronological_train)}")
    print(f"Benign training records used to fit Isolation Forest: {len(benign_train)}")
    print(f"Malicious training records excluded from model fitting: {len(malicious_train_excluded)}")
    print(f"Test Benign count: {test_benign_count}")
    print(f"Test Malicious count: {test_malicious_count}")
    print(f"\nFinal feature list used: {final_features}")

    X_train_benign = benign_train.loc[:, final_features].copy()
    X_test = chronological_test.loc[:, final_features].copy()
    row_ids_test = chronological_test[ROW_ID_COLUMN].copy()

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X_train_benign_processed = preprocessor.fit_transform(X_train_benign)
    X_test_processed = preprocessor.transform(X_test)

    model = IsolationForest(random_state=42)
    model.fit(X_train_benign_processed)

    raw_predictions = model.predict(X_test_processed)
    y_test_pred = pd.Series((raw_predictions == -1).astype(int), index=chronological_test.index)

    # decision_function is higher for normal traffic, so we invert it so that
    # larger anomaly_score means more likely Malicious/anomalous.
    anomaly_scores = pd.Series(-model.decision_function(X_test_processed), index=chronological_test.index)
    metrics = compute_metrics(y_test, y_test_pred, anomaly_scores)

    save_metrics(
        total_train_records=len(chronological_train),
        benign_train_records=len(benign_train),
        malicious_train_excluded=len(malicious_train_excluded),
        test_benign_count=test_benign_count,
        test_malicious_count=test_malicious_count,
        metrics=metrics,
    )
    save_predictions(
        row_ids=row_ids_test,
        y_true=y_test,
        y_pred=y_test_pred,
        anomaly_scores=anomaly_scores,
    )

    print("\nEvaluation metrics:")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1-score: {metrics['f1_score']:.4f}")
    print(f"Test False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(
        f"Predicted anomalies: {metrics['predicted_anomalies']} "
        f"({metrics['predicted_anomaly_rate']:.2%} of test records)"
    )
    print(f"Benign test records incorrectly flagged as anomalies: {metrics['benign_test_flagged_as_anomaly']}")
    print(f"Malicious test records successfully detected: {metrics['malicious_test_detected']}")
    print("\nConfusion matrix [TN FP; FN TP]:")
    print(
        f"{metrics['true_negatives']} {metrics['false_positives']}\n"
        f"{metrics['false_negatives']} {metrics['true_positives']}"
    )
    print("\nClassification report:")
    print(metrics["classification_report"])
    print(f"Saved metrics to: {METRICS_OUTPUT_FILE}")
    print(f"Saved predictions to: {PREDICTIONS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
