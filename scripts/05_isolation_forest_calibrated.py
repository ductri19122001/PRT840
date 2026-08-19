from __future__ import annotations

import os
from pathlib import Path

import numpy as np
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = BASE_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

try:
    import matplotlib

    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None


INPUT_FILE = BASE_DIR / "outputs" / "clean_network_flows.csv"
METADATA_FILE = BASE_DIR / "outputs" / "cti_mapping_metadata.csv"
FEATURE_RECOMMENDATIONS_FILE = BASE_DIR / "outputs" / "feature_analysis" / "feature_relevance_recommendations.csv"
METRICS_OUTPUT_FILE = BASE_DIR / "outputs" / "isolation_forest_calibrated_metrics.csv"
PREDICTIONS_OUTPUT_FILE = BASE_DIR / "outputs" / "isolation_forest_calibrated_predictions.csv"
BASELINE_METRICS_FILE = BASE_DIR / "outputs" / "isolation_forest_chronological_metrics.csv"

CALIBRATION_QUANTILE = 0.99
BENIGN_TEST_SIZE = 0.20
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
]
TEST_DATES = [
    "2023-02-22",
    "2023-02-23",
    "2023-02-24",
    "2023-02-25",
    "2023-02-26",
]


def load_approved_features() -> tuple[list[str], list[str]]:
    if not FEATURE_RECOMMENDATIONS_FILE.exists():
        return DEFAULT_NUMERICAL_FEATURES, DEFAULT_CATEGORICAL_FEATURES

    recommendations = pd.read_csv(FEATURE_RECOMMENDATIONS_FILE)
    approved = recommendations.loc[recommendations["recommendation"] != "DROP", "feature"].tolist()
    numerical = [feature for feature in DEFAULT_NUMERICAL_FEATURES if feature in approved]
    categorical = [feature for feature in DEFAULT_CATEGORICAL_FEATURES if feature in approved]
    return numerical, categorical


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


def build_preprocessor(numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
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


def save_metrics(
    total_train_records: int,
    benign_train_records_total: int,
    benign_fit_records: int,
    benign_calibration_records: int,
    malicious_train_excluded: int,
    calibrated_threshold: float,
    calibration_quantile: float,
    calibration_anomaly_rate: float,
    test_benign_count: int,
    test_malicious_count: int,
    metrics: dict[str, object],
) -> None:
    metrics_rows = [
        {"metric": "total_chronological_train_records", "value": total_train_records},
        {"metric": "benign_train_records_total", "value": benign_train_records_total},
        {"metric": "benign_fit_records_used_for_if_training", "value": benign_fit_records},
        {"metric": "benign_calibration_records", "value": benign_calibration_records},
        {"metric": "malicious_train_records_excluded", "value": malicious_train_excluded},
        {"metric": "calibrated_threshold", "value": calibrated_threshold},
        {"metric": "calibration_quantile", "value": calibration_quantile},
        {"metric": "calibration_anomaly_rate", "value": calibration_anomaly_rate},
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
    capture_dates: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    anomaly_scores: pd.Series,
    calibrated_threshold: float,
) -> None:
    predictions = pd.DataFrame(
        {
            "flow_row_id": row_ids.to_numpy(),
            "capture_date": capture_dates.astype(str).to_numpy(),
            "true_label": y_true.to_numpy(),
            "predicted_label": y_pred.to_numpy(),
            "anomaly_score": anomaly_scores.to_numpy(),
            "calibrated_threshold": calibrated_threshold,
        }
    ).sort_values("flow_row_id")
    predictions.to_csv(PREDICTIONS_OUTPUT_FILE, index=False)


def load_metrics_lookup(metrics_file: Path) -> dict[str, float]:
    metrics_dataframe = pd.read_csv(metrics_file)
    return dict(zip(metrics_dataframe["metric"], metrics_dataframe["value"]))


def print_comparison_table(default_metrics: dict[str, float], calibrated_metrics: dict[str, float]) -> None:
    rows = [
        ("Accuracy", "test_accuracy"),
        ("Precision", "test_precision"),
        ("Recall", "test_recall"),
        ("F1", "test_f1_score"),
        ("FPR", "test_false_positive_rate"),
        ("ROC-AUC", "test_roc_auc"),
        ("TN", "true_negatives"),
        ("FP", "false_positives"),
        ("FN", "false_negatives"),
        ("TP", "true_positives"),
        ("Predicted anomalies", "predicted_anomalies"),
    ]

    print("\nMetric                 Default IF       Calibrated IF")
    for label, key in rows:
        print(f"{label:<22}{default_metrics[key]:>12.6f}{calibrated_metrics[key]:>20.6f}")


def percentage_at_or_above_threshold(scores: pd.Series, threshold: float) -> float:
    if len(scores) == 0:
        return 0.0
    return float((scores >= threshold).mean())


def plot_score_distribution(
    benign_scores: pd.Series,
    malicious_scores: pd.Series,
    threshold: float,
) -> None:
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        benign_scores,
        bins=50,
        density=True,
        alpha=0.6,
        color="#4f81bd",
        label="Benign test anomaly scores",
    )
    ax.hist(
        malicious_scores,
        bins=50,
        density=True,
        alpha=0.6,
        color="#c0504d",
        label="Malicious test anomaly scores",
    )
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Calibrated threshold",
    )
    ax.set_title("Calibrated Isolation Forest Anomaly Score Distribution")
    ax.set_xlabel("Anomaly score (higher score = more anomalous)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main() -> None:
    dataframe = load_modeling_dataframe_with_capture_dates()
    numerical_features, categorical_features = load_approved_features()
    final_features = numerical_features + categorical_features

    validate_chronological_split(dataframe)
    print("Isolation Forest benign-only calibrated-threshold experiment")
    print(f"Input dataset: {INPUT_FILE}")
    print(f"Metadata dataset: {METADATA_FILE}")
    print("Anomaly score definition: anomaly_score = -model.decision_function(X)")
    print("Interpretation: higher anomaly_score = more anomalous.")
    print(f"Training dates: {TRAIN_DATES}")
    print(f"Testing dates: {TEST_DATES}")
    print(f"Final feature list used: {final_features}")

    train_mask = dataframe[CAPTURE_DATE_COLUMN].isin(TRAIN_DATES)
    test_mask = dataframe[CAPTURE_DATE_COLUMN].isin(TEST_DATES)

    chronological_train = dataframe.loc[train_mask].copy()
    chronological_test = dataframe.loc[test_mask].copy()
    benign_train = chronological_train.loc[chronological_train[TARGET_COLUMN] == 0].copy()
    malicious_train_excluded = chronological_train.loc[chronological_train[TARGET_COLUMN] == 1].copy()

    benign_fit, benign_calibration = train_test_split(
        benign_train,
        test_size=BENIGN_TEST_SIZE,
        random_state=42,
    )

    print(f"\nBenign fit records: {len(benign_fit)}")
    print(f"Benign calibration records: {len(benign_calibration)}")
    print(f"Excluded malicious training-period records: {len(malicious_train_excluded)}")

    X_fit = benign_fit.loc[:, final_features].copy()
    X_calibration = benign_calibration.loc[:, final_features].copy()
    X_test = chronological_test.loc[:, final_features].copy()
    y_test = chronological_test[TARGET_COLUMN].astype(int)

    test_benign_scores_count = int((y_test == 0).sum())
    test_malicious_scores_count = int((y_test == 1).sum())

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X_fit_processed = preprocessor.fit_transform(X_fit)
    X_calibration_processed = preprocessor.transform(X_calibration)
    X_test_processed = preprocessor.transform(X_test)

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_fit_processed)

    calibration_anomaly_scores = pd.Series(
        -model.decision_function(X_calibration_processed),
        index=benign_calibration.index,
    )
    calibrated_threshold = float(np.quantile(calibration_anomaly_scores, CALIBRATION_QUANTILE))
    calibration_predicted_anomaly_rate = percentage_at_or_above_threshold(
        calibration_anomaly_scores,
        calibrated_threshold,
    )

    print(f"\nCalibrated threshold (99th percentile of benign calibration anomaly scores): {calibrated_threshold:.6f}")
    print(
        "Benign calibration records classified as anomalous at this threshold: "
        f"{calibration_predicted_anomaly_rate:.2%}"
    )

    test_anomaly_scores = pd.Series(
        -model.decision_function(X_test_processed),
        index=chronological_test.index,
    )
    y_test_pred = pd.Series(
        (test_anomaly_scores >= calibrated_threshold).astype(int),
        index=chronological_test.index,
    )

    metrics = compute_metrics(y_test, y_test_pred, test_anomaly_scores)

    save_metrics(
        total_train_records=len(chronological_train),
        benign_train_records_total=len(benign_train),
        benign_fit_records=len(benign_fit),
        benign_calibration_records=len(benign_calibration),
        malicious_train_excluded=len(malicious_train_excluded),
        calibrated_threshold=calibrated_threshold,
        calibration_quantile=CALIBRATION_QUANTILE,
        calibration_anomaly_rate=calibration_predicted_anomaly_rate,
        test_benign_count=test_benign_scores_count,
        test_malicious_count=test_malicious_scores_count,
        metrics=metrics,
    )
    save_predictions(
        row_ids=chronological_test[ROW_ID_COLUMN],
        capture_dates=chronological_test[CAPTURE_DATE_COLUMN],
        y_true=y_test,
        y_pred=y_test_pred,
        anomaly_scores=test_anomaly_scores,
        calibrated_threshold=calibrated_threshold,
    )

    benign_test_scores = test_anomaly_scores.loc[y_test == 0]
    malicious_test_scores = test_anomaly_scores.loc[y_test == 1]

    print("\nEvaluation metrics on untouched chronological test set:")
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
    print(f"Benign test records flagged as anomaly: {metrics['benign_test_flagged_as_anomaly']}")
    print(f"Malicious test records detected: {metrics['malicious_test_detected']}")
    print("\nConfusion matrix [TN FP; FN TP]:")
    print(
        f"{metrics['true_negatives']} {metrics['false_positives']}\n"
        f"{metrics['false_negatives']} {metrics['true_positives']}"
    )

    print("\nDiagnostic anomaly-score summary:")
    print(f"Median anomaly score - benign calibration: {calibration_anomaly_scores.median():.6f}")
    print(f"Median anomaly score - benign test: {benign_test_scores.median():.6f}")
    print(f"Median anomaly score - malicious test: {malicious_test_scores.median():.6f}")
    print(f"Mean anomaly score - benign test: {benign_test_scores.mean():.6f}")
    print(f"Mean anomaly score - malicious test: {malicious_test_scores.mean():.6f}")
    print(
        "Percentage beyond calibrated threshold - benign test: "
        f"{percentage_at_or_above_threshold(benign_test_scores, calibrated_threshold):.2%}"
    )
    print(
        "Percentage beyond calibrated threshold - malicious test: "
        f"{percentage_at_or_above_threshold(malicious_test_scores, calibrated_threshold):.2%}"
    )

    calibrated_metrics = load_metrics_lookup(METRICS_OUTPUT_FILE)
    if BASELINE_METRICS_FILE.exists():
        default_metrics = load_metrics_lookup(BASELINE_METRICS_FILE)
        print_comparison_table(default_metrics, calibrated_metrics)
    else:
        print(
            "\nBaseline comparison skipped because the default Isolation Forest metrics file was not found: "
            f"{BASELINE_METRICS_FILE}"
        )

    print("\nClassification report:")
    print(metrics["classification_report"])

    plot_score_distribution(benign_test_scores, malicious_test_scores, calibrated_threshold)

    print(f"Saved metrics to: {METRICS_OUTPUT_FILE}")
    print(f"Saved predictions to: {PREDICTIONS_OUTPUT_FILE}")
    if plt is not None:
        print("Visualisations displayed interactively.")


if __name__ == "__main__":
    main()
