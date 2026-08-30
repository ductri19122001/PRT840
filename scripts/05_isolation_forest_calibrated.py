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
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


INPUT_FILE = BASE_DIR / "outputs" / "clean_network_flows.csv"
METADATA_FILE = BASE_DIR / "outputs" / "cti_mapping_metadata.csv"
METRICS_OUTPUT_FILE = BASE_DIR / "outputs" / "isolation_forest_calibrated_metrics.csv"
PREDICTIONS_OUTPUT_FILE = BASE_DIR / "outputs" / "isolation_forest_calibrated_predictions.csv"
BASELINE_METRICS_FILE = BASE_DIR / "outputs" / "isolation_forest_chronological_metrics.csv"

CALIBRATION_QUANTILE = 0.99
BENIGN_TEST_SIZE = 0.20
NUMERICAL_FEATURES = [
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
CATEGORICAL_FEATURES = ["proto", "service", "conn_state", "history"]
TARGET_COLUMN = "label"
ROW_ID_COLUMN = "flow_row_id"
CAPTURE_DATE_COLUMN = "capture_date"
DETAILED_LABEL_COLUMN = "detailedlabel"
TRAIN_DATES = ["2023-02-20", "2023-02-21"]
TEST_DATES = ["2023-02-22", "2023-02-23", "2023-02-24", "2023-02-25", "2023-02-26"]
ATTACK_LABELS = {
    "RedLineStealer": {"match": "redline", "small_sample": False},
    "AgentTesla": {"match": "agenttesla", "small_sample": True},
    "LockBit": {"match": "lockbit", "small_sample": True},
}


def verify_unique_row_ids(dataframe: pd.DataFrame, source_name: str) -> None:
    if dataframe[ROW_ID_COLUMN].duplicated().any():
        duplicate_count = int(dataframe[ROW_ID_COLUMN].duplicated().sum())
        raise ValueError(f"{source_name} contains {duplicate_count} duplicated {ROW_ID_COLUMN} values.")


def load_modeling_dataframe_with_metadata() -> pd.DataFrame:
    modeling_dataframe = pd.read_csv(INPUT_FILE)
    metadata_dataframe = pd.read_csv(
        METADATA_FILE,
        usecols=[ROW_ID_COLUMN, CAPTURE_DATE_COLUMN, DETAILED_LABEL_COLUMN],
    )
    verify_unique_row_ids(modeling_dataframe, str(INPUT_FILE))
    verify_unique_row_ids(metadata_dataframe, str(METADATA_FILE))
    merged_dataframe = modeling_dataframe.merge(
        metadata_dataframe,
        on=ROW_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if len(merged_dataframe) != len(modeling_dataframe):
        raise ValueError("Joining metadata changed the number of modelling records.")
    if merged_dataframe[CAPTURE_DATE_COLUMN].isna().any():
        raise ValueError("Some modelling records are missing capture_date after the metadata join.")
    return merged_dataframe


def validate_chronological_split(dataframe: pd.DataFrame) -> None:
    available_dates = set(dataframe[CAPTURE_DATE_COLUMN].astype(str).unique())
    missing_train_dates = [date for date in TRAIN_DATES if date not in available_dates]
    missing_test_dates = [date for date in TEST_DATES if date not in available_dates]
    overlap = sorted(set(TRAIN_DATES) & set(TEST_DATES))

    if missing_train_dates:
        raise ValueError(f"TRAIN_DATES contain dates not present in the dataset: {missing_train_dates}")
    if missing_test_dates:
        raise ValueError(f"TEST_DATES contain dates not present in the dataset: {missing_test_dates}")
    if overlap:
        raise ValueError(f"TRAIN_DATES and TEST_DATES overlap: {overlap}")


def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, anomaly_scores: pd.Series) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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


def percentage_at_or_above_threshold(scores: pd.Series, threshold: float) -> float:
    if len(scores) == 0:
        return 0.0
    return float((scores >= threshold).mean())


def build_attack_detection_rows(test_dataframe: pd.DataFrame, y_pred: pd.Series) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    detailed_labels = (
        test_dataframe[DETAILED_LABEL_COLUMN]
        .fillna("")
        .astype(str)
        .str.replace(r"[^A-Za-z0-9]+", "", regex=True)
        .str.lower()
    )
    for attack_name, metadata in ATTACK_LABELS.items():
        mask = detailed_labels.str.contains(metadata["match"], na=False)
        attack_total = int(mask.sum())
        detected = int(y_pred.loc[mask].sum()) if attack_total else 0
        detection_rate = (detected / attack_total) if attack_total else 0.0
        rows.append(
            {
                "attack_name": attack_name,
                "match_rule": metadata["match"],
                "test_attack_rows": attack_total,
                "detected_rows": detected,
                "missed_rows": attack_total - detected,
                "detection_rate": detection_rate,
                "small_sample_note": (
                    "Very small-sample descriptive result only."
                    if metadata["small_sample"]
                    else ""
                ),
            }
        )
    return rows


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
    attack_rows: list[dict[str, object]],
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
    for row in attack_rows:
        attack_slug = row["attack_name"].lower()
        metrics_rows.extend(
            [
                {"metric": f"{attack_slug}_test_attack_rows", "value": row["test_attack_rows"]},
                {"metric": f"{attack_slug}_detected_rows", "value": row["detected_rows"]},
                {"metric": f"{attack_slug}_detection_rate", "value": row["detection_rate"]},
            ]
        )
    pd.DataFrame(metrics_rows).to_csv(METRICS_OUTPUT_FILE, index=False)


def save_predictions(
    test_dataframe: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    anomaly_scores: pd.Series,
    calibrated_threshold: float,
) -> None:
    predictions = pd.DataFrame(
        {
            ROW_ID_COLUMN: test_dataframe[ROW_ID_COLUMN].to_numpy(),
            CAPTURE_DATE_COLUMN: test_dataframe[CAPTURE_DATE_COLUMN].astype(str).to_numpy(),
            DETAILED_LABEL_COLUMN: test_dataframe[DETAILED_LABEL_COLUMN].fillna("").to_numpy(),
            "true_label": y_true.to_numpy(),
            "predicted_label": y_pred.to_numpy(),
            "anomaly_score": anomaly_scores.to_numpy(),
            "calibrated_threshold": calibrated_threshold,
        }
    ).sort_values(ROW_ID_COLUMN)
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


def plot_score_distribution(benign_scores: pd.Series, malicious_scores: pd.Series, threshold: float) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(benign_scores, bins=50, density=True, alpha=0.6, color="#4f81bd", label="Benign test anomaly scores")
    ax.hist(malicious_scores, bins=50, density=True, alpha=0.6, color="#c0504d", label="Malicious test anomaly scores")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label="Calibrated threshold")
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
    dataframe = load_modeling_dataframe_with_metadata()
    validate_chronological_split(dataframe)

    print("Isolation Forest benign-only calibrated-threshold experiment")
    print(f"Input dataset: {INPUT_FILE}")
    print(f"Metadata dataset: {METADATA_FILE}")
    print("Feature set adaptation note: use the full 14-feature VM3 baseline unchanged for VM2 comparability.")
    print("Anomaly score definition: anomaly_score = -model.decision_function(X)")
    print("Interpretation: higher anomaly_score = more anomalous.")
    print(f"Training dates: {TRAIN_DATES}")
    print(f"Testing dates: {TEST_DATES}")
    print(f"Final feature list used: {NUMERICAL_FEATURES + CATEGORICAL_FEATURES}")

    chronological_train = dataframe.loc[dataframe[CAPTURE_DATE_COLUMN].isin(TRAIN_DATES)].copy()
    chronological_test = dataframe.loc[dataframe[CAPTURE_DATE_COLUMN].isin(TEST_DATES)].copy()
    benign_train = chronological_train.loc[chronological_train[TARGET_COLUMN] == 0].copy()
    malicious_train_excluded = chronological_train.loc[chronological_train[TARGET_COLUMN] == 1].copy()

    benign_fit, benign_calibration = train_test_split(
        benign_train,
        test_size=BENIGN_TEST_SIZE,
        random_state=42,
    )

    print(f"\nTraining benign count: {len(benign_train)}")
    print(f"Calibration benign count: {len(benign_calibration)}")
    print(f"Excluded malicious training-period records: {len(malicious_train_excluded)}")

    X_fit = benign_fit.loc[:, NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    X_calibration = benign_calibration.loc[:, NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    X_test = chronological_test.loc[:, NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y_test = chronological_test[TARGET_COLUMN].astype(int)

    test_benign_count = int((y_test == 0).sum())
    test_malicious_count = int((y_test == 1).sum())

    preprocessor = build_preprocessor()
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

    calibration_anomaly_scores = pd.Series(-model.decision_function(X_calibration_processed), index=benign_calibration.index)
    calibrated_threshold = float(np.quantile(calibration_anomaly_scores, CALIBRATION_QUANTILE))
    calibration_predicted_anomaly_rate = percentage_at_or_above_threshold(calibration_anomaly_scores, calibrated_threshold)

    test_anomaly_scores = pd.Series(-model.decision_function(X_test_processed), index=chronological_test.index)
    y_test_pred = pd.Series((test_anomaly_scores >= calibrated_threshold).astype(int), index=chronological_test.index)

    metrics = compute_metrics(y_test, y_test_pred, test_anomaly_scores)
    attack_rows = build_attack_detection_rows(chronological_test, y_test_pred)

    save_metrics(
        total_train_records=len(chronological_train),
        benign_train_records_total=len(benign_train),
        benign_fit_records=len(benign_fit),
        benign_calibration_records=len(benign_calibration),
        malicious_train_excluded=len(malicious_train_excluded),
        calibrated_threshold=calibrated_threshold,
        calibration_quantile=CALIBRATION_QUANTILE,
        calibration_anomaly_rate=calibration_predicted_anomaly_rate,
        test_benign_count=test_benign_count,
        test_malicious_count=test_malicious_count,
        metrics=metrics,
        attack_rows=attack_rows,
    )
    save_predictions(
        test_dataframe=chronological_test,
        y_true=y_test,
        y_pred=y_test_pred,
        anomaly_scores=test_anomaly_scores,
        calibrated_threshold=calibrated_threshold,
    )

    benign_test_scores = test_anomaly_scores.loc[y_test == 0]
    malicious_test_scores = test_anomaly_scores.loc[y_test == 1]

    print(f"\nCalibrated threshold (99th percentile of benign calibration anomaly scores): {calibrated_threshold:.6f}")
    print(
        "Benign calibration records classified as anomalous at this threshold: "
        f"{calibration_predicted_anomaly_rate:.2%}"
    )
    print("\nEvaluation metrics on untouched chronological test set:")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1-score: {metrics['f1_score']:.4f}")
    print(f"Test False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Predicted anomaly count: {metrics['predicted_anomalies']}")
    print(f"Benign flagged as anomaly: {metrics['benign_test_flagged_as_anomaly']}")
    print(f"Malicious detected: {metrics['malicious_test_detected']}")
    print(f"Test benign count: {test_benign_count}")
    print(f"Test malicious count: {test_malicious_count}")
    print("\nConfusion matrix [TN FP; FN TP]:")
    print(f"{metrics['true_negatives']} {metrics['false_positives']}\n{metrics['false_negatives']} {metrics['true_positives']}")
    print("\nAttack-specific descriptive detection counts:")
    print(pd.DataFrame(attack_rows).to_string(index=False))

    print("\nDiagnostic anomaly-score summary:")
    print(f"Median anomaly score - benign calibration: {calibration_anomaly_scores.median():.6f}")
    print(f"Median anomaly score - benign test: {benign_test_scores.median():.6f}")
    print(f"Median anomaly score - malicious test: {malicious_test_scores.median():.6f}")
    print(f"Mean anomaly score - benign test: {benign_test_scores.mean():.6f}")
    print(f"Mean anomaly score - malicious test: {malicious_test_scores.mean():.6f}")
    print(f"Percentage beyond calibrated threshold - benign test: {percentage_at_or_above_threshold(benign_test_scores, calibrated_threshold):.2%}")
    print(f"Percentage beyond calibrated threshold - malicious test: {percentage_at_or_above_threshold(malicious_test_scores, calibrated_threshold):.2%}")

    calibrated_metrics = load_metrics_lookup(METRICS_OUTPUT_FILE)
    if BASELINE_METRICS_FILE.exists():
        default_metrics = load_metrics_lookup(BASELINE_METRICS_FILE)
        print_comparison_table(default_metrics, calibrated_metrics)
    else:
        print(f"\nBaseline comparison skipped because the default Isolation Forest metrics file was not found: {BASELINE_METRICS_FILE}")

    print("\nClassification report:")
    print(metrics["classification_report"])
    visualisations_displayed = False
    try:
        plot_score_distribution(benign_test_scores, malicious_test_scores, calibrated_threshold)
        visualisations_displayed = True
    except Exception as exc:
        # VM2 automation may run in a non-GUI environment where Tk/Tcl is unavailable.
        print(f"Visualisations skipped because the environment could not open matplotlib windows: {exc}")
    print(f"Saved metrics to: {METRICS_OUTPUT_FILE}")
    print(f"Saved predictions to: {PREDICTIONS_OUTPUT_FILE}")
    if visualisations_displayed:
        print("Visualisations displayed interactively.")


if __name__ == "__main__":
    main()
