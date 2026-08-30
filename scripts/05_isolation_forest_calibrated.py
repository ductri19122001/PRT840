from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
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

EXPERIMENT_NAME = "VM2"
PROTOCOL_NAME = "common_random_holdout_primary"
RANDOM_STATE = 42
BENIGN_FIT_FRACTION = 0.70
BENIGN_CALIBRATION_FRACTION = 0.15
BENIGN_TEST_FRACTION = 0.15
CALIBRATION_QUANTILE = 0.99

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
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "label"
ROW_ID_COLUMN = "flow_row_id"
CAPTURE_DATE_COLUMN = "capture_date"
DETAILED_LABEL_COLUMN = "detailedlabel"

IF_MODEL_PARAMETERS = {
    "n_estimators": 200,
    "contamination": "auto",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

ATTACK_LABELS = {
    "RedLineStealer": {"match": "redlinestealer", "small_sample": False},
    "AgentTesla": {"match": "agenttesla", "small_sample": True},
    "LockBit": {"match": "lockbit", "small_sample": True},
}

# The prior date-based chronological split is retained in repository history as a
# secondary robustness reference. This script intentionally runs only the common
# benign-only random hold-out protocol so VM3 and VM2 remain directly comparable.


def verify_unique_row_ids(dataframe: pd.DataFrame, source_name: str) -> None:
    if dataframe[ROW_ID_COLUMN].duplicated().any():
        duplicate_count = int(dataframe[ROW_ID_COLUMN].duplicated().sum())
        raise ValueError(f"{source_name} contains {duplicate_count} duplicate {ROW_ID_COLUMN} values.")


def load_modeling_dataframe_with_metadata() -> tuple[pd.DataFrame, int]:
    modeling_dataframe = pd.read_csv(INPUT_FILE)
    metadata_columns = [ROW_ID_COLUMN, CAPTURE_DATE_COLUMN]
    metadata_header = pd.read_csv(METADATA_FILE, nrows=0)
    if DETAILED_LABEL_COLUMN in metadata_header.columns:
        metadata_columns.append(DETAILED_LABEL_COLUMN)
    metadata_dataframe = pd.read_csv(METADATA_FILE, usecols=metadata_columns)

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
    if DETAILED_LABEL_COLUMN not in merged_dataframe:
        merged_dataframe[DETAILED_LABEL_COLUMN] = ""

    numeric_labels = pd.to_numeric(merged_dataframe[TARGET_COLUMN], errors="coerce")
    valid_label_mask = numeric_labels.isin([0, 1])
    discarded_records = int((~valid_label_mask).sum())
    merged_dataframe = merged_dataframe.loc[valid_label_mask].copy()
    merged_dataframe[TARGET_COLUMN] = numeric_labels.loc[valid_label_mask].astype(int)
    return merged_dataframe, discarded_records


def validate_feature_contract(dataframe: pd.DataFrame) -> None:
    expected_features = [
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
        "proto",
        "service",
        "conn_state",
        "history",
    ]
    missing_features = [feature for feature in expected_features if feature not in dataframe.columns]
    if missing_features:
        raise ValueError(f"Input data is missing required baseline features: {missing_features}")
    if ALL_FEATURES != expected_features or len(ALL_FEATURES) != 14:
        raise ValueError("The experiment must use the fixed 14-feature baseline.")
    if not np.isclose(BENIGN_FIT_FRACTION + BENIGN_CALIBRATION_FRACTION + BENIGN_TEST_FRACTION, 1.0):
        raise ValueError("The benign split fractions must sum to 1.0.")


def prepare_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    features = dataframe.loc[:, ALL_FEATURES].copy()
    for feature in NUMERICAL_FEATURES:
        features[feature] = pd.to_numeric(features[feature], errors="coerce")
    for feature in CATEGORICAL_FEATURES:
        features[feature] = features[feature].where(features[feature].notna(), "missing").astype(str)
    return features


def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_benign_records(benign_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benign_fit, benign_temporary = train_test_split(
        benign_dataframe,
        test_size=BENIGN_CALIBRATION_FRACTION + BENIGN_TEST_FRACTION,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    benign_calibration, benign_test = train_test_split(
        benign_temporary,
        test_size=BENIGN_TEST_FRACTION / (BENIGN_CALIBRATION_FRACTION + BENIGN_TEST_FRACTION),
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    if min(len(benign_fit), len(benign_calibration), len(benign_test)) == 0:
        raise ValueError("The benign-only split produced an empty partition.")
    return benign_fit.copy(), benign_calibration.copy(), benign_test.copy()


def build_final_test_set(benign_test: pd.DataFrame, malicious_dataframe: pd.DataFrame) -> pd.DataFrame:
    final_test = pd.concat([benign_test, malicious_dataframe], axis=0)
    return final_test.sample(frac=1.0, random_state=RANDOM_STATE).copy()


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, anomaly_scores: pd.Series) -> dict[str, object]:
    if set(y_true.unique()) != {0, 1}:
        raise ValueError("Final test data must contain both benign and malicious records.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    predicted_anomalies = int((y_pred == 1).sum())
    malicious_detection_rate = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": false_positive_rate,
        "roc_auc": roc_auc_score(y_true, anomaly_scores),
        "average_precision": average_precision_score(y_true, anomaly_scores),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "specificity": specificity,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_anomalies": predicted_anomalies,
        "predicted_anomaly_rate": predicted_anomalies / len(y_pred),
        "benign_test_flagged_as_anomaly": int(((y_true == 0) & (y_pred == 1)).sum()),
        "malicious_test_detected": int(((y_true == 1) & (y_pred == 1)).sum()),
        "malicious_detection_rate": malicious_detection_rate,
    }


def percentage_above_threshold(scores: pd.Series, threshold: float) -> float:
    return float((scores > threshold).mean()) if len(scores) else 0.0


def build_attack_detection_rows(test_dataframe: pd.DataFrame, y_pred: pd.Series) -> list[dict[str, object]]:
    malicious_mask = test_dataframe[TARGET_COLUMN].eq(1)
    normalised_labels = (
        test_dataframe[DETAILED_LABEL_COLUMN]
        .fillna("")
        .astype(str)
        .str.replace(r"[^A-Za-z0-9]+", "", regex=True)
        .str.lower()
    )
    rows: list[dict[str, object]] = []
    for attack_name, attack_metadata in ATTACK_LABELS.items():
        attack_mask = malicious_mask & normalised_labels.str.contains(
            attack_metadata["match"],
            case=False,
            regex=False,
            na=False,
        )
        total_records = int(attack_mask.sum())
        detected_records = int(y_pred.loc[attack_mask].sum()) if total_records else 0
        rows.append(
            {
                "attack_name": attack_name,
                "test_attack_records": total_records,
                "detected_records": detected_records,
                "missed_records": total_records - detected_records,
                "detection_rate": detected_records / total_records if total_records else 0.0,
                "small_sample_note": (
                    "Very small-sample descriptive result only."
                    if attack_metadata["small_sample"]
                    else ""
                ),
            }
        )
    return rows


def attack_metric_slug(attack_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", attack_name.lower()).strip("_")


def save_metrics(
    benign_total_records: int,
    malicious_total_records: int,
    benign_fit_records: int,
    benign_calibration_records: int,
    benign_test_records: int,
    malicious_test_records: int,
    calibrated_threshold: float,
    calibration_anomaly_rate: float,
    metrics: dict[str, object],
    attack_rows: list[dict[str, object]],
) -> None:
    metrics_rows = [
        {"metric": "evaluation_protocol", "value": PROTOCOL_NAME},
        {"metric": "random_state", "value": RANDOM_STATE},
        {"metric": "benign_total_records", "value": benign_total_records},
        {"metric": "malicious_total_records", "value": malicious_total_records},
        {"metric": "benign_fit_records_used_for_if_training", "value": benign_fit_records},
        {"metric": "benign_calibration_records", "value": benign_calibration_records},
        {"metric": "benign_test_records", "value": benign_test_records},
        {"metric": "malicious_test_records", "value": malicious_test_records},
        {"metric": "calibrated_threshold", "value": calibrated_threshold},
        {"metric": "calibration_quantile", "value": CALIBRATION_QUANTILE},
        {"metric": "calibration_anomaly_rate", "value": calibration_anomaly_rate},
        {"metric": "test_accuracy", "value": metrics["accuracy"]},
        {"metric": "test_precision", "value": metrics["precision"]},
        {"metric": "test_recall", "value": metrics["recall"]},
        {"metric": "test_f1_score", "value": metrics["f1_score"]},
        {"metric": "test_false_positive_rate", "value": metrics["false_positive_rate"]},
        {"metric": "test_roc_auc", "value": metrics["roc_auc"]},
        {"metric": "test_average_precision", "value": metrics["average_precision"]},
        {"metric": "test_balanced_accuracy", "value": metrics["balanced_accuracy"]},
        {"metric": "test_specificity", "value": metrics["specificity"]},
        {"metric": "true_negatives", "value": metrics["true_negatives"]},
        {"metric": "false_positives", "value": metrics["false_positives"]},
        {"metric": "false_negatives", "value": metrics["false_negatives"]},
        {"metric": "true_positives", "value": metrics["true_positives"]},
        {"metric": "predicted_anomalies", "value": metrics["predicted_anomalies"]},
        {"metric": "predicted_anomaly_rate", "value": metrics["predicted_anomaly_rate"]},
        {"metric": "benign_test_flagged_as_anomaly", "value": metrics["benign_test_flagged_as_anomaly"]},
        {"metric": "malicious_test_detected", "value": metrics["malicious_test_detected"]},
        {"metric": "malicious_detection_rate", "value": metrics["malicious_detection_rate"]},
    ]
    for attack_row in attack_rows:
        attack_slug = attack_metric_slug(str(attack_row["attack_name"]))
        metrics_rows.extend(
            [
                {"metric": f"attack_{attack_slug}_test_records", "value": attack_row["test_attack_records"]},
                {"metric": f"attack_{attack_slug}_detected_records", "value": attack_row["detected_records"]},
                {"metric": f"attack_{attack_slug}_detection_rate", "value": attack_row["detection_rate"]},
                {"metric": f"attack_{attack_slug}_small_sample_note", "value": attack_row["small_sample_note"]},
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
            "evaluation_protocol": PROTOCOL_NAME,
        }
    ).sort_values(ROW_ID_COLUMN)
    predictions.to_csv(PREDICTIONS_OUTPUT_FILE, index=False)


def print_validation_checks(
    benign_fit: pd.DataFrame,
    benign_calibration: pd.DataFrame,
    benign_test: pd.DataFrame,
    malicious_dataframe: pd.DataFrame,
    final_test: pd.DataFrame,
    calibration_scores: pd.Series,
    model: IsolationForest,
) -> None:
    fit_ids = set(benign_fit[ROW_ID_COLUMN])
    calibration_ids = set(benign_calibration[ROW_ID_COLUMN])
    benign_test_ids = set(benign_test[ROW_ID_COLUMN])
    malicious_ids = set(malicious_dataframe[ROW_ID_COLUMN])
    final_malicious_ids = set(final_test.loc[final_test[TARGET_COLUMN].eq(1), ROW_ID_COLUMN])
    partitions_do_not_overlap = not (
        fit_ids & calibration_ids or fit_ids & benign_test_ids or calibration_ids & benign_test_ids
    )
    model_parameters_match = all(model.get_params()[key] == value for key, value in IF_MODEL_PARAMETERS.items())
    checks = [
        ("No malicious record in IF training", benign_fit[TARGET_COLUMN].eq(0).all()),
        ("No malicious record in threshold calibration", benign_calibration[TARGET_COLUMN].eq(0).all()),
        ("Train/calibration/benign-test IDs do not overlap", partitions_do_not_overlap),
        ("All malicious records appear only in final test", malicious_ids == final_malicious_ids),
        ("Preprocessing fitted only on benign training", benign_fit[TARGET_COLUMN].eq(0).all()),
        (
            "Threshold calculated only from benign calibration",
            calibration_scores.index.equals(benign_calibration.index)
            and benign_calibration[TARGET_COLUMN].eq(0).all(),
        ),
        ("random_state is 42 for all random operations", RANDOM_STATE == 42),
        ("Exactly the same 14 baseline features are used", len(ALL_FEATURES) == 14),
        ("Isolation Forest parameters match the shared contract", model_parameters_match),
        ("Threshold quantile is 0.99", CALIBRATION_QUANTILE == 0.99),
    ]
    print()
    print("Protocol validation checks:")
    for check_name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'} - {check_name}")
    if not all(passed for _, passed in checks):
        raise RuntimeError("One or more common protocol validation checks failed.")


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
    dataframe, discarded_records = load_modeling_dataframe_with_metadata()
    validate_feature_contract(dataframe)

    benign_dataframe = dataframe.loc[dataframe[TARGET_COLUMN].eq(0)].copy()
    malicious_dataframe = dataframe.loc[dataframe[TARGET_COLUMN].eq(1)].copy()
    if benign_dataframe.empty or malicious_dataframe.empty:
        raise ValueError("The common protocol requires at least one benign and one malicious record.")

    benign_fit, benign_calibration, benign_test = split_benign_records(benign_dataframe)
    final_test = build_final_test_set(benign_test, malicious_dataframe)

    print(f"{EXPERIMENT_NAME} Isolation Forest calibrated baseline")
    print(f"Evaluation protocol: {PROTOCOL_NAME}")
    print(f"Input dataset: {INPUT_FILE}")
    print("Anomaly score definition: anomaly_score = -model.decision_function(X)")
    print("Prediction rule: anomaly_score > calibrated_threshold")
    print(f"Fixed 14-feature baseline: {ALL_FEATURES}")
    print(f"Unknown or invalid labels excluded: {discarded_records}")
    print(
        "Benign split counts: "
        f"fit={len(benign_fit)}, calibration={len(benign_calibration)}, test={len(benign_test)}"
    )
    print(f"All malicious records reserved for final test: {len(malicious_dataframe)}")

    X_fit = prepare_feature_frame(benign_fit)
    X_calibration = prepare_feature_frame(benign_calibration)
    X_test = prepare_feature_frame(final_test)
    y_test = final_test[TARGET_COLUMN].astype(int)

    preprocessor = build_preprocessor()
    X_fit_processed = preprocessor.fit_transform(X_fit)
    X_calibration_processed = preprocessor.transform(X_calibration)
    X_test_processed = preprocessor.transform(X_test)

    model = IsolationForest(**IF_MODEL_PARAMETERS)
    model.fit(X_fit_processed)

    calibration_scores = pd.Series(
        -model.decision_function(X_calibration_processed),
        index=benign_calibration.index,
    )
    calibrated_threshold = float(np.quantile(calibration_scores, CALIBRATION_QUANTILE))
    calibration_anomaly_rate = percentage_above_threshold(calibration_scores, calibrated_threshold)

    test_anomaly_scores = pd.Series(
        -model.decision_function(X_test_processed),
        index=final_test.index,
    )
    y_test_pred = pd.Series(
        (test_anomaly_scores > calibrated_threshold).astype(int),
        index=final_test.index,
    )

    metrics = compute_metrics(y_test, y_test_pred, test_anomaly_scores)
    attack_rows = build_attack_detection_rows(final_test, y_test_pred)
    print_validation_checks(
        benign_fit,
        benign_calibration,
        benign_test,
        malicious_dataframe,
        final_test,
        calibration_scores,
        model,
    )

    save_metrics(
        benign_total_records=len(benign_dataframe),
        malicious_total_records=len(malicious_dataframe),
        benign_fit_records=len(benign_fit),
        benign_calibration_records=len(benign_calibration),
        benign_test_records=len(benign_test),
        malicious_test_records=len(malicious_dataframe),
        calibrated_threshold=calibrated_threshold,
        calibration_anomaly_rate=calibration_anomaly_rate,
        metrics=metrics,
        attack_rows=attack_rows,
    )
    save_predictions(
        test_dataframe=final_test,
        y_true=y_test,
        y_pred=y_test_pred,
        anomaly_scores=test_anomaly_scores,
        calibrated_threshold=calibrated_threshold,
    )

    benign_test_scores = test_anomaly_scores.loc[y_test.eq(0)]
    malicious_test_scores = test_anomaly_scores.loc[y_test.eq(1)]
    print()
    print(f"Calibrated threshold (benign calibration q={CALIBRATION_QUANTILE:.2f}): {calibrated_threshold:.6f}")
    print(f"Calibration anomaly rate: {calibration_anomaly_rate:.2%}")
    print()
    print("Primary comparable metrics:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Malicious recall/detection rate: {metrics['malicious_detection_rate']:.4f}")
    print(f"False positive rate: {metrics['false_positive_rate']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(
        "Interpret raw accuracy and precision with class prevalence in mind; "
        "do not use them alone to compare VM performance."
    )
    print()
    print("Confusion matrix [TN FP; FN TP]:")
    print(f"{metrics['true_negatives']} {metrics['false_positives']}")
    print(f"{metrics['false_negatives']} {metrics['true_positives']}")
    print()
    print("Attack-specific descriptive results (detailedlabel is reporting-only):")
    print(pd.DataFrame(attack_rows).to_string(index=False))
    print()
    print("Classification report:")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    if plt is not None:
        try:
            plot_score_distribution(benign_test_scores, malicious_test_scores, calibrated_threshold)
            print("Interactive anomaly-score distribution displayed.")
        except Exception as exc:
            print(f"Interactive visualisation skipped: {exc}")

    print(f"Saved metrics to: {METRICS_OUTPUT_FILE}")
    print(f"Saved predictions to: {PREDICTIONS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
