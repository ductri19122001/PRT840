from __future__ import annotations

import os
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
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = BASE_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


INPUT_FILE = BASE_DIR / "outputs" / "clean_network_flows.csv"
METADATA_FILE = BASE_DIR / "outputs" / "cti_mapping_metadata.csv"
METRICS_OUTPUT_FILE = BASE_DIR / "outputs" / "decision_tree_chronological_metrics.csv"
PREDICTIONS_OUTPUT_FILE = BASE_DIR / "outputs" / "decision_tree_chronological_predictions.csv"
FEATURE_IMPORTANCE_OUTPUT_FILE = BASE_DIR / "outputs" / "decision_tree_feature_importance.csv"

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


def build_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )


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


def print_capture_date_distribution(dataframe: pd.DataFrame) -> None:
    distribution = (
        dataframe.assign(class_name=dataframe[TARGET_COLUMN].map({0: "Benign", 1: "Malicious"}))
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
    print("\nBenign/Malicious distribution for every capture_date:")
    print(distribution.to_string())


def print_label_distribution(split_name: str, labels: pd.Series) -> None:
    distribution = labels.map({0: "Benign", 1: "Malicious"}).value_counts().reindex(["Benign", "Malicious"], fill_value=0)
    print(f"\n{split_name} class distribution:")
    print(distribution.to_string())


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {
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
    train_accuracy: float,
    train_f1: float,
    test_metrics: dict[str, object],
    train_samples: int,
    test_samples: int,
    attack_rows: list[dict[str, object]],
) -> None:
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
        {"metric": "train_samples", "value": train_samples},
        {"metric": "test_samples", "value": test_samples},
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


def save_predictions(test_dataframe: pd.DataFrame, y_pred: pd.Series, y_prob: pd.Series) -> None:
    predictions = pd.DataFrame(
        {
            ROW_ID_COLUMN: test_dataframe[ROW_ID_COLUMN].to_numpy(),
            CAPTURE_DATE_COLUMN: test_dataframe[CAPTURE_DATE_COLUMN].astype(str).to_numpy(),
            DETAILED_LABEL_COLUMN: test_dataframe[DETAILED_LABEL_COLUMN].fillna("").to_numpy(),
            "true_label": test_dataframe[TARGET_COLUMN].to_numpy(),
            "predicted_label": y_pred.to_numpy(),
            "predicted_probability": y_prob,
        }
    ).sort_values(ROW_ID_COLUMN)
    predictions.to_csv(PREDICTIONS_OUTPUT_FILE, index=False)


def map_transformed_feature_to_original(transformed_name: str) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "", 1)
    if transformed_name.startswith("cat__"):
        cleaned_name = transformed_name.replace("cat__", "", 1)
        for categorical_feature in CATEGORICAL_FEATURES:
            if cleaned_name == categorical_feature or cleaned_name.startswith(f"{categorical_feature}_"):
                return categorical_feature
        return cleaned_name
    return transformed_name


def create_feature_importance_outputs(pipeline: Pipeline) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_feature_names = list(preprocessor.get_feature_names_out())
    raw_importances = classifier.feature_importances_

    aggregated_importances: dict[str, float] = {
        feature: 0.0 for feature in NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    }
    for transformed_name, importance in zip(transformed_feature_names, raw_importances):
        original_feature = map_transformed_feature_to_original(transformed_name)
        aggregated_importances[original_feature] = aggregated_importances.get(original_feature, 0.0) + float(importance)

    feature_importance = (
        pd.DataFrame({"feature": list(aggregated_importances.keys()), "importance": list(aggregated_importances.values())})
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_OUTPUT_FILE, index=False)

    if plt is None:
        return

    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        plot_dataframe = feature_importance.sort_values("importance", ascending=True, kind="stable")
        ax.barh(plot_dataframe["feature"], plot_dataframe["importance"], color="#1f4e79")
        ax.set_title("Decision Tree - Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"Feature-importance visualisation skipped because the environment could not open matplotlib windows: {exc}")


def create_confusion_matrix_figure(y_true: pd.Series, y_pred: pd.Series) -> None:
    if plt is None:
        return
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("Decision Tree - Confusion Matrix\nChronological Test (2023-02-22 to 2023-02-26)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Benign", "Malicious"])
    ax.set_yticks([0, 1], labels=["Benign", "Malicious"])

    labels = [["TN", "FP"], ["FN", "TP"]]
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            text_color = "white" if value > threshold else "black"
            ax.text(column_index, row_index, f"{labels[row_index][column_index]}\n{value}", ha="center", va="center", color=text_color)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def create_roc_curve_figure(y_true: pd.Series, y_prob: pd.Series) -> None:
    if plt is None:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f4e79", linewidth=2, label=f"Decision Tree (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Random classifier")
    ax.set_title("Decision Tree - ROC Curve\nChronological Test (2023-02-22 to 2023-02-26)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def create_daily_performance_figure(test_dataframe: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series) -> None:
    if plt is None:
        return
    daily_dataframe = pd.DataFrame(
        {
            CAPTURE_DATE_COLUMN: test_dataframe[CAPTURE_DATE_COLUMN].astype(str).to_numpy(),
            "true_label": y_true.to_numpy(),
            "predicted_label": y_pred.to_numpy(),
        }
    )

    daily_rows: list[dict[str, float | str]] = []
    for capture_date in TEST_DATES:
        daily_subset = daily_dataframe.loc[daily_dataframe[CAPTURE_DATE_COLUMN] == capture_date].copy()
        if daily_subset.empty:
            daily_rows.append({"capture_date": capture_date, "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1_score": float("nan")})
            continue

        y_true_day = daily_subset["true_label"].astype(int)
        y_pred_day = daily_subset["predicted_label"].astype(int)
        has_positive_class = bool((y_true_day == 1).any())
        daily_rows.append(
            {
                "capture_date": capture_date,
                "accuracy": accuracy_score(y_true_day, y_pred_day),
                "precision": precision_score(y_true_day, y_pred_day, zero_division=0) if has_positive_class else float("nan"),
                "recall": recall_score(y_true_day, y_pred_day, zero_division=0) if has_positive_class else float("nan"),
                "f1_score": f1_score(y_true_day, y_pred_day, zero_division=0) if has_positive_class else float("nan"),
            }
        )

    performance_dataframe = pd.DataFrame(daily_rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = range(len(performance_dataframe))
    ax.plot(x_positions, performance_dataframe["accuracy"], marker="o", linewidth=2, label="Accuracy")
    ax.plot(x_positions, performance_dataframe["precision"], marker="o", linewidth=2, label="Precision")
    ax.plot(x_positions, performance_dataframe["recall"], marker="o", linewidth=2, label="Recall")
    ax.plot(x_positions, performance_dataframe["f1_score"], marker="o", linewidth=2, label="F1")
    ax.set_title("Decision Tree - Daily Chronological Performance")
    ax.set_xlabel("Capture date")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x_positions), labels=performance_dataframe["capture_date"])
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main() -> None:
    dataframe = load_modeling_dataframe_with_metadata()
    validate_chronological_split(dataframe)

    print("Decision Tree chronological binary baseline")
    print(f"Input dataset: {INPUT_FILE}")
    print(f"Metadata dataset: {METADATA_FILE}")
    print("Feature set adaptation note: use the full 14-feature VM3 baseline unchanged for VM2 comparability.")
    print(f"Final feature list used: {NUMERICAL_FEATURES + CATEGORICAL_FEATURES}")
    print(f"\nTrain dates: {TRAIN_DATES}")
    print(f"Test dates: {TEST_DATES}")
    print_capture_date_distribution(dataframe)

    train_dataframe = dataframe.loc[dataframe[CAPTURE_DATE_COLUMN].isin(TRAIN_DATES)].copy()
    test_dataframe = dataframe.loc[dataframe[CAPTURE_DATE_COLUMN].isin(TEST_DATES)].copy()

    X_train = train_dataframe.loc[:, NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    X_test = test_dataframe.loc[:, NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y_train = train_dataframe[TARGET_COLUMN].astype(int)
    y_test = test_dataframe[TARGET_COLUMN].astype(int)

    print(f"\nTrain sample count: {len(X_train)}")
    print(f"Test sample count: {len(X_test)}")
    print_label_distribution("Train", y_train)
    print_label_distribution("Test", y_test)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_train_pred = pd.Series(pipeline.predict(X_train), index=y_train.index)
    y_test_pred = pd.Series(pipeline.predict(X_test), index=y_test.index)
    y_test_prob = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=y_test.index)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)
    attack_rows = build_attack_detection_rows(test_dataframe, y_test_pred)

    save_metrics(train_accuracy, train_f1, test_metrics, len(X_train), len(X_test), attack_rows)
    save_predictions(test_dataframe, y_test_pred, y_test_prob)

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
    print(f"{test_metrics['true_negatives']} {test_metrics['false_positives']}\n{test_metrics['false_negatives']} {test_metrics['true_positives']}")
    print("\nAttack-specific descriptive detection counts:")
    print(pd.DataFrame(attack_rows).to_string(index=False))
    print("\nClassification report:")
    print(test_metrics["classification_report"])

    create_feature_importance_outputs(pipeline)
    visualisations_displayed = False
    for visualisation_step in (
        lambda: create_confusion_matrix_figure(y_test, y_test_pred),
        lambda: create_roc_curve_figure(y_test, y_test_prob),
        lambda: create_daily_performance_figure(test_dataframe, y_test, y_test_pred),
    ):
        try:
            visualisation_step()
            visualisations_displayed = True
        except Exception as exc:
            # VM2 automation may run in a non-GUI environment where Tk/Tcl is unavailable.
            print(f"Visualisations skipped because the environment could not open matplotlib windows: {exc}")
            break

    print(f"Saved metrics to: {METRICS_OUTPUT_FILE}")
    print(f"Saved predictions to: {PREDICTIONS_OUTPUT_FILE}")
    if visualisations_displayed:
        print("Visualisations displayed interactively.")


if __name__ == "__main__":
    main()
