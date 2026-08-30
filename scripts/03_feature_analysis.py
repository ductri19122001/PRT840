from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "outputs" / "clean_network_flows.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "feature_analysis"

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
LABEL_NAMES = {0: "Benign", 1: "Malicious"}
TOP_CATEGORY_COUNT = 10


def load_modelling_data() -> pd.DataFrame:
    dataframe = pd.read_csv(INPUT_FILE)
    expected_labels = set(LABEL_NAMES)
    observed_labels = set(dataframe[TARGET_COLUMN].dropna().unique())
    if not observed_labels.issubset(expected_labels):
        raise ValueError(f"Unexpected labels found in {TARGET_COLUMN}: {sorted(observed_labels)}")
    return dataframe


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator / denominator)


def summarize_numerical_feature(dataframe: pd.DataFrame, feature: str) -> dict[str, object]:
    series = pd.to_numeric(dataframe[feature], errors="coerce")
    benign = series[dataframe[TARGET_COLUMN] == 0]
    malicious = series[dataframe[TARGET_COLUMN] == 1]

    overall_std = float(series.std(skipna=True))
    benign_mean = float(benign.mean(skipna=True))
    malicious_mean = float(malicious.mean(skipna=True))
    benign_std = float(benign.std(skipna=True))
    malicious_std = float(malicious.std(skipna=True))
    pooled_std = np.sqrt(np.nanmean([benign_std**2, malicious_std**2]))
    mean_gap = abs(malicious_mean - benign_mean)
    effect_size = safe_ratio(mean_gap, pooled_std)
    missing_percentage = float(series.isna().mean() * 100.0)
    dominant_value_ratio = float(series.value_counts(normalize=True, dropna=False).iloc[0]) if len(series) else 0.0
    unique_non_null = int(series.nunique(dropna=True))

    summary: dict[str, object] = {
        "feature": feature,
        "type": "numerical",
        "missing_percentage": round(missing_percentage, 4),
        "unique_non_null": unique_non_null,
        "dominant_value_ratio": round(dominant_value_ratio, 6),
        "overall_mean": round(float(series.mean(skipna=True)), 6),
        "overall_median": round(float(series.median(skipna=True)), 6),
        "overall_std": round(overall_std, 6),
        "overall_min": round(float(series.min(skipna=True)), 6),
        "overall_q25": round(float(series.quantile(0.25)), 6),
        "overall_q50": round(float(series.quantile(0.50)), 6),
        "overall_q75": round(float(series.quantile(0.75)), 6),
        "overall_max": round(float(series.max(skipna=True)), 6),
        "benign_mean": round(benign_mean, 6),
        "benign_median": round(float(benign.median(skipna=True)), 6),
        "benign_std": round(benign_std, 6),
        "benign_min": round(float(benign.min(skipna=True)), 6),
        "benign_q25": round(float(benign.quantile(0.25)), 6),
        "benign_q50": round(float(benign.quantile(0.50)), 6),
        "benign_q75": round(float(benign.quantile(0.75)), 6),
        "benign_max": round(float(benign.max(skipna=True)), 6),
        "malicious_mean": round(malicious_mean, 6),
        "malicious_median": round(float(malicious.median(skipna=True)), 6),
        "malicious_std": round(malicious_std, 6),
        "malicious_min": round(float(malicious.min(skipna=True)), 6),
        "malicious_q25": round(float(malicious.quantile(0.25)), 6),
        "malicious_q50": round(float(malicious.quantile(0.50)), 6),
        "malicious_q75": round(float(malicious.quantile(0.75)), 6),
        "malicious_max": round(float(malicious.max(skipna=True)), 6),
        "mean_gap": round(mean_gap, 6),
        "standardized_mean_gap": round(effect_size, 6),
        "relevance_evidence": (
            f"effect_size={effect_size:.3f}; "
            f"dominant_value_ratio={dominant_value_ratio:.3f}; "
            "retain fixed VM3 baseline feature set for comparability"
        ),
        "recommendation": "KEEP",
    }
    return summary


def summarize_categorical_feature(dataframe: pd.DataFrame, feature: str) -> tuple[dict[str, object], pd.DataFrame]:
    series = dataframe[feature].astype("string")
    normalized = series.fillna("<MISSING>")
    counts = normalized.value_counts(dropna=False)
    top_categories = counts.head(TOP_CATEGORY_COUNT).index.tolist()
    rows: list[dict[str, object]] = []
    strongest_category: str | None = None
    strongest_gap = -1.0

    for category in top_categories:
        mask = normalized == category
        subset = dataframe.loc[mask, TARGET_COLUMN]
        benign_count = int((subset == 0).sum())
        malicious_count = int((subset == 1).sum())
        total_count = benign_count + malicious_count
        malicious_rate = safe_ratio(malicious_count, total_count)
        association_gap = abs(malicious_rate - float(dataframe[TARGET_COLUMN].mean()))

        if association_gap > strongest_gap:
            strongest_gap = association_gap
            strongest_category = str(category)

        rows.append(
            {
                "feature": feature,
                "category": category,
                "count": total_count,
                "percentage": round(float(mask.mean() * 100.0), 4),
                "benign_count": benign_count,
                "malicious_count": malicious_count,
                "benign_percentage_within_category": round(safe_ratio(benign_count * 100.0, total_count), 4),
                "malicious_percentage_within_category": round(safe_ratio(malicious_count * 100.0, total_count), 4),
            }
        )

    top_category_frame = pd.DataFrame(rows)
    base_malicious_rate = float(dataframe[TARGET_COLUMN].mean())
    missing_percentage = float(series.isna().mean() * 100.0)
    unique_categories = int(series.nunique(dropna=True))
    dominant_value_ratio = float(counts.iloc[0] / len(dataframe)) if len(dataframe) else 0.0

    evidence_parts = [
        f"unique_categories={unique_categories}",
        f"dominant_value_ratio={dominant_value_ratio:.3f}",
        "retain fixed VM3 baseline feature set for comparability",
    ]

    if strongest_category is not None:
        strongest_row = top_category_frame.loc[top_category_frame["category"] == strongest_category].iloc[0]
        strongest_rate = strongest_row["malicious_percentage_within_category"] / 100.0
        leaning = "Malicious" if strongest_rate >= base_malicious_rate else "Benign"
        evidence_parts.append(
            f"top association: {strongest_category} -> {leaning} "
            f"({strongest_row['malicious_percentage_within_category']:.1f}% malicious)"
        )

    summary = {
        "feature": feature,
        "type": "categorical",
        "missing_percentage": round(missing_percentage, 4),
        "unique_categories": unique_categories,
        "dominant_value_ratio": round(dominant_value_ratio, 6),
        "top_categories": ", ".join(str(category) for category in top_categories[:5]),
        "relevance_evidence": "; ".join(evidence_parts),
        "recommendation": "KEEP",
    }
    return summary, top_category_frame


def build_recommendation_table(
    numerical_rows: list[dict[str, object]],
    categorical_rows: list[dict[str, object]],
) -> pd.DataFrame:
    combined_rows = []
    for row in numerical_rows + categorical_rows:
        combined_rows.append(
            {
                "feature": row["feature"],
                "type": row["type"],
                "missing_percentage": row["missing_percentage"],
                "relevance_evidence": row["relevance_evidence"],
                "recommendation": row["recommendation"],
            }
        )
    return pd.DataFrame(combined_rows)


def print_terminal_summary(
    dataframe: pd.DataFrame,
    numerical_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    label_counts = dataframe[TARGET_COLUMN].map(LABEL_NAMES).value_counts().rename_axis("label").reset_index(name="count")
    print(f"Feature analysis dataset: {INPUT_FILE}")
    print(f"Rows analysed: {len(dataframe)}")
    print("\nClass distribution:")
    print(label_counts.to_string(index=False))
    print("\nNumerical feature summary:")
    print(numerical_summary.loc[:, ["feature", "missing_percentage", "standardized_mean_gap", "recommendation"]].to_string(index=False))
    print("\nCategorical feature summary:")
    print(categorical_summary.loc[:, ["feature", "missing_percentage", "unique_categories", "recommendation"]].to_string(index=False))
    print("\nConcise feature relevance table:")
    print(recommendations.to_string(index=False))


def main() -> None:
    dataframe = load_modelling_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    numerical_rows = [summarize_numerical_feature(dataframe, feature) for feature in NUMERICAL_FEATURES]
    numerical_summary = pd.DataFrame(numerical_rows)

    categorical_rows: list[dict[str, object]] = []
    categorical_distribution_frames: list[pd.DataFrame] = []
    for feature in CATEGORICAL_FEATURES:
        row, detail_frame = summarize_categorical_feature(dataframe, feature)
        categorical_rows.append(row)
        categorical_distribution_frames.append(detail_frame)

    categorical_summary = pd.DataFrame(categorical_rows)
    categorical_detail = pd.concat(categorical_distribution_frames, ignore_index=True)
    recommendations = build_recommendation_table(numerical_rows, categorical_rows)

    numerical_summary.to_csv(OUTPUT_DIR / "numerical_feature_summary.csv", index=False)
    categorical_summary.merge(categorical_detail, on="feature", how="left").to_csv(
        OUTPUT_DIR / "categorical_feature_summary.csv",
        index=False,
    )
    recommendations.to_csv(OUTPUT_DIR / "feature_relevance_recommendations.csv", index=False)

    print_terminal_summary(dataframe, numerical_summary, categorical_summary, recommendations)
    print(f"\nSaved numerical summary to: {OUTPUT_DIR / 'numerical_feature_summary.csv'}")
    print(f"Saved categorical summary to: {OUTPUT_DIR / 'categorical_feature_summary.csv'}")
    print(f"Saved recommendations to: {OUTPUT_DIR / 'feature_relevance_recommendations.csv'}")


if __name__ == "__main__":
    main()
