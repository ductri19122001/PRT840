from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = BASE_DIR / "CTU-SME-11" / "Experiment-VM-Microsoft-Windows7full-2"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "conn_log_profile_summary.csv"
TARGET_FILENAME = "conn.log.labeled"
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


def find_conn_logs(root: Path) -> list[Path]:
    files = sorted(root.rglob(TARGET_FILENAME))
    if not files:
        raise FileNotFoundError(f"No {TARGET_FILENAME!r} files found under {root}")
    return files


def parse_zeek_fields(file_path: Path) -> list[str]:
    fields: list[str] | None = None

    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#fields"):
                fields = line.rstrip("\r\n").split("\t")[1:]
                break

    if not fields:
        raise ValueError(f"Missing #fields header in {file_path}")

    return fields


def load_conn_log(file_path: Path) -> pd.DataFrame:
    fields = parse_zeek_fields(file_path)
    dataframe = pd.read_csv(
        file_path,
        sep="\t",
        comment="#",
        header=None,
        names=fields,
        dtype=str,
        na_values=["-", "(empty)"],
        keep_default_na=False,
        low_memory=False,
    )
    dataframe["source_file"] = str(file_path)
    dataframe["capture_date"] = file_path.parent.parent.name
    return dataframe


def distribution_frame(series: pd.Series, section: str) -> pd.DataFrame:
    counts = series.fillna("<MISSING>").value_counts(dropna=False)
    frame = counts.rename_axis("item").reset_index(name="value")
    frame.insert(0, "section", section)
    return frame


def build_numeric_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in NUMERICAL_FEATURES:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        rows.extend(
            [
                {"section": f"numeric_summary:{column}", "item": "non_null_count", "value": int(series.notna().sum())},
                {"section": f"numeric_summary:{column}", "item": "missing_count", "value": int(series.isna().sum())},
                {"section": f"numeric_summary:{column}", "item": "mean", "value": float(series.mean(skipna=True))},
                {"section": f"numeric_summary:{column}", "item": "median", "value": float(series.median(skipna=True))},
                {"section": f"numeric_summary:{column}", "item": "std", "value": float(series.std(skipna=True))},
                {"section": f"numeric_summary:{column}", "item": "min", "value": float(series.min(skipna=True))},
                {"section": f"numeric_summary:{column}", "item": "max", "value": float(series.max(skipna=True))},
            ]
        )
    return pd.DataFrame(rows)


def save_summary_report(dataframe: pd.DataFrame) -> None:
    label_distribution = distribution_frame(dataframe["label"], "label_distribution")
    detailed_label_distribution = distribution_frame(
        dataframe["detailedlabel"], "detailed_label_distribution"
    )
    missing_values = (
        dataframe.isna()
        .sum()
        .sort_values(ascending=False)
        .rename_axis("item")
        .reset_index(name="value")
    )
    missing_values.insert(0, "section", "missing_values")

    date_counts = (
        dataframe.groupby("capture_date")
        .size()
        .rename("value")
        .rename_axis("item")
        .reset_index()
    )
    date_counts.insert(0, "section", "records_per_date")

    date_label_counts = (
        dataframe.assign(label=dataframe["label"].fillna("<MISSING>"))
        .groupby(["capture_date", "label"], dropna=False)
        .size()
        .rename("value")
        .reset_index()
    )
    date_label_counts["item"] = date_label_counts["capture_date"] + " | " + date_label_counts["label"].astype(str)
    date_label_counts = date_label_counts.loc[:, ["item", "value"]]
    date_label_counts.insert(0, "section", "records_per_date_and_label")

    timestamp_series = pd.to_datetime(dataframe["ts"], unit="s", utc=True, errors="coerce")
    summary_rows = [
        {"section": "dataset", "item": "total_rows", "value": len(dataframe)},
        {"section": "dataset", "item": "total_columns", "value": dataframe.shape[1]},
        {"section": "dataset", "item": "files_discovered", "value": dataframe["source_file"].nunique()},
        {"section": "dataset", "item": "duplicate_rows", "value": int(dataframe.duplicated().sum())},
        {"section": "dataset", "item": "timestamp_min_utc", "value": timestamp_series.min().isoformat()},
        {"section": "dataset", "item": "timestamp_max_utc", "value": timestamp_series.max().isoformat()},
    ]

    dtype_rows = [
        {"section": "column_dtype", "item": column, "value": str(dtype)}
        for column, dtype in dataframe.dtypes.items()
    ]

    report = pd.concat(
        [
            pd.DataFrame(summary_rows),
            pd.DataFrame(dtype_rows),
            missing_values,
            date_counts,
            date_label_counts,
            label_distribution,
            detailed_label_distribution,
            build_numeric_summary(dataframe),
        ],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False)


def main() -> None:
    conn_logs = find_conn_logs(DATASET_ROOT)
    dataframes = [load_conn_log(file_path) for file_path in conn_logs]
    combined_df = pd.concat(dataframes, ignore_index=True)

    print(f"Files discovered: {len(conn_logs)}")
    print(f"Total rows: {len(combined_df)}")
    print("\nRecords per date:")
    print(combined_df.groupby("capture_date").size().to_string())
    print("\nLabel distribution by date:")
    print(
        combined_df.assign(label=combined_df["label"].fillna("<MISSING>"))
        .groupby(["capture_date", "label"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    print("\nDetailed label distribution:")
    print(combined_df["detailedlabel"].fillna("<MISSING>").value_counts(dropna=False).to_string())
    print("\nMissing values per column:")
    print(combined_df.isna().sum().sort_values(ascending=False).to_string())

    save_summary_report(combined_df)
    print(f"\nSummary report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
