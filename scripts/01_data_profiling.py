from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DATASET_ROOT = Path("Experiment-VM-Microsoft-Windows7full-3")
OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "conn_log_profile_summary.csv"
TARGET_FILENAME = "conn.log.labeled"


def find_conn_logs(root: Path) -> list[Path]:
    files = sorted(root.rglob(TARGET_FILENAME))
    if not files:
        raise FileNotFoundError(f"No {TARGET_FILENAME!r} files found under {root}")
    return files


def parse_zeek_header(file_path: Path) -> tuple[list[str], list[str]]:
    fields: list[str] | None = None
    types: list[str] | None = None

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]
            elif line.startswith("#types"):
                types = line.rstrip("\n").split("\t")[1:]
                break

    if not fields:
        raise ValueError(f"Missing #fields header in {file_path}")
    if not types:
        raise ValueError(f"Missing #types header in {file_path}")
    if len(fields) != len(types):
        raise ValueError(f"Header length mismatch in {file_path}")

    return fields, types


def zeek_type_to_dtype(zeek_type: str) -> str:
    numeric_types = {"count", "int", "port"}
    float_types = {"double", "interval", "time"}

    if zeek_type in numeric_types:
        return "Int64"
    if zeek_type in float_types:
        return "float64"
    if zeek_type == "bool":
        return "boolean"
    return "string"


def build_dtype_map(columns: Iterable[str], zeek_types: Iterable[str]) -> dict[str, str]:
    return {column: zeek_type_to_dtype(zeek_type) for column, zeek_type in zip(columns, zeek_types)}


def load_conn_log(file_path: Path) -> pd.DataFrame:
    fields, zeek_types = parse_zeek_header(file_path)
    dtype_map = build_dtype_map(fields, zeek_types)

    dataframe = pd.read_csv(
        file_path,
        sep="\t",
        comment="#",
        header=None,
        names=fields,
        na_values="-",
        keep_default_na=False,
        dtype=dtype_map,
        low_memory=False,
    )

    dataframe["source_file"] = str(file_path)
    dataframe["capture_date"] = file_path.parent.parent.name

    return dataframe


def distribution_frame(series: pd.Series, metric_name: str) -> pd.DataFrame:
    counts = series.fillna("<MISSING>").value_counts(dropna=False)
    frame = counts.rename_axis("value").reset_index(name="count")
    frame.insert(0, "metric", metric_name)
    return frame


def save_summary_report(
    dataframe: pd.DataFrame,
    label_distribution: pd.DataFrame,
    detailed_label_distribution: pd.DataFrame,
    missing_values: pd.Series,
    duplicate_count: int,
    timestamp_min: pd.Timestamp,
    timestamp_max: pd.Timestamp,
) -> None:
    summary_rows: list[dict[str, object]] = [
        {"section": "dataset", "item": "total_rows", "value": len(dataframe)},
        {"section": "dataset", "item": "total_columns", "value": dataframe.shape[1]},
        {"section": "dataset", "item": "duplicate_rows", "value": duplicate_count},
        {"section": "dataset", "item": "timestamp_min_utc", "value": timestamp_min.isoformat()},
        {"section": "dataset", "item": "timestamp_max_utc", "value": timestamp_max.isoformat()},
    ]

    summary_rows.extend(
        {"section": "column_dtype", "item": column, "value": str(dtype)}
        for column, dtype in dataframe.dtypes.items()
    )
    summary_rows.extend(
        {"section": "missing_values", "item": column, "value": int(count)}
        for column, count in missing_values.items()
    )
    summary_rows.extend(
        {"section": "label_distribution", "item": row["value"], "value": int(row["count"])}
        for _, row in label_distribution.iterrows()
    )
    summary_rows.extend(
        {"section": "detailed_label_distribution", "item": row["value"], "value": int(row["count"])}
        for _, row in detailed_label_distribution.iterrows()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_FILE, index=False)


def main() -> None:
    conn_logs = find_conn_logs(DATASET_ROOT)
    dataframes = [load_conn_log(file_path) for file_path in conn_logs]
    combined_df = pd.concat(dataframes, ignore_index=True)

    label_distribution = distribution_frame(combined_df["label"], "label")
    detailed_label_distribution = distribution_frame(combined_df["detailedlabel"], "detailedlabel")
    missing_values = combined_df.isna().sum().sort_values(ascending=False)
    duplicate_count = int(combined_df.duplicated().sum())

    timestamp_series = pd.to_datetime(combined_df["ts"], unit="s", utc=True, errors="coerce")
    timestamp_min = timestamp_series.min()
    timestamp_max = timestamp_series.max()

    if pd.isna(timestamp_min) or pd.isna(timestamp_max):
        raise ValueError("Unable to compute timestamp range from the ts column.")

    print(f"Files discovered: {len(conn_logs)}")
    print(f"Total rows: {combined_df.shape[0]}")
    print(f"Total columns: {combined_df.shape[1]}")
    print("\nColumn names:")
    print(list(combined_df.columns))
    print("\nData types:")
    print(combined_df.dtypes)
    print("\nLabel distribution:")
    print(label_distribution.to_string(index=False))
    print("\nDetailed-label distribution:")
    print(detailed_label_distribution.to_string(index=False))
    print("\nMissing values per column:")
    print(missing_values.to_string())
    print(f"\nDuplicate row count: {duplicate_count}")
    print("\nTimestamp range (UTC):")
    print(f"Start: {timestamp_min}")
    print(f"End:   {timestamp_max}")

    save_summary_report(
        dataframe=combined_df,
        label_distribution=label_distribution,
        detailed_label_distribution=detailed_label_distribution,
        missing_values=missing_values,
        duplicate_count=duplicate_count,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
    )
    print(f"\nSummary report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
