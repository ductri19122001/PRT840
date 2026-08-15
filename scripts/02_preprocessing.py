from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DATASET_ROOT = Path("Experiment-VM-Microsoft-Windows7full-3")
OUTPUT_DIR = Path("outputs")
TARGET_FILENAME = "conn.log.labeled"
MODELLING_OUTPUT_FILE = OUTPUT_DIR / "clean_network_flows.csv"
CTI_METADATA_OUTPUT_FILE = OUTPUT_DIR / "cti_mapping_metadata.csv"

METADATA_COLUMNS = [
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "source_file",
    "capture_date",
    "label",
    "detailedlabel",
]

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
FEATURES_TO_EXCLUDE = [
    "detailedlabel",
    "id.orig_h",
    "id.resp_h",
    "uid",
    "source_file",
    "capture_date",
    "local_orig",
    "local_resp",
    "tunnel_parents",
]
LABEL_MAPPING = {"Benign": 0, "Malicious": 1}
MISSING_CHECK_COLUMNS = ["duration", "orig_bytes", "resp_bytes"]


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


def print_missing_distribution_by_label_and_state(dataframe: pd.DataFrame) -> None:
    for column in MISSING_CHECK_COLUMNS:
        distribution = (
            dataframe.assign(is_missing=dataframe[column].isna())
            .groupby(["label", "conn_state", "is_missing"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["label", "conn_state", "is_missing"])
        )
        print(f"\nMissing-value distribution for {column} by label and conn_state:")
        print(distribution.to_string(index=False))


def print_samples_per_capture_date(dataframe: pd.DataFrame) -> None:
    distribution = (
        dataframe.groupby(["capture_date", "label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["capture_date", "label"])
    )
    print("\nBenign and malicious samples per capture_date:")
    print(distribution.to_string(index=False))


def build_metadata_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    metadata = dataframe.loc[:, METADATA_COLUMNS].copy()
    metadata.insert(0, "flow_row_id", metadata.index)
    return metadata


def build_modelling_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    filtered_df = dataframe.loc[dataframe["label"] != "Unknown"].copy()
    filtered_df["label"] = filtered_df["label"].map(LABEL_MAPPING).astype("Int64")

    missing_labels = int(filtered_df["label"].isna().sum())
    if missing_labels:
        raise ValueError(f"Unexpected label values remain after mapping: {missing_labels}")

    filtered_df[CATEGORICAL_FEATURES] = filtered_df[CATEGORICAL_FEATURES].fillna("MISSING")

    modelling_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["label"]
    modelling_df = filtered_df.loc[:, modelling_columns].copy()
    modelling_df.insert(0, "flow_row_id", filtered_df.index)

    excluded_columns = [column for column in FEATURES_TO_EXCLUDE if column in filtered_df.columns]
    print("\nExcluded from ML features:")
    print(excluded_columns)
    print("\nLeakage note:")
    print(
        "No scaler, imputer, or encoder is fit here. Numeric missing values are preserved, "
        "and categorical missing values are replaced with the explicit category 'MISSING'."
    )

    return modelling_df


def save_outputs(modelling_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelling_df.to_csv(MODELLING_OUTPUT_FILE, index=False)
    metadata_df.to_csv(CTI_METADATA_OUTPUT_FILE, index=False)


def main() -> None:
    conn_logs = find_conn_logs(DATASET_ROOT)
    dataframes = [load_conn_log(file_path) for file_path in conn_logs]
    combined_df = pd.concat(dataframes, ignore_index=True)

    print(f"Files discovered: {len(conn_logs)}")
    print(f"Combined rows before filtering: {len(combined_df)}")
    print(f"Combined columns before filtering: {combined_df.shape[1]}")

    print_missing_distribution_by_label_and_state(combined_df)
    print_samples_per_capture_date(combined_df.loc[combined_df["label"].isin(LABEL_MAPPING)].copy())

    metadata_df = build_metadata_dataframe(combined_df)
    metadata_df = metadata_df.loc[metadata_df["label"] != "Unknown"].reset_index(drop=True)
    metadata_df["flow_row_id"] = metadata_df.index

    modelling_df = build_modelling_dataframe(combined_df)
    modelling_df["flow_row_id"] = range(len(modelling_df))

    print(f"\nUnknown rows removed from modelling baseline: {(combined_df['label'] == 'Unknown').sum()}")
    print(f"Rows retained for modelling: {len(modelling_df)}")
    print(f"Rows retained for CTI metadata mapping: {len(metadata_df)}")
    print("\nModelling dataframe columns:")
    print(list(modelling_df.columns))
    print("\nCTI metadata dataframe columns:")
    print(list(metadata_df.columns))

    save_outputs(modelling_df, metadata_df)
    print(f"\nSaved modelling dataframe to: {MODELLING_OUTPUT_FILE}")
    print(f"Saved CTI metadata dataframe to: {CTI_METADATA_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
