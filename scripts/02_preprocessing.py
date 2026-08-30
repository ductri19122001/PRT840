from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = BASE_DIR / "CTU-SME-11" / "Experiment-VM-Microsoft-Windows7full-2"
OUTPUT_DIR = BASE_DIR / "outputs"
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

CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "conn_state",
    "history",
]

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

LABEL_MAPPING = {
    "Benign": 0,
    "Malicious": 1,
}

MISSING_CHECK_COLUMNS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
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
        low_memory=False,
    )

    dataframe["source_file"] = str(file_path)
    dataframe["capture_date"] = file_path.parent.parent.name

    return dataframe


def convert_numerical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    for column in NUMERICAL_FEATURES:
        if column not in dataframe.columns:
            raise KeyError(f"Required numerical feature {column!r} is missing from the dataset.")
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    return dataframe


def prepare_categorical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    for column in CATEGORICAL_FEATURES:
        if column not in dataframe.columns:
            raise KeyError(f"Required categorical feature {column!r} is missing from the dataset.")
        dataframe[column] = dataframe[column].fillna("missing").astype(str)

    return dataframe


def print_missing_distribution_by_label_and_state(dataframe: pd.DataFrame) -> None:
    print("\nMissing-value distribution for selected numerical features:")

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
    available_columns = [column for column in METADATA_COLUMNS if column in dataframe.columns]
    return dataframe.loc[:, available_columns].copy()


def build_modelling_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    filtered_df = dataframe.loc[dataframe["label"].isin(LABEL_MAPPING.keys())].copy()
    filtered_df["label"] = filtered_df["label"].map(LABEL_MAPPING).astype("Int64")

    if int(filtered_df["label"].isna().sum()):
        raise ValueError("Unexpected label values remain after mapping.")

    filtered_df = convert_numerical_features(filtered_df)
    filtered_df = prepare_categorical_features(filtered_df)

    modelling_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["label"]
    return filtered_df.loc[:, modelling_columns].copy()


def validate_alignment(modelling_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    if len(modelling_df) != len(metadata_df):
        raise ValueError("Modelling and metadata dataframes have different row counts.")
    if modelling_df["flow_row_id"].duplicated().any():
        raise ValueError("Duplicate flow_row_id values found in modelling dataframe.")
    if metadata_df["flow_row_id"].duplicated().any():
        raise ValueError("Duplicate flow_row_id values found in metadata dataframe.")


def save_outputs(modelling_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelling_df.to_csv(MODELLING_OUTPUT_FILE, index=False)
    metadata_df.to_csv(CTI_METADATA_OUTPUT_FILE, index=False)


def main() -> None:
    conn_logs = find_conn_logs(DATASET_ROOT)
    print(f"Files discovered: {len(conn_logs)}")
    for file_path in conn_logs:
        print(f"  - {file_path}")

    dataframes = [load_conn_log(file_path) for file_path in conn_logs]
    combined_df = pd.concat(dataframes, ignore_index=True)

    print(f"\nCombined rows before filtering: {len(combined_df)}")
    print(f"Combined columns before filtering: {combined_df.shape[1]}")

    diagnostic_df = convert_numerical_features(combined_df)
    print_missing_distribution_by_label_and_state(diagnostic_df)

    binary_rows = combined_df.loc[combined_df["label"].isin(LABEL_MAPPING.keys())].copy()
    print_samples_per_capture_date(binary_rows)

    unknown_count = int((combined_df["label"] == "Unknown").sum())
    binary_df = combined_df.loc[combined_df["label"].isin(LABEL_MAPPING.keys())].copy()
    binary_df = binary_df.reset_index(drop=True)
    binary_df.insert(0, "flow_row_id", range(len(binary_df)))

    metadata_df = build_metadata_dataframe(binary_df)
    metadata_df.insert(0, "flow_row_id", binary_df["flow_row_id"])

    modelling_df = build_modelling_dataframe(binary_df)
    modelling_df.insert(0, "flow_row_id", binary_df["flow_row_id"])

    excluded_columns = [column for column in FEATURES_TO_EXCLUDE if column in combined_df.columns]
    print("\nExcluded from baseline ML features:")
    print(excluded_columns)
    print("\nPreprocessing / leakage note:")
    print(
        "Raw Zeek missing markers '-' and '(empty)' are treated as missing. "
        "Numerical features are converted using pd.to_numeric(errors='coerce'). "
        "Categorical missing values are represented explicitly as 'missing'. "
        "No numerical imputer, encoder, scaler, or ML model is fitted in this script. "
        "Training-only learned preprocessing must be performed later inside the model pipeline."
    )

    validate_alignment(modelling_df, metadata_df)

    print(f"\nUnknown rows removed from binary baseline: {unknown_count}")
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
