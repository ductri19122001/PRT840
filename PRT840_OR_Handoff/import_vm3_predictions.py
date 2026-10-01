"""Validate and import Duc's full original VM3 test predictions without retraining."""
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

from merge_or import (decision, integer, keyed, read_csv, read_json, sha256,
                      validate_predictions, write_csv, write_json)

ROOT = Path(__file__).resolve().parent


def normalize_rows(rows, protocol):
    required = {"flow_row_id", "true_label", "predicted_label", "predicted_probability"}
    normalized = []
    keyed(rows, "flow_row_id", "supplied VM3 export")
    for row in rows:
        if not required.issubset(row):
            raise ValueError("Supplied export is missing prediction columns")
        if row.get("evaluation_protocol", protocol) != protocol:
            raise ValueError("Supplied export declares an incompatible evaluation protocol")
        integer(row["flow_row_id"])
        decision(row["true_label"])
        decision(row["predicted_label"])
        probability = Decimal(row["predicted_probability"])
        if not probability.is_finite() or not 0 <= probability <= 1:
            raise ValueError("Prediction probability is outside [0, 1]")
        normalized.append({**row, "evaluation_protocol": protocol})
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "inputs/vm3/decision_tree_vm3_original_80_20_predictions.csv")
    args = parser.parse_args()
    source = args.input.resolve()
    config = read_json(ROOT / "config.json")
    spec = config["datasets"]["VM3"]
    original = read_csv(source)
    rows = normalize_rows(original, spec["dt_protocol"])
    metadata = keyed(read_csv(ROOT / spec["metadata"]), "flow_row_id", "VM3 metadata")
    indexed = validate_predictions(rows, metadata, spec["dt_protocol"], source.name)
    if len(indexed) != spec["recorded_dt_evaluation_count"]:
        raise ValueError("Supplied row count does not match the recorded original VM3 test population")
    samples = read_csv(ROOT / spec["dt_recorded_samples"])
    for sample in samples:
        fid = integer(sample["flow_row_id"])
        if fid not in indexed or any(Decimal(sample[f]) != Decimal(indexed[fid][f])
                                    for f in ("true_label", "predicted_label", "predicted_probability")):
            raise ValueError(f"Supplied export disagrees with recorded sample flow {fid}")
    output = ROOT / spec["dt_predictions"]
    if source == output:
        raise ValueError("Normalized output must remain separate from the original supplied file")
    write_csv(output, rows, list(rows[0]))
    write_json(ROOT / "inputs/vm3/dt_import_manifest.json", {
        "source_file": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name,
        "source_sha256": sha256(source), "source_columns": list(original[0]),
        "normalized_file": str(output.relative_to(ROOT)), "normalized_sha256": sha256(output),
        "records": len(rows), "matching_recorded_examples": len(samples),
        "identity_and_label_validation": "all records match the pinned VM3 metadata population",
        "evaluation_protocol": spec["dt_protocol"],
        "protocol_provenance": "The supplied filename identifies the original 80/20 export. Its rows agree with all recorded stratified_80_20 examples. The evaluation_protocol column is added from this declared import context; it was absent from the supplied CSV.",
        "unchanged_prediction_values": True, "retrained_model": False,
        "training_feature_provenance": "not established by a prediction CSV"})
    print(f"Imported {len(rows):,} predictions; {len(samples)} recorded examples agree.")


if __name__ == "__main__":
    main()
