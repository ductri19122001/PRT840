"""Merge saved DT/IF decisions and build network cases triggered by positive alerts.

Python 3.10+; standard library only. No training or model inference is performed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META_FIELDS = ("ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "capture_date")
FEATURE_FIELDS = ("duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "proto", "service", "conn_state", "history")
MISSING = {"", "-", "missing", "nan", "None"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def integer(value):
    number = Decimal(str(value))
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"Expected an integer, got {value!r}")
    return int(number)


def keyed(rows, key, source):
    result = {}
    for row in rows:
        value = integer(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key} {value} in {source}")
        result[value] = row
    return result


def decision(value):
    value = integer(value)
    if value not in (0, 1):
        raise ValueError(f"Prediction is not binary: {value}")
    return value


def or_decision(dt, forest):
    # A missing prediction is unknown. One positive is sufficient for OR.
    if dt == 1 or forest == 1:
        return 1
    if dt == 0 and forest == 0:
        return 0
    return None


def flag_category(dt, forest):
    if dt is not None and forest is not None:
        return {(1, 1): "both_positive", (1, 0): "dt_positive_if_negative",
                (0, 1): "dt_negative_if_positive", (0, 0): "both_negative"}[(dt, forest)]
    if dt == 1:
        return "dt_positive_if_unavailable"
    if forest == 1:
        return "dt_unavailable_if_positive"
    return "undetermined"


def ground_truth(meta):
    return {"Benign": 0, "Malicious": 1}[meta["label"]]


def validate_predictions(rows, metadata, protocol, source):
    indexed = keyed(rows, "flow_row_id", source)
    for flow_id, row in indexed.items():
        if flow_id not in metadata:
            raise ValueError(f"{source}: flow {flow_id} is absent from this dataset")
        if row.get("evaluation_protocol") != protocol:
            raise ValueError(f"{source}: wrong protocol for flow {flow_id}: {row.get('evaluation_protocol')!r}")
        if decision(row["true_label"]) != ground_truth(metadata[flow_id]):
            raise ValueError(f"{source}: ground-truth mismatch for flow {flow_id}")
        decision(row["predicted_label"])
    return indexed


def utc_timestamp(value):
    seconds = Decimal(value)
    whole = int(seconds)
    microseconds = int((seconds - whole) * 1000000)
    return (datetime.fromtimestamp(whole, tz=timezone.utc) + timedelta(microseconds=microseconds)).isoformat(timespec="microseconds")


def network_row(dataset, flow_id, meta, features):
    row = {"dataset": dataset, "flow_row_id": flow_id,
           **{field: meta[field] for field in META_FIELDS},
           **{field: features[field] for field in FEATURE_FIELDS}}
    row["id.orig_p"] = integer(row["id.orig_p"])
    row["id.resp_p"] = integer(row["id.resp_p"])
    row["timestamp_utc"] = utc_timestamp(row["ts"])
    return row


def merge_dataset(dataset, metadata, features, dt, forest):
    if set(metadata) != set(features):
        raise ValueError(f"{dataset}: feature and metadata populations differ")
    if len({m["uid"] for m in metadata.values()}) != len(metadata):
        raise ValueError(f"{dataset}: duplicate Zeek UID")
    rows = []
    for flow_id in sorted(metadata):
        meta, feature = metadata[flow_id], features[flow_id]
        for field in ("id.orig_p", "id.resp_p"):
            if integer(meta[field]) != integer(feature[field]):
                raise ValueError(f"{dataset}: port mismatch for flow {flow_id}")
        if decision(feature["label"]) != ground_truth(meta):
            raise ValueError(f"{dataset}: feature label mismatch for flow {flow_id}")
        a = decision(dt[flow_id]["predicted_label"]) if flow_id in dt else None
        b = decision(forest[flow_id]["predicted_label"]) if flow_id in forest else None
        rows.append({**network_row(dataset, flow_id, meta, feature),
                     "dt_available": a is not None, "dt_prediction": a,
                     "if_available": b is not None, "if_prediction": b,
                     "jointly_evaluated": a is not None and b is not None,
                     "or_alert": or_decision(a, b), "flag_category": flag_category(a, b)})
    return rows


def classify(metrics, predicted, truth):
    metrics[{(1, 1): "tp", (1, 0): "fp", (0, 1): "fn", (0, 0): "tn"}[(predicted, truth)]] += 1


def overlap_summary(rows, metadata, dt_kind):
    joint = [r for r in rows if r["jointly_evaluated"]]
    truth = Counter(ground_truth(metadata[r["flow_row_id"]]) for r in joint)
    categories = Counter(r["flag_category"] for r in joint)
    metrics = {name: Counter({key: 0 for key in ("tp", "fp", "fn", "tn")}) for name in ("dt", "if", "or")}
    for row in joint:
        actual = ground_truth(metadata[row["flow_row_id"]])
        for name, field in (("dt", "dt_prediction"), ("if", "if_prediction"), ("or", "or_alert")):
            classify(metrics[name], row[field], actual)
    result = {"processed_dataset_records": len(rows), "dt_records_loaded": sum(r["dt_available"] for r in rows),
              "if_records_loaded": sum(r["if_available"] for r in rows), "dt_source_coverage": dt_kind,
              "jointly_evaluated_records": len(joint), "jointly_evaluated_malicious": truth[1],
              "jointly_evaluated_benign": truth[0], "joint_flag_categories": dict(categories),
              "dt_only_available": sum(r["dt_available"] and not r["if_available"] for r in rows),
              "if_only_available": sum(r["if_available"] and not r["dt_available"] for r in rows),
              "neither_available": sum(not r["dt_available"] and not r["if_available"] for r in rows),
              "confirmed_positive_or_alerts": sum(r["or_alert"] == 1 for r in rows),
              "confirmed_negative_or_decisions": sum(r["or_alert"] == 0 for r in rows),
              "undetermined_or_decisions": sum(r["or_alert"] is None for r in rows),
              "joint_counts_only": {k: dict(v) for k, v in metrics.items()},
              "overall_ensemble_accuracy": None, "overall_ensemble_false_positive_rate": None,
              "evaluation_limit": "Saved test populations differ. Joint counts concern their intersection, not a shared full test set. These counts cannot establish full-population ensemble accuracy or false-positive rates."}
    return result


def matches_scope(row, spec):
    scope = spec["scope"]
    if row["dataset"] != spec["dataset"]:
        return False
    for field in ("id.orig_h", "id.resp_h", "id.resp_p", "proto"):
        if str(row[field]) != str(scope[field]):
            return False
    day = row["timestamp_utc"][:10] if scope["time_basis"] == "utc_flow_start_date" else row["capture_date"]
    return day == scope["date"]


def build_case(rows, spec):
    context = [r for r in rows if matches_scope(r, spec)]
    triggers = [r for r in context if r["or_alert"] == 1]
    # An endpoint/date request alone cannot create an alert-led case.
    if not triggers:
        return None
    return {"case_id": spec["case_id"], "context": context, "triggers": triggers}


def public_flow(row):
    return {field: row[field] for field in ("dataset", "flow_row_id", "uid", "timestamp_utc", "capture_date", "ts", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", *FEATURE_FIELDS)}


def numeric_summary(rows, field):
    values = [Decimal(str(r[field])) for r in rows if str(r[field]) not in MISSING and r[field] is not None]
    if any(not v.is_finite() for v in values):
        raise ValueError(f"Nonfinite values in {field}")
    total = sum(values, Decimal(0))
    value = int(total) if total == total.to_integral_value() else str(total)
    return {"sum_available_values": value, "available_rows": len(values), "missing_rows": len(rows) - len(values)}


def observation_summary(rows):
    return {"context_flow_count": len(rows), "first_seen_utc": min(r["timestamp_utc"] for r in rows),
            "last_seen_utc": max(r["timestamp_utc"] for r in rows),
            "protocol_counts": dict(sorted(Counter(r["proto"] for r in rows).items())),
            "service_labels": dict(sorted(Counter("unset" if r["service"] in MISSING else r["service"] for r in rows).items())),
            "connection_states": dict(sorted(Counter(r["conn_state"] for r in rows).items())),
            "orig_bytes": numeric_summary(rows, "orig_bytes"), "resp_bytes": numeric_summary(rows, "resp_bytes"),
            "orig_packets": numeric_summary(rows, "orig_pkts"), "resp_packets": numeric_summary(rows, "resp_pkts")}


def retrieve_references(records, observations, top_k=5):
    # Network service labels drive lookup; family names and ground truth are excluded.
    query = "http https web protocols" if observations["service_labels"].get("http", 0) else "tcp network protocol"
    stop = {"the", "and", "or", "to", "of", "in", "a", "an", "for", "on", "with", "by", "is", "are", "as", "may", "can", "be"}
    def tokens(text):
        return [t for t in re.findall(r"[a-z][a-z0-9]+", text.lower()) if t not in stop]
    eligible = [r for r in records if set(r.get("platforms", [])) != {"PRE"}]
    counts = [Counter(tokens(" ".join([r["name"], " ".join(r["tactics"]), r["description"]]))) for r in eligible]
    lengths = [sum(c.values()) for c in counts]
    average = sum(lengths) / len(lengths)
    df = Counter(t for count in counts for t in count)
    ranked = []
    for record, count, length in zip(eligible, counts, lengths):
        score = sum(math.log(1 + (len(eligible) - df[t] + .5) / (df[t] + .5)) * count[t] * 2.5 /
                    (count[t] + 1.5 * (.25 + .75 * length / average)) for t in sorted(set(tokens(query))) if count[t])
        if score:
            ranked.append((score, record))
    ranked.sort(key=lambda value: (-value[0], value[1]["id"]))
    selected = [record for _, record in ranked[:top_k]]
    excerpts = [{"technique_id": r["id"], "name": r["name"], "tactics": r["tactics"],
                 "description": r["description"][:1600], "description_truncated": len(r["description"]) > 1600,
                 "source_url": r["url"]} for r in selected]
    return excerpts, {"algorithm": "BM25", "k1": 1.5, "b": .75, "query": query,
                      "query_source": "Network service facet only; lookup synonyms are not observed traffic.",
                      "selected_ids": [r["id"] for r in selected],
                      "limitation": "Retrieved definitions are reference material; retrieval does not establish observed ATT&CK behaviour."}


PROMPT = """You assist a human analyst interpreting a network case opened by ML alerts.
The alerts indicate suspicion, not confirmed malicious behaviour. Network records,
retrospective CTI and ATT&CK definitions are separate kinds of evidence. Dataset
labels are withheld. An endpoint association does not prove every connection is
malicious, that a host is compromised, or that a particular technique occurred.
TCP, a port number or a Zeek service label does not establish web C2, exfiltration,
credential theft or an attack stage. Zero available responder payload bytes does
not establish zero responder packets. Do not invent HTTP content or host events.
Only map a technique when supplied behavioural evidence supports its defining
behaviour. Otherwise say insufficient evidence; an empty mapping list is valid.
References are definitions, not observations. Identify uncertainty and missing
evidence. Suggest proportionate triage, corroboration and preservation actions,
with prerequisites for any disruptive response. Do not report a complete attack
chain or automatic containment. Do not recommend running suspicious files.
Return one JSON object with: observed_facts, supported_attack_mappings,
uncertainties, suggested_response_actions, human_review_required (true).

INPUT
"""


def prepare_payload(case, spec, records, cti):
    flows = [public_flow(r) for r in case["context"]]
    observations = observation_summary(flows)
    references, retrieval = retrieve_references(records, observations)
    # The case name, supplied family annotations and ground truth never enter this payload.
    payload = {"schema_version": "1.0", "case_id": spec["case_id"],
               "scope": spec["scope"], "ml_trigger_summary": {"or_alert_count": len(case["triggers"]),
               "jointly_evaluated_alert_count": sum(r["jointly_evaluated"] for r in case["triggers"]),
               "flag_categories": dict(Counter(r["flag_category"] for r in case["triggers"]))},
               "network_observations": observations, "retrospective_cti": cti,
               "attack_reference_material": references,
               "evidence_limits": ["Connection logs and saved detector decisions are supplied; no process, credential-access, file-content or HTTP request/response evidence is supplied.",
                                   "The processed dataset includes only records with binary dataset labels. Unknown-labelled records are absent from these snapshots.",
                                   "ML triggers and ATT&CK retrieval are not ground truth. Retrospective CTI is context, not proof of individual flow content.",
                                   "This is a retrospective replay for predeclared study endpoints and dates; it does not establish a historical automatic handoff."]}
    return payload, retrieval, flows


def verify_sources(root, manifest):
    for source in manifest["files"]:
        path = root / source["local_path"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {source['local_path']}")
        if "uncompressed_sha256" in source:
            digest = hashlib.sha256()
            with gzip.open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != source["uncompressed_sha256"]:
                raise ValueError(f"Uncompressed source checksum mismatch: {source['local_path']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--vm3-dt", type=Path, help="Original full VM3 stratified_80_20 DT prediction CSV, when available")
    args = parser.parse_args(argv)
    root = args.config.resolve().parent
    config, manifest = read_json(args.config), read_json(root / "source_manifest.json")
    verify_sources(root, manifest)
    protected = [root / "inputs", root / "references"]
    if args.output_dir.resolve() == root or any(args.output_dir.resolve() == p or p in args.output_dir.resolve().parents for p in protected):
        raise ValueError("Output directory cannot replace the package, inputs or references")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_status.json", {"status": "preparing", "llm_inputs_ready": False})
    all_rows, summaries = [], {}
    external_inputs = []
    for dataset, spec in config["datasets"].items():
        metadata = keyed(read_csv(root / spec["metadata"]), "flow_row_id", spec["metadata"])
        features = keyed(read_csv(root / spec["features"]), "flow_row_id", spec["features"])
        dt_path = args.vm3_dt.resolve() if dataset == "VM3" and args.vm3_dt else root / spec["dt_predictions"]
        dt_kind = "full_saved_prediction_export" if dataset == "VM3" and args.vm3_dt else spec["dt_coverage"]
        dt = validate_predictions(read_csv(dt_path), metadata, spec["dt_protocol"], str(dt_path.name))
        forest = validate_predictions(read_csv(root / spec["if_predictions"]), metadata, spec["if_protocol"], spec["if_predictions"])
        if dataset == "VM3" and dt_kind == "full_saved_prediction_export":
            expected = config["datasets"]["VM3"]["recorded_dt_evaluation_count"]
            if len(dt) != expected:
                raise ValueError(f"Expected {expected} records in the original full VM3 DT export, got {len(dt)}")
            # Every available recorded example must agree with the replacement.
            sample = read_csv(root / spec.get("dt_recorded_samples", spec["dt_predictions"]))
            for row in sample:
                fid = integer(row["flow_row_id"])
                if fid not in dt or decision(dt[fid]["predicted_label"]) != decision(row["predicted_label"]):
                    raise ValueError(f"Full VM3 DT export disagrees with recorded sample flow {fid}")
            if args.vm3_dt:
                external_inputs.append({"file_name": dt_path.name, "sha256": sha256(dt_path), "records": len(dt)})
        rows = merge_dataset(dataset, metadata, features, dt, forest)
        all_rows.extend(rows)
        summaries[dataset] = overlap_summary(rows, metadata, dt_kind)
    fieldnames = list(all_rows[0])
    write_csv(output / "or_decisions.csv.gz", all_rows, fieldnames)
    alerts = [row for row in all_rows if row["or_alert"] == 1]
    write_csv(output / "or_alerts.csv", alerts, fieldnames)
    groups = Counter((r["dataset"], r["id.orig_h"], r["id.resp_h"], r["id.resp_p"], r["proto"], r["timestamp_utc"][:10]) for r in alerts)
    group_fields = ("dataset", "source_ip", "destination_ip", "destination_port", "protocol", "utc_flow_start_date", "or_alert_count")
    write_csv(output / "alert_case_index.csv", [dict(zip(group_fields, (*key, count))) for key, count in sorted(groups.items())], group_fields)
    write_json(output / "model_overlap_summary.json", summaries)
    corpus_path = root / config["attack_corpus"]
    corpus_manifest = read_json(root / config["attack_corpus_manifest"])
    if sha256(corpus_path) != corpus_manifest["normalised_sha256"]:
        raise ValueError("ATT&CK corpus checksum mismatch")
    records = read_json(corpus_path)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate ATT&CK technique IDs")
    cases = []
    for spec in config["cases"]:
        case = build_case(all_rows, spec)
        case_dir = output / "cases" / spec["folder"]
        if case is None:
            status = {"case_id": spec["case_id"], "status": "no_positive_or_alert", "llm_input_prepared": False}
            write_json(case_dir / "handoff_trace.json", status)
            for filename in ("llm_input.json", "llm_prompt.txt", "context_flows.csv", "trigger_alerts.csv", "retrieval_trace.json"):
                (case_dir / filename).unlink(missing_ok=True)
            cases.append({"case_id": spec["case_id"], "status": "no_positive_or_alert"})
            continue
        cti = read_json(root / spec["cti_context"])
        payload, retrieval, flows = prepare_payload(case, spec, records, cti)
        write_csv(case_dir / "trigger_alerts.csv", case["triggers"], fieldnames)
        write_csv(case_dir / "context_flows.csv", flows, list(flows[0]))
        write_json(case_dir / "llm_input.json", payload)
        write_json(case_dir / "retrieval_trace.json", retrieval)
        (case_dir / "llm_prompt.txt").write_text(PROMPT + json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        trace = {"case_id": spec["case_id"], "study_name": spec["study_name"], "dataset": spec["dataset"],
                 "scope": spec["scope"], "or_alert_count": len(case["triggers"]), "context_flow_count": len(flows),
                 "trigger_flow_ids": [r["flow_row_id"] for r in case["triggers"]],
                 "trigger_uids": [r["uid"] for r in case["triggers"]],
                 "context_rule": "All available processed records matching the same endpoint and declared date scope, irrespective of model outcomes or family labels.",
                 "input_sha256": sha256(case_dir / "llm_input.json"),
                 "prompt_sha256": sha256(case_dir / "llm_prompt.txt"),
                 "context_csv_sha256": sha256(case_dir / "context_flows.csv"),
                 "trigger_csv_sha256": sha256(case_dir / "trigger_alerts.csv"),
                 "historical_run_status": "Earlier LLM runs used preselected cases; these newly generated inputs have not been submitted to Ollama by this script.",
                 "status": "alert_led_input_prepared"}
        write_json(case_dir / "handoff_trace.json", trace)
        cases.append(trace)
    run = {"schema_version": "1.0", "operation": "saved_prediction_or_merge_and_retrospective_case_preparation",
           "trained_models": False, "executed_model_inference": False, "executed_llm_generation": False,
           "missing_prediction_policy": "1 OR unknown = 1; 0 OR unknown = unknown; unknown is never replaced with 0.",
           "join_key": ["dataset", "flow_row_id"], "evaluation_protocols": config["datasets"],
           "source_manifest_sha256": sha256(root / "source_manifest.json"), "config_sha256": sha256(args.config),
           "script_sha256": sha256(Path(__file__)), "external_inputs": external_inputs,
           "processed_rows": len(all_rows), "positive_or_alerts": len(alerts), "automatic_endpoint_day_groups": len(groups),
           "cases": cases}
    write_json(output / "run_manifest.json", run)
    write_json(output / "run_status.json", {"status": "prepared", "llm_inputs_ready": True, "run_manifest_sha256": sha256(output / "run_manifest.json")})
    print(json.dumps({"positive_or_alerts": len(alerts), "cases": [{"case_id": c["case_id"], "status": c["status"], "or_alert_count": c.get("or_alert_count"), "context_flow_count": c.get("context_flow_count")} for c in cases]}, indent=2))


if __name__ == "__main__":
    main()
