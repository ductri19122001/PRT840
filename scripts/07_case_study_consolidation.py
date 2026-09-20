"""Consolidate saved flow evidence without fitting models or making threat inferences.

Run from any directory. Only the four outputs in outputs/case_studies are written.
Selection is deterministic: up to three lowest flow IDs per joint outcome and
per experiment's benign false positives, plus two IF-only examples per outcome.
All AgentTesla/LockBit records are included as small-sample descriptive evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ROOTS = {"VM3": BASE, "VM2": BASE.parent / "CTU-SME-11_Windows7full-2"}
OUT = BASE / "outputs" / "case_studies"
FEATURES = ["id.orig_p", "id.resp_p", "duration", "orig_bytes", "resp_bytes",
            "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
            "proto", "service", "conn_state", "history"]
META = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
        "source_file", "capture_date", "label", "detailedlabel"]
ATTACKS = {"VM3": {"RemcosRAT": "remcosrat"},
           "VM2": {"RedLineStealer": "redlinestealer", "AgentTesla": "agenttesla", "LockBit": "lockbit"}}
EXPERIMENTS = {
    "vm3_dt": {"dataset": "VM3", "training_dataset": "VM3", "model": "DT", "protocol": "stratified_80_20", "stem": "decision_tree_stratified", "script": "04.1_decision_tree_visualisation.py", "scope": "VM3_internal"},
    "vm3_if": {"dataset": "VM3", "training_dataset": "VM3", "model": "IF", "protocol": "common_random_holdout_primary", "stem": "isolation_forest_calibrated", "script": "05_isolation_forest_calibrated.py", "scope": "VM3_internal"},
    "vm2_dt": {"dataset": "VM2", "training_dataset": "VM2", "model": "DT", "protocol": "stratified_80_20", "stem": "decision_tree_stratified", "script": "04.1_decision_tree_visualisation.py", "scope": "VM2_standalone"},
    "vm2_if": {"dataset": "VM2", "training_dataset": "VM2", "model": "IF", "protocol": "common_random_holdout_primary", "stem": "isolation_forest_calibrated", "script": "05_isolation_forest_calibrated.py", "scope": "VM2_standalone"},
    "vm2_external_if": {"dataset": "VM2", "training_dataset": "VM3", "model": "IF", "protocol": "vm3_to_vm2_external_frozen_if", "stem": "isolation_forest_cross_vm_validation_vm2", "script": "06_isolation_forest_cross_vm_validation.py", "scope": "VM2_external"},
}
PAIRS = {"vm3_dt_if": ("vm3_dt", "vm3_if"),
         "vm2_standalone_dt_if": ("vm2_dt", "vm2_if"),
         "vm2_dt_external_if": ("vm2_dt", "vm2_external_if")}
CATEGORIES = {"A": "DT correct + IF correct", "B": "DT correct + IF missed",
              "C": "DT missed + IF correct", "D": "DT missed + IF missed"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader)


def index_rows(rows):
    result = {}
    for row in rows:
        row_id = int(row["flow_row_id"])
        if row_id in result:
            raise ValueError(f"Duplicate flow_row_id: {row_id}")
        result[row_id] = row
    return result


def number(value):
    if value in (None, ""):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite numeric evidence")
    return value


def joint_category(true_label, dt, forest):
    if dt is None or forest is None:
        return "unavailable"
    if true_label != 1:
        return "not_applicable_benign"
    return {(1, 1): "A", (1, 0): "B", (0, 1): "C", (0, 0): "D"}[
        (dt["predicted_label"], forest["predicted_label"])]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    for (dt, forest), expected in { (1, 1): "A", (1, 0): "B", (0, 1): "C", (0, 0): "D"}.items():
        assert joint_category(1, {"predicted_label": dt}, {"predicted_label": forest}) == expected
    assert joint_category(1, None, {"predicted_label": 0}) == "unavailable"
    assert joint_category(1, {"predicted_label": 1}, None) == "unavailable"
    assert joint_category(0, {"predicted_label": 1}, {"predicted_label": 1}) == "not_applicable_benign"
    try:
        index_rows([{"flow_row_id": "1"}, {"flow_row_id": "1"}])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate IDs accepted")
    assert number("") is None
    print("PASS: joint categories, unavailable predictions, benign handling, duplicate IDs, missing numerics")


def main():
    required_inputs = {}
    for root in ROOTS.values():
        for name in ("clean_network_flows.csv", "cti_mapping_metadata.csv"):
            required_inputs[root / "outputs" / name] = root / "scripts" / "02_preprocessing.py"
    for exp_id, spec in EXPERIMENTS.items():
        root = ROOTS["VM3"] if exp_id == "vm2_external_if" else ROOTS[spec["dataset"]]
        producer = root / "scripts" / spec["script"]
        required_inputs[root / "outputs" / (spec["stem"] + "_predictions.csv")] = producer
        metrics_name = "isolation_forest_cross_vm_validation_metrics.csv" if exp_id == "vm2_external_if" else spec["stem"] + "_metrics.csv"
        required_inputs[root / "outputs" / metrics_name] = producer
    missing = {path: producer for path, producer in required_inputs.items() if not path.is_file()}
    if missing:
        lines = ["Missing required experiment outputs:"]
        lines.extend(f"  {path}" for path in missing)
        lines.append("Restore these files, or generate them with the original experiment scripts:")
        lines.extend(f'  python "{producer}"' for producer in dict.fromkeys(missing.values()))
        lines.append("Then rerun script 07. Predictions from other protocols cannot replace these files.")
        raise SystemExit("\n".join(lines))

    validations = []
    sources = {}

    def check(name, condition):
        validations.append({"check": name, "status": "PASS" if condition else "FAIL"})
        if not condition:
            raise ValueError(f"FAIL: {name}")

    def source(path):
        path = path.resolve()
        sources[str(path)] = digest(path)
        return read_csv(path)

    protected = {str(p): digest(p) for root in ROOTS.values()
                 for p in (root / "scripts").glob("*.py") if re.match(r"0[1-6]", p.name)}
    clean, metadata = {}, {}
    for dataset, root in ROOTS.items():
        clean[dataset] = index_rows(source(root / "outputs" / "clean_network_flows.csv"))
        metadata[dataset] = index_rows(source(root / "outputs" / "cti_mapping_metadata.csv"))
        check(f"{dataset}: unique cleaned/metadata IDs and identical populations", clean[dataset].keys() == metadata[dataset].keys())
        check(f"{dataset}: required features and metadata columns", all(set(FEATURES + ["label"]) <= r.keys() for r in clean[dataset].values()) and all(set(META) <= r.keys() for r in metadata[dataset].values()))
        check(f"{dataset}: binary labels agree with metadata", all(r["label"] in ("0", "1") and metadata[dataset][i]["label"] == {"0": "Benign", "1": "Malicious"}[r["label"]] for i, r in clean[dataset].items()))

    predictions = {}
    registry = {}
    for exp_id, spec in EXPERIMENTS.items():
        dataset = spec["dataset"]
        exp_root = ROOTS["VM3"] if exp_id == "vm2_external_if" else ROOTS[dataset]
        path = exp_root / "outputs" / (spec["stem"] + "_predictions.csv")
        metrics_path = exp_root / "outputs" / ("isolation_forest_cross_vm_validation_metrics.csv" if exp_id == "vm2_external_if" else spec["stem"] + "_metrics.csv")
        raw = index_rows(source(path))
        metric_rows = source(metrics_path)
        if exp_id == "vm2_external_if":
            metric = next(r for r in metric_rows if r["row_type"] == "evaluation_metrics" and r["evaluation_dataset"] == "VM2_external")
            expected_count = int(float(metric["records_total"]))
            threshold = number(metric["threshold"])
            expected_cm = [int(float(metric[k])) for k in ("tn", "fp", "fn", "tp")]
        else:
            metric = {r["metric"]: r["value"] for r in metric_rows}
            expected_count = int(metric["test_samples"]) if spec["model"] == "DT" else int(metric["benign_test_records"]) + int(metric["malicious_test_records"])
            threshold = number(metric.get("calibrated_threshold"))
            expected_cm = [int(float(metric[k])) for k in ("true_negatives", "false_positives", "false_negatives", "true_positives")]
            check(f"{exp_id}: metrics protocol and random_state", metric["evaluation_protocol"] == spec["protocol"] and metric["random_state"] == "42")
        parsed = {}
        cm = Counter()
        for row_id, r in raw.items():
            if row_id not in clean[dataset] or r["true_label"] != clean[dataset][row_id]["label"]:
                raise ValueError(f"Prediction identity/label mismatch: {exp_id}:{row_id}")
            true, pred = int(r["true_label"]), int(r["predicted_label"])
            if pred not in (0, 1):
                raise ValueError("Invalid predicted label")
            for field in ("capture_date", "detailedlabel"):
                if field in r and r[field] != metadata[dataset][row_id][field]:
                    raise ValueError(f"Metadata mismatch: {exp_id}:{row_id}:{field}")
            if exp_id == "vm2_external_if":
                if r["dataset_source"] != "VM2" or r["evaluation_dataset"] != "VM2_external":
                    raise ValueError("External evaluation provenance mismatch")
            elif r["evaluation_protocol"] != spec["protocol"]:
                raise ValueError("Prediction protocol mismatch")
            probability, score = number(r.get("predicted_probability")), number(r.get("anomaly_score"))
            if spec["model"] == "DT":
                if probability is None or not 0 <= probability <= 1 or pred != int(probability > 0.5):
                    raise ValueError("Invalid DT probability/prediction")
            else:
                saved_threshold = number(r.get("frozen_threshold", r.get("calibrated_threshold")))
                if score is None or threshold is None or saved_threshold != threshold or pred != int(score > threshold):
                    raise ValueError("Invalid IF score/threshold/prediction")
            parsed[row_id] = {"identity": {"dataset": dataset, "evaluation_protocol": spec["protocol"], "experiment_id": exp_id, "flow_row_id": row_id},
                              "true_label": true, "predicted_label": pred, "predicted_probability": probability,
                              "anomaly_score": score, "threshold": threshold}
            cm[true, pred] += 1
        predictions[exp_id] = parsed
        check(f"{exp_id}: unique IDs, labels, metadata, provenance and prediction rules", len(raw) == len(parsed))
        check(f"{exp_id}: population and confusion matrix match saved metrics", len(parsed) == expected_count and [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]] == expected_cm)
        if spec["model"] == "IF":
            check(f"{exp_id}: every malicious flow evaluated", all(i in parsed for i, r in clean[dataset].items() if r["label"] == "1"))
        registry[exp_id] = {**spec, "random_state": 42, "prediction_file": str(path), "metrics_file": str(metrics_path),
                            "model_script": str(exp_root / "scripts" / spec["script"]), "evaluated_records": len(parsed),
                            "evaluated_benign": cm[0, 0] + cm[0, 1], "evaluated_malicious": cm[1, 0] + cm[1, 1],
                            "threshold": threshold,
                            "population_description": "All valid VM2 records; frozen VM3 preprocessing/model/99th-percentile threshold" if exp_id == "vm2_external_if" else "Label-stratified 20% test; 80% supervised training" if spec["model"] == "DT" else "15% benign test plus all malicious; 70% benign fit and 15% benign calibration"}
    check("External IF covers all VM2 records", predictions["vm2_external_if"].keys() == clean["VM2"].keys())
    check("External IF threshold equals VM3 calibrated threshold", registry["vm2_external_if"]["threshold"] == registry["vm3_if"]["threshold"])

    selected = defaultdict(set)
    coverage = []
    groups = {}
    for dataset, attacks in ATTACKS.items():
        for attack, token in attacks.items():
            groups[dataset, attack] = sorted(i for i, r in clean[dataset].items() if r["label"] == "1" and token in re.sub(r"[^a-z0-9]", "", metadata[dataset][i]["detailedlabel"].lower()))
    check("Primary attacks exist", bool(groups["VM3", "RemcosRAT"]) and bool(groups["VM2", "RedLineStealer"]))
    for (dataset, attack), ids in groups.items():
        if attack in ("AgentTesla", "LockBit"):
            for i in ids:
                selected[dataset, i].add(f"descriptive_small_sample:{attack}")
            continue
        for pair, (dt_id, if_id) in PAIRS.items():
            if EXPERIMENTS[dt_id]["dataset"] != dataset:
                continue
            buckets = defaultdict(list)
            for i in ids:
                buckets[joint_category(1, predictions[dt_id].get(i), predictions[if_id].get(i))].append(i)
            for category in CATEGORIES:
                chosen = buckets[category][:3]
                coverage.append({"dataset": dataset, "attack": attack, "pair": pair, "category": category, "eligible": len(buckets[category]), "selected": len(chosen)})
                for i in chosen:
                    selected[dataset, i].add(f"{pair}:{category}")
            for outcome in (0, 1):
                candidates = [i for i in ids if i not in predictions[dt_id] and i in predictions[if_id] and predictions[if_id][i]["predicted_label"] == outcome]
                for i in candidates[:2]:
                    selected[dataset, i].add(f"{pair}:DT_unavailable_IF_{'detected' if outcome else 'missed'}")
    for exp_id, rows in predictions.items():
        false_positives = sorted(i for i, r in rows.items() if r["true_label"] == 0 and r["predicted_label"] == 1)
        for i in false_positives[:3]:
            selected[EXPERIMENTS[exp_id]["dataset"], i].add(f"benign_false_positive:{exp_id}")

    attack_lookup = {(dataset, i): attack for (dataset, attack), ids in groups.items() for i in ids}
    cases, flat_rows = [], []
    for (dataset, row_id), reasons in sorted(selected.items()):
        raw, meta = clean[dataset][row_id], metadata[dataset][row_id]
        label = int(raw["label"])
        attack = attack_lookup.get((dataset, row_id), "Benign")
        evidence = {}
        flat = {"dataset": dataset, "flow_row_id": row_id, "case_label": attack, "true_label": label,
                "selection_reasons": ";".join(sorted(reasons)), "descriptive_only": attack in ("AgentTesla", "LockBit")}
        meta_values = {k: meta.get(k) or None for k in META}
        feature_values = {k: number(raw[k]) if k in FEATURES[:10] else raw[k] or None for k in FEATURES}
        flat.update({f"metadata_{k}": v for k, v in meta_values.items()})
        flat.update({f"feature_{k}": v for k, v in feature_values.items()})
        for exp_id, spec in EXPERIMENTS.items():
            applicable = spec["dataset"] == dataset
            prediction = predictions[exp_id].get(row_id) if applicable else None
            reason = None if prediction else "not_in_saved_evaluation_population" if applicable else "different_dataset"
            evidence[exp_id] = {"available": prediction is not None, "unavailable_reason": reason, "prediction": prediction}
            flat[f"{exp_id}_available"] = prediction is not None
            flat[f"{exp_id}_unavailable_reason"] = reason
            flat[f"{exp_id}_evaluation_protocol"] = spec["protocol"]
            flat[f"{exp_id}_prediction_identity"] = json.dumps(prediction["identity"], sort_keys=True) if prediction else None
            for field in ("predicted_label", "predicted_probability", "anomaly_score", "threshold"):
                flat[f"{exp_id}_{field}"] = prediction[field] if prediction else None
        joint = {}
        for pair, (dt_id, if_id) in PAIRS.items():
            category = joint_category(label, predictions[dt_id].get(row_id), predictions[if_id].get(row_id)) if EXPERIMENTS[dt_id]["dataset"] == dataset else "not_applicable_dataset"
            joint[pair] = category
            flat[pair + "_category"] = category
        cases.append({"dataset": dataset, "flow_row_id": row_id, "case_label": attack, "true_label": label,
                      "descriptive_only": attack in ("AgentTesla", "LockBit"), "selection_reasons": sorted(reasons),
                      "metadata": meta_values, "features": feature_values, "model_evidence": evidence, "joint_categories": joint})
        flat_rows.append(flat)

    summary = []
    for dataset in ROOTS:
        populations = {attack: ids for (ds, attack), ids in groups.items() if ds == dataset}
        populations["Benign"] = sorted(i for i, r in clean[dataset].items() if r["label"] == "0")
        for attack, ids in populations.items():
            for exp_id, spec in EXPERIMENTS.items():
                if spec["dataset"] != dataset:
                    continue
                evaluated = [predictions[exp_id][i] for i in ids if i in predictions[exp_id]]
                positives = sum(r["predicted_label"] for r in evaluated)
                selected_ids = [i for i in ids if (dataset, i) in selected]
                summary.append({"dataset": dataset, "case_label": attack, "experiment_id": exp_id,
                                "evaluation_scope": spec["scope"], "evaluation_protocol": spec["protocol"],
                                "training_dataset": spec["training_dataset"], "dataset_records": len(ids),
                                "evaluated_records": len(evaluated), "unavailable_records": len(ids) - len(evaluated),
                                "predicted_malicious_records": positives,
                                "detected_attack_records": positives if attack != "Benign" else None,
                                "missed_attack_records": len(evaluated) - positives if attack != "Benign" else None,
                                "false_positive_records": positives if attack == "Benign" else None,
                                "detection_rate_evaluated": positives / len(evaluated) if evaluated and attack != "Benign" else None,
                                "false_positive_rate_evaluated": positives / len(evaluated) if evaluated and attack == "Benign" else None,
                                "selected_cases": len(selected_ids), "selected_cases_evaluated": sum(i in predictions[exp_id] for i in selected_ids),
                                "descriptive_only": attack in ("AgentTesla", "LockBit"), "prediction_source": registry[exp_id]["prediction_file"]})

    check("Selected dataset/flow identities are unique", len(cases) == len({(r["dataset"], r["flow_row_id"]) for r in cases}))
    check("A-D categories require both exact experiment predictions", all(cat not in CATEGORIES or (case["model_evidence"][PAIRS[pair][0]]["available"] and case["model_evidence"][PAIRS[pair][1]]["available"]) for case in cases for pair, cat in case["joint_categories"].items()))
    check("Unavailable predictions remain null; available evidence retains full identity", all((item["prediction"] is None and item["unavailable_reason"] is not None) if not item["available"] else item["prediction"]["identity"] == {"dataset": case["dataset"], "evaluation_protocol": EXPERIMENTS[exp_id]["protocol"], "experiment_id": exp_id, "flow_row_id": case["flow_row_id"]} for case in cases for exp_id, item in case["model_evidence"].items()))
    check("Every selected benign case is a saved false positive", all(any(e["available"] and e["prediction"]["predicted_label"] == 1 for e in case["model_evidence"].values()) for case in cases if case["true_label"] == 0))
    check("Summary denominators reconcile", all(r["evaluated_records"] + r["unavailable_records"] == r["dataset_records"] and r["selected_cases_evaluated"] <= r["selected_cases"] for r in summary))
    check("All input CSV hashes unchanged", all(digest(Path(p)) == h for p, h in sources.items()))
    check("Scripts 01-06 unchanged in both projects", all(digest(Path(p)) == h for p, h in protected.items()))
    counts = Counter(case["case_label"] for case in cases)
    unavailable = {exp: sum(not c["model_evidence"][exp]["available"] for c in cases if c["dataset"] == spec["dataset"]) for exp, spec in EXPERIMENTS.items()}
    missing_fields = Counter(f"{section}.{field}" for case in cases for section in ("metadata", "features") for field, value in case[section].items() if value is None)
    policy = {"ordering": "dataset then integer flow_row_id", "joint_cases_per_attack_pair_category": 3,
              "if_only_cases_per_attack_pair_outcome": 2, "benign_false_positives_per_experiment": 3,
              "tie_break": "lowest integer flow_row_id", "descriptive_cases": "all AgentTesla and LockBit flows",
              "deduplication": "one case per dataset/flow_row_id; preserve all selection reasons",
              "unavailable": "JSON null / CSV blank with explicit availability and reason; never substitute predictions",
              "summary_denominator": "all saved evaluated records of each attack/benign group, not selected representatives",
              "representativeness": "illustrative deterministic examples, not a random or exhaustive sample"}
    document = {"schema_version": "1.0", "content_type": "observed_flow_and_saved_model_evidence_only",
                "label_origin": "provided dataset labels; not inferred attribution", "features": FEATURES,
                "selection_policy": policy, "joint_category_definitions": CATEGORIES,
                "experiment_registry": registry, "input_sha256": sources,
                "protected_script_sha256": protected, "selection_counts": dict(counts),
                "joint_category_coverage": coverage, "unavailable_predictions_in_selected_applicable_cases": unavailable,
                "missing_observed_fields": dict(missing_fields), "attack_summary": summary,
                "cases": cases, "validations": validations}
    payload = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    check("Evidence JSON serializes without NaN/Infinity", json.loads(payload)["cases"] == cases)
    # Regenerate after the serialization validation so every PASS is exported.
    payload = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    report = ["CASE STUDY CONSOLIDATION - SAVED EVIDENCE ONLY", "No training, inference, MITRE mapping, attack chains, Ollama calls or response generation performed.",
              "VM3 DT: stratified 80/20. VM3 IF: benign-only 70/15/15; all malicious records tested.",
              "VM2 standalone DT/IF and VM3 -> VM2 external IF retain independent identities.",
              "The VM2 DT/external IF pair combines VM2 standalone DT with frozen VM3 IF; it is not a cross-VM DT experiment.",
              "Source consistency checks do not independently reproduce model fitting or establish historical run provenance.",
              "Selection policy: " + json.dumps(policy, sort_keys=True), "Selected unique cases: " + json.dumps(dict(counts), sort_keys=True),
              "Unavailable applicable predictions: " + json.dumps(unavailable, sort_keys=True),
              "Missing observed fields: " + json.dumps(dict(missing_fields), sort_keys=True),
              "DT anomaly_score/threshold and IF predicted_probability are not applicable and remain null/blank.",
              "Models for the other dataset are marked different_dataset, even if numeric flow IDs coincide.",
              "No fitted model artifacts, calibration score vectors or training membership files are introduced.",
              "JOINT CATEGORY COVERAGE (zero eligible means unavailable, not synthesized):"]
    report.extend(json.dumps(r, sort_keys=True) for r in coverage)
    report.append("VALIDATIONS:")
    report.extend(f"{v['status']}: {v['check']}" for v in validations)
    report.append("INPUT SHA256:")
    report.extend(f"{h}  {p}" for p, h in sorted(sources.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "case_study_flows.csv", flat_rows)
    write_csv(OUT / "case_study_attack_summary.csv", summary)
    (OUT / "case_study_llm_input.json").write_text(payload, encoding="utf-8")
    (OUT / "case_study_consolidation_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:13]))
    print("\n".join(f"{v['status']}: {v['check']}" for v in validations))
    print()
    print(f"Saved case-study flows ({len(flat_rows)} rows) to: {OUT / 'case_study_flows.csv'}")
    print(f"Saved attack summary ({len(summary)} rows) to: {OUT / 'case_study_attack_summary.csv'}")
    print(f"Saved LLM evidence input to: {OUT / 'case_study_llm_input.json'}")
    print(f"Saved consolidation report to: {OUT / 'case_study_consolidation_report.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run focused invariant tests without reading/writing project data")
    args = parser.parse_args()
    self_test() if args.self_test else main()
