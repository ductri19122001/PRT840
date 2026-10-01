"""Independently reconcile saved outputs to the source snapshots (standard library)."""
from __future__ import annotations

import csv
import argparse
import gzip
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--vm3-dt", type=Path, help="Same original full VM3 DT CSV supplied when generating these outputs")
    args = parser.parse_args()
    config = read_json(ROOT / "config.json")
    out = args.output_dir.resolve()
    run = read_json(out / "run_manifest.json")
    if run["external_inputs"]:
        if not args.vm3_dt:
            raise SystemExit("These outputs use the full VM3 DT export. Supply the same file with --vm3-dt to verify them.")
        if digest(args.vm3_dt) != run["external_inputs"][0]["sha256"]:
            raise SystemExit("The supplied full VM3 DT export does not match the generation manifest.")
    elif args.vm3_dt:
        raise SystemExit("These outputs used the bundled prediction sources. Run merge_or.py with the external file before verifying it.")
    checks = []
    def check(name, condition):
        checks.append({"check": name, "status": "PASS" if condition else "FAIL"})
        if not condition:
            raise AssertionError(name)

    for item in read_json(ROOT / "source_manifest.json")["files"]:
        check("Source checksum: " + item["local_path"], digest(ROOT / item["local_path"]) == item["sha256"])
    decisions = read_csv(out / "or_decisions.csv.gz")
    alerts = read_csv(out / "or_alerts.csv")
    key = lambda r: (r["dataset"], int(r["flow_row_id"]))
    decisions_by_id = {key(r): r for r in decisions}
    alert_ids = {key(r) for r in alerts}
    check("One merged row per dataset/flow identity", len(decisions_by_id) == len(decisions))
    check("Alert export contains positive decisions only", all(r["or_alert"] == "1" for r in alerts))
    metadata = {}
    expected_positive = set()
    expected_negative = set()
    expected_all = set()
    for dataset, spec in config["datasets"].items():
        metadata[dataset] = {int(r["flow_row_id"]): r for r in read_csv(ROOT / spec["metadata"])}
        dt_path = args.vm3_dt if dataset == "VM3" and args.vm3_dt else ROOT / spec["dt_predictions"]
        dt = {int(r["flow_row_id"]): int(r["predicted_label"]) for r in read_csv(dt_path)}
        forest = {int(r["flow_row_id"]): int(r["predicted_label"]) for r in read_csv(ROOT / spec["if_predictions"])}
        positives = {i for i, label in dt.items() if label == 1} | {i for i, label in forest.items() if label == 1}
        negatives = {i for i, label in dt.items() if label == 0} & {i for i, label in forest.items() if label == 0}
        expected_positive.update((dataset, i) for i in positives)
        expected_negative.update((dataset, i) for i in negatives)
        expected_all.update((dataset, i) for i in metadata[dataset])
        check(dataset + " loaded DT/IF decisions agree with source CSVs", all(
            decisions_by_id[(dataset, i)]["dt_prediction"] == str(dt[i]) if i in dt else decisions_by_id[(dataset, i)]["dt_prediction"] == ""
            for i in metadata[dataset]) and all(
            decisions_by_id[(dataset, i)]["if_prediction"] == str(forest[i]) if i in forest else decisions_by_id[(dataset, i)]["if_prediction"] == ""
            for i in metadata[dataset]))
        check(dataset + " merged identities agree with metadata", all(
            decisions_by_id[(dataset, i)]["uid"] == m["uid"] and
            all(decisions_by_id[(dataset, i)][field] == m[field] for field in ("id.orig_h", "id.resp_h", "id.resp_p", "capture_date"))
            for i, m in metadata[dataset].items()))
    check("Merged population exactly matches the available processed datasets", set(decisions_by_id) == expected_all)
    check("Alert IDs exactly equal the union of source-model positive IDs", alert_ids == expected_positive)
    check("Negative OR decisions exactly equal the intersection of source-model negative IDs", {key(r) for r in decisions if r["or_alert"] == "0"} == expected_negative)
    check("All other merged decisions remain undetermined", {key(r) for r in decisions if r["or_alert"] == ""} == expected_all - expected_positive - expected_negative)

    original = read_json(ROOT / "inputs/original_case_study_consolidation.json")
    recorded = {c["flow_row_id"]: c["model_evidence"]["vm3_dt"]["prediction"]["predicted_label"]
                for c in original["cases"] if c["dataset"] == "VM3" and c["model_evidence"]["vm3_dt"]["available"]}
    vm3_spec = config["datasets"]["VM3"]
    extracted = {int(r["flow_row_id"]): int(r["predicted_label"]) for r in read_csv(ROOT / vm3_spec.get("dt_recorded_samples", vm3_spec["dt_predictions"]))}
    check("VM3 DT examples exactly reproduce the recorded sample, without invented rows", recorded == extracted and len(extracted) == 11)
    if vm3_spec["dt_coverage"] == "full_saved_prediction_export":
        imported = read_json(ROOT / "inputs/vm3/dt_import_manifest.json")
        raw = read_csv(ROOT / imported["source_file"])
        normalized = read_csv(ROOT / imported["normalized_file"])
        check("Original supplied VM3 export checksum agrees", digest(ROOT / imported["source_file"]) == imported["source_sha256"])
        check("Normalized VM3 export checksum agrees", digest(ROOT / imported["normalized_file"]) == imported["normalized_sha256"])
        check("Full VM3 export retains all 27,953 original prediction rows", len(raw) == len(normalized) == imported["records"] == vm3_spec["recorded_dt_evaluation_count"])
        check("Normalization changes no original prediction values", all(all(n[k] == r[k] for k in r) and n["evaluation_protocol"] == vm3_spec["dt_protocol"] for r, n in zip(raw, normalized)))
        full = {int(r["flow_row_id"]): int(r["predicted_label"]) for r in normalized}
        check("Full VM3 export agrees with every previously recorded example", all(full.get(i) == p for i, p in extracted.items()))

    case_counts = []
    for spec in config["cases"]:
        directory = out / "cases" / spec["folder"]
        context = read_csv(directory / "context_flows.csv")
        triggers = read_csv(directory / "trigger_alerts.csv")
        trace = read_json(directory / "handoff_trace.json")
        payload = read_json(directory / "llm_input.json")
        scope = spec["scope"]
        expected_context = set()
        for i, meta in metadata[spec["dataset"]].items():
            endpoint_matches = all(str(meta[f]) == str(scope[f]) for f in ("id.orig_h", "id.resp_h", "id.resp_p"))
            merged = decisions_by_id[(spec["dataset"], i)]
            date = merged["timestamp_utc"][:10] if scope["time_basis"] == "utc_flow_start_date" else meta["capture_date"]
            if endpoint_matches and date == scope["date"] and merged["proto"] == scope["proto"]:
                expected_context.add((spec["dataset"], i))
        context_ids, trigger_ids = {key(r) for r in context}, {key(r) for r in triggers}
        check(spec["folder"] + " context uses the complete declared endpoint/date scope", context_ids == expected_context and len(context_ids) == len(context))
        check(spec["folder"] + " trigger set equals source positive IDs within its context", trigger_ids == expected_positive & expected_context and len(trigger_ids) == len(triggers) and len(trigger_ids) > 0)
        check(spec["folder"] + " trace retains exact trigger flow identities", set(trace["trigger_flow_ids"]) == {i for _, i in trigger_ids} and set(trace["trigger_uids"]) == {r["uid"] for r in triggers})
        check(spec["folder"] + " context does not contain dataset annotations", all("label" not in r and "detailedlabel" not in r and "source_file" not in r for r in context))
        for name, field in [("llm_input.json", "input_sha256"), ("llm_prompt.txt", "prompt_sha256"), ("context_flows.csv", "context_csv_sha256"), ("trigger_alerts.csv", "trigger_csv_sha256")]:
            check(spec["folder"] + " prepared checksum: " + name, digest(directory / name) == trace[field])
        actual_summary = payload["network_observations"]
        check(spec["folder"] + " input and trace counts reconcile", actual_summary["context_flow_count"] == len(context) == trace["context_flow_count"] and payload["ml_trigger_summary"]["or_alert_count"] == len(triggers) == trace["or_alert_count"])
        check(spec["folder"] + " connection-state counts reconcile", actual_summary["connection_states"] == dict(Counter(r["conn_state"] for r in context)))
        for field in ["orig_bytes", "resp_bytes"]:
            available = [Decimal(r[field]) for r in context if r[field] not in ("", "-", "nan", "missing")]
            check(spec["folder"] + " byte totals and missing values reconcile: " + field,
                  Decimal(str(actual_summary[field]["sum_available_values"])) == sum(available, Decimal(0)) and actual_summary[field]["missing_rows"] == len(context) - len(available))
        encoded_prompt = (directory / "llm_prompt.txt").read_text()
        check(spec["folder"] + " prompt contains the exact prepared input", json.loads(encoded_prompt.split("\nINPUT\n", 1)[1]) == payload)
        check(spec["folder"] + " payload contains no dataset-family annotation", not any("From_malicious-" in str(x) for x in [encoded_prompt]) and '"detailedlabel"' not in encoded_prompt)
        check(spec["folder"] + " reference IDs are unique official definitions", len({r["technique_id"] for r in payload["attack_reference_material"]}) == len(payload["attack_reference_material"]) and all(r["source_url"].startswith("https://attack.mitre.org/techniques/") for r in payload["attack_reference_material"]))
        case_counts.append({"case_id": spec["case_id"], "triggers": len(triggers), "context": len(context)})

    original_redline = read_csv(ROOT / "inputs/redline_endpoint_evidence.csv")
    new_redline = read_csv(out / "cases/redlinestealer_case2/context_flows.csv")
    check("New RedLine context exactly matches every UID in the prior supplied evidence CSV", {r["uid"] for r in original_redline} == {r["uid"] for r in new_redline} and len(new_redline) == 2755)
    before = {r["uid"]: r for r in original_redline}
    check("RedLine endpoint/connection evidence agrees with the prior supplied CSV", all(all(r[f] == before[r["uid"]][f] for f in ("id.orig_h", "id.resp_h", "id.resp_p", "conn_state", "proto")) for r in new_redline))
    for field in ["orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]:
        check("RedLine per-flow numeric evidence agrees: " + field, all(
            (r[field] == "" and before[r["uid"]][field] == "") or
            (r[field] != "" and before[r["uid"]][field] != "" and Decimal(r[field]) == Decimal(before[r["uid"]][field]))
            for r in new_redline))
    check("Preparation itself performs no training, ML inference or LLM generation", run["trained_models"] is False and run["executed_model_inference"] is False and run["executed_llm_generation"] is False)
    check("Manifest population and alert counts reconcile", run["processed_rows"] == len(decisions) and run["positive_or_alerts"] == len(alerts))
    recorded_generations = []
    for manifest_path in sorted((out / "cases").glob("*/runs/*/generation_manifest.json")):
        directory = manifest_path.parent
        generation = read_json(manifest_path)
        label = directory.parent.parent.name + "/" + directory.name
        for name, field in [("llm_input.json", "input_sha256"), ("llm_prompt.txt", "prompt_sha256")]:
            check(label + " generation snapshot checksum: " + name, digest(directory / name) == generation[field])
        snapshot = read_json(directory / "handoff_trace.json")
        for name, field in [("trigger_alerts.csv", "trigger_csv_sha256"), ("context_flows.csv", "context_csv_sha256")]:
            check(label + " generation flow snapshot checksum: " + name, digest(directory / name) == snapshot[field])
        request = read_json(directory / "ollama_request.json")
        check(label + " exact saved prompt was submitted", request["prompt"] == (directory / "llm_prompt.txt").read_text() and request["model"] == generation["model"] and request["options"] == generation["options"])
        check(label + " client source snapshot checksum agrees", generation["client_script_sha256"] == digest(directory / "run_ollama_client.py"))
        if generation["status"] == "generated_review_pending":
            response = read_json(directory / "ollama_api_response.json")
            text = (directory / "response_raw.txt").read_text()
            check(label + " raw and parsed model responses agree", response["response"] == text and json.loads(text) == read_json(directory / "response.json") and digest(directory / "response_raw.txt") == generation["response_sha256"])
            check(label + " generation is complete and not token-truncated", response.get("done") is True and response.get("done_reason") != "length" and bool(text.strip()))
            identity = read_json(directory / "model_identity.json")
            check(label + " captured runtime and model digest agree", generation["model_identity_sha256"] == digest(directory / "model_identity.json") and generation["model_digest"] == identity["model"]["digest"] and generation["ollama_version"] == identity["ollama_version"])
            if "runtime_environment_sha256" in generation:
                runtime_path = out / generation.get("runtime_environment_file", "runtime_environment.json")
                runtime = read_json(runtime_path)
                check(label + " runtime and official registry manifest agree", generation["runtime_environment_sha256"] == digest(runtime_path) and generation["model_digest"] == runtime["model"]["digest"] == runtime["model_download_verification"]["manifest_sha256"])
        recorded_generations.append({"case_id": generation["case_id"], "run": str(directory.relative_to(out)), "status": generation["status"],
                                     "matches_current_prepared_input": generation["input_sha256"] == digest(directory.parent.parent / "llm_input.json")})
    execution_path = out / "llm_execution_manifest.json"
    if execution_path.exists():
        execution = read_json(execution_path)
        for item in execution["cases"]:
            selected = ROOT / item["generation_manifest"]
            generation = read_json(selected)
            case_dir = out / "cases" / item["case_directory"]
            review = read_json(case_dir / "evidence_review.json")
            check(item["case_directory"] + " selected generation matches the current case input", generation["input_sha256"] == digest(case_dir / "llm_input.json") and generation["prompt_sha256"] == digest(case_dir / "llm_prompt.txt"))
            check(item["case_directory"] + " current review is bound to the selected input and raw response", review["input_sha256"] == generation["input_sha256"] and review["model_response_sha256"] == generation["response_sha256"])
        for assessment in execution.get("evidence_assessments", []):
            check(assessment["case_id"] + " assessment checksum agrees with the execution manifest", digest(out / assessment["path"]) == assessment["sha256"])
    report = {"status": "PASS", "source_and_output_checks_passed": len(checks), "processed_rows": len(decisions),
              "positive_or_alerts_from_available_predictions": len(alerts), "cases": case_counts,
              "boundary_test_command": "python -m unittest -v test_handoff.py", "boundary_tests_passed_at_package_creation": 21,
              "llm_client_check": "The client boundary test uses a simulated API. Actual local generation records are listed separately; artifact checks do not validate their interpretations.",
              "recorded_local_generations": recorded_generations, "checks": checks}
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
