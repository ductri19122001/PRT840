"""Boundary checks for missing predictions, source identity and case generation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from merge_or import (build_case, integer, keyed, merge_dataset, numeric_summary,
                      or_decision, public_flow, read_csv, sha256, validate_predictions, write_csv, write_json)
import run_ollama
from import_vm3_predictions import normalize_rows


def fixtures(dataset="VM2"):
    metadata = {1: {"ts": "1676988000.000001", "uid": dataset + "_1", "id.orig_h": "192.168.1.1",
                   "id.orig_p": "50001", "id.resp_h": "203.0.113.1", "id.resp_p": "2567",
                   "capture_date": "2023-02-21", "label": "Malicious", "detailedlabel": "LABEL_SENTINEL"}}
    feature = {"id.orig_p": "50001.0", "id.resp_p": "2567.0", "duration": "1.0", "orig_bytes": "25",
               "resp_bytes": "", "missed_bytes": "0", "orig_pkts": "1", "orig_ip_bytes": "65", "resp_pkts": "0",
               "resp_ip_bytes": "0", "proto": "tcp", "service": "missing", "conn_state": "S0", "history": "S", "label": "1"}
    prediction = {"flow_row_id": "1", "true_label": "1", "predicted_label": "1", "evaluation_protocol": "stratified_80_20"}
    scope = {"case_id": "fixture", "dataset": dataset, "scope": {"id.orig_h": "192.168.1.1", "id.resp_h": "203.0.113.1",
             "id.resp_p": 2567, "proto": "tcp", "time_basis": "capture_date", "date": "2023-02-21"}}
    return metadata, {1: feature}, {1: prediction}, scope


class SourceAndHandoffChecks(unittest.TestCase):
    def test_import_preserves_prediction_values_and_declares_missing_protocol(self):
        row = {"flow_row_id": "12", "true_label": "1", "predicted_label": "0", "predicted_probability": "0.25"}
        original = copy.deepcopy(row)
        imported = normalize_rows([row], "stratified_80_20")
        self.assertEqual(row, original)
        self.assertEqual({k: imported[0][k] for k in row}, original)
        self.assertEqual(imported[0]["evaluation_protocol"], "stratified_80_20")

    def test_import_rejects_a_conflicting_declared_protocol(self):
        row = {"flow_row_id": "12", "true_label": "1", "predicted_label": "0", "predicted_probability": "0.25", "evaluation_protocol": "chronological"}
        with self.assertRaisesRegex(ValueError, "incompatible"):
            normalize_rows([row], "stratified_80_20")

    def test_import_rejects_an_invalid_probability(self):
        row = {"flow_row_id": "12", "true_label": "1", "predicted_label": "0", "predicted_probability": "NaN"}
        with self.assertRaisesRegex(ValueError, "probability"):
            normalize_rows([row], "stratified_80_20")

    def test_compressed_csv_preserves_missing_versus_zero_and_flow_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.csv.gz"
            rows = [{"flow_row_id": "1", "bytes": ""}, {"flow_row_id": "2", "bytes": "0"}]
            write_csv(path, rows, ["flow_row_id", "bytes"])
            self.assertEqual(read_csv(path), rows)
    def test_missing_negative_does_not_become_a_confirmed_negative(self):
        metadata, features, dt, scope = fixtures()
        dt[1]["predicted_label"] = "0"
        rows = merge_dataset("VM2", metadata, features, dt, {})
        self.assertIsNone(rows[0]["or_alert"])
        self.assertIsNone(build_case(rows, scope))

    def test_one_positive_opens_case_even_with_other_prediction_missing(self):
        metadata, features, dt, scope = fixtures()
        rows = merge_dataset("VM2", metadata, features, dt, {})
        case = build_case(rows, scope)
        self.assertEqual([r["flow_row_id"] for r in case["triggers"]], [1])
        self.assertFalse(case["triggers"][0]["if_available"])

    def test_case_keeps_related_negative_flow_as_context_only(self):
        metadata, features, dt, scope = fixtures()
        metadata[2] = {**metadata[1], "uid": "VM2_2", "id.orig_p": "50002", "label": "Benign"}
        features[2] = {**features[1], "id.orig_p": "50002", "label": "0"}
        dt[2] = {**dt[1], "flow_row_id": "2", "true_label": "0", "predicted_label": "0"}
        forest = {2: {"predicted_label": "0"}}
        case = build_case(merge_dataset("VM2", metadata, features, dt, forest), scope)
        self.assertEqual(len(case["context"]), 2)
        self.assertEqual(len(case["triggers"]), 1)

    def test_changing_available_prediction_changes_case_creation(self):
        metadata, features, dt, scope = fixtures()
        dt[1]["predicted_label"] = "0"
        self.assertIsNone(build_case(merge_dataset("VM2", metadata, features, dt, {}), scope))
        dt[1]["predicted_label"] = "1"
        self.assertIsNotNone(build_case(merge_dataset("VM2", metadata, features, dt, {}), scope))

    def test_same_integer_id_in_other_dataset_cannot_trigger_this_case(self):
        metadata, features, dt, scope = fixtures("VM3")
        rows = merge_dataset("VM3", metadata, features, dt, {})
        scope["dataset"] = "VM2"
        self.assertIsNone(build_case(rows, scope))

    def test_unknown_or_is_not_inferred_from_ground_truth(self):
        metadata, features, _, scope = fixtures()
        rows = merge_dataset("VM2", metadata, features, {}, {})
        self.assertIsNone(rows[0]["or_alert"])
        self.assertIsNone(build_case(rows, scope))

    def test_dataset_annotations_do_not_enter_network_payload(self):
        metadata, features, dt, _ = fixtures()
        exported = public_flow(merge_dataset("VM2", metadata, features, dt, {})[0])
        self.assertNotIn("label", exported)
        self.assertNotIn("detailedlabel", exported)
        self.assertNotIn("LABEL_SENTINEL", json.dumps(exported))

    def test_duplicate_prediction_identity_is_rejected(self):
        metadata, _, dt, _ = fixtures()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_predictions([dt[1], dt[1]], metadata, "stratified_80_20", "fixture")

    def test_protocol_mismatch_is_rejected(self):
        metadata, _, dt, _ = fixtures()
        with self.assertRaisesRegex(ValueError, "wrong protocol"):
            validate_predictions([dt[1]], metadata, "chronological", "fixture")

    def test_ground_truth_mismatch_is_rejected(self):
        metadata, _, dt, _ = fixtures()
        dt[1]["true_label"] = "0"
        with self.assertRaisesRegex(ValueError, "ground-truth mismatch"):
            validate_predictions([dt[1]], metadata, "stratified_80_20", "fixture")

    def test_prediction_for_missing_flow_is_rejected(self):
        metadata, _, dt, _ = fixtures()
        dt[1]["flow_row_id"] = "999"
        with self.assertRaisesRegex(ValueError, "absent"):
            validate_predictions([dt[1]], metadata, "stratified_80_20", "fixture")

    def test_available_sum_preserves_missing_byte_values(self):
        self.assertEqual(numeric_summary([{"bytes": "0"}, {"bytes": ""}, {"bytes": "25"}], "bytes"),
                         {"sum_available_values": 25, "available_rows": 2, "missing_rows": 1})

    def test_utc_window_is_distinct_from_capture_date(self):
        metadata, features, dt, scope = fixtures()
        metadata[1]["ts"] = "1677024070.463027"  # 22 Feb, still in the 21 Feb capture.
        rows = merge_dataset("VM2", metadata, features, dt, {})
        self.assertIsNotNone(build_case(rows, scope))
        scope["scope"]["time_basis"] = "utc_flow_start_date"
        self.assertIsNone(build_case(rows, scope))


class OllamaClientChecks(unittest.TestCase):
    def test_inactive_case_cannot_submit_a_stale_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "handoff_trace.json", {"case_id": "fixture", "status": "no_positive_or_alert", "or_alert_count": 0})
            (root / "llm_prompt.txt").write_text("stale earlier case")
            with self.assertRaisesRegex(ValueError, "positive OR triggers"):
                run_ollama.validate_case(root)
    def test_client_submits_exact_prepared_prompt_and_retains_case_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, content in [("llm_input.json", '{"case_id":"fixture"}'), ("llm_prompt.txt", "fixture prepared prompt"),
                                      ("context_flows.csv", "uid\nfixture\n"), ("trigger_alerts.csv", "uid\nfixture\n")]:
                (root / filename).write_text(content)
            trace = {"case_id": "fixture", "status": "alert_led_input_prepared", "or_alert_count": 1,
                     "input_sha256": sha256(root / "llm_input.json"), "prompt_sha256": sha256(root / "llm_prompt.txt"),
                     "context_csv_sha256": sha256(root / "context_flows.csv"), "trigger_csv_sha256": sha256(root / "trigger_alerts.csv")}
            write_json(root / "handoff_trace.json", trace)
            answer = {"observed_facts": {}, "supported_attack_mappings": [], "uncertainties": ["fixture"],
                      "suggested_response_actions": [], "human_review_required": True}
            with patch.object(run_ollama, "request_json", side_effect=[{"version": "fixture"},
                              {"models": [{"name": "fixture-model", "digest": "fixture-digest"}]},
                              {"response": json.dumps(answer), "done": True, "done_reason": "stop"}]) as call:
                run_ollama.main(["--case", str(root), "--model", "fixture-model"])
            submitted = call.call_args_list[2].args[2]
            self.assertEqual(submitted["prompt"], "fixture prepared prompt")
            self.assertEqual(submitted["model"], "fixture-model")
            run_manifest = json.loads(next((root / "runs").glob("*/generation_manifest.json")).read_text())
            self.assertEqual(run_manifest["input_sha256"], trace["input_sha256"])
            self.assertEqual(run_manifest["status"], "generated_review_pending")
            self.assertEqual(run_manifest["semantic_validation"], "not_performed")
            self.assertEqual(run_manifest["model_digest"], "fixture-digest")
            self.assertEqual(run_manifest["model_identity_sha256"], sha256(next((root / "runs").glob("*/model_identity.json"))))

    def test_empty_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            run_ollama.validate_response("")

    def test_response_without_review_is_rejected(self):
        with self.assertRaises(ValueError):
            run_ollama.validate_response(json.dumps({"observed_facts": {}, "supported_attack_mappings": [],
                                                   "uncertainties": [], "suggested_response_actions": [], "human_review_required": False}))


if __name__ == "__main__":
    unittest.main()
