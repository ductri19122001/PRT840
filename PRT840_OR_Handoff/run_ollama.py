"""Submit one prepared alert-led case to a local Ollama server.

Called separately from merge_or.py. See https://docs.ollama.com/api/generate.
Successful JSON parsing does not validate the model's interpretation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from merge_or import read_json, sha256, write_json


def request_json(base_url, endpoint, payload=None, timeout=30):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(base_url.rstrip("/") + endpoint, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_case(case_dir):
    trace = read_json(case_dir / "handoff_trace.json")
    if trace.get("status") != "alert_led_input_prepared" or trace.get("or_alert_count", 0) <= 0:
        raise ValueError("This directory does not contain a prepared case with positive OR triggers")
    for filename, field in (("llm_input.json", "input_sha256"), ("llm_prompt.txt", "prompt_sha256"), ("context_flows.csv", "context_csv_sha256"), ("trigger_alerts.csv", "trigger_csv_sha256")):
        if sha256(case_dir / filename) != trace[field]:
            raise ValueError(f"Prepared case checksum mismatch: {filename}")
    return trace


def validate_response(text):
    if not text.strip():
        raise ValueError("Ollama returned an empty response")
    value = json.loads(text)
    required = {"observed_facts", "supported_attack_mappings", "uncertainties", "suggested_response_actions", "human_review_required"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("Response is missing required JSON fields")
    if value["human_review_required"] is not True:
        raise ValueError("Response must retain human review")
    for field in ("supported_attack_mappings", "uncertainties", "suggested_response_actions"):
        if not isinstance(value[field], list):
            raise ValueError(f"Response field {field} must be a list")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="Directory containing handoff_trace.json and llm_prompt.txt")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    case_dir = args.case.resolve()
    trace = validate_case(case_dir)
    prompt = (case_dir / "llm_prompt.txt").read_text(encoding="utf-8")
    request = {"model": args.model, "prompt": prompt, "stream": False, "format": "json",
               "options": {"seed": args.seed, "temperature": 0, "num_ctx": 8192, "num_predict": 2048,
                           "num_thread": 8, "num_batch": 64, "num_gpu": 0}}
    run_dir = case_dir / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    for filename in ("llm_input.json", "llm_prompt.txt", "handoff_trace.json", "trigger_alerts.csv", "context_flows.csv"):
        shutil.copyfile(case_dir / filename, run_dir / filename)
    shutil.copyfile(Path(__file__), run_dir / "run_ollama_client.py")
    write_json(run_dir / "ollama_request.json", request)
    manifest = {"case_id": trace["case_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model, "options": request["options"], "input_sha256": trace["input_sha256"],
                "prompt_sha256": trace["prompt_sha256"], "client_script_sha256": sha256(run_dir / "run_ollama_client.py"),
                "status": "requested", "semantic_validation": "not_performed"}
    write_json(run_dir / "generation_manifest.json", manifest)
    try:
        manifest["ollama_version"] = request_json(args.base_url, "/api/version")
        tags = request_json(args.base_url, "/api/tags")
        model = next((m for m in tags.get("models", []) if m.get("name") in (args.model, args.model + ":latest")), None)
        if model is None or not model.get("digest"):
            raise ValueError("The requested local model has no recorded digest")
        write_json(run_dir / "model_identity.json", {"ollama_version": manifest["ollama_version"], "model": model,
                                                    "capture_method": "local API before generation"})
        manifest["model_digest"] = model["digest"]
        manifest["model_identity_sha256"] = sha256(run_dir / "model_identity.json")
        write_json(run_dir / "generation_manifest.json", manifest)
        response = request_json(args.base_url, "/api/generate", request, timeout=args.timeout)
        write_json(run_dir / "ollama_api_response.json", response)
        text = response.get("response", "")
        (run_dir / "response_raw.txt").write_text(text, encoding="utf-8")
        if response.get("done") is not True:
            raise ValueError("Ollama did not report a completed generation")
        if response.get("done_reason") == "length":
            raise ValueError("Ollama reached the output token limit")
        parsed = validate_response(text)
        write_json(run_dir / "response.json", parsed)
        manifest.update(status="generated_review_pending", response_sha256=sha256(run_dir / "response_raw.txt"),
                        done_reason=response.get("done_reason"), prompt_eval_count=response.get("prompt_eval_count"),
                        eval_count=response.get("eval_count"))
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        manifest.update(status="generation_failed", error=str(exc), completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(run_dir / "generation_manifest.json", manifest)
        raise SystemExit(f"Generation failed: {exc}. Details: {run_dir.name}")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "generation_manifest.json", manifest)
    print(f"Generated; evidence review pending. Results: {run_dir}")


if __name__ == "__main__":
    main()
