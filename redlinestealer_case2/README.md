# RedLineStealer Case Study 2

This case contains the VM2 endpoint evidence, Duc's saved pilot, and one completed fixed-reference versus retrieved-reference comparison using local `llama3.1:8b`.

The two new responses are valid JSON and copy the supplied numerical facts correctly. Both make an unsupported web C2 assertion. The retrieved response also duplicates `T1048.001` and omits `T1090.004` and `T1048`. This experiment does not establish an improvement in semantic accuracy from retrieval. Independent validation of the mappings remains pending.

## Contents

| Path | Content |
| --- | --- |
| `PRT840_RedLineStealer_Case_Study_2_Draft.docx` | Evidence, method, actual results and limitations |
| `source/duc_baseline/` | Seven unchanged source files from the reviewed repository commit |
| `source/source_manifest.json` | Source commit, file hashes and scope |
| `analysis/` | Recomputed flow summaries, figure, verification results and mapping assessment |
| `inputs/common_evidence.json` | The evidence supplied to both conditions |
| `references/` | Pinned official ATT&CK corpus, provenance and MITRE licence |
| `runs/paired_seed42/` | Prepared prompts, API requests, actual responses, model metadata and audits |
| `derive_evidence.py` | Recomputes the supplied CSV aggregates and common evidence |
| `compare.py` | Prepares, verifies, runs and audits a matched pair |
| `SHA256SUMS.txt` | Package file checksums |

## Evidence

The selection contains 2,755 unique connection records from `192.168.1.106` to `4.234.116.12:2567` over TCP, starting between 21 February 2023 06:50:05.443686962 UTC and 22 February 2023 00:01:10.463027 UTC. The CSV includes 2,358 HTTP service labels and 397 unset labels.

Available byte values sum to 885,750 originator bytes and zero responder bytes. Each byte column has 134 missing values, which are not imputed as zero. The selection contains 4,709 responder packets. The byte sum does not establish an absence of replies or responder payload across all records.

The team CTI register records a retrospective file-to-IP relationship and a separate absence of a port-specific match. It does not independently establish the malware family in these flows or a successful attack. Original capture files were not independently re-extracted for this review. The supplied `redline_endpoint_timeline.csv` duplicates the evidence CSV; the hourly series in `analysis/` is a new aggregate.

## Recorded experiment

- Model: `llama3.1:8b`, Q4_K_M.
- Model manifest digest: `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`.
- Ollama: `0.35.0`, Linux CPU execution on 30 September 2026.
- One generation per condition; seed 42; temperature 0.
- Context 8,192 tokens; output limit 2,304 tokens; remaining explicit options are in each request.
- Fixed condition followed by retrieved condition; common evidence, instructions, schema and settings are identical.
- Five references per condition. BM25 retrieval uses the HTTP service facet, excludes PRE-only techniques and does not use malware names or CTI in its query.
- The official ATT&CK source checksum and all normalised records were independently verified. The pinned version is 19.2 with 697 active techniques.

The pilot uses different evidence detail, prompt wording, output controls, settings and Ollama version. Comparing it with a new response cannot isolate a retrieval effect. The new pair is a single example and does not estimate statistical significance or general accuracy. Its second request uses cached prompt processing, so durations are descriptive rather than a speed benchmark.

## Reproducing the comparison

Python 3.10 or later and a running Ollama installation are required. The scripts use the Python standard library. On Windows, the official installer is available at [ollama.com/download/windows](https://ollama.com/download/windows). Start Ollama and install the model:

```text
ollama pull llama3.1:8b
```

Run the following from this directory, choosing a new output directory for each execution:

```text
python compare.py prepare --out runs/new_comparison
python compare.py verify --out runs/new_comparison
python compare.py run --out runs/new_comparison
```

The default API address is `http://127.0.0.1:11434`. `--base-url` can select another locally authorised endpoint. Existing runs are protected against overwriting. Runtime and model metadata are recorded for every execution; matching a model name alone does not establish the same model digest or runtime.

`python derive_evidence.py` regenerates the derived numerical evidence from the preserved source CSV. It does not modify saved prompts or executed responses. No response action is executed by either script.

## Interpreting the audits

`validation.json` checks JSON structure, exact copied facts, requested candidate coverage, response phases, generation completion and recorded context budget. Passing these checks does not establish that an ATT&CK mapping is true or that a response action is useful. The fixed response passes all automated checks despite its unsupported C2 claim. The separate assessment in `analysis/` records the substantive issues for independent review.

## Sources

Source repository: [PRT840 CTI--Remaining-Results at the reviewed commit](https://github.com/ductri19122001/PRT840/tree/f0fe547ba4d67e2ec5de6b900abc39ac9560f5c0/CTU-SME-11_Windows7full-2/outputs/case_studies/redline_cti).

ATT&CK provenance: `references/corpus_manifest.json`. Model API: [Ollama generate documentation](https://docs.ollama.com/api/generate). Connection field semantics: [Zeek Conn Info documentation](https://docs.zeek.org/en/current/scripts/base/protocols/conn/main.zeek.html).
