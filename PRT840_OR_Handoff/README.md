# PRT840 — OR alerts and case preparation

This package joins saved Decision Tree (DT) and Isolation Forest (IF) predictions, applies the OR rule, and prepares network evidence for cases with positive alerts. It connects saved detector decisions to explicit case inputs for local LLM interpretation.

The implementation is a retrospective replay created on 1 October 2026 and updated that day with Duc's full original VM3 Decision Tree test export. Earlier RemcosRAT and RedLineStealer case experiments used preselected traffic and retain that historical provenance.

## Local generation results

Current generation records use local Ollama 0.35.0 and Llama 3.1 8B, Q4_K_M. Remcos was regenerated after the full VM3 import changed its trigger evidence; RedLine's input and successful response are unchanged. Earlier Remcos input and response snapshots remain in its `runs/` directory. The exact inputs, requests, raw responses, model identity and generation settings are retained. `outputs/llm_execution_manifest.json` identifies the response associated with each current input.

| Case | Model technique proposals | Accepted as supported in the accompanying assessment | Interpretation result |
| --- | ---: | ---: | --- |
| Remcos endpoint scope | 1 | 0 | Correct endpoint scope, but trigger and traffic aggregates omitted; T1095 lacks evidence for confirmed C2 behaviour |
| RedLine endpoint scope | 5 | 0 | All five retrieved definitions were repeated as supported mappings without behavioural justification |

These are descriptive results from one generation for each current input, with the earlier Remcos response retained as history. They do not establish a general model accuracy or error rate. The evidence assessments retain the unsupported claims and explain the missing evidence; independent review remains pending. The short response-action lists are triage suggestions, not validated incident-response playbooks.

`OR_Handoff_Results.md` explains the findings. `docs/PRT840_System_Architecture.docx` describes the implemented replay and its limits.

## Included results

| Study scope | Positive alert triggers | Wider context flows | Trigger evidence |
| --- | ---: | ---: | --- |
| VM3: 66.63.168.35:5888, flow starts on 22 February 2023 UTC | 6,765 | 33,744 | 6,759 DT-positive/IF-negative; two DT-negative/IF-positive; four IF-positive with DT unavailable |
| VM2: 4.234.116.12:2567, 21 February 2023 capture | 590 | 2,755 | DT positive and IF negative on all 590 |

Context includes all available processed flows matching the declared endpoint and date scope, including flows without a positive model decision. The remaining context flows are not counted as detected attacks. The VM2 capture contains flows starting shortly after midnight on 22 February, so its scope uses the capture date rather than the UTC flow-start date.

The new VM3 context concerns the alerted endpoint only. The three connections to 178.237.33.50:80 in the earlier 33,747-flow Remcos study are outside this new scope. Earlier LLM responses must not be attached to the new input as though they were generated from it.

## Merge rule and coverage

Predictions are joined by `(dataset, flow_row_id)`. VM2 and VM3 are kept separate; evaluation protocols are checked against the configuration. The source label is used only to check record identity and summarise the evaluation population. Malware-family annotations do not select alerts or case context, and dataset labels are withheld from LLM inputs.

| DT | IF | OR decision |
| --- | --- | --- |
| Positive | Any outcome, including unavailable | Positive alert |
| Any outcome, including unavailable | Positive | Positive alert |
| Negative | Negative | Negative |
| Negative | Unavailable | Undetermined |
| Unavailable | Negative or unavailable | Undetermined |

The package contains the full saved VM2 prediction exports, the VM3 IF export and all 27,953 rows of the original VM3 Decision Tree 80/20 test export supplied by Duc. Every supplied flow ID and source label matches the pinned VM3 metadata. All 11 previously recorded examples agree in label, prediction and probability.

The supplied CSV is retained unchanged as `inputs/vm3/decision_tree_vm3_original_80_20_predictions.csv`. It contains flow IDs, labels, predictions and probabilities, with no evaluation-protocol column. The normalized import adds `stratified_80_20` from the declared original-export context and agreement with the recorded examples. The source values are unchanged; `dt_import_manifest.json` records this transformation. The CSV does not independently establish the exact training code or features. Predictions for flows outside the saved evaluation populations remain unavailable.

The saved baseline notebook also includes source and destination port numbers as model features. That source does not substantiate the earlier statement that ports were removed from all training runs. `outputs/detector_source_review.json` records the exact source and feature list; the final report needs to identify the model version used for its results.

VM2 has 1,941 jointly evaluated flows and VM3 has 15,794. Both intersections contain only malicious-labelled records; the saved benign test populations do not overlap. VM3 has 15,782 DT-only positives, six positives from both models, three IF-only positives and three negatives from both. These intersections cannot establish overall ensemble accuracy or a false-positive rate. `model_overlap_summary.json` retains the counts and leaves those overall estimates unset.

All source data here come from the existing processed binary-label population; unknown-labelled records were omitted by the original preprocessing and are not available as case context in this package. Numeric missing values remain missing and are counted separately from observed zero values.

## Run the preparation

Extract the package and run these commands from the `PRT840_OR_Handoff` directory. Python 3.10 or later is required; no third-party packages are needed.

```sh
python merge_or.py
```

This verifies source checksums, creates the merged decisions and positive alert list, indexes every alerted endpoint/day group, and prepares the two declared study cases. A case input is created only when its scope contains a positive alert. The endpoint/date scopes are declared for these retrospective studies; they are not claims of independent discovery or retrospective CTI being available during the original capture.

The full VM3 export is already imported and used by default. To reproduce its normalization:

```sh
python import_vm3_predictions.py
```

Import checks retain the original file separately, reject conflicting declared protocols, validate probabilities and record identities, and compare the 11 saved examples. The implementation does not reconstruct unavailable predictions or train a replacement model.

## Outputs

| File | Purpose |
| --- | --- |
| `outputs/or_decisions.csv.gz` | Every processed flow, model availability, individual predictions and three-state OR outcome |
| `outputs/or_alerts.csv` | Positive OR alerts from the available saved predictions |
| `outputs/alert_case_index.csv` | Automatically grouped alerted endpoints and UTC dates across both datasets |
| `outputs/model_overlap_summary.json` | Coverage and model-overlap counts, with evaluation limitations |
| `outputs/cases/*/trigger_alerts.csv` | Exact model alerts triggering each study case |
| `outputs/cases/*/context_flows.csv` | Wider endpoint/time context, with flow IDs and UIDs and no dataset annotations |
| `outputs/cases/*/llm_input.json` | Network summary, alert count, retrospective CTI and ATT&CK reference material |
| `outputs/cases/*/llm_prompt.txt` | Prepared prompt for Ollama |
| `outputs/cases/*/handoff_trace.json` | Trigger identities, scope, counts and input/prompt checksums |
| `outputs/run_manifest.json` | Preparation provenance and explicit execution status |
| `outputs/verification.json` | Checks of the packaged outputs against the source snapshots |
| `outputs/llm_execution_manifest.json` | Actual local generation outcomes and evidence-assessment links |
| `outputs/runtime_environment.json` | Ollama version, exact model digest, CPU environment and official model checksum records |
| `outputs/runtime_environment_full_vm3_update.json` | Runtime and official download verification for the corrected Remcos generation |
| `outputs/cases/*/runs/*/` | Exact input/flow snapshots, request, raw response and generation provenance |
| `outputs/cases/*/evidence_review.json` | Assessment bound to the exact input and raw-response hashes |
| `inputs/vm3/dt_import_manifest.json` | Original-export checksum and explicit normalization provenance |
| `history/sample_only_replay.json` | Earlier preparation, coverage and source manifests combined without changing their values |
| `history/startup_attempt/` | Failed local connection attempt before server readiness; no model generation occurred |

The historical metadata describes the earlier replay with 11 available VM3 DT examples. Its corresponding six-trigger Remcos input, response and assessment remain under `outputs/cases/remcosrat_case1/runs/20261001T054106_637034Z/`. That input differs from the current case. The current execution manifest identifies the response associated with each current input; RedLine is unchanged.

The bundled MITRE ATT&CK Enterprise 19.2 reference is pinned by its existing corpus manifest. BM25 lookup uses the observed network-service facet. Retrieved definitions are not observed behaviour or validated technique mappings. Lexical retrieval can return weakly relevant techniques; a model may return an empty supported-mapping list when the evidence is insufficient.

Large source and decision CSVs are gzip-compressed. The scripts read these files directly; they do not need to be decompressed before running.

## Submit a prepared case to Ollama

Preparation itself does not call an LLM. Each current input has a separately recorded local generation; the earlier Remcos generation remains historical. To create an additional recorded response, use a running local Ollama server with `llama3.1:8b` installed:

```sh
python run_ollama.py --case outputs/cases/remcosrat_case1
python run_ollama.py --case outputs/cases/redlinestealer_case2
```

The client checks the prepared evidence hashes and saves the exact input, prompt, trigger/context CSVs, request, raw API response, model identity and generation metadata in a new run directory. Settings are seed 42, temperature 0, context 8,192 tokens, output limit 2,048 tokens, eight CPU threads and batch size 64. Existing runs are retained. Empty, incomplete or structurally invalid responses are recorded as failures. Successful parsing is recorded as `generated_review_pending`; it does not establish that a technique or response instruction is supported by the evidence. The accompanying assessments are separate from this structural check.

The Ollama request follows the [official generation API](https://docs.ollama.com/api/generate). Source commits and file checksums are recorded in `source_manifest.json`; the MITRE source URL and licence are included under `references/`.

## Checks

```sh
python -m unittest -v test_handoff.py
python verify_outputs.py
```

Checks cover missing predictions, incompatible protocols, duplicate or mismatched flow identities, separation of alerts and context, date boundaries, missing byte values, annotation exclusion and submission of the exact prepared prompt. The client test uses a simulated API response; it is not an Ollama model run.
