# PRT840 OR handoff and local interpretation results

1 October 2026

The replay now includes the full original VM3 Decision Tree test export supplied by Duc. Saved Decision Tree and Isolation Forest decisions are connected to exact case inputs and recorded local Ollama responses. The Remcos trigger count increases from six to 6,765 because previously unavailable Decision Tree predictions have been added. Its wider traffic context is unchanged. The generated ATT&CK proposals still lack sufficient behavioural evidence for confirmed mappings.

## Prediction import and OR handoff

The supplied CSV contains 27,953 unique test records. Every flow ID and source label matches the pinned VM3 metadata, and all 11 previously recorded examples agree in label, prediction and probability. The original file is retained unchanged. A separate import adds an evaluation-protocol column from the declared original 80/20 export context; that column was absent from the supplied file. Import provenance and both file checksums are recorded. The prediction CSV does not establish the exact training code or features.

Predictions are joined by dataset and flow row ID. Any available positive produces an OR alert. Two negatives produce a negative decision. Other combinations with unavailable predictions remain undetermined. Source labels are used for identity checks and evaluation summaries; family annotations do not select triggers or context and do not enter the LLM input.

| Case | Trigger alerts | Context flows | Trigger contributions |
| --- | ---: | ---: | --- |
| Remcos endpoint 66.63.168.35:5888, 22 February 2023 UTC | 6,765 | 33,744 | 6,759 DT-positive/IF-negative; two DT-negative/IF-positive; four IF-positive with DT unavailable |
| RedLine endpoint 4.234.116.12:2567, 21 February 2023 capture | 590 | 2,755 | DT-positive/IF-negative on all 590 |

A declared endpoint and date scope becomes a case only when it contains a positive alert. Exact trigger rows remain separate from the wider matching context. Context totals are not counts of detected attacks. The Remcos scope excludes three web-endpoint connections included in the older 33,747-flow preselected study.

## Detector overlap

| Dataset | Jointly evaluated records | DT-positive/IF-negative | Both positive | DT-negative/IF-positive | Both negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| VM2 | 1,941 | 1,940 | 0 | 1 | 0 |
| VM3 | 15,794 | 15,782 | 6 | 3 | 3 |

Both intersections contain only malicious-labelled records. The saved benign test populations do not overlap. The VM3 OR rule flags 15,791 of its 15,794 shared malicious-labelled test records, including three missed by DT. These descriptive counts show why retaining alerts from either model is useful within the observed intersection. They cannot establish overall ensemble accuracy or a false-positive rate.

Across all available saved predictions, the merged population contains 220,219 processed records and 17,983 positive OR alerts: 2,062 from VM2 and 15,921 from VM3. Predictions outside the saved test populations remain unavailable. The implementation retains unknown outcomes rather than replacing missing predictions with normal decisions. Records with unknown dataset labels were excluded by the original preprocessing and are unavailable here.

## Local generation

The corrected Remcos input was submitted once to Ollama 0.35.0 with Llama 3.1 8B, Q4_K_M, on CPU. It completed with valid required JSON fields and `done_reason: stop`, producing 247 output tokens. The current RedLine input and its completed response remain unchanged.

Both current runs use seed 42, temperature 0, an 8,192-token context, a 2,048-token output limit, eight CPU threads, batch size 64 and the same model digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`. Official distribution and model-layer checksums were verified. Exact inputs, prompts, requests, raw responses and runtime identities are retained.

The earlier six-trigger Remcos response remains attached to its original input under its historical run directory. The current execution manifest selects the response for each current input. One failed client connection before server readiness is retained separately; no model generation occurred in that attempt. No retries were performed to seek a better interpretation.

## Interpretation assessment

| Current case | Generated proposals | Assessment against supplied evidence |
| --- | --- | --- |
| Remcos | T1095 Non-Application Layer Protocol | TCP flow records and retrospective endpoint CTI do not establish adversary C2 use of a non-application-layer channel. No mapping is accepted as supported in the accompanying assessment. |
| RedLine | T1071.001, T1001.003, T1048.001, T1090.004 and T1048 | The five proposals repeat the retrieved reference list. The supplied evidence does not demonstrate web C2, service impersonation, encrypted exfiltration, domain fronting or alternative-protocol exfiltration. No mapping is accepted as supported. |

The corrected Remcos response accurately states the endpoint, TCP protocol and date scope. It omits the 6,765 alerts, model contributions, 33,744 context-flow count and all supplied traffic aggregates. It retains insufficient-evidence caveats for four other retrieved candidates concerning password attacks and an exhaustion flood, but still lists T1095 as supported without the necessary behavioural justification.

The RedLine response preserves the endpoint and retrospective CTI caveats but omits important traffic aggregates and the trigger/context distinction. It incorrectly treats the observed destination port as uncertain; the uncertainty concerns its malware association. Referring to an affected host also overstates the evidence of compromise.

Both responses offer generic evidence-collection or monitoring suggestions. Neither supplies a validated incident-response playbook or reconstructs a confirmed attack chain. Assistant-assisted assessments are bound to the exact input and raw-response hashes; independent team review remains pending. An unsupported proposal means the supplied evidence does not establish that technique.

These current runs do not estimate general model accuracy or demonstrate improved retrieval performance. Supplying official ATT&CK definitions did not prevent unsupported claims. Adding the missing predictions repairs the detector-to-case evidence path, while the interpretation limitations remain documented.

## Verification and remaining reporting work

All 21 boundary tests and 108 source, output and generation-provenance checks passed. The checks reconcile record identities, OR outcomes, case scopes, trigger/context separation, normalization and exact request/response hashes. They do not validate the model's interpretation.

The full VM3 prediction export is now included. The saved baseline notebook still includes source and destination ports among its model features, so the exact detector version needs reconciliation with the Week 9 minutes describing their removal. That question is not answered by a prediction CSV. The report should retain the differing test-population limits and complete independent assessment before describing any mapping or response instruction as validated.

## Source records

Prediction and CTI source commits and checksums: `source_manifest.json`. Original VM3 import provenance: `inputs/vm3/dt_import_manifest.json`. Official ATT&CK corpus provenance: `references/corpus_manifest.json`. Current generation and assessment links: `outputs/llm_execution_manifest.json`. Historical replay metadata: `history/sample_only_replay.json`.

Relevant official definitions include [T1095](https://attack.mitre.org/techniques/T1095/), [T1071.001](https://attack.mitre.org/techniques/T1071/001/), [T1001.003](https://attack.mitre.org/techniques/T1001/003/), [T1048.001](https://attack.mitre.org/techniques/T1048/001/), [T1090.004](https://attack.mitre.org/techniques/T1090/004/) and [T1048](https://attack.mitre.org/techniques/T1048/).
