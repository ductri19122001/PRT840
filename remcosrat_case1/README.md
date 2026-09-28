# Case Study 1: RemcosRAT evidence review

This directory contains a retrospective analysis of RemcosRAT-labelled network records from the CTU-SME-11 Windows VM3 dataset, with contextual cyber threat intelligence (CTI) and saved local language-model outputs. The principal traffic window is 22 February 2023 UTC.

## Contents

| Directory | Contents |
| --- | --- |
| `report/` | Case study evidence review, methods, limitations and references. |
| `evidence/` | Case summary, endpoint aggregates, selected flow examples, 15-minute timeline and forensic graph. |
| `cti/` | Endpoint evidence register, daily match summary and the script used for CTI enrichment. |
| `llm_legacy/` | Prompts, raw responses and run records for the two initial Ollama runs. |
| `experiments/` | Three subsequent development experiment archives, including prompts, responses and validation records. |

`SHA256SUMS.txt` lists checksums for the packaged files.

## Evidence summary

The 22 February slice contains 33,747 records selected using the dataset's retrospective labels. Of these, 33,744 involve `66.63.168.35:5888` and three involve `178.237.33.50:80`. The flow counts and byte counters describe recorded network activity; they do not establish malware execution, successful command and control, or exfiltration.

The CTI evidence links `66.63.168.35:5888` to a RemcosRAT configuration extracted from an archive and its contained executable. Those two records describe one sample chain, rather than independent corroboration of every network flow. The historical match does not establish when the configuration was available to defenders during the 2023 traffic window.

The two initial LLM runs and three later experiment versions are preserved as generated. The later versions produced ten responses in total: six in version 1, two in version 2 and two in version 3. Automated checks of output format or ATT&CK references do not establish that a proposed technique, attack sequence or response action is supported by the traffic. Independent assessment of the generated interpretations remains pending.

## Scope and reproducibility

This package documents a selected case and development experiments. It does not demonstrate a complete live path from a classifier alert through CTI attribution to a validated playbook. The report distinguishes observed traffic, contextual CTI, model output and analyst inference.

The original dataset and preprocessing inputs are outside this compact package. The `cti/build_enrichment.py` script requires the project input files `cti_mapping_metadata(2).csv` and `clean_network_flows(2).csv` in its parent directory to regenerate its joined outputs. The full source references and experiment details are in the report and the saved run records.
