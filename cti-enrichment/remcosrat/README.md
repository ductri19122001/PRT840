# RemcosRAT CTI Enrichment

## Overview

This package contains the CTI enrichment work for the RemcosRAT case study in the PRT840 research project.

The enrichment links network-flow records with external cyber threat intelligence evidence to support the subsequent LLM-based case-study analysis.

## Required Inputs

* `clean_network_flows.csv` — processed network-flow dataset.
* `cti_mapping_metadata.csv` — CTI mapping metadata.

## Enrichment Method

The enrichment script joins CTI mapping metadata with processed network flows and identifies connections associated with the endpoint `66.63.168.35:5888`.

A MalwareBazaar report identifies this endpoint in a malware sample's C2 configuration.

The enrichment identified 78,942 matching network connections.

## How to Run

1. Install the Python dependencies required by `build_enrichment.py`.
2. Make the two required input CSV files available at the paths expected by the script.
3. Run the enrichment script:

```bash
python build_enrichment.py
```

Check the script for any required command-line arguments or path configuration before running.

## Output Files

| File                                 | Description                                     |
| ------------------------------------ | ----------------------------------------------- |
| `cti_evidence_register.csv`          | External CTI evidence and source information    |
| `remcos_endpoint_enriched_flows.csv` | CTI-enriched network-flow records               |
| `remcos_evidence_only_flows.csv`     | Evidence-only records prepared for LLM analysis |
| `case_daily_summary.csv`             | Daily case summary                              |

## Temporal Evidence Limitation

The MalwareBazaar report identifies the endpoint in a RemcosRAT sample's C2 configuration. However, the date on which this configuration became publicly available has not yet been verified.

The CTI evidence must therefore be treated as retrospective unless its availability during the February 2023 traffic period can be established.

The endpoint match alone does not establish that every associated connection is malicious.

## LLM Case Study

The CTI evidence register was used by Duc for LLM Run 2.

The enriched network-flow outputs were not used in that run.
