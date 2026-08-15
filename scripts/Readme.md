# Scripts Guide

This folder contains two Python scripts for working with the `conn.log.labeled` files under `Experiment-VM-Microsoft-Windows7full-3`.

## Prerequisites

- Python 3.10 or newer installed
- `pip` available in your terminal
- The dataset folder `Experiment-VM-Microsoft-Windows7full-3` present in the project root

## Install pandas

Open a terminal in the project root (`D:\CTU\CTU-SME-11`) and run:

```powershell
python -m pip install pandas
```

If your machine uses `py` instead of `python`, run:

```powershell
py -m pip install pandas
```

You can verify the installation with:

```powershell
python -m pip show pandas
```

## Run Script 01: Data Profiling

This script scans all `conn.log.labeled` files, combines them, prints profiling information, and saves a summary CSV file.

Run from the project root:

```powershell
python scripts/01_data_profiling.py
```

Alternative on Windows:

```powershell
py scripts/01_data_profiling.py
```

### Expected output

- Console output showing:
  - number of files discovered
  - total rows and columns
  - column names and data types
  - label and detailed-label distributions
  - missing values
  - duplicate row count
  - timestamp range
- Output file created:
  - `outputs/conn_log_profile_summary.csv`

## Run Script 02: Preprocessing

This script prepares two outputs:

- a cleaned modelling dataset
- a CTI metadata mapping file

Run from the project root:

```powershell
python scripts/02_preprocessing.py
```

Alternative on Windows:

```powershell
py scripts/02_preprocessing.py
```

### Expected output

- Console output showing:
  - number of files discovered
  - combined rows and columns
  - missing-value distribution by label and connection state
  - benign and malicious sample counts per capture date
  - removed `Unknown` rows
  - retained rows for modelling and metadata mapping
- Output files created:
  - `outputs/clean_network_flows.csv`
  - `outputs/cti_mapping_metadata.csv`

## Recommended Run Order

Run the scripts in this order:

1. `python scripts/01_data_profiling.py`
2. `python scripts/02_preprocessing.py`

This helps you inspect the dataset first and then generate the cleaned outputs.

## Common Issue

If you see an error such as `ModuleNotFoundError: No module named 'pandas'`, install pandas again using:

```powershell
python -m pip install pandas
```

If you see a file-not-found error, make sure you are running the command from the project root and that the folder `Experiment-VM-Microsoft-Windows7full-3` exists there.
