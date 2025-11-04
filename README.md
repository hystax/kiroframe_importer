## *The Kiroframe ML profiling data importing tool by Hystax*

This tool exports metadata from an MLflow Tracking Server and imports it into **Kiroframe** for central profiling, analytics, and model governance.

---

## What it does

- Connects to an MLflow Tracking URI and enumerates **experiments**,  **runs** and **models**.
- Exports:
  - `runs.csv` — flat table of all runs for quick ingestion.
  - `runs.json` — per-run record with tags/params/metrics (latest + history).
  - `experiments_runs_datasets.json` — per-run dataset usage/production rows.
  - `experiment_summary.json` — per-experiment stats (owners, time range, metrics/params/tag keys).
  - `artifacts_manifest.json` — where artifacts were downloaded (optional).
  - `summary.json` — overall export stats.
  - `models.json` — registered models with versions (optional).
- Optionally **downloads run artifacts** (experimental).
- Pushes the exported data into **Kiroframe** via API.
- Supports re-import

---
## Prerequisites

- Python **3.10+**
- Network access to your MLflow Tracking Server and Kiroframe endpoint
- A valid **Profiling token**
- Disk space for the export (artifacts optional)
---

## Quick start (env-driven, recommended)


- install python venv package (assuming you're using deb-based distro)
```bash
  sudo apt update && sudo apt install python3-venv
```

 - create venv for importer
```bash
 python3 -m venv .venv
```

- activate venv

```bash
  source .venv/bin/activate
```

- install kiroframe-importer

```bash
  pip install kiroframe_importer
```

 - create env file (*params.env*)
```bash

export TRACKING_URI=http://10.10.0.1:5000 # MLflow URL
export KIROFRAME_URL=https://my.kiroframe.com # Kiroframe URL
export PROFILING_TOKEN=51b10959-3e3b-4ab3-bff8-18dfb57de104 # Insert Valid ML token here
export IMPORT_TYPE=mlflow
export EXPORT_BASE_DIR=/tmp/mlf_exports
export EXPORT_MODELS=true
export INCLUDE_SCHEMA_PROFILE=true
export DOWNLOAD_ARTIFACTS=false
export EXP_PAGE_SIZE=200
export RUN_PAGE_SIZE=1000
```
- source env file

```bash
  source params.env
```
- run the importer
```bash
 kiroframe-importer
```

On first run it creates a timestamped export folder inside `EXPORT_BASE_DIR`, then imports to Kiroframe.

---

## CLI usage

```bash
 kiroframe-importer [OPTIONS]
```

**Options**

| Flag                                                       | Env                      | Default | Description                                                             |
|------------------------------------------------------------|--------------------------|---|-------------------------------------------------------------------------|
| `--import-type`                                            | `IMPORT_TYPE`            | `mlflow` | Import type.                                                            |
| `--tracking-url`                                           | `TRACKING_URI`           | — (required) | MLflow Tracking URI.                                                    |
| `--kiroframe-url`                                          | `KIROFRAME_URL`          | — (required) | Kiroframe base URL.                                             |
| `--profiling-token`                                        | `PROFILING_TOKEN`        | — (required) | ML token used for import.                                       |
| `--base-dir`                                               | `EXPORT_BASE_DIR`        | `~/mlflow_export` | Export base folder; a dated subfolder is created within.                |
| `--include-schema-profile` / `--no-include-schema-profile` | `INCLUDE_SCHEMA_PROFILE` | `true` | Include dataset `schema`/`profile` if present in MLflow dataset inputs. |
| `--export-models` / `--no-export-models`                   | `EXPORT_MODELS`          | `true` | Dump `models.json` with registered models & versions.                   |
| `--download-artifacts` / `--no-download-artifacts`         | `DOWNLOAD_ARTIFACTS`     | `false` | Download run artifacts and record in manifest.                          |
| `--experiment-page-size`                                   | `EXP_PAGE_SIZE`          | `200` | Page size for listing experiments.                                      |
| `--run-page-size`                                          | `RUN_PAGE_SIZE`          | `1000` | Page size for listing runs per experiment.                              |
| `--import-to-kiroframe`                                    | —                        | `true` | If set, import to Kiroframe after export.                               |
| `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`          | —                        | `INFO` | Console & file verbosity.                                               |

**Required core parameters**  
At least these must be present via CLI or env:

- `--tracking-url` **or** `TRACKING_URI`
- `--kiroframe-url` **or** `KIROFRAME_URL`
- `--profiling-token` **or** `PROFILING_TOKEN`

If any is missing the program prints an error and exits with **code 2**.

---

## Running from a single command

**Export + Import (most common):**
```bash
 kiroframe-importer --tracking-url http://10.10.0.1:5000   --kiroframe-url https://my.kiroframe.com   --profiling-token "$PROFILING_TOKEN"   --base-dir /tmp/mlf_exports   --export-models   --no-download-artifacts
```

**Export only (inspect files first):**
```bash
 kiroframe-importer --no-import-to-kiroframe --no-download-artifacts
```

---

## Output files & structure

A new directory is created per run: `YYYYMMDD_HHMM_microseconds`, e.g.:

```
/tmp/mlf_exports/20250915_1012_123456/
├── artifacts/                              # run artifacts if enabled
│   └── <run_id>/...
├── artifacts_manifest.json                 # mapping run_id -> local path
├── experiment_summary.json                 # JSON; one row per experiment
├── experiments_runs_datasets.json          # JSON; dataset usage/production rows
├── export.log                              # detailed log file of the export/import tasks
├── models.json                             # registered models + versions (optional)
├── runs.csv                                # flat table of all runs (from search_runs)
├── runs_basic.csv                          # minimal per-run summary for quick scans
├── runs.json                               # JSON; per-run full record
└── summary.json                            # export counters & MLflow version
```

**Notes**
- Timestamps are recorded both in **ms epoch** (used for import) and **ISO 8601 (UTC)** where applicable.
- `experiments_runs_datasets.json` includes dataset **role** (`used`/`produced`) inferred from `mlflow.data.context`.

---

## What gets imported to Kiroframe

The importer posts three payload types to Kiroframe:

1. **Experiments** (`experiment_summary.json` rows) — creates/updates tasks,metrics.
2. **Runs** (`runs.json` rows) — imports runs, metric logs, datasets.
3. **Models** (`models.json`) — registered models + versions (experimental, when `--export-models`).

## Logging

- Console logs respect `--log-level`.
- A rotating file handler isn’t used; a single `export.log` is attached in the export folder with extended context: timestamp, module, file, line.

---

## Security & networking

- Tokens are read from env or CLI; **avoid shell history leaks** (prefer env vars).
- If you need a proxy, try following envs: `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`.

---

## Performance tips

- try to increase `--run-page-size` for servers with many runs per experiment.
- If artifacts are large, keep `--no-download-artifacts` during initial import.
- Run close to the MLflow server to reduce latency on metric history calls.

---

## Troubleshooting

**“ERROR: missing required parameter(s)” and exit code 2**  
Provide `--tracking-url`, `--kiroframe-url`, and `--profiling-token` (or their env vars).

**MLflow auth errors / 401 / 403**  
Your MLflow server may require a token or Basic Auth; include credentials in the URI or set envs your MLflow store expects.

**Artifacts download fails**  
- Keep running without `--download-artifacts`.  
- Verify `artifact_location` and MLflow backend store permissions.

**models.json empty**  
No registered models found or `--no-export-models` was set.

---

## Exit codes

- `0` — success
- `2` — missing required parameters (core validation)
- `>0` — unhandled error (see `export.log`)

---

## FAQ

**Q: Can I re-run and only import new runs?**  
A: The exporter always creates a fresh timestamped folder. Kiroframe deduplication behavior supports re-imports, it should be idempotent.

**Q: Does it move or delete anything on MLflow?**  
A: No. It only reads via MLflow’s public client APIs and (optionally) downloads data.

---

## Developer notes

- The importer selects the backend class from `kiroframe_importer.modules.modules` by `--import-type` (default: `mlflow`) and instantiates `MLF`.
- Paging is supported for both experiments and runs; `search_runs` token support is auto-detected by inspecting the client signature.
- `get_models()` gathers:
  - model metadata
  - **all** versions (latest-first if MLflow supports `order_by`)
  - helper maps are built internally for potential future enrichment.

---

## Versioning & compatibility

- Tested with MLflow **2.x**, **3.x** APIs (`MlflowClient`).
- Older stores without `search_experiments`/`page_token` are supported via fallbacks.

---
## License & support

This importer is distributed as part of the Kiroframe data tooling. For enterprise support and integrations, contact your Hystax/Kiroframe representative.