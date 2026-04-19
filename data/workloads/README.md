# Workload Samples

This directory contains small sample workload files used for:

- parser / loader checks
- example commands
- repository smoke tests
- inspection by reviewers

Each sample preserves the same JSONL schema used by the larger benchmark workloads behind our experiments.

Subdirectories:

- `tpch/`
- `tpcds/`
- `job/`

The current samples are aligned to the real paper workload families rather than generic benchmark placeholders:

- `tpch/tpch_good17_phased_10000.sample.jsonl`
- `tpcds/llmindexadvisor_tpcds_2k_phase_drift_repeat2.duckdb_valid_templated.sample.jsonl`
- `job/job_workload.duckdb_valid_phased_1130.sample.jsonl`
