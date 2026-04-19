# Data Assets

This directory documents the benchmark assets used by LogicDB and ships small, review-friendly workload samples that can be inspected directly from the repository.

## What is included here

- `benchmarks.json`: benchmark metadata, official links, and LogicDB sample workload pointers
- `workloads/tpch/*.sample.jsonl`: sample slices from the paper's `good17_phased_10000` TPC-H workload
- `workloads/tpcds/*.sample.jsonl`: sample slices from the paper's paired TPC-DS `repeat2` workload
- `workloads/job/*.sample.jsonl`: sample slices from the paper's paired JOB `phased_1130` workload

These samples are intentionally small. They are meant for inspection, parser validation, CI smoke tests, and demonstrating the expected workload schema.

## Paper-aligned workload provenance

The public samples in this directory are aligned to the exact workload families used in our paper experiments:

- TPC-H mainline: `tpch_good17_phased_10000`
- TPC-DS paired row: `llmindexadvisor_tpcds_2k_phase_drift_repeat2.duckdb_valid_templated`
- JOB paired row: `job_workload.duckdb_valid_phased_1130`

The shipped files are reduced slices for inspection and testing. They preserve the same row schema and naming as the corresponding paper workloads.

## Full datasets and benchmark kits

LogicDB does not vendor full benchmark datasets or official benchmark kits into this repository.

- TPC-H official benchmark page: https://www.tpc.org/tpch/
- TPC-DS official benchmark page: https://www.tpc.org/tpcds/
- Join Order Benchmark repository: https://github.com/gregrahn/join-order-benchmark

The TPC benchmark specifications and generators should be obtained from TPC according to their licensing and usage terms.

## Expected workload format

TPC-H / TPC-DS style JSONL:

```json
{"qid": 0, "segment_id": 0, "template": 8, "seed": 1019045725, "sql": "SELECT ..."}
```

JOB style JSONL:

```json
{"id": "17f", "query": "SELECT ...", "sql": "SELECT ..."}
```

## Notes

- The sample workloads in this directory are sufficient for repository-level tests and examples.
- For paper-scale runs, point LogicDB to your full local benchmark data and workload files.
