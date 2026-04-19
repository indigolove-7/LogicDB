# LogicDB

LogicDB is an open-source research codebase for the system described in our paper: an LLM-native analytical stack that combines a foreground analytics agent with a background physical-optimization controller through one typed action space, `OpenOps`.

This repository is structured as a public research system repo. Reviewers and users should be able to inspect the code paths, understand the benchmark assets, and run the main public entrypoints directly.

## Core components

- Foreground SQL-free analytical serving over structured and table-centric tasks
- A typed OpenOps controller over layout, index, and reusable artifact actions
- Layout planning and application on DuckDB / Parquet-style tabular substrates
- DuckDB-oriented index candidate generation, selection, and build
- Materialized-view style candidate generation, build, and rewrite routing
- Skill mining and retrieval from successful trajectories

## Repository layout

- `src/logicdb/agent`: foreground analytics agent, tools, evaluation helpers, semantic cache
- `src/logicdb/openops`: typed action records and orchestration logic
- `src/logicdb/layout`: layout planning and layout application operators
- `src/logicdb/physical`: index / MV planning, build, rewrite, and cost heuristics
- `src/logicdb/skills`: mined skill bank retrieval and offline skill mining
- `src/logicdb/data`: dataset loaders and tabular carrier helpers
- `src/logicdb/cli`: CLI entrypoints
- `data`: benchmark metadata and repository-shipped sample workloads
- `examples`: runnable examples and sample workload files
- `docs`: architecture notes
- `scripts`: verification helpers

## Installation

```bash
cd /data1/lzs/LogicDB
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev,retrieval,system]"
```

## Environment variables

Dataset roots:

- `LOGICDB_SPIDER_ROOT`
- `LOGICDB_BIRD_ROOT`
- `LOGICDB_WTQ_ROOT`
- `LOGICDB_TABFACT_ROOT`

LLM access:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

Without `OPENAI_API_KEY`, the `demo-openops` pipeline still runs by falling back to deterministic heuristic layout/index planning. The foreground agent path still expects an LLM.

## Benchmark assets

The repository includes benchmark metadata and inspectable workload samples under [data/README.md](/data1/lzs/LogicDB/data/README.md).

- Benchmark metadata: [benchmarks.json](/data1/lzs/LogicDB/data/benchmarks.json)
- TPC-H paper workload sample: [tpch_good17_phased_10000.sample.jsonl](/data1/lzs/LogicDB/data/workloads/tpch/tpch_good17_phased_10000.sample.jsonl)
- TPC-DS paper workload sample: [llmindexadvisor_tpcds_2k_phase_drift_repeat2.duckdb_valid_templated.sample.jsonl](/data1/lzs/LogicDB/data/workloads/tpcds/llmindexadvisor_tpcds_2k_phase_drift_repeat2.duckdb_valid_templated.sample.jsonl)
- JOB paper workload sample: [job_workload.duckdb_valid_phased_1130.sample.jsonl](/data1/lzs/LogicDB/data/workloads/job/job_workload.duckdb_valid_phased_1130.sample.jsonl)

These files are small repository-safe slices taken from the exact workload families used in the paper:

- TPC-H mainline workload: `good17_phased_10000`
- TPC-DS paired workload: `llmindexadvisor_tpcds_2k_phase_drift_repeat2.duckdb_valid_templated`
- JOB paired workload: `job_workload.duckdb_valid_phased_1130`

The full benchmark datasets and official generators should be obtained from their original sources:

- TPC-H: https://www.tpc.org/tpch/
- TPC-DS: https://www.tpc.org/tpcds/
- JOB: https://github.com/gregrahn/join-order-benchmark

## Quickstart

### 1. OpenOps demo on a local DuckDB database

```bash
cd /data1/lzs/LogicDB
python examples/build_demo_duckdb.py
logicdb demo-openops \
  --duckdb-path examples/demo/demo.duckdb \
  --workload-jsonl examples/demo/workload.jsonl \
  --workdir examples/demo/run \
  --out examples/demo/result.json
```

This runs:

- schema introspection
- workload summarization
- layout planning and apply
- index candidate generation and build
- MV candidate generation / build / rewrite demo
- OpenOps action selection and actual commit

The demo assets are generated locally by [build_demo_duckdb.py](/data1/lzs/LogicDB/examples/build_demo_duckdb.py) and are not checked into the repository.

### 2. Python API

```python
from logicdb import LogicDBConfig, run_openops_demo_from_jsonl

cfg = LogicDBConfig(
    duckdb_path="examples/demo/demo.duckdb",
    workdir="examples/demo/run_api",
)
result = run_openops_demo_from_jsonl(cfg, "examples/demo/workload.jsonl")
print(result.to_dict()["openops_plan"])
```

### 3. Verify OpenOps execution path

```bash
PYTHONPATH=/data1/lzs/LogicDB/src python scripts/verify_openops.py
```

This check rebuilds the demo database if needed and verifies that OpenOps does not just score actions: it selects actions, commits them, and applies real layout, index, and MV maintenance work.

### 4. Foreground analytics agent

```bash
logicdb agent \
  --dataset spider \
  --idx 0 \
  --model gpt-5-mini
```

For this path, set `LOGICDB_SPIDER_ROOT` or pass `--spider-root`.

### 5. Skill mining

```bash
logicdb mine-skills \
  --predictions runs/wtq/predictions.jsonl \
  --trajectory-dir runs/wtq/trajectories \
  --dataset wtq \
  --out runs/skills/wtq_skill_bank.json
```

## Notes on reproducibility

- `demo-openops` is runnable without an API key because it has heuristic fallback planners.
- In the public runner, OpenOps is the control path that selects and commits maintenance actions.
- The public demo is intentionally configured so that reviewers can observe real actuation, not only recommendation output.
- The LLM-enabled paths are closer to the paper’s original architecture.
- DuckDB is the main public backend in this release.
- Some modules preserve compatibility with older internal environment variables, but the public names all use the `LOGICDB_*` prefix.

## License

MIT
