# Architecture

LogicDB has two coupled control planes.

## Foreground

The foreground plane answers one analytical request at a time.

- Ground schema and table evidence
- Read tabular data
- Execute code-centric analysis steps
- Validate outputs and repair if needed

Main files:

- `src/logicdb/agent/core.py`
- `src/logicdb/agent/tools.py`
- `src/logicdb/skills/library.py`

## Background

The background plane watches workload evidence and considers heavier maintenance actions.

- Layout reorganization
- Index build / selection
- Reusable artifact or MV-like materialization
- Admissibility under budget, dependency, conflict, and risk constraints

Main files:

- `src/logicdb/openops/orchestrator.py`
- `src/logicdb/layout/planner.py`
- `src/logicdb/layout/operators.py`
- `src/logicdb/physical/index_duckdb.py`
- `src/logicdb/physical/mv_duckdb.py`

## Shared bridge

The bridge between these planes is a typed action abstraction.

- Candidate actions are normalized into `OpenOpsAction`
- The controller builds a dependency/conflict DAG
- The controller selects a feasible plan under budget and risk constraints

## Public demo path

`src/logicdb/runner.py` is the main public entrypoint for showing the full stack.

- It introspects a DuckDB database
- Summarizes a workload
- Builds physical-action proposals
- Applies selected maintenance operations
- Produces a compact result object for inspection

