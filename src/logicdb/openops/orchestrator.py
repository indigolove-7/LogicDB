#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenOps orchestration core for four-dimensional optimization.

This module provides a claim-safe orchestration implementation:
- DAG-based planning over typed actions (Schema/Index/Pre-cache/Manipulation)
- asynchronous planning compatibility (outside this module)
- guarded serial commit (execution width defaults to 1)

The design intentionally separates:
1) proposal graph construction (dependencies/conflicts/guards)
2) plan selection under budget + risk constraints
3) guarded commit with rollback hooks
"""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class ActionFamily(str, Enum):
    SCHEMA = "schema"
    INDEX = "index"
    PRECACHE = "precache"
    MANIPULATION = "manipulation"


class EdgeType(str, Enum):
    DEPENDS_ON = "depends_on"
    CONFLICTS_WITH = "conflicts_with"
    GUARD = "guard"


@dataclass
class OpenOpsAction:
    """
    Unified action record in OpenOps.

    `schema` uses normalized notation to keep table rows compact in paper tables,
    e.g. "execute_code(code, inputs)".
    """

    action_id: str
    family: ActionFamily
    schema: str
    target: str
    params: Dict[str, Any] = field(default_factory=dict)

    expected_gain_ms: float = 0.0
    expected_apply_ms: float = 0.0
    risk_score: float = 0.0  # [0, 1]
    schema_epoch: Optional[int] = None
    expected_cpu_ms: float = 0.0
    expected_io_mb: float = 0.0
    expected_mem_mb: float = 0.0
    is_heavy: bool = False

    depends_on: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    resource_tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def utility(self, risk_weight: float = 0.35) -> float:
        penalty = float(self.risk_score) * float(risk_weight) * max(0.0, float(self.expected_gain_ms))
        return float(self.expected_gain_ms) - float(self.expected_apply_ms) - penalty


@dataclass(frozen=True)
class DAGEdge:
    src: str
    dst: str
    edge_type: EdgeType
    reason: str = ""


@dataclass
class ActionDAG:
    nodes: Dict[str, OpenOpsAction] = field(default_factory=dict)
    edges: List[DAGEdge] = field(default_factory=list)

    def add_node(self, action: OpenOpsAction) -> None:
        self.nodes[action.action_id] = action

    def add_edge(self, edge: DAGEdge) -> None:
        self.edges.append(edge)

    def dependency_edges(self) -> List[DAGEdge]:
        return [e for e in self.edges if e.edge_type == EdgeType.DEPENDS_ON]

    def conflict_pairs(self) -> Set[Tuple[str, str]]:
        out: Set[Tuple[str, str]] = set()
        for e in self.edges:
            if e.edge_type != EdgeType.CONFLICTS_WITH:
                continue
            a, b = sorted((e.src, e.dst))
            if a != b:
                out.add((a, b))
        return out

    def indegree(self) -> Dict[str, int]:
        deg = {nid: 0 for nid in self.nodes}
        for e in self.dependency_edges():
            if e.dst in deg:
                deg[e.dst] += 1
        return deg

    def outgoing_dep(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for e in self.dependency_edges():
            if e.src in out and e.dst in self.nodes:
                out[e.src].append(e.dst)
        return out


@dataclass
class OrchestrationState:
    window_id: int
    schema_epoch: int
    budget_total_ms: float = 0.0
    budget_by_family_ms: Dict[ActionFamily, float] = field(default_factory=dict)
    risk_tolerance: float = 0.65
    evidence: Dict[str, Any] = field(default_factory=dict)
    physical: Optional["PhysicalState"] = None
    io_budget_mb: float = 0.0
    cpu_budget_ms: float = 0.0
    mem_headroom_mb: float = 0.0


@dataclass
class PhysicalState:
    """
    Runtime physical signals used by control-path admission.

    All fields are optional-friendly through conservative defaults.
    """

    cpu_util_pct: float = 0.0
    cpu_iowait_pct: float = 0.0
    mem_used_pct: float = 0.0
    disk_util_pct: float = 0.0
    running_queries: int = 0
    query_p95_ms: float = 0.0
    active_threads: int = 0
    max_threads: int = 0
    thermal_throttled: bool = False
    backend: str = "unknown"

    def pressure_score(self) -> float:
        # Weighted pressure proxy in [0, 1+] for robust gating.
        cpu = max(0.0, min(1.0, self.cpu_util_pct / 100.0))
        mem = max(0.0, min(1.0, self.mem_used_pct / 100.0))
        io = max(0.0, min(1.0, self.disk_util_pct / 100.0))
        wait = max(0.0, min(1.0, self.cpu_iowait_pct / 100.0))
        q = max(0.0, min(1.0, self.running_queries / 16.0))
        return (0.30 * cpu) + (0.25 * mem) + (0.20 * io) + (0.15 * wait) + (0.10 * q)


def collect_local_physical_state(*, backend: str = "duckdb") -> PhysicalState:
    """
    Best-effort local host snapshot.
    Safe fallback when psutil is unavailable.
    """
    if psutil is None:
        # /proc-based fallback to keep physical guards useful without optional deps.
        cpu_util = 0.0
        mem_used = 0.0
        disk_used = 0.0
        try:
            cpus = max(1, int(os.cpu_count() or 1))
            load1 = float(os.getloadavg()[0])
            cpu_util = max(0.0, min(100.0, (load1 / float(cpus)) * 100.0))
        except Exception:
            cpu_util = 0.0
        try:
            mem_total_kb = 0.0
            mem_avail_kb = 0.0
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for ln in f:
                    if ln.startswith("MemTotal:"):
                        mem_total_kb = float((ln.split() or [0, 0])[1])
                    elif ln.startswith("MemAvailable:"):
                        mem_avail_kb = float((ln.split() or [0, 0])[1])
            if mem_total_kb > 0.0:
                mem_used = max(0.0, min(100.0, (1.0 - (mem_avail_kb / mem_total_kb)) * 100.0))
        except Exception:
            mem_used = 0.0
        try:
            du = shutil.disk_usage("/")
            if du.total > 0:
                disk_used = max(0.0, min(100.0, (float(du.used) / float(du.total)) * 100.0))
        except Exception:
            disk_used = 0.0
        return PhysicalState(
            cpu_util_pct=float(cpu_util),
            cpu_iowait_pct=0.0,
            mem_used_pct=float(mem_used),
            disk_util_pct=float(disk_used),
            running_queries=0,
            query_p95_ms=0.0,
            active_threads=0,
            max_threads=max(1, int(os.cpu_count() or 1)),
            thermal_throttled=False,
            backend=backend,
        )
    try:
        vm = psutil.virtual_memory()
        cpu = float(psutil.cpu_percent(interval=0.1))
        disk = float(psutil.disk_usage("/").percent)
        # iowait is platform-dependent.
        iowait = 0.0
        try:
            ct = psutil.cpu_times_percent(interval=0.0)
            iowait = float(getattr(ct, "iowait", 0.0) or 0.0)
        except Exception:
            iowait = 0.0
        active_threads = 0
        max_threads = int(psutil.cpu_count(logical=True) or 0)
        return PhysicalState(
            cpu_util_pct=cpu,
            cpu_iowait_pct=iowait,
            mem_used_pct=float(vm.percent),
            disk_util_pct=disk,
            running_queries=0,
            query_p95_ms=0.0,
            active_threads=active_threads,
            max_threads=max_threads,
            thermal_throttled=False,
            backend=backend,
        )
    except Exception:
        return PhysicalState(backend=backend)


@dataclass
class ActionOutcome:
    ok: bool
    latency_delta_ms: float = 0.0
    error: Optional[str] = None
    rollback_suggested: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationPlan:
    selected_ids: List[str]
    rejected_ids: Dict[str, str] = field(default_factory=dict)
    score_sum: float = 0.0


@dataclass
class MaintenanceConfig:
    """
    Slow-timescale maintenance policy for physical optimization families.
    """

    interval_windows: int = 8
    min_net_gain_ms: float = 80.0
    min_hot_query_ratio: float = 0.12
    min_drift_score: float = 0.35
    max_pressure_for_maintenance: float = 0.72
    max_apply_share: float = 0.85  # fraction of maintenance budget_total_ms


@dataclass
class MaintenanceDecision:
    run: bool
    reason: str
    estimated_gain_ms: float = 0.0
    estimated_apply_ms: float = 0.0
    estimated_net_ms: float = 0.0
    interval_due: bool = False
    cost_trigger: bool = False
    pressure_score: float = 0.0


@dataclass
class CommitRecord:
    action_id: str
    family: ActionFamily
    ok: bool
    rollback: bool
    reason: str = ""
    latency_delta_ms: float = 0.0


@dataclass
class CommitResult:
    applied_ids: List[str] = field(default_factory=list)
    rolled_back_ids: List[str] = field(default_factory=list)
    records: List[CommitRecord] = field(default_factory=list)
    execution_mode: str = "serial"
    commit_width: int = 1
    levels: List[List[str]] = field(default_factory=list)
    commit_wall_ms: float = 0.0


@dataclass
class DualTimescalePlan:
    """
    Planning outcome for decoupled maintenance vs online intent execution.
    """

    online_dag: ActionDAG
    online_plan: OrchestrationPlan
    maintenance_decision: MaintenanceDecision
    maintenance_dag: Optional[ActionDAG] = None
    maintenance_plan: Optional[OrchestrationPlan] = None


ExecuteFn = Callable[[OpenOpsAction, OrchestrationState], ActionOutcome]
RollbackFn = Callable[[OpenOpsAction, OrchestrationState], None]


class OpenOpsOrchestrator:
    """
    Four-dimensional orchestrator with DAG planning and guarded commit.

    Important: by default this is a DAG planner with serial commit (width=1),
    which matches current production-safe behavior.
    """

    def __init__(
        self,
        *,
        risk_weight: float = 0.35,
        min_expected_gain_ms: float = 0.0,
        commit_width: int = 1,
        max_physical_pressure: float = 0.88,
    ) -> None:
        self.risk_weight = float(risk_weight)
        self.min_expected_gain_ms = float(min_expected_gain_ms)
        self.commit_width = max(1, int(commit_width))
        self.max_physical_pressure = float(max_physical_pressure)

    def build_dag(self, candidates: Iterable[OpenOpsAction]) -> ActionDAG:
        dag = ActionDAG()
        for a in candidates:
            dag.add_node(a)

        node_ids = set(dag.nodes.keys())
        for a in dag.nodes.values():
            for dep in a.depends_on:
                if dep in node_ids and dep != a.action_id:
                    dag.add_edge(
                        DAGEdge(
                            src=dep,
                            dst=a.action_id,
                            edge_type=EdgeType.DEPENDS_ON,
                            reason="explicit_dependency",
                        )
                    )
            for c in a.conflicts_with:
                if c in node_ids and c != a.action_id:
                    dag.add_edge(
                        DAGEdge(
                            src=a.action_id,
                            dst=c,
                            edge_type=EdgeType.CONFLICTS_WITH,
                            reason="explicit_conflict",
                        )
                    )

        # Implicit conflicts on exclusive resource tags.
        by_tag: Dict[str, List[str]] = {}
        for a in dag.nodes.values():
            for tag in a.resource_tags:
                by_tag.setdefault(str(tag), []).append(a.action_id)
        for tag, ids in by_tag.items():
            if len(ids) < 2:
                continue
            ids_sorted = sorted(set(ids))
            for i in range(len(ids_sorted)):
                for j in range(i + 1, len(ids_sorted)):
                    dag.add_edge(
                        DAGEdge(
                            src=ids_sorted[i],
                            dst=ids_sorted[j],
                            edge_type=EdgeType.CONFLICTS_WITH,
                            reason=f"resource_conflict:{tag}",
                        )
                    )
        return dag

    def _admissible(self, action: OpenOpsAction, st: OrchestrationState) -> Tuple[bool, str]:
        if action.schema_epoch is not None and int(action.schema_epoch) != int(st.schema_epoch):
            return False, "schema_epoch_mismatch"
        if action.expected_gain_ms < self.min_expected_gain_ms:
            return False, "insufficient_expected_gain"
        if action.risk_score > max(0.0, min(1.0, st.risk_tolerance)):
            return False, "risk_above_tolerance"
        # Physical-layer gate: reject heavy mutations under host pressure.
        if st.physical is not None:
            p = float(st.physical.pressure_score())
            if st.physical.thermal_throttled and action.is_heavy:
                return False, "thermal_throttled"
            if action.is_heavy and p > float(self.max_physical_pressure):
                return False, "physical_overload"
            if st.io_budget_mb > 0.0 and float(action.expected_io_mb) > float(st.io_budget_mb):
                return False, "exceed_io_budget"
            if st.cpu_budget_ms > 0.0 and float(action.expected_cpu_ms) > float(st.cpu_budget_ms):
                return False, "exceed_cpu_budget"
            if st.mem_headroom_mb > 0.0 and float(action.expected_mem_mb) > float(st.mem_headroom_mb):
                return False, "exceed_mem_headroom"
        return True, ""

    @staticmethod
    def _kahn_topological_order(dag: ActionDAG) -> List[str]:
        indeg = dag.indegree()
        out = dag.outgoing_dep()
        ready = sorted([nid for nid, d in indeg.items() if d == 0])
        order: List[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for nxt in sorted(out.get(nid, [])):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
                    ready.sort()
        # Cycle-safe fallback: append missing nodes by lexical order.
        if len(order) < len(dag.nodes):
            rem = sorted([nid for nid in dag.nodes if nid not in set(order)])
            order.extend(rem)
        return order

    def select_plan(self, dag: ActionDAG, st: OrchestrationState) -> OrchestrationPlan:
        rejected: Dict[str, str] = {}
        candidates: Dict[str, OpenOpsAction] = {}
        for nid, a in dag.nodes.items():
            ok, why = self._admissible(a, st)
            if not ok:
                rejected[nid] = why
                continue
            candidates[nid] = a

        if not candidates:
            return OrchestrationPlan(selected_ids=[], rejected_ids=rejected, score_sum=0.0)

        # Work on subgraph of admissible nodes.
        sub = ActionDAG(nodes={k: v for k, v in candidates.items()}, edges=[])
        for e in dag.edges:
            if e.src in sub.nodes and e.dst in sub.nodes:
                sub.add_edge(e)

        topo = self._kahn_topological_order(sub)
        conflicts = sub.conflict_pairs()
        selected: List[str] = []
        selected_set: Set[str] = set()
        blocked_by_conflict: Set[str] = set()
        score_sum = 0.0

        remain_total = float(st.budget_total_ms) if st.budget_total_ms > 0 else float("inf")
        remain_family: Dict[ActionFamily, float] = {}
        for fam in ActionFamily:
            b = st.budget_by_family_ms.get(fam, 0.0)
            remain_family[fam] = float(b) if b > 0 else float("inf")

        for nid in topo:
            if nid in blocked_by_conflict:
                rejected[nid] = "conflict_blocked"
                continue
            a = sub.nodes[nid]

            # Dependency closure: all required predecessors must be selected.
            missing = [d for d in a.depends_on if d in sub.nodes and d not in selected_set]
            if missing:
                rejected[nid] = "missing_dependency"
                continue

            util = a.utility(self.risk_weight)
            if util <= 0:
                rejected[nid] = "non_positive_utility"
                continue
            if float(a.expected_apply_ms) > remain_total:
                rejected[nid] = "exceed_total_budget"
                continue
            if float(a.expected_apply_ms) > remain_family[a.family]:
                rejected[nid] = "exceed_family_budget"
                continue

            selected.append(nid)
            selected_set.add(nid)
            score_sum += util
            remain_total -= float(a.expected_apply_ms)
            remain_family[a.family] -= float(a.expected_apply_ms)

            for x, y in conflicts:
                if x == nid and y not in selected_set:
                    blocked_by_conflict.add(y)
                if y == nid and x not in selected_set:
                    blocked_by_conflict.add(x)

        for nid in blocked_by_conflict:
            if nid not in rejected and nid not in selected_set:
                rejected[nid] = "conflict_blocked"

        return OrchestrationPlan(selected_ids=selected, rejected_ids=rejected, score_sum=score_sum)

    def suggest_runtime_control_actions(self, st: OrchestrationState) -> List[OpenOpsAction]:
        """
        Generate low-risk physical control actions within the existing 4 families.

        We keep these in `Manipulation` because they tune execution control-path
        rather than introducing a new optimization dimension.
        """
        ps = st.physical
        if ps is None:
            return []
        out: List[OpenOpsAction] = []
        pressure = ps.pressure_score()
        # High pressure: reduce parallelism and hold heavy maintenance.
        if pressure >= 0.82:
            out.append(
                OpenOpsAction(
                    action_id=f"M_ctl_threads_down_w{st.window_id}",
                    family=ActionFamily.MANIPULATION,
                    schema="set_runtime_knob(name, value, scope)",
                    target="engine_threads",
                    params={"name": "threads", "value": "downshift"},
                    expected_gain_ms=35.0,
                    expected_apply_ms=3.0,
                    expected_cpu_ms=2.0,
                    expected_io_mb=0.0,
                    expected_mem_mb=0.0,
                    risk_score=0.08,
                    is_heavy=False,
                )
            )
            out.append(
                OpenOpsAction(
                    action_id=f"M_ctl_hold_heavy_w{st.window_id}",
                    family=ActionFamily.MANIPULATION,
                    schema="hold(reason, duration)",
                    target="heavy_maintenance_window",
                    params={"reason": "physical_pressure", "duration": "short"},
                    expected_gain_ms=20.0,
                    expected_apply_ms=1.0,
                    risk_score=0.05,
                    is_heavy=False,
                )
            )
        # Low pressure + query backlog: can scale up compute.
        if pressure <= 0.45 and ps.running_queries >= 4:
            out.append(
                OpenOpsAction(
                    action_id=f"M_ctl_threads_up_w{st.window_id}",
                    family=ActionFamily.MANIPULATION,
                    schema="set_runtime_knob(name, value, scope)",
                    target="engine_threads",
                    params={"name": "threads", "value": "upshift"},
                    expected_gain_ms=30.0,
                    expected_apply_ms=4.0,
                    risk_score=0.12,
                    is_heavy=False,
                )
            )
        return out

    @staticmethod
    def split_timescale_candidates(
        candidates: Iterable[OpenOpsAction],
    ) -> Tuple[List[OpenOpsAction], List[OpenOpsAction]]:
        """
        Split actions into slow-timescale maintenance vs online execution.

        Default policy:
        - Schema/Index/Pre-cache -> maintenance loop
        - Manipulation -> online loop
        Optional override per action:
        - metadata["timescale"] in {"maintenance", "online"}
        """
        maintenance: List[OpenOpsAction] = []
        online: List[OpenOpsAction] = []
        for a in candidates:
            ts = str(a.metadata.get("timescale", "")).strip().lower()
            if ts == "maintenance":
                maintenance.append(a)
                continue
            if ts == "online":
                online.append(a)
                continue
            if a.family in (ActionFamily.SCHEMA, ActionFamily.INDEX, ActionFamily.PRECACHE):
                maintenance.append(a)
            else:
                online.append(a)
        return maintenance, online

    def decide_maintenance_cycle(
        self,
        *,
        maintenance_candidates: Iterable[OpenOpsAction],
        st: OrchestrationState,
        cfg: Optional[MaintenanceConfig] = None,
    ) -> MaintenanceDecision:
        """
        Decide whether to run physical-maintenance planning this window.

        Trigger logic:
        1) interval due, or
        2) cost trigger from history evidence (hot query ratio / drift) and positive net utility
        while respecting pressure/budget guards.
        """
        policy = cfg or MaintenanceConfig()
        cands = list(maintenance_candidates)
        if not cands:
            return MaintenanceDecision(run=False, reason="no_maintenance_candidates")

        pressure = float(st.physical.pressure_score()) if st.physical is not None else 0.0
        if pressure > float(policy.max_pressure_for_maintenance):
            return MaintenanceDecision(
                run=False,
                reason="defer_high_pressure",
                pressure_score=pressure,
            )

        est_gain = 0.0
        est_apply = 0.0
        for a in cands:
            ok, _ = self._admissible(a, st)
            if not ok:
                continue
            util = float(a.utility(self.risk_weight))
            if util <= 0.0:
                continue
            est_gain += float(a.expected_gain_ms)
            est_apply += float(a.expected_apply_ms)
        est_net = est_gain - est_apply

        interval = max(1, int(policy.interval_windows))
        interval_due = (int(st.window_id) % interval == 0)
        hot_ratio = float((st.evidence or {}).get("hot_query_ratio", 0.0) or 0.0)
        drift_score = float((st.evidence or {}).get("template_drift_score", 0.0) or 0.0)
        cost_trigger = (
            est_net >= float(policy.min_net_gain_ms)
            and (hot_ratio >= float(policy.min_hot_query_ratio) or drift_score >= float(policy.min_drift_score))
        )

        if st.budget_total_ms > 0.0 and est_apply > float(policy.max_apply_share) * float(st.budget_total_ms):
            return MaintenanceDecision(
                run=False,
                reason="defer_excessive_apply_cost",
                estimated_gain_ms=est_gain,
                estimated_apply_ms=est_apply,
                estimated_net_ms=est_net,
                interval_due=interval_due,
                cost_trigger=cost_trigger,
                pressure_score=pressure,
            )

        run = (interval_due or cost_trigger) and (est_net > 0.0)
        reason = "interval_due" if run and interval_due else ("cost_trigger" if run else "not_due")
        return MaintenanceDecision(
            run=bool(run),
            reason=reason,
            estimated_gain_ms=est_gain,
            estimated_apply_ms=est_apply,
            estimated_net_ms=est_net,
            interval_due=bool(interval_due),
            cost_trigger=bool(cost_trigger),
            pressure_score=pressure,
        )

    def plan_dual_timescale(
        self,
        *,
        candidates: Iterable[OpenOpsAction],
        st: OrchestrationState,
        cfg: Optional[MaintenanceConfig] = None,
    ) -> DualTimescalePlan:
        """
        Build separate plans for:
        - slow maintenance loop (Schema/Index/Pre-cache)
        - per-intent online loop (Manipulation)
        """
        maintenance, online = self.split_timescale_candidates(candidates)
        decision = self.decide_maintenance_cycle(
            maintenance_candidates=maintenance,
            st=st,
            cfg=cfg,
        )

        # Online plan always exists for current intent execution.
        online_dag = self.build_dag(online)
        online_plan = self.select_plan(online_dag, st)

        if not decision.run:
            return DualTimescalePlan(
                online_dag=online_dag,
                online_plan=online_plan,
                maintenance_decision=decision,
                maintenance_dag=None,
                maintenance_plan=None,
            )

        maint_dag = self.build_dag(maintenance)
        maint_plan = self.select_plan(maint_dag, st)
        return DualTimescalePlan(
            online_dag=online_dag,
            online_plan=online_plan,
            maintenance_decision=decision,
            maintenance_dag=maint_dag,
            maintenance_plan=maint_plan,
        )

    def commit_plan(
        self,
        plan: OrchestrationPlan,
        dag: ActionDAG,
        st: OrchestrationState,
        *,
        execute_fn: ExecuteFn,
        rollback_fn: Optional[RollbackFn] = None,
        rollback_on_fail: bool = True,
        rollback_on_regression: bool = True,
        regression_threshold_ms: float = 0.0,
    ) -> CommitResult:
        """
        Execute selected actions under guarded serial commit.

        `commit_width` is reserved for future extension. Current default and
        recommended production mode is width=1 to keep actuation deterministic.
        """

        t0 = time.perf_counter()
        result = CommitResult(execution_mode="serial", commit_width=max(1, int(self.commit_width)))
        selected = list(plan.selected_ids)
        if not selected:
            result.commit_wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return result

        levels = self._selected_levels(selected_ids=selected, dag=dag)
        result.levels = [list(x) for x in levels]
        step = max(1, int(self.commit_width))
        if step > 1:
            result.execution_mode = "level_parallel"

        def _finalize_one(aid: str, out: ActionOutcome) -> None:
            action = dag.nodes[aid]
            rollback = False
            reason = ""
            if not out.ok:
                reason = str(out.error or "execute_failed")
                rollback = rollback_on_fail
            elif rollback_on_regression and out.latency_delta_ms < -abs(float(regression_threshold_ms)):
                reason = "latency_regression"
                rollback = True
            elif bool(out.rollback_suggested):
                reason = "guard_requested_rollback"
                rollback = True

            if rollback and rollback_fn is not None:
                rollback_fn(action, st)
                result.rolled_back_ids.append(aid)
            elif out.ok:
                result.applied_ids.append(aid)

            result.records.append(
                CommitRecord(
                    action_id=aid,
                    family=action.family,
                    ok=bool(out.ok),
                    rollback=bool(rollback),
                    reason=reason,
                    latency_delta_ms=float(out.latency_delta_ms),
                )
            )

        for level in levels:
            if not level:
                continue
            if step <= 1 or len(level) <= 1:
                for aid in level:
                    out = execute_fn(dag.nodes[aid], st)
                    _finalize_one(aid, out)
                continue

            # Execute same-level independent actions in bounded parallelism.
            for i in range(0, len(level), step):
                chunk = list(level[i : i + step])
                outcomes: Dict[str, ActionOutcome] = {}
                with ThreadPoolExecutor(max_workers=len(chunk)) as ex:
                    fut_map = {
                        ex.submit(execute_fn, dag.nodes[aid], st): aid
                        for aid in chunk
                    }
                    for fut, aid in list(fut_map.items()):
                        try:
                            outcomes[aid] = fut.result()
                        except Exception as exc:
                            outcomes[aid] = ActionOutcome(
                                ok=False,
                                latency_delta_ms=0.0,
                                error=f"parallel_execute_failed:{type(exc).__name__}:{exc}",
                            )
                # Keep deterministic record order.
                for aid in chunk:
                    _finalize_one(aid, outcomes.get(aid, ActionOutcome(ok=False, error="missing_outcome")))
        result.commit_wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return result

    @staticmethod
    def _selected_levels(*, selected_ids: List[str], dag: ActionDAG) -> List[List[str]]:
        """
        Build dependency levels for selected nodes.
        Nodes in the same level have no selected dependency relation.
        """
        selected = [str(x) for x in selected_ids if str(x) in dag.nodes]
        if not selected:
            return []
        sel_set = set(selected)
        indeg: Dict[str, int] = {nid: 0 for nid in selected}
        out: Dict[str, List[str]] = {nid: [] for nid in selected}
        for e in dag.dependency_edges():
            s, d = str(e.src), str(e.dst)
            if s in sel_set and d in sel_set and s != d:
                indeg[d] = int(indeg.get(d, 0) + 1)
                out.setdefault(s, []).append(d)
        ready = sorted([nid for nid, d in indeg.items() if d == 0])
        levels: List[List[str]] = []
        used: Set[str] = set()
        while ready:
            lv = list(ready)
            levels.append(lv)
            used.update(lv)
            nxt: List[str] = []
            for nid in lv:
                for d in sorted(out.get(nid, [])):
                    indeg[d] -= 1
                    if indeg[d] == 0:
                        nxt.append(d)
            ready = sorted(set(nxt))
        # Cycle-safe fallback: append any remaining nodes as singleton levels.
        rem = sorted([nid for nid in selected if nid not in used])
        for nid in rem:
            levels.append([nid])
        return levels


def _demo_executor(action: OpenOpsAction, _st: OrchestrationState) -> ActionOutcome:
    """
    Deterministic demo executor.
    Positive `expected_gain_ms` is interpreted as latency reduction.
    """
    # Simulate one risky index candidate to show rollback path.
    if action.family == ActionFamily.INDEX and action.risk_score >= 0.9:
        return ActionOutcome(ok=True, latency_delta_ms=-80.0, rollback_suggested=True)
    return ActionOutcome(ok=True, latency_delta_ms=max(0.0, action.expected_gain_ms))


def _demo_rollback(_action: OpenOpsAction, _st: OrchestrationState) -> None:
    return None


def _demo() -> None:
    st = OrchestrationState(
        window_id=12,
        schema_epoch=3,
        budget_total_ms=900.0,
        budget_by_family_ms={
            ActionFamily.SCHEMA: 500.0,
            ActionFamily.INDEX: 250.0,
            ActionFamily.PRECACHE: 300.0,
            ActionFamily.MANIPULATION: 200.0,
        },
        risk_tolerance=0.85,
        physical=PhysicalState(
            cpu_util_pct=61.0,
            cpu_iowait_pct=7.0,
            mem_used_pct=68.0,
            disk_util_pct=44.0,
            running_queries=5,
            query_p95_ms=230.0,
            active_threads=16,
            max_threads=32,
            thermal_throttled=False,
            backend="duckdb",
        ),
        io_budget_mb=1024.0,
        cpu_budget_ms=600.0,
        mem_headroom_mb=4096.0,
    )
    actions = [
        OpenOpsAction(
            action_id="S1",
            family=ActionFamily.SCHEMA,
            schema="reorganize_layout(table, spec)",
            target="orders",
            expected_gain_ms=420.0,
            expected_apply_ms=280.0,
            expected_cpu_ms=180.0,
            expected_io_mb=880.0,
            expected_mem_mb=700.0,
            risk_score=0.35,
            resource_tags={"exclusive:orders"},
            is_heavy=True,
        ),
        OpenOpsAction(
            action_id="I1",
            family=ActionFamily.INDEX,
            schema="build_index(table, keys, include, type)",
            target="orders",
            expected_gain_ms=160.0,
            expected_apply_ms=120.0,
            expected_cpu_ms=70.0,
            expected_io_mb=240.0,
            expected_mem_mb=320.0,
            risk_score=0.30,
            depends_on=["S1"],
            is_heavy=True,
        ),
        OpenOpsAction(
            action_id="I2",
            family=ActionFamily.INDEX,
            schema="build_index(table, keys, include, type)",
            target="lineitem",
            expected_gain_ms=110.0,
            expected_apply_ms=130.0,
            expected_cpu_ms=90.0,
            expected_io_mb=260.0,
            expected_mem_mb=300.0,
            risk_score=0.95,
            is_heavy=True,
        ),
        OpenOpsAction(
            action_id="P1",
            family=ActionFamily.PRECACHE,
            schema="create_artifact(name, spec, budget)",
            target="mv_orders_customer",
            expected_gain_ms=220.0,
            expected_apply_ms=260.0,
            expected_cpu_ms=150.0,
            expected_io_mb=700.0,
            expected_mem_mb=980.0,
            risk_score=0.45,
            conflicts_with=["I2"],
            is_heavy=True,
        ),
        OpenOpsAction(
            action_id="M1",
            family=ActionFamily.MANIPULATION,
            schema="execute_code(code, inputs)",
            target="final_answer_branch",
            expected_gain_ms=40.0,
            expected_apply_ms=8.0,
            expected_cpu_ms=5.0,
            expected_io_mb=0.0,
            expected_mem_mb=32.0,
            risk_score=0.10,
            depends_on=["I1"],
        ),
    ]

    orch = OpenOpsOrchestrator(risk_weight=0.35, min_expected_gain_ms=10.0, commit_width=1)
    actions.extend(orch.suggest_runtime_control_actions(st))
    dag = orch.build_dag(actions)
    plan = orch.select_plan(dag, st)
    out = orch.commit_plan(
        plan,
        dag,
        st,
        execute_fn=_demo_executor,
        rollback_fn=_demo_rollback,
        rollback_on_fail=True,
        rollback_on_regression=True,
        regression_threshold_ms=10.0,
    )

    print("[OpenOps Demo] selected:", plan.selected_ids)
    print("[OpenOps Demo] rejected:", plan.rejected_ids)
    for r in out.records:
        print(
            f"- {r.action_id} ({r.family.value}) ok={int(r.ok)} "
            f"rollback={int(r.rollback)} reason={r.reason or 'none'}"
        )


if __name__ == "__main__":
    _demo()
