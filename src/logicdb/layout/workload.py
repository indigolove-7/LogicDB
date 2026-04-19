#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workload Characterization for Layout Optimization

This module provides comprehensive workload feature extraction for data layout
optimization decisions. Features are designed to be:
1. Multi-dimensional (suitable for academic paper presentation)
2. Precisely extracted using sqlglot parser
3. Actionable for layout decision-making

Feature Categories:
- Access Patterns: scan, filter, join, aggregation
- Selectivity: filter selectivity, join cardinality
- Temporal: time range predicates, partition pruning potential
- Dimensional: multi-dimensional filter correlation
- Computational: CPU vs I/O bound indicators
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp


# =============================================================================
# Data Structures for Features
# =============================================================================

@dataclass
class ColumnAccessPattern:
    """Per-column access pattern in queries"""
    scan_count: int = 0                    # How many queries scan this column
    filter_count: int = 0                  # Equality or range filters
    join_count: int = 0                    # Used as join key
    group_by_count: int = 0                # GROUP BY
    order_by_count: int = 0                # ORDER BY
    
    # Filter types
    equality_filter_count: int = 0         # col = value
    range_filter_count: int = 0            # col > x, col BETWEEN
    in_filter_count: int = 0               # col IN (...)
    like_filter_count: int = 0             # col LIKE '...'
    
    # Aggregation types
    sum_count: int = 0
    avg_count: int = 0
    min_max_count: int = 0
    count_count: int = 0


@dataclass
class TableAccessPattern:
    """Per-table access pattern across workload"""
    query_count: int = 0                   # How many queries access this table
    total_weight: float = 0.0              # Sum of query weights
    
    # Access types
    full_scan_count: int = 0               # No filters
    filtered_scan_count: int = 0           # Has WHERE predicates
    join_probe_count: int = 0              # Used as join probe side
    join_build_count: int = 0              # Used as join build side
    
    # Column-level patterns
    columns: Dict[str, ColumnAccessPattern] = field(default_factory=dict)
    
    # Temporal analysis
    has_temporal_predicates: bool = False
    temporal_columns: Set[str] = field(default_factory=set)
    
    # Join relationships
    join_with_tables: Dict[str, int] = field(default_factory=dict)  # table -> count


@dataclass
class QueryFeatures:
    """Comprehensive features for a single query"""
    query_id: str
    weight: float = 1.0
    parsed_ok: bool = False
    
    # Basic statistics
    num_tables: int = 0
    num_joins: int = 0
    num_predicates: int = 0
    num_aggregations: int = 0
    num_subqueries: int = 0
    
    # Table access
    tables_accessed: Set[str] = field(default_factory=set)
    fact_tables: Set[str] = field(default_factory=set)      # Large tables
    dimension_tables: Set[str] = field(default_factory=set) # Small tables
    
    # Predicate analysis
    filter_predicates: Dict[str, List[str]] = field(default_factory=dict)  # table -> [col]
    predicate_patterns: Dict[str, List[str]] = field(default_factory=dict)  # table -> [normalized predicate pattern]
    join_predicates: Dict[Tuple[str, str], List[Tuple[str, str]]] = field(default_factory=dict)  # (t1,t2) -> [(col1,col2)]
    
    # Temporal features
    has_time_range_filter: bool = False
    time_filter_columns: Set[Tuple[str, str]] = field(default_factory=set)  # (table, col)
    estimated_time_selectivity: Optional[float] = None
    
    # Multi-dimensional filtering
    multi_dim_filters: Dict[str, Set[str]] = field(default_factory=dict)  # table -> {cols}
    max_filter_dimensions: int = 0
    
    # Join characteristics
    join_types: List[str] = field(default_factory=list)  # inner, left, right
    join_selectivity_hints: Dict[str, str] = field(default_factory=dict)  # table -> 'high'/'medium'/'low'
    
    # Aggregation patterns
    group_by_columns: Dict[str, Set[str]] = field(default_factory=dict)  # table -> {cols}
    agg_functions: List[str] = field(default_factory=list)  # SUM, AVG, etc.
    
    # Sorting
    order_by_columns: Dict[str, List[str]] = field(default_factory=dict)  # table -> [cols]
    
    # Layout hints
    partition_candidate_columns: Dict[str, Set[str]] = field(default_factory=dict)  # table -> {cols}
    clustering_candidate_columns: Dict[str, Set[str]] = field(default_factory=dict)  # table -> {cols}
    zorder_candidate_columns: Dict[str, Set[str]] = field(default_factory=dict)  # table -> {cols}


@dataclass
class WorkloadFeatures:
    """Aggregated features for entire workload"""
    num_queries: int = 0
    total_weight: float = 0.0
    
    # Per-table patterns
    tables: Dict[str, TableAccessPattern] = field(default_factory=dict)
    
    # Workload-level statistics
    avg_joins_per_query: float = 0.0
    avg_predicates_per_query: float = 0.0
    avg_tables_per_query: float = 0.0
    
    # Temporal workload characteristics
    time_range_query_ratio: float = 0.0     # Fraction with time filters
    partition_pruning_potential: float = 0.0 # Estimated benefit from partitioning
    
    # Join patterns
    join_graph: Dict[Tuple[str, str], int] = field(default_factory=dict)  # (t1,t2) -> frequency
    star_schema_detected: bool = False
    snowflake_schema_detected: bool = False
    
    # Layout recommendations (aggregated)
    recommended_partition_columns: Dict[str, Set[str]] = field(default_factory=dict)
    recommended_clustering_columns: Dict[str, Set[str]] = field(default_factory=dict)
    recommended_zorder_columns: Dict[str, Set[str]] = field(default_factory=dict)
    
    # MDDL: Top-K predicates per table (for predicate-driven layouts)
    top_predicates: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)  # table -> [(predicate_sql, frequency)]
    
    # Per-query features (optional, for detailed analysis)
    query_features: List[QueryFeatures] = field(default_factory=list)


# =============================================================================
# SQL Parsing Utilities
# =============================================================================

def _normalize_identifier(s: str) -> str:
    """Normalize table/column names to lowercase"""
    return s.lower().strip() if s else ""


def _collect_table_aliases(root: exp.Expression) -> Dict[str, str]:
    """Build alias -> base_table mapping"""
    alias_map = {}
    for table_node in root.find_all(exp.Table):
        base = _normalize_identifier(table_node.this.name if hasattr(table_node.this, 'name') else str(table_node.this))
        alias = _normalize_identifier(table_node.alias_or_name or base)
        alias_map[alias] = base
    return alias_map


def _infer_unqualified_column_table(col_name: str, alias_map: Dict[str, str]) -> Optional[str]:
    """
    Infer table for unqualified column references.
    TPC-H queries often use unqualified columns like `o_orderdate`, `l_shipdate`.
    """
    tables = set(alias_map.values())
    if not tables:
        return None
    if len(tables) == 1:
        return next(iter(tables))

    prefix = ""
    if "_" in col_name:
        prefix = col_name.split("_", 1)[0].lower()

    tpch_prefix_map = {
        "c": "customer",
        "l": "lineitem",
        "n": "nation",
        "o": "orders",
        "p": "part",
        "ps": "partsupp",
        "r": "region",
        "s": "supplier",
    }
    mapped = tpch_prefix_map.get(prefix)
    if mapped in tables:
        return mapped

    # Generic fallback: if exactly one table starts with the same prefix.
    if prefix:
        cands = [t for t in tables if t.startswith(prefix)]
        if len(cands) == 1:
            return cands[0]

    return None


def _resolve_column_table(col: exp.Column, alias_map: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Resolve Column to (table, column)"""
    col_name = _normalize_identifier(col.name)
    table_ref = _normalize_identifier(col.table or "")
    
    if table_ref:
        base_table = alias_map.get(table_ref, table_ref)
        return (base_table, col_name)
    inferred_table = _infer_unqualified_column_table(col_name, alias_map)
    if inferred_table:
        return (inferred_table, col_name)
    return None


def _is_temporal_column(col_name: str) -> bool:
    """Heuristic: detect temporal columns by name"""
    temporal_keywords = ['date', 'time', 'timestamp', 'year', 'month', 'day', 'shipdate', 'orderdate', 'receiptdate']
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in temporal_keywords)


def _is_join_predicate(pred: exp.Expression, alias_map: Dict[str, str]) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
    """
    Detect join predicate: t1.col1 = t2.col2
    Returns ((t1, col1), (t2, col2)) or None
    """
    if not isinstance(pred, exp.EQ):
        return None
    
    left, right = pred.left, pred.right
    if isinstance(left, exp.Column) and isinstance(right, exp.Column):
        left_tc = _resolve_column_table(left, alias_map)
        right_tc = _resolve_column_table(right, alias_map)
        
        if left_tc and right_tc and left_tc[0] != right_tc[0]:
            return (left_tc, right_tc)
    
    return None


def _extract_literal_value(node: exp.Expression) -> Optional[Any]:
    """Extract Python value from literal expression"""
    if isinstance(node, exp.Literal):
        if node.is_string:
            val = node.name.strip("'\"")
            # Try to parse as date
            try:
                return date.fromisoformat(val)
            except:
                return val
        # Numeric
        try:
            num = float(node.name)
            return int(num) if num == int(num) else num
        except:
            return node.name
    
    # Handle CAST
    if isinstance(node, exp.Cast):
        return _extract_literal_value(node.this)
    
    # Handle DATE '...'
    if isinstance(node, exp.Anonymous) and node.name.upper() == "DATE":
        if node.expressions:
            return _extract_literal_value(node.expressions[0])
    
    return None


# =============================================================================
# Single Query Feature Extraction
# =============================================================================

class QueryCharacterizer:
    """Extract features from a single SQL query"""
    
    def __init__(self, sql: str, query_id: str = "Q", weight: float = 1.0):
        self.sql = sql
        self.query_id = query_id
        self.weight = weight
        self.parsed = self._parse_sql(sql)
        self.alias_map = _collect_table_aliases(self.parsed) if self.parsed else {}
    
    def _parse_sql(self, sql: str) -> Optional[exp.Expression]:
        """Parse SQL with fallback dialects"""
        for dialect in ["duckdb", "postgres", "ansi"]:
            try:
                return sqlglot.parse_one(sql, read=dialect)
            except:
                continue
        return None
    
    def extract_features(self) -> QueryFeatures:
        """Extract comprehensive query features"""
        if not self.parsed:
            return QueryFeatures(query_id=self.query_id, weight=self.weight, parsed_ok=False)
        
        features = QueryFeatures(query_id=self.query_id, weight=self.weight, parsed_ok=True)
        
        # Basic statistics
        features.num_tables = len(list(self.parsed.find_all(exp.Table)))
        # NOTE: exp.Join may miss implicit joins written as `FROM t1, t2, ... WHERE t1.k=t2.k`.
        # We'll compute join count from join predicates after _extract_joins().
        features.num_joins = 0
        features.num_subqueries = len(list(self.parsed.find_all(exp.Subquery)))
        
        # Tables accessed
        features.tables_accessed = set(self.alias_map.values())
        
        # Predicates
        self._extract_predicates(features)
        
        # Joins
        self._extract_joins(features)
        # More faithful for TPC-H: count actual join predicates found (t1.col=t2.col)
        features.num_joins = sum(len(v) for v in features.join_predicates.values())
        
        # Aggregations
        self._extract_aggregations(features)
        
        # Sorting
        self._extract_sorting(features)
        
        # Layout hints
        self._generate_layout_hints(features)
        
        return features
    
    def _extract_predicates(self, features: QueryFeatures) -> None:
        """Extract WHERE/HAVING predicates"""
        predicate_roots = []
        
        for where_node in self.parsed.find_all(exp.Where):
            if where_node.this:
                predicate_roots.append(where_node.this)
        
        for having_node in self.parsed.find_all(exp.Having):
            if having_node.this:
                predicate_roots.append(having_node.this)
        
        for root in predicate_roots:
            for node in root.walk():
                # Skip join predicates (handled separately)
                if _is_join_predicate(node, self.alias_map):
                    continue
                
                # Filter predicates on columns
                if isinstance(node, (exp.EQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.Between)):
                    self._handle_filter_predicate(node, features)
                elif isinstance(node, exp.In):
                    self._handle_in_predicate(node, features)
                elif isinstance(node, (exp.Like, exp.ILike)):
                    self._handle_like_predicate(node, features)
        
        features.num_predicates = sum(len(v) for v in features.filter_predicates.values())
    
    def _handle_filter_predicate(self, node: exp.Expression, features: QueryFeatures) -> None:
        """Handle equality and range filter predicates"""
        col = None
        literal = None
        
        if isinstance(node, (exp.EQ, exp.LT, exp.LTE, exp.GT, exp.GTE)):
            left, right = node.left, node.right
            if isinstance(left, exp.Column) and not isinstance(right, exp.Column):
                col = left
                literal = _extract_literal_value(right)
            elif isinstance(right, exp.Column) and not isinstance(left, exp.Column):
                col = right
                literal = _extract_literal_value(left)
        
        elif isinstance(node, exp.Between):
            if isinstance(node.this, exp.Column):
                col = node.this
                # Extract range bounds
                low = _extract_literal_value(node.args.get("low"))
                high = _extract_literal_value(node.args.get("high"))
                literal = (low, high)
        
        if col:
            tc = _resolve_column_table(col, self.alias_map)
            if tc:
                table, col_name = tc
                features.filter_predicates.setdefault(table, []).append(col_name)
                features.multi_dim_filters.setdefault(table, set()).add(col_name)

                # Predicate pattern (normalized, no literals)
                if isinstance(node, exp.EQ):
                    pat = f"{col_name} = ?"
                elif isinstance(node, (exp.LT, exp.LTE)):
                    pat = f"{col_name} < ?"
                elif isinstance(node, (exp.GT, exp.GTE)):
                    pat = f"{col_name} > ?"
                elif isinstance(node, exp.Between):
                    pat = f"{col_name} BETWEEN ? AND ?"
                else:
                    pat = f"{col_name} ? ?"
                features.predicate_patterns.setdefault(table, []).append(pat)
                
                # Check temporal
                if _is_temporal_column(col_name):
                    features.has_time_range_filter = True
                    features.time_filter_columns.add((table, col_name))
                    
                    # Estimate selectivity from literal if date range
                    if isinstance(literal, tuple) and len(literal) == 2:
                        low, high = literal
                        if isinstance(low, date) and isinstance(high, date):
                            days = (high - low).days
                            # Rough selectivity: assume data spans ~7 years (TPC-H typical)
                            features.estimated_time_selectivity = min(1.0, days / (7 * 365))
    
    def _handle_in_predicate(self, node: exp.In, features: QueryFeatures) -> None:
        """Handle IN (...) predicates"""
        if isinstance(node.this, exp.Column):
            tc = _resolve_column_table(node.this, self.alias_map)
            if tc:
                table, col_name = tc
                features.filter_predicates.setdefault(table, []).append(col_name)
                features.multi_dim_filters.setdefault(table, set()).add(col_name)
                features.predicate_patterns.setdefault(table, []).append(f"{col_name} IN (?)")
    
    def _handle_like_predicate(self, node: exp.Expression, features: QueryFeatures) -> None:
        """Handle LIKE predicates"""
        if isinstance(node.this, exp.Column):
            tc = _resolve_column_table(node.this, self.alias_map)
            if tc:
                table, col_name = tc
                features.filter_predicates.setdefault(table, []).append(col_name)
                features.predicate_patterns.setdefault(table, []).append(f"{col_name} LIKE ?")
    
    def _extract_joins(self, features: QueryFeatures) -> None:
        """Extract join predicates and types"""
        # Join types
        for join_node in self.parsed.find_all(exp.Join):
            join_type = join_node.args.get("kind", "inner")
            features.join_types.append(str(join_type).lower())
        
        # Join predicates (from WHERE and ON clauses)
        all_conditions = []
        
        for where_node in self.parsed.find_all(exp.Where):
            if where_node.this:
                all_conditions.append(where_node.this)
        
        for join_node in self.parsed.find_all(exp.Join):
            if join_node.args.get("on"):
                all_conditions.append(join_node.args["on"])
        
        for cond_root in all_conditions:
            for node in cond_root.walk():
                join_pred = _is_join_predicate(node, self.alias_map)
                if join_pred:
                    (t1, c1), (t2, c2) = join_pred
                    # Normalize table order
                    if t1 > t2:
                        t1, c1, t2, c2 = t2, c2, t1, c1
                    
                    features.join_predicates.setdefault((t1, t2), []).append((c1, c2))
    
    def _extract_aggregations(self, features: QueryFeatures) -> None:
        """Extract GROUP BY and aggregation functions"""
        # GROUP BY
        for group_node in self.parsed.find_all(exp.Group):
            for expr in group_node.expressions:
                if isinstance(expr, exp.Column):
                    tc = _resolve_column_table(expr, self.alias_map)
                    if tc:
                        table, col = tc
                        features.group_by_columns.setdefault(table, set()).add(col)
        
        # Aggregation functions
        agg_types = {
            exp.Sum: "SUM",
            exp.Avg: "AVG",
            exp.Min: "MIN",
            exp.Max: "MAX",
            exp.Count: "COUNT",
        }
        
        for agg_class, agg_name in agg_types.items():
            for _ in self.parsed.find_all(agg_class):
                features.agg_functions.append(agg_name)
        
        features.num_aggregations = len(features.agg_functions)
    
    def _extract_sorting(self, features: QueryFeatures) -> None:
        """Extract ORDER BY columns"""
        for order_node in self.parsed.find_all(exp.Order):
            for expr in order_node.expressions:
                if isinstance(expr, exp.Ordered):
                    col = expr.this
                    if isinstance(col, exp.Column):
                        tc = _resolve_column_table(col, self.alias_map)
                        if tc:
                            table, col_name = tc
                            features.order_by_columns.setdefault(table, []).append(col_name)
    
    def _generate_layout_hints(self, features: QueryFeatures) -> None:
        """Generate layout optimization hints from query patterns"""
        # Partition candidates: temporal columns with range filters
        for table, col in features.time_filter_columns:
            features.partition_candidate_columns.setdefault(table, set()).add(col)
        
        # Clustering candidates: columns in ORDER BY or single-dimensional filters
        for table, cols in features.order_by_columns.items():
            features.clustering_candidate_columns.setdefault(table, set()).update(cols)
        
        for table, cols in features.filter_predicates.items():
            if len(set(cols)) <= 2:  # Single or two-column filters
                features.clustering_candidate_columns.setdefault(table, set()).update(set(cols))
        
        # Z-order candidates: multi-dimensional filters (3+ columns)
        for table, cols in features.multi_dim_filters.items():
            if len(cols) >= 3:
                features.zorder_candidate_columns.setdefault(table, set()).update(cols)
        
        features.max_filter_dimensions = max(
            (len(cols) for cols in features.multi_dim_filters.values()),
            default=0
        )


# =============================================================================
# Workload Aggregation
# =============================================================================

class WorkloadCharacterizer:
    """Aggregate features from multiple queries"""
    
    def __init__(self):
        self.query_features: List[QueryFeatures] = []
    
    def add_query(self, sql: str, query_id: str = "Q", weight: float = 1.0) -> QueryFeatures:
        """Add a query and extract its features"""
        qc = QueryCharacterizer(sql, query_id, weight)
        qf = qc.extract_features()
        self.query_features.append(qf)
        return qf
    
    def aggregate(self) -> WorkloadFeatures:
        """Aggregate all query features into workload-level features"""
        wf = WorkloadFeatures()
        wf.num_queries = len(self.query_features)
        wf.total_weight = sum(qf.weight for qf in self.query_features)
        wf.query_features = self.query_features
        
        if wf.num_queries == 0:
            return wf
        
        # Aggregate per-table patterns
        for qf in self.query_features:
            for table in qf.tables_accessed:
                tap = wf.tables.setdefault(table, TableAccessPattern())
                tap.query_count += 1
                tap.total_weight += qf.weight
                
                # Filter vs full scan
                if table in qf.filter_predicates and qf.filter_predicates[table]:
                    tap.filtered_scan_count += 1
                else:
                    tap.full_scan_count += 1
                
                # Temporal
                if any(tc[0] == table for tc in qf.time_filter_columns):
                    tap.has_temporal_predicates = True
                    tap.temporal_columns.update(tc[1] for tc in qf.time_filter_columns if tc[0] == table)
                
                # Column-level patterns
                for col in qf.filter_predicates.get(table, []):
                    cap = tap.columns.setdefault(col, ColumnAccessPattern())
                    cap.scan_count += 1
                    cap.filter_count += 1
                
                for col in qf.group_by_columns.get(table, set()):
                    tap.columns.setdefault(col, ColumnAccessPattern()).group_by_count += 1
                
                for col in qf.order_by_columns.get(table, []):
                    tap.columns.setdefault(col, ColumnAccessPattern()).order_by_count += 1
            
            # Join relationships
            for (t1, t2), join_cols in qf.join_predicates.items():
                wf.join_graph[(t1, t2)] = wf.join_graph.get((t1, t2), 0) + 1

                # Join extraction can surface tables that were not recorded in
                # tables_accessed (e.g., benchmark-specific aliases/CTEs).
                # Normalize them into the workload map before updating join stats.
                tap1 = wf.tables.setdefault(t1, TableAccessPattern())
                tap2 = wf.tables.setdefault(t2, TableAccessPattern())
                tap1.join_with_tables[t2] = tap1.join_with_tables.get(t2, 0) + 1
                tap2.join_with_tables[t1] = tap2.join_with_tables.get(t1, 0) + 1
                
                # Mark join columns
                for c1, c2 in join_cols:
                    wf.tables[t1].columns.setdefault(c1, ColumnAccessPattern()).join_count += 1
                    wf.tables[t2].columns.setdefault(c2, ColumnAccessPattern()).join_count += 1
        
        # Workload statistics
        wf.avg_joins_per_query = sum(qf.num_joins for qf in self.query_features) / wf.num_queries
        wf.avg_predicates_per_query = sum(qf.num_predicates for qf in self.query_features) / wf.num_queries
        wf.avg_tables_per_query = sum(qf.num_tables for qf in self.query_features) / wf.num_queries
        
        wf.time_range_query_ratio = sum(1 for qf in self.query_features if qf.has_time_range_filter) / wf.num_queries
        
        # Aggregate layout recommendations
        for qf in self.query_features:
            for table, cols in qf.partition_candidate_columns.items():
                wf.recommended_partition_columns.setdefault(table, set()).update(cols)
            for table, cols in qf.clustering_candidate_columns.items():
                wf.recommended_clustering_columns.setdefault(table, set()).update(cols)
            for table, cols in qf.zorder_candidate_columns.items():
                wf.recommended_zorder_columns.setdefault(table, set()).update(cols)
        
        # Partition pruning potential (estimated from temporal queries)
        if wf.time_range_query_ratio > 0:
            avg_time_sel = sum(qf.estimated_time_selectivity or 0.5 for qf in self.query_features if qf.has_time_range_filter) / max(1, sum(1 for qf in self.query_features if qf.has_time_range_filter))
            wf.partition_pruning_potential = wf.time_range_query_ratio * (1.0 - avg_time_sel)
        
        # Extract top-K predicates per table for MDDL
        wf.top_predicates = self._extract_top_predicates(top_k=5)
        
        return wf
    
    def _extract_top_predicates(self, top_k: int = 5) -> Dict[str, List[Tuple[str, int]]]:
        """
        Extract top-K most frequent predicates per table for MDDL.
        Returns table -> [(predicate_pattern, frequency)]
        
        Note: This is a simplified version. Full implementation would need
        to parse predicates from SQL and normalize them.
        """
        predicate_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for qf in self.query_features:
            for table, pats in getattr(qf, "predicate_patterns", {}).items():
                for pat in pats:
                    predicate_counts[table][pat] += 1
        
        # Get top-K per table
        result = {}
        for table, preds in predicate_counts.items():
            sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
            result[table] = sorted_preds[:top_k]
        
        return result


# =============================================================================
# Export Functions
# =============================================================================

def characterize_workload_to_json(queries: List[Tuple[str, str, float]], output_path: str) -> WorkloadFeatures:
    """
    Characterize a workload and export to JSON
    
    Args:
        queries: List of (sql, query_id, weight) tuples
        output_path: Path to write JSON output
    
    Returns:
        WorkloadFeatures object
    """
    wc = WorkloadCharacterizer()
    
    for sql, qid, weight in queries:
        wc.add_query(sql, qid, weight)
    
    wf = wc.aggregate()
    
    # Convert to JSON-serializable format
    output = {
        "summary": {
            "num_queries": wf.num_queries,
            "total_weight": wf.total_weight,
            "avg_joins_per_query": wf.avg_joins_per_query,
            "avg_predicates_per_query": wf.avg_predicates_per_query,
            "avg_tables_per_query": wf.avg_tables_per_query,
            "time_range_query_ratio": wf.time_range_query_ratio,
            "partition_pruning_potential": wf.partition_pruning_potential,
        },
        "tables": {},
        "join_graph": {f"{t1}<->{t2}": count for (t1, t2), count in wf.join_graph.items()},
        "top_predicates": {table: [{"predicate": pred, "frequency": freq} for pred, freq in preds] 
                          for table, preds in wf.top_predicates.items()},
        "layout_recommendations": {
            "partition": {t: list(cols) for t, cols in wf.recommended_partition_columns.items()},
            "clustering": {t: list(cols) for t, cols in wf.recommended_clustering_columns.items()},
            "zorder": {t: list(cols) for t, cols in wf.recommended_zorder_columns.items()},
        },
    }
    
    # Per-table details
    for table, tap in wf.tables.items():
        output["tables"][table] = {
            "query_count": tap.query_count,
            "total_weight": tap.total_weight,
            "full_scan_count": tap.full_scan_count,
            "filtered_scan_count": tap.filtered_scan_count,
            "has_temporal_predicates": tap.has_temporal_predicates,
            "temporal_columns": list(tap.temporal_columns),
            "join_with_tables": tap.join_with_tables,
            "top_filtered_columns": sorted(
                [(col, cap.filter_count) for col, cap in tap.columns.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "top_join_columns": sorted(
                [(col, cap.join_count) for col, cap in tap.columns.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return wf


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    import duckdb
    
    parser = argparse.ArgumentParser(description="Characterize SQL workload for layout optimization")
    parser.add_argument("--db", required=True, help="DuckDB database file")
    parser.add_argument("--queries", default="tpch", help="'tpch' or path to queries JSON")
    parser.add_argument("--output", default="workload_features.json", help="Output JSON path")
    args = parser.parse_args()
    
    con = duckdb.connect(args.db)
    
    queries = []
    if args.queries == "tpch":
        con.execute("INSTALL tpch;")
        con.execute("LOAD tpch;")
        rows = con.execute("SELECT query_nr, query FROM tpch_queries() ORDER BY query_nr;").fetchall()
        queries = [(str(row[1]).strip().rstrip(";"), f"Q{int(row[0]):02d}", 1.0) for row in rows]
    else:
        with open(args.queries, "r") as f:
            data = json.load(f)
            queries = [(q["sql"], q.get("id", f"Q{i}"), q.get("weight", 1.0)) for i, q in enumerate(data["queries"])]
    
    print(f"Characterizing {len(queries)} queries...")
    wf = characterize_workload_to_json(queries, args.output)
    
    print(f"\nWorkload Summary:")
    print(f"  Queries: {wf.num_queries}")
    print(f"  Avg tables/query: {wf.avg_tables_per_query:.2f}")
    print(f"  Avg joins/query: {wf.avg_joins_per_query:.2f}")
    print(f"  Avg predicates/query: {wf.avg_predicates_per_query:.2f}")
    print(f"  Time range query ratio: {wf.time_range_query_ratio:.2%}")
    print(f"  Partition pruning potential: {wf.partition_pruning_potential:.2%}")
    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
