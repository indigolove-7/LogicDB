from __future__ import annotations

import json
from pathlib import Path

import duckdb


def build_demo_assets(root: Path | None = None) -> dict[str, str]:
    root = (root or Path(__file__).resolve().parent / "demo").resolve()
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "demo.duckdb"
    workload_path = root / "workload.jsonl"

    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE customers AS
            SELECT
              i AS customer_id,
              CASE
                WHEN i % 7 = 0 THEN 'enterprise'
                WHEN i % 3 = 0 THEN 'midmarket'
                ELSE 'consumer'
              END AS segment,
              CASE
                WHEN i % 4 = 0 THEN 'US'
                WHEN i % 4 = 1 THEN 'EU'
                WHEN i % 4 = 2 THEN 'APAC'
                ELSE 'LATAM'
              END AS region,
              DATE '2023-01-01' + CAST((i % 365) AS INTEGER) AS signup_date
            FROM range(1, 120001) t(i)
            """
        )
        con.execute(
            """
            CREATE TABLE orders AS
            SELECT
              i AS order_id,
              1 + (i % 120000) AS customer_id,
              DATE '2024-01-01' + CAST((i % 365) AS INTEGER) AS order_date,
              CAST(40 + (i % 19) * 11 + (i % 7) * 3 AS DOUBLE) AS revenue,
              CASE
                WHEN i % 5 = 0 THEN 'complete'
                WHEN i % 5 = 1 THEN 'pending'
                WHEN i % 5 = 2 THEN 'shipped'
                ELSE 'processing'
              END AS status,
              CASE WHEN i % 6 < 3 THEN 'web' ELSE 'store' END AS channel
            FROM range(1, 240001) t(i)
            """
        )
    finally:
        con.close()

    repeated_queries = [
        (
            1,
            "SELECT region, date_trunc('month', order_date) AS month, SUM(revenue) AS total_revenue "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE order_date >= DATE '2024-03-01' AND status = 'complete' "
            "GROUP BY region, month ORDER BY region, month",
        ),
        (
            2,
            "SELECT segment, channel, COUNT(*) AS order_cnt, SUM(revenue) AS total_revenue "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE order_date >= DATE '2024-05-01' AND status IN ('pending', 'processing') "
            "GROUP BY segment, channel ORDER BY segment, channel",
        ),
        (
            3,
            "SELECT c.customer_id, c.region, COUNT(*) AS complete_orders, AVG(revenue) AS avg_revenue "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE status = 'complete' AND order_date >= DATE '2024-07-01' "
            "GROUP BY c.customer_id, c.region ORDER BY complete_orders DESC, c.customer_id LIMIT 100",
        ),
    ]

    workload = []
    qid = 0
    for phase_id, (template_id, sql) in enumerate(repeated_queries):
        for repeat in range(3):
            workload.append(
                {
                    "qid": qid,
                    "segment_id": phase_id,
                    "template": template_id,
                    "sql": sql,
                    "instance_id": f"demo_t{template_id}_r{repeat + 1}",
                }
            )
            qid += 1

    with workload_path.open("w", encoding="utf-8") as f:
        for row in workload:
            f.write(json.dumps(row) + "\n")

    return {"duckdb_path": str(db_path), "workload_jsonl": str(workload_path)}


def main() -> None:
    print(json.dumps(build_demo_assets(), indent=2))


if __name__ == "__main__":
    main()
