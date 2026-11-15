"""Demo Spark thật: CSV có schema, chất lượng, Parquet phân vùng, SQL và kế hoạch đọc.

Chỉ collect tối đa 288 nhóm và 10 hàng xem trước. Dữ liệu nhân bản tách khỏi ML.
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, RAW, save_json, sha256, environment
from .data import load_data
from .spark_data import read_seoul
from .spark_pipeline import session

SIZES = [8760, 100000, 1000000]
MONTH = "2018-10"


def reference_aggregates(rows):
    """Đối chiếu độc lập bằng trọng số, chỉ giữ 8.760 hàng ở driver."""
    original = load_data().reset_index()
    original["weight"] = rows // len(original) + (np.arange(len(original)) < rows % len(original))
    original["observation_month"] = original.timestamp.dt.strftime("%Y-%m")
    original["weighted_total"] = original["count"] * original.weight
    original["weighted_temperature"] = original.temperature * original.weight
    result = original.groupby(["observation_month", "hour"]).agg(
        rows=("weight", "sum"), total=("weighted_total", "sum"),
        temperature_sum=("weighted_temperature", "sum")).reset_index()
    result = result.loc[result.rows > 0].copy()
    result["avg_temperature"] = result.temperature_sum / result.rows
    return result.drop(columns="temperature_sum").sort_values(["observation_month", "hour"]).reset_index(drop=True)


def run(rows=8760):
    if rows not in SIZES:
        raise ValueError(f"Chọn quy mô trong {SIZES}.")
    from pyspark.sql import functions as F, Window
    start = time.perf_counter()
    spark = session("SeoulBike-ETL-Demo")
    base = None
    temp_dir = None
    output = ROOT / f"results/lab/{rows}"
    output.mkdir(parents=True, exist_ok=True)
    try:
        clean, quality = read_seoul(spark)
        base = (clean.withColumn("source_row_id", F.row_number().over(Window.orderBy("timestamp")) - 1)
                .withColumn("observation_month", F.date_format("timestamp", "yyyy-MM"))
                .select("source_row_id", "timestamp", "observation_month", "hour", "count",
                        "temperature", "humidity", "wind", "rain", "snow", "functioning").cache())
        source_rows = base.count()
        if source_rows != 8760:
            raise ValueError("Demo này dành cho đúng phiên bản Seoul 8.760 giờ đã đóng gói.")
        quality_s = time.perf_counter() - start
        # Broadcast chỉ áp dụng bảng gốc nhỏ. Range tạo tải trên Spark, không tạo 1 triệu hàng ở Python.
        copies = spark.range(rows).withColumnRenamed("id", "row_id")
        copies = copies.withColumn("source_row_id", F.col("row_id") % F.lit(source_rows))
        expanded = copies.join(F.broadcast(base), "source_row_id").withColumn(
            "copy_id", F.floor(F.col("row_id") / F.lit(source_rows)).cast("long"))
        expand_plan = expanded._jdf.queryExecution().toString()
        before_partitions = expanded.rdd.getNumPartitions()
        parent = ROOT / "data/lab"
        parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=".building-", dir=parent))
        parquet = temp_dir / "silver.parquet"
        t = time.perf_counter()
        expanded.write.mode("overwrite").partitionBy("observation_month").parquet(str(parquet))
        write_s = time.perf_counter() - t
        silver = spark.read.parquet(str(parquet))
        silver.createOrReplaceTempView("bike_demo_silver")
        sql = """SELECT observation_month, CAST(hour AS INT) AS hour,
            COUNT(*) AS rows, SUM(count) AS total, AVG(temperature) AS avg_temperature
            FROM bike_demo_silver GROUP BY observation_month, hour
            ORDER BY observation_month, hour"""
        t = time.perf_counter()
        aggregation = spark.sql(sql)
        # Bảng nhóm bị chặn ở 12 tháng x 24 giờ. Không toPandas trên tập nhân bản.
        aggregate = pd.DataFrame([r.asDict() for r in aggregation.collect()])
        sql_s = time.perf_counter() - t
        expected = reference_aggregates(rows)
        aggregate = aggregate.sort_values(["observation_month", "hour"]).reset_index(drop=True)
        if aggregate[["observation_month", "hour"]].to_dict("records") != expected[["observation_month", "hour"]].to_dict("records"):
            raise AssertionError("Spark/pandas không trùng nhóm tháng và giờ.")
        np.testing.assert_allclose(aggregate[["rows", "total", "avg_temperature"]],
            expected[["rows", "total", "avg_temperature"]], rtol=1e-10, atol=1e-8)
        if int(aggregate.rows.sum()) != rows:
            raise AssertionError("Số hàng sau khi ghi/đọc Parquet không đúng.")
        selected = silver.filter(F.col("observation_month") == MONTH).select("timestamp", "hour", "count")
        selected_count = selected.count()
        selected_plan = selected._jdf.queryExecution().toString()
        expected_count = int(expected.loc[expected.observation_month.eq(MONTH), "rows"].sum())
        if selected_count != expected_count:
            raise AssertionError("Truy vấn tháng 10 không đúng số hàng.")
        folders = sorted(p for p in parquet.glob("observation_month=*") if p.is_dir())
        files = list(parquet.rglob("*.parquet"))
        selected_files = list((parquet / f"observation_month={MONTH}").glob("*.parquet"))
        pruning = "PartitionFilters:" in selected_plan and MONTH in selected_plan
        if not pruning:
            raise AssertionError("Kế hoạch Spark chưa ghi nhận bộ lọc phân vùng.")
        preview = [r.asDict() for r in selected.orderBy("timestamp", "hour").limit(10).collect()]
        # File plan được đổi sang đường dẫn kết quả ổn định trước khi bàn giao.
        final_dir = parent / f"rows_{rows}"
        final_parquet = final_dir / "silver.parquet"
        def stable(plan):
            return plan.replace(str(parquet), str(final_parquet))
        summary = {
            "status": "passed", "utc": environment()["utc"], "environment": environment(),
            "source_rows": source_rows, "processed_rows": rows, "replicated": rows != source_rows,
            "scope": "Spark local[2] ETL demo; repeated rows only for load, never used for training/testing",
            "source_sha256": sha256(RAW), "quality": quality, "range_partitions": before_partitions,
            "month_partitions": len(folders), "parquet_files": len(files),
            "parquet_bytes": sum(p.stat().st_size for p in files),
            "partition_months": [p.name.split("=", 1)[1] for p in folders],
            "query_month": MONTH, "query_rows": selected_count,
            "query_partition_files": len(selected_files),
            "partition_filter_in_plan": pruning, "aggregate_groups": len(aggregate),
            "aggregates_match_pandas": True, "collected_preview_rows": len(preview),
            "timing_s": {"startup_and_quality": quality_s, "write_parquet": write_s,
                         "sql_collect": sql_s, "total": time.perf_counter() - start},
            "timing_scope": "One demonstration run, includes checks and IO; distinct from the 3-repeat engine benchmark",
            "parquet_path": str(final_parquet.relative_to(ROOT)),
            "schema": silver.schema.simpleString(),
            "sql": sql, "preview": preview,
        }
        # Chỉ cập nhật kết quả hoàn chỉnh. Giữ bộ đã chạy thành công nếu một bước trước thất bại.
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.rename(final_dir)
        temp_dir = None
        aggregate.to_csv(output / "gold_month_hour.csv", index=False)
        (output / "plan_expand.txt").write_text(stable(expand_plan), encoding="utf-8")
        (output / "plan_aggregation.txt").write_text(stable(aggregation._jdf.queryExecution().toString()), encoding="utf-8")
        (output / "plan_partition_filter.txt").write_text(stable(selected_plan), encoding="utf-8")
        save_json(output / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ["status", "processed_rows", "month_partitions",
            "query_rows", "aggregates_match_pandas", "parquet_path"]}, ensure_ascii=False), flush=True)
        return summary
    finally:
        if base is not None:
            base.unpersist()
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, choices=SIZES, default=8760)
    run(parser.parse_args().rows)
