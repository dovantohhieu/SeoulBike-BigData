"""Một tiến trình riêng cho mỗi lượt đo. Khởi động Spark nằm ngoài etl_s nhưng trong wall_s."""
import argparse
import time
from pathlib import Path
from .common import save_json

def run(engine, path, output):
    t0 = time.perf_counter()
    if engine == "pandas":
        import pandas as pd
        startup = time.perf_counter()-t0
        t1 = time.perf_counter()
        df = pd.read_csv(path, dtype={"row_id":"int64", "hour":"int32", "count":"float64", "temperature":"float64", "humidity":"float64"})
        df = df.dropna(subset=["count", "temperature", "humidity", "hour"])
        df = df[(df["count"] >= 0) & df["hour"].between(0,23) & df["humidity"].between(0,100)]
        result = df.groupby("hour").agg(rows=("row_id","size"), total=("count","sum"),
                    avg_temperature=("temperature","mean")).reset_index().sort_values("hour")
        rows = result.to_dict("records")
        elapsed = time.perf_counter()-t1
    else:
        from .spark_pipeline import session
        from pyspark.sql import functions as F
        spark = session("ETLBenchmark")
        startup = time.perf_counter()-t0
        t1 = time.perf_counter()
        df = spark.read.schema("row_id long, hour int, count double, temperature double, humidity double").option("header",True).csv(str(path))
        df = df.na.drop(subset=["count","temperature","humidity","hour"])
        df = df.filter((F.col("count") >= 0) & F.col("hour").between(0,23) & F.col("humidity").between(0,100))
        # collect bắt buộc thực thi DAG và trả cùng 24 nhóm như pandas.
        rows = [r.asDict() for r in df.groupBy("hour").agg(F.count("row_id").alias("rows"),
                F.sum("count").alias("total"), F.avg("temperature").alias("avg_temperature")).orderBy("hour").collect()]
        elapsed = time.perf_counter()-t1
        spark.stop()
    save_json(output, {"engine":engine, "startup_s":startup, "etl_s":elapsed,
        "worker_s":time.perf_counter()-t0, "input_bytes":Path(path).stat().st_size, "aggregates":rows})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["pandas","spark"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.engine, a.input, a.output)
