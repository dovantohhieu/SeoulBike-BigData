"""Đọc và xác thực CSV Seoul bằng Spark, dừng khi hỏng thay vì âm thầm dropna."""
from functools import reduce
from .common import RAW
from .data import RENAME, NUMERIC


def read_seoul(spark, path=RAW):
    from pyspark.sql import functions as F, types as T
    schema = T.StructType([T.StructField(c, T.StringType(), True) for c in RENAME])
    raw = (spark.read.schema(schema).option("header", True)
           .option("enforceSchema", False).option("mode", "FAILFAST")
           .option("encoding", "ISO-8859-1").csv(str(path)))
    df = raw.select(*[F.col(f"`{a}`").alias(b) for a, b in RENAME.items()])
    for c in NUMERIC:
        df = df.withColumn(c, F.col(c).cast("double"))
    df = df.withColumn("timestamp", F.to_timestamp(F.concat_ws(" ", F.col("date"),
        F.format_string("%02d:00:00", F.col("hour").cast("int"))), "dd/MM/yyyy HH:mm:ss"))
    invalid = [F.col(c).isNull() | F.isnan(c) | (F.abs(F.col(c)) == float("inf")) for c in NUMERIC]
    invalid += [F.col("timestamp").isNull(), ~F.col("hour").between(0, 23),
                ~F.col("humidity").between(0, 100), F.col("count") < 0]
    invalid += [F.col(c) < 0 for c in ["wind", "visibility", "solar", "rain", "snow"]]
    invalid += [F.col(c) != F.floor(c) for c in ["count", "hour", "humidity", "visibility"]]
    for c, domain in {"season": ["Winter", "Spring", "Summer", "Autumn"],
                      "holiday": ["Holiday", "No Holiday"], "functioning": ["Yes", "No"]}.items():
        invalid.append(F.col(c).isNull() | ~F.col(c).isin(domain))
    bad = reduce(lambda a, b: a | b, invalid)
    s = df.agg(F.count("*").alias("rows"),
        F.sum(F.when(bad, 1).otherwise(0)).alias("invalid_rows"),
        F.countDistinct("timestamp").alias("unique_hours"),
        F.min("timestamp").alias("first"), F.max("timestamp").alias("last")).first()
    if not s.rows:
        raise ValueError("CSV Seoul đang rỗng.")
    if s.invalid_rows:
        raise ValueError(f"Spark phát hiện {s.invalid_rows} hàng không hợp lệ. Chưa tạo dữ liệu dẫn xuất.")
    duplicates = s.rows - s.unique_hours
    if duplicates:
        raise ValueError(f"Spark phát hiện {duplicates} giờ trùng.")
    expected = int((s.last - s.first).total_seconds() // 3600) + 1
    if expected != s.rows:
        raise ValueError(f"Spark phát hiện {expected - s.rows} giờ thiếu. Không tạo lag trên chuỗi có lỗ hổng.")
    return df, {"raw_rows": s.rows, "invalid_rows": s.invalid_rows,
                "duplicate_hours": duplicates, "missing_hours": expected - s.rows,
                "first": str(s.first), "last": str(s.last)}
