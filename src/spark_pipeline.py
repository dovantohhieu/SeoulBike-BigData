"""Spark SQL làm sạch/tạo lag và Spark ML Pipeline huấn luyện hồi quy."""
import os
import time
import numpy as np
import pandas as pd
from .common import ROOT, RAW, FEATURES, FOLDS, TEST_START, WEATHER, LAGS, WINDOWS, FEATURE_DECIMALS, save_json
from .spark_data import read_seoul
from .metrics import regression_metrics

def session(name="SeoulBike"):
    from pyspark.sql import SparkSession
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    # Chuỗi nguồn là giờ Seoul không DST. Khóa timezone kỹ thuật để toPandas
    # không áp dụng DST của máy chủ lên các timestamp không kèm múi giờ.
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    spark = (SparkSession.builder.master("local[2]").appName(name)
        .config("spark.driver.memory", "1g").config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .config("spark.default.parallelism", "2").config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def spark_features(spark):
    from pyspark.sql import functions as F, Window
    df, quality = read_seoul(spark)
    save_json(ROOT / "results/spark_data_quality.json", quality)
    # Một chuỗi duy nhất: cửa sổ này cần sắp thứ tự toàn cục. Không dùng nó để tuyên bố mở rộng tuyến tính.
    order = Window.orderBy("timestamp")
    for lag in LAGS:
        df = df.withColumn(f"lag_{lag}", F.lag("count", lag).over(order))
    for size in WINDOWS:
        frame = order.rowsBetween(-size, -1)
        df = df.withColumn(f"mean_{size}", F.when(F.count("count").over(frame) == size, F.avg("count").over(frame)))
    for c in WEATHER:
        df = df.withColumn(f"{c}_lag1", F.lag(c, 1).over(order))
    dow = (F.dayofweek("timestamp")+5) % 7
    for label, col, period in [("hour", F.hour("timestamp"), 24), ("dow", dow, 7),
                               ("month", F.month("timestamp"), 12)]:
        df = df.withColumn(label+"_sin", F.sin(col * (2*np.pi/period)))
        df = df.withColumn(label+"_cos", F.cos(col * (2*np.pi/period)))
    df = df.withColumn("weekend", (dow >= 5).cast("double"))
    # Chênh lệch sin/cos rất nhỏ gần 0 có thể đổi nhánh của cây. Quy ước cùng độ chính xác.
    for c in FEATURES:
        df = df.withColumn(c, F.bround(c, FEATURE_DECIMALS))
    return df.select("timestamp", "count", "functioning", *FEATURES).dropna()

def estimator():
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor
    return Pipeline(stages=[VectorAssembler(inputCols=FEATURES, outputCol="features"),
        RandomForestRegressor(featuresCol="features", labelCol="count", predictionCol="prediction",
            numTrees=60, maxDepth=8, minInstancesPerNode=3, seed=42)])

def run():
    from pyspark.sql import functions as F
    spark = session()
    try:
        data = spark_features(spark).cache()
        n = data.count()
        # So sánh độc lập mọi đặc trưng với pandas, không chỉ so số hàng.
        sf = data.orderBy("timestamp").toPandas().set_index("timestamp")
        pf = pd.read_csv(ROOT / "data/processed/features.csv", parse_dates=["timestamp"]).set_index("timestamp")
        if not sf.index.equals(pf.index):
            raise AssertionError("Timestamp Spark và pandas không trùng.")
        np.testing.assert_allclose(sf[FEATURES], pf[FEATURES], rtol=1e-10, atol=1e-9)
        np.testing.assert_array_equal(sf["count"], pf["count"])
        save_json(ROOT / "results/spark_feature_parity.json", {"rows":n, "features":len(FEATURES),
            "all_values_equal_within_tolerance":True, "max_abs_difference":float(np.abs(sf[FEATURES]-pf[FEATURES]).max().max())})
        data.write.mode("overwrite").parquet(str(ROOT / "data/processed/spark_features.parquet"))
        cv = []
        for fold, (start,end) in enumerate(FOLDS,1):
            train = data.filter(F.col("timestamp") < start)
            val = data.filter((F.col("timestamp") >= start) & (F.col("timestamp") < end))
            t = time.perf_counter()
            model = estimator().fit(train)
            out = model.transform(val).select("timestamp", "count", "prediction").orderBy("timestamp").toPandas()
            cv.append({"model":"SparkRandomForest", "fold":fold, "train_n":train.count(),
                       "validation_start":start, "validation_end_exclusive":end,
                       "fit_predict_s":time.perf_counter()-t,
                       **regression_metrics(out["count"], np.maximum(0,out["prediction"]))})
            print("SparkRandomForest",fold,round(cv[-1]["MAE"],3),flush=True)
        pd.DataFrame(cv).to_csv(ROOT / "results/cv_spark.csv", index=False)
        train = data.filter(F.col("timestamp") < TEST_START)
        test = data.filter(F.col("timestamp") >= TEST_START)
        t = time.perf_counter()
        model = estimator().fit(train)
        fit_s = time.perf_counter()-t
        model.write().overwrite().save(str(ROOT / "models/SparkRandomForest"))
        out = model.transform(test).select("timestamp", "count", "prediction").orderBy("timestamp").toPandas()
        out["prediction"] = np.maximum(0,out["prediction"])
        out.to_csv(ROOT / "results/predictions_spark.csv", index=False)
        pd.DataFrame([{"model":"SparkRandomForest", "fit_s":fit_s,
            **regression_metrics(out["count"],out["prediction"])}]).to_csv(ROOT / "results/test_spark.csv",index=False)
    finally:
        spark.stop()

if __name__ == "__main__":
    run()
