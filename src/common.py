from pathlib import Path
import json
import hashlib
import platform
from datetime import datetime, timezone
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/SeoulBikeData.csv"
TEST_START = "2018-10-01"
FOLDS = [("2018-06-01", "2018-07-01"), ("2018-07-01", "2018-08-01"), ("2018-08-01", "2018-09-01")]
LAGS = [1, 2, 3, 24, 48, 168]
WINDOWS = [6, 24, 168]
# Thống nhất ranh giới split của cây giữa numpy và Spark (sin/cos gần 0).
FEATURE_DECIMALS = 10
WEATHER = ["temperature", "humidity", "wind", "rain", "snow"]
FEATURES = ([f"lag_{x}" for x in LAGS] + [f"mean_{x}" for x in WINDOWS]
            + [f"{x}_lag1" for x in WEATHER]
            + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "weekend"])

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def environment():
    packages = {}
    for p in ["pandas", "numpy", "scikit-learn", "pyspark", "streamlit", "psutil", "joblib", "matplotlib"]:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = None
    return {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "python": platform.python_version(), "packages": packages, "raw_sha256": sha256(RAW)}
