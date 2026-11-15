import json
import joblib
import numpy as np
from .common import ROOT, FEATURES
from .data import next_features

def metadata():
    return json.loads((ROOT / "models/registry.json").read_text(encoding="utf-8"))

def load_model(name=None):
    name = name or metadata()["selected_model"]
    if name == "SparkRandomForest":
        from .spark_pipeline import session
        from pyspark.ml import PipelineModel
        session("ForecastDemo")
        return PipelineModel.load(str(ROOT / "models/SparkRandomForest"))
    allowed = {"Ridge", "RandomForest", "HistGradientBoosting"}
    if name not in allowed:
        raise ValueError("Tên mô hình không hợp lệ.")
    return joblib.load(ROOT / f"models/{name}.joblib")

def predict_frame(model, x, name):
    if name == "SparkRandomForest":
        from .spark_pipeline import session
        spark=session("ForecastDemo")
        frame=x[FEATURES].reset_index(drop=True).copy()
        frame["__forecast_row"]=np.arange(len(frame),dtype="int64")
        values=spark.createDataFrame(frame)
        out=model.transform(values).orderBy("__forecast_row").select("prediction").collect()
        return np.maximum(0, np.array([r.prediction for r in out]))
    return np.maximum(0, model.predict(x[FEATURES]))

def forecast(history, model=None, name=None):
    name = name or metadata()["selected_model"]
    model = model if model is not None else load_model(name)
    x = next_features(history)
    return x.index[0], float(predict_frame(model,x,name)[0]), x
