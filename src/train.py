"""CV mở rộng theo tháng; cố định cấu hình, chọn theo MAE CV trước khi đọc test."""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from .common import ROOT, FEATURES, FEATURE_DECIMALS, TEST_START, FOLDS, save_json, environment
from .data import prepare
from .metrics import regression_metrics

def candidates():
    return {"Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
            "RandomForest": RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=42, n_jobs=2),
            "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                min_samples_leaf=30, learning_rate=0.06, early_stopping=False, random_state=42)}

def run():
    _, data = prepare()
    protocol = {"horizon_hours": 1, "mode": "rolling-origin with observed history, fixed model in each fold",
                "folds": FOLDS, "test_start": TEST_START, "test_end": str(data.index.max()),
                "selection_metric": "mean unweighted fold MAE", "random_seed": 42,
                "features": FEATURES, "feature_decimal_places": FEATURE_DECIMALS, "hyperparameter_search": False,
                "development_disclosure": "An earlier feasibility pilot inspected an overlapping final period. This is a reproducible historical backtest, not an untouched external test or prospective validation.",
                "candidates": {k: str(v) for k,v in candidates().items()},
                "spark_candidate": "RandomForestRegressor(numTrees=60,maxDepth=8,minInstancesPerNode=3,seed=42)"}
    save_json(ROOT / "results/protocol.json", protocol)
    save_json(ROOT / "results/environment.json", environment())
    cv = []
    for name, model in candidates().items():
        for fold, (start, end) in enumerate(FOLDS, 1):
            train = data.loc[data.index < start]
            val = data.loc[(data.index >= start) & (data.index < end)]
            t = time.perf_counter()
            model.fit(train[FEATURES], train["count"])
            pred = np.maximum(0, model.predict(val[FEATURES]))
            cv.append({"model": name, "fold": fold, "train_n": len(train), "validation_start": start,
                       "validation_end_exclusive": end, "fit_predict_s": time.perf_counter()-t,
                       **regression_metrics(val["count"], pred)})
            print(name, fold, round(cv[-1]["MAE"], 3), flush=True)
    pd.DataFrame(cv).to_csv(ROOT / "results/cv_sklearn.csv", index=False)
    train, test = data.loc[data.index < TEST_START], data.loc[data.index >= TEST_START]
    preds = test[["count", "functioning"]].copy()
    stats = []
    for lag in [1, 24, 168]:
        name = f"Naive_{lag}h"
        preds[name] = test[f"lag_{lag}"]
        stats.append({"model": name, "fit_s": 0.0, **regression_metrics(test["count"], preds[name])})
    for name, model in candidates().items():
        t = time.perf_counter()
        model.fit(train[FEATURES], train["count"])
        elapsed = time.perf_counter()-t
        preds[name] = np.maximum(0, model.predict(test[FEATURES]))
        joblib.dump(model, ROOT / f"models/{name}.joblib", compress=3)
        stats.append({"model": name, "fit_s": elapsed, **regression_metrics(test["count"], preds[name])})
    preds.to_csv(ROOT / "results/predictions_sklearn.csv", index_label="timestamp")
    pd.DataFrame(stats).to_csv(ROOT / "results/test_sklearn.csv", index=False)
    save_json(ROOT / "results/splits.json", {"train_n":len(train), "test_n":len(test),
        "train_start":str(train.index.min()), "train_end":str(train.index.max()),
        "test_start":str(test.index.min()), "test_end":str(test.index.max())})

if __name__ == "__main__":
    run()
