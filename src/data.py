"""Dữ liệu một chuỗi theo giờ. Dừng rõ ràng khi đầu vào hỏng, không tự bịa giờ thiếu."""
import numpy as np
import pandas as pd
from .common import RAW, ROOT, LAGS, WINDOWS, WEATHER, FEATURES, FEATURE_DECIMALS, save_json, sha256

RENAME = {"Date": "date", "Rented Bike Count": "count", "Hour": "hour",
          "Temperature(°C)": "temperature", "Humidity(%)": "humidity", "Wind speed (m/s)": "wind",
          "Visibility (10m)": "visibility", "Dew point temperature(°C)": "dew_point",
          "Solar Radiation (MJ/m2)": "solar", "Rainfall(mm)": "rain", "Snowfall (cm)": "snow",
          "Seasons": "season", "Holiday": "holiday", "Functioning Day": "functioning"}
NUMERIC = list(RENAME.values())[1:11]

def validate_hourly(df, allow_last_target_missing=False):
    if df.empty:
        raise ValueError("Dữ liệu lịch sử đang rỗng.")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise ValueError("Cột thời gian không hợp lệ.")
    if df.index.duplicated().any():
        raise ValueError("Có giờ bị trùng. Cần xử lý trước khi tạo lag.")
    if len(df) > 1 and not (df.index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("Chuỗi phải liên tục từng giờ và sắp tăng dần.")
    check = df.iloc[:-1] if allow_last_target_missing else df
    vals = check[["count"] + WEATHER].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Dữ liệu lịch sử bị thiếu hoặc có giá trị vô hạn.")
    if (check["count"] < 0).any() or (check["count"] % 1 != 0).any():
        raise ValueError("Số lượt thuê phải là số nguyên không âm.")
    if not check["humidity"].between(0, 100).all():
        raise ValueError("Độ ẩm phải từ 0 đến 100%.")
    if (check[["wind", "rain", "snow"]] < 0).any().any():
        raise ValueError("Gió, mưa, tuyết không thể âm.")
    return df

def load_data(path=RAW):
    raw = pd.read_csv(path, encoding="latin1", keep_default_na=False)
    if len(raw.columns) != 14 or set(raw.columns) != set(RENAME):
        raise ValueError("CSV phải đúng 14 cột của SeoulBikeData.csv gốc.")
    df = raw.rename(columns=RENAME)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if not np.isfinite(df[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError("Cột số bị thiếu hoặc có giá trị vô hạn.")
    if (df[["visibility", "solar"]] < 0).any().any():
        raise ValueError("Tầm nhìn và bức xạ mặt trời không thể âm.")
    for c in ["humidity", "visibility"]:
        if (df[c] % 1 != 0).any():
            raise ValueError(f"{c} phải là số nguyên theo schema nguồn.")
    if not df["hour"].between(0, 23).all() or (df["hour"] % 1 != 0).any():
        raise ValueError("Giờ phải là số nguyên từ 0 đến 23.")
    for col, allowed in {"season": {"Winter", "Spring", "Summer", "Autumn"},
                         "holiday": {"Holiday", "No Holiday"}, "functioning": {"Yes", "No"}}.items():
        if not set(df[col]).issubset(allowed):
            raise ValueError(f"Giá trị phân loại không hợp lệ: {col}")
    df["timestamp"] = pd.to_datetime(df["date"], format="%d/%m/%Y") + pd.to_timedelta(df["hour"], unit="h")
    df = df.set_index("timestamp").sort_index()
    validate_hourly(df)
    return df

def feature_frame(df):
    """X_t chỉ dùng lịch ở t và quan sát <= t-1. Chưa loại các hàng đầu thiếu lag."""
    x = pd.DataFrame(index=df.index)
    for lag in LAGS:
        x[f"lag_{lag}"] = df["count"].shift(lag)
    for window in WINDOWS:
        x[f"mean_{window}"] = df["count"].shift(1).rolling(window, min_periods=window).mean()
    for c in WEATHER:
        x[f"{c}_lag1"] = df[c].shift(1)
    for label, values, period in [("hour", df.index.hour, 24), ("dow", df.index.dayofweek, 7),
                                  ("month", df.index.month, 12)]:
        x[f"{label}_sin"] = np.sin(2 * np.pi * values / period)
        x[f"{label}_cos"] = np.cos(2 * np.pi * values / period)
    x["weekend"] = (df.index.dayofweek >= 5).astype(float)
    return x[FEATURES].round(FEATURE_DECIMALS)

def next_features(history):
    """Dự báo giờ kế tiếp sau hàng cuối; yêu cầu ít nhất 168 giờ lịch sử."""
    validate_hourly(history)
    if len(history) < max(LAGS):
        raise ValueError("Cần ít nhất 168 giờ lịch sử liên tục.")
    timestamp = history.index[-1] + pd.Timedelta(hours=1)
    extended = history.reindex(history.index.append(pd.DatetimeIndex([timestamp])))
    extended.index.name = history.index.name
    x = feature_frame(extended).iloc[[-1]]
    if x.isna().any().any():
        raise ValueError("Không đủ dữ liệu để tạo đặc trưng.")
    return x

def prepare():
    for folder in ["data/processed", "models", "results"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    df = load_data()
    x = feature_frame(df)
    data = x.join(df[["count", "functioning"]]).dropna()
    data.to_csv(ROOT / "data/processed/features.csv", index_label="timestamp")
    df.to_csv(ROOT / "data/processed/clean.csv", index_label="timestamp")
    quality = {"raw_rows": len(df), "raw_columns": 14, "feature_count": len(FEATURES),
               "feature_rows": len(data), "initial_hours_removed": len(df)-len(data),
               "first": str(df.index.min()), "last": str(df.index.max()),
               "missing_cells": int(df.isna().sum().sum()), "duplicate_hours": int(df.index.duplicated().sum()),
               "missing_hours": 0, "zero_targets": int((df["count"] == 0).sum()),
               "closed_hours": int((df["functioning"] == "No").sum()),
               "zero_but_operating": int(((df["count"] == 0) & (df["functioning"] == "Yes")).sum()),
               "sha256": sha256(RAW)}
    save_json(ROOT / "results/data_quality.json", quality)
    return df, data

if __name__ == "__main__":
    prepare()
