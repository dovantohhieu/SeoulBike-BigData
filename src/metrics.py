import numpy as np

def regression_metrics(y, prediction):
    y, p = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nhãn và dự báo phải hữu hạn, không rỗng và cùng kích thước.")
    e = y-p
    positive = y > 0
    return {"n": len(y), "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAPE_positive_pct": float(np.mean(np.abs(e[positive]/y[positive]))*100) if positive.any() else None,
            "MAPE_n": int(positive.sum()), "MAPE_coverage_pct": float(positive.mean()*100),
            "WAPE_pct": float(np.abs(e).sum()/np.abs(y).sum()*100) if np.abs(y).sum() else None}
