"""Kết hợp kết quả đã chạy; mọi bảng/biểu đồ dùng chung các CSV gốc."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT, FEATURES, FEATURE_DECIMALS, save_json
from .data import load_data
from .metrics import regression_metrics

def run():
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    cv=pd.concat([pd.read_csv(ROOT/f"results/cv_{k}.csv") for k in ["sklearn","spark"]],ignore_index=True)
    cv.to_csv(ROOT/"results/cv_all.csv",index=False)
    summary=cv.groupby("model").agg(MAE_cv=("MAE","mean"),RMSE_cv=("RMSE","mean"),
        MAE_std=("MAE","std"),folds=("fold","count")).sort_values("MAE_cv").reset_index()
    summary.to_csv(ROOT/"results/cv_summary.csv",index=False)
    selected=summary.iloc[0]["model"]
    names=["Ridge","RandomForest","HistGradientBoosting","SparkRandomForest"]
    # Chọn model bằng validation, không thay sang model tốt nhất trên test.
    save_json(ROOT/"models/registry.json",{"selected_model":selected,"model_names":names,
        "features":FEATURES,"feature_decimal_places":FEATURE_DECIMALS,"horizon_hours":1,"selection":"minimum mean MAE over three chronological validation folds",
        "fit_end_exclusive":"2018-10-01","requires_java_for_spark":True})
    scores=pd.concat([pd.read_csv(ROOT/f"results/test_{k}.csv") for k in ["sklearn","spark"]],ignore_index=True)
    scores["selected_by_cv"]=scores.model.eq(selected)
    scores.to_csv(ROOT/"results/test_metrics.csv",index=False)
    pred=pd.read_csv(ROOT/"results/predictions_sklearn.csv",parse_dates=["timestamp"]).set_index("timestamp")
    sp=pd.read_csv(ROOT/"results/predictions_spark.csv",parse_dates=["timestamp"]).set_index("timestamp")
    assert pred.index.equals(sp.index)
    np.testing.assert_array_equal(pred["count"],sp["count"])
    pred["SparkRandomForest"]=sp["prediction"]
    pred.to_csv(ROOT/"results/predictions.csv",index_label="timestamp")
    groups=[]
    for label,mask in [("all",np.ones(len(pred),bool)),("operating",pred.functioning.eq("Yes")),
                       ("closed",pred.functioning.eq("No")),("peak_07_09_17_19",pred.index.hour.isin([7,8,9,17,18,19]))]:
        sub=pred.loc[mask]
        groups.append({"group":label,**regression_metrics(sub["count"],sub[selected])})
    pd.DataFrame(groups).to_csv(ROOT/"results/selected_error_groups.csv",index=False)
    err=pd.DataFrame({"hour":pred.index.hour,"abs_error":abs(pred["count"]-pred[selected])}).groupby("hour").abs_error.mean()
    err.to_csv(ROOT/"results/error_by_hour.csv")
    worst=pred.assign(abs_error=abs(pred["count"]-pred[selected])).sort_values("abs_error",ascending=False).head(20)
    worst.to_csv(ROOT/"results/largest_errors.csv")
    selected_metrics=scores.set_index("model").loc[selected].to_dict()
    baseline=scores.set_index("model").loc["Naive_1h"]
    save_json(ROOT/"results/conclusions.json",{"selected_model":selected,"selected_metrics":selected_metrics,
        "MAE_reduction_vs_naive_1h_pct":float((1-selected_metrics["MAE"]/baseline["MAE"])*100),
        "best_test_model_descriptive_only":scores.sort_values("MAE").iloc[0]["model"],
        "warning":"Model selection uses CV, even if another model has lower test MAE."})
    df=load_data()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold","figure.dpi":150})
    teal="#007f85"; orange="#dd6e42"; navy="#193449"
    def save(name):
        plt.tight_layout();plt.savefig(ROOT/f"figures/{name}.png",bbox_inches="tight");plt.close()
    plt.figure(figsize=(9,3.5));df["count"].resample("D").mean().plot(color=teal)
    plt.ylabel("Lượt thuê trung bình mỗi giờ");plt.xlabel("Ngày");plt.title("Nhu cầu thay đổi theo mùa trong một năm")
    save("daily_demand")
    plt.figure(figsize=(9,3.4))
    for weekend,label,col in [(False,"Ngày trong tuần",teal),(True,"Cuối tuần",orange)]:
        s=df.loc[(df.index.dayofweek>=5)==weekend]
        s.groupby(s.index.hour)["count"].mean().plot(label=label,color=col,marker="o",markersize=3)
    plt.xlabel("Giờ Seoul");plt.ylabel("Lượt thuê trung bình");plt.legend();plt.xticks(range(0,24,2));save("hourly_pattern")
    plt.figure(figsize=(9,3.4));plt.hist(df["count"],bins=45,color=teal,edgecolor="white")
    plt.xlabel("Lượt thuê trong một giờ");plt.ylabel("Số giờ");plt.title("Phân phối biến mục tiêu");save("target_distribution")
    pivot=cv.pivot(index="validation_start",columns="model",values="MAE")
    ax=pivot.plot.bar(figsize=(9,3.7),color=[teal,orange,navy,"#91a5aa"]);ax.set_ylabel("MAE (lượt/giờ)");ax.set_xlabel("Tháng validation");plt.xticks(rotation=0)
    plt.legend(fontsize=8,ncol=2);save("cv_scores")
    ordered=scores.sort_values("MAE",ascending=False)
    plt.figure(figsize=(9,3.6));plt.barh(ordered.model,ordered.MAE,color=[orange if m==selected else teal for m in ordered.model]);plt.xlabel("MAE (lượt/giờ)");save("test_mae")
    seg=pred.iloc[:168]
    ax=seg[["count",selected,"Naive_1h"]].rename(columns={"count":"Thực tế","Naive_1h":"Giờ trước"}).plot(figsize=(9,3.8),color=[navy,teal,orange],alpha=.9)
    ax.set_ylabel("Lượt thuê");ax.set_xlabel("Giờ Seoul, 7 ngày đầu tập kiểm tra");plt.legend(fontsize=8);save("test_week")
    plt.figure(figsize=(9,3.2));err.plot.bar(color=teal);plt.ylabel("MAE (lượt/giờ)");plt.xlabel("Giờ cần dự báo");plt.xticks(rotation=0);save("error_by_hour")
    bench=pd.read_csv(ROOT/"results/benchmark_summary.csv")
    for metric,label,name in [("etl_median_s","Giây ETL (trục log)","benchmark_time"),("peak_rss_median_mib","MiB RSS đỉnh (trục log)","benchmark_memory")]:
        ax=bench.pivot(index="size",columns="engine",values=metric).plot.bar(figsize=(9,3.4),color=[teal,orange],logy=True)
        ax.set_ylabel(label);ax.set_xlabel("Số dòng, dữ liệu nhân bản");plt.xticks(rotation=0);save(name)
    hourly=df.groupby(df.index.hour)["count"].mean()
    daily=df["count"].resample("D").mean()
    save_json(ROOT/"results/presentation_data.json",{"scores":scores.to_dict("records"),"cv":summary.to_dict("records"),
        "cv_folds":cv.to_dict("records"),"bench":bench.to_dict("records"),"groups":groups,
        "selected":selected,"hourly":{"labels":[str(x) for x in hourly.index],"values":hourly.tolist()},
        "daily":{"labels":[str(x.date()) for x in daily.index],"values":daily.tolist()},
        "week":{"labels":[str(x) for x in seg.index],"actual":seg["count"].tolist(),"model":seg[selected].tolist(),"naive":seg["Naive_1h"].tolist()}})
    from .offline import build_offline
    build_offline(pred,scores,bench,selected)
    print("Selected by CV:",selected,flush=True)

if __name__=="__main__":run()
