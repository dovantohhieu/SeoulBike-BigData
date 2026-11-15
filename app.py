"""Dashboard cục bộ: chạy python -m streamlit run app.py từ thư mục dự án."""
import json
import importlib.util
import shutil
import os
import subprocess
import sys
import pandas as pd
import streamlit as st
from src.common import ROOT, TEST_START
from src.data import load_data
from src.predict import metadata, load_model, forecast

st.set_page_config(page_title="Seoul Bike · Dự báo theo giờ",page_icon="🚲",layout="wide")
st.title("Dự báo nhu cầu xe đạp Seoul")
st.caption("Đề tài 4 · Xử lý dữ liệu lớn · Dự báo trước 1 giờ")

@st.cache_data
def read_assets():
    df=load_data()
    predictions=pd.read_csv(ROOT/"results/predictions.csv",parse_dates=["timestamp"]).set_index("timestamp")
    scores=pd.read_csv(ROOT/"results/test_metrics.csv")
    cv=pd.read_csv(ROOT/"results/cv_summary.csv")
    bench=pd.read_csv(ROOT/"results/benchmark_summary.csv")
    return df,predictions,scores,cv,bench

@st.cache_resource
def get_model(name):
    return load_model(name)

try:
    df,predictions,scores,cv,bench=read_assets()
    reg=metadata()
except (FileNotFoundError,ValueError) as e:
    st.error(f"Chưa đủ kết quả. Chạy python run_project.py --all trước. Chi tiết: {e}")
    st.stop()

page=st.sidebar.radio("Nội dung",["Tổng quan dữ liệu","Xử lý Big Data trực tiếp","Thử dự báo","Đánh giá mô hình","Spark và hiệu năng","Hướng dẫn demo"])
st.sidebar.write("**Mô hình được chọn**")
st.sidebar.write(reg["selected_model"])
st.sidebar.caption("Chọn bằng MAE trung bình của 3 tập validation theo thời gian.")

if page == "Tổng quan dữ liệu":
    a,b,c,d=st.columns(4)
    a.metric("Giờ quan sát",f"{len(df):,}")
    b.metric("Giờ ngừng hoạt động",int((df.functioning=="No").sum()))
    c.metric("Đặc trưng đầu vào",21)
    d.metric("Chân trời dự báo","1 giờ")
    st.write("Dữ liệu lịch sử từ 01/12/2017 đến 30/11/2018. Giờ theo thời gian địa phương Seoul, không chuyển sang giờ Việt Nam.")
    st.subheader("Lượt thuê trung bình theo ngày")
    st.line_chart(df["count"].resample("D").mean(),y_label="Lượt/giờ")
    left,right=st.columns(2)
    with left:
        st.subheader("Theo giờ trong ngày")
        st.bar_chart(df.groupby(df.index.hour)["count"].mean(),y_label="Lượt/giờ")
    with right:
        st.subheader("Theo tháng")
        st.bar_chart(df.groupby(df.index.to_period("M").astype(str))["count"].mean(),y_label="Lượt/giờ")
    st.dataframe(df.reset_index().head(48),hide_index=True,width="stretch")
    st.caption("Nguồn: UCI Seoul Bike Sharing Demand, DOI 10.24432/C5F62R, CC BY 4.0. Nhân bản dữ liệu chỉ dành cho phép đo ETL.")

elif page == "Xử lý Big Data trực tiếp":
    st.subheader("CSV → kiểm tra chất lượng → Parquet theo tháng → Spark SQL")
    st.write("Bấm chạy để thực thi Spark trên máy. Có thể chọn dữ liệu thật hoặc tăng tải bằng cách lặp dữ liệu. Tập lặp nằm riêng và không được đưa vào huấn luyện hay đánh giá dự báo.")
    size=st.selectbox("Số hàng xử lý",[8760,100000,1000000],format_func=lambda n:f"{n:,} hàng"+(" · dữ liệu thật" if n==8760 else " · nhân bản thử tải"))
    ready=bool(importlib.util.find_spec("pyspark") and shutil.which("java"))
    if not ready:
        st.info("Đang xem kết quả đã chạy. Để chạy mới, cài requirements.txt và Java 17 theo README, hoặc mở notebook trên Colab.")
    if st.button("Chạy Spark ETL",disabled=not ready,type="primary"):
        with st.spinner("Spark đang đọc CSV, xác thực, ghi Parquet và chạy SQL…"):
            try:
                process=subprocess.run([sys.executable,"-m","src.bigdata_lab","--rows",str(size)],cwd=ROOT,
                    capture_output=True,text=True,timeout=240,
                    env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"})
                if process.returncode:
                    st.error("Spark chưa hoàn thành. Xem log bên dưới; kiểm tra Java 17 và môi trường.")
                else:
                    st.success("Đã chạy Spark thật và đối chiếu kết quả SQL với pandas.")
                with st.expander("Nhật ký lần chạy vừa rồi",expanded=bool(process.returncode)):
                    st.code((process.stdout+process.stderr)[-16000:],language="text")
            except subprocess.TimeoutExpired:
                st.error("Lần chạy vượt 240 giây. Dùng quy mô 8.760 hàng hoặc chạy lệnh trong terminal để kiểm tra.")
    folder=ROOT/f"results/lab/{size}"
    if (folder/"summary.json").exists():
        info=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        st.caption("Kết quả lần chạy thành công được lưu lúc "+info["utc"]+". Bảng chỉ cập nhật khi lần chạy mới thành công.")
        a,b,c,d=st.columns(4)
        a.metric("Số hàng xử lý",f"{info['processed_rows']:,}")
        b.metric("Phân vùng theo tháng",info["month_partitions"])
        c.metric("Nhóm tháng × giờ",info["aggregate_groups"])
        d.metric("SQL khớp pandas","Đạt" if info["aggregates_match_pandas"] else "Chưa đạt")
        with st.expander("1. Chất lượng dữ liệu nguồn và schema",expanded=True):
            st.json(info["quality"])
            st.code(info["schema"],language="text")
        with st.expander("2. Parquet có phân vùng"):
            st.write(f"{info['parquet_files']} tệp Parquet, {info['parquet_bytes']/1024**2:.2f} MiB. Đường dẫn: `{info['parquet_path']}`.")
            st.write(info["partition_months"])
            st.caption("Phân vùng thư mục theo tháng hỗ trợ bỏ qua các tháng không cần đọc. Nó khác với số task/partition khi Spark đang tính toán.")
        st.subheader("3. Thành quả tổng hợp bằng Spark SQL")
        result=pd.read_csv(folder/"gold_month_hour.csv")
        month=st.selectbox("Tháng xem kết quả",sorted(result.observation_month.unique()))
        part=result.loc[result.observation_month.eq(month)].set_index("hour")
        st.bar_chart(part[["total"]],y_label="Tổng lượt thuê" if size==8760 else "Tổng trên dữ liệu lặp (chỉ thử tải)")
        st.dataframe(part.reset_index().round(3),hide_index=True,width="stretch")
        st.download_button("Tải kết quả Spark SQL",(folder/"gold_month_hour.csv").read_bytes(),f"spark_sql_{size}.csv","text/csv")
        with st.expander("4. SQL và kế hoạch thực thi"):
            st.code(info["sql"],language="sql")
            st.write(f"Lọc tháng {info['query_month']} trả {info['query_rows']:,} hàng. Tháng này có {info['query_partition_files']}/{info['parquet_files']} tệp Parquet. Plan có PartitionFilters.")
            st.caption("Số tệp trên là đếm từ thư mục phân vùng. Đây không phải số byte I/O đo thực tế.")
            for name,label in [("plan_partition_filter.txt","Đọc cột cần thiết và lọc phân vùng"),("plan_aggregation.txt","GroupBy và Exchange"),("plan_expand.txt","Range và broadcast bảng nguồn nhỏ")]:
                st.write(label)
                st.code((folder/name).read_text(encoding="utf-8"),language="text")
        st.caption("Các thời gian của màn hình này là một lượt demo có kiểm tra, không dùng thay bảng benchmark 3 lần lặp. local[2] chạy trên một máy.")
    else:
        st.info("Quy mô này chưa có kết quả lưu. Bấm Chạy Spark ETL sau khi cài môi trường đầy đủ.")

elif page == "Thử dự báo":
    mode=st.radio("Nguồn lịch sử",["Phát lại một giờ trong tập kiểm tra","Giờ kế tiếp sau bộ dữ liệu","CSV lịch sử của bạn"])
    available=[n for n in reg["model_names"] if n != "SparkRandomForest" or (importlib.util.find_spec("pyspark") and shutil.which("java"))]
    default=reg["selected_model"] if reg["selected_model"] in available else available[0]
    name=st.selectbox("Mô hình",available,index=available.index(default))
    if "SparkRandomForest" not in available:
        st.caption("Để dự báo trực tiếp bằng SparkRandomForest, cài requirements.txt và Java 17. Các mô hình sklearn vẫn hoạt động.")
    history=None
    actual=None
    if mode == "Phát lại một giờ trong tập kiểm tra":
        day=st.date_input("Ngày cần dự báo",value=pd.Timestamp(TEST_START).date(),min_value=predictions.index.min().date(),max_value=predictions.index.max().date())
        hour=st.slider("Giờ cần dự báo",0,23,8)
        target=pd.Timestamp(day)+pd.Timedelta(hours=hour)
        history=df.loc[df.index<target]
        actual=float(df.loc[target,"count"])
        st.info("Mỗi lần dự báo chỉ sử dụng số liệu đến giờ trước đó. Giá trị thực tế chỉ xuất hiện sau khi mô hình đã tính xong.")
    elif mode == "Giờ kế tiếp sau bộ dữ liệu":
        history=df
        st.info("Dự báo 01/12/2018 00:00 từ lịch sử đến 30/11/2018 23:00. Bộ dữ liệu không có nhãn thực tế cho giờ này.")
    else:
        upload=st.file_uploader("CSV đúng cấu trúc SeoulBikeData.csv, tối thiểu 168 giờ liên tục",type=["csv"])
        if upload:
            try: history=load_data(upload)
            except (ValueError,UnicodeError) as e: st.error(str(e))
        st.caption("Cần cột Date dạng ngày/tháng/năm và đủ 14 cột như dữ liệu mẫu. Không cần biết thời tiết của giờ tương lai.")
    if history is not None and st.button("Tính dự báo",type="primary"):
        try:
            ts,value,x=forecast(history,get_model(name),name)
            st.success(f"{ts:%d/%m/%Y %H:%M} · Dự báo {value:,.1f} lượt thuê")
            a,b,c=st.columns(3)
            a.metric("Dự báo",f"{value:,.1f}")
            b.metric("Giờ trước",f"{history.iloc[-1]['count']:,.0f}")
            if actual is not None: c.metric("Thực tế",f"{actual:,.0f}",delta=f"Sai số tuyệt đối {abs(actual-value):.1f}",delta_color="off")
            st.line_chart(history["count"].tail(168),y_label="Lượt thuê")
            with st.expander("21 đặc trưng dùng trong dự báo"):
                st.dataframe(x.T.rename(columns={ts:"Giá trị"}))
            st.caption("Kết quả kỳ vọng có thể là số thập phân. Mô hình không biết trước giờ đóng dịch vụ nếu thông tin này chưa có trong lịch sử.")
        except (ValueError,RuntimeError,ModuleNotFoundError) as e:
            st.error(str(e))

elif page == "Đánh giá mô hình":
    st.write("Tập kiểm tra: 01/10–30/11/2018. Mô hình giữ nguyên trọng số, lịch sử quan sát cập nhật sau mỗi giờ.")
    st.dataframe(scores.round(3),hide_index=True,width="stretch")
    st.warning("MAPE_positive_pct chỉ tính trên giờ có nhãn > 0; cột MAPE_n và coverage ghi số giờ được tính. MAE và RMSE gồm cả giờ bằng 0.")
    st.subheader("MAE validation theo thời gian")
    st.dataframe(cv.round(3),hide_index=True,width="stretch")
    chosen=st.multiselect("Đường dự báo",reg["model_names"]+["Naive_1h"],default=[reg["selected_model"],"Naive_1h"])
    dates=st.date_input("Khoảng xem",value=(predictions.index.min().date(),(predictions.index.min()+pd.Timedelta(days=6)).date()),min_value=predictions.index.min().date(),max_value=predictions.index.max().date())
    if isinstance(dates,(tuple,list)) and len(dates)==2:
        segment=predictions.loc[str(dates[0]):str(dates[1])]
        st.line_chart(segment[["count"]+chosen],y_label="Lượt thuê")
    st.download_button("Tải toàn bộ dự báo CSV",(ROOT/"results/predictions.csv").read_bytes(),"predictions.csv","text/csv")
    st.caption("Đây là backtest lịch sử có thể tái lập. Một pilot trước đó từng xem giai đoạn chồng lấn, nên không gọi đây là kiểm định độc lập chưa từng được xem.")

elif page == "Spark và hiệu năng":
    st.write("Cùng CSV, cùng kiểu dữ liệu, cùng lọc hợp lệ và groupBy giờ. Mỗi cấu hình chạy 3 lần trong tiến trình mới. Spark chạy local[2] trên một máy.")
    st.dataframe(bench.round(3),hide_index=True,width="stretch")
    st.bar_chart(bench.pivot(index="size",columns="engine",values="etl_median_s"),y_label="Giây ETL, trung vị")
    st.bar_chart(bench.pivot(index="size",columns="engine",values="peak_rss_median_mib"),y_label="MiB RSS đỉnh, trung vị")
    st.caption("etl_s gồm đọc CSV và thực thi đến kết quả 24 nhóm, không gồm khởi động Spark. wall_s gồm khởi động và kết thúc. RSS lấy mẫu cả Python và JVM; có thể đếm trùng trang nhớ chia sẻ. Không xóa cache hệ điều hành.")
    st.write("Dữ liệu 100 nghìn và 1 triệu dòng là bản lặp có row_id riêng để đo tải. Huấn luyện và đánh giá chỉ dùng 8.760 quan sát gốc.")

else:
    st.markdown("""1. Mở **Tổng quan dữ liệu**: giới thiệu mục tiêu và 295 giờ đóng dịch vụ.
2. Mở **Xử lý Big Data trực tiếp**, chọn 8.760 hàng và bấm **Chạy Spark ETL**. Cho xem kiểm tra chất lượng, 12 phân vùng Parquet, biểu đồ SQL và PartitionFilters.
3. Có thể đổi sang 1 triệu hàng để minh họa tăng tải; bảng này là dữ liệu nhân bản.
4. Mở **Thử dự báo**: chọn một giờ kiểm tra, bấm **Tính dự báo** và so với thực tế.
5. Mở **Đánh giá mô hình**: giải thích CV theo thời gian, baseline và MAPE khi có số 0.
6. Mở **Spark và hiệu năng**: trình bày số đo thật và giới hạn của Spark chạy một máy.

Để chạy lại toàn bộ: `python run_project.py --all`.

Nếu chưa cài Python, mở `DEMO_OFFLINE.html` trong trình duyệt để xem kết quả đã lưu. Bản offline không huấn luyện hoặc suy luận mô hình mới.""")
