# Dự báo nhu cầu thuê xe đạp Seoul trước một giờ

Đề tài 4, học phần Xử lý dữ liệu lớn. Mã nguồn thực hiện thu thập, EDA, làm sạch, tạo đặc trưng, mô hình hóa, đánh giá và demo bằng notebook. Spark tham gia vào kiểm tra dữ liệu, Window, Parquet phân vùng theo tháng, Spark SQL và ML Pipeline.

Báo cáo Word/PDF 16 trang theo bảy phần, 18 slide, notebook có đầu ra và tài liệu học cho bốn người được bàn giao trong gói ZIP riêng. Kho này chứa mã nguồn dạng văn bản. Dữ liệu, mô hình và kết quả được tái tạo tại máy khi chạy.

## Chạy demo chính

Dùng Python 3.12 và Java 17, khuyến nghị RAM từ 8 GB. Với Windows, nên chạy Spark trong WSL2. Không cần GPU.

```bash
git clone https://github.com/dovantohhieu/SeoulBike-BigData.git
cd SeoulBike-BigData
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
python -m pip check
java -version
python -m ipykernel install --user --name seoul-bike --display-name "Python 3.12 Seoul Bike"
python -m jupyter lab notebooks/Demo_BigData.ipynb
```

Chọn kernel **Python 3.12 Seoul Bike**, rồi chọn **Restart Kernel and Run All**. Giữ `RETRAIN_MODELS=True` và `MEASURE_BENCHMARK=True` trong lần chạy từ mã nguồn. Ô đầu tải CSV chính thức nếu chưa có và kiểm tra SHA-256 trước khi xử lý. Mỗi lần chạy lưu kết quả trong `runs/notebook_<thời điểm>/`.

Notebook gồm 28 ô, trong đó 16 ô code. Nó huấn luyện lại sklearn/Spark, chạy lab 8.760 / 100.000 / 1.000.000 hàng, đo 18 lượt benchmark và chạy 17 kiểm thử. MAE/RMSE được đối chiếu với bảng tham chiếu gồm bảy mô hình/baseline tại `tests/fixtures/reference_metrics.csv`. Đây là bảng số liệu nhỏ đã công bố trong báo cáo, không phải các bản ghi người dùng.

Gói ZIP đầy đủ đã có CSV, mô hình và các bảng benchmark, nên hỗ trợ chế độ đặt hai cờ thành `False` để trình bày nhanh. Trong notebook, đổi `MONTH` rồi chạy lại ô truy vấn để Spark thực sự đọc/lọc Parquet và xuất plan của tháng được chọn. Có thể đổi `TARGET` để dự báo từ lịch sử trước giờ cần dự báo.

## Chạy tự động và mở dashboard

```bash
python scripts/execute_notebook.py --output runs/check/Demo_BigData.ipynb
```

Nếu môi trường kiểm tra chặn socket kernel Jupyter, dùng thêm `--in-process`. Tùy chọn này thực thi bằng IPython kernel trong một tiến trình Python mới, không thay thế việc thử giao diện Jupyter trên máy trình bày.

Để tạo các tệp mô hình, bảng, hình và HTML trực tiếp ở thư mục dự án trước khi dùng dashboard:

```bash
python scripts/bootstrap.py --all
python -m streamlit run app.py
```

Lệnh bootstrap huấn luyện lại và đo lại. Các số đo thời gian/bộ nhớ phụ thuộc máy, vì vậy giữ riêng với bảng benchmark trong báo cáo. CI **Verify Seoul Bike notebook** chạy notebook từ mã nguồn trên Python 3.12/Java 17. CI chỉ kiểm tra thực thi, không xuất bản mô hình hoặc tài liệu.

## Nguồn và phương pháp

Dùng [UCI Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), DOI [10.24432/C5F62R](https://doi.org/10.24432/C5F62R), giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CSV có **8.760 dòng và 14 cột**, từ 01/12/2017 đến 30/11/2018. Nguồn, checksum và ghi công nằm trong `data/DATA_SOURCE.json` và `data/ATTRIBUTION.md`.

Chuỗi có 295 giờ nhãn 0 do ngừng hoạt động; các giờ này được giữ. Khi dự báo giờ t, đầu vào chỉ dùng quan sát đến t−1 và lịch đã biết của t. Có 21 đặc trưng, sau 168 giờ khởi tạo còn 8.592 mẫu. CV theo tháng 6/7/8, fit cuối trước tháng 10, test 1.464 giờ tháng 10–11, không xáo trộn.

HistGradientBoosting được chọn bằng MAE validation. Báo cáo ghi MAE backtest 72,073 và RMSE 118,343 lượt/giờ. MAE/RMSE dùng đủ 1.464 giờ; MAPE chỉ dùng 1.313 giờ dương và công bố coverage 89,686%. RF có test tốt hơn vẫn được công bố, còn lựa chọn mặc định giữ theo CV.

Phần ETL so sánh pandas/Spark trên cùng tác vụ, ba quy mô, hai engine và ba lần lặp. Đo cả thời gian materialize và RSS gồm Python/JVM. Các hàng lặp chỉ tăng tải ETL, không tham gia huấn luyện hoặc đánh giá dự báo. Spark chạy `local[2]` trên một máy; Window của chuỗi đơn có điểm nghẽn thứ tự toàn cục. Bài không suy ra hiệu năng cụm nhiều máy. Pilot ban đầu từng xem giai đoạn chồng lấn test, nên kết quả được công bố là backtest lịch sử.

Tài liệu đề tài: https://drive.google.com/file/d/1bHKkSoT1uwLvbT70im-9z-k-TtMjlNgu/view
