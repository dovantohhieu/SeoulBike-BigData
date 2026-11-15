# Trình diễn bằng notebook

Mở `Demo_BigData.ipynb` sau khi cài Python 3.12, Java 17 và `requirements-notebook.txt`. Notebook đã có đầu ra để đọc; chọn **Restart Kernel and Run All** để kiểm tra từ đầu.

```bash
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name seoul-bike --display-name "Python 3.12 Seoul Bike"
python -m jupyter lab notebooks/Demo_BigData.ipynb
```

Chọn kernel vừa tạo. Môi trường Windows nên chạy Spark trong WSL2 hoặc dùng môi trường Linux phù hợp. Chưa có lần kiểm thử trực tiếp trên máy Windows hoặc tài khoản Colab của nhóm.

## Chạy đầy đủ và trình bày nhanh

Ở chế độ mặc định, `RETRAIN_MODELS=True` và `MEASURE_BENCHMARK=True`: huấn luyện lại sklearn/Spark và đo ba quy mô, hai engine, ba lần lặp. Mỗi lần chạy ghi vào một thư mục `runs/notebook_<thời điểm>` riêng.

Khi trình bày ngắn, đặt hai cờ thành `False` và `LAB_SIZES=[8760]` ở ô đầu, sau đó chạy lần lượt. Các ô ETL vẫn chạy Spark thật. Mô hình và benchmark khi đó lấy từ bản đính kèm; notebook ghi rõ việc này. Có thể dùng chế độ đầy đủ để thầy xem quá trình huấn luyện hoặc đo lại.

## Chia phần cho bốn người

| Người | Phần notebook | Điều cần giải thích |
|---|---|---|
| 1 | Mở đầu và mục 1–2 | Mục tiêu, nguồn UCI, 14 cột, chu kỳ và ý nghĩa nhãn |
| 2 | Mục 3–4 | Kiểm tra lưới giờ, giữ số 0 hợp lệ, 21 đặc trưng và không dùng tương lai |
| 3 | Mục 5–6 | Ba fold theo thời gian, lựa chọn bằng CV, MAE/RMSE và MAPE dương |
| 4 | Mục 7 | Ba quy mô ETL, Parquet/SQL/plan, 18 lượt benchmark và ô thử dự báo |

## Thao tác nên cho thầy xem

1. Chạy ô nguồn: SHA-256 khớp, 8.760 hàng và 14 cột.
2. Chạy ô làm sạch: 295 giờ nhãn 0 được giữ; thử bỏ một giờ thì chương trình từ chối.
3. Chạy ô đặc trưng: đối chiếu lag/rolling của 08:00 với lịch sử đến 07:00.
4. Chạy phần Spark ETL trên 8.760 hàng. Cho xem 12 tháng, 288 nhóm SQL, CSV tháng 10 và `PartitionFilters`/`ReadSchema`.
5. Đổi `MONTH`, chạy lại ô để Spark đọc Parquet, tổng hợp và xuất plan đúng tháng vừa chọn. `VIEW_ROWS` mặc định lấy quy mô đầu trong `LAB_SIZES`. Với dữ liệu thật, tháng 10 có 744 giờ.
6. Ở ô dự báo, dùng `TARGET='2018-10-01 08:00'` và HGB: dự báo khoảng 1.783,01, thực tế 2.101. Thử `2018-10-03 00:00` để giải thích sai số lớn lúc mở lại dịch vụ.
7. Cho xem bảng benchmark, nói rõ đó là số đo mới hay bảng đính kèm. Cả hai đều tách dữ liệu lặp khỏi huấn luyện.

## Khi một ô lỗi

- Sai kernel hoặc thiếu thư viện: chọn Python 3.12, cài đúng file yêu cầu rồi restart kernel.
- Thiếu Java: kiểm tra `java -version`, `JAVA_HOME` và PATH; dùng Java 17.
- CSV không khớp SHA hoặc sai schema: dừng để kiểm tra nguồn, không bỏ kiểm tra cho chạy tiếp.
- Chạy lại tạo thư mục `runs/` mới: đây là chủ ý để giữ nguyên số liệu của bản báo cáo. Có thể xóa các thư mục thử do mình tạo sau khi đã lưu kết quả cần giữ.
- Bản HTML chỉ hiển thị kết quả lưu. Dashboard là phương án thao tác phụ trợ, không thay thế notebook chính.
