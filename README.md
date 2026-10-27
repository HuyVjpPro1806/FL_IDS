# FL_IDS — UNSW, BoT và CIC

Project chính phát hiện xâm nhập nhị phân trên ba client **UNSW, BoT và CIC**, so sánh **Local, Centralized và Federated Learning** bằng ba mô hình **DNN, LSTM và CNN**. Điểm chính của cách tiếp cận này là chuẩn hóa Log + Z-score riêng tại mỗi client trước khi học.

## Quy trình

`CSV → chọn 39 feature → Log → Z-score theo client → clip [-8,8] → tuning → huấn luyện → đánh giá`.

Mean/std được tính riêng cho mỗi feature của từng client, chỉ trên tập fit; validation/test dùng lại đúng scaler đã fit. FL dùng một bộ trọng số mô hình toàn cục, client/server Adam và tổng hợp đồng đều: UNSW = BoT = CIC = 1/3, không phụ thuộc số mẫu fit. Cả ba mô hình dùng chung pipeline và ngân sách. Chính sách tổng hợp được ghi rõ bằng `client_weighting: uniform` trong `configs/tuning.json` và áp dụng cho cả tuning/training cuối.

| Kịch bản | Ngân sách |
|---|---|
| FL | 15 round × 3 local epoch |
| Local | Tối đa 45 epoch |
| Centralized | Tối đa 45 epoch |

Local/Centralized dùng early stopping theo val_loss, patience 5 và khôi phục trọng số tốt nhất. Tuning 10 trial cho mỗi mô hình, chỉ chọn bằng validation. Trial đầu DNN: **8–4–2, dropout 0,5, batch 1024, client Adam LR 0,0005, server Adam LR 0,02**. Đây là cấu hình khởi đầu; các trial tiếp theo vẫn tìm trong search space. LSTM và CNN dùng cấu hình neo riêng trong configs/tuning.json. Ngưỡng phân loại 0,5; seed 42.

## Chạy notebook

Mở `start_notebooks.cmd` để dùng kernel Python WSL `/opt/flids-tf` và GPU, rồi chạy theo thứ tự:

1. `00_environment.ipynb`: kiểm tra môi trường, GPU và cấu hình.
2. `01_preprocess.ipynb`: chuẩn bị dữ liệu và kiểm tra scaler/split.
3. `02_tune_models.ipynb`: tuning DNN, LSTM, CNN trên validation.
4. `03_train_models.ipynb`: huấn luyện Local/Centralized/FL với tham số được chọn.
5. `04_results.ipynb`: xem accuracy, precision, recall, F1, FAR, AUC và tỷ lệ phát hiện theo loại tấn công.

Dữ liệu chuẩn bị và các fit hoàn tất được tái sử dụng khi checksum và protocol khớp. Kết quả lưu trong `runs/main/`. Thay ngân sách/search space trong `configs/tuning.json` trước khi bắt đầu study; study đã tồn tại cần run_directory mới.

## Dữ liệu

CIC có 1.000.000 mẫu thật, gồm 500.000 benign và 500.000 attack, chọn không hoàn lại với seed 42. Các client được undersampling cân bằng trước khi chia outer train/test. Validation được tách theo nhóm input trùng trong train; không loại các bản ghi trùng.

| Client | Fit | Validation | Test |
|---|---:|---:|---:|
| UNSW | 120.369 | 12.705 | 57.032 |
| BoT | 168.102 | 20.949 | 81.023 |
| CIC | 637.019 | 62.981 | 300.000 |

`dataset/` chứa CSV nguồn; `data_splits/` lưu row ID, nhãn và metadata phân chia cố định. Các thống kê chuẩn hóa được lưu tại `runs/main/prepared/<client>/scaler.json`. Dùng `feature_preprocessing.transform_raw` để chuẩn hóa đầu vào raw khi inference, theo đúng thứ tự 39 feature.

## Tổ chức code

- `models.py`: DNN, LSTM, CNN và Adam.
- `notebook_workflow.py`: tuning, training và evaluation dùng chung.
- `prepare.py`, `feature_preprocessing.py`: đọc CSV, chia tập cố định và chuẩn hóa.
- `metrics.py`: chỉ số đánh giá nhị phân.
- `runtime.py`, `process_runner.py`: môi trường GPU/TFF và quản lý tiến trình.
- `configs/tuning.json`: ngân sách, tham số neo và search space cho ba mô hình.
- `configs/base.json`: nguồn dữ liệu và cấu hình tham khảo.
- `PROTOCOL.md`: mô tả giao thức đầy đủ.

## Chạy bằng CLI

Trong WSL: `bash launch.sh prepare`, `bash launch.sh doctor`, `bash launch.sh tune --model dnn`, `bash launch.sh train --model dnn`, `bash launch.sh results`. Thay dnn bằng lstm hoặc cnn khi cần. Trên Windows có thể dùng các lệnh tương ứng qua `run.cmd`.
