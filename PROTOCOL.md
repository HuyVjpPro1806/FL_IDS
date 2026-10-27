# Giao thức FL IDS Log + Z-score

## Đầu vào và chuẩn hóa

Ba client UNSW/BoT/CIC có cùng 39 feature, cùng thứ tự cột, nhãn nhị phân benign/attack. Đọc giá trị raw float64 từ CSV. Với mỗi feature, dùng signed log `sign(x)*ln(1+abs(x))`; với giá trị không âm là `ln(1+x)`. Tính mean và population std (ddof=0) trên fit của từng client, sau đó Z-score, clip [-8,8] và chuyển float32. Std <1e-12 dùng scale=1. Validation/test không tham gia fit scaler.

Undersampling cân bằng trước outer split 70/30. Validation là fold đầu của StratifiedGroupKFold(10, shuffle=True, seed=42) trong outer train, nhóm theo hash của 39 raw feature. Giữ nguyên duplicate; nhóm input fit/validation không giao nhau. Row ID nguồn được cố định trong data_splits. Các split này từng được dùng để đánh giá trước đây; đây không phải tập test mới chưa từng được xem.

## Tuning và training

Trial đầu DNN: **8–4–2, dropout 0,5, batch 1024, client Adam LR 0,0005, server Adam LR 0,02**. Đây là cấu hình khởi đầu; các trial tiếp theo vẫn tìm trong search space. LSTM và CNN dùng cấu hình neo riêng trong configs/tuning.json.

DNN/LSTM/CNN dùng chung workflow, mỗi model có 10 trial: một trial neo, bốn trial random và năm trial TPE. Mỗi trial chạy ba Local, một Centralized và một FL. Điểm chọn là F1 trung bình đều của 12 đánh giá validation: 6 Local chéo client, 3 Centralized và 3 FL. Tie-break bằng F1 FL client thấp nhất, rồi recall trung bình.

FL chạy 15 round, 3 local epoch mỗi round, mọi client tham gia. Client/server dùng Adam; tổng hợp đồng đều, UNSW = BoT = CIC = 1/3 cho mọi trial và model. Dùng trọng số round cuối. Chỉ có một model toàn cục và không fine-tune riêng sau FL.

Local/Centralized tối đa 45 epoch, early stopping val_loss patience 5, restore minimum-val_loss weights; không refit trên toàn bộ outer train. Local theo dõi validation client nguồn; Centralized theo dõi validation gộp. Mỗi mẫu được chuẩn hóa bằng scaler của client nguồn trong mọi kịch bản.

Test chỉ dùng sau khi chọn tham số bằng validation. Ngưỡng 0,5; seed 42. Báo cáo tất cả ba client bằng accuracy, precision, recall, F1, FAR, AUC và recall theo loại tấn công. Kết quả CSV lưu trong runs/main/results/.

## Trọng số client được dùng ở đâu?

Trong mỗi round FL:

1. Server gửi cùng model toàn cục w_t cho ba client.
2. Mỗi client học 3 local epoch trên dữ liệu của mình bằng client Adam, thu được model w_i.
3. Client gửi cập nhật model về server. Dữ liệu raw không được gửi để tổng hợp.
4. Server lấy trung bình đồng đều các model delta: g = [(w_t - w_UNSW) + (w_t - w_BoT) + (w_t - w_CIC)] / 3.
5. Server Adam dùng g và trạng thái momentum/variance đang giữ để cập nhật model toàn cục, rồi bắt đầu round tiếp theo. Không đơn thuần đặt global model bằng trung bình ba model local, vì server optimizer là Adam.

Đây là hệ số dùng khi gộp cập nhật client, khác với các tham số của mạng neural mà model học. Trọng số client cố định 1/3 ở mọi round, không được học hay điều chỉnh bằng validation. Mỗi client vẫn train trên toàn bộ tập fit riêng nên số batch và số bước local có thể khác nhau.

Cấu hình thực chạy nằm trong configs/tuning.json (`client_weighting: uniform`) và notebook_workflow.make_process (`ClientWeighting.UNIFORM`). Các giá trị client_weighting trong configs/base.json chỉ mô tả cấu hình tham khảo.

## Kiến trúc

DNN dùng Dense/ReLU; LSTM dùng GPU CuDNNLSTM tanh/sigmoid với input reshape (1,39); CNN dùng trục 39 feature, không phải chuỗi thời gian. Cả ba được định nghĩa trong models.py và dùng cấu hình trong configs/tuning.json.

Các tham số trong configs/base.json là thông tin tham khảo, không điều khiển ngân sách training của workflow. Tài liệu nguồn nằm trong reference/.
