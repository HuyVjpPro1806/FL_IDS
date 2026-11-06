# FL_IDS · CIC All-Attack Dirichlet (α = 0,5)

FL_IDS là dự án phát hiện xâm nhập nhị phân trên NF-CSE-CIC-IDS2018-v2. Dự án phân bổ dữ liệu cho ba client theo Dirichlet và đánh giá DNN, CNN ở các kịch bản Local, Centralized và Federated Learning. Chi tiết thiết kế và tiêu chí đánh giá nằm trong [PROTOCOL.md](PROTOCOL.md).

## Dữ liệu và cách chia

1. Đọc toàn bộ NF-CSE-CIC-IDS2018-v2 với 39 predictor; bỏ IP/port khỏi đầu vào.
2. Làm sạch toàn nguồn: nhóm input raw 39 feature, loại toàn nhóm nếu có nhãn binary 0 và 1; giữ dòng nguồn đầu tiên của mỗi input còn lại. Khác loại Attack nhưng đều Attack=1 được giữ trong metadata.
3. Giữ **toàn bộ Attack sạch**; lấy Benign không hoàn lại theo seed cho đủ **tối đa 1.500.000 mẫu**. Nếu Attack vượt trần thì báo lỗi yêu cầu tăng trần, không tự bỏ Attack.
4. Với CSV này: **544.863 Attack + 955.137 Benign = 1.500.000**, Attack khoảng **36,32%**. Đây là tỷ lệ do lấy hết Attack, khác tỷ lệ Attack tự nhiên của nguồn sạch là 6,07%.
5. Hash input và seed quyết định ranh giới fit/validation/test toàn cục, tỷ lệ mục tiêu 63/7/30. Do chia bằng hash, số mẫu thực tế dao động quanh tỷ lệ mục tiêu.
6. Rút Dirichlet đối xứng `Dir(0.5,0.5,0.5)` cho mỗi nhóm metadata Benign, BruteForce, Bot, DoS, DDoS, Infiltration, WebAttack. Dùng cùng bộ tỷ lệ cho cả ba split, làm tròn bằng largest remainder. Mỗi mẫu chỉ được giao một lần.
7. **Không áp quota số mẫu, tỷ lệ Attack hoặc sự hiện diện nhóm hiếm cho từng client.** Benign cũng là một nhóm khi chia; model vẫn chỉ học nhãn Benign/Attack.
8. Đọc lại raw CSV của các dòng đã chọn và fit Log → Z-score riêng từ fit của từng client; clip [-8,8], đóng băng scaler cho validation/test.

Toàn pool có đủ 14 loại Attack gốc. Một client hoặc một split có thể thiếu loại hiếm. Metadata giữ nhãn đại diện, mask các nhãn nguồn và cờ đa nghĩa; báo cáo theo Attack có phần support/recall không đa nghĩa. Không suy ra dữ liệu client là một tổ chức thật.

## Model, tuning và đánh giá

DNN + CNN, sigmoid, BCE thường, threshold 0,5. Client Adam; server FedAvg trực tiếp qua SGD lr=1, không momentum; không server Adam.

Tuning 10 trial/model, tổng 20 trial. Mỗi trial gồm 3 Local + Centralized + FedAvg theo số mẫu fit thực tế. Không gian tìm kiếm và cấu hình khởi điểm nằm trong `configs/tuning.json`. Chọn theo F1 Attack trung bình của 12 đánh giá validation (6 Local chéo, 3 Centralized, 3 FL). Nếu bằng điểm, ưu tiên F1 thấp nhất trong ba client FL rồi đến recall trung bình.

Sau tuning: 3 Local + Centralized + FedAvg weighted + FedAvg uniform. Weighted dùng số mẫu fit thực tế; uniform 1/3. Giữ chung tham số/khởi tạo/seed/data giữa hai FL. FL 15×3 và lấy round cuối; Local/Centralized tối đa 45 epoch, early stopping val_loss patience 5, restore epoch tốt nhất. Test chỉ đánh giá cuối.

## Chạy

Các lệnh Windows dùng WSL Ubuntu 22.04 và môi trường Python tại `/opt/flids-tf`. Phiên bản thư viện được ghi trong `requirements.txt`.

Đặt CSV nguồn tại `dataset/NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2.csv`. `configs/base.json` khai báo đường dẫn tương đối và SHA-256 của CSV. Thư mục `dataset/` và các kết quả sinh ra trong `runs/` được lưu cục bộ.

Mở `start_notebooks.cmd` từ thư mục dự án để chạy JupyterLab trên cổng **8893**, rồi chạy notebook **00 → 01 → 02 → 03 → 04**. Notebook 04 có bảng Local đánh giá chéo và FL trên cùng test-client. Có thể chạy từng bước từ dòng lệnh:

```bat
run.cmd prepare
run.cmd audit
run.cmd tune
run.cmd train
run.cmd results
```

Dữ liệu đã chuẩn bị được tái sử dụng khi identity và checksum hợp lệ. Nếu đã có kết quả huấn luyện, `run.cmd results` đọc các kết quả hiện có.

## Tệp chính

- `configs/base.json`: trần pool, chọn toàn Attack, Dirichlet α/seed và ranh giới split.
- `configs/tuning.json`: hai model, search space và lịch huấn luyện.
- `data_splits/pool_selection.json`: bằng chứng giữ toàn Attack, số Benign và hash tập Attack row ID.
- `data_splits/partition.json`, `split_counts.csv`: tỷ lệ rút và số mẫu thực tế.
- `runs/cic_all_attack_dirichlet_a05_max1p5m/prepared/`: feature, binary label, metadata và scaler mới.
- `.../tuning/` và `.../results/`: tham số đã chọn, nhật ký tuning và kết quả đánh giá.

Kiểm tra dữ liệu đã chuẩn bị: `run.cmd audit`. Thay alpha/seed/chính sách làm sạch sau khi có prepared cần tạo run_directory mới.

## CNN FedAdam theo số mẫu train — tuning server learning rate

Cell cuối notebook 03 chạy riêng CNN bằng `fedadam_search.py`, cấu hình `configs/fedadam_weighted.json`. Giữ cấu hình CNN đã chọn bằng validation và client Adam LR 0,0003; chỉ thử 20 server learning rate: 0,0001; 0,0002; 0,0003; 0,0005; 0,0007; 0,001; 0,0015; 0,002; 0,003; 0,004; 0,005; 0,007; 0,01; 0,015; 0,02; 0,03; 0,04; 0,05; 0,07; 0,1. Mức 0,01 được chạy trước. Mỗi trial khởi tạo mới cùng seed, học 15 vòng × 3 local epoch, tổng hợp NUM_EXAMPLES theo số mẫu fit. Server Adam beta1=0,9, beta2=0,999, epsilon=1e-7.

Chọn F1 Attack validation trung bình ba client tại vòng 15; hòa thì ưu tiên F1 client thấp nhất, recall trung bình, FAR trung bình thấp hơn, rồi trial sớm hơn. Việc tính điểm trung bình các client không thay đổi trọng số tổng hợp lúc train. Hoàn tất đủ 20 trial mới khởi tạo lại model, train với LR tốt nhất, rồi test một lần. Không dùng test để chọn LR, round hoặc threshold.

Tuning riêng: `runs/cic_all_attack_dirichlet_a05_max1p5m/tuning/fedadam_weighted/cnn/` chứa `trials.csv`, `best_params.json`, tiến độ và log từng trial. Test riêng: `.../results/binary/cnn/fedadam_weighted/`. Cell có thể tiếp tục từ các trial đã hoàn tất khi code/config/data identity khớp. Không chạy lại DNN, Local hoặc Centralized.

