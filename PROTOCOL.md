# Protocol — all clean Attack, max 1.5m, Dirichlet 0.5

Pool: toàn bộ binary-positive input sau làm sạch toàn nguồn; Benign lấy không hoàn lại để lấp phần còn lại của trần 1.500.000. Kiểm tra số Attack và SHA-256 của tập row ID Attack trước/sau phân phối. CSV nguồn không bị ghi đè.

Input: 39 feature raw float64; hash input uint64 bằng pandas. Loại binary-conflict group, giữ row ID nguồn nhỏ nhất cho mỗi nhóm còn lại. Hash equality là tiêu chí nhóm; collision có thể gộp nhóm. Nhãn Attack khác nhau nhưng cùng binary=1 không phải conflict; giữ mask nhãn nguồn và cờ đa nghĩa.

Chia global fit/validation/test bằng SplitMix64(seed=42), tỷ lệ mục tiêu 63/7/30. Tất cả input giống nhau có cùng ranh giới global; audit cuối yêu cầu không còn input trùng trong hay giữa 9 phân hoạch. Pool có thể giao nhiều split của thí nghiệm khác, nhưng cùng hash rule giữ global train/validation/test boundary khi seed/quy tắc không đổi.

Dirichlet trên 7 nhóm metadata gồm Benign và 6 nhóm Attack. Một vector ba phần tử với alpha=(0.5,0.5,0.5) cho mỗi nhóm; seed=42. Dùng cùng vector cho mọi global split; largest remainder để có số nguyên. Không ép số mẫu hoặc tỷ lệ Attack từng client; không ép loại hiếm phải có mặt. Tỷ lệ thực tế được lưu cùng vector gốc. Label model chỉ 0/1.

Scaler: fit-only per-client signed log1p, population Z-score, std gần 0 thay bằng 1, clip [-8,8], float32. Validation/test dùng scaler đã đóng băng của client đích.

Tuning mới: DNN/CNN 10 trial mỗi model, 3 Local + Centralized + weighted FedAvg trong mỗi trial. Mục tiêu 12 validation F1 Attack: bỏ Local tự đánh giá, giữ 6 Local chéo + 3 Centralized + 3 FL; tie-break min FL-client F1 rồi mean recall. Giữ anchor/search space gốc, không server LR tuning. Adam ở client, trung bình server trực tiếp. FL 15 round × 3 local epoch, checkpoint round cuối; Local/Centralized tối đa 45 epoch, patience 5, restore min validation loss.

Train/test: 3 Local + Centralized + weighted/uniform FedAvg, chung tham số đã chọn và initialization. Weighted theo số mẫu fit thực tế, uniform 1/3. Không chọn model/threshold/seed bằng test. Báo cáo pooled và từng test-client; Local chéo phải so FL trên cùng test-client. Đọc recall từng Attack cùng support và phần metadata không đa nghĩa.

Pool này tăng tỷ trọng Attack so với nguồn sạch để giữ hết Attack; không diễn giải là lấy mẫu đại diện tự nhiên. Non-IID là mô phỏng theo nhóm Attack/Benign, không phải client theo tổ chức hoặc phiên mạng thực.

## CNN FedAdam theo số mẫu train: 20 server learning rate

Thí nghiệm bổ sung độc lập study FedAvg: `fedadam_search.py`, cấu hình `configs/fedadam_weighted.json`. Giữ cấu hình CNN đã chọn, client Adam LR=0,0003, seed, initial model weights, BCE, threshold 0,5, fit rows và lịch 15×3. Chỉ tuning 20 server LR được khai báo trước; beta1=0,9, beta2=0,999, epsilon=1e-7. Tổng hợp NUM_EXAMPLES; tỷ trọng client lấy từ số mẫu fit thực tế, không dùng 1/3.

Mỗi trial chỉ fit và validation, đánh giá validation ở round cuối. Chọn mean F1 Attack trên ba validation-client, tie-break min client F1, mean recall, lower mean FAR, earlier trial. Không chạy Local/Centralized để chọn riêng server LR. Trọng số ngang nhau khi tính điểm validation không phải Uniform aggregation. Cả 20 trial phải hoàn tất trước final. Sau lựa chọn, khởi tạo lại từ seed gốc, train 15×3 và test một lần; không dùng test để chọn LR/round/threshold. Lưu nguồn study, cấu hình, trọng số client, hash initialization và bảng đầy đủ 20 trial. Không đổi `configs/tuning.json` hay core code của study baseline.
