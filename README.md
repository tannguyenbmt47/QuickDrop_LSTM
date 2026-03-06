# Phân tích cảm xúc trên tập IMDB bằng LSTM — So sánh Adam và QuickProp

**Môn học:** Toán cho Trí tuệ Nhân tạo  

---

## 1. Giới thiệu bài toán

Phân tích cảm xúc (sentiment analysis) là bài toán phân loại văn bản thành hai lớp: tích cực (positive) và tiêu cực (negative). Bài toán này được sử dụng rộng rãi trong xử lý ngôn ngữ tự nhiên (NLP).

Tập dữ liệu sử dụng là **IMDB Movie Reviews** gồm 50.000 đánh giá phim đã được gán nhãn positive/negative. Thí nghiệm sử dụng **2000 mẫu huấn luyện** và **1000 mẫu kiểm tra**. Mục tiêu chính là so sánh hai phương pháp tối ưu trong hai chế độ huấn luyện khác nhau (mini-batch và full-batch), không nhằm đạt kết quả tối ưu trên tập dữ liệu.

## 2. Mô hình và cách tiếp cận

### 2.1. Kiến trúc mô hình

Mô hình sử dụng là **LSTM (Long Short-Term Memory)** với cấu trúc:

- **Embedding**: vocabulary = 5000, dimension = 32
- **LSTM**: 1 layer, hidden size = 100
- **Fully Connected**: 100 → 1 (hàm kích hoạt sigmoid)
- **Hàm mất mát**: Binary Cross Entropy

Cả hai phương pháp tối ưu đều sử dụng cùng bộ trọng số khởi tạo (seed = 42) để đảm bảo tính công bằng khi so sánh.

### 2.2. Hai chế độ huấn luyện

Thí nghiệm được thực hiện với hai chế độ:

1. **Mini-batch** (`lstm_minibatch.py`): batch size = 64, 40 epochs. Mỗi epoch duyệt qua nhiều mini-batch, cập nhật trọng số sau mỗi batch.
2. **Full-batch** (`lstm_fullbatch.py`): tích lũy gradient trên toàn bộ tập huấn luyện (gradient accumulation với sub-batch = 128 để tiết kiệm bộ nhớ), cập nhật trọng số 1 lần mỗi epoch, 60 epochs.

### 2.3. Hai phương pháp tối ưu

1. **Adam** (Kingma & Ba, 2015) — sử dụng gradient kết hợp momentum bậc nhất, bậc hai và adaptive learning rate.
2. **QuickProp** (Fahlman, 1989) — sử dụng xấp xỉ parabol dựa trên gradient hiện tại và gradient trước đó để xác định bước cập nhật.

## 3. Lý thuyết QuickProp

### 3.1. Nguyên lý

QuickProp xấp xỉ hàm loss theo từng tham số bằng một parabol (đa thức bậc 2) dựa trên gradient tại hai bước liên tiếp. Từ đó xác định đỉnh parabol làm điểm cập nhật tiếp theo.

Giả sử tại bước $t$, ta có:
- $g_t$ — gradient hiện tại
- $g_{t-1}$ — gradient ở bước trước
- $\Delta w^{(t-1)}$ — bước cập nhật trước đó

### 3.2. Công thức cập nhật

$$\Delta w^{(t)} = \frac{g_t}{g_{t-1} - g_t} \cdot \Delta w^{(t-1)}$$

Ý tưởng: nếu coi loss là parabol theo $w$, hai gradient $g_{t-1}$ và $g_t$ xác định được hai điểm trên parabol đó. Tỷ số $\frac{g_t}{g_{t-1} - g_t}$ ước lượng vị trí đỉnh parabol (cực tiểu) dựa trên tốc độ thay đổi gradient.


### 3.3. So sánh với Adam

| Đặc điểm | Adam | QuickProp |
|-----------|------|-----------|
| Loại | Gradient bậc nhất + momentum | Xấp xỉ bậc hai (parabol) |
| Thông tin sử dụng | Gradient + trung bình trượt bậc 1, 2 | Gradient hiện tại + gradient trước |
| Adaptive | Có (per-parameter learning rate) | Có (bước cập nhật tỷ lệ với đường cong loss) |
| Hyperparameter | lr, $\beta_1$, $\beta_2$ | lr, max_delta |

## 4. Kết quả thực nghiệm

### 4.0. Biểu đồ so sánh tổng quan

#### So sánh Accuracy tốt nhất

![So sánh accuracy tốt nhất](images/best_accuracy.png)

#### Train Loss theo epoch

![Train Loss](images/train_loss.png)

#### Validation Loss theo epoch

![Validation Loss](images/val_loss.png)

#### Validation Accuracy theo epoch

![Validation Accuracy](images/val_accuracy.png)

### 4.1. Thí nghiệm 1 — Mini-batch (lstm_minibatch.py)

**Cấu hình:** batch size = 64, 40 epochs, cập nhật trọng số mỗi mini-batch.

#### Adam (mini-batch)

| Chỉ số | Giá trị |
|--------|---------|
| Epoch tốt nhất | 19 |
| Val accuracy cao nhất | 70.70% |
| Val loss tại epoch tốt nhất | 1.0803 |
| Thời gian | 13.1s |

Train loss giảm nhanh từ 0.692 (epoch 1) xuống 0.002 (epoch 40). Val loss tăng từ 0.65 (epoch 7) lên 1.80 (epoch 40), cho thấy overfitting nghiêm trọng. Val accuracy đạt đỉnh 70.70% tại epoch 19, sau đó dao động trong khoảng 68–70%.

#### QuickProp (mini-batch)

| Chỉ số | Giá trị |
|--------|---------|
| Epoch tốt nhất | 27 |
| Val accuracy cao nhất | 54.20% |
| Val loss tại epoch tốt nhất | 0.6987 |
| Thời gian | 15.4s |

Train loss dao động quanh 0.69–0.71 trong suốt 40 epoch, giảm không đáng kể. Val accuracy dao động trong khoảng 50–54%, mô hình gần như không học được.

#### So sánh mini-batch

| Phương pháp | Best Val Loss | Best Acc | Epoch | Thời gian |
|-------------|---------------|----------|-------|-----------|
| Adam | 1.0803 | 70.70% | 19 | 13.1s |
| QuickProp | 0.6987 | 54.20% | 27 | 15.4s |

### 4.2. Thí nghiệm 2 — Full-batch (lstm_fullbatch.py)

**Cấu hình:** gradient accumulation trên toàn bộ 2000 mẫu (sub-batch = 128), cập nhật trọng số 1 lần mỗi epoch, 60 epochs.

Mục đích: QuickProp ban đầu được thiết kế cho full-batch training (Fahlman, 1989). Thí nghiệm này kiểm tra liệu gradient ổn định hơn có cải thiện hiệu suất QuickProp hay không.

#### Adam (full-batch)

| Chỉ số | Giá trị |
|--------|---------|
| Epoch tốt nhất | 56 |
| Val accuracy cao nhất | 67.10% |
| Val loss tại epoch tốt nhất | 0.6658 |
| Thời gian | 18.7s |

Train loss giảm chậm và đều từ 0.694 (epoch 1) xuống 0.432 (epoch 60). Val loss giảm từ 0.695 xuống 0.666 rồi dao động nhẹ. Val accuracy tăng dần từ 48.3% lên 67.1% tại epoch 56. So với mini-batch, Adam full-batch hội tụ chậm hơn đáng kể do chỉ cập nhật 1 lần mỗi epoch thay vì 31 lần (2000/64 ≈ 31 mini-batch).

#### QuickProp (full-batch)

| Chỉ số | Giá trị |
|--------|---------|
| Epoch tốt nhất | 30 |
| Val accuracy cao nhất | 52.70% |
| Val loss tại epoch tốt nhất | 0.9469 |
| Thời gian | 12.4s |

Train loss dao động mạnh, xuất hiện các đỉnh cao (1.22 ở epoch 15, 1.28 ở epoch 32). Val loss rất bất ổn, dao động từ 0.69 đến 1.27. Val accuracy dao động quanh 49–52%, thấp hơn cả phiên bản mini-batch (54.20%).

#### So sánh full-batch

| Phương pháp | Best Val Loss | Best Acc | Epoch | Thời gian |
|-------------|---------------|----------|-------|-----------|
| Adam (full-batch) | 0.6658 | 67.10% | 56 | 18.7s |
| QuickProp (full-batch) | 0.9469 | 52.70% | 30 | 12.4s |

### 4.3. So sánh tổng hợp 4 cấu hình

| Phương pháp | Chế độ | Best Val Loss | Best Acc | Epoch | Thời gian |
|-------------|--------|---------------|----------|-------|-----------|
| Adam | mini-batch | 1.0803 | **70.70%** | 19 | 13.1s |
| Adam | full-batch | 0.6658 | 67.10% | 56 | 18.7s |
| QuickProp | mini-batch | 0.6987 | 54.20% | 27 | 15.4s |
| QuickProp | full-batch | 0.9469 | 52.70% | 30 | 12.4s |

## 5. Nhận xét

### 5.1. Adam

- **Mini-batch**: accuracy 70.70%, hội tụ nhanh (best ở epoch 19)
  - Train loss: 0.692 → 0.002 (giảm mạnh)
  - Val loss: 0.65 → 1.80 (tăng → overfitting nặng)
  - Cập nhật trọng số ~31 lần/epoch → tổng ~1240 bước trong 40 epoch
- **Full-batch**: accuracy 67.10%, hội tụ chậm hơn (best ở epoch 56)
  - Train loss: 0.694 → 0.432 (giảm đều)
  - Val loss: 0.695 → 0.666 (ổn định → overfitting nhẹ)
  - Cập nhật trọng số 1 lần/epoch → tổng 60 bước
- **Nhận xét**: mini-batch hội tụ nhanh hơn nhờ nhiều bước cập nhật, nhưng dễ overfitting trên tập nhỏ. Full-batch ổn định hơn, gap train-val loss nhỏ.

### 5.2. QuickProp

- **Mini-batch**: accuracy 54.20% (best ở epoch 27)
  - Train loss: dao động 0.69–0.71, gần như không giảm
  - Val accuracy: dao động 50–54%
- **Full-batch**: accuracy 52.70% (best ở epoch 30)
  - Train loss: dao động mạnh, xuất hiện đỉnh 1.22 (epoch 15), 1.28 (epoch 32)
  - Val loss: bất ổn, dao động 0.69–1.27
- **Nhận xét**: QuickProp không hội tụ ở cả hai chế độ. Full-batch không cải thiện mà còn kém hơn mini-batch.

### 5.3. Nguyên nhân QuickProp kém trên LSTM

- **Loss landscape phi lồi**: LSTM có nhiều điểm yên ngựa và cực tiểu địa phương → xấp xỉ parabol chỉ chính xác trong vùng lân cận nhỏ
- **Thiếu cơ chế ổn định**: Adam dùng exponential moving average, QuickProp chỉ dựa trên 2 gradient liên tiếp → nhạy cảm với nhiễu
- **Gradient vanishing/exploding**: gradient qua nhiều time step thay đổi biên độ lớn → tỷ số $\frac{g_t}{g_{t-1} - g_t}$ không ổn định


### 5.4. Overfitting

- **Adam mini-batch**: overfitting rõ rệt (train loss 0.002, val loss 1.80)
- **Adam full-batch**: overfitting nhẹ (train loss 0.432, val loss 0.666)
- **QuickProp**: không overfitting ở cả hai chế độ — nhưng do underfitting (mô hình chưa học được)

### 5.5. Kết luận

- **Adam**: hội tụ tốt ở cả hai chế độ — 70.70% (mini-batch), 67.10% (full-batch)
- **QuickProp**: không hội tụ hiệu quả — 54.20% (mini-batch), 52.70% (full-batch)
- Xấp xỉ parabol bậc hai không phù hợp với loss landscape phi lồi, nhiều chiều của LSTM
- QuickProp có thể phù hợp hơn cho mô hình đơn giản (MLP nông) hoặc bài toán có loss landscape lồi

## 6. Hướng dẫn chạy

```bash
pip install torch numpy keras tensorflow matplotlib

# Chế độ mini-batch
python3 lstm_minibatch.py

# Chế độ full-batch
python3 lstm_fullbatch.py

# Vẽ biểu đồ so sánh
python3 plot_results.py
```


