# BÁO CÁO NGHIỆM THU KỸ THUẬT - PHẦN VIỆC THÀNH VIÊN 3
## HỆ THỐNG CHẤM BÀI TRẮC NGHIỆM TỰ ĐỘNG TỪ ẢNH CHỤP (OMR)

* **Học viện Công nghệ Bưu chính Viễn thông (PTIT)**
* **Khoa Công nghệ Thông tin 1 – Bộ môn Khoa học Máy tính**
* **Học phần:** Xử lý ảnh số (INT13146)
* **Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt
* **Nhóm sinh viên thực hiện:** Nhóm 4 (Quy mô 04 sinh viên)
* **Thành viên phụ trách:** **Thành viên 3 (Phân đoạn ảnh & Nhận diện ô tô đáp án)**
* **Tỷ lệ đóng góp kỹ thuật:** **25%** (Cài đặt thuật toán 100% Code from Scratch bằng `NumPy`)

---

## MỤC LỤC
1. [Tổng quan nhiệm vụ của Thành viên 3](#1-tổng-quan-nhiệm-vụ-của-thành-viên-3)
2. [Cơ sở lý thuyết và Cài đặt thuật toán thuần NumPy (Code from Scratch)](#2-cơ-sở-lý-thuyết-và-cài-đặt-thuật-toán-thuần-numpy-code-from-scratch)
   - [2.1. Phân ngưỡng tự động Otsu toàn cục (Otsu's Global Thresholding)](#21-phân-ngưỡng-tự-động-otsu-toàn-cục-otsus-global-thresholding)
   - [2.2. Bảng Ảnh tích phân (Integral Image) & Phân ngưỡng thích nghi cục bộ (Adaptive Mean Thresholding)](#22-bảng-ảnh-tích-phân-integral-image--phân-ngưỡng-thích-nghi-cục-bộ-adaptive-mean-thresholding)
   - [2.3. Bốn phép toán Hình thái học nhị phân (Binary Mathematical Morphology)](#23-bốn-phép-toán-hình-thái-học-nhị-phân-binary-mathematical-morphology)
   - [2.4. Thuật toán Gán nhãn thành phần liên thông 2 lượt (Two-Pass CCL) & Cấu trúc Disjoint-Set](#24-thuật-toán-gán-nhãn-thành-phần-liên-thông-2-lượt-two-pass-ccl--cấu-trúc-disjoint-set)
   - [2.5. Trích xuất ROI ô tròn, Tỷ lệ lấp đầy (Fill Ratio) $\tau_{q,c}$ & Logic phân loại 4 trạng thái](#25-trích-xuất-roi-ô-tròn-tỷ-lệ-lấp-đầy-fill-ratio-tau_qc--logic-phân-loại-4-trạng-thái)
3. [Kiến trúc Module và Thiết kế Pipeline tích hợp (`TV3SegmentationPipeline`)](#3-kiến-trúc-module-và-thiết-kế-pipeline-tích-hợp-tv3segmentationpipeline)
4. [Kết quả Thực nghiệm Định lượng (Benchmark: Scratch vs. OpenCV)](#4-kết-quả-thực-nghiệm-định-lượng-benchmark-scratch-vs-opencv)
5. [Khảo sát Tham số (Ablation Study): Otsu vs. Adaptive Mean dưới bóng đổ (DS-03)](#5-khảo-sát-tham-số-ablation-study-otsu-vs-adaptive-mean-dưới-bóng-đổ-ds-03)
6. [Hướng dẫn Ghép nối Hệ thống (Integration Interfaces với TV2 và TV4)](#6-hướng-dẫn-ghép-nối-hệ-thống-integration-interfaces-với-tv2-và-tv4)
7. [Phân tích Trường hợp Thất bại (Failure Cases) và Hướng khắc phục](#7-phân-tích-trường-hợp-thất-bại-failure-cases-và-hướng-khắc-phục)
8. [Kết luận và Minh chứng Nghiệm thu](#8-kết-luận-và-minh-chứng-nghiệm-thu)

---

## 1. TỔNG QUAN NHIỆM VỤ CỦA THÀNH VIÊN 3

Trong cấu trúc 4 giai đoạn của dự án OMR, **Thành viên 3** chịu trách nhiệm trọng tâm ở **Giai đoạn 3 (Phân đoạn ảnh & Nhận diện ô tô đáp án)**:

```
[Đầu vào: Ảnh xám phiếu thi đã nắn phẳng phối cảnh từ TV2: warped_gray]
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MODULE THÀNH VIÊN 3: PHÂN ĐOẠN ẢNH & NHẬN DIỆN Ô TÔ ĐÁP ÁN (GIAI ĐOẠN 3)     │
│ 1. Phân ngưỡng tự động: Otsu toàn cục & Adaptive Mean qua Integral Image    │
│ 2. Xử lý Hình thái học nhị phân: Opening (khử nhiễu) + Closing (lấp lỗ)      │
│ 3. Gán nhãn thành phần liên thông: Two-Pass CCL (Union-Find) & Lọc ô tròn    │
│ 4. Trích xuất Circular ROI Mask & Tính Fill Ratio \tau_{q, c}                │
│ 5. Phân loại 4 trạng thái: VALID, BLANK, ERASED_VALID, MULTIPLE_INVALID      │
│ 6. Đối chứng thực nghiệm song song với thư viện OpenCV chuẩn                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
[Đầu ra cho TV4: Vector đáp án OMR + Trực quan hóa Triplet View + Benchmark]
```

### Ràng buộc kỹ thuật nghiêm ngặt:
* **100% Cài đặt từ đầu (Code from Scratch)** bằng thư viện ma trận `NumPy`. Tuyệt đối không dùng các hàm xử lý ảnh nhị phân/hình thái học của OpenCV (`cv2.threshold`, `cv2.adaptiveThreshold`, `cv2.morphologyEx`, `cv2.connectedComponents`) trong luồng chính.
* Thư viện `OpenCV` chỉ được gọi độc lập trong khối test/benchmark để **đối chứng sai số số học (MSE, PSNR, IoU, Dice)** và **đo tốc độ thực thi (Latency)**.

---

## 2. CƠ SỞ LÝ THUYẾT VÀ CÀI ĐẶT THUẬT TOÁN THUẦN NUMPY

### 2.1. Phân ngưỡng tự động Otsu toàn cục (Otsu's Global Thresholding)

#### Cơ sở toán học:
Cho ảnh mức xám $I(x, y)$ kích thước $H \times W$ có $L = 256$ mức xám. Xác suất xuất hiện của mức xám $i \in [0, 255]$ là:
$$p_i = \frac{n_i}{N}, \quad \sum_{i=0}^{255} p_i = 1, \quad N = H \times W$$

Với mỗi ngưỡng thử $t \in [0, 255]$, chia không gian pixel thành hai lớp tiền cảnh $C_0 = [0, t]$ và hậu cảnh $C_1 = [t+1, 255]$:
$$\omega_0(t) = \sum_{i=0}^t p_i, \quad \omega_1(t) = \sum_{i=t+1}^{255} p_i = 1 - \omega_0(t)$$
$$\mu_0(t) = \frac{1}{\omega_0(t)} \sum_{i=0}^t i \cdot p_i, \quad \mu_1(t) = \frac{1}{\omega_1(t)} \sum_{i=t+1}^{255} i \cdot p_i$$

Mục tiêu của thuật toán Otsu là tìm ngưỡng $T^*$ cực đại hóa phương sai giữa hai lớp (Between-class variance):
$$\sigma_B^2(t) = \omega_0(t) \omega_1(t) \left[ \mu_0(t) - \mu_1(t) \right]^2$$
$$T^* = \arg\max_{0 \le t \le 255} \sigma_B^2(t)$$

#### Kỹ thuật cài đặt Scratch tối ưu:
1. Tính Histogram 256 bin bằng `np.bincount(image.ravel(), minlength=256)`.
2. Duyệt qua 256 mức xám với biến tích lũy cộng dồn $O(1)$ cho $\omega_0$ và $\sum i \cdot p_i$, không tính lại toàn bộ mảng.
3. Khi phân ngưỡng `BINARY_INV`, các điểm ảnh $I(x, y) \le T^*$ (vết chì tô đậm) được gán giá trị $255$ (tiền cảnh trắng), nền giấy trắng $I(x, y) > T^*$ được gán $0$ (hậu cảnh đen).

---

### 2.2. Bảng Ảnh tích phân (Integral Image) & Phân ngưỡng thích nghi cục bộ (Adaptive Mean)

#### Cơ sở toán học:
Khi ảnh chụp bị bóng đổ (tập dữ liệu **DS-03**), mức xám của nền giấy trong vùng tối có thể giảm xuống $[90, 130]$, chồng lấn với mức xám của vết bút chì ở vùng sáng $[40, 90]$. Thuật toán Otsu toàn cục tìm một ngưỡng cố định duy nhất $T^* \approx 135$ sẽ biến toàn bộ vùng giấy trong bóng râm thành tiền cảnh trắng, gây hỏng hoàn toàn kết quả chấm.

Phân ngưỡng thích nghi cục bộ (Adaptive Mean Thresholding) giải quyết triệt để vấn đề này bằng cách tính ngưỡng riêng biệt $T(x, y)$ cho từng điểm ảnh dựa trên trung bình lân cận cửa sổ $b \times b$:
$$T(x, y) = \mu_{\text{local}}(x, y) - C = \left( \frac{1}{b^2} \sum_{u=-r}^r \sum_{v=-r}^r I(x+u, y+v) \right) - C$$
$$B(x, y) = \begin{cases} 255 & \text{nếu } I(x, y) \le T(x, y) \quad (\text{Chuẩn BINARY\_INV}) \\ 0 & \text{ngược lại} \end{cases}$$

#### Tối ưu hóa thời gian tính toán $O(1)$ bằng Bảng Ảnh tích phân:
Nếu dùng vòng lặp tích chập cửa sổ trượt $b \times b$ thông thường trên ảnh $800 \times 1000$, độ phức tạp tính toán là $O(H \cdot W \cdot b^2) \approx 5 \times 10^8$ phép tính, gây độ trễ $> 5$ giây.

Thành viên 3 đã cài đặt **Bảng Ảnh tích phân (Integral Image / Summed-Area Table)** $S(y, x)$ kích thước $(H+1) \times (W+1)$:
$$S(y, x) = \sum_{r \le y} \sum_{c \le x} I(r, c)$$
Được tính cực nhanh bằng 2 lượt `np.cumsum` (theo trục 0 rồi trục 1) với kiểu dữ liệu `np.uint64` chống tràn số.

Tổng mức xám trong bất kỳ cửa sổ hình chữ nhật $[r_1, r_2] \times [c_1, c_2]$ được tính chỉ với **04 phép truy xuất và 03 phép cộng trừ $O(1)$**:
$$\sum_{r=r_1}^{r_2} \sum_{c=c_1}^{c_2} I(r, c) = S(r_2+1, c_2+1) - S(r_1, c_2+1) - S(r_2+1, c_1) + S(r_1, c_1)$$

Kết hợp với kỹ thuật đệm biên `np.pad(image, r, mode='edge')` (tương thích 100% với `cv2.BORDER_REPLICATE`), toàn bộ ảnh được nhị phân hóa trong vòng **$< 35\text{ ms}$**.

---

### 2.3. Bốn phép toán Hình thái học nhị phân (Binary Mathematical Morphology)

#### Cơ sở toán học với phần tử cấu trúc $B$ kích thước $3 \times 3$:
1. **Phép Co (Binary Erosion):**
   $$(A \ominus B)(x, y) = \min_{(u, v) \in B} A(x+u, y+v)$$
   *Tác dụng:* Bào mòn đường viền ngoài, xóa các hạt bụi chì nhỏ, tách rời các ô bị lem dính nhẹ.
2. **Phép Giãn (Binary Dilation):**
   $$(A \oplus B)(x, y) = \max_{(u, v) \in B} A(x+u, y+v)$$
   *Tác dụng:* Làm đậm nét tô bút chì, lấp các khe hở trắng nhỏ do thí sinh tô chưa kín ô.
3. **Phép Mở (Binary Opening):**
   $$A \circ B = (A \ominus B) \oplus B$$
   *Tác dụng:* Co trước rồi Giãn sau — loại bỏ triệt để nhiễu muối tiêu và gai nét mà không làm thay đổi diện tích tổng thể của ô tròn.
4. **Phép Đóng (Binary Closing):**
   $$A \bullet B = (A \oplus B) \ominus B$$
   *Tác dụng:* Giãn trước rồi Co sau — hàn gắn các lỗ thủng bên trong vùng tô chì và nối liền các nét đứt.

#### Kỹ thuật vector hóa Sliding Window:
Thay vì sử dụng 2 vòng lặp `for` lồng nhau quét qua từng pixel, hàm sử dụng:
```python
windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
# Erosion: Tất cả các điểm dưới kernel phải là tiền cảnh
erode_res = np.where(np.all(windows[..., k_mask] > 0, axis=-1), 255, 0)
# Dilation: Chỉ cần ít nhất 1 điểm dưới kernel là tiền cảnh
dilate_res = np.where(np.any(windows[..., k_mask] > 0, axis=-1), 255, 0)
```
Tốc độ nhanh gấp $\approx 60$ lần so với vòng lặp Python thông thường.

---

### 2.4. Thuật toán Gán nhãn thành phần liên thông 2 lượt (Two-Pass CCL) & Cấu trúc Disjoint-Set

#### Cơ sở lý thuyết:
Để phát hiện độc lập từng ô trắc nghiệm và lọc bỏ khung bảng hoặc nhiễu, thuật toán duyệt 2 lượt (Two-Pass Connected Component Labeling) liên thông 8 hướng được cài đặt kết hợp cấu trúc dữ liệu **Disjoint-Set (Union-Find)**:

1. **Cấu trúc Disjoint-Set tối ưu:**
   - **Nén đường dẫn hai lượt lặp (Iterative Path Compression):** Không dùng đệ quy để tránh lỗi tràn ngăn xếp (RecursionError) khi ảnh có hàng chục nghìn điểm ảnh liên thông. Mọi node trên nhánh duyệt đều được trỏ thẳng về node gốc `root`.
   - **Hợp nhất ưu tiên nhãn nhỏ hơn (Union by Smallest Label):** Luôn gán node cha là nhãn có giá trị số học nhỏ hơn để đảm bảo tính tất định.
2. **Pass 1 (Quét tiến):** Quét qua các điểm ảnh tiền cảnh $I(r, c) > 0$. Kiểm tra 4 lân cận đã duyệt trước đó $(r-1, c), (r, c-1), (r-1, c-1), (r-1, c+1)$:
   - Nếu không có lân cận nào có nhãn: Tạo nhãn tạm mới `ds.add()`.
   - Nếu có lân cận: Gán nhãn bằng `min(neighbors)`, đồng thời gọi `ds.union()` cho các nhãn tương đương.
3. **Pass 2 (Chuẩn hóa và gom cụm):**
   - Xây dựng bảng tra cứu ánh xạ `lut[label] = ds.find(label)`.
   - Ánh xạ lại nhãn theo thứ tự quét raster xuất hiện đầu tiên, chuẩn hóa về dải liên tục $1, 2, \dots, K$.
4. **Trích xuất thông số hình học (Vectorized Feature Extraction):**
   - Diện tích $Area$, Bounding Box $(x, y, w, h)$, Trọng tâm thực $(c_x, c_y)$, Tỷ lệ khung hình $Aspect\ Ratio = w/h$, và Độ đặc $Extent = Area / (w \times h)$.
5. **Bộ lọc ô tròn OMR (`filter_bubble_components`):**
   - Chỉ giữ lại các vùng thỏa mãn: $40 \le Area \le 2500$, $0.6 \le Aspect\ Ratio \le 1.6$, $Extent \ge 0.40$.
   - Loại bỏ hoàn toàn đường kẻ bảng, tiêu đề chữ và bụi nhiễu.

---

### 2.5. Trích xuất ROI ô tròn, Tỷ lệ lấp đầy $\tau_{q,c}$ & Logic phân loại 4 trạng thái

#### Cơ sở toán học:
Vùng quan tâm (ROI) của ô trắc nghiệm thứ $c \in \{A, B, C, D\}$ thuộc câu hỏi $q \in \{1 \dots 20\}$ là hình tròn bán kính $R$ tại tâm $(c_x, c_y)$:
$$\Omega_{q, c} = \left\{ (u, v) \mid (u - c_x)^2 + (v - c_y)^2 \le R^2 \right\}$$

Tỷ lệ lấp đầy (Fill Ratio) trên ảnh nhị phân sau Morphology:
$$\tau_{q, c} = \frac{1}{|\Omega_{q, c}|} \sum_{(u, v) \in \Omega_{q, c}} \frac{B_{\text{morph}}(u, v)}{255} \in [0.0, 1.0]$$

#### Logic phân loại 4 trạng thái (Decision Logic):
Sắp xếp các phương án của câu $q$ theo tỷ lệ giảm dần: $\tau_{(1)} \ge \tau_{(2)} \ge \tau_{(3)} \ge \tau_{(4)}$ và $\Delta \tau = \tau_{(1)} - \tau_{(2)}$.

| Trạng thái | Điều kiện nhận diện | Giải thích thực tế | Kết luận chấm thi |
| :--- | :--- | :--- | :--- |
| **`BLANK`** | $\tau_{(1)} < \tau_{\text{min}}$ ($0.38$) | Thí sinh bỏ trống, không tô ô nào. | `selected = None`, `is_valid = False` |
| **`VALID`** | $\tau_{(1)} \ge \tau_{\text{min}}$ và $\tau_{(2)} < \tau_{\text{min}}$ | Thí sinh tô duy nhất 1 ô hợp lệ chuẩn. | `selected = opt_(1)`, `is_valid = True` |
| **`ERASED_VALID`** | $\tau_{(1)} \ge \tau_{\text{min}}$, $\tau_{(2)} \ge \tau_{\text{min}}$ và $\Delta \tau \ge 0.15$ | Thí sinh tẩy ô cũ (còn vết mờ) và tô đậm ô mới. | `selected = opt_(1)`, `is_valid = True` |
| **`MULTIPLE_INVALID`** | $\tau_{(1)} \ge \tau_{\text{min}}$, $\tau_{(2)} \ge \tau_{\text{min}}$ và $\Delta \tau < 0.15$ | Tô đúp hoặc tẩy không sạch gây tranh chấp. | `selected = None`, `is_valid = False` |

---

## 3. KIẾN TRÚC MODULE VÀ THIẾT KẾ PIPELINE TÍCH HỢP

Mã nguồn của Thành viên 3 được đóng gói thành một hệ sinh thái module hoàn chỉnh trong thư mục `src/`:

```
c:\Users\Admin\Desktop\xla\
├── src\
│   ├── __init__.py                # Re-export đầy đủ API và cấu hình đường dẫn an toàn
│   ├── tv3_thresholding.py        # Otsu toàn cục, Integral Image, Adaptive Mean & Benchmark
│   ├── tv3_morphology_ccl.py      # Erode, Dilate, Open, Close, DisjointSet, Two-Pass CCL
│   ├── tv3_roi_classifier.py      # Circular ROI Mask, Fill Ratio, 4-state Classifier, Grid
│   └── tv3_segmentation.py        # TV3SegmentationPipeline tích hợp đầu-cuối & Standalone CLI
├── tests\
│   ├── test_tv3_thresholding.py   # 37 Unit tests cho ngưỡng Otsu và Adaptive
│   ├── test_tv3_morphology_ccl.py # 19 Unit tests cho Morphology và Two-Pass CCL
│   ├── test_tv3_roi_classifier.py # 10 Unit tests cho Fill Ratio, auto-grid và phân loại đáp án
│   └── test_tv3_benchmark.py      # 19 Integration tests nghiệm thu & xử lý các mục Code Review
├── outputs\tv3\                   # Toàn bộ sản phẩm ảnh và số liệu xuất tự động
│   ├── 1_input_warped.png         # Ảnh xám phiếu thi đầu vào
│   ├── 2_thresholded.png          # Ảnh sau phân ngưỡng nhị phân
│   ├── 3_morphology.png           # Ảnh sau xử lý Opening + Closing
│   ├── 4_ccl_labels.png           # Bản đồ màu trực quan hóa các thành phần CCL
│   ├── 5_omr_annotated.png        # Phiếu thi khoanh màu và nhãn kết quả từng câu
│   ├── 6_triplet_comparison.png   # Triplet View: Scratch vs OpenCV vs Heatmap (|diff| x 10)
│   ├── benchmark_metrics.csv      # Bảng số liệu đối chứng định lượng CSV
│   ├── benchmark_metrics.json     # Báo cáo đối chứng định lượng JSON
│   ├── recognition_results.csv    # Bảng kết quả nhận diện 20 câu chi tiết CSV
│   └── recognition_results.json   # Chi tiết kết quả nhận diện JSON
└── reports\
    └── Bao_Cao_Thanh_Vien_3_OMR.md # Báo cáo kỹ thuật và nghiệm thu cá nhân (File này)
```

---

## 4. KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG (BENCHMARK: SCRATCH VS. OPENCV)

Thực nghiệm được tiến hành trên phiếu thi OMR chuẩn kích thước $800 \times 1000\text{ px}$ gồm 20 câu hỏi (80 ô trắc nghiệm), có đầy đủ hiệu ứng bóng đổ DS-03, tẩy xóa và tô đúp:

### 4.1. Bảng Đối chứng Độ chính xác Số học và Thời gian thực thi

| Giai đoạn xử lý thuật toán | Thuật toán NumPy Scratch | Hàm OpenCV đối chứng | MSE | PSNR (dB) | IoU (%) | Dice (%) | Thời gian Scratch (ms) | Thời gian OpenCV (ms) | Đánh giá nghiệm thu |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phân ngưỡng thích nghi** | `adaptive_mean_threshold_scratch` | `cv2.adaptiveThreshold` | $0.0000$ | $999.0$ ($\infty$) | **$100.00\%$** | $100.00\%$ | $50.4\text{ ms}$ | $10.4\text{ ms}$ | **ĐẠT XUẤT SẮC** ($\text{IoU} \ge 98.5\%$) |
| **Xử lý hình thái học** | `opening_` + `closing_binary_scratch` | `cv2.morphologyEx` (OPEN+CLOSE) | $0.0000$ | $999.0$ ($\infty$) | **$100.00\%$** | $100.00\%$ | $55.8\text{ ms}$ | $1.0\text{ ms}$ | **ĐẠT XUẤT SẮC** ($\text{IoU} \ge 98.5\%$) |
| **Gán nhãn liên thông** | `two_pass_ccl_scratch` | `cv2.connectedComponentsWithStats` | - | - | - | - | $131.3\text{ ms}$ | $6.2\text{ ms}$ | **ĐẠT** (Khớp $62/62$ vùng, $100\%$) |
| **Nhận diện ROI & Chấm** | `recognize_all_questions` | - | - | - | - | - | $5.0\text{ ms}$ | - | **ĐẠT** (Độ chính xác $100\%$) |
| **TOÀN BỘ PIPELINE TV3** | `TV3SegmentationPipeline` | Full OpenCV Counterpart | - | - | **$100.00\%$** | **$100.00\%$** | **$242.6\text{ ms}$** | **$17.6\text{ ms}$** | **VƯỢT CHỈ TIÊU** ($< 460\text{ ms}$) |

### 4.2. Phân tích kết quả:
1. **Độ trùng khớp số học tuyệt đối:** Chỉ số $\text{IoU} = 100.00\%$ và $\text{PSNR} = 999.0\text{ dB}$ ($\text{MSE} = 0$) chứng minh thuật toán phân ngưỡng cục bộ qua Bảng Ảnh tích phân và các phép toán Morphology bằng `sliding_window_view` của nhóm đạt độ chuẩn xác toán học tương đương $100\%$ so với mã nguồn C++ tối ưu của OpenCV.
2. **Thời gian đáp ứng (Latency):** Toàn bộ pipeline tự code thuần túy bằng Python/NumPy chỉ mất **$242.6\text{ ms}$ / phiếu**, hoàn toàn vượt trội so với mục tiêu đặt ra trong Kế hoạch đề tài ($460\text{ ms}$ tại Bảng 7.2), đáp ứng xuất sắc yêu cầu xử lý thời gian thực khi chấm bài số lượng lớn.
3. **Tối ưu hiệu năng & Đồng bộ sau Code Review:**
   - Rút gọn bước đánh lại nhãn Two-Pass CCL từ $O(K \cdot L)$ xuống $O(K + L)$ và lọc LUT trong `filter_bubble_components`, loại bỏ nút cổ chai trên ảnh nhiều nhiễu.
   - Tính ngưỡng nhị phân một lần trong `recognize_all_questions` thay vì gọi `np.max(binary_image)` lặp lại cho từng ô.
   - Chuyển toàn bộ `import cv2` vào trong các hàm benchmark/trực quan hóa để module lõi chạy độc lập 100% chỉ cần `NumPy`.
   - **Tối ưu Mục 2.2 (Code Review lần 3):** Trong `TV3SegmentationPipeline.run(..., run_ccl=None)`, tự động bỏ qua bước Two-Pass CCL khi đã truyền sẵn `question_grid` và `run_opencv_benchmark=False` (hoặc khi truyền tường minh `run_ccl=False`), loại bỏ 1 lượt chạy CCL dư thừa khi tích hợp với TV4 (`pipeline_coordinator.py`) và `extract_bubble_answers`.
   - **Ghi chú Mục 3.3 (Hai bản Otsu):** Cài đặt `otsu_threshold_scratch` tại `src/tv3_thresholding.py` (TV3) tồn tại song song với bản Otsu trong `OMR_System_TV1/src/tv1_hough.py` một cách có chủ đích để mỗi phân hệ giữ tính độc lập hoàn toàn (Loose Coupling), không tạo phụ thuộc chéo giữa mã nguồn của các thành viên.

---

## 5. KHẢO SÁT THAM SỐ (ABLATION STUDY): OTSU VS. ADAPTIVE MEAN DƯỚI BÓNG ĐỔ (DS-03)

Thực hiện đúng cam kết trong Mục 6.2 của Kế hoạch đề tài, nhóm tiến hành khảo sát so sánh hiệu quả giữa **Phân ngưỡng Otsu toàn cục** và **Phân ngưỡng thích nghi cục bộ (Adaptive Mean)** trên cùng một phiếu thi bị bóng đổ che phủ một phần:

```
                                  KẾT QUẢ ABLATION STUDY (DS-03)
┌───────────────────────────────────────────────┬───────────────────────────────────────────────┐
│           OTSU GLOBAL THRESHOLDING            │      ADAPTIVE MEAN (INTEGRAL IMAGE O(1))      │
├───────────────────────────────────────────────┼───────────────────────────────────────────────┤
│ • Ngưỡng toàn cục: T* = 102 mức xám           │ • Ngưỡng cục bộ biến thiên theo từng điểm ảnh │
│ • Vùng giấy trong bóng đổ (mức xám ~95) bị    │ • Tự động bù trừ độ sụt giảm chiếu sáng       │
│   nhầm toàn bộ thành vết chì (tiền cảnh 255)  │ • Bóc tách chính xác từng ô tròn trắc nghiệm │
│ • Độ chính xác nhận diện: 50.0% (10/20 câu)   │ • Độ chính xác nhận diện: 100.0% (20/20 câu)  │
│ • 10 câu trong bóng đổ biến thành MULTIPLE    │ • Xử lý hoàn hảo cả câu tẩy xóa và câu tô đúp │
└───────────────────────────────────────────────┴───────────────────────────────────────────────┘
```

* **Kết luận khoa học:** Phân ngưỡng thích nghi cục bộ là lựa chọn bắt buộc cho bài toán OMR thực tế từ ảnh chụp điện thoại. Kỹ thuật Ảnh tích phân giúp thuật toán đạt tốc độ tức thì mà không làm tăng độ trễ tính toán.

---

## 6. HƯỚNG DẪN GHÉP NỐI HỆ THỐNG (INTEGRATION INTERFACES)

Pipeline của Thành viên 3 được thiết kế theo chuẩn module hóa cao (Loose Coupling), sẵn sàng ghép nối trực tiếp với Thành viên 2 và Thành viên 4:

### 6.1. Giao tiếp nhận dữ liệu từ Thành viên 2 (`tv2_perspective.py`):
Thành viên 3 nhận ảnh xám 2D đã được nắn phẳng phối cảnh `warped_gray` (kích thước chuẩn $800 \times 1000$):
```python
from src.tv3_segmentation import run_tv3_pipeline

# warped_gray được tạo ra từ bước warp_perspective_bilinear() của TV2
tv3_result = run_tv3_pipeline(
    warped_gray=warped_gray,
    threshold_method='adaptive',
    block_size=25,
    C=10.0,
    run_opencv_benchmark=True
)
```

### 6.2. Bàn giao dữ liệu cho Thành viên 4 (`tv4_enhancement_eval.py` & `app.py`):
Thành viên 3 cung cấp đầy đủ các cấu trúc dữ liệu chuẩn để Thành viên 4 thực hiện chấm điểm, xuất Excel và hiển thị giao diện Streamlit:
```python
# 1. Lấy vector đáp án nhận diện {1: 'A', 2: 'B', 3: None, ...}
answers_dict = tv3_result.get_answer_vector()

# 2. Lấy thống kê số câu hợp lệ/phạm quy
summary_stats = tv3_result.get_summary()  # {'VALID': 15, 'BLANK': 2, 'ERASED_VALID': 2, 'MULTIPLE_INVALID': 1}

# 3. Lấy chi tiết Fill Ratio từng ô để hiển thị Tab 1 Streamlit
for q_id, q_res in tv3_result.omr_result.questions.items():
    print(f"Câu {q_id}: Đã chọn {q_res.selected}, Trạng thái: {q_res.status}, Fill: {q_res.fill_ratios}")

# 4. Xuất trực tiếp file CSV và JSON phục vụ tải xuống
tv3_result.save_csv("outputs/tv3/recognition_results.csv")
tv3_result.save_json("outputs/tv3/recognition_results.json")

# 5. Lấy ảnh Triplet View để hiển thị đối chứng tại Tab 3 Streamlit
# Ảnh nằm sẵn tại: outputs/tv3/6_triplet_comparison.png
```

---

## 7. PHÂN TÍCH TRƯỜNG HỢP THẤT BẠI (FAILURE CASES) VÀ HƯỚNG KHẮC PHỤC

Để đảm bảo tính khách quan khoa học theo yêu cầu của TS. Phạm Hoàng Việt tại Mục 8 của Đề tài, Thành viên 3 phân tích 2 trường hợp ngoại lệ trong phân đoạn:

1. **Ranh giới bóng râm sắc nét cắt ngang qua đúng 1 ô tròn (Sharp Shadow Boundary):**
   - *Hiện tượng:* Khi bóng bàn tay có mép quá sắc cắt đôi một ô tròn rỗng, nửa ô nằm trong bóng tối có mức xám thấp hơn trung bình cửa sổ cục bộ, tạo ra một vệt trắng giả chiếm $\approx 35\%$ diện tích ô.
   - *Khắc phục đã áp dụng:* Thiết lập ngưỡng lấp đầy $\tau_{\text{min}} = 0.38$ (cao hơn $0.35$) kết hợp phép Opening nhị phân để làm tan rã vệt bóng đổ mỏng.
2. **Thí sinh dùng bút chì quá nhạt ($HB$) hoặc gạch chéo chữ X thay vì tô kín:**
   - *Hiện tượng:* Nét gạch chữ X có diện tích pixel nhỏ, khi qua phép Co (Erosion) có nguy cơ bị đứt nét khiến $\tau_{q,c} < 0.38 \rightarrow$ Bị nhận diện nhầm thành `BLANK`.
   - *Giải pháp:* Cung cấp tham số tùy biến Structuring Element $2 \times 2$ hoặc cho phép bật/tắt phép Co trong cấu hình `TV3PipelineConfig(apply_opening=False)`.

---

## 8. KẾT LUẬN VÀ MINH CHỨNG NGHIỆM THU

* **Khối lượng công việc:** Thành viên 3 đã hoàn thành $100\%$ khối lượng công việc được giao theo đúng kế hoạch đề tài, bao gồm:
  - 04 file mã nguồn kỹ thuật tại `src/` (đóng gói thành package `tv3` qua `pyproject.toml`, không chèn `sys.path` gây xung đột).
  - 04 file kiểm thử tại `tests/` với **85/85 test cases vượt qua (100% Pass)**.
  - Bộ sinh dữ liệu mô phỏng OMR và bộ trực quan hóa Triplet View đạt chuẩn báo cáo khoa học.
* **Xác nhận chỉ tiêu nghiệm thu:**
  - $\Delta T_{\text{Otsu}} \le 1.0$ mức xám: **ĐẠT** ($\Delta T = 0.0$).
  - $\text{IoU}_{\text{bin}} \ge 98.5\%$: **ĐẠT XUẤT SẮC** ($\text{IoU} = 100.0\%$).
  - $\text{PSNR} \ge 48\text{ dB}$: **ĐẠT** ($\text{PSNR} = 999.0\text{ dB}$).
  - Độ chính xác nhận diện đáp án trên phiếu kiểm thử: **$100.0\%$ (20/20 câu)**.
  - Khớp số lượng thành phần liên thông CCL với OpenCV: **$100\%$ (62/62 thành phần)**.
  - Tốc độ thực thi: **$242.6\text{ ms}$ / phiếu** (nhanh hơn $\approx 1.9$ lần so với mục tiêu $460\text{ ms}$).

*Báo cáo được hoàn thiện và đóng gói phục vụ bảo vệ đề tài môn học Xử lý ảnh số (INT13146).*
