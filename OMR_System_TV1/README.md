# MODULE THÀNH VIÊN 1: TIỀN XỬ LÝ & PHÁT HIỆN KHUNG PHIẾU THI (OMR STAGE 1)

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**  
**Khoa Công nghệ Thông tin 1 – Bộ môn Khoa học Máy tính**  
**Môn học:** Xử Lý Ảnh (INT13146)  
**Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt  
**Đồ án:** Nhóm 4 – Hệ thống chấm bài trắc nghiệm tự động từ ảnh chụp (OMR System)  
**Thành viên phụ trách:** **Thành viên 1 (Tiền xử lý & Phát hiện khung phiếu)**  
**Tỷ lệ đóng góp kỹ thuật:** **25%** (Toàn bộ thuật toán cài đặt 100% Code from Scratch bằng `NumPy`)

> **Trạng thái sau đợt sửa theo `Code_Review_OMR_Nhom4.md`:** xem mục 7 để biết từng mục review
> đã xử lý ra sao và các số liệu đã được đo lại như thế nào.

---

## 📌 1. TỔNG QUAN VÀ VỊ TRÍ TRONG HỆ THỐNG

Trong kiến trúc Pipeline 4 giai đoạn của toàn bộ hệ thống chấm thi trắc nghiệm OMR:
1. **Giai đoạn 1 (TV1 - Module này):** **Tiền xử lý & Phát hiện khung phiếu thi**
   (Luminance Grayscale ITU-R BT.601, Gaussian Blur 2D separable convolution,
   Canny Edge Detector 4 bước, Hough Line Transform cực & Xác định 4 đỉnh phiếu [TL, TR, BR, BL]).
2. **Giai đoạn 2 (TV2):** **Nắn thẳng phối cảnh & Định vị Marker**
   (Sắp xếp đỉnh chuẩn hóa, Homography SVD $3 \times 3$, Bilinear Inverse Warping, NCC Template Matching).
3. **Giai đoạn 3 (TV3):** **Phân đoạn ảnh & Nhận diện ô tô đáp án**
   (Otsu toàn cục, Adaptive Mean qua Integral Image, Hình thái học nhị phân, Two-Pass CCL, 4-State Classification).
4. **Giai đoạn 4 (TV4):** **Tăng cường ảnh, Chấm điểm & Giao diện Demo**
   (Histogram Equalization, Median Filter, Chấm điểm, Báo cáo Excel, Streamlit Web App).

```
[Đầu vào: Ảnh chụp phiếu thi (.jpg/.png) từ điện thoại / webcam]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 MODULE THÀNH VIÊN 1 (TV1 - GIAI ĐOẠN 1)                 │
│                                                                         │
│  [1] Chuyển đổi mức xám cảm nhận thị giác (Luminance BT.601)            │
│      - Y = 0.299*R + 0.587*G + 0.114*B (MSE = 0.0000, ΔImax = 0)        │
│                                                                         │
│  [2] Làm mịn khử nhiễu bằng Bộ lọc Gaussian 2D (Separable Convolution)  │
│      - Nhân Gaussian 1D: g(u) = exp(-u^2 / (2*sigma^2))                 │
│      - Tích chập 2 lượt O(2k) thay vì O(k^2) + Đệm biên Reflect 101     │
│      - PSNR = 69.54 dB so với cv2.GaussianBlur()                        │
│                                                                         │
│  [3] Phát hiện biên Canny 4 bước hoàn chỉnh (Canny Edge Detector)       │
│      - Bước 1: Gradient Sobel 3x3 L2 (Gx, Gy, M = sqrt(Gx^2 + Gy^2))    │
│      - Bước 2: Triệt tiêu không cực đại (NMS) lượng tử hóa 4 hướng góc  │
│      - Bước 3: Phân ngưỡng kép (Double Thresholding: T_low & T_high)    │
│      - Bước 4: Nối biên trễ 8-lân cận (Hysteresis, vector hóa)          │
│      - Độ trùng khớp: IoU = 99.02%, Dice = 0.9951                       │
│                                                                         │
│  [4] Biến đổi Hough đường thẳng cực & Xác định 4 đỉnh góc khung phiếu   │
│      - Không gian tham số cực: rho = x*cos(theta) + y*sin(theta)        │
│      - Bỏ phiếu vector hóa qua bincount & NMS trong không gian tham số  │
│      - Chấm điểm từng đường bằng số điểm biên thật nằm trên đường       │
│      - Giải hệ Cramer tìm giao điểm 2 cạnh dọc x 2 cạnh ngang           │
│      - Xuất 4 tọa độ chuẩn hóa [TL, TR, BR, BL]                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
[Bàn giao cho TV2: Tọa độ 4 góc [TL, TR, BR, BL] + Ảnh xám đã tiền xử lý]
```

---

## 🔬 2. CƠ SỞ LÝ THUYẾT VÀ CÀI ĐẶT THUẬT TOÁN 100% NUMPY (CODE FROM SCRATCH)

Toàn bộ thuật toán lõi được tự cài đặt từ đầu bằng công thức toán học và mảng đa chiều `NumPy`,
**tuyệt đối không sử dụng hàm xử lý có sẵn của `OpenCV` trong luồng xử lý chính**. `OpenCV` chỉ được
gọi độc lập trong bộ đo benchmark để đối chứng số học và thời gian thực thi.

### 2.1. Chuyển đổi mức xám Luminance ITU-R BT.601 (`tv1_grayscale_blur.py`)
- **Cơ sở toán học:** Thị giác con người nhạy cảm nhất với ánh sáng màu xanh lục ($G$), sau đó đến
  màu đỏ ($R$) và thấp nhất với màu xanh lam ($B$). Tổ hợp tuyến tính bảo toàn độ sáng cảm nhận:
  $$I_{\text{gray}}(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$
- **Tối ưu hóa:** Dùng `np.float64` trung gian chống tràn số, vector hóa trên mảng 3 chiều, sau đó
  `np.round` rồi ép kiểu `np.uint8`.
- **Độ chính xác:** Khớp chính xác từng điểm ảnh so với `cv2.cvtColor(BGR2GRAY)`
  ($\text{MSE} = 0.0000$, sai khác cực đại $0$ mức xám trên toàn bộ $800\,000$ pixel).

### 2.2. Bộ lọc Gaussian 2D & Tích chập phân tách (Separable 1D Convolution)
- **Cơ sở toán học:** Hàm mật độ xác suất Gaussian 2D đối xứng tâm, phân rã được thành tích ngoài
  của hai nhân 1D: $G(u, v) = g(u) \cdot g(v)$.
- **Kỹ thuật tối ưu hóa:**
  1. Tích chập theo hàng bằng $g(u)$, rồi theo cột bằng $g(v)$: $O(H \cdot W \cdot 2k)$ thay vì
     $O(H \cdot W \cdot k^2)$.
  2. Đệm biên phản xạ đối xứng chuẩn `BORDER_REFLECT_101` (`fedcba | abcdefgh | hgfedc`)
     bằng `np.pad(..., mode='reflect')`, triệt tiêu viền đen giả ở biên ảnh.
  3. Trích xuất cửa sổ trượt qua `np.lib.stride_tricks.sliding_window_view`.
- **Độ chính xác:** $\text{MSE} = 0.0072$, $\text{PSNR} = 69.54\text{ dB}$ so với `cv2.GaussianBlur()`
  (vượt tiêu chuẩn nghiệm thu $> 48.0\text{ dB}$). Sai khác cực đại chỉ 1 mức xám do quy ước làm tròn
  số thực (`round half to even` của NumPy vs `round half away from zero` của OpenCV).

### 2.3. Thuật toán Phát hiện biên Canny 4 bước hoàn chỉnh (`tv1_canny.py`)
1. **Bước 1 – Sobel $3 \times 3$ và Gradient L2:**
   $$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad
   K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
   $$M(x, y) = \sqrt{G_x^2 + G_y^2}, \quad \theta = \text{atan2}(G_y, G_x)$$
   $K_y$ dùng dấu ngược với một số giáo trình vì ở đây trục $y$ hướng xuống (quy ước ảnh số);
   cách này khớp 100% với `cv2.Sobel(..., cv2.CV_16S)`.
   *Cài đặt:* 9 lát (slice) của cửa sổ $3\times3$ được cộng trực tiếp thay cho
   `sliding_window_view` — kết quả trùng khít ($\text{sai số} = 0$) nhưng nhanh hơn ~2.4 lần.
2. **Bước 2 – Triệt tiêu không cực đại (NMS):** Lượng tử hóa góc thành 4 hướng
   ($0^\circ$, $45^\circ$, $90^\circ$, $135^\circ$) bằng so sánh tỉ lệ $|G_y|$ với
   $0.4142|G_x|$ và $2.4142|G_x|$; dùng bất đẳng thức nghiêm ngặt một bên
   ($m > n_1$ và $m \ge n_2$) để triệt tiêu biên kép trên sườn dốc phẳng.
3. **Bước 3 – Phân ngưỡng kép:** $M_{\text{NMS}} \ge T_{\text{high}} \Rightarrow$ biên mạnh;
   $T_{\text{low}} \le M_{\text{NMS}} < T_{\text{high}} \Rightarrow$ biên yếu; còn lại loại bỏ.
4. **Bước 4 – Nối biên trễ (Hysteresis):** Đây là phép **tái tạo hình thái học**
   (morphological reconstruction) trên mặt nạ biên yếu: lặp phép hợp 8 bản dịch của mặt nạ
   hiện tại rồi giao với mặt nạ yếu, dừng khi hội tụ. Kết quả **trùng khít** với duyệt BFS
   từng điểm (đã kiểm chứng bằng test) nhưng vector hóa hoàn toàn nên nhanh hơn ~10 lần.
- **Độ chính xác:** $\text{IoU} = 99.02\%$, $\text{Dice} = 0.9951$ so với `cv2.Canny(L2gradient=True)`.

### 2.4. Biến đổi Hough đường thẳng cực & Định vị 4 góc phiếu (`tv1_hough.py`)
- **Phương trình Hesse cực:** $\rho = x \cos\theta + y \sin\theta$ với $\theta \in [0, \pi)$
  và $\rho \in [-D, D]$ ($D = \sqrt{H^2 + W^2}$).
- **Bỏ phiếu vector hóa:** Chuyển toàn bộ điểm biên qua `np.outer`, tính chỉ số phẳng và tích lũy
  bằng `np.bincount`.
- **Khử cực đại phụ (`apply_parameter_space_nms`):** Gộp các đường có
  $\Delta\theta < 5^\circ$ và $\Delta\rho < 10\text{ px}$. Hàm này được tách riêng để **dùng lại
  cho cả kết quả `cv2.HoughLines`** khi đối chứng (xem mục 5).
- **Chấm điểm đường bằng support thật (`score_lines_by_edge_support`):** Mảng tích lũy chỉ đếm
  phiếu trong ô $(\rho, \theta)$ rời rạc nên hai đường gần nhau rất khó phân biệt. Hàm này đếm
  trực tiếp số pixel biên có khoảng cách tới đường $\le 1.5\text{ px}$ (chia lô để giới hạn bộ nhớ).
- **Dựng tứ giác từ Hough (`detect_quad_from_hough_lines`)** — chính là bước 4 của kế hoạch:
  1. Chấm điểm toàn bộ đường, tách họ **cạnh dọc** ($|\cos\theta| \ge \cos 20^\circ$) và
     **cạnh ngang** ($|\sin\theta| \ge \cos 20^\circ$).
  2. Trong mỗi họ chọn cặp đường có **tổng support lớn nhất** mà khoảng cách giữa chúng
     $\ge 50\%$ kích thước ảnh (tức 2 cạnh đối diện của khung phiếu).
  3. Tính 4 giao điểm bằng `intersect_lines_polar` (quy tắc Cramer:
     $x = \frac{\rho_1 \sin\theta_2 - \rho_2 \sin\theta_1}{\sin(\theta_2 - \theta_1)}$,
     $y = \frac{\rho_2 \cos\theta_1 - \rho_1 \cos\theta_2}{\sin(\theta_2 - \theta_1)}$).
  4. Kiểm tra hợp lệ hình học: tứ giác nằm trong ảnh và diện tích $\ge 30\%$ khung hình.
- **Dự phòng (`detect_quad_from_boundary_scratch`):** Phân ngưỡng Otsu → lấy **điểm biên cực trị
  theo hàng/cột** (`collect_boundary_extremes`, tối đa $2H + 2W$ điểm và **luôn chứa 4 đỉnh góc**)
  → bao lồi Andrew's Monotone Chain → **RDP trên đa giác khép kín** (`simplify_closed_polygon`,
  tách đôi tại đỉnh xa nhất) → 4 đỉnh.

### 2.5. Nhận biết "nền bàn tối" và chọn nhánh xử lý
Hàm `detect_sheet_corners` có hai nhánh, và việc chọn nhánh dựa trên **thống kê toàn dải viền**
chứ không dựa vào 4 pixel góc:

| Tiêu chí | Ý nghĩa | Ngưỡng |
| :--- | :--- | :--- |
| 1 | **Cả bốn** dải viền đều tối hơn ngưỡng Otsu | $\ge 20$ mức xám |
| 2 | **Mỗi** dải viền có tỉ lệ pixel thuộc lớp tối của Otsu | $\ge 60\%$ |
| 3 | Cạnh **sáng nhất** vẫn tối hơn trung vị vùng giữa ảnh | $< 60\%$ |

Chỉ cần **một** tiêu chí đúng là kết luận "có nền bàn tối". Ngưỡng Otsu được cài đặt từ đầu
(`otsu_threshold_scratch`, khớp `cv2.threshold(THRESH_OTSU)` trên cả 5 ảnh mẫu).

**Vì sao phải xét riêng từng phía thay vì gộp bốn dải làm một trung vị:** nền bàn tối thật sự bao
quanh phiếu thì **mọi phía** đều tối. Ảnh DS-05 là tờ phiếu phủ kín khung hình chỉ bị bóng đổ ở
phía dưới: dải trên sáng 245, ba dải còn lại tối 80. Nếu gộp chung, trung vị rơi vào vùng tối và
tờ phiếu bị cắt thành tứ giác thụt vào trong — **sai 67.19 px**. Yêu cầu cả bốn phía cùng tối loại
đúng trường hợp bóng đổ (DS-05: $67.19 \to 0.00$ px) mà vẫn nhận đúng DS-02 (cả 4 dải ≈ 45).

| Ảnh | Trung vị 4 dải (T/B/L/R) | Vùng giữa | Otsu | Kết luận |
| :--- | :--- | :---: | :---: | :--- |
| DS-01 Chuẩn | 245 / 245 / 245 / 245 | 245 | 141 | Không nền tối |
| DS-02 Nghiêng bàn tối | 45 / 45 / 45 / 45 | 245 | 144 | **Nền tối** |
| DS-03 Thiếu sáng | 142 / 142 / 142 / 142 | 142 | 86 | Không nền tối |
| DS-04 Tô sai + tẩy | 245 / 245 / 245 / 245 | 245 | 141 | Không nền tối |
| DS-05 Bóng đổ | **245 / 80 / 245 / 80** | 80 | 157 | Không nền tối |

---

## 📂 3. CẤU TRÚC THƯ MỤC MODULE

```
OMR_System_TV1/
├── README.md                      # Tài liệu kỹ thuật chi tiết của Thành viên 1
├── requirements.txt               # Thư viện phụ thuộc (numpy, opencv-python, matplotlib)
├── test_tv1.py                    # Script kiểm thử độc lập 6 bài test (tương tự test_tv2.py)
├── src/                           # Mã nguồn lõi 100% Code from Scratch bằng NumPy
│   ├── __init__.py                # Package init & re-export toàn bộ API
│   ├── tv1_grayscale_blur.py      # Chuyển mức xám Luminance & Lọc Gaussian 2D separable
│   ├── tv1_canny.py               # Canny 4 bước (Sobel, NMS, Double Thresh, Hysteresis)
│   ├── tv1_hough.py               # Hough cực, NMS tham số, Cramer, Convex Hull, RDP, Otsu
│   └── tv1_preprocessing.py       # Pipeline tích hợp TV1, detect_sheet_corners & Benchmark
├── tests/                         # Bộ kiểm thử tự động toàn diện (33 Unit Tests)
│   ├── __init__.py
│   ├── test_tv1_grayscale_blur.py # 5 tests: trọng số BT.601, tính chất kernel, PSNR > 48 dB
│   ├── test_tv1_canny.py          # 5 tests: Sobel Gx/Gy, NMS, Hysteresis, IoU >= 98%
│   ├── test_tv1_hough.py          # 11 tests: Hough, Cramer, hull, RDP (hở/khép kín), Otsu, support
│   └── test_tv1_benchmark.py      # 12 tests: nghiệm thu 5 thuật toán, PASS/FAIL, latency, nhánh DS-02, bóng đổ DS-05
├── reports/                       # Báo cáo kỹ thuật chi tiết
│   └── Bao_Cao_Thanh_Vien_1_OMR.md
└── outputs/                       # Thư mục xuất sản phẩm và ảnh trung gian
    └── tv1/
        ├── 1_input_sample.png          # Ảnh phiếu thi đầu vào
        ├── 2_grayscale.png             # Ảnh mức xám Luminance BT.601
        ├── 3_gaussian_blur.png         # Ảnh sau làm mịn Gaussian 2D
        ├── 4_sobel_gradient.png        # Bản đồ độ lớn Gradient Sobel L2
        ├── 5_nms.png                   # Bản đồ sau triệt tiêu không cực đại NMS
        ├── 6_canny_edges.png           # Ảnh biên nhị phân Canny hoàn chỉnh
        ├── 7_hough_accumulator.png     # Không gian tham số cực Hough (rho, theta)
        ├── 8_sheet_corners.png         # Ảnh gốc đánh dấu 4 đỉnh góc phiếu [TL, TR, BR, BL]
        ├── 9_triplet_comparison.png    # Triplet View: Scratch vs OpenCV vs Heatmap sai khác
        ├── benchmark_metrics.csv       # Bảng số liệu benchmark đối chứng
        └── benchmark_metrics.json      # Metadata chi tiết sai số và độ trễ
```

---

## 🚀 4. HƯỚNG DẪN CÀI ĐẶT & CHẠY THỬ NGHIỆM

### Bước 1: Cài đặt thư viện phụ thuộc
```powershell
cd OMR_System_TV1
pip install -r requirements.txt
```

### Bước 2: Chạy script kiểm thử CLI độc lập (6 bài test toàn diện)
```powershell
python test_tv1.py
```
*Kết quả:* cả 6 bài kiểm thử đều `[PASS]`, bao gồm cả kiểm chứng nhánh Canny + Hough trên ảnh
nghiêng DS-02.

### Bước 3: Chạy bộ kiểm thử tự động Unit Tests (33 Unit Tests)
```powershell
python -m unittest discover tests -v
```
*Kết quả:* **33/33 tests PASSED** trong $\approx 1.8\text{ s}$.

### Bước 4: Chạy Demo Pipeline & Xuất toàn bộ ảnh nghiệm thu ra `outputs/tv1/`
```powershell
python src/tv1_preprocessing.py
```

> **Ghi chú Windows:** nếu console báo `UnicodeEncodeError`, đặt `$env:PYTHONIOENCODING='utf-8'`
> trước khi chạy (lỗi này cũng đã gặp ở module TV2).

---

## 📊 5. KẾT QUẢ ĐỐI CHỨNG THỰC NGHIỆM (SCRATCH vs. OPENCV)

Số liệu dưới đây đo trên ảnh phiếu chuẩn $1000 \times 800$ (`01_B21DCCN001_Chuan.png`),
Python 3.14 / NumPy 2.5 / OpenCV 5.0, lấy **giá trị tốt nhất trong 3 lần chạy**:

| Module | Thuật toán Scratch (`NumPy`) | Hàm `OpenCV` đối chứng | Chỉ số đối chứng | Thời gian Scratch | Thời gian OpenCV | Đánh giá |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **TV1** | `rgb_to_gray_scratch()` | `cv2.cvtColor(BGR2GRAY)` | MSE $= 0.0000$, $\Delta I_{\max} = 0$ | $18.2\text{ ms}$ | $2.7\text{ ms}$ | **PASS** |
| **TV1** | `gaussian_blur_scratch()` | `cv2.GaussianBlur(5x5, 1.4)` | PSNR $= 69.54\text{ dB}$, $\Delta I_{\max} = 1$ | $38.8\text{ ms}$ | $0.8\text{ ms}$ | **PASS** |
| **TV1** | `canny_edge_detector()` | `cv2.Canny(L2gradient=True)` | IoU $= 99.02\%$, Dice $= 0.9951$ | $69.8\text{ ms}$ | $3.1\text{ ms}$ | **PASS** |
| **TV1** | `hough_lines_standard()` | `cv2.HoughLines(rho=1, theta=1°)$ + NMS | Recall $= 100.0\%$, Precision $= 91.2\%$ | $146.9\text{ ms}$ | $6.3\text{ ms}$ | **PASS** |
| **TV1** | `detect_sheet_corners()` | Contour Quad / `minAreaRect` | $E_{\text{corner}}$: $0.00\text{ px}$ (DS-01), $1.93\text{ px}$ (DS-02), $0.00\text{ px}$ (DS-05) | $23.6\text{ ms}$ | $4.5\text{ ms}$ | **PASS** |

### Nhận xét về phép đo Hough
Cách so sánh cũ (`|len(Scratch) − len(OpenCV)| ≤ 5`) là **sai về bản chất**: `cv2.HoughLines`
không khử cực đại phụ nên trả về 79 đường, trong khi `hough_lines_standard` đã NMS nên chỉ còn
34 đường. Sai khác số lượng không nói lên điều gì về độ chính xác. Do đó phép đo được thay bằng:

1. Chuẩn hóa danh sách của OpenCV bằng **cùng quy trình NMS** (`apply_parameter_space_nms`).
2. So khớp theo tham số hình học `(rho, theta)`, chuẩn hóa góc về $[0, \pi)$ để
   $(\rho, \theta)$ và $(-\rho, \theta + \pi)$ được coi là cùng một đường.
3. Báo cáo **Recall** (bao nhiêu % đường của OpenCV được Scratch tìm lại) và
   **Precision** (bao nhiêu % đường của Scratch có trong OpenCV), dung sai $2^\circ / 5\text{ px}$.

Kết quả: Recall $= 100.0\%$, Precision $= 91.2\%$ — tập đường chính trùng nhau hoàn toàn.

### Nhận xét về thời gian chạy
Tổng giai đoạn 1 (Scratch) trên 5 ảnh mẫu: **trung bình $208\text{ ms}$, lớn nhất $377\text{ ms}$**
(ảnh DS-02 vì phải chạy thêm Hough). So với trước khi tối ưu (1.1–1.7 s) đã giảm khoảng 4–6 lần nhờ:

- Sobel dùng slicing thay `sliding_window_view` (91 ms → 37 ms).
- Hysteresis vector hóa thay BFS từng pixel (95 ms → 9 ms).
- `detect_sheet_corners` nhận `precomputed_gray`/`precomputed_edges` để **không chạy lại Canny**.

---

## 🔗 6. GIAO TIẾP TÍCH HỢP HỆ THỐNG (INTEGRATION WITH TV2 & TV4)

Module TV1 cung cấp các API chuẩn hóa:

- **`rgb_to_gray_scratch(image, is_bgr=True)`** — TV4 gọi trong Stage 1 của `pipeline_coordinator.py`.
- **`gaussian_blur_scratch(image, kernel_size, sigma)`** — TV4 gọi đối chứng trong `benchmark_all_modules()`.
- **`canny_edge_scratch(image, low_threshold, high_threshold)`** — TV4 gọi đối chứng Canny.
- **`detect_sheet_corners(image, use_scratch=True, precomputed_gray=None, precomputed_edges=None)`**
  — trả về mảng `float32` shape $(4, 2)$ theo thứ tự `[TL, TR, BR, BL]`, chuyển tiếp trực tiếp vào
  `warp_perspective_sheet()` của TV2.

```python
# TV1 phát hiện góc (truyền gray/edges đã tính để tránh chạy lại Canny)
corners = detect_sheet_corners(raw_image, use_scratch=True)

# TV2 nắn thẳng phối cảnh
warped_sheet = warp_perspective_sheet(gray_image, corners, target_width=800, target_height=1000, use_scratch=True)
```

### Ranh giới trách nhiệm với TV2 (quan trọng)
TV1 **chỉ** trả về 4 góc hình học. Việc **sửa hướng phiếu** (0°/90°/180°/270°) thuộc về TV2
(`detect_markers_and_correct_rotation` + NCC). Code Review mục 1.2 đã chỉ ra kịch bản DS-02 xoay
90°/180° **chưa được xử lý ở pipeline end-to-end**; nguyên nhân nằm ở việc TV2 chưa được gọi trong
pipeline, và thiết kế phiếu hiện tại có 4 marker **giống nhau** nên không thể suy ra hướng xoay.
Đây là việc của TV2 + thiết kế phiếu, không phải của module TV1.

---

## ✅ 7. ĐỐI CHIẾU VỚI `Code_Review_OMR_Nhom4.md`

| Mục review | Nội dung Code Review | Trạng thái sau khi sửa |
| :--- | :--- | :--- |
| **1.2** | `detect_sheet_corners` chỉ phân ngưỡng cố định `gray >= 100`, không dùng Canny/Hough; `hough_lines_standard` chỉ chạy trong benchmark | **Đã sửa.** `detect_sheet_corners` chạy Canny + Hough + giao điểm Cramer; `detect_quad_from_hough_lines` là đường chính, phân ngưỡng + hull + RDP chỉ còn là dự phòng |
| **1.5** | `"status": "PASS" if ... else "PASS"` ở dòng 435 và 458 | **Đã sửa.** Cả hai nhánh `else` đổi thành `"FAIL"`; có test hồi quy `test_benchmark_status_can_fail` |
| **2.2** | RDP nhận đa giác khép kín nên luôn trả 2 điểm; lấy mẫu raster 3000 điểm có thể bỏ sót đỉnh góc | **Đã sửa.** `ramer_douglas_peucker` từ chối đa giác khép kín; thêm `simplify_closed_polygon` tách đôi tại đỉnh xa nhất; thay lấy mẫu raster bằng `collect_boundary_extremes` (min/max theo hàng và cột) |
| **2.3** | Kiểm tra "nền bàn tối" chỉ dựa vào 4 pixel góc, ngưỡng cố định 100 | **Đã sửa.** Dùng trung vị toàn dải viền + 3 tiêu chí (Otsu, tỉ lệ pixel tối, so với vùng giữa ảnh), xét **riêng từng phía**; có test với 4 pixel góc bị làm sáng. Quá trình nghiệm thu còn phát hiện thêm lỗi **bóng đổ một phía** (DS-05) và đã sửa: $67.19 \to 0.00$ px |
| **2.7** | TV1 chạy 1.1–1.7 s/phiếu; test fail vì vượt 1.5 s | **Đã sửa.** Còn $133\text{ ms}$ (DS-01) … $377\text{ ms}$ (DS-02); toàn bộ 33 test chạy trong 1.8 s |
| **3 (docstring)** | Tài liệu ghi miền $\theta$ của Hough là $[-\pi/2, \pi/2)$ trong khi code dùng $[0, \pi)$ | **Đã đồng bộ tài liệu** theo code ($[0, \pi)$), vì $[0,\pi)$ là quy ước Hesse chuẩn cho ảnh số |
| **3 (dọn repo)** | `.pyc` trong `__pycache__` của TV1 vẫn nằm trong git | Ngoài phạm vi module (việc của người quản lý repo) |

Các mục Code Review **không thuộc TV1** (1.1, 1.3, 1.4, 1.6, 2.1, 2.4, 2.5, 2.6, 2.8, 2.9) cần
Thành viên 2/3/4 xử lý. Trong đó mục **2.5 và 2.6** có ảnh hưởng tới TV1: TV4 đang giữ bản sao
`tv1_preprocessing.py` (bản rút gọn, không có nhánh Hough) và cả ba module đều dùng package tên
`src` — nếu chạy chung một tiến trình có thể nạp nhầm module. Khi hợp nhất cần dùng bản trong
`OMR_System_TV1/src/` làm nguồn duy nhất.

---

## ⚠️ 8. GIỚI HẠN ĐÃ BIẾT

1. **Hướng xoay phiếu không do TV1 quyết định.** TV1 trả 4 góc của khung phiếu; nếu phiếu bị xoay
   90°/180°, `detect_sheet_corners` vẫn trả tứ giác đúng hình học nhưng thứ tự `[TL, TR, BR, BL]`
   theo nghĩa "góc ảnh", không theo nghĩa "góc nội dung phiếu". Việc chuẩn hóa hướng cần NCC của TV2
   cộng với một marker khác biệt trên phiếu.
2. **Benchmark corner chỉ đo trên kịch bản phiếu chiếm toàn khung.** Với ảnh DS-01/DS-03/DS-04/DS-05
   ảnh mẫu, cả hai chế độ đều trả 4 góc khung hình nên $E_{\text{corner}} = 0$ là kết quả đúng theo
   thiết kế, không phải "làm đẹp số" — nhưng nó cũng có nghĩa phép đo này **không** kiểm chứng được
   độ chính xác hình học của nhánh Hough. Nhánh Hough chỉ được kiểm chứng trên DS-02
   ($1.93\text{ px}$) và trên ảnh tổng hợp trong unit test.
3. **Chưa có Ground Truth độc lập.** Sai số góc hiện so với baseline OpenCV, chưa so với nhãn do
   người đo. Code Review mục 2.9 cũng nêu repo thiếu dataset 60 ảnh thật.
4. **Hough chạy lại trong `detect_sheet_corners`** khi gọi độc lập (không truyền `precomputed_edges`)
   nên mất thêm ~150 ms. Trong pipeline thì không, vì đã truyền `edges` sẵn.
5. **PSNR của Gaussian (69.54 dB) chỉ đo độ trung thực cài đặt**, không đo khả năng khử nhiễu.
   Khảo sát khử nhiễu thật (thêm nhiễu Gauss $\sigma{=}18$ + muối tiêu 2% rồi đo với ảnh sạch) nằm ở
   `reports/Bao_Cao_Thanh_Vien_1_OMR.md` mục 5.1: $5\times5/\sigma{=}1.4$ giảm mảnh nhiễu rời từ 387
   xuống 17.
