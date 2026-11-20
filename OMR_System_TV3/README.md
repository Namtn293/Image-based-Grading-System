# MODULE THÀNH VIÊN 3: PHÂN ĐOẠN ẢNH & NHẬN DIỆN Ô TÔ ĐÁP ÁN (OMR STAGE 3)

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**  
**Khoa Công nghệ Thông tin 1 – Bộ môn Khoa học Máy tính**  
**Môn học:** Xử Lý Ảnh (INT13146)  
**Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt  
**Đồ án:** Nhóm 4 – Hệ thống chấm bài trắc nghiệm tự động từ ảnh chụp (OMR System)  
**Thành viên phụ trách:** **Thành viên 3 (Phân đoạn ảnh & Nhận diện ô tô)**  
**Tỷ lệ đóng góp kỹ thuật:** **25%** (Cài đặt thuật toán 100% Code from Scratch bằng `NumPy`)  

---

## 📌 1. TỔNG QUAN VÀ VỊ TRÍ TRONG HỆ THỐNG

Trong kiến trúc Pipeline 4 giai đoạn của toàn hệ thống OMR:
1. **Giai đoạn 1 (TV1):** Tiền xử lý & Phát hiện 4 góc phiếu thi (Grayscale, Gaussian, Canny, Hough Transform).
2. **Giai đoạn 2 (TV2):** Nắn thẳng phối cảnh & Định vị Marker (Homography SVD, Bilinear Inverse Warping, NCC Template Matching).
3. **Giai đoạn 3 (TV3 - Module này):** **Phân đoạn ảnh & Nhận diện ô tô đáp án** (Otsu, Adaptive Mean via Integral Image, Morphology, Two-Pass CCL, Circular ROI, 4-State Classification).
4. **Giai đoạn 4 (TV4):** Tăng cường ảnh, Chấm điểm & Giao diện Demo (Histogram Equalization, Median Filter, Grading, Excel Report, Streamlit Web App).

```
[Đầu vào: Ảnh xám phiếu thi đã nắn phẳng 800x1000 từ TV2]
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              MODULE THÀNH VIÊN 3 (TV3)                   │
│                                                          │
│  [1] Phân ngưỡng nhị phân tự động (Thresholding)         │
│      - Otsu toàn cục (Tối đa hóa phương sai liên lớp)    │
│      - Adaptive Mean cục bộ qua Integral Image O(1)      │
│                                                          │
│  [2] Xử lý hình thái học nhị phân (Morphology)           │
│      - Opening (Co trước Giãn sau: xóa bụi chì, nhiễu)   │
│      - Closing (Giãn trước Co sau: lấp lỗ hổng vết tô)   │
│                                                          │
│  [3] Gán nhãn thành phần liên thông (Two-Pass CCL)       │
│      - Quét 2 lượt kết hợp cây cấu trúc Disjoint-Set     │
│      - Trích xuất bounding box, diện tích & tâm ô tô     │
│                                                          │
│  [4] Trích xuất ROI & Nhận diện ô tô (Classification)    │
│      - Circular Mask bán kính R chuẩn hóa                │
│      - Tỷ lệ lấp đầy: tau = sum(ROI * Mask) / (pi * R^2) │
│      - Phân loại: VALID, BLANK, ERASED_VALID, MULTIPLE   │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
[Đầu ra: Vector đáp án nhận diện {1: 'A', ...} + Bảng tỷ lệ tau + Ảnh chú thích]
```

---

## 🔬 2. CƠ SỞ LÝ THUYẾT VÀ THUẬT TOÁN 100% NUMPY (CODE FROM SCRATCH)

Toàn bộ thuật toán lõi được tự cài đặt từ đầu bằng công thức toán học và mảng đa chiều `NumPy`, **tuyệt đối không sử dụng hàm xử lý có sẵn của `OpenCV` trong luồng xử lý chính**. `OpenCV` chỉ được gọi độc lập trong bộ đo benchmark để đối chứng số học và thời gian thực thi.

### 2.1. Phân ngưỡng tự động Otsu toàn cục (`tv3_thresholding.py`)
- **Nguyên lý:** Quét qua 256 mức xám $t \in [0, 255]$ để tìm ngưỡng $T^*$ tối đa hóa phương sai giữa hai lớp (Between-class Variance):
  $$\sigma_B^2(t) = \omega_0(t) \cdot \omega_1(t) \cdot [\mu_0(t) - \mu_1(t)]^2$$
- **Vector hóa:** Tích lũy xác suất và giá trị trung bình qua `np.cumsum` đạt độ phức tạp $O(L)$ với $L=256$, thời gian thực thi $< 1 \text{ ms}$.
- **Sai khác với OpenCV:** $\Delta T = |T_{\text{scratch}} - T_{\text{cv}}| = 0$ mức xám.

### 2.2. Bảng ảnh tích phân (Integral Image) & Phân ngưỡng thích nghi cục bộ
- **Nguyên lý:** Phân ngưỡng thích nghi cục bộ giải quyết bài toán chiếu sáng không đều và bóng đổ xiên (kịch bản DS-03):
  $$T(x, y) = \mu_{W}(x, y) - C$$
- **Tối ưu hóa $O(1)$:** Xây dựng bảng ảnh tích phân $II(x, y) = \sum_{r \le y, c \le x} I(r, c)$ bằng `np.cumsum`, tính tổng mức xám của bất kỳ cửa sổ $b \times b$ nào chỉ với 4 phép truy xuất:
  $$\Sigma = II(y_2, x_2) - II(y_1-1, x_2) - II(y_2, x_1-1) + II(y_1-1, x_1-1)$$
- **Độ chính xác:** Khớp 100% so với `cv2.adaptiveThreshold` ($IoU = 100.0\%$).

### 2.3. Bốn phép toán Hình thái học nhị phân (`tv3_morphology_ccl.py`)
- Tự sinh phần tử cấu trúc dạng Chữ nhật (`MORPH_RECT`), Chữ thập (`MORPH_CROSS`), Ellipse (`MORPH_ELLIPSE`).
- **Co nhị phân (Erosion):** Điểm ảnh giữ mức 255 khi toàn bộ kernel nằm trọn trong vùng foreground.
- **Giãn nhị phân (Dilation):** Điểm ảnh chuyển thành 255 nếu ít nhất một phần tử kernel giao với foreground.
- **Phép Mở (Opening):** $A \circ B = (A \ominus B) \oplus B$ (Khử nhiễu hạt, bụi chì, viền rách).
- **Phép Đóng (Closing):** $A \bullet B = (A \oplus B) \ominus B$ (Nối liền nét đứt, hàn gắn lỗ rỗng trong vết chì tô).
- **Độ chính xác:** Khớp 100% ma trận ảnh so với `cv2.morphologyEx` ($IoU = 100.0\%$, $PSNR = 999 \text{ dB}$).

### 2.4. Gán nhãn thành phần liên thông 2 lượt (Two-Pass CCL)
- Hỗ trợ liên thông 4 lân cận (4-connectivity) và 8 lân cận (8-connectivity).
- **Lượt 1 (First Pass):** Quét từ trên xuống dưới, trái sang phải; gán nhãn tạm thời và lưu quan hệ tương đương vào cấu trúc rừng cây rời rạc (`DisjointSet` với Path Compression).
- **Lượt 2 (Second Pass):** Chuẩn hóa toàn bộ nhãn về gốc đại diện duy nhất (`find()`).
- Trích xuất tự động: Số lượng liên thông, tọa độ Bounding Box $[x, y, w, h]$, diện tích và tọa độ tâm $(\bar{x}, \bar{y})$.
- **Độ chính xác:** Khớp chính xác 100% số lượng thành phần liên thông so với `cv2.connectedComponentsWithStats`.

### 2.5. Trích xuất Circular ROI & Bộ phân loại 4 trạng thái (`tv3_roi_classifier.py`)
- Tạo Circular Mask số học: $\mathcal{M}(u, v) = \mathbb{I}\left(u^2 + v^2 \le R_{\text{inner}}^2\right)$.
- Tính tỷ lệ lấp đầy:
  $$\tau_{q, c} = \frac{\sum_{(u, v)} \text{ROI}_{q, c}(u, v) \cdot \mathcal{M}(u, v)}{255 \cdot \sum \mathcal{M}(u, v)}$$
- **Phân loại 4 trạng thái OMR chuẩn:**
  1. `VALID`: Có đúng 1 ô đạt $\tau \ge \tau_{\min}$ và vượt trội so với ô kế tiếp $\Delta \tau \ge \Delta \tau_{\min}$.
  2. `BLANK`: Toàn bộ các lựa chọn đều có $\tau < \tau_{\min}$ (Thí sinh bỏ trống).
  3. `ERASED_VALID`: Thí sinh tẩy xóa chọn lại (1 ô mờ $\tau_{\text{erased}} \in [0.35, 0.50]$ và 1 ô tô đè $\tau_{\text{active}} \ge 0.70$).
  4. `MULTIPLE_INVALID`: Tô đúp phạm quy (2 ô trở lên cùng vượt ngưỡng mà không có ô nổi trội).

---

## 📂 3. CẤU TRÚC THƯ MỤC MODULE

```
OMR_System_TV3/
├── README.md                      # Tài liệu kỹ thuật chi tiết của Thành viên 3
├── requirements.txt               # Thư viện phụ thuộc (numpy, opencv-python, pytest)
├── .gitignore                     # Bỏ qua cache và file tạm
├── Ke_Hoach_De_Tai_OMR_Nhom4.md   # Đề cương chi tiết & Kế hoạch kỹ thuật
├── src/                           # Mã nguồn chính (100% Code from Scratch bằng NumPy)
│   ├── __init__.py                # Package init & re-export toàn bộ API
│   ├── tv3_thresholding.py        # Otsu & Adaptive Mean via Integral Image
│   ├── tv3_morphology_ccl.py      # Morphology (Erode, Dilate, Open, Close) & Two-Pass CCL
│   ├── tv3_roi_classifier.py      # Circular ROI, Fill Ratio tau & 4-State Classifier
│   └── tv3_segmentation.py        # Pipeline tích hợp, Demo CLI & Adapter cho TV4
├── tests/                         # Bộ kiểm thử tự động toàn diện (85 Unit Tests)
│   ├── __init__.py
│   ├── test_tv3_thresholding.py   # Kiểm thử Otsu, Integral Image & Adaptive Mean
│   ├── test_tv3_morphology_ccl.py # Kiểm thử Structuring Element, Morphology & CCL
│   ├── test_tv3_roi_classifier.py # Kiểm thử ROI, Mask tròn & logic 4 trạng thái
│   └── test_tv3_benchmark.py      # Kiểm thử Pipeline, Acceptance Criteria & Benchmark
├── reports/                       # Báo cáo nghiệm thu kỹ thuật
│   └── Bao_Cao_Thanh_Vien_3_OMR.md# Báo cáo kỹ thuật chi tiết 26KB
└── outputs/                       # Thư mục xuất sản phẩm và ảnh trực quan
    └── tv3/
        ├── 1_input_warped.png          # Ảnh xám phiếu thi đầu vào
        ├── 2_thresholded.png           # Ảnh sau phân ngưỡng nhị phân
        ├── 3_morphology.png            # Ảnh sau phép Mở & Đóng
        ├── 4_ccl_labels.png            # Bản đồ nhãn màu vùng liên thông CCL
        ├── 5_omr_annotated.png         # Ảnh phiếu thi khoanh tròn và chú thích đáp án
        ├── 6_triplet_comparison.png    # Triplet View: Scratch vs OpenCV vs Heatmap
        ├── benchmark_metrics.csv       # Bảng số liệu benchmark đối chứng
        ├── benchmark_metrics.json      # Metadata chi tiết thời gian và sai số
        ├── recognition_results.csv     # Bảng kết quả nhận diện 20 câu
        └── recognition_results.json    # Chi tiết fill ratio và trạng thái 20 câu
```

---

## 🚀 4. HƯỚNG DẪN CÀI ĐẶT & CHẠY THỬ NGHIỆM

### Bước 1: Cài đặt thư viện phụ thuộc
```powershell
cd OMR_System_TV3
pip install -r requirements.txt
```

### Bước 2: Chạy bộ kiểm thử tự động (85 Unit Tests)
```powershell
pytest -v
```
Kết quả kiểm thử đạt **85/85 PASSED (100%)**.

### Bước 3: Chạy script Demo & Xuất toàn bộ ảnh nghiệm thu
```powershell
python src/tv3_segmentation.py
```
Script sẽ tự động:
1. Sinh ảnh phiếu thi mô phỏng thực tế 20 câu (gồm câu tô chuẩn, bỏ trống, tẩy xóa, tô đúp, có bóng đổ).
2. Chạy toàn bộ pipeline phân đoạn và nhận diện bằng thuật toán Scratch.
3. Xuất 6 ảnh minh họa các bước trung gian vào `outputs/tv3/`.
4. Xuất 2 file kết quả định dạng CSV và JSON (`benchmark_metrics.csv`, `recognition_results.csv`).
5. In bảng kết quả chi tiết từng câu và các chỉ số đo đạc đối chứng lên màn hình console.

---

## 📊 5. KẾT QUẢ ĐỊNH LƯỢNG & BENCHMARK SO VỚI OPENCV

| Module kỹ thuật | Thuật toán Scratch (NumPy) | Thuật toán OpenCV đối chứng | Chỉ số tương đồng | Sai số / Đánh giá | Trạng thái nghiệm thu |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Phân ngưỡng toàn cục** | `otsu_threshold_scratch` | `cv2.threshold(THRESH_OTSU)` | $\Delta T = \|T_{\text{scratch}} - T_{\text{cv}}\|$ | $\Delta T = 0$ mức xám | **ĐẠT XUẤT SẮC** |
| **Phân ngưỡng thích nghi** | `adaptive_mean_threshold_scratch` | `cv2.adaptiveThreshold` | IoU, Dice, PSNR | $\text{IoU} = 100.0\%$, $\text{PSNR} = 999 \text{ dB}$ | **ĐẠT XUẤT SẮC** |
| **Hình thái học Co/Giãn** | `erode_binary_scratch`, `dilate` | `cv2.erode`, `cv2.dilate` | IoU, Dice | $\text{IoU} = 100.0\%$, $\text{Dice} = 100.0\%$ | **ĐẠT XUẤT SẮC** |
| **Hình thái học Mở/Đóng** | `opening_binary`, `closing_binary` | `cv2.morphologyEx` | IoU, PSNR | $\text{IoU} = 100.0\%$, $\text{PSNR} = 999 \text{ dB}$ | **ĐẠT XUẤT SẮC** |
| **Gán nhãn liên thông** | `two_pass_ccl_scratch` | `cv2.connectedComponentsWithStats` | Số lượng liên thông $N$ | Khớp 100% ($62/62$ components) | **ĐẠT XUẤT SẮC** |
| **Nhận diện đáp án OMR** | `TV3SegmentationPipeline` | Ground Truth 20 câu | Accuracy (%) | **100.00%** (20/20 câu) | **ĐẠT XUẤT SẮC** |

### Khảo sát phân loại 4 trạng thái trên 20 câu hỏi thực nghiệm:
- **Tô chuẩn (`VALID`):** 15 câu (Nhận diện chính xác 100%).
- **Bỏ trống (`BLANK`):** 2 câu (Nhận diện đúng trạng thái `BLANK`, không báo lỗi).
- **Tẩy xóa chọn lại (`ERASED_VALID`):** 2 câu (Phân biệt chính xác ô tẩy mờ $\tau \approx 0.42$ và ô chọn lại $\tau \approx 0.85$).
- **Tô đúp phạm quy (`MULTIPLE_INVALID`):** 1 câu (Phát hiện chính xác thí sinh tô 2 ô cùng lúc).

---

## 🔗 6. GIAO TIẾP TÍCH HỢP HỆ THỐNG (API CHO TV2 & TV4)

Module Thành viên 3 cung cấp hàm tích hợp chuẩn `extract_bubble_answers(...)` để ghép nối trực tiếp với Pipeline Coordinator của TV4:

```python
from src.tv3_segmentation import extract_bubble_answers

# Nhận ảnh phiếu xám đã nắn phẳng từ Thành viên 2:
answers, fill_ratios, vis_warped = extract_bubble_answers(
    warped_gray=warped_sheet_from_tv2,
    total_questions=20,
    fill_threshold=0.38,
    use_scratch=True
)

# answers: Dict[int, str] ví dụ {1: 'A', 2: 'B', ..., 5: 'BLANK', 15: 'MULTIPLE'}
# fill_ratios: Dict[int, Dict[str, float]]
# vis_warped: np.ndarray (Ảnh BGR kèm vòng tròn xanh lá/đỏ chú thích)
```
