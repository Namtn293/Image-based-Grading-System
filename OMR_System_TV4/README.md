# HỆ THỐNG CHẤM BÀI TRẮC NGHIỆM TỰ ĐỘNG - PHÂN HỆ THÀNH VIÊN 4

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**  
**Khoa Công nghệ Thông tin 1 – Bộ môn Khoa học Máy tính**  
**Môn học:** Xử Lý Ảnh (INT13146)  
**Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt  
**Phân hệ phụ trách:** Thành viên 4 – Tăng cường ảnh, Chấm điểm, Đánh giá chất lượng & Giao diện ứng dụng  
**Tỷ lệ đóng góp kỹ thuật:** 25% khối lượng đề tài  

---

## 1. NỘI DUNG KỸ THUẬT VÀ NHIỆM VỤ CỦA THÀNH VIÊN 4

Theo đề cương phân công BTL, Thành viên 4 đảm nhiệm các phần việc cốt lõi sau:

1. **Thuật toán Cân bằng lược đồ mức xám (Histogram Equalization):**
   - Tự cài đặt từ đầu bằng `NumPy` dựa trên hàm phân phối tích lũy xác suất thực tế (CDF - Cumulative Distribution Function).
   - Tăng cường độ tương phản cho các bài thi bị chụp trong điều kiện ánh sáng yếu, thiếu sáng (Kịch bản 3 và 5).
2. **Bộ lọc trung vị khử nhiễu (Median Filter 2D):**
   - Tự cài đặt thuật toán cửa sổ trượt $k \times k$ kết hợp cơ chế đệm biên đối xứng (Reflect Padding) để khử nhiễu xung muối tiêu (Salt-and-Pepper noise) mà vẫn bảo toàn sắc nét cạnh ô tròn.
3. **Module xử lý ma trận mức xám dạng File CSV:**
   - Đọc và xuất ma trận điểm ảnh mức xám dạng số học $[0 \dots 255]$ ra file `.csv`, phục vụ nghiệm thu chi tiết từng pixel theo yêu cầu của giảng viên tại Slide 10–12.
4. **Bộ đo đạc đối chứng định lượng (Quantitative Benchmark):**
   - Độc lập chạy các hàm thư viện chuẩn `OpenCV` song song với thuật toán tự cài đặt của cả 4 thành viên.
   - Tự động tính toán các chỉ số toán học: Sai số toàn phương trung bình (**MSE**), Tỷ số tín hiệu trên nhiễu đỉnh (**PSNR**), Độ tương đồng diện tích (**IoU/Dice**), Độ lệch ngưỡng $\Delta\tau$, và Thời gian thực thi (Latency).
5. **Logic chấm điểm & Xuất báo cáo Excel chuyên nghiệp:**
   - Thuật toán chấm điểm thang 10, phân loại chính xác các trạng thái: Đúng, Sai, Bỏ trống (Blank) và Phạm quy (Multiple marks).
   - Cơ chế phát hiện tự động cấu trúc số câu trên phiếu để cảnh báo nếu người dùng nạp nhầm phiếu 20 câu với đáp án 10 câu (hoặc ngược lại).
   - Tự động xuất bảng điểm tổng hợp ra file Excel `Ket_Qua_Cham_Thi.xlsx` với 2 sheet có định dạng màu trực quan.
6. **Điều phối tích hợp hệ thống & Giao diện Web tương tác:**
   - Xây dựng luồng điều phối tích hợp (`pipeline_coordinator.py`) kết nối trơn tru 4 phân hệ.
   - Xây dựng bảng điều khiển Web đa năng bằng Streamlit (`app.py` và gói `ui/`) phục vụ bảo vệ đồ án.

---

## 2. CƠ SỞ TOÁN HỌC CÁC THUẬT TOÁN TỰ CÀI ĐẶT

### 2.1. Cân bằng lược đồ mức xám (Histogram Equalization)
Cho ảnh mức xám $I \in [0, L-1]$ kích thước $M \times N$ (với $L = 256$):
1. **Tính mật độ xác suất mức xám:**
   $$p_r(r_k) = \frac{n_k}{M \cdot N}, \quad k = 0, 1, \dots, L-1$$
2. **Tính hàm phân phối tích lũy (CDF):**
   $$\text{CDF}(k) = \sum_{j=0}^{k} p_r(r_j)$$
3. **Ánh xạ mức xám chuẩn hóa (Cumulative Histogram Mapping):**
   $$s_k = T(r_k) = \text{round}\left( \frac{\text{CDF}(k) - \text{CDF}_{\min}}{1 - \text{CDF}_{\min}} \cdot (L - 1) \right)$$
   *Kết quả thực nghiệm trên ảnh kiểm thử: Sai số $\text{MSE} = 0.0$, $\text{PSNR} = 100.0\text{ dB}$, khớp tuyệt đối $100\%$ so với hàm `cv2.equalizeHist()`.*

### 2.2. Bộ lọc trung vị không gian (2D Median Filter)
Với cửa sổ lân cận kích thước $k \times k$ (với $k = 2r + 1$):
1. **Đệm biên đối xứng (Reflect Padding):**
   $$I_{\text{pad}}(x, y) = I(x', y')$$
2. **Tính giá trị trung vị trong lân cận:**
   $$I_{\text{out}}(x, y) = \text{median}\left( \{ I_{\text{pad}}(x + u, y + v) \mid -r \le u, v \le r \} \right)$$
   *Kết quả thực nghiệm: Loại bỏ triệt để $100\%$ các điểm nhiễu muối tiêu nhân tạo trên ma trận mẫu, chỉ số $\text{PSNR} = 100.0\text{ dB}$ trên vùng ảnh thực.*

### 2.3. Các chỉ số đo đạc đối chứng toán học
* **Sai số toàn phương trung bình (Mean Squared Error - MSE):**
  $$\text{MSE} = \frac{1}{M \cdot N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} \left( I_{\text{cài đặt}}(x, y) - I_{\text{OpenCV}}(x, y) \right)^2$$
* **Tỷ số tín hiệu cực đại trên nhiễu (Peak Signal-to-Noise Ratio - PSNR):**
  $$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{255^2}{\text{MSE}} \right) \quad (\text{dB})$$
* **Chỉ số tương đồng vùng nhị phân (Intersection over Union - IoU):**
  $$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{\sum (A \land B)}{\sum (A \lor B)}$$
* **Độ lệch ngưỡng Otsu ($\Delta\tau$):**
  $$\Delta\tau = |\tau_{\text{scratch}} - \tau_{\text{OpenCV}}|$$

---

## 3. CẤU TRÚC THƯ MỤC PHÂN HỆ 4

```
OMR_System_TV4/
├── app.py                      # Ứng dụng Web Streamlit (Controller & Layout Router)
├── run_demo.py                 # Script CLI kiểm thử & in bảng số liệu đối chứng
├── generate_samples.py         # Script sinh bộ dữ liệu mẫu (DS-01 .. DS-05 & CSV)
├── requirements.txt            # Danh sách thư viện phụ thuộc của dự án
├── README.md                   # Tài liệu kỹ thuật chi tiết của Phân hệ 4
├── data/                       # Dữ liệu kiểm thử
│   ├── answer_key.csv          # File đáp án chuẩn (tùy chỉnh số câu)
│   ├── sample_matrix.csv       # File ma trận mức xám 60x60
│   ├── sample_matrix_15x15.csv # File ma trận mức xám 15x15 có nhiễu muối tiêu
│   ├── Ket_Qua_Cham_Thi.xlsx   # File Excel kết quả chấm bài thi tổng hợp
│   └── samples/                # Bộ 5 ảnh phiếu thi trắc nghiệm mẫu (DS-01 .. DS-05)
├── src/                        # Thư mục mã nguồn thuật toán (Modular Architecture)
│   ├── __init__.py             # Package init & re-export
│   ├── tv4_filters.py          # Thuật toán Scratch: Median Filter & Histogram Equalization
│   ├── tv4_metrics.py          # Bộ đo đạc định lượng: MSE, PSNR, IoU, Dice, Delta-tau, Ablation
│   ├── tv4_grader.py           # Logic chấm điểm OMR, Ma trận CSV & Xuất báo cáo Excel
│   ├── tv4_enhancement_eval.py # Facade module (Đảm bảo tương thích ngược 100%)
│   └── pipeline_coordinator.py # Luồng điều phối tích hợp toàn hệ thống (TV1 -> TV4)
├── ui/                         # Giao diện Web Modular (Zero Raw Emojis, Pure Light SaaS)
│   ├── __init__.py             # Package UI init
│   ├── styles.py               # Hệ thống CSS Light SaaS & cấu hình Matplotlib
│   ├── tab1_grading.py         # Tab 1: Chấm bài trắc nghiệm & Giám sát Human-in-the-Loop
│   ├── tab2_stepper.py         # Tab 2: Stepper kiểm thử từng bước Pipeline liên hoàn
│   ├── tab3_benchmark.py       # Tab 3: Đối chứng Triplet View & Khảo sát Ablation
│   └── tab4_statistics.py      # Tab 4: Thống kê KPI, biểu đồ phổ điểm & tỷ lệ làm đúng
└── tests/                      # Bộ kiểm thử tự động của Module 4
    ├── __init__.py
    └── test_tv4_algorithms.py  # 20 Unit Tests kiểm thử toàn diện
```

---

## 4. HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG

### Bước 1: Cài đặt môi trường
```powershell
pip install -r requirements.txt
```

### Bước 2: Chạy bộ kiểm thử tự động (20 Unit Tests)
```powershell
python -m unittest discover -s tests -v
```
Toàn bộ 20 bài kiểm thử bao phủ toàn diện:
- Tính đúng đắn toán học của Cân bằng sáng CDF và Lọc trung vị.
- Đọc/ghi ma trận CSV dạng số học với các trường hợp dữ liệu bất thường.
- Logic tính điểm thang 10, phân loại ô tô, cảnh báo bỏ trống / tô phạm quy.
- Cơ chế tự động phát hiện lệch cấu trúc câu hỏi giữa ảnh phiếu thi và file đáp án.
- Kiểm thử tích hợp trọn vẹn luồng Pipeline qua 5 kịch bản ảnh thực tế.

### Bước 3: Chạy nghiệm thu CLI & xuất báo cáo Excel
```powershell
python run_demo.py
```
Hệ thống sẽ chạy và in trực tiếp ra màn hình terminal:
- Bảng đối chứng chỉ số MSE, PSNR, thời gian thực thi của cả 4 phân hệ.
- Bảng điểm chi tiết từng câu của sinh viên.
- Xuất file bảng điểm tổng hợp `data/Ket_Qua_Cham_Thi.xlsx`.

### Bước 4: Khởi chạy Giao diện Web tương tác
```powershell
streamlit run app.py
```
Truy cập địa chỉ `http://localhost:8501` trên trình duyệt:
- **Tab 1: Chấm Bài Trắc Nghiệm:** Chấm bài tự động, hiển thị ảnh trực quan, kiểm duyệt thủ công và tải file Excel.
- **Tab 2: Kiểm Thử Thuật Toán Từng Phân Hệ:** Xem trước ảnh/ma trận số học trước và sau xử lý, tải file CSV kết quả.
- **Tab 3: Đối Chứng Thuật Toán vs OpenCV:** So sánh trực quan và định lượng toán học (MSE, PSNR, IoU) với OpenCV.
- **Tab 4: Thống Kê & Phổ Điểm:** Biểu đồ phân bố điểm số và tỷ lệ đúng từng câu hỏi.
