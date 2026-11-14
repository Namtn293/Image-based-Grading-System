# HỆ THỐNG CHẤM BÀI THI TRẮC NGHIỆM TỰ ĐỘNG (OMR VISION DASHBOARD)

**Hệ Thống Nhận Diện & Đánh Giá Thị Giác Máy Tính Tự Động**  
**Sản phẩm hoàn thiện:** Module 4 - Tăng cường ảnh, Chấm điểm, Benchmark & Dashboard Quản trị  

---

## 📌 1. KIẾN TRÚC HỆ THỐNG 4 GIAI ĐOẠN (PIPELINE ARCHITECTURE)
Hệ thống được thiết kế theo mô hình xử lý ảnh nối tầng 4 giai đoạn độc lập:

1. **Giai đoạn 1 (Tiền xử lý & Trích xuất biên):**
   - Chuyển đổi ảnh mức xám, làm mịn Gaussian (`gaussian_blur_scratch`).
   - Tính Gradient Sobel, triệt tiêu cực đại cục bộ NMS, ngưỡng kép và liên kết biên Hysteresis (`canny_edge_detector_scratch`).
2. **Giai đoạn 2 (Định vị & Nắn thẳng phối cảnh):**
   - Phát hiện 4 điểm neo fiducial góc phiếu bằng khớp mẫu chuẩn hóa tương quan chéo (`match_template_ncc_scratch`).
   - Ước lượng ma trận Homography bằng phép phân tích suy biến SVD (`find_homography_svd_scratch`) và nội suy song tuyến (`warp_perspective_scratch`).
3. **Giai đoạn 3 (Phân ngưỡng & Trích xuất Bubble):**
   - Phân ngưỡng tự động Otsu toàn cục (`otsu_threshold_scratch`).
   - Phân ngưỡng thích nghi cục bộ bằng ảnh tích lũy Integral Image (`adaptive_mean_threshold_scratch`).
   - Phép toán hình thái học Opening/Closing khử nhiễu (`morphology_open_scratch`, `morphology_close_scratch`).
4. **Giai đoạn 4 (Tăng cường ảnh, Chấm điểm, Benchmark & Giao diện):**
   - Cân bằng biểu đồ mức xám (`histogram_equalization_scratch`) và lọc trung vị 2D (`median_filter_scratch`, `_pad_2d_scratch`).
   - Đọc/xuất ma trận xám dạng file CSV phục vụ kiểm thử dữ liệu số học.
   - Bộ đo Benchmark định lượng đối chứng OpenCV (MSE, PSNR, IoU, Dice, $\Delta T$, Latency) & Ablation Study.
   - Logic chấm điểm thang 10, phân loại lỗi tô mờ/tô phạm quy và xuất báo cáo Excel 2 sheet định dạng màu trực quan.
   - Điều phối toàn bộ Pipeline (`pipeline_coordinator.py`) và Dashboard Web tương tác (`app.py`).

---

## 📂 2. CẤU TRÚC THƯ MỤC DỰ ÁN
```
OMR_System_TV4/
├── app.py                      # Giao diện Web Streamlit 4 Tab hoàn chỉnh
├── run_demo.py                 # Script CLI kiểm thử & in bảng số liệu benchmark
├── generate_samples.py         # Script sinh bộ dữ liệu mẫu (phiếu thi + ma trận CSV)
├── requirements.txt            # Danh sách thư viện phụ thuộc của dự án
├── .gitignore                  # Cấu hình bỏ qua file rác / cache của Git
├── README.md                   # Tài liệu hướng dẫn sử dụng & cơ sở lý thuyết
├── data/                       # Thư mục dữ liệu kiểm thử
│   ├── answer_key.csv          # File đáp án chuẩn 20 câu hỏi
│   ├── sample_matrix.csv       # File ma trận mức xám phục vụ kiểm chuẩn CSV
│   └── samples/                # Bộ 5 ảnh phiếu thi trắc nghiệm mẫu (DS-01 .. DS-05)
├── src/                        # Thư mục mã nguồn chính của Module 4
│   ├── __init__.py
│   ├── tv4_enhancement_eval.py # Module 4: Tăng cường ảnh, Chấm điểm, Benchmark & Excel
│   └── pipeline_coordinator.py # Luồng điều phối tích hợp toàn hệ thống
└── tests/                      # Bộ kiểm thử tự động của Module 4
    ├── __init__.py
    └── test_tv4_algorithms.py  # 15 Unit Tests kiểm thử toàn diện thuật toán TV4
```

---

## 🚀 3. HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN

### Bước 1: Cài đặt thư viện phụ thuộc
```powershell
pip install -r requirements.txt
```

### Bước 2: Chạy kiểm thử tự động (Unit Tests)
```powershell
python -m unittest discover tests -v
```

### Bước 3: Chạy demo kiểm thử CLI và xuất bảng điểm Excel
```powershell
python run_demo.py
```
*Kết quả sẽ in đầy đủ bảng số liệu MSE, PSNR, thời gian thực thi, điểm số bài thi mẫu và sinh file `data/Ket_Qua_Cham_Thi.xlsx`.*

### Bước 4: Khởi chạy Giao diện Web tương tác
```powershell
streamlit run app.py
```
*Trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8501`.*

---

## 🔬 4. KẾT QUẢ ĐỐI CHỨNG VỚI OPENCV (BENCHMARK)

| Thuật toán | Thời gian Scratch (NumPy) | Thời gian OpenCV (C++) | Sai số MSE | PSNR (dB) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cân bằng Histogram** | $\approx 3.8\text{ ms}$ | $\approx 0.6\text{ ms}$ | **$0.0000$** | **$100.0\text{ dB}$** | ✅ Khớp tuyệt đối công thức OpenCV |
| **Lọc trung vị ($3 \times 3$)** | $\approx 8.6\text{ ms}$ | $\approx 0.2\text{ ms}$ | **$0.0000$** | **$100.0\text{ dB}$** | ✅ Khớp tuyệt đối vùng ảnh thực |
| **I/O Ma trận CSV** | $\approx 1.2\text{ ms}$ | — | **$0.0000$** | **$\infty$** | ✅ Khớp ma trận mức xám chuẩn xác |
| **Chấm điểm Pipeline (đo thực)** | $\approx 320 - 450\text{ ms}$ | $\approx 8 - 14\text{ ms}$ | — | — | ✅ Tối ưu hiệu năng ($< 0.5\text{s}$/phiếu, tự chuẩn hóa $H=1000$) |
