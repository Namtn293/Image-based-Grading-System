# HỆ THỐNG CHẤM BÀI TRẮC NGHIỆM TỰ ĐỘNG TỪ ẢNH CHỤP (OMR SYSTEM) - NHÓM 4

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**  
**Khoa Công nghệ Thông tin 1 – Bộ môn Khoa học Máy tính**  
**Môn học:** Xử Lý Ảnh (INT13146)  
**Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt  
**Đồ án:** Nhóm 4 (Quy mô: 04 sinh viên, tỷ lệ đóng góp kỹ thuật: 25%/thành viên)  

---

## 📌 1. TỔNG QUAN DỰ ÁN & KIẾN TRÚC PIPELINE 4 GIAI ĐOẠN

Dự án xây dựng hệ thống nhận diện và chấm bài trắc nghiệm tự động từ ảnh chụp camera điện thoại (hoặc ma trận mức xám dạng `.csv`). Toàn bộ thuật toán xử lý ảnh trong luồng chính được **cài đặt từ đầu (code from scratch) bằng `NumPy`**, tuyệt đối không sử dụng các hàm xử lý có sẵn của `OpenCV`. Thư viện `OpenCV` (`cv2`) chỉ được sử dụng độc lập để so sánh đối chứng số học (MSE, PSNR, IoU) và thời gian thực thi (Latency).

```
[Đầu vào: Ảnh chụp phiếu thi (.jpg/.png) hoặc Ma trận mức xám (.csv)]
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1 (THÀNH VIÊN 1): Tiền xử lý & Phát hiện khung phiếu thi      │
│  - Chuyển mức xám Luminance (ITU-R BT.601)                              │
│  - Bộ lọc Gaussian 2D (Tích chập 2D Spatial Convolution)                │
│  - Phát hiện biên Canny 4 bước (Sobel -> NMS -> Double Thresh -> Hyst)  │
│  - Biến đổi Hough (Hough Line Transform) xác định 4 góc phiếu          │
└─────────────────────────────────────────────────────────────────────────┘
          │ (Tọa độ 4 đỉnh khung phiếu: TL, TR, BR, BL)
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2 (THÀNH VIÊN 2): Nắn thẳng phối cảnh & Định vị Marker        │
│  - Chuẩn hóa thứ tự 4 đỉnh & ước lượng Homography 3x3 bằng SVD         │
│  - Ánh xạ ngược (Inverse Warping) & Nội suy song tuyến (Bilinear)       │
│  - So khớp mẫu NCC (Normalized Cross-Correlation) định vị 4 Marker góc  │
└─────────────────────────────────────────────────────────────────────────┘
          │ (Ảnh phiếu đã nắn phẳng 800x1000 chuẩn hóa)
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3 (THÀNH VIÊN 3): Phân đoạn ảnh & Nhận diện ô tô đáp án       │
│  - Phân ngưỡng tự động Otsu toàn cục & Phân ngưỡng thích nghi cục bộ    │
│  - Toán tử Hình thái học: Erosion, Dilation, Opening, Closing           │
│  - Gán nhãn thành phần liên thông 2 lượt (Two-Pass CCL)                 │
│  - Trích xuất ROI từng ô A/B/C/D & phân loại theo tỷ lệ lấp đầy (Fill)  │
└─────────────────────────────────────────────────────────────────────────┘
          │ (Vector đáp án nhận diện + Trạng thái Đúng/Sai/Trống/Phạm quy)
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 4 (THÀNH VIÊN 4): Tăng cường ảnh, Chấm điểm & Giao diện Demo  │
│  - Cân bằng biểu đồ mức xám (Histogram Equalization dùng CDF chuẩn)      │
│  - Bộ lọc trung vị (Median Filter 2D) khử nhiễu xung muối tiêu          │
│  - Module I/O ma trận CSV phục vụ nghiệm thu theo Slide 10–12           │
│  - Đối chiếu đáp án chuẩn, chấm điểm thang 10 & xuất báo cáo Excel XLSX │
│  - Bộ đo Benchmark định lượng (MSE, PSNR, IoU, Latency) vs OpenCV       │
│  - Tích hợp Pipeline Coordinator & Giao diện Web tương tác Streamlit    │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
[Đầu ra: Ảnh chấm trực quan + Báo cáo Benchmark + File Excel Ket_Qua_Cham_Thi.xlsx]
```

---

## 📂 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```
Image-based-Grading-System/
├── Ke_Hoach_De_Tai_OMR_Nhom4.md     # Đề cương chi tiết & Kế hoạch đồ án Nhóm 4
├── README.md                        # Hướng dẫn tổng quan toàn dự án
├── OMR_System_TV2/                  # Không gian làm việc & Module của Thành viên 2 (Nắn phối cảnh & Marker)
│   ├── README.md                    # Hướng dẫn kỹ thuật TV2
│   ├── tv2_perspective.py           # Thuật toán Homography SVD, Bilinear Warping & NCC
│   └── test_tv2.py                  # Bộ kiểm thử TV2
├── OMR_System_TV3/                  # Không gian làm việc & Module của Thành viên 3 (Phân đoạn & Ô tô)
│   ├── README.md                    # Hướng dẫn kỹ thuật TV3
│   ├── requirements.txt             # Thư viện phụ thuộc
│   ├── src/                         # Mã nguồn Scratch (Otsu, Adaptive Mean, Morphology, CCL, Classifier)
│   ├── tests/                       # 69 Unit Tests kiểm thử toàn diện
│   ├── reports/                     # Báo cáo kỹ thuật chi tiết
│   └── outputs/tv3/                 # Ảnh các bước xử lý trung gian & file kết quả CSV/JSON
└── OMR_System_TV4/                  # Không gian làm việc & Module của Thành viên 4
    ├── app.py                       # Giao diện Web tương tác Streamlit (4 Tab đầy đủ)
    ├── run_demo.py                  # Script CLI kiểm thử & in bảng số liệu benchmark
    ├── generate_samples.py          # Script sinh bộ dữ liệu mẫu (DS-01 .. DS-05 & CSV)
    ├── requirements.txt             # Thư viện phụ thuộc của dự án
    ├── README.md                    # Tài liệu kỹ thuật chi tiết của Thành viên 4
    ├── data/                        # Dữ liệu kiểm thử
    │   ├── answer_key.csv           # Đáp án chuẩn 20 câu hỏi
    │   ├── sample_matrix.csv        # Ma trận mức xám phục vụ nghiệm thu Slide 10–12
    │   ├── Ket_Qua_Cham_Thi.xlsx    # Bảng điểm tổng hợp xuất ra Excel (2 sheet có màu)
    │   └── samples/                 # 5 kịch bản ảnh phiếu thi mẫu (DS-01 .. DS-05)
    ├── src/                         # Mã nguồn Module Thành viên 4
    │   ├── tv4_enhancement_eval.py  # Thuật toán TV4 (Hist Eq, Median, Benchmark, Excel)
    │   └── pipeline_coordinator.py  # Luồng điều phối tích hợp toàn hệ thống
    └── tests/                       # Bộ kiểm thử tự động của Thành viên 4
        └── test_tv4_algorithms.py   # 15 Unit Tests kiểm thử toàn diện
```

---

## 🚀 3. HƯỚNG DẪN CÀI ĐẶT & CHẠY ỨNG DỤNG

### Bước 1: Cài đặt thư viện phụ thuộc
```powershell
cd OMR_System_TV4
pip install -r requirements.txt
```

### Bước 2: Chạy bộ kiểm thử tự động (15 Unit Tests)
```powershell
python -m unittest discover tests -v
```

### Bước 3: Chạy script kiểm thử CLI & xuất bảng điểm Excel
```powershell
python run_demo.py
```

### Bước 4: Khởi chạy Giao diện Web Streamlit
```powershell
streamlit run app.py
```
Ứng dụng sẽ tự động mở tại `http://localhost:8501` với 4 Tab:
1. **Tab 1:** Chấm bài trắc nghiệm (Hỗ trợ 5 kịch bản mẫu hoặc tải ảnh tùy chọn, kiểm duyệt thủ công & xuất file Excel).
2. **Tab 2:** Kiểm thử thuật toán & File CSV (Xem ma trận trước/sau, lọc nhiễu, cân bằng sáng, tải CSV/PNG theo Slide 10–12).
3. **Tab 3:** Đối chứng trực quan Scratch (NumPy) vs OpenCV (Triplet View, Bảng đối chứng 4 module, Khảo sát tham số Ablation Study).
4. **Tab 4:** Thống kê phổ điểm & Phân tích độ khó từng câu hỏi.
