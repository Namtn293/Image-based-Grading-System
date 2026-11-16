# HỆ THỐNG CHẤM BÀI TRẮC NGHIỆM TỰ ĐỘNG TỪ ẢNH CHỤP (OMR SYSTEM) - NHÓM 4

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**  
**Khoa Công nghệ Thông tin 1 – Bộ môn Khoa học Máy tính**  
**Môn học:** Xử Lý Ảnh (INT13146)  
**Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt  
**Đề tài:** Hệ thống Chấm bài Trắc nghiệm Tự động từ Ảnh chụp (Automatic Optical Mark Recognition System – OMR)  
**Quy mô nhóm:** 04 sinh viên (Tỷ lệ đóng góp kỹ thuật: 25% / thành viên)  

---

## 📌 1. TỔNG QUAN DỰ ÁN & KIẾN TRÚC PIPELINE 4 GIAI ĐOẠN

Dự án xây dựng hệ thống tự động nhận diện và chấm bài thi trắc nghiệm từ ảnh chụp camera điện thoại (hoặc nạp ma trận mức xám dạng file `.csv`). Toàn bộ thuật toán xử lý ảnh trong luồng chấm chính được **tự cài đặt từ đầu (code from scratch) bằng thư viện `NumPy`**, tuyệt đối không dùng các hàm xử lý tắt của `OpenCV`. Thư viện `OpenCV` (`cv2`) chỉ được sử dụng độc lập làm thước đo chuẩn (Baseline) để đối chứng định lượng về sai số toán học (MSE, PSNR, IoU) và thời gian thực thi.

```
[Đầu vào: Ảnh chụp phiếu thi (.jpg/.png) hoặc Ma trận mức xám (.csv)]
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1 (THÀNH VIÊN 1): Tiền xử lý & Phát hiện khung phiếu thi      │
│  - Chuyển mức xám thị giác chuẩn Luminance (ITU-R BT.601)               │
│  - Bộ lọc Gaussian 2D (Tích chập không gian 2D Convolution)             │
│  - Phát hiện biên Canny 4 bước (Sobel 3x3 -> NMS -> Ngưỡng kép -> Hyst) │
│  - Biến đổi Hough xác định 4 góc khung phiếu (TL, TR, BR, BL)           │
└─────────────────────────────────────────────────────────────────────────┘
          │ (Tọa độ 4 đỉnh góc phiếu thi)
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2 (THÀNH VIÊN 2): Nắn thẳng phối cảnh & Định vị Marker        │
│  - Chuẩn hóa thứ tự 4 góc & lập hệ phương trình Ah = 0                  │
│  - Ước lượng ma trận Homography 3x3 bằng phép giải SVD                  │
│  - Ánh xạ ngược (Inverse Warping) & Nội suy song tuyến (Bilinear)       │
│  - So khớp mẫu NCC (Normalized Cross-Correlation) định vị 4 Marker góc  │
└─────────────────────────────────────────────────────────────────────────┘
          │ (Ảnh phiếu đã nắn phẳng 800x1000 chuẩn hóa)
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3 (THÀNH VIÊN 3): Phân đoạn ảnh & Nhận diện ô tô đáp án       │
│  - Phân ngưỡng tự động Otsu toàn cục & Phân ngưỡng thích nghi cục bộ    │
│  - Toán tử Hình thái học: Co (Erosion), Giãn (Dilation), Mở, Đóng       │
│  - Gán nhãn thành phần liên thông 2 lượt (Two-Pass CCL)                 │
│  - Trích xuất ô A/B/C/D & phân loại theo tỷ lệ diện tích tô (Fill Ratio)│
└─────────────────────────────────────────────────────────────────────────┘
          │ (Vector đáp án nhận diện + Trạng thái Đúng/Sai/Trống/Phạm quy)
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 4 (THÀNH VIÊN 4): Tăng cường ảnh, Chấm điểm, Báo cáo & GUI    │
│  - Cân bằng biểu đồ mức xám (Histogram Equalization dùng hàm CDF chuẩn) │
│  - Bộ lọc trung vị (Median Filter 2D) khử nhiễu xung muối tiêu          │
│  - Module I/O ma trận mức xám CSV phục vụ nghiệm thu theo Slide 10–12   │
│  - Đối chiếu đáp án chuẩn, chấm điểm thang 10 & xuất báo cáo Excel XLSX │
│  - Bộ đo Benchmark định lượng (MSE, PSNR, IoU, Latency) vs OpenCV       │
│  - Tích hợp Pipeline Coordinator & Giao diện Web tương tác Streamlit    │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
[Đầu ra: Ảnh chấm trực quan + Báo cáo Đối chứng + File Excel Ket_Qua_Cham_Thi.xlsx]
```

---

## 📂 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```
Image-based-Grading-System/
├── Ke_Hoach_De_Tai_OMR_Nhom4.md     # Đề cương chi tiết & Kế hoạch đồ án Nhóm 4
├── README.md                        # Tài liệu tổng quan toàn bộ đồ án
├── OMR_System_TV1/                  # Module Thành viên 1: Tiền xử lý & Khung phiếu
│   ├── README.md                    # Tài liệu kỹ thuật TV1
│   ├── src/                         # Mã nguồn (Grayscale, Gaussian Blur, Canny, Hough)
│   └── tests/                       # Unit tests kiểm thử thuật toán TV1
├── OMR_System_TV2/                  # Module Thành viên 2: Nắn phối cảnh & Marker
│   ├── README.md                    # Tài liệu kỹ thuật TV2
│   ├── tv2_perspective.py           # Thuật toán Homography SVD, Bilinear Warping & NCC
│   └── test_tv2.py                  # Bộ kiểm thử thuật toán TV2
├── OMR_System_TV3/                  # Module Thành viên 3: Phân đoạn & Ô tô đáp án
│   ├── README.md                    # Tài liệu kỹ thuật TV3
│   ├── requirements.txt             # Thư viện phụ thuộc
│   ├── src/                         # Mã nguồn (Otsu, Adaptive Mean, Morphology, CCL)
│   ├── tests/                       # Unit tests kiểm thử TV3
│   ├── reports/                     # Báo cáo kỹ thuật chi tiết
│   └── outputs/tv3/                 # Ảnh trung gian & file kết quả CSV/JSON
└── OMR_System_TV4/                  # Module Thành viên 4: Cải thiện ảnh, Chấm điểm & Dashboard
    ├── app.py                       # Giao diện Web tương tác Streamlit (4 Tab đầy đủ)
    ├── run_demo.py                  # Script CLI kiểm thử & in bảng số liệu đối chứng
    ├── generate_samples.py          # Script sinh bộ dữ liệu mẫu (DS-01 .. DS-05 & CSV)
    ├── requirements.txt             # Thư viện phụ thuộc của hệ thống
    ├── README.md                    # Tài liệu kỹ thuật chi tiết của Thành viên 4
    ├── data/                        # Thư mục dữ liệu kiểm thử
    │   ├── answer_key.csv           # File đáp án chuẩn (hỗ trợ linh hoạt số câu)
    │   ├── sample_matrix.csv        # Ma trận mức xám phục vụ nghiệm thu Slide 10–12
    │   ├── sample_matrix_15x15.csv  # Ma trận 15x15 kiểm thử lọc trung vị
    │   ├── Ket_Qua_Cham_Thi.xlsx    # Bảng điểm tổng hợp xuất ra Excel (2 sheet có định dạng)
    │   └── samples/                 # 5 kịch bản ảnh phiếu thi mẫu (DS-01 .. DS-05)
    ├── src/                         # Mã nguồn tích hợp và thuật toán Thành viên 4
    │   ├── tv4_enhancement_eval.py  # Thuật toán TV4 (Hist Eq, Median, Benchmark, Excel)
    │   └── pipeline_coordinator.py  # Luồng điều phối tích hợp toàn hệ thống
    └── tests/                       # Bộ kiểm thử tự động của Thành viên 4
        └── test_tv4_algorithms.py   # 20 Unit Tests kiểm thử toàn diện
```

---

## 🔬 3. KẾT QUẢ ĐỐI CHỨNG ĐỊNH LƯỢNG VỚI OPENCV (BENCHMARK)

Bảng đo đạc thực nghiệm giữa thuật toán tự cài đặt bằng `NumPy` và thư viện chuẩn `OpenCV` trên cùng dữ liệu ảnh thử nghiệm:

| Phân hệ | Thuật toán tự cài đặt (NumPy) | Sai số MSE | PSNR (dB) | IoU (%) | Đánh giá toán học |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Phân hệ 1** | Làm mờ Gauss 2D (5x5) | $0.0072$ | $69.54\text{ dB}$ | — | Khớp xấp xỉ nhân Gauss giải tích |
| **Phân hệ 1** | Phát hiện biên Canny | $23.08$ | $34.50\text{ dB}$ | $99.0\%$ | Khớp biên Sobel & triệt cực đại NMS |
| **Phân hệ 1** | Hough xác định 4 đỉnh khung | **$0.0000$** | **$100.0\text{ dB}$** | — | Khớp tuyệt đối tọa độ 4 góc |
| **Phân hệ 2** | Nắn phẳng phối cảnh SVD + Bilinear | $0.1116$ | $57.65\text{ dB}$ | — | Khớp nội suy tọa độ thực |
| **Phân hệ 3** | Phân ngưỡng Otsu toàn cục | **$0.0000$** | **$100.0\text{ dB}$** | **$100.0\%$** | Khớp tuyệt đối ngưỡng tối ưu $T^*$ |
| **Phân hệ 3** | Phân ngưỡng thích nghi Adaptive Mean | **$0.0000$** | **$100.0\text{ dB}$** | **$100.0\%$** | Khớp tuyệt đối qua Integral Image |
| **Phân hệ 3** | Phép mở hình thái học (Opening 3x3) | **$0.0000$** | **$100.0\text{ dB}$** | **$100.0\%$** | Khớp tuyệt đối Erosion + Dilation |
| **Phân hệ 4** | Cân bằng sáng (Histogram Equalization) | **$0.0000$** | **$100.0\text{ dB}$** | — | Khớp tuyệt đối hàm phân phối CDF |
| **Phân hệ 4** | Bộ lọc trung vị (Median Filter 3x3) | **$0.0000$** | **$100.0\text{ dB}$** | — | Khớp tuyệt đối phép lọc trung vị |

---

## 🚀 4. HƯỚNG DẪN CÀI ĐẶT & CHẠY ỨNG DỤNG

### Bước 1: Cài đặt thư viện phụ thuộc
```powershell
cd OMR_System_TV4
pip install -r requirements.txt
```

### Bước 2: Chạy bộ kiểm thử tự động (20 Unit Tests)
```powershell
python -m unittest discover tests -v
```
*(Yêu cầu kết quả: 20/20 test cases đạt trạng thái OK 100%).*

### Bước 3: Chạy script kiểm thử CLI & xuất bảng điểm Excel
```powershell
python run_demo.py
```
*(Script sẽ in bảng đo đạc đối chứng, tính điểm 5 bài thi mẫu và tự động sinh file `data/Ket_Qua_Cham_Thi.xlsx`).*

### Bước 4: Khởi chạy Giao diện Web tương tác Streamlit
```powershell
streamlit run app.py
```
Ứng dụng sẽ tự động mở tại `http://localhost:8501` với 4 Tab chức năng:
1. **🎯 Chấm Bài Trắc Nghiệm (Pipeline Tự Động):**
   - Hỗ trợ chấm từng bài hoặc chấm hàng loạt (Batch).
   - Tự động nhận diện cấu trúc phiếu và cảnh báo nếu số câu trên phiếu lệch với file đáp án CSV.
   - Cho phép giám thị kiểm duyệt và xác nhận đáp án chỉ với 1 cú click.
   - Xuất bảng điểm tổng hợp ra file Excel (`Ket_Qua_Cham_Thi.xlsx`).
2. **🔬 Kiểm Thử Thuật Toán Từng Phân Hệ:**
   - Kiểm thử độc lập từng thuật toán của 4 phân hệ.
   - Hỗ trợ tải ảnh chụp thực tế HOẶC nạp trực tiếp file ma trận mức xám `.csv` (hiển thị bảng số thực nghiệm pixel theo Slide 10–12).
   - Xuất kết quả đã xử lý ra file ảnh `.png` hoặc file ma trận `.csv`.
3. **⚖️ Đối Chứng Thuật Toán vs OpenCV:**
   - Đối chiếu trực quan song song ảnh đầu ra của thuật toán tự cài đặt và thư viện chuẩn OpenCV.
   - Hiển thị bản đồ sai khác tuyệt đối $|I_{\text{cài đặt}} - I_{\text{OpenCV}}|$ và đo đạc tự động các chỉ số MSE, PSNR, IoU.
   - Khảo sát độ nhạy tham số ngưỡng tô đậm $\tau_{\min}$ và hiệu quả của phân ngưỡng thích nghi trên ảnh bóng đổ.
4. **📊 Thống Kê & Phổ Điểm:**
   - Trực quan hóa phổ điểm toàn khóa thi (0 - 10 điểm).
   - Thống kê tỷ lệ trả lời đúng theo từng câu hỏi để phân tích độ khó của đề thi.
