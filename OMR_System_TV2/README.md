# OMR SYSTEM - THÀNH VIÊN 2 (TV2)
## Nắn thẳng phối cảnh & Định vị Marker (Perspective & Registration)

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**  
**Môn học:** Xử lý ảnh số (INT13146)  
**Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt  

---

### 1. Giới thiệu Module
Module này phụ trách toàn bộ Giai đoạn 2 của pipeline hệ thống chấm trắc nghiệm tự động OMR:
- Nhận tọa độ 4 góc khung phiếu từ Thành viên 1.
- Nắn thẳng phối cảnh (Perspective Transform) đưa phiếu thi về dạng phẳng chuẩn (800 x 1000).
- So khớp mẫu NCC xác định Marker và chuẩn hóa góc xoay phiếu về 0° trước khi bàn giao cho Thành viên 3.

> **Ràng buộc:** Toàn bộ thuật toán được cài đặt từ đầu (**100% Code from Scratch bằng NumPy**), không gọi hàm xử lý của thư viện OpenCV trong luồng chính.

---

### 2. Danh sách thuật toán tự cài đặt trong `tv2_perspective.py`

1. **`order_corners(pts)`**: Sắp xếp chuẩn hóa 4 đỉnh tứ giác theo thứ tự `[Top-Left, Top-Right, Bottom-Right, Bottom-Left]` dựa trên tổng $(x+y)$ và hiệu $(y-x)$, đồng thời tự tính kích thước phẳng tối ưu $(W, H)$ bằng khoảng cách Euclid.
2. **`solve_homography_svd(src, dst)`**: Lập hệ 8 phương trình thuần nhất $Ah=0$ ($A \in \mathbb{R}^{8 \times 9}$) và giải ma trận Homography $3 \times 3$ bằng phân tích suy biến SVD (`np.linalg.svd`). Sai số so với `cv2.getPerspectiveTransform` chỉ $\approx 10^{-10}$.
3. **`warp_perspective_bilinear(img, H, out_size)`**: Ánh xạ ngược (Inverse Warping với $H^{-1}$) kết hợp Nội suy song tuyến tính (Bilinear Interpolation). Được vector hóa 100% bằng NumPy (`np.meshgrid`, broadcasting). Đạt $\text{PSNR} = 87.34\text{ dB}$, $\text{MSE} = 0.0001$.
4. **`match_template_ncc(img, template)`**: So khớp mẫu Tương quan chéo chuẩn hóa (Normalized Cross-Correlation - NCC), tương đương `cv2.matchTemplate(TM_CCOEFF_NORMED)`. Sử dụng `sliding_window_view` tối ưu tốc độ.
5. **`detect_markers_and_correct_rotation(warped_img, anchor_template)`**: Quét 4 góc tìm Marker đặc biệt để tự động nhận diện góc xoay ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) và tự động xoay phiếu về đúng chiều chuẩn $0^\circ$.
6. **`process_perspective_pipeline(...)` & `warp_perspective_sheet(...)`**: Các hàm đóng gói pipeline sẵn sàng kết nối trực tiếp với Pipeline Coordinator chung của nhóm.

---

### 3. Quy cách Đầu vào & Đầu ra (Input / Output Specifications)

#### 🔹 Đầu vào (Input) của Module TV2:
1. **Ảnh gốc (`img`)**: Ảnh chụp phiếu thi từ smartphone (mức xám $H \times W$ hoặc ảnh màu RGB $H \times W \times 3$). Có thể bị nghiêng góc từ $15^\circ \rightarrow 45^\circ$, ngược hoặc xoay ngang.
2. **Tọa độ 4 đỉnh khung phiếu (`corners` / `raw_corners`)**: Mảng NumPy shape `(4, 2)` chứa 4 cặp tọa độ $(x, y)$ của 4 góc phiếu thi do Thành viên 1 phát hiện và bàn giao.
3. *(Tùy chọn)* **Mẫu Marker định vị (`anchor_template`)**: Ảnh ma trận mẫu của Marker góc dùng để so khớp NCC xác định hướng xoay.
4. *(Tùy chọn)* **Kích thước chuẩn đích (`target_size`)**: Tuple `(W_dst, H_dst)` (chuẩn mặc định là $800 \times 1000$ pixels).

#### 🔹 Đầu ra (Output) của Module TV2:
1. **Ảnh phiếu thi đã nắn phẳng (`warped_sheet` / `warped_gray`)**: 
   - Ảnh nắn thẳng hình học loại bỏ biến dạng phối cảnh, chuẩn hóa về kích thước $800 \times 1000$ pixels.
   - Chiều xoay được chuẩn hóa về đúng $0^\circ$ (chuẩn bị bàn giao cho Thành viên 3 phân đoạn và chấm ô tô).
2. **Thông tin metadata (`info`)**:
   - `ordered_corners`: 4 đỉnh đã sắp xếp theo thứ tự `[TL, TR, BR, BL]`.
   - `homography_matrix`: Ma trận biến đổi phối cảnh $H$ kích thước $3 \times 3$.
   - `detected_angle`: Góc xoay phát hiện ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
   - `marker_scores`: Điểm tương quan NCC tại 4 góc.

#### 📂 Dữ liệu minh họa Input / Output trên đĩa:
- Thư mục dữ liệu mẫu đầu vào: `OMR_System_TV2/data/input/`
  - `sample_tilted_sheet.png`: Ảnh chụp phiếu bị nghiêng phối cảnh.
  - `input_corners.txt`: Tọa độ 4 đỉnh nhận từ TV1.
- Thư mục kết quả nắn phẳng đầu ra: `OMR_System_TV2/data/output/`
  - `warped_sheet_scratch.png`: Ảnh nắn phẳng bằng thuật toán Scratch của TV2.
  - `warped_sheet_opencv.png`: Ảnh nắn phẳng bằng OpenCV đối chứng.

---

### 4. Hướng dẫn chạy kiểm thử Unit Benchmark
Chạy script kiểm thử đối chứng Scratch vs. OpenCV:

```bash
python test_tv2.py
```

Kết quả mong đợi:
- Tất cả 6 bài kiểm thử đều báo `[PASS]`.
- PSNR $> 48\text{ dB}$ (thực tế đạt $> 87\text{ dB}$).
- Sai số SVD $< 10^{-4}$ (thực tế đạt $10^{-10}$).


