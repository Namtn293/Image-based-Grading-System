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

### 3. Hướng dẫn chạy kiểm thử Unit Benchmark
Chạy script kiểm thử đối chứng Scratch vs. OpenCV:

```bash
python test_tv2.py
```

Kết quả mong đợi:
- Tất cả 6 bài kiểm thử đều báo `[PASS]`.
- PSNR $> 48\text{ dB}$ (thực tế đạt $> 87\text{ dB}$).
- Sai số SVD $< 10^{-4}$ (thực tế đạt $10^{-10}$).

