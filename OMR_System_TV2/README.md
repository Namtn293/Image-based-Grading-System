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

> **Ràng buộc:** Toàn bộ thuật toán được cài đặt từ đầu (**100% Code from Scratch bằng NumPy**). Gói `tv2` **không import OpenCV**; OpenCV chỉ nằm trong `tv2/baseline_opencv.py` và `tv2/benchmark.py` để đối chứng (có test tự động kiểm tra điều này).

---

### 2. Cấu trúc thư mục

```
OMR_System_TV2/
├── tv2_perspective.py          # Lớp vỏ tương thích: TV4 vẫn `import tv2_perspective` như cũ
├── tv2/                        # Mã nguồn chính (100% NumPy)
│   ├── config.py               # PerspectiveConfig và các hằng số
│   ├── geometry.py             # order_corners, quad_size, destination_corners
│   ├── homography.py           # solve_homography_svd (Normalized DLT + SVD)
│   ├── warping.py              # warp_perspective_bilinear (Inverse Warping + Bilinear)
│   ├── template_matching.py    # match_template_ncc (NCC qua ảnh tích phân)
│   ├── registration.py         # detect_markers_and_correct_rotation, corners_for_angle
│   ├── pipeline.py             # process_perspective_pipeline, warp_perspective_sheet
│   ├── baseline_opencv.py      # Các hàm OpenCV đối chứng (chỉ nạp khi cần)
│   └── benchmark.py            # Benchmark Scratch vs. OpenCV (Bảng 6.1)
├── tests/                      # Bộ kiểm thử pytest (40 test)
├── data/input/                 # Ảnh phiếu nghiêng + tọa độ 4 góc từ TV1
├── data/output/                # Ảnh nắn phẳng Scratch / OpenCV (sinh bởi benchmark)
├── outputs/tv2/                # benchmark_metrics.csv/json, triplet_warp.png
├── reports/Bao_Cao_Thanh_Vien_2_OMR.md
├── conftest.py  requirements.txt  README.md
```

---

### 3. Danh sách thuật toán tự cài đặt

1. **`order_corners(pts)`** (`geometry.py`): Sắp xếp 4 đỉnh `[TL, TR, BR, BL]` theo tổng $(x+y)$ và hiệu $(y-x)$. Nếu quy tắc này chọn trùng đỉnh (tứ giác xoay gần $45^\circ$) thì chuyển sang sắp xếp theo góc cực quanh trọng tâm. Đồng thời tính kích thước phẳng $(W, H)$ theo cạnh dài nhất.
2. **`solve_homography_svd(src, dst)`** (`homography.py`): Lập hệ $Ah=0$ ($A \in \mathbb{R}^{8 \times 9}$) sau khi **chuẩn hóa Hartley** tọa độ, giải bằng SVD, rồi khôi phục $H = T_{dst}^{-1} H_n T_{src}$. Báo `ValueError` khi 4 điểm thẳng hàng hoặc trùng nhau.
3. **`warp_perspective_bilinear(img, H, out_size)`** (`warping.py`): Ánh xạ ngược với $H^{-1}$ kết hợp nội suy song tuyến tính, vector hóa 100%. Điểm lân cận ngoài ảnh nguồn mang giá trị 0 (giống `BORDER_CONSTANT` của OpenCV).
4. **`match_template_ncc(img, template)`** (`template_matching.py`): NCC tương đương `cv2.TM_CCOEFF_NORMED`. Tử số tính bằng $\sum I \cdot (T - \bar T)$ trên view cửa sổ trượt; mẫu số tính bằng 2 ảnh tích phân (`int64`, chính xác tuyệt đối với ảnh số nguyên). Bộ nhớ đỉnh ~4.5 MB thay vì ~300 MB.
5. **`detect_markers_and_correct_rotation(warped_img, anchor_template)`** (`registration.py`): Quét 4 góc tìm Anchor Marker để nhận diện góc xoay $0^\circ/90^\circ/180^\circ/270^\circ$ và xoay về $0^\circ$.
6. **`process_perspective_pipeline(...)`, `warp_perspective_sheet(...)`** (`pipeline.py`): Hàm đóng gói. Khi có `anchor_template`, pipeline dò hướng trên ảnh nắn thử, gán lại 4 góc cho hướng $0^\circ$ rồi nắn **một lần** từ ảnh gốc — ảnh ra luôn đúng `target_size`, không méo khi phiếu xoay $90^\circ/270^\circ$.

---

### 4. Quy cách Đầu vào & Đầu ra

#### 🔹 Đầu vào
1. **Ảnh gốc (`img`)**: mức xám $H \times W$ hoặc ảnh màu $H \times W \times 3$ (mặc định BGR như `cv2.imread`; đổi bằng `color_order="rgb"`).
2. **Tọa độ 4 đỉnh (`corners` / `raw_corners`)**: mảng `(4, 2)` (chấp nhận cả `(4, 1, 2)` của contour) do TV1 bàn giao.
3. *(Tùy chọn)* **`anchor_template`**: mẫu Anchor Marker (mức xám 2D). **Lưu ý thiết kế phiếu:** Anchor Marker ở góc trên-phải phải khác mẫu 3 marker còn lại, nếu không NCC không thể phân biệt hướng.
4. *(Tùy chọn)* **`target_size`** `(W, H)` (mặc định $800 \times 1000$) hoặc một `PerspectiveConfig`.

#### 🔹 Đầu ra
1. **Ảnh phiếu đã nắn phẳng**: kích thước `target_size`, hướng $0^\circ$, bàn giao cho TV3.
2. **Metadata `info`**: `ordered_corners`, `homography_matrix`, `output_size`, `detected_angle`, `marker_scores`.

#### 📂 Dữ liệu minh họa
- `data/input/sample_tilted_sheet.png`, `data/input/input_corners.txt`: ảnh phiếu nghiêng và 4 góc từ TV1.
- `data/output/warped_sheet_scratch.png`, `data/output/warped_sheet_opencv.png`: kết quả nắn phẳng (sinh lại bằng benchmark).

---

### 5. Cách sử dụng

```python
import cv2, numpy as np
from tv2 import warp_perspective_sheet, process_perspective_pipeline, PerspectiveConfig

img = cv2.imread("data/input/sample_tilted_sheet.png", cv2.IMREAD_GRAYSCALE)
corners = np.loadtxt("data/input/input_corners.txt")

sheet = warp_perspective_sheet(img, corners)                       # 1000 x 800, Scratch
sheet, info = process_perspective_pipeline(img, corners, anchor_template=anchor,
                                           config=PerspectiveConfig(target_size=(800, 1000)))
```

Mã cũ `from tv2_perspective import ...` vẫn chạy bình thường.

---

### 6. Kiểm thử & Benchmark

```bash
cd OMR_System_TV2
pip install -r requirements.txt
python -m pytest -q          # 40 test
python -m tv2.benchmark      # Xuất outputs/tv2/* và data/output/*
```

Kết quả benchmark trên `data/input/sample_tilted_sheet.png` (đích $800 \times 1000$):

| Thuật toán | Sai số so với OpenCV | Kết quả |
| :--- | :--- | :---: |
| Homography (Normalized DLT + SVD) | $\max\lvert H_{sc} - H_{cv}\rvert = 7.1 \times 10^{-14}$ | PASS |
| Inverse Warping + Bilinear | PSNR 85.89 dB, lệch tối đa 1 mức xám | PASS |
| NCC (template $32 \times 32$) | $5.3 \times 10^{-5}$ trên cửa sổ có nội dung, đỉnh trùng khớp | PASS |
| Pipeline end-to-end | PSNR 85.89 dB | PASS |

Chi tiết và phân tích: [reports/Bao_Cao_Thanh_Vien_2_OMR.md](reports/Bao_Cao_Thanh_Vien_2_OMR.md).
