# BÁO CÁO NGHIỆM THU KỸ THUẬT - PHẦN VIỆC THÀNH VIÊN 2
## HỆ THỐNG CHẤM BÀI TRẮC NGHIỆM TỰ ĐỘNG TỪ ẢNH CHỤP (OMR)

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**  
**Môn học:** Xử lý ảnh số (INT13146)  
**Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt  
**Phụ trách:** Thành viên 2 — Giai đoạn 2: Nắn thẳng phối cảnh & Định vị Marker

---

## MỤC LỤC
1. [Tổng quan nhiệm vụ](#1-tổng-quan-nhiệm-vụ)
2. [Cơ sở lý thuyết và cài đặt](#2-cơ-sở-lý-thuyết-và-cài-đặt)
3. [Kiến trúc module](#3-kiến-trúc-module)
4. [Kết quả thực nghiệm định lượng](#4-kết-quả-thực-nghiệm-định-lượng)
5. [Kiểm thử](#5-kiểm-thử)
6. [Hạn chế và hướng phát triển](#6-hạn-chế-và-hướng-phát-triển)

---

## 1. TỔNG QUAN NHIỆM VỤ

**Đầu vào:** ảnh chụp phiếu (mức xám hoặc màu) và 4 góc khung phiếu do TV1 phát hiện.  
**Đầu ra:** ảnh phiếu phẳng $800 \times 1000$ ở hướng $0^\circ$ cho TV3, kèm ma trận $H$ và góc xoay phát hiện được.

Bốn thuật toán tự cài đặt theo đề cương (mục 5.2):
1. Sắp xếp chuẩn hóa 4 đỉnh tứ giác.
2. Lập hệ $Ah = 0$ và giải Homography $3 \times 3$ bằng SVD.
3. Ánh xạ ngược (Inverse Warping) và nội suy song tuyến tính.
4. So khớp mẫu NCC định vị Marker và xoay chuẩn hướng.

**Ràng buộc:** gói `tv2` chỉ dùng NumPy; OpenCV chỉ dùng để đối chứng (`tv2/baseline_opencv.py`, `tv2/benchmark.py`). Test `test_core_package_does_not_import_opencv` kiểm tra tự động rằng `import tv2` không nạp `cv2`.

---

## 2. CƠ SỞ LÝ THUYẾT VÀ CÀI ĐẶT

### 2.1. Sắp xếp 4 đỉnh `[TL, TR, BR, BL]`
- TL có $(x+y)$ nhỏ nhất, BR có $(x+y)$ lớn nhất; TR có $(y-x)$ nhỏ nhất, BL có $(y-x)$ lớn nhất.
- **Xử lý suy biến:** khi tứ giác xoay gần $45^\circ$, hai quy tắc trên có thể chọn trùng một đỉnh. Khi phát hiện trùng, module chuyển sang sắp xếp theo góc cực $\operatorname{atan2}(y - \bar y, x - \bar x)$ quanh trọng tâm (theo chiều kim đồng hồ trong hệ trục ảnh), rồi xoay vòng để đỉnh có $(x+y)$ nhỏ nhất đứng đầu.
- Kích thước phẳng: $W = \max(\lVert TR - TL\rVert, \lVert BR - BL\rVert)$, $H = \max(\lVert BL - TL\rVert, \lVert BR - TR\rVert)$.
- Tọa độ đích đặt tại tâm pixel biên: $[0, W-1] \times [0, H-1]$, cùng quy ước với `cv2.getPerspectiveTransform` (tránh lệch 1 pixel).

### 2.2. Homography bằng Normalized DLT + SVD
Mỗi cặp $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$ cho 2 phương trình:
$$\begin{bmatrix} -x_i & -y_i & -1 & 0 & 0 & 0 & x'_i x_i & x'_i y_i & x'_i \\ 0 & 0 & 0 & -x_i & -y_i & -1 & y'_i x_i & y'_i y_i & y'_i \end{bmatrix} \mathbf{h} = \mathbf{0}$$

- **Chuẩn hóa Hartley:** tọa độ pixel cỡ $10^3$ khiến các cột của $A$ chênh nhau tới $\sim 10^6$ lần. Trước khi lập $A$, mỗi tập điểm được dời trọng tâm về gốc và co giãn để khoảng cách trung bình bằng $\sqrt 2$ (ma trận $T_{src}, T_{dst}$). Sau khi giải được $H_n$: $H = T_{dst}^{-1} H_n T_{src}$, rồi chuẩn hóa $h_{33} = 1$.
- Nghiệm $\mathbf h$ là hàng cuối của $V^T$ trong $A = U \Sigma V^T$.
- **Phát hiện suy biến:** nếu $\sigma_8 < 10^{-10} \sigma_1$ (3 điểm thẳng hàng hoặc trùng nhau) thì báo `ValueError` thay vì trả về $H$ sai.
- Ma trận $A$ được lập vector hóa (`build_dlt_matrix`), không dùng vòng lặp.

### 2.3. Inverse Warping + Bilinear
Với mỗi điểm đích $(x', y')$: $[\tilde x, \tilde y, \tilde w]^T = H^{-1}[x', y', 1]^T$, $x = \tilde x / \tilde w$, $y = \tilde y / \tilde w$. Với $\alpha = x - \lfloor x \rfloor$, $\beta = y - \lfloor y \rfloor$:
$$I(x', y') = (1-\alpha)(1-\beta) I_{00} + \alpha(1-\beta) I_{10} + (1-\alpha)\beta I_{01} + \alpha\beta I_{11}$$

- Vector hóa toàn bộ lưới đích (`np.meshgrid` + nhân ma trận $3 \times N$).
- **Xử lý mép ảnh nguồn:** điểm lân cận nằm ngoài ảnh mang giá trị 0 (tương đương `BORDER_CONSTANT`), nên dải 1 pixel sát mép được nội suy mờ dần như OpenCV. Trong trường hợp phiếu chạm mép ảnh, cách này nâng PSNR so với OpenCV từ 33.5 dB (khi cắt cụt) lên 79.7 dB.

### 2.4. NCC tối ưu bộ nhớ
$$\text{NCC}(x,y) = \frac{\sum (I - \bar I_{x,y})(T - \bar T)}{\sqrt{\sum (I - \bar I_{x,y})^2 \cdot \sum (T - \bar T)^2}}$$

- **Tử số:** vì $\sum (T - \bar T) = 0$ nên $\sum (I - \bar I)(T - \bar T) = \sum I \cdot (T - \bar T)$. Tính bằng `einsum` trực tiếp trên view `sliding_window_view`, không tạo mảng 4 chiều trung gian.
- **Mẫu số:** $N \sum (I - \bar I)^2 = N \sum I^2 - (\sum I)^2$, với $\sum I$ và $\sum I^2$ của mọi cửa sổ lấy trong $O(1)$ từ 2 ảnh tích phân. Ảnh số nguyên được tính bằng `int64` nên chính xác tuyệt đối; cửa sổ phẳng có phương sai đúng bằng 0 và được gán NCC = 0.
- Bộ nhớ đỉnh với vùng $250 \times 200$, template $32 \times 32$: **~4.5 MB** (bản trước: ~300 MB mỗi góc).

### 2.5. Định hướng phiếu bằng Anchor Marker
- Anchor Marker đặt ở góc TR khi phiếu ở hướng $0^\circ$. Góc có NCC cực đại cho biết phiếu đã xoay: TR $\to 0^\circ$, BR $\to 90^\circ$, BL $\to 180^\circ$, TL $\to 270^\circ$ (thuận chiều kim đồng hồ).
- **Nắn một lần:** pipeline nắn thử theo đúng tỉ lệ khung đang thấy để dò Marker; sau đó dịch vòng 4 đỉnh $-(\text{góc}/90)$ vị trí để gán đúng `[TL, TR, BR, BL]` của phiếu ở hướng $0^\circ$ và nắn lại trực tiếp từ ảnh gốc. Ảnh ra luôn đúng `target_size` (không méo khi xoay $90^\circ/270^\circ$) và chỉ chịu một lần nội suy.

---

## 3. KIẾN TRÚC MODULE

```
TV1 (4 góc) ──> order_corners ──> solve_homography_svd ──> warp_perspective_bilinear ──┐
                                                                                        │ (nếu có anchor_template)
                     detect_markers_and_correct_rotation <── match_template_ncc <──────┘
                                   │ góc xoay
                                   ▼
                corners_for_angle ──> Homography + Warp lần 2 ──> TV3 (ảnh 800 x 1000, hướng 0°)
```

| File | Nội dung |
| :--- | :--- |
| `tv2/config.py` | `PerspectiveConfig` (target_size, search_margin_ratio, color_order, degenerate_tol) |
| `tv2/geometry.py` | `order_corners`, `quad_size`, `destination_corners` |
| `tv2/homography.py` | `build_dlt_matrix`, `solve_homography_svd` |
| `tv2/warping.py` | `warp_perspective_bilinear` |
| `tv2/template_matching.py` | `match_template_ncc` |
| `tv2/registration.py` | `to_gray`, `detect_markers_and_correct_rotation`, `corners_for_angle` |
| `tv2/pipeline.py` | `process_perspective_pipeline`, `warp_perspective_sheet` |
| `tv2/baseline_opencv.py`, `tv2/benchmark.py` | Đối chứng OpenCV (ngoài luồng chính) |
| `tv2_perspective.py` | Lớp vỏ tương thích cho TV4 |

---

## 4. KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG

Ảnh `data/input/sample_tilted_sheet.png`, kích thước đích $800 \times 1000$, thời gian là trung vị của 5 lần chạy. Sinh lại bằng `python -m tv2.benchmark` (số liệu gốc: `outputs/tv2/benchmark_metrics.csv`).

### Bảng 1: TV2 Scratch vs. OpenCV

| Thuật toán | Hàm OpenCV đối chứng | Sai số | Scratch (ms) | OpenCV (ms) | Kết quả |
| :--- | :--- | :--- | ---: | ---: | :---: |
| Homography 3x3 (Normalized DLT + SVD) | `cv2.getPerspectiveTransform` | $\max\lvert\Delta H\rvert = 7.1 \times 10^{-14}$ | 0.35 | 0.01 | PASS |
| Inverse Warping + Bilinear | `cv2.warpPerspective` | MSE 0.00017, PSNR 85.89 dB, lệch tối đa 1 | 310.9 | 2.8 | PASS |
| NCC (template 32x32) | `cv2.matchTemplate(TM_CCOEFF_NORMED)` | $5.3 \times 10^{-5}$, đỉnh trùng (47, 21) | 37.3 | 2.0 | PASS |
| Pipeline end-to-end | `getPerspectiveTransform` + `warpPerspective` | PSNR 85.89 dB | 259.9 | 2.0 | PASS |

Ngưỡng nghiệm thu: PSNR > 48 dB (mục 7.1), sai số $H$ và NCC < $10^{-4}$.

### Nhận xét
- **Homography** khớp OpenCV tới sai số máy ($10^{-14}$) nhờ chuẩn hóa Hartley.
- **Warp** chỉ lệch tối đa 1 mức xám ở một số ít pixel do làm tròn số thực, PSNR vượt xa ngưỡng 48 dB. Ảnh so sánh 3 khung (Scratch | OpenCV | sai khác $\times 10$): `outputs/tv2/triplet_warp.png`.
- **NCC:** ở các cửa sổ gần như phẳng (độ lệch chuẩn < 2 mức xám, vùng giấy trắng) NCC có dạng $0/0$; OpenCV tính bằng float32 nên dao động tới $5.5 \times 10^{-3}$, trong khi bản Scratch tính chính xác bằng `int64` và trả về 0. Vì vậy sai số được đo trên các cửa sổ có nội dung (std $\ge 2$), nơi hai bên khớp tới $5.3 \times 10^{-5}$.
- **Tốc độ:** bản Scratch chậm hơn OpenCV khoảng 100 lần ở bước warp (~0.3 s cho ảnh $800 \times 1000$), chủ yếu do truy xuất chỉ số (fancy indexing) và tính toán `float64` trên ~800 nghìn điểm. Tốc độ này bằng bản trước khi refactor.

---

## 5. KIỂM THỬ

`python -m pytest -q` trong `OMR_System_TV2` — **40 test, tất cả PASS**:

| File | Nội dung kiểm thử |
| :--- | :--- |
| `test_tv2_geometry.py` | Sắp xếp góc, tứ giác $45^\circ$, đầu vào dạng contour, đầu vào lỗi, tọa độ đích |
| `test_tv2_homography.py` | Khớp OpenCV, ánh xạ đúng 4 góc, tọa độ pixel lớn, 4 điểm suy biến |
| `test_tv2_warping.py` | Khớp OpenCV (ảnh xám, ảnh màu, phiếu chạm mép), $H$ suy biến, góc phiếu rơi đúng pixel biên |
| `test_tv2_template_matching.py` | Khớp OpenCV, tìm đúng đỉnh, cửa sổ/template phẳng, ảnh float, bộ nhớ < 20 MB |
| `test_tv2_registration.py` | Nhận diện đúng 4 góc xoay, ảnh màu RGB, dịch vòng góc |
| `test_tv2_pipeline.py` | Xoay 0/90/180/270 qua pipeline cho ảnh ra trùng khớp phiếu gốc, `PerspectiveConfig`, lớp vỏ tương thích, gói `tv2` không nạp OpenCV |
| `test_tv2_benchmark.py` | Toàn bộ Bảng 1 đạt PASS |

---

## 6. HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

1. **Thiết kế phiếu:** việc định hướng phụ thuộc vào Anchor Marker ở góc TR khác mẫu 3 marker còn lại. Mẫu phiếu hiện tại của nhóm dùng 4 ô đen giống nhau nên cần bổ sung một marker khác biệt.
2. **Tích hợp TV4:** `warp_perspective_sheet` đã hỗ trợ tham số `anchor_template`, nhưng `pipeline_coordinator.py` của TV4 chưa truyền tham số này, nên pipeline chung chưa tự sửa được phiếu xoay $90^\circ/180^\circ$.
3. **Mặt phẳng cứng:** Homography giả định tờ giấy phẳng; giấy nhàu hoặc cong (Failure Case 2 của đề cương) cần thêm điểm căn chỉnh cục bộ.
4. **Tốc độ warp:** có thể giảm thời gian bằng tính toán `float32` hoặc chỉ nội suy trong vùng hình chữ nhật bao của phiếu.
