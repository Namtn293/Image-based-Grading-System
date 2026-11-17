# BÁO CÁO NGHIỆM THU KỸ THUẬT - PHẦN VIỆC THÀNH VIÊN 1
## HỆ THỐNG CHẤM BÀI TRẮC NGHIỆM TỰ ĐỘNG TỪ ẢNH CHỤP (OMR)

* **Học viện Công nghệ Bưu chính Viễn thông (PTIT)**
* **Khoa Công nghệ Thông tin 1 – Bộ môn Khoa học Máy tính**
* **Học phần:** Xử lý ảnh số (INT13146)
* **Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt
* **Nhóm sinh viên thực hiện:** Nhóm 4 (Quy mô 04 sinh viên)
* **Thành viên phụ trách:** **Thành viên 1 (Tiền xử lý & Phát hiện khung phiếu thi)**
* **Tỷ lệ đóng góp kỹ thuật:** **25%** (Cài đặt thuật toán 100% Code from Scratch bằng `NumPy`)

---

## MỤC LỤC
1. [Tổng quan nhiệm vụ của Thành viên 1](#1-tổng-quan-nhiệm-vụ-của-thành-viên-1)
2. [Cơ sở lý thuyết và Cài đặt thuật toán thuần NumPy (Code from Scratch)](#2-cơ-sở-lý-thuyết-và-cài-đặt-thuật-toán-thuần-numpy-code-from-scratch)
   - [2.1. Chuyển đổi mức xám cảm nhận thị giác (Luminance ITU-R BT.601)](#21-chuyển-đổi-mức-xám-cảm-nhận-thị-giác-luminance-itu-r-bt601)
   - [2.2. Bộ lọc Gaussian 2D & Tích chập không gian phân tách (Separable 1D Convolution)](#22-bộ-lọc-gaussian-2d--tích-chập-không-gian-phân-tách-separable-1d-convolution)
   - [2.3. Thuật toán Phát hiện biên Canny đầy đủ 4 bước](#23-thuật-toán-phát-hiện-biên-canny-đầy-đủ-4-bước)
   - [2.4. Biến đổi Hough đường thẳng cực và Định vị 4 góc phiếu](#24-biến-đổi-hough-đường-thẳng-cực-và-định-vị-4-góc-phiếu)
   - [2.5. Phân loại kịch bản: phiếu phủ kín khung hay nằm trên nền bàn tối](#25-phân-loại-kịch-bản-phiếu-phủ-kín-khung-hay-nằm-trên-nền-bàn-tối)
3. [Kiến trúc Module và Thiết kế Pipeline](#3-kiến-trúc-module-và-thiết-kế-pipeline)
4. [Kết quả Thực nghiệm Định lượng (Benchmark: Scratch vs. OpenCV)](#4-kết-quả-thực-nghiệm-định-lượng-benchmark-scratch-vs-opencv)
5. [Khảo sát Tham số (Ablation Study)](#5-khảo-sát-tham-số-ablation-study)
6. [Hướng dẫn Ghép nối Hệ thống (Integration Interfaces với TV2 và TV4)](#6-hướng-dẫn-ghép-nối-hệ-thống-integration-interfaces-với-tv2-và-tv4)
7. [Phân tích Trường hợp Thất bại (Failure Cases) và Hướng khắc phục](#7-phân-tích-trường-hợp-thất-bại-failure-cases-và-hướng-khắc-phục)
8. [Đối chiếu với Code Review và Kết luận Nghiệm thu](#8-đối-chiếu-với-code-review-và-kết-luận-nghiệm-thu)

---

## 1. TỔNG QUAN NHIỆM VỤ CỦA THÀNH VIÊN 1

Trong kiến trúc tổng thể 4 giai đoạn của Đề tài, **Thành viên 1** đảm nhiệm **Giai đoạn 1
(Preprocessing & Sheet Detection)**: chuyển ảnh màu về mức xám, làm mịn nhiễu, phát hiện biên và
**định vị chính xác 4 đỉnh góc của tờ phiếu**. Đây là mắt xích khởi đầu: nếu 4 góc sai thì phép
nắn phối cảnh của TV2 sẽ cắt/nhòe vùng ô tô, kéo theo TV3 phân ngưỡng sai và TV4 chấm điểm sai.

```
[Đầu vào: Ảnh chụp phiếu thi (.jpg/.png) từ điện thoại / webcam]
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODULE THÀNH VIÊN 1: TIỀN XỬ LÝ & PHÁT HIỆN KHUNG PHIẾU THI (GIAI ĐOẠN 1)   │
│ 1. Chuyển đổi RGB → mức xám Luminance ITU-R BT.601                          │
│ 2. Làm mịn khử nhiễu: Bộ lọc Gaussian 2D (tích chập phân tách 2 lượt)        │
│ 3. Phát hiện biên Canny 4 bước (Sobel L2 → NMS → Ngưỡng kép → Hysteresis)   │
│ 4. Biến đổi Hough cực (ρ = x·cosθ + y·sinθ) + NMS không gian tham số        │
│ 5. Chấm điểm đường theo support biên thật, chọn cặp cạnh đối diện           │
│ 6. Giải hệ Cramer → 4 giao điểm → tứ giác [TL, TR, BR, BL]                  │
│ 7. Đối chứng định lượng song song với OpenCV (chỉ trong benchmark)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
[Đầu ra cho TV2: corners shape (4,2) float32 + ảnh xám đã tiền xử lý]
```

### Ràng buộc kỹ thuật (Theo Đề cương môn học):
* **100% cài đặt từ đầu** bằng `NumPy`. Không gọi `cv2.cvtColor`, `cv2.GaussianBlur`, `cv2.Sobel`,
  `cv2.Canny`, `cv2.HoughLines`, `cv2.findContours` trong luồng xử lý chính.
* `OpenCV` chỉ được gọi trong khối benchmark để đo sai số và thời gian đối chứng.

> **Ghi chú minh bạch:** bản báo cáo trước ghi *"tuyệt đối không thay đổi bất kỳ nội dung nào của
> các file đã có sẵn trong dự án"*. Điều đó **không còn đúng** sau đợt sửa theo Code Review, vì lỗi
> nằm chính trong mã nguồn TV1 nên buộc phải sửa. Toàn bộ thay đổi được liệt kê ở mục 8.

---

## 2. CƠ SỞ LÝ THUYẾT VÀ CÀI ĐẶT THUẬT TOÁN THUẦN NUMPY

### 2.1. Chuyển đổi mức xám cảm nhận thị giác (Luminance ITU-R BT.601)

**Cơ sở toán học:** mắt người nhạy nhất với ánh sáng lục ($\approx 555$ nm), sau đó đến đỏ và kém
nhất với lam. Chuẩn ITU-R BT.601 định nghĩa:

$$I_{\text{gray}}(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$

với $\sum w_c = 0.299 + 0.587 + 0.114 = 1.0$ (bảo toàn dải động $[0, 255]$).

**Cài đặt:** dùng `np.float64` trung gian, vector hóa toàn ảnh, `np.round` rồi ép `np.uint8`.

**Đối chứng:** $\text{MSE} = 0.0000$, sai khác cực đại $0$ mức xám trên toàn bộ $800{,}000$ pixel
so với `cv2.cvtColor(BGR2GRAY)` — khớp tuyệt đối, không phải "xấp xỉ".

---

### 2.2. Bộ lọc Gaussian 2D & Tích chập phân tách (Separable 1D Convolution)

**Cơ sở toán học:** nhân Gaussian 2D đẳng hướng, bán kính $r = \lfloor k/2 \rfloor$:

$$G(u, v) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{u^2+v^2}{2\sigma^2}\right), \qquad
K_{\text{gauss}} = \frac{G}{\sum G}$$

Do $\exp(a+b) = \exp(a)\exp(b)$, nhân 2D phân rã thành tích ngoài của hai nhân 1D:
$G(u,v) = g(u)\,g(v)$.

**Tối ưu:** lọc hàng rồi lọc cột ⇒ $2k$ phép nhân-cộng/pixel thay vì $k^2$. Với $k=5$: 10 thay vì
25 (giảm 2.5 lần). Đệm biên `BORDER_REFLECT_101` (`fedcba|abcdefgh|hgfedc`) qua
`np.pad(mode='reflect')` để triệt tiêu viền tối giả. Cửa sổ trượt dùng
`np.lib.stride_tricks.sliding_window_view` (không copy bộ nhớ).

**Đối chứng:** $\text{PSNR} = 69.54$ dB, sai khác cực đại $1$ mức xám. Chênh lệch 1 mức xám là do
quy ước làm tròn `round half to even` (NumPy) vs `round half away from zero` (C++), **không phải
sai số thuật toán**.

> **Đính chính so với báo cáo trước:** báo cáo cũ dùng PSNR $69.54$ dB như "bằng chứng lọc nhiễu
> tốt". Sai. Đại lượng đó chỉ đo **độ trung thực cài đặt** (Scratch có khớp OpenCV không), không
> đo khả năng khử nhiễu. Khảo sát khử nhiễu thật nằm ở mục 5.1, đo PSNR so với **ảnh sạch gốc**.

---

### 2.3. Thuật toán Phát hiện biên Canny đầy đủ 4 bước

Canny (1986) tối ưu 3 tiêu chí: tỉ lệ lỗi thấp, định vị chính xác, đáp ứng đơn trị.

**Bước 1 – Sobel $3\times3$ + gradient L2:**

$$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \qquad
K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

$$M = \sqrt{G_x^2 + G_y^2}, \qquad \theta = \operatorname{atan2}(G_y, G_x)$$

Dùng L2 (chuẩn Euclid) thay vì xấp xỉ L1 để định vị biên chính xác hơn. $G_x, G_y$ khớp
$0.0$ sai số so với `cv2.Sobel(..., CV_16S)`.

*Tối ưu:* 9 lát cắt của cửa sổ $3\times3$ cộng trực tiếp, thay cho
`sliding_window_view` + `einsum`. Kết quả **trùng khít** nhưng nhanh hơn ~2.4 lần
(91.7 ms → 37.4 ms trên ảnh $1000\times800$).

**Bước 2 – Triệt tiêu không cực đại (NMS):** lượng tử hóa $\theta$ thành 4 hướng. Thay vì tính
góc rồi so sánh, dùng tỉ lệ để tránh `arctan` tốn kém:

- Sector ngang: $|G_y| \le \tan(22.5^\circ)|G_x| \approx 0.4142|G_x|$ → so với $(r, c\pm1)$
- Sector dọc: $|G_y| \ge \tan(67.5^\circ)|G_x| \approx 2.4142|G_x|$ → so với $(r\pm1, c)$
- Sector chéo: dấu của $G_x \cdot G_y$ quyết định so với $(r\pm1, c\pm1)$

Dùng bất đẳng thức **nghiêm ngặt một phía** ($m > n_1$ và $m \ge n_2$) để tránh biên dày 2 pixel
trên các đỉnh bằng phẳng (plateau).

**Bước 3 – Ngưỡng kép:** $M_{\text{NMS}} \ge T_{\text{high}}$ → biên mạnh;
$T_{\text{low}} \le M_{\text{NMS}} < T_{\text{high}}$ → biên yếu; còn lại loại.

**Bước 4 – Nối biên trễ (Hysteresis):** đây là **tái tạo hình thái học** trên mặt nạ biên yếu:
lặp phép hợp 8 bản dịch của mặt nạ hiện tại rồi giao với mặt nạ yếu, dừng khi hội tụ. Tương đương
chính xác duyệt BFS 8-lân cận (đã kiểm chứng bằng test `assert np.array_equal`), nhưng vector hóa
hoàn toàn: **95 ms → 9 ms**.

**Đối chứng:** $\text{IoU} = 99.02\%$, $\text{Dice} = 0.9951$ so với `cv2.Canny(L2gradient=True)`.

---

### 2.4. Biến đổi Hough đường thẳng cực và Định vị 4 góc phiếu

**Phương trình Hesse:** mỗi điểm biên $(x_i, y_i)$ thỏa vô số đường; trong tọa độ cực:

$$\rho = x_i\cos\theta + y_i\sin\theta, \qquad \theta \in [0, \pi), \quad \rho \in [-D, D]$$

với $D = \sqrt{W^2+H^2}$. Các điểm thẳng hàng cho các đường cong sin cùng đi qua một điểm
$(\rho^*, \theta^*)$.

```
Mặt phẳng ảnh (x, y)                    Không gian tham số cực (rho, theta)
  y ^                                     rho ^
    |   * (x1, y1)                            |       /\
    |     * (x2, y2)      Hough Transform     |      /  \  <- Giao điểm cực đại
    |       * (x3, y3)   ─────────────────>   |     /    \    (rho*, theta*)
    └──────────> x                            0 ────┴────┴────> theta (0 → pi)
```

> **Đính chính:** báo cáo cũ ghi miền $\theta \in [-\pi/2, \pi/2)$. Code dùng $[0, \pi)$ — là quy
> ước Hesse chuẩn cho ảnh số (một đường không phân biệt hai chiều). Tài liệu đã sửa theo code.

**Bỏ phiếu vector hóa:** $R = X\cos\Theta^T + Y\sin\Theta^T \in \mathbb{R}^{N\times180}$ qua
`np.outer`, ánh xạ sang chỉ số 1D rồi `np.bincount`.

**NMS trong không gian tham số** (`apply_parameter_space_nms`): gộp các đường có
$\Delta\theta \le 5^\circ$ và $\Delta\rho \le 10$ px, giữ đường nhiều phiếu nhất. Hàm được tách
riêng để **dùng lại cho cả kết quả `cv2.HoughLines`** khi đối chứng (mục 4).

**Chấm điểm theo support thật** (`score_lines_by_edge_support`): mảng tích lũy chỉ đếm phiếu trong
ô rời rạc, nên hai đường lệch 1–2 px có điểm gần như nhau. Hàm đếm trực tiếp số pixel biên cách
đường $\le 1.5$ px (chia lô $\le 20{,}000$ điểm để giới hạn bộ nhớ). Nhờ vậy chọn đúng cạnh khung
phiếu thay vì cạnh của ô tô bên trong.

**Dựng tứ giác** (`detect_quad_from_hough_lines`):
1. Tách họ **cạnh dọc** ($|\cos\theta| \ge \cos 20^\circ$) và **cạnh ngang**
   ($|\sin\theta| \ge \cos 20^\circ$) — khung phiếu chụp tay nghiêng tối đa ~20°.
2. Trong mỗi họ, chọn cặp đường có **tổng support lớn nhất** mà khoảng cách $\ge 50\%$ kích thước
   ảnh (tức hai cạnh **đối diện**, không phải hai cạnh kề).
3. Tính 4 giao điểm bằng `intersect_lines_polar` (Cramer):

$$x = \frac{\rho_1\sin\theta_2 - \rho_2\sin\theta_1}{\sin(\theta_2-\theta_1)}, \qquad
y = \frac{\rho_2\cos\theta_1 - \rho_1\cos\theta_2}{\sin(\theta_2-\theta_1)}$$

với $|D| = |\sin(\theta_2-\theta_1)| > 10^{-4}$ (hai đường không song song).

4. Kiểm tra hợp lệ: tứ giác nằm trong ảnh, diện tích $\ge 30\%$ khung hình.

**Dự phòng** (`detect_quad_from_boundary_scratch`): Otsu → **điểm biên cực trị theo hàng/cột**
(`collect_boundary_extremes`, tối đa $2H+2W$ điểm, **luôn chứa 4 đỉnh**) → bao lồi Andrew's
Monotone Chain $O(N\log N)$ → **RDP trên đa giác khép kín** (`simplify_closed_polygon`, tách đôi tại
đỉnh xa nhất) → 4 đỉnh → chuẩn hóa thứ tự TL/TR/BR/BL.

> **Đính chính quan trọng (Code Review 2.2):** báo cáo cũ ghi "RDP với $\epsilon = 0.025 \times$
> chu vi" như một bước đang chạy tốt. Thực tế code cũ truyền **đa giác khép kín** vào RDP, mà RDP
> trên đường khép kín luôn thu về 2 điểm (điểm đầu = điểm cuối) nên bước này vô hiệu. Nay
> `ramer_douglas_peucker` **ném `ValueError`** nếu nhận đa giác khép kín, và đường dự phòng dùng
> `simplify_closed_polygon` tách đôi trước khi RDP. Ngoài ra việc lấy mẫu raster 3000 điểm dọc
> biên có thể **bỏ sót đỉnh góc**; đã thay bằng cực trị theo hàng/cột (luôn giữ đỉnh).

---

### 2.5. Phân loại kịch bản: phiếu phủ kín khung hay nằm trên nền bàn tối

Hai nhánh xử lý khác nhau hoàn toàn, nên chọn sai nhánh là sai toàn bộ kết quả. Bộ phân loại dùng
**thống kê toàn dải viền** (không dùng 4 pixel góc như code cũ) và ngưỡng **Otsu tự cài đặt**
(không dùng hằng số 100).

| Tiêu chí | Nội dung | Ngưỡng |
| :---: | :--- | :--- |
| 1 | **Cả bốn** dải viền đều tối hơn ngưỡng Otsu | $\ge 20$ mức xám |
| 2 | **Mỗi** dải viền có tỉ lệ pixel thuộc lớp tối của Otsu | $\ge 60\%$ |
| 3 | Cạnh **sáng nhất** vẫn tối hơn trung vị vùng giữa ảnh | $< 60\%$ |

Chỉ cần một tiêu chí đúng → kết luận "có nền bàn tối bao quanh".

**Phát hiện mới trong quá trình nghiệm thu (chưa có trong Code Review):** phiên bản đầu của bộ
phân loại gộp cả bốn dải viền thành **một** trung vị. Điều này đúng với DS-02 nhưng **sai với
DS-05** (bóng đổ không đều): dải trên sáng 245, ba dải còn lại tối 80, nên trung vị gộp rơi vào
vùng tối và ảnh bị đẩy sang nhánh Hough. Kết quả: tứ giác thụt vào trong, **sai 67.19 px** so với
baseline. Bản sửa yêu cầu **mọi phía** cùng tối (tiêu chí 1 và 2 xét riêng từng dải), đưa sai số
DS-05 về **0.00 px** mà không ảnh hưởng DS-02. Hai test hồi quy mới khóa lại hành vi này.

Bảng quyết định thực đo trên 5 ảnh mẫu:

| Ảnh | Trung vị 4 dải (T/B/L/R) | Vùng giữa | Otsu | Kết luận | Nhánh |
| :--- | :--- | :---: | :---: | :---: | :--- |
| DS-01 Chuẩn | 245 / 245 / 245 / 245 | 245 | 141 | Không nền tối | Toàn khung |
| DS-02 Nghiêng bàn tối | 45 / 45 / 45 / 45 | 245 | 144 | **Nền tối** | Canny + Hough |
| DS-03 Thiếu sáng | 142 / 142 / 142 / 142 | 142 | 86 | Không nền tối | Toàn khung |
| DS-04 Tô sai + tẩy | 245 / 245 / 245 / 245 | 245 | 141 | Không nền tối | Toàn khung |
| DS-05 Bóng đổ | **245 / 80 / 245 / 80** | 80 | 157 | Không nền tối | Toàn khung |

---

## 3. KIẾN TRÚC MODULE VÀ THIẾT KẾ PIPELINE

```
OMR_System_TV1/src/
├── __init__.py                # Re-export 32 API công khai
├── tv1_grayscale_blur.py      # Luminance BT.601 + Gaussian 2D separable
├── tv1_canny.py               # Sobel → NMS → Ngưỡng kép → Hysteresis vector hóa
├── tv1_hough.py               # Hough cực, NMS tham số, support scoring, Cramer,
│                              #   convex hull, RDP hở/khép kín, Otsu tự cài
└── tv1_preprocessing.py       # detect_sheet_corners, pipeline tích hợp, benchmark
```

`process_preprocessing_pipeline(image, use_scratch=True)` trả về dict:

| Khóa | Nội dung |
| :--- | :--- |
| `'gray'` | Ảnh mức xám `uint8` |
| `'blurred'` | Ảnh sau lọc Gaussian |
| `'edges'` | Ảnh biên nhị phân Canny |
| `'corners'` | `float32` shape $(4, 2)$, thứ tự `[TL, TR, BR, BL]` |
| `'latencies'` | Độ trễ từng công đoạn (ms) |

**Thay đổi thiết kế quan trọng:** `detect_sheet_corners` nhận thêm `precomputed_gray` và
`precomputed_edges`. Trước đây pipeline tính ảnh xám, tính biên Canny, rồi gọi
`detect_sheet_corners` khiến Canny **chạy lần thứ hai** (~70 ms lãng phí). Nay pipeline truyền lại
kết quả đã có. Test `test_pipeline_does_not_run_canny_twice` khóa lại điều này.

---

## 4. KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG (BENCHMARK: SCRATCH vs. OPENCV)

Môi trường: Windows 11, Python 3.14.0, NumPy 2.5.4, OpenCV 5.0.0. Ảnh chuẩn $1000\times800$
(`01_B21DCCN001_Chuan.png`). Mỗi phép đo lấy **giá trị tốt nhất trong 3 lần chạy**.

### Bảng 1: Nghiệm thu định lượng TV1

| STT | Thuật toán | Hàm Scratch | Hàm OpenCV | Chỉ số đối chứng | Thời gian Scratch | Thời gian OpenCV | Kết luận |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | Chuyển mức xám | `rgb_to_gray_scratch()` | `cv2.cvtColor` | MSE $=0.0000$, $\Delta I_{\max}=0$ | $18.2$ ms | $2.7$ ms | **PASS** |
| **2** | Gaussian 2D | `gaussian_blur_scratch()` | `cv2.GaussianBlur` | PSNR $=69.54$ dB, $\Delta I_{\max}=1$ | $38.8$ ms | $0.8$ ms | **PASS** |
| **3** | Canny 4 bước | `canny_edge_detector()` | `cv2.Canny` | IoU $=99.02\%$, Dice $=0.9951$ | $69.8$ ms | $3.1$ ms | **PASS** |
| **4** | Hough cực | `hough_lines_standard()` | `cv2.HoughLines` + NMS | Recall $=100.0\%$, Precision $=91.2\%$ | $146.9$ ms | $6.3$ ms | **PASS** |
| **5** | Định vị 4 góc | `detect_sheet_corners()` | Contour quad (OpenCV) | $E=0.00$ px (DS-01), $1.93$ px (DS-02), $0.00$ px (DS-05) | $23.6$ ms | $4.5$ ms | **PASS** |

### Nhận xét về phép đo Hough (đã sửa phương pháp)

Báo cáo cũ ghi Hough đạt "MSE $=0.0000$, PSNR $=999.00$ dB" và mô tả "bắt trọn 4 cạnh viền". Đó
**không phải một phép đo hợp lệ**: hai tập đường thẳng không cùng số phần tử và không cùng thứ tự
nên không thể tính MSE/PSNR theo pixel. Cách cũ so `|len(Scratch) − len(OpenCV)| ≤ 5` cũng sai về
bản chất, vì `cv2.HoughLines` **không** NMS (trả 79 đường) còn Scratch có NMS (34 đường).

Phương pháp đúng đang dùng:
1. Chuẩn hóa danh sách OpenCV bằng **cùng** quy trình `apply_parameter_space_nms` → 31 đường.
2. So khớp theo tham số hình học $(\rho, \theta)$, chuẩn hóa $\theta$ về $[0, \pi)$ để
   $(\rho,\theta)$ và $(-\rho, \theta+\pi)$ là một đường; dung sai $2^\circ$ / $5$ px.
3. Báo cáo **Recall** (% đường OpenCV được tìm lại) và **Precision** (% đường Scratch có trong
   OpenCV).

Kết quả: **Recall $=100.0\%$** (không bỏ sót đường nào của OpenCV), **Precision $=91.2\%$** (phần
lớn đường Scratch trùng OpenCV; phần dư là các đường phụ có support thật nhưng OpenCV gộp mất).
Với mục tiêu *dựng khung phiếu*, recall mới là chỉ số quan trọng — và đạt tuyệt đối.

### Nhận xét về độ trễ

| Ảnh | Xám | Gaussian | Canny | Định vị góc | **Tổng** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| DS-01 Chuẩn ($800\times1000$) | 17.7 | 39.6 | 66.7 | 4.0 | **128.0 ms** |
| DS-02 Nghiêng ($920\times1120$) | 24.1 | 53.0 | 91.2 | 209.0 | **377.3 ms** |
| DS-05 Bóng đổ ($800\times1000$) | 17.7 | 40.5 | 84.0 | 4.0 | **~146 ms** |

Trung bình 5 ảnh mẫu ≈ **208 ms**, lớn nhất **377 ms** — đạt yêu cầu $<600$ ms/phiếu và thấp hơn
nhiều so với $1.5$ s mà Code Review mục 2.7 nêu. So với bản trước khi tối ưu (1.1–1.7 s), giảm
4–6 lần nhờ: Sobel dùng slicing (91.7 → 37.4 ms), hysteresis vector hóa (95 → 9 ms), và pipeline
không chạy Canny hai lần.

> **Trung thực về con số:** báo cáo cũ ghi "297 ms/phiếu" cho toàn giai đoạn 1 trong khi bảng của
> chính báo cáo đó cộng lại đã hơn 480 ms. Con số 297 ms không tái lập được. Bảng trên là số đo
> lại thực tế.

---

## 5. KHẢO SÁT THAM SỐ (ABLATION STUDY)

Toàn bộ số liệu dưới đây được đo lại trong đợt nghiệm thu này. Các bảng cũ trong báo cáo trước
chứa số liệu **không tái lập được** (ví dụ PSNR $74.21$ dB cho kernel $3\times3$ trong khi đo lại
chỉ được $71.13$ dB trên cùng ảnh) nên đã được thay thế.

### 5.1. Khảo sát Gaussian: khử nhiễu thật (không phải độ trung thực cài đặt)

Thí nghiệm: lấy ảnh sạch DS-01 làm chuẩn, thêm nhiễu Gauss $\sigma = 18$ **và** nhiễu muối tiêu
2%, rồi đo PSNR của ảnh đã lọc **so với ảnh sạch gốc**.

Ảnh nhiễu ban đầu: PSNR $=19.15$ dB, phương sai Laplacian $=17{,}683$ (ảnh sạch: $2{,}571$).

| Kernel | $\sigma$ | PSNR vs ảnh sạch | Mảnh nhiễu $\le 5$ px | Thời gian | Đánh giá |
| :---: | :---: | ---: | ---: | ---: | :--- |
| $3\times3$ | 0.8 | 23.83 dB | 387 | 28.4 ms | Còn nhiều mảnh nhiễu rời rạc |
| $3\times3$ | 1.4 | 23.29 dB | 192 | 28.9 ms | Vẫn còn 192 mảnh rời |
| $5\times5$ | 0.8 | 23.79 dB | 348 | 37.7 ms | Chưa đủ mạnh cho nhiễu muối tiêu |
| **$5\times5$** | **1.4** | **22.28 dB** | **17** | **38.8 ms** | **Cân bằng tốt: nhiễu rời giảm 96%** |
| $5\times5$ | 2.0 | 21.56 dB | 24 | 36.5 ms | Mịn hơn nhưng bắt đầu mất chi tiết |
| $7\times7$ | 1.4 | 21.81 dB | 11 | 46.8 ms | Ít nhiễu nhất nhưng tốn thời gian hơn 20% |

**Kết luận:** $5\times5$, $\sigma = 1.4$ là điểm cân bằng. $3\times3$ để lại quá nhiều mảnh nhiễu
(rất hại cho Hough vì mỗi mảnh nhiễu đều bỏ phiếu); $7\times7$ chỉ giảm thêm 6 mảnh nhiễu nhưng
chậm hơn 20% và làm mờ nét bút bi (quan trọng với TV3).

> Lưu ý về cách đọc PSNR: PSNR $5\times5/\sigma{=}1.4$ *thấp hơn* $3\times3$ vì lọc mạnh làm mất
> chi tiết tần số cao, trong khi nhiễu còn lại ở $3\times3$ lại "tình cờ" gần ảnh sạch về mặt
> năng lượng. Vì vậy bảng này bổ sung cột **mảnh nhiễu rời rạc** — chỉ số phản ánh đúng tác động
> lên bước Hough phía sau.

### 5.2. Khảo sát cặp ngưỡng Canny $T_{\text{low}}, T_{\text{high}}$

Đo trên DS-03 (thiếu sáng + nhiễu hạt) với cùng ảnh xám và cùng Gaussian:

| Cặp ngưỡng | Mật độ biên | Số mảnh nhiễu $\le 5$ px | Số mảnh nhiễu $\le 20$ px | IoU vs OpenCV | Thời gian |
| :---: | ---: | ---: | ---: | ---: | ---: |
| $(30, 80)$ | 4.75% | 44 | — | 99.08% (DS-01) | 108 ms |
| **$(50, 150)$** | **3.62%** | **2** | **—** | **99.02%** | **108 ms** |
| $(80, 200)$ | 3.60% | 0 | — | 99.02% | 106 ms |
| $(100, 250)$ | 1.92% | — | — | — | 197 ms |

Trên ảnh sạch DS-01: $(30,80)$ sinh 47 mảnh nhiễu $\le 5$ px, $(50,150)$ chỉ còn 7, $(80,200)$ còn
2. Chọn $(50, 150)$: giảm 96% mảnh nhiễu so với $(30,80)$ mà **không** làm mất biên thật (IoU với
OpenCV vẫn $99.02\%$, số điểm biên chênh lệch chỉ 143 pixel trên 800{,}000).

> Báo cáo cũ ghi $(30,80)$ cho IoU $96.8\%$ và $(80,200)$ cho $95.4\%$. Đo lại cho thấy cả hai đều
> xấp xỉ $99\%$ — vì Scratch và OpenCV dùng chung ngưỡng nên IoU ít nhạy với ngưỡng. Chỉ số phân
> biệt thực sự là **số mảnh nhiễu rời rạc**, đã thêm vào bảng.

---

## 6. HƯỚNG DẪN GHÉP NỐI HỆ THỐNG (INTEGRATION VỚI TV2 VÀ TV4)

### 6.1. Với TV2 (Nắn phối cảnh)

```python
# TV1: phát hiện 4 góc, tái dùng gray/edges nếu đã có để tránh chạy lại Canny
result = process_preprocessing_pipeline(raw_image, use_scratch=True)
corners = result["corners"]          # float32 (4, 2) theo thứ tự [TL, TR, BR, BL]

# TV2: nắn thẳng phối cảnh
warped = warp_perspective_sheet(result["gray"], corners, target_width=800, target_height=1000, use_scratch=True)
```

Hợp đồng dữ liệu: `corners` là `float32`, thứ tự cố định TL → TR → BR → BL, đúng thứ tự
`order_corners()` của TV2 mong đợi. Không cần chuyển đổi trung gian.

### 6.2. Với TV4 (Pipeline Coordinator)

- `rgb_to_gray_scratch` — gọi trong Stage 1 của `pipeline_coordinator.py`
- `gaussian_blur_scratch`, `canny_edge_scratch` — gọi trong `benchmark_all_modules()`
- `detect_sheet_corners` — gọi để lấy 4 góc trước khi warp

### 6.3. Ranh giới trách nhiệm với TV2 (quan trọng, Code Review 1.2)

TV1 **chỉ** trả về 4 góc hình học. Việc **chuẩn hóa hướng phiếu** (0°/90°/180°/270°) thuộc TV2
(`detect_markers_and_correct_rotation` + NCC). Code Review đã chỉ ra kịch bản xoay 90°/180°
**chưa được xử lý end-to-end**; nguyên nhân là TV2 chưa được gọi trong pipeline, cộng với thiết kế
phiếu hiện tại có 4 marker **giống nhau** nên về nguyên tắc không thể suy ra hướng xoay. Đây là
việc của TV2 và thiết kế phiếu — không phải lỗi của module TV1.

---

## 7. PHÂN TÍCH TRƯỜNG HỢP THẤT BẠI (FAILURE CASES)

| STT | Trường hợp | Nguyên nhân gốc rễ | Trạng thái xử lý |
| :---: | :--- | :--- | :--- |
| **1** | Phiếu trên mặt bàn trắng, tương phản $\Delta I < 15$ | Gradient Sobel dưới $T_{\text{low}}$, Canny bắt khung viền in bên trong thay vì mép giấy | **Chưa xử lý triệt để.** Hiện dựa vào bộ phân loại dải viền: nếu cả 4 phía đều sáng gần bằng thân phiếu thì trả về 4 góc khung hình (giả định phiếu phủ kín ảnh). Nếu phiếu **không** phủ kín và nền cũng sáng thì không có tín hiệu hình học nào để tách — cần dataset thật để đánh giá. |
| **2** | Rung nhòe chuyển động (motion blur) | Gradient bị nhòe theo một phương, biên đứt đoạn | **Chưa cài đặt.** Đề xuất: kernel nhỏ hơn ($3\times3$) + ngưỡng thích nghi theo phân vị độ lớn gradient. Chưa thử nghiệm vì chưa có ảnh mẫu bị nhòe chuyển động thật. |
| **3** | Bóng đổ mạnh một phía (DS-05) | Bộ phân loại gộp trung vị 4 dải viền nên bị bóng đổ đánh lừa | **Đã sửa trong đợt này.** Yêu cầu cả bốn phía cùng tối; sai số DS-05: $67.19 \to 0.00$ px. Có 2 test hồi quy. |
| **4** | Phiếu xoay 90°/180° | 4 marker giống nhau nên NCC không phân biệt được hướng | **Ngoài phạm vi TV1** (TV2 + thiết kế phiếu). |

---

## 8. ĐỐI CHIẾU VỚI CODE REVIEW VÀ KẾT LUẬN NGHIỆM THU

### 8.1. Các mục Code Review thuộc phạm vi TV1

| Mục | Nội dung Code Review | Xử lý |
| :--- | :--- | :--- |
| **1.2** | `detect_sheet_corners` chỉ ngưỡng cố định `gray >= 100`, Canny/Hough không được dùng ngoài benchmark | **Đã sửa.** Nhánh Hough là đường chính: Canny → Hough → chấm support → cặp cạnh đối diện → Cramer. Ngưỡng cố định + hull + RDP chỉ còn là dự phòng. |
| **1.5** | `"status": "PASS" if ... else "PASS"` (luôn PASS) tại 2 chỗ | **Đã sửa.** Cả hai nhánh `else` thành `"FAIL"`. Test `test_benchmark_status_can_fail` chèn số liệu giả để xác nhận FAIL thật sự xuất hiện. |
| **2.2** | RDP nhận đa giác khép kín → luôn trả 2 điểm; lấy mẫu raster bỏ sót đỉnh | **Đã sửa.** `ramer_douglas_peucker` ném `ValueError` với đa giác khép kín; thêm `simplify_closed_polygon`; thay raster bằng cực trị hàng/cột (luôn chứa 4 đỉnh). |
| **2.3** | Kiểm tra nền bàn tối dựa 4 pixel góc + hằng số 100 | **Đã sửa.** Trung vị 4 dải viền + Otsu + 3 tiêu chí, và **phát hiện thêm** lỗi bóng đổ một phía (mục 2.5). |
| **2.7** | TV1 chạy 1.1–1.7 s, vượt ngưỡng test 1.5 s | **Đã sửa.** Còn 128 ms (DS-01) … 377 ms (DS-02); 33 test chạy trong 1.8 s. |
| **3 (docstring)** | Tài liệu ghi $\theta \in [-\pi/2, \pi/2)$ nhưng code dùng $[0, \pi)$ | **Đã đồng bộ tài liệu** theo code. |
| **3 (repo)** | `.pyc` trong `__pycache__` còn trong git | Ngoài phạm vi module. |

### 8.2. Những điểm báo cáo cũ tuyên bố sai và đã đính chính

Vì yêu cầu trung thực học thuật, các tuyên bố không kiểm chứng được trong báo cáo trước được ghi
lại rõ ràng:

1. "Toàn bộ thuật toán 100% from scratch" — **đúng**, đã kiểm chứng bằng rà soát import: luồng
   chính chỉ dùng `numpy`, `cv2` xuất hiện trong hàm `*_cv` đối chứng và trong
   `detect_sheet_corners(use_scratch=False)`.
2. "Không thay đổi bất kỳ file nào đã có" — **sai sau đợt sửa**. Đã sửa 4 file nguồn, 1 file test,
   thêm 1 script CLI; chi tiết ở 8.3.
3. "18/18 unit tests" — **đã lỗi thời**. Hiện là **33/33**.
4. Hough "MSE $= 0.0000$, PSNR $= 999.00$ dB" — **không hợp lệ về phương pháp**, đã thay bằng
   Recall/Precision theo $(\rho, \theta)$.
5. PSNR $69.54$ dB được trình bày như bằng chứng khử nhiễu — **sai ý nghĩa**, đó là độ trung thực
   cài đặt. Khảo sát khử nhiễu thật ở mục 5.1.
6. "297 ms/phiếu" — **không tái lập được**, đã thay bằng bảng đo thực tế (128–377 ms).
7. Bảng ablation Gaussian và Canny cũ — **số liệu không tái lập**, đã đo lại toàn bộ.

### 8.3. Danh sách thay đổi mã nguồn trong đợt này

| File | Thay đổi |
| :--- | :--- |
| `src/tv1_canny.py` | Sobel viết lại bằng slicing (giữ nguyên kết quả, nhanh 2.4×); hysteresis vector hóa 8 bản dịch (nhanh ~10×, kết quả `array_equal` với BFS); bỏ import `deque` không dùng |
| `src/tv1_hough.py` | Thêm `otsu_threshold_scratch`, `collect_boundary_extremes`, `simplify_closed_polygon`, `apply_parameter_space_nms`, `score_lines_by_edge_support`, `detect_quad_from_hough_lines`; `ramer_douglas_peucker` từ chối đa giác khép kín; `hough_lines_standard` dùng chung NMS |
| `src/tv1_preprocessing.py` | Thêm `_border_band_stats`, `_is_sheet_surrounded_by_dark_desk`, `calculate_line_match_rate`; `detect_sheet_corners` nhận `precomputed_gray`/`precomputed_edges` và chạy Canny→Hough; benchmark có nhánh FAIL thật; demo xuất `7_hough_accumulator.png` |
| `src/__init__.py` | Re-export 32 API |
| `tests/test_tv1_hough.py` | +6 test (RDP khép kín, simplify, cực trị biên, Otsu, quad từ Hough, NMS tham số) |
| `tests/test_tv1_benchmark.py` | Viết lại; +2 test hồi quy bóng đổ DS-05 |
| `test_tv1.py` | Thêm kiểm thử nhánh Hough DS-02 và bóng đổ DS-05 |

### 8.4. Minh chứng bàn giao

- **Mã nguồn:** `OMR_System_TV1/src/` (4 module + `__init__.py`)
- **Kiểm thử:** `OMR_System_TV1/tests/` — **33/33 PASSED** trong 1.8 s
- **Script nghiệm thu nhanh:** `python test_tv1.py` — 6 bài test, tất cả `[PASS]`
- **Sản phẩm trực quan:** `outputs/tv1/` gồm `1_input_sample.png` … `9_triplet_comparison.png`
  (đã có `7_hough_accumulator.png`), kèm `benchmark_metrics.csv` và `benchmark_metrics.json`
- **Tài liệu:** `README.md` và báo cáo này

### 8.5. Kết luận

Giai đoạn 1 hoàn thành đúng phạm vi: chuyển mức xám, lọc Gaussian, Canny 4 bước, Hough cực và
định vị 4 góc phiếu — tất cả cài đặt thuần `NumPy`, đối chứng định lượng với OpenCV đạt hoặc vượt
mọi chỉ tiêu (MSE 0, PSNR > 69 dB, IoU 99.02%, Hough Recall 100%, sai số góc ≤ 1.93 px), độ trễ
128–377 ms/phiếu.

Các lỗi Code Review chỉ ra trong phạm vi TV1 (1.2, 1.5, 2.2, 2.3, 2.7) đã được sửa và có test hồi
quy khóa lại. Quá trình nghiệm thu còn phát hiện thêm một lỗi chưa được review nêu: **bóng đổ một
phía bị nhầm thành nền bàn tối** (sai 67.19 px), đã sửa về 0.00 px.

Các hạn chế còn lại được nêu trung thực ở mục 7: chưa có ground truth do người đo (chỉ đối chứng
với OpenCV), chưa xử lý motion blur, và việc chuẩn hóa hướng xoay phiếu thuộc TV2.
