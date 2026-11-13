# BÁO CÁO ĐỀ TÀI PROJECT MÔN XỬ LÝ ẢNH (INT13146)

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**  
**Khoa Công nghệ Thông tin 1 – Bộ môn Khoa học Máy tính**  
**Giảng viên hướng dẫn:** TS. Phạm Hoàng Việt  
**Tên đề tài:** Hệ thống Chấm bài Trắc nghiệm Tự động từ Ảnh chụp (Automatic Optical Mark Recognition System – OMR)  
**Quy mô nhóm:** 04 sinh viên  

---

## MỤC LỤC
1. [Phát biểu bài toán và Ứng dụng thực tế](#1-phát-biểu-bài-toán-và-ứng-dụng-thực-tế)
2. [Xây dựng Bộ dữ liệu (Dataset)](#2-xây-dựng-bộ-dữ-liệu-dataset)
3. [Phân tích đặc điểm dữ liệu đầu vào](#3-phân-tích-đặc-điểm-dữ-liệu-đầu-vào)
4. [Đề xuất Pipeline xử lý ảnh](#4-đề-xuất-pipeline-xử-lý-ảnh)
5. [Cơ sở lý thuyết và Cài đặt thuật toán (Code from Scratch)](#5-cơ-sở-lý-thuyết-và-cài-đặt-thuật-toán-code-from-scratch)
6. [Thiết kế thí nghiệm](#6-thiết-kế-thí-nghiệm)
7. [Đánh giá Định lượng và Định tính](#7-đánh-giá-định-lượng-và-định-tính)
8. [Phân tích các trường hợp thất bại (Failure Cases)](#8-phân-tích-các-trường-hợp-thất-bại-failure-cases)
9. [Xây dựng Giao diện ứng dụng và Kịch bản Demo](#9-xây-dựng-giao-diện-ứng-dụng-và-kịch-bản-demo)
10. [Phân công công việc, Tỷ lệ % đóng góp và Lộ trình thực hiện](#10-phân-công-công-việc-tỷ-lệ--đóng-góp-và-lộ-trình-thực-hiện)

---

## 1. PHÁT BIỂU BÀI TOÁN VÀ ỨNG DỤNG THỰC TẾ

### 1.1. Đặt vấn đề và Ứng dụng thực tế
Hình thức kiểm tra trắc nghiệm khách quan trên giấy (Multiple-Choice Question – MCQ) được áp dụng rộng rãi tại các trường phổ thông, đại học và trung tâm khảo thí. Tuy nhiên:
* Máy chấm thi chuyên dụng (Optical Mark Recognition – OMR) có chi phí đầu tư cao và yêu cầu giấy in phôi đặc biệt.
* Chấm thủ công bằng mắt thường hoặc bìa đục lỗ tốn nhiều thời gian, dễ xảy ra sai sót khi số lượng bài thi lên tới hàng trăm phiếu.

Việc xây dựng **Hệ thống Chấm bài Trắc nghiệm Tự động từ Ảnh chụp Điện thoại** giúp giáo viên và giảng viên sử dụng trực tiếp camera smartphone để chụp phiếu trả lời, tự động nắn thẳng phối cảnh, nhận diện đáp án tô chì/bút bi và xuất bảng điểm thống kê.

### 1.2. Phát biểu bài toán dưới góc độ Xử lý ảnh số
* **Đầu vào (Input):**
  * Ảnh màu RGB $I_{\text{raw}} \in \mathbb{R}^{H \times W \times 3}$ chụp tờ phiếu trả lời trắc nghiệm từ điện thoại (hoặc ma trận ảnh mức xám nạp từ file `.csv` phục vụ kiểm thử thuật toán).
  * Bộ đáp án chuẩn (Answer Key) lưu dưới dạng file `.csv` gồm mã câu hỏi và đáp án đúng $K = \{k_1, k_2, \dots, k_N\}$, với $k_i \in \{\text{A, B, C, D}\}$.
* **Ràng buộc kỹ thuật (Theo quy định môn học):**
  * Toàn bộ các thuật toán xử lý ảnh trong giáo trình (Chuyển đổi mức xám, Lọc không gian, Cân bằng Histogram, Phát hiện biên, Biến đổi Hough, Biến đổi phối cảnh Homography & Nội suy, So khớp mẫu NCC, Phân ngưỡng Otsu/Adaptive, Xử lý hình thái học, Gán nhãn vùng liên thông) đều phải được **cài đặt từ đầu (code from scratch) bằng `NumPy`**, tuyệt đối không gọi các hàm xử lý có sẵn của `OpenCV` trong luồng chính.
  * Thư viện `OpenCV` (`cv2`) chỉ được sử dụng độc lập để **so sánh đối chứng** về độ chính xác toán học và thời gian thực thi.
* **Đầu ra (Output):**
  * Ảnh phiếu thi đã được nắn phẳng và hiển thị trực quan kết quả chấm (khoanh xanh đáp án đúng, khoanh đỏ đáp án sai, khoanh vàng câu bỏ trống hoặc tô phạm quy).
  * Ma trận ảnh kết quả trung gian hỗ trợ xuất ra file `.csv` hoặc file ảnh `.png/.jpg`.
  * Bảng điểm tổng hợp và thống kê phổ điểm xuất ra file Excel (`Ket_Qua_Cham_Thi.xlsx`).

---

## 2. XÂY DỰNG BỘ DỮ LIỆU (DATASET)

Nhóm tự thiết kế mẫu phiếu trả lời trắc nghiệm chuẩn gồm **20 câu hỏi** (4 lựa chọn A, B, C, D mỗi câu), có khung viền chữ nhật bao quanh vùng trả lời và **04 ô vuông định vị (Fiducial Markers)** màu đen ở 4 góc phiếu.

Bộ dữ liệu thực nghiệm gồm **60 ảnh thực tế** (và **05 ma trận mức xám chuẩn dạng `.csv`**), được chia thành 5 tập kịch bản kiểm thử (mỗi tập 12 ảnh):

| Mã tập | Tên kịch bản dữ liệu | Mô tả điều kiện thu thập | Mục đích kiểm thử thuật toán |
| :--- | :--- | :--- | :--- |
| **DS-01** | **Ảnh chuẩn (Baseline)** | Chụp vuông góc ($\theta < 5^\circ$), ánh sáng đều, tô bút chì 2B đậm chuẩn, nền bàn tối màu tương phản với giấy trắng. | Kiểm tra độ chính xác cơ sở của toàn bộ pipeline. |
| **DS-02** | **Biến dạng phối cảnh (Perspective Tilt)** | Chụp nghiêng góc từ $15^\circ \rightarrow 45^\circ$, phiếu bị xoay ngang $90^\circ$ hoặc ngược $180^\circ$. | Kiểm thử thuật toán Canny + Hough tìm đỉnh, Homography SVD, Bilinear Warping và NCC Marker. |
| **DS-03** | **Chiếu sáng không đều & Bóng đổ (Uneven Lighting)** | Chụp dưới ánh đèn bàn tạo vùng chói cục bộ hoặc có bóng tay/điện thoại đè lên mặt phiếu. | Kiểm thử hiệu quả của Cân bằng Histogram và Phân ngưỡng thích nghi (Adaptive Thresholding) so với Otsu toàn cục. |
| **DS-04** | **Nhiễu hạt & Tô mờ (Low Contrast / Noise)** | Chụp trong phòng thiếu sáng (camera sinh nhiễu muối tiêu/Gaussian), thí sinh tô bút chì HB nhạt hoặc dùng bút bi xanh. | Kiểm thử Bộ lọc Gaussian, Median Filter và độ nhạy của bước tính Fill Ratio. |
| **DS-05** | **Trường hợp ngoại lệ (Edge & Failure Cases)** | Thí sinh tẩy xóa lem nhem còn vết đen, tô 2 đáp án trên cùng 1 câu, bỏ trống câu, hoặc tờ giấy bị quăn mép/nhàu. | Kiểm thử phép toán Hình thái học (Morphology), Connected Components và logic bắt lỗi phạm quy. |

---

## 3. PHÂN TÍCH ĐẶC ĐIỂM DỮ LIỆU ĐẦU VÀO

Trước khi thiết kế pipeline, nhóm tiến hành phân tích đặc trưng số học của ảnh đầu vào để làm cơ sở lựa chọn thuật toán và tham số:

1. **Đặc trưng kích thước không gian (Spatial Resolution):**
   * Ảnh gốc từ smartphone có kích thước lớn ($3000 \times 4000\text{ px}$). Nếu chạy tích chập 2D và nội suy trực tiếp trên kích thước này bằng Python sẽ gây độ trễ lớn.
   * Do đường kính mỗi ô tròn đáp án chiếm khoảng $1.8\%$ chiều cao tờ phiếu, việc chuẩn hóa chiều cao ảnh về $H = 1000\text{ px}$ giúp mỗi ô đáp án vẫn giữ được kích thước $\approx 18 \times 18\text{ px}$ ($\approx 324\text{ pixels}$), đủ lớn để phân biệt chính xác ô được tô và ô trống mà giảm được $\approx 12$ lần khối lượng tính toán.
2. **Đặc trưng phân bố mức xám (Histogram Distribution):**
   * Trên tập **DS-01 (Đủ sáng)**, biểu đồ Histogram có dạng **hai đỉnh (bimodal)** rõ rệt: đỉnh nền giấy trắng tập trung ở dải mức xám $[210, 250]$ và đỉnh chữ in/ô tô chì nằm ở dải $[20, 95]$. Với dạng này, thuật toán Otsu toàn cục tìm được ngưỡng tối ưu $T^* \approx 145$ rất chính xác.
   * Trên tập **DS-03 (Bóng đổ)**, vùng giấy nằm trong bóng râm có mức xám giảm xuống $[110, 150]$, chồng lấp với mức xám của vết bút chì ở vùng sáng ($[80, 130]$). Khi đó Histogram mất tính hai đỉnh toàn cục $\rightarrow$ Bắt buộc phải kết hợp **Cân bằng Histogram** và **Phân ngưỡng thích nghi cục bộ (Adaptive Mean Thresholding)**.
3. **Đặc trưng hình học và Nhiễu (Geometry & Noise):**
   * Phép chiếu phối cảnh khi chụp nghiêng làm các đường thẳng song song của khung phiếu hội tụ về điểm tụ (vanishing point), biến hình chữ nhật thành tứ giác lồi bất kỳ $\rightarrow$ Không thể dùng phép xoay 2D thông thường (Affine $2 \times 3$) mà phải dùng **Biến đổi phối cảnh Homography $3 \times 3$ (8 bậc tự do)**.
   * Vết tô bút chì thường có các khe hở trắng nhỏ bên trong nét gạch hoặc các chấm bụi chì rơi ngoài ô $\rightarrow$ Cần **Phép Đóng (Closing)** để lấp đầy lỗ hổng bên trong ô tô và **Phép Mở (Opening)** để xóa nhiễu hạt nhỏ.

---

## 4. ĐỀ XUẤT PIPELINE XỬ LÝ ẢNH

Hệ thống được thiết kế theo kiến trúc tuyến tính gồm **04 Giai đoạn (4 Modules)**, tương ứng với phần việc kỹ thuật độc lập của 04 thành viên:

```
[Đầu vào: Ảnh chụp phiếu thi (.jpg/.png) hoặc Ma trận mức xám (.csv)]
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1 (THÀNH VIÊN 1): Tiền xử lý & Phát hiện khung phiếu thi      │
│  1. Chuyển đổi ảnh màu RGB sang ảnh mức xám (Luminance Grayscale)       │
│  2. Làm mịn khử nhiễu bằng Bộ lọc Gaussian 2D (2D Convolution)          │
│  3. Phát hiện biên Canny (Gradient Sobel -> NMS -> Double Threshold)    │
│  4. Biến đổi Hough (Hough Line Transform) & Tìm giao điểm 4 đỉnh phiếu  │
└─────────────────────────────────────────────────────────────────────────┘
          │ (Tọa độ 4 đỉnh khung phiếu: TL, TR, BR, BL)
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2 (THÀNH VIÊN 2): Nắn thẳng phối cảnh & Căn chỉnh Marker      │
│  1. Chuẩn hóa thứ tự 4 đỉnh & Tính kích thước phẳng đích (W_dst, H_dst) │
│  2. Lập hệ phương trình Ah = 0 & Giải ma trận Homography 3x3 bằng SVD   │
│  3. Ánh xạ ngược (Inverse Warping) kết hợp Nội suy song tuyến tính      │
│  4. So khớp mẫu (Template Matching NCC) tìm 4 Marker & Xoay chuẩn hướng │
└─────────────────────────────────────────────────────────────────────────┘
          │ (Ảnh phiếu đã nắn phẳng và chuẩn hóa hướng 0°)
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3 (THÀNH VIÊN 3): Phân đoạn ảnh & Nhận diện ô tô đáp án       │
│  1. Phân ngưỡng tự động Otsu toàn cục & Phân ngưỡng thích nghi cục bộ   │
│  2. Toán tử Hình thái học: Erosion, Dilation, Opening, Closing          │
│  3. Gán nhãn thành phần liên thông (Two-Pass Connected Components)      │
│  4. Trích xuất ROI từng ô A/B/C/D & Phân loại theo tỷ lệ lấp đầy (Fill) │
└─────────────────────────────────────────────────────────────────────────┘
          │ (Vector đáp án nhận diện + Trạng thái hợp lệ từng câu)
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 4 (THÀNH VIÊN 4): Tăng cường ảnh, Chấm điểm & Giao diện Demo  │
│  1. Cân bằng biểu đồ mức xám (Histogram Equalization) xử lý ảnh tối     │
│  2. Bộ lọc trung vị (Median Filter) khử nhiễu muối tiêu                 │
│  3. Đối chiếu đáp án chuẩn, chấm điểm, vẽ kết quả trực quan & xuất XLSX │
│  4. Tích hợp GUI Streamlit (Hỗ trợ ảnh thật & CSV) + Benchmark OpenCV   │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
[Đầu ra: Ảnh chấm trực quan + Báo cáo Benchmark Scratch vs. OpenCV + File Excel]
```

---

## 5. CƠ SỞ LÝ THUYẾT VÀ CÀI ĐẶT THUẬT TOÁN (CODE FROM SCRATCH)

Toàn bộ các thuật toán dưới đây được cài đặt thuần túy bằng phép toán ma trận trên `NumPy`. Nhóm áp dụng kỹ thuật vector hóa (`np.lib.stride_tricks.sliding_window_view`, `np.einsum`, broadcasting) để tối ưu tốc độ thay vì dùng vòng lặp `for` lồng nhau chậm chạp.

---

### 5.1. Module Thành viên 1: Tiền xử lý & Phát hiện khung phiếu (Sheet Detection)

#### (1) Chuyển đổi ảnh mức xám (Grayscale Conversion)
Từ ảnh màu 3 kênh $I_{\text{RGB}}(x, y) = [R(x,y), G(x,y), B(x,y)]$, mức xám cảm nhận theo thị giác người (chuẩn ITU-R BT.601) được tính bằng tổ hợp tuyến tính:
$$I_{\text{gray}}(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$

#### (2) Bộ lọc Gaussian & Phép tích chập 2D (Gaussian Blur & 2D Convolution)
* **Sinh ma trận nhân Gaussian (Gaussian Kernel) kích thước $k \times k$ (với $k = 2r + 1$):**
  $$G(u, v) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{u^2 + v^2}{2\sigma^2}\right), \quad u, v \in [-r, r]$$
  Chuẩn hóa ma trận nhân để tổng trọng số bằng $1$: $K_{\text{gauss}} = \frac{G}{\sum_{u=-r}^{r}\sum_{v=-r}^{r} G(u, v)}$.
* **Tích chập 2D (2D Spatial Convolution) có đệm biên đối xứng (Reflect Padding):**
  $$I_{\text{blur}}(x, y) = (I_{\text{gray}} * K_{\text{gauss}})(x, y) = \sum_{u=-r}^{r} \sum_{v=-r}^{r} I_{\text{padded}}(x + r - u, y + r - v) \cdot K_{\text{gauss}}(u, v)$$

#### (3) Thuật toán Phát hiện biên Canny (4 bước chuẩn giáo trình)
* **Bước 1 – Tính đạo hàm bậc nhất bằng toán tử Sobel $3 \times 3$:**
  $$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix}$$
  $$G_x = I_{\text{blur}} * K_x, \quad G_y = I_{\text{blur}} * K_y$$
  Độ lớn biên (Gradient Magnitude) và hướng pháp tuyến biên (Gradient Direction):
  $$M(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}, \quad \theta(x, y) = \text{atan2}(G_y(x, y), G_x(x, y)) \in [0^\circ, 180^\circ)$$
* **Bước 2 – Triệt tiêu không cực đại (Non-Maximum Suppression – NMS):**
  Lượng tử hóa góc $\theta(x, y)$ về 4 hướng chính: $0^\circ$ (ngang), $45^\circ$ (chéo phụ), $90^\circ$ (dọc), $135^\circ$ (chéo chính). Tại mỗi điểm $(x, y)$, so sánh $M(x, y)$ với 2 điểm lân cận dọc theo hướng gradient. Nếu $M(x, y)$ không phải cực đại cục bộ thì gán $M_{\text{NMS}}(x, y) = 0$.
* **Bước 3 & 4 – Phân ngưỡng kép (Double Thresholding) và Nối biên trễ (Hysteresis):**
  Sử dụng hai ngưỡng $T_{\text{high}}$ và $T_{\text{low}} = 0.4 \cdot T_{\text{high}}$:
  * Điểm biên mạnh (Strong Edge): $M_{\text{NMS}}(x, y) \ge T_{\text{high}} \Rightarrow E(x, y) = 255$.
  * Điểm biên yếu (Weak Edge): $T_{\text{low}} \le M_{\text{NMS}}(x, y) < T_{\text{high}} \Rightarrow E(x, y) = 75$.
  * Điểm loại bỏ: $M_{\text{NMS}}(x, y) < T_{\text{low}} \Rightarrow E(x, y) = 0$.
  * Duyệt lan truyền 8-lân cận (BFS/DFS) từ các điểm biên mạnh: mọi điểm biên yếu kết nối liên thông với biên mạnh được nâng lên $255$, các điểm biên yếu cô lập bị xóa về $0$.

#### (4) Biến đổi Hough tìm đường thẳng (Hough Line Transform) & Xác định 4 đỉnh
* Chuyển mỗi điểm biên $(x, y)$ sang không gian tham số cực $(\rho, \theta)$ theo phương trình đường thẳng chuẩn:
  $$\rho = x \cos\theta + y \sin\theta, \quad \theta \in [-\frac{\pi}{2}, \frac{\pi}{2}), \quad \rho \in [-D, D] \text{ với } D = \sqrt{W^2 + H^2}$$
* Khởi tạo mảng tích lũy (Accumulator Array) $A(\rho_i, \theta_j)$ và bỏ phiếu (voting). Chọn 4 cực đại lớn nhất tương ứng với 2 cạnh ngang và 2 cạnh dọc của khung phiếu, sau đó giải hệ phương trình tuyến tính tìm 4 giao điểm (4 góc khung phiếu).

---

### 5.2. Module Thành viên 2: Nắn thẳng phối cảnh & Định vị Marker (Perspective & Registration)

#### (1) Chuẩn hóa tọa độ 4 đỉnh khung phiếu
Giả sử tập 4 đỉnh tìm được là $P = \{p_1, p_2, p_3, p_4\}$ với $p_i = (x_i, y_i)$. Ta sắp xếp theo quy tắc hình học:
* Đỉnh **Top-Left ($p_{\text{TL}}$)** có tổng $(x_i + y_i)$ nhỏ nhất; **Bottom-Right ($p_{\text{BR}}$)** có tổng $(x_i + y_i)$ lớn nhất.
* Đỉnh **Top-Right ($p_{\text{TR}}$)** có hiệu $(y_i - x_i)$ nhỏ nhất; **Bottom-Left ($p_{\text{BL}}$)** có hiệu $(y_i - x_i)$ lớn nhất.

#### (2) Ước lượng Ma trận Biến đổi Phối cảnh Homography $3 \times 3$ bằng SVD
Quan hệ giữa điểm nguồn $(x, y)$ và điểm đích trên mặt phẳng chuẩn $(x', y')$ trong tọa độ thuần nhất:
$$s \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = H \mathbf{x}$$
Khử hệ số tỷ lệ $s$, mỗi cặp điểm tương ứng $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$ sinh ra 2 phương trình tuyến tính:
$$\begin{bmatrix} -x_i & -y_i & -1 & 0 & 0 & 0 & x'_i x_i & x'_i y_i & x'_i \\ 0 & 0 & 0 & -x_i & -y_i & -1 & y'_i x_i & y'_i y_i & y'_i \end{bmatrix} \mathbf{h} = \mathbf{0}$$
Ghép 4 cặp đỉnh được hệ phương trình thuần nhất $A \mathbf{h} = \mathbf{0}$ với $A \in \mathbb{R}^{8 \times 9}$. Phân tích giá trị kỳ dị $A = U \Sigma V^T$ (dùng `np.linalg.svd`), nghiệm $\mathbf{h}$ chính là cột cuối cùng của $V$ (hàng cuối của $V^T$ ứng với giá trị kỳ dị nhỏ nhất), sau đó chuẩn hóa $H = H / h_{33}$.

#### (3) Ánh xạ ngược & Nội suy song tuyến tính (Inverse Warping & Bilinear Interpolation)
Để tránh hiện tượng lỗ thủng pixel (holes) khi biến đổi thuận, nhóm tạo lưới tọa độ đích $(x', y') \in [0, W_{\text{dst}}-1] \times [0, H_{\text{dst}}-1]$ và nhân với ma trận nghịch đảo $H^{-1}$ để tìm tọa độ thực $(x, y)$ trên ảnh gốc:
$$\begin{bmatrix} \tilde{x} & \tilde{y} & \tilde{w} \end{bmatrix}^T = H^{-1} \begin{bmatrix} x' & y' & 1 \end{bmatrix}^T \implies x = \frac{\tilde{x}}{\tilde{w}}, \quad y = \frac{\tilde{y}}{\tilde{w}}$$
Đặt $x_0 = \lfloor x \rfloor, x_1 = x_0 + 1, y_0 = \lfloor y \rfloor, y_1 = y_0 + 1$ và phần lẻ $\alpha = x - x_0, \beta = y - y_0$. Giá trị mức xám tại $(x', y')$ được nội suy từ 4 điểm nguyên lân cận:
$$I_{\text{warp}}(x', y') = (1 - \alpha)(1 - \beta)I(x_0, y_0) + \alpha(1 - \beta)I(x_1, y_0) + (1 - \alpha)\beta I(x_0, y_1) + \alpha\beta I(x_1, y_1)$$

#### (4) So khớp mẫu Tương quan chéo chuẩn hóa (Normalized Cross-Correlation – NCC)
Để kiểm tra hướng xoay của phiếu (0°, 90°, 180°, 270°) và tinh chỉnh khung lưới câu hỏi, nhóm cài đặt thuật toán NCC giữa mẫu Marker góc $T(u, v)$ kích thước $m \times n$ và vùng ảnh $I(x+u, y+v)$:
$$\text{NCC}(x, y) = \frac{\sum_{u,v} \left[I(x+u, y+v) - \bar{I}_{x,y}\right] \left[T(u, v) - \bar{T}\right]}{\sqrt{\sum_{u,v} \left[I(x+u, y+v) - \bar{I}_{x,y}\right]^2 \cdot \sum_{u,v} \left[T(u, v) - \bar{T}\right]^2}}$$
Điểm có $\text{NCC}(x, y) \in [-1, 1]$ đạt cực đại tại 4 vùng góc chính là tâm của 4 Marker định vị.

---

### 5.3. Module Thành viên 3: Phân đoạn ảnh & Nhận diện ô tô (Segmentation & Mark Recognition)

#### (1) Phân ngưỡng tự động Otsu (Otsu's Global Thresholding)
* Tính phân bố xác suất chuẩn hóa từ biểu đồ mức xám: $p_i = \frac{n_i}{N}$ với $i \in [0, 255]$.
* Với mỗi ngưỡng thử $t \in [0, 255]$, chia pixel thành lớp tiền cảnh $C_0 = [0, t]$ và hậu cảnh $C_1 = [t+1, 255]$:
  $$\omega_0(t) = \sum_{i=0}^{t} p_i, \quad \omega_1(t) = 1 - \omega_0(t)$$
  $$\mu_0(t) = \frac{\sum_{i=0}^{t} i \cdot p_i}{\omega_0(t)}, \quad \mu_1(t) = \frac{\sum_{i=t+1}^{255} i \cdot p_i}{\omega_1(t)}$$
* Tìm ngưỡng tối ưu $T^*$ cực đại hóa phương sai giữa hai lớp (Between-class variance):
  $$T^* = \arg\max_{0 \le t \le 255} \sigma_B^2(t) = \arg\max_{0 \le t \le 255} \left[ \omega_0(t)\omega_1(t)\left(\mu_0(t) - \mu_1(t)\right)^2 \right]$$

#### (2) Phân ngưỡng thích nghi cục bộ (Adaptive Mean Thresholding – Dùng ảnh tích phân)
Để xử lý bóng đổ, ngưỡng tại mỗi điểm $(x, y)$ được tính bằng trung bình cục bộ trong cửa sổ $b \times b$ quanh nó trừ đi hằng số $C$. Nhóm cài đặt bảng **Ảnh tích phân (Integral Image)** $S(x, y) = \sum_{i \le x, j \le y} I(i, j)$ giúp tính tổng cửa sổ bất kỳ chỉ với $O(1)$ phép cộng trừ:
$$T(x, y) = \frac{1}{b^2} \sum_{u=-r}^{r} \sum_{v=-r}^{r} I(x+u, y+v) - C, \quad B(x, y) = \begin{cases} 255 & \text{nếu } I(x, y) < T(x, y) \text{ (Đảo nền đen - chữ trắng)} \\ 0 & \text{ngược lại} \end{cases}$$

#### (3) Xử lý hình thái học nhị phân (Morphological Operations)
Với phần tử cấu trúc (Structuring Element) $S$ kích thước $3 \times 3$:
* **Phép Co (Erosion):** $(B \ominus S)(x, y) = \min_{(u,v) \in S} B(x+u, y+v)$ — làm mảnh viền khung tròn in sẵn và xóa chấm nhiễu rời rạc.
* **Phép Giãn (Dilation):** $(B \oplus S)(x, y) = \max_{(u,v) \in S} B(x+u, y+v)$ — làm đầy vùng bút chì tô.
* **Phép Mở (Opening):** $B \circ S = (B \ominus S) \oplus S$.
* **Phép Đóng (Closing):** $B \bullet S = (B \oplus S) \ominus S$.

#### (4) Gán nhãn thành phần liên thông (Two-Pass Connected Component Labeling) & Tính Fill Ratio
* Cài đặt thuật toán duyệt 2 lượt (Two-Pass CCL) kèm cấu trúc dữ liệu **Disjoint-Set (Union-Find)** để tách các vùng pixel trắng liên thông, lọc bỏ các thành phần có diện tích hoặc tỷ lệ khung hình (Aspect Ratio) không phải ô tròn trắc nghiệm.
* Với câu hỏi thứ $q \in \{1 \dots 20\}$ và phương án $c \in \{\text{A, B, C, D}\}$, tính tỷ lệ lấp đầy (Fill Ratio) trong mặt nạ hình tròn bán kính $R$:
  $$\tau_{q, c} = \frac{1}{|\Omega_{q, c}|} \sum_{(x, y) \in \Omega_{q, c}} \frac{B_{\text{morph}}(x, y)}{255}$$
* **Quy tắc phân loại hợp lệ:**
  * Phương án $c$ được coi là có tô nếu $\tau_{q, c} \ge \tau_{\text{min}}$ (với $\tau_{\text{min}} = 0.38$).
  * Nếu câu $q$ không có phương án nào $\ge \tau_{\text{min}} \Rightarrow$ **Bỏ trống (Blank)**.
  * Nếu câu $q$ có $\ge 2$ phương án đạt ngưỡng và độ chênh lệch giữa 2 ô cao nhất $\Delta \tau < 0.15 \Rightarrow$ **Tô nhiều đáp án / Phạm quy (Multiple Marked)**.
  * Nếu có 2 ô vượt ngưỡng nhưng $\Delta \tau \ge 0.15$ (do thí sinh tẩy 1 ô còn vết mờ và tô đậm ô khác) $\Rightarrow$ Chọn ô có $\tau_{q, c}$ lớn nhất.

---

### 5.4. Module Thành viên 4: Tăng cường chất lượng ảnh, Chấm điểm & Tích hợp hệ thống

#### (1) Cân bằng biểu đồ mức xám (Histogram Equalization)
Biến đổi mức xám đầu vào $k \in [0, 255]$ sang mức xám mới $s_k$ thông qua hàm phân phối tích lũy chuẩn hóa cực tiểu ($\text{CDF}_{\min}$) nhằm trải đều dải động cho ảnh chụp thiếu sáng (khớp 100% với chuẩn `cv2.equalizeHist` của OpenCV):
$$\text{CDF}(k) = \sum_{j=0}^{k} n_j, \qquad s_k = T(k) = \text{round}\left( \frac{\text{CDF}(k) - \text{CDF}_{\min}}{H \times W - \text{CDF}_{\min}} \times 255 \right)$$
*(với $\text{CDF}_{\min}$ là giá trị tích lũy tại mức xám khác 0 đầu tiên $k_{\min}$, và $s_{k_{\min}} = 0$; trường hợp ảnh đồng nhất hoàn toàn $H \times W = \text{CDF}_{\min}$ thì giữ nguyên mức xám gốc).*

#### (2) Bộ lọc trung vị (Median Filter)
Quét cửa sổ lân cận $k \times k$ qua từng pixel, sắp xếp mảng $k^2$ giá trị mức xám và lấy phần tử trung vị (phần tử thứ $\lfloor k^2 / 2 \rfloor$). Giúp triệt tiêu hoàn toàn nhiễu muối tiêu (Salt-and-Pepper Noise) mà không làm nhòe biên chữ như bộ lọc trung bình.

#### (3) Đối chiếu đáp án, Chấm điểm & Xuất báo cáo
* So sánh vector đáp án nhận diện $\hat{Y} = [\hat{y}_1, \hat{y}_2, \dots, \hat{y}_N]$ với đáp án chuẩn $K = [k_1, k_2, \dots, k_N]$.
* Tính tổng điểm trên thang 10: $\text{Score} = \frac{10}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i == k_i)$.
* Vẽ đè lớp thông tin (Overlay) lên ảnh phiếu đã nắn phẳng và kết xuất báo cáo tổng hợp ra file `.xlsx`.

---

## 6. THIẾT KẾ THÍ NGHIỆM

Để đánh giá khách quan chất lượng các thuật toán tự cài đặt so với thư viện chuẩn `OpenCV`, nhóm thiết kế **03 nhóm thí nghiệm**:

### 6.1. Thí nghiệm 1: Kiểm chứng độ chính xác số học từng hàm (Unit Benchmark: Scratch vs. OpenCV)
Chạy độc lập từng hàm tự code bằng `NumPy` và hàm tương ứng của `OpenCV` trên cùng một tập ảnh đầu vào (và ma trận `.csv` mẫu) với cùng bộ tham số:

| Thành viên | Thuật toán Code from Scratch (`NumPy`) | Hàm `OpenCV` đối chứng | Tham số cố định khi thí nghiệm |
| :--- | :--- | :--- | :--- |
| **TV1** | `grayscale_luminance()`<br>`gaussian_blur_2d()`<br>`canny_edge_detector()`<br>`hough_lines_standard()` | `cv2.cvtColor(COLOR_RGB2GRAY)`<br>`cv2.GaussianBlur()`<br>`cv2.Canny()`<br>`cv2.HoughLines()` | Kernel $k = 5 \times 5, \sigma = 1.4$<br>$T_{\text{low}} = 50, T_{\text{high}} = 150$<br>$\Delta\rho = 1\text{ px}, \Delta\theta = \pi/180$ |
| **TV2** | `solve_homography_svd()`<br>`warp_perspective_bilinear()`<br>`match_template_ncc()` | `cv2.getPerspectiveTransform()`<br>`cv2.warpPerspective()`<br>`cv2.matchTemplate(TM_CCOEFF_NORMED)` | Kích thước đích $800 \times 1000$<br>Nội suy Bilinear<br>Template Marker $32 \times 32$ |
| **TV3** | `otsu_threshold()`<br>`adaptive_mean_threshold()`<br>`morphology_op()`<br>`connected_components_2pass()` | `cv2.threshold(THRESH_OTSU)`<br>`cv2.adaptiveThreshold(ADAPTIVE_THRESH_MEAN_C)`<br>`cv2.morphologyEx()`<br>`cv2.connectedComponents()` | Block size $b = 25, C = 10$<br>Structuring Element $3 \times 3$<br>Liên thông 8 hướng (8-connectivity) |
| **TV4** | `histogram_equalization()`<br>`median_filter_2d()` | `cv2.equalizeHist()`<br>`cv2.medianBlur()` | Dải mức xám $[0, 255]$<br>Kích thước cửa sổ $k = 3 \times 3$ |

### 6.2. Thí nghiệm 2: Khảo sát tham số (Ablation Study)
* So sánh **Phân ngưỡng Otsu toàn cục** vs. **Phân ngưỡng thích nghi (Adaptive Mean)** trên tập ảnh bóng đổ **DS-03**.
* Khảo sát ảnh hưởng của ngưỡng lấp đầy $\tau_{\text{min}} \in \{0.25, 0.30, 0.35, 0.38, 0.45, 0.50\}$ đến tỷ lệ nhận diện nhầm vết tẩy xóa (False Positive) và bỏ sót vết chì mờ (False Negative).

### 6.3. Thí nghiệm 3: Đánh giá hiệu năng toàn diện (End-to-End Pipeline Evaluation)
Đo độ chính xác chấm điểm và thời gian chạy trung bình (ms/phiếu) của toàn bộ pipeline khi chạy ở chế độ **100% Code Scratch** so với chế độ **100% OpenCV** trên cả 5 tập dữ liệu (`DS-01` $\rightarrow$ `DS-05`).

---

## 7. ĐÁNH GIÁ ĐỊNH LƯỢNG VÀ ĐỊNH TÍNH

### 7.1. Các chỉ số Đánh giá Định lượng (Quantitative Metrics)

1. **Đánh giá sai số mức pixel (Dành cho Lọc ảnh, Cân bằng sáng, Biến đổi phối cảnh):**
   * **Sai số bình phương trung bình (MSE):**
     $$\text{MSE} = \frac{1}{HW} \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} \left[ I_{\text{scratch}}(x, y) - I_{\text{OpenCV}}(x, y) \right]^2$$
   * **Tỷ số tín hiệu cực đại trên nhiễu (PSNR – đơn vị dB):**
     $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$$
     *(Mục tiêu nghiệm thu: $\text{MSE} < 1.0$, $\text{PSNR} > 48\text{ dB}$ — sai lệch chỉ do làm tròn số thực dấu phẩy động ở biên ảnh).*
2. **Đánh giá độ trùng khớp phân đoạn nhị phân (Dành cho Canny, Otsu, Adaptive, Morphology):**
   * **Độ lệch ngưỡng Otsu:** $\Delta T = |T^*_{\text{scratch}} - T^*_{\text{OpenCV}}|$ *(Mục tiêu: $\Delta T \le 1$ mức xám)*.
   * **Hệ số giao trên hợp (IoU / Jaccard Index) và Dice Score:**
     $$\text{IoU} = \frac{|B_{\text{scratch}} \cap B_{\text{OpenCV}}|}{|B_{\text{scratch}} \cup B_{\text{OpenCV}}|}, \quad \text{Dice} = \frac{2 |B_{\text{scratch}} \cap B_{\text{OpenCV}}|}{|B_{\text{scratch}}| + |B_{\text{OpenCV}}|}$$
     *(Mục tiêu nghiệm thu: $\text{IoU} \ge 98.5\%$).*
3. **Đánh giá độ chính xác hình học & Nhận diện OMR:**
   * **Sai số định vị góc phiếu (Corner Localization Error):** Khoảng cách Euclid trung bình (px) giữa 4 đỉnh tìm được bằng Code Scratch và Ground Truth: $E_{\text{corner}} = \frac{1}{4}\sum_{i=1}^4 \| \hat{p}_i - p_i^{\text{GT}} \|_2$.
   * **Độ chính xác nhận diện ô tô (Mark Detection Accuracy):**
     $$\text{Accuracy} = \frac{\text{Tổng số câu nhận diện đúng đáp án}}{\text{Tổng số câu hỏi kiểm thử (60 phiếu } \times \text{ 20 câu = 1200 câu)}} \times 100\%$$

### 7.2. Bảng Kỳ vọng Kết quả Thực nghiệm (Mẫu bảng số liệu báo cáo)

| Tập dữ liệu | Số câu hỏi | Accuracy (Otsu toàn cục) | Accuracy (Adaptive + Morphology Scratch) | Accuracy (OpenCV) | Thời gian chạy Scratch (s/phiếu) | Thời gian chạy OpenCV (s/phiếu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DS-01 (Ảnh chuẩn)** | 240 | $100.0\%$ | **$100.0\%$** | $100.0\%$ | $0.42\text{ s}$ | $0.04\text{ s}$ |
| **DS-02 (Nghiêng $15^\circ-45^\circ$)** | 240 | $98.8\%$ | **$99.2\%$** | $99.2\%$ | $0.48\text{ s}$ | $0.05\text{ s}$ |
| **DS-03 (Bóng đổ / Chói sáng)** | 240 | $76.5\%$ | **$98.3\%$** | $98.8\%$ | $0.45\text{ s}$ | $0.04\text{ s}$ |
| **DS-04 (Nhiễu / Chì nhạt)** | 240 | $91.2\%$ | **$97.5\%$** | $97.9\%$ | $0.51\text{ s}$ | $0.05\text{ s}$ |
| **DS-05 (Tẩy xóa / Ngoại lệ)** | 240 | $88.0\%$ | **$96.7\%$** | $97.1\%$ | $0.46\text{ s}$ | $0.05\text{ s}$ |
| **Trung bình toàn tập** | **1200** | **$90.9\%$** | **$98.34\%$** | **$98.60\%$** | **$0.46\text{ s}$** | **$0.046\text{ s}$** |

### 7.3. Phương pháp Đánh giá Định tính (Qualitative Evaluation)
Trong báo cáo và trên giao diện phần mềm, mỗi bước xử lý đều được trực quan hóa bằng **bộ 3 ảnh đặt cạnh nhau (Triplet Comparison View)**:
1. **Ảnh kết quả từ Code Scratch (`NumPy`)**.
2. **Ảnh kết quả từ thư viện `OpenCV`**.
3. **Bản đồ nhiệt sai khác tuyệt đối (Difference Heatmap):** $\Delta I(x, y) = |I_{\text{scratch}}(x, y) - I_{\text{OpenCV}}(x, y)|$ (nhân hệ số khuếch đại $\times 10$ để mắt thường quan sát rõ các điểm khác biệt nhỏ ở rìa biên nếu có).

---

## 8. PHÂN TÍCH CÁC TRƯỜNG HỢP THẤT BẠI (FAILURE CASES)

Việc chỉ rõ giới hạn của thuật toán là yêu cầu quan trọng để đánh giá tính trung thực khoa học của dự án. Nhóm phân tích 04 trường hợp hệ thống hoạt động kém hoặc thất bại:

| STT | Trường hợp thất bại (Failure Case) | Nguyên nhân gốc rễ (Root Cause trong Pipeline) | Giải pháp khắc phục đề xuất |
| :---: | :--- | :--- | :--- |
| **1** | **Phiếu đặt trên mặt bàn màu trắng trùng màu giấy** | Độ tương phản mức xám giữa mép giấy và mặt bàn quá thấp ($\Delta I < 15$), khiến gradient Sobel yếu, Canny không bắt được biên ngoài và Hough Transform bắt nhầm vào đường kẻ bảng bên trong phiếu. | Yêu cầu khung viền đen đậm bao quanh vùng làm bài (Fiducial Border), hoặc tìm trực tiếp 4 Marker góc bằng NCC trước khi cắt phiếu. |
| **2** | **Giấy thi bị nhàu nát hoặc cong phồng ở giữa (Non-planar deformation)** | Phép biến đổi Homography $3 \times 3$ giả định tờ giấy là một **mặt phẳng cứng (planar surface)**. Khi giấy bị phồng cong 3D, lưới tọa độ nội suy ở vùng tâm phiếu bị lệch vài pixel so với vị trí thực của ô tròn. | Bổ sung các điểm căn chỉnh phụ (Local Alignment Markers) dọc theo từng hàng câu hỏi thay vì chỉ dùng 4 góc ngoài cùng. |
| **3** | **Bóng râm có biên sắc nét cắt ngang qua đúng 1 ô đáp án** | Khi ranh giới bóng râm đi qua giữa một ô tròn trống, nửa tối của ô có mức xám thấp hơn trung bình cửa sổ cục bộ, khiến Adaptive Thresholding tạo ra một vệt trắng chiếm $\approx 35 - 40\%$ diện tích ô (gần chạm ngưỡng $\tau_{\text{min}}$). | Áp dụng phép chia nền hình thái học (Morphological Top-Hat / Black-Hat Transform) với kernel lớn trước khi phân ngưỡng để loại bỏ hoàn toàn gradient chiếu sáng. |
| **4** | **Thí sinh dùng bút chì quá nhạt hoặc gạch chữ X mảnh thay vì tô kín ô** | Phép Co (Erosion) trong bước khử nhiễu làm đứt nét gạch mảnh, khiến tỷ lệ lấp đầy $\tau_{q,c}$ tụt xuống dưới ngưỡng $0.38 \rightarrow$ Hệ thống báo nhầm là câu bỏ trống (False Negative). | Giảm kích thước Structuring Element của phép Opening xuống $2 \times 2$ và kết hợp kiểm tra độ lệch chuẩn mức xám bên trong lòng ô tròn. |

---

## 9. XÂY DỰNG GIAO DIỆN ỨNG DỤNG VÀ KỊCH BẢN DEMO

Ứng dụng được xây dựng bằng `Streamlit` với giao diện trực quan chia làm **04 Tab chức năng** đáp ứng đầy đủ tiêu chí nghiệm thu của giảng viên:

1. **Tab 1 – Chấm bài Trắc nghiệm (Single & Batch Grading):**
   * Cho phép tải lên 1 ảnh phiếu thi hoặc cả thư mục ảnh, cùng file đáp án chuẩn (`answer_key.csv`).
   * Tuỳ chọn chế độ chạy: **`Run From Scratch (NumPy)`** hoặc **`Run OpenCV Baseline`**.
   * Hiển thị ảnh phiếu đã chấm (xanh/đỏ/vàng), bảng chi tiết từng câu (kèm giá trị Fill Ratio $\tau_{q,c}$ của cả 4 ô A, B, C, D) và nút tải xuống bảng điểm `Ket_Qua_Cham_Thi.xlsx`.
2. **Tab 2 – Kiểm thử từng Thuật toán & Hỗ trợ File CSV (Algorithm Inspector):**
   * Cho phép giảng viên **tải lên ảnh thật (`.jpg/.png`) HOẶC file ma trận mức xám (`.csv`)** theo đúng yêu cầu nghiệm thu tại Slide 10–12.
   * Cho phép điều chỉnh thanh trượt tham số trực tiếp (kích thước kernel Gaussian $k$, độ lệch chuẩn $\sigma$, ngưỡng Canny $T_{\text{low}}/T_{\text{high}}$, cửa sổ Adaptive $b$, hằng số $C$).
   * Cho phép **lưu kết quả đầu ra dưới dạng file ảnh hoặc xuất ma trận số ra file `.csv`**.
3. **Tab 3 – Đối chứng trực quan Scratch vs. OpenCV (Side-by-Side Benchmark):**
   * Hiển thị song song ảnh đầu ra của hàm tự code và hàm `OpenCV` ở từng giai đoạn, kèm theo các chỉ số đo tự động trực tiếp trên giao diện: **MSE, PSNR (dB), IoU (%), Độ lệch ngưỡng $\Delta T$, và Thời gian chạy (ms)**.
4. **Tab 4 – Phân tích Thống kê & Kiểm duyệt thủ công (Analytics & Human-in-the-loop):**
   * Biểu đồ phân bố điểm số, thống kê các câu hỏi nhiều sinh viên sai nhất, và danh sách các phiếu có cảnh báo (tô mờ/tẩy xóa) để giám thị xác nhận lại chỉ với 1 cú click.

---

## 10. PHÂN CÔNG CÔNG VIỆC, TỶ LỆ % ĐÓNG GÓP VÀ LỘ TRÌNH THỰC HIỆN

### 10.1. Bảng Phân công Công việc và Tỷ lệ % Đóng góp Kỹ thuật
*(Lưu ý theo đúng quy định của giảng viên tại Slide 9: **Tỷ lệ % đóng góp chỉ tính trên nội dung kỹ thuật và cài đặt thuật toán, tuyệt đối không tính công viết báo cáo hay làm slides** — cả 4 thành viên cùng viết báo cáo và chuẩn bị slide cho phần thuật toán của mình).*

| Thành viên | Phụ trách Module | Các thuật toán tự cài đặt (Code from Scratch bằng `NumPy`) | File mã nguồn phụ trách | Tỷ lệ đóng góp |
| :--- | :--- | :--- | :--- | :---: |
| **Thành viên 1** | **Tiền xử lý & Phát hiện khung phiếu** | 1. Chuyển đổi mức xám Luminance.<br>2. Sinh Gaussian Kernel & Tích chập 2D (2D Convolution).<br>3. Phát hiện biên Canny đầy đủ 4 bước (Sobel, NMS, Double Threshold, Hysteresis).<br>4. Biến đổi Hough tìm đường thẳng & Giao điểm 4 góc. | `src/tv1_preprocessing.py` | **25%** |
| **Thành viên 2** | **Biến đổi phối cảnh & Định vị Marker** | 1. Thuật toán sắp xếp chuẩn hóa 4 đỉnh tứ giác.<br>2. Lập ma trận hệ phương trình $Ah=0$ & Giải Homography $3 \times 3$ bằng SVD.<br>3. Ánh xạ ngược (Inverse Warping) & Nội suy Bilinear.<br>4. So khớp mẫu NCC (Normalized Cross-Correlation) định vị 4 Marker & Xoay chuẩn hướng. | `src/tv2_perspective.py` | **25%** |
| **Thành viên 3** | **Phân đoạn ảnh & Nhận diện ô tô** | 1. Phân ngưỡng tự động Otsu toàn cục (tối đa hóa $\sigma_B^2$).<br>2. Phân ngưỡng thích nghi cục bộ (Adaptive Mean với Integral Image).<br>3. Bốn phép toán Hình thái học (Erosion, Dilation, Opening, Closing).<br>4. Gán nhãn thành phần liên thông 2 lượt (Two-Pass CCL) & Phân loại Fill Ratio. | `src/tv3_segmentation.py` | **25%** |
| **Thành viên 4** | **Tăng cường ảnh, Chấm điểm & Benchmark** | 1. Cân bằng biểu đồ mức xám (Histogram Equalization bằng CDF).<br>2. Bộ lọc trung vị 2D (Median Filter) khử nhiễu muối tiêu.<br>3. Module đọc/ghi ma trận `.csv`, chấm điểm, xuất Excel.<br>4. Bộ đo Benchmark tự động (MSE, PSNR, IoU, Latency) đối chứng `OpenCV` & Giao diện `Streamlit`. | `src/tv4_enhancement_eval.py`<br>`app.py` | **25%** |

---

### 10.2. Lộ trình Thực hiện Dự án (Bám sát mốc thời gian Chương 0)

| Giai đoạn | Mốc thời gian | Nội dung công việc cụ thể | Sản phẩm đầu ra (Deliverables) |
| :--- | :--- | :--- | :--- |
| **Giai đoạn 1** | **Trước tiết 19** | Chốt đề tài, phân công 4 thành viên, thiết kế mẫu phiếu 20 câu có 4 Marker góc, in ấn và chụp bộ dữ liệu 60 ảnh + tạo 5 file test `.csv`. | Đăng ký đề tài; Thư mục `dataset/` đủ 5 kịch bản. |
| **Giai đoạn 2** | **Tiết 19 – Tiết 24** | Từng thành viên cài đặt các thuật toán `NumPy` from scratch của mình và viết script kiểm thử đối chứng với hàm `OpenCV` tương ứng. | 4 module độc lập trong `src/` đạt $\text{PSNR} > 48\text{ dB}$, $\text{IoU} > 98\%$. |
| **Giai đoạn 3** | **Tiết 25 – Tiết 28** | Tích hợp 4 module vào pipeline hoàn chỉnh; xây dựng giao diện `Streamlit` (hỗ trợ ảnh thật & CSV); chạy thực nghiệm trên 60 ảnh để xuất bảng số liệu. | Ứng dụng `app.py` chạy mượt; Bảng kết quả định lượng & ảnh định tính. |
| **Giai đoạn 4** | **Trước khi kết thúc tiết 30** | Hoàn thiện quyển báo cáo chính thức, đóng gói mã nguồn sạch kèm hướng dẫn chạy (`README.md`), upload lên Google Drive của lớp. | File Báo cáo PDF + Source code nộp đúng hạn tiết 30. |
| **Giai đoạn 5** | **Tiết 34 trở đi** | Thiết kế slide thuyết trình (12–15 phút), chuẩn bị kịch bản live demo (gồm cả test cases thành công và failure cases), ôn tập công thức toán học để bảo vệ vấn đáp cá nhân. | Slide thuyết trình + Demo trực tiếp trước giảng viên. |
