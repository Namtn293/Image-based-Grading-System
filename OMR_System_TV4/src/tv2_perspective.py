"""
Module: TV2 - Nắn thẳng phối cảnh & Định vị Marker (Perspective & Registration)
Đề tài: Hệ thống Chấm bài Trắc nghiệm Tự động (OMR)
Môn học: Xử lý ảnh số (INT13146) - Học viện Công nghệ Bưu chính Viễn thông (PTIT)
Giảng viên hướng dẫn: TS. Phạm Hoàng Việt

YÊU CẦU BẮT BUỘC:
- 100% Thuật toán tự cài đặt từ đầu (code from scratch) bằng NumPy.
- KHÔNG import hay sử dụng bất kỳ hàm xử lý ảnh có sẵn nào của OpenCV trong module này
  (OpenCV chỉ được import cục bộ trong nhánh đối chứng use_scratch=False).
"""

from typing import List, Optional, Tuple
import numpy as np


# ==============================================================================
# 1. SẮP XẾP 4 ĐỈNH & TÍNH KÍCH THƯỚC ĐÍCH
# ==============================================================================

def _order_by_angle(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 đỉnh theo góc cực quanh trọng tâm (dự phòng khi quy tắc tổng/hiệu suy biến).

    Trong hệ tọa độ ảnh (trục y hướng xuống), sắp xếp tăng dần theo atan2(dy, dx)
    cho thứ tự cùng chiều kim đồng hồ: TL -> TR -> BR -> BL. Sau đó xoay vòng để
    đỉnh có tổng (x + y) nhỏ nhất đứng đầu.
    """
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def _quad_size(ordered: np.ndarray) -> Tuple[int, int]:
    """Tính kích thước phẳng đích (W_dst, H_dst) từ 4 đỉnh đã sắp xếp [TL, TR, BR, BL]."""
    tl, tr, br, bl = ordered.astype(np.float64)
    w_dst = int(round(max(np.hypot(*(tr - tl)), np.hypot(*(br - bl)))))
    h_dst = int(round(max(np.hypot(*(bl - tl)), np.hypot(*(br - tr)))))
    return max(w_dst, 1), max(h_dst, 1)


def order_corners(pts: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Sắp xếp 4 đỉnh của tứ giác theo thứ tự chuẩn:
    [Top-Left (TL), Top-Right (TR), Bottom-Right (BR), Bottom-Left (BL)]
    và tính kích thước (W_dst, H_dst) cho ảnh phẳng đích.

    Bản chất toán học:
    - TL: Điểm có tổng (x + y) nhỏ nhất.
    - BR: Điểm có tổng (x + y) lớn nhất.
    - TR: Điểm có hiệu (y - x) nhỏ nhất (tương đương x - y lớn nhất).
    - BL: Điểm có hiệu (y - x) lớn nhất.

    Khi tứ giác bị xoay gần 45°, quy tắc tổng/hiệu có thể chọn trùng một đỉnh cho
    hai vị trí. Trường hợp đó được phát hiện và chuyển sang sắp xếp theo góc cực
    quanh trọng tâm để luôn trả về 4 đỉnh phân biệt.

    Args:
        pts (np.ndarray): Mảng shape (4, 2) chứa 4 tọa độ (x, y) bất kỳ.

    Returns:
        ordered (np.ndarray): Mảng shape (4, 2) theo thứ tự [TL, TR, BR, BL], dtype float32.
        (W_dst, H_dst) (Tuple[int, int]): Chiều rộng và chiều cao đích (làm tròn int).
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size == 8:
        pts = pts.reshape(4, 2)  # chấp nhận cả dạng (4, 1, 2) của contour
    if pts.shape != (4, 2):
        raise ValueError(f"Tọa độ đầu vào phải có shape (4, 2), nhận được: {pts.shape}")

    s = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]
    idx = [int(np.argmin(s)), int(np.argmin(diff)), int(np.argmax(s)), int(np.argmax(diff))]

    if len(set(idx)) == 4:
        ordered = pts[idx]
    else:
        # Quy tắc tổng/hiệu bị suy biến (trùng đỉnh) -> sắp xếp theo góc cực
        ordered = _order_by_angle(pts)

    return ordered.astype(np.float32), _quad_size(ordered)


# ==============================================================================
# 2. ƯỚC LƯỢNG HOMOGRAPHY BẰNG DLT + SVD
# ==============================================================================

def _normalization_matrix(pts: np.ndarray) -> np.ndarray:
    """Ma trận chuẩn hóa Hartley: dời trọng tâm về gốc, co giãn để khoảng cách TB = sqrt(2)."""
    center = pts.mean(axis=0)
    mean_dist = np.mean(np.sqrt(np.sum((pts - center) ** 2, axis=1)))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    return np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def solve_homography_svd(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Ước lượng ma trận biến đổi phối cảnh Homography H (3x3) từ 4 cặp điểm
    tương ứng bằng thuật toán Direct Linear Transformation (DLT) và phân tích SVD.

    Bản chất toán học:
    Mỗi cặp điểm (x_i, y_i) <-> (x'_i, y'_i) sinh ra 2 phương trình tuyến tính:
    [-x_i, -y_i, -1,    0,    0,  0,  x'_i * x_i, x'_i * y_i, x'_i] * h = 0
    [   0,    0,  0, -x_i, -y_i, -1,  y'_i * x_i, y'_i * y_i, y'_i] * h = 0

    Ghép 4 cặp điểm được hệ A * h = 0 (với A shape (8, 9)).
    Giải nghiệm h bằng SVD (A = U * Sigma * V^T):
    Nghiệm h tối ưu là vector cột cuối của V (hàng cuối của V^T).

    Ổn định số học (Normalized DLT - Hartley):
    Tọa độ pixel cỡ hàng nghìn khiến các cột của A chênh nhau tới ~10^6 lần.
    Trước khi lập A, điểm nguồn và đích được chuẩn hóa bằng T_src, T_dst;
    sau khi giải được H_n, khôi phục H = T_dst^(-1) * H_n * T_src.

    Args:
        src (np.ndarray): 4 điểm trên ảnh nguồn, shape (4, 2).
        dst (np.ndarray): 4 điểm trên ảnh đích, shape (4, 2).

    Returns:
        H (np.ndarray): Ma trận Homography shape (3, 3), chuẩn hóa H[2, 2] = 1.0.

    Raises:
        ValueError: Nếu 4 điểm suy biến (3 điểm thẳng hàng / trùng nhau) khiến H không xác định.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src và dst đều phải có shape (4, 2).")

    T_src = _normalization_matrix(src)
    T_dst = _normalization_matrix(dst)
    src_n = (T_src @ np.column_stack([src, np.ones(4)]).T).T[:, :2]
    dst_n = (T_dst @ np.column_stack([dst, np.ones(4)]).T).T[:, :2]

    A = np.zeros((8, 9), dtype=np.float64)
    for i in range(4):
        x, y = src_n[i]
        u, v = dst_n[i]
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    # Phân tích Singular Value Decomposition (SVD)
    _, sigma, vt = np.linalg.svd(A)

    # Hạng của A phải bằng 8; nếu giá trị kỳ dị thứ 8 ~ 0 thì nghiệm không duy nhất
    if sigma[-1] < 1e-10 * sigma[0]:
        raise ValueError("4 điểm tương ứng bị suy biến (thẳng hàng hoặc trùng nhau), không giải được Homography.")

    H_n = vt[-1, :].reshape((3, 3))
    H = np.linalg.inv(T_dst) @ H_n @ T_src

    if np.abs(H[2, 2]) < 1e-12:
        raise ValueError("Homography suy biến (h33 ~ 0).")
    return H / H[2, 2]


# ==============================================================================
# 3. INVERSE WARPING + NỘI SUY SONG TUYẾN TÍNH
# ==============================================================================

def warp_perspective_bilinear(
    img: np.ndarray,
    H: np.ndarray,
    out_size: Tuple[int, int]
) -> np.ndarray:
    """
    Nắn thẳng ảnh bằng phương pháp Ánh xạ ngược (Inverse Warping)
    kết hợp Nội suy song tuyến tính (Bilinear Interpolation).

    Vectorized 100% bằng NumPy, không dùng vòng lặp for.

    Bản chất toán học:
    1. Quét từng điểm (x', y') trên ảnh phẳng đích.
    2. Chiếu ngược về ảnh nguồn: [x, y, 1]^T ~ H^(-1) * [x', y', 1]^T.
    3. Tọa độ thực (x, y) được nội suy mức xám từ 4 điểm lân cận:
       I(x, y) = (1 - a)(1 - b)*Ia + a*(1 - b)*Ib + (1 - a)*b*Ic + a*b*Id
       với a = x - floor(x), b = y - floor(y).

    Args:
        img (np.ndarray): Ảnh đầu vào (H_src, W_src) hoặc (H_src, W_src, C).
        H (np.ndarray): Ma trận Homography (3, 3).
        out_size (Tuple[int, int]): Kích thước ảnh đích (W_dst, H_dst).

    Returns:
        output (np.ndarray): Ảnh đã nắn thẳng, shape (H_dst, W_dst) cùng số kênh, dtype uint8.
    """
    w_dst, h_dst = int(out_size[0]), int(out_size[1])
    if w_dst <= 0 or h_dst <= 0:
        raise ValueError(f"Kích thước đích không hợp lệ: {out_size}")
    h_src, w_src = img.shape[:2]
    is_color = (img.ndim == 3)

    # 1. Ma trận nghịch đảo H^(-1) để ánh xạ ngược từ đích về nguồn
    try:
        H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Ma trận Homography suy biến, không thể nghịch đảo.") from exc

    # 2. Lưới tọa độ đích (Meshgrid)
    y_dst_grid, x_dst_grid = np.meshgrid(
        np.arange(h_dst, dtype=np.float64),
        np.arange(w_dst, dtype=np.float64),
        indexing="ij"
    )

    coords_dst = np.stack(
        [x_dst_grid.ravel(), y_dst_grid.ravel(), np.ones(h_dst * w_dst, dtype=np.float64)],
        axis=0
    )

    # 3. Tìm tọa độ tương ứng trên ảnh nguồn
    coords_src = H_inv @ coords_dst

    w_scale = coords_src[2, :]
    w_scale = np.where(np.abs(w_scale) < 1e-10, 1e-10, w_scale)

    x_src = (coords_src[0, :] / w_scale).reshape((h_dst, w_dst))
    y_src = (coords_src[1, :] / w_scale).reshape((h_dst, w_dst))

    # 4. 4 điểm nguyên lân cận bao quanh (x, y)
    x0 = np.floor(x_src).astype(np.int64)
    y0 = np.floor(y_src).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    alpha = x_src - x0
    beta = y_src - y0

    # Điểm nằm ngoài ảnh nguồn sẽ gán màu đen (0), tương đương BORDER_CONSTANT của OpenCV
    mask_valid = (x_src >= 0.0) & (x_src <= w_src - 1.0) & (y_src >= 0.0) & (y_src <= h_src - 1.0)

    x0_c = np.clip(x0, 0, w_src - 1)
    x1_c = np.clip(x1, 0, w_src - 1)
    y0_c = np.clip(y0, 0, h_src - 1)
    y1_c = np.clip(y1, 0, h_src - 1)

    if is_color:
        alpha = alpha[..., np.newaxis]
        beta = beta[..., np.newaxis]
        mask_valid = mask_valid[..., np.newaxis]

    Ia = img[y0_c, x0_c].astype(np.float64)
    Ib = img[y0_c, x1_c].astype(np.float64)
    Ic = img[y1_c, x0_c].astype(np.float64)
    Id = img[y1_c, x1_c].astype(np.float64)

    interp = (
        (1.0 - alpha) * (1.0 - beta) * Ia
        + alpha * (1.0 - beta) * Ib
        + (1.0 - alpha) * beta * Ic
        + alpha * beta * Id
    )
    interp = np.where(mask_valid, interp, 0.0)
    return np.clip(np.round(interp), 0, 255).astype(np.uint8)


# ==============================================================================
# 4. SO KHỚP MẪU NCC
# ==============================================================================

def _box_sum(integral: np.ndarray, h: int, w: int) -> np.ndarray:
    """Tổng mọi cửa sổ h x w trong O(1) mỗi vị trí từ ảnh tích phân có viền 0."""
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def match_template_ncc(
    img: np.ndarray,
    template: np.ndarray
) -> np.ndarray:
    """
    So khớp mẫu bằng Tương quan chéo chuẩn hóa (Normalized Cross-Correlation - NCC).
    Tương đương với phương thức cv2.TM_CCOEFF_NORMED trong OpenCV.

    Bản chất toán học:
    NCC(x, y) = Sum( (I - I_mean) * (T - T_mean) ) / sqrt( Sum(I - I_mean)^2 * Sum(T - T_mean)^2 )
    Giá trị NCC thuộc [-1.0, 1.0]. Càng gần 1.0 nghĩa là mẫu khớp càng chính xác.

    Tối ưu hóa bộ nhớ & tốc độ:
    - Tử số: vì Sum(T - T_mean) = 0 nên Sum((I - I_mean)(T - T_mean)) = Sum(I * (T - T_mean)).
      Tính bằng einsum trực tiếp trên view cửa sổ trượt, KHÔNG tạo mảng 4 chiều trung gian.
    - Mẫu số: Sum(I - I_mean)^2 = Sum(I^2) - (Sum I)^2 / N, với Sum I và Sum I^2 của mọi cửa sổ
      lấy trong O(1) từ 2 ảnh tích phân. Với ảnh số nguyên, phép tính dùng int64 nên chính xác tuyệt đối
      (cửa sổ phẳng cho phương sai đúng bằng 0, không bị sai số làm tròn).

    Args:
        img (np.ndarray): Ảnh mức xám 2D (H, W).
        template (np.ndarray): Ảnh mẫu 2D (m, n) với m <= H, n <= W.

    Returns:
        ncc_map (np.ndarray): Bản đồ tương quan float32 shape (H - m + 1, W - n + 1).
    """
    if img.ndim != 2 or template.ndim != 2:
        raise ValueError("match_template_ncc yêu cầu ảnh và template đều là mảng 2D mức xám.")

    h_img, w_img = img.shape
    h_tmpl, w_tmpl = template.shape

    if h_tmpl > h_img or w_tmpl > w_img:
        raise ValueError("Kích thước template không được lớn hơn kích thước ảnh!")

    n_px = h_tmpl * w_tmpl
    out_shape = (h_img - h_tmpl + 1, w_img - w_tmpl + 1)

    # 1. Template trừ trung bình
    tmpl_f = template.astype(np.float64)
    t_zero = tmpl_f - tmpl_f.mean()
    t_norm = np.sqrt(np.sum(t_zero ** 2))
    if t_norm < 1e-10:
        return np.zeros(out_shape, dtype=np.float32)

    # 2. Tử số: Sum(I * (T - T_mean)) trên từng cửa sổ (view, không sao chép)
    img_f = img.astype(np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(img_f, (h_tmpl, w_tmpl))
    numerator = np.einsum("ijkl,kl->ij", windows, t_zero)

    # 3. Mẫu số: chuẩn của cửa sổ trừ trung bình qua 2 ảnh tích phân
    if np.issubdtype(img.dtype, np.integer):
        img_i = img.astype(np.int64)
        acc_dtype = np.int64
    else:
        img_i = img_f
        acc_dtype = np.float64

    integral = np.zeros((h_img + 1, w_img + 1), dtype=acc_dtype)
    integral_sq = np.zeros((h_img + 1, w_img + 1), dtype=acc_dtype)
    integral[1:, 1:] = np.cumsum(np.cumsum(img_i, axis=0), axis=1)
    integral_sq[1:, 1:] = np.cumsum(np.cumsum(img_i * img_i, axis=0), axis=1)

    win_sum = _box_sum(integral, h_tmpl, w_tmpl)
    win_sum_sq = _box_sum(integral_sq, h_tmpl, w_tmpl)

    # N * Sum(I - I_mean)^2 = N * Sum(I^2) - (Sum I)^2   (chính xác với số nguyên)
    win_var_n = (n_px * win_sum_sq - win_sum * win_sum).astype(np.float64)
    win_var_n = np.maximum(win_var_n, 0.0)
    win_norm = np.sqrt(win_var_n / n_px)

    # 4. NCC map (cửa sổ phẳng có phương sai 0 -> gán 0)
    denominator = win_norm * t_norm
    valid = denominator > 1e-10
    ncc_map = np.zeros(out_shape, dtype=np.float32)
    ncc_map[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)

    return np.clip(ncc_map, -1.0, 1.0)


# ==============================================================================
# 5. ĐỊNH VỊ MARKER & XOAY CHUẨN HƯỚNG
# ==============================================================================

def detect_markers_and_correct_rotation(
    warped_img: np.ndarray,
    anchor_template: np.ndarray,
    search_margin_ratio: float = 0.25
) -> Tuple[np.ndarray, int, List[Tuple[str, float]]]:
    """
    Xác định vị trí Marker đặc biệt (Anchor Marker) ở 4 góc để tự động
    nhận biết góc xoay của phiếu thi (0°, 90°, 180°, 270°) và xoay phiếu về 0°.

    Giả định thiết kế phiếu chuẩn:
    - Marker đặc biệt (Anchor Marker) được đặt ở góc Top-Right (TR) ở trạng thái chuẩn 0°.
      (3 góc còn lại dùng marker khác mẫu, nếu không thì không thể phân biệt hướng.)
    - Khi phiếu bị xoay (np.rot90 xoay NGƯỢC chiều kim đồng hồ với k > 0):
      + Anchor ở TR  --> 0°   (Đúng chiều, không xoay).
      + Anchor ở BR  --> Phiếu xoay 90° thuận chiều kim đồng hồ  --> Xoay lại bằng np.rot90(k=1).
      + Anchor ở BL  --> Phiếu xoay 180°                         --> Xoay lại bằng np.rot90(k=2).
      + Anchor ở TL  --> Phiếu xoay 270° thuận (= 90° ngược)     --> Xoay lại bằng np.rot90(k=3).

    Args:
        warped_img (np.ndarray): Ảnh phiếu đã được nắn thẳng (mức xám hoặc màu).
        anchor_template (np.ndarray): Mẫu Marker đặc biệt (mức xám 2D).
        search_margin_ratio (float): Tỷ lệ vùng góc dùng để quét tìm Marker (mặc định 25%).

    Returns:
        corrected_img (np.ndarray): Ảnh đã được xoay chuẩn hướng 0°.
        detected_angle (int): Góc xoay phát hiện được (0, 90, 180, hoặc 270).
        corner_scores (List[Tuple[str, float]]): Điểm số NCC cực đại tại 4 góc [TL, TR, BR, BL].
    """
    gray = warped_img
    if gray.ndim == 3:
        # Chuyển mức xám BT.601 (giả định BGR như cv2.imread) chỉ để so khớp
        gray = (0.114 * gray[..., 0] + 0.587 * gray[..., 1] + 0.299 * gray[..., 2])

    h, w = gray.shape[:2]
    mh = int(h * search_margin_ratio)
    mw = int(w * search_margin_ratio)

    rois = {
        "TL": gray[0:mh, 0:mw],
        "TR": gray[0:mh, w - mw:w],
        "BR": gray[h - mh:h, w - mw:w],
        "BL": gray[h - mh:h, 0:mw],
    }

    max_scores = {}
    for corner, roi in rois.items():
        if roi.shape[0] >= anchor_template.shape[0] and roi.shape[1] >= anchor_template.shape[1]:
            max_scores[corner] = float(np.max(match_template_ncc(roi, anchor_template)))
        else:
            max_scores[corner] = -1.0

    best_corner = max(max_scores, key=max_scores.get)
    angle = {"TR": 0, "BR": 90, "BL": 180, "TL": 270}[best_corner]
    k_fix = {0: 0, 90: 1, 180: 2, 270: 3}[angle]
    corrected_img = np.rot90(warped_img, k=k_fix).copy() if k_fix else warped_img.copy()

    return corrected_img, angle, list(max_scores.items())


def _corners_for_angle(ordered: np.ndarray, angle: int) -> np.ndarray:
    """Gán lại 4 đỉnh trên ảnh gốc cho đúng vị trí [TL, TR, BR, BL] của phiếu ở hướng 0°.

    Phiếu xoay thuận chiều kim đồng hồ `angle` độ thì đỉnh TL thật nằm ở vị trí
    thứ (angle / 90) theo chiều kim đồng hồ -> dịch vòng mảng đỉnh đi -(angle / 90).
    """
    return np.roll(ordered, -(angle // 90), axis=0)


# ==============================================================================
# 6. HÀM ĐÓNG GÓI PIPELINE TV2
# ==============================================================================

def process_perspective_pipeline(
    img: np.ndarray,
    raw_corners: np.ndarray,
    anchor_template: Optional[np.ndarray] = None,
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, dict]:
    """
    Hàm đóng gói toàn bộ Pipeline của Thành viên 2 (TV2) thành 1 hàm duy nhất:
    Nhận đầu vào từ TV1 -> Xử lý toàn bộ -> Trả ảnh phẳng chuẩn 0° cho TV3.

    Các bước tự động:
    1. Sắp xếp 4 đỉnh theo thứ tự chuẩn [TL, TR, BR, BL].
    2. Xác định kích thước ảnh phẳng (W_dst, H_dst).
    3. Ước lượng ma trận Homography bằng SVD.
    4. Nắn phẳng ảnh bằng Inverse Warping + Bilinear Interpolation.
    5. (Tùy chọn) Nhận diện Marker bằng NCC để tìm góc xoay. Khi phiếu bị xoay,
       4 đỉnh được gán lại cho đúng hướng 0° rồi nắn lại MỘT lần từ ảnh gốc,
       nên ảnh ra luôn đúng kích thước target_size (không bị méo khi xoay 90°/270°)
       và chỉ chịu một lần nội suy.

    Args:
        img (np.ndarray): Ảnh gốc đầu vào (mức xám hoặc màu).
        raw_corners (np.ndarray): 4 đỉnh tứ giác nhận từ TV1 (shape 4x2).
        anchor_template (np.ndarray, optional): Ảnh mẫu Marker đặc biệt để xoay hướng.
        target_size (Tuple[int, int], optional): Kích thước ép buộc (W, H). Nếu None, tự tính.

    Returns:
        warped_sheet (np.ndarray): Ảnh phiếu thi phẳng chuẩn 0° bàn giao cho TV3.
        info (dict): Metadata chi tiết (H_matrix, ordered_corners, detected_angle, v.v.).
    """
    ordered_pts, auto_size = order_corners(raw_corners)

    def _warp(corners: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        w_out, h_out = size
        # Đích là các tâm pixel biên: [0, W-1] x [0, H-1] (khớp cv2.getPerspectiveTransform)
        dst_pts = np.array([
            [0.0, 0.0],
            [w_out - 1.0, 0.0],
            [w_out - 1.0, h_out - 1.0],
            [0.0, h_out - 1.0],
        ], dtype=np.float64)
        H_mat = solve_homography_svd(corners, dst_pts)
        return warp_perspective_bilinear(img, H_mat, (w_out, h_out)), H_mat

    detected_angle = 0
    scores: List[Tuple[str, float]] = []

    if anchor_template is None:
        out_size = tuple(target_size) if target_size is not None else auto_size
        warped, H = _warp(ordered_pts, out_size)
        final_corners = ordered_pts
    else:
        # Lượt 1: nắn theo đúng tỉ lệ khung hình đang thấy trên ảnh để dò Marker
        preview_size = auto_size
        if target_size is not None:
            tw, th = target_size
            quad_landscape = auto_size[0] > auto_size[1]
            target_landscape = tw > th
            preview_size = (tw, th) if quad_landscape == target_landscape else (th, tw)
        preview, _ = _warp(ordered_pts, preview_size)
        _, detected_angle, scores = detect_markers_and_correct_rotation(preview, anchor_template)

        # Lượt 2: gán lại 4 đỉnh cho hướng 0° và nắn trực tiếp từ ảnh gốc
        final_corners = _corners_for_angle(ordered_pts, detected_angle)
        out_size = tuple(target_size) if target_size is not None else _quad_size(final_corners)
        warped, H = _warp(final_corners, out_size)

    info = {
        "ordered_corners": final_corners,
        "homography_matrix": H,
        "output_size": out_size,
        "detected_angle": detected_angle,
        "marker_scores": scores,
    }
    return warped, info


def warp_perspective_sheet(
    image: np.ndarray,
    corners: np.ndarray,
    target_width: int = 800,
    target_height: int = 1000,
    use_scratch: bool = True,
    anchor_template: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Hàm chuẩn giao tiếp kết nối trực tiếp với Pipeline Coordinator chung của nhóm:
    Hỗ trợ gọi ở chế độ use_scratch=True (NumPy Scratch của TV2) hoặc use_scratch=False (OpenCV đối chứng).

    Args:
        image (np.ndarray): Ảnh đầu vào (mức xám hoặc màu).
        corners (np.ndarray): Tọa độ 4 góc (shape 4x2).
        target_width (int): Chiều rộng ảnh chuẩn (mặc định 800).
        target_height (int): Chiều cao ảnh chuẩn (mặc định 1000).
        use_scratch (bool): True = dùng 100% NumPy của TV2; False = dùng OpenCV.
        anchor_template (np.ndarray, optional): Nếu truyền vào (chế độ Scratch), tự động
            nhận diện và sửa hướng xoay 90°/180°/270° bằng NCC.

    Returns:
        warped (np.ndarray): Ảnh đã nắn thẳng, shape (target_height, target_width).
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)

    if use_scratch:
        warped, _ = process_perspective_pipeline(
            image, pts, anchor_template=anchor_template,
            target_size=(target_width, target_height)
        )
        return warped

    import cv2
    ordered_pts, _ = order_corners(pts)
    dst_pts = np.array([
        [0.0, 0.0],
        [float(target_width - 1), 0.0],
        [float(target_width - 1), float(target_height - 1)],
        [0.0, float(target_height - 1)]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    return cv2.warpPerspective(image, matrix, (target_width, target_height))
