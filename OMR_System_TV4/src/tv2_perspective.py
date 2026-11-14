"""
Module: TV2 - Nắn thẳng phối cảnh & Định vị Marker (Perspective & Registration)
Đề tài: Hệ thống Chấm bài Trắc nghiệm Tự động (OMR)
Môn học: Xử lý ảnh số (INT13146) - Học viện Công nghệ Bưu chính Viễn thông (PTIT)
Giảng viên hướng dẫn: TS. Phạm Hoàng Việt

YÊU CẦU BẮT BUỘC:
- 100% Thuật toán tự cài đặt từ đầu (code from scratch) bằng NumPy.
- KHÔNG import hay sử dụng bất kỳ hàm xử lý ảnh có sẵn nào của OpenCV trong module này.
"""

from typing import List, Tuple
import numpy as np


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

    Args:
        pts (np.ndarray): Mảng shape (4, 2) chứa 4 tọa độ (x, y) bất kỳ.

    Returns:
        ordered (np.ndarray): Mảng shape (4, 2) theo thứ tự [TL, TR, BR, BL], dtype float32.
        (W_dst, H_dst) (Tuple[int, int]): Chiều rộng và chiều cao đích đã làm tròn int.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Tọa độ đầu vào phải có shape (4, 2), nhận được: {pts.shape}")

    ordered = np.zeros((4, 2), dtype=np.float32)

    # 1. Tính tổng (x + y)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # Top-Left (TL)
    ordered[2] = pts[np.argmax(s)]  # Bottom-Right (BR)

    # 2. Tính hiệu (y - x)
    diff = pts[:, 1] - pts[:, 0]
    ordered[1] = pts[np.argmin(diff)]  # Top-Right (TR)
    ordered[3] = pts[np.argmax(diff)]  # Bottom-Left (BL)

    tl, tr, br, bl = ordered

    # 3. Tính chiều rộng tối đa (W_dst) bằng khoảng cách Euclid
    width_top = np.sqrt(np.sum((tr - tl) ** 2))
    width_bottom = np.sqrt(np.sum((br - bl) ** 2))
    w_dst = int(max(width_top, width_bottom))

    # 4. Tính chiều cao tối đa (H_dst)
    height_left = np.sqrt(np.sum((bl - tl) ** 2))
    height_right = np.sqrt(np.sum((br - tr) ** 2))
    h_dst = int(max(height_left, height_right))

    return ordered, (w_dst, h_dst)


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

    Args:
        src (np.ndarray): 4 điểm trên ảnh nguồn, shape (4, 2).
        dst (np.ndarray): 4 điểm trên ảnh đích, shape (4, 2).

    Returns:
        H (np.ndarray): Ma trận Homography shape (3, 3), chuẩn hóa H[2, 2] = 1.0.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src và dst đều phải có shape (4, 2).")

    A = np.zeros((8, 9), dtype=np.float64)

    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        # Hàng chẵn
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        # Hàng lẻ
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    # Phân tích Singular Value Decomposition (SVD)
    _, _, vt = np.linalg.svd(A)

    # Nghiệm h là hàng cuối cùng của Vt (ứng với giá trị kỳ dị nhỏ nhất)
    h = vt[-1, :]

    H = h.reshape((3, 3))

    # Chuẩn hóa để H[2, 2] = 1.0
    if np.abs(H[2, 2]) > 1e-10:
        H /= H[2, 2]

    return H


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
    w_dst, h_dst = out_size
    h_src, w_src = img.shape[:2]
    is_color = (img.ndim == 3)

    # 1. Tính ma trận nghịch đảo H^(-1) để ánh xạ ngược từ đích về nguồn
    H_inv = np.linalg.inv(H)

    # 2. Tạo lưới tọa độ đích (Meshgrid)
    y_dst_grid, x_dst_grid = np.meshgrid(
        np.arange(h_dst, dtype=np.float64),
        np.arange(w_dst, dtype=np.float64),
        indexing="ij"
    )

    # Stack thành tọa độ thuần nhất: shape (3, N) với N = h_dst * w_dst
    coords_dst = np.stack(
        [x_dst_grid.ravel(), y_dst_grid.ravel(), np.ones(h_dst * w_dst, dtype=np.float64)],
        axis=0
    )

    # 3. Nhân ma trận để tìm tọa độ tương ứng trên ảnh nguồn
    coords_src = H_inv @ coords_dst

    # Chuẩn hóa chia cho hệ số tỷ lệ w
    w_scale = coords_src[2, :]
    # Tránh chia cho 0
    w_scale = np.where(np.abs(w_scale) < 1e-10, 1e-10, w_scale)

    x_src = (coords_src[0, :] / w_scale).reshape((h_dst, w_dst))
    y_src = (coords_src[1, :] / w_scale).reshape((h_dst, w_dst))

    # 4. Xác định 4 điểm nguyên lân cận bao quanh (x, y)
    x0 = np.floor(x_src).astype(np.int32)
    y0 = np.floor(y_src).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Trọng số phần lẻ cho nội suy
    alpha = x_src - x0
    beta = y_src - y0

    # Mặt nạ kiểm tra pixel có nằm trong phạm vi ảnh nguồn không
    mask_valid = (x_src >= 0.0) & (x_src <= w_src - 1.0) & (y_src >= 0.0) & (y_src <= h_src - 1.0)

    # Kẹp chỉ số trong biên hợp lệ để không bị IndexError
    x0_c = np.clip(x0, 0, w_src - 1)
    x1_c = np.clip(x1, 0, w_src - 1)
    y0_c = np.clip(y0, 0, h_src - 1)
    y1_c = np.clip(y1, 0, h_src - 1)

    # 5. Thực hiện nội suy Bilinear
    if not is_color:
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
        # Điểm ngoài biên gán màu đen (0)
        interp = np.where(mask_valid, interp, 0.0)
        output = np.clip(np.round(interp), 0, 255).astype(np.uint8)
    else:
        # Xử lý cho ảnh màu 3 kênh
        alpha = alpha[..., np.newaxis]
        beta = beta[..., np.newaxis]
        mask_valid = mask_valid[..., np.newaxis]

        Ia = img[y0_c, x0_c, :].astype(np.float64)
        Ib = img[y0_c, x1_c, :].astype(np.float64)
        Ic = img[y1_c, x0_c, :].astype(np.float64)
        Id = img[y1_c, x1_c, :].astype(np.float64)

        interp = (
            (1.0 - alpha) * (1.0 - beta) * Ia
            + alpha * (1.0 - beta) * Ib
            + (1.0 - alpha) * beta * Ic
            + alpha * beta * Id
        )
        interp = np.where(mask_valid, interp, 0.0)
        output = np.clip(np.round(interp), 0, 255).astype(np.uint8)

    return output


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

    Tối ưu hóa:
    Áp dụng np.lib.stride_tricks.sliding_window_view để vector hóa cửa sổ trượt.

    Args:
        img (np.ndarray): Ảnh mức xám 2D (H, W).
        template (np.ndarray): Ảnh mẫu 2D (m, n) với m <= H, n <= W.

    Returns:
        ncc_map (np.ndarray): Bản đồ tương quan shape (H - m + 1, W - n + 1).
    """
    img_f = img.astype(np.float64)
    tmpl_f = template.astype(np.float64)

    h_img, w_img = img_f.shape[:2]
    h_tmpl, w_tmpl = tmpl_f.shape[:2]

    if h_tmpl > h_img or w_tmpl > w_img:
        raise ValueError("Kích thước template không được lớn hơn kích thước ảnh!")

    # 1. Trừ trung bình và chuẩn hóa template
    t_mean = np.mean(tmpl_f)
    t_zero = tmpl_f - t_mean
    t_norm = np.sqrt(np.sum(t_zero ** 2))

    if t_norm < 1e-10:
        return np.zeros((h_img - h_tmpl + 1, w_img - w_tmpl + 1), dtype=np.float32)

    # 2. Tạo mảng cửa sổ trượt 4D: shape (H - m + 1, W - n + 1, m, n)
    windows = np.lib.stride_tricks.sliding_window_view(img_f, (h_tmpl, w_tmpl))

    # 3. Tính mean và trừ mean của từng cửa sổ
    win_mean = np.mean(windows, axis=(-2, -1), keepdims=True)
    win_zero = windows - win_mean

    # 4. Tính chuẩn độ lệch (Norm) của từng cửa sổ
    win_norm = np.sqrt(np.sum(win_zero ** 2, axis=(-2, -1)))

    # 5. Tính tích vô hướng tương quan (Numerator)
    # Tương đương Sum(win_zero * t_zero) trên 2 trục cuối
    numerator = np.sum(win_zero * t_zero, axis=(-2, -1))

    # 6. Mẫu số
    denominator = win_norm * t_norm

    # 7. Tính NCC map (tránh chia cho 0)
    valid_denom = denominator > 1e-10
    ncc_map = np.zeros_like(numerator, dtype=np.float32)
    ncc_map[valid_denom] = (numerator[valid_denom] / denominator[valid_denom]).astype(np.float32)

    # Kẹp trong đoạn chuẩn [-1.0, 1.0] để khử sai số số thực
    return np.clip(ncc_map, -1.0, 1.0)


def detect_markers_and_correct_rotation(
    warped_img: np.ndarray,
    anchor_template: np.ndarray,
    search_margin_ratio: float = 0.25
) -> Tuple[np.ndarray, int, List[Tuple[int, int]]]:
    """
    Xác định vị trí Marker đặc biệt (Anchor Marker) ở 4 góc để tự động
    nhận biết góc xoay của phiếu thi (0°, 90°, 180°, 270°) và xoay phiếu về 0°.

    Giả định thiết kế phiếu chuẩn:
    - Marker đặc biệt (Anchor Marker) được đặt ở góc Top-Right (TR) ở trạng thái chuẩn 0°.
    - Khi phiếu bị xoay:
      + Anchor ở TR  --> 0° (Đúng chiều, không xoay).
      + Anchor ở BR  --> Xoay 90° cùng chiều kim đồng hồ --> Xoay lại bằng rot90 k=3.
      + Anchor ở BL  --> Xoay 180°                      --> Xoay lại bằng rot90 k=2.
      + Anchor ở TL  --> Xoay 270° (hoặc 90° ngược)     --> Xoay lại bằng rot90 k=1.

    Args:
        warped_img (np.ndarray): Ảnh phiếu đã được nắn thẳng.
        anchor_template (np.ndarray): Mẫu Marker đặc biệt.
        search_margin_ratio (float): Tỷ lệ vùng góc dùng để quét tìm Marker (mặc định 25%).

    Returns:
        corrected_img (np.ndarray): Ảnh đã được xoay chuẩn hướng 0°.
        detected_angle (int): Góc xoay phát hiện được (0, 90, 180, hoặc 270).
        corner_scores (List[Tuple[str, float]]): Điểm số NCC tại 4 góc [TL, TR, BR, BL].
    """
    h, w = warped_img.shape[:2]
    mh = int(h * search_margin_ratio)
    mw = int(w * search_margin_ratio)

    # Cắt 4 vùng góc (ROIs) để tìm kiếm Marker
    # 0: TL, 1: TR, 2: BR, 3: BL
    rois = {
        "TL": warped_img[0:mh, 0:mw],
        "TR": warped_img[0:mh, w - mw:w],
        "BR": warped_img[h - mh:h, w - mw:w],
        "BL": warped_img[h - mh:h, 0:mw],
    }

    max_scores = {}
    for corner, roi in rois.items():
        if roi.shape[0] >= anchor_template.shape[0] and roi.shape[1] >= anchor_template.shape[1]:
            ncc = match_template_ncc(roi, anchor_template)
            max_scores[corner] = float(np.max(ncc))
        else:
            max_scores[corner] = -1.0

    # Góc có điểm NCC cao nhất với anchor template
    best_corner = max(max_scores, key=max_scores.get)

    if best_corner == "TR":
        angle = 0
        corrected_img = warped_img.copy()
    elif best_corner == "BR":
        # Anchor từ TR đã dịch chuyển xuống BR -> Phiếu bị xoay 90 độ thuận chiều kim đồng hồ
        # Khắc phục: Xoay ngược chiều kim đồng hồ 90 độ (np.rot90 k=1)
        angle = 90
        corrected_img = np.rot90(warped_img, k=1)
    elif best_corner == "BL":
        # Anchor từ TR đã dịch chuyển sang BL -> Phiếu bị xoay 180 độ
        # Khắc phục: Xoay 180 độ (np.rot90 k=2)
        angle = 180
        corrected_img = np.rot90(warped_img, k=2)
    elif best_corner == "TL":
        # Anchor từ TR đã dịch chuyển sang TL -> Phiếu bị xoay 270 độ thuận (90 độ ngược)
        # Khắc phục: Xoay thuận chiều kim đồng hồ 90 độ (np.rot90 k=3)
        angle = 270
        corrected_img = np.rot90(warped_img, k=3)
    else:
        angle = 0
        corrected_img = warped_img.copy()

    return corrected_img, angle, list(max_scores.items())


def process_perspective_pipeline(
    img: np.ndarray,
    raw_corners: np.ndarray,
    anchor_template: np.ndarray = None,
    target_size: Tuple[int, int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Hàm đóng gói toàn bộ Pipeline của Thành viên 2 (TV2) thành 1 hàm duy nhất:
    Nhận đầu vào từ TV1 -> Xử lý toàn bộ -> Trả ảnh phẳng chuẩn 0° cho TV3.

    Các bước tự động:
    1. Sắp xếp 4 đỉnh theo thứ tự chuẩn [TL, TR, BR, BL].
    2. Xác định kích thước ảnh phẳng (W_dst, H_dst).
    3. Ước lượng ma trận Homography bằng SVD.
    4. Nắn phẳng ảnh bằng Inverse Warping + Bilinear Interpolation.
    5. (Tùy chọn) Nhận diện Marker và tự động xoay chuẩn hướng 0° nếu có anchor_template.

    Args:
        img (np.ndarray): Ảnh gốc đầu vào (mức xám hoặc màu).
        raw_corners (np.ndarray): 4 đỉnh tứ giác nhận từ TV1 (shape 4x2).
        anchor_template (np.ndarray, optional): Ảnh mẫu Marker đặc biệt để xoay hướng.
        target_size (Tuple[int, int], optional): Kích thước ép buộc (W, H). Nếu None, tự tính.

    Returns:
        warped_sheet (np.ndarray): Ảnh phiếu thi phẳng chuẩn 0° bàn giao cho TV3.
        info (dict): Metadata chi tiết (H_matrix, ordered_corners, detected_angle, v.v.).
    """
    # Bước 1: Sắp xếp 4 góc
    ordered_pts, auto_size = order_corners(raw_corners)
    dst_w, dst_h = target_size if target_size is not None else auto_size

    # Tọa độ 4 góc phẳng đích
    dst_pts = np.array([
        [0.0, 0.0],
        [float(dst_w), 0.0],
        [float(dst_w), float(dst_h)],
        [0.0, float(dst_h)]
    ], dtype=np.float32)

    # Bước 2: Giải ma trận Homography bằng SVD
    H = solve_homography_svd(ordered_pts, dst_pts)

    # Bước 3: Nắn phẳng ảnh
    warped = warp_perspective_bilinear(img, H, (dst_w, dst_h))

    # Bước 4: Tự động sửa góc xoay (nếu có template)
    detected_angle = 0
    scores = []
    if anchor_template is not None:
        warped, detected_angle, scores = detect_markers_and_correct_rotation(warped, anchor_template)

    info = {
        "ordered_corners": ordered_pts,
        "homography_matrix": H,
        "output_size": (dst_w, dst_h),
        "detected_angle": detected_angle,
        "marker_scores": scores
    }

    return warped, info


def warp_perspective_sheet(
    image: np.ndarray,
    corners: np.ndarray,
    target_width: int = 800,
    target_height: int = 1000,
    use_scratch: bool = True
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

    Returns:
        warped (np.ndarray): Ảnh đã nắn thẳng, shape (target_height, target_width).
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    ordered_pts, _ = order_corners(pts)
    dst_pts = np.array([
        [0.0, 0.0],
        [float(target_width - 1), 0.0],
        [float(target_width - 1), float(target_height - 1)],
        [0.0, float(target_height - 1)]
    ], dtype=np.float32)

    if use_scratch:
        H = solve_homography_svd(ordered_pts, dst_pts)
        return warp_perspective_bilinear(image, H, (target_width, target_height))
    else:
        import cv2
        matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
        return cv2.warpPerspective(image, matrix, (target_width, target_height))


