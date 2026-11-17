"""Module: TV1 - Biến đổi Hough (Hough Line Transform) & Xác định 4 đỉnh phiếu
Đề tài: Hệ thống Chấm bài Trắc nghiệm Tự động (OMR System)
Môn học: Xử lý ảnh số (INT13146) - Học viện Công nghệ Bưu chính Viễn thông (PTIT)
Giảng viên hướng dẫn: TS. Phạm Hoàng Việt
Thành viên phụ trách: Thành viên 1 (Tiền xử lý & Phát hiện khung phiếu)

YÊU CẦU BẮT BUỘC:
- 100% Cài đặt từ đầu (Code from Scratch) bằng NumPy:
    1. Biến đổi không gian tham số cực Hesse (rho, theta): rho = x*cos(theta) + y*sin(theta)
    2. Bỏ phiếu vector hóa trong mảng tích lũy (Accumulator Array)
    3. Trích xuất cực đại cục bộ (Peak detection với Non-Maximum Suppression trong không gian tham số)
    4. Phân loại đường thẳng ngang/dọc, tính giao điểm hình học 4 góc khung phiếu
    5. Cung cấp thuật toán đa giác xấp xỉ hình học (Andrew's Monotone Chain Convex Hull + RDP)
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np


def hough_lines_standard(
    edge_image: np.ndarray,
    rho_res: float = 1.0,
    theta_res: float = np.pi / 180.0,
    threshold: int = 100,
    min_theta_diff: float = np.pi / 36.0,
    min_rho_diff: float = 10.0
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """Biến đổi Hough đường thẳng dạng chuẩn trong không gian tham số cực (Standard Hough Transform).

    Cơ sở toán học:
      Đường thẳng trong mặt phẳng ảnh (x, y) được biểu diễn bằng phương trình tham số cực Hesse:
        rho = x * cos(theta) + y * sin(theta)
      với:
        theta in [0, pi): Góc pháp tuyến của đường thẳng với trục Ox.
        rho in [-D, D]: Khoảng cách có dấu từ gốc tọa độ (0, 0) tới đường thẳng, D = sqrt(H^2 + W^2).

      Mỗi điểm biên (x_i, y_i) ánh xạ thành một đường hình sin trong không gian (rho, theta).
      Giao điểm của nhiều đường hình sin tương ứng với cực đại trong Mảng tích lũy (Accumulator Array).

    Args:
        edge_image: Ảnh nhị phân biên 2D uint8 (các điểm biên mang giá trị > 0).
        rho_res: Độ phân giải khoảng cách rho (mặc định 1.0 pixel).
        theta_res: Độ phân giải góc theta (mặc định pi / 180 rad = 1 độ).
        threshold: Số phiếu tối thiểu để một ô tích lũy được công nhận là đường thẳng.
        min_theta_diff: Khoảng cách góc tối thiểu giữa 2 đường thẳng phân biệt (khử cực đại phụ).
        min_rho_diff: Khoảng cách rho tối thiểu giữa 2 đường thẳng phân biệt.

    Returns:
        lines: Danh sách các cặp (rho, theta) sắp xếp giảm dần theo số phiếu.
        accumulator: Ma trận tích lũy 2D shape (num_rhos, num_thetas).
    """
    img = np.asarray(edge_image)
    if img.ndim != 2:
        raise ValueError(f"hough_lines_standard yêu cầu ảnh nhị phân 2D, nhận được shape {img.shape}")

    h, w = img.shape
    diag_len = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))

    # Tạo các trục tọa độ không gian tham số
    rhos = np.arange(-diag_len, diag_len + 1, rho_res, dtype=np.float64)
    thetas = np.arange(0.0, np.pi, theta_res, dtype=np.float64)

    num_rhos = len(rhos)
    num_thetas = len(thetas)

    # Tìm tọa độ các điểm biên (y = row, x = col)
    y_idxs, x_idxs = np.nonzero(img)
    num_pts = len(x_idxs)

    if num_pts == 0:
        return [], np.zeros((num_rhos, num_thetas), dtype=np.int32)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Bỏ phiếu vector hóa 100%:
    # r_matrix shape: (num_pts, num_thetas)
    r_matrix = np.outer(x_idxs, cos_t) + np.outer(y_idxs, sin_t)
    r_idx_matrix = np.round((r_matrix + diag_len) / rho_res).astype(np.int64)

    theta_idx_matrix = np.broadcast_to(
        np.arange(num_thetas, dtype=np.int64),
        r_idx_matrix.shape
    )

    valid_mask = (r_idx_matrix >= 0) & (r_idx_matrix < num_rhos)
    flat_indices = r_idx_matrix[valid_mask] * num_thetas + theta_idx_matrix[valid_mask]

    counts = np.bincount(flat_indices, minlength=num_rhos * num_thetas)
    accumulator = counts.reshape((num_rhos, num_thetas)).astype(np.int32)

    # Trích xuất các cực đại cục bộ vượt ngưỡng threshold
    candidate_r_idxs, candidate_t_idxs = np.where(accumulator >= threshold)
    if len(candidate_r_idxs) == 0:
        return [], accumulator

    # Sắp xếp các ứng viên giảm dần theo số phiếu
    vote_counts = accumulator[candidate_r_idxs, candidate_t_idxs]
    sort_order = np.argsort(vote_counts)[::-1]

    # Non-Maximum Suppression trên không gian tham số cực để loại bỏ các đỉnh lân cận cùng đường
    ordered_candidates: List[Tuple[float, float]] = [
        (float(rhos[candidate_r_idxs[idx]]), float(thetas[candidate_t_idxs[idx]]))
        for idx in sort_order
    ]
    selected_lines = apply_parameter_space_nms(
        ordered_candidates,
        min_theta_diff=min_theta_diff,
        min_rho_diff=min_rho_diff
    )

    return selected_lines, accumulator


def intersect_lines_polar(
    line1: Tuple[float, float],
    line2: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """Giải hệ phương trình tuyến tính tìm giao điểm của 2 đường thẳng trong hệ tọa độ cực.

    Cơ sở toán học:
      Hệ 2 phương trình:
        cos(theta_1) * x + sin(theta_1) * y = rho_1
        cos(theta_2) * x + sin(theta_2) * y = rho_2

      Định thức ma trận hệ số (Determinant):
        D = cos(theta_1)*sin(theta_2) - sin(theta_1)*cos(theta_2) = sin(theta_2 - theta_1)

      Nghiệm giải tích theo quy tắc Cramer:
        x = (rho_1 * sin(theta_2) - rho_2 * sin(theta_1)) / D
        y = (rho_2 * cos(theta_1) - rho_1 * cos(theta_2)) / D

    Args:
        line1: (rho1, theta1)
        line2: (rho2, theta2)

    Returns:
        (x, y) là tọa độ giao điểm trên mặt phẳng ảnh, hoặc None nếu hai đường thẳng gần song song.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2

    det = np.sin(theta2 - theta1)
    if abs(det) < 1e-4:  # Hai đường thẳng song song hoặc trùng nhau
        return None

    x = (rho1 * np.sin(theta2) - rho2 * np.sin(theta1)) / det
    y = (rho2 * np.cos(theta1) - rho1 * np.cos(theta2)) / det

    return float(x), float(y)


def order_quad_corners(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp chuẩn hóa 4 đỉnh tứ giác theo thứ tự hình học [TL, TR, BR, BL].

    Quy tắc chuẩn tương thích 100% với Module Thành viên 2 (TV2):
      - Top-Left (TL): Điểm có tổng (x + y) nhỏ nhất.
      - Bottom-Right (BR): Điểm có tổng (x + y) lớn nhất.
      - Top-Right (TR): Điểm có hiệu (y - x) nhỏ nhất (x - y lớn nhất).
      - Bottom-Left (BL): Điểm có hiệu (y - x) lớn nhất.

    Args:
        pts: Mảng shape (4, 2) chứa 4 điểm bất kỳ.

    Returns:
        Mảng float32 shape (4, 2) theo thứ tự [TL, TR, BR, BL].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]  # y - x

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Thuật toán Chuỗi đơn điệu Andrew (Andrew's Monotone Chain 2D Convex Hull).

    Độ phức tạp O(N log N), cài đặt thuần NumPy/Python không dùng scipy/cv2.

    Args:
        points: Mảng các điểm 2D shape (N, 2).

    Returns:
        Mảng các đỉnh bao lồi theo thứ tự ngược chiều kim đồng hồ, shape (M, 2).
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= 3:
        return pts.astype(np.float32)

    # Sắp xếp các điểm theo hoành độ x, sau đó đến tung độ y
    sorted_pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Xây dựng nhánh dưới (Lower hull)
    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    # Xây dựng nhánh trên (Upper hull)
    upper = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float32)


def ramer_douglas_peucker(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """Thuật toán Ramer-Douglas-Peucker (RDP) xấp xỉ rút gọn đường gấp khúc từ đầu bằng NumPy.

    LƯU Ý QUAN TRỌNG: hàm này chỉ đúng với ĐA GIÁC HỞ (polyline), tức điểm đầu và điểm
    cuối phải KHÁC nhau. Nếu truyền vào một đa giác khép kín (điểm đầu trùng điểm cuối),
    "đoạn thẳng" tham chiếu giữa hai đầu suy biến thành một điểm, khoảng cách từ mọi đỉnh
    tới nó luôn bằng 0, nên hàm luôn trả về đúng 2 điểm. Muốn rút gọn đa giác khép kín,
    dùng :func:`simplify_closed_polygon` (tách đôi tại đỉnh xa nhất).

    Args:
        pts: Mảng các đỉnh đường gấp khúc hở shape (N, 2).
        epsilon: Sai số khoảng cách ngưỡng cực đại.

    Returns:
        Mảng các đỉnh đã rút gọn (luôn giữ điểm đầu và điểm cuối).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) <= 2:
        return pts

    start, end = pts[0], pts[-1]
    # Đa giác khép kín bị truyền nhầm: đoạn tham chiếu suy biến -> không thể rút gọn đúng.
    if np.allclose(start, end):
        raise ValueError(
            "ramer_douglas_peucker chỉ nhận đa giác hở (điểm đầu khác điểm cuối). "
            "Với đa giác khép kín, hãy dùng simplify_closed_polygon()."
        )

    def point_line_distance(pt, p_start, p_end):
        x0, y0 = pt
        x1, y1 = p_start
        x2, y2 = p_end
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = np.hypot(y2 - y1, x2 - x1)
        return num / den if den > 1e-8 else 0.0

    dmax = 0.0
    index = 0

    for i in range(1, len(pts) - 1):
        d = point_line_distance(pts[i], start, end)
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        rec1 = ramer_douglas_peucker(pts[:index + 1], epsilon)
        rec2 = ramer_douglas_peucker(pts[index:], epsilon)
        return np.vstack([rec1[:-1], rec2])
    else:
        return np.array([start, end], dtype=np.float64)


def simplify_closed_polygon(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """Rút gọn ĐA GIÁC KHÉP KÍN bằng RDP: tách thành hai nửa hở tại đỉnh xa nhất.

    Cách làm đúng theo đề xuất của Code Review (mục 2.2):
      1. Bỏ điểm trùng lặp cuối (nếu đa giác được khép kín sẵn).
      2. Tìm đỉnh xa đỉnh 0 nhất, gọi là ``far``.
      3. Chạy RDP trên nửa hở ``pts[0..far]`` và nửa hở ``pts[far..0]``.
      4. Ghép lại, bỏ điểm nối trùng.

    Args:
        pts: Các đỉnh của đa giác khép kín shape (N, 2), có thể chứa hoặc không chứa
             điểm lặp lại của đỉnh đầu ở cuối mảng.
        epsilon: Sai số khoảng cách ngưỡng cực đại.

    Returns:
        Mảng các đỉnh đã rút gọn shape (M, 2) (không lặp lại đỉnh đầu ở cuối).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return pts

    # Bỏ đỉnh khép kín trùng đỉnh đầu
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    n = len(pts)
    if n < 3:
        return pts

    # Đỉnh xa đỉnh 0 nhất -> chia đa giác thành 2 nửa hở không suy biến
    far = int(np.argmax(np.hypot(pts[:, 0] - pts[0, 0], pts[:, 1] - pts[0, 1])))

    first_half = pts[:far + 1]                 # hở: pts[0] ... pts[far]
    second_half = np.vstack([pts[far:], pts[:1]])  # hở: pts[far] ... pts[0]

    simplified_first = ramer_douglas_peucker(first_half, epsilon)
    simplified_second = ramer_douglas_peucker(second_half, epsilon)

    # Ghép và bỏ đỉnh nối trùng (đỉnh ``far`` và đỉnh 0 bị lặp)
    return np.vstack([simplified_first[:-1], simplified_second[:-1]])


def otsu_threshold_scratch(gray: np.ndarray) -> int:
    """Tính ngưỡng Otsu toàn cục từ đầu bằng NumPy (dùng cho bước tách nền).

    Cực đại hóa phương sai giữa hai lớp:
        sigma_B^2(t) = omega_0(t) * omega_1(t) * (mu_0(t) - mu_1(t))^2

    Hàm này KHÔNG dùng cv2.threshold: chỉ dùng ``np.bincount`` và tổng tích lũy, độ phức
    tạp O(256 + H*W). Cùng công thức với module TV3, nhưng TV1 cần một bản cục bộ để
    ``detect_sheet_corners`` không phụ thuộc vào module của thành viên khác.

    Args:
        gray: Ảnh mức xám 2D (uint8 hoặc mảng số).

    Returns:
        Ngưỡng tối ưu trong khoảng [0, 255].
    """
    values = np.clip(np.asarray(gray), 0, 255).astype(np.uint8).ravel()
    hist = np.bincount(values, minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128

    prob = hist / total
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(prob)                 # omega_0(t)
    mu = np.cumsum(prob * levels)           # mu_0(t) * omega_0(t)
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    between_var = (mu_total * omega - mu) ** 2 / denom

    return int(np.argmax(between_var))


def collect_boundary_extremes(binary_mask: np.ndarray) -> np.ndarray:
    """Thu thập các điểm biên CỰC TRỊ theo từng hàng và từng cột của mặt nạ nhị phân.

    Thay cho cách lấy mẫu raster cũ (``np.linspace`` trên thứ tự ``argwhere``): cách cũ
    lấy mẫu theo thứ tự quét ảnh nên có thể BỎ SÓT đúng các đỉnh góc của tứ giác, khiến
    bao lồi bị "vát góc" và RDP không tìm được 4 đỉnh. Ở đây với mỗi hàng có pixel biên
    ta lấy min/max cột, mỗi cột có pixel biên ta lấy min/max hàng, tức tối đa 2H + 2W
    điểm nhưng LUÔN chứa 4 đỉnh góc thật.

    Args:
        binary_mask: Mặt nạ nhị phân 2D (giá trị khác 0 là vùng quan tâm).

    Returns:
        Mảng (N, 2) tọa độ (x, y) các điểm biên cực trị.
    """
    mask = np.asarray(binary_mask) > 0
    h, w = mask.shape

    # Điểm biên = pixel foreground có ít nhất 1 lân cận 4 hướng là background
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    border = mask & ~inner

    pts = []

    # Cực trị theo hàng: x nhỏ nhất và lớn nhất của mỗi hàng có biên
    rows = np.flatnonzero(border.any(axis=1))
    if rows.size:
        sub = border[rows, :]
        cols_min = np.argmax(sub, axis=1)
        cols_max = w - 1 - np.argmax(sub[:, ::-1], axis=1)
        pts.append(np.column_stack([cols_min, rows]))
        pts.append(np.column_stack([cols_max, rows]))

    # Cực trị theo cột: y nhỏ nhất và lớn nhất của mỗi cột có biên
    cols = np.flatnonzero(border.any(axis=0))
    if cols.size:
        sub = border[:, cols]
        rows_min = np.argmax(sub, axis=0)
        rows_max = h - 1 - np.argmax(sub[::-1, :], axis=0)
        pts.append(np.column_stack([cols, rows_min]))
        pts.append(np.column_stack([cols, rows_max]))

    if not pts:
        return np.empty((0, 2), dtype=np.float64)

    return np.vstack(pts).astype(np.float64)


def detect_quad_from_boundary_scratch(
    binary_mask: np.ndarray,
    img_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Tìm 4 góc của tờ phiếu trên nền tối bằng Bao lồi và Xấp xỉ đa giác RDP (100% Scratch).

    Args:
        binary_mask: Mặt nạ nhị phân vùng giấy phiếu (giá trị 255 ở vùng giấy).
        img_shape: (H, W) của ảnh.

    Returns:
        Tọa độ 4 góc shape (4, 2) đã sắp xếp [TL, TR, BR, BL], hoặc None nếu không tìm thấy.
    """
    h, w = img_shape

    # Lấy điểm biên cực trị theo hàng/cột (luôn chứa 4 đỉnh góc thật)
    pts = collect_boundary_extremes(binary_mask)
    if len(pts) < 50:
        return None

    hull = convex_hull_2d(pts)
    if len(hull) < 4:
        return None

    # Chu vi đa giác khép kín (dùng để chọn epsilon tương đối)
    hull_closed = np.vstack([hull, hull[0]])
    perimeter = float(np.sum(np.hypot(np.diff(hull_closed[:, 0]), np.diff(hull_closed[:, 1]))))

    # RDP trên đa giác KHÉP KÍN (tách đôi tại đỉnh xa nhất) thay vì đưa đa giác khép kín
    # vào RDP của đa giác hở như code cũ (khi đó hàm luôn trả về 2 điểm).
    for eps_factor in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
        quad = simplify_closed_polygon(hull, eps_factor * perimeter)
        if len(quad) == 4:
            return order_quad_corners(quad)

    # Dự phòng: chọn 4 điểm cực trị của bao lồi theo tổng/hiệu tọa độ
    s = hull[:, 0] + hull[:, 1]
    diff = hull[:, 1] - hull[:, 0]
    tl = hull[np.argmin(s)]
    br = hull[np.argmax(s)]
    tr = hull[np.argmin(diff)]
    bl = hull[np.argmax(diff)]
    return order_quad_corners(np.array([tl, tr, br, bl], dtype=np.float32))


def apply_parameter_space_nms(
    lines: List[Tuple[float, float]],
    min_theta_diff: float = np.pi / 36.0,
    min_rho_diff: float = 10.0
) -> List[Tuple[float, float]]:
    """Khử cực đại phụ trong không gian tham số cho một danh sách đường ĐÃ sắp xếp giảm dần.

    Dùng để chuẩn hóa kết quả của thư viện khác (ví dụ ``cv2.HoughLines``) theo ĐÚNG quy
    trình của :func:`hough_lines_standard` trước khi so sánh. Nếu không chuẩn hóa, OpenCV
    trả về rất nhiều cực đại phụ nằm sát nhau trong không gian (rho, theta), làm phép đo
    độ trùng khớp bị sai lệch dù tập đường chính hoàn toàn giống nhau.

    Args:
        lines: Danh sách (rho, theta) đã sắp xếp giảm dần theo số phiếu.
        min_theta_diff: Khoảng cách góc tối thiểu giữa 2 đường phân biệt (radian).
        min_rho_diff: Khoảng cách rho tối thiểu giữa 2 đường phân biệt (pixel).

    Returns:
        Danh sách (rho, theta) sau khi loại các cực đại phụ lân cận.
    """
    selected: List[Tuple[float, float]] = []
    for rho, theta in lines:
        is_duplicate = False
        for s_rho, s_theta in selected:
            d_theta = abs(theta - s_theta)
            if d_theta > np.pi / 2:
                d_theta = np.pi - d_theta
            if d_theta < min_theta_diff and abs(rho - s_rho) < min_rho_diff:
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append((float(rho), float(theta)))
    return selected


def score_lines_by_edge_support(
    lines: List[Tuple[float, float]],
    edge_image: np.ndarray,
    tolerance: float = 1.5,
    max_points: int = 20000
) -> List[Tuple[int, float, float]]:
    """Chấm điểm từng đường thẳng Hough bằng số điểm biên thật sự nằm trên đường đó.

    Mảng tích lũy Hough chỉ đếm phiếu trong một ô (rho, theta) rời rạc, nên hai đường
    có số phiếu gần bằng nhau rất khó phân biệt. Đếm trực tiếp số pixel biên có khoảng
    cách tới đường <= ``tolerance`` cho thứ hạng ổn định hơn nhiều.

    Bộ nhớ của ma trận khoảng cách là O(num_lines x num_points). Với ảnh rất lớn, số điểm
    biên có thể lên tới hàng trăm nghìn, nên hàm lấy mẫu thưa đều (stride cố định) để giới
    hạn ở ``max_points`` điểm — thứ hạng đường gần như không đổi vì mẫu vẫn trải khắp ảnh.

    Args:
        lines: Danh sách (rho, theta) do :func:`hough_lines_standard` trả về.
        edge_image: Ảnh biên nhị phân đã dùng để bỏ phiếu.
        tolerance: Bán kính dung sai (pixel).
        max_points: Số điểm biên tối đa dùng để chấm điểm.

    Returns:
        Danh sách (support, rho, theta) sắp xếp giảm dần theo support.
    """
    img = np.asarray(edge_image)
    ys, xs = np.nonzero(img)
    if xs.size == 0 or not lines:
        return []

    if xs.size > max_points:
        stride = int(np.ceil(xs.size / float(max_points)))
        xs = xs[::stride]
        ys = ys[::stride]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    cos_t = np.array([np.cos(th) for _, th in lines], dtype=np.float64)
    sin_t = np.array([np.sin(th) for _, th in lines], dtype=np.float64)
    rhos = np.array([rho for rho, _ in lines], dtype=np.float64)

    support = np.zeros(len(lines), dtype=np.int64)
    # Chia lô theo đường để đỉnh bộ nhớ không vượt quá ~32 MB
    batch = max(1, int(2_000_000 // max(xs.size, 1)))
    for start in range(0, len(lines), batch):
        stop = min(start + batch, len(lines))
        dist = np.abs(
            cos_t[start:stop, None] * xs[None, :]
            + sin_t[start:stop, None] * ys[None, :]
            - rhos[start:stop, None]
        )
        support[start:stop] = (dist <= tolerance).sum(axis=1)

    scored = [(int(support[i]), float(lines[i][0]), float(lines[i][1])) for i in range(len(lines))]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def detect_quad_from_hough_lines(
    lines: List[Tuple[float, float]],
    edge_image: np.ndarray,
    img_shape: Tuple[int, int],
    angle_tolerance_deg: float = 20.0,
    min_span_ratio: float = 0.5,
    support_tolerance: float = 1.5
) -> Optional[np.ndarray]:
    """Dựng 4 đỉnh khung phiếu từ 4 đường Hough mạnh nhất (2 cạnh dọc + 2 cạnh ngang).

    Quy trình đúng theo kế hoạch mục 5.1(4) và đề xuất của Code Review (mục 1.2):
      1. Chấm điểm từng đường bằng số điểm biên thật sự nằm trên nó.
      2. Tách thành họ cạnh dọc (pháp tuyến gần trục x) và họ cạnh ngang (gần trục y).
      3. Trong mỗi họ, chọn cặp đường có tổng support lớn nhất mà khoảng cách giữa chúng
         đủ lớn (>= ``min_span_ratio`` kích thước ảnh) — tức 2 cạnh đối diện của khung.
      4. Tính 4 giao điểm bằng :func:`intersect_lines_polar` (quy tắc Cramer).

    Args:
        lines: Danh sách (rho, theta) từ Hough.
        edge_image: Ảnh biên dùng để tính support.
        img_shape: (H, W) của ảnh.
        angle_tolerance_deg: Dung sai lệch hướng so với trục (độ).
        min_span_ratio: Khoảng cách tối thiểu giữa 2 cạnh đối diện, theo tỉ lệ kích thước ảnh.
        support_tolerance: Bán kính dung sai khi đếm điểm biên trên đường (pixel).

    Returns:
        4 góc shape (4, 2) đã sắp xếp [TL, TR, BR, BL], hoặc None nếu không dựng được.
    """
    h, w = img_shape
    if not lines:
        return None

    scored = score_lines_by_edge_support(lines, edge_image, tolerance=support_tolerance)
    if len(scored) < 4:
        return None

    cos_tol = np.cos(np.deg2rad(angle_tolerance_deg))
    vertical: List[Tuple[int, float, float]] = []
    horizontal: List[Tuple[int, float, float]] = []
    for support, rho, theta in scored:
        if abs(np.cos(theta)) >= cos_tol:
            vertical.append((support, rho, theta))
        elif abs(np.sin(theta)) >= cos_tol:
            horizontal.append((support, rho, theta))

    if len(vertical) < 2 or len(horizontal) < 2:
        return None

    def line_offset(item: Tuple[int, float, float], axis: str) -> Optional[float]:
        """Giao điểm của đường với trục giữa ảnh -> dùng để đo khoảng cách giữa 2 cạnh."""
        _, rho, theta = item
        cos_v, sin_v = np.cos(theta), np.sin(theta)
        if axis == "V":  # x = const
            if abs(cos_v) < 1e-9:
                return None
            return float((rho - (h * 0.5) * sin_v) / cos_v)
        if abs(sin_v) < 1e-9:
            return None
        return float((rho - (w * 0.5) * cos_v) / sin_v)

    def best_opposite_pair(
        family: List[Tuple[int, float, float]],
        axis: str,
        min_span: float
    ) -> Optional[Tuple[int, Tuple[int, float, float], Tuple[int, float, float]]]:
        best = None
        offsets = [line_offset(item, axis) for item in family]
        for i in range(len(family)):
            if offsets[i] is None:
                continue
            for j in range(i + 1, len(family)):
                if offsets[j] is None:
                    continue
                if abs(offsets[i] - offsets[j]) < min_span:
                    continue
                total = family[i][0] + family[j][0]
                if best is None or total > best[0]:
                    best = (total, family[i], family[j])
        return best

    pair_v = best_opposite_pair(vertical, "V", min_span_ratio * w)
    pair_h = best_opposite_pair(horizontal, "H", min_span_ratio * h)
    if pair_v is None or pair_h is None:
        return None

    corners = []
    for v_line in (pair_v[1], pair_v[2]):
        for h_line in (pair_h[1], pair_h[2]):
            pt = intersect_lines_polar((v_line[1], v_line[2]), (h_line[1], h_line[2]))
            if pt is None:
                return None
            corners.append(pt)

    quad = order_quad_corners(np.array(corners, dtype=np.float32))

    # Kiểm tra tính hợp lệ hình học: tứ giác phải nằm trong (hoặc sát) khung ảnh
    xs, ys = quad[:, 0], quad[:, 1]
    margin_x, margin_y = 0.03 * w, 0.03 * h
    if xs.min() < -margin_x or xs.max() > w - 1 + margin_x:
        return None
    if ys.min() < -margin_y or ys.max() > h - 1 + margin_y:
        return None

    area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    if area / float(h * w) < 0.30:
        return None

    return quad
