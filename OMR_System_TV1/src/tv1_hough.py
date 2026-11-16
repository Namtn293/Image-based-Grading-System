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
    selected_lines: List[Tuple[float, float]] = []

    for idx in sort_order:
        r_i = candidate_r_idxs[idx]
        t_i = candidate_t_idxs[idx]
        cur_rho = float(rhos[r_i])
        cur_theta = float(thetas[t_i])

        # Kiểm tra xem có quá gần với đường thẳng đã chọn nào trước đó không
        is_duplicate = False
        for s_rho, s_theta in selected_lines:
            d_theta = abs(cur_theta - s_theta)
            if d_theta > np.pi / 2:
                d_theta = np.pi - d_theta
            d_rho = abs(cur_rho - s_rho)

            if d_theta < min_theta_diff and d_rho < min_rho_diff:
                is_duplicate = True
                break

        if not is_duplicate:
            selected_lines.append((cur_rho, cur_theta))

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
    """Thuật toán Ramer-Douglas-Peucker (RDP) xấp xỉ rút gọn đa giác từ đầu bằng NumPy.

    Args:
        pts: Mảng các đỉnh đường gấp khúc/đa giác shape (N, 2).
        epsilon: Sai số khoảng cách ngưỡng cực đại.

    Returns:
        Mảng các đỉnh đã rút gọn.
    """
    if len(pts) <= 2:
        return pts

    def point_line_distance(pt, start, end):
        x0, y0 = pt
        x1, y1 = start
        x2, y2 = end
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = np.hypot(y2 - y1, x2 - x1)
        return num / den if den > 1e-8 else 0.0

    dmax = 0.0
    index = 0
    start, end = pts[0], pts[-1]

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
        return np.array([start, end], dtype=np.float32)


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
    diff_v = binary_mask[1:, :] != binary_mask[:-1, :]
    diff_h = binary_mask[:, 1:] != binary_mask[:, :-1]

    edge_mask = np.zeros_like(binary_mask, dtype=bool)
    edge_mask[1:, :] |= diff_v
    edge_mask[:, 1:] |= diff_h

    edge_pts = np.argwhere(edge_mask)
    if len(edge_pts) < 50:
        return None

    # Đổi sang hệ tọa độ (x, y)
    pts = np.column_stack([edge_pts[:, 1], edge_pts[:, 0]])

    # Lấy mẫu tối đa 3000 điểm để thuật toán Convex Hull chạy < 5 ms
    if len(pts) > 3000:
        sample_indices = np.linspace(0, len(pts) - 1, 3000, dtype=int)
        pts = pts[sample_indices]

    hull = convex_hull_2d(pts)
    if len(hull) < 4:
        return None

    # Khép kín bao lồi để tính chu vi
    hull_closed = np.vstack([hull, hull[0]])
    perimeter = float(np.sum(np.hypot(np.diff(hull_closed[:, 0]), np.diff(hull_closed[:, 1]))))

    # Thử nghiệm các hệ số epsilon để tìm đa giác đúng 4 cạnh
    for eps_factor in [0.03, 0.025, 0.02, 0.035, 0.04, 0.015]:
        simplified = ramer_douglas_peucker(hull_closed, eps_factor * perimeter)
        quad = simplified[:-1] if np.allclose(simplified[0], simplified[-1]) else simplified
        if len(quad) == 4:
            ordered = order_quad_corners(quad)
            return ordered

    # Dự phòng: Chọn 4 điểm cực trị trên bao lồi theo 4 góc
    s = hull[:, 0] + hull[:, 1]
    diff = hull[:, 1] - hull[:, 0]
    tl = hull[np.argmin(s)]
    br = hull[np.argmax(s)]
    tr = hull[np.argmin(diff)]
    bl = hull[np.argmax(diff)]
    return order_quad_corners(np.array([tl, tr, br, bl], dtype=np.float32))
